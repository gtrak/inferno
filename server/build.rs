//! Build script for `inferno-server`.
//!
//! The hot path is Zig. `zig build` produces the static library
//! `libinferno_core.a` (with the CUDA cubins embedded) and this script points
//! the Rust linker at it, so `cargo build` alone rebuilds the whole engine.
//!
//! The engine C ABI (`inferno_core_init`, `inferno_core_submit`, ...) lands in
//! issue I-15. Until the archive defines those symbols the crate cannot link
//! against them, so this script reports what it found as a `cfg` and
//! `src/engine/ffi.rs` swaps in the "core ABI not linked" fallbacks. The switch
//! is derived from the archive itself, never from a hand-maintained flag: the
//! day I-15 exports the ABI the fallbacks disappear without an edit here.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Optimization mode of the Zig library. The gate builds the Rust side with
/// `--release`; the core keeps its safety checks in both profiles.
const ZIG_OPTIMIZE: &str = "ReleaseSafe";

/// Every symbol of the model/sequence handle ABI (I-036) that
/// `src/engine/handles.rs` declares.
///
/// Unlike the engine ABI below, this one is not optional: the Rust handle module
/// has no fallback, so the archive must define all of it.
const HANDLE_ABI_SYMBOLS: [&str; 7] = [
    "inferno_model_open",
    "inferno_model_close",
    "inferno_seq_open",
    "inferno_seq_close",
    "inferno_seq_reset",
    "inferno_seq_step",
    "inferno_handle_counts",
];

/// Every symbol of the engine C ABI that `src/engine/ffi.rs` declares.
const ENGINE_ABI_SYMBOLS: [&str; 7] = [
    "inferno_core_init",
    "inferno_core_shutdown",
    "inferno_core_version",
    "inferno_core_submit",
    "inferno_core_cancel",
    "inferno_core_set_token_cb",
    "inferno_core_metrics",
];

fn main() {
    let manifest_directory =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let core_directory = manifest_directory
        .parent()
        .expect("server/ always sits inside the workspace root")
        .join("core");
    let output_directory = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let zig_prefix = output_directory.join("zig");

    let status = Command::new("zig")
        .current_dir(&core_directory)
        .arg("build")
        .arg(format!("-Doptimize={ZIG_OPTIMIZE}"))
        .arg("--prefix")
        .arg(&zig_prefix)
        .status()
        .expect("failed to run `zig`; Zig 0.14.1 must be on PATH");
    assert!(
        status.success(),
        "zig build of the inference core failed with {status}"
    );

    let library = zig_prefix.join("lib").join("libinferno_core.a");
    assert!(
        library.is_file(),
        "zig build did not produce {}",
        library.display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        zig_prefix.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=inferno_core");

    let archive = std::fs::read(&library).expect("the Zig static library is readable");

    println!("cargo::rustc-check-cfg=cfg(inferno_core_abi_absent)");
    if !defines_engine_abi(&archive) {
        println!("cargo:rustc-cfg=inferno_core_abi_absent");
        println!(
            "cargo:warning=libinferno_core.a does not export the engine C ABI yet (issue I-15); \
             building with the not-implemented fallbacks, so only `--engine mock` will serve"
        );
    }

    assert_handle_abi(&archive);

    // The cubins are part of the archive, so kernel sources and the Zig build
    // graph are inputs of this crate as well.
    println!(
        "cargo:rerun-if-changed={}",
        core_directory.join("build.zig").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        core_directory.join("build.zig.zon").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        core_directory.join("src").display()
    );
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}

/// True when the archive's symbol table carries every engine ABI name.
///
/// A byte scan is enough: the archive's string table holds the mangled names of
/// the symbols it defines, and the core never *references* its own exported C
/// ABI, so a hit means the symbol is defined rather than merely wanted.
fn defines_engine_abi(archive: &[u8]) -> bool {
    ENGINE_ABI_SYMBOLS
        .iter()
        .all(|symbol| contains(archive, symbol.as_bytes()))
}

/// Fails the build with a sentence, not a linker error, when the archive is
/// missing part of the handle ABI that `src/engine/handles.rs` links against.
fn assert_handle_abi(archive: &[u8]) {
    let missing: Vec<&str> = HANDLE_ABI_SYMBOLS
        .iter()
        .copied()
        .filter(|symbol| !contains(archive, symbol.as_bytes()))
        .collect();
    assert!(
        missing.is_empty(),
        "libinferno_core.a does not export the handle C ABI ({missing:?}); \
         core/src/c_api.zig must `export` those symbols"
    );
}

/// Reports whether `haystack` contains `needle`.
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    assert!(!needle.is_empty());
    if needle.len() > haystack.len() {
        return false;
    }
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}
