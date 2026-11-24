//! Build script for `inferno-server`.
//!
//! The hot path is Zig. `zig build` produces the static library
//! `libinferno_core.a` (with the CUDA cubins embedded) and this script points
//! the Rust linker at it, so `cargo build` alone rebuilds the whole engine.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Optimization mode of the Zig library. The gate builds the Rust side with
/// `--release`; the core keeps its safety checks in both profiles.
const ZIG_OPTIMIZE: &str = "ReleaseSafe";

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

    println!(
        "cargo:rustc-link-search=native={}",
        zig_prefix.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=inferno_core");

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
