//! `inferno-server`: the Rust side of the engine.
//!
//! This binary exercises the C ABI of the Zig core: the device probe and the
//! version string. The HTTP surface arrives in a later issue.

use std::ffi::{CStr, c_char};
use std::process::ExitCode;

/// Room for the probe document of a fully populated machine.
const PROBE_BUFFER_BYTES: usize = 64 * 1024;

/// Conventional exit code for a bad command line.
const EXIT_USAGE: u8 = 2;

// These declarations must match `core/src/c_api.zig` exactly.
unsafe extern "C" {
    /// Writes a JSON device probe into `buffer`; returns the bytes written, or
    /// 0 if the driver could not be queried.
    fn inferno_probe_devices(buffer: *mut u8, capacity: u32) -> u32;

    /// Version of the embedded core library, as a NUL terminated C string.
    fn inferno_version() -> *const c_char;
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match arguments.as_slice() {
        [flag] if flag.as_str() == "--probe" => probe_devices(),
        [flag] if flag.as_str() == "--version" => {
            print_version();
            ExitCode::SUCCESS
        }
        _ => {
            print_usage();
            ExitCode::from(EXIT_USAGE)
        }
    }
}

fn print_usage() {
    println!("usage: inferno-server --probe | --version");
}

fn print_version() {
    // SAFETY: the core returns a pointer to a static, NUL terminated string.
    let version = unsafe { CStr::from_ptr(inferno_version()) };
    println!("{}", version.to_string_lossy());
}

/// Prints the JSON device probe from the core library.
fn probe_devices() -> ExitCode {
    let mut buffer = vec![0_u8; PROBE_BUFFER_BYTES];
    let capacity = u32::try_from(buffer.len()).expect("the probe buffer fits in u32");

    // SAFETY: the buffer is valid for `capacity` bytes and the core never
    // writes past that bound.
    let written = unsafe { inferno_probe_devices(buffer.as_mut_ptr(), capacity) };
    if written == 0 {
        eprintln!("inferno-server: device probe failed; see the core log for details");
        return ExitCode::FAILURE;
    }

    let document = &buffer[..written as usize];
    let document = match std::str::from_utf8(document) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("inferno-server: probe document is not UTF-8: {error}");
            return ExitCode::FAILURE;
        }
    };
    println!("{document}");
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_declaration_matches_the_core_library() {
        // SAFETY: the core returns a pointer to a static, NUL terminated string.
        let version = unsafe { CStr::from_ptr(inferno_version()) };
        assert!(version.to_string_lossy().starts_with("inferno "));
    }

    #[test]
    fn probe_declaration_writes_a_device_document() {
        let mut buffer = vec![0_u8; PROBE_BUFFER_BYTES];
        let capacity = u32::try_from(buffer.len()).expect("the probe buffer fits in u32");

        // SAFETY: the buffer is valid for `capacity` bytes and the core never
        // writes past that bound.
        let written = unsafe { inferno_probe_devices(buffer.as_mut_ptr(), capacity) };
        assert!(written > 0, "the probe reported an error");

        let document =
            std::str::from_utf8(&buffer[..written as usize]).expect("the probe document is UTF-8");
        assert!(document.contains("\"devices\""));
    }
}
