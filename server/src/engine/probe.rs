//! The device probe ABI of the core (issue I-001).
//!
//! `inferno_probe_devices` and `inferno_version` are part of the core library
//! from the build skeleton on, independently of the engine ABI in [`super::ffi`].

use std::ffi::{CStr, c_char};

/// Room for the probe document of a fully populated machine.
const PROBE_BUFFER_BYTES: usize = 64 * 1024;

// These declarations must match `core/src/c_api.zig` exactly.
unsafe extern "C" {
    /// Writes a JSON device probe into `buffer`; returns the bytes written, or
    /// 0 if the driver could not be queried.
    fn inferno_probe_devices(buffer: *mut u8, capacity: u32) -> u32;

    /// Version of the embedded core library, as a NUL terminated C string.
    fn inferno_version() -> *const c_char;
}

/// The core library's version string.
pub fn version() -> String {
    // SAFETY: the core returns a pointer to a static, NUL terminated string.
    let version = unsafe { CStr::from_ptr(inferno_version()) };
    version.to_string_lossy().into_owned()
}

/// The core's JSON device probe.
pub fn probe_document() -> Result<String, String> {
    let mut buffer = vec![0_u8; PROBE_BUFFER_BYTES];
    let capacity = u32::try_from(buffer.len()).expect("the probe buffer fits in u32");

    // SAFETY: the buffer is valid for `capacity` bytes and the core never writes
    // past that bound.
    let written = unsafe { inferno_probe_devices(buffer.as_mut_ptr(), capacity) };
    if written == 0 {
        return Err("the core could not query the driver; see its log".to_owned());
    }
    let document = &buffer[..written as usize];
    String::from_utf8(document.to_vec())
        .map_err(|error| format!("the probe document is not UTF-8: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probe_sees_this_machine() {
        let document = probe_document().expect("the probe runs");
        assert!(document.contains("\"devices\""), "{document}");
    }

    #[test]
    fn the_version_is_the_inferno_core() {
        assert!(version().starts_with("inferno "), "{}", version());
    }
}
