//! Fallbacks for the engine C ABI.
//!
//! Compiled only while `libinferno_core.a` does not export the engine ABI (issue
//! I-15); `server/build.rs` decides that from the archive's symbol table. They
//! carry the same names and signatures as `ffi.rs`, so the rest of the server
//! compiles against one spelling and the failure is a plain "not implemented"
//! error instead of a link error. `--engine mock` is the supported mode until
//! then.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, c_void};

use super::{CORE_ERROR_NOT_IMPLEMENTED, inferno_engine, inferno_token_cb};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_core_init(
    _config_json: *const c_char,
    _len: u32,
) -> *mut inferno_engine {
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_core_shutdown(_engine: *mut inferno_engine) {}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_core_version() -> *const c_char {
    c"inferno-server (core ABI not linked yet: issue I-15)".as_ptr()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_core_submit(
    _engine: *mut inferno_engine,
    _req_id: u64,
    _req_json: *const c_char,
    _len: u32,
) -> i32 {
    CORE_ERROR_NOT_IMPLEMENTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_core_cancel(_engine: *mut inferno_engine, _req_id: u64) -> i32 {
    CORE_ERROR_NOT_IMPLEMENTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_core_set_token_cb(
    _engine: *mut inferno_engine,
    _callback: inferno_token_cb,
    _ctx: *mut c_void,
) {
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_core_metrics(
    _engine: *mut inferno_engine,
    _buf: *mut c_char,
    _cap: u32,
) -> i32 {
    CORE_ERROR_NOT_IMPLEMENTED
}
