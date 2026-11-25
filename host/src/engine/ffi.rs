//! The C ABI of the Zig core.
//!
//! This declaration is the fixed contract with `core/src/c_api.zig`; it is
//! implemented by issue I-15 and must not drift:
//!
//! ```c
//! typedef struct inferno_engine inferno_engine;
//! typedef void (*inferno_token_cb)(void* ctx, uint64_t req_id, uint32_t token_id, uint32_t flags);
//! extern inferno_engine* inferno_core_init(const char* config_json, uint32_t len);
//! extern void            inferno_core_shutdown(inferno_engine*);
//! extern const char*     inferno_core_version(void);
//! extern int32_t         inferno_core_submit(inferno_engine*, uint64_t req_id,
//!                                            const char* req_json, uint32_t len);
//! extern int32_t         inferno_core_cancel(inferno_engine*, uint64_t req_id);
//! extern void            inferno_core_set_token_cb(inferno_engine*, inferno_token_cb, void* ctx);
//! extern int32_t         inferno_core_metrics(inferno_engine*, char* buf, uint32_t cap);
//! ```
//!
//! The callback runs on a Zig worker thread: it may only push into the request's
//! channel, never hold a lock across the FFI boundary and never unwind. Only
//! plain C types cross the boundary and every buffer carries its capacity.

use std::ffi::c_void;

/// Opaque engine handle owned by the core.
#[repr(C)]
pub struct inferno_engine {
    _private: [u8; 0],
}

/// Token callback: `flags` carries the [`super::TOKEN_FLAG_*`] bits.
///
/// The name is the C ABI's, deliberately not Rust style.
#[allow(non_camel_case_types)]
pub type inferno_token_cb =
    Option<unsafe extern "C" fn(ctx: *mut c_void, req_id: u64, token_id: u32, flags: u32)>;

/// Negative code the core returns while the ABI is stubbed out.
pub const CORE_ERROR_NOT_IMPLEMENTED: i32 = -104;

/// True when the linked `libinferno_core.a` exports the engine ABI.
///
/// `host/build.rs` decides this from the archive's symbol table, so it flips to
/// `true` by itself the day issue I-15 lands.
pub const CORE_ABI_LINKED: bool = cfg!(not(inferno_core_abi_absent));

/// The linked core exports the ABI: declare it and let the linker bind it.
#[cfg(not(inferno_core_abi_absent))]
mod linked {
    use super::{inferno_engine, inferno_token_cb};
    use std::ffi::{c_char, c_void};

    unsafe extern "C" {
        pub fn inferno_core_init(config_json: *const c_char, len: u32) -> *mut inferno_engine;
        pub fn inferno_core_shutdown(engine: *mut inferno_engine);
        pub fn inferno_core_version() -> *const c_char;
        pub fn inferno_core_submit(
            engine: *mut inferno_engine,
            req_id: u64,
            req_json: *const c_char,
            len: u32,
        ) -> i32;
        pub fn inferno_core_cancel(engine: *mut inferno_engine, req_id: u64) -> i32;
        pub fn inferno_core_set_token_cb(
            engine: *mut inferno_engine,
            callback: inferno_token_cb,
            ctx: *mut c_void,
        );
        pub fn inferno_core_metrics(engine: *mut inferno_engine, buf: *mut c_char, cap: u32)
        -> i32;
    }
}

#[cfg(not(inferno_core_abi_absent))]
pub(crate) use linked::*;

/// The core has no engine ABI yet: same names, "not implemented" answers.
#[cfg(inferno_core_abi_absent)]
#[path = "ffi_absent.rs"]
mod absent;

#[cfg(inferno_core_abi_absent)]
pub(crate) use absent::*;
