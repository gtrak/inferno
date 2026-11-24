//! `dlopen`/`dlsym` access to a shared library.
//!
//! The CUDA driver is opened at run time instead of being linked, so the
//! server binary stays loadable (and buildable) on hosts without a GPU
//! toolchain in the link path.

const std = @import("std");

const assert = std.debug.assert;

/// `RTLD_NOW` from `<dlfcn.h>`: resolve every symbol before returning.
const rtld_now: c_int = 2;

pub const DlError = error{
    /// The shared library is not installed or not visible to the loader.
    library_missing,
    /// The library exists but does not export the requested symbol.
    symbol_missing,
};

pub const Library = struct {
    handle: *anyopaque,

    /// Resolves one symbol. The caller casts the address to a function pointer
    /// of the exact C signature the library documents.
    pub fn symbol(
        library: *const Library,
        name: [*:0]const u8,
    ) DlError!*anyopaque {
        assert(name[0] != 0);
        assert(@intFromPtr(library.handle) != 0);

        const address = dlsym(library.handle, name);
        if (address) |resolved| {
            return resolved;
        } else {
            return DlError.symbol_missing;
        }
    }
};

/// Opens `name` and keeps it open for the lifetime of the process.
pub fn open(name: [*:0]const u8) DlError!Library {
    assert(name[0] != 0);
    assert(rtld_now > 0);

    const handle = dlopen(name, rtld_now);
    if (handle) |library_handle| {
        assert(@intFromPtr(library_handle) != 0);
        return .{ .handle = library_handle };
    } else {
        return DlError.library_missing;
    }
}

extern "c" fn dlopen(name: [*:0]const u8, flags: c_int) ?*anyopaque;
extern "c" fn dlsym(handle: *anyopaque, name: [*:0]const u8) ?*anyopaque;

test "open finds the CUDA driver library and its cuInit symbol" {
    const library = try open("libcuda.so.1");
    try std.testing.expect(@intFromPtr(library.handle) != 0);

    const init_address = try library.symbol("cuInit");
    try std.testing.expect(@intFromPtr(init_address) != 0);
}

test "symbol reports a missing symbol as an error" {
    const library = try open("libcuda.so.1");
    const missing = library.symbol("inferno_symbol_that_does_not_exist");
    try std.testing.expectError(DlError.symbol_missing, missing);
}
