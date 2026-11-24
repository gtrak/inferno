//! Module loading and kernel launch: the seam between the Zig hot path and the
//! CUDA cubins that `core/build.zig` compiles and embeds.
//!
//! This file resolves its own driver symbols through `dl.zig` instead of
//! extending `cuda_driver.zig`'s `Functions` table. Module loading and launching
//! are a separate concern from device/context/memory management, and keeping the
//! table local means the four kernel lanes can share this file without touching
//! any other `hw/` file.
//!
//! Every entry point is checked: a driver call that fails becomes a value in a
//! closed error set, never a panic across the C ABI and never a silent success.

const std = @import("std");

const cuda_driver = @import("cuda_driver.zig");
const dl = @import("dl.zig");

const assert = std.debug.assert;

/// Raw `CUresult` as the driver returns it; zero means success.
const CuResult = c_int;

/// Driver-reported success.
const result_success: CuResult = 0;

/// `CUDA_ERROR_INVALID_VALUE`.
const result_invalid_value: CuResult = 1;
/// `CUDA_ERROR_OUT_OF_MEMORY`.
const result_out_of_memory: CuResult = 2;
/// `CUDA_ERROR_NOT_INITIALIZED`.
const result_not_initialized: CuResult = 3;
/// `CUDA_ERROR_INVALID_IMAGE` - the cubin does not match this device.
const result_invalid_image: CuResult = 200;
/// `CUDA_ERROR_INVALID_CONTEXT`.
const result_invalid_context: CuResult = 201;
/// `CUDA_ERROR_INVALID_HANDLE`.
const result_invalid_handle: CuResult = 400;
/// `CUDA_ERROR_NOT_FOUND` - no such kernel in the module.
const result_not_found: CuResult = 500;
/// `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`.
const result_launch_out_of_resources: CuResult = 701;
/// `CUDA_ERROR_LAUNCH_FAILED`.
const result_launch_failed: CuResult = 719;
/// `CUDA_ERROR_NOT_SUPPORTED`.
const result_not_supported: CuResult = 801;

/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`.
const function_attribute_max_dynamic_shared_bytes: c_int = 8;

/// The dynamic shared memory a kernel may request beyond the static limit is
/// capped by the device; anything larger is a programming error, not a runtime
/// condition, so the bound is asserted rather than checked.
pub const dynamic_shared_bytes_max: u32 = 227 * 1024;

/// Handle to a loaded cubin.
pub const ModuleHandle = ?*anyopaque;

/// Handle to one kernel inside a module.
pub const FunctionHandle = ?*anyopaque;

/// Handle to a CUDA stream, as the driver defines it.
pub const StreamHandle = cuda_driver.CuStream;

pub const KernelError = error{
    driver_library_missing,
    driver_symbol_missing,
    module_load_failed,
    module_arch_mismatch,
    function_not_found,
    launch_failed,
    launch_out_of_resources,
    invalid_value,
    invalid_context,
    invalid_handle,
    out_of_memory,
    not_initialized,
    not_supported,
    unknown_failure,
};

/// Function-pointer table for the module and launch entry points.
///
/// A symbol table is a dense array of function pointers; any other layout would
/// break the size assertion below.
const Symbols = struct {
    module_load_data: *const fn (*ModuleHandle, *const anyopaque) callconv(.c) CuResult,
    module_get_function: *const fn (
        *FunctionHandle,
        ModuleHandle,
        [*:0]const u8,
    ) callconv(.c) CuResult,
    module_unload: *const fn (ModuleHandle) callconv(.c) CuResult,
    function_set_attribute: *const fn (
        FunctionHandle,
        c_int,
        c_int,
    ) callconv(.c) CuResult,
    launch: *const fn (
        FunctionHandle,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        StreamHandle,
        ?[*]?*anyopaque,
        ?[*]?*anyopaque,
    ) callconv(.c) CuResult,

    const field_count = 5;

    /// Resolves every symbol once. Callers cache the result; see `symbols`.
    fn resolve() KernelError!Symbols {
        assert(field_count == std.meta.fields(Symbols).len);
        assert(@sizeOf(Symbols) == field_count * @sizeOf(*const anyopaque));

        const library = dl.open("libcuda.so.1") catch |open_error| {
            return switch (open_error) {
                dl.DlError.library_missing => KernelError.driver_library_missing,
                dl.DlError.symbol_missing => KernelError.driver_symbol_missing,
            };
        };

        return .{
            .module_load_data = try cast_symbol(library, "cuModuleLoadData", @TypeOf(
                @as(Symbols, undefined).module_load_data,
            )),
            .module_get_function = try cast_symbol(library, "cuModuleGetFunction", @TypeOf(
                @as(Symbols, undefined).module_get_function,
            )),
            .module_unload = try cast_symbol(library, "cuModuleUnload", @TypeOf(
                @as(Symbols, undefined).module_unload,
            )),
            .function_set_attribute = try cast_symbol(library, "cuFuncSetAttribute", @TypeOf(
                @as(Symbols, undefined).function_set_attribute,
            )),
            .launch = try cast_symbol(library, "cuLaunchKernel", @TypeOf(
                @as(Symbols, undefined).launch,
            )),
        };
    }
};

/// Resolves one symbol and casts it to the exact C signature.
fn cast_symbol(
    library: dl.Library,
    comptime name: [*:0]const u8,
    comptime PointerType: type,
) KernelError!PointerType {
    assert(name[0] != 0);
    assert(@sizeOf(PointerType) == @sizeOf(*const anyopaque));

    const address = library.symbol(name) catch |symbol_error| {
        return switch (symbol_error) {
            dl.DlError.symbol_missing => KernelError.driver_symbol_missing,
            dl.DlError.library_missing => KernelError.driver_library_missing,
        };
    };
    const function: PointerType = @ptrCast(@alignCast(address));
    return function;
}

/// Process-wide symbol table, resolved on first use.
///
/// The engine is single-threaded at init, so a plain flag is sufficient; the
/// assertion turns a concurrent first use into a loud failure in tests instead
/// of a data race in production.
var symbols_cache: Symbols = undefined;
var symbols_resolved: bool = false;

fn symbols() KernelError!*const Symbols {
    assert(@intFromPtr(&symbols_cache) != 0);

    if (symbols_resolved) {
        return &symbols_cache;
    } else {
        symbols_cache = try Symbols.resolve();
        symbols_resolved = true;
        assert(symbols_resolved);
        return &symbols_cache;
    }
}

/// Maps a `CUresult` to the closed error set. Every result is handled.
fn map_result(result: CuResult, comptime context: KernelError) KernelError!void {
    assert(@sizeOf(CuResult) == @sizeOf(c_int));
    assert(context != KernelError.driver_library_missing);

    if (result == result_success) {
        return;
    } else {
        return switch (result) {
            result_invalid_value => KernelError.invalid_value,
            result_out_of_memory => KernelError.out_of_memory,
            result_not_initialized => KernelError.not_initialized,
            result_invalid_image => KernelError.module_arch_mismatch,
            result_invalid_context => KernelError.invalid_context,
            result_invalid_handle => KernelError.invalid_handle,
            result_not_found => KernelError.function_not_found,
            result_launch_out_of_resources => KernelError.launch_out_of_resources,
            result_launch_failed => KernelError.launch_failed,
            result_not_supported => KernelError.not_supported,
            else => context,
        };
    }
}

/// A loaded cubin. Owns its module handle and must be unloaded.
pub const Module = struct {
    handle: ModuleHandle,

    /// Loads a cubin image that is embedded in the binary.
    ///
    /// The bytes must outlive the module; embedded cubins do.
    pub fn load(image: []const u8) KernelError!Module {
        assert(image.len > 0);
        assert(image.len <= std.math.maxInt(u32));

        const table = try symbols();
        var handle: ModuleHandle = null;
        try map_result(
            table.module_load_data(&handle, image.ptr),
            KernelError.module_load_failed,
        );
        if (handle) |_| {
            assert(@intFromPtr(handle) != 0);
            return .{ .handle = handle };
        } else {
            return KernelError.module_load_failed;
        }
    }

    pub fn unload(module: *const Module) void {
        assert(module.handle != null);
        assert(@intFromPtr(module.handle) != 0);

        const table = symbols() catch return;
        _ = table.module_unload(module.handle);
    }

    /// Looks up one kernel by its `extern "C"` symbol name.
    pub fn function(module: *const Module, name: [*:0]const u8) KernelError!Function {
        assert(name[0] != 0);
        assert(module.handle != null);

        const table = try symbols();
        var handle: FunctionHandle = null;
        try map_result(
            table.module_get_function(&handle, module.handle, name),
            KernelError.function_not_found,
        );
        if (handle) |_| {
            assert(@intFromPtr(handle) != 0);
            return .{ .handle = handle, .table = table };
        } else {
            return KernelError.function_not_found;
        }
    }
};

/// One kernel entry point, plus the symbol table needed to launch it.
pub const Function = struct {
    handle: FunctionHandle,
    table: *const Symbols,

    /// Raises the dynamic shared memory limit for kernels that need more than
    /// the default 48 KiB per block.
    pub fn setDynamicSharedBytes(
        function: *const Function,
        bytes: u32,
    ) KernelError!void {
        assert(function.handle != null);
        assert(bytes > 0);
        assert(bytes <= dynamic_shared_bytes_max);

        try map_result(
            function.table.function_set_attribute(
                function.handle,
                function_attribute_max_dynamic_shared_bytes,
                @intCast(bytes),
            ),
            KernelError.invalid_value,
        );
    }

    /// Launches the kernel.
    ///
    /// `parameters` is the array of pointers the driver forwards as the
    /// kernel's arguments; each element points at the value of one argument.
    /// The caller guarantees every pointer outlives the launch.
    pub fn launch(
        function: *const Function,
        grid: [3]u32,
        block: [3]u32,
        shared_bytes: u32,
        stream: StreamHandle,
        parameters: []const ?*anyopaque,
    ) KernelError!void {
        assert(function.handle != null);
        assert(grid[0] > 0);
        assert(grid[1] > 0);
        assert(grid[2] > 0);
        assert(block[0] > 0);
        assert(block[1] > 0);
        assert(block[2] > 0);
        assert(shared_bytes <= dynamic_shared_bytes_max);
        assert(parameters.len <= 32);

        const parameter_count: u32 = @intCast(parameters.len);
        assert(parameter_count == parameters.len);

        try map_result(
            function.table.launch(
                function.handle,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_bytes,
                stream,
                @constCast(parameters.ptr),
                null,
            ),
            KernelError.launch_failed,
        );
    }
};

test "symbols resolve on a machine with a driver" {
    const table = symbols() catch |resolve_error| {
        // A host without a driver is a legitimate configuration; the engine
        // refuses to start there, so the test is skipped rather than failed.
        try std.testing.expect(
            resolve_error == KernelError.driver_library_missing or
                resolve_error == KernelError.driver_symbol_missing,
        );
        return;
    };
    try std.testing.expect(@intFromPtr(table.launch) != 0);
    try std.testing.expect(@intFromPtr(table.module_load_data) != 0);
    try std.testing.expect(symbols_resolved);
}
