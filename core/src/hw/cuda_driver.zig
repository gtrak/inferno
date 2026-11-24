//! Minimal CUDA driver API layer: enough to enumerate this host's GPUs.
//!
//! Only the symbols the probe needs are resolved (see `Functions`). Every call
//! checks its `CUresult` and maps failures into `DriverError`, so callers see a
//! closed set of errors instead of raw driver codes.

const std = @import("std");

const dl = @import("dl.zig");

const assert = std.debug.assert;

/// SONAME of the library installed by the NVIDIA driver package.
const cuda_library_name = "libcuda.so.1";

/// procfs entry of the kernel module. It is the only source of the display
/// driver version: `cuDriverGetVersion` reports the CUDA API version instead.
const procfs_version_path = "/proc/driver/nvidia/version";

/// Generous bound for the single line in `procfs_version_path`.
const procfs_version_bytes_max = 512;

/// Version text used when procfs cannot be read.
const unknown_version_text = "unknown";

/// Values from `cuda.h` (CUDA 13.2).
const cu_success: CuResult = 0;
const cu_error_invalid_value: CuResult = 1;
const cu_error_out_of_memory: CuResult = 2;
const cu_error_not_initialized: CuResult = 3;
const cu_error_invalid_device: CuResult = 101;
const cu_error_not_supported: CuResult = 801;

/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` and `..._MINOR`.
const attribute_compute_capability_major: c_int = 75;
const attribute_compute_capability_minor: c_int = 76;

/// Upper bound on the devices reported; this machine has four.
pub const device_count_max: u32 = 64;

/// Long enough for any device name the driver reports.
pub const device_name_bytes_max: u32 = 128;

/// Long enough for a driver version such as "610.57.04".
pub const driver_version_bytes_max: u32 = 32;

/// C types from `cuda.h` that cross this boundary.
pub const CuResult = c_int;
pub const CuDevice = c_int;

pub const DriverError = error{
    driver_library_missing,
    driver_symbol_missing,
    driver_not_initialized,
    driver_invalid_value,
    driver_invalid_device,
    driver_out_of_memory,
    driver_not_supported,
    driver_unknown_failure,
};

pub const DriverVersionError = error{
    version_unavailable,
};

/// One GPU as reported by the driver.
pub const Device = struct {
    index: u32,
    name: [device_name_bytes_max]u8,
    name_length: u32,
    compute_capability_major: u32,
    compute_capability_minor: u32,
    vram_bytes: u64,

    pub fn name_slice(device: *const Device) []const u8 {
        assert(device.name_length <= device_name_bytes_max);
        assert(device.name_length > 0);
        return device.name[0..device.name_length];
    }
};

/// Display driver version, for example "610.57.04".
pub const DriverVersion = struct {
    text: [driver_version_bytes_max]u8,
    length: u32,

    pub fn text_slice(version: *const DriverVersion) []const u8 {
        assert(version.length <= driver_version_bytes_max);
        assert(version.length > 0);
        return version.text[0..version.length];
    }

    /// The version reported when procfs is unreadable.
    pub fn unknown() DriverVersion {
        assert(unknown_version_text.len <= driver_version_bytes_max);
        assert(unknown_version_text.len > 0);

        const end = unknown_version_text.len;
        var version: DriverVersion = .{
            .text = [_]u8{0} ** driver_version_bytes_max,
            .length = end,
        };
        @memcpy(version.text[0..end], unknown_version_text);
        return version;
    }
};

/// Fills `devices` with every device the driver reports, returning the count.
///
/// The driver is initialized on first use; devices beyond `devices.len` are not
/// reported.
pub fn query_devices(devices: []Device) DriverError!u32 {
    assert(devices.len > 0);
    assert(devices.len <= device_count_max);

    const table = try load();
    try check_result(table.init(0));
    var driver_device_count: c_int = 0;
    try check_result(table.device_count_get(&driver_device_count));
    if (driver_device_count > 0) {
        const reported: u32 = @intCast(driver_device_count);
        const used = @min(reported, @as(u32, @intCast(devices.len)));
        for (0..used) |index| {
            try fill_device(table, &devices[index], @intCast(index));
        }
        assert(used > 0);
        return used;
    } else {
        return DriverError.driver_unknown_failure;
    }
}

/// Returns the CUDA API version of the driver, for example 13030 for 13.3.
pub fn cuda_api_version() DriverError!u32 {
    assert(cuda_library_name.len > 0);
    assert(@sizeOf(CuResult) == @sizeOf(c_int));

    const table = try load();
    var version: c_int = 0;
    try check_result(table.driver_version_get(&version));
    if (version > 0) {
        assert(version > 0);
        return @intCast(version);
    } else {
        return DriverError.driver_unknown_failure;
    }
}

/// Reads the display driver version from the kernel module's procfs entry.
///
/// `version` is written only on success, so callers can pre-load a fallback.
pub fn display_driver_version(version: *DriverVersion) DriverVersionError!void {
    assert(procfs_version_path.len > 0);
    assert(driver_version_bytes_max > 0);

    var contents: [procfs_version_bytes_max]u8 = undefined;
    const file = std.fs.openFileAbsolute(procfs_version_path, .{}) catch {
        return DriverVersionError.version_unavailable;
    };
    defer file.close();

    const contents_length = file.readAll(&contents) catch {
        return DriverVersionError.version_unavailable;
    };
    const token = first_version_token(contents[0..contents_length]) orelse {
        return DriverVersionError.version_unavailable;
    };
    if (token.len <= driver_version_bytes_max) {
        @memcpy(version.text[0..token.len], token);
        version.length = @intCast(token.len);
    } else {
        return DriverVersionError.version_unavailable;
    }
    assert(version.length > 0);
}

/// Driver entry points resolved from `cuda_library_name`.
const Functions = struct {
    init: *const fn (c_uint) callconv(.c) CuResult,
    driver_version_get: *const fn (*c_int) callconv(.c) CuResult,
    device_count_get: *const fn (*c_int) callconv(.c) CuResult,
    device_get: *const fn (*CuDevice, c_int) callconv(.c) CuResult,
    device_name_get: *const fn ([*]u8, c_int, CuDevice) callconv(.c) CuResult,
    attribute_get: *const fn (*c_int, c_int, CuDevice) callconv(.c) CuResult,
    total_memory_get: *const fn (*u64, CuDevice) callconv(.c) CuResult,
};

const LoadState = enum { unloaded, loaded, failed };

/// The symbol table is resolved once per process and then read-only.
var driver_functions: Functions = undefined;
var load_state: LoadState = .unloaded;
var load_error: DriverError = error.driver_library_missing;
var load_mutex: std.Thread.Mutex = .{};

/// Returns the cached driver symbol table, loading it on first use.
fn load() DriverError!*const Functions {
    assert(@sizeOf(Functions) == 7 * @sizeOf(*const anyopaque));
    assert(cuda_library_name.len > 0);

    load_mutex.lock();
    defer load_mutex.unlock();

    switch (load_state) {
        .loaded => return &driver_functions,
        .failed => return load_error,
        .unloaded => {
            load_locked();
            if (load_state == .loaded) {
                return &driver_functions;
            } else {
                return load_error;
            }
        },
    }
}

/// Opens the driver and fills the symbol table; the caller holds `load_mutex`.
fn load_locked() void {
    assert(load_state == .unloaded);
    assert(@sizeOf(Functions) > 0);

    const library = dl.open(cuda_library_name) catch |err| {
        load_error = map_dl_error(err);
        load_state = .failed;
        return;
    };
    const table = resolve_symbols(&library) catch |err| {
        load_error = map_dl_error(err);
        load_state = .failed;
        return;
    };
    driver_functions = table;
    load_state = .loaded;
    assert(load_state == .loaded);
}

/// Resolves every driver function this layer calls.
fn resolve_symbols(library: *const dl.Library) dl.DlError!Functions {
    assert(cuda_library_name.len > 0);
    assert(@sizeOf(Functions) > 0);

    return .{
        .init = @ptrCast(try library.symbol("cuInit")),
        .driver_version_get = @ptrCast(
            try library.symbol("cuDriverGetVersion"),
        ),
        .device_count_get = @ptrCast(try library.symbol("cuDeviceGetCount")),
        .device_get = @ptrCast(try library.symbol("cuDeviceGet")),
        .device_name_get = @ptrCast(try library.symbol("cuDeviceGetName")),
        .attribute_get = @ptrCast(try library.symbol("cuDeviceGetAttribute")),
        .total_memory_get = @ptrCast(try library.symbol("cuDeviceTotalMem_v2")),
    };
}

/// Fills one device from the driver.
fn fill_device(
    table: *const Functions,
    device: *Device,
    index: u32,
) DriverError!void {
    assert(index < device_count_max);
    assert(device.name.len == device_name_bytes_max);

    var driver_device: CuDevice = 0;
    try check_result(table.device_get(&driver_device, @intCast(index)));

    var name: [device_name_bytes_max]u8 = [_]u8{0} ** device_name_bytes_max;
    const name_bytes: c_int = @intCast(name.len);
    try check_result(table.device_name_get(&name, name_bytes, driver_device));

    var major: c_int = 0;
    var minor: c_int = 0;
    try check_result(table.attribute_get(
        &major,
        attribute_compute_capability_major,
        driver_device,
    ));
    try check_result(table.attribute_get(
        &minor,
        attribute_compute_capability_minor,
        driver_device,
    ));

    var vram_bytes: u64 = 0;
    try check_result(table.total_memory_get(&vram_bytes, driver_device));

    device.* = .{
        .index = index,
        .name = name,
        .name_length = name_length_of(&name),
        .compute_capability_major = @intCast(major),
        .compute_capability_minor = @intCast(minor),
        .vram_bytes = vram_bytes,
    };
    assert(device.name_length > 0);
}

/// Length of the NUL terminated name inside the fixed-size buffer.
fn name_length_of(name: *const [device_name_bytes_max]u8) u32 {
    assert(name.len == device_name_bytes_max);
    assert(device_name_bytes_max > 0);

    const length = std.mem.indexOfScalar(u8, name, 0) orelse name.len;
    assert(length <= name.len);
    return @intCast(length);
}

/// First whitespace separated token that starts with a digit: the "610.57.04"
/// in the kernel module's "NVRM version: ..." line.
fn first_version_token(contents: []const u8) ?[]const u8 {
    assert(procfs_version_bytes_max >= contents.len);
    assert(driver_version_bytes_max > 0);

    var iterator = std.mem.tokenizeAny(u8, contents, " \t\r\n");
    while (iterator.next()) |token| {
        assert(token.len > 0);
        if (std.ascii.isDigit(token[0])) {
            return token;
        } else {
            // Not the version token; keep scanning to the end of the line.
        }
    }
    return null;
}

/// Turns a driver result code into an error.
fn check_result(result: CuResult) DriverError!void {
    assert(@sizeOf(CuResult) == @sizeOf(c_int));
    assert(cu_success == 0);

    if (result == cu_success) {
        return;
    } else {
        return map_result(result);
    }
}

/// Maps every driver code this layer can see onto `DriverError`.
fn map_result(result: CuResult) DriverError {
    assert(result != cu_success);
    assert(@sizeOf(CuResult) == 4);

    return switch (result) {
        cu_error_invalid_value => DriverError.driver_invalid_value,
        cu_error_out_of_memory => DriverError.driver_out_of_memory,
        cu_error_not_initialized => DriverError.driver_not_initialized,
        cu_error_invalid_device => DriverError.driver_invalid_device,
        cu_error_not_supported => DriverError.driver_not_supported,
        else => DriverError.driver_unknown_failure,
    };
}

fn map_dl_error(err: dl.DlError) DriverError {
    assert(cuda_library_name.len > 0);
    assert(@sizeOf(dl.DlError) > 0);

    return switch (err) {
        error.library_missing => DriverError.driver_library_missing,
        error.symbol_missing => DriverError.driver_symbol_missing,
    };
}

test "query_devices enumerates this machine's GPUs" {
    var devices: [device_count_max]Device = undefined;
    const count = query_devices(devices[0..]) catch |err| switch (err) {
        // A host without a driver is a supported deployment, not a failure.
        error.driver_library_missing => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expect(count > 0);
    try std.testing.expect(count <= device_count_max);
    try std.testing.expect(devices[0].name_length > 0);
    try std.testing.expect(devices[0].vram_bytes > 0);
}

test "the symbol table is resolved once and cached" {
    const first = try load();
    const second = try load();
    try std.testing.expect(first == second);
    try std.testing.expect(load_state == .loaded);
}

test "cuda_api_version reports a plausible CUDA version" {
    const version = cuda_api_version() catch |err| switch (err) {
        error.driver_library_missing => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expect(version >= 12000);
}

test "display_driver_version parses procfs" {
    var version: DriverVersion = undefined;
    try display_driver_version(&version);
    const text = version.text_slice();
    try std.testing.expect(text.len > 0);
    try std.testing.expect(std.ascii.isDigit(text[0]));
}

test "first_version_token skips non-version words" {
    const line = "NVRM version: NVIDIA UNIX Open Kernel Module" ++
        " for x86_64  610.57.04";
    const maybe_token = first_version_token(line);
    const token = maybe_token orelse return error.TestExpectedToken;
    try std.testing.expectEqualStrings("610.57.04", token);
    try std.testing.expect(first_version_token("no digits here") == null);
}

test "map_result covers representative driver codes" {
    try std.testing.expectEqual(
        DriverError.driver_invalid_value,
        map_result(cu_error_invalid_value),
    );
    try std.testing.expectEqual(
        DriverError.driver_unknown_failure,
        map_result(999),
    );
}

test "DriverVersion.unknown is a usable fallback" {
    const version = DriverVersion.unknown();
    try std.testing.expectEqualStrings("unknown", version.text_slice());
}
