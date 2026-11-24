//! CUDA driver API surface: the ABI, the symbol table and the error taxonomy.
//!
//! `hw/cuda_context.zig` wraps these raw entry points in typed handles, so the
//! rest of the engine never passes a bare `CUcontext` or `CUstream` around.
//! Every entry point returns a `CUresult`; `check_result` maps it onto the
//! closed `DriverError` set, so no driver code is ever silently dropped.

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

/// `CUresult` values from `cuda.h` (CUDA 13.2) that this layer maps.
const cu_success: CuResult = 0;
const cu_error_invalid_value: CuResult = 1;
const cu_error_out_of_memory: CuResult = 2;
const cu_error_not_initialized: CuResult = 3;
const cu_error_no_device: CuResult = 100;
const cu_error_invalid_device: CuResult = 101;
const cu_error_invalid_context: CuResult = 201;
const cu_error_invalid_handle: CuResult = 400;
const cu_error_peer_access_already_enabled: CuResult = 704;
const cu_error_peer_access_not_enabled: CuResult = 705;
const cu_error_not_permitted: CuResult = 800;
const cu_error_not_supported: CuResult = 801;

/// `CUdevice_attribute` values queried for the hardware probe.
const attribute_clock_khz: c_int = 13;
const attribute_multiprocessor_count: c_int = 16;
const attribute_memory_clock_khz: c_int = 36;
const attribute_l2_cache_bytes: c_int = 38;
const attribute_compute_capability_major: c_int = 75;
const attribute_compute_capability_minor: c_int = 76;

/// Upper bound on the devices reported; this machine has four.
pub const device_count_max: u32 = 64;

/// Long enough for any device name the driver reports.
pub const device_name_bytes_max: u32 = 128;

/// Long enough for a driver version such as "610.57.04".
pub const driver_version_bytes_max: u32 = 32;

/// Long enough for a bus id such as "0000:05:00.0" plus its terminator.
pub const pci_bus_id_bytes_max: u32 = 16;

/// C types from `cuda.h` that cross this boundary.
pub const CuResult = c_int;
pub const CuDevice = c_int;
pub const CuContext = *anyopaque;
pub const CuStream = *anyopaque;
pub const CuEvent = *anyopaque;
pub const DevicePtr = u64;

/// Enabling peer access twice is not a failure; the caller ignores this code.
pub const peer_access_already_enabled_result: CuResult =
    cu_error_peer_access_already_enabled;

pub const DriverError = error{
    driver_library_missing,
    driver_symbol_missing,
    driver_not_initialized,
    driver_no_device,
    driver_invalid_value,
    driver_invalid_device,
    driver_invalid_context,
    driver_invalid_handle,
    driver_out_of_memory,
    /// Peer access was already on, or is not on, when the call needed the
    /// other state.
    driver_peer_access,
    driver_not_permitted,
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
    multiprocessor_count: u32,
    l2_cache_bytes: u64,
    clock_khz: u32,
    memory_clock_khz: u32,
    pci_bus_id: [pci_bus_id_bytes_max]u8,
    pci_bus_id_length: u32,
    handle: CuDevice,

    pub fn name_slice(device: *const Device) []const u8 {
        assert(device.name_length <= device_name_bytes_max);
        assert(device.name_length > 0);
        return device.name[0..device.name_length];
    }

    /// PCI address, for example "0000:05:00.0"; also the sysfs directory name.
    pub fn pci_bus_id_slice(device: *const Device) []const u8 {
        assert(device.pci_bus_id_length <= pci_bus_id_bytes_max);
        assert(device.pci_bus_id_length > 0);
        return device.pci_bus_id[0..device.pci_bus_id_length];
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

/// `cuDevice*` entry points.
pub const DeviceFunctions = struct {
    count_get: *const fn (*c_int) callconv(.c) CuResult,
    get: *const fn (*CuDevice, c_int) callconv(.c) CuResult,
    name_get: *const fn ([*]u8, c_int, CuDevice) callconv(.c) CuResult,
    attribute_get: *const fn (
        *c_int,
        c_int,
        CuDevice,
    ) callconv(.c) CuResult,
    total_memory_get: *const fn (*u64, CuDevice) callconv(.c) CuResult,
    pci_bus_id_get: *const fn ([*]u8, c_int, CuDevice) callconv(.c) CuResult,
    can_access_peer: *const fn (
        *c_int,
        CuDevice,
        CuDevice,
    ) callconv(.c) CuResult,
};

/// `cuCtx*` and primary context entry points.
pub const ContextFunctions = struct {
    retain_primary: *const fn (*CuContext, CuDevice) callconv(.c) CuResult,
    release_primary: *const fn (CuDevice) callconv(.c) CuResult,
    set_current: *const fn (CuContext) callconv(.c) CuResult,
    get_current: *const fn (*CuContext) callconv(.c) CuResult,
    enable_peer_access: *const fn (
        CuContext,
        c_uint,
    ) callconv(.c) CuResult,
};

/// `cuMem*` allocation and copy entry points.
pub const MemoryFunctions = struct {
    alloc: *const fn (*DevicePtr, usize) callconv(.c) CuResult,
    free: *const fn (DevicePtr) callconv(.c) CuResult,
    info: *const fn (*usize, *usize) callconv(.c) CuResult,
    host_alloc: *const fn (
        *?*anyopaque,
        usize,
        c_uint,
    ) callconv(.c) CuResult,
    host_free: *const fn (*anyopaque) callconv(.c) CuResult,
    copy_h_to_d_async: *const fn (
        DevicePtr,
        *const anyopaque,
        usize,
        CuStream,
    ) callconv(.c) CuResult,
    copy_d_to_h_async: *const fn (
        *anyopaque,
        DevicePtr,
        usize,
        CuStream,
    ) callconv(.c) CuResult,
    copy_d_to_d_async: *const fn (
        DevicePtr,
        DevicePtr,
        usize,
        CuStream,
    ) callconv(.c) CuResult,
    copy_peer_async: *const fn (
        DevicePtr,
        CuContext,
        DevicePtr,
        CuContext,
        usize,
        CuStream,
    ) callconv(.c) CuResult,
};

/// `cuStream*` entry points.
pub const StreamFunctions = struct {
    create: *const fn (*CuStream, c_uint) callconv(.c) CuResult,
    destroy: *const fn (CuStream) callconv(.c) CuResult,
    synchronize: *const fn (CuStream) callconv(.c) CuResult,
};

/// `cuEvent*` entry points.
pub const EventFunctions = struct {
    create: *const fn (*CuEvent, c_uint) callconv(.c) CuResult,
    destroy: *const fn (CuEvent) callconv(.c) CuResult,
    record: *const fn (CuEvent, CuStream) callconv(.c) CuResult,
    synchronize: *const fn (CuEvent) callconv(.c) CuResult,
    elapsed_ms: *const fn (*f32, CuEvent, CuEvent) callconv(.c) CuResult,
};

/// Driver entry points resolved from `cuda_library_name`. Signatures are the C
/// ones from `cuda.h`, so `size_t` stays `usize` and flags stay `c_uint`.
pub const Functions = struct {
    init: *const fn (c_uint) callconv(.c) CuResult,
    driver_version_get: *const fn (*c_int) callconv(.c) CuResult,
    device: DeviceFunctions,
    context: ContextFunctions,
    memory: MemoryFunctions,
    stream: StreamFunctions,
    event: EventFunctions,
};

/// A symbol table is a dense array of function pointers; a field with any
/// other layout would break the size invariants asserted below.
fn is_pointer_table(comptime Table: type) bool {
    assert(@sizeOf(*const anyopaque) > 0);
    assert(std.meta.fields(Table).len > 0);

    return @sizeOf(Table) ==
        std.meta.fields(Table).len * @sizeOf(*const anyopaque);
}

comptime {
    // The top-level table is two function pointers plus the five groups.
    assert(@sizeOf(Functions) == @sizeOf(DeviceFunctions) +
        @sizeOf(ContextFunctions) +
        @sizeOf(MemoryFunctions) +
        @sizeOf(StreamFunctions) +
        @sizeOf(EventFunctions) +
        2 * @sizeOf(*const anyopaque));
    assert(is_pointer_table(DeviceFunctions));
    assert(is_pointer_table(ContextFunctions));
    assert(is_pointer_table(MemoryFunctions));
    assert(is_pointer_table(StreamFunctions));
    assert(is_pointer_table(EventFunctions));
}

const LoadState = enum { unloaded, loaded, failed };

/// The symbol table is resolved once per process and then read-only.
var driver_functions: Functions = undefined;
var load_state: LoadState = .unloaded;
var load_error: DriverError = error.driver_library_missing;
var load_mutex: std.Thread.Mutex = .{};

/// Returns the cached driver symbol table, loading it on first use.
pub fn functions() DriverError!*const Functions {
    assert(cuda_library_name.len > 0);
    assert(@sizeOf(Functions) > 0);

    return load();
}

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
    try check_result(table.device.count_get(&driver_device_count));
    if (driver_device_count > 0) {
        const reported: u32 = @intCast(driver_device_count);
        const used = @min(reported, @as(u32, @intCast(devices.len)));
        for (0..used) |index| {
            try fill_device(table, &devices[index], @intCast(index));
        }
        assert(used > 0);
        return used;
    } else {
        return DriverError.driver_no_device;
    }
}

/// Whether `from` may reach `to` over peer-to-peer copy.
pub fn can_access_peer(from: CuDevice, to: CuDevice) DriverError!bool {
    assert(from >= 0);
    assert(to >= 0);

    const table = try load();
    var can_access: c_int = 0;
    try check_result(table.device.can_access_peer(&can_access, from, to));
    if (can_access == 0) {
        return false;
    } else if (can_access == 1) {
        return true;
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

/// Returns the cached driver symbol table, loading it on first use.
fn load() DriverError!*const Functions {
    assert(cuda_library_name.len > 0);
    assert(@sizeOf(Functions) > 0);

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
        .device = try resolve_device_symbols(library),
        .context = try resolve_context_symbols(library),
        .memory = try resolve_memory_symbols(library),
        .stream = try resolve_stream_symbols(library),
        .event = try resolve_event_symbols(library),
    };
}

fn resolve_device_symbols(
    library: *const dl.Library,
) dl.DlError!DeviceFunctions {
    assert(cuda_library_name.len > 0);
    assert(@sizeOf(DeviceFunctions) > 0);

    return .{
        .count_get = @ptrCast(try library.symbol("cuDeviceGetCount")),
        .get = @ptrCast(try library.symbol("cuDeviceGet")),
        .name_get = @ptrCast(try library.symbol("cuDeviceGetName")),
        .attribute_get = @ptrCast(
            try library.symbol("cuDeviceGetAttribute"),
        ),
        .total_memory_get = @ptrCast(
            try library.symbol("cuDeviceTotalMem_v2"),
        ),
        .pci_bus_id_get = @ptrCast(
            try library.symbol("cuDeviceGetPCIBusId"),
        ),
        .can_access_peer = @ptrCast(
            try library.symbol("cuDeviceCanAccessPeer"),
        ),
    };
}

fn resolve_context_symbols(
    library: *const dl.Library,
) dl.DlError!ContextFunctions {
    assert(cuda_library_name.len > 0);
    assert(@sizeOf(ContextFunctions) > 0);

    return .{
        .retain_primary = @ptrCast(
            try library.symbol("cuDevicePrimaryCtxRetain"),
        ),
        .release_primary = @ptrCast(
            try library.symbol("cuDevicePrimaryCtxRelease_v2"),
        ),
        .set_current = @ptrCast(try library.symbol("cuCtxSetCurrent")),
        .get_current = @ptrCast(try library.symbol("cuCtxGetCurrent")),
        .enable_peer_access = @ptrCast(
            try library.symbol("cuCtxEnablePeerAccess"),
        ),
    };
}

fn resolve_memory_symbols(
    library: *const dl.Library,
) dl.DlError!MemoryFunctions {
    assert(cuda_library_name.len > 0);
    assert(@sizeOf(MemoryFunctions) > 0);

    return .{
        .alloc = @ptrCast(try library.symbol("cuMemAlloc_v2")),
        .free = @ptrCast(try library.symbol("cuMemFree_v2")),
        .info = @ptrCast(try library.symbol("cuMemGetInfo_v2")),
        .host_alloc = @ptrCast(try library.symbol("cuMemHostAlloc")),
        .host_free = @ptrCast(try library.symbol("cuMemFreeHost")),
        .copy_h_to_d_async = @ptrCast(
            try library.symbol("cuMemcpyHtoDAsync_v2"),
        ),
        .copy_d_to_h_async = @ptrCast(
            try library.symbol("cuMemcpyDtoHAsync_v2"),
        ),
        .copy_d_to_d_async = @ptrCast(
            try library.symbol("cuMemcpyDtoDAsync_v2"),
        ),
        .copy_peer_async = @ptrCast(
            try library.symbol("cuMemcpyPeerAsync"),
        ),
    };
}

fn resolve_stream_symbols(
    library: *const dl.Library,
) dl.DlError!StreamFunctions {
    assert(cuda_library_name.len > 0);
    assert(@sizeOf(StreamFunctions) > 0);

    return .{
        .create = @ptrCast(try library.symbol("cuStreamCreate")),
        .destroy = @ptrCast(try library.symbol("cuStreamDestroy_v2")),
        .synchronize = @ptrCast(
            try library.symbol("cuStreamSynchronize"),
        ),
    };
}

fn resolve_event_symbols(
    library: *const dl.Library,
) dl.DlError!EventFunctions {
    assert(cuda_library_name.len > 0);
    assert(@sizeOf(EventFunctions) > 0);

    return .{
        .create = @ptrCast(try library.symbol("cuEventCreate")),
        .destroy = @ptrCast(try library.symbol("cuEventDestroy_v2")),
        .record = @ptrCast(try library.symbol("cuEventRecord")),
        .synchronize = @ptrCast(try library.symbol("cuEventSynchronize")),
        .elapsed_ms = @ptrCast(try library.symbol("cuEventElapsedTime")),
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

    var handle: CuDevice = 0;
    try check_result(table.device.get(&handle, @intCast(index)));

    var name: [device_name_bytes_max]u8 = [_]u8{0} ** device_name_bytes_max;
    const name_bytes: c_int = @intCast(name.len);
    try check_result(table.device.name_get(&name, name_bytes, handle));

    var vram_bytes: u64 = 0;
    try check_result(table.device.total_memory_get(&vram_bytes, handle));

    var bus_id: [pci_bus_id_bytes_max]u8 = [_]u8{0} ** pci_bus_id_bytes_max;
    try check_result(table.device.pci_bus_id_get(
        &bus_id,
        @intCast(bus_id.len),
        handle,
    ));

    const major = try attribute_u32(
        table,
        handle,
        attribute_compute_capability_major,
    );
    const minor = try attribute_u32(
        table,
        handle,
        attribute_compute_capability_minor,
    );
    const multiprocessor_count = try attribute_u32(
        table,
        handle,
        attribute_multiprocessor_count,
    );
    const l2_cache_bytes: u64 = try attribute_u32(
        table,
        handle,
        attribute_l2_cache_bytes,
    );
    const clock_khz = try attribute_u32(table, handle, attribute_clock_khz);
    const memory_clock_khz = try attribute_u32(
        table,
        handle,
        attribute_memory_clock_khz,
    );

    device.* = .{
        .index = index,
        .name = name,
        .name_length = name_length_of(&name),
        .compute_capability_major = major,
        .compute_capability_minor = minor,
        .vram_bytes = vram_bytes,
        .multiprocessor_count = multiprocessor_count,
        .l2_cache_bytes = l2_cache_bytes,
        .clock_khz = clock_khz,
        .memory_clock_khz = memory_clock_khz,
        .pci_bus_id = bus_id,
        .pci_bus_id_length = name_length_of(&bus_id),
        .handle = handle,
    };
    assert(device.name_length > 0);
    assert(device.multiprocessor_count > 0);
}

/// Reads one device attribute the driver reports as a non-negative integer.
fn attribute_u32(
    table: *const Functions,
    handle: CuDevice,
    attribute: c_int,
) DriverError!u32 {
    assert(attribute >= 0);
    assert(@sizeOf(c_int) == 4);

    var value: c_int = 0;
    try check_result(table.device.attribute_get(&value, attribute, handle));
    if (value >= 0) {
        return @intCast(value);
    } else {
        return DriverError.driver_unknown_failure;
    }
}

/// Length of a NUL terminated string inside its fixed-size buffer.
fn name_length_of(name: anytype) u32 {
    assert(name.len > 0);
    assert(name.len <= device_name_bytes_max);

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
pub fn check_result(result: CuResult) DriverError!void {
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
        cu_error_no_device => DriverError.driver_no_device,
        cu_error_invalid_device => DriverError.driver_invalid_device,
        cu_error_invalid_context => DriverError.driver_invalid_context,
        cu_error_invalid_handle => DriverError.driver_invalid_handle,
        cu_error_peer_access_already_enabled => DriverError.driver_peer_access,
        cu_error_peer_access_not_enabled => DriverError.driver_peer_access,
        cu_error_not_permitted => DriverError.driver_not_permitted,
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
    try std.testing.expect(devices[0].multiprocessor_count > 0);
    try std.testing.expect(devices[0].l2_cache_bytes > 0);
    try std.testing.expect(devices[0].clock_khz > 0);
    try std.testing.expect(devices[0].memory_clock_khz > 0);
    try std.testing.expect(devices[0].pci_bus_id_length > 0);
    try std.testing.expectEqual(
        @as(u8, '0'),
        devices[0].pci_bus_id_slice()[0],
    );
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
        DriverError.driver_out_of_memory,
        map_result(cu_error_out_of_memory),
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
