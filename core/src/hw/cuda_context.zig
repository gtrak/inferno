//! Typed handles for the CUDA driver resources the engine drives.
//!
//! Each type owns exactly one driver resource and exposes the operations that
//! resource supports; the raw entry points stay behind `hw/cuda_driver.zig`.
//! Contexts are primary contexts, so they are shared with any other library in
//! the process (NVML, cuBLAS, a future graph capture) instead of forking the
//! device state. Buffers, streams and events belong to whichever context is
//! current when they are created, which is why every constructor asserts that
//! the caller made one current first.

const std = @import("std");

const cuda_driver = @import("cuda_driver.zig");

const assert = std.debug.assert;

/// `CU_STREAM_NON_BLOCKING`: streams created by the probe never synchronize
/// with the legacy default stream, so four concurrent streams stay concurrent.
const stream_flags_non_blocking: c_uint = 1;

/// `CU_MEMHOSTALLOC_PORTABLE`: the pinned host mapping is usable from every
/// context in the process, not just the one that allocated it.
const memhostalloc_portable: c_uint = 0x01;

/// `CU_EVENT_DEFAULT`: timing is enabled, which `cuEventElapsedTime` needs.
const event_flags_default: c_uint = 0;

pub const DriverError = cuda_driver.DriverError;

/// A retained primary context for one device.
pub const Context = struct {
    handle: cuda_driver.CuContext,
    device_index: u32,

    /// Retains (and thereby initializes) the device's primary context.
    pub fn retain(device: *const cuda_driver.Device) DriverError!Context {
        assert(device.index < cuda_driver.device_count_max);
        assert(device.handle >= 0);

        const table = try cuda_driver.functions();
        var handle: cuda_driver.CuContext = undefined;
        try cuda_driver.check_result(
            table.context.retain_primary(&handle, device.handle),
        );
        assert(@intFromPtr(handle) != 0);
        return .{ .handle = handle, .device_index = device.index };
    }

    /// Releases the retained primary context; the device stays usable by other
    /// libraries because the primary context lives in the driver.
    pub fn release(context: *const Context) void {
        assert(@intFromPtr(context.handle) != 0);
        assert(context.device_index < cuda_driver.device_count_max);

        const table = cuda_driver.functions() catch |err| {
            std.log.err(
                "cannot release context on GPU{d}: {s}",
                .{ context.device_index, @errorName(err) },
            );
            return;
        };
        cuda_driver.check_result(
            table.context.release_primary(@intCast(context.device_index)),
        ) catch |err| {
            std.log.err(
                "cuDevicePrimaryCtxRelease_v2 GPU{d}: {s}",
                .{ context.device_index, @errorName(err) },
            );
            return;
        };
    }

    /// Makes this context current for the calling thread, then verifies it.
    pub fn set_current(context: *const Context) DriverError!void {
        assert(@intFromPtr(context.handle) != 0);
        assert(context.device_index < cuda_driver.device_count_max);

        const table = try cuda_driver.functions();
        try cuda_driver.check_result(
            table.context.set_current(context.handle),
        );
        assert(try context.is_current());
    }

    /// Whether this context is the current one; the paired half of
    /// `set_current`.
    pub fn is_current(context: *const Context) DriverError!bool {
        assert(@intFromPtr(context.handle) != 0);
        assert(context.device_index < cuda_driver.device_count_max);

        const table = try cuda_driver.functions();
        var current: cuda_driver.CuContext = undefined;
        try cuda_driver.check_result(table.context.get_current(&current));
        assert(!is_null_handle(current));
        return current == context.handle;
    }

    /// Free device bytes as `cuMemGetInfo` reports them for this context.
    pub fn free_bytes(context: *const Context) DriverError!u64 {
        assert(@intFromPtr(context.handle) != 0);
        assert(context.device_index < cuda_driver.device_count_max);

        const table = try cuda_driver.functions();
        var free: usize = 0;
        var total: usize = 0;
        try cuda_driver.check_result(table.memory.info(&free, &total));
        assert(total > 0);
        return free;
    }

    /// Lets work in this context reach `peer`'s memory in place.
    ///
    /// Enabling twice is not an error: a previous probe run in the same process
    /// may already have done it.
    pub fn enable_peer_access(
        context: *const Context,
        peer: *const Context,
    ) DriverError!void {
        assert(context.device_index != peer.device_index);
        assert(@intFromPtr(peer.handle) != 0);

        try context.set_current();
        const table = try cuda_driver.functions();
        const result = table.context.enable_peer_access(peer.handle, 0);
        if (result == cuda_driver.peer_access_already_enabled_result) {
            return;
        } else {
            try cuda_driver.check_result(result);
        }
    }
};

/// A device allocation owned by the current context.
pub const DeviceBuffer = struct {
    pointer: cuda_driver.DevicePtr,
    bytes: u64,

    /// Allocates `bytes` of device memory in the current context.
    pub fn allocate(bytes: u64) DriverError!DeviceBuffer {
        assert(bytes > 0);
        assert(bytes % 4096 == 0);

        const table = try cuda_driver.functions();
        var pointer: cuda_driver.DevicePtr = 0;
        try cuda_driver.check_result(
            table.memory.alloc(&pointer, @intCast(bytes)),
        );
        assert(pointer != 0);
        return .{ .pointer = pointer, .bytes = bytes };
    }

    /// Frees the allocation; the owning context must be current.
    pub fn free(buffer: *const DeviceBuffer) void {
        assert(buffer.pointer != 0);
        assert(buffer.bytes > 0);

        const table = cuda_driver.functions() catch |err| {
            std.log.err(
                "cannot free a device buffer: {s}",
                .{@errorName(err)},
            );
            return;
        };
        cuda_driver.check_result(
            table.memory.free(buffer.pointer),
        ) catch |err| {
            std.log.err("cuMemFree_v2: {s}", .{@errorName(err)});
            return;
        };
    }
};

/// A page-locked host buffer that device copies can read and write directly.
pub const PinnedHostBuffer = struct {
    pointer: [*]u8,
    bytes: u64,

    /// Allocates `bytes` of pinned host memory.
    pub fn allocate(bytes: u64) DriverError!PinnedHostBuffer {
        assert(bytes > 0);
        assert(bytes % 4096 == 0);

        const table = try cuda_driver.functions();
        var pointer: ?*anyopaque = null;
        try cuda_driver.check_result(table.memory.host_alloc(
            &pointer,
            @intCast(bytes),
            memhostalloc_portable,
        ));
        const handle = pointer orelse return DriverError.driver_out_of_memory;
        assert(@intFromPtr(handle) != 0);
        return .{ .pointer = @ptrCast(handle), .bytes = bytes };
    }

    /// Releases the pinned host memory.
    pub fn free(buffer: *const PinnedHostBuffer) void {
        assert(@intFromPtr(buffer.pointer) != 0);
        assert(buffer.bytes > 0);

        const table = cuda_driver.functions() catch |err| {
            std.log.err(
                "cannot free a pinned buffer: {s}",
                .{@errorName(err)},
            );
            return;
        };
        cuda_driver.check_result(
            table.memory.host_free(@ptrCast(buffer.pointer)),
        ) catch |err| {
            std.log.err("cuMemFreeHost: {s}", .{@errorName(err)});
            return;
        };
    }

    pub fn slice(buffer: *const PinnedHostBuffer) []u8 {
        assert(@intFromPtr(buffer.pointer) != 0);
        assert(buffer.bytes > 0);
        return buffer.pointer[0..buffer.bytes];
    }
};

/// An asynchronous copy queue owned by the current context.
pub const Stream = struct {
    handle: cuda_driver.CuStream,

    pub fn create() DriverError!Stream {
        assert(stream_flags_non_blocking > 0);
        assert(@sizeOf(cuda_driver.CuStream) > 0);

        const table = try cuda_driver.functions();
        var handle: cuda_driver.CuStream = undefined;
        try cuda_driver.check_result(table.stream.create(
            &handle,
            stream_flags_non_blocking,
        ));
        assert(@intFromPtr(handle) != 0);
        return .{ .handle = handle };
    }

    /// Destroys the queue; the owning context must be current.
    pub fn destroy(stream: *const Stream) void {
        assert(@intFromPtr(stream.handle) != 0);
        assert(@sizeOf(cuda_driver.CuStream) > 0);

        const table = cuda_driver.functions() catch |err| {
            std.log.err("cannot destroy a stream: {s}", .{@errorName(err)});
            return;
        };
        cuda_driver.check_result(
            table.stream.destroy(stream.handle),
        ) catch |err| {
            std.log.err("cuStreamDestroy_v2: {s}", .{@errorName(err)});
            return;
        };
    }

    /// Blocks until every queued operation finished.
    pub fn synchronize(stream: *const Stream) DriverError!void {
        assert(@intFromPtr(stream.handle) != 0);
        assert(@sizeOf(cuda_driver.CuStream) > 0);

        const table = try cuda_driver.functions();
        try cuda_driver.check_result(
            table.stream.synchronize(stream.handle),
        );
        assert(@intFromPtr(stream.handle) != 0);
    }

    /// Enqueues a host-to-device copy of `bytes` bytes.
    pub fn copy_h_to_d(
        stream: *const Stream,
        destination: *const DeviceBuffer,
        source: *const PinnedHostBuffer,
        bytes: u64,
    ) DriverError!void {
        assert(bytes > 0);
        assert(bytes <= destination.bytes);
        assert(bytes <= source.bytes);

        const table = try cuda_driver.functions();
        try cuda_driver.check_result(table.memory.copy_h_to_d_async(
            destination.pointer,
            @ptrCast(source.pointer),
            @intCast(bytes),
            stream.handle,
        ));
        assert(destination.pointer != 0);
    }

    /// Enqueues a device-to-host copy of `bytes` bytes.
    pub fn copy_d_to_h(
        stream: *const Stream,
        destination: *const PinnedHostBuffer,
        source: *const DeviceBuffer,
        bytes: u64,
    ) DriverError!void {
        assert(bytes > 0);
        assert(bytes <= destination.bytes);
        assert(bytes <= source.bytes);

        const table = try cuda_driver.functions();
        try cuda_driver.check_result(table.memory.copy_d_to_h_async(
            @ptrCast(destination.pointer),
            source.pointer,
            @intCast(bytes),
            stream.handle,
        ));
        assert(source.pointer != 0);
    }

    /// Enqueues a device-to-device copy of `bytes` bytes. With peer access
    /// enabled the two buffers may live on different devices.
    pub fn copy_d_to_d(
        stream: *const Stream,
        destination: *const DeviceBuffer,
        source: *const DeviceBuffer,
        bytes: u64,
    ) DriverError!void {
        assert(bytes > 0);
        assert(bytes <= destination.bytes);
        assert(bytes <= source.bytes);

        const table = try cuda_driver.functions();
        try cuda_driver.check_result(table.memory.copy_d_to_d_async(
            destination.pointer,
            source.pointer,
            @intCast(bytes),
            stream.handle,
        ));
        assert(destination.bytes > 0);
    }
};

/// A timestamp recorded on a stream, used for device-side elapsed time.
pub const Event = struct {
    handle: cuda_driver.CuEvent,

    pub fn create() DriverError!Event {
        assert(event_flags_default == 0);
        assert(@sizeOf(cuda_driver.CuEvent) > 0);

        const table = try cuda_driver.functions();
        var handle: cuda_driver.CuEvent = undefined;
        try cuda_driver.check_result(table.event.create(
            &handle,
            event_flags_default,
        ));
        assert(@intFromPtr(handle) != 0);
        return .{ .handle = handle };
    }

    /// Destroys the event; the owning context must be current.
    pub fn destroy(event: *const Event) void {
        assert(@intFromPtr(event.handle) != 0);
        assert(@sizeOf(cuda_driver.CuEvent) > 0);

        const table = cuda_driver.functions() catch |err| {
            std.log.err("cannot destroy an event: {s}", .{@errorName(err)});
            return;
        };
        cuda_driver.check_result(
            table.event.destroy(event.handle),
        ) catch |err| {
            std.log.err("cuEventDestroy_v2: {s}", .{@errorName(err)});
            return;
        };
    }

    /// Records this event on `stream`, after the work already queued there.
    pub fn record(event: *const Event, stream: *const Stream) DriverError!void {
        assert(@intFromPtr(event.handle) != 0);
        assert(@intFromPtr(stream.handle) != 0);

        const table = try cuda_driver.functions();
        try cuda_driver.check_result(
            table.event.record(event.handle, stream.handle),
        );
        assert(@intFromPtr(event.handle) != 0);
    }

    /// Blocks until the event has been recorded.
    pub fn synchronize(event: *const Event) DriverError!void {
        assert(@intFromPtr(event.handle) != 0);
        assert(@sizeOf(cuda_driver.CuEvent) > 0);

        const table = try cuda_driver.functions();
        try cuda_driver.check_result(
            table.event.synchronize(event.handle),
        );
        assert(@intFromPtr(event.handle) != 0);
    }

    /// Nanoseconds between two recorded events on the same device.
    pub fn elapsed_nanoseconds(
        start: *const Event,
        end: *const Event,
    ) DriverError!u64 {
        assert(@intFromPtr(start.handle) != 0);
        assert(@intFromPtr(end.handle) != 0);

        const table = try cuda_driver.functions();
        var milliseconds: f32 = 0;
        try cuda_driver.check_result(
            table.event.elapsed_ms(&milliseconds, start.handle, end.handle),
        );
        if (milliseconds >= 0) {
            assert(@as(f64, milliseconds) <= 1e9);
            return @intFromFloat(@as(f64, milliseconds) * 1_000_000.0);
        } else {
            return DriverError.driver_unknown_failure;
        }
    }
};

fn is_null_handle(handle: cuda_driver.CuContext) bool {
    assert(@sizeOf(cuda_driver.CuContext) > 0);
    assert(@sizeOf(*anyopaque) > 0);
    return @intFromPtr(handle) == 0;
}

test "a primary context, stream, buffers and events work on device 0" {
    var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
    const queried = cuda_driver.query_devices(devices[0..]);
    const count = queried catch |err| switch (err) {
        error.driver_library_missing => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expect(count > 0);

    const context = try Context.retain(&devices[0]);
    defer context.release();
    try context.set_current();
    try std.testing.expect(try context.is_current());

    const free_before = try context.free_bytes();

    const buffer = try DeviceBuffer.allocate(4096);
    const staged = try PinnedHostBuffer.allocate(4096);
    const stream = try Stream.create();
    const start = try Event.create();
    const end = try Event.create();

    @memset(staged.slice(), 0x7f);
    try stream.copy_h_to_d(&buffer, &staged, 4096);
    try stream.synchronize();
    try std.testing.expect(buffer.bytes == 4096);

    try start.record(&stream);
    try stream.copy_d_to_h(&staged, &buffer, 4096);
    try end.record(&stream);
    try end.synchronize();
    const elapsed = try Event.elapsed_nanoseconds(&start, &end);
    try std.testing.expect(elapsed > 0);

    // Freed explicitly in reverse order so the leak check below sees the
    // memory returned to the driver.
    end.destroy();
    start.destroy();
    stream.destroy();
    staged.free();
    buffer.free();
    const free_after = try context.free_bytes();
    try std.testing.expect(free_before <= free_after + 32 * 1024 * 1024);
}
