//! Guarded device allocations: payload first, a canary band after it.
//!
//! A device overflow cannot be prevented by a host type, so the honest
//! mechanism is detection: put 2 MiB of a known pattern immediately after the
//! payload, and scan it on demand (`verifyGuardsAll`). A kernel that writes
//! past its buffer tears the canary on the first run instead of showing up as
//! three different `max_abs_err` values sixteen runs later, named by the
//! allocation that was overrun.
//!
//! `compute-sanitizer memcheck` does not substitute for this: an overrun that
//! stays inside a larger allocation is not a driver error, and the I-028
//! incident (`State.indexScoreBytes`, 1.4 MB past a 64 KiB allocation) was
//! exactly that shape.
//!
//! The band is debug-only. With `INFERNO_GUARD_BANDS` unset, `allocate`
//! forwards straight to `DeviceBuffer.allocate` and the registry stays empty,
//! so release pays nothing but the branch. With it set, the engine calls
//! `verifyGuardsAll` after each decoder layer.
//!
//! The registry is a fixed array: allocation happens at init, never after, and
//! an overflow of the array is an error rather than a silent reinterpretation.
//! All entries are assumed to belong to the current context, which is true of
//! the executor (`State` on device 0); multi-rank allocations do not use this
//! module, and would need a context per entry before they could.
//!
//! `allocate` is also the one place a device allocation becomes a capability
//! handle: it returns `epochs.Writable`, so the caller cannot read the buffer
//! until it seals it, and it registers the buffer's write epoch (I-028 §5a).
//! Both mechanisms live on this same call so that a new allocation cannot get
//! one without the other.

const std = @import("std");

const cuda_context = @import("cuda_context.zig");
const epochs = @import("epochs.zig");

const assert = std.debug.assert;

/// Bytes of canary after every guarded payload.
pub const guard_bytes: u64 = 2 * 1024 * 1024;

/// Byte the canary is painted with.
pub const canary_byte: u8 = 0xa5;

/// Environment variable that turns guarded allocation on.
pub const enable_variable = "INFERNO_GUARD_BANDS";

/// Bytes one verification copy moves.
const staging_bytes: u64 = 1 << 20;

/// Guarded allocations one process may hold; the production graph needs fewer
/// than 200, and `State` frees its allocations at deinit.
const entry_capacity: u32 = 512;

pub const GuardError = error{
    /// The canary after one payload does not match the painted pattern.
    guard_band_torn,
    /// More guarded allocations live than the fixed registry can name.
    guard_registry_full,
    /// The payload is registered under another device context. Freeing or
    /// scanning it here would act on a coincidence of virtual addresses.
    guard_wrong_context,
};

/// One guarded allocation: the payload the caller sees, plus its label and the
/// context that owns it.
const Entry = struct {
    payload: cuda_context.DeviceBuffer,
    label: []const u8,
    scope: epochs.Scope,
};

/// Registry order is allocation order within a scope; `verifyGuardsAll` walks
/// the current scope's prefix-filtered view.
var entries: [entry_capacity]Entry = undefined;
var entry_count: u32 = 0;

/// Serialises the registry and the one staging window across the multi-rank
/// test's threads. Debug/test only: a release build never populates a registry.
var registry_mutex: std.Thread.Mutex = .{};

/// Test override; `null` means "follow `enable_variable`".
var enabled_override: ?bool = null;

/// One pinned staging window shared by every paint and scan. `PORTABLE`, so a
/// copy issued from any context may use it; the mutex keeps two contexts from
/// memsetting it at once.
var staging: cuda_context.PinnedHostBuffer = undefined;
var staging_ready: bool = false;

/// Whether guarded allocation is active.
pub fn enabled() bool {
    assert(enable_variable.len > 0);
    assert(guard_bytes > 0);

    if (enabled_override) |value| {
        return value;
    } else {
        const text = std.posix.getenv(enable_variable) orelse return false;
        return text.len > 0 and !std.mem.eql(u8, text, "0");
    }
}

/// Overrides `enabled` for tests; pass `null` to return to the environment.
pub fn setEnabledForTest(value: ?bool) void {
    assert(entry_count <= entry_capacity);
    assert(enable_variable.len > 0);

    enabled_override = value;
}

/// Allocates `payload_bytes` plus, when enabled, a canary band the caller
/// never sees, and registers the payload's write epoch. `payload_bytes` must
/// already be a whole allocation.
pub fn allocate(
    label: []const u8,
    payload_bytes: u64,
    stream: *const cuda_context.Stream,
    policy: epochs.Policy,
) !epochs.Writable {
    assert(label.len > 0);
    assert(payload_bytes > 0);
    assert(payload_bytes % 4096 == 0);

    if (!enabled()) {
        const plain = try cuda_context.DeviceBuffer.allocate(payload_bytes);
        errdefer plain.free();
        return epochs.open(label, plain, policy);
    } else {
        assert(entry_count <= entry_capacity);
    }

    const scope = epochs.currentScope();
    registry_mutex.lock();
    defer registry_mutex.unlock();
    if (entry_count >= entry_capacity) {
        return GuardError.guard_registry_full;
    } else {
        assert(entry_count < entry_capacity);
    }

    const allocation = try cuda_context.DeviceBuffer.allocate(
        payload_bytes + guard_bytes,
    );
    errdefer allocation.free();
    const payload = cuda_context.DeviceBuffer{
        .pointer = allocation.pointer,
        .bytes = payload_bytes,
    };
    try paintGuard(&payload, stream);
    entries[entry_count] = .{
        .payload = payload,
        .label = label,
        .scope = scope,
    };
    entry_count += 1;
    assert(entry_count <= entry_capacity);
    return epochs.open(label, payload, policy) catch |open_error| {
        entry_count -= 1;
        return open_error;
    };
}

/// Frees a payload `allocate` returned; other buffers free directly.
///
/// The registry is context-scoped (I-033): a pointer registered under another
/// context is refused with `guard_wrong_context` rather than freed against a
/// coincidence of virtual addresses.
pub fn free(buffer: *const cuda_context.DeviceBuffer) GuardError!void {
    assert(buffer.pointer != 0);
    assert(buffer.bytes > 0);
    assert(entry_count <= entry_capacity);

    // Release build with the bands off: no registry exists, so freeing must
    // stay driver-free. (A test/debug build keeps the epoch registry even
    // with the bands off, and the scope check is load-bearing there.)
    if (comptime !epochs.tracking) {
        if (!enabled()) {
            buffer.free();
            return;
        }
    }

    const scope = epochs.currentScope();
    registry_mutex.lock();
    defer registry_mutex.unlock();

    if (takeEntry(scope, buffer.pointer)) |entry| {
        assert(entry.payload.bytes == buffer.bytes);
        epochs.forget(buffer.pointer);
        const allocation = cuda_context.DeviceBuffer{
            .pointer = entry.payload.pointer,
            .bytes = entry.payload.bytes + guard_bytes,
        };
        allocation.free();
        if (entry_count == 0) {
            releaseStaging();
        } else {
            assert(entry_count > 0);
        }
        return;
    }

    if (foreignEntry(scope, buffer.pointer)) |foreign| {
        return refuseForeignFree(buffer, foreign.label, foreign.scope, scope);
    }
    // The bands were off at allocation, so this scope owns no guard entry, but
    // the epoch registry (test/debug builds) still knows the pointer.
    if (epochs.foreignEntry(buffer.pointer, scope)) |foreign| {
        return refuseForeignFree(buffer, foreign.label, foreign.scope, scope);
    }

    // Not guarded in this scope: a buffer allocated while bands were off, or
    // one that never went through `allocate`.
    epochs.forget(buffer.pointer);
    buffer.free();
}

/// Scans every guard band on `stream`. Returns the first torn band, naming the
/// allocation and the offset inside its payload where the write landed.
pub fn verifyGuardsAll(stream: *const cuda_context.Stream) !void {
    assert(@intFromPtr(stream.handle) != 0);
    assert(entry_count <= entry_capacity);

    if (!enabled() or entry_count == 0) {
        return;
    } else {
        assert(entry_count > 0);
    }

    const scope = epochs.currentScope();
    registry_mutex.lock();
    defer registry_mutex.unlock();
    const stage = try stagingWindow();
    for (entries[0..entry_count]) |entry| {
        if (entry.scope.eql(scope)) {
            try verifyEntry(stream, entry, stage);
        }
    }
    assert(entry_count <= entry_capacity);
}

/// The number of live guarded allocations in the current context; the guard
/// tests read it.
pub fn liveCount() u32 {
    assert(entry_count <= entry_capacity);
    assert(entry_capacity > 0);

    const scope = epochs.currentScope();
    registry_mutex.lock();
    defer registry_mutex.unlock();
    var count: u32 = 0;
    for (entries[0..entry_count]) |entry| {
        if (entry.scope.eql(scope)) {
            count += 1;
        }
    }
    return count;
}

/// Logs and names a cross-context free, then returns the refusal.
fn refuseForeignFree(
    buffer: *const cuda_context.DeviceBuffer,
    label: []const u8,
    owner: epochs.Scope,
    current: epochs.Scope,
) GuardError {
    assert(buffer.pointer != 0);
    assert(label.len > 0);

    var owner_text: [64]u8 = undefined;
    var current_text: [64]u8 = undefined;
    std.log.warn(
        "guarded free refused: buffer '{s}' at 0x{x} belongs to {s}, " ++
            "but the current context is {s}; not freeing another " ++
            "context's allocation",
        .{
            label,
            buffer.pointer,
            epochs.scopeText(owner, owner_text[0..]),
            epochs.scopeText(current, current_text[0..]),
        },
    );
    return GuardError.guard_wrong_context;
}

/// The guard entry for `pointer` under a scope other than `scope`, or `null`.
/// The caller holds `registry_mutex`.
fn foreignEntry(scope: epochs.Scope, pointer: u64) ?Entry {
    assert(pointer != 0);
    assert(entry_count <= entry_capacity);

    for (entries[0..entry_count]) |entry| {
        if (entry.payload.pointer == pointer and !entry.scope.eql(scope)) {
            return entry;
        }
    }
    return null;
}

/// Paints the band after `payload` with `canary_byte`.
fn paintGuard(
    payload: *const cuda_context.DeviceBuffer,
    stream: *const cuda_context.Stream,
) !void {
    assert(payload.bytes > 0);
    assert(guard_bytes % 4096 == 0);

    const stage = try stagingWindow();
    @memset(stage.slice(), canary_byte);
    const band = bandAfter(payload);
    var painted: u64 = 0;
    while (painted < band.bytes) {
        const chunk: u64 = @min(stage.bytes, band.bytes - painted);
        const destination = sliceOf(&band, painted, chunk);
        try stream.copy_h_to_d(&destination, stage, chunk);
        painted += chunk;
    }
    try stream.synchronize();
    assert(painted == guard_bytes);
}

/// Scans one entry's band and returns `guard_band_torn` on the first mismatch.
fn verifyEntry(
    stream: *const cuda_context.Stream,
    entry: Entry,
    stage: *const cuda_context.PinnedHostBuffer,
) !void {
    assert(entry.label.len > 0);
    assert(entry.payload.bytes > 0);

    const band = bandAfter(&entry.payload);
    var scanned: u64 = 0;
    while (scanned < band.bytes) {
        const chunk: u64 = @min(stage.bytes, band.bytes - scanned);
        const source = sliceOf(&band, scanned, chunk);
        try stream.copy_d_to_h(&stage.*, &source, chunk);
        try stream.synchronize();
        const bytes = stage.slice()[0..@intCast(chunk)];
        const tear = firstMismatch(bytes);
        if (tear) |offset| {
            std.log.warn(
                "guard band torn: '{s}' wrote past its payload " ++
                    "at +{d} bytes (payload {d} bytes, band byte {d} != {d})",
                .{
                    entry.label,
                    scanned + offset,
                    entry.payload.bytes,
                    bytes[@intCast(offset)],
                    canary_byte,
                },
            );
            return GuardError.guard_band_torn;
        } else {
            assert(tear == null);
        }
        scanned += chunk;
    }
    assert(scanned == band.bytes);
}

/// Offset of the first byte that is not the canary, or `null`.
fn firstMismatch(bytes: []const u8) ?u64 {
    assert(staging_bytes > 0);
    assert(canary_byte == 0xa5);

    for (bytes, 0..) |byte, index| {
        if (byte != canary_byte) {
            return @intCast(index);
        } else {
            assert(byte == canary_byte);
        }
    }
    return null;
}

/// The canary band that follows `payload`.
fn bandAfter(
    payload: *const cuda_context.DeviceBuffer,
) cuda_context.DeviceBuffer {
    assert(payload.pointer != 0);
    assert(payload.bytes > 0);

    return .{
        .pointer = payload.pointer + payload.bytes,
        .bytes = guard_bytes,
    };
}

/// A `bytes`-long view starting `offset` bytes into `buffer`.
fn sliceOf(
    buffer: *const cuda_context.DeviceBuffer,
    offset: u64,
    bytes: u64,
) cuda_context.DeviceBuffer {
    assert(offset + bytes <= buffer.bytes);
    assert(bytes > 0);

    return .{ .pointer = buffer.pointer + offset, .bytes = bytes };
}

/// Removes the entry for `pointer` in `scope`, if one is registered. The
/// caller holds `registry_mutex`.
fn takeEntry(scope: epochs.Scope, pointer: u64) ?Entry {
    assert(pointer != 0);
    assert(entry_count <= entry_capacity);

    for (entries[0..entry_count], 0..) |entry, index| {
        if (entry.payload.pointer == pointer and entry.scope.eql(scope)) {
            const found = entry;
            entry_count -= 1;
            entries[index] = entries[entry_count];
            return found;
        }
    }
    return null;
}

/// The shared pinned staging window, allocated once.
fn stagingWindow() !*cuda_context.PinnedHostBuffer {
    assert(staging_bytes % 4096 == 0);
    assert(staging_bytes <= guard_bytes);

    if (!staging_ready) {
        staging = try cuda_context.PinnedHostBuffer.allocate(staging_bytes);
        @memset(staging.slice(), canary_byte);
        staging_ready = true;
    } else {
        assert(staging.bytes == staging_bytes);
    }
    return &staging;
}

/// Releases the staging window when the last guarded allocation goes away, so
/// a later context (the next test, or a re-init) does not reuse a mapping that
/// belonged to the previous one.
fn releaseStaging() void {
    assert(entry_count == 0);
    assert(!staging_ready or staging.bytes == staging_bytes);

    if (staging_ready) {
        staging.free();
        staging_ready = false;
    } else {
        assert(!staging_ready);
    }
}

/// Zeroes a guarded payload and its band before the test frees it.
///
/// The guard test deliberately writes a non-zero pattern into device memory.
/// Releasing that memory back to the driver with the pattern still in it would
/// hand the next allocation 0xa5 bytes, and the engine's tests (I-028 R1) have
/// places that wrongly assume `cudaMalloc` zeroed them. Cleanup is not part of
/// the guard mechanism: it keeps this test from perturbing its neighbours.
fn wipeForFree(
    buffer: cuda_context.DeviceBuffer,
    stream: *const cuda_context.Stream,
) !void {
    assert(buffer.pointer != 0);
    assert(buffer.bytes > 0);

    const stage = try stagingWindow();
    @memset(stage.slice(), 0);
    const allocation = cuda_context.DeviceBuffer{
        .pointer = buffer.pointer,
        .bytes = buffer.bytes + guard_bytes,
    };
    var wiped: u64 = 0;
    while (wiped < allocation.bytes) {
        const chunk: u64 = @min(stage.bytes, allocation.bytes - wiped);
        const destination = sliceOf(&allocation, wiped, chunk);
        try stream.copy_h_to_d(&destination, stage, chunk);
        wiped += chunk;
    }
    try stream.synchronize();
    assert(wiped == allocation.bytes);
}

/// Retains device 0, or skips when this host has no driver.
fn testContext() !?cuda_context.Context {
    assert(@sizeOf(cuda_context.Context) > 0);
    assert(@sizeOf(cuda_context.Stream) > 0);

    const cuda_driver = @import("cuda_driver.zig");
    var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
    const queried = cuda_driver.query_devices(devices[0..]);
    const count = queried catch |err| switch (err) {
        error.driver_library_missing => return null,
        else => return err,
    };
    if (count == 0) {
        return null;
    } else {
        assert(count > 0);
    }
    const context = try cuda_context.Context.retain(&devices[0]);
    try context.set_current();
    return context;
}

test "a 1.4 MB write past a 64 KiB payload tears the guard band" {
    const maybe_context = try testContext();
    if (maybe_context == null) {
        return error.SkipZigTest;
    } else {
        assert(maybe_context != null);
    }
    const context = maybe_context.?;
    defer context.release();

    setEnabledForTest(true);
    defer setEnabledForTest(null);
    const stream = try cuda_context.Stream.create();
    defer stream.destroy();

    // The incident's shape: a 64 KiB payload that the score kernel writes
    // 1.4 MB into. Host copies stand in for the kernel here; both are an
    // unchecked write into the band.
    const writable = try allocate("index_scores", 64 * 1024, &stream, .must_write);
    const payload = writable.seal().device();
    defer free(&payload) catch {};
    try verifyGuardsAll(&stream);
    assert(liveCount() == 1);

    const written: u64 = 1400 * 1024;
    const stage = try cuda_context.PinnedHostBuffer.allocate(written);
    defer stage.free();
    @memset(stage.slice(), 0x5a);
    const overrun = cuda_context.DeviceBuffer{
        .pointer = payload.pointer,
        .bytes = written,
    };
    try stream.copy_h_to_d(&overrun, &stage, written);
    try stream.synchronize();

    try std.testing.expectError(
        GuardError.guard_band_torn,
        verifyGuardsAll(&stream),
    );
    try std.testing.expect(liveCount() == 1);
    try wipeForFree(payload, &stream);
}

test "a clean payload verifies and frees its registry entry" {
    const maybe_context = try testContext();
    if (maybe_context == null) {
        return error.SkipZigTest;
    } else {
        assert(maybe_context != null);
    }
    const context = maybe_context.?;
    defer context.release();

    setEnabledForTest(true);
    defer setEnabledForTest(null);
    const stream = try cuda_context.Stream.create();
    defer stream.destroy();

    const before = liveCount();
    const writable = try allocate("rope_cos", 4096, &stream, .must_write);
    const payload = writable.seal().device();
    try verifyGuardsAll(&stream);
    assert(liveCount() == before + 1);
    try wipeForFree(payload, &stream);
    try free(&payload);
    assert(liveCount() == before);
    try verifyGuardsAll(&stream);
}

test "a guarded payload from another context is refused here" {
    const maybe_context = try testContext();
    if (maybe_context == null) {
        return error.SkipZigTest;
    } else {
        assert(maybe_context != null);
    }
    const context = maybe_context.?;
    defer context.release();

    setEnabledForTest(true);
    defer setEnabledForTest(null);
    const stream = try cuda_context.Stream.create();
    defer stream.destroy();

    const real = epochs.currentScope();
    var foreign_id = real.id + 1;
    if (foreign_id == 0) {
        foreign_id = 2;
    }
    const foreign = epochs.Scope{ .id = foreign_id };
    try std.testing.expect(!foreign.eql(real));

    const before = liveCount();
    const writable = try allocate("rope_cos", 4096, &stream, .must_write);
    const payload = writable.seal().device();
    try std.testing.expectEqual(before + 1, liveCount());

    // Ask from a context that does not own the payload: the registry must not
    // resolve the address, and the refusal must not free anything.
    epochs.setScopeForTest(foreign);
    defer epochs.setScopeForTest(null);
    try std.testing.expectEqual(@as(u32, 0), liveCount());
    try std.testing.expectError(
        GuardError.guard_wrong_context,
        free(&payload),
    );
    try std.testing.expectEqual(@as(u32, 0), liveCount());
    // Scanning the foreign scope either: it has no entries to walk.
    try verifyGuardsAll(&stream);

    // In its own context the same payload is intact and frees normally.
    epochs.setScopeForTest(null);
    try std.testing.expectEqual(before + 1, liveCount());
    try wipeForFree(payload, &stream);
    try free(&payload);
    try std.testing.expectEqual(before, liveCount());
}

test "two live contexts each scan their own band and a cross-context free is refused" {
    const cuda_driver = @import("cuda_driver.zig");
    var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
    const queried = cuda_driver.query_devices(devices[0..]);
    const device_count = queried catch |err| switch (err) {
        error.driver_library_missing => return error.SkipZigTest,
        else => return err,
    };
    if (device_count < 2) {
        return error.SkipZigTest;
    } else {
        assert(device_count >= 2);
    }

    const context_a = try cuda_context.Context.retain(&devices[0]);
    defer context_a.release();
    const context_b = try cuda_context.Context.retain(&devices[1]);
    defer context_b.release();
    setEnabledForTest(true);
    defer setEnabledForTest(null);

    try context_a.set_current();
    epochs.nameScope(0);
    const stream_a = try cuda_context.Stream.create();
    defer {
        context_a.set_current() catch {};
        stream_a.destroy();
    }
    const writable_a = try allocate("rope_cos", 4096, &stream_a, .must_write);
    const payload_a = writable_a.seal().device();

    try context_b.set_current();
    epochs.nameScope(1);
    const stream_b = try cuda_context.Stream.create();
    defer {
        context_b.set_current() catch {};
        stream_b.destroy();
    }
    const writable_b = try allocate("rope_sin", 4096, &stream_b, .must_write);
    const payload_b = writable_b.seal().device();

    // Each context's registry sees exactly its own allocation.
    try std.testing.expectEqual(@as(u32, 1), liveCount());
    try verifyGuardsAll(&stream_b);

    // Tear B's band. A's scan must stay clean; a global registry would see
    // B's tear from A and this test would fail.
    const stage = try cuda_context.PinnedHostBuffer.allocate(4096);
    defer stage.free();
    @memset(stage.slice(), 0x5a);
    const tear = cuda_context.DeviceBuffer{
        .pointer = payload_b.pointer + payload_b.bytes,
        .bytes = 1,
    };
    try stream_b.copy_h_to_d(&tear, &stage, 1);
    try stream_b.synchronize();
    try std.testing.expectError(
        GuardError.guard_band_torn,
        verifyGuardsAll(&stream_b),
    );
    try context_a.set_current();
    try verifyGuardsAll(&stream_a);
    try std.testing.expectEqual(@as(u32, 1), liveCount());

    // A's payload under B's context: refused, naming GPU1 and GPU0.
    try context_b.set_current();
    try std.testing.expectError(
        GuardError.guard_wrong_context,
        free(&payload_a),
    );
    try std.testing.expectEqual(@as(u32, 1), liveCount());

    // Each payload frees in the context that owns it.
    try context_a.set_current();
    try wipeForFree(payload_a, &stream_a);
    try free(&payload_a);
    try context_b.set_current();
    try wipeForFree(payload_b, &stream_b);
    try free(&payload_b);
}
