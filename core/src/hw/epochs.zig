//! Capability handles and the per-buffer write epoch (I-028 §5a/§5b).
//!
//! The hazard this closes is *not* a device out-of-bounds write — nothing in
//! Zig prevents that — but a **read of a buffer whose content is void**. Two of
//! tonight's five bugs had that shape: `State.reset` zeroed every allocation
//! including the mRoPE tables, and `Model.reset` did not refill them, so the
//! next `applyRope` rotated by `cos = 0, sin = 0`. Nothing crashed; the answer
//! was merely wrong, and it was wrong differently on different runs.
//!
//! So every allocation carries an `Epoch`. It starts `never`, becomes `written`
//! only when a declared data path writes it, and returns to `never` when
//! `State.reset` zeroes it — because a zero fill carries no meaning, which is
//! exactly what blanket zeroing used to hide. A launch that declares `read` on
//! a `never` buffer fails with `read_before_write`, naming the kernel and the
//! buffer:
//!
//!     read before write: kernel 'attention.apply_rope' reads 'rope_cos',
//!     which no data path has written since allocation or reset
//!
//! **Cost is zero outside test and debug builds.** `tracking` is a comptime
//! `false` for a release build of the library, so `open`, `check` and the
//! registry itself are not emitted — `strings zig-out/lib/libinferno_core.a`
//! finds neither log format below. `zig build test` keeps them, because it runs
//! at `ReleaseSafe`: gating on `.Debug` alone would have made every check in
//! this file dead in this project's own gate, which is how three guards got
//! through tonight as decoration.
//!
//! **Honest limits.** Zig has no field privacy and no linear types, so
//! `Writable` cannot be made *unreadable* — `.pointer` is a field and fields are
//! public. What the pair does give: `guards.allocate` returns `Writable` and
//! `Writable` has no read accessor, so the only way to obtain a readable
//! `Sealed` is `seal()`, and `fillVoid` makes a zero fill satisfy nothing.
//! Reaching past that takes a deliberate bypass, which is the issue's stated
//! goal. The load-bearing check is the epoch, not the type.

const std = @import("std");
const builtin = @import("builtin");

const cuda_context = @import("cuda_context.zig");
const cuda_driver = @import("cuda_driver.zig");

const assert = std.debug.assert;

/// Whether epoch tracking is compiled in. Test and debug builds keep it; a
/// release build of the library emits none of it.
pub const tracking: bool = builtin.is_test or builtin.mode == .Debug;

/// How a launch uses one buffer.
pub const Access = enum {
    /// The kernel only reads it.
    read,
    /// The kernel only writes it.
    write,
    /// The kernel reads and writes it; a read is still a read.
    read_write,
};

/// What a buffer's contract says about its contents at allocation.
pub const Policy = enum {
    /// Every read needs a real write first. This is the default, and it is the
    /// policy that makes a missing write visible.
    must_write,
    /// Zero content *is* the contract (MoE padding, KV tile tails). The
    /// exemption is an allocation-time property, never a per-reset operation.
    initially_zero,
};

/// Whether a buffer's content has ever been produced by a data path.
pub const Epoch = enum {
    /// Allocated, or reset: the contents are void and may not be read.
    never,
    /// A declared write has landed since allocation or the last reset.
    written,
};

/// Live allocations the registry can name. `State` allocates about 70; the
/// kernel unit tests add their own, each of which frees its entry.
const capacity: u32 = 1024;

/// Slot value for a handle whose buffer is not registered (release builds).
const unregistered: u32 = std.math.maxInt(u32);

pub const EpochError = error{
    /// A launch declared `read` on a buffer whose epoch is `never`.
    read_before_write,
    /// A launch declared a buffer that `open` never registered, so the check
    /// cannot judge it. Silence here would make the whole mechanism vacuous.
    epoch_unknown_buffer,
    /// More live allocations than the fixed registry can name.
    epoch_registry_full,
};

/// One buffer a launch touches, and how.
pub const Decl = struct {
    label: []const u8,
    pointer: cuda_driver.DevicePtr,
    access: Access,
};

/// One registered allocation.
const Entry = struct {
    label: []const u8,
    pointer: cuda_driver.DevicePtr,
    bytes: u64,
    policy: Policy,
    epoch: Epoch,
};

var entries: [capacity]Entry = undefined;
var entry_count: u32 = 0;

/// A buffer that may only be written, never read: the §5a capability.
///
/// There is no `slice`, no `device()`, no download and no read accessor of any
/// kind. The only way to obtain a readable `Sealed` is `seal()`, so the door
/// `guards.allocate` opens cannot be walked through into a kernel by accident.
/// Fills go through `fillVoid`, which is where the "a zero fill is not a write"
/// rule lives.
pub const Writable = struct {
    label: []const u8,
    pointer: cuda_driver.DevicePtr,
    bytes: u64,
    slot: u32,

    /// The payload size in bytes, which is not a read of the contents.
    pub fn size(self: Writable) u64 {
        assert(self.bytes > 0);
        assert(self.label.len > 0);

        return self.bytes;
    }

    /// Consumes the writable view and returns the readable one.
    pub fn seal(self: Writable) Sealed {
        assert(self.bytes > 0);
        assert(self.label.len > 0);

        return .{
            .label = self.label,
            .pointer = self.pointer,
            .bytes = self.bytes,
            .slot = self.slot,
        };
    }
};

/// A buffer whose contents a kernel may read, because it was sealed.
pub const Sealed = struct {
    label: []const u8,
    pointer: cuda_driver.DevicePtr,
    bytes: u64,
    slot: u32,

    /// The payload size in bytes.
    pub fn size(self: Sealed) u64 {
        assert(self.bytes > 0);
        assert(self.label.len > 0);

        return self.bytes;
    }

    /// Re-opens a write view; `Writable.fill` still leaves the epoch `never`,
    /// so a reset cannot use this to satisfy a later read.
    pub fn openWrite(self: Sealed) Writable {
        assert(self.pointer != 0);
        assert(self.bytes > 0);

        return .{
            .label = self.label,
            .pointer = self.pointer,
            .bytes = self.bytes,
            .slot = self.slot,
        };
    }

    /// The raw view for the host-copy paths.
    pub fn device(self: Sealed) cuda_context.DeviceBuffer {
        assert(self.pointer != 0);
        assert(self.bytes > 0);

        return .{ .pointer = self.pointer, .bytes = self.bytes };
    }

    /// This buffer's epoch; `written` when tracking is compiled out.
    pub fn written(self: Sealed) Epoch {
        assert(self.bytes > 0);
        assert(self.label.len > 0);

        if (comptime tracking) {
            return epochOf(self.pointer) orelse Epoch.never;
        } else {
            return Epoch.written;
        }
    }
};

/// Registers `buffer` and returns its capability handle.
pub fn open(
    label: []const u8,
    buffer: cuda_context.DeviceBuffer,
    policy: Policy,
) EpochError!Writable {
    assert(label.len > 0);
    assert(buffer.pointer != 0);
    assert(buffer.bytes > 0);

    if (comptime tracking) {
        return openTracked(label, buffer, policy);
    } else {
        return .{
            .label = label,
            .pointer = buffer.pointer,
            .bytes = buffer.bytes,
            .slot = unregistered,
        };
    }
}

/// Drops the entry for a freed pointer, so a re-allocation cannot inherit it.
pub fn forget(pointer: cuda_driver.DevicePtr) void {
    assert(pointer != 0);

    if (comptime tracking) {
        forgetTracked(pointer);
    }
}

/// Sets one buffer's epoch to `never`; `unknown` pointers are not an error.
pub fn markEmpty(pointer: cuda_driver.DevicePtr) void {
    assert(pointer != 0);

    if (comptime tracking) {
        writeEpoch(pointer, .never);
    }
}

/// Marks one buffer written; an unregistered pointer is left alone.
pub fn markWritten(pointer: cuda_driver.DevicePtr) void {
    assert(pointer != 0);

    if (comptime tracking) {
        writeEpoch(pointer, .written);
    }
}

/// Zeroes a payload and leaves its epoch `never`.
///
/// This is `State.reset`: every allocation is zeroed, and every allocation's
/// content is thereby void. That is the whole point of the check — a blanket
/// zero fill used to look like "the buffer has been written", which is exactly
/// how the mRoPE tables were read as `cos = 0, sin = 0` without a word.
pub fn fillVoid(
    buffer: cuda_context.DeviceBuffer,
    stream: *const cuda_context.Stream,
    staging: *const cuda_context.PinnedHostBuffer,
) !void {
    assert(buffer.pointer != 0);
    assert(buffer.bytes > 0);

    try fillZeros(buffer, stream, staging);
    if (comptime tracking) {
        writeEpoch(buffer.pointer, .never);
    }
}

/// The epoch of a registered pointer, or `null` when it is not registered.
pub fn epochOf(pointer: cuda_driver.DevicePtr) ?Epoch {
    assert(pointer != 0);

    if (comptime tracking) {
        const slot = slotOf(pointer) orelse return null;
        return entries[slot].epoch;
    } else {
        return null;
    }
}

/// The registered label of a pointer, or `null`.
pub fn labelOf(pointer: cuda_driver.DevicePtr) ?[]const u8 {
    assert(pointer != 0);

    if (comptime tracking) {
        const slot = slotOf(pointer) orelse return null;
        return entries[slot].label;
    } else {
        return null;
    }
}

/// Live registered allocations; the tests read it.
pub fn liveCount() u32 {
    assert(capacity > 0);

    if (comptime tracking) {
        assert(entry_count <= capacity);
        return entry_count;
    } else {
        return 0;
    }
}

/// Enforces a launch's declared accesses, then records its writes.
///
/// `kernel` is the kernel's own name: the point of the error is that it says
/// which launch to look at. Reads are checked before any write is recorded, so
/// a kernel that both reads and writes one buffer is still refused.
pub fn check(kernel: []const u8, decls: []const Decl) EpochError!void {
    assert(kernel.len > 0);
    assert(decls.len > 0);

    if (comptime tracking) {
        return checkTracked(kernel, decls);
    } else {
        return;
    }
}

/// `open`, with the registry. Only reachable from a build that tracks epochs.
fn openTracked(
    label: []const u8,
    buffer: cuda_context.DeviceBuffer,
    policy: Policy,
) EpochError!Writable {
    assert(label.len > 0);
    assert(buffer.bytes > 0);
    assert(entry_count <= capacity);

    // A second `open` of a live pointer returns the existing entry: a buffer
    // has one epoch, whoever asks for it.
    if (slotOf(buffer.pointer)) |slot| {
        assert(entries[slot].pointer == buffer.pointer);
        return .{
            .label = entries[slot].label,
            .pointer = entries[slot].pointer,
            .bytes = entries[slot].bytes,
            .slot = slot,
        };
    }
    if (entry_count >= capacity) {
        return EpochError.epoch_registry_full;
    } else {
        assert(entry_count < capacity);
    }
    entries[entry_count] = .{
        .label = label,
        .pointer = buffer.pointer,
        .bytes = buffer.bytes,
        .policy = policy,
        .epoch = .never,
    };
    const slot = entry_count;
    entry_count += 1;
    assert(entry_count <= capacity);
    return .{
        .label = label,
        .pointer = buffer.pointer,
        .bytes = buffer.bytes,
        .slot = slot,
    };
}

/// `forget`, with the registry.
fn forgetTracked(pointer: cuda_driver.DevicePtr) void {
    assert(pointer != 0);
    assert(entry_count <= capacity);

    if (slotOf(pointer)) |slot| {
        entry_count -= 1;
        entries[slot] = entries[entry_count];
    }
    assert(entry_count <= capacity);
}

/// Sets one registered buffer's epoch; a foreign pointer is left alone.
fn writeEpoch(pointer: cuda_driver.DevicePtr, epoch: Epoch) void {
    assert(pointer != 0);
    assert(entry_count <= capacity);

    if (slotOf(pointer)) |slot| {
        entries[slot].epoch = epoch;
    }
}

/// `check`, with the registry: reads first, then the writes are recorded.
fn checkTracked(kernel: []const u8, decls: []const Decl) EpochError!void {
    assert(kernel.len > 0);
    assert(decls.len > 0);
    assert(decls.len <= 16);

    for (decls) |decl| {
        assert(decl.label.len > 0);
        assert(decl.pointer != 0);
        const slot = slotOf(decl.pointer) orelse {
            std.log.warn(
                "unknown buffer: kernel '{s}' declares '{s}' at 0x{x}, which no " ++
                    "allocation registered; the epoch check cannot judge it",
                .{ kernel, decl.label, decl.pointer },
            );
            return EpochError.epoch_unknown_buffer;
        };
        const entry = entries[slot];
        if (reads(decl.access) and entry.epoch == .never and entry.policy == .must_write) {
            std.log.warn(
                "read before write: kernel '{s}' reads '{s}', which no data " ++
                    "path has written since allocation or reset " ++
                    "({d} bytes at 0x{x})",
                .{ kernel, decl.label, entry.bytes, entry.pointer },
            );
            return EpochError.read_before_write;
        } else {
            assert(!reads(decl.access) or
                entry.epoch == .written or
                entry.policy == .initially_zero);
        }
    }
    for (decls) |decl| {
        if (writes(decl.access)) {
            writeEpoch(decl.pointer, .written);
        } else {
            assert(!writes(decl.access));
        }
    }
}

/// Whether an access reads its buffer.
fn reads(access: Access) bool {
    assert(@sizeOf(Access) == 1);
    assert(unregistered > capacity);

    return access == .read or access == .read_write;
}

/// Whether an access writes its buffer.
fn writes(access: Access) bool {
    assert(@sizeOf(Access) == 1);
    assert(capacity > 0);

    return access == .write or access == .read_write;
}

/// Registry slot holding `pointer`, or `null`.
fn slotOf(pointer: cuda_driver.DevicePtr) ?u32 {
    assert(pointer != 0);
    assert(entry_count <= capacity);

    for (entries[0..entry_count], 0..) |entry, slot| {
        if (entry.pointer == pointer) {
            return @intCast(slot);
        } else {
            assert(entry.pointer != pointer);
        }
    }
    return null;
}

/// Zeroes a payload through pinned staging, chunk by chunk.
fn fillZeros(
    buffer: cuda_context.DeviceBuffer,
    stream: *const cuda_context.Stream,
    staging: *const cuda_context.PinnedHostBuffer,
) !void {
    assert(buffer.pointer != 0);
    assert(buffer.bytes > 0);
    assert(staging.bytes > 0);

    @memset(staging.slice(), 0);
    var written: u64 = 0;
    while (written < buffer.bytes) {
        const chunk: u64 = @min(staging.bytes, buffer.bytes - written);
        const destination = cuda_context.DeviceBuffer{
            .pointer = buffer.pointer + written,
            .bytes = chunk,
        };
        try stream.copy_h_to_d(&destination, staging, chunk);
        written += chunk;
    }
    try stream.synchronize();
    assert(written == buffer.bytes);
}

/// A buffer for the host-side tests: no device memory is touched by `open`,
/// `seal`, `check` or `markAllEmpty`, so a bogus pointer is enough.
fn testSealed(label: []const u8, pointer: cuda_driver.DevicePtr) !Sealed {
    assert(label.len > 0);
    assert(pointer != 0);

    const buffer = cuda_context.DeviceBuffer{ .pointer = pointer, .bytes = 4096 };
    const writable = try open(label, buffer, .must_write);
    return writable.seal();
}

test "a read before any write is refused, naming the kernel and the buffer" {
    const before = liveCount();
    const sealed = try testSealed("rope_cos", 0x1000);
    defer forget(sealed.pointer);

    try std.testing.expect(sealed.written() == .never);
    try std.testing.expectEqual(@as(u32, before + 1), liveCount());

    const decls = [_]Decl{
        .{ .label = "rope_cos", .pointer = sealed.pointer, .access = .read },
    };
    try std.testing.expectError(
        EpochError.read_before_write,
        check("attention.apply_rope", decls[0..]),
    );
    // Still never: a refused launch records nothing.
    try std.testing.expect(sealed.written() == .never);
}

test "a write then a read is accepted, and the read sees the write" {
    const before = liveCount();
    const sealed = try testSealed("rope_cos", 0x2000);
    defer forget(sealed.pointer);

    const producing = [_]Decl{
        .{ .label = "rope_cos", .pointer = sealed.pointer, .access = .write },
    };
    try check("attention.build_rope_tables", producing[0..]);
    try std.testing.expect(sealed.written() == .written);

    const consuming = [_]Decl{
        .{ .label = "rope_cos", .pointer = sealed.pointer, .access = .read },
    };
    try check("attention.apply_rope", consuming[0..]);
    try std.testing.expectEqual(@as(u32, before + 1), liveCount());
    try std.testing.expectEqual(@as(?[]const u8, "rope_cos"), labelOf(sealed.pointer));
    forget(sealed.pointer);
    try std.testing.expectEqual(before, liveCount());
}

test "read_write needs a prior write, because a read is still a read" {
    const sealed = try testSealed("attn_key", 0x3000);
    defer forget(sealed.pointer);

    const decls = [_]Decl{
        .{ .label = "attn_key", .pointer = sealed.pointer, .access = .read_write },
    };
    try std.testing.expectError(
        EpochError.read_before_write,
        check("attention.apply_rope", decls[0..]),
    );
}

test "a fill does not satisfy a later read: the reset bug, closed" {
    const sealed = try testSealed("rope_cos", 0x4000);
    defer forget(sealed.pointer);

    const producing = [_]Decl{
        .{ .label = "rope_cos", .pointer = sealed.pointer, .access = .write },
    };
    try check("attention.build_rope_tables", producing[0..]);
    try std.testing.expect(sealed.written() == .written);

    // `State.reset`'s blanket zero loop: a fill through the writable view.
    markEmpty(sealed.pointer);
    try std.testing.expect(sealed.written() == .never);

    const consuming = [_]Decl{
        .{ .label = "rope_cos", .pointer = sealed.pointer, .access = .read },
    };
    try std.testing.expectError(
        EpochError.read_before_write,
        check("attention.apply_rope", consuming[0..]),
    );

    // And the correct code — reset, then rebuild the tables — passes.
    try check("attention.build_rope_tables", producing[0..]);
    try check("attention.apply_rope", consuming[0..]);
}

test "a declared buffer the registry does not know is an error, not a pass" {
    const decls = [_]Decl{
        .{ .label = "mystery", .pointer = 0x5000, .access = .read },
    };
    try std.testing.expectError(
        EpochError.epoch_unknown_buffer,
        check("some.kernel", decls[0..]),
    );
}

test "an initially_zero buffer may be read with no write at all" {
    const buffer = cuda_context.DeviceBuffer{ .pointer = 0x6000, .bytes = 4096 };
    const writable = try open("moe_pad", buffer, .initially_zero);
    defer forget(writable.pointer);
    const sealed = writable.seal();

    const decls = [_]Decl{
        .{ .label = "moe_pad", .pointer = sealed.pointer, .access = .read },
    };
    try check("moe.grouped_gemm", decls[0..]);
    try std.testing.expect(sealed.written() == .never);
}

test "tracking is compiled in for tests, and the registry survives a reset" {
    // The newer bug in this repository was a guard that could never fire. If
    // `tracking` were false here, every check above would pass vacuously; the
    // suite asserts the other direction.
    try std.testing.expect(tracking);
    const before = liveCount();
    const sealed = try testSealed("index_scores", 0x7000);
    try std.testing.expectEqual(before + 1, liveCount());
    markEmpty(sealed.pointer);
    try std.testing.expect(sealed.written() == .never);
    forget(sealed.pointer);
    try std.testing.expectEqual(before, liveCount());
}
