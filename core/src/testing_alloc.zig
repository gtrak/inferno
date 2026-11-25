//! Allocator policy for tests that touch bulk data (I-035).
//!
//! `std.testing.allocator` maps every allocation so it can report leaks with provenance.
//! That is the right default. It is the wrong allocator for a path that allocates once per
//! tensor: over the real checkpoint (299,545 tensors) it turned a **478 ms** manifest build
//! into a run that did not finish in ten minutes. The suite then looks hung rather than slow.
//!
//! Two instruments, in the order to reach for them:
//!
//!   var a = testing_alloc.arena();      // bulk data: one chunk allocator over the tracker
//!   defer a.deinit();                   //   leak detection survives at chunk granularity
//!
//!   var b = testing_alloc.budget(1_000); // any path: fail if it allocates more than N times
//!   defer b.deinit();                    //   turns a perf cliff into an assertion

const std = @import("std");

/// Environment variable that turns the overrun diary back on.
///
/// Off by default because of how the test runner reports a failure: Zig 0.14's
/// `std.Build.Step.Run` attaches *everything the test process wrote to stderr
/// since the previous test result* to the next test that fails. A passing test
/// that prints therefore renames a later failure's message. Measured: the
/// deliberate-overrun test below printed five lines, and the shard test that
/// genuinely failed with `expected 11, found 4` was reported as
/// `failed: budget exceeded: 5 allocations ...`. The failing assertion and the
/// reported reason came from different tests.
const diary_variable = "INFERNO_ALLOC_DEBUG";

/// An arena over the tracked allocator. Bulk paths allocate freely inside it; the tracker
/// still sees the chunks, so a leaked arena still fails the test.
pub fn arena() std.heap.ArenaAllocator {
    return std.heap.ArenaAllocator.init(std.testing.allocator);
}

/// Wraps the tracked allocator and fails the test if it is asked for more than `limit`
/// allocations. Use this to assert that a data path allocates O(1) times, not O(n).
pub fn budget(limit: usize) Budget {
    assertLimit(limit);
    return .{ .child = std.testing.allocator, .limit = limit };
}

fn assertLimit(limit: usize) void {
    std.debug.assert(limit > 0);
}

pub const Budget = struct {
    child: std.mem.Allocator,
    limit: usize,
    count: usize = 0,
    overruns: usize = 0,

    pub fn allocator(self: *Budget) std.mem.Allocator {
        assertLimit(self.limit);
        return .{ .ptr = self, .vtable = &vtable };
    }

    pub fn deinit(self: *Budget) void {
        // No postcondition here: exceeding the budget is already reported through
        // `exceeded`, and a test that deliberately overruns must be able to pass by
        // observing it.
        std.debug.assert(self.limit > 0);
    }

    pub fn exceeded(self: *const Budget) bool {
        return self.count > self.limit;
    }

    const vtable = std.mem.Allocator.VTable{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };

    fn alloc(context: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *Budget = @ptrCast(@alignCast(context));
        self.count += 1;
        if (self.count > self.limit) {
            self.overruns += 1;
            reportOverrun(self.count, self.limit);
            // Null here is `error.OutOfMemory` at the caller, so the test that ran
            // out of budget still fails on its own name with its own stack trace.
            return null;
        }
        return self.child.vtable.alloc(self.child.ptr, len, alignment, ra);
    }

    fn resize(
        context: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        ra: usize,
    ) bool {
        const self: *Budget = @ptrCast(@alignCast(context));
        return self.child.vtable.resize(self.child.ptr, memory, alignment, new_len, ra);
    }

    fn remap(
        context: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        ra: usize,
    ) ?[*]u8 {
        const self: *Budget = @ptrCast(@alignCast(context));
        return self.child.vtable.remap(self.child.ptr, memory, alignment, new_len, ra);
    }

    fn free(context: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *Budget = @ptrCast(@alignCast(context));
        self.child.vtable.free(self.child.ptr, memory, alignment, ra);
    }
};

/// Silent unless `INFERNO_ALLOC_DEBUG` is set in the environment. The message is
/// the diagnosis a lane hunting a per-element allocation wants; the default stream
/// is kept clean so a failure is reported against the test that failed.
fn reportOverrun(count: usize, limit: usize) void {
    assertLimit(limit);
    std.debug.assert(count > limit);

    if (std.posix.getenv(diary_variable) == null) {
        return;
    } else {
        std.debug.print(
            "budget overrun: {d} allocations (limit {d}); this path allocates per " ++
                "element - pass an arena or a caller-owned scratch buffer\n",
            .{ count, limit },
        );
    }
}

test "the budget fails a path that allocates per element" {
    var b = budget(4);
    defer b.deinit();
    const a = b.allocator();
    const first = try a.alloc(u8, 8);
    a.free(first);
    var overrun = false;
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        if (a.alloc(u8, 8) catch null) |slice| {
            a.free(slice);
        } else {
            overrun = true;
        }
    }
    try std.testing.expect(overrun);
    try std.testing.expect(b.exceeded());
    // Every refusal is counted, so a caller can assert on the shape of the overrun
    // rather than reading it off a print.
    try std.testing.expectEqual(@as(usize, 5), b.overruns);
    try std.testing.expectEqual(@as(usize, 9), b.count);
}

test "the arena allocates chunks, not per call" {
    var counted = budget(8);
    defer counted.deinit();
    var arena_over_counted = std.heap.ArenaAllocator.init(counted.allocator());
    defer arena_over_counted.deinit();
    const a = arena_over_counted.allocator();
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        _ = try a.alloc(u8, 64);
    }
    try std.testing.expect(counted.count <= counted.limit);
}
