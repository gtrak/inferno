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

    pub fn allocator(self: *Budget) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    pub fn deinit(self: *Budget) void {
        // No postcondition here: exceeding the budget is already reported by `alloc`, and a
        // test that deliberately overruns must be able to pass by observing it.
        std.debug.assert(self.count > 0 or self.limit > 0);
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
            std.debug.print(
                "budget exceeded: {d} allocations (limit {d}); this path allocates per " ++
                    "element - pass an arena or a caller-owned scratch buffer\n",
                .{ self.count, self.limit },
            );
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
