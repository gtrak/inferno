//! A restorable copy of one sequence's recurrent state.
//!
//! The Gated DeltaNet state is the expensive half of rollback: 36 linear
//! layers x 48 value heads x 128 x 128 f32 is about 113 MB per sequence in
//! production. Speculation will take this snapshot before a draft round and
//! restore it after a rejection, so the copy must stay on the device (a host
//! round trip would be four milliseconds of PCIe per direction) and it must
//! belong to the sequence, not to `State`: `State.reset` starts a *new*
//! sequence and must not free or invalidate a snapshot the caller still owns.
//!
//! The KV and indexer caches are deliberately **not** copied. They are
//! addressed by absolute position: a restore only has to rewind the logical
//! length (`State.length`) and every later append overwrites the rows a
//! diverged draft wrote. Copying them would be the wrong ownership — those
//! rows are the arena's, and their meaning is the position the snapshot
//! restores. The GDN conv ring and the PLE conv ring *are* copied: they shift
//! in place, so position alone cannot reconstruct them.
//!
//! Ownership: `Snapshot` owns one shadow device buffer per captured live
//! buffer and a host copy of the PLE n-gram stream. It borrows the live
//! buffers' addresses from the model; the model must outlive the snapshot and
//! must not be re-initialised in place. `deinit` frees only the shadows.

const std = @import("std");

const cuda_context = @import("../hw/api.zig").cuda_context;
const epochs = @import("../hw/epochs.zig");
const generate = @import("generate.zig");
const guards = @import("../hw/buffer.zig");
const ple = @import("../kernels/ple.zig");
const state_module = @import("state.zig");

const assert = std.debug.assert;

/// One live cache buffer and the shadow that holds its captured bytes.
///
/// The shadow is `Sealed`, not a raw `DeviceBuffer`: it goes through
/// `guards.allocate` so the guard band and write epoch cover the snapshot too.
/// A restore of a shadow no capture has written fails the epoch check instead
/// of copying whatever the allocation happened to hold.
const Entry = struct {
    live: cuda_context.DeviceBuffer,
    shadow: epochs.Sealed,
};

/// The per-sequence state a speculation round must be able to rewind.
pub const Snapshot = struct {
    allocator: std.mem.Allocator,
    entries: []Entry,
    /// The PLE n-gram history, which is host state with no device buffer.
    id_stream: ple.IdStream,
    /// `State.length` at capture: the KV/indexer append position.
    length: u32,
    /// `State.pooled_blocks` at capture.
    pooled_blocks: u32,
    /// The sequence's own position, so a restore rewinds more than the model.
    generated: u32,
    cached: u32,
    pending: i64,
    /// The sequence epoch at capture; a restore across a `begin` is refused.
    epoch: u64,
    captured: bool,

    /// Allocates the shadows for `sequence`'s model. No allocation after this.
    pub fn init(
        allocator: std.mem.Allocator,
        sequence: *const generate.Sequence,
    ) !Snapshot {
        assert(allocator.vtable != undefined);
        assert(sequence.model.state.layer_count > 0);
        assert(sequence.model.plan.layer_count > 0);

        const count = entryCount(sequence);
        const entries = try allocator.alloc(Entry, count);
        errdefer allocator.free(entries);
        var filled: usize = 0;
        errdefer {
            for (entries[0..filled]) |entry| {
                freeShadow(entry.shadow);
            }
        }
        const stream = &sequence.model.ple.stream;
        try appendEntry(
            entries,
            &filled,
            sequence.model.ple_block.buffers.conv_state,
            stream,
        );
        for (0..sequence.model.plan.layer_count) |index| {
            if (sequence.model.state.layers[index].gdn) |*cache| {
                try appendEntry(entries, &filled, cache.conv, stream);
                try appendEntry(entries, &filled, cache.recurrent, stream);
            } else {
                assert(sequence.model.state.layers[index].attention != null);
            }
        }
        if (filled != entries.len) {
            return error.SnapshotLayoutMismatch;
        }
        return .{
            .allocator = allocator,
            .entries = entries,
            .id_stream = undefined,
            .length = 0,
            .pooled_blocks = 0,
            .generated = 0,
            .cached = 0,
            .pending = 0,
            .epoch = 0,
            .captured = false,
        };
    }

    /// Frees the shadows; the live buffers belong to the model.
    pub fn deinit(self: *Snapshot) void {
        assert(self.allocator.vtable != undefined);
        assert(self.entries.len > 0);

        const entries = self.entries;
        self.entries = entries[0..0];
        for (entries) |entry| {
            freeShadow(entry.shadow);
        }
        self.allocator.free(entries);
    }

    /// Copies the sequence's recurrent state into the shadows.
    pub fn capture(self: *Snapshot, sequence: *generate.Sequence) !void {
        assert(self.entries.len > 0);
        assert(sequence.model.plan.layer_count > 0);
        assert(sequence.cached == sequence.model.state.length);
        assert(sequence.epoch > 0);
        try self.verify(sequence);

        const stream = &sequence.model.ple.stream;
        for (self.entries) |entry| {
            assert(entry.shadow.size() >= entry.live.bytes);
            const shadow = entry.shadow.device();
            try stream.copy_d_to_d(&shadow, &entry.live, entry.live.bytes);
        }
        try stream.synchronize();
        for (self.entries) |entry| {
            epochs.markWritten(entry.shadow.device().pointer);
        }
        self.id_stream = sequence.model.id_stream;
        self.length = sequence.model.state.length;
        self.pooled_blocks = sequence.model.state.pooled_blocks;
        self.generated = sequence.generated;
        self.cached = sequence.cached;
        self.pending = sequence.pending;
        self.epoch = sequence.epoch;
        self.captured = true;
        assert(self.captured);
    }

    /// Copies the shadows back and rewinds the sequence's logical position.
    ///
    /// The rows a diverged run wrote past `length` stay in the KV/indexer
    /// arenas; they cannot be read while the length is restored, and the
    /// ordinary append path overwrites them when the continuation reaches
    /// them. That is the position-as-meaning contract this restore relies on.
    pub fn restore(self: *Snapshot, sequence: *generate.Sequence) !void {
        assert(self.captured);
        assert(self.entries.len > 0);
        if (sequence.epoch != self.epoch) {
            return error.SnapshotSequenceMismatch;
        }
        try self.verify(sequence);

        const stream = &sequence.model.ple.stream;
        for (self.entries) |entry| {
            assert(entry.live.bytes <= entry.shadow.size());
            assert(entry.shadow.written() == .written);
            const shadow = entry.shadow.device();
            try stream.copy_d_to_d(&entry.live, &shadow, entry.live.bytes);
        }
        try stream.synchronize();
        sequence.model.id_stream = self.id_stream;
        sequence.model.state.length = self.length;
        sequence.model.state.pooled_blocks = self.pooled_blocks;
        sequence.generated = self.generated;
        sequence.cached = self.cached;
        sequence.pending = self.pending;
        assert(sequence.model.state.length == self.length);
        assert(sequence.cached == sequence.model.state.length);
    }

    /// Rejects a snapshot taken from another model or after a re-layout.
    fn verify(
        self: *const Snapshot,
        sequence: *const generate.Sequence,
    ) !void {
        assert(self.entries.len > 0);
        assert(sequence.model.plan.layer_count > 0);

        if (self.entries.len != entryCount(sequence)) {
            return error.SnapshotLayoutMismatch;
        }
        var index: usize = 0;
        if (self.entries[index].live.pointer !=
            sequence.model.ple_block.buffers.conv_state.pointer)
        {
            return error.SnapshotLayoutMismatch;
        }
        index += 1;
        for (0..sequence.model.plan.layer_count) |layer| {
            if (sequence.model.state.layers[layer].gdn) |*cache| {
                if (self.entries[index].live.pointer != cache.conv.pointer) {
                    return error.SnapshotLayoutMismatch;
                }
                index += 1;
                if (self.entries[index].live.pointer != cache.recurrent.pointer) {
                    return error.SnapshotLayoutMismatch;
                }
                index += 1;
            }
        }
        if (index != self.entries.len) {
            return error.SnapshotLayoutMismatch;
        }
        assert(index == self.entries.len);
    }
};

/// Buffers one snapshot shadows: the PLE conv ring plus each GDN pair.
fn entryCount(sequence: *const generate.Sequence) usize {
    assert(sequence.model.plan.layer_count > 0);
    assert(sequence.model.state.layer_count == sequence.model.plan.layer_count);

    var count: usize = 1;
    for (0..sequence.model.plan.layer_count) |index| {
        if (sequence.model.state.layers[index].gdn != null) {
            count += 2;
        }
    }
    assert(count > 1);
    return count;
}

/// Allocates one shadow for `live` and records the pair.
fn appendEntry(
    entries: []Entry,
    filled: *usize,
    live: cuda_context.DeviceBuffer,
    stream: *const cuda_context.Stream,
) !void {
    assert(filled.* < entries.len);
    assert(live.bytes > 0);

    const writable = try guards.allocate(
        "snapshot.shadow",
        state_module.allocationBytes(live.bytes),
        stream,
        .must_write,
    );
    entries[filled.*] = .{ .live = live, .shadow = writable.seal() };
    filled.* += 1;
    assert(filled.* <= entries.len);
}

/// Releases a shadow allocated through `guards`.
fn freeShadow(shadow: epochs.Sealed) void {
    assert(shadow.size() > 0);
    assert(shadow.pointer != 0);

    const payload = shadow.device();
    guards.free(&payload);
}
