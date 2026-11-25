//! Model and sequence handles for the C ABI (issue I-036).
//!
//! Rust decides the *policy*: which model is live, when a sequence may step,
//! when a handle stops being usable. This file is the *mechanism* — two
//! fixed-capacity tables with a generation counter, no allocation outside
//! `openModel`, and every entry point answers with a code, because a panic on
//! this side aborts the Rust process that called it.
//!
//! A caller-visible handle is a C pointer to one `Cell` of a static pool. The
//! cell names the slot it was issued for and the generation it was issued in.
//! An entry point honours a handle only when the cell lies inside the right
//! pool, is the slot's newest issuance, and still carries the slot's
//! generation. That gives three refusals the plain "index into an array" handle
//! cannot:
//!
//! * a closed handle is refused with a code and never loaded;
//! * a handle whose slot has been reissued is refused instead of quietly
//!   driving the sequence that took the slot (the ABA bug);
//! * closing a model orphans its sequences, so `step` can answer
//!   `model_closed` instead of reading a released CUDA context.
//!
//! A forged pointer is rejected by an integer range test before anything is
//! loaded through it. The one hazard a pointer handle cannot remove is that a
//! pool cell is eventually reused; that needs `sequence_cells_max` further
//! openings of the table while the caller keeps the stale handle, and it is
//! documented rather than pretended away.
//!
//! The tables are static so the model built in one has the address its interior
//! pointers were bound to (I-032), and so the step path allocates nothing.
//!
//! A step or a reset that *fails* inside the graph invalidates the sequence: its
//! own bookkeeping is no longer trustworthy (a retry would replay the previous
//! id), so the handle answers `stale_handle` from then on and the caller opens a
//! new one. That is the same rule as closing, applied to a failure.

const std = @import("std");

const exec = @import("exec.zig");
const generate = @import("generate.zig");
/// The committed tiny fixture, read by the tests below and by nothing else.
const tiny_check = @import("tiny_check.zig");

const assert = std.debug.assert;

/// Models open at once.
pub const models_max: u32 = 2;

/// Sequences open at once, summed over every model.
pub const sequences_max: u32 = 4;

/// Prompt ids a sequence may be opened with. The tiny arena holds 512.
pub const prompt_ids_max: u32 = 512;

/// Sequence handles issued before the pool overwrites a cell.
pub const sequence_cells_max: u32 = 128;

/// Model handles issued before the pool overwrites a cell.
pub const model_cells_max: u32 = 64;

/// Floats `stepSequence` writes into `out_stats`.
pub const stats_len: u32 = 3;

/// Largest manifest document `openModel` reads.
pub const manifest_bytes_max: u32 = 4096;

/// A slot with no current issuance.
const no_cell: u32 = std.math.maxInt(u32);

/// The only model this build can run: the committed tiny fixtures. The real
/// checkpoint needs a forward that does not exist yet (I-017c).
pub const tiny_manifest = "{\"model\":\"tiny\"}";

/// Result codes of the handle ABI.
///
/// `server/src/engine/handles.rs` mirrors these numbers, and a test on each side
/// pins them, so the two declarations cannot drift silently.
pub const Code = enum(i32) {
    ok = 0,
    /// A null argument, an out-of-range id, or a zero-length document.
    invalid_argument = -1,
    /// Every slot of the table is taken.
    out_of_handles = -2,
    /// The handle names no live object: closed, or its slot was reissued.
    stale_handle = -3,
    /// The model a sequence belongs to has been closed.
    model_closed = -4,
    /// The model could not be opened: no device, or a failed load.
    model_unavailable = -5,
    /// The manifest names a model this build cannot run.
    unsupported_model = -6,
    /// The prompt was empty.
    empty_prompt = -7,
    /// The prompt does not fit one prefill.
    prompt_too_long = -8,
    /// The sequence reached the end of the model's arena.
    context_full = -9,
    /// The forward failed for a reason without a code of its own.
    step_failed = -10,
};

/// Failures the handle entry points raise. The C ABI converts each to a `Code`;
/// nothing here escapes as a panic.
pub const HandleError = error{
    InvalidArgument,
    OutOfHandles,
    StaleHandle,
    ModelClosed,
    ModelUnavailable,
    UnsupportedModel,
    EmptyPrompt,
    PromptTooLong,
    ContextFull,
    StepFailed,
};

/// Filled by `readCounts`: what the tables hold, and what they have issued.
///
/// The Rust side reads this to prove a dropped sequence left nothing behind;
/// `leaked_models` is the model's own allocator reporting, at close, that
/// something it handed out was never returned.
pub const Counts = extern struct {
    models_live: u32,
    models_opened: u32,
    sequences_live: u32,
    sequences_opened: u32,
    sequences_orphaned: u32,
    leaked_models: u32,
};

/// One issued handle. Its address is the C pointer; `slot` and `generation` are
/// what make a stale handle detectable without dereferencing anything else.
pub const Cell = extern struct {
    slot: u32,
    generation: u32,
};

/// What a sequence slot holds.
pub const Status = enum(u8) {
    /// No sequence; the slot is the next one `openSequence` takes.
    free,
    /// A live sequence over a live model.
    live,
    /// A sequence whose model was closed; `step` answers `model_closed` and the
    /// next `openSequence` reclaims the slot.
    orphaned,
};

/// One model slot. `model` is built where it lives, never moved.
const ModelSlot = struct {
    model: exec.Model = undefined,
    /// The model's owner. A debug allocator, so closing the model reports an
    /// allocation the graph returned nothing for — the leak half of I-036.
    allocator: std.heap.DebugAllocator(.{}) = .{},
    generation: u32 = 0,
    cell: u32 = no_cell,
    live: bool = false,
};

/// One sequence slot, with the prompt it replays on `reset`.
const SequenceSlot = struct {
    sequence: generate.Sequence = undefined,
    prompt: [prompt_ids_max]i64 = [_]i64{0} ** prompt_ids_max,
    prompt_length: u32 = 0,
    model_slot: u32 = 0,
    model_generation: u32 = 0,
    generation: u32 = 0,
    cell: u32 = no_cell,
    status: Status = .free,
};

/// The handle tables.
pub const Table = struct {
    models: [models_max]ModelSlot = [_]ModelSlot{.{}} ** models_max,
    sequences: [sequences_max]SequenceSlot =
        [_]SequenceSlot{.{}} ** sequences_max,
    model_cells: [model_cells_max]Cell =
        [_]Cell{.{ .slot = 0, .generation = 0 }} ** model_cells_max,
    sequence_cells: [sequence_cells_max]Cell =
        [_]Cell{.{ .slot = 0, .generation = 0 }} ** sequence_cells_max,
    model_cell_next: u32 = 0,
    sequence_cell_next: u32 = 0,
    models_opened: u32 = 0,
    sequences_opened: u32 = 0,
    leaked_models: u32 = 0,
};

/// The one table; static, so every address in it outlives the process.
var table: Table = .{};

/// The table, for the entry points and for `collapse` in a test.
pub fn instance() *Table {
    assert(models_max > 0);
    assert(@sizeOf(Table) > 0);
    return &table;
}

/// The C code for `err`. Total by construction: an error without a code of its
/// own becomes `step_failed`, so a failure can never be reported as success.
pub fn codeFor(err: anyerror) i32 {
    assert(err != error.NoError);
    assert(@intFromEnum(Code.step_failed) < 0);

    return switch (err) {
        error.InvalidArgument => @intFromEnum(Code.invalid_argument),
        error.OutOfHandles => @intFromEnum(Code.out_of_handles),
        error.StaleHandle => @intFromEnum(Code.stale_handle),
        error.ModelClosed => @intFromEnum(Code.model_closed),
        error.ModelUnavailable => @intFromEnum(Code.model_unavailable),
        error.UnsupportedModel => @intFromEnum(Code.unsupported_model),
        error.EmptyPrompt => @intFromEnum(Code.empty_prompt),
        error.PromptTooLong => @intFromEnum(Code.prompt_too_long),
        error.ContextFull => @intFromEnum(Code.context_full),
        else => @intFromEnum(Code.step_failed),
    };
}

/// The handle error for a failure inside the model graph. `ContextFull` is the
/// only forward error with a meaning of its own here.
fn forwardError(err: anyerror) HandleError {
    assert(err != error.NoError);
    assert(exec.tokens_max > 0);

    return switch (err) {
        error.ContextFull => error.ContextFull,
        else => error.StepFailed,
    };
}

/// The pool index `pointer` names, or null when it is not a cell boundary of
/// `cells`. The test is on the integer address, so a forged handle is refused
/// before the table loads it.
fn cellIndex(cells: []const Cell, pointer: *const Cell) ?u32 {
    assert(cells.len > 0);
    assert(cells.len <= std.math.maxInt(u32));

    const base = @intFromPtr(&cells[0]);
    const address = @intFromPtr(pointer);
    const width = @sizeOf(Cell);
    if (address < base) {
        return null;
    } else {
        assert(address >= base);
    }
    const offset = address - base;
    if (offset >= width * cells.len) {
        return null;
    } else if (offset % width != 0) {
        return null;
    } else {
        return @intCast(offset / width);
    }
}

/// Issues the next cell of `cells` for `(slot, generation)`.
///
/// The pool is a ring. A cell is only overwritten after `cells.len` further
/// issuances, and a reissued cell carries a new generation, so the two checks
/// in `resolveSequence` refuse every stale handle until the ring comes round to
/// the same slot with the same generation counter.
fn issueCell(cells: []Cell, next: *u32, slot: u32, generation: u32) *Cell {
    assert(cells.len > 0);
    assert(next.* < cells.len);

    const count: u32 = @intCast(cells.len);
    const index = next.*;
    cells[index] = .{ .slot = slot, .generation = generation };
    next.* = (index + 1) % count;
    assert(next.* < count);
    return &cells[index];
}

/// The index of the model `handle` names, or `stale_handle`.
fn resolveModel(t: *Table, handle: *const Cell) HandleError!u32 {
    assert(models_max > 0);
    assert(model_cells_max > 0);

    const index = cellIndex(t.model_cells[0..], handle) orelse
        return error.StaleHandle;
    const cell = &t.model_cells[index];
    const slot_index = cell.slot;
    if (slot_index >= models_max) {
        return error.StaleHandle;
    } else {
        assert(slot_index < models_max);
    }
    const slot = &t.models[slot_index];
    if (slot.cell != index) {
        return error.StaleHandle;
    } else if (slot.generation != cell.generation) {
        return error.StaleHandle;
    } else if (!slot.live) {
        return error.StaleHandle;
    } else {
        assert(slot.live);
    }
    return slot_index;
}

/// The index of the sequence `handle` names, or `stale_handle`. An orphaned slot
/// still resolves: its handle is valid, its model is not.
fn resolveSequence(t: *Table, handle: *const Cell) HandleError!u32 {
    assert(sequences_max > 0);
    assert(sequence_cells_max > 0);

    const index = cellIndex(t.sequence_cells[0..], handle) orelse
        return error.StaleHandle;
    const cell = &t.sequence_cells[index];
    const slot_index = cell.slot;
    if (slot_index >= sequences_max) {
        return error.StaleHandle;
    } else {
        assert(slot_index < sequences_max);
    }
    const slot = &t.sequences[slot_index];
    if (slot.cell != index) {
        return error.StaleHandle;
    } else if (slot.generation != cell.generation) {
        return error.StaleHandle;
    } else if (slot.status == .free) {
        return error.StaleHandle;
    } else {
        assert(slot.status != .free);
    }
    return slot_index;
}

/// The index of the sequence `handle` names, refusing one whose model is gone.
///
/// The model checks come before any field of the model is read, which is what
/// makes "model closed with a live sequence" a code instead of a use-after-free.
fn requireLiveSequence(t: *Table, handle: *const Cell) HandleError!u32 {
    assert(sequences_max > 0);
    assert(models_max > 0);

    const index = try resolveSequence(t, handle);
    const slot = &t.sequences[index];
    switch (slot.status) {
        .free => return error.StaleHandle,
        .orphaned => return error.ModelClosed,
        .live => {
            const model = &t.models[slot.model_slot];
            if (model.generation != slot.model_generation) {
                return error.ModelClosed;
            } else if (!model.live) {
                return error.ModelClosed;
            } else {
                assert(model.live);
            }
        },
    }
    return index;
}

/// The lowest free or orphaned sequence slot, or null when the table is full.
fn freeSequenceSlot(t: *Table) ?u32 {
    assert(sequences_max > 0);
    assert(t.sequences.len == sequences_max);

    for (&t.sequences, 0..) |*slot, index| {
        switch (slot.status) {
            .free, .orphaned => return @intCast(index),
            .live => assert(slot.status == .live),
        }
    }
    return null;
}

/// The lowest free model slot, or null when the table is full.
fn freeModelSlot(t: *Table) ?u32 {
    assert(models_max > 0);
    assert(t.models.len == models_max);

    for (&t.models, 0..) |*slot, index| {
        if (!slot.live) {
            return @intCast(index);
        } else {
            assert(slot.live);
        }
    }
    return null;
}

/// Requires `json` to be the manifest of a model this build can run.
///
/// The document is parsed, not pattern-matched: a caller that asks for the real
/// checkpoint must be refused, not handed the tiny model. Anything that does not
/// parse, or does not carry a `model` string, is `invalid_argument`; a `model`
/// this build does not implement is `unsupported_model`.
fn requireTinyManifest(allocator: std.mem.Allocator, json: []const u8) HandleError!void {
    assert(json.len > 0);
    assert(allocator.vtable != undefined);

    var document = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{},
    ) catch return error.InvalidArgument;
    defer document.deinit();

    if (document.value != .object) {
        return error.InvalidArgument;
    } else {
        assert(document.value == .object);
    }
    const named = document.value.object.get("model") orelse
        return error.InvalidArgument;
    if (named != .string) {
        return error.InvalidArgument;
    } else if (!std.mem.eql(u8, named.string, "tiny")) {
        return error.UnsupportedModel;
    } else {
        assert(named.string.len > 0);
    }
}

/// Releases what a model slot holds and returns it to free.
///
/// The sequences bound to the model are orphaned first, so their handles keep
/// resolving and answer `model_closed` instead of touching a released context.
fn releaseModel(t: *Table, index: u32) void {
    assert(index < models_max);
    assert(t.models[index].live);

    const slot = &t.models[index];
    for (&t.sequences, 0..) |*sequence, sequence_index| {
        assert(sequence_index < sequences_max);
        if (sequence.status == .live and
            sequence.model_slot == index and
            sequence.model_generation == slot.generation)
        {
            sequence.status = .orphaned;
        } else {
            assert(sequence.status != .live or sequence.model_slot != index or
                sequence.model_generation != slot.generation);
        }
    }
    slot.model.deinit();
    const check = slot.allocator.deinit();
    slot.allocator = .{};
    slot.live = false;
    slot.generation +%= 1;
    if (check == .leak) {
        t.leaked_models +%= 1;
        std.log.err(
            "inferno_model_close: slot {d} leaked an allocation; " ++
                "the allocator's report above names it",
            .{index},
        );
    } else {
        assert(check == .ok);
    }
}

/// Frees sequence slot `index` and invalidates its handle.
fn releaseSequence(t: *Table, index: u32) void {
    assert(index < sequences_max);
    assert(t.sequences[index].status != .free);

    const slot = &t.sequences[index];
    slot.status = .free;
    slot.prompt_length = 0;
    // The cell is left in place: the handle still resolves to this slot and is
    // then refused by the generation it no longer carries.
    slot.generation +%= 1;
    assert(slot.status == .free);
}

/// Frees the allocator of a model slot that never became live.
fn abandonModel(t: *Table, slot: *ModelSlot) void {
    assert(!slot.live);
    assert(t.models.len == models_max);

    const check = slot.allocator.deinit();
    slot.allocator = .{};
    if (check == .leak) {
        t.leaked_models +%= 1;
    } else {
        assert(check == .ok);
    }
}

/// Opens the model `json` describes and returns its handle.
///
/// The manifest is a JSON object with a `model` key; `"tiny"` is the committed
/// fixture and the only forward this build has (I-017c adds the checkpoint).
/// The ABI's signature has no error out-parameter, so the reason a caller gets
/// null is logged here — and the caller is Rust, which logs the same fact.
pub fn openModel(json: ?[*]const u8, length: u32) HandleError!*Cell {
    assert(models_max > 0);
    assert(model_cells_max > 0);

    if (json == null) {
        return error.InvalidArgument;
    } else if (length == 0) {
        return error.InvalidArgument;
    } else if (length > manifest_bytes_max) {
        return error.InvalidArgument;
    } else {
        assert(json != null);
    }
    const t = instance();
    const index = if (freeModelSlot(t)) |found| found else return error.OutOfHandles;
    const slot = &t.models[index];
    assert(!slot.live);
    slot.allocator = .{};
    const allocator = slot.allocator.allocator();
    requireTinyManifest(allocator, json.?[0..length]) catch |err| {
        abandonModel(t, slot);
        return err;
    };
    slot.model.init(allocator) catch |err| {
        std.log.err("inferno_model_open: {s}", .{@errorName(err)});
        abandonModel(t, slot);
        return error.ModelUnavailable;
    };
    slot.generation +%= 1;
    slot.live = true;
    const cell = issueCell(
        t.model_cells[0..],
        &t.model_cell_next,
        index,
        slot.generation,
    );
    slot.cell = cellIndex(t.model_cells[0..], cell) orelse no_cell;
    t.models_opened +%= 1;
    assert(slot.live);
    return cell;
}

/// Closes the model `handle` names. Idempotent: a stale handle frees nothing.
pub fn closeModel(handle: ?*const Cell) void {
    assert(models_max > 0);
    assert(model_cells_max > 0);

    if (handle) |cell| {
        const t = instance();
        if (resolveModel(t, cell)) |index| {
            releaseModel(t, index);
        } else |_| {
            // A second close, or a handle whose slot was reissued: nothing to do.
        }
    } else {
        assert(handle == null);
    }
}

/// Opens a sequence over `model` and prefills `ids`.
///
/// `count` is bounded by `exec.tokens_max` because this prefill is one forward:
/// the chunked prefill that would lift it belongs to the runtime work, and a
/// longer prompt would trip the executor's own bound instead of returning a code.
pub fn openSequence(
    model: ?*const Cell,
    ids: ?[*]const u32,
    count: u32,
) HandleError!*Cell {
    assert(sequences_max > 0);
    assert(prompt_ids_max > 0);

    const model_handle = model orelse return error.InvalidArgument;
    const t = instance();
    const model_index = try resolveModel(t, model_handle);
    const model_slot = &t.models[model_index];
    const arena = model_slot.model.state.arena_length;
    if (count == 0) {
        return error.EmptyPrompt;
    } else if (count > prompt_ids_max) {
        return error.PromptTooLong;
    } else if (count > exec.tokens_max) {
        return error.PromptTooLong;
    } else if (count > arena) {
        return error.PromptTooLong;
    } else {
        assert(count <= prompt_ids_max);
    }
    const id_slice = ids orelse return error.InvalidArgument;
    const index = if (freeSequenceSlot(t)) |found| found else return error.OutOfHandles;
    const slot = &t.sequences[index];
    if (slot.status == .orphaned) {
        // The sequence this slot held is already unreachable: its model is gone.
        slot.status = .free;
    } else {
        assert(slot.status == .free);
    }
    for (id_slice[0..count], 0..) |id, position| {
        assert(position < @as(usize, prompt_ids_max));
        if (id >= model_slot.model.config.vocab_size) {
            return error.InvalidArgument;
        } else {
            assert(id < model_slot.model.config.vocab_size);
        }
        slot.prompt[position] = @intCast(id);
    }
    slot.prompt_length = count;
    slot.model_slot = model_index;
    slot.model_generation = model_slot.generation;
    const sequence = generate.Sequence.init(&model_slot.model);
    slot.sequence = sequence;
    prefill(slot) catch |err| {
        slot.prompt_length = 0;
        return err;
    };
    slot.generation +%= 1;
    slot.status = .live;
    const cell = issueCell(
        t.sequence_cells[0..],
        &t.sequence_cell_next,
        index,
        slot.generation,
    );
    slot.cell = cellIndex(t.sequence_cells[0..], cell) orelse no_cell;
    t.sequences_opened +%= 1;
    assert(slot.status == .live);
    return cell;
}

/// Prefills the slot's prompt from a fresh cache state.
fn prefill(slot: *SequenceSlot) HandleError!void {
    assert(slot.prompt_length > 0);
    assert(slot.prompt_length <= prompt_ids_max);

    slot.sequence.begin(slot.prompt[0..slot.prompt_length]) catch |err| {
        std.log.err("inferno prefill: {s}", .{@errorName(err)});
        return forwardError(err);
    };
    assert(slot.sequence.cached == slot.prompt_length);
}

/// Closes the sequence `handle` names. Idempotent: a stale handle frees nothing.
pub fn closeSequence(handle: ?*const Cell) void {
    assert(sequences_max > 0);
    assert(sequence_cells_max > 0);

    if (handle) |cell| {
        const t = instance();
        if (resolveSequence(t, cell)) |index| {
            releaseSequence(t, index);
        } else |_| {
            // A second close, or a handle whose slot was reissued: nothing to do.
        }
    } else {
        assert(handle == null);
    }
}

/// Returns the sequence to start-of-sequence and replays its prompt.
///
/// The prompt is kept so a reset is a *replay*: the same ids come out, which is
/// the property the reset contract is checked against, and it needs no second
/// ABI call to hand the tokens back.
pub fn resetSequence(handle: ?*const Cell) HandleError!void {
    assert(sequences_max > 0);
    assert(prompt_ids_max > 0);

    const cell = handle orelse return error.InvalidArgument;
    const t = instance();
    const index = try requireLiveSequence(t, cell);
    const slot = &t.sequences[index];
    const model_slot = &t.models[slot.model_slot];
    slot.sequence = generate.Sequence.init(&model_slot.model);
    prefill(slot) catch |err| {
        // A half-replayed sequence is not left behind for the next call to trip
        // over: the handle is invalidated, and the caller reopens.
        releaseSequence(t, index);
        return err;
    };
}

/// Produces the sequence's next id into `out_token` and its counters into
/// `out_stats` (see `stats_len`).
pub fn stepSequence(
    handle: ?*const Cell,
    out_token: ?*u32,
    out_stats: ?[*]f32,
    stats_cap: u32,
) HandleError!void {
    assert(stats_len > 0);
    assert(sequences_max > 0);

    const cell = handle orelse return error.InvalidArgument;
    const token_out = out_token orelse return error.InvalidArgument;
    const t = instance();
    const index = try requireLiveSequence(t, cell);
    const slot = &t.sequences[index];
    const model_slot = &t.models[slot.model_slot];
    const before = model_slot.model.timings.total_ns;
    const id = slot.sequence.step() catch |err| {
        // The same rule as `resetSequence`: a failed step leaves the sequence's
        // bookkeeping suspect (a retry would replay the last id), so the handle
        // is invalidated instead of becoming a trap for the next call.
        releaseSequence(t, index);
        return forwardError(err);
    };
    assert(id >= 0);
    assert(id < model_slot.model.config.vocab_size);
    token_out.* = @intCast(id);
    const step_ns = model_slot.model.timings.total_ns - before;
    writeStats(out_stats, stats_cap, slot, step_ns);
}

/// Writes `generated`, `cached` and the step's wall time in microseconds. A
/// null or short `out_stats` is legal and writes the prefix it has room for.
fn writeStats(
    out_stats: ?[*]f32,
    stats_cap: u32,
    slot: *const SequenceSlot,
    step_ns: u64,
) void {
    assert(stats_len > 0);
    assert(slot.prompt_length > 0);

    const values = [stats_len]f32{
        @floatFromInt(slot.sequence.generated),
        @floatFromInt(slot.sequence.cached),
        @as(f32, @floatFromInt(step_ns)) / 1000.0,
    };
    if (out_stats) |stats| {
        const written = @min(stats_cap, stats_len);
        for (values[0..written], 0..) |value, index| {
            assert(index < @as(usize, stats_len));
            stats[index] = value;
        }
    } else {
        assert(out_stats == null);
    }
}

/// Fills `out` with the tables' counters.
pub fn readCounts(t: *Table, out: *Counts) void {
    assert(models_max > 0);
    assert(sequences_max > 0);

    out.* = .{
        .models_live = 0,
        .models_opened = t.models_opened,
        .sequences_live = 0,
        .sequences_opened = t.sequences_opened,
        .sequences_orphaned = 0,
        .leaked_models = t.leaked_models,
    };
    for (&t.models) |*slot| {
        if (slot.live) {
            out.models_live += 1;
        } else {
            assert(!slot.live);
        }
    }
    for (&t.sequences) |*slot| {
        switch (slot.status) {
            .free => assert(slot.status == .free),
            .live => out.sequences_live += 1,
            .orphaned => out.sequences_orphaned += 1,
        }
    }
}

/// Releases every live model and sequence.
///
/// The tests call it from `defer` so one failing case cannot leave a slot
/// occupied for the next; a server calls it at shutdown.
pub fn collapse(t: *Table) void {
    assert(models_max > 0);
    assert(sequences_max > 0);

    for (&t.sequences, 0..) |*slot, index| {
        assert(index < sequences_max);
        if (slot.status == .free) {
            assert(slot.status == .free);
        } else {
            releaseSequence(t, @intCast(index));
        }
    }
    for (&t.models, 0..) |*slot, index| {
        assert(index < models_max);
        if (slot.live) {
            releaseModel(t, @intCast(index));
        } else {
            assert(!slot.live);
        }
    }
}

/// The committed prompt and its sixteen greedy ids, as `u32` ids for the ABI.
fn goldenPrompt(allocator: std.mem.Allocator) ![]u32 {
    assert(allocator.vtable != undefined);
    assert(tiny_check.greedy_tokens > 0);

    var golden = try tiny_check.Golden.open(allocator);
    defer golden.close();
    const ids = try allocator.alloc(u32, golden.prompt.len);
    for (golden.prompt, ids) |id, *slot| {
        assert(id >= 0);
        slot.* = @intCast(id);
    }
    assert(ids.len > 0);
    return ids;
}

/// Steps `handle` once and returns the id it produced.
fn stepOnce(handle: *const Cell) !u32 {
    assert(@intFromPtr(handle) != 0);
    assert(stats_len > 0);

    var token: u32 = 0;
    var stats: [stats_len]f32 = [_]f32{0.0} ** stats_len;
    try stepSequence(handle, &token, stats[0..], stats_len);
    assert(token < 4 * 1024);
    assert(stats[0] >= 1.0);
    return token;
}

test "the handle ABI reproduces the golden greedy ids" {
    const t = instance();
    defer collapse(t);
    const allocator = std.testing.allocator;
    const prompt = try goldenPrompt(allocator);
    defer allocator.free(prompt);

    const model = try openModel(tiny_manifest, tiny_manifest.len);
    const sequence = try openSequence(model, prompt.ptr, @intCast(prompt.len));
    var ids: [tiny_check.greedy_tokens]u32 = [_]u32{0} ** tiny_check.greedy_tokens;
    for (&ids) |*id| {
        id.* = try stepOnce(sequence);
        std.debug.print("{d} ", .{id.*});
    }
    std.debug.print("\n", .{});
    // The architecture gate's ids, straight out of the C ABI.
    const expected = [_]u32{
        169, 252, 39, 151, 132, 239, 223, 20,
        198, 120, 47, 121, 121, 144, 96,  19,
    };
    try std.testing.expectEqualSlices(u32, expected[0..], ids[0..]);
    const counts = countsNow(t);
    try std.testing.expectEqual(@as(u32, 1), counts.models_live);
    try std.testing.expectEqual(@as(u32, 1), counts.sequences_live);
}

/// The code `result` refused with. A call that succeeded is a failed test: the
/// point of the assertion is that the refusal happened at all.
fn refusalCode(result: HandleError!void) !i32 {
    assert(stats_len > 0);
    assert(@intFromEnum(Code.stale_handle) < 0);

    result catch |err| return codeFor(err);
    return error.TestUnexpectedResult;
}

/// The counters of `t`, for the tests.
fn countsNow(t: *Table) Counts {
    assert(models_max > 0);
    assert(sequences_max > 0);

    var counts: Counts = undefined;
    readCounts(t, &counts);
    return counts;
}

test "a closed sequence refuses a step" {
    const t = instance();
    defer collapse(t);
    const allocator = std.testing.allocator;
    const prompt = try goldenPrompt(allocator);
    defer allocator.free(prompt);

    const model = try openModel(tiny_manifest, tiny_manifest.len);
    const sequence = try openSequence(model, prompt.ptr, @intCast(prompt.len));
    const first = try stepOnce(sequence);
    try std.testing.expectEqual(@as(u32, 169), first);

    closeSequence(sequence);
    var token: u32 = 0xdead_beef;
    var stats: [stats_len]f32 = [_]f32{-1.0} ** stats_len;
    const refused = stepSequence(sequence, &token, stats[0..], stats_len);
    try std.testing.expectError(error.StaleHandle, refused);
    // Refused means refused: the out-parameters were not touched, and the code
    // is the documented one rather than a crash.
    try std.testing.expectEqual(@as(u32, 0xdead_beef), token);
    try std.testing.expectEqual(@as(f32, -1.0), stats[0]);
    try std.testing.expectEqual(@as(i32, -3), try refusalCode(refused));
    // Closing twice is a no-op, not a second free.
    closeSequence(sequence);
    closeSequence(sequence);
    try std.testing.expectEqual(
        @as(u32, 0),
        countsNow(t).sequences_live,
    );
}

test "a stale handle cannot drive the sequence that took its slot" {
    const t = instance();
    defer collapse(t);
    const allocator = std.testing.allocator;
    const prompt = try goldenPrompt(allocator);
    defer allocator.free(prompt);

    const model = try openModel(tiny_manifest, tiny_manifest.len);
    const abandoned = try openSequence(model, prompt.ptr, @intCast(prompt.len));
    closeSequence(abandoned);
    // The freed slot is the lowest free one, so this reissue is the ABA case:
    // same slot, same table entry, different handle.
    const replacement = try openSequence(model, prompt.ptr, @intCast(prompt.len));
    try std.testing.expect(abandoned != replacement);

    var token: u32 = 0;
    try std.testing.expectError(
        error.StaleHandle,
        stepSequence(abandoned, &token, null, 0),
    );
    try std.testing.expectError(
        error.StaleHandle,
        resetSequence(abandoned),
    );
    // The live handle is unaffected: the stale one did not consume its step.
    const first = try stepOnce(replacement);
    try std.testing.expectEqual(@as(u32, 169), first);
    try std.testing.expectEqual(@as(u32, 1), countsNow(t).sequences_live);
}

test "closing the model invalidates its live sequence with a code, not a crash" {
    const t = instance();
    defer collapse(t);
    const allocator = std.testing.allocator;
    const prompt = try goldenPrompt(allocator);
    defer allocator.free(prompt);

    const model = try openModel(tiny_manifest, tiny_manifest.len);
    const sequence = try openSequence(model, prompt.ptr, @intCast(prompt.len));
    try std.testing.expectEqual(@as(u32, 169), try stepOnce(sequence));

    // The C ABI cannot refuse this close — Rust's borrow checker is what stops
    // it — so the sequence must survive the model with a defined answer.
    closeModel(model);
    // A second close of the same handle is a no-op, not a second free.
    closeModel(model);
    var token: u32 = 0;
    var stats: [stats_len]f32 = [_]f32{0.0} ** stats_len;
    const refused = stepSequence(sequence, &token, stats[0..], stats_len);
    try std.testing.expectError(error.ModelClosed, refused);
    try std.testing.expectEqual(@as(i32, -4), try refusalCode(refused));
    try std.testing.expectError(error.ModelClosed, resetSequence(sequence));
    const counts = countsNow(t);
    try std.testing.expectEqual(@as(u32, 0), counts.models_live);
    try std.testing.expectEqual(@as(u32, 0), counts.sequences_live);
    try std.testing.expectEqual(@as(u32, 1), counts.sequences_orphaned);

    // Closing the orphan is safe and frees the slot.
    closeSequence(sequence);
    try std.testing.expectEqual(@as(u32, 0), countsNow(t).sequences_orphaned);
    // The table is usable again, which is what "defined error" buys.
    const reopened = try openModel(tiny_manifest, tiny_manifest.len);
    const again = try openSequence(reopened, prompt.ptr, @intCast(prompt.len));
    try std.testing.expectEqual(@as(u32, 169), try stepOnce(again));
}

test "reset replays the prompt through the handle" {
    const t = instance();
    defer collapse(t);
    const allocator = std.testing.allocator;
    const prompt = try goldenPrompt(allocator);
    defer allocator.free(prompt);

    const model = try openModel(tiny_manifest, tiny_manifest.len);
    const sequence = try openSequence(model, prompt.ptr, @intCast(prompt.len));
    const first = try stepOnce(sequence);
    const second = try stepOnce(sequence);
    try resetSequence(sequence);
    var token: u32 = 0;
    var stats: [stats_len]f32 = [_]f32{0.0} ** stats_len;
    try stepSequence(sequence, &token, stats[0..], stats_len);
    try std.testing.expectEqual(first, token);
    try std.testing.expectEqual(@as(f32, 1.0), stats[0]);
    try std.testing.expectEqual(
        @as(f32, @floatFromInt(prompt.len)),
        stats[1],
    );
    try std.testing.expectEqual(second, try stepOnce(sequence));
}

test "the ABI refuses a null or forged handle without loading it" {
    const t = instance();
    defer collapse(t);

    var token: u32 = 0;
    try std.testing.expectError(
        error.InvalidArgument,
        stepSequence(null, &token, null, 0),
    );
    try std.testing.expectEqual(
        @as(i32, -1),
        codeFor(error.InvalidArgument),
    );
    try std.testing.expectError(
        error.InvalidArgument,
        openSequence(null, null, 1),
    );
    try std.testing.expectError(
        error.InvalidArgument,
        openModel(null, 0),
    );
    // A pointer outside the pool is refused by the address test: nothing is
    // loaded through it, which is what keeps a forged handle from crashing.
    var forged: Cell = .{ .slot = 0, .generation = 1 };
    try std.testing.expectError(
        error.StaleHandle,
        stepSequence(&forged, &token, null, 0),
    );
    try std.testing.expectError(
        error.StaleHandle,
        resetSequence(&forged),
    );
    closeSequence(&forged);
    closeModel(&forged);
    try std.testing.expectEqual(@as(u32, 0), countsNow(t).sequences_live);
}

test "the manifest names a model this build can run" {
    const t = instance();
    defer collapse(t);

    const before = countsNow(t);
    // A document that parses but asks for the real checkpoint must be refused,
    // not answered with the tiny model.
    const real = "{\"model\":\"qwen3.8-flash-next-nvfp4\"}";
    try std.testing.expectError(
        error.UnsupportedModel,
        openModel(real, real.len),
    );
    const truncated = "{\"model\":";
    try std.testing.expectError(
        error.InvalidArgument,
        openModel(truncated, truncated.len),
    );
    const nameless = "{\"path\":\"tiny\"}";
    try std.testing.expectError(
        error.InvalidArgument,
        openModel(nameless, nameless.len),
    );
    const after = countsNow(t);
    try std.testing.expectEqual(@as(u32, 0), after.models_live);
    try std.testing.expectEqual(before.models_opened, after.models_opened);
}

test "the codes are the numbers the Rust side mirrors" {
    try std.testing.expectEqual(@as(i32, 0), @intFromEnum(Code.ok));
    try std.testing.expectEqual(@as(i32, -1), @intFromEnum(Code.invalid_argument));
    try std.testing.expectEqual(@as(i32, -2), @intFromEnum(Code.out_of_handles));
    try std.testing.expectEqual(@as(i32, -3), @intFromEnum(Code.stale_handle));
    try std.testing.expectEqual(@as(i32, -4), @intFromEnum(Code.model_closed));
    try std.testing.expectEqual(@as(i32, -5), @intFromEnum(Code.model_unavailable));
    try std.testing.expectEqual(@as(i32, -6), @intFromEnum(Code.unsupported_model));
    try std.testing.expectEqual(@as(i32, -7), @intFromEnum(Code.empty_prompt));
    try std.testing.expectEqual(@as(i32, -8), @intFromEnum(Code.prompt_too_long));
    try std.testing.expectEqual(@as(i32, -9), @intFromEnum(Code.context_full));
    try std.testing.expectEqual(@as(i32, -10), @intFromEnum(Code.step_failed));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(Cell));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Counts));
}
