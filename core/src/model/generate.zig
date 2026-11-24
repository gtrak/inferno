//! Greedy decoding: the argmax id on the device, never a host logit transfer.
//!
//! `greedy` has two forms, both of which the end-to-end check requires and
//! compares:
//!
//! * `recompute` forwards the whole sequence from a fresh cache on every step.
//!   It is the slow, obviously-correct form and is what catches a cache bug by
//!   disagreeing with the cached form;
//! * `cached` prefills the prompt and then forwards one token per step, with
//!   the KV, indexer and GDN caches carried across steps.
//!
//! `Sequence` is the runtime form: it owns one sequence's logical position and
//! runs the cached discipline through a stop surface (EOS, token ceiling, a
//! caller predicate). The caches themselves stay owned by `exec.Model`, which
//! allocates them once; `Sequence.begin` resets them for the next request.
//!
//! Every form takes the next id from `ew_argmax` on the device: the host reads
//! one `u32`, not the vocabulary.

const std = @import("std");

const exec = @import("exec.zig");

const assert = std.debug.assert;

/// Which cache discipline a greedy run uses.
pub const Form = enum {
    recompute,
    cached,
};

/// A caller-supplied stop test over a generated id.
///
/// `index` is zero-based within the run. The context is opaque to the loop, so
/// one predicate function can carry per-request state without an allocation.
pub const Predicate = *const fn (id: i64, index: u32, context: ?*anyopaque) bool;

/// Why a run stopped.
pub const StopReason = enum {
    /// The generated id equalled `Stop.eos`.
    eos,
    /// The run produced `Stop.max_tokens` ids.
    max_tokens,
    /// The caller predicate returned true.
    predicate,
    /// The sequence reached the arena's last position before the next forward.
    context_limit,
    /// The caller's output slice filled before any other stop condition.
    output_full,
};

/// The stop surface: EOS id, token ceiling, and an optional predicate.
///
/// Sampling policy is deliberately absent: the id is always `ew_argmax`'s.
/// The stop tests run in the order `eos`, `max_tokens`, `predicate`, so an id
/// that is both EOS and the ceiling reports `eos`.
pub const Stop = struct {
    eos: ?i64 = null,
    max_tokens: u32,
    predicate: ?Predicate = null,
    context: ?*anyopaque = null,

    /// The reason `id` at `index` stops the run, or null to continue.
    pub fn check(self: *const Stop, id: i64, index: u32) ?StopReason {
        assert(self.max_tokens > 0);
        assert(index < self.max_tokens);

        if (self.eos) |eos| {
            assert(eos >= 0);
            if (id == eos) {
                return .eos;
            } else {
                assert(id != eos);
            }
        }
        if (index + 1 == self.max_tokens) {
            return .max_tokens;
        } else {
            assert(index + 1 < self.max_tokens);
        }
        if (self.predicate) |predicate| {
            if (predicate(id, index, self.context)) {
                return .predicate;
            }
        }
        return null;
    }
};

/// How many ids a run produced and why it stopped.
pub const Run = struct {
    count: usize,
    reason: StopReason,
};

/// One sequence's decode position over an `exec.Model`'s caches.
///
/// `init` only records the logical position; `begin` resets the model's caches
/// and prefills. Neither allocates. `nextId` is the only host read, and it is
/// one `u32` off the device's argmax.
pub const Sequence = struct {
    model: *exec.Model,
    /// Tokens in the caches: the prompt plus every id fed back.
    cached: u32,
    /// Ids produced since `begin`.
    generated: u32,
    /// The most recent id; the next `step` feeds it unless this is the first.
    pending: i64,
    /// Incremented by every `begin`. A snapshot records it, so a restore
    /// across a reset — which invalidates the captured KV positions — is a
    /// refusal rather than a silent resurrection of the old sequence.
    epoch: u64,

    /// Binds a decode position to `model`, whose caches it reuses.
    pub fn init(model: *exec.Model) Sequence {
        assert(model.state.arena_length > 0);
        assert(model.plan.layer_count > 0);

        return .{
            .model = model,
            .cached = 0,
            .generated = 0,
            .pending = 0,
            .epoch = 0,
        };
    }

    /// Resets the caches to start-of-sequence and prefills `prompt`.
    ///
    /// The KV, indexer and GDN allocations are `exec.Model`'s and are reused:
    /// a second request costs a reset, never an allocation.
    pub fn begin(self: *Sequence, prompt: []const i64) !void {
        assert(self.model.state.arena_length > 0);
        assert(self.model.plan.layer_count > 0);
        if (prompt.len == 0) {
            return error.EmptyPrompt;
        }
        if (prompt.len > @as(usize, self.model.state.arena_length)) {
            return error.PromptTooLong;
        }
        try self.model.reset();
        try self.model.forward(prompt);
        self.epoch +%= 1;
        self.cached = @intCast(prompt.len);
        self.generated = 0;
        self.pending = 0;
        assert(self.cached == @as(u32, @intCast(prompt.len)));
    }

    /// Produces the next greedy id, extending the caches by one token.
    ///
    /// The first step reads the prefill's logits; every later step first
    /// forwards the previous id. The returned id is not in the caches yet.
    pub fn step(self: *Sequence) !i64 {
        assert(self.cached > 0);
        assert(self.cached == self.model.state.length);
        if (self.generated > 0) {
            if (self.cached + 1 > self.model.state.arena_length) {
                return error.ContextFull;
            }
            const single = [_]i64{self.pending};
            try self.model.forward(&single);
            self.cached += 1;
        }
        const id = try nextId(self.model);
        assert(self.cached == self.model.state.length);
        self.pending = id;
        self.generated += 1;
        return id;
    }

    /// Runs until `stop` fires, the arena ends, or `out` fills.
    ///
    /// On `context_limit` the sequence is exhausted: `out[0..result.count]`
    /// holds every id produced before the limit. The caller's `out` is the
    /// only storage; a `Sequence` never allocates.
    pub fn run(self: *Sequence, stop: Stop, out: []i64) !Run {
        assert(stop.max_tokens > 0);
        assert(out.len > 0);
        assert(self.cached > 0);

        var count: usize = 0;
        while (count < out.len and count < stop.max_tokens) {
            const id = self.step() catch |err| switch (err) {
                error.ContextFull => return .{
                    .count = count,
                    .reason = .context_limit,
                },
                else => return err,
            };
            out[count] = id;
            count += 1;
            if (stop.check(id, @intCast(count - 1))) |reason| {
                assert(count > 0);
                return .{ .count = count, .reason = reason };
            }
        }
        if (count == stop.max_tokens) {
            return .{ .count = count, .reason = .max_tokens };
        } else {
            assert(count == out.len);
            return .{ .count = count, .reason = .output_full };
        }
    }
};

/// Runs `count` greedy steps after `prompt`, writing the ids.
pub fn greedy(
    model: *exec.Model,
    prompt: []const i64,
    count: u32,
    out: []i64,
    form: Form,
) !void {
    assert(prompt.len > 0);
    assert(count > 0);
    assert(out.len == count);

    switch (form) {
        .recompute => try greedyRecompute(model, prompt, out),
        .cached => try greedyCached(model, prompt, out),
    }
    assert(out.len == count);
}

/// Full-recompute decoding: every step forwards the whole prefix.
fn greedyRecompute(
    model: *exec.Model,
    prompt: []const i64,
    out: []i64,
) !void {
    assert(prompt.len > 0);
    assert(model.sequence.len >= prompt.len + out.len);

    for (0..out.len) |step| {
        try model.reset();
        @memcpy(model.sequence[0..prompt.len], prompt);
        @memcpy(model.sequence[prompt.len..][0..step], out[0..step]);
        try model.forward(model.sequence[0 .. prompt.len + step]);
        out[step] = try nextId(model);
    }
    assert(out.len > 0);
}

/// Incremental decoding: prefill once, then one token per step.
fn greedyCached(
    model: *exec.Model,
    prompt: []const i64,
    out: []i64,
) !void {
    assert(prompt.len > 0);
    assert(out.len > 0);

    try model.reset();
    try model.forward(prompt);
    out[0] = try nextId(model);
    for (1..out.len) |step| {
        const single = [_]i64{out[step - 1]};
        try model.forward(&single);
        out[step] = try nextId(model);
    }
    assert(out.len > 0);
}

/// The argmax id of the last forward, read back as one word.
pub fn nextId(model: *exec.Model) !i64 {
    assert(model.state.argmax_index.bytes >= 4);

    var index = [_]u32{0};
    try model.state.downloadRaw(
        &model.state.argmax_index,
        std.mem.sliceAsBytes(index[0..]),
    );
    const id: i64 = @intCast(index[0]);
    assert(id >= 0);
    assert(id < model.config.vocab_size);
    return id;
}
