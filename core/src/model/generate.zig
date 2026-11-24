//! Greedy decoding: the argmax id on the device, never a host logit transfer.
//!
//! Two forms, both of which the end-to-end check requires and compares:
//!
//! * `recompute` forwards the whole sequence from a fresh cache on every step.
//!   It is the slow, obviously-correct form and is what catches a cache bug by
//!   disagreeing with the cached form;
//! * `cached` prefills the prompt and then forwards one token per step, with
//!   the KV, indexer and GDN caches carried across steps.
//!
//! Both take the next id from `ew_argmax` on the device: the host reads one
//! `u32`, not the vocabulary.

const std = @import("std");

const exec = @import("exec.zig");

const assert = std.debug.assert;

/// Which cache discipline a greedy run uses.
pub const Form = enum {
    recompute,
    cached,
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
