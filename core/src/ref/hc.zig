//! Hyper-connections (`Qwen4ExpTextGatedResidual`, reference lines 1017-1053).
//!
//! The residual stream is `hc_count` copies of the hidden vector. A block reads
//! a low-rank, sigmoid-weighted mix of the normalised streams and writes its
//! output back into every stream through per-stream injection weights. The
//! final mixer before `lm_head` has `use_combine=false`: it returns the mixed
//! single stream and no injection weights.

const std = @import("std");

const fixtures = @import("fixtures.zig");
const ops = @import("ops.zig");

const assert = std.debug.assert;

/// Weights of one hyper-connection block.
pub const Weights = struct {
    /// `hc_norm.weight`, length `hc_count * hidden_size`; the norm is per
    /// stream (`group_size = hidden_size`) and scales by `1 + weight`.
    norm: []const f32,
    /// `input_mix_weight_down.weight`, `[hc_lowrank, hyper]`.
    mix_down: []const f32,
    /// `input_mix_weight_up.weight`, `[hyper, hc_lowrank]`.
    mix_up: []const f32,
    /// `block_inject_weight.weight`, `[hc_count, hyper]`; null for the mixer.
    inject: ?[]const f32,
};

/// Per-token buffers for one block.
pub const Scratch = struct {
    normed: []f32,
    mix_lowrank: []f32,
    mix_weight: []f32,

    /// Allocates the buffers for a `tokens_max` forward.
    pub fn init(
        allocator: std.mem.Allocator,
        config: *const fixtures.RefConfig,
    ) !Scratch {
        assert(config.hidden_size > 0);
        assert(config.hc_count > 0);

        const tokens: usize = config.max_position_embeddings;
        const hyper: usize = config.hyper_size();
        const lowrank: usize = config.hc_lowrank;

        const normed = try allocator.alloc(f32, tokens * hyper);
        errdefer allocator.free(normed);
        const mix_lowrank = try allocator.alloc(f32, tokens * lowrank);
        errdefer allocator.free(mix_lowrank);
        const mix_weight = try allocator.alloc(f32, tokens * hyper);
        errdefer allocator.free(mix_weight);

        return .{
            .normed = normed,
            .mix_lowrank = mix_lowrank,
            .mix_weight = mix_weight,
        };
    }

    /// Releases the buffers.
    pub fn deinit(
        self: *Scratch,
        allocator: std.mem.Allocator,
    ) void {
        assert(self.normed.len > 0);
        assert(self.mix_lowrank.len > 0);

        allocator.free(self.mix_weight);
        allocator.free(self.mix_lowrank);
        allocator.free(self.normed);
    }
};

/// Writes the mixed single stream into `mixed` and, when `injection` is given,
/// the per-stream injection weights.
pub fn forward(
    hyper_input: []const f32,
    tokens: u32,
    mixed: []f32,
    injection: ?[]f32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(mixed.len == tokens * config.hidden_size);

    forward_normalise(hyper_input, tokens, weights, scratch, config);
    forward_mix(hyper_input, tokens, mixed, weights, scratch, config);
    if (weights.inject) |inject_weight| {
        if (injection) |target| {
            forward_inject(tokens, target, inject_weight, scratch, config);
        } else {
            // No buffer supplied: the caller only wants the mixed stream.
        }
    } else {
        assert(injection == null);
    }
    assert(mixed.len == tokens * config.hidden_size);
}

/// Normalises both streams into `scratch.normed`.
fn forward_normalise(
    hyper_input: []const f32,
    tokens: u32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(hyper_input.len == tokens * config.hyper_size());

    const hyper: usize = config.hyper_size();
    for (0..tokens) |t| {
        const row = hyper_input[t * hyper ..][0..hyper];
        const normed = scratch.normed[t * hyper ..][0..hyper];
        ops.rmsnorm(
            row,
            weights.norm,
            normed,
            config.hidden_size,
            config.rms_norm_eps,
        );
    }
    assert(scratch.normed.len >= tokens * hyper);
}

/// Computes the sigmoid mix weights and the mixed single stream.
fn forward_mix(
    hyper_input: []const f32,
    tokens: u32,
    mixed: []f32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(mixed.len == tokens * config.hidden_size);

    const hyper: usize = config.hyper_size();
    const lowrank: usize = config.hc_lowrank;
    const hidden: usize = config.hidden_size;
    const streams: usize = config.hc_count;
    const divisor = @as(f32, @floatFromInt(streams));
    assert(divisor >= 1.0);

    ops.matmul(
        scratch.normed[0 .. tokens * hyper],
        weights.mix_down,
        scratch.mix_lowrank[0 .. tokens * lowrank],
        .{
            .rows = tokens,
            .in_features = config.hyper_size(),
            .out_features = config.hc_lowrank,
        },
    );
    // silu(down(x) / hc_count), then the up projection and a sigmoid.
    for (scratch.mix_lowrank[0 .. tokens * lowrank]) |*value| {
        value.* = ops.silu(value.* / divisor);
    }
    ops.matmul(
        scratch.mix_lowrank[0 .. tokens * lowrank],
        weights.mix_up,
        scratch.mix_weight[0 .. tokens * hyper],
        .{
            .rows = tokens,
            .in_features = config.hc_lowrank,
            .out_features = config.hyper_size(),
        },
    );
    for (scratch.mix_weight[0 .. tokens * hyper]) |*value| {
        value.* = ops.sigmoid(value.*);
    }
    assert(scratch.mix_weight.len >= tokens * hyper);

    // Mean over the streams of `weight * normed`, one hidden vector per token.
    for (0..tokens) |t| {
        const row = t * hyper;
        for (0..hidden) |h| {
            var total: f32 = 0.0;
            for (0..streams) |c| {
                const index = row + c * hidden + h;
                total += scratch.mix_weight[index] * scratch.normed[index];
            }
            mixed[t * hidden + h] = total / divisor;
        }
    }
    assert(hyper_input.len == tokens * hyper);
}

/// Computes the per-stream injection weights, `2 * sigmoid(w(x) / hc_count)`.
fn forward_inject(
    tokens: u32,
    injection: []f32,
    inject_weight: []const f32,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(injection.len == tokens * config.hc_count);

    const hyper: usize = config.hyper_size();
    const streams: usize = config.hc_count;
    const divisor = @as(f32, @floatFromInt(streams));

    ops.matmul(
        scratch.normed[0 .. tokens * hyper],
        inject_weight,
        injection[0 .. tokens * streams],
        .{
            .rows = tokens,
            .in_features = config.hyper_size(),
            .out_features = config.hc_count,
        },
    );
    for (injection[0 .. tokens * streams]) |*value| {
        value.* = 2.0 * ops.sigmoid(value.* / divisor);
    }
    assert(injection.len == tokens * streams);
}

test "the mixer reproduces a hand-computed two-stream mix" {
    const config = try fixtures.load_config(std.testing.allocator);
    var scratch = try Scratch.init(std.testing.allocator, &config);
    defer scratch.deinit(std.testing.allocator);

    // Identity normalisation: a zero norm weight scales by `1 + 0`.
    var norm: [256]f32 = [_]f32{0.0} ** 256;
    var mix_down: [16 * 256]f32 = [_]f32{0.0} ** (16 * 256);
    var mix_up: [256 * 16]f32 = [_]f32{0.0} ** (256 * 16);
    // A single low-rank channel that copies the first stream.
    for (0..128) |h| {
        mix_down[h] = 1.0;
        mix_up[h * 16] = 20.0;
        mix_up[(128 + h) * 16] = 20.0;
    }
    const weights = Weights{
        .norm = &norm,
        .mix_down = &mix_down,
        .mix_up = &mix_up,
        .inject = null,
    };

    var input: [256]f32 = undefined;
    for (0..256) |i| {
        input[i] = 1.0;
    }
    var mixed: [128]f32 = undefined;
    forward(&input, 1, &mixed, null, &weights, &scratch, &config);
    // Both streams are equal, so the mean of the sigmoid mix is 1.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mixed[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mixed[127], 1e-5);
}
