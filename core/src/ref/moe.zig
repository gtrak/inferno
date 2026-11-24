//! Sparse mixture of experts (`TopKRouter`, `Experts`, `SparseMoeBlock`).
//!
//! Reference lines 935-1015. The router softmaxes its logits in f32, takes the
//! top `num_experts_per_tok` *probabilities* (not logits), renormalises them,
// and
//! the experts accumulate `down(silu(gate) * up) * weight` per token in expert
//! order. The shared expert is gated by `sigmoid(shared_expert_gate(x))` and
//! added to the routed sum.

const std = @import("std");

const fixtures = @import("fixtures.zig");
const ops = @import("ops.zig");

const assert = std.debug.assert;

/// Most experts any layer may have (the real checkpoint uses 512).
pub const experts_max: u32 = 512;

/// Weights of one sparse MoE block.
pub const Weights = struct {
    /// `mlp.gate.weight`, `[num_experts, hidden]`.
    router: []const f32,
    /// Per-expert `mlp.experts.E.gate_proj.weight`, `[intermediate, hidden]`.
    gate_proj: [experts_max][]const f32,
    /// Per-expert `mlp.experts.E.up_proj.weight`, `[intermediate, hidden]`.
    up_proj: [experts_max][]const f32,
    /// Per-expert `mlp.experts.E.down_proj.weight`, `[hidden, intermediate]`.
    down_proj: [experts_max][]const f32,
    /// `mlp.shared_expert.gate_proj.weight`, `[shared_intermediate, hidden]`.
    shared_gate: []const f32,
    /// `mlp.shared_expert.up_proj.weight`, `[shared_intermediate, hidden]`.
    shared_up: []const f32,
    /// `mlp.shared_expert.down_proj.weight`, `[hidden, shared_intermediate]`.
    shared_down: []const f32,
    /// `mlp.shared_expert_gate.weight`, `[1, hidden]`.
    shared_gate_logit: []const f32,
};

/// Per-token buffers for one block.
pub const Scratch = struct {
    router_logits: []f32,
    probs: []f32,
    topk_values: []f32,
    topk_indices: []u32,
    /// One expert's `[gate | up]` projection output.
    gate_up: []f32,
    /// One expert's activated intermediate vector.
    hidden: []f32,
    /// One expert's `down` output, before the routing weight.
    down: []f32,
    /// The routed sum over the top-k experts.
    expert_out: []f32,
    /// The gated shared-expert output, `sigmoid(gate) * down(silu(g) * up)`.
    shared_out: []f32,
    /// The shared expert's `[gate | up]` output and activated vector.
    shared_gate_up: []f32,
    shared_hidden: []f32,

    /// Allocates the buffers for a `tokens_max` forward.
    pub fn init(
        allocator: std.mem.Allocator,
        config: *const fixtures.RefConfig,
    ) !Scratch {
        assert(config.num_experts > 0);
        assert(config.num_experts_per_tok <= config.num_experts);

        const tokens: usize = config.max_position_embeddings;
        const hidden: usize = config.hidden_size;
        const experts: usize = config.num_experts;
        const top_k: usize = config.num_experts_per_tok;
        const intermediate: usize = config.moe_intermediate_size;
        const shared: usize = config.shared_expert_intermediate_size;

        const router_logits = try allocator.alloc(f32, tokens * experts);
        errdefer allocator.free(router_logits);
        const probs = try allocator.alloc(f32, tokens * experts);
        errdefer allocator.free(probs);
        const topk_values = try allocator.alloc(f32, tokens * top_k);
        errdefer allocator.free(topk_values);
        const topk_indices = try allocator.alloc(u32, tokens * top_k);
        errdefer allocator.free(topk_indices);
        const gate_up = try allocator.alloc(f32, 2 * intermediate);
        errdefer allocator.free(gate_up);
        const activated = try allocator.alloc(f32, intermediate);
        errdefer allocator.free(activated);
        const down = try allocator.alloc(f32, hidden);
        errdefer allocator.free(down);
        const expert_out = try allocator.alloc(f32, tokens * hidden);
        errdefer allocator.free(expert_out);
        const shared_out = try allocator.alloc(f32, tokens * hidden);
        errdefer allocator.free(shared_out);
        const shared_gate_up = try allocator.alloc(f32, 2 * shared);
        errdefer allocator.free(shared_gate_up);
        const shared_hidden = try allocator.alloc(f32, shared);
        errdefer allocator.free(shared_hidden);

        return .{
            .router_logits = router_logits,
            .probs = probs,
            .topk_values = topk_values,
            .topk_indices = topk_indices,
            .gate_up = gate_up,
            .hidden = activated,
            .down = down,
            .expert_out = expert_out,
            .shared_out = shared_out,
            .shared_gate_up = shared_gate_up,
            .shared_hidden = shared_hidden,
        };
    }

    /// Releases the buffers.
    pub fn deinit(
        self: *Scratch,
        allocator: std.mem.Allocator,
    ) void {
        assert(self.router_logits.len > 0);
        assert(self.gate_up.len > 0);

        allocator.free(self.shared_hidden);
        allocator.free(self.shared_gate_up);
        allocator.free(self.shared_out);
        allocator.free(self.expert_out);
        allocator.free(self.down);
        allocator.free(self.hidden);
        allocator.free(self.gate_up);
        allocator.free(self.topk_indices);
        allocator.free(self.topk_values);
        allocator.free(self.probs);
        allocator.free(self.router_logits);
    }
};

/// One block forward: routed experts plus the gated shared expert.
pub fn forward(
    input: []const f32,
    tokens: u32,
    output: []f32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(input.len == tokens * config.hidden_size);
    assert(output.len == tokens * config.hidden_size);

    forward_route(input, tokens, weights, scratch, config);
    forward_experts(input, tokens, weights, scratch, config);
    forward_shared(input, tokens, weights, scratch, config);
    const hidden: usize = config.hidden_size;
    for (0..tokens * hidden) |i| {
        output[i] = scratch.expert_out[i] + scratch.shared_out[i];
    }
    assert(output.len == tokens * hidden);
}

/// Softmaxes the router logits and keeps the top-k probabilities.
pub fn forward_route(
    input: []const f32,
    tokens: u32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(weights.router.len == config.num_experts * config.hidden_size);

    const experts: usize = config.num_experts;
    const top_k: usize = config.num_experts_per_tok;

    ops.matmul(
        input,
        weights.router,
        scratch.router_logits[0 .. tokens * experts],
        .{
            .rows = tokens,
            .in_features = config.hidden_size,
            .out_features = config.num_experts,
        },
    );
    for (0..tokens) |t| {
        const logits = scratch.router_logits[t * experts ..][0..experts];
        const probs = scratch.probs[t * experts ..][0..experts];
        @memcpy(probs, logits);
        ops.softmax(probs);
        const values = scratch.topk_values[t * top_k ..][0..top_k];
        const indices = scratch.topk_indices[t * top_k ..][0..top_k];
        topk_descending(probs, values, indices);
        if (config.norm_topk_prob) {
            var total: f32 = 0.0;
            for (values) |value| {
                total += value;
            }
            assert(total > 0.0);
            for (values) |*value| {
                value.* /= total;
            }
        } else {
            assert(!config.norm_topk_prob);
        }
    }
    assert(scratch.topk_indices.len >= tokens * top_k);
}

/// Selects the `values.len` largest entries, descending, ties by lower index.
fn topk_descending(
    source: []const f32,
    values: []f32,
    indices: []u32,
) void {
    assert(source.len > 0);
    assert(values.len == indices.len);
    assert(values.len <= source.len);
    assert(source.len <= experts_max);

    var taken: [experts_max]bool = undefined;
    for (0..experts_max) |i| {
        taken[i] = false;
    }
    for (values, 0..) |*value, slot| {
        var best: u32 = 0;
        var best_value = -std.math.inf(f32);
        for (source, 0..) |candidate, index| {
            if (taken[index]) {
                continue;
            } else {
                if (candidate > best_value) {
                    best_value = candidate;
                    best = @intCast(index);
                } else {
                    continue;
                }
            }
        }
        taken[best] = true;
        value.* = best_value;
        indices[slot] = best;
    }
    assert(values.len == indices.len);
}

/// Accumulates the routed experts in expert order, as the reference does.
pub fn forward_experts(
    input: []const f32,
    tokens: u32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(scratch.expert_out.len >= tokens * config.hidden_size);

    const hidden: usize = config.hidden_size;
    const top_k: usize = config.num_experts_per_tok;
    @memset(scratch.expert_out[0 .. tokens * hidden], 0.0);

    for (0..config.num_experts) |expert| {
        for (0..tokens) |t| {
            for (0..top_k) |slot| {
                const routed = scratch.topk_indices[t * top_k + slot];
                if (routed == expert) {
                    const weight = scratch.topk_values[t * top_k + slot];
                    forward_expert(
                        input,
                        t,
                        weight,
                        expert,
                        weights,
                        scratch,
                        config,
                    );
                } else {
                    continue;
                }
            }
        }
    }
    assert(scratch.expert_out.len >= tokens * hidden);
}

/// Adds one (token, expert, weight) contribution into `scratch.expert_out`.
fn forward_expert(
    input: []const f32,
    token: usize,
    weight: f32,
    expert: usize,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(weight > 0.0);
    assert(expert < config.num_experts);

    const hidden: usize = config.hidden_size;
    const intermediate: usize = config.moe_intermediate_size;
    const row = input[token * hidden ..][0..hidden];
    const gate_up = scratch.gate_up[0 .. 2 * intermediate];
    ops.matmul(
        row,
        weights.gate_proj[expert],
        gate_up[0..intermediate],
        .{
            .rows = 1,
            .in_features = config.hidden_size,
            .out_features = config.moe_intermediate_size,
        },
    );
    ops.matmul(
        row,
        weights.up_proj[expert],
        gate_up[intermediate..][0..intermediate],
        .{
            .rows = 1,
            .in_features = config.hidden_size,
            .out_features = config.moe_intermediate_size,
        },
    );
    for (0..intermediate) |i| {
        scratch.hidden[i] = ops.silu(gate_up[i]) * gate_up[intermediate + i];
    }
    ops.matmul(
        scratch.hidden[0..intermediate],
        weights.down_proj[expert],
        scratch.down,
        .{
            .rows = 1,
            .in_features = config.moe_intermediate_size,
            .out_features = config.hidden_size,
        },
    );
    for (0..hidden) |i| {
        scratch.expert_out[token * hidden + i] +=
            scratch.down[i] * weight;
    }
    assert(scratch.down.len == hidden);
}

/// The shared expert, gated by `sigmoid(shared_expert_gate(x))`.
pub fn forward_shared(
    input: []const f32,
    tokens: u32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(weights.shared_gate_logit.len == config.hidden_size);

    const hidden: usize = config.hidden_size;
    const shared: usize = config.shared_expert_intermediate_size;
    for (0..tokens) |t| {
        const row = input[t * hidden ..][0..hidden];
        const gate_up = scratch.shared_gate_up[0 .. 2 * shared];
        ops.matmul(
            row,
            weights.shared_gate,
            gate_up[0..shared],
            .{
                .rows = 1,
                .in_features = config.hidden_size,
                .out_features = config.shared_expert_intermediate_size,
            },
        );
        ops.matmul(
            row,
            weights.shared_up,
            gate_up[shared..][0..shared],
            .{
                .rows = 1,
                .in_features = config.hidden_size,
                .out_features = config.shared_expert_intermediate_size,
            },
        );
        for (0..shared) |i| {
            scratch.shared_hidden[i] =
                ops.silu(gate_up[i]) * gate_up[shared + i];
        }
        ops.matmul(
            scratch.shared_hidden[0..shared],
            weights.shared_down,
            scratch.down,
            .{
                .rows = 1,
                .in_features = config.shared_expert_intermediate_size,
                .out_features = config.hidden_size,
            },
        );
        const gate = ops.sigmoid(ops.dot(row, weights.shared_gate_logit));
        for (0..hidden) |i| {
            scratch.shared_out[t * hidden + i] = scratch.down[i] * gate;
        }
    }
    assert(scratch.shared_out.len >= tokens * hidden);
}

test "topk_descending returns the largest probabilities in order" {
    var values: [2]f32 = undefined;
    var indices: [2]u32 = undefined;
    topk_descending(&[4]f32{ 0.1, 0.5, 0.3, 0.5 }, &values, &indices);
    try std.testing.expectEqual(@as(u32, 1), indices[0]);
    try std.testing.expectEqual(@as(u32, 3), indices[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), values[0], 1e-6);
}

test "the router renormalises its top-k probabilities" {
    const config = try fixtures.load_config(std.testing.allocator);
    var scratch = try Scratch.init(std.testing.allocator, &config);
    defer scratch.deinit(std.testing.allocator);

    var router: [4 * 128]f32 = [_]f32{0.0} ** (4 * 128);
    // Expert e only sees hidden dim e, so the input alone decides the ranking.
    for (0..4) |e| {
        router[e * 128 + e] = 1.0;
    }
    var weights = empty_weights();
    weights.router = &router;
    var input: [128]f32 = [_]f32{0.0} ** 128;
    input[0] = 3.0;
    input[1] = 1.0;
    forward_route(&input, 1, &weights, &scratch, &config);
    try std.testing.expectEqual(@as(u32, 0), scratch.topk_indices[0]);
    try std.testing.expectEqual(@as(u32, 1), scratch.topk_indices[1]);
    const values = scratch.topk_values[0..2];
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0),
        values[0] + values[1],
        1e-6,
    );
    try std.testing.expect(values[0] > values[1]);
}

/// All-null weight set, so a test can fill in only what it exercises.
fn empty_weights() Weights {
    var weights = Weights{
        .router = &[_]f32{},
        .gate_proj = undefined,
        .up_proj = undefined,
        .down_proj = undefined,
        .shared_gate = &[_]f32{},
        .shared_up = &[_]f32{},
        .shared_down = &[_]f32{},
        .shared_gate_logit = &[_]f32{},
    };
    for (0..experts_max) |e| {
        weights.gate_proj[e] = &[_]f32{};
        weights.up_proj[e] = &[_]f32{};
        weights.down_proj[e] = &[_]f32{};
    }
    assert(weights.gate_proj[0].len == 0);
    return weights;
}
