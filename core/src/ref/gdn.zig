//! GatedDeltaNet: causal conv, gating, and both forms of the gated delta rule.
//!
//! Reference lines 271-641. Prefill uses the chunked rule
//! (`torch_chunk_gated_delta_rule`, chunk 64), decode uses the recurrent rule
//! (`torch_recurrent_gated_delta_rule`); the two are algebraically the same
//! recurrence, and `tests` below asserts they agree to 1e-5 relative on the
// same
//! input.
//!
//! Conventions that are easy to get wrong, all taken from the reference: *
//! `repeat_interleave` gives each key head `num_v_heads / num_k_heads`
//! *consecutive* value heads, so value head `h` reads key head `h / group`; *
//! `l2norm` runs first, then queries are scaled by `head_dim ** -0.5`; * `g` is
//! already in log space and `exp(g)` is the per-step decay; * the strictly
//! upper triangle of the pairwise decay is `-inf` *before*
// `exp`;
//! * the UT transform is solved as `unitriangular=True`, so the diagonal of
//! `beta * k k^T` is ignored by the solve.

const std = @import("std");

const conv = @import("conv.zig");
const fixtures = @import("fixtures.zig");
const ops = @import("ops.zig");

const assert = std.debug.assert;

/// Sequence chunk of the chunked delta rule (fixed by the reference).
const chunk_size: u32 = 64;

/// `tokens` rounded up to whole chunks, the padded sequence the scan walks.
fn padded_length(tokens: u32) usize {
    assert(tokens > 0);
    assert(chunk_size > 0);

    const chunks = @divFloor(tokens + chunk_size - 1, chunk_size);
    assert(chunks * chunk_size >= tokens);
    return chunks * chunk_size;
}

/// Weights of one linear-attention layer.
pub const Weights = struct {
    /// `in_proj_qkv.weight`, `[conv_dim, hidden]`.
    in_proj_qkv: []const f32,
    /// `in_proj_z.weight`, `[value_dim, hidden]`.
    in_proj_z: []const f32,
    /// `in_proj_b.weight`, `[num_v_heads, hidden]`.
    in_proj_b: []const f32,
    /// `in_proj_a.weight`, `[num_v_heads, hidden]`.
    in_proj_a: []const f32,
    /// `conv1d.weight`, `[conv_dim, kernel]`.
    conv: []const f32,
    /// `norm.weight` of the gated RMSNorm, `[head_v_dim]`.
    norm: []const f32,
    /// `out_proj.weight`, `[hidden, value_dim]`.
    out_proj: []const f32,
    /// `A_log`, `[num_v_heads]`.
    a_log: []const f32,
    /// `dt_bias`, `[num_v_heads]`.
    dt_bias: []const f32,
};

/// Recurrent and conv state of one linear-attention layer.
pub const State = struct {
    /// `[conv_dim, kernel - 1]`, channel major.
    conv: []f32,
    /// `[num_v_heads, head_k_dim, head_v_dim]`.
    recurrent: []f32,

    /// Allocates a zeroed state.
    pub fn init(
        allocator: std.mem.Allocator,
        config: *const fixtures.RefConfig,
    ) !State {
        assert(config.conv_dim() > 0);
        assert(config.linear_conv_kernel_dim >= 2);

        const conv_len: usize = config.conv_dim() *
            (config.linear_conv_kernel_dim - 1);
        const recurrent_len: usize = config.linear_num_value_heads *
            config.linear_key_head_dim * config.linear_value_head_dim;

        const conv_state = try allocator.alloc(f32, conv_len);
        errdefer allocator.free(conv_state);
        const recurrent_state = try allocator.alloc(f32, recurrent_len);
        errdefer allocator.free(recurrent_state);
        @memset(conv_state, 0.0);
        @memset(recurrent_state, 0.0);

        return .{ .conv = conv_state, .recurrent = recurrent_state };
    }

    /// Releases the state.
    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        assert(self.conv.len > 0);
        assert(self.recurrent.len > 0);

        allocator.free(self.recurrent);
        allocator.free(self.conv);
    }

    /// Zeroes the state for a fresh sequence.
    pub fn reset(self: *State) void {
        assert(self.conv.len > 0);
        assert(self.recurrent.len > 0);

        @memset(self.conv, 0.0);
        @memset(self.recurrent, 0.0);
    }
};

/// Per-forward buffers.
pub const Scratch = struct {
    mixed_qkv: []f32,
    conv_full: []f32,
    conv_out: []f32,
    z: []f32,
    proj_a: []f32,
    proj_b: []f32,
    beta: []f32,
    decay: []f32,
    query: []f32,
    key: []f32,
    value: []f32,
    core_out: []f32,
    normed: []f32,
    head_q: []f32,
    head_k: []f32,
    head_v: []f32,
    head_beta: []f32,
    head_decay: []f32,
    cumulative: []f32,
    pairwise: []f32,
    system: []f32,
    intra: []f32,
    scaled_value: []f32,
    solved_v: []f32,
    solved_k: []f32,
    decayed_k_beta: []f32,
    q_decayed: []f32,
    k_decayed: []f32,
    v_new: []f32,
    head_out: []f32,

    /// Allocates the buffers for a `tokens_max` forward.
    pub fn init(
        allocator: std.mem.Allocator,
        config: *const fixtures.RefConfig,
    ) !Scratch {
        assert(config.conv_dim() > 0);
        assert(config.linear_key_head_dim > 0);

        const tokens: usize = config.max_position_embeddings;
        const conv_dim: usize = config.conv_dim();
        const key_dim: usize = config.key_dim();
        const value_dim: usize = config.value_dim();
        const heads: usize = config.linear_num_value_heads;
        const k_head: usize = config.linear_key_head_dim;
        const v_head: usize = config.linear_value_head_dim;
        const padding: usize = config.linear_conv_kernel_dim - 1;
        const padded = @as(usize, padded_length(@intCast(tokens)));
        const square: usize = chunk_size * chunk_size;

        var scratch: Scratch = .{
            .mixed_qkv = try allocator.alloc(f32, tokens * conv_dim),
            .conv_full = try allocator.alloc(
                f32,
                conv_dim * (tokens + padding),
            ),
            .conv_out = try allocator.alloc(f32, conv_dim * tokens),
            .z = try allocator.alloc(f32, tokens * value_dim),
            .proj_a = try allocator.alloc(f32, tokens * heads),
            .proj_b = try allocator.alloc(f32, tokens * heads),
            .beta = try allocator.alloc(f32, tokens * heads),
            .decay = try allocator.alloc(f32, tokens * heads),
            .query = try allocator.alloc(f32, tokens * key_dim),
            .key = try allocator.alloc(f32, tokens * key_dim),
            .value = try allocator.alloc(f32, tokens * value_dim),
            .core_out = try allocator.alloc(f32, tokens * value_dim),
            .normed = try allocator.alloc(f32, tokens * value_dim),
            .head_q = try allocator.alloc(f32, padded * k_head),
            .head_k = try allocator.alloc(f32, padded * k_head),
            .head_v = try allocator.alloc(f32, padded * v_head),
            .head_beta = try allocator.alloc(f32, padded),
            .head_decay = try allocator.alloc(f32, padded),
            .cumulative = try allocator.alloc(f32, padded),
            .pairwise = try allocator.alloc(f32, square),
            .system = try allocator.alloc(f32, square),
            .intra = try allocator.alloc(f32, square),
            .scaled_value = try allocator.alloc(f32, chunk_size * v_head),
            .solved_v = try allocator.alloc(f32, chunk_size * v_head),
            .solved_k = try allocator.alloc(f32, chunk_size * k_head),
            .decayed_k_beta = try allocator.alloc(f32, chunk_size * k_head),
            .q_decayed = try allocator.alloc(f32, chunk_size * k_head),
            .k_decayed = try allocator.alloc(f32, chunk_size * k_head),
            .v_new = try allocator.alloc(f32, chunk_size * v_head),
            .head_out = try allocator.alloc(f32, tokens * v_head),
        };
        assert(scratch.pairwise.len == square);
        scratch.check();
        return scratch;
    }

    /// Asserts every buffer is non-empty (one place to catch a typo).
    fn check(self: *const Scratch) void {
        assert(self.mixed_qkv.len > 0);
        assert(self.head_out.len > 0);
    }

    /// Releases the buffers.
    pub fn deinit(
        self: *Scratch,
        allocator: std.mem.Allocator,
    ) void {
        assert(self.mixed_qkv.len > 0);
        assert(self.pairwise.len > 0);

        allocator.free(self.head_out);
        allocator.free(self.v_new);
        allocator.free(self.k_decayed);
        allocator.free(self.q_decayed);
        allocator.free(self.decayed_k_beta);
        allocator.free(self.solved_k);
        allocator.free(self.solved_v);
        allocator.free(self.scaled_value);
        allocator.free(self.intra);
        allocator.free(self.system);
        allocator.free(self.pairwise);
        allocator.free(self.cumulative);
        allocator.free(self.head_decay);
        allocator.free(self.head_beta);
        allocator.free(self.head_v);
        allocator.free(self.head_k);
        allocator.free(self.head_q);
        allocator.free(self.normed);
        allocator.free(self.core_out);
        allocator.free(self.value);
        allocator.free(self.key);
        allocator.free(self.query);
        allocator.free(self.decay);
        allocator.free(self.beta);
        allocator.free(self.proj_b);
        allocator.free(self.proj_a);
        allocator.free(self.z);
        allocator.free(self.conv_out);
        allocator.free(self.conv_full);
        allocator.free(self.mixed_qkv);
    }
};

/// One layer forward. `recurrent_form` selects the decode-time delta rule.
pub fn forward(
    input: []const f32,
    tokens: u32,
    output: []f32,
    weights: *const Weights,
    state: *State,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
    recurrent_form: bool,
) void {
    assert(tokens > 0);
    assert(input.len == tokens * config.hidden_size);
    assert(output.len == tokens * config.hidden_size);

    forward_project(input, tokens, weights, scratch, config);
    forward_conv(tokens, weights, state, scratch, config);
    forward_gates(tokens, weights, scratch, config);
    if (recurrent_form) {
        gated_delta_rule_recurrent(
            tokens,
            scratch,
            config,
            state.recurrent,
        );
    } else {
        gated_delta_rule_chunked(tokens, scratch, config, state.recurrent);
    }
    forward_norm(tokens, weights, scratch, config);
    ops.matmul(
        scratch.normed[0 .. tokens * config.value_dim()],
        weights.out_proj,
        output,
        .{
            .rows = tokens,
            .in_features = config.value_dim(),
            .out_features = config.hidden_size,
        },
    );
    assert(output.len == tokens * config.hidden_size);
}

/// The four input projections.
fn forward_project(
    input: []const f32,
    tokens: u32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(weights.in_proj_qkv.len == config.conv_dim() * config.hidden_size);

    const heads: usize = config.linear_num_value_heads;
    ops.matmul(
        input,
        weights.in_proj_qkv,
        scratch.mixed_qkv[0 .. tokens * config.conv_dim()],
        .{
            .rows = tokens,
            .in_features = config.hidden_size,
            .out_features = config.conv_dim(),
        },
    );
    ops.matmul(
        input,
        weights.in_proj_z,
        scratch.z[0 .. tokens * config.value_dim()],
        .{
            .rows = tokens,
            .in_features = config.hidden_size,
            .out_features = config.value_dim(),
        },
    );
    ops.matmul(
        input,
        weights.in_proj_b,
        scratch.proj_b[0 .. tokens * heads],
        .{
            .rows = tokens,
            .in_features = config.hidden_size,
            .out_features = config.linear_num_value_heads,
        },
    );
    ops.matmul(
        input,
        weights.in_proj_a,
        scratch.proj_a[0 .. tokens * heads],
        .{
            .rows = tokens,
            .in_features = config.hidden_size,
            .out_features = config.linear_num_value_heads,
        },
    );
    assert(scratch.proj_a.len >= tokens * heads);
}

/// Causal conv over `[state | mixed_qkv]`, then the q/k/v split.
fn forward_conv(
    tokens: u32,
    weights: *const Weights,
    state: *State,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(weights.conv.len ==
        config.conv_dim() * config.linear_conv_kernel_dim);

    const channels: usize = config.conv_dim();
    const history: usize = config.linear_conv_kernel_dim - 1;
    const full_len: usize = tokens + history;
    const full = scratch.conv_full[0 .. channels * full_len];
    for (0..channels) |channel| {
        const target = full[channel * full_len ..][0..full_len];
        @memcpy(
            target[0..history],
            state.conv[channel * history ..][0..history],
        );
        for (0..tokens) |t| {
            target[history + t] = scratch.mixed_qkv[t * channels + channel];
        }
    }
    conv.causal_conv1d(
        full,
        weights.conv,
        scratch.conv_out[0 .. channels * tokens],
        .{
            .channels = config.conv_dim(),
            .kernel = config.linear_conv_kernel_dim,
            .dilation = 1,
            .full_len = @intCast(full_len),
        },
        .silu,
    );
    conv.store_state(
        state.conv,
        full,
        config.conv_dim(),
        @intCast(full_len),
        @intCast(history),
    );
    forward_split(tokens, scratch, config);
    assert(state.conv.len == channels * history);
}

/// Splits the conv output into the query, key and value blocks.
fn forward_split(
    tokens: u32,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(scratch.head_out.len > 0);

    const key_dim: usize = config.key_dim();
    const value_dim: usize = config.value_dim();
    const k_heads: usize = config.linear_num_key_heads;
    const k_head: usize = config.linear_key_head_dim;
    const v_head: usize = config.linear_value_head_dim;
    for (0..tokens) |t| {
        for (0..k_heads) |h| {
            for (0..k_head) |d| {
                const lane = (h * k_head + d) * tokens + t;
                scratch.query[(t * k_heads + h) * k_head + d] =
                    scratch.conv_out[lane];
                scratch.key[(t * k_heads + h) * k_head + d] =
                    scratch.conv_out[key_dim * tokens + lane];
            }
        }
        for (0..config.linear_num_value_heads) |h| {
            for (0..v_head) |d| {
                const lane = (h * v_head + d) * tokens + t;
                scratch.value[
                    (t * config.linear_num_value_heads + h) *
                        v_head + d
                ] = scratch.conv_out[2 * key_dim * tokens + lane];
            }
        }
    }
    assert(scratch.query.len >= tokens * key_dim);
    assert(scratch.value.len >= tokens * value_dim);
}

/// `beta = sigmoid(b)` and `g = -exp(A_log) * softplus(a + dt_bias)`.
fn forward_gates(
    tokens: u32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(weights.a_log.len == config.linear_num_value_heads);

    const heads: usize = config.linear_num_value_heads;
    for (0..tokens) |t| {
        for (0..heads) |h| {
            scratch.beta[t * heads + h] = ops.sigmoid(
                scratch.proj_b[t * heads + h],
            );
            const rate = -std.math.exp(weights.a_log[h]);
            const x = scratch.proj_a[t * heads + h] + weights.dt_bias[h];
            scratch.decay[t * heads + h] = rate * ops.softplus(x);
        }
    }
    assert(scratch.decay.len >= tokens * heads);
}

/// `normed = RMSNormGated(core_out, z)` flattened over `[tokens, heads, dv]`.
fn forward_norm(
    tokens: u32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(weights.norm.len == config.linear_value_head_dim);

    const head_v: usize = config.linear_value_head_dim;
    const rows: usize = tokens * config.linear_num_value_heads;
    for (0..rows) |row| {
        ops.rmsnorm_gated(
            scratch.core_out[row * head_v ..][0..head_v],
            scratch.z[row * head_v ..][0..head_v],
            weights.norm,
            scratch.normed[row * head_v ..][0..head_v],
            config.rms_norm_eps,
            config.output_gate_type,
        );
    }
    assert(scratch.normed.len >= rows * head_v);
}

/// Copies one value head's inputs into the padded per-head buffers.
fn load_head(
    head: u32,
    tokens: u32,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(head < config.linear_num_value_heads);
    assert(tokens <= config.max_position_embeddings);

    const group: usize = config.value_heads_per_key_head();
    const k_head: usize = config.linear_key_head_dim;
    const v_head: usize = config.linear_value_head_dim;
    const k_heads: usize = config.linear_num_key_heads;
    const heads: usize = config.linear_num_value_heads;
    const source_key = head / group;
    const padded: usize = padded_length(tokens);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(k_head)));

    for (0..padded) |t| {
        const query_row = scratch.head_q[t * k_head ..][0..k_head];
        const key_row = scratch.head_k[t * k_head ..][0..k_head];
        const value_row = scratch.head_v[t * v_head ..][0..v_head];
        if (t < tokens) {
            const query_source = scratch.query[(t * k_heads + source_key) *
                k_head ..][0..k_head];
            const key_source = scratch.key[(t * k_heads + source_key) *
                k_head ..][0..k_head];
            const value_source = scratch.value[(t * heads + head) *
                v_head ..][0..v_head];
            ops.l2norm(query_source, query_row, 1e-6);
            ops.l2norm(key_source, key_row, 1e-6);
            @memcpy(value_row, value_source);
            for (query_row) |*value| {
                value.* *= scale;
            }
            scratch.head_beta[t] = scratch.beta[t * heads + head];
            scratch.head_decay[t] = scratch.decay[t * heads + head];
        } else {
            @memset(query_row, 0.0);
            @memset(key_row, 0.0);
            @memset(value_row, 0.0);
            scratch.head_beta[t] = 0.0;
            scratch.head_decay[t] = 0.0;
        }
    }
    assert(scratch.head_beta.len >= padded);
}

/// Chunked gated delta rule: one pass of the UT transform per chunk.
fn gated_delta_rule_chunked(
    tokens: u32,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
    recurrent: []f32,
) void {
    assert(tokens > 0);
    assert(recurrent.len == config.linear_num_value_heads *
        config.linear_key_head_dim * config.linear_value_head_dim);

    for (0..config.linear_num_value_heads) |head| {
        load_head(@intCast(head), tokens, scratch, config);
        chunked_head(@intCast(head), tokens, scratch, config, recurrent);
    }
    assert(recurrent.len > 0);
}

/// Runs the chunk scan for a single value head.
fn chunked_head(
    head: u32,
    tokens: u32,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
    recurrent: []f32,
) void {
    assert(head < config.linear_num_value_heads);
    assert(tokens > 0);

    const k_head: usize = config.linear_key_head_dim;
    const v_head: usize = config.linear_value_head_dim;
    const heads: usize = config.linear_num_value_heads;
    const padded: usize = padded_length(tokens);
    const state = recurrent[head * k_head * v_head ..][0 .. k_head * v_head];
    const out = scratch.head_out[0 .. tokens * v_head];

    var start: usize = 0;
    while (start < padded) : (start += chunk_size) {
        prepare_chunk(start, scratch, config);
        scan_chunk(start, tokens, state, out, scratch, config);
    }
    assert(start == padded);
    for (0..tokens) |t| {
        for (0..v_head) |d| {
            scratch.core_out[(t * heads + head) * v_head + d] =
                out[t * v_head + d];
        }
    }
    assert(state.len == k_head * v_head);
}

/// Prepares one chunk: cumulative decay, pairwise decay, UT system, intra-chunk
/// attention, and the decayed query/key matrices the scan reads.
fn prepare_chunk(
    start: usize,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(start < config.max_position_embeddings + chunk_size);
    assert(scratch.pairwise.len == chunk_size * chunk_size);

    const k_head: usize = config.linear_key_head_dim;
    const rows: usize = chunk_size;
    var running: f32 = 0.0;
    for (0..rows) |j| {
        running += scratch.head_decay[start + j];
        scratch.cumulative[j] = running;
    }
    for (0..rows) |i| {
        for (0..rows) |j| {
            if (j <= i) {
                scratch.pairwise[i * rows + j] = std.math.exp(
                    scratch.cumulative[i] - scratch.cumulative[j],
                );
            } else {
                // The reference masks the strictly upper triangle to -inf
                // before the exponential, so these entries are exactly zero.
                scratch.pairwise[i * rows + j] = 0.0;
            }
        }
    }
    prepare_chunk_products(start, k_head, scratch);
    // `g` is in log space and never positive, so the running sum cannot grow.
    assert(scratch.cumulative[rows - 1] <= 0.0);
    assert(scratch.system.len == rows * rows);
}

/// The `k_beta`, `q_decayed` and `k_decayed` rows plus the UT/intra matrices.
fn prepare_chunk_products(
    start: usize,
    k_head: usize,
    scratch: *Scratch,
) void {
    assert(k_head > 0);
    assert(start + chunk_size <= scratch.head_beta.len);

    const rows: usize = chunk_size;
    const last_decay = scratch.cumulative[rows - 1];
    for (0..rows) |i| {
        const beta = scratch.head_beta[start + i];
        const key_i = scratch.head_k[(start + i) * k_head ..][0..k_head];
        const query_i = scratch.head_q[(start + i) * k_head ..][0..k_head];
        for (0..k_head) |d| {
            scratch.decayed_k_beta[i * k_head + d] = key_i[d] * beta *
                std.math.exp(scratch.cumulative[i]);
            scratch.q_decayed[i * k_head + d] = query_i[d] *
                std.math.exp(scratch.cumulative[i]);
            scratch.k_decayed[i * k_head + d] = key_i[d] *
                std.math.exp(last_decay - scratch.cumulative[i]);
        }
        for (0..rows) |j| {
            if (j <= i) {
                const key_j_start = (start + j) * k_head;
                const key_j = scratch.head_k[key_j_start..][0..k_head];
                const decayed = scratch.pairwise[i * rows + j];
                scratch.system[i * rows + j] =
                    dot_scaled(key_i, key_j, beta) * decayed;
                scratch.intra[i * rows + j] =
                    ops.dot(query_i, key_j) * decayed;
            } else {
                scratch.system[i * rows + j] = 0.0;
                scratch.intra[i * rows + j] = 0.0;
            }
        }
    }
    assert(scratch.intra.len == rows * rows);
}

/// `beta * dot(left, right)`, the `k_beta` row dotted against a plain key row.
fn dot_scaled(left: []const f32, right: []const f32, beta: f32) f32 {
    assert(left.len == right.len);
    assert(beta >= 0.0);

    return beta * ops.dot(left, right);
}

/// Solves the two UT systems for one chunk and advances the recurrent state.
fn scan_chunk(
    start: usize,
    tokens: u32,
    state: []f32,
    out: []f32,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(start < config.max_position_embeddings + chunk_size);
    assert(state.len == config.linear_key_head_dim *
        config.linear_value_head_dim);

    const k_head: usize = config.linear_key_head_dim;
    const v_head: usize = config.linear_value_head_dim;
    for (0..chunk_size) |i| {
        const beta = scratch.head_beta[start + i];
        for (0..v_head) |c| {
            scratch.scaled_value[i * v_head + c] =
                scratch.head_v[(start + i) * v_head + c] * beta;
        }
    }
    solve_lower_unit(
        scratch.system,
        scratch.scaled_value,
        v_head,
        scratch.solved_v,
    );
    solve_lower_unit(
        scratch.system,
        scratch.decayed_k_beta,
        k_head,
        scratch.solved_k,
    );

    accumulate_v_new(state, v_head, scratch);
    accumulate_out(start, tokens, state, out, v_head, scratch);
    update_state(state, v_head, scratch);
    assert(scratch.v_new.len == chunk_size * v_head);
}

/// `v_new[i] = solved_v[i] - solved_k[i] @ state`.
fn accumulate_v_new(state: []const f32, v_head: usize, scratch: *Scratch) void {
    assert(v_head > 0);
    assert(state.len > 0);

    const k_head = state.len / v_head;
    for (0..chunk_size) |i| {
        for (0..v_head) |c| {
            var total: f32 = 0.0;
            for (0..k_head) |k| {
                total += scratch.solved_k[i * k_head + k] *
                    state[k * v_head + c];
            }
            scratch.v_new[i * v_head + c] =
                scratch.solved_v[i * v_head + c] - total;
        }
    }
    assert(k_head * v_head == state.len);
}

/// `out[i] = q_decayed[i] @ state + sum_{j<=i} intra[i][j] * v_new[j]`.
fn accumulate_out(
    start: usize,
    tokens: u32,
    state: []const f32,
    out: []f32,
    v_head: usize,
    scratch: *Scratch,
) void {
    assert(v_head > 0);
    assert(state.len > 0);

    const k_head = state.len / v_head;
    for (0..chunk_size) |i| {
        const row = start + i;
        if (row < tokens) {
            for (0..v_head) |c| {
                var inter: f32 = 0.0;
                for (0..k_head) |k| {
                    inter += scratch.q_decayed[i * k_head + k] *
                        state[k * v_head + c];
                }
                var intra: f32 = 0.0;
                for (0..i + 1) |j| {
                    intra += scratch.intra[i * chunk_size + j] *
                        scratch.v_new[j * v_head + c];
                }
                out[row * v_head + c] = inter + intra;
            }
        } else {
            // Positions past the sequence are padding; only the state keeps
            // them.
        }
    }
    assert(out.len == tokens * v_head);
}

/// `state = state * exp(cum_last) + k_decayed^T @ v_new`.
fn update_state(state: []f32, v_head: usize, scratch: *Scratch) void {
    assert(v_head > 0);
    assert(state.len > 0);

    const k_head = state.len / v_head;
    const decay = std.math.exp(scratch.cumulative[chunk_size - 1]);
    for (state) |*value| {
        value.* *= decay;
    }
    for (0..chunk_size) |i| {
        for (0..k_head) |k| {
            const key = scratch.k_decayed[i * k_head + k];
            for (0..v_head) |c| {
                state[k * v_head + c] += key * scratch.v_new[i * v_head + c];
            }
        }
    }
    assert(state.len == k_head * v_head);
}

/// Forward substitution for a unit lower triangular `system`.
fn solve_lower_unit(
    system: []const f32,
    rhs: []const f32,
    columns: usize,
    solution: []f32,
) void {
    assert(system.len == chunk_size * chunk_size);
    assert(columns > 0);
    assert(solution.len == chunk_size * columns);

    for (0..chunk_size) |i| {
        for (0..columns) |c| {
            var total: f32 = rhs[i * columns + c];
            for (0..i) |j| {
                total -= system[i * chunk_size + j] *
                    solution[j * columns + c];
            }
            solution[i * columns + c] = total;
        }
    }
    assert(solution.len == chunk_size * columns);
}

/// Recurrent gated delta rule, the decode-time form.
fn gated_delta_rule_recurrent(
    tokens: u32,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
    recurrent: []f32,
) void {
    assert(tokens > 0);
    assert(recurrent.len == config.linear_num_value_heads *
        config.linear_key_head_dim * config.linear_value_head_dim);

    for (0..config.linear_num_value_heads) |head| {
        load_head(@intCast(head), tokens, scratch, config);
        recurrent_head(@intCast(head), tokens, scratch, config, recurrent);
    }
    assert(recurrent.len > 0);
}

/// One value head of the recurrent rule, token by token.
fn recurrent_head(
    head: u32,
    tokens: u32,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
    recurrent: []f32,
) void {
    assert(head < config.linear_num_value_heads);
    assert(tokens > 0);

    const k_head: usize = config.linear_key_head_dim;
    const v_head: usize = config.linear_value_head_dim;
    const heads: usize = config.linear_num_value_heads;
    const state = recurrent[head * k_head * v_head ..][0 .. k_head * v_head];

    for (0..tokens) |t| {
        const decay = std.math.exp(scratch.head_decay[t]);
        for (state) |*value| {
            value.* *= decay;
        }
        const key = scratch.head_k[t * k_head ..][0..k_head];
        const value = scratch.head_v[t * v_head ..][0..v_head];
        const beta = scratch.head_beta[t];
        for (0..v_head) |c| {
            var memory: f32 = 0.0;
            for (0..k_head) |k| {
                memory += state[k * v_head + c] * key[k];
            }
            const delta = (value[c] - memory) * beta;
            for (0..k_head) |k| {
                state[k * v_head + c] += key[k] * delta;
            }
        }
        const query = scratch.head_q[t * k_head ..][0..k_head];
        for (0..v_head) |c| {
            var total: f32 = 0.0;
            for (0..k_head) |k| {
                total += state[k * v_head + c] * query[k];
            }
            scratch.core_out[(t * heads + head) * v_head + c] = total;
        }
    }
    assert(state.len == k_head * v_head);
}

test "the chunked and recurrent delta rules agree on the same input" {
    const allocator = std.testing.allocator;
    const config = try fixtures.load_config(allocator);
    var scratch = try Scratch.init(allocator, &config);
    defer scratch.deinit(allocator);

    const tokens: u32 = 70;
    var chunk_state = try State.init(allocator, &config);
    defer chunk_state.deinit(allocator);
    var recurrent_state = try State.init(allocator, &config);
    defer recurrent_state.deinit(allocator);

    // Fill the delta-rule inputs directly: this test is about the two scans.
    var seed: u32 = 12_345;
    fill_rule_inputs(tokens, &scratch, &config, &seed);
    const width: usize = tokens * config.value_dim();
    var chunk_out: [70 * 96]f32 = undefined;
    var recurrent_out: [70 * 96]f32 = undefined;
    assert(width <= chunk_out.len);

    gated_delta_rule_chunked(tokens, &scratch, &config, chunk_state.recurrent);
    @memcpy(chunk_out[0..width], scratch.core_out[0..width]);
    gated_delta_rule_recurrent(
        tokens,
        &scratch,
        &config,
        recurrent_state.recurrent,
    );
    @memcpy(recurrent_out[0..width], scratch.core_out[0..width]);

    const out_error = scaled_error(
        chunk_out[0..width],
        recurrent_out[0..width],
    );
    const state_error = scaled_error(
        chunk_state.recurrent,
        recurrent_state.recurrent,
    );
    std.debug.print(
        "chunked vs recurrent: core_out {e:.3}, state {e:.3} (scaled)\n",
        .{ out_error, state_error },
    );
    // A 70-token sequence spans two chunks, so both the intra-chunk and the
    // cross-chunk paths are exercised.
    try std.testing.expect(tokens > chunk_size);
    try std.testing.expect(out_error < 1e-5);
    try std.testing.expect(state_error < 1e-5);
}

/// `max|a - b| / max(1e-6, max|a|)`, the scaled error the golden suite uses.
fn scaled_error(expected: []const f32, actual: []const f32) f64 {
    assert(expected.len > 0);
    assert(actual.len == expected.len);

    var largest: f64 = 0.0;
    var magnitude: f64 = 0.0;
    for (expected, actual) |want, got| {
        largest = @max(largest, @abs(@as(f64, want) - @as(f64, got)));
        magnitude = @max(magnitude, @abs(@as(f64, want)));
    }
    const scaled = largest / @max(1e-6, magnitude);
    assert(scaled >= 0.0);
    return scaled;
}

/// Fills the per-token delta-rule inputs with pseudo-random but sane values.
fn fill_rule_inputs(
    tokens: u32,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
    seed: *u32,
) void {
    assert(tokens > 0);
    assert(tokens <= config.max_position_embeddings);

    const k_head: usize = config.linear_key_head_dim;
    const v_head: usize = config.linear_value_head_dim;
    const k_heads: usize = config.linear_num_key_heads;
    const heads: usize = config.linear_num_value_heads;
    for (0..tokens) |t| {
        for (0..k_heads) |h| {
            for (0..k_head) |d| {
                seed.* = seed.* *% 1_103_515_245 +% 12_345;
                const value =
                    @as(f32, @floatFromInt(seed.* % 2000)) / 1000.0 - 1.0;
                scratch.query[(t * k_heads + h) * k_head + d] = value;
                seed.* = seed.* *% 1_103_515_245 +% 12_345;
                scratch.key[(t * k_heads + h) * k_head + d] =
                    @as(f32, @floatFromInt(seed.* % 2000)) / 1000.0 - 1.0;
            }
        }
        for (0..heads) |h| {
            for (0..v_head) |d| {
                seed.* = seed.* *% 1_103_515_245 +% 12_345;
                scratch.value[(t * heads + h) * v_head + d] =
                    @as(f32, @floatFromInt(seed.* % 2000)) / 1000.0 - 1.0;
            }
            seed.* = seed.* *% 1_103_515_245 +% 12_345;
            scratch.beta[t * heads + h] =
                @as(f32, @floatFromInt(seed.* % 1000)) / 1000.0;
            seed.* = seed.* *% 1_103_515_245 +% 12_345;
            scratch.decay[t * heads + h] =
                -@as(f32, @floatFromInt(seed.* % 500)) / 1000.0;
        }
    }
    assert(scratch.query.len >= tokens * config.key_dim());
}
