//! Per-Layer Embedding: hashed n-gram ids, the gather, and the dilated conv.
//!
//! Reference lines 1055-1272. The n-gram id of head `h` for token `t` is `XOR_s
//! (shifted_s(t) * multiplier_s) mod head_vocab_size[h] +
// head_offset[h]`,
//! computed in int64 with wrapping multiplication and floored `remainder`, on a
//! token history that resets at every EOS (a segment boundary never looks back
//! into the previous segment).

const std = @import("std");

const conv = @import("conv.zig");
const fixtures = @import("fixtures.zig");
const ops = @import("ops.zig");

const assert = std.debug.assert;

/// Most shards the n-gram table may be split into.
pub const shards_max: u32 = 128;

/// The splitmix64 constants of the reference.
const splitmix_gamma: u64 = 0x9E3779B97F4A7C15;
const splitmix_m1: u64 = 0xBF58476D1CE4E5B9;
const splitmix_m2: u64 = 0x94D049BB133111EB;
const prime_seed_stride: u64 = 10007;

/// Weights of one PLE block.
pub const Weights = struct {
    /// Row blocks of the n-gram table in id order (`split_ngram_parts` shards),
    /// each `[rows, head_dim]`.
    shards: [shards_max][]const f32,
    shard_count: u32,
    /// Columns of one table row (`ple_embed_dim / ngram_heads`).
    head_dim: u32,
    /// `layer_multipliers`, `[ngram_size]`.
    multipliers: []const i64,
    /// `ngram_heads_vocab_sizes`, `[ngram_heads]`.
    head_vocab_sizes: []const i64,
    /// `ngram_heads_offsets`, `[ngram_heads]`.
    head_offsets: []const i64,
    /// `key_proj.weight`, `[hyper, ple_embed_dim]`.
    key_proj: []const f32,
    /// `value_proj.weight`, `[hidden, ple_embed_dim]`.
    value_proj: []const f32,
    /// `norm_key.weight`, `[hyper]`.
    norm_key: []const f32,
    /// `norm_query.weight`, `[hyper]`.
    norm_query: []const f32,
    /// `norm_conv.weight`, `[hyper]`.
    norm_conv: []const f32,
    /// `conv1d.weight`, `[hyper, ple_conv_kernel_size]`.
    conv: []const f32,
};

/// The n-gram context and dilated-conv state of one PLE layer.
pub const State = struct {
    /// `[context_len]`, the token ids the next forward reads as history.
    context: []i64,
    /// `[hyper, (ple_conv_kernel_size - 1) * ngram_size]`, channel major.
    conv: []f32,

    /// Allocates the state, with the context pre-filled with EOS.
    pub fn init(
        allocator: std.mem.Allocator,
        config: *const fixtures.RefConfig,
    ) !State {
        assert(config.ngram_context_len() > 0);
        assert(config.ple_conv_state_len() > 0);

        const context = try allocator.alloc(
            i64,
            config.ngram_context_len(),
        );
        errdefer allocator.free(context);
        const conv_state = try allocator.alloc(
            f32,
            config.hyper_size() * config.ple_conv_state_len(),
        );
        errdefer allocator.free(conv_state);
        @memset(context, config.eos_token_id);
        @memset(conv_state, 0.0);

        return .{ .context = context, .conv = conv_state };
    }

    /// Releases the state.
    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        assert(self.context.len > 0);
        assert(self.conv.len > 0);

        allocator.free(self.conv);
        allocator.free(self.context);
    }

    /// Restores the initial state (EOS context, zero conv).
    pub fn reset(self: *State, config: *const fixtures.RefConfig) void {
        assert(self.context.len > 0);
        assert(self.conv.len > 0);

        @memset(self.context, config.eos_token_id);
        @memset(self.conv, 0.0);
    }
};

/// Per-forward buffers.
pub const Scratch = struct {
    history: []i64,
    shifted: []i64,
    mixed: []i64,
    ngram_ids: []i64,
    embeddings: []f32,
    key_normed: []f32,
    query_normed: []f32,
    value: []f32,
    gate: []f32,
    gated_value: []f32,
    gated_value_normed: []f32,
    conv_full: []f32,
    conv_out: []f32,

    /// Allocates the buffers for a `tokens_max` forward.
    pub fn init(
        allocator: std.mem.Allocator,
        config: *const fixtures.RefConfig,
    ) !Scratch {
        assert(config.ngram_size >= 2);
        assert(config.ple_embed_dim > 0);

        const tokens: usize = config.max_position_embeddings;
        const history: usize = tokens + config.ngram_context_len();
        const heads: usize = config.ngram_heads();
        const hyper: usize = config.hyper_size();
        const conv_len: usize = hyper * (tokens + config.ple_conv_state_len());

        const buffer = try allocator.alloc(i64, history * config.ngram_size);
        errdefer allocator.free(buffer);
        const shifted = try allocator.alloc(i64, history * config.ngram_size);
        errdefer allocator.free(shifted);
        const mixed = try allocator.alloc(i64, history * config.ngram_size);
        errdefer allocator.free(mixed);
        const ngram_ids = try allocator.alloc(i64, tokens * heads);
        errdefer allocator.free(ngram_ids);
        const embeddings = try allocator.alloc(
            f32,
            tokens * config.ple_embed_dim,
        );
        errdefer allocator.free(embeddings);
        const key_normed = try allocator.alloc(f32, tokens * hyper);
        errdefer allocator.free(key_normed);
        const query_normed = try allocator.alloc(f32, tokens * hyper);
        errdefer allocator.free(query_normed);
        const value = try allocator.alloc(f32, tokens * config.hidden_size);
        errdefer allocator.free(value);
        const gate = try allocator.alloc(f32, tokens * config.hc_count);
        errdefer allocator.free(gate);
        const gated_value = try allocator.alloc(f32, tokens * hyper);
        errdefer allocator.free(gated_value);
        const gated_value_normed = try allocator.alloc(f32, tokens * hyper);
        errdefer allocator.free(gated_value_normed);
        const conv_full = try allocator.alloc(f32, conv_len);
        errdefer allocator.free(conv_full);
        const conv_out = try allocator.alloc(f32, hyper * tokens);
        errdefer allocator.free(conv_out);

        return .{
            .history = buffer,
            .shifted = shifted,
            .mixed = mixed,
            .ngram_ids = ngram_ids,
            .embeddings = embeddings,
            .key_normed = key_normed,
            .query_normed = query_normed,
            .value = value,
            .gate = gate,
            .gated_value = gated_value,
            .gated_value_normed = gated_value_normed,
            .conv_full = conv_full,
            .conv_out = conv_out,
        };
    }

    /// Releases the buffers.
    pub fn deinit(self: *Scratch, allocator: std.mem.Allocator) void {
        assert(self.history.len > 0);
        assert(self.conv_out.len > 0);

        allocator.free(self.conv_out);
        allocator.free(self.conv_full);
        allocator.free(self.gated_value_normed);
        allocator.free(self.gated_value);
        allocator.free(self.gate);
        allocator.free(self.value);
        allocator.free(self.query_normed);
        allocator.free(self.key_normed);
        allocator.free(self.embeddings);
        allocator.free(self.ngram_ids);
        allocator.free(self.mixed);
        allocator.free(self.shifted);
        allocator.free(self.history);
    }
};

/// One PLE forward: `gated_value + silu(dilated_conv(norm_conv(gated_value)))`.
pub fn forward(
    hidden: []const f32,
    input_ids: []const i64,
    tokens: u32,
    output: []f32,
    weights: *const Weights,
    state: *State,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(input_ids.len == tokens);
    assert(hidden.len == tokens * config.hyper_size());

    forward_ids(input_ids, tokens, weights, state, scratch, config);
    forward_gather(tokens, weights, scratch, config);
    forward_gates(hidden, tokens, weights, scratch, config);
    forward_conv(tokens, weights, state, scratch, config);
    for (0..tokens * config.hyper_size()) |i| {
        output[i] = scratch.gated_value[i];
    }
    for (0..config.hyper_size()) |c| {
        for (0..tokens) |t| {
            output[t * config.hyper_size() + c] +=
                scratch.conv_out[c * tokens + t];
        }
    }
    assert(output.len == tokens * config.hyper_size());
}

/// Derives the n-gram ids from the (EOS-aware) token history.
fn forward_ids(
    input_ids: []const i64,
    tokens: u32,
    weights: *const Weights,
    state: *State,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(state.context.len == config.ngram_context_len());

    const ngram_size: usize = config.ngram_size;
    const heads: usize = config.ngram_heads();
    const context: usize = config.ngram_context_len();
    const history_len: usize = context + tokens;

    @memcpy(scratch.history[0..context], state.context);
    @memcpy(scratch.history[context..][0..tokens], input_ids);
    @memcpy(
        state.context,
        scratch.history[history_len - context ..][0..context],
    );

    const history = scratch.history[0..history_len];
    for (0..ngram_size) |shift| {
        shift_right_ignore_eos(
            history,
            shift,
            config.eos_token_id,
            scratch.shifted[shift * history_len ..][0..history_len],
        );
    }
    derive_ids(
        history_len,
        tokens,
        weights,
        scratch,
        config,
    );
    assert(scratch.ngram_ids.len >= tokens * heads);
}

/// Hashes the shifted token history into one id per (token, n-gram head).
fn derive_ids(
    history_len: usize,
    tokens: u32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(history_len > tokens);
    assert(weights.multipliers.len == config.ngram_size);

    const heads_per_ngram: usize = config.heads_per_ngram;
    const ngram_size: usize = config.ngram_size;
    const heads: usize = config.ngram_heads();
    const offset_in_token: usize = history_len - tokens;

    for (0..history_len) |position| {
        var mixed: i64 = 0;
        var ngram: usize = 2;
        while (ngram <= ngram_size) : (ngram += 1) {
            mixed = scratch.shifted[position];
            mixed *%= weights.multipliers[0];
            for (1..ngram) |s| {
                const term = scratch.shifted[s * history_len + position];
                mixed ^= term *% weights.multipliers[s];
            }
            const first_head = (ngram - 2) * heads_per_ngram;
            for (0..heads_per_ngram) |k| {
                const head = first_head + k;
                const modulus = weights.head_vocab_sizes[head];
                assert(modulus > 0);
                const id = @mod(mixed, modulus);
                if (position >= offset_in_token) {
                    const t = position - offset_in_token;
                    scratch.ngram_ids[t * heads + head] =
                        id + weights.head_offsets[head];
                } else {
                    // Positions of the context only feed shifts upstream.
                }
            }
        }
    }
    assert(heads == heads_per_ngram * (ngram_size - 1));
}

/// `_shift_right_ignore_eos`: shift `shift` positions, restarting at each EOS.
fn shift_right_ignore_eos(
    token_ids: []const i64,
    shift: usize,
    eos_token_id: i64,
    out: []i64,
) void {
    assert(token_ids.len > 0);
    assert(out.len == token_ids.len);
    assert(eos_token_id >= 0);

    if (shift == 0) {
        @memcpy(out, token_ids);
        return;
    } else {
        assert(shift > 0);
    }
    var previous_eos: i64 = -1;
    for (token_ids, 0..) |token, position| {
        const position_i64: i64 = @intCast(position);
        const source = position_i64 - @as(i64, @intCast(shift));
        const segment_start = previous_eos + 1;
        const in_segment = position_i64 - segment_start;
        if (source >= 0) {
            if (in_segment >= @as(i64, @intCast(shift))) {
                out[position] = token_ids[@intCast(source)];
            } else {
                out[position] = eos_token_id;
            }
        } else {
            out[position] = eos_token_id;
        }
        if (token == eos_token_id) {
            previous_eos = position_i64;
        } else {
            assert(token != eos_token_id);
        }
    }
    assert(out.len == token_ids.len);
}

/// Gathers one table row per (token, head) into the flat embedding vector.
fn forward_gather(
    tokens: u32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(weights.shard_count > 0);

    const heads: usize = config.ngram_heads();
    const head_dim: usize = config.ngram_head_dim();
    assert(weights.head_dim == head_dim);
    const rows_per_shard = weights.shards[0].len / head_dim;
    assert(rows_per_shard > 0);
    for (0..tokens * heads) |row| {
        const id = scratch.ngram_ids[row];
        assert(id >= 0);
        const index: u64 = @intCast(id);
        const shard: usize = @intCast(@divFloor(index, rows_per_shard));
        const offset: usize = @intCast(@mod(index, rows_per_shard));
        assert(shard < weights.shard_count);
        const source = weights.shards[shard][offset * head_dim ..][0..head_dim];
        @memcpy(scratch.embeddings[row * head_dim ..][0..head_dim], source);
    }
    assert(scratch.embeddings.len >= tokens * config.ple_embed_dim);
}

/// `key_proj`/`value_proj`, the per-stream signed-sqrt gate and the gated
// value.
fn forward_gates(
    hidden: []const f32,
    tokens: u32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(hidden.len == tokens * config.hyper_size());

    const hyper: usize = config.hyper_size();
    const hidden_size: usize = config.hidden_size;
    ops.matmul(
        scratch.embeddings[0 .. tokens * config.ple_embed_dim],
        weights.key_proj,
        scratch.key_normed[0 .. tokens * hyper],
        .{
            .rows = tokens,
            .in_features = config.ple_embed_dim,
            .out_features = config.hyper_size(),
        },
    );
    ops.matmul(
        scratch.embeddings[0 .. tokens * config.ple_embed_dim],
        weights.value_proj,
        scratch.value[0 .. tokens * hidden_size],
        .{
            .rows = tokens,
            .in_features = config.ple_embed_dim,
            .out_features = config.hidden_size,
        },
    );
    forward_gate(hidden, tokens, weights, scratch, config);
    assert(scratch.gated_value.len >= tokens * hyper);
}

/// `norm_key(norm_query(hidden))` per stream, then the signed-sqrt gate.
fn forward_gate(
    hidden: []const f32,
    tokens: u32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(hidden.len == tokens * config.hyper_size());

    const hyper: usize = config.hyper_size();
    const hidden_size: usize = config.hidden_size;
    const streams: usize = config.hc_count;
    const divisor = @sqrt(@as(f32, @floatFromInt(hidden_size)));
    for (0..tokens) |t| {
        const key = scratch.key_normed[t * hyper ..][0..hyper];
        const query = scratch.query_normed[t * hyper ..][0..hyper];
        const normed_query = hidden[t * hyper ..][0..hyper];
        ops.rmsnorm(
            key,
            weights.norm_key,
            key,
            config.hidden_size,
            config.rms_norm_eps,
        );
        ops.rmsnorm(
            normed_query,
            weights.norm_query,
            query,
            config.hidden_size,
            config.rms_norm_eps,
        );
        for (0..streams) |c| {
            var total: f32 = 0.0;
            for (0..hidden_size) |h| {
                total += key[c * hidden_size + h] * query[c * hidden_size + h];
            }
            const scaled = total / divisor;
            const magnitude = @max(@abs(scaled), 1e-6);
            const signed: f32 = if (scaled < 0.0) -1.0 else 1.0;
            scratch.gate[t * streams + c] = signed * @sqrt(magnitude);
        }
        for (0..streams) |c| {
            const weight = ops.sigmoid(scratch.gate[t * streams + c]);
            for (0..hidden_size) |h| {
                scratch.gated_value[t * hyper + c * hidden_size + h] =
                    weight * scratch.value[t * hidden_size + h];
            }
        }
    }
    assert(scratch.gated_value.len >= tokens * hyper);
}

/// `norm_conv` over the flattened gated value, then the dilated depthwise conv.
fn forward_conv(
    tokens: u32,
    weights: *const Weights,
    state: *State,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(weights.conv.len ==
        config.hyper_size() * config.ple_conv_kernel_size);

    const hyper: usize = config.hyper_size();
    const history: usize = config.ple_conv_state_len();
    const full_len: usize = tokens + history;
    for (0..tokens) |t| {
        ops.rmsnorm(
            scratch.gated_value[t * hyper ..][0..hyper],
            weights.norm_conv,
            scratch.gated_value_normed[t * hyper ..][0..hyper],
            config.hidden_size,
            config.rms_norm_eps,
        );
    }
    const full = scratch.conv_full[0 .. hyper * full_len];
    for (0..hyper) |channel| {
        const target = full[channel * full_len ..][0..full_len];
        @memcpy(
            target[0..history],
            state.conv[channel * history ..][0..history],
        );
        for (0..tokens) |t| {
            target[history + t] =
                scratch.gated_value_normed[t * hyper + channel];
        }
    }
    conv.causal_conv1d(
        full,
        weights.conv,
        scratch.conv_out[0 .. hyper * tokens],
        .{
            .channels = config.hyper_size(),
            .kernel = config.ple_conv_kernel_size,
            .dilation = config.ngram_size,
            .full_len = @intCast(full_len),
        },
        .silu,
    );
    conv.store_state(
        state.conv,
        full,
        config.hyper_size(),
        @intCast(full_len),
        @intCast(history),
    );
    assert(state.conv.len == hyper * history);
}

/// `_splitmix64` from the reference, on wrapping u64.
pub fn splitmix64(value: u64) u64 {
    assert(value <= std.math.maxInt(u64));
    assert(splitmix_gamma > 0);

    var mixed = value +% splitmix_gamma;
    mixed = (mixed ^ (mixed >> 30)) *% splitmix_m1;
    mixed = (mixed ^ (mixed >> 27)) *% splitmix_m2;
    return mixed ^ (mixed >> 31);
}

/// `_build_layer_multipliers`: one odd multiplier per n-gram order.
pub fn build_layer_multipliers(
    unigram_vocab_size: u64,
    ngram_size: u32,
    ple_layer_index: u32,
    seed: u64,
    out: []i64,
) void {
    assert(ngram_size >= 2);
    assert(out.len >= ngram_size);

    const max_long: u64 = (1 << 63) - 1;
    const vocabulary = @max(unigram_vocab_size, 1);
    const multiplier_max = max_long / vocabulary;
    const half_bound = @max(multiplier_max / 2, 1);
    const base_seed = seed +% prime_seed_stride *% ple_layer_index;
    for (0..ngram_size) |index| {
        const value = base_seed +% splitmix_gamma *% (index + 1);
        out[index] = @intCast(2 * (splitmix64(value) % half_bound) + 1);
    }
    assert(half_bound >= 1);
}

/// `_is_prime`, by trial division up to the integer square root.
pub fn is_prime(value: u64) bool {
    assert(value <= 1 << 40);
    assert(value > 0);

    if (value < 2) {
        return false;
    } else {
        assert(value >= 2);
    }
    if (value % 2 == 0) {
        return value == 2;
    } else {
        assert(value % 2 != 0);
    }
    var divisor: u64 = 3;
    while (divisor * divisor <= value) : (divisor += 2) {
        if (value % divisor == 0) {
            return false;
        } else {
            continue;
        }
    }
    return true;
}

/// `_find_nth_prime_after`: the `count`-th prime strictly after `start`.
pub fn find_nth_prime_after(start: u64, count: u32) u64 {
    assert(count >= 1);
    assert(start <= 1 << 32);

    var prime = start;
    for (0..count) |_| {
        prime += 1;
        while (!is_prime(prime)) {
            prime += 1;
        }
    }
    assert(is_prime(prime));
    return prime;
}

test "the derived n-gram head sizes match the checkpoint buffers" {
    const allocator = std.testing.allocator;
    const config = try fixtures.load_config(allocator);
    var tensors = try fixtures.open_checkpoint(allocator);
    defer tensors.close(allocator);

    const prefix = "model.language_model.layers.1.ple.ple_embedding.";
    const sizes = try tensors.tensor_i64(
        prefix ++ "ngram_heads_vocab_sizes",
    );
    const offsets = try tensors.tensor_i64(
        prefix ++ "ngram_heads_offsets",
    );
    var total: u64 = 0;
    for (0..config.ngram_heads()) |head| {
        const global_head = head;
        const size = find_nth_prime_after(
            config.ngram_vocab_size_base - 1,
            @intCast(global_head + 1),
        );
        try std.testing.expectEqual(@as(i64, @intCast(size)), sizes[head]);
        try std.testing.expectEqual(@as(i64, @intCast(total)), offsets[head]);
        total += size;
    }
    // 1000-based primes: 1009, 1013, 1019, 1021 -> 4062 padded up to a
    // multiple of 8 for the sharded table.
    try std.testing.expectEqual(@as(u64, 4062), total);
}

test "the derived layer multipliers match the checkpoint buffer" {
    const allocator = std.testing.allocator;
    const config = try fixtures.load_config(allocator);
    var tensors = try fixtures.open_checkpoint(allocator);
    defer tensors.close(allocator);

    const stored = try tensors.tensor_i64(
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
    );
    var derived: [8]i64 = undefined;
    build_layer_multipliers(
        config.vocab_size,
        config.ngram_size,
        config.ple_layer_index(1).?,
        config.seed,
        &derived,
    );
    try std.testing.expectEqual(@as(usize, 3), stored.len);
    for (stored, 0..) |value, i| {
        try std.testing.expectEqual(value, derived[i]);
    }
}

test "shift_right_ignore_eos restarts the window at an EOS" {
    var out: [5]i64 = undefined;
    shift_right_ignore_eos(&[5]i64{ 7, 255, 9, 10, 11 }, 1, 255, &out);
    try std.testing.expectEqual(@as(i64, 255), out[0]);
    // Position 1 still sees position 0; the EOS at position 1 only closes the
    // segment that starts at position 2.
    try std.testing.expectEqual(@as(i64, 7), out[1]);
    try std.testing.expectEqual(@as(i64, 255), out[2]);
    try std.testing.expectEqual(@as(i64, 9), out[3]);
    try std.testing.expectEqual(@as(i64, 10), out[4]);
}

test "shift zero is the identity" {
    var out: [3]i64 = undefined;
    shift_right_ignore_eos(&[3]i64{ 1, 2, 3 }, 0, 255, &out);
    try std.testing.expectEqual([3]i64{ 1, 2, 3 }, out);
}

test "is_prime and find_nth_prime_after agree with the reference" {
    try std.testing.expectEqual(@as(u64, 1009), find_nth_prime_after(1000, 1));
    try std.testing.expectEqual(@as(u64, 1013), find_nth_prime_after(1000, 2));
    try std.testing.expectEqual(@as(u64, 1019), find_nth_prime_after(1000, 3));
    try std.testing.expect(!is_prime(1));
    try std.testing.expect(is_prime(2));
    try std.testing.expect(!is_prime(9));
}
