//! The QSA indexer selection and the gated GQA attention that consumes it.
//!
//! Reference lines 688-916. The indexer projections are computed first: it
//! pools the per-token index keys into `compress_ratio`-sized blocks of the
//! *visible* prefix (the trailing incomplete block is appended raw), scores
// each
//! block with `relu(q . k).sum() / sqrt(head_dim)` over its heads, and keeps
// the
//! top `budget / compress_ratio` blocks. That turns into a token-level boolean
//! mask which narrows the causal mask of the attention that follows.
//!
//! The query projection splits into `(query, gate)` per head; the gate is
//! `sigmoid` applied after the attention (before `o_proj`).

const std = @import("std");

const fixtures = @import("fixtures.zig");
const ops = @import("ops.zig");
const rope = @import("rope.zig");

const assert = std.debug.assert;

/// Weights of one indexed-attention layer.
pub const Weights = struct {
    /// `q_proj.weight`, `[heads * head_dim * 2, hidden]`; the second half of
    /// each head block is the output gate.
    q_proj: []const f32,
    /// `k_proj.weight`, `[kv_heads * head_dim, hidden]`.
    k_proj: []const f32,
    /// `v_proj.weight`, `[kv_heads * head_dim, hidden]`.
    v_proj: []const f32,
    /// `o_proj.weight`, `[hidden, heads * head_dim]`.
    o_proj: []const f32,
    /// `q_norm.weight`, `[head_dim]`.
    q_norm: []const f32,
    /// `k_norm.weight`, `[head_dim]`.
    k_norm: []const f32,
    /// `indexer.index_qk_proj.weight`, `[(n_heads + kv_heads) * dim, hidden]`.
    index_qk_proj: []const f32,
    /// `indexer.q_layernorm.weight`, `[index_dim]`.
    q_layernorm: []const f32,
    /// `indexer.k_layernorm.weight`, `[index_dim]`.
    k_layernorm: []const f32,
};

/// KV and indexer-key caches of one indexed-attention layer.
pub const State = struct {
    /// `[cache_len, kv_heads * head_dim]`, post-RoPE keys.
    key: []f32,
    /// `[cache_len, kv_heads * head_dim]`.
    value: []f32,
    /// `[cache_len, index_dim]`, the indexer's per-token keys.
    index_key: []f32,
    /// Number of tokens written so far.
    length: u32,

    /// Allocates an empty cache with room for `cache_len` tokens.
    pub fn init(
        allocator: std.mem.Allocator,
        config: *const fixtures.RefConfig,
        cache_len: u32,
    ) !State {
        assert(cache_len > 0);
        assert(config.indexer_head_dim > 0);

        const kv_dim: usize = config.num_key_value_heads * config.head_dim;
        const index_dim: usize = config.indexer_head_dim;
        const tokens: usize = cache_len;

        const key = try allocator.alloc(f32, tokens * kv_dim);
        errdefer allocator.free(key);
        const value = try allocator.alloc(f32, tokens * kv_dim);
        errdefer allocator.free(value);
        const index_key = try allocator.alloc(f32, tokens * index_dim);
        errdefer allocator.free(index_key);
        @memset(key, 0.0);
        @memset(value, 0.0);
        @memset(index_key, 0.0);

        return .{
            .key = key,
            .value = value,
            .index_key = index_key,
            .length = 0,
        };
    }

    /// Releases the caches.
    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        assert(self.key.len > 0);
        assert(self.index_key.len > 0);

        allocator.free(self.index_key);
        allocator.free(self.value);
        allocator.free(self.key);
        self.length = 0;
    }

    /// Forgets the cached tokens.
    pub fn reset(self: *State) void {
        assert(self.key.len > 0);
        assert(self.index_key.len > 0);

        self.length = 0;
    }
};

/// Per-forward buffers.
pub const Scratch = struct {
    qkv: []f32,
    query: []f32,
    key: []f32,
    value: []f32,
    gate: []f32,
    attn_out: []f32,
    output: []f32,
    row: []f32,
    index_qk: []f32,
    index_query: []f32,
    pooled: []f32,
    block_scores: []f32,
    block_order: []u32,
    selected: []i32,
    mask: []bool,

    /// Allocates the buffers for a `tokens_max` forward and `cache_len` cache.
    pub fn init(
        allocator: std.mem.Allocator,
        config: *const fixtures.RefConfig,
        cache_len: u32,
    ) !Scratch {
        assert(config.head_dim > 0);
        assert(cache_len > 0);

        const tokens: usize = config.max_position_embeddings;
        const cache: usize = cache_len;
        const heads: usize = config.num_attention_heads;
        const kv_heads: usize = config.num_key_value_heads;
        const head_dim: usize = config.head_dim;
        const index_heads: usize = config.indexer_n_heads +
            config.indexer_kv_heads;
        const index_dim: usize = config.indexer_head_dim;
        const blocks = cache / config.indexer_compress_ratio + 1;

        const qkv = try allocator.alloc(f32, tokens * heads * head_dim * 2);
        errdefer allocator.free(qkv);
        const query = try allocator.alloc(f32, tokens * heads * head_dim);
        errdefer allocator.free(query);
        const key = try allocator.alloc(f32, tokens * kv_heads * head_dim);
        errdefer allocator.free(key);
        const value = try allocator.alloc(f32, tokens * kv_heads * head_dim);
        errdefer allocator.free(value);
        const gate = try allocator.alloc(f32, tokens * heads * head_dim);
        errdefer allocator.free(gate);
        const attn_out = try allocator.alloc(f32, tokens * heads * head_dim);
        errdefer allocator.free(attn_out);
        const output = try allocator.alloc(f32, tokens * config.hidden_size);
        errdefer allocator.free(output);
        const row = try allocator.alloc(f32, cache);
        errdefer allocator.free(row);
        const index_qk = try allocator.alloc(
            f32,
            tokens * index_heads * index_dim,
        );
        errdefer allocator.free(index_qk);
        const index_query = try allocator.alloc(
            f32,
            tokens * config.indexer_n_heads * index_dim,
        );
        errdefer allocator.free(index_query);
        const pooled = try allocator.alloc(f32, blocks * index_dim);
        errdefer allocator.free(pooled);
        const block_scores = try allocator.alloc(f32, blocks);
        errdefer allocator.free(block_scores);
        const block_order = try allocator.alloc(u32, blocks);
        errdefer allocator.free(block_order);
        const selected = try allocator.alloc(
            i32,
            tokens * config.indexer_selected_slots(),
        );
        errdefer allocator.free(selected);
        const mask = try allocator.alloc(bool, tokens * cache);
        errdefer allocator.free(mask);

        return .{
            .qkv = qkv,
            .query = query,
            .key = key,
            .value = value,
            .gate = gate,
            .attn_out = attn_out,
            .output = output,
            .row = row,
            .index_qk = index_qk,
            .index_query = index_query,
            .pooled = pooled,
            .block_scores = block_scores,
            .block_order = block_order,
            .selected = selected,
            .mask = mask,
        };
    }

    /// Releases the buffers.
    pub fn deinit(self: *Scratch, allocator: std.mem.Allocator) void {
        assert(self.qkv.len > 0);
        assert(self.mask.len > 0);

        allocator.free(self.mask);
        allocator.free(self.selected);
        allocator.free(self.block_order);
        allocator.free(self.block_scores);
        allocator.free(self.pooled);
        allocator.free(self.index_query);
        allocator.free(self.index_qk);
        allocator.free(self.output);
        allocator.free(self.row);
        allocator.free(self.attn_out);
        allocator.free(self.gate);
        allocator.free(self.value);
        allocator.free(self.key);
        allocator.free(self.query);
        allocator.free(self.qkv);
    }
};

/// Storage dtype of the pooled block keys.
///
/// The model pools in f32 and casts the result back to the cache dtype
/// (`key_groups.float().mean(dim=1).to(raw_keys.dtype)`) before the norm and the
/// rotation, so on a bf16 cache the pooled keys a query is scored against are
/// bf16. The graph itself stays f32 throughout; this only names the cast.
pub const PooledDtype = enum {
    /// f32 pooled keys, what `forward` computes.
    f32,
    /// The cache dtype: round the pooled key to bf16 at the cast points.
    cache_dtype,
};

/// RoPE tables for the positions this forward covers.
pub const Rope = struct {
    /// `[positions, rotary_dim]`, positions `0..positions`.
    cos: []const f32,
    sin: []const f32,
};

/// One layer forward: indexer mask, then gated GQA over the caches.
pub fn forward(
    input: []const f32,
    tokens: u32,
    positions: []const u32,
    output: []f32,
    weights: *const Weights,
    state: *State,
    scratch: *Scratch,
    table: Rope,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(positions.len == tokens);
    assert(input.len == tokens * config.hidden_size);

    forward_indexer(
        input,
        tokens,
        positions,
        weights,
        state,
        scratch,
        table,
        config,
    );
    forward_projections(
        input,
        tokens,
        positions,
        weights,
        state,
        scratch,
        table,
        config,
    );
    forward_attend(tokens, positions, state, scratch, config);
    forward_output(tokens, weights, scratch, config);
    @memcpy(output, scratch.attn_out[0 .. tokens * config.hidden_size]);
    assert(output.len == tokens * config.hidden_size);
}

/// Runs the indexer: fills `scratch.selected` and `scratch.mask`.
pub fn forward_indexer(
    input: []const f32,
    tokens: u32,
    positions: []const u32,
    weights: *const Weights,
    state: *State,
    scratch: *Scratch,
    table: Rope,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(state.length + tokens <=
        state.index_key.len / config.indexer_head_dim);

    const queries: usize = config.indexer_n_heads;
    const index_dim: usize = config.indexer_head_dim;
    const kv_heads: usize = config.indexer_kv_heads;
    const projected = queries + kv_heads;
    const rotary: usize = config.rotary_dim();

    ops.matmul(
        input,
        weights.index_qk_proj,
        scratch.index_qk[0 .. tokens * projected * index_dim],
        .{
            .rows = tokens,
            .in_features = config.hidden_size,
            .out_features = @intCast(projected * index_dim),
        },
    );
    // The index keys are cached raw (un-normalised) and normalised on read.
    for (0..tokens) |t| {
        const source_start = (t * projected + queries) * index_dim;
        const source = scratch.index_qk[source_start..][0..index_dim];
        const target_start = (state.length + t) * index_dim;
        const target = state.index_key[target_start..][0..index_dim];
        @memcpy(target, source);
    }
    state.length += tokens;
    for (0..tokens) |t| {
        const position: usize = positions[t];
        const cos = table.cos[position * rotary ..][0..rotary];
        const sin = table.sin[position * rotary ..][0..rotary];
        for (0..queries) |head| {
            const source_start = (t * projected + head) * index_dim;
            const source = scratch.index_qk[source_start..][0..index_dim];
            const normed = scratch.row[0..index_dim];
            ops.rmsnorm(
                source,
                weights.q_layernorm,
                normed,
                @intCast(index_dim),
                config.rms_norm_eps,
            );
            const target_start = (t * queries + head) * index_dim;
            const target = scratch.index_query[target_start..][0..index_dim];
            rope.apply(normed, cos, sin, target);
        }
    }
    forward_select(
        tokens,
        positions,
        weights,
        state,
        scratch,
        table,
        config,
        .f32,
    );
    assert(state.length > 0);
}

/// Block pooling, block scoring and the token-level selection mask.
///
/// `pooled_dtype` selects the storage the pooled keys are scored in.
pub fn forward_select(
    tokens: u32,
    positions: []const u32,
    weights: *const Weights,
    state: *const State,
    scratch: *Scratch,
    table: Rope,
    config: *const fixtures.RefConfig,
    pooled_dtype: PooledDtype,
) void {
    assert(tokens > 0);
    assert(state.length == positions[tokens - 1] + 1);

    const ratio: usize = config.indexer_compress_ratio;
    const slots: usize = config.indexer_selected_slots();
    const kv_length: usize = state.length;

    @memset(scratch.mask[0 .. tokens * kv_length], false);
    for (0..tokens) |t| {
        const visible: usize = positions[t] + 1;
        const blocks: usize = visible / ratio;
        const selected = scratch.selected[t * slots ..][0..slots];
        @memset(selected, -1);
        const count = select_blocks(
            t,
            blocks,
            config.indexer_block_topk(),
            weights,
            state,
            scratch,
            table,
            config,
            pooled_dtype,
        );
        // The incomplete trailing block is appended without pooling.
        const tail = visible - blocks * ratio;
        assert(tail < ratio);
        var written: usize = 0;
        for (0..count) |slot| {
            const block = scratch.block_order[slot];
            for (0..ratio) |j| {
                const token = block * ratio + j;
                scratch.mask[t * kv_length + token] = true;
                selected[written] = @intCast(token);
                written += 1;
            }
        }
        for (0..tail) |j| {
            const token = blocks * ratio + j;
            scratch.mask[t * kv_length + token] = true;
            selected[written] = @intCast(token);
            written += 1;
        }
        assert(written <= slots);
    }
    assert(scratch.mask.len >= tokens * kv_length);
}

/// Casts one pooled key component to the storage it is held in.
fn castPooled(value: f32, dtype: PooledDtype) f32 {
    assert(@sizeOf(f32) == 4);
    assert(@sizeOf(u16) == 2);

    switch (dtype) {
        .f32 => return value,
        .cache_dtype => return ops.round_bf16(value),
    }
}

/// Scores the visible blocks and writes the top-k block order.
///
/// The score is `sum_heads relu(q_h . pooled_block) / sqrt(index_dim)`, taken
// on
/// the *normalised and rotated* pooled block keys.
fn select_blocks(
    token: usize,
    blocks: usize,
    top: usize,
    weights: *const Weights,
    state: *const State,
    scratch: *Scratch,
    table: Rope,
    config: *const fixtures.RefConfig,
    pooled_dtype: PooledDtype,
) u32 {
    assert(token < config.max_position_embeddings);
    assert(blocks * config.indexer_head_dim <= scratch.pooled.len);

    if (blocks == 0) {
        return 0;
    } else {
        assert(blocks > 0);
    }
    const index_dim: usize = config.indexer_head_dim;
    const ratio: usize = config.indexer_compress_ratio;
    const queries: usize = config.indexer_n_heads;
    const rotary: usize = config.rotary_dim();
    for (0..blocks) |block| {
        const pooled = scratch.pooled[block * index_dim ..][0..index_dim];
        @memset(pooled, 0.0);
        for (0..ratio) |j| {
            const source_start = (block * ratio + j) * index_dim;
            const source = state.index_key[source_start..][0..index_dim];
            for (source, 0..) |value, d| {
                pooled[d] += value;
            }
        }
        for (pooled) |*value| {
            value.* = castPooled(value.* / @as(f32, @floatFromInt(ratio)), pooled_dtype);
        }
        ops.rmsnorm(
            pooled,
            weights.k_layernorm,
            pooled,
            @intCast(index_dim),
            config.rms_norm_eps,
        );
        // The block key rotates at the position of the block's first token.
        const start: usize = block * ratio;
        const cos = table.cos[start * rotary ..][0..rotary];
        const sin = table.sin[start * rotary ..][0..rotary];
        rope.apply(pooled, cos, sin, pooled);
        for (pooled) |*value| {
            value.* = castPooled(value.*, pooled_dtype);
        }
    }
    for (0..blocks) |block| {
        scratch.block_scores[block] = block_score(
            token,
            block,
            queries,
            index_dim,
            scratch,
        );
    }
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(index_dim)));
    for (0..blocks) |block| {
        scratch.block_scores[block] *= scale;
    }
    const count = @min(blocks, top);
    top_blocks(scratch.block_scores[0..blocks], count, scratch.block_order);
    assert(count <= blocks);
    return @intCast(count);
}

/// Sum over heads of `relu(q_h . pooled_block)`, before the `1/sqrt(dim)`
// scale.
fn block_score(
    token: usize,
    block: usize,
    queries: usize,
    index_dim: usize,
    scratch: *Scratch,
) f32 {
    assert(queries > 0);
    assert((block + 1) * index_dim <= scratch.pooled.len);

    const pooled = scratch.pooled[block * index_dim ..][0..index_dim];
    var total: f32 = 0.0;
    for (0..queries) |head| {
        const query = scratch.index_query[(token * queries + head) *
            index_dim ..][0..index_dim];
        const dot = ops.dot(query, pooled);
        total += if (dot > 0.0) dot else 0.0;
    }
    assert(total >= 0.0);
    return total;
}

/// Writes the indices of the `count` largest scores, descending.
///
/// Ties go to the *later* block. `torch.topk` does not define an order for
/// equal scores; the golden trace shows it keeping the later block in every tie
/// this fixture produces (score 0.0 is the `relu` floor, so ties are common),
/// and this reproduces that.
fn top_blocks(scores: []const f32, count: usize, order: []u32) void {
    assert(scores.len > 0);
    assert(count <= scores.len);
    assert(order.len >= count);

    var taken: [4096]bool = undefined;
    assert(scores.len <= taken.len);
    for (0..scores.len) |i| {
        taken[i] = false;
    }
    for (0..count) |slot| {
        var best: usize = 0;
        var best_score = -std.math.inf(f32);
        for (scores, 0..) |score, index| {
            if (taken[index]) {
                continue;
            } else {
                if (score >= best_score) {
                    best_score = score;
                    best = index;
                } else {
                    continue;
                }
            }
        }
        taken[best] = true;
        order[slot] = @intCast(best);
    }
    assert(count <= scores.len);
}

/// Query/key/value projections, per-head norms, RoPE and the cache append.
fn forward_projections(
    input: []const f32,
    tokens: u32,
    positions: []const u32,
    weights: *const Weights,
    state: *State,
    scratch: *Scratch,
    table: Rope,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(positions.len == tokens);

    const heads: usize = config.num_attention_heads;
    const head_dim: usize = config.head_dim;
    const kv_dim: usize = config.num_key_value_heads * config.head_dim;

    ops.matmul(
        input,
        weights.q_proj,
        scratch.qkv[0 .. tokens * heads * head_dim * 2],
        .{
            .rows = tokens,
            .in_features = config.hidden_size,
            .out_features = @intCast(heads * head_dim * 2),
        },
    );
    ops.matmul(
        input,
        weights.k_proj,
        scratch.key[0 .. tokens * kv_dim],
        .{
            .rows = tokens,
            .in_features = config.hidden_size,
            .out_features = config.num_key_value_heads * config.head_dim,
        },
    );
    ops.matmul(
        input,
        weights.v_proj,
        scratch.value[0 .. tokens * kv_dim],
        .{
            .rows = tokens,
            .in_features = config.hidden_size,
            .out_features = config.num_key_value_heads * config.head_dim,
        },
    );
    forward_heads(tokens, positions, weights, state, scratch, table, config);
    assert(state.length > 0);
}

/// Per-head norms, RoPE and the KV cache append.
fn forward_heads(
    tokens: u32,
    positions: []const u32,
    weights: *const Weights,
    state: *State,
    scratch: *Scratch,
    table: Rope,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(positions.len == tokens);

    const heads: usize = config.num_attention_heads;
    const kv_heads: usize = config.num_key_value_heads;
    const kv_dim: usize = kv_heads * config.head_dim;
    const rotary: usize = config.rotary_dim();
    // The indexer already appended its own keys, so `state.length` counts this
    // forward's tokens too; the KV write starts at `past`.
    const past: usize = state.length - tokens;

    for (0..tokens) |t| {
        const position: usize = positions[t];
        const cos = table.cos[position * rotary ..][0..rotary];
        const sin = table.sin[position * rotary ..][0..rotary];
        for (0..heads) |head| {
            forward_query_head(t, head, cos, sin, weights, scratch, config);
        }
        for (0..kv_heads) |head| {
            forward_key_head(t, head, past, cos, sin, weights, state, scratch, config);
        }
        for (0..kv_dim) |d| {
            state.value[(past + t) * kv_dim + d] =
                scratch.value[t * kv_dim + d];
        }
    }
    assert(state.length > 0);
}

/// One query head: split off the gate, normalise, rotate.
fn forward_query_head(
    token: usize,
    head: usize,
    cos: []const f32,
    sin: []const f32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(head < config.num_attention_heads);
    assert(cos.len == config.rotary_dim());

    const head_dim: usize = config.head_dim;
    const heads: usize = config.num_attention_heads;
    const base = (token * heads + head) * head_dim * 2;
    const query_source = scratch.qkv[base..][0..head_dim];
    const gate_source = scratch.qkv[base + head_dim ..][0..head_dim];
    const normed = scratch.row[0..head_dim];
    const target_start = (token * heads + head) * head_dim;
    const target = scratch.query[target_start..][0..head_dim];
    ops.rmsnorm(
        query_source,
        weights.q_norm,
        normed,
        @intCast(head_dim),
        config.rms_norm_eps,
    );
    rope.apply(normed, cos, sin, target);
    for (0..head_dim) |d| {
        scratch.gate[target_start + d] = gate_source[d];
    }
}

/// One key head: normalise, rotate, and write into the KV cache.
fn forward_key_head(
    token: usize,
    head: usize,
    past: usize,
    cos: []const f32,
    sin: []const f32,
    weights: *const Weights,
    state: *State,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(head < config.num_key_value_heads);
    assert(cos.len == config.rotary_dim());

    const head_dim: usize = config.head_dim;
    const kv_heads: usize = config.num_key_value_heads;
    const kv_dim: usize = kv_heads * head_dim;
    const source_start = (token * kv_heads + head) * head_dim;
    const source = scratch.key[source_start..][0..head_dim];
    const normed = scratch.row[0..head_dim];
    const target_start = (past + token) * kv_dim + head * head_dim;
    const target = state.key[target_start..][0..head_dim];
    ops.rmsnorm(
        source,
        weights.k_norm,
        normed,
        @intCast(head_dim),
        config.rms_norm_eps,
    );
    rope.apply(normed, cos, sin, target);
}

/// Causal, indexer-masked GQA.
fn forward_attend(
    tokens: u32,
    positions: []const u32,
    state: *const State,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(state.length > 0);

    const heads: usize = config.num_attention_heads;
    const head_dim: usize = config.head_dim;
    const kv_dim: usize = config.num_key_value_heads * config.head_dim;
    const groups: usize = config.num_key_value_groups();
    const kv_length: usize = state.length;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    for (0..tokens) |t| {
        const visible: usize = positions[t] + 1;
        for (0..heads) |head| {
            const source_key = head / groups;
            const query_start = (t * heads + head) * head_dim;
            const query = scratch.query[query_start..][0..head_dim];
            for (0..kv_length) |kv| {
                const allowed = scratch.mask[t * kv_length + kv];
                const offset = kv * kv_dim + source_key * head_dim;
                if (allowed) {
                    const key = state.key[offset..][0..head_dim];
                    scratch.row[kv] = ops.dot(query, key) * scale;
                } else {
                    scratch.row[kv] = -std.math.inf(f32);
                }
            }
            for (visible..kv_length) |kv| {
                scratch.row[kv] = -std.math.inf(f32);
            }
            ops.softmax(scratch.row[0..kv_length]);
            const target_start = (t * heads + head) * head_dim;
            const target = scratch.attn_out[target_start..][0..head_dim];
            @memset(target, 0.0);
            for (0..kv_length) |kv| {
                const weight = scratch.row[kv];
                const offset = kv * kv_dim + source_key * head_dim;
                const value = state.value[offset..][0..head_dim];
                for (value, 0..) |component, d| {
                    target[d] += weight * component;
                }
            }
        }
    }
    assert(state.length == kv_length);
}

/// `attn_out * sigmoid(gate)`, then `o_proj`.
fn forward_output(
    tokens: u32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(weights.o_proj.len == config.hidden_size *
        config.num_attention_heads * config.head_dim);

    const width: usize = config.num_attention_heads * config.head_dim;
    for (0..tokens * width) |i| {
        scratch.attn_out[i] *= ops.sigmoid(scratch.gate[i]);
    }
    ops.matmul(
        scratch.attn_out[0 .. tokens * width],
        weights.o_proj,
        scratch.output[0 .. tokens * config.hidden_size],
        .{
            .rows = tokens,
            .in_features = @intCast(width),
            .out_features = config.hidden_size,
        },
    );
    @memcpy(
        scratch.attn_out[0 .. tokens * config.hidden_size],
        scratch.output[0 .. tokens * config.hidden_size],
    );
    assert(scratch.output.len >= tokens * config.hidden_size);
}
