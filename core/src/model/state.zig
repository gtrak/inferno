//! The caches and buffers one sequence's forward runs on.
//!
//! Everything is allocated once, at init, and reused by every forward: the
//! per-forward activation buffers (they are scratch, overwritten each time) and
//! the per-layer caches (KV, indexer keys, GDN conv ring and recurrent state),
//! which survive across decode steps until `reset`. That is the tiger-style
//! "no allocation after init" rule applied to the graph, and it is also why the
//! sizes here are derived from the config rather than measured per forward.

const std = @import("std");

const attention_kernels = @import("../kernels/attention.zig");
const cuda_context = @import("../hw/cuda_context.zig");
const cuda_driver = @import("../hw/cuda_driver.zig");
const fixtures = @import("../ref/fixtures.zig");
const plan_module = @import("plan.zig");

const assert = std.debug.assert;

/// Tokens one forward may cover: a prefill of the tiny prompt plus fourteen
/// decode steps fits in 32; 64 leaves room for the full-recompute variant with
/// margin, and the arena still bounds the total sequence length.
pub const tokens_max: u32 = 64;

/// Bytes of pinned host staging every upload and download goes through.
const staging_bytes: u64 = 1 << 20;

/// Rounds a byte count up to a whole device allocation.
pub fn allocationBytes(bytes: u64) u64 {
    assert(bytes > 0);
    assert(@alignOf(u64) == 8);

    return (bytes + 4095) / 4096 * 4096;
}

/// One device buffer of at least `bytes`, rounded to the allocation size.
fn allocate(bytes: u64) !cuda_context.DeviceBuffer {
    assert(bytes > 0);

    const buffer = try cuda_context.DeviceBuffer.allocate(allocationBytes(bytes));
    assert(buffer.pointer != 0);
    return buffer;
}

/// One attention layer's caches.
pub const AttentionCache = struct {
    /// bf16 `[arena, kv_heads * head_dim]`, post-RoPE keys.
    key: cuda_context.DeviceBuffer,
    /// bf16 `[arena, kv_heads * head_dim]`.
    value: cuda_context.DeviceBuffer,
    /// bf16 `[arena, index_dim]`: the indexer's per-token keys.
    index_keys: cuda_context.DeviceBuffer,
    /// bf16 `[arena / compress_ratio, index_dim]`: pooled block keys.
    index_pooled: cuda_context.DeviceBuffer,
};

/// One Gated DeltaNet layer's caches.
pub const GdnCache = struct {
    /// f32 `[conv_dim, conv_kernel - 1]`, the causal conv's history ring.
    conv: cuda_context.DeviceBuffer,
    /// f32 `[value_heads, key_head_dim, value_head_dim]`.
    recurrent: cuda_context.DeviceBuffer,
};

/// One layer's persistent state; only the slot of its kind is filled.
pub const LayerCache = struct {
    attention: ?AttentionCache = null,
    gdn: ?GdnCache = null,
};

/// Everything one sequence's forward needs.
pub const State = struct {
    allocator: std.mem.Allocator,
    /// The queue the host copies are issued on; the executor owns it.
    stream_handle: *cuda_context.Stream,
    config: fixtures.RefConfig,
    plan: plan_module.Plan,
    arena_length: u32,
    allocations: std.ArrayList(cuda_context.DeviceBuffer),
    staging: cuda_context.PinnedHostBuffer,

    // Per-forward activations, all `[tokens, ...]` for `tokens <= tokens_max`.
    /// f32 `[tokens, hyper]`: the multi-stream residual.
    hyper: cuda_context.DeviceBuffer,
    /// f32 `[tokens, hc_count]`.
    injection: cuda_context.DeviceBuffer,
    /// f32 `[tokens, hidden]`: a block's single-stream input.
    stream_in: cuda_context.DeviceBuffer,
    /// f32 `[tokens, hidden]`: a block's single-stream output.
    stream_out: cuda_context.DeviceBuffer,
    /// bf16 `[tokens, hidden]`: the same stream for the bf16 kernels.
    stream_bf16: cuda_context.DeviceBuffer,
    /// f32 `[tokens, hidden]`: the final mixer's output and the logits input.
    mixed: cuda_context.DeviceBuffer,
    /// f32 `[tokens, vocab]`.
    logits: cuda_context.DeviceBuffer,
    /// u32 `[1]` and f32 `[1]`: the device-side argmax.
    argmax_index: cuda_context.DeviceBuffer,
    argmax_value: cuda_context.DeviceBuffer,
    /// i32 `[tokens]`: absolute positions of this forward.
    positions: cuda_context.DeviceBuffer,
    /// i32 `[1]`: gathered rows past a query's causal horizon.
    violations: cuda_context.DeviceBuffer,
    /// f32 `[arena, rotary_dim]` and int32 `[arena]` per mRoPE axis.
    rope_cos: cuda_context.DeviceBuffer,
    rope_sin: cuda_context.DeviceBuffer,
    rope_positions: cuda_context.DeviceBuffer,

    // Gated DeltaNet scratch.
    /// f32 `[tokens, conv_dim]`, `[tokens, value_dim]` and `[tokens, heads]`.
    gdn_qkv: cuda_context.DeviceBuffer,
    gdn_z: cuda_context.DeviceBuffer,
    gdn_a: cuda_context.DeviceBuffer,
    gdn_b: cuda_context.DeviceBuffer,
    /// f32 `[tokens, heads]` each.
    gdn_beta: cuda_context.DeviceBuffer,
    gdn_decay: cuda_context.DeviceBuffer,
    /// bf16 `[tokens, conv_dim]`: conv input and output.
    gdn_conv_in: cuda_context.DeviceBuffer,
    gdn_conv_out: cuda_context.DeviceBuffer,
    /// f32 `[tokens, key_dim]` each.
    gdn_query: cuda_context.DeviceBuffer,
    gdn_key: cuda_context.DeviceBuffer,
    /// f32 `[tokens, value_dim]`.
    gdn_value: cuda_context.DeviceBuffer,
    /// bf16 `[tokens, value_dim]`: the delta rule output, the gate, the normed
    /// output.
    gdn_core: cuda_context.DeviceBuffer,
    gdn_gate: cuda_context.DeviceBuffer,
    gdn_normed: cuda_context.DeviceBuffer,
    /// f32 `[tokens, value_dim]`: the normed output upcast for `out_proj`.
    gdn_normed_f32: cuda_context.DeviceBuffer,

    // Indexed-attention scratch.
    /// f32 `[tokens, heads * head_dim]` each: the query projection and gate.
    attn_query: cuda_context.DeviceBuffer,
    attn_gate: cuda_context.DeviceBuffer,
    /// f32 `[tokens, kv_dim]` each.
    attn_key: cuda_context.DeviceBuffer,
    attn_value: cuda_context.DeviceBuffer,
    /// bf16 `[tokens, heads * head_dim]`, `[tokens, kv_dim]`: normed and roped.
    attn_query_bf16: cuda_context.DeviceBuffer,
    attn_key_bf16: cuda_context.DeviceBuffer,
    attn_value_bf16: cuda_context.DeviceBuffer,
    /// bf16 `[tokens, index_heads, index_dim]`.
    index_query: cuda_context.DeviceBuffer,
    /// f32 `[tokens, projected_width]`.
    index_projected: cuda_context.DeviceBuffer,
    /// f32 `[indexerScoreBytes(tokens_max, arena / ratio)]`.
    index_scores: cuda_context.DeviceBuffer,
    /// u64 `[tokens, block_topk]`.
    index_partial: cuda_context.DeviceBuffer,
    /// i32 `[tokens, slots]`.
    index_selected: cuda_context.DeviceBuffer,
    /// i32 `[slots]` and bf16 `[slots, kv_dim]` each: one row's gather.
    gather_index: cuda_context.DeviceBuffer,
    gathered_key: cuda_context.DeviceBuffer,
    gathered_value: cuda_context.DeviceBuffer,
    /// bf16 `[tokens, heads * head_dim]`: the attention output.
    attn_out: cuda_context.DeviceBuffer,
    /// f32 `[tokens, heads * head_dim]`: the same, gated.
    attn_out_f32: cuda_context.DeviceBuffer,
    attn_gated: cuda_context.DeviceBuffer,

    // MoE scratch.
    /// f32 `[tokens, num_experts]`.
    moe_logits: cuda_context.DeviceBuffer,
    /// u32 `[tokens, top_k]` and f32 `[tokens, top_k]`.
    moe_indices: cuda_context.DeviceBuffer,
    moe_weights: cuda_context.DeviceBuffer,
    /// f32 `[tokens, 2 * intermediate]` and `[tokens, intermediate]`.
    moe_gate_up: cuda_context.DeviceBuffer,
    moe_act: cuda_context.DeviceBuffer,
    /// f32 `[experts, tokens, hidden]`.
    moe_expert_out: cuda_context.DeviceBuffer,
    /// f32 `[tokens, 2 * shared]`, `[tokens, shared]`, `[tokens, hidden]`.
    moe_shared_gate_up: cuda_context.DeviceBuffer,
    moe_shared_act: cuda_context.DeviceBuffer,
    moe_shared_out: cuda_context.DeviceBuffer,

    /// Per-layer caches, one entry per hidden layer.
    layers: [fixtures.layers_max]LayerCache = undefined,
    layer_count: u32 = 0,
    /// Tokens written to the caches so far.
    length: u32 = 0,
    /// Complete pooled indexer blocks so far.
    pooled_blocks: u32 = 0,

    /// Allocates every buffer and cache for `config`'s plan.
    pub fn init(
        allocator: std.mem.Allocator,
        queue: *cuda_context.Stream,
        config: fixtures.RefConfig,
        plan: plan_module.Plan,
    ) !State {
        assert(config.num_hidden_layers > 0);
        assert(plan.layer_count == config.num_hidden_layers);

        const arena_length = config.max_position_embeddings;
        const staging = try cuda_context.PinnedHostBuffer.allocate(staging_bytes);
        errdefer staging.free();
        var state = State{
            .allocator = allocator,
            .stream_handle = queue,
            .config = config,
            .plan = plan,
            .arena_length = arena_length,
            .allocations = std.ArrayList(cuda_context.DeviceBuffer).init(allocator),
            .staging = staging,
            .hyper = undefined,
            .injection = undefined,
            .stream_in = undefined,
            .stream_out = undefined,
            .stream_bf16 = undefined,
            .mixed = undefined,
            .logits = undefined,
            .argmax_index = undefined,
            .argmax_value = undefined,
            .positions = undefined,
            .violations = undefined,
            .rope_cos = undefined,
            .rope_sin = undefined,
            .rope_positions = undefined,
            .gdn_qkv = undefined,
            .gdn_z = undefined,
            .gdn_a = undefined,
            .gdn_b = undefined,
            .gdn_beta = undefined,
            .gdn_decay = undefined,
            .gdn_conv_in = undefined,
            .gdn_conv_out = undefined,
            .gdn_query = undefined,
            .gdn_key = undefined,
            .gdn_value = undefined,
            .gdn_core = undefined,
            .gdn_gate = undefined,
            .gdn_normed = undefined,
            .gdn_normed_f32 = undefined,
            .attn_query = undefined,
            .attn_gate = undefined,
            .attn_key = undefined,
            .attn_value = undefined,
            .attn_query_bf16 = undefined,
            .attn_key_bf16 = undefined,
            .attn_value_bf16 = undefined,
            .index_query = undefined,
            .index_projected = undefined,
            .index_scores = undefined,
            .index_partial = undefined,
            .index_selected = undefined,
            .gather_index = undefined,
            .gathered_key = undefined,
            .gathered_value = undefined,
            .attn_out = undefined,
            .attn_out_f32 = undefined,
            .attn_gated = undefined,
            .moe_logits = undefined,
            .moe_indices = undefined,
            .moe_weights = undefined,
            .moe_gate_up = undefined,
            .moe_act = undefined,
            .moe_expert_out = undefined,
            .moe_shared_gate_up = undefined,
            .moe_shared_act = undefined,
            .moe_shared_out = undefined,
        };
        errdefer state.release();
        try state.allocateForwardBuffers();
        try state.allocateRope();
        try state.allocateLayers();
        assert(state.length == 0);
        assert(state.pooled_blocks == 0);
        return state;
    }

    /// Frees every buffer, the staging window and the caches.
    pub fn deinit(self: *State) void {
        assert(self.layer_count > 0);
        assert(self.allocations.items.len > 0);

        self.release();
        self.layer_count = 0;
    }

    /// Forgets every cached token: the caches return to their init state.
    pub fn reset(self: *State) !void {
        assert(self.arena_length > 0);
        assert(self.layer_count > 0);

        for (self.allocations.items) |buffer| {
            try self.zero(buffer.pointer, buffer.bytes);
        }
        self.length = 0;
        self.pooled_blocks = 0;
        assert(self.length == 0);
    }

    /// Frees the buffers; safe on a partially constructed state.
    fn release(self: *State) void {
        assert(self.allocator.vtable != undefined);

        for (self.allocations.items) |buffer| {
            buffer.free();
        }
        self.allocations.deinit();
        self.staging.free();
    }

    /// Allocates the per-forward activation buffers.
    fn allocateForwardBuffers(self: *State) !void {
        assert(tokens_max > 0);
        assert(self.config.hidden_size > 0);

        const tokens: u64 = tokens_max;
        const hidden: u64 = self.config.hidden_size;
        const hyper: u64 = self.config.hyper_size();
        const heads: u64 = self.config.num_attention_heads;
        const head_dim: u64 = self.config.head_dim;
        const kv_dim: u64 = self.config.num_key_value_heads * head_dim;
        const width: u64 = heads * head_dim;
        const index_dim: u64 = self.config.indexer_head_dim;
        const index_heads: u64 = self.config.indexer_n_heads;
        const projected: u64 = index_heads + self.config.indexer_kv_heads;
        const slots: u64 = self.config.indexer_selected_slots();
        const block_topk: u64 = self.config.indexer_block_topk();
        const experts: u64 = self.config.num_experts;
        const top_k: u64 = self.config.num_experts_per_tok;
        const intermediate: u64 = self.config.moe_intermediate_size;
        const shared: u64 = self.config.shared_expert_intermediate_size;

        self.hyper = try self.push(tokens * hyper * 4);
        self.injection = try self.push(tokens * self.config.hc_count * 4);
        self.stream_in = try self.push(tokens * hidden * 4);
        self.stream_out = try self.push(tokens * hidden * 4);
        self.stream_bf16 = try self.push(tokens * hidden * 2);
        self.mixed = try self.push(tokens * hidden * 4);
        self.logits = try self.push(tokens * self.config.vocab_size * 4);
        self.argmax_index = try self.push(4);
        self.argmax_value = try self.push(4);
        self.positions = try self.push(tokens * 4);
        self.violations = try self.push(4);
        self.rope_positions = try self.push(self.arena_length * 4);

        const conv_dim: u64 = self.config.conv_dim();
        const value_dim: u64 = self.config.value_dim();
        const key_dim: u64 = self.config.key_dim();
        const gdn_heads: u64 = self.config.linear_num_value_heads;
        self.gdn_qkv = try self.push(tokens * conv_dim * 4);
        self.gdn_z = try self.push(tokens * value_dim * 4);
        self.gdn_a = try self.push(tokens * gdn_heads * 4);
        self.gdn_b = try self.push(tokens * gdn_heads * 4);
        self.gdn_beta = try self.push(tokens * gdn_heads * 4);
        self.gdn_decay = try self.push(tokens * gdn_heads * 4);
        self.gdn_conv_in = try self.push(tokens * conv_dim * 2);
        self.gdn_conv_out = try self.push(tokens * conv_dim * 2);
        self.gdn_query = try self.push(tokens * key_dim * 4);
        self.gdn_key = try self.push(tokens * key_dim * 4);
        self.gdn_value = try self.push(tokens * value_dim * 4);
        self.gdn_core = try self.push(tokens * value_dim * 2);
        self.gdn_gate = try self.push(tokens * value_dim * 2);
        self.gdn_normed = try self.push(tokens * value_dim * 2);
        self.gdn_normed_f32 = try self.push(tokens * value_dim * 4);

        self.attn_query = try self.push(tokens * width * 4);
        self.attn_gate = try self.push(tokens * width * 4);
        self.attn_key = try self.push(tokens * kv_dim * 4);
        self.attn_value = try self.push(tokens * kv_dim * 4);
        self.attn_query_bf16 = try self.push(tokens * width * 2);
        self.attn_key_bf16 = try self.push(tokens * kv_dim * 2);
        self.attn_value_bf16 = try self.push(tokens * kv_dim * 2);
        self.index_query = try self.push(tokens * index_heads * index_dim * 2);
        self.index_projected = try self.push(tokens * projected * index_dim * 4);
        self.index_scores = try self.push(self.indexScoreBytes());
        self.index_partial = try self.push(tokens * block_topk * 8);
        self.index_selected = try self.push(tokens * slots * 4);
        self.gather_index = try self.push(slots * 4);
        self.gathered_key = try self.push(slots * kv_dim * 2);
        self.gathered_value = try self.push(slots * kv_dim * 2);
        self.attn_out = try self.push(tokens * width * 2);
        self.attn_out_f32 = try self.push(tokens * width * 4);
        self.attn_gated = try self.push(tokens * width * 4);

        self.moe_logits = try self.push(tokens * experts * 4);
        self.moe_indices = try self.push(tokens * top_k * 4);
        self.moe_weights = try self.push(tokens * top_k * 4);
        self.moe_gate_up = try self.push(tokens * 2 * intermediate * 4);
        self.moe_act = try self.push(tokens * intermediate * 4);
        self.moe_expert_out = try self.push(experts * tokens * hidden * 4);
        self.moe_shared_gate_up = try self.push(tokens * 2 * shared * 4);
        self.moe_shared_act = try self.push(tokens * shared * 4);
        self.moe_shared_out = try self.push(tokens * hidden * 4);
        assert(self.staging.bytes >= staging_bytes);
    }

    /// Bytes the indexer's streaming score buffer needs: one
    /// `score_group_blocks`-wide group per row, as `attention.zig` sizes it.
    ///
    /// The kernel writes at `scores[row * score_group_blocks + block]`, so the
    /// stride is the group width, not the context's block count. Sizing this
    /// buffer by the block count under-allocates it by two orders of magnitude
    /// at the production context and the score kernel then writes past its end
    /// into whatever the next allocation is.
    fn indexScoreBytes(self: *const State) u64 {
        assert(self.arena_length > 0);
        assert(self.config.indexer_compress_ratio > 0);

        const blocks = self.arena_length / self.config.indexer_compress_ratio;
        return attention_kernels.Attention.indexerScoreBytes(tokens_max, blocks);
    }

    /// Allocates the mRoPE cos/sin tables and their position grid.
    fn allocateRope(self: *State) !void {
        assert(self.arena_length > 0);
        assert(self.config.rotary_dim() > 0);

        const rotary: u64 = self.config.rotary_dim();
        self.rope_cos = try self.push(self.arena_length * rotary * 4);
        self.rope_sin = try self.push(self.arena_length * rotary * 4);
        assert(self.rope_positions.bytes >= @as(u64, self.arena_length) * 4);
    }

    /// Allocates one cache set per hidden layer.
    fn allocateLayers(self: *State) !void {
        assert(self.config.num_hidden_layers <= fixtures.layers_max);
        assert(self.arena_length > 0);

        for (0..self.config.num_hidden_layers) |index| {
            self.layers[index] = try self.allocateLayer(@intCast(index));
            self.layer_count += 1;
        }
        assert(self.layer_count == self.config.num_hidden_layers);
    }

    /// Allocates the cache set of layer `index`.
    fn allocateLayer(self: *State, index: u32) !LayerCache {
        assert(index < self.config.num_hidden_layers);
        const kind: plan_module.BlockKind = switch (self.config.layer_types[index]) {
            .linear_attention => .linear_attention,
            .indexed_attention => .indexed_attention,
        };
        assert(self.plan.layer(index).kind == kind);

        const kv_dim: u64 = self.config.num_key_value_heads * self.config.head_dim;
        const index_dim: u64 = self.config.indexer_head_dim;
        const ratio: u64 = self.config.indexer_compress_ratio;
        const arena: u64 = self.arena_length;
        var cache = LayerCache{};
        switch (self.plan.layer(index).kind) {
            .indexed_attention => cache.attention = .{
                .key = try self.push(arena * kv_dim * 2),
                .value = try self.push(arena * kv_dim * 2),
                .index_keys = try self.push(arena * index_dim * 2),
                .index_pooled = try self.push((arena / ratio) * index_dim * 2),
            },
            .linear_attention => cache.gdn = .{
                .conv = try self.push(
                    self.config.conv_dim() *
                        (self.config.linear_conv_kernel_dim - 1) * 4,
                ),
                .recurrent = try self.push(
                    self.config.linear_num_value_heads *
                        self.config.linear_key_head_dim *
                        self.config.linear_value_head_dim * 4,
                ),
            },
        }
        return cache;
    }

    /// Allocates one buffer and tracks it for `deinit`.
    fn push(self: *State, bytes: u64) !cuda_context.DeviceBuffer {
        assert(bytes > 0);

        const buffer = try allocate(bytes);
        errdefer buffer.free();
        try self.allocations.append(buffer);
        return buffer;
    }

    /// Uploads `values` into `buffer` through the pinned staging window.
    pub fn uploadF32(self: *State, buffer: *const cuda_context.DeviceBuffer, values: []const f32) !void {
        assert(values.len > 0);
        assert(values.len * 4 <= buffer.bytes);

        const bytes = std.mem.sliceAsBytes(values);
        var written: u64 = 0;
        while (written < bytes.len) {
            const chunk: u64 = @min(self.staging.bytes, bytes.len - written);
            @memcpy(
                self.staging.slice()[0..@intCast(chunk)],
                bytes[@intCast(written)..][0..@intCast(chunk)],
            );
            const destination = view(
                buffer,
                written,
                chunk,
            );
            try self.stream().copy_h_to_d(&destination, &self.staging, chunk);
            written += chunk;
        }
        try self.stream().synchronize();
        assert(written == bytes.len);
    }

    /// Uploads `values` into `buffer` as i32 words.
    pub fn uploadI32(self: *State, buffer: *const cuda_context.DeviceBuffer, values: []const i32) !void {
        assert(values.len > 0);
        assert(values.len * 4 <= buffer.bytes);

        try self.uploadRaw(buffer, std.mem.sliceAsBytes(values));
    }

    /// Uploads raw bytes into `buffer`.
    pub fn uploadRaw(self: *State, buffer: *const cuda_context.DeviceBuffer, bytes: []const u8) !void {
        assert(bytes.len > 0);
        assert(bytes.len <= buffer.bytes);

        var written: u64 = 0;
        while (written < bytes.len) {
            const chunk: u64 = @min(self.staging.bytes, bytes.len - written);
            @memcpy(
                self.staging.slice()[0..@intCast(chunk)],
                bytes[@intCast(written)..][0..@intCast(chunk)],
            );
            const destination = view(buffer, written, chunk);
            try self.stream().copy_h_to_d(&destination, &self.staging, chunk);
            written += chunk;
        }
        try self.stream().synchronize();
        assert(written == bytes.len);
    }

    /// Downloads `out.len` f32 values from `buffer`.
    pub fn downloadF32(self: *State, buffer: *const cuda_context.DeviceBuffer, out: []f32) !void {
        assert(out.len > 0);
        assert(out.len * 4 <= buffer.bytes);

        try self.downloadRaw(buffer, std.mem.sliceAsBytes(out));
    }

    /// Downloads `bytes.len` raw bytes from `buffer`.
    pub fn downloadRaw(self: *State, buffer: *const cuda_context.DeviceBuffer, out: []u8) !void {
        assert(out.len > 0);
        assert(out.len <= buffer.bytes);

        var read: u64 = 0;
        while (read < out.len) {
            const chunk: u64 = @min(self.staging.bytes, out.len - read);
            const source = view(buffer, read, chunk);
            try self.stream().copy_d_to_h(&self.staging, &source, chunk);
            try self.stream().synchronize();
            @memcpy(
                out[@intCast(read)..][0..@intCast(chunk)],
                self.staging.slice()[0..@intCast(chunk)],
            );
            read += chunk;
        }
        assert(read == out.len);
    }

    /// Zeroes `bytes` of device memory starting at `pointer`.
    fn zero(self: *State, pointer: cuda_driver.DevicePtr, bytes: u64) !void {
        assert(pointer != 0);
        assert(bytes > 0);
        assert(self.staging.bytes > 0);

        const buffer = cuda_context.DeviceBuffer{ .pointer = pointer, .bytes = bytes };
        @memset(self.staging.slice(), 0);
        var written: u64 = 0;
        while (written < bytes) {
            const chunk: u64 = @min(self.staging.bytes, bytes - written);
            const destination = view(&buffer, written, chunk);
            try self.stream().copy_h_to_d(&destination, &self.staging, chunk);
            written += chunk;
        }
        try self.stream().synchronize();
        assert(written == bytes);
    }

    /// The queue every host copy is issued on.
    fn stream(self: *State) *cuda_context.Stream {
        assert(self.config.hidden_size > 0);

        return self.stream_handle;
    }
};

/// A device buffer view: `bytes` starting `offset` bytes into `buffer`.
pub fn view(
    buffer: *const cuda_context.DeviceBuffer,
    offset: u64,
    bytes: u64,
) cuda_context.DeviceBuffer {
    assert(offset + bytes <= buffer.bytes);
    assert(bytes > 0);

    return .{ .pointer = buffer.pointer + offset, .bytes = bytes };
}

/// The stream every host copy is issued on.
fn stream(self: *State) *cuda_context.Stream {
    assert(self.stream_handle != undefined);
    assert(self.config.hidden_size > 0);
    return self.stream_handle;
}
