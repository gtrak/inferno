//! Shard-aware weight loading: which of the checkpoint's tensors each rank
//! opens, and the byte refusal that stops a run which would not fit.
//!
//! `mem/placement.zig` owns the sharding *decision* and the byte *rule*;
//! `weights/manifest.zig` owns the typed checkpoint, file by file. This file is
//! the one place that applies the decision to the real tensors and then holds
//! the result to the rule, per rank, before a byte is copied.
//!
//! Two things it deliberately does **not** do, and one thing it does that the
//! tiny fixture path could not:
//!
//! * it does not re-decide what is replicated, tensor parallel or expert
//!   parallel — `shardingFor` maps the manifest's roles onto
//!   `placement.Sharding`, one row per role, and each row's comment is the
//!   reason;
//! * it does not load the whole checkpoint on every rank: a rank opens its own
//!   expert shard, its own rows of the projections that are tensor parallel,
//!   and one whole copy of what is replicated;
//! * it refuses a plan whose per-rank device set exceeds that rank's
//!   **measured** free VRAM, through the same `placement.verifyRanks` the
//!   planned table is held to.
//!
//! The numbers come from the checkpoint's own dtypes, not from the requantised
//! plan. Only the routed experts are NVFP4 (U8 pairs, one F8_E4M3 scale per 16
//! values, two f32 scales per matrix); attention, GDN, the hyper-connections,
//! the router, the shared expert, `embed_tokens` and `lm_head` are BF16. The
//! shipped expert slab is therefore 2 764 824 B against the 2 611 212 B
//! `placement.expertSlabBytes()` budgets for at per-32 (I-020) — 5.8 % more,
//! which the host tier has to absorb. `plan` reports both, so the difference is
//! visible rather than assumed.
//!
//! `lm_head`, the GDN `out_proj` and the shared expert's gate/up are split over
//! **output rows**, so each rank's slice is a contiguous run of rows in its
//! file: the shard a rank opens is a byte range, not a repack. The shared
//! expert's down projection is column-split by shape and so is replicated: the
//! gathered activation is reduced by running it whole on every rank, which is
//! the design's choice (`model/multi.zig` records the bandwidth argument).
//!
//! `embed_tokens` and the 51.2 GB n-gram table are **host-resident**: both are
//! gather-only, one row per token, so they stay in the read-only mapping and
//! cost no rank any VRAM. The MTP head and the vision tower are not loaded at
//! all; they are counted and reported as deferred.
//!
//! The rules this file is built to (plan.md §8) hold the same way they do in
//! `kernels/`: the GPU work goes through `hw/api.zig`, never the raw driver, so
//! the CUDA mass stays inside `hw/`.

const std = @import("std");

const cuda_context = @import("../hw/api.zig").cuda_context;
const cuda_driver = @import("../hw/api.zig").cuda_driver;
const manifest = @import("manifest.zig");
const placement = @import("../mem/placement.zig");
const safetensors = @import("safetensors.zig");

const assert = std.debug.assert;

/// Ranks the engine runs on: one per GPU, straight from the placement.
pub const rank_count: u32 = placement.rank_count;

/// A slab id that is not an expert slab.
pub const no_slab: u32 = std.math.maxInt(u32);

/// Alignment of one tensor inside a rank's device arena. The checkpoint's own
/// layout is 8-byte aligned; 16 keeps a vectorised kernel's loads aligned
/// without padding the arena measurably.
pub const alignment_bytes: u64 = 16;

/// Bytes one staged read moves from the mapping to the device.
pub const read_chunk_bytes: u64 = 1 << 26;

pub const ShardError = error{
    shard_not_divisible,
    slab_layout_mismatch,
    tensor_bytes_disagree,
    rank_bytes_disagree,
    extent_out_of_order,
};

/// Where a rank keeps one tensor.
pub const Storage = enum {
    /// Copied into the rank's device arena, in the region that cannot move.
    device_fixed,
    /// Copied into an expert slab: the hot ones on device, the cold ones in the
    /// host tier.
    device_slab,
    /// Left in the read-only mapping: gather-only, never device-resident.
    mapping,
};

/// One tensor of the checkpoint, as one rank has to handle it.
pub const Item = struct {
    ref: manifest.TensorRef,
    role: manifest.Role,
    layer: ?u32 = null,
    /// First row of `ref` this rank opens; every slice is rows, because a
    /// safetensors tensor is row-major and a row range is a byte range.
    first_row: u64 = 0,
    rows: u64 = 0,
    storage: Storage = .device_fixed,
    /// Expert slab this item belongs to; `no_slab` for everything else.
    slab: u32 = no_slab,
};

/// Rows of one tensor a rank opens.
pub const RowSlice = struct {
    first: u64,
    rows: u64,
};

/// A sink for one walk over the checkpoint's tensors.
const Sink = struct {
    user: *anyopaque,
    add: *const fn (user: *anyopaque, item: Item) anyerror!void,

    fn push(self: Sink, item: Item) !void {
        assert(item.ref.byte_len > 0);
        assert(item.rows > 0);
        try self.add(self.user, item);
    }
};

/// Bytes one row of `ref` occupies; a rank-0 tensor is one row.
pub fn rowBytes(ref: manifest.TensorRef) u64 {
    assert(ref.byte_len > 0);
    assert(ref.dims.rank <= safetensors.rank_max);

    const rows: u64 = if (ref.dims.rank == 0) 1 else ref.dims.values[0];
    assert(rows > 0);
    const bytes = @divExact(ref.byte_len, rows);
    assert(bytes > 0);
    return bytes;
}

/// Bytes of `ref` one rank opens.
pub fn itemBytes(item: Item) u64 {
    assert(item.rows > 0);
    assert(item.ref.byte_len > 0);

    return item.rows * rowBytes(item.ref);
}

// --- the policy -------------------------------------------------------------

/// The placement row one manifest role belongs to.
///
/// This is `mem/placement.zig`'s decision read against the checkpoint's own
/// tensor names: no row here decides anything new, it names which decided row a
/// tensor falls under. `placement_table` budgets the *requantised* sizes;
/// `plan` measures the real ones.
pub fn shardingFor(role: manifest.Role) placement.Sharding {
    assert(manifest.all_roles.len == std.meta.fields(manifest.Role).len);

    return switch (role) {
        // Gather-only, one row per token: the table and its index stay in the
        // mapping, and every rank reads the same pages.
        .embed_tokens, .ngram_table, .ngram_index, .ngram_scale => .host_resident,
        // Row split by vocabulary, gathered and reduced on device.
        .lm_head => .tensor_parallel,
        // 128 consecutive experts per rank, in every layer.
        .expert => .expert_parallel,
        // Cheaper to copy than to reduce (attention, the indexer, the two GQA
        // key/value heads) or unsplittable (the router, so every rank routes
        // identically; the hyper-connection mixer, whose 10240-wide stream is
        // the residual itself).
        .attn_hyper_connection,
        .mlp_hyper_connection,
        .hyper_connection_mixer,
        .router_gate,
        .shared_expert,
        .shared_expert_gate,
        .q_proj,
        .k_proj,
        .v_proj,
        .o_proj,
        .attn_norm,
        .indexer,
        .ple_projection,
        .ple_norm,
        .ple_conv1d,
        => .replicated,
        // The GDN block: `out_proj` is split over output rows, the qkv/z/b/a
        // projections are replicated because their output is per-head state.
        // `walkAttention` tells the two apart; the row is the block's.
        .linear_attn => .tensor_parallel,
        // The MTP head and the vision tower are not this engine's to load.
        .deferred => .host_resident,
    };
}

/// Rows of `ref` a rank opens under `sharding`, or null when it opens none.
fn shardRows(
    ref: manifest.TensorRef,
    sharding: placement.Sharding,
    rank: u32,
) !?RowSlice {
    assert(rank < rank_count);
    assert(ref.byte_len > 0);

    if (ref.dims.rank == 0) {
        return .{ .first = 0, .rows = 1 };
    } else {
        assert(ref.dims.rank <= safetensors.rank_max);
    }
    const rows: u64 = ref.dims.values[0];
    switch (sharding) {
        .replicated => return .{ .first = 0, .rows = rows },
        .tensor_parallel => {
            if (rows % rank_count != 0) {
                std.debug.print(
                    "shard: {d} rows do not split over {d} ranks\n",
                    .{ rows, rank_count },
                );
                return ShardError.shard_not_divisible;
            } else {
                assert(rows % rank_count == 0);
            }
            const per_rank = rows / rank_count;
            const first = per_rank * rank;
            assert(first + per_rank <= rows);
            return .{ .first = first, .rows = per_rank };
        },
        // An expert's rows belong to exactly one rank: the walker emits the
        // experts it owns and never asks a row split for one.
        .expert_parallel, .host_resident => return null,
    }
}

// --- the walk ---------------------------------------------------------------

/// Every text tensor of the checkpoint, as `rank` has to handle it.
///
/// One walk, two sinks: the accounting pass sizes the run and the extent pass
/// builds the load list, so the bytes the report prints and the bytes the
/// loader copies are the same numbers by construction.
pub fn walkRank(
    subject: *const manifest.Manifest,
    rank: u32,
    sink: Sink,
) !void {
    assert(rank < rank_count);
    assert(subject.layers.len == subject.text.config.num_hidden_layers);

    try walkGlobal(subject, rank, sink);
    for (subject.layers) |*layer| {
        assert(layer.index < subject.layers.len);
        try walkHyper(&layer.attn_hyper_connection, layer.index, sink);
        try walkHyper(&layer.mlp_hyper_connection, layer.index, sink);
        try walkAttention(subject, layer, rank, sink);
        try walkMoe(subject, layer, rank, sink);
        if (layer.ple) |*ple| {
            try walkPle(subject, ple, layer.index, sink);
        } else {
            assert(layer.ple == null);
        }
    }
}

/// `embed_tokens` (mapped), `lm_head` (row split) and the final mixer (replica).
fn walkGlobal(
    subject: *const manifest.Manifest,
    rank: u32,
    sink: Sink,
) !void {
    assert(subject.layers.len > 0);
    assert(rank < rank_count);

    try emit(subject.embed_tokens, .embed_tokens, null, rank, sink);
    try emit(subject.lm_head, .lm_head, null, rank, sink);
    const mixer = &subject.final_mixer;
    try emitFull(mixer.norm, .hyper_connection_mixer, null, sink);
    try emitFull(mixer.mix_down, .hyper_connection_mixer, null, sink);
    try emitFull(mixer.mix_up, .hyper_connection_mixer, null, sink);
}

/// One hyper-connection block: four replicated tensors.
fn walkHyper(
    block: *const manifest.HyperConnection,
    layer: u32,
    sink: Sink,
) !void {
    assert(layer < placement.layer_count);
    assert(block.block_inject.byte_len > 0);

    const role: manifest.Role = .attn_hyper_connection;
    try emitFull(block.block_inject, role, layer, sink);
    try emitFull(block.norm, role, layer, sink);
    try emitFull(block.mix_down, role, layer, sink);
    try emitFull(block.mix_up, role, layer, sink);
}

/// The attention slot: GatedDeltaNet's tensors, or indexed attention's.
fn walkAttention(
    subject: *const manifest.Manifest,
    layer: *const manifest.Layer,
    rank: u32,
    sink: Sink,
) !void {
    assert(subject.layers.len > 0);
    assert(layer.index < subject.layers.len);

    switch (layer.kind) {
        .linear_attention => {
            const attention = &layer.linear_attn;
            // The one projection whose output rows are the hidden stream, so
            // its slice is the rank's own dense slice.
            try emitSliced(
                attention.out_proj,
                .linear_attn,
                layer.index,
                rank,
                sink,
            );
            try emitFull(attention.in_proj_qkv, .linear_attn, layer.index, sink);
            try emitFull(attention.in_proj_z, .linear_attn, layer.index, sink);
            try emitFull(attention.in_proj_a, .linear_attn, layer.index, sink);
            try emitFull(attention.in_proj_b, .linear_attn, layer.index, sink);
            try emitFull(attention.conv1d, .linear_attn, layer.index, sink);
            try emitFull(attention.a_log, .linear_attn, layer.index, sink);
            try emitFull(attention.dt_bias, .linear_attn, layer.index, sink);
            try emitFull(attention.norm, .linear_attn, layer.index, sink);
        },
        .full_attention => {
            const attention = &layer.self_attn;
            try emitFull(attention.q_proj, .q_proj, layer.index, sink);
            try emitFull(attention.k_proj, .k_proj, layer.index, sink);
            try emitFull(attention.v_proj, .v_proj, layer.index, sink);
            try emitFull(attention.o_proj, .o_proj, layer.index, sink);
            try emitFull(attention.q_norm, .attn_norm, layer.index, sink);
            try emitFull(attention.k_norm, .attn_norm, layer.index, sink);
            try emitFull(attention.indexer_qk_proj, .indexer, layer.index, sink);
            try emitFull(attention.indexer_q_norm, .indexer, layer.index, sink);
            try emitFull(attention.indexer_k_norm, .indexer, layer.index, sink);
        },
    }
}

/// The MoE block: router, shared expert and this rank's expert shard.
fn walkMoe(
    subject: *const manifest.Manifest,
    layer: *const manifest.Layer,
    rank: u32,
    sink: Sink,
) !void {
    assert(layer.index < subject.layers.len);
    assert(rank < rank_count);

    try emitFull(layer.router_gate, .router_gate, layer.index, sink);
    try emitSliced(
        layer.shared_expert.gate_proj,
        .shared_expert,
        layer.index,
        rank,
        sink,
    );
    try emitSliced(
        layer.shared_expert.up_proj,
        .shared_expert,
        layer.index,
        rank,
        sink,
    );
    // Column split by shape: the gathered activation is reduced by running the
    // down projection whole on every rank (see this file's header).
    try emitFull(layer.shared_expert.down_proj, .shared_expert, layer.index, sink);
    try emitFull(layer.shared_expert_gate, .shared_expert_gate, layer.index, sink);
    try walkExperts(subject, layer, rank, sink);
}

/// This rank's consecutive expert shard: `experts_per_rank` of them.
fn walkExperts(
    subject: *const manifest.Manifest,
    layer: *const manifest.Layer,
    rank: u32,
    sink: Sink,
) !void {
    assert(subject.layers.len > 0);
    assert(rank < rank_count);

    if (layer.experts.len % rank_count != 0) {
        return ShardError.shard_not_divisible;
    } else {
        assert(layer.experts.len % rank_count == 0);
    }
    const per_rank: u32 = @intCast(layer.experts.len / rank_count);
    assert(per_rank == placement.experts_per_rank);
    const first: u32 = per_rank * rank;
    for (0..per_rank) |offset| {
        const local: u32 = @intCast(offset);
        const expert = &layer.experts[first + local];
        const slab = layer.index * per_rank + local;
        try emitSlab(expert.gate_proj, layer.index, slab, sink);
        try emitSlab(expert.up_proj, layer.index, slab, sink);
        try emitSlab(expert.down_proj, layer.index, slab, sink);
    }
}

/// The PLE layer: projections replicated, the n-gram table and its index in
/// the mapping.
fn walkPle(
    subject: *const manifest.Manifest,
    ple: *const manifest.Ple,
    layer: u32,
    sink: Sink,
) !void {
    assert(subject.layers.len > 0);
    assert(ple.embedding_shards.len > 0);

    try emitFull(ple.key_proj, .ple_projection, layer, sink);
    try emitFull(ple.value_proj, .ple_projection, layer, sink);
    try emitFull(ple.conv1d, .ple_conv1d, layer, sink);
    try emitFull(ple.norm_key, .ple_norm, layer, sink);
    try emitFull(ple.norm_query, .ple_norm, layer, sink);
    try emitFull(ple.norm_conv, .ple_norm, layer, sink);
    try emitFull(ple.heads_offsets, .ngram_index, layer, sink);
    try emitFull(ple.heads_vocab_sizes, .ngram_index, layer, sink);
    try emitFull(ple.layer_multipliers, .ngram_index, layer, sink);
    try emitFull(ple.embedding_scale, .ngram_scale, layer, sink);
    for (ple.embedding_shards) |ref| {
        try emitFull(ref, .ngram_table, layer, sink);
    }
}

/// Emits one tensor with the policy its role's placement row names.
fn emit(
    ref: manifest.TensorRef,
    role: manifest.Role,
    layer: ?u32,
    rank: u32,
    sink: Sink,
) !void {
    assert(!ref.isEmpty());
    assert(rank < rank_count);

    const sharding = shardingFor(role);
    const slice = try shardRows(ref, sharding, rank);
    if (slice) |rows| {
        assert(rows.rows > 0);
        try sink.push(.{
            .ref = ref,
            .role = role,
            .layer = layer,
            .first_row = rows.first,
            .rows = rows.rows,
            .storage = storageFor(role),
        });
    } else {
        assert(slice == null);
    }
}

/// Emits one tensor's rows split over the ranks.
fn emitSliced(
    ref: manifest.TensorRef,
    role: manifest.Role,
    layer: u32,
    rank: u32,
    sink: Sink,
) !void {
    assert(!ref.isEmpty());
    assert(rank < rank_count);

    const slice = (try shardRows(ref, .tensor_parallel, rank)) orelse {
        return ShardError.shard_not_divisible;
    };
    try sink.push(.{
        .ref = ref,
        .role = role,
        .layer = layer,
        .first_row = slice.first,
        .rows = slice.rows,
        .storage = storageFor(role),
    });
}

/// Emits one tensor whole, on every rank.
fn emitFull(
    ref: manifest.TensorRef,
    role: manifest.Role,
    layer: ?u32,
    sink: Sink,
) !void {
    assert(!ref.isEmpty());
    assert(ref.dims.rank <= safetensors.rank_max);

    try sink.push(.{
        .ref = ref,
        .role = role,
        .layer = layer,
        .first_row = 0,
        .rows = if (ref.dims.rank == 0) 1 else ref.dims.values[0],
        .storage = storageFor(role),
    });
}

/// Emits one NVFP4 matrix and its three scales as part of a slab.
fn emitSlab(
    projection: manifest.QuantizedProjection,
    layer: u32,
    slab: u32,
    sink: Sink,
) !void {
    assert(slab != no_slab);
    assert(!projection.weight.isEmpty());

    try slabPart(projection.weight, layer, slab, sink);
    try slabPart(projection.weight_scale, layer, slab, sink);
    try slabPart(projection.weight_scale_2, layer, slab, sink);
    try slabPart(projection.input_scale, layer, slab, sink);
}

/// Emits one tensor of a slab whole; the slab is the unit that moves.
fn slabPart(
    ref: manifest.TensorRef,
    layer: u32,
    slab: u32,
    sink: Sink,
) !void {
    assert(slab != no_slab);
    assert(!ref.isEmpty());

    try sink.push(.{
        .ref = ref,
        .role = .expert,
        .layer = layer,
        .first_row = 0,
        .rows = if (ref.dims.rank == 0) 1 else ref.dims.values[0],
        .storage = .device_slab,
        .slab = slab,
    });
}

/// A role's tensors are in the mapping or on the device, and a slab part is
/// only ever an expert's.
fn storageFor(role: manifest.Role) Storage {
    assert(manifest.all_roles.len > 0);
    assert(@intFromEnum(role) < manifest.all_roles.len);

    if (shardingFor(role) == .host_resident) {
        return .mapping;
    } else {
        return .device_fixed;
    }
}

// --- the accounting ---------------------------------------------------------

/// Per-rank bytes, summed over the walk.
pub const Accounting = struct {
    /// Device bytes this rank would hold with every slab on the device.
    device_bytes_all: u64 = 0,
    /// Of which the region that cannot move: everything that is not a slab.
    fixed_bytes: u64 = 0,
    /// Bytes this rank gathers out of the read-only mapping.
    mapping_bytes: u64 = 0,
    /// Slabs this rank owns.
    slab_count: u32 = 0,
    /// Tensors this rank opens or maps.
    tensors: u64 = 0,
    per_role: [manifest.all_roles.len]manifest.RoleTotal =
        [_]manifest.RoleTotal{.{}} ** manifest.all_roles.len,
};

const CountSink = struct {
    accounting: Accounting = .{},

    fn sink(self: *CountSink) Sink {
        assert(@intFromPtr(self) != 0);
        return .{ .user = @ptrCast(self), .add = add };
    }

    fn add(user: *anyopaque, item: Item) anyerror!void {
        const self: *CountSink = @ptrCast(@alignCast(user));
        const bytes = itemBytes(item);
        switch (item.storage) {
            .device_fixed => {
                self.accounting.fixed_bytes += bytes;
                self.accounting.device_bytes_all += bytes;
            },
            .device_slab => {
                if (item.slab == no_slab) {
                    return ShardError.slab_layout_mismatch;
                } else {
                    assert(item.slab != no_slab);
                }
                self.accounting.device_bytes_all += bytes;
                self.accounting.slab_count = @max(
                    self.accounting.slab_count,
                    item.slab + 1,
                );
            },
            .mapping => self.accounting.mapping_bytes += bytes,
        }
        self.accounting.tensors += 1;
        self.accounting.per_role[@intFromEnum(item.role)].add(bytes);
    }
};

/// Sizes one rank's share of the checkpoint from the typed manifest.
pub fn account(
    subject: *const manifest.Manifest,
    rank: u32,
) !Accounting {
    assert(rank < rank_count);
    assert(subject.layers.len > 0);

    var counter = CountSink{};
    try walkRank(subject, rank, counter.sink());
    const slab = try slabBytes(subject);
    if (counter.accounting.slab_count == 0) {
        return ShardError.slab_layout_mismatch;
    } else {
        assert(counter.accounting.slab_count > 0);
    }
    // Every slab part was counted into `device_bytes_all`; they must add up to
    // whole slabs, which is what lets the loader size an arena from the count.
    const slab_total = @as(u64, counter.accounting.slab_count) * slab;
    if (counter.accounting.device_bytes_all !=
        counter.accounting.fixed_bytes + slab_total)
    {
        std.debug.print(
            "shard: expert bytes {d} B are not {d} slabs of {d} B\n",
            .{
                counter.accounting.device_bytes_all -
                    counter.accounting.fixed_bytes,
                counter.accounting.slab_count,
                slab,
            },
        );
        return ShardError.slab_layout_mismatch;
    } else {
        assert(counter.accounting.device_bytes_all ==
            counter.accounting.fixed_bytes + slab_total);
    }
    try checkCompleteness(&counter.accounting, subject);
    return counter.accounting;
}

/// Bytes one (layer, expert) slab holds, from the checkpoint's own shapes:
/// gate, up and down as NVFP4 with both scale tensors each.
pub fn slabBytes(subject: *const manifest.Manifest) !u64 {
    assert(subject.layers.len > 0);
    assert(subject.layers[0].experts.len > 0);

    const expert = &subject.layers[0].experts[0];
    const projections = [_]manifest.QuantizedProjection{
        expert.gate_proj,
        expert.up_proj,
        expert.down_proj,
    };
    var total: u64 = 0;
    for (projections) |projection| {
        total += projection.weight.byte_len;
        total += projection.weight_scale.byte_len;
        total += projection.weight_scale_2.byte_len;
        total += projection.input_scale.byte_len;
    }
    if (total == 0) {
        return ShardError.slab_layout_mismatch;
    } else {
        assert(total > 0);
    }
    return total;
}

/// Bytes every rank's share of the text path should total: the whole checkpoint
/// text bytes, less what the splits take away three quarters of.
///
/// Derived from the manifest's own role totals and typed refs, never from the
/// walk, so a tensor the walk drops or double-counts fails here.
fn expectedRankBytes(subject: *const manifest.Manifest) !u64 {
    assert(subject.layers.len > 0);
    assert(subject.total(.deferred).count > 0);

    var text_bytes: u64 = 0;
    for (manifest.all_roles) |role| {
        if (role == .deferred) {
            continue;
        } else {
            assert(role != .deferred);
        }
        text_bytes += subject.total(role).bytes;
    }
    var split_bytes: u64 = subject.total(.expert).bytes +
        subject.total(.lm_head).bytes;
    for (subject.layers) |*layer| {
        switch (layer.kind) {
            .linear_attention => {
                split_bytes += layer.linear_attn.out_proj.byte_len;
            },
            .full_attention => {
                // Indexed attention is replicated whole.
            },
        }
        split_bytes += layer.shared_expert.gate_proj.byte_len;
        split_bytes += layer.shared_expert.up_proj.byte_len;
    }
    if (split_bytes == 0 or split_bytes > text_bytes) {
        return ShardError.tensor_bytes_disagree;
    } else {
        assert(split_bytes <= text_bytes);
    }
    const kept = text_bytes - split_bytes;
    return kept + @divExact(split_bytes, rank_count);
}

/// Every text tensor is on the device or in the mapping, exactly once, and the
/// split ones are a quarter on this rank.
fn checkCompleteness(
    accounting: *const Accounting,
    subject: *const manifest.Manifest,
) !void {
    assert(accounting.tensors > 0);
    assert(accounting.device_bytes_all > 0);

    const expected = try expectedRankBytes(subject);
    const actual = accounting.device_bytes_all + accounting.mapping_bytes;
    if (actual != expected) {
        std.debug.print(
            "shard: rank accounts for {d} B against {d} B expected\n",
            .{ actual, expected },
        );
        return ShardError.tensor_bytes_disagree;
    } else {
        assert(actual == expected);
    }
}

// --- the load list ----------------------------------------------------------

/// One byte range this rank copies into its arena.
pub const Extent = struct {
    /// Which tensor, and which of its rows.
    ref: manifest.TensorRef,
    role: manifest.Role,
    layer: ?u32 = null,
    first_row: u64,
    rows: u64,
    /// Byte offset inside the rank's device arena.
    slot: u64,
    /// Slab this extent belongs to; `no_slab` for the fixed region.
    slab: u32 = no_slab,
};

const ExtentSink = struct {
    extents: std.ArrayList(Extent),
    /// Bytes of the region that cannot move, page-rounded: where slabs start.
    fixed_region_bytes: u64,
    slab_stride_bytes: u64,
    hot_slabs: u32,
    /// Running offset inside the fixed region.
    fixed_cursor: u64 = 0,
    /// The slab the previous item belonged to, and its running offset.
    slab: u32 = no_slab,
    slab_cursor: u64 = 0,

    fn sink(self: *ExtentSink) Sink {
        assert(@intFromPtr(self) != 0);
        return .{ .user = @ptrCast(self), .add = add };
    }

    fn add(user: *anyopaque, item: Item) anyerror!void {
        const self: *ExtentSink = @ptrCast(@alignCast(user));
        const bytes = itemBytes(item);
        switch (item.storage) {
            .device_fixed => {
                const slot = std.mem.alignForward(
                    u64,
                    self.fixed_cursor,
                    alignment_bytes,
                );
                self.fixed_cursor = slot + bytes;
                try self.append(item, slot, no_slab);
            },
            .device_slab => {
                if (item.slab == no_slab) {
                    return ShardError.slab_layout_mismatch;
                } else {
                    assert(item.slab != no_slab);
                }
                if (item.slab >= self.hot_slabs) {
                    // Cold: the tier holds it in pinned host memory, so the
                    // device arena has no slot for it.
                    return;
                } else {
                    assert(item.slab < self.hot_slabs);
                }
                if (item.slab != self.slab) {
                    try self.openSlab(item.slab);
                } else {
                    assert(item.slab == self.slab);
                }
                const base = self.fixed_region_bytes +
                    @as(u64, item.slab) * self.slab_stride_bytes;
                const slot = std.mem.alignForward(
                    u64,
                    base + self.slab_cursor,
                    alignment_bytes,
                );
                self.slab_cursor = slot + bytes - base;
                assert(self.slab_cursor <= self.slab_stride_bytes);
                try self.append(item, slot, item.slab);
            },
            .mapping => {
                // Gathered from the mapping: not an extent.
            },
        }
    }

    /// Starts a new slab; the walk visits slabs in ascending order.
    fn openSlab(self: *ExtentSink, slab: u32) !void {
        assert(slab != no_slab);
        assert(slab < self.hot_slabs);

        if (self.slab != no_slab and slab <= self.slab) {
            return ShardError.extent_out_of_order;
        } else {
            assert(self.slab == no_slab or slab > self.slab);
        }
        self.slab = slab;
        self.slab_cursor = 0;
    }

    fn append(self: *ExtentSink, item: Item, slot: u64, slab: u32) !void {
        assert(item.rows > 0);
        assert(slot % alignment_bytes == 0);

        try self.extents.append(.{
            .ref = item.ref,
            .role = item.role,
            .layer = item.layer,
            .first_row = item.first_row,
            .rows = item.rows,
            .slot = slot,
            .slab = slab,
        });
    }
};

/// Builds one rank's load list: the fixed region, then the hot slabs.
fn collectExtents(
    allocator: std.mem.Allocator,
    subject: *const manifest.Manifest,
    rank: u32,
    accounting: *const Accounting,
    slab_bytes_total: u64,
    hot_slabs: u32,
) ![]Extent {
    assert(rank < rank_count);
    assert(hot_slabs <= accounting.slab_count);

    var collector = ExtentSink{
        .extents = std.ArrayList(Extent).init(allocator),
        .fixed_region_bytes = std.mem.alignForward(
            u64,
            accounting.fixed_bytes,
            placement.page_bytes,
        ),
        .slab_stride_bytes = placement.slabStrideBytes(slab_bytes_total),
        .hot_slabs = hot_slabs,
    };
    errdefer collector.extents.deinit();
    try walkRank(subject, rank, collector.sink());
    // The fixed region holds exactly the bytes the accounting pass summed, plus
    // the padding between tensors.
    assert(collector.fixed_cursor <= collector.fixed_region_bytes);
    if (hot_slabs > 0) {
        assert(collector.slab == hot_slabs - 1);
    } else {
        assert(collector.slab == no_slab);
    }
    return collector.extents.toOwnedSlice();
}

// --- the plan ---------------------------------------------------------------

/// One rank's shard: what it opens, where it goes, and how much is left over.
pub const RankShard = struct {
    rank: u32,
    accounting: Accounting,
    /// The measured-VRAM decision, from the same rule the planned table uses.
    placement: placement.RankPlacement,
    extents: []Extent,
    /// Bytes of the device arena: the fixed region plus the hot slabs.
    arena_bytes: u64,
    /// Bytes actually copied into it; the difference from `arena_bytes` is
    /// alignment padding.
    resident_bytes: u64,

    pub fn free(self: *RankShard, allocator: std.mem.Allocator) void {
        assert(self.rank < rank_count);
        allocator.free(self.extents);
        self.extents = &.{};
    }
};

/// Every rank's shard, plus the checkpoint totals the report prints.
pub const Shards = struct {
    ranks: [rank_count]RankShard,
    /// One copy of every text tensor the engine holds on a device, hot or cold.
    device_bytes_checkpoint: u64,
    /// One copy of every gather-only tensor, shared by every rank's mapping.
    mapping_bytes_checkpoint: u64,
    /// Bytes the engine does not load (MTP head, vision tower).
    deferred_bytes: u64,
    /// Bytes of the checkpoint's tensor payload.
    tensor_bytes: u64,
    /// The shipped expert slab against the planned one: the difference the tier
    /// has to absorb.
    slab_bytes: u64,
    slab_bytes_planned: u64,
    slab_stride_bytes: u64,
    slab_count: u32,
    context_tokens: u32,

    pub fn free(self: *Shards, allocator: std.mem.Allocator) void {
        assert(self.ranks.len == rank_count);
        for (&self.ranks) |*rank| {
            rank.free(allocator);
        }
    }
};

/// Sizes every rank from the checkpoint's own bytes and that rank's measured
/// free VRAM. Refuses, rather than degrades, when the fixed set cannot fit.
pub fn plan(
    allocator: std.mem.Allocator,
    loaded: *const manifest.Loaded,
    vrams: [rank_count]u64,
    context_tokens: u32,
) !*Shards {
    assert(loaded.files.len > 0);
    assert(context_tokens > 0);

    const subject = &loaded.manifest;
    const self = try allocator.create(Shards);
    errdefer allocator.destroy(self);
    self.* = .{
        .ranks = undefined,
        .device_bytes_checkpoint = 0,
        .mapping_bytes_checkpoint = 0,
        .deferred_bytes = subject.total(.deferred).bytes,
        .tensor_bytes = loaded.tensorBytesTotal(),
        .slab_bytes = try slabBytes(subject),
        .slab_bytes_planned = placement.expertSlabBytes(),
        .slab_stride_bytes = 0,
        .slab_count = 0,
        .context_tokens = context_tokens,
    };
    var ready: u32 = 0;
    errdefer for (self.ranks[0..ready]) |*rank| rank.free(allocator);
    for (0..rank_count) |index| {
        self.ranks[index] = try planRank(
            allocator,
            subject,
            @intCast(index),
            vrams[index],
            context_tokens,
            self.slab_bytes,
        );
        ready += 1;
    }
    assert(ready == rank_count);
    // The four shards are shaped symmetrically, so the totals must be equal.
    const first = self.ranks[0].accounting;
    for (&self.ranks, 0..) |*rank, index| {
        if (rank.accounting.device_bytes_all != first.device_bytes_all or
            rank.accounting.mapping_bytes != first.mapping_bytes)
        {
            std.debug.print(
                "shard: rank {d} accounts for a different checkpoint total\n",
                .{index},
            );
            return ShardError.rank_bytes_disagree;
        } else {
            assert(rank.accounting.slab_count == first.slab_count);
        }
    }
    self.device_bytes_checkpoint = first.device_bytes_all;
    self.mapping_bytes_checkpoint = first.mapping_bytes;
    self.slab_count = first.slab_count;
    self.slab_stride_bytes = placement.slabStrideBytes(self.slab_bytes);
    return self;
}

/// Sizes one rank: account, then let the placement decide hot against cold.
fn planRank(
    allocator: std.mem.Allocator,
    subject: *const manifest.Manifest,
    rank: u32,
    vram_bytes: u64,
    context_tokens: u32,
    slab_bytes_total: u64,
) !RankShard {
    assert(rank < rank_count);
    assert(vram_bytes > 0);

    const accounting = try account(subject, rank);
    const record = try placement.placeMeasured(.{
        .vram_bytes = vram_bytes,
        .fixed_bytes = accounting.fixed_bytes,
        .slab_bytes = slab_bytes_total,
        .slab_count = accounting.slab_count,
        .context_tokens = context_tokens,
    });
    const extents = try collectExtents(
        allocator,
        subject,
        rank,
        &accounting,
        slab_bytes_total,
        record.hot_slab_count,
    );
    errdefer allocator.free(extents);
    const stride = placement.slabStrideBytes(slab_bytes_total);
    const arena_bytes = std.mem.alignForward(
        u64,
        accounting.fixed_bytes,
        placement.page_bytes,
    ) + @as(u64, record.hot_slab_count) * stride;
    if (arena_bytes > vram_bytes) {
        // `placeMeasured` guarantees weights + kv + state fits; the arena is
        // the weights alone, so this cannot trip unless the arithmetic drifted.
        return placement.PlacementError.rank_exceeds_vram;
    } else {
        assert(arena_bytes <= vram_bytes);
    }
    var resident: u64 = 0;
    for (extents) |extent| {
        resident += extent.rows * rowBytes(extent.ref);
    }
    return .{
        .rank = rank,
        .accounting = accounting,
        .placement = record,
        .extents = extents,
        .arena_bytes = arena_bytes,
        .resident_bytes = resident,
    };
}

/// The refusal: every rank under its measured free VRAM, and a tier where the
/// checkpoint needs one. The same function `placement.verify` runs.
pub fn verify(
    subject: *const Shards,
    dram_tier_enabled: bool,
) placement.PlacementError!void {
    assert(subject.ranks.len == rank_count);
    assert(subject.context_tokens > 0);

    var records: [rank_count]placement.RankPlacement = undefined;
    for (&subject.ranks, 0..) |*rank, index| {
        records[index] = rank.placement;
    }
    try placement.verifyRanks(records[0..], dram_tier_enabled);
}

/// Bits per weight of one shipped slab, for the report.
fn slabBitsPerWeight(subject: *const Shards) struct { whole: u64, fraction: u64 } {
    assert(subject.slab_bytes > 0);
    // Three 640 x 2560 matrices of four-bit weights per slab.
    const values: u64 = 3 * placement.expert_intermediate * placement.expert_hidden;
    const scaled = subject.slab_bytes * 800 / values;
    return .{ .whole = scaled / 100, .fraction = scaled % 100 };
}

/// Writes the shard report: per rank what it opens, what it holds on the device
/// and what is left, then the checkpoint totals.
pub fn render(subject: *const Shards, writer: anytype) !void {
    assert(subject.ranks.len == rank_count);
    assert(subject.tensor_bytes > 0);

    const bits = slabBitsPerWeight(subject);
    try writer.print(
        "shard plan: {d} ranks, context {d} tokens, shipped experts " ++
            "{d}.{d:0>2} b/p, planned {d}.{d:0>2} b/p\n",
        .{
            rank_count,
            subject.context_tokens,
            bits.whole,
            bits.fraction,
            subject.slab_bytes_planned * 800 /
                (3 * placement.expert_intermediate * placement.expert_hidden) /
                100,
            subject.slab_bytes_planned * 800 /
                (3 * placement.expert_intermediate * placement.expert_hidden) %
                100,
        },
    );
    for (&subject.ranks) |*rank| {
        try renderRank(rank, subject, writer);
    }
    try writer.print(
        "shard totals: {d} B tensor payload | {d} B text on device | " ++
            "{d} B gathered from the mapping | {d} B deferred\n",
        .{
            subject.tensor_bytes,
            subject.device_bytes_checkpoint,
            subject.mapping_bytes_checkpoint,
            subject.deferred_bytes,
        },
    );
    try writer.print(
        "        slab {d} B on disk, {d} B stride in an arena, " ++
            "{d} slabs against {d} B planned\n",
        .{
            subject.slab_bytes,
            subject.slab_stride_bytes,
            subject.slab_count,
            subject.slab_bytes_planned,
        },
    );
}

/// One rank's lines.
fn renderRank(
    rank: *const RankShard,
    subject: *const Shards,
    writer: anytype,
) !void {
    assert(rank.rank < rank_count);
    assert(rank.arena_bytes <= rank.placement.vram_bytes);

    const record = &rank.placement;
    const accounting = &rank.accounting;
    try writer.print(
        "  rank {d}: vram {d} B | arena {d} B | headroom {d} B | " ++
            "{d} extents, {d} B copied\n",
        .{
            rank.rank,
            record.vram_bytes,
            rank.arena_bytes,
            record.vram_bytes - record.device_bytes,
            rank.extents.len,
            rank.resident_bytes,
        },
    );
    try writer.print(
        "    fixed {d} B | hot {d} of {d} slabs ({d} B) | cold {d} slabs " ++
            "-> tier {d} B | kv fp8 {d} B | state {d} B\n",
        .{
            accounting.fixed_bytes,
            record.hot_slab_count,
            accounting.slab_count,
            @as(u64, record.hot_slab_count) * subject.slab_stride_bytes,
            record.cold_slab_count,
            record.tier_bytes,
            record.kv_bytes,
            record.state_bytes,
        },
    );
    try writer.print(
        "    mapping (gather-only) {d} B per rank\n",
        .{accounting.mapping_bytes},
    );
    for (manifest.all_roles) |role| {
        const total = accounting.per_role[@intFromEnum(role)];
        if (total.count == 0) {
            continue;
        } else {
            assert(total.count > 0);
        }
        try writer.print(
            "      {s: <24} {d: >7} tensors {d: >13} B {s}\n",
            .{
                @tagName(role),
                total.count,
                total.bytes,
                @tagName(shardingFor(role)),
            },
        );
    }
}

// --- the load ---------------------------------------------------------------

/// One primary context per rank and the free VRAM each reported before
/// anything was allocated.
pub const Devices = struct {
    contexts: [rank_count]cuda_context.Context = undefined,
    names: [rank_count][]const u8 = .{ "", "", "", "" },
    free_bytes: [rank_count]u64 = undefined,
    ready: u32 = 0,

    /// Opens one primary context per rank and measures what is free. Null when
    /// there is no driver or not enough cards: the caller skips, it does not
    /// fail a machine that has no GPUs.
    pub fn open() !?Devices {
        assert(rank_count > 0);
        assert(rank_count <= cuda_driver.device_count_max);

        var devices: [cuda_driver.device_count_max]cuda_driver.Device =
            undefined;
        const count = cuda_driver.query_devices(devices[0..]) catch |err| switch (err) {
            error.driver_library_missing => return null,
            else => return err,
        };
        if (count < rank_count) {
            return null;
        } else {
            assert(count >= rank_count);
        }
        var self = Devices{};
        errdefer self.close();
        for (0..rank_count) |index| {
            const rank: u32 = @intCast(index);
            self.contexts[rank] = try cuda_context.Context.retain(
                &devices[rank],
            );
            self.names[rank] = devices[rank].name_slice();
            self.ready += 1;
            try self.contexts[rank].set_current();
            self.free_bytes[rank] = try self.contexts[rank].free_bytes();
        }
        assert(self.ready == rank_count);
        return self;
    }

    pub fn close(self: *Devices) void {
        assert(self.ready <= rank_count);
        for (self.contexts[0..self.ready]) |*context| {
            context.release();
        }
        self.ready = 0;
    }
};

/// What one rank's load did.
pub const RankLoad = struct {
    bytes: u64 = 0,
    extents: u64 = 0,
    elapsed_ns: u64 = 0,
    /// Sampled witness over the bytes that were copied, so a short or misplaced
    /// read is caught without hashing 13 GB per rank.
    witness: u64 = 0,
    free_after_bytes: u64 = 0,
};

/// The whole load.
pub const LoadReport = struct {
    ranks: [rank_count]RankLoad = [_]RankLoad{.{}} ** rank_count,
    bytes_total: u64 = 0,
    elapsed_ns: u64 = 0,
};

/// Copies every rank's shard into one arena per rank and reports the bytes, the
/// time and what is left free.
///
/// One arena per rank, all four held at once: this is the fleet's resident set,
/// not a per-card check. Every extent is read out of the read-only mapping and
/// copied through a pinned staging buffer, so the checkpoint's read rate and the
/// link rate are both measured rather than assumed.
pub fn load(
    allocator: std.mem.Allocator,
    subject: *const Shards,
    loaded: *const manifest.Loaded,
    devices: *Devices,
    staging_bytes: u64,
) !LoadReport {
    assert(subject.ranks.len == rank_count);
    assert(devices.ready == rank_count);
    assert(staging_bytes > 0);

    var arenas: [rank_count]cuda_context.DeviceBuffer = undefined;
    var allocated: u32 = 0;
    errdefer for (arenas[0..allocated]) |*arena| arena.free();
    for (&subject.ranks, 0..) |*shard, index| {
        try devices.contexts[index].set_current();
        arenas[index] = try cuda_context.DeviceBuffer.allocate(
            std.mem.alignForward(u64, shard.arena_bytes, placement.page_bytes),
        );
        allocated += 1;
    }
    assert(allocated == rank_count);

    var report = LoadReport{};
    const started = std.time.nanoTimestamp();
    for (&subject.ranks, 0..) |*shard, index| {
        const rank: u32 = @intCast(index);
        report.ranks[rank] = try loadRank(
            allocator,
            shard,
            loaded,
            &devices.contexts[rank],
            &arenas[rank],
            staging_bytes,
        );
        report.bytes_total += report.ranks[rank].bytes;
        report.ranks[rank].free_after_bytes =
            try devices.contexts[rank].free_bytes();
    }
    report.elapsed_ns = @intCast(std.time.nanoTimestamp() - started);
    for (arenas[0..allocated]) |*arena| {
        arena.free();
    }
    assert(report.bytes_total > 0);
    return report;
}

/// One rank's arena: copy every extent, keep a witness.
fn loadRank(
    allocator: std.mem.Allocator,
    shard: *const RankShard,
    loaded: *const manifest.Loaded,
    context: *const cuda_context.Context,
    arena: *const cuda_context.DeviceBuffer,
    staging_bytes: u64,
) !RankLoad {
    assert(allocator.vtable != undefined);
    assert(shard.extents.len > 0);

    try context.set_current();
    const stream = try cuda_context.Stream.create();
    defer stream.destroy();
    const staging = try cuda_context.PinnedHostBuffer.allocate(
        std.mem.alignForward(u64, staging_bytes, placement.page_bytes),
    );
    defer staging.free();
    var result = RankLoad{};
    const started = std.time.nanoTimestamp();
    for (shard.extents) |extent| {
        result.bytes += try copyExtent(loaded, extent, arena, &stream, &staging);
        result.extents += 1;
        result.witness = witnessOf(result.witness, loaded, extent);
    }
    try stream.synchronize();
    result.elapsed_ns = @intCast(std.time.nanoTimestamp() - started);
    assert(result.bytes == shard.resident_bytes);
    return result;
}

/// Copies one extent in staging-sized chunks: mapping -> pinned -> device.
fn copyExtent(
    loaded: *const manifest.Loaded,
    extent: Extent,
    arena: *const cuda_context.DeviceBuffer,
    stream: *const cuda_context.Stream,
    staging: *const cuda_context.PinnedHostBuffer,
) !u64 {
    assert(extent.rows > 0);
    assert(arena.pointer != 0);

    const row_bytes = rowBytes(extent.ref);
    const source_start = extent.ref.file_offset + extent.first_row * row_bytes;
    const bytes = extent.rows * row_bytes;
    assert(source_start + bytes <=
        extent.ref.file_offset + extent.ref.byte_len);
    assert(extent.slot + bytes <= arena.bytes);
    const window = staging.slice();
    assert(window.len > 0);
    var copied: u64 = 0;
    while (copied < bytes) {
        const chunk = @min(bytes - copied, window.len);
        assert(chunk > 0);
        const to = extent.slot + copied;
        try readMapping(
            loaded,
            extent.ref.file_index,
            source_start + copied,
            window[0..chunk],
        );
        try stream.copy_h_to_d(
            &cuda_context.DeviceBuffer{ .pointer = arena.pointer + to, .bytes = chunk },
            &cuda_context.PinnedHostBuffer{ .pointer = window.ptr, .bytes = chunk },
            chunk,
        );
        copied += chunk;
    }
    assert(copied == bytes);
    return bytes;
}

/// Reads `count` bytes at absolute offset `from` out of one mapped file.
fn readMapping(
    loaded: *const manifest.Loaded,
    file_index: u32,
    from: u64,
    into: []u8,
) !void {
    assert(file_index < loaded.files.len);
    assert(into.len > 0);
    assert(into.len <= read_chunk_bytes);

    const file = &loaded.files[file_index];
    const start: usize = @intCast(from);
    if (start + into.len > file.bytes.len) {
        return ShardError.tensor_bytes_disagree;
    } else {
        assert(start + into.len <= file.bytes.len);
    }
    // Touching the mapping is the read: the page fault brings the bytes in.
    @memcpy(into, file.bytes[start..][0..into.len]);
}

/// A sampled witness over one extent's bytes: its first and last byte and one
/// in the middle. Not a checksum of the file; a witness that the range copied
/// is the range planned.
fn witnessOf(
    seed: u64,
    loaded: *const manifest.Loaded,
    extent: Extent,
) u64 {
    assert(extent.rows > 0);
    assert(extent.ref.file_index < loaded.files.len);

    const row_bytes = rowBytes(extent.ref);
    const start = extent.ref.file_offset + extent.first_row * row_bytes;
    const bytes = extent.rows * row_bytes;
    const file = &loaded.files[extent.ref.file_index];
    const first: usize = @intCast(start);
    const last: usize = @intCast(start + bytes - 1);
    var witness = seed;
    for ([_]usize{ first, last, (first + last) / 2 }) |at| {
        assert(at < file.bytes.len);
        witness = witness *% 1_099_511_628_211 +% file.bytes[at];
    }
    assert(last >= first);
    return witness;
}

// --- tests ------------------------------------------------------------------

test "the policy maps every role onto the placement's sharding" {
    try std.testing.expectEqual(
        @as(usize, placement.rank_count),
        @as(usize, rank_count),
    );
    // Every role is decided; none falls through to a default. The counts are
    // pinned so a role added without a decision changes this test.
    var replicated: u32 = 0;
    var parallel: u32 = 0;
    var host: u32 = 0;
    for (manifest.all_roles) |role| {
        switch (shardingFor(role)) {
            .replicated => replicated += 1,
            .tensor_parallel, .expert_parallel => parallel += 1,
            .host_resident => {
                if (role == .deferred) {
                    // Counted, never loaded.
                } else {
                    host += 1;
                }
            },
        }
    }
    try std.testing.expectEqual(@as(u32, 11), host);
    try std.testing.expectEqual(@as(u32, 3), parallel);
    try std.testing.expectEqual(@as(u32, 8), replicated);
}

test "a row slice is contiguous, whole and disjoint across the ranks" {
    const ref = manifest.TensorRef{
        .dims = .{ .values = .{ 2_560, 6_144, 0, 0, 0 }, .rank = 2 },
        .byte_len = 2 * 2_560 * 6_144,
    };
    try std.testing.expectEqual(@as(u64, 12_288), rowBytes(ref));
    var covered: u64 = 0;
    var previous_end: u64 = 0;
    for (0..rank_count) |index| {
        const slice = (try shardRows(ref, .tensor_parallel, @intCast(index))).?;
        try std.testing.expectEqual(@as(u64, 640), slice.rows);
        try std.testing.expectEqual(previous_end, slice.first);
        previous_end = slice.first + slice.rows;
        covered += slice.rows;
    }
    try std.testing.expectEqual(@as(u64, 2_560), covered);
    for (0..rank_count) |index| {
        const slice = (try shardRows(ref, .replicated, @intCast(index))).?;
        try std.testing.expectEqual(@as(u64, 2_560), slice.rows);
    }
    try std.testing.expectEqual(
        @as(?RowSlice, null),
        try shardRows(ref, .host_resident, 0),
    );
    var ragged = ref;
    ragged.dims.values[0] = 2_561;
    ragged.byte_len = 2 * 2_561 * 6_144;
    try std.testing.expectError(
        ShardError.shard_not_divisible,
        shardRows(ragged, .tensor_parallel, 0),
    );
}

/// The checkpoint's directory, or null when it is not on this machine.
fn checkpointDirectory(allocator: std.mem.Allocator) !?[]u8 {
    const variable = "INFERNO_MODEL_DIR";
    // No hard-coded default: a 132 GB artifact must never be scanned by the default
    // test suite. This test parses a 31 MB index with 299,545 tensors plus ten shard
    // headers, which made `zig build test` look hung. Opt in explicitly.
    const directory = std.posix.getenv(variable) orelse {
        std.debug.print(
            "SKIP: set INFERNO_MODEL_DIR to run the real-checkpoint tests\n",
            .{},
        );
        return null;
    };
    assert(directory.len > 0);
    const index = try std.fs.path.join(
        allocator,
        &.{ directory, "model.safetensors.index.json" },
    );
    defer allocator.free(index);
    const handle = std.fs.cwd().openFile(index, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    handle.close();
    return try allocator.dupe(u8, directory);
}

test "the real checkpoint's shard plan is sized against measured VRAM" {
    const allocator = std.testing.allocator;
    const maybe_directory = try checkpointDirectory(allocator);
    if (maybe_directory == null) {
        std.debug.print("SKIP: no checkpoint (set INFERNO_MODEL_DIR)\n", .{});
        return error.SkipZigTest;
    }
    const directory = maybe_directory.?;
    defer allocator.free(directory);

    var loaded = try manifest.build(allocator, directory);
    defer manifest.unload(&loaded);
    try std.testing.expectEqual(
        @as(u64, 132_639_846_394),
        loaded.tensorBytesTotal(),
    );

    // Measured, not assumed: one context per card, free VRAM as it is now.
    var devices = (try Devices.open()) orelse {
        std.debug.print(
            "SKIP: the driver does not report {d} GPUs\n",
            .{rank_count},
        );
        return error.SkipZigTest;
    };
    defer devices.close();

    const plan_shards = try plan(allocator, &loaded, devices.free_bytes, 4_096);
    defer {
        plan_shards.free(allocator);
        allocator.destroy(plan_shards);
    }
    try verify(plan_shards, true);
    // Without a tier the real checkpoint cannot start on this machine: that
    // refusal is the measured situation, not a preference.
    try std.testing.expectError(
        placement.PlacementError.dram_tier_required,
        verify(plan_shards, false),
    );

    // The shipped slab is the NVFP4 one, not the planned per-32 requant.
    try std.testing.expectEqual(@as(u64, 2_764_824), plan_shards.slab_bytes);
    try std.testing.expectEqual(
        @as(u64, 2_611_212),
        plan_shards.slab_bytes_planned,
    );
    try std.testing.expectEqual(
        @as(u64, 2_768_896),
        plan_shards.slab_stride_bytes,
    );
    try std.testing.expectEqual(@as(u32, 6_144), plan_shards.slab_count);

    // Every rank: one copy of what is replicated, a quarter of what is not,
    // and its own expert shard. Pinned to the checkpoint's own bytes.
    for (&plan_shards.ranks) |*rank| {
        try std.testing.expectEqual(
            @as(u64, 6_585_173_760),
            rank.accounting.fixed_bytes,
        );
        try std.testing.expectEqual(
            @as(u64, 16_987_078_656),
            rank.accounting.device_bytes_all - rank.accounting.fixed_bytes,
        );
        try std.testing.expectEqual(
            @as(u64, 52_471_644_442),
            rank.accounting.mapping_bytes,
        );
        try std.testing.expect(rank.arena_bytes <= rank.placement.vram_bytes);
        try std.testing.expectEqual(
            rank.placement.hot_slab_count + rank.placement.cold_slab_count,
            @as(u32, 6_144),
        );
        try std.testing.expect(rank.resident_bytes <= rank.arena_bytes);
        try std.testing.expect(
            rank.resident_bytes > rank.accounting.fixed_bytes / 2,
        );
        // The tier has to hold the rest, and it is not small.
        try std.testing.expect(rank.placement.tier_bytes > 1_000_000_000);
    }
    try std.testing.expectEqual(
        @as(u64, 129_043_957_786),
        plan_shards.device_bytes_checkpoint,
    );
    try std.testing.expectEqual(
        @as(u64, 3_595_888_608),
        plan_shards.deferred_bytes,
    );

    const buffer = try allocator.alloc(u8, 64 * 1024);
    defer allocator.free(buffer);
    var stream = std.io.fixedBufferStream(buffer);
    try render(plan_shards, stream.writer());
    const text = stream.getWritten();
    try std.testing.expect(
        std.mem.indexOf(u8, text, "shard totals:") != null,
    );
    try std.testing.expect(std.mem.indexOf(u8, text, "rank 3:") != null);
    try std.testing.expect(
        std.mem.indexOf(u8, text, "4.50 b/p") != null,
    );
    std.debug.print("{s}", .{text});
}

test "the real shard plan loads into the four cards and reports headroom" {
    const allocator = std.testing.allocator;
    const enabled = std.posix.getenv("INFERNO_SHARD_LOAD") orelse "";
    if (!std.mem.eql(u8, enabled, "1")) {
        std.debug.print(
            "SKIP: full shard load (set INFERNO_SHARD_LOAD=1; reads ~52 GB)\n",
            .{},
        );
        return error.SkipZigTest;
    }
    const maybe_directory = try checkpointDirectory(allocator);
    if (maybe_directory == null) {
        std.debug.print("SKIP: no checkpoint (set INFERNO_MODEL_DIR)\n", .{});
        return error.SkipZigTest;
    }
    const directory = maybe_directory.?;
    defer allocator.free(directory);

    var loaded = try manifest.build(allocator, directory);
    defer manifest.unload(&loaded);
    var devices = (try Devices.open()) orelse {
        std.debug.print(
            "SKIP: the driver does not report {d} GPUs\n",
            .{rank_count},
        );
        return error.SkipZigTest;
    };
    defer devices.close();

    const plan_shards = try plan(allocator, &loaded, devices.free_bytes, 4_096);
    defer {
        plan_shards.free(allocator);
        allocator.destroy(plan_shards);
    }
    try verify(plan_shards, true);
    const report = try load(allocator, plan_shards, &loaded, &devices, read_chunk_bytes);
    var expected: u64 = 0;
    for (&plan_shards.ranks) |*rank| {
        expected += rank.resident_bytes;
    }
    try std.testing.expectEqual(expected, report.bytes_total);
    for (&report.ranks, 0..) |*rank, index| {
        try std.testing.expect(rank.bytes > 0);
        try std.testing.expect(rank.elapsed_ns > 0);
        try std.testing.expect(rank.witness != 0);
        try std.testing.expect(rank.free_after_bytes < devices.free_bytes[index]);
    }
    const gigabytes = @as(f64, @floatFromInt(report.bytes_total)) / 1e9;
    const seconds = @as(f64, @floatFromInt(report.elapsed_ns)) / 1e9;
    std.debug.print(
        "shard load: {d} B in {d:.3} s = {d:.2} GB/s across {d} ranks\n",
        .{ report.bytes_total, seconds, gigabytes / seconds, rank_count },
    );
    for (&report.ranks, 0..) |*rank, index| {
        std.debug.print(
            "  rank {d}: {d} B in {d:.3} s, {d} extents, witness {x}, " ++
                "{d} B free after ({d} B headroom)\n",
            .{
                index,
                rank.bytes,
                @as(f64, @floatFromInt(rank.elapsed_ns)) / 1e9,
                rank.extents,
                rank.witness,
                rank.free_after_bytes,
                plan_shards.ranks[index].placement.vram_bytes -
                    plan_shards.ranks[index].placement.device_bytes,
            },
        );
    }
}
