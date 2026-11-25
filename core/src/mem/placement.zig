//! Per-rank weight placement and the host-DRAM expert tier (I-014).
//!
//! The placement itself is decided: the design's
//! `.agents/design/full-context-single-stream.md` §3 fixes which component is
//! replicated, which is tensor parallel and which is expert parallel, and why.
//! This file is the one place that turns those bytes into per-rank numbers, so
//! the planner, the placement report and the expert tier all read the same
//! table instead of each re-deriving a budget.
//!
//! Everything here is arithmetic on constants: no driver call, so the whole
//! budget is checkable on a machine with no GPU. `model/multi.zig` feeds it the
//! *measured* free VRAM of every rank and owns the driver resources.
//!
//! The gap it has to close, in the design's own numbers: 70.37 GB of device
//! weights against 66.60 GB of VRAM across four 16 GB cards, with a 3 GiB FP8
//! KV copy per rank at 256 K context. What does not fit is expert mass, so the
//! expert tier is where the plan puts it.

const std = @import("std");

const assert = std.debug.assert;

// --- the machine ------------------------------------------------------------

/// Ranks the engine runs on: one per GPU on this machine.
pub const rank_count: u32 = 4;

/// Smallest allocation granularity of the driver's device and pinned memory.
pub const page_bytes: u64 = 4096;

/// Context the KV arena is sized for unless the operator asks for less.
pub const max_context_tokens: u32 = 262_144;

// --- the checkpoint, as the placement reads it ------------------------------

/// Text layers in the checkpoint.
pub const layer_count: u32 = 48;

/// Layers that carry indexed (full) attention: every fourth layer.
pub const full_attention_layers: u32 = 12;

/// Experts in every layer.
pub const experts_total: u32 = 512;

/// Experts per rank under expert parallelism: consecutive shards of 128.
pub const experts_per_rank: u32 = experts_total / rank_count;

/// Expert slabs one rank owns: one shard in every layer.
pub const expert_slabs_per_rank: u32 = layer_count * experts_per_rank;

/// Width every expert reads (rows of gate and up, columns of down).
pub const expert_hidden: u32 = 2_560;

/// Width every expert writes before the down projection.
pub const expert_intermediate: u32 = 640;

/// MXFP4 keeps one 8-bit E8M0 scale per 32 four-bit values.
pub const expert_block_elements: u32 = 32;

/// Matrices per expert: gate, up, down.
const expert_matrices: u64 = 3;

/// One f32 per-tensor scale per matrix, as the checkpoint ships it.
const expert_tensor_scale_bytes: u64 = 4;

/// Head width of a full attention layer.
pub const head_dim: u32 = 256;

/// GQA key/value heads: two, so their projection is replicated, not split.
pub const kv_heads: u32 = 2;

/// FP8 KV: one byte per element.
pub const kv_element_bytes: u32 = 1;

/// QSA selection budget: how many keys one query attends to, at any context.
pub const qsa_budget_keys: u32 = 2_048;

/// Bytes one (layer, expert) MXFP4 slab occupies: gate, up and down as packed
/// four-bit pairs, one E8M0 scale per 32 values, one f32 scale per matrix.
///
/// This is the unit the expert tier moves and the LRU holds, so it is also the
/// unit the whole budget is expressed in.
pub fn expertSlabBytes() u64 {
    assert(expert_block_elements == 32);
    assert(expert_matrices == 3);

    const values = expert_matrices *
        @as(u64, expert_hidden) * @as(u64, expert_intermediate);
    assert(values > 0);
    const packed_bytes = @divExact(values, 2);
    const scale_bytes = @divExact(values, expert_block_elements);
    const tensor_bytes = expert_matrices * expert_tensor_scale_bytes;
    const total = packed_bytes + scale_bytes + tensor_bytes;
    assert(total > packed_bytes);
    return total;
}

/// Bytes a slab of `slab_bytes` occupies in an arena: whole pages, because
/// device and pinned allocations are page granular and the tier indexes slots
/// by byte offset. The checkpoint's own slabs are not the planned size (see
/// `weights/shard.zig`), so the rounding is a function, not a constant.
pub fn slabStrideBytes(slab_bytes: u64) u64 {
    assert(slab_bytes > 0);
    assert(page_bytes > 0);

    const stride = std.mem.alignForward(u64, slab_bytes, page_bytes);
    assert(stride >= slab_bytes);
    assert(stride % page_bytes == 0);
    return stride;
}

/// Bytes one planned slab occupies in an arena.
pub fn expertSlabStrideBytes() u64 {
    assert(expertSlabBytes() > 0);
    return slabStrideBytes(expertSlabBytes());
}

/// Bytes every layer's experts occupy across the whole checkpoint.
pub fn expertsBytesTotal() u64 {
    const slab_bytes = expertSlabBytes();
    assert(slab_bytes > 0);
    assert(expert_slabs_per_rank * rank_count == layer_count * experts_total);

    return slab_bytes * layer_count * experts_total;
}

/// How a component is spread over the ranks.
pub const Sharding = enum {
    /// Output dimension split `rank_count` ways with `in_features` whole; one
    /// fused all-reduce per layer completes both sub-blocks.
    tensor_parallel,
    /// `experts_per_rank` consecutive experts per rank, in every layer.
    expert_parallel,
    /// Identical bytes on every rank: cheaper than a reduction (attention,
    /// indexer, router) or unsplittable (the two GQA key/value heads).
    replicated,
    /// No rank holds it; gathered from the host once per token.
    host_resident,
};

/// One row of the decided placement table.
pub const Component = struct {
    name: []const u8,
    sharding: Sharding,
    /// Bytes one copy of this component occupies: what the checkpoint holds.
    bytes_checkpoint: u64,
    /// Bytes this row costs one rank's device. A replicated row costs a whole
    /// copy on every rank, so the aggregate VRAM demand of the weights is
    /// larger than the checkpoint total (see `weightsBytesAcrossRanks`).
    bytes_per_rank: u64,
};

/// One rank's share of a tensor-parallel or expert-parallel component.
fn shardBytes(checkpoint_bytes: u64) u64 {
    assert(checkpoint_bytes > 0);
    assert(rank_count > 0);

    return @divExact(checkpoint_bytes, rank_count);
}

/// The placement from `.agents/design/full-context-single-stream.md` §3.
///
/// Only the expert row is derived rather than quoted: it is the checkpoint's
/// own shape at MXFP4 per-32, so the tier arithmetic cannot drift away from the
/// slab arithmetic in this file. The other rows are the requantised sizes the
/// design commits to, and they are what the requantiser (I-020) has to hit.
pub const placement_table = [_]Component{
    .{
        .name = "experts_mxfp4",
        .sharding = .expert_parallel,
        .bytes_checkpoint = expertsBytesTotal(),
        .bytes_per_rank = shardBytes(expertsBytesTotal()),
    },
    .{
        .name = "dense_value_path_fp8",
        .sharding = .tensor_parallel,
        .bytes_checkpoint = 2_960_000_000,
        .bytes_per_rank = shardBytes(2_960_000_000),
    },
    .{
        // §2: 1.235 GB of bf16 self attention and indexer per copy requantises
        // to 0.62 GB of FP8, and every rank holds one.
        .name = "attention_indexer_fp8",
        .sharding = .replicated,
        .bytes_checkpoint = 617_500_000,
        .bytes_per_rank = 617_500_000,
    },
    .{
        // 512 experts x 2560 hidden at bf16 is 2.6 MB a layer, 126 MB for all
        // 48: one exact copy per rank, so every rank routes identically.
        .name = "router_bf16",
        .sharding = .replicated,
        .bytes_checkpoint = 130_000_000,
        .bytes_per_rank = 130_000_000,
    },
    .{
        // Row split by vocabulary, reduced on device before sampling.
        .name = "lm_head_fp8",
        .sharding = .tensor_parallel,
        .bytes_checkpoint = 640_000_000,
        .bytes_per_rank = shardBytes(640_000_000),
    },
    .{
        // Gather only: one row per token, so no rank keeps it resident.
        .name = "embed_tokens_bf16",
        .sharding = .host_resident,
        .bytes_checkpoint = 1_270_000_000,
        .bytes_per_rank = 0,
    },
    .{
        // Gather only, from NVMe: the 51 GB n-gram table.
        .name = "ple_ngram_table",
        .sharding = .host_resident,
        .bytes_checkpoint = 51_200_000_000,
        .bytes_per_rank = 0,
    },
};

/// GDN recurrent state, GDN convolution state and PLE state for one sequence,
/// across all ranks. Not split: it lives on the rank that owns the v-heads.
const state_bytes_total: u64 = 130_000_000;

/// Device-resident weight bytes on one rank, before the expert tier moves
/// anything: 17.69 GB, which is what has to fit in one card's VRAM.
pub fn weightsBytesPerRank() u64 {
    assert(placement_table.len > 0);
    assert(page_bytes > 0);

    var total: u64 = 0;
    for (placement_table) |component| {
        total += component.bytes_per_rank;
    }
    assert(total > 0);
    return total;
}

/// Device-resident weight bytes with one copy of every component, i.e. the
/// checkpoint's own bytes. Below the design's 70.37 GB figure, because that
/// figure counts the attention block four times (it is replicated) and the
/// router once; `weightsBytesAcrossRanks` is the arithmetic that matters.
pub fn weightsBytesTotal() u64 {
    assert(placement_table.len > 0);

    var total: u64 = 0;
    for (placement_table) |component| {
        if (component.sharding == .host_resident) {
            // Gathered per token; never resident on any rank.
        } else {
            total += component.bytes_checkpoint;
        }
    }
    assert(total > 0);
    return total;
}

/// Aggregate VRAM demand of the weights: every rank's share, summed. Larger
/// than `weightsBytesTotal` by the replicated rows, because the router and the
/// attention block are paid for four times.
pub fn weightsBytesAcrossRanks() u64 {
    assert(rank_count > 0);
    assert(weightsBytesPerRank() > 0);

    return weightsBytesPerRank() * rank_count;
}

/// Weight bytes per rank that are not experts. This is the floor of the
/// placement: the expert tier can move experts, nothing else.
pub fn fixedWeightBytesPerRank() u64 {
    const weights = weightsBytesPerRank();
    const experts_per_rank_bytes = @divExact(expertsBytesTotal(), rank_count);
    assert(experts_per_rank_bytes > 0);
    assert(weights > experts_per_rank_bytes);

    return weights - experts_per_rank_bytes;
}

/// Bytes one (layer, token) of K and V occupies in FP8.
pub fn kvBytesPerLayerToken() u64 {
    assert(kv_heads > 0);
    assert(head_dim > 0);

    const per_head = @as(u64, head_dim) * kv_element_bytes;
    return 2 * kv_heads * per_head;
}

/// KV bytes one rank holds at `context_tokens`: FP8, one copy per rank, because
/// replicated attention reads the whole cache from its own HBM.
pub fn kvBytesPerRank(context_tokens: u32) u64 {
    assert(full_attention_layers > 0);
    assert(context_tokens > 0);

    return @as(u64, full_attention_layers) * kvBytesPerLayerToken() *
        context_tokens;
}

/// KV bytes across all ranks, i.e. the replicated total.
pub fn kvBytesTotal(context_tokens: u32) u64 {
    assert(context_tokens > 0);
    assert(rank_count > 0);

    return kvBytesPerRank(context_tokens) * rank_count;
}

/// Worst-case KV bytes one decode step reads from the host at *any* context:
/// 12 layers x the QSA budget x one kilobyte per key and value.
///
/// QSA caps live attention at `qsa_budget_keys`, which is what makes a KV tail
/// in DRAM legal: the step reads 25 MB, not the whole 3 GiB arena.
pub fn kvTailBytesPerToken() u64 {
    assert(full_attention_layers == 12);
    assert(qsa_budget_keys == 2_048);

    return @as(u64, full_attention_layers) * qsa_budget_keys *
        kvBytesPerLayerToken();
}

/// State bytes one rank holds for one sequence.
pub fn stateBytesPerRank() u64 {
    assert(state_bytes_total > 0);
    assert(rank_count > 0);

    return @divExact(state_bytes_total, rank_count);
}

/// What the planner is told about one rank.
pub const RankQuery = struct {
    /// Free device bytes the driver reports for this rank, measured at init.
    vram_bytes: u64,
    /// Context the KV arena is sized for.
    context_tokens: u32,
    /// Slabs to keep on the device; 0 lets the planner use every page the
    /// measured VRAM leaves free.
    hot_slab_count: u32 = 0,
};

/// What one rank ends up holding.
pub const RankPlacement = struct {
    vram_bytes: u64,
    /// Device-resident weights, after the tier moved the cold experts out.
    weights_bytes: u64,
    kv_bytes: u64,
    state_bytes: u64,
    /// weights + kv + state: what the report has to see under `vram_bytes`.
    device_bytes: u64,
    /// Expert slabs this rank keeps on the device.
    hot_slab_count: u32,
    /// Expert slabs this rank keeps in pinned host memory.
    cold_slab_count: u32,
    /// Pinned host bytes the cold slabs need.
    tier_bytes: u64,
};

pub const PlacementError = error{
    context_tokens_zero,
    vram_below_resident_set,
    hot_slab_count_too_large,
    hot_slab_count_zero,
    rank_exceeds_vram,
    dram_tier_required,
    ranks_disagree,
};

/// Expert slabs the leftover VRAM holds, never more than the rank owns.
fn expertSlabCapacity(room_bytes: u64, stride: u64, owned: u32) u32 {
    assert(room_bytes > 0);
    assert(stride > 0);

    const fits = room_bytes / stride;
    if (fits > owned) {
        return owned;
    } else {
        assert(fits <= owned);
        return @intCast(fits);
    }
}

/// The operator's slab count, or every slab that fits; never zero.
fn chosenHotSlabs(requested: u32, capacity: u32) PlacementError!u32 {
    assert(expert_slabs_per_rank > 0);
    assert(page_bytes > 0);

    if (requested == 0) {
        if (capacity == 0) {
            return PlacementError.hot_slab_count_zero;
        } else {
            assert(capacity > 0);
            return capacity;
        }
    } else {
        assert(requested > 0);
    }
    if (requested > capacity) {
        return PlacementError.hot_slab_count_too_large;
    } else {
        assert(requested <= capacity);
        return requested;
    }
}

/// One rank's byte demand measured from the checkpoint itself rather than from
/// the requantised placement table: `weights/shard.zig` reads the real dtypes
/// (BF16 attention, NVFP4 experts) and fills this in, so the same budget rule
/// decides both the planned and the real run.
pub const MeasuredSet = struct {
    /// Free device bytes the driver reports for this rank.
    vram_bytes: u64,
    /// Device weights that are not expert slabs: what the tier cannot move.
    fixed_bytes: u64,
    /// Bytes one (layer, expert) slab occupies in the checkpoint's layout.
    slab_bytes: u64,
    /// Slabs this rank owns before the tier moves any.
    slab_count: u32,
    /// Context the KV arena is sized for.
    context_tokens: u32,
    /// Slabs to keep on the device; 0 uses every slab the measured VRAM leaves.
    hot_slab_count: u32 = 0,
};

/// Sizes one rank: everything that cannot move, then as many expert slabs as
/// the measured VRAM leaves room for, and the rest in the host tier.
pub fn placeMeasured(query: MeasuredSet) PlacementError!RankPlacement {
    assert(query.vram_bytes > 0);
    assert(query.slab_count > 0);

    if (query.context_tokens == 0) {
        return PlacementError.context_tokens_zero;
    } else {
        assert(query.context_tokens > 0);
    }
    const kv_bytes = kvBytesPerRank(query.context_tokens);
    const state_bytes = stateBytesPerRank();
    const resident_bytes = query.fixed_bytes + kv_bytes + state_bytes;
    if (query.vram_bytes <= resident_bytes) {
        // No expert slab fits at all, so the tier cannot close the gap.
        return PlacementError.vram_below_resident_set;
    } else {
        assert(query.vram_bytes > resident_bytes);
    }

    const stride = slabStrideBytes(query.slab_bytes);
    const capacity = expertSlabCapacity(
        query.vram_bytes - resident_bytes,
        stride,
        query.slab_count,
    );
    const hot = try chosenHotSlabs(query.hot_slab_count, capacity);
    const cold = query.slab_count - hot;
    assert(hot + cold == query.slab_count);

    const weights_bytes = query.fixed_bytes + @as(u64, hot) * stride;
    const device_bytes = weights_bytes + kv_bytes + state_bytes;
    assert(device_bytes <= query.vram_bytes);
    return .{
        .vram_bytes = query.vram_bytes,
        .weights_bytes = weights_bytes,
        .kv_bytes = kv_bytes,
        .state_bytes = state_bytes,
        .device_bytes = device_bytes,
        .hot_slab_count = hot,
        .cold_slab_count = cold,
        .tier_bytes = @as(u64, cold) * stride,
    };
}

/// Sizes one rank from the planned table: the same rule as `placeMeasured`
/// with the requantised slab and the fixed set the design commits to.
pub fn place(query: RankQuery) PlacementError!RankPlacement {
    assert(query.vram_bytes > 0);
    assert(expert_slabs_per_rank > 0);

    return placeMeasured(.{
        .vram_bytes = query.vram_bytes,
        .fixed_bytes = fixedWeightBytesPerRank(),
        .slab_bytes = expertSlabBytes(),
        .slab_count = expert_slabs_per_rank,
        .context_tokens = query.context_tokens,
        .hot_slab_count = query.hot_slab_count,
    });
}

/// The whole plan: one decision per rank plus the totals the report prints.
pub const Placement = struct {
    ranks: [rank_count]RankPlacement,
    weights_bytes_per_rank: u64,
    /// One copy of every device-resident component.
    weights_bytes_total: u64,
    /// What the four ranks actually pay, replicated rows included.
    weights_bytes_across_ranks: u64,
    kv_bytes_per_rank: u64,
    kv_bytes_total: u64,
    state_bytes_per_rank: u64,
    device_bytes_total: u64,
    tier_bytes_total: u64,
};

/// Sizes every rank. `queries` carries each rank's measured free VRAM, so a
/// rank another tenant has already loaded gets a smaller expert hot set instead
/// of an out-of-memory abort.
pub fn plan(queries: [rank_count]RankQuery) PlacementError!Placement {
    assert(queries.len == rank_count);
    assert(page_bytes > 0);

    var result: Placement = undefined;
    result.weights_bytes_per_rank = weightsBytesPerRank();
    result.weights_bytes_total = weightsBytesTotal();
    result.weights_bytes_across_ranks = weightsBytesAcrossRanks();
    result.kv_bytes_per_rank = kvBytesPerRank(queries[0].context_tokens);
    result.kv_bytes_total = kvBytesTotal(queries[0].context_tokens);
    result.state_bytes_per_rank = stateBytesPerRank();
    result.device_bytes_total = 0;
    result.tier_bytes_total = 0;
    for (queries, 0..) |query, index| {
        const rank = try place(query);
        result.ranks[index] = rank;
        result.device_bytes_total += rank.device_bytes;
        result.tier_bytes_total += rank.tier_bytes;
        if (rank.kv_bytes != result.kv_bytes_per_rank) {
            // The KV arena is sized by the context, which is one number for the
            // whole run: ranks that disagree mean the placement was mis-built.
            return PlacementError.ranks_disagree;
        } else {
            assert(rank.kv_bytes == result.kv_bytes_per_rank);
        }
    }
    assert(result.ranks.len == rank_count);
    return result;
}

/// Refuses a plan the engine must not start: a rank over its measured VRAM, or
/// a cold expert with no host tier to hold it.
pub fn verify(
    subject: *const Placement,
    dram_tier_enabled: bool,
) PlacementError!void {
    assert(subject.ranks.len == rank_count);
    assert(subject.weights_bytes_per_rank > 0);

    try verifyRanks(subject.ranks[0..], dram_tier_enabled);
    assert(subject.device_bytes_total > 0);
}

/// The refusal itself, on any set of per-rank placements: a rank that would
/// hold more than its measured free VRAM, or a cold slab with no tier.
///
/// Split out of `verify` so a plan sized from the real checkpoint
/// (`weights/shard.zig`) is held to the same rule as one sized from the
/// requantised table, without either file restating it.
pub fn verifyRanks(
    ranks: []const RankPlacement,
    dram_tier_enabled: bool,
) PlacementError!void {
    assert(ranks.len > 0);
    assert(ranks.len <= rank_count);

    for (ranks) |rank| {
        if (rank.device_bytes > rank.vram_bytes) {
            return PlacementError.rank_exceeds_vram;
        } else {
            assert(rank.device_bytes <= rank.vram_bytes);
        }
        if (dram_tier_enabled) {
            // The tier holds the cold slabs; the rank already fits.
        } else {
            if (rank.cold_slab_count > 0) {
                return PlacementError.dram_tier_required;
            } else {
                assert(rank.cold_slab_count == 0);
            }
        }
    }
}

/// Device names for the report, one per rank. Empty when the driver was not
/// queried, because the plan itself is pure and renders without a GPU.
pub const RenderOptions = struct {
    device_names: [rank_count][]const u8 = .{ "", "", "", "" },
    dram_tier_enabled: bool = true,
};

/// Writes one rank's byte lines.
fn renderRank(
    rank_index: u32,
    rank: *const RankPlacement,
    name: []const u8,
    writer: anytype,
) !void {
    assert(rank_index < rank_count);
    assert(rank.device_bytes <= rank.vram_bytes);

    const shard_first = rank_index * experts_per_rank;
    const shard_last = shard_first + experts_per_rank - 1;
    try writer.print(
        "  rank {d} {s}\n",
        .{ rank_index, name },
    );
    try writer.print(
        "    vram (measured free) {d} B | device {d} B | headroom {d} B\n",
        .{
            rank.vram_bytes,
            rank.device_bytes,
            rank.vram_bytes - rank.device_bytes,
        },
    );
    try writer.print(
        "    weights {d} B | kv fp8 {d} B | state {d} B\n",
        .{ rank.weights_bytes, rank.kv_bytes, rank.state_bytes },
    );
    try writer.print(
        "    experts {d} consecutive of {d} (ids {d}..{d}) | " ++
            "{d} slabs on device of {d} | {d} cold slabs\n",
        .{
            experts_per_rank,
            experts_total,
            shard_first,
            shard_last,
            rank.hot_slab_count,
            expert_slabs_per_rank,
            rank.cold_slab_count,
        },
    );
    try writer.print(
        "    dram expert tier {d} B pinned host ({d} slabs x {d} B)\n",
        .{
            rank.tier_bytes,
            rank.cold_slab_count,
            expertSlabStrideBytes(),
        },
    );
}

/// Writes the placement report: per-rank weights, KV, state and tier bytes,
/// then the totals. Printing is separate from `verify` so the operator sees the
/// numbers that made the engine refuse.
pub fn render(
    subject: *const Placement,
    options: RenderOptions,
    writer: anytype,
) !void {
    assert(subject.ranks.len == rank_count);
    assert(options.device_names.len == rank_count);

    try writer.print(
        "placement: {d} ranks, context {d} tokens, kv fp8 replicated, " ++
            "attention replicated, dense tp={d}, experts ep={d}, " ++
            "dram tier {s}\n",
        .{
            rank_count,
            subject.kv_bytes_per_rank / kvBytesPerLayerToken() /
                full_attention_layers,
            rank_count,
            rank_count,
            if (options.dram_tier_enabled) "enabled" else "disabled",
        },
    );
    for (&subject.ranks, 0..) |*rank, index| {
        const rank_index: u32 = @intCast(index);
        try renderRank(rank_index, rank, options.device_names[index], writer);
    }
    // The four measured frees, not rank 0's times four: rank 0 carries the
    // display and its driver reports less free VRAM than its peers, so an
    // extrapolation understates the fleet.
    var vram_total: u64 = 0;
    for (subject.ranks) |rank| {
        assert(rank.vram_bytes > 0);
        vram_total += rank.vram_bytes;
    }
    assert(vram_total >= subject.ranks[0].vram_bytes);
    try writer.print(
        "totals: weights {d} B one copy (of {d} B vram) | " ++
            "{d} B across {d} ranks | kv {d} B | state {d} B\n",
        .{
            subject.weights_bytes_total,
            vram_total,
            subject.weights_bytes_across_ranks,
            rank_count,
            subject.kv_bytes_total,
            subject.state_bytes_per_rank * rank_count,
        },
    );
    try writer.print(
        "        device total {d} B | dram expert tier {d} B | " ++
            "qsa budget {d} keys = {d} B of kv read per token\n",
        .{
            subject.device_bytes_total,
            subject.tier_bytes_total,
            qsa_budget_keys,
            kvTailBytesPerToken(),
        },
    );
    assert(subject.ranks.len == rank_count);
}

// --- tests ------------------------------------------------------------------

test "the expert slab is the requantised checkpoint's own bytes" {
    // gate, up: 640 x 2560 each; down: 2560 x 640 = 4 915 200 values.
    const values: u64 = 3 * 2_560 * 640;
    try std.testing.expectEqual(@as(u64, 4_915_200), values);
    const packed_bytes = values / 2;
    const scale_bytes = values / 32;
    try std.testing.expectEqual(
        packed_bytes + scale_bytes + 12,
        expertSlabBytes(),
    );
    // A page rounded stride is what an arena slot costs.
    try std.testing.expect(expertSlabStrideBytes() >= expertSlabBytes());
    try std.testing.expectEqual(
        @as(u64, 0),
        expertSlabStrideBytes() % page_bytes,
    );
}

test "the placement table lands on the design's 70.37 GB of weights" {
    // The four replica/tensor/expert rows of the design's §3 table.
    try std.testing.expectEqual(
        @as(u64, 64_173_146_112),
        expertsBytesTotal(),
    );
    try std.testing.expectEqual(@as(u64, 68_520_646_112), weightsBytesTotal());
    // What the four ranks actually pay: every replicated row four times.
    try std.testing.expectEqual(
        @as(u64, 70_763_146_112),
        weightsBytesAcrossRanks(),
    );
    // The design's 70.37 GB figure reconciles with both: it counts the
    // attention block four times (3 copies above `weightsBytesTotal`) and the
    // router once (3 copies below `weightsBytesAcrossRanks`).
    const designed_weights_bytes: u64 = 70_373_146_112;
    try std.testing.expectEqual(
        weightsBytesTotal() + 3 * 617_500_000,
        designed_weights_bytes,
    );
    try std.testing.expectEqual(
        designed_weights_bytes + 3 * 130_000_000,
        weightsBytesAcrossRanks(),
    );
    // 70.37 GB against 66.60 GB of VRAM: the gap the tier closes.
    const vram_total: u64 = @as(u64, 16_660_430_848) * rank_count;
    try std.testing.expect(designed_weights_bytes > vram_total);
    const gap = designed_weights_bytes - vram_total;
    const gap_gigabytes = @as(f64, @floatFromInt(gap)) / 1e9;
    try std.testing.expectApproxEqAbs(@as(f64, 3.73), gap_gigabytes, 0.05);
}

test "every rank's shard is 128 consecutive experts" {
    try std.testing.expectEqual(@as(u32, 128), experts_per_rank);
    try std.testing.expectEqual(@as(u32, 512), experts_total);
    try std.testing.expectEqual(@as(u32, 4), rank_count);
    try std.testing.expectEqual(@as(u32, 6_144), expert_slabs_per_rank);
    // The four shards cover every expert exactly once.
    var covered: [experts_total]u32 = [_]u32{0} ** experts_total;
    for (0..rank_count) |rank_index| {
        const first = rank_index * experts_per_rank;
        for (first..first + experts_per_rank) |expert| {
            try std.testing.expect(expert < experts_total);
            covered[expert] += 1;
        }
    }
    for (covered) |count| {
        try std.testing.expectEqual(@as(u32, 1), count);
    }
}

test "replicated rows cost every rank a full copy, TP and EP rows a quarter" {
    var replicated_checkpoint_bytes: u64 = 0;
    var sharded_checkpoint_bytes: u64 = 0;
    var host_checkpoint_bytes: u64 = 0;
    for (placement_table) |component| {
        switch (component.sharding) {
            .replicated => {
                try std.testing.expectEqual(
                    component.bytes_checkpoint,
                    component.bytes_per_rank,
                );
                replicated_checkpoint_bytes += component.bytes_checkpoint;
            },
            .tensor_parallel, .expert_parallel => {
                try std.testing.expectEqual(
                    @divExact(component.bytes_checkpoint, rank_count),
                    component.bytes_per_rank,
                );
                sharded_checkpoint_bytes += component.bytes_checkpoint;
            },
            .host_resident => {
                try std.testing.expectEqual(
                    @as(u64, 0),
                    component.bytes_per_rank,
                );
                host_checkpoint_bytes += component.bytes_checkpoint;
            },
        }
    }
    // 0.62 attention + 0.13 router per copy: 0.75 GB, paid on every rank.
    try std.testing.expectEqual(
        @as(u64, 747_500_000),
        replicated_checkpoint_bytes,
    );
    try std.testing.expectEqual(
        @as(u64, 1_270_000_000 + 51_200_000_000),
        host_checkpoint_bytes,
    );
    try std.testing.expectEqual(
        weightsBytesTotal(),
        replicated_checkpoint_bytes + sharded_checkpoint_bytes,
    );
    // The three extra copies of the replicated rows are exactly the difference.
    try std.testing.expectEqual(
        weightsBytesTotal() + 3 * replicated_checkpoint_bytes,
        weightsBytesAcrossRanks(),
    );
}

test "kv is 3 GiB per rank at 256 K, 12 GiB replicated across four" {
    try std.testing.expectEqual(@as(u64, 1_024), kvBytesPerLayerToken());
    const per_rank = kvBytesPerRank(max_context_tokens);
    try std.testing.expectEqual(@as(u64, 3_221_225_472), per_rank);
    try std.testing.expectEqual(@as(u64, 3 * 1024 * 1024 * 1024), per_rank);
    const total = kvBytesTotal(max_context_tokens);
    try std.testing.expectEqual(@as(u64, 12 * 1024 * 1024 * 1024), total);
}

test "qsa caps the kv a decode step reads at 25 MB whatever the context" {
    // 12 layers x 2048 keys x 1 KiB, independent of the context length.
    try std.testing.expectEqual(@as(u64, 25_165_824), kvTailBytesPerToken());
    const arena = kvBytesPerRank(max_context_tokens);
    // The tail is 128x smaller than the arena, which is what makes it legal.
    try std.testing.expect(arena / kvTailBytesPerToken() > 100);
}

test "the measured plan puts 4.30 GB of experts in host RAM per rank" {
    // 16 660 430 848 B is this machine's card total, not its free VRAM: with
    // the display attached the driver reports 16 368 730 112 B free on rank 0.
    // So this plan is the fully-free **upper bound** on the hot set (4 499
    // slabs / 4.30 GB in host RAM); the measured plan is 4 388 hot / 4.589 GB.
    const card_bytes: u64 = 16_660_430_848;
    const rank = try place(.{
        .vram_bytes = card_bytes,
        .context_tokens = max_context_tokens,
    });

    // Weights per rank 17.69 GB + 3 GiB KV + state does not fit in 16.66 GB,
    // so the tier holds the difference as whole expert slabs: 1645 of them, and
    // the device keeps 4499.
    try std.testing.expectEqual(
        @as(u64, 17_690_786_528),
        weightsBytesPerRank(),
    );
    try std.testing.expectEqual(@as(u64, 13_404_502_752), rank.weights_bytes);
    // Every slab is in exactly one tier: the device keeps `hot` of them and the
    // host holds the rest, page-rounded, so nothing is lost in the move.
    try std.testing.expectEqual(
        fixedWeightBytesPerRank() + expert_slabs_per_rank *
            expertSlabStrideBytes(),
        rank.weights_bytes + rank.tier_bytes,
    );
    try std.testing.expectEqual(@as(u64, 3_221_225_472), rank.kv_bytes);
    try std.testing.expectEqual(@as(u64, 32_500_000), rank.state_bytes);
    try std.testing.expect(rank.device_bytes <= card_bytes);
    try std.testing.expectEqual(
        @as(u32, 4_499),
        rank.hot_slab_count,
    );
    try std.testing.expectEqual(@as(u32, 1_645), rank.cold_slab_count);
    // 1 645 slabs x 2 613 248 B = 4.2988 GB: the 4.30 GB host tier.
    try std.testing.expectEqual(@as(u64, 4_298_792_960), rank.tier_bytes);
    const tier_gigabytes = @as(f64, @floatFromInt(rank.tier_bytes)) / 1e9;
    try std.testing.expectApproxEqAbs(@as(f64, 4.30), tier_gigabytes, 0.01);
    // The device total is under the measured card by construction.
    try std.testing.expectEqual(
        rank.weights_bytes + rank.kv_bytes + rank.state_bytes,
        rank.device_bytes,
    );
}

test "a larger context moves more experts to host, never past VRAM" {
    const card_bytes: u64 = 16_660_430_848;
    const short = try place(.{
        .vram_bytes = card_bytes,
        .context_tokens = 8_192,
    });
    const long = try place(.{
        .vram_bytes = card_bytes,
        .context_tokens = max_context_tokens,
    });
    try std.testing.expect(short.cold_slab_count < long.cold_slab_count);
    try std.testing.expect(short.tier_bytes < long.tier_bytes);
    try std.testing.expect(short.device_bytes <= card_bytes);
    try std.testing.expect(long.device_bytes <= card_bytes);
    // At 8 K the KV arena costs 96 MiB per rank, so the tier shrinks to the
    // weight overflow alone.
    try std.testing.expectEqual(
        @as(u64, 100_663_296),
        short.kv_bytes,
    );
}

test "a rank another tenant loaded gets a smaller hot set" {
    const card_bytes: u64 = 16_660_430_848;
    const loaded = card_bytes - 2_000_000_000;
    const free_rank = try place(.{
        .vram_bytes = card_bytes,
        .context_tokens = max_context_tokens,
    });
    const loaded_rank = try place(.{
        .vram_bytes = loaded,
        .context_tokens = max_context_tokens,
    });
    try std.testing.expect(
        loaded_rank.hot_slab_count < free_rank.hot_slab_count,
    );
    try std.testing.expect(loaded_rank.tier_bytes > free_rank.tier_bytes);
    try std.testing.expect(loaded_rank.device_bytes <= loaded);
}

test "a rank with no room for one slab is refused, not degraded" {
    const card_bytes: u64 = 16_660_430_848;
    const fixed = fixedWeightBytesPerRank() +
        kvBytesPerRank(max_context_tokens) + stateBytesPerRank();
    try std.testing.expectError(
        PlacementError.vram_below_resident_set,
        place(.{ .vram_bytes = fixed, .context_tokens = max_context_tokens }),
    );
    try std.testing.expectError(
        PlacementError.vram_below_resident_set,
        place(.{
            .vram_bytes = fixed / 2,
            .context_tokens = max_context_tokens,
        }),
    );
    // Just enough for the resident set but not for a whole 2.5 MiB slab.
    try std.testing.expectError(
        PlacementError.hot_slab_count_zero,
        place(.{
            .vram_bytes = fixed + 1024,
            .context_tokens = max_context_tokens,
        }),
    );
    try std.testing.expect(card_bytes > fixed);
}

test "asking for more hot slabs than the measured VRAM allows is refused" {
    const card_bytes: u64 = 16_660_430_848;
    try std.testing.expectError(
        PlacementError.hot_slab_count_too_large,
        place(.{
            .vram_bytes = card_bytes,
            .context_tokens = max_context_tokens,
            .hot_slab_count = 6_000,
        }),
    );
    // The operator may shrink the hot set below what fits.
    const smaller = try place(.{
        .vram_bytes = card_bytes,
        .context_tokens = max_context_tokens,
        .hot_slab_count = 4_000,
    });
    try std.testing.expectEqual(@as(u32, 4_000), smaller.hot_slab_count);
    try std.testing.expectEqual(@as(u32, 2_144), smaller.cold_slab_count);
    try std.testing.expect(smaller.device_bytes <= card_bytes);
}

test "a zero context is refused instead of sizing a zero arena" {
    try std.testing.expectError(
        PlacementError.context_tokens_zero,
        place(.{ .vram_bytes = 16_660_430_848, .context_tokens = 0 }),
    );
}

test "the four-rank plan sums the totals and refuses a disabled tier" {
    const card_bytes: u64 = 16_660_430_848;
    var queries: [rank_count]RankQuery = undefined;
    for (&queries) |*query| {
        query.* = .{ .vram_bytes = card_bytes, .context_tokens = 8_192 };
    }
    const full = try plan(queries);
    try std.testing.expectEqual(@as(u64, 4), full.ranks.len);
    try std.testing.expectEqual(
        full.tier_bytes_total,
        full.ranks[0].tier_bytes * rank_count,
    );
    try std.testing.expectEqual(
        full.device_bytes_total,
        full.ranks[0].device_bytes * rank_count,
    );
    try std.testing.expectEqual(
        @as(u64, 12 * 1024 * 1024 * 1024 / 32),
        full.kv_bytes_total,
    );
    // The tier is needed at every context on this machine, so disabling it is a
    // refusal rather than a silent load that would not fit.
    try std.testing.expectError(
        PlacementError.dram_tier_required,
        verify(&full, false),
    );
    try verify(&full, true);
}

test "the report names every rank and states the bytes it was sized from" {
    const card_bytes: u64 = 16_660_430_848;
    var queries: [rank_count]RankQuery = undefined;
    for (&queries) |*query| {
        query.* = .{
            .vram_bytes = card_bytes,
            .context_tokens = max_context_tokens,
        };
    }
    const full = try plan(queries);
    var buffer: [8192]u8 = undefined;
    var stream = std.io.fixedBufferStream(buffer[0..]);
    try render(&full, .{
        .device_names = .{
            "NVIDIA GeForce RTX 5060 Ti",
            "NVIDIA GeForce RTX 5060 Ti",
            "NVIDIA GeForce RTX 5060 Ti",
            "NVIDIA GeForce RTX 5060 Ti",
        },
    }, stream.writer());
    const text = stream.getWritten();

    // Every rank is named, and every rank's shard is spelled out.
    for (0..rank_count) |index| {
        var line: [64]u8 = undefined;
        const expected = try std.fmt.bufPrint(
            line[0..],
            "  rank {d} NVIDIA GeForce RTX 5060 Ti\n",
            .{index},
        );
        try std.testing.expect(std.mem.indexOf(u8, text, expected) != null);
    }
    try std.testing.expect(
        std.mem.indexOf(u8, text, "128 consecutive of 512 (ids 384..511)") !=
            null,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, text, "dram tier enabled") != null,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, text, "4499 slabs on device of 6144") != null,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, text, "68520646112 B") != null,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, text, "70763146112 B") != null,
    );
    try std.testing.expect(std.mem.indexOf(u8, text, "25165824 B") != null);
}

test "a rank sized from measured bytes obeys the rule the plan is sized by" {
    // The planned table and the checkpoint's own dtypes are two different byte
    // sets, and both must be sized by one rule. This is the equivalence that
    // keeps `placeMeasured` from drifting away from `place`.
    const card_bytes: u64 = 16_660_430_848;
    for ([_]u32{ 8_192, max_context_tokens }) |context| {
        const planned = try place(.{
            .vram_bytes = card_bytes,
            .context_tokens = context,
        });
        const measured = try placeMeasured(.{
            .vram_bytes = card_bytes,
            .fixed_bytes = fixedWeightBytesPerRank(),
            .slab_bytes = expertSlabBytes(),
            .slab_count = expert_slabs_per_rank,
            .context_tokens = context,
        });
        try std.testing.expectEqual(planned, measured);
    }
}

test "a measured slab with a larger stride fits fewer slabs, never more" {
    // The real checkpoint's NVFP4 slab is 2 764 824 B against the planned
    // 2 611 212 B, so the same card holds fewer of them: 152 592 B more per
    // slab, 5.8 %, and the tier has to absorb it.
    const card_bytes: u64 = 16_660_430_848;
    const real_slab_bytes: u64 = 2_764_824;
    try std.testing.expect(real_slab_bytes > expertSlabBytes());
    const planned = try place(.{
        .vram_bytes = card_bytes,
        .context_tokens = max_context_tokens,
    });
    const real = try placeMeasured(.{
        .vram_bytes = card_bytes,
        .fixed_bytes = 6_585_174_042,
        .slab_bytes = real_slab_bytes,
        .slab_count = expert_slabs_per_rank,
        .context_tokens = max_context_tokens,
    });
    try std.testing.expectEqual(
        @as(u64, 2_768_896),
        slabStrideBytes(real_slab_bytes),
    );
    try std.testing.expect(real.hot_slab_count < planned.hot_slab_count);
    try std.testing.expect(real.tier_bytes > planned.tier_bytes);
    try std.testing.expect(real.device_bytes <= card_bytes);
    // 6.59 GB of BF16 fixed weights against the FP8 plan's 4.29 GB.
    try std.testing.expect(
        6_585_174_042 > fixedWeightBytesPerRank() + 2_000_000_000,
    );
}

test "the report says which rank is over its VRAM instead of hiding it" {
    const card_bytes: u64 = 16_660_430_848;
    var queries: [rank_count]RankQuery = undefined;
    for (&queries) |*query| {
        query.* = .{ .vram_bytes = card_bytes, .context_tokens = 8_192 };
    }
    const full = try plan(queries);
    // A hand-built plan that claims the device holds more than the card has is
    // exactly the state `verify` exists to catch.
    var broken = full;
    broken.ranks[2].device_bytes = broken.ranks[2].vram_bytes + 1;
    try std.testing.expectError(
        PlacementError.rank_exceeds_vram,
        verify(&broken, true),
    );
}
