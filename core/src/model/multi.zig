//! Four-rank execution: per-rank setup, peer access, the placement report and
//! the host-DRAM expert tier (I-014).
//!
//! This file is the plumbing the forward pass lands on. It brings up one
//! context and three streams per rank (compute, comms, prefetch), enables peer
//! access between every ordered pair of ranks and *counts* the pairs that
//! cannot reach each other instead of failing on them, sizes the placement from
//! rank's **measured** free VRAM rather than the datasheet, and owns the expert
//! tier: the expert slabs that do not fit stay in pinned host memory, and the
//! ones that do fit are an LRU hot set that a router result fetches into
//! asynchronously.
//!
//! Two tiers, one slab per tier, is the whole trick: a fetched cold slab swaps
//! with a displaced hot slab, so `device + host == slab_count` always holds and
//! no expert's bytes are ever lost. `SwapSet` is the bookkeeping and is pure,
//! so the swap protocol is tested without a GPU; `ExpertTier` is the same
//! protocol driving real pinned and device memory.
//!
//! The forward pass itself is `Forward`: four ranks, one graph, every rank
//! running the same layer sequence on its own card and exchanging the slices
//! its shard does not hold. The call order is the one the plumbing was built
//! for: `Ranks.bringUp(options)` -> `Ranks.announcePlacement` -> per layer
//! `ExpertTier.planLayer(selected, moves)` on the router result ->
//! `ExpertTier.prefetch(moves)` on the rank's `prefetch` stream ->
//! `ExpertTier.waitReady()` before the expert products.
//!
//! What is split is `mem/placement.zig`'s decision, not a new one: attention,
//! the indexer, the router and the hyper-connection mixer are replicated;
//! the dense projections are tensor parallel over their output rows; the
//! experts are expert parallel over consecutive shards. The tiny fixture runs
//! the same policy with its own counts (4 layers, 4 experts, hidden 128), which
//! is what makes the gate possible on four 16 GB cards.
//!
//! Experts are **gathered** per owner, not summed across ranks: 10 selected
//! expert outputs of 5,120 B each is 51.2 KB/token against ~5.1 KB for one
//! reduction - 10x the bytes, ~839 MB per layer per 16K chunk in production.
//! That is the honest cost of not having a reduction kernel. I-030 registers
//! the kernel, the fused all-reduce and the ordering contract; the gather stays
//! behind a flag as the reference path.
//!
//! **Why the tensor split is column parallel and gathered, not row parallel
//! and reduced: bandwidth.** At the measured 7.2 GB/s PCIe (M-001, cold tier,
//! near the Gen4 x4 link rate) one gathered boundary sends 3/4 of an activation
//! per rank - 3,840 B for a 2,560-wide bf16 stream - against 7,680 B for a ring
//! all-reduce of the same boundary. Over 48 layers that is ~230 KB/token/rank
//! gathered (~32 us at 7.2 GB/s) against ~369-737 KB/token/rank reduced
//! (~51-102 us), for the same activations. The price is redundant compute: a
//! row-parallel follow-on projection cannot be split, so it runs on every rank.
//! Only one such projection exists in this graph, the shared expert's down
//! projection, and at tensor-core rates 3 extra copies of it cost ~53 ms per
//! 48-layer 16K chunk (indicative, two lanes hold the GPUs) against ~629 ms of
//! PCIe saved per chunk. At the *measured* FFMA shared-expert rate the redundant
//! copy costs ~1.3 s per chunk and would flip the decision, which is why moving
//! that block off FFMA is a registered lever (plan.md 7c, L6) rather than a
//! detail.
//!
//! A second property, recorded because it is useful to a bisect: every slice is
//! a complete dot product over the full contraction, so a gathered result is
//! numerically the single-device result. That is a consequence of the choice,
//! not its reason.
//!
//! `src/c_api.zig` imports this file from its `test` block, so `zig build test`
//! runs these tests and `tools/gate.sh` fails on them.

const std = @import("std");

const cuda_context = @import("../hw/api.zig").cuda_context;
const cuda_driver = @import("../hw/api.zig").cuda_driver;
const exec = @import("exec.zig");
const fixtures = @import("../ref/fixtures.zig");
const generate = @import("generate.zig");
const kernel = @import("../hw/kernel.zig");
const load = @import("load.zig");
const placement = @import("../mem/placement.zig");
const tiny_check = @import("tiny_check.zig");
const generated_cubins = @import("generated_cubins");

const assert = std.debug.assert;

/// Ranks the engine runs on: one per GPU, straight from the placement.
pub const rank_count: u32 = placement.rank_count;

/// A slab id that this rank does not own.
pub const no_slab: u32 = std.math.maxInt(u32);

/// A slot index that does not exist.
pub const no_slot: u32 = std.math.maxInt(u32);

/// How many bytes one expert slab holds, from the placement's geometry.
pub const slab_bytes: u64 = placement.expertSlabBytes();

/// What the engine may do with the experts a rank cannot hold.
pub const Options = struct {
    /// `--no-dram-tier`: refuse to start instead of moving cold experts to host
    /// memory. The tier is what makes this checkpoint fit, so refusing is the
    /// honest response to "do not touch host RAM".
    dram_tier_enabled: bool = true,
    /// `--expert-hot-slabs=N`: expert slabs to keep on the device; 0 sizes them
    /// from the measured free VRAM.
    hot_slab_count: u32 = 0,
    /// `--context-tokens=N`: context the KV arena is sized for.
    context_tokens: u32 = placement.max_context_tokens,
};

pub const OptionError = error{
    unknown_argument,
    argument_value_missing,
    argument_value_invalid,
};

/// Parses the engine's own flags out of an argument list. Positional arguments
/// belong to the caller and are ignored here.
pub fn parseOptions(arguments: []const []const u8) OptionError!Options {
    assert(arguments.len < 4096);
    assert(placement.max_context_tokens > 0);

    var options = Options{};
    for (arguments) |argument| {
        if (std.mem.eql(u8, argument, "--no-dram-tier")) {
            options.dram_tier_enabled = false;
        } else if (std.mem.startsWith(u8, argument, "--expert-hot-slabs=")) {
            options.hot_slab_count = try parseCount(
                argument,
                "--expert-hot-slabs=".len,
            );
        } else if (std.mem.startsWith(u8, argument, "--context-tokens=")) {
            options.context_tokens = try parseCount(
                argument,
                "--context-tokens=".len,
            );
        } else if (std.mem.startsWith(u8, argument, "--")) {
            // A flag this engine does not know is a caller bug, not a default.
            return OptionError.unknown_argument;
        } else {
            // Positional: not ours.
        }
    }
    assert(options.context_tokens > 0);
    return options;
}

/// Parses the positive decimal value that follows `prefix_bytes` in `argument`.
fn parseCount(argument: []const u8, prefix_bytes: u32) OptionError!u32 {
    assert(prefix_bytes > 0);
    assert(argument.len >= prefix_bytes);

    const text = argument[prefix_bytes..];
    if (text.len == 0) {
        return OptionError.argument_value_missing;
    } else {
        assert(text.len > 0);
    }
    const value = std.fmt.parseInt(u32, text, 10) catch {
        return OptionError.argument_value_invalid;
    };
    if (value == 0) {
        return OptionError.argument_value_invalid;
    } else {
        assert(value > 0);
    }
    return value;
}

/// Bytes of one embedded cubin, by table index.
fn cubinBytes(index: u32) []const u8 {
    assert(index < generated_cubins.cubins.len);
    assert(@sizeOf(generated_cubins.Cubin) > 0);

    const entry = generated_cubins.cubins[index];
    const bytes = entry.bytes[0..entry.length];
    assert(bytes.len > 0);
    return bytes;
}

/// The slab id of one of this rank's experts in one layer.
///
/// Slab ids are dense over the shard, which is what lets the tier index them
/// with a plain array instead of a hash table.
pub fn slabId(layer: u32, expert_in_shard: u32) u32 {
    assert(layer < placement.layer_count);
    assert(expert_in_shard < placement.experts_per_rank);

    const id = layer * placement.experts_per_rank + expert_in_shard;
    assert(id < placement.expert_slabs_per_rank);
    return id;
}

/// The layer a slab belongs to.
pub fn slabLayer(slab: u32) u32 {
    assert(slab < placement.expert_slabs_per_rank);
    assert(placement.experts_per_rank > 0);

    const layer = slab / placement.experts_per_rank;
    assert(layer < placement.layer_count);
    return layer;
}

/// The shard-local expert index a slab belongs to.
pub fn slabExpertInShard(slab: u32) u32 {
    assert(slab < placement.expert_slabs_per_rank);
    assert(placement.experts_per_rank > 0);

    const expert = slab % placement.experts_per_rank;
    assert(expert < placement.experts_per_rank);
    return expert;
}

/// The shard-local index of a router expert id, or null when another rank owns
/// it. Because the shards are consecutive, this needs no communication.
pub fn expertInShard(rank_index: u32, expert: u32) ?u32 {
    assert(rank_index < rank_count);
    assert(expert < placement.experts_total);

    const first = rank_index * placement.experts_per_rank;
    if (expert < first) {
        return null;
    } else if (expert >= first + placement.experts_per_rank) {
        return null;
    } else {
        const local = expert - first;
        assert(local < placement.experts_per_rank);
        return local;
    }
}

/// One rank's driver resources.
pub const Rank = struct {
    index: u32,
    device: cuda_driver.Device,
    context: cuda_context.Context,
    /// The stream the model graph runs on.
    compute: cuda_context.Stream,
    /// The stream peer copies and MoE dispatch and combine run on, so a
    /// reduction never queues behind the previous layer's kernels.
    comms: cuda_context.Stream,
    /// The stream the expert tier prefetches on.
    prefetch: cuda_context.Stream,
    /// Whether this rank may read each peer's memory in place.
    reachable: [rank_count]bool,
    /// One loaded cubin per entry of the embedded table.
    modules: [generated_cubins.cubins.len]kernel.Module,
    /// Free device bytes the driver reported once the context was retained.
    free_bytes: u64,

    pub fn init(devices: []const cuda_driver.Device, index: u32) !Rank {
        assert(index < devices.len);
        assert(index < rank_count);

        var rank: Rank = undefined;
        rank.index = index;
        rank.device = devices[index];
        rank.context = try cuda_context.Context.retain(&devices[index]);
        try rank.context.set_current();
        rank.compute = try cuda_context.Stream.create();
        rank.comms = try cuda_context.Stream.create();
        rank.prefetch = try cuda_context.Stream.create();
        rank.reachable = [_]bool{false} ** rank_count;
        // Measured, not assumed: another tenant's context may already hold part
        // of this card, and the placement has to see the bytes that are left.
        rank.free_bytes = try rank.context.free_bytes();
        for (&rank.modules, 0..) |*module, cubin_index| {
            const index_in_table: u32 = @intCast(cubin_index);
            module.* = try kernel.Module.load(cubinBytes(index_in_table));
        }
        assert(rank.modules.len == generated_cubins.cubins.len);
        return rank;
    }

    pub fn deinit(rank: *Rank) void {
        assert(rank.index < rank_count);
        assert(rank.modules.len > 0);

        rank.context.set_current() catch |err| {
            std.log.err(
                "rank {d}: cannot make its context current to tear down: {s}",
                .{ rank.index, @errorName(err) },
            );
            return;
        };
        for (&rank.modules) |*module| {
            module.unload();
        }
        rank.prefetch.destroy();
        rank.comms.destroy();
        rank.compute.destroy();
        rank.context.release();
    }
};

/// Enables peer access for every ordered pair of ranks and returns how many
/// pairs could not be connected.
///
/// A pair that cannot reach each other is a *loud* fallback, not a failure: the
/// engine stages that pair's traffic through host memory and reports the count,
/// because one slow pair is better than no engine and a silent one is worse
/// than both.
pub fn enablePeerAccess(ranks: []Rank) u32 {
    assert(ranks.len > 0);
    assert(ranks.len <= rank_count);

    var fallbacks: u32 = 0;
    for (ranks) |*from| {
        for (ranks) |*to| {
            if (from.index == to.index) {
                // Every rank trivially reaches its own memory.
            } else {
                fallbacks += connectPair(from, to);
            }
        }
    }
    assert(fallbacks <= ranks.len * (ranks.len - 1));
    return fallbacks;
}

/// Connects one ordered pair, returning 1 when it needs host staging.
fn connectPair(from: *Rank, to: *Rank) u32 {
    assert(from.index != to.index);
    assert(to.index < rank_count);

    const can_reach = cuda_driver.can_access_peer(
        from.device.handle,
        to.device.handle,
    ) catch |err| {
        std.log.err(
            "rank {d} cannot query peer access to rank {d}: {s}",
            .{ from.index, to.index, @errorName(err) },
        );
        return 1;
    };
    if (can_reach) {
        from.context.enable_peer_access(&to.context) catch |err| {
            std.log.err(
                "rank {d} cannot enable peer access to rank {d}: {s}",
                .{ from.index, to.index, @errorName(err) },
            );
            return 1;
        };
        from.reachable[to.index] = true;
        assert(from.reachable[to.index]);
        return 0;
    } else {
        std.log.warn(
            "P2P unavailable rank {d} -> rank {d}: staging that pair " ++
                "through host memory",
            .{ from.index, to.index },
        );
        return 1;
    }
}

/// How big an expert slab is and how many of them one rank owns.
pub const SlabGeometry = struct {
    slab_count: u32,
    slab_bytes: u64,
    /// Page rounded: what one arena slot occupies.
    slab_stride_bytes: u64,

    /// The checkpoint's geometry: every layer, this rank's 128 experts.
    pub fn checkpoint() SlabGeometry {
        const geometry = SlabGeometry{
            .slab_count = placement.expert_slabs_per_rank,
            .slab_bytes = placement.expertSlabBytes(),
            .slab_stride_bytes = placement.expertSlabStrideBytes(),
        };
        assert(geometry.slab_count > 0);
        assert(geometry.slab_stride_bytes >= geometry.slab_bytes);
        return geometry;
    }
};

/// One planned move of the tier: fetch `fetch_slab` from `host_slot` into
/// `device_slot`, after writing what that slot held back into the same host
/// slot. `displaced_slab` is `no_slab` when the device slot was still free.
pub const Move = struct {
    fetch_slab: u32,
    host_slot: u32,
    device_slot: u32,
    displaced_slab: u32,
};

pub const SwapError = error{
    slab_out_of_range,
    slot_out_of_range,
    slab_already_placed,
    slot_occupied,
    slab_not_loaded,
    tier_not_populated,
    move_buffer_too_small,
    bookkeeping_inconsistent,
};

/// Which slab sits where, across the device hot set and the pinned host tier.
///
/// The invariant is that every slab this rank owns is in exactly one tier, so
/// the host arena never mirrors the whole shard and no expert is ever dropped.
pub const SwapSet = struct {
    geometry: SlabGeometry,
    /// slab -> device slot.
    device_slot_of_slab: []u32,
    /// device slot -> slab.
    slab_of_device_slot: []u32,
    /// slab -> host slot.
    host_slot_of_slab: []u32,
    /// host slot -> slab.
    slab_of_host_slot: []u32,
    /// LRU chain over device slots: `previous` towards the most recently used
    /// slot, `next` towards the eviction candidate.
    previous: []u32,
    next: []u32,
    head: u32,
    tail: u32,
    device_resident: u32,
    host_resident: u32,
    planned_moves: u64,

    pub fn init(
        allocator: std.mem.Allocator,
        geometry: SlabGeometry,
        device_slot_count: u32,
        host_slot_count: u32,
    ) std.mem.Allocator.Error!SwapSet {
        assert(geometry.slab_count > 0);
        assert(device_slot_count + host_slot_count == geometry.slab_count);

        var set: SwapSet = undefined;
        set.geometry = geometry;
        set.device_slot_of_slab = try allocator.alloc(u32, geometry.slab_count);
        set.host_slot_of_slab = try allocator.alloc(u32, geometry.slab_count);
        set.slab_of_device_slot = try allocator.alloc(u32, device_slot_count);
        set.slab_of_host_slot = try allocator.alloc(u32, host_slot_count);
        set.previous = try allocator.alloc(u32, device_slot_count);
        set.next = try allocator.alloc(u32, device_slot_count);
        @memset(set.device_slot_of_slab, no_slot);
        @memset(set.host_slot_of_slab, no_slot);
        @memset(set.slab_of_device_slot, no_slab);
        @memset(set.slab_of_host_slot, no_slab);
        @memset(set.previous, no_slot);
        @memset(set.next, no_slot);
        set.head = no_slot;
        set.tail = no_slot;
        set.device_resident = 0;
        set.host_resident = 0;
        set.planned_moves = 0;
        assert(set.device_resident == 0);
        return set;
    }

    pub fn deinit(set: *SwapSet, allocator: std.mem.Allocator) void {
        assert(set.geometry.slab_count > 0);
        assert(set.next.len > 0);

        allocator.free(set.next);
        allocator.free(set.previous);
        allocator.free(set.slab_of_host_slot);
        allocator.free(set.slab_of_device_slot);
        allocator.free(set.host_slot_of_slab);
        allocator.free(set.device_slot_of_slab);
    }

    /// Records that `slab` already sits in device slot `slot`, as the loader
    /// leaves the hot part of the shard.
    pub fn admit(set: *SwapSet, slab: u32, slot: u32) SwapError!void {
        assert(set.geometry.slab_count > 0);
        assert(set.slab_of_device_slot.len > 0);

        if (slab >= set.geometry.slab_count) {
            return SwapError.slab_out_of_range;
        } else if (slot >= set.slab_of_device_slot.len) {
            return SwapError.slot_out_of_range;
        } else if (set.device_slot_of_slab[slab] != no_slot) {
            return SwapError.slab_already_placed;
        } else if (set.slab_of_device_slot[slot] != no_slab) {
            return SwapError.slot_occupied;
        } else {
            assert(set.slab_of_device_slot[slot] == no_slab);
        }
        set.device_slot_of_slab[slab] = slot;
        set.slab_of_device_slot[slot] = slab;
        set.host_slot_of_slab[slab] = no_slot;
        set.pushHead(slot);
        set.device_resident += 1;
        assert(set.slab_of_device_slot[slot] == slab);
    }

    /// Records that `slab` already sits in pinned host slot `slot`.
    pub fn stage(set: *SwapSet, slab: u32, slot: u32) SwapError!void {
        assert(set.geometry.slab_count > 0);
        assert(set.slab_of_host_slot.len > 0);

        if (slab >= set.geometry.slab_count) {
            return SwapError.slab_out_of_range;
        } else if (slot >= set.slab_of_host_slot.len) {
            return SwapError.slot_out_of_range;
        } else if (set.host_slot_of_slab[slab] != no_slot) {
            return SwapError.slab_already_placed;
        } else if (set.slab_of_host_slot[slot] != no_slab) {
            return SwapError.slot_occupied;
        } else {
            assert(set.slab_of_host_slot[slot] == no_slab);
        }
        set.host_slot_of_slab[slab] = slot;
        set.slab_of_host_slot[slot] = slab;
        set.device_slot_of_slab[slab] = no_slot;
        set.host_resident += 1;
        assert(set.slab_of_host_slot[slot] == slab);
    }

    /// Plans the moves one routing result needs: every selected slab this rank
    /// owns that is cold is fetched from its host slot into the least recently
    /// used device slot, whose occupant is written back first. Resident slabs
    /// are only touched, so the LRU order follows the routing, not the clock.
    pub fn planLayer(
        set: *SwapSet,
        selected: []const u32,
        moves: []Move,
    ) SwapError!u32 {
        assert(set.geometry.slab_count > 0);
        assert(moves.len > 0);

        var count: u32 = 0;
        for (selected) |slab| {
            if (slab >= set.geometry.slab_count) {
                return SwapError.slab_out_of_range;
            } else {
                assert(slab < set.geometry.slab_count);
            }
            if (set.device_slot_of_slab[slab] != no_slot) {
                set.touch(set.device_slot_of_slab[slab]);
            } else {
                if (count >= moves.len) {
                    return SwapError.move_buffer_too_small;
                } else {
                    assert(count < moves.len);
                }
                moves[count] = try set.planFetch(slab);
                count += 1;
            }
        }
        assert(count <= moves.len);
        return count;
    }

    /// Reserves a device slot for one cold slab and describes the swap.
    fn planFetch(set: *SwapSet, slab: u32) SwapError!Move {
        assert(slab < set.geometry.slab_count);
        assert(set.device_slot_of_slab[slab] == no_slot);

        const host_slot = set.host_slot_of_slab[slab];
        if (host_slot == no_slot) {
            return SwapError.slab_not_loaded;
        } else {
            assert(host_slot < set.slab_of_host_slot.len);
        }
        if (set.tail == no_slot) {
            return SwapError.tier_not_populated;
        } else {
            assert(set.tail < set.slab_of_device_slot.len);
        }

        const device_slot = set.tail;
        const displaced = set.slab_of_device_slot[device_slot];
        set.unlink(device_slot);
        // The victim takes the host slot the fetched slab leaves: the tiers
        // exchange one slab, so neither total ever changes.
        set.device_slot_of_slab[displaced] = no_slot;
        set.host_slot_of_slab[displaced] = host_slot;
        set.slab_of_host_slot[host_slot] = displaced;
        set.device_slot_of_slab[slab] = device_slot;
        set.slab_of_device_slot[device_slot] = slab;
        set.host_slot_of_slab[slab] = no_slot;
        set.pushHead(device_slot);
        set.planned_moves += 1;
        assert(set.slab_of_device_slot[device_slot] == slab);
        assert(set.slab_of_host_slot[host_slot] == displaced);
        return .{
            .fetch_slab = slab,
            .host_slot = host_slot,
            .device_slot = device_slot,
            .displaced_slab = displaced,
        };
    }

    /// Moves a device slot to the most recently used end of the chain.
    fn touch(set: *SwapSet, slot: u32) void {
        assert(slot < set.slab_of_device_slot.len);
        assert(set.slab_of_device_slot[slot] != no_slab);

        set.unlink(slot);
        set.pushHead(slot);
        assert(set.head == slot);
    }

    /// Pushes an unchained device slot onto the most recently used end.
    fn pushHead(set: *SwapSet, slot: u32) void {
        assert(slot < set.next.len);
        assert(set.next[slot] == no_slot);

        set.previous[slot] = no_slot;
        set.next[slot] = set.head;
        if (set.head != no_slot) {
            set.previous[set.head] = slot;
        } else {
            set.tail = slot;
        }
        set.head = slot;
        assert(set.head == slot);
    }

    /// Removes a chained device slot from the LRU list.
    fn unlink(set: *SwapSet, slot: u32) void {
        assert(slot < set.next.len);
        assert(set.head != no_slot);

        const before = set.previous[slot];
        const after = set.next[slot];
        if (before != no_slot) {
            set.next[before] = after;
        } else {
            assert(set.head == slot);
            set.head = after;
        }
        if (after != no_slot) {
            set.previous[after] = before;
        } else {
            assert(set.tail == slot);
            set.tail = before;
        }
        set.previous[slot] = no_slot;
        set.next[slot] = no_slot;
        assert(set.next[slot] == no_slot);
    }

    /// Checks the two-tier invariant: every slab is in exactly one tier, the
    /// reverse maps agree, and the LRU chain holds exactly the resident slots.
    pub fn verify(set: *const SwapSet) SwapError!void {
        assert(set.geometry.slab_count > 0);
        assert(set.slab_of_host_slot.len > 0);

        var device_count: u32 = 0;
        var host_count: u32 = 0;
        for (0..set.geometry.slab_count) |index| {
            const slab: u32 = @intCast(index);
            const device_slot = set.device_slot_of_slab[slab];
            const host_slot = set.host_slot_of_slab[slab];
            if (device_slot != no_slot) {
                if (host_slot != no_slot) {
                    return SwapError.bookkeeping_inconsistent;
                } else {
                    assert(host_slot == no_slot);
                }
                if (set.slab_of_device_slot[device_slot] != slab) {
                    return SwapError.bookkeeping_inconsistent;
                } else {
                    assert(set.slab_of_device_slot[device_slot] == slab);
                }
                device_count += 1;
            } else if (host_slot != no_slot) {
                if (set.slab_of_host_slot[host_slot] != slab) {
                    return SwapError.bookkeeping_inconsistent;
                } else {
                    assert(set.slab_of_host_slot[host_slot] == slab);
                }
                host_count += 1;
            } else {
                // A slab nobody holds has lost its bytes.
                return SwapError.bookkeeping_inconsistent;
            }
        }
        if (device_count != set.device_resident) {
            return SwapError.bookkeeping_inconsistent;
        } else if (host_count != set.host_resident) {
            return SwapError.bookkeeping_inconsistent;
        } else {
            assert(device_count == set.device_resident);
        }
        try set.verifyChain();
    }

    /// Walks the LRU chain: it must be a simple list of the resident slots.
    fn verifyChain(set: *const SwapSet) SwapError!void {
        assert(set.previous.len == set.next.len);
        assert(set.previous.len > 0);

        var walked: u32 = 0;
        var cursor = set.tail;
        while (cursor != no_slot) {
            if (walked >= set.slab_of_device_slot.len) {
                return SwapError.bookkeeping_inconsistent;
            } else {
                assert(walked < set.slab_of_device_slot.len);
            }
            if (set.slab_of_device_slot[cursor] == no_slab) {
                return SwapError.bookkeeping_inconsistent;
            } else {
                assert(set.slab_of_device_slot[cursor] != no_slab);
            }
            walked += 1;
            cursor = set.previous[cursor];
        }
        if (walked != set.device_resident) {
            return SwapError.bookkeeping_inconsistent;
        } else {
            assert(walked == set.device_resident);
        }
    }

    /// Device slots that hold a slab.
    pub fn deviceSlotsInUse(set: *const SwapSet) u32 {
        assert(set.geometry.slab_count > 0);
        assert(set.device_resident <= set.slab_of_device_slot.len);

        return set.device_resident;
    }

    /// Host slots that hold a slab.
    pub fn hostSlotsInUse(set: *const SwapSet) u32 {
        assert(set.geometry.slab_count > 0);
        assert(set.host_resident <= set.slab_of_host_slot.len);

        return set.host_resident;
    }
};

pub const TierError = SwapError || cuda_context.DriverError;

/// The comms bench has to leave a readable report behind.
pub const ReportError = error{report_not_written};

/// Pinned staging slabs the tier rotates through. A swap writes the displaced
/// slab into the host slot the fetched slab is leaving, so the fetched bytes
/// have to be read out somewhere first; two slots is the smallest rotation that
/// lets the CPU refill one while the device copy still reads the other.
const bounce_slots: usize = 2;

/// One rank's host-DRAM expert tier: cold slabs in pinned host memory, hot
/// slabs in one contiguous device arena, and the swap protocol that moves a
/// slab between them on the prefetch stream.
///
/// A batch of moves is one token's routing on this rank, not its whole cold
/// set: the router keeps `top_k_total = 10` of the 512 experts in every one of
/// the 48 layers (`kernels/moe.zig`), so a rank fetches ~120 slabs for one
/// token (48 layers x 10 / 4 ranks) while its cold set is 1 645 slabs at 256 K
/// context. Sizing a move buffer for the cold set sizes it for the wrong
/// quantity.
pub const ExpertTier = struct {
    geometry: SlabGeometry,
    device_slot_count: u32,
    host_slot_count: u32,
    host: cuda_context.PinnedHostBuffer,
    device: cuda_context.DeviceBuffer,
    /// Rotating pinned staging: `bounce_free[i]` is recorded at the end of a
    /// batch's prefetch, after the device copy that reads `bounce[i]` and after
    /// every writeback the batch queued, so the next refill of that slot knows
    /// when both the copy and the host slot it is about to read are settled.
    bounce: [bounce_slots]cuda_context.PinnedHostBuffer,
    bounce_free: [bounce_slots]cuda_context.Event,
    stream: cuda_context.Stream,
    /// Whether `deinit` destroys `stream` (the tier created it) or leaves it to
    /// its owner (the engine passed the rank's prefetch queue).
    owns_stream: bool,
    ready: cuda_context.Event,
    swap: SwapSet,
    /// Bytes fetched from host memory since init.
    fetch_bytes: u64,
    /// Bytes written back to host memory since init.
    writeback_bytes: u64,

    pub fn init(
        allocator: std.mem.Allocator,
        geometry: SlabGeometry,
        device_slot_count: u32,
        host_slot_count: u32,
    ) !ExpertTier {
        return initOn(
            allocator,
            geometry,
            device_slot_count,
            host_slot_count,
            null,
        );
    }

    /// `init` on a queue the caller already owns: the engine runs the tier on
    /// the rank's dedicated `prefetch` stream, so a fetch never queues behind
    /// that rank's compute and the tier does not create a second queue.
    pub fn initOn(
        allocator: std.mem.Allocator,
        geometry: SlabGeometry,
        device_slot_count: u32,
        host_slot_count: u32,
        stream: ?*cuda_context.Stream,
    ) !ExpertTier {
        assert(device_slot_count > 0);
        assert(host_slot_count > 0);
        assert(device_slot_count + host_slot_count == geometry.slab_count);

        const device_bytes = @as(u64, device_slot_count) *
            geometry.slab_stride_bytes;
        const host_bytes = @as(u64, host_slot_count) *
            geometry.slab_stride_bytes;
        var tier: ExpertTier = undefined;
        tier.geometry = geometry;
        tier.device_slot_count = device_slot_count;
        tier.host_slot_count = host_slot_count;
        tier.owns_stream = stream == null;
        tier.device = try cuda_context.DeviceBuffer.allocate(device_bytes);
        tier.host = try cuda_context.PinnedHostBuffer.allocate(host_bytes);
        for (&tier.bounce) |*bounce| {
            bounce.* = try cuda_context.PinnedHostBuffer.allocate(
                geometry.slab_stride_bytes,
            );
        }
        for (&tier.bounce_free) |*free| {
            free.* = try cuda_context.Event.create();
        }
        tier.stream = if (stream) |given| given.* else try cuda_context.Stream.create();
        tier.ready = try cuda_context.Event.create();
        tier.swap = try SwapSet.init(
            allocator,
            geometry,
            device_slot_count,
            host_slot_count,
        );
        tier.fetch_bytes = 0;
        tier.writeback_bytes = 0;
        assert(tier.device.bytes >= device_bytes);
        return tier;
    }

    pub fn deinit(tier: *ExpertTier, allocator: std.mem.Allocator) void {
        assert(tier.device.bytes > 0);
        assert(tier.host.bytes > 0);

        tier.swap.deinit(allocator);
        tier.ready.destroy();
        if (tier.owns_stream) {
            tier.stream.destroy();
        } else {
            assert(!tier.owns_stream);
        }
        for (&tier.bounce_free) |*free| {
            free.destroy();
        }
        for (&tier.bounce) |*bounce| {
            bounce.free();
        }
        tier.host.free();
        tier.device.free();
    }

    /// The slab resident in a device slot, if any. The forward resolves the
    /// weights it computes from here, so an unfetched slab is an error at the
    /// product, not a silently stale matrix.
    pub fn residentSlot(tier: *const ExpertTier, slab: u32) ?u32 {
        assert(slab < tier.geometry.slab_count);
        assert(tier.swap.device_slot_of_slab.len == tier.geometry.slab_count);

        const slot = tier.swap.device_slot_of_slab[slab];
        if (slot == no_slot) {
            return null;
        } else {
            assert(slot < tier.device_slot_count);
            return slot;
        }
    }

    /// The bytes of one device slot, for the expert products to read.
    pub fn slotBuffer(
        tier: *const ExpertTier,
        slot: u32,
    ) cuda_context.DeviceBuffer {
        assert(slot < tier.device_slot_count);
        assert(tier.geometry.slab_bytes > 0);
        return tier.deviceSlot(slot);
    }

    /// The device arena a slot lives in.
    fn deviceSlot(
        tier: *const ExpertTier,
        slot: u32,
    ) cuda_context.DeviceBuffer {
        assert(slot < tier.device_slot_count);
        assert(tier.geometry.slab_bytes <= tier.device.bytes);

        return .{
            .pointer = tier.device.pointer +
                slot * tier.geometry.slab_stride_bytes,
            .bytes = tier.geometry.slab_bytes,
        };
    }

    /// Byte offset of one host slot in its arena.
    fn hostOffset(tier: *const ExpertTier, slot: u32) u64 {
        assert(slot < tier.host_slot_count);
        assert(tier.geometry.slab_stride_bytes > 0);

        const offset = @as(u64, slot) * tier.geometry.slab_stride_bytes;
        assert(offset + tier.geometry.slab_bytes <= tier.host.bytes);
        return offset;
    }

    /// The pinned host memory a slot lives in.
    fn hostSlot(
        tier: *const ExpertTier,
        slot: u32,
    ) cuda_context.PinnedHostBuffer {
        assert(slot < tier.host_slot_count);
        assert(tier.geometry.slab_bytes <= tier.host.bytes);

        return .{
            .pointer = tier.host.pointer + tier.hostOffset(slot),
            .bytes = tier.geometry.slab_bytes,
        };
    }

    /// The pinned bytes of one host slot, as the CPU sees them.
    fn hostSlotBytes(tier: *const ExpertTier, slot: u32) []const u8 {
        assert(slot < tier.host_slot_count);
        assert(tier.geometry.slab_bytes <= tier.host.bytes);

        const offset = tier.hostOffset(slot);
        return tier.host.slice()[offset .. offset + tier.geometry.slab_bytes];
    }

    /// Copies one hot slab's checkpoint bytes into the device arena. The loader
    /// calls this once per slab that fits.
    pub fn uploadHotSlab(
        tier: *ExpertTier,
        slab: u32,
        device_slot: u32,
        bytes: []const u8,
    ) TierError!void {
        assert(bytes.len > 0);
        assert(bytes.len <= slab_bytes);

        const staged = try cuda_context.PinnedHostBuffer.allocate(
            std.mem.alignForward(u64, bytes.len, placement.page_bytes),
        );
        defer staged.free();
        @memcpy(staged.slice()[0..bytes.len], bytes);
        const target = tier.deviceSlot(device_slot);
        try tier.stream.copy_h_to_d(&target, &staged, bytes.len);
        // The staging buffer must outlive the copy.
        try tier.stream.synchronize();
        try tier.swap.admit(slab, device_slot);
        assert(tier.swap.deviceSlotsInUse() > 0);
    }

    /// Copies one cold slab's checkpoint bytes into its pinned host slot.
    pub fn stageColdSlab(
        tier: *ExpertTier,
        slab: u32,
        host_slot: u32,
        bytes: []const u8,
    ) TierError!void {
        assert(bytes.len > 0);
        assert(bytes.len <= slab_bytes);

        const offset = tier.hostOffset(host_slot);
        const arena = tier.host.slice();
        if (offset + bytes.len > arena.len) {
            return SwapError.slot_out_of_range;
        } else {
            assert(offset + bytes.len <= arena.len);
        }
        @memcpy(arena[offset .. offset + bytes.len], bytes);
        try tier.swap.stage(slab, host_slot);
        assert(tier.swap.hostSlotsInUse() > 0);
    }

    /// Plans the moves one layer's routing result needs.
    pub fn planLayer(
        tier: *ExpertTier,
        selected: []const u32,
        moves: []Move,
    ) SwapError!u32 {
        assert(tier.geometry.slab_count > 0);
        assert(moves.len > 0);

        const count = try tier.swap.planLayer(selected, moves);
        assert(count <= moves.len);
        return count;
    }

    /// Issues one layer's planned moves on the prefetch stream: the fetched
    /// slab is read out of its host slot into a staging slab, the displaced
    /// slab is written into that host slot, and the staging slab is copied into
    /// the freed device slot. Order matters: writing the victim first would
    /// overwrite the very bytes being fetched.
    ///
    /// The staging slabs rotate, because `copy_h_to_d` is asynchronous: reusing
    /// one staging slab would let the CPU refill it for the next move while the
    /// copy for this one is still reading it, which hands the device the wrong
    /// expert's weights for every move but the last in the batch.
    ///
    /// The event lets the compute stream wait only on these copies.
    ///
    /// Every staging slab's event is recorded again at the end of the batch.
    /// Recording it beside the copy that read the slab is not enough: the
    /// writeback of a *later* move in the same batch can still be filling a
    /// host slot, and the next batch's refill reads host slots. Move 1 of batch
    /// N+1 waits on the event move 1 of batch N recorded, while move 2 of batch
    /// N queued its writeback after that record, so a host slot written back at
    /// the tail of one batch could be read by the CPU at the top of the next.
    pub fn prefetch(tier: *ExpertTier, moves: []const Move) TierError!void {
        assert(moves.len > 0);
        assert(tier.geometry.slab_bytes > 0);

        for (moves, 0..) |move, index| {
            const bounce = &tier.bounce[index % bounce_slots];
            const free = &tier.bounce_free[index % bounce_slots];
            // Waits for the copy that last read this staging slab; the first
            // move waits on an event no copy has recorded, which is free.
            try free.synchronize();
            @memcpy(
                bounce.slice()[0..tier.geometry.slab_bytes],
                tier.hostSlotBytes(move.host_slot),
            );
            if (move.displaced_slab == no_slab) {
                // A slot that was never used holds nothing to write back.
            } else {
                const origin = tier.deviceSlot(move.device_slot);
                const destination = tier.hostSlot(move.host_slot);
                try tier.stream.copy_d_to_h(
                    &destination,
                    &origin,
                    tier.geometry.slab_bytes,
                );
                tier.writeback_bytes += tier.geometry.slab_bytes;
            }
            const origin: cuda_context.PinnedHostBuffer = .{
                .pointer = bounce.pointer,
                .bytes = tier.geometry.slab_bytes,
            };
            const destination = tier.deviceSlot(move.device_slot);
            try tier.stream.copy_h_to_d(
                &destination,
                &origin,
                tier.geometry.slab_bytes,
            );
            try free.record(&tier.stream);
            tier.fetch_bytes += tier.geometry.slab_bytes;
        }
        try tier.ready.record(&tier.stream);
        // After the last writeback the batch queued, so waiting on any staging
        // slab's event also waits for every host slot this batch filled.
        for (&tier.bounce_free) |*free| {
            try free.record(&tier.stream);
        }
        assert(tier.fetch_bytes > 0);
    }

    /// Blocks until the prefetched slabs are on the device. The compute stream
    /// waits on `ready` in the graph; this is the plumbing's synchronous form.
    pub fn waitReady(tier: *const ExpertTier) TierError!void {
        assert(tier.geometry.slab_bytes > 0);
        assert(tier.fetch_bytes > 0);

        try tier.ready.synchronize();
        assert(tier.fetch_bytes > 0);
    }
};

/// The four ranks, the placement they were sized from and the peer links the
/// driver could not establish.
pub const Ranks = struct {
    ranks: [rank_count]Rank,
    plan: placement.Placement,
    peer_fallbacks: u32,
    options: Options,

    /// Brings every rank up, connects the peers and sizes the placement from
    /// the free VRAM the driver reports for each of them.
    pub fn bringUp(options: Options) !Ranks {
        assert(options.context_tokens > 0);
        assert(rank_count > 0);

        var devices: [cuda_driver.device_count_max]cuda_driver.Device =
            undefined;
        const device_count = try cuda_driver.query_devices(devices[0..]);
        if (device_count < rank_count) {
            std.log.err(
                "the engine needs {d} GPUs and the driver reports {d}",
                .{ rank_count, device_count },
            );
            return error.not_enough_devices;
        } else {
            assert(device_count >= rank_count);
        }

        var ranks: Ranks = undefined;
        ranks.options = options;
        for (&ranks.ranks, 0..) |*rank, index| {
            rank.* = try Rank.init(devices[0..device_count], @intCast(index));
        }
        ranks.peer_fallbacks = enablePeerAccess(ranks.ranks[0..]);
        ranks.plan = try placement.plan(queriesFor(ranks.ranks[0..], options));
        assert(ranks.plan.ranks.len == rank_count);
        return ranks;
    }

    pub fn deinit(ranks: *Ranks) void {
        assert(ranks.ranks.len == rank_count);
        assert(ranks.plan.ranks.len == rank_count);

        for (&ranks.ranks) |*rank| {
            rank.deinit();
        }
    }

    /// Prints the placement report and refuses to start if a rank would exceed
    /// its measured VRAM, or if cold experts have no tier to live in.
    pub fn announcePlacement(ranks: *const Ranks, writer: anytype) !void {
        assert(ranks.ranks.len == rank_count);
        assert(ranks.options.context_tokens > 0);

        var names: [rank_count][]const u8 = .{ "", "", "", "" };
        for (&ranks.ranks, 0..) |*rank, index| {
            names[index] = rank.device.name_slice();
        }
        try writer.print(
            "peer access: {d} links to host staging\n",
            .{ranks.peer_fallbacks},
        );
        try placement.render(&ranks.plan, .{
            .device_names = names,
            .dram_tier_enabled = ranks.options.dram_tier_enabled,
        }, writer);
        ranks.verify() catch |err| {
            try writer.print("refusing to start: {s}\n", .{@errorName(err)});
            return err;
        };
        assert(ranks.plan.ranks.len == rank_count);
    }

    /// The refusal gate: every rank under its measured VRAM, and a tier where
    /// there are cold experts.
    pub fn verify(ranks: *const Ranks) placement.PlacementError!void {
        assert(ranks.ranks.len == rank_count);
        assert(ranks.plan.weights_bytes_per_rank > 0);

        try placement.verify(&ranks.plan, ranks.options.dram_tier_enabled);
        assert(ranks.peer_fallbacks <= rank_count * (rank_count - 1));
    }
};

/// One placement query per rank, each with that rank's measured free VRAM.
fn queriesFor(
    ranks: []const Rank,
    options: Options,
) [rank_count]placement.RankQuery {
    assert(ranks.len == rank_count);
    assert(options.context_tokens > 0);

    var queries: [rank_count]placement.RankQuery = undefined;
    for (ranks, 0..) |*rank, index| {
        queries[index] = .{
            .vram_bytes = rank.free_bytes,
            .context_tokens = options.context_tokens,
            .hot_slab_count = options.hot_slab_count,
        };
    }
    assert(queries.len == rank_count);
    return queries;
}

// --- the four-rank forward ---------------------------------------------------

const empty_buffer = cuda_context.DeviceBuffer{ .pointer = 0, .bytes = 0 };

/// A device allocation rounded to the driver's granularity: every slice buffer
/// here is smaller than a page at fixture shapes.
fn allocateDevice(bytes: u64) !cuda_context.DeviceBuffer {
    assert(bytes > 0);
    assert(placement.page_bytes > 0);

    return cuda_context.DeviceBuffer.allocate(
        std.mem.alignForward(u64, bytes, placement.page_bytes),
    );
}

/// A device sub-range as its own buffer, with the bounds checked.
fn view(
    buffer: *const cuda_context.DeviceBuffer,
    offset: u64,
    bytes: u64,
) cuda_context.DeviceBuffer {
    assert(bytes > 0);
    assert(buffer.pointer != 0);
    assert(offset + bytes <= buffer.bytes);

    return .{ .pointer = buffer.pointer + offset, .bytes = bytes };
}

/// One expert's bytes in the tier: `gate_up` then `down`, each f32 in the
/// loader's own layout, so a slab is a byte range and not a re-interpretation.
pub const ExpertGeometry = struct {
    gate_up_bytes: u64,
    down_bytes: u64,
    slab_bytes: u64,
    slab_stride_bytes: u64,

    pub fn derive(config: fixtures.RefConfig) ExpertGeometry {
        assert(config.hidden_size > 0);
        assert(config.moe_intermediate_size > 0);

        const hidden: u64 = config.hidden_size;
        const intermediate: u64 = config.moe_intermediate_size;
        const gate_up_bytes = 2 * intermediate * hidden * 4;
        const down_bytes = hidden * intermediate * 4;
        const expert_bytes = gate_up_bytes + down_bytes;
        return .{
            .gate_up_bytes = gate_up_bytes,
            .down_bytes = down_bytes,
            .slab_bytes = expert_bytes,
            .slab_stride_bytes = std.mem.alignForward(
                u64,
                expert_bytes,
                placement.page_bytes,
            ),
        };
    }
};

/// `mem/placement.zig`'s sharding policy instantiated on the fixture's counts.
///
/// The production table is the same policy at 48 layers and 512 experts; this
/// file never re-decides which component is replicated, tensor parallel or
/// expert parallel, only how many rows and experts the fixture's shapes give
/// each rank. Tensors are split over their output rows and experts into
/// consecutive shards, exactly as the production table does.
pub const Shard = struct {
    experts_per_rank: u32,
    dense_rows: u32,
    shared_rows: u32,
    head_rows: u32,
    layer_count: u32,

    /// Returns an error rather than a ragged split: a fixture whose counts do
    /// not divide over the ranks cannot run this policy, and a remainder would
    /// silently leave part of the model on no rank.
    pub fn derive(config: fixtures.RefConfig, ranks: u32) !Shard {
        assert(config.num_hidden_layers > 0);
        assert(ranks > 1);

        if (config.num_experts % ranks != 0 or
            config.hidden_size % ranks != 0 or
            config.shared_expert_intermediate_size % ranks != 0 or
            config.vocab_size % ranks != 0)
        {
            return error.shard_not_divisible;
        }
        const experts_per_rank = config.num_experts / ranks;
        if (experts_per_rank == 0) {
            return error.shard_not_divisible;
        }
        return .{
            .experts_per_rank = experts_per_rank,
            .dense_rows = config.hidden_size / ranks,
            .shared_rows = config.shared_expert_intermediate_size / ranks,
            .head_rows = config.vocab_size / ranks,
            .layer_count = config.num_hidden_layers,
        };
    }

    /// Expert slabs one rank owns: its shard in every layer.
    pub fn expertSlabs(self: Shard) u32 {
        assert(self.layer_count > 0);
        assert(self.experts_per_rank > 0);
        return self.layer_count * self.experts_per_rank;
    }

    /// First expert id of `rank`'s consecutive shard.
    pub fn expertFirst(self: Shard, rank: u32) u32 {
        assert(rank < rank_count);
        assert(self.experts_per_rank > 0);

        const first = rank * self.experts_per_rank;
        assert(first + self.experts_per_rank <= placement.experts_total);
        return first;
    }

    /// Whether `expert` is in `rank`'s shard.
    pub fn ownsExpert(self: Shard, rank: u32, expert: u32) bool {
        assert(rank < rank_count);
        assert(expert < placement.experts_total);

        const first = self.expertFirst(rank);
        return expert >= first and expert < first + self.experts_per_rank;
    }

    /// First output row of `rank`'s dense slice.
    pub fn denseFirst(self: Shard, rank: u32) u32 {
        assert(rank < rank_count);
        assert(self.dense_rows > 0);
        return rank * self.dense_rows;
    }

    /// First shared-expert intermediate column of `rank`'s slice.
    pub fn sharedFirst(self: Shard, rank: u32) u32 {
        assert(rank < rank_count);
        assert(self.shared_rows > 0);
        return rank * self.shared_rows;
    }

    /// First vocabulary row of `rank`'s `lm_head` slice.
    pub fn headFirst(self: Shard, rank: u32) u32 {
        assert(rank < rank_count);
        assert(self.head_rows > 0);
        return rank * self.head_rows;
    }

    /// Slab id of the `local`-th expert of `layer` in this rank's shard.
    pub fn slabId(self: Shard, layer: u32, local: u32) u32 {
        assert(layer < self.layer_count);
        assert(local < self.experts_per_rank);

        const id = layer * self.experts_per_rank + local;
        assert(id < self.expertSlabs());
        return id;
    }
};

/// How the tier splits this fixture's slabs between device and host.
///
/// The production split is `placement.place`'s answer from measured VRAM. The
/// fixture's slabs are 48 KiB, so both tiers are exercised explicitly instead
/// of letting the measured card hold all four of them and never fetch.
pub const TierSplit = struct {
    device_slabs: u32,
    host_slabs: u32,
};

/// One rank's exchange scratch: what the graph writes its slice into, and the
/// weight shard the shared expert's two projections read.
const Slices = struct {
    dense: cuda_context.DeviceBuffer = empty_buffer,
    shared_gate_up: cuda_context.DeviceBuffer = empty_buffer,
    shared_act: cuda_context.DeviceBuffer = empty_buffer,
    head: cuda_context.DeviceBuffer = empty_buffer,
    shared_weight: [fixtures.layers_max]load.Matrix = undefined,
    shared_weight_count: u32 = 0,

    fn free(self: *Slices) void {
        assert(self.shared_weight_count <= fixtures.layers_max);
        assert(@sizeOf(load.Matrix) > 0);

        for (self.shared_weight[0..self.shared_weight_count]) |matrix| {
            if (matrix.data.pointer != 0) {
                matrix.data.free();
            } else {
                assert(matrix.data.pointer == 0);
            }
        }
        if (self.head.pointer != 0) {
            self.head.free();
        }
        if (self.shared_act.pointer != 0) {
            self.shared_act.free();
        }
        if (self.shared_gate_up.pointer != 0) {
            self.shared_gate_up.free();
        }
        if (self.dense.pointer != 0) {
            self.dense.free();
        }
        self.* = Slices{};
    }
};

/// One rank thread's hook state: which forward it serves and which rank it is.
const Hook = struct {
    forward: *Forward,
    rank: u32,
};

/// A host barrier for the four rank threads.
///
/// Every gather must run after **all** ranks have queued their slice, because
/// a peer copy reads the other ranks' device memory. The wait carries a
/// timeout: a rank that failed early would otherwise deadlock the gate instead
/// of failing it, and a barrier timeout is fatal to the run either way (the
/// arrived count is not restored, so no later boundary can pass).
const Barrier = struct {
    mutex: std.Thread.Mutex = .{},
    condition: std.Thread.Condition = .{},
    parties: u32,
    arrived: u32 = 0,
    generation: u32 = 0,
    timeout_ns: u64 = 30 * std.time.ns_per_s,

    fn init(parties: u32) Barrier {
        assert(parties > 0);
        assert(parties <= placement.rank_count);
        return .{ .parties = parties };
    }

    fn wait(self: *Barrier) !void {
        assert(self.parties > 0);
        assert(self.arrived < self.parties);

        self.mutex.lock();
        defer self.mutex.unlock();
        const generation = self.generation;
        self.arrived += 1;
        if (self.arrived == self.parties) {
            self.arrived = 0;
            self.generation +%= 1;
            self.condition.broadcast();
        } else {
            while (self.generation == generation) {
                self.condition.timedWait(&self.mutex, self.timeout_ns) catch |err| switch (err) {
                    error.Timeout => return error.barrier_timeout,
                };
            }
        }
        assert(self.arrived < self.parties);
    }
};

/// One rank's thread body: its ids and whatever it failed with.
const RankRun = struct {
    forward: *Forward,
    rank: u32,
    prompt: []const i64,
    count: u32,
    out: []i64,
    failure: ?anyerror = null,
};

/// The four-rank graph: one `exec.Model` per card, the exchange slices the
/// tensor split needs, one host-DRAM tier per rank, and the hooks that turn a
/// peer copy into a gathered activation.
///
/// Threads: one per rank, each with its own primary context current. No two
/// threads touch the same rank's fields: `models[rank]`, `slices[rank]`,
/// `tiers[rank]` and `moves[rank]` are single-writer by construction.
pub const Forward = struct {
    allocator: std.mem.Allocator,
    options: Options,
    config: fixtures.RefConfig,
    shard: Shard,
    geometry: ExpertGeometry,
    tier_split: TierSplit,
    ranks: Ranks = undefined,
    ranks_ready: bool = false,
    models: [rank_count]exec.Model = undefined,
    model_count: u32 = 0,
    slices: [rank_count]Slices = undefined,
    slice_count: u32 = 0,
    tiers: [rank_count]ExpertTier = undefined,
    tier_count: u32 = 0,
    hooks: [rank_count]Hook = undefined,
    moves: [rank_count][placement.experts_per_rank]Move = undefined,
    slab_scratch: [rank_count][placement.experts_per_rank]u32 = undefined,
    barrier: Barrier,
    /// Test only: skip the peers' slices, to prove the exchange is load-bearing.
    sabotage: bool = false,
    placement_report: [32 * 1024]u8 = undefined,
    placement_report_len: usize = 0,

    /// Brings the four ranks up, prints the placement, builds one model per
    /// rank and stages each rank's expert shard into its tier.
    pub fn init(
        allocator: std.mem.Allocator,
        options: Options,
        split: TierSplit,
    ) !*Forward {
        assert(options.context_tokens > 0);
        assert(rank_count == placement.rank_count);

        const config = try fixtures.load_config(allocator);
        const shard = try Shard.derive(config, rank_count);
        if (shard.experts_per_rank > placement.experts_per_rank) {
            return error.shard_too_large;
        }
        if (split.device_slabs == 0 or split.host_slabs == 0) {
            return error.tier_split_mismatch;
        }
        if (split.device_slabs + split.host_slabs != shard.expertSlabs()) {
            return error.tier_split_mismatch;
        }

        const self = try allocator.create(Forward);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .options = options,
            .config = config,
            .shard = shard,
            .geometry = ExpertGeometry.derive(config),
            .tier_split = split,
            .barrier = Barrier.init(rank_count),
        };
        self.ranks = try Ranks.bringUp(options);
        self.ranks_ready = true;
        errdefer self.release();

        try self.announce();
        for (0..rank_count) |index| {
            const rank: u32 = @intCast(index);
            self.models[rank] = try exec.Model.initOnDevice(allocator, rank);
            self.model_count += 1;
        }
        for (0..rank_count) |index| {
            const rank: u32 = @intCast(index);
            try self.ranks.ranks[rank].context.set_current();
            try self.allocateSlices(rank);
            self.slice_count += 1;
        }
        for (0..rank_count) |index| {
            const rank: u32 = @intCast(index);
            try self.buildTier(rank);
            self.tier_count += 1;
        }
        for (0..rank_count) |index| {
            const rank: u32 = @intCast(index);
            self.buildCluster(rank);
        }
        assert(self.model_count == rank_count);
        assert(self.tier_count == rank_count);
        assert(self.slice_count == rank_count);
        return self;
    }

    pub fn deinit(self: *Forward) void {
        assert(self.allocator.vtable != undefined);
        assert(self.model_count <= rank_count);
        self.release();
        self.allocator.destroy(self);
    }

    /// The placement report `announce` rendered.
    pub fn placementReport(self: *const Forward) []const u8 {
        assert(self.placement_report_len > 0);
        assert(self.placement_report_len <= self.placement_report.len);
        return self.placement_report[0..self.placement_report_len];
    }

    /// Prints and verifies the placement, keeping the text for the gate.
    fn announce(self: *Forward) !void {
        assert(self.ranks_ready);
        assert(self.placement_report.len > 0);

        var stream = std.io.fixedBufferStream(self.placement_report[0..]);
        try self.ranks.announcePlacement(stream.writer());
        self.placement_report_len = stream.getWritten().len;
        assert(self.placement_report_len > 0);
    }

    /// Allocates one rank's exchange slices and copies its shared-expert
    /// weight shard out of the loaded checkpoint.
    fn allocateSlices(self: *Forward, rank: u32) !void {
        assert(rank < rank_count);
        assert(self.slice_count == rank);

        const tokens: u64 = exec.tokens_max;
        const hidden = self.config.hidden_size;
        const shared = self.config.shared_expert_intermediate_size;
        const rows = self.shard.shared_rows;
        const first = self.shard.sharedFirst(rank);

        var slices = Slices{};
        errdefer slices.free();
        slices.dense = try allocateDevice(tokens * self.shard.dense_rows * 4);
        slices.shared_gate_up = try allocateDevice(tokens * 2 * rows * 4);
        slices.shared_act = try allocateDevice(tokens * rows * 4);
        slices.head = try allocateDevice(tokens * self.shard.head_rows * 4);

        const model = &self.models[rank];
        const row_bytes: u64 = @as(u64, rows) * hidden * 4;
        for (0..self.shard.layer_count) |layer_index| {
            const layer: u32 = @intCast(layer_index);
            const source = model.weights.layers[layer].moe.shared_gate_up;
            const data = try allocateDevice(2 * row_bytes);
            slices.shared_weight[layer_index] = .{
                .data = data,
                .rows = 2 * rows,
                .cols = hidden,
            };
            slices.shared_weight_count += 1;
            // The gate rows then the up rows of the same intermediate
            // columns: the layout `moe_silu_mul` reads as one `[rows][2n]`.
            const gate_source = view(
                &source.data,
                @as(u64, first) * hidden * 4,
                row_bytes,
            );
            const up_source = view(
                &source.data,
                @as(u64, shared + first) * hidden * 4,
                row_bytes,
            );
            const gate_target = view(&data, 0, row_bytes);
            const up_target = view(&data, row_bytes, row_bytes);
            try model.ple.stream.copy_d_to_d(&gate_target, &gate_source, row_bytes);
            try model.ple.stream.copy_d_to_d(&up_target, &up_source, row_bytes);
            try model.ple.stream.synchronize();
        }
        self.slices[rank] = slices;
    }

    /// Builds one rank's expert tier and stages its shard into it: the first
    /// `device_slabs` slabs on the card, the rest pinned in host memory.
    fn buildTier(self: *Forward, rank: u32) !void {
        assert(rank < rank_count);
        assert(self.tier_count == rank);

        const context = &self.ranks.ranks[rank].context;
        try context.set_current();
        const geometry = SlabGeometry{
            .slab_count = self.shard.expertSlabs(),
            .slab_bytes = self.geometry.slab_bytes,
            .slab_stride_bytes = self.geometry.slab_stride_bytes,
        };
        var tier = try ExpertTier.initOn(
            self.allocator,
            geometry,
            self.tier_split.device_slabs,
            self.tier_split.host_slabs,
            &self.ranks.ranks[rank].prefetch,
        );
        errdefer tier.deinit(self.allocator);

        const slab = try self.allocator.alloc(u8, self.geometry.slab_bytes);
        defer self.allocator.free(slab);
        const model = &self.models[rank];
        var device_slot: u32 = 0;
        var host_slot: u32 = 0;
        for (0..self.shard.layer_count) |layer_index| {
            const layer: u32 = @intCast(layer_index);
            const weights = &model.weights.layers[layer].moe;
            var local: u32 = 0;
            while (local < self.shard.experts_per_rank) : (local += 1) {
                const expert = self.shard.expertFirst(rank) + local;
                assert(expert < self.config.num_experts);
                try model.state.downloadRaw(
                    &weights.expert_gate_up[expert].data,
                    slab[0..self.geometry.gate_up_bytes],
                );
                try model.state.downloadRaw(
                    &weights.expert_down[expert].data,
                    slab[self.geometry.gate_up_bytes..self.geometry.slab_bytes],
                );
                const slab_id = self.shard.slabId(layer, local);
                if (device_slot < self.tier_split.device_slabs) {
                    try tier.uploadHotSlab(slab_id, device_slot, slab);
                    device_slot += 1;
                } else {
                    try tier.stageColdSlab(slab_id, host_slot, slab);
                    host_slot += 1;
                }
            }
        }
        try tier.swap.verify();
        assert(device_slot == self.tier_split.device_slabs);
        assert(host_slot == self.tier_split.host_slabs);
        self.tiers[rank] = tier;
    }

    /// Points one rank's model at its slices and hooks.
    fn buildCluster(self: *Forward, rank: u32) void {
        assert(rank < rank_count);
        assert(self.tier_count == rank_count);
        assert(self.config.num_experts_per_tok > 0);

        const shard = self.shard;
        self.hooks[rank] = .{ .forward = self, .rank = rank };
        const slices = &self.slices[rank];
        self.models[rank].cluster = .{
            .user = @ptrCast(&self.hooks[rank]),
            .dense_first = shard.denseFirst(rank),
            .dense_rows = shard.dense_rows,
            .shared_first = shard.sharedFirst(rank),
            .shared_rows = shard.shared_rows,
            .shared_gate_up = slices.shared_weight[0..self.shard.layer_count],
            .head_first = shard.headFirst(rank),
            .head_rows = shard.head_rows,
            .expert_first = shard.expertFirst(rank),
            .expert_count = shard.experts_per_rank,
            .dense_slice = slices.dense,
            .shared_gate_up_slice = slices.shared_gate_up,
            .shared_act_slice = slices.shared_act,
            .head_slice = slices.head,
            .gather_dense = gatherDense,
            .gather_shared = gatherShared,
            .gather_head = gatherHead,
            .prepare_experts = prepareExperts,
            .expert_weights = expertWeights,
            .gather_experts = gatherExperts,
        };
    }

    /// Runs `count` greedy steps on every rank, concurrently, and returns each
    /// rank's ids. Every rank runs the same graph and must produce the same
    /// ids; the hooks are what make that a cross-device result rather than four
    /// copies of one.
    pub fn greedy(
        self: *Forward,
        prompt: []const i64,
        count: u32,
        out: [rank_count][]i64,
    ) !void {
        assert(prompt.len > 0);
        assert(count > 0);
        for (out) |row| {
            assert(row.len >= count);
        }

        var runs: [rank_count]RankRun = undefined;
        var threads: [rank_count]std.Thread = undefined;
        for (&runs, 0..) |*run, index| {
            run.* = .{
                .forward = self,
                .rank = @intCast(index),
                .prompt = prompt,
                .count = count,
                .out = out[index],
            };
        }
        var started: u32 = 0;
        while (started < rank_count) : (started += 1) {
            threads[started] = std.Thread.spawn(
                .{},
                runRank,
                .{&runs[started]},
            ) catch |err| {
                var joined: u32 = 0;
                while (joined < started) : (joined += 1) {
                    threads[joined].join();
                }
                return err;
            };
        }
        for (threads) |thread| {
            thread.join();
        }
        for (runs) |run| {
            if (run.failure) |err| {
                return err;
            } else {
                assert(run.failure == null);
            }
        }
        assert(started == rank_count);
    }

    /// One rank's thread body.
    fn runOne(self: *Forward, run: *RankRun) !void {
        assert(run.rank < rank_count);
        assert(run.out.len >= run.count);

        try self.ranks.ranks[run.rank].context.set_current();
        try generate.greedy(
            &self.models[run.rank],
            run.prompt,
            run.count,
            run.out,
            .cached,
        );
    }

    /// Releases every rank in reverse construction order.
    fn release(self: *Forward) void {
        assert(self.allocator.vtable != undefined);
        assert(self.tier_count <= rank_count);

        if (self.ranks_ready) {
            var index: u32 = 0;
            while (index < rank_count) : (index += 1) {
                const rank = index;
                self.ranks.ranks[rank].context.set_current() catch |err| {
                    std.log.err(
                        "rank {d}: cannot make its context current to tear down: {s}",
                        .{ rank, @errorName(err) },
                    );
                    continue;
                };
                if (rank < self.model_count) {
                    self.models[rank].deinit();
                } else {
                    assert(rank >= self.model_count);
                }
                if (rank < self.slice_count) {
                    self.slices[rank].free();
                } else {
                    assert(rank >= self.slice_count);
                }
                if (rank < self.tier_count) {
                    self.tiers[rank].deinit(self.allocator);
                } else {
                    assert(rank >= self.tier_count);
                }
            }
            self.ranks.deinit();
        } else {
            assert(!self.ranks_ready);
        }
        self.placement_report_len = 0;
    }
};

// --- hook implementations ----------------------------------------------------

fn castHook(user: *anyopaque) *Hook {
    assert(@intFromPtr(user) != 0);
    assert(@alignOf(Hook) > 1);
    return @ptrCast(@alignCast(user));
}

fn runRank(run: *RankRun) void {
    assert(run.forward.model_count == rank_count);
    assert(run.rank < rank_count);
    run.forward.runOne(run) catch |err| {
        run.failure = err;
    };
}

/// Whether this rank's own slice is complete, then every rank's is.
fn reachBoundary(f: *Forward, rank: u32) !void {
    assert(rank < rank_count);
    assert(f.model_count == rank_count);

    try f.models[rank].ple.stream.synchronize();
    try f.barrier.wait();
}

/// Copies every rank's disjoint row slice into `target`'s column band.
///
/// A peer copy runs on this rank's `comms` stream and reads the peer's buffer
/// directly, so the peer must have finished writing it (the barrier above) and
/// this rank must have peer access to it (checked here, not assumed).
fn gatherSlices(
    f: *Forward,
    rank: u32,
    target: *const cuda_context.DeviceBuffer,
    sources: []const cuda_context.DeviceBuffer,
    tokens: u32,
    columns: u32,
    slice_columns: u32,
) !void {
    assert(rank < rank_count);
    assert(sources.len == rank_count);
    assert(tokens > 0);
    assert(slice_columns * rank_count == columns);

    const column_bytes: u64 = @as(u64, slice_columns) * 4;
    const row_bytes: u64 = @as(u64, columns) * 4;
    const context = &f.ranks.ranks[rank].context;
    const stream = &f.ranks.ranks[rank].comms;
    for (0..rank_count) |peer_index| {
        const peer: u32 = @intCast(peer_index);
        if (f.sabotage and peer != rank) {
            continue;
        } else if (peer != rank) {
            assert(f.ranks.ranks[rank].reachable[peer]);
        } else {
            assert(peer == rank);
        }
        if (sources[peer].bytes < @as(u64, tokens) * column_bytes) {
            return error.slice_too_small;
        } else {
            assert(sources[peer].bytes >= @as(u64, tokens) * column_bytes);
        }
        for (0..tokens) |token_index| {
            const token: u64 = @intCast(token_index);
            const source = view(
                &sources[peer],
                token * column_bytes,
                column_bytes,
            );
            const destination = view(
                target,
                token * row_bytes + @as(u64, peer) * column_bytes,
                column_bytes,
            );
            if (peer == rank) {
                try stream.copy_d_to_d(&destination, &source, column_bytes);
            } else {
                try stream.copy_peer(
                    &destination,
                    context,
                    &source,
                    &f.ranks.ranks[peer].context,
                    column_bytes,
                );
            }
        }
    }
    try stream.synchronize();
}

fn gatherDense(
    user: *anyopaque,
    target: *const cuda_context.DeviceBuffer,
    tokens: u32,
    columns: u32,
) anyerror!void {
    const hook = castHook(user);
    const f = hook.forward;
    assert(tokens > 0);
    assert(columns > 0);
    try reachBoundary(f, hook.rank);
    var sources: [rank_count]cuda_context.DeviceBuffer = undefined;
    for (&sources, 0..) |*source, index| {
        source.* = f.slices[index].dense;
    }
    try gatherSlices(
        f,
        hook.rank,
        target,
        sources[0..],
        tokens,
        columns,
        f.shard.dense_rows,
    );
}

fn gatherShared(
    user: *anyopaque,
    target: *const cuda_context.DeviceBuffer,
    tokens: u32,
    columns: u32,
) anyerror!void {
    const hook = castHook(user);
    const f = hook.forward;
    assert(tokens > 0);
    assert(columns > 0);
    try reachBoundary(f, hook.rank);
    var sources: [rank_count]cuda_context.DeviceBuffer = undefined;
    for (&sources, 0..) |*source, index| {
        source.* = f.slices[index].shared_act;
    }
    try gatherSlices(
        f,
        hook.rank,
        target,
        sources[0..],
        tokens,
        columns,
        f.shard.shared_rows,
    );
}

fn gatherHead(
    user: *anyopaque,
    target: *const cuda_context.DeviceBuffer,
    tokens: u32,
    columns: u32,
) anyerror!void {
    const hook = castHook(user);
    const f = hook.forward;
    assert(tokens > 0);
    assert(columns > 0);
    try reachBoundary(f, hook.rank);
    var sources: [rank_count]cuda_context.DeviceBuffer = undefined;
    for (&sources, 0..) |*source, index| {
        source.* = f.slices[index].head;
    }
    try gatherSlices(
        f,
        hook.rank,
        target,
        sources[0..],
        tokens,
        columns,
        f.shard.head_rows,
    );
}

/// Gathers every rank's expert range of `moe_expert_out` into `target`. The
/// ranges are consecutive, so one copy per peer carries a whole shard.
fn gatherExperts(
    user: *anyopaque,
    target: *const cuda_context.DeviceBuffer,
    tokens: u32,
    hidden: u32,
) anyerror!void {
    const hook = castHook(user);
    const f = hook.forward;
    const rank = hook.rank;
    assert(tokens > 0);
    assert(hidden > 0);

    try reachBoundary(f, rank);
    const per_rank = f.shard.experts_per_rank;
    const slot_bytes: u64 = @as(u64, per_rank) * tokens * hidden * 4;
    const context = &f.ranks.ranks[rank].context;
    const stream = &f.ranks.ranks[rank].comms;
    for (0..rank_count) |peer_index| {
        const peer: u32 = @intCast(peer_index);
        if (f.sabotage and peer != rank) {
            continue;
        } else if (peer != rank) {
            assert(f.ranks.ranks[rank].reachable[peer]);
        } else {
            assert(peer == rank);
        }
        const offset: u64 = @as(u64, peer) * per_rank * tokens * hidden * 4;
        const source = view(
            &f.models[peer].state.moe_expert_out,
            offset,
            slot_bytes,
        );
        const destination = view(target, offset, slot_bytes);
        if (peer == rank) {
            try stream.copy_d_to_d(&destination, &source, slot_bytes);
        } else {
            try stream.copy_peer(
                &destination,
                context,
                &source,
                &f.ranks.ranks[peer].context,
                slot_bytes,
            );
        }
    }
    try stream.synchronize();
}

/// Fetches the selected experts of this rank's shard, on the rank's prefetch
/// stream, before the layer's expert products read their slots.
fn prepareExperts(
    user: *anyopaque,
    layer: u32,
    selected: []const u32,
) anyerror!void {
    const hook = castHook(user);
    const f = hook.forward;
    const rank = hook.rank;
    assert(layer < f.shard.layer_count);
    assert(selected.len > 0);

    // A slot the previous layer's products may still be reading must not be
    // overwritten: the prefetch stream is independent of the compute stream.
    try f.models[rank].ple.stream.synchronize();
    const scratch = f.slab_scratch[rank][0..];
    var count: u32 = 0;
    for (selected) |expert| {
        if (f.shard.ownsExpert(rank, expert)) {
            const local = expert - f.shard.expertFirst(rank);
            if (count >= scratch.len) {
                return error.selection_too_large;
            } else {
                assert(count < scratch.len);
            }
            scratch[count] = f.shard.slabId(layer, local);
            count += 1;
        } else {
            assert(!f.shard.ownsExpert(rank, expert));
        }
    }
    if (count == 0) {
        return;
    } else {
        assert(count > 0);
    }
    const moves = f.moves[rank][0..];
    const planned = try f.tiers[rank].planLayer(scratch[0..count], moves);
    if (planned == 0) {
        return;
    } else {
        assert(planned > 0);
    }
    try f.tiers[rank].prefetch(moves[0..planned]);
    try f.tiers[rank].waitReady();
}

/// The weights of one selected expert this rank owns, from the tier's slot.
fn expertWeights(
    user: *anyopaque,
    layer: u32,
    expert: u32,
) anyerror!exec.ExpertSlice {
    const hook = castHook(user);
    const f = hook.forward;
    const rank = hook.rank;
    assert(layer < f.shard.layer_count);
    assert(expert < f.config.num_experts);

    if (!f.shard.ownsExpert(rank, expert)) {
        return error.expert_not_owned;
    } else {
        assert(f.shard.ownsExpert(rank, expert));
    }
    const local = expert - f.shard.expertFirst(rank);
    const slab = f.shard.slabId(layer, local);
    const slot = f.tiers[rank].residentSlot(slab) orelse
        return error.expert_not_resident;
    const arena = f.tiers[rank].slotBuffer(slot);
    const hidden = f.config.hidden_size;
    const intermediate = f.config.moe_intermediate_size;
    return .{
        .gate_up = .{
            .data = view(&arena, 0, f.geometry.gate_up_bytes),
            .rows = 2 * intermediate,
            .cols = hidden,
        },
        .down = .{
            .data = view(&arena, f.geometry.gate_up_bytes, f.geometry.down_bytes),
            .rows = hidden,
            .cols = intermediate,
        },
    };
}

// --- comms bench -------------------------------------------------------------

/// Repetitions per ordered pair: the issue asks for a median of 6.
pub const comms_repetitions: usize = 6;

/// What this harness asserts about a single copy: a copy slower than this is a
/// link that does not work, not one that is slow.
pub const comms_liveness_us: f64 = 10_000.0;

/// The issue's per-token bound: one token's activation across four stage
/// boundaries.
pub const token_comms_bound_us: f64 = 1_500.0;

/// Stage boundaries one token crosses: four ranks in a ring, one hop each.
pub const token_stages: u32 = 4;

/// Bytes one token's activation occupies per boundary: hidden 2560 in bf16.
pub const token_activation_bytes: u64 = 5_120;

/// Payloads the bench copies: one token's BF16 hidden state (2 560 values) and
/// one MiB, a prefill chunk's worth of dispatch traffic.
pub const comms_payload_bytes = [_]u64{ token_activation_bytes, 1_048_576 };

/// Where the report is written, relative to the build root and to the
/// repository root (the gate may run from either).
const comms_report_paths = [_][]const u8{
    "../bench/comm.json",
    "bench/comm.json",
};

/// One ordered pair's peer-copy latency, in microseconds.
pub const PairLatency = struct {
    from: u32,
    to: u32,
    median_us: f64,
    max_us: f64,
};

/// One token's four stage boundaries: the ring's per-hop medians and the sum
/// the issue's bound is stated over.
pub const TokenChain = struct {
    hops: [token_stages]PairLatency,
    total_us: f64,
    bytes_total: u64,
    within_bound: bool,
};

/// Measures the per-token chain the 1.5 ms bound is about: four stage
/// boundaries, one 5,120 B activation copy per hop, four ranks in a ring.
///
/// The hops are issued and waited one at a time, which is the conservative
/// reading of a pipeline boundary: each stage's compute cannot start before the
/// previous stage's activation arrived. The total is the sum of the per-hop
/// medians, so a figure that fits means no boundary serialised.
fn measureTokenChain(
    ranks: *Ranks,
    scratch: []const cuda_context.DeviceBuffer,
) !TokenChain {
    assert(ranks.ranks.len == rank_count);
    assert(scratch.len == rank_count);
    assert(token_stages == rank_count);
    assert(token_activation_bytes <= scratch[0].bytes);

    const staged = try cuda_context.PinnedHostBuffer.allocate(
        std.mem.alignForward(u64, token_activation_bytes, placement.page_bytes),
    );
    defer staged.free();

    var chain = TokenChain{
        .hops = undefined,
        .total_us = 0.0,
        .bytes_total = 0,
        .within_bound = false,
    };
    for (0..token_stages) |index| {
        const from: u32 = @intCast(index);
        const to: u32 = (from + 1) % rank_count;
        chain.hops[index] = try measurePeerCopy(
            &ranks.ranks[from],
            &ranks.ranks[to],
            token_activation_bytes,
            &scratch[from],
            &scratch[to],
            &staged,
        );
        chain.total_us += chain.hops[index].median_us;
        chain.bytes_total += token_activation_bytes;
    }
    chain.within_bound = chain.total_us <= token_comms_bound_us;
    assert(chain.total_us >= 0.0);
    assert(chain.bytes_total == token_stages * token_activation_bytes);
    return chain;
}

/// How many ordered pairs could be measured: every pair that can reach.
pub fn reachablePairCount(ranks: *const Ranks) u32 {
    assert(ranks.ranks.len == rank_count);
    assert(rank_count > 1);

    var count: u32 = 0;
    for (&ranks.ranks) |*from| {
        for (&ranks.ranks) |*to| {
            if (from.index != to.index and from.reachable[to.index]) {
                count += 1;
            }
        }
    }
    assert(count <= rank_count * (rank_count - 1));
    return count;
}

/// Times one ordered peer copy `comms_repetitions` times on the source rank's
/// comms stream, then reads the copy back to prove the bytes arrived.
fn measurePeerCopy(
    from: *Rank,
    to: *Rank,
    bytes: u64,
    source: *const cuda_context.DeviceBuffer,
    destination: *const cuda_context.DeviceBuffer,
    staged: *const cuda_context.PinnedHostBuffer,
) !PairLatency {
    assert(from.index != to.index);
    assert(bytes > 0);
    assert(bytes <= source.bytes);
    assert(bytes <= destination.bytes);

    const length: usize = @intCast(bytes);
    // One pattern byte per source rank, so a stale copy is a visible mismatch.
    const pattern: u8 = @intCast(0x40 + from.index);
    try from.context.set_current();
    @memset(staged.slice()[0..length], pattern);
    try from.comms.copy_h_to_d(source, staged, bytes);
    try from.comms.synchronize();

    const start = try cuda_context.Event.create();
    defer start.destroy();
    const end = try cuda_context.Event.create();
    defer end.destroy();

    var samples: [comms_repetitions]f64 = undefined;
    for (&samples) |*sample| {
        try start.record(&from.comms);
        try from.comms.copy_peer(
            destination,
            &to.context,
            source,
            &from.context,
            bytes,
        );
        try end.record(&from.comms);
        try from.comms.synchronize();
        const nanos = try cuda_context.Event.elapsed_nanoseconds(&start, &end);
        sample.* = @as(f64, @floatFromInt(nanos)) / std.time.ns_per_us;
    }

    // The copies just timed have to have moved the bytes to the peer's card.
    try to.context.set_current();
    try to.compute.copy_d_to_h(staged, destination, bytes);
    try to.compute.synchronize();
    if (equalPattern(staged.slice()[0..length], pattern)) {
        assert(staged.bytes >= bytes);
    } else {
        std.log.err(
            "rank {d} to rank {d}: {d} bytes did not arrive",
            .{ from.index, to.index, bytes },
        );
        return error.peer_copy_mismatch;
    }

    std.mem.sort(f64, samples[0..], {}, std.sort.asc(f64));
    const middle = comms_repetitions / 2;
    assert(samples[middle] >= samples[0]);
    return .{
        .from = from.index,
        .to = to.index,
        .median_us = (samples[middle - 1] + samples[middle]) / 2.0,
        .max_us = samples[comms_repetitions - 1],
    };
}

/// Measures every reachable ordered pair at one payload size, writes them as a
/// JSON array and returns the slowest copy it saw, in microseconds.
fn writePayloadLatency(
    ranks: *Ranks,
    bytes: u64,
    scratch: []const cuda_context.DeviceBuffer,
    writer: anytype,
) !f64 {
    assert(bytes > 0);
    assert(scratch.len == rank_count);
    assert(bytes <= scratch[0].bytes);

    const staged_bytes = std.mem.alignForward(
        u64,
        bytes,
        placement.page_bytes,
    );
    const staged = try cuda_context.PinnedHostBuffer.allocate(staged_bytes);
    defer staged.free();

    try writer.print(
        "    {{\n      \"bytes\": {d},\n      \"pairs\": [\n",
        .{bytes},
    );
    var slowest_us: f64 = 0.0;
    var first = true;
    for (&ranks.ranks) |*from| {
        for (&ranks.ranks) |*to| {
            if (from.index == to.index) {
                // A rank copies to its peers, not to itself.
            } else if (!from.reachable[to.index]) {
                std.log.warn(
                    "bench: rank {d} cannot reach rank {d}, not measured",
                    .{ from.index, to.index },
                );
            } else {
                const pair = try measurePeerCopy(
                    from,
                    to,
                    bytes,
                    &scratch[from.index],
                    &scratch[to.index],
                    &staged,
                );
                if (first) {
                    // The separator goes before every pair but the first.
                } else {
                    try writer.writeAll(",\n");
                }
                first = false;
                try writer.print(
                    "        {{\"from\": {d}, \"to\": {d}, " ++
                        "\"median_us\": {d:.3}, \"max_us\": {d:.3}}}",
                    .{ pair.from, pair.to, pair.median_us, pair.max_us },
                );
                if (pair.max_us > slowest_us) {
                    slowest_us = pair.max_us;
                } else {
                    assert(pair.max_us <= slowest_us);
                }
            }
        }
    }
    try writer.writeAll("\n      ]\n    }");
    assert(!first);
    return slowest_us;
}

/// Measures the peer-copy matrix the forward pass's dispatch and combine will
/// ride on, writes it as JSON and returns the slowest copy, in microseconds.
///
/// One peer copy is all this measures per ordered pair; the per-token block
/// below sums the four stage boundaries at the production activation size. It
/// is **indicative**: two other lanes hold GPUs, so the figure carries its
/// contention in the report and must be re-taken on a quiet box before it is
/// quoted. The fused dense all-reduce and the MoE combine all-reduce are not
/// measured because they are not implemented: this tree gathers disjoint slices
/// for bandwidth reasons and the reduction kernel is I-030.
pub const CommsResult = struct {
    slowest_us: f64,
    token: TokenChain,
};

pub fn writeCommsBench(ranks: *Ranks, writer: anytype) !CommsResult {
    assert(ranks.ranks.len == rank_count);
    assert(comms_payload_bytes.len > 0);
    assert(comms_report_paths.len > 0);

    const scratch_bytes = comms_payload_bytes[comms_payload_bytes.len - 1];
    var scratch: [rank_count]cuda_context.DeviceBuffer = undefined;
    for (&ranks.ranks) |*rank| {
        try rank.context.set_current();
        scratch[rank.index] = try cuda_context.DeviceBuffer.allocate(
            scratch_bytes,
        );
    }
    defer freeScratch(ranks, scratch[0..]);

    try writer.writeAll(
        "{\n  \"harness\": \"core/src/model/multi.zig comms bench\",\n" ++
            "  \"measured\": \"peer copy per ordered pair, through the hw " ++
            "layer, on the source rank's comms stream\",\n",
    );
    try writer.print(
        "  \"repetitions\": {d},\n  \"measured_pairs\": {d},\n",
        .{ comms_repetitions, reachablePairCount(ranks) },
    );
    try writer.writeAll("  \"payloads\": [\n");
    var slowest_us: f64 = 0.0;
    for (comms_payload_bytes, 0..) |bytes, index| {
        if (index > 0) {
            try writer.writeAll(",\n");
        }
        const payload_slowest = try writePayloadLatency(
            ranks,
            bytes,
            scratch[0..],
            writer,
        );
        if (payload_slowest > slowest_us) {
            slowest_us = payload_slowest;
        }
    }
    const chain = try measureTokenChain(ranks, scratch[0..]);
    try writer.print(
        "\n  ],\n  \"liveness_bound_ms\": {d:.1},\n" ++
            "  \"liveness_ok\": {s},\n" ++
            "  \"per_token\": {{\n" ++
            "    \"topology\": \"ring: one activation copy per stage " ++
            "boundary, four ranks\",\n" ++
            "    \"contention\": \"indicative: measured while other lanes " ++
            "may hold the GPUs; re-take on a quiet box before quoting\",\n" ++
            "    \"stages\": {d},\n" ++
            "    \"activation_bytes\": {d},\n" ++
            "    \"total_bytes\": {d},\n" ++
            "    \"hops\": [\n",
        .{
            comms_liveness_us / 1_000.0,
            if (slowest_us <= comms_liveness_us) "true" else "false",
            token_stages,
            token_activation_bytes,
            chain.bytes_total,
        },
    );
    for (&chain.hops, 0..) |*hop, index| {
        if (index > 0) {
            try writer.writeAll(",\n");
        }
        try writer.print(
            "      {{\"from\": {d}, \"to\": {d}, \"median_us\": {d:.3}, " ++
                "\"max_us\": {d:.3}}}",
            .{ hop.from, hop.to, hop.median_us, hop.max_us },
        );
    }
    // `liveness_ok` is a link check and `within_bound` is the issue's gate.
    // The two are separate keys because a working link can still be slow, and
    // `not_measured` keeps the two reductions this tree does not have visible
    // instead of renaming the gather into them.
    try writer.print(
        "\n    ],\n" ++
            "    \"total_median_us\": {d:.3},\n" ++
            "    \"bound_us\": {d:.1},\n" ++
            "    \"within_bound\": {s}\n" ++
            "  }},\n" ++
            "  \"not_measured\": [\n" ++
            "    \"fused dense all-reduce: not implemented, this tree " ++
            "gathers disjoint slices for bandwidth reasons and the " ++
            "reduction kernel is I-030\",\n" ++
            "    \"MoE combine all-reduce: not implemented, this tree " ++
            "gathers per-expert outputs and the reduction kernel is " ++
            "I-030\"\n" ++
            "  ]\n}}\n",
        .{
            chain.total_us,
            token_comms_bound_us,
            if (chain.within_bound) "true" else "false",
        },
    );
    assert(slowest_us >= 0.0);
    assert(chain.total_us >= 0.0);
    return .{ .slowest_us = slowest_us, .token = chain };
}

/// Frees the bench scratch, each buffer in the context that owns it.
fn freeScratch(ranks: *Ranks, scratch: []const cuda_context.DeviceBuffer) void {
    assert(scratch.len == rank_count);
    assert(ranks.ranks.len == rank_count);

    for (&ranks.ranks) |*rank| {
        rank.context.set_current() catch |err| {
            std.log.err(
                "rank {d}: cannot free the bench scratch: {s}",
                .{ rank.index, @errorName(err) },
            );
            continue;
        };
        scratch[rank.index].free();
    }
}

/// Writes the bench report to the first of `paths` it can create, so it lands
/// in `bench/comm.json` from either working directory.
///
/// Fatal on failure: a report no path accepted means the gate would go green
/// with no numbers on disk, so this returns an error instead of printing the
/// document and hoping someone reads the scrollback. `paths` is a parameter
/// rather than `comms_report_paths` itself so a test can hand it a list nothing
/// can be created under.
fn writeCommsReport(
    report: []const u8,
    paths: []const []const u8,
) ReportError!void {
    assert(report.len > 0);
    assert(paths.len > 0);

    for (paths) |path| {
        const file = std.fs.cwd().createFile(path, .{}) catch |err| {
            std.log.warn(
                "cannot create {s}: {s}",
                .{ path, @errorName(err) },
            );
            continue;
        };
        defer file.close();
        file.writeAll(report) catch |err| {
            std.log.warn("cannot write {s}: {s}", .{ path, @errorName(err) });
            return ReportError.report_not_written;
        };
        std.debug.print("wrote {s} ({d} bytes)\n", .{ path, report.len });
        return;
    }
    // Warn, not err: a log line cannot fail a build, and the error below is
    // what does. An `err` line here would also fail the test that proves this
    // failure is fatal.
    std.log.warn(
        "the comms report reached none of {d} paths, first {s}",
        .{ paths.len, paths[0] },
    );
    return ReportError.report_not_written;
}

/// Index of the first id where two greedy runs differ, or null when they agree.
/// Diagnostic only: the gate is the committed golden ids.
fn firstIdDifference(left: []const i64, right: []const i64) ?usize {
    assert(left.len == right.len);
    assert(left.len > 0);

    for (left, right, 0..) |a, b, index| {
        if (a != b) {
            return index;
        } else {
            assert(a == b);
        }
    }
    return null;
}

// --- tests ------------------------------------------------------------------

test "parseOptions reads the flags the spec names" {
    const defaults = try parseOptions(&.{});
    try std.testing.expect(defaults.dram_tier_enabled);
    try std.testing.expectEqual(@as(u32, 0), defaults.hot_slab_count);
    try std.testing.expectEqual(
        placement.max_context_tokens,
        defaults.context_tokens,
    );

    const tuned = try parseOptions(&.{
        "--no-dram-tier",
        "--expert-hot-slabs=4000",
        "--context-tokens=8192",
        "prompt.txt",
    });
    try std.testing.expect(!tuned.dram_tier_enabled);
    try std.testing.expectEqual(@as(u32, 4_000), tuned.hot_slab_count);
    try std.testing.expectEqual(@as(u32, 8_192), tuned.context_tokens);
}

test "parseOptions refuses flags and values it cannot honour" {
    try std.testing.expectError(
        OptionError.unknown_argument,
        parseOptions(&.{"--dram-tier=maybe"}),
    );
    try std.testing.expectError(
        OptionError.argument_value_missing,
        parseOptions(&.{"--expert-hot-slabs="}),
    );
    // A hot set of zero slabs is a tier refusal, not a silent default.
    try std.testing.expectError(
        OptionError.argument_value_invalid,
        parseOptions(&.{"--expert-hot-slabs=0"}),
    );
    try std.testing.expectError(
        OptionError.argument_value_invalid,
        parseOptions(&.{"--context-tokens=lots"}),
    );
}

test "a comms report no path accepts is an error, not a printed document" {
    // The gate reads bench/comm.json; a run that could not write one has to
    // fail here instead of scrolling the numbers past the operator.
    try std.testing.expectError(
        ReportError.report_not_written,
        writeCommsReport("{}\n", &.{"no-such-dir/comm.json"}),
    );
    // The control: the same report does land when a path exists.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const directory = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(directory);
    const path = try std.fs.path.join(
        std.testing.allocator,
        &.{ directory, "comm.json" },
    );
    defer std.testing.allocator.free(path);
    try writeCommsReport("{}\n", &.{path});
    const written = try tmp.dir.statFile("comm.json");
    try std.testing.expectEqual(@as(u64, 3), written.size);
}

test "slab ids are dense over the shard and invert back" {
    try std.testing.expectEqual(
        @as(u32, 6_144),
        placement.expert_slabs_per_rank,
    );
    try std.testing.expectEqual(@as(u32, 0), slabId(0, 0));
    try std.testing.expectEqual(@as(u32, 127), slabId(0, 127));
    try std.testing.expectEqual(@as(u32, 128), slabId(1, 0));
    try std.testing.expectEqual(@as(u32, 6_143), slabId(47, 127));

    var slab: u32 = 0;
    while (slab < placement.expert_slabs_per_rank) : (slab += 1) {
        try std.testing.expectEqual(
            slab,
            slabId(slabLayer(slab), slabExpertInShard(slab)),
        );
    }
}

test "the router's expert ids map onto consecutive shards without comms" {
    // Rank 0 owns 0..127, rank 3 owns 384..511, and nothing is in two shards.
    try std.testing.expectEqual(@as(?u32, 0), expertInShard(0, 0));
    try std.testing.expectEqual(@as(?u32, 127), expertInShard(0, 127));
    try std.testing.expectEqual(@as(?u32, null), expertInShard(0, 128));
    try std.testing.expectEqual(@as(?u32, 0), expertInShard(3, 384));
    try std.testing.expectEqual(@as(?u32, 127), expertInShard(3, 511));

    var owners: [placement.experts_total]u32 = [_]u32{0} **
        placement.experts_total;
    var expert: u32 = 0;
    while (expert < placement.experts_total) : (expert += 1) {
        var rank_index: u32 = 0;
        while (rank_index < rank_count) : (rank_index += 1) {
            if (expertInShard(rank_index, expert) != null) {
                owners[expert] += 1;
            } else {
                assert(expertInShard(rank_index, expert) == null);
            }
        }
    }
    for (owners) |count| {
        try std.testing.expectEqual(@as(u32, 1), count);
    }
}

/// A tier of `slab_count` slabs split into two halves, for the pure tests.
fn testSwapSet(
    allocator: std.mem.Allocator,
    slab_count: u32,
    device_slots: u32,
    host_slots: u32,
) !SwapSet {
    assert(slab_count == device_slots + host_slots);
    assert(slab_count > 0);

    return SwapSet.init(allocator, .{
        .slab_count = slab_count,
        .slab_bytes = 4096,
        .slab_stride_bytes = 4096,
    }, device_slots, host_slots);
}

test "the swap set keeps every slab in exactly one tier" {
    var set = try testSwapSet(std.testing.allocator, 6, 4, 2);
    defer set.deinit(std.testing.allocator);

    try set.admit(0, 0);
    try set.admit(1, 1);
    try set.admit(2, 2);
    try set.admit(3, 3);
    try set.stage(4, 0);
    try set.stage(5, 1);
    try set.verify();
    try std.testing.expectEqual(@as(u32, 4), set.deviceSlotsInUse());
    try std.testing.expectEqual(@as(u32, 2), set.hostSlotsInUse());
    // Slot 3 was admitted last, so it is the most recently used; slot 0 is the
    // eviction candidate.
    try std.testing.expectEqual(@as(u32, 3), set.head);
    try std.testing.expectEqual(@as(u32, 0), set.tail);
}

test "a cold slab evicts the least recently used slot and swaps its host slot" {
    var set = try testSwapSet(std.testing.allocator, 6, 4, 2);
    defer set.deinit(std.testing.allocator);

    try set.admit(0, 0);
    try set.admit(1, 1);
    try set.admit(2, 2);
    try set.admit(3, 3);
    try set.stage(4, 0);
    try set.stage(5, 1);

    var moves: [8]Move = undefined;
    // Slab 1 is used, so slab 0 becomes the eviction candidate.
    const touched = try set.planLayer(&.{1}, moves[0..]);
    try std.testing.expectEqual(@as(u32, 0), touched);
    // Slab 4 is cold: it takes slot 0 and slab 0 takes host slot 0.
    const fetched = try set.planLayer(&.{4}, moves[0..]);
    try std.testing.expectEqual(@as(u32, 1), fetched);
    try std.testing.expectEqual(@as(u32, 4), moves[0].fetch_slab);
    try std.testing.expectEqual(@as(u32, 0), moves[0].device_slot);
    try std.testing.expectEqual(@as(u32, 0), moves[0].host_slot);
    try std.testing.expectEqual(@as(u32, 0), moves[0].displaced_slab);
    try set.verify();

    // Slab 1 was touched last, so slot 2 (slab 2) is next.
    const second = try set.planLayer(&.{5}, moves[0..]);
    try std.testing.expectEqual(@as(u32, 1), second);
    try std.testing.expectEqual(@as(u32, 2), moves[0].device_slot);
    try std.testing.expectEqual(@as(u32, 1), moves[0].host_slot);
    try std.testing.expectEqual(@as(u32, 2), moves[0].displaced_slab);
    try set.verify();
    try std.testing.expectEqual(@as(u32, 4), set.deviceSlotsInUse());
    try std.testing.expectEqual(@as(u32, 2), set.hostSlotsInUse());
    try std.testing.expectEqual(@as(u64, 2), set.planned_moves);
}

test "a resident slab is touched, never fetched twice in one result" {
    var set = try testSwapSet(std.testing.allocator, 4, 2, 2);
    defer set.deinit(std.testing.allocator);

    try set.admit(0, 0);
    try set.admit(1, 1);
    try set.stage(2, 0);
    try set.stage(3, 1);

    var moves: [8]Move = undefined;
    // Two tokens in the same batch may select the same expert once each; only
    // the first selection costs a fetch.
    const planned = try set.planLayer(&.{ 0, 2, 2, 0 }, moves[0..]);
    try std.testing.expectEqual(@as(u32, 1), planned);
    try std.testing.expectEqual(@as(u32, 2), moves[0].fetch_slab);
    try std.testing.expectEqual(@as(u32, 1), moves[0].displaced_slab);
    try set.verify();
}

test "a slab nobody loaded is refused instead of fetched from nowhere" {
    var set = try testSwapSet(std.testing.allocator, 4, 2, 2);
    defer set.deinit(std.testing.allocator);

    try set.admit(0, 0);
    try set.admit(1, 1);
    try set.stage(2, 0);
    // Slab 3 is in neither tier: the loader has not run for it yet.
    var moves: [4]Move = undefined;
    try std.testing.expectError(
        SwapError.slab_not_loaded,
        set.planLayer(&.{3}, moves[0..]),
    );
    try std.testing.expectError(
        SwapError.bookkeeping_inconsistent,
        set.verify(),
    );
    // Once it has bytes, the same selection swaps it in.
    try set.stage(3, 1);
    try set.verify();
    try std.testing.expectEqual(
        @as(u32, 1),
        try set.planLayer(&.{3}, moves[0..]),
    );
}

test "a move buffer that cannot hold the plan is an error, not a truncation" {
    var set = try testSwapSet(std.testing.allocator, 6, 3, 3);
    defer set.deinit(std.testing.allocator);

    try set.admit(0, 0);
    try set.admit(1, 1);
    try set.admit(2, 2);
    try set.stage(3, 0);
    try set.stage(4, 1);
    try set.stage(5, 2);

    var one_move: [1]Move = undefined;
    try std.testing.expectError(
        SwapError.move_buffer_too_small,
        set.planLayer(&.{ 3, 4 }, one_move[0..]),
    );
    // Zero is not a valid slab id, but 6144 is past the end of the shard.
    try std.testing.expectError(
        SwapError.slab_out_of_range,
        set.planLayer(&.{6}, one_move[0..]),
    );
}

test "a slot may hold one slab and a slab one slot" {
    var set = try testSwapSet(std.testing.allocator, 4, 2, 2);
    defer set.deinit(std.testing.allocator);

    try set.admit(0, 0);
    try std.testing.expectError(SwapError.slot_occupied, set.admit(1, 0));
    try std.testing.expectError(SwapError.slab_already_placed, set.admit(0, 1));
    try std.testing.expectError(SwapError.slot_out_of_range, set.admit(1, 2));
    try set.admit(1, 1);
    try set.stage(2, 0);
    try std.testing.expectError(SwapError.slot_occupied, set.stage(3, 0));
    try std.testing.expectError(SwapError.slab_already_placed, set.stage(2, 1));
    try set.stage(3, 1);
    try set.verify();
}

test "an empty hot set cannot fetch at all" {
    var set = try SwapSet.init(std.testing.allocator, .{
        .slab_count = 2,
        .slab_bytes = 4096,
        .slab_stride_bytes = 4096,
    }, 1, 1);
    defer set.deinit(std.testing.allocator);

    try set.stage(0, 0);
    var moves: [2]Move = undefined;
    // The device half is empty, so there is no victim to displace.
    try std.testing.expectError(
        SwapError.tier_not_populated,
        set.planLayer(&.{0}, moves[0..]),
    );
}

test "four ranks come up, report their placement and stay under VRAM" {
    var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
    const queried = cuda_driver.query_devices(devices[0..]);
    const device_count = queried catch |err| switch (err) {
        error.driver_library_missing => return error.SkipZigTest,
        else => return err,
    };
    if (device_count < rank_count) {
        // A machine with fewer cards is a configuration, not a defect.
        return error.SkipZigTest;
    } else {
        assert(device_count >= rank_count);
    }

    const options = Options{ .context_tokens = 8_192 };
    var ranks = try Ranks.bringUp(options);
    defer ranks.deinit();

    var buffer: [8192]u8 = undefined;
    var stream = std.io.fixedBufferStream(buffer[0..]);
    try ranks.announcePlacement(stream.writer());
    const text = stream.getWritten();

    // Non-vacuity: every rank is named and holds its own 128 experts.
    for (&ranks.ranks, 0..) |*rank, index| {
        try std.testing.expect(rank.device.name_slice().len > 0);
        try std.testing.expect(rank.free_bytes > 0);
        var line: [96]u8 = undefined;
        const expected = try std.fmt.bufPrint(
            line[0..],
            "  rank {d} ",
            .{index},
        );
        try std.testing.expect(std.mem.indexOf(u8, text, expected) != null);
    }
    try std.testing.expect(
        std.mem.indexOf(u8, text, "128 consecutive of 512") != null,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, text, "dram tier enabled") != null,
    );
    // Every rank's plan fits the free memory that rank actually reported.
    for (&ranks.plan.ranks) |*rank| {
        try std.testing.expect(rank.device_bytes <= rank.vram_bytes);
    }
    // This machine's four cards are inside one bifurcated slot and the modified
    // driver enables peer access, so no pair may need host staging.
    try std.testing.expectEqual(@as(u32, 0), ranks.peer_fallbacks);
    for (&ranks.ranks) |*rank| {
        var peer: u32 = 0;
        while (peer < rank_count) : (peer += 1) {
            if (peer == rank.index) {
                try std.testing.expect(!rank.reachable[peer]);
            } else {
                try std.testing.expect(rank.reachable[peer]);
            }
        }
    }
    try std.testing.expect(
        std.mem.indexOf(u8, text, "refusing to start") == null,
    );
}

test "the tier moves a cold slab to the device and writes the victim back" {
    var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
    const queried = cuda_driver.query_devices(devices[0..]);
    const device_count = queried catch |err| switch (err) {
        error.driver_library_missing => return error.SkipZigTest,
        else => return err,
    };
    if (device_count < 1) {
        return error.SkipZigTest;
    } else {
        assert(device_count >= 1);
    }

    const context = try cuda_context.Context.retain(&devices[0]);
    defer context.release();
    try context.set_current();

    // Four slabs, two on the device and two in pinned host memory: the smallest
    // tier that can perform a swap.
    const geometry = SlabGeometry{
        .slab_count = 4,
        .slab_bytes = slab_bytes,
        .slab_stride_bytes = placement.expertSlabStrideBytes(),
    };
    var tier = try ExpertTier.init(std.testing.allocator, geometry, 2, 2);
    defer tier.deinit(std.testing.allocator);

    try tier.uploadHotSlab(0, 0, slabPattern(0xa0));
    try tier.uploadHotSlab(1, 1, slabPattern(0xb1));
    try tier.stageColdSlab(2, 0, slabPattern(0xc2));
    try tier.stageColdSlab(3, 1, slabPattern(0xd3));
    try tier.swap.verify();

    // Selecting slab 2 evicts the least recently used slab 0 into host slot 0.
    var moves: [4]Move = undefined;
    const planned = try tier.planLayer(&.{2}, moves[0..]);
    try std.testing.expectEqual(@as(u32, 1), planned);
    try std.testing.expectEqual(@as(u32, 0), moves[0].displaced_slab);
    try tier.prefetch(moves[0..planned]);
    try tier.waitReady();
    try std.testing.expectEqual(slab_bytes, tier.fetch_bytes);
    try std.testing.expectEqual(slab_bytes, tier.writeback_bytes);

    // The device slot now holds slab 2's bytes and host slot 0 holds slab 0's.
    const fetched = try readDeviceSlot(&tier, moves[0].device_slot);
    defer std.testing.allocator.free(fetched);
    try std.testing.expect(equalPattern(fetched, 0xc2));
    const displaced = tier.hostSlotBytes(moves[0].host_slot);
    try std.testing.expect(equalPattern(displaced, 0xa0));
    try tier.swap.verify();
    try std.testing.expectEqual(@as(u32, 2), tier.swap.deviceSlotsInUse());
    try std.testing.expectEqual(@as(u32, 2), tier.swap.hostSlotsInUse());
}

test "a two-move batch fetches both slabs, not the last one twice" {
    var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
    const queried = cuda_driver.query_devices(devices[0..]);
    const device_count = queried catch |err| switch (err) {
        error.driver_library_missing => return error.SkipZigTest,
        else => return err,
    };
    if (device_count < 1) {
        return error.SkipZigTest;
    } else {
        assert(device_count >= 1);
    }

    const context = try cuda_context.Context.retain(&devices[0]);
    defer context.release();
    try context.set_current();

    // Two hot slabs and two cold ones: the smallest tier that can plan a batch
    // of two moves, which is what a layer's routing produces at steady state.
    const geometry = SlabGeometry{
        .slab_count = 4,
        .slab_bytes = slab_bytes,
        .slab_stride_bytes = placement.expertSlabStrideBytes(),
    };
    var tier = try ExpertTier.init(std.testing.allocator, geometry, 2, 2);
    defer tier.deinit(std.testing.allocator);

    try tier.uploadHotSlab(0, 0, slabPattern(0xa0));
    try tier.uploadHotSlab(1, 1, slabPattern(0xb1));
    try tier.stageColdSlab(2, 0, slabPattern(0xc2));
    try tier.stageColdSlab(3, 1, slabPattern(0xd3));
    try tier.swap.verify();

    // Both selected slabs are cold, so each one plans its own move.
    var moves: [4]Move = undefined;
    const planned = try tier.planLayer(&.{ 2, 3 }, moves[0..]);
    try std.testing.expectEqual(@as(u32, 2), planned);
    try std.testing.expectEqual(@as(u32, 2), moves[0].fetch_slab);
    try std.testing.expectEqual(@as(u32, 3), moves[1].fetch_slab);
    try tier.prefetch(moves[0..planned]);
    try tier.waitReady();
    try std.testing.expectEqual(2 * slab_bytes, tier.fetch_bytes);
    try std.testing.expectEqual(2 * slab_bytes, tier.writeback_bytes);

    // Each move landed its own bytes. Sharing one staging slab across the two
    // device copies leaves both device slots holding the last fetched slab, so
    // the first assertion is what fails when the rotation regresses.
    const first = try readDeviceSlot(&tier, moves[0].device_slot);
    defer std.testing.allocator.free(first);
    try std.testing.expect(equalPattern(first, 0xc2));
    const second = try readDeviceSlot(&tier, moves[1].device_slot);
    defer std.testing.allocator.free(second);
    try std.testing.expect(equalPattern(second, 0xd3));
    try std.testing.expect(!std.mem.eql(u8, first, second));

    // The victims went to the host slots the fetched slabs left.
    const first_victim = tier.hostSlotBytes(moves[0].host_slot);
    try std.testing.expect(equalPattern(first_victim, 0xa0));
    const second_victim = tier.hostSlotBytes(moves[1].host_slot);
    try std.testing.expect(equalPattern(second_victim, 0xb1));
    try tier.swap.verify();
    try std.testing.expectEqual(@as(u32, 2), tier.swap.deviceSlotsInUse());
    try std.testing.expectEqual(@as(u32, 2), tier.swap.hostSlotsInUse());
}

/// The time one move's two copies take on the tier's prefetch stream with
/// nothing else queued. Three samples, the smallest: a control is a floor, not
/// a median, and the gap it is compared against is a whole move of work.
fn measureMoveCopiesNanos(tier: *const ExpertTier) !u64 {
    assert(tier.geometry.slab_bytes > 0);
    assert(tier.geometry.slab_stride_bytes >= tier.geometry.slab_bytes);

    const start = try cuda_context.Event.create();
    defer start.destroy();
    const end = try cuda_context.Event.create();
    defer end.destroy();
    const host = try cuda_context.PinnedHostBuffer.allocate(
        tier.geometry.slab_stride_bytes,
    );
    defer host.free();
    const device = try cuda_context.DeviceBuffer.allocate(
        tier.geometry.slab_stride_bytes,
    );
    defer device.free();
    const source = tier.deviceSlot(0);
    var control_ns: u64 = std.math.maxInt(u64);
    for (0..3) |_| {
        try start.record(&tier.stream);
        try tier.stream.copy_d_to_h(&host, &source, tier.geometry.slab_bytes);
        try tier.stream.copy_h_to_d(&device, &host, tier.geometry.slab_bytes);
        try end.record(&tier.stream);
        try end.synchronize();
        const sample = try cuda_context.Event.elapsed_nanoseconds(&start, &end);
        if (sample < control_ns) {
            control_ns = sample;
        } else {
            assert(sample >= control_ns);
        }
    }
    assert(control_ns > 0);
    return control_ns;
}

test "a staging slab's reuse waits for the batch's last writeback" {
    var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
    const queried = cuda_driver.query_devices(devices[0..]);
    const device_count = queried catch |err| switch (err) {
        error.driver_library_missing => return error.SkipZigTest,
        else => return err,
    };
    if (device_count < 1) {
        return error.SkipZigTest;
    } else {
        assert(device_count >= 1);
    }

    const context = try cuda_context.Context.retain(&devices[0]);
    defer context.release();
    try context.set_current();

    // Three of each, so one batch plans three moves and the rotation is used at
    // indices 0, 1, 0: the batch ends on a move whose writeback is queued
    // *after* the event staging slab 1 recorded beside its own copy.
    const geometry = SlabGeometry{
        .slab_count = 6,
        .slab_bytes = slab_bytes,
        .slab_stride_bytes = placement.expertSlabStrideBytes(),
    };
    var tier = try ExpertTier.init(std.testing.allocator, geometry, 3, 3);
    defer tier.deinit(std.testing.allocator);

    try tier.uploadHotSlab(0, 0, slabPattern(0xa0));
    try tier.uploadHotSlab(1, 1, slabPattern(0xb1));
    try tier.uploadHotSlab(2, 2, slabPattern(0xc2));
    try tier.stageColdSlab(3, 0, slabPattern(0xd3));
    try tier.stageColdSlab(4, 1, slabPattern(0xe4));
    try tier.stageColdSlab(5, 2, slabPattern(0xf5));
    try tier.swap.verify();

    var moves: [4]Move = undefined;
    const planned = try tier.planLayer(&.{ 3, 4, 5 }, moves[0..]);
    try std.testing.expectEqual(@as(u32, 3), planned);
    try tier.prefetch(moves[0..planned]);

    // `probe` is queued once the whole batch has been issued, so the stream
    // time between a staging slab's event and the probe is the work still
    // pending when the next batch refills that slab. A refill reads host
    // memory, and the last move's writeback is queued after slab 1's event.
    const probe = try cuda_context.Event.create();
    defer probe.destroy();
    try probe.record(&tier.stream);
    try probe.synchronize();
    const gap_ns = try cuda_context.Event.elapsed_nanoseconds(
        &tier.bounce_free[1],
        &probe,
    );

    // The batch still did its job, so the ordering is the only subject here.
    try tier.waitReady();
    const fetched = try readDeviceSlot(&tier, moves[0].device_slot);
    defer std.testing.allocator.free(fetched);
    try std.testing.expect(equalPattern(fetched, 0xd3));

    const control_ns = try measureMoveCopiesNanos(&tier);
    std.debug.print(
        "prefetch ordering: staging event {d} ns before the batch ended, " ++
            "one move of copies {d} ns\n",
        .{ gap_ns, control_ns },
    );
    // A quarter of one move: recording the event beside its own copy instead of
    // at the end of the batch leaves the last move's copies in the gap.
    try std.testing.expect(gap_ns * 4 <= control_ns);
}

test "four ranks copy to each other and the matrix lands in bench/comm.json" {
    var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
    const queried = cuda_driver.query_devices(devices[0..]);
    const device_count = queried catch |err| switch (err) {
        error.driver_library_missing => return error.SkipZigTest,
        else => return err,
    };
    if (device_count < rank_count) {
        return error.SkipZigTest;
    } else {
        assert(device_count >= rank_count);
    }

    var ranks = try Ranks.bringUp(.{ .context_tokens = 8_192 });
    defer ranks.deinit();

    var buffer: [64 * 1024]u8 = undefined;
    var stream = std.io.fixedBufferStream(buffer[0..]);
    const result = try writeCommsBench(&ranks, stream.writer());
    const text = stream.getWritten();

    // The number is printed before the assertions, so a refusal shows it.
    std.debug.print(
        "comms bench: slowest peer copy {d:.1} us, {d} payload sizes, " ++
            "per-token chain {d:.3} us over {d} boundaries (bound {d:.0} us, " ++
            "indicative under contention)\n",
        .{
            result.slowest_us,
            comms_payload_bytes.len,
            result.token.total_us,
            token_stages,
            token_comms_bound_us,
        },
    );

    // Non-vacuity: an ordered pair was measured and its bytes verified.
    try std.testing.expect(std.mem.indexOf(u8, text, "\"median_us\"") != null);
    try std.testing.expect(
        std.mem.indexOf(u8, text, "\"from\": 0, \"to\": 1") != null,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, text, "\"liveness_ok\": true") != null,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, text, "\"liveness_bound_ms\": 10.0") != null,
    );
    // The issue's gate: four boundaries, one activation each, under 1.5 ms.
    try std.testing.expect(
        std.mem.indexOf(u8, text, "\"per_token\": {") != null,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, text, "\"stages\": 4") != null,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, text, "\"activation_bytes\": 5120") != null,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, text, "\"total_bytes\": 20480") != null,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, text, "\"within_bound\": true") != null,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, text, "\"bound_us\": 1500.0") != null,
    );
    // Not a quiet-box figure: the contention travels with the number.
    try std.testing.expect(
        std.mem.indexOf(u8, text, "indicative: measured while other lanes") !=
            null,
    );
    try std.testing.expect(result.token.within_bound);
    try std.testing.expect(result.token.total_us > 0.0);
    try std.testing.expect(result.token.bytes_total == 20_480);
    // The total is a sum of real hops, not a constant: it can never be smaller
    // than its slowest hop's median.
    for (&result.token.hops) |*hop| {
        try std.testing.expect(result.token.total_us >= hop.median_us);
    }
    try std.testing.expect(
        std.mem.indexOf(u8, text, "\"not_measured\": [") != null,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, text, "per-token comms total") == null,
    );
    try std.testing.expect(result.slowest_us <= comms_liveness_us);
    try writeCommsReport(text, comms_report_paths[0..]);
}

test "a fixture whose counts do not divide over the ranks is refused" {
    const allocator = std.testing.allocator;
    var config = try fixtures.load_config(allocator);
    const whole = try Shard.derive(config, rank_count);
    try std.testing.expectEqual(
        config.num_experts / rank_count,
        whole.experts_per_rank,
    );
    try std.testing.expectEqual(
        @as(u32, 4) * whole.experts_per_rank,
        whole.expertSlabs(),
    );
    // Non-vacuity: one count that does not divide is an error, not a ragged
    // split that leaves a remainder on no rank.
    config.vocab_size += 1;
    try std.testing.expectError(
        error.shard_not_divisible,
        Shard.derive(config, rank_count),
    );
    config.vocab_size -= 1;
    config.num_experts = rank_count - 1;
    try std.testing.expectError(
        error.shard_not_divisible,
        Shard.derive(config, rank_count),
    );
}

test "four ranks run the tiny forward and reproduce the single-device greedy ids" {
    var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
    const queried = cuda_driver.query_devices(devices[0..]);
    const device_count = queried catch |err| switch (err) {
        error.driver_library_missing => return error.SkipZigTest,
        else => return err,
    };
    if (device_count < rank_count) {
        return error.SkipZigTest;
    } else {
        assert(device_count >= rank_count);
    }

    const allocator = std.testing.allocator;
    var golden = try tiny_check.Golden.open(allocator);
    defer golden.close();
    const tokens = tiny_check.greedy_tokens;
    try std.testing.expectEqual(@as(u32, 16), tokens);

    // One device slot and three host slabs per rank: the ownership barrier is
    // exercised by the forward itself, not only by the synthetic slab tests.
    var forward = try Forward.init(
        allocator,
        .{ .context_tokens = 8_192 },
        .{ .device_slabs = 1, .host_slabs = 3 },
    );
    defer forward.deinit();

    // The placement report is part of the forward path: every rank is named
    // with its production shard before a single kernel runs.
    const report = forward.placementReport();
    for (0..rank_count) |index| {
        var line: [96]u8 = undefined;
        const expected = try std.fmt.bufPrint(line[0..], "  rank {d} ", .{index});
        try std.testing.expect(std.mem.indexOf(u8, report, expected) != null);
    }
    try std.testing.expect(
        std.mem.indexOf(u8, report, "128 consecutive of 512") != null,
    );

    var ids: [rank_count][tokens]i64 = undefined;
    var rows: [rank_count][]i64 = undefined;
    for (&rows, 0..) |*row, index| {
        row.* = ids[index][0..];
    }
    try forward.greedy(golden.prompt, tokens, rows);

    // 1. The committed ids, on every rank.
    for (rows) |row| {
        try std.testing.expectEqualSlices(i64, golden.ids[0..tokens], row);
    }
    // 2. Printed diagnostic, not a gate: whether the four ranks agree with
    // each other. The pass/fail criterion is the golden ids above; agreement
    // only localises a fault to one rank's exchange.
    for (rows[1..], 1..) |row, index| {
        if (!std.mem.eql(i64, rows[0], row)) {
            std.debug.print(
                "DIAG rank {d} disagrees with rank 0: first difference at {d}\n",
                .{ index, firstIdDifference(rows[0], row) orelse tokens },
            );
        }
    }
    // 3. The expert tier fed the graph: the cold slabs its router selected
    // were fetched and the victims written back.
    var fetched: u64 = 0;
    var written_back: u64 = 0;
    for (&forward.tiers) |*tier| {
        fetched += tier.fetch_bytes;
        written_back += tier.writeback_bytes;
    }
    try std.testing.expect(fetched > 0);

    // 4. The tier's own half, deterministically: a slab that is cold now is
    // fetched on the prefetch stream and is resident afterwards.
    try forward.ranks.ranks[0].context.set_current();
    var cold: ?u32 = null;
    var slab: u32 = 0;
    while (slab < forward.shard.expertSlabs()) : (slab += 1) {
        if (forward.tiers[0].residentSlot(slab) == null) {
            cold = slab;
            break;
        } else {
            assert(forward.tiers[0].residentSlot(slab) != null);
        }
    }
    const wanted = cold orelse return error.no_cold_slab;
    var moves: [placement.experts_per_rank]Move = undefined;
    const planned = try forward.tiers[0].planLayer(&.{wanted}, moves[0..]);
    try std.testing.expect(planned >= 1);
    const before = forward.tiers[0].fetch_bytes;
    try forward.tiers[0].prefetch(moves[0..planned]);
    try forward.tiers[0].waitReady();
    try std.testing.expect(forward.tiers[0].residentSlot(wanted) != null);
    try std.testing.expect(forward.tiers[0].fetch_bytes > before);

    // The gate's evidence, printed before anything else runs.
    std.debug.print(
        "multi-rank greedy (4 ranks, 1 hot + 3 host slabs each):\n",
        .{},
    );
    for (ids, 0..) |row, index| {
        std.debug.print("  rank {d}:", .{index});
        for (row) |id| {
            std.debug.print(" {d}", .{id});
        }
        std.debug.print("\n", .{});
    }
    std.debug.print(
        "  golden first {d} ids equal on every rank; tier: {d} B fetched, " ++
            "{d} B written back over {d} ranks\n",
        .{tokens} ++ .{ fetched, written_back, rank_count },
    );

    // 5. Non-vacuity: the exchange is load-bearing. With the peers' slices
    // skipped, every rank keeps only its own rows, so its argmax can only name
    // an id inside its own vocabulary band and no rank reaches the golden.
    for (&rows, 0..) |*row, index| {
        row.* = ids[index][0..];
    }
    forward.sabotage = true;
    try forward.greedy(golden.prompt, tokens, rows);
    forward.sabotage = false;
    var diverged = false;
    for (rows) |row| {
        if (!std.mem.eql(i64, row, golden.ids[0..tokens])) {
            diverged = true;
        } else {
            assert(std.mem.eql(i64, row, golden.ids[0..tokens]));
        }
    }
    try std.testing.expect(diverged);
    std.debug.print("  sabotaged (peers' slices skipped):", .{});
    for (rows) |row| {
        std.debug.print(" [{d} {d} ...]", .{ row[0], row[1] });
    }
    std.debug.print("\n", .{});
}

/// A slab-sized buffer filled with one byte value, for the round-trip test.
var slab_pattern: [slab_bytes]u8 = undefined;

fn slabPattern(value: u8) []const u8 {
    assert(slab_pattern.len > 0);
    assert(value != 0);

    @memset(slab_pattern[0..], value);
    return slab_pattern[0..];
}

fn equalPattern(bytes: []const u8, value: u8) bool {
    assert(bytes.len > 0);
    assert(value != 0);

    for (bytes) |sample| {
        if (sample == value) {
            // The rest of the slab must agree with its first byte.
        } else {
            return false;
        }
    }
    return true;
}

fn readDeviceSlot(tier: *const ExpertTier, device_slot: u32) ![]u8 {
    assert(device_slot < tier.device_slot_count);
    assert(tier.geometry.slab_bytes > 0);

    const staged = try cuda_context.PinnedHostBuffer.allocate(
        tier.geometry.slab_stride_bytes,
    );
    defer staged.free();
    const source = tier.deviceSlot(device_slot);
    try tier.stream.copy_d_to_h(&staged, &source, tier.geometry.slab_bytes);
    try tier.stream.synchronize();
    const bytes = try std.testing.allocator.alloc(u8, tier.geometry.slab_bytes);
    @memcpy(bytes, staged.slice()[0..tier.geometry.slab_bytes]);
    assert(bytes.len == tier.geometry.slab_bytes);
    return bytes;
}
