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
//! The forward pass itself is *not* here yet: the graph, the fused dense
//! all-reduce and the MoE dispatch/combine arrive on this seam next, and no
//! throughput number is claimed until they do.
//!
//! Merge seam: nothing here imports the I-013 graph (`model/plan.zig`,
//! `model/exec.zig`, `model/load.zig`, `model/state.zig`, `model/mod.zig`), so
//! this file compiles and tests on its own. `src/c_api.zig` imports it from its
//! `test` block, so `zig build test` runs these tests and `tools/gate.sh` fails
//! on them. TODO(I-014), next pass: after `Ranks.bringUp(options)` and
//! `Ranks.announcePlacement`, `model/exec.zig` calls
//! `ExpertTier.planLayer(selected, moves)` on each rank's router result and
//! `ExpertTier.prefetch(moves)` on that rank's `prefetch` stream, then waits on
//! `ExpertTier.waitReady` before the grouped GEMM.

const std = @import("std");

const cuda_context = @import("../hw/api.zig").cuda_context;
const cuda_driver = @import("../hw/api.zig").cuda_driver;
const kernel = @import("../hw/kernel.zig");
const placement = @import("../mem/placement.zig");
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
        tier.stream = try cuda_context.Stream.create();
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
        tier.stream.destroy();
        for (&tier.bounce_free) |*free| {
            free.destroy();
        }
        for (&tier.bounce) |*bounce| {
            bounce.free();
        }
        tier.host.free();
        tier.device.free();
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

// --- comms bench -------------------------------------------------------------

/// Repetitions per ordered pair: the issue asks for a median of 6.
pub const comms_repetitions: usize = 6;

/// What this harness asserts. The issue's bound (per-token comms <= 1.5 ms)
/// needs the forward pass and an idle box, so it is **not measured here**: a
/// copy slower than this is a link that does not work, not one that is slow.
pub const comms_liveness_us: f64 = 10_000.0;

/// Payloads the bench copies: one token's BF16 hidden state (2 560 values) and
/// one MiB, a prefill chunk's worth of dispatch traffic.
pub const comms_payload_bytes = [_]u64{ 5_120, 1_048_576 };

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
/// One peer copy is all this measures: not a layer's fused dense all-reduce and
/// not the MoE dispatch/combine, which need the reduction kernel and the graph
/// (I-014b). The issue's per-token comms bound (<= 1.5 ms) is therefore
/// **unmeasured** and no number here may be read as one.
pub fn writeCommsBench(ranks: *Ranks, writer: anytype) !f64 {
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
    // `liveness_ok` is a link check, not the issue's gate: the per-token bound
    // is 1.5 ms and nothing in this lane measures it, which is what
    // `not_measured` says in the report the operator reads next to the flag.
    try writer.print(
        "\n  ],\n  \"liveness_bound_ms\": {d:.1},\n" ++
            "  \"liveness_ok\": {s},\n" ++
            "  \"not_measured\": [\n" ++
            "    \"per-token comms total (bound 1.5 ms): not measured, " ++
            "needs the per-layer all-reduce and the MoE dispatch/combine " ++
            "on an idle box (I-014b)\",\n" ++
            "    \"fused dense all-reduce latency: not measured, no " ++
            "reduction kernel in this lane (I-014b)\",\n" ++
            "    \"MoE dispatch/combine latency: not measured, needs the " ++
            "forward pass (I-014b)\"\n" ++
            "  ]\n}}\n",
        .{
            comms_liveness_us / 1_000.0,
            if (slowest_us <= comms_liveness_us) "true" else "false",
        },
    );
    assert(slowest_us >= 0.0);
    return slowest_us;
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

    var buffer: [32 * 1024]u8 = undefined;
    var stream = std.io.fixedBufferStream(buffer[0..]);
    const slowest_us = try writeCommsBench(&ranks, stream.writer());
    const text = stream.getWritten();

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
    // The per-token bound is named as unmeasured rather than renamed into the
    // liveness flag: no key may report it as satisfied.
    try std.testing.expect(
        std.mem.indexOf(u8, text, "\"within_bound\"") == null,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, text, "per-token comms total (bound 1.5 ms)") !=
            null,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, text, "\"not_measured\": [") != null,
    );
    // Liveness, not the issue's 1.5 ms per-token bound: see `writeCommsBench`.
    try std.testing.expect(slowest_us <= comms_liveness_us);
    std.debug.print(
        "comms bench: slowest peer copy {d:.1} us, {d} payload sizes\n",
        .{ slowest_us, comms_payload_bytes.len },
    );
    try writeCommsReport(text, comms_report_paths[0..]);
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
