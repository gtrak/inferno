//! MoE launch wrappers, the block executor and the I-010 harness.
//!
//! `src/kernels/moe.cu` holds the device code. This file owns the launch
//! geometry, the buffer contracts, the host-side setup the exact reference path
//! needs, and the tests that check every piece against `src/ref/moe.zig`.
//!
//! The routed path is fully stream-ordered: the router logits GEMM, the top-k
//! selection, the stable expert sort, the tile map, the gathered FP4 activation
//! quantiser, both grouped `OMMA.SF` GEMMs, the sigmoid-gated shared expert and
//! the weighted combine all launch with shapes derived on the host from the
//! request shape alone. The permutation output never leaves the device; the
//! grouped GEMM's grid is the deterministic upper bound below and row tiles
//! past the live count exit immediately, which keeps the block CUDA-graph
//! friendly for decode.
//!
//! The exact reference path (`Exact`) is host-driven by design: it dequantises
//! one expert's weights to bf16 at a time (an e2m1 value times a ue4m3 scale
//! needs six mantissa bits, so bf16 is lossless) and folds the per-tensor f32
//! scale into the FFMA epilogue, so the only residual against the oracle is f32
//! accumulation order. It is the numerical reference and the fallback, not a
//! throughput path.
//!
//! The block's output is f32: the router's renormalised weights, the combine
//! accumulator and the per-token result all stay in the engine's activation
//! dtype, which for this oracle-checked path is f32. Rounding the block to bf16
//! is a model-level dtype decision for I-013, not part of a kernel gate.

const std = @import("std");

const cuda_context = @import("../hw/api.zig").cuda_context;
const cuda_driver = @import("../hw/api.zig").cuda_driver;
const kernel = @import("../hw/kernel.zig");
const generated_cubins = @import("generated_cubins");
const pack_nvfp4 = @import("../weights/pack_nvfp4.zig");
const quant = @import("../weights/quant.zig");
const ref_fixtures = @import("../ref/fixtures.zig");
const ref_moe = @import("../ref/moe.zig");
const ref_ops = @import("../ref/ops.zig");
// The host weight recipe (e2m1/e4m3 rounding and the per-16 quantiser) is the
// one the I-006b harness already proved reproduces the GPU quantiser.
const host_quant = @import("gemm.zig");

const assert = std.debug.assert;

/// Cubin compiled from `src/kernels/moe.cu`.
pub const cubin_name = "moe";

/// The checkpoint's text-path MoE shape (`nvidia/Qwen3.8-Flash-Next-NVFP4`).
pub const hidden_size: u32 = 2560;
pub const expert_intermediate: u32 = 640;
pub const shared_intermediate: u32 = 640;
pub const experts_total: u32 = 512;
pub const top_k_total: u32 = 10;

/// Threads per grouped-GEMM block: one warp owns one 16-row tile.
pub const grouped_threads: u32 = 32;

/// Threads per router logits block and rows that block covers.
pub const router_threads: u32 = 128;
pub const router_rows: u32 = 64;

/// Contraction step the bf16 tensor-core GEMMs copy per pipeline stage.
pub const bf16_k_step: u32 = 32;

/// Threads per router selection block (one token per block).
pub const router_topk_threads: u32 = 128;

/// Threads per permute and tile-map block: one thread per expert.
pub const permute_threads: u32 = 512;

/// Threads per combine block (one token per block).
pub const combine_threads: u32 = 256;

/// Threads and output tile of the exact-path GEMM.
pub const dense_threads: u32 = 256;
pub const dense_rows: u32 = 64;
pub const dense_columns: u32 = 64;

/// Contraction depth of one block-scaled FP4 MMA.
pub const nvfp4_k: u32 = 64;

/// Weights sharing one ue4m3 scale.
pub const nvfp4_block: u32 = 16;

/// Rows one MMA covers, i.e. the expert row-padding granularity.
pub const mma_rows: u32 = 16;

/// Bytes pages round to; `cuMemAlloc` and the pinned staging both need it.
const page_bytes: u64 = 4096;

/// Per-tensor scale of the synthetic NVFP4 weights the harness quantises.
///
/// ModelOpt's recipe for a tensor bounded by 1: `bound / (6 * 448)`. It only
/// has to be a positive constant: the oracle dequantises the same bytes with
/// the same scale, so the value cancels out of the comparison.
pub const weight_scale2: f32 = 1.0 / (6.0 * quant.fp8_max_magnitude);

const KernelError = kernel.KernelError || cuda_driver.DriverError;

/// Finds the `moe` cubin in the generated table.
pub fn cubinBytes() ![]const u8 {
    assert(generated_cubins.cubins.len > 0);
    assert(@sizeOf(generated_cubins.Cubin) > 0);

    for (generated_cubins.cubins) |entry| {
        if (std.mem.eql(u8, std.mem.span(entry.name), cubin_name)) {
            const bytes = entry.bytes[0..entry.length];
            assert(bytes.len > 0);
            return bytes;
        }
    }
    return error.MoeCubinMissing;
}

/// Rounds a byte count up to a whole page.
fn pageRounded(bytes: u64) u64 {
    assert(page_bytes > 0);
    assert(bytes > 0);

    return (bytes + page_bytes - 1) / page_bytes * page_bytes;
}

/// Allocates a page-rounded device buffer.
fn allocate(bytes: u64) KernelError!cuda_context.DeviceBuffer {
    assert(bytes > 0);
    assert(bytes <= std.math.maxInt(u32));

    const buffer = try cuda_context.DeviceBuffer.allocate(pageRounded(bytes));
    assert(buffer.bytes >= bytes);
    return buffer;
}

/// A sub-range of a device allocation, for per-expert strides.
fn view(
    buffer: *const cuda_context.DeviceBuffer,
    offset: u64,
    bytes: u64,
) cuda_context.DeviceBuffer {
    assert(offset % 4 == 0);
    assert(bytes > 0);
    assert(offset + bytes <= buffer.bytes);

    return .{ .pointer = buffer.pointer + offset, .bytes = bytes };
}

/// Uploads `contents` into `destination` at byte `offset`.
fn uploadAt(
    stream: *const cuda_context.Stream,
    destination: *const cuda_context.DeviceBuffer,
    offset: u64,
    contents: []const u8,
) KernelError!void {
    assert(contents.len > 0);
    assert(offset % page_bytes == 0);
    assert(offset + contents.len <= destination.bytes);

    const staged = try cuda_context.PinnedHostBuffer.allocate(
        pageRounded(contents.len),
    );
    defer staged.free();
    @memcpy(staged.slice()[0..contents.len], contents);
    const target = view(destination, offset, contents.len);
    try stream.copy_h_to_d(&target, &staged, contents.len);
    // Page-locked memory must outlive the copy. `cuMemFreeHost` does not wait
    // for pending work, and the exact path queues one upload per expert, so the
    // staging buffer is drained before it is released.
    try stream.synchronize();
}

/// Downloads `length` bytes from `source` at byte `offset` into `destination`.
fn downloadAt(
    stream: *const cuda_context.Stream,
    destination: []u8,
    source: *const cuda_context.DeviceBuffer,
    offset: u64,
) KernelError!void {
    assert(destination.len > 0);
    assert(offset + destination.len <= source.bytes);

    const staged = try cuda_context.PinnedHostBuffer.allocate(
        pageRounded(destination.len),
    );
    defer staged.free();
    const origin = view(source, offset, destination.len);
    try stream.copy_d_to_h(&staged, &origin, destination.len);
    try stream.synchronize();
    @memcpy(destination, staged.slice()[0..destination.len]);
}

/// Downloads the start of `source` into `destination`.
fn download(
    stream: *const cuda_context.Stream,
    destination: []u8,
    source: *const cuda_context.DeviceBuffer,
) KernelError!void {
    assert(destination.len > 0);
    assert(destination.len <= source.bytes);

    try downloadAt(stream, destination, source, 0);
}

/// The fifteen kernels of `moe.cu`, resolved once and launched many times.
pub const Kernels = struct {
    module: kernel.Module,
    router_logits: kernel.Function,
    shared_gemm: kernel.Function,
    shared_gemm_split: kernel.Function,
    router_topk: kernel.Function,
    permute: kernel.Function,
    build_tiles: kernel.Function,
    quantize_gathered: kernel.Function,
    grouped_gemm: kernel.Function,
    silu_quantize: kernel.Function,
    combine: kernel.Function,
    dense_f32: kernel.Function,
    gather_f32: kernel.Function,
    convert_f32: kernel.Function,
    silu_mul: kernel.Function,
    silu_split: kernel.Function,

    pub fn init(image: []const u8) KernelError!Kernels {
        assert(image.len > 0);
        assert(image.len % 4 == 0);

        const module = try kernel.Module.load(image);
        return .{
            .module = module,
            .router_logits = try module.function("moe_router_logits"),
            .shared_gemm = try module.function("moe_shared_gemm_bf16"),
            .shared_gemm_split = try module.function("moe_shared_gemm_bf16_split"),
            .router_topk = try module.function("moe_router_topk"),
            .permute = try module.function("moe_permute"),
            .build_tiles = try module.function("moe_build_tiles"),
            .quantize_gathered = try module.function("moe_quantize_gathered"),
            .grouped_gemm = try module.function("moe_grouped_gemm"),
            .silu_quantize = try module.function("moe_silu_quantize"),
            .combine = try module.function("moe_combine"),
            .dense_f32 = try module.function("moe_dense_f32_gemm"),
            .gather_f32 = try module.function("moe_gather_f32"),
            .convert_f32 = try module.function("moe_convert_f32"),
            .silu_mul = try module.function("moe_silu_mul"),
            .silu_split = try module.function("moe_silu_split_bf16"),
        };
    }

    pub fn deinit(kernels: *const Kernels) void {
        assert(@intFromPtr(kernels.module.handle) != 0);
        assert(@intFromPtr(kernels.grouped_gemm.handle) != 0);

        kernels.module.unload();
    }
};

/// Expert count rounded up to the 64-column router/logits tile.
pub fn paddedExperts(n_experts: u32) u32 {
    assert(n_experts > 0);
    assert(n_experts <= experts_total);

    return (n_experts + dense_columns - 1) / dense_columns * dense_columns;
}

/// Row tiles the grouped GEMM may need for `tokens`, padded per hit expert.
///
/// The true count is `sum ceil(count_e / 16)`; this bound adds one padding tile
/// for every expert that can be hit, which is enough to size the buffers and the
/// launch grid without a device-to-host round trip.
pub fn tilesUpper(tokens: u32, top_k: u32, n_experts: u32) u32 {
    assert(tokens > 0);
    assert(top_k > 0);
    assert(n_experts > 0);

    const pairs = tokens * top_k;
    const hit_max = @min(n_experts, pairs);
    return (pairs + mma_rows - 1) / mma_rows + hit_max;
}

/// Gathered rows those tiles cover.
pub fn rowsUpper(tokens: u32, top_k: u32, n_experts: u32) u32 {
    const tiles = tilesUpper(tokens, top_k, n_experts);
    assert(tiles > 0);

    return tiles * mma_rows;
}

/// Launches the router logits GEMM.
pub fn launchRouterLogits(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    x: *const cuda_context.DeviceBuffer,
    gate: *const cuda_context.DeviceBuffer,
    logits: *const cuda_context.DeviceBuffer,
    tokens: u32,
    n_padded: u32,
    k_cols: u32,
) KernelError!void {
    assert(tokens > 0);
    assert(n_padded > 0 and n_padded % dense_columns == 0);
    assert(k_cols % 32 == 0);
    assert(logits.bytes >= @as(u64, tokens) * n_padded * 4);

    var x_pointer = x.pointer;
    var gate_pointer = gate.pointer;
    var logits_pointer = logits.pointer;
    var m_rows: i32 = @intCast(tokens);
    var columns: i32 = @intCast(n_padded);
    var contraction: i32 = @intCast(k_cols);
    var arguments = [_]?*anyopaque{
        &x_pointer,
        &gate_pointer,
        &logits_pointer,
        &m_rows,
        &columns,
        &contraction,
    };
    const grid = [3]u32{
        n_padded / dense_columns,
        (tokens + router_rows - 1) / router_rows,
        1,
    };
    try kernels.router_logits.launch(
        grid,
        .{ router_threads, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Launches a shared-expert GEMM on the bf16 tensor cores: `c = a x b^T`.
///
/// Both operands are bf16 as the checkpoint ships them and the MMA accumulates
/// in f32, so this is the same arithmetic as the FFMA path with a different
/// accumulation order. `n_cols` must be a whole 64-column tile and `k_cols` a
/// whole 32-deep copy step, which 1280x2560 and 2560x640 both are.
pub fn launchSharedGemm(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    a: *const cuda_context.DeviceBuffer,
    b: *const cuda_context.DeviceBuffer,
    c: *const cuda_context.DeviceBuffer,
    rows: u32,
    n_cols: u32,
    k_cols: u32,
) KernelError!void {
    assert(rows > 0);
    assert(n_cols > 0 and n_cols % dense_columns == 0);
    assert(k_cols % bf16_k_step == 0);
    assert(a.bytes >= @as(u64, rows) * k_cols * 2);
    assert(c.bytes >= @as(u64, rows) * n_cols * 4);

    var a_pointer = a.pointer;
    var b_pointer = b.pointer;
    var c_pointer = c.pointer;
    var row_count: i32 = @intCast(rows);
    var columns: i32 = @intCast(n_cols);
    var contraction: i32 = @intCast(k_cols);
    var arguments = [_]?*anyopaque{
        &a_pointer,
        &b_pointer,
        &c_pointer,
        &row_count,
        &columns,
        &contraction,
    };
    try kernels.shared_gemm.launch(
        .{ n_cols / dense_columns, (rows + router_rows - 1) / router_rows, 1 },
        .{ router_threads, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Launches the shared-expert down GEMM with the f32 activation given as a
/// high/low bf16 pair: `c = (a_high + a_low) x b^T`, both products into one f32
/// accumulator. Together the terms carry 2^-18 relative of the f32 value, so
/// the operand keeps sixteen mantissa bits on a kernel whose input dtype has
/// eight.
pub fn launchSharedGemmSplit(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    a_high: *const cuda_context.DeviceBuffer,
    a_low: *const cuda_context.DeviceBuffer,
    b: *const cuda_context.DeviceBuffer,
    c: *const cuda_context.DeviceBuffer,
    rows: u32,
    n_cols: u32,
    k_cols: u32,
) KernelError!void {
    assert(rows > 0);
    assert(n_cols > 0 and n_cols % dense_columns == 0);
    assert(k_cols % bf16_k_step == 0);
    assert(a_high.bytes >= @as(u64, rows) * k_cols * 2);
    assert(a_low.bytes == a_high.bytes);
    assert(c.bytes >= @as(u64, rows) * n_cols * 4);

    var high_pointer = a_high.pointer;
    var low_pointer = a_low.pointer;
    var b_pointer = b.pointer;
    var c_pointer = c.pointer;
    var row_count: i32 = @intCast(rows);
    var columns: i32 = @intCast(n_cols);
    var contraction: i32 = @intCast(k_cols);
    var arguments = [_]?*anyopaque{
        &high_pointer,
        &low_pointer,
        &b_pointer,
        &c_pointer,
        &row_count,
        &columns,
        &contraction,
    };
    try kernels.shared_gemm_split.launch(
        .{ n_cols / dense_columns, (rows + router_rows - 1) / router_rows, 1 },
        .{ router_threads, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Launches the router softmax + top-k selection.
pub fn launchRouterTopk(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    logits: *const cuda_context.DeviceBuffer,
    indices: *const cuda_context.DeviceBuffer,
    weights: *const cuda_context.DeviceBuffer,
    tokens: u32,
    logit_stride: u32,
    n_experts: u32,
    top_k: u32,
) KernelError!void {
    assert(tokens > 0);
    assert(top_k > 0 and top_k <= n_experts);
    assert(n_experts <= experts_total);
    assert(indices.bytes >= @as(u64, tokens) * top_k * 4);

    var logits_pointer = logits.pointer;
    var indices_pointer = indices.pointer;
    var weights_pointer = weights.pointer;
    var stride: i32 = @intCast(logit_stride);
    var experts: i32 = @intCast(n_experts);
    var slots: i32 = @intCast(top_k);
    var arguments = [_]?*anyopaque{
        &logits_pointer,
        &indices_pointer,
        &weights_pointer,
        &stride,
        &experts,
        &slots,
    };
    try kernels.router_topk.launch(
        .{ tokens, 1, 1 },
        .{ router_topk_threads, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Launches the stable expert sort.
pub fn launchPermute(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    indices: *const cuda_context.DeviceBuffer,
    tokens: u32,
    top_k: u32,
    n_experts: u32,
    row_index: *const cuda_context.DeviceBuffer,
    assign_pos: *const cuda_context.DeviceBuffer,
    expert_offsets: *const cuda_context.DeviceBuffer,
    expert_counts: *const cuda_context.DeviceBuffer,
    capacity_rows: u32,
) KernelError!void {
    assert(tokens > 0);
    assert(top_k > 0);
    assert(n_experts > 0 and n_experts <= experts_total);
    assert(capacity_rows >= rowsUpper(tokens, top_k, n_experts));
    assert(row_index.bytes >= @as(u64, capacity_rows) * 4);
    assert(expert_offsets.bytes >= @as(u64, n_experts + 1) * 4);

    var indices_pointer = indices.pointer;
    var row_pointer = row_index.pointer;
    var pos_pointer = assign_pos.pointer;
    var offset_pointer = expert_offsets.pointer;
    var count_pointer = expert_counts.pointer;
    var m_rows: i32 = @intCast(tokens);
    var slots: i32 = @intCast(top_k);
    var experts: i32 = @intCast(n_experts);
    var capacity: i32 = @intCast(capacity_rows);
    var arguments = [_]?*anyopaque{
        &indices_pointer,
        &m_rows,
        &slots,
        &experts,
        &row_pointer,
        &pos_pointer,
        &offset_pointer,
        &count_pointer,
        &capacity,
    };
    try kernels.permute.launch(
        .{ 1, 1, 1 },
        .{ permute_threads, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Launches the per-row-tile expert map.
pub fn launchBuildTiles(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    expert_offsets: *const cuda_context.DeviceBuffer,
    expert_counts: *const cuda_context.DeviceBuffer,
    n_experts: u32,
    tiles: u32,
    tile_expert: *const cuda_context.DeviceBuffer,
    tile_row_base: *const cuda_context.DeviceBuffer,
    tile_rows_live: *const cuda_context.DeviceBuffer,
) KernelError!void {
    assert(n_experts > 0 and n_experts <= experts_total);
    assert(tiles > 0);
    assert(tile_expert.bytes >= @as(u64, tiles) * 4);

    var offset_pointer = expert_offsets.pointer;
    var count_pointer = expert_counts.pointer;
    var expert_pointer = tile_expert.pointer;
    var base_pointer = tile_row_base.pointer;
    var live_pointer = tile_rows_live.pointer;
    var experts: i32 = @intCast(n_experts);
    var total: i32 = @intCast(tiles);
    var arguments = [_]?*anyopaque{
        &offset_pointer,
        &count_pointer,
        &experts,
        &total,
        &expert_pointer,
        &base_pointer,
        &live_pointer,
    };
    try kernels.build_tiles.launch(
        .{ 1, 1, 1 },
        .{ permute_threads, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Launches the gathered activation quantiser for the gate_up stage.
pub fn launchQuantizeGathered(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    x: *const cuda_context.DeviceBuffer,
    row_index: *const cuda_context.DeviceBuffer,
    tiles: u32,
    k_cols: u32,
    frag: *const cuda_context.DeviceBuffer,
    scales: *const cuda_context.DeviceBuffer,
) KernelError!void {
    assert(tiles > 0);
    assert(k_cols % nvfp4_k == 0);
    const rows = tiles * mma_rows;
    assert(frag.bytes >= pack_nvfp4.activationFragmentWords(rows, k_cols) * 4);
    assert(scales.bytes >= pack_nvfp4.activationScaleWords(rows, k_cols) * 4);

    var x_pointer = x.pointer;
    var row_pointer = row_index.pointer;
    var frag_pointer = frag.pointer;
    var scale_pointer = scales.pointer;
    var contraction: i32 = @intCast(k_cols);
    var arguments = [_]?*anyopaque{
        &x_pointer,
        &row_pointer,
        &contraction,
        &frag_pointer,
        &scale_pointer,
    };
    try kernels.quantize_gathered.launch(
        .{ k_cols / nvfp4_k, tiles, 1 },
        .{ grouped_threads, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// One expert weight matrix, in both the checkpoint and the fragment layout.
///
/// The device buffers hold `n_experts` slots; only the experts a run routes to
/// have to be filled, because the grouped GEMM never reads a slot the tile map
/// does not name.
pub const ExpertWeights = struct {
    gate_up_frag: cuda_context.DeviceBuffer,
    gate_up_scale: cuda_context.DeviceBuffer,
    down_frag: cuda_context.DeviceBuffer,
    down_scale: cuda_context.DeviceBuffer,

    /// Bytes one expert's gate_up fragment matrix occupies.
    pub fn gateUpFragmentBytes() u64 {
        assert(hidden_size % nvfp4_k == 0);
        assert(2 * expert_intermediate % 8 == 0);
        return pack_nvfp4.weightFragmentWords(2 * expert_intermediate, hidden_size) * 4;
    }

    /// Bytes one expert's gate_up scales occupy.
    pub fn gateUpScaleBytes() u64 {
        return pack_nvfp4.weightScaleWords(2 * expert_intermediate, hidden_size) * 4;
    }

    /// Bytes one expert's down fragment matrix occupies.
    pub fn downFragmentBytes() u64 {
        return pack_nvfp4.weightFragmentWords(hidden_size, expert_intermediate) * 4;
    }

    /// Bytes one expert's down scales occupy.
    pub fn downScaleBytes() u64 {
        return pack_nvfp4.weightScaleWords(hidden_size, expert_intermediate) * 4;
    }

    pub fn init(n_experts: u32) KernelError!ExpertWeights {
        assert(n_experts > 0 and n_experts <= experts_total);
        assert(gateUpFragmentBytes() % page_bytes == 0);

        var weights: ExpertWeights = undefined;
        weights.gate_up_frag = try allocate(gateUpFragmentBytes() * n_experts);
        weights.gate_up_scale = try allocate(gateUpScaleBytes() * n_experts);
        weights.down_frag = try allocate(downFragmentBytes() * n_experts);
        weights.down_scale = try allocate(downScaleBytes() * n_experts);
        return weights;
    }

    pub fn deinit(weights: *ExpertWeights) void {
        assert(weights.gate_up_frag.bytes > 0);
        assert(weights.down_scale.bytes > 0);

        weights.down_scale.free();
        weights.down_frag.free();
        weights.gate_up_scale.free();
        weights.gate_up_frag.free();
    }

    /// True when the four per-expert strides are whole pages, which `uploadAt`
    /// requires and the checkpoint layout guarantees.
    pub fn stridesArePageAligned() bool {
        return gateUpFragmentBytes() % page_bytes == 0 and
            gateUpScaleBytes() % page_bytes == 0 and
            downFragmentBytes() % page_bytes == 0 and
            downScaleBytes() % page_bytes == 0;
    }
};

/// Launches one grouped block-scaled FP4 GEMM over gathered rows.
pub fn launchGroupedGemm(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    a_frag: *const cuda_context.DeviceBuffer,
    a_scale: *const cuda_context.DeviceBuffer,
    b_frag: *const cuda_context.DeviceBuffer,
    b_scale: *const cuda_context.DeviceBuffer,
    tile_expert: *const cuda_context.DeviceBuffer,
    tile_row_base: *const cuda_context.DeviceBuffer,
    tile_rows_live: *const cuda_context.DeviceBuffer,
    weight_scale: f32,
    output: *const cuda_context.DeviceBuffer,
    tiles: u32,
    n_cols: u32,
    k_cols: u32,
) KernelError!void {
    assert(tiles > 0);
    assert(n_cols % dense_columns == 0);
    assert(k_cols % nvfp4_k == 0);
    assert(weight_scale > 0);
    assert(output.bytes >= @as(u64, tiles) * mma_rows * n_cols * 4);

    var a_frag_pointer = a_frag.pointer;
    var a_scale_pointer = a_scale.pointer;
    var b_frag_pointer = b_frag.pointer;
    var b_scale_pointer = b_scale.pointer;
    var tile_expert_pointer = tile_expert.pointer;
    var tile_base_pointer = tile_row_base.pointer;
    var tile_live_pointer = tile_rows_live.pointer;
    var scale = weight_scale;
    var output_pointer = output.pointer;
    var columns: i32 = @intCast(n_cols);
    var contraction: i32 = @intCast(k_cols);
    var arguments = [_]?*anyopaque{
        &a_frag_pointer,
        &a_scale_pointer,
        &b_frag_pointer,
        &b_scale_pointer,
        &tile_expert_pointer,
        &tile_base_pointer,
        &tile_live_pointer,
        &scale,
        &output_pointer,
        &columns,
        &contraction,
    };
    try kernels.grouped_gemm.launch(
        .{ n_cols / dense_columns, tiles, 1 },
        .{ grouped_threads, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Launches silu(gate) * up and its FP4 activation quantisation.
pub fn launchSiluQuantize(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    gate_up: *const cuda_context.DeviceBuffer,
    row_index: *const cuda_context.DeviceBuffer,
    tiles: u32,
    intermediate: u32,
    frag: *const cuda_context.DeviceBuffer,
    scales: *const cuda_context.DeviceBuffer,
) KernelError!void {
    assert(tiles > 0);
    assert(intermediate % nvfp4_k == 0);
    const rows = tiles * mma_rows;
    assert(frag.bytes >= pack_nvfp4.activationFragmentWords(rows, intermediate) * 4);
    assert(scales.bytes >= pack_nvfp4.activationScaleWords(rows, intermediate) * 4);

    var gate_up_pointer = gate_up.pointer;
    var row_pointer = row_index.pointer;
    var frag_pointer = frag.pointer;
    var scale_pointer = scales.pointer;
    var width: i32 = @intCast(intermediate);
    var arguments = [_]?*anyopaque{
        &gate_up_pointer,
        &row_pointer,
        &width,
        &frag_pointer,
        &scale_pointer,
    };
    try kernels.silu_quantize.launch(
        .{ intermediate / nvfp4_k, tiles, 1 },
        .{ grouped_threads, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Launches the combine: routed scatter-add plus the gated shared expert.
pub fn launchCombine(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    indices: *const cuda_context.DeviceBuffer,
    assign_pos: *const cuda_context.DeviceBuffer,
    router_weights: *const cuda_context.DeviceBuffer,
    expert_offsets: *const cuda_context.DeviceBuffer,
    down_out: *const cuda_context.DeviceBuffer,
    shared_out: *const cuda_context.DeviceBuffer,
    shared_gate_logit: *const cuda_context.DeviceBuffer,
    x: *const cuda_context.DeviceBuffer,
    output: *const cuda_context.DeviceBuffer,
    tokens: u32,
    top_k: u32,
    k_cols: u32,
) KernelError!void {
    assert(tokens > 0);
    assert(top_k > 0);
    assert(output.bytes >= @as(u64, tokens) * k_cols * 4);

    var indices_pointer = indices.pointer;
    var pos_pointer = assign_pos.pointer;
    var weight_pointer = router_weights.pointer;
    var offset_pointer = expert_offsets.pointer;
    var down_pointer = down_out.pointer;
    var shared_pointer = shared_out.pointer;
    var gate_pointer = shared_gate_logit.pointer;
    var x_pointer = x.pointer;
    var output_pointer = output.pointer;
    var slots: i32 = @intCast(top_k);
    var columns: i32 = @intCast(k_cols);
    var arguments = [_]?*anyopaque{
        &indices_pointer,
        &pos_pointer,
        &weight_pointer,
        &offset_pointer,
        &down_pointer,
        &shared_pointer,
        &gate_pointer,
        &x_pointer,
        &slots,
        &columns,
        &output_pointer,
    };
    try kernels.combine.launch(
        .{ tokens, 1, 1 },
        .{ combine_threads, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Launches the exact path's f32 FFMA GEMM over bf16-dequantised weights.
pub fn launchDenseF32(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    a: *const cuda_context.DeviceBuffer,
    b: *const cuda_context.DeviceBuffer,
    c: *const cuda_context.DeviceBuffer,
    epilogue_scale: f32,
    rows: u32,
    n_cols: u32,
    k_cols: u32,
) KernelError!void {
    assert(rows > 0);
    assert(n_cols % dense_columns == 0);
    assert(k_cols % mma_rows == 0);
    assert(c.bytes >= @as(u64, rows) * n_cols * 4);

    var a_pointer = a.pointer;
    var b_pointer = b.pointer;
    var c_pointer = c.pointer;
    var scale = epilogue_scale;
    var row_count: i32 = @intCast(rows);
    var columns: i32 = @intCast(n_cols);
    var contraction: i32 = @intCast(k_cols);
    var arguments = [_]?*anyopaque{
        &a_pointer,
        &b_pointer,
        &scale,
        &c_pointer,
        &row_count,
        &columns,
        &contraction,
    };
    const grid = [3]u32{
        n_cols / dense_columns,
        (rows + dense_rows - 1) / dense_rows,
        1,
    };
    try kernels.dense_f32.launch(
        grid,
        .{ dense_threads, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Grid for the flat elementwise kernels.
fn elementwiseGrid(count: u64) u32 {
    assert(count > 0);

    const blocks = (count + 255) / 256;
    return @intCast(@min(blocks, 65535));
}

/// Launches the bf16 -> f32 gather of the exact path.
pub fn launchGatherF32(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    x: *const cuda_context.DeviceBuffer,
    row_index: *const cuda_context.DeviceBuffer,
    rows: u32,
    k_cols: u32,
    out: *const cuda_context.DeviceBuffer,
) KernelError!void {
    assert(rows > 0);
    assert(k_cols > 0);
    assert(out.bytes >= @as(u64, rows) * k_cols * 4);

    var x_pointer = x.pointer;
    var row_pointer = row_index.pointer;
    var out_pointer = out.pointer;
    var row_count: i32 = @intCast(rows);
    var columns: i32 = @intCast(k_cols);
    var arguments = [_]?*anyopaque{
        &x_pointer,
        &row_pointer,
        &row_count,
        &columns,
        &out_pointer,
    };
    try kernels.gather_f32.launch(
        .{ elementwiseGrid(@as(u64, rows) * k_cols), 1, 1 },
        .{ 256, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Launches the bf16 -> f32 conversion used by the shared expert.
pub fn launchConvertF32(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    x: *const cuda_context.DeviceBuffer,
    count: u32,
    out: *const cuda_context.DeviceBuffer,
) KernelError!void {
    assert(count > 0);
    assert(out.bytes >= @as(u64, count) * 4);

    var x_pointer = x.pointer;
    var out_pointer = out.pointer;
    var elements: i32 = @intCast(count);
    var arguments = [_]?*anyopaque{ &x_pointer, &elements, &out_pointer };
    try kernels.convert_f32.launch(
        .{ elementwiseGrid(count), 1, 1 },
        .{ 256, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Launches silu(gate) * up in f32 for the exact path.
pub fn launchSiluMul(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    gate_up: *const cuda_context.DeviceBuffer,
    rows: u32,
    intermediate: u32,
    act: *const cuda_context.DeviceBuffer,
) KernelError!void {
    assert(rows > 0);
    assert(intermediate > 0);
    assert(act.bytes >= @as(u64, rows) * intermediate * 4);

    var gate_up_pointer = gate_up.pointer;
    var act_pointer = act.pointer;
    var row_count: i32 = @intCast(rows);
    var width: i32 = @intCast(intermediate);
    var arguments = [_]?*anyopaque{
        &gate_up_pointer,
        &row_count,
        &width,
        &act_pointer,
    };
    try kernels.silu_mul.launch(
        .{ elementwiseGrid(@as(u64, rows) * intermediate), 1, 1 },
        .{ 256, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Launches silu(gate) * up split into a high and a low bf16 term, the
/// tensor-core down stage's activation operand.
pub fn launchSiluSplit(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    gate_up: *const cuda_context.DeviceBuffer,
    rows: u32,
    intermediate: u32,
    high: *const cuda_context.DeviceBuffer,
    low: *const cuda_context.DeviceBuffer,
) KernelError!void {
    assert(rows > 0);
    assert(intermediate > 0);
    assert(gate_up.bytes >= @as(u64, rows) * 2 * intermediate * 4);
    assert(high.bytes >= @as(u64, rows) * intermediate * 2);
    assert(low.bytes == high.bytes);

    var gate_up_pointer = gate_up.pointer;
    var high_pointer = high.pointer;
    var low_pointer = low.pointer;
    var row_count: i32 = @intCast(rows);
    var width: i32 = @intCast(intermediate);
    var arguments = [_]?*anyopaque{
        &gate_up_pointer,
        &row_count,
        &width,
        &high_pointer,
        &low_pointer,
    };
    try kernels.silu_split.launch(
        .{ elementwiseGrid(@as(u64, rows) * intermediate), 1, 1 },
        .{ 256, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Every device buffer one MoE block needs for a given maximum token count.
pub const Stage = struct {
    tokens_max: u32,
    n_experts: u32,
    top_k: u32,
    n_padded: u32,
    tiles_max: u32,
    rows_capacity_max: u32,

    gate: cuda_context.DeviceBuffer,
    logits: cuda_context.DeviceBuffer,
    indices: cuda_context.DeviceBuffer,
    router_weights: cuda_context.DeviceBuffer,
    row_index: cuda_context.DeviceBuffer,
    assign_pos: cuda_context.DeviceBuffer,
    expert_offsets: cuda_context.DeviceBuffer,
    expert_counts: cuda_context.DeviceBuffer,
    tile_expert: cuda_context.DeviceBuffer,
    tile_row_base: cuda_context.DeviceBuffer,
    tile_rows_live: cuda_context.DeviceBuffer,
    a_frag: cuda_context.DeviceBuffer,
    a_scale: cuda_context.DeviceBuffer,
    gate_up: cuda_context.DeviceBuffer,
    act_frag: cuda_context.DeviceBuffer,
    act_scale: cuda_context.DeviceBuffer,
    down_out: cuda_context.DeviceBuffer,
    x: cuda_context.DeviceBuffer,
    x_f32: cuda_context.DeviceBuffer,
    out: cuda_context.DeviceBuffer,

    shared_gate_up: cuda_context.DeviceBuffer,
    shared_down: cuda_context.DeviceBuffer,
    shared_gate_logit: cuda_context.DeviceBuffer,
    shared_gate_up_out: cuda_context.DeviceBuffer,
    shared_act: cuda_context.DeviceBuffer,
    shared_act_high: cuda_context.DeviceBuffer,
    shared_act_low: cuda_context.DeviceBuffer,
    shared_out: cuda_context.DeviceBuffer,

    pub fn init(tokens_max: u32, n_experts: u32, top_k: u32) KernelError!Stage {
        assert(tokens_max > 0);
        assert(n_experts > 0 and n_experts <= experts_total);
        assert(top_k > 0 and top_k <= n_experts);
        assert(hidden_size % nvfp4_k == 0);
        assert(expert_intermediate % nvfp4_k == 0);

        const n_padded = paddedExperts(n_experts);
        const tiles = tilesUpper(tokens_max, top_k, n_experts);
        const rows = tiles * mma_rows;
        var stage: Stage = undefined;
        stage.tokens_max = tokens_max;
        stage.n_experts = n_experts;
        stage.top_k = top_k;
        stage.n_padded = n_padded;
        stage.tiles_max = tiles;
        stage.rows_capacity_max = rows;

        stage.gate = try allocate(@as(u64, n_padded) * hidden_size * 2);
        stage.logits = try allocate(@as(u64, tokens_max) * n_padded * 4);
        stage.indices = try allocate(@as(u64, tokens_max) * top_k * 4);
        stage.router_weights = try allocate(@as(u64, tokens_max) * top_k * 4);
        stage.row_index = try allocate(@as(u64, rows) * 4);
        stage.assign_pos = try allocate(@as(u64, tokens_max) * top_k * 4);
        stage.expert_offsets = try allocate(@as(u64, n_experts + 1) * 4);
        stage.expert_counts = try allocate(@as(u64, n_experts) * 4);
        stage.tile_expert = try allocate(@as(u64, tiles) * 4);
        stage.tile_row_base = try allocate(@as(u64, tiles) * 4);
        stage.tile_rows_live = try allocate(@as(u64, tiles) * 4);
        // The activation fragments are sized by the layout definition, not by
        // the packed-checkpoint arithmetic: the scale layout is broadcast, so a
        // 16x64 tile needs 32 words (128 B), twice the packed scale bytes.
        const gate_up_frag_words = pack_nvfp4.activationFragmentWords(rows, hidden_size);
        const gate_up_scale_words = pack_nvfp4.activationScaleWords(rows, hidden_size);
        const down_frag_words = pack_nvfp4.activationFragmentWords(rows, expert_intermediate);
        const down_scale_words = pack_nvfp4.activationScaleWords(rows, expert_intermediate);
        stage.a_frag = try allocate(gate_up_frag_words * 4);
        stage.a_scale = try allocate(gate_up_scale_words * 4);
        stage.gate_up = try allocate(@as(u64, rows) * 2 * expert_intermediate * 4);
        stage.act_frag = try allocate(down_frag_words * 4);
        stage.act_scale = try allocate(down_scale_words * 4);
        stage.down_out = try allocate(@as(u64, rows) * hidden_size * 4);
        stage.x = try allocate(@as(u64, tokens_max) * hidden_size * 2);
        stage.x_f32 = try allocate(@as(u64, tokens_max) * hidden_size * 4);
        stage.out = try allocate(@as(u64, tokens_max) * hidden_size * 4);

        stage.shared_gate_up = try allocate(
            @as(u64, 2 * shared_intermediate) * hidden_size * 2,
        );
        stage.shared_down = try allocate(
            @as(u64, hidden_size) * shared_intermediate * 2,
        );
        stage.shared_gate_logit = try allocate(@as(u64, hidden_size) * 4);
        stage.shared_gate_up_out = try allocate(
            @as(u64, tokens_max) * 2 * shared_intermediate * 4,
        );
        stage.shared_act = try allocate(
            @as(u64, tokens_max) * shared_intermediate * 4,
        );
        stage.shared_act_high = try allocate(
            @as(u64, tokens_max) * shared_intermediate * 2,
        );
        stage.shared_act_low = try allocate(
            @as(u64, tokens_max) * shared_intermediate * 2,
        );
        stage.shared_out = try allocate(@as(u64, tokens_max) * hidden_size * 4);
        return stage;
    }

    pub fn deinit(stage: *Stage) void {
        assert(stage.gate.bytes > 0);
        assert(stage.out.bytes > 0);

        stage.shared_out.free();
        stage.shared_act_low.free();
        stage.shared_act_high.free();
        stage.shared_act.free();
        stage.shared_gate_up_out.free();
        stage.shared_gate_logit.free();
        stage.shared_down.free();
        stage.shared_gate_up.free();
        stage.out.free();
        stage.x_f32.free();
        stage.x.free();
        stage.down_out.free();
        stage.act_scale.free();
        stage.act_frag.free();
        stage.gate_up.free();
        stage.a_scale.free();
        stage.a_frag.free();
        stage.tile_rows_live.free();
        stage.tile_row_base.free();
        stage.tile_expert.free();
        stage.expert_counts.free();
        stage.expert_offsets.free();
        stage.assign_pos.free();
        stage.row_index.free();
        stage.router_weights.free();
        stage.indices.free();
        stage.logits.free();
        stage.gate.free();
    }

    /// Tiles and gathered rows one run of `tokens` uses.
    pub fn plan(stage: *const Stage, tokens: u32) struct { tiles: u32, rows: u32 } {
        assert(tokens > 0 and tokens <= stage.tokens_max);

        const tiles = tilesUpper(tokens, stage.top_k, stage.n_experts);
        assert(tiles <= stage.tiles_max);
        const rows = tiles * mma_rows;
        assert(rows <= stage.rows_capacity_max);
        return .{ .tiles = tiles, .rows = rows };
    }

    /// Which shared-expert implementation a launch runs. The tensor-core path
    /// is the shipped one; the FFMA path is the exact reference it is compared
    /// against, and stays available so a numerics regression has a witness.
    pub const SharedPath = enum { tensor_core, exact_ffma };

    /// The sigmoid-gated shared expert, plus the bf16 -> f32 conversion of `x`
    /// that `moe_combine` reads for the gate dot.
    pub fn runShared(
        stage: *Stage,
        kernels: *const Kernels,
        stream: *const cuda_context.Stream,
        tokens: u32,
        path: SharedPath,
    ) KernelError!void {
        assert(tokens > 0 and tokens <= stage.tokens_max);

        try launchConvertF32(
            kernels,
            stream,
            &stage.x,
            tokens * hidden_size,
            &stage.x_f32,
        );
        switch (path) {
            .tensor_core => try runSharedTensorCore(stage, kernels, stream, tokens),
            .exact_ffma => try runSharedExactFfma(stage, kernels, stream, tokens),
        }
    }

    /// gate_up on the bf16 tensor cores, silu*up split into two bf16 terms,
    /// then the down projection as two accumulations into one f32 result.
    fn runSharedTensorCore(
        stage: *Stage,
        kernels: *const Kernels,
        stream: *const cuda_context.Stream,
        tokens: u32,
    ) KernelError!void {
        assert(tokens > 0 and tokens <= stage.tokens_max);

        try launchSharedGemm(
            kernels,
            stream,
            &stage.x,
            &stage.shared_gate_up,
            &stage.shared_gate_up_out,
            tokens,
            2 * shared_intermediate,
            hidden_size,
        );
        try launchSiluSplit(
            kernels,
            stream,
            &stage.shared_gate_up_out,
            tokens,
            shared_intermediate,
            &stage.shared_act_high,
            &stage.shared_act_low,
        );
        try launchSharedGemmSplit(
            kernels,
            stream,
            &stage.shared_act_high,
            &stage.shared_act_low,
            &stage.shared_down,
            &stage.shared_out,
            tokens,
            hidden_size,
            shared_intermediate,
        );
    }

    /// The exact path: f32 FFMA over bf16-dequantised weights, unchanged.
    fn runSharedExactFfma(
        stage: *Stage,
        kernels: *const Kernels,
        stream: *const cuda_context.Stream,
        tokens: u32,
    ) KernelError!void {
        assert(tokens > 0 and tokens <= stage.tokens_max);

        try launchDenseF32(
            kernels,
            stream,
            &stage.x_f32,
            &stage.shared_gate_up,
            &stage.shared_gate_up_out,
            1.0,
            tokens,
            2 * shared_intermediate,
            hidden_size,
        );
        try launchSiluMul(
            kernels,
            stream,
            &stage.shared_gate_up_out,
            tokens,
            shared_intermediate,
            &stage.shared_act,
        );
        try launchDenseF32(
            kernels,
            stream,
            &stage.shared_act,
            &stage.shared_down,
            &stage.shared_out,
            1.0,
            tokens,
            hidden_size,
            shared_intermediate,
        );
    }

    /// Router, top-k, stable sort and the tile map.
    pub fn runDispatch(
        stage: *Stage,
        kernels: *const Kernels,
        stream: *const cuda_context.Stream,
        tokens: u32,
    ) KernelError!void {
        assert(tokens > 0 and tokens <= stage.tokens_max);
        const planned = stage.plan(tokens);

        try launchRouterLogits(
            kernels,
            stream,
            &stage.x,
            &stage.gate,
            &stage.logits,
            tokens,
            stage.n_padded,
            hidden_size,
        );
        try launchRouterTopk(
            kernels,
            stream,
            &stage.logits,
            &stage.indices,
            &stage.router_weights,
            tokens,
            stage.n_padded,
            stage.n_experts,
            stage.top_k,
        );
        try launchPermute(
            kernels,
            stream,
            &stage.indices,
            tokens,
            stage.top_k,
            stage.n_experts,
            &stage.row_index,
            &stage.assign_pos,
            &stage.expert_offsets,
            &stage.expert_counts,
            planned.rows,
        );
        try launchBuildTiles(
            kernels,
            stream,
            &stage.expert_offsets,
            &stage.expert_counts,
            stage.n_experts,
            planned.tiles,
            &stage.tile_expert,
            &stage.tile_row_base,
            &stage.tile_rows_live,
        );
    }

    /// The routed FP4 path: quantise, gate_up, silu*up + quantise, down.
    pub fn runRouted(
        stage: *Stage,
        kernels: *const Kernels,
        stream: *const cuda_context.Stream,
        weights: *const ExpertWeights,
        tokens: u32,
    ) KernelError!void {
        assert(tokens > 0 and tokens <= stage.tokens_max);
        const planned = stage.plan(tokens);

        try launchQuantizeGathered(
            kernels,
            stream,
            &stage.x,
            &stage.row_index,
            planned.tiles,
            hidden_size,
            &stage.a_frag,
            &stage.a_scale,
        );
        try launchGroupedGemm(
            kernels,
            stream,
            &stage.a_frag,
            &stage.a_scale,
            &weights.gate_up_frag,
            &weights.gate_up_scale,
            &stage.tile_expert,
            &stage.tile_row_base,
            &stage.tile_rows_live,
            weight_scale2,
            &stage.gate_up,
            planned.tiles,
            2 * expert_intermediate,
            hidden_size,
        );
        try launchSiluQuantize(
            kernels,
            stream,
            &stage.gate_up,
            &stage.row_index,
            planned.tiles,
            expert_intermediate,
            &stage.act_frag,
            &stage.act_scale,
        );
        try launchGroupedGemm(
            kernels,
            stream,
            &stage.act_frag,
            &stage.act_scale,
            &weights.down_frag,
            &weights.down_scale,
            &stage.tile_expert,
            &stage.tile_row_base,
            &stage.tile_rows_live,
            weight_scale2,
            &stage.down_out,
            planned.tiles,
            hidden_size,
            expert_intermediate,
        );
    }

    /// The routed scatter-add and the gated shared expert: the block's epilogue.
    /// Reads what `runShared` and `runRouted` left in the stage, so it is the
    /// one term of the block that cannot run first.
    pub fn runCombine(
        stage: *Stage,
        kernels: *const Kernels,
        stream: *const cuda_context.Stream,
        tokens: u32,
    ) KernelError!void {
        assert(tokens > 0 and tokens <= stage.tokens_max);
        assert(stage.out.bytes >= @as(u64, tokens) * hidden_size * 4);

        try launchCombine(
            kernels,
            stream,
            &stage.indices,
            &stage.assign_pos,
            &stage.router_weights,
            &stage.expert_offsets,
            &stage.down_out,
            &stage.shared_out,
            &stage.shared_gate_logit,
            &stage.x_f32,
            &stage.out,
            tokens,
            stage.top_k,
            hidden_size,
        );
    }

    /// The whole block: dispatch, routed FP4 experts, shared expert, combine.
    pub fn run(
        stage: *Stage,
        kernels: *const Kernels,
        stream: *const cuda_context.Stream,
        weights: *const ExpertWeights,
        tokens: u32,
    ) KernelError!void {
        assert(tokens > 0 and tokens <= stage.tokens_max);

        try stage.runDispatch(kernels, stream, tokens);
        try stage.runRouted(kernels, stream, weights, tokens);
        try stage.runShared(kernels, stream, tokens, .tensor_core);
        try stage.runCombine(kernels, stream, tokens);
    }
};

// ---------------------------------------------------------------------------
// Device context.
// ---------------------------------------------------------------------------

/// One open device, its stream and the MoE kernels.
const Harness = struct {
    context: cuda_context.Context,
    stream: cuda_context.Stream,
    kernels: Kernels,
    device: cuda_driver.Device,

    fn open() !Harness {
        var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
        const count = try cuda_driver.query_devices(devices[0..]);
        assert(count > 0);
        assert(count <= cuda_driver.device_count_max);

        var harness: Harness = undefined;
        harness.device = devices[0];
        harness.context = try cuda_context.Context.retain(&devices[0]);
        try harness.context.set_current();
        harness.stream = try cuda_context.Stream.create();
        harness.kernels = try Kernels.init(try cubinBytes());
        assert(@intFromPtr(harness.stream.handle) != 0);
        return harness;
    }

    fn close(harness: *Harness) void {
        assert(@intFromPtr(harness.stream.handle) != 0);
        assert(@intFromPtr(harness.kernels.grouped_gemm.handle) != 0);

        harness.kernels.deinit();
        harness.stream.destroy();
        harness.context.release();
    }
};

// ---------------------------------------------------------------------------
// Deterministic host fixtures.
//
// Nothing here is random at test time: one LCG stream, seeded per expert, so a
// failure reproduces exactly and any expert's weights can be regenerated
// without holding the other 511.
// ---------------------------------------------------------------------------

/// Linear congruential generator: one fixed stream, no library dependency.
const Generator = struct {
    state: u32,

    fn init(seed: u32) Generator {
        assert(@sizeOf(u32) == 4);
        return .{ .state = if (seed == 0) 0x9e37_79b9 else seed };
    }

    fn next(self: *Generator) u32 {
        assert(self.state != 0);
        self.state = self.state *% 1_664_525 +% 1_013_904_223;
        return self.state;
    }

    /// A value uniform in `[-1, 1)`.
    fn signed(self: *Generator) f32 {
        assert(@sizeOf(f32) == 4);
        const raw: i32 = @bitCast(self.next());
        const value = @as(f32, @floatFromInt(raw >> 8)) / 8_388_608.0;
        assert(value >= -1.0);
        assert(value < 1.0);
        return value;
    }
};

/// Seed of one expert's weight stream; experts are independent.
fn expertSeed(expert: u32) u32 {
    assert(expert < experts_total);
    return 0x9e37_79b9 ^ (expert *% 0x85eb_ca6b);
}

/// Expert seed of the test input stream and the shared expert's stream.
const input_seed: u32 = 0x77aa_bbcc;
const shared_seed: u32 = 0x5eed_5eed;
const second_input_seed: u32 = 0x1234_abcd;

/// Fills one `[rows, k]` matrix with row-scaled uniform values.
fn fillMatrix(gen: *Generator, values: []f32, k: u32, bound: f32) void {
    assert(values.len % k == 0);
    assert(bound > 0);

    const rows = values.len / k;
    for (0..rows) |row| {
        const scale = bound * (1.0 + 0.3 * gen.signed());
        for (values[row * k ..][0..k]) |*value| {
            value.* = gen.signed() * scale;
        }
    }
}

/// Rounds a buffer to bf16 and back, so the GPU and the oracle see the same
/// inputs and the upload can be a plain cast.
fn roundToBf16(values: []f32) void {
    assert(values.len > 0);

    for (values) |*value| {
        value.* = quant.bf16_to_f32(quant.f32_to_bf16(value.*));
    }
}

/// Generates one expert's checkpoint weights: `gate_up [1280, 2560]` (gate rows
/// first, then up) and `down [2560, 640]`, both f32 before quantization.
fn generateExpert(expert: u32, gate_up: []f32, down: []f32) void {
    assert(expert < experts_total);
    assert(gate_up.len == @as(usize, 2 * expert_intermediate) * hidden_size);
    assert(down.len == @as(usize, hidden_size) * expert_intermediate);

    var gen = Generator.init(expertSeed(expert));
    fillMatrix(&gen, gate_up, hidden_size, 0.02);
    fillMatrix(&gen, down, expert_intermediate, 0.02);
}

/// The NVFP4 checkpoint layout of one expert's two weight matrices.
const ExpertPacked = struct {
    gate_up: []u8,
    scales_gate_up: []u8,
    down: []u8,
    scales_down: []u8,

    const gate_up_bytes = @as(usize, 2 * expert_intermediate) * hidden_size / 2;
    const gate_up_scale_bytes = @as(usize, 2 * expert_intermediate) * hidden_size / nvfp4_block;
    const down_bytes = @as(usize, hidden_size) * expert_intermediate / 2;
    const down_scale_bytes = @as(usize, hidden_size) * expert_intermediate / nvfp4_block;

    fn init(allocator: std.mem.Allocator) !ExpertPacked {
        assert(gate_up_bytes > 0);
        assert(down_bytes > 0);

        return .{
            .gate_up = try allocator.alloc(u8, gate_up_bytes),
            .scales_gate_up = try allocator.alloc(u8, gate_up_scale_bytes),
            .down = try allocator.alloc(u8, down_bytes),
            .scales_down = try allocator.alloc(u8, down_scale_bytes),
        };
    }

    fn deinit(quantized: *ExpertPacked, allocator: std.mem.Allocator) void {
        assert(quantized.gate_up.len > 0);
        assert(quantized.down.len > 0);

        allocator.free(quantized.scales_down);
        allocator.free(quantized.down);
        allocator.free(quantized.scales_gate_up);
        allocator.free(quantized.gate_up);
    }

    /// Quantises both matrices with the checkpoint's per-16 recipe.
    fn fill(quantized: *ExpertPacked, gate_up: []const f32, down: []const f32) void {
        assert(gate_up.len == @as(usize, 2 * expert_intermediate) * hidden_size);
        assert(down.len == @as(usize, hidden_size) * expert_intermediate);

        quantizeMatrix(gate_up, 2 * expert_intermediate, hidden_size, quantized.gate_up, quantized.scales_gate_up);
        quantizeMatrix(down, hidden_size, expert_intermediate, quantized.down, quantized.scales_down);
    }
};

/// Quantises one `[n, k]` f32 matrix into the checkpoint layout.
fn quantizeMatrix(
    values: []const f32,
    n: u32,
    k: u32,
    packed_matrix: []u8,
    scales: []u8,
) void {
    assert(values.len == @as(usize, n) * k);
    assert(packed_matrix.len == values.len / 2);
    assert(scales.len == values.len / nvfp4_block);

    for (0..n) |row| {
        host_quant.quantizeWeightRow(
            values[row * k ..][0..k],
            weight_scale2,
            packed_matrix[row * (k / 2) ..][0 .. k / 2],
            scales[row * (k / nvfp4_block) ..][0 .. k / nvfp4_block],
        );
    }
}

/// Dequantises one `[n, k]` packed matrix into bf16 (lossless for fp4 x ue4m3).
fn dequantizeRows(
    packed_matrix: []const u8,
    scales: []const u8,
    n: u32,
    k: u32,
    row: []f32,
    out: []u16,
) void {
    assert(row.len >= k);
    assert(out.len == @as(usize, n) * k);
    assert(packed_matrix.len == @as(usize, n) * k / 2);

    const blocks = k / nvfp4_block;
    for (0..n) |index| {
        // No per-tensor scale here: fp4 x block-scale is exact in bf16, and
        // the f32 per-tensor scale rides in the GEMM epilogue.
        quant.dequant_nvfp4_row(
            packed_matrix[index * (k / 2) ..][0 .. k / 2],
            @ptrCast(scales[index * blocks ..][0..blocks]),
            1.0,
            row[0..k],
        );
        for (0..k) |column| {
            out[index * k + column] = quant.f32_to_bf16(row[column]);
        }
    }
}

/// One hit expert: its original f32 weights (for the W4A4 report), the packed
/// bytes every oracle and the exact path read, and its place in the sorted list.
const HostExpert = struct {
    expert: u32,
    count: u32,
    row_base: u32,
    original_gate_up: []f32,
    original_down: []f32,
    quantized: ExpertPacked,

    fn init(
        allocator: std.mem.Allocator,
        expert: u32,
        count: u32,
        row_base: u32,
    ) !HostExpert {
        assert(expert < experts_total);
        assert(row_base % mma_rows == 0);

        var host = HostExpert{
            .expert = expert,
            .count = count,
            .row_base = row_base,
            .original_gate_up = try allocator.alloc(
                f32,
                @as(usize, 2 * expert_intermediate) * hidden_size,
            ),
            .original_down = try allocator.alloc(
                f32,
                @as(usize, hidden_size) * expert_intermediate,
            ),
            .quantized = try ExpertPacked.init(allocator),
        };
        generateExpert(expert, host.original_gate_up, host.original_down);
        host.quantized.fill(host.original_gate_up, host.original_down);
        return host;
    }

    fn deinit(host: *HostExpert, allocator: std.mem.Allocator) void {
        assert(host.expert < experts_total);
        assert(host.original_gate_up.len > 0);

        host.quantized.deinit(allocator);
        allocator.free(host.original_down);
        allocator.free(host.original_gate_up);
    }
};

/// Fragment-order scratch reused while uploading one expert at a time.
const FragmentScratch = struct {
    gate_up: []u32,
    scales_gate_up: []u32,
    down: []u32,
    scales_down: []u32,

    fn init(allocator: std.mem.Allocator) !FragmentScratch {
        assert(pack_nvfp4.k_tile == nvfp4_k);
        assert(pack_nvfp4.block_size == nvfp4_block);

        return .{
            .gate_up = try allocator.alloc(u32, @intCast(pack_nvfp4.weightFragmentWords(2 * expert_intermediate, hidden_size))),
            .scales_gate_up = try allocator.alloc(u32, @intCast(pack_nvfp4.weightScaleWords(2 * expert_intermediate, hidden_size))),
            .down = try allocator.alloc(u32, @intCast(pack_nvfp4.weightFragmentWords(hidden_size, expert_intermediate))),
            .scales_down = try allocator.alloc(u32, @intCast(pack_nvfp4.weightScaleWords(hidden_size, expert_intermediate))),
        };
    }

    fn deinit(scratch: *FragmentScratch, allocator: std.mem.Allocator) void {
        assert(scratch.gate_up.len > 0);
        assert(scratch.scales_down.len > 0);

        allocator.free(scratch.scales_down);
        allocator.free(scratch.down);
        allocator.free(scratch.scales_gate_up);
        allocator.free(scratch.gate_up);
    }
};

/// Uploads one expert's weights into its slot of every device buffer.
fn uploadExpert(
    stream: *const cuda_context.Stream,
    weights: *const ExpertWeights,
    scratch: *FragmentScratch,
    host: *const HostExpert,
) KernelError!void {
    assert(host.expert < experts_total);
    assert(weights.gate_up_frag.bytes >= ExpertWeights.gateUpFragmentBytes());

    pack_nvfp4.packWeights(
        host.quantized.gate_up,
        host.quantized.scales_gate_up,
        2 * expert_intermediate,
        hidden_size,
        scratch.gate_up,
        scratch.scales_gate_up,
    );
    pack_nvfp4.packWeights(
        host.quantized.down,
        host.quantized.scales_down,
        hidden_size,
        expert_intermediate,
        scratch.down,
        scratch.scales_down,
    );

    const expert: u64 = host.expert;
    try uploadAt(
        stream,
        &weights.gate_up_frag,
        expert * ExpertWeights.gateUpFragmentBytes(),
        std.mem.sliceAsBytes(scratch.gate_up),
    );
    try uploadAt(
        stream,
        &weights.gate_up_scale,
        expert * ExpertWeights.gateUpScaleBytes(),
        std.mem.sliceAsBytes(scratch.scales_gate_up),
    );
    try uploadAt(
        stream,
        &weights.down_frag,
        expert * ExpertWeights.downFragmentBytes(),
        std.mem.sliceAsBytes(scratch.down),
    );
    try uploadAt(
        stream,
        &weights.down_scale,
        expert * ExpertWeights.downScaleBytes(),
        std.mem.sliceAsBytes(scratch.scales_down),
    );
}

/// Generates, packs and uploads one entry per routed expert.
fn prepareExperts(
    allocator: std.mem.Allocator,
    stream: *const cuda_context.Stream,
    weights: *const ExpertWeights,
    experts: []const u32,
    counts: []const u32,
    offsets: []const u32,
) ![]HostExpert {
    assert(experts.len > 0);
    assert(counts.len == experts_total);
    assert(offsets.len == experts_total + 1);

    const hosts = try allocator.alloc(HostExpert, experts.len);
    var scratch = try FragmentScratch.init(allocator);
    defer scratch.deinit(allocator);

    var prepared: usize = 0;
    errdefer {
        for (hosts[0..prepared]) |*host| host.deinit(allocator);
        allocator.free(hosts);
    }
    for (experts, 0..) |expert, index| {
        hosts[index] = try HostExpert.init(
            allocator,
            expert,
            counts[expert],
            offsets[expert],
        );
        prepared += 1;
        try uploadExpert(stream, weights, &scratch, &hosts[index]);
    }
    try stream.synchronize();
    assert(prepared == experts.len);
    return hosts;
}

/// Releases a `prepareExperts` list.
fn releaseExperts(allocator: std.mem.Allocator, hosts: []HostExpert) void {
    assert(hosts.len > 0);

    for (hosts) |*host| host.deinit(allocator);
    allocator.free(hosts);
}

/// The shared expert's host weights (`[gate | up]`, `down`, and the gate row).
const SharedHost = struct {
    gate_up: []f32,
    down: []f32,
    gate_logit: []f32,

    fn init(allocator: std.mem.Allocator) !SharedHost {
        assert(shared_intermediate > 0);

        const shared = SharedHost{
            .gate_up = try allocator.alloc(f32, @as(usize, 2 * shared_intermediate) * hidden_size),
            .down = try allocator.alloc(f32, @as(usize, hidden_size) * shared_intermediate),
            .gate_logit = try allocator.alloc(f32, hidden_size),
        };
        var gen = Generator.init(shared_seed);
        fillMatrix(&gen, shared.gate_up, hidden_size, 0.02);
        fillMatrix(&gen, shared.down, shared_intermediate, 0.02);
        for (shared.gate_logit) |*value| {
            value.* = gen.signed() * 0.02;
        }
        roundToBf16(shared.gate_up);
        roundToBf16(shared.down);
        return shared;
    }

    fn deinit(shared: *SharedHost, allocator: std.mem.Allocator) void {
        assert(shared.gate_up.len > 0);
        assert(shared.gate_logit.len > 0);

        allocator.free(shared.gate_logit);
        allocator.free(shared.down);
        allocator.free(shared.gate_up);
    }
};

/// The router gate: `[experts_total, hidden]`, bf16-rounded.
fn fillGate(values: []f32, variant: GateVariant) void {
    assert(values.len == @as(usize, experts_total) * hidden_size);
    assert(@intFromEnum(variant) <= @intFromEnum(GateVariant.equal));

    for (0..experts_total) |expert| {
        const group: u32 = switch (variant) {
            .random => @intCast(expert),
            .groups => @intCast(expert / 32),
            .pairs => @intCast(expert / 2),
            .equal => 0,
        };
        var gen = Generator.init(expertSeed(group) ^ 0x51ed_270b);
        for (values[expert * hidden_size ..][0..hidden_size]) |*value| {
            value.* = gen.signed() * 0.02;
        }
    }
    roundToBf16(values);
}

/// The gate shapes the router test exercises.
const GateVariant = enum { random, groups, pairs, equal };

/// Entries in {-1, 0, 1}: every router dot product is then an integer below
/// 2^24, which f32 represents exactly in any accumulation order. The GPU's
/// tiled MMA and the oracle's sequential dot therefore produce bit-identical
/// logits, and the only residual in the routing weights is the softmax's
/// `exp` implementation (about one ulp).
fn exactEntry(gen: *Generator) f32 {
    assert(@sizeOf(f32) == 4);

    return @as(f32, @floatFromInt(@as(i32, @intCast(gen.next() % 3)) - 1));
}

/// Integer-valued router gate with the requested tie structure.
fn fillExactGate(values: []f32, variant: GateVariant) void {
    assert(values.len == @as(usize, experts_total) * hidden_size);

    for (0..experts_total) |expert| {
        const group: u32 = switch (variant) {
            .random => @intCast(expert),
            .groups => @intCast(expert / 32),
            .pairs => @intCast(expert / 2),
            .equal => 0,
        };
        var gen = Generator.init(expertSeed(group) ^ 0x51ed_270b);
        for (values[expert * hidden_size ..][0..hidden_size]) |*value| {
            value.* = exactEntry(&gen);
        }
    }
}

/// Integer-valued router input rows.
fn fillExactInput(x_bf16: []u16, x_f32: []f32) void {
    assert(x_bf16.len == x_f32.len);
    assert(x_bf16.len % hidden_size == 0);

    var gen = Generator.init(input_seed ^ 0x2f1e_3d4c);
    for (x_bf16, x_f32) |*half, *value| {
        const sample = exactEntry(&gen);
        half.* = quant.f32_to_bf16(sample);
        value.* = quant.bf16_to_f32(half.*);
        assert(value.* == sample);
    }
}

/// Uploads f32 values that are already bf16-representable as bf16.
fn uploadBf16(
    allocator: std.mem.Allocator,
    stream: *const cuda_context.Stream,
    destination: *const cuda_context.DeviceBuffer,
    values: []const f32,
) !void {
    assert(values.len > 0);
    assert(@as(u64, values.len) * 2 <= destination.bytes);

    const half = try allocator.alloc(u16, values.len);
    defer allocator.free(half);
    for (values, 0..) |value, index| {
        half[index] = quant.f32_to_bf16(value);
        assert(quant.bf16_to_f32(half[index]) == value);
    }
    try uploadAt(stream, destination, 0, std.mem.sliceAsBytes(half));
}

/// Uploads f32 values verbatim.
fn uploadF32(
    stream: *const cuda_context.Stream,
    destination: *const cuda_context.DeviceBuffer,
    values: []const f32,
) !void {
    assert(values.len > 0);
    assert(@as(u64, values.len) * 4 <= destination.bytes);

    try uploadAt(stream, destination, 0, std.mem.sliceAsBytes(values));
}

/// Downloads a device buffer into a host slice.
fn downloadSlice(
    stream: *const cuda_context.Stream,
    destination: anytype,
    source: *const cuda_context.DeviceBuffer,
) !void {
    assert(destination.len > 0);

    try downloadAt(stream, std.mem.sliceAsBytes(destination), source, 0);
}

/// One comparison of a kernel output against the oracle.
const Comparison = struct {
    max_abs_err: f64,
    max_reference: f64,

    fn relative(comparison: *const Comparison) f64 {
        assert(comparison.max_reference >= 0);

        if (comparison.max_reference > 0) {
            return comparison.max_abs_err / comparison.max_reference;
        } else {
            return std.math.inf(f64);
        }
    }
};

/// Compares two f32 buffers elementwise.
fn compareOutput(actual: []const f32, expected: []const f32) Comparison {
    assert(actual.len == expected.len);
    assert(actual.len > 0);

    var max_abs_err: f64 = 0;
    var max_reference: f64 = 0;
    for (actual, expected) |value, reference| {
        max_abs_err = @max(max_abs_err, @abs(@as(f64, value) - @as(f64, reference)));
        max_reference = @max(max_reference, @abs(@as(f64, reference)));
    }
    return .{ .max_abs_err = max_abs_err, .max_reference = max_reference };
}

/// Whether an oracle run gets the original f32 weights or the dequantised ones.
const Basis = enum { as_multiplied, original };

/// The f32 weights the oracle multiplies, one resident buffer per hit expert.
const OracleExperts = struct {
    gate_up: [][]f32,
    down: [][]f32,

    fn init(
        allocator: std.mem.Allocator,
        hosts: []const HostExpert,
        basis: Basis,
    ) !OracleExperts {
        assert(hosts.len > 0);
        assert(@intFromEnum(basis) <= @intFromEnum(Basis.original));

        var experts = OracleExperts{
            .gate_up = try allocator.alloc([]f32, hosts.len),
            .down = try allocator.alloc([]f32, hosts.len),
        };
        var prepared: usize = 0;
        errdefer {
            for (experts.gate_up[0..prepared]) |buffer| allocator.free(buffer);
            for (experts.down[0..prepared]) |buffer| allocator.free(buffer);
            allocator.free(experts.down);
            allocator.free(experts.gate_up);
        }
        for (hosts, 0..) |host, index| {
            experts.gate_up[index] = try allocator.alloc(
                f32,
                @as(usize, 2 * expert_intermediate) * hidden_size,
            );
            experts.down[index] = try allocator.alloc(
                f32,
                @as(usize, hidden_size) * expert_intermediate,
            );
            prepared += 1;
            switch (basis) {
                .original => {
                    @memcpy(experts.gate_up[index], host.original_gate_up);
                    @memcpy(experts.down[index], host.original_down);
                },
                .as_multiplied => {
                    dequantizeMatrix(
                        host.quantized.gate_up,
                        host.quantized.scales_gate_up,
                        2 * expert_intermediate,
                        hidden_size,
                        experts.gate_up[index],
                    );
                    dequantizeMatrix(
                        host.quantized.down,
                        host.quantized.scales_down,
                        hidden_size,
                        expert_intermediate,
                        experts.down[index],
                    );
                },
            }
        }
        assert(prepared == hosts.len);
        return experts;
    }

    fn deinit(experts: *OracleExperts, allocator: std.mem.Allocator) void {
        assert(experts.gate_up.len > 0);
        assert(experts.down.len == experts.gate_up.len);

        for (experts.down) |buffer| allocator.free(buffer);
        for (experts.gate_up) |buffer| allocator.free(buffer);
        allocator.free(experts.down);
        allocator.free(experts.gate_up);
    }
};

/// Dequantises one packed `[n, k]` matrix to f32.
fn dequantizeMatrix(
    packed_matrix: []const u8,
    scales: []const u8,
    n: u32,
    k: u32,
    out: []f32,
) void {
    assert(out.len == @as(usize, n) * k);
    assert(packed_matrix.len == out.len / 2);

    const blocks = k / nvfp4_block;
    for (0..n) |row| {
        quant.dequant_nvfp4_row(
            packed_matrix[row * (k / 2) ..][0 .. k / 2],
            @ptrCast(scales[row * blocks ..][0..blocks]),
            weight_scale2,
            out[row * k ..][0..k],
        );
    }
}

/// Builds the oracle's weight view: every unhit expert points at an empty slice,
/// which `forward_experts` never dereferences.
fn oracleWeights(
    experts: *const OracleExperts,
    hosts: []const HostExpert,
    router: []const f32,
    shared: *const SharedHost,
) ref_moe.Weights {
    assert(hosts.len == experts.gate_up.len);
    assert(router.len == @as(usize, experts_total) * hidden_size);

    var weights = ref_moe.Weights{
        .router = router,
        .gate_proj = undefined,
        .up_proj = undefined,
        .down_proj = undefined,
        .shared_gate = shared.gate_up[0 .. @as(usize, shared_intermediate) * hidden_size],
        .shared_up = shared.gate_up[@as(usize, shared_intermediate) * hidden_size ..],
        .shared_down = shared.down,
        .shared_gate_logit = shared.gate_logit,
    };
    for (&weights.gate_proj) |*slice| {
        slice.* = &[_]f32{};
    }
    for (&weights.up_proj) |*slice| {
        slice.* = &[_]f32{};
    }
    for (&weights.down_proj) |*slice| {
        slice.* = &[_]f32{};
    }
    for (hosts, 0..) |host, index| {
        const expert: usize = host.expert;
        const gate_up = experts.gate_up[index];
        weights.gate_proj[expert] = gate_up[0 .. @as(usize, expert_intermediate) * hidden_size];
        weights.up_proj[expert] = gate_up[@as(usize, expert_intermediate) * hidden_size ..];
        weights.down_proj[expert] = experts.down[index];
    }
    assert(weights.shared_up.len == @as(usize, shared_intermediate) * hidden_size);
    return weights;
}

/// The as-multiplied routed reference: for every (token, slot) assignment it
/// multiplies exactly the operands the GPU multiplies — the quantised activation
/// row, the dequantised gate_up weights, silu(gate)*up quantised per 16, and the
/// dequantised down weights — and accumulates the routing weight in f32. It is
/// the same arithmetic `ref_moe.forward_expert` performs, with the reference's
/// own primitives, plus the intermediate quantisation the FP4 path adds; the
/// monolithic reference cannot express that step.
fn buildRoutedReference(
    allocator: std.mem.Allocator,
    experts: *const OracleExperts,
    hosts: []const HostExpert,
    indices: []const u32,
    router_weights: []const f32,
    x_quantized: []const f32,
    tokens: u32,
    out: []f32,
) !void {
    assert(out.len == @as(usize, tokens) * hidden_size);
    assert(indices.len == @as(usize, tokens) * top_k_total);
    assert(router_weights.len == indices.len);

    var lookup: [experts_total]?usize = [_]?usize{null} ** experts_total;
    for (hosts, 0..) |host, index| {
        lookup[host.expert] = index;
    }
    const gate = try allocator.alloc(f32, expert_intermediate);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, expert_intermediate);
    defer allocator.free(up);
    const act = try allocator.alloc(f32, expert_intermediate);
    defer allocator.free(act);
    const down = try allocator.alloc(f32, hidden_size);
    defer allocator.free(down);
    var packed_row: [expert_intermediate / 2]u8 = undefined;
    var scale_row: [expert_intermediate / nvfp4_block]u8 = undefined;

    @memset(out, 0);
    for (0..tokens) |token| {
        const input = x_quantized[token * hidden_size ..][0..hidden_size];
        for (0..top_k_total) |slot| {
            const pair = token * top_k_total + slot;
            const expert = indices[pair];
            assert(expert < experts_total);
            const index = lookup[expert] orelse return error.MissingExpert;
            const gate_up = experts.gate_up[index];
            ref_ops.matmul(input, gate_up[0 .. @as(usize, expert_intermediate) * hidden_size], gate, .{
                .rows = 1,
                .in_features = hidden_size,
                .out_features = expert_intermediate,
            });
            ref_ops.matmul(input, gate_up[@as(usize, expert_intermediate) * hidden_size ..], up, .{
                .rows = 1,
                .in_features = hidden_size,
                .out_features = expert_intermediate,
            });
            for (act, gate, up) |*value, gate_value, up_value| {
                value.* = ref_ops.silu(gate_value) * up_value;
            }
            quantizeActivationRow(act, packed_row[0..], scale_row[0..]);
            ref_ops.matmul(act, experts.down[index], down, .{
                .rows = 1,
                .in_features = expert_intermediate,
                .out_features = hidden_size,
            });
            const weight = router_weights[pair];
            for (0..hidden_size) |column| {
                out[token * hidden_size + column] += down[column] * weight;
            }
        }
    }
}

/// The exact BF16-dequantised reference path's device scratch.
const Exact = struct {
    gathered: cuda_context.DeviceBuffer,
    act: cuda_context.DeviceBuffer,
    weights: cuda_context.DeviceBuffer,
    host_gate_up: []u16,
    host_down: []u16,
    row: []f32,

    fn init(allocator: std.mem.Allocator, rows: u32) !Exact {
        assert(rows > 0);
        // The scratch holds one dequantised bf16 matrix: gate_up is the larger.
        const weights_bytes = @as(u64, 2 * expert_intermediate) * hidden_size * 2;

        const exact = Exact{
            .gathered = try allocate(@as(u64, rows) * hidden_size * 4),
            .act = try allocate(@as(u64, rows) * expert_intermediate * 4),
            .weights = try allocate(weights_bytes),
            .host_gate_up = try allocator.alloc(u16, @as(usize, 2 * expert_intermediate) * hidden_size),
            .host_down = try allocator.alloc(u16, @as(usize, hidden_size) * expert_intermediate),
            .row = try allocator.alloc(f32, hidden_size),
        };
        assert(exact.host_gate_up.len > 0);
        return exact;
    }

    fn deinit(exact: *Exact, allocator: std.mem.Allocator) void {
        assert(exact.gathered.bytes > 0);
        assert(exact.row.len > 0);

        allocator.free(exact.row);
        allocator.free(exact.host_down);
        allocator.free(exact.host_gate_up);
        exact.weights.free();
        exact.act.free();
        exact.gathered.free();
    }
};

/// Runs the exact BF16-dequantised reference: gathered f32 activations, an f32
/// FFMA gate_up, silu*up in f32, an f32 FFMA down, then the shared expert and
/// the same combine the FP4 path uses. `shared` picks which shared-expert
/// implementation runs in the same block, so the two can be compared on
/// identical routed output.
fn runExact(
    stage: *Stage,
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    exact: *Exact,
    hosts: []const HostExpert,
    tokens: u32,
    shared: Stage.SharedPath,
) KernelError!void {
    assert(tokens > 0);
    assert(exact.host_gate_up.len > 0);

    const planned = stage.plan(tokens);
    try stage.runDispatch(kernels, stream, tokens);
    try launchGatherF32(
        kernels,
        stream,
        &stage.x,
        &stage.row_index,
        planned.rows,
        hidden_size,
        &exact.gathered,
    );

    for (hosts) |host| {
        if (host.count == 0) {
            continue;
        }
        const count = host.count;
        const offset = host.row_base;
        assert(count <= mma_rows * planned.tiles);
        assert(offset + count <= planned.rows);

        dequantizeRows(
            host.quantized.gate_up,
            host.quantized.scales_gate_up,
            2 * expert_intermediate,
            hidden_size,
            exact.row,
            exact.host_gate_up,
        );
        try uploadAt(stream, &exact.weights, 0, std.mem.sliceAsBytes(exact.host_gate_up));
        const gate_up_in = view(&exact.gathered, @as(u64, offset) * hidden_size * 4, @as(u64, count) * hidden_size * 4);
        const gate_up_out = view(&stage.gate_up, @as(u64, offset) * 2 * expert_intermediate * 4, @as(u64, count) * 2 * expert_intermediate * 4);
        try launchDenseF32(
            kernels,
            stream,
            &gate_up_in,
            &exact.weights,
            &gate_up_out,
            weight_scale2,
            count,
            2 * expert_intermediate,
            hidden_size,
        );
        const act = view(&exact.act, @as(u64, offset) * expert_intermediate * 4, @as(u64, count) * expert_intermediate * 4);
        try launchSiluMul(
            kernels,
            stream,
            &gate_up_out,
            count,
            expert_intermediate,
            &act,
        );

        dequantizeRows(
            host.quantized.down,
            host.quantized.scales_down,
            hidden_size,
            expert_intermediate,
            exact.row,
            exact.host_down,
        );
        try uploadAt(stream, &exact.weights, 0, std.mem.sliceAsBytes(exact.host_down));
        const down_out = view(&stage.down_out, @as(u64, offset) * hidden_size * 4, @as(u64, count) * hidden_size * 4);
        try launchDenseF32(
            kernels,
            stream,
            &act,
            &exact.weights,
            &down_out,
            weight_scale2,
            count,
            hidden_size,
            expert_intermediate,
        );
    }

    try stage.runShared(kernels, stream, tokens, shared);
    try launchCombine(
        kernels,
        stream,
        &stage.indices,
        &stage.assign_pos,
        &stage.router_weights,
        &stage.expert_offsets,
        &stage.down_out,
        &stage.shared_out,
        &stage.shared_gate_logit,
        &stage.x_f32,
        &stage.out,
        tokens,
        stage.top_k,
        hidden_size,
    );
}

/// Fills every expert slot by copying the first slot, so the timed kernel moves
/// the bytes of a 512-expert layer without generating 512 distinct matrices.
fn broadcastExperts(
    stream: *const cuda_context.Stream,
    weights: *const ExpertWeights,
    n_experts: u32,
    source_expert: u32,
) KernelError!void {
    assert(n_experts > 1);
    assert(n_experts <= experts_total);
    assert(source_expert < n_experts);

    const copies = [_]struct { buffer: *const cuda_context.DeviceBuffer, stride: u64 }{
        .{ .buffer = &weights.gate_up_frag, .stride = ExpertWeights.gateUpFragmentBytes() },
        .{ .buffer = &weights.gate_up_scale, .stride = ExpertWeights.gateUpScaleBytes() },
        .{ .buffer = &weights.down_frag, .stride = ExpertWeights.downFragmentBytes() },
        .{ .buffer = &weights.down_scale, .stride = ExpertWeights.downScaleBytes() },
    };
    for (copies) |copy| {
        // Slot 0 is the doubling template.
        if (source_expert != 0) {
            const origin = view(copy.buffer, copy.stride * source_expert, copy.stride);
            const slot_zero = view(copy.buffer, 0, copy.stride);
            try stream.copy_d_to_d(&slot_zero, &origin, copy.stride);
        }
        var filled: u32 = 1;
        while (filled < n_experts) {
            const chunk = @min(filled, n_experts - filled);
            const bytes = copy.stride * chunk;
            const source = view(copy.buffer, 0, bytes);
            const destination = view(copy.buffer, copy.stride * filled, bytes);
            try stream.copy_d_to_d(&destination, &source, bytes);
            filled += chunk;
        }
    }
    try stream.synchronize();
}

// ---------------------------------------------------------------------------
// Host inputs and the reference-config subset the oracle needs.
// ---------------------------------------------------------------------------

/// The `RefConfig` fields `ref/moe.zig` reads; everything else stays default.
fn oracleConfig(tokens: u32) ref_fixtures.RefConfig {
    assert(tokens > 0);
    assert(normTopkProb());

    return .{
        .hidden_size = hidden_size,
        .num_experts = experts_total,
        .num_experts_per_tok = top_k_total,
        .moe_intermediate_size = expert_intermediate,
        .shared_expert_intermediate_size = shared_intermediate,
        .norm_topk_prob = true,
        .output_gate_type = .sigmoid,
        .max_position_embeddings = tokens,
    };
}

/// The checkpoint's router normalises its top-k probabilities.
fn normTopkProb() bool {
    assert(top_k_total < experts_total);
    return true;
}

/// Inputs of one block evaluation, kept in the dtypes the GPU and oracle share.
const Fixture = struct {
    tokens: u32,
    gate: []f32,
    x_bf16: []u16,
    x_f32: []f32,
    x_quantized: []f32,
    shared: SharedHost,

    fn init(
        allocator: std.mem.Allocator,
        tokens: u32,
        seed: u32,
    ) !Fixture {
        assert(tokens > 0);
        assert(seed != 0);

        const fixture: Fixture = .{
            .tokens = tokens,
            .gate = try allocator.alloc(f32, @as(usize, experts_total) * hidden_size),
            .x_bf16 = try allocator.alloc(u16, @as(usize, tokens) * hidden_size),
            .x_f32 = try allocator.alloc(f32, @as(usize, tokens) * hidden_size),
            .x_quantized = try allocator.alloc(f32, @as(usize, tokens) * hidden_size),
            .shared = try SharedHost.init(allocator),
        };
        fillGate(fixture.gate, .random);
        fillInput(fixture.x_bf16, fixture.x_f32, seed);
        quantizeInput(fixture.x_f32, fixture.x_quantized);
        return fixture;
    }

    fn deinit(fixture: *Fixture, allocator: std.mem.Allocator) void {
        assert(fixture.gate.len > 0);
        assert(fixture.x_quantized.len > 0);

        fixture.shared.deinit(allocator);
        allocator.free(fixture.x_quantized);
        allocator.free(fixture.x_f32);
        allocator.free(fixture.x_bf16);
        allocator.free(fixture.gate);
    }

    /// Uploads the router gate and the shared expert into a stage.
    fn uploadWeights(
        fixture: *const Fixture,
        allocator: std.mem.Allocator,
        stream: *const cuda_context.Stream,
        stage: *const Stage,
    ) !void {
        assert(fixture.shared.gate_up.len > 0);
        assert(stage.tokens_max >= fixture.tokens);

        try uploadBf16(allocator, stream, &stage.gate, fixture.gate);
        try uploadBf16(allocator, stream, &stage.shared_gate_up, fixture.shared.gate_up);
        try uploadBf16(allocator, stream, &stage.shared_down, fixture.shared.down);
        try uploadF32(stream, &stage.shared_gate_logit, fixture.shared.gate_logit);
    }

    /// Uploads one input stream (`seed` selects the realisation).
    fn uploadInput(
        fixture: *const Fixture,
        allocator: std.mem.Allocator,
        stream: *const cuda_context.Stream,
        stage: *const Stage,
        seed: u32,
    ) !void {
        assert(fixture.x_bf16.len > 0);
        assert(stage.x.bytes >= @as(u64, fixture.tokens) * hidden_size * 2);

        if (seed == input_seed) {
            try uploadAt(stream, &stage.x, 0, std.mem.sliceAsBytes(fixture.x_bf16));
        } else {
            const other = try allocator.alloc(
                u16,
                @as(usize, fixture.tokens) * hidden_size,
            );
            defer allocator.free(other);
            const other_f32 = try allocator.alloc(
                f32,
                @as(usize, fixture.tokens) * hidden_size,
            );
            defer allocator.free(other_f32);
            fillInput(other, other_f32, seed);
            try uploadAt(stream, &stage.x, 0, std.mem.sliceAsBytes(other));
        }
    }
};

/// Fills `[tokens, hidden]` bf16 input rows and their f32 mirror.
fn fillInput(x_bf16: []u16, x_f32: []f32, seed: u32) void {
    assert(x_bf16.len == x_f32.len);
    assert(x_bf16.len % hidden_size == 0);
    assert(seed != 0);

    var gen = Generator.init(seed);
    const tokens = x_bf16.len / hidden_size;
    for (0..tokens) |token| {
        const scale = 1.0 + 0.5 * gen.signed();
        for (0..hidden_size) |column| {
            const value = gen.signed() * scale;
            const half = quant.f32_to_bf16(value);
            x_bf16[token * hidden_size + column] = half;
            x_f32[token * hidden_size + column] = quant.bf16_to_f32(half);
        }
    }
}

/// Quantises each input row in place with the kernel's per-16 FP4 recipe, so
/// the oracle multiplies exactly the values the MMA multiplies.
fn quantizeInput(x_f32: []const f32, x_quantized: []f32) void {
    assert(x_f32.len == x_quantized.len);
    assert(x_f32.len % hidden_size == 0);

    var packed_row: [hidden_size / 2]u8 = undefined;
    var scales: [hidden_size / nvfp4_block]u8 = undefined;
    const tokens = x_f32.len / hidden_size;
    for (0..tokens) |token| {
        const row = x_quantized[token * hidden_size ..][0..hidden_size];
        @memcpy(row, x_f32[token * hidden_size ..][0..hidden_size]);
        quantizeActivationRow(row, packed_row[0..], scales[0..]);
    }
}

/// Quantises then dequantises one activation row with the kernel's per-16
/// `scale_vec::4X` recipe (`amax / 6` as ue4m3, no per-tensor scale). Mirrors
/// the recipe the I-006b harness established.
fn quantizeActivationRow(row: []f32, packed_row: []u8, scales: []u8) void {
    assert(row.len % nvfp4_block == 0);
    assert(packed_row.len == row.len / 2);
    assert(scales.len == row.len / nvfp4_block);

    host_quant.quantizeWeightRow(row, 1.0, packed_row, scales);
    quant.dequant_nvfp4_row(packed_row, @ptrCast(scales), 1.0, row);
}

/// The oracle's router-only weight view, for tests that exercise routing alone.
fn routeWeights(router: []const f32) ref_moe.Weights {
    assert(router.len > 0);

    var weights = ref_moe.Weights{
        .router = router,
        .gate_proj = undefined,
        .up_proj = undefined,
        .down_proj = undefined,
        .shared_gate = &[_]f32{},
        .shared_up = &[_]f32{},
        .shared_down = &[_]f32{},
        .shared_gate_logit = &[_]f32{},
    };
    for (&weights.gate_proj) |*slice| {
        slice.* = &[_]f32{};
    }
    for (&weights.up_proj) |*slice| {
        slice.* = &[_]f32{};
    }
    for (&weights.down_proj) |*slice| {
        slice.* = &[_]f32{};
    }
    return weights;
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

// The synthetic tie cases below are the router's tie-breaking gate. They
// compare against `ref/moe.zig`'s `topk_descending`, which selects the lowest
// index of an equal group. Inferno fixes that order deliberately; torch does
// not promise one, and says so itself - the `torch.topk` docstring reads "the
// indices of tied elements are not guaranteed to be stable" - and on CPU
// torch 2.15.0+cpu the top 10 of 512 equal probabilities is experts 344 337
// 338 339 340 341 342 343 350 349, not the ten lowest. The repro is in the
// `moe_router_topk` header in moe.cu. The oracle is therefore the only tie
// contract this kernel can be gated against.
test "the router matches the oracle on random and tied logits" {
    const allocator = std.testing.allocator;
    var harness = try Harness.open();
    defer harness.close();

    const tokens: u32 = 16;
    var stage = try Stage.init(tokens, experts_total, top_k_total);
    defer stage.deinit();
    var fixture = try Fixture.init(allocator, tokens, input_seed);
    defer fixture.deinit(allocator);
    fillExactInput(fixture.x_bf16, fixture.x_f32);
    try fixture.uploadWeights(allocator, &harness.stream, &stage);
    try fixture.uploadInput(allocator, &harness.stream, &stage, input_seed);

    const config = oracleConfig(tokens);
    var scratch = try ref_moe.Scratch.init(allocator, &config);
    defer scratch.deinit(allocator);

    const indices = try allocator.alloc(u32, @as(usize, tokens) * top_k_total);
    defer allocator.free(indices);
    const weights = try allocator.alloc(f32, @as(usize, tokens) * top_k_total);
    defer allocator.free(weights);

    const variants = [_]GateVariant{ .random, .groups, .pairs, .equal };
    for (variants) |variant| {
        fillGate(fixture.gate, variant);
        try uploadBf16(allocator, &harness.stream, &stage.gate, fixture.gate);
        try launchRouterLogits(
            &harness.kernels,
            &harness.stream,
            &stage.x,
            &stage.gate,
            &stage.logits,
            tokens,
            stage.n_padded,
            hidden_size,
        );
        try launchRouterTopk(
            &harness.kernels,
            &harness.stream,
            &stage.logits,
            &stage.indices,
            &stage.router_weights,
            tokens,
            stage.n_padded,
            stage.n_experts,
            stage.top_k,
        );
        try downloadSlice(&harness.stream, indices, &stage.indices);
        try downloadSlice(&harness.stream, weights, &stage.router_weights);

        var router = routeWeights(fixture.gate);
        ref_moe.forward_route(fixture.x_f32, tokens, &router, &scratch, &config);

        var worst_relative: f64 = 0;
        for (0..@as(usize, tokens) * top_k_total) |index| {
            try std.testing.expectEqual(scratch.topk_indices[index], indices[index]);
            const expected = scratch.topk_values[index];
            const actual = weights[index];
            worst_relative = @max(
                worst_relative,
                @abs(@as(f64, actual) - @as(f64, expected)) / @abs(@as(f64, expected)),
            );
        }
        if (worst_relative > 1e-6) {
            std.debug.print(
                "  FAIL router {s}: relative weight error {e:.3}\n",
                .{ @tagName(variant), worst_relative },
            );
            return error.RouterWeightMismatch;
        }
        std.debug.print(
            "router {s:<7} {d} tokens: indices exact, max relative weight error {e:.3}\n",
            .{ @tagName(variant), tokens, worst_relative },
        );

        // Tie structure, asserted directly as well as through the oracle: a
        // group of equal experts must select the lowest indices in ascending
        // order, never a later member of the group.
        switch (variant) {
            .equal => for (0..top_k_total) |slot| {
                try std.testing.expectEqual(@as(u32, @intCast(slot)), indices[slot]);
            },
            .groups => {
                const group = indices[0] / 32;
                for (0..top_k_total) |slot| {
                    try std.testing.expectEqual(
                        group * 32 + @as(u32, @intCast(slot)),
                        indices[slot],
                    );
                }
                std.debug.print(
                    "  group tie: top-10 = experts {d}..{d} (lowest indices of the winning group)\n",
                    .{ indices[0], indices[top_k_total - 1] },
                );
            },
            else => {},
        }
    }

    // Realistic magnitudes: the two engines sum each 2560-term logit in
    // different f32 orders, so their logits agree to ~2e-6 relative. The
    // selection must still be exact; the weight residual is reported because
    // the 1e-6 weight contract is pinned above on identical logits.
    fillGate(fixture.gate, .random);
    fillInput(fixture.x_bf16, fixture.x_f32, input_seed);
    try uploadBf16(allocator, &harness.stream, &stage.gate, fixture.gate);
    try fixture.uploadInput(allocator, &harness.stream, &stage, input_seed);
    try launchRouterLogits(
        &harness.kernels,
        &harness.stream,
        &stage.x,
        &stage.gate,
        &stage.logits,
        tokens,
        stage.n_padded,
        hidden_size,
    );
    try launchRouterTopk(
        &harness.kernels,
        &harness.stream,
        &stage.logits,
        &stage.indices,
        &stage.router_weights,
        tokens,
        stage.n_padded,
        stage.n_experts,
        stage.top_k,
    );
    try downloadSlice(&harness.stream, indices, &stage.indices);
    try downloadSlice(&harness.stream, weights, &stage.router_weights);
    const logits = try allocator.alloc(f32, @as(usize, tokens) * stage.n_padded);
    defer allocator.free(logits);
    try downloadSlice(&harness.stream, logits, &stage.logits);
    var router = routeWeights(fixture.gate);
    ref_moe.forward_route(fixture.x_f32, tokens, &router, &scratch, &config);
    const logit_comparison = compareOutput(
        logits[0 .. @as(usize, tokens) * stage.n_padded],
        scratch.router_logits[0 .. @as(usize, tokens) * experts_total],
    );
    var realistic_weights: f64 = 0;
    for (0..@as(usize, tokens) * top_k_total) |index| {
        try std.testing.expectEqual(scratch.topk_indices[index], indices[index]);
        realistic_weights = @max(
            realistic_weights,
            @abs(@as(f64, weights[index]) -
                @as(f64, scratch.topk_values[index])) / @abs(@as(f64, scratch.topk_values[index])),
        );
    }
    std.debug.print(
        "router realistic 16 tokens: indices exact, logit relative error {e:.3} (f32 accumulation order), weight residual {e:.3}\n",
        .{ logit_comparison.relative(), realistic_weights },
    );
}

test "the permute builds a stable expert-sorted list" {
    const allocator = std.testing.allocator;
    var harness = try Harness.open();
    defer harness.close();

    const tokens: u32 = 64;
    var stage = try Stage.init(tokens, experts_total, top_k_total);
    defer stage.deinit();
    const pairs: usize = @as(usize, tokens) * top_k_total;
    const planned = stage.plan(tokens);

    // Expert 7 is routed by the first 24 tokens twice; those pairs must come out
    // in token order, which is what makes the sort stable.
    var indices: [pairs]u32 = undefined;
    for (0..tokens) |token| {
        for (0..top_k_total) |slot| {
            indices[token * top_k_total + slot] =
                if (token < 24 and slot < 2)
                    7
                else
                    @intCast((token * 37 + slot * 101 + 11) % experts_total);
        }
    }
    try uploadAt(
        &harness.stream,
        &stage.indices,
        0,
        std.mem.sliceAsBytes(indices[0..]),
    );
    try launchPermute(
        &harness.kernels,
        &harness.stream,
        &stage.indices,
        tokens,
        top_k_total,
        experts_total,
        &stage.row_index,
        &stage.assign_pos,
        &stage.expert_offsets,
        &stage.expert_counts,
        planned.rows,
    );

    const counts = try allocator.alloc(u32, experts_total);
    defer allocator.free(counts);
    const offsets = try allocator.alloc(u32, experts_total + 1);
    defer allocator.free(offsets);
    const positions = try allocator.alloc(u32, pairs);
    defer allocator.free(positions);
    const rows = try allocator.alloc(u32, planned.rows);
    defer allocator.free(rows);
    try downloadSlice(&harness.stream, counts, &stage.expert_counts);
    try downloadSlice(&harness.stream, offsets, &stage.expert_offsets);
    try downloadSlice(&harness.stream, positions, &stage.assign_pos);
    try downloadSlice(&harness.stream, rows, &stage.row_index);

    // CPU reference: the same counting sort, in pair order.
    var expected_counts: [experts_total]u32 = [_]u32{0} ** experts_total;
    for (indices[0..]) |expert| {
        expected_counts[expert] += 1;
    }
    var expected_offsets: [experts_total + 1]u32 = undefined;
    var cursor: u32 = 0;
    for (0..experts_total) |expert| {
        expected_offsets[expert] = cursor;
        try std.testing.expectEqual(@as(u32, 0), cursor % mma_rows);
        cursor += (expected_counts[expert] + mma_rows - 1) / mma_rows * mma_rows;
    }
    expected_offsets[experts_total] = cursor;
    try std.testing.expect(cursor <= planned.rows);
    try std.testing.expectEqual(@as(u32, 0), cursor % mma_rows);

    const expected_rows = try allocator.alloc(u32, planned.rows);
    defer allocator.free(expected_rows);
    @memset(expected_rows, 0xffffffff);
    var expected_positions: [pairs]u32 = undefined;
    for (0..experts_total) |expert| {
        var position = expected_offsets[expert];
        for (indices[0..], 0..) |routed, pair| {
            if (routed == expert) {
                expected_rows[position] = @intCast(pair / top_k_total);
                expected_positions[pair] = position - expected_offsets[expert];
                position += 1;
            }
        }
    }
    var assignments: u32 = 0;
    for (0..experts_total) |expert| {
        try std.testing.expectEqual(expected_counts[expert], counts[expert]);
        try std.testing.expectEqual(expected_offsets[expert], offsets[expert]);
        assignments += counts[expert];
    }
    try std.testing.expectEqual(@as(u32, @intCast(pairs)), assignments);
    try std.testing.expectEqual(expected_offsets[experts_total], offsets[experts_total]);

    // The gathered list: stable expert order, then padding out to the tile.
    for (0..experts_total) |expert| {
        const start = expected_offsets[expert];
        const count = expected_counts[expert];
        var row = start + count;
        while (row < expected_offsets[expert + 1]) : (row += 1) {
            try std.testing.expectEqual(@as(u32, 0xffffffff), rows[row]);
        }
    }
    // Slots past the live tile count are invalid too: the grouped GEMM's grid
    // covers the upper bound, and every such block must exit without work.
    for (cursor..planned.rows) |row| {
        try std.testing.expectEqual(@as(u32, 0xffffffff), rows[row]);
    }
    try std.testing.expectEqualSlices(u32, expected_rows, rows);
    try std.testing.expectEqualSlices(u32, expected_positions[0..], positions);
    try std.testing.expectEqualSlices(u32, expected_offsets[0..], offsets);

    // Stable within the repeated expert: the two slots of a token stay
    // adjacent, in token order (non-decreasing, since both name one token).
    for (1..counts[7]) |index| {
        try std.testing.expect(rows[offsets[7] + index - 1] <= rows[offsets[7] + index]);
    }
    std.debug.print(
        "permute: {d} assignments, {d} padding rows, {d} hit experts, stable expert 7 = {d} rows\n",
        .{
            assignments,
            offsets[experts_total] - assignments,
            experts_total - std.mem.count(u32, counts, &[_]u32{0}),
            counts[7],
        },
    );
}

/// One full-block evaluation: the device stage, the prepared weights, the host
/// fixtures and the oracle scratch all live here.
const Session = struct {
    allocator: std.mem.Allocator,
    harness: *Harness,
    tokens: u32,
    stage: Stage,
    weights: ExpertWeights,
    fixture: Fixture,
    config: ref_fixtures.RefConfig,
    scratch: ref_moe.Scratch,
    indices: []u32,
    router_weights: []f32,
    counts: []u32,
    offsets: []u32,
    hosts: ?[]HostExpert,
    output: []f32,
    reference: []f32,

    fn open(
        allocator: std.mem.Allocator,
        harness: *Harness,
        tokens: u32,
    ) !Session {
        assert(tokens > 0);
        assert(tokens <= 2048);

        const config = oracleConfig(tokens);
        var session: Session = .{
            .allocator = allocator,
            .harness = harness,
            .tokens = tokens,
            .stage = try Stage.init(tokens, experts_total, top_k_total),
            .weights = try ExpertWeights.init(experts_total),
            .fixture = try Fixture.init(allocator, tokens, input_seed),
            .config = config,
            .scratch = try ref_moe.Scratch.init(allocator, &config),
            .indices = try allocator.alloc(u32, @as(usize, tokens) * top_k_total),
            .router_weights = try allocator.alloc(f32, @as(usize, tokens) * top_k_total),
            .counts = try allocator.alloc(u32, experts_total),
            .offsets = try allocator.alloc(u32, experts_total + 1),
            .hosts = null,
            .output = try allocator.alloc(f32, @as(usize, tokens) * hidden_size),
            .reference = try allocator.alloc(f32, @as(usize, tokens) * hidden_size),
        };
        try session.fixture.uploadWeights(allocator, &harness.stream, &session.stage);
        try session.fixture.uploadInput(allocator, &harness.stream, &session.stage, input_seed);
        return session;
    }

    fn close(session: *Session) void {
        assert(session.output.len > 0);
        assert(session.stage.tokens_max == session.tokens);

        if (session.hosts) |hosts| {
            releaseExperts(session.allocator, hosts);
        }
        session.allocator.free(session.reference);
        session.allocator.free(session.output);
        session.allocator.free(session.offsets);
        session.allocator.free(session.counts);
        session.allocator.free(session.router_weights);
        session.allocator.free(session.indices);
        session.scratch.deinit(session.allocator);
        session.fixture.deinit(session.allocator);
        session.weights.deinit();
        session.stage.deinit();
    }

    /// Runs the router on the first input, checks it against the oracle, then on
    /// a second input so the weight preparation covers the union of both.
    fn route(session: *Session) !void {
        const tokens = session.tokens;
        const stream = &session.harness.stream;
        const kernels = &session.harness.kernels;
        try session.stage.runDispatch(kernels, stream, tokens);
        try downloadSlice(stream, session.indices, &session.stage.indices);
        try downloadSlice(stream, session.router_weights, &session.stage.router_weights);
        try downloadSlice(stream, session.counts, &session.stage.expert_counts);
        try downloadSlice(stream, session.offsets, &session.stage.expert_offsets);

        var router = routeWeights(session.fixture.gate);
        ref_moe.forward_route(
            session.fixture.x_f32,
            tokens,
            &router,
            &session.scratch,
            &session.config,
        );
        var worst: f64 = 0;
        for (0..@as(usize, tokens) * top_k_total) |index| {
            try std.testing.expectEqual(
                session.scratch.topk_indices[index],
                session.indices[index],
            );
            const expected = session.scratch.topk_values[index];
            worst = @max(
                worst,
                @abs(@as(f64, session.router_weights[index]) -
                    @as(f64, expected)) / @abs(@as(f64, expected)),
            );
        }
        // The selection is the gate: indices must match exactly. The weights
        // are reported rather than gated at 1e-6 here because the two engines
        // sum each 2560-term logit in different orders, so their f32 logits
        // differ by ~1e-6 relative; the router test pins the 1e-6 weight
        // contract on exact-logit inputs.
        if (worst > 1e-4) {
            return error.RouterWeightMismatch;
        }
        std.debug.print(
            "block router {d} tokens {d} experts: indices exact vs oracle, max relative weight residual {e:.3} (f32 logit accumulation order)\n",
            .{ tokens, experts_total, worst },
        );

        try session.fixture.uploadInput(
            session.allocator,
            stream,
            &session.stage,
            second_input_seed,
        );
        try session.stage.runDispatch(kernels, stream, tokens);
        const second = try session.allocator.alloc(u32, @as(usize, tokens) * top_k_total);
        defer session.allocator.free(second);
        try downloadSlice(stream, second, &session.stage.indices);

        var seen: [experts_total]bool = [_]bool{false} ** experts_total;
        var union_list: [experts_total]u32 = undefined;
        var union_len: usize = 0;
        for ([_][]const u32{ session.indices, second }) |list| {
            for (list) |expert| {
                assert(expert < experts_total);
                if (!seen[expert]) {
                    seen[expert] = true;
                    union_list[union_len] = expert;
                    union_len += 1;
                }
            }
        }
        session.hosts = try prepareExperts(
            session.allocator,
            stream,
            &session.weights,
            union_list[0..union_len],
            session.counts,
            session.offsets,
        );
        try session.fixture.uploadInput(
            session.allocator,
            stream,
            &session.stage,
            input_seed,
        );
        std.debug.print(
            "prepared {d} distinct experts ({d} assignments over {d} experts, {d} padded rows)\n",
            .{ union_len, tokens * top_k_total, session.counts.len, session.offsets[experts_total] },
        );
    }

    /// The native FP4 block against an as-multiplied f32 oracle: the same
    /// quantised activations and dequantised weights at both stages. Then the
    /// labelled W4A4 trade against the model's unquantised reference.
    fn checkFp4(session: *Session) !void {
        const allocator = session.allocator;
        const hosts = session.hosts orelse return error.ExpertsNotPrepared;
        try session.stage.run(
            &session.harness.kernels,
            &session.harness.stream,
            &session.weights,
            session.tokens,
        );
        try downloadSlice(&session.harness.stream, session.output, &session.stage.out);

        var multiplied = try OracleExperts.init(allocator, hosts, .as_multiplied);
        defer multiplied.deinit(allocator);
        try buildRoutedReference(
            allocator,
            &multiplied,
            hosts,
            session.indices,
            session.router_weights,
            session.fixture.x_quantized,
            session.tokens,
            session.reference,
        );
        var weights = oracleWeights(&multiplied, hosts, session.fixture.gate, &session.fixture.shared);
        ref_moe.forward_shared(
            session.fixture.x_f32,
            session.tokens,
            &weights,
            &session.scratch,
            &session.config,
        );
        for (0..session.reference.len) |index| {
            session.reference[index] += session.scratch.shared_out[index];
        }
        const comparison = compareOutput(session.output, session.reference);
        std.debug.print(
            "MoE FP4 block: max_abs={e:.3} max|ref|={e:.3} relative={e:.3} (gate 2e-2, as-multiplied operands)\n",
            .{ comparison.max_abs_err, comparison.max_reference, comparison.relative() },
        );
        if (comparison.relative() > 2e-2) {
            return error.Fp4BlockMismatch;
        }

        // The model reference: the checkpoint's original f32 weights and the
        // unquantised activations. The gap is the W4A4 trade, reported only.
        var original = try OracleExperts.init(allocator, hosts, .original);
        defer original.deinit(allocator);
        var original_weights = oracleWeights(&original, hosts, session.fixture.gate, &session.fixture.shared);
        ref_moe.forward_route(
            session.fixture.x_f32,
            session.tokens,
            &original_weights,
            &session.scratch,
            &session.config,
        );
        ref_moe.forward_experts(
            session.fixture.x_f32,
            session.tokens,
            &original_weights,
            &session.scratch,
            &session.config,
        );
        ref_moe.forward_shared(
            session.fixture.x_f32,
            session.tokens,
            &original_weights,
            &session.scratch,
            &session.config,
        );
        fillReference(session);
        const trade = compareOutput(session.output, session.reference);
        std.debug.print(
            "MoE FP4 block vs f32 model reference: relative={e:.3} max_abs={e:.3} (labelled W4A4 trade, not a gate)\n",
            .{ trade.relative(), trade.max_abs_err },
        );
    }

    /// Prints the hit histogram and proves the block is not vacuous.
    fn checkNonVacuity(session: *Session) !void {
        const allocator = session.allocator;
        const stream = &session.harness.stream;
        var nonzero: u32 = 0;
        var assignments: u32 = 0;
        var max_count: u32 = 0;
        for (session.counts) |count| {
            if (count > 0) {
                nonzero += 1;
            }
            assignments += count;
            max_count = @max(max_count, count);
        }
        std.debug.print("expert hit histogram ({d} non-zero, sum {d}):", .{ nonzero, assignments });
        for (session.counts, 0..) |count, expert| {
            if (count > 0) {
                std.debug.print(" {d}:{d}", .{ expert, count });
            }
        }
        std.debug.print("\n", .{});
        try std.testing.expectEqual(
            session.tokens * top_k_total,
            assignments,
        );
        try std.testing.expect(nonzero > 1);
        try std.testing.expect(max_count >= 1);

        const planned = session.stage.plan(session.tokens);
        const padded = session.offsets[experts_total];
        try std.testing.expect(padded <= planned.rows);
        const rows = try allocator.alloc(u32, planned.rows);
        defer allocator.free(rows);
        try downloadSlice(stream, rows, &session.stage.row_index);
        var invalid: u32 = 0;
        for (rows) |row| {
            if (row == 0xffffffff) {
                invalid += 1;
            }
        }
        // Padding inside the hit experts' last tiles, plus the slots between the
        // live tile count and the grid's upper bound, which must all be invalid.
        try std.testing.expectEqual(
            padded - assignments + (planned.rows - padded),
            invalid,
        );
        std.debug.print(
            "padded rows: {d} of {d} gathered rows ({d} invalid markers: {d} in-expert + {d} past the live tile count, 16-row MMA alignment)\n",
            .{ padded - assignments, planned.rows, invalid, padded - assignments, planned.rows - padded },
        );

        try session.fixture.uploadInput(
            allocator,
            stream,
            &session.stage,
            second_input_seed,
        );
        try session.stage.run(
            &session.harness.kernels,
            stream,
            &session.weights,
            session.tokens,
        );
        const second = try allocator.alloc(f32, @as(usize, session.tokens) * hidden_size);
        defer allocator.free(second);
        try downloadSlice(stream, second, &session.stage.out);
        var differing: usize = 0;
        for (session.output, second) |first, other| {
            if (@as(u32, @bitCast(first)) != @as(u32, @bitCast(other))) {
                differing += 1;
            }
        }
        if (differing == 0) {
            return error.IdenticalOutputs;
        }
        std.debug.print(
            "two different inputs: {d} of {d} outputs differ\n",
            .{ differing, second.len },
        );
        try session.fixture.uploadInput(allocator, stream, &session.stage, input_seed);
    }

    /// The exact BF16-dequantised reference path against the same oracle, with
    /// both shared-expert implementations on the same routed output: the FFMA
    /// path first (the pre-tensor-core number), then the shipped tensor-core
    /// path, which is the one gated at 5e-4.
    fn checkExact(session: *Session) !void {
        const allocator = session.allocator;
        const hosts = session.hosts orelse return error.ExpertsNotPrepared;
        const planned = session.stage.plan(session.tokens);
        var exact = try Exact.init(allocator, planned.rows);
        defer exact.deinit(allocator);

        const ffma_output = try allocator.alloc(f32, @as(usize, session.tokens) * hidden_size);
        defer allocator.free(ffma_output);
        try runExact(
            &session.stage,
            &session.harness.kernels,
            &session.harness.stream,
            &exact,
            hosts,
            session.tokens,
            .exact_ffma,
        );
        try downloadSlice(&session.harness.stream, ffma_output, &session.stage.out);

        var multiplied = try OracleExperts.init(allocator, hosts, .as_multiplied);
        defer multiplied.deinit(allocator);
        var weights = oracleWeights(&multiplied, hosts, session.fixture.gate, &session.fixture.shared);
        ref_moe.forward_route(
            session.fixture.x_f32,
            session.tokens,
            &weights,
            &session.scratch,
            &session.config,
        );
        ref_moe.forward_experts(
            session.fixture.x_f32,
            session.tokens,
            &weights,
            &session.scratch,
            &session.config,
        );
        ref_moe.forward_shared(
            session.fixture.x_f32,
            session.tokens,
            &weights,
            &session.scratch,
            &session.config,
        );
        fillReference(session);
        const ffma = compareOutput(ffma_output, session.reference);
        std.debug.print(
            "MoE exact BF16-dequant block, shared expert on the FFMA path: max_abs={e:.3} max|ref|={e:.3} relative={e:.3} (gate 5e-4)\n",
            .{ ffma.max_abs_err, ffma.max_reference, ffma.relative() },
        );

        try runExact(
            &session.stage,
            &session.harness.kernels,
            &session.harness.stream,
            &exact,
            hosts,
            session.tokens,
            .tensor_core,
        );
        try downloadSlice(&session.harness.stream, session.output, &session.stage.out);
        const comparison = compareOutput(session.output, session.reference);
        std.debug.print(
            "MoE exact BF16-dequant block, shared expert on the BF16 tensor-core path: max_abs={e:.3} max|ref|={e:.3} relative={e:.3} (gate 5e-4)\n",
            .{ comparison.max_abs_err, comparison.max_reference, comparison.relative() },
        );
        const shared_delta = compareOutput(session.output, ffma_output);
        std.debug.print(
            "MoE shared expert tensor-core vs FFMA on the same inputs: max_abs={e:.3} relative={e:.3} (gate 1e-4; operand split 2^-18, accumulation order)\n",
            .{ shared_delta.max_abs_err, shared_delta.relative() },
        );
        // The witness gate. The tensor-core path may move the last digits of the
        // f32 result, not approach the block gate: one part in 1e4 of the FFMA
        // path's own output is the whole budget, and it fires if the shared
        // expert ever stops splitting its down-projection activation.
        if (shared_delta.relative() > 1e-4) {
            return error.SharedPathDivergence;
        }
        if (comparison.relative() > 5e-4) {
            return error.ExactBlockMismatch;
        }
    }
};

/// Sums the oracle's routed and shared outputs into the comparison buffer.
fn fillReference(session: *Session) void {
    assert(session.scratch.expert_out.len >= @as(usize, session.tokens) * hidden_size);
    assert(session.reference.len == @as(usize, session.tokens) * hidden_size);

    for (0..session.reference.len) |index| {
        session.reference[index] =
            session.scratch.expert_out[index] + session.scratch.shared_out[index];
    }
}

/// Which side of the DRAM tier the expert weights are on for a timed phase.
const Cache = enum { warm, cold };

/// Which part of the block a measurement times.
const Phase = enum { block, routed, shared, dispatch, combine };

/// One phase's `samples_per_measurement` launches: median and worst.
const Timing = struct {
    median_ns: u64,
    max_ns: u64,
};

/// One RTX 5060 Ti's block-scaled FP4 tensor-core peak: the denominator
/// `plan.md` §7c quotes routed efficiency against.
const fp4_peak_tflops: f64 = 192.5;

/// The shared expert's cost per token per layer before it moved onto the bf16
/// tensor cores: M-001's measured 20,261.8 us at M=16,384 warm on one GPU,
/// i.e. 1.237 us. This box's own pre-change run of the same sweep measured
/// 1.292 us, so the bound holds the shipped path to the better of the two.
const shared_micros_per_token_before: f64 = 1.237;

/// Text layers that run this block (`mem/placement.zig`'s `layer_count`). The
/// sweep measures one layer; the derived row multiplies by this and nothing
/// else.
const model_moe_layers: u32 = 48;

/// The cold tier's source: every expert byte in pinned host memory, taken from
/// the bytes the device holds so a cold phase differs from a warm one only in
/// the transfer.
///
/// The kernels read device pointers, so an evicted slab cannot be executed at
/// all; the cold cost is the fetch that makes it resident again. That fetch is
/// a real PCIe transfer of the same bytes, timed inside the phase, and the warm
/// run of the same binary in the same process is its control: if the two do not
/// differ by the transfer, the bytes never left the device.
const ExpertHostCopy = struct {
    gate_up_frag: cuda_context.PinnedHostBuffer,
    gate_up_scale: cuda_context.PinnedHostBuffer,
    down_frag: cuda_context.PinnedHostBuffer,
    down_scale: cuda_context.PinnedHostBuffer,

    fn capture(
        stream: *const cuda_context.Stream,
        weights: *const ExpertWeights,
    ) cuda_driver.DriverError!ExpertHostCopy {
        assert(weights.gate_up_frag.bytes > 0);
        assert(weights.down_scale.bytes > 0);

        const gate_up_frag = try cuda_context.PinnedHostBuffer.allocate(weights.gate_up_frag.bytes);
        errdefer gate_up_frag.free();
        const gate_up_scale = try cuda_context.PinnedHostBuffer.allocate(weights.gate_up_scale.bytes);
        errdefer gate_up_scale.free();
        const down_frag = try cuda_context.PinnedHostBuffer.allocate(weights.down_frag.bytes);
        errdefer down_frag.free();
        const down_scale = try cuda_context.PinnedHostBuffer.allocate(weights.down_scale.bytes);
        errdefer down_scale.free();

        try stream.copy_d_to_h(&gate_up_frag, &weights.gate_up_frag, weights.gate_up_frag.bytes);
        try stream.copy_d_to_h(&gate_up_scale, &weights.gate_up_scale, weights.gate_up_scale.bytes);
        try stream.copy_d_to_h(&down_frag, &weights.down_frag, weights.down_frag.bytes);
        try stream.copy_d_to_h(&down_scale, &weights.down_scale, weights.down_scale.bytes);
        try stream.synchronize();
        return .{
            .gate_up_frag = gate_up_frag,
            .gate_up_scale = gate_up_scale,
            .down_frag = down_frag,
            .down_scale = down_scale,
        };
    }

    fn release(copy: *ExpertHostCopy) void {
        assert(copy.gate_up_frag.bytes > 0);
        assert(copy.down_scale.bytes > 0);

        copy.down_scale.free();
        copy.down_frag.free();
        copy.gate_up_scale.free();
        copy.gate_up_frag.free();
    }

    fn bytes(copy: *const ExpertHostCopy) u64 {
        assert(copy.gate_up_frag.bytes > 0);
        assert(copy.down_scale.bytes > 0);

        return copy.gate_up_frag.bytes + copy.gate_up_scale.bytes +
            copy.down_frag.bytes + copy.down_scale.bytes;
    }

    /// Enqueues the tier's H2D fetch on the block's stream.
    fn fetch(
        copy: *const ExpertHostCopy,
        stream: *const cuda_context.Stream,
        weights: *const ExpertWeights,
    ) cuda_driver.DriverError!void {
        assert(copy.bytes() > 0);
        assert(weights.gate_up_frag.bytes == copy.gate_up_frag.bytes);

        try stream.copy_h_to_d(&weights.gate_up_frag, &copy.gate_up_frag, copy.gate_up_frag.bytes);
        try stream.copy_h_to_d(&weights.gate_up_scale, &copy.gate_up_scale, copy.gate_up_scale.bytes);
        try stream.copy_h_to_d(&weights.down_frag, &copy.down_frag, copy.down_frag.bytes);
        try stream.copy_h_to_d(&weights.down_scale, &copy.down_scale, copy.down_scale.bytes);
    }
};

/// Runs what a phase reads but does not time. Only the routed phase needs it:
/// dispatch is a measured term of its own, so it is prepared outside the region.
fn preparePhase(
    phase: Phase,
    harness: *Harness,
    stage: *Stage,
    tokens: u32,
) KernelError!void {
    assert(tokens > 0);
    assert(tokens <= stage.tokens_max);

    switch (phase) {
        .routed => try stage.runDispatch(&harness.kernels, &harness.stream, tokens),
        else => {},
    }
}

/// Runs one timed phase.
fn runPhase(
    phase: Phase,
    harness: *Harness,
    stage: *Stage,
    weights: *const ExpertWeights,
    tokens: u32,
) KernelError!void {
    assert(tokens > 0);
    assert(tokens <= stage.tokens_max);

    switch (phase) {
        .block => try stage.run(&harness.kernels, &harness.stream, weights, tokens),
        .routed => try stage.runRouted(&harness.kernels, &harness.stream, weights, tokens),
        .shared => try stage.runShared(&harness.kernels, &harness.stream, tokens, .tensor_core),
        .dispatch => try stage.runDispatch(&harness.kernels, &harness.stream, tokens),
        .combine => try stage.runCombine(&harness.kernels, &harness.stream, tokens),
    }
}

/// Timed launches per measurement.
const samples_per_measurement: u32 = 7;

/// Runs one phase several times and returns the median and the worst sample.
///
/// A cold phase enqueues the tier fetch inside the timed region, after the
/// untimed prepare, so the sample is what production pays for a miss: the
/// transfer plus the kernels.
fn timePhase(
    harness: *Harness,
    stage: *Stage,
    weights: *const ExpertWeights,
    tokens: u32,
    phase: Phase,
    cache: Cache,
    host_copy: ?*const ExpertHostCopy,
) (KernelError || error{ImpossibleTiming})!Timing {
    assert(tokens > 0);
    assert(tokens <= stage.tokens_max);
    assert(samples_per_measurement >= 3);
    assert(cache == .warm or host_copy != null);

    try preparePhase(phase, harness, stage, tokens);
    if (cache == .cold) try host_copy.?.fetch(&harness.stream, weights);
    try runPhase(phase, harness, stage, weights, tokens);
    try harness.stream.synchronize();

    const start = try cuda_context.Event.create();
    defer start.destroy();
    const end = try cuda_context.Event.create();
    defer end.destroy();

    var measured: [samples_per_measurement]u64 = undefined;
    for (&measured) |*sample| {
        try preparePhase(phase, harness, stage, tokens);
        try start.record(&harness.stream);
        if (cache == .cold) try host_copy.?.fetch(&harness.stream, weights);
        try runPhase(phase, harness, stage, weights, tokens);
        try end.record(&harness.stream);
        try end.synchronize();
        sample.* = try cuda_context.Event.elapsed_nanoseconds(&start, &end);
        if (sample.* == 0) {
            return error.ImpossibleTiming;
        }
    }
    std.mem.sort(u64, measured[0..], {}, std.sort.asc(u64));
    return .{
        .median_ns = measured[samples_per_measurement / 2],
        .max_ns = measured[samples_per_measurement - 1],
    };
}

/// Times the cold fetch on its own: the tier's PCIe transfer with no kernel
/// behind it, which is the number the cold phases are checked against.
fn timeFetch(
    harness: *Harness,
    host_copy: *const ExpertHostCopy,
    weights: *const ExpertWeights,
) (KernelError || error{ImpossibleTiming})!Timing {
    assert(host_copy.bytes() > 0);
    assert(samples_per_measurement >= 3);

    try host_copy.fetch(&harness.stream, weights);
    try harness.stream.synchronize();

    const start = try cuda_context.Event.create();
    defer start.destroy();
    const end = try cuda_context.Event.create();
    defer end.destroy();

    var measured: [samples_per_measurement]u64 = undefined;
    for (&measured) |*sample| {
        try start.record(&harness.stream);
        try host_copy.fetch(&harness.stream, weights);
        try end.record(&harness.stream);
        try end.synchronize();
        sample.* = try cuda_context.Event.elapsed_nanoseconds(&start, &end);
        if (sample.* == 0) {
            return error.ImpossibleTiming;
        }
    }
    std.mem.sort(u64, measured[0..], {}, std.sort.asc(u64));
    return .{
        .median_ns = measured[samples_per_measurement / 2],
        .max_ns = measured[samples_per_measurement - 1],
    };
}

/// One size's warm terms, all medians in nanoseconds.
const WarmTerms = struct {
    block: Timing,
    dispatch: Timing,
    routed: Timing,
    shared: Timing,
    combine: Timing,
};

/// One size's cold terms and the fetch that stands behind them.
const ColdTerms = struct {
    fetch: Timing,
    routed: Timing,
    block: Timing,
};

/// Times the warm block and each term of it. The block runs first so the
/// buffers its epilogue reads are valid for the term measurements.
fn measureWarm(
    harness: *Harness,
    stage: *Stage,
    weights: *const ExpertWeights,
    tokens: u32,
) (KernelError || error{ImpossibleTiming})!WarmTerms {
    assert(tokens > 0);
    assert(tokens <= stage.tokens_max);

    var terms: WarmTerms = undefined;
    terms.block = try timePhase(harness, stage, weights, tokens, .block, .warm, null);
    terms.dispatch = try timePhase(harness, stage, weights, tokens, .dispatch, .warm, null);
    terms.routed = try timePhase(harness, stage, weights, tokens, .routed, .warm, null);
    terms.shared = try timePhase(harness, stage, weights, tokens, .shared, .warm, null);
    terms.combine = try timePhase(harness, stage, weights, tokens, .combine, .warm, null);
    return terms;
}

/// Times the cold tier: the fetch alone, then routed and the whole block with
/// that fetch inside the region.
fn measureCold(
    harness: *Harness,
    stage: *Stage,
    weights: *const ExpertWeights,
    host_copy: *const ExpertHostCopy,
    tokens: u32,
) (KernelError || error{ImpossibleTiming})!ColdTerms {
    assert(tokens > 0);
    assert(host_copy.bytes() > 0);

    var terms: ColdTerms = undefined;
    terms.fetch = try timeFetch(harness, host_copy, weights);
    terms.routed = try timePhase(harness, stage, weights, tokens, .routed, .cold, host_copy);
    terms.block = try timePhase(harness, stage, weights, tokens, .block, .cold, host_copy);
    return terms;
}

/// Routed-expert FLOPs of one block: gate_up plus down for each assignment.
fn routedFlops(tokens: u32) f64 {
    assert(tokens > 0);

    const width: f64 = @floatFromInt(2 * expert_intermediate);
    const height: f64 = @floatFromInt(hidden_size);
    const down_k: f64 = @floatFromInt(expert_intermediate);
    const per_row = 2.0 * (width * height + height * down_k);
    return @as(f64, @floatFromInt(tokens)) *
        @as(f64, @floatFromInt(top_k_total)) * per_row;
}

/// Bytes `Stage.init` allocates for `tokens_max`, page-rounded per buffer: the
/// working set a sweep size needs, reported when it does not fit. The MoE block
/// is KV-free, so this is activations and gathered rows only. Mirrors the
/// terms in `Stage.init`.
fn stageBytes(tokens_max: u32, n_experts: u32, top_k: u32) u64 {
    assert(tokens_max > 0);
    assert(n_experts > 0 and n_experts <= experts_total);

    const n_padded = paddedExperts(n_experts);
    const tiles = tilesUpper(tokens_max, top_k, n_experts);
    const rows = tiles * mma_rows;
    const gate_up_frag_words = pack_nvfp4.activationFragmentWords(rows, hidden_size);
    const gate_up_scale_words = pack_nvfp4.activationScaleWords(rows, hidden_size);
    const down_frag_words = pack_nvfp4.activationFragmentWords(rows, expert_intermediate);
    const down_scale_words = pack_nvfp4.activationScaleWords(rows, expert_intermediate);

    var total: u64 = 0;
    total += pageRounded(@as(u64, n_padded) * hidden_size * 2);
    total += pageRounded(@as(u64, tokens_max) * n_padded * 4);
    total += pageRounded(@as(u64, tokens_max) * top_k * 4);
    total += pageRounded(@as(u64, tokens_max) * top_k * 4);
    total += pageRounded(@as(u64, rows) * 4);
    total += pageRounded(@as(u64, tokens_max) * top_k * 4);
    total += pageRounded(@as(u64, n_experts + 1) * 4);
    total += pageRounded(@as(u64, n_experts) * 4);
    total += pageRounded(@as(u64, tiles) * 4);
    total += pageRounded(@as(u64, tiles) * 4);
    total += pageRounded(@as(u64, tiles) * 4);
    total += pageRounded(gate_up_frag_words * 4);
    total += pageRounded(gate_up_scale_words * 4);
    total += pageRounded(@as(u64, rows) * 2 * expert_intermediate * 4);
    total += pageRounded(down_frag_words * 4);
    total += pageRounded(down_scale_words * 4);
    total += pageRounded(@as(u64, rows) * hidden_size * 4);
    total += pageRounded(@as(u64, tokens_max) * hidden_size * 2);
    total += pageRounded(@as(u64, tokens_max) * hidden_size * 4);
    total += pageRounded(@as(u64, tokens_max) * hidden_size * 4);
    total += pageRounded(@as(u64, 2 * shared_intermediate) * hidden_size * 2);
    total += pageRounded(@as(u64, hidden_size) * shared_intermediate * 2);
    total += pageRounded(@as(u64, hidden_size) * 4);
    total += pageRounded(@as(u64, tokens_max) * 2 * shared_intermediate * 4);
    total += pageRounded(@as(u64, tokens_max) * shared_intermediate * 4);
    total += pageRounded(@as(u64, tokens_max) * shared_intermediate * 2);
    total += pageRounded(@as(u64, tokens_max) * shared_intermediate * 2);
    total += pageRounded(@as(u64, tokens_max) * hidden_size * 4);
    assert(total > 0);
    assert(total >= @as(u64, rows) * hidden_size * 4);
    return total;
}

/// Bytes the resident expert tier occupies: the four per-expert buffers.
fn expertBytes(weights: *const ExpertWeights) u64 {
    assert(weights.gate_up_frag.bytes > 0);
    assert(weights.down_scale.bytes > 0);

    return weights.gate_up_frag.bytes + weights.gate_up_scale.bytes +
        weights.down_frag.bytes + weights.down_scale.bytes;
}

/// Prints one phase row: p50, worst sample, both rates and the per-layer
/// per-token cost. Every row names its cache state and its parallelism.
fn reportPhase(tokens: u32, phase: Phase, cache: Cache, timing: Timing) void {
    assert(tokens > 0);
    assert(timing.median_ns > 0);
    assert(timing.max_ns >= timing.median_ns);

    const median_us = @as(f64, @floatFromInt(timing.median_ns)) / 1e3;
    const max_us = @as(f64, @floatFromInt(timing.max_ns)) / 1e3;
    const seconds = @as(f64, @floatFromInt(timing.median_ns)) / 1e9;
    const rows_per_second = @as(f64, @floatFromInt(tokens)) / seconds;
    const micros_per_token = median_us / @as(f64, @floatFromInt(tokens));
    std.debug.print(
        "MoE M={d:<5} 1gpu {s:<4} {s:<8} p50 {d:>9.1} us max {d:>9.1} us = {d:>7.0} tok/s = {d:>6.3} us/token/layer\n",
        .{ tokens, @tagName(cache), @tagName(phase), median_us, max_us, rows_per_second, micros_per_token },
    );
}

/// Routed-only efficiency against the card's block-scaled FP4 peak. Cold rows
/// are not printed through here: they include a PCIe fetch, not kernel time.
fn reportRoutedEfficiency(tokens: u32, cache: Cache, timing: Timing) void {
    assert(tokens > 0);
    assert(timing.median_ns > 0);
    assert(fp4_peak_tflops > 0.0);

    const seconds = @as(f64, @floatFromInt(timing.median_ns)) / 1e9;
    const tflops = routedFlops(tokens) / seconds / 1e12;
    const percent = tflops / fp4_peak_tflops * 100.0;
    std.debug.print(
        "MoE M={d:<5} 1gpu {s:<4} routed efficiency {d:>6.2} TFLOPS = {d:>5.1}% of the {d:.1} TFLOPS OMMA.SF peak\n",
        .{ tokens, @tagName(cache), tflops, percent, fp4_peak_tflops },
    );
}

/// Prints the cold fetch: every expert byte of one layer, host to device.
fn reportFetch(tokens: u32, timing: Timing, bytes: u64) void {
    assert(tokens > 0);
    assert(bytes > 0);
    assert(timing.median_ns > 0);

    const median_us = @as(f64, @floatFromInt(timing.median_ns)) / 1e3;
    const max_us = @as(f64, @floatFromInt(timing.max_ns)) / 1e3;
    const seconds = @as(f64, @floatFromInt(timing.median_ns)) / 1e9;
    const gigabytes = @as(f64, @floatFromInt(bytes)) / 1e9;
    const rate = gigabytes / seconds;
    const micros_per_token = median_us / @as(f64, @floatFromInt(tokens));
    std.debug.print(
        "MoE M={d:<5} 1gpu cold fetch p50 {d:>9.1} us max {d:>9.1} us for {d:>6.3} GB = {d:>6.1} GB/s from pinned host DRAM = {d:>6.3} us/token/layer\n",
        .{ tokens, median_us, max_us, gigabytes, rate, micros_per_token },
    );
}

/// The one derived row: the measured block times the model's 48 MoE layers.
fn reportDerived(tokens: u32, block: Timing) void {
    assert(tokens > 0);
    assert(block.median_ns > 0);
    assert(model_moe_layers > 0);

    const seconds = @as(f64, @floatFromInt(block.median_ns)) / 1e9;
    const model_seconds = seconds * @as(f64, @floatFromInt(model_moe_layers));
    const model_rate = @as(f64, @floatFromInt(tokens)) / model_seconds;
    const micros_per_token = model_seconds * 1e6 / @as(f64, @floatFromInt(tokens));
    std.debug.print(
        "MoE M={d:<5} 1gpu warm derived {d}-layer MoE only: {d:>7.1} us/token = {d:>7.0} tok/s " ++
            "(measured one-layer block x {d}; excludes attention, PLE, lm_head)\n",
        .{ tokens, model_moe_layers, micros_per_token, model_rate, model_moe_layers },
    );
}

/// Prints every warm row for one size and checks that the term sum closes on
/// the block, so a term measured in the wrong state cannot pass unnoticed.
fn reportWarm(tokens: u32, terms: WarmTerms) !void {
    assert(tokens > 0);
    assert(terms.block.median_ns > 0);
    assert(terms.combine.median_ns > 0);

    reportPhase(tokens, .block, .warm, terms.block);
    reportPhase(tokens, .dispatch, .warm, terms.dispatch);
    reportPhase(tokens, .routed, .warm, terms.routed);
    reportRoutedEfficiency(tokens, .warm, terms.routed);
    reportPhase(tokens, .shared, .warm, terms.shared);
    try checkSharedBound(tokens, terms.shared);
    reportPhase(tokens, .combine, .warm, terms.combine);

    const sum_ns = terms.dispatch.median_ns + terms.routed.median_ns +
        terms.shared.median_ns + terms.combine.median_ns;
    const sum_us = @as(f64, @floatFromInt(sum_ns)) / 1e3;
    const block_us = @as(f64, @floatFromInt(terms.block.median_ns)) / 1e3;
    std.debug.print(
        "MoE M={d:<5} 1gpu warm terms sum {d:>9.1} us vs block {d:>9.1} us ({d:>5.1}%)\n",
        .{ tokens, sum_us, block_us, sum_us / block_us * 100.0 },
    );
    reportDerived(tokens, terms.block);
}

/// Asserts the shared expert's per-token per-layer cost against the measurement
/// taken before it left the scalar FFMA path. This bound is the reason the
/// tensor-core path exists, so it has to be able to fail: a shared expert that
/// fell back, lost its 64-column tile or picked up a per-token launch overhead
/// lands back above it.
fn checkSharedBound(tokens: u32, shared: Timing) !void {
    assert(tokens > 0);
    assert(shared.median_ns > 0);
    assert(shared_micros_per_token_before > 0.0);

    const micros_per_token =
        @as(f64, @floatFromInt(shared.median_ns)) / 1e3 / @as(f64, @floatFromInt(tokens));
    std.debug.print(
        "MoE M={d:<5} 1gpu warm shared bound: {d:>6.3} us/token/layer vs {d:.3} measured before the " ++
            "tensor-core path = {d:>5.2}x better (bound fails above {d:.3})\n",
        .{
            tokens,
            micros_per_token,
            shared_micros_per_token_before,
            shared_micros_per_token_before / micros_per_token,
            shared_micros_per_token_before,
        },
    );
    if (micros_per_token >= shared_micros_per_token_before) {
        return error.SharedExpertRegression;
    }
}

/// Prints the cold rows for one size: the fetch, then routed and block with
/// that fetch inside the measured region.
fn reportCold(tokens: u32, cold: ColdTerms, host_copy: *const ExpertHostCopy) void {
    assert(tokens > 0);
    assert(cold.fetch.median_ns > 0);
    assert(host_copy.bytes() > 0);

    reportFetch(tokens, cold.fetch, host_copy.bytes());
    reportPhase(tokens, .routed, .cold, cold.routed);
    reportPhase(tokens, .block, .cold, cold.block);
}

/// The warm run is the cold run's control: a cold phase has to be slower than
/// the warm one by roughly the transfer, or the "fetch" never reached the bus.
fn checkColdControl(tokens: u32, warm: WarmTerms, cold: ColdTerms) !void {
    assert(tokens > 0);
    assert(warm.block.median_ns > 0);
    assert(cold.fetch.median_ns > 0);

    const block_extra_ns: i128 = @as(i128, cold.block.median_ns) - @as(i128, warm.block.median_ns);
    const routed_extra_ns: i128 = @as(i128, cold.routed.median_ns) - @as(i128, warm.routed.median_ns);
    const fetch_us = @as(f64, @floatFromInt(cold.fetch.median_ns)) / 1e3;
    const block_us = @as(f64, @floatFromInt(block_extra_ns)) / 1e3;
    const routed_us = @as(f64, @floatFromInt(routed_extra_ns)) / 1e3;
    std.debug.print(
        "MoE M={d:<5} 1gpu cold control: block +{d:>8.1} us ({d:>5.0}% of fetch), " ++
            "routed +{d:>8.1} us ({d:>5.0}% of fetch), fetch {d:>8.1} us\n",
        .{
            tokens,
            block_us,
            block_us / fetch_us * 100.0,
            routed_us,
            routed_us / fetch_us * 100.0,
            fetch_us,
        },
    );
    try std.testing.expect(block_extra_ns > 0);
    try std.testing.expect(routed_extra_ns > 0);
    // Paying under half the measured transfer would mean the cold row is
    // really warm, which is the failure this whole pairing exists to catch.
    try std.testing.expect(block_extra_ns * 2 >= @as(i128, cold.fetch.median_ns));
}

/// Proves a size really routes across the full expert count and prints the
/// gathered-row working set that implies.
fn reportRouting(
    allocator: std.mem.Allocator,
    harness: *Harness,
    stage: *Stage,
    tokens: u32,
) !void {
    assert(tokens > 0);
    assert(stage.tokens_max >= tokens);

    try stage.runDispatch(&harness.kernels, &harness.stream, tokens);
    const counts = try allocator.alloc(u32, experts_total);
    defer allocator.free(counts);
    try downloadSlice(&harness.stream, counts, &stage.expert_counts);
    var hit: u32 = 0;
    for (counts) |count| {
        if (count > 0) hit += 1;
    }
    std.debug.print(
        "MoE M={d:<5} 1gpu routing {d} of {d} experts hit, {d} assignments, {d} gathered rows\n",
        .{ tokens, hit, experts_total, tokens * top_k_total, rowsUpper(tokens, top_k_total, experts_total) },
    );
    try std.testing.expect(hit >= experts_total * 3 / 4);
}

/// One chunk size: allocate its stage, prove routing, time the warm terms and
/// then the cold tier. A size that does not fit is reported with its byte
/// count and skipped, never silently dropped.
fn sweepSize(
    allocator: std.mem.Allocator,
    harness: *Harness,
    weights: *const ExpertWeights,
    fixture: *const Fixture,
    host_copy: ?*const ExpertHostCopy,
    x_bf16: []const u16,
    tokens: u32,
) !void {
    assert(tokens > 0);
    assert(x_bf16.len >= @as(usize, tokens) * hidden_size);

    const required = stageBytes(tokens, experts_total, top_k_total);
    var stage = Stage.init(tokens, experts_total, top_k_total) catch |err| {
        std.debug.print(
            "MoE M={d:<5} 1gpu SKIP: stage allocation failed ({s}); {d} B = {d:.2} GB " ++
                "of KV-free activations plus {d:.2} GB of resident experts would be needed\n",
            .{
                tokens,
                @errorName(err),
                required,
                @as(f64, @floatFromInt(required)) / 1e9,
                @as(f64, @floatFromInt(expertBytes(weights))) / 1e9,
            },
        );
        return;
    };
    defer stage.deinit();

    try fixture.uploadWeights(allocator, &harness.stream, &stage);
    try uploadAt(
        &harness.stream,
        &stage.x,
        0,
        std.mem.sliceAsBytes(x_bf16[0 .. @as(usize, tokens) * hidden_size]),
    );

    try reportRouting(allocator, harness, &stage, tokens);
    const warm = try measureWarm(harness, &stage, weights, tokens);
    try reportWarm(tokens, warm);

    const copy = host_copy orelse {
        std.debug.print(
            "MoE M={d:<5} 1gpu cold SKIP: no pinned host capture of the expert tier\n",
            .{tokens},
        );
        return;
    };
    const cold = try measureCold(harness, &stage, weights, copy, tokens);
    reportCold(tokens, cold, copy);
    try checkColdControl(tokens, warm, cold);
}

/// Reports the block at the chunk sizes prefill actually runs, warm and cold.
fn checkThroughput(
    allocator: std.mem.Allocator,
    harness: *Harness,
    weights: *const ExpertWeights,
    fixture: *const Fixture,
    source_expert: u32,
) !void {
    assert(experts_total > 0);
    assert(source_expert < experts_total);

    const max_tokens: u32 = 32768;
    const x_bf16 = try allocator.alloc(u16, @as(usize, max_tokens) * hidden_size);
    defer allocator.free(x_bf16);
    const x_f32 = try allocator.alloc(f32, @as(usize, max_tokens) * hidden_size);
    defer allocator.free(x_f32);
    fillInput(x_bf16, x_f32, input_seed ^ 0x9e37_79b9);
    try broadcastExperts(&harness.stream, weights, experts_total, source_expert);

    var host_copy: ?ExpertHostCopy = ExpertHostCopy.capture(&harness.stream, weights) catch |err| blk: {
        std.debug.print(
            "MoE cold tier unavailable: pinned host capture failed ({s}); warm rows only\n",
            .{@errorName(err)},
        );
        break :blk null;
    };
    defer {
        if (host_copy) |*copy| copy.release();
    }

    std.debug.print(
        "MoE prefill chunk sweep on one GPU: {s} (PCI {s}, {d:.1} GB VRAM). " ++
            "1gpu = single GPU, 1 rank, EP=1 (all {d} experts resident), no TP, no replication.\n",
        .{
            harness.device.name_slice(),
            harness.device.pci_bus_id_slice(),
            @as(f64, @floatFromInt(harness.device.vram_bytes)) / 1e9,
            experts_total,
        },
    );
    std.debug.print(
        "One block = one MoE layer. `cold` fetches the layer's {d:.4} GB expert tier from pinned host DRAM " ++
            "over PCIe inside the timed region, then runs the same kernels; `warm` reads the resident device copy.\n",
        .{@as(f64, @floatFromInt(expertBytes(weights))) / 1e9},
    );

    const shapes = [_]u32{ 2048, 4096, 8192, 16384, 32768 };
    for (shapes) |tokens| {
        const copy = if (host_copy) |*copy| copy else null;
        try sweepSize(allocator, harness, weights, fixture, copy, x_bf16, tokens);
    }
}

test "the MoE block matches the fp32 oracle at the real shape" {
    const allocator = std.testing.allocator;
    var harness = try Harness.open();
    defer harness.close();

    const tokens: u32 = 4;
    var session = try Session.open(allocator, &harness, tokens);
    defer session.close();
    try session.route();
    try session.checkFp4();
    try session.checkNonVacuity();
    try session.checkExact();

    const hosts = session.hosts orelse return error.ExpertsNotPrepared;
    try checkThroughput(
        allocator,
        &harness,
        &session.weights,
        &session.fixture,
        hosts[0].expert,
    );
}
