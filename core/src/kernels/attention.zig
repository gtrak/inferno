//! Launch wrappers for the indexed (QSA) attention kernels, and the harness
//! that checks them against the fp32 reference.
//!
//! `src/kernels/attention.cu` holds the device code; this file owns the launch
//! geometry, the buffer contracts and the copies. Every buffer describes one
//! sequence: the KV arena is a single contiguous `[arena_length, kv_heads,
//! 2 * head_dim]` region stored once (the target design replicates attention
//! across ranks rather than tensor-parallelising it), and positions are
//! absolute, so the same arena serves prefill and decode.
//!
//! The tests at the bottom are the harness of issue I-008: they run every
//! kernel on the GPU and compare against `src/ref/attention.zig` (the fp32
//! oracle), against a brute-force reference top-k, and against the
//! `layer2/attn/*` goldens of the HF trace. They run under `zig build test` and
//! under `zig build attntest`, which is the issue's own gate and builds this
//! file alone.

const std = @import("std");

const cuda_context = @import("../hw/cuda_context.zig");
const cuda_driver = @import("../hw/cuda_driver.zig");
const kernel = @import("../hw/kernel.zig");
const generated_cubins = @import("generated_cubins");
const ops = @import("../ref/ops.zig");
const reference_attention = @import("../ref/attention.zig");
const reference_fixtures = @import("../ref/fixtures.zig");
const reference_rope = @import("../ref/rope.zig");
const quant = @import("../weights/quant.zig");

const assert = std.debug.assert;

/// Cubin compiled from `src/kernels/attention.cu`.
const cubin_name = "attention";

/// Query heads one prefill block attends to; matches `prefill_warps` in the
/// device file, which is also the number of warps per block.
const prefill_warps: u32 = 4;

/// Warps one decode block spreads the selection over.
const decode_warps: u32 = 8;

/// Query rows one tensor-core prefill block attends to; mirrors `mma_rows` in
/// the device file.
const prefill_mma_rows: u32 = 16;

/// Threads per tensor-core prefill block; mirrors `mma_threads`.
const prefill_mma_threads: u32 = 128;

/// Widest head dim the tensor-core prefill path handles; the V tile has to stay
/// inside the static shared-memory budget.
const prefill_mma_head_dim_max: u32 = 256;

/// Fewest query rows the tensor-core path needs to pay for its 16-row m-tile.
/// A single-row launch is a decode-shaped step and stays on the warp-scalar
/// kernel, which does not pad fifteen rows of tensor-core work and keeps the
/// `decode == prefill` invariant on identical arithmetic.
const prefill_mma_min_rows: u32 = 8;

/// Threads per norm, pool and score block.
const index_threads: u32 = 128;

/// Blocks one CTA scores in the per-block score stage; it mirrors
/// `scores_per_cta` in `attention.cu`.
const scores_per_cta: u32 = 16;

/// Threads per selection block.
const select_threads: u32 = 512;

/// Selected slots the decode kernel holds in shared memory.
const selected_slots_max: u32 = 2064;

/// Block count the selection can keep; `block_topk` is `budget / ratio`.
const block_topk_max: u32 = 512;

/// Head dims one warp pass covers.
const dims_per_warp_pass: u32 = 64;

/// Passes the kernels unroll; `head_dim <= lane_passes_max * 64`.
const lane_passes_max: u32 = 8;

/// Blocks one indexer score group covers. The score buffer is a fixed
/// `min(rows, indexer_row_chunk) * score_group_blocks` fp32 allocation, so it
/// does not change with the context at all: past 131 072 tokens at
/// `compress_ratio` 4 the wrapper folds several groups into the running top-k
/// instead of growing the buffer.
pub const score_group_blocks: u32 = 32768;

/// Rows one streaming pass scores at once, so the fixed score buffer stays
/// bounded even at a full batch prefill.
pub const indexer_row_chunk: u32 = 256;

/// Indexer heads whose partial dots fit in one block's shared memory.
const index_heads_max: u32 = 32;

/// Longest rope table this layer asks for: the model's context limit.
const positions_max: u32 = 1 << 20;

/// Threads per rope table block.
const rope_table_threads: u32 = 64;

/// Threads per rope application block.
const rope_apply_threads: u32 = 256;

/// Bytes one device allocation rounds up to (`cuMemAlloc` alignment).
const allocation_granularity: u64 = 4096;

/// The shapes one indexed-attention layer runs at.
///
/// The kernels take every extent as an argument, so the tiny golden fixture
/// (4 q heads, head_dim 32, indexer dim 16) and the real checkpoint (24 heads,
/// head_dim 256, indexer dim 128) share one implementation.
pub const Shapes = struct {
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    index_heads: u32,
    index_kv_heads: u32,
    index_dim: u32,
    compress_ratio: u32,
    budget: u32,
    /// Dims each head rotates: `head_dim * partial_rotary_factor`.
    rotary_dim: u32,
    /// `rope_parameters.mrope_section`: temporal, height, width.
    section: [3]u32,
    /// `rms_norm_eps`, used by the indexer's own norms.
    eps: f32,
    /// `rope_parameters.rope_theta`.
    theta: f32,

    /// nvidia/Qwen3.8-Flash-Next-NVFP4, `text_config`.
    pub const real: Shapes = .{
        .heads = 24,
        .kv_heads = 2,
        .head_dim = 256,
        .index_heads = 4,
        .index_kv_heads = 1,
        .index_dim = 128,
        .compress_ratio = 4,
        .budget = 2048,
        .rotary_dim = 64,
        .section = .{ 11, 11, 10 },
        .eps = 1e-6,
        .theta = 10_000_000.0,
    };

    /// Slots of the `selected` list per query: `budget + ratio - 1`.
    pub fn slots(self: Shapes) u32 {
        assert(self.budget > 0);
        assert(self.compress_ratio > 0);

        return self.budget + self.compress_ratio - 1;
    }

    /// Blocks the indexer keeps: `budget / compress_ratio`.
    pub fn blockTopk(self: Shapes) u32 {
        assert(self.budget > 0);
        assert(self.compress_ratio > 0);

        return @divExact(self.budget, self.compress_ratio);
    }

    /// Query heads one kv head serves.
    pub fn group(self: Shapes) u32 {
        assert(self.heads > 0);
        assert(self.kv_heads > 0);

        return @divExact(self.heads, self.kv_heads);
    }

    /// Columns `index_qk_proj` produces: the index queries plus the token key.
    pub fn projectedWidth(self: Shapes) u32 {
        assert(self.index_heads > 0);
        assert(self.index_kv_heads > 0);

        return (self.index_heads + self.index_kv_heads) * self.index_dim;
    }

    /// Warp passes one head takes.
    pub fn lanePasses(self: Shapes) u32 {
        assert(self.head_dim > 0);
        assert(self.head_dim % 2 == 0);

        return (self.head_dim + dims_per_warp_pass - 1) / dims_per_warp_pass;
    }

    /// Rejects shapes the kernels cannot launch.
    pub fn validate(self: Shapes) void {
        assert(self.heads > 0);
        assert(self.heads % self.kv_heads == 0);
        assert(self.head_dim > 0);
        assert(self.head_dim % 2 == 0);
        assert(self.lanePasses() <= lane_passes_max);
        assert(self.index_heads > 0);
        assert(self.index_heads <= index_heads_max);
        assert(self.index_dim > 0);
        assert(self.index_dim % 2 == 0);
        assert(self.rotary_dim > 0);
        assert(self.rotary_dim % 2 == 0);
        assert(self.rotary_dim <= self.head_dim);
        assert(self.rotary_dim <= self.index_dim);
        assert(self.slots() <= selected_slots_max);
        assert(self.blockTopk() <= block_topk_max);
        assert(self.blockTopk() > 0);
        assert(self.section[0] > 0);
        assert(self.eps > 0.0);
        assert(self.theta > 1.0);
    }
};

/// Frequency lanes one rotated head has: `rotary_dim / 2`.
pub fn rotaryPairs(rotary_dim: u32) u32 {
    assert(rotary_dim > 0);
    assert(rotary_dim % 2 == 0);
    return rotary_dim / 2;
}

/// Entry points of the attention cubin.
pub const Kernels = struct {
    module: kernel.Module,
    rope_table_build: kernel.Function,
    rope_apply_rotary: kernel.Function,
    indexer_project: kernel.Function,
    indexer_norm_rope_q: kernel.Function,
    indexer_append_key: kernel.Function,
    indexer_pool_blocks: kernel.Function,
    indexer_block_scores: kernel.Function,
    indexer_select: kernel.Function,
    indexer_group_topk: kernel.Function,
    indexer_expand: kernel.Function,
    attn_prefill: kernel.Function,
    attn_prefill_mma: kernel.Function,
    attn_decode: kernel.Function,

    /// Loads the cubin and resolves every entry point; the context must be
    /// current.
    pub fn init() !Kernels {
        assert(cubin_name.len > 0);
        assert(generated_cubins.cubins.len > 0);

        const image = try cubinImage(cubin_name);
        const module = try kernel.Module.load(image);
        errdefer module.unload();
        return .{
            .module = module,
            .rope_table_build = try module.function("rope_table_build"),
            .rope_apply_rotary = try module.function("rope_apply_rotary"),
            .indexer_project = try module.function("indexer_project"),
            .indexer_norm_rope_q = try module.function("indexer_norm_rope_q"),
            .indexer_append_key = try module.function("indexer_append_key"),
            .indexer_pool_blocks = try module.function("indexer_pool_blocks"),
            .indexer_block_scores = try module.function(
                "indexer_block_scores",
            ),
            .indexer_select = try module.function("indexer_select"),
            .indexer_group_topk = try module.function("indexer_group_topk"),
            .indexer_expand = try module.function("indexer_expand"),
            .attn_prefill = try module.function("attn_prefill"),
            .attn_prefill_mma = try module.function("attn_prefill_mma"),
            .attn_decode = try module.function("attn_decode"),
        };
    }

    /// Unloads the module; the owning context must be current.
    pub fn deinit(self: *Kernels) void {
        assert(self.module.handle != null);
        assert(@intFromPtr(self.module.handle) != 0);

        self.module.unload();
    }
};

/// The image of the cubin named `name` inside the static library.
fn cubinImage(name: []const u8) ![]const u8 {
    assert(name.len > 0);
    assert(generated_cubins.cubins.len > 0);

    for (generated_cubins.cubins) |cubin| {
        const text = std.mem.span(cubin.name);
        if (std.mem.eql(u8, text, name)) {
            assert(cubin.length > 0);
            return cubin.bytes[0..cubin.length];
        } else {
            continue;
        }
    }
    return error.MissingCubin;
}

/// Arguments of `rope_table_build`.
pub const RopeTableArgs = struct {
    cos: cuda_driver.DevicePtr,
    sin: cuda_driver.DevicePtr,
    /// One int32 `[rows]` position grid per mRoPE axis.
    grids: [3]cuda_driver.DevicePtr,
    /// int32 `[3]`: `mrope_section`.
    section: cuda_driver.DevicePtr,
    rows: u32,
    rotary_dim: u32,
    theta: f32,
};

/// Arguments of `rope_apply_rotary`.
pub const RopeApplyArgs = struct {
    /// bf16 `[rows, heads, head_dim]`, rotated in place.
    values: cuda_driver.DevicePtr,
    cos: cuda_driver.DevicePtr,
    sin: cuda_driver.DevicePtr,
    /// int32 `[rows]`.
    positions: cuda_driver.DevicePtr,
    rows: u32,
    heads: u32,
    head_dim: u32,
    rotary_dim: u32,
};

/// Arguments of the whole indexer pipeline.
pub const IndexerArgs = struct {
    /// bf16 `[rows, in_features]`.
    hidden: cuda_driver.DevicePtr,
    /// bf16 `[projectedWidth, in_features]`, the TP shard of `index_qk_proj`.
    weight: cuda_driver.DevicePtr,
    /// f32 `[rows, projectedWidth]` scratch.
    projected: cuda_driver.DevicePtr,
    /// f32 `[index_dim]` each.
    q_layernorm: cuda_driver.DevicePtr,
    k_layernorm: cuda_driver.DevicePtr,
    /// bf16 `[rows, index_heads, index_dim]` out.
    queries: cuda_driver.DevicePtr,
    /// bf16 `[arena_length, index_dim]` cache, appended in place.
    index_keys: cuda_driver.DevicePtr,
    /// bf16 `[arena_length / ratio, index_dim]` scratch.
    pooled: cuda_driver.DevicePtr,
    /// f32 `[rows, min(blocks, score_group_blocks)]` scratch: one streaming
    /// group's block scores, overwritten by the next group.
    scores: cuda_driver.DevicePtr,
    /// u64 `[rows, block_topk]` scratch: the running per-row top-k, packed as
    /// `(score bits, absolute block index)` and sorted descending.
    partial: cuda_driver.DevicePtr,
    /// int32 `[rows, slots]` out.
    selected: cuda_driver.DevicePtr,
    /// int32 `[rows]` absolute positions.
    positions: cuda_driver.DevicePtr,
    cos: cuda_driver.DevicePtr,
    sin: cuda_driver.DevicePtr,
    rows: u32,
    in_features: u32,
    arena_length: u32,
    /// First pooled block this launch computes: the engine pools a block once,
    /// when it completes, so old raw keys are never read again.
    first_block: u32,
    shapes: Shapes,
};

/// Arguments of `attn_prefill`.
///
/// The working set is the dense gather of the selected tokens; the kernel never
/// sees the master KV cache, which the design keeps partly in DRAM.
pub const PrefillArgs = struct {
    /// bf16 `[rows, heads, head_dim]`, already normed and rotated.
    queries: cuda_driver.DevicePtr,
    /// bf16 `[count, kv_heads, head_dim]`, gathered keys.
    gathered_k: cuda_driver.DevicePtr,
    /// bf16 `[count, kv_heads, head_dim]`, gathered values.
    gathered_v: cuda_driver.DevicePtr,
    /// int32 `[count]`: absolute token position of each gathered row.
    gather_index: cuda_driver.DevicePtr,
    /// int32 `[rows]`: absolute positions of the queries, ascending.
    ///
    /// A prefill chunk may share one union working set, so a gathered row can
    /// be past an earlier row of the chunk and is masked for it; the last row's
    /// position is the launch's causal horizon and the one `violations` counts
    /// against.
    positions: cuda_driver.DevicePtr,
    /// bf16 `[rows, heads, head_dim]` out.
    out: cuda_driver.DevicePtr,
    /// Optional int32 `[1]`: counts gathered rows past the launch's causal
    /// horizon, once each, so a nonzero value means the list was corrupt.
    violations: ?cuda_driver.DevicePtr,
    /// Optional u32 `[rows, ceil(count / 32)]`: per-row selection bitmap over
    /// the shared union gather. Bit `slot` set means this row's QSA selection
    /// contains the gathered row; the MMA path masks every cleared bit, which is
    /// what makes prefill exactly QSA at `rows > 1`. Null keeps causal-only
    /// masking (the union semantics).
    selection_mask: ?cuda_driver.DevicePtr = null,
    rows: u32,
    count: u32,
    shapes: Shapes,
};

/// Arguments of `attn_decode`.
pub const DecodeArgs = struct {
    /// bf16 `[heads, head_dim]`: the one new query token.
    query: cuda_driver.DevicePtr,
    /// bf16 `[count, kv_heads, head_dim]`, gathered keys.
    gathered_k: cuda_driver.DevicePtr,
    /// bf16 `[count, kv_heads, head_dim]`, gathered values.
    gathered_v: cuda_driver.DevicePtr,
    /// int32 `[count]`: absolute token position of each gathered row.
    gather_index: cuda_driver.DevicePtr,
    /// Absolute position of the query token, for the causal check.
    position: u32,
    /// bf16 `[heads, head_dim]` out.
    out: cuda_driver.DevicePtr,
    /// Optional int32 `[1]`: counts gathered rows past the query's position,
    /// once each; a nonzero value means the selection list was corrupt.
    violations: ?cuda_driver.DevicePtr,
    count: u32,
    shapes: Shapes,
};

/// The kernels plus the stream they are launched on.
pub const Attention = struct {
    kernels: Kernels,
    stream: cuda_context.Stream,

    /// Loads the cubin and creates the queue; the context must be current.
    pub fn init() !Attention {
        assert(cubin_name.len > 0);
        assert(generated_cubins.cubins.len > 0);

        const kernels = try Kernels.init();
        errdefer {
            var owned = kernels;
            owned.deinit();
        }
        const stream = try cuda_context.Stream.create();
        assert(@intFromPtr(stream.handle) != 0);
        return .{ .kernels = kernels, .stream = stream };
    }

    /// Releases the queue and the module.
    pub fn deinit(self: *Attention) void {
        assert(@intFromPtr(self.stream.handle) != 0);
        assert(@intFromPtr(self.kernels.module.handle) != 0);

        self.stream.destroy();
        self.kernels.deinit();
    }

    /// Fills the `[rows, rotary_dim]` cos/sin tables.
    pub fn buildRopeTables(
        self: *const Attention,
        args: RopeTableArgs,
    ) !void {
        assert(args.rows > 0);
        assert(args.rows <= positions_max);
        assert(args.rotary_dim > 0);
        assert(args.rotary_dim % 2 == 0);

        var cos = args.cos;
        var sin = args.sin;
        var temporal = args.grids[0];
        var height = args.grids[1];
        var width = args.grids[2];
        var section = args.section;
        var rows: u32 = args.rows;
        var rotary_dim: u32 = args.rotary_dim;
        var theta: f32 = args.theta;
        const threads = @min(rotaryPairs(args.rotary_dim), rope_table_threads);
        const arguments = [_]?*anyopaque{
            @ptrCast(&cos),     @ptrCast(&sin),  @ptrCast(&temporal),   @ptrCast(&height), @ptrCast(&width),
            @ptrCast(&section), @ptrCast(&rows), @ptrCast(&rotary_dim), @ptrCast(&theta),
        };
        try self.kernels.rope_table_build.launch(
            .{ args.rows, 1, 1 },
            .{ threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// Rotates the first `rotary_dim` dims of every head in place.
    pub fn applyRope(
        self: *const Attention,
        args: RopeApplyArgs,
    ) !void {
        assert(args.rows > 0);
        assert(args.heads > 0);
        assert(args.rotary_dim > 0);
        assert(args.rotary_dim % 2 == 0);

        var values = args.values;
        var cos = args.cos;
        var sin = args.sin;
        var positions = args.positions;
        var rows: u32 = args.rows;
        var heads: u32 = args.heads;
        var head_dim: u32 = args.head_dim;
        var rotary_dim: u32 = args.rotary_dim;
        const pairs = rotaryPairs(args.rotary_dim);
        const total: u64 = @as(u64, args.rows) * args.heads * pairs;
        const blocks: u32 = @intCast(
            std.math.divCeil(u64, total, rope_apply_threads) catch unreachable,
        );
        const arguments = [_]?*anyopaque{
            @ptrCast(&values), @ptrCast(&cos),      @ptrCast(&sin),        @ptrCast(&positions), @ptrCast(&rows),
            @ptrCast(&heads),  @ptrCast(&head_dim), @ptrCast(&rotary_dim),
        };
        try self.kernels.rope_apply_rotary.launch(
            .{ blocks, 1, 1 },
            .{ rope_apply_threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// Runs the six indexer stages: project, norm+rotate, append, pool, score,
    /// select.
    ///
    /// The block scores stream: each pass scores at most `score_group_blocks`
    /// blocks into the reusable `scores` buffer, folds that group's top-k into
    /// the running `partial` keys, and the last pass expands the partial keys
    /// into `selected`. The score buffer therefore never grows with the
    /// context.
    pub fn indexerScores(
        self: *const Attention,
        args: IndexerArgs,
    ) !void {
        assert(args.rows > 0);
        assert(args.arena_length > 0);
        assert(args.in_features > 0);
        assert(args.shapes.index_dim > 0);
        args.shapes.validate();

        const shapes = args.shapes;
        const width = shapes.projectedWidth();
        const blocks = args.arena_length / shapes.compress_ratio;
        assert(args.first_block <= blocks);
        try self.project(&args, width);
        try self.normQueries(&args);
        try self.appendKeys(&args);
        if (blocks > args.first_block) {
            try self.poolBlocks(&args, blocks - args.first_block);
        } else {
            // No block completed this step, so there is nothing new to pool: the
            // resident pooled keys already cover every complete block.
        }
        // The score buffer is a fixed group width, processed one row chunk at a
        // time: a full batch prefill therefore uses the same bytes as a decode
        // step. The partial top-k seeds the first group's merge, so 0 is the
        // padding key that sorts last.
        const capacity = score_group_blocks;
        assert(capacity > 0);
        assert(indexer_row_chunk > 0);
        var row_first: u32 = 0;
        while (row_first < args.rows) {
            const rows_local = @min(args.rows - row_first, indexer_row_chunk);
            var stage_args = args;
            stage_args.rows = rows_local;
            stage_args.queries += @as(u64, row_first) *
                shapes.index_heads * shapes.index_dim * 2;
            stage_args.positions += @as(u64, row_first) * 4;
            stage_args.partial += @as(u64, row_first) *
                shapes.blockTopk() * 8;
            stage_args.selected += @as(u64, row_first) * shapes.slots() * 4;
            if (blocks > 0) {
                var first: u32 = 0;
                while (first < blocks) {
                    const count = @min(blocks - first, score_group_blocks);
                    // The block scores read *this* step's queries, which change
                    // every step, so they are recomputed even when no new block
                    // completed. One bounded group per launch keeps the score
                    // buffer context-independent.
                    try self.scoreBlocks(&stage_args, first, count, capacity);
                    try self.groupTopk(&stage_args, first, count, capacity);
                    first += count;
                }
            } else {
                // An empty cache has no block to score; the expansion still
                // writes the trailing tokens.
            }
            try self.expandSelected(&stage_args);
            row_first += rows_local;
        }
        assert(args.rows > 0);
    }

    /// Bytes one indexer call's score buffer needs: a fixed row chunk by the
    /// fixed streaming group width, never the context's block count or the
    /// batch's row count.
    pub fn indexerScoreBytes(rows: u32, blocks: u32) u64 {
        assert(rows > 0);
        assert(score_group_blocks > 0);
        assert(blocks <= positions_max);

        const chunk_rows = @min(rows, indexer_row_chunk);
        return @as(u64, chunk_rows) * score_group_blocks * 4;
    }

    /// Bytes of the running per-row partial top-k buffer.
    pub fn indexerPartialBytes(rows: u32, shapes: Shapes) u64 {
        assert(rows > 0);
        assert(shapes.blockTopk() > 0);

        return @as(u64, rows) * shapes.blockTopk() * 8;
    }

    /// Stage one: `projected = hidden @ weight^T` in fp32.
    fn project(
        self: *const Attention,
        args: *const IndexerArgs,
        width: u32,
    ) !void {
        assert(args.rows > 0);
        assert(width > 0);

        var hidden = args.hidden;
        var weight = args.weight;
        var projected = args.projected;
        var rows: u32 = args.rows;
        var in_features: u32 = args.in_features;
        var out_features: u32 = width;
        const columns = (width + index_threads - 1) / index_threads;
        const arguments = [_]?*anyopaque{
            @ptrCast(&hidden), @ptrCast(&weight), @ptrCast(&projected), @ptrCast(&rows), @ptrCast(&in_features), @ptrCast(&out_features),
        };
        try self.kernels.indexer_project.launch(
            .{ args.rows, columns, 1 },
            .{ index_threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// Stage two: per-head RMSNorm plus partial RoPE on the index queries.
    fn normQueries(
        self: *const Attention,
        args: *const IndexerArgs,
    ) !void {
        assert(args.rows > 0);
        assert(args.shapes.index_heads > 0);

        const shapes = args.shapes;
        var projected = args.projected;
        var layernorm = args.q_layernorm;
        var queries = args.queries;
        var cos = args.cos;
        var sin = args.sin;
        var positions = args.positions;
        var rows: u32 = args.rows;
        var index_heads: u32 = shapes.index_heads;
        var index_kv_heads: u32 = shapes.index_kv_heads;
        var index_dim: u32 = shapes.index_dim;
        var rotary_dim: u32 = shapes.rotary_dim;
        var eps: f32 = shapes.eps;
        const arguments = [_]?*anyopaque{
            @ptrCast(&projected),  @ptrCast(&layernorm), @ptrCast(&queries),     @ptrCast(&cos),            @ptrCast(&sin),
            @ptrCast(&positions),  @ptrCast(&rows),      @ptrCast(&index_heads), @ptrCast(&index_kv_heads), @ptrCast(&index_dim),
            @ptrCast(&rotary_dim), @ptrCast(&eps),
        };
        try self.kernels.indexer_norm_rope_q.launch(
            .{ args.rows, shapes.index_heads, 1 },
            .{ index_threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// Stage three: append the step's raw token keys to the index key cache.
    fn appendKeys(
        self: *const Attention,
        args: *const IndexerArgs,
    ) !void {
        assert(args.rows > 0);
        assert(args.arena_length > 0);

        const shapes = args.shapes;
        var projected = args.projected;
        var index_keys = args.index_keys;
        var positions = args.positions;
        var rows: u32 = args.rows;
        var index_heads: u32 = shapes.index_heads;
        var index_kv_heads: u32 = shapes.index_kv_heads;
        var index_dim: u32 = shapes.index_dim;
        var arena_length: u32 = args.arena_length;
        const arguments = [_]?*anyopaque{
            @ptrCast(&projected),      @ptrCast(&index_keys), @ptrCast(&positions),    @ptrCast(&rows), @ptrCast(&index_heads),
            @ptrCast(&index_kv_heads), @ptrCast(&index_dim),  @ptrCast(&arena_length),
        };
        try self.kernels.indexer_append_key.launch(
            .{ args.rows, 1, 1 },
            .{ index_threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// Stage four: pool, normalise and rotate `blocks` blocks from `first_block`.
    ///
    /// The engine calls this once per newly completed block, so the raw keys of
    /// older tokens are never read again and need not be retained.
    fn poolBlocks(
        self: *const Attention,
        args: *const IndexerArgs,
        blocks: u32,
    ) !void {
        assert(blocks > 0);
        assert(args.shapes.compress_ratio > 0);

        const shapes = args.shapes;
        var index_keys = args.index_keys;
        var layernorm = args.k_layernorm;
        var pooled = args.pooled;
        var cos = args.cos;
        var sin = args.sin;
        var first_block: u32 = args.first_block;
        var block_count: u32 = blocks;
        var index_dim: u32 = shapes.index_dim;
        var ratio: u32 = shapes.compress_ratio;
        var rotary_dim: u32 = shapes.rotary_dim;
        var eps: f32 = shapes.eps;
        const arguments = [_]?*anyopaque{
            @ptrCast(&index_keys),  @ptrCast(&layernorm), @ptrCast(&pooled),
            @ptrCast(&cos),         @ptrCast(&sin),       @ptrCast(&first_block),
            @ptrCast(&block_count), @ptrCast(&index_dim), @ptrCast(&ratio),
            @ptrCast(&rotary_dim),  @ptrCast(&eps),
        };
        try self.kernels.indexer_pool_blocks.launch(
            .{ blocks, 1, 1 },
            .{ index_threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// Stage five: `sum_heads relu(q . pooled) / sqrt(index_dim)` per block,
    /// for one bounded streaming group.
    ///
    /// `first` is the group's first absolute block; `count` blocks are scored,
    /// stored at their offset inside the `capacity`-wide group buffer.
    fn scoreBlocks(
        self: *const Attention,
        args: *const IndexerArgs,
        first: u32,
        count: u32,
        capacity: u32,
    ) !void {
        assert(count > 0);
        assert(capacity >= count);
        assert(args.shapes.index_dim > 0);

        const shapes = args.shapes;
        var queries = args.queries;
        var pooled = args.pooled;
        var scores = args.scores;
        var rows: u32 = args.rows;
        var block_count: u32 = count;
        var first_block: u32 = first;
        var score_capacity: u32 = capacity;
        var index_heads: u32 = shapes.index_heads;
        var index_dim: u32 = shapes.index_dim;
        var scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(index_dim)));
        const chunks = (count + scores_per_cta - 1) / scores_per_cta;
        const arguments = [_]?*anyopaque{
            @ptrCast(&queries),     @ptrCast(&pooled),      @ptrCast(&scores),         @ptrCast(&rows),
            @ptrCast(&block_count), @ptrCast(&first_block), @ptrCast(&score_capacity), @ptrCast(&index_heads),
            @ptrCast(&index_dim),   @ptrCast(&scale),
        };
        try self.kernels.indexer_block_scores.launch(
            .{ chunks, args.rows, 1 },
            .{ shapes.index_heads * 32, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// Stage five and a half: merge one score group into the running top-k.
    fn groupTopk(
        self: *const Attention,
        args: *const IndexerArgs,
        first: u32,
        count: u32,
        capacity: u32,
    ) !void {
        assert(count > 0);
        assert(capacity >= count);
        assert(args.shapes.blockTopk() > 0);

        const shapes = args.shapes;
        var scores = args.scores;
        var positions = args.positions;
        var partial = args.partial;
        var rows: u32 = args.rows;
        var group_capacity: u32 = capacity;
        var first_block: u32 = first;
        var group_blocks: u32 = count;
        var ratio: u32 = shapes.compress_ratio;
        var block_topk: u32 = shapes.blockTopk();
        const arguments = [_]?*anyopaque{
            @ptrCast(&scores),       @ptrCast(&positions),      @ptrCast(&partial),
            @ptrCast(&rows),         @ptrCast(&group_capacity), @ptrCast(&first_block),
            @ptrCast(&group_blocks), @ptrCast(&ratio),          @ptrCast(&block_topk),
        };
        try self.kernels.indexer_group_topk.launch(
            .{ args.rows, 1, 1 },
            .{ select_threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// Stage six of the streaming path: expand the partial top-k to tokens.
    fn expandSelected(
        self: *const Attention,
        args: *const IndexerArgs,
    ) !void {
        assert(args.rows > 0);
        assert(args.shapes.blockTopk() > 0);

        const shapes = args.shapes;
        var partial = args.partial;
        var positions = args.positions;
        var selected = args.selected;
        var rows: u32 = args.rows;
        var slots: u32 = shapes.slots();
        var ratio: u32 = shapes.compress_ratio;
        var block_topk: u32 = shapes.blockTopk();
        const arguments = [_]?*anyopaque{
            @ptrCast(&partial),    @ptrCast(&positions), @ptrCast(&selected),
            @ptrCast(&rows),       @ptrCast(&slots),     @ptrCast(&ratio),
            @ptrCast(&block_topk),
        };
        try self.kernels.indexer_expand.launch(
            .{ args.rows, 1, 1 },
            .{ select_threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// Stage six: top-k blocks, expanded to tokens, plus the trailing tail.
    ///
    /// Takes the four buffers it reads and writes instead of the whole
    /// `IndexerArgs`, so the stage can be launched on its own: the tie-break
    /// contract of the top-k is checked with a synthetic score row.
    pub fn select(
        self: *const Attention,
        scores: cuda_driver.DevicePtr,
        positions: cuda_driver.DevicePtr,
        selected: cuda_driver.DevicePtr,
        rows: u32,
        blocks: u32,
        shapes: Shapes,
    ) !void {
        assert(rows > 0);
        assert(shapes.slots() > 0);
        assert(shapes.blockTopk() > 0);

        var scores_arg = scores;
        var positions_arg = positions;
        var selected_arg = selected;
        var rows_arg: u32 = rows;
        var block_count: u32 = blocks;
        var slots: u32 = shapes.slots();
        var ratio: u32 = shapes.compress_ratio;
        var block_topk: u32 = shapes.blockTopk();
        const arguments = [_]?*anyopaque{
            @ptrCast(&scores_arg), @ptrCast(&positions_arg), @ptrCast(&selected_arg),
            @ptrCast(&rows_arg),   @ptrCast(&block_count),   @ptrCast(&slots),
            @ptrCast(&ratio),      @ptrCast(&block_topk),
        };
        try self.kernels.indexer_select.launch(
            .{ rows, 1, 1 },
            .{ select_threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// Dense prefill attention over an explicitly gathered working set.
    ///
    /// The tensor-core path handles the model's head geometries; a shape it
    /// cannot tile falls back to the warp-scalar kernel, which is the same
    /// maths with the same masking.
    pub fn prefill(self: *const Attention, args: PrefillArgs) !void {
        assert(args.rows > 0);
        assert(args.count > 0);
        assert(args.count <= selected_slots_max);
        args.shapes.validate();

        if (prefillUsesMma(args.shapes, args.rows)) {
            try self.prefillMma(args);
        } else {
            try self.prefillScalar(args);
        }
    }

    /// The tensor-core prefill: one CTA per (head, 16 query rows).
    fn prefillMma(self: *const Attention, args: PrefillArgs) !void {
        assert(prefillUsesMma(args.shapes, args.rows));
        assert(prefill_mma_threads % 32 == 0);

        const shapes = args.shapes;
        var queries = args.queries;
        var gathered_k = args.gathered_k;
        var gathered_v = args.gathered_v;
        var gather_index = args.gather_index;
        var positions = args.positions;
        var selection_mask: cuda_driver.DevicePtr = args.selection_mask orelse 0;
        var out = args.out;
        var rows: u32 = args.rows;
        var heads: u32 = shapes.heads;
        var kv_heads: u32 = shapes.kv_heads;
        var head_dim: u32 = shapes.head_dim;
        var count: u32 = args.count;
        var scale: f32 = attentionScale(shapes);
        var violations: cuda_driver.DevicePtr = args.violations orelse 0;
        const arguments = [_]?*anyopaque{
            @ptrCast(&queries),
            @ptrCast(&gathered_k),
            @ptrCast(&gathered_v),
            @ptrCast(&gather_index),
            @ptrCast(&positions),
            @ptrCast(&selection_mask),
            @ptrCast(&out),
            @ptrCast(&rows),
            @ptrCast(&heads),
            @ptrCast(&kv_heads),
            @ptrCast(&head_dim),
            @ptrCast(&count),
            @ptrCast(&scale),
            @ptrCast(&violations),
        };
        const row_blocks = (args.rows + prefill_mma_rows - 1) / prefill_mma_rows;
        try self.kernels.attn_prefill_mma.launch(
            .{ shapes.heads, row_blocks, 1 },
            .{ prefill_mma_threads, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// The warp-scalar prefill: one warp per query head, one row per block.
    fn prefillScalar(self: *const Attention, args: PrefillArgs) !void {
        assert(!prefillUsesMma(args.shapes, args.rows));
        assert(prefill_warps > 0);

        const shapes = args.shapes;
        var queries = args.queries;
        var gathered_k = args.gathered_k;
        var gathered_v = args.gathered_v;
        var gather_index = args.gather_index;
        var positions = args.positions;
        var out = args.out;
        var rows: u32 = args.rows;
        var heads: u32 = shapes.heads;
        var kv_heads: u32 = shapes.kv_heads;
        var head_dim: u32 = shapes.head_dim;
        var count: u32 = args.count;
        var passes: u32 = shapes.lanePasses();
        var scale: f32 = attentionScale(shapes);
        var selection_mask: cuda_driver.DevicePtr = args.selection_mask orelse 0;
        var violations: cuda_driver.DevicePtr = args.violations orelse 0;
        const arguments = [_]?*anyopaque{
            @ptrCast(&queries),
            @ptrCast(&gathered_k),
            @ptrCast(&gathered_v),
            @ptrCast(&gather_index),
            @ptrCast(&positions),
            @ptrCast(&out),
            @ptrCast(&rows),
            @ptrCast(&heads),
            @ptrCast(&kv_heads),
            @ptrCast(&head_dim),
            @ptrCast(&count),
            @ptrCast(&passes),
            @ptrCast(&scale),
            @ptrCast(&selection_mask),
            @ptrCast(&violations),
        };
        const columns = (shapes.heads + prefill_warps - 1) / prefill_warps;
        try self.kernels.attn_prefill.launch(
            .{ columns, args.rows, 1 },
            .{ prefill_warps * 32, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }

    /// Dense single-token decode over the same kind of gathered working set.
    pub fn decode(self: *const Attention, args: DecodeArgs) !void {
        assert(args.count > 0);
        assert(args.count <= selected_slots_max);
        args.shapes.validate();

        const shapes = args.shapes;
        var query = args.query;
        var gathered_k = args.gathered_k;
        var gathered_v = args.gathered_v;
        var gather_index = args.gather_index;
        var count: u32 = args.count;
        var position: u32 = args.position;
        var out = args.out;
        var heads: u32 = shapes.heads;
        var kv_heads: u32 = shapes.kv_heads;
        var head_dim: u32 = shapes.head_dim;
        var passes: u32 = shapes.lanePasses();
        var scale: f32 = attentionScale(shapes);
        var violations: cuda_driver.DevicePtr = args.violations orelse 0;
        const arguments = [_]?*anyopaque{
            @ptrCast(&query),
            @ptrCast(&gathered_k),
            @ptrCast(&gathered_v),
            @ptrCast(&gather_index),
            @ptrCast(&count),
            @ptrCast(&position),
            @ptrCast(&out),
            @ptrCast(&heads),
            @ptrCast(&kv_heads),
            @ptrCast(&head_dim),
            @ptrCast(&passes),
            @ptrCast(&scale),
            @ptrCast(&violations),
        };
        try self.kernels.attn_decode.launch(
            .{ shapes.heads, 1, 1 },
            .{ decode_warps * 32, 1, 1 },
            0,
            self.stream.handle,
            arguments[0..],
        );
    }
};

/// `1 / sqrt(head_dim)`, the attention scaling of the reference.
pub fn attentionScale(shapes: Shapes) f32 {
    assert(shapes.head_dim > 0);
    assert(shapes.head_dim <= 1 << 20);

    return 1.0 / @sqrt(@as(f32, @floatFromInt(shapes.head_dim)));
}

/// Whether `prefill` can take the tensor-core path for `shapes` and `rows`: the
/// head dim must split into 32 dims per warp, the V tile must fit, and the
/// launch must be wide enough to fill the 16-row m-tile.
pub fn prefillUsesMma(shapes: Shapes, rows: u32) bool {
    assert(shapes.head_dim > 0);
    assert(shapes.kv_heads > 0);
    assert(rows > 0);

    return shapes.head_dim % 32 == 0 and
        shapes.head_dim <= prefill_mma_head_dim_max and
        rows >= prefill_mma_min_rows;
}

/// Host-side tripwire for the indexer's causal guarantee (issue I-008 §1a).
///
/// The attention kernels skip and count a gathered row that sits past the
/// launch's causal horizon, so a corrupt selection cannot read the future — but
/// it also cannot fail there, which is what this check is for: the host refuses
/// a selection list that reaches past the horizon. The horizon is the last
/// query position of the launch, which is also the query's own position when one
/// query is in flight; it is inclusive, because the indexer's visible set is
/// `position + 1` tokens.
pub fn checkSelectionWithinPosition(
    selected: []const i32,
    position: u32,
) error{FutureTokenSelected}!void {
    assert(selected.len > 0);
    assert(position < std.math.maxInt(i32));

    for (selected) |index| {
        if (index >= 0) {
            if (@as(u32, @intCast(index)) > position) {
                return error.FutureTokenSelected;
            } else {
                // A cached token at or before the query: what the indexer
                // promises.
            }
        } else {
            // `-1` padding: this slot holds no token.
        }
    }
    assert(selected.len > 0);
}

/// Rounds `bytes` up to the driver's allocation granularity.
pub fn roundToAllocation(bytes: u64) u64 {
    assert(bytes > 0);
    assert(allocation_granularity > 0);

    const rounded = (bytes + allocation_granularity - 1) /
        allocation_granularity * allocation_granularity;
    assert(rounded >= bytes);
    return rounded;
}

/// Replaces every value with its BF16 round trip, so a f32 oracle sees exactly
/// the numbers the kernels read back out of the arena.
pub fn roundTripBf16(values: []f32) void {
    assert(values.len > 0);
    assert(@sizeOf(u16) == 2);

    for (values) |*value| {
        value.* = quant.bf16_to_f32(quant.f32_to_bf16(value.*));
    }
}

// ---------------------------------------------------------------------------
// Harness: device buffers, the fp32 oracles, and the fixture readers.
//
// Everything below runs only from `zig build test`. It is written for clarity
// over speed: the buffers are allocated per case and freed with the harness.
// ---------------------------------------------------------------------------

/// Largest single transfer the harness will attempt; a full-context arena is
/// half a gigabyte, so this is a sanity bound, not a staging allocation.
const staging_max_bytes: u64 = 4 * 1024 * 1024 * 1024;

/// Pinned staging window; transfers loop over it in chunks so a full-context
/// arena never pins hundreds of megabytes.
const staging_chunk_bytes: u64 = 32 * 1024 * 1024;

/// A GPU to run on: one context, one queue, one loaded cubin.
const Harness = struct {
    allocator: std.mem.Allocator,
    devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined,
    context: cuda_context.Context = undefined,
    attention: Attention = undefined,
    allocations: std.ArrayList(cuda_context.DeviceBuffer),

    /// Opens device 0 and loads the kernels.
    ///
    /// A host without the CUDA driver skips the test instead of failing it,
    /// which is how the rest of the repository treats a GPU-less machine.
    fn open(allocator: std.mem.Allocator) !Harness {
        assert(allocator.vtable != undefined);
        assert(cuda_driver.device_count_max > 0);

        var harness = Harness{
            .allocator = allocator,
            .allocations = std.ArrayList(cuda_context.DeviceBuffer).init(allocator),
        };
        const count = cuda_driver.query_devices(harness.devices[0..]) catch |err| switch (err) {
            error.driver_library_missing => return error.SkipZigTest,
            else => return err,
        };
        assert(count > 0);
        harness.context = try cuda_context.Context.retain(&harness.devices[0]);
        errdefer harness.context.release();
        try harness.context.set_current();
        harness.attention = try Attention.init();
        assert(harness.attention.stream.handle != undefined);
        return harness;
    }

    fn deinit(self: *Harness) void {
        assert(self.attention.stream.handle != undefined);
        assert(@intFromPtr(self.context.handle) != 0);

        for (self.allocations.items) |buffer| {
            buffer.free();
        }
        self.allocations.deinit();
        self.attention.deinit();
        self.context.release();
    }

    /// Copies `bytes` to a device buffer and tracks it for cleanup.
    ///
    /// The transfer loops over a bounded pinned window, so an arena of several
    /// hundred megabytes costs one small staging buffer instead of one huge
    /// pinned allocation.
    fn upload(self: *Harness, bytes: []const u8) !cuda_driver.DevicePtr {
        assert(bytes.len > 0);
        assert(bytes.len <= staging_max_bytes);
        assert(staging_chunk_bytes > 0);

        const rounded: u64 = @intCast(roundToAllocation(@intCast(bytes.len)));
        const buffer = try cuda_context.DeviceBuffer.allocate(rounded);
        errdefer buffer.free();
        const window: u64 = @min(rounded, staging_chunk_bytes);
        const staged = try cuda_context.PinnedHostBuffer.allocate(window);
        defer staged.free();
        var written: u64 = 0;
        while (written < rounded) {
            const size: usize = @intCast(@min(window, rounded - written));
            const view = staged.slice()[0..size];
            @memset(view, 0);
            const available = @min(size, bytes.len - @as(usize, @intCast(written)));
            @memcpy(view[0..available], bytes[@intCast(written)..][0..available]);
            try self.attention.stream.copy_h_to_d(
                &.{ .pointer = buffer.pointer + written, .bytes = size },
                &staged,
                size,
            );
            // The next chunk reuses this pinned window, so the copy that still
            // owns it has to land first.
            try self.attention.stream.synchronize();
            written += size;
        }
        try self.allocations.append(buffer);
        assert(buffer.pointer != 0);
        return buffer.pointer;
    }

    /// Copies a device buffer back into `out`, in the same bounded window.
    fn download(
        self: *Harness,
        pointer: cuda_driver.DevicePtr,
        out: []u8,
    ) !void {
        assert(out.len > 0);
        assert(pointer != 0);
        assert(staging_chunk_bytes > 0);

        const window: u64 = @min(
            @as(u64, @intCast(roundToAllocation(@intCast(out.len)))),
            staging_chunk_bytes,
        );
        const staged = try cuda_context.PinnedHostBuffer.allocate(window);
        defer staged.free();
        var read: usize = 0;
        while (read < out.len) {
            const size: usize = @min(@as(usize, @intCast(window)), out.len - read);
            try self.attention.stream.copy_d_to_h(
                &staged,
                &.{ .pointer = pointer + read, .bytes = size },
                size,
            );
            try self.attention.stream.synchronize();
            @memcpy(out[read..][0..size], staged.slice()[0..size]);
            read += size;
        }
        assert(read == out.len);
    }

    fn uploadF32(self: *Harness, values: []const f32) !cuda_driver.DevicePtr {
        assert(values.len > 0);
        assert(@sizeOf(f32) == 4);

        return self.upload(std.mem.sliceAsBytes(values));
    }

    fn uploadBf16(self: *Harness, values: []const f32) !cuda_driver.DevicePtr {
        assert(values.len > 0);
        assert(values.len <= 1 << 28);

        const encoded = try self.allocator.alloc(u16, values.len);
        defer self.allocator.free(encoded);
        for (values, 0..) |value, index| {
            encoded[index] = quant.f32_to_bf16(value);
        }
        return self.upload(std.mem.sliceAsBytes(encoded));
    }

    fn uploadI32(self: *Harness, values: []const i32) !cuda_driver.DevicePtr {
        assert(values.len > 0);
        assert(@sizeOf(i32) == 4);

        return self.upload(std.mem.sliceAsBytes(values));
    }

    fn downloadF32(self: *Harness, pointer: cuda_driver.DevicePtr, out: []f32) !void {
        assert(out.len > 0);
        assert(pointer != 0);

        try self.download(pointer, std.mem.sliceAsBytes(out));
    }

    fn downloadBf16(self: *Harness, pointer: cuda_driver.DevicePtr, out: []f32) !void {
        assert(out.len > 0);
        assert(pointer != 0);

        const encoded = try self.allocator.alloc(u16, out.len);
        defer self.allocator.free(encoded);
        try self.download(pointer, std.mem.sliceAsBytes(encoded));
        for (encoded, 0..) |value, index| {
            out[index] = quant.bf16_to_f32(value);
        }
    }

    fn downloadI32(self: *Harness, pointer: cuda_driver.DevicePtr, out: []i32) !void {
        assert(out.len > 0);
        assert(pointer != 0);

        try self.download(pointer, std.mem.sliceAsBytes(out));
    }

    /// An uninitialised device buffer for scratch or kernel output.
    fn allocate(self: *Harness, bytes: u64) !cuda_driver.DevicePtr {
        assert(bytes > 0);
        assert(bytes <= staging_max_bytes);

        const buffer = try cuda_context.DeviceBuffer.allocate(roundToAllocation(bytes));
        try self.allocations.append(buffer);
        assert(buffer.pointer != 0);
        return buffer.pointer;
    }
};

// ---------------------------------------------------------------------------
// Fixture configuration.
// ---------------------------------------------------------------------------

/// The shapes one `RefConfig` runs at.
///
/// `config.rotary_dim()` is `int(head_dim * partial_rotary_factor)` and the
/// reference rotates both the attention heads and the indexer with it, so the
/// indexer's rotary width follows `head_dim`, not `index_dim`.
fn shapesFromConfig(config: reference_fixtures.RefConfig) Shapes {
    assert(config.head_dim > 0);
    assert(config.num_attention_heads > 0);
    assert(config.indexer_head_dim > 0);

    return .{
        .heads = config.num_attention_heads,
        .kv_heads = config.num_key_value_heads,
        .head_dim = config.head_dim,
        .index_heads = config.indexer_n_heads,
        .index_kv_heads = config.indexer_kv_heads,
        .index_dim = config.indexer_head_dim,
        .compress_ratio = config.indexer_compress_ratio,
        .budget = config.indexer_budget,
        .rotary_dim = config.rotary_dim(),
        .section = config.mrope_section,
        .eps = config.rms_norm_eps,
        .theta = config.rope_theta,
    };
}

/// A `RefConfig` holding the real checkpoint's indexed-attention numbers.
///
/// The real `config.json` is a 132 GB checkpoint away, so the oracle runs on a
/// config literal; the drift guard below pins it to `Shapes.real`.
fn realConfig(hidden_size: u32, positions: u32) reference_fixtures.RefConfig {
    assert(hidden_size > 0);
    assert(positions > 0);

    var config = reference_fixtures.RefConfig{};
    config.hidden_size = hidden_size;
    config.max_position_embeddings = positions;
    config.num_attention_heads = 24;
    config.num_key_value_heads = 2;
    config.head_dim = 256;
    config.partial_rotary_factor = 0.25;
    config.rope_theta = 10_000_000.0;
    config.mrope_section = .{ 11, 11, 10 };
    config.rms_norm_eps = 1e-6;
    config.indexer_budget = 2048;
    config.indexer_compress_ratio = 4;
    config.indexer_head_dim = 128;
    config.indexer_kv_heads = 1;
    config.indexer_n_heads = 4;
    return config;
}

// ---------------------------------------------------------------------------
// fp32 oracles.
// ---------------------------------------------------------------------------

/// Prints the error of `actual` against `expected` and enforces `limit`.
fn reportError(
    label: []const u8,
    actual: []const f32,
    expected: []const f32,
    limit: f32,
) !void {
    assert(label.len > 0);
    assert(actual.len == expected.len);
    assert(actual.len > 0);
    assert(limit > 0.0);

    var largest: f32 = 0.0;
    var magnitude: f32 = 0.0;
    for (actual, expected) |got, want| {
        largest = @max(largest, @abs(got - want));
        magnitude = @max(magnitude, @abs(want));
    }
    const relative = if (magnitude > 0.0) largest / magnitude else largest;
    std.debug.print(
        "{s:<46} max_abs_err {e:.3} max|ref| {e:.3} rel {e:.3}\n",
        .{ label, largest, magnitude, relative },
    );
    try std.testing.expect(relative <= limit);
}

/// Prints the relative difference between two float rows without asserting.
///
/// Used where one side is deliberately stored in a narrower dtype: the number
/// documents the size of the cast, it is not a gate.
fn printRelativeDifference(
    label: []const u8,
    actual: []const f32,
    expected: []const f32,
) void {
    assert(label.len > 0);
    assert(actual.len == expected.len);
    assert(actual.len > 0);

    var largest: f32 = 0.0;
    var magnitude: f32 = 0.0;
    for (actual, expected) |got, want| {
        largest = @max(largest, @abs(got - want));
        magnitude = @max(magnitude, @abs(want));
    }
    const relative = if (magnitude > 0.0) largest / magnitude else largest;
    std.debug.print(
        "{s:<46} max_abs_err {e:.3} max|ref| {e:.3} rel {e:.3}\n",
        .{ label, largest, magnitude, relative },
    );
    assert(relative >= 0.0);
}

/// fp32 reference for the dense gathered attention of every query row.
fn attendOracle(
    allocator: std.mem.Allocator,
    queries: []const f32,
    gathered_k: []const f32,
    gathered_v: []const f32,
    gather_index: []const i32,
    positions: []const u32,
    out: []f32,
    shapes: Shapes,
) !void {
    assert(queries.len % shapes.heads == 0);
    assert(gathered_k.len == gathered_v.len);
    assert(out.len == queries.len);

    const scale = attentionScale(shapes);
    const weights = try allocator.alloc(f32, gather_index.len);
    defer allocator.free(weights);
    for (positions, 0..) |position, row| {
        for (0..shapes.heads) |head| {
            const query = queries[(row * shapes.heads + head) * shapes.head_dim ..][0..shapes.head_dim];
            const target = out[(row * shapes.heads + head) * shapes.head_dim ..][0..shapes.head_dim];
            attendRow(
                query,
                gathered_k,
                gathered_v,
                gather_index,
                @intCast(head / shapes.group()),
                shapes,
                position,
                weights,
                target,
                scale,
            );
        }
    }
}

/// One head's masked softmax over the gathered working set, in fp32.
fn attendRow(
    query: []const f32,
    gathered_k: []const f32,
    gathered_v: []const f32,
    gather_index: []const i32,
    kv_head: u32,
    shapes: Shapes,
    position: u32,
    weights: []f32,
    target: []f32,
    scale: f32,
) void {
    assert(query.len == shapes.head_dim);
    assert(weights.len == gather_index.len);
    assert(target.len == shapes.head_dim);

    const kv_dim = shapes.kv_heads * shapes.head_dim;
    const head_offset = kv_head * shapes.head_dim;
    for (gather_index, 0..) |index, slot| {
        if (index < 0 or @as(u32, @intCast(index)) > position) {
            weights[slot] = -std.math.inf(f32);
            continue;
        }
        const base = slot * kv_dim + head_offset;
        weights[slot] = ops.dot(query, gathered_k[base..][0..shapes.head_dim]) *
            scale;
    }
    ops.softmax(weights);
    @memset(target, 0.0);
    for (gather_index, 0..) |index, slot| {
        if (index < 0 or @as(u32, @intCast(index)) > position) {
            continue;
        }
        const base = slot * kv_dim + head_offset;
        const value = gathered_v[base..][0..shapes.head_dim];
        for (value, 0..) |component, dim| {
            target[dim] += weights[slot] * component;
        }
    }
}

/// The reference top-k plus expansion, transcribed from the oracle's
/// `select_blocks` and `top_blocks`: scores descending, ties to the higher
/// block index, then the trailing incomplete block's raw indices.
fn selectOracle(
    scores: []const f32,
    visible: u32,
    block_topk: u32,
    ratio: u32,
    out: []i32,
    taken: []bool,
) void {
    assert(out.len > 0);
    assert(ratio > 0);
    assert(visible > 0);

    @memset(out, -1);
    const complete = visible / ratio;
    const tail = visible - complete * ratio;
    const take = @min(complete, block_topk);
    assert(scores.len >= complete);
    // Scratch instead of a fixed array: a full-context query has 65 536 blocks.
    assert(taken.len >= complete);
    @memset(taken[0..complete], false);
    var written: usize = 0;
    assert(written == 0);
    for (0..take) |_| {
        var best: usize = 0;
        var best_score = -std.math.inf(f32);
        for (scores[0..complete], 0..) |score, index| {
            if (taken[index]) {
                continue;
            }
            if (score >= best_score) {
                best_score = score;
                best = index;
            }
        }
        taken[best] = true;
        for (0..ratio) |item| {
            out[written] = @intCast(best * ratio + item);
            written += 1;
        }
    }
    for (0..tail) |item| {
        out[written] = @intCast(complete * ratio + item);
        written += 1;
    }
    assert(written <= out.len);
}

/// f32 reference of one index key pool: mean, cast, norm, rotate.
fn poolOracle(
    pooled: []f32,
    index_keys: []const f32,
    k_layernorm: []const f32,
    cos: []const f32,
    sin: []const f32,
    block: u32,
    shapes: Shapes,
    row: []f32,
) void {
    assert(pooled.len == shapes.index_dim);
    assert(k_layernorm.len == shapes.index_dim);
    assert(row.len == shapes.index_dim);

    const index_dim = shapes.index_dim;
    for (0..index_dim) |dim| {
        var total: f32 = 0.0;
        for (0..shapes.compress_ratio) |item| {
            const source = (block * shapes.compress_ratio + item) * index_dim;
            total += index_keys[source + dim];
        }
        pooled[dim] = quant.bf16_to_f32(
            quant.f32_to_bf16(total / @as(f32, @floatFromInt(shapes.compress_ratio))),
        );
    }
    ops.rmsnorm(pooled, k_layernorm, row, index_dim, shapes.eps);
    const position = block * shapes.compress_ratio;
    const start = @as(usize, position) * shapes.rotary_dim;
    reference_rope.apply(
        row,
        cos[start..][0..shapes.rotary_dim],
        sin[start..][0..shapes.rotary_dim],
        pooled,
    );
    roundTripBf16(pooled);
}

/// The indexer's block scores for one row, in fp32, from the same bf16 values
/// the kernels read.
fn scoreOracle(
    scores: []f32,
    queries: []const f32,
    pooled: []const f32,
    row: u32,
    blocks: u32,
    shapes: Shapes,
) void {
    assert(scores.len >= blocks);
    assert(queries.len >= (row + 1) * shapes.index_heads * shapes.index_dim);
    assert(pooled.len >= blocks * shapes.index_dim);

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(shapes.index_dim)));
    for (0..blocks) |block| {
        const key = pooled[block * shapes.index_dim ..][0..shapes.index_dim];
        var total: f32 = 0.0;
        for (0..shapes.index_heads) |head| {
            const offset = (row * shapes.index_heads + head) * shapes.index_dim;
            const dot = ops.dot(queries[offset..][0..shapes.index_dim], key);
            total += if (dot > 0.0) dot else 0.0;
        }
        scores[block] = total * scale;
    }
}

// ---------------------------------------------------------------------------
// Test support.
// ---------------------------------------------------------------------------

/// Deterministic values in [-1, 1).
fn randomValues(
    allocator: std.mem.Allocator,
    count: usize,
    seed: u64,
) ![]f32 {
    assert(count > 0);
    assert(seed > 0);

    const values = try allocator.alloc(f32, count);
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (values) |*value| {
        value.* = random.float(f32) * 2.0 - 1.0;
    }
    return values;
}

/// Scales `values` in place, so pooled keys and scores stay in a sane range.
fn scaleValues(values: []f32, factor: f32) void {
    assert(values.len > 0);
    assert(factor > 0.0);

    for (values) |*value| {
        value.* *= factor;
    }
}

/// Builds the rope tables on the GPU for `rows` positions and returns them on
/// the host; the three grids are offset so the mRoPE interleave is exercised.
fn buildTables(
    harness: *Harness,
    shapes: Shapes,
    rows: u32,
    offsets: [3]u32,
    cos: []f32,
    sin: []f32,
) !void {
    assert(rows > 0);
    assert(cos.len == rows * shapes.rotary_dim);
    assert(sin.len == cos.len);

    const allocator = harness.allocator;
    const grids = try allocator.alloc(u32, rows * 3);
    defer allocator.free(grids);
    const sections = try allocator.alloc(i32, 3);
    defer allocator.free(sections);
    const encoded = try allocator.alloc(i32, rows * 3);
    defer allocator.free(encoded);
    for (0..rows) |row| {
        grids[row] = @intCast(row + offsets[0]);
        grids[rows + row] = @intCast(row + offsets[1]);
        grids[2 * rows + row] = @intCast(row + offsets[2]);
    }
    for (grids, 0..) |value, index| {
        encoded[index] = @intCast(value);
    }
    for (shapes.section, 0..) |width, index| {
        sections[index] = @intCast(width);
    }
    const device_grids = try harness.uploadI32(encoded);
    const device_section = try harness.uploadI32(sections);
    const stride: u64 = @as(u64, rows) * @sizeOf(i32);
    const device_cos = try harness.allocate(@as(u64, rows) * shapes.rotary_dim * 4);
    const device_sin = try harness.allocate(@as(u64, rows) * shapes.rotary_dim * 4);
    try harness.attention.buildRopeTables(.{
        .cos = device_cos,
        .sin = device_sin,
        .grids = .{ device_grids, device_grids + stride, device_grids + 2 * stride },
        .section = device_section,
        .rows = rows,
        .rotary_dim = shapes.rotary_dim,
        .theta = shapes.theta,
    });
    try harness.downloadF32(device_cos, cos);
    try harness.downloadF32(device_sin, sin);
    assert(cos.len == rows * shapes.rotary_dim);
}

/// The tables `src/ref/rope.zig` computes for the same grids `buildTables`
/// uses.
fn referenceTables(
    allocator: std.mem.Allocator,
    shapes: Shapes,
    rows: u32,
    offsets: [3]u32,
    cos: []f32,
    sin: []f32,
) !void {
    assert(rows > 0);
    assert(cos.len == rows * shapes.rotary_dim);
    assert(sin.len == cos.len);

    const pairs = rotaryPairs(shapes.rotary_dim);
    const frequencies = try allocator.alloc(f32, pairs);
    defer allocator.free(frequencies);
    const temporal = try allocator.alloc(u32, rows);
    defer allocator.free(temporal);
    const height = try allocator.alloc(u32, rows);
    defer allocator.free(height);
    const width = try allocator.alloc(u32, rows);
    defer allocator.free(width);
    for (0..rows) |row| {
        temporal[row] = @intCast(row + offsets[0]);
        height[row] = @intCast(row + offsets[1]);
        width[row] = @intCast(row + offsets[2]);
    }
    reference_rope.inv_freq_init(shapes.theta, frequencies);
    reference_rope.cos_sin(
        .{ temporal, height, width },
        frequencies,
        shapes.section,
        cos,
        sin,
    );
    assert(frequencies[0] == 1.0);
}

/// Checks the rope tables and their application for one shape set.
fn checkRope(harness: *Harness, shapes: Shapes, rows: u32) !void {
    assert(rows > 0);
    assert(shapes.rotary_dim > 0);

    const allocator = harness.allocator;
    const rotary = shapes.rotary_dim;
    shapes.validate();
    const actual_cos = try allocator.alloc(f32, rows * rotary);
    defer allocator.free(actual_cos);
    const actual_sin = try allocator.alloc(f32, rows * rotary);
    defer allocator.free(actual_sin);
    try buildTables(harness, shapes, rows, .{ 0, 7, 13 }, actual_cos, actual_sin);
    const cos = try allocator.alloc(f32, rows * rotary);
    defer allocator.free(cos);
    const sin = try allocator.alloc(f32, rows * rotary);
    defer allocator.free(sin);
    try referenceTables(allocator, shapes, rows, .{ 0, 7, 13 }, cos, sin);
    try reportError("rope cos table", actual_cos, cos, 1.0e-4);
    try reportError("rope sin table", actual_sin, sin, 1.0e-4);

    const count = @as(usize, rows) * shapes.heads * shapes.head_dim;
    const values = try randomValues(allocator, count, 0x51ce_0001);
    defer allocator.free(values);
    roundTripBf16(values);
    const device_values = try harness.uploadBf16(values);
    const position_ids = try allocator.alloc(i32, rows);
    defer allocator.free(position_ids);
    for (position_ids, 0..) |*slot, row| {
        slot.* = @intCast(row);
    }
    const device_positions = try harness.uploadI32(position_ids);
    const device_cos = try harness.uploadF32(actual_cos);
    const device_sin = try harness.uploadF32(actual_sin);
    try harness.attention.applyRope(.{
        .values = device_values,
        .cos = device_cos,
        .sin = device_sin,
        .positions = device_positions,
        .rows = rows,
        .heads = shapes.heads,
        .head_dim = shapes.head_dim,
        .rotary_dim = rotary,
    });
    const applied = try allocator.alloc(f32, count);
    defer allocator.free(applied);
    try harness.downloadBf16(device_values, applied);

    const expected = try allocator.alloc(f32, count);
    defer allocator.free(expected);
    @memcpy(expected, values);
    for (0..rows) |row| {
        for (0..shapes.heads) |head| {
            const offset = (row * shapes.heads + head) * shapes.head_dim;
            reference_rope.apply(
                values[offset..][0..shapes.head_dim],
                actual_cos[row * rotary ..][0..rotary],
                actual_sin[row * rotary ..][0..rotary],
                expected[offset..][0..shapes.head_dim],
            );
            roundTripBf16(expected[offset..][0..shapes.head_dim]);
        }
    }
    try reportError("rope apply (bf16 storage)", applied, expected, 5.0e-3);
    assert(applied.len == expected.len);
}

/// One attention case: a block of queries plus the dense gathered working set
/// the kernels consume.
///
/// The master KV cache is not part of the contract: the design keeps a hot
/// window in VRAM and gathers the DRAM tail on demand, so a case is just the
/// gather output plus the absolute token position of each gathered row.
const AttentionCase = struct {
    /// bf16-rounded `[rows, heads, head_dim]`.
    queries: []f32,
    /// bf16-rounded `[count, kv_heads, head_dim]` each.
    gathered_k: []f32,
    gathered_v: []f32,
    /// `[count]`: absolute token position of each gathered row, -1 if empty.
    gather_index: []i32,
    /// `[rows]`: absolute position of each query.
    positions: []u32,
    /// `[rows, heads, head_dim]`: the fp32 oracle's expectation.
    out: []f32,

    fn rows(self: AttentionCase) u32 {
        assert(self.positions.len > 0);
        assert(self.queries.len % self.positions.len == 0);
        return @intCast(self.positions.len);
    }

    fn count(self: AttentionCase) u32 {
        assert(self.gather_index.len > 0);
        assert(self.gathered_k.len == self.gathered_v.len);
        return @intCast(self.gather_index.len);
    }
};

/// Allocates and fills one attention case.
///
/// `last_position` is the absolute position of the last query; the gather index
/// spans the visible prefix up to it, so the causal check is exercised for the
/// earlier rows of a chunk.
fn makeAttentionCase(
    allocator: std.mem.Allocator,
    shapes: Shapes,
    count: u32,
    rows: u32,
    last_position: u32,
    seed: u64,
) !AttentionCase {
    assert(count > 0);
    assert(rows > 0);
    assert(last_position + 1 >= rows);

    const kv_dim = @as(usize, shapes.kv_heads) * shapes.head_dim;
    const queries = try randomValues(
        allocator,
        @as(usize, rows) * shapes.heads * shapes.head_dim,
        seed,
    );
    errdefer allocator.free(queries);
    const gathered_k = try randomValues(
        allocator,
        @as(usize, count) * kv_dim,
        seed + 1,
    );
    errdefer allocator.free(gathered_k);
    const gathered_v = try randomValues(
        allocator,
        @as(usize, count) * kv_dim,
        seed + 2,
    );
    errdefer allocator.free(gathered_v);
    const gather_index = try allocator.alloc(i32, count);
    errdefer allocator.free(gather_index);
    const positions = try allocator.alloc(u32, rows);
    errdefer allocator.free(positions);
    const out = try allocator.alloc(f32, queries.len);
    errdefer allocator.free(out);
    roundTripBf16(queries);
    roundTripBf16(gathered_k);
    roundTripBf16(gathered_v);
    @memset(out, 0.0);
    const visible = @as(u64, last_position) + 1;
    for (gather_index, 0..) |*index, slot| {
        index.* = @intCast((@as(u64, slot) * visible) / count);
    }
    for (positions, 0..) |*position, row| {
        // `last_position + 1 - rows` first: the chunk may start at position 0.
        position.* = last_position + 1 - rows + @as(u32, @intCast(row));
    }
    assert(positions[rows - 1] == last_position);
    assert(gather_index[count - 1] <= last_position);
    return .{
        .queries = queries,
        .gathered_k = gathered_k,
        .gathered_v = gathered_v,
        .gather_index = gather_index,
        .positions = positions,
        .out = out,
    };
}

/// Frees one attention case.
fn freeAttentionCase(case: AttentionCase, allocator: std.mem.Allocator) void {
    assert(case.queries.len > 0);
    assert(case.gathered_k.len > 0);

    allocator.free(case.out);
    allocator.free(case.positions);
    allocator.free(case.gather_index);
    allocator.free(case.gathered_v);
    allocator.free(case.gathered_k);
    allocator.free(case.queries);
}

/// The host buffers the indexer pipeline consumes, shared by the GPU run and
/// the fp32 oracle so both see exactly the same numbers.
const IndexerInput = struct {
    shapes: Shapes,
    rows: u32,
    in_features: u32,
    arena_length: u32,
    /// bf16-rounded `[rows, in_features]`.
    hidden: []const f32,
    /// bf16-rounded `[projectedWidth, in_features]`.
    weight: []const f32,
    q_layernorm: []const f32,
    k_layernorm: []const f32,
    /// bf16-rounded `[arena_length, index_dim]`.
    index_keys: []const f32,
    positions: []const u32,
    /// Cos/sin tables, `[arena_length, rotary_dim]` each.
    cos: []const f32,
    sin: []const f32,
};

/// Runs the six indexer stages on the GPU and downloads the selection.
fn runIndexer(
    harness: *Harness,
    input: IndexerInput,
    selected: []i32,
    scores_out: ?[]f32,
    args_out: ?*IndexerArgs,
    probe_pooled: ?[]f32,
    probe_queries: ?[]f32,
) !void {
    assert(input.rows > 0);
    assert(input.arena_length > 0);
    assert(selected.len == @as(usize, input.rows) * input.shapes.slots());

    const shapes = input.shapes;
    const allocator = harness.allocator;
    const width = shapes.projectedWidth();
    const blocks = input.arena_length / shapes.compress_ratio;
    const encoded = try allocator.alloc(i32, input.rows);
    defer allocator.free(encoded);
    for (input.positions, 0..) |position, row| {
        encoded[row] = @intCast(position);
    }
    const projected = try harness.allocate(@as(u64, input.rows) * width * 4);
    const pooled = try harness.allocate(
        @as(u64, blocks) * shapes.index_dim * 2 + 4,
    );
    const scores = try harness.allocate(
        Attention.indexerScoreBytes(input.rows, blocks) + 4,
    );
    const partial = try harness.allocate(
        Attention.indexerPartialBytes(input.rows, shapes),
    );
    const queries_pointer = try harness.allocate(
        @as(u64, input.rows) * shapes.index_heads * shapes.index_dim * 2,
    );
    const selected_device = try harness.allocate(
        @as(u64, selected.len) * 4,
    );
    const args = IndexerArgs{
        .hidden = try harness.uploadBf16(input.hidden),
        .weight = try harness.uploadBf16(input.weight),
        .projected = projected,
        .q_layernorm = try harness.uploadF32(input.q_layernorm),
        .k_layernorm = try harness.uploadF32(input.k_layernorm),
        .queries = queries_pointer,
        .index_keys = try harness.uploadBf16(input.index_keys),
        .pooled = pooled,
        .scores = scores,
        .partial = partial,
        .selected = selected_device,
        .positions = try harness.uploadI32(encoded),
        .cos = try harness.uploadF32(input.cos),
        .sin = try harness.uploadF32(input.sin),
        .rows = input.rows,
        .in_features = input.in_features,
        .arena_length = input.arena_length,
        .first_block = 0,
        .shapes = shapes,
    };
    try harness.attention.indexerScores(args);
    try harness.downloadI32(selected_device, selected);
    if (probe_pooled) |out| {
        try harness.downloadBf16(args.pooled, out);
    }
    if (probe_queries) |out| {
        try harness.downloadBf16(args.queries, out);
    }
    if (args_out) |out| {
        out.* = args;
    }
    if (scores_out) |out| {
        // The device score buffer is one fixed group wide, so a row's scores
        // are `score_group_blocks` apart even when the last group is shorter.
        const capacity = score_group_blocks;
        const first = if (blocks == 0) 0 else ((blocks - 1) / capacity) * capacity;
        const used = blocks - first;
        assert(out.len == @as(usize, input.rows) * used);
        for (0..input.rows) |row| {
            try harness.downloadF32(
                scores + @as(u64, row) * capacity * 4,
                out[row * used ..][0..used],
            );
        }
    }
}

/// The device buffers of a resident indexer cache, kept across steps.
///
/// `runIndexer` allocates and discards them per call, which is what a one-shot
/// comparison wants. A decode loop instead keeps the key cache, the pooled
/// blocks and the scratch alive and launches the six stages again on every
/// step, so that is what this type models: `open` uploads the step-invariant
/// buffers, and `step` runs one step's rows against them.
const IndexerSession = struct {
    harness: *Harness,
    shapes: Shapes,
    in_features: u32,
    /// bf16 `[arena_length, index_dim]`, appended in place by every step.
    index_keys: cuda_driver.DevicePtr,
    weight: cuda_driver.DevicePtr,
    q_layernorm: cuda_driver.DevicePtr,
    k_layernorm: cuda_driver.DevicePtr,
    cos: cuda_driver.DevicePtr,
    sin: cuda_driver.DevicePtr,
    projected: cuda_driver.DevicePtr,
    pooled: cuda_driver.DevicePtr,
    scores: cuda_driver.DevicePtr,
    partial: cuda_driver.DevicePtr,
    queries: cuda_driver.DevicePtr,
    selected: cuda_driver.DevicePtr,

    /// Uploads the step-invariant buffers and sizes the scratch for a step of at
    /// most `rows` rows and a cache of at most `blocks` pooled blocks.
    fn open(
        harness: *Harness,
        input: IndexerInput,
        rows: u32,
        blocks: u32,
    ) !IndexerSession {
        assert(rows > 0);
        assert(blocks > 0);
        assert(input.arena_length >= rows);

        const shapes = input.shapes;
        assert(input.index_keys.len >= @as(usize, input.arena_length) * shapes.index_dim);
        assert(input.cos.len >= @as(usize, input.arena_length) * shapes.rotary_dim);
        const width = shapes.projectedWidth();
        return .{
            .harness = harness,
            .shapes = shapes,
            .in_features = input.in_features,
            .index_keys = try harness.uploadBf16(input.index_keys),
            .weight = try harness.uploadBf16(input.weight),
            .q_layernorm = try harness.uploadF32(input.q_layernorm),
            .k_layernorm = try harness.uploadF32(input.k_layernorm),
            .cos = try harness.uploadF32(input.cos),
            .sin = try harness.uploadF32(input.sin),
            .projected = try harness.allocate(@as(u64, rows) * width * 4),
            .pooled = try harness.allocate(
                @as(u64, blocks) * shapes.index_dim * 2,
            ),
            .scores = try harness.allocate(
                Attention.indexerScoreBytes(rows, blocks),
            ),
            .partial = try harness.allocate(
                Attention.indexerPartialBytes(rows, shapes),
            ),
            .queries = try harness.allocate(@as(u64, rows) * shapes.index_heads * shapes.index_dim * 2),
            .selected = try harness.allocate(@as(u64, rows) * shapes.slots() * 4),
        };
    }

    /// Runs the six stages for `rows` new tokens at `positions`, with
    /// `arena_length` tokens in the cache and `first_block` blocks already
    /// pooled.
    fn step(
        self: *const IndexerSession,
        hidden: []const f32,
        positions: []const u32,
        arena_length: u32,
        first_block: u32,
    ) !void {
        assert(positions.len > 0);
        assert(self.in_features > 0);
        assert(hidden.len == positions.len * @as(usize, self.in_features));
        assert(first_block <= arena_length / self.shapes.compress_ratio);

        const rows: u32 = @intCast(positions.len);
        const encoded = try self.harness.allocator.alloc(i32, rows);
        defer self.harness.allocator.free(encoded);
        for (positions, 0..) |position, row| {
            encoded[row] = @intCast(position);
        }
        try self.harness.attention.indexerScores(.{
            .hidden = try self.harness.uploadBf16(hidden),
            .weight = self.weight,
            .projected = self.projected,
            .q_layernorm = self.q_layernorm,
            .k_layernorm = self.k_layernorm,
            .queries = self.queries,
            .index_keys = self.index_keys,
            .pooled = self.pooled,
            .scores = self.scores,
            .partial = self.partial,
            .selected = self.selected,
            .positions = try self.harness.uploadI32(encoded),
            .cos = self.cos,
            .sin = self.sin,
            .rows = rows,
            .in_features = self.in_features,
            .arena_length = arena_length,
            .first_block = first_block,
            .shapes = self.shapes,
        });
        assert(@as(usize, rows) == positions.len);
    }

    /// Copies the first step row's selection list back to the host.
    fn downloadSelection(self: *const IndexerSession, out: []i32) !void {
        assert(out.len == self.shapes.slots());
        assert(self.selected != 0);

        try self.harness.downloadI32(self.selected, out);
    }
};

/// fp32 reference of the whole indexer pipeline: projection, per-head norm and
/// rotary on the queries, the key cache append, block pooling, block scores and
/// the reference top-k.
fn indexerOracle(
    allocator: std.mem.Allocator,
    input: IndexerInput,
    selected: []i32,
    scores_out: ?[]f32,
    probe_pooled: ?[]f32,
    probe_queries: ?[]f32,
) !void {
    assert(input.rows > 0);
    assert(selected.len == @as(usize, input.rows) * input.shapes.slots());

    const shapes = input.shapes;
    const rows: usize = input.rows;
    const width = shapes.projectedWidth();
    const index_dim = shapes.index_dim;
    const blocks = input.arena_length / shapes.compress_ratio;
    const projected = try allocator.alloc(f32, rows * width);
    defer allocator.free(projected);
    ops.matmul(input.hidden, input.weight, projected, .{
        .rows = input.rows,
        .in_features = input.in_features,
        .out_features = width,
    });
    const queries = try allocator.alloc(
        f32,
        rows * shapes.index_heads * index_dim,
    );
    defer allocator.free(queries);
    const keys = try allocator.dupe(f32, input.index_keys);
    defer allocator.free(keys);
    const row_buffer = try allocator.alloc(f32, index_dim);
    defer allocator.free(row_buffer);
    const pooled = try allocator.alloc(f32, blocks * index_dim + 1);
    defer allocator.free(pooled);
    const scores = try allocator.alloc(f32, blocks + 1);
    defer allocator.free(scores);
    try oracleQueriesAndKeys(input, projected, queries, keys, row_buffer);
    if (probe_queries) |out| {
        @memcpy(out, queries);
    }
    for (0..blocks) |block| {
        poolOracle(
            pooled[block * index_dim ..][0..index_dim],
            keys,
            input.k_layernorm,
            input.cos,
            input.sin,
            @intCast(block),
            shapes,
            row_buffer,
        );
    }
    if (probe_pooled) |out| {
        @memcpy(out[0 .. blocks * index_dim], pooled[0 .. blocks * index_dim]);
    }
    try oracleSelections(input, queries, pooled, scores, selected, scores_out);
}

/// Block scores and the reference top-k for every query of `input`.
fn oracleSelections(
    input: IndexerInput,
    queries: []const f32,
    pooled: []const f32,
    scores: []f32,
    selected: []i32,
    scores_out: ?[]f32,
) !void {
    assert(input.rows > 0);
    assert(scores.len > 0);
    assert(selected.len == @as(usize, input.rows) * input.shapes.slots());

    const shapes = input.shapes;
    const blocks = input.arena_length / shapes.compress_ratio;
    const taken = try std.testing.allocator.alloc(bool, blocks + 1);
    defer std.testing.allocator.free(taken);
    for (0..input.rows) |row| {
        const visible = input.positions[row] + 1;
        const visible_blocks = visible / shapes.compress_ratio;
        scoreOracle(
            scores,
            queries,
            pooled,
            @intCast(row),
            visible_blocks,
            shapes,
        );
        if (scores_out) |out| {
            @memcpy(
                out[row * blocks ..][0..visible_blocks],
                scores[0..visible_blocks],
            );
        }
        selectOracle(
            scores[0..visible_blocks],
            visible,
            shapes.blockTopk(),
            shapes.compress_ratio,
            selected[row * shapes.slots() ..][0..shapes.slots()],
            taken,
        );
    }
}

/// Per-head norm and rotary on the index queries, plus the raw token key
/// append, in fp32 with the storage roundings the kernels perform.
fn oracleQueriesAndKeys(
    input: IndexerInput,
    projected: []const f32,
    queries: []f32,
    keys: []f32,
    row_buffer: []f32,
) !void {
    assert(input.rows > 0);
    assert(projected.len >= @as(usize, input.rows) * input.shapes.projectedWidth());

    const shapes = input.shapes;
    const width = shapes.projectedWidth();
    const index_dim = shapes.index_dim;
    const rotary = shapes.rotary_dim;
    for (0..input.rows) |row| {
        const position = input.positions[row];
        const cos = input.cos[@as(usize, position) * rotary ..][0..rotary];
        const sin = input.sin[@as(usize, position) * rotary ..][0..rotary];
        for (0..shapes.index_heads) |head| {
            const source = projected[row * width + head * index_dim ..][0..index_dim];
            ops.rmsnorm(
                source,
                input.q_layernorm,
                row_buffer,
                index_dim,
                shapes.eps,
            );
            const target = queries[(row * shapes.index_heads + head) * index_dim ..][0..index_dim];
            reference_rope.apply(row_buffer, cos, sin, target);
            roundTripBf16(target);
        }
        const token_key = projected[row * width + shapes.index_heads * index_dim ..][0..index_dim];
        for (token_key, 0..) |value, dim| {
            keys[@as(usize, position) * index_dim + dim] =
                quant.bf16_to_f32(quant.f32_to_bf16(value));
        }
    }
    assert(keys.len > 0);
}

/// Counts the non-padding entries of one selection row.
fn selectedCount(row: []const i32) u32 {
    assert(row.len > 0);
    assert(row.len <= 1 << 24);

    var count: u32 = 0;
    for (row) |index| {
        if (index >= 0) {
            count += 1;
        }
    }
    assert(count <= row.len);
    return count;
}

/// The non-padding entries of `row`, ascending.
fn filteredSorted(allocator: std.mem.Allocator, row: []const i32) ![]i32 {
    assert(row.len > 0);
    assert(allocator.vtable != undefined);

    var count: usize = 0;
    for (row) |index| {
        if (index >= 0) {
            count += 1;
        }
    }
    const sorted = try allocator.alloc(i32, count);
    var written: usize = 0;
    for (row) |index| {
        if (index >= 0) {
            sorted[written] = index;
            written += 1;
        }
    }
    assert(written == count);
    std.mem.sort(i32, sorted, {}, std.sort.asc(i32));
    return sorted;
}

/// Smallest fraction of a selection that has to agree when the two sides store
/// different precision. Observed 0.999 on the fixture this runs on, so the floor
/// leaves 0.9% of headroom and still fails on a real selection regression; see
/// the header of "the indexer selection and the attention output match the fp32
/// oracle" for why the bf16 side is the faithful one.
const overlap_floor: f64 = 0.99;

/// Elements two ascending, duplicate-free slices have in common.
fn commonElements(actual: []const i32, expected: []const i32) usize {
    assert(actual.len > 0);
    assert(expected.len > 0);

    var shared: usize = 0;
    var left: usize = 0;
    var right: usize = 0;
    while (left < actual.len and right < expected.len) {
        if (actual[left] == expected[right]) {
            shared += 1;
            left += 1;
            right += 1;
        } else if (actual[left] < expected[right]) {
            left += 1;
        } else {
            right += 1;
        }
    }
    assert(shared <= actual.len);
    return shared;
}

/// Reports how much of `actual` is also in `expected`, and fails below
/// `overlap_floor`. Used where the two sides deliberately store different
/// precision, so exact equality is not the question.
///
/// The floor is not decoration: it sits 0.9% below the observed agreement
/// (0.999 at the fixture this runs on) and far above the agreement a stale or
/// broken score row produces, so a regression cannot hide under it.
fn reportOverlap(
    label: []const u8,
    actual: []const i32,
    expected: []const i32,
) !void {
    assert(label.len > 0);
    assert(actual.len > 0);
    assert(expected.len == actual.len);

    const allocator = std.testing.allocator;
    const actual_set = try filteredSorted(allocator, actual);
    defer allocator.free(actual_set);
    const expected_set = try filteredSorted(allocator, expected);
    defer allocator.free(expected_set);
    try std.testing.expectEqual(expected_set.len, actual_set.len);
    const shared = commonElements(actual_set, expected_set);
    const fraction = @as(f64, @floatFromInt(shared)) /
        @as(f64, @floatFromInt(actual_set.len));
    assert(shared <= actual_set.len);
    std.debug.print(
        "{s:<46} {d}/{d} shared ({d:.3})\n",
        .{ label, shared, actual_set.len, fraction },
    );
    try std.testing.expect(fraction >= overlap_floor);
}

/// Checks a selection list against the oracle: sets first, then order.
///
/// Only the set is asserted here: the fixtures this runs on are real scores, so
/// two blocks never tie and the order column below is informational. The order
/// contract (a tie goes to the higher block index, as `torch.topk` and the
/// reference's `top_blocks` do it) is asserted by "the indexer top-k breaks
/// ties the way the reference does", which feeds exact ties through the same
/// stage.
fn expectSelection(
    label: []const u8,
    actual: []const i32,
    expected: []const i32,
) !void {
    assert(label.len > 0);
    assert(actual.len == expected.len);
    assert(actual.len > 0);

    const allocator = std.testing.allocator;
    var ordering: u32 = 0;
    for (actual, expected) |got, want| {
        if (got != want) {
            ordering += 1;
        }
    }
    const actual_set = try filteredSorted(allocator, actual);
    defer allocator.free(actual_set);
    const expected_set = try filteredSorted(allocator, expected);
    defer allocator.free(expected_set);
    // Set difference, not a positional one: a single element present on only
    // one side shifts every later rank, and counting those shifts as
    // mismatches overstates the difference by orders of magnitude.
    const shared = commonElements(actual_set, expected_set);
    const difference = actual_set.len - shared;
    std.debug.print(
        "{s:<46} entries {d}, selected {d}, set_difference {d}, " ++
            "list_order {d}\n",
        .{ label, actual.len, selectedCount(actual), difference, ordering },
    );
    try std.testing.expectEqual(expected_set.len, actual_set.len);
    try std.testing.expectEqual(@as(usize, 0), difference);
}

/// A random indexer input that owns its host buffers.
const RandomIndexer = struct {
    shapes: Shapes,
    rows: u32,
    in_features: u32,
    arena_length: u32,
    hidden: []f32,
    weight: []f32,
    q_layernorm: []f32,
    k_layernorm: []f32,
    index_keys: []f32,
    positions: []u32,
    cos: []f32,
    sin: []f32,

    /// Fills every buffer; the values are BF16-rounded where the kernels store
    /// BF16, so the oracle and the GPU read the same numbers.
    fn make(
        harness: *Harness,
        shapes: Shapes,
        arena_length: u32,
        rows: u32,
        in_features: u32,
        seed: u64,
    ) !RandomIndexer {
        assert(rows > 0);
        assert(arena_length >= rows);
        assert(in_features > 0);

        const allocator = harness.allocator;
        const rotary = shapes.rotary_dim;
        const hidden = try randomValues(
            allocator,
            @as(usize, rows) * in_features,
            seed,
        );
        errdefer allocator.free(hidden);
        const weight = try randomValues(
            allocator,
            @as(usize, shapes.projectedWidth()) * in_features,
            seed + 1,
        );
        errdefer allocator.free(weight);
        const q_layernorm = try randomValues(allocator, shapes.index_dim, seed + 2);
        errdefer allocator.free(q_layernorm);
        const k_layernorm = try randomValues(allocator, shapes.index_dim, seed + 3);
        errdefer allocator.free(k_layernorm);
        const index_keys = try randomValues(
            allocator,
            @as(usize, arena_length) * shapes.index_dim,
            seed + 4,
        );
        errdefer allocator.free(index_keys);
        const positions = try allocator.alloc(u32, rows);
        errdefer allocator.free(positions);
        const cos = try allocator.alloc(f32, @as(usize, arena_length) * rotary);
        errdefer allocator.free(cos);
        const sin = try allocator.alloc(f32, @as(usize, arena_length) * rotary);
        errdefer allocator.free(sin);

        scaleValues(weight, 0.05);
        scaleValues(q_layernorm, 0.1);
        scaleValues(k_layernorm, 0.1);
        roundTripBf16(hidden);
        roundTripBf16(weight);
        roundTripBf16(index_keys);
        for (positions, 0..) |*slot, row| {
            slot.* = arena_length - rows + @as(u32, @intCast(row));
        }
        try buildTables(harness, shapes, arena_length, .{ 0, 0, 0 }, cos, sin);
        assert(positions[rows - 1] == arena_length - 1);
        return .{
            .shapes = shapes,
            .rows = rows,
            .in_features = in_features,
            .arena_length = arena_length,
            .hidden = hidden,
            .weight = weight,
            .q_layernorm = q_layernorm,
            .k_layernorm = k_layernorm,
            .index_keys = index_keys,
            .positions = positions,
            .cos = cos,
            .sin = sin,
        };
    }

    /// Releases every buffer.
    fn deinit(self: *const RandomIndexer, allocator: std.mem.Allocator) void {
        assert(self.rows > 0);
        assert(self.cos.len > 0);

        allocator.free(self.sin);
        allocator.free(self.cos);
        allocator.free(self.positions);
        allocator.free(self.index_keys);
        allocator.free(self.k_layernorm);
        allocator.free(self.q_layernorm);
        allocator.free(self.weight);
        allocator.free(self.hidden);
    }

    /// The borrow the pipelines take.
    fn view(self: *const RandomIndexer) IndexerInput {
        assert(self.rows > 0);
        assert(self.arena_length > 0);

        return .{
            .shapes = self.shapes,
            .rows = self.rows,
            .in_features = self.in_features,
            .arena_length = self.arena_length,
            .hidden = self.hidden,
            .weight = self.weight,
            .q_layernorm = self.q_layernorm,
            .k_layernorm = self.k_layernorm,
            .index_keys = self.index_keys,
            .positions = self.positions,
            .cos = self.cos,
            .sin = self.sin,
        };
    }
};

/// Prints the largest block-score difference between the kernel and the
/// oracle for the last streaming group. The selection is the gate; this only
/// shows how much headroom the arithmetic has, and the score buffer only holds
/// one group, so the comparison is over the group the wrapper scored last.
fn reportScoreDifference(
    actual: []const f32,
    expected: []const f32,
    positions: []const u32,
    blocks: u32,
    used: u32,
    first_block: u32,
    ratio: u32,
) void {
    assert(actual.len == @as(usize, @intCast(used)) * positions.len);
    assert(expected.len >= @as(usize, @intCast(blocks)) * positions.len);
    assert(positions.len > 0);
    assert(ratio > 0);
    assert(used <= blocks);
    assert(first_block + used == blocks);

    var largest: f32 = 0.0;
    var row_of_largest: usize = 0;
    var block_of_largest: usize = 0;
    for (positions, 0..) |position, row| {
        const visible_blocks = (position + 1) / ratio;
        const in_group = if (visible_blocks > first_block)
            @min(visible_blocks - first_block, used)
        else
            0;
        for (0..in_group) |local| {
            const index = row * used + local;
            const expected_index = row * blocks + first_block + local;
            const delta = @abs(actual[index] - expected[expected_index]);
            if (delta > largest) {
                largest = delta;
                row_of_largest = row;
                block_of_largest = first_block + local;
            }
        }
    }
    const local = block_of_largest - first_block;
    const index = row_of_largest * used + local;
    const expected_index = row_of_largest * blocks + block_of_largest;
    std.debug.print(
        "indexer block scores (last {d}-block group): max_abs_diff {e:.3} " ++
            "at row {d} block {d} (kernel {e:.4} oracle {e:.4})\n",
        .{
            used,
            largest,
            row_of_largest,
            block_of_largest,
            actual[index],
            expected[expected_index],
        },
    );
}

/// Checks the indexer's selection against the reference top-k at real shapes.
fn checkIndexerSelection(
    harness: *Harness,
    shapes: Shapes,
    arena_length: u32,
    rows: u32,
    in_features: u32,
    seed: u64,
) !void {
    assert(rows >= 3);
    assert(arena_length > rows);
    assert(in_features > 0);

    const allocator = harness.allocator;
    shapes.validate();
    var random = try RandomIndexer.make(
        harness,
        shapes,
        arena_length,
        rows,
        in_features,
        seed,
    );
    defer random.deinit(allocator);
    const input = random.view();
    const selected = try allocator.alloc(i32, @as(usize, rows) * shapes.slots());
    defer allocator.free(selected);
    const blocks = arena_length / shapes.compress_ratio;
    const first = if (blocks == 0) 0 else ((blocks - 1) / score_group_blocks) * score_group_blocks;
    const used = blocks - first;
    const scores = try allocator.alloc(f32, @as(usize, rows) * used);
    defer allocator.free(scores);
    const pooled_count = @as(usize, blocks) * shapes.index_dim;
    const query_count = @as(usize, rows) * shapes.index_heads * shapes.index_dim;
    const probe = try allocator.alloc(f32, 2 * (pooled_count + query_count));
    defer allocator.free(probe);
    @memset(probe, 0.0);
    const oracle_pooled = probe[pooled_count + query_count ..][0..pooled_count];
    const oracle_queries = probe[pooled_count..][0..query_count];
    try runIndexer(harness, input, selected, scores, null, probe[0..pooled_count], probe[2 * pooled_count + query_count ..]);
    const oracle = try allocator.alloc(i32, selected.len);
    defer allocator.free(oracle);
    const oracle_scores = try allocator.alloc(f32, @as(usize, rows) * blocks);
    defer allocator.free(oracle_scores);
    @memset(oracle_scores, 0.0);
    try indexerOracle(
        allocator,
        input,
        oracle,
        oracle_scores,
        probe[pooled_count + query_count ..],
        probe[pooled_count .. pooled_count + query_count],
    );
    reportScoreDifference(
        scores,
        oracle_scores,
        random.positions,
        blocks,
        used,
        first,
        shapes.compress_ratio,
    );
    const kernel_pooled = probe[0..pooled_count];
    const kernel_queries = probe[2 * pooled_count + query_count ..];
    var pooled_exact: u32 = 0;
    for (kernel_pooled, oracle_pooled) |got, want| {
        if (got != want) {
            pooled_exact += 1;
        }
    }
    var query_exact: u32 = 0;
    for (kernel_queries[0..query_count], oracle_queries) |got, want| {
        if (got != want) {
            query_exact += 1;
        }
    }
    // Operand agreement: the selection is the gate, this shows how much
    // headroom it has. A mismatch here is one bf16 ulp on a rounding boundary,
    // which is what can reorder two blocks whose scores are within noise.
    std.debug.print(
        "indexer operands: pooled {d}/{d} differ, queries {d}/{d} differ\n",
        .{ pooled_exact, pooled_count, query_exact, query_count },
    );
    try expectSelection("indexer selection (real shapes)", selected, oracle);
    for (rows - 3..rows) |row| {
        std.debug.print(
            "query {d}: selected {d} tokens, first {d}, last {d}\n",
            .{
                row,
                selectedCount(selected[row * shapes.slots() ..][0..shapes.slots()]),
                selected[row * shapes.slots()],
                selected[(row + 1) * shapes.slots() - 1],
            },
        );
    }
}

/// Runs the indexer across two steps over one resident cache: a prefill that
/// completes `tokens / compress_ratio` blocks, then a decode step that completes
/// none, because `first_block` already equals the block count. Three of every
/// four decode steps at `compress_ratio` 4 look like the second one, and its
/// selection has to come from its own query rather than from the previous
/// step's score row.
fn checkNoNewBlockStep(
    harness: *Harness,
    shapes: Shapes,
    tokens: u32,
    in_features: u32,
    seed: u64,
) !void {
    assert(tokens > shapes.budget);
    assert(tokens % shapes.compress_ratio == 0);
    assert(in_features > 0);

    const allocator = harness.allocator;
    shapes.validate();
    const ratio = shapes.compress_ratio;
    const blocks = tokens / ratio;
    var random = try RandomIndexer.make(
        harness,
        shapes,
        tokens + 1,
        tokens,
        in_features,
        seed,
    );
    defer random.deinit(allocator);
    // The prefill covers positions 0 .. tokens-1, so the second step is the one
    // token at `tokens`.
    for (random.positions, 0..) |*position, row| {
        position.* = @intCast(row);
    }
    var session = try IndexerSession.open(
        harness,
        random.view(),
        tokens,
        blocks,
    );
    try session.step(random.hidden, random.positions, tokens, 0);
    const next = [_]u32{tokens};
    try session.step(random.hidden[0..in_features], next[0..], tokens + 1, blocks);
    const selected = try allocator.alloc(i32, shapes.slots());
    defer allocator.free(selected);
    try session.downloadSelection(selected);
    // The fp32 reference reads the cache the kernels actually built: the
    // prefill's appended keys come back from the device, so its pooling sees the
    // same bf16 key values the kernel pooled.
    const keys = try allocator.alloc(
        f32,
        @as(usize, tokens + 1) * shapes.index_dim,
    );
    defer allocator.free(keys);
    try harness.downloadBf16(session.index_keys, keys);
    const oracle_input = IndexerInput{
        .shapes = shapes,
        .rows = 1,
        .in_features = in_features,
        .arena_length = tokens + 1,
        .hidden = random.hidden[0..in_features],
        .weight = random.weight,
        .q_layernorm = random.q_layernorm,
        .k_layernorm = random.k_layernorm,
        .index_keys = keys,
        .positions = next[0..],
        .cos = random.cos,
        .sin = random.sin,
    };
    const expected = try allocator.alloc(i32, shapes.slots());
    defer allocator.free(expected);
    try indexerOracle(allocator, oracle_input, expected, null, null, null);
    std.debug.print(
        "step with no new block: {d} blocks pooled, {d} visible, " ++
            "selected {d} tokens\n",
        .{ blocks, (tokens + 1) / ratio, selectedCount(selected) },
    );
    try expectSelection("indexer selection after a step with no new block", selected, expected);
}

/// Blocks of the tie-break fixture: the odd ones share the top score.
const tie_blocks: u32 = 8;

/// Fills a score row with exact ties and nothing else in between.
///
/// A real fixture never produces a tie, so without a row like this the ordering
/// half of acceptance #2 is unmeasured. `take` = 2 then lands inside the
/// four-way top group (the index digits decide the order) and `take` = 4 covers
/// the group exactly (the radix select resolves early and must still order by
/// index).
fn tieScores(scores: []f32) void {
    assert(scores.len > 1);
    assert(scores.len % 2 == 0);

    for (scores, 0..) |*score, block| {
        score.* = if (block % 2 == 1) 3.0 else 1.0;
    }
}

/// Runs the selection stage alone on a tied score row and compares the emitted
/// token list with the reference top-k, position by position.
fn checkSelectionTieBreak(
    harness: *Harness,
    shapes: Shapes,
    blocks: u32,
) !void {
    assert(blocks % 2 == 0);
    assert(shapes.blockTopk() < blocks);

    const allocator = harness.allocator;
    const ratio = shapes.compress_ratio;
    const visible = blocks * ratio;
    const scores = try allocator.alloc(f32, blocks);
    defer allocator.free(scores);
    tieScores(scores);
    const selected = try allocator.alloc(i32, shapes.slots());
    defer allocator.free(selected);
    @memset(selected, -1);
    const encoded = [_]i32{@intCast(visible - 1)};
    const scores_device = try harness.uploadF32(scores);
    const positions_device = try harness.uploadI32(encoded[0..]);
    const selected_device = try harness.uploadI32(selected);
    try harness.attention.select(
        scores_device,
        positions_device,
        selected_device,
        1,
        blocks,
        shapes,
    );
    try harness.downloadI32(selected_device, selected);
    const expected = try allocator.alloc(i32, shapes.slots());
    defer allocator.free(expected);
    const taken = try allocator.alloc(bool, blocks);
    defer allocator.free(taken);
    selectOracle(scores, visible, shapes.blockTopk(), ratio, expected, taken);
    std.debug.print(
        "tie-break take {d} of {d} tied blocks: kernel",
        .{ shapes.blockTopk(), blocks },
    );
    const span = @as(usize, shapes.blockTopk()) * ratio;
    assert(span <= selected.len);
    for (0..shapes.blockTopk()) |item| {
        const first = selected[item * ratio];
        std.debug.print(" {d}", .{@divTrunc(first, @as(i32, @intCast(ratio)))});
    }
    std.debug.print("\n", .{});
    try std.testing.expectEqualSlices(i32, expected, selected);
}

/// Asserts how many gathered rows a launch reported past the launch's causal
/// horizon (its last query position).
///
/// The kernels skip such a row instead of reading it, so the counter is the
/// only evidence that the selection they were given was causal; a corrupted
/// list would otherwise pass every numeric check by masking itself.
fn expectViolations(
    harness: *Harness,
    counter: cuda_driver.DevicePtr,
    expected: u32,
    label: []const u8,
) !void {
    assert(label.len > 0);
    assert(counter != 0);

    var count = [_]i32{0};
    try harness.downloadI32(counter, count[0..]);
    std.debug.print(
        "{s:<46} violations {d} (expected {d})\n",
        .{ label, count[0], expected },
    );
    try std.testing.expectEqual(@as(i32, @intCast(expected)), count[0]);
}

/// Runs one attention case and checks it against the fp32 oracle.
fn checkAttentionCase(
    harness: *Harness,
    shapes: Shapes,
    case: AttentionCase,
) !void {
    assert(case.rows() > 0);
    assert(case.count() > 0);

    const allocator = harness.allocator;
    const rows = case.rows();
    const encoded = try allocator.alloc(i32, rows);
    defer allocator.free(encoded);
    for (case.positions, 0..) |position, row| {
        encoded[row] = @intCast(position);
    }
    // The host is the one that refuses a selection reaching past its query: the
    // kernels only skip and count such a row, they do not fail on it. The bound
    // is the chunk's last position, because a prefill chunk may share one union
    // working set across its rows (`attendGatheredRow` shares none, but this
    // sweep does).
    try checkSelectionWithinPosition(
        case.gather_index,
        case.positions[rows - 1],
    );
    const violations = try harness.uploadI32(&[_]i32{0});
    const args = PrefillArgs{
        .queries = try harness.uploadBf16(case.queries),
        .gathered_k = try harness.uploadBf16(case.gathered_k),
        .gathered_v = try harness.uploadBf16(case.gathered_v),
        .gather_index = try harness.uploadI32(case.gather_index),
        .positions = try harness.uploadI32(encoded),
        .out = try harness.allocate(@as(u64, case.queries.len) * 2),
        .violations = violations,
        .rows = rows,
        .count = case.count(),
        .shapes = shapes,
    };
    try harness.attention.prefill(args);
    const actual = try allocator.alloc(f32, case.queries.len);
    defer allocator.free(actual);
    try harness.downloadBf16(args.out, actual);
    try expectViolations(harness, violations, 0, "prefill");
    try attendOracle(
        allocator,
        case.queries,
        case.gathered_k,
        case.gathered_v,
        case.gather_index,
        case.positions,
        case.out,
        shapes,
    );
    var label: [96]u8 = undefined;
    const text = try std.fmt.bufPrint(
        &label,
        "prefill count={d} rows={d} pos<={d}",
        .{ case.count(), rows, case.positions[rows - 1] },
    );
    try reportError(text, actual, case.out, 4.0e-3);
    if (rows == 1) {
        try checkDecodeInvariant(harness, shapes, case, text);
    }
}

/// Runs decode for the single query of `case` and checks it against the oracle
/// and against the row `attn_prefill` produced for the same position.
fn checkDecodeInvariant(
    harness: *Harness,
    shapes: Shapes,
    case: AttentionCase,
    label: []const u8,
) !void {
    assert(case.rows() == 1);
    assert(label.len > 0);

    const allocator = harness.allocator;
    const row_elements = @as(usize, shapes.heads) * shapes.head_dim;
    const gathered_k = try harness.uploadBf16(case.gathered_k);
    const gathered_v = try harness.uploadBf16(case.gathered_v);
    const gather_index = try harness.uploadI32(case.gather_index);
    const positions = try harness.uploadI32(&[_]i32{@intCast(case.positions[0])});
    const prefill_out = try harness.allocate(@as(u64, case.queries.len) * 2);
    const prefill_violations = try harness.uploadI32(&[_]i32{0});
    try harness.attention.prefill(.{
        .queries = try harness.uploadBf16(case.queries),
        .gathered_k = gathered_k,
        .gathered_v = gathered_v,
        .gather_index = gather_index,
        .positions = positions,
        .out = prefill_out,
        .violations = prefill_violations,
        .rows = 1,
        .count = case.count(),
        .shapes = shapes,
    });
    const prefill = try allocator.alloc(f32, case.queries.len);
    defer allocator.free(prefill);
    try harness.downloadBf16(prefill_out, prefill);
    try expectViolations(harness, prefill_violations, 0, "prefill");
    const out = try harness.allocate(@as(u64, row_elements) * 2);
    const decode_violations = try harness.uploadI32(&[_]i32{0});
    try harness.attention.decode(.{
        .query = try harness.uploadBf16(case.queries),
        .gathered_k = gathered_k,
        .gathered_v = gathered_v,
        .gather_index = gather_index,
        .position = case.positions[0],
        .out = out,
        .violations = decode_violations,
        .count = case.count(),
        .shapes = shapes,
    });
    const decoded = try allocator.alloc(f32, row_elements);
    defer allocator.free(decoded);
    try harness.downloadBf16(out, decoded);
    try expectViolations(harness, decode_violations, 0, "decode");
    var oracle_label: [96]u8 = undefined;
    try reportError(
        try std.fmt.bufPrint(&oracle_label, "{s} decode", .{label}),
        decoded,
        case.out,
        4.0e-3,
    );
    var invariant_label: [96]u8 = undefined;
    try reportError(
        try std.fmt.bufPrint(&invariant_label, "{s} decode==prefill", .{label}),
        decoded,
        prefill[0..row_elements],
        4.0e-3,
    );
}

/// A case whose selection was corrupted by hand: the last gathered row claims
/// to be the token right after the query's position.
///
/// This is the failure the issue's §1(a) tripwire exists for. The kernels mask
/// such a row, so a corrupt selection can never read the future; what must also
/// happen is that it is noticed — on the host by
/// `checkSelectionWithinPosition`, and on the device by the counter.
fn makeCorruptedCase(
    allocator: std.mem.Allocator,
    shapes: Shapes,
    count: u32,
    position: u32,
    seed: u64,
) !AttentionCase {
    assert(count > 1);
    assert(shapes.heads > 0);

    const case = try makeAttentionCase(
        allocator,
        shapes,
        count,
        1,
        position,
        seed,
    );
    assert(case.gather_index.len == count);
    // The last row of the working set now points one past the query.
    case.gather_index[count - 1] = @intCast(position + 1);
    return case;
}

/// Checks the corrupt case end to end: host refusal, device count, oracle.
fn checkCorruptedSelection(harness: *Harness, shapes: Shapes) !void {
    assert(shapes.heads > 0);
    assert(shapes.budget > 1);

    const allocator = harness.allocator;
    const position: u32 = 4095;
    const case = try makeCorruptedCase(
        allocator,
        shapes,
        shapes.slots(),
        position,
        0x51ce_0a00,
    );
    defer freeAttentionCase(case, allocator);
    try std.testing.expectError(
        error.FutureTokenSelected,
        checkSelectionWithinPosition(case.gather_index, case.positions[0]),
    );
    try attendOracle(
        allocator,
        case.queries,
        case.gathered_k,
        case.gathered_v,
        case.gather_index,
        case.positions,
        case.out,
        shapes,
    );
    try checkCorruptedPrefill(harness, shapes, case);
    try checkCorruptedDecode(harness, shapes, case);
}

/// Runs `attn_prefill` on the corrupt case: one violation, no future token.
fn checkCorruptedPrefill(
    harness: *Harness,
    shapes: Shapes,
    case: AttentionCase,
) !void {
    assert(case.rows() == 1);
    assert(case.count() > 1);

    const allocator = harness.allocator;
    const only = [_]i32{@intCast(case.positions[0])};
    const positions = try harness.uploadI32(only[0..]);
    const violations = try harness.uploadI32(&[_]i32{0});
    const out = try harness.allocate(@as(u64, case.queries.len) * 2);
    try harness.attention.prefill(.{
        .queries = try harness.uploadBf16(case.queries),
        .gathered_k = try harness.uploadBf16(case.gathered_k),
        .gathered_v = try harness.uploadBf16(case.gathered_v),
        .gather_index = try harness.uploadI32(case.gather_index),
        .positions = positions,
        .out = out,
        .violations = violations,
        .rows = 1,
        .count = case.count(),
        .shapes = shapes,
    });
    const actual = try allocator.alloc(f32, case.queries.len);
    defer allocator.free(actual);
    try harness.downloadBf16(out, actual);
    try expectViolations(harness, violations, 1, "corrupt selection, prefill");
    try reportError(
        "corrupt selection, prefill vs oracle",
        actual,
        case.out,
        4.0e-3,
    );
}

/// Runs `attn_decode` on the corrupt case: one violation, no future token.
fn checkCorruptedDecode(
    harness: *Harness,
    shapes: Shapes,
    case: AttentionCase,
) !void {
    assert(case.rows() == 1);
    assert(case.count() > 1);

    const allocator = harness.allocator;
    const row_elements = @as(usize, shapes.heads) * shapes.head_dim;
    const violations = try harness.uploadI32(&[_]i32{0});
    const out = try harness.allocate(@as(u64, row_elements) * 2);
    try harness.attention.decode(.{
        .query = try harness.uploadBf16(case.queries),
        .gathered_k = try harness.uploadBf16(case.gathered_k),
        .gathered_v = try harness.uploadBf16(case.gathered_v),
        .gather_index = try harness.uploadI32(case.gather_index),
        .position = case.positions[0],
        .out = out,
        .violations = violations,
        .count = case.count(),
        .shapes = shapes,
    });
    const actual = try allocator.alloc(f32, row_elements);
    defer allocator.free(actual);
    try harness.downloadBf16(out, actual);
    try expectViolations(harness, violations, 1, "corrupt selection, decode");
    try reportError(
        "corrupt selection, decode vs oracle",
        actual,
        case.out,
        4.0e-3,
    );
}

/// Arguments one `attn_decode` launch needs, built from a case.
fn decodeArgs(
    harness: *Harness,
    shapes: Shapes,
    case: AttentionCase,
) !DecodeArgs {
    assert(case.rows() == 1);
    assert(case.count() > 0);

    return .{
        .query = try harness.uploadBf16(case.queries),
        .gathered_k = try harness.uploadBf16(case.gathered_k),
        .gathered_v = try harness.uploadBf16(case.gathered_v),
        .gather_index = try harness.uploadI32(case.gather_index),
        .position = case.positions[0],
        .out = try harness.allocate(
            @as(u64, shapes.heads) * shapes.head_dim * 2,
        ),
        .violations = null,
        .count = case.count(),
        .shapes = shapes,
    };
}

/// Times `attn_decode` for the single query of `case`. The 300 us/launch
/// target of the issue is reported, not enforced: the number moves with
/// whatever else is running on the GPU.
fn reportDecodeLatency(
    harness: *Harness,
    shapes: Shapes,
    case: AttentionCase,
) !void {
    assert(case.rows() == 1);
    assert(shapes.heads > 0);

    const args = try decodeArgs(harness, shapes, case);
    const micros = try measure(harness, DecodeDriver{ .args = args });
    std.debug.print(
        "attn_decode count={d} pos={d}: {d:.1} us per launch " ++
            "(median of {d} samples of {d})\n",
        .{
            args.count,
            args.position,
            micros,
            timing_samples,
            timing_launches_per_sample,
        },
    );
    try std.testing.expect(micros > 0.0);
}

/// Times a multi-row prefill of `case` and returns the microseconds per launch.
fn reportPrefillLatency(
    harness: *Harness,
    shapes: Shapes,
    case: AttentionCase,
) !f64 {
    assert(case.rows() > 1);
    assert(shapes.heads > 0);

    const allocator = harness.allocator;
    const encoded = try allocator.alloc(i32, case.rows());
    defer allocator.free(encoded);
    for (case.positions, 0..) |position, row| {
        encoded[row] = @intCast(position);
    }
    const args = PrefillArgs{
        .queries = try harness.uploadBf16(case.queries),
        .gathered_k = try harness.uploadBf16(case.gathered_k),
        .gathered_v = try harness.uploadBf16(case.gathered_v),
        .gather_index = try harness.uploadI32(case.gather_index),
        .positions = try harness.uploadI32(encoded),
        .out = try harness.allocate(@as(u64, case.queries.len) * 2),
        .violations = null,
        .rows = case.rows(),
        .count = case.count(),
        .shapes = shapes,
    };
    const micros = try measure(harness, PrefillDriver{ .args = args });
    const per_token = micros / @as(f64, @floatFromInt(args.rows));
    std.debug.print(
        "attn_prefill count={d} rows={d}: {d:.1} us per launch = " ++
            "{d:.2} us/token/layer{s}\n",
        .{
            args.count,
            args.rows,
            micros,
            per_token,
            if (prefillUsesMma(shapes, args.rows)) "" else " (warp-scalar)",
        },
    );
    try std.testing.expect(micros > 0.0);
    return micros;
}

test "the indexed attention layer matches the HF golden" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();
    var fixtures = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer fixtures.deinit();
    try checkGoldenLayer(&harness, fixtures.allocator());
}

/// Runs one full indexed-attention layer of the tiny model and compares the
/// result with the HF trace: the indexer's selection exactly, the rotary
/// intermediates, and `layer2/attn/out`.
fn checkGoldenLayer(
    harness: *Harness,
    fixtures: std.mem.Allocator,
) !void {
    assert(fixtures.vtable != undefined);
    assert(harness.allocator.vtable != undefined);

    const config = try reference_fixtures.load_config(fixtures);
    const shapes = shapesFromConfig(config);
    shapes.validate();
    const tokens: u32 = 12;
    const hidden_size = config.hidden_size;
    const trace = try reference_fixtures.open_trace(fixtures);
    const checkpoint = try reference_fixtures.open_checkpoint(fixtures);
    const mixed = (try trace.tensor("layer2/hc_attn/mixed")).values;
    const golden_out = (try trace.tensor("layer2/attn/out")).values;
    const golden_gate = (try trace.tensor("layer2/attn/gate")).values;
    const golden_keys = (try trace.tensor("layer2/attn/k")).values;
    const golden_queries = (try trace.tensor("layer2/attn/q")).values;
    const golden_selected = try trace.tensor_i32(
        "layer2/attn/indexer/selected",
    );
    const weights = try LayerTwoWeights.load(&checkpoint);
    const built = try projectGoldenQueries(fixtures, shapes, config, mixed, weights, tokens);
    try compareGoldenRotary(harness, fixtures, shapes, built, golden_queries, golden_keys, tokens);
    const selected = try fixtures.alloc(i32, tokens * shapes.slots());
    try runGoldenIndexer(
        harness,
        fixtures,
        shapes,
        config,
        mixed,
        weights,
        tokens,
        selected,
    );
    try expectSelection("layer2/attn/indexer/selected", selected, golden_selected);
    const produced = try runGoldenAttention(
        harness,
        fixtures,
        shapes,
        built,
        selected,
        tokens,
    );
    try reportError("layer2/attn/gate (fp32 projection)", built.gate, golden_gate, 1.0e-4);
    const gated = try fixtures.alloc(f32, tokens * shapes.heads * shapes.head_dim);
    for (produced, built.gate, 0..) |value, gate, index| {
        gated[index] = value * ops.sigmoid(gate);
    }
    const out = try fixtures.alloc(f32, tokens * hidden_size);
    ops.matmul(gated, weights.o_proj, out, .{
        .rows = tokens,
        .in_features = shapes.heads * shapes.head_dim,
        .out_features = hidden_size,
    });
    try reportError("layer2/attn/out (bf16 kernels)", out, golden_out, 4.0e-3);
    assert(mixed.len == tokens * hidden_size);
}

/// Layer-2 projections of the tiny model, normed on the host: `q_norm` and
/// `k_norm` belong to I-012 (grouped RMSNorm), so this harness only supplies
/// the BF16 operands the indexed-attention kernels consume.
const GoldenProjections = struct {
    /// `[tokens, heads, head_dim]`, normed and BF16-rounded.
    q: []f32,
    /// `[tokens, kv_heads, head_dim]`, normed and BF16-rounded.
    k: []f32,
    /// `[tokens, kv_heads, head_dim]`, BF16-rounded.
    v: []f32,
    /// `[tokens, heads, head_dim]`, kept in fp32 as the reference does.
    gate: []f32,
};

/// Projects `mixed` with the layer-2 weights and normalises every head.
fn projectGoldenQueries(
    fixtures: std.mem.Allocator,
    shapes: Shapes,
    config: reference_fixtures.RefConfig,
    mixed: []const f32,
    weights: LayerTwoWeights,
    tokens: u32,
) !GoldenProjections {
    assert(tokens > 0);
    assert(mixed.len == @as(usize, tokens) * config.hidden_size);

    const head_dim = shapes.head_dim;
    const kv_heads = shapes.kv_heads;
    const heads = shapes.heads;
    const qkV = try fixtures.alloc(f32, @as(usize, tokens) * heads * head_dim * 2);
    const keys = try fixtures.alloc(f32, @as(usize, tokens) * kv_heads * head_dim);
    const values = try fixtures.alloc(f32, @as(usize, tokens) * kv_heads * head_dim);
    ops.matmul(mixed, weights.q_proj, qkV, .{
        .rows = tokens,
        .in_features = config.hidden_size,
        .out_features = heads * head_dim * 2,
    });
    ops.matmul(mixed, weights.k_proj, keys, .{
        .rows = tokens,
        .in_features = config.hidden_size,
        .out_features = kv_heads * head_dim,
    });
    ops.matmul(mixed, weights.v_proj, values, .{
        .rows = tokens,
        .in_features = config.hidden_size,
        .out_features = kv_heads * head_dim,
    });
    const queries = try fixtures.alloc(f32, @as(usize, tokens) * heads * head_dim);
    const gate = try fixtures.alloc(f32, @as(usize, tokens) * heads * head_dim);
    const normed = try fixtures.alloc(f32, head_dim);
    for (0..tokens) |token| {
        for (0..heads) |head| {
            const base = (token * heads + head) * head_dim * 2;
            ops.rmsnorm(
                qkV[base..][0..head_dim],
                weights.q_norm,
                normed,
                head_dim,
                shapes.eps,
            );
            const target = (token * heads + head) * head_dim;
            @memcpy(queries[target..][0..head_dim], normed);
            @memcpy(gate[target..][0..head_dim], qkV[base + head_dim ..][0..head_dim]);
        }
        for (0..kv_heads) |head| {
            const base = (token * kv_heads + head) * head_dim;
            ops.rmsnorm(
                keys[base..][0..head_dim],
                weights.k_norm,
                normed,
                head_dim,
                shapes.eps,
            );
            @memcpy(keys[base..][0..head_dim], normed);
        }
    }
    roundTripBf16(queries);
    roundTripBf16(keys);
    roundTripBf16(values);
    assert(queries.len == @as(usize, tokens) * heads * head_dim);
    return .{ .q = queries, .k = keys, .v = values, .gate = gate };
}

/// Applies the partial mRoPE to `q` and `k` on the GPU and compares the result
/// with the golden `layer2/attn/q` and `layer2/attn/k`, which are stored
/// head-major.
fn compareGoldenRotary(
    harness: *Harness,
    fixtures: std.mem.Allocator,
    shapes: Shapes,
    built: GoldenProjections,
    golden_queries: []const f32,
    golden_keys: []const f32,
    tokens: u32,
) !void {
    assert(tokens > 0);
    assert(golden_queries.len == @as(usize, tokens) * shapes.heads * shapes.head_dim);

    const head_dim = shapes.head_dim;
    const rotary = shapes.rotary_dim;
    const cos = try fixtures.alloc(f32, @as(usize, tokens) * rotary);
    const sin = try fixtures.alloc(f32, @as(usize, tokens) * rotary);
    try buildTables(harness, shapes, tokens, .{ 0, 0, 0 }, cos, sin);
    const encoded = try fixtures.alloc(i32, tokens);
    for (encoded, 0..) |*slot, token| {
        slot.* = @intCast(token);
    }
    const device_positions = try harness.uploadI32(encoded);
    const device_cos = try harness.uploadF32(cos);
    const device_sin = try harness.uploadF32(sin);
    const device_queries = try harness.uploadBf16(built.q);
    const device_keys = try harness.uploadBf16(built.k);
    try harness.attention.applyRope(.{
        .values = device_queries,
        .cos = device_cos,
        .sin = device_sin,
        .positions = device_positions,
        .rows = tokens,
        .heads = shapes.heads,
        .head_dim = head_dim,
        .rotary_dim = rotary,
    });
    try harness.attention.applyRope(.{
        .values = device_keys,
        .cos = device_cos,
        .sin = device_sin,
        .positions = device_positions,
        .rows = tokens,
        .heads = shapes.kv_heads,
        .head_dim = head_dim,
        .rotary_dim = rotary,
    });
    try harness.downloadBf16(device_queries, built.q);
    try harness.downloadBf16(device_keys, built.k);
    try compareGoldenTensor(
        fixtures,
        "layer2/attn/q (rope on bf16)",
        built.q,
        golden_queries,
        tokens,
        shapes.heads,
        head_dim,
    );
    try compareGoldenTensor(
        fixtures,
        "layer2/attn/k (rope on bf16)",
        built.k,
        golden_keys,
        tokens,
        shapes.kv_heads,
        head_dim,
    );
}

/// Compares one token-major result with a head-major golden tensor.
fn compareGoldenTensor(
    fixtures: std.mem.Allocator,
    label: []const u8,
    actual: []const f32,
    golden: []const f32,
    tokens: u32,
    heads: u32,
    head_dim: u32,
) !void {
    assert(label.len > 0);
    assert(actual.len == @as(usize, tokens) * heads * head_dim);
    assert(golden.len == actual.len);

    const transposed = try fixtures.alloc(f32, actual.len);
    transposeHeads(transposed, golden, tokens, heads, head_dim);
    try reportError(label, actual, transposed, 8.0e-3);
}

/// Rewrites a golden `[heads, tokens, dim]` tensor as `[tokens, heads, dim]`.
fn transposeHeads(
    out: []f32,
    source: []const f32,
    tokens: u32,
    heads: u32,
    head_dim: u32,
) void {
    assert(out.len == @as(usize, tokens) * heads * head_dim);
    assert(source.len == out.len);
    assert(heads > 0);

    for (0..tokens) |token| {
        for (0..heads) |head| {
            for (0..head_dim) |dim| {
                out[(token * heads + head) * head_dim + dim] =
                    source[(head * tokens + token) * head_dim + dim];
            }
        }
    }
}

/// Runs the indexer on the golden layer input.
fn runGoldenIndexer(
    harness: *Harness,
    fixtures: std.mem.Allocator,
    shapes: Shapes,
    config: reference_fixtures.RefConfig,
    mixed: []const f32,
    weights: LayerTwoWeights,
    tokens: u32,
    selected: []i32,
) !void {
    assert(tokens > 0);
    assert(selected.len == @as(usize, tokens) * shapes.slots());
    assert(mixed.len == @as(usize, tokens) * config.hidden_size);

    const hidden = try fixtures.alloc(f32, @as(usize, tokens) * config.hidden_size);
    @memcpy(hidden, mixed);
    roundTripBf16(hidden);
    const keys = try fixtures.alloc(f32, @as(usize, tokens) * shapes.index_dim);
    @memset(keys, 0.0);
    const positions = try fixtures.alloc(u32, tokens);
    for (positions, 0..) |*slot, token| {
        slot.* = @intCast(token);
    }
    const cos = try fixtures.alloc(f32, @as(usize, tokens) * shapes.rotary_dim);
    const sin = try fixtures.alloc(f32, @as(usize, tokens) * shapes.rotary_dim);
    try buildTables(harness, shapes, tokens, .{ 0, 0, 0 }, cos, sin);
    const input = IndexerInput{
        .shapes = shapes,
        .rows = tokens,
        .in_features = config.hidden_size,
        .arena_length = tokens,
        .hidden = hidden,
        .weight = weights.index_qk_proj,
        .q_layernorm = weights.q_layernorm,
        .k_layernorm = weights.k_layernorm,
        .index_keys = keys,
        .positions = positions,
        .cos = cos,
        .sin = sin,
    };
    try runIndexer(harness, input, selected, null, null, null, null);
    assert(positions[tokens - 1] == tokens - 1);
}

/// Runs the golden layer's attention row by row: each row gathers its own
/// selection out of the reference's K/V and calls the kernel once.
fn runGoldenAttention(
    harness: *Harness,
    fixtures: std.mem.Allocator,
    shapes: Shapes,
    built: GoldenProjections,
    selected: []const i32,
    tokens: u32,
) ![]f32 {
    assert(tokens > 0);
    assert(selected.len == @as(usize, tokens) * shapes.slots());

    const row_elements = @as(usize, shapes.heads) * shapes.head_dim;
    const produced = try fixtures.alloc(f32, @as(usize, tokens) * row_elements);
    for (0..tokens) |token| {
        try attendGatheredRow(
            harness,
            shapes,
            built.q[token * row_elements ..][0..row_elements],
            built.k,
            built.v,
            selected[token * shapes.slots() ..][0..shapes.slots()],
            @intCast(token),
            produced[token * row_elements ..][0..row_elements],
        );
    }
    return produced;
}

const oracle_hidden_size: u32 = 256;

/// KV arena of the oracle-driven tests: 4096 tokens is 1024 pooled blocks,
/// twice `block_topk`, so the top-k has a real choice to make.
const oracle_arena_length: u32 = 4096;

/// Queries the oracle-driven tests run.
const oracle_tokens: u32 = 8;

// Acceptance #2 of I-008: the indexer's `selected` lists must match the
// reference exactly, and the pooled key storage is bf16 on purpose.
//
// The model pools in f32 and then casts the result back to the cache dtype —
// `key_groups.float().mean(dim=1).to(raw_keys.dtype)` — which for this model IS
// bf16. Keeping the kernels' pooled keys bf16 therefore tracks the real bf16
// model *more* closely than our f32 CPU oracle does, and the ~0.1% of blocks
// that move when the two are compared at a top-k boundary are an artifact of
// comparing against an f32 oracle, not a kernel defect. Storing pooled keys in
// f32 instead would cost +0.93 ms/token at 262 144 tokens (about 20% of the
// decode budget) to chase that artifact, so the residual is accepted and the
// comparison against the f32 oracle is pinned at >= 0.99 set agreement
// (`overlap_floor`), while exact integer set equality is required against the
// reference wherever both sides use the same dtype — `ref/attention.zig` with
// `PooledDtype.cache_dtype`, and the fp32 transcription on identical inputs.
// Both exact gates run below and both report set_difference 0.
test "the indexer selection and the attention output match the fp32 oracle" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();
    var fixtures = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer fixtures.deinit();
    try checkOracleLayer(&harness, fixtures.allocator());
}

/// Runs one whole indexed-attention layer through `src/ref/attention.zig` and
/// checks the kernels against its selection and its output.
fn checkOracleLayer(
    harness: *Harness,
    allocator: std.mem.Allocator,
) !void {
    assert(allocator.vtable != undefined);
    assert(oracle_tokens > 0);

    const config = realConfig(oracle_hidden_size, oracle_arena_length);
    const shapes = shapesFromConfig(config);
    shapes.validate();
    try std.testing.expectEqual(
        @as(u32, 1024),
        oracle_arena_length / shapes.compress_ratio,
    );
    try std.testing.expect(shapes.blockTopk() < 1024);

    const weights = try oracleWeights(allocator, config, shapes);
    var layers = try oracleLayerState(allocator, config, shapes);
    defer layers.deinit(allocator);
    const table_cos = try allocator.alloc(
        f32,
        @as(usize, oracle_arena_length) * shapes.rotary_dim,
    );
    defer allocator.free(table_cos);
    const table_sin = try allocator.alloc(f32, table_cos.len);
    defer allocator.free(table_sin);
    try buildTables(harness, shapes, oracle_arena_length, .{ 0, 0, 0 }, table_cos, table_sin);

    var oracle = try runOracleLayer(allocator, config, shapes, weights, layers, table_cos, table_sin);
    defer oracle.deinit(allocator);
    try checkSelectionAgainstOracle(harness, config, shapes, weights, layers, table_cos, table_sin, oracle);
    try checkLayerAgainstOracle(harness, config, shapes, weights, oracle);
}

/// Everything one oracle layer forward produces that the checks consume.
const OracleLayer = struct {
    /// int32 `[tokens, slots]`, the reference's selection, and the one its own
    /// layer output was computed with.
    selected: []i32,
    /// int32 `[tokens, slots]`, re-selected from the storage the kernels really
    /// have: bf16 index keys, bf16 roped queries and bf16 pooled block keys
    /// (review-gate §5). This is the exact-equality gate.
    selected_storage: []i32,
    /// f32 `[tokens, heads * head_dim]`, the reference's post-attention gate.
    gate: []f32,
    /// f32 `[tokens * heads * head_dim]`, the reference's roped queries.
    query: []f32,
    /// f32 `[arena_length, kv_heads * head_dim]`, keys after the append.
    key: []f32,
    /// f32 `[arena_length, kv_heads * head_dim]`, values after the append.
    value: []f32,
    /// f32 `[tokens, hidden]`, the reference's layer output.
    output: []f32,
    /// f32 `[blocks]`, the reference's block scores for its last query, with
    /// the pooled keys in the cache dtype.
    block_scores: []f32,
    /// f32 `[blocks]`, the same row with f32 pooled keys: the size of the cast
    /// the model performs, which is the whole of the residual below.
    block_scores_f32: []f32,

    fn deinit(self: *OracleLayer, allocator: std.mem.Allocator) void {
        assert(self.selected.len > 0);
        assert(self.output.len > 0);

        allocator.free(self.block_scores_f32);
        allocator.free(self.block_scores);
        allocator.free(self.selected_storage);
        allocator.free(self.output);
        allocator.free(self.value);
        allocator.free(self.key);
        allocator.free(self.query);
        allocator.free(self.gate);
        allocator.free(self.selected);
    }
};

/// The buffers the oracle's `State` and `Scratch` own during a forward.
const OracleScratch = struct {
    state: reference_attention.State,
    scratch: reference_attention.Scratch,

    fn deinit(self: *OracleScratch, allocator: std.mem.Allocator) void {
        assert(self.state.key.len > 0);
        assert(self.scratch.mask.len > 0);

        self.scratch.deinit(allocator);
        self.state.deinit(allocator);
    }
};

/// Random layer-2-shaped weights on the host, BF16-rounded so the kernels and
/// the oracle read the same operands.
fn oracleWeights(
    allocator: std.mem.Allocator,
    config: reference_fixtures.RefConfig,
    shapes: Shapes,
) !reference_attention.Weights {
    assert(config.hidden_size > 0);
    assert(shapes.heads > 0);

    const hidden: usize = config.hidden_size;
    const head_dim: usize = shapes.head_dim;
    const q_rows: usize = shapes.heads * head_dim * 2;
    const kv_rows: usize = shapes.kv_heads * head_dim;
    const scale: f32 = 0.02;
    const q_proj = try randomValues(allocator, q_rows * hidden, 0x51ce_0a00);
    errdefer allocator.free(q_proj);
    const o_proj = try randomValues(
        allocator,
        hidden * shapes.heads * head_dim,
        0x51ce_0a01,
    );
    errdefer allocator.free(o_proj);
    const k_proj = try randomValues(allocator, kv_rows * hidden, 0x51ce_0a02);
    errdefer allocator.free(k_proj);
    const v_proj = try randomValues(allocator, kv_rows * hidden, 0x51ce_0a03);
    errdefer allocator.free(v_proj);
    const q_norm = try randomValues(allocator, head_dim, 0x51ce_0a04);
    errdefer allocator.free(q_norm);
    const k_norm = try randomValues(allocator, head_dim, 0x51ce_0a05);
    errdefer allocator.free(k_norm);
    const index_qk_proj = try randomValues(
        allocator,
        @as(usize, shapes.projectedWidth()) * hidden,
        0x51ce_0a06,
    );
    errdefer allocator.free(index_qk_proj);
    const q_layernorm = try randomValues(allocator, shapes.index_dim, 0x51ce_0a07);
    errdefer allocator.free(q_layernorm);
    const k_layernorm = try randomValues(allocator, shapes.index_dim, 0x51ce_0a08);
    errdefer allocator.free(k_layernorm);
    for ([_][]f32{
        q_proj,      o_proj, k_proj,        v_proj,
        q_norm,      k_norm, index_qk_proj, q_layernorm,
        k_layernorm,
    }) |values| {
        if (values.len > 64) {
            scaleValues(values, scale);
        } else {
            scaleValues(values, 0.1);
        }
        roundTripBf16(values);
    }
    return .{
        .q_proj = q_proj,
        .k_proj = k_proj,
        .v_proj = v_proj,
        .o_proj = o_proj,
        .q_norm = q_norm,
        .k_norm = k_norm,
        .index_qk_proj = index_qk_proj,
        .q_layernorm = q_layernorm,
        .k_layernorm = k_layernorm,
    };
}

/// The hidden states, the pre-filled caches and the positions of one layer.
const OracleInputs = struct {
    hidden: []f32,
    index_keys: []f32,
    key: []f32,
    value: []f32,
    positions: []u32,

    fn deinit(self: *OracleInputs, allocator: std.mem.Allocator) void {
        assert(self.hidden.len > 0);
        assert(self.positions.len > 0);

        allocator.free(self.positions);
        allocator.free(self.value);
        allocator.free(self.key);
        allocator.free(self.index_keys);
        allocator.free(self.hidden);
    }
};

/// Fills the layer input: BF16-rounded hidden states, pre-filled KV and index
/// key caches, and the positions of the last `oracle_tokens` rows.
fn oracleLayerState(
    allocator: std.mem.Allocator,
    config: reference_fixtures.RefConfig,
    shapes: Shapes,
) !OracleInputs {
    assert(oracle_tokens > 0);
    assert(oracle_arena_length > oracle_tokens);

    const kv_dim = @as(usize, shapes.kv_heads) * shapes.head_dim;
    const hidden = try randomValues(
        allocator,
        @as(usize, oracle_tokens) * config.hidden_size,
        0x51ce_0b00,
    );
    errdefer allocator.free(hidden);
    const index_keys = try randomValues(
        allocator,
        @as(usize, oracle_arena_length) * shapes.index_dim,
        0x51ce_0b01,
    );
    errdefer allocator.free(index_keys);
    const key = try randomValues(
        allocator,
        @as(usize, oracle_arena_length) * kv_dim,
        0x51ce_0b02,
    );
    errdefer allocator.free(key);
    const value = try randomValues(
        allocator,
        @as(usize, oracle_arena_length) * kv_dim,
        0x51ce_0b03,
    );
    errdefer allocator.free(value);
    const positions = try allocator.alloc(u32, oracle_tokens);
    errdefer allocator.free(positions);
    roundTripBf16(hidden);
    roundTripBf16(index_keys);
    roundTripBf16(key);
    roundTripBf16(value);
    for (positions, 0..) |*position, index| {
        position.* = oracle_arena_length - oracle_tokens + @as(u32, @intCast(index));
    }
    assert(positions[oracle_tokens - 1] == oracle_arena_length - 1);
    return .{
        .hidden = hidden,
        .index_keys = index_keys,
        .key = key,
        .value = value,
        .positions = positions,
    };
}

/// Runs the reference layer forward once and copies out what the checks need.
fn runOracleLayer(
    allocator: std.mem.Allocator,
    config: reference_fixtures.RefConfig,
    shapes: Shapes,
    weights: reference_attention.Weights,
    inputs: OracleInputs,
    table_cos: []const f32,
    table_sin: []const f32,
) !OracleLayer {
    assert(inputs.positions.len == oracle_tokens);
    assert(table_cos.len >= @as(usize, oracle_arena_length) * shapes.rotary_dim);

    const kv_dim = @as(usize, shapes.kv_heads) * shapes.head_dim;
    var scratch = OracleScratch{
        .state = try reference_attention.State.init(
            allocator,
            &config,
            oracle_arena_length,
        ),
        .scratch = undefined,
    };
    errdefer scratch.state.deinit(allocator);
    scratch.scratch = try reference_attention.Scratch.init(
        allocator,
        &config,
        oracle_arena_length,
    );
    defer scratch.deinit(allocator);
    const prefill = @as(usize, inputs.positions[0]) * shapes.index_dim;
    @memcpy(scratch.state.index_key[0..prefill], inputs.index_keys[0..prefill]);
    @memcpy(
        scratch.state.key[0 .. @as(usize, inputs.positions[0]) * kv_dim],
        inputs.key[0 .. @as(usize, inputs.positions[0]) * kv_dim],
    );
    @memcpy(
        scratch.state.value[0 .. @as(usize, inputs.positions[0]) * kv_dim],
        inputs.value[0 .. @as(usize, inputs.positions[0]) * kv_dim],
    );
    scratch.state.length = inputs.positions[0];
    const output = try allocator.alloc(
        f32,
        @as(usize, oracle_tokens) * config.hidden_size,
    );
    errdefer allocator.free(output);
    reference_attention.forward(
        inputs.hidden,
        oracle_tokens,
        inputs.positions,
        output,
        &weights,
        &scratch.state,
        &scratch.scratch,
        .{ .cos = table_cos, .sin = table_sin },
        &config,
    );
    // §5 of the review gate: both sides get identical inputs. The reference
    // keeps its caches and its roped queries in fp32; the kernels store them as
    // bf16, so the reference's copies are rounded to bf16 here (outside its
    // code) and its selection is recomputed from those. Its pooling, scoring
    // and top-k stay untouched fp32 reference code.
    const selected = try allocator.dupe(i32, scratch.scratch.selected[0 .. @as(usize, oracle_tokens) * shapes.slots()]);
    errdefer allocator.free(selected);
    const block_scores_f32 = try allocator.dupe(f32, scratch.scratch.block_scores);
    errdefer allocator.free(block_scores_f32);
    roundTripBf16(scratch.state.index_key);
    roundTripBf16(scratch.scratch.index_query);
    // The model pools in f32 and casts the pooled keys back to the cache dtype,
    // which is bf16 in the arena, before it norms and rotates them; the kernels
    // store them exactly so. Re-selecting with that cast puts both sides on the
    // same storage, and the selection then has to match exactly.
    reference_attention.forward_select(
        oracle_tokens,
        inputs.positions,
        &weights,
        &scratch.state,
        &scratch.scratch,
        .{ .cos = table_cos, .sin = table_sin },
        &config,
        .cache_dtype,
    );
    const result = OracleLayer{
        .selected = selected,
        .selected_storage = try allocator.dupe(i32, scratch.scratch.selected[0 .. @as(usize, oracle_tokens) * shapes.slots()]),
        .gate = try allocator.dupe(f32, scratch.scratch.gate[0 .. @as(usize, oracle_tokens) * shapes.heads * shapes.head_dim]),
        .query = try allocator.dupe(f32, scratch.scratch.query),
        .key = try allocator.dupe(f32, scratch.state.key),
        .value = try allocator.dupe(f32, scratch.state.value),
        .output = output,
        .block_scores = try allocator.dupe(f32, scratch.scratch.block_scores),
        .block_scores_f32 = block_scores_f32,
    };
    assert(result.selected.len == @as(usize, oracle_tokens) * shapes.slots());
    return result;
}

/// My indexer against the reference's selection.
///
/// Three comparisons, because they prove different things:
///
/// * `src/ref/attention.zig` fed the storage the kernels actually have (bf16
///   index keys, bf16 roped queries, bf16 pooled block keys) must select the
///   *same set*: that is the requirement-level gate, and it holds.
/// * A test-local fp32 transcription of the same pipeline, on the same inputs,
///   must also agree: it catches a change in the algorithm itself.
/// * The reference with fully fp32 pooled keys can only agree up to storage
///   precision (see the test's header): that agreement is pinned at
///   `overlap_floor` = 0.99 against an observed 0.999, which is 0.9% of headroom
///   under the measurement and far above what a broken score row produces.
fn checkSelectionAgainstOracle(
    harness: *Harness,
    config: reference_fixtures.RefConfig,
    shapes: Shapes,
    weights: reference_attention.Weights,
    inputs: OracleInputs,
    table_cos: []const f32,
    table_sin: []const f32,
    oracle: OracleLayer,
) !void {
    assert(inputs.hidden.len > 0);
    assert(oracle.selected.len > 0);

    const allocator = harness.allocator;
    const selected = try allocator.alloc(i32, oracle.selected.len);
    defer allocator.free(selected);
    const input = IndexerInput{
        .shapes = shapes,
        .rows = oracle_tokens,
        .in_features = config.hidden_size,
        .arena_length = oracle_arena_length,
        .hidden = inputs.hidden,
        .weight = weights.index_qk_proj,
        .q_layernorm = weights.q_layernorm,
        .k_layernorm = weights.k_layernorm,
        .index_keys = inputs.index_keys,
        .positions = inputs.positions,
        .cos = table_cos,
        .sin = table_sin,
    };
    try runIndexer(harness, input, selected, null, null, null, null);
    const transcribed = try allocator.alloc(i32, oracle.selected.len);
    defer allocator.free(transcribed);
    const blocks = oracle_arena_length / shapes.compress_ratio;
    const scores = try allocator.alloc(f32, @as(usize, oracle_tokens) * blocks);
    defer allocator.free(scores);
    @memset(scores, 0.0);
    try indexerOracle(allocator, input, transcribed, scores, null, null);
    const last_row = oracle_tokens - 1;
    const visible_blocks = (inputs.positions[last_row] + 1) / shapes.compress_ratio;
    // The transcription and the reference now run the same cast in the same
    // order, so this row is expected to be exactly zero; the tolerance only
    // admits last-bit differences in the reduction order.
    try reportError(
        "block scores: transcription vs reference",
        scores[last_row * blocks ..][0..visible_blocks],
        oracle.block_scores[0..visible_blocks],
        4.0e-3,
    );
    printRelativeDifference(
        "block scores: fp32 pooled vs cache dtype (cause of the drift)",
        oracle.block_scores_f32[0..visible_blocks],
        oracle.block_scores[0..visible_blocks],
    );
    // Exact gate: the same algorithm on the same storage precision.
    try expectSelection("indexer selection vs fp32 transcription", selected, transcribed);
    // Requirement-level gate: the reference itself, on the storage the kernels
    // have (bf16 cache dtype for the keys, the queries and the pooled keys).
    try expectSelection(
        "indexer selection vs reference, kernel storage",
        selected,
        oracle.selected_storage,
    );
    // The reference with fp32 pooled keys, which the real model does not have:
    // the blocks that move are the ones whose scores sit inside one bf16 ulp of
    // the top-k boundary. Pinned at `overlap_floor`.
    try reportOverlap(
        "selection vs f32-pooled oracle (floor 0.99)",
        selected,
        oracle.selected,
    );
}

/// My prefill attention against the oracle's layer output: the oracle's own
/// selection and roped queries drive the kernel, and the gate plus `o_proj`
/// (I-012/I-006a) are applied on the host.
fn checkLayerAgainstOracle(
    harness: *Harness,
    config: reference_fixtures.RefConfig,
    shapes: Shapes,
    weights: reference_attention.Weights,
    oracle: OracleLayer,
) !void {
    assert(oracle.query.len > 0);
    assert(oracle.output.len > 0);

    const allocator = harness.allocator;
    const heads = shapes.heads;
    const head_dim = shapes.head_dim;
    const row_elements = @as(usize, oracle_tokens) * heads * head_dim;
    const produced = try allocator.alloc(f32, row_elements);
    defer allocator.free(produced);
    for (0..oracle_tokens) |token| {
        try attendGatheredRow(
            harness,
            shapes,
            oracle.query[token * heads * head_dim ..][0 .. heads * head_dim],
            oracle.key,
            oracle.value,
            oracle.selected[token * shapes.slots() ..][0..shapes.slots()],
            @intCast(oracle_arena_length - oracle_tokens + @as(u32, @intCast(token))),
            produced[token * heads * head_dim ..][0 .. heads * head_dim],
        );
    }
    const gated = try allocator.alloc(f32, row_elements);
    defer allocator.free(gated);
    for (produced, oracle.gate, 0..) |value, gate, index| {
        gated[index] = value * ops.sigmoid(gate);
    }
    const layer_out = try allocator.alloc(
        f32,
        @as(usize, oracle_tokens) * config.hidden_size,
    );
    defer allocator.free(layer_out);
    ops.matmul(gated, weights.o_proj, layer_out, .{
        .rows = oracle_tokens,
        .in_features = heads * head_dim,
        .out_features = config.hidden_size,
    });
    try reportError(
        "whole layer vs fp32 oracle (bf16 kernels)",
        layer_out,
        oracle.output,
        4.0e-3,
    );
}

/// Gathers one row's selection out of a contiguous test source and runs
/// `attn_prefill` for that row alone.
///
/// Per-row gathered sets are what keep prefill exactly QSA when the queries of
/// a chunk select different tokens. A chunk that shares one selection (the
/// design's union gather) sets that selection as the working set and passes
/// `rows > 1` instead.
fn attendGatheredRow(
    harness: *Harness,
    shapes: Shapes,
    query: []const f32,
    key: []const f32,
    value: []const f32,
    selected: []const i32,
    position: u32,
    out: []f32,
) !void {
    assert(query.len == @as(usize, shapes.heads) * shapes.head_dim);
    assert(selected.len > 0);
    assert(out.len == query.len);

    const allocator = harness.allocator;
    const kv_dim = @as(usize, shapes.kv_heads) * shapes.head_dim;
    const gathered_k = try allocator.alloc(f32, selected.len * kv_dim);
    defer allocator.free(gathered_k);
    const gathered_v = try allocator.alloc(f32, selected.len * kv_dim);
    defer allocator.free(gathered_v);
    for (selected, 0..) |index, slot| {
        if (index < 0) {
            @memset(gathered_k[slot * kv_dim ..][0..kv_dim], 0.0);
            @memset(gathered_v[slot * kv_dim ..][0..kv_dim], 0.0);
            continue;
        }
        const source = @as(usize, @intCast(index)) * kv_dim;
        @memcpy(
            gathered_k[slot * kv_dim ..][0..kv_dim],
            key[source..][0..kv_dim],
        );
        @memcpy(
            gathered_v[slot * kv_dim ..][0..kv_dim],
            value[source..][0..kv_dim],
        );
    }
    roundTripBf16(gathered_k);
    roundTripBf16(gathered_v);
    const queries = try allocator.dupe(f32, query);
    defer allocator.free(queries);
    roundTripBf16(queries);
    const device_out = try harness.allocate(@as(u64, query.len) * 2);
    try harness.attention.prefill(.{
        .queries = try harness.uploadBf16(queries),
        .gathered_k = try harness.uploadBf16(gathered_k),
        .gathered_v = try harness.uploadBf16(gathered_v),
        .gather_index = try harness.uploadI32(selected),
        .positions = try harness.uploadI32(&[_]i32{@intCast(position)}),
        .out = device_out,
        .violations = null,
        .rows = 1,
        .count = @intCast(selected.len),
        .shapes = shapes,
    });
    try harness.downloadBf16(device_out, out);
    assert(out.len == queries.len);
}

// ---------------------------------------------------------------------------
// Timing and the tiny layer-2 weights.
// ---------------------------------------------------------------------------

/// Batches one timing sample: how many launches per sample, and how many
/// samples the median is taken over.
/// Odd, and at least the review gate's six samples, so the middle value is a
/// real median rather than one of two central samples.
const timing_samples: u32 = 7;
const timing_launches_per_sample: u32 = 4;

/// A launch `measure` can repeat; each driver binds one argument set.
const PrefillDriver = struct {
    args: PrefillArgs,

    fn run(self: PrefillDriver, harness: *Harness) !void {
        assert(self.args.rows > 0);
        assert(self.args.count > 0);
        try harness.attention.prefill(self.args);
    }
};

const DecodeDriver = struct {
    args: DecodeArgs,

    fn run(self: DecodeDriver, harness: *Harness) !void {
        assert(self.args.count > 0);
        assert(self.args.shapes.heads > 0);
        try harness.attention.decode(self.args);
    }
};

const IndexerDriver = struct {
    args: IndexerArgs,

    fn run(self: IndexerDriver, harness: *Harness) !void {
        assert(self.args.rows > 0);
        assert(self.args.arena_length > 0);
        try harness.attention.indexerScores(self.args);
    }
};

/// The single-CTA append stage, used as the timing calibration: its cost is one
/// launch and one row of work, so it measures the device, not the indexer.
const AppendDriver = struct {
    args: IndexerArgs,

    fn run(self: AppendDriver, harness: *Harness) !void {
        assert(self.args.rows > 0);
        try harness.attention.appendKeys(&self.args);
    }
};

/// The per-block score stage alone, so the cost report can name it.
const BlockScoreDriver = struct {
    args: IndexerArgs,
    first: u32,
    count: u32,
    capacity: u32,

    fn run(self: BlockScoreDriver, harness: *Harness) !void {
        assert(self.count > 0);
        assert(self.args.rows > 0);
        try harness.attention.scoreBlocks(
            &self.args,
            self.first,
            self.count,
            self.capacity,
        );
    }
};

/// One group's top-k merge, so the cost report can name it.
const GroupTopkDriver = struct {
    args: IndexerArgs,
    first: u32,
    count: u32,
    capacity: u32,

    fn run(self: GroupTopkDriver, harness: *Harness) !void {
        assert(self.count > 0);
        assert(self.args.rows > 0);
        try harness.attention.groupTopk(
            &self.args,
            self.first,
            self.count,
            self.capacity,
        );
    }
};

/// The final expansion of the partial top-k into the token list.
const ExpandDriver = struct {
    args: IndexerArgs,

    fn run(self: ExpandDriver, harness: *Harness) !void {
        assert(self.args.rows > 0);
        try harness.attention.expandSelected(&self.args);
    }
};

/// Median microseconds per launch of `driver.run`, over `timing_samples`
/// samples of `timing_launches_per_sample` launches each.
fn measure(harness: *Harness, driver: anytype) !f64 {
    assert(timing_samples > 0);
    assert(timing_launches_per_sample > 0);

    try driver.run(harness);
    try harness.attention.stream.synchronize();
    const start = try cuda_context.Event.create();
    defer start.destroy();
    const end = try cuda_context.Event.create();
    defer end.destroy();
    var measured = [_]f64{0.0} ** timing_samples;
    for (&measured) |*sample| {
        try start.record(&harness.attention.stream);
        for (0..timing_launches_per_sample) |_| {
            try driver.run(harness);
        }
        try end.record(&harness.attention.stream);
        try end.synchronize();
        const elapsed = try cuda_context.Event.elapsed_nanoseconds(&start, &end);
        sample.* = @as(f64, @floatFromInt(elapsed)) / 1000.0 /
            @as(f64, @floatFromInt(timing_launches_per_sample));
    }
    std.mem.sort(f64, measured[0..], {}, std.sort.asc(f64));
    const median = measured[timing_samples / 2];
    assert(median > 0.0);
    return median;
}

/// One projected tensor of the tiny layer-2 attention block.
const LayerTwoWeights = struct {
    q_proj: []const f32,
    k_proj: []const f32,
    v_proj: []const f32,
    o_proj: []const f32,
    q_norm: []const f32,
    k_norm: []const f32,
    index_qk_proj: []const f32,
    q_layernorm: []const f32,
    k_layernorm: []const f32,

    /// Reads every `self_attn` tensor of decoder layer 2.
    fn load(tensors: *const reference_fixtures.Tensors) !LayerTwoWeights {
        assert(tensors.entry_count > 0);
        assert(@sizeOf(LayerTwoWeights) > 0);

        const prefix = "model.language_model.layers.2.self_attn.";
        var name: [128]u8 = undefined;
        return .{
            .q_proj = try tensorF32(
                tensors,
                try std.fmt.bufPrint(&name, "{s}q_proj.weight", .{prefix}),
            ),
            .k_proj = try tensorF32(
                tensors,
                try std.fmt.bufPrint(&name, "{s}k_proj.weight", .{prefix}),
            ),
            .v_proj = try tensorF32(
                tensors,
                try std.fmt.bufPrint(&name, "{s}v_proj.weight", .{prefix}),
            ),
            .o_proj = try tensorF32(
                tensors,
                try std.fmt.bufPrint(&name, "{s}o_proj.weight", .{prefix}),
            ),
            .q_norm = try tensorF32(
                tensors,
                try std.fmt.bufPrint(&name, "{s}q_norm.weight", .{prefix}),
            ),
            .k_norm = try tensorF32(
                tensors,
                try std.fmt.bufPrint(&name, "{s}k_norm.weight", .{prefix}),
            ),
            .index_qk_proj = try tensorF32(
                tensors,
                try std.fmt.bufPrint(
                    &name,
                    "{s}indexer.index_qk_proj.weight",
                    .{prefix},
                ),
            ),
            .q_layernorm = try tensorF32(
                tensors,
                try std.fmt.bufPrint(
                    &name,
                    "{s}indexer.q_layernorm.weight",
                    .{prefix},
                ),
            ),
            .k_layernorm = try tensorF32(
                tensors,
                try std.fmt.bufPrint(
                    &name,
                    "{s}indexer.k_layernorm.weight",
                    .{prefix},
                ),
            ),
        };
    }
};

/// Views one checkpoint tensor as f32 through the reference reader.
fn tensorF32(
    tensors: *const reference_fixtures.Tensors,
    name: []const u8,
) ![]const f32 {
    assert(name.len > 0);
    assert(tensors.entry_count > 0);

    const tensor = try tensors.tensor(name);
    if (tensor.values.len == 0) {
        return error.EmptyTensor;
    }
    return tensor.values;
}

// ---------------------------------------------------------------------------
// The numerical gates: rope, the indexer selection, and the attention kernel
// at sweep and production magnitude.
// ---------------------------------------------------------------------------

/// The model's context limit: 262 144 tokens, i.e. 65 536 pooled index blocks
/// at `compress_ratio` 4. This is the scale the selection kernel has to run at.
const full_context_tokens: u32 = 262_144;

/// Layers of the model that run indexed (QSA) attention: 12 of 48.
const indexed_attention_layers: u32 = 12;

/// Per-token ceiling of the whole indexer at the context limit, across every
/// indexed layer, for a decode step that completes no new block.
///
/// The floor with replicated attention is 0.473 ms/token: each rank reads all 12
/// layers of 16.78 MB bf16 pooled keys at this machine's GDDR7 rate. Scoring a
/// quarter of the blocks per rank and all-gathering a 512-entry partial top-k
/// divides that to 0.118 ms/token, but it makes the indexer a distributed
/// concern and so belongs to I-014 (TP/EP).
///
/// The streaming score buffer (I-025) measured 3.18 ms/token with the score
/// buffer flat at 32 KB per row, so this bound is the regression guard the
/// issue permits: achieved + 10 %, 3 500 us/token, and not the 0.6 ms target.
/// The gap and the remaining fix are reported by `checkFullContextIndexerCost`,
/// which prints the three stages and their measured shares; each has a fix:
///
/// * `indexer_select` (streaming), 121 us: two radix selects over 32 768-block
///   groups plus the merge, still one CTA per row per group scanning the score
///   row digit by digit at ~20 GB/s. The fix is a two-phase radix (per-CTA
///   partial histograms and partial top-k merged by a second kernel) — total
///   ordering must stay descending-score with ties to the higher block index,
///   exactly as `torch.topk`; per-CTA counts have to sum to `block_topk` or the
///   token list comes out short; and the boundary has to be narrowed from the
///   merged candidate set, not per CTA. A previous attempt produced
///   `set_difference 2824 of 16396`, so it was reverted rather than landed
///   half-debugged; the selection set-equality gates in this file caught it.
///   That parallel radix is item 5 of I-025 and is not in this pass.
/// * `indexer_project`, 68 us: one dependent bf16 load per element over a 5 KB
///   strided weight row. The fix is the I-006a GEMM.
/// * `indexer_block_scores`, 76 us: still latency-bound at one block in flight
///   per warp. The streaming split costs one extra launch pair per group, which
///   is why the guard is set from the achieved number rather than the old 2.95
///   ms floor.
///
/// The guard is the full +10 % and not a tighter number because run-to-run
/// timing on this machine spreads by ~3 %: the probe bench runs concurrently
/// with the core tests under `zig build test`, so a 2 % guard was inside the
/// noise and failed a gate on one run of an unchanged tree.
const indexer_micros_per_token_limit: f64 = 3_500.0;

/// Prefill attention ceiling in microseconds per token per layer: the 5 000
/// tok/s target's 200 us/token budget over the 12 indexed layers. The
/// tensor-core path measured 4.83 us/token/layer at the production shape (256
/// rows, 2 048 gathered keys), down from 36.6 for the warp-scalar kernel. This
/// is the issue's gate itself, not an achieved + margin bound: the measured
/// value is in microseconds, so a 5.10 us/token run has to fail. The bound is
/// asserted only when a one-CTA calibration launch is clean; a contended device
/// inflates every number and the measured value is printed either way.
const prefill_micros_per_token_limit: f64 = 5.0;

/// Cost of one one-CTA launch on an idle device. `compute-sanitizer` and a
/// device shared with another lane both inflate every launch by an order of
/// magnitude, so the timing bounds are only asserted when this holds; the
/// measured numbers are printed either way.
const timing_clean_micros_max: f64 = 20.0;

/// The same ceiling for a cold cache, i.e. the first step after a prefill, which
/// pools every block in one launch. That is a per-prefill cost, not a per-token
/// one, so it is bounded separately and loosely.
const indexer_cold_micros_per_token_limit: f64 = 7_200.0;

/// One row of the attention sweep: working-set size, query rows, and the
/// absolute position of the last query.
const SweepCase = struct {
    count: u32,
    rows: u32,
    last_position: u32,
};

test "harness round trip crosses the staging window intact" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();

    const count: usize = 40 * 1024 * 1024;
    const source = try std.testing.allocator.alloc(u8, count);
    defer std.testing.allocator.free(source);
    for (source, 0..) |*byte, index| {
        byte.* = @truncate(index * 31 + 7);
    }
    const pointer = try harness.upload(source);
    const back = try std.testing.allocator.alloc(u8, count);
    defer std.testing.allocator.free(back);
    try harness.download(pointer, back);
    var differing: u32 = 0;
    for (source, back) |want, got| {
        if (want != got) {
            differing += 1;
        }
    }
    std.debug.print(
        "round trip: {d} of {d} bytes differ\n",
        .{ differing, count },
    );
    try std.testing.expectEqual(@as(u32, 0), differing);
}

test "rope tables and rotation match the fp32 reference" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();
    var fixtures = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer fixtures.deinit();

    const config = try reference_fixtures.load_config(fixtures.allocator());
    try checkRope(&harness, Shapes.real, 37);
    try checkRope(&harness, shapesFromConfig(config), 12);
}

test "the indexer selection matches the brute-force reference top-k" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();

    try checkIndexerSelection(&harness, Shapes.real, 4096, 8, 2560, 0x51ce_0100);
}

test "the indexer top-k breaks ties the way the reference does" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();

    // Same geometry as the model, two budgets: `take` inside the tie group, and
    // `take` exactly covering it.
    var inside = Shapes.real;
    inside.budget = 2 * Shapes.real.compress_ratio;
    inside.validate();
    try checkSelectionTieBreak(&harness, inside, tie_blocks);
    var covering = Shapes.real;
    covering.budget = 4 * Shapes.real.compress_ratio;
    covering.validate();
    try checkSelectionTieBreak(&harness, covering, tie_blocks);
}

test "a corrupt selection is refused on the host and counted on the device" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();

    try checkCorruptedSelection(&harness, Shapes.real);
}

test "the indexer rescores blocks on a step that completes no new block" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();

    // Real geometry and the real budget: 1024 blocks compete for 512 block
    // slots, so the selection is decided by the scores rather than by there being
    // fewer blocks than slots.
    try checkNoNewBlockStep(&harness, Shapes.real, 2 * Shapes.real.budget, 256, 0x51ce_0800);
}

test "the indexer selection scales to a full-context block count" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();
    try std.testing.expectEqual(
        @as(u32, 65_536),
        full_context_tokens / Shapes.real.compress_ratio,
    );
    try checkIndexerSelection(
        &harness,
        Shapes.real,
        full_context_tokens,
        3,
        2560,
        0x51ce_0600,
    );
}

test "attention matches the fp32 oracle across the working-set sweep" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();

    // Real head geometry throughout; the working-set size sweeps from a short
    // context to the full QSA budget, and the query block from one row (decode)
    // to a thousand.
    const sweep = [_]SweepCase{
        .{ .count = 64, .rows = 1, .last_position = 127 },
        .{ .count = 512, .rows = 1, .last_position = 1023 },
        .{ .count = 128, .rows = 256, .last_position = 1023 },
        .{ .count = 32, .rows = 1024, .last_position = 1023 },
        .{ .count = 2048, .rows = 1, .last_position = 4095 },
        .{ .count = 64, .rows = 256, .last_position = 4095 },
        .{ .count = 32, .rows = 1024, .last_position = 4095 },
    };
    for (sweep, 0..) |case, index| {
        const built = try makeAttentionCase(
            harness.allocator,
            Shapes.real,
            case.count,
            case.rows,
            case.last_position,
            0x51ce_0200 + index,
        );
        defer freeAttentionCase(built, harness.allocator);
        try checkAttentionCase(&harness, Shapes.real, built);
    }
}

test "attention at full context matches the fp32 oracle and the prefill row" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();

    // Production magnitude: the full QSA working set (2048 keys per query), the
    // model's head geometry, and absolute positions at the end of a 262 144
    // token context, where an indexing or off-by-one bug would show up.
    const case = try makeAttentionCase(
        std.testing.allocator,
        Shapes.real,
        Shapes.real.budget,
        1,
        full_context_tokens - 1,
        0x51ce_0700,
    );
    defer freeAttentionCase(case, std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 262_143), case.positions[0]);
    try checkAttentionCase(&harness, Shapes.real, case);
}

test "the indexer holds its per-token budget at the context limit" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();

    try checkFullContextIndexerCost(&harness);
}

test "the indexer at 4096 selects about the budget and decode stays fast" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();
    var fixtures = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer fixtures.deinit();
    try checkLongContextSelection(&harness, fixtures.allocator());
}

/// Times the six indexer stages at the context limit and checks the per-token
/// cost of every indexed layer against `indexer_micros_per_token_limit`.
fn checkFullContextIndexerCost(harness: *Harness) !void {
    assert(indexed_attention_layers > 0);
    assert(indexer_micros_per_token_limit > 0.0);

    const shapes = Shapes.real;
    const arena_length = full_context_tokens;
    const allocator = harness.allocator;
    shapes.validate();
    const blocks = arena_length / shapes.compress_ratio;
    try std.testing.expectEqual(@as(u32, 65_536), blocks);
    var random = try RandomIndexer.make(
        harness,
        shapes,
        arena_length,
        1,
        2560,
        0x51ce_0a00,
    );
    defer random.deinit(allocator);
    const selected = try allocator.alloc(i32, shapes.slots());
    defer allocator.free(selected);
    var args: IndexerArgs = undefined;
    try runIndexer(harness, random.view(), selected, null, &args, null, null);
    // One-off: a prefill that pools every block in a single launch. An engine
    // pools a block once, so this cost is not the per-token one; it is printed
    // because it is what a cold cache pays.
    const cold_micros = try measure(harness, IndexerDriver{ .args = args });
    // Steady state: the decode step that completes no new block, which is three
    // of every four steps at `compress_ratio` 4. The pooled keys are resident, so
    // this is the projection, the append, the block scores and the selection.
    args.first_block = blocks;
    const micros = try measure(harness, IndexerDriver{ .args = args });
    // One streaming group's score and merge, scaled by the group count: the
    // wrapper launches one of each per group, so this is the stage's share of
    // the whole call.
    const capacity = @min(blocks, score_group_blocks);
    const groups = (blocks + score_group_blocks - 1) / score_group_blocks;
    const first = blocks - capacity;
    const scored_group = try measure(harness, BlockScoreDriver{
        .args = args,
        .first = first,
        .count = capacity,
        .capacity = capacity,
    });
    const merged_group = try measure(harness, GroupTopkDriver{
        .args = args,
        .first = first,
        .count = capacity,
        .capacity = capacity,
    });
    const expanded = try measure(harness, ExpandDriver{ .args = args });
    const scored = scored_group * @as(f64, @floatFromInt(groups));
    const selected_micros = merged_group * @as(f64, @floatFromInt(groups)) +
        expanded;
    assert(scored > 0.0);
    assert(selected_micros > 0.0);
    const per_token = micros * @as(f64, @floatFromInt(indexed_attention_layers));
    const cold_per_token = cold_micros *
        @as(f64, @floatFromInt(indexed_attention_layers));
    const fixed = micros - scored - selected_micros;
    const score_bytes = Attention.indexerScoreBytes(args.rows, blocks);
    std.debug.print(
        "indexer at {d} blocks ({d} groups of {d}): score buffer {d} B " ++
            "flat; steady state {d:.1} us = block_scores {d:.1} + select {d:.1} + " ++
            "project/norm/append {d:.1} ({d:.1} us/token across {d} layers, " ++
            "limit {d:.0}); cold cache {d:.1} us ({d:.1} us/token, limit {d:.0})\n",
        .{
            blocks,
            groups,
            capacity,
            score_bytes,
            micros,
            scored,
            selected_micros,
            fixed,
            per_token,
            indexed_attention_layers,
            indexer_micros_per_token_limit,
            cold_micros,
            cold_per_token,
            indexer_cold_micros_per_token_limit,
        },
    );
    try std.testing.expectEqual(@as(u32, 1), args.rows);
    const calibration = try measure(harness, AppendDriver{ .args = args });
    if (calibration > timing_clean_micros_max) {
        std.debug.print(
            "timing bound skipped: one launch costs {d:.1} us (clean " ++
                "{d:.1} us), so the device is instrumented or contended\n",
            .{ calibration, timing_clean_micros_max },
        );
        return;
    } else {
        assert(calibration <= timing_clean_micros_max);
    }
    try std.testing.expect(per_token <= indexer_micros_per_token_limit);
    try std.testing.expect(cold_per_token <= indexer_cold_micros_per_token_limit);
}

/// Runs the indexer at 4096 tokens, feeds its selection straight into the
/// gather list of both attention kernels, and reports the selection count, the
/// decode/prefill invariant and the per-kernel times.
fn checkLongContextSelection(
    harness: *Harness,
    allocator: std.mem.Allocator,
) !void {
    assert(allocator.vtable != undefined);
    const shapes = Shapes.real;
    const arena_length: u32 = 4096;
    shapes.validate();
    assert(arena_length > shapes.budget);
    assert(@intFromPtr(harness.attention.stream.handle) != 0);

    var random = try RandomIndexer.make(
        harness,
        shapes,
        arena_length,
        1,
        2560,
        0x51ce_0300,
    );
    defer random.deinit(harness.allocator);
    const selected = try allocator.alloc(i32, shapes.slots());
    var indexer_args: IndexerArgs = undefined;
    try runIndexer(harness, random.view(), selected, null, &indexer_args, null, null);
    const count = selectedCount(selected);
    std.debug.print(
        "S={d} budget={d}: indexer selected {d} tokens of {d} visible\n",
        .{ arena_length, shapes.budget, count, arena_length },
    );
    try std.testing.expect(count > 0);
    try std.testing.expect(count <= shapes.slots());
    try std.testing.expect(count + shapes.compress_ratio - 1 >= shapes.budget);
    try std.testing.expect(count < arena_length);

    // The indexer's selection *is* the gather list: each entry is an absolute
    // token position, and the attention kernel never learns where that token
    // lives.
    const case = try makeAttentionCase(
        harness.allocator,
        shapes,
        shapes.slots(),
        1,
        arena_length - 1,
        0x51ce_0400,
    );
    defer freeAttentionCase(case, harness.allocator);
    @memcpy(case.gather_index, selected);
    try checkAttentionCase(harness, shapes, case);
    try reportDecodeLatency(harness, shapes, case);
    const indexer_micros = try measure(harness, IndexerDriver{ .args = indexer_args });
    std.debug.print(
        "indexer_scores S={d} rows={d}: {d:.1} us per launch\n",
        .{ arena_length, indexer_args.rows, indexer_micros },
    );
    const wide = try makeAttentionCase(
        harness.allocator,
        shapes,
        shapes.budget,
        256,
        arena_length - 1,
        0x51ce_0500,
    );
    defer freeAttentionCase(wide, harness.allocator);
    const prefill_micros = try reportPrefillLatency(harness, shapes, wide);
    const per_token = prefill_micros / @as(f64, @floatFromInt(wide.rows()));
    const calibration = try measure(harness, AppendDriver{ .args = indexer_args });
    if (calibration > timing_clean_micros_max) {
        std.debug.print(
            "prefill bound skipped: one launch costs {d:.1} us (clean " ++
                "{d:.1} us), so the device is instrumented or contended\n",
            .{ calibration, timing_clean_micros_max },
        );
    } else {
        try std.testing.expect(per_token <= prefill_micros_per_token_limit);
    }
    try std.testing.expect(indexer_micros > 0.0);
}

/// Runs one union-gather case with a per-row selection bitmap and checks the
/// output against the reference's per-row masked attention.
///
/// The gathered working set is a 2048-token arena; each row selects only a
/// slice of it, so the bitmap is what makes the kernel compute this row's QSA
/// rather than the union's. The same launch without a bitmap is the union
/// baseline: it must differ, or the test would not be measuring the mask.
fn checkRowSelectionExactness(
    harness: *Harness,
    shapes: Shapes,
    rows: u32,
    selected_per_row: u32,
    seed: u64,
) !void {
    assert(rows > 0);
    assert(selected_per_row > 0);
    assert(selected_per_row <= shapes.budget);

    const allocator = harness.allocator;
    const count = shapes.budget;
    const kv_dim = @as(usize, shapes.kv_heads) * shapes.head_dim;
    const positions = try allocator.alloc(i32, rows);
    defer allocator.free(positions);
    const gather_index = try allocator.alloc(i32, count);
    defer allocator.free(gather_index);
    const gathered_k = try randomValues(
        allocator,
        @as(usize, count) * kv_dim,
        seed,
    );
    defer allocator.free(gathered_k);
    const gathered_v = try randomValues(
        allocator,
        @as(usize, count) * kv_dim,
        seed + 1,
    );
    defer allocator.free(gathered_v);
    const queries = try randomValues(
        allocator,
        @as(usize, rows) * shapes.heads * shapes.head_dim,
        seed + 2,
    );
    defer allocator.free(queries);
    roundTripBf16(gathered_k);
    roundTripBf16(gathered_v);
    roundTripBf16(queries);
    for (gather_index, 0..) |*slot, index| slot.* = @intCast(index);
    @memset(positions, @intCast(count - 1));

    // Each row selects a deterministic stride through the union; the union over
    // the chunk covers it. The bitmap is the one the engine would build from
    // the indexer's per-row list.
    const mask_words = (count + 31) / 32;
    const mask = try allocator.alloc(u32, @as(usize, rows) * mask_words);
    defer allocator.free(mask);
    @memset(mask, 0);
    const union_seen = try allocator.alloc(bool, count);
    defer allocator.free(union_seen);
    @memset(union_seen, false);
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (0..rows) |row| {
        var picked: u32 = 0;
        while (picked < selected_per_row) {
            const slot = random.uintLessThan(u32, count);
            const word = row * mask_words + slot / 32;
            const bit: u32 = @as(u32, 1) << @intCast(slot % 32);
            if (mask[word] & bit != 0) {
                continue;
            }
            mask[word] |= bit;
            union_seen[slot] = true;
            picked += 1;
        }
    }
    var union_count: u32 = 0;
    for (union_seen) |seen| {
        if (seen) union_count += 1;
    }

    const device_mask = try harness.upload(std.mem.sliceAsBytes(mask));
    const encoded_positions = try allocator.alloc(i32, rows);
    defer allocator.free(encoded_positions);
    for (positions, 0..) |position, row| encoded_positions[row] = position;
    const row_elements = @as(usize, shapes.heads) * shapes.head_dim;
    const device_out = try harness.allocate(@as(u64, queries.len) * 2);
    try harness.attention.prefill(.{
        .queries = try harness.uploadBf16(queries),
        .gathered_k = try harness.uploadBf16(gathered_k),
        .gathered_v = try harness.uploadBf16(gathered_v),
        .gather_index = try harness.uploadI32(gather_index),
        .positions = try harness.uploadI32(encoded_positions),
        .out = device_out,
        .violations = null,
        .selection_mask = device_mask,
        .rows = rows,
        .count = count,
        .shapes = shapes,
    });
    const actual = try allocator.alloc(f32, queries.len);
    defer allocator.free(actual);
    try harness.downloadBf16(device_out, actual);

    // Reference: each row's own masked softmax over the selected slots.
    const expected = try allocator.alloc(f32, queries.len);
    defer allocator.free(expected);
    const sub_k = try allocator.alloc(f32, @as(usize, selected_per_row) * kv_dim);
    defer allocator.free(sub_k);
    const sub_v = try allocator.alloc(f32, @as(usize, selected_per_row) * kv_dim);
    defer allocator.free(sub_v);
    const sub_index = try allocator.alloc(i32, selected_per_row);
    defer allocator.free(sub_index);
    for (0..rows) |row| {
        var written: u32 = 0;
        for (0..count) |slot| {
            const word = row * mask_words + slot / 32;
            if (((mask[word] >> @as(u5, @intCast(slot % 32))) & 1) == 0) {
                continue;
            }
            @memcpy(
                sub_k[written * kv_dim ..][0..kv_dim],
                gathered_k[slot * kv_dim ..][0..kv_dim],
            );
            @memcpy(
                sub_v[written * kv_dim ..][0..kv_dim],
                gathered_v[slot * kv_dim ..][0..kv_dim],
            );
            sub_index[written] = @intCast(slot);
            written += 1;
        }
        try std.testing.expectEqual(selected_per_row, written);
        const query = queries[row * row_elements ..][0..row_elements];
        try attendOracle(
            allocator,
            query,
            sub_k,
            sub_v,
            sub_index,
            @as([]const u32, &.{count - 1}),
            expected[row * row_elements ..][0..row_elements],
            shapes,
        );
    }
    std.debug.print(
        "row mask rows={d} union={d} budget={d} union/budget {d:.3} " ++
            "selected/row={d}\n",
        .{
            rows,
            union_count,
            shapes.budget,
            @as(f64, @floatFromInt(union_count)) /
                @as(f64, @floatFromInt(shapes.budget)),
            selected_per_row,
        },
    );
    try reportError("row mask prefill vs per-row oracle", actual, expected, 4.0e-3);

    // The union baseline, with no mask: it must differ, otherwise the bitmap
    // changed nothing and this test is vacuous.
    const union_out = try harness.allocate(@as(u64, queries.len) * 2);
    try harness.attention.prefill(.{
        .queries = try harness.uploadBf16(queries),
        .gathered_k = try harness.uploadBf16(gathered_k),
        .gathered_v = try harness.uploadBf16(gathered_v),
        .gather_index = try harness.uploadI32(gather_index),
        .positions = try harness.uploadI32(encoded_positions),
        .out = union_out,
        .violations = null,
        .rows = rows,
        .count = count,
        .shapes = shapes,
    });
    const union_actual = try allocator.alloc(f32, queries.len);
    defer allocator.free(union_actual);
    try harness.downloadBf16(union_out, union_actual);
    var largest: f32 = 0.0;
    for (union_actual, actual) |got, want| {
        largest = @max(largest, @abs(got - want));
    }
    std.debug.print(
        "row mask: union-only differs by max_abs {e:.3} (must be nonzero)\n",
        .{largest},
    );
    try std.testing.expect(largest > 4.0e-3);
}

/// The indexer's selection at multiple row counts, against
/// `src/ref/attention.zig` itself with `PooledDtype.cache_dtype`: the reference
/// runs on the same bf16 index keys, bf16 roped queries and bf16 pooled block
/// keys the kernels store, so the two selections have to agree exactly.
fn checkSelectionAgainstReferenceRows(
    harness: *Harness,
    shapes: Shapes,
    rows: u32,
    in_features: u32,
    seed: u64,
) !void {
    assert(rows > 0);
    assert(in_features > 0);

    const allocator = harness.allocator;
    const arena: u32 = 4096;
    shapes.validate();
    const config = realConfig(in_features, arena);
    var random = try RandomIndexer.make(
        harness,
        shapes,
        arena,
        rows,
        in_features,
        seed,
    );
    defer random.deinit(allocator);
    const weights = reference_attention.Weights{
        .q_proj = &[_]f32{},
        .k_proj = &[_]f32{},
        .v_proj = &[_]f32{},
        .o_proj = &[_]f32{},
        .q_norm = &[_]f32{},
        .k_norm = &[_]f32{},
        .index_qk_proj = random.weight,
        .q_layernorm = random.q_layernorm,
        .k_layernorm = random.k_layernorm,
    };
    var state = try reference_attention.State.init(allocator, &config, arena);
    defer state.deinit(allocator);
    var scratch = try reference_attention.Scratch.init(allocator, &config, arena);
    defer scratch.deinit(allocator);
    const first = random.positions[0];
    @memcpy(
        state.index_key[0 .. @as(usize, first) * shapes.index_dim],
        random.index_keys[0 .. @as(usize, first) * shapes.index_dim],
    );
    state.length = first;
    reference_attention.forward_indexer(
        random.hidden,
        rows,
        random.positions,
        &weights,
        &state,
        &scratch,
        .{ .cos = random.cos, .sin = random.sin },
        &config,
    );
    // The kernels store the caches and the queries as bf16; round the
    // reference's copies outside its code so both sides select from the same
    // storage (review-gate §5), then re-select with the cache dtype.
    roundTripBf16(state.index_key);
    roundTripBf16(scratch.index_query);
    reference_attention.forward_select(
        rows,
        random.positions,
        &weights,
        &state,
        &scratch,
        .{ .cos = random.cos, .sin = random.sin },
        &config,
        .cache_dtype,
    );
    const count = @as(usize, rows) * shapes.slots();
    const expected = try allocator.alloc(i32, count);
    defer allocator.free(expected);
    @memcpy(expected, scratch.selected[0..count]);
    const selected = try allocator.alloc(i32, count);
    defer allocator.free(selected);
    try runIndexer(harness, random.view(), selected, null, null, null, null);
    var label: [96]u8 = undefined;
    try expectSelection(
        try std.fmt.bufPrint(&label, "indexer selection vs reference, rows={d}", .{rows}),
        selected,
        expected,
    );
}

test "the indexer selection matches the reference at rows above one" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();

    for ([_]u32{ 1, 64, 256 }) |rows| {
        try checkSelectionAgainstReferenceRows(
            &harness,
            Shapes.real,
            rows,
            256,
            0x51ce_1000 + rows,
        );
    }
}

test "the per-row bitmap makes prefill exactly QSA at rows above one" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();

    // Real head geometry and the real budget, so the union/budget ratio is the
    // production one; each row keeps a 64-token slice of the 2048-token union.
    for ([_]u32{ 1, 64, 256 }) |rows| {
        try checkRowSelectionExactness(&harness, Shapes.real, rows, 64, 0x51ce_0e00 + rows);
    }
}

/// The contexts the score-buffer gate reports: 4 K, 32 K and the model limit.
const score_buffer_contexts = [_]u32{ 4096, 32_768, 262_144 };

test "the indexer score buffer stays flat as the context grows" {
    var harness = try Harness.open(std.testing.allocator);
    defer harness.deinit();

    // The buffer is a fixed row chunk by a fixed block group, so its bytes are
    // the same at every context. The old shape was `rows * blocks * 4`, which
    // at 4096 rows and 256 K is the 1.07 GB the issue opens with.
    var previous: u64 = 0;
    for (score_buffer_contexts) |context| {
        const blocks = context / Shapes.real.compress_ratio;
        const bytes = Attention.indexerScoreBytes(1, blocks);
        const wide = Attention.indexerScoreBytes(indexer_row_chunk, blocks);
        std.debug.print(
            "indexer score buffer at {d} tokens ({d} blocks): {d} B for one " ++
                "row, {d} B at {d} rows (old shape would be {d} B)\n",
            .{
                context,
                blocks,
                bytes,
                wide,
                indexer_row_chunk,
                @as(u64, indexer_row_chunk) * blocks * 4,
            },
        );
        if (previous != 0) {
            try std.testing.expectEqual(previous, bytes);
        }
        previous = bytes;
        try std.testing.expectEqual(
            @as(u64, indexer_row_chunk) * score_group_blocks * 4,
            wide,
        );
    }

    // And the streaming path is exact across groups: 163 840 tokens is two
    // 32 768-block groups, so the merge really runs.
    try checkIndexerSelection(&harness, Shapes.real, 163_840, 3, 2560, 0x51ce_0f00);
}
