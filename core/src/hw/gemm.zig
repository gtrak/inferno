//! Typed launches for the prefill GEMM family in `src/kernels/gemm.cu`.
//!
//! Each wrapper owns the launch geometry of one kernel and the shape checks that
//! kernel requires, so a caller cannot launch with a K that the tile does not
//! divide or an M that overflows the grid. The block sizes here are the ones the
//! CUDA file asserts at compile time: 128 threads (four warps, one 16-row strip
//! each) over 64x64 output tiles.

const std = @import("std");

const cuda_context = @import("cuda_context.zig");
const cuda_driver = @import("cuda_driver.zig");
const kernel = @import("kernel.zig");

const assert = std.debug.assert;

/// A launch can fail either in the load/launch layer or in the driver layer
/// (allocation, pinned staging, stream sync).
pub const KernelError = kernel.KernelError || cuda_driver.DriverError;

/// Threads per block, matching `threads_per_block` in `gemm.cu`.
pub const threads_per_block: u32 = 128;

/// Output tile edges, matching `tile_m` and `tile_n` in `gemm.cu`.
pub const tile_m: u32 = 64;
pub const tile_n: u32 = 64;

/// Contraction depth of one native FP4 MMA, i.e. the k step of the fast path.
pub const nvfp4_k: u32 = 64;

/// Weights sharing one ue4m3 scale, i.e. the `.scale_vec::4X` block.
pub const nvfp4_block: u32 = 16;

/// Rows the blocked kernels round `M` up to.
pub const m_granularity: u32 = 64;

/// Largest M the prefill path supports (decode dispatches to the GEMV family).
pub const m_max: u32 = 8192;

/// Words of the repacked weight fragment buffer for `[N, K]`.
///
/// The layout contract itself is defined in `weights/pack_nvfp4.zig`; these
/// three functions exist so the driver layer can size a buffer without
/// importing the weight layer (the probe compiles `hw/` on its own). The
/// harness asserts the two spellings agree.
pub fn weightFragmentWords(n: u32, k: u32) u64 {
    assert(k % nvfp4_k == 0);
    assert(n % (nvfp4_k / 8) == 0);

    const tiles: u64 = @as(u64, k / nvfp4_k) * @as(u64, n / 8);
    return tiles * 32 * 2;
}

/// Words of the repacked weight scale buffer for `[N, K]`.
pub fn weightScaleWords(n: u32, k: u32) u64 {
    assert(k % nvfp4_k == 0);
    assert(n % 8 == 0);

    const tiles: u64 = @as(u64, k / nvfp4_k) * @as(u64, n / 8);
    return tiles * 8;
}

/// Words of the quantised activation fragment buffer for `[M, K]`.
pub fn activationFragmentWords(m: u32, k: u32) u64 {
    assert(m % 16 == 0);
    assert(k % nvfp4_k == 0);

    const tiles: u64 = @as(u64, m / 16) * @as(u64, k / nvfp4_k);
    return tiles * 32 * 4;
}

/// Words of the activation scale buffer for `[M, K]`.
pub fn activationScaleWords(m: u32, k: u32) u64 {
    assert(m % 16 == 0);
    assert(k % nvfp4_k == 0);

    const tiles: u64 = @as(u64, m / 16) * @as(u64, k / nvfp4_k);
    return tiles * 32;
}

/// Matrix shape of one GEMM: `A[m][k] x B[n][k] -> C[m][n]`.
pub const Shape = struct {
    m: u32,
    n: u32,
    k: u32,

    /// Rejects shapes the tiles cannot cover. The real model satisfies these;
    /// a caller that does not is a bug, not a case to handle.
    pub fn validate(shape: Shape) KernelError!void {
        assert(m_max > 0);
        assert(nvfp4_k > 0);

        if (shape.m == 0 or shape.m > m_max) {
            return KernelError.invalid_value;
        } else if (shape.n % tile_n != 0) {
            return KernelError.invalid_value;
        } else if (shape.k % nvfp4_k != 0) {
            return KernelError.invalid_value;
        } else {
            assert(shape.k % nvfp4_k == 0);
        }
    }

    /// `M` rounded up to a whole number of block tiles: the last tile is
    /// padded with zero rows and its output dropped.
    pub fn paddedRows(shape: Shape) u32 {
        assert(shape.m > 0);
        assert(shape.m <= m_max);

        const tiles = (shape.m + m_granularity - 1) / m_granularity;
        assert(tiles * m_granularity >= shape.m);
        return tiles * m_granularity;
    }
};

/// The five kernels of `gemm.cu`, resolved once and launched many times.
pub const Kernels = struct {
    module: kernel.Module,
    quantize_activations: kernel.Function,
    nvfp4_mma: kernel.Function,
    dequantize_weights: kernel.Function,
    fp8_e4m3: kernel.Function,
    bf16: kernel.Function,

    /// Loads a cubin image (the `gemm` entry of the generated table).
    pub fn init(image: []const u8) KernelError!Kernels {
        assert(image.len > 0);
        assert(image.len % 4 == 0);

        const module = try kernel.Module.load(image);
        return .{
            .module = module,
            .quantize_activations = try module.function("quantize_activations_nvfp4"),
            .nvfp4_mma = try module.function("gemm_nvfp4_mma"),
            .dequantize_weights = try module.function("dequantize_nvfp4_weights"),
            .fp8_e4m3 = try module.function("gemm_fp8_e4m3"),
            .bf16 = try module.function("gemm_bf16"),
        };
    }

    /// Unloads the module; every kernel handle becomes invalid.
    pub fn deinit(kernels: *const Kernels) void {
        assert(@intFromPtr(kernels.module.handle) != 0);
        assert(@intFromPtr(kernels.bf16.handle) != 0);

        kernels.module.unload();
    }

    /// Grid for a 64x64-tiled kernel over `shape`: one block per output tile.
    pub fn tiledGrid(shape: Shape) [3]u32 {
        assert(shape.n % tile_n == 0);
        assert(shape.k % nvfp4_k == 0);

        return .{ shape.n / tile_n, shape.paddedRows() / tile_m, 1 };
    }
};

/// Bytes of one pinned staging allocation are a multiple of this.
pub const staging_page_bytes: u64 = 4096;

/// Rounds a byte count up to a whole staging page.
fn stagingBytes(bytes: u64) u64 {
    assert(staging_page_bytes > 0);
    assert(bytes > 0);

    return (bytes + staging_page_bytes - 1) / staging_page_bytes *
        staging_page_bytes;
}

/// Uploads `bytes` into `destination` through a pinned staging buffer.
///
/// `cuMemcpyHtoDAsync` requires page-locked source memory, so every transfer in
/// this tree stages through one pinned allocation.
pub fn upload(
    stream: *const cuda_context.Stream,
    destination: *const cuda_context.DeviceBuffer,
    bytes: []const u8,
) KernelError!void {
    assert(bytes.len > 0);
    assert(bytes.len <= destination.bytes);

    const staged = try cuda_context.PinnedHostBuffer.allocate(
        stagingBytes(bytes.len),
    );
    defer staged.free();
    const slice = staged.slice();
    @memcpy(slice[0..bytes.len], bytes);
    try stream.copy_h_to_d(destination, &staged, bytes.len);
}

/// Downloads `bytes` from `source` into `destination` through pinned memory.
pub fn download(
    stream: *const cuda_context.Stream,
    destination: []u8,
    source: *const cuda_context.DeviceBuffer,
) KernelError!void {
    assert(destination.len > 0);
    assert(destination.len <= source.bytes);

    const staged = try cuda_context.PinnedHostBuffer.allocate(
        stagingBytes(destination.len),
    );
    defer staged.free();
    try stream.copy_d_to_h(&staged, source, destination.len);
    try stream.synchronize();
    @memcpy(destination, staged.slice()[0..destination.len]);
}

/// Launches the activation quantiser: bf16 [M,K] plus its fragment-order
/// scratch buffers produced by `weights/pack_nvfp4.zig`.
pub fn quantizeActivations(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    a: *const cuda_context.DeviceBuffer,
    a_frag: *const cuda_context.DeviceBuffer,
    a_scales: *const cuda_context.DeviceBuffer,
    shape: Shape,
) KernelError!void {
    assert(a.bytes > 0);
    assert(a_frag.bytes > 0);
    assert(a_scales.bytes > 0);

    try shape.validate();
    var m_live: i32 = @intCast(shape.m);
    var k_columns: i32 = @intCast(shape.k);
    var a_pointer = a.pointer;
    var frag_pointer = a_frag.pointer;
    var scale_pointer = a_scales.pointer;
    var arguments = [_]?*anyopaque{
        &a_pointer,
        &m_live,
        &k_columns,
        &frag_pointer,
        &scale_pointer,
    };
    const grid = [3]u32{ shape.k / nvfp4_k, shape.paddedRows() / 16, 1 };
    const block = [3]u32{ 32, 1, 1 };
    try kernels.quantize_activations.launch(grid, block, 0, stream.handle, &arguments);
}

/// Launches the native block-scaled FP4 GEMM (W4A4).
pub fn launchNvfp4Mma(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    a_frag: *const cuda_context.DeviceBuffer,
    a_scales: *const cuda_context.DeviceBuffer,
    b_frag: *const cuda_context.DeviceBuffer,
    b_scales: *const cuda_context.DeviceBuffer,
    scale2: f32,
    c: *const cuda_context.DeviceBuffer,
    shape: Shape,
) KernelError!void {
    assert(b_frag.bytes > 0);
    assert(c.bytes > 0);

    try shape.validate();
    var a_frag_pointer = a_frag.pointer;
    var a_scale_pointer = a_scales.pointer;
    var b_frag_pointer = b_frag.pointer;
    var b_scale_pointer = b_scales.pointer;
    var weight_scale = scale2;
    var c_pointer = c.pointer;
    var m_rows: i32 = @intCast(shape.m);
    var n_columns: i32 = @intCast(shape.n);
    var k_columns: i32 = @intCast(shape.k);
    var arguments = [_]?*anyopaque{
        &a_frag_pointer,
        &a_scale_pointer,
        &b_frag_pointer,
        &b_scale_pointer,
        &weight_scale,
        &c_pointer,
        &m_rows,
        &n_columns,
        &k_columns,
    };
    try kernels.nvfp4_mma.launch(
        Kernels.tiledGrid(shape),
        .{ threads_per_block, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Launches the weight dequantiser: packed FP4 plus ue4m3 scales to bf16.
///
/// `destination` holds `N * K` bf16 values. The conversion is lossless, so the
/// bf16 buffer is the same matrix the checkpoint describes.
pub fn launchDequantizeWeights(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    b: *const cuda_context.DeviceBuffer,
    b_scales: *const cuda_context.DeviceBuffer,
    destination: *const cuda_context.DeviceBuffer,
    shape: Shape,
) KernelError!void {
    assert(b.bytes > 0);
    assert(destination.bytes >= @as(u64, shape.n) * shape.k * 2);

    try shape.validate();
    var b_pointer = b.pointer;
    var scale_pointer = b_scales.pointer;
    var out_pointer = destination.pointer;
    var n_columns: i32 = @intCast(shape.n);
    var k_columns: i32 = @intCast(shape.k);
    var arguments = [_]?*anyopaque{
        &b_pointer,
        &scale_pointer,
        &out_pointer,
        &n_columns,
        &k_columns,
    };
    const blocks_per_row = shape.k / nvfp4_block;
    const grid = [3]u32{
        (blocks_per_row + threads_per_block - 1) / threads_per_block,
        shape.n,
        1,
    };
    try kernels.dequantize_weights.launch(
        grid,
        .{ threads_per_block, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Launches the exact reference path: dequantise the weights to bf16, then run
/// the bf16 kernel with the per-tensor weight scale folded into its epilogue.
pub fn launchNvfp4DequantBf16(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    a: *const cuda_context.DeviceBuffer,
    b: *const cuda_context.DeviceBuffer,
    b_scales: *const cuda_context.DeviceBuffer,
    dequantised: *const cuda_context.DeviceBuffer,
    scale2: f32,
    c: *const cuda_context.DeviceBuffer,
    shape: Shape,
) KernelError!void {
    assert(b.bytes > 0);
    assert(c.bytes > 0);

    try shape.validate();
    try launchDequantizeWeights(kernels, stream, b, b_scales, dequantised, shape);
    try launchBf16(kernels, stream, a, dequantised, scale2, c, shape);
}

/// Launches the e4m3 GEMM with a per-tensor weight scale.
pub fn launchFp8E4m3(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    a: *const cuda_context.DeviceBuffer,
    b: *const cuda_context.DeviceBuffer,
    b_scale: *const cuda_context.DeviceBuffer,
    c: *const cuda_context.DeviceBuffer,
    shape: Shape,
) KernelError!void {
    assert(b_scale.bytes >= @sizeOf(f32));
    assert(c.bytes > 0);

    try shape.validate();
    var a_pointer = a.pointer;
    var b_pointer = b.pointer;
    var scale_pointer = b_scale.pointer;
    var c_pointer = c.pointer;
    var m_rows: i32 = @intCast(shape.m);
    var n_columns: i32 = @intCast(shape.n);
    var k_columns: i32 = @intCast(shape.k);
    var arguments = [_]?*anyopaque{
        &a_pointer,
        &b_pointer,
        &scale_pointer,
        &c_pointer,
        &m_rows,
        &n_columns,
        &k_columns,
    };
    try kernels.fp8_e4m3.launch(
        Kernels.tiledGrid(shape),
        .{ threads_per_block, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

/// Launches the bf16 GEMM.
pub fn launchBf16(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    a: *const cuda_context.DeviceBuffer,
    b: *const cuda_context.DeviceBuffer,
    scale: f32,
    c: *const cuda_context.DeviceBuffer,
    shape: Shape,
) KernelError!void {
    assert(a.bytes > 0);
    assert(b.bytes > 0);

    try shape.validate();
    var a_pointer = a.pointer;
    var b_pointer = b.pointer;
    var epilogue_scale = scale;
    var c_pointer = c.pointer;
    var m_rows: i32 = @intCast(shape.m);
    var n_columns: i32 = @intCast(shape.n);
    var k_columns: i32 = @intCast(shape.k);
    var arguments = [_]?*anyopaque{
        &a_pointer,
        &b_pointer,
        &epilogue_scale,
        &c_pointer,
        &m_rows,
        &n_columns,
        &k_columns,
    };
    try kernels.bf16.launch(
        Kernels.tiledGrid(shape),
        .{ threads_per_block, 1, 1 },
        0,
        stream.handle,
        &arguments,
    );
}

test "Shape rejects a K the FP4 tile does not divide" {
    const shape = Shape{ .m = 16, .n = 640, .k = 2560 + 32 };
    try std.testing.expectError(
        KernelError.invalid_value,
        shape.validate(),
    );
}

test "Shape rejects an N the tile does not divide" {
    const shape = Shape{ .m = 16, .n = 640 + 8, .k = 2560 };
    try std.testing.expectError(
        KernelError.invalid_value,
        shape.validate(),
    );
}

test "Shape rejects M outside 1..8192" {
    try std.testing.expectError(
        KernelError.invalid_value,
        (Shape{ .m = 0, .n = 640, .k = 2560 }).validate(),
    );
    try std.testing.expectError(
        KernelError.invalid_value,
        (Shape{ .m = m_max + 1, .n = 640, .k = 2560 }).validate(),
    );
}

test "paddedRows rounds M up to a whole block tile" {
    try std.testing.expectEqual(
        @as(u32, 64),
        (Shape{ .m = 1, .n = 64, .k = 64 }).paddedRows(),
    );
    try std.testing.expectEqual(
        @as(u32, 128),
        (Shape{ .m = 65, .n = 64, .k = 64 }).paddedRows(),
    );
    try std.testing.expectEqual(
        @as(u32, 2048),
        (Shape{ .m = 2048, .n = 64, .k = 64 }).paddedRows(),
    );
}

test "tiledGrid covers N and the padded row count" {
    const shape = Shape{ .m = 1, .n = 640, .k = 2560 };
    const grid = Kernels.tiledGrid(shape);
    try std.testing.expectEqual(@as(u32, 10), grid[0]);
    try std.testing.expectEqual(@as(u32, 1), grid[1]);
}

test "stagingBytes rounds up to a whole page" {
    try std.testing.expectEqual(@as(u64, 4096), stagingBytes(1));
    try std.testing.expectEqual(@as(u64, 8192), stagingBytes(4097));
}
