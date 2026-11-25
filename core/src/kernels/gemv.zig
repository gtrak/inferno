//! Zig side of the decode GEMV family: typed launches and the validation
//! harness.
//!
//! `kernels/gemv.cu` holds the device code; this file owns everything that
//! decides. Dimension, alignment and capacity preconditions are assertions (a
//! wrong shape is a programming error, not a runtime condition); anything the
//! driver can refuse comes back in a closed `KernelError` set.
//!
//! The harness at the bottom is the only part that allocates. It builds random
//! matrices at the real model shapes, quantises them, runs the kernel, and
//! judges it twice: against the fp32 oracle in `ref/ops.zig` (correctness) and
//! against the measured HBM bandwidth in `bench/hw-probe.json` (speed). Decode
//! is bandwidth bound (THEORY.md §2), so the speed half is not optional.
//!
//! Timed iterations cycle over several identical copies of the weight matrix:
//! the smallest test shape is 14.75 MB and this GPU has 32 MiB of L2, so a
//! single copy would be measured out of cache and report a bandwidth the HBM
//! cannot deliver. A control kernel (`gemv_stream_read`) reads exactly the same
//! bytes with no arithmetic, which is the achievable ceiling for that size.

const std = @import("std");

const cuda_context = @import("../hw/api.zig").cuda_context;
const cuda_driver = @import("../hw/api.zig").cuda_driver;
const kernel = @import("../hw/kernel.zig");
const ops = @import("../ref/ops.zig");
const quant = @import("../weights/quant.zig");

const generated_cubins = @import("generated_cubins");

const assert = std.debug.assert;

/// Threads per block in `kernels/gemv.cu` (`gemv_block_threads`).
const block_threads: u32 = 256;

/// Lanes that share one output row (`gemv_lanes_per_row`).
const lanes_per_row: u32 = 8;

/// Output rows covered by one block (`gemv_rows_per_block`).
const rows_per_block: u32 = 32;

/// Largest decode batch the batched kernel specialises for.
const max_decode_batch: u32 = 32;

/// Contracted elements per NVFP4 block scale.
const nvfp4_block_elems: u32 = 16;

/// Side of the FP8 blockwise inverse-scale tile.
const fp8_scale_tile: u32 = 128;

/// Alignment every weight and activation buffer must satisfy for `uint4` loads.
const vector_alignment: u64 = 16;

/// `cuMemAlloc` granularity, and the size `DeviceBuffer.allocate` demands.
const allocation_granularity: u64 = 4096;

/// Bytes per BF16 element in device memory.
const bf16_bytes: u64 = 2;

/// Timed samples per case; the median is reported.
const timing_samples: usize = 7;

/// Measurement sweeps are opt-in: `zig build test` asserts, INFERNO_BENCH=1 measures. The extra
/// shapes only report throughput, so the default suite does not need their oracle cost.
fn benchMode() bool {
    return std.posix.getenv("INFERNO_BENCH") != null;
}

/// Upper bound on the copies of one weight matrix the harness cycles through
/// while timing, so every timed read misses L2.
const timing_copies: usize = 32;

/// Device budget for those copies; larger shapes already exceed L2 on their own.
const timing_copies_bytes_max: u64 = 1024 * 1024 * 1024;

/// L2 size the timing loop must defeat, used when `bench/hw-probe.json` is
/// absent or does not report one: 32 MiB, this machine's RTX 5060 Ti (GB206).
/// The figure is machine-specific; the probe carries the real value elsewhere.
const l2_bytes_default: u64 = 32 * 1024 * 1024;

/// Tier 2 of the I-06a ladder: accepted error of a dequantising GEMV against
/// the fp32 oracle computed from the *dequantised* weights, at the acceptance
/// shape. What is left is f32 accumulation order plus the BF16 output rounding,
/// which the oracle mirrors.
const dequantised_error_max: f64 = 2e-3;

/// The same comparison for every other shape, bounded by the output format
/// rather than by a constant: one BF16 step of the largest output is at most
/// 2^-7 of it, so a shape whose outputs are small cannot be held to 2e-3 (that
/// is below the format's own resolution there). Non-vacuous: the kernel must be
/// within one step of the oracle on the kernel's own operands.
const dequantised_ratio_max: f64 = 1.0 / 128.0;

/// Outputs are BF16, so the strongest statement the format allows is "within
/// half a BF16 step of the f32 oracle". One step is all that separates two
/// adjacent BF16 values, and the kernel's own arithmetic error is ~2e-7 of that.
const ulp_error_max: f64 = 0.5 + 1e-3;

/// Scale of the random weight matrices: the checkpoint's weights are O(0.02).
const weight_scale: f32 = 0.05;

/// Scale of the random activation vectors.
const activation_scale: f32 = 1.0;

/// Per-tensor NVFP4 second-level scale used by the harness.
const harness_weight_scale_2: f32 = 1.0;

/// Key in `bench/hw-probe.json` holding read-plus-write HBM traffic per GPU.
const measured_hbm_key = "d2d_hbm_gib_s";

/// Key in `bench/hw-probe.json` holding the per-GPU records.
const measured_devices_key = "device_info";

/// Key in a `measured_devices_key` record holding that GPU's L2 size in bytes.
const measured_l2_key = "l2_bytes";

/// Paths `bench/hw-probe.json` may live at, relative to the build root
/// (`core/`) and to the repository root: the gate may run from either.
const hardware_probe_paths = [_][]const u8{
    "../bench/hw-probe.json",
    "bench/hw-probe.json",
};

/// Cap on the bytes read from `bench/hw-probe.json`.
const hardware_probe_bytes_max: usize = 1 << 20;

/// Blocks the control kernel launches: four per SM on a 36 SM device.
const control_blocks: u32 = 4 * 36;

comptime {
    assert(block_threads % 32 == 0);
    assert(block_threads / 32 * (32 / lanes_per_row) == rows_per_block);
    assert(rows_per_block % lanes_per_row == 0);
    assert(nvfp4_block_elems == quant.nvfp4_block_size);
    assert(timing_samples > 1);
}

/// A matrix shape as the model uses it: `n` output rows, `k` contracted
/// elements.
pub const Shape = struct {
    n: u32,
    k: u32,

    /// Bytes of packed NVFP4 weights with their block scales.
    pub fn nvfp4Bytes(shape: Shape) u64 {
        assert(shape.n > 0);
        assert(shape.k % nvfp4_block_elems == 0);

        const per_row = shape.k / 2 + shape.k / nvfp4_block_elems;
        const total = @as(u64, shape.n) * per_row;
        assert(total > 0);
        return total;
    }

    /// Bytes of an FP8 blockwise weight matrix with its tile scales.
    pub fn fp8Bytes(shape: Shape) u64 {
        assert(shape.n > 0);
        assert(shape.k > 0);

        const scale_cols = (shape.k + fp8_scale_tile - 1) / fp8_scale_tile;
        const scale_rows = (shape.n + fp8_scale_tile - 1) / fp8_scale_tile;
        const total = @as(u64, shape.n) * shape.k +
            @as(u64, scale_rows) * scale_cols * bf16_bytes;
        assert(total >= @as(u64, shape.n) * shape.k);
        return total;
    }

    /// Bytes of BF16 weights.
    pub fn bf16Bytes(shape: Shape) u64 {
        assert(shape.n > 0);
        assert(shape.k > 0);

        const total = @as(u64, shape.n) * shape.k * bf16_bytes;
        assert(total > 0);
        return total;
    }
};

/// The cubin image of `kernels/gemv.cu` as `core/build.zig` embedded it.
fn embeddedCubin(name: []const u8) ?[]const u8 {
    assert(name.len > 0);
    assert(generated_cubins.cubins.len > 0);

    for (generated_cubins.cubins) |entry| {
        if (std.mem.eql(u8, std.mem.span(entry.name), name)) {
            assert(entry.length > 0);
            return entry.bytes[0..entry.length];
        } else {
            continue;
        }
    }
    return null;
}

/// Every entry point of `kernels/gemv.cu`, loaded once from the embedded cubin.
pub const Kernels = struct {
    module: kernel.Module,
    nvfp4: kernel.Function,
    nvfp4_batch: kernel.Function,
    fp8_blockwise: kernel.Function,
    bf16: kernel.Function,
    stream_read: kernel.Function,

    /// Loads the module and resolves all five kernels.
    pub fn load() kernel.KernelError!Kernels {
        const image = embeddedCubin("gemv") orelse
            return kernel.KernelError.module_load_failed;
        const module = try kernel.Module.load(image);
        errdefer module.unload();
        assert(module.handle != null);

        const loaded: Kernels = .{
            .module = module,
            .nvfp4 = try module.function("gemv_nvfp4_bf16"),
            .nvfp4_batch = try module.function("gemv_nvfp4_bf16_batch"),
            .fp8_blockwise = try module.function("gemv_fp8_blockwise_bf16"),
            .bf16 = try module.function("gemv_bf16"),
            .stream_read = try module.function("gemv_stream_read"),
        };
        assert(loaded.module.handle != null);
        return loaded;
    }

    pub fn unload(kernels: *const Kernels) void {
        assert(kernels.module.handle != null);
        assert(@intFromPtr(kernels.module.handle) != 0);

        kernels.module.unload();
    }
};

/// Rows per launch for `n` output rows; the kernel bounds-checks the last one.
fn gridRows(n: u32) u32 {
    assert(n > 0);
    assert(rows_per_block > 0);

    const blocks = (n + rows_per_block - 1) / rows_per_block;
    assert(blocks > 0);
    assert(blocks <= n);
    return blocks;
}

/// A sub-view of a device buffer beginning `offset` bytes in.
fn viewAfter(buffer: cuda_context.DeviceBuffer, offset: u64) cuda_context.DeviceBuffer {
    assert(offset < buffer.bytes);
    assert(offset % vector_alignment == 0);

    const view: cuda_context.DeviceBuffer = .{
        .pointer = buffer.pointer + offset,
        .bytes = buffer.bytes - offset,
    };
    assert(view.bytes > 0);
    return view;
}

/// Asserts every precondition of the NVFP4 kernels: shape contract, `uint4`
/// alignment and buffer capacity.
fn checkNvfp4Buffers(
    weight_packed: *const cuda_context.DeviceBuffer,
    weight_scales: *const cuda_context.DeviceBuffer,
    x: *const cuda_context.DeviceBuffer,
    y: *const cuda_context.DeviceBuffer,
    shape: Shape,
    batch: u32,
) void {
    assert(shape.n > 0);
    assert(shape.n % lanes_per_row == 0);
    assert(shape.k > 0);
    assert(shape.k % nvfp4_block_elems == 0);
    assert(batch > 0);
    assert(batch <= max_decode_batch);

    assert(weight_packed.pointer % vector_alignment == 0);
    assert(x.pointer % vector_alignment == 0);
    assert(y.pointer % vector_alignment == 0);
    assert(weight_packed.bytes >= @as(u64, shape.n) * (shape.k / 2));
    assert(weight_scales.bytes >=
        @as(u64, shape.n) * (shape.k / nvfp4_block_elems));
    assert(x.bytes >= @as(u64, batch) * shape.k * bf16_bytes);
    assert(y.bytes >= @as(u64, batch) * shape.n * bf16_bytes);
}

/// Launches `gemv_nvfp4_bf16`: one NVFP4 weight matrix against one activation
/// row, `y[o] = sum_k x[k] * dequant(w[o, k])`.
pub fn launchGemvNvfp4(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    weight_packed: *const cuda_context.DeviceBuffer,
    weight_scales: *const cuda_context.DeviceBuffer,
    weight_scale_2: f32,
    x: *const cuda_context.DeviceBuffer,
    y: *const cuda_context.DeviceBuffer,
    n: u32,
    k: u32,
) kernel.KernelError!void {
    assert(n > 0);
    assert(k > 0);
    assert(@intFromPtr(stream.handle) != 0);

    const shape: Shape = .{ .n = n, .k = k };
    checkNvfp4Buffers(weight_packed, weight_scales, x, y, shape, 1);

    var packed_arg = weight_packed.pointer;
    var scales_arg = weight_scales.pointer;
    var scale_2_arg = weight_scale_2;
    var x_arg = x.pointer;
    var y_arg = y.pointer;
    var n_arg: c_int = @intCast(n);
    var k_arg: c_int = @intCast(k);
    var parameters = [_]?*anyopaque{
        @ptrCast(&packed_arg),
        @ptrCast(&scales_arg),
        @ptrCast(&scale_2_arg),
        @ptrCast(&x_arg),
        @ptrCast(&y_arg),
        @ptrCast(&n_arg),
        @ptrCast(&k_arg),
    };
    assert(parameters.len == 7);

    try kernels.nvfp4.launch(
        .{ gridRows(n), 1, 1 },
        .{ block_threads, 1, 1 },
        0,
        stream.handle,
        parameters[0..],
    );
}

/// Launches `gemv_nvfp4_bf16_batch`: one weight matrix against `batch`
/// activation rows, so a batched decode re-reads no weights.
pub fn launchGemvNvfp4Batch(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    weight_packed: *const cuda_context.DeviceBuffer,
    weight_scales: *const cuda_context.DeviceBuffer,
    weight_scale_2: f32,
    x: *const cuda_context.DeviceBuffer,
    y: *const cuda_context.DeviceBuffer,
    n: u32,
    k: u32,
    batch: u32,
) kernel.KernelError!void {
    assert(n > 0);
    assert(k > 0);
    assert(batch > 0);
    assert(batch <= max_decode_batch);

    const shape: Shape = .{ .n = n, .k = k };
    checkNvfp4Buffers(weight_packed, weight_scales, x, y, shape, batch);

    var packed_arg = weight_packed.pointer;
    var scales_arg = weight_scales.pointer;
    var scale_2_arg = weight_scale_2;
    var x_arg = x.pointer;
    var y_arg = y.pointer;
    var n_arg: c_int = @intCast(n);
    var k_arg: c_int = @intCast(k);
    var batch_arg: c_int = @intCast(batch);
    var parameters = [_]?*anyopaque{
        @ptrCast(&packed_arg),
        @ptrCast(&scales_arg),
        @ptrCast(&scale_2_arg),
        @ptrCast(&x_arg),
        @ptrCast(&y_arg),
        @ptrCast(&n_arg),
        @ptrCast(&k_arg),
        @ptrCast(&batch_arg),
    };
    assert(parameters.len == 8);

    try kernels.nvfp4_batch.launch(
        .{ gridRows(n), 1, 1 },
        .{ block_threads, 1, 1 },
        0,
        stream.handle,
        parameters[0..],
    );
}

/// Launches `gemv_fp8_blockwise_bf16`: E4M3 weights with one BF16 inverse scale
/// per 128x128 tile, the layout of the MTP and requantised dense paths.
pub fn launchGemvFp8Blockwise(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    weight: *const cuda_context.DeviceBuffer,
    scale_inverse: *const cuda_context.DeviceBuffer,
    x: *const cuda_context.DeviceBuffer,
    y: *const cuda_context.DeviceBuffer,
    n: u32,
    k: u32,
) kernel.KernelError!void {
    assert(n > 0);
    assert(k > 0);
    assert(k % nvfp4_block_elems == 0);

    const scale_cols = (k + fp8_scale_tile - 1) / fp8_scale_tile;
    const scale_rows = (n + fp8_scale_tile - 1) / fp8_scale_tile;
    assert(weight.bytes >= @as(u64, n) * k);
    assert(scale_inverse.bytes >= @as(u64, scale_rows) * scale_cols * bf16_bytes);
    assert(x.bytes >= @as(u64, k) * bf16_bytes);
    assert(y.bytes >= @as(u64, n) * bf16_bytes);
    assert(weight.pointer % vector_alignment == 0);
    assert(x.pointer % vector_alignment == 0);

    var weight_arg = weight.pointer;
    var scale_arg = scale_inverse.pointer;
    var x_arg = x.pointer;
    var y_arg = y.pointer;
    var n_arg: c_int = @intCast(n);
    var k_arg: c_int = @intCast(k);
    var parameters = [_]?*anyopaque{
        @ptrCast(&weight_arg),
        @ptrCast(&scale_arg),
        @ptrCast(&x_arg),
        @ptrCast(&y_arg),
        @ptrCast(&n_arg),
        @ptrCast(&k_arg),
    };
    assert(parameters.len == 6);

    try kernels.fp8_blockwise.launch(
        .{ gridRows(n), 1, 1 },
        .{ block_threads, 1, 1 },
        0,
        stream.handle,
        parameters[0..],
    );
}

/// Launches `gemv_bf16`: the router, the lm_head fallback and the tests.
pub fn launchGemvBf16(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    weight: *const cuda_context.DeviceBuffer,
    x: *const cuda_context.DeviceBuffer,
    y: *const cuda_context.DeviceBuffer,
    n: u32,
    k: u32,
) kernel.KernelError!void {
    assert(n > 0);
    assert(k > 0);
    assert(k % lanes_per_row == 0);

    assert(weight.bytes >= @as(u64, n) * k * bf16_bytes);
    assert(x.bytes >= @as(u64, k) * bf16_bytes);
    assert(y.bytes >= @as(u64, n) * bf16_bytes);
    assert(weight.pointer % vector_alignment == 0);
    assert(x.pointer % vector_alignment == 0);

    var weight_arg = weight.pointer;
    var x_arg = x.pointer;
    var y_arg = y.pointer;
    var n_arg: c_int = @intCast(n);
    var k_arg: c_int = @intCast(k);
    var parameters = [_]?*anyopaque{
        @ptrCast(&weight_arg),
        @ptrCast(&x_arg),
        @ptrCast(&y_arg),
        @ptrCast(&n_arg),
        @ptrCast(&k_arg),
    };
    assert(parameters.len == 5);

    try kernels.bf16.launch(
        .{ gridRows(n), 1, 1 },
        .{ block_threads, 1, 1 },
        0,
        stream.handle,
        parameters[0..],
    );
}

/// Launches `gemv_stream_read` over `bytes` of `source`: the control run that
/// reports the read bandwidth this GPU delivers for that buffer.
fn launchStreamRead(
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    source: *const cuda_context.DeviceBuffer,
    sink: *const cuda_context.DeviceBuffer,
    bytes: u64,
) kernel.KernelError!void {
    assert(bytes > 0);
    assert(bytes % vector_alignment == 0);
    assert(source.bytes >= bytes);
    assert(sink.bytes >= @sizeOf(f32));

    var source_arg = source.pointer;
    var sink_arg = sink.pointer;
    var words_arg = bytes / vector_alignment;
    var parameters = [_]?*anyopaque{
        @ptrCast(&source_arg),
        @ptrCast(&sink_arg),
        @ptrCast(&words_arg),
    };
    assert(parameters.len == 3);

    try kernels.stream_read.launch(
        .{ control_blocks, 1, 1 },
        .{ block_threads, 1, 1 },
        0,
        stream.handle,
        parameters[0..],
    );
}

test {
    _ = @import("fp4_mma_probe.zig");
}

// ---------------------------------------------------------------------------
// Validation harness
// ---------------------------------------------------------------------------

/// Which kernel a timed launch runs.
const Family = enum { nvfp4, nvfp4_batch, fp8_blockwise, bf16, stream_read };

/// Error against the oracle and speed against the measured HBM bandwidth, for
/// one kernel at one shape.
const Case = struct {
    family: Family,
    shape: Shape,
    batch: u32,
    /// Against the oracle rounded to BF16: both sides are BF16, so this is the
    /// like-for-like comparison the numeric rule asks for.
    dequantised: ErrorStats,
    /// Against the unrounded f32 oracle: shows the BF16 output floor, which is
    /// half a BF16 step of the largest output and not a kernel defect.
    unrounded: ErrorStats,
    /// Against the original fp32 weights, i.e. the quantisation error the test
    /// data itself carries. Informational.
    source: ErrorStats,
    ulp_error: f64,
    output_magnitude: f32,
    bandwidth_gbs: f64,
    control_gbs: f64,
    fraction_of_hbm: f64,
};

/// One kernel configuration, ready to launch against any copy of the weights.
const TimedLaunch = struct {
    kernels: *const Kernels,
    stream: *const cuda_context.Stream,
    family: Family,
    copies: []const cuda_context.DeviceBuffer,
    weights_bytes: u64,
    weight_scale_2: f32,
    x: *const cuda_context.DeviceBuffer,
    y: *const cuda_context.DeviceBuffer,
    sink: *const cuda_context.DeviceBuffer,
    shape: Shape,
    batch: u32,
    copy_offset: usize,

    fn launchCopy(timed: TimedLaunch, index: usize) !void {
        assert(timed.weights_bytes > 0);
        assert(timed.copy_offset < timed.copies.len);

        const copy = (timed.copy_offset + index) % timed.copies.len;
        assert(copy < timed.copies.len);
        assert(timed.copies[copy].bytes >= timed.weights_bytes);

        const weights = timed.copies[copy];
        switch (timed.family) {
            .nvfp4 => try launchGemvNvfp4(
                timed.kernels,
                timed.stream,
                &weights,
                &viewAfter(weights, timed.weights_bytes),
                timed.weight_scale_2,
                timed.x,
                timed.y,
                timed.shape.n,
                timed.shape.k,
            ),
            .nvfp4_batch => try launchGemvNvfp4Batch(
                timed.kernels,
                timed.stream,
                &weights,
                &viewAfter(weights, timed.weights_bytes),
                timed.weight_scale_2,
                timed.x,
                timed.y,
                timed.shape.n,
                timed.shape.k,
                timed.batch,
            ),
            .fp8_blockwise => try launchGemvFp8Blockwise(
                timed.kernels,
                timed.stream,
                &weights,
                &viewAfter(weights, timed.weights_bytes),
                timed.x,
                timed.y,
                timed.shape.n,
                timed.shape.k,
            ),
            .bf16 => try launchGemvBf16(
                timed.kernels,
                timed.stream,
                &weights,
                timed.x,
                timed.y,
                timed.shape.n,
                timed.shape.k,
            ),
            .stream_read => try launchStreamRead(
                timed.kernels,
                timed.stream,
                &weights,
                timed.sink,
                weights.bytes,
            ),
        }
    }
};

/// Everything one case needs on the host: the quantised matrix, the activation
/// rows, both references and the read-back buffer.
const HostCase = struct {
    allocator: std.mem.Allocator,
    family: Family,
    shape: Shape,
    batch: u32,
    blob: []u8,
    weights_bytes: u64,
    weight_scale_2: f32,
    activations: []f32,
    x_bf16: []u16,
    dequantised: []f32,
    source: []f32,
    measured: []u16,

    fn deinit(host: *HostCase) void {
        assert(host.blob.len > 0);
        assert(host.measured.len > 0);

        host.allocator.free(host.blob);
        host.allocator.free(host.activations);
        host.allocator.free(host.x_bf16);
        host.allocator.free(host.dequantised);
        host.allocator.free(host.source);
        host.allocator.free(host.measured);
    }
};

/// Driver context, stream and kernels, closed in reverse order.
const Harness = struct {
    context: cuda_context.Context,
    stream: cuda_context.Stream,
    kernels: Kernels,

    /// Skips the calling test when this host has no usable driver.
    fn open() !Harness {
        var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
        const count = cuda_driver.query_devices(devices[0..]) catch |err| switch (err) {
            error.driver_library_missing => return error.SkipZigTest,
            else => return err,
        };
        assert(count > 0);

        const context = try cuda_context.Context.retain(&devices[0]);
        errdefer context.release();
        try context.set_current();
        const stream = try cuda_context.Stream.create();
        const kernels = Kernels.load() catch |err| switch (err) {
            kernel.KernelError.driver_library_missing => return error.SkipZigTest,
            else => return err,
        };
        assert(kernels.module.handle != null);
        return .{ .context = context, .stream = stream, .kernels = kernels };
    }

    fn close(harness: *Harness) void {
        assert(harness.kernels.module.handle != null);
        assert(@intFromPtr(harness.stream.handle) != 0);

        harness.kernels.unload();
        harness.stream.destroy();
        harness.context.release();
    }
};

/// Rounds a byte count up to the allocation granularity.
fn allocBytes(needed: u64) u64 {
    assert(needed > 0);
    assert(allocation_granularity > 0);

    const rounded = std.mem.alignForward(u64, needed, allocation_granularity);
    assert(rounded >= needed);
    return rounded;
}

/// Allocates device memory and fills it from `bytes`.
fn upload(
    stream: *const cuda_context.Stream,
    bytes: []const u8,
) !cuda_context.DeviceBuffer {
    assert(bytes.len > 0);
    assert(bytes.len <= std.math.maxInt(u32));

    const length = allocBytes(bytes.len);
    const buffer = try cuda_context.DeviceBuffer.allocate(length);
    errdefer buffer.free();
    const staged = try cuda_context.PinnedHostBuffer.allocate(length);
    defer staged.free();

    @memcpy(staged.slice()[0..bytes.len], bytes);
    try stream.copy_h_to_d(&buffer, &staged, bytes.len);
    try stream.synchronize();
    assert(buffer.pointer % vector_alignment == 0);
    return buffer;
}

/// Copies `bytes` of `buffer` back into `destination`.
fn download(
    stream: *const cuda_context.Stream,
    buffer: *const cuda_context.DeviceBuffer,
    destination: []u8,
) !void {
    assert(destination.len > 0);
    assert(destination.len <= buffer.bytes);

    const length = allocBytes(destination.len);
    const staged = try cuda_context.PinnedHostBuffer.allocate(length);
    defer staged.free();

    try stream.copy_d_to_h(&staged, buffer, destination.len);
    try stream.synchronize();
    @memcpy(destination, staged.slice()[0..destination.len]);
    assert(destination.len > 0);
}

/// Median of `values`; the slice is sorted in place.
fn median(values: []u64) u64 {
    assert(values.len > 0);
    assert(values.len <= 64);

    std.mem.sort(u64, values, {}, std.sort.asc(u64));
    const middle = values.len / 2;
    assert(middle < values.len);
    return values[middle];
}

/// GB/s from bytes and nanoseconds.
fn gigabytesPerSecond(bytes: u64, nanoseconds: u64) f64 {
    assert(bytes > 0);
    assert(nanoseconds > 0);

    const seconds = @as(f64, @floatFromInt(nanoseconds)) * 1e-9;
    const rate = @as(f64, @floatFromInt(bytes)) / seconds / 1e9;
    assert(rate > 0.0);
    return rate;
}

/// Measured single-GPU HBM traffic, as `bench/hw-probe.json` states it (GiB/s,
/// counted as read plus write) and converted to the decimal GB/s this harness
/// reports. Both numbers are printed, so no result silently mixes the units.
const MeasuredHbm = struct {
    gib_per_second: f64,
    gigabytes_per_second: f64,
};

/// Parses `bench/hw-probe.json`, or null when the probe has not been run here
/// (the file is gitignored, so a clean checkout has none).
fn hardwareProbe() ?std.json.Parsed(std.json.Value) {
    assert(hardware_probe_paths.len > 0);
    assert(hardware_probe_bytes_max > 0);

    for (hardware_probe_paths) |path| {
        const file = std.fs.cwd().openFile(path, .{}) catch continue;
        defer file.close();

        const contents = file.readToEndAlloc(
            std.heap.page_allocator,
            hardware_probe_bytes_max,
        ) catch continue;
        defer std.heap.page_allocator.free(contents);

        const parsed = std.json.parseFromSlice(
            std.json.Value,
            std.heap.page_allocator,
            contents,
            .{},
        ) catch continue;
        assert(parsed.value != .null);
        return parsed;
    }
    return null;
}

/// Reads `bench/hw-probe.json`, or null when the probe has not been run there
/// (the file is gitignored).
fn measuredHbm() ?MeasuredHbm {
    assert(@sizeOf(f64) == 8);
    assert(measured_hbm_key.len > 0);

    const parsed = hardwareProbe() orelse return null;
    defer parsed.deinit();
    if (parsed.value == .object) {
        assert(parsed.value == .object);
    } else {
        return null;
    }

    const rates = parsed.value.object.get(measured_hbm_key) orelse return null;
    if (rates == .array and rates.array.items.len > 0) {
        assert(rates.array.items.len > 0);
    } else {
        return null;
    }
    const gib_per_second = rates.array.items[0].float;
    const gigabytes = gib_per_second * (1024.0 * 1024.0 * 1024.0) / 1e9;
    if (gigabytes > 1.0) {
        return .{
            .gib_per_second = gib_per_second,
            .gigabytes_per_second = gigabytes,
        };
    } else {
        return null;
    }
}

/// The first GPU's `l2_bytes` from `bench/hw-probe.json`, or null when the file
/// is absent or does not report one.
fn measuredL2Bytes() ?u64 {
    assert(measured_devices_key.len > 0);
    assert(measured_l2_key.len > 0);

    const parsed = hardwareProbe() orelse return null;
    defer parsed.deinit();
    if (parsed.value == .object) {
        assert(parsed.value == .object);
    } else {
        return null;
    }

    const devices = parsed.value.object.get(measured_devices_key) orelse
        return null;
    if (devices == .array and devices.array.items.len > 0) {
        assert(devices.array.items.len > 0);
    } else {
        return null;
    }

    const first = devices.array.items[0];
    if (first == .object) {
        assert(first == .object);
    } else {
        return null;
    }
    const reported = first.object.get(measured_l2_key) orelse return null;
    if (reported == .integer and reported.integer > 0) {
        assert(reported.integer > 0);
        return @intCast(reported.integer);
    } else {
        return null;
    }
}

/// L2 size the timing loop has to defeat: the probe's figure when there is one,
/// else `l2_bytes_default`. A timed read must miss L2 or the reported GB/s is
/// not HBM bandwidth.
fn l2Bytes() u64 {
    assert(l2_bytes_default > 0);
    assert(measured_l2_key.len > 0);

    const measured = measuredL2Bytes() orelse l2_bytes_default;
    assert(measured > 0);
    return measured;
}

/// Byte offset of the scale section inside a blob holding weights then scales.
fn weightsBytesFor(shape: Shape, family: Family) u64 {
    assert(shape.n > 0);
    assert(shape.k > 0);

    const bytes = switch (family) {
        .nvfp4, .nvfp4_batch => @as(u64, shape.n) * (shape.k / 2),
        .fp8_blockwise => @as(u64, shape.n) * shape.k,
        .bf16 => shape.bf16Bytes(),
        .stream_read => @as(u64, shape.n) * (shape.k / 2),
    };
    assert(bytes > 0);
    return bytes;
}

/// Deterministic normal samples with `torch.randn` semantics: zero mean, unit
/// variance, scaled.
fn fillNormal(random: std.Random, values: []f32, scale: f32) void {
    assert(values.len > 0);
    assert(scale > 0.0);

    for (values) |*value| {
        const sample = random.floatNorm(f32) * scale;
        value.* = if (std.math.isFinite(sample)) sample else 0.0;
    }
}

/// Nearest E4M3 (FN) code to a non-negative `value`, saturating at the ends of
/// the format. Used only to pick a block scale the checkpoint format can store.
fn nearestE4m3Code(value: f32) u8 {
    assert(value >= 0.0);
    assert(value < 1e9);

    if (value <= 0.0) {
        return 0;
    } else {
        assert(value > 0.0);
    }
    // Below the smallest normal (2^-6) E4M3 steps in units of 2^-9, codes 1..7.
    const subnormal_step = std.math.scalbn(@as(f32, 1.0), -9);
    if (value < std.math.scalbn(@as(f32, 1.0), -6)) {
        const steps = @round(value / subnormal_step);
        const clamped = @min(@max(steps, 1.0), 7.0);
        assert(clamped >= 1.0);
        return @intFromFloat(clamped);
    } else {
        assert(value >= std.math.scalbn(@as(f32, 1.0), -6));
    }
    var exponent: i32 = @intFromFloat(@floor(std.math.log2(value)));
    if (exponent > 8) {
        return 0x7e; // 448, the largest E4M3 value
    } else {
        assert(exponent >= -6);
    }
    const step = std.math.scalbn(@as(f32, 1.0), exponent);
    var mantissa: i32 = @intFromFloat(@round((value / step - 1.0) * 8.0));
    if (mantissa >= 8) {
        exponent += 1;
        mantissa = 0;
    } else {
        assert(mantissa >= 0);
    }
    if (exponent > 8) {
        return 0x7e;
    } else {
        assert(exponent <= 8);
    }
    const code: u8 = @intCast((exponent + 7) << 3 | mantissa);
    return code;
}

/// Nearest E2M1 code (sign in bit 3) to `ratio`, using the magnitude grid of
/// `quant.fp4_values`.
fn nearestE2m1Code(ratio: f32) u8 {
    assert(ratio == ratio);
    assert(quant.fp4_values.len == 16);

    const magnitude = @abs(ratio);
    var best: u8 = 0;
    var best_error = std.math.floatMax(f32);
    for (0..8) |index| {
        const candidate = @abs(quant.fp4_values[index]);
        const error_value = @abs(magnitude - candidate);
        if (error_value < best_error) {
            best_error = error_value;
            best = @intCast(index);
        } else {
            continue;
        }
    }
    const sign: u8 = if (ratio < 0.0) 8 else 0;
    assert(best < 8);
    return best | sign;
}

/// Test-only NVFP4 quantiser: one block scale per 16 weights, taken from the
/// exact-power-of-two E4M3 codes, weights rounded to the nearest E2M1 value.
/// The production requantiser is a separate issue; this exists so the harness
/// has a realistic, non-uniform quantisation error to report.
fn quantiseNvfp4Row(
    weights: []const f32,
    packed_weights: []u8,
    scales: []u8,
    scale_2: f32,
) void {
    assert(weights.len % nvfp4_block_elems == 0);
    assert(packed_weights.len == weights.len / 2);
    assert(scales.len * nvfp4_block_elems == weights.len);

    for (scales, 0..) |*scale_out, block| {
        const block_weights = weights[block * nvfp4_block_elems ..][0..nvfp4_block_elems];
        var largest: f32 = 0.0;
        for (block_weights) |value| {
            largest = @max(largest, @abs(value));
        }
        const wanted = if (largest > 0.0) largest / 6.0 / scale_2 else 1.0;
        const scale_code = nearestE4m3Code(wanted);
        const step = quant.f8_e4m3_to_f32(@enumFromInt(scale_code));
        scale_out.* = scale_code;
        assert(step >= 0.0);

        for (block_weights, 0..) |value, index| {
            const code = nearestE2m1Code(value / (step * scale_2));
            const byte_index = block * (nvfp4_block_elems / 2) + index / 2;
            if (index % 2 == 0) {
                packed_weights[byte_index] = code;
            } else {
                packed_weights[byte_index] |= code << 4;
            }
        }
    }
}

/// Allocates a host case and fills its activation rows; the weights are left
/// zeroed for the family-specific preparation.
fn newHostCase(
    allocator: std.mem.Allocator,
    shape: Shape,
    batch: u32,
    family: Family,
    blob_len: u64,
    activation_random: std.Random,
) !HostCase {
    assert(shape.n > 0);
    assert(shape.k > 0);
    assert(batch > 0);

    const elements: usize = @as(usize, batch) * shape.n;
    var host: HostCase = .{
        .allocator = allocator,
        .family = family,
        .shape = shape,
        .batch = batch,
        .blob = try allocator.alloc(u8, @intCast(blob_len)),
        .weights_bytes = weightsBytesFor(shape, family),
        .weight_scale_2 = harness_weight_scale_2,
        .activations = try allocator.alloc(f32, @as(usize, batch) * shape.k),
        .x_bf16 = try allocator.alloc(u16, @as(usize, batch) * shape.k),
        .dequantised = try allocator.alloc(f32, elements),
        .source = try allocator.alloc(f32, elements),
        .measured = try allocator.alloc(u16, elements),
    };
    errdefer host.deinit();
    @memset(host.blob, 0);

    fillNormal(activation_random, host.activations, activation_scale);
    // The kernels read BF16 activations, so the oracle must see exactly the
    // values the hardware does: round here, once.
    for (host.x_bf16, host.activations) |*bits, *value| {
        bits.* = quant.f32_to_bf16(value.*);
        value.* = quant.bf16_to_f32(bits.*);
    }
    assert(host.blob.len == blob_len);
    return host;
}

/// Runs the fp32 oracle for one output row against both references.
fn oracleRow(
    host: *HostCase,
    row: usize,
    original: []const f32,
    dequantised: []const f32,
) void {
    assert(row < host.shape.n);
    assert(original.len == host.shape.k);
    assert(dequantised.len == host.shape.k);

    for (0..host.batch) |index| {
        const activation = host.activations[index * host.shape.k ..][0..host.shape.k];
        const shape: ops.MatmulShape = .{
            .rows = 1,
            .in_features = host.shape.k,
            .out_features = 1,
        };
        ops.matmul(
            activation,
            original,
            host.source[index * host.shape.n + row ..][0..1],
            shape,
        );
        ops.matmul(
            activation,
            dequantised,
            host.dequantised[index * host.shape.n + row ..][0..1],
            shape,
        );
    }
}

/// Prepares an NVFP4 case: quantise normally distributed weights, then build the
/// oracle from the dequantised row and from the original f32 row.
fn prepareNvfp4(
    allocator: std.mem.Allocator,
    shape: Shape,
    batch: u32,
    weights_random: std.Random,
    activation_random: std.Random,
) !HostCase {
    assert(shape.k % nvfp4_block_elems == 0);
    assert(batch > 0);
    assert(batch <= max_decode_batch);

    const family: Family = if (batch == 1) .nvfp4 else .nvfp4_batch;
    var host = try newHostCase(
        allocator,
        shape,
        batch,
        family,
        shape.nvfp4Bytes(),
        activation_random,
    );
    errdefer host.deinit();

    const original = try allocator.alloc(f32, shape.k);
    defer allocator.free(original);
    const dequantised = try allocator.alloc(f32, shape.k);
    defer allocator.free(dequantised);

    for (0..shape.n) |row| {
        fillNormal(weights_random, original, weight_scale);
        const packed_row = host.blob[row * (shape.k / 2) ..][0 .. shape.k / 2];
        const scale_offset = host.weights_bytes + row * (shape.k / nvfp4_block_elems);
        const scales = host.blob[scale_offset..][0 .. shape.k / nvfp4_block_elems];
        quantiseNvfp4Row(original, packed_row, scales, host.weight_scale_2);
        quant.dequant_nvfp4_row(
            packed_row,
            @ptrCast(scales),
            host.weight_scale_2,
            dequantised,
        );
        oracleRow(&host, row, original, dequantised);
    }
    assert(host.blob.len == shape.nvfp4Bytes());
    return host;
}

/// Prepares an FP8 blockwise case. Random valid E4M3 bytes and exact-power-of-
/// two BF16 tile scales; the reference is the exact dequantised matrix, because
/// the FP8 checkpoint weights *are* the quantised values.
fn prepareFp8(
    allocator: std.mem.Allocator,
    shape: Shape,
    weights_random: std.Random,
    activation_random: std.Random,
) !HostCase {
    assert(shape.n > 0);
    assert(shape.k % nvfp4_block_elems == 0);

    var host = try newHostCase(
        allocator,
        shape,
        single_row_batch,
        .fp8_blockwise,
        shape.fp8Bytes(),
        activation_random,
    );
    errdefer host.deinit();

    const scale_cols = (shape.k + fp8_scale_tile - 1) / fp8_scale_tile;
    const scale_rows = (shape.n + fp8_scale_tile - 1) / fp8_scale_tile;
    const scale_bytes = host.blob[host.weights_bytes..][0 .. scale_rows * scale_cols * 2];
    const scales = std.mem.bytesAsSlice(u16, scale_bytes);
    for (scales) |*scale| {
        const step = weights_random.intRangeAtMost(u8, 0, 2);
        const exponent = fp8_scale_inv_exponent_min + @as(i32, step);
        assert(exponent <= fp8_scale_inv_exponent_max);
        scale.* = @intCast((exponent + 127) << 7);
    }
    for (host.blob[0..host.weights_bytes]) |*byte| {
        const exponent = weights_random.intRangeAtMost(u8, 0, 14);
        const mantissa = weights_random.intRangeAtMost(u8, 0, 7);
        const sign: u8 = if (weights_random.boolean()) 0x80 else 0x00;
        byte.* = sign | (exponent << 3) | mantissa;
    }

    const dequantised = try allocator.alloc(f32, shape.k);
    defer allocator.free(dequantised);
    for (0..shape.n) |row| {
        const row_weights = host.blob[row * shape.k ..][0..shape.k];
        const row_scales = scales[(row / fp8_scale_tile) * scale_cols ..][0..scale_cols];
        for (row_weights, 0..) |byte, column| {
            const scale = quant.bf16_to_f32(row_scales[column / fp8_scale_tile]);
            dequantised[column] = quant.f8_e4m3_to_f32(@enumFromInt(byte)) * scale;
        }
        oracleRow(&host, row, dequantised, dequantised);
    }
    assert(host.blob.len == shape.fp8Bytes());
    return host;
}

/// Batch used by the non-NVFP4 families, which only need the single-row path.
const single_row_batch: u32 = 1;

/// Exponents of the FP8 blockwise inverse scales the harness uses. A 128x128
/// tile of N(0, 0.05) weights peaks near 0.15, and the tile scale is
/// `amax / 448`, i.e. ~3.3e-4: these powers of two put the dequantised test
/// weights at model magnitude instead of across the whole E4M3 range, which
/// keeps the comparison a production-magnitude one.
const fp8_scale_inv_exponent_min: i32 = -13;
const fp8_scale_inv_exponent_max: i32 = -11;

/// Prepares a BF16 case: the weights are BF16 already, so the reference is the
/// exact BF16 matrix and both errors are zero by construction.
fn prepareBf16(
    allocator: std.mem.Allocator,
    shape: Shape,
    weights_random: std.Random,
    activation_random: std.Random,
) !HostCase {
    assert(shape.n > 0);
    assert(shape.k % lanes_per_row == 0);

    var host = try newHostCase(
        allocator,
        shape,
        single_row_batch,
        .bf16,
        shape.bf16Bytes(),
        activation_random,
    );
    errdefer host.deinit();

    const bits = std.mem.bytesAsSlice(u16, host.blob);
    const dequantised = try allocator.alloc(f32, shape.k);
    defer allocator.free(dequantised);
    const original = try allocator.alloc(f32, shape.k);
    defer allocator.free(original);
    for (0..shape.n) |row| {
        const row_bits = bits[row * shape.k ..][0..shape.k];
        fillNormal(weights_random, original, weight_scale);
        for (row_bits, original) |*value, sample| {
            value.* = quant.f32_to_bf16(sample);
        }
        for (row_bits, dequantised) |value, *decoded| {
            decoded.* = quant.bf16_to_f32(value);
        }
        oracleRow(&host, row, dequantised, dequantised);
    }
    assert(host.blob.len == shape.bf16Bytes());
    return host;
}

/// One comparison: the absolute error, the magnitude it is measured against,
/// and their ratio. The absolute pair travels with the ratio because a small
/// reference magnitude inflates the ratio under cancellation, so a scaled figure
/// on its own is not evidence.
const ErrorStats = struct {
    /// Largest |measured - reference| over the compared elements.
    absolute: f32,
    /// Largest |reference|, the divisor of `ratio`.
    magnitude: f32,
    /// `absolute / magnitude`, or 0 when the reference is all zeros.
    ratio: f64,
};

/// Compares the measured BF16 output against `reference`.
fn errorStats(measured: []const u16, reference: []const f32) ErrorStats {
    assert(measured.len == reference.len);
    assert(measured.len > 0);

    var magnitude: f32 = 0.0;
    var absolute: f32 = 0.0;
    for (measured, reference) |bits, expected| {
        const value = quant.bf16_to_f32(bits);
        magnitude = @max(magnitude, @abs(expected));
        absolute = @max(absolute, @abs(value - expected));
    }
    assert(absolute >= 0.0);
    const ratio = if (magnitude > 0.0)
        @as(f64, absolute) / @as(f64, magnitude)
    else
        0.0;
    assert(ratio >= 0.0);
    return .{ .absolute = absolute, .magnitude = magnitude, .ratio = ratio };
}

/// Largest |measured|, i.e. the output magnitude the comparison is judged at.
fn outputMagnitude(measured: []const u16) f32 {
    assert(measured.len > 0);
    assert(measured.len <= 1 << 24);

    var largest: f32 = 0.0;
    for (measured) |bits| {
        largest = @max(largest, @abs(quant.bf16_to_f32(bits)));
    }
    assert(largest >= 0.0);
    return largest;
}

/// One BF16 step at the magnitude of `reference`.
fn bf16Step(reference: f32) f32 {
    assert(reference == reference);
    assert(@abs(reference) < std.math.floatMax(f32));

    const bits = quant.f32_to_bf16(reference);
    const exponent: i32 = @intCast((bits >> 7) & 0xff);
    const step = if (exponent == 0)
        std.math.scalbn(@as(f32, 1.0), bf16_subnormal_exponent)
    else
        std.math.scalbn(@as(f32, 1.0), exponent - 134);
    assert(step > 0.0);
    return step;
}

/// Exponent of one BF16 subnormal step (2^-133).
const bf16_subnormal_exponent: i32 = -133;

/// Largest error between the measured output and the f32 oracle, in BF16 steps
/// of the largest output. This is the tightest check the BF16 output format
/// allows: half a step means every element is the correctly rounded value of the
/// f32 oracle, whatever the cancellation inside the dot product.
fn ulpError(measured: []const u16, reference: []const f32) f64 {
    assert(measured.len == reference.len);
    assert(measured.len > 0);

    var largest: f32 = 0.0;
    for (reference) |expected| {
        largest = @max(largest, @abs(expected));
    }
    if (largest == 0.0) {
        return 0.0;
    } else {
        assert(largest > 0.0);
    }
    const step = bf16Step(largest);
    var worst: f64 = 0.0;
    for (measured, reference) |bits, expected| {
        const value = quant.bf16_to_f32(bits);
        worst = @max(worst, @as(f64, @abs(value - expected)) / @as(f64, step));
    }
    assert(worst >= 0.0);
    return worst;
}

/// Reports one case and enforces the acceptance thresholds.
fn reportCase(case: Case) !void {
    assert(case.shape.n > 0);
    assert(case.batch > 0);

    std.debug.print(
        "  {s:<14} n={d:<6} k={d:<5} m={d:<2} {d:7.1} GB/s " ++
            "({d:5.1}% of HBM, control {d:7.1} GB/s)\n" ++
            "    |y|max={e:.3} |dequantised: abs={e:.3} rel={e:.3}" ++
            " (mag {e:.3}) |f32 oracle: abs={e:.3} rel={e:.3}" ++
            " (mag {e:.3}) |source: abs={e:.3} rel={e:.3}" ++
            " (mag {e:.3}) |bf16 steps={d:.3}\n",
        .{
            @tagName(case.family),
            case.shape.n,
            case.shape.k,
            case.batch,
            case.bandwidth_gbs,
            case.fraction_of_hbm * 100.0,
            case.control_gbs,
            case.output_magnitude,
            case.dequantised.absolute,
            case.dequantised.ratio,
            case.dequantised.magnitude,
            case.unrounded.absolute,
            case.unrounded.ratio,
            case.unrounded.magnitude,
            case.source.absolute,
            case.source.ratio,
            case.source.magnitude,
            case.ulp_error,
        },
    );
    // The I-06a ladder. Tier 1/2: a correct kernel is within half a BF16 step
    // of the f32 oracle on its own operands (`ulp_error`), and within one step
    // of that oracle rounded to the output format. Tier 3 (`case.source`, the
    // error against the original fp32 weights) is printed and never asserted:
    // per-16 E2M1 on Gaussian weights cannot beat ~8-10 %, so a bound there
    // would grade the test quantiser, not the kernel.
    try std.testing.expect(case.ulp_error <= ulp_error_max);
    try std.testing.expect(case.dequantised.ratio <= dequantised_ratio_max);
    try std.testing.expect(case.output_magnitude > 0.0);
    try std.testing.expect(case.bandwidth_gbs > 0.0);
    try std.testing.expect(case.control_gbs > 0.0);
}

/// Copies of one weight matrix: enough to defeat the L2 of `bytes_per_copy`'s
/// shape, within budget.
fn timingCopyCount(bytes_per_copy: u64, l2: u64) usize {
    assert(bytes_per_copy > 0);
    assert(l2 > 0);

    const target = 4 * l2;
    if (bytes_per_copy >= target) {
        return 1;
    } else {
        const wanted = target / bytes_per_copy + 1;
        const count = @min(wanted, timing_copies);
        const copies_bytes = count * bytes_per_copy;
        if (copies_bytes > timing_copies_bytes_max) {
            return 1;
        } else {
            return @intCast(count);
        }
    }
}

/// Median nanoseconds of `timing_samples` launches, cycling the copies.
fn timeMedian(launch: TimedLaunch) !u64 {
    assert(launch.copies.len > 0);
    assert(timing_samples > 1);

    // Warm up: loads the cubin, faults the buffers in and fills L2 once.
    try launch.launchCopy(0);
    try launch.stream.synchronize();

    var samples: [timing_samples]u64 = undefined;
    for (&samples, 0..) |*sample, index| {
        const start = try cuda_context.Event.create();
        defer start.destroy();
        const end = try cuda_context.Event.create();
        defer end.destroy();

        try start.record(launch.stream);
        try launch.launchCopy((index + 1) % launch.copies.len);
        try end.record(launch.stream);
        try end.synchronize();
        sample.* = try cuda_context.Event.elapsed_nanoseconds(&start, &end);
    }
    return median(samples[0..]);
}

/// Uploads the weight blob once, then clones it into the remaining copies so
/// every timed launch reads a cold, identical matrix.
fn uploadCopies(
    stream: *const cuda_context.Stream,
    blob: []const u8,
    copies: []cuda_context.DeviceBuffer,
) !void {
    assert(blob.len > 0);
    assert(copies.len > 0);

    for (copies) |*copy| {
        copy.* = try upload(stream, blob);
    }
    try stream.synchronize();
    assert(copies[0].bytes >= blob.len);
}

/// Runs one case end to end: launch, verify against the oracle, then time the
/// kernel and the control run on the same bytes.
fn runCase(harness: *Harness, host: *HostCase, hbm: ?MeasuredHbm) !Case {
    assert(host.blob.len > 0);
    assert(host.shape.n > 0);

    var copies: [timing_copies]cuda_context.DeviceBuffer = undefined;
    const copy_count = timingCopyCount(allocBytes(host.blob.len), l2Bytes());
    try uploadCopies(&harness.stream, host.blob, copies[0..copy_count]);
    defer for (copies[0..copy_count]) |*copy| copy.free();

    const x = try upload(&harness.stream, std.mem.sliceAsBytes(host.x_bf16));
    defer x.free();
    const y_bytes = @as(u64, host.batch) * host.shape.n * bf16_bytes;
    const y = try cuda_context.DeviceBuffer.allocate(allocBytes(y_bytes));
    defer y.free();
    const sink = try cuda_context.DeviceBuffer.allocate(allocation_granularity);
    defer sink.free();

    var launch: TimedLaunch = .{
        .kernels = &harness.kernels,
        .stream = &harness.stream,
        .family = host.family,
        .copies = copies[0..copy_count],
        .weights_bytes = host.weights_bytes,
        .weight_scale_2 = host.weight_scale_2,
        .x = &x,
        .y = &y,
        .sink = &sink,
        .shape = host.shape,
        .batch = host.batch,
        .copy_offset = 0,
    };
    try launch.launchCopy(0);
    try harness.stream.synchronize();
    try download(&harness.stream, &y, std.mem.sliceAsBytes(host.measured));

    var result = try compareOutputs(host);
    const nanoseconds = try timeMedian(launch);
    // The control reads a disjoint half of the pool, so it cannot be served from
    // the lines the kernel run above just pulled into L2.
    launch.family = .stream_read;
    launch.copy_offset = copy_count / 2;
    const control_nanoseconds = try timeMedian(launch);

    result.bandwidth_gbs = gigabytesPerSecond(host.blob.len, nanoseconds);
    result.control_gbs = gigabytesPerSecond(host.blob.len, control_nanoseconds);
    result.fraction_of_hbm = if (hbm) |measured|
        result.bandwidth_gbs / measured.gigabytes_per_second
    else
        0.0;
    assert(result.bandwidth_gbs > 0.0);
    return result;
}

/// Compares the downloaded output against the oracle and the original weights.
fn compareOutputs(host: *const HostCase) !Case {
    assert(host.measured.len > 0);
    assert(host.dequantised.len == host.measured.len);

    // The output format is BF16, so the reference the kernel can reproduce is
    // the oracle rounded to BF16; the unrounded oracle is reported as well.
    const rounded = try host.allocator.alloc(f32, host.dequantised.len);
    defer host.allocator.free(rounded);
    for (rounded, host.dequantised) |*value, original| {
        value.* = quant.bf16_to_f32(quant.f32_to_bf16(original));
    }

    const result: Case = .{
        .family = host.family,
        .shape = host.shape,
        .batch = host.batch,
        .dequantised = errorStats(host.measured, rounded),
        .unrounded = errorStats(host.measured, host.dequantised),
        .source = errorStats(host.measured, host.source),
        .ulp_error = ulpError(host.measured, host.dequantised),
        .output_magnitude = outputMagnitude(host.measured),
        .bandwidth_gbs = 0.0,
        .control_gbs = 0.0,
        .fraction_of_hbm = 0.0,
    };
    assert(result.ulp_error >= 0.0);
    return result;
}

/// Prints the head of the output and checks it is not degenerate.
fn checkNonVacuity(host: *const HostCase) !void {
    assert(host.measured.len > 0);
    assert(host.measured.len == host.batch * host.shape.n);

    var negatives: usize = 0;
    var positives: usize = 0;
    for (host.measured) |bits| {
        const value = quant.bf16_to_f32(bits);
        if (value < 0.0) {
            negatives += 1;
        } else if (value > 0.0) {
            positives += 1;
        } else {
            continue;
        }
    }
    std.debug.print("    y[0..4] =", .{});
    for (host.measured[0..4]) |bits| {
        std.debug.print(" {d:.5}", .{quant.bf16_to_f32(bits)});
    }
    std.debug.print("   (negative {d}, positive {d})\n", .{ negatives, positives });
    try std.testing.expect(negatives > 0);
    try std.testing.expect(positives > 0);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// The launch wrappers must reject nothing the model needs: these byte counts
// are the checkpoint layouts the kernels assume.
test "shape byte counts follow the checkpoint layouts" {
    const shape: Shape = .{ .n = 12, .k = 2560 };
    try std.testing.expectEqual(@as(u64, 12 * (1280 + 160)), shape.nvfp4Bytes());
    try std.testing.expectEqual(@as(u64, 12 * 2560 + 20 * 2), shape.fp8Bytes());
    try std.testing.expectEqual(@as(u64, 12 * 2560 * 2), shape.bf16Bytes());
    try std.testing.expectEqual(@as(u32, 1), gridRows(rows_per_block));
    try std.testing.expectEqual(@as(u32, 2), gridRows(rows_per_block + 1));
}

test "nvfp4 decode gemv matches the fp32 oracle at the real model shapes" {
    var harness = try Harness.open();
    defer harness.close();

    const hbm = measuredHbm();
    if (hbm) |measured| {
        std.debug.print(
            "measured single-GPU HBM, read+write: {d:.2} GiB/s = " ++
                "{d:.1} GB/s (bench/hw-probe.json)\n",
            .{ measured.gib_per_second, measured.gigabytes_per_second },
        );
    } else {
        std.debug.print(
            "bench/hw-probe.json absent: reporting GB/s without grading it\n",
            .{},
        );
    }
    // The timing loop must cycle enough copies of a weight matrix to miss L2;
    // state the figure it used, and where it came from, so the assumption is
    // never silent.
    if (measuredL2Bytes()) |measured| {
        std.debug.print(
            "timing loop defeats the probe's L2: {d} MiB\n",
            .{measured >> 20},
        );
    } else {
        std.debug.print(
            "timing loop defeats {d} MiB of L2 (machine-specific default: " ++
                "no bench/hw-probe.json)\n",
            .{l2_bytes_default >> 20},
        );
    }

    const allocator = std.testing.allocator;
    var weights_random = std.Random.DefaultPrng.init(0x006a_0001);
    var activation_random = std.Random.DefaultPrng.init(0x006a_0002);
    // Only index 0 is GRADED below; every other shape merely reports throughput. So the full
    // list is a measurement, not an assertion: lm_head's 248320x2560 alone costs ~21 s of fp32
    // oracle (2.5 GB of host comparison) for a line nobody gates on.
    // Default: the acceptance shape plus the partial-block case. INFERNO_BENCH=1 sweeps all.
    const all_shapes = [_]Shape{
        .{ .n = 10240, .k = 2560 }, // dense projection, the acceptance shape
        .{ .n = 6144, .k = 2560 }, // gate/up of an expert group
        .{ .n = 12288, .k = 2560 }, // wide gate/up
        .{ .n = 2560, .k = 6144 }, // down projection
        .{ .n = 248320, .k = 2560 }, // lm_head, the largest matrix
        .{ .n = 40, .k = 2560 }, // n % 8 == 0 but not % 32: partial last block
    };
    const default_shapes = [_]Shape{ all_shapes[0], all_shapes[5] };
    const shapes: []const Shape = if (benchMode()) &all_shapes else &default_shapes;

    var acceptance_fraction: f64 = 0.0;
    var acceptance_error: f64 = std.math.floatMax(f64);
    for (shapes, 0..) |shape, index| {
        var host = try prepareNvfp4(
            allocator,
            shape,
            1,
            weights_random.random(),
            activation_random.random(),
        );
        defer host.deinit();
        const result = try runCase(&harness, &host, hbm);
        try reportCase(result);
        try checkNonVacuity(&host);
        if (index == 0) {
            acceptance_fraction = result.fraction_of_hbm;
            acceptance_error = result.dequantised.ratio;
        } else {
            continue;
        }
    }

    // Spec gates at the acceptance shape, [10240, 2560]: the dequantisation
    // must be exact to 2e-3 of the reference, and the hot path must reach 55%
    // of this GPU's measured HBM bandwidth.
    try std.testing.expect(acceptance_error <= dequantised_error_max);
    // Spec gate: an ungraded bandwidth gate must never read as a passed one, so
    // the fall-through is a visible SKIP line.
    if (hbm != null) {
        try std.testing.expect(acceptance_fraction >= 0.55);
    } else {
        std.debug.print(
            "SKIP: bandwidth gate not graded (no bench/hw-probe.json)\n",
            .{},
        );
    }
}

test "batched nvfp4 gemv reads the weights once for the whole batch" {
    var harness = try Harness.open();
    defer harness.close();

    const hbm = measuredHbm();
    const allocator = std.testing.allocator;
    var weights_random = std.Random.DefaultPrng.init(0x006a_0003);
    var activation_random = std.Random.DefaultPrng.init(0x006a_0004);
    const cases = [_]struct { shape: Shape, batch: u32 }{
        .{ .shape = .{ .n = 640, .k = 2560 }, .batch = 3 }, // partial-batch clamp
        .{ .shape = .{ .n = 640, .k = 2560 }, .batch = 16 },
        .{ .shape = .{ .n = 10240, .k = 2560 }, .batch = 8 }, // production size
    };

    for (cases) |case| {
        var host = try prepareNvfp4(
            allocator,
            case.shape,
            case.batch,
            weights_random.random(),
            activation_random.random(),
        );
        defer host.deinit();
        try reportCase(try runCase(&harness, &host, hbm));
    }
}

test "nvfp4 gemv handles k % 32 == 16 through the 8-byte load path" {
    var harness = try Harness.open();
    defer harness.close();

    const allocator = std.testing.allocator;
    var weights_random = std.Random.DefaultPrng.init(0x006a_0005);
    var activation_random = std.Random.DefaultPrng.init(0x006a_0006);
    var host = try prepareNvfp4(
        allocator,
        .{ .n = 64, .k = 272 },
        1,
        weights_random.random(),
        activation_random.random(),
    );
    defer host.deinit();
    try reportCase(try runCase(&harness, &host, measuredHbm()));
}

test "fp8 blockwise decode gemv matches the fp32 oracle" {
    var harness = try Harness.open();
    defer harness.close();

    const hbm = measuredHbm();
    const allocator = std.testing.allocator;
    var weights_random = std.Random.DefaultPrng.init(0x006a_0007);
    var activation_random = std.Random.DefaultPrng.init(0x006a_0008);
    const shapes = [_]Shape{
        .{ .n = 10240, .k = 2560 },
        .{ .n = 2560, .k = 6144 },
    };

    for (shapes) |shape| {
        var host = try prepareFp8(
            allocator,
            shape,
            weights_random.random(),
            activation_random.random(),
        );
        defer host.deinit();
        try reportCase(try runCase(&harness, &host, hbm));
    }
}

test "bf16 decode gemv matches the fp32 oracle" {
    var harness = try Harness.open();
    defer harness.close();

    const hbm = measuredHbm();
    const allocator = std.testing.allocator;
    var weights_random = std.Random.DefaultPrng.init(0x006a_0009);
    var activation_random = std.Random.DefaultPrng.init(0x006a_000a);
    const shapes = [_]Shape{
        .{ .n = 10240, .k = 2560 }, // production dense projection
        .{ .n = 2560, .k = 640 }, // small contracted extent
    };

    for (shapes) |shape| {
        var host = try prepareBf16(
            allocator,
            shape,
            weights_random.random(),
            activation_random.random(),
        );
        defer host.deinit();
        try reportCase(try runCase(&harness, &host, hbm));
        try checkNonVacuity(&host);
    }
}

test "two different activation rows give different output" {
    var harness = try Harness.open();
    defer harness.close();

    const allocator = std.testing.allocator;
    const shape: Shape = .{ .n = 512, .k = 640 };
    // Two instances with the same seed produce the same weights: a `std.Random`
    // value is an interface over shared state, so it must not be copied.
    var first_weights = std.Random.DefaultPrng.init(0x006a_000b);
    var second_weights = std.Random.DefaultPrng.init(0x006a_000b);
    var first_random = std.Random.DefaultPrng.init(0x006a_000c);
    var second_random = std.Random.DefaultPrng.init(0x006a_000d);

    var first = try prepareNvfp4(
        allocator,
        shape,
        1,
        first_weights.random(),
        first_random.random(),
    );
    defer first.deinit();
    var second = try prepareNvfp4(
        allocator,
        shape,
        1,
        second_weights.random(),
        second_random.random(),
    );
    defer second.deinit();

    // Same weights, different activations: the kernels must notice.
    try std.testing.expectEqualSlices(u8, first.blob, second.blob);
    try std.testing.expect(!std.mem.eql(f32, first.activations, second.activations));

    const hbm = measuredHbm();
    _ = try runCase(&harness, &first, hbm);
    _ = try runCase(&harness, &second, hbm);
    try std.testing.expect(!std.mem.eql(u16, first.measured, second.measured));
}
