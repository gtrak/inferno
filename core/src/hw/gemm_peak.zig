//! Peak tensor throughput of this GPU, measured with the GEMM kernels the
//! engine actually runs.
//!
//! THEORY.md §3 carried spec-sheet guesses ("M-todo") for the bf16, fp8 and
//! NVFP4 tensor peaks. This module replaces them with numbers: the same kernel
//! family, at `M = N = K = 4096`, median of six runs after one warm-up, so the
//! reported peak and the reported achieved TFLOPS of a real shape are measured
//! the same way and are comparable.
//!
//! The buffers hold arbitrary but *valid* values (no NaN scale bytes): the
//! measurement is about the instruction stream, not the arithmetic, and a NaN
//! would still cost the same.

const std = @import("std");

const cuda_context = @import("cuda_context.zig");
const cuda_driver = @import("cuda_driver.zig");
const gemm = @import("gemm.zig");

const assert = std.debug.assert;

pub const Error = gemm.KernelError ||
    cuda_driver.DriverError ||
    std.mem.Allocator.Error ||
    error{
        impossible_timing,
        timer_unsupported,
    };

/// Runs per measurement, after one warm-up.
pub const sample_count: u32 = 6;

/// Side of the square peak matrix.
pub const side: u32 = 4096;

/// Bytes of one device allocation are a multiple of this.
const page_bytes: u64 = 4096;

/// FLOP of one `M x N x K` matrix product: two per multiply-add.
pub fn flops(shape: gemm.Shape) f64 {
    assert(shape.m > 0);
    assert(shape.n > 0);

    return 2.0 * @as(f64, @floatFromInt(shape.m)) *
        @as(f64, @floatFromInt(shape.n)) *
        @as(f64, @floatFromInt(shape.k));
}

/// The four measured ceilings, in TFLOPS.
pub const Peaks = struct {
    bf16: f64,
    fp8: f64,
    nvfp4_mma: f64,
    nvfp4_dequant_bf16: f64,

    /// Per-element access by name, for the report and the JSON document.
    pub fn slices(peaks: *const Peaks) [4]f64 {
        assert(peaks.bf16 >= 0);
        assert(peaks.nvfp4_mma >= 0);

        return .{ peaks.bf16, peaks.fp8, peaks.nvfp4_mma, peaks.nvfp4_dequant_bf16 };
    }
};

/// Buffers one peak run needs, all square `side` matrices.
const Buffers = struct {
    a: cuda_context.DeviceBuffer,
    a_frag: cuda_context.DeviceBuffer,
    a_scales: cuda_context.DeviceBuffer,
    w_packed: cuda_context.DeviceBuffer,
    w_scales: cuda_context.DeviceBuffer,
    w_frag: cuda_context.DeviceBuffer,
    w_frag_scales: cuda_context.DeviceBuffer,
    w_e4m3: cuda_context.DeviceBuffer,
    w_bf16: cuda_context.DeviceBuffer,
    w_scale: cuda_context.DeviceBuffer,
    c: cuda_context.DeviceBuffer,
};

/// Pseudo-random host bytes, deliberately avoiding the E4M3 NaN codes.
fn fillValid(bytes: []u8, seed: u32) void {
    assert(bytes.len > 0);
    assert(seed != 0);

    var state = seed;
    for (bytes) |*byte| {
        state = state *% 1_664_525 +% 1_013_904_223;
        byte.* = @truncate(state >> 16);
        // 0x7f and 0xff are the only E4M3 NaN codes.
        if (byte.* == 0x7f or byte.* == 0xff) {
            byte.* = 0x7e;
        }
    }
}

/// Rounds a byte count up to a whole allocation page.
fn paged(bytes: u64) u64 {
    assert(page_bytes > 0);
    assert(bytes > 0);

    return (bytes + page_bytes - 1) / page_bytes * page_bytes;
}

/// Measures the four peaks; the caller owns the device context and stream.
pub fn measure(
    allocator: std.mem.Allocator,
    kernels: *const gemm.Kernels,
    stream: *const cuda_context.Stream,
) Error!Peaks {
    assert(side % gemm.tile_n == 0);
    assert(side % gemm.nvfp4_k == 0);

    const shape = gemm.Shape{ .m = side, .n = side, .k = side };
    try shape.validate();
    var buffers = try allocate(kernels, stream, allocator, shape);
    defer release(stream, &buffers);

    var peaks: Peaks = .{
        .bf16 = 0,
        .fp8 = 0,
        .nvfp4_mma = 0,
        .nvfp4_dequant_bf16 = 0,
    };
    const kernel_flops = flops(shape);
    peaks.bf16 = try time(stream, kernel_flops, .bf16, kernels, &buffers, shape);
    peaks.fp8 = try time(stream, kernel_flops, .fp8, kernels, &buffers, shape);
    peaks.nvfp4_mma = try time(stream, kernel_flops, .nvfp4_mma, kernels, &buffers, shape);
    peaks.nvfp4_dequant_bf16 = try time(
        stream,
        kernel_flops,
        .nvfp4_dequant_bf16,
        kernels,
        &buffers,
        shape,
    );
    return peaks;
}

/// Which kernel a timing refers to.
const KernelChoice = enum { bf16, fp8, nvfp4_mma, nvfp4_dequant_bf16 };

/// Launches one kernel once.
fn launchOnce(
    choice: KernelChoice,
    kernels: *const gemm.Kernels,
    buffers: *const Buffers,
    stream: *const cuda_context.Stream,
    shape: gemm.Shape,
) Error!void {
    assert(shape.k % gemm.nvfp4_k == 0);
    assert(buffers.c.bytes > 0);

    switch (choice) {
        .bf16 => try gemm.launchBf16(
            kernels,
            stream,
            &buffers.a,
            &buffers.w_bf16,
            1.0,
            &buffers.c,
            shape,
        ),
        .fp8 => try gemm.launchFp8E4m3(
            kernels,
            stream,
            &buffers.a,
            &buffers.w_e4m3,
            &buffers.w_scale,
            &buffers.c,
            shape,
        ),
        .nvfp4_mma => try gemm.launchNvfp4Mma(
            kernels,
            stream,
            &buffers.a_frag,
            &buffers.a_scales,
            &buffers.w_frag,
            &buffers.w_frag_scales,
            1.0,
            &buffers.c,
            shape,
        ),
        .nvfp4_dequant_bf16 => try gemm.launchNvfp4DequantBf16(
            kernels,
            stream,
            &buffers.a,
            &buffers.w_packed,
            &buffers.w_scales,
            &buffers.w_bf16,
            1.0,
            &buffers.c,
            shape,
        ),
    }
}

/// Median TFLOPS of one kernel: one warm-up, then `sample_count` timed runs.
fn time(
    stream: *const cuda_context.Stream,
    kernel_flops: f64,
    choice: KernelChoice,
    kernels: *const gemm.Kernels,
    buffers: *const Buffers,
    shape: gemm.Shape,
) Error!f64 {
    assert(kernel_flops > 0);
    assert(sample_count >= 6);

    try launchOnce(choice, kernels, buffers, stream, shape);
    try stream.synchronize();

    const start = try cuda_context.Event.create();
    defer start.destroy();
    const end = try cuda_context.Event.create();
    defer end.destroy();

    var samples: [sample_count]u64 = undefined;
    for (&samples) |*sample| {
        try start.record(stream);
        try launchOnce(choice, kernels, buffers, stream, shape);
        try end.record(stream);
        try end.synchronize();
        const elapsed = try cuda_context.Event.elapsed_nanoseconds(&start, &end);
        if (elapsed == 0) {
            return Error.impossible_timing;
        } else {
            sample.* = elapsed;
        }
    }
    const nanoseconds = median(samples[0..]);
    assert(nanoseconds > 0);
    const seconds = @as(f64, @floatFromInt(nanoseconds)) / 1e9;
    const tflops = kernel_flops / seconds / 1e12;
    assert(tflops > 0);
    return tflops;
}

/// Median of a sample array; the caller's buffer is scratch and gets sorted.
pub fn median(samples: []u64) u64 {
    assert(samples.len > 0);
    assert(samples.len <= 64);

    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    const middle = @divFloor(samples.len, 2);
    assert(middle < samples.len);
    return samples[middle];
}

/// Allocates and fills every peak buffer.
fn allocate(
    kernels: *const gemm.Kernels,
    stream: *const cuda_context.Stream,
    allocator: std.mem.Allocator,
    shape: gemm.Shape,
) Error!Buffers {
    assert(shape.m == side);
    assert(shape.n == side);

    const rows = shape.paddedRows();
    const a_bytes = @as(u64, rows) * shape.k * 2;
    const a_frag_bytes = gemm.activationFragmentWords(rows, shape.k) * 4;
    const a_scale_bytes = gemm.activationScaleWords(rows, shape.k) * 4;
    const c_bytes = @as(u64, shape.m) * shape.n * 4;
    const packed_bytes = @as(u64, shape.n) * shape.k / 2;
    const scale_bytes = @as(u64, shape.n) * shape.k / gemm.nvfp4_block;
    const fragment_bytes = gemm.weightFragmentWords(shape.n, shape.k) * 4;
    const fragment_scale_bytes = gemm.weightScaleWords(shape.n, shape.k) * 4;

    var buffers: Buffers = .{
        .a = try cuda_context.DeviceBuffer.allocate(paged(a_bytes)),
        .a_frag = try cuda_context.DeviceBuffer.allocate(paged(a_frag_bytes)),
        .a_scales = try cuda_context.DeviceBuffer.allocate(paged(a_scale_bytes)),
        .w_packed = try cuda_context.DeviceBuffer.allocate(paged(packed_bytes)),
        .w_scales = try cuda_context.DeviceBuffer.allocate(paged(scale_bytes)),
        .w_frag = try cuda_context.DeviceBuffer.allocate(paged(fragment_bytes)),
        .w_frag_scales = try cuda_context.DeviceBuffer.allocate(paged(fragment_scale_bytes)),
        .w_e4m3 = try cuda_context.DeviceBuffer.allocate(paged(@as(u64, shape.n) * shape.k)),
        .w_bf16 = try cuda_context.DeviceBuffer.allocate(paged(@as(u64, shape.n) * shape.k * 2)),
        .w_scale = try cuda_context.DeviceBuffer.allocate(page_bytes),
        .c = try cuda_context.DeviceBuffer.allocate(paged(c_bytes)),
    };
    try fillAll(stream, allocator, &buffers);
    try gemm.quantizeActivations(
        kernels,
        stream,
        &buffers.a,
        &buffers.a_frag,
        &buffers.a_scales,
        shape,
    );
    try stream.synchronize();
    assert(buffers.c.bytes >= c_bytes);
    return buffers;
}

/// Frees every peak buffer, newest first.
fn release(stream: *const cuda_context.Stream, buffers: *Buffers) void {
    assert(@intFromPtr(stream.handle) != 0);
    assert(buffers.c.pointer != 0);

    buffers.c.free();
    buffers.w_scale.free();
    buffers.w_bf16.free();
    buffers.w_e4m3.free();
    buffers.w_frag_scales.free();
    buffers.w_frag.free();
    buffers.w_scales.free();
    buffers.w_packed.free();
    buffers.a_scales.free();
    buffers.a_frag.free();
    buffers.a.free();
}

/// Copies pseudo-random bytes into every buffer, and the unit weight scale.
fn fillAll(
    stream: *const cuda_context.Stream,
    allocator: std.mem.Allocator,
    buffers: *const Buffers,
) Error!void {
    assert(buffers.w_scale.bytes >= @sizeOf(f32));
    assert(buffers.a.bytes > 0);

    const targets = [_]*const cuda_context.DeviceBuffer{
        &buffers.a,
        &buffers.a_frag,
        &buffers.a_scales,
        &buffers.w_packed,
        &buffers.w_scales,
        &buffers.w_frag,
        &buffers.w_frag_scales,
        &buffers.w_e4m3,
        &buffers.w_bf16,
        &buffers.c,
    };
    for (targets, 0..) |target, index| {
        const host = try allocator.alloc(u8, @intCast(target.bytes));
        defer allocator.free(host);
        fillValid(host, @intCast(0x10 + index));
        try gemm.upload(stream, target, host);
    }
    // The unit weight scale is a truth value, not random noise.
    const unit: [4]u8 = @bitCast(@as(f32, 1.0));
    try gemm.upload(stream, &buffers.w_scale, unit[0..]);
    assert(buffers.w_scale.bytes >= @sizeOf(f32));
}

test "flops counts two per multiply-add" {
    const shape = gemm.Shape{ .m = 2, .n = 3, .k = 4 };
    try std.testing.expectEqual(@as(f64, 48), flops(shape));
}

test "median picks the middle of an odd count" {
    var samples = [_]u64{ 9, 1, 5 };
    try std.testing.expectEqual(@as(u64, 5), median(samples[0..]));
}

test "paged rounds up to a whole page" {
    try std.testing.expectEqual(@as(u64, 4096), paged(1));
    try std.testing.expectEqual(@as(u64, 8192), paged(4097));
}
