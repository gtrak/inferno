//! Prefill GEMM harness: every kernel in `gemm.cu` against the f32 oracle.
//!
//! Accuracy is reported as three separate comparisons, never conflated, because
//! they answer three different questions and only two of them are gates
//! (`.agents/tasks/review-gate.md` §5). Every comparison gives both sides the
//! same inputs: the oracle's operands are rounded to the kernel's dtype first.
//! The output is f32 (see the `C` note in `gemm.cu`), so there is no half-bf16
//! floor to sit above; the engine's dtype policy is a model-level question for
//! I-013, never part of a kernel comparison.
//!
//! The three tiers:
//!
//!   tier 1  OPERAND-EXACT, assert <= 1e-6. Both sides get the operands the
//!           kernel actually multiplies (activations quantised with the
//!           kernel's own per-16 recipe, weights dequantised from the repacked
//!           FP4). This pins the kernel machinery: the weight repack, the
//!           A/B/SFA/SFB fragment layouts and the epilogue. Measured 5e-8.
//!   tier 2  W4A16, assert <= 5e-4. The dequantising path against an f32 oracle
//!           on the dequantised weights, with bf16 activations. The
//!           dequantisation is lossless — an e2m1 value times a ue4m3 scale
//!           needs six mantissa bits and bf16 has eight — so the only residual
//!           is f32 accumulation order. Measured 1e-6.
//! Every case also re-runs its kernel and requires bit-identical output, which
//! is what pins the cp.async pipeline and the zero-filled padding.
//!
//!   tier 3  W4A4 against the checkpoint's original f32 weights, REPORTED ONLY:
//!           "labelled W4A4 quality trade - NOT a kernel gate". Measured
//!           1.2e-1 .. 1.8e-1, and that number is a property of FP4, not of this
//!           code: E2M1 rounding of both iid-uniform operands gives ~10 %
//!           per-element RMS, and for iid data the dot-product error scales
//!           with sqrt(K) exactly like the signal, so it never shrinks with K.
//!           W4A4 is a *model-level* quality question, gated at I-020 by
//!           delta-NLL and router top-k agreement; if it proves too lossy the
//!           fallback is `kind::f8f6f4` e2m1 x e4m3 (W4A8), also available here.
//!
//! Two shapes are compared in full; every other shape is compared on a per-tile
//! sample (a row from every 64-row block, two columns from every 64-column
//! block), which catches a tile-, warp- or block-dependent bug without spending
//! minutes in the oracle. Throughput is reported per case against the peak
//! measured in the same run by `hw/gemm_peak.zig`; that peak is also what
//! `inferno-probe` records in `bench/hw-probe.json`.

const std = @import("std");

const cuda_context = @import("../hw/api.zig").cuda_context;
const cuda_driver = @import("../hw/api.zig").cuda_driver;
const gemm = @import("../hw/gemm.zig");
const gemm_peak = @import("../hw/gemm_peak.zig");
const ref_fixtures = @import("../ref/fixtures.zig");
const ref_ops = @import("../ref/ops.zig");
const pack_nvfp4 = @import("../weights/pack_nvfp4.zig");
const quant = @import("../weights/quant.zig");
const generated_cubins = @import("generated_cubins");

const assert = std.debug.assert;

/// Activation row counts the harness exercises, from decode-sized to prefill.
pub const case_rows = [_]u32{ 1, 16, 128, 512, 2048 };

/// Weight shapes the harness exercises: the real projections of the checkpoint.
pub const case_columns = [_]struct { n: u32, k: u32 }{
    .{ .n = 10240, .k = 2560 },
    .{ .n = 6144, .k = 2560 },
    .{ .n = 2560, .k = 6144 },
    .{ .n = 12288, .k = 2560 },
    .{ .n = 640, .k = 2560 },
};

/// Tier 1: the kernel machinery, against the operands it really multiplies.
const kernel_machinery_tolerance: f64 = 1e-6;
/// Round-off budget of the bf16 path: the operands are exact, so only the
/// accumulation order differs.
const bf16_tolerance: f64 = 1e-4;
/// Budget of the e4m3 path. The operands are exact, but the e4m3 MMA truncates
/// its internal accumulator: measured 1.3e-3 of max|ref| at K=2560 against an
/// exact f64 dot product, and flat in K from 128 to 2560.
const fp8_tolerance: f64 = 5e-3;
/// Tier 2: the dequantising NVFP4 path.
const nvfp4_dequant_tolerance: f64 = 5e-4;

/// At or below this many rows the whole matrix is compared, not a sample.
const dense_row_limit: u32 = 16;

/// Which distribution the synthetic fixtures draw from.
///
/// `uniform` is the bounded [-1, 1) distribution the error budgets in the header
/// are reasoned about: every 16-value block has the same amax, which is the
/// worst case for FP4. `realistic` is the shape real tensors have - roughly
/// Gaussian, per-row scale variation, a heavy tail - and is what the
/// production-magnitude case uses, because a value distribution that never
/// stresses the dynamic range is as weak a detector as a tiny shape.
const ValueModel = enum {
    uniform,
    realistic,

    /// A bound on |value|, from which the per-tensor scale follows.
    fn bound(model: ValueModel) f32 {
        return switch (model) {
            .uniform => 1.0,
            .realistic => 2.0,
        };
    }
};

/// Weight magnitudes are uniform in `[-1, 1)` for the bounded model.
const weight_bound: f32 = 1.0;

/// Seed of the fixture generator, so a failure reproduces exactly.
const fixture_seed: u32 = 0x5eed_1234;

/// Runs per timed measurement in the shape table.
const timing_samples: u32 = 3;

/// First component of the output matrix, printed with every case.
const report_components: u32 = 4;

/// Finds the `gemm` cubin in the generated table.
pub fn cubinBytes() ![]const u8 {
    assert(generated_cubins.cubins.len > 0);
    assert(@sizeOf(generated_cubins.Cubin) > 0);

    for (generated_cubins.cubins) |entry| {
        if (std.mem.eql(u8, std.mem.span(entry.name), "gemm")) {
            const bytes = entry.bytes[0..entry.length];
            assert(bytes.len > 0);
            return bytes;
        }
    }
    return error.GemmCubinMissing;
}

/// Linear congruential generator: one fixed stream, no library dependency.
fn nextRandom(state: *u32) u32 {
    assert(state.* != 0);
    assert(fixture_seed != 0);

    state.* = state.* *% 1_664_525 +% 1_013_904_223;
    return state.*;
}

/// A value uniform in `[-1, 1)`: from that bound the per-tensor scale follows
/// without a second pass over the weights.
fn uniformSigned(state: *u32) f32 {
    assert(fixture_seed != 0);
    assert(weight_bound == 1.0);
    assert(ValueModel.uniform.bound() == weight_bound);

    const raw: i32 = @bitCast(nextRandom(state));
    const result = @as(f32, @floatFromInt(raw >> 8)) / 8_388_608.0;
    assert(result >= -1.0);
    assert(result < 1.0);
    return result;
}

/// A roughly standard normal sample: the sum of three uniforms, rescaled. It
/// costs three generator steps instead of a transcendental, and the heavy tail
/// the FP4 trade cares about is added explicitly by the caller.
fn standardNormal(state: *u32) f32 {
    assert(fixture_seed != 0);

    var total: f32 = 0;
    for (0..3) |_| {
        total += (uniformSigned(state) + 1.0) * 0.5;
    }
    const result = (total - 1.5) * 2.0;
    assert(@abs(result) < 3.0);
    return result;
}

/// One realistic weight value: a per-tensor magnitude typical of a projection,
/// per-row variation, and a sparse heavy tail.
fn realisticWeight(state: *u32, row_scale: f32) f32 {
    assert(row_scale > 0);
    assert(row_scale <= 0.1);

    var value = standardNormal(state) * row_scale;
    if (nextRandom(state) % 1000 == 0) {
        value *= 15.0;
    }
    assert(@abs(value) <= ValueModel.realistic.bound());
    return value;
}

/// One realistic activation value: unit-ish scale with per-row variation and a
/// sparse heavy tail, which is what makes per-16 dynamic scaling earn its keep.
fn realisticActivation(state: *u32, row_scale: f32) f32 {
    assert(row_scale > 0);
    assert(row_scale <= 8.0);

    var value = standardNormal(state) * row_scale;
    if (nextRandom(state) % 1000 == 0) {
        value *= 10.0;
    }
    return value;
}

/// Rounds to nearest with ties to even, for a non-negative f32 up to 16.
fn roundHalfEven(value: f32) f32 {
    assert(value >= 0);
    assert(value <= 16.0);

    const floored = @floor(value);
    const fraction = value - floored;
    if (fraction > 0.5) {
        return floored + 1.0;
    } else if (fraction < 0.5) {
        return floored;
    } else {
        // A tie rounds to whichever neighbour is even.
        const half = floored / 2.0;
        return if (@floor(half) == half) floored else floored + 1.0;
    }
}

/// Rounds an f32 to e4m3 with ties to even and saturation at 448, matching
/// `__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3)` in the kernel.
pub fn e4m3Encode(value: f32) u8 {
    assert(value == value);
    assert(@abs(value) <= std.math.floatMax(f32));

    const sign: u8 = if (value < 0) 0x80 else 0x00;
    const magnitude = @abs(value);
    if (magnitude >= quant.fp8_max_magnitude) {
        // 448 = 0b0_1111_110 is the largest finite e4m3 value.
        return sign | 0x7e;
    }
    if (magnitude < 0x1p-10) {
        return sign;
    }
    if (magnitude < 0x1p-6) {
        // Subnormal e4m3: the step is 2^-9, so the mantissa is the count of it.
        const steps = magnitude * 512.0;
        const rounded = roundHalfEven(steps);
        assert(rounded <= 8.0);
        return sign | @as(u8, @intFromFloat(rounded));
    }

    const bits: u32 = @bitCast(magnitude);
    const exponent: i32 = @as(i32, @intCast((bits >> 23) & 0xff)) - 127;
    const dropped = bits & 0xfffff;
    var mantissa: u32 = (bits >> 20) & 0x7;
    if (dropped > 0x80000 or (dropped == 0x80000 and (mantissa & 1) == 1)) {
        mantissa += 1;
    }
    var field: i32 = exponent + 7;
    if (mantissa == 8) {
        mantissa = 0;
        field += 1;
    }
    if (field > 15 or (field == 15 and mantissa > 6)) {
        return sign | 0x7e;
    }
    assert(field >= 1);
    assert(mantissa <= 7);
    return sign | (@as(u8, @intCast(field)) << 3) | @as(u8, @intCast(mantissa));
}

/// Rounds to the nearest e2m1 grid point: the kernel's midpoint rule.
pub fn encodeE2m1(value: f32) u8 {
    assert(value == value);

    const magnitude = @abs(value);
    var code: u8 = 0;
    if (magnitude >= 5.0) {
        code = 7;
    } else if (magnitude >= 3.5) {
        code = 6;
    } else if (magnitude >= 2.5) {
        code = 5;
    } else if (magnitude >= 1.75) {
        code = 4;
    } else if (magnitude >= 1.25) {
        code = 3;
    } else if (magnitude >= 0.75) {
        code = 2;
    } else if (magnitude >= 0.25) {
        code = 1;
    } else {
        code = 0;
    }
    return if (value < 0) code | 8 else code;
}

/// Quantises one f32 weight row into the checkpoint's NVFP4 layout with the same
/// recipe the GPU uses for activations: the block scale is `amax / 6` as ue4m3
/// (here divided by the per-tensor scale that the epilogue re-applies), and the
/// nibbles are formed against the *stored* scale.
pub fn quantizeWeightRow(
    row: []const f32,
    scale2: f32,
    packed_weights: []u8,
    scales: []u8,
) void {
    assert(row.len % quant.nvfp4_block_size == 0);
    assert(packed_weights.len == row.len / 2);
    assert(scales.len == row.len / quant.nvfp4_block_size);
    assert(scale2 > 0);

    const block = quant.nvfp4_block_size;
    for (scales, 0..) |*scale_byte, block_index| {
        const values = row[block_index * block ..][0..block];
        var largest: f32 = 0;
        for (values) |value| {
            largest = @max(largest, @abs(value));
        }
        const requested = if (largest > 0) largest / 6.0 / scale2 else 1.0;
        scale_byte.* = e4m3Encode(requested);
        const stored = quant.f8_e4m3_to_f32(@enumFromInt(scale_byte.*)) * scale2;
        for (values, 0..) |value, offset| {
            const nibble = encodeE2m1(if (stored > 0) value / stored else 0.0);
            const index = block_index * (block / 2) + offset / 2;
            if (offset % 2 == 0) {
                packed_weights[index] = nibble;
            } else {
                packed_weights[index] |= nibble << 4;
            }
        }
    }
    assert(packed_weights.len == row.len / 2);
}

/// One open device, its stream and the five GEMM kernels.
const Harness = struct {
    context: cuda_context.Context,
    stream: cuda_context.Stream,
    kernels: gemm.Kernels,
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
        harness.kernels = try gemm.Kernels.init(try cubinBytes());
        assert(@intFromPtr(harness.stream.handle) != 0);
        return harness;
    }

    fn close(harness: *Harness) void {
        assert(@intFromPtr(harness.stream.handle) != 0);
        assert(@intFromPtr(harness.kernels.bf16.handle) != 0);

        harness.kernels.deinit();
        harness.stream.destroy();
        harness.context.release();
    }
};

/// A device allocation with a host mirror in 32-bit words.
const Staged = struct {
    device: cuda_context.DeviceBuffer,
    words: []u32,

    fn init(
        allocator: std.mem.Allocator,
        harness: *Harness,
        bytes: u64,
    ) !Staged {
        assert(bytes > 0);
        assert(bytes % 4 == 0);

        const rounded = (bytes + gemm.staging_page_bytes - 1) /
            gemm.staging_page_bytes * gemm.staging_page_bytes;
        const words = try allocator.alloc(u32, @intCast(rounded / 4));
        @memset(words, 0);
        const device = try cuda_context.DeviceBuffer.allocate(rounded);
        try gemm.upload(&harness.stream, &device, std.mem.sliceAsBytes(words));
        assert(words.len == rounded / 4);
        return .{ .device = device, .words = words };
    }

    fn deinit(staged: *Staged, allocator: std.mem.Allocator) void {
        assert(staged.words.len > 0);
        assert(staged.device.pointer != 0);

        staged.device.free();
        allocator.free(staged.words);
    }

    fn floats(staged: *const Staged) []const f32 {
        assert(staged.words.len > 0);
        return @ptrCast(staged.words);
    }
};

/// Host fixtures for one `[N, K]` weight matrix, in every layout under test.
const Weights = struct {
    n: u32,
    k: u32,
    scale2: f32,
    packed_weights: []u8,
    scales: []u8,
    e4m3: []u8,
    bf16: []u16,
    fragment: []u32,
    fragment_scales: []u32,
    /// Columns the oracle checks, and their original f32 values.
    sampled: []u32,
    sampled_f32: []f32,
    model: ValueModel,
    device_packed: cuda_context.DeviceBuffer,
    device_scales: cuda_context.DeviceBuffer,
    device_fragment: cuda_context.DeviceBuffer,
    device_fragment_scales: cuda_context.DeviceBuffer,
    device_e4m3: cuda_context.DeviceBuffer,
    device_bf16: cuda_context.DeviceBuffer,
    /// bf16 mirror the exact FP4 path dequantises into.
    device_dequantised: cuda_context.DeviceBuffer,

    fn init(
        allocator: std.mem.Allocator,
        harness: *Harness,
        n: u32,
        k: u32,
        model: ValueModel,
    ) !Weights {
        assert(n % gemm.tile_n == 0);
        assert(k % gemm.nvfp4_k == 0);

        // ModelOpt's per-tensor scale for a tensor bounded by the model's bound.
        const scale2 = model.bound() / (6.0 * quant.fp8_max_magnitude);
        const buffers = pack_nvfp4.weightBuffers(n, k);
        const columns = try sampleColumns(allocator, n);
        errdefer allocator.free(columns);

        var weights: Weights = .{
            .n = n,
            .k = k,
            .scale2 = scale2,
            .packed_weights = try allocator.alloc(u8, @intCast(buffers.packed_bytes)),
            .scales = try allocator.alloc(u8, @intCast(buffers.scale_bytes)),
            .e4m3 = try allocator.alloc(u8, @as(usize, n) * k),
            .bf16 = try allocator.alloc(u16, @as(usize, n) * k),
            .fragment = try allocator.alloc(
                u32,
                @intCast(pack_nvfp4.weightFragmentWords(n, k)),
            ),
            .fragment_scales = try allocator.alloc(
                u32,
                @intCast(pack_nvfp4.weightScaleWords(n, k)),
            ),
            .sampled = columns,
            .sampled_f32 = try allocator.alloc(f32, columns.len * k),
            .model = model,
            .device_packed = undefined,
            .device_scales = undefined,
            .device_fragment = undefined,
            .device_fragment_scales = undefined,
            .device_e4m3 = undefined,
            .device_bf16 = undefined,
            .device_dequantised = undefined,
        };
        try generateWeights(allocator, &weights);

        weights.device_packed = try allocateAndUpload(&harness.stream, weights.packed_weights);
        weights.device_scales = try allocateAndUpload(&harness.stream, weights.scales);
        weights.device_fragment = try allocateAndUpload(
            &harness.stream,
            std.mem.sliceAsBytes(weights.fragment),
        );
        weights.device_fragment_scales = try allocateAndUpload(
            &harness.stream,
            std.mem.sliceAsBytes(weights.fragment_scales),
        );
        weights.device_e4m3 = try allocateAndUpload(&harness.stream, weights.e4m3);
        weights.device_bf16 = try allocateAndUpload(
            &harness.stream,
            std.mem.sliceAsBytes(weights.bf16),
        );
        const dequantised_bytes = @as(usize, n) * k * 2;
        const placeholder = try allocator.alloc(u8, dequantised_bytes);
        defer allocator.free(placeholder);
        @memset(placeholder, 0);
        weights.device_dequantised = try allocateAndUpload(
            &harness.stream,
            placeholder,
        );
        return weights;
    }

    fn deinit(weights: *Weights, allocator: std.mem.Allocator) void {
        assert(weights.sampled.len > 0);
        assert(weights.n % gemm.tile_n == 0);

        weights.device_dequantised.free();
        weights.device_bf16.free();
        weights.device_e4m3.free();
        weights.device_fragment_scales.free();
        weights.device_fragment.free();
        weights.device_scales.free();
        weights.device_packed.free();
        allocator.free(weights.sampled_f32);
        allocator.free(weights.fragment_scales);
        allocator.free(weights.fragment);
        allocator.free(weights.bf16);
        allocator.free(weights.e4m3);
        allocator.free(weights.scales);
        allocator.free(weights.packed_weights);
        allocator.free(weights.sampled);
    }
};

/// Allocates a device buffer and uploads `contents` into it.
fn allocateAndUpload(
    stream: *const cuda_context.Stream,
    contents: []const u8,
) !cuda_context.DeviceBuffer {
    assert(contents.len > 0);
    assert(contents.len <= std.math.maxInt(u32));

    const device = try cuda_context.DeviceBuffer.allocate(@intCast(
        (@as(u64, contents.len) + gemm.staging_page_bytes - 1) /
            gemm.staging_page_bytes * gemm.staging_page_bytes,
    ));
    try gemm.upload(stream, &device, contents);
    assert(device.bytes >= contents.len);
    return device;
}

/// Columns of the weight matrix the oracle checks: two per 64-column block.
fn sampleColumns(allocator: std.mem.Allocator, n: u32) ![]u32 {
    assert(n % gemm.tile_n == 0);
    assert(n > 0);

    const blocks = n / gemm.tile_n;
    const columns = try allocator.alloc(u32, blocks * 2);
    for (0..blocks) |block| {
        const first: u32 = @intCast(block * gemm.tile_n);
        columns[block * 2] = first;
        columns[block * 2 + 1] = first + gemm.tile_n / 2;
    }
    assert(columns.len == blocks * 2);
    return columns;
}

/// Rows of the activation matrix the oracle checks: one per 64-row block plus
/// the last row, or every row when `M <= dense_row_limit`.
fn sampleRows(allocator: std.mem.Allocator, m: u32) ![]u32 {
    assert(m > 0);
    assert(m <= gemm.m_max);

    if (m <= dense_row_limit) {
        const rows = try allocator.alloc(u32, m);
        for (rows, 0..) |*row, index| {
            row.* = @intCast(index);
        }
        return rows;
    }
    const blocks = (m + gemm.tile_m - 1) / gemm.tile_m;
    const rows = try allocator.alloc(u32, blocks + 1);
    for (0..blocks) |block| {
        const row: u32 = @intCast(block * gemm.tile_m);
        rows[block] = if (row < m) row else m - 1;
    }
    rows[blocks] = m - 1;
    assert(rows.len == blocks + 1);
    return rows;
}

/// Fills every host layout of one weight matrix from the same f32 stream.
fn generateWeights(allocator: std.mem.Allocator, weights: *Weights) !void {
    assert(weights.k % quant.nvfp4_block_size == 0);
    assert(weights.sampled_f32.len == weights.sampled.len * weights.k);

    var state = fixture_seed;
    const row = try allocator.alloc(f32, weights.k);
    defer allocator.free(row);

    var sampled_index: usize = 0;
    for (0..weights.n) |n| {
        // Each output row carries its own magnitude, as a trained projection
        // does; the uniform model keeps every row identical on purpose.
        const row_scale = 0.02 * (1.0 + 0.3 * uniformSigned(&state));
        for (row) |*value| {
            value.* = switch (weights.model) {
                .uniform => uniformSigned(&state) * weight_bound,
                .realistic => realisticWeight(&state, row_scale),
            };
        }
        quantizeWeightRow(
            row,
            weights.scale2,
            weights.packed_weights[n * weights.k / 2 ..][0 .. weights.k / 2],
            weights.scales[n * (weights.k / quant.nvfp4_block_size) ..][0..(weights.k / quant.nvfp4_block_size)],
        );
        for (row, 0..) |value, column| {
            weights.e4m3[n * weights.k + column] = e4m3Encode(value);
            weights.bf16[n * weights.k + column] = quant.f32_to_bf16(value);
        }
        if (sampled_index < weights.sampled.len and
            weights.sampled[sampled_index] == n)
        {
            @memcpy(
                weights.sampled_f32[sampled_index * weights.k ..][0..weights.k],
                row,
            );
            sampled_index += 1;
        }
    }
    assert(sampled_index == weights.sampled.len);

    pack_nvfp4.packWeights(
        weights.packed_weights,
        weights.scales,
        weights.n,
        weights.k,
        weights.fragment,
        weights.fragment_scales,
    );
}

/// Host fixtures for one `[M, K]` activation matrix, plus its FP4 scratch.
const Activations = struct {
    shape: gemm.Shape,
    bf16: []u16,
    sampled: []u32,
    sampled_f32: []f32,
    device: cuda_context.DeviceBuffer,
    device_fragment: cuda_context.DeviceBuffer,
    device_scales: cuda_context.DeviceBuffer,

    fn init(
        allocator: std.mem.Allocator,
        harness: *Harness,
        shape: gemm.Shape,
        model: ValueModel,
    ) !Activations {
        assert(shape.m > 0);
        assert(shape.k % gemm.nvfp4_k == 0);

        const rows: usize = @intCast(shape.paddedRows());
        const columns = try sampleRows(allocator, shape.m);
        errdefer allocator.free(columns);

        var state = fixture_seed ^ 0x9e37_79b9;
        const bf16 = try allocator.alloc(u16, rows * shape.k);
        for (0..@as(usize, shape.m)) |row| {
            const row_scale = 1.0 + 0.5 * uniformSigned(&state);
            const values = bf16[row * shape.k ..][0..shape.k];
            for (values) |*value| {
                const sample = switch (model) {
                    .uniform => uniformSigned(&state) * weight_bound,
                    .realistic => realisticActivation(&state, row_scale),
                };
                value.* = quant.f32_to_bf16(sample);
            }
        }
        // The rows past `M` are never read: the kernels zero-fill them.
        const sampled_f32 = try allocator.alloc(f32, columns.len * shape.k);
        for (columns, 0..) |row, slot| {
            for (0..shape.k) |column| {
                sampled_f32[slot * shape.k + column] =
                    quant.bf16_to_f32(bf16[@as(usize, row) * shape.k + column]);
            }
        }

        const fragment_bytes = pack_nvfp4.activationFragmentWords(
            shape.paddedRows(),
            shape.k,
        ) * 4;
        const scale_bytes = pack_nvfp4.activationScaleWords(
            shape.paddedRows(),
            shape.k,
        ) * 4;
        const zero_fragment = try allocator.alloc(u8, @intCast(fragment_bytes));
        defer allocator.free(zero_fragment);
        @memset(zero_fragment, 0);
        const zero_scales = try allocator.alloc(u8, @intCast(scale_bytes));
        defer allocator.free(zero_scales);
        @memset(zero_scales, 0);

        return .{
            .shape = shape,
            .bf16 = bf16,
            .sampled = columns,
            .sampled_f32 = sampled_f32,
            .device = try allocateAndUpload(&harness.stream, std.mem.sliceAsBytes(bf16)),
            .device_fragment = try allocateAndUpload(&harness.stream, zero_fragment),
            .device_scales = try allocateAndUpload(&harness.stream, zero_scales),
        };
    }

    fn deinit(activations: *Activations, allocator: std.mem.Allocator) void {
        assert(activations.sampled.len > 0);
        assert(activations.bf16.len > 0);

        activations.device_scales.free();
        activations.device_fragment.free();
        activations.device.free();
        allocator.free(activations.sampled_f32);
        allocator.free(activations.sampled);
        allocator.free(activations.bf16);
    }
};

/// The four kernels under test.
const KernelId = enum {
    bf16,
    fp8,
    nvfp4_dequant_bf16,
    nvfp4_mma,

    fn nameOf(id: KernelId) []const u8 {
        return switch (id) {
            .bf16 => "gemm_bf16",
            .fp8 => "gemm_fp8_e4m3",
            .nvfp4_dequant_bf16 => "gemm_nvfp4_bf16",
            .nvfp4_mma => "gemm_nvfp4_mma",
        };
    }

    fn toleranceOf(id: KernelId) f64 {
        return switch (id) {
            .bf16 => bf16_tolerance,
            .fp8 => fp8_tolerance,
            .nvfp4_dequant_bf16 => nvfp4_dequant_tolerance,
            .nvfp4_mma => kernel_machinery_tolerance,
        };
    }

    fn peakOf(id: KernelId, peaks: *const gemm_peak.Peaks) f64 {
        assert(peaks.nvfp4_mma > 0);

        return switch (id) {
            .bf16 => peaks.bf16,
            .fp8 => peaks.fp8,
            .nvfp4_dequant_bf16 => peaks.nvfp4_dequant_bf16,
            .nvfp4_mma => peaks.nvfp4_mma,
        };
    }
};

/// The e4m3 path is unscaled: the weights carry their own magnitude.
const fp8_unit_scale: f32 = 1.0;

/// Everything one case needs, alive for the length of the case.
const Case = struct {
    harness: *Harness,
    weights: *Weights,
    activations: *Activations,
    output: Staged,
    unit_scale: cuda_context.DeviceBuffer,

    fn init(
        allocator: std.mem.Allocator,
        harness: *Harness,
        weights: *Weights,
        activations: *Activations,
    ) !Case {
        assert(weights.n % gemm.tile_n == 0);
        assert(activations.shape.m > 0);

        const shape = activations.shape;
        const unit: [4]u8 = @bitCast(fp8_unit_scale);
        return .{
            .harness = harness,
            .weights = weights,
            .activations = activations,
            .output = try Staged.init(allocator, harness, @as(u64, shape.m) * shape.n * 4),
            .unit_scale = try allocateAndUpload(&harness.stream, unit[0..]),
        };
    }

    fn deinit(case: *Case, allocator: std.mem.Allocator) void {
        assert(case.weights.n > 0);
        assert(case.activations.shape.m > 0);

        case.unit_scale.free();
        case.output.deinit(allocator);
    }

    /// Launches one kernel once.
    fn launch(case: *Case, id: KernelId) !void {
        assert(case.activations.shape.k % gemm.nvfp4_k == 0);
        assert(case.output.device.pointer != 0);

        const kernels = &case.harness.kernels;
        const stream = &case.harness.stream;
        const shape = case.activations.shape;
        switch (id) {
            .bf16 => try gemm.launchBf16(
                kernels,
                stream,
                &case.activations.device,
                &case.weights.device_bf16,
                1.0,
                &case.output.device,
                shape,
            ),
            .fp8 => try gemm.launchFp8E4m3(
                kernels,
                stream,
                &case.activations.device,
                &case.weights.device_e4m3,
                &case.unit_scale,
                &case.output.device,
                shape,
            ),
            .nvfp4_dequant_bf16 => try gemm.launchNvfp4DequantBf16(
                kernels,
                stream,
                &case.activations.device,
                &case.weights.device_packed,
                &case.weights.device_scales,
                &case.weights.device_dequantised,
                case.weights.scale2,
                &case.output.device,
                shape,
            ),
            .nvfp4_mma => try gemm.launchNvfp4Mma(
                kernels,
                stream,
                &case.activations.device_fragment,
                &case.activations.device_scales,
                &case.weights.device_fragment,
                &case.weights.device_fragment_scales,
                case.weights.scale2,
                &case.output.device,
                shape,
            ),
        }
    }

    /// One warm-up, then the median of `timing_samples` timed launches.
    fn time(case: *Case, id: KernelId, flops: f64) !f64 {
        assert(flops > 0);
        assert(timing_samples >= 3);

        const stream = &case.harness.stream;
        try case.launch(id);
        try stream.synchronize();

        const start = try cuda_context.Event.create();
        defer start.destroy();
        const end = try cuda_context.Event.create();
        defer end.destroy();

        var samples: [timing_samples]u64 = undefined;
        for (&samples) |*sample| {
            try start.record(stream);
            try case.launch(id);
            try end.record(stream);
            try end.synchronize();
            const elapsed = try cuda_context.Event.elapsed_nanoseconds(&start, &end);
            if (elapsed == 0) {
                return error.ImpossibleTiming;
            } else {
                sample.* = elapsed;
            }
        }
        const nanoseconds = gemm_peak.median(samples[0..]);
        const seconds = @as(f64, @floatFromInt(nanoseconds)) / 1e9;
        const tflops = flops / seconds / 1e12;
        assert(tflops > 0);
        return tflops;
    }

    /// Downloads the whole output matrix into the host mirror.
    fn download(case: *Case) ![]const f32 {
        assert(case.output.words.len > 0);
        assert(case.output.device.bytes >= case.output.words.len * 4);

        try gemm.download(
            &case.harness.stream,
            std.mem.sliceAsBytes(case.output.words),
            &case.output.device,
        );
        return case.output.floats();
    }
};

/// One kernel/shape result.
const Result = struct {
    max_abs_err: f64,
    max_reference: f64,
    samples: u32,
    distinct: u32,
    tflops: f64,
    peak: f64,

    fn relative(result: *const Result) f64 {
        assert(result.samples > 0);

        if (result.max_reference > 0) {
            return result.max_abs_err / result.max_reference;
        } else {
            // A zero reference means the oracle produced nothing: never a pass.
            return std.math.inf(f64);
        }
    }
};

/// Widens either engine's sample to f64 without a same-type `@floatCast`.
fn widen(value: anytype) f64 {
    if (@TypeOf(value) == f64) {
        return value;
    } else {
        return @floatCast(value);
    }
}

/// Compares the sampled outputs against one oracle engine's values.
fn compare(
    case: *Case,
    output: []const f32,
    reference: anytype,
) !Result {
    assert(reference.len > 0);
    assert(output.len >= @as(usize, case.activations.shape.m) * case.activations.shape.n);

    const shape = case.activations.shape;
    const rows = case.activations.sampled;
    const columns = case.weights.sampled;
    assert(rows.len * columns.len == reference.len);

    var max_abs_err: f64 = 0;
    var max_reference: f64 = 0;
    for (rows, 0..) |row, row_slot| {
        for (columns, 0..) |column, column_slot| {
            const actual = output[@as(usize, row) * shape.n + column];
            const expected: f64 =
                widen(reference[row_slot * columns.len + column_slot]);
            max_abs_err = @max(max_abs_err, @abs(@as(f64, actual) - expected));
            max_reference = @max(max_reference, @abs(expected));
        }
    }

    // Non-vacuity: a kernel that wrote one tile would leave a constant field.
    var distinct: u32 = 0;
    var previous = output[0];
    var index: usize = 0;
    while (index < output.len) : (index += 997) {
        if (output[index] != previous) {
            distinct += 1;
            previous = output[index];
        }
    }
    if (distinct == 0) {
        return error.ConstantOutput;
    }
    return .{
        .max_abs_err = max_abs_err,
        .max_reference = max_reference,
        .samples = @intCast(rows.len * columns.len),
        .distinct = distinct,
        .tflops = 0,
        .peak = 0,
    };
}

/// Which operands the oracle multiplies.
///
/// `as_multiplied` is what the kernel actually multiplies: for the FP4 path
/// that is the quantised (and therefore dequantised) activation and weight, so
/// a mismatch there is a kernel or repack defect. `original` is the f32 matrix
/// before quantization, so a mismatch there is the W4A4 trade itself.
const Basis = enum { as_multiplied, original };

/// Fills `out` with the activation values `id` multiplies, in f32.
fn fillActivationRow(
    id: KernelId,
    basis: Basis,
    activations: *const Activations,
    slot: usize,
    out: []f32,
) void {
    assert(out.len == activations.shape.k);
    assert(slot < activations.sampled.len);

    const k = activations.shape.k;
    const row: usize = activations.sampled[slot];
    const source = activations.bf16[row * k ..][0..k];
    switch (id) {
        .bf16 => for (source, 0..) |value, index| {
            out[index] = quant.bf16_to_f32(value);
        },
        .fp8 => for (source, 0..) |value, index| {
            // The kernel converts the bf16 value to e4m3 in registers.
            const byte = e4m3Encode(quant.bf16_to_f32(value));
            out[index] = quant.f8_e4m3_to_f32(@enumFromInt(byte));
        },
        .nvfp4_dequant_bf16, .nvfp4_mma => @memcpy(
            out,
            activations.sampled_f32[slot * k ..][0..k],
        ),
    }
    if (id == .nvfp4_mma and basis == .as_multiplied) {
        // The fast path quantises the activations itself, per 16 values, with
        // no per-tensor scale; reproduce that grid exactly.
        var packed_weights: [activation_row_bytes_max]u8 = undefined;
        var scales: [activation_scale_bytes_max]u8 = undefined;
        quantizeActivationRow(
            out,
            packed_weights[0 .. out.len / 2],
            scales[0 .. out.len / quant.nvfp4_block_size],
        );
    }
}

/// Room for one activation row quantised at the largest k the harness runs.
const activation_row_bytes_max: u32 = 6144 / 2;
const activation_scale_bytes_max: u32 = 6144 / 16;

/// Quantises then dequantises an activation row in place, with the kernel's
/// per-16 `scale_vec::4X` recipe (`amax / 6` as ue4m3, no per-tensor scale).
fn quantizeActivationRow(
    row: []f32,
    packed_weights: []u8,
    scales: []u8,
) void {
    assert(row.len % quant.nvfp4_block_size == 0);
    assert(packed_weights.len == row.len / 2);
    assert(scales.len == row.len / quant.nvfp4_block_size);

    quantizeWeightRow(row, 1.0, packed_weights, scales);
    quant.dequant_nvfp4_row(
        packed_weights,
        @ptrCast(scales),
        1.0,
        row,
    );
}

/// Fills `out` with the weight values `id` multiplies, in f32.
fn fillWeightRow(
    id: KernelId,
    basis: Basis,
    weights: *const Weights,
    slot: usize,
    out: []f32,
) void {
    assert(out.len == weights.k);
    assert(slot < weights.sampled.len);

    const k = weights.k;
    const column: usize = weights.sampled[slot];
    switch (id) {
        .bf16 => for (weights.bf16[column * k ..][0..k], 0..) |value, index| {
            out[index] = quant.bf16_to_f32(value);
        },
        .fp8 => for (weights.e4m3[column * k ..][0..k], 0..) |byte, index| {
            out[index] = quant.f8_e4m3_to_f32(@enumFromInt(byte)) * fp8_unit_scale;
        },
        .nvfp4_dequant_bf16 => quant.dequant_nvfp4_row(
            weights.packed_weights[column * k / 2 ..][0 .. k / 2],
            @ptrCast(weights.scales[column * (k / quant.nvfp4_block_size) ..][0..(k / quant.nvfp4_block_size)]),
            weights.scale2,
            out,
        ),
        .nvfp4_mma => switch (basis) {
            // The weight the kernel holds: the dequantised FP4 matrix.
            .as_multiplied => quant.dequant_nvfp4_row(
                weights.packed_weights[column * k / 2 ..][0 .. k / 2],
                @ptrCast(weights.scales[column * (k / quant.nvfp4_block_size) ..][0..(k / quant.nvfp4_block_size)]),
                weights.scale2,
                out,
            ),
            // The checkpoint's f32 weights before quantization.
            .original => @memcpy(out, weights.sampled_f32[slot * k ..][0..k]),
        },
    }
}

/// The oracle's two CPU engines, evaluated on the same operands.
///
/// Both are CPU references; neither is a GPU implementation, which is the
/// property plan.md §8.2 requires. `cpu` is `ref/ops.zig`'s `dot` - the
/// sanctioned I-05 reference, f32 accumulated in order, which is the arithmetic
/// the PyTorch reference performs. `precise` accumulates in f64, so it is
/// sharper than *any* f32 summation order and can therefore judge the kernel's
/// own accumulation order; that is what the tightest tier needs. A case fails
/// if the two engines disagree, which is how a broken oracle shows up instead of
/// looking like a kernel defect.
const References = struct {
    precise: []f64,
    cpu: []f32,

    fn deinit(references: *const References, allocator: std.mem.Allocator) void {
        assert(references.precise.len == references.cpu.len);
        assert(references.precise.len > 0);

        allocator.free(references.cpu);
        allocator.free(references.precise);
    }
};

/// The largest disagreement the two CPU engines may have, relative to max|precise|.
/// It is the f32 reference's own accumulation noise, and it bounds how much a
/// comparison against `precise` can differ from a comparison against `cpu`.
const oracle_agreement_tolerance: f64 = 1e-4;

/// The f32 oracle values for every sampled (row, column) pair of one case.
fn referenceValues(
    allocator: std.mem.Allocator,
    id: KernelId,
    basis: Basis,
    case: *Case,
) !References {
    assert(case.activations.sampled.len > 0);
    assert(case.weights.sampled.len > 0);

    const rows = case.activations.sampled.len;
    const columns = case.weights.sampled.len;
    const k = case.activations.shape.k;
    const activation_row = try allocator.alloc(f32, k);
    defer allocator.free(activation_row);
    const weight_row = try allocator.alloc(f32, k);
    defer allocator.free(weight_row);
    const precise = try allocator.alloc(f64, rows * columns);
    errdefer allocator.free(precise);
    const cpu = try allocator.alloc(f32, rows * columns);
    errdefer allocator.free(cpu);

    for (0..rows) |row_slot| {
        fillActivationRow(id, basis, case.activations, row_slot, activation_row);
        for (0..columns) |column_slot| {
            fillWeightRow(id, basis, case.weights, column_slot, weight_row);
            var total: f64 = 0;
            for (activation_row, weight_row) |a, b| {
                total += @as(f64, a) * @as(f64, b);
            }
            precise[row_slot * columns + column_slot] = total;
            cpu[row_slot * columns + column_slot] =
                ref_ops.dot(activation_row, weight_row);
        }
    }
    assert(precise.len == rows * columns);
    return .{ .precise = precise, .cpu = cpu };
}

/// Requires the two CPU engines to agree, so a broken oracle fails loudly.
fn expectOracleAgreement(references: *const References) !void {
    assert(references.precise.len == references.cpu.len);
    assert(references.cpu.len > 0);

    var largest: f64 = 0;
    var disagreement: f64 = 0;
    for (references.precise, references.cpu) |precise, cpu| {
        largest = @max(largest, @abs(precise));
        disagreement = @max(disagreement, @abs(precise - @as(f64, cpu)));
    }
    if (largest <= 0) {
        // An all-zero oracle is a broken oracle, never a pass.
        return error.EmptyOracle;
    }
    if (disagreement / largest > oracle_agreement_tolerance) {
        std.debug.print(
            "  FAIL oracle engines disagree: {e:.3} > {e:.3} of max|ref|\n",
            .{ disagreement / largest, oracle_agreement_tolerance },
        );
        return error.OracleDisagreement;
    }
    assert(disagreement >= 0);
}

/// Runs one kernel on one shape: times it, checks it against the oracle, prints
/// one line and returns the result.
fn checkKernel(
    allocator: std.mem.Allocator,
    case: *Case,
    id: KernelId,
    peaks: *const gemm_peak.Peaks,
) !Result {
    assert(case.activations.shape.k % gemm.nvfp4_k == 0);
    assert(case.activations.sampled.len > 0);

    const shape = case.activations.shape;
    // One launch before the first read: the output buffer starts zeroed.
    try case.launch(id);
    try case.harness.stream.synchronize();
    const output = try case.download();

    const multiplied = try referenceValues(allocator, id, .as_multiplied, case);
    defer multiplied.deinit(allocator);
    try expectOracleAgreement(&multiplied);
    var result = try compare(case, output, multiplied.precise);
    // For every tier whose budget is above the f32 engine's own noise, the
    // sanctioned I-05 reference gates the result as well.
    var sanctioned = try compare(case, output, multiplied.cpu);
    sanctioned.tflops = 0;

    // Acceptance: the same input must produce bit-identical output. The copy
    // is taken before the timed launches, so this also proves the pipeline is
    // deterministic across repeated runs, not just across two.
    const first = try allocator.dupe(f32, output);
    defer allocator.free(first);
    result.tflops = try case.time(id, gemm_peak.flops(shape));
    sanctioned.tflops = result.tflops;
    sanctioned.peak = result.peak;
    for (first, try case.download()) |before, after| {
        if (@as(u32, @bitCast(before)) != @as(u32, @bitCast(after))) {
            std.debug.print(
                "  FAIL {s} at M={d} N={d} K={d}: output is not bit-identical " ++
                    "across launches ({d} vs {d})\n",
                .{ KernelId.nameOf(id), shape.m, shape.n, shape.k, before, after },
            );
            return error.NotRepeatable;
        }
    }
    result.peak = KernelId.peakOf(id, peaks);

    std.debug.print(
        "  M={d:<5} N={d:<6} K={d:<5} {s:<18} err/|ref|={e:.3} (f64) " ++
            "{e:.3} (ref/ops.zig) max_abs={e:.3} max|ref|={e:.3} ({d} refs, " ++
            "{d} distinct, bit-identical) {d:.2} TFLOPS = {d:.1}% of peak",
        .{
            shape.m,
            shape.n,
            shape.k,
            KernelId.nameOf(id),
            result.relative(),
            sanctioned.relative(),
            result.max_abs_err,
            result.max_reference,
            result.samples,
            result.distinct,
            result.tflops,
            if (result.peak > 0) result.tflops / result.peak * 100.0 else 0.0,
        },
    );
    if (id == .nvfp4_mma) {
        // The labelled W4A4 trade: the same output against the checkpoint's
        // f32 weights and f32 activations, i.e. before quantization.
        const original = try referenceValues(allocator, id, .original, case);
        defer original.deinit(allocator);
        try expectOracleAgreement(&original);
        const trade = try compare(case, output, original.precise);
        std.debug.print(
            " vs_original={e:.3} (f64) max_abs={e:.3} max|ref|={e:.3} " ++
                "(labelled W4A4 trade, not a gate)",
            .{ trade.relative(), trade.max_abs_err, trade.max_reference },
        );
    }
    std.debug.print("  C[0][0..{d}] =", .{report_components});
    for (output[0..report_components]) |value| {
        std.debug.print(" {d:.6}", .{value});
    }
    std.debug.print("\n", .{});

    // The f32 engine's noise is the only thing keeping it from gating the
    // tightest tier; everywhere else it must pass the same budget.
    const gauged = if (id.toleranceOf() >= oracle_agreement_tolerance)
        @max(result.relative(), sanctioned.relative())
    else
        result.relative();
    if (gauged > id.toleranceOf()) {
        std.debug.print(
            "  FAIL {s} at M={d} N={d} K={d}: {e:.3} > {e:.3}\n",
            .{
                KernelId.nameOf(id),
                shape.m,
                shape.n,
                shape.k,
                gauged,
                id.toleranceOf(),
            },
        );
        return error.ToleranceExceeded;
    }
    return result;
}

test "e4m3Encode rounds to nearest even and saturates" {
    try std.testing.expectEqual(@as(u8, 0x00), e4m3Encode(0.0));
    try std.testing.expectEqual(@as(u8, 0x38), e4m3Encode(1.0));
    try std.testing.expectEqual(@as(u8, 0xb8), e4m3Encode(-1.0));
    try std.testing.expectEqual(@as(u8, 0x7e), e4m3Encode(448.0));
    try std.testing.expectEqual(@as(u8, 0x7e), e4m3Encode(1000.0));
    try std.testing.expectEqual(@as(u8, 0xfe), e4m3Encode(-1000.0));
    // 1 + 1/16 is the tie between 1.0 and 1.125; even is 1.0.
    try std.testing.expectEqual(@as(u8, 0x38), e4m3Encode(1.0625));
    // 1 + 3/16 is the tie between 1.125 and 1.25; even is 1.25.
    try std.testing.expectEqual(@as(u8, 0x3a), e4m3Encode(1.1875));
    // The subnormal step is 2^-9.
    try std.testing.expectEqual(@as(u8, 0x01), e4m3Encode(0x1p-9));
    try std.testing.expectEqual(@as(u8, 0x02), e4m3Encode(0x1p-8));
}

test "encodeE2m1 uses the kernel's midpoints" {
    try std.testing.expectEqual(@as(u8, 0), encodeE2m1(0.0));
    try std.testing.expectEqual(@as(u8, 1), encodeE2m1(0.3));
    try std.testing.expectEqual(@as(u8, 2), encodeE2m1(0.9));
    try std.testing.expectEqual(@as(u8, 7), encodeE2m1(6.0));
    try std.testing.expectEqual(@as(u8, 0x8 | 7), encodeE2m1(-6.0));
}

test "quantizeWeightRow lands within one e2m1 step of the input" {
    const k: u32 = 32;
    var row: [k]f32 = undefined;
    for (&row, 0..) |*value, index| {
        value.* = @as(f32, @floatFromInt(index)) / 16.0 - 1.0;
    }
    var packed_weights: [k / 2]u8 = undefined;
    var scales: [k / quant.nvfp4_block_size]u8 = undefined;
    const scale2: f32 = weight_bound / (6.0 * quant.fp8_max_magnitude);
    quantizeWeightRow(row[0..], scale2, packed_weights[0..], scales[0..]);

    var out: [k]f32 = undefined;
    quant.dequant_nvfp4_row(packed_weights[0..], @ptrCast(scales[0..]), scale2, out[0..]);
    // Rounding to the e2m1 grid costs at most half a step, and a step is
    // amax/6 for the 16-value block the value belongs to.
    const block = quant.nvfp4_block_size;
    for (0..k / block) |block_index| {
        var largest: f32 = 0;
        for (row[block_index * block ..][0..block]) |value| {
            largest = @max(largest, @abs(value));
        }
        for (row[block_index * block ..][0..block], out[block_index * block ..][0..block]) |expected, actual| {
            // The e2m1 grid is 0, .5, 1, 1.5, 2, 3, 4, 6: near the top a step
            // is 2 units of the block scale, so half a step is amax/6.
            try std.testing.expect(@abs(expected - actual) <= largest / 6.0 + 1e-6);
        }
    }
}

test "sample plans cover every tile of a shape" {
    const allocator = std.testing.allocator;
    const rows = try sampleRows(allocator, 2048);
    defer allocator.free(rows);
    try std.testing.expectEqual(@as(usize, 33), rows.len);
    try std.testing.expectEqual(@as(u32, 2047), rows[rows.len - 1]);

    const dense = try sampleRows(allocator, 16);
    defer allocator.free(dense);
    try std.testing.expectEqual(@as(usize, 16), dense.len);

    const columns = try sampleColumns(allocator, 10240);
    defer allocator.free(columns);
    try std.testing.expectEqual(@as(usize, 320), columns.len);
}

test {
    // The repack is part of this contract: its tests run with the harness.
    _ = @import("../weights/pack_nvfp4.zig");
}

test "the GEMM peaks at 4096 are measured, not assumed" {
    var harness = try Harness.open();
    defer harness.close();

    const peaks = try gemm_peak.measure(
        std.testing.allocator,
        &harness.kernels,
        &harness.stream,
    );
    std.debug.print(
        "peaks at M=N=K={d}: bf16 {d:.2} TFLOPS, fp8 {d:.2}, nvfp4_mma {d:.2}, " ++
            "nvfp4_dequant_bf16 {d:.2}\n",
        .{ gemm_peak.side, peaks.bf16, peaks.fp8, peaks.nvfp4_mma, peaks.nvfp4_dequant_bf16 },
    );
    try std.testing.expect(peaks.bf16 > 0);
    try std.testing.expect(peaks.fp8 > 0);
    try std.testing.expect(peaks.nvfp4_mma > 0);
    try std.testing.expect(peaks.nvfp4_dequant_bf16 > 0);
    // The whole point of the issue: native FP4 must beat the bf16-dequant path.
    try std.testing.expect(peaks.nvfp4_mma > peaks.nvfp4_dequant_bf16);
}

test "every GEMM kernel matches the f32 oracle across the shape table" {
    const allocator = std.testing.allocator;
    var harness = try Harness.open();
    defer harness.close();

    const peaks = try gemm_peak.measure(
        allocator,
        &harness.kernels,
        &harness.stream,
    );
    std.debug.print(
        "peaks at M=N=K={d}: bf16 {d:.2} TFLOPS, fp8 {d:.2}, nvfp4_mma {d:.2}, " ++
            "nvfp4_dequant_bf16 {d:.2}\n",
        .{ gemm_peak.side, peaks.bf16, peaks.fp8, peaks.nvfp4_mma, peaks.nvfp4_dequant_bf16 },
    );

    var mma_total: f64 = 0;
    var dequant_total: f64 = 0;
    for (case_columns) |nk| {
        var weights = try Weights.init(allocator, &harness, nk.n, nk.k, .uniform);
        defer weights.deinit(allocator);
        std.debug.print(
            "-- N={d} K={d} (scale2={e:.4})\n",
            .{ nk.n, nk.k, weights.scale2 },
        );
        for (case_rows) |rows| {
            const shape = gemm.Shape{ .m = rows, .n = nk.n, .k = nk.k };
            try runShape(allocator, &harness, &weights, &peaks, shape);
        }
    }

    // The payoff of native FP4 MMA: the ratio of the two paths at the largest
    // shape, which is the input to the prefill ceiling in THEORY.md §3.
    {
        const shape = gemm.Shape{
            .m = case_rows[case_rows.len - 1],
            .n = case_columns[0].n,
            .k = case_columns[0].k,
        };
        var weights = try Weights.init(allocator, &harness, shape.n, shape.k, .uniform);
        defer weights.deinit(allocator);
        var activations = try Activations.init(allocator, &harness, shape, .uniform);
        defer activations.deinit(allocator);
        var case = try Case.init(allocator, &harness, &weights, &activations);
        defer case.deinit(allocator);
        try gemm.quantizeActivations(
            &harness.kernels,
            &harness.stream,
            &activations.device,
            &activations.device_fragment,
            &activations.device_scales,
            shape,
        );
        const flops = gemm_peak.flops(shape);
        mma_total = try case.time(.nvfp4_mma, flops);
        dequant_total = try case.time(.nvfp4_dequant_bf16, flops);
        std.debug.print(
            "payoff at M={d} N={d} K={d}: nvfp4_mma {d:.2} / nvfp4_dequant_bf16 " ++
                "{d:.2} TFLOPS = {d:.2}x\n",
            .{ shape.m, shape.n, shape.k, mma_total, dequant_total, mma_total / dequant_total },
        );
        try std.testing.expect(mma_total > dequant_total);
    }
}

/// Runs the four kernels on one shape and reports the floors.
fn runShape(
    allocator: std.mem.Allocator,
    harness: *Harness,
    weights: *Weights,
    peaks: *const gemm_peak.Peaks,
    shape: gemm.Shape,
) !void {
    assert(shape.n == weights.n);
    assert(shape.k == weights.k);

    var activations = try Activations.init(allocator, harness, shape, weights.model);
    defer activations.deinit(allocator);
    var case = try Case.init(allocator, harness, weights, &activations);
    defer case.deinit(allocator);

    // The FP4 fast path consumes the quantised activations, produced once here.
    try gemm.quantizeActivations(
        &harness.kernels,
        &harness.stream,
        &activations.device,
        &activations.device_fragment,
        &activations.device_scales,
        shape,
    );
    try harness.stream.synchronize();

    const ids = [_]KernelId{ .bf16, .fp8, .nvfp4_dequant_bf16, .nvfp4_mma };
    for (ids) |id| {
        _ = try checkKernel(allocator, &case, id, peaks);
    }
    assert(shape.m > 0);
}

test "the driver and weight layers size the FP4 buffers identically" {
    // `hw/gemm.zig` sizes buffers for the probe, which compiles `hw/` alone;
    // `weights/pack_nvfp4.zig` owns the layout and sizes the same buffers for
    // the repack. Both spellings must agree for every shape under test.
    for (case_rows) |rows| {
        for (case_columns) |nk| {
            const padded = gemm.Shape{ .m = rows, .n = nk.n, .k = nk.k };
            try std.testing.expectEqual(
                pack_nvfp4.activationFragmentWords(padded.paddedRows(), nk.k),
                gemm.activationFragmentWords(padded.paddedRows(), nk.k),
            );
            try std.testing.expectEqual(
                pack_nvfp4.activationScaleWords(padded.paddedRows(), nk.k),
                gemm.activationScaleWords(padded.paddedRows(), nk.k),
            );
            try std.testing.expectEqual(
                pack_nvfp4.weightFragmentWords(nk.n, nk.k),
                gemm.weightFragmentWords(nk.n, nk.k),
            );
            try std.testing.expectEqual(
                pack_nvfp4.weightScaleWords(nk.n, nk.k),
                gemm.weightScaleWords(nk.n, nk.k),
            );
        }
    }
}

/// The tiny checkpoint's own tensors, in every layout the four kernels take.
///
/// Real values matter here: the synthetic table uses a bounded uniform
/// distribution so its error budgets can be reasoned about, while this case runs
/// the committed golden weights, which have the magnitudes and the outliers of
/// an actual projection.
const Golden = struct {
    shape: gemm.Shape,
    a: cuda_context.DeviceBuffer,
    a_fragment: cuda_context.DeviceBuffer,
    a_scales: cuda_context.DeviceBuffer,
    weights_bf16: cuda_context.DeviceBuffer,
    weights_e4m3: cuda_context.DeviceBuffer,
    weights_packed: cuda_context.DeviceBuffer,
    weights_scales: cuda_context.DeviceBuffer,
    weights_fragment: cuda_context.DeviceBuffer,
    weights_fragment_scales: cuda_context.DeviceBuffer,
    weights_dequantised: cuda_context.DeviceBuffer,
    unit_scale: cuda_context.DeviceBuffer,
    output: Staged,
    /// Host mirrors of what each kernel multiplies, for `ref/ops.zig`. The
    /// activations appear twice because `gemm_fp8_e4m3` converts them to e4m3
    /// in registers, so its oracle must multiply the converted values.
    a_f32: []f32,
    a_e4m3_f32: []f32,
    weights_bf16_f32: []f32,
    weights_e4m3_f32: []f32,
    weights_dequantised_f32: []f32,
    scale2: f32,
};

/// Builds every layout of the golden case from the checkpoint's f32 tensors.
fn buildGolden(
    allocator: std.mem.Allocator,
    harness: *Harness,
    token_rows: []const f32,
    weight_rows: []const f32,
    shape: gemm.Shape,
) !Golden {
    assert(shape.m > 0 and shape.n > 0 and shape.k > 0);
    assert(token_rows.len >= shape.m * shape.k);
    assert(weight_rows.len == shape.n * shape.k);

    const k = shape.k;
    var largest: f32 = 0;
    for (weight_rows) |value| {
        largest = @max(largest, @abs(value));
    }
    try std.testing.expect(largest > 0);
    // ModelOpt's per-tensor scale for the weights' own maximum.
    const scale2 = largest / (6.0 * quant.fp8_max_magnitude);

    const a_f32 = try allocator.alloc(f32, shape.m * k);
    errdefer allocator.free(a_f32);
    const a_e4m3_f32 = try allocator.alloc(f32, shape.m * k);
    errdefer allocator.free(a_e4m3_f32);
    const weights_bf16_f32 = try allocator.alloc(f32, shape.n * k);
    errdefer allocator.free(weights_bf16_f32);
    const weights_e4m3_f32 = try allocator.alloc(f32, shape.n * k);
    errdefer allocator.free(weights_e4m3_f32);
    const weights_dequantised_f32 = try allocator.alloc(f32, shape.n * k);
    errdefer allocator.free(weights_dequantised_f32);

    const a_bf16 = try allocator.alloc(u16, shape.m * k);
    defer allocator.free(a_bf16);
    const weights_bf16 = try allocator.alloc(u16, shape.n * k);
    defer allocator.free(weights_bf16);
    const weights_e4m3 = try allocator.alloc(u8, shape.n * k);
    defer allocator.free(weights_e4m3);
    const weights_packed = try allocator.alloc(u8, shape.n * k / 2);
    defer allocator.free(weights_packed);
    const weights_scales = try allocator.alloc(u8, shape.n * k / quant.nvfp4_block_size);
    defer allocator.free(weights_scales);
    const weights_fragment = try allocator.alloc(
        u32,
        @intCast(pack_nvfp4.weightFragmentWords(shape.n, k)),
    );
    defer allocator.free(weights_fragment);
    const weights_fragment_scales = try allocator.alloc(
        u32,
        @intCast(pack_nvfp4.weightScaleWords(shape.n, k)),
    );
    defer allocator.free(weights_fragment_scales);
    const zero_fragment = try allocator.alloc(
        u8,
        @intCast(pack_nvfp4.activationFragmentWords(shape.paddedRows(), k) * 4),
    );
    defer allocator.free(zero_fragment);
    @memset(zero_fragment, 0);
    const zero_scales = try allocator.alloc(
        u8,
        @intCast(pack_nvfp4.activationScaleWords(shape.paddedRows(), k) * 4),
    );
    defer allocator.free(zero_scales);
    @memset(zero_scales, 0);

    // The activations are the checkpoint's token embeddings, rounded to bf16
    // because that is the type the kernels take.
    for (0..shape.m * k) |index| {
        const rounded = quant.f32_to_bf16(token_rows[index]);
        a_bf16[index] = rounded;
        a_f32[index] = quant.bf16_to_f32(rounded);
        // The e4m3 operand the fp8 kernel forms in registers.
        const byte = e4m3Encode(a_f32[index]);
        a_e4m3_f32[index] = quant.f8_e4m3_to_f32(@enumFromInt(byte));
    }
    for (0..shape.n * k) |index| {
        const rounded = quant.f32_to_bf16(weight_rows[index]);
        weights_bf16[index] = rounded;
        weights_bf16_f32[index] = quant.bf16_to_f32(rounded);
        weights_e4m3[index] = e4m3Encode(weight_rows[index]);
        weights_e4m3_f32[index] = quant.f8_e4m3_to_f32(@enumFromInt(weights_e4m3[index]));
    }
    for (0..shape.n) |row| {
        const block = k / quant.nvfp4_block_size;
        quantizeWeightRow(
            weight_rows[row * k ..][0..k],
            scale2,
            weights_packed[row * k / 2 ..][0 .. k / 2],
            weights_scales[row * block ..][0..block],
        );
        quant.dequant_nvfp4_row(
            weights_packed[row * k / 2 ..][0 .. k / 2],
            @ptrCast(weights_scales[row * block ..][0..block]),
            scale2,
            weights_dequantised_f32[row * k ..][0..k],
        );
    }
    pack_nvfp4.packWeights(
        weights_packed,
        weights_scales,
        shape.n,
        k,
        weights_fragment,
        weights_fragment_scales,
    );

    const unit: [4]u8 = @bitCast(@as(f32, 1.0));
    var golden: Golden = .{
        .shape = shape,
        .a = try allocateAndUpload(&harness.stream, std.mem.sliceAsBytes(a_bf16)),
        .a_fragment = try allocateAndUpload(&harness.stream, zero_fragment),
        .a_scales = try allocateAndUpload(&harness.stream, zero_scales),
        .weights_bf16 = try allocateAndUpload(&harness.stream, std.mem.sliceAsBytes(weights_bf16)),
        .weights_e4m3 = try allocateAndUpload(&harness.stream, weights_e4m3),
        .weights_packed = try allocateAndUpload(&harness.stream, weights_packed),
        .weights_scales = try allocateAndUpload(&harness.stream, weights_scales),
        .weights_fragment = try allocateAndUpload(&harness.stream, std.mem.sliceAsBytes(weights_fragment)),
        .weights_fragment_scales = try allocateAndUpload(&harness.stream, std.mem.sliceAsBytes(weights_fragment_scales)),
        .weights_dequantised = try allocateAndUpload(&harness.stream, std.mem.sliceAsBytes(weights_bf16)),
        .unit_scale = try allocateAndUpload(&harness.stream, unit[0..]),
        .output = try Staged.init(allocator, harness, @as(u64, shape.m) * shape.n * 4),
        .a_f32 = a_f32,
        .a_e4m3_f32 = a_e4m3_f32,
        .weights_bf16_f32 = weights_bf16_f32,
        .weights_e4m3_f32 = weights_e4m3_f32,
        .weights_dequantised_f32 = weights_dequantised_f32,
        .scale2 = scale2,
    };
    try gemm.quantizeActivations(
        &harness.kernels,
        &harness.stream,
        &golden.a,
        &golden.a_fragment,
        &golden.a_scales,
        shape,
    );
    try harness.stream.synchronize();
    assert(golden.shape.n == shape.n);
    return golden;
}

fn deinitGolden(golden: *Golden, allocator: std.mem.Allocator) void {
    assert(golden.shape.m > 0);
    assert(golden.a_f32.len == golden.shape.m * golden.shape.k);

    golden.output.deinit(allocator);
    golden.unit_scale.free();
    golden.weights_dequantised.free();
    golden.weights_fragment_scales.free();
    golden.weights_fragment.free();
    golden.weights_scales.free();
    golden.weights_packed.free();
    golden.weights_e4m3.free();
    golden.weights_bf16.free();
    golden.a_scales.free();
    golden.a_fragment.free();
    golden.a.free();
    allocator.free(golden.a_e4m3_f32);
    allocator.free(golden.weights_dequantised_f32);
    allocator.free(golden.weights_e4m3_f32);
    allocator.free(golden.weights_bf16_f32);
    allocator.free(golden.a_f32);
}

/// How far an output may sit from the sum of its products' magnitudes, i.e.
/// from the size of the numbers the accumulator actually held. This is the
/// scale an MMA's internal accumulator is judged against: rounding inside the
/// instruction is relative to the partial sums, not to the result, and a real
/// weight matrix makes the partial sums much larger than the result.
const accumulator_tolerance: f64 = 2e-3;

/// Compares the whole output matrix against `ref/ops.zig`'s matmul, on both the
/// result scale and the accumulator scale.
fn expectGoldenMatch(
    allocator: std.mem.Allocator,
    harness: *Harness,
    golden: *Golden,
    activations: []const f32,
    weights: []const f32,
    label: []const u8,
    tolerance: f64,
) !void {
    assert(activations.len == golden.shape.m * golden.shape.k);
    assert(weights.len == golden.shape.n * golden.shape.k);

    const expected = try allocator.alloc(f32, golden.shape.m * golden.shape.n);
    defer allocator.free(expected);
    ref_ops.matmul(activations, weights, expected, .{
        .rows = golden.shape.m,
        .in_features = golden.shape.k,
        .out_features = golden.shape.n,
    });

    const output = try gemm.download(
        &harness.stream,
        std.mem.sliceAsBytes(golden.output.words),
        &golden.output.device,
    );
    const actual = golden.output.floats();
    _ = output;
    var largest: f64 = 0;
    var worst: f64 = 0;
    for (expected, actual) |reference, value| {
        largest = @max(largest, @abs(@as(f64, reference)));
        worst = @max(worst, @abs(@as(f64, reference) - @as(f64, value)));
    }
    try std.testing.expect(largest > 0);
    const relative = worst / largest;
    const held = try expectAccumulatorScale(golden, activations, weights, worst);
    std.debug.print(
        "  golden {s:<18} err/|ref|={e:.3} err/|sum||a*w|={e:.3} " ++
            "(max_abs={e:.3} max|ref|={e:.3}) against ref/ops.zig matmul " ++
            "({d}x{d}x{d}, full tensor)\n",
        .{
            label,
            relative,
            held,
            worst,
            largest,
            golden.shape.m,
            golden.shape.n,
            golden.shape.k,
        },
    );
    if (relative > tolerance) {
        std.debug.print(
            "  FAIL golden {s}: err/|ref| {e:.3} > {e:.3}\n",
            .{ label, relative, tolerance },
        );
        return error.GoldenToleranceExceeded;
    }
    if (held > accumulator_tolerance) {
        std.debug.print(
            "  FAIL golden {s}: err/|sum||a*w| {e:.3} > {e:.3}\n",
            .{ label, held, accumulator_tolerance },
        );
        return error.GoldenAccumulatorExceeded;
    }
    assert(relative >= 0);
}

/// The error relative to the largest sum of product magnitudes in the matrix.
fn expectAccumulatorScale(
    golden: *const Golden,
    activations: []const f32,
    weights: []const f32,
    worst: f64,
) !f64 {
    assert(activations.len == golden.shape.m * golden.shape.k);
    assert(weights.len == golden.shape.n * golden.shape.k);
    assert(worst >= 0);

    var largest_sum: f64 = 0;
    for (0..golden.shape.m) |row| {
        const a_row = activations[row * golden.shape.k ..][0..golden.shape.k];
        for (0..golden.shape.n) |column| {
            const w_row = weights[column * golden.shape.k ..][0..golden.shape.k];
            var sum: f64 = 0;
            for (a_row, w_row) |a, w| {
                sum += @abs(@as(f64, a) * @as(f64, w));
            }
            largest_sum = @max(largest_sum, sum);
        }
    }
    try std.testing.expect(largest_sum > 0);
    assert(largest_sum > 0);
    return worst / largest_sum;
}

test "the GEMM family matches ref/ops.zig on the tiny checkpoint's own weights" {
    const allocator = std.testing.allocator;
    var harness = try Harness.open();
    defer harness.close();

    // Real committed checkpoint data, reached through the absolute fixture path
    // the build passes in, so no test depends on the working directory.
    var tensors = try ref_fixtures.open_checkpoint(allocator);
    defer tensors.close(allocator);
    const tokens = try tensors.tensor("model.language_model.embed_tokens.weight");
    const weights = try tensors.tensor("lm_head.weight");
    try std.testing.expectEqual(@as(u32, 2), tokens.rank);
    try std.testing.expectEqual(@as(u32, 2), weights.rank);

    // `lm_head.weight` is [N, K]; the first M embedding rows are the activations.
    const shape = gemm.Shape{
        .m = tokens.shape[0] / 2,
        .n = weights.shape[0],
        .k = weights.shape[1],
    };
    try shape.validate();
    var golden = try buildGolden(allocator, &harness, tokens.values, weights.values, shape);
    defer deinitGolden(&golden, allocator);

    const activations = golden.a_f32;
    try gemm.launchBf16(
        &harness.kernels,
        &harness.stream,
        &golden.a,
        &golden.weights_bf16,
        1.0,
        &golden.output.device,
        shape,
    );
    try harness.stream.synchronize();
    try expectGoldenMatch(
        allocator,
        &harness,
        &golden,
        activations,
        golden.weights_bf16_f32,
        "gemm_bf16",
        bf16_tolerance,
    );

    try gemm.launchFp8E4m3(
        &harness.kernels,
        &harness.stream,
        &golden.a,
        &golden.weights_e4m3,
        &golden.unit_scale,
        &golden.output.device,
        shape,
    );
    try harness.stream.synchronize();
    try expectGoldenMatch(
        allocator,
        &harness,
        &golden,
        golden.a_e4m3_f32,
        golden.weights_e4m3_f32,
        "gemm_fp8_e4m3",
        fp8_tolerance,
    );

    try gemm.launchNvfp4DequantBf16(
        &harness.kernels,
        &harness.stream,
        &golden.a,
        &golden.weights_packed,
        &golden.weights_scales,
        &golden.weights_dequantised,
        golden.scale2,
        &golden.output.device,
        shape,
    );
    try harness.stream.synchronize();
    try expectGoldenMatch(
        allocator,
        &harness,
        &golden,
        activations,
        golden.weights_dequantised_f32,
        "gemm_nvfp4_bf16",
        nvfp4_dequant_tolerance,
    );

    // The fast path multiplies the quantised activation, so the oracle has to
    // see that operand: this is the tier-1 comparison, not the W4A4 trade.
    const quantised_a = try allocator.dupe(f32, activations);
    defer allocator.free(quantised_a);
    const scratch_packed = try allocator.alloc(u8, shape.k / 2);
    defer allocator.free(scratch_packed);
    const scratch_scales = try allocator.alloc(u8, shape.k / quant.nvfp4_block_size);
    defer allocator.free(scratch_scales);
    for (0..shape.m) |row| {
        quantizeActivationRow(
            quantised_a[row * shape.k ..][0..shape.k],
            scratch_packed,
            scratch_scales,
        );
    }
    try gemm.launchNvfp4Mma(
        &harness.kernels,
        &harness.stream,
        &golden.a_fragment,
        &golden.a_scales,
        &golden.weights_fragment,
        &golden.weights_fragment_scales,
        golden.scale2,
        &golden.output.device,
        shape,
    );
    try harness.stream.synchronize();
    try expectGoldenMatch(
        allocator,
        &harness,
        &golden,
        quantised_a,
        golden.weights_dequantised_f32,
        "gemm_nvfp4_mma",
        kernel_machinery_tolerance,
    );
}

test "the GEMM family holds at production magnitude with realistic values" {
    const allocator = std.testing.allocator;
    var harness = try Harness.open();
    defer harness.close();
    const peaks = try gemm_peak.measure(
        allocator,
        &harness.kernels,
        &harness.stream,
    );

    // An expert gate/up projection of the checkpoint - 10240 outputs over 2560
    // inputs - with a full prefill batch of 2048 rows, and values drawn the way
    // a trained tensor is (Gaussian, per-row magnitude, a heavy tail) rather
    // than from the bounded uniform fixture. A shape this size with values that
    // never stress the dynamic range would be as weak a detector of an indexing
    // bug as a tiny fixture is.
    const shape = gemm.Shape{ .m = 2048, .n = 10240, .k = 2560 };
    std.debug.print(
        "-- production magnitude, realistic values: M={d} N={d} K={d}\n",
        .{ shape.m, shape.n, shape.k },
    );
    var weights = try Weights.init(allocator, &harness, shape.n, shape.k, .realistic);
    defer weights.deinit(allocator);
    try runShape(allocator, &harness, &weights, &peaks, shape);
}
