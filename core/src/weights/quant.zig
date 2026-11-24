//! Scalar decoders for the checkpoint's block-quantised weight formats.
//!
//! Two layouts ship in the checkpoint:
//!
//! * NVFP4 (the expert stacks): 4-bit E2M1 weights packed two per byte, low
//!   nibble first, with one E4M3 scale per `nvfp4_block_size` consecutive
//!   weights along the contracted dimension and one FP32 second-level scale per
//!   tensor. `w = fp4 * (block_scale * scale_2)`.
//! * FP8 E4M3 blockwise (the MTP layer): one BF16 inverse scale per 128x128
//!   tile. `w = fp8 * scale_inv`.
//!
//! Everything here is scalar and table-driven on purpose: these decoders are
//! the reference the GPU dequantisation kernels are checked against, so they
//! must be exact, independent of those kernels, and cheap to reason about.
//! `tools/ref-trace/quant_oracle.py` generates the golden cases in
//! `tests/golden/quant/` with an independent numpy implementation.

const std = @import("std");

const assert = std.debug.assert;

const safetensors = @import("safetensors.zig");

/// An E4M3 (FN) byte as stored by ModelOpt; decode with `f8_e4m3_to_f32`.
pub const F8E4M3 = enum(u8) { _ };

/// Weights sharing one E4M3 block scale in the NVFP4 expert layout.
pub const nvfp4_block_size: u32 = 16;

/// Side of the FP8 inverse-scale tile used by the MTP experts.
pub const fp8_block_size: u32 = 128;

/// Largest magnitude E2M1 represents.
pub const fp4_max_magnitude: f32 = 6.0;

/// Largest magnitude E4M3 (FN) represents: E4M3 has no infinity.
pub const fp8_max_magnitude: f32 = 448.0;

/// Sign bit of an f32, needed because arithmetic cannot produce -0.0 from 0.0.
const f32_sign_bit: u32 = 0x8000_0000;

/// Bias of the f32 exponent field, and of the E4M3 exponent field.
const f32_exponent_bias: u32 = 127;
const e4m3_exponent_bias: u32 = 7;

/// 2^-9, the E4M3 subnormal step, as an exact f32.
const e4m3_subnormal_step: f32 = @bitCast(@as(u32, 118) << 23);

/// E2M1 by 4-bit code: bit 3 is the sign and code 8 is negative zero.
pub const fp4_values: [16]f32 = blk: {
    const magnitudes = [8]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 };
    var table: [16]f32 = undefined;
    for (0..16) |code| {
        const magnitude = magnitudes[code % 8];
        const bits: u32 = @bitCast(magnitude);
        table[code] = @bitCast(if (code < 8) bits else bits | f32_sign_bit);
    }
    break :blk table;
};

/// A dense matrix shape; block layouts need both extents.
pub const Extent = struct {
    rows: u32,
    cols: u32,

    pub fn elementCount(self: Extent) u64 {
        assert(self.rows > 0);
        assert(self.cols > 0);

        const count = @as(u64, self.rows) * @as(u64, self.cols);
        assert(count > 0);
        return count;
    }
};

/// Applies a sign bit without arithmetic, so -0.0 survives.
fn withSign(magnitude: f32, negative: bool) f32 {
    assert(magnitude >= 0.0);
    assert(@as(u32, @bitCast(magnitude)) & f32_sign_bit == 0);

    const bits: u32 = @bitCast(magnitude);
    const signed: f32 = @bitCast(if (negative) bits | f32_sign_bit else bits);
    assert(if (negative) @as(u32, @bitCast(signed)) & f32_sign_bit != 0 else true);
    return signed;
}

/// 2^(exponent - 7) as an exact f32 for the E4M3 normal range.
fn e4m3ExponentScale(exponent: u32) f32 {
    assert(exponent > 0);
    assert(exponent < 16);

    const biased = f32_exponent_bias - e4m3_exponent_bias + exponent;
    const scale: f32 = @bitCast(biased << 23);
    assert(scale >= 0.0078125);
    return scale;
}

/// Decodes one E4M3 (FN) byte. The only NaN code is `0x7f`/`0xff`.
pub fn f8_e4m3_to_f32(scale: F8E4M3) f32 {
    const bits: u8 = @intFromEnum(scale);
    assert(@sizeOf(F8E4M3) == 1);

    const negative = (bits & 0x80) != 0;
    const exponent: u32 = @as(u32, (bits >> 3) & 0x0f);
    const mantissa: u32 = @as(u32, bits & 0x07);

    var magnitude: f32 = undefined;
    if (exponent == 0) {
        // Subnormal: mantissa * 2^-9, exact in f32.
        magnitude = @as(f32, @floatFromInt(mantissa)) * e4m3_subnormal_step;
    } else if (exponent == 0x0f) {
        if (mantissa == 0x07) {
            return std.math.nan(f32);
        } else {
            assert(mantissa < 0x07);
        }
        magnitude = (1.0 + @as(f32, @floatFromInt(mantissa)) / 8.0) *
            e4m3ExponentScale(exponent);
    } else {
        assert(exponent < 0x0f);
        magnitude = (1.0 + @as(f32, @floatFromInt(mantissa)) / 8.0) *
            e4m3ExponentScale(exponent);
    }
    assert(magnitude <= fp8_max_magnitude);
    assert(!std.math.isNan(magnitude));
    return withSign(magnitude, negative);
}

/// Decodes one 4-bit E2M1 code.
pub fn fp4_e2m1_decode(nibble: u8) f32 {
    assert(nibble < 16);
    assert(fp4_values.len == 16);

    const value = fp4_values[nibble];
    assert(@abs(value) <= fp4_max_magnitude);
    return value;
}

/// Decodes a packed byte into its two weights; the low nibble comes first,
/// which is the order ModelOpt packs along the contracted dimension.
pub fn fp4x2_decode(packed_pair: u8, first_out: *f32, second_out: *f32) void {
    assert(fp4_values.len == 16);
    assert(@intFromPtr(first_out) != @intFromPtr(second_out));

    first_out.* = fp4_e2m1_decode(packed_pair & 0x0f);
    second_out.* = fp4_e2m1_decode(packed_pair >> 4);
    assert(@abs(second_out.*) <= fp4_max_magnitude);
}

/// Reinterprets a BF16 bit pattern as f32; every BF16 value is an f32 value.
pub fn bf16_to_f32(value: u16) f32 {
    assert(@sizeOf(u16) == 2);
    assert(@sizeOf(f32) == 4);

    const bits: u32 = @as(u32, value) << 16;
    const result: f32 = @bitCast(bits);
    assert(@as(u32, @bitCast(result)) == bits);
    return result;
}

/// Rounds an f32 to the nearest BF16 value, ties to even.
pub fn f32_to_bf16(value: f32) u16 {
    assert(@sizeOf(f32) == 4);
    assert(@sizeOf(u16) == 2);

    const bits: u32 = @bitCast(value);
    const magnitude = bits & 0x7fff_ffff;
    // NaN must stay NaN, so quiet it instead of rounding into an infinity.
    if (magnitude > 0x7f80_0000) {
        return @truncate((bits >> 16) | 0x0040);
    } else {
        assert(magnitude <= 0x7f80_0000);
    }
    // 0x7fff is the largest remainder that must round up, and the extra bit
    // breaks exact ties towards an even mantissa.
    const rounded = bits +% (0x7fff +% ((bits >> 16) & 1));
    const half = @as(u16, @truncate(rounded >> 16));
    assert(half != 0 or bits & 0x7fff_0000 == 0);
    return half;
}

/// Dequantises one row of a packed NVFP4 matrix.
///
/// `packed` holds `out.len / 2` bytes, `scales` holds one E4M3 scale per 16
/// consecutive weights, and `scale_2` is the per-tensor FP32 scale. The product
/// `block_scale * scale_2` is formed once per block, before the weights are
/// multiplied in: that is the order the oracle uses, so the results are
/// bit-identical.
pub fn dequant_nvfp4_row(
    packed_weights: []const u8,
    scales: []const F8E4M3,
    scale_2: f32,
    out: []f32,
) void {
    assert(packed_weights.len > 0);
    assert(out.len == packed_weights.len * 2);
    assert(scales.len * nvfp4_block_size == out.len);

    const packed_per_block = nvfp4_block_size / 2;
    for (scales, 0..) |scale, block| {
        const combined = f8_e4m3_to_f32(scale) * scale_2;
        const first = block * nvfp4_block_size;
        for (0..packed_per_block) |offset| {
            const packed_index = block * packed_per_block + offset;
            var lower: f32 = undefined;
            var upper: f32 = undefined;
            fp4x2_decode(packed_weights[packed_index], &lower, &upper);
            out[first + offset * 2] = lower * combined;
            out[first + offset * 2 + 1] = upper * combined;
        }
    }
    assert(out.len % nvfp4_block_size == 0);
}

/// Number of 128-wide tiles needed to cover `size` columns or rows.
fn tileCount(size: u32) u32 {
    assert(size > 0);
    assert(fp8_block_size == 128);

    const wide = (@as(u64, size) + fp8_block_size - 1) / fp8_block_size;
    const tiles: u32 = @intCast(wide);
    assert(@as(u64, tiles) * fp8_block_size >= size);
    return tiles;
}

/// Dequantises an FP8 E4M3 matrix that carries one BF16 inverse scale per
/// 128x128 tile, which is the layout of the MTP experts.
pub fn dequant_fp8_blockwise(
    weight: []const F8E4M3,
    scale_inv: []const u16,
    extent: Extent,
    out: []f32,
) void {
    assert(extent.rows > 0);
    assert(extent.cols > 0);
    assert(weight.len == extent.elementCount());
    assert(out.len == weight.len);

    const scale_cols = tileCount(extent.cols);
    const scale_rows = tileCount(extent.rows);
    assert(scale_inv.len == scale_rows * scale_cols);

    for (0..extent.rows) |row| {
        const scale_row = row / fp8_block_size;
        for (0..extent.cols) |col| {
            const scale_col = col / fp8_block_size;
            const scale = bf16_to_f32(scale_inv[scale_row * scale_cols + scale_col]);
            const index = row * extent.cols + col;
            out[index] = f8_e4m3_to_f32(weight[index]) * scale;
        }
    }
    assert(out.len == extent.elementCount());
}

/// Golden fixtures embedded by `core/build.zig`, so the tests never depend on
/// the working directory or on the checkpoint being downloaded.
pub const golden_nvfp4 = @embedFile("golden_quant_nvfp4");
pub const golden_fp8_blockwise = @embedFile("golden_quant_fp8_blockwise");

/// Bytes of the tensor called `name`, which must be in the fixture.
fn namedBytes(
    header: *const safetensors.Header,
    file: *const safetensors.File,
    name: []const u8,
) ![]const u8 {
    assert(name.len > 0);
    assert(header.tensors.len > 0);

    for (header.tensors) |*tensor| {
        if (std.mem.eql(u8, tensor.name, name)) {
            return file.tensorBytes(tensor);
        } else {
            continue;
        }
    }
    std.debug.print("quant: fixture has no tensor {s}\n", .{name});
    return error.missing_tensor;
}

/// Little endian f32 decoding of a fixture tensor.
fn readF32(allocator: std.mem.Allocator, bytes: []const u8) ![]f32 {
    assert(bytes.len > 0);
    assert(bytes.len % 4 == 0);

    const values = try allocator.alloc(f32, bytes.len / 4);
    for (values, 0..) |*value, index| {
        const bits = std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little);
        value.* = @bitCast(bits);
    }
    assert(values.len * 4 == bytes.len);
    return values;
}

/// Little endian BF16 decoding of a fixture tensor, as raw bit patterns.
fn readBf16(allocator: std.mem.Allocator, bytes: []const u8) ![]u16 {
    assert(bytes.len > 0);
    assert(bytes.len % 2 == 0);

    const values = try allocator.alloc(u16, bytes.len / 2);
    for (values, 0..) |*value, index| {
        value.* = std.mem.readInt(u16, bytes[index * 2 ..][0..2], .little);
    }
    assert(values.len * 2 == bytes.len);
    return values;
}

/// Byte for byte copy of a fixture tensor into `F8E4M3` values.
fn readF8(allocator: std.mem.Allocator, bytes: []const u8) ![]F8E4M3 {
    assert(bytes.len > 0);
    assert(@sizeOf(F8E4M3) == 1);

    const values = try allocator.alloc(F8E4M3, bytes.len);
    for (values, bytes) |*value, byte| {
        value.* = @enumFromInt(byte);
    }
    assert(values.len == bytes.len);
    return values;
}

test "the E2M1 table is the sixteen value grid" {
    const expected = [16]f32{
        0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    };
    for (expected, 0..) |value, code| {
        const decoded = fp4_e2m1_decode(@intCast(code));
        try std.testing.expectEqual(@as(u32, @bitCast(value)), @as(u32, @bitCast(decoded)));
    }
    // Code 8 is negative zero, which must survive the round trip.
    try std.testing.expect(std.math.signbit(fp4_e2m1_decode(8)));
    try std.testing.expect(!std.math.signbit(fp4_e2m1_decode(0)));
}

test "a packed E2M1 byte splits low nibble first" {
    var lower: f32 = undefined;
    var upper: f32 = undefined;

    fp4x2_decode(0xa3, &lower, &upper);
    try std.testing.expectEqual(@as(f32, 1.5), lower);
    try std.testing.expectEqual(@as(f32, -1.0), upper);

    fp4x2_decode(0xd3, &lower, &upper);
    try std.testing.expectEqual(@as(f32, 1.5), lower);
    try std.testing.expectEqual(@as(f32, -3.0), upper);

    // Low nibble 0 is +0.0, high nibble 8 is -0.0: both are zero, and the sign
    // bit is the only way to tell them apart.
    fp4x2_decode(0x80, &lower, &upper);
    try std.testing.expect(!std.math.signbit(lower));
    try std.testing.expect(std.math.signbit(upper));
}

test "E4M3 decodes its boundaries" {
    try std.testing.expectEqual(@as(f32, 0.0), f8_e4m3_to_f32(@enumFromInt(0x00)));
    try std.testing.expect(std.math.signbit(f8_e4m3_to_f32(@enumFromInt(0x80))));
    try std.testing.expectEqual(fp8_max_magnitude, f8_e4m3_to_f32(@enumFromInt(0x7e)));
    try std.testing.expectEqual(-fp8_max_magnitude, f8_e4m3_to_f32(@enumFromInt(0xfe)));
    try std.testing.expectEqual(@as(f32, 1.0), f8_e4m3_to_f32(@enumFromInt(0x38)));
    try std.testing.expectEqual(@as(f32, 1.875), f8_e4m3_to_f32(@enumFromInt(0x3f)));
    // The smallest subnormal and the smallest normal.
    try std.testing.expectEqual(@as(f32, 0.001953125), f8_e4m3_to_f32(@enumFromInt(0x01)));
    try std.testing.expectEqual(@as(f32, 0.015625), f8_e4m3_to_f32(@enumFromInt(0x08)));
    try std.testing.expect(std.math.isNan(f8_e4m3_to_f32(@enumFromInt(0x7f))));
    try std.testing.expect(std.math.isNan(f8_e4m3_to_f32(@enumFromInt(0xff))));
}

test "BF16 conversion is exact and rounds to nearest even" {
    const one = f32_to_bf16(1.0);
    try std.testing.expectEqual(@as(u16, 0x3f80), one);
    try std.testing.expectEqual(@as(f32, 1.0), bf16_to_f32(one));

    // Exactly halfway between 1.0 and 1 + 2^-7: ties to the even mantissa.
    const halfway = 1.0 + std.math.pow(f32, 2.0, -8.0);
    try std.testing.expectEqual(@as(u16, 0x3f80), f32_to_bf16(halfway));
    // One step above halfway must round up.
    const above = halfway + std.math.pow(f32, 2.0, -20.0);
    try std.testing.expectEqual(@as(u16, 0x3f81), f32_to_bf16(above));

    // Negative values keep their sign and NaN stays quiet instead of infiniting.
    try std.testing.expectEqual(@as(u16, 0xbf80), f32_to_bf16(-1.0));
    const quiet = f32_to_bf16(std.math.nan(f32)) & 0x7fc0;
    try std.testing.expectEqual(@as(u16, 0x7fc0), quiet);
    try std.testing.expectEqual(@as(u16, 0x7f80), f32_to_bf16(std.math.inf(f32)));
}

test "the NVFP4 golden case dequantises bit exactly" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var file = try safetensors.File.initFromBytes("nvfp4_case", golden_nvfp4);
    defer file.close();
    const header = try safetensors.parseHeader(allocator, &file, 0);
    try std.testing.expect(header.dataComplete(&file));

    const packed_weights = try namedBytes(&header, &file, "packed");
    const scales = try readF8(allocator, try namedBytes(&header, &file, "scales"));
    const scale_2 = (try readF32(allocator, try namedBytes(&header, &file, "scale_2")))[0];
    const expected = try readF32(allocator, try namedBytes(&header, &file, "reference"));
    try std.testing.expectEqual(scales.len, @as(usize, 2));

    const out = try allocator.alloc(f32, expected.len);
    dequant_nvfp4_row(packed_weights, scales, scale_2, out);

    for (expected, out) |want, got| {
        try std.testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(got)));
    }
    // The pattern includes a weight that quantises to negative zero, which is
    // the case a widening tolerance would silently hide.
    try std.testing.expect(std.math.signbit(out[15]));
}

test "the FP8 blockwise golden case dequantises bit exactly" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var file = try safetensors.File.initFromBytes("fp8_blockwise_case", golden_fp8_blockwise);
    defer file.close();
    const header = try safetensors.parseHeader(allocator, &file, 0);
    try std.testing.expect(header.dataComplete(&file));

    const weight = try readF8(allocator, try namedBytes(&header, &file, "weight"));
    const scale_inv = try readBf16(allocator, try namedBytes(&header, &file, "scale_inv"));
    const expected = try readF32(allocator, try namedBytes(&header, &file, "reference"));
    try std.testing.expectEqual(scale_inv.len, @as(usize, 4));

    const extent = Extent{ .rows = 256, .cols = 256 };
    const out = try allocator.alloc(f32, expected.len);
    dequant_fp8_blockwise(weight, scale_inv, extent, out);

    for (expected, out) |want, got| {
        try std.testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(got)));
    }
}
