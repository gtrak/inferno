//! Repacking the checkpoint's NVFP4 weights into tensor-core fragment order.
//!
//! The checkpoint stores a matrix as `[N, K/2]` packed nibbles (low nibble first,
//! i.e. ModelOpt's order along the contracted dimension) plus one E4M3 scale per
//! 16 weights as `[N, K/16]`. The block-scaled MMA on sm_120a wants something
//! else entirely: every lane holds two 32-bit words of 16 nibbles and one 32-bit
//! word of four scales, and the mapping from (lane, word) to (n, k) is fixed by
//! `SFBLayout` in `.agents/design/mma-sm120-nvfp4.md`.
//!
//! Repacking here, once at load, is what lets the hot loop in `gemm.cu` be pure
//! `ld.shared` + `OMMA.SF`: no shuffles and no nibble surgery per k step. The
//! mapping below is the single definition of that order. `unpackWeights` is its
//! inverse, used by the tests to prove the repack is a permutation and that the
//! dequantised values are unchanged bit for bit; the GPU is checked separately
//! against the f32 oracle in `core/src/kernels/gemm.zig`.

const std = @import("std");

const quant = @import("quant.zig");

const assert = std.debug.assert;

/// Contraction depth of one block-scaled FP4 MMA.
pub const k_tile: u32 = 64;

/// Columns one block-scaled FP4 MMA covers.
pub const n_tile: u32 = 8;

/// Lanes in a warp, i.e. the MMA's thread count.
pub const warp_size: u32 = 32;

/// Words per lane of the B operand: 16 nibbles.
pub const b_words_per_lane: u32 = 2;

/// Words per lane of the A operand: 32 nibbles.
pub const a_words_per_lane: u32 = 4;

/// Weights sharing one E4M3 scale.
pub const block_size: u32 = quant.nvfp4_block_size;

/// Scale words one B tile needs: one per column of the tile.
pub const b_scale_words_per_tile: u32 = n_tile;

/// Scale words one A tile needs: one per lane.
pub const a_scale_words_per_tile: u32 = warp_size;

/// Bytes one E4M3 scale word carries.
const scale_bytes_per_word: u32 = 4;

/// Size of the repacked weight tensor in 32-bit words, for `[N, K]`.
pub fn weightFragmentWords(n: u32, k: u32) u64 {
    assert(k > 0 and k % k_tile == 0);
    assert(n > 0 and n % n_tile == 0);

    const tiles: u64 = @as(u64, k / k_tile) * @as(u64, n / n_tile);
    assert(tiles > 0);
    return tiles * warp_size * b_words_per_lane;
}

/// Size of the repacked weight scales in 32-bit words, for `[N, K]`.
pub fn weightScaleWords(n: u32, k: u32) u64 {
    assert(k > 0 and k % k_tile == 0);
    assert(block_size == 16);

    const tiles: u64 = @as(u64, k / k_tile) * @as(u64, n / n_tile);
    assert(tiles > 0);
    return tiles * b_scale_words_per_tile;
}

/// Size of the quantised activation tensor in 32-bit words, for `[M, K]` with
/// `M` already padded to a whole number of 16-row tiles.
pub fn activationFragmentWords(m: u32, k: u32) u64 {
    assert(m > 0 and m % 16 == 0);
    assert(k > 0 and k % k_tile == 0);

    const tiles: u64 = @as(u64, m / 16) * @as(u64, k / k_tile);
    assert(tiles > 0);
    return tiles * warp_size * a_words_per_lane;
}

/// Size of the activation scales in 32-bit words, for `[M, K]` with padded `M`.
pub fn activationScaleWords(m: u32, k: u32) u64 {
    assert(m > 0 and m % 16 == 0);
    assert(k > 0 and k % k_tile == 0);

    const tiles: u64 = @as(u64, m / 16) * @as(u64, k / k_tile);
    assert(tiles > 0);
    return tiles * a_scale_words_per_tile;
}

/// Reads four consecutive bytes as one word: byte b carries the numbers of
/// k, k+1 packed low nibble first, which is the order the MMA reads.
fn readWord(bytes: []const u8, offset: usize) u32 {
    assert(bytes.len >= offset + 4);
    assert(offset + 4 <= bytes.len);

    return @as(u32, bytes[offset]) |
        (@as(u32, bytes[offset + 1]) << 8) |
        (@as(u32, bytes[offset + 2]) << 16) |
        (@as(u32, bytes[offset + 3]) << 24);
}

/// Writes one word back as four consecutive bytes.
fn writeWord(bytes: []u8, offset: usize, word: u32) void {
    assert(bytes.len >= offset + 4);
    assert(offset + 4 <= bytes.len);

    bytes[offset] = @truncate(word);
    bytes[offset + 1] = @truncate(word >> 8);
    bytes[offset + 2] = @truncate(word >> 16);
    bytes[offset + 3] = @truncate(word >> 24);
}

/// Repacks row-major NVFP4 weights into fragment order.
///
/// `packed` is `[N, K/2]`, `scales` is `[N, K/16]`; `fragment` and
/// `fragment_scales` must be exactly the sizes `weightFragmentWords` and
/// `weightScaleWords` report.
pub fn packWeights(
    packed_weights: []const u8,
    scales: []const u8,
    n: u32,
    k: u32,
    fragment: []u32,
    fragment_scales: []u32,
) void {
    assert(packed_weights.len == @as(usize, n) * k / 2);
    assert(scales.len == @as(usize, n) * k / block_size);
    assert(fragment.len == weightFragmentWords(n, k));
    assert(fragment_scales.len == weightScaleWords(n, k));

    const n_tiles = n / n_tile;
    const k_tiles = k / k_tile;
    for (0..k_tiles) |k_tile_index| {
        for (0..n_tiles) |n_tile_index| {
            const tile: usize = @intCast(k_tile_index * n_tiles + n_tile_index);
            packWeightTile(packed_weights, scales, k, n, tile, n_tile_index, k_tile_index, fragment, fragment_scales);
        }
    }
    assert(fragment.len == weightFragmentWords(n, k));
}

/// Repacks the 8 columns and 64 contraction steps of one B tile.
fn packWeightTile(
    packed_weights: []const u8,
    scales: []const u8,
    k: u32,
    n: u32,
    tile: usize,
    n_tile_index: usize,
    k_tile_index: usize,
    fragment: []u32,
    fragment_scales: []u32,
) void {
    assert(tile < fragment.len / (warp_size * b_words_per_lane));
    assert(n_tile_index < n / n_tile);

    const word_base = tile * warp_size * b_words_per_lane;
    var lane: usize = 0;
    while (lane < warp_size) : (lane += 1) {
        const column = n_tile_index * n_tile + lane / 4;
        const quad = lane % 4;
        var word: usize = 0;
        while (word < b_words_per_lane) : (word += 1) {
            // The lane's word `r` covers k = k_tile * 64 + 8 * quad + 32 * r.
            const k_start = k_tile_index * k_tile + 8 * quad + 32 * word;
            const source = column * (k / 2) + k_start / 2;
            fragment[word_base + lane * b_words_per_lane + word] =
                readWord(packed_weights, source);
        }
    }

    const scale_base = tile * b_scale_words_per_tile;
    var column_local: usize = 0;
    while (column_local < b_scale_words_per_tile) : (column_local += 1) {
        const column = n_tile_index * n_tile + column_local;
        const source = column * (k / block_size) +
            k_tile_index * (k_tile / block_size);
        fragment_scales[scale_base + column_local] = readWord(scales, source);
    }
    assert(scale_base + b_scale_words_per_tile <= fragment_scales.len);
}

/// Inverse of `packWeights`: rebuilds the checkpoint layout from fragments.
pub fn unpackWeights(
    fragment: []const u32,
    fragment_scales: []const u32,
    n: u32,
    k: u32,
    packed_weights: []u8,
    scales: []u8,
) void {
    assert(fragment.len == weightFragmentWords(n, k));
    assert(fragment_scales.len == weightScaleWords(n, k));
    assert(packed_weights.len == @as(usize, n) * k / 2);
    assert(scales.len == @as(usize, n) * k / block_size);

    const n_tiles = n / n_tile;
    const k_tiles = k / k_tile;
    for (0..k_tiles) |k_tile_index| {
        for (0..n_tiles) |n_tile_index| {
            const tile: usize = @intCast(k_tile_index * n_tiles + n_tile_index);
            unpackWeightTile(fragment, fragment_scales, k, n, tile, n_tile_index, k_tile_index, packed_weights, scales);
        }
    }
    assert(packed_weights.len == @as(usize, n) * k / 2);
}

/// Inverse of `packWeightTile`.
fn unpackWeightTile(
    fragment: []const u32,
    fragment_scales: []const u32,
    k: u32,
    n: u32,
    tile: usize,
    n_tile_index: usize,
    k_tile_index: usize,
    packed_weights: []u8,
    scales: []u8,
) void {
    assert(tile * warp_size * b_words_per_lane < fragment.len);
    assert(n_tile_index < n / n_tile);

    const word_base = tile * warp_size * b_words_per_lane;
    var lane: usize = 0;
    while (lane < warp_size) : (lane += 1) {
        const column = n_tile_index * n_tile + lane / 4;
        const quad = lane % 4;
        var word: usize = 0;
        while (word < b_words_per_lane) : (word += 1) {
            const k_start = k_tile_index * k_tile + 8 * quad + 32 * word;
            const destination = column * (k / 2) + k_start / 2;
            writeWord(packed_weights, destination, fragment[word_base + lane * b_words_per_lane + word]);
        }
    }

    const scale_base = tile * b_scale_words_per_tile;
    var column_local: usize = 0;
    while (column_local < b_scale_words_per_tile) : (column_local += 1) {
        const column = n_tile_index * n_tile + column_local;
        const destination = column * (k / block_size) +
            k_tile_index * (k_tile / block_size);
        writeWord(scales, destination, fragment_scales[scale_base + column_local]);
    }
    assert(scale_base + b_scale_words_per_tile <= fragment_scales.len);
}

/// Sizes of the four buffers one `[N, K]` weight matrix needs, in bytes.
pub const WeightBuffers = struct {
    packed_bytes: u64,
    scale_bytes: u64,
    fragment_bytes: u64,
    fragment_scale_bytes: u64,
};

/// Buffer sizes for a `[N, K]` weight matrix.
pub fn weightBuffers(n: u32, k: u32) WeightBuffers {
    assert(k % k_tile == 0);
    assert(n % n_tile == 0);

    return .{
        .packed_bytes = @as(u64, n) * k / 2,
        .scale_bytes = @as(u64, n) * k / block_size,
        .fragment_bytes = weightFragmentWords(n, k) * 4,
        .fragment_scale_bytes = weightScaleWords(n, k) * 4,
    };
}

test {
    // The layout helpers are checked by the round-trip tests below; referencing
    // the decoder here keeps the two definitions of the block size linked.
    assert(scale_bytes_per_word * 4 == block_size);
    assert(a_scale_words_per_tile == warp_size);
}

/// Deterministic pseudo-random bytes: a linear congruential generator, so a
/// failure reproduces exactly.
fn fillPseudorandom(bytes: []u8, seed: u32) void {
    assert(bytes.len > 0);
    assert(seed != 0);

    var state = seed;
    for (bytes) |*byte| {
        state = state *% 1_664_525 +% 1_013_904_223;
        byte.* = @truncate(state >> 16);
    }
}

test "the repack round-trips every packed_weights weight and scale byte" {
    const n: u32 = 16;
    const k: u32 = 128;
    const buffers = weightBuffers(n, k);

    const allocator = std.testing.allocator;
    const packed_weights = try allocator.alloc(u8, @intCast(buffers.packed_bytes));
    defer allocator.free(packed_weights);
    const scales = try allocator.alloc(u8, @intCast(buffers.scale_bytes));
    defer allocator.free(scales);
    fillPseudorandom(packed_weights, 0x5eed);
    fillPseudorandom(scales, 0x1dea);

    const fragment = try allocator.alloc(u32, @intCast(weightFragmentWords(n, k)));
    defer allocator.free(fragment);
    const fragment_scales =
        try allocator.alloc(u32, @intCast(weightScaleWords(n, k)));
    defer allocator.free(fragment_scales);

    packWeights(packed_weights, scales, n, k, fragment, fragment_scales);

    const rebuilt = try allocator.alloc(u8, packed_weights.len);
    defer allocator.free(rebuilt);
    const rebuilt_scales = try allocator.alloc(u8, scales.len);
    defer allocator.free(rebuilt_scales);
    unpackWeights(fragment, fragment_scales, n, k, rebuilt, rebuilt_scales);

    try std.testing.expectEqualSlices(u8, packed_weights, rebuilt);
    try std.testing.expectEqualSlices(u8, scales, rebuilt_scales);
}

test "the repacked weights dequantise to the same f32 values" {
    const n: u32 = 8;
    const k: u32 = 64;
    const allocator = std.testing.allocator;

    const packed_weights = try allocator.alloc(u8, @as(usize, n) * k / 2);
    defer allocator.free(packed_weights);
    const scales = try allocator.alloc(u8, @as(usize, n) * k / block_size);
    defer allocator.free(scales);
    fillPseudorandom(packed_weights, 0xc0ffee);
    fillPseudorandom(scales, 0xbadf00d);

    const fragment = try allocator.alloc(u32, @intCast(weightFragmentWords(n, k)));
    defer allocator.free(fragment);
    const fragment_scales =
        try allocator.alloc(u32, @intCast(weightScaleWords(n, k)));
    defer allocator.free(fragment_scales);
    packWeights(packed_weights, scales, n, k, fragment, fragment_scales);

    const rebuilt = try allocator.alloc(u8, packed_weights.len);
    defer allocator.free(rebuilt);
    const rebuilt_scales = try allocator.alloc(u8, scales.len);
    defer allocator.free(rebuilt_scales);
    unpackWeights(fragment, fragment_scales, n, k, rebuilt, rebuilt_scales);

    // The oracle's decoder sees the same bytes, so it must see the same values.
    const scale2: f32 = 0.001953125;
    var original_row: [k]f32 = undefined;
    var rebuilt_row: [k]f32 = undefined;
    for (0..n) |row| {
        const packed_row = packed_weights[row * k / 2 ..][0 .. k / 2];
        const scale_row = scales[row * (k / block_size) ..][0 .. k / block_size];
        quant.dequant_nvfp4_row(
            packed_row,
            @ptrCast(scale_row),
            scale2,
            original_row[0..],
        );
        quant.dequant_nvfp4_row(
            rebuilt[row * k / 2 ..][0 .. k / 2],
            @ptrCast(rebuilt_scales[row * (k / block_size) ..][0 .. k / block_size]),
            scale2,
            rebuilt_row[0..],
        );
        for (original_row, rebuilt_row) |expected, actual| {
            try std.testing.expectEqual(expected, actual);
        }
    }
}

test "fragment and scale sizes match the documented tile arithmetic" {
    // One 64-deep k tile of eight columns: 32 lanes x 2 words, and 8 scales.
    try std.testing.expectEqual(@as(u64, 64), weightFragmentWords(8, 64));
    try std.testing.expectEqual(@as(u64, 8), weightScaleWords(8, 64));
    // 16 rows and 64 columns of activations: four lanes' worth of words per
    // row half, so 128 words and 32 scale words.
    try std.testing.expectEqual(@as(u64, 128), activationFragmentWords(16, 64));
    try std.testing.expectEqual(@as(u64, 32), activationScaleWords(16, 64));
    // A 2048x2560 weight matrix is 2.62 MB packed and 1.31 MB repacked, i.e.
    // the same bytes plus nothing.
    const buffers = weightBuffers(2048, 2560);
    try std.testing.expectEqual(buffers.packed_bytes, buffers.fragment_bytes);
    try std.testing.expectEqual(
        buffers.scale_bytes,
        buffers.fragment_scale_bytes,
    );
}
