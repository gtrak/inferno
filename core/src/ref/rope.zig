//! Partial rotary embeddings with mRoPE frequency recomposition.
//!
//! The reference (`Qwen4ExpTextRotaryEmbedding`, lines 119-158) builds cos/sin
//! for three position grids (temporal, height, width), overwrites the height
// and
//! width lanes of the temporal grid, then concatenates the half-width frequency
//! vector with itself so it broadcasts over a full rotary block. Only `head_dim
//! * partial_rotary_factor` dims of each head rotate; the rest pass through.

const std = @import("std");

const assert = std.debug.assert;

/// Frequencies per position: `1 / theta^(2i/dim)` for `i` in `0..dim/2`.
///
/// `dim` is the number of rotated dims, i.e. `2 * frequencies.len`.
pub fn inv_freq_init(rope_theta: f32, frequencies: []f32) void {
    assert(rope_theta > 1.0);
    assert(frequencies.len > 0);

    const dim: u32 = @intCast(2 * frequencies.len);
    const dim_float: f32 = @floatFromInt(dim);
    for (frequencies, 0..) |*frequency, i| {
        const exponent = @as(f32, @floatFromInt(2 * i)) / dim_float;
        frequency.* = 1.0 / std.math.pow(f32, rope_theta, exponent);
    }
    assert(frequencies[0] == 1.0);
}

/// Which position grid drives frequency `index` after recomposition.
///
/// The reference copies `freq[1, ..., d::3]` into the temporal grid for as many
/// lanes as `mrope_section[1] * 3` covers, then does the same for the width
/// grid; every other lane keeps the temporal entry. The width grid is applied
/// last, so it wins any overlap.
pub fn mrope_grid(index: u32, mrope_section: [3]u32) u32 {
    assert(mrope_section[0] > 0);
    assert(index < 1 << 20);

    var grid: u32 = 0;
    if (is_section_lane(index, 1, mrope_section[1])) {
        grid = 1;
    } else {
        assert(grid == 0);
    }
    if (is_section_lane(index, 2, mrope_section[2])) {
        grid = 2;
    } else {
        assert(grid <= 1);
    }
    assert(grid <= 2);
    return grid;
}

/// True when `index` is lane `offset` of a strided grid of `width` triples.
fn is_section_lane(index: u32, offset: u32, width: u32) bool {
    assert(offset < 3);
    assert(width <= 1 << 20);

    if (index < offset) {
        return false;
    } else {
        if ((index - offset) % 3 != 0) {
            return false;
        } else {
            return index < width * 3;
        }
    }
}

/// Builds cos/sin tables of shape `[positions, 2 * frequencies.len]`.
///
/// `positions` holds the three grids (temporal, height, width); the text-only
/// path passes the same positions three times, which makes the recomposition a
/// no-op, but the interleave is still exercised.
pub fn cos_sin(
    positions: [3][]const u32,
    frequencies: []const f32,
    mrope_section: [3]u32,
    cos: []f32,
    sin: []f32,
) void {
    assert(frequencies.len > 0);
    assert(cos.len == sin.len);
    assert(positions[0].len > 0);

    const count: usize = positions[0].len;
    const rotated: usize = 2 * frequencies.len;
    assert(cos.len == count * rotated);
    assert(positions[1].len == count);
    assert(positions[2].len == count);

    for (0..count) |t| {
        for (frequencies, 0..) |frequency, index| {
            const grid = mrope_grid(@intCast(index), mrope_section);
            const position: f32 = @floatFromInt(positions[grid][t]);
            const angle = position * frequency;
            const cosine = std.math.cos(angle);
            const sine = std.math.sin(angle);
            cos[t * rotated + index] = cosine;
            cos[t * rotated + index + frequencies.len] = cosine;
            sin[t * rotated + index] = sine;
            sin[t * rotated + index + frequencies.len] = sine;
        }
    }
    assert(cos.len == count * rotated);
}

/// Rotates the first `rotated` dims of one head by the given angle row.
///
/// `rotate_half` splits the rotated block in two halves and returns `(-x2,
/// x1)`; the caller's `out` receives the full head.
pub fn apply(
    head: []const f32,
    cos: []const f32,
    sin: []const f32,
    out: []f32,
) void {
    assert(head.len > 0);
    assert(cos.len == sin.len);
    assert(out.len == head.len);

    const rotated = cos.len;
    assert(rotated <= head.len);
    assert(rotated % 2 == 0);

    const half = rotated / 2;
    for (0..half) |j| {
        const low = head[j];
        const high = head[j + half];
        out[j] = low * cos[j] - high * sin[j];
        out[j + half] = high * cos[j + half] + low * sin[j + half];
    }
    for (rotated..head.len) |j| {
        out[j] = head[j];
    }
    assert(out.len == head.len);
}

test "inverse frequencies follow the 1/theta^(2i/dim) schedule" {
    var frequencies: [4]f32 = undefined;
    inv_freq_init(10000.0, &frequencies);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), frequencies[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), frequencies[1], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), frequencies[2], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.001), frequencies[3], 1e-7);
}

test "mrope_grid sends lane 1 to height and lane 2 to width" {
    const section = [3]u32{ 2, 1, 1 };
    try std.testing.expectEqual(@as(u32, 0), mrope_grid(0, section));
    try std.testing.expectEqual(@as(u32, 1), mrope_grid(1, section));
    try std.testing.expectEqual(@as(u32, 2), mrope_grid(2, section));
    try std.testing.expectEqual(@as(u32, 0), mrope_grid(3, section));
}

test "mrope_grid honours a wider height section" {
    const section = [3]u32{ 4, 2, 1 };
    try std.testing.expectEqual(@as(u32, 1), mrope_grid(1, section));
    try std.testing.expectEqual(@as(u32, 2), mrope_grid(2, section));
    try std.testing.expectEqual(@as(u32, 1), mrope_grid(4, section));
    try std.testing.expectEqual(@as(u32, 0), mrope_grid(5, section));
}

test "cos_sin duplicates both halves and interleaves the grids" {
    const section = [3]u32{ 2, 1, 1 };
    const frequencies = [4]f32{ 1.0, 1.0, 1.0, 1.0 };
    const temporal = [2]u32{ 0, 1 };
    const height = [2]u32{ 0, 50 };
    const width = [2]u32{ 0, 50 };
    var cos: [2 * 8]f32 = undefined;
    var sin: [2 * 8]f32 = undefined;
    cos_sin(
        .{ &temporal, &height, &width },
        &frequencies,
        section,
        &cos,
        &sin,
    );
    // Frequency 0 is 1.0, so the angle is the position itself.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cos[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sin[0], 1e-6);
    // Lane 1 reads the height grid, lane 2 the width grid, so both see 50.
    try std.testing.expectApproxEqAbs(
        std.math.cos(@as(f32, 50.0)),
        cos[1 * 8 + 1],
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        std.math.cos(@as(f32, 50.0)),
        cos[1 * 8 + 2],
        1e-6,
    );
    // The second half of the frequency block repeats the first.
    try std.testing.expectEqual(cos[1 * 8 + 1], cos[1 * 8 + 5]);
}

test "apply rotates half a head and passes the rest through" {
    // rotary dims 4 of 8, cos = sin = the unit angle 0.
    const head = [8]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const cos = [4]f32{ 1.0, 1.0, 1.0, 1.0 };
    const sin = [4]f32{ 0.0, 0.0, 0.0, 0.0 };
    var out: [8]f32 = undefined;
    apply(&head, &cos, &sin, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out[4], 1e-6);
}

test "apply is a quarter turn for a 90 degree angle" {
    // A rotation by +90 degrees maps (x1, x2) to (-x2, x1).
    const head = [4]f32{ 1.0, 0.0, 0.0, 0.0 };
    const quarter = std.math.pi / 2.0;
    const cos = [2]f32{ std.math.cos(quarter), std.math.cos(quarter) };
    const sin = [2]f32{ std.math.sin(quarter), std.math.sin(quarter) };
    var out: [4]f32 = undefined;
    apply(&head, &cos, &sin, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[1], 1e-6);
}
