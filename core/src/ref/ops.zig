//! Elementwise and small dense primitives for the fp32 reference.
//!
//! The arithmetic here is a transcription of the PyTorch ops the reference
//! model calls, in f32, without fusion: `x * (1 + w)` RMSNorm, the gated
//! RMSNorm, `silu`, `sigmoid`, `softplus` with PyTorch's threshold, full-row
//! softmax, L2 normalisation, and a transposed-weight matmul (`x @ w^T`).

const std = @import("std");

const assert = std.debug.assert;

/// Largest value `softplus` evaluates the logarithm for; above it, `x`.
const softplus_threshold: f32 = 20.0;

/// Activations the model applies elementwise after a projection.
pub const Activation = enum {
    none,
    silu,
    sigmoid,

    /// Applies the activation to `value`.
    pub fn apply(self: Activation, value: f32) f32 {
        assert(value == value);
        assert(value <= std.math.floatMax(f32));

        return switch (self) {
            .none => value,
            .silu => silu(value),
            .sigmoid => sigmoid(value),
        };
    }
};

/// `x * sigmoid(x)`, PyTorch's `SiLU`.
pub fn silu(x: f32) f32 {
    assert(x == x);
    assert(x <= std.math.floatMax(f32));

    const result = x / (1.0 + std.math.exp(-x));
    assert(result == result);
    return result;
}

/// Logistic function, computed the way PyTorch does (`1 / (1 + exp(-x))`).
pub fn sigmoid(x: f32) f32 {
    assert(x == x);
    assert(x <= std.math.floatMax(f32));

    const result = 1.0 / (1.0 + std.math.exp(-x));
    assert(result >= 0.0);
    return result;
}

/// `log(1 + exp(x))`, with PyTorch's linear regime above the threshold.
pub fn softplus(x: f32) f32 {
    assert(x == x);
    assert(x <= std.math.floatMax(f32));

    if (x > softplus_threshold) {
        assert(x > 0.0);
        return x;
    } else {
        const result = @log(1.0 + std.math.exp(x));
        assert(result >= 0.0);
        return result;
    }
}

/// RMSNorm over `group_size` slices of `row`, scaled by `1 + weight`.
///
/// `group_size == 0` normalises the whole row (the reference's
/// `group_size=None`). The tiny model uses `group_size=hidden_size` for the
/// hyper-connection norms and `None` for every other RMSNorm.
pub fn rmsnorm(
    row: []const f32,
    weight: []const f32,
    out: []f32,
    group_size: u32,
    eps: f32,
) void {
    assert(row.len > 0);
    assert(out.len == row.len);
    assert(weight.len == row.len);
    assert(eps >= 0.0);
    assert(eps <= 1.0);

    const columns = row.len;
    const group: usize = if (group_size == 0) columns else group_size;
    assert(group > 0);
    assert(columns % group == 0);

    var start: usize = 0;
    while (start < columns) : (start += group) {
        var sum_squares: f32 = 0.0;
        for (row[start..][0..group]) |value| {
            sum_squares += value * value;
        }
        const mean = sum_squares / @as(f32, @floatFromInt(group));
        const scale = 1.0 / @sqrt(mean + eps);
        for (row[start..][0..group], 0..) |value, i| {
            out[start + i] = value * scale * (1.0 + weight[start + i]);
        }
    }
    assert(start == columns);
}

/// `RMSNormGated`: whole-row RMSNorm, times `weight`, gated by `act(z)`.
///
/// The gate activation is `config.output_gate_type` for this checkpoint, which
/// is `"sigmoid"`; the reference computes it in f32 before the multiply.
pub fn rmsnorm_gated(
    row: []const f32,
    z: []const f32,
    weight: []const f32,
    out: []f32,
    eps: f32,
    activation: Activation,
) void {
    assert(row.len > 0);
    assert(z.len == row.len);
    assert(out.len == row.len);
    assert(weight.len == row.len);

    var sum_squares: f32 = 0.0;
    for (row) |value| {
        sum_squares += value * value;
    }
    const mean = sum_squares / @as(f32, @floatFromInt(row.len));
    const scale = 1.0 / @sqrt(mean + eps);
    for (row, 0..) |value, i| {
        out[i] = (weight[i] * (value * scale)) * activation.apply(z[i]);
    }
    assert(out.len == row.len);
}

/// `x * rsqrt(sum(x^2) + eps)` over the whole row.
pub fn l2norm(row: []const f32, out: []f32, eps: f32) void {
    assert(row.len > 0);
    assert(out.len == row.len);
    assert(eps >= 0.0);
    assert(eps <= 1.0);

    var sum_squares: f32 = 0.0;
    for (row) |value| {
        sum_squares += value * value;
    }
    const scale = 1.0 / @sqrt(sum_squares + eps);
    for (row, 0..) |value, i| {
        out[i] = value * scale;
    }
    assert(out.len == row.len);
}

/// Softmax over the whole slice, in place.
///
/// A row whose entries are all `-inf` (a fully masked query) yields zeros
/// rather than NaN: the reference reaches the same result because its mask
/// always leaves the diagonal visible.
pub fn softmax(values: []f32) void {
    assert(values.len > 0);
    assert(values.len <= 1 << 20);

    var largest = values[0];
    for (values[1..]) |value| {
        largest = @max(largest, value);
    }
    if (largest == -std.math.inf(f32)) {
        @memset(values, 0.0);
        assert(values[0] == 0.0);
        return;
    } else {
        assert(largest > -std.math.inf(f32));
    }

    var total: f32 = 0.0;
    for (values) |*value| {
        value.* = std.math.exp(value.* - largest);
        total += value.*;
    }
    assert(total > 0.0);
    const inverse = 1.0 / total;
    for (values) |*value| {
        value.* *= inverse;
    }
    assert(total * inverse > 0.5);
}

/// Shape of one `x @ weight^T` product: `weight` is `[out, in]`.
pub const MatmulShape = struct {
    rows: u32,
    in_features: u32,
    out_features: u32,
};

/// `out[r][o] = sum_i x[r][i] * weight[o][i]` (PyTorch `nn.Linear`, no bias).
pub fn matmul(
    x: []const f32,
    weight: []const f32,
    out: []f32,
    shape: MatmulShape,
) void {
    assert(shape.rows > 0);
    assert(shape.in_features > 0);
    assert(shape.out_features > 0);

    const rows: usize = shape.rows;
    const in_features: usize = shape.in_features;
    const out_features: usize = shape.out_features;
    assert(x.len == rows * in_features);
    assert(weight.len == out_features * in_features);
    assert(out.len == rows * out_features);

    for (0..rows) |row| {
        const row_values = x[row * in_features ..][0..in_features];
        for (0..out_features) |column| {
            const row_weight = weight[column * in_features ..][0..in_features];
            out[row * out_features + column] = dot(row_values, row_weight);
        }
    }
    assert(out.len == rows * out_features);
}

/// Rounds an f32 to the nearest bf16 value and back, ties to even.
///
/// The arena stores keys in bf16 and the model casts its pooled index keys to
/// that dtype, so the reference needs the same rounding at those points. It
/// mirrors `weights/quant.zig`, which the reference cannot import: a module root
/// may not reach outside `src/ref/`.
pub fn round_bf16(value: f32) f32 {
    assert(@sizeOf(f32) == 4);
    assert(@sizeOf(u16) == 2);

    const bits: u32 = @bitCast(value);
    const magnitude = bits & 0x7fff_ffff;
    // NaN must stay NaN, so quiet it instead of rounding into an infinity.
    if (magnitude > 0x7f80_0000) {
        const quiet: u32 = (bits & 0xffff_0000) | 0x0040_0000;
        return @bitCast(quiet);
    } else {
        assert(magnitude <= 0x7f80_0000);
    }
    // 0x7fff is the largest remainder that must round up, and the extra bit
    // breaks exact ties towards an even mantissa.
    const rounded = bits +% (0x7fff +% ((bits >> 16) & 1));
    const half: u32 = rounded & 0xffff_0000;
    const result: f32 = @bitCast(half);
    assert(@as(u32, @bitCast(result)) == half);
    return result;
}

/// Single-vector matrix product: `out[o] = sum_i x[i] * weight[o][i]`.
pub fn gemv(
    x: []const f32,
    weight: []const f32,
    out: []f32,
    shape: MatmulShape,
) void {
    assert(shape.rows == 1);
    assert(x.len > 0);
    assert(out.len == shape.out_features);

    matmul(x, weight, out, shape);
    assert(out.len == shape.out_features);
}

/// Dot product of two equal-length vectors, accumulated in f32 in order.
pub fn dot(left: []const f32, right: []const f32) f32 {
    assert(left.len > 0);
    assert(right.len == left.len);

    var total: f32 = 0.0;
    for (left, right) |a, b| {
        total += a * b;
    }
    assert(total == total);
    return total;
}

/// Copies `count` rows of `dim` values from `table` at `ids` into `out`.
pub fn embed_gather(
    table: []const f32,
    dim: u32,
    ids: []const i64,
    out: []f32,
) void {
    assert(dim > 0);
    assert(ids.len > 0);
    assert(out.len == ids.len * dim);

    const columns: usize = dim;
    const rows: usize = table.len / columns;
    for (ids, 0..) |id, i| {
        assert(id >= 0);
        assert(id < rows);
        const source = table[@as(usize, @intCast(id)) * columns ..][0..columns];
        @memcpy(out[i * columns ..][0..columns], source);
    }
    assert(out.len == ids.len * columns);
}

/// Copy of `source` scaled by `factor`, written to `out`.
pub fn scale_into(source: []const f32, factor: f32, out: []f32) void {
    assert(source.len > 0);
    assert(out.len == source.len);
    assert(factor == factor);

    for (source, 0..) |value, i| {
        out[i] = value * factor;
    }
    assert(out.len == source.len);
}

/// Index of the first largest value (PyTorch `argmax` on a tie).
pub fn argmax(values: []const f32) u32 {
    assert(values.len > 0);
    assert(values.len <= 1 << 20);

    var best_index: u32 = 0;
    var best = values[0];
    for (values[1..], 1..) |value, i| {
        if (value > best) {
            best = value;
            best_index = @intCast(i);
        } else {
            continue;
        }
    }
    assert(best_index < values.len);
    return best_index;
}

test "silu and sigmoid match their defining formulas" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), silu(0.0), 1e-9);
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.7310586),
        silu(1.0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid(0.0), 1e-9);
    try std.testing.expectApproxEqAbs(
        @as(f32, -0.26894143),
        silu(-1.0),
        1e-6,
    );
}

test "softplus switches to the linear regime above 20" {
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.6931472),
        softplus(0.0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), softplus(25.0), 1e-6);
}

test "rmsnorm normalises per group and scales by 1 + weight" {
    var out: [4]f32 = undefined;
    const weight = [4]f32{ 0.0, 0.0, 1.0, -0.5 };
    rmsnorm(&[4]f32{ 1.0, 1.0, 2.0, 2.0 }, &weight, &out, 2, 0.0);
    // Group 0: mean square 1 -> scale 1 -> 1 * (1 + 0).
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[1], 1e-6);
    // Group 1: mean square 4 -> scale 1/2 -> 2 * 0.5 * (1 + w).
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[3], 1e-6);
}

test "rmsnorm_gated multiplies the normed row by the gated activation" {
    var out: [2]f32 = undefined;
    const weight = [2]f32{ 1.0, 1.0 };
    rmsnorm_gated(
        &[2]f32{ 3.0, 4.0 },
        &[2]f32{ 0.0, 0.0 },
        &weight,
        &out,
        0.0,
        .silu,
    );
    // scale = 1/5, so the normed row is (0.6, 0.8); silu(0) = 0.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1e-6);
}

test "l2norm makes a unit vector" {
    var out: [3]f32 = undefined;
    l2norm(&[3]f32{ 3.0, 0.0, 4.0 }, &out, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), out[2], 1e-6);
}

test "softmax of a fully masked row is zero, not NaN" {
    const masked = -std.math.inf(f32);
    var values = [3]f32{ masked, masked, masked };
    softmax(&values);
    for (values) |value| {
        try std.testing.expectEqual(@as(f32, 0.0), value);
    }
}

test "softmax sums to one" {
    var values = [3]f32{ 1.0, 2.0, 3.0 };
    softmax(&values);
    const total = values[0] + values[1] + values[2];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-6);
}

test "matmul applies the transposed weight layout of nn.Linear" {
    // weight is [out=2, in=3].
    const weight = [6]f32{ 1.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
    var out: [4]f32 = undefined;
    matmul(
        &[6]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 },
        &weight,
        &out,
        .{ .rows = 2, .in_features = 3, .out_features = 2 },
    );
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), out[3], 1e-6);
}

test "embed_gather copies whole rows" {
    const table = [6]f32{ 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 };
    var out: [4]f32 = undefined;
    embed_gather(&table, 2, &[2]i64{ 2, 0 }, &out);
    try std.testing.expectEqual([4]f32{ 2.0, 2.0, 0.0, 0.0 }, out);
}

test "argmax keeps the first index on a tie" {
    try std.testing.expectEqual(@as(u32, 1), argmax(&[3]f32{ 0.0, 5.0, 5.0 }));
    try std.testing.expectEqual(@as(u32, 2), argmax(&[3]f32{ 0.0, 1.0, 2.0 }));
}
