//! Causal depthwise conv1d, the state handling around it, and the dilated
//! variant the PLE block uses.
//!
//! The reference calls `F.conv1d(..., padding=k-1, groups=channels)` and slices
//! the trailing `seq_len` positions, which is the same as convolving the window
//! `[state | input]` with no padding. Everything is expressed on that window so
//! prefill (zero state) and decode (real state) run the same arithmetic.

const std = @import("std");

const ops = @import("ops.zig");

const assert = std.debug.assert;

/// Geometry of one causal conv over a `[channels, full_len]` window.
///
/// `full_len` counts the retained state plus the new positions, so the output
/// has `full_len - (kernel - 1) * dilation` positions. Non-dilated convs use
/// `dilation = 1`.
pub const ConvShape = struct {
    channels: u32,
    kernel: u32,
    dilation: u32,
    full_len: u32,
};

/// Number of output positions for `shape`.
pub fn out_len(shape: ConvShape) u32 {
    assert(shape.channels > 0);
    assert(shape.kernel > 0);
    assert(shape.dilation > 0);

    const span = (shape.kernel - 1) * shape.dilation;
    assert(span < shape.full_len);
    return shape.full_len - span;
}

/// `out[c][t] = sum_j weight[c][j] * full[c][t + j * dilation]`.
///
/// `full` and `out` are channel-major, which is the layout the reference holds
/// between the two `transpose(1, 2)` calls around the conv.
pub fn causal_conv1d(
    full: []const f32,
    weight: []const f32,
    out: []f32,
    shape: ConvShape,
    activation: ?ops.Activation,
) void {
    assert(shape.channels > 0);
    assert(shape.kernel > 0);
    assert(shape.dilation > 0);

    const channels: usize = shape.channels;
    const kernel: usize = shape.kernel;
    const dilation: usize = shape.dilation;
    const full_len: usize = shape.full_len;
    const positions: usize = out_len(shape);
    assert(full.len == channels * full_len);
    assert(weight.len == channels * kernel);
    assert(out.len == channels * positions);

    for (0..channels) |channel| {
        const window = full[channel * full_len ..][0..full_len];
        const taps = weight[channel * kernel ..][0..kernel];
        for (0..positions) |t| {
            var total: f32 = 0.0;
            for (taps, 0..) |tap, j| {
                total += tap * window[t + j * dilation];
            }
            out[channel * positions + t] = if (activation) |act|
                act.apply(total)
            else
                total;
        }
    }
    assert(out.len == channels * positions);
}

/// Writes `[state | input]` into `full`, the window `causal_conv1d` reads.
pub fn prepend_state(
    state: []const f32,
    input: []const f32,
    full: []f32,
    channels: u32,
    state_len: u32,
    input_len: u32,
) void {
    assert(channels > 0);
    assert(state_len > 0);
    assert(input_len > 0);

    const rows: usize = channels;
    const history: usize = state_len;
    const fresh: usize = input_len;
    assert(state.len == rows * history);
    assert(input.len == rows * fresh);
    assert(full.len == rows * (history + fresh));

    for (0..rows) |channel| {
        const width = history + fresh;
        const target = full[channel * width ..][0..width];
        @memcpy(target[0..history], state[channel * history ..][0..history]);
        @memcpy(target[history..], input[channel * fresh ..][0..fresh]);
    }
    assert(full.len == rows * (history + fresh));
}

/// Keeps the last `state_len` columns of `full` as the next state.
pub fn store_state(
    state: []f32,
    full: []const f32,
    channels: u32,
    full_len: u32,
    state_len: u32,
) void {
    assert(channels > 0);
    assert(state_len > 0);
    assert(full_len >= state_len);

    const rows: usize = channels;
    const window: usize = full_len;
    const keep: usize = state_len;
    assert(state.len == rows * keep);
    assert(full.len == rows * window);

    for (0..rows) |channel| {
        const source = full[channel * window + (window - keep) ..][0..keep];
        @memcpy(state[channel * keep ..][0..keep], source);
    }
    assert(state.len == rows * keep);
}

test "a causal conv leaves the first positions short of left context" {
    // weight = [1, 10, 100] over the window [0, 0, 1, 2, 3].
    const full = [5]f32{ 0.0, 0.0, 1.0, 2.0, 3.0 };
    const weight = [3]f32{ 1.0, 10.0, 100.0 };
    var out: [3]f32 = undefined;
    causal_conv1d(
        &full,
        &weight,
        &out,
        .{ .channels = 1, .kernel = 3, .dilation = 1, .full_len = 5 },
        null,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 210.0), out[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 321.0), out[2], 1e-4);
}

test "the dilated variant skips every other position" {
    const full = [5]f32{ 0.0, 0.0, 1.0, 2.0, 3.0 };
    const weight = [2]f32{ 1.0, 10.0 };
    var out: [3]f32 = undefined;
    causal_conv1d(
        &full,
        &weight,
        &out,
        .{ .channels = 1, .kernel = 2, .dilation = 2, .full_len = 5 },
        null,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), out[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 31.0), out[2], 1e-4);
}

test "the silu activation is applied after the sum" {
    const full = [2]f32{ 0.0, 0.0 };
    const weight = [1]f32{0.0};
    var out: [2]f32 = undefined;
    causal_conv1d(
        &full,
        &weight,
        &out,
        .{ .channels = 1, .kernel = 1, .dilation = 1, .full_len = 2 },
        .silu,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1e-6);
}

test "prepend_state and store_state are a sliding window" {
    var full: [2 * 5]f32 = undefined;
    const state = [2 * 3]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const input = [2 * 2]f32{ 7.0, 8.0, 9.0, 10.0 };
    prepend_state(&state, &input, &full, 2, 3, 2);
    try std.testing.expectEqual(
        [5]f32{ 1.0, 2.0, 3.0, 7.0, 8.0 },
        full[0..5].*,
    );

    var next: [2 * 3]f32 = undefined;
    store_state(&next, &full, 2, 5, 3);
    try std.testing.expectEqual([3]f32{ 3.0, 7.0, 8.0 }, next[0..3].*);
    try std.testing.expectEqual([3]f32{ 6.0, 9.0, 10.0 }, next[3..6].*);
}
