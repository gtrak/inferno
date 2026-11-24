//! Launch wrappers and the acceptance suite for the Gated DeltaNet kernels.
//!
//! `core/src/kernels/gdn.cu` holds the four kernels; this file owns their
//! launch geometry, the dynamic shared-memory sizes they need, and the tests
//! that compare them against the fp32 CPU oracle (`core/src/ref/gdn.zig`) and
//! against the committed HF golden trace.
//!
//! Dtypes follow the reference: `query`, `key` and `value` are f32, because
//! `modeling_qwen4_exp.py:309-313` upcasts them before the delta rule and keeps
//! them f32 through `l2norm`; `beta`, `decay` and the recurrent state are the
//! rule's own f32 arithmetic; only activations that cross a kernel boundary (the
//! conv input and output, `out` and the gate `z`) are bf16.
//!
//! Layouts, all token-major unless the state is named:
//!
//!   conv     x, out         `[tokens][conv_dim]` bf16
//!            weight         `[conv_dim][kernel]` f32
//!            state          `[conv_dim][kernel - 1]` f32, the per-sequence ring
//!   prefill  query, key     `[tokens][num_k_heads][k_head]` f32, l2-normalised
//!                           and scaled by `k_head ** -0.5`
//!            value          `[tokens][num_v_heads][v_head]` f32
//!            out            `[tokens][num_v_heads][v_head]` bf16
//!            beta, decay    `[tokens][num_v_heads]` f32
//!            state          `[num_v_heads][k_head][v_head]` f32
//!   decode   query, key     `[num_k_heads][k_head]` f32, one token
//!            value          `[num_v_heads][v_head]` f32, one token
//!            gate           `[num_v_heads][v_head]` bf16, one token
//!            out            `[num_v_heads][v_head]` bf16, one token
//!            beta, decay    `[num_v_heads]` f32
//!            state          `[num_v_heads][k_head][v_head]` f32
//!   norm     values, gate   `[rows][v_head]` bf16; out `[rows][v_head]` bf16
//!            weight         `[v_head]` f32
//!
//! A layer costs three launches for prefill (`gdn_conv`, `gdn_prefill`,
//! `gdn_gate_norm`) and two for decode: `gdn_conv` with one token, then
//! `gdn_decode`, which applies the gated norm inside the block that owns the
//! value head. Decode is the hot path (36 layers per emitted token), so the
//! norm is fused there rather than launched again.

const std = @import("std");

const cuda_context = @import("../hw/api.zig").cuda_context;
const cuda_driver = @import("../hw/api.zig").cuda_driver;
const kernel = @import("../hw/kernel.zig");
const generated_cubins = @import("generated_cubins");

const assert = std.debug.assert;

/// Chunk of the UT transform; `ref/gdn.zig` fixes the same value.
const chunk_size: u32 = 64;

/// Largest head dimension the kernels are compiled for.
const head_dim_max: u32 = 128;

/// `gdn_conv` geometry, mirrored from `gdn.cu`: 32 channels, up to 8 tokens
/// per block. A single-token step launches one token slot, not eight.
const conv_channels_per_block: u32 = 32;
const conv_token_tile_max: u32 = 8;

/// Floats of scratch the block-wide reduction in `gdn.cu` needs.
const reduction_scratch_floats: u32 = 32;

/// Dynamic shared memory one block may opt into on this GPU: 99 KiB, read from
/// `cudaDeviceProp::sharedMemPerBlockOptin` (`sharedMemPerMultiprocessor` is
/// 100 KiB). The prefill and decode tiles are sized against it, so a layout
/// change that overflows is a load-time failure, not a launch-time one.
const dynamic_shared_bytes_max: u32 = 99 * 1024;

/// Block widths, mirrored from `gdn.cu`.
const prefill_threads: u32 = 2 * head_dim_max;
const gate_norm_threads: u32 = head_dim_max;

/// Block width of `gdn_decode`.
///
/// One thread per value column does the delta rule, but the kernel's traffic is
/// the 128x128 fp32 state moving to and from global memory, so the block is
/// widened to eight columns' worth of threads: they take part in the
/// cooperative load and store and idle during the column phase, which lifts the
/// measured state bandwidth at the production shape from 125 GB/s at 128
/// threads to 330 GB/s (of the 427 GB/s the I-002 probe measured) at 1024.
/// Threads above `v_head` still reach every barrier, which is why the kernel
/// guards the column phase on `threadIdx.x`.
const decode_threads: u32 = 8 * head_dim_max;

/// Name `core/build.zig` gives the cubin compiled from `gdn.cu`.
const cubin_name = "gdn";

/// Kernel names, which are the `extern "C"` symbols inside the cubin.
const name_conv = "gdn_conv";
const name_prefill = "gdn_prefill";
const name_decode = "gdn_decode";
const name_gate_norm = "gdn_gate_norm";

/// Shapes of one layer's delta rule.
pub const DeltaShape = struct {
    tokens: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_head: u32,
    v_head: u32,

    /// Rejects the shapes the kernels cannot express.
    pub fn check(self: DeltaShape) void {
        assert(self.tokens > 0);
        assert(self.k_head > 0);
        assert(self.k_head <= head_dim_max);
        assert(self.v_head > 0);
        assert(self.v_head <= head_dim_max);
        assert(self.num_k_heads > 0);
        assert(self.num_v_heads % self.num_k_heads == 0);
    }
};

/// Device pointers of one `gdn_conv` launch.
pub const ConvTensors = struct {
    x: cuda_driver.DevicePtr,
    weight: cuda_driver.DevicePtr,
    state: cuda_driver.DevicePtr,
    out: cuda_driver.DevicePtr,

    fn check(self: ConvTensors) void {
        assert(self.x != 0);
        assert(self.weight != 0);
        assert(self.state != 0);
        assert(self.out != 0);
    }
};

/// Device pointers of one `gdn_prefill` launch.
pub const PrefillTensors = struct {
    query: cuda_driver.DevicePtr,
    key: cuda_driver.DevicePtr,
    value: cuda_driver.DevicePtr,
    beta: cuda_driver.DevicePtr,
    decay: cuda_driver.DevicePtr,
    state: cuda_driver.DevicePtr,
    out: cuda_driver.DevicePtr,

    fn check(self: PrefillTensors) void {
        assert(self.query != 0);
        assert(self.key != 0);
        assert(self.value != 0);
        assert(self.state != 0);
        assert(self.out != 0);
    }
};

/// Device pointers of one `gdn_decode` launch, gate and norm weight included
/// because the kernel applies the gated norm itself.
pub const DecodeTensors = struct {
    query: cuda_driver.DevicePtr,
    key: cuda_driver.DevicePtr,
    value: cuda_driver.DevicePtr,
    beta: cuda_driver.DevicePtr,
    decay: cuda_driver.DevicePtr,
    state: cuda_driver.DevicePtr,
    gate: cuda_driver.DevicePtr,
    norm_weight: cuda_driver.DevicePtr,
    out: cuda_driver.DevicePtr,

    fn check(self: DecodeTensors) void {
        assert(self.query != 0);
        assert(self.state != 0);
        assert(self.gate != 0);
        assert(self.out != 0);
    }
};

/// Device pointers of one `gdn_gate_norm` launch.
pub const GateNormTensors = struct {
    values: cuda_driver.DevicePtr,
    gate: cuda_driver.DevicePtr,
    weight: cuda_driver.DevicePtr,
    out: cuda_driver.DevicePtr,

    fn check(self: GateNormTensors) void {
        assert(self.values != 0);
        assert(self.gate != 0);
        assert(self.weight != 0);
        assert(self.out != 0);
    }
};

/// The four kernel entry points of one loaded `gdn` module.
pub const Kernels = struct {
    module: kernel.Module,
    conv: kernel.Function,
    prefill: kernel.Function,
    decode: kernel.Function,
    gate_norm: kernel.Function,

    /// Loads the cubin `core/build.zig` embedded for `gdn.cu`.
    ///
    /// The context must be current. The dynamic shared-memory limit is raised
    /// once, here, to the production head size: the driver treats it as a
    /// per-function maximum, so every smaller request (the tiny fixture shapes
    /// this suite runs) stays legal and no launch pays for the call.
    pub fn load() !Kernels {
        const image = cubinImage() orelse return error.GdnCubinMissing;
        const module = try kernel.Module.load(image);
        errdefer module.unload();

        const entry = try loadEntryPoints(module);
        const kernels = Kernels{
            .module = module,
            .conv = entry.conv,
            .prefill = entry.prefill,
            .decode = entry.decode,
            .gate_norm = entry.gate_norm,
        };
        try kernels.prefill.setDynamicSharedBytes(maxPrefillSharedBytes());
        try kernels.decode.setDynamicSharedBytes(maxDecodeSharedBytes());
        assert(kernels.module.handle != null);
        return kernels;
    }

    /// Unloads the module; the context must still be current.
    pub fn deinit(kernels: *const Kernels) void {
        assert(kernels.module.handle != null);
        assert(@intFromPtr(kernels.conv.handle) != 0);

        kernels.module.unload();
    }
};

/// The four entry points, looked up together so a partial load can be undone by
/// the caller's `errdefer`.
const EntryPoints = struct {
    conv: kernel.Function,
    prefill: kernel.Function,
    decode: kernel.Function,
    gate_norm: kernel.Function,
};

/// Looks every kernel up by its `extern "C"` name.
fn loadEntryPoints(module: kernel.Module) !EntryPoints {
    assert(module.handle != null);
    assert(name_conv.len > 0);

    return .{
        .conv = try module.function(name_conv),
        .prefill = try module.function(name_prefill),
        .decode = try module.function(name_decode),
        .gate_norm = try module.function(name_gate_norm),
    };
}

/// The `gdn` cubin image, or null when the build embedded no such kernel.
fn cubinImage() ?[]const u8 {
    assert(cubin_name.len > 0);
    assert(generated_cubins.cubins.len > 0);

    for (&generated_cubins.cubins) |*entry| {
        if (std.mem.eql(u8, std.mem.span(entry.name), cubin_name)) {
            const image = entry.bytes[0..entry.length];
            assert(image.len > 0);
            return image;
        }
    }
    return null;
}

/// Dynamic shared bytes `gdn_prefill` needs for `shape`: the state tile, one
/// chunk of output, the four chunk vectors (`cum`, `beta`, the UT row and the
/// intra-chunk attention column) and the solved `k_cumdecay` row.
fn prefillSharedBytes(shape: DeltaShape) u32 {
    shape.check();
    assert(shape.num_k_heads > 0);
    assert(chunk_size > 0);

    const floats = shape.k_head * shape.v_head +
        chunk_size * shape.v_head +
        4 * chunk_size +
        shape.k_head;
    const bytes = floats * @sizeOf(f32);
    assert(bytes <= dynamic_shared_bytes_max);
    return bytes;
}

/// Dynamic shared bytes `gdn_decode` needs for `shape`: the state tile, the
/// query and key rows, the output row and the reduction scratch.
fn decodeSharedBytes(shape: DeltaShape) u32 {
    shape.check();
    assert(shape.v_head <= head_dim_max);

    // The recurrent state tile is 128 x 128 fp32 = 64 KiB at the production
    // head size, the largest single buffer of the four kernels.
    const floats = shape.k_head * shape.v_head +
        2 * shape.k_head +
        shape.v_head +
        reduction_scratch_floats;
    const bytes = floats * @sizeOf(f32);
    assert(bytes <= dynamic_shared_bytes_max);
    return bytes;
}

/// Dynamic shared bytes `gdn_gate_norm` needs for one row of `width`.
fn gateNormSharedBytes(width: u32) u32 {
    assert(width > 0);
    assert(width <= head_dim_max);

    return (width + reduction_scratch_floats) * @sizeOf(f32);
}

/// Largest `gdn_prefill` shared request, for the production head dimensions.
fn maxPrefillSharedBytes() u32 {
    assert(head_dim_max > 0);
    assert(chunk_size > 0);

    return prefillSharedBytes(.{
        .tokens = 1,
        .num_k_heads = 1,
        .num_v_heads = 1,
        .k_head = head_dim_max,
        .v_head = head_dim_max,
    });
}

/// Largest `gdn_decode` shared request, for the production head dimensions.
fn maxDecodeSharedBytes() u32 {
    assert(head_dim_max > 0);
    assert(chunk_size == 64);

    return decodeSharedBytes(.{
        .tokens = 1,
        .num_k_heads = 1,
        .num_v_heads = 1,
        .k_head = head_dim_max,
        .v_head = head_dim_max,
    });
}

/// Launches `gdn_conv` over `tokens` tokens of `conv_dim` channels.
///
/// One launch covers a whole prefill and a single-token decode; the kernel
/// keeps the history ring per channel, so the state is correct either way.
pub fn launchConv(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    tensors: ConvTensors,
    tokens: u32,
    conv_dim: u32,
) kernel.KernelError!void {
    tensors.check();
    assert(tokens > 0);
    assert(conv_dim > 0);

    var x = tensors.x;
    var weight = tensors.weight;
    var state = tensors.state;
    var out = tensors.out;
    var count: i32 = @intCast(tokens);
    var channels: i32 = @intCast(conv_dim);
    var parameters = [_]?*anyopaque{
        @ptrCast(&x),
        @ptrCast(&weight),
        @ptrCast(&state),
        @ptrCast(&out),
        @ptrCast(&count),
        @ptrCast(&channels),
    };
    assert(parameters.len == 6);

    const grid = [3]u32{
        std.math.divCeil(u32, conv_dim, conv_channels_per_block) catch unreachable,
        1,
        1,
    };
    const tile = @min(tokens, conv_token_tile_max);
    try kernels.conv.launch(
        grid,
        .{ conv_channels_per_block, tile, 1 },
        0,
        stream,
        parameters[0..],
    );
}

/// Launches `gdn_prefill` over the whole sequence, one block per value head.
pub fn launchPrefill(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    tensors: PrefillTensors,
    shape: DeltaShape,
) kernel.KernelError!void {
    tensors.check();
    shape.check();
    assert(shape.num_v_heads > 0);
    assert(shape.tokens > 0);

    var query = tensors.query;
    var key = tensors.key;
    var value = tensors.value;
    var beta = tensors.beta;
    var decay = tensors.decay;
    var state = tensors.state;
    var out = tensors.out;
    var tokens: i32 = @intCast(shape.tokens);
    var k_heads: i32 = @intCast(shape.num_k_heads);
    var v_heads: i32 = @intCast(shape.num_v_heads);
    var k_head: i32 = @intCast(shape.k_head);
    var v_head: i32 = @intCast(shape.v_head);
    var parameters = [_]?*anyopaque{
        @ptrCast(&query),
        @ptrCast(&key),
        @ptrCast(&value),
        @ptrCast(&beta),
        @ptrCast(&decay),
        @ptrCast(&state),
        @ptrCast(&out),
        @ptrCast(&tokens),
        @ptrCast(&k_heads),
        @ptrCast(&v_heads),
        @ptrCast(&k_head),
        @ptrCast(&v_head),
    };
    assert(parameters.len == 12);

    try kernels.prefill.launch(
        .{ shape.num_v_heads, 1, 1 },
        .{ prefill_threads, 1, 1 },
        prefillSharedBytes(shape),
        stream,
        parameters[0..],
    );
}

/// Launches `gdn_decode` for one token and one value head per block; the gated
/// norm of each head is part of the same launch.
pub fn launchDecode(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    tensors: DecodeTensors,
    shape: DeltaShape,
    epsilon: f32,
    activation: i32,
) kernel.KernelError!void {
    tensors.check();
    shape.check();
    assert(shape.tokens == 1);
    assert(epsilon > 0.0);

    var query = tensors.query;
    var key = tensors.key;
    var value = tensors.value;
    var beta = tensors.beta;
    var decay = tensors.decay;
    var state = tensors.state;
    var gate = tensors.gate;
    var norm_weight = tensors.norm_weight;
    var out = tensors.out;
    var k_heads: i32 = @intCast(shape.num_k_heads);
    var k_head: i32 = @intCast(shape.k_head);
    var v_head: i32 = @intCast(shape.v_head);
    var variance_epsilon = epsilon;
    var gate_activation = activation;
    var parameters = [_]?*anyopaque{
        @ptrCast(&query),
        @ptrCast(&key),
        @ptrCast(&value),
        @ptrCast(&beta),
        @ptrCast(&decay),
        @ptrCast(&state),
        @ptrCast(&gate),
        @ptrCast(&norm_weight),
        @ptrCast(&out),
        @ptrCast(&k_heads),
        @ptrCast(&k_head),
        @ptrCast(&v_head),
        @ptrCast(&variance_epsilon),
        @ptrCast(&gate_activation),
    };
    assert(parameters.len == 14);

    try kernels.decode.launch(
        .{ shape.num_v_heads, 1, 1 },
        .{ decode_threads, 1, 1 },
        decodeSharedBytes(shape),
        stream,
        parameters[0..],
    );
}

/// Launches `gdn_gate_norm` over `rows` rows of `width` values.
pub fn launchGateNorm(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    tensors: GateNormTensors,
    rows: u32,
    width: u32,
    epsilon: f32,
    activation: i32,
) kernel.KernelError!void {
    tensors.check();
    assert(rows > 0);
    assert(width > 0);
    assert(width <= head_dim_max);

    var values = tensors.values;
    var gate = tensors.gate;
    var weight = tensors.weight;
    var out = tensors.out;
    var row_width: i32 = @intCast(width);
    var variance_epsilon = epsilon;
    var gate_activation = activation;
    var parameters = [_]?*anyopaque{
        @ptrCast(&values),
        @ptrCast(&gate),
        @ptrCast(&weight),
        @ptrCast(&out),
        @ptrCast(&row_width),
        @ptrCast(&variance_epsilon),
        @ptrCast(&gate_activation),
    };
    assert(parameters.len == 7);

    try kernels.gate_norm.launch(
        .{ rows, 1, 1 },
        .{ gate_norm_threads, 1, 1 },
        gateNormSharedBytes(width),
        stream,
        parameters[0..],
    );
}

// ---------------------------------------------------------------------------
// Acceptance suite.
//
// Every case runs the same three-stage chain the engine runs, in the dtypes the
// engine uses (f32 q/k/v, bf16 store, fp32 recurrent state), and compares it
// against the fp32 CPU oracle in `core/src/ref/gdn.zig`. The two
// reference-of-record checks are the per-stage error table against the oracle
// and the invariant that one `gdn_prefill` over S tokens and S `gdn_decode`
// calls agree at every position: the second is what catches chunking, decay and
// padding mistakes.
// ---------------------------------------------------------------------------

const fixtures = @import("../ref/fixtures.zig");
const ref_conv = @import("../ref/conv.zig");
const ref_gdn = @import("../ref/gdn.zig");
const ref_ops = @import("../ref/ops.zig");

/// Bytes of pinned host memory a suite stages uploads through.
const staging_bytes: u64 = 16 * 1024 * 1024;

/// Bytes of device memory a suite allocates once and bumps within.
const device_bytes: u64 = 64 * 1024 * 1024;

/// Alignment of every tensor inside both arenas.
const tensor_alignment: u64 = 256;

/// The lengths the suite runs. The non-multiples of 64 walk the padding path
/// of the chunked form; 1024 covers sixteen chunks and the cross-chunk scan.
const tested_lengths = [_]u32{ 1, 63, 64, 65, 128, 1024 };

/// The issue's tolerance for a kernel against the oracle under the dtype
/// contract (f32 q/k/v, bf16 out and gate, fp32 state).
const value_tolerance: f64 = 4.0e-3;

/// The issue's tolerance for the final recurrent state, relative.
const state_tolerance: f64 = 1.0e-3;

/// bf16 bits of `value`, rounded to nearest even.
fn bf16Bits(value: f32) u16 {
    assert(value == value);
    assert(@abs(value) <= std.math.floatMax(f32));

    const bits: u32 = @bitCast(value);
    const rounded = bits +% 0x7fff +% ((bits >> 16) & 1);
    return @intCast(rounded >> 16);
}

/// The f32 value the bf16 encoding `bits` represents.
fn bf16Value(bits: u16) f32 {
    assert(@sizeOf(u16) == 2);
    assert(@sizeOf(f32) == 4);

    return @bitCast(@as(u32, bits) << 16);
}

/// Scaled difference between an oracle result and a kernel result.
///
/// `scaled` is the metric the issue gates on; `absolute` and `reference_max`
/// are printed with it because the scaled figure is dominated by rounding
/// whenever the tensor itself is small (the tiny fixture's `core_attn_out` is
/// about 1e-4, so an absolute error of 1e-7 already reads as 1e-3).
const Comparison = struct {
    absolute: f64,
    scaled: f64,
    reference_max: f64,
    index: u32,
    exact: u32,
    elements: u32,
};

/// Compares two equally shaped f32 views. `scaled` is the metric the issue
/// gates on: `max|expected - actual| / max(1e-6, max|expected|)`.
///
/// The oracle is expected to be finite; a non-finite kernel value scores an
/// infinite error. Without that, `NaN > largest` is false and a NaN out of the
/// triangular solve would contribute zero error and pass every gate.
fn compare(expected: []const f32, actual: []const f32) Comparison {
    assert(expected.len > 0);
    assert(actual.len == expected.len);
    for (expected) |want| {
        assert(std.math.isFinite(want));
    }

    var largest: f64 = 0.0;
    var magnitude: f64 = 0.0;
    var worst: u32 = 0;
    var exact: u32 = 0;
    for (expected, actual, 0..) |want, got, index| {
        const difference = if (std.math.isFinite(got))
            @abs(@as(f64, want) - @as(f64, got))
        else
            std.math.inf(f64);
        if (difference > largest) {
            largest = difference;
            worst = @intCast(index);
        }
        magnitude = @max(magnitude, @abs(@as(f64, want)));
        if (want == got) {
            exact += 1;
        }
    }
    const scaled = largest / @max(1e-6, magnitude);
    assert(scaled >= 0.0);
    assert(exact <= expected.len);
    return .{
        .absolute = largest,
        .scaled = scaled,
        .reference_max = magnitude,
        .index = worst,
        .exact = exact,
        .elements = @intCast(expected.len),
    };
}

test "compare scores a non-finite kernel value as an infinite error" {
    const expected = [_]f32{ 0.25, 0.5, -0.125 };
    const exact = [_]f32{ 0.25, 0.5, -0.125 };
    const poisoned = [_]f32{ 0.25, std.math.nan(f32), -0.125 };

    // `NaN > largest` is false, so without the finiteness test the NaN element
    // contributes zero error and the row stays under the 4e-3 gate.
    const clean = compare(expected[0..], exact[0..]);
    const broken = compare(expected[0..], poisoned[0..]);
    try std.testing.expect(clean.scaled == 0.0);
    try std.testing.expect(!(broken.scaled <= value_tolerance));
    try std.testing.expect(broken.index == 1);
}

/// Prints one comparison, with the absolute error and the reference magnitude
/// the scaled figure is divided by.
fn report(label: []const u8, tokens: u32, result: Comparison) void {
    assert(label.len > 0);
    assert(result.elements > 0);

    std.debug.print(
        "  {s:<11} S={d:<5} scaled {e:.3}  abs {e:.3}  ref {e:.3}  {d}/{d}\n",
        .{
            label,
            tokens,
            result.scaled,
            result.absolute,
            result.reference_max,
            result.exact,
            result.elements,
        },
    );
}

/// Prints one comparison and fails when it is over `tolerance`.
fn expectWithin(
    label: []const u8,
    tokens: u32,
    result: Comparison,
    tolerance: f64,
) !void {
    assert(label.len > 0);
    assert(tolerance > 0.0);

    report(label, tokens, result);
    try std.testing.expect(result.scaled <= tolerance);
}

/// A CUDA context with one stream and two bump arenas, reused by every case.
const Harness = struct {
    context: cuda_context.Context,
    stream: cuda_context.Stream,
    staging: cuda_context.PinnedHostBuffer,
    device: cuda_context.DeviceBuffer,
    kernels: Kernels,
    staged: u64,
    allocated: u64,

    /// Retains GPU 0 and loads the `gdn` cubin; skips on a host without a driver.
    fn init() !Harness {
        var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
        const count = cuda_driver.query_devices(devices[0..]) catch |err| switch (err) {
            error.driver_library_missing => return error.SkipZigTest,
            else => return err,
        };
        try std.testing.expect(count > 0);

        const context = try cuda_context.Context.retain(&devices[0]);
        errdefer context.release();
        try context.set_current();
        const stream = try cuda_context.Stream.create();
        errdefer stream.destroy();
        const staging = try cuda_context.PinnedHostBuffer.allocate(
            staging_bytes,
        );
        errdefer staging.free();
        const device = try cuda_context.DeviceBuffer.allocate(device_bytes);
        errdefer device.free();
        const kernels = try Kernels.load();

        return .{
            .context = context,
            .stream = stream,
            .staging = staging,
            .device = device,
            .kernels = kernels,
            .staged = 0,
            .allocated = 0,
        };
    }

    /// Releases everything the harness owns.
    fn deinit(harness: *Harness) void {
        assert(harness.staging.bytes > 0);
        assert(harness.device.bytes > 0);

        harness.kernels.deinit();
        harness.device.free();
        harness.staging.free();
        harness.stream.destroy();
        harness.context.release();
    }

    /// Reserves `bytes` of device memory and returns its address.
    fn allocate(harness: *Harness, bytes: u64) !cuda_driver.DevicePtr {
        assert(bytes > 0);
        assert(harness.allocated <= harness.device.bytes);

        const aligned = std.mem.alignForward(u64, harness.allocated, tensor_alignment);
        if (aligned + bytes > harness.device.bytes) {
            return error.DeviceArenaFull;
        }
        harness.allocated = aligned + bytes;
        return harness.device.pointer + aligned;
    }

    /// Reserves `bytes` of the pinned arena and returns the slice to fill.
    fn stage(harness: *Harness, bytes: u64) ![]u8 {
        assert(bytes > 0);
        assert(harness.staged <= harness.staging.bytes);

        const aligned = std.mem.alignForward(u64, harness.staged, tensor_alignment);
        if (aligned + bytes > harness.staging.bytes) {
            return error.StagingArenaFull;
        }
        harness.staged = aligned + bytes;
        const region = harness.staging.slice()[@intCast(aligned)..][0..@intCast(bytes)];
        assert(region.len == bytes);
        return region;
    }

    /// Enqueues a host-to-device copy of a staged region.
    fn enqueue(
        harness: *Harness,
        destination: cuda_driver.DevicePtr,
        region: []u8,
    ) !void {
        assert(destination != 0);
        assert(region.len > 0);

        const sink = cuda_context.DeviceBuffer{
            .pointer = destination,
            .bytes = region.len,
        };
        const view = cuda_context.PinnedHostBuffer{
            .pointer = region.ptr,
            .bytes = region.len,
        };
        try harness.stream.copy_h_to_d(&sink, &view, region.len);
    }

    /// Stages `source` into an existing device buffer and starts its copy.
    fn write(
        harness: *Harness,
        destination: cuda_driver.DevicePtr,
        comptime Element: type,
        source: []const Element,
    ) !void {
        assert(destination != 0);
        assert(source.len > 0);
        assert(@sizeOf(Element) > 0);

        const bytes = std.mem.sliceAsBytes(source);
        const region = try harness.stage(bytes.len);
        @memcpy(region, bytes);
        try harness.enqueue(destination, region);
    }

    /// Stages `source`, allocates device memory for it and starts its copy.
    fn upload(
        harness: *Harness,
        comptime Element: type,
        source: []const Element,
    ) !cuda_driver.DevicePtr {
        assert(source.len > 0);
        assert(@sizeOf(Element) > 0);

        const destination = try harness.allocate(std.mem.sliceAsBytes(source).len);
        try harness.write(destination, Element, source);
        return destination;
    }

    /// Allocates `count` zeroed elements of the device arena.
    fn zeros(
        harness: *Harness,
        comptime Element: type,
        count: u32,
    ) !cuda_driver.DevicePtr {
        assert(count > 0);
        assert(@sizeOf(Element) > 0);

        const bytes: u64 = @as(u64, count) * @sizeOf(Element);
        const destination = try harness.allocate(bytes);
        const region = try harness.stage(bytes);
        @memset(region, 0);
        try harness.enqueue(destination, region);
        return destination;
    }

    /// Copies `source` back into `destination` and waits for it.
    fn download(
        harness: *Harness,
        comptime Element: type,
        source: cuda_driver.DevicePtr,
        destination: []Element,
    ) !void {
        assert(source != 0);
        assert(destination.len > 0);

        const bytes = std.mem.sliceAsBytes(destination);
        const region = try harness.stage(bytes.len);
        const view = cuda_context.PinnedHostBuffer{
            .pointer = region.ptr,
            .bytes = region.len,
        };
        const origin = cuda_context.DeviceBuffer{
            .pointer = source,
            .bytes = region.len,
        };
        try harness.stream.copy_d_to_h(&view, &origin, region.len);
        try harness.stream.synchronize();
        @memcpy(bytes, region);
    }

    /// Ends a case: waits for its work, then rewinds both arenas.
    fn reset(harness: *Harness) !void {
        assert(harness.staged <= harness.staging.bytes);
        assert(harness.allocated <= harness.device.bytes);

        try harness.stream.synchronize();
        harness.staged = 0;
        harness.allocated = 0;
    }
};

/// Loads the tiny fixture config, raising its position cap to the longest
/// sequence this suite runs.
///
/// The field only sizes the reference's scratch buffers (the fixture caps them
/// at 512 tokens); no arithmetic reads it.
fn loadConfig(
    allocator: std.mem.Allocator,
    max_tokens: u32,
) !fixtures.RefConfig {
    assert(max_tokens > 0);
    assert(max_tokens >= tested_lengths[0]);

    var config = try fixtures.load_config(allocator);
    if (config.max_position_embeddings < max_tokens) {
        config.max_position_embeddings = max_tokens;
    } else {
        // Already long enough; keep the fixture's value.
    }
    assert(config.max_position_embeddings >= max_tokens);
    return config;
}

/// Views one linear-attention tensor of `layer` in the tiny checkpoint.
fn layerTensor(
    tensors: *const fixtures.Tensors,
    layer: u32,
    name: []const u8,
) !fixtures.Tensor {
    assert(name.len > 0);
    assert(layer < fixtures.layers_max);

    var buffer: [128]u8 = undefined;
    const full = try std.fmt.bufPrint(
        buffer[0..],
        "model.language_model.layers.{d}.linear_attn.{s}",
        .{ layer, name },
    );
    const tensor = try tensors.tensor(full);
    assert(tensor.values.len > 0);
    return tensor;
}

/// Loads one layer's linear-attention weights out of the tiny checkpoint.
fn loadGdnWeights(
    tensors: *const fixtures.Tensors,
    layer: u32,
) !ref_gdn.Weights {
    assert(layer < fixtures.layers_max);
    assert(tensors.entry_count > 0);

    const weights = ref_gdn.Weights{
        .in_proj_qkv = (try layerTensor(tensors, layer, "in_proj_qkv.weight")).values,
        .in_proj_z = (try layerTensor(tensors, layer, "in_proj_z.weight")).values,
        .in_proj_b = (try layerTensor(tensors, layer, "in_proj_b.weight")).values,
        .in_proj_a = (try layerTensor(tensors, layer, "in_proj_a.weight")).values,
        .conv = (try layerTensor(tensors, layer, "conv1d.weight")).values,
        .norm = (try layerTensor(tensors, layer, "norm.weight")).values,
        .out_proj = (try layerTensor(tensors, layer, "out_proj.weight")).values,
        .a_log = (try layerTensor(tensors, layer, "A_log")).values,
        .dt_bias = (try layerTensor(tensors, layer, "dt_bias")).values,
    };
    assert(weights.norm.len > 0);
    return weights;
}

/// Fills `input` with deterministic values in `[-1, 1)`: a small LCG, so a
/// failure is reproducible and the values exercise both signs.
fn fillInput(input: []f32, seed_init: u32) void {
    assert(input.len > 0);
    assert(seed_init != 0);

    var seed = seed_init;
    for (input) |*slot| {
        seed = seed *% 1_103_515_245 +% 12_345;
        slot.* = @as(f32, @floatFromInt(seed % 2000)) / 1000.0 - 1.0;
    }
}

/// Fills a bf16 buffer with the same deterministic pattern as `fillInput`.
fn fillBf16(bits: []u16, seed_init: u32) void {
    assert(bits.len > 0);
    assert(seed_init != 0);

    var seed = seed_init;
    for (bits) |*slot| {
        seed = seed *% 1_103_515_245 +% 12_345;
        const value = @as(f32, @floatFromInt(seed % 2000)) / 1000.0 - 1.0;
        slot.* = bf16Bits(value);
    }
}

/// The delta rule's inputs for one sequence, in the kernels' dtypes.
const DeltaInputs = struct {
    query: []f32,
    key: []f32,
    value: []f32,
    beta: []const f32,
    decay: []const f32,

    fn deinit(inputs: *const DeltaInputs, allocator: std.mem.Allocator) void {
        assert(inputs.query.len > 0);
        assert(inputs.value.len > 0);

        allocator.free(inputs.value);
        allocator.free(inputs.key);
        allocator.free(inputs.query);
    }
};

/// The delta-rule shape of a layer as the fixture config reports it.
fn deltaShape(config: *const fixtures.RefConfig, tokens: u32) DeltaShape {
    assert(tokens > 0);
    assert(config.linear_num_key_heads > 0);

    const shape = DeltaShape{
        .tokens = tokens,
        .num_k_heads = config.linear_num_key_heads,
        .num_v_heads = config.linear_num_value_heads,
        .k_head = config.linear_key_head_dim,
        .v_head = config.linear_value_head_dim,
    };
    shape.check();
    return shape;
}

/// q/k/v as the projections produced them: `[tokens][heads][dim]`, f32,
/// before l2-normalisation.
const RawDelta = struct {
    query: []const f32,
    key: []const f32,
    value: []const f32,
};

/// l2-normalises q and k (eps 1e-6), scales q by `k_head ** -0.5` and copies v,
/// exactly as `ref/gdn.zig`'s `load_head` does, leaving all three in f32 as the
/// reference does.
///
/// This is the layer's elementwise work (the projections and I-012 own it, not
/// the four kernels here); modelling it on the host keeps the prefill chain,
/// the decode chain and the oracle on byte-identical delta-rule inputs, which
/// is what makes the prefill-versus-decode invariant a test of the delta rule.
fn normaliseDelta(
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    raw: RawDelta,
    beta: []const f32,
    decay: []const f32,
    tokens: u32,
) !DeltaInputs {
    assert(tokens > 0);
    assert(raw.query.len == @as(usize, tokens) * config.key_dim());
    assert(raw.value.len == @as(usize, tokens) * config.value_dim());
    assert(beta.len >= @as(usize, tokens) * config.linear_num_value_heads);

    const k_heads = config.linear_num_key_heads;
    const k_head = config.linear_key_head_dim;
    const heads = config.linear_num_value_heads;
    const v_head = config.linear_value_head_dim;
    const key_dim = config.key_dim();
    const value_dim = config.value_dim();
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(k_head)));

    const query = try allocator.alloc(f32, @as(usize, tokens) * key_dim);
    errdefer allocator.free(query);
    const key = try allocator.alloc(f32, @as(usize, tokens) * key_dim);
    errdefer allocator.free(key);
    const value = try allocator.alloc(f32, @as(usize, tokens) * value_dim);
    errdefer allocator.free(value);
    const scratch_row = try allocator.alloc(f32, k_head);
    defer allocator.free(scratch_row);

    for (0..tokens) |t| {
        for (0..k_heads) |head| {
            const offset = (t * k_heads + head) * k_head;
            ref_ops.l2norm(raw.query[offset..][0..k_head], scratch_row, 1e-6);
            for (0..k_head) |dim| {
                query[offset + dim] = scratch_row[dim] * scale;
            }
            ref_ops.l2norm(raw.key[offset..][0..k_head], scratch_row, 1e-6);
            for (0..k_head) |dim| {
                key[offset + dim] = scratch_row[dim];
            }
        }
        @memcpy(
            value[t * value_dim ..][0..value_dim],
            raw.value[t * value_dim ..][0..value_dim],
        );
    }
    assert(value.len == @as(usize, tokens) * heads * v_head);
    return .{
        .query = query,
        .key = key,
        .value = value,
        .beta = beta,
        .decay = decay,
    };
}

/// Splits a token-major bf16 conv output into raw q/k/v and normalises it.
fn prepareDeltaInputs(
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    conv_out: []const u16,
    beta: []const f32,
    decay: []const f32,
    tokens: u32,
) !DeltaInputs {
    assert(tokens > 0);
    assert(conv_out.len == @as(usize, tokens) * config.conv_dim());

    const key_dim = config.key_dim();
    const value_dim = config.value_dim();
    const conv_dim = config.conv_dim();
    const total: usize = @as(usize, tokens) * conv_dim;
    const raw = try allocator.alloc(f32, total);
    defer allocator.free(raw);
    for (0..total) |index| {
        raw[index] = bf16Value(conv_out[index]);
    }
    const query = try allocator.alloc(f32, @as(usize, tokens) * key_dim);
    defer allocator.free(query);
    const key = try allocator.alloc(f32, @as(usize, tokens) * key_dim);
    defer allocator.free(key);
    const value = try allocator.alloc(f32, @as(usize, tokens) * value_dim);
    defer allocator.free(value);
    for (0..tokens) |t| {
        @memcpy(query[t * key_dim ..][0..key_dim], raw[t * conv_dim ..][0..key_dim]);
        @memcpy(
            key[t * key_dim ..][0..key_dim],
            raw[t * conv_dim + key_dim ..][0..key_dim],
        );
        @memcpy(
            value[t * value_dim ..][0..value_dim],
            raw[t * conv_dim + 2 * key_dim ..][0..value_dim],
        );
    }
    return try normaliseDelta(
        allocator,
        config,
        .{ .query = query, .key = key, .value = value },
        beta,
        decay,
        tokens,
    );
}

/// The reference's channel-major conv output as a token-major bf16 buffer.
fn tokenMajorConvOutput(
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    channel_major: []const f32,
    tokens: u32,
) ![]u16 {
    assert(tokens > 0);
    assert(channel_major.len == @as(usize, tokens) * config.conv_dim());

    const conv_dim: usize = config.conv_dim();
    const rows: usize = @as(usize, tokens) * conv_dim;
    const output = try allocator.alloc(u16, rows);
    for (0..tokens) |t| {
        for (0..conv_dim) |channel| {
            output[t * conv_dim + channel] = bf16Bits(channel_major[channel * tokens + t]);
        }
    }
    assert(output.len == rows);
    return output;
}

/// The reference conv over the same bf16-rounded input the kernel is given,
/// starting from an empty history ring: writes the expected token-major output
/// and the expected history ring.
fn referenceConv(
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    weights: *const ref_gdn.Weights,
    rounded: []const f32,
    tokens: u32,
    expected: []f32,
    expected_state: []f32,
) !void {
    assert(tokens > 0);
    assert(rounded.len == @as(usize, tokens) * config.conv_dim());

    const conv_dim: usize = config.conv_dim();
    const taps: usize = config.linear_conv_kernel_dim;
    const history: usize = taps - 1;
    const window_len: usize = @as(usize, tokens) + history;
    assert(expected.len == @as(usize, tokens) * conv_dim);
    assert(expected_state.len == conv_dim * history);

    const window = try allocator.alloc(f32, conv_dim * window_len);
    defer allocator.free(window);
    @memset(window, 0.0);
    for (0..conv_dim) |channel| {
        for (0..tokens) |t| {
            window[channel * window_len + history + t] =
                rounded[t * conv_dim + channel];
        }
    }

    const channel_major = try allocator.alloc(f32, conv_dim * tokens);
    defer allocator.free(channel_major);
    ref_conv.causal_conv1d(
        window,
        weights.conv,
        channel_major,
        .{
            .channels = @intCast(conv_dim),
            .kernel = @intCast(taps),
            .dilation = 1,
            .full_len = @intCast(window_len),
        },
        .silu,
    );
    ref_conv.store_state(
        expected_state,
        window,
        @intCast(conv_dim),
        @intCast(window_len),
        @intCast(history),
    );
    for (0..tokens) |t| {
        for (0..conv_dim) |channel| {
            expected[t * conv_dim + channel] = channel_major[channel * tokens + t];
        }
    }
}

/// Compares `gdn_conv` and its history ring against the reference conv.
///
/// Both sides start from the same bf16-rounded input, so what remains is the
/// f32 ordering inside the taps plus the kernel's bf16 output store. The ring
/// holds copies of input values, so it is compared bit for bit.
fn checkConv(
    harness: *Harness,
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    weights: *const ref_gdn.Weights,
    scratch: *const ref_gdn.Scratch,
    tokens: u32,
) !void {
    const conv_dim: usize = config.conv_dim();
    const history: usize = config.linear_conv_kernel_dim - 1;
    const elements: usize = @as(usize, tokens) * conv_dim;
    assert(scratch.mixed_qkv.len >= elements);

    const input = try allocator.alloc(u16, elements);
    defer allocator.free(input);
    const rounded = try allocator.alloc(f32, elements);
    defer allocator.free(rounded);
    const rings = try allocator.alloc(f32, 2 * conv_dim * history);
    defer allocator.free(rings);
    const expected_state = rings[0 .. conv_dim * history];
    const actual_state = rings[conv_dim * history ..];
    const expected = try allocator.alloc(f32, elements);
    defer allocator.free(expected);
    const actual = try allocator.alloc(f32, elements);
    defer allocator.free(actual);
    const actual_bits = try allocator.alloc(u16, elements);
    defer allocator.free(actual_bits);

    for (0..elements) |index| {
        input[index] = bf16Bits(scratch.mixed_qkv[index]);
        rounded[index] = bf16Value(input[index]);
    }
    try referenceConv(
        allocator,
        config,
        weights,
        rounded,
        tokens,
        expected,
        expected_state,
    );

    const device_input = try harness.upload(u16, input);
    const device_weight = try harness.upload(f32, weights.conv);
    const device_state = try harness.zeros(f32, @intCast(conv_dim * history));
    const device_out = try harness.zeros(u16, @intCast(elements));
    try launchConv(
        &harness.kernels,
        harness.stream.handle,
        .{
            .x = device_input,
            .weight = device_weight,
            .state = device_state,
            .out = device_out,
        },
        tokens,
        @intCast(conv_dim),
    );

    try harness.download(u16, device_out, actual_bits);
    for (actual_bits, 0..) |bits, index| actual[index] = bf16Value(bits);
    try expectWithin("conv", tokens, compare(expected, actual), value_tolerance);

    try harness.download(f32, device_state, actual_state);
    for (expected_state, actual_state) |want, got| {
        try std.testing.expectEqual(want, got);
    }
}

/// Compares `gdn_prefill` against the oracle's chunked scan, both starting from
/// a zeroed recurrent state.
fn checkPrefill(
    harness: *Harness,
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    state: *const ref_gdn.State,
    scratch: *const ref_gdn.Scratch,
    inputs: *const DeltaInputs,
    tokens: u32,
) !void {
    const shape = deltaShape(config, tokens);
    const value_rows: usize = @as(usize, tokens) * shape.num_v_heads * shape.v_head;
    const state_elements: usize = shape.num_v_heads * shape.k_head * shape.v_head;
    assert(state.recurrent.len == state_elements);
    assert(scratch.core_out.len >= value_rows);

    const device_query = try harness.upload(f32, inputs.query);
    const device_key = try harness.upload(f32, inputs.key);
    const device_value = try harness.upload(f32, inputs.value);
    const device_beta = try harness.upload(f32, inputs.beta[0 .. @as(usize, tokens) * shape.num_v_heads]);
    const device_decay = try harness.upload(f32, inputs.decay[0 .. @as(usize, tokens) * shape.num_v_heads]);
    const device_state = try harness.zeros(f32, @intCast(state_elements));
    const device_out = try harness.zeros(u16, @intCast(value_rows));
    try launchPrefill(
        &harness.kernels,
        harness.stream.handle,
        .{
            .query = device_query,
            .key = device_key,
            .value = device_value,
            .beta = device_beta,
            .decay = device_decay,
            .state = device_state,
            .out = device_out,
        },
        shape,
    );

    const bits = try allocator.alloc(u16, value_rows);
    defer allocator.free(bits);
    const actual = try allocator.alloc(f32, value_rows);
    defer allocator.free(actual);
    try harness.download(u16, device_out, bits);
    for (bits, 0..) |element, index| actual[index] = bf16Value(element);
    try expectWithin(
        "prefill",
        tokens,
        compare(scratch.core_out[0..value_rows], actual),
        value_tolerance,
    );

    const actual_state = try allocator.alloc(f32, state_elements);
    defer allocator.free(actual_state);
    try harness.download(f32, device_state, actual_state);
    try expectWithin(
        "state",
        tokens,
        compare(state.recurrent, actual_state),
        value_tolerance,
    );
}

/// Compares `gdn_gate_norm` against the oracle's own row function, fed with the
/// same bf16-rounded values the kernel reads.
fn checkGateNorm(
    harness: *Harness,
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    weights: *const ref_gdn.Weights,
    scratch: *const ref_gdn.Scratch,
    tokens: u32,
) !void {
    const width: usize = config.linear_value_head_dim;
    const rows: usize = @as(usize, tokens) * config.linear_num_value_heads;
    const elements: usize = rows * width;
    assert(scratch.normed.len >= elements);

    const values = try allocator.alloc(u16, elements);
    defer allocator.free(values);
    const gate = try allocator.alloc(u16, elements);
    defer allocator.free(gate);
    const actual = try allocator.alloc(f32, elements);
    defer allocator.free(actual);
    const bits = try allocator.alloc(u16, elements);
    defer allocator.free(bits);
    for (0..elements) |index| {
        values[index] = bf16Bits(scratch.core_out[index]);
        gate[index] = bf16Bits(scratch.z[index]);
    }

    // The oracle is the reference's own per-row `rmsnorm_gated`, applied to
    // exactly the values the kernel reads: the bf16 quantisation of the inputs
    // is the engine's dtype policy, not kernel error.
    const expected = try normedExpectation(
        allocator,
        config,
        weights,
        scratch.core_out,
        scratch.z,
        tokens,
    );
    defer allocator.free(expected);

    const device_values = try harness.upload(u16, values);
    const device_gate = try harness.upload(u16, gate);
    const device_weight = try harness.upload(f32, weights.norm);
    const device_out = try harness.zeros(u16, @intCast(elements));
    try launchGateNorm(
        &harness.kernels,
        harness.stream.handle,
        .{
            .values = device_values,
            .gate = device_gate,
            .weight = device_weight,
            .out = device_out,
        },
        @intCast(rows),
        @intCast(width),
        config.rms_norm_eps,
        @intFromEnum(config.output_gate_type),
    );
    try harness.download(u16, device_out, bits);
    for (bits, 0..) |element, index| actual[index] = bf16Value(element);
    try expectWithin(
        "gate_norm",
        tokens,
        compare(expected, actual),
        value_tolerance,
    );
}

/// Prints the `beta` and `g` ranges of the case and fails when either is
/// vacuous: a zeroed `g` would leave the whole decay path untested.
fn reportGates(
    config: *const fixtures.RefConfig,
    scratch: *const ref_gdn.Scratch,
    tokens: u32,
) !void {
    const used: usize = @as(usize, tokens) * config.linear_num_value_heads;
    assert(scratch.beta.len >= used);
    assert(scratch.decay.len >= used);

    var beta_low: f32 = 1.0;
    var beta_high: f32 = 0.0;
    var decay_low: f32 = 0.0;
    var decay_high: f32 = -std.math.floatMax(f32);
    for (0..used) |index| {
        beta_low = @min(beta_low, scratch.beta[index]);
        beta_high = @max(beta_high, scratch.beta[index]);
        decay_low = @min(decay_low, scratch.decay[index]);
        decay_high = @max(decay_high, scratch.decay[index]);
    }
    std.debug.print(
        "  beta    S={d:<5} [{d:.4}, {d:.4}]   g [{d:.4}, {d:.4}]\n",
        .{ tokens, beta_low, beta_high, decay_low, decay_high },
    );
    // `beta = sigmoid(b)` lives in (0, 1) and `g = -exp(A_log) * softplus(..)`
    // is strictly negative once `A_log` is finite and `softplus` is positive.
    try std.testing.expect(beta_low > 0.0);
    try std.testing.expect(beta_high < 1.0);
    try std.testing.expect(decay_high <= 0.0);
    try std.testing.expect(decay_low < -1.0e-4);
}

test "gdn kernels match the CPU reference at every tested length" {
    const allocator = std.testing.allocator;
    var harness = try Harness.init();
    defer harness.deinit();

    const longest = tested_lengths[tested_lengths.len - 1];
    const config = try loadConfig(allocator, longest);
    var tensors = try fixtures.open_checkpoint(allocator);
    defer tensors.close(allocator);
    const weights = try loadGdnWeights(&tensors, 0);

    var state = try ref_gdn.State.init(allocator, &config);
    defer state.deinit(allocator);
    var scratch = try ref_gdn.Scratch.init(allocator, &config);
    defer scratch.deinit(allocator);
    const input = try allocator.alloc(f32, @as(usize, longest) * config.hidden_size);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, @as(usize, longest) * config.hidden_size);
    defer allocator.free(output);

    std.debug.print(
        "gdn vs the fp32 CPU oracle (f32 q/k/v, bf16 store, fp32 state)\n",
        .{},
    );
    for (tested_lengths) |tokens| {
        const rows: usize = @as(usize, tokens) * config.hidden_size;
        fillInput(input[0..rows], 7 + tokens);
        state.reset();
        ref_gdn.forward(
            input[0..rows],
            tokens,
            output[0..rows],
            &weights,
            &state,
            &scratch,
            &config,
            false,
        );

        try reportGates(&config, &scratch, tokens);
        try checkConv(&harness, allocator, &config, &weights, &scratch, tokens);
        const conv_elements: usize = @as(usize, tokens) * config.conv_dim();
        const conv_out = try tokenMajorConvOutput(
            allocator,
            &config,
            scratch.conv_out[0..conv_elements],
            tokens,
        );
        defer allocator.free(conv_out);
        const inputs = try prepareDeltaInputs(
            allocator,
            &config,
            conv_out,
            scratch.beta,
            scratch.decay,
            tokens,
        );
        defer inputs.deinit(allocator);
        // Re-run the oracle on the same dtype-rounded inputs before comparing,
        // so `scratch.core_out` and the recurrent state below are the oracle's
        // answer for the values the kernel actually reads.
        oracleDeltaRule(
            &scratch,
            &config,
            state.recurrent,
            conv_out,
            tokens,
            false,
        );
        try checkPrefill(&harness, allocator, &config, &state, &scratch, &inputs, tokens);
        try checkGateNorm(&harness, allocator, &config, &weights, &scratch, tokens);
        try checkLayer(&harness, allocator, &config, &weights, &scratch, &inputs, tokens);
        try harness.reset();
    }
}

/// The conv result of one chain, read back to the host.
const ConvResult = struct {
    output: []u16,
    ring: []f32,

    fn deinit(result: *const ConvResult, allocator: std.mem.Allocator) void {
        assert(result.output.len > 0);
        assert(result.ring.len > 0);

        allocator.free(result.ring);
        allocator.free(result.output);
    }
};

/// One `gdn_conv` launch over the whole sequence, starting from an empty ring.
fn convSequence(
    harness: *Harness,
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    weight: []const f32,
    input: []const u16,
    tokens: u32,
) !ConvResult {
    assert(tokens > 0);
    assert(input.len == @as(usize, tokens) * config.conv_dim());

    const conv_dim: usize = config.conv_dim();
    const history: usize = config.linear_conv_kernel_dim - 1;
    const elements: usize = @as(usize, tokens) * conv_dim;
    const device_input = try harness.upload(u16, input);
    const device_weight = try harness.upload(f32, weight);
    const device_ring = try harness.zeros(f32, @intCast(conv_dim * history));
    const device_out = try harness.zeros(u16, @intCast(elements));
    try launchConv(
        &harness.kernels,
        harness.stream.handle,
        .{
            .x = device_input,
            .weight = device_weight,
            .state = device_ring,
            .out = device_out,
        },
        tokens,
        @intCast(conv_dim),
    );

    const output = try allocator.alloc(u16, elements);
    errdefer allocator.free(output);
    const ring = try allocator.alloc(f32, conv_dim * history);
    errdefer allocator.free(ring);
    try harness.download(u16, device_out, output);
    try harness.download(f32, device_ring, ring);
    return .{ .output = output, .ring = ring };
}

/// The same conv as `tokens` single-token launches: the decode path, which has
/// to produce the identical output and the identical history ring.
fn convPerToken(
    harness: *Harness,
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    weight: []const f32,
    input: []const u16,
    tokens: u32,
) !ConvResult {
    assert(tokens > 0);
    assert(input.len == @as(usize, tokens) * config.conv_dim());

    const conv_dim: usize = config.conv_dim();
    const history: usize = config.linear_conv_kernel_dim - 1;
    const elements: usize = @as(usize, tokens) * conv_dim;
    const token_bytes: u64 = conv_dim * @sizeOf(u16);
    const device_input = try harness.upload(u16, input);
    const device_weight = try harness.upload(f32, weight);
    const device_ring = try harness.zeros(f32, @intCast(conv_dim * history));
    const device_out = try harness.zeros(u16, @intCast(elements));

    for (0..tokens) |token| {
        const offset = @as(u64, @intCast(token)) * token_bytes;
        try launchConv(
            &harness.kernels,
            harness.stream.handle,
            .{
                .x = device_input + offset,
                .weight = device_weight,
                .state = device_ring,
                .out = device_out + offset,
            },
            1,
            @intCast(conv_dim),
        );
    }

    const output = try allocator.alloc(u16, elements);
    errdefer allocator.free(output);
    const ring = try allocator.alloc(f32, conv_dim * history);
    errdefer allocator.free(ring);
    try harness.download(u16, device_out, output);
    try harness.download(f32, device_ring, ring);
    return .{ .output = output, .ring = ring };
}

/// One run of the layer's delta rule, in whichever form.
const ChainResult = struct {
    normed: []f32,
    state: []f32,

    fn deinit(result: *const ChainResult, allocator: std.mem.Allocator) void {
        assert(result.normed.len > 0);
        assert(result.state.len > 0);

        allocator.free(result.state);
        allocator.free(result.normed);
    }
};

/// Everything one chain run needs besides the harness.
const ChainInputs = struct {
    delta: *const DeltaInputs,
    gate: []const u16,
    norm_weight: []const f32,
    shape: DeltaShape,
    epsilon: f32,
    activation: i32,
};

/// Prefill chain: `gdn_prefill` over the sequence, then `gdn_gate_norm`.
fn prefillChain(
    harness: *Harness,
    allocator: std.mem.Allocator,
    inputs: ChainInputs,
) !ChainResult {
    const shape = inputs.shape;
    shape.check();
    assert(inputs.norm_weight.len == shape.v_head);
    assert(inputs.epsilon > 0.0);

    const rows: usize = @as(usize, shape.tokens) * shape.num_v_heads;
    const value_rows: usize = rows * shape.v_head;
    const state_elements: usize = shape.num_v_heads * shape.k_head * shape.v_head;
    const device_state = try harness.zeros(f32, @intCast(state_elements));
    const device_core = try harness.zeros(u16, @intCast(value_rows));
    try launchPrefill(
        &harness.kernels,
        harness.stream.handle,
        .{
            .query = try harness.upload(f32, inputs.delta.query),
            .key = try harness.upload(f32, inputs.delta.key),
            .value = try harness.upload(f32, inputs.delta.value),
            .beta = try harness.upload(f32, inputs.delta.beta[0..rows]),
            .decay = try harness.upload(f32, inputs.delta.decay[0..rows]),
            .state = device_state,
            .out = device_core,
        },
        shape,
    );
    const device_normed = try harness.zeros(u16, @intCast(value_rows));
    try launchGateNorm(
        &harness.kernels,
        harness.stream.handle,
        .{
            .values = device_core,
            .gate = try harness.upload(u16, inputs.gate),
            .weight = try harness.upload(f32, inputs.norm_weight),
            .out = device_normed,
        },
        @intCast(rows),
        shape.v_head,
        inputs.epsilon,
        inputs.activation,
    );

    return try readChain(harness, allocator, device_normed, device_state, value_rows, state_elements);
}

/// Decode chain: one `gdn_decode` per token, which applies the gated norm
/// itself, carrying the recurrent state across steps.
fn decodeChain(
    harness: *Harness,
    allocator: std.mem.Allocator,
    inputs: ChainInputs,
) !ChainResult {
    const shape = inputs.shape;
    shape.check();
    assert(shape.tokens > 0);
    assert(inputs.norm_weight.len == shape.v_head);

    const rows: usize = @as(usize, shape.tokens) * shape.num_v_heads;
    const value_rows: usize = rows * shape.v_head;
    const state_elements: usize = shape.num_v_heads * shape.k_head * shape.v_head;
    const query_width: usize = shape.num_k_heads * shape.k_head;
    const value_width: usize = shape.num_v_heads * shape.v_head;

    const device_state = try harness.zeros(f32, @intCast(state_elements));
    const device_query = try harness.zeros(f32, @intCast(query_width));
    const device_key = try harness.zeros(f32, @intCast(query_width));
    const device_value = try harness.zeros(f32, @intCast(value_width));
    const device_gate = try harness.zeros(u16, @intCast(value_width));
    const device_beta = try harness.zeros(f32, @intCast(shape.num_v_heads));
    const device_decay = try harness.zeros(f32, @intCast(shape.num_v_heads));
    const device_out = try harness.zeros(u16, @intCast(value_width));
    const device_weight = try harness.upload(f32, inputs.norm_weight);

    const normed = try allocator.alloc(f32, value_rows);
    errdefer allocator.free(normed);
    const bits = try allocator.alloc(u16, value_width);
    defer allocator.free(bits);
    const buffers = DecodeTokenBuffers{
        .query = device_query,
        .key = device_key,
        .value = device_value,
        .gate = device_gate,
        .beta = device_beta,
        .decay = device_decay,
        .state = device_state,
        .norm_weight = device_weight,
        .out = device_out,
    };
    const single = ChainInputs{
        .delta = inputs.delta,
        .gate = inputs.gate,
        .norm_weight = inputs.norm_weight,
        .shape = .{
            .tokens = 1,
            .num_k_heads = shape.num_k_heads,
            .num_v_heads = shape.num_v_heads,
            .k_head = shape.k_head,
            .v_head = shape.v_head,
        },
        .epsilon = inputs.epsilon,
        .activation = inputs.activation,
    };
    for (0..shape.tokens) |token| {
        const value_row = token * value_width;
        try writeDecodeToken(harness, buffers, inputs, token);
        try runDecodeStep(harness, buffers, single, bits);
        for (bits, 0..) |element, index| {
            normed[value_row + index] = bf16Value(element);
        }
    }

    const state = try allocator.alloc(f32, state_elements);
    errdefer allocator.free(state);
    try harness.download(f32, device_state, state);
    assert(normed.len == value_rows);
    return .{ .normed = normed, .state = state };
}

/// The device buffers one decode step reads and writes.
const DecodeTokenBuffers = struct {
    query: cuda_driver.DevicePtr,
    key: cuda_driver.DevicePtr,
    value: cuda_driver.DevicePtr,
    gate: cuda_driver.DevicePtr,
    beta: cuda_driver.DevicePtr,
    decay: cuda_driver.DevicePtr,
    state: cuda_driver.DevicePtr,
    norm_weight: cuda_driver.DevicePtr,
    out: cuda_driver.DevicePtr,

    fn tensors(buffers: DecodeTokenBuffers) DecodeTensors {
        assert(buffers.state != 0);
        assert(buffers.out != 0);

        return .{
            .query = buffers.query,
            .key = buffers.key,
            .value = buffers.value,
            .beta = buffers.beta,
            .decay = buffers.decay,
            .state = buffers.state,
            .gate = buffers.gate,
            .norm_weight = buffers.norm_weight,
            .out = buffers.out,
        };
    }
};

/// Launches one decode step and reads its output row back.
fn runDecodeStep(
    harness: *Harness,
    buffers: DecodeTokenBuffers,
    inputs: ChainInputs,
    bits: []u16,
) !void {
    assert(bits.len == @as(usize, inputs.shape.num_v_heads) * inputs.shape.v_head);
    assert(inputs.shape.tokens == 1);

    try launchDecode(
        &harness.kernels,
        harness.stream.handle,
        buffers.tensors(),
        inputs.shape,
        inputs.epsilon,
        inputs.activation,
    );
    try harness.download(u16, buffers.out, bits);
}

/// Writes one token's delta-rule and gate inputs into the decode buffers.
fn writeDecodeToken(
    harness: *Harness,
    buffers: DecodeTokenBuffers,
    inputs: ChainInputs,
    token: usize,
) !void {
    assert(token < inputs.shape.tokens);
    assert(inputs.shape.k_head > 0);

    const shape = inputs.shape;
    const query_width: usize = shape.num_k_heads * shape.k_head;
    const value_width: usize = shape.num_v_heads * shape.v_head;
    const value_row = token * value_width;
    const head_row = token * shape.num_v_heads;
    try harness.write(buffers.query, f32, inputs.delta.query[token * query_width ..][0..query_width]);
    try harness.write(buffers.key, f32, inputs.delta.key[token * query_width ..][0..query_width]);
    try harness.write(buffers.value, f32, inputs.delta.value[value_row..][0..value_width]);
    try harness.write(buffers.gate, u16, inputs.gate[value_row..][0..value_width]);
    try harness.write(buffers.beta, f32, inputs.delta.beta[head_row..][0..shape.num_v_heads]);
    try harness.write(buffers.decay, f32, inputs.delta.decay[head_row..][0..shape.num_v_heads]);
}

/// Reads a chain's output rows and final state back to the host.
fn readChain(
    harness: *Harness,
    allocator: std.mem.Allocator,
    device_normed: cuda_driver.DevicePtr,
    device_state: cuda_driver.DevicePtr,
    value_rows: usize,
    state_elements: usize,
) !ChainResult {
    assert(device_normed != 0);
    assert(device_state != 0);

    const normed = try allocator.alloc(f32, value_rows);
    errdefer allocator.free(normed);
    const bits = try allocator.alloc(u16, value_rows);
    defer allocator.free(bits);
    const state = try allocator.alloc(f32, state_elements);
    errdefer allocator.free(state);
    try harness.download(u16, device_normed, bits);
    for (bits, 0..) |element, index| normed[index] = bf16Value(element);
    try harness.download(f32, device_state, state);
    assert(state.len == state_elements);
    return .{ .normed = normed, .state = state };
}

/// Rounds the reference's gate projection (`z`) into the kernels' dtype.
fn bf16GateRows(
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    gate: []const f32,
    tokens: u32,
) ![]u16 {
    assert(tokens > 0);
    assert(gate.len >= @as(usize, tokens) * config.value_dim());

    const elements: usize = @as(usize, tokens) * config.value_dim();
    const rounded = try allocator.alloc(u16, elements);
    for (0..elements) |index| {
        rounded[index] = bf16Bits(gate[index]);
    }
    assert(rounded.len == elements);
    return rounded;
}

test "one gdn_prefill and repeated gdn_decode agree at every position" {
    const allocator = std.testing.allocator;
    var harness = try Harness.init();
    defer harness.deinit();

    const longest = tested_lengths[tested_lengths.len - 1];
    const config = try loadConfig(allocator, longest);
    var tensors = try fixtures.open_checkpoint(allocator);
    defer tensors.close(allocator);
    const weights = try loadGdnWeights(&tensors, 0);
    var state = try ref_gdn.State.init(allocator, &config);
    defer state.deinit(allocator);
    var scratch = try ref_gdn.Scratch.init(allocator, &config);
    defer scratch.deinit(allocator);
    const input = try allocator.alloc(f32, @as(usize, longest) * config.hidden_size);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, @as(usize, longest) * config.hidden_size);
    defer allocator.free(output);

    std.debug.print(
        "prefill vs S decode steps (f32 q/k/v, bf16 store, fp32 state)\n",
        .{},
    );
    for (tested_lengths) |tokens| {
        const conv_elements: usize = @as(usize, tokens) * config.conv_dim();
        const rows: usize = @as(usize, tokens) * config.hidden_size;
        fillInput(input[0..rows], 7 + tokens);
        state.reset();
        ref_gdn.forward(
            input[0..rows],
            tokens,
            output[0..rows],
            &weights,
            &state,
            &scratch,
            &config,
            false,
        );

        const sequence = try allocator.alloc(u16, conv_elements);
        defer allocator.free(sequence);
        for (0..conv_elements) |index| {
            sequence[index] = bf16Bits(scratch.mixed_qkv[index]);
        }
        const conv = try checkConvRing(
            &harness,
            allocator,
            &config,
            &weights,
            sequence,
            tokens,
        );
        defer conv.deinit(allocator);
        try checkDeltaInvariant(
            &harness,
            allocator,
            &config,
            &weights,
            &scratch,
            conv.output,
            tokens,
        );
        try harness.reset();
    }
}

/// Runs the conv as one prefill and as one launch per token and requires both
/// to agree bit for bit, both in the output and in the history ring.
///
/// Returns the prefill conv result: that is the sequence the delta-rule chains
/// below consume, so the invariant test runs on the kernel's own conv output
/// rather than on a host recomputation of it.
fn checkConvRing(
    harness: *Harness,
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    weights: *const ref_gdn.Weights,
    sequence: []const u16,
    tokens: u32,
) !ConvResult {
    assert(tokens > 0);
    assert(sequence.len == @as(usize, tokens) * config.conv_dim());

    const one = try convSequence(harness, allocator, config, weights.conv, sequence, tokens);
    errdefer one.deinit(allocator);
    const many = try convPerToken(harness, allocator, config, weights.conv, sequence, tokens);
    defer many.deinit(allocator);

    for (one.output, many.output) |expected, actual| {
        try std.testing.expectEqual(expected, actual);
    }
    for (one.ring, many.ring) |expected, actual| {
        try std.testing.expectEqual(expected, actual);
    }
    std.debug.print(
        "  conv    S={d:<5} prefill and {d} decode steps identical, ring identical\n",
        .{ tokens, tokens },
    );
    return one;
}

/// Runs both delta-rule chains on identical inputs, compares them with each
/// other, and gates each against the oracle's matching scan — `gdn_prefill`
/// against the chunked form (in `checkPrefill`), `gdn_decode` against the
/// recurrent one below.
///
/// Consumes `scratch` mutably: the oracle's recurrent run overwrites
/// `scratch.core_out` (with the decoded sequence's rule output), so callers must
/// not read the chunked result out of it afterwards.
fn checkDeltaInvariant(
    harness: *Harness,
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    weights: *const ref_gdn.Weights,
    scratch: *ref_gdn.Scratch,
    conv_out: []const u16,
    tokens: u32,
) !void {
    assert(tokens > 0);
    assert(scratch.beta.len >= @as(usize, tokens) * config.linear_num_value_heads);

    const shape = deltaShape(config, tokens);
    const delta = try prepareDeltaInputs(
        allocator,
        config,
        conv_out,
        scratch.beta,
        scratch.decay,
        tokens,
    );
    defer delta.deinit(allocator);
    const gate = try bf16GateRows(allocator, config, scratch.z, tokens);
    defer allocator.free(gate);

    const inputs = ChainInputs{
        .delta = &delta,
        .gate = gate,
        .norm_weight = weights.norm,
        .shape = shape,
        .epsilon = config.rms_norm_eps,
        .activation = @intFromEnum(config.output_gate_type),
    };
    const prefill = try prefillChain(harness, allocator, inputs);
    defer prefill.deinit(allocator);
    const decode = try decodeChain(harness, allocator, inputs);
    defer decode.deinit(allocator);

    // Per position, because a chunk-boundary bug shows up as one bad row.
    const stride: usize = @as(usize, shape.num_v_heads) * shape.v_head;
    var worst_position: u32 = 0;
    var worst: f64 = 0.0;
    for (0..tokens) |token| {
        const result = compare(
            prefill.normed[token * stride ..][0..stride],
            decode.normed[token * stride ..][0..stride],
        );
        if (result.scaled > worst) {
            worst = result.scaled;
            worst_position = @intCast(token);
        }
    }
    const overall = compare(prefill.normed, decode.normed);
    const final_state = compare(prefill.state, decode.state);
    std.debug.print(
        "  output  S={d:<5} worst position {d:<5} scaled {e:.3}  (all {e:.3})\n",
        .{ tokens, worst_position, worst, overall.scaled },
    );
    std.debug.print(
        "  state   S={d:<5} scaled {e:.3}\n",
        .{ tokens, final_state.scaled },
    );
    try std.testing.expect(worst <= value_tolerance);
    try std.testing.expect(final_state.scaled <= state_tolerance);
    try checkDecode(allocator, config, weights, scratch, conv_out, &decode, tokens);
}

/// `gdn_decode` against the oracle's recurrent rule, on the same bf16-rounded
/// inputs the kernel reads. The contract names the decode kernel at
/// `value_tolerance`; the prefill-vs-decode rows above only bound it by two
/// tolerances together.
///
/// Overwrites `scratch.core_out` with the decoded sequence's rule output.
fn checkDecode(
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    weights: *const ref_gdn.Weights,
    scratch: *ref_gdn.Scratch,
    conv_out: []const u16,
    decode: *const ChainResult,
    tokens: u32,
) !void {
    assert(tokens > 0);
    assert(decode.normed.len > 0);

    const shape = deltaShape(config, tokens);
    const state_elements: usize = shape.num_v_heads * shape.k_head * shape.v_head;
    const oracle_state = try allocator.alloc(f32, state_elements);
    defer allocator.free(oracle_state);
    oracleDeltaRule(scratch, config, oracle_state, conv_out, tokens, true);
    const expected = try normedExpectation(
        allocator,
        config,
        weights,
        scratch.core_out,
        scratch.z,
        tokens,
    );
    defer allocator.free(expected);
    std.debug.print("  decode vs the oracle's recurrent rule\n", .{});
    try expectWithin(
        "decode",
        tokens,
        compare(expected, decode.normed),
        value_tolerance,
    );
    try expectWithin(
        "dec-state",
        tokens,
        compare(oracle_state, decode.state),
        state_tolerance,
    );
}

/// Compares the whole chain (`gdn_prefill` then `gdn_gate_norm`) against the
/// oracle's `normed`, which is what the layer's out_projection consumes.
///
/// This row is the well-conditioned one: `RMSNormGated` is scale invariant, so
/// the comparison is insensitive to the magnitude of `core_attn_out` (about
/// 1e-4 for the tiny fixture) in a way that comparing `core_out` alone is not.
fn checkLayer(
    harness: *Harness,
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    weights: *const ref_gdn.Weights,
    scratch: *const ref_gdn.Scratch,
    delta: *const DeltaInputs,
    tokens: u32,
) !void {
    const shape = deltaShape(config, tokens);
    const rows: usize = @as(usize, tokens) * shape.num_v_heads;
    const elements: usize = rows * shape.v_head;
    assert(scratch.normed.len >= elements);

    const gate = try bf16GateRows(allocator, config, scratch.z, tokens);
    defer allocator.free(gate);
    const inputs = ChainInputs{
        .delta = delta,
        .gate = gate,
        .norm_weight = weights.norm,
        .shape = shape,
        .epsilon = config.rms_norm_eps,
        .activation = @intFromEnum(config.output_gate_type),
    };
    const chain = try prefillChain(harness, allocator, inputs);
    defer chain.deinit(allocator);

    const expected = try normedExpectation(
        allocator,
        config,
        weights,
        scratch.core_out,
        scratch.z,
        tokens,
    );
    defer allocator.free(expected);
    try expectWithin(
        "chain",
        tokens,
        compare(expected, chain.normed),
        value_tolerance,
    );
    // Informational only, never a gate: `scratch.normed` is the oracle's own
    // end-to-end result from f32 (unquantised) activations, so this row carries
    // the whole engine dtype policy, not the kernel.
    report(
        "chain-f32",
        tokens,
        compare(scratch.normed[0..elements], chain.normed),
    );
}

/// The oracle's `RMSNormGated` over the rows it is given, on the bf16-rounded
/// inputs the kernel reads. `values` is the fp32 `core_attn_out`, `gate` the
/// fp32 gate projection; both are rounded per element, which is the engine's
/// activation dtype and not kernel error.
fn normedExpectation(
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    weights: *const ref_gdn.Weights,
    values: []const f32,
    gate: []const f32,
    tokens: u32,
) ![]f32 {
    const width: usize = config.linear_value_head_dim;
    const rows: usize = @as(usize, tokens) * config.linear_num_value_heads;
    assert(values.len >= rows * width);
    assert(gate.len >= rows * width);
    assert(weights.norm.len == width);

    const expected = try allocator.alloc(f32, rows * width);
    errdefer allocator.free(expected);
    const row_values = try allocator.alloc(f32, width);
    defer allocator.free(row_values);
    const row_gate = try allocator.alloc(f32, width);
    defer allocator.free(row_gate);
    for (0..rows) |row| {
        for (0..width) |dim| {
            row_values[dim] = bf16Value(bf16Bits(values[row * width + dim]));
            row_gate[dim] = bf16Value(bf16Bits(gate[row * width + dim]));
        }
        ref_ops.rmsnorm_gated(
            row_values,
            row_gate,
            weights.norm,
            expected[row * width ..][0..width],
            config.rms_norm_eps,
            config.output_gate_type,
        );
    }
    assert(expected.len == rows * width);
    return expected;
}

/// Re-runs the oracle's delta rule on exactly the delta-rule inputs the kernel
/// reads: the same bf16-rounded conv output (which the oracle then
/// l2-normalises itself), the same gates, and a zeroed recurrent state.
/// `recurrent_form` selects the decode-time rule, which is the only oracle path
/// `gdn_decode` can be gated against at all — comparing it against
/// `gdn_prefill` bounds it by two tolerances, not one.
///
/// This is the same-input comparison the review gate asks for: the engine's
/// activation dtype is on both sides, so what is left to disagree is the scan.
fn oracleDeltaRule(
    scratch: *ref_gdn.Scratch,
    config: *const fixtures.RefConfig,
    recurrent: []f32,
    conv_out: []const u16,
    tokens: u32,
    recurrent_form: bool,
) void {
    assert(tokens > 0);
    assert(conv_out.len == @as(usize, tokens) * config.conv_dim());

    const conv_dim: usize = config.conv_dim();
    const key_dim: usize = config.key_dim();
    const value_dim: usize = config.value_dim();
    for (0..tokens) |t| {
        for (0..key_dim) |lane| {
            const source = bf16Value(conv_out[t * conv_dim + lane]);
            scratch.query[t * key_dim + lane] = source;
            scratch.key[t * key_dim + lane] =
                bf16Value(conv_out[t * conv_dim + key_dim + lane]);
        }
        for (0..value_dim) |lane| {
            scratch.value[t * value_dim + lane] =
                bf16Value(conv_out[t * conv_dim + 2 * key_dim + lane]);
        }
    }
    @memset(recurrent, 0.0);
    if (recurrent_form) {
        ref_gdn.gated_delta_rule_recurrent(tokens, scratch, config, recurrent);
    } else {
        ref_gdn.gated_delta_rule_chunked(tokens, scratch, config, recurrent);
    }
    assert(recurrent.len > 0);
}

/// Hidden size of the synthetic layer the production-shape tests build.
///
/// Only the delta rule's dimensions are the model's; the projections are the
/// subject of other issues, so a 256-wide hidden keeps the oracle cheap without
/// changing a single dimension the four kernels see.
const production_hidden: u32 = 256;

/// Cursor over a synthetic weight buffer.
const Filler = struct {
    storage: []f32,
    cursor: usize,
    seed: u32,

    /// Takes `count` values with magnitude `scale` and advances the cursor.
    fn take(filler: *Filler, count: usize, scale: f32) []f32 {
        assert(count > 0);
        assert(filler.cursor + count <= filler.storage.len);

        const slice = filler.storage[filler.cursor..][0..count];
        filler.cursor += count;
        for (slice) |*slot| {
            filler.seed = filler.seed *% 1_103_515_245 +% 12_345;
            slot.* = (@as(f32, @floatFromInt(filler.seed % 2000)) / 1000.0 - 1.0) * scale;
        }
        assert(slice.len == count);
        return slice;
    }
};

/// A synthetic linear-attention layer at the model's delta-rule dimensions.
const ProductionLayer = struct {
    storage: []f32,
    weights: ref_gdn.Weights,
    config: fixtures.RefConfig,

    /// Builds the config and the weights of one layer.
    fn init(allocator: std.mem.Allocator, tokens_max: u32) !ProductionLayer {
        assert(tokens_max > 0);
        assert(production_hidden > 0);

        const config = productionConfig(tokens_max);
        const conv_dim: usize = config.conv_dim();
        const value_dim: usize = config.value_dim();
        const heads: usize = config.linear_num_value_heads;
        const hidden: usize = config.hidden_size;
        const total = conv_dim * hidden +
            value_dim * hidden +
            2 * heads * hidden +
            conv_dim * config.linear_conv_kernel_dim +
            config.linear_value_head_dim +
            hidden * value_dim +
            2 * heads;
        const storage = try allocator.alloc(f32, total);
        errdefer allocator.free(storage);

        var filler = Filler{ .storage = storage, .cursor = 0, .seed = 20_260_124 };
        const qkv_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden)));
        const weights = ref_gdn.Weights{
            .in_proj_qkv = filler.take(conv_dim * hidden, qkv_scale),
            .in_proj_z = filler.take(value_dim * hidden, qkv_scale),
            .in_proj_b = filler.take(heads * hidden, qkv_scale),
            .in_proj_a = filler.take(heads * hidden, qkv_scale),
            .conv = filler.take(conv_dim * config.linear_conv_kernel_dim, 0.5),
            .norm = filler.take(config.linear_value_head_dim, 1.0),
            .out_proj = filler.take(hidden * value_dim, qkv_scale),
            .a_log = filler.take(heads, 1.0),
            .dt_bias = filler.take(heads, 1.0),
        };
        assert(filler.cursor == total);
        assert(weights.in_proj_qkv.len == conv_dim * hidden);
        return .{ .storage = storage, .weights = weights, .config = config };
    }

    fn deinit(layer: *const ProductionLayer, allocator: std.mem.Allocator) void {
        assert(layer.storage.len > 0);
        assert(layer.weights.norm.len == 128);

        allocator.free(layer.storage);
    }
};

/// The model's delta-rule geometry with a small hidden size.
fn productionConfig(tokens_max: u32) fixtures.RefConfig {
    assert(tokens_max > 0);
    assert(production_hidden > 0);

    var config = fixtures.RefConfig{};
    config.hidden_size = production_hidden;
    config.num_hidden_layers = 1;
    config.linear_num_key_heads = 16;
    config.linear_num_value_heads = 48;
    config.linear_key_head_dim = 128;
    config.linear_value_head_dim = 128;
    config.linear_conv_kernel_dim = 4;
    config.rms_norm_eps = 1.0e-6;
    config.output_gate_type = .sigmoid;
    config.max_position_embeddings = tokens_max;
    assert(config.conv_dim() == 10240);
    return config;
}

test "gdn matches the oracle at the production head dimensions" {
    const allocator = std.testing.allocator;
    var harness = try Harness.init();
    defer harness.deinit();

    // Three real rows in a 64-row chunk: the padding path at the real head
    // size, where an indexing or decay bug actually shows.
    const tokens: u32 = 3;
    var layer = try ProductionLayer.init(allocator, 8);
    defer layer.deinit(allocator);
    const config = layer.config;
    var state = try ref_gdn.State.init(allocator, &config);
    defer state.deinit(allocator);
    var scratch = try ref_gdn.Scratch.init(allocator, &config);
    defer scratch.deinit(allocator);
    const input = try allocator.alloc(f32, @as(usize, tokens) * config.hidden_size);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, @as(usize, tokens) * config.hidden_size);
    defer allocator.free(output);

    fillInput(input, 991);
    state.reset();
    ref_gdn.forward(input, tokens, output, &layer.weights, &state, &scratch, &config, false);

    std.debug.print(
        "gdn at the model's head dimensions (16x128 key, 48x128 value)\n",
        .{},
    );
    try reportGates(&config, &scratch, tokens);
    try checkConv(&harness, allocator, &config, &layer.weights, &scratch, tokens);
    try harness.reset();
    const conv_elements: usize = @as(usize, tokens) * config.conv_dim();
    const conv_out = try tokenMajorConvOutput(
        allocator,
        &config,
        scratch.conv_out[0..conv_elements],
        tokens,
    );
    defer allocator.free(conv_out);
    const inputs = try prepareDeltaInputs(
        allocator,
        &config,
        conv_out,
        scratch.beta,
        scratch.decay,
        tokens,
    );
    defer inputs.deinit(allocator);
    oracleDeltaRule(&scratch, &config, state.recurrent, conv_out, tokens, false);
    try checkPrefill(&harness, allocator, &config, &state, &scratch, &inputs, tokens);
    try checkGateNorm(&harness, allocator, &config, &layer.weights, &scratch, tokens);
    try checkLayer(&harness, allocator, &config, &layer.weights, &scratch, &inputs, tokens);
    try harness.reset();
    try checkDeltaInvariant(
        &harness,
        allocator,
        &config,
        &layer.weights,
        &scratch,
        conv_out,
        tokens,
    );
    try harness.reset();
}

test "gdn matches the HF golden trace at twelve tokens" {
    const allocator = std.testing.allocator;
    var harness = try Harness.init();
    defer harness.deinit();

    // Twelve tokens is one chunk: this ties the kernels to the trace the model
    // was exported with, independently of the fp32 oracle.
    const tokens: u32 = 12;
    const config = try loadConfig(allocator, tokens);
    var checkpoint = try fixtures.open_checkpoint(allocator);
    defer checkpoint.close(allocator);
    var trace = try fixtures.open_trace(allocator);
    defer trace.close(allocator);
    const weights = try loadGdnWeights(&checkpoint, 0);

    const raw = RawDelta{
        .query = (try trace.tensor("layer0/gdn/q")).values,
        .key = (try trace.tensor("layer0/gdn/k")).values,
        .value = (try trace.tensor("layer0/gdn/v")).values,
    };
    const beta = (try trace.tensor("layer0/gdn/beta")).values;
    const decay = (try trace.tensor("layer0/gdn/g")).values;
    const gate = try bf16GateRows(allocator, &config, (try trace.tensor("layer0/gdn/z")).values, tokens);
    defer allocator.free(gate);
    const expected = (try trace.tensor("layer0/gdn/normed")).values;
    const inputs = try normaliseDelta(allocator, &config, raw, beta, decay, tokens);
    defer inputs.deinit(allocator);

    const chain = try prefillChain(
        &harness,
        allocator,
        .{
            .delta = &inputs,
            .gate = gate,
            .norm_weight = weights.norm,
            .shape = deltaShape(&config, tokens),
            .epsilon = config.rms_norm_eps,
            .activation = @intFromEnum(config.output_gate_type),
        },
    );
    defer chain.deinit(allocator);

    // The golden is f32 end to end: every float tensor in
    // `tests/golden/tiny/trace.safetensors` is F32 (`layer0/gdn/normed`
    // included), so HF's `hidden_states.to(input_dtype)` inside
    // `RMSNormGated.forward` is a no-op, and `zig build refcheck` pins the fp32
    // oracle to this trace at 4.2e-7 scaled. The kernel, however, reads a bf16
    // gate `z` and stores its output in bf16, so comparing `normed` with the
    // kernel output directly would charge the engine's dtype policy to the
    // kernel: that is the 3.95e-3 to 5.35e-3 the `chain-f32` rows report.
    //
    // The gate below applies that policy to the expectation instead: it runs the
    // reference's own `RMSNormGated` over the golden's `core_out` and `z`
    // rounded to bf16 (the dtype the kernel reads) and puts the result through
    // the same bf16 store, so the only thing left to disagree is the f32
    // reduction order inside the two norms. `value_tolerance` is the contract
    // row's bound; the raw golden is reported underneath, never gated.
    std.debug.print("gdn vs the HF golden trace (layer 0, 12 tokens)\n", .{});
    const policy = try normedExpectation(
        allocator,
        &config,
        &weights,
        (try trace.tensor("layer0/gdn/core_out")).values,
        (try trace.tensor("layer0/gdn/z")).values,
        tokens,
    );
    defer allocator.free(policy);
    const stored = try allocator.alloc(f32, policy.len);
    defer allocator.free(stored);
    for (policy, stored) |value, *slot| {
        slot.* = bf16Value(bf16Bits(value));
    }
    try expectWithin(
        "hf-golden",
        tokens,
        compare(stored, chain.normed),
        value_tolerance,
    );
    report("hf-f32", tokens, compare(expected, chain.normed));
}

/// The delta-rule inputs and gate rows the oracle's `forward` implies for one
/// input sequence.
const PreparedCase = struct {
    delta: DeltaInputs,
    gate: []u16,

    fn deinit(prepared: *const PreparedCase, allocator: std.mem.Allocator) void {
        assert(prepared.gate.len > 0);
        assert(prepared.delta.query.len > 0);

        allocator.free(prepared.gate);
        prepared.delta.deinit(allocator);
    }
};

/// Runs the oracle's `forward` over `input` and derives the kernel's inputs from
// the conv output it produced.
fn prepareCase(
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
    weights: *const ref_gdn.Weights,
    state: *ref_gdn.State,
    scratch: *ref_gdn.Scratch,
    input: []const f32,
    output: []f32,
    tokens: u32,
    seed: u32,
) !PreparedCase {
    assert(tokens > 0);
    assert(input.len == @as(usize, tokens) * config.hidden_size);

    fillInput(@constCast(input), seed);
    state.reset();
    ref_gdn.forward(input, tokens, output, weights, state, scratch, config, false);

    const conv_elements: usize = @as(usize, tokens) * config.conv_dim();
    const conv_out = try tokenMajorConvOutput(
        allocator,
        config,
        scratch.conv_out[0..conv_elements],
        tokens,
    );
    defer allocator.free(conv_out);
    const delta = try prepareDeltaInputs(
        allocator,
        config,
        conv_out,
        scratch.beta,
        scratch.decay,
        tokens,
    );
    errdefer delta.deinit(allocator);
    const gate = try bf16GateRows(allocator, config, scratch.z, tokens);
    return .{ .delta = delta, .gate = gate };
}

test "gdn_prefill output depends on the sequence it is given" {
    const allocator = std.testing.allocator;
    var harness = try Harness.init();
    defer harness.deinit();

    // Non-vacuity: two different sequences through the same weights must give
    // clearly different outputs, or the suite is comparing constants.
    const tokens: u32 = 64;
    const config = try loadConfig(allocator, tokens);
    var tensors = try fixtures.open_checkpoint(allocator);
    defer tensors.close(allocator);
    const weights = try loadGdnWeights(&tensors, 0);
    var state = try ref_gdn.State.init(allocator, &config);
    defer state.deinit(allocator);
    var scratch = try ref_gdn.Scratch.init(allocator, &config);
    defer scratch.deinit(allocator);
    const rows: usize = @as(usize, tokens) * config.hidden_size;
    const input = try allocator.alloc(f32, rows);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, rows);
    defer allocator.free(output);
    const shape = deltaShape(&config, tokens);

    var difference: f64 = 0.0;
    var previous: ?ChainResult = null;
    defer if (previous) |chain| chain.deinit(allocator);
    for (0..2) |step| {
        const prepared = try prepareCase(
            allocator,
            &config,
            &weights,
            &state,
            &scratch,
            input,
            output,
            tokens,
            31 + @as(u32, @intCast(step)),
        );
        defer prepared.deinit(allocator);
        const chain = try prefillChain(
            &harness,
            allocator,
            .{
                .delta = &prepared.delta,
                .gate = prepared.gate,
                .norm_weight = weights.norm,
                .shape = shape,
                .epsilon = config.rms_norm_eps,
                .activation = @intFromEnum(config.output_gate_type),
            },
        );
        if (previous) |earlier| {
            const comparison = compare(earlier.normed, chain.normed);
            difference = comparison.scaled;
            std.debug.print(
                "  two inputs  S={d:<5} differ by scaled {e:.3}  abs {e:.3}\n",
                .{ tokens, comparison.scaled, comparison.absolute },
            );
            chain.deinit(allocator);
        } else {
            previous = chain;
        }
    }
    // Far above any rounding: the outputs are genuinely input dependent.
    try std.testing.expect(difference > 0.1);
}

// Bounds check of the padded chunk.
//
// A sequence that is not a multiple of the chunk leaves the tail rows of the
// last chunk as padding, and `gdn_prefill`'s UT-transform dot-product loop
// walks all 64 rows of a chunk whether they hold tokens or not. Every load of
// `query`, `key` and `value` there is behind a `row_valid` / `other_valid`
// test, so the kernel must never address element `tokens` of a sequence
// tensor. This is the only test that can see a regression of that guard:
// padding rows are discarded from the output and skipped by the state update,
// so an unguarded load changes no number the oracle can compare.
//
// `key` is therefore given a device allocation of its own whose last byte is
// the tensor's last byte, so a load past the tensor is a load past the
// allocation, which `compute-sanitizer --tool memcheck` reports as an
// out-of-bounds read. There is no per-kernel build step, so the tool is pointed
// at the unit-test binary the `test` step runs (`zig build test --verbose`
// prints its path). Without the `row_valid` guard that run reports
// `Invalid __global__ read of size 4` at the unguarded load; with it, the whole
// suite reports `ERROR SUMMARY: 0 errors`.
//
// The assertions below are what a plain `zig build test` sees on its own: the
// launch must not raise an illegal-address error, and the 56 real rows must
// come back finite and non-zero. A driver that maps the page after the
// allocation swallows the over-read, which is why the tool is the proof.
test "gdn_prefill over a padded chunk reads nothing past the key tensor" {
    const allocator = std.testing.allocator;
    var harness = try Harness.init();
    defer harness.deinit();

    // 56 tokens fill 56 rows of the single chunk; rows 56..63 are padding.
    const tokens: u32 = 56;
    const config = try loadConfig(allocator, tokens);
    const shape = deltaShape(&config, tokens);
    const key_elements: usize = @as(usize, tokens) * shape.num_k_heads * shape.k_head;
    const value_elements: usize = @as(usize, tokens) * shape.num_v_heads * shape.v_head;
    const state_elements: usize =
        @as(usize, shape.num_v_heads) * shape.k_head * shape.v_head;
    assert(key_elements > 0);
    assert(shape.tokens % chunk_size != 0);

    const query = try allocator.alloc(f32, key_elements);
    defer allocator.free(query);
    const key = try allocator.alloc(f32, key_elements);
    defer allocator.free(key);
    const value = try allocator.alloc(f32, value_elements);
    defer allocator.free(value);
    const beta = try allocator.alloc(f32, @as(usize, tokens) * shape.num_v_heads);
    defer allocator.free(beta);
    const decay = try allocator.alloc(f32, @as(usize, tokens) * shape.num_v_heads);
    defer allocator.free(decay);
    fillInput(query, 101);
    fillInput(key, 211);
    fillInput(value, 307);
    // `beta` in (0, 1) and a decay well away from zero: the substitution and the
    // chunk decay both do work, so the padding rows reach the guarded loads with
    // live numbers rather than with zeros that would hide an over-read.
    for (beta) |*slot| slot.* = 0.5;
    for (decay) |*slot| slot.* = -0.05;

    const device_query = try harness.upload(f32, query);
    const device_value = try harness.upload(f32, value);
    const device_beta = try harness.upload(f32, beta);
    const device_decay = try harness.upload(f32, decay);
    const device_state = try harness.zeros(f32, @intCast(state_elements));
    const device_out = try harness.zeros(u16, @intCast(value_elements));

    // The one allocation whose end is a tensor's end: `key` sits in the last
    // `key_bytes` of a page-aligned buffer, so element `tokens` of the tensor is
    // the first byte past the allocation.
    const key_bytes: u64 = @as(u64, key_elements) * @sizeOf(f32);
    const key_buffer = try cuda_context.DeviceBuffer.allocate(
        std.mem.alignForward(u64, key_bytes, 4096),
    );
    defer key_buffer.free();
    assert(key_buffer.bytes >= key_bytes);
    const device_key = key_buffer.pointer + (key_buffer.bytes - key_bytes);
    assert(device_key % @sizeOf(f32) == 0);
    try harness.write(device_key, f32, key);

    try launchPrefill(
        &harness.kernels,
        harness.stream.handle,
        .{
            .query = device_query,
            .key = device_key,
            .value = device_value,
            .beta = device_beta,
            .decay = device_decay,
            .state = device_state,
            .out = device_out,
        },
        shape,
    );
    // A load past the `key` allocation that the driver faults on fails here,
    // as an illegal-address error, instead of being absorbed by the slack of an
    // arena the way an in-arena over-read would be.
    try harness.stream.synchronize();

    const bits = try allocator.alloc(u16, value_elements);
    defer allocator.free(bits);
    try harness.download(u16, device_out, bits);
    var live: u32 = 0;
    for (bits) |element| {
        try std.testing.expect(std.math.isFinite(bf16Value(element)));
        if (element != 0) {
            live += 1;
        }
    }
    // The 56 real rows were written by the kernel, so the launch did the work
    // the bound above is meant to cover.
    try std.testing.expect(live > 0);

    const actual_state = try allocator.alloc(f32, state_elements);
    defer allocator.free(actual_state);
    try harness.download(f32, device_state, actual_state);
    for (actual_state) |element| {
        try std.testing.expect(std.math.isFinite(element));
    }
    std.debug.print(
        "  padded key  S={d:<5} {d} padding rows, {d} live outputs, no over-read\n",
        .{ tokens, chunk_size - (shape.tokens % chunk_size), live },
    );
    try harness.reset();
}
