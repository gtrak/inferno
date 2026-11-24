//! Launch wrappers and the acceptance suite for the elementwise kernels.
//!
//! `core/src/kernels/elementwise.cu` holds the twelve kernels; this file owns
//! their launch geometry, the dynamic shared memory the hyper-connection block
//! needs, and the tests that compare them against the fp32 CPU oracle in
//! `core/src/ref/` with identical inputs.
//!
//! Shapes, all token-major:
//!
//!   rmsnorm   x, out        `[rows][columns]`, grouped by `group` columns
//!             weight        `[columns]`, applied as `1 + w`
//!   gated     x, z, out     `[rows][columns]`; whole-row norm, times `act(z)`
//!   rope      positions     three `[tokens]` int32 grids (temporal, H, W)
//!             frequencies   `[rotated / 2]` f32, from `rope.inv_freq_init`
//!             cos, sin      `[tokens][rotated]` f32
//!             q, k          `[tokens][heads][head_dim]` f32, rotated in place
//!   hyper     input         `[tokens][streams * hidden]` f32
//!             mixed         `[tokens][hidden]` f32
//!             injection     `[tokens][streams]` f32, absent for the mixer
//!   sampler   logits        `[vocab]` f32, plus `[history_len]` int32 ids
//!
//! The sampler's retained-set definition is documented in `elementwise.cu` and
//! re-derived independently in this file's tests, because a sampler has no CPU
//! oracle in `ref/`: the definition is the specification, and the test checks
//! the kernel against a host implementation of the same definition.

const std = @import("std");

const cuda_context = @import("../hw/api.zig").cuda_context;
const cuda_driver = @import("../hw/api.zig").cuda_driver;
const kernel = @import("../hw/kernel.zig");
const generated_cubins = @import("generated_cubins");

const fixtures = @import("../ref/fixtures.zig");
const ref_hc = @import("../ref/hc.zig");
const ref_ops = @import("../ref/ops.zig");
const ref_rope = @import("../ref/rope.zig");

const assert = std.debug.assert;

/// Name `core/build.zig` gives the cubin compiled from `elementwise.cu`.
const cubin_name = "elementwise";

/// Kernel names, which are the `extern "C"` symbols inside the cubin.
const name_rmsnorm = "ew_rmsnorm_grouped";
const name_rmsnorm_bf16 = "ew_rmsnorm_grouped_bf16";
const name_rmsnorm_gated = "ew_rmsnorm_gated";
const name_rope_build = "ew_rope_build_tables";
const name_rope_apply_qk = "ew_rope_apply_qk";
const name_rope_apply_q = "ew_rope_apply_q";
const name_hyper_mix = "ew_hyper_mix";
const name_silu_mul = "ew_silu_mul";
const name_sigmoid_gate = "ew_sigmoid_gate";
const name_softmax = "ew_softmax_rows";
const name_argmax = "ew_argmax";
const name_sampler = "ew_sampler";

/// Threads per block, mirrored from `elementwise.cu`.
const norm_threads: u32 = 256;

/// Threads per block of the RoPE kernels: one head rotates at a time.
const rope_threads: u32 = 32;

/// Threads per block of the sampler; the whole vocabulary belongs to one block.
const sampler_threads: u32 = 256;

/// Floats a block reduction needs: one partial per warp of a 256-thread block,
/// plus the total.
const reduction_scratch: u32 = 9;

/// Dynamic shared memory one block may opt into on this GPU: 99 KiB, read from
/// `cudaDeviceProp::sharedMemPerBlockOptin`. The hyper-connection block asks
/// for 83 KiB at the production shape, so the limit is raised at load time and
/// a layout change that overflows fails there rather than at a launch.
const dynamic_shared_bytes_max: u32 = 99 * 1024;

/// Largest column count the norm kernels are compiled for: the HC stream is
/// `hc_count * hidden_size` = 10240, and nothing else in the model is wider.
const norm_columns_max: u32 = 16384;

/// Largest group the norm kernels handle: one 2560-dim hidden vector.
const norm_group_max: u32 = 8192;

/// Largest `hc_lowrank` the hyper-connection block is compiled for.
const hyper_lowrank_max: u32 = 1024;

/// The `activation` code the kernels take, from `ops.Activation`'s order.
fn activationCode(activation: ref_ops.Activation) i32 {
    assert(activation == activation);
    assert(@intFromEnum(activation) <= 2);

    return switch (activation) {
        .none => 0,
        .silu => 1,
        .sigmoid => 2,
    };
}

test "the kernel activation codes follow the reference enum order" {
    // The CUDA side hard-codes 0/1/2 in that order, so a reordering of
    // `ops.Activation` has to fail here rather than silently applying the wrong
    // gate.
    try std.testing.expectEqual(@as(i32, 0), activationCode(.none));
    try std.testing.expectEqual(@as(i32, 1), activationCode(.silu));
    try std.testing.expectEqual(@as(i32, 2), activationCode(.sigmoid));
    try std.testing.expectEqual(
        @as(u32, 1),
        @intFromEnum(ref_ops.Activation.silu),
    );
    try std.testing.expectEqual(
        @as(u32, 2),
        @intFromEnum(ref_ops.Activation.sigmoid),
    );
}

/// Shape of one grouped RMSNorm: `rows` rows of `columns`, normalised in groups
/// of `group` columns (the whole row when `group == columns`).
pub const NormShape = struct {
    rows: u32,
    columns: u32,
    group: u32,

    /// Rejects the shapes the kernels cannot express.
    pub fn check(self: NormShape) void {
        assert(self.rows > 0);
        assert(self.columns > 0);
        assert(self.columns <= norm_columns_max);
        assert(self.group > 0);
        assert(self.group <= norm_group_max);
        assert(self.columns % self.group == 0);
    }

    /// Blocks one launch needs: one per (row, group).
    fn blocks(self: NormShape) u32 {
        self.check();
        const groups = self.columns / self.group;
        const total = self.rows * groups;
        assert(total > 0);
        return total;
    }
};

/// Device pointers of one `ew_rmsnorm_grouped` launch.
pub const NormTensors = struct {
    x: cuda_driver.DevicePtr,
    weight: cuda_driver.DevicePtr,
    out: cuda_driver.DevicePtr,

    fn check(self: NormTensors) void {
        assert(self.x != 0);
        assert(self.weight != 0);
        assert(self.out != 0);
    }
};

/// Shape of one partial-rotary mRoPE application.
pub const RopeShape = struct {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    /// Dims rotated: `int(head_dim * partial_rotary_factor)` = 64 of 256.
    rotated: u32,
    /// `rope_parameters.mrope_section`: temporal, height, width.
    section: [3]u32,

    /// Rejects the shapes the kernels cannot express.
    pub fn check(self: RopeShape) void {
        assert(self.tokens > 0);
        assert(self.heads > 0);
        assert(self.rotated > 0);
        assert(self.rotated % 2 == 0);
        assert(self.rotated <= self.head_dim);
        assert(self.section[0] > 0);
        // The sections may exceed the frequency count: the reference clips each
        // strided grid at `2 * frequencies.len`, and the indexer's 128-dim head
        // rotates only 32 dims while the checkpoint still declares 11/11/10.
        assert(self.section[1] <= 8 * self.rotated);
        assert(self.section[2] <= 8 * self.rotated);
    }

    /// Frequencies the cos/sin table needs: one per rotated pair.
    fn frequencyCount(self: RopeShape) u32 {
        self.check();
        return self.rotated / 2;
    }
};

/// Device pointers of one `ew_rope_build_tables` launch.
pub const RopeTableTensors = struct {
    positions_t: cuda_driver.DevicePtr,
    positions_h: cuda_driver.DevicePtr,
    positions_w: cuda_driver.DevicePtr,
    frequencies: cuda_driver.DevicePtr,
    cos: cuda_driver.DevicePtr,
    sin: cuda_driver.DevicePtr,

    fn check(self: RopeTableTensors) void {
        assert(self.positions_t != 0);
        assert(self.positions_h != 0);
        assert(self.positions_w != 0);
        assert(self.frequencies != 0);
        assert(self.cos != 0);
        assert(self.sin != 0);
    }
};

/// Device pointers of one rotated-tensor launch.
pub const RopeTensors = struct {
    q: cuda_driver.DevicePtr,
    k: cuda_driver.DevicePtr,
    cos: cuda_driver.DevicePtr,
    sin: cuda_driver.DevicePtr,

    fn check(self: RopeTensors) void {
        assert(self.q != 0);
        assert(self.k != 0);
        assert(self.cos != 0);
        assert(self.sin != 0);
    }
};

/// Shape of one hyper-connection block.
pub const HyperShape = struct {
    tokens: u32,
    hidden: u32,
    streams: u32,
    lowrank: u32,

    /// Rejects the shapes the kernels cannot express.
    pub fn check(self: HyperShape) void {
        assert(self.tokens > 0);
        assert(self.hidden > 0);
        assert(self.streams > 0);
        assert(self.lowrank > 0);
        assert(self.lowrank <= hyper_lowrank_max);
        assert(self.streams * self.hidden <= norm_columns_max);
    }

    /// Width of the stream: `hc_count * hidden_size`.
    fn hyper(self: HyperShape) u32 {
        self.check();
        return self.streams * self.hidden;
    }
};

/// Device pointers of one `ew_hyper_mix` launch. `inject_weight` and
/// `injection` are zero for the final mixer (`use_combine = false`).
pub const HyperTensors = struct {
    input: cuda_driver.DevicePtr,
    norm_weight: cuda_driver.DevicePtr,
    mix_down: cuda_driver.DevicePtr,
    mix_up: cuda_driver.DevicePtr,
    inject_weight: cuda_driver.DevicePtr,
    mixed: cuda_driver.DevicePtr,
    injection: cuda_driver.DevicePtr,

    fn check(self: HyperTensors) void {
        assert(self.input != 0);
        assert(self.norm_weight != 0);
        assert(self.mix_down != 0);
        assert(self.mix_up != 0);
        assert(self.mixed != 0);
        assert((self.inject_weight == 0) == (self.injection == 0));
    }
};

/// Everything the sampler needs beyond the logits of one token.
pub const SampleParams = struct {
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    seed: u32,
    /// Which draw of this sequence this is; part of the RNG state, so the same
    /// `(seed, draw)` always replays the same token.
    draw: u32,

    /// Rejects the parameters the kernel cannot express.
    pub fn check(self: SampleParams) void {
        assert(self.temperature >= 0.0);
        assert(self.temperature <= 100.0);
        assert(self.top_k >= 1);
        assert(self.top_p >= 0.0);
        assert(self.top_p <= 1.0);
        assert(self.min_p >= 0.0);
        assert(self.min_p <= 1.0);
        assert(self.repetition_penalty >= 1.0);
        assert(self.repetition_penalty <= 100.0);
    }
};

/// Device pointers of one `ew_sampler` launch.
pub const SamplerTensors = struct {
    logits: cuda_driver.DevicePtr,
    history: cuda_driver.DevicePtr,
    /// Caller-owned scratch of one f32 per vocabulary entry.
    scaled: cuda_driver.DevicePtr,
    id: cuda_driver.DevicePtr,
    logprob: cuda_driver.DevicePtr,

    fn check(self: SamplerTensors, history_len: u32) void {
        assert(self.logits != 0);
        assert(self.scaled != 0);
        assert(self.id != 0);
        assert(self.logprob != 0);
        if (history_len > 0) {
            assert(self.history != 0);
        } else {
            assert(self.history == 0);
        }
    }
};

/// Every kernel entry point of one loaded `elementwise` module.
pub const Kernels = struct {
    module: kernel.Module,
    rmsnorm: kernel.Function,
    rmsnorm_bf16: kernel.Function,
    rmsnorm_gated: kernel.Function,
    rope_build: kernel.Function,
    rope_apply_qk: kernel.Function,
    rope_apply_q: kernel.Function,
    hyper_mix: kernel.Function,
    silu_mul: kernel.Function,
    sigmoid_gate: kernel.Function,
    softmax: kernel.Function,
    argmax: kernel.Function,
    sampler: kernel.Function,

    /// Loads the cubin `core/build.zig` embedded for `elementwise.cu`.
    ///
    /// The context must be current. The dynamic shared-memory limit is raised
    /// once, to the production hyper-connection shape: the driver treats it as
    /// a per-function maximum, so the smaller fixture shapes stay legal.
    pub fn load() !Kernels {
        const image = cubinImage() orelse return error.ElementwiseCubinMissing;
        const module = try kernel.Module.load(image);
        errdefer module.unload();

        const entry = try loadEntryPoints(module);
        const kernels = Kernels{
            .module = module,
            .rmsnorm = entry.rmsnorm,
            .rmsnorm_bf16 = entry.rmsnorm_bf16,
            .rmsnorm_gated = entry.rmsnorm_gated,
            .rope_build = entry.rope_build,
            .rope_apply_qk = entry.rope_apply_qk,
            .rope_apply_q = entry.rope_apply_q,
            .hyper_mix = entry.hyper_mix,
            .silu_mul = entry.silu_mul,
            .sigmoid_gate = entry.sigmoid_gate,
            .softmax = entry.softmax,
            .argmax = entry.argmax,
            .sampler = entry.sampler,
        };
        try kernels.hyper_mix.setDynamicSharedBytes(maxHyperSharedBytes());
        assert(kernels.module.handle != null);
        return kernels;
    }

    /// Unloads the module; the context must still be current.
    pub fn deinit(kernels: *const Kernels) void {
        assert(kernels.module.handle != null);
        assert(@intFromPtr(kernels.rmsnorm.handle) != 0);

        kernels.module.unload();
    }
};

/// The twelve entry points, looked up together so a partial load can be undone
/// by the caller's `errdefer`.
const EntryPoints = struct {
    rmsnorm: kernel.Function,
    rmsnorm_bf16: kernel.Function,
    rmsnorm_gated: kernel.Function,
    rope_build: kernel.Function,
    rope_apply_qk: kernel.Function,
    rope_apply_q: kernel.Function,
    hyper_mix: kernel.Function,
    silu_mul: kernel.Function,
    sigmoid_gate: kernel.Function,
    softmax: kernel.Function,
    argmax: kernel.Function,
    sampler: kernel.Function,
};

/// Looks every kernel up by its `extern "C"` name.
fn loadEntryPoints(module: kernel.Module) !EntryPoints {
    assert(module.handle != null);
    assert(name_rmsnorm.len > 0);

    return .{
        .rmsnorm = try module.function(name_rmsnorm),
        .rmsnorm_bf16 = try module.function(name_rmsnorm_bf16),
        .rmsnorm_gated = try module.function(name_rmsnorm_gated),
        .rope_build = try module.function(name_rope_build),
        .rope_apply_qk = try module.function(name_rope_apply_qk),
        .rope_apply_q = try module.function(name_rope_apply_q),
        .hyper_mix = try module.function(name_hyper_mix),
        .silu_mul = try module.function(name_silu_mul),
        .sigmoid_gate = try module.function(name_sigmoid_gate),
        .softmax = try module.function(name_softmax),
        .argmax = try module.function(name_argmax),
        .sampler = try module.function(name_sampler),
    };
}

/// The `elementwise` cubin image, or null when the build embedded no such
/// kernel.
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

/// Bytes of dynamic shared memory one norm block needs.
fn normSharedBytes() u32 {
    assert(reduction_scratch > 0);
    assert(@sizeOf(f32) == 4);

    return reduction_scratch * @sizeOf(f32);
}

/// Floats of dynamic shared memory `ew_hyper_mix` needs for `shape`: the
/// normalised stream, the sigmoid mix weights, the low-rank vector and the
/// reduction scratch.
fn hyperSharedBytes(shape: HyperShape) u32 {
    shape.check();
    const floats = 2 * shape.hyper() + shape.lowrank + reduction_scratch;
    const bytes = floats * @sizeOf(f32);
    assert(bytes <= dynamic_shared_bytes_max);
    return bytes;
}

/// Largest `ew_hyper_mix` shared request, for the production shape.
fn maxHyperSharedBytes() u32 {
    assert(hyper_lowrank_max > 0);
    return hyperSharedBytes(.{
        .tokens = 1,
        .hidden = 2560,
        .streams = 4,
        .lowrank = 320,
    });
}

/// Launches `ew_rmsnorm_grouped` over `shape`.
pub fn launchRmsnorm(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    tensors: NormTensors,
    shape: NormShape,
    epsilon: f32,
) kernel.KernelError!void {
    tensors.check();
    shape.check();
    assert(epsilon > 0.0);

    var x = tensors.x;
    var weight = tensors.weight;
    var out = tensors.out;
    var rows: i32 = @intCast(shape.rows);
    var columns: i32 = @intCast(shape.columns);
    var group: i32 = @intCast(shape.group);
    var variance_epsilon = epsilon;
    var parameters = [_]?*anyopaque{
        @ptrCast(&x),
        @ptrCast(&weight),
        @ptrCast(&out),
        @ptrCast(&rows),
        @ptrCast(&columns),
        @ptrCast(&group),
        @ptrCast(&variance_epsilon),
    };
    assert(parameters.len == 7);

    try kernels.rmsnorm.launch(
        .{ shape.blocks(), 1, 1 },
        .{ norm_threads, 1, 1 },
        normSharedBytes(),
        stream,
        parameters[0..],
    );
}

/// Launches `ew_rmsnorm_grouped_bf16` over `shape`.
pub fn launchRmsnormBf16(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    tensors: NormTensors,
    shape: NormShape,
    epsilon: f32,
) kernel.KernelError!void {
    tensors.check();
    shape.check();
    assert(epsilon > 0.0);

    var x = tensors.x;
    var weight = tensors.weight;
    var out = tensors.out;
    var rows: i32 = @intCast(shape.rows);
    var columns: i32 = @intCast(shape.columns);
    var group: i32 = @intCast(shape.group);
    var variance_epsilon = epsilon;
    var parameters = [_]?*anyopaque{
        @ptrCast(&x),
        @ptrCast(&weight),
        @ptrCast(&out),
        @ptrCast(&rows),
        @ptrCast(&columns),
        @ptrCast(&group),
        @ptrCast(&variance_epsilon),
    };
    assert(parameters.len == 7);

    try kernels.rmsnorm_bf16.launch(
        .{ shape.blocks(), 1, 1 },
        .{ norm_threads, 1, 1 },
        normSharedBytes(),
        stream,
        parameters[0..],
    );
}

/// Launches `ew_rmsnorm_gated` over `shape`; `z` is the pre-activation gate.
pub fn launchRmsnormGated(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    values: cuda_driver.DevicePtr,
    gate: cuda_driver.DevicePtr,
    weight: cuda_driver.DevicePtr,
    out: cuda_driver.DevicePtr,
    shape: NormShape,
    epsilon: f32,
    activation: ref_ops.Activation,
) kernel.KernelError!void {
    shape.check();
    assert(values != 0);
    assert(gate != 0);
    assert(weight != 0);
    assert(out != 0);

    var x = values;
    var z = gate;
    var w = weight;
    var result = out;
    var rows: i32 = @intCast(shape.rows);
    var columns: i32 = @intCast(shape.columns);
    var variance_epsilon = epsilon;
    var gate_activation = activationCode(activation);
    var parameters = [_]?*anyopaque{
        @ptrCast(&x),
        @ptrCast(&z),
        @ptrCast(&w),
        @ptrCast(&result),
        @ptrCast(&rows),
        @ptrCast(&columns),
        @ptrCast(&variance_epsilon),
        @ptrCast(&gate_activation),
    };
    assert(parameters.len == 8);

    try kernels.rmsnorm_gated.launch(
        .{ shape.rows, 1, 1 },
        .{ norm_threads, 1, 1 },
        normSharedBytes(),
        stream,
        parameters[0..],
    );
}

/// Launches `ew_rope_build_tables` for `shape`.
pub fn launchRopeBuild(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    tensors: RopeTableTensors,
    shape: RopeShape,
) kernel.KernelError!void {
    tensors.check();
    shape.check();

    var positions_t = tensors.positions_t;
    var positions_h = tensors.positions_h;
    var positions_w = tensors.positions_w;
    var tokens: i32 = @intCast(shape.tokens);
    var frequencies = tensors.frequencies;
    var frequency_count: i32 = @intCast(shape.frequencyCount());
    var section_h: i32 = @intCast(shape.section[1]);
    var section_w: i32 = @intCast(shape.section[2]);
    var cos = tensors.cos;
    var sin = tensors.sin;
    var parameters = [_]?*anyopaque{
        @ptrCast(&positions_t),
        @ptrCast(&positions_h),
        @ptrCast(&positions_w),
        @ptrCast(&tokens),
        @ptrCast(&frequencies),
        @ptrCast(&frequency_count),
        @ptrCast(&section_h),
        @ptrCast(&section_w),
        @ptrCast(&cos),
        @ptrCast(&sin),
    };
    assert(parameters.len == 10);

    try kernels.rope_build.launch(
        .{ shape.tokens, 1, 1 },
        .{ norm_threads, 1, 1 },
        0,
        stream,
        parameters[0..],
    );
}

/// Launches `ew_rope_apply_qk`: query and key rotated in one launch.
pub fn launchRopeApplyQk(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    tensors: RopeTensors,
    shape: RopeShape,
) kernel.KernelError!void {
    tensors.check();
    shape.check();

    var q = tensors.q;
    var k = tensors.k;
    var cos = tensors.cos;
    var sin = tensors.sin;
    var tokens: i32 = @intCast(shape.tokens);
    var heads: i32 = @intCast(shape.heads);
    var head_dim: i32 = @intCast(shape.head_dim);
    var rotated: i32 = @intCast(shape.rotated);
    var parameters = [_]?*anyopaque{
        @ptrCast(&q),
        @ptrCast(&k),
        @ptrCast(&cos),
        @ptrCast(&sin),
        @ptrCast(&tokens),
        @ptrCast(&heads),
        @ptrCast(&head_dim),
        @ptrCast(&rotated),
    };
    assert(parameters.len == 8);

    try kernels.rope_apply_qk.launch(
        .{ shape.tokens, shape.heads, 1 },
        .{ rope_threads, 1, 1 },
        0,
        stream,
        parameters[0..],
    );
}

/// Launches `ew_rope_apply_q`: the indexer's query-only variant.
pub fn launchRopeApplyQ(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    q: cuda_driver.DevicePtr,
    cos: cuda_driver.DevicePtr,
    sin: cuda_driver.DevicePtr,
    shape: RopeShape,
) kernel.KernelError!void {
    shape.check();
    assert(q != 0);
    assert(cos != 0);
    assert(sin != 0);

    var query = q;
    var table_cos = cos;
    var table_sin = sin;
    var tokens: i32 = @intCast(shape.tokens);
    var heads: i32 = @intCast(shape.heads);
    var head_dim: i32 = @intCast(shape.head_dim);
    var rotated: i32 = @intCast(shape.rotated);
    var parameters = [_]?*anyopaque{
        @ptrCast(&query),
        @ptrCast(&table_cos),
        @ptrCast(&table_sin),
        @ptrCast(&tokens),
        @ptrCast(&heads),
        @ptrCast(&head_dim),
        @ptrCast(&rotated),
    };
    assert(parameters.len == 7);

    try kernels.rope_apply_q.launch(
        .{ shape.tokens, shape.heads, 1 },
        .{ rope_threads, 1, 1 },
        0,
        stream,
        parameters[0..],
    );
}

/// Launches `ew_hyper_mix` for every token of `shape`.
pub fn launchHyperMix(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    tensors: HyperTensors,
    shape: HyperShape,
    epsilon: f32,
) kernel.KernelError!void {
    tensors.check();
    shape.check();
    assert(epsilon > 0.0);

    var input = tensors.input;
    var norm_weight = tensors.norm_weight;
    var mix_down = tensors.mix_down;
    var mix_up = tensors.mix_up;
    var inject_weight = tensors.inject_weight;
    var mixed = tensors.mixed;
    var injection = tensors.injection;
    var tokens: i32 = @intCast(shape.tokens);
    var hidden: i32 = @intCast(shape.hidden);
    var streams: i32 = @intCast(shape.streams);
    var lowrank: i32 = @intCast(shape.lowrank);
    var variance_epsilon = epsilon;
    var parameters = [_]?*anyopaque{
        @ptrCast(&input),
        @ptrCast(&norm_weight),
        @ptrCast(&mix_down),
        @ptrCast(&mix_up),
        @ptrCast(&inject_weight),
        @ptrCast(&mixed),
        @ptrCast(&injection),
        @ptrCast(&tokens),
        @ptrCast(&hidden),
        @ptrCast(&streams),
        @ptrCast(&lowrank),
        @ptrCast(&variance_epsilon),
    };
    assert(parameters.len == 12);

    try kernels.hyper_mix.launch(
        .{ shape.tokens, 1, 1 },
        .{ norm_threads, 1, 1 },
        hyperSharedBytes(shape),
        stream,
        parameters[0..],
    );
}

/// Blocks one elementwise launch uses: one per 256 elements, at least one.
fn elementwiseBlocks(count: u32) u32 {
    assert(count > 0);
    const per_block: u32 = norm_threads;
    const blocks = std.math.divCeil(u32, count, per_block) catch unreachable;
    assert(blocks > 0);
    return blocks;
}

/// Launches `ew_silu_mul`: `out = silu(a) * b`.
pub fn launchSiluMul(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    a: cuda_driver.DevicePtr,
    b: cuda_driver.DevicePtr,
    out: cuda_driver.DevicePtr,
    count: u32,
) kernel.KernelError!void {
    assert(a != 0);
    assert(b != 0);
    assert(out != 0);
    assert(count > 0);

    var input = a;
    var gate = b;
    var result = out;
    var elements: i32 = @intCast(count);
    var parameters = [_]?*anyopaque{
        @ptrCast(&input),
        @ptrCast(&gate),
        @ptrCast(&result),
        @ptrCast(&elements),
    };
    assert(parameters.len == 4);

    try kernels.silu_mul.launch(
        .{ elementwiseBlocks(count), 1, 1 },
        .{ norm_threads, 1, 1 },
        0,
        stream,
        parameters[0..],
    );
}

/// Launches `ew_sigmoid_gate`: `out = sigmoid(a)`.
pub fn launchSigmoidGate(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    a: cuda_driver.DevicePtr,
    out: cuda_driver.DevicePtr,
    count: u32,
) kernel.KernelError!void {
    assert(a != 0);
    assert(out != 0);
    assert(count > 0);

    var input = a;
    var result = out;
    var elements: i32 = @intCast(count);
    var parameters = [_]?*anyopaque{
        @ptrCast(&input),
        @ptrCast(&result),
        @ptrCast(&elements),
    };
    assert(parameters.len == 3);

    try kernels.sigmoid_gate.launch(
        .{ elementwiseBlocks(count), 1, 1 },
        .{ norm_threads, 1, 1 },
        0,
        stream,
        parameters[0..],
    );
}

/// Launches `ew_softmax_rows` over `[rows][columns]`, in place.
pub fn launchSoftmax(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    values: cuda_driver.DevicePtr,
    rows: u32,
    columns: u32,
) kernel.KernelError!void {
    assert(values != 0);
    assert(rows > 0);
    assert(columns > 0);
    assert(columns <= norm_columns_max);

    var data = values;
    var row_count: i32 = @intCast(rows);
    var width: i32 = @intCast(columns);
    var parameters = [_]?*anyopaque{
        @ptrCast(&data),
        @ptrCast(&row_count),
        @ptrCast(&width),
    };
    assert(parameters.len == 3);

    try kernels.softmax.launch(
        .{ rows, 1, 1 },
        .{ norm_threads, 1, 1 },
        normSharedBytes(),
        stream,
        parameters[0..],
    );
}

/// Launches `ew_argmax`: the largest value's index, lowest index on a tie.
pub fn launchArgmax(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    values: cuda_driver.DevicePtr,
    index: cuda_driver.DevicePtr,
    value: cuda_driver.DevicePtr,
    count: u32,
) kernel.KernelError!void {
    assert(values != 0);
    assert(index != 0);
    assert(value != 0);
    assert(count > 0);

    var input = values;
    var elements: i32 = @intCast(count);
    var out_index = index;
    var out_value = value;
    var parameters = [_]?*anyopaque{
        @ptrCast(&input),
        @ptrCast(&elements),
        @ptrCast(&out_index),
        @ptrCast(&out_value),
    };
    assert(parameters.len == 4);

    try kernels.argmax.launch(
        .{ 1, 1, 1 },
        .{ norm_threads, 1, 1 },
        0,
        stream,
        parameters[0..],
    );
}

/// Launches `ew_sampler` for one token of one sequence.
pub fn launchSampler(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    tensors: SamplerTensors,
    vocab: u32,
    history_len: u32,
    params: SampleParams,
) kernel.KernelError!void {
    tensors.check(history_len);
    params.check();
    assert(vocab > 0);

    var logits = tensors.logits;
    var history = tensors.history;
    var history_length: i32 = @intCast(history_len);
    var vocabulary: i32 = @intCast(vocab);
    var penalty = params.repetition_penalty;
    var temperature = params.temperature;
    var top_k: i32 = @intCast(params.top_k);
    var top_p = params.top_p;
    var min_p = params.min_p;
    var seed = params.seed;
    var draw = params.draw;
    var scaled = tensors.scaled;
    var out_id = tensors.id;
    var out_logprob = tensors.logprob;
    var parameters = [_]?*anyopaque{
        @ptrCast(&logits),
        @ptrCast(&history),
        @ptrCast(&history_length),
        @ptrCast(&vocabulary),
        @ptrCast(&penalty),
        @ptrCast(&temperature),
        @ptrCast(&top_k),
        @ptrCast(&top_p),
        @ptrCast(&min_p),
        @ptrCast(&seed),
        @ptrCast(&draw),
        @ptrCast(&scaled),
        @ptrCast(&out_id),
        @ptrCast(&out_logprob),
    };
    assert(parameters.len == 14);

    try kernels.sampler.launch(
        .{ 1, 1, 1 },
        .{ sampler_threads, 1, 1 },
        0,
        stream,
        parameters[0..],
    );
}

// ---------------------------------------------------------------------------
// Acceptance suite.
//
// Every case hands the kernel and the fp32 CPU oracle the same inputs. The
// tolerances are the output format's: `value_tolerance` for a f32 store, and
// the half-ulp bound plus `bf16_tolerance` for the one bf16 store, because half
// a bf16 step is 4e-3 relative and a tighter absolute bound is unsatisfiable
// there by construction.
// ---------------------------------------------------------------------------

/// Bytes of pinned host memory the suite stages uploads through. One case holds
/// both 13 MiB production hyper-connection matrices at once, and the arena only
/// rewinds at the end of a case.
const staging_bytes: u64 = 40 * 1024 * 1024;

/// Bytes of device memory a suite allocates once and bumps within.
const device_bytes: u64 = 96 * 1024 * 1024;

/// Alignment of every tensor inside both arenas.
const tensor_alignment: u64 = 256;

/// The issue's tolerance for an f32 kernel against the f32 oracle, relative to
/// the tensor's largest magnitude.
const value_tolerance: f64 = 1.0e-5;

/// The issue's bf16 round-trip tolerance, relative to the tensor's largest
/// magnitude. One bf16 step at |y| ~ 11 is 0.0625, so this is the coarse bound;
/// `ulp_tolerance` is the one that decides correctness.
const bf16_tolerance: f64 = 4.0e-3;

/// Correct rounding: a bf16 store must land within half a bf16 step of the f32
/// oracle.
const ulp_tolerance: f64 = 0.5;

/// xorshift64* stream for the fixtures; the kernels never see the generator.
const Random = struct {
    state: u64,

    fn init(seed: u64) Random {
        assert(seed != 0);
        assert(@sizeOf(u64) == 8);

        return .{ .state = seed };
    }

    /// Next 32 raw bits.
    fn next(self: *Random) u32 {
        assert(self.state != 0);
        assert(@sizeOf(u32) == 4);

        var value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        const product = value *% 2685821657736338717;
        return @truncate(product >> 32);
    }

    /// Uniform in `[-1, 1)` with 24 significant bits.
    fn signed(self: *Random) f32 {
        const raw = self.next();
        assert(@sizeOf(f32) == 4);
        assert(raw <= std.math.maxInt(u32));

        const unit = @as(f32, @floatFromInt(raw >> 8)) * (1.0 / 8388608.0);
        assert(unit >= 0.0);
        assert(unit < 2.0);
        return unit - 1.0;
    }
};

/// Fills `out` with a pseudo-random sequence in `[-scale, scale)`.
fn fillRandom(out: []f32, seed: u64, scale: f32) void {
    assert(out.len > 0);
    assert(scale > 0.0);

    var random = Random.init(seed);
    for (out) |*value| {
        value.* = random.signed() * scale;
    }
}

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

/// One bf16 step at the magnitude of `value`.
///
/// bf16 keeps eight significand bits, so the step is `2^(exponent - 7)`; the
/// subnormal range shares the smallest normal step.
fn bf16Step(value: f32) f64 {
    const magnitude: u32 = @bitCast(@abs(value));
    const exponent_field = magnitude >> 23;
    assert(exponent_field <= 0xff);

    const exponent: i32 = if (exponent_field == 0)
        -126
    else
        @as(i32, @intCast(exponent_field)) - 127;
    const step = std.math.pow(f64, 2.0, @floatFromInt(exponent - 7));
    assert(step > 0.0);
    return step;
}

/// Distance between a stored bf16 value and the f32 oracle, in bf16 steps.
fn bf16UlpError(actual: f32, expected: f32) f64 {
    assert(actual == actual);
    assert(expected == expected);

    const difference = @abs(@as(f64, actual) - @as(f64, expected));
    const step = bf16Step(expected);
    assert(step > 0.0);
    return difference / step;
}

/// Worse-case ulp distance of a bf16 result against the f32 oracle.
fn bf16UlpWorst(actual: []const f32, expected: []const f32) f64 {
    assert(actual.len == expected.len);
    assert(actual.len > 0);

    var worst: f64 = 0.0;
    for (actual, expected) |got, want| {
        worst = @max(worst, bf16UlpError(got, want));
    }
    assert(worst >= 0.0);
    return worst;
}

/// Scaled difference between an oracle result and a kernel result.
const Comparison = struct {
    absolute: f64,
    scaled: f64,
    reference_max: f64,
    index: u32,
    exact: u32,
    elements: u32,
};

/// Compares two equally shaped f32 views: `max|expected - actual|` divided by
/// `max(1e-6, max|expected|)`. A non-finite kernel value scores an infinite
/// error: without that, `NaN > largest` is false and a NaN would contribute
/// zero error and pass every gate.
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

    const clean = compare(expected[0..], exact[0..]);
    const broken = compare(expected[0..], poisoned[0..]);
    try std.testing.expect(clean.scaled == 0.0);
    try std.testing.expect(!(broken.scaled <= value_tolerance));
    try std.testing.expect(broken.index == 1);
}

/// Prints one comparison, with the absolute error and the reference magnitude
/// the scaled figure is divided by.
fn report(label: []const u8, result: Comparison) void {
    assert(label.len > 0);
    assert(result.elements > 0);

    std.debug.print(
        "  {s:<16} scaled {e:.3}  abs {e:.3}  ref {e:.3}  {d}/{d}\n",
        .{
            label,
            result.scaled,
            result.absolute,
            result.reference_max,
            result.exact,
            result.elements,
        },
    );
}

/// Prints one comparison and fails when it is over `tolerance`.
fn expectWithin(label: []const u8, result: Comparison, tolerance: f64) !void {
    assert(label.len > 0);
    assert(tolerance > 0.0);

    report(label, result);
    try std.testing.expect(result.scaled <= tolerance);
}

/// Prints one scalar pair and fails when it is off by more than `tolerance`
/// relative to its own magnitude.
fn expectClose(label: []const u8, expected: f32, actual: f32, tolerance: f64) !void {
    assert(label.len > 0);
    assert(tolerance > 0.0);
    assert(expected == expected);

    const scale = @max(1.0, @abs(@as(f64, expected)));
    const error_value = @abs(@as(f64, expected) - @as(f64, actual)) / scale;
    std.debug.print(
        "  {s:<16} expected {e:.6}  actual {e:.6}  rel {e:.3}\n",
        .{ label, expected, actual, error_value },
    );
    try std.testing.expect(error_value <= tolerance);
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

    /// Retains GPU 0 and loads the `elementwise` cubin; skips on a host without
    /// a driver.
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

    /// Stages `source`, allocates device memory for it and starts its copy.
    fn upload(
        harness: *Harness,
        comptime Element: type,
        source: []const Element,
    ) !cuda_driver.DevicePtr {
        assert(source.len > 0);
        assert(@sizeOf(Element) > 0);

        const bytes = std.mem.sliceAsBytes(source);
        const destination = try harness.allocate(bytes.len);
        const region = try harness.stage(bytes.len);
        @memcpy(region, bytes);
        try harness.enqueue(destination, region);
        return destination;
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

    /// Allocates `count` elements of the device arena.
    fn reserve(
        harness: *Harness,
        comptime Element: type,
        count: u32,
    ) !cuda_driver.DevicePtr {
        assert(count > 0);
        assert(@sizeOf(Element) > 0);

        return harness.allocate(@as(u64, count) * @sizeOf(Element));
    }

    /// Copies `source` back and waits for it.
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

test "silu_mul, sigmoid_gate, softmax and argmax match the fp32 oracle" {
    var harness = try Harness.init();
    defer harness.deinit();
    const allocator = std.testing.allocator;

    const count: u32 = 1024;
    const a = try allocator.alloc(f32, count);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, count);
    defer allocator.free(b);
    const expected = try allocator.alloc(f32, count);
    defer allocator.free(expected);
    const actual = try allocator.alloc(f32, count);
    defer allocator.free(actual);

    fillRandom(a, 0x51_1, 4.0);
    fillRandom(b, 0x51_2, 4.0);
    for (a, b, 0..) |value, gate, index| {
        expected[index] = ref_ops.silu(value) * gate;
    }
    const device_a = try harness.upload(f32, a);
    const device_b = try harness.upload(f32, b);
    const device_out = try harness.reserve(f32, count);
    try launchSiluMul(&harness.kernels, harness.stream.handle, device_a, device_b, device_out, count);
    try harness.download(f32, device_out, actual);
    try expectWithin("silu_mul", compare(expected, actual), value_tolerance);

    for (a, 0..) |value, index| {
        expected[index] = ref_ops.sigmoid(value);
    }
    try launchSigmoidGate(&harness.kernels, harness.stream.handle, device_a, device_out, count);
    try harness.download(f32, device_out, actual);
    try expectWithin("sigmoid_gate", compare(expected, actual), value_tolerance);

    // Softmax over four rows of 256, in place.
    const rows: u32 = 4;
    const columns: u32 = 256;
    const values = try allocator.alloc(f32, rows * columns);
    defer allocator.free(values);
    fillRandom(values, 0x51_3, 6.0);
    // Upload first: the kernel softmaxes in place, so the host copy must still be
    // the raw scores when the device receives them.
    const device_values = try harness.upload(f32, values);
    for (0..rows) |row| {
        const start = row * columns;
        ref_ops.softmax(values[start..][0..columns]);
    }
    try launchSoftmax(&harness.kernels, harness.stream.handle, device_values, rows, columns);
    try harness.download(f32, device_values, actual[0 .. rows * columns]);
    try expectWithin(
        "softmax",
        compare(values, actual[0 .. rows * columns]),
        value_tolerance,
    );

    // Argmax: a unique maximum, then an exact tie at two indices, which must
    // resolve to the lower index.
    fillRandom(values, 0x51_4, 1.0);
    values[300] = 3.5;
    values[700] = 3.5;
    const device_index = try harness.reserve(i32, 1);
    const device_value = try harness.reserve(f32, 1);
    const device_scan = try harness.upload(f32, values);
    try launchArgmax(&harness.kernels, harness.stream.handle, device_scan, device_index, device_value, count);
    var index_result = [_]i32{0};
    var value_result = [_]f32{0.0};
    try harness.download(i32, device_index, index_result[0..]);
    try harness.download(f32, device_value, value_result[0..]);
    try std.testing.expectEqual(@as(i32, 300), index_result[0]);
    try std.testing.expectEqual(@as(f32, 3.5), value_result[0]);

    values[700] = 3.0;
    try harness.write(device_scan, f32, values);
    try launchArgmax(&harness.kernels, harness.stream.handle, device_scan, device_index, device_value, count);
    try harness.download(i32, device_index, index_result[0..]);
    try std.testing.expectEqual(@as(i32, 300), index_result[0]);

    try harness.reset();
}

/// One grouped-norm case: upload, launch, download and compare against
/// `ops.rmsnorm` with the same group size and epsilon.
fn checkRmsnorm(
    harness: *Harness,
    allocator: std.mem.Allocator,
    shape: NormShape,
    epsilon: f32,
    seed: u64,
    label: []const u8,
) !void {
    shape.check();
    assert(epsilon > 0.0);
    assert(label.len > 0);

    const elements: u32 = shape.rows * shape.columns;
    const x = try allocator.alloc(f32, elements);
    defer allocator.free(x);
    const weight = try allocator.alloc(f32, shape.columns);
    defer allocator.free(weight);
    const expected = try allocator.alloc(f32, elements);
    defer allocator.free(expected);
    const actual = try allocator.alloc(f32, elements);
    defer allocator.free(actual);

    fillRandom(x, seed, 2.0);
    fillRandom(weight, seed +% 1, 0.08);
    for (0..shape.rows) |row| {
        const start = row * shape.columns;
        ref_ops.rmsnorm(
            x[start..][0..shape.columns],
            weight,
            expected[start..][0..shape.columns],
            shape.group,
            epsilon,
        );
    }

    const device_x = try harness.upload(f32, x);
    const device_weight = try harness.upload(f32, weight);
    const device_out = try harness.reserve(f32, elements);
    const tensors = NormTensors{
        .x = device_x,
        .weight = device_weight,
        .out = device_out,
    };
    try launchRmsnorm(&harness.kernels, harness.stream.handle, tensors, shape, epsilon);
    try harness.download(f32, device_out, actual);
    try expectWithin(label, compare(expected, actual), value_tolerance);
    try harness.reset();
}

test "rmsnorm matches the oracle with a group size and the 1 + w convention" {
    var harness = try Harness.init();
    defer harness.deinit();
    const allocator = std.testing.allocator;

    // The hyper-connection shape: four 2560-dim groups of the 10240-dim stream.
    try checkRmsnorm(
        &harness,
        allocator,
        .{ .rows = 3, .columns = 10240, .group = 2560 },
        1e-6,
        0x12_1,
        "rmsnorm hc",
    );
    // A whole-row norm, which is what every non-HC RMSNorm uses.
    try checkRmsnorm(
        &harness,
        allocator,
        .{ .rows = 5, .columns = 2560, .group = 2560 },
        1e-6,
        0x12_2,
        "rmsnorm row",
    );
    // The tiny fixture's shape, so the group path is also exercised off the
    // production widths.
    try checkRmsnorm(
        &harness,
        allocator,
        .{ .rows = 4, .columns = 256, .group = 128 },
        1e-6,
        0x12_3,
        "rmsnorm tiny",
    );
}

test "the bf16 norm store is correctly rounded" {
    var harness = try Harness.init();
    defer harness.deinit();
    const allocator = std.testing.allocator;

    const shape = NormShape{ .rows = 2, .columns = 10240, .group = 2560 };
    const epsilon: f32 = 1e-6;
    const elements: u32 = shape.rows * shape.columns;

    // Dyadic operands: every product and partial sum of the reduction is exact
    // in f32, so the kernel and the oracle cannot disagree by a rounding step
    // and the ulp figure measures the store alone.
    const dyadic = [_]f32{ -1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0 };
    const x = try allocator.alloc(f32, elements);
    defer allocator.free(x);
    const weight = try allocator.alloc(f32, shape.columns);
    defer allocator.free(weight);
    const expected = try allocator.alloc(f32, elements);
    defer allocator.free(expected);
    const actual = try allocator.alloc(f32, elements);
    defer allocator.free(actual);

    for (x, 0..) |*value, index| {
        value.* = dyadic[index % dyadic.len];
    }
    for (weight, 0..) |*value, index| {
        value.* = dyadic[(index % dyadic.len)] * 0.0625;
    }
    for (0..shape.rows) |row| {
        const start = row * shape.columns;
        ref_ops.rmsnorm(
            x[start..][0..shape.columns],
            weight,
            expected[start..][0..shape.columns],
            shape.group,
            epsilon,
        );
    }

    const half = try allocator.alloc(u16, elements);
    defer allocator.free(half);
    for (x, 0..) |value, index| {
        // Every operand is exactly representable in bf16, so the kernel sees
        // the same numbers the oracle does.
        half[index] = bf16Bits(value);
        try std.testing.expectEqual(value, bf16Value(half[index]));
    }

    const device_x = try harness.upload(u16, half);
    const device_weight = try harness.upload(f32, weight);
    const device_out = try harness.reserve(u16, elements);
    const tensors = NormTensors{
        .x = device_x,
        .weight = device_weight,
        .out = device_out,
    };
    try launchRmsnormBf16(&harness.kernels, harness.stream.handle, tensors, shape, epsilon);
    const stored = try allocator.alloc(u16, elements);
    defer allocator.free(stored);
    try harness.download(u16, device_out, stored);
    for (stored, 0..) |bits, index| {
        actual[index] = bf16Value(bits);
    }

    try expectWithin("rmsnorm bf16", compare(expected, actual), bf16_tolerance);
    const worst_ulp = bf16UlpWorst(actual, expected);
    std.debug.print(
        "  {s:<16} worst {d:.4} bf16 ulp (bound 0.5)\n",
        .{ "bf16 ulp", worst_ulp },
    );
    try std.testing.expect(worst_ulp <= ulp_tolerance);
    try harness.reset();
}

test "rmsnorm_gated norms before multiplying by the activated gate" {
    var harness = try Harness.init();
    defer harness.deinit();
    const allocator = std.testing.allocator;

    const shape = NormShape{ .rows = 4, .columns = 2560, .group = 2560 };
    const epsilon: f32 = 1e-6;
    const elements: u32 = shape.rows * shape.columns;

    const x = try allocator.alloc(f32, elements);
    defer allocator.free(x);
    const z = try allocator.alloc(f32, elements);
    defer allocator.free(z);
    const weight = try allocator.alloc(f32, shape.columns);
    defer allocator.free(weight);
    const expected = try allocator.alloc(f32, elements);
    defer allocator.free(expected);
    const actual = try allocator.alloc(f32, elements);
    defer allocator.free(actual);

    fillRandom(x, 0x31_1, 2.0);
    fillRandom(z, 0x31_2, 6.0);
    fillRandom(weight, 0x31_3, 0.08);

    const device_x = try harness.upload(f32, x);
    const device_z = try harness.upload(f32, z);
    const device_weight = try harness.upload(f32, weight);
    const device_out = try harness.reserve(f32, elements);

    const activations = [_]ref_ops.Activation{ .sigmoid, .silu };
    for (activations) |activation| {
        for (0..shape.rows) |row| {
            const start = row * shape.columns;
            ref_ops.rmsnorm_gated(
                x[start..][0..shape.columns],
                z[start..][0..shape.columns],
                weight,
                expected[start..][0..shape.columns],
                epsilon,
                activation,
            );
        }
        try launchRmsnormGated(
            &harness.kernels,
            harness.stream.handle,
            device_x,
            device_z,
            device_weight,
            device_out,
            shape,
            epsilon,
            activation,
        );
        try harness.download(f32, device_out, actual);
        const label = switch (activation) {
            .sigmoid => "gated sigmoid",
            .silu => "gated silu",
            .none => "gated none",
        };
        try expectWithin(label, compare(expected, actual), value_tolerance);
    }
    try harness.reset();
}

/// Counts dims that changed bit for bit, either inside the rotated prefix or in
/// the untouched tail (`within_rotated = false`). The tail count must be zero:
/// a full-rotary bug is otherwise invisible in an aggregate error.
fn countBitChanges(
    original: []const f32,
    updated: []const f32,
    shape: RopeShape,
    comptime within_rotated: bool,
) u32 {
    assert(original.len == updated.len);
    assert(original.len == shape.tokens * shape.heads * shape.head_dim);

    const first: u32 = if (within_rotated) 0 else shape.rotated;
    const last: u32 = if (within_rotated) shape.rotated else shape.head_dim;
    var changes: u32 = 0;
    for (0..shape.tokens) |token| {
        for (0..shape.heads) |head| {
            const base = (token * shape.heads + head) * shape.head_dim;
            for (first..last) |dim| {
                const before: u32 = @bitCast(original[base + dim]);
                const after: u32 = @bitCast(updated[base + dim]);
                if (before != after) {
                    changes += 1;
                }
            }
        }
    }
    assert(changes <= original.len);
    return changes;
}

test "partial rotary with the mRoPE interleave matches the reference" {
    var harness = try Harness.init();
    defer harness.deinit();
    const allocator = std.testing.allocator;

    // The production geometry: 64 of 256 dims rotate, and the 32 frequencies
    // are assigned to the three grids 11/11/10.
    const shape = RopeShape{
        .tokens = 5,
        .heads = 4,
        .head_dim = 256,
        .rotated = 64,
        .section = .{ 11, 11, 10 },
    };
    shape.check();
    const frequency_count = shape.frequencyCount();
    assert(frequency_count == 32);

    const frequencies = try allocator.alloc(f32, frequency_count);
    defer allocator.free(frequencies);
    ref_rope.inv_freq_init(10_000_000.0, frequencies);

    // Three distinct grids: the text path repeats the temporal position, but a
    // multimodal prompt interleaves height and width lanes, which is the case
    // that a text-only fixture cannot see.
    const temporal = [_]u32{ 0, 1, 2, 3, 4 };
    const height = [_]u32{ 7, 7, 3, 9, 1 };
    const width = [_]u32{ 5, 2, 8, 4, 6 };
    var temporal_i32 = [_]i32{ 0, 1, 2, 3, 4 };
    var height_i32 = [_]i32{ 7, 7, 3, 9, 1 };
    var width_i32 = [_]i32{ 5, 2, 8, 4, 6 };
    for (0..shape.tokens) |token| {
        temporal_i32[token] = @intCast(temporal[token]);
        height_i32[token] = @intCast(height[token]);
        width_i32[token] = @intCast(width[token]);
    }

    const table_elements: u32 = shape.tokens * shape.rotated;
    const cos_expected = try allocator.alloc(f32, table_elements);
    defer allocator.free(cos_expected);
    const sin_expected = try allocator.alloc(f32, table_elements);
    defer allocator.free(sin_expected);
    const cos_actual = try allocator.alloc(f32, table_elements);
    defer allocator.free(cos_actual);
    const sin_actual = try allocator.alloc(f32, table_elements);
    defer allocator.free(sin_actual);

    ref_rope.cos_sin(
        .{ temporal[0..], height[0..], width[0..] },
        frequencies,
        shape.section,
        cos_expected,
        sin_expected,
    );

    const device_positions_t = try harness.upload(i32, temporal_i32[0..]);
    const device_positions_h = try harness.upload(i32, height_i32[0..]);
    const device_positions_w = try harness.upload(i32, width_i32[0..]);
    const device_frequencies = try harness.upload(f32, frequencies);
    const device_cos = try harness.reserve(f32, table_elements);
    const device_sin = try harness.reserve(f32, table_elements);
    const tables = RopeTableTensors{
        .positions_t = device_positions_t,
        .positions_h = device_positions_h,
        .positions_w = device_positions_w,
        .frequencies = device_frequencies,
        .cos = device_cos,
        .sin = device_sin,
    };
    try launchRopeBuild(&harness.kernels, harness.stream.handle, tables, shape);
    try harness.download(f32, device_cos, cos_actual);
    try harness.download(f32, device_sin, sin_actual);
    try expectWithin("rope cos", compare(cos_expected, cos_actual), value_tolerance);
    try expectWithin("rope sin", compare(sin_expected, sin_actual), value_tolerance);

    // The interleave itself: lane 1 of each triple must read the height grid and
    // lane 2 the width grid, so with three distinct grids the table cannot be
    // reproduced by a plain temporal broadcast.
    var interleaved: u32 = 0;
    for (0..shape.tokens) |token| {
        for (0..frequency_count) |lane| {
            const grid = ref_rope.mrope_grid(@intCast(lane), shape.section);
            const position = switch (grid) {
                0 => temporal[token],
                1 => height[token],
                else => width[token],
            };
            const angle = @as(f32, @floatFromInt(position)) * frequencies[lane];
            const index = token * shape.rotated + lane;
            try std.testing.expectApproxEqAbs(
                @cos(angle),
                cos_actual[index],
                1e-6,
            );
            if (grid != 0) {
                interleaved += 1;
            }
        }
    }
    std.debug.print(
        "  {s:<16} {d}/{d} frequencies read a non-temporal grid\n",
        .{ "mrope lanes", interleaved, shape.tokens * frequency_count },
    );
    try std.testing.expect(interleaved > 0);

    // Now the rotation itself, against `rope.apply` head by head.
    const elements: u32 = shape.tokens * shape.heads * shape.head_dim;
    const q_input = try allocator.alloc(f32, elements);
    defer allocator.free(q_input);
    const k_input = try allocator.alloc(f32, elements);
    defer allocator.free(k_input);
    const expected = try allocator.alloc(f32, elements);
    defer allocator.free(expected);
    const actual = try allocator.alloc(f32, elements);
    defer allocator.free(actual);

    fillRandom(q_input, 0x77_1, 3.0);
    fillRandom(k_input, 0x77_2, 3.0);
    for (0..shape.tokens) |token| {
        const table = token * shape.rotated;
        for (0..shape.heads) |head| {
            const base = (token * shape.heads + head) * shape.head_dim;
            ref_rope.apply(
                q_input[base..][0..shape.head_dim],
                cos_expected[table..][0..shape.rotated],
                sin_expected[table..][0..shape.rotated],
                expected[base..][0..shape.head_dim],
            );
        }
    }

    const device_q = try harness.upload(f32, q_input);
    const device_k = try harness.upload(f32, k_input);
    const rope_tensors = RopeTensors{
        .q = device_q,
        .k = device_k,
        .cos = device_cos,
        .sin = device_sin,
    };
    try launchRopeApplyQk(
        &harness.kernels,
        harness.stream.handle,
        rope_tensors,
        shape,
    );
    try harness.download(f32, device_q, actual);
    try expectWithin("rope q", compare(expected, actual), value_tolerance);

    // The k tensor is rotated too, so the query's result must be unchanged by
    // the addition of k and both must agree with the same oracle.
    const k_actual = try allocator.alloc(f32, elements);
    defer allocator.free(k_actual);
    try harness.download(f32, device_k, k_actual);
    for (0..shape.tokens) |token| {
        const table = token * shape.rotated;
        for (0..shape.heads) |head| {
            const base = (token * shape.heads + head) * shape.head_dim;
            ref_rope.apply(
                k_input[base..][0..shape.head_dim],
                cos_expected[table..][0..shape.rotated],
                sin_expected[table..][0..shape.rotated],
                expected[base..][0..shape.head_dim],
            );
        }
    }
    try expectWithin("rope k", compare(expected, k_actual), value_tolerance);

    // Dims 64..255 must be bit-identical to the input, and the rotated prefix
    // must actually have moved.
    const tail_changes = countBitChanges(q_input, actual, shape, false);
    const prefix_changes = countBitChanges(q_input, actual, shape, true);
    std.debug.print(
        "  {s:<16} prefix changed {d}/{d}, tail changed {d}/{d}\n",
        .{
            "rope coverage",
            prefix_changes,
            shape.tokens * shape.heads * shape.rotated,
            tail_changes,
            shape.tokens * shape.heads * (shape.head_dim - shape.rotated),
        },
    );
    try std.testing.expectEqual(@as(u32, 0), tail_changes);
    for (0..shape.tokens) |token| {
        for (0..shape.heads) |head| {
            const base = (token * shape.heads + head) * shape.head_dim;
            for (shape.rotated..shape.head_dim) |dim| {
                // Bit equality, not approximate equality: the tail is a copy.
                try std.testing.expectEqual(
                    @as(u32, @bitCast(q_input[base + dim])),
                    @as(u32, @bitCast(actual[base + dim])),
                );
            }
        }
    }
    const rotated_total = shape.tokens * shape.heads * shape.rotated;
    try std.testing.expect(prefix_changes * 2 > rotated_total);

    try harness.reset();
}

test "the query-only rotary leaves the indexer's keys alone" {
    var harness = try Harness.init();
    defer harness.deinit();
    const allocator = std.testing.allocator;

    const shape = RopeShape{
        .tokens = 3,
        .heads = 4,
        .head_dim = 128,
        .rotated = 32,
        .section = .{ 11, 11, 10 },
    };
    shape.check();
    const frequency_count = shape.frequencyCount();
    const frequencies = try allocator.alloc(f32, frequency_count);
    defer allocator.free(frequencies);
    ref_rope.inv_freq_init(10_000_000.0, frequencies);

    const positions = [_]u32{ 0, 1, 2 };
    var positions_i32 = [_]i32{ 0, 1, 2 };
    for (0..shape.tokens) |token| {
        positions_i32[token] = @intCast(positions[token]);
    }
    const table_elements: u32 = shape.tokens * shape.rotated;
    const cos_expected = try allocator.alloc(f32, table_elements);
    defer allocator.free(cos_expected);
    const sin_expected = try allocator.alloc(f32, table_elements);
    defer allocator.free(sin_expected);
    ref_rope.cos_sin(
        .{ positions[0..], positions[0..], positions[0..] },
        frequencies,
        shape.section,
        cos_expected,
        sin_expected,
    );

    const device_positions = try harness.upload(i32, positions_i32[0..]);
    const device_frequencies = try harness.upload(f32, frequencies);
    const device_cos = try harness.reserve(f32, table_elements);
    const device_sin = try harness.reserve(f32, table_elements);
    try launchRopeBuild(&harness.kernels, harness.stream.handle, .{
        .positions_t = device_positions,
        .positions_h = device_positions,
        .positions_w = device_positions,
        .frequencies = device_frequencies,
        .cos = device_cos,
        .sin = device_sin,
    }, shape);

    const elements: u32 = shape.tokens * shape.heads * shape.head_dim;
    const q_input = try allocator.alloc(f32, elements);
    defer allocator.free(q_input);
    const expected = try allocator.alloc(f32, elements);
    defer allocator.free(expected);
    const actual = try allocator.alloc(f32, elements);
    defer allocator.free(actual);
    fillRandom(q_input, 0x99_1, 3.0);
    for (0..shape.tokens) |token| {
        const table = token * shape.rotated;
        for (0..shape.heads) |head| {
            const base = (token * shape.heads + head) * shape.head_dim;
            ref_rope.apply(
                q_input[base..][0..shape.head_dim],
                cos_expected[table..][0..shape.rotated],
                sin_expected[table..][0..shape.rotated],
                expected[base..][0..shape.head_dim],
            );
        }
    }

    // The keys live in the same allocation the query does, so a kernel that
    // rotated them too would be caught here rather than at the model level.
    const device_q = try harness.upload(f32, q_input);
    const device_k = try harness.upload(f32, q_input);
    try launchRopeApplyQ(
        &harness.kernels,
        harness.stream.handle,
        device_q,
        device_cos,
        device_sin,
        shape,
    );
    try harness.download(f32, device_q, actual);
    try expectWithin("rope q only", compare(expected, actual), value_tolerance);
    const k_actual = try allocator.alloc(f32, elements);
    defer allocator.free(k_actual);
    try harness.download(f32, device_k, k_actual);
    for (q_input, k_actual) |before, after| {
        try std.testing.expectEqual(
            @as(u32, @bitCast(before)),
            @as(u32, @bitCast(after)),
        );
    }
    try harness.reset();
}

/// One hyper-connection case: upload, launch, download and compare both the
/// mixed stream and the injection weights against `hc.forward`.
fn checkHyperConnection(
    harness: *Harness,
    allocator: std.mem.Allocator,
    shape: HyperShape,
    seed: u64,
    label: []const u8,
) !void {
    shape.check();
    assert(label.len > 0);

    const hyper = shape.hyper();
    const mixed_elements: u32 = shape.tokens * shape.hidden;
    const injection_elements: u32 = shape.tokens * shape.streams;

    var config = try fixtures.load_config(allocator);
    config.hidden_size = shape.hidden;
    config.hc_count = shape.streams;
    config.hc_lowrank = shape.lowrank;
    config.max_position_embeddings = shape.tokens;
    const epsilon = config.rms_norm_eps;

    const input = try allocator.alloc(f32, shape.tokens * hyper);
    defer allocator.free(input);
    const norm_weight = try allocator.alloc(f32, hyper);
    defer allocator.free(norm_weight);
    const mix_down = try allocator.alloc(f32, shape.lowrank * hyper);
    defer allocator.free(mix_down);
    const mix_up = try allocator.alloc(f32, hyper * shape.lowrank);
    defer allocator.free(mix_up);
    const inject_weight = try allocator.alloc(f32, shape.streams * hyper);
    defer allocator.free(inject_weight);
    const expected_mixed = try allocator.alloc(f32, mixed_elements);
    defer allocator.free(expected_mixed);
    const expected_injection = try allocator.alloc(f32, injection_elements);
    defer allocator.free(expected_injection);
    const actual_mixed = try allocator.alloc(f32, mixed_elements);
    defer allocator.free(actual_mixed);
    const actual_injection = try allocator.alloc(f32, injection_elements);
    defer allocator.free(actual_injection);

    fillRandom(input, seed, 1.0);
    // The checkpoint initialises the HC norm weight to zeros, so a small
    // perturbation is the realistic case: `1 + w` stays near 1.
    fillRandom(norm_weight, seed +% 1, 0.05);
    const input_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hyper)));
    const lowrank_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(shape.lowrank)));
    fillRandom(mix_down, seed +% 2, input_scale);
    fillRandom(mix_up, seed +% 3, lowrank_scale);
    fillRandom(inject_weight, seed +% 4, input_scale);

    var scratch = try ref_hc.Scratch.init(allocator, &config);
    defer scratch.deinit(allocator);
    const weights = ref_hc.Weights{
        .norm = norm_weight,
        .mix_down = mix_down,
        .mix_up = mix_up,
        .inject = inject_weight,
    };
    ref_hc.forward(
        input,
        shape.tokens,
        expected_mixed,
        expected_injection,
        &weights,
        &scratch,
        &config,
    );

    const device_input = try harness.upload(f32, input);
    const device_norm = try harness.upload(f32, norm_weight);
    const device_down = try harness.upload(f32, mix_down);
    const device_up = try harness.upload(f32, mix_up);
    const device_inject = try harness.upload(f32, inject_weight);
    const device_mixed = try harness.reserve(f32, mixed_elements);
    const device_injection = try harness.reserve(f32, injection_elements);
    const tensors = HyperTensors{
        .input = device_input,
        .norm_weight = device_norm,
        .mix_down = device_down,
        .mix_up = device_up,
        .inject_weight = device_inject,
        .mixed = device_mixed,
        .injection = device_injection,
    };
    try launchHyperMix(
        &harness.kernels,
        harness.stream.handle,
        tensors,
        shape,
        epsilon,
    );
    try harness.download(f32, device_mixed, actual_mixed);
    try harness.download(f32, device_injection, actual_injection);
    try expectWithin(label, compare(expected_mixed, actual_mixed), value_tolerance);
    try expectWithin(
        "hc injection",
        compare(expected_injection, actual_injection),
        value_tolerance,
    );

    // Non-vacuity: `2 * sigmoid(...)` is strictly inside (0, 2), and a fixture
    // that saturated at one end would make the comparison above trivially easy.
    var lowest: f32 = 2.0;
    var highest: f32 = 0.0;
    for (actual_injection) |value| {
        lowest = @min(lowest, value);
        highest = @max(highest, value);
    }
    std.debug.print(
        "  {s:<16} injected weights in ({d:.6}, {d:.6})\n",
        .{ "hc range", lowest, highest },
    );
    try std.testing.expect(lowest > 0.0);
    try std.testing.expect(highest < 2.0);
    try std.testing.expect(highest - lowest > 0.001);

    // `use_combine = false`: the same kernel with no injection weight, which is
    // the final mixer before `lm_head`.
    const mixer_weights = ref_hc.Weights{
        .norm = norm_weight,
        .mix_down = mix_down,
        .mix_up = mix_up,
        .inject = null,
    };
    ref_hc.forward(
        input,
        shape.tokens,
        expected_mixed,
        null,
        &mixer_weights,
        &scratch,
        &config,
    );
    const mixer = HyperTensors{
        .input = device_input,
        .norm_weight = device_norm,
        .mix_down = device_down,
        .mix_up = device_up,
        .inject_weight = 0,
        .mixed = device_mixed,
        .injection = 0,
    };
    try launchHyperMix(
        &harness.kernels,
        harness.stream.handle,
        mixer,
        shape,
        epsilon,
    );
    try harness.download(f32, device_mixed, actual_mixed);
    try expectWithin("hc mixer", compare(expected_mixed, actual_mixed), value_tolerance);
    try harness.reset();
}

test "the hyper-connection block matches the oracle with and without injection" {
    var harness = try Harness.init();
    defer harness.deinit();
    const allocator = std.testing.allocator;

    // The tiny fixture's shape: two 128-dim streams and a 16-wide low rank.
    try checkHyperConnection(
        &harness,
        allocator,
        .{ .tokens = 3, .hidden = 128, .streams = 2, .lowrank = 16 },
        0x44_1,
        "hc tiny",
    );
    // The production shape: four 2560-dim streams (10240 wide) and a 320-wide
    // low rank, which is the 83 KiB shared-memory case.
    try checkHyperConnection(
        &harness,
        allocator,
        .{ .tokens = 2, .hidden = 2560, .streams = 4, .lowrank = 320 },
        0x44_2,
        "hc production",
    );
}

/// Host re-derivation of the sampler's retained-set definition.
///
/// The definition lives in `elementwise.cu`: the three filters are value
/// thresholds, so the retained set is a pure function of the scores and no
/// ordering enters it. This struct computes the same thresholds by sorting a
/// copy of the scores, which is the brute-force form of the same predicate.
const SamplerOracle = struct {
    scaled: []f32,
    retained: []bool,
    threshold: f32,
    maximum: f32,
    total_mass: f64,
    retained_mass: f64,

    fn init(
        allocator: std.mem.Allocator,
        logits: []const f32,
        history: []const u32,
        params: SampleParams,
    ) !SamplerOracle {
        params.check();
        const vocab: u32 = @intCast(logits.len);
        assert(vocab > 0);

        const scaled = try allocator.alloc(f32, vocab);
        errdefer allocator.free(scaled);
        const retained = try allocator.alloc(bool, vocab);
        errdefer allocator.free(retained);
        const sorted = try allocator.alloc(f32, vocab);
        defer allocator.free(sorted);

        @memcpy(scaled, logits);
        for (history, 0..) |id, index| {
            var first = true;
            for (history[0..index]) |earlier| {
                if (earlier == id) {
                    first = false;
                }
            }
            if (first) {
                const value = scaled[id];
                if (value > 0.0) {
                    scaled[id] = value / params.repetition_penalty;
                } else {
                    scaled[id] = value * params.repetition_penalty;
                }
            }
        }
        if (params.temperature > 0.0) {
            for (scaled) |*value| {
                value.* /= params.temperature;
            }
        }

        var maximum: f32 = scaled[0];
        for (scaled) |value| {
            maximum = @max(maximum, value);
        }
        @memcpy(sorted, scaled);
        std.mem.sort(f32, sorted, {}, descending);

        var total: f64 = 0.0;
        for (scaled) |value| {
            total += std.math.exp(@as(f64, value) - @as(f64, maximum));
        }
        assert(total >= 1.0);

        const kept = @min(params.top_k, vocab);
        var threshold: f32 = sorted[kept - 1];
        if (params.top_p < 1.0) {
            // The nucleus is the smallest value threshold whose normalised mass
            // reaches `top_p`; ties at the threshold are all retained.
            var mass: f64 = 0.0;
            var count: u32 = 0;
            while (count < vocab) : (count += 1) {
                mass += std.math.exp(
                    @as(f64, sorted[count]) - @as(f64, maximum),
                );
                if (mass >= @as(f64, params.top_p) * total) {
                    break;
                }
            }
            assert(count < vocab);
            threshold = @max(threshold, sorted[count]);
        }
        if (params.min_p > 0.0) {
            threshold = @max(threshold, maximum + @log(params.min_p));
        }

        var mass: f64 = 0.0;
        for (scaled, 0..) |value, index| {
            if (value >= threshold) {
                retained[index] = true;
                mass += std.math.exp(@as(f64, value) - @as(f64, maximum));
            } else {
                retained[index] = false;
            }
        }
        assert(mass >= 1.0);
        return .{
            .scaled = scaled,
            .retained = retained,
            .threshold = threshold,
            .maximum = maximum,
            .total_mass = total,
            .retained_mass = mass,
        };
    }

    fn deinit(self: *SamplerOracle, allocator: std.mem.Allocator) void {
        assert(self.scaled.len > 0);
        assert(self.retained.len == self.scaled.len);

        allocator.free(self.retained);
        allocator.free(self.scaled);
    }

    /// Log probability the kernel must report for `id` under the retained set:
    /// the log-softmax of `scaled` over that set alone, which is
    /// `(score - maximum) - log(sum exp(score - maximum))`.
    fn expectedLogprob(self: *const SamplerOracle, id: u32) f32 {
        assert(id < self.scaled.len);
        assert(self.retained_mass >= 1.0);

        const offset = @as(f64, self.scaled[id]) - @as(f64, self.maximum);
        return @floatCast(offset - @log(self.retained_mass));
    }

    /// Log probability the greedy path reports: the argmax's log-softmax over
    /// the whole vocabulary, because a point mass has no filtered set.
    fn expectedGreedyLogprob(self: *const SamplerOracle, id: u32) f32 {
        assert(id < self.scaled.len);
        assert(self.total_mass >= 1.0);

        const offset = @as(f64, self.scaled[id]) - @as(f64, self.maximum);
        assert(offset <= 1e-6);
        return @floatCast(offset - @log(self.total_mass));
    }

    /// Number of retained candidates.
    fn retainedCount(self: *const SamplerOracle) u32 {
        var count: u32 = 0;
        for (self.retained) |keep| {
            if (keep) {
                count += 1;
            }
        }
        assert(count > 0);
        return count;
    }
};

/// Descending f32 order, for the oracle's threshold by rank.
fn descending(_: void, a: f32, b: f32) bool {
    assert(a == a);
    assert(b == b);

    return a > b;
}

/// Result of one sampler launch.
const Draw = struct {
    id: u32,
    logprob: f32,
};

/// Runs one draw and copies the id and logprob back.
fn runDraw(
    harness: *Harness,
    tensors: SamplerTensors,
    vocab: u32,
    history_len: u32,
    params: SampleParams,
) !Draw {
    assert(vocab > 0);
    params.check();

    try launchSampler(
        &harness.kernels,
        harness.stream.handle,
        tensors,
        vocab,
        history_len,
        params,
    );
    var id = [_]i32{0};
    var logprob = [_]f32{0.0};
    try harness.download(i32, tensors.id, id[0..]);
    try harness.download(f32, tensors.logprob, logprob[0..]);
    try std.testing.expect(id[0] >= 0);
    const result = Draw{ .id = @intCast(id[0]), .logprob = logprob[0] };
    assert(result.id < vocab);
    return result;
}

/// Device tensors of one sampler fixture, allocated inside the harness arena.
const SamplerFixture = struct {
    tensors: SamplerTensors,
    vocab: u32,
    history_len: u32,

    fn init(
        harness: *Harness,
        logits: []const f32,
        history: []const u32,
    ) !SamplerFixture {
        assert(logits.len > 0);
        assert(logits.len <= std.math.maxInt(u32));

        const vocab: u32 = @intCast(logits.len);
        const device_logits = try harness.upload(f32, logits);
        const device_history = if (history.len > 0)
            try harness.upload(u32, history)
        else
            @as(cuda_driver.DevicePtr, 0);
        return .{
            .tensors = .{
                .logits = device_logits,
                .history = device_history,
                .scaled = try harness.reserve(f32, vocab),
                .id = try harness.reserve(i32, 1),
                .logprob = try harness.reserve(f32, 1),
            },
            .vocab = vocab,
            .history_len = @intCast(history.len),
        };
    }

    fn draw(
        self: *const SamplerFixture,
        harness: *Harness,
        params: SampleParams,
    ) !Draw {
        assert(self.vocab > 0);
        assert(self.history_len <= self.vocab);

        return runDraw(harness, self.tensors, self.vocab, self.history_len, params);
    }
};

/// Parameters with no filtering: the distribution the logits define.
fn unfilteredParams(temperature: f32, seed: u32, draw: u32, vocab: u32) SampleParams {
    assert(temperature >= 0.0);
    assert(vocab > 0);

    return .{
        .temperature = temperature,
        .top_k = vocab,
        .top_p = 1.0,
        .min_p = 0.0,
        .repetition_penalty = 1.0,
        .seed = seed,
        .draw = draw,
    };
}

test "the sampler replays a seed and samples the distribution it was given" {
    var harness = try Harness.init();
    defer harness.deinit();
    const allocator = std.testing.allocator;

    const vocab: u32 = 2048;
    const logits = try allocator.alloc(f32, vocab);
    defer allocator.free(logits);
    // A decaying ramp with the production vocabulary's width: the mass is
    // concentrated on the first few hundred ids, so both the replay check and
    // the distribution check below have something to measure.
    for (logits, 0..) |*value, index| {
        value.* = -0.05 * @as(f32, @floatFromInt(index));
    }

    const fixture = try SamplerFixture.init(&harness, logits, &[_]u32{});
    var oracle = try SamplerOracle.init(
        allocator,
        logits,
        &[_]u32{},
        unfilteredParams(1.0, 7, 0, vocab),
    );
    defer oracle.deinit(allocator);
    try std.testing.expectEqual(vocab, oracle.retainedCount());

    var first: [100]u32 = undefined;
    var replay: [100]u32 = undefined;
    for (0..100) |index| {
        const params = unfilteredParams(1.0, 1234, @intCast(index), vocab);
        first[index] = (try fixture.draw(&harness, params)).id;
    }
    for (0..100) |index| {
        const params = unfilteredParams(1.0, 1234, @intCast(index), vocab);
        replay[index] = (try fixture.draw(&harness, params)).id;
    }
    try std.testing.expectEqualSlices(u32, first[0..], replay[0..]);

    // A different seed has to move the sequence, otherwise the seed never
    // reaches the generator and the replay check above is vacuous.
    var moved: u32 = 0;
    for (0..100) |index| {
        const params = unfilteredParams(1.0, 4321, @intCast(index), vocab);
        const other = (try fixture.draw(&harness, params)).id;
        if (other != first[index]) {
            moved += 1;
        }
    }
    std.debug.print("  {s:<16} {d}/100 draws change with the seed\n", .{ "seed", moved });
    try std.testing.expect(moved > 50);

    // The issue's non-vacuity check: the id distribution over 1000 draws.
    const counts = try allocator.alloc(u32, vocab);
    defer allocator.free(counts);
    @memset(counts, 0);
    var total: f64 = 0.0;
    var moments: f64 = 0.0;
    for (0..1000) |index| {
        const params = unfilteredParams(1.0, 99, @intCast(index), vocab);
        const result = try fixture.draw(&harness, params);
        counts[result.id] += 1;
        total += 1.0;
        moments += @as(f64, @floatFromInt(result.id));
    }
    var distinct: u32 = 0;
    var busiest: u32 = 0;
    for (counts, 0..) |count, index| {
        if (count > 0) {
            distinct += 1;
        }
        if (count > counts[busiest]) {
            busiest = @intCast(index);
        }
    }
    std.debug.print(
        "  {s:<16} {d} distinct ids, busiest id {d} ({d}/1000)\n",
        .{ "distribution", distinct, busiest, counts[busiest] },
    );
    try std.testing.expect(distinct > 5);

    // The mean is the cheap summary that catches a sampler which draws from the
    // wrong set: 1000 draws put it within a few percent of the analytic mean.
    var reference_total: f64 = 0.0;
    var reference_moment: f64 = 0.0;
    for (logits, 0..) |value, index| {
        const probability = std.math.exp(@as(f64, value) - @as(f64, oracle.maximum));
        reference_total += probability;
        reference_moment += probability * @as(f64, @floatFromInt(index));
    }
    const empirical_mean = moments / total;
    const reference_mean = reference_moment / reference_total;
    std.debug.print(
        "  {s:<16} empirical mean {d:.3}, analytic {d:.3}\n",
        .{ "mean", empirical_mean, reference_mean },
    );
    try std.testing.expect(@abs(empirical_mean - reference_mean) / reference_mean < 0.1);
    try harness.reset();
}

test "the greedy path is argmax, and a tied maximum keeps the lowest id" {
    var harness = try Harness.init();
    defer harness.deinit();
    const allocator = std.testing.allocator;

    const vocab: u32 = 512;
    const logits = try allocator.alloc(f32, vocab);
    defer allocator.free(logits);
    fillRandom(logits, 0x63_1, 1.0);
    // Two exact maxima: the tie has to resolve to the lower id, not to whichever
    // element the reduction happens to visit first.
    logits[100] = 5.0;
    logits[500] = 5.0;
    logits[511] = 5.0;

    const fixture = try SamplerFixture.init(&harness, logits, &[_]u32{});
    var oracle = try SamplerOracle.init(
        allocator,
        logits,
        &[_]u32{},
        unfilteredParams(0.0, 1, 0, vocab),
    );
    defer oracle.deinit(allocator);

    // Greedy: every seed and every draw index gives the same token.
    for (0..8) |index| {
        const params = SampleParams{
            .temperature = 0.0,
            .top_k = 1,
            .top_p = 0.0,
            .min_p = 0.0,
            .repetition_penalty = 1.0,
            .seed = @intCast(index + 1),
            .draw = @intCast(index),
        };
        const result = try fixture.draw(&harness, params);
        try std.testing.expectEqual(@as(u32, 100), result.id);
        try expectClose(
            "greedy logprob",
            oracle.expectedGreedyLogprob(result.id),
            result.logprob,
            value_tolerance,
        );
    }

    // `top_k = 1` at temperature 1 is the same token, from the filtered path.
    for (0..8) |index| {
        const params = SampleParams{
            .temperature = 1.0,
            .top_k = 1,
            .top_p = 1.0,
            .min_p = 0.0,
            .repetition_penalty = 1.0,
            .seed = @intCast(index + 1),
            .draw = @intCast(index),
        };
        const result = try fixture.draw(&harness, params);
        // The three tied maxima share one second-order candidate set, so the
        // draw lands inside it and nowhere else.
        try std.testing.expect(logits[result.id] == 5.0);
    }

    // A unique maximum, so `top_k = 1` has exactly one candidate.
    logits[100] = 6.0;
    try harness.write(fixture.tensors.logits, f32, logits);
    for (0..8) |index| {
        const params = SampleParams{
            .temperature = 1.0,
            .top_k = 1,
            .top_p = 1.0,
            .min_p = 0.0,
            .repetition_penalty = 1.0,
            .seed = @intCast(index + 1),
            .draw = @intCast(index),
        };
        const result = try fixture.draw(&harness, params);
        try std.testing.expectEqual(@as(u32, 100), result.id);
    }
    try harness.reset();
}

test "top_k, top_p and min_p keep exactly the host-computed sets" {
    var harness = try Harness.init();
    defer harness.deinit();
    const allocator = std.testing.allocator;

    const vocab: u32 = 2048;
    const logits = try allocator.alloc(f32, vocab);
    defer allocator.free(logits);
    for (logits, 0..) |*value, index| {
        value.* = -0.02 * @as(f32, @floatFromInt(index));
    }

    const fixture = try SamplerFixture.init(&harness, logits, &[_]u32{});
    const cases = [_]SampleParams{
        .{
            .temperature = 1.0,
            .top_k = 3,
            .top_p = 1.0,
            .min_p = 0.0,
            .repetition_penalty = 1.0,
            .seed = 5,
            .draw = 0,
        },
        .{
            .temperature = 1.0,
            .top_k = vocab,
            .top_p = 0.5,
            .min_p = 0.0,
            .repetition_penalty = 1.0,
            .seed = 5,
            .draw = 0,
        },
        .{
            .temperature = 1.0,
            .top_k = vocab,
            .top_p = 1.0,
            .min_p = 0.9,
            .repetition_penalty = 1.0,
            .seed = 5,
            .draw = 0,
        },
        .{
            .temperature = 2.0,
            .top_k = vocab,
            .top_p = 0.9,
            .min_p = 0.0,
            .repetition_penalty = 1.0,
            .seed = 5,
            .draw = 0,
        },
    };
    for (cases) |params| {
        var oracle = try SamplerOracle.init(allocator, logits, &[_]u32{}, params);
        defer oracle.deinit(allocator);
        const retained = oracle.retainedCount();
        std.debug.print(
            "  {s:<16} k {d} p {d:.2} min {d:.2} -> {d} retained\n",
            .{
                "filter",
                params.top_k,
                params.top_p,
                params.min_p,
                retained,
            },
        );
        // The retained set is a prefix of the ramp, so its size is the whole
        // story; every draw must land inside it, and the logprob must match the
        // re-computed log-softmax of that set.
        for (0..16) |index| {
            var drawn = params;
            drawn.draw = @intCast(index);
            drawn.seed = @intCast(index + 11);
            const result = try fixture.draw(&harness, drawn);
            try std.testing.expect(oracle.retained[result.id]);
            try expectClose(
                "filter logprob",
                oracle.expectedLogprob(result.id),
                result.logprob,
                value_tolerance,
            );
        }
        // Non-vacuity: not every id may be retained, or the case proves nothing.
        try std.testing.expect(retained < vocab);
    }
    try harness.reset();
}

test "the repetition penalty is applied once per distinct id" {
    var harness = try Harness.init();
    defer harness.deinit();
    const allocator = std.testing.allocator;

    const vocab: u32 = 256;
    const logits = try allocator.alloc(f32, vocab);
    defer allocator.free(logits);
    fillRandom(logits, 0x87_1, 1.0);
    logits[7] = 4.0;
    logits[3] = 3.0;

    const history = [_]u32{ 7, 7, 7 };
    const fixture = try SamplerFixture.init(&harness, logits, history[0..]);
    var oracle = try SamplerOracle.init(allocator, logits, history[0..], .{
        .temperature = 0.0,
        .top_k = 1,
        .top_p = 1.0,
        .min_p = 0.0,
        .repetition_penalty = 4.0,
        .seed = 3,
        .draw = 0,
    });
    defer oracle.deinit(allocator);

    const params = SampleParams{
        .temperature = 0.0,
        .top_k = 1,
        .top_p = 1.0,
        .min_p = 0.0,
        .repetition_penalty = 4.0,
        .seed = 3,
        .draw = 0,
    };
    const penalised = try fixture.draw(&harness, params);
    // 4.0 / 4.0 drops id 7 below id 3, which carries no penalty.
    try std.testing.expectEqual(@as(u32, 3), penalised.id);

    // A single occurrence divides once too: three occurrences must not compound
    // the penalty into 4^3.
    const once = [_]u32{7};
    const single = try SamplerFixture.init(&harness, logits, once[0..]);
    const repeated = try fixture.draw(&harness, params);
    const singular = try single.draw(&harness, params);
    try std.testing.expectEqual(singular.id, repeated.id);
    // The penalty case samples greedily, so the reported logprob is the
    // argmax's log-softmax over the whole (penalised) vocabulary.
    try std.testing.expect(penalised.logprob <= 0.0);
    try expectClose(
        "penalty logprob",
        oracle.expectedGreedyLogprob(penalised.id),
        penalised.logprob,
        value_tolerance,
    );
    try harness.reset();
}
