//! Launch wrappers for `model/glue.cu`, and their oracle checks.
//!
//! The model graph needs a handful of operations the block kernels leave to
//! their caller: the dtype cast at every block boundary, the KV row gather the
//! attention kernel consumes, the hyper-connection residual injection, the
//! attention output gate, the routed/shared MoE combine, and the Gated
//! DeltaNet l2-normalise and beta/decay steps. Each is small enough that the
//! f32 CPU expression is its oracle; the test at the bottom runs every kernel
//! against that expression on the same inputs (`.agents/tasks/review-gate.md`
//! section 5).

const std = @import("std");

const cuda_context = @import("../hw/cuda_context.zig");
const cuda_driver = @import("../hw/cuda_driver.zig");
const generated_cubins = @import("generated_cubins");
const kernel = @import("../hw/kernel.zig");
const quant = @import("../weights/quant.zig");
const ref_ops = @import("../ref/ops.zig");

const assert = std.debug.assert;

/// Cubin compiled from `src/model/glue.cu`.
const cubin_name = "glue";

/// Threads per glue block; mirrors `glue_threads` in the device file.
pub const glue_threads: u32 = 256;

/// Bytes one device allocation rounds up to.
const allocation_granularity: u64 = 4096;

const KernelError = kernel.KernelError || cuda_driver.DriverError ||
    error{GlueCubinMissing};

/// Finds the `glue` cubin in the generated table.
fn cubinImage() ![]const u8 {
    assert(cubin_name.len > 0);
    assert(generated_cubins.cubins.len > 0);

    for (generated_cubins.cubins) |entry| {
        if (std.mem.eql(u8, std.mem.span(entry.name), cubin_name)) {
            const bytes = entry.bytes[0..entry.length];
            assert(bytes.len > 0);
            return bytes;
        } else {
            continue;
        }
    }
    return error.GlueCubinMissing;
}

/// Every entry point of the model glue cubin.
pub const Kernels = struct {
    module: kernel.Module,
    cast_f32_bf16: kernel.Function,
    gather_bf16: kernel.Function,
    inject: kernel.Function,
    sigmoid_mul: kernel.Function,
    gdn_prepare: kernel.Function,
    gdn_decay: kernel.Function,
    moe_combine: kernel.Function,

    /// Loads the cubin and resolves every entry point; the context must be
    /// current.
    pub fn load() KernelError!Kernels {
        assert(cubin_name.len > 0);

        const module = try kernel.Module.load(try cubinImage());
        errdefer module.unload();
        return .{
            .module = module,
            .cast_f32_bf16 = try module.function("model_cast_f32_bf16"),
            .gather_bf16 = try module.function("model_gather_bf16"),
            .inject = try module.function("model_inject"),
            .sigmoid_mul = try module.function("model_sigmoid_mul"),
            .gdn_prepare = try module.function("model_gdn_prepare"),
            .gdn_decay = try module.function("model_gdn_decay"),
            .moe_combine = try module.function("model_moe_combine"),
        };
    }

    /// Unloads the module; the context must still be current.
    pub fn deinit(kernels: *const Kernels) void {
        assert(kernels.module.handle != null);
        assert(@intFromPtr(kernels.moe_combine.handle) != 0);

        kernels.module.unload();
    }
};

/// Grid for a flat elementwise launch over `count` items.
fn flatGrid(count: u64) u32 {
    assert(count > 0);
    assert(glue_threads > 0);

    const blocks = (count + glue_threads - 1) / glue_threads;
    return @intCast(@min(blocks, 65535));
}

/// Rounds `bytes` up to a whole device allocation.
pub fn allocationBytes(bytes: u64) u64 {
    assert(bytes > 0);
    assert(allocation_granularity > 0);

    return (bytes + allocation_granularity - 1) / allocation_granularity *
        allocation_granularity;
}

/// Launches `model_cast_f32_bf16`.
pub fn launchCastF32Bf16(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    source: cuda_driver.DevicePtr,
    destination: cuda_driver.DevicePtr,
    count: u32,
) KernelError!void {
    assert(source != 0);
    assert(destination != 0);
    assert(count > 0);

    var from = source;
    var to = destination;
    var elements: i32 = @intCast(count);
    const arguments = [_]?*anyopaque{
        @ptrCast(&from), @ptrCast(&to), @ptrCast(&elements),
    };
    try kernels.cast_f32_bf16.launch(
        .{ flatGrid(count), 1, 1 },
        .{ glue_threads, 1, 1 },
        0,
        stream,
        arguments[0..],
    );
}

/// Launches `model_gather_bf16`: `destination[row] = source[index[row]]`, with
/// a zero row for a negative index.
pub fn launchGatherBf16(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    source: cuda_driver.DevicePtr,
    index: cuda_driver.DevicePtr,
    count: u32,
    width: u32,
    destination: cuda_driver.DevicePtr,
) KernelError!void {
    assert(source != 0);
    assert(index != 0);
    assert(destination != 0);
    assert(count > 0);
    assert(width > 0);

    var from = source;
    var rows = index;
    var to = destination;
    var row_count: i32 = @intCast(count);
    var row_width: i32 = @intCast(width);
    const arguments = [_]?*anyopaque{
        @ptrCast(&from),
        @ptrCast(&rows),
        @ptrCast(&row_count),
        @ptrCast(&row_width),
        @ptrCast(&to),
    };
    try kernels.gather_bf16.launch(
        .{ flatGrid(@as(u64, count) * width), 1, 1 },
        .{ glue_threads, 1, 1 },
        0,
        stream,
        arguments[0..],
    );
}

/// Launches `model_inject`: the hyper-connection residual update.
pub fn launchInject(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    block: cuda_driver.DevicePtr,
    injection: cuda_driver.DevicePtr,
    hyper: cuda_driver.DevicePtr,
    tokens: u32,
    hidden: u32,
    streams: u32,
) KernelError!void {
    assert(block != 0);
    assert(injection != 0);
    assert(hyper != 0);
    assert(tokens > 0);
    assert(hidden > 0);
    assert(streams > 0);

    var source = block;
    var weights = injection;
    var target = hyper;
    var token_count: i32 = @intCast(tokens);
    var width: i32 = @intCast(hidden);
    var stream_count: i32 = @intCast(streams);
    const arguments = [_]?*anyopaque{
        @ptrCast(&source),      @ptrCast(&weights), @ptrCast(&target),
        @ptrCast(&token_count), @ptrCast(&width),   @ptrCast(&stream_count),
    };
    const per_token = @as(u64, hidden) * streams;
    const blocks: u32 = @intCast(@min(
        (per_token + glue_threads - 1) / glue_threads,
        65535,
    ));
    try kernels.inject.launch(
        .{ blocks, tokens, 1 },
        .{ glue_threads, 1, 1 },
        0,
        stream,
        arguments[0..],
    );
}

/// Launches `model_sigmoid_mul`.
pub fn launchSigmoidMul(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    values: cuda_driver.DevicePtr,
    gate: cuda_driver.DevicePtr,
    out: cuda_driver.DevicePtr,
    count: u32,
) KernelError!void {
    assert(values != 0);
    assert(gate != 0);
    assert(out != 0);
    assert(count > 0);

    var source = values;
    var gate_arg = gate;
    var target = out;
    var elements: i32 = @intCast(count);
    const arguments = [_]?*anyopaque{
        @ptrCast(&source),   @ptrCast(&gate_arg), @ptrCast(&target),
        @ptrCast(&elements),
    };
    try kernels.sigmoid_mul.launch(
        .{ flatGrid(count), 1, 1 },
        .{ glue_threads, 1, 1 },
        0,
        stream,
        arguments[0..],
    );
}

/// Launches `model_gdn_prepare`.
pub fn launchGdnPrepare(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    conv_out: cuda_driver.DevicePtr,
    query: cuda_driver.DevicePtr,
    key: cuda_driver.DevicePtr,
    value: cuda_driver.DevicePtr,
    tokens: u32,
    k_heads: u32,
    v_heads: u32,
    k_head: u32,
    v_head: u32,
    q_scale: f32,
    eps: f32,
) KernelError!void {
    assert(conv_out != 0);
    assert(query != 0);
    assert(key != 0);
    assert(value != 0);
    assert(tokens > 0);
    assert(k_heads > 0);
    assert(v_heads > 0);
    assert(k_head > 0);
    assert(v_head > 0);
    assert(eps > 0);

    var conv = conv_out;
    var q = query;
    var k = key;
    var v = value;
    var token_count: i32 = @intCast(tokens);
    var key_heads: i32 = @intCast(k_heads);
    var value_heads: i32 = @intCast(v_heads);
    var key_dim: i32 = @intCast(k_head);
    var value_dim: i32 = @intCast(v_head);
    var scale: f32 = q_scale;
    var epsilon: f32 = eps;
    const arguments = [_]?*anyopaque{
        @ptrCast(&conv),        @ptrCast(&q),           @ptrCast(&k),
        @ptrCast(&v),           @ptrCast(&token_count), @ptrCast(&key_heads),
        @ptrCast(&value_heads), @ptrCast(&key_dim),     @ptrCast(&value_dim),
        @ptrCast(&scale),       @ptrCast(&epsilon),
    };
    const rows = tokens * (2 * k_heads + v_heads);
    try kernels.gdn_prepare.launch(
        .{ rows, 1, 1 },
        .{ glue_threads, 1, 1 },
        0,
        stream,
        arguments[0..],
    );
}

/// Launches `model_gdn_decay`.
pub fn launchGdnDecay(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    proj_a: cuda_driver.DevicePtr,
    proj_b: cuda_driver.DevicePtr,
    a_log: cuda_driver.DevicePtr,
    dt_bias: cuda_driver.DevicePtr,
    beta: cuda_driver.DevicePtr,
    decay: cuda_driver.DevicePtr,
    tokens: u32,
    heads: u32,
) KernelError!void {
    assert(proj_a != 0);
    assert(proj_b != 0);
    assert(a_log != 0);
    assert(dt_bias != 0);
    assert(beta != 0);
    assert(decay != 0);
    assert(tokens > 0);
    assert(heads > 0);
    assert(heads <= 1024);

    var a = proj_a;
    var b = proj_b;
    var log = a_log;
    var bias = dt_bias;
    var beta_out = beta;
    var decay_out = decay;
    var token_count: i32 = @intCast(tokens);
    var head_count: i32 = @intCast(heads);
    const arguments = [_]?*anyopaque{
        @ptrCast(&a),           @ptrCast(&b),          @ptrCast(&log),
        @ptrCast(&bias),        @ptrCast(&beta_out),   @ptrCast(&decay_out),
        @ptrCast(&token_count), @ptrCast(&head_count),
    };
    try kernels.gdn_decay.launch(
        .{ tokens, 1, 1 },
        .{ glue_threads, 1, 1 },
        0,
        stream,
        arguments[0..],
    );
}

/// Launches `model_moe_combine`.
pub fn launchMoeCombine(
    kernels: *const Kernels,
    stream: kernel.StreamHandle,
    expert_out: cuda_driver.DevicePtr,
    indices: cuda_driver.DevicePtr,
    router_weights: cuda_driver.DevicePtr,
    shared_out: cuda_driver.DevicePtr,
    gate_logit: cuda_driver.DevicePtr,
    x: cuda_driver.DevicePtr,
    out: cuda_driver.DevicePtr,
    tokens: u32,
    hidden: u32,
    top_k: u32,
) KernelError!void {
    assert(expert_out != 0);
    assert(indices != 0);
    assert(router_weights != 0);
    assert(shared_out != 0);
    assert(gate_logit != 0);
    assert(x != 0);
    assert(out != 0);
    assert(tokens > 0);
    assert(hidden > 0);
    assert(top_k > 0);

    var experts = expert_out;
    var slots = indices;
    var weights = router_weights;
    var shared = shared_out;
    var logit = gate_logit;
    var input = x;
    var destination = out;
    var token_count: i32 = @intCast(tokens);
    var width: i32 = @intCast(hidden);
    var top: i32 = @intCast(top_k);
    const arguments = [_]?*anyopaque{
        @ptrCast(&experts),     @ptrCast(&slots),       @ptrCast(&weights),
        @ptrCast(&shared),      @ptrCast(&logit),       @ptrCast(&input),
        @ptrCast(&destination), @ptrCast(&token_count), @ptrCast(&width),
        @ptrCast(&top),
    };
    try kernels.moe_combine.launch(
        .{ tokens, 1, 1 },
        .{ glue_threads, 1, 1 },
        0,
        stream,
        arguments[0..],
    );
}

// ---------------------------------------------------------------------------
// Oracle checks. The f32 expression each kernel computes is the oracle.
// ---------------------------------------------------------------------------

/// One device, its stream, the glue cubin and every allocation the test makes.
const TestGpu = struct {
    context: cuda_context.Context,
    stream: cuda_context.Stream,
    kernels: Kernels,
    allocations: std.ArrayList(cuda_context.DeviceBuffer),

    fn open() !TestGpu {
        var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
        const count = try cuda_driver.query_devices(devices[0..]);
        assert(count > 0);

        var gpu = TestGpu{
            .context = try cuda_context.Context.retain(&devices[0]),
            .stream = undefined,
            .kernels = undefined,
            .allocations = std.ArrayList(cuda_context.DeviceBuffer).init(
                std.testing.allocator,
            ),
        };
        try gpu.context.set_current();
        gpu.stream = try cuda_context.Stream.create();
        gpu.kernels = try Kernels.load();
        return gpu;
    }

    fn close(self: *TestGpu) void {
        assert(self.allocations.items.len > 0);
        assert(@intFromPtr(self.stream.handle) != 0);

        for (self.allocations.items) |buffer| {
            buffer.free();
        }
        self.allocations.deinit();
        self.kernels.deinit();
        self.stream.destroy();
        self.context.release();
    }

    fn allocate(self: *TestGpu, bytes: u64) !cuda_context.DeviceBuffer {
        assert(bytes > 0);

        const buffer = try cuda_context.DeviceBuffer.allocate(
            allocationBytes(bytes),
        );
        try self.allocations.append(buffer);
        return buffer;
    }

    fn upload(self: *TestGpu, bytes: []const u8) !cuda_context.DeviceBuffer {
        assert(bytes.len > 0);

        const buffer = try self.allocate(bytes.len);
        const staged = try cuda_context.PinnedHostBuffer.allocate(buffer.bytes);
        defer staged.free();
        @memset(staged.slice(), 0);
        @memcpy(staged.slice()[0..bytes.len], bytes);
        try self.stream.copy_h_to_d(&buffer, &staged, buffer.bytes);
        try self.stream.synchronize();
        return buffer;
    }

    fn download(self: *TestGpu, source: *const cuda_context.DeviceBuffer, out: []u8) !void {
        assert(out.len > 0);
        assert(out.len <= source.bytes);

        const staged = try cuda_context.PinnedHostBuffer.allocate(
            allocationBytes(out.len),
        );
        defer staged.free();
        try self.stream.copy_d_to_h(&staged, source, out.len);
        try self.stream.synchronize();
        @memcpy(out, staged.slice()[0..out.len]);
    }
};

/// Uploads f32 values into a fresh device buffer.
fn uploadF32(gpu: *TestGpu, values: []const f32) !cuda_context.DeviceBuffer {
    assert(values.len > 0);
    return gpu.upload(std.mem.sliceAsBytes(values));
}

/// Uploads bf16 bits into a fresh device buffer.
fn uploadBf16(gpu: *TestGpu, values: []const u16) !cuda_context.DeviceBuffer {
    assert(values.len > 0);
    return gpu.upload(std.mem.sliceAsBytes(values));
}

/// Downloads f32 values from `source`.
fn downloadF32(
    gpu: *TestGpu,
    source: *const cuda_context.DeviceBuffer,
    out: []f32,
) !void {
    assert(out.len > 0);
    try gpu.download(source, std.mem.sliceAsBytes(out));
}

/// Downloads bf16 bits from `source`.
fn downloadBf16(
    gpu: *TestGpu,
    source: *const cuda_context.DeviceBuffer,
    out: []u16,
) !void {
    assert(out.len > 0);
    try gpu.download(source, std.mem.sliceAsBytes(out));
}

/// A small deterministic generator, so a failure reproduces.
const Generator = struct {
    state: u32,

    fn next(self: *Generator) u32 {
        assert(self.state != 0);
        self.state = self.state *% 1_664_525 +% 1_013_904_223;
        return self.state;
    }

    fn signed(self: *Generator) f32 {
        const raw: i32 = @bitCast(self.next());
        return @as(f32, @floatFromInt(raw >> 8)) / 8_388_608.0;
    }

    fn bf16(self: *Generator) u16 {
        return quant.f32_to_bf16(self.signed() * 2.0);
    }
};

test "the glue kernels match their f32 expressions" {
    var gpu = try TestGpu.open();
    defer gpu.close();
    var generator = Generator{ .state = 0x51ce_9001 };

    // 1. f32 -> bf16 cast.
    var source: [257]f32 = undefined;
    for (&source) |*value| {
        value.* = generator.signed() * 8.0;
    }
    const cast_source = try uploadF32(&gpu, &source);
    const cast_output = try gpu.allocate(source.len * 2);
    try launchCastF32Bf16(
        &gpu.kernels,
        gpu.stream.handle,
        cast_source.pointer,
        cast_output.pointer,
        source.len,
    );
    var cast_bits: [257]u16 = undefined;
    try downloadBf16(&gpu, &cast_output, &cast_bits);
    for (source, cast_bits) |value, bits| {
        try std.testing.expectEqual(quant.f32_to_bf16(value), bits);
    }

    // 2. bf16 row gather, with one negative padding slot.
    const rows: u32 = 5;
    const width: u32 = 7;
    var matrix: [rows * width]u16 = undefined;
    for (&matrix) |*bits| {
        bits.* = generator.bf16();
    }
    const index = [_]i32{ 4, -1, 0, 3, -1, 2 };
    const matrix_device = try uploadBf16(&gpu, &matrix);
    const index_buffer = try gpu.upload(std.mem.sliceAsBytes(&index));
    const gather_output = try gpu.allocate(index.len * width * 2);
    try launchGatherBf16(
        &gpu.kernels,
        gpu.stream.handle,
        matrix_device.pointer,
        index_buffer.pointer,
        index.len,
        width,
        gather_output.pointer,
    );
    var gathered: [index.len * width]u16 = undefined;
    try downloadBf16(&gpu, &gather_output, &gathered);
    for (index, 0..) |row, slot| {
        for (0..width) |column| {
            const expected: u16 = if (row < 0)
                0
            else
                matrix[@as(usize, @intCast(row)) * width + column];
            try std.testing.expectEqual(expected, gathered[slot * width + column]);
        }
    }

    // 3. residual injection.
    const tokens: u32 = 3;
    const hidden: u32 = 8;
    const streams: u32 = 2;
    const hyper_size: u32 = tokens * hidden * streams;
    const block_values = [_]f32{ 1, -2, 3, -4, 5, -6, 7, -8, 2, 4, -6, 8, -10, 12, -14, 16, 1, 1, 1, 1, 1, 1, 1, 1 };
    const injection_values = [_]f32{ 0.5, 1.5, -0.25, 2.0, 0.75, -1.0 };
    var hyper_values: [hyper_size]f32 = undefined;
    for (&hyper_values) |*value| {
        value.* = generator.signed();
    }
    const block_device = try uploadF32(&gpu, &block_values);
    const injection_device = try uploadF32(&gpu, &injection_values);
    const hyper_device = try uploadF32(&gpu, &hyper_values);
    try launchInject(
        &gpu.kernels,
        gpu.stream.handle,
        block_device.pointer,
        injection_device.pointer,
        hyper_device.pointer,
        tokens,
        hidden,
        streams,
    );
    var injected: [hyper_size]f32 = undefined;
    try downloadF32(&gpu, &hyper_device, &injected);
    for (0..tokens) |t| {
        for (0..streams) |c| {
            for (0..hidden) |h| {
                const expected = hyper_values[t * hidden * streams + c * hidden + h] +
                    block_values[t * hidden + h] * injection_values[t * streams + c];
                try std.testing.expectApproxEqAbs(
                    expected,
                    injected[t * hidden * streams + c * hidden + h],
                    1e-6,
                );
            }
        }
    }

    // 4. attention output gate.
    var values: [64]f32 = undefined;
    var gates: [64]f32 = undefined;
    for (&values, &gates) |*value, *gate| {
        value.* = generator.signed() * 4.0;
        gate.* = generator.signed() * 6.0;
    }
    const values_device = try uploadF32(&gpu, &values);
    const gates_device = try uploadF32(&gpu, &gates);
    const gated_device = try gpu.allocate(values.len * 4);
    try launchSigmoidMul(
        &gpu.kernels,
        gpu.stream.handle,
        values_device.pointer,
        gates_device.pointer,
        gated_device.pointer,
        values.len,
    );
    var gated: [64]f32 = undefined;
    try downloadF32(&gpu, &gated_device, &gated);
    for (values, gates, gated) |value, gate, got| {
        try std.testing.expectApproxEqAbs(
            value * ref_ops.sigmoid(gate),
            got,
            1e-6,
        );
    }
}

test "the Gated DeltaNet preprocessing kernels match the reference" {
    var gpu = try TestGpu.open();
    defer gpu.close();
    var generator = Generator{ .state = 0x51ce_9002 };

    const tokens: u32 = 3;
    const k_heads: u32 = 2;
    const v_heads: u32 = 6;
    const k_head: u32 = 16;
    const v_head: u32 = 16;
    const key_dim = k_heads * k_head;
    const value_dim = v_heads * v_head;
    const conv_dim = 2 * key_dim + value_dim;
    var conv: [tokens * conv_dim]u16 = undefined;
    for (&conv) |*bits| {
        bits.* = generator.bf16();
    }
    const conv_device = try uploadBf16(&gpu, &conv);
    const query_device = try gpu.allocate(tokens * key_dim * 4);
    const key_device = try gpu.allocate(tokens * key_dim * 4);
    const value_device = try gpu.allocate(tokens * value_dim * 4);
    const q_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(k_head)));
    const eps: f32 = 1e-6;
    try launchGdnPrepare(
        &gpu.kernels,
        gpu.stream.handle,
        conv_device.pointer,
        query_device.pointer,
        key_device.pointer,
        value_device.pointer,
        tokens,
        k_heads,
        v_heads,
        k_head,
        v_head,
        q_scale,
        eps,
    );
    var query: [tokens * key_dim]f32 = undefined;
    var key: [tokens * key_dim]f32 = undefined;
    var value: [tokens * value_dim]f32 = undefined;
    try downloadF32(&gpu, &query_device, &query);
    try downloadF32(&gpu, &key_device, &key);
    try downloadF32(&gpu, &value_device, &value);
    var row_input: [k_head]f32 = undefined;
    var row_expected: [k_head]f32 = undefined;
    for (0..tokens) |t| {
        for (0..k_heads) |head| {
            const src = t * conv_dim + head * k_head;
            for (0..k_head) |dim| {
                row_input[dim] = quant.bf16_to_f32(conv[src + dim]);
            }
            ref_ops.l2norm(&row_input, &row_expected, eps);
            for (0..k_head) |dim| {
                const want = row_expected[dim] * q_scale;
                const got = query[(t * k_heads + head) * k_head + dim];
                if (@abs(want - got) > 1e-6) {
                    std.debug.print("query t={d} h={d} d={d} want={e} got={e}\n", .{
                        t, head, dim, want, got,
                    });
                }
                try std.testing.expectApproxEqAbs(want, got, 1e-6);
            }
            const key_src = t * conv_dim + key_dim + head * k_head;
            for (0..k_head) |dim| {
                row_input[dim] = quant.bf16_to_f32(conv[key_src + dim]);
            }
            ref_ops.l2norm(&row_input, &row_expected, eps);
            for (0..k_head) |dim| {
                const want = row_expected[dim];
                const got = key[(t * k_heads + head) * k_head + dim];
                if (@abs(want - got) > 1e-6) {
                    var squares: f32 = 0.0;
                    for (row_input) |item| squares += item * item;
                    std.debug.print(
                        "key t={d} h={d} d={d} want={e} got={e} sum={e} x0={e} x1={e}\n",
                        .{
                            t,
                            head,
                            dim,
                            want,
                            got,
                            squares,
                            row_input[0],
                            row_input[1],
                        },
                    );
                }
                try std.testing.expectApproxEqAbs(want, got, 1e-6);
            }
        }
        for (0..v_heads) |head| {
            const src = t * conv_dim + 2 * key_dim + head * v_head;
            for (0..v_head) |dim| {
                try std.testing.expectEqual(
                    quant.bf16_to_f32(conv[src + dim]),
                    value[(t * v_heads + head) * v_head + dim],
                );
            }
        }
    }

    // The gates.
    const heads: u32 = 6;
    var proj_a: [tokens * heads]f32 = undefined;
    var proj_b: [tokens * heads]f32 = undefined;
    var a_log: [heads]f32 = undefined;
    var dt_bias: [heads]f32 = undefined;
    for (&proj_a, &proj_b) |*a, *b| {
        a.* = generator.signed() * 3.0;
        b.* = generator.signed() * 3.0;
    }
    for (&a_log, &dt_bias) |*a, *bias| {
        a.* = generator.signed() * 2.0;
        bias.* = generator.signed() * 0.5;
    }
    const a_device = try uploadF32(&gpu, &proj_a);
    const b_device = try uploadF32(&gpu, &proj_b);
    const log_device = try uploadF32(&gpu, &a_log);
    const bias_device = try uploadF32(&gpu, &dt_bias);
    const beta_device = try gpu.allocate(proj_a.len * 4);
    const decay_device = try gpu.allocate(proj_a.len * 4);
    try launchGdnDecay(
        &gpu.kernels,
        gpu.stream.handle,
        a_device.pointer,
        b_device.pointer,
        log_device.pointer,
        bias_device.pointer,
        beta_device.pointer,
        decay_device.pointer,
        tokens,
        heads,
    );
    var beta: [tokens * heads]f32 = undefined;
    var decay: [tokens * heads]f32 = undefined;
    try downloadF32(&gpu, &beta_device, &beta);
    try downloadF32(&gpu, &decay_device, &decay);
    for (0..tokens) |t| {
        for (0..heads) |head| {
            const index = t * heads + head;
            try std.testing.expectApproxEqAbs(
                ref_ops.sigmoid(proj_b[index]),
                beta[index],
                1e-6,
            );
            const expected = -@exp(a_log[head]) *
                ref_ops.softplus(proj_a[index] + dt_bias[head]);
            if (@abs(expected - decay[index]) > 1e-6) {
                std.debug.print(
                    "decay t={d} head={d} a={e} log={e} bias={e} want={e} got={e}\n",
                    .{
                        t,
                        head,
                        proj_a[index],
                        a_log[head],
                        dt_bias[head],
                        expected,
                        decay[index],
                    },
                );
            }
            try std.testing.expectApproxEqAbs(expected, decay[index], 1e-6);
        }
    }
}

test "the MoE combine matches the reference expression" {
    var gpu = try TestGpu.open();
    defer gpu.close();
    var generator = Generator{ .state = 0x51ce_9003 };

    const tokens: u32 = 4;
    const hidden: u32 = 8;
    const experts: u32 = 3;
    const top_k: u32 = 2;
    var expert_out: [experts * tokens * hidden]f32 = undefined;
    var shared: [tokens * hidden]f32 = undefined;
    var x: [tokens * hidden]f32 = undefined;
    var gate_logit: [hidden]f32 = undefined;
    for (&expert_out) |*expert| {
        expert.* = generator.signed();
    }
    for (&shared, &x) |*shared_value, *input| {
        shared_value.* = generator.signed();
        input.* = generator.signed();
    }
    for (&gate_logit) |*value| {
        value.* = generator.signed();
    }
    const indices = [_]u32{ 0, 2, 1, 0, 2, 1, 0, 2 };
    const weights = [_]f32{ 0.6, 0.4, 0.3, 0.7, 0.5, 0.5, 0.9, 0.1 };

    const expert_device = try uploadF32(&gpu, &expert_out);
    const shared_device = try uploadF32(&gpu, &shared);
    const x_device = try uploadF32(&gpu, &x);
    const logit_device = try uploadF32(&gpu, &gate_logit);
    const index_device = try gpu.upload(std.mem.sliceAsBytes(&indices));
    const weight_device = try uploadF32(&gpu, &weights);
    const out_device = try gpu.allocate(tokens * hidden * 4);
    try launchMoeCombine(
        &gpu.kernels,
        gpu.stream.handle,
        expert_device.pointer,
        index_device.pointer,
        weight_device.pointer,
        shared_device.pointer,
        logit_device.pointer,
        x_device.pointer,
        out_device.pointer,
        tokens,
        hidden,
        top_k,
    );
    var combined: [tokens * hidden]f32 = undefined;
    try downloadF32(&gpu, &out_device, &combined);
    for (0..tokens) |t| {
        var dot: f32 = 0;
        for (0..hidden) |column| {
            dot += x[t * hidden + column] * gate_logit[column];
        }
        const gate = ref_ops.sigmoid(dot);
        for (0..hidden) |column| {
            var total: f32 = 0;
            for (0..top_k) |slot| {
                const expert = indices[t * top_k + slot];
                total += expert_out[
                    (@as(usize, expert) * tokens + t) * hidden +
                        column
                ] * weights[t * top_k + slot];
            }
            try std.testing.expectApproxEqAbs(
                total + gate * shared[t * hidden + column],
                combined[t * hidden + column],
                1e-6,
            );
        }
    }
}
