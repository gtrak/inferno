//! Loads the tiny checkpoint into the device tensors the executor reads.
//!
//! The file is opened with `weights/safetensors.zig` — the reader the
//! production `weights/manifest.zig` is built on — and every tensor is checked
//! against the shape the plan derives before it is uploaded. The production
//! `weights/manifest.zig` cannot open this fixture: it reconciles a
//! `model.safetensors.index.json` the fixture directory does not have and it
//! asserts the production nested config, so the loader keeps the tiny
//! `ref/fixtures.zig` config and the production safetensors reader.
//!
//! The bytes stay in f32: the tiny graph runs its matrix products through
//! `ple_project`, which is f32 in and f32 weights, so no weight rounding enters
//! the model path. The one exception is the indexer's `index_qk_proj`, because
//! `indexer_project` reads bf16 operands, exactly as the kernel contract in
//! `kernels/attention.zig` requires.
//!
//! Three shapes are rearranged here, once, rather than per forward:
//!
//! * `q_proj` is split into `q_only` and `q_gate`, so each head's RMSNorm sees
//!   a contiguous row (the checkpoint interleaves the two halves per head);
//! * `q_norm` is tiled to one copy per head, because the grouped RMSNorm kernel
//!   indexes its weight by the group's column offset inside the row;
//! * nothing else: every projection keeps its exact extent, because the f32
//!   matmul the engine uses (`ple_project`) takes arbitrary extents.

const std = @import("std");

const cuda_context = @import("../hw/api.zig").cuda_context;
const cuda_driver = @import("../hw/api.zig").cuda_driver;
const fixtures = @import("../ref/fixtures.zig");
const quant = @import("../weights/quant.zig");
const safetensors = @import("../weights/safetensors.zig");

const assert = std.debug.assert;

/// Absolute directory of the committed tiny fixtures, injected by the build.
const fixtures_root: []const u8 = @import("fixtures_dir").path;

/// Tensor names of the text path live under this prefix.
const model_prefix = "model.language_model.";

/// Longest tensor name this loader builds.
const name_bytes_max: u32 = 192;

/// One device matrix in `nn.Linear` layout: `[rows, cols]`, row-major.
pub const Matrix = struct {
    data: cuda_context.DeviceBuffer,
    rows: u32,
    cols: u32,

    fn check(self: Matrix) void {
        assert(self.rows > 0);
        assert(self.cols > 0);
        assert(self.data.bytes >= @as(u64, self.rows) * self.cols * 4);
    }
};

/// One hyper-connection block's device weights.
pub const HcWeights = struct {
    /// f32 `[hyper]`, the per-stream norm; `ops.rmsnorm` scales by `1 + w`.
    norm: cuda_context.DeviceBuffer,
    /// `[hc_lowrank, hyper]`.
    mix_down: Matrix,
    /// `[hyper, hc_lowrank]`.
    mix_up: Matrix,
    /// `[hc_count, hyper]`; null for the final mixer.
    inject: ?Matrix,
};

/// One Gated DeltaNet block's device weights.
pub const GdnWeights = struct {
    /// `[padded conv_dim, hidden]`, zero rows below `conv_dim`.
    in_proj_qkv: Matrix,
    /// `[padded value_dim, hidden]`.
    in_proj_z: Matrix,
    /// `[padded heads, hidden]`, zero rows below `heads`.
    in_proj_b: Matrix,
    in_proj_a: Matrix,
    /// f32 `[conv_dim, conv_kernel]`, channel-major.
    conv: cuda_context.DeviceBuffer,
    /// f32 `[value_head_dim]`.
    norm: cuda_context.DeviceBuffer,
    /// `[hidden, value_dim]`.
    out_proj: Matrix,
    /// f32 `[value_heads]` each.
    a_log: cuda_context.DeviceBuffer,
    dt_bias: cuda_context.DeviceBuffer,
};

/// One indexed-attention block's device weights. `q_only` and `q_gate` are the
/// per-head halves of the interleaved `q_proj`, split so each head's RMSNorm
/// sees a contiguous row.
pub const AttentionWeights = struct {
    q_only: Matrix,
    q_gate: Matrix,
    /// `[padded kv_dim, hidden]`, zero rows below `kv_dim`.
    k_proj: Matrix,
    v_proj: Matrix,
    /// `[hidden, heads * head_dim]`.
    o_proj: Matrix,
    /// f32 `[heads * head_dim]`: `q_norm` repeated once per head, the grouped
    /// norm's weight layout.
    q_norm: cuda_context.DeviceBuffer,
    /// f32 `[kv_dim]`.
    k_norm: cuda_context.DeviceBuffer,
    /// bf16 `[projected width, hidden]`.
    index_qk_proj: cuda_context.DeviceBuffer,
    /// f32 `[index_dim]` each.
    q_layernorm: cuda_context.DeviceBuffer,
    k_layernorm: cuda_context.DeviceBuffer,
};

/// One sparse MoE block's device weights.
pub const MoeWeights = struct {
    /// `[padded experts, hidden]`, zero rows below `num_experts`.
    router: Matrix,
    /// `[2 * intermediate, hidden]`: gate rows then up rows.
    shared_gate_up: Matrix,
    /// `[hidden, intermediate]`.
    shared_down: Matrix,
    /// f32 `[hidden]`, the shared expert's gate projection.
    shared_gate_logit: cuda_context.DeviceBuffer,
    /// One `[2 * intermediate, hidden]` matrix per expert.
    expert_gate_up: []Matrix,
    /// One `[hidden, intermediate]` matrix per expert.
    expert_down: []Matrix,
};

/// One PLE block's device weights.
pub const PleWeights = struct {
    /// `[hyper, ple_embed_dim]`.
    key_proj: Matrix,
    /// `[hidden, ple_embed_dim]`.
    value_proj: Matrix,
    /// f32 `[hyper]` each.
    norm_key: cuda_context.DeviceBuffer,
    norm_query: cuda_context.DeviceBuffer,
    norm_conv: cuda_context.DeviceBuffer,
    /// `[hyper, conv_kernel]`, channel-major.
    conv: cuda_context.DeviceBuffer,
};

/// One decoder layer's weights; only the slots of its kind are filled.
pub const LayerWeights = struct {
    attn_hyper: HcWeights,
    mlp_hyper: HcWeights,
    moe: MoeWeights,
    gdn: ?GdnWeights = null,
    attention: ?AttentionWeights = null,
    ple: ?PleWeights = null,
};

/// Every text weight, device-resident except the embedding table.
pub const Weights = struct {
    allocator: std.mem.Allocator,
    allocations: std.ArrayList(cuda_context.DeviceBuffer),
    /// The per-expert `Matrix` arrays, which are host slices: one entry per
    /// matrix list a layer allocated.
    matrix_lists: std.ArrayList([]Matrix),
    /// `[vocab, hidden]`, host-side: the engine gathers embedding rows on the
    /// host, as the production placement plan keeps them off the device.
    embed_tokens: []f32,
    lm_head: Matrix,
    mixer: HcWeights,
    layers: [fixtures.layers_max]LayerWeights = undefined,
    layer_count: u32 = 0,

    /// Opens the committed checkpoint and loads every text weight.
    pub fn load(
        allocator: std.mem.Allocator,
        stream: *const cuda_context.Stream,
        config: *const fixtures.RefConfig,
    ) !Weights {
        assert(config.num_hidden_layers > 0);
        assert(config.num_hidden_layers <= fixtures.layers_max);

        const path = try checkpointPath(allocator);
        defer allocator.free(path);
        var checkpoint = try Checkpoint.open(allocator, path);
        defer checkpoint.close();

        var weights = Weights{
            .allocator = allocator,
            .allocations = std.ArrayList(cuda_context.DeviceBuffer).init(allocator),
            .matrix_lists = std.ArrayList([]Matrix).init(allocator),
            .embed_tokens = &[_]f32{},
            .lm_head = undefined,
            .mixer = undefined,
            .layer_count = 0,
        };
        errdefer weights.release();

        weights.embed_tokens = try allocator.dupe(
            f32,
            try checkpoint.f32Values("model.language_model.embed_tokens.weight"),
        );
        weights.lm_head = try loadMatrix(
            &weights,
            stream,
            &checkpoint,
            "",
            "lm_head.weight",
            config.vocab_size,
            config.hidden_size,
        );
        weights.mixer = try loadHyper(
            &weights,
            stream,
            &checkpoint,
            "model.language_model.hyper_connection_mixer.",
            config,
            false,
        );
        for (0..config.num_hidden_layers) |index| {
            weights.layers[index] = try loadLayer(
                &weights,
                stream,
                &checkpoint,
                config,
                @intCast(index),
            );
        }
        weights.layer_count = config.num_hidden_layers;
        assert(weights.embed_tokens.len == config.vocab_size * config.hidden_size);
        return weights;
    }

    /// Frees every device buffer and the host embedding copy.
    pub fn deinit(self: *Weights) void {
        assert(self.layer_count > 0);
        assert(self.embed_tokens.len > 0);

        self.release();
        self.layer_count = 0;
    }

    /// Frees the allocations and forgets the embedding values.
    fn release(self: *Weights) void {
        assert(self.allocator.vtable != undefined);

        if (self.embed_tokens.len > 0) {
            self.allocator.free(self.embed_tokens);
            self.embed_tokens = &[_]f32{};
        }
        for (self.allocations.items) |buffer| {
            buffer.free();
        }
        self.allocations.deinit();
        for (self.matrix_lists.items) |list| {
            self.allocator.free(list);
        }
        self.matrix_lists.deinit();
    }
};

/// One mapped checkpoint plus the arena its header lives in.
pub const Checkpoint = struct {
    arena: std.heap.ArenaAllocator,
    file: safetensors.File,
    header: safetensors.Header,

    /// Maps `path` and parses its header.
    pub fn open(allocator: std.mem.Allocator, path: []const u8) !Checkpoint {
        assert(path.len > 0);
        assert(std.mem.endsWith(u8, path, ".safetensors"));

        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const file = try safetensors.File.open(path);
        errdefer {
            var closing = file;
            closing.close();
        }
        const header = try safetensors.parseHeader(arena.allocator(), &file, 0);
        if (!header.dataComplete(&file)) {
            return error.checkpoint_incomplete;
        }
        return .{ .arena = arena, .file = file, .header = header };
    }

    /// Releases the mapping and the header arena.
    pub fn close(self: *Checkpoint) void {
        assert(self.header.tensors.len > 0);
        assert(self.arena.queryCapacity() > 0);

        self.file.close();
        self.arena.deinit();
    }

    /// Finds one tensor by name.
    pub fn find(self: *const Checkpoint, name: []const u8) !*const safetensors.Tensor {
        assert(name.len > 0);
        assert(self.header.tensors.len > 0);

        for (self.header.tensors) |*tensor| {
            if (std.mem.eql(u8, tensor.name, name)) {
                return tensor;
            } else {
                continue;
            }
        }
        std.debug.print("checkpoint: no tensor {s}\n", .{name});
        return error.MissingTensor;
    }

    /// One f32 tensor's values.
    pub fn f32Values(self: *const Checkpoint, name: []const u8) ![]const f32 {
        assert(name.len > 0);
        assert(@sizeOf(f32) == 4);

        const tensor = try self.find(name);
        if (tensor.dtype != .f32) {
            return error.WrongDtype;
        }
        assert(tensor.file_offset % @sizeOf(f32) == 0);
        return asF32(self.file.tensorBytes(tensor));
    }

    /// One f32 vector's values: a rank-one tensor of `length`, or its `1 x
    /// length` spelling.
    pub fn f32Vector(
        self: *const Checkpoint,
        name: []const u8,
        length: u32,
    ) ![]const f32 {
        assert(name.len > 0);
        assert(length > 0);

        const tensor = try self.find(name);
        if (tensor.dtype != .f32) {
            return error.WrongDtype;
        }
        const matches = switch (tensor.dims.rank) {
            1 => tensor.dims.values[0] == length,
            2 => tensor.dims.values[0] == 1 and tensor.dims.values[1] == length,
            else => false,
        };
        if (!matches) {
            std.debug.print("checkpoint: {s} is rank {d}, expected {d} values\n", .{
                name,
                tensor.dims.rank,
                length,
            });
            return error.WrongShape;
        }
        return asF32(self.file.tensorBytes(tensor));
    }

    /// One convolution kernel's values, flattened channel-major: a
    /// `[channels, kernel]` tensor or its `[channels, 1, kernel]` spelling.
    pub fn f32Conv(
        self: *const Checkpoint,
        name: []const u8,
        channels: u32,
        kernel: u32,
    ) ![]const f32 {
        assert(name.len > 0);
        assert(channels > 0);
        assert(kernel > 0);

        const tensor = try self.find(name);
        if (tensor.dtype != .f32) {
            return error.WrongDtype;
        }
        const matches = switch (tensor.dims.rank) {
            2 => tensor.dims.values[0] == channels and
                tensor.dims.values[1] == kernel,
            3 => tensor.dims.values[0] == channels and
                tensor.dims.values[1] == 1 and
                tensor.dims.values[2] == kernel,
            else => false,
        };
        if (!matches) {
            std.debug.print("checkpoint: {s} is rank {d}, expected {d}x1x{d}\n", .{
                name,
                tensor.dims.rank,
                channels,
                kernel,
            });
            return error.WrongShape;
        }
        return asF32(self.file.tensorBytes(tensor));
    }

    /// One f32 tensor's values, checked against an expected rank-two extent.
    pub fn f32Matrix(
        self: *const Checkpoint,
        name: []const u8,
        rows: u32,
        cols: u32,
    ) ![]const f32 {
        assert(name.len > 0);
        assert(rows > 0);
        assert(cols > 0);

        const tensor = try self.find(name);
        if (tensor.dtype != .f32) {
            return error.WrongDtype;
        }
        if (tensor.dims.rank != 2) {
            return error.WrongShape;
        }
        if (tensor.dims.values[0] != rows or tensor.dims.values[1] != cols) {
            std.debug.print("checkpoint: {s} is {d}x{d}, expected {d}x{d}\n", .{
                name,
                tensor.dims.values[0],
                tensor.dims.values[1],
                rows,
                cols,
            });
            return error.WrongShape;
        }
        return asF32(self.file.tensorBytes(tensor));
    }
};

/// Views mapped tensor bytes as f32 values.
///
/// The mapping is page aligned and every f32 tensor starts at a multiple of
/// four, which is what makes the alignment cast safe; the assert states it
/// rather than assuming it.
fn asF32(bytes: []const u8) []const f32 {
    assert(bytes.len % @sizeOf(f32) == 0);
    assert(@intFromPtr(bytes.ptr) % @sizeOf(f32) == 0);

    const pointer: [*]const f32 = @ptrCast(@alignCast(bytes.ptr));
    return pointer[0 .. bytes.len / @sizeOf(f32)];
}

/// The checkpoint's file path, from the build-injected fixture directory.
pub fn checkpointPath(allocator: std.mem.Allocator) ![]u8 {
    assert(fixtures_root.len > 0);

    return std.fmt.allocPrint(allocator, "{s}/checkpoint.safetensors", .{
        fixtures_root,
    });
}

/// Rounds a byte count up to a whole device allocation.
pub fn allocationBytes(bytes: u64) u64 {
    assert(bytes > 0);
    assert(@sizeOf(f64) == 8);

    return (bytes + 4095) / 4096 * 4096;
}

/// Uploads `bytes` into a fresh device buffer tracked by `weights`.
fn upload(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    bytes: []const u8,
) !cuda_context.DeviceBuffer {
    assert(bytes.len > 0);
    assert(weights.allocations.allocator.vtable != undefined);

    const buffer = try cuda_context.DeviceBuffer.allocate(
        allocationBytes(@intCast(bytes.len)),
    );
    errdefer buffer.free();
    const staged = try cuda_context.PinnedHostBuffer.allocate(buffer.bytes);
    defer staged.free();
    @memset(staged.slice(), 0);
    @memcpy(staged.slice()[0..bytes.len], bytes);
    try stream.copy_h_to_d(&buffer, &staged, buffer.bytes);
    try weights.allocations.append(buffer);
    return buffer;
}

/// Uploads f32 values as a device vector.
fn uploadF32(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    values: []const f32,
) !cuda_context.DeviceBuffer {
    assert(values.len > 0);
    assert(@sizeOf(f32) == 4);

    return upload(weights, stream, std.mem.sliceAsBytes(values));
}

/// Uploads f32 values as a device matrix.
fn uploadMatrix(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    values: []const f32,
    rows: u32,
    cols: u32,
) !Matrix {
    assert(values.len == @as(usize, rows) * cols);
    assert(rows > 0);
    assert(cols > 0);

    const matrix = Matrix{
        .data = try uploadF32(weights, stream, values),
        .rows = rows,
        .cols = cols,
    };
    matrix.check();
    return matrix;
}

/// Uploads f32 values rounded to bf16.
fn uploadBf16(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    values: []const f32,
) !cuda_context.DeviceBuffer {
    assert(values.len > 0);
    assert(@sizeOf(u16) == 2);

    const allocator = weights.allocations.allocator;
    const bits = try allocator.alloc(u16, values.len);
    defer allocator.free(bits);
    for (values, bits) |value, *slot| {
        slot.* = quant.f32_to_bf16(value);
    }
    return upload(weights, stream, std.mem.sliceAsBytes(bits));
}

/// Builds `prefix ++ suffix` in `buffer`.
fn tensorName(
    buffer: []u8,
    prefix: []const u8,
    suffix: []const u8,
) ![]const u8 {
    assert(prefix.len + suffix.len <= buffer.len);
    assert(suffix.len > 0);

    const name = try std.fmt.bufPrint(buffer, "{s}{s}", .{ prefix, suffix });
    assert(name.len == prefix.len + suffix.len);
    return name;
}

/// Builds `"{s}{d}."` layer prefixes in `buffer`.
fn layerPrefix(buffer: []u8, layer: u32) ![]const u8 {
    assert(buffer.len > model_prefix.len);

    const prefix = try std.fmt.bufPrint(
        buffer,
        model_prefix ++ "layers.{d}.",
        .{layer},
    );
    assert(prefix.len > model_prefix.len);
    return prefix;
}

/// Loads one f32 matrix under `prefix`, checking its extent.
fn loadMatrix(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    checkpoint: *const Checkpoint,
    prefix: []const u8,
    suffix: []const u8,
    rows: u32,
    cols: u32,
) !Matrix {
    assert(suffix.len > 0);
    assert(rows > 0);

    var name: [name_bytes_max]u8 = undefined;
    const full = try tensorName(&name, prefix, suffix);
    const values = try checkpoint.f32Matrix(full, rows, cols);
    return uploadMatrix(weights, stream, values, rows, cols);
}

/// Loads one f32 vector under `prefix`, checking its length.
fn loadVector(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    checkpoint: *const Checkpoint,
    prefix: []const u8,
    suffix: []const u8,
    length: u32,
) !cuda_context.DeviceBuffer {
    assert(suffix.len > 0);
    assert(length > 0);

    var name: [name_bytes_max]u8 = undefined;
    const full = try tensorName(&name, prefix, suffix);
    const values = try checkpoint.f32Vector(full, length);
    return uploadF32(weights, stream, values);
}

/// Loads one convolution kernel under `prefix`, flattened channel-major.
fn loadConvWeights(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    checkpoint: *const Checkpoint,
    prefix: []const u8,
    suffix: []const u8,
    channels: u32,
    kernel: u32,
) !cuda_context.DeviceBuffer {
    assert(suffix.len > 0);
    assert(channels > 0);

    var name: [name_bytes_max]u8 = undefined;
    const full = try tensorName(&name, prefix, suffix);
    const values = try checkpoint.f32Conv(full, channels, kernel);
    return uploadF32(weights, stream, values);
}

/// Loads one hyper-connection block (`inject` only when `use_combine`).
fn loadHyper(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    checkpoint: *const Checkpoint,
    prefix: []const u8,
    config: *const fixtures.RefConfig,
    use_combine: bool,
) !HcWeights {
    assert(prefix.len > 0);
    assert(config.hc_count > 0);

    const hyper = config.hyper_size();
    return .{
        .norm = try loadVector(
            weights,
            stream,
            checkpoint,
            prefix,
            "hc_norm.weight",
            hyper,
        ),
        .mix_down = try loadMatrix(
            weights,
            stream,
            checkpoint,
            prefix,
            "input_mix_weight_down.weight",
            config.hc_lowrank,
            hyper,
        ),
        .mix_up = try loadMatrix(
            weights,
            stream,
            checkpoint,
            prefix,
            "input_mix_weight_up.weight",
            hyper,
            config.hc_lowrank,
        ),
        .inject = if (use_combine) try loadMatrix(
            weights,
            stream,
            checkpoint,
            prefix,
            "block_inject_weight.weight",
            config.hc_count,
            hyper,
        ) else null,
    };
}

/// Loads a `gate_proj`/`up_proj` pair as the single `[2 * inter, hidden]`
/// matrix `moe_silu_mul` reads: gate rows, then up rows.
fn loadGateUp(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    checkpoint: *const Checkpoint,
    prefix: []const u8,
    gate_suffix: []const u8,
    up_suffix: []const u8,
    intermediate: u32,
    hidden: u32,
) !Matrix {
    assert(gate_suffix.len > 0);
    assert(up_suffix.len > 0);
    assert(intermediate > 0);

    var name: [name_bytes_max]u8 = undefined;
    const gate = try checkpoint.f32Matrix(
        try tensorName(&name, prefix, gate_suffix),
        intermediate,
        hidden,
    );
    const up = try checkpoint.f32Matrix(
        try tensorName(&name, prefix, up_suffix),
        intermediate,
        hidden,
    );
    const allocator = weights.allocations.allocator;
    const rows = 2 * intermediate;
    const combined = try allocator.alloc(f32, @as(usize, rows) * hidden);
    defer allocator.free(combined);
    @memcpy(combined[0..gate.len], gate);
    @memcpy(combined[gate.len..][0..up.len], up);
    return uploadMatrix(weights, stream, combined, rows, hidden);
}

/// Loads the router, the per-expert matrices and the shared expert.
fn loadMoe(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    checkpoint: *const Checkpoint,
    prefix: []const u8,
    config: *const fixtures.RefConfig,
) !MoeWeights {
    assert(config.num_experts > 0);
    assert(config.moe_intermediate_size > 0);

    const hidden = config.hidden_size;
    const intermediate = config.moe_intermediate_size;
    const shared = config.shared_expert_intermediate_size;
    var gate_up = try weights.allocator.alloc(Matrix, config.num_experts);
    errdefer weights.allocator.free(gate_up);
    try weights.matrix_lists.append(gate_up);
    var down = try weights.allocator.alloc(Matrix, config.num_experts);
    errdefer weights.allocator.free(down);
    try weights.matrix_lists.append(down);
    var suffixes: [2][name_bytes_max]u8 = undefined;
    for (0..config.num_experts) |expert| {
        var gate_name: [name_bytes_max]u8 = undefined;
        const gate_suffix = try std.fmt.bufPrint(
            &gate_name,
            "mlp.experts.{d}.gate_proj.weight",
            .{expert},
        );
        var up_name: [name_bytes_max]u8 = undefined;
        const up_suffix = try std.fmt.bufPrint(
            &up_name,
            "mlp.experts.{d}.up_proj.weight",
            .{expert},
        );
        gate_up[expert] = try loadGateUp(
            weights,
            stream,
            checkpoint,
            prefix,
            gate_suffix,
            up_suffix,
            intermediate,
            hidden,
        );
        down[expert] = try loadMatrix(
            weights,
            stream,
            checkpoint,
            prefix,
            try std.fmt.bufPrint(
                &suffixes[1],
                "mlp.experts.{d}.down_proj.weight",
                .{expert},
            ),
            hidden,
            intermediate,
        );
    }
    return .{
        .router = try loadMatrix(weights, stream, checkpoint, prefix, "mlp.gate.weight", config.num_experts, hidden),
        .shared_gate_up = try loadGateUp(
            weights,
            stream,
            checkpoint,
            prefix,
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            shared,
            hidden,
        ),
        .shared_down = try loadMatrix(
            weights,
            stream,
            checkpoint,
            prefix,
            "mlp.shared_expert.down_proj.weight",
            hidden,
            shared,
        ),
        .shared_gate_logit = try loadVector(
            weights,
            stream,
            checkpoint,
            prefix,
            "mlp.shared_expert_gate.weight",
            hidden,
        ),
        .expert_gate_up = gate_up,
        .expert_down = down,
    };
}

/// Loads the Gated DeltaNet weights.
fn loadGdn(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    checkpoint: *const Checkpoint,
    prefix: []const u8,
    config: *const fixtures.RefConfig,
) !GdnWeights {
    assert(config.linear_num_value_heads > 0);
    assert(config.linear_conv_kernel_dim > 1);

    const hidden = config.hidden_size;
    const heads = config.linear_num_value_heads;
    const conv = config.conv_dim();
    const value = config.value_dim();
    return .{
        .in_proj_qkv = try loadMatrix(weights, stream, checkpoint, prefix, "linear_attn.in_proj_qkv.weight", conv, hidden),
        .in_proj_z = try loadMatrix(weights, stream, checkpoint, prefix, "linear_attn.in_proj_z.weight", value, hidden),
        .in_proj_b = try loadMatrix(weights, stream, checkpoint, prefix, "linear_attn.in_proj_b.weight", heads, hidden),
        .in_proj_a = try loadMatrix(weights, stream, checkpoint, prefix, "linear_attn.in_proj_a.weight", heads, hidden),
        .conv = try loadConvWeights(
            weights,
            stream,
            checkpoint,
            prefix,
            "linear_attn.conv1d.weight",
            conv,
            config.linear_conv_kernel_dim,
        ),
        .norm = try loadVector(
            weights,
            stream,
            checkpoint,
            prefix,
            "linear_attn.norm.weight",
            config.linear_value_head_dim,
        ),
        .out_proj = try loadMatrix(
            weights,
            stream,
            checkpoint,
            prefix,
            "linear_attn.out_proj.weight",
            hidden,
            value,
        ),
        .a_log = try loadVector(
            weights,
            stream,
            checkpoint,
            prefix,
            "linear_attn.A_log",
            heads,
        ),
        .dt_bias = try loadVector(
            weights,
            stream,
            checkpoint,
            prefix,
            "linear_attn.dt_bias",
            heads,
        ),
    };
}

/// Loads the indexed-attention weights, indexer included.
fn loadAttention(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    checkpoint: *const Checkpoint,
    prefix: []const u8,
    config: *const fixtures.RefConfig,
) !AttentionWeights {
    assert(config.num_attention_heads > 0);
    assert(config.indexer_n_heads > 0);

    const hidden = config.hidden_size;
    const heads = config.num_attention_heads;
    const head_dim = config.head_dim;
    const kv_dim = config.num_key_value_heads * head_dim;
    const width = heads * head_dim;
    var name: [name_bytes_max]u8 = undefined;
    const q_proj = try checkpoint.f32Matrix(
        try tensorName(&name, prefix, "self_attn.q_proj.weight"),
        width * 2,
        hidden,
    );
    return .{
        .q_only = try uploadSplit(
            weights,
            stream,
            q_proj,
            heads,
            head_dim,
            2 * head_dim,
            0,
            hidden,
        ),
        .q_gate = try uploadSplit(
            weights,
            stream,
            q_proj,
            heads,
            head_dim,
            2 * head_dim,
            head_dim,
            hidden,
        ),
        .k_proj = try loadMatrix(weights, stream, checkpoint, prefix, "self_attn.k_proj.weight", kv_dim, hidden),
        .v_proj = try loadMatrix(weights, stream, checkpoint, prefix, "self_attn.v_proj.weight", kv_dim, hidden),
        .o_proj = try loadMatrix(
            weights,
            stream,
            checkpoint,
            prefix,
            "self_attn.o_proj.weight",
            hidden,
            width,
        ),
        .q_norm = try uploadTiled(
            weights,
            stream,
            try checkpoint.f32Vector(
                try tensorName(&name, prefix, "self_attn.q_norm.weight"),
                head_dim,
            ),
            heads,
        ),
        .k_norm = try loadVector(
            weights,
            stream,
            checkpoint,
            prefix,
            "self_attn.k_norm.weight",
            head_dim,
        ),
        .index_qk_proj = try uploadBf16(
            weights,
            stream,
            try checkpoint.f32Matrix(
                try tensorName(
                    &name,
                    prefix,
                    "self_attn.indexer.index_qk_proj.weight",
                ),
                (config.indexer_n_heads + config.indexer_kv_heads) *
                    config.indexer_head_dim,
                hidden,
            ),
        ),
        .q_layernorm = try loadVector(
            weights,
            stream,
            checkpoint,
            prefix,
            "self_attn.indexer.q_layernorm.weight",
            config.indexer_head_dim,
        ),
        .k_layernorm = try loadVector(
            weights,
            stream,
            checkpoint,
            prefix,
            "self_attn.indexer.k_layernorm.weight",
            config.indexer_head_dim,
        ),
    };
}

/// Copies one `head_dim`-wide slice out of every `stride`-wide head block of
/// `source` into a dense `[heads * head_dim, cols]` matrix.
fn uploadSplit(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    source: []const f32,
    heads: u32,
    head_dim: u32,
    stride: u32,
    offset: u32,
    cols: u32,
) !Matrix {
    assert(source.len == @as(usize, heads) * stride * cols);
    assert(offset + head_dim <= stride);
    assert(cols > 0);

    const allocator = weights.allocations.allocator;
    const rows = heads * head_dim;
    const split = try allocator.alloc(f32, @as(usize, rows) * cols);
    defer allocator.free(split);
    for (0..heads) |head| {
        const target = @as(usize, head) * head_dim * cols;
        const from = (@as(usize, head) * stride + offset) * cols;
        @memcpy(
            split[target..][0 .. @as(usize, head_dim) * cols],
            source[from..][0 .. @as(usize, head_dim) * cols],
        );
    }
    return uploadMatrix(weights, stream, split, rows, cols);
}

/// Uploads `values` repeated `copies` times: the grouped RMSNorm's per-group
/// weight layout.
fn uploadTiled(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    values: []const f32,
    copies: u32,
) !cuda_context.DeviceBuffer {
    assert(values.len > 0);
    assert(copies > 0);

    const allocator = weights.allocations.allocator;
    const tiled = try allocator.alloc(f32, values.len * copies);
    defer allocator.free(tiled);
    for (0..copies) |copy| {
        @memcpy(tiled[copy * values.len ..][0..values.len], values);
    }
    return uploadF32(weights, stream, tiled);
}

/// Loads the PLE weights of one layer.
fn loadPle(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    checkpoint: *const Checkpoint,
    prefix: []const u8,
    config: *const fixtures.RefConfig,
) !PleWeights {
    assert(config.ple_embed_dim > 0);
    assert(config.ple_conv_kernel_size > 1);

    const hyper = config.hyper_size();
    return .{
        .key_proj = try loadMatrix(
            weights,
            stream,
            checkpoint,
            prefix,
            "ple.key_proj.weight",
            hyper,
            config.ple_embed_dim,
        ),
        .value_proj = try loadMatrix(
            weights,
            stream,
            checkpoint,
            prefix,
            "ple.value_proj.weight",
            config.hidden_size,
            config.ple_embed_dim,
        ),
        .norm_key = try loadVector(
            weights,
            stream,
            checkpoint,
            prefix,
            "ple.norm_key.weight",
            hyper,
        ),
        .norm_query = try loadVector(
            weights,
            stream,
            checkpoint,
            prefix,
            "ple.norm_query.weight",
            hyper,
        ),
        .norm_conv = try loadVector(
            weights,
            stream,
            checkpoint,
            prefix,
            "ple.norm_conv.weight",
            hyper,
        ),
        .conv = try loadConvWeights(
            weights,
            stream,
            checkpoint,
            prefix,
            "ple.conv1d.weight",
            hyper,
            config.ple_conv_kernel_size,
        ),
    };
}

/// Loads every weight of one decoder layer.
fn loadLayer(
    weights: *Weights,
    stream: *const cuda_context.Stream,
    checkpoint: *const Checkpoint,
    config: *const fixtures.RefConfig,
    layer: u32,
) !LayerWeights {
    assert(layer < config.num_hidden_layers);
    assert(config.num_hidden_layers > 0);

    var buffer: [name_bytes_max]u8 = undefined;
    const prefix = try layerPrefix(&buffer, layer);
    var hyper_buffer: [name_bytes_max]u8 = undefined;
    const attn_hyper_prefix = try std.fmt.bufPrint(
        &hyper_buffer,
        "{s}attn_hyper_connection.",
        .{prefix},
    );
    var layer_weights = LayerWeights{
        .attn_hyper = try loadHyper(
            weights,
            stream,
            checkpoint,
            attn_hyper_prefix,
            config,
            true,
        ),
        .mlp_hyper = undefined,
        .moe = undefined,
    };
    var mlp_buffer: [name_bytes_max]u8 = undefined;
    const mlp_hyper_prefix = try std.fmt.bufPrint(
        &mlp_buffer,
        "{s}mlp_hyper_connection.",
        .{prefix},
    );
    layer_weights.mlp_hyper = try loadHyper(
        weights,
        stream,
        checkpoint,
        mlp_hyper_prefix,
        config,
        true,
    );
    layer_weights.moe = try loadMoe(
        weights,
        stream,
        checkpoint,
        prefix,
        config,
    );
    switch (config.layer_types[layer]) {
        .linear_attention => layer_weights.gdn = try loadGdn(
            weights,
            stream,
            checkpoint,
            prefix,
            config,
        ),
        .indexed_attention => layer_weights.attention = try loadAttention(
            weights,
            stream,
            checkpoint,
            prefix,
            config,
        ),
    }
    if (config.ple_layer_index(layer) != null) {
        layer_weights.ple = try loadPle(
            weights,
            stream,
            checkpoint,
            prefix,
            config,
        );
    } else {
        assert(layer_weights.ple == null);
    }
    return layer_weights;
}

test "the tiny checkpoint loads every text weight at the plan's shape" {
    const allocator = std.testing.allocator;
    const config = try fixtures.load_config(allocator);

    var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
    const count = try cuda_driver.query_devices(devices[0..]);
    try std.testing.expect(count > 0);
    var context = try cuda_context.Context.retain(&devices[0]);
    defer context.release();
    try context.set_current();
    var stream = try cuda_context.Stream.create();
    defer stream.destroy();

    var weights = try Weights.load(allocator, &stream, &config);
    defer weights.deinit();
    try stream.synchronize();
    try std.testing.expectEqual(config.num_hidden_layers, weights.layer_count);

    // The embedding and the head are the checkpoint's own bytes.
    const checkpoint_file = try checkpointPath(allocator);
    defer allocator.free(checkpoint_file);
    var checkpoint = try Checkpoint.open(allocator, checkpoint_file);
    defer checkpoint.close();
    const embed = try checkpoint.f32Values(
        "model.language_model.embed_tokens.weight",
    );
    try std.testing.expectEqualSlices(f32, embed, weights.embed_tokens);
    try std.testing.expectEqual(config.vocab_size, weights.lm_head.rows);

    // The interleaved `q_proj` split gives back the checkpoint's rows, in the
    // order every head's RMSNorm expects.
    const width = config.num_attention_heads * config.head_dim;
    var attention_layers: u32 = 0;
    var gdn_layers: u32 = 0;
    for (0..config.num_hidden_layers) |layer| {
        if (weights.layers[layer].attention) |attention| {
            attention_layers += 1;
            try std.testing.expectEqual(width, attention.q_only.rows);
            try std.testing.expectEqual(width, attention.q_gate.rows);
            try std.testing.expect(attention.q_norm.bytes >=
                @as(u64, config.num_attention_heads * config.head_dim * 4));
            try std.testing.expectEqual(
                @as(u64, (config.indexer_n_heads + config.indexer_kv_heads) *
                    config.indexer_head_dim * config.hidden_size * 2),
                attention.index_qk_proj.bytes,
            );
        } else {
            gdn_layers += 1;
            const gdn = weights.layers[layer].gdn.?;
            try std.testing.expectEqual(
                @as(u32, config.conv_dim()),
                gdn.in_proj_qkv.rows,
            );
            try std.testing.expectEqual(
                @as(u32, config.value_dim()),
                gdn.out_proj.cols,
            );
            try std.testing.expect(gdn.a_log.bytes >=
                @as(u64, config.linear_num_value_heads) * 4);
        }
        try std.testing.expectEqual(
            @as(u32, config.num_experts),
            weights.layers[layer].moe.expert_down.len,
        );
        try std.testing.expectEqual(
            @as(u32, 2 * config.moe_intermediate_size),
            weights.layers[layer].moe.expert_gate_up[0].rows,
        );
    }
    try std.testing.expectEqual(@as(u32, 1), attention_layers);
    try std.testing.expectEqual(@as(u32, 3), gdn_layers);
    // `ple_layer_ids` is 1-based, so id 1 is layer index 0.
    const ple_layer = config.ple_layer_ids[0] - 1;
    try std.testing.expect(weights.layers[ple_layer].ple != null);
    try std.testing.expect(weights.layers[0].ple == null);
}
