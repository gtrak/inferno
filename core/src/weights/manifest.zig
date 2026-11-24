//! The typed tensor table of the checkpoint, and the `manifest.json` report.
//!
//! `build` walks the header of every safetensors file next to `config.json`,
//! classifies each tensor into a closed set of `Role`s, checks its dtype and
//! extents against the dimensions derived from the config, and stores a
//! `TensorRef` in the slot the model graph will read it from. A tensor that
//! matches no role is a hard error: silently ignoring a weight is how a model
//! loads with a zeroed gate.
//!
//! Nothing here copies weight bytes. A `TensorRef` is `{file_index,
//! file_offset,
//! byte_len, dtype, dims}`, so the whole 132 GB checkpoint is described by
//! about 12 MB of typed references.
//!
//! The real test is reconciliation: the sum of every role's bytes must equal
//! `metadata.total_size` from the checkpoint's own
//! `model.safetensors.index.json`,
//! and every file's header must account for its size exactly. A misread layout
//! (say, scales counted as F32) breaks the total, and a tensor that reached the
//! wrong role breaks a derived per-role count.

const std = @import("std");

const config = @import("config.zig");
const quant = @import("quant.zig");
const safetensors = @import("safetensors.zig");

const assert = std.debug.assert;

/// Suffix of the files that hold weights.
const weights_extension = ".safetensors";

/// The checkpoint's own weight map, whose `metadata.total_size` every byte in
/// this manifest is reconciled against.
const index_file_name = "model.safetensors.index.json";

/// The real index is 31 MiB; the bound only has to be finite and generous.
const index_bytes_max: u32 = 1 << 27;

/// A shard still being written is picked up under this name.
const partial_extension = ".safetensors.part";

/// Prefix of every text layer's tensors.
const layers_prefix = "model.language_model.layers.";

/// Tensors that live outside the layers.
const embed_tokens_name = "model.language_model.embed_tokens.weight";
const lm_head_name = "lm_head.weight";
const final_mixer_prefix = "model.language_model.hyper_connection_mixer.";

/// Sub-trees that are recognised, counted, and not used yet.
const deferred_prefixes = [_][]const u8{ "mtp.", "model.visual." };

/// Every tensor belongs to exactly one role.
pub const Role = enum {
    embed_tokens,
    lm_head,
    attn_hyper_connection,
    mlp_hyper_connection,
    hyper_connection_mixer,
    router_gate,
    shared_expert,
    shared_expert_gate,
    expert,
    linear_attn,
    q_proj,
    k_proj,
    v_proj,
    o_proj,
    attn_norm,
    indexer,
    ple_projection,
    ple_norm,
    ple_conv1d,
    ngram_index,
    ngram_scale,
    ngram_table,
    deferred,
};

/// One entry per `Role`, in declaration order.
pub const all_roles = [_]Role{
    .embed_tokens,
    .lm_head,
    .attn_hyper_connection,
    .mlp_hyper_connection,
    .hyper_connection_mixer,
    .router_gate,
    .shared_expert,
    .shared_expert_gate,
    .expert,
    .linear_attn,
    .q_proj,
    .k_proj,
    .v_proj,
    .o_proj,
    .attn_norm,
    .indexer,
    .ple_projection,
    .ple_norm,
    .ple_conv1d,
    .ngram_index,
    .ngram_scale,
    .ngram_table,
    .deferred,
};

comptime {
    assert(all_roles.len == std.meta.fields(Role).len);
}

/// Per-role tensor count and byte total; index with `@intFromEnum(role)`.
pub const RoleTotal = struct {
    count: u64 = 0,
    bytes: u64 = 0,

    pub fn add(self: *RoleTotal, bytes: u64) void {
        assert(bytes > 0);
        assert(self.bytes <= std.math.maxInt(u64) - bytes);

        self.count += 1;
        self.bytes += bytes;
    }
};

/// A tensor's location and layout: everything a kernel needs to be pointed at
/// it, and nothing else. An all-zero ref means "this slot was never filled".
pub const TensorRef = struct {
    dtype: safetensors.Dtype = .u8,
    dims: safetensors.Dims = .{},
    file_index: u32 = 0,
    file_offset: u64 = 0,
    byte_len: u64 = 0,

    pub fn fromTensor(tensor: *const safetensors.Tensor) TensorRef {
        assert(tensor.byte_len > 0);
        assert(tensor.dims.rank <= safetensors.rank_max);

        const ref = TensorRef{
            .dtype = tensor.dtype,
            .dims = tensor.dims,
            .file_index = tensor.file_index,
            .file_offset = tensor.file_offset,
            .byte_len = tensor.byte_len,
        };
        assert(ref.byte_len == tensor.byte_len);
        return ref;
    }

    pub fn isEmpty(self: TensorRef) bool {
        assert(self.dims.rank <= safetensors.rank_max);
        // A filled ref always starts after the length prefix of its file; a
        // scalar tensor is rank zero but never empty, so rank cannot decide.
        assert((self.byte_len == 0) == (self.file_offset == 0));

        return self.byte_len == 0;
    }
};

/// The four tensors of a hyper-connection block. The final mixer has no
/// injection weight because it does not feed another stream.
pub const HyperConnection = struct {
    block_inject: TensorRef = .{},
    norm: TensorRef = .{},
    mix_down: TensorRef = .{},
    mix_up: TensorRef = .{},
};

/// Three tensors; the final mixer leaves `block_inject` empty.
pub const FinalMixer = struct {
    norm: TensorRef = .{},
    mix_down: TensorRef = .{},
    mix_up: TensorRef = .{},
};

pub const SharedExpert = struct {
    gate_proj: TensorRef = .{},
    up_proj: TensorRef = .{},
    down_proj: TensorRef = .{},
};

/// One NVFP4 projection: packed 4-bit weights plus its three scale tensors.
pub const QuantizedProjection = struct {
    weight: TensorRef = .{},
    weight_scale: TensorRef = .{},
    weight_scale_2: TensorRef = .{},
    input_scale: TensorRef = .{},
};

pub const Expert = struct {
    gate_proj: QuantizedProjection = .{},
    up_proj: QuantizedProjection = .{},
    down_proj: QuantizedProjection = .{},
};

/// GatedDeltaNet projection and state tensors of a linear attention layer.
pub const LinearAttention = struct {
    in_proj_qkv: TensorRef = .{},
    in_proj_z: TensorRef = .{},
    in_proj_a: TensorRef = .{},
    in_proj_b: TensorRef = .{},
    out_proj: TensorRef = .{},
    conv1d: TensorRef = .{},
    a_log: TensorRef = .{},
    dt_bias: TensorRef = .{},
    norm: TensorRef = .{},
};

/// Indexed attention tensors of a full attention layer, indexer included.
pub const SelfAttention = struct {
    q_proj: TensorRef = .{},
    k_proj: TensorRef = .{},
    v_proj: TensorRef = .{},
    o_proj: TensorRef = .{},
    q_norm: TensorRef = .{},
    k_norm: TensorRef = .{},
    indexer_qk_proj: TensorRef = .{},
    indexer_q_norm: TensorRef = .{},
    indexer_k_norm: TensorRef = .{},
};

/// The PLE layer: hashed n-gram lookup plus its projections.
pub const Ple = struct {
    embedding_shards: []TensorRef = &.{},
    embedding_scale: TensorRef = .{},
    heads_offsets: TensorRef = .{},
    heads_vocab_sizes: TensorRef = .{},
    layer_multipliers: TensorRef = .{},
    key_proj: TensorRef = .{},
    value_proj: TensorRef = .{},
    conv1d: TensorRef = .{},
    norm_key: TensorRef = .{},
    norm_query: TensorRef = .{},
    norm_conv: TensorRef = .{},
};

/// One text layer, typed. Only the slots its `kind` uses are filled.
pub const Layer = struct {
    index: u32 = 0,
    kind: config.LayerType = .linear_attention,
    attn_hyper_connection: HyperConnection = .{},
    mlp_hyper_connection: HyperConnection = .{},
    router_gate: TensorRef = .{},
    shared_expert: SharedExpert = .{},
    shared_expert_gate: TensorRef = .{},
    experts: []Expert = &.{},
    linear_attn: LinearAttention = .{},
    self_attn: SelfAttention = .{},
    ple: ?Ple = null,
};

/// What one weight file contributed.
pub const FileReport = struct {
    name: []const u8,
    tensors: u64 = 0,
    tensor_bytes: u64 = 0,
    header_bytes: u64 = 0,
    file_bytes: u64 = 0,
    /// False while the shard is still downloading: its header is complete, its
    /// data section is not.
    data_present: bool = true,
};

/// The whole checkpoint, typed.
pub const Manifest = struct {
    text: config.TextModel,
    embed_tokens: TensorRef = .{},
    lm_head: TensorRef = .{},
    final_mixer: FinalMixer = .{},
    layers: []Layer = &.{},
    totals: [all_roles.len]RoleTotal = [_]RoleTotal{.{}} ** all_roles.len,
    files: []FileReport = &.{},
    deferred_tensors: u64 = 0,
    deferred_bytes: u64 = 0,
    /// `metadata.total_size` of `model.safetensors.index.json`: the
    /// checkpoint's
    /// own statement of how many weight bytes there are. The manifest must
    /// reconcile against it exactly, which is the strongest available check
    /// that every tensor was classified once and its dtype read correctly.
    index_total_size: u64 = 0,
    /// Rows of the logical n-gram table after padding, and per shard.
    ngram_table_rows: u64 = 0,
    ngram_shard_rows: u64 = 0,

    /// Bytes and count of every role that describes an attention projection,
    /// for the reporting aggregate that THEORY.md budgets as "self_attn".
    pub fn selfAttentionTotal(self: *const Manifest) RoleTotal {
        assert(all_roles.len == std.meta.fields(Role).len);
        assert(self.totals.len == all_roles.len);

        var attention: RoleTotal = .{};
        const attention_roles = [_]Role{
            .q_proj, .k_proj, .v_proj, .o_proj, .attn_norm, .indexer,
        };
        for (attention_roles) |role| {
            attention.count += self.totals[@intFromEnum(role)].count;
            attention.bytes += self.totals[@intFromEnum(role)].bytes;
        }
        assert(attention.count > 0);
        return attention;
    }

    pub fn total(self: *const Manifest, role: Role) RoleTotal {
        assert(@intFromEnum(role) < self.totals.len);
        assert(self.totals.len == all_roles.len);

        return self.totals[@intFromEnum(role)];
    }
};

/// The parsed checkpoint: types, metadata, and the open mappings.
pub const Loaded = struct {
    manifest: Manifest,
    files: []safetensors.File,
    headers: []safetensors.Header,

    /// Bytes of tensor payload across every file.
    pub fn tensorBytesTotal(self: *const Loaded) u64 {
        assert(self.manifest.files.len == self.files.len);
        assert(self.files.len > 0);

        var total: u64 = 0;
        for (self.manifest.files) |report| {
            total += report.tensor_bytes;
        }
        assert(total > 0);
        return total;
    }

    /// View of one tensor's bytes, straight out of the read-only mapping.
    pub fn tensorBytes(self: *const Loaded, ref: TensorRef) []const u8 {
        assert(!ref.isEmpty());
        assert(ref.file_index < self.files.len);

        const file = &self.files[ref.file_index];
        const offset: usize = @intCast(ref.file_offset);
        const length: usize = @intCast(ref.byte_len);
        const bytes = file.bytes[offset..][0..length];
        assert(bytes.len == ref.byte_len);
        return bytes;
    }
};

/// Tensor plus the layer it belongs to, when it belongs to one.
const Classified = struct {
    role: Role,
    layer: ?u32 = null,
};

/// `metadata` of `model.safetensors.index.json`; the weight map itself is not
/// read because the manifest walks the safetensors headers instead.
const IndexFile = struct {
    metadata: struct {
        total_size: u64,
    },
};

/// Total weight bytes the checkpoint's index claims.
fn readIndexTotalSize(
    allocator: std.mem.Allocator,
    model_directory: []const u8,
) !u64 {
    assert(model_directory.len > 0);
    assert(index_bytes_max > 0);

    const path = try std.fs.path.join(
        allocator,
        &.{ model_directory, index_file_name },
    );
    const bytes = try std.fs.cwd().readFileAlloc(
        allocator,
        path,
        index_bytes_max,
    );
    // `weight_map` is 300k entries of data we deliberately re-derive from the
    // headers, so it is skipped rather than parsed.
    const parsed = std.json.parseFromSliceLeaky(
        IndexFile,
        allocator,
        bytes,
        .{ .ignore_unknown_fields = true },
    ) catch |err| {
        std.debug.print("manifest: cannot parse {s}: {s}\n", .{
            index_file_name,
            @errorName(err),
        });
        return error.malformed_index;
    };
    if (parsed.metadata.total_size == 0) {
        return error.malformed_index;
    } else {
        assert(parsed.metadata.total_size > 0);
    }
    return parsed.metadata.total_size;
}

/// Reads `config.json`, maps every weight file, and types the whole checkpoint.
pub fn build(
    allocator: std.mem.Allocator,
    model_directory: []const u8,
) !Loaded {
    assert(model_directory.len > 0);
    assert(weights_extension.len > 0);

    const text = try readTextModel(allocator, model_directory);
    const names = try listWeightsFiles(allocator, model_directory);

    const files = try allocator.alloc(safetensors.File, names.len);
    const headers = try allocator.alloc(safetensors.Header, names.len);
    var reports = try allocator.alloc(FileReport, names.len);
    try mapFiles(allocator, model_directory, names, files, headers, reports);

    var manifest = Manifest{
        .text = text,
        .layers = try allocator.alloc(Layer, text.config.num_hidden_layers),
        .files = reports,
        .index_total_size = try readIndexTotalSize(allocator, model_directory),
    };
    try initLayers(allocator, &manifest);

    var layer_counts = try allocator.alloc(u64, manifest.layers.len);
    @memset(layer_counts, 0);
    var tensor_total: u64 = 0;
    for (headers, files, 0..) |*header, *file, file_index| {
        for (header.tensors) |*tensor| {
            const classified = try classify(&manifest, tensor);
            record(&manifest, classified, tensor);
            reports[file_index].tensors += 1;
            reports[file_index].tensor_bytes += tensor.byte_len;
            tensor_total += 1;
            if (classified.layer) |layer_index| {
                layer_counts[layer_index] += 1;
            } else {
                const unlayered = classified.role == .embed_tokens or
                    classified.role == .lm_head or
                    classified.role == .hyper_connection_mixer or
                    classified.role == .deferred;
                assert(unlayered);
            }
        }
        try checkFileTotals(reports[file_index], file, header);
    }

    var loaded = Loaded{
        .manifest = manifest,
        .files = files,
        .headers = headers,
    };
    try validateNgramTable(allocator, &loaded);
    try validateCompleteness(&loaded, layer_counts, tensor_total);
    assert(loaded.files.len == names.len);
    return loaded;
}

/// Closes every mapping; the typed table stays valid for reads until then.
pub fn unload(loaded: *Loaded) void {
    assert(loaded.files.len == loaded.headers.len);
    assert(loaded.files.len > 0);

    for (loaded.files) |*file| {
        file.close();
    }
}

/// Loads `config.json` and validates the text section.
fn readTextModel(
    allocator: std.mem.Allocator,
    model_directory: []const u8,
) !config.TextModel {
    assert(model_directory.len > 0);
    assert(model_directory.len > 1);

    const path = try std.fs.path.join(
        allocator,
        &.{ model_directory, "config.json" },
    );
    const file_config = try config.load(allocator, path);
    const text = try config.TextModel.init(file_config.text_config);
    assert(text.config.num_hidden_layers > 0);
    return text;
}

/// Names of every weight file, sorted, so `file_index` is reproducible.
fn listWeightsFiles(
    allocator: std.mem.Allocator,
    model_directory: []const u8,
) ![]const []const u8 {
    assert(model_directory.len > 0);
    assert(partial_extension.len > weights_extension.len);

    var directory = try std.fs.cwd().openDir(
        model_directory,
        .{ .iterate = true },
    );
    defer directory.close();

    var names = std.ArrayList([]const u8).init(allocator);
    var iterator = directory.iterate();
    while (try iterator.next()) |entry| {
        const is_weights = entry.kind == .file and
            (std.mem.endsWith(u8, entry.name, weights_extension) or
                std.mem.endsWith(u8, entry.name, partial_extension));
        if (is_weights) {
            try names.append(try allocator.dupe(u8, entry.name));
        } else {
            assert(!std.mem.endsWith(u8, entry.name, partial_extension));
        }
    }
    if (names.items.len == 0) {
        std.debug.print("manifest: no {s} files in {s}\n", .{
            weights_extension,
            model_directory,
        });
        return error.no_weights;
    } else {
        assert(names.items.len > 0);
    }

    const file_names = try names.toOwnedSlice();
    std.mem.sort([]const u8, file_names, {}, lessByName);
    assert(file_names.len > 0);
    return file_names;
}

/// Byte-wise name order.
fn lessByName(_: void, lhs: []const u8, rhs: []const u8) bool {
    assert(lhs.len > 0);
    assert(rhs.len > 0);

    return std.mem.lessThan(u8, lhs, rhs);
}

/// Maps and parses every file; a shard that is still downloading is reported
/// as incomplete instead of failing the build.
fn mapFiles(
    allocator: std.mem.Allocator,
    model_directory: []const u8,
    names: []const []const u8,
    files: []safetensors.File,
    headers: []safetensors.Header,
    reports: []FileReport,
) !void {
    assert(names.len == files.len);
    assert(names.len == headers.len);

    for (names, 0..) |name, file_index| {
        const path = try std.fs.path.join(
            allocator,
            &.{ model_directory, name },
        );
        files[file_index] = try safetensors.File.open(path);
        headers[file_index] = try safetensors.parseHeader(
            allocator,
            &files[file_index],
            @intCast(file_index),
        );
        const file = &files[file_index];
        reports[file_index] = .{
            .name = name,
            .header_bytes = file.header_len,
            .file_bytes = file.bytes.len,
            .data_present = headers[file_index].dataComplete(file),
        };
        if (reports[file_index].data_present) {
            assert(reports[file_index].file_bytes > file.dataStart());
        } else {
            std.debug.print("manifest: {s} is incomplete ({d} bytes)\n", .{
                name,
                file.bytes.len,
            });
        }
    }
    assert(reports.len == names.len);
}

/// Allocates the per-layer expert tables and the PLE shard table.
fn initLayers(allocator: std.mem.Allocator, manifest: *Manifest) !void {
    assert(manifest.layers.len > 0);
    assert(manifest.text.config.num_experts > 0);

    for (manifest.layers, 0..) |*layer, index| {
        layer.* = .{
            .index = @intCast(index),
            .kind = manifest.text.layerType(@intCast(index)),
            .experts = try allocator.alloc(
                Expert,
                manifest.text.config.num_experts,
            ),
        };
        for (layer.experts) |*expert| {
            expert.* = .{};
        }
        if (index == manifest.text.ple_layer_index) {
            layer.ple = .{
                .embedding_shards = try allocator.alloc(
                    TensorRef,
                    manifest.text.config.split_ngram_parts,
                ),
            };
        } else {
            assert(layer.ple == null);
        }
    }
    assert(manifest.layers[manifest.text.ple_layer_index].ple != null);
}

/// Attributes one tensor to a role and to the byte total of that role.
fn record(
    manifest: *Manifest,
    classified: Classified,
    tensor: *const safetensors.Tensor,
) void {
    assert(tensor.byte_len > 0);
    assert(@intFromEnum(classified.role) < manifest.totals.len);

    manifest.totals[@intFromEnum(classified.role)].add(tensor.byte_len);
    if (classified.role == .deferred) {
        manifest.deferred_tensors += 1;
        manifest.deferred_bytes += tensor.byte_len;
    } else {
        assert(classified.role != .deferred);
    }
}

/// The header must account for the file exactly once its data has arrived.
fn checkFileTotals(
    report: FileReport,
    file: *const safetensors.File,
    header: *const safetensors.Header,
) !void {
    assert(file.header_len > 0);
    assert(report.tensor_bytes > 0);

    if (report.data_present) {
        const expected = safetensors.length_prefix_bytes + file.header_len +
            report.tensor_bytes;
        if (expected != file.bytes.len) {
            std.debug.print("manifest: {s}: header says {d}, file is {d}\n", .{
                report.name,
                expected,
                file.bytes.len,
            });
            return error.file_size_mismatch;
        } else {
            assert(expected == file.bytes.len);
        }
    } else {
        assert(file.bytes.len < header.data_end);
    }
    assert(report.tensor_bytes == sumTensors(header));
}

/// Sum of the byte lengths the header declares.
fn sumTensors(header: *const safetensors.Header) u64 {
    assert(header.tensors.len > 0);
    assert(header.data_end > 0);

    var total: u64 = 0;
    for (header.tensors) |*tensor| {
        total += tensor.byte_len;
    }
    assert(total > 0);
    return total;
}

/// Checks dtype and extents, then stores the reference in `target`.
///
/// `extents` is an anonymous array literal of the exact expected shape, so a
/// call site reads like the tensor it describes. Filling a slot twice is a
/// bug (two tensors with the same role), so that assertion fires first.
fn place(
    target: *TensorRef,
    tensor: *const safetensors.Tensor,
    role: Role,
    dtype: safetensors.Dtype,
    extents: anytype,
) !void {
    assert(target.isEmpty());
    assert(extents.len <= safetensors.rank_max);

    if (tensor.dtype != dtype) {
        std.debug.print("manifest: {s} is {s}, expected {s}\n", .{
            tensor.name,
            @tagName(tensor.dtype),
            @tagName(dtype),
        });
        return error.wrong_dtype;
    } else {
        assert(tensor.dtype == dtype);
    }
    if (!tensor.dims.equals(&extents)) {
        std.debug.print("manifest: {s} has shape {any}, expected {any}\n", .{
            tensor.name,
            tensor.dims.values[0..tensor.dims.rank],
            extents,
        });
        return error.wrong_shape;
    } else {
        assert(tensor.dims.equals(&extents));
    }
    target.* = TensorRef.fromTensor(tensor);
    assert(!target.isEmpty());
    assert(@intFromEnum(role) < all_roles.len);
}

/// Classifies one top-level tensor name.
fn classify(
    manifest: *Manifest,
    tensor: *const safetensors.Tensor,
) !Classified {
    assert(tensor.name.len > 0);
    assert(manifest.layers.len > 0);

    const name = tensor.name;
    const model = manifest.text;
    const config_values = &model.config;
    if (std.mem.eql(u8, name, embed_tokens_name)) {
        try place(&manifest.embed_tokens, tensor, .embed_tokens, .bf16, .{
            config_values.vocab_size,
            config_values.hidden_size,
        });
        return .{ .role = .embed_tokens };
    } else if (std.mem.eql(u8, name, lm_head_name)) {
        try place(&manifest.lm_head, tensor, .lm_head, .bf16, .{
            config_values.vocab_size,
            config_values.hidden_size,
        });
        return .{ .role = .lm_head };
    } else if (std.mem.startsWith(u8, name, final_mixer_prefix)) {
        const mixer = &manifest.final_mixer;
        const tail = name[final_mixer_prefix.len..];
        try fillFinalMixer(mixer, tensor, tail, &model);
        return .{ .role = .hyper_connection_mixer };
    } else if (std.mem.startsWith(u8, name, layers_prefix)) {
        return try classifyLayer(manifest, tensor);
    } else if (isDeferred(name)) {
        // Recognised, counted, unused: `mtp` and the vision tower.
        return .{ .role = .deferred };
    } else {
        std.debug.print("manifest: unrecognised tensor {s}\n", .{name});
        return error.unknown_tensor;
    }
}

/// True for the sub-trees that are not part of the text hot path yet.
fn isDeferred(name: []const u8) bool {
    assert(deferred_prefixes.len > 0);
    assert(name.len > 0);

    for (deferred_prefixes) |prefix| {
        if (std.mem.startsWith(u8, name, prefix)) {
            return true;
        } else {
            continue;
        }
    }
    return false;
}

/// The final hyper-connection mixer has no block injection weight.
fn fillFinalMixer(
    mixer: *FinalMixer,
    tensor: *const safetensors.Tensor,
    tail: []const u8,
    model: *const config.TextModel,
) !void {
    assert(tail.len > 0);
    assert(model.hc_hidden_size > 0);

    const hidden = model.config.hidden_size;
    const lowrank = model.config.hc_lowrank;
    if (std.mem.eql(u8, tail, "hc_norm.weight")) {
        try place(
            &mixer.norm,
            tensor,
            .hyper_connection_mixer,
            .bf16,
            .{model.hc_hidden_size},
        );
    } else if (std.mem.eql(u8, tail, "input_mix_weight_down.weight")) {
        try place(&mixer.mix_down, tensor, .hyper_connection_mixer, .bf16, .{
            lowrank,
            model.hc_hidden_size,
        });
    } else if (std.mem.eql(u8, tail, "input_mix_weight_up.weight")) {
        try place(&mixer.mix_up, tensor, .hyper_connection_mixer, .bf16, .{
            model.hc_hidden_size,
            lowrank,
        });
    } else {
        std.debug.print(
            "manifest: unexpected final mixer tensor {s}\n",
            .{tail},
        );
        return error.unknown_tensor;
    }
    assert(model.hc_hidden_size == model.config.hc_count * hidden);
}

/// A layer tensor: dispatch on the module that owns it.
fn classifyLayer(
    manifest: *Manifest,
    tensor: *const safetensors.Tensor,
) !Classified {
    assert(std.mem.startsWith(u8, tensor.name, layers_prefix));
    assert(manifest.layers.len == manifest.text.config.num_hidden_layers);

    const parsed = try parseIndexedName(tensor.name[layers_prefix.len..]);
    if (parsed.index >= manifest.layers.len) {
        std.debug.print("manifest: {s} is past layer {d}\n", .{
            tensor.name,
            manifest.layers.len,
        });
        return error.unknown_tensor;
    } else {
        assert(parsed.index < manifest.layers.len);
    }
    const layer = &manifest.layers[parsed.index];
    const model = &manifest.text;
    const tail = parsed.tail;

    var role: Role = .deferred;
    if (std.mem.startsWith(u8, tail, linear_attn_prefix)) {
        role = try fillLinearAttention(
            layer,
            tensor,
            tail[linear_attn_prefix.len..],
            model,
        );
    } else if (std.mem.startsWith(u8, tail, self_attn_prefix)) {
        role = try fillSelfAttention(
            layer,
            tensor,
            tail[self_attn_prefix.len..],
            model,
        );
    } else if (std.mem.startsWith(u8, tail, attn_hc_prefix)) {
        role = try fillHyperConnection(
            &layer.attn_hyper_connection,
            tensor,
            tail[attn_hc_prefix.len..],
            model,
            .attn_hyper_connection,
        );
    } else if (std.mem.startsWith(u8, tail, mlp_hc_prefix)) {
        role = try fillHyperConnection(
            &layer.mlp_hyper_connection,
            tensor,
            tail[mlp_hc_prefix.len..],
            model,
            .mlp_hyper_connection,
        );
    } else if (std.mem.startsWith(u8, tail, experts_prefix)) {
        role = try fillExpert(layer, tensor, tail[experts_prefix.len..], model);
    } else if (std.mem.startsWith(u8, tail, ple_prefix)) {
        role = try fillPle(layer, tensor, tail[ple_prefix.len..], model);
    } else if (std.mem.startsWith(u8, tail, shared_expert_prefix)) {
        role = try fillSharedExpert(
            layer,
            tensor,
            tail[shared_expert_prefix.len..],
            model,
        );
    } else if (std.mem.eql(u8, tail, router_gate_tail)) {
        try place(&layer.router_gate, tensor, .router_gate, .bf16, .{
            model.config.num_experts,
            model.config.hidden_size,
        });
        role = .router_gate;
    } else if (std.mem.eql(u8, tail, shared_expert_gate_tail)) {
        try place(
            &layer.shared_expert_gate,
            tensor,
            .shared_expert_gate,
            .bf16,
            .{ 1, model.config.hidden_size },
        );
        role = .shared_expert_gate;
    } else {
        std.debug.print(
            "manifest: unrecognised layer tensor {s}\n",
            .{tensor.name},
        );
        return error.unknown_tensor;
    }
    assert(role != .deferred);
    return .{ .role = role, .layer = parsed.index };
}

const linear_attn_prefix = "linear_attn.";
const self_attn_prefix = "self_attn.";
const attn_hc_prefix = "attn_hyper_connection.";
const mlp_hc_prefix = "mlp_hyper_connection.";
const experts_prefix = "mlp.experts.";
const ple_prefix = "ple.";
const shared_expert_prefix = "mlp.shared_expert.";
const router_gate_tail = "mlp.gate.weight";
const shared_expert_gate_tail = "mlp.shared_expert_gate.weight";

/// A name of the form `{digits}.{tail}`; `tail` may itself be indexed.
const IndexedName = struct {
    index: u32,
    tail: []const u8,
};

/// Splits a leading decimal index off a dotted path.
fn parseIndexedName(name: []const u8) !IndexedName {
    assert(name.len > 0);
    assert(name.len < 128);

    const dot = std.mem.indexOfScalar(u8, name, '.') orelse {
        std.debug.print("manifest: {s} has no index\n", .{name});
        return error.unknown_tensor;
    };
    if (dot == 0) {
        return error.unknown_tensor;
    } else {
        assert(dot > 0);
    }
    const index = std.fmt.parseInt(u32, name[0..dot], 10) catch {
        std.debug.print("manifest: {s} has a bad index\n", .{name});
        return error.unknown_tensor;
    };
    const tail = name[dot + 1 ..];
    if (tail.len == 0) {
        return error.unknown_tensor;
    } else {
        assert(tail.len > 0);
    }
    assert(tail.len < name.len);
    return .{ .index = index, .tail = tail };
}

/// GatedDeltaNet tensors, all BF16.
fn fillLinearAttention(
    layer: *Layer,
    tensor: *const safetensors.Tensor,
    tail: []const u8,
    model: *const config.TextModel,
) !Role {
    assert(tail.len > 0);
    assert(model.linear_qkv_rows == model.hc_hidden_size);

    const attention = &layer.linear_attn;
    const hidden = model.config.hidden_size;
    const value_heads = model.config.linear_num_value_heads;
    const head_dim = model.config.linear_value_head_dim;
    if (std.mem.eql(u8, tail, "in_proj_qkv.weight")) {
        try place(&attention.in_proj_qkv, tensor, .linear_attn, .bf16, .{
            model.linear_qkv_rows,
            hidden,
        });
    } else if (std.mem.eql(u8, tail, "in_proj_z.weight")) {
        try place(&attention.in_proj_z, tensor, .linear_attn, .bf16, .{
            model.linear_z_rows,
            hidden,
        });
    } else if (std.mem.eql(u8, tail, "in_proj_a.weight")) {
        try place(&attention.in_proj_a, tensor, .linear_attn, .bf16, .{
            value_heads, hidden,
        });
    } else if (std.mem.eql(u8, tail, "in_proj_b.weight")) {
        try place(
            &attention.in_proj_b,
            tensor,
            .linear_attn,
            .bf16,
            .{ value_heads, hidden },
        );
    } else if (std.mem.eql(u8, tail, "out_proj.weight")) {
        try place(&attention.out_proj, tensor, .linear_attn, .bf16, .{
            hidden,
            model.linear_z_rows,
        });
    } else if (std.mem.eql(u8, tail, "conv1d.weight")) {
        try place(&attention.conv1d, tensor, .linear_attn, .bf16, .{
            model.linear_qkv_rows,
            1,
            model.config.linear_conv_kernel_dim,
        });
    } else if (std.mem.eql(u8, tail, "A_log")) {
        try place(
            &attention.a_log,
            tensor,
            .linear_attn,
            .bf16,
            .{value_heads},
        );
    } else if (std.mem.eql(u8, tail, "dt_bias")) {
        try place(
            &attention.dt_bias,
            tensor,
            .linear_attn,
            .bf16,
            .{value_heads},
        );
    } else if (std.mem.eql(u8, tail, "norm.weight")) {
        try place(&attention.norm, tensor, .linear_attn, .bf16, .{head_dim});
    } else {
        std.debug.print(
            "manifest: unexpected linear_attn tensor {s}\n",
            .{tail},
        );
        return error.unknown_tensor;
    }
    assert(layer.kind == .linear_attention or layer.kind == .full_attention);
    return .linear_attn;
}

/// Indexed attention tensors, including the QSA indexer.
fn fillSelfAttention(
    layer: *Layer,
    tensor: *const safetensors.Tensor,
    tail: []const u8,
    model: *const config.TextModel,
) !Role {
    assert(tail.len > 0);
    assert(model.q_proj_rows == 2 * model.o_proj_cols);

    const attention = &layer.self_attn;
    const hidden = model.config.hidden_size;
    const head_dim = model.config.head_dim;
    const indexer_dim = model.config.indexer_head_dim;
    if (std.mem.eql(u8, tail, "q_proj.weight")) {
        // A query block and an output gate per head, interleaved.
        try place(
            &attention.q_proj,
            tensor,
            .q_proj,
            .bf16,
            .{ model.q_proj_rows, hidden },
        );
        return .q_proj;
    } else if (std.mem.eql(u8, tail, "k_proj.weight")) {
        try place(
            &attention.k_proj,
            tensor,
            .k_proj,
            .bf16,
            .{ model.kv_proj_rows, hidden },
        );
        return .k_proj;
    } else if (std.mem.eql(u8, tail, "v_proj.weight")) {
        try place(
            &attention.v_proj,
            tensor,
            .v_proj,
            .bf16,
            .{ model.kv_proj_rows, hidden },
        );
        return .v_proj;
    } else if (std.mem.eql(u8, tail, "o_proj.weight")) {
        try place(
            &attention.o_proj,
            tensor,
            .o_proj,
            .bf16,
            .{ hidden, model.o_proj_cols },
        );
        return .o_proj;
    } else if (std.mem.eql(u8, tail, "q_norm.weight")) {
        try place(&attention.q_norm, tensor, .attn_norm, .bf16, .{head_dim});
        return .attn_norm;
    } else if (std.mem.eql(u8, tail, "k_norm.weight")) {
        try place(&attention.k_norm, tensor, .attn_norm, .bf16, .{head_dim});
        return .attn_norm;
    } else if (std.mem.eql(u8, tail, "indexer.index_qk_proj.weight")) {
        try place(&attention.indexer_qk_proj, tensor, .indexer, .bf16, .{
            model.indexer_qk_rows,
            hidden,
        });
        return .indexer;
    } else if (std.mem.eql(u8, tail, "indexer.q_layernorm.weight")) {
        try place(
            &attention.indexer_q_norm,
            tensor,
            .indexer,
            .bf16,
            .{indexer_dim},
        );
        return .indexer;
    } else if (std.mem.eql(u8, tail, "indexer.k_layernorm.weight")) {
        try place(
            &attention.indexer_k_norm,
            tensor,
            .indexer,
            .bf16,
            .{indexer_dim},
        );
        return .indexer;
    } else {
        std.debug.print("manifest: unexpected self_attn tensor {s}\n", .{tail});
        return error.unknown_tensor;
    }
}

/// One hyper-connection block: mixing weights on the 4 x hidden stream.
fn fillHyperConnection(
    target: *HyperConnection,
    tensor: *const safetensors.Tensor,
    tail: []const u8,
    model: *const config.TextModel,
    role: Role,
) !Role {
    assert(tail.len > 0);
    assert(model.config.hc_lowrank > 0);

    const lowrank = model.config.hc_lowrank;
    const width = model.hc_hidden_size;
    if (std.mem.eql(u8, tail, "block_inject_weight.weight")) {
        try place(&target.block_inject, tensor, role, .bf16, .{
            model.config.hc_count,
            width,
        });
    } else if (std.mem.eql(u8, tail, "hc_norm.weight")) {
        try place(&target.norm, tensor, role, .bf16, .{width});
    } else if (std.mem.eql(u8, tail, "input_mix_weight_down.weight")) {
        try place(&target.mix_down, tensor, role, .bf16, .{ lowrank, width });
    } else if (std.mem.eql(u8, tail, "input_mix_weight_up.weight")) {
        try place(&target.mix_up, tensor, role, .bf16, .{ width, lowrank });
    } else {
        std.debug.print(
            "manifest: unexpected hyper connection tensor {s}\n",
            .{tail},
        );
        return error.unknown_tensor;
    }
    assert(role == .attn_hyper_connection or role == .mlp_hyper_connection);
    return role;
}

/// The shared expert of a layer: BF16, unlike the routed experts.
fn fillSharedExpert(
    layer: *Layer,
    tensor: *const safetensors.Tensor,
    tail: []const u8,
    model: *const config.TextModel,
) !Role {
    assert(tail.len > 0);
    assert(model.config.shared_expert_intermediate_size > 0);

    const shared = &layer.shared_expert;
    const hidden = model.config.hidden_size;
    const intermediate = model.config.shared_expert_intermediate_size;
    if (std.mem.eql(u8, tail, "gate_proj.weight")) {
        try place(
            &shared.gate_proj,
            tensor,
            .shared_expert,
            .bf16,
            .{ intermediate, hidden },
        );
    } else if (std.mem.eql(u8, tail, "up_proj.weight")) {
        try place(
            &shared.up_proj,
            tensor,
            .shared_expert,
            .bf16,
            .{ intermediate, hidden },
        );
    } else if (std.mem.eql(u8, tail, "down_proj.weight")) {
        try place(
            &shared.down_proj,
            tensor,
            .shared_expert,
            .bf16,
            .{ hidden, intermediate },
        );
    } else {
        std.debug.print(
            "manifest: unexpected shared expert tensor {s}\n",
            .{tail},
        );
        return error.unknown_tensor;
    }
    assert(intermediate == model.config.shared_expert_intermediate_size);
    return .shared_expert;
}

/// One routed expert's NVFP4 projection: packed pairs plus three scales.
///
/// Every expert tensor is `{proj}.{field}`; the three projections differ only
/// in
/// which extent is contracted, so the field handling is shared.
fn fillExpert(
    layer: *Layer,
    tensor: *const safetensors.Tensor,
    tail: []const u8,
    model: *const config.TextModel,
) !Role {
    assert(tail.len > 0);
    assert(quant.nvfp4_block_size == 16);

    const parsed = try parseIndexedName(tail);
    if (parsed.index >= layer.experts.len) {
        std.debug.print("manifest: expert {d} of {d}\n", .{
            parsed.index,
            layer.experts.len,
        });
        return error.unknown_tensor;
    } else {
        assert(parsed.index < layer.experts.len);
    }
    const expert = &layer.experts[parsed.index];

    const dot = std.mem.indexOfScalar(u8, parsed.tail, '.') orelse {
        return error.unknown_tensor;
    };
    const name = parsed.tail[0..dot];
    const field = parsed.tail[dot + 1 ..];
    const hidden = model.config.hidden_size;
    const intermediate = model.config.moe_intermediate_size;

    const is_down = std.mem.eql(u8, name, "down_proj");
    const projection = if (std.mem.eql(u8, name, "gate_proj"))
        &expert.gate_proj
    else if (std.mem.eql(u8, name, "up_proj"))
        &expert.up_proj
    else if (is_down)
        &expert.down_proj
    else
        return error.unknown_tensor;
    // gate and up map hidden -> intermediate, down maps it back.
    const rows = if (is_down) hidden else intermediate;
    const contracted = if (is_down) intermediate else hidden;
    try placeExpertField(projection, tensor, field, rows, contracted);
    assert(contracted > 0);
    return .expert;
}

/// Stores one of the four tensors of an NVFP4 projection.
///
/// The packed weight keeps two weights per byte, so its contracted extent
/// halves; the E4M3 block scale covers 16 weights along that same dimension.
fn placeExpertField(
    projection: *QuantizedProjection,
    tensor: *const safetensors.Tensor,
    field: []const u8,
    rows: u32,
    contracted: u32,
) !void {
    assert(field.len > 0);
    assert(contracted % quant.nvfp4_block_size == 0);

    const packed_columns = contracted / 2;
    const scale_columns = contracted / quant.nvfp4_block_size;
    if (std.mem.eql(u8, field, "weight")) {
        try place(
            &projection.weight,
            tensor,
            .expert,
            .u8,
            .{ rows, packed_columns },
        );
    } else if (std.mem.eql(u8, field, "weight_scale")) {
        try place(
            &projection.weight_scale,
            tensor,
            .expert,
            .f8_e4m3,
            .{ rows, scale_columns },
        );
    } else if (std.mem.eql(u8, field, "weight_scale_2")) {
        try place(&projection.weight_scale_2, tensor, .expert, .f32, .{});
    } else if (std.mem.eql(u8, field, "input_scale")) {
        try place(&projection.input_scale, tensor, .expert, .f32, .{});
    } else {
        std.debug.print("manifest: unexpected expert field {s}\n", .{field});
        return error.unknown_tensor;
    }
    assert(packed_columns * 2 == contracted);
}

/// The PLE layer: n-gram table shards, index buffers and projections.
fn fillPle(
    layer: *Layer,
    tensor: *const safetensors.Tensor,
    tail: []const u8,
    model: *const config.TextModel,
) !Role {
    assert(tail.len > 0);
    assert(model.ngram_heads > 0);

    const ple = if (layer.ple) |*payload| payload else {
        std.debug.print(
            "manifest: {s} is not on a PLE layer\n",
            .{tensor.name},
        );
        return error.unknown_tensor;
    };
    const width = model.hc_hidden_size;
    const embed_dim = model.config.ple_embed_dim;
    const heads = model.ngram_heads;
    var role: Role = .ple_projection;
    if (std.mem.eql(u8, tail, "key_proj.weight")) {
        try place(
            &ple.key_proj,
            tensor,
            .ple_projection,
            .bf16,
            .{ width, embed_dim },
        );
    } else if (std.mem.eql(u8, tail, "value_proj.weight")) {
        try place(&ple.value_proj, tensor, .ple_projection, .bf16, .{
            model.config.hidden_size,
            embed_dim,
        });
    } else if (std.mem.eql(u8, tail, "conv1d.weight")) {
        role = try placePleConv1d(ple, tensor, model, width);
    } else if (std.mem.eql(u8, tail, "norm_key.weight")) {
        try place(&ple.norm_key, tensor, .ple_norm, .bf16, .{width});
        role = .ple_norm;
    } else if (std.mem.eql(u8, tail, "norm_query.weight")) {
        try place(&ple.norm_query, tensor, .ple_norm, .bf16, .{width});
        role = .ple_norm;
    } else if (std.mem.eql(u8, tail, "norm_conv.weight")) {
        try place(&ple.norm_conv, tensor, .ple_norm, .bf16, .{width});
        role = .ple_norm;
    } else if (std.mem.eql(u8, tail, "ple_embedding.ngram_heads_offsets")) {
        try place(&ple.heads_offsets, tensor, .ngram_index, .i64, .{heads});
        role = .ngram_index;
    } else if (std.mem.eql(u8, tail, "ple_embedding.ngram_heads_vocab_sizes")) {
        try place(&ple.heads_vocab_sizes, tensor, .ngram_index, .i64, .{heads});
        role = .ngram_index;
    } else if (std.mem.eql(u8, tail, "ple_embedding.layer_multipliers")) {
        try place(&ple.layer_multipliers, tensor, .ngram_index, .i64, .{
            model.config.ngram_size,
        });
        role = .ngram_index;
    } else if (std.mem.eql(
        u8,
        tail,
        "ple_embedding.ngram_embedding.weight_scale",
    )) {
        // One BF16 scale for the whole table.
        try place(&ple.embedding_scale, tensor, .ngram_scale, .bf16, .{1});
        role = .ngram_scale;
    } else if (std.mem.startsWith(u8, tail, shard_prefix)) {
        role = try placeNgramShard(
            ple,
            tensor,
            tail[shard_prefix.len..],
            model,
        );
    } else {
        std.debug.print("manifest: unexpected PLE tensor {s}\n", .{tail});
        return error.unknown_tensor;
    }
    assert(role != .deferred);
    return role;
}

const shard_prefix = "ple_embedding.ngram_embedding.shard_";

/// The dilated depthwise conv over the PLE stream.
fn placePleConv1d(
    ple: *Ple,
    tensor: *const safetensors.Tensor,
    model: *const config.TextModel,
    width: u32,
) !Role {
    assert(width > 0);
    assert(model.config.ple_conv_kernel_size > 0);

    try place(&ple.conv1d, tensor, .ple_conv1d, .bf16, .{
        width,
        1,
        model.config.ple_conv_kernel_size,
    });
    return .ple_conv1d;
}

/// One of the `split_ngram_parts` row ranges of the logical n-gram table.
fn placeNgramShard(
    ple: *Ple,
    tensor: *const safetensors.Tensor,
    tail: []const u8,
    model: *const config.TextModel,
) !Role {
    assert(tail.len > 0);
    assert(model.config.split_ngram_parts > 0);

    const parsed = try parseIndexedName(tail);
    if (parsed.index >= ple.embedding_shards.len) {
        return error.unknown_tensor;
    } else {
        assert(parsed.index < ple.embedding_shards.len);
    }
    if (!std.mem.eql(u8, parsed.tail, "weight")) {
        return error.unknown_tensor;
    } else {
        assert(parsed.tail.len > 0);
    }
    // The row count depends on the n-gram vocabularies, which are validated
    // after every file has been walked (`validateNgramTable`).
    if (tensor.dtype != .f8_e4m3) {
        std.debug.print("manifest: {s} is {s}, expected F8_E4M3\n", .{
            tensor.name,
            @tagName(tensor.dtype),
        });
        return error.wrong_dtype;
    } else {
        assert(tensor.dtype == .f8_e4m3);
    }
    if (tensor.dims.rank != 2) {
        return error.wrong_shape;
    } else {
        assert(tensor.dims.rank == 2);
    }
    ple.embedding_shards[parsed.index] = TensorRef.fromTensor(tensor);
    assert(!ple.embedding_shards[parsed.index].isEmpty());
    return .ngram_table;
}

/// Little endian decoding of an I64 buffer, used for the n-gram index.
fn readI64(allocator: std.mem.Allocator, bytes: []const u8) ![]i64 {
    assert(bytes.len > 0);
    assert(bytes.len % 8 == 0);

    const values = try allocator.alloc(i64, bytes.len / 8);
    for (values, 0..) |*value, index| {
        value.* = std.mem.readInt(i64, bytes[index * 8 ..][0..8], .little);
    }
    assert(values.len * 8 == bytes.len);
    return values;
}

/// Derives the logical n-gram table shape from the head vocabularies, and
/// checks that the 128 shard tensors tile it.
///
/// The head sizes are primes, so the concatenated table is padded up to a
/// multiple of `make_ngram_vocab_size_divisible_by` and then split into
/// `split_ngram_parts` equal row ranges. That is how the reference builds it,
/// and the padding is why the shard row count is not a round number of
/// millions.
/// Sum of the per-head vocabularies, with the offset buffer cross-checked
/// against them: the table is one array of rows, so offsets must be prefixes.
fn ngramTableRowCount(
    allocator: std.mem.Allocator,
    loaded: *const Loaded,
    ple: *const Ple,
    model: *const config.TextModel,
) !i64 {
    assert(model.ngram_heads > 0);
    assert(loaded.files.len > 0);

    const head_count: usize = @intCast(model.ngram_heads);
    const sizes = try readI64(
        allocator,
        loaded.tensorBytes(ple.heads_vocab_sizes),
    );
    const offsets = try readI64(
        allocator,
        loaded.tensorBytes(ple.heads_offsets),
    );
    if (sizes.len != head_count) {
        return error.wrong_shape;
    } else {
        assert(sizes.len == head_count);
    }
    if (offsets.len != head_count) {
        return error.wrong_shape;
    } else {
        assert(offsets.len == head_count);
    }

    var total: i64 = 0;
    for (sizes, offsets) |size, offset| {
        if (size <= 0) {
            return error.wrong_shape;
        } else {
            assert(size > 0);
        }
        if (offset != total) {
            std.debug.print("manifest: n-gram offset {d}, expected {d}\n", .{
                offset,
                total,
            });
            return error.wrong_shape;
        } else {
            assert(offset == total);
        }
        total += size;
    }
    assert(total > 0);
    return total;
}

fn validateNgramTable(allocator: std.mem.Allocator, loaded: *Loaded) !void {
    assert(loaded.manifest.layers.len > 0);
    assert(loaded.manifest.text.ngram_embed_dim > 0);

    const manifest = &loaded.manifest;
    const model = &manifest.text;
    const ple_layer = &manifest.layers[model.ple_layer_index];
    const ple = if (ple_layer.ple) |*payload| payload else {
        return error.unknown_tensor;
    };
    if (ple.heads_vocab_sizes.isEmpty()) {
        return error.unknown_tensor;
    } else {
        assert(!ple.heads_vocab_sizes.isEmpty());
    }

    const total = try ngramTableRowCount(allocator, loaded, ple, model);
    const divisor = model.config.make_ngram_vocab_size_divisible_by;
    const padded = blk: {
        const quotient = std.math.divCeil(i64, total, @as(i64, divisor)) catch
            return error.wrong_shape;
        break :blk quotient * @as(i64, divisor);
    };
    const parts = model.config.split_ngram_parts;
    if (@rem(padded, @as(i64, parts)) != 0) {
        return error.wrong_shape;
    } else {
        assert(@rem(padded, @as(i64, parts)) == 0);
    }
    manifest.ngram_table_rows = @intCast(padded);
    manifest.ngram_shard_rows = @intCast(@divExact(padded, parts));

    for (ple.embedding_shards) |shard| {
        if (shard.isEmpty()) {
            return error.unknown_tensor;
        } else {
            assert(!shard.isEmpty());
        }
        if (!shard.dims.equals(&.{
            @as(u32, @intCast(manifest.ngram_shard_rows)),
            model.ngram_embed_dim,
        })) {
            std.debug.print("manifest: n-gram shard shape {any}\n", .{
                shard.dims.values[0..shard.dims.rank],
            });
            return error.wrong_shape;
        } else {
            assert(shard.dims.rank == 2);
        }
    }
    assert(manifest.ngram_table_rows > 0);
}

/// The headline invariant: the per-role totals account for every tensor and for
/// every byte the checkpoint's own index claims, with nothing counted twice.
///
/// The derived reporting aggregates (`kv_proj_total`, `self_attn`) deliberately
/// overlap roles, so they are not part of `totals` and never enter this sum.
fn reconcileBytes(loaded: *const Loaded, tensor_total: u64) !void {
    assert(tensor_total > 0);
    assert(loaded.manifest.index_total_size > 0);

    const manifest = &loaded.manifest;
    var role_tensors: u64 = 0;
    var role_bytes: u64 = 0;
    for (manifest.totals) |total| {
        role_tensors += total.count;
        role_bytes += total.bytes;
    }
    if (role_tensors != tensor_total) {
        return error.incomplete_model;
    } else {
        assert(role_tensors == tensor_total);
    }
    if (role_bytes != loaded.tensorBytesTotal()) {
        return error.incomplete_model;
    } else {
        assert(role_bytes == loaded.tensorBytesTotal());
    }
    if (role_bytes != manifest.index_total_size) {
        std.debug.print("manifest: {d} bytes classified, index claims {d}\n", .{
            role_bytes,
            manifest.index_total_size,
        });
        return error.index_mismatch;
    } else {
        assert(role_bytes == manifest.index_total_size);
    }
    assert(role_bytes > 0);
}

/// Every slot the model graph reads must be filled exactly once.
fn validateCompleteness(
    loaded: *const Loaded,
    layer_counts: []const u64,
    tensor_total: u64,
) !void {
    assert(layer_counts.len == loaded.manifest.layers.len);
    assert(tensor_total > 0);

    const manifest = &loaded.manifest;
    const model = &manifest.text;
    const config_values = &model.config;
    var expected: u64 = 5; // embed_tokens, lm_head and three mixer tensors.
    for (layer_counts, 0..) |count, index| {
        const layer = &manifest.layers[index];
        const has_ple = index == model.ple_layer_index;
        const layer_expected = expectedLayerTensorCount(model, has_ple);
        if (count != layer_expected) {
            std.debug.print(
                "manifest: layer {d} has {d} tensors, expected {d}\n",
                .{ index, count, layer_expected },
            );
            return error.incomplete_layer;
        } else {
            assert(count == layer_expected);
        }
        try checkLayerSlots(layer, model);
        expected += layer_expected;
    }
    if (tensor_total != expected + manifest.deferred_tensors) {
        std.debug.print(
            "manifest: {d} tensors, expected {d} text + {d} deferred\n",
            .{ tensor_total, expected, manifest.deferred_tensors },
        );
        return error.incomplete_model;
    } else {
        assert(tensor_total == expected + manifest.deferred_tensors);
    }

    try reconcileBytes(loaded, tensor_total);
    // The routed experts are the bulk of the checkpoint: 512 experts, three
    // projections, four tensors each, on every layer.
    const expert_tensors = @as(u64, config_values.num_hidden_layers) *
        config_values.num_experts * 3 * 4;
    if (manifest.total(.expert).count != expert_tensors) {
        return error.incomplete_model;
    } else {
        assert(manifest.total(.expert).count == expert_tensors);
    }
    assert(manifest.lm_head.dims.equals(&.{
        config_values.vocab_size,
        config_values.hidden_size,
    }));
}

/// Tensors of one layer, from the closed set of roles that layer must have.
fn expectedLayerTensorCount(model: *const config.TextModel, has_ple: bool) u64 {
    assert(model.config.num_experts > 0);
    assert(model.config.split_ngram_parts > 0);

    // Two hyper connections (4 each), the router, the shared expert and its
    // gate, the expert stacks, and the attention block (9 tensors, linear or
    // self attention).
    const projections: u64 = 4;
    var count: u64 = 4 + 4 + 1 + 3 + 1 + 9;
    count += @as(u64, model.config.num_experts) * 3 * projections;
    if (has_ple) {
        // Six projections and norms, three index buffers, the table scale, and
        // one tensor per shard.
        count += 6 + 3 + 1 + model.config.split_ngram_parts;
    } else {
        assert(!has_ple);
    }
    assert(count >= 9);
    return count;
}

/// True when every `TensorRef` field of `value` has been filled.
fn allFilled(comptime T: type, value: *const T) bool {
    assert(std.meta.fields(T).len > 0);
    assert(@sizeOf(T) > 0);

    inline for (std.meta.fields(T)) |field| {
        if (@TypeOf(@field(value, field.name)) == TensorRef) {
            if (@field(value, field.name).isEmpty()) {
                return false;
            } else {
                assert(!@field(value, field.name).isEmpty());
            }
        } else {
            assert(@TypeOf(@field(value, field.name)) != TensorRef);
        }
    }
    return true;
}

/// True when every `TensorRef` field of `value` is still empty.
fn allEmpty(comptime T: type, value: *const T) bool {
    assert(std.meta.fields(T).len > 0);
    assert(@sizeOf(T) > 0);

    inline for (std.meta.fields(T)) |field| {
        if (@TypeOf(@field(value, field.name)) == TensorRef) {
            if (!@field(value, field.name).isEmpty()) {
                return false;
            } else {
                assert(@field(value, field.name).isEmpty());
            }
        } else {
            assert(@TypeOf(@field(value, field.name)) != TensorRef);
        }
    }
    return true;
}

/// Checks that the slots this layer's kind uses are filled and the others
/// empty.
fn checkLayerSlots(layer: *const Layer, model: *const config.TextModel) !void {
    assert(layer.index < model.config.num_hidden_layers);
    assert(layer.experts.len == model.config.num_experts);

    const attention_filled = if (layer.kind == .linear_attention)
        allFilled(LinearAttention, &layer.linear_attn)
    else
        allFilled(SelfAttention, &layer.self_attn);
    const other_empty = if (layer.kind == .linear_attention)
        allEmpty(SelfAttention, &layer.self_attn)
    else
        allEmpty(LinearAttention, &layer.linear_attn);
    if (!attention_filled) {
        return error.incomplete_layer;
    } else {
        assert(attention_filled);
    }
    if (!other_empty) {
        return error.incomplete_layer;
    } else {
        assert(other_empty);
    }
    if (!allFilled(HyperConnection, &layer.attn_hyper_connection)) {
        return error.incomplete_layer;
    } else {
        assert(allFilled(HyperConnection, &layer.mlp_hyper_connection));
    }
    if (!allFilled(SharedExpert, &layer.shared_expert)) {
        return error.incomplete_layer;
    } else {
        assert(!layer.router_gate.isEmpty());
    }
    for (layer.experts) |*expert| {
        if (!allFilled(QuantizedProjection, &expert.gate_proj)) {
            return error.incomplete_expert;
        } else {
            assert(allFilled(QuantizedProjection, &expert.up_proj));
        }
        if (!allFilled(QuantizedProjection, &expert.down_proj)) {
            return error.incomplete_expert;
        } else {
            assert(!expert.down_proj.weight.isEmpty());
        }
    }
    assert(layer.shared_expert_gate.byte_len > 0);
}

/// Renders the manifest document: dimensions, per-role totals, per-file totals.
pub fn renderJson(allocator: std.mem.Allocator, loaded: *const Loaded) ![]u8 {
    assert(loaded.files.len > 0);
    assert(loaded.manifest.layers.len > 0);

    var buffer = std.ArrayList(u8).init(allocator);
    errdefer buffer.deinit();
    const writer = buffer.writer();

    try writer.writeAll("{\n");
    try writeDims(writer, &loaded.manifest);
    try writeRoles(writer, &loaded.manifest);
    try writeFiles(writer, loaded);
    try writeSummary(writer, loaded);
    try writeReconciliation(writer, loaded);
    try writer.writeAll("}\n");

    const document = try buffer.toOwnedSlice();
    assert(document.len > 0);
    assert(std.mem.endsWith(u8, document, "}\n"));
    return document;
}

/// Writes the integer dimensions the kernels are sized from.
fn writeDims(writer: anytype, manifest: *const Manifest) !void {
    assert(manifest.layers.len > 0);
    assert(manifest.text.hc_hidden_size > 0);

    const model = &manifest.text;
    const config_values = &model.config;
    try writer.print("  \"dims\": {{\n", .{});
    try writer.print(
        "    \"num_hidden_layers\": {d},\n",
        .{config_values.num_hidden_layers},
    );
    try writer.print(
        "    \"hidden_size\": {d},\n",
        .{config_values.hidden_size},
    );
    try writer.print("    \"head_dim\": {d},\n", .{config_values.head_dim});
    try writer.print("    \"vocab_size\": {d},\n", .{config_values.vocab_size});
    try writer.print(
        "    \"num_experts\": {d},\n",
        .{config_values.num_experts},
    );
    try writer.print("    \"num_experts_per_tok\": {d},\n", .{
        config_values.num_experts_per_tok,
    });
    try writer.print("    \"moe_intermediate_size\": {d},\n", .{
        config_values.moe_intermediate_size,
    });
    try writer.print("    \"shared_expert_intermediate_size\": {d},\n", .{
        config_values.shared_expert_intermediate_size,
    });
    try writer.print("    \"hc_count\": {d},\n", .{config_values.hc_count});
    try writer.print("    \"hc_lowrank\": {d},\n", .{config_values.hc_lowrank});
    try writer.print("    \"hc_hidden_size\": {d},\n", .{model.hc_hidden_size});
    try writer.print("    \"q_proj_rows\": {d},\n", .{model.q_proj_rows});
    try writer.print("    \"kv_proj_rows\": {d},\n", .{model.kv_proj_rows});
    try writer.print("    \"o_proj_cols\": {d},\n", .{model.o_proj_cols});
    try writer.print(
        "    \"linear_qkv_rows\": {d},\n",
        .{model.linear_qkv_rows},
    );
    try writer.print("    \"linear_z_rows\": {d},\n", .{model.linear_z_rows});
    try writer.print(
        "    \"indexer_qk_rows\": {d},\n",
        .{model.indexer_qk_rows},
    );
    try writer.print("    \"linear_attention_layers\": {d},\n", .{
        model.linear_attention_layers,
    });
    try writer.print("    \"full_attention_layers\": {d},\n", .{
        model.full_attention_layers,
    });
    try writer.print(
        "    \"ple_layer_index\": {d},\n",
        .{model.ple_layer_index},
    );
    try writer.print(
        "    \"ple_embed_dim\": {d},\n",
        .{config_values.ple_embed_dim},
    );
    try writer.print("    \"ngram_heads\": {d},\n", .{model.ngram_heads});
    try writer.print(
        "    \"ngram_embed_dim\": {d},\n",
        .{model.ngram_embed_dim},
    );
    try writer.print(
        "    \"ngram_table_rows\": {d},\n",
        .{manifest.ngram_table_rows},
    );
    try writer.print(
        "    \"ngram_shard_rows\": {d},\n",
        .{manifest.ngram_shard_rows},
    );
    try writer.print("    \"layer_types\": [", .{});
    for (0..config_values.num_hidden_layers) |index| {
        const kind = model.layerType(@intCast(index));
        if (index > 0) {
            try writer.print(", ", .{});
        } else {
            assert(index == 0);
        }
        try writer.print("\"{s}\"", .{layerTypeName(kind)});
    }
    try writer.print("]\n", .{});
    try writer.print("  }},\n", .{});
    assert(std.meta.fields(Role).len == all_roles.len);
}

/// Name a layer kind carries in the manifest.
///
/// `config.json` calls the 12 layers that carry the QSA indexer
/// `full_attention`; the reference model and the issued layer taxonomy call
/// them `indexed_attention`, because every one of them overlays an indexer mask
/// on the causal mask. The manifest uses the taxonomy name and nothing else.
fn layerTypeName(kind: config.LayerType) []const u8 {
    assert(model_layer_type_names.len == 2);
    assert(kind == .linear_attention or kind == .full_attention);

    if (kind == .full_attention) {
        return "indexed_attention";
    } else {
        return "linear_attention";
    }
}

const model_layer_type_names = [_][]const u8{
    "linear_attention",
    "indexed_attention",
};

/// Writes one `{count, bytes}` object per role, in declaration order.
fn writeRoles(writer: anytype, manifest: *const Manifest) !void {
    assert(manifest.totals.len == all_roles.len);
    assert(manifest.layers.len > 0);

    try writer.print("  \"roles\": {{\n", .{});
    for (all_roles, 0..) |role, index| {
        const total = manifest.totals[@intFromEnum(role)];
        try writer.print("    \"{s}\": {{\"count\": {d}, \"bytes\": {d}}}", .{
            @tagName(role),
            total.count,
            total.bytes,
        });
        try writer.print(",\n", .{});
        assert(index < all_roles.len);
    }
    // Reporting-only aggregates. They overlap the roles above, so `total_bytes`
    // and every sum in this file exclude them: an aggregate that is silently
    // added into a total is how a wrong byte budget gets written.
    const key_and_value = manifest.total(.k_proj);
    const value = manifest.total(.v_proj);
    try writer.print(
        "    \"kv_proj_total\": {{\"count\": {d}, \"bytes\": {d}, " ++
            "\"derived\": true}},\n",
        .{
            key_and_value.count + value.count,
            key_and_value.bytes + value.bytes,
        },
    );
    const attention = manifest.selfAttentionTotal();
    try writer.print(
        "    \"self_attn\": {{\"count\": {d}, \"bytes\": {d}, " ++
            "\"derived\": true}}\n",
        .{ attention.count, attention.bytes },
    );
    try writer.print("  }},\n", .{});
    assert(all_roles.len > 0);
}

/// Writes the per-file totals, which must add up to the file sizes.
fn writeFiles(writer: anytype, loaded: *const Loaded) !void {
    assert(loaded.manifest.files.len == loaded.files.len);
    assert(loaded.files.len > 0);

    try writer.print("  \"files\": [\n", .{});
    for (loaded.manifest.files, 0..) |report, index| {
        try writer.print(
            "    {{\"name\": \"{s}\", \"tensors\": {d}, " ++
                "\"tensor_bytes\": {d}, " ++
                "\"header_bytes\": {d}, \"file_bytes\": {d}, " ++
                "\"data_present\": {s}}}",
            .{
                report.name,
                report.tensors,
                report.tensor_bytes,
                report.header_bytes,
                report.file_bytes,
                if (report.data_present) "true" else "false",
            },
        );
        if (index + 1 == loaded.files.len) {
            try writer.print("\n", .{});
        } else {
            try writer.print(",\n", .{});
        }
    }
    try writer.print("  ],\n", .{});
    assert(loaded.files.len > 0);
}

/// Writes the aggregate totals and the deferred sub-tree totals.
fn writeSummary(writer: anytype, loaded: *const Loaded) !void {
    assert(loaded.files.len > 0);
    assert(loaded.tensorBytesTotal() > 0);

    var header_bytes: u64 = 0;
    var file_bytes: u64 = 0;
    for (loaded.manifest.files) |report| {
        header_bytes += report.header_bytes;
        file_bytes += report.file_bytes;
    }
    try writer.print("  \"totals\": {{\n", .{});
    try writer.print(
        "    \"tensor_bytes\": {d},\n",
        .{loaded.tensorBytesTotal()},
    );
    try writer.print("    \"header_bytes\": {d},\n", .{header_bytes});
    try writer.print("    \"file_bytes\": {d},\n", .{file_bytes});
    try writer.print("    \"deferred_tensors\": {d},\n", .{
        loaded.manifest.deferred_tensors,
    });
    try writer.print(
        "    \"deferred_bytes\": {d}\n",
        .{loaded.manifest.deferred_bytes},
    );
    try writer.print("  }},\n", .{});
    // A shard that is still arriving holds fewer bytes than its header
    // declares, so only the header side is monotone here.
    assert(header_bytes > 0);
    assert(loaded.tensorBytesTotal() > 0);
}

/// The two numbers the gate compares: what we classified, and what the
/// checkpoint's own index claims.
fn writeReconciliation(writer: anytype, loaded: *const Loaded) !void {
    assert(loaded.manifest.index_total_size > 0);
    assert(loaded.tensorBytesTotal() > 0);

    const classified = loaded.tensorBytesTotal();
    try writer.print("  \"total_bytes\": {d},\n", .{classified});
    try writer.print("  \"index_total_size\": {d}\n", .{
        loaded.manifest.index_total_size,
    });
    assert(classified == loaded.manifest.index_total_size);
}
