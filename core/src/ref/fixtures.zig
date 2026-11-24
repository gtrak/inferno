//! Self-contained readers for the I-04 golden fixtures.
//!
//! Issue I-003 owns `core/src/weights/`; the reference implementation must not
//! depend on it, so this file carries its own tiny `config.json` parser and its
//! own safetensors reader. Nothing here is meant for the hot path: the whole
//! file (checkpoint or trace) is read once into an aligned buffer at init.
//!
//! Both readers take an allocator only at init; every later call works on the
//! buffers that init produced.

const std = @import("std");

const ops = @import("ops.zig");

const assert = std.debug.assert;

/// Longest tensor name in the tiny checkpoint is 78 bytes; leave headroom.
const name_bytes_max: u32 = 128;

/// The tiny checkpoint holds 152 tensors. The bound is what the table stores.
const entries_max: u32 = 512;

/// A safetensors file starts with a little-endian u64 header length.
const header_length_bytes: u64 = 8;

/// Rank of the shapes this reader accepts (expert tensors are rank 3).
const shape_rank_max: u32 = 4;

/// Longest layer list the fixed config arrays hold.
pub const layers_max: u32 = 64;

/// Element types that appear in the fixtures.
pub const DType = enum {
    f32,
    bf16,
    i64,
    i32,
    bool_,

    /// Maps the safetensors tag onto the enum.
    fn parse(tag: []const u8) !DType {
        assert(tag.len > 0);
        assert(tag.len <= 8);
        if (std.mem.eql(u8, tag, "F32")) {
            return .f32;
        } else if (std.mem.eql(u8, tag, "BF16")) {
            return .bf16;
        } else if (std.mem.eql(u8, tag, "I64")) {
            return .i64;
        } else if (std.mem.eql(u8, tag, "I32")) {
            return .i32;
        } else if (std.mem.eql(u8, tag, "BOOL")) {
            return .bool_;
        } else {
            std.debug.print("unsupported safetensors dtype {s}\n", .{tag});
            return error.UnsupportedDType;
        }
    }

    /// Bytes per element.
    fn size(self: DType) u32 {
        assert(@intFromEnum(self) <= @intFromEnum(DType.bool_));
        return switch (self) {
            .f32, .i32 => 4,
            .bf16 => 2,
            .i64 => 8,
            .bool_ => 1,
        };
    }
};

/// Maps a config activation name onto `ops.Activation`.
fn parse_activation(tag: []const u8) !ops.Activation {
    assert(tag.len > 0);
    assert(tag.len <= 16);

    if (std.mem.eql(u8, tag, "silu")) {
        return .silu;
    } else if (std.mem.eql(u8, tag, "sigmoid")) {
        return .sigmoid;
    } else {
        std.debug.print("unsupported activation {s}\n", .{tag});
        return error.UnsupportedActivation;
    }
}

/// What a decoder layer does in the attention slot.
pub const LayerType = enum {
    linear_attention,
    indexed_attention,

    fn parse(tag: []const u8) !LayerType {
        assert(tag.len > 0);
        assert(tag.len <= 32);
        if (std.mem.eql(u8, tag, "linear_attention")) {
            return .linear_attention;
        } else if (std.mem.eql(u8, tag, "indexed_attention")) {
            return .indexed_attention;
        } else {
            std.debug.print("unsupported layer type {s}\n", .{tag});
            return error.UnsupportedLayerType;
        }
    }
};

/// The subset of `Qwen4ExpTextConfig` the reference arithmetic needs.
///
/// Field names follow the config file; the derived sizes (`key_dim`, ...) are
/// functions below rather than stored fields so there is one spelling each.
pub const RefConfig = struct {
    hidden_size: u32 = 0,
    hc_count: u32 = 0,
    hc_lowrank: u32 = 0,
    num_hidden_layers: u32 = 0,
    layer_types: [layers_max]LayerType = undefined,
    vocab_size: u32 = 0,
    num_attention_heads: u32 = 0,
    num_key_value_heads: u32 = 0,
    head_dim: u32 = 0,
    attention_bias: bool = false,
    linear_num_key_heads: u32 = 0,
    linear_num_value_heads: u32 = 0,
    linear_key_head_dim: u32 = 0,
    linear_value_head_dim: u32 = 0,
    linear_conv_kernel_dim: u32 = 0,
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,
    norm_topk_prob: bool = false,
    rms_norm_eps: f32 = 0.0,
    output_gate_type: ops.Activation = .sigmoid,
    eos_token_id: i64 = 0,
    max_position_embeddings: u32 = 0,
    rope_theta: f32 = 0.0,
    partial_rotary_factor: f32 = 0.0,
    mrope_section: [3]u32 = .{ 0, 0, 0 },
    indexer_budget: u32 = 0,
    indexer_compress_ratio: u32 = 0,
    indexer_head_dim: u32 = 0,
    indexer_kv_heads: u32 = 0,
    indexer_n_heads: u32 = 0,
    ple_layer_ids: [layers_max]u32 = undefined,
    ple_layer_ids_count: u32 = 0,
    ple_embed_dim: u32 = 0,
    ple_conv_kernel_size: u32 = 0,
    ngram_size: u32 = 0,
    heads_per_ngram: u32 = 0,
    ngram_vocab_size_base: u32 = 0,
    make_ngram_vocab_size_divisible_by: u32 = 0,
    split_ngram_parts: u32 = 0,
    seed: u64 = 0,

    /// Parses the tiny `config.json`.
    pub fn parse(allocator: std.mem.Allocator, text: []const u8) !RefConfig {
        assert(text.len > 0);
        assert(text[0] == '{');

        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            allocator,
            text,
            .{},
        );
        defer parsed.deinit();
        const root = parsed.value.object;

        var config = RefConfig{};
        config.hidden_size = try config_u32(root, "hidden_size");
        config.hc_count = try config_u32(root, "hc_count");
        config.hc_lowrank = try config_u32(root, "hc_lowrank");
        config.num_hidden_layers = try config_u32(root, "num_hidden_layers");
        config.vocab_size = try config_u32(root, "vocab_size");
        config.num_attention_heads = try config_u32(
            root,
            "num_attention_heads",
        );
        config.num_key_value_heads = try config_u32(
            root,
            "num_key_value_heads",
        );
        config.head_dim = try config_u32(root, "head_dim");
        config.attention_bias =
            config_bool(root, "attention_bias") orelse false;
        config.linear_num_key_heads = try config_u32(
            root,
            "linear_num_key_heads",
        );
        config.linear_num_value_heads = try config_u32(
            root,
            "linear_num_value_heads",
        );
        config.linear_key_head_dim = try config_u32(
            root,
            "linear_key_head_dim",
        );
        config.linear_value_head_dim = try config_u32(
            root,
            "linear_value_head_dim",
        );
        config.linear_conv_kernel_dim = try config_u32(
            root,
            "linear_conv_kernel_dim",
        );
        config.num_experts = try config_u32(root, "num_experts");
        config.num_experts_per_tok = try config_u32(
            root,
            "num_experts_per_tok",
        );
        config.moe_intermediate_size = try config_u32(
            root,
            "moe_intermediate_size",
        );
        config.shared_expert_intermediate_size = try config_u32(
            root,
            "shared_expert_intermediate_size",
        );
        config.norm_topk_prob =
            config_bool(root, "norm_topk_prob") orelse false;
        config.output_gate_type = try parse_activation(
            root.get("output_gate_type").?.string,
        );
        config.rms_norm_eps = config_f32(root, "rms_norm_eps") orelse 1e-6;
        config.eos_token_id = config_i64(root, "eos_token_id") orelse 0;
        config.max_position_embeddings = try config_u32(
            root,
            "max_position_embeddings",
        );
        config.indexer_budget = try config_u32(root, "indexer_budget");
        config.indexer_compress_ratio = try config_u32(
            root,
            "indexer_compress_ratio",
        );
        config.indexer_head_dim = try config_u32(root, "indexer_head_dim");
        config.indexer_kv_heads = try config_u32(root, "indexer_kv_heads");
        config.indexer_n_heads = try config_u32(root, "indexer_n_heads");
        config.ple_embed_dim = try config_u32(root, "ple_embed_dim");
        config.ple_conv_kernel_size = try config_u32(
            root,
            "ple_conv_kernel_size",
        );
        config.ngram_size = try config_u32(root, "ngram_size");
        config.heads_per_ngram = try config_u32(root, "heads_per_ngram");
        config.ngram_vocab_size_base = try config_u32(
            root,
            "ngram_vocab_size_base",
        );
        config.make_ngram_vocab_size_divisible_by = try config_u32(
            root,
            "make_ngram_vocab_size_divisible_by",
        );
        config.split_ngram_parts = try config_u32(root, "split_ngram_parts");
        config.seed = @intCast(config_i64(root, "seed") orelse 0);
        try config_parse_rope(root.get("rope_parameters").?, &config);
        try config_parse_layer_types(root.get("layer_types").?, &config);
        try config_parse_ple_layer_ids(root.get("ple_layer_ids").?, &config);

        assert(config.hidden_size > 0);
        assert(config.hc_count > 0);
        return config;
    }

    /// Number of scalar values one token occupies in the hyper-connection
    // state.
    pub fn hyper_size(self: *const RefConfig) u32 {
        assert(self.hc_count > 0);
        assert(self.hidden_size > 0);
        return self.hc_count * self.hidden_size;
    }

    /// `linear_key_head_dim * linear_num_key_heads`.
    pub fn key_dim(self: *const RefConfig) u32 {
        assert(self.linear_num_key_heads > 0);
        assert(self.linear_key_head_dim > 0);
        return self.linear_key_head_dim * self.linear_num_key_heads;
    }

    /// `linear_value_head_dim * linear_num_value_heads`.
    pub fn value_dim(self: *const RefConfig) u32 {
        assert(self.linear_num_value_heads > 0);
        assert(self.linear_value_head_dim > 0);
        return self.linear_value_head_dim * self.linear_num_value_heads;
    }

    /// Channels the GDN depthwise conv mixes: q, k and v concatenated.
    pub fn conv_dim(self: *const RefConfig) u32 {
        assert(self.key_dim() > 0);
        assert(self.value_dim() > 0);
        return 2 * self.key_dim() + self.value_dim();
    }

    /// How many value heads each key head serves (repeat_interleave factor).
    pub fn value_heads_per_key_head(self: *const RefConfig) u32 {
        assert(self.linear_num_value_heads > 0);
        assert(self.linear_num_key_heads > 0);
        return @divExact(
            self.linear_num_value_heads,
            self.linear_num_key_heads,
        );
    }

    /// How many query heads each key/value head serves in the GQA attention.
    pub fn num_key_value_groups(self: *const RefConfig) u32 {
        assert(self.num_attention_heads > 0);
        assert(self.num_key_value_heads > 0);
        return @divExact(self.num_attention_heads, self.num_key_value_heads);
    }

    /// Dims rotated by RoPE: `int(head_dim * partial_rotary_factor)`.
    pub fn rotary_dim(self: *const RefConfig) u32 {
        assert(self.partial_rotary_factor > 0.0);
        assert(self.partial_rotary_factor <= 1.0);
        const scaled = @as(f32, @floatFromInt(self.head_dim)) *
            self.partial_rotary_factor;
        return @intFromFloat(scaled);
    }

    /// Number of rotary frequencies per position (`rotary_dim / 2`).
    pub fn rotary_freqs(self: *const RefConfig) u32 {
        assert(self.rotary_dim() > 0);
        assert(self.rotary_dim() % 2 == 0);
        return @divExact(self.rotary_dim(), 2);
    }

    /// Head count of the n-gram block: `(ngram_size - 1) * heads_per_ngram`.
    pub fn ngram_heads(self: *const RefConfig) u32 {
        assert(self.ngram_size >= 2);
        assert(self.heads_per_ngram >= 1);
        return (self.ngram_size - 1) * self.heads_per_ngram;
    }

    /// Embedding columns each n-gram head gathers.
    pub fn ngram_head_dim(self: *const RefConfig) u32 {
        assert(self.ple_embed_dim > 0);
        assert(self.ngram_heads() > 0);
        return @divExact(self.ple_embed_dim, self.ngram_heads());
    }

    /// Tokens of history the n-gram hash reads (`ngram_size - 1`).
    pub fn ngram_context_len(self: *const RefConfig) u32 {
        assert(self.ngram_size >= 2);
        assert(self.ngram_size <= 8);
        return self.ngram_size - 1;
    }

    /// Length of the PLE depthwise conv state: `(k - 1) * dilation`.
    pub fn ple_conv_state_len(self: *const RefConfig) u32 {
        assert(self.ple_conv_kernel_size >= 2);
        assert(self.ngram_size >= 2);
        return (self.ple_conv_kernel_size - 1) * self.ngram_size;
    }

    /// Index of the PLE layer for decoder layer `layer_index`, if it has one.
    ///
    /// `ple_layer_ids` holds 1-based text layer ids, so layer `L` is a PLE
    /// layer when `L + 1` appears in the list.
    pub fn ple_layer_index(
        self: *const RefConfig,
        layer_index: u32,
    ) ?u32 {
        assert(layer_index < self.num_hidden_layers);
        assert(self.ple_layer_ids_count <= layers_max);
        for (self.ple_layer_ids[0..self.ple_layer_ids_count], 0..) |id, i| {
            if (id == layer_index + 1) {
                return @intCast(i);
            } else {
                continue;
            }
        }
        return null;
    }

    /// Blocks the indexer may select (`indexer_budget / compress_ratio`).
    pub fn indexer_block_topk(self: *const RefConfig) u32 {
        assert(self.indexer_budget > 0);
        assert(self.indexer_compress_ratio > 0);
        return @divExact(self.indexer_budget, self.indexer_compress_ratio);
    }

    /// Slots of the `selected_token_indices` buffer per query.
    pub fn indexer_selected_slots(self: *const RefConfig) u32 {
        assert(self.indexer_block_topk() > 0);
        assert(self.indexer_compress_ratio > 0);
        return self.indexer_budget + self.indexer_compress_ratio - 1;
    }
};

/// Reads the nested `rope_parameters` object into `config`.
fn config_parse_rope(object: std.json.Value, config: *RefConfig) !void {
    assert(config.head_dim > 0);
    assert(object == .object);

    const root = object.object;
    config.rope_theta = config_f32(root, "rope_theta") orelse 10000.0;
    config.partial_rotary_factor = config_f32(
        root,
        "partial_rotary_factor",
    ) orelse 1.0;
    const section = root.get("mrope_section").?;
    assert(section.array.items.len == 3);
    for (section.array.items, 0..) |item, i| {
        config.mrope_section[i] = @intCast(item.integer);
    }
    assert(config.mrope_section[0] > 0);
}

/// Reads the `layer_types` string array into `config`.
fn config_parse_layer_types(object: std.json.Value, config: *RefConfig) !void {
    assert(object == .array);
    assert(object.array.items.len <= layers_max);

    const items = object.array.items;
    for (items, 0..) |item, i| {
        config.layer_types[i] = try LayerType.parse(item.string);
    }
    assert(items.len == config.num_hidden_layers);
    config.num_hidden_layers = @intCast(items.len);
}

/// Reads the `ple_layer_ids` integer array into `config`.
fn config_parse_ple_layer_ids(
    object: std.json.Value,
    config: *RefConfig,
) !void {
    assert(object == .array);
    assert(object.array.items.len <= layers_max);

    const items = object.array.items;
    for (items, 0..) |item, i| {
        config.ple_layer_ids[i] = @intCast(item.integer);
    }
    config.ple_layer_ids_count = @intCast(items.len);
    assert(config.ple_layer_ids_count == items.len);
}

/// Reads a required unsigned integer field.
fn config_u32(object: std.json.ObjectMap, key: []const u8) !u32 {
    assert(key.len > 0);
    assert(object.count() > 0);

    const value = object.get(key) orelse return error.MissingConfigField;
    assert(value == .integer);
    return @intCast(value.integer);
}

/// Reads an optional unsigned integer field.
fn config_i64(object: std.json.ObjectMap, key: []const u8) ?i64 {
    assert(key.len > 0);
    assert(object.count() > 0);

    const value = object.get(key) orelse return null;
    if (value == .integer) {
        return value.integer;
    } else {
        return null;
    }
}

/// Reads an optional float field, accepting an integer spelling too.
fn config_f32(object: std.json.ObjectMap, key: []const u8) ?f32 {
    assert(key.len > 0);
    assert(object.count() > 0);

    const value = object.get(key) orelse return null;
    if (value == .float) {
        return @floatCast(value.float);
    } else if (value == .integer) {
        return @floatFromInt(value.integer);
    } else {
        return null;
    }
}

/// Reads an optional `true`/`false` field.
fn config_bool(object: std.json.ObjectMap, key: []const u8) ?bool {
    assert(key.len > 0);
    assert(object.count() > 0);

    const value = object.get(key) orelse return null;
    if (value == .bool) {
        return value.bool;
    } else {
        return null;
    }
}

/// One tensor header entry: where its bytes are and what shape they form.
pub const Entry = struct {
    name: [name_bytes_max]u8 = [_]u8{0} ** name_bytes_max,
    name_length: u32 = 0,
    shape: [shape_rank_max]u32 = .{ 0, 0, 0, 0 },
    rank: u32 = 0,
    dtype: DType = .f32,
    offset: u64 = 0,
    element_count: u32 = 0,
};

/// A tensor as the reference uses it: f32 values and their shape.
pub const Tensor = struct {
    values: []const f32,
    shape: [shape_rank_max]u32,
    rank: u32,
};

/// A whole safetensors file, parsed into a lookup table of views.
pub const Tensors = struct {
    bytes: []align(8) u8,
    data_start: u64,
    entries: []Entry,
    entry_count: u32,

    /// Reads `path` and indexes every tensor header.
    pub fn open(allocator: std.mem.Allocator, path: []const u8) !Tensors {
        assert(path.len > 0);
        assert(path.len < 4096);

        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const length = try file.getEndPos();
        assert(length > header_length_bytes);

        const bytes = try allocator.alignedAlloc(u8, 8, @intCast(length));
        errdefer allocator.free(bytes);
        const read = try file.readAll(bytes);
        assert(read == length);

        const entries = try allocator.alloc(Entry, entries_max);
        errdefer allocator.free(entries);

        var tensors = Tensors{
            .bytes = bytes,
            .data_start = 0,
            .entries = entries,
            .entry_count = 0,
        };
        try tensors.parse_header(allocator);
        assert(tensors.entry_count > 0);
        return tensors;
    }

    /// Releases both allocations; the caller owns nothing else.
    pub fn close(self: *Tensors, allocator: std.mem.Allocator) void {
        assert(self.entry_count > 0);
        assert(self.entries.len == entries_max);

        allocator.free(self.entries);
        allocator.free(self.bytes);
    }

    /// Decodes the JSON header into `entries` and records `data_start`.
    fn parse_header(self: *Tensors, allocator: std.mem.Allocator) !void {
        assert(self.bytes.len > header_length_bytes);
        assert(self.entry_count == 0);

        const header_length = std.mem.readInt(u64, self.bytes[0..8], .little);
        assert(header_length > 2);
        const header_end = header_length_bytes + header_length;
        assert(header_end <= self.bytes.len);
        self.data_start = header_end;

        const text = self.bytes[header_length_bytes..@intCast(header_end)];
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            allocator,
            text,
            .{},
        );
        defer parsed.deinit();

        var iterator = parsed.value.object.iterator();
        while (iterator.next()) |item| {
            if (std.mem.eql(u8, item.key_ptr.*, "__metadata__")) {
                continue;
            } else {
                try self.parse_entry(item.key_ptr.*, &item.value_ptr.*);
            }
        }
        assert(self.entry_count <= entries_max);
    }

    /// Fills one `Entry` from a `name -> {dtype, shape, data_offsets}` object.
    fn parse_entry(
        self: *Tensors,
        name: []const u8,
        value: *const std.json.Value,
    ) !void {
        assert(name.len > 0);
        assert(self.entry_count < entries_max);

        const object = value.object;
        const slot = &self.entries[self.entry_count];
        if (name.len > name_bytes_max) {
            return error.TensorNameTooLong;
        } else {
            @memcpy(slot.name[0..name.len], name);
        }
        slot.name_length = @intCast(name.len);
        slot.dtype = try DType.parse(object.get("dtype").?.string);

        const shape = object.get("shape").?.array.items;
        if (shape.len > shape_rank_max) {
            return error.TensorRankTooLarge;
        } else {
            slot.rank = @intCast(shape.len);
        }
        var element_count: u32 = 1;
        for (shape, 0..) |dim, i| {
            slot.shape[i] = @intCast(dim.integer);
            element_count *= slot.shape[i];
        }
        slot.element_count = element_count;

        const offsets = object.get("data_offsets").?.array.items;
        assert(offsets.len == 2);
        slot.offset = @intCast(offsets[0].integer);
        const expected: u64 = @as(u64, element_count) * slot.dtype.size();
        assert(offsets[1].integer - offsets[0].integer == expected);
        self.entry_count += 1;
    }

    /// Looks up an entry by name, or null when the file has no such tensor.
    pub fn find(self: *const Tensors, name: []const u8) ?*const Entry {
        assert(name.len > 0);
        assert(self.entry_count > 0);

        for (self.entries[0..self.entry_count]) |*entry| {
            if (std.mem.eql(u8, entry.name[0..entry.name_length], name)) {
                return entry;
            } else {
                continue;
            }
        }
        return null;
    }

    /// Views the named tensor as f32, converting bf16 on the way out.
    ///
    /// bf16 tensors are rejected here: the fixtures are f32 and the one bf16
    /// path (the real `lm_head`) belongs to the weights layer.
    pub fn tensor(self: *const Tensors, name: []const u8) !Tensor {
        assert(name.len > 0);
        assert(self.data_start > 0);

        const entry = self.find(name) orelse {
            std.debug.print("missing tensor {s}\n", .{name});
            return error.MissingTensor;
        };
        if (entry.dtype != .f32) {
            std.debug.print("tensor {s} is not f32\n", .{name});
            return error.TensorDType;
        } else {
            assert(entry.dtype == .f32);
        }
        const address = @intFromPtr(self.bytes.ptr) +
            self.data_start + entry.offset;
        if (address % @alignOf(f32) != 0) {
            return error.TensorMisaligned;
        } else {
            assert(address % @alignOf(f32) == 0);
        }
        const pointer: [*]const f32 = @ptrFromInt(address);
        assert(entry.element_count == entry.shape[0] * product(entry));
        return .{
            .values = pointer[0..entry.element_count],
            .shape = entry.shape,
            .rank = entry.rank,
        };
    }

    /// Views the named boolean tensor as raw bytes (0 or 1).
    pub fn tensor_bool(self: *const Tensors, name: []const u8) ![]const u8 {
        assert(name.len > 0);
        assert(self.data_start > 0);

        const entry = self.find(name) orelse {
            std.debug.print("missing tensor {s}\n", .{name});
            return error.MissingTensor;
        };
        if (entry.dtype != .bool_) {
            std.debug.print("tensor {s} is not bool\n", .{name});
            return error.TensorDType;
        } else {
            assert(entry.dtype == .bool_);
        }
        const address = @intFromPtr(self.bytes.ptr) +
            self.data_start + entry.offset;
        const pointer: [*]const u8 = @ptrFromInt(address);
        return pointer[0..entry.element_count];
    }

    /// Views the named int32 tensor.
    pub fn tensor_i32(self: *const Tensors, name: []const u8) ![]const i32 {
        assert(name.len > 0);
        assert(self.data_start > 0);

        const entry = self.find(name) orelse {
            std.debug.print("missing tensor {s}\n", .{name});
            return error.MissingTensor;
        };
        if (entry.dtype != .i32) {
            std.debug.print("tensor {s} is not i32\n", .{name});
            return error.TensorDType;
        } else {
            assert(entry.dtype == .i32);
        }
        const address = @intFromPtr(self.bytes.ptr) +
            self.data_start + entry.offset;
        if (address % @alignOf(i32) != 0) {
            return error.TensorMisaligned;
        } else {
            assert(address % @alignOf(i32) == 0);
        }
        const pointer: [*]const i32 = @ptrFromInt(address);
        return pointer[0..entry.element_count];
    }

    /// Views the named int64 tensor.
    pub fn tensor_i64(self: *const Tensors, name: []const u8) ![]const i64 {
        assert(name.len > 0);
        assert(self.data_start > 0);

        const entry = self.find(name) orelse {
            std.debug.print("missing tensor {s}\n", .{name});
            return error.MissingTensor;
        };
        if (entry.dtype != .i64) {
            std.debug.print("tensor {s} is not i64\n", .{name});
            return error.TensorDType;
        } else {
            assert(entry.dtype == .i64);
        }
        const address = @intFromPtr(self.bytes.ptr) +
            self.data_start + entry.offset;
        if (address % @alignOf(i64) != 0) {
            return error.TensorMisaligned;
        } else {
            assert(address % @alignOf(i64) == 0);
        }
        const pointer: [*]const i64 = @ptrFromInt(address);
        return pointer[0..entry.element_count];
    }
};

/// Product of the shape dimensions from index 1 on (used to re-derive count).
fn product(entry: *const Entry) u32 {
    assert(entry.rank > 0);
    assert(entry.rank <= shape_rank_max);

    var total: u32 = 1;
    for (entry.shape[1..entry.rank]) |dim| {
        total *= dim;
    }
    assert(total >= 1);
    return total;
}

test "the tiny config parses into the documented shape" {
    const allocator = std.testing.allocator;
    const config = try load_config(allocator);
    try std.testing.expectEqual(@as(u32, 128), config.hidden_size);
    try std.testing.expectEqual(@as(u32, 2), config.hc_count);
    try std.testing.expectEqual(@as(u32, 256), config.hyper_size());
    try std.testing.expectEqual(@as(u32, 4), config.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 160), config.conv_dim());
    try std.testing.expectEqual(@as(u32, 3), config.value_heads_per_key_head());
    try std.testing.expectEqual(@as(u32, 4), config.num_key_value_groups());
}

test "the tiny config derives the rope, ple and indexer sizes" {
    const allocator = std.testing.allocator;
    const config = try load_config(allocator);
    try std.testing.expectEqual(@as(u32, 8), config.rotary_dim());
    try std.testing.expectEqual(@as(u32, 4), config.rotary_freqs());
    try std.testing.expectEqual(@as(u32, 2), config.mrope_section[0]);
    try std.testing.expectEqual(@as(u32, 1), config.mrope_section[1]);
    try std.testing.expectEqual(@as(u32, 1), config.mrope_section[2]);
    try std.testing.expectEqual(@as(u32, 4), config.ngram_heads());
    try std.testing.expectEqual(@as(u32, 8), config.ngram_head_dim());
    try std.testing.expectEqual(@as(u32, 9), config.ple_conv_state_len());
    try std.testing.expectEqual(@as(u32, 4), config.indexer_block_topk());
    try std.testing.expectEqual(@as(u32, 9), config.indexer_selected_slots());
}

test "only text layer 1 carries a PLE block" {
    const allocator = std.testing.allocator;
    const config = try load_config(allocator);
    try std.testing.expectEqual(@as(?u32, null), config.ple_layer_index(0));
    try std.testing.expectEqual(@as(?u32, 0), config.ple_layer_index(1));
    try std.testing.expectEqual(@as(?u32, null), config.ple_layer_index(2));
    try std.testing.expectEqual(
        LayerType.indexed_attention,
        config.layer_types[2],
    );
}

test "the checkpoint reader finds the expert and n-gram tensors" {
    const allocator = std.testing.allocator;
    var tensors = try open_checkpoint(allocator);
    defer tensors.close(allocator);

    try std.testing.expectEqual(@as(u32, 152), tensors.entry_count);
    const embedding = try tensors.tensor(
        "model.language_model.embed_tokens.weight",
    );
    try std.testing.expectEqual(@as(u32, 256), embedding.shape[0]);
    try std.testing.expectEqual(@as(u32, 128), embedding.shape[1]);

    const up = try tensors.tensor(
        "model.language_model.layers.0.mlp.experts.3.up_proj.weight",
    );
    try std.testing.expectEqual(@as(u32, 32), up.shape[0]);
    try std.testing.expectEqual(@as(u32, 128), up.shape[1]);

    const shard = try tensors.tensor(
        "model.language_model.layers.1.ple.ple_embedding" ++
            ".ngram_embedding.shard_1.weight",
    );
    try std.testing.expectEqual(@as(u32, 2032), shard.shape[0]);
    const multipliers = try tensors.tensor_i64(
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
    );
    try std.testing.expectEqual(@as(usize, 3), multipliers.len);
}

/// Loads the tiny config from the fixture directory the build passed in.
pub fn load_config(allocator: std.mem.Allocator) !RefConfig {
    assert(fixtures_directory.len > 0);
    assert(std.mem.endsWith(u8, fixtures_directory, "tiny"));

    const path = try std.fmt.allocPrint(
        allocator,
        "{s}/config.json",
        .{fixtures_directory},
    );
    defer allocator.free(path);
    const text = try read_fixture_text(allocator, path);
    defer allocator.free(text);
    assert(text.len > 0);
    return RefConfig.parse(allocator, text);
}

/// Opens the tiny checkpoint-layout safetensors file.
pub fn open_checkpoint(allocator: std.mem.Allocator) !Tensors {
    assert(fixtures_directory.len > 0);
    assert(std.mem.endsWith(u8, fixtures_directory, "tiny"));

    const path = try std.fmt.allocPrint(
        allocator,
        "{s}/checkpoint.safetensors",
        .{fixtures_directory},
    );
    defer allocator.free(path);
    assert(path.len > 0);
    return Tensors.open(allocator, path);
}

/// Opens the golden trace file.
pub fn open_trace(allocator: std.mem.Allocator) !Tensors {
    assert(fixtures_directory.len > 0);
    assert(std.mem.endsWith(u8, fixtures_directory, "tiny"));

    const path = try std.fmt.allocPrint(
        allocator,
        "{s}/trace.safetensors",
        .{fixtures_directory},
    );
    defer allocator.free(path);
    assert(path.len > 0);
    return Tensors.open(allocator, path);
}

/// Reads a whole fixture file as text.
fn read_fixture_text(
    allocator: std.mem.Allocator,
    path: []const u8,
) ![]u8 {
    assert(path.len > 0);
    assert(path.len < 4096);

    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const length = try file.getEndPos();
    assert(length > 0);
    const buffer = try allocator.alloc(u8, @intCast(length));
    errdefer allocator.free(buffer);
    const read = try file.readAll(buffer);
    assert(read == length);
    return buffer;
}

/// Directory of the golden fixtures, injected by `core/build.zig`.
const fixtures_directory: []const u8 = @import("fixtures_dir").path;
