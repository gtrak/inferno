//! The checkpoint's `config.json`, parsed into explicit types.
//!
//! `TextConfig` mirrors every key of `text_config` so that a missing or renamed
//! key is a parse error instead of a silent default: the engine reads these
//! numbers to size every buffer, and a default would be a wrong buffer.
//! `vision_config` and `quantization_config` are parsed for the same reason but
//! are unused until the vision tower lands.
//!
//! `TextModel` is the validated form: it recomputes the dimensions the weight
//! manifest and the kernels index with, and asserts the relations that this
//! checkpoint satisfies (see the assertions in `validateAttention`,
//! `validateLinearAttention` and `validatePle`).

const std = @import("std");

const assert = std.debug.assert;

/// The checkpoint's 8-bit floats are the OCP E4M3 (FN) variant.
pub const fp8_dtype_name = "float8_e4m3fn";

/// Kind of attention a text layer uses, straight from `layer_types`.
pub const LayerType = enum {
    linear_attention,
    full_attention,
};

/// `rope_parameters`: partial rotary width and the mRoPE section split.
pub const RopeParameters = struct {
    mrope_interleaved: bool,
    mrope_section: []const u32,
    partial_rotary_factor: f32,
    rope_theta: f64,
    rope_type: []const u8,
};

/// `text_config.mtp`: one speculative-decoding layer, deferred for now.
pub const MtpConfig = struct {
    hybrid: bool,
    layer_types: []const LayerType,
    mtp_use_hidden_state_from_layer: ?u32,
    num_hidden_layers: u32,
    rope_theta: f64,
};

/// Every key of `text_config`.
pub const TextConfig = struct {
    attention_bias: bool,
    attention_dropout: f32,
    bos_token_id: u32,
    dtype: []const u8,
    eos_token_id: u32,
    full_attention_interval: u32,
    hc_count: u32,
    hc_lowrank: u32,
    head_dim: u32,
    heads_per_ngram: u32,
    hidden_act: []const u8,
    hidden_size: u32,
    indexer_budget: u32,
    indexer_compress_ratio: u32,
    indexer_head_dim: u32,
    indexer_kv_heads: u32,
    indexer_n_heads: u32,
    initializer_range: f32,
    layer_types: []const LayerType,
    linear_conv_kernel_dim: u32,
    linear_key_head_dim: u32,
    linear_num_key_heads: u32,
    linear_num_value_heads: u32,
    linear_value_head_dim: u32,
    make_ngram_vocab_size_divisible_by: u32,
    mamba_ssm_dtype: []const u8,
    max_position_embeddings: u32,
    model_type: []const u8,
    moe_intermediate_size: u32,
    mtp: MtpConfig,
    mtp_num_hidden_layers: u32,
    mtp_use_dedicated_embeddings: bool,
    ngram_size: u32,
    ngram_vocab_size_base: u32,
    norm_topk_prob: bool,
    num_attention_heads: u32,
    num_experts: u32,
    num_experts_per_tok: u32,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    number_of_conv_states: u32,
    output_gate_type: []const u8,
    output_router_logits: bool,
    pad_token_id: ?u32,
    partial_rotary_factor: f32,
    ple_conv_kernel_size: u32,
    ple_embed_dim: u32,
    ple_layer_ids: []const u32,
    rms_norm_eps: f32,
    rope_parameters: RopeParameters,
    router_aux_loss_coef: f32,
    seed: u32,
    shared_expert_intermediate_size: u32,
    split_ngram_parts: u32,
    tie_word_embeddings: bool,
    use_cache: bool,
    vocab_size: u32,
};

/// Every key of `vision_config`; deferred, so nothing validates it yet.
pub const VisionConfig = struct {
    deepstack_visual_indexes: []const u32,
    depth: u32,
    dtype: []const u8,
    hidden_act: []const u8,
    hidden_size: u32,
    in_channels: u32,
    initializer_range: f32,
    intermediate_size: u32,
    model_type: []const u8,
    num_heads: u32,
    num_position_embeddings: u32,
    out_hidden_size: u32,
    patch_size: u32,
    spatial_merge_size: u32,
    temporal_patch_size: u32,
};

/// Every top-level key of `config.json`.
pub const Config = struct {
    architectures: []const []const u8,
    dtype: []const u8,
    image_token_id: u32,
    language_model_only: bool,
    model_type: []const u8,
    /// Recognised and unused: the quantisation layout is read from tensor names
    /// and dtypes, not from this section.
    quantization_config: std.json.Value,
    text_config: TextConfig,
    tie_word_embeddings: bool,
    transformers_version: []const u8,
    video_token_id: u32,
    vision_config: VisionConfig,
    vision_end_token_id: u32,
    vision_start_token_id: u32,
};

/// Validated text configuration plus the dimensions derived from it.
pub const TextModel = struct {
    config: TextConfig,
    /// `hc_count * hidden_size`: width of every hyper-connection stream.
    hc_hidden_size: u32,
    /// Rows of `q_proj`: a query block and a gate block per head.
    q_proj_rows: u32,
    /// Rows of `k_proj` and of `v_proj`.
    kv_proj_rows: u32,
    /// Columns of `o_proj`: the heads' queries without their gates.
    o_proj_cols: u32,
    /// Rows of `in_proj_qkv`: q and k, then the value heads.
    linear_qkv_rows: u32,
    /// Rows of `in_proj_z`, equal to the value rows of `in_proj_qkv`.
    linear_z_rows: u32,
    /// Rows of `indexer.index_qk_proj`: indexer queries then indexer keys.
    indexer_qk_rows: u32,
    /// Number of n-gram heads feeding the PLE embedding.
    ngram_heads: u32,
    /// Columns of every n-gram table shard, `ple_embed_dim / ngram_heads`.
    ngram_embed_dim: u32,
    /// Zero-based index of the single PLE layer.
    ple_layer_index: u32,
    full_attention_layers: u32,
    linear_attention_layers: u32,

    pub fn init(config: TextConfig) !TextModel {
        assert(config.num_hidden_layers > 0);
        assert(config.hidden_size > 0);

        const model = TextModel{
            .config = config,
            .hc_hidden_size = try hyperConnectionWidth(&config),
            .q_proj_rows = try queryProjectionRows(&config),
            .kv_proj_rows = config.num_key_value_heads * config.head_dim,
            .o_proj_cols = config.num_attention_heads * config.head_dim,
            .linear_qkv_rows = try linearAttentionRows(&config),
            .linear_z_rows = config.linear_num_value_heads * config.linear_value_head_dim,
            .indexer_qk_rows = try indexerRows(&config),
            .ngram_heads = try ngramHeadCount(&config),
            .ngram_embed_dim = try ngramEmbedDim(&config),
            .ple_layer_index = try pleLayerIndex(&config),
            .full_attention_layers = try countLayerTypes(&config, .full_attention),
            .linear_attention_layers = try countLayerTypes(&config, .linear_attention),
        };
        try validateMisc(&model);
        assert(model.full_attention_layers + model.linear_attention_layers ==
            config.num_hidden_layers);
        return model;
    }

    /// Whether text layer `layer_index` is linear attention or full attention.
    pub fn layerType(self: *const TextModel, layer_index: u32) LayerType {
        assert(layer_index < self.config.num_hidden_layers);
        assert(self.config.layer_types.len == self.config.num_hidden_layers);

        return self.config.layer_types[layer_index];
    }
};

/// Parses `bytes` as a checkpoint `config.json`.
pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !Config {
    assert(bytes.len > 0);
    assert(bytes.len <= config_bytes_max);

    const config = std.json.parseFromSliceLeaky(Config, allocator, bytes, .{}) catch |err| {
        std.debug.print("config: cannot parse: {s}\n", .{@errorName(err)});
        return error.malformed_config;
    };
    assert(config.text_config.num_hidden_layers > 0);
    return config;
}

/// Reads and parses `path`.
pub fn load(allocator: std.mem.Allocator, path: []const u8) !Config {
    assert(path.len > 0);
    assert(config_bytes_max > 0);

    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, config_bytes_max);
    assert(bytes.len <= config_bytes_max);
    return parse(allocator, bytes);
}

/// The real file is 30 KiB; this bound only has to be generous and finite.
pub const config_bytes_max: u32 = 1 << 20;

/// `hc_count * hidden_size`, which is also the linear attention q/k/v width.
fn hyperConnectionWidth(config: *const TextConfig) !u32 {
    assert(config.hc_count > 0);
    assert(config.hidden_size > 0);

    const width = std.math.mul(u32, config.hc_count, config.hidden_size) catch {
        return error.invalid_config;
    };
    const key_width = try std.math.mul(
        u32,
        config.linear_num_key_heads * config.linear_key_head_dim,
        2,
    );
    const value_width = config.linear_num_value_heads * config.linear_value_head_dim;
    // This is a real invariant of the checkpoint: the residual stream is 4 x
    // 2560 wide and the GDN input projection produces exactly that many rows,
    // so the two are the same buffer layout. If this ever fails, the layer
    // widths changed and every downstream shape assumption is wrong.
    if (width != key_width + value_width) {
        std.debug.print("config: hc width {d} != linear attention {d}\n", .{
            width,
            key_width + value_width,
        });
        return error.invalid_config;
    } else {
        assert(width == key_width + value_width);
    }
    assert(width > 0);
    return width;
}

/// Rows of `q_proj`, which emits a query and a gate per head.
fn queryProjectionRows(config: *const TextConfig) !u32 {
    assert(config.num_attention_heads > 0);
    assert(config.head_dim > 0);

    const rows = std.math.mul(
        u32,
        config.num_attention_heads * config.head_dim,
        2,
    ) catch return error.invalid_config;
    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        return error.invalid_config;
    } else {
        assert(config.num_attention_heads % config.num_key_value_heads == 0);
    }
    assert(rows % 2 == 0);
    return rows;
}

/// Rows of `in_proj_qkv`: a query and a key per key head, values per value head.
fn linearAttentionRows(config: *const TextConfig) !u32 {
    assert(config.linear_num_key_heads > 0);
    assert(config.linear_num_value_heads > 0);

    if (config.linear_num_value_heads % config.linear_num_key_heads != 0) {
        return error.invalid_config;
    } else {
        assert(config.linear_num_value_heads % config.linear_num_key_heads == 0);
    }
    const per_key_head = config.linear_key_head_dim * 2;
    const key_rows = config.linear_num_key_heads * per_key_head;
    const value_rows = config.linear_num_value_heads * config.linear_value_head_dim;
    const rows = std.math.add(u32, key_rows, value_rows) catch {
        return error.invalid_config;
    };
    assert(rows > value_rows);
    return rows;
}

/// Rows of `indexer.index_qk_proj`: the indexer queries plus one key head.
fn indexerRows(config: *const TextConfig) !u32 {
    assert(config.indexer_n_heads > 0);
    assert(config.indexer_head_dim > 0);

    const heads = std.math.add(
        u32,
        config.indexer_n_heads,
        config.indexer_kv_heads,
    ) catch return error.invalid_config;
    const rows = std.math.mul(u32, heads, config.indexer_head_dim) catch {
        return error.invalid_config;
    };
    assert(rows > config.indexer_n_heads * config.indexer_head_dim);
    return rows;
}

/// Number of n-gram heads: one per (order, head) pair above the unigram.
fn ngramHeadCount(config: *const TextConfig) !u32 {
    assert(config.ngram_size >= 2);
    assert(config.heads_per_ngram > 0);

    const heads = std.math.mul(
        u32,
        config.ngram_size - 1,
        config.heads_per_ngram,
    ) catch return error.invalid_config;
    assert(heads > 0);
    return heads;
}

/// Columns of an n-gram table shard.
fn ngramEmbedDim(config: *const TextConfig) !u32 {
    assert(config.ple_embed_dim > 0);
    assert(config.split_ngram_parts > 0);

    const heads = try ngramHeadCount(config);
    if (config.ple_embed_dim % heads != 0) {
        return error.invalid_config;
    } else {
        assert(config.ple_embed_dim % heads == 0);
    }
    const width = config.ple_embed_dim / heads;
    assert(width > 0);
    return width;
}

/// Zero-based PLE layer index; `ple_layer_ids` are 1-based layer numbers.
fn pleLayerIndex(config: *const TextConfig) !u32 {
    assert(config.ple_layer_ids.len == 1);
    assert(config.num_hidden_layers > 0);

    const one_based = config.ple_layer_ids[0];
    if (one_based == 0) {
        return error.invalid_config;
    } else {
        assert(one_based > 0);
    }
    const index = one_based - 1;
    if (index >= config.num_hidden_layers) {
        return error.invalid_config;
    } else {
        assert(index < config.num_hidden_layers);
    }
    return index;
}

/// Counts one kind of layer, cross-checked against `full_attention_interval`.
fn countLayerTypes(config: *const TextConfig, kind: LayerType) !u32 {
    assert(config.layer_types.len > 0);
    assert(config.full_attention_interval > 0);

    if (config.layer_types.len != config.num_hidden_layers) {
        return error.invalid_config;
    } else {
        assert(config.layer_types.len == config.num_hidden_layers);
    }
    var count: u32 = 0;
    for (config.layer_types, 0..) |entry, position| {
        const is_last_of_interval =
            (position + 1) % config.full_attention_interval == 0;
        // Every interval ends with a full attention layer and nothing else is
        // one, so the pattern is derivable from the interval alone.
        if (is_last_of_interval != (entry == .full_attention)) {
            std.debug.print("config: layer {d} is {s}\n", .{
                position,
                @tagName(entry),
            });
            return error.invalid_config;
        } else {
            assert(is_last_of_interval == (entry == .full_attention));
        }
        if (entry == kind) {
            count += 1;
        } else {
            assert(entry != kind);
        }
    }
    assert(count <= config.num_hidden_layers);
    return count;
}

/// Checks the scalar relations that do not size a single tensor.
fn validateMisc(model: *const TextModel) !void {
    assert(model.config.rms_norm_eps > 0);
    assert(model.config.vocab_size > 0);

    const config = &model.config;
    const head_dim: f32 = @floatFromInt(config.head_dim);
    const rope_halves = head_dim * config.partial_rotary_factor / 2.0;
    const section_total = config.rope_parameters.mrope_section[0] +
        config.rope_parameters.mrope_section[1] +
        config.rope_parameters.mrope_section[2];
    // The rotary half dimension is split between the three mRoPE sections, so
    // their sum *is* the rotary width; 256 * 0.25 / 2 = 32 = 11 + 11 + 10.
    if (rope_halves != @as(f32, @floatFromInt(section_total))) {
        std.debug.print("config: rotary halves {d} != mRoPE sections {d}\n", .{
            rope_halves,
            section_total,
        });
        return error.invalid_config;
    } else {
        assert(rope_halves == @as(f32, @floatFromInt(section_total)));
    }
    if (config.partial_rotary_factor != config.rope_parameters.partial_rotary_factor) {
        return error.invalid_config;
    } else {
        assert(config.partial_rotary_factor ==
            config.rope_parameters.partial_rotary_factor);
    }
    // Three conv states per layer: the GDN causal conv, the PLE short conv and
    // the PLE n-gram token history (see the reference implementation).
    if (config.number_of_conv_states != 3) {
        return error.invalid_config;
    } else {
        assert(config.number_of_conv_states == 3);
    }
    if (config.num_experts_per_tok > config.num_experts) {
        return error.invalid_config;
    } else {
        assert(config.num_experts_per_tok <= config.num_experts);
    }
    assert(config.head_dim % 2 == 0);
}
/// A copy of the checkpoint's config.json, with the unused quantisation
/// section trimmed: the fixture exercises the parse and the invariants,
/// and `zig build manifest` exercises the real file.
const test_config_json =
    \\{
    \\  "architectures": ["Qwen4ExpForConditionalGeneration"],
    \\  "dtype": "bfloat16",
    \\  "image_token_id": 248056,
    \\  "language_model_only": false,
    \\  "model_type": "qwen4_exp",
    \\  "quantization_config": {"config_groups": {}, "ignore": []},
    \\  "tie_word_embeddings": false,
    \\  "transformers_version": "5.16.0.dev0",
    \\  "video_token_id": 248057,
    \\  "vision_end_token_id": 248054,
    \\  "vision_start_token_id": 248053,
    \\  "vision_config": {
    \\    "deepstack_visual_indexes": [],
    \\    "depth": 27,
    \\    "dtype": "bfloat16",
    \\    "hidden_act": "gelu_pytorch_tanh",
    \\    "hidden_size": 1152,
    \\    "in_channels": 3,
    \\    "initializer_range": 0.02,
    \\    "intermediate_size": 4304,
    \\    "model_type": "qwen4_exp_vision",
    \\    "num_heads": 16,
    \\    "num_position_embeddings": 2304,
    \\    "out_hidden_size": 2560,
    \\    "patch_size": 16,
    \\    "spatial_merge_size": 2,
    \\    "temporal_patch_size": 2
    \\  },
    \\  "text_config": {
    \\    "attention_bias": false,
    \\    "attention_dropout": 0.0,
    \\    "bos_token_id": 248044,
    \\    "dtype": "bfloat16",
    \\    "eos_token_id": 248044,
    \\    "full_attention_interval": 4,
    \\    "hc_count": 4,
    \\    "hc_lowrank": 320,
    \\    "head_dim": 256,
    \\    "heads_per_ngram": 8,
    \\    "hidden_act": "silu",
    \\    "hidden_size": 2560,
    \\    "indexer_budget": 2048,
    \\    "indexer_compress_ratio": 4,
    \\    "indexer_head_dim": 128,
    \\    "indexer_kv_heads": 1,
    \\    "indexer_n_heads": 4,
    \\    "initializer_range": 0.02,
    \\    "linear_conv_kernel_dim": 4,
    \\    "linear_key_head_dim": 128,
    \\    "linear_num_key_heads": 16,
    \\    "linear_num_value_heads": 48,
    \\    "linear_value_head_dim": 128,
    \\    "make_ngram_vocab_size_divisible_by": 128,
    \\    "mamba_ssm_dtype": "float32",
    \\    "max_position_embeddings": 262144,
    \\    "model_type": "qwen4_exp_text",
    \\    "moe_intermediate_size": 640,
    \\    "mtp_num_hidden_layers": 1,
    \\    "mtp_use_dedicated_embeddings": false,
    \\    "ngram_size": 3,
    \\    "ngram_vocab_size_base": 20000000,
    \\    "norm_topk_prob": true,
    \\    "num_attention_heads": 24,
    \\    "num_experts": 512,
    \\    "num_experts_per_tok": 10,
    \\    "num_hidden_layers": 48,
    \\    "num_key_value_heads": 2,
    \\    "number_of_conv_states": 3,
    \\    "output_gate_type": "sigmoid",
    \\    "output_router_logits": false,
    \\    "pad_token_id": null,
    \\    "partial_rotary_factor": 0.25,
    \\    "ple_conv_kernel_size": 4,
    \\    "ple_embed_dim": 2560,
    \\    "ple_layer_ids": [2],
    \\    "rms_norm_eps": 1e-06,
    \\    "router_aux_loss_coef": 0.001,
    \\    "seed": 1234,
    \\    "shared_expert_intermediate_size": 640,
    \\    "split_ngram_parts": 128,
    \\    "tie_word_embeddings": false,
    \\    "use_cache": true,
    \\    "vocab_size": 248320,
    \\    "rope_parameters": {
    \\      "mrope_interleaved": true,
    \\      "mrope_section": [11, 11, 10],
    \\      "partial_rotary_factor": 0.25,
    \\      "rope_theta": 10000000,
    \\      "rope_type": "default"
    \\    },
    \\    "mtp": {
    \\      "hybrid": true,
    \\      "layer_types": ["full_attention"],
    \\      "mtp_use_hidden_state_from_layer": null,
    \\      "num_hidden_layers": 1,
    \\      "rope_theta": 10000000
    \\    },
    \\    "layer_types": [
    \\      "linear_attention", "linear_attention", "linear_attention",
    \\      "full_attention", "linear_attention", "linear_attention",
    \\      "linear_attention", "full_attention", "linear_attention",
    \\      "linear_attention", "linear_attention", "full_attention",
    \\      "linear_attention", "linear_attention", "linear_attention",
    \\      "full_attention", "linear_attention", "linear_attention",
    \\      "linear_attention", "full_attention", "linear_attention",
    \\      "linear_attention", "linear_attention", "full_attention",
    \\      "linear_attention", "linear_attention", "linear_attention",
    \\      "full_attention", "linear_attention", "linear_attention",
    \\      "linear_attention", "full_attention", "linear_attention",
    \\      "linear_attention", "linear_attention", "full_attention",
    \\      "linear_attention", "linear_attention", "linear_attention",
    \\      "full_attention", "linear_attention", "linear_attention",
    \\      "linear_attention", "full_attention", "linear_attention",
    \\      "linear_attention", "linear_attention", "full_attention"
    \\    ]
    \\  }
    \\}
;

test "the checkpoint config yields the documented text dimensions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = try parse(allocator, test_config_json);
    const model = try TextModel.init(config.text_config);

    try std.testing.expectEqual(@as(u32, 48), model.config.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 2560), model.config.hidden_size);
    try std.testing.expectEqual(@as(u32, 248320), model.config.vocab_size);
    try std.testing.expectEqual(@as(u32, 10240), model.hc_hidden_size);
    try std.testing.expectEqual(@as(u32, 12288), model.q_proj_rows);
    try std.testing.expectEqual(@as(u32, 512), model.kv_proj_rows);
    try std.testing.expectEqual(@as(u32, 6144), model.o_proj_cols);
    try std.testing.expectEqual(@as(u32, 10240), model.linear_qkv_rows);
    try std.testing.expectEqual(@as(u32, 6144), model.linear_z_rows);
    try std.testing.expectEqual(@as(u32, 640), model.indexer_qk_rows);
    try std.testing.expectEqual(@as(u32, 16), model.ngram_heads);
    try std.testing.expectEqual(@as(u32, 160), model.ngram_embed_dim);
    try std.testing.expectEqual(@as(u32, 1), model.ple_layer_index);
    try std.testing.expectEqual(@as(u32, 12), model.full_attention_layers);
    try std.testing.expectEqual(@as(u32, 36), model.linear_attention_layers);
}

test "the layer kind follows the full attention interval" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = try parse(allocator, test_config_json);
    const model = try TextModel.init(config.text_config);

    try std.testing.expectEqual(LayerType.linear_attention, model.layerType(1));
    try std.testing.expectEqual(LayerType.full_attention, model.layerType(3));
    try std.testing.expectEqual(LayerType.full_attention, model.layerType(47));
    try std.testing.expectEqual(LayerType.linear_attention, model.layerType(46));
}

test "the PLE layer id is one based" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = try parse(allocator, test_config_json);
    try std.testing.expectEqualSlices(u32, &.{2}, config.text_config.ple_layer_ids);
    const model = try TextModel.init(config.text_config);
    try std.testing.expectEqual(@as(u32, 1), model.ple_layer_index);
}

test "a config missing a key is rejected instead of defaulted" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    try std.testing.expectError(
        error.malformed_config,
        parse(allocator, "{\"text_config\":{}}"),
    );
}
