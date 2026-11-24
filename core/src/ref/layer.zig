//! One decoder layer: PLE injection, hyper-connection + attention + injection,
//! hyper-connection + MoE + injection.
//!
//! Reference lines 1274-1325. `hyper_input` is the multi-stream residual; the
//! block writes its single-stream output back into every stream through the
//! injection weights of the block that produced it.

const std = @import("std");

const attention = @import("attention.zig");
const fixtures = @import("fixtures.zig");
const gdn = @import("gdn.zig");
const hc = @import("hc.zig");
const moe = @import("moe.zig");
const ple = @import("ple.zig");

const assert = std.debug.assert;

/// Longest tensor name this loader builds.
const name_bytes_max: u32 = 192;

/// Weights of one decoder layer.
pub const Weights = struct {
    kind: fixtures.LayerType,
    /// Present for `linear_attention` layers.
    gdn: gdn.Weights,
    /// Present for `indexed_attention` layers.
    attention: attention.Weights,
    moe: moe.Weights,
    /// Present on the layer named by `ple_layer_ids`.
    ple: ?ple.Weights,
    attn_hyper: hc.Weights,
    mlp_hyper: hc.Weights,
};

/// State of one decoder layer (all of it cache-resident across steps).
pub const State = struct {
    gdn: gdn.State,
    attention: attention.State,
    /// Present on the layer named by `ple_layer_ids`.
    ple: ?ple.State,

    /// Allocates the state of layer `layer_index`.
    pub fn init(
        allocator: std.mem.Allocator,
        config: *const fixtures.RefConfig,
        layer_index: u32,
    ) !State {
        assert(layer_index < config.num_hidden_layers);
        assert(config.head_dim > 0);

        var state = State{
            .gdn = try gdn.State.init(allocator, config),
            .attention = undefined,
            .ple = null,
        };
        state.attention = attention.State.init(
            allocator,
            config,
            config.max_position_embeddings,
        ) catch |err| {
            state.gdn.deinit(allocator);
            return err;
        };
        if (config.ple_layer_index(layer_index) != null) {
            state.ple = try ple.State.init(allocator, config);
        } else {
            assert(state.ple == null);
        }
        return state;
    }

    /// Releases the state.
    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        assert(self.gdn.conv.len > 0);
        assert(self.attention.key.len > 0);

        if (self.ple) |*ple_state| {
            ple_state.deinit(allocator);
        } else {
            assert(self.ple == null);
        }
        self.attention.deinit(allocator);
        self.gdn.deinit(allocator);
    }

    /// Forgets every cached token.
    pub fn reset(self: *State, config: *const fixtures.RefConfig) void {
        assert(self.gdn.conv.len > 0);
        assert(self.attention.key.len > 0);

        self.gdn.reset();
        self.attention.reset();
        if (self.ple) |*ple_state| {
            ple_state.reset(config);
        } else {
            assert(self.ple == null);
        }
    }
};

/// Per-layer views into a checkpoint, by name.
pub const Loader = struct {
    tensors: *const fixtures.Tensors,
    prefix: [name_bytes_max]u8 = undefined,
    prefix_length: u32 = 0,

    /// Prepares a loader for `model.language_model.layers.{layer}.`.
    pub fn init(
        tensors: *const fixtures.Tensors,
        layer: u32,
    ) !Loader {
        assert(layer < fixtures.layers_max);
        assert(tensors.entry_count > 0);

        var loader = Loader{ .tensors = tensors };
        const prefix = try std.fmt.bufPrint(
            &loader.prefix,
            "model.language_model.layers.{d}.",
            .{layer},
        );
        loader.prefix_length = @intCast(prefix.len);
        assert(loader.prefix_length > 0);
        return loader;
    }

    /// Prepares a loader for an explicit `prefix` (the final mixer, say).
    pub fn init_root(
        tensors: *const fixtures.Tensors,
        prefix: []const u8,
    ) !Loader {
        assert(prefix.len > 0);
        assert(prefix.len < name_bytes_max);

        var loader = Loader{ .tensors = tensors };
        const copied = try std.fmt.bufPrint(&loader.prefix, "{s}", .{prefix});
        loader.prefix_length = @intCast(copied.len);
        assert(loader.prefix_length == prefix.len);
        return loader;
    }

    /// Loads an f32 tensor whose name is `prefix ++ suffix`.
    pub fn tensor_f32(self: *Loader, suffix: []const u8) ![]const f32 {
        assert(suffix.len > 0);
        assert(self.prefix_length > 0);

        const full_name = try self.name(suffix);
        const tensor = try self.tensors.tensor(full_name);
        return tensor.values;
    }

    /// Loads an i64 tensor whose name is `prefix ++ suffix`.
    pub fn tensor_i64(self: *Loader, suffix: []const u8) ![]const i64 {
        assert(suffix.len > 0);
        assert(self.prefix_length > 0);

        const full_name = try self.name(suffix);
        return self.tensors.tensor_i64(full_name);
    }

    /// Builds `prefix ++ suffix` in `self.prefix`; the slice is short-lived.
    fn name(self: *Loader, suffix: []const u8) ![]const u8 {
        assert(suffix.len > 0);
        assert(self.prefix_length < name_bytes_max);

        const start: usize = self.prefix_length;
        const tail = try std.fmt.bufPrint(
            self.prefix[start..],
            "{s}",
            .{suffix},
        );
        assert(start + tail.len <= name_bytes_max);
        return self.prefix[0 .. start + tail.len];
    }
};

/// Loads every weight of layer `layer_index` from the checkpoint.
pub fn load(
    tensors: *const fixtures.Tensors,
    layer_index: u32,
    config: *const fixtures.RefConfig,
) !Weights {
    assert(layer_index < config.num_hidden_layers);
    assert(tensors.entry_count > 0);

    var loader = try Loader.init(tensors, layer_index);
    const kind = config.layer_types[layer_index];
    var weights = Weights{
        .kind = kind,
        .gdn = undefined,
        .attention = undefined,
        .moe = try load_moe(&loader, config),
        .ple = null,
        .attn_hyper = try load_hyper(&loader, "attn_hyper_connection.", true),
        .mlp_hyper = try load_hyper(&loader, "mlp_hyper_connection.", true),
    };
    if (kind == .linear_attention) {
        weights.gdn = try load_gdn(&loader);
    } else {
        weights.attention = try load_attention(&loader, config);
    }
    if (config.ple_layer_index(layer_index) != null) {
        weights.ple = try load_ple(&loader, config);
    } else {
        assert(weights.ple == null);
    }
    assert(weights.kind == kind);
    return weights;
}

/// Loads the router, the per-expert matrices and the shared expert.
fn load_moe(
    loader: *Loader,
    config: *const fixtures.RefConfig,
) !moe.Weights {
    assert(config.num_experts > 0);
    assert(config.num_experts <= moe.experts_max);

    var weights = moe.Weights{
        .router = try loader.tensor_f32("mlp.gate.weight"),
        .gate_proj = undefined,
        .up_proj = undefined,
        .down_proj = undefined,
        .shared_gate = try loader.tensor_f32(
            "mlp.shared_expert.gate_proj.weight",
        ),
        .shared_up = try loader.tensor_f32("mlp.shared_expert.up_proj.weight"),
        .shared_down = try loader.tensor_f32(
            "mlp.shared_expert.down_proj.weight",
        ),
        .shared_gate_logit = try loader.tensor_f32(
            "mlp.shared_expert_gate.weight",
        ),
    };
    var suffixes: [3][name_bytes_max]u8 = undefined;
    for (0..config.num_experts) |expert| {
        weights.gate_proj[expert] = try loader.tensor_f32(try std.fmt.bufPrint(
            &suffixes[0],
            "mlp.experts.{d}.gate_proj.weight",
            .{expert},
        ));
        weights.up_proj[expert] = try loader.tensor_f32(try std.fmt.bufPrint(
            &suffixes[1],
            "mlp.experts.{d}.up_proj.weight",
            .{expert},
        ));
        weights.down_proj[expert] = try loader.tensor_f32(try std.fmt.bufPrint(
            &suffixes[2],
            "mlp.experts.{d}.down_proj.weight",
            .{expert},
        ));
    }
    // The remaining expert slots stay unset; nothing indexes past
    // `num_experts`.
    for (config.num_experts..moe.experts_max) |expert| {
        weights.gate_proj[expert] = &[_]f32{};
        weights.up_proj[expert] = &[_]f32{};
        weights.down_proj[expert] = &[_]f32{};
    }
    assert(weights.gate_proj[0].len == config.moe_intermediate_size *
        config.hidden_size);
    return weights;
}

/// Loads one hyper-connection block (`use_combine` false for the mixer).
pub fn load_hyper(
    loader: *Loader,
    comptime prefix: []const u8,
    use_combine: bool,
) !hc.Weights {
    assert(prefix.len > 0);
    assert(prefix.len < name_bytes_max);

    var suffixes: [3][name_bytes_max]u8 = undefined;
    return .{
        .norm = try loader.tensor_f32(try std.fmt.bufPrint(
            &suffixes[0],
            "{s}hc_norm.weight",
            .{prefix},
        )),
        .mix_down = try loader.tensor_f32(try std.fmt.bufPrint(
            &suffixes[1],
            "{s}input_mix_weight_down.weight",
            .{prefix},
        )),
        .mix_up = try loader.tensor_f32(try std.fmt.bufPrint(
            &suffixes[2],
            "{s}input_mix_weight_up.weight",
            .{prefix},
        )),
        .inject = if (use_combine) try loader.tensor_f32(try std.fmt.bufPrint(
            &suffixes[0],
            "{s}block_inject_weight.weight",
            .{prefix},
        )) else null,
    };
}

/// Loads the GatedDeltaNet weights.
fn load_gdn(loader: *Loader) !gdn.Weights {
    return .{
        .in_proj_qkv = try loader.tensor_f32("linear_attn.in_proj_qkv.weight"),
        .in_proj_z = try loader.tensor_f32("linear_attn.in_proj_z.weight"),
        .in_proj_b = try loader.tensor_f32("linear_attn.in_proj_b.weight"),
        .in_proj_a = try loader.tensor_f32("linear_attn.in_proj_a.weight"),
        .conv = try loader.tensor_f32("linear_attn.conv1d.weight"),
        .norm = try loader.tensor_f32("linear_attn.norm.weight"),
        .out_proj = try loader.tensor_f32("linear_attn.out_proj.weight"),
        .a_log = try loader.tensor_f32("linear_attn.A_log"),
        .dt_bias = try loader.tensor_f32("linear_attn.dt_bias"),
    };
}

/// Loads the indexed-attention weights, indexer included.
fn load_attention(
    loader: *Loader,
    config: *const fixtures.RefConfig,
) !attention.Weights {
    assert(config.indexer_n_heads > 0);
    assert(config.indexer_kv_heads > 0);

    const weights: attention.Weights = .{
        .q_proj = try loader.tensor_f32("self_attn.q_proj.weight"),
        .k_proj = try loader.tensor_f32("self_attn.k_proj.weight"),
        .v_proj = try loader.tensor_f32("self_attn.v_proj.weight"),
        .o_proj = try loader.tensor_f32("self_attn.o_proj.weight"),
        .q_norm = try loader.tensor_f32("self_attn.q_norm.weight"),
        .k_norm = try loader.tensor_f32("self_attn.k_norm.weight"),
        .index_qk_proj = try loader.tensor_f32(
            "self_attn.indexer.index_qk_proj.weight",
        ),
        .q_layernorm = try loader.tensor_f32(
            "self_attn.indexer.q_layernorm.weight",
        ),
        .k_layernorm = try loader.tensor_f32(
            "self_attn.indexer.k_layernorm.weight",
        ),
    };
    assert(weights.index_qk_proj.len == (config.indexer_n_heads +
        config.indexer_kv_heads) * config.indexer_head_dim *
        config.hidden_size);
    return weights;
}

/// Loads the PLE weights, gathering the n-gram table shards.
fn load_ple(
    loader: *Loader,
    config: *const fixtures.RefConfig,
) !ple.Weights {
    assert(config.split_ngram_parts >= 1);
    assert(config.split_ngram_parts <= ple.shards_max);

    const prefix = "ple.ple_embedding.ngram_embedding.shard_";
    var suffixes: [ple.shards_max][name_bytes_max]u8 = undefined;
    var shards: [ple.shards_max][]const f32 = undefined;
    for (0..config.split_ngram_parts) |shard| {
        const suffix = try std.fmt.bufPrint(
            &suffixes[shard],
            "{s}{d}.weight",
            .{ prefix, shard },
        );
        shards[shard] = try loader.tensor_f32(suffix);
    }
    for (config.split_ngram_parts..ple.shards_max) |shard| {
        shards[shard] = &[_]f32{};
    }
    return .{
        .shards = shards,
        .shard_count = config.split_ngram_parts,
        .head_dim = config.ngram_head_dim(),
        .multipliers = try loader.tensor_i64(
            "ple.ple_embedding.layer_multipliers",
        ),
        .head_vocab_sizes = try loader.tensor_i64(
            "ple.ple_embedding.ngram_heads_vocab_sizes",
        ),
        .head_offsets = try loader.tensor_i64(
            "ple.ple_embedding.ngram_heads_offsets",
        ),
        .key_proj = try loader.tensor_f32("ple.key_proj.weight"),
        .value_proj = try loader.tensor_f32("ple.value_proj.weight"),
        .norm_key = try loader.tensor_f32("ple.norm_key.weight"),
        .norm_query = try loader.tensor_f32("ple.norm_query.weight"),
        .norm_conv = try loader.tensor_f32("ple.norm_conv.weight"),
        .conv = try loader.tensor_f32("ple.conv1d.weight"),
    };
}

/// Optional copies of the intermediates the golden suite compares.
///
/// The per-block buffers are reused, so a caller that wants `hc_attn/mixed`,
/// `hc_attn/inject_weights` or the attention block output has to ask for a copy
/// while the value is still live.
pub const Probe = struct {
    /// `[tokens, hidden]`: the hyper-connection mix entering the block.
    block_mixed: ?[]f32 = null,
    /// `[tokens, hc_count]`: the block's injection weights.
    block_inject: ?[]f32 = null,
    /// `[tokens, hidden]`: the attention block output, before injection.
    block_out: ?[]f32 = null,
    /// `[tokens, hyper]`: the PLE output, when the layer has a PLE block.
    ple_out: ?[]f32 = null,
};

/// Copies `source` into `target` when the probe asked for it.
fn record(target: ?[]f32, source: []const f32) void {
    assert(source.len > 0);
    assert(source.len <= 1 << 20);

    if (target) |destination| {
        assert(destination.len >= source.len);
        @memcpy(destination[0..source.len], source);
    } else {
        // Not requested.
    }
}

/// One decoder layer forward, in place on the multi-stream residual.
pub fn forward(
    hyper: []f32,
    tokens: u32,
    positions: []const u32,
    input_ids: []const i64,
    weights: *const Weights,
    state: *State,
    scratch: *Scratch,
    table: attention.Rope,
    config: *const fixtures.RefConfig,
    recurrent_form: bool,
    probe: ?*Probe,
) void {
    assert(tokens > 0);
    assert(positions.len == tokens);
    assert(hyper.len == tokens * config.hyper_size());

    forward_ple(hyper, tokens, input_ids, weights, state, scratch, config, probe);
    forward_attention_half(
        hyper,
        tokens,
        positions,
        weights,
        state,
        scratch,
        table,
        config,
        recurrent_form,
        probe,
    );
    forward_mlp_half(hyper, tokens, weights, scratch, config);
    assert(hyper.len == tokens * config.hyper_size());
}

/// Adds the PLE block's output into the residual, when the layer has one.
fn forward_ple(
    hyper: []f32,
    tokens: u32,
    input_ids: []const i64,
    weights: *const Weights,
    state: *State,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
    probe: ?*Probe,
) void {
    assert(tokens > 0);
    assert(input_ids.len == tokens);

    const hyper_size: usize = tokens * config.hyper_size();
    if (weights.ple) |*ple_weights| {
        if (state.ple) |*ple_state| {
            ple.forward(
                hyper,
                input_ids,
                tokens,
                scratch.hyper_out[0..hyper_size],
                ple_weights,
                ple_state,
                &scratch.ple_scratch,
                config,
            );
            if (probe) |p| {
                record(p.ple_out, scratch.hyper_out[0..hyper_size]);
            } else {
                // Not requested.
            }
            for (hyper, scratch.hyper_out[0..hyper_size]) |*value, addend| {
                value.* += addend;
            }
        } else {
            // A PLE weight block without PLE state is a construction error.
            assert(state.ple == null);
        }
    } else {
        assert(weights.ple == null);
    }
}

/// Hyper-connection mix, the attention block, and the injection back.
fn forward_attention_half(
    hyper: []f32,
    tokens: u32,
    positions: []const u32,
    weights: *const Weights,
    state: *State,
    scratch: *Scratch,
    table: attention.Rope,
    config: *const fixtures.RefConfig,
    recurrent_form: bool,
    probe: ?*Probe,
) void {
    assert(tokens > 0);
    assert(positions.len == tokens);

    const hidden: usize = tokens * config.hidden_size;
    forward_block(hyper, tokens, &weights.attn_hyper, scratch, config);
    if (probe) |p| {
        record(p.block_mixed, scratch.stream_in[0..hidden]);
        record(p.block_inject, scratch.injection[0 .. tokens * config.hc_count]);
    } else {
        // Not requested.
    }
    forward_attention(
        tokens,
        positions,
        weights,
        state,
        scratch,
        table,
        config,
        recurrent_form,
    );
    if (probe) |p| {
        record(p.block_out, scratch.stream_out[0..hidden]);
    } else {
        // Not requested.
    }
    forward_inject(
        hyper,
        tokens,
        scratch.stream_out,
        scratch.injection,
        config,
    );
}

/// Hyper-connection mix, the MoE block, and the injection back.
fn forward_mlp_half(
    hyper: []f32,
    tokens: u32,
    weights: *const Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(hyper.len == tokens * config.hyper_size());

    const hidden: usize = tokens * config.hidden_size;
    forward_block(hyper, tokens, &weights.mlp_hyper, scratch, config);
    moe.forward(
        scratch.stream_in[0..hidden],
        tokens,
        scratch.stream_out[0..hidden],
        &weights.moe,
        &scratch.moe_scratch,
        config,
    );
    forward_inject(
        hyper,
        tokens,
        scratch.stream_out,
        scratch.injection,
        config,
    );
}

/// Hyper-connection mix of the residual into the block's single stream.
fn forward_block(
    hyper: []f32,
    tokens: u32,
    block_weights: *const hc.Weights,
    scratch: *Scratch,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(scratch.stream_in.len >= tokens * config.hidden_size);

    hc.forward(
        hyper,
        tokens,
        scratch.stream_in[0 .. tokens * config.hidden_size],
        scratch.injection[0 .. tokens * config.hc_count],
        block_weights,
        &scratch.hc_scratch,
        config,
    );
    assert(scratch.injection.len >= tokens * config.hc_count);
}

/// Runs whichever attention block this layer has.
fn forward_attention(
    tokens: u32,
    positions: []const u32,
    weights: *const Weights,
    state: *State,
    scratch: *Scratch,
    table: attention.Rope,
    config: *const fixtures.RefConfig,
    recurrent_form: bool,
) void {
    assert(tokens > 0);
    assert(scratch.stream_in.len >= tokens * config.hidden_size);

    if (weights.kind == .linear_attention) {
        gdn.forward(
            scratch.stream_in[0 .. tokens * config.hidden_size],
            tokens,
            scratch.stream_out[0 .. tokens * config.hidden_size],
            &weights.gdn,
            &state.gdn,
            &scratch.gdn_scratch,
            config,
            recurrent_form,
        );
    } else {
        attention.forward(
            scratch.stream_in[0 .. tokens * config.hidden_size],
            tokens,
            positions,
            scratch.stream_out[0 .. tokens * config.hidden_size],
            &weights.attention,
            &state.attention,
            &scratch.attention_scratch,
            table,
            config,
        );
    }
    assert(scratch.stream_out.len > 0);
}

/// Adds `block[t, :] * injection[t, c]` into every stream of the residual.
fn forward_inject(
    hyper: []f32,
    tokens: u32,
    block: []const f32,
    injection: []const f32,
    config: *const fixtures.RefConfig,
) void {
    assert(tokens > 0);
    assert(block.len >= tokens * config.hidden_size);

    const hidden: usize = config.hidden_size;
    const streams: usize = config.hc_count;
    for (0..tokens) |t| {
        for (0..streams) |c| {
            const weight = injection[t * streams + c];
            for (0..hidden) |h| {
                hyper[t * streams * hidden + c * hidden + h] +=
                    block[t * hidden + h] * weight;
            }
        }
    }
    assert(hyper.len >= tokens * streams * hidden);
}

/// Per-layer buffers, reused by every layer in turn.
pub const Scratch = struct {
    hc_scratch: hc.Scratch,
    gdn_scratch: gdn.Scratch,
    attention_scratch: attention.Scratch,
    moe_scratch: moe.Scratch,
    ple_scratch: ple.Scratch,
    /// `[tokens, hidden]`: the block's single-stream input.
    stream_in: []f32,
    /// `[tokens, hidden]`: the block's single-stream output.
    stream_out: []f32,
    /// `[tokens, hyper]`: the PLE output before it joins the residual.
    hyper_out: []f32,
    /// `[tokens, hc_count]`: the current block's injection weights.
    injection: []f32,

    /// Allocates the buffers for one sequence.
    pub fn init(
        allocator: std.mem.Allocator,
        config: *const fixtures.RefConfig,
    ) !Scratch {
        assert(config.max_position_embeddings > 0);
        assert(config.hc_count > 0);

        const tokens: usize = config.max_position_embeddings;

        const scratch = Scratch{
            .hc_scratch = try hc.Scratch.init(allocator, config),
            .gdn_scratch = try gdn.Scratch.init(allocator, config),
            .attention_scratch = try attention.Scratch.init(
                allocator,
                config,
                config.max_position_embeddings,
            ),
            .moe_scratch = try moe.Scratch.init(allocator, config),
            .ple_scratch = try ple.Scratch.init(allocator, config),
            .stream_in = try allocator.alloc(f32, tokens * config.hidden_size),
            .stream_out = try allocator.alloc(f32, tokens * config.hidden_size),
            .hyper_out = try allocator.alloc(f32, tokens * config.hyper_size()),
            .injection = try allocator.alloc(f32, tokens * config.hc_count),
        };
        assert(scratch.hyper_out.len == tokens * config.hyper_size());
        return scratch;
    }

    /// Releases everything.
    pub fn deinit(self: *Scratch, allocator: std.mem.Allocator) void {
        assert(self.hyper_out.len > 0);
        assert(self.injection.len > 0);

        allocator.free(self.injection);
        allocator.free(self.hyper_out);
        allocator.free(self.stream_out);
        allocator.free(self.stream_in);
        self.ple_scratch.deinit(allocator);
        self.moe_scratch.deinit(allocator);
        self.attention_scratch.deinit(allocator);
        self.gdn_scratch.deinit(allocator);
        self.hc_scratch.deinit(allocator);
    }
};
