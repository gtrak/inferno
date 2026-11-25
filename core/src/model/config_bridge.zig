//! Bridges the production checkpoint config onto the reference config.
//!
//! `model/plan.zig` and `model/exec.zig` were written against
//! `ref/fixtures.zig`'s fixed-size `RefConfig`, because that is the shape every
//! kernel was validated against. The real checkpoint validates against
//! `weights/config.zig`'s nested `TextConfig`, which is slice-based and uses the
//! checkpoint's own vocabulary for two names (`full_attention` instead of
//! `indexed_attention`).
//!
//! Rather than fork the plan and the executor on config type - which would give
//! two spellings of the layer schedule and invite exactly the drift that bug
//! class #1 was - the real config is translated once, at load, into the
//! reference shape. One plan, one executor, and the tiny fixture keeps working.
//!
//! The translation asserts the production config's own identity
//! (`hc_hidden_size == 2 * key_dim + value_dim`) is *already* enforced by
//! `weights/config.zig`; it re-checks the pieces the plan depends on and adds no
//! new arithmetic.

const std = @import("std");

const config = @import("../weights/config.zig");
const fixtures = @import("../ref/fixtures.zig");
const ops = @import("../ref/ops.zig");

const assert = std.debug.assert;

/// The mRoPE split has three sections in this model family.
const mrope_section_count: usize = 3;

/// Translates a validated production text config into the reference shape.
///
/// Fails when the checkpoint carries geometry the fixed-size reference cannot
/// hold, rather than truncating: `layers_max` is a compile-time bound, and a
/// silent truncation here would drop layers from the forward.
pub fn toRefConfig(text: *const config.TextConfig) !fixtures.RefConfig {
    assert(text.num_hidden_layers > 0);
    assert(text.layer_types.len == text.num_hidden_layers);
    assert(text.hidden_size > 0);

    if (text.num_hidden_layers > fixtures.layers_max) {
        std.debug.print(
            "checkpoint has {d} layers; the reference plan holds {d}\n",
            .{ text.num_hidden_layers, fixtures.layers_max },
        );
        return error.TooManyLayers;
    } else {
        // Fits.
    }

    var result = fixtures.RefConfig{};
    result.hidden_size = text.hidden_size;
    result.hc_count = text.hc_count;
    result.hc_lowrank = text.hc_lowrank;
    result.num_hidden_layers = text.num_hidden_layers;
    result.vocab_size = text.vocab_size;
    result.num_attention_heads = text.num_attention_heads;
    result.num_key_value_heads = text.num_key_value_heads;
    result.head_dim = text.head_dim;
    result.attention_bias = text.attention_bias;
    result.linear_num_key_heads = text.linear_num_key_heads;
    result.linear_num_value_heads = text.linear_num_value_heads;
    result.linear_key_head_dim = text.linear_key_head_dim;
    result.linear_value_head_dim = text.linear_value_head_dim;
    result.linear_conv_kernel_dim = text.linear_conv_kernel_dim;
    result.num_experts = text.num_experts;
    result.num_experts_per_tok = text.num_experts_per_tok;
    result.moe_intermediate_size = text.moe_intermediate_size;
    result.shared_expert_intermediate_size = text.shared_expert_intermediate_size;
    result.norm_topk_prob = text.norm_topk_prob;
    result.rms_norm_eps = text.rms_norm_eps;
    result.eos_token_id = @intCast(text.eos_token_id);
    result.max_position_embeddings = text.max_position_embeddings;
    result.indexer_budget = text.indexer_budget;
    result.indexer_compress_ratio = text.indexer_compress_ratio;
    result.indexer_head_dim = text.indexer_head_dim;
    result.indexer_kv_heads = text.indexer_kv_heads;
    result.indexer_n_heads = text.indexer_n_heads;
    result.ple_embed_dim = text.ple_embed_dim;
    result.ple_conv_kernel_size = text.ple_conv_kernel_size;
    result.ngram_size = text.ngram_size;
    result.heads_per_ngram = text.heads_per_ngram;
    result.ngram_vocab_size_base = text.ngram_vocab_size_base;
    result.make_ngram_vocab_size_divisible_by =
        text.make_ngram_vocab_size_divisible_by;
    result.split_ngram_parts = text.split_ngram_parts;
    result.seed = text.seed;

    try copyLayerTypes(text, &result);
    try copyRope(text, &result);
    try copyPleLayers(text, &result);
    result.output_gate_type = try activation(text.output_gate_type);

    return result;
}

/// Layer kinds differ by name only: the checkpoint calls the indexed block
/// `full_attention`, the reference calls it `indexed_attention`.
fn copyLayerTypes(
    text: *const config.TextConfig,
    result: *fixtures.RefConfig,
) !void {
    assert(text.layer_types.len == result.num_hidden_layers);
    assert(result.num_hidden_layers <= fixtures.layers_max);

    for (text.layer_types, 0..) |kind, index| {
        result.layer_types[index] = switch (kind) {
            .linear_attention => .linear_attention,
            .full_attention => .indexed_attention,
        };
    }
}

/// Rotary parameters live under `rope_parameters` in the checkpoint. The
/// `partial_rotary_factor` is duplicated at the top level; both spellings must
/// agree or the two consumers would disagree silently.
fn copyRope(
    text: *const config.TextConfig,
    result: *fixtures.RefConfig,
) !void {
    assert(text.rope_parameters.rope_theta > 0.0);
    assert(text.partial_rotary_factor > 0.0);

    const rope = &text.rope_parameters;
    if (rope.partial_rotary_factor != text.partial_rotary_factor) {
        std.debug.print(
            "partial_rotary_factor disagrees: {d} under rope_parameters, " ++
                "{d} at top level\n",
            .{ rope.partial_rotary_factor, text.partial_rotary_factor },
        );
        return error.RotaryFactorMismatch;
    } else {
        // Both spellings agree.
    }

    if (rope.mrope_section.len != mrope_section_count) {
        std.debug.print(
            "mrope_section has {d} entries; {d} expected\n",
            .{ rope.mrope_section.len, mrope_section_count },
        );
        return error.BadMropeSection;
    } else {
        // The split has the expected arity.
    }

    result.rope_theta = @floatCast(rope.rope_theta);
    result.partial_rotary_factor = rope.partial_rotary_factor;
    for (rope.mrope_section, 0..) |section, index| {
        result.mrope_section[index] = section;
    }
}

/// The PLE layer list is what `Plan` uses to decide which layer owns the block;
/// an empty list would leave the model without its n-gram embeddings.
fn copyPleLayers(
    text: *const config.TextConfig,
    result: *fixtures.RefConfig,
) !void {
    assert(text.ple_layer_ids.len <= fixtures.layers_max);

    if (text.ple_layer_ids.len == 0) {
        std.debug.print("checkpoint declares no PLE layers\n", .{});
        return error.NoPleLayers;
    } else {
        // At least one layer owns the block.
    }

    for (text.ple_layer_ids, 0..) |id, index| {
        if (id >= result.num_hidden_layers) {
            std.debug.print(
                "PLE layer id {d} is outside {d} layers\n",
                .{ id, result.num_hidden_layers },
            );
            return error.PleLayerOutOfRange;
        } else {
            result.ple_layer_ids[index] = id;
        }
    }
    result.ple_layer_ids_count = @intCast(text.ple_layer_ids.len);
}

/// `output_gate_type` is a string in the checkpoint and an enum in the
/// reference.
fn activation(tag: []const u8) !ops.Activation {
    assert(tag.len > 0);
    assert(tag.len <= 16);

    if (std.mem.eql(u8, tag, "silu")) {
        return .silu;
    } else if (std.mem.eql(u8, tag, "sigmoid")) {
        return .sigmoid;
    } else {
        std.debug.print("unsupported output_gate_type {s}\n", .{tag});
        return error.UnsupportedActivation;
    }
}

/// Reads the checkpoint's config file, or null when no checkpoint is named.
pub fn loadReal(
    allocator: std.mem.Allocator,
) !?fixtures.RefConfig {
    assert(allocator.ptr != undefined);

    const variable = "INFERNO_MODEL_DIR";
    const directory = std.posix.getenv(variable) orelse return null;
    assert(directory.len > 0);

    const path = try std.fs.path.join(
        allocator,
        &.{ directory, "config.json" },
    );
    defer allocator.free(path);

    var file = std.fs.cwd().openFile(path, .{}) catch |problem| {
        std.debug.print("no config.json at {s}: {s}\n", .{ path, @errorName(problem) });
        return null;
    };
    defer file.close();

    const bytes = try file.readToEndAlloc(allocator, 1 << 20);
    defer allocator.free(bytes);

    // `parse` is leaky on purpose: its slices point into `allocator`, which the
    // caller owns. The translation copies every value the plan keeps, so the
    // config's own storage is free to die with the caller's arena.
    const parsed = try config.parse(allocator, bytes);
    return try toRefConfig(&parsed.text_config);
}

test "the real checkpoint config translates into the reference shape" {
    // An arena, not a budget: `weights/config.zig`'s `parse` is leaky ON PURPOSE (its
    // slices point into the caller's allocator), so a tracked allocator reports the
    // intended leak. `testing_alloc.trackedBudget` is for asserting that a path you own
    // allocates O(1) times; it is not for leaky-by-design parsers. The cliff that cost an
    // evening is guarded where it applies: G3 requires an arena at the bulk entry points.
    var arena = @import("../testing_alloc.zig").arena();
    defer arena.deinit();
    const allocator = arena.allocator();

    const maybe = try loadReal(allocator);
    if (maybe == null) {
        std.debug.print("SKIP: set INFERNO_MODEL_DIR to check the real config\n", .{});
        return error.SkipZigTest;
    } else {
        // A checkpoint was named.
    }
    const real = maybe.?;

    // The documented geometry of this checkpoint (plan.md, model card). These
    // are the numbers the whole placement, tier and prefill arithmetic assume,
    // so they are pinned here rather than read back from the same file.
    try std.testing.expectEqual(@as(u32, 48), real.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 2560), real.hidden_size);
    try std.testing.expectEqual(@as(u32, 248320), real.vocab_size);
    try std.testing.expectEqual(@as(u32, 512), real.num_experts);
    try std.testing.expectEqual(@as(u32, 10), real.num_experts_per_tok);
    try std.testing.expectEqual(@as(u32, 640), real.moe_intermediate_size);
    try std.testing.expectEqual(@as(u32, 256), real.head_dim);
    try std.testing.expectEqual(@as(u32, 24), real.num_attention_heads);
    try std.testing.expectEqual(@as(u32, 2), real.num_key_value_heads);
    try std.testing.expectEqual(@as(u32, 262144), real.max_position_embeddings);
    try std.testing.expectEqual(@as(usize, 3), real.mrope_section.len);
    try std.testing.expectEqual(
        [3]u32{ 11, 11, 10 },
        real.mrope_section,
    );

    // The hybrid schedule: 36 linear-attention layers and 12 indexed ones.
    var linear: u32 = 0;
    var indexed: u32 = 0;
    for (real.layer_types[0..real.num_hidden_layers]) |kind| {
        switch (kind) {
            .linear_attention => linear += 1,
            .indexed_attention => indexed += 1,
        }
    }
    try std.testing.expectEqual(@as(u32, 36), linear);
    try std.testing.expectEqual(@as(u32, 12), indexed);

    // PLE owns exactly one layer. The id is 1-BASED: the checkpoint says
    // `ple_layer_ids: [2]` and its tensors are named
    // `model.language_model.layers.1.ple.*`, so id 2 means layer index 1.
    // `RefConfig.ple_layer_index` already reads it that way (`id == index + 1`);
    // assuming the 0-based HF convention here would silently move the whole PLE
    // block one layer.
    try std.testing.expectEqual(@as(u32, 1), real.ple_layer_ids_count);
    try std.testing.expectEqual(@as(u32, 2), real.ple_layer_ids[0]);

    // The translated config must drive the real plan, not just parse.
    const plan = try @import("plan.zig").Plan.init(real);
    try std.testing.expectEqual(@as(u32, 48), plan.layer_count);
    var ple_layers: u32 = 0;
    for (0..plan.layer_count) |index| {
        if (plan.layers[index].has_ple) {
            ple_layers += 1;
        }
    }
    try std.testing.expectEqual(@as(u32, 1), ple_layers);

    // The plan reports a 0-based layer index, so the 1-based config id maps to
    // `id - 1`. Layer 1 is also where the checkpoint's PLE tensors live.
    try std.testing.expectEqual(@as(u32, 1), plan.ple_layer);
    try std.testing.expectEqual(real.ple_layer_ids[0] - 1, plan.ple_layer);
}
