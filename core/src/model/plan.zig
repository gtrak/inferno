//! The static execution plan of one text model.
//!
//! Everything a forward needs is decided here, once: which layers exist, which
//! of them use the linear-attention (Gated DeltaNet) block and which the
//! indexed-attention block, and which layer owns the PLE block. `exec.zig`
//! walks this plan; no layer type or PLE index is read from the config during a
//! forward.
//!
//! The plan is derived from the tiny fixture's own `config.json`, which is the
//! flat text configuration. `weights/config.zig`'s `TextModel` validates the
//! *production* nested checkpoint, but it asserts the real model's identity
//! `hc_hidden_size == 2 * key_dim + value_dim`, which the random tiny model does
//! not satisfy (256 versus 160). Using the fixture config keeps one source of
//! truth for the shapes every kernel already validated against.

const std = @import("std");

const fixtures = @import("../ref/fixtures.zig");

const assert = std.debug.assert;

/// What the attention slot of one layer runs.
pub const BlockKind = enum {
    linear_attention,
    indexed_attention,
};

/// One decoder layer's static shape.
pub const LayerPlan = struct {
    kind: BlockKind,
    /// True on the single layer named by `ple_layer_ids`.
    has_ple: bool,
};

/// The whole plan: one entry per hidden layer.
pub const Plan = struct {
    config: fixtures.RefConfig,
    layers: [fixtures.layers_max]LayerPlan = undefined,
    layer_count: u32 = 0,
    ple_layer: u32 = 0,

    /// Derives the plan from a validated tiny config.
    pub fn init(config: fixtures.RefConfig) !Plan {
        assert(config.num_hidden_layers > 0);
        assert(config.num_hidden_layers <= fixtures.layers_max);

        var plan = Plan{ .config = config };
        var ple_count: u32 = 0;
        for (0..config.num_hidden_layers) |index| {
            const kind: BlockKind = switch (config.layer_types[index]) {
                .linear_attention => .linear_attention,
                .indexed_attention => .indexed_attention,
            };
            const has_ple = config.ple_layer_index(@intCast(index)) != null;
            plan.layers[index] = .{ .kind = kind, .has_ple = has_ple };
            if (has_ple) {
                plan.ple_layer = @intCast(index);
                ple_count += 1;
            } else {
                assert(!has_ple);
            }
        }
        if (ple_count != 1) {
            return error.PleLayerCount;
        }
        plan.layer_count = config.num_hidden_layers;
        assert(plan.layers[plan.ple_layer].has_ple);
        return plan;
    }

    /// The layer plan at `index`.
    pub fn layer(self: *const Plan, index: u32) LayerPlan {
        assert(index < self.layer_count);
        assert(self.layer_count == self.config.num_hidden_layers);

        return self.layers[index];
    }
};
