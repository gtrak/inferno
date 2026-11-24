//! The full text graph: embed, repeat into hyper-connections, the decoder
//! layers, the final mixer, and `lm_head`. Plus the caches `generate` needs.
//!
//! Reference lines 1375-1520 (`Qwen4ExpTextModel.forward`). There is no final
//! RMSNorm: `hyper_connection_mixer` is the last block before `lm_head`.

const std = @import("std");

const fixtures = @import("fixtures.zig");
const hc = @import("hc.zig");
const layer = @import("layer.zig");
const gdn = @import("gdn.zig");
const attention = @import("attention.zig");
const ops = @import("ops.zig");
const ple = @import("ple.zig");
const rope = @import("rope.zig");

const assert = std.debug.assert;

/// All weights of the text model, resolved to slices of one checkpoint.
pub const Weights = struct {
    /// `embed_tokens.weight`, `[vocab, hidden]`.
    embed_tokens: []const f32,
    /// `lm_head.weight`, `[vocab, hidden]`.
    lm_head: []const f32,
    layers: [fixtures.layers_max]layer.Weights,
    /// `hyper_connection_mixer`, with `use_combine = false`.
    mixer: hc.Weights,
};

/// Per-layer caches plus the number of tokens written so far.
pub const Cache = struct {
    layers: [fixtures.layers_max]layer.State,
    /// Layers actually in use (`num_hidden_layers`).
    layer_count: u32,
    length: u32,

    /// Allocates the caches for `config`.
    pub fn init(
        allocator: std.mem.Allocator,
        config: *const fixtures.RefConfig,
    ) !Cache {
        assert(config.num_hidden_layers > 0);
        assert(config.num_hidden_layers <= fixtures.layers_max);

        var cache = Cache{ .layers = undefined, .layer_count = 0, .length = 0 };
        var created: u32 = 0;
        errdefer {
            var index: u32 = 0;
            while (index < created) : (index += 1) {
                cache.layers[index].deinit(allocator);
            }
        }
        for (0..config.num_hidden_layers) |index| {
            cache.layers[index] = try layer.State.init(
                allocator,
                config,
                @intCast(index),
            );
            created += 1;
        }
        cache.layer_count = created;
        assert(cache.layer_count == config.num_hidden_layers);
        return cache;
    }

    /// Releases the caches.
    pub fn deinit(self: *Cache, allocator: std.mem.Allocator) void {
        assert(self.layer_count > 0);
        assert(self.layers[0].gdn.conv.len > 0);

        for (self.layers[0..self.layer_count]) |*slot| {
            slot.deinit(allocator);
        }
    }

    /// Forgets every cached token.
    pub fn reset(self: *Cache, config: *const fixtures.RefConfig) void {
        assert(config.num_hidden_layers > 0);
        assert(config.num_hidden_layers <= fixtures.layers_max);

        self.length = 0;
        for (self.layers[0..config.num_hidden_layers]) |*slot| {
            slot.reset(config);
        }
    }
};

/// Per-forward buffers: the global activation space plus the per-layer set.
pub const Scratch = struct {
    layer: layer.Scratch,
    /// `[max_position, hyper]`: the multi-stream residual.
    hyper: []f32,
    /// `[max_position, hidden]`: the token embedding.
    embed: []f32,
    /// `[max_position, hidden]`: the final mixer output.
    mixed: []f32,
    /// `[max_position, vocab]`: the logits of the last forward.
    logits: []f32,
    /// `[max_position, rotary_dim]`: mRoPE cos/sin for every position.
    cos: []f32,
    sin: []f32,
    /// `[max_position]` i64: the ids of the last forward.
    ids: []i64,
    /// `[max_position]`: the absolute positions of the last forward.
    positions: []u32,

    /// Allocates everything a forward needs.
    pub fn init(
        allocator: std.mem.Allocator,
        config: *const fixtures.RefConfig,
    ) !Scratch {
        assert(config.max_position_embeddings > 0);
        assert(config.vocab_size > 0);

        const tokens: usize = config.max_position_embeddings;
        const hyper: usize = config.hyper_size();
        const rotary: usize = config.rotary_dim();

        var scratch = Scratch{
            .layer = try layer.Scratch.init(allocator, config),
            .hyper = try allocator.alloc(f32, tokens * hyper),
            .embed = try allocator.alloc(f32, tokens * config.hidden_size),
            .mixed = try allocator.alloc(f32, tokens * config.hidden_size),
            .logits = try allocator.alloc(f32, tokens * config.vocab_size),
            .cos = try allocator.alloc(f32, tokens * rotary),
            .sin = try allocator.alloc(f32, tokens * rotary),
            .ids = try allocator.alloc(i64, tokens),
            .positions = try allocator.alloc(u32, tokens),
        };
        for (0..tokens) |position| {
            scratch.positions[position] = @intCast(position);
        }
        scratch.fill_rope(config);
        assert(scratch.cos.len == tokens * rotary);
        return scratch;
    }

    /// Precomputes the mRoPE tables for every position the model can visit.
    fn fill_rope(self: *Scratch, config: *const fixtures.RefConfig) void {
        assert(self.cos.len > 0);
        assert(self.sin.len == self.cos.len);

        const tokens: usize = config.max_position_embeddings;
        var frequencies: [64]f32 = undefined;
        assert(config.rotary_freqs() <= frequencies.len);
        const freqs = frequencies[0..config.rotary_freqs()];
        rope.inv_freq_init(config.rope_theta, freqs);
        // Text-only: the temporal, height and width grids are the same
        // positions, so the mRoPE interleave is a no-op on the values.
        const positions = self.positions[0..tokens];
        rope.cos_sin(
            .{ positions, positions, positions },
            freqs,
            config.mrope_section,
            self.cos,
            self.sin,
        );
        assert(self.sin.len == tokens * config.rotary_dim());
    }

    /// Releases everything.
    pub fn deinit(self: *Scratch, allocator: std.mem.Allocator) void {
        assert(self.cos.len > 0);
        assert(self.logits.len > 0);

        allocator.free(self.positions);
        allocator.free(self.ids);
        allocator.free(self.sin);
        allocator.free(self.cos);
        allocator.free(self.logits);
        allocator.free(self.mixed);
        allocator.free(self.embed);
        allocator.free(self.hyper);
        self.layer.deinit(allocator);
    }
};

/// Loads every weight of the text model from `tensors`.
pub fn load_weights(
    target: *Weights,
    tensors: *const fixtures.Tensors,
    config: *const fixtures.RefConfig,
) !void {
    assert(config.num_hidden_layers > 0);
    assert(tensors.entry_count > 0);

    const embedding = try tensors.tensor(
        "model.language_model.embed_tokens.weight",
    );
    target.embed_tokens = embedding.values;
    const head = try tensors.tensor("lm_head.weight");
    target.lm_head = head.values;
    var mixer_loader = try layer.Loader.init_root(
        tensors,
        "model.language_model.",
    );
    target.mixer = try layer.load_hyper(
        &mixer_loader,
        "hyper_connection_mixer.",
        false,
    );
    for (0..config.num_hidden_layers) |index| {
        target.layers[index] = try layer.load(tensors, @intCast(index), config);
    }
    assert(target.mixer.inject == null);
}

/// The model: config, weights and caches.
pub const Model = struct {
    config: fixtures.RefConfig,
    weights: Weights,
    cache: Cache,

    /// Loads the weights and allocates the caches.
    pub fn init(
        self: *Model,
        allocator: std.mem.Allocator,
        tensors: *const fixtures.Tensors,
    ) !void {
        assert(tensors.entry_count > 0);
        assert(tensors.data_start > 0);

        const config = try fixtures.load_config(allocator);
        self.* = .{
            .config = config,
            .weights = undefined,
            .cache = try Cache.init(allocator, &config),
        };
        errdefer self.cache.deinit(allocator);
        try load_weights(&self.weights, tensors, &self.config);
        assert(self.cache.length == 0);
    }

    /// Releases the caches.
    pub fn deinit(self: *Model, allocator: std.mem.Allocator) void {
        assert(self.cache.layer_count == self.config.num_hidden_layers);
        assert(self.config.num_hidden_layers > 0);

        self.cache.deinit(allocator);
    }

    /// One forward over `tokens` new positions, writing `scratch.logits`.
    pub fn forward(
        self: *Model,
        ids: []const i64,
        tokens: u32,
        scratch: *Scratch,
    ) !void {
        assert(tokens > 0);
        assert(ids.len == tokens);
        assert(self.cache.length + tokens <=
            self.config.max_position_embeddings);

        const hyper: usize = self.config.hyper_size();
        const hidden: usize = self.config.hidden_size;
        const past: u32 = self.cache.length;
        for (0..tokens) |t| {
            scratch.ids[t] = ids[t];
            scratch.positions[t] = past + @as(u32, @intCast(t));
        }
        ops.embed_gather(
            self.weights.embed_tokens,
            self.config.hidden_size,
            ids,
            scratch.embed[0 .. tokens * hidden],
        );
        for (0..tokens) |t| {
            for (0..self.config.hc_count) |c| {
                for (0..hidden) |h| {
                    scratch.hyper[t * hyper + c * hidden + h] =
                        scratch.embed[t * hidden + h];
                }
            }
        }
        const table = attention.Rope{
            .cos = scratch.cos,
            .sin = scratch.sin,
        };
        const recurrent_form = tokens == 1 and past > 0;
        for (0..self.config.num_hidden_layers) |index| {
            layer.forward(
                scratch.hyper[0 .. tokens * hyper],
                tokens,
                scratch.positions[0..tokens],
                ids,
                &self.weights.layers[index],
                &self.cache.layers[index],
                &scratch.layer,
                table,
                &self.config,
                recurrent_form,
                null,
            );
        }
        hc.forward(
            scratch.hyper[0 .. tokens * hyper],
            tokens,
            scratch.mixed[0 .. tokens * hidden],
            null,
            &self.weights.mixer,
            &scratch.layer.hc_scratch,
            &self.config,
        );
        ops.matmul(
            scratch.mixed[0 .. tokens * hidden],
            self.weights.lm_head,
            scratch.logits[0 .. tokens * self.config.vocab_size],
            .{
                .rows = tokens,
                .in_features = self.config.hidden_size,
                .out_features = self.config.vocab_size,
            },
        );
        self.cache.length += tokens;
        assert(self.cache.length == past + tokens);
    }

    /// Greedy continuation of `prompt` for `count` tokens.
    pub fn generate(
        self: *Model,
        prompt: []const i64,
        count: u32,
        out: []i64,
        scratch: *Scratch,
    ) !void {
        assert(prompt.len > 0);
        assert(count > 0);
        assert(out.len == count);

        self.cache.reset(&self.config);
        try self.forward(prompt, @intCast(prompt.len), scratch);
        const vocab: usize = self.config.vocab_size;
        var next: i64 = @intCast(ops.argmax(
            scratch.logits[(prompt.len - 1) * vocab ..][0..vocab],
        ));
        out[0] = next;
        for (1..count) |step| {
            const single = [_]i64{next};
            try self.forward(&single, 1, scratch);
            next = @intCast(ops.argmax(scratch.logits[0..vocab]));
            out[step] = next;
        }
        assert(out.len == count);
    }
};
