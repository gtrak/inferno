//! The forward pass: embed → decoder layers → final mixer → `lm_head`.
//!
//! Every step is a kernel launch; the only host work a forward does is the
//! embedding gather (the table stays host-side, as the production placement
//! plan keeps it), the n-gram id derivation the PLE block reads, and the
//! position upload. The graph is walked from `plan.zig`, so no layer kind or
//! PLE index is read from the config during a forward.
//!
//! Two streams are in play. Everything the executor launches goes on the PLE
//! module's queue, which is also the queue `ple_project` — the f32 matmul the
//! whole dense path uses — runs on. The indexed-attention module owns its own
//! queue, so the executor fences around it: the encodings are short, and one
//! fence per layer is cheaper than threading events through every launch.

const std = @import("std");

const attention = @import("../kernels/attention.zig");
const cuda_context = @import("../hw/api.zig").cuda_context;
const cuda_driver = @import("../hw/api.zig").cuda_driver;
const guards = @import("../hw/buffer.zig");
const elementwise = @import("../kernels/elementwise.zig");
const fixtures = @import("../ref/fixtures.zig");
const gdn = @import("../kernels/gdn.zig");
const glue = @import("glue.zig");
const kernel = @import("../hw/kernel.zig");
const load = @import("load.zig");
const moe = @import("../kernels/moe.zig");
const plan_module = @import("plan.zig");
const ple = @import("../kernels/ple.zig");
const state_module = @import("state.zig");

const assert = std.debug.assert;

/// Wall time per graph phase, in nanoseconds. Information only: the check
/// prints it, no gate compares it.
pub const Timings = struct {
    embed_ns: u64 = 0,
    attention_ns: u64 = 0,
    moe_ns: u64 = 0,
    head_ns: u64 = 0,
    total_ns: u64 = 0,

    /// Adds `other` into `self`.
    pub fn accumulate(self: *Timings, other: Timings) void {
        self.embed_ns += other.embed_ns;
        self.attention_ns += other.attention_ns;
        self.moe_ns += other.moe_ns;
        self.head_ns += other.head_ns;
        self.total_ns += other.total_ns;
    }
};

/// Tokens one forward may cover; the executor's own bound, re-exported so a
/// multi-rank driver can size its exchange buffers from one spelling.
pub const tokens_max: u32 = state_module.tokens_max;

/// The device bytes of one expert the tier has placed: `gate_up` then `down`,
/// in the layout the loading path already uses, so a shard is a byte range and
/// not a re-interpretation.
pub const ExpertSlice = struct {
    gate_up: load.Matrix,
    down: load.Matrix,
};

/// The tensor-parallel slices and peer hooks of a graph split across ranks.
///
/// A `null` cluster on `Model` is the single-device graph I-013 gates, and every
/// branch that reads this field is skipped. A non-null cluster means this rank
/// owns an output-row slice of the projections that are tensor parallel, a
/// range of the experts and a vocabulary slice, and it must exchange them with
/// its peers through `model/multi.zig`.
///
/// **Why column parallel and gathered, not row parallel and reduced: it is a
/// bandwidth decision.** At the measured 7.2 GB/s PCIe (M-001, cold tier) one
/// gathered boundary costs 3/4 of an activation per rank - 3,840 B for a
/// 2,560-wide bf16 stream - against 7,680 B for a ring all-reduce of the same
/// boundary. Over 48 layers that is ~32 us/token gathered against ~51-102
/// us/token reduced for the same activations. The price is redundant compute:
/// the shared expert's down projection would be row parallel and is instead
/// run whole on every rank - 3 extra copies, ~53 ms per 48-layer 16K chunk at
/// tensor-core rates (indicative; two lanes hold the GPUs) against ~629 ms of
/// PCIe saved per chunk. At the measured FFMA shared-expert rate the
/// redundancy would cost ~1.3 s per chunk and flip the decision, which is why
/// moving that block off FFMA is a registered lever (plan.md 7c, L6). I-030
/// registers the reduction kernel and the all-reduce path this tree lacks.
///
/// A property worth recording for a bisect, but not the reason: every slice is
/// a complete dot product over the full contraction, so a gathered slice equals
/// the same rows of the single-device product. That is a consequence of the
/// split, not its justification.
pub const Cluster = struct {
    /// The rank's hook state, opaque to the graph.
    user: *anyopaque,
    /// This rank's first output row and row count of the GDN `out_proj`.
    dense_first: u32,
    dense_rows: u32,
    /// This rank's first intermediate column and column count of the shared
    /// expert's gate and up projections.
    shared_first: u32,
    shared_rows: u32,
    /// One weight shard per layer, `[2 * shared_rows, hidden]`: this rank's
    /// gate rows followed by this rank's up rows, in the gate-up layout
    /// `moe_silu_mul` reads.
    shared_gate_up: []const load.Matrix,
    /// This rank's first logit row and row count of `lm_head`.
    head_first: u32,
    head_rows: u32,
    /// The half-open expert range this rank owns.
    expert_first: u32,
    expert_count: u32,
    /// Scratch the graph writes and the hooks gather out of.
    dense_slice: cuda_context.DeviceBuffer,
    shared_gate_up_slice: cuda_context.DeviceBuffer,
    shared_act_slice: cuda_context.DeviceBuffer,
    head_slice: cuda_context.DeviceBuffer,
    /// Copy every rank's slice into `target`'s column band, then return with
    /// the copy complete. Called with this rank's compute stream current.
    gather_dense: *const fn (
        user: *anyopaque,
        target: *const cuda_context.DeviceBuffer,
        tokens: u32,
        columns: u32,
    ) anyerror!void,
    gather_shared: *const fn (
        user: *anyopaque,
        target: *const cuda_context.DeviceBuffer,
        tokens: u32,
        columns: u32,
    ) anyerror!void,
    gather_head: *const fn (
        user: *anyopaque,
        target: *const cuda_context.DeviceBuffer,
        tokens: u32,
        columns: u32,
    ) anyerror!void,
    /// Fetches the experts `selected` names that this rank owns into their
    /// device slots. Called before the layer's expert products run.
    prepare_experts: *const fn (
        user: *anyopaque,
        layer: u32,
        selected: []const u32,
    ) anyerror!void,
    /// The weights of an expert this rank owns, from the tier's device slot.
    expert_weights: *const fn (
        user: *anyopaque,
        layer: u32,
        expert: u32,
    ) anyerror!ExpertSlice,
    /// Copies every rank's expert range of `moe_expert_out` into `target`.
    gather_experts: *const fn (
        user: *anyopaque,
        target: *const cuda_context.DeviceBuffer,
        tokens: u32,
        hidden: u32,
    ) anyerror!void,

    /// Whether `expert` belongs to this rank's shard.
    pub fn ownsExpert(self: *const Cluster, expert: u32) bool {
        assert(expert < std.math.maxInt(u32));
        assert(self.expert_count > 0);

        return expert >= self.expert_first and
            expert < self.expert_first + self.expert_count;
    }
};

/// The whole text model on one device.
pub const Model = struct {
    allocator: std.mem.Allocator,
    config: fixtures.RefConfig,
    plan: plan_module.Plan,
    context: cuda_context.Context,
    weights: load.Weights,
    state: state_module.State,
    glue_kernels: glue.Kernels,
    elementwise_kernels: elementwise.Kernels,
    gdn_kernels: gdn.Kernels,
    moe_kernels: moe.Kernels,
    attention: attention.Attention,
    ple: ple.Ple,
    table: ple.Table,
    pool: *ple.GatherPool,
    ple_block: ple.Block,
    head_table: ple.HeadTable,
    id_stream: ple.IdStream,
    spec: ple.HashSpec,
    /// i32 `[3]`: the mRoPE section, as the table builder reads it.
    rope_section: cuda_context.DeviceBuffer,
    /// int32 `[arena]`: the text-only position grid, shared by all three axes.
    rope_grid: cuda_context.DeviceBuffer,
    /// Host scratch: the embedding gather and the multi-stream residual.
    embed_host: []f32,
    hyper_host: []f32,
    /// Host scratch: the n-gram ids of one forward.
    ngram_ids: []i64,
    /// Host scratch: the absolute positions of one forward.
    positions_host: []i32,
    /// Host scratch: the sequence a recompute-form step forwards.
    sequence: []i64,
    /// Host scratch: the router's selected expert ids of one forward, read
    /// back so the expert tier can prefetch them.
    moe_selected: []u32,
    /// Non-null when the graph runs split across ranks; see `Cluster`.
    cluster: ?Cluster = null,
    timings: Timings,

    /// Opens device 0, loads the checkpoint and builds the caches.
    pub fn init(allocator: std.mem.Allocator) !Model {
        return initOnDevice(allocator, 0);
    }

    /// Opens one device by index: one rank's copy of the graph. Every buffer,
    /// module and stream this model owns belongs to that device's context.
    pub fn initOnDevice(allocator: std.mem.Allocator, device_index: u32) !Model {
        assert(device_index < cuda_driver.device_count_max);

        const config = try fixtures.load_config(allocator);
        const plan = try plan_module.Plan.init(config);

        var devices: [cuda_driver.device_count_max]cuda_driver.Device = undefined;
        const count = try cuda_driver.query_devices(devices[0..]);
        if (count == 0) {
            return error.NoDevice;
        }
        if (device_index >= count) {
            return error.NoDevice;
        }
        const context = try cuda_context.Context.retain(&devices[device_index]);
        errdefer context.release();
        try context.set_current();

        var model = Model{
            .allocator = allocator,
            .config = config,
            .plan = plan,
            .context = context,
            .weights = undefined,
            .state = undefined,
            .glue_kernels = undefined,
            .elementwise_kernels = undefined,
            .gdn_kernels = undefined,
            .moe_kernels = undefined,
            .attention = undefined,
            .ple = undefined,
            .table = undefined,
            .pool = undefined,
            .ple_block = undefined,
            .head_table = undefined,
            .id_stream = undefined,
            .spec = undefined,
            .rope_section = undefined,
            .rope_grid = undefined,
            .embed_host = &[_]f32{},
            .hyper_host = &[_]f32{},
            .ngram_ids = &[_]i64{},
            .positions_host = &[_]i32{},
            .sequence = &[_]i64{},
            .moe_selected = &[_]u32{},
            .cluster = null,
            .timings = .{},
        };
        errdefer model.release();

        model.ple = try ple.Ple.init();
        errdefer model.ple.deinit();
        model.weights = try load.Weights.load(
            allocator,
            &model.ple.stream,
            &config,
        );
        errdefer model.weights.deinit();
        model.state = try state_module.State.init(
            allocator,
            &model.ple.stream,
            config,
            plan,
        );
        errdefer model.state.deinit();
        model.glue_kernels = try glue.Kernels.load();
        errdefer model.glue_kernels.deinit();
        model.elementwise_kernels = try elementwise.Kernels.load();
        errdefer model.elementwise_kernels.deinit();
        model.gdn_kernels = try gdn.Kernels.load();
        errdefer model.gdn_kernels.deinit();
        model.moe_kernels = try moe.Kernels.init(try moe.cubinBytes());
        errdefer model.moe_kernels.deinit();
        model.attention = try attention.Attention.init();
        errdefer model.attention.deinit();
        try model.allocateRopeTables();
        try model.openPle();
        try model.allocateHostScratch();
        assert(model.state.length == 0);
        return model;
    }

    /// Releases every module, cache and host allocation.
    pub fn deinit(self: *Model) void {
        assert(self.config.num_hidden_layers > 0);
        assert(self.embed_host.len > 0);

        self.release();
        self.context.release();
    }

    /// Releases in reverse construction order; safe on a partial model.
    fn release(self: *Model) void {
        assert(self.allocator.vtable != undefined);

        if (self.moe_selected.len > 0) {
            self.allocator.free(self.moe_selected);
            self.moe_selected = &[_]u32{};
        }
        if (self.sequence.len > 0) {
            self.allocator.free(self.sequence);
            self.sequence = &[_]i64{};
        }
        if (self.positions_host.len > 0) {
            self.allocator.free(self.positions_host);
            self.positions_host = &[_]i32{};
        }
        if (self.ngram_ids.len > 0) {
            self.allocator.free(self.ngram_ids);
            self.ngram_ids = &[_]i64{};
        }
        if (self.hyper_host.len > 0) {
            self.allocator.free(self.hyper_host);
            self.hyper_host = &[_]f32{};
        }
        if (self.embed_host.len > 0) {
            self.allocator.free(self.embed_host);
            self.embed_host = &[_]f32{};
        }
        if (self.rope_grid.pointer != 0) {
            self.rope_grid.free();
        }
        if (self.rope_section.pointer != 0) {
            self.rope_section.free();
        }
        self.ple_block.deinit();
        self.pool.destroy(self.allocator);
        self.table.close();
        self.attention.deinit();
        self.moe_kernels.deinit();
        self.gdn_kernels.deinit();
        self.elementwise_kernels.deinit();
        self.glue_kernels.deinit();
        self.state.deinit();
        self.weights.deinit();
        self.ple.deinit();
    }

    /// Allocates the mRoPE buffers and fills the cos/sin tables.
    fn allocateRopeTables(self: *Model) !void {
        assert(self.config.rotary_dim() > 0);
        assert(self.state.arena_length > 0);

        const arena = self.state.arena_length;
        self.rope_section = try cuda_context.DeviceBuffer.allocate(4096);
        errdefer self.rope_section.free();
        self.rope_grid = try cuda_context.DeviceBuffer.allocate(
            state_module.allocationBytes(@as(u64, arena) * 4),
        );
        errdefer self.rope_grid.free();
        try self.fillRopeTables();
        assert(self.rope_grid.bytes >= @as(u64, arena) * 4);
    }

    /// (Re)builds the mRoPE cos/sin tables for every position the arena holds.
    ///
    /// The tables are a function of the config and the position grid, not of
    /// the tokens a sequence has seen, so this is re-run whenever the buffers
    /// holding them are cleared — `State.reset` zeroes every per-sequence
    /// allocation, these two included, and a zeroed table would rotate every
    /// query and key of every later attention layer to zero.
    fn fillRopeTables(self: *Model) !void {
        assert(self.config.rotary_dim() > 0);
        assert(self.state.arena_length > 0);

        const arena = self.state.arena_length;
        var section = [_]i32{
            @intCast(self.config.mrope_section[0]),
            @intCast(self.config.mrope_section[1]),
            @intCast(self.config.mrope_section[2]),
        };
        try self.state.uploadI32(&self.rope_section, section[0..]);
        const positions = try self.allocator.alloc(i32, arena);
        defer self.allocator.free(positions);
        for (positions, 0..) |*slot, index| {
            slot.* = @intCast(index);
        }
        try self.state.uploadI32(&self.rope_grid, positions);
        try self.attention.buildRopeTables(.{
            .cos = self.state.rope_cos,
            .sin = self.state.rope_sin,
            .grids = .{
                self.rope_grid.pointer,
                self.rope_grid.pointer,
                self.rope_grid.pointer,
            },
            .section = self.rope_section.pointer,
            .rows = arena,
            .rotary_dim = self.config.rotary_dim(),
            .theta = self.config.rope_theta,
        });
        try self.attention.stream.synchronize();
        assert(self.state.rope_cos.bytes >= @as(u64, arena) * self.config.rotary_dim() * 4);
    }

    /// Opens the n-gram table and builds the PLE block of the PLE layer.
    fn openPle(self: *Model) !void {
        assert(self.config.ngram_size >= 2);
        assert(self.plan.layer_count > 0);

        self.spec = .{
            .ngram_size = self.config.ngram_size,
            .heads_per_ngram = self.config.heads_per_ngram,
            .eos_token_id = self.config.eos_token_id,
            .seed = self.config.seed,
            .vocab_size = self.config.vocab_size,
            .ngram_vocab_size_base = self.config.ngram_vocab_size_base,
            .make_ngram_vocab_size_divisible_by = self.config.make_ngram_vocab_size_divisible_by,
            .split_ngram_parts = self.config.split_ngram_parts,
            .ple_layer_index = 0,
        };
        try self.spec.validate();
        self.head_table = try ple.deriveHeadTable(&self.spec);
        self.id_stream = ple.IdStream.init(&self.spec);
        const path = try load.checkpointPath(self.allocator);
        defer self.allocator.free(path);
        self.table = try ple.Table.open(
            path,
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_",
            self.config.split_ngram_parts,
            self.config.ngram_head_dim(),
            .f32,
        );
        self.pool = try ple.GatherPool.create(self.allocator, &self.table, 4);
        const weights = self.weights.layers[self.plan.ple_layer].ple.?;
        self.ple_block = try ple.Block.init(
            &self.ple,
            &self.table,
            self.pool,
            .{
                .key_proj = weights.key_proj.data.pointer,
                .value_proj = weights.value_proj.data.pointer,
                .norm_key = weights.norm_key.pointer,
                .norm_query = weights.norm_query.pointer,
                .norm_conv = weights.norm_conv.pointer,
                .taps = weights.conv.pointer,
                .table_format = .f32,
                .table_scale = 1.0,
            },
            .{
                .ple_embed_dim = self.config.ple_embed_dim,
                .hidden_size = self.config.hidden_size,
                .hc_count = self.config.hc_count,
                .ngram_heads = self.config.ngram_heads(),
                .head_dim = self.config.ngram_head_dim(),
                .conv_kernel = self.config.ple_conv_kernel_size,
                .conv_dilation = self.config.ngram_size,
                .eps = self.config.rms_norm_eps,
            },
            state_module.tokens_max,
        );
        assert(self.head_table.head_count == self.config.ngram_heads());
    }

    /// Allocates the host scratch a forward writes into.
    fn allocateHostScratch(self: *Model) !void {
        assert(state_module.tokens_max > 0);
        assert(self.config.hidden_size > 0);

        const tokens: usize = state_module.tokens_max;
        self.embed_host = try self.allocator.alloc(
            f32,
            tokens * self.config.hidden_size,
        );
        errdefer self.allocator.free(self.embed_host);
        self.hyper_host = try self.allocator.alloc(
            f32,
            tokens * self.config.hyper_size(),
        );
        errdefer self.allocator.free(self.hyper_host);
        self.ngram_ids = try self.allocator.alloc(
            i64,
            tokens * self.config.ngram_heads(),
        );
        errdefer self.allocator.free(self.ngram_ids);
        self.positions_host = try self.allocator.alloc(i32, tokens);
        errdefer self.allocator.free(self.positions_host);
        self.sequence = try self.allocator.alloc(
            i64,
            self.config.max_position_embeddings,
        );
        self.moe_selected = try self.allocator.alloc(
            u32,
            tokens * self.config.num_experts_per_tok,
        );
        assert(self.sequence.len >= tokens);
    }

    /// Forgets every cached token: caches, the PLE conv state and the n-gram
    /// history all return to their start-of-sequence state, and the mRoPE
    /// tables — which `State.reset` clears with everything else — are rebuilt.
    pub fn reset(self: *Model) !void {
        assert(self.state.length >= 0);
        assert(self.ple_block.capacity_tokens > 0);

        try self.state.reset();
        try self.fillRopeTables();
        try self.ple_block.reset();
        self.id_stream.reset();
    }

    /// One forward over `ids`, writing the logits of every row.
    pub fn forward(self: *Model, ids: []const i64) !void {
        var timer = try std.time.Timer.start();
        const tokens: u32 = @intCast(ids.len);
        assert(tokens > 0);
        assert(tokens <= state_module.tokens_max);
        const past = self.state.length;
        assert(past + tokens <= self.config.max_position_embeddings);
        assert(self.state.length == past);

        try self.loadEmbedding(ids, tokens);
        self.timings.embed_ns += timer.lap();
        try self.forwardLayers(ids, tokens, past);
        try self.forwardHead(tokens);
        if (guards.enabled()) {
            try self.state.verifyGuards();
        } else {
            assert(!guards.enabled());
        }
        self.timings.total_ns += timer.read();
        self.state.length = past + tokens;
        assert(self.state.length == past + tokens);
    }

    /// Gathers the embedding rows and repeats them into every residual stream.
    fn loadEmbedding(self: *Model, ids: []const i64, tokens: u32) !void {
        assert(ids.len == tokens);
        assert(self.embed_host.len >= tokens * self.config.hidden_size);

        const hidden: usize = self.config.hidden_size;
        const hyper: usize = self.config.hyper_size();
        const streams: usize = self.config.hc_count;
        for (ids, 0..) |id, token| {
            assert(id >= 0);
            assert(id < self.config.vocab_size);
            const row = self.weights.embed_tokens[@as(usize, @intCast(id)) * hidden ..][0..hidden];
            @memcpy(self.embed_host[token * hidden ..][0..hidden], row);
        }
        for (0..tokens) |token| {
            for (0..streams) |stream| {
                const target = token * hyper + stream * hidden;
                @memcpy(
                    self.hyper_host[target..][0..hidden],
                    self.embed_host[token * hidden ..][0..hidden],
                );
            }
        }
        try self.state.uploadF32(
            &self.state.hyper,
            self.hyper_host[0 .. tokens * hyper],
        );
        assert(self.hyper_host.len >= tokens * hyper);
    }

    /// Walks the plan: every decoder layer, then the final mixer.
    fn forwardLayers(self: *Model, ids: []const i64, tokens: u32, past: u32) !void {
        assert(ids.len == tokens);
        assert(tokens > 0);

        try self.uploadPositions(tokens, past);
        const recurrent_form = tokens == 1 and past > 0;
        for (0..self.plan.layer_count) |index| {
            try self.forwardLayer(@intCast(index), ids, tokens, past, recurrent_form);
            // Debug only (`INFERNO_GUARD_BANDS`): a kernel that wrote past an
            // allocation tears that allocation's canary, and the first layer
            // to do it names the buffer instead of corrupting a later one.
            if (guards.enabled()) {
                try self.state.verifyGuards();
            } else {
                assert(!guards.enabled());
            }
        }
        try self.forwardMixer(tokens);
        assert(self.state.hyper.bytes > 0);
    }

    /// Uploads the absolute positions of this forward.
    fn uploadPositions(self: *Model, tokens: u32, past: u32) !void {
        assert(tokens > 0);
        assert(past + tokens <= self.config.max_position_embeddings);

        assert(self.positions_host.len >= tokens);
        for (self.positions_host[0..tokens], 0..) |*slot, index| {
            slot.* = @intCast(past + @as(u32, @intCast(index)));
        }
        try self.state.uploadI32(
            &self.state.positions,
            self.positions_host[0..tokens],
        );
        assert(self.state.positions.bytes >= @as(u64, tokens) * 4);
    }

    /// One decoder layer: PLE, hyper-connection + block + injection, twice.
    fn forwardLayer(
        self: *Model,
        index: u32,
        ids: []const i64,
        tokens: u32,
        past: u32,
        recurrent_form: bool,
    ) !void {
        assert(index < self.plan.layer_count);
        assert(ids.len == tokens);

        const layer_plan = self.plan.layer(index);
        const weights = &self.weights.layers[index];
        if (layer_plan.has_ple) {
            try self.runPle(ids, tokens);
        } else {
            assert(weights.ple == null);
        }
        try self.hyperMix(&weights.attn_hyper, tokens, &self.state.stream_in);
        switch (layer_plan.kind) {
            .linear_attention => try self.gdnBlock(index, tokens, recurrent_form),
            .indexed_attention => try self.attentionBlock(index, tokens, past),
        }
        try self.inject(tokens);
        try self.hyperMix(&weights.mlp_hyper, tokens, &self.state.stream_in);
        try self.moeBlock(index, tokens);
        try self.inject(tokens);
        assert(layer_plan.kind == self.plan.layer(index).kind);
    }

    /// Runs the PLE block: it adds its output into the residual itself.
    fn runPle(self: *Model, ids: []const i64, tokens: u32) !void {
        assert(tokens > 0);
        assert(self.head_table.head_count == self.config.ngram_heads());

        const heads: usize = self.config.ngram_heads();
        assert(self.ngram_ids.len >= tokens * heads);
        self.id_stream.appendAll(
            &self.head_table,
            ids,
            self.ngram_ids[0 .. tokens * heads],
        );
        try self.ple_block.run(
            self.ngram_ids[0 .. tokens * heads],
            tokens,
            self.state.hyper.pointer,
            self.state.hyper.pointer,
        );
        assert(self.id_stream.count >= tokens);
    }

    /// The hyper-connection mix of the residual into `mixed`.
    fn hyperMix(
        self: *Model,
        weights: *const load.HcWeights,
        tokens: u32,
        mixed: *const cuda_context.DeviceBuffer,
    ) !void {
        assert(tokens > 0);
        assert(weights.mix_down.rows == self.config.hc_lowrank);

        const inject_weight = if (weights.inject) |matrix| matrix.data.pointer else 0;
        const injection = if (weights.inject != null)
            self.state.injection.pointer
        else
            0;
        try elementwise.launchHyperMix(
            &self.elementwise_kernels,
            self.ple.stream.handle,
            .{
                .input = self.state.hyper.pointer,
                .norm_weight = weights.norm.pointer,
                .mix_down = weights.mix_down.data.pointer,
                .mix_up = weights.mix_up.data.pointer,
                .inject_weight = inject_weight,
                .mixed = mixed.pointer,
                .injection = injection,
            },
            .{
                .tokens = tokens,
                .hidden = self.config.hidden_size,
                .streams = self.config.hc_count,
                .lowrank = self.config.hc_lowrank,
            },
            self.config.rms_norm_eps,
        );
    }

    /// Adds the block output back into every residual stream.
    fn inject(self: *Model, tokens: u32) !void {
        assert(tokens > 0);
        assert(self.state.injection.bytes >=
            @as(u64, tokens) * self.config.hc_count * 4);

        try glue.launchInject(
            &self.glue_kernels,
            self.ple.stream.handle,
            self.state.stream_out.pointer,
            self.state.injection.pointer,
            self.state.hyper.pointer,
            tokens,
            self.config.hidden_size,
            self.config.hc_count,
        );
    }

    /// The final mixer before `lm_head`: no injection weights.
    fn forwardMixer(self: *Model, tokens: u32) !void {
        assert(tokens > 0);
        assert(self.weights.mixer.inject == null);

        try self.hyperMix(&self.weights.mixer, tokens, &self.state.mixed);
    }

    /// `lm_head` and the device-side argmax of the last row.
    fn forwardHead(self: *Model, tokens: u32) !void {
        assert(tokens > 0);
        assert(self.weights.lm_head.rows == self.config.vocab_size);

        var timer = try std.time.Timer.start();
        if (self.cluster) |cluster| {
            // Row split by vocabulary. The fixture's 256-wide head is gathered
            // whole because it is 1 KiB; at the production vocabulary that
            // gather would be ~745 KB/token/rank and is NOT the production
            // path - the row-split head needs the on-device top-k reduction
            // registered in I-030.
            assert(cluster.head_rows > 0);
            assert(self.config.vocab_size % cluster.head_rows == 0);
            assert(cluster.head_first + cluster.head_rows <= self.config.vocab_size);
            const hidden: u64 = self.config.hidden_size;
            const sliced = load.Matrix{
                .data = state_module.view(
                    &self.weights.lm_head.data,
                    @as(u64, cluster.head_first) * hidden * 4,
                    @as(u64, cluster.head_rows) * hidden * 4,
                ),
                .rows = cluster.head_rows,
                .cols = self.config.hidden_size,
            };
            try self.project(
                &self.state.mixed,
                sliced,
                &cluster.head_slice,
                tokens,
            );
            try cluster.gather_head(
                cluster.user,
                &self.state.logits,
                tokens,
                self.config.vocab_size,
            );
        } else {
            try self.project(
                &self.state.mixed,
                self.weights.lm_head,
                &self.state.logits,
                tokens,
            );
        }
        const vocab: u64 = self.config.vocab_size;
        const last = state_module.view(
            &self.state.logits,
            @as(u64, tokens - 1) * vocab * 4,
            vocab * 4,
        );
        try elementwise.launchArgmax(
            &self.elementwise_kernels,
            self.ple.stream.handle,
            last.pointer,
            self.state.argmax_index.pointer,
            self.state.argmax_value.pointer,
            self.config.vocab_size,
        );
        self.timings.head_ns += timer.lap();
        assert(last.bytes == vocab * 4);
    }

    /// `out = x @ weight^T` through the f32 matmul the PLE module carries.
    fn project(
        self: *Model,
        x: *const cuda_context.DeviceBuffer,
        weight: load.Matrix,
        out: *const cuda_context.DeviceBuffer,
        rows: u32,
    ) !void {
        assert(rows > 0);
        assert(weight.rows > 0);
        assert(weight.cols > 0);
        assert(weight.cols * @as(u64, rows) * 4 <= x.bytes);
        assert(weight.rows * @as(u64, rows) * 4 <= out.bytes);

        try self.ple.project(.{
            .x = x.pointer,
            .weight = weight.data.pointer,
            .out = out.pointer,
            .rows = rows,
            .in_features = weight.cols,
            .out_features = weight.rows,
        });
    }

    /// The routed MoE block over `stream_in`, writing `stream_out`.
    fn moeBlock(self: *Model, index: u32, tokens: u32) !void {
        assert(index < self.plan.layer_count);
        assert(tokens > 0);

        var timer = try std.time.Timer.start();
        const weights = &self.weights.layers[index].moe;
        try self.project(
            &self.state.stream_in,
            weights.router,
            &self.state.moe_logits,
            tokens,
        );
        try moe.launchRouterTopk(
            &self.moe_kernels,
            &self.ple.stream,
            &self.state.moe_logits,
            &self.state.moe_indices,
            &self.state.moe_weights,
            tokens,
            self.config.num_experts,
            self.config.num_experts,
            self.config.num_experts_per_tok,
        );
        try self.sharedExpert(index, weights, tokens);
        if (self.cluster) |cluster| {
            // Expert parallel: this rank computes the experts of its shard that
            // the router selected, from the bytes the expert tier placed, and
            // every rank gathers all shards' slots before combining. The
            // combine reads only the selected slots, so an unselected slot's
            // staleness is unobservable, and no partial sum is added: each
            // selected slot is computed exactly once, by its owner.
            const selected = try self.selectedExperts(tokens);
            try cluster.prepare_experts(cluster.user, index, selected);
            var local: u32 = 0;
            while (local < cluster.expert_count) : (local += 1) {
                const expert = cluster.expert_first + local;
                assert(expert < self.config.num_experts);
                if (selectedExpert(selected, expert)) {
                    const slice = try cluster.expert_weights(
                        cluster.user,
                        index,
                        expert,
                    );
                    try self.routedExpertMatrices(
                        slice.gate_up,
                        slice.down,
                        expert,
                        tokens,
                    );
                } else {
                    assert(!selectedExpert(selected, expert));
                }
            }
            try cluster.gather_experts(
                cluster.user,
                &self.state.moe_expert_out,
                tokens,
                self.config.hidden_size,
            );
        } else {
            for (0..self.config.num_experts) |expert| {
                try self.routedExpert(weights, @intCast(expert), tokens);
            }
        }
        try glue.launchMoeCombine(
            &self.glue_kernels,
            self.ple.stream.handle,
            self.state.moe_expert_out.pointer,
            self.state.moe_indices.pointer,
            self.state.moe_weights.pointer,
            self.state.moe_shared_out.pointer,
            weights.shared_gate_logit.pointer,
            self.state.stream_in.pointer,
            self.state.stream_out.pointer,
            tokens,
            self.config.hidden_size,
            self.config.num_experts_per_tok,
        );
        self.timings.moe_ns += timer.lap();
    }

    /// The sigmoid-gated shared expert: `shared_out` without the gate.
    ///
    /// Split across ranks this is column parallel over the intermediate width:
    /// each rank computes its gate and up rows over the full hidden (so each
    /// product is complete), the silu product of its own columns, and the
    /// gather rebuilds the whole activation before the shared down projection
    /// runs. The down projection therefore reads exactly the single-device
    /// activation, in the same column order.
    fn sharedExpert(
        self: *Model,
        index: u32,
        weights: *const load.MoeWeights,
        tokens: u32,
    ) !void {
        assert(index < self.plan.layer_count);
        assert(tokens > 0);
        assert(weights.shared_gate_up.rows ==
            2 * self.config.shared_expert_intermediate_size);

        if (self.cluster) |cluster| {
            assert(index < cluster.shared_gate_up.len);
            assert(cluster.shared_rows > 0);
            assert(self.config.shared_expert_intermediate_size %
                cluster.shared_rows == 0);
            assert(cluster.shared_first + cluster.shared_rows <=
                self.config.shared_expert_intermediate_size);
            const shard = cluster.shared_gate_up[index];
            try self.project(
                &self.state.stream_in,
                shard,
                &cluster.shared_gate_up_slice,
                tokens,
            );
            try moe.launchSiluMul(
                &self.moe_kernels,
                &self.ple.stream,
                &cluster.shared_gate_up_slice,
                tokens,
                cluster.shared_rows,
                &cluster.shared_act_slice,
            );
            try cluster.gather_shared(
                cluster.user,
                &self.state.moe_shared_act,
                tokens,
                self.config.shared_expert_intermediate_size,
            );
            try self.project(
                &self.state.moe_shared_act,
                weights.shared_down,
                &self.state.moe_shared_out,
                tokens,
            );
        } else {
            try self.project(
                &self.state.stream_in,
                weights.shared_gate_up,
                &self.state.moe_shared_gate_up,
                tokens,
            );
            try moe.launchSiluMul(
                &self.moe_kernels,
                &self.ple.stream,
                &self.state.moe_shared_gate_up,
                tokens,
                self.config.shared_expert_intermediate_size,
                &self.state.moe_shared_act,
            );
            try self.project(
                &self.state.moe_shared_act,
                weights.shared_down,
                &self.state.moe_shared_out,
                tokens,
            );
        }
        assert(self.state.moe_shared_out.bytes >=
            @as(u64, tokens) * self.config.hidden_size * 4);
    }

    /// The router's selected expert ids of this forward, read back to the host.
    ///
    /// The ids are what the expert tier prefetches on; the combine still reads
    /// them on the device, so this copy exists for the placement decision, not
    /// for the arithmetic.
    fn selectedExperts(self: *Model, tokens: u32) ![]const u32 {
        assert(tokens > 0);
        assert(self.cluster != null);
        assert(self.config.num_experts_per_tok > 0);

        const count: usize = @as(usize, tokens) *
            self.config.num_experts_per_tok;
        assert(self.moe_selected.len >= count);
        try self.state.downloadRaw(
            &self.state.moe_indices,
            std.mem.sliceAsBytes(self.moe_selected[0..count]),
        );
        for (self.moe_selected[0..count]) |expert| {
            assert(expert < self.config.num_experts);
        }
        return self.moe_selected[0..count];
    }

    /// One routed expert, from the checkpoint's own matrices.
    fn routedExpert(
        self: *Model,
        weights: *const load.MoeWeights,
        expert: u32,
        tokens: u32,
    ) !void {
        assert(expert < self.config.num_experts);
        assert(tokens > 0);

        try self.routedExpertMatrices(
            weights.expert_gate_up[expert],
            weights.expert_down[expert],
            expert,
            tokens,
        );
    }

    /// One routed expert, written into its slot of the `[expert][token]`
    /// combine buffer. The engine runs every expert densely on one device: the
    /// router's weights are what select, and `moe_combine` applies them, so the
    /// sum is the reference's slot-order sum even when two experts would have
    /// been dispatched. Split across ranks the matrices come from the expert
    /// tier's device slot instead, but the arithmetic is the same one.
    fn routedExpertMatrices(
        self: *Model,
        gate_up: load.Matrix,
        down: load.Matrix,
        expert: u32,
        tokens: u32,
    ) !void {
        assert(expert < self.config.num_experts);
        assert(tokens > 0);
        assert(gate_up.rows == 2 * self.config.moe_intermediate_size);
        assert(down.rows == self.config.hidden_size);
        assert(down.cols == self.config.moe_intermediate_size);

        const hidden: u64 = self.config.hidden_size;
        // `model_moe_combine` reads `[experts][token][hidden]` packed by this
        // forward's token count, so the slot stride is `tokens`, not the
        // buffer's `tokens_max` capacity: a wider stride would make the
        // combine read another expert's (or an earlier forward's) rows for
        // every expert but the first.
        const offset = @as(u64, expert) * tokens * hidden * 4;
        const slot = state_module.view(
            &self.state.moe_expert_out,
            offset,
            @as(u64, tokens) * hidden * 4,
        );
        try self.project(
            &self.state.stream_in,
            gate_up,
            &self.state.moe_gate_up,
            tokens,
        );
        try moe.launchSiluMul(
            &self.moe_kernels,
            &self.ple.stream,
            &self.state.moe_gate_up,
            tokens,
            self.config.moe_intermediate_size,
            &self.state.moe_act,
        );
        try self.project(&self.state.moe_act, down, &slot, tokens);
        assert(self.state.moe_expert_out.bytes >= offset + tokens * hidden * 4);
    }

    /// The indexed-attention block over `stream_in`, writing `stream_out`.
    fn attentionBlock(self: *Model, index: u32, tokens: u32, past: u32) !void {
        assert(index < self.plan.layer_count);
        assert(tokens > 0);

        var timer = try std.time.Timer.start();
        const weights = &self.weights.layers[index].attention.?;
        const cache = &self.state.layers[index].attention.?;
        const heads: u32 = self.config.num_attention_heads;
        const head_dim: u32 = self.config.head_dim;
        const width: u32 = heads * head_dim;
        const kv_dim: u32 = self.config.num_key_value_heads * head_dim;

        try self.project(&self.state.stream_in, weights.q_only, &self.state.attn_query, tokens);
        try self.project(&self.state.stream_in, weights.q_gate, &self.state.attn_gate, tokens);
        try self.project(&self.state.stream_in, weights.k_proj, &self.state.attn_key, tokens);
        try self.project(&self.state.stream_in, weights.v_proj, &self.state.attn_value, tokens);
        try elementwise.launchRmsnorm(
            &self.elementwise_kernels,
            self.ple.stream.handle,
            .{
                .x = self.state.attn_query.pointer,
                .weight = weights.q_norm.pointer,
                .out = self.state.attn_out_f32.pointer,
            },
            .{ .rows = tokens, .columns = width, .group = head_dim },
            self.config.rms_norm_eps,
        );
        try elementwise.launchRmsnorm(
            &self.elementwise_kernels,
            self.ple.stream.handle,
            .{
                .x = self.state.attn_key.pointer,
                .weight = weights.k_norm.pointer,
                .out = self.state.gdn_normed_f32.pointer,
            },
            .{ .rows = tokens, .columns = kv_dim, .group = head_dim },
            self.config.rms_norm_eps,
        );
        try glue.launchCastF32Bf16(
            &self.glue_kernels,
            self.ple.stream.handle,
            self.state.attn_out_f32.pointer,
            self.state.attn_query_bf16.pointer,
            tokens * width,
        );
        try glue.launchCastF32Bf16(
            &self.glue_kernels,
            self.ple.stream.handle,
            self.state.gdn_normed_f32.pointer,
            self.state.attn_key_bf16.pointer,
            tokens * kv_dim,
        );
        try glue.launchCastF32Bf16(
            &self.glue_kernels,
            self.ple.stream.handle,
            self.state.attn_value.pointer,
            self.state.attn_value_bf16.pointer,
            tokens * kv_dim,
        );
        try self.ropeAndCache(tokens, past, cache);
        try self.indexer(tokens, past, weights, cache);
        try self.attend(tokens, past, cache);
        try self.attentionOutput(tokens, weights);
        self.timings.attention_ns += timer.lap();
    }

    /// Applies mRoPE to the normed q/k and appends k/v to the layer's cache.
    fn ropeAndCache(
        self: *Model,
        tokens: u32,
        past: u32,
        cache: *const state_module.AttentionCache,
    ) !void {
        assert(tokens > 0);
        assert(past + tokens <= self.config.max_position_embeddings);

        const head_dim: u32 = self.config.head_dim;
        const kv_dim: u32 = self.config.num_key_value_heads * head_dim;
        // The bf16 casts this block runs were queued on the executor's stream
        // and the attention module owns its own, so the hand-over needs a
        // fence: without it the rotation reads the casts' output while they
        // are still running.
        try self.ple.stream.synchronize();
        try self.attention.applyRope(.{
            .values = self.state.attn_query_bf16.pointer,
            .cos = self.state.rope_cos,
            .sin = self.state.rope_sin,
            .positions = self.state.positions.pointer,
            .rows = tokens,
            .heads = self.config.num_attention_heads,
            .head_dim = head_dim,
            .rotary_dim = self.config.rotary_dim(),
        });
        try self.attention.applyRope(.{
            .values = self.state.attn_key_bf16.pointer,
            .cos = self.state.rope_cos,
            .sin = self.state.rope_sin,
            .positions = self.state.positions.pointer,
            .rows = tokens,
            .heads = self.config.num_key_value_heads,
            .head_dim = head_dim,
            .rotary_dim = self.config.rotary_dim(),
        });
        try self.attention.stream.synchronize();
        // The keys are roped in place, so the cache append is a straight copy
        // of this step's rows into the arena's `past` offset.
        const bytes: u64 = @as(u64, tokens) * kv_dim * 2;
        const key_source = state_module.view(&self.state.attn_key_bf16, 0, bytes);
        const value_source = state_module.view(&self.state.attn_value_bf16, 0, bytes);
        const key_target = state_module.view(
            &cache.key,
            @as(u64, past) * kv_dim * 2,
            bytes,
        );
        const value_target = state_module.view(
            &cache.value,
            @as(u64, past) * kv_dim * 2,
            bytes,
        );
        try self.ple.stream.copy_d_to_d(&key_target, &key_source, bytes);
        try self.ple.stream.copy_d_to_d(&value_target, &value_source, bytes);
        assert(cache.key.bytes >= (@as(u64, past + tokens)) * kv_dim * 2);
    }

    /// Runs the indexer and, per row, the gathered attention.
    fn indexer(
        self: *Model,
        tokens: u32,
        past: u32,
        weights: *const load.AttentionWeights,
        cache: *const state_module.AttentionCache,
    ) !void {
        assert(tokens > 0);
        assert(past + tokens <= self.config.max_position_embeddings);

        const ratio: u32 = self.config.indexer_compress_ratio;
        try glue.launchCastF32Bf16(
            &self.glue_kernels,
            self.ple.stream.handle,
            self.state.stream_in.pointer,
            self.state.stream_bf16.pointer,
            tokens * self.config.hidden_size,
        );
        try self.ple.stream.synchronize();
        try self.attention.indexerScores(.{
            .hidden = self.state.stream_bf16.pointer,
            .weight = weights.index_qk_proj.pointer,
            .projected = self.state.index_projected.pointer,
            .q_layernorm = weights.q_layernorm.pointer,
            .k_layernorm = weights.k_layernorm.pointer,
            .queries = self.state.index_query.pointer,
            .index_keys = cache.index_keys.pointer,
            .pooled = cache.index_pooled.pointer,
            .scores = self.state.index_scores.pointer,
            .partial = self.state.index_partial.pointer,
            .selected = self.state.index_selected.pointer,
            .positions = self.state.positions.pointer,
            .cos = self.state.rope_cos.pointer,
            .sin = self.state.rope_sin.pointer,
            .rows = tokens,
            .in_features = self.config.hidden_size,
            .arena_length = past + tokens,
            .first_block = past / ratio,
            .shapes = self.attentionShapes(),
        });
        try self.attention.stream.synchronize();
        assert(self.state.index_scores.bytes > 0);
    }

    /// Attention for every row of this step: gather the row's selection out of
    /// the KV cache and run one `attn_prefill` over it.
    ///
    /// The gathers run on the attention module's queue, not the executor's,
    /// because their output is the *next* row's input: `gathered_key` and
    /// `gathered_value` are one buffer pair, so row `r + 1`'s gather overwrites
    /// what row `r`'s prefill is reading. On one queue the next gather cannot
    /// start before this row's prefill finished, and the loop needs no host
    /// fence at all. The single fence before it is the row-independent
    /// hand-over: the KV append the executor queued, which every row reads.
    fn attend(
        self: *Model,
        tokens: u32,
        past: u32,
        cache: *const state_module.AttentionCache,
    ) !void {
        assert(tokens > 0);
        assert(self.attentionShapes().slots() > 0);

        const head_dim: u32 = self.config.head_dim;
        const kv_dim: u32 = self.config.num_key_value_heads * head_dim;
        const width: u32 = self.config.num_attention_heads * head_dim;
        const slots: u32 = self.attentionShapes().slots();
        const ratio: u32 = self.config.indexer_compress_ratio;
        const block_topk: u32 = self.config.indexer_block_topk();
        // The KV rows this loop gathers were appended on the executor's queue.
        try self.ple.stream.synchronize();
        for (0..tokens) |row| {
            const visible: u32 = past + @as(u32, @intCast(row)) + 1;
            const count = selectionCount(visible, ratio, block_topk);
            assert(count > 0);
            assert(count <= slots);
            const index = state_module.view(
                &self.state.index_selected,
                @as(u64, row) * slots * 4,
                @as(u64, count) * 4,
            );
            try self.gatherRow(cache, index, count, kv_dim);
            try self.attention.prefill(.{
                .queries = state_module.view(
                    &self.state.attn_query_bf16,
                    @as(u64, row) * width * 2,
                    @as(u64, width) * 2,
                ).pointer,
                .gathered_k = self.state.gathered_key.pointer,
                .gathered_v = self.state.gathered_value.pointer,
                .gather_index = index.pointer,
                .positions = state_module.view(
                    &self.state.positions,
                    @as(u64, row) * 4,
                    4,
                ).pointer,
                .out = state_module.view(
                    &self.state.attn_out,
                    @as(u64, row) * width * 2,
                    @as(u64, width) * 2,
                ).pointer,
                .violations = self.state.violations.pointer,
                .rows = 1,
                .count = count,
                .shapes = self.attentionShapes(),
            });
        }
        try self.attention.stream.synchronize();
        assert(self.state.gathered_key.bytes >= @as(u64, slots) * kv_dim * 2);
    }

    /// Gathers one row's selected keys and values into the shared gather
    /// buffers the row's prefill reads. The two launches share the attention
    /// queue with that prefill and with the next row's gather, which is what
    /// orders the overwrite after the last read.
    fn gatherRow(
        self: *Model,
        cache: *const state_module.AttentionCache,
        index: cuda_context.DeviceBuffer,
        count: u32,
        kv_dim: u32,
    ) !void {
        assert(count > 0);
        assert(kv_dim > 0);
        assert(index.bytes >= @as(u64, count) * 4);

        try glue.launchGatherBf16(
            &self.glue_kernels,
            self.attention.stream.handle,
            cache.key.pointer,
            index.pointer,
            count,
            kv_dim,
            self.state.gathered_key.pointer,
        );
        try glue.launchGatherBf16(
            &self.glue_kernels,
            self.attention.stream.handle,
            cache.value.pointer,
            index.pointer,
            count,
            kv_dim,
            self.state.gathered_value.pointer,
        );
        assert(self.state.gathered_key.bytes >= @as(u64, count) * kv_dim * 2);
    }

    /// `attn_out * sigmoid(gate)`, then `o_proj` into `stream_out`.
    fn attentionOutput(
        self: *Model,
        tokens: u32,
        weights: *const load.AttentionWeights,
    ) !void {
        assert(tokens > 0);
        assert(weights.o_proj.rows == self.config.hidden_size);

        const width: u32 = self.config.num_attention_heads * self.config.head_dim;
        const elements: u32 = tokens * width;
        const source = state_module.view(
            &self.state.attn_out,
            0,
            @as(u64, elements) * 2,
        );
        const target = state_module.view(
            &self.state.attn_out_f32,
            0,
            @as(u64, elements) * 4,
        );
        try moe.launchConvertF32(
            &self.moe_kernels,
            &self.ple.stream,
            &source,
            elements,
            &target,
        );
        try glue.launchSigmoidMul(
            &self.glue_kernels,
            self.ple.stream.handle,
            target.pointer,
            self.state.attn_gate.pointer,
            self.state.attn_gated.pointer,
            elements,
        );
        try self.ple.project(.{
            .x = self.state.attn_gated.pointer,
            .weight = weights.o_proj.data.pointer,
            .out = self.state.stream_out.pointer,
            .rows = tokens,
            .in_features = width,
            .out_features = self.config.hidden_size,
        });
        assert(self.state.attn_gated.bytes >= @as(u64, elements) * 4);
    }

    /// The Gated DeltaNet block over `stream_in`, writing `stream_out`.
    fn gdnBlock(self: *Model, index: u32, tokens: u32, recurrent_form: bool) !void {
        assert(index < self.plan.layer_count);
        assert(tokens > 0);

        var timer = try std.time.Timer.start();
        const weights = &self.weights.layers[index].gdn.?;
        const cache = &self.state.layers[index].gdn.?;
        const config = &self.config;
        const heads: u32 = config.linear_num_value_heads;
        const k_heads: u32 = config.linear_num_key_heads;
        const k_head: u32 = config.linear_key_head_dim;
        const v_head: u32 = config.linear_value_head_dim;
        const conv_dim: u32 = config.conv_dim();
        const value_dim: u32 = config.value_dim();

        try self.project(&self.state.stream_in, weights.in_proj_qkv, &self.state.gdn_qkv, tokens);
        try self.project(&self.state.stream_in, weights.in_proj_z, &self.state.gdn_z, tokens);
        try self.project(&self.state.stream_in, weights.in_proj_b, &self.state.gdn_b, tokens);
        try self.project(&self.state.stream_in, weights.in_proj_a, &self.state.gdn_a, tokens);
        try glue.launchCastF32Bf16(
            &self.glue_kernels,
            self.ple.stream.handle,
            self.state.gdn_qkv.pointer,
            self.state.gdn_conv_in.pointer,
            tokens * conv_dim,
        );
        try gdn.launchConv(
            &self.gdn_kernels,
            self.ple.stream.handle,
            .{
                .x = self.state.gdn_conv_in.pointer,
                .weight = weights.conv.pointer,
                .state = cache.conv.pointer,
                .out = self.state.gdn_conv_out.pointer,
            },
            tokens,
            conv_dim,
        );
        const q_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(k_head)));
        try glue.launchGdnPrepare(
            &self.glue_kernels,
            self.ple.stream.handle,
            self.state.gdn_conv_out.pointer,
            self.state.gdn_query.pointer,
            self.state.gdn_key.pointer,
            self.state.gdn_value.pointer,
            tokens,
            k_heads,
            heads,
            k_head,
            v_head,
            q_scale,
            config.rms_norm_eps,
        );
        try glue.launchGdnDecay(
            &self.glue_kernels,
            self.ple.stream.handle,
            self.state.gdn_a.pointer,
            self.state.gdn_b.pointer,
            weights.a_log.pointer,
            weights.dt_bias.pointer,
            self.state.gdn_beta.pointer,
            self.state.gdn_decay.pointer,
            tokens,
            heads,
        );
        try glue.launchCastF32Bf16(
            &self.glue_kernels,
            self.ple.stream.handle,
            self.state.gdn_z.pointer,
            self.state.gdn_gate.pointer,
            tokens * value_dim,
        );
        const shape = gdn.DeltaShape{
            .tokens = tokens,
            .num_k_heads = k_heads,
            .num_v_heads = heads,
            .k_head = k_head,
            .v_head = v_head,
        };
        if (recurrent_form) {
            try gdn.launchDecode(
                &self.gdn_kernels,
                self.ple.stream.handle,
                .{
                    .query = self.state.gdn_query.pointer,
                    .key = self.state.gdn_key.pointer,
                    .value = self.state.gdn_value.pointer,
                    .beta = self.state.gdn_beta.pointer,
                    .decay = self.state.gdn_decay.pointer,
                    .state = cache.recurrent.pointer,
                    .gate = self.state.gdn_gate.pointer,
                    .norm_weight = weights.norm.pointer,
                    .out = self.state.gdn_core.pointer,
                },
                shape,
                config.rms_norm_eps,
                @intFromEnum(config.output_gate_type),
            );
        } else {
            try gdn.launchPrefill(
                &self.gdn_kernels,
                self.ple.stream.handle,
                .{
                    .query = self.state.gdn_query.pointer,
                    .key = self.state.gdn_key.pointer,
                    .value = self.state.gdn_value.pointer,
                    .beta = self.state.gdn_beta.pointer,
                    .decay = self.state.gdn_decay.pointer,
                    .state = cache.recurrent.pointer,
                    .out = self.state.gdn_core.pointer,
                },
                shape,
            );
            try gdn.launchGateNorm(
                &self.gdn_kernels,
                self.ple.stream.handle,
                .{
                    .values = self.state.gdn_core.pointer,
                    .gate = self.state.gdn_gate.pointer,
                    .weight = weights.norm.pointer,
                    .out = self.state.gdn_normed.pointer,
                },
                tokens * heads,
                v_head,
                config.rms_norm_eps,
                @intFromEnum(config.output_gate_type),
            );
        }
        const normed = if (recurrent_form)
            &self.state.gdn_core
        else
            &self.state.gdn_normed;
        const source = state_module.view(
            normed,
            0,
            @as(u64, tokens) * value_dim * 2,
        );
        const target = state_module.view(
            &self.state.gdn_normed_f32,
            0,
            @as(u64, tokens) * value_dim * 4,
        );
        try moe.launchConvertF32(
            &self.moe_kernels,
            &self.ple.stream,
            &source,
            tokens * value_dim,
            &target,
        );
        if (self.cluster) |cluster| {
            // Column parallel: this rank computes its output rows over the
            // full contraction, then the peers' rows are gathered in. The
            // gathered stream feeds the residual add, so nothing downstream
            // re-projects it and no compute is duplicated here (see
            // `Cluster`'s note on the shared expert, where one is).
            assert(cluster.dense_rows > 0);
            assert(config.hidden_size % cluster.dense_rows == 0);
            assert(cluster.dense_first + cluster.dense_rows <= config.hidden_size);
            const sliced = load.Matrix{
                .data = state_module.view(
                    &weights.out_proj.data,
                    @as(u64, cluster.dense_first) * value_dim * 4,
                    @as(u64, cluster.dense_rows) * value_dim * 4,
                ),
                .rows = cluster.dense_rows,
                .cols = value_dim,
            };
            try self.project(&target, sliced, &cluster.dense_slice, tokens);
            try cluster.gather_dense(
                cluster.user,
                &self.state.stream_out,
                tokens,
                config.hidden_size,
            );
        } else {
            try self.ple.project(.{
                .x = target.pointer,
                .weight = weights.out_proj.data.pointer,
                .out = self.state.stream_out.pointer,
                .rows = tokens,
                .in_features = value_dim,
                .out_features = config.hidden_size,
            });
        }
        self.timings.attention_ns += timer.lap();
        assert(self.state.gdn_query.bytes >= @as(u64, tokens) * k_heads * k_head * 4);
    }

    /// The attention kernel's shape set, from the tiny config.
    fn attentionShapes(self: *const Model) attention.Shapes {
        assert(self.config.head_dim > 0);
        assert(self.config.indexer_head_dim > 0);

        return .{
            .heads = self.config.num_attention_heads,
            .kv_heads = self.config.num_key_value_heads,
            .head_dim = self.config.head_dim,
            .index_heads = self.config.indexer_n_heads,
            .index_kv_heads = self.config.indexer_kv_heads,
            .index_dim = self.config.indexer_head_dim,
            .compress_ratio = self.config.indexer_compress_ratio,
            .budget = self.config.indexer_budget,
            .rotary_dim = self.config.rotary_dim(),
            .section = self.config.mrope_section,
            .eps = self.config.rms_norm_eps,
            .theta = self.config.rope_theta,
        };
    }

    /// DIAGNOSTIC (temporary): runs three full forwards from a reset, keeping
    /// the residual stream after every phase and the logits, and prints how
    /// far consecutive runs drifted at each phase.
    pub fn diagnoseReproducibility(self: *Model, ids: []const i64) !void {
        assert(ids.len > 0);
        assert(self.plan.layer_count > 0);

        const tokens: u32 = @intCast(ids.len);
        const phases: usize = self.plan.layer_count + 2;
        const width: usize = tokens * self.config.hyper_size();
        const mixed_length: usize = tokens * self.config.hidden_size;
        const logits_length: usize = tokens * self.config.vocab_size;
        const rounds: usize = 3;
        assert(logits_length <= width);
        const host = try self.allocator.alloc(f32, rounds * phases * width);
        defer self.allocator.free(host);

        for (0..rounds) |round| {
            try self.reset();
            try self.loadEmbedding(ids, tokens);
            try self.uploadPositions(tokens, 0);
            for (0..self.plan.layer_count) |index| {
                try self.forwardLayer(
                    @intCast(index),
                    ids,
                    tokens,
                    0,
                    false,
                );
                const sink = host[(round * phases + index) * width ..][0..width];
                try self.state.downloadF32(&self.state.hyper, sink);
            }
            try self.forwardMixer(tokens);
            const mixed = state_module.view(
                &self.state.mixed,
                0,
                @as(u64, mixed_length) * 4,
            );
            const mixed_sink = host[(round * phases + self.plan.layer_count) * width ..][0..mixed_length];
            try self.state.downloadF32(&mixed, mixed_sink);
            try self.forwardHead(tokens);
            const head_sink = host[(round * phases + phases - 1) * width ..][0..logits_length];
            try self.state.downloadF32(&self.state.logits, head_sink);
        }
        for (1..rounds) |round| {
            for (0..phases) |phase| {
                const first = host[((round - 1) * phases + phase) * width ..][0..width];
                const second = host[(round * phases + phase) * width ..][0..width];
                var max_abs: f64 = 0.0;
                var differing: usize = 0;
                for (second, first) |value, reference| {
                    const delta = @abs(@as(f64, value) - @as(f64, reference));
                    max_abs = @max(max_abs, delta);
                    if (@as(u32, @bitCast(value)) != @as(u32, @bitCast(reference))) {
                        differing += 1;
                    } else {
                        assert(@as(u32, @bitCast(value)) == @as(u32, @bitCast(reference)));
                    }
                }
                std.debug.print(
                    "DIAG round {d} phase {d}: max_abs {e:.3} differing {d}\n",
                    .{ round, phase, max_abs, differing },
                );
            }
        }
        assert(phases == self.plan.layer_count + 2);
    }
};

/// How many tokens one row's indexer selection lists.
///
/// The indexer keeps `min(blocks, block_topk)` complete blocks plus the
/// incomplete trailing block, which is the prefix of `selected` it writes; the
/// rest of the slot list stays at its -1 fill and must not be gathered.
/// Whether the router selected `expert` for one of this forward's tokens.
fn selectedExpert(selected: []const u32, expert: u32) bool {
    assert(selected.len > 0);
    assert(expert < std.math.maxInt(u32));

    for (selected) |candidate| {
        if (candidate == expert) {
            return true;
        } else {
            assert(candidate != expert);
        }
    }
    return false;
}

/// How many tokens one row's indexer selection lists.
pub fn selectionCount(visible: u32, ratio: u32, block_topk: u32) u32 {
    assert(visible > 0);
    assert(ratio > 0);
    assert(block_topk > 0);

    const blocks = visible / ratio;
    const tail = visible - blocks * ratio;
    assert(tail < ratio);
    const kept = @min(blocks, block_topk);
    return kept * ratio + tail;
}
