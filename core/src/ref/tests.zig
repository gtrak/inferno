//! Golden comparison suite for the fp32 CPU reference (issue I-005).
//!
//! Everything here compares the Zig reference against
//! `tests/golden/tiny/trace.safetensors`, which the Hugging Face reference
//! produced, and against `tests/golden/tiny/greedy_ids.json`.
//!
//! Tolerances, from the issue:
//!
//! * per tensor: `max |err| / max(1e-6, max |golden|) <= 2e-5`;
//! * the fixture is bit-exact only under `torch.set_num_threads(1)`, the
//!   setting `tests/golden/README.md` records and re-measures (at 1 thread the
//!   reference reproduces all 224 float tensors exactly). Left at the default
//!   thread count the reference itself moves - ~4.5e-8 (prefill) and
//!   ~3.0e-8 (decode) on `final/logits`, ~2.4e-7 over every captured tensor -
//!   so compare bit-exactly only against that setting and otherwise use
//!   `rtol`/`atol` ~ 1e-6 in fp32; the gate above is that bound scaled by
//!   magnitude.
//!
//! The prefill pass is driven layer by layer so every intermediate is still
//! live when it is compared; a second test then runs the whole model through
//! `model.forward` and `model.generate`, which is what pins the end-to-end
//! result to the golden logits and the greedy ids.

const std = @import("std");

const attention = @import("attention.zig");
const fixtures = @import("fixtures.zig");
const gdn = @import("gdn.zig");
const hc = @import("hc.zig");
const layer = @import("layer.zig");
const model = @import("model.zig");
const ops = @import("ops.zig");
const ple = @import("ple.zig");

const assert = std.debug.assert;

/// Largest scaled error a tensor may show (issue I-005).
const relative_limit: f64 = 2e-5;

/// Floor of the error scale, so near-zero tensors are compared absolutely.
const error_scale_min: f64 = 1e-6;

/// Prefill ids of the golden trace (vocab is 256, so no tokenizer is needed).
const prefill_ids = [12]i64{ 13, 4, 199, 42, 7, 88, 211, 30, 96, 155, 61, 240 };

/// The id the reference feeds for its single decode step.
const decode_id = [1]i64{169};

/// Greedy tokens the reference produces from `prefill_ids`.
const greedy_count: u32 = 16;

/// Longest tensor name the harness builds.
const name_bytes_max: u32 = 96;

/// Accumulates the error table and remembers the worst tensor.
const Report = struct {
    worst_name: [name_bytes_max]u8 = [_]u8{0} ** name_bytes_max,
    worst_length: u32 = 0,
    worst_scaled: f64 = 0.0,
    worst_absolute: f64 = 0.0,
    checked: u32 = 0,
    failures: u32 = 0,

    /// Compares one tensor and prints `name max_abs_err scaled_err`.
    fn tensor_f32(
        self: *Report,
        tensor_name: []const u8,
        golden: []const f32,
        actual: []const f32,
    ) !void {
        assert(tensor_name.len > 0);
        assert(tensor_name.len < name_bytes_max);
        if (golden.len != actual.len) {
            std.debug.print(
                "{s:<44} LENGTH {d} vs {d}\n",
                .{ tensor_name, golden.len, actual.len },
            );
            self.checked += 1;
            self.failures += 1;
            return;
        } else {
            assert(golden.len == actual.len);
        }

        var largest: f64 = 0.0;
        var magnitude: f64 = 0.0;
        for (golden, actual) |want, got| {
            const difference = @abs(@as(f64, want) - @as(f64, got));
            largest = @max(largest, difference);
            magnitude = @max(magnitude, @abs(@as(f64, want)));
        }
        const scaled = largest / @max(error_scale_min, magnitude);
        std.debug.print(
            "{s:<44} {e:.3} {e:.3}\n",
            .{ tensor_name, largest, scaled },
        );
        if (scaled > self.worst_scaled) {
            @memcpy(self.worst_name[0..tensor_name.len], tensor_name);
            self.worst_length = @intCast(tensor_name.len);
            self.worst_scaled = scaled;
            self.worst_absolute = largest;
        } else {
            assert(scaled <= self.worst_scaled);
        }
        self.checked += 1;
        if (scaled <= relative_limit) {
            assert(scaled <= relative_limit);
        } else {
            self.failures += 1;
        }
        try std.testing.expect(difference_is_finite(largest));
    }

    /// Compares an integer tensor exactly.
    fn tensor_i64(
        self: *Report,
        tensor_name: []const u8,
        golden: anytype,
        actual: anytype,
    ) !void {
        assert(tensor_name.len > 0);
        assert(tensor_name.len < name_bytes_max);
        if (golden.len != actual.len) {
            std.debug.print(
                "{s:<44} LENGTH {d} vs {d}\n",
                .{ tensor_name, golden.len, actual.len },
            );
            self.checked += 1;
            self.failures += 1;
            return;
        } else {
            assert(golden.len == actual.len);
        }

        var mismatches: u32 = 0;
        for (golden, actual) |want, got| {
            if (want != got) {
                mismatches += 1;
            } else {
                continue;
            }
        }
        std.debug.print(
            "{s:<44} {d} mismatch(es)\n",
            .{ tensor_name, mismatches },
        );
        self.checked += 1;
        if (mismatches == 0) {
            assert(mismatches == 0);
        } else {
            self.failures += 1;
        }
    }

    /// Compares a boolean mask exactly.
    fn tensor_bool(
        self: *Report,
        tensor_name: []const u8,
        golden: []const u8,
        actual: []const bool,
    ) !void {
        assert(tensor_name.len > 0);
        assert(tensor_name.len < name_bytes_max);
        if (golden.len != actual.len) {
            std.debug.print(
                "{s:<44} LENGTH {d} vs {d}\n",
                .{ tensor_name, golden.len, actual.len },
            );
            self.checked += 1;
            self.failures += 1;
            return;
        } else {
            assert(golden.len == actual.len);
        }

        var mismatches: u32 = 0;
        var selected: u32 = 0;
        for (golden, actual) |want, got| {
            if (want != 0) selected += 1;
            if ((want != 0) != got) {
                mismatches += 1;
            } else {
                continue;
            }
        }
        std.debug.print(
            "{s:<44} {d} mismatch(es), {d} selected\n",
            .{ tensor_name, mismatches, selected },
        );
        self.checked += 1;
        if (mismatches == 0) {
            assert(mismatches == 0);
        } else {
            self.failures += 1;
        }
    }

    /// True when the largest absolute error is a real number.
    fn difference_is_finite(largest: f64) bool {
        assert(largest >= 0.0);
        assert(largest <= std.math.inf(f64));
        return largest == largest;
    }

    /// Prints the headline: the worst tensor and its scaled error.
    fn summary(self: *const Report, label: []const u8) void {
        assert(label.len > 0);
        assert(self.checked > 0);

        std.debug.print(
            "== {s}: {d} tensors checked, worst {s} scaled {e:.3} " ++
                "(abs {e:.3})\n",
            .{
                label,
                self.checked,
                self.worst_name[0..self.worst_length],
                self.worst_scaled,
                self.worst_absolute,
            },
        );
        std.debug.print(
            "== {s}: {d} failing tensors\n",
            .{ label, self.failures },
        );
    }
};

/// The model, its scratch space, the two fixture files and the error table.
const Harness = struct {
    allocator: std.mem.Allocator,
    checkpoint: fixtures.Tensors,
    trace: fixtures.Tensors,
    model: model.Model,
    scratch: model.Scratch,
    report: Report = .{},
    probe_storage: ProbeStorage,
    saved_hyper: []f32,
    transposed: []f32,
    /// `[conv_dim, kernel - 1]` staging view of the trace's conv state,
    /// which is one column wider than the state the reference keeps (see
    /// `compare_gdn_state`).
    conv_view: []f32,

    /// Opens the fixtures and builds the model.
    fn init(allocator: std.mem.Allocator) !Harness {
        assert(std.mem.endsWith(u8, fixtures_directory, "tiny"));
        assert(fixtures_directory.len > 0);

        var checkpoint = try fixtures.open_checkpoint(allocator);
        errdefer checkpoint.close(allocator);
        var trace = try fixtures.open_trace(allocator);
        errdefer trace.close(allocator);

        var harness = Harness{
            .allocator = allocator,
            .checkpoint = checkpoint,
            .trace = trace,
            .model = undefined,
            .scratch = undefined,
            .probe_storage = undefined,
            .saved_hyper = &[_]f32{},
            .transposed = &[_]f32{},
            .conv_view = &[_]f32{},
        };
        try harness.model.init(allocator, &harness.checkpoint);
        harness.scratch = try model.Scratch.init(
            allocator,
            &harness.model.config,
        );
        const config = &harness.model.config;
        harness.probe_storage = try ProbeStorage.init(allocator, config);
        harness.saved_hyper = try allocator.alloc(
            f32,
            config.max_position_embeddings * config.hyper_size(),
        );
        harness.transposed = try allocator.alloc(
            f32,
            config.max_position_embeddings * config.hyper_size(),
        );
        assert(config.linear_conv_kernel_dim >= 2);
        harness.conv_view = try allocator.alloc(
            f32,
            config.conv_dim() * (config.linear_conv_kernel_dim - 1),
        );
        assert(harness.scratch.hyper.len > 0);
        assert(harness.conv_view.len > 0);
        return harness;
    }

    /// Releases everything `init` allocated.
    fn deinit(self: *Harness) void {
        assert(self.scratch.hyper.len > 0);
        assert(self.saved_hyper.len > 0);
        assert(self.conv_view.len > 0);

        self.allocator.free(self.conv_view);
        self.allocator.free(self.transposed);
        self.allocator.free(self.saved_hyper);
        self.probe_storage.deinit(self.allocator);
        self.scratch.deinit(self.allocator);
        self.model.deinit(self.allocator);
        self.trace.close(self.allocator);
        self.checkpoint.close(self.allocator);
    }

    /// Fetches a golden f32 tensor by name.
    fn golden(self: *const Harness, name: []const u8) ![]const f32 {
        assert(name.len > 0);
        assert(self.trace.entry_count > 0);

        const tensor = try self.trace.tensor(name);
        return tensor.values;
    }

    /// Runs one forward of `tokens` ids layer by layer, comparing as it goes.
    fn run(
        self: *Harness,
        prefix: []const u8,
        ids: []const i64,
        tokens: u32,
    ) !void {
        assert(prefix.len <= 8);
        assert(tokens > 0);
        assert(ids.len == tokens);

        const config = &self.model.config;
        const hidden: usize = config.hidden_size;
        const hyper: usize = config.hyper_size();
        const past: u32 = self.model.cache.length;
        try std.testing.expect(past + tokens <= config.max_position_embeddings);

        var name_buffer: [name_bytes_max]u8 = undefined;
        for (0..tokens) |t| {
            self.scratch.positions[t] = past + @as(u32, @intCast(t));
        }
        ops.embed_gather(
            self.model.weights.embed_tokens,
            config.hidden_size,
            ids,
            self.scratch.embed[0 .. tokens * hidden],
        );
        for (0..tokens) |t| {
            for (0..config.hc_count) |c| {
                for (0..hidden) |h| {
                    self.scratch.hyper[t * hyper + c * hidden + h] =
                        self.scratch.embed[t * hidden + h];
                }
            }
        }
        const embed_label = try section_name(
            prefix,
            "embed/output",
            &name_buffer,
        );
        try self.report.tensor_f32(
            embed_label,
            try self.golden(embed_label),
            self.scratch.embed[0 .. tokens * hidden],
        );
        // The rotary tables cover every position the caches hold so far.
        const rotary: usize = config.rotary_dim();
        const table_label = try section_name(
            prefix,
            "rotary/cos",
            &name_buffer,
        );
        try self.report.tensor_f32(
            table_label,
            try self.golden(table_label),
            self.scratch.cos[0 .. (past + tokens) * rotary],
        );
        const table = attention.Rope{
            .cos = self.scratch.cos,
            .sin = self.scratch.sin,
        };
        const recurrent_form = tokens == 1 and past > 0;
        for (0..config.num_hidden_layers) |index| {
            try self.compare_layer_in(prefix, @intCast(index), tokens);
            layer.forward(
                self.scratch.hyper[0 .. tokens * hyper],
                tokens,
                self.scratch.positions[0..tokens],
                ids,
                &self.model.weights.layers[index],
                &self.model.cache.layers[index],
                &self.scratch.layer,
                table,
                config,
                recurrent_form,
                &self.probe_storage.probe,
            );
            try self.compare_layer(prefix, @intCast(index), tokens);
        }
        hc.forward(
            self.scratch.hyper[0 .. tokens * hyper],
            tokens,
            self.scratch.mixed[0 .. tokens * hidden],
            null,
            &self.model.weights.mixer,
            &self.scratch.layer.hc_scratch,
            config,
        );
        ops.matmul(
            self.scratch.mixed[0 .. tokens * hidden],
            self.model.weights.lm_head,
            self.scratch.logits[0 .. tokens * config.vocab_size],
            .{
                .rows = tokens,
                .in_features = config.hidden_size,
                .out_features = config.vocab_size,
            },
        );
        const mixed_label = try section_name(
            prefix,
            "final/mixer_out",
            &name_buffer,
        );
        try self.report.tensor_f32(
            mixed_label,
            try self.golden(mixed_label),
            self.scratch.mixed[0 .. tokens * hidden],
        );
        const logits_label = try section_name(
            prefix,
            "final/logits",
            &name_buffer,
        );
        try self.report.tensor_f32(
            logits_label,
            try self.golden(logits_label),
            self.scratch.logits[0 .. tokens * config.vocab_size],
        );
        self.model.cache.length += tokens;
        assert(self.model.cache.length == past + tokens);
    }

    /// Copies the residual before the layer so `layer{L}/in` can be compared.
    fn compare_layer_in(
        self: *Harness,
        prefix: []const u8,
        index: u32,
        tokens: u32,
    ) !void {
        assert(index < self.model.config.num_hidden_layers);
        assert(tokens > 0);

        const hyper: usize = self.model.config.hyper_size();
        const length = tokens * hyper;
        @memcpy(self.saved_hyper[0..length], self.scratch.hyper[0..length]);
        var label_buffer: [name_bytes_max]u8 = undefined;
        const label = try sections(
            prefix,
            &label_buffer,
            "layer{d}/in",
            .{index},
        );
        try self.report.tensor_f32(
            label,
            try self.golden(label),
            self.saved_hyper[0..length],
        );
    }

    /// Compares every recorded intermediate of layer `index`.
    fn compare_layer(
        self: *Harness,
        prefix: []const u8,
        index: u32,
        tokens: u32,
    ) !void {
        assert(index < self.model.config.num_hidden_layers);
        assert(tokens > 0);

        const config = &self.model.config;
        const hidden: usize = config.hidden_size;
        const hyper: usize = config.hyper_size();
        const kind = config.layer_types[index];
        var buffers: [8][name_bytes_max]u8 = undefined;
        const out = try sections(prefix, &buffers[0], "layer{d}/out", .{index});
        try self.report.tensor_f32(
            out,
            try self.golden(out),
            self.scratch.hyper[0 .. tokens * hyper],
        );

        const mixed = try sections(
            prefix,
            &buffers[1],
            "layer{d}/hc_attn/mixed",
            .{index},
        );
        try self.report.tensor_f32(
            mixed,
            try self.golden(mixed),
            self.probe_storage.attn_mixed[0 .. tokens * hidden],
        );
        const inject = try sections(
            prefix,
            &buffers[2],
            "layer{d}/hc_attn/inject_weights",
            .{index},
        );
        try self.report.tensor_f32(
            inject,
            try self.golden(inject),
            self.probe_storage.attn_inject[0 .. tokens * config.hc_count],
        );
        const mlp_mixed = try sections(
            prefix,
            &buffers[3],
            "layer{d}/hc_mlp/mixed",
            .{index},
        );
        try self.report.tensor_f32(
            mlp_mixed,
            try self.golden(mlp_mixed),
            self.scratch.layer.stream_in[0 .. tokens * hidden],
        );
        const moe_out = try sections(
            prefix,
            &buffers[4],
            "layer{d}/moe/out",
            .{index},
        );
        try self.report.tensor_f32(
            moe_out,
            try self.golden(moe_out),
            self.scratch.layer.stream_out[0 .. tokens * hidden],
        );
        if (kind == .linear_attention) {
            try self.compare_gdn(prefix, index, tokens);
        } else {
            try self.compare_attention(prefix, index, tokens);
        }
        if (config.ple_layer_index(index) != null) {
            try self.compare_ple(prefix, index, tokens);
        } else {
            assert(self.model.weights.layers[index].ple == null);
        }
        try self.compare_moe(prefix, index, tokens);
        assert(self.report.checked > 0);
    }

    /// Compares the GatedDeltaNet intermediates of layer `index`.
    fn compare_gdn(
        self: *Harness,
        prefix: []const u8,
        index: u32,
        tokens: u32,
    ) !void {
        assert(index < self.model.config.num_hidden_layers);
        assert(tokens > 0);

        const config = &self.model.config;
        const heads: usize = config.linear_num_value_heads;
        const k_heads: usize = config.linear_num_key_heads;
        const k_head: usize = config.linear_key_head_dim;
        const v_head: usize = config.linear_value_head_dim;
        const source = &self.scratch.layer.gdn_scratch;
        var buffers: [10][name_bytes_max]u8 = undefined;
        const block_out = try sections(
            prefix,
            &buffers[0],
            "layer{d}/gdn/out",
            .{index},
        );
        try self.report.tensor_f32(
            block_out,
            try self.golden(block_out),
            self.probe_storage.block_out[0 .. tokens * config.hidden_size],
        );
        inline for (.{
            .{ "q", "layer{d}/gdn/q" },
            .{ "k", "layer{d}/gdn/k" },
            .{ "v", "layer{d}/gdn/v" },
            .{ "g", "layer{d}/gdn/g" },
            .{ "beta", "layer{d}/gdn/beta" },
            .{ "core_out", "layer{d}/gdn/core_out" },
            .{ "z", "layer{d}/gdn/z" },
            .{ "normed", "layer{d}/gdn/normed" },
            .{ "conv_out", "layer{d}/gdn/conv_out" },
        }, 1..) |entry, slot| {
            const name_text = try sections(
                prefix,
                &buffers[slot],
                entry[1],
                .{index},
            );
            const actual = gdn_slice(source, entry[0], tokens, config);
            try self.report.tensor_f32(
                name_text,
                try self.golden(name_text),
                actual,
            );
        }
        assert(k_heads * k_head > 0);
        assert(heads * v_head > 0);
        try self.compare_gdn_state(prefix, index);
    }

    /// Compares the GDN cache state of layer `index` after the forward.
    ///
    /// These two tensors are the whole cross-step contract of a
    /// linear-attention layer - the GDN kernels of I-009 are handed exactly
    /// them - and before this comparison nothing in the suite read them at
    /// all. `recurrent` is `[num_v_heads, head_k_dim, head_v_dim]` in both;
    /// the conv state needs the column drop that `drop_leading_column`
    /// explains.
    fn compare_gdn_state(
        self: *Harness,
        prefix: []const u8,
        index: u32,
    ) !void {
        assert(index < self.model.config.num_hidden_layers);
        assert(self.model.config.layer_types[index] == .linear_attention);

        const state = &self.model.cache.layers[index].gdn;
        assert(state.conv.len > 0);
        assert(state.recurrent.len > 0);

        var buffers: [2][name_bytes_max]u8 = undefined;
        const recurrent_label = try sections(
            prefix,
            &buffers[0],
            "layer{d}/gdn/recurrent_state",
            .{index},
        );
        try self.report.tensor_f32(
            recurrent_label,
            try self.golden(recurrent_label),
            state.recurrent,
        );
        const conv_label = try sections(
            prefix,
            &buffers[1],
            "layer{d}/gdn/conv_state",
            .{index},
        );
        const golden_conv = try self.golden(conv_label);
        try self.report.tensor_f32(
            conv_label,
            self.drop_leading_column(golden_conv),
            state.conv,
        );
    }

    /// Returns `captured` without the leading column that no conv reads.
    ///
    /// The HF cache keeps `linear_conv_kernel_dim` columns of conv input - one
    /// more than a causal conv of that kernel needs - while the reference keeps
    /// exactly `kernel - 1`, the minimum a continuation requires. The extra
    /// column is one step of history we never read: `gdn/conv_out` matches
    /// through prefill *and* the decode step on `kernel - 1` columns alone. So
    /// the comparable part is the trailing `kernel - 1` columns of every
    /// channel, repacked into `conv_view`.
    fn drop_leading_column(
        self: *Harness,
        captured: []const f32,
    ) []const f32 {
        const config = &self.model.config;
        const channels: usize = config.conv_dim();
        const history: usize = config.linear_conv_kernel_dim - 1;
        const stored: usize = config.linear_conv_kernel_dim;
        assert(captured.len == channels * stored);
        assert(self.conv_view.len == channels * history);

        for (0..channels) |channel| {
            const source = captured[channel * stored + 1 ..][0..history];
            @memcpy(self.conv_view[channel * history ..][0..history], source);
        }
        assert(self.conv_view.len == channels * history);
        return self.conv_view;
    }

    /// Compares the indexed-attention intermediates of layer `index`.
    fn compare_attention(
        self: *Harness,
        prefix: []const u8,
        index: u32,
        tokens: u32,
    ) !void {
        assert(index < self.model.config.num_hidden_layers);
        assert(tokens > 0);

        const config = &self.model.config;
        const source = &self.scratch.layer.attention_scratch;
        const state = &self.model.cache.layers[index].attention;
        const heads: usize = config.num_attention_heads;
        const head_dim: usize = config.head_dim;
        const kv_dim: usize = config.num_key_value_heads * config.head_dim;
        const kv_length: usize = state.length;
        var buffers: [12][name_bytes_max]u8 = undefined;

        const out_label = try sections(
            prefix,
            &buffers[0],
            "layer{d}/attn/out",
            .{index},
        );
        try self.report.tensor_f32(
            out_label,
            try self.golden(out_label),
            self.probe_storage.block_out[0 .. tokens * config.hidden_size],
        );
        // `attn/q` is stored head-major by the reference: [heads, tokens, dim].
        const query_label = try sections(
            prefix,
            &buffers[1],
            "layer{d}/attn/q",
            .{index},
        );
        transpose_heads(
            self.transposed[0 .. heads * tokens * head_dim],
            source.query[0 .. tokens * heads * head_dim],
            heads,
            tokens,
            head_dim,
        );
        try self.report.tensor_f32(
            query_label,
            try self.golden(query_label),
            self.transposed[0 .. heads * tokens * head_dim],
        );
        const key_label = try sections(
            prefix,
            &buffers[2],
            "layer{d}/attn/k",
            .{index},
        );
        try self.report.tensor_f32(
            key_label,
            try self.golden(key_label),
            state.key[0 .. kv_length * kv_dim],
        );
        const value_label = try sections(
            prefix,
            &buffers[3],
            "layer{d}/attn/v",
            .{index},
        );
        try self.report.tensor_f32(
            value_label,
            try self.golden(value_label),
            state.value[0 .. kv_length * kv_dim],
        );
        const gate_label = try sections(
            prefix,
            &buffers[4],
            "layer{d}/attn/gate",
            .{index},
        );
        try self.report.tensor_f32(
            gate_label,
            try self.golden(gate_label),
            source.gate[0 .. tokens * heads * head_dim],
        );
        const query_index_label = try sections(
            prefix,
            &buffers[5],
            "layer{d}/attn/indexer/q",
            .{index},
        );
        try self.report.tensor_f32(
            query_index_label,
            try self.golden(query_index_label),
            source.index_query[0 .. tokens * config.indexer_n_heads *
                config.indexer_head_dim],
        );
        const keys_label = try sections(
            prefix,
            &buffers[6],
            "layer{d}/attn/indexer/raw_keys",
            .{index},
        );
        try self.report.tensor_f32(
            keys_label,
            try self.golden(keys_label),
            state.index_key[0 .. kv_length * config.indexer_head_dim],
        );
        const selection_label = try sections(
            prefix,
            &buffers[7],
            "layer{d}/attn/mask",
            .{index},
        );
        const mask_values = try self.trace.tensor_bool(selection_label);
        try self.report.tensor_bool(
            selection_label,
            mask_values,
            source.mask[0 .. tokens * kv_length],
        );
        const selected_label = try sections(
            prefix,
            &buffers[8],
            "layer{d}/attn/indexer/selected",
            .{index},
        );
        try self.report.tensor_i64(
            selected_label,
            try self.trace.tensor_i32(selected_label),
            source.selected[0 .. tokens * config.indexer_selected_slots()],
        );
        assert(kv_length > 0);
    }

    /// Compares the PLE intermediates of layer `index`.
    fn compare_ple(
        self: *Harness,
        prefix: []const u8,
        index: u32,
        tokens: u32,
    ) !void {
        assert(index < self.model.config.num_hidden_layers);
        assert(tokens > 0);

        const config = &self.model.config;
        const source = &self.scratch.layer.ple_scratch;
        const hyper: usize = config.hyper_size();
        var buffers: [10][name_bytes_max]u8 = undefined;

        const out_label = try sections(
            prefix,
            &buffers[0],
            "layer{d}/ple/out",
            .{index},
        );
        try self.report.tensor_f32(
            out_label,
            try self.golden(out_label),
            self.probe_storage.ple_out[0 .. tokens * hyper],
        );
        const ids_label = try sections(
            prefix,
            &buffers[1],
            "layer{d}/ple/ngram_ids",
            .{index},
        );
        try self.report.tensor_i64(
            ids_label,
            try self.trace.tensor_i64(ids_label),
            source.ngram_ids[0 .. tokens * config.ngram_heads()],
        );
        inline for (.{
            .{ "embeddings", "layer{d}/ple/embeddings" },
            .{ "key_normed", "layer{d}/ple/key_normed" },
            .{ "query_normed", "layer{d}/ple/query_normed" },
            .{ "value", "layer{d}/ple/value" },
            .{ "gate", "layer{d}/ple/gate" },
            .{ "gated_value", "layer{d}/ple/gated_value" },
            .{ "gated_value_normed", "layer{d}/ple/gated_value_normed" },
            .{ "conv_out", "layer{d}/ple/conv_out" },
        }, 2..) |entry, slot| {
            const text = try sections(
                prefix,
                &buffers[slot],
                entry[1],
                .{index},
            );
            const actual = ple_slice(source, entry[0], tokens, config);
            try self.report.tensor_f32(text, try self.golden(text), actual);
        }
        assert(source.conv_out.len >= tokens * hyper);
    }

    /// Compares the MoE intermediates of layer `index`.
    fn compare_moe(
        self: *Harness,
        prefix: []const u8,
        index: u32,
        tokens: u32,
    ) !void {
        assert(index < self.model.config.num_hidden_layers);
        assert(tokens > 0);

        const config = &self.model.config;
        const source = &self.scratch.layer.moe_scratch;
        const hidden: usize = config.hidden_size;
        const experts: usize = config.num_experts;
        const top_k: usize = config.num_experts_per_tok;
        var buffers: [8][name_bytes_max]u8 = undefined;

        const logits_label = try sections(
            prefix,
            &buffers[0],
            "layer{d}/moe/router_logits",
            .{index},
        );
        try self.report.tensor_f32(
            logits_label,
            try self.golden(logits_label),
            source.router_logits[0 .. tokens * experts],
        );
        const weights_label = try sections(
            prefix,
            &buffers[1],
            "layer{d}/moe/router_weights",
            .{index},
        );
        try self.report.tensor_f32(
            weights_label,
            try self.golden(weights_label),
            source.topk_values[0 .. tokens * top_k],
        );
        const indices_label = try sections(
            prefix,
            &buffers[2],
            "layer{d}/moe/router_indices",
            .{index},
        );
        var indices: [1024]i64 = undefined;
        assert(tokens * top_k <= indices.len);
        for (0..tokens * top_k) |i| {
            indices[i] = source.topk_indices[i];
        }
        try self.report.tensor_i64(
            indices_label,
            try self.trace.tensor_i64(indices_label),
            indices[0 .. tokens * top_k],
        );
        const expert_label = try sections(
            prefix,
            &buffers[3],
            "layer{d}/moe/expert_out",
            .{index},
        );
        try self.report.tensor_f32(
            expert_label,
            try self.golden(expert_label),
            source.expert_out[0 .. tokens * hidden],
        );
        const shared_label = try sections(
            prefix,
            &buffers[4],
            "layer{d}/moe/shared_out",
            .{index},
        );
        try self.report.tensor_f32(
            shared_label,
            try self.golden(shared_label),
            source.shared_out[0 .. tokens * hidden],
        );
        assert(source.expert_out.len >= tokens * hidden);
    }
};

/// Buffers the probe copies its intermediates into.
const ProbeStorage = struct {
    attn_mixed: []f32,
    attn_inject: []f32,
    block_out: []f32,
    ple_out: []f32,
    probe: layer.Probe = .{},

    /// Allocates the probe buffers and points the probe at them.
    fn init(
        allocator: std.mem.Allocator,
        config: *const fixtures.RefConfig,
    ) !ProbeStorage {
        assert(config.max_position_embeddings > 0);
        assert(config.hidden_size > 0);

        const tokens: usize = config.max_position_embeddings;
        var storage = ProbeStorage{
            .attn_mixed = try allocator.alloc(f32, tokens * config.hidden_size),
            .attn_inject = try allocator.alloc(f32, tokens * config.hc_count),
            .block_out = try allocator.alloc(f32, tokens * config.hidden_size),
            .ple_out = try allocator.alloc(f32, tokens * config.hyper_size()),
        };
        storage.probe = .{
            .block_mixed = storage.attn_mixed,
            .block_inject = storage.attn_inject,
            .block_out = storage.block_out,
            .ple_out = storage.ple_out,
        };
        assert(storage.probe.block_out != null);
        return storage;
    }

    /// Releases the probe buffers.
    fn deinit(self: *ProbeStorage, allocator: std.mem.Allocator) void {
        assert(self.attn_mixed.len > 0);
        assert(self.ple_out.len > 0);

        allocator.free(self.ple_out);
        allocator.free(self.block_out);
        allocator.free(self.attn_inject);
        allocator.free(self.attn_mixed);
    }
};

/// Maps a short key onto the scratch buffer that holds it.
fn gdn_slice(
    source: *const gdn.Scratch,
    comptime key: []const u8,
    tokens: u32,
    config: *const fixtures.RefConfig,
) []const f32 {
    assert(tokens > 0);
    assert(key.len > 0);

    const width: usize = if (comptime std.mem.eql(u8, key, "q"))
        config.key_dim()
    else if (comptime std.mem.eql(u8, key, "k"))
        config.key_dim()
    else if (comptime std.mem.eql(u8, key, "v"))
        config.value_dim()
    else if (comptime std.mem.eql(u8, key, "z"))
        config.value_dim()
    else if (comptime std.mem.eql(u8, key, "g"))
        config.linear_num_value_heads
    else if (comptime std.mem.eql(u8, key, "beta"))
        config.linear_num_value_heads
    else if (comptime std.mem.eql(u8, key, "core_out"))
        config.value_dim()
    else if (comptime std.mem.eql(u8, key, "normed"))
        config.value_dim()
    else if (comptime std.mem.eql(u8, key, "conv_out"))
        config.conv_dim()
    else
        unreachable;

    const length = tokens * width;
    if (comptime std.mem.eql(u8, key, "q")) {
        return source.query[0..length];
    } else if (comptime std.mem.eql(u8, key, "k")) {
        return source.key[0..length];
    } else if (comptime std.mem.eql(u8, key, "v")) {
        return source.value[0..length];
    } else if (comptime std.mem.eql(u8, key, "z")) {
        return source.z[0..length];
    } else if (comptime std.mem.eql(u8, key, "g")) {
        return source.decay[0..length];
    } else if (comptime std.mem.eql(u8, key, "beta")) {
        return source.beta[0..length];
    } else if (comptime std.mem.eql(u8, key, "core_out")) {
        return source.core_out[0..length];
    } else if (comptime std.mem.eql(u8, key, "normed")) {
        return source.normed[0..length];
    } else {
        return source.conv_out[0..length];
    }
}

/// Maps a short key onto the PLE scratch buffer that holds it.
fn ple_slice(
    source: *const ple.Scratch,
    comptime key: []const u8,
    tokens: u32,
    config: *const fixtures.RefConfig,
) []const f32 {
    assert(tokens > 0);
    assert(key.len > 0);

    const width: usize = if (comptime std.mem.eql(u8, key, "embeddings"))
        config.ple_embed_dim
    else if (comptime std.mem.eql(u8, key, "value"))
        config.hidden_size
    else if (comptime std.mem.eql(u8, key, "gate"))
        config.hc_count
    else if (comptime std.mem.eql(u8, key, "conv_out"))
        config.hyper_size()
    else
        config.hyper_size();

    const length = tokens * width;
    if (comptime std.mem.eql(u8, key, "embeddings")) {
        return source.embeddings[0..length];
    } else if (comptime std.mem.eql(u8, key, "key_normed")) {
        return source.key_normed[0..length];
    } else if (comptime std.mem.eql(u8, key, "query_normed")) {
        return source.query_normed[0..length];
    } else if (comptime std.mem.eql(u8, key, "value")) {
        return source.value[0..length];
    } else if (comptime std.mem.eql(u8, key, "gate")) {
        return source.gate[0..length];
    } else if (comptime std.mem.eql(u8, key, "gated_value")) {
        return source.gated_value[0..length];
    } else if (comptime std.mem.eql(u8, key, "gated_value_normed")) {
        return source.gated_value_normed[0..length];
    } else {
        return source.conv_out[0..length];
    }
}

/// Reorders `[tokens, heads, dim]` into `[heads, tokens, dim]`.
fn transpose_heads(
    target: []f32,
    source: []const f32,
    heads: usize,
    tokens: usize,
    head_dim: usize,
) void {
    assert(source.len == heads * tokens * head_dim);
    assert(target.len == source.len);

    for (0..heads) |head| {
        for (0..tokens) |token| {
            for (0..head_dim) |dimension| {
                target[(head * tokens + token) * head_dim + dimension] =
                    source[(token * heads + head) * head_dim + dimension];
            }
        }
    }
    assert(target.len == source.len);
}

/// Formats `format` into `buffer` and returns the text.
fn format_name(
    buffer: *[name_bytes_max]u8,
    comptime format: []const u8,
    args: anytype,
) ![]const u8 {
    assert(buffer.len == name_bytes_max);
    assert(format.len > 0);

    return std.fmt.bufPrint(buffer, format, args);
}

/// `prefix ++ formatted`, for the trace's section prefixes.
fn sections(
    prefix: []const u8,
    buffer: *[name_bytes_max]u8,
    comptime format: []const u8,
    args: anytype,
) ![]const u8 {
    assert(prefix.len <= 8);
    assert(format.len > 0);

    var text: [name_bytes_max]u8 = undefined;
    const formatted = try std.fmt.bufPrint(&text, format, args);
    return section_name(prefix, formatted, buffer);
}

/// `prefix ++ text`, for the trace's section prefixes.
fn section_name(
    prefix: []const u8,
    text: []const u8,
    buffer: *[name_bytes_max]u8,
) ![]const u8 {
    assert(prefix.len <= 8);
    assert(text.len > 0);

    return std.fmt.bufPrint(buffer, "{s}{s}", .{ prefix, text });
}

test "prefill and one decode step match the golden trace" {
    const allocator = std.testing.allocator;
    var harness = try Harness.init(allocator);
    defer harness.deinit();

    std.debug.print(
        "{s:<44} {s:<11} {s}\n",
        .{ "tensor", "max_abs_err", "scaled_err" },
    );
    try harness.run("", &prefill_ids, prefill_ids.len);
    // Non-vacuity: the run has to be selective, not degenerate. Checked here,
    // before the decode step overwrites `logits[0..vocab]`, so every prefill
    // position is still live in the logits buffer.
    try expect_non_vacuous(&harness);
    try harness.run("decode/", &decode_id, 1);
    harness.report.summary("golden trace");
    try std.testing.expectEqual(@as(u32, 0), harness.report.failures);
}

/// Asserts the run is not degenerate: a selective mask, varied routing, and
/// logits with real spread.
fn expect_non_vacuous(harness: *Harness) !void {
    assert(harness.report.checked > 0);
    assert(harness.model.config.num_hidden_layers > 0);

    const config = &harness.model.config;
    // The last layer is linear attention, so read layer 2's selection from the
    // golden mask rather than from the (reused) scratch buffer.
    const mask = try harness.trace.tensor_bool("layer2/attn/mask");
    var selected: u32 = 0;
    for (mask) |value| {
        if (value != 0) selected += 1;
    }
    std.debug.print(
        "layer2/attn/mask selects {d} of {d} positions\n",
        .{ selected, mask.len },
    );
    try std.testing.expect(selected > 0);
    try std.testing.expect(selected < mask.len);

    const indices = try harness.trace.tensor_i64("layer2/moe/router_indices");
    const positions: u32 = @intCast(indices.len / 2);
    const distinct = count_top2_rows(indices, true);
    const ordered = count_top2_rows(indices, false);
    std.debug.print(
        "router picks {d} distinct top-2 expert sets over {d} positions " ++
            "({d} distinct ordered pairs)\n",
        .{ distinct, positions, ordered },
    );
    for ([_]usize{ 0, 5, 11 }) |position| {
        std.debug.print(
            "router top-2 at position {d}: {d}, {d}\n",
            .{
                position,
                indices[position * 2],
                indices[position * 2 + 1],
            },
        );
    }
    // Strict bounds, not the tautologies this check used to hold. With the
    // golden trace every one of them is live: 5 distinct sets over 12
    // positions, 6 distinct ordered pairs. So `distinct > 1` fails when every
    // position routes to the same pair, `distinct < positions` when routing is
    // a bijection, and `ordered > distinct` when the trace never picks a pair
    // in two different orders - which is what makes the slot a routed expert
    // lands in observable.
    try std.testing.expect(distinct > 1);
    try std.testing.expect(distinct < positions);
    try std.testing.expect(ordered > distinct);

    var largest: f32 = 0.0;
    var greatest: f32 = -std.math.floatMax(f32);
    var smallest: f32 = std.math.floatMax(f32);
    const vocab: usize = config.vocab_size;
    const live = harness.scratch.logits[0 .. prefill_ids.len * vocab];
    for (live) |value| {
        largest = @max(largest, @abs(value));
        greatest = @max(greatest, value);
        smallest = @min(smallest, value);
    }
    const spread = greatest - smallest;
    std.debug.print(
        "logits over {d} positions: max |logit| = {e:.3}, " ++
            "max = {e:.3}, min = {e:.3}, spread (max - min) = {e:.3}\n",
        .{ prefill_ids.len, largest, greatest, smallest, spread },
    );
    try std.testing.expect(largest > 0.05);
    try std.testing.expect(spread > 0.05);
}

/// One routing row: the two expert ids as `{first, second}`. With `canonical`
/// the row is an unordered set, so `{2, 3}` and `{3, 2}` coincide.
const Top2Row = [2]i64;

/// `position`'s routing row, canonicalised when `canonical` is set.
fn top2_row(indices: []const i64, position: u32, canonical: bool) Top2Row {
    assert(indices.len % 2 == 0);
    assert(position * 2 + 2 <= indices.len);

    const base = position * 2;
    const first = indices[base];
    const second = indices[base + 1];
    if (canonical) {
        return .{ @min(first, second), @max(first, second) };
    } else {
        return .{ first, second };
    }
}

/// Whether `position`'s row already appeared at an earlier position.
fn top2_seen_before(indices: []const i64, position: u32, canonical: bool) bool {
    assert(indices.len % 2 == 0);
    assert(position * 2 + 2 <= indices.len);

    const wanted = top2_row(indices, position, canonical);
    var earlier: u32 = 0;
    while (earlier < position) : (earlier += 1) {
        const candidate = top2_row(indices, earlier, canonical);
        if (std.mem.eql(i64, &candidate, &wanted)) {
            return true;
        } else {
            continue;
        }
    }
    return false;
}

/// Rows that are not a repeat of an earlier row (as a set when `canonical`).
fn count_top2_rows(indices: []const i64, canonical: bool) u32 {
    assert(indices.len > 0);
    assert(indices.len % 2 == 0);

    const positions: u32 = @intCast(indices.len / 2);
    var distinct: u32 = 0;
    for (0..positions) |position| {
        if (top2_seen_before(indices, @intCast(position), canonical)) {
            continue;
        } else {
            distinct += 1;
        }
    }
    assert(distinct <= positions);
    return distinct;
}

test "the whole model reproduces the golden logits and the greedy ids" {
    const allocator = std.testing.allocator;
    var harness = try Harness.init(allocator);
    defer harness.deinit();

    try harness.model.forward(&prefill_ids, prefill_ids.len, &harness.scratch);
    const logits = try harness.golden("final/logits");
    const vocab: usize = harness.model.config.vocab_size;
    try harness.report.tensor_f32(
        "model.forward/final/logits",
        logits,
        harness.scratch.logits[0 .. prefill_ids.len * vocab],
    );

    var generated: [greedy_count]i64 = undefined;
    try harness.model.generate(
        &prefill_ids,
        greedy_count,
        &generated,
        &harness.scratch,
    );
    harness.report.checked += 1;
    const expected = try load_greedy_ids(allocator);
    try std.testing.expectEqual(greedy_count, expected.count);
    std.debug.print("generated ids:", .{});
    for (generated) |id| {
        std.debug.print(" {d}", .{id});
    }
    std.debug.print("\n", .{});
    for (0..greedy_count) |step| {
        try std.testing.expectEqual(expected.ids[step], generated[step]);
    }
    harness.report.summary("end to end (logits + greedy ids)");
    try std.testing.expectEqual(@as(u32, 0), harness.report.failures);
}

/// `greedy_ids.json` from the fixture directory.
const GreedyIds = struct {
    ids: [greedy_count]i64 = [_]i64{0} ** greedy_count,
    count: u32 = 0,
};

/// Parses the greedy continuation the HF reference produced.
fn load_greedy_ids(allocator: std.mem.Allocator) !GreedyIds {
    assert(fixtures_directory.len > 0);
    assert(std.mem.endsWith(u8, fixtures_directory, "tiny"));

    const path = try std.fmt.allocPrint(
        allocator,
        "{s}/greedy_ids.json",
        .{fixtures_directory},
    );
    defer allocator.free(path);
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const length = try file.getEndPos();
    assert(length > 0);
    const text = try allocator.alloc(u8, @intCast(length));
    defer allocator.free(text);
    const read = try file.readAll(text);
    assert(read == length);

    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        text,
        .{},
    );
    defer parsed.deinit();
    const list = parsed.value.object.get("generated_ids").?.array.items;
    try std.testing.expect(list.len == greedy_count);
    var result = GreedyIds{};
    for (list, 0..) |item, index| {
        result.ids[index] = item.integer;
    }
    result.count = @intCast(list.len);
    return result;
}

/// Directory of the golden fixtures, injected by `core/build.zig`.
const fixtures_directory: []const u8 = @import("fixtures_dir").path;
