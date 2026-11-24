//! The fp32 CPU reference implementation of the Qwen4Exp text graph.
//!
//! This is the executable specification of the model: every GPU kernel in
//! inferno is validated against it (and, through it, against the Hugging Face
//! golden traces). It is deliberately plain: scalar f32 math, one buffer per
//! intermediate, no fusion, no threads. Speed is irrelevant; agreement with
//! `tools/ref-trace/reference/modeling_qwen4_exp.py` is the only goal.
//!
//! Every buffer is allocated once, in `Scratch.init`, from `RefConfig`; no
//! forward allocates.

const std = @import("std");

const assert = std.debug.assert;

pub const fixtures = @import("fixtures.zig");
pub const ops = @import("ops.zig");
pub const conv = @import("conv.zig");
pub const rope = @import("rope.zig");
pub const hc = @import("hc.zig");
pub const moe = @import("moe.zig");
pub const gdn = @import("gdn.zig");
pub const attention = @import("attention.zig");
pub const ple = @import("ple.zig");
pub const layer = @import("layer.zig");
pub const model = @import("model.zig");

/// Every buffer the reference needs, sized from the config at init.
pub const Scratch = model.Scratch;

/// Allocates the scratch buffers for `config`.
pub fn init_scratch(
    allocator: std.mem.Allocator,
    config: *const fixtures.RefConfig,
) !Scratch {
    assert(config.hidden_size > 0);
    assert(config.max_position_embeddings > 0);

    const scratch = try Scratch.init(allocator, config);
    assert(scratch.hyper.len > 0);
    return scratch;
}

/// Releases the scratch buffers.
pub fn deinit_scratch(
    scratch: *Scratch,
    allocator: std.mem.Allocator,
) void {
    assert(scratch.hyper.len > 0);
    assert(scratch.logits.len > 0);

    scratch.deinit(allocator);
}

test {
    _ = fixtures;
    _ = ops;
    _ = conv;
    _ = rope;
    _ = hc;
    _ = moe;
    _ = gdn;
    _ = attention;
    _ = ple;
    _ = layer;
    _ = model;
}
