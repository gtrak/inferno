//! `core/src/model/` — the Qwen4Exp text graph as a GPU executor.
//!
//! The graph is split the way the work is: `plan.zig` decides the layer
//! sequence once, `load.zig` puts the checkpoint on the device, `state.zig`
//! owns the per-sequence caches, `exec.zig` walks one forward, `generate.zig`
//! runs the greedy decode loop, and `glue.zig` carries the handful of kernels
//! the block kernels take as given. `tiny_check.zig` is the end-to-end check
//! (issue I-013): prefill logits against the committed trace, then greedy ids
//! against the committed ids.

test {
    _ = @import("plan.zig");
    _ = @import("load.zig");
    _ = @import("glue.zig");
    _ = @import("state.zig");
    _ = @import("exec.zig");
    _ = @import("generate.zig");
    _ = @import("tiny_check.zig");
}
