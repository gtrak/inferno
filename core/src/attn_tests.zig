//! Root of the `attntest` step: the indexed-attention harness of issue I-008.
//!
//! The tests live beside the wrappers they check (`kernels/attention.zig`) and
//! are already part of `zig build test` through `c_api.zig`; this root exists so
//! that `zig build attntest` runs that one harness on its own, without
//! compiling the rest of the test graph. It sits in `src/` rather than next to
//! the file it pulls in because a module may not reach outside its root's
//! directory, and the harness imports `../hw/` and `../ref/`.

test {
    _ = @import("kernels/attention.zig");
}
