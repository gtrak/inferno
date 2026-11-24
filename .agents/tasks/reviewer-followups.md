# Reviewer follow-ups (non-blocking suggestions carried forward)

From the I-001 and I-004 review gates. Fold these into the issue named; do not
re-open the merged work.

## I-002 (hw probe) — from I-001 review
1. `core/src/c_api.zig` `inferno_probe_devices`: `assert(capacity > 0)` aborts
   across the FFI boundary; the documented contract is "return 0". Make
   `capacity == 0` return 0 without touching the buffer, keep truncation for
   `cap >= 2`, and always NUL-terminate.
2. Replace the `@export`ed data symbol `inferno_cubin_table` with a typed
   `pub fn cubin_table() *const [N]Cubin` accessor plus a `export fn
   inferno_cubin_count() u32`, so no exported symbol lacks length information.
3. `core/src/kernels/probe.cu` comment: claim only what the tests do
   ("nvcc -> cubin -> embed pipeline works"), not "driver-load" (that is I-002).

## I-005 (CPU reference) — from I-004 review
4. `tests/golden/README.md` records `torch.set_num_threads(1)`. Consequence for
   consumers: only compare **bit-exactly** under `set_num_threads(1)`; at default
   threads the reference itself differs by ~4.5e-8 (prefill) / ~3.0e-8 (decode),
   so use `rtol/atol ≈ 1e-6` in fp32. Write this into `core/src/ref/tests.zig`
   as a comment next to the tolerances.
5. `tests/golden/tiny/greedy_ids.json` does **not** exist yet. Generate it by
   adding a `--greedy` mode to `tools/ref-trace/trace.py` (16 greedy tokens from
   the golden input, `torch.set_num_threads(1)`, no sampling) and commit it; that
   file is the greedy-equivalence gate for I-005 and later for the GPU paths.
6. The trace contains 104 contract names + 16 extra diagnostic captures per
   section (120 per section, 240 total). Do not treat the extras as missing
   contract names.

## Project bookkeeping (orchestrator)
7. `plan.md` §5 layout said `src/cuda/`; the merged tree uses `src/hw/`.
   Follow the tree, not the old plan text.
