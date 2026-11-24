# I-015 — single-sequence runtime: arena, cancel/preempt, admission

## Read first
`.agents/design/full-context-single-stream.md` §1: **concurrent decode is not
required.** This issue replaces the original "scheduler + continuous batching +
prefix cache" scope with a single-sequence runtime. Do not build a batcher.

## Working agreement
- Sole writer. tiger-style binding.
- Do not edit `core/build.zig`; add exactly one `_ = @import("sched/runtime.zig");`
  line to the test block in `core/src/c_api.zig`.

## What to build
`core/src/sched/runtime.zig` + `core/src/mem/arena.zig`:
1. **One KV/state arena** sized from the requested context at submit time; growth
   policy that either fits or is refused before the first token (no mid-request OOM).
2. **Admission**: at most one generating sequence; others queue (FIFO) or are
   rejected with a typed error. A queue-depth metric.
3. **Cancel**: cooperative, checked at every layer boundary; frees the arena.
4. **Preemption**: a long prefill must be chunked so a queued request's TTFT does not
   starve — chunk size configurable.
5. Context-length accounting: refuse a prompt that cannot fit, with the exact numbers
   in the error.

## Gates
```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 900 zig build runtime-check) ; echo RT=$?
```
1. Submit a sequence, cancel it mid-decode, assert the arena is returned to the pool
   (byte-exact free-list accounting) and the callback reports a final error token.
2. Submit two sequences; assert the second is queued and starts only after the first
   finishes or is cancelled; assert no interleaving of token callbacks.
3. Request a context larger than the arena ⇒ typed refusal before any compute.
4. Non-vacuity: print the arena free bytes before/after (must return to the initial
   value) and the queue-depth metric transitions.

## Report format
Implemented / Changed files / Validation / Open risks / Recommended next step.
