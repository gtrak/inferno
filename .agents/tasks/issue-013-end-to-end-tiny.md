# I-013 — single-GPU end-to-end on the tiny model: greedy ids bit-identical to HF

## Why this issue gates everything downstream
First time the whole graph runs on GPU. Before this passes, we do not know that our
reading of the architecture is right; after it, the remaining work is placement and
speed. Treat a mismatch as an architecture bug, never as "numerics".

## Working agreement
- Sole writer. tiger-style binding. Single GPU only (device 0) — multi-GPU is I-014.
- Consumes: `core/src/hw/` (driver layer), `core/src/weights/` (checkpoint reader),
  `core/src/kernels/` (all kernels), `core/src/ref/` (fp32 oracle), `tests/golden/tiny/`.
- You may create `core/src/model/` freely. Do not edit `core/build.zig`; add exactly
  one `_ = @import("model/mod.zig");` line to the test block in `core/src/c_api.zig`.

## What to build
`core/src/model/` — the Qwen4Exp text graph as a GPU executor:
1. `plan.zig` — a static execution plan: per-layer kernel sequence derived from
   `weights/config.zig` (`layer_types`, PLE layer index), with buffers sized once at
   init (tiger-style: no allocation after init).
2. `exec.zig` — the forward pass: embed gather → 4 layers (HC → attention/GDN → HC →
   MoE, plus PLE on layer index 1) → final HC mixer → `lm_head` → logits.
3. `state.zig` — the caches for one sequence: GDN conv + recurrent state, KV,
   indexer keys, n-gram context, PLE conv state.
4. `generate.zig` — greedy decode loop (argmax on device, no host logit transfer).
5. Load the tiny checkpoint (`tests/golden/tiny/checkpoint.safetensors`) and the
   reference weights the same way production will, i.e. through `weights/manifest.zig`.

## Gates
```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build test) ; echo CORETEST=$?
python3 - <<'EOF'
import json; d=json.load(open("tests/golden/tiny/greedy_ids.json"))
print("golden greedy:", d["ids"]); assert len(d["ids"]) >= 16
EOF
```
**Amended after a lane escalation:** the original gate named `(cd core && zig build tiny-check)`,
which contradicted this spec's own "do not edit core/build.zig" and, worse, is a command
`tools/gate.sh` never runs — it would have made the architecture gate a test that CI silently
skips. The check is therefore a **normal `test` block** reachable from `core/src/c_api.zig`'s
merged import list, so `bash tools/gate.sh` → `zig build test` executes it. No new build step.
It must **assert**, and report its wall time.

It must print and assert:
1. **Prefill logits** vs `tests/golden/tiny/trace.safetensors::final/logits`:
   `max_abs_err/max|ref| <= 4e-3` (bf16 GPU vs fp32 oracle).
2. **Greedy ids: EXACT equality** with `tests/golden/tiny/greedy_ids.json` for 16
   tokens, both (a) full-recompute each step and (b) with the incremental caches.
   Assert with `expectEqualSlices` and put the first mismatching index and both ids in the
   failure message — a printed id list is not a gate.
   Doubling as a cache-correctness check: (a) and (b) must also agree with each other.
3. **Non-vacuity**: print the 16 ids, the number of distinct ids (must be >1), the
   max |logit|, and the KV length after generation (must equal prompt+16).
4. Report per-phase wall time (embed / attention-block / MoE / lm_head) and total
   prefill + decode ms, for information only.

## Report format
Implemented / Changed files / Validation (the three checks, verbatim) / Open risks
(any place the GPU path and the oracle disagree in a way you had to investigate) /
Recommended next step.

## Resume state (2026-11-24, first attempt timed out at 90 min)

The first attempt did **not** hang — it spent the full budget on design and scaffolding and
died mid-`core/src/model/load.zig`. Work that exists in the worktree and is worth keeping:

| file | lines | state |
|---|---|---|
| `core/src/model/plan.zig` | 78 | execution plan, appears complete |
| `core/src/model/glue.cu` | 262 | layer wiring kernels |
| `core/src/model/glue.zig` | 856 | host side; **fails `zig fmt --check`** |
| `core/src/model/load.zig` | 385 | written last, never compiled |

Also decided during that attempt and worth not re-deriving: the tiny config has
`rotary_dim=8`, `index_dim=16`, so rope halves do not match the production shape —
`validateMisc` must not assume the 2560/256/0.25 production ratios.

**What remains, in order:**
1. `zig fmt core/src/model/glue.zig` and get the whole tree to compile
   (`bash tools/gate.sh` currently fails at the first step, `zig fmt --check`).
2. `_ = @import("model/...zig");` — one line in `core/src/c_api.zig`'s merged test block.
3. The end-to-end test: 16 tokens, assert `expectEqualSlices` against
   `tests/golden/tiny/greedy_ids.json`, both full-recompute and incremental-cache paths.
4. Match the ids. If they mismatch, bisect per layer against the trace tensors and report
   the first divergence — do not adjust a kernel to make ids agree.

If this times out again, the next attempt should be given a **2.5 h budget** and start at
item 1, not at design: 1,581 lines of wiring already exist.

## Exact remaining blocker (diagnosed 2026-11-24, second timeout)

The tree is at **4,676 lines** across `model/{plan,glue,load,exec,state,generate,tiny_check}.zig`,
`c_api.zig` is wired, and **185/186 tests pass**. Exactly one test fails, and it is the
end-to-end one:

```
error: 'model.tiny_check.test.the tiny model reproduces the golden logits and the golden
greedy ids' failed: config: cannot parse: MissingField
```

**Cause, verified against the real checkpoint:** `~/models/qwen38-flash-next-nvfp4/config.json`
nests the model under `"text_config"` and carries **both** `full_attention_interval: 4` and a
48-entry `layer_types`. `tests/golden/tiny/config.json` is **flat** (no `text_config` wrapper)
and has `layer_types` but **no** `full_attention_interval`. The production parser
(`weights/config.zig`, whose doc comment shows the nested shape) therefore rejects the fixture.

**Do this, and only this:**
1. In `tiny_check.zig`, wrap the flat fixture as `{ "text_config": { ... } }` before parsing,
   and derive `full_attention_interval` from `layer_types` (both are present in production, so
   this is faithful, not a fiction). Put a comment saying the fixture is flat and why the test
   adapts rather than the loader.
2. **Do not change `weights/config.zig`** and **do not change the committed fixture** — the
   fixture regeneration is follow-up work and must not block the architecture gate.
3. Then run the ids. If they mismatch, bisect per layer against `trace.safetensors` and report
   the first divergence; do not adjust a kernel to make them agree.

**Budget note:** two 90-minute attempts have died on this task, both after real progress, and
neither was stuck. Start at step 1 above. Do not re-derive the wiring, the plan, the loader or
the glue — they exist and 185 tests pin them.

## Prefill numbers in the tree are NOT valid measurements

The lanes ran alongside other GPU work. Any throughput printed by these lanes is indicative
only; M-001 is the measurement of record.

## SUPERVISOR RULINGS on the rev-e2e-p2 blocking findings — 2026-11-24

Read this before acting. Two of the six findings are **mine**, not the lane's, and must not
consume your clock.

**P1-4 (`zig build tiny-check` does not exist) — MOOT, my spec defect.** The gate was amended
before this review; the reviewer read a stale copy. The end-to-end check is a normal test block
wired through `c_api.zig`, run by `zig build test`, which `tools/gate.sh` already runs. **Do not
add a build step.** Adding one would put the architecture gate behind a command CI never calls.

**P1-5 (`core/build.zig` was edited in breach of the fence) — AUTHORISED, and it was me.** I
generalised the cubin pass to walk `kernel_directories = { "src/kernels", "src/model" }` so that
`src/model/glue.cu` becomes a cubin; before that, `glue.zig` could only ever return
`GlueCubinMissing`. The table is verified to contain `glue`. Do not revert it, do not re-add the
step, and do not re-litigate the fence.

**P1-6 (`weights/manifest.zig` cannot load the tiny fixture) — ACCEPTED DEVIATION, no action.**
The requirement is unsatisfiable: the fixture has no `model.safetensors.index.json`, and
`weights/config.zig::hyperConnectionWidth` rejects the tiny shape (256 vs 160). Loading through
`safetensors.zig` and documenting the deviation in-tree is the correct resolution. **Do not
weaken `manifest.zig` or `config.zig` to accept the fixture** — that would make the production
loader accept a shape it must reject. Report it as residual risk.

**P1-3 (vacuous `max_logit > 0.0` over `reference`) — FIX IT.** Iterate `produced`. This is the
same vacuity class that has now appeared six times in this project; an assertion that can pass on
an all-zero model is not a test.

### The two that are actually the work, in priority order

**P1-2 — nondeterminism first.** `forward` gives max_abs_err 1.423e-2 / 7.814e-3 / 9.803e-3
across runs of the *same tree*. Until that is fixed the greedy-id gate cannot be a gate at all.
Diagnose it, do not paper over it: run the forward **twice in one process on identical inputs**
and compare **layer by layer**, reporting the first layer whose output differs between the two
runs. That localises it to a kernel or a launch. Prime suspects: a kernel launched without a
stream dependency, a workspace/device buffer reused across layers without reinitialisation, an
uninitialised allocation, or a split reduction whose order varies.

**P1-1 — prefill logits 4.402e-2 against a 4e-3 gate (~14.6 bf16 ulp).** Only after P1-2 is
resolved, because a nondeterministic forward cannot be measured to one number. If it persists
once deterministic, bisect per layer against `trace.safetensors` and report the first divergent
tensor and its magnitude — **do not adjust a kernel to make logits agree.**

## OPEN BUG found by the diagnostic: `reset()` is not idempotent

While hunting the residual nondeterminism the lane added a temporary
`Model.diagnoseReproducibility` that runs three full forwards from a reset and prints the
residual-stream drift per phase. Its own output:

- the **first** reset+forward is correct (`ulp_error 0.07`),
- the **fourth** reset+forward diverges at **`ulp_error 88.23`**, max_abs_err 1.723e-1.

The 20/20 clean runs did not catch this because each test process resets once. This is a
**serving-path bug**: every request resets, so request 4 onward would be silently wrong. It is
almost certainly the same mechanism as the earlier "reset zeroes the mRoPE tables" defect
returning by a different route — something `reset` clears is not restored on the second pass.

Do not work around it. Reproduce it directly: `reset(); forward(); reset(); forward();` and
compare the two forwards. Then find which buffer differs between pass 1 and pass 2 and fix the
reset (or the fill-after-reset), do not special-case the test.

The diagnostic call sites have been removed from the measured path so the gate reflects the
shipping behaviour; the function itself is kept for the reproduction.

## RESOLVED: `reset()` was not idempotent — chunked copy never advanced its destination

Found by repeat-run diagnosis, localised by hashing buffers across passes, fixed at the mechanism.

`ple.Block.reset()`'s chunked H2D loop copied **every chunk to offset 0** of `conv_state`. The
tiny fixture's staging window is 8,192 B (`64 tokens x 2 heads x 16 dims x 4 B`) while
`conv_state` is 12,288 B, so iteration 1 wrote `[0,8192)` and iteration 2 re-wrote `[0,4096)`;
**`[8192,12288)` was never written.** Probe after a reset: exactly 256 nonzero floats, the first
at index 2048 = byte offset 8192 = the ring's newest column. Because `plan.ple_layer == 1`, that
stale column was convolved into the first rows of the PLE layer on every pass after the first:
pass 1 read a fresh ring, pass 2+ read pass 1's column and agreed with each other — which is why
the first `reset()`+`forward()` was correct and the fourth diverged at ulp 88.23.

The fix advances the destination by `written` and clears **exactly the same byte count** as
before, so this is an offset fix, not a widening. Verified non-vacuous: a reviewer swapped only
`ple.zig` between HEAD and working tree in a `/tmp` copy and reproduced the failure to the digit
(`logit 0 is -5.080317e-2 then -2.5817446e-2`).

**This is the third bug of the same family tonight** — `reset` clearing a container but not the
meaning of its contents:

| bug | what cleared | what did not |
|---|---|---|
| mRoPE cos/sin tables | the buffers | the refill after clearing |
| PLE block buffer | the allocations | the block's index state |
| `conv_state` (this one) | part of the buffer | the chunk offset, so 4 KiB survived |

## OPEN: `reset()` still zeroes every allocation including the KV cache

`State.reset` zeroes all tracked allocations, which at 262,144 tokens is the KV cache: **3 GiB per
rank, ~7.5 ms at the measured 397.6 GiB/s**, against a 5.3 ms decode step. Invisible on the tiny
fixture (a few MB). Real data should be written before read; only true padding needs zeroing, once
at allocation. Zeros are indistinguishable from valid data, so this masks rather than fixes — the
same class as the `max_logit > 0.0`-over-reference vacuity.

Also: `logits_ulps_max = 16.0` now sits **230x above the measured 0.07 ulp**, so it can never bind
and the effective gate is the relative 4e-3. Tighten it to a value that binds (~2 ulp) or drop it
and keep one gate. A gate that cannot fail is decoration.
