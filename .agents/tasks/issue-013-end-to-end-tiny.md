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

## OPEN: residual nondeterminism (2026-11-24, gauntlet evidence)

Determinism is **NOT** fixed. Sixteen repeat runs of one binary produced:

```
ulp_error 6.75  x14
ulp_error 6.70  x 1   (/tmp/run_9.log)
ulp_error 5.91  x 1   (/tmp/run_8.log)
```

and a later invocation showed `ulp_error 7.79, max_abs_err 1.522e-2` while all sixteen logs carry
`max_abs_err 7.652e-3`. The greedy ids are identical in every run, so the ids gate holds; the
*forward* still varies. Flake rate ~12 %.

The 1.4 MB score-buffer overrun and the missing `applyRope` fence were real and are fixed, but
they were not the whole story.

### Do these in order, cheapest first

1. **`CUDA_LAUNCH_BLOCKING=1` on the e2e test.** One env var, no code change. If the spread
   vanishes, the remaining bug is **host-side async ordering** — another unordered dependency
   between the executor stream and a module stream. If it persists, it is memory (uninitialised
   read, a second sizing mismatch) or an intra-kernel race.
2. **`compute-sanitizer --tool racecheck`** — catches shared-memory and warp-sync hazards inside
   kernels, and names the kernel and line. This is the one tool never run on this path.
   Then `--tool memcheck`. Note `memcheck` reported 0 errors despite a 1.4 MB overrun, because
   the write stayed inside a larger allocation, so its silence proves less than it looks.
3. **Bisect by synchronisation.** If step 1 implicates ordering, add an explicit
   `cudaStreamSynchronize` after every launch behind an env flag, confirm determinism, then
   re-add syncs one at a time until the spread returns. That localises the missing dependency
   to a single launch pair — do not guess which one.

### Verification rule

Flake rate is ~12 %, so three runs cannot clear a fix. Verify the final configuration with
**>= 20 runs once**, record the count and every distinct value, and do not repeat the loop per
gate afterwards. See the verification-effort rule in `plan.md`.

### Do not

- Do not widen the ulp ceiling (16) to absorb the spread. The spread is the bug.
- Do not touch `core/build.zig`, add a `tiny-check` step, or relax any tolerance.
- Do not "fix" it by adding a blanket `cudaDeviceSynchronize` and calling it done — that hides
  the dependency rather than stating it, and it would serialise the real pipeline.
