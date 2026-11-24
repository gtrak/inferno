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
(cd core && timeout 1800 zig build tiny-check) ; echo TINY=$?
python3 - <<'EOF'
import json; d=json.load(open("tests/golden/tiny/greedy_ids.json"))
print("golden greedy:", d["ids"]); assert len(d["ids"]) >= 16
EOF
```
`tiny-check` must print and assert:
1. **Prefill logits** vs `tests/golden/tiny/trace.safetensors::final/logits`:
   `max_abs_err/max|ref| <= 4e-3` (bf16 GPU vs fp32 oracle).
2. **Greedy ids: EXACT equality** with `tests/golden/tiny/greedy_ids.json` for 16
   tokens, both (a) full-recompute each step and (b) with the incremental caches.
   Doubling as a cache-correctness check: (a) and (b) must also agree with each other.
3. **Non-vacuity**: print the 16 ids, the number of distinct ids (must be >1), the
   max |logit|, and the KV length after generation (must equal prompt+16).
4. Report per-phase wall time (embed / attention-block / MoE / lm_head) and total
   prefill + decode ms, for information only.

## Report format
Implemented / Changed files / Validation (the three checks, verbatim) / Open risks
(any place the GPU path and the oracle disagree in a way you had to investigate) /
Recommended next step.
