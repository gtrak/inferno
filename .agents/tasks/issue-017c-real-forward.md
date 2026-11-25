# I-017c — real-weight forward: first coherent tokens

**Predecessors (done):** I-017a loads the real checkpoint on all four ranks with extents read back;
I-017b drives the reference `Plan` from the real config. Neither produces logits.

**Goal:** one forward pass over the *real* checkpoint's weights, greedy-decoded, producing tokens that
are finite, deterministic across two runs, plausible in magnitude, and — the only real quality
statement available at this stage — coherent enough to name itself.

There is **no golden token id for the real checkpoint** at this issue. Do not invent one and do not
compare against another GPU path. The numeric authorities in order are:

1. `core/src/ref/` (fp32 CPU oracle) with identical dtype-rounded inputs, for any single kernel.
2. The published bf16 trace (`jyohukuchan/qwen38-flash-next-bf16-logits`, 2048 positions) for
   end-to-end quality — that is I-031's grader, at **short context** (plan.md §7i).

## Re-scoped by the target state (`.agents/design/target-state.md`)

Loading is Rust's job now, so this issue must **not** grow a Zig-side real-checkpoint loader — that would
create exactly the second production path the target state forbids. What this issue owns is the other
half: **the executor's weight source becomes an interface**, so `exec.Model` can be fed either the tiny
fixture or a real weights view (dtype, tier, rank, offset) resolved by Rust. Incoherence #5 in the target
state is this issue.

## Where the seam actually is

`model/exec.zig` (1745 lines) is written for the tiny fixture: `fixtures.RefConfig` plus **f32**
weights uploaded by `model/load.zig` (1227 lines, f32 and a single bf16 exception). The real
checkpoint is a dtype zoo — this is the gap:

| component | real dtype | where it has to bind |
|---|---|---|
| attention / main linears | bf16 | `load.zig` has one bf16 path (indexer `index_qk_proj`); every other matrix is f32 |
| routed MoE experts | NVFP4 W4A4, group 16, MSE-calibrated | no upload path; kernels exist (`I-006a/B`, `OMMA.SF`) but nothing feeds them from the checkpoint |
| shared expert | bf16 | tensor-core path exists, upload path does not |
| PLE n-gram table | FP8, 47.7 GiB, **host-resident** | nothing reads it; it is the largest single object in the model |
| KV cache | FP8 (our placement choice) | pool exists (I-015); wiring to real weights is untested |

**Do not** try to make `load.zig` generic over dtype as step one. The tractable order is:

1. **Embedding → layer 0 → lm_head**, one token, real bf16 weights, experts bypassed. This proves
   offset arithmetic, dtype rounding, and device upload against real bytes. Logits will be wrong as
   *model output* (no experts, no PLE) but must be finite and deterministic.
2. **One full layer** of each kind: a linear-attention (GDN) layer and an indexed-attention layer.
3. **MoE from the hot tier**, NVFP4 experts dequantised by the existing kernels, cold tier stubbed.
4. **All 48 layers**, then PLE, then greedy decode of a short prompt.

## Hard constraints

- Expert tiering is **required, not an optimisation**: 123.53 GiB of weights against 61.1 GiB of VRAM.
  `mem/placement.zig` and `model/multi.zig` already model the hot/cold split; the forward must read
  through that, never assume a tensor is resident.
- **413 KB of headroom** was measured after the shard load. Any KV or workspace allocation in this
  issue has to come from the tier machinery or the run will fail at `cuMemAlloc` with no useful error.
- The checkpoint path comes from `INFERNO_MODEL_DIR` only. No default path, ever (gate-enforced by
  review, not by a check).
- `ple_layer_ids` are **1-based**: `[2]` is layer index 1, confirmed by the tensor names
  `model.language_model.layers.1.ple.*`. A 0-based reading silently moves the PLE block one layer.
- Bulk paths take an arena; `std.testing.allocator` on a bulk path is G3-blocked and >1250× slower
  (measured: 478 ms vs >600 s unfinished).
- Iterate with `INFERNO_GUARD_BANDS=0`; verify without it. `INFERNO_BENCH=1` for sweeps.
- The real-checkpoint tests stay behind `INFERNO_MODEL_DIR`; the 52 GB load behind
  `INFERNO_SHARD_LOAD=1`. A default suite run must never touch 132 GB.

## Acceptance

- Two runs of the same prompt produce **identical** token ids (determinism).
- Logits finite; `max|logit|` reported and in a sane range (single digits to low tens, not 1e3/NaN).
- Per-layer relative-RMS against the fp32 oracle for whichever layers are wired, reported as a number.
- The full 48-layer decode names itself or continues coherently; if it does not, report the layer at
  which drift starts (cosine is a **localiser only** — scale-invariant, saturates, 0.999+ on degraded
  models; §7f).
- Timing reported, not gated: load wall time, per-token cost, and which tier served each expert.

## Explicitly out of scope

Tokenizer (Rust owns it; feed token ids), the HTTP surface, speculative decoding, and any performance
work. First make it *right*, at whatever speed it runs.
