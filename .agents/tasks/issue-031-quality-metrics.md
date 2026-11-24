# I-031 — model-quality measurement: KL, ΔPPL, and per-layer drift

**Why this exists.** Every gate in this project is *kernel validation*: a kernel against the fp32
oracle with identical inputs, in ulp or relative terms. That answers "does this kernel implement the
same math." It cannot answer **"is the approximating pipeline still the same model"** — which is
exactly what FP8 KV, NVFP4 experts, and a TP split put at risk.

The quantisation ceiling (FP8 weights, FP8 KV, native 4-bit experts) was settled by *argument*, not
measurement. This issue makes it measurable, and gives the project a way to answer "did that change
cost quality" with a number instead of an opinion.

## The reference problem, stated first

ΔPPL and KL both need a **full-precision reference**, and we do not have one: the checkpoint on disk
is already quantised, and a bf16 copy is ~265 GB. So:

**Generate one reference trace, once, and cache it.** Run HF transformers over the real checkpoint
with CPU/offloaded weights for a **fixed prompt set** and a **fixed number of positions**, and store
logits as a safetensors trace alongside the prompts. Slow once, free forever, reproducible. The
machinery already exists in `tools/ref-trace/` (it produced the 240-tensor tiny trace).

**Also measure the reference's own noise floor** — run the HF reference twice, once with
`torch.set_num_threads(1)` and once at default. The difference is the irreducible instability of the
reference itself, and it is **the correct scale for the gate**. We learned this the hard way: an
earlier tolerance was set at 1e-6 when the reference legitimately moved 4.5e-8 with thread count.
A quality gate whose threshold is tighter than the reference's own variance is decoration.

## Metrics, and what each is for

| metric | role | notes |
|---|---|---|
| **KL(p_ref ‖ p_ours)** per position | **primary grader** | on temperature-scaled or top-k-clipped distributions — raw full-vocab KL over 248,320 tokens is dominated by tail noise. Report mean, p50, p99 and max with its position |
| **ΔPPL** on the fixed prompt set | headline number for the record | coarse but it is the number anyone outside this project will ask for |
| **per-layer relative RMS** `‖x̂−x‖/‖x‖` | **primary per-layer metric** | NOT scale-invariant, so it actually degrades. This is the honest layer number |
| **router top-k agreement rate** | MoE quality | fraction of positions picking the same 10 experts as the reference. 4-bit expert quality risk is mostly misrouting |
| **QSA selected-key overlap** | attention quality | we already compute `set_difference`; here it becomes a rate at 4K/32K/128K/256K |
| **per-layer cosine similarity** | **diagnostic only — do not gate on an absolute value** | scale-invariant and saturating: a quietly degraded model still reads 0.999+. Gate only on *regression* below a recorded baseline |

**Cosine caution, written down so nobody re-litigates it:** 0.99 → 0.5 is a screaming signal and
worth failing; 0.9995 → 0.9985 is not, and treating it as a quality score will make a broken run look
excellent. KL is the grader; cosine is the localiser.

## Deliverables

1. `tools/ref-trace/` extension: emit a **real-model** logits trace for a fixed prompt set
   (prompt ids, logits per position, the config, the HF version, the thread count) plus the
   twice-run noise floor. Document how to regenerate it and why it is expensive.
2. A test (not a script — it must run under `zig build test`) that loads our logits and the reference
   trace and reports the table above, asserting:
   - **KL mean and p99 within the noise floor budget**, where the budget is derived from the two-run
     reference spread rather than invented;
   - **ΔPPL within a stated bound**;
   - **router top-k agreement above a stated floor**;
   - **relative RMS per layer** below a stated bound;
   - cosine **recorded**, with a regression bound only.
3. A way to run the same measurement per precision choice, so the ceiling can be revisited with data:
   FP8 KV vs BF16 KV, and (if ever revisited) FP8 vs NVFP4 experts. This is the experiment that
   turns the M3 rejection from principle into evidence.

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build test) ; echo CORETEST=$?
```

Non-vacuity: every threshold must be shown to fail at least once. Perturb a logit vector by a known
KL and paste the refusal; do not assert a bound nobody has seen fire. Two gates tonight were 1,000x
and 230x too loose to ever bind.

## Non-goals

- No task benchmarks (MMLU, HumanEval) in this issue. KL and ΔPPL are the right first instruments.
- Do not touch any existing kernel tolerance or oracle comparison. This is a *second layer* of
  assurance, not a replacement for kernel validation.
- Do not compare against another inference engine's output as if it were a reference. The reference
  is HF with the checkpoint's own semantics, produced by us, cached, with its noise floor recorded.
