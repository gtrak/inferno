# I-027 — SIMD the fp32 CPU oracle (`@Vector`)

**Why:** `core/src/ref/` is the oracle every kernel is validated against, and it is scalar.
`ref/ops.zig:267` is `for (left, right) |a, b| { total += a * b; }`. `refcheck` pushes 206
traced tensors through it on every gate run, and every kernel lane's oracle comparison uses
the same code. Zig has had `@Vector` the whole time and the tree uses it zero times
(`grep -c '@Vector' core/src -r` → 0). This is CPU-only work: it does not touch a GPU, so it
is safe to run while other lanes hold the devices.

## Scope

`core/src/ref/ops.zig` and, only where it clearly pays, `ref/attention.zig`, `ref/gdn.zig`,
`ref/moe.zig`, `ref/rope.zig`, `ref/hc.zig`.

Convert the reduction and elementwise inner loops to `@Vector(8, f32)` (or 16 where the
measurement says so) with explicit horizontal reduction at the end. Keep the code readable:
one helper for a strided dot, one for an elementwise map, rather than a macro soup.

## The hard constraint: the oracle's answers must not move

A SIMD reduction reassociates the sum, so f32 results **will** differ in the last bits unless
you deliberately preserve order. That is acceptable only within the existing tolerances, and
you must *prove* it rather than assert it:

1. `(cd core && timeout 900 zig build refcheck) ; echo REFCHECK=$?` → `REFCHECK=0`,
   0 failing tensors, and report the **worst scaled error before and after** (it is
   5.6e-7-class today).
2. Report the **maximum relative delta between the scalar oracle and the vector oracle**
   across all 206 traced tensors. If any tensor moves by more than 1e-6 relative, stop and
   report it — do not "fix" it by widening a tolerance.
3. **Do not change any tolerance value** in `ref/tests.zig` or anywhere else. Not one digit.
4. Anything that must stay bit-exact (packing, quantisation code copies, index math) stays
   scalar. Say which parts you deliberately left scalar and why.

## Report

- Wall time of `zig build refcheck` before and after, median of 3.
- Wall time of `zig build test` before and after.
- The `@Vector` count in the tree before and after.
- Which functions you vectorised and the per-function speedup where you can measure it.

## Gate

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build test) ; echo CORETEST=$?
(cd core && timeout 900 zig build refcheck) ; echo REFCHECK=$?
python3 -c "import subprocess,sys; n=subprocess.run(['grep','-rc','@Vector','core/src'],capture_output=True,text=True).stdout; print(n)"
```

## Accepted exception (supervisor, 2026-11-24)

The 1e-6 relative bar above was **exceeded** and the exception is granted, on the record:

- Worst per-tensor **scaled** delta: **1.753e-6** (decode/layer0/gdn/core_out). On that tensor
  max|ref| = 7.47e-5, so the absolute movement is **1.31e-10** — the "large" relative figure is
  an artefact of small magnitudes, and it lands on cancellation-amplified GDN l2norm/delta
  tensors, exactly where reassociation bites hardest.
- Worst absolute movement over all 193 float tensors: 1.907e-6 (layer0/gdn/g, max|ref| 20).
- **The comparison that matters improved:** worst scaled error against the HF golden went
  1.363e-6 → **6.901e-7**, 0 failing tensors, no tolerance touched.
- Greedy ids remain **bit-identical** (16/16), so the architecture gate is unaffected.

Reasoning: the scalar serial order is not ground truth — HF's matmuls run through cuBLAS with an
unspecified accumulation order, so an 8-lane reassociation is arguably closer to the compliant
path. Reverting (bit-exact but 5-8x slower) chases a fiction. A shape-dependent hybrid (SIMD only
for rows >= 64) was **rejected**: an oracle whose numerics depend on tensor length produces
failures nobody can reason about.

**Conditions carried into the code:** the ~9 % consumption of the tightest kernel budget (2e-5)
is stated in a comment at `reduceDot`/`reduceSquares`, and the kernel-observed margins before and
after are reported so no kernel silently went from comfortable to marginal.

**Lesson for future bars:** state them relative to the tensor's own scale, or state them absolute.
"relative to max|ref|" and "absolute" differ by five orders of magnitude on small-magnitude
tensors and I wrote one where I meant the other.
