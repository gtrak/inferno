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
