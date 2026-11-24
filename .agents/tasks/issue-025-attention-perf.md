# I-025 — attention performance + exactness additions (carried from I-008)

Scope that main's `bfcc488` amendment made binding "for the follow-up". Judged against
`core/src/ref/` and the committed goldens exactly as every other kernel lane.

## Baselines as delivered (measured, I-008 round 3)

| item | measured | target |
|---|---|---|
| `attn_prefill` (rows=256, count=2048) | **36.6 µs/token/layer** (9361 µs/launch) | ≤ 5 µs/token/layer |
| indexer steady state | **245.7 µs/launch → 2.95 ms/token** = block_scores 73.1 + select 102.6 + project/norm/append 70.0 | ≤ 0.6 ms/token |
| indexer cold cache | 543.8 µs/launch → 6.5 ms/token | per-prefill, looser bound |
| indexer scores buffer | `M × n_blocks` fp32 = **1.07 GB** at M=4096, 256K | context-independent |
| `gathered_k` / `gathered_v` | two separate arrays | one interleaved array |
| prefill QSA at rows > 1 | one shared union working set, causally masked only | exact per-row selection |

## What to build

1. **MMA prefill attention.** Warp-scalar at 27.5→36.6 µs/token/layer misses the 5,000 tok/s
   prefill target (200 µs/token budget) by 7x. The arithmetic is only 0.60 GFLOP/token, so
   at the measured 46.3 TFLOPS/GPU fp8 rate this is ~3 µs/token. Use
   `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` on the already-fp8 gathered K/V —
   no dequantise step. Gate: **≤ 5 µs/token/layer** at the production shape.
2. **Per-row selection bitmap over the union gather** so prefill is exactly QSA at rows > 1.
   Per-row gathering is infeasible (4096 rows × 2048 keys × 1 KB = 8.4 GB per layer per
   chunk), so keep the union gather and mask. **Print the union/budget ratio** so chunk size
   can be tuned toward 1.
3. **Streaming top-k for the indexer** so the scores buffer is context-independent:
   process block groups (e.g. 8192), keep a running per-row partial top-`block_topk`, merge.
   **Assert peak score-buffer bytes do not grow with context.**
4. **Interleave K and V** into one `[count, 2, kv_heads, head_dim]` working set: one H2D,
   single addressing.
5. **Indexer parallelism** (from the I-008 diagnosis, recorded next to the timing constant):
   - `indexer_select` is a **single CTA** scanning 65 536 scores at ~20 GB/s (102.6 µs).
     Fix: two-phase radix (per-CTA partial histograms + partial top-k merged by a second
     kernel). **Ordering must stay descending score with ties to the higher block index, as
     `torch.topk`; per-CTA `take` counts must sum to `block_topk`; the boundary must be
     narrowed from the merged candidate set, not per-CTA.** A previous attempt broke this
     and produced `set_difference 2824 of 16396`.
   - `indexer_block_scores` already scores 16 blocks per CTA (93.4 → 73.1 µs); keep it.

## Non-goals (explicitly I-014, not here)

Sharding block scoring across ranks (0.473 → 0.118 ms/token at 256 K) is a distributed
concern and belongs to I-014. Do not implement it here; the single-GPU floor of
**0.473 ms/token** stands.

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build test) ; echo CORETEST=$?
(cd core && timeout 900 zig build refcheck) ; echo REF=$?
```
1. Exact QSA at rows > 1: selection set equality against `ref/attention.zig` (cache_dtype
   mode) at rows ∈ {1, 64, 256}, with the bitmap active. Report union/budget per case.
2. Numerics unchanged: the existing prefill/decode/`decode@p == prefill row p` gates stay
   green at their existing tolerances. **No tolerance may be widened.**
3. Perf: prefill ≤5 µs/token/layer; indexer ≤0.6 ms/token steady state (or achieved +10 %
   as a regression guard, with the gap reported and the remaining fix named).
4. Peak indexer score-buffer bytes printed at 4 K / 32 K / 256 K and asserted flat.
5. `compute-sanitizer --tool memcheck` over the core test binary: 0 errors.

## Report format

Implemented / Changed files / Validation (perf table + exactness + buffer bytes) /
Open risks / Recommended next step.
