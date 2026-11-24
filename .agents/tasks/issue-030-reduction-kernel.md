# I-030 — device reduction / fused all-reduce kernel (and the expert-combine bandwidth it fixes)

Carried out of I-014b, where the all-gather design was chosen for bit-exactness and the cost was
measured rather than argued.

## Why it exists

I-014b uses **column-parallel + all-gather** for the dense TP path because it is *bit-exact*: each
rank accumulates a disjoint output slice over the **full K dimension**, so the accumulation order
matches the single-GPU kernel and no reduction is reassociated. That is the only way to hold gate 1
(16-token greedy equality) exactly.

The expert EP path has no such luxury. Each rank computes only its shard's selected experts, and
`moe_combine` needs every expert's contribution. I-014b **gathers** those outputs so the combine sums
them in a deterministic expert-index order on every rank. The cost:

| approach | traffic per token | at chunk 16,384, per layer |
|---|---:|---:|
| gather 10 expert outputs (5,120 B each) | **51.2 KB** | **~839 MB** |
| reduce, then broadcast the result | **5.1 KB** | ~84 MB |

**10x.** Fine on the tiny fixture (6 experts, hidden 128); unacceptable in production.

## Work

1. **A reduction kernel** in `core/src/kernels/reduce.cu` — sum a list of equally-shaped f32/bf16
   buffers into one, with a **specified summation order** (index order, not arrival order) so that a
   reader can reason about the result. No `atomicAdd`.
2. **A fused all-reduce** on the `comms` streams: reduce-scatter + all-gather, so a per-layer dense
   all-reduce costs one pass rather than two.
3. Wire the expert-combine path to use the reduce instead of the gather once it exists, and keep the
   gather behind a flag as the **bit-exact reference** for comparison.
4. Assert the traffic and the ordering: the reduced result must equal the gathered result **when the
   summation order is the same**, which is the property that makes the substitution safe.

Note: adding a `.cu` requires **no `core/build.zig` edit** — the cubin pass globs
`{ "src/kernels", "src/model" }` and embeds every `.cu` it finds.

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build test) ; echo CORETEST=$?
```

Non-vacuity: prove the order assertion fails if the reduction order changes, and paste the refusal.

## Non-goals

Do not change gate 1's 16 ids. If the reduce path cannot reproduce them, the reduce is a *production*
path validated against the fp32 oracle, and the gather stays the test path — say so explicitly rather
than adjusting the ids.
