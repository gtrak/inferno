# I-026 — decode attention and indexer: the serial bottleneck

**Status:** open. **Priority: above I-014**, because it is 72 % of the per-token budget and
because I-014's sharding multiplies whatever this produces.

## Why this exists

The I-025 pass hit its prefill target (36.6 → 4.82 µs/token/layer) and cut the score buffer
from 1.07 GB to 128 KiB/row flat. But summing the *measured* decode components gives

    7,654 us/token = 131 tok/s at 4K context

which is below the 150 tok/s target before context grows. 72 % of it is serial:

| component | us/token | layers | per launch |
|---|---|---|---|
| indexer | 3,183.8 | 12 | 265.3 |
| attn_decode @ pos 4095 | 2,349.6 | 12 | 195.8 |
| gdn | 731.0 | 36 | 20.3 |
| experts (arithmetic) | 1,300.0 | 48 | — |
| ple gather p50 | 90.0 | 1 | 90.0 |

Attention/indexer are **replicated across the 4 ranks** (a deliberate I-008 placement choice
that removed the attention all-reduce). The consequence is that TP does **not** divide this
5.5 ms. This issue attacks the number itself; I-014 shards it.

## Item 1 — attn_decode is ~10x too slow (highest suspicion)

Measured `attn_decode count=2051 pos=4095: 195.8 us per launch` (median of 7 samples of 4).

QSA caps a decode step at **2048 selected keys**. So the work is a gather+dot of
2048 keys x 2 kv heads x 256 head_dim ≈ 1.05M MACs per query row, plus a softmax. That is
single-digit-microsecond work at any sane rate; 195.8 us implies ~5 GFLOP/s equivalent.

**Find the cause before optimising.** Candidates, in the order I would check:
1. The 2048 bound is not actually enforced on this path, so it is reading a much larger
   fraction of the 4096-token context (check the selected-count actually passed to the kernel).
2. The gathered working-set copy is being rebuilt per layer per token instead of incrementally.
3. Non-coalesced gather of 256-float rows per key (one 1 KiB row per thread block is fine;
   one row per warp reading 4-byte elements is not).
4. Occupancy: a single-CTA-per-query-row launch at grid=1 leaves 3 GPUs' worth of SMs idle on
   the replicating rank.

Report the per-launch breakdown the way the indexer already does
(`block_scores + select + project/norm/append`), so the next pass has a target.

## Item 2 — indexer regressed 2960.7 → 3180.6 us/token

The I-025 streaming top-k made the buffer context-independent but cost 7 % wall clock, and
the guard was raised to achieved+10 % (3250 → 3500) to admit it. That was the correct short-term
call — a perf gate must not fail on run-to-run noise — but the regression must be recovered.

Measured breakdown at 65536 blocks: `block_scores 76.2 + select 120.8 + project/norm/append 68.3
= 265.3 us/launch`. The zero-allocation target is an **incremental** top-k: the running partial
is maintained per token, so a decode step only scores the *new* block and merges, rather than
re-scoring the whole 65536-block set each step. That is the difference between O(context) and
O(1) per token, and it is the same trick already approved for the pooled indexer keys.

Non-negotiable: selection must stay **exactly** equal to `ref/attention.zig` `.cache_dtype` at
context 4K/32K/256K, `set_difference == 0`, with the documented tie rule (higher block index
wins, matching `torch.topk`). Speed that changes the selected set is a correctness bug.

## Item 3 — the numbers must be measured at 256K, not extrapolated

Every figure above is at 4K. Before this issue closes, report decode step time at
context 4K / 32K / 128K / 256K for attn_decode and the indexer, and state which ones are flat
by construction and which grow. If any component grows with context at 256K, say so with the
number — that is a findings-level fact, not a footnote.

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build test) ; echo CORETEST=$?
(cd core && timeout 900 zig build refcheck) ; echo REFCHECK=$?
```

Any perf assertion you add must be **proven non-vacuous**: set the limit below the achieved
value, watch it fail, paste that output, then restore. A gate nobody has seen fail is decoration.

## Non-goals

- Sharding across ranks (I-014). Do not do it here; just report numbers that sharding will divide.
- Changing the placement decision (replicated attention). Re-opening it needs a design doc.
- Quantising the indexer keys further.
