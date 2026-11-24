# Design target: full context, single stream (user directive 2026-11-24)

Targets: **full 262,144-token context**, **single-stream decode ≥ 150 tok/s**,
**prefill ≥ 5,000 tok/s**. Concurrent decode is *not* required if single-stream is
fast enough. KV may be quantised if the trade is right.

## 1. What this simplifies

- Decode needs no batch axis: every decode kernel specialises to M=1 (GEMV /
  `OMMA.SF` at M=16 padding only if profitable), no grouped-GEMM padding cliffs,
  no admission control beyond queueing, **one contiguous KV arena** instead of a
  paged pool. I-015 shrinks from "scheduler + continuous batching + prefix cache"
  to "one sequence, one arena, cancel/preempt".
- **The DRAM expert tier becomes structural, not optional** (§3). With one stream
  the routing distribution is maximally skewed, which is the regime where a
  hot-expert cache works best.
- Speculative decoding stops being a stretch goal and becomes the natural way to
  buy extra speed (single stream is exactly its best case).

## 2. KV must not be tensor-parallel-replicated

Measured KV is 27 KB/token (bf16) / 13.5 KB/token (fp8). With TP=4 attention every
rank needs the whole cache, i.e. ×4:

| ctx | bf16 ×4 | fp8 ×4 | **fp8, 1 copy** |
|---:|---:|---:|---:|
| 32 K | 3.62 GB | 1.81 GB | 0.45 GB |
| 64 K | 7.25 GB | 3.62 GB | 0.91 GB |
| 128 K | 14.50 GB | 7.25 GB | 1.81 GB |
| **256 K** | **28.99 GB** | 14.50 GB | **3.62 GB** |

28.99 GB does not exist on this machine. **Decision: FP8 KV, stored ONCE, and the
attention block is REPLICATED across ranks rather than tensor-parallel.**

Replicating the attention weights (self_attn + indexer, 1.235 GB bf16 → 0.62 GB
FP8, ×4 = 2.47 GB, i.e. +1.85 GB versus TP) buys back 3/4 of the KV cost, and has
a second, larger benefit: **every rank then produces the identical post-attention
hidden state, so there is no all-reduce after the attention block at all.** The
only per-layer communication left is MoE dispatch/combine.

Dense value path (linear_attn, hyper-connections, shared expert) stays TP=4,
because it is 5.9 GB and TP saves ~4 GB there; its all-reduce is fused into one
message per layer.

## 3. VRAM budget at 256 K

| item | GB |
|---|---:|
| experts (MXFP4 per-32) | 64.18 |
| dense value path, FP8, TP-split | 2.96 |
| attention + indexer, FP8, replicated ×4 | 2.47 |
| router (BF16, exact) | 0.13 |
| `lm_head` FP8 | 0.64 |
| **weights** | **70.37** |
| KV FP8 @ 256 K, 1 copy | 3.62 |
| GDN state + conv + PLE | 0.13 |

66.6 GB available ⇒ **~7.5 GB of weights must live outside VRAM** at full context
(~4.8 GB at 64 K, 0 at very short contexts). That is the DRAM expert tier, and it
is why the tier is structural.

## 4. Expected speed, and where 150 tok/s actually lands

Decode floor (measured 1792 GB/s aggregate): **5.00 GB/token → 2.79 ms → 358 tok/s**
ceiling with dense-FP8 + MXFP4 experts.

| context | offloaded experts | added DRAM fetch | est. token time | est. decode |
|---:|---:|---:|---:|---:|
| ≤ 8 K | ~0 GB | — | 2.8 ms ÷ 0.7 + 1.0 comms ≈ 5.0 ms | **~200 tok/s** |
| 64 K | 4.8 GB | ~1.0 ms | ≈ 6.0 ms | **~165 tok/s** |
| 256 K | 7.5 GB | ~1.5–2.5 ms | ≈ 7.0–8.0 ms | **~125–145 tok/s** |

So the honest target is **150+ everywhere up to ~64 K, ~125–145 at full 256 K**,
with the shortfall at max context coming entirely from the DRAM tier needed to
hold the KV arena. Closing it further would require per-row-FP4 experts
(60.5 GB, −3.7 GB) at a real quality cost — a measured trade, not a default.

**Prefill ≥5,000 tok/s is comfortable**: the QSA indexer caps attention at ~2,048
selected keys per query instead of O(ctx), so prefill cost stays ≈13 GFLOP/token
at any context (~65 TFLOPS at 5,000 tok/s ≈ 8 % of the FP4/FP8 peak). The risk is
kernel efficiency in the indexer (gather-heavy), not the arithmetic.

## 5. Speed degrades with context, by design

VRAM is partitioned per request: the KV arena is sized to the requested context and
the expert hot set receives the remainder. So decode speed is a **curve**, not a
constant, and the engine must report it (`inferno_ctx_tokens`, hot-expert count) so
the degradation is visible rather than mysterious. That is the honest product
statement for this machine.

## 6. Consequences for the issue plan

- I-014 must implement **replicated attention + FP8 KV stored once**, not
  conventional TP attention (this supersedes the open question in `THEORY.md` §4).
- I-015 is cut down to a single-sequence arena + cancel/preempt.
- I-021 (DRAM expert tier) is on the **critical path** for full-length context,
  not a later optimisation.
- I-022 (speculative decoding) becomes the main lever for going beyond ~200 tok/s,
  and no longer competes with batching for the win.
