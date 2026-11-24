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

## 7. KV tiering: hot window in VRAM, tail in DRAM (user directive 2026-11-24)

**Naive full-KV streaming is impossible**: 3.22 GB (fp8) per *token* would be 110 ms.
But QSA changes the working set, and this is the key insight:

| quantity | value |
|---|---|
| full FP8 KV at 256 K, one copy | 3.22 GB |
| **QSA budget per query** | **2,048 tokens** |
| per-(layer, token) K+V, fp8 | 1,024 B |
| **worst-case KV actually read per decode step** | 12 layers × 2048 × 1 KB = **25.2 MB** |
| that at 28.9 GB/s of measured host bandwidth | **0.87 ms/token** |
| pooled indexer keys at 256 K, **bf16**, per layer | **16.8 MB** (f32: 33.6 MB) |
| all 12 layers' pooled keys read per token | **201 MB bf16 / 403 MB f32** = 0.112 / 0.225 ms |
| hot window of 32 K context in VRAM (fp8) | 0.40 GB |
| **VRAM freed vs fully-resident 256 K** | **2.82 GB** |

So attention reads ~25 MB/token, not 3.2 GB/token — a ~128× reduction, entirely
because the indexer caps attention at 2,048 selected keys and selection is
**per layer**.

### Design

1. **Pooled indexer keys stay in VRAM** (8.4 MB at 256 K). They are maintained
   *incrementally* — when a 4-token block completes, its pooled key is computed once
   and stored — instead of the reference's re-pool of all raw keys every step. This
   also means **raw indexer keys need not be retained at all** beyond the current
   partial block, saving 3 KB/token of the 13.5 KB/token.
2. **A hot window of KV (e.g. the last 32 K positions) stays in VRAM.** Selection has
   strong recency locality (nearby queries select nearby blocks), so the hot window
   should absorb most selected tokens and the DRAM read is far below the 25 MB worst
   case. Size it by measurement, not by guess.
3. **The DRAM tail is gathered on demand** into a compact per-step VRAM working set:
   a dense `[budget, kv_head, head_dim]` K and V block per layer. The CPU issues the
   scattered 1 KB reads (12 cores, otherwise idle) into pinned staging, then one H2D
   per layer.
4. **Layout requirement:** K and V for one (layer, token) must be contiguous, 1 KB
   per token per layer, so a gather is one read and not two. Layout
   `[layer][token][2 kv heads][256] K then V`.
5. **Prefill**: gather by the *union* of the chunk's selections, not per query — with
   a 4 K chunk the union is a few hundred MB at most and amortises over thousands of
   tokens, so prefill pays ~0.2 s of DRAM traffic for a 256 K prompt.

### Consequences

- **The attention kernel must consume an explicitly gathered K/V working set**, never
  assume the master cache is contiguous or fully resident. This decouples the kernel
  from the storage tier and is *simpler* to implement.
- Freeing 2.82 GB of VRAM shrinks the DRAM expert tier from ~7.5 GB to ~4.7 GB at
  256 K, which also speeds up decode. The two tiers compete for the same DRAM
  bandwidth, so budget them together: worst case ~25 MB (KV) + ~40 MB (cold experts)
  per token ≈ 2.2 ms at 28.9 GB/s, against a ~2.8 ms HBM floor — i.e. decode remains
  HBM-bound at max context, which is exactly where we want it.
- Consequence for I-008 (in flight): attention takes a gather list, not an arena.

## 8. Correction: the pooled-key re-read dominates long-context state traffic

An earlier note in this file said fp32 pooled keys would cost ~7 ms/token. That was
wrong — it applied DRAM bandwidth to a buffer that is VRAM-resident. Measured
arithmetic at 256 K:

| item | per token |
|---|---:|
| pooled indexer keys, 12 layers, **bf16** | 201 MB → 0.112 ms |
| pooled indexer keys, 12 layers, f32 | 403 MB → 0.225 ms |
| **selected KV working set (the §7 gather)** | **25 MB → 0.87 ms of *host* bandwidth** |

So at long context the **indexer's pooled-key re-read (201 MB) is 8× larger than the
KV working set (25 MB)** — the indexer, not the KV gather, is the dominant per-token
*state* cost. It is cheap (0.112 ms) only because it is VRAM-resident and therefore
runs at HBM speed rather than host speed; that is precisely why pooled keys must stay
in VRAM and why they must stay **bf16**.

bf16 is also the *faithful* choice: the reference pools in f32 and casts to
`raw_keys.dtype`, which in the real bf16 model is bf16. Our f32 CPU oracle is the less
faithful of the two, so the observed 0.1 % selection drift is an oracle artifact.

Indexer compute is trivial at any context (0.067 GFLOP per layer per query at 256 K ⇒
0.36 µs at the measured fp8 rate), so a correct `indexer_scores` is memory-bound at
~0.11 ms/token across 12 layers. Gate: ≤0.35 ms/token at the 256 K block count.
