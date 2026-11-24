# THEORY.md — ceilings for this machine

Per `.agents/skills/inference-optimisation/references/ceilings.md`: compute the
maximum first. Every number below is either **queried** (marked Q) or a
**spec-sheet guess to be replaced by measurement** (marked M-todo). Measurement
lands in `bench/hw-probe.json` (issue I-02) and `ledger.jsonl`.

## 1. Machine (Q, 2026-11-24)

| item | value |
|---|---|
| GPU | 4 × NVIDIA GeForce RTX 5060 Ti, 16311 MiB each = 65244 MiB ≈ 68.4 GB total |
| compute capability | sm_120 (GB206), `-arch=sm_120a` for block-scaled mma |
| max memory clock (Q) | 15501 MHz |
| memory bandwidth per GPU (M-todo) | spec 448 GB/s (128-bit GDDR7) → aggregate 1792 GB/s |
| PCIe | **one bifurcated Gen4 x16 = 4 × x4**, all four GPUs on the same root complex; P2P enabled by modified driver |
| host↔device / device↔device aggregate (M-todo) | ≈25 GB/s raw Gen4 x16 → ~20–23 GB/s effective, **shared** by all 4 GPUs |
| CPU | Ryzen 9 5900X, 12C/24T, DDR4 dual-channel; measured STREAM-like ≈35–45 GB/s (M-todo) |
| system RAM | 60 GB total, ~56 GB available |

**Consequence:** PCIe is a bottleneck for any tier that is not VRAM. Weights that
must be read every token belong in VRAM. Host RAM is used only for (a) the
51.2 GB n-gram table (gather-only, ~64 KB/token from NVMe/RAM) and (b)
`embed_tokens` (gather-only, 5 KB/token), plus optionally cold experts (lever L5).

## 2. Decode ceiling (memory-bandwidth bound)

Bytes read per emitted token with all weights resident in VRAM:

| variant | bytes/token | ms/token @1792 GB/s | tok/s ceiling |
|---|---:|---:|---:|
| as shipped (bf16 dense, fp4 experts) | 9.92 GB | 5.53 | **181** |
| dense FP8 (L2) | 5.67 GB | 3.16 | **316** |
| + lm_head FP8, experts per-32 scales (L3) | 5.66 GB | 3.16 | 316 |

Breakdown (as shipped): experts 10/512 × 67.95 = 1.327 · dense 7.26 · lm_head
1.271 · PLE projections 0.065.

**Practical ceiling** = 50–65 % of the above (kernel efficiency, 4-GPU latency,
PCIe comms): **35–70 tok/s single-stream** for the as-shipped layout. Any claim
above 181 tok/s is a measurement error.

Speculative decoding (L7) divides the target reads by the accepted length but adds
the draft cost; with an n-gram/prompt-lookup draft (no weights) the realistic
multiplier is 1.4–2.0× at ≥0.6 acceptance.

## 3. Prefill ceiling (compute bound)

Active parameters per token: experts 10/512 × 40.3 B = 0.79 B, dense 3.63 B,
shared expert 0.24 B → **≈4.4 B active** → 8.8 GFLOP/token, plus attention
≈4·L·h·s = 4 × 12 × 2560 × s.

| BF16 tensor peak (M-todo) | ≈50 TFLOPS/GPU → 200 TFLOPS aggregate |
|---|---|
| FP4 tensor peak (M-todo) | ≈4× BF16 → 800 TFLOPS aggregate |
| practical (30–50 %) | 60–100 TFLOPS BF16-equivalent |

At 32 K prompt: ≈13 GFLOP/token ⇒ **1.5–4 K tok/s** prefill; short prompts
(≤2 K) are dominated by expert GEMV inefficiency, not FLOPs.

**TTFT floor** = one forward pass of the last prompt token + tokenize + schedule
≈ 10–20 ms once the context is cached.

## 4. State memory (caps concurrency)

| item | per sequence | note |
|---|---:|---|
| KV (12 full-attn layers, 2 kv heads × 256 × bf16) | 12 KB/token | 128 K ctx → 1.5 GB |
| indexer keys (12 layers × 128 dim bf16) | 3 KB/token | 128 K ctx → 0.4 GB |
| GDN recurrent state (36 layers × 48 heads × 128 × 128 fp32) | **113 MB** | state, not KV, caps concurrency |
| GDN conv state (36 × 3 × 10240 bf16) | 2.2 MB | |
| PLE conv state + n-gram context | ~0.2 MB | |

With ~1.5–2 GB VRAM left after weights: **8–16 concurrent sequences** at moderate
context, or 1 × 128 K. GDN state in bf16 halves the state cost (lever, needs a
quality/equivalence gate).

## 5. Quality gates that must hold at every measured point

- Greedy output equivalence against the CPU reference (I-05) on the tiny model:
  bit-identical token ids for 32 tokens.
- Real-model needle-in-haystack at 50 % and ~100 % of the working context.
- Real-model chat + tools round-trip, no degenerate/looping output.
- Any quantization lever (L2, L3) additionally needs: same greedy 64-token
  continuation as the unquantized path on ≥8 fixed prompts, and no increase in
  perplexity proxy beyond the threshold recorded in its issue.
