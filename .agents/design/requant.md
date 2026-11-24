# Requantisation strategy (I-020)

Goal: make the text path fit in **66.6 GB** of measured VRAM without a permanent
per-token PCIe tax, and as a side effect roughly halve decode bytes/token.

## 0. The gap, precisely

| item | as shipped | fits? |
|---|---:|---|
| experts (NVFP4: 4 b/param + E4M3 per-16) | 67.95 GB | |
| dense value path BF16 (linear_attn 4.173, self_attn+indexer 1.235, hyper 1.272, shared 0.471) | 7.15 GB | |
| router `mlp.gate` BF16 | 0.126 GB | |
| `lm_head` BF16 | 1.271 GB | |
| **total text path** | **76.5 GB** | **no — 10 GB over** |
| + GDN state / KV / workspaces | ~1.5 GB | |
| usable VRAM (measured, 4 × 16,660,430,848 B − contexts) | 66.6 GB | |

`embed_tokens` (1.27 GB) and the 51.2 GB n-gram table never consume VRAM: both are
gather-only (1 row/token and 16 rows/token).

## 1. The governing principle: discrete decisions stay exact

A 1e-3 weight perturbation in a **value** path adds tolerable noise. The same
perturbation in a path that makes a **discrete decision** can flip the decision and
cause a large output divergence. So:

| must stay exact (BF16/F32) | why |
|---|---|
| `mlp.gate` router (0.126 GB) | a flipped top-10 changes the entire forward pass |
| `self_attn.indexer.*` projections + layernorms | indexer selects a **discrete token set**; a different set = different attention |
| `linear_attn.A_log`, `dt_bias` | exponentiate into the forget/decay gate; compounding over thousands of steps |
| all norms (`rms_norm`, `hc_norm`, `q/k_norm`, PLE norms) | multiplicative, and `1 + w` doubles sensitivity |
| `conv1d` weights (main + PLE), `ple.*_proj` | tiny (≈70 MB) — no reason to touch |
| `embed_tokens`, n-gram table | already 1–2 bytes and gather-only |

This is also why I-008's gate demands **exact integer set equality** for the indexer
selection and I-013 demands bit-identical greedy ids — those are the detectors that
tell us a discrete path drifted.

## 2. Formats

### Experts → **MXFP4: E2M1 weights, E8M0 scale per 32**
- Native on this GPU: `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0`
  → SASS `OMMA.SF.16864.F32.E2M1.E2M1.E8` (verified, `.agents/design/mma-sm120-nvfp4.md`).
- 4.25 b/param vs NVFP4's 4.5 → 67.95 → **64.18 GB** (−3.77 GB). Per-32 is the
  *coarsest* block FP4 MMA supports (`VS ∈ {16,32}`), so this is the floor.
- Quantise from the **dequantised NVFP4 float**, never from the 4-bit codes, so the
  only added error is the coarser scale granularity.
- **Rejected:** per-row FP4 (≈4.01 b/param, saves 7.4 GB) — one scale per output
  row makes outliers crush small values, far outside the format's design envelope.

### Dense value path → **E4M3 with one F32 scale per output row (W8A16 decode, W8A8 prefill)**
- 7.15 → **3.575 GB** (−3.58 GB).
- Per-output-row scaling is mathematically free: `y[n] = s[n] · Σ w_q[n,k]·x[k]`,
  so the scale folds into the accumulator finalise / GEMM epilogue. **Nothing is
  applied inside the k-loop** (contrast NVFP4, where the hardware must scale every
  16 elements — which is exactly why block-scaled MMA exists).
- Offline static weight quantisation, no calibration data required.
- Decode: W8A16 GEMV, dequantise E4M3→BF16 in registers (a shift, not a multiply).
- Prefill: prefer W8A8 (`mma.sync.m16n8k32.f32.e4m3.e4m3`, 2× bf16 rate) with
  dynamic per-token activation scaling; keep the W8A16 dequant path as the
  numerical reference.

### `lm_head` → **E4M3 per output row** (1.271 → 0.636 GB)
Sampling only reads the logit tail; per-row scaling protects it. `lm_head` in FP4
is rejected — rare-token logits are exactly where 4 bits hurts.

### Hyper-connections → E4M3 per output row (1.272 → 0.636 GB)
Pure value paths. Second-cheapest rung only if needed.

## 3. The ladder — stop at the cheapest rung that fits

| rung | change | weights | +state ≈1.5 | fits 66.6? | decode bytes/token |
|---|---|---:|---:|---|---:|
| 0 | as shipped (baseline quality) | 76.50 | 78.0 | no (−11.4) | 9.92 GB |
| 1 | dense + lm_head → E4M3 | 72.23 | 73.7 | no (−7.1) | 6.34 GB |
| 2 | + experts → MXFP4 per-32 | 68.46 | 70.0 | no (−3.4) | 5.62 GB |
| 3 | + hyper-connections → E4M3 | 67.83 | 69.3 | no (−2.7) | 5.30 GB |
| 4 | + 2.8 GB of coldest experts offloaded (L4/L5) | 65.03 | 66.5 | **yes** | 5.30 GB |

Rung 4 only fits by moving ~4 % of expert bytes out of VRAM. That is why **L5
(compute cold experts on the idle CPU) is the preferred closer**: 14 cold
expert-invocations/token × 2.4 MB read from DDR4 at ~40 GB/s with 12 cores costs
~0.1–0.2 ms/token, whereas streaming the same bytes over the measured 26.9 GiB/s
PCIe costs ~2.7 ms/token *forever*. If L5 does not land, rung 4 costs ~25 % of
decode throughput.

**Decision rule:** implement rungs 1→3 unconditionally (pure wins, no behavioural
risk), measure, then choose between L5 and PCIe offload for the remainder.
Deprioritise L3/L4 and go straight to L5 if the expert requant's quality gate is
marginal.

## 4. Quality gates (measured, per rung, against the rung-0 engine)

Requantisation changes weights, so greedy ids will **not** be bit-identical — that
is expected and must not be treated as failure. Gates, in order of sensitivity:

1. **Router top-k agreement ≥ 99.5 %** of (token, layer) decisions — the discrete
   guard, and cheap to measure.
2. **Indexer selection Jaccard ≥ 0.98** per query.
3. **ΔNLL/token < 0.02 nats** (≈ <1 % perplexity) over a fixed ~100 KB held-out
   corpus, teacher-forced.
4. **Task set**: 20 held-out prompts with checkable answers, no regression beyond
   noise.
5. Greedy continuation on 8 fixed prompts: report first-divergence position; a
   divergence before token ~48 is a red flag to investigate.

Keep the original 132 GB checkpoint on disk so every rung is A/B-able and
revertible without re-downloading.

## 5. Mechanics and cost

- One offline pass, `tools/requant`, reading 76.5 GB and writing ~66 GB; I/O
  bound, tens of minutes. Output is our own shard format + manifest, cached next to
  the checkpoint; **never converted at startup**.
- Disk (measured after the download completed): 122 GB free with the 133 GB original
  in place. A ~66 GB converted copy therefore fits with ~56 GB spare; the converter
  must check free space first and refuse to start if it would leave < 20 GB.
- Determinism: the converter must be bit-reproducible (fixed rounding: RNE to the
  target grid) so a rebuilt artefact is byte-identical — asserted by hashing.
- The converted weights are validated by re-running the **manifest** gate
  (I-003's `total_bytes == index_total_size` equivalent) plus the decoders'
  oracle tests, before any model load.
