# Lever survey — what we are and are not doing (2026-11-24)

Scope: 4×RTX 5060 Ti (66.6 GB VRAM, HBM 397.6 GiB/s/GPU), single stream, full
256 K context, targets ≥150 tok/s decode / ≥5 000 tok/s prefill.

## 1. Speculation is the biggest lever, and it is on the table

Decode is **weight-read bound**: 5.00 GB/token ⇒ 2.79 ms floor. Verify *k* candidate
tokens inside one weight read and throughput scales with the accepted length:

| scheme | ms/pass | accepted | ms/token | tok/s |
|---|---:|---:|---:|---:|
| no speculation | 2.79 | 1.0 | 2.79 | 358 |
| k=4 | ~3.15 | 2.5 | 1.26 | ~790 |
| k=4 | ~3.15 | 2.0 | 1.58 | ~635 |
| k=5 | ~3.27 | 3.0 | 1.09 | ~915 |

(The extra per-candidate cost, ~0.12 ms, is attention over the gathered KV working
set plus the verify GEMM; the weight read is paid once.) `lm_head` also amortises:
0.64 GB per pass covers all k candidates.

**This promotes speculation from "L7 nice-to-have" to a primary component.** Note the
target is already met without it, so it is upside rather than risk — but if we want
materially more than ~150 at full context, this is the only lever that gets there.

### Draft sources, ranked for *this* machine

1. **Prompt-lookup / n-gram** — zero weights, zero VRAM, works on code, editing,
   quoting and repetition. Highest value per byte. Implement first.
2. **MTP head from the checkpoint** (`mtp.*`, 1 layer, FP8 blockwise, ~2.65 GB) — a
   genuine EAGLE-style draft head: `fc_hidden`+`fc_embedding`, norms, one
   attention+MoE layer, own hyper-connection mixer. **Risks:** no reference
   implementation exists (HF ignores `mtp.*`), so semantics must be reverse-engineered
   from shapes; and 2.65 GB is 2.65 GB we would have to take from the expert hot set.
   Try only after the KV/DRAM budget is measured.
3. **Self-speculation by layer skip** — draft with a prefix of the 48 layers, verify
   with all of them. Zero extra weights, so it costs no VRAM at all; acceptance is
   lower. Best VRAM-free option if MTP does not fit.

### The landmine: GDN state cannot be rolled back

Speculation writes KV **and** the GDN recurrent state for rejected tokens. The delta
rule updates a 128×128 fp32 state **in place**; undoing it requires the pre-state.
Therefore I-022 must **snapshot the recurrent state per speculative round**
(113 MB/seq ≈ 0.06 ms at HBM speed) or make the update reversible (numerically
unattractive). KV rollback is free (just reset the length). Any MTP/lookahead design
must budget this snapshot, and the snapshot itself scales with k if we want
per-candidate rollback of intermediate states — measure before choosing.

## 2. Missing from the plan, in descending payoff

| # | technique | why it matters | cost |
|---|---|---|---|
| M1 | **Kernel fusion / graph node budget** | ~1 000 nodes × 1–2 µs GPU-side dispatch = **1.0–2.0 ms**, comparable to the entire 2.79 ms floor. CUDA graphs remove *CPU* launch overhead, not per-node GPU dispatch. Target **≤300 nodes** (≤6–7 fused kernels per layer) and measure replay overhead. | engineering |
| M2 | **Prefix caching** (dropped with batching — wrongly) | interactive chat re-prefills the whole conversation every turn. Caching the prefix makes TTFT ≈ 0 for turn N+1. Prefix KV already lives in the DRAM tier, so it composes with §7 KV tiering. | small |
| ~~M3~~ | ~~Dense value path → NVFP4~~ | **REJECTED by owner (2026-11-24): a quality hit.** The dense path feeds the GDN recurrence, where 4-bit error compounds over thousands of steps; FP8 is the quality floor for it. Cost of rejecting: the decode floor stays 2.79 ms (not 1.79) and 1.79 GB of VRAM stays committed, so the DRAM expert tier carries ~4.2 GB instead of ~2.4 GB at 256 K. Accepted. | — |
| M4 | **GPUDirect Storage for the 51 GB n-gram table** | the PLE gather currently needs a CPU round trip plus H2D. GPU-initiated NVMe reads remove both. Only worth it if I-011's measured gather latency is a real fraction of the token time. | medium |
| M5 | **FP8 K/V in the attention MMA** | gathered K/V are already fp8, so QK^T can use `mma.sync.m16n8k32.e4m3` at 2× bf16 rate instead of dequantising to bf16. Pure win at long context where attention share grows. | small |
| M6 | **GDN state in bf16** | 113 → 56.6 MB per sequence; frees ~57 MB/seq for the expert tier. Needs an equivalence gate (recurrence, so it compounds). | small |
| M7 | **L2 persistence window on hot expert weights** | ~30 MB of L2 per GPU; pinning the most-read weight pages may shave a few % of HBM traffic. Free to try, measure or drop. | tiny |
| M8 | **Decode megakernel** (persistent kernel looping all 48 layers) | eliminates node dispatch entirely; the "optimal" answer to M1. Very high complexity, and it fights per-layer data-dependent routing (top-k ids). Stretch only. | large |
| M9 | **UVM/`cuMemAllocManaged` for the KV tail** | could replace explicit prefetch. Likely worse on consumer parts (migration heuristics), but one experiment settles it. | small |

## 2b. Consequence of rejecting M3: the quality budget is now closed

Quantisation stops at **FP8 weights** (dense path + `lm_head`) and **FP8 KV**
(both sanctioned, both standard). Experts stay at their native 4-bit, requantised
only to a coarser *block* (per-16 → per-32 MXFP4, same 4-bit grid, no extra
activation quantisation). **No further quantisation lever is available.**

Therefore every remaining speed lever must be **quality-neutral**:
speculation, kernel fusion / graph-node reduction, prefix caching, FP8-KV MMA,
GPUDirect Storage, L2 persistence, GDN-state bf16. That is a useful constraint: it
removes the whole class of "faster but a bit worse" decisions from the hill-climb
and makes the ledger honest — any future gain is either mechanical or it is a
quality trade requiring the owner's sign-off.

## 3. Deliberately not doing (so they are decisions, not omissions)

- **Continuous batching / concurrency** — dropped by directive; speculation replaces
  it as the throughput lever, and the VRAM it would need is better spent on KV.
- **Separate draft model** — 1–4 GB VRAM we do not have, for less acceptance than MTP.
- **Medusa / trained EAGLE heads** — we cannot train.
- **Structured sparsity, prompt compression, int4 *activation* calibration** — either
  model-changing or of unproven payoff here.
- **Any quantisation beyond FP8 weights / FP8 KV / native 4-bit experts** (M3 in
  particular) — rejected on quality grounds; see §2b.
- **Interleaved prefill/decode overlap** — meaningless with one sequence.
