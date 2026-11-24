# inferno — from-scratch inference engine for Qwen3.8-Flash-Next-NVFP4 on this machine

Owner: orchestrator (plans, reviews, commits). Implementation: delegated worker
subagents. Review gate: `gate-reviewer`. Library skills in `.agents/skills/` are
authoritative (tiger-style for all Zig; inference-optimisation for measurement,
ceilings, ledger).

## 1. Target

Serve `nvidia/Qwen3.8-Flash-Next-NVFP4` (multi-modal MoE, 48 text layers,
512 experts/layer, top-10) with a from-scratch engine:

| layer | language | owns |
|---|---|---|
| hot path | **Zig** | CUDA driver API, streams, graphs, all kernel launches, model graph, MoE routing/dispatch, KV + recurrent state, weight placement, expert cache, sampling, request scheduling |
| rest | **Rust** | HTTP/SSE server, OpenAI-compatible API, tokenizer, config, metrics (Prometheus), tracing, control-plane placement planning |

GPU kernels are CUDA C++ compiled by `nvcc -arch=sm_120a` at build time into
cubins, embedded into the Zig binary and loaded via the CUDA **Driver** API.
Rationale: Zig cannot emit PTX/NVPTX; CUDA C++ is the only practical way to
target Blackwell tensor cores. Everything that *decides or schedules* work is Zig.
`-arch=sm_120a` is **mandatory** — plain `sm_120` rejects the block-scaled FP4 MMA
we depend on (`.agents/design/mma-sm120-nvfp4.md`).

## 2. Hardware (measured / queried, see THEORY.md)

- 4 × RTX 5060 Ti 16 GB (GB206, sm_120a, GDDR7, mem clock 15501 MHz).
- **All four GPUs hang off ONE bifurcated PCIe 4.0 x16 (4×4). P2P is enabled by
  a modified driver.** Aggregate host/user-visible PCIe bandwidth ≈ 25 GB/s, not
  the nominal per-slot rate. Host↔device and device↔device traffic is a scarce,
  shared resource — this drives nearly every design decision below.
- Ryzen 9 5900X (12C/24T), 60 GB DDR4, 251 GB free NVMe (931 GB Samsung 980 PRO).
- CUDA 13.2 (`/usr/local/cuda-13.2`), driver 610.57, Rust 1.100, Zig 0.14.1,
  Python 3.14 + `uv`.

## 3. Weight budget (exact, from safetensors headers)

Model files total 132.64 GB (`~/models/qwen38-flash-next-nvfp4`).

| component | GB | notes |
|---|---:|---|
| MoE experts (48×512×{gate,up,down}) | 67.95 | NVFP4: U8 packed pairs + F8_E4M3 per-16 block scale (7.55) + F32 per-tensor scale |
| PLE n-gram table (128 shards, layer 2) | 51.20 | F8_E4M3 [320M, 160], gather-only (16 rows/token) |
| linear_attn (36 layers) | 4.16 | BF16 |
| self_attn + indexer (12 layers) | 1.23 | BF16 |
| hyper-connections (48 layers × 2 blocks) | 1.27 | BF16, 10240-dim |
| shared expert + router | 0.60 | BF16 |
| embed_tokens + lm_head | 2.54 | BF16, untied |
| MTP (1 layer) + PLE projections | ~2.72 | FP8 blockwise / BF16 |
| vision tower | 0.68 | BF16, deferred |

Text-path hot bytes (per token, as shipped): experts 1.327 (10/512) + dense 7.26
+ lm_head 1.271 ≈ **9.9 GB/token** → ~5.5 ms at aggregate GDDR7 bandwidth.

**VRAM is 67.9 GB usable weight budget for 77.8 GB of text weights.** The gap is
closed by (a) keeping the 51 GB n-gram table on NVMe (gather-only, mmap), (b)
keeping `embed_tokens` host-side (gather-only), (c) requantizing the dense path,
(d) an expert hot/cold split with an idle-CPU fallback path (see §6 levers).
Everything is config-driven; correctness first, then hill-climb.

## 4. Architecture of the engine

```
Rust: axum HTTP/SSE ── tokenizer ── request queue ── metrics/tracing
        │  C ABI (static link, in-process)
Zig core: scheduler → per-sequence state → model graph executor
        │            ├── weight placement / expert cache (VRAM + host)
        │            ├── KV cache + GDN recurrent state pool
        │            ├── CUDA driver: streams, events, graphs, P2P
        │            └── kernels (cubins from nvcc)
4 × GPU: TP=4 for dense, EP=4 for experts (contiguous 128-expert shards)
```

Parallelism: **TP=4 dense + EP=4 experts**, all device↔device movement over P2P
(driver supports it). Target ≤1.5 ms/token of communication latency. Layer
pipelining is *not* used for a single stream (it cannot shorten the critical path).

## 5. Layout

```
core/                     Zig hot path (static lib, C ABI)
  build.zig               also drives nvcc for kernels/*.cu (-arch=sm_120a)
  src/c_api.zig           exported C ABI surface
  src/hw/                 driver bindings, streams, events, graphs, peer copy (I-001)
  src/bench/              hardware probe binary (I-002)
  src/weights/            safetensors, config, manifest, quant, expert repack (I-003)
  src/ref/                fp32 CPU reference = the GPU oracle (I-005)
  src/kernels/*.cu        CUDA kernels (sm_120a)
  src/model/              qwen4_exp graph: gdn, indexed attention, moe, hc, ple
  src/mem/                placement planner, expert cache, kv/state pool
  src/sched/              scheduler, batching, sampling
server/                   Rust binary; links core as a static lib
tools/ref-trace/          Python: HF reference traces + tiny random models
tools/requant/            offline dense/expert requantization
tools/gate.sh             build + test + clippy + zig fmt + kernel compile
tests/golden/             committed golden fixtures
THEORY.md PLAN-log.md ledger.jsonl configs/ best-config.json   (inference-optimisation skill)
```

## 6. Optimization levers (registered, not yet measured — see PLAN-log.md)

| id | lever | est. payoff | risk |
|---|---|---|---|
| L1 | CUDA graphs for decode (whole 48-layer step captured) | 1.5–2× c1 | none |
| L2 | dense path → FP8 (4.36 GB) then NVFP4 if needed | frees 4.4 GB VRAM | small quality |
| L3 | experts → per-32 block scales (4.25 b/p, 64.2 GB) | frees 3.7 GB VRAM | needs eval gate |
| L4 | hot/cold expert split, statistics-driven | fits in VRAM | ~2-5% tok on PCIe |
| L5 | **cold experts computed on the idle CPU** (12 cores, ~40 GB/s DDR4 otherwise unused) instead of PCIe | avoids 1–4 ms/token | implementation |
| L6 | fused QKV/gate-up GEMMs + native `OMMA.SF` NVFP4 block-scaled prefill MMA (verified available on `sm_120a`, see `.agents/design/mma-sm120-nvfp4.md`) | ~4× bf16 tensor throughput for prefill | kernel work |
| L7 | speculative decoding (prompt-lookup n-gram; MTP later, no HF reference for MTP) | 1.5–2.5× c1 | quality-neutral |
| L8 | indexer top-k selection early exit at short context | short-ctx gains | none |
| L9 | batch/pack decode across sequences (continuous batching) | 2–4× cN | scheduler |

## 7. Issues (dependency order)

Done: **I-001** build skeleton (gates green; probe sees 4 × 16.66 GB, driver
610.57.04) · **I-004** oracle (240 golden tensors, transformers 5.24.0.dev0 @
d8f10a8, tiny checkpoint in both HF and published-checkpoint layouts).
Carried-forward reviewer suggestions: `.agents/tasks/reviewer-followups.md`.

Wave 1 — foundation (parallel-safe, disjoint dirs):
- **I-01** build skeleton: Cargo workspace + Zig static lib + nvcc pipeline + `tools/gate.sh` + hardware probe (device query, P2P, measured PCIe/GDDR bandwidth). Lane A.
- **I-04** `tools/ref-trace`: uv venv, CPU torch + transformers(main) reference, tiny random-weight Qwen4Exp models, golden fixtures per module. Lane B.

Wave 2 — data + correctness:
- **I-02** CUDA driver layer in Zig (context/dev/stream/event/module/buffer/peer-copy, error taxonomy, `zig build test` on real GPUs). Lane A.
- **I-03** weights: safetensors mmap reader, config/manifest, NVFP4 + FP8 blockwise layouts, placement planner input. Lane A.
- **I-05a** CPU reference: GatedDeltaNet (chunk + recurrent) matches goldens.
- **I-05b** CPU reference: indexed attention (partial RoPE, mRoPE, QSA indexer, gated output) matches goldens.
- **I-05c** CPU reference: router + 512-expert MoE + shared expert matches goldens.
- **I-05d** CPU reference: hyper-connections (10240-dim, low-rank) + PLE n-gram embedding matches goldens.
- **I-05e** CPU reference: full layer + full model end-to-end greedy tokens match HF.

Wave 3 — GPU kernels (each gated against the CPU reference, not against itself):
- **I-06** BF16/FP8/NVFP4 GEMV+GEMM (dequant-fused; block-scaled mma for prefill).
- **I-07** RMSNorm (grouped + gated), RoPE (partial + mRoPE), SiLU-mul, sigmoid gates, softmax/sampling.
- **I-08** attention: prefill causal GQA with indexer mask overlay; decode with KV cache.
- **I-09** GDN: chunked scan prefill, recurrent decode, causal conv1d (k=4) state update.
- **I-10** MoE: router top-k, permute/group, grouped fp4 GEMM, combine, shared expert.
- **I-11** PLE: 16-row n-gram gather (host, async), key/value proj, dilated depthwise conv, inject.

Wave 4 — engine:
- **I-12** single-GPU end-to-end tiny-model generation matching HF (greedy, bit-stable).
- **I-13** multi-GPU TP=4/EP=4 with P2P allreduce/dispatch; validated vs I-12 outputs.
- **I-14** KV + recurrent-state pool, scheduler, continuous batching, prefix cache.
- **I-15** sampling (top-k/top-p/temp/repetition), C ABI, streaming token out.

Wave 5 — server (can start after I-01):
- **I-16** Rust server: axum, `/v1/chat/completions` (stream + non-stream), `/health`, `/metrics`.
- **I-17** tokenizer (HF `tokenizers`), chat template rendering, stop sequences, usage accounting.
- **I-18** config/logging/tracing/graceful shutdown/backpressure.

Wave 6 — real model + hill-climb:
- **I-19** load the real 132 GB checkpoint, first coherent tokens, placement report.
- **I-20** acceptance gates: needle at 50%/100% context, chat + tools round-trip, soak.
- **I-21** CUDA graphs + kernel tuning round 1 (L1, L6).
- **I-22** requantization tool (L2, L3) with quality gate.
- **I-23** expert hot/cold split + CPU-expert path (L4, L5).
- **I-24** speculative decoding (L7).
- **I-25** vision tower (deferred, optional).

## 8. Non-negotiable

0. **The core is dependency-free and its unsafe mass stays contained.** Enforced by
   `tools/gate.sh` "structural invariants": `core/build.zig.zon` must be
   `.dependencies = .{}`, `Cargo.lock` must hold exactly one package, and no CUDA
   symbol may appear outside `core/src/hw/`. Rationale: Zig was chosen for the
   core because our unsafe mass is pointer-shaped (driver handles, mmap'd page
   tables, intrusive LRU/free-lists, coalesced staging buffers) and Zig wraps more
   of that behind a smaller surface than Rust, where the same unsafety relocates
   into index-slab modelling, `Pin`, lifetime workarounds and `unsafe impl Send`.
   Rust keeps the *other* mass — dataflow between threads and across the network —
   where `Send`/`Sync` and the type system still buy real guarantees.
   Corollary: because Zig safety is a build mode (`ReleaseSafe`) rather than a
   compiler guarantee, the assertion discipline below is load-bearing, not
   decorative.
1. Every Zig function: ≥2 assertions, no recursion, no dynamic allocation after
   init, ≤70 lines, explicit integer widths (tiger-style).
1b. **No `assert`/panic may cross the C ABI.** Exported functions return error
   codes; a panic on a driver error path would abort the Rust server. (Found by
   the I-001 gate review.)
2. Every kernel is validated against the **CPU reference** (I-05) or the **HF
   golden trace** (I-04) — never against another GPU implementation.
3. No gate may be weakened to make it pass; the orchestrator re-runs every gate.
4. Numbers only enter PLAN-log.md with a median-of-N and a control run.
5. Workers never decide product/architecture; they escalate via
   `contact_supervisor` with `reason: "need_decision"`.
