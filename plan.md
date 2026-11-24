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

## 7. Issue inventory (single source of truth — renumbering drifted from the original draft)

`[x]` merged & gated · `[~]` implemented, in review · `[s]` spec written, queued · `[ ]` not yet specified

### Foundation — done
- [x] **I-001** build skeleton: Cargo workspace, Zig static lib, nvcc→cubin→embed, FFI probe
- [x] **I-004** oracle: HF reference traces, tiny random model in HF + checkpoint layouts, 240 goldens

### Wave 2 — in review
- [~] **I-002** hardware probe (HBM/PCIe/P2P/GEMM peaks → replaces every M-todo in THEORY.md)
- [~] **I-003** weights: safetensors, config, tensor manifest, NVFP4/FP8 decoders
- [~] **I-005** CPU reference: full Qwen4Exp text graph in f32 = the GPU oracle
- [~] **I-007** Rust server: axum + OpenAI API + tokenizer + metrics (MockEngine backend)

### Wave 3 — kernels (each gated against the CPU reference, never against another GPU path)
- [s] **I-006a** decode GEMV family (NVFP4 / FP8 blockwise / BF16) + batched decode
- [s] **I-006b** prefill GEMM: native `OMMA.SF` NVFP4 + exact BF16-dequant reference path
- [s] **I-008** indexed (QSA) attention: indexer selection, prefill sparse attention, decode
- [s] **I-009** Gated DeltaNet: causal conv, chunked UT-scan prefill, recurrent decode
- [ ] **I-010** MoE: router top-k, permute/group, grouped NVFP4 GEMM, combine, shared expert
- [ ] **I-011** PLE: n-gram id derivation, 16-row gather, key/value proj, dilated depthwise conv
- [ ] **I-012** elementwise/structural: RMSNorm (grouped, gated), partial+mRoPE, SiLU-mul, sigmoid gates, sampler

### Wave 4 — engine
- [ ] **I-013** single-GPU end-to-end tiny model: greedy ids bit-identical to HF  ← integration milestone
- [ ] **I-014** multi-GPU TP=4 dense + EP=4 experts over P2P; must match I-013 token-for-token
- [ ] **I-015** KV + GDN-state pool, scheduler, continuous batching, prefix cache
- [ ] **I-016** sampling, C ABI, token streaming → wires I-007 to the real engine end to end

### Wave 5 — real model + hill-climb
- [ ] **I-017** load the real 132 GB checkpoint; placement report; first coherent tokens
- [ ] **I-018** acceptance gates: needle at 50 %/100 % context, chat + tools round-trip, soak
- [ ] **I-019** CUDA graphs + kernel tuning round 1 (L1, L6)
- [ ] **I-020** requantiser: dense→FP8 (L2), experts→per-32 MXFP4 (L3), with quality gate
- [ ] **I-021** expert hot/cold split + CPU-expert path (L4, L5) — the VRAM gap closer
- [ ] **I-022** speculative decoding: prompt-lookup n-gram (L7)
- [ ] **I-023** vision tower (deferred; text path does not need it)

### Critical path to "the real model answering over HTTP"
I-006a → I-006b → I-008/I-009/I-010/I-011/I-012 → I-013 → I-014 → I-016 → I-017
(10 issues, with I-014 effectively mandatory: 78 GB of text weights cannot fit in
66.6 GB of VRAM on one GPU, and host streaming over 4×x4 PCIe costs ~40 ms/GB/token).

## 7b. Worker model routing (user directive 2026-11-24, binding)

| work class | model | notes |
|---|---|---|
| **all worker lanes** (hard and straightforward) | **`opencode-go/deepseek-v4.1-flash`** ("DeepSeek V4.1 Flash 1017", 1M ctx) | **user directive 2026-11-24: workers use the opencode-go provider.** The earlier "one slot only" reservation is superseded — lanes may run concurrently on this provider; watch for rate limits rather than assuming the slot is exclusive |
| `synthetic/syn:large:text` | **review gate and reviewer roles only** | **the same model as `opencode-go/deepseek-v4.1-flash` on a different provider** — not a second opinion, see below |
| `desktop` | **OFFLINE** | see the note below |
| **review gate** | `synthetic/syn:large:text`, always | never downgrade it to a small model. **Correction 2026-11-24: this is the same model as the worker on a different provider**, so the gate is *not* an independent second opinion — see the note below on why it still works |

`local/local` (gary-agents) was returning `Bad Gateway` on 2026-11-24 and must not
be used for dispatched work until it answers `/v1/models` again.

### Why the review gate works even though it is the same model

I claimed for most of a night that the gate was a *different family* from the implementer. It is
not: `synthetic/syn:large:text` and `opencode-go/deepseek-v4.1-flash` are the same model on
different providers. Every reviewer on a deepseek lane — I-010 MoE, I-013, I-014 — was the same
model grading its own work.

The gate nevertheless caught the MoE gate-2 spec defect, the `max_logit`-over-reference vacuity,
the I-014 prefetch race, the "torch is not installed" false claim, and the 1,000x-too-loose perf
gate. So its value is **procedural, not model-diverse**:

- a **fresh context** with no memory of the implementer's reasoning or its self-justifications,
- an **adversarial instruction** ("judge the tree by execution, not the report"),
- **read-only** access, so it cannot fix what it finds and must report it,
- and the orchestrator **re-runs every gate** independently afterwards.

That is a real check on *claim quality* — a report that says "torch is not installed" dies because
someone runs the command. It is **not** a check on the model's blind spots, which the implementer
and reviewer share. Where a genuine second opinion matters (spec interpretation, gate design), the
only different-family model available was `desktop` (Qwen3.8-Flash-Next), now offline and unavailable.

**`desktop` is offline from 2026-11-24 (user directive): the machine is needed for other
work.** It served its purpose — I-024 cleanup passed first time with zero blocking, all eight
findings pinned by tests — and must not be dispatched to again. Mechanical work routes to
`syn:large:text` instead.

**This is an availability decision, not a measurement-hygiene one.** An earlier version of this
note claimed `desktop` contended for the same GPUs and thereby explained the ~3 % run-to-run
spread that forced a perf guard widening. That was wrong: `desktop` is `gary-desktop`, a
**separate machine** from `gary-agents`, sharing no GPU, VRAM or PCIe with it. The spread
remains unexplained and must not be attributed to this box.

**Spec defect pattern, recorded:** three issues (I-013, I-014, I-014b) specified a named
`zig build <something>` gate. `tools/gate.sh` only ever runs `zig build test`, so every such gate
would have pointed at a step that either does not exist or that CI never invokes. All gates are
now normal test blocks wired through `c_api.zig`.

**Verification effort is proportional to evidence (user directive 2026-11-24).** Do not run slow
repeat loops by ritual. A single gate run plus `zig build test` is the default. Repeat-run
determinism checks are justified **only** by evidence:

- a result that differed between two runs (the I-013 case: `max_abs_err` 1.423e-2 / 7.814e-3 /
  9.803e-3 from one tree — that warranted 16 runs),
- a change touching async ordering, stream fences, allocation sizing or shared device buffers,
- or a new failure nobody can reproduce on demand.

Once determinism has been established for a component, record the evidence and stop re-proving
it. Re-establish only when one of the above fires again. Sixteen identical runs a second time
proves nothing that the first sixteen did not, and it costs twelve minutes per gate.

The real benchmark rule is unchanged and independent: **no kernel or benchmark lane may run
concurrently with another on `gary-agents`.** Conflating the two cost a false causal claim in
this file, which is the same error class as the earlier bandwidth-tier mistakes — asserting a
mechanism without checking the topology.

## 7c. Measured baselines and the bar we are chasing

Two external reference points, recorded as targets (the peer engine on this box is
deliberately unidentified — do not go looking for it, the numbers are what matter):

| platform | decode | prefill | notes |
|---|---|---|---|
| 1x RTX 4090, 64 GB DDR5 | 75 tok/s | 1,200 tok/s | Flash-Next IQ4_XS, weight offload implied |
| **this box, peer engine** | **100-150 tok/s** | **5,000 tok/s** | Flash-Next, 4x 5060 Ti, full 256K context |

**Where inferno is, from measured components only:**

| | value | vs bar |
|---|---|---|
| decode at **full 262,144 context** | **131 tok/s** (7.65 ms/token) | **in band** |
| prefill | **not yet measured at a comparable chunk size** | unknown |

The decode figure is measured, not modelled: the indexer's 3,183.8 us/token is at
65,536 blocks = 262,144 tokens x block 4, i.e. the *full* advertised context, and GDN is
measured, and experts are arithmetic. It rests on one assumption that I-026 Item 1 must
confirm: that QSA's 2048-key cap actually bounds `attn_decode`.

**The earlier "2,690 tok/s prefill" figure is withdrawn.** It was computed at M=2048 and was
unfair to us in two separate ways:

1. **Chunk size.** All 512 experts are hit at M=2048 *and* at M=16384, so the 1.4157 GB of
   expert weights per layer is read either way — 8x more tokens per byte at 16 K. The peer
   engine's optimum is 16,384 and our only measurement is at 2,048, so the two numbers were
   never comparable.
2. **The PLE term.** 90 us *per token* x 2,048 = 184 ms, which I charged at 24 % of prefill.
   That is a serialised table lookup per token; batched it is ~0.1 ms for the whole chunk.
   Charging it per token made prefill look 24 % worse than the design intends.

### Measured at last: MoE across chunk sizes (M-001, 1 GPU, EP=1, warm cache)

| M | block us | routed us | routed TFLOPS | % of 192.5 | shared us | dispatch us | combine us | block x48 us/token | MoE-only tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 9,321.6 | 4,647.9 | 43.32 | 22.5 % | 2,573.0 | 1,528.4 | 587.3 | 218.5 | 4,577 |
| 8,192 | 29,486.9 | 11,294.7 | 71.30 | 37.0 % | 10,189.5 | 5,777.0 | 2,355.8 | 172.8 | 5,788 |
| 16,384 | 57,728.4 | 21,477.4 | **74.99** | **39.0 %** | 20,261.8 | 11,466.0 | 4,708.4 | **169.1** | **5,913** |
| 32,768 | 116,348.2 | 41,887.7 | 76.90 | 39.9 % | 40,569.5 | 24,645.8 | 9,416.3 | 170.4 | 5,867 |

Term sum vs block is 100.1-100.4 % at every size, so the decomposition is a control, not a guess.

**Three corrections to this file, all of them mine:**

1. **"MoE runs at 17 % of FP4 peak" was wrong** — 32.1 TFLOPS was *dispatch + routed*, not the GEMMs.
   Routed alone is **43.3 TFLOPS (22.5 %)** at 2K and **75.0 (39.0 %)** at 16K. I conflated a
   composite with a kernel efficiency, which is the same error as conflating whole-block with
   routed-only.
2. **The chunk-size lever is not 8x for warm compute.** Warm block throughput rises 219.7 K to
   283.8 K tok/s (+29 %); routed-only rises +73 %. The block is **occupancy/compute-bound, not
   weight-read-bound**. The 8x argument applies to the *DRAM tier*: cold fetch falls from 95.7 to
   11.96 us/token between 2K and 16K.
3. **The 3,550 tok/s projection was too pessimistic.** Anchored at 16K: MoE 169.1-176.0 us/token
   + attention 57.8 + ~0 PLE = **226.9-233.8 us/token = 4,277-4,407 tok/s** on **one GPU**.

**Topology caveat, stated because it is the whole comparison:** all rows are **1 GPU, EP=1, no TP,
no replication**. The peer engine's 5,000 tok/s is a **4-GPU** warm figure. We are at ~4.3-4.4 K
prefill on a single card; a like-for-like comparison is not yet available and must not be implied.

**Cold tier, measured rather than assumed:** 7.2 GB/s across pinned host DRAM (near the Gen4 x4 link
rate of ~7.9 GB/s). The production 4.30 GB per-rank overflow therefore costs **0.595 s/chunk =
36.4 us/token at 16K** (290.5 us/token at 2K) — per rank. The all-cold rows are an upper bound
(1.4156 GB/layer is ~16x the production overflow) and must not be multiplied by 48 as a production
figure.

**The one lever that reaches 5,000:** at 16K, shared expert + combine is **73-76 us/token (43 % of
the block)** with pure shared at 59-62 us/token across 48 layers. Killing the shared FFMA cost is
worth up to 59-62 us/token and even halving it reaches **~4,930-5,070 tok/s**. That is the single
highest-value prefill change and it remains quality-neutral: the BF16 shared expert only sits on the
FFMA path because the 5e-4 exactness gate forbids quantising it.

**Cache warmth is our DRAM tier, and they pay it too.****Cache warmth is our DRAM tier, and they pay it too.** The peer engine's 5,000 is a warm
number. Our 4.30 GB expert overflow is read once per chunk: 2.1 MB/token at chunk 2,048
versus 262 KB/token at 16,384. Large chunks are not a tuning knob — they are what makes the
DRAM tier survivable at all. Every prefill figure we report must state cold or warm.

### The three levers, in payoff order

1. **MoE prefill runs at 17 % of FP4 peak** (32.1 TFLOPS routed of 192.5), and **3.3 ms of
   its 9.5509 ms block is the BF16 shared expert on the FFMA path** — 35 % of the block doing
   scalar math because the 5e-4 exactness gate forbids quantising it. A tensor-core shared
   path plus better grouped-GEMM occupancy is worth ~3x and is the only route to 5,000.
2. **Batch the PLE gather.** 90 us/token x 2,048 = 184 ms per chunk, i.e. 24 % of prefill,
   for a table lookup. One batched gather replaces 2,048 serialised ones.
3. **Speculation** stays the decode lever: k=4 at ~2.5 accepted takes 131 tok/s to ~330, with
   the known landmine that GDN state updates in place and needs a 113 MB/seq snapshot per round.

Quality-neutral, all three. The quantisation ceiling does not move.

## 7d. Invariant triage (user directive 2026-11-24)

When a bug is found, the fix is not the fix. Classify the *hazard*, then pick the cheapest
mechanism that makes recurrence impossible or loud, in this order:

1. **Make the invalid state unrepresentable.** Comptime-checked shapes, capability handles whose
   methods do not exist in the wrong state, single-owner types with private fields. Zero runtime
   cost, so prefer it whenever it is expressible.
2. **Runtime guard, if cheap.** A per-launch epoch check, a canary band, a NaN poison, an
   assertion on a declared invariant. Accept it when the cost is bounded and it fires on the first
   run rather than the fourteenth. Debug-only is fine when release cost is not acceptable, provided
   the debug form fails loudly.
3. **Documented contract + a test that can see the violation.** Last resort, and only for hazards
   where 1 and 2 are genuinely inexpressible — device-side OOB is the standing example.

**Never accept a blanket operation that hides a class of bug.** Zeroing every buffer before use is
the worked example: it makes a missing write invisible, converts a crash into a plausible wrong
answer, and costs 3 GiB per reset at full context. It is rule 3 masquerading as rule 2.

Corollary: the guard must be easier to satisfy than the workaround, or lanes route around it.
Three lanes asked to edit `core/build.zig` to dodge a fence in one night.

## 7e. Bit-exactness is a test technique, not a requirement (user correction 2026-11-24)

I had been treating **bit-identical output versus our own single-GPU path** as an architecture gate
and letting it choose the parallelisation. It is neither: the user's targets are 262,144 context,
>= 150 tok/s decode, >= 5,000 prefill and quality-neutral quantisation. **Bit-exactness is my
testing convenience.**

Separate the two claims that were being conflated:

| claim | status |
|---|---|
| greedy ids equal the **HF golden** ids | **a real requirement** - it is the correctness gate |
| ids bit-identical to **our own single-GPU run** | a **test technique**: it makes a mismatch unambiguous (fault is in placement, not kernels) at a cost worth paying *only* when it is free |

Consequences:
- Gates compare against the fixture/oracle, never against a second configuration of our own code.
  "The 4-rank path must match the 1-rank path exactly" may be used as a **diagnostic**, but a
  placement choice must never be justified by it alone.
- A parallelisation is chosen on **measured** grounds - bytes on the link, redundant compute, occupancy
  - and the resulting numerical difference from another configuration is reported, not gated.
- The replacement for bit-exactness as an architecture gate is the fp32 oracle plus the output-format
  ulp budget; see the tier-1 tolerance rule in I-013/#8.

This matters because the difference is not free. At our measured 7.2 GB/s PCIe:
column-parallel + all-gather of a 1,280-wide bf16 intermediate is ~1,920 B/token/rank, while
row-parallel + all-reduce of a 2,560-wide activation is ~7,680 B/token/rank - but all-gather then
duplicates the whole down-projection on every rank (~53 ms per 48-layer 16K chunk of redundant
compute). Either can win; **measure it, do not derive it from a test property.**

## 7f. Two layers of assurance (user direction 2026-11-24)

Everything gated in this project so far is **kernel validation**: a kernel against the fp32 oracle
with identical inputs, in ulp/relative terms. That answers *"does this kernel implement the same
math."* It does not answer *"is the approximating pipeline still the same model"* — which is what FP8
KV, NVFP4 experts and a TP split actually risk.

Both layers are required and they are not interchangeable:

| layer | question | instrument |
|---|---|---|
| kernel validation | is the math right? | fp32 oracle, identical inputs, ulp/relative bounds |
| **model quality** | is it still the same model? | **KL vs reference, ΔPPL, per-layer relative RMS, router top-k agreement** (I-031) |

**KL is the grader; per-layer cosine is the localiser.** Cosine is scale-invariant and saturates —
a quietly degraded model still reads 0.999+ — so it may be gated only on regression against a
recorded baseline, never on an absolute value. 0.99 → 0.5 is a real signal; 0.9995 → 0.9985 is not.

Any quality threshold must be derived from the **reference's own two-run noise floor**, not invented.
An earlier tolerance here was set at 1e-6 when the reference legitimately moved 4.5e-8 with thread
count; a gate tighter than the reference's variance is decoration.

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
