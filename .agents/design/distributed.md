# Design: 4-GPU execution on one bifurcated PCIe 4.0 x16 (4 × x4), P2P enabled

Status: binding contract for issues I-13 onward. Numbers marked (M) come from
`bench/hw-probe.json` (I-02) and must be re-measured, never assumed.

## 1. Why this shape and not another

All four GPUs share **one** Gen4 x16 link (≈25 GB/s raw, ~20 GB/s effective, in
*aggregate*, M). Device↔device goes over the same root complex (P2P works, but
bandwidth is that same shared budget). Consequences:

- **Per-token VRAM-resident bytes dominate.** Never put per-token-read weights
  outside VRAM. (THEORY.md §2: 9.92 GB/token as shipped.)
- **Aggregate GDDR7 bandwidth is only useful if a single layer's weights are
  split across GPUs** (tensor parallel), not if layers are split (pipeline
  parallel): a single token traverses every layer sequentially, so pipeline
  parallel would make decode latency `bytes / per-GPU_bw`, i.e. 4× worse.
- **Communication must be latency-optimised, not bandwidth-optimised.** The
  per-token payload is tiny (a 2560-element bf16 activation is 5 KiB); the cost
  is the number of round trips (~10 µs each over P2P, M).

## 2. Placement

- **Dense (attention / linear-attention / hyper-connections / router / shared
  expert): tensor parallel, TP = 4.**
  - `in_proj_qkv [10240,2560]`, `in_proj_z [6144,2560]`, `out_proj [2560,6144]`,
    `q_proj [12288,2560]`, `k/v_proj [512,2560]`, `o_proj [2560,6144]`,
    `hc down/up [320,10240]/[10240,320]`, `shared_expert [640,2560]×2 / [2560,640]`:
    split the **output** dimension column-wise (`out_features/4` per rank), keep
    `in_features` whole; the corresponding `out_proj`/`down_proj` are split
    row-wise so a single all-reduce per sub-block completes the operation.
  - Head-parallel where natural: 24 q-heads → 6 per rank; 2 kv-heads cannot be
    split (GQA) → **replicate k/v_proj** (0.031 GB total, negligible) on every
    rank, or split the KV head's 256 dims and all-reduce. Replication is cheaper.
- **Experts: expert parallel, EP = 4.** 512 experts → 128 consecutive experts per
  rank, all 48 layers. `router` is replicated (0.126 GB) so every rank computes
  the same top-10 independently — **no routing all-gather is needed**.
- **`lm_head`**: row-split (each rank computes `vocab/4` logits) then a
  partial-argmax/all-gather of the top-k scores; never materialise full logits on
  one rank when sampling on GPU.
- **`embed_tokens`**: host-resident, gather 1 row/token (5 KiB), H2D once per
  prefill and once per decode step. (Or: replicate per-rank the 248320×2560 bf16
  table only if VRAM allows after measurement — 1.27 GB × 4 is too much.)
- **PLE n-gram table**: NVMe mmap, gather 16 rows/token on the host, H2D 2.5 KiB.
- **States**: GDN recurrent state (113 MB/seq) is **not** split — it lives on the
  rank that owns the v-heads (48 v-heads → 12 per rank ⇒ 28 MB/seq/rank).

## 3. Per-token communication budget

| message | per layer | per token (48 layers) | bytes |
|---|---:|---:|---:|
| dense all-reduce (attn + mlp outputs, fused) | 2 × 5 KiB | 96 | 480 KiB |
| EP dispatch: token activation to each rank owning a selected expert | 1 × 5 KiB broadcast | 48 | 240 KiB |
| EP combine: weighted partial sums back to all ranks | 1 × 5 KiB all-reduce | 48 | 240 KiB |

≈1 MiB/token of PCIe traffic — 50 µs of *bandwidth* but ~1–1.5 ms of *latency*
at ~10 µs/round-trip if issued naively. Therefore:

1. **Batch the two all-reduces per layer into one** (concatenate attn+mlp outputs
   before the reduction → one 10 KiB reduce instead of two 5 KiB ones).
2. **Overlap dispatch with the previous layer's dense compute** using separate
   streams and events (dispatch for layer L+1 depends only on layer L's output).
3. Use **one unidirectional ring per direction** over P2P with `cuMemcpyPeerAsync`
   on dedicated streams; never bounce through host memory.
4. Target: **≤1.5 ms/token** of measured communication, enforced by a benchmark
   gate (`bench/comm.json`) before any hill-climb claims count.

## 4. MoE dispatch protocol (EP = 4, top-10, 512 experts)

1. Every rank computes the router (replicated weights) → identical top-10 ids and
   weights; no communication.
2. Each rank builds, for each destination rank, the list of (token, expert, weight)
   assignments it owns. Because all ranks have identical routing, the send list
   per destination is derivable **without an all-to-all of ids**.
3. `cuMemcpyPeerAsync` the packed activation block to each destination (at most 3
   peers, all issued concurrently).
4. Each rank runs its owned experts' grouped FP4 GEMM, scales by the routing
   weight, then **all-reduces** the (partial) output so every rank ends with the
   full sum. (All-reduce rather than gather-then-scatter: same bytes, one op, and
   it keeps the next layer's dense path ready on every rank.)
5. Add the shared-expert output (computed locally, TP) and continue.

Prefill uses the same protocol with tokens × top-10 rows; the history must be
grouped by expert before the GEMM (sorted permutation on device).

## 5. Correctness contract

The multi-GPU path must be **bit-comparable with the single-GPU path** (I-12) on
greedy generation: same token ids for ≥32 tokens on the tiny model. Any deviation
means a placement or reduction bug, not "numerics".

## 6. Failure modes to design for

- P2P unavailable between a pair at runtime → fall back to host staging for that
  pair, with a loud warning in metrics (`inferno_p2p_fallback_total`).
- One GPU OOMs → refuse to start (do not silently degrade); print the placement
  report with per-rank byte totals.
- A rank's kernel fault must abort the whole process (no partial state).
