# I-014 — 4-GPU execution: replicated attention + FP8 KV stored once, EP experts, DRAM tier

## Read first (binding)
`.agents/design/full-context-single-stream.md` and `.agents/design/distributed.md`,
then `THEORY.md` §4. The user's target is **single-stream, full 256K context,
≥150 tok/s decode, ≥5000 tok/s prefill**. The allocation below follows from that and
supersedes conventional tensor parallelism.

## Working agreement
- Sole writer. tiger-style binding. You run on all 4 GPUs.
- Do not edit `core/build.zig`; add exactly one `_ = @import("model/multi.zig");`
  line to the test block in `core/src/c_api.zig`.

## The placement (decided — implement, do not re-litigate)
| component | placement |
|---|---|
| attention + indexer (BF16/FP8, 1.24 GB) | **replicated on all 4 ranks** |
| KV + indexer key cache | **FP8, stored ONCE** in a single contiguous per-sequence arena; every rank reads the same arena (replicate the *buffer*, not the compute: each rank holds its own copy only if it cannot address rank 0's — prefer P2P reads, else replicate at 1/4 the precision cost) |
| dense value path (linear_attn, hyper, shared expert) | **TP=4**, one fused all-reduce per layer for both sub-blocks |
| experts (512, MXFP4) | **EP=4**, 128 consecutive experts per rank, all 48 layers |
| router | replicated (identical top-k on every rank ⇒ no routing all-gather) |
| `lm_head` | row-split, on-device argmax reduction across ranks |
| GDN state | on the rank owning the v-heads (12 v-heads each) |

Because attention is replicated, **every rank produces the identical post-attention
hidden state, so there is no attention all-reduce.** The only per-layer comms are
the fused dense all-reduce and the MoE dispatch/combine.

## What to build
`core/src/model/multi.zig` (+ `core/src/mem/placement.zig`):
1. Per-rank context/stream/module setup; P2P enablement with a loud fallback.
2. The forward pass over 4 ranks with the placement above; per-layer comms via
   `cuMemcpyPeerAsync` on dedicated streams, event-ordered, overlapped.
3. A **placement report** printed at init: per-rank byte totals for weights, KV and
   state, plus the DRAM tier size, refusing to start if any rank would exceed its
   VRAM (measured, not assumed).
4. The **DRAM expert tier**: experts that do not fit stay in host RAM (pinned,
   `cuMemHostAlloc`) and are prefetched asynchronously on the router result;
   an LRU hot set of tunable size. Include a `--no-dram-tier` flag that refuses to
   start instead of silently OOMing.

## Gates
```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 2400 zig build multi-check) ; echo MULTI=$?
```
1. **Token-for-token equality with I-013**: the same greedy ids for 16 tokens on the
   tiny model, 4 ranks vs 1 rank. Any divergence is a placement/reduction bug.
2. Comms benchmark printed and written to `bench/comm.json`: per-layer all-reduce
   latency and MoE dispatch/combine latency, median of 6. **Assert the per-token
   comms total ≤ 1.5 ms.**
3. Placement report shows every rank under its measured VRAM total.
4. Non-vacuity: print all 4 ranks' device names and that each holds 128 experts.

## Report format
Implemented / Changed files / Validation (equality with I-013, comms numbers) /
Open risks / Recommended next step.
