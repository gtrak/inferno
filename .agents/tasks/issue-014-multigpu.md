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
`core/src/model/multi.zig` (+ `core/src/mem/placement.zig`), plus two purely additive `hw/`
seams those files need: `core/src/hw/cuda_driver.zig` and `core/src/hw/cuda_context.zig`
(`cuMemcpyPeerAsync` and peer access; 11 + 31 insertions, 0 deletions). Engine code must not
name a CUDA symbol — that is `tools/gate.sh`'s structural invariant — so peer copy can only
live behind `hw/`. The files are named here so the next reviewer does not read a legitimate
change as scope creep.

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
(cd core && timeout 2400 zig build test) ; echo CORETEST=$?
```
There is **no named build step** for this issue and there will not be one: `tools/gate.sh`
runs `zig build test`, never a named step, so the check has to be a normal test block
reachable from `c_api.zig`'s import list. This block used to specify `zig build multi-check`,
which `core/build.zig` has never had — it returns `error: no step named 'multi-check'` and
exits 1. Do not specify a named step again.

1. **Token-for-token equality with I-013**: the same greedy ids for 16 tokens on the
   tiny model, 4 ranks vs 1 rank. Any divergence is a placement/reduction bug.
   **NOT DELIVERED — carried to I-014b** (no 4-rank forward pass exists yet; see Status).
2. Comms benchmark printed and written to `bench/comm.json`: per-layer all-reduce
   latency and MoE dispatch/combine latency, median of 6. **Assert the per-token
   comms total ≤ 1.5 ms.**
   **PARTIAL — carried to I-014b** (see Status). Every ordered pair's peer copy is measured
   and written; `liveness_ok` is a link check, and the 1.5 ms per-token assert is not made.
3. Placement report shows every rank under its measured VRAM total.
4. Non-vacuity: print all 4 ranks' device names and that each holds 128 experts.

## Report format
Implemented / Changed files / Validation (equality with I-013, comms numbers) /
Open risks / Recommended next step.

## Status — PARTIAL, amended 2026-11-24

**Delivered (plumbing pass; the tree is green, acceptance is the orchestrator's call):**
`core/src/mem/placement.zig` (1,004 lines, 15 tests) and `core/src/model/multi.zig` (2,055 lines,
17 tests) — per-rank placement from measured VRAM,
the refusal path, the host-DRAM expert tier with two-tier LRU, prefetch double-buffering with two
pinned staging slabs and per-slot CUDA events, the peer-copy matrix in `bench/comm.json`, and the
placement/non-vacuity report (gates 3 and 4).

**One defect found and fixed across two passes.** The prefetch raced: an async copy could have
its source slab refilled underneath it. The double-buffer fix left a second, narrower hole — each
staging slab's event was recorded beside the copy that read it, so everything the batch queued
after that point (the last move's copies and the writebacks behind them) fell outside it, and a
next batch's refill could read a host slot the previous batch was still writing back. The fix
re-records every staging event after `ready.record`, so a refill waits for the whole batch.
Pinned by a GPU round-trip test that measures stream time between `bounce_free[1]` and a probe
queued after the batch: 511 ns with the fix against 729 920 / 729 951 ns for one move of copies
in two runs, while an independent replay of the unfixed sequence gives 304 127 ns against a
294 943 ns control — the assertion fails without the fix.

**Carried forward to `.agents/tasks/issue-014b-multigpu-forward.md`:**
- item 2, the 4-rank forward pass with per-layer `cuMemcpyPeerAsync` on dedicated streams
- gate 1, 16-token greedy equality against I-013
- gate 2's per-token `<= 1.5 ms` assert (the peer-copy matrix and the liveness check are here)

**Why it is a dependency and not a preference:** item 2 needs `model/exec.zig`. There is no
`model/plan.zig` / `exec.zig` / `state.zig` / `generate.zig` anywhere on this base — I-013 has not
merged — so a forward pass written here would compile in the worktree and break on `main`. Gate
2's assert additionally needs the fused dense all-reduce and the MoE dispatch/combine, and
`core/src/kernels/` has no vector-add or reduction kernel to build them from, so it is I-014b
work plus a quiet box (another lane on the GPUs breaks a 1.5 ms bound).

**Measured numbers this lane produced, which are load-bearing for the budget** (arithmetic, not
throughput claims): KV 3 GiB per rank at 256 K (12.00 GiB replicated x4); QSA tail
25,165,824 B of KV read per token; 4,499 expert slabs hot at the full-card upper bound and 4,388
on measured rank-0 free VRAM; 120 slabs fetched per token; 12 reachable ordered pairs, slowest
1 MiB peer copy 149.056 us (max over the 12 pairs, median of 6 each).
**No throughput claim is made anywhere in this lane.**
