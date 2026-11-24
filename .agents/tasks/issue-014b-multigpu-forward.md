# I-014b — multi-GPU forward pass, peer comms, and the comms gate

Carried forward from **partial I-014** (plumbing merged: `mem/placement.zig`,
`model/multi.zig`, prefetch double-buffering with per-slot events). This issue is the rest.

**Blocked on I-013 merging.** `model/exec.zig` lives only in slot-1's worktree; the 4-rank
forward cannot be written against it until it is in main. Gate 2 is additionally blocked on a
**quiet box** (see the metric rules in `plan.md` §7c) — the `<= 1.5 ms` comms assert must not be
taken while another lane is running GPU work, or it will fail for reasons unrelated to the code.

## Work

1. **4-rank forward pass**, one layer at a time, with per-layer `cuMemcpyPeerAsync` of the
   activation between stage boundaries on dedicated streams. Pipeline-parallel across ranks for
   the dense path, EP for experts, replicated attention and router (the placement that
   `placement.zig` already decides as data).
2. **Gate 1: 16-token greedy equality** against I-013's `greedy_ids.json`. Same ids as the
   single-GPU path, bit-identical. If they differ, the fault is in the placement or the peer
   copy, not in the kernels — bisect per layer.
3. **Gate 2: comms bench** — four-rank `cuMemcpyPeerAsync`, `bench/comm.json`, assert
   `<= 1.5 ms` per token of cross-stage activation traffic at the production activation size
   (hidden 2560, bf16, so 5 KiB per token per boundary x 3 boundaries).

## Known numbers to respect

- KV **3 GiB per rank** at 256 K (12.00 GiB replicated x4).
- **QSA tail 25,165,824 B** of KV read per token (12 layers x 2048 keys x 1 KiB).
- Experts: 67.95 GB across 4 ranks, 4.30 GB overflow to host DRAM at 256 K.
- Measured P2P latency 6.8 us; per-GPU HBM 397.6 GiB/s (95 % of spec).
- Activation per boundary at hidden 2560 bf16 = 5,120 B/token. At 4 boundaries that is ~20 KiB
  per token, so `<= 1.5 ms` is a very loose bound — if it fails, something is serialising.

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build test) ; echo CORETEST=$?
```
There is **no named build step** for this issue and there will not be one: `tools/gate.sh` runs
`zig build test`, so the check must be a normal test block reachable from `c_api.zig`'s import
list. Three issues have now specified a named step by mistake; do not repeat it.
