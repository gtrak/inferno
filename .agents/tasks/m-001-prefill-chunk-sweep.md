# M-001 — measure prefill at the chunk size that matters (16 K)

**Status: queued. Do not run while any other lane holds a GPU.** A prefill throughput number
taken under contention is worthless, and taking one after refusing the `desktop` lane's
stability measurement for exactly that reason would be inconsistent.

## Why

Every prefill figure the project currently holds is at **M=2048**, and all 512 experts are hit
at that size *and* at 16 K, so the 1.4157 GB of expert weights per layer is read either way.
The peer engine's optimum chunk is **16,384**; we have never measured above 2,048. The
projection in `plan.md` §7c is therefore unanchored at the one size that matters.

## What to measure

`core/src/kernels/moe.zig` already prints a block-throughput table. Extend the sweep to
M = 2048, 4096, 8192, 16384, 32768 and report, per size:

1. **Cold**: expert weights evicted before the run (the 4.30 GB overflow really comes from
   host DRAM). Use the evict-then-measure device the earlier PLE lane built — warm vs evicted
   on the same binary, with the control shown — rather than `mincore`, which was found to give
   internally inconsistent residency readings under a contended cache.
2. **Warm**: repeat on a resident cache.
3. Per-term breakdown in us/token: routed, shared expert, dispatch/permute, combine.
4. **Routed FP4 efficiency as a % of the 192.5 TFLOPS `OMMA.SF` peak**, at each size. The
   measured 32.1 TFLOPS at M=2048 is 17 % and is the single largest unknown in the projection.
5. The same sweep for the shared expert with the BF16 FFMA path, so the cost of leaving it
   unquantised is a measured number at the target chunk size rather than an extrapolation.

## Reporting rules

- State cold or warm on **every** number. The peer engine's 5,000 tok/s is a warm figure.
- Report the whole-block figure *and* the routed-only figure separately; conflating them is
  what produced the withdrawn 2,690.
- If a size fails to allocate (KV plus activations plus 16 K rows of gathered experts), say so
  with the byte count rather than silently skipping it.
- Do not change any kernel or tolerance. This is a measurement task.

## Gate

`bash tools/gate.sh ; echo EXIT=$?` must stay EXIT=0 — the sweep is additive reporting.
