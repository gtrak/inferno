# I-035 — testing strategy: what is a test, what is a measurement, and where the wall clock goes

## What is already done (measured, this session)

| | before | after |
|---|---|---|
| `zig build test` (bands on, default env) | **107 s** | **59 s** |
| tests | 815 | 815 — **nothing removed** |
| graded assertions | unchanged | unchanged |

Two changes achieved it, both principled rather than by deletion:

1. **`moe.zig`**: the chunk sweep ran M = 2048/4096/8192/16384/32768. Only the log differed per size;
   the assertions run on the first. The sweep is now `{2048}` unless **`INFERNO_BENCH=1`**.
2. **`gemv.zig`**: only `shapes[0]` (10240x2560) is **graded** — the file literally `continue`s for
   every other shape after printing throughput. The 248320x2560 lm_head case alone cost **21 s** in
   fp32 oracle preparation (636M elements, 2.5 GB host) for a line nobody gates on. Default is now
   `{10240, 2560}` + `{40, 2560}`; the rest need `INFERNO_BENCH=1`.

**Measurement trap found along the way, twice:** `zig build -Doptimize=ReleaseSafe` does **not** build
the test artifact, so `ls -t .zig-cache/o/*/test` silently returned a stale binary and showed "no
improvement" after a real improvement. And piping `zig build test` through awk gives **one timestamp
for every line** — zig writes in a burst, so per-test timing only works by running the **test binary
directly**. Any future timing claim must state which of these two mistakes it avoided.

## The four things this issue is for

### 1. Remove tests that are not tests
A *test* asserts. A *measurement* reports. Right now they are mixed in the same function, so a
throughput line's cost is paid by every gate run and its number is taken on whatever box happens to be
busy — which `plan.md` §7c forbids. **Rule: if a case only calls `report*` and never `expect*`, it is a
measurement and must not run by default.** Ongoing audit: every `for (shapes) |shape|` loop where the
body only reports.

### 2. Separate slow from fast
Add explicit targets so a lane can run only what it owns:
`zig build test-fast` (no GPU, no sweeps), `zig build test-kernels`, `zig build test-model`,
`zig build test-bench` (the sweeps, quiet box only). `zig build test` stays the full gate.

### 3. Parallelise across the four GPUs
See I-034: only `model/multi.zig` and `mem/placement.zig` need all four ranks. Everything else is
single-device and currently pinned to device 0 while three cards idle. Same-device concurrency stays
forbidden — that is the flake actually measured (2-of-5 failures).

### 4. Root cause: loading the full checkpoint should be milliseconds, and is not
The I-017 branch's `weights.shard.test.the real checkpoint's shard plan is sized against measured VRAM`
was the single slowest thing observed (>150 s), and it begins with `manifest.build()` over
**299,545 tensors / a 31 MB index**.

**Leading hypothesis, not yet confirmed:** per-tensor `classify()` is a chain of prefix comparisons and
each hit calls `place()`; at 299k tensors that is millions of comparisons plus any per-tensor
allocation. Under `std.testing.allocator` — which validates every allocation and tracks frees — that
alone can turn a ~50 ms build into tens of seconds.

**Decisive measurement, in this order:**
1. Wrap `manifest.build` in a timing harness (arena allocator, as `manifest_main.zig` already does) and
   print per-phase times: index parse, `mapFiles`, the tensor loop, `validateNgramTable`,
   `validateCompleteness`. That alone may end the investigation.
2. Count allocations in the tensor loop with a counting allocator. If it is ~millions, the fix is to
   make `classify` allocation-free (parse the layer index into a stack buffer; `std.mem.indexOfScalar`
   over a fixed separator instead of splitting) and to run heavy manifest tests on an **arena**, not
   `std.testing.allocator`.
3. If the loop is clean, the cost is in `safetensors` header parsing of ten 12 GB shards — check
   whether headers are read once or re-scanned per tensor.

**Acceptance for this item: `manifest.build` on the real checkpoint under 500 ms**, and a test that
asserts it, so the next regression is visible.

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 600 zig build test) ; echo CORETEST=$?
INFERNO_BENCH=1 (cd core && timeout 1800 zig build test) ; echo BENCH=$?   # quiet box only
```
