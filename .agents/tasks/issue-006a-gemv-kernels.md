# I-06a — decode GEMV family (NVFP4 / FP8 / BF16) + Blackwell FP4-MMA capability probe

## Launch infrastructure (shared, already written — do not reinvent)

`core/src/hw/kernel.zig` exists on main and is the only way to load a cubin and
launch a kernel. Copy it into your worktree (`cp /home/gary/dev/inferno/core/src/hw/kernel.zig core/src/hw/`)
and add exactly TWO lines to the test block in `core/src/c_api.zig`:
`_ = @import("hw/kernel.zig");` and `_ = @import("kernels/<yours>.zig");`.

```zig
var module = try kernel.Module.load(cubin_bytes);   // decompress-free: bytes are embedded
defer module.unload();
var fn_ = try module.function("your_kernel");       // extern "C" name from the .cu
try fn_.setDynamicSharedBytes(64 * 1024);           // only if you need > 48 KiB/block
var params = [_]?*anyopaque{ @ptrCast(&ptr), @ptrCast(&n) };
try fn_.launch(.{ gx, 1, 1 }, .{ bx, 1, 1 }, shared_bytes, stream.handle, &params);
```

`core/build.zig` already globs `src/kernels/*.cu` into the embedded cubin table, so
adding a `.cu` file needs no build change. **Do not edit `core/build.zig`.**
There is no per-kernel build step: your tests run under `zig build test`.

## Working agreement

- Sole writer in your worktree. `.agents/skills/tiger-style/` binding for Zig.
  CUDA C++ kernels follow the same spirit: named constants, checked launch
  configs, no `printf` in hot code, `__restrict__`, no uninitialised reads.
- Builds on I-01 (`core/build.zig`, nvcc → cubin → `@embedFile`), I-02 (driver
  layer, measured bandwidths in `bench/hw-probe.json`), I-03 (`weights/quant.zig`
  fp4/fp8 decoders), I-05 (`ref/ops.zig` for the CPU oracle).
- Decode is bandwidth-bound (THEORY.md §2): the only thing that matters is
  bytes-read per unit time. Every kernel here must be judged by achieved GB/s
  against the measured HBM bandwidth, not by "it runs".

## What to build

### 1. `core/src/kernels/fp4_mma_probe.cu` — capability probe (10 minutes, then move on)

**Verified by the orchestrator on this machine** (`ptxas` V13.2.86, `cuobjdump -sass`).
Full details, exact PTX strings, operand constraints and SASS: read
`.agents/design/mma-sm120-nvfp4.md`. Summary:

| instruction | SASS | verdict |
|---|---|---|
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | HMMA | ACCEPTED |
| `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` | FP8 | ACCEPTED |
| `mma.sync.aligned.kind::f8f6f4.m16n8k32.f32.e2m1.e2m1.f32` | `QMMA.16832` | ACCEPTED |
| `...m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X....ue4m3` | `OMMA.SF.16864.UE4M3.4X` | **ACCEPTED — native NVFP4** |
| `...scale_vec::2X....ue8m0` (MXFP4, per-32) | `OMMA.SF.16864.E8` | ACCEPTED |
| same instructions with `-arch=sm_120` (no `a`) | — | REJECTED |

Your job on this sub-task: put these lines in `fp4_mma_probe.cu`, add
`-arch=sm_120a` as an assertion in the build, write the verdict table into
`bench/fp4-mma-probe.txt`, and **move on**. Do not re-derive it. If a line stops
compiling, the cause is almost certainly operand kinds (`byte_id`/`thread_id`
are `u16` **immediates**), not a missing architectural feature.

### 2. `core/src/kernels/gemv.cu` — the decode hot path

```c++
// NVFP4 weights, bf16 activations, fp32 accumulate
__global__ void gemv_nvfp4_bf16(
    const uint8_t* __restrict__ w_packed,   // [N, K/2] row-major, low nibble first
    const uint8_t* __restrict__ w_scales,   // [N, K/16] e4m3, one per 16 contracted elems
    float w_scale_2,                        // per-tensor F32 scale
    const __nv_bfloat16* __restrict__ x,    // [K]
    __nv_bfloat16* __restrict__ y,          // [N]
    int n, int k);
```

plus:
- `gemv_fp8_blockwise_bf16(w: e4m3 [N,K], scale_inv: bf16 [N/128, K/128], x, y, n, k)`
  for the FP8 path (used by MTP and by our own requantized dense path, L2);
- `gemv_bf16(w: bf16 [N,K], x, y, n, k)` (router, lm_head fallback, tests);
- a batched variant `gemv_nvfp4_bf16_batch(..., int m)` for decode batches
  (`m ≤ 32`), where `x` is `[m, K]` and `y` is `[m, N]`, so a batched decode does
  not re-read weights per sequence.

Design requirements:
- one warp per output row (or a row-tile with `__shfl` reduction) so the weight
  stream is perfectly coalesced; **vectorised 16-byte loads** (`uint4`);
- dequantise with a 16-entry e2m1→bf16 lookup plus a shared-memory or
  constant-memory scale table; accumulate in `float`;
- `__launch_bounds__`, no shared-memory bank conflicts on `x` (stage `x` into
  shared memory once per block if it helps);
- support arbitrary `K` multiple of 16 (`k % 16 == 0`) and `N` any multiple of 8;
  assert host-side.

### 3. `core/src/kernels/gemv.zig` — Zig launch + validation harness

- Typed launch wrappers (`launchGemvNvfp4(stream, ...)`) that check every
  dimension, alignment and buffer-size precondition with assertions and return a
  closed error set on driver failure.
- `zig build gemvtest` runs, for each kernel: random matrices at the **real
  shapes** — `[10240,2560]`, `[6144,2560]`, `[2560,6144]`, `[12288,2560]`,
  `[640,2560]`, `[2560,640]`, `[512,2560]`, `[248320,2560]` (subsample if VRAM
  is tight, min 2 shapes per kernel), input from `torch.randn`-equivalent
  deterministic PRNG — and compares against the CPU oracle from `ref/ops.zig`
  in f32.

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1200 zig build test) ; echo CORETEST=$?
```

The test suite (reached through the `c_api.zig` import — there is no per-kernel build
step) must print, per shape/kernel, `max_abs_err / max|ref|` on a **three-tier
ladder** (see `.agents/tasks/review-gate.md` §5):
- tier 1 operand-exact (both sides get the kernel's own quantised operands):
  **assert ≤ 1e-6** — repack, LUT decode, block-scale application.
- tier 2 vs `ref` on the **dequantised** weights: **assert ≤ 2e-3**.
- tier 3 vs the **original fp32** weights: **report only, never assert**. Per-16
  E2M1 on Gaussian data cannot beat ~8-10 %; the earlier `<= 2e-2` wording in this
  spec was an error (it describes the exact path, not 4-bit weights).
Plus **achieved GB/s**, median of ≥6 iterations.

Acceptance: for `gemv_nvfp4_bf16` at `[10240,2560]` (3.3 MB of weights) the
achieved bandwidth must be **≥ 55 % of the measured single-GPU HBM bandwidth**
from `bench/hw-probe.json`. If it is below 55 %, iterate on the kernel; report the
best achieved and what the bottleneck was if you cannot reach it.

Non-vacuity: print `y[0..4]` and confirm at least one negative and one positive
value, and confirm two different inputs give different outputs.

## Report format

Implemented / Changed files / Validation (the error+bandwidth table, and the
FP4-MMA verdict) / Open risks / Recommended next step.
