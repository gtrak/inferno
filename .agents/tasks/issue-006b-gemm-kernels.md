# I-06b — prefill GEMM: FP4→BF16 dequant + HMMA, and native FP8 GEMM

## Working agreement

- Sole writer in your worktree. `.agents/skills/tiger-style/` binding for Zig;
  CUDA C++ with named constants, checked launch configs, `__restrict__`,
  `__launch_bounds__`.
- Builds on I-01 (nvcc pipeline), I-02 (measured bandwidths), I-03
  (`weights/quant.zig`), I-05 (`ref/ops.zig` matmul oracle), I-06a (GEMV family
  and the MMA probe result).
- **Settled fact (do not re-litigate):** `sm_120a` has *no* block-scaled FP4 MMA.
  Accepted: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` and
  `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`.
- Prefill is compute bound (THEORY.md §3). The metric is achieved TFLOPS against
  the measured peak, plus correctness against the f32 oracle.

## What to build

### 1. `core/src/kernels/gemm.cu`

```c++
// A: bf16 [M,K] (activations)  B: nvfp4 [N,K] (weights, row-major, low nibble first)
// scales: e4m3 [N, K/16], scale2: per-tensor f32.  C: bf16/f32 [M,N]
__global__ void gemm_nvfp4_bf16(const __nv_bfloat16* A, const uint8_t* B,
                                const uint8_t* Bscales, float scale2,
                                __nv_bfloat16* C, int M, int N, int K);

__global__ void gemm_fp8_e4m3(const __nv_bfloat16* A, const uint8_t* B /*e4m3 [N,K]*/,
                              const float* Bscale /*per-tensor*/, __nv_bfloat16* C,
                              int M, int N, int K);

__global__ void gemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B,
                          __nv_bfloat16* C, int M, int N, int K);
```

Requirements:
- Tile: `BM=128, BN=128` (or 64/128 depending on K), stages = 3, `cp.async`
  double/triple buffering, `ldmatrix.x4`, `m16n8k16` HMMA with fp32 accumulate,
  swizzled shared-memory layout to avoid bank conflicts.
- The NVFP4 kernel dequantises the weight tile **into shared memory as BF16**
  on the fly. This is exact (e2m1 × e4m3 needs ≤6 mantissa bits), so assert in a
  unit test that `dequant(fp4)*scale` round-trips exactly through bf16.
- Support `M` from 1 to 8192 (a small-M path that falls back to a warp-per-row
  GEMV from I-06a is acceptable and preferred over a degenerate tiling).
- `K % 16 == 0`, `N % 64 == 0` asserted host-side (the real model satisfies this).
- Accumulate in `float`; write `bf16` (a `float` output variant for `lm_head`
  comparison is useful — add `bool out_f32` via a separate kernel).

### 2. `core/src/kernels/gemm.zig`

Typed launch wrappers with assertions + a `zig build gemmtest` harness that, for
every kernel, compares against the f32 oracle at these shapes:
`M ∈ {1, 16, 128, 512, 2048}` × `(N,K) ∈ {(10240,2560), (6144,2560), (2560,6144), (12288,2560), (640,2560)}`.

Report per case: `max_abs_err/max|ref|`, achieved TFLOPS, and the achieved
fraction of the peak from `bench/hw-probe.json`.

### 3. Peak throughput measurement

Add to `core/src/bench/main.zig` (the I-02 probe binary) a bf16-GEMM and an
fp8-GEMM peak measurement at `M=N=K=4096`, median of 6, written into
`bench/hw-probe.json` as `gemm_peak_tflops: {bf16, fp8, nvfp4_dequant}`.
**These are the numbers THEORY.md §3 calls M-todo** — report them.

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build gemmtest) ; echo GEMM=$?
(cd core && timeout 600 ./zig-out/bin/inferno-probe >/dev/null && python3 -c "
import json; d=json.load(open('../bench/hw-probe.json'))
print(d['gemm_peak_tflops'])") ; echo PEAK=$?
```

Acceptance:
1. Numerical: `max_abs_err/max|ref| <= 2e-3` for the NVFP4 path (vs oracle on the
   *dequantised* weights) and `<= 5e-4` for bf16/fp8 paths.
2. Performance floors **for a first correct implementation** (tuning is a later
   issue): `gemm_nvfp4_bf16` at `(M=2048,N=10240,K=2560)` must reach **≥ 25 %**
   of measured bf16 peak; `gemm_fp8_e4m3` must reach **≥ 30 %** of measured fp8
   peak; `gemm_bf16` **≥ 40 %** of bf16 peak. Report the real numbers even if a
   floor is missed, with the profiler-free reasoning (occupancy, smem, dequant
   instruction count).
3. Non-vacuity: print `C[0][0..3]`, check the output is not constant, and check
   that `M=1` and `M=2048` both agree with the oracle (guards against a
   batch-dependent bug).

## Report format

Implemented / Changed files / Validation (error + TFLOPS table, and the measured
peaks) / Open risks / Recommended next step (what you would tune first).
