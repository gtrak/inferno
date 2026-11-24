# I-06b — prefill GEMM: native block-scaled NVFP4 (`OMMA.SF`) + exact BF16 reference path

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

- Sole writer in your worktree. `.agents/skills/tiger-style/` binding for Zig;
  CUDA C++ with named constants, checked launch configs, `__restrict__`,
  `__launch_bounds__`.
- Builds on I-01 (nvcc pipeline), I-02 (measured bandwidths), I-03
  (`weights/quant.zig`), I-05 (`ref/ops.zig` matmul oracle), I-06a (GEMV family).
- **Read `.agents/design/mma-sm120-nvfp4.md` before writing a line of CUDA.** It
  has the exact PTX operand list, the SASS, the thread-fragment layouts and the
  nibble-placement trap. The vendored CUTLASS headers it cites are in
  `.agents/reference/cutlass/` — read-only documentation, never compiled.
- Settled facts: `sm_120a` has native block-scaled FP4 MMA in two scale formats
  (`ue4m3` = NVFP4 per-16, matching the checkpoint; `ue8m0` = MXFP4);
  `-arch=sm_120a` is mandatory.
- Prefill is compute bound (THEORY.md §3). Report achieved TFLOPS against measured
  peak **and** correctness against the f32 oracle.

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
- Three kernels, one API:
  1. **`gemm_nvfp4_mma`** — the fast path: `OMMA.SF` m16n8k64 with `ue4m3` scales
     (`ALayout`/`BLayout`/`SFALayout`/`SFBLayout` as documented). Activations are
     quantised to NVFP4 on the fly (per-16 dynamic scales, `scale_vec::4X`),
     `weight_scale_2` folded into the epilogue. **This is a labelled W4A4 quality
     trade** — see the gate.
  2. **`gemm_nvfp4_bf16`** — the exact reference path: dequantise fp4×scale to
     BF16 in shared memory (lossless: e2m1 × e4m3 needs ≤6 mantissa bits) then
     `m16n8k16` HMMA. Slower, and the numerical baseline.
  3. **`gemm_fp8_e4m3`** and **`gemm_bf16`** — plain kernels for the dense path
     and for testing.
- Tile `BM=128, BN=128` (adapt to `K`), 3–4 stage `cp.async` pipelining,
  swizzled shared-memory layout, `ldmatrix` where the format allows it.
- **Weight repacking (offline, at load):** convert the checkpoint's row-major
  `[N, K/2]` (low nibble first) + `[N, K/16]` E4M3 scales into the MMA fragment
  order for `A` and `SF`, so the hot loop issues no shuffles. Ship it as
  `core/src/weights/pack_nvfp4.zig` with a Zig unit test. **Validate the repack
  numerically against the f32 oracle — never by inspecting the layout by eye.**
  If the repack is wrong the oracle test fails; fix the repack, not the oracle.
- Support `M` from 1 to 8192. For `M < 16` prefer dispatching to the I-06a GEMV
  rather than running a degenerate tile.
- `K % 64 == 0` and `N % 64 == 0` asserted host-side (the real model satisfies
  this). Accumulate in `float`.

### 2. `core/src/kernels/gemm.zig`

Typed launch wrappers with assertions + a test suite (reached through the
`c_api.zig` import; there is no per-kernel build step) that, for
every kernel, compares against the f32 oracle at these shapes:
`M ∈ {1, 16, 128, 512, 2048}` × `(N,K) ∈ {(10240,2560), (6144,2560), (2560,6144), (12288,2560), (640,2560)}`.

Report per case: `max_abs_err/max|ref|`, achieved TFLOPS, and the achieved
fraction of the peak from `bench/hw-probe.json`.

### 3. Peak throughput measurement

Add to `core/src/bench/main.zig` (the I-02 probe binary) a peak measurement at
`M=N=K=4096` for bf16 (`HMMA`), fp8, and NVFP4 (`OMMA.SF`), median of 6, written
into `bench/hw-probe.json` as
`gemm_peak_tflops: {bf16, fp8, nvfp4_mma, nvfp4_dequant_bf16}`.
**These replace the M-todo numbers in THEORY.md §3 — report them.**

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build test) ; echo CORETEST=$?
(cd core && timeout 600 ./zig-out/bin/inferno-probe >/dev/null && python3 -c "
import json; d=json.load(open('../bench/hw-probe.json'))
print(d['gemm_peak_tflops'])") ; echo PEAK=$?
```

Acceptance:
1. Numerical — a **three-tier ladder**, because kernel correctness and the engine's
   dtype policy are different questions (`.agents/tasks/review-gate.md` §5):
   - **tier 1, operand-exact** (both sides get the kernel's own quantised
     operands): the kernel machinery — repack, fragment layouts, SFA/SFB, epilogue —
     must match at **the output format's precision**: `ulp_error <= 0.5` (correctly
     rounded bf16), or ≤1e-6 only if the output is f32. This is the real correctness
     gate.
   - **tier 2, W4A16** (`gemm_nvfp4_bf16` vs the oracle on dequantised weights,
     activations bf16): `max_abs_err/max|ref| <= 5e-4`.
   - **tier 3, W4A4 vs the original fp32 weights**: **reported, NOT asserted.**
     E2M1 is a 3-bit grid, so both-operand quantisation gives ~10 % per-element RMS
     and, for iid data, dot-product error scales with sqrt(K) exactly like the
     signal — it does not shrink with K. 1.1e-1..1.4e-1 is the *correct* value for
     any correct W4A4 kernel, confirmed independently by the gate reviewer's
     from-scratch numpy oracle (1.374e-1). The earlier `<= 2e-2` wording here was an
     error in this spec: it describes the exact path, not W4A4.
     W4A4 quality is gated at **I-020** (ΔNLL, router top-k agreement). If it fails
     there the fallback is `kind::f8f6f4` e2m1×e4m3 (W4A8), available on this GPU.
   Also required: internal repeatability (same input ⇒ bit-identical output).
2. Performance floors **must be asserted, not merely printed** — an unasserted
   floor is decoration (gate finding). Assert at **half** the intended floor so a
   catastrophic regression fails while run-to-run noise on a shared GPU does not:
   `gemm_nvfp4_mma` at `(M=2048,N=10240,K=2560)` ≥ 10 % of measured NVFP4 peak;
   `gemm_nvfp4_bf16` ≥ 20 % of measured bf16 peak; `gemm_fp8_e4m3` ≥ 15 % of
   measured fp8 peak. Print the achieved percentage and the intended (20/40/30 %)
   figure alongside. Report real numbers even when a floor is missed, with a
   profiler-free explanation (occupancy, smem, dequant or repack overhead).
3. Non-vacuity: print `C[0][0..3]`, verify the output is not constant, and verify
   `M=1` and `M=2048` both agree with the oracle (guards a batch-dependent bug).
4. Report the ratio `tflops(gemm_nvfp4_mma) / tflops(gemm_nvfp4_bf16)` — this is
   the measured payoff of native FP4 MMA and the input to the prefill ceiling in
   THEORY.md §3.

## Report format

Implemented / Changed files / Validation (error + TFLOPS table, and the measured
peaks) / Open risks / Recommended next step (what you would tune first).
