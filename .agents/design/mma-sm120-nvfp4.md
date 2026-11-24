# SM120 (consumer Blackwell, GB206 = RTX 5060 Ti) tensor-core MMA — verified facts

Verified on this machine: CUDA 13.2 `nvcc`/`ptxas` V13.2.86, `-arch=sm_120a`,
inspected with `cuobjdump -sass`. Read-only CUTLASS reference vendored in
`.agents/reference/cutlass/`.

## 1. The `a` suffix is mandatory

```
-arch=sm_120a   -> accepted, SASS emitted
-arch=sm_120    -> ptxas: "Feature '.kind::mxf4nvf4' not supported on .target 'sm_120'"
```

Block-scaled FP4 MMA is an **architecture-specific** feature. Our build must
therefore use `-arch=sm_120a` everywhere (and `tools/gate.sh` must fail if the
cubin target is not `sm_120a`).

## 2. What compiles, and what SASS it becomes

| PTX (abridged) | SASS | verdict |
|---|---|---|
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | `HMMA` | OK |
| `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` | FP8 MMA | OK |
| `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32` | `QMMA.16832.F32.E2M1.E2M1` | OK |
| `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` | `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` | **OK — this is native NVFP4** |
| `...kind::mxf4nvf4.block_scale.scale_vec::4X....ue8m0` | `OMMA.SF.16864.F32.E2M1.E2M1.E8.4X` | OK (MXFP4, per-16 ue8m0) |
| `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64....ue8m0` | `OMMA.SF.16864.F32.E2M1.E2M1.E8` | OK (per-32 ue8m0 scales) |
| `tcgen05.mma` (5th-gen, `kind::mxf4nvf4`) | — | SM100/103 only, **not** SM120 |

**Conclusion: NVFP4 (per-16 E4M3 block scales, exactly the ModelOpt checkpoint
format) has native tensor-core support on this GPU at m16n8k64.** No
dequantise-to-BF16 is needed for throughput. The checkpoint's `weight_scale`
(E4M3, one per 16 weights) is the format the hardware wants.

## 3. Exact operand structure of the block-scaled MMA

From `mma_sm120.hpp` `SM120_16x8x64_TN_VS` (CUTLASS): `D float[4]`,
`A uint32_t[4]`, `B uint32_t[2]`, `C float[4]`, `SFA uint32_t[1]`,
`SFB uint32_t[1]`, plus two **u16 immediates per operand** (`byte_id`, `thread_id`),
which CUTLASS always passes as `0`:

```c++
asm volatile(
 "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
 ".f32.e2m1.e2m1.f32.ue4m3 "
 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13},"
 "{%14}, {%15,%16}, {%17}, {%18,%19};" ... );
```

**The trap that produced a wrong first answer here:** `%15/%16/%18/%19` are
`byte_id`/`thread_id` **immediates of type u16** (`"h"((uint16_t)0)`), not
registers. Passing a `uint32_t` register there makes `ptxas` report the
misleading `error: Unexpected instruction types specified for 'mma'`, which looks
exactly like "unsupported instruction". If a probe fails, re-check the operand
kinds before concluding the hardware lacks the feature.

## 4. Thread-fragment layouts (`mma_traits_sm120.hpp`)

For `SM120_16x8x64_TN_VS`, `Shape_MNK = 16x8x64`, `ThrID = 32`:

```
ALayout   = Layout<Shape<Shape<_4,_8>, Shape<_8,_2,_2>>, Stride<Stride<_128,_1>, Stride<_16,_8,_512>>>  // (T32,V32) -> (M16,K64)
BLayout   = Layout<Shape<Shape<_4,_8>, Shape<_8,_2>>,    Stride<Stride<_64,_1>,  Stride<_8,_256>>>      // (T32,V16) -> (M16,K64)
SFALayout = Layout<Shape<Shape<_2,_2,_8>,_64>,           Stride<Stride<_8,_0,_1>,_16>>                  // (T32,V64) -> (M16,K64)
SFBLayout = Layout<Shape<Shape<_4,_8>,_64>,              Stride<Stride<_0,_1>, _8>>                     // (T32,V64) -> (N8,K64)
CLayout   = SM80_16x8_Row                                                                                // (T32,V4)  -> (M16,N8)
```

Notes that follow directly from these:
- A is **32 four-bit values per thread** (4 × u32 = 32 nibbles), covering
  `m ∈ {t1, t1+8}` and `k ∈ {8·t0 … }`; B is 16 values (2 × u32).
- The SF layouts are *broadcast*: `SFALayout` has a `_0` stride on one mode
  ("effectively 16 threads") and `SFBLayout` likewise ("effectively 8 threads"),
  so a 32-bit SFA register carries 4 ue4m3 scales that several lanes share.
  Do not "fix" this by giving each lane its own scale — it changes the meaning.
- **`ldmatrix` nibble placement:** for FP4, `ld.matrix.b4x16_p64` puts the nibble
  in the **low** 4 bits of each 8-bit container, but MMA expects it **shifted
  left by 2**. CUTLASS provides `fp4_shift_A`/`fp4_shift_B` for this. If we
  repack weights offline into the fragment order we avoid `ldmatrix` for FP4
  entirely — which is the recommended path (see below).

## 5. Consequences for the engine

1. **Expert prefill GEMM:** native block-scaled NVFP4 `OMMA.SF` (m16n8k64) with
   the checkpoint's own E4M3 per-16 scales, activations quantised to FP4 with
   per-16 scales (W4A4, the deployment mode NVFP4 models are calibrated for;
   this is a *labelled* quality trade, gated by greedy-continuation equality on
   fixed prompts). Expected ≈4× BF16 tensor throughput.
2. **Weight repacking at load:** convert the checkpoint's row-major
   `[N, K/2]` + `[N, K/16]` layout into the MMA fragment order for `A` and `SF`
   offline (once, at load), so the hot loop is pure `ld.global`/`ld.shared` +
   `OMMA.SF` with no shuffles. Verify the repack **numerically** against the f32
   oracle — never by inspecting the layout by eye.
3. **Fallback / numerical reference:** the dequantise-to-BF16 + `HMMA`
   (m16n8k16) path stays in the tree as the exact reference kernel and as the
   fallback for any architecture without `*a` features.
4. **Decode GEMV:** still FFMA-based (bandwidth bound); `OMMA.SF` at M=1 wastes
   the tensor core. A `QMMA`/`OMMA` path becomes interesting only for batched
   decode (M ≥ 16).
5. **Prefill throughput ceiling** (THEORY.md §3) must be restated in terms of FP4
   tensor throughput, not BF16; measure it (I-02 extension) rather than guessing.
