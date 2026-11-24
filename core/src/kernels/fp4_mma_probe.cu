// Capability probe for the SM120 (consumer Blackwell) tensor-core MMA family.
//
// Every kernel here issues exactly one `mma.sync` instruction and writes its
// accumulator fragment, so `ptxas` must emit the instruction (it cannot be
// optimised away) and `cuobjdump -sass` names it. The Zig harness
// `src/kernels/fp4_mma_probe.zig` dumps that SASS and records the verdict table
// in `bench/fp4-mma-probe.txt`.
//
// Operand structure and the `.kind::mxf4nvf4` / `.kind::f8f6f4` spellings are
// taken from `.agents/reference/cutlass/mma_sm120.hpp` (read-only reference),
// which is the same information as `.agents/design/mma-sm120-nvfp4.md`.
//
// The one trap worth repeating: the last four operands of a block-scaled MMA
// are the `byte_id`/`thread_id` *immediates*, typed `u16` ("h"), always 0.
// Passing them as u32 registers makes ptxas report "Unexpected instruction
// types specified for 'mma'", which reads like "unsupported instruction".

// Block-scaled FP4 MMA is an architecture-specific (`sm_120a`) feature: plain
// `sm_120` rejects `.kind::mxf4nvf4` with "Feature not supported on .target".
// This is the build-time assertion that the whole kernel directory is compiled
// with the mandatory `a` suffix (core/build.zig `cuda_arch`).
#if !defined(__CUDA_ARCH_SPECIFIC__)
#error "inferno kernels must be compiled with -arch=sm_120a"
#endif

#include <cstdint>

// A warp is 32 lanes; the probe needs exactly one converged warp.
static constexpr int probe_lanes = 32;

// Elements each kernel writes: four f32 accumulator registers per lane.
static constexpr int probe_outputs_per_lane = 4;

// Nonzero fragment fill, so a probe that silently skipped the MMA would write
// values that differ from the ones we list in the verdict table.
__device__ __forceinline__ uint32_t probe_pattern(int lane, int register_index) {
    return 0x3f800000u + static_cast<uint32_t>(lane * 8 + register_index);
}

// m16n8k16 bf16 x bf16 -> f32. A: 4 x u32 (bf16x2), B: 2 x u32, C/D: 4 x f32.
extern "C" __global__ void __launch_bounds__(probe_lanes)
fp4_probe_mma_bf16_m16n8k16(float *out) {
    const int lane = static_cast<int>(threadIdx.x);

    const uint32_t a0 = probe_pattern(lane, 0);
    const uint32_t a1 = probe_pattern(lane, 1);
    const uint32_t a2 = probe_pattern(lane, 2);
    const uint32_t a3 = probe_pattern(lane, 3);
    const uint32_t b0 = probe_pattern(lane, 4);
    const uint32_t b1 = probe_pattern(lane, 5);

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));

    const int base = lane * probe_outputs_per_lane;
    out[base + 0] = d0;
    out[base + 1] = d1;
    out[base + 2] = d2;
    out[base + 3] = d3;
}

// m16n8k32 e4m3 x e4m3 -> f32 (the FP8 path of the dense/MTP GEMM).
extern "C" __global__ void __launch_bounds__(probe_lanes)
fp4_probe_mma_fp8_m16n8k32(float *out) {
    const int lane = static_cast<int>(threadIdx.x);

    const uint32_t a0 = probe_pattern(lane, 0);
    const uint32_t a1 = probe_pattern(lane, 1);
    const uint32_t a2 = probe_pattern(lane, 2);
    const uint32_t a3 = probe_pattern(lane, 3);
    const uint32_t b0 = probe_pattern(lane, 4);
    const uint32_t b1 = probe_pattern(lane, 5);

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));

    const int base = lane * probe_outputs_per_lane;
    out[base + 0] = d0;
    out[base + 1] = d1;
    out[base + 2] = d2;
    out[base + 3] = d3;
}

// m16n8k32 e2m1 x e2m1 -> f32: unscaled FP4 MMA (no block scales).
extern "C" __global__ void __launch_bounds__(probe_lanes)
fp4_probe_mma_f8f6f4_m16n8k32(float *out) {
    const int lane = static_cast<int>(threadIdx.x);

    const uint32_t a0 = probe_pattern(lane, 0);
    const uint32_t a1 = probe_pattern(lane, 1);
    const uint32_t a2 = probe_pattern(lane, 2);
    const uint32_t a3 = probe_pattern(lane, 3);
    const uint32_t b0 = probe_pattern(lane, 4);
    const uint32_t b1 = probe_pattern(lane, 5);

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));

    const int base = lane * probe_outputs_per_lane;
    out[base + 0] = d0;
    out[base + 1] = d1;
    out[base + 2] = d2;
    out[base + 3] = d3;
}

// m16n8k64 e2m1 x e2m1 -> f32 with UE4M3 block scales, `scale_vec::4X`
// (one scale per 16 contracted elements): this is *exactly* the ModelOpt NVFP4
// checkpoint format, so it is the instruction the expert GEMM wants.
extern "C" __global__ void __launch_bounds__(probe_lanes)
fp4_probe_mma_mxf4nvf4_4x_ue4m3_m16n8k64(float *out) {
    const int lane = static_cast<int>(threadIdx.x);

    // byte_id / thread_id are u16 immediates, never registers.
    const uint16_t byte_id = 0;
    const uint16_t thread_id = 0;

    const uint32_t a0 = probe_pattern(lane, 0);
    const uint32_t a1 = probe_pattern(lane, 1);
    const uint32_t a2 = probe_pattern(lane, 2);
    const uint32_t a3 = probe_pattern(lane, 3);
    const uint32_t b0 = probe_pattern(lane, 4);
    const uint32_t b1 = probe_pattern(lane, 5);
    const uint32_t scale_a = 0x3c003c00u;
    const uint32_t scale_b = 0x3c003c00u;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
          "r"(scale_a), "h"(byte_id), "h"(thread_id),
          "r"(scale_b), "h"(byte_id), "h"(thread_id));

    const int base = lane * probe_outputs_per_lane;
    out[base + 0] = d0;
    out[base + 1] = d1;
    out[base + 2] = d2;
    out[base + 3] = d3;
}

// m16n8k64 e2m1 x e2m1 -> f32 with UE8M0 block scales, `scale_vec::2X`
// (one power-of-two scale per 32 contracted elements): the MXFP4 format of the
// per-32 expert requantisation lever (plan.md L3).
extern "C" __global__ void __launch_bounds__(probe_lanes)
fp4_probe_mma_mxf4nvf4_2x_ue8m0_m16n8k64(float *out) {
    const int lane = static_cast<int>(threadIdx.x);

    const uint16_t byte_id = 0;
    const uint16_t thread_id = 0;

    const uint32_t a0 = probe_pattern(lane, 0);
    const uint32_t a1 = probe_pattern(lane, 1);
    const uint32_t a2 = probe_pattern(lane, 2);
    const uint32_t a3 = probe_pattern(lane, 3);
    const uint32_t b0 = probe_pattern(lane, 4);
    const uint32_t b1 = probe_pattern(lane, 5);
    const uint32_t scale_a = 0x7f007f00u;
    const uint32_t scale_b = 0x7f007f00u;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
          "r"(scale_a), "h"(byte_id), "h"(thread_id),
          "r"(scale_b), "h"(byte_id), "h"(thread_id));

    const int base = lane * probe_outputs_per_lane;
    out[base + 0] = d0;
    out[base + 1] = d1;
    out[base + 2] = d2;
    out[base + 3] = d3;
}
