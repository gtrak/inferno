// Prefill GEMM family for sm_120a (GB206).
//
// Four kernels, one contract: `C[m][n] = sum_k A[m][k] * B[n][k]`, accumulated
// in f32. `C` is f32 because a bf16 result rounds to 3.9e-3 relative, and the
// exactness gate for the dequantising path is 5e-4.
//
//   1. gemm_nvfp4_mma       native block-scaled FP4 tensor core (OMMA.SF),
//                           m16n8k64 with per-16 ue4m3 scales: W4A4, the
//                           deployment mode, gated at 2e-2.
//   2. gemm_nvfp4_bf16      the numerical reference: `dequantize_nvfp4_weights`
//                           turns fp4 x ue4m3 into bf16 exactly (the product
//                           needs six mantissa bits), then `gemm_bf16` runs.
//                           The dequantisation is a whole-tensor pass because a
//                           per-tile one repeats it M/64 times.
//   3. gemm_fp8_e4m3        e4m3 tensor core (m16n8k32), per-tensor B scale.
//   4. gemm_bf16            bf16 tensor core (m16n8k16), the dense path.
//
// Thread-fragment layouts: every lane loads its own fragment words with plain
// `ld` from buffers already arranged in fragment order, so the hot loops hold
// no `ldmatrix`, no shuffles and no byte surgery. The weight side is repacked
// once at load time by `core/src/weights/pack_nvfp4.zig`; the activation side
// is repacked by `quantize_activations_nvfp4` below. The layouts were verified
// against the hardware rather than read off the reference; the tests in
// `core/src/kernels/gemm.zig` are the standing proof.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

// ---------------------------------------------------------------------------
// MMA shapes. Properties of the ISA, not tuning knobs.
// ---------------------------------------------------------------------------

// Block-scaled FP4 MMA: a 16x8 output tile, a 64-deep contraction, and four
// per-16 scale bytes on each operand.
constexpr int omma_m = 16;
constexpr int omma_n = 8;
constexpr int omma_k = 64;
constexpr int omma_a_words = 4; // 32 nibbles per lane
constexpr int omma_b_words = 2; // 16 nibbles per lane
constexpr int nvfp4_block_k = 16;
constexpr int nvfp4_scale_bytes = omma_k / nvfp4_block_k;

// bf16 and e4m3 MMA: a 16x8 output tile, 16 or 32 deep.
constexpr int hmma_m = 16;
constexpr int hmma_n = 8;
constexpr int hmma_k = 16;
constexpr int qmma_m = 16;
constexpr int qmma_n = 8;
constexpr int qmma_k = 32;

constexpr int warp_size = 32;

// ---------------------------------------------------------------------------
// Block tiling: 64x64 output tiles, four warps, one warp per 16-row strip.
// ---------------------------------------------------------------------------

constexpr int tile_m = 64;
constexpr int tile_n = 64;
constexpr int warps_per_block = tile_m / hmma_m;
constexpr int m_tiles_per_block = warps_per_block;
constexpr int threads_per_block = warps_per_block * warp_size;
constexpr int omma_n_tiles = tile_n / omma_n;
constexpr int hmma_n_tiles = tile_n / hmma_n;
constexpr int qmma_n_tiles = tile_n / qmma_n;

// cp.async pipeline depth: the copy for tile i+2 runs while tile i is consumed.
constexpr int stages = 3;

// Shared-memory row strides in elements, chosen so a 16 B cp.async stays
// aligned and the per-lane fragment reads land on distinct banks. bf16 rows are
// padded from 32 to 40 elements (80 B), e4m3 rows from 32 to 48 bytes.
constexpr int bf16_tile_k = 32;
constexpr int bf16_row_elements = bf16_tile_k + 8;
constexpr int qmma_tile_k = 32;
constexpr int qmma_row_elements = qmma_tile_k + 16;

// 16 B chunks one stage copy needs, and how many each thread moves.
constexpr int bf16_row_chunks = bf16_tile_k * 2 / 16;
constexpr int bf16_slab_chunks = tile_m * bf16_row_chunks;
constexpr int qmma_row_chunks = qmma_tile_k / 16;
constexpr int qmma_slab_chunks = tile_m * qmma_row_chunks;

// Fragment-order words the block consumes per 64-deep k step of the FP4 path:
// eight 8-column tiles, 32 lanes, two words each.
constexpr int nvfp4_b_stage_words = omma_n_tiles * warp_size * omma_b_words;
constexpr int nvfp4_b_chunks = nvfp4_b_stage_words * 4 / 16;
constexpr int nvfp4_b_scale_stage_words = omma_n_tiles * omma_n;

// Fragment-order sizes of the two whole-tensor FP4 buffers, in words.
constexpr int nvfp4_a_words_per_tile = warp_size * omma_a_words;
constexpr int nvfp4_b_words_per_tile = warp_size * omma_b_words;
constexpr int nvfp4_b_scale_words_per_tile = omma_n;

static_assert(threads_per_block == 128, "stage copies assume 128 threads");
static_assert(nvfp4_b_chunks == threads_per_block,
              "one 16 B B-fragment chunk per thread per stage");
static_assert(nvfp4_b_scale_stage_words <= threads_per_block,
              "B scale words must fit one per thread");
static_assert(bf16_slab_chunks == 2 * threads_per_block,
              "two 16 B chunks per thread per bf16 slab");
static_assert(qmma_slab_chunks == threads_per_block,
              "one 16 B chunk per thread per e4m3 slab");
static_assert(bf16_row_elements % 8 == 0, "16 B alignment of a bf16 row");
static_assert(qmma_row_elements % 16 == 0, "16 B alignment of an e4m3 row");
static_assert(nvfp4_scale_bytes == 4, "one ue4m3 x4 scale word per lane");
static_assert(tile_n % omma_n == 0 && tile_n % hmma_n == 0);
static_assert(tile_m % omma_m == 0 && tile_m % hmma_m == 0);
static_assert(bf16_tile_k % hmma_k == 0 && qmma_tile_k == qmma_k);
static_assert(stages >= 3, "a two-stage pipeline would not overlap copy and MMA");

// ---------------------------------------------------------------------------
// cp.async helpers.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
    const uint32_t address =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address),
                 "l"(gmem));
}

// The same copy, zero-filled when `live` is false: rows past the caller's `M`
// must read as zeros rather than as whatever the previous stage left behind.
__device__ __forceinline__ void cp_async16_pred(void *smem, const void *gmem,
                                               bool live) {
    const uint32_t address =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int bytes = live ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(address),
                 "l"(gmem), "r"(bytes));
}

__device__ __forceinline__ void cp_async4(void *smem, const void *gmem) {
    const uint32_t address =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(address),
                 "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

// Waits until at most `pending` copy groups are still in flight.
template <int pending>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(pending));
}

// ---------------------------------------------------------------------------
// FP4 and FP8 codecs. Decoding is a table so the GPU and `quant.zig` agree on
// every bit; encoding rounds to nearest on the same grid.
// ---------------------------------------------------------------------------

// e2m1: code & 7 selects the magnitude, bit 3 is the sign.
__device__ __forceinline__ float e2m1_decode(uint32_t code) {
    constexpr float magnitude[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float value = magnitude[code & 7u];
    return (code & 8u) ? -value : value;
}

// Magnitudes are compared against the midpoints between grid points, which is
// round-to-nearest in e2m1.
__device__ __forceinline__ uint32_t e2m1_encode(float value) {
    const float magnitude = fabsf(value);
    uint32_t code;
    if (magnitude >= 5.f) {
        code = 7u;
    } else if (magnitude >= 3.5f) {
        code = 6u;
    } else if (magnitude >= 2.5f) {
        code = 5u;
    } else if (magnitude >= 1.75f) {
        code = 4u;
    } else if (magnitude >= 1.25f) {
        code = 3u;
    } else if (magnitude >= 0.75f) {
        code = 2u;
    } else if (magnitude >= 0.25f) {
        code = 1u;
    } else {
        code = 0u;
    }
    return value < 0.f ? (code | 8u) : code;
}

// e4m3 (finite, bias 7).
__device__ __forceinline__ float e4m3_decode(uint32_t bits) {
    const uint32_t sign = (bits >> 7) & 1u;
    const uint32_t exponent = (bits >> 3) & 0xfu;
    const uint32_t mantissa = bits & 7u;
    const float magnitude =
        exponent == 0u
            ? ldexpf(static_cast<float>(mantissa), -9)
            : ldexpf(1.f + static_cast<float>(mantissa) / 8.f,
                     static_cast<int>(exponent) - 7);
    return sign ? -magnitude : magnitude;
}

__device__ __forceinline__ uint32_t e4m3_encode(float value) {
    return static_cast<uint32_t>(
        __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float value) {
    return __float2bfloat16(value);
}

// Four e4m3 values as one 32-bit operand word (k, k+1, k+2, k+3).
__device__ __forceinline__ uint32_t pack_e4m3x4(float x0, float x1, float x2,
                                                float x3) {
    return e4m3_encode(x0) | (e4m3_encode(x1) << 8) | (e4m3_encode(x2) << 16) |
           (e4m3_encode(x3) << 24);
}

// Two adjacent bf16 (k, k+1) as one operand register.
__device__ __forceinline__ uint32_t bf16_pair(const __nv_bfloat16 *row, int k) {
    const uint32_t low = __bfloat16_as_ushort(row[k]);
    const uint32_t high = __bfloat16_as_ushort(row[k + 1]);
    return low | (high << 16);
}

// Four adjacent bf16 converted to one e4m3 operand register (k, k+1, k+2, k+3).
// Saturation at the e4m3 range is the whole of the e4m3 activation contract.
__device__ __forceinline__ uint32_t convert_bf16x4(const __nv_bfloat16 *row) {
    const __nv_bfloat162 low = *reinterpret_cast<const __nv_bfloat162 *>(row);
    const __nv_bfloat162 high = *reinterpret_cast<const __nv_bfloat162 *>(row + 2);
    return pack_e4m3x4(__low2float(low), __high2float(low), __low2float(high),
                       __high2float(high));
}

// ---------------------------------------------------------------------------
// The three MMA instructions.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void omma_nvfp4(float d[4], const uint32_t a[4],
                                           uint32_t b0, uint32_t b1,
                                           uint32_t scale_a, uint32_t scale_b) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(d[0]),
          "f"(d[1]), "f"(d[2]), "f"(d[3]), "r"(scale_a),
          "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)),
          "r"(scale_b), "h"(static_cast<uint16_t>(0)),
          "h"(static_cast<uint16_t>(0)));
}

__device__ __forceinline__ void hmma_bf16(float d[4], const uint32_t a[4],
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(d[0]),
          "f"(d[1]), "f"(d[2]), "f"(d[3]));
}

__device__ __forceinline__ void qmma_e4m3(float d[4], const uint32_t a[4],
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(d[0]),
          "f"(d[1]), "f"(d[2]), "f"(d[3]));
}

// ---------------------------------------------------------------------------
// FP4 layout contract, shared with `pack_nvfp4.zig`.
//
// Quantised activations, word index
//     ((m_tile * k_tiles + k_tile) * warp_size + lane) * 4 + reg
// holds for lane `l`, register `r` the eight nibbles (low nibble first)
//     m = m_tile * 16 + l / 4 + 8 * (r & 1),  k = k_tile * 64 + 8 * (l % 4)
//                                                + 32 * (r >> 1) ... + 8
// and the scale word at ((m_tile * k_tiles + k_tile) * warp_size + lane) holds
// the four ue4m3 bytes of m = m_tile * 16 + 8 * (l & 1) + l / 4, k blocks 0..3.
//
// Weights, word index
//     ((k_tile * n_tiles + n_tile) * warp_size + lane) * 2 + reg
// holds for lane `l`, register `r`
//     n = n_tile * 8 + l / 4,  k = k_tile * 64 + 8 * (l % 4) + 32 * r ... + 8
// and the scale word at ((k_tile * n_tiles + n_tile) * 8 + n_local) holds the
// four ue4m3 bytes of n = n_tile * 8 + n_local.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Activation quantization: bf16 [M,K] -> FP4 fragment order + ue4m3 scales.
// ---------------------------------------------------------------------------

// One warp per (16-row, 64-column) tile. Each thread owns two whole 16-wide
// quantization blocks, so a block maximum needs no cross-lane reduction and the
// four scales of a row land in the lane the MMA reads them from.
extern "C" __global__ void __launch_bounds__(warp_size)
quantize_activations_nvfp4(const __nv_bfloat16 *__restrict__ a, int m_live,
                           int k_cols, uint32_t *__restrict__ a_frag,
                           uint32_t *__restrict__ a_scale_frag) {
    const int lane = threadIdx.x;
    const int k_tile = blockIdx.x;
    const int m_tile = blockIdx.y;
    const int k_tiles = k_cols / omma_k;
    const int k_base = k_tile * omma_k;
    const int m_base = m_tile * omma_m;
    const int scale_base = (m_tile * k_tiles + k_tile) * warp_size;
    const int frag_base = scale_base * omma_a_words;

    // 64 quantization blocks per tile, two per thread: (row, k block).
    for (int step = 0; step < 2; ++step) {
        const int block = lane + step * warp_size;
        const int row = block / nvfp4_scale_bytes;
        const int k_block = block % nvfp4_scale_bytes;
        const int m = m_base + row;
        const int k0 = k_base + k_block * nvfp4_block_k;
        const bool live = m < m_live;

        float values[nvfp4_block_k];
        float largest = 0.f;
        for (int j = 0; j < nvfp4_block_k; ++j) {
            values[j] = live
                            ? bf16_to_f32(
                                  a[static_cast<int64_t>(m) * k_cols + k0 + j])
                            : 0.f;
            largest = fmaxf(largest, fabsf(values[j]));
        }

        // The NVFP4 recipe: a block's scale is amax/6 encoded as ue4m3, and the
        // nibbles are formed against the *stored* scale so that a reader
        // dequantising with the stored byte gets back what was quantised.
        const float requested = largest > 0.f ? largest / 6.f : 1.f;
        const uint32_t scale_byte = e4m3_encode(requested);
        const float stored = e4m3_decode(scale_byte);

        uint32_t packed[2] = {0u, 0u};
        for (int j = 0; j < nvfp4_block_k; ++j) {
            const uint32_t code = e2m1_encode(values[j] / stored);
            packed[j / 8] |= code << (4 * (j % 8));
        }

        // Bytes 0..3 carry k0..k0+7 (lane `low`), bytes 4..7 carry k0+8..k0+15
        // (lane `low + 1`). Both live in the register selected by the row half
        // and the k-half of the block.
        const int reg = (k_block / 2) * 2 + (row / 8);
        const int lane_low = 4 * (row % 8) + 2 * (k_block % 2);
        a_frag[frag_base + lane_low * omma_a_words + reg] = packed[0];
        a_frag[frag_base + (lane_low + 1) * omma_a_words + reg] = packed[1];

        // SFALayout is m = 8 * (lane & 1) + lane / 4, so the lane that carries
        // row m's four scale bytes is 4 * (m % 8) + m / 8; the odd lane of the
        // pair is the broadcast twin and carries the same word.
        const int lane_scale = 4 * (row % 8) + (row / 8);
        uint8_t *scale_out =
            reinterpret_cast<uint8_t *>(&a_scale_frag[scale_base + lane_scale]);
        scale_out[k_block] = static_cast<uint8_t>(scale_byte);
    }
}

// ---------------------------------------------------------------------------
// Shared slab copies for the bf16 and e4m3 paths.
// ---------------------------------------------------------------------------

// Copies one 64x32 bf16 slab of `source` (row-major [rows][k_cols]) into the
// padded shared tile. `row_base + row >= rows_live` copies zeros.
__device__ __forceinline__ void copy_bf16_slab(__nv_bfloat16 *tile,
                                              const __nv_bfloat16 *source,
                                              int row_base, int rows_live,
                                              int column_base, int k_cols) {
    for (int index = 0; index < 2; ++index) {
        const int chunk = threadIdx.x + index * threads_per_block;
        const int row = chunk / bf16_row_chunks;
        const int element = (chunk % bf16_row_chunks) * 8;
        const int64_t offset =
            static_cast<int64_t>(row_base + row) * k_cols + column_base + element;
        cp_async16_pred(tile + row * bf16_row_elements + element,
                        source + offset, row < rows_live);
    }
}

// Copies one 64x32 e4m3 slab; `n_cols` is a multiple of the tile so the whole
// slab is always in bounds.
__device__ __forceinline__ void copy_e4m3_slab(uint8_t *tile,
                                               const uint8_t *source,
                                               int row_base, int column_base,
                                               int k_cols) {
    const int chunk = threadIdx.x;
    const int row = chunk / qmma_row_chunks;
    const int element = (chunk % qmma_row_chunks) * 16;
    const int64_t offset =
        static_cast<int64_t>(row_base + row) * k_cols + column_base + element;
    cp_async16(tile + row * qmma_row_elements + element, source + offset);
}

// ---------------------------------------------------------------------------
// 1. Native block-scaled FP4: the fast path.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void nvfp4_prefetch_b(uint32_t *dst,
                                                uint32_t *dst_scale,
                                                const uint32_t *src_frag,
                                                const uint32_t *src_scale,
                                                int k_tile, int n_tile0,
                                                int n_tiles) {
    const int64_t frag_base =
        (static_cast<int64_t>(k_tile) * n_tiles + n_tile0) *
        nvfp4_b_words_per_tile;
    cp_async16(&dst[threadIdx.x * 4], &src_frag[frag_base + threadIdx.x * 4]);
    if (threadIdx.x < nvfp4_b_scale_stage_words) {
        const int64_t scale_base =
            (static_cast<int64_t>(k_tile) * n_tiles + n_tile0) *
            nvfp4_b_scale_words_per_tile;
        cp_async4(&dst_scale[threadIdx.x], &src_scale[scale_base + threadIdx.x]);
    }
}

extern "C" __global__ void __launch_bounds__(threads_per_block)
gemm_nvfp4_mma(const uint32_t *__restrict__ a_frag,
               const uint32_t *__restrict__ a_scale_frag,
               const uint32_t *__restrict__ b_frag,
               const uint32_t *__restrict__ b_scale_frag, float scale2,
               float *__restrict__ c, int m_rows, int n_cols, int k_cols) {
    __shared__ uint32_t b_stage[stages][nvfp4_b_stage_words];
    __shared__ uint32_t b_scale_stage[stages][nvfp4_b_scale_stage_words];

    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const int k_tiles = k_cols / omma_k;
    const int n_tiles = n_cols / omma_n;
    const int n_tile0 = blockIdx.x * omma_n_tiles;
    const int m_tile = blockIdx.y * m_tiles_per_block + warp;

    // One warp owns one 16-row tile; its fragment words advance by one tile
    // (32 lanes x 4 words) per k step.
    const int64_t a_base =
        static_cast<int64_t>(m_tile) * k_tiles * nvfp4_a_words_per_tile +
        lane * omma_a_words;

    for (int stage = 0; stage < stages - 1; ++stage) {
        if (stage < k_tiles) {
            nvfp4_prefetch_b(b_stage[stage], b_scale_stage[stage], b_frag,
                             b_scale_frag, stage, n_tile0, n_tiles);
        }
        cp_async_commit();
    }

    float acc[omma_n_tiles][4];
    for (int n_tile = 0; n_tile < omma_n_tiles; ++n_tile) {
        for (int value = 0; value < 4; ++value) {
            acc[n_tile][value] = 0.f;
        }
    }

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        cp_async_wait<stages - 2>();
        __syncthreads();
        const int prefetch = k_tile + stages - 1;
        if (prefetch < k_tiles) {
            nvfp4_prefetch_b(b_stage[prefetch % stages],
                             b_scale_stage[prefetch % stages], b_frag,
                             b_scale_frag, prefetch, n_tile0, n_tiles);
        }
        // Unconditional, so the wait above always refers to the same stage even
        // in the tail where nothing is prefetched.
        cp_async_commit();

        const uint32_t *b_words = &b_stage[k_tile % stages][0];
        const uint32_t *b_scales = &b_scale_stage[k_tile % stages][0];
        const int64_t a_tile_base =
            a_base + static_cast<int64_t>(k_tile) * nvfp4_a_words_per_tile;
        uint32_t a[omma_a_words];
        for (int word = 0; word < omma_a_words; ++word) {
            a[word] = a_frag[a_tile_base + word];
        }
        // The scale word walks one per lane per k step.
        const uint32_t scale_a = a_scale_frag[a_tile_base / omma_a_words];

        for (int n_tile = 0; n_tile < omma_n_tiles; ++n_tile) {
            const uint32_t b0 =
                b_words[(n_tile * warp_size + lane) * omma_b_words];
            const uint32_t b1 =
                b_words[(n_tile * warp_size + lane) * omma_b_words + 1];
            const uint32_t scale_b = b_scales[n_tile * omma_n + lane / 4];
            omma_nvfp4(acc[n_tile], a, b0, b1, scale_a, scale_b);
        }
    }

    // Epilogue: fold the per-tensor weight scale in and write f32. Rows past
    // `m_rows` come from a degenerate tile and are dropped.
    const int row_low = blockIdx.y * tile_m + warp * omma_m + lane / 4;
    const int column = blockIdx.x * tile_n + 2 * (lane % 4);
    for (int n_tile = 0; n_tile < omma_n_tiles; ++n_tile) {
        const int n = column + n_tile * omma_n;
        if (row_low < m_rows) {
            float *destination =
                c + static_cast<int64_t>(row_low) * n_cols + n;
            destination[0] = acc[n_tile][0] * scale2;
            destination[1] = acc[n_tile][1] * scale2;
        }
        if (row_low + 8 < m_rows) {
            float *destination =
                c + static_cast<int64_t>(row_low + 8) * n_cols + n;
            destination[0] = acc[n_tile][2] * scale2;
            destination[1] = acc[n_tile][3] * scale2;
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Exact reference path: dequantise FP4 x ue4m3 to bf16, then HMMA.
// ---------------------------------------------------------------------------

// Dequantises the whole packed weight matrix to bf16 once, before the GEMM.
//
// The product of an e2m1 value with a ue4m3 scale needs six mantissa bits, so
// bf16 holds it exactly and the conversion is lossless. Doing it here rather
// than inside each 64x32 shared-memory tile matters: a tiled kernel dequantises
// the same weight `M/64` times (64x redundant work at M=4096), which measured at
// 4.3 TFLOPS against 33 TFLOPS for the bf16 MMA it feeds.
extern "C" __global__ void __launch_bounds__(threads_per_block)
dequantize_nvfp4_weights(const uint8_t *__restrict__ packed_weights,
                        const uint8_t *__restrict__ scales,
                        __nv_bfloat16 *__restrict__ out, int n_cols, int k_cols) {
    const int blocks_per_row = k_cols / nvfp4_block_k;
    const int column_block = blockIdx.x * threads_per_block + threadIdx.x;
    const int row = blockIdx.y;
    if (column_block >= blocks_per_row || row >= n_cols) {
        return;
    }
    const int64_t block_index = static_cast<int64_t>(row) * blocks_per_row + column_block;
    const uint32_t *source =
        reinterpret_cast<const uint32_t *>(packed_weights + block_index * 8);
    const float block_scale = e4m3_decode(scales[block_index]);
    const uint32_t low = source[0];
    const uint32_t high = source[1];

    uint32_t words[8];
    for (int word = 0; word < 2; ++word) {
        const uint32_t nibbles = word == 0 ? low : high;
        for (int pair = 0; pair < 4; ++pair) {
            const uint32_t byte = (nibbles >> (8 * pair)) & 0xffu;
            const uint32_t first = byte & 0xfu;
            const uint32_t second = byte >> 4;
            const uint32_t low_half = __bfloat16_as_ushort(
                f32_to_bf16(e2m1_decode(first) * block_scale));
            const uint32_t high_half = __bfloat16_as_ushort(
                f32_to_bf16(e2m1_decode(second) * block_scale));
            words[word * 4 + pair] = low_half | (high_half << 16);
        }
    }
    __nv_bfloat16 *destination =
        out + static_cast<int64_t>(row) * k_cols + column_block * nvfp4_block_k;
    *reinterpret_cast<uint4 *>(destination) =
        make_uint4(words[0], words[1], words[2], words[3]);
    *reinterpret_cast<uint4 *>(destination + 8) =
        make_uint4(words[4], words[5], words[6], words[7]);
}

// ---------------------------------------------------------------------------
// 3. e4m3 tensor core. The activations are converted to e4m3 in registers, so
// no scratch tensor is needed; the weights scale is folded in the epilogue.
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(threads_per_block)
gemm_fp8_e4m3(const __nv_bfloat16 *__restrict__ a, const uint8_t *__restrict__ b,
              const float *__restrict__ b_scale, float *__restrict__ c, int m_rows,
              int n_cols, int k_cols) {
    __shared__ __nv_bfloat16 a_stage[stages][tile_m * bf16_row_elements];
    __shared__ uint8_t b_stage[stages][tile_m * qmma_row_elements];

    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const int group = lane / 4;
    const int quad = lane % 4;
    const int k_steps = k_cols / qmma_tile_k;
    const int m0 = blockIdx.y * tile_m;
    const int n0 = blockIdx.x * tile_n;
    const int rows_live = m_rows - m0 < tile_m ? m_rows - m0 : tile_m;
    const float scale = b_scale[0];

    for (int stage = 0; stage < stages - 1; ++stage) {
        if (stage < k_steps) {
            copy_bf16_slab(a_stage[stage], a, m0, rows_live,
                           stage * qmma_tile_k, k_cols);
            copy_e4m3_slab(b_stage[stage], b, n0, stage * qmma_tile_k, k_cols);
        }
        cp_async_commit();
    }

    float acc[qmma_n_tiles][4];
    for (int n_tile = 0; n_tile < qmma_n_tiles; ++n_tile) {
        for (int value = 0; value < 4; ++value) {
            acc[n_tile][value] = 0.f;
        }
    }

    const int row_low = warp * qmma_m + group;
    const int row_high = row_low + 8;
    for (int k_step = 0; k_step < k_steps; ++k_step) {
        cp_async_wait<stages - 2>();
        __syncthreads();
        const int prefetch = k_step + stages - 1;
        if (prefetch < k_steps) {
            copy_bf16_slab(a_stage[prefetch % stages], a, m0, rows_live,
                           prefetch * qmma_tile_k, k_cols);
            copy_e4m3_slab(b_stage[prefetch % stages], b, n0,
                           prefetch * qmma_tile_k, k_cols);
        }
        cp_async_commit();

        const __nv_bfloat16 *a_tile = &a_stage[k_step % stages][0];
        const uint8_t *b_tile = &b_stage[k_step % stages][0];
        const __nv_bfloat16 *a_low = a_tile + row_low * bf16_row_elements;
        const __nv_bfloat16 *a_high = a_tile + row_high * bf16_row_elements;
        const int k_low = 4 * quad;
        const int k_high = k_low + 16;
        uint32_t a_words[4];
        a_words[0] = convert_bf16x4(a_low + k_low);
        a_words[1] = convert_bf16x4(a_high + k_low);
        a_words[2] = convert_bf16x4(a_low + k_high);
        a_words[3] = convert_bf16x4(a_high + k_high);
        for (int n_tile = 0; n_tile < qmma_n_tiles; ++n_tile) {
            const uint8_t *b_row =
                b_tile + (n_tile * qmma_n + group) * qmma_row_elements;
            const uint32_t b0 = *reinterpret_cast<const uint32_t *>(b_row + k_low);
            const uint32_t b1 = *reinterpret_cast<const uint32_t *>(b_row + k_high);
            qmma_e4m3(acc[n_tile], a_words, b0, b1);
        }
    }

    const int m_low = m0 + row_low;
    const int m_high = m0 + row_high;
    const int column = n0 + 2 * quad;
    for (int n_tile = 0; n_tile < qmma_n_tiles; ++n_tile) {
        const int n = column + n_tile * qmma_n;
        if (m_low < m_rows) {
            float *destination = c + static_cast<int64_t>(m_low) * n_cols + n;
            destination[0] = acc[n_tile][0] * scale;
            destination[1] = acc[n_tile][1] * scale;
        }
        if (m_high < m_rows) {
            float *destination = c + static_cast<int64_t>(m_high) * n_cols + n;
            destination[0] = acc[n_tile][2] * scale;
            destination[1] = acc[n_tile][3] * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// 4. bf16 tensor core.
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(threads_per_block)
gemm_bf16(const __nv_bfloat16 *__restrict__ a, const __nv_bfloat16 *__restrict__ b,
          float scale, float *__restrict__ c, int m_rows, int n_cols,
          int k_cols) {
    __shared__ __nv_bfloat16 a_stage[stages][tile_m * bf16_row_elements];
    __shared__ __nv_bfloat16 b_stage[stages][tile_m * bf16_row_elements];

    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const int group = lane / 4;
    const int quad = lane % 4;
    const int k_steps = k_cols / bf16_tile_k;
    const int m0 = blockIdx.y * tile_m;
    const int n0 = blockIdx.x * tile_n;
    const int rows_live = m_rows - m0 < tile_m ? m_rows - m0 : tile_m;

    for (int stage = 0; stage < stages - 1; ++stage) {
        if (stage < k_steps) {
            copy_bf16_slab(a_stage[stage], a, m0, rows_live,
                           stage * bf16_tile_k, k_cols);
            copy_bf16_slab(b_stage[stage], b, n0, tile_n, stage * bf16_tile_k,
                           k_cols);
        }
        cp_async_commit();
    }

    float acc[hmma_n_tiles][4];
    for (int n_tile = 0; n_tile < hmma_n_tiles; ++n_tile) {
        for (int value = 0; value < 4; ++value) {
            acc[n_tile][value] = 0.f;
        }
    }

    const int row_low = warp * hmma_m + group;
    const int row_high = row_low + 8;
    for (int k_step = 0; k_step < k_steps; ++k_step) {
        cp_async_wait<stages - 2>();
        __syncthreads();
        const int prefetch = k_step + stages - 1;
        if (prefetch < k_steps) {
            copy_bf16_slab(a_stage[prefetch % stages], a, m0, rows_live,
                           prefetch * bf16_tile_k, k_cols);
            copy_bf16_slab(b_stage[prefetch % stages], b, n0, tile_n,
                           prefetch * bf16_tile_k, k_cols);
        }
        cp_async_commit();

        const __nv_bfloat16 *a_tile = &a_stage[k_step % stages][0];
        const __nv_bfloat16 *b_tile = &b_stage[k_step % stages][0];
        for (int half = 0; half < bf16_tile_k / hmma_k; ++half) {
            const int k_low = half * hmma_k + 2 * quad;
            const uint32_t a_words[4] = {
                bf16_pair(a_tile + row_low * bf16_row_elements, k_low),
                bf16_pair(a_tile + row_high * bf16_row_elements, k_low),
                bf16_pair(a_tile + row_low * bf16_row_elements, k_low + 8),
                bf16_pair(a_tile + row_high * bf16_row_elements, k_low + 8),
            };
            for (int n_tile = 0; n_tile < hmma_n_tiles; ++n_tile) {
                const __nv_bfloat16 *b_row =
                    b_tile + (n_tile * hmma_n + group) * bf16_row_elements;
                hmma_bf16(acc[n_tile], a_words, bf16_pair(b_row, k_low),
                          bf16_pair(b_row, k_low + 8));
            }
        }
    }

    const int m_low = m0 + row_low;
    const int m_high = m0 + row_high;
    const int column = n0 + 2 * quad;
    for (int n_tile = 0; n_tile < hmma_n_tiles; ++n_tile) {
        const int n = column + n_tile * hmma_n;
        if (m_low < m_rows) {
            float *destination = c + static_cast<int64_t>(m_low) * n_cols + n;
            destination[0] = acc[n_tile][0] * scale;
            destination[1] = acc[n_tile][1] * scale;
        }
        if (m_high < m_rows) {
            float *destination = c + static_cast<int64_t>(m_high) * n_cols + n;
            destination[0] = acc[n_tile][2] * scale;
            destination[1] = acc[n_tile][3] * scale;
        }
    }
}

} // namespace
