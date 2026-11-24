// Decode GEMV family: NVFP4 / FP8-blockwise / BF16 weights, BF16 activations,
// f32 accumulation.
//
// Decode is bandwidth bound (THEORY.md §2): a token reads every weight byte
// exactly once and no arithmetic is reusable, so the only thing that matters is
// bytes read per second. Every kernel below therefore streams its weight matrix
// with 8 lanes per output row (perfectly coalesced 64 B per row group, 256 B per
// warp instruction), and the activation vector is read through L1/L2 - for the
// real K the whole `x` is 5 KB (K=2560), so re-reading it per row is an L1 hit
// and costs no HBM traffic. That is why there is no shared-memory staging of
// `x`: staging would only move the same bytes from L1 to shared memory.
//
// 16-bit `x` is widened with a shift (`bf16 -> f32` is `word << 16`, exactly),
// and 4-bit / 8-bit weights are decoded through per-block lookup tables built
// once into shared memory (256 E4M3 entries, 16 E2M1 entries).
//
// Shapes: `k % 16 == 0` and `n % 8 == 0` are the contract (asserted host-side in
// `gemv.zig`); the aligned path uses `uint4` (16 B) weight loads when the row
// stride is 16 B aligned (`k % 32 == 0`, which is true for every shape of this
// model) and falls back to `uint2` (8 B) loads for `k % 32 == 16`.

#if !defined(__CUDA_ARCH_SPECIFIC__)
#error "inferno kernels must be compiled with -arch=sm_120a"
#endif

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Threads per block: one warp per four output rows, eight warps.
static constexpr int gemv_block_threads = 256;

// Lanes covering one output row; the warp therefore spans four rows.
static constexpr int gemv_lanes_per_row = 8;

static constexpr int gemv_rows_per_warp = 32 / gemv_lanes_per_row;
static constexpr int gemv_warps_per_block = gemv_block_threads / 32;
static constexpr int gemv_rows_per_block = gemv_rows_per_warp * gemv_warps_per_block;

// E4M3 is a byte-wide code, E2M1 a nibble-wide one.
static constexpr int e4m3_table_entries = 256;
static constexpr int e2m1_table_entries = 16;

// A decode batch is a handful of sequences; beyond this the kernel becomes
// shared-memory bound and the prefill GEMM (I-06b) should run instead.
static constexpr int gemv_max_decode_batch = 32;

// Weights sharing one NVFP4 block scale, i.e. the natural 8-byte load unit.
static constexpr int nvfp4_block_elems = 16;
static constexpr int nvfp4_scale_elems = nvfp4_block_elems;

// One BF16 inverse scale per 128x128 tile in the FP8 blockwise layout.
static constexpr int fp8_scale_tile = 128;

// Decode tables shared by every warp of a block.
struct DecodeTables {
    float e4m3[e4m3_table_entries];
    float e2m1[e2m1_table_entries];
};

// The E2M1 value grid: code 0..7 is the magnitude, bit 3 is the sign. This is
// the same table as `weights/quant.zig` `fp4_values`.
__device__ constexpr float e2m1_magnitudes[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__device__ __forceinline__ float e2m1_value(int code) {
    const float magnitude = e2m1_magnitudes[code & 7];
    return ((code & 8) != 0) ? -magnitude : magnitude;
}

// Builds the per-block decode tables. Every E4M3 value is exact in f32, and the
// hardware conversion used here is the FN (finite, no infinity) format the
// checkpoint uses, so the tables match `weights/quant.zig` bit for bit.
__device__ __forceinline__ void fill_decode_tables(DecodeTables *tables) {
    const int thread = static_cast<int>(threadIdx.x);
    for (int code = thread; code < e4m3_table_entries; code += gemv_block_threads) {
        const __half decoded = __nv_cvt_fp8_to_halfraw(
            static_cast<__nv_fp8_storage_t>(code), __NV_E4M3);
        tables->e4m3[code] = __half2float(decoded);
    }
    if (thread < e2m1_table_entries) {
        tables->e2m1[thread] = e2m1_value(thread);
    }
}

// BF16 to f32 is a left shift of the bit pattern: exact, one instruction.
__device__ __forceinline__ float bf16_low(uint32_t packed) {
    return __uint_as_float(packed << 16);
}

__device__ __forceinline__ float bf16_high(uint32_t packed) {
    return __uint_as_float(packed & 0xffff0000u);
}

// Eight packed E2M1 weights (one u32, low nibble first) against eight BF16
// activations (four u32).
__device__ __forceinline__ float dot_e2m1_8(
    uint32_t packed, uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
    const float *e2m1) {
    float sum = 0.0f;
    sum = fmaf(e2m1[packed & 0x0fu], bf16_low(x0), sum);
    sum = fmaf(e2m1[(packed >> 4) & 0x0fu], bf16_high(x0), sum);
    sum = fmaf(e2m1[(packed >> 8) & 0x0fu], bf16_low(x1), sum);
    sum = fmaf(e2m1[(packed >> 12) & 0x0fu], bf16_high(x1), sum);
    sum = fmaf(e2m1[(packed >> 16) & 0x0fu], bf16_low(x2), sum);
    sum = fmaf(e2m1[(packed >> 20) & 0x0fu], bf16_high(x2), sum);
    sum = fmaf(e2m1[(packed >> 24) & 0x0fu], bf16_low(x3), sum);
    sum = fmaf(e2m1[packed >> 28], bf16_high(x3), sum);
    return sum;
}

// Four E4M3 weights (one u32) against four BF16 activations.
__device__ __forceinline__ float dot_e4m3_4(
    uint32_t packed, uint32_t x0, uint32_t x1, const float *e4m3) {
    float sum = 0.0f;
    sum = fmaf(e4m3[packed & 0xffu], bf16_low(x0), sum);
    sum = fmaf(e4m3[(packed >> 8) & 0xffu], bf16_high(x0), sum);
    sum = fmaf(e4m3[(packed >> 16) & 0xffu], bf16_low(x1), sum);
    sum = fmaf(e4m3[packed >> 24], bf16_high(x1), sum);
    return sum;
}

// Eight BF16 weights (one uint4) against eight BF16 activations (one uint4).
__device__ __forceinline__ float dot_bf16_8(uint4 w, uint4 x) {
    float sum = 0.0f;
    sum = fmaf(bf16_low(w.x), bf16_low(x.x), sum);
    sum = fmaf(bf16_high(w.x), bf16_high(x.x), sum);
    sum = fmaf(bf16_low(w.y), bf16_low(x.y), sum);
    sum = fmaf(bf16_high(w.y), bf16_high(x.y), sum);
    sum = fmaf(bf16_low(w.z), bf16_low(x.z), sum);
    sum = fmaf(bf16_high(w.z), bf16_high(x.z), sum);
    sum = fmaf(bf16_low(w.w), bf16_low(x.w), sum);
    sum = fmaf(bf16_high(w.w), bf16_high(x.w), sum);
    return sum;
}

// Sums one row group's partial products and hands the total to lane 0.
// The shuffle mask covers exactly the eight lanes of this row, so a block whose
// last row group is out of range stays inside the CUDA shuffle contract.
__device__ __forceinline__ float reduce_row_group(float partial, uint32_t group_mask) {
    partial += __shfl_xor_sync(group_mask, partial, 1);
    partial += __shfl_xor_sync(group_mask, partial, 2);
    partial += __shfl_xor_sync(group_mask, partial, 4);
    return partial;
}

// Positions inside the block: each lane owns (row, chunk) where `chunk_lane`
// selects which of the eight strided weight chunks it streams.
struct RowSlot {
    int row;
    int chunk_lane;
    uint32_t group_mask;
};

__device__ __forceinline__ RowSlot row_slot() {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row_in_warp = lane / gemv_lanes_per_row;
    RowSlot slot;
    slot.row = static_cast<int>(blockIdx.x) * gemv_rows_per_block +
               warp * gemv_rows_per_warp + row_in_warp;
    slot.chunk_lane = lane % gemv_lanes_per_row;
    slot.group_mask = static_cast<uint32_t>(0xffu << (row_in_warp * gemv_lanes_per_row));
    return slot;
}

// NVFP4 weight row against `kBatch` activation rows.
//
// `x` is [m, k] and `y` is [m, n]; a sequence index of `m - 1` is used for the
// unused slots of a partial batch, so no load ever leaves the batch while the
// accumulators stay branch free.
template <int kBatch>
__device__ __forceinline__ void gemv_nvfp4_rows(
    const uint8_t *__restrict__ w_packed,
    const uint8_t *__restrict__ w_scales,
    float w_scale_2,
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ y,
    int n, int k, int m,
    const DecodeTables &tables) {
    const RowSlot slot = row_slot();
    if (slot.row >= n) {
        return;
    }

    const uint8_t *w_row = w_packed + static_cast<size_t>(slot.row) *
                                        static_cast<size_t>(k / 2);
    const uint8_t *s_row = w_scales + static_cast<size_t>(slot.row) *
                                        static_cast<size_t>(k / nvfp4_scale_elems);
    float acc[kBatch];
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
        acc[b] = 0.0f;
    }

    if ((k & 31) == 0) {
        // Aligned path: 16-byte weight loads, two scale blocks per load.
        const int units = k / 32;
        for (int unit = slot.chunk_lane; unit < units; unit += gemv_lanes_per_row) {
            const uint4 packed = *reinterpret_cast<const uint4 *>(w_row + unit * 16);
            const uint16_t scale_pair =
                *reinterpret_cast<const uint16_t *>(s_row + unit * 2);
            const float scale_lo = tables.e4m3[scale_pair & 0xffu] * w_scale_2;
            const float scale_hi = tables.e4m3[scale_pair >> 8] * w_scale_2;
            const int column = unit * 32;
#pragma unroll
            for (int b = 0; b < kBatch; ++b) {
                const int seq = (b < m) ? b : (m - 1);
                // One uint4 is eight BF16 activations: four of them cover the
                // 32 elements of this unit.
                const __nv_bfloat16 *x_unit =
                    x + static_cast<size_t>(seq) * static_cast<size_t>(k) + column;
                const uint4 xa = *reinterpret_cast<const uint4 *>(x_unit);
                const uint4 xb = *reinterpret_cast<const uint4 *>(x_unit + 8);
                const uint4 xc = *reinterpret_cast<const uint4 *>(x_unit + 16);
                const uint4 xd = *reinterpret_cast<const uint4 *>(x_unit + 24);
                const float lo = dot_e2m1_8(packed.x, xa.x, xa.y, xa.z, xa.w,
                                            tables.e2m1) +
                                 dot_e2m1_8(packed.y, xb.x, xb.y, xb.z, xb.w,
                                            tables.e2m1);
                const float hi = dot_e2m1_8(packed.z, xc.x, xc.y, xc.z, xc.w,
                                            tables.e2m1) +
                                 dot_e2m1_8(packed.w, xd.x, xd.y, xd.z, xd.w,
                                            tables.e2m1);
                acc[b] = fmaf(lo, scale_lo, acc[b]);
                acc[b] = fmaf(hi, scale_hi, acc[b]);
            }
        }
    } else {
        // `k % 32 == 16`: the row stride is only 8-byte aligned, so the 16
        // elements of one scale block are loaded as a uint2.
        const int units = k / nvfp4_block_elems;
        for (int unit = slot.chunk_lane; unit < units; unit += gemv_lanes_per_row) {
            const uint2 packed = *reinterpret_cast<const uint2 *>(w_row + unit * 8);
            const float scale = tables.e4m3[s_row[unit]] * w_scale_2;
            const int column = unit * nvfp4_block_elems;
#pragma unroll
            for (int b = 0; b < kBatch; ++b) {
                const int seq = (b < m) ? b : (m - 1);
                const uint4 xa = *reinterpret_cast<const uint4 *>(
                    x + static_cast<size_t>(seq) * static_cast<size_t>(k) + column);
                const uint4 xb = *reinterpret_cast<const uint4 *>(
                    x + static_cast<size_t>(seq) * static_cast<size_t>(k) + column + 8);
                const float partial = dot_e2m1_8(packed.x, xa.x, xa.y, xa.z, xa.w,
                                                 tables.e2m1) +
                                      dot_e2m1_8(packed.y, xb.x, xb.y, xb.z, xb.w,
                                                 tables.e2m1);
                acc[b] = fmaf(partial, scale, acc[b]);
            }
        }
    }

#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
        const float total = reduce_row_group(acc[b], slot.group_mask);
        if (slot.chunk_lane == 0 && b < m) {
            y[static_cast<size_t>(b) * static_cast<size_t>(n) + slot.row] =
                __float2bfloat16(total);
        }
    }
}

// Decode GEMV: `y[n] = sum_k x[k] * dequant(w[n, k])`, one row per eight lanes.
extern "C" __global__ void __launch_bounds__(gemv_block_threads)
gemv_nvfp4_bf16(const uint8_t *__restrict__ w_packed,
                const uint8_t *__restrict__ w_scales,
                float w_scale_2,
                const __nv_bfloat16 *__restrict__ x,
                __nv_bfloat16 *__restrict__ y,
                int n, int k) {
    __shared__ DecodeTables tables;
    fill_decode_tables(&tables);
    __syncthreads();
    gemv_nvfp4_rows<1>(w_packed, w_scales, w_scale_2, x, y, n, k, 1, tables);
}

// Batched decode GEMV: the weights are read once for up to 32 sequences.
extern "C" __global__ void __launch_bounds__(gemv_block_threads)
gemv_nvfp4_bf16_batch(const uint8_t *__restrict__ w_packed,
                      const uint8_t *__restrict__ w_scales,
                      float w_scale_2,
                      const __nv_bfloat16 *__restrict__ x,
                      __nv_bfloat16 *__restrict__ y,
                      int n, int k, int m) {
    __shared__ DecodeTables tables;
    fill_decode_tables(&tables);
    __syncthreads();

    if (m <= 1) {
        gemv_nvfp4_rows<1>(w_packed, w_scales, w_scale_2, x, y, n, k, m, tables);
    } else if (m <= 2) {
        gemv_nvfp4_rows<2>(w_packed, w_scales, w_scale_2, x, y, n, k, m, tables);
    } else if (m <= 4) {
        gemv_nvfp4_rows<4>(w_packed, w_scales, w_scale_2, x, y, n, k, m, tables);
    } else if (m <= 8) {
        gemv_nvfp4_rows<8>(w_packed, w_scales, w_scale_2, x, y, n, k, m, tables);
    } else if (m <= 16) {
        gemv_nvfp4_rows<16>(w_packed, w_scales, w_scale_2, x, y, n, k, m, tables);
    } else {
        gemv_nvfp4_rows<gemv_max_decode_batch>(
            w_packed, w_scales, w_scale_2, x, y, n, k, m, tables);
    }
}

// FP8 E4M3 weights with one BF16 inverse scale per 128x128 tile (the MTP and
// requantised dense layout): `y[n] = sum_k x[k] * w[n, k] * scale_inv[...]`.
extern "C" __global__ void __launch_bounds__(gemv_block_threads)
gemv_fp8_blockwise_bf16(const uint8_t *__restrict__ w,
                        const __nv_bfloat16 *__restrict__ scale_inv,
                        const __nv_bfloat16 *__restrict__ x,
                        __nv_bfloat16 *__restrict__ y,
                        int n, int k) {
    __shared__ DecodeTables tables;
    fill_decode_tables(&tables);
    __syncthreads();

    const RowSlot slot = row_slot();
    if (slot.row >= n) {
        return;
    }

    // scale_inv is [ceil(n / 128), ceil(k / 128)], row major.
    const int scale_cols = (k + fp8_scale_tile - 1) / fp8_scale_tile;
    const int scale_row = slot.row / fp8_scale_tile;
    const uint8_t *w_row = w + static_cast<size_t>(slot.row) * static_cast<size_t>(k);
    const __nv_bfloat16 *scale_row_ptr =
        scale_inv + static_cast<size_t>(scale_row) * static_cast<size_t>(scale_cols);

    const int units = k / nvfp4_block_elems;  // one uint4 == 16 E4M3 weights
    float acc = 0.0f;
    for (int unit = slot.chunk_lane; unit < units; unit += gemv_lanes_per_row) {
        const uint4 packed = *reinterpret_cast<const uint4 *>(w_row + unit * 16);
        const int column = unit * nvfp4_block_elems;
        const uint32_t scale_bits =
            __bfloat16_as_ushort(scale_row_ptr[column / fp8_scale_tile]);
        const float scale = __uint_as_float(scale_bits << 16);
        const uint4 xa = *reinterpret_cast<const uint4 *>(x + column);
        const uint4 xb = *reinterpret_cast<const uint4 *>(x + column + 8);
        const float partial = dot_e4m3_4(packed.x, xa.x, xa.y, tables.e4m3) +
                              dot_e4m3_4(packed.y, xa.z, xa.w, tables.e4m3) +
                              dot_e4m3_4(packed.z, xb.x, xb.y, tables.e4m3) +
                              dot_e4m3_4(packed.w, xb.z, xb.w, tables.e4m3);
        acc = fmaf(partial, scale, acc);
    }

    const float total = reduce_row_group(acc, slot.group_mask);
    if (slot.chunk_lane == 0) {
        y[slot.row] = __float2bfloat16(total);
    }
}

// BF16 weights: the router, the lm_head fallback and the numerical tests.
extern "C" __global__ void __launch_bounds__(gemv_block_threads)
gemv_bf16(const __nv_bfloat16 *__restrict__ w,
          const __nv_bfloat16 *__restrict__ x,
          __nv_bfloat16 *__restrict__ y,
          int n, int k) {
    const RowSlot slot = row_slot();
    if (slot.row >= n) {
        return;
    }

    const __nv_bfloat16 *w_row =
        w + static_cast<size_t>(slot.row) * static_cast<size_t>(k);
    const int units = k / 8;  // one uint4 == 8 BF16 weights
    float acc = 0.0f;
    for (int unit = slot.chunk_lane; unit < units; unit += gemv_lanes_per_row) {
        const uint4 packed = *reinterpret_cast<const uint4 *>(w_row + unit * 8);
        const uint4 values = *reinterpret_cast<const uint4 *>(x + unit * 8);
        acc += dot_bf16_8(packed, values);
    }

    const float total = reduce_row_group(acc, slot.group_mask);
    if (slot.chunk_lane == 0) {
        y[slot.row] = __float2bfloat16(total);
    }
}

// Control run for the bandwidth harness: streams `bytes` of `src` with the same
// 16-byte granularity as the GEMV weight loop and no arithmetic or stores, so
// the measured figure is the achievable read bandwidth for that buffer size.
extern "C" __global__ void __launch_bounds__(gemv_block_threads)
gemv_stream_read(const uint4 *__restrict__ src, float *__restrict__ sink,
                 unsigned long long words) {
    uint32_t sum = 0u;
    for (unsigned long long index = static_cast<unsigned long long>(blockIdx.x) *
                                        static_cast<unsigned long long>(blockDim.x) +
                                    static_cast<unsigned long long>(threadIdx.x);
         index < words;
         index += static_cast<unsigned long long>(gridDim.x) *
                  static_cast<unsigned long long>(blockDim.x)) {
        const uint4 value = src[index];
        sum ^= value.x ^ value.y ^ value.z ^ value.w;
    }
    // The comparison can never hold for these buffers, so nothing is stored and
    // the loop cannot be optimised away.
    if (sum == 0x9e3779b9u) {
        sink[0] = 1.0f;
    }
}
