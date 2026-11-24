// Sparse mixture-of-experts kernels for sm_120a (GB206).
//
// One contract: `SparseMoeBlock` (reference `modeling_qwen4_exp.py` lines
// 935-1015). The block routes each token to its top-10 experts, runs each hit
// expert's MLP over the gathered rows with the native block-scaled FP4 tensor
// core, scales and scatters the results back in f32, and adds the shared expert
// gated by `sigmoid(shared_expert_gate(x))`.
//
//   moe_router_logits      bf16 [M,2560] x bf16 [E,2560]^T -> f32 logits
//   moe_shared_gemm_bf16    bf16 [M,K] x bf16 [N,K]^T -> f32, shared expert
//   moe_shared_gemm_bf16_split  same, activation given as a high/low bf16 pair
//   moe_router_topk        f32 softmax, top-k on the probabilities, renormalise
//   moe_permute            stable expert-sorted assignment list + padded offsets
//   moe_build_tiles        per-row-tile expert map for the grouped GEMM
//   moe_quantize_gathered  gathered bf16 rows -> FP4 fragments + ue4m3 scales
//   moe_grouped_gemm       OMMA.SF m16n8k64, one expert per 16-row tile
//   moe_silu_quantize      silu(gate)*up in f32, fed straight into FP4 fragments
//   moe_combine            weighted scatter-add + sigmoid-gated shared expert
//   moe_dense_f32_gemm     exact FFMA path over bf16-dequantised weights
//   moe_gather_f32         bf16 -> f32 gathered rows, zeros for padding
//   moe_convert_f32        bf16 -> f32, part of the exact path's setup
//   moe_silu_mul           silu(gate)*up in f32 for the exact path
//   moe_silu_split_bf16    the tensor-core down stage's high/low activation
//
// The FP4 fragment layouts are the ones `core/src/kernels/gemm.cu` verified
// against the hardware and `core/src/weights/pack_nvfp4.zig` defines; this file
// consumes exactly those layouts. The grouped GEMM differs from the dense one
// only in where a row tile's activation and weight fragments start: one warp,
// one expert, one 16-row tile, 64 output columns.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>

namespace {

// ---------------------------------------------------------------------------
// MMA shapes and block tiling. Properties of the ISA, not tuning knobs.
// ---------------------------------------------------------------------------

constexpr int warp_size = 32;
constexpr int omma_m = 16;
constexpr int omma_n = 8;
constexpr int omma_k = 64;
constexpr int omma_a_words = 4;  // 32 nibbles per lane
constexpr int omma_b_words = 2;  // 16 nibbles per lane
constexpr int nvfp4_block_k = 16;
constexpr int nvfp4_scale_bytes = omma_k / nvfp4_block_k;

constexpr int hmma_m = 16;
constexpr int hmma_n = 8;
constexpr int hmma_k = 16;

// One warp owns a 16-row x 64-column output tile of the grouped GEMM, so the
// expert padding granularity is the MMA's 16 rows rather than a block tile.
constexpr int tile_m = omma_m;
constexpr int tile_n = 64;
constexpr int omma_n_tiles = tile_n / omma_n;
constexpr int threads_per_block = warp_size;
constexpr int stages = 3;

// B fragment words one 64-column block consumes per 64-deep k step.
constexpr int b_stage_words = omma_n_tiles * warp_size * omma_b_words;
constexpr int b_scale_stage_words = omma_n_tiles * omma_n;
constexpr int b_chunks = b_stage_words * 4 / 16;

// The dense router/f32 GEMM tiles.
constexpr int router_threads = 128;
constexpr int router_tile_m = 64;
constexpr int router_tile_n = 64;
constexpr int router_warps = router_tile_m / hmma_m;
constexpr int router_n_tiles = router_tile_n / hmma_n;
constexpr int bf16_tile_k = 32;
constexpr int bf16_row_elements = bf16_tile_k + 8;
constexpr int bf16_row_chunks = bf16_tile_k * 2 / 16;
constexpr int bf16_slab_chunks = router_tile_m * bf16_row_chunks;

constexpr int dense_threads = 256;
constexpr int dense_tile_m = 64;
constexpr int dense_tile_n = 64;
constexpr int dense_tile_k = 16;

// Router selection and permute limits, matching `ref/moe.zig`'s `experts_max`.
constexpr int experts_static_max = 512;
constexpr int router_threads_max = 128;
constexpr int permute_threads = 512;
constexpr int combine_threads = 256;

// An invalid gathered row: `row_index` uses this for MMA padding.
constexpr uint32_t row_invalid = 0xffffffffu;

static_assert(threads_per_block == warp_size, "one warp per grouped tile");
static_assert(b_chunks == threads_per_block * 4,
              "four 16 B chunks per thread per stage");
static_assert(b_scale_stage_words == threads_per_block * 2,
              "two scale words per thread per stage");
static_assert(nvfp4_scale_bytes == 4, "one ue4m3 x4 scale word per lane");
static_assert(router_warps * warp_size == router_threads, "router block size");
static_assert(bf16_slab_chunks == 2 * router_threads,
              "two 16 B chunks per thread per bf16 slab");
static_assert(dense_tile_m % 16 == 0 && dense_tile_n % 16 == 0);
static_assert(stages >= 3, "a two-stage pipeline would not overlap copy and MMA");

// ---------------------------------------------------------------------------
// cp.async helpers (identical protocol to `gemm.cu`).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
    const uint32_t address =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address),
                 "l"(gmem));
}

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

template <int pending>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(pending));
}

// ---------------------------------------------------------------------------
// Codecs and MMA instructions (identical to `gemm.cu`, which verified them).
// ---------------------------------------------------------------------------

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

__device__ __forceinline__ uint32_t bf16_pair(const __nv_bfloat16 *row, int k) {
    const uint32_t low = __bfloat16_as_ushort(row[k]);
    const uint32_t high = __bfloat16_as_ushort(row[k + 1]);
    return low | (high << 16);
}

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

// silu, computed the way the reference does: x / (1 + exp(-x)).
__device__ __forceinline__ float silu(float value) {
    return value / (1.f + expf(-value));
}

__device__ __forceinline__ float sigmoid(float value) {
    return 1.f / (1.f + expf(-value));
}

// ---------------------------------------------------------------------------
// 1. BF16 tensor-core GEMM: bf16 [M,K] x bf16 [N,K]^T -> f32 [M,N].
//
// One body, three entry points: the router logits over the expert gate rows and
// the shared expert's gate_up and down projections. Both operands are bf16 and
// the MMA accumulates in f32, so the only residual against the oracle is f32
// accumulation order - the same term the exact FFMA path carries.
//
// `splits` selects how the activation operand is presented:
//   1 - the activation is exactly bf16 (the router's `x`, and the shared
//       gate_up's `x`, which the checkpoint ships as bf16 already), so one
//       HMMA per k step is the whole product.
//   2 - the activation is f32 (`silu(gate)*up`, the shared down projection's
//       operand). A single bf16 rounding would cost 2^-9 relative on every
//       element, which the 5e-4 block gate cannot afford to absorb, so the
//       caller splits it into a high and a low bf16 term whose sum reproduces
//       the f32 value to 2^-18 relative, and both products accumulate into the
//       same f32 accumulator.
//
// Two terms is where the split stops, and that was measured, not assumed: a
// three-term split (2^-27, indistinguishable from the FFMA path's f32 operand)
// moved the block's observed error from 5.475e-6 to 5.856e-6 - no improvement,
// because the residual is the MMA's own accumulation order, not the operand.
// The second term is not optional: with the activation rounded to a single bf16
// the block's observed error is 1.380e-3, and the 5e-4 gate fails.
//
// `n_cols` is both the output width and its stride: the router's logit row is
// `n_padded` wide (the caller zero-fills the gate rows past `n_experts` and
// ignores those logits), and the shared expert's outputs are exactly as wide as
// they are long.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void copy_bf16_slab(__nv_bfloat16 *tile,
                                              const __nv_bfloat16 *source,
                                              int row_base, int rows_live,
                                              int column_base, int k_cols) {
    for (int index = 0; index < 2; ++index) {
        const int chunk = threadIdx.x + index * router_threads;
        const int row = chunk / bf16_row_chunks;
        const int element = (chunk % bf16_row_chunks) * 8;
        const int64_t offset =
            static_cast<int64_t>(row_base + row) * k_cols + column_base + element;
        cp_async16_pred(tile + row * bf16_row_elements + element,
                        source + offset, row < rows_live);
    }
}

// Where one (term, stage) activation slab starts. `splits` terms are pipelined
// together, so a k step's copies for all terms commit as one group and the MMA
// consumes them as one.
__device__ __forceinline__ __nv_bfloat16 *activation_slab(__nv_bfloat16 *base,
                                                         int part, int stage) {
    return base + (part * stages + stage) * (router_tile_m * bf16_row_elements);
}

template <int splits>
__device__ __forceinline__ void bf16_gemm_tile(
    const __nv_bfloat16 *a, const __nv_bfloat16 *a_low,
    const __nv_bfloat16 *b, float *c, int m_rows, int n_cols, int k_cols) {
    // 64 x 64 output tile, one warp per 16 rows.
    __shared__ __nv_bfloat16 a_stage[splits][stages][router_tile_m * bf16_row_elements];
    __shared__ __nv_bfloat16 b_stage[stages][router_tile_m * bf16_row_elements];

    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const int group = lane / 4;
    const int quad = lane % 4;
    const int k_steps = k_cols / bf16_tile_k;
    const int m0 = blockIdx.y * router_tile_m;
    const int n0 = blockIdx.x * router_tile_n;
    const int rows_live = m_rows - m0 < router_tile_m ? m_rows - m0 : router_tile_m;

    for (int stage = 0; stage < stages - 1; ++stage) {
        if (stage < k_steps) {
            for (int part = 0; part < splits; ++part) {
                copy_bf16_slab(activation_slab(&a_stage[0][0][0], part, stage),
                               part == 0 ? a : a_low, m0, rows_live,
                               stage * bf16_tile_k, k_cols);
            }
            copy_bf16_slab(b_stage[stage], b, n0, router_tile_n,
                           stage * bf16_tile_k, k_cols);
        }
        cp_async_commit();
    }

    float acc[router_n_tiles][4];
    for (int n_tile = 0; n_tile < router_n_tiles; ++n_tile) {
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
            for (int part = 0; part < splits; ++part) {
                copy_bf16_slab(activation_slab(&a_stage[0][0][0], part, prefetch % stages),
                               part == 0 ? a : a_low, m0, rows_live,
                               prefetch * bf16_tile_k, k_cols);
            }
            copy_bf16_slab(b_stage[prefetch % stages], b, n0, router_tile_n,
                           prefetch * bf16_tile_k, k_cols);
        }
        cp_async_commit();

        const __nv_bfloat16 *b_tile = &b_stage[k_step % stages][0];
        for (int part = 0; part < splits; ++part) {
            const __nv_bfloat16 *a_tile =
                activation_slab(&a_stage[0][0][0], part, k_step % stages);
            for (int half = 0; half < bf16_tile_k / hmma_k; ++half) {
                const int k_low = half * hmma_k + 2 * quad;
                const uint32_t a_words[4] = {
                    bf16_pair(a_tile + row_low * bf16_row_elements, k_low),
                    bf16_pair(a_tile + row_high * bf16_row_elements, k_low),
                    bf16_pair(a_tile + row_low * bf16_row_elements, k_low + 8),
                    bf16_pair(a_tile + row_high * bf16_row_elements, k_low + 8),
                };
                for (int n_tile = 0; n_tile < router_n_tiles; ++n_tile) {
                    const __nv_bfloat16 *b_row =
                        b_tile + (n_tile * hmma_n + group) * bf16_row_elements;
                    hmma_bf16(acc[n_tile], a_words, bf16_pair(b_row, k_low),
                              bf16_pair(b_row, k_low + 8));
                }
            }
        }
    }

    const int m_low = m0 + row_low;
    const int m_high = m0 + row_high;
    const int column = n0 + 2 * quad;
    for (int n_tile = 0; n_tile < router_n_tiles; ++n_tile) {
        const int n = column + n_tile * hmma_n;
        if (m_low < m_rows) {
            float *destination = c + static_cast<int64_t>(m_low) * n_cols + n;
            destination[0] = acc[n_tile][0];
            destination[1] = acc[n_tile][1];
        }
        if (m_high < m_rows) {
            float *destination = c + static_cast<int64_t>(m_high) * n_cols + n;
            destination[0] = acc[n_tile][2];
            destination[1] = acc[n_tile][3];
        }
    }
}

// Router logits: one 64-column block of the expert gate rows per block.
extern "C" __global__ void __launch_bounds__(router_threads)
moe_router_logits(const __nv_bfloat16 *__restrict__ a,
                  const __nv_bfloat16 *__restrict__ b, float *__restrict__ logits,
                  int m_rows, int n_padded, int k_cols) {
    bf16_gemm_tile<1>(a, nullptr, b, logits, m_rows, n_padded, k_cols);
}

// Shared expert gate_up: the activation is the checkpoint's bf16 `x`.
extern "C" __global__ void __launch_bounds__(router_threads)
moe_shared_gemm_bf16(const __nv_bfloat16 *__restrict__ a,
                     const __nv_bfloat16 *__restrict__ b,
                     float *__restrict__ c, int rows, int n_cols, int k_cols) {
    bf16_gemm_tile<1>(a, nullptr, b, c, rows, n_cols, k_cols);
}

// Shared expert down: the f32 activation arrives as a high/low bf16 pair.
extern "C" __global__ void __launch_bounds__(router_threads)
moe_shared_gemm_bf16_split(const __nv_bfloat16 *__restrict__ a_high,
                           const __nv_bfloat16 *__restrict__ a_low,
                           const __nv_bfloat16 *__restrict__ b,
                           float *__restrict__ c, int rows, int n_cols,
                           int k_cols) {
    bf16_gemm_tile<2>(a_high, a_low, b, c, rows, n_cols, k_cols);
}

// ---------------------------------------------------------------------------
// 2. Router top-k: softmax in f32, top-k on the probabilities, renormalise.
//
// Ties are broken exactly like `ref/moe.zig`'s `topk_descending`: the scan
// keeps the first maximum, so equal probabilities select the lowest expert
// index. Inferno fixes that order deliberately; `torch.topk` does not, and
// its own docstring says so: "the indices of tied elements are not
// guaranteed to be stable". Measured on CPU torch 2.15.0+cpu (the version
// these fixtures were generated with, see tests/golden/README.md); the repro
// below runs from the repo root and three runs agreed:
//
//   cd tools/ref-trace && ./.venv/bin/python - <<'PY'
//   import torch
//   f = lambda n: torch.topk(torch.ones(1, n), 10)[1][0].tolist()
//   print(f(16)); print(f(512))
//   PY
//   # -> [8, 15, 14, 13, 12, 11, 10, 9, 2, 3]
//   #    [344, 337, 338, 339, 340, 341, 342, 343, 350, 349]
//
// That venv is CPU-only (`torch.cuda.is_available()` is False), so nothing
// is claimed here about torch's tie order on a GPU. The oracle, not torch,
// is what this kernel is gated against. One block per token: 128 threads
// reduce the max, thread 0 sums the probabilities in index order (the
// reference's summation order, needed for the 1e-6 weight contract), then
// runs the selection serially over shared memory (10 x 512 comparisons).
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(router_threads_max)
moe_router_topk(const float *__restrict__ logits, uint32_t *__restrict__ indices,
                float *__restrict__ weights, int logit_stride, int n_experts,
                int top_k) {
    __shared__ float probs[experts_static_max];
    __shared__ uint8_t taken[experts_static_max];
    __shared__ float selected[experts_static_max];
    __shared__ float partial[router_threads_max / warp_size];
    __shared__ float largest;
    __shared__ float total;

    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    if (n_experts > experts_static_max || top_k > n_experts) {
        return;
    }
    const float *row = logits + static_cast<int64_t>(token) * logit_stride;

    float local = -INFINITY;
    for (int e = tid; e < n_experts; e += router_threads_max) {
        local = fmaxf(local, row[e]);
    }
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, offset));
    }
    if (tid % warp_size == 0) {
        partial[tid / warp_size] = local;
    }
    __syncthreads();
    if (tid == 0) {
        float value = partial[0];
        for (int w = 1; w < router_threads_max / warp_size; ++w) {
            value = fmaxf(value, partial[w]);
        }
        largest = value;
    }
    __syncthreads();

    for (int e = tid; e < n_experts; e += router_threads_max) {
        probs[e] = expf(row[e] - largest);
    }
    __syncthreads();
    // Summed sequentially in index order, not as a tree: the reference sums its
    // probability vector in that order, and the router-weight gate is 1e-6
    // relative, which a different summation order can miss.
    if (tid == 0) {
        float value = 0.f;
        for (int e = 0; e < n_experts; ++e) {
            value += probs[e];
        }
        total = value;
    }
    __syncthreads();

    const float inverse = 1.f / total;
    for (int e = tid; e < n_experts; e += router_threads_max) {
        probs[e] *= inverse;
        taken[e] = 0u;
    }
    __syncthreads();

    if (tid == 0) {
        // Selection in the reference's order: strictly greater keeps the lower
        // index of an equal pair, and the taken mask makes each pick unique.
        for (int slot = 0; slot < top_k; ++slot) {
            float best_value = -INFINITY;
            int best = 0;
            for (int e = 0; e < n_experts; ++e) {
                if (!taken[e] && probs[e] > best_value) {
                    best_value = probs[e];
                    best = e;
                }
            }
            taken[best] = 1u;
            selected[slot] = best_value;
            indices[static_cast<int64_t>(token) * top_k + slot] =
                static_cast<uint32_t>(best);
        }
        // `norm_topk_prob`: divide by the sum of the selected values, summed in
        // selection order exactly as the reference does, then cast.
        float norm = 0.f;
        for (int slot = 0; slot < top_k; ++slot) {
            norm += selected[slot];
        }
        const float scale = 1.f / norm;
        for (int slot = 0; slot < top_k; ++slot) {
            weights[static_cast<int64_t>(token) * top_k + slot] =
                selected[slot] * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Permute: stable expert-sorted assignment list.
//
// One block, one thread per expert. Thread e counts its pairs, a prefix sum in
// shared memory gives each expert a 16-row-aligned offset, then thread e walks
// the pairs in input order (token-major, slot-minor) and scatters, which is the
// stable order. Rows past an expert's count, and every tile past the live tile
// count, are marked `row_invalid`.
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(permute_threads)
moe_permute(const uint32_t *__restrict__ indices, int m_rows, int top_k,
            int n_experts, uint32_t *__restrict__ row_index,
            uint32_t *__restrict__ assign_pos,
            uint32_t *__restrict__ expert_offsets,
            uint32_t *__restrict__ expert_counts, int capacity_rows) {
    __shared__ uint32_t counts[experts_static_max];
    __shared__ uint32_t offsets[experts_static_max];

    const int expert = threadIdx.x;
    const int pairs = m_rows * top_k;
    if (expert < n_experts) {
        uint32_t count = 0;
        for (int pair = 0; pair < pairs; ++pair) {
            count += static_cast<uint32_t>(indices[pair] ==
                                           static_cast<uint32_t>(expert));
        }
        counts[expert] = count;
        expert_counts[expert] = count;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t prefix = 0;
        for (int e = 0; e < n_experts; ++e) {
            offsets[e] = prefix;
            expert_offsets[e] = prefix;
            const uint32_t padded = (counts[e] + (tile_m - 1)) / tile_m * tile_m;
            prefix += padded;
        }
        expert_offsets[n_experts] = prefix;
    }
    __syncthreads();

    if (expert < n_experts) {
        uint32_t cursor = offsets[expert];
        for (int pair = 0; pair < pairs; ++pair) {
            if (indices[pair] == static_cast<uint32_t>(expert)) {
                row_index[cursor] = static_cast<uint32_t>(pair / top_k);
                assign_pos[pair] = cursor - offsets[expert];
                cursor += 1;
            }
        }
        const uint32_t end = expert_offsets[expert + 1];
        for (uint32_t row = cursor; row < end; ++row) {
            row_index[row] = row_invalid;
        }
    }
    __syncthreads();

    // Every slot the grouped GEMM may visit must name an expert or be invalid.
    for (int row = threadIdx.x + expert_offsets[n_experts]; row < capacity_rows;
         row += permute_threads) {
        row_index[row] = row_invalid;
    }
}

// ---------------------------------------------------------------------------
// 4. Tile map: per gathered row tile, the expert and the live row count.
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(permute_threads)
moe_build_tiles(const uint32_t *__restrict__ expert_offsets,
                const uint32_t *__restrict__ expert_counts, int n_experts,
                int total_tiles, int *__restrict__ tile_expert,
                int *__restrict__ tile_row_base,
                int *__restrict__ tile_rows_live) {
    const int expert = threadIdx.x;
    for (int tile = expert; tile < total_tiles; tile += permute_threads) {
        tile_expert[tile] = -1;
    }
    __syncthreads();

    if (expert < n_experts) {
        const uint32_t base = expert_offsets[expert];
        const uint32_t end = expert_offsets[expert + 1];
        const int count = static_cast<int>(expert_counts[expert]);
        for (uint32_t row = base; row < end; row += tile_m) {
            const int tile = static_cast<int>(row / tile_m);
            const int done = static_cast<int>(row - base);
            tile_expert[tile] = expert;
            tile_row_base[tile] = static_cast<int>(row);
            tile_rows_live[tile] = min(tile_m, count - done);
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Gathered activation quantisation (gate_up stage).
//
// Same 16-row x 64-column tile and fragment layout as
// `quantize_activations_nvfp4` in `gemm.cu`, but each row is read through
// `row_index`; an invalid row quantises to zeros, which is what the MMA
// padding lanes must multiply.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void quantize_block(uint32_t packed[2],
                                               uint32_t *scale_byte,
                                               const float values[16]) {
    float largest = 0.f;
    for (int j = 0; j < nvfp4_block_k; ++j) {
        largest = fmaxf(largest, fabsf(values[j]));
    }
    const float requested = largest > 0.f ? largest / 6.f : 1.f;
    const uint32_t encoded = e4m3_encode(requested);
    const float stored = e4m3_decode(encoded);
    packed[0] = 0u;
    packed[1] = 0u;
    for (int j = 0; j < nvfp4_block_k; ++j) {
        const uint32_t code = e2m1_encode(values[j] / stored);
        packed[j / 8] |= code << (4 * (j % 8));
    }
    *scale_byte = encoded;
}

// Writes the two words and the scale of one (row, k block) into fragment order.
__device__ __forceinline__ void write_fragment(uint32_t *frag, uint32_t *scales,
                                               int frag_base, int scale_base,
                                               int row, int k_block,
                                               const uint32_t packed[2],
                                               uint32_t scale_byte) {
    const int reg = (k_block / 2) * 2 + (row / 8);
    const int lane_low = 4 * (row % 8) + 2 * (k_block % 2);
    frag[frag_base + lane_low * omma_a_words + reg] = packed[0];
    frag[frag_base + (lane_low + 1) * omma_a_words + reg] = packed[1];
    const int lane_scale = 4 * (row % 8) + (row / 8);
    uint8_t *scale_out =
        reinterpret_cast<uint8_t *>(&scales[scale_base + lane_scale]);
    scale_out[k_block] = static_cast<uint8_t>(scale_byte);
}

extern "C" __global__ void __launch_bounds__(warp_size)
moe_quantize_gathered(const __nv_bfloat16 *__restrict__ x,
                      const uint32_t *__restrict__ row_index, int k_cols,
                      uint32_t *__restrict__ frag,
                      uint32_t *__restrict__ scale_frag) {
    const int lane = threadIdx.x;
    const int k_tile = blockIdx.x;
    const int m_tile = blockIdx.y;
    const int k_tiles = k_cols / omma_k;
    const int k_base = k_tile * omma_k;
    const int scale_base = (m_tile * k_tiles + k_tile) * warp_size;
    const int frag_base = scale_base * omma_a_words;

    for (int step = 0; step < 2; ++step) {
        const int block = lane + step * warp_size;
        const int row = block / nvfp4_scale_bytes;
        const int k_block = block % nvfp4_scale_bytes;
        const uint32_t token = row_index[m_tile * omma_m + row];
        const bool live = token != row_invalid;
        const int k0 = k_base + k_block * nvfp4_block_k;

        float values[nvfp4_block_k];
        for (int j = 0; j < nvfp4_block_k; ++j) {
            values[j] = live ? bf16_to_f32(
                                   x[static_cast<int64_t>(token) * k_cols + k0 + j])
                             : 0.f;
        }
        uint32_t packed[2];
        uint32_t scale_byte;
        quantize_block(packed, &scale_byte, values);
        write_fragment(frag, scale_frag, frag_base, scale_base, row, k_block,
                       packed, scale_byte);
    }
}

// ---------------------------------------------------------------------------
// 6. Grouped NVFP4 GEMM.
//
// Grid: x = 64-column blocks, y = gathered row tiles. Each block reads its
// expert and row base from the tile map, streams that expert's B fragments
// through shared memory in a three-stage cp.async pipeline, and runs
// `OMMA.SF` m16n8k64 over the whole contraction. The epilogue folds the
// per-tensor weight scale into f32 and drops padding rows.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void prefetch_b(uint32_t *dst, uint32_t *dst_scale,
                                           const uint32_t *src_frag,
                                           const uint32_t *src_scale,
                                           int64_t frag_expert_base,
                                           int64_t scale_expert_base,
                                           int k_tile, int n_tile0, int n_tiles,
                                           int lane) {
    const int64_t frag_base =
        frag_expert_base +
        (static_cast<int64_t>(k_tile) * n_tiles + n_tile0) * 64;
    for (int index = 0; index < 4; ++index) {
        const int chunk = lane + index * warp_size;
        cp_async16(&dst[chunk * 4], &src_frag[frag_base + chunk * 4]);
    }
    const int64_t scale_base =
        scale_expert_base +
        (static_cast<int64_t>(k_tile) * n_tiles + n_tile0) * omma_n;
    cp_async4(&dst_scale[lane], &src_scale[scale_base + lane]);
    cp_async4(&dst_scale[lane + warp_size], &src_scale[scale_base + lane + warp_size]);
}

extern "C" __global__ void __launch_bounds__(threads_per_block)
moe_grouped_gemm(const uint32_t *__restrict__ a_frag,
                 const uint32_t *__restrict__ a_scale_frag,
                 const uint32_t *__restrict__ b_frag,
                 const uint32_t *__restrict__ b_scale_frag,
                 const int *__restrict__ tile_expert,
                 const int *__restrict__ tile_row_base,
                 const int *__restrict__ tile_rows_live, float scale2,
                 float *__restrict__ c, int n_cols, int k_cols) {
    __shared__ uint32_t b_stage[stages][b_stage_words];
    __shared__ uint32_t b_scale_stage[stages][b_scale_stage_words];

    const int lane = threadIdx.x;
    const int row_tile = blockIdx.y;
    const int expert = tile_expert[row_tile];
    if (expert < 0) {
        return;
    }
    const int row_base = tile_row_base[row_tile];
    const int rows_live = tile_rows_live[row_tile];
    const int k_tiles = k_cols / omma_k;
    const int n_tiles = n_cols / omma_n;
    const int n_tile0 = blockIdx.x * omma_n_tiles;

    const int64_t frag_expert_base =
        static_cast<int64_t>(expert) * k_tiles * n_tiles * 64;
    const int64_t scale_expert_base =
        static_cast<int64_t>(expert) * k_tiles * n_tiles * omma_n;
    const int64_t a_tile_index = static_cast<int64_t>(row_base / omma_m) * k_tiles;
    const int64_t a_base = a_tile_index * (warp_size * omma_a_words);

    for (int stage = 0; stage < stages - 1; ++stage) {
        if (stage < k_tiles) {
            prefetch_b(b_stage[stage], b_scale_stage[stage], b_frag,
                       b_scale_frag, frag_expert_base, scale_expert_base, stage,
                       n_tile0, n_tiles, lane);
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
            prefetch_b(b_stage[prefetch % stages], b_scale_stage[prefetch % stages],
                       b_frag, b_scale_frag, frag_expert_base,
                       scale_expert_base, prefetch, n_tile0, n_tiles, lane);
        }
        cp_async_commit();

        const uint32_t *b_words = &b_stage[k_tile % stages][0];
        const uint32_t *b_scales = &b_scale_stage[k_tile % stages][0];
        const int64_t a_tile_base =
            a_base + static_cast<int64_t>(k_tile) * (warp_size * omma_a_words);
        uint32_t a[omma_a_words];
        for (int word = 0; word < omma_a_words; ++word) {
            a[word] = a_frag[a_tile_base + lane * omma_a_words + word];
        }
        const uint32_t scale_a =
            a_scale_frag[(a_tile_index + k_tile) * warp_size + lane];

        for (int n_tile = 0; n_tile < omma_n_tiles; ++n_tile) {
            const uint32_t b0 =
                b_words[(n_tile * warp_size + lane) * omma_b_words];
            const uint32_t b1 =
                b_words[(n_tile * warp_size + lane) * omma_b_words + 1];
            const uint32_t scale_b = b_scales[n_tile * omma_n + lane / 4];
            omma_nvfp4(acc[n_tile], a, b0, b1, scale_a, scale_b);
        }
    }

    const int row_low = row_base + lane / 4;
    const int column = blockIdx.x * tile_n + 2 * (lane % 4);
    for (int n_tile = 0; n_tile < omma_n_tiles; ++n_tile) {
        const int n = column + n_tile * omma_n;
        if (lane / 4 < rows_live) {
            float *destination = c + static_cast<int64_t>(row_low) * n_cols + n;
            destination[0] = acc[n_tile][0] * scale2;
            destination[1] = acc[n_tile][1] * scale2;
        }
        if (lane / 4 + 8 < rows_live) {
            float *destination =
                c + static_cast<int64_t>(row_low + 8) * n_cols + n;
            destination[0] = acc[n_tile][2] * scale2;
            destination[1] = acc[n_tile][3] * scale2;
        }
    }
}

// ---------------------------------------------------------------------------
// 7. silu(gate) * up in f32, quantised straight into the down stage's
// activation fragments. Invalid gathered rows write zeros.
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(warp_size)
moe_silu_quantize(const float *__restrict__ gate_up,
                  const uint32_t *__restrict__ row_index, int intermediate,
                  uint32_t *__restrict__ frag,
                  uint32_t *__restrict__ scale_frag) {
    const int lane = threadIdx.x;
    const int k_tile = blockIdx.x;
    const int m_tile = blockIdx.y;
    const int k_tiles = intermediate / omma_k;
    const int k_base = k_tile * omma_k;
    const int width = 2 * intermediate;
    const int scale_base = (m_tile * k_tiles + k_tile) * warp_size;
    const int frag_base = scale_base * omma_a_words;

    for (int step = 0; step < 2; ++step) {
        const int block = lane + step * warp_size;
        const int row = block / nvfp4_scale_bytes;
        const int k_block = block % nvfp4_scale_bytes;
        const int gathered = m_tile * omma_m + row;
        const uint32_t token = row_index[gathered];
        const bool live = token != row_invalid;
        const int k0 = k_base + k_block * nvfp4_block_k;

        float values[nvfp4_block_k];
        for (int j = 0; j < nvfp4_block_k; ++j) {
            const int64_t source = static_cast<int64_t>(gathered) * width + k0 + j;
            values[j] = live ? silu(gate_up[source]) * gate_up[source + intermediate]
                             : 0.f;
        }
        uint32_t packed[2];
        uint32_t scale_byte;
        quantize_block(packed, &scale_byte, values);
        write_fragment(frag, scale_frag, frag_base, scale_base, row, k_block,
                       packed, scale_byte);
    }
}

// ---------------------------------------------------------------------------
// 8. Combine: routing-weighted scatter-add of the per-expert down outputs, plus
// the shared expert scaled by sigmoid(x . shared_expert_gate).
//
// One block per token, 256 threads: the block reduces the shared gate dot once,
// then each thread sums its columns over the token's top-k assignments in slot
// order (deterministic, no atomics) and adds the gated shared output.
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(combine_threads)
moe_combine(const uint32_t *__restrict__ indices,
            const uint32_t *__restrict__ assign_pos,
            const float *__restrict__ router_weights,
            const uint32_t *__restrict__ expert_offsets,
            const float *__restrict__ down_out,
            const float *__restrict__ shared_out,
            const float *__restrict__ shared_gate_logit,
            const float *__restrict__ x, int top_k, int k_cols,
            float *__restrict__ out) {
    __shared__ float partial[combine_threads / warp_size];
    __shared__ float gate_value;

    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const float *row = x + static_cast<int64_t>(token) * k_cols;

    float local = 0.f;
    for (int column = tid; column < k_cols; column += combine_threads) {
        local += row[column] * shared_gate_logit[column];
    }
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        local += __shfl_down_sync(0xffffffffu, local, offset);
    }
    if (tid % warp_size == 0) {
        partial[tid / warp_size] = local;
    }
    __syncthreads();
    if (tid == 0) {
        float dot = 0.f;
        for (int w = 0; w < combine_threads / warp_size; ++w) {
            dot += partial[w];
        }
        gate_value = sigmoid(dot);
    }
    __syncthreads();

    for (int column = tid; column < k_cols; column += combine_threads) {
        float total = 0.f;
        for (int slot = 0; slot < top_k; ++slot) {
            const int pair = token * top_k + slot;
            const uint32_t expert = indices[pair];
            const int64_t gathered =
                static_cast<int64_t>(expert_offsets[expert]) + assign_pos[pair];
            total += down_out[gathered * k_cols + column] * router_weights[pair];
        }
        out[static_cast<int64_t>(token) * k_cols + column] =
            total + gate_value * shared_out[static_cast<int64_t>(token) * k_cols + column];
    }
}

// ---------------------------------------------------------------------------
// 9. Exact reference path: bf16-dequantised weights, f32 arithmetic.
//
// The dequantised weight matrix is uploaded as bf16 without the per-tensor
// scale: an e2m1 value times a ue4m3 block scale needs six mantissa bits, so
// bf16 holds it exactly, and the per-tensor f32 scale is folded into the
// epilogue where it costs no precision. The f32 FFMA product below is then the
// same arithmetic as the CPU oracle. The kernel is not timed as a throughput
// path: it is the numerical reference.
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(dense_threads)
moe_dense_f32_gemm(const float *__restrict__ a, const __nv_bfloat16 *__restrict__ b,
                   float scale, float *__restrict__ c, int rows, int n_cols,
                   int k_cols) {
    __shared__ float a_tile[dense_tile_m][dense_tile_k + 1];
    __shared__ float b_tile[dense_tile_n][dense_tile_k + 1];

    const int lane = threadIdx.x % 16;
    const int warp = threadIdx.x / 16;
    const int m0 = blockIdx.y * dense_tile_m;
    const int n0 = blockIdx.x * dense_tile_n;

    float acc[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.f;
        }
    }

    for (int k0 = 0; k0 < k_cols; k0 += dense_tile_k) {
        for (int index = 0; index < 4; ++index) {
            const int position = threadIdx.x + index * dense_threads;
            const int row = position / dense_tile_k;
            const int column = position % dense_tile_k;
            const int global_row = m0 + row;
            a_tile[row][column] =
                global_row < rows
                    ? a[static_cast<int64_t>(global_row) * k_cols + k0 + column]
                    : 0.f;
            const int global_column = n0 + row;
            b_tile[row][column] =
                global_column < n_cols
                    ? bf16_to_f32(b[static_cast<int64_t>(global_column) * k_cols +
                                    k0 + column])
                    : 0.f;
        }
        __syncthreads();
        for (int kk = 0; kk < dense_tile_k; ++kk) {
            for (int i = 0; i < 4; ++i) {
                const float left = a_tile[warp * 4 + i][kk];
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += left * b_tile[lane * 4 + j][kk];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < 4; ++i) {
        const int row = m0 + warp * 4 + i;
        if (row < rows) {
            for (int j = 0; j < 4; ++j) {
                c[static_cast<int64_t>(row) * n_cols + n0 + lane * 4 + j] =
                    acc[i][j] * scale;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 10. Setup kernels for the exact path: gather + convert + silu-mul.
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(256)
moe_gather_f32(const __nv_bfloat16 *__restrict__ x,
               const uint32_t *__restrict__ row_index, int rows_padded,
               int k_cols, float *__restrict__ out) {
    const int64_t count = static_cast<int64_t>(rows_padded) * k_cols;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index / k_cols);
        const int column = static_cast<int>(index % k_cols);
        const uint32_t token = row_index[row];
        out[index] = token == row_invalid
                         ? 0.f
                         : bf16_to_f32(x[static_cast<int64_t>(token) * k_cols + column]);
    }
}

extern "C" __global__ void __launch_bounds__(256)
moe_convert_f32(const __nv_bfloat16 *__restrict__ x, int count,
                float *__restrict__ out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        out[index] = bf16_to_f32(x[index]);
    }
}

extern "C" __global__ void __launch_bounds__(256)
moe_silu_mul(const float *__restrict__ gate_up, int rows, int intermediate,
             float *__restrict__ act) {
    const int64_t count = static_cast<int64_t>(rows) * intermediate;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index / intermediate);
        const int column = static_cast<int>(index % intermediate);
        const int64_t base = static_cast<int64_t>(row) * 2 * intermediate + column;
        act[index] = silu(gate_up[base]) * gate_up[base + intermediate];
    }
}

// The tensor-core path's activation operand: silu(gate)*up split into a high
// and a low bf16 term. `high + low` reproduces the f32 value to 2^-18 relative,
// and `moe_shared_gemm_bf16_split` accumulates both products into one f32
// accumulator, so the down projection's operand carries sixteen mantissa bits
// instead of the eight a single bf16 rounding would leave.
extern "C" __global__ void __launch_bounds__(256)
moe_silu_split_bf16(const float *__restrict__ gate_up, int rows,
                    int intermediate, __nv_bfloat16 *__restrict__ high,
                    __nv_bfloat16 *__restrict__ low) {
    const int64_t count = static_cast<int64_t>(rows) * intermediate;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index / intermediate);
        const int column = static_cast<int>(index % intermediate);
        const int64_t base = static_cast<int64_t>(row) * 2 * intermediate + column;
        const float value = silu(gate_up[base]) * gate_up[base + intermediate];
        const __nv_bfloat16 upper = __float2bfloat16(value);
        high[index] = upper;
        low[index] = __float2bfloat16(value - __bfloat162float(upper));
    }
}

} // namespace
