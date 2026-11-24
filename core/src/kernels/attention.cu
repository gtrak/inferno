// Indexed (QSA) attention: tensor-core sparse prefill attention, single-token
// decode, the indexer's streaming block selection, and the partial mRoPE tables
// that feed them.
//
// Reference: tools/ref-trace/reference/modeling_qwen4_exp.py lines 643-916
// (`apply_rotary_pos_emb`, `Qwen4ExpTextQSAIndexer`, `Qwen4ExpTextAttention`).
// The fp32 CPU oracle this file is checked against is core/src/ref/attention.zig.
//
// Buffer contracts. Every shape is a launch argument; the constants below are
// the bounds the kernels check before touching memory.
//
//   gathered_k  bf16 [count, kv_heads, head_dim]  selected keys, dense
//   gathered_v  bf16 [count, kv_heads, head_dim]  their values, dense
//   gather_index int32 [count]  absolute token position of each gathered row
//   positions    int32 [rows]   absolute position of each query
//   selection_mask uint32 [rows, ceil(count / 32)]  per-row QSA bitmap, optional
//   index keys  bf16 [arena_length, index_dim]  raw (un-normalised) token keys
//   pooled      bf16 [blocks, index_dim]         one pooled key per complete block
//   scores      f32  [rows, 32768]               one streaming group's scores
//   partial     u64  [rows, block_topk]          running packed top-k
//   selected    int32 [rows, slots], -1 padded, slots = budget + ratio - 1
//
// Prefill runs on the tensor cores (`attn_prefill_mma`): m16n8k16 bf16 QK^T and
// fp16 PV over a 16-query-row m-tile, with four warps splitting the 64-key tile
// and the output dims. The warp-scalar `attn_prefill` remains the fallback for
// head dims the tiling cannot take and for one-row launches, which would pad
// fifteen rows of tensor-core work. Both honour the per-row selection bitmap,
// which is what makes a shared union gather exactly QSA at rows > 1.
//
// Attention never sees the master KV cache. The design keeps a hot window in
// VRAM and a DRAM tail that is gathered on demand (`.agents/design/
// full-context-single-stream.md` section 7), so the kernels take an explicit
// dense working set of at most `budget` rows and do plain dense attention over
// it. `gather_index` is read for one purpose only: the causal check. A row
// whose position is past the query's own is skipped and counted, so a
// corrupted gather cannot read the future.
//
// The optional `violations` counter is a tripwire for a corrupted selection:
// it tallies every gathered row that is past the launch's causal horizon, once
// each, by the first head of the row, so its value does not depend on the head
// count. Prefill's horizon is the position of its *last* query row, because a
// prefill chunk may share one union working set across its rows: rows are
// masked per query, so only an index past the chunk itself is corrupt. A
// legitimate launch therefore always reports zero, and the host asserts it.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

/// Lanes per warp; every reduction below is warp-wide.
constexpr int warp_size = 32;

/// One `__nv_bfloat162` per lane per pass.
constexpr int pair_width = 2;

/// Head dims covered by one warp pass.
constexpr int dims_per_warp_pass = warp_size * pair_width;

/// Passes unrolled per head; head_dim <= 512 keeps the accumulators in
/// registers (the loops below are constant-bound so they never spill).
constexpr int lane_passes_max = 8;

constexpr int head_dim_max = dims_per_warp_pass * lane_passes_max;

/// `budget + compress_ratio - 1` for the real model: 2048 + 4 - 1.
constexpr int selected_slots_max = 2064;

/// Query heads one prefill block attends to (one warp each).
constexpr int prefill_warps = 4;

/// Warps one decode block spreads the selection over.
constexpr int decode_warps = 8;

/// Radix-select histogram bins and passes: 8 passes of 8 bits over a u64 key,
/// of which the first four are the score and the last four the block index.
constexpr int radix_bins = 256;
constexpr int radix_digits = 8;
constexpr int score_digits = 4;

/// Largest block count `indexer_select` can keep; `block_topk` is 512.
constexpr int select_topk_max = 512;

/// Keys one streaming merge keeps in shared memory: the running partial top-k
/// plus one group's top-k.
constexpr int merge_keys_max = 2 * select_topk_max;

/// Blocks one CTA scores: the per-block work is 128 multiply-adds, so a CTA per
/// block leaves the GPU scheduling 65 536 near-empty CTAs at the context limit.
constexpr int scores_per_cta = 16;

// ---------------------------------------------------------------------------
// Prefill attention MMA tiling.
// ---------------------------------------------------------------------------

/// Query rows one prefill CTA attends to, and the bf16 MMA's m extent.
constexpr int mma_rows = 16;

/// Warps per prefill CTA: one 16-row m-tile shared by four warps, which split
/// the key tile for QK^T and the output dims for PV.
constexpr int mma_warps = 4;
constexpr int mma_threads = mma_warps * warp_size;

/// Keys one CTA iteration covers, and the keys one warp scores in it.
constexpr int mma_keys_per_tile = 64;
constexpr int mma_keys_per_warp = mma_keys_per_tile / mma_warps;  // 16

/// bf16 MMA contraction depth, and the key steps one PV tile takes.
constexpr int mma_k = 16;
constexpr int mma_key_steps_pv = mma_keys_per_tile / mma_k;  // 4

/// Output dims one warp accumulates, and the widest head the kernel takes.
constexpr int mma_dims_per_warp = 64;
constexpr int mma_dim_tiles = mma_dims_per_warp / 8;  // 8
constexpr int mma_head_dim_max = mma_dims_per_warp * mma_warps;  // 256

/// Shared row strides, padded so the 16 B tile reads spread over the banks.
constexpr int mma_p_row_elements = mma_keys_per_tile + 8;   // 72
constexpr int mma_v_row_elements = mma_head_dim_max + 8;    // 264

/// Query rows one prefill CTA's softmax loop assigns to one thread group.
constexpr int mma_softmax_threads = 8;
constexpr int mma_softmax_keys = mma_keys_per_tile / mma_softmax_threads;

/// Added to every radix key so that padding (0) sorts below every real key.
constexpr unsigned long long key_offset = 1;

/// Values below this are never read; a fade guard, not a numerical bound.
constexpr float negative_infinity = -__builtin_huge_valf();

/// Rounds a float pair into the packed bf16 storage layout.
__device__ __forceinline__ void store_bf16_pair(
    __nv_bfloat16* base,
    int dim,
    float2 value) {
    *reinterpret_cast<__nv_bfloat162*>(base + dim) =
        __floats2bfloat162_rn(value.x, value.y);
}

/// Reads two adjacent bf16 values as floats.
__device__ __forceinline__ float2 load_bf16_pair(
    const __nv_bfloat16* base,
    int dim) {
    return __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162*>(base + dim));
}

/// Shared-memory address of `pointer`, the form `ldmatrix` takes.
__device__ __forceinline__ unsigned int shared_address(const void* pointer) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(pointer));
}

/// Loads four 8x8 bf16 tiles in MMA A-fragment order.
__device__ __forceinline__ void ldmatrix_x4(
    unsigned int* registers,
    const void* address) {
    const unsigned int shared = shared_address(address);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(registers[0]), "=r"(registers[1]),
          "=r"(registers[2]), "=r"(registers[3])
        : "r"(shared));
}

/// Loads two 8x8 bf16 tiles transposed, in m16n8k16 B-fragment order.
__device__ __forceinline__ void ldmatrix_x2_trans(
    unsigned int* registers,
    const void* address) {
    const unsigned int shared = shared_address(address);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(registers[0]), "=r"(registers[1])
        : "r"(shared));
}

/// m16n8k16 bf16 x bf16 -> f32, folded into the accumulator.
__device__ __forceinline__ void mma_bf16(
    float* accumulator,
    const unsigned int* a,
    const unsigned int* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(accumulator[0]), "+f"(accumulator[1]),
          "+f"(accumulator[2]), "+f"(accumulator[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

/// m16n8k16 fp16 x fp16 -> f32, folded into the accumulator. The PV product
/// uses this: the softmax weights live in [0, 1], where fp16 keeps ten mantissa
/// bits against bf16's eight, and the gathered V is a bf16 value that fp16
/// represents exactly for every normal magnitude. That keeps the tensor-core
/// numerator within the bf16-output tolerance of the f32 reference.
__device__ __forceinline__ void mma_f16(
    float* accumulator,
    const unsigned int* a,
    const unsigned int* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(accumulator[0]), "+f"(accumulator[1]),
          "+f"(accumulator[2]), "+f"(accumulator[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

/// The dims one lane owns in warp pass `pass`.
__device__ __forceinline__ int lane_dim(int pass, int lane) {
    return pass * dims_per_warp_pass + lane * pair_width;
}

/// Sum across the warp; every lane returns the total.
__device__ __forceinline__ float warp_sum(float value) {
    float total = value;
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        total += __shfl_down_sync(0xffffffffu, total, offset);
    }
    total = __shfl_sync(0xffffffffu, total, 0);
    return total;
}

/// Largest value across the warp; every lane returns the maximum.
__device__ __forceinline__ float warp_max(float value) {
    float largest = value;
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        largest = fmaxf(largest, __shfl_down_sync(0xffffffffu, largest, offset));
    }
    largest = __shfl_sync(0xffffffffu, largest, 0);
    return largest;
}

/// Largest value across the warp, then across the warps of one block.
__device__ __forceinline__ float block_max(float value, float* scratch) {
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const float per_warp = warp_max(value);
    if (lane == 0) {
        scratch[warp] = per_warp;
    }
    __syncthreads();
    const int warps = (blockDim.x + warp_size - 1) / warp_size;
    float largest = negative_infinity;
    for (int index = 0; index < warps; ++index) {
        largest = fmaxf(largest, scratch[index]);
    }
    return largest;
}

/// Sum across the warp, then across the warps of one block.
__device__ __forceinline__ float block_sum(float value, float* scratch) {
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const float per_warp = warp_sum(value);
    if (lane == 0) {
        scratch[warp] = per_warp;
    }
    __syncthreads();
    const int warps = (blockDim.x + warp_size - 1) / warp_size;
    float total = 0.0f;
    for (int index = 0; index < warps; ++index) {
        total += scratch[index];
    }
    return total;
}

/// `x * rsqrt(mean(x^2) + eps) * (1 + weight)`, in fp32, as the reference does.
__device__ __forceinline__ float rmsnorm_scale(
    const float* row,
    int width,
    float eps,
    float* scratch) {
    float squares = 0.0f;
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        squares += row[index] * row[index];
    }
    const float total = block_sum(squares, scratch);
    const float mean = total / static_cast<float>(width);
    return rsqrtf(mean + eps);
}

/// Which of the three position grids drives frequency `index`.
///
/// The height lanes are overwritten first and the width lanes last, so width
/// wins any overlap; this mirrors `rope.mrope_grid` in the fp32 reference.
__device__ __forceinline__ int mrope_grid(int index, const int* section) {
    int grid = 0;
    if (index % 3 == 1 && index < section[1] * 3) {
        grid = 1;
    }
    if (index % 3 == 2 && index < section[2] * 3) {
        grid = 2;
    }
    return grid;
}

/// `1 / theta^(2i/rotary_dim)`, the frequency of lane `index`.
__device__ __forceinline__ float inv_frequency(
    int index,
    int rotary_dim,
    float theta) {
    const float exponent =
        2.0f * static_cast<float>(index) / static_cast<float>(rotary_dim);
    return 1.0f / powf(theta, exponent);
}

/// `low * cos - high * sin`, rounded at each step.
///
/// The explicit round-to-nearest multiplies stop `nvcc` contracting the pair
/// into an FMA, which would round differently from the fp32 reference and land
/// a different bf16 value on the rounding boundary.
__device__ __forceinline__ float rotate_low(float low, float high, float cos, float sin) {
    return __fsub_rn(__fmul_rn(low, cos), __fmul_rn(high, sin));
}

/// `high * cos + low * sin`, rounded at each step.
__device__ __forceinline__ float rotate_high(float low, float high, float cos, float sin) {
    return __fadd_rn(__fmul_rn(high, cos), __fmul_rn(low, sin));
}

/// One dim of a pooled block key after the mean, the cast back and the norm.
///
/// The mean is taken in fp32 and cast back to the cache dtype before the norm,
/// exactly as `key_groups.float().mean(dim=1).to(raw_keys.dtype)` does.
__device__ __forceinline__ float pooled_normed(
    const __nv_bfloat16* first,
    const float* k_layernorm,
    float scale,
    int dim,
    int index_dim,
    int ratio) {
    float total = 0.0f;
    for (int row = 0; row < ratio; ++row) {
        total += __bfloat162float(first[static_cast<long long>(row) * index_dim + dim]);
    }
    const float mean =
        __bfloat162float(__float2bfloat16(total / static_cast<float>(ratio)));
    return mean * scale * (1.0f + k_layernorm[dim]);
}

}  // namespace

/// Builds the `[positions, rotary_dim]` cos/sin tables (kernel 2 of the spec).
///
/// `positions` holds the three position grids; the text path passes the same
/// grid three times, which makes the recomposition a no-op.
extern "C" __global__ void rope_table_build(
    float* cos_table,
    float* sin_table,
    const int* temporal,
    const int* height,
    const int* width,
    const int* section,
    int positions,
    int rotary_dim,
    float theta) {
    const int row = blockIdx.x;
    const int half = rotary_dim / 2;
    if (row >= positions) {
        return;
    }
    for (int index = threadIdx.x; index < half; index += blockDim.x) {
        int grid = mrope_grid(index, section);
        int position = temporal[row];
        if (grid == 1) {
            position = height[row];
        } else if (grid == 2) {
            position = width[row];
        }
        const float angle =
            static_cast<float>(position) * inv_frequency(index, rotary_dim, theta);
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        cos_table[row * rotary_dim + index] = cosine;
        cos_table[row * rotary_dim + index + half] = cosine;
        sin_table[row * rotary_dim + index] = sine;
        sin_table[row * rotary_dim + index + half] = sine;
    }
}

/// Rotates the first `rotary_dim` dims of every head in place.
///
/// One thread per rotated pair: `rows * heads * rotary_dim / 2` threads.
extern "C" __global__ void rope_apply_rotary(
    __nv_bfloat16* values,
    const float* cos_table,
    const float* sin_table,
    const int* positions,
    int rows,
    int heads,
    int head_dim,
    int rotary_dim) {
    const int half = rotary_dim / 2;
    const long long total =
        static_cast<long long>(rows) * heads * static_cast<long long>(half);
    const long long index =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int slot = static_cast<int>(index % half);
    const long long above = index / half;
    const int head = static_cast<int>(above % heads);
    const int row = static_cast<int>(above / heads);
    const int position = positions[row];
    const float* cos = cos_table + static_cast<long long>(position) * rotary_dim;
    const float* sin = sin_table + static_cast<long long>(position) * rotary_dim;
    __nv_bfloat16* head_values =
        values + (static_cast<long long>(row) * heads + head) * head_dim;
    const float low = __bfloat162float(head_values[slot]);
    const float high = __bfloat162float(head_values[slot + half]);
    head_values[slot] = __float2bfloat16(
        rotate_low(low, high, cos[slot], sin[slot]));
    head_values[slot + half] = __float2bfloat16(
        rotate_high(low, high, cos[slot + half], sin[slot + half]));
}

/// `out = hidden @ weight^T`, bf16 operands with fp32 accumulation.
///
/// `weight` is `[out_features, in_features]`. One block per (row, 128 output
/// columns); the indexer's projection is small enough that this is not the
/// hot path, and I-006a owns the real GEMM.
extern "C" __global__ void indexer_project(
    const __nv_bfloat16* hidden,
    const __nv_bfloat16* weight,
    float* out,
    int rows,
    int in_features,
    int out_features) {
    const int row = blockIdx.x;
    const int column = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows || column >= out_features) {
        return;
    }
    const __nv_bfloat16* source = hidden + static_cast<long long>(row) * in_features;
    const __nv_bfloat16* row_weight =
        weight + static_cast<long long>(column) * in_features;
    float total = 0.0f;
    int base = 0;
    for (; base + 1 < in_features; base += pair_width) {
        const float2 a = load_bf16_pair(source, base);
        const float2 b = load_bf16_pair(row_weight, base);
        total += a.x * b.x + a.y * b.y;
    }
    if (base < in_features) {
        total += __bfloat162float(source[base]) *
            __bfloat162float(row_weight[base]);
    }
    out[static_cast<long long>(row) * out_features + column] = total;
}

/// Per-head RMSNorm plus partial RoPE on the indexer queries.
///
/// One block per (row, index head). The raw token keys stay untouched: the
/// reference caches them un-normalised and normalises the pooled reads.
extern "C" __global__ void indexer_norm_rope_q(
    const float* projected,
    const float* q_layernorm,
    __nv_bfloat16* queries,
    const float* cos_table,
    const float* sin_table,
    const int* positions,
    int rows,
    int index_heads,
    int index_kv_heads,
    int index_dim,
    int rotary_dim,
    float eps) {
    __shared__ float scratch[warp_size];
    __shared__ float scale;

    const int row = blockIdx.x;
    const int head = blockIdx.y;
    if (row >= rows || head >= index_heads) {
        return;
    }
    const int projected_width = (index_heads + index_kv_heads) * index_dim;
    const float* source =
        projected + static_cast<long long>(row) * projected_width +
        static_cast<long long>(head) * index_dim;

    // The normed row is recomputed from the same source twice (scale, then
    // apply) so it never needs a shared staging buffer.
    scale = rmsnorm_scale(source, index_dim, eps, scratch);
    __syncthreads();

    const int position = positions[row];
    const float* cos = cos_table + static_cast<long long>(position) * rotary_dim;
    const float* sin = sin_table + static_cast<long long>(position) * rotary_dim;
    __nv_bfloat16* target =
        queries + (static_cast<long long>(row) * index_heads + head) * index_dim;
    const int half = rotary_dim / 2;
    for (int slot = threadIdx.x; slot < half; slot += blockDim.x) {
        const float low = source[slot] * scale * (1.0f + q_layernorm[slot]);
        const float high =
            source[slot + half] * scale * (1.0f + q_layernorm[slot + half]);
        target[slot] = __float2bfloat16(
            rotate_low(low, high, cos[slot], sin[slot]));
        target[slot + half] = __float2bfloat16(
            rotate_high(low, high, cos[slot + half], sin[slot + half]));
    }
    for (int dim = rotary_dim + threadIdx.x; dim < index_dim;
         dim += blockDim.x) {
        target[dim] = __float2bfloat16(
            source[dim] * scale * (1.0f + q_layernorm[dim]));
    }
}

/// Appends this step's raw index keys to the per-sequence index key cache.
extern "C" __global__ void indexer_append_key(
    const float* projected,
    __nv_bfloat16* index_keys,
    const int* positions,
    int rows,
    int index_heads,
    int index_kv_heads,
    int index_dim,
    int arena_length) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int projected_width = (index_heads + index_kv_heads) * index_dim;
    const float* source =
        projected + static_cast<long long>(row) * projected_width +
        static_cast<long long>(index_heads) * index_dim;
    const int position = positions[row];
    if (position < 0 || position >= arena_length) {
        return;
    }
    __nv_bfloat16* target =
        index_keys + static_cast<long long>(position) * index_dim;
    for (int dim = threadIdx.x; dim < index_dim; dim += blockDim.x) {
        target[dim] = __float2bfloat16(source[dim]);
    }
}

/// Pools, normalises and rotates `blocks` completed blocks of index keys,
/// starting at `first_block`.
///
/// The mean is taken in fp32 and cast back to the cache dtype before the norm,
/// exactly as `key_groups.float().mean(dim=1).to(raw_keys.dtype)` does. The
/// result lands in the resident pooled array at its absolute block index, so a
/// block is pooled exactly once and its raw keys can then be dropped.
extern "C" __global__ void indexer_pool_blocks(
    const __nv_bfloat16* index_keys,
    const float* k_layernorm,
    __nv_bfloat16* pooled,
    const float* cos_table,
    const float* sin_table,
    int first_block,
    int blocks,
    int index_dim,
    int ratio,
    int rotary_dim,
    float eps) {
    __shared__ float scratch[warp_size];
    __shared__ float scale;

    // `blocks` is how many blocks this launch pools, starting at `first_block`:
    // the design pools a block once, when it completes, so the raw keys of
    // older tokens are never read again (and need not be kept).
    if (blockIdx.x >= blocks) {
        return;
    }
    const int block = first_block + blockIdx.x;
    const __nv_bfloat16* first =
        index_keys + static_cast<long long>(block) * ratio * index_dim;
    float squares = 0.0f;
    for (int dim = threadIdx.x; dim < index_dim; dim += blockDim.x) {
        float total = 0.0f;
        for (int row = 0; row < ratio; ++row) {
            total += __bfloat162float(first[static_cast<long long>(row) * index_dim + dim]);
        }
        const float mean = __bfloat162float(__float2bfloat16(total / ratio));
        squares += mean * mean;
    }
    const float sum_squares = block_sum(squares, scratch);
    const float mean_square = sum_squares / static_cast<float>(index_dim);
    scale = rsqrtf(mean_square + eps);
    __syncthreads();

    // The block key rotates at the position of its first token, in fp32, and is
    // stored once: the reference normes and rotates in fp32 as well.
    const int position = block * ratio;
    const float* cos = cos_table + static_cast<long long>(position) * rotary_dim;
    const float* sin = sin_table + static_cast<long long>(position) * rotary_dim;
    __nv_bfloat16* target = pooled + static_cast<long long>(block) * index_dim;
    const int half = rotary_dim / 2;
    for (int slot = threadIdx.x; slot < half; slot += blockDim.x) {
        const float low = pooled_normed(first, k_layernorm, scale, slot, index_dim, ratio);
        const float high =
            pooled_normed(first, k_layernorm, scale, slot + half, index_dim, ratio);
        target[slot] = __float2bfloat16(
            rotate_low(low, high, cos[slot], sin[slot]));
        target[slot + half] = __float2bfloat16(
            rotate_high(low, high, cos[slot + half], sin[slot + half]));
    }
    for (int dim = rotary_dim + threadIdx.x; dim < index_dim;
         dim += blockDim.x) {
        target[dim] = __float2bfloat16(
            pooled_normed(first, k_layernorm, scale, dim, index_dim, ratio));
    }
}

/// `score[row][block] = sum_heads relu(q_h . pooled_block) / sqrt(index_dim)`.
///
/// One CTA per (`scores_per_cta` blocks, row) and one warp per block: the warp
/// loads the block's key once, dots it against every index head (each query is
/// 1 KB and stays in L1), and lane 0 writes the score. Keeping the whole score
/// in one warp is what lets the loop run without a barrier; a shared-memory
/// reduction per block cost a barrier pair per 128 multiply-adds.
extern "C" __global__ void indexer_block_scores(
    const __nv_bfloat16* queries,
    const __nv_bfloat16* pooled,
    float* scores,
    int rows,
    int blocks,
    int first_block,
    int capacity,
    int index_heads,
    int index_dim,
    float scale) {
    const int row = blockIdx.y;
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const int warps = blockDim.x / warp_size;
    if (row >= rows || warps == 0) {
        return;
    }
    // The score row is the streaming group buffer: `blocks` blocks starting at
    // `first_block`, stored at their offset inside the group so the next group
    // can reuse the same `[rows, capacity]` allocation.
    const int first = blockIdx.x * scores_per_cta;
    const int limit = first + scores_per_cta;
    const int last = limit < blocks ? limit : blocks;
    for (int local = first + warp; local < last; local += warps) {
        const int block = first_block + local;
        const __nv_bfloat16* key = pooled + static_cast<long long>(block) * index_dim;
        float total = 0.0f;
        for (int head = 0; head < index_heads; ++head) {
            const __nv_bfloat16* query = queries +
                (static_cast<long long>(row) * index_heads + head) * index_dim;
            float partial = 0.0f;
            for (int dim = lane_dim(0, lane); dim < index_dim;
                 dim += dims_per_warp_pass) {
                if (dim + 1 < index_dim) {
                    const float2 a = load_bf16_pair(query, dim);
                    const float2 b = load_bf16_pair(key, dim);
                    partial += a.x * b.x + a.y * b.y;
                } else {
                    partial += __bfloat162float(query[dim]) *
                        __bfloat162float(key[dim]);
                }
            }
            const float dot = warp_sum(partial);
            total += dot > 0.0f ? dot : 0.0f;
        }
        const float scaled = total * scale;
        if (lane == 0) {
            scores[static_cast<long long>(row) * capacity + local] =
                scaled > 0.0f ? scaled : 0.0f;
        }
    }
}

/// Key ordering used by the selection: (score bits, block index), descending.
///
/// Every score is non-negative, so its bit pattern is monotone as an unsigned
/// integer and a radix select over the packed key orders exactly as
/// `torch.topk` does with ties broken by the higher block index.
__device__ __forceinline__ unsigned long long select_key(float score, int index) {
    const unsigned int bits = __float_as_uint(score);
    return (static_cast<unsigned long long>(bits) << 32) |
        static_cast<unsigned int>(index);
}

/// The bin where the descending suffix count first reaches `rank`, and how many
/// entries sit strictly above that bin.
///
/// Warp 0 does the whole search: `radix_bins / warp_size` bins per lane, a
/// warp-shuffle prefix over the lane sums, then at most that many comparisons
/// per lane. That replaces a 256-bin serial walk, which is what lets one query
/// scan 65 536 pooled blocks without the search cost growing with the bin
/// count. Exactly one lane matches, so the two stores cannot race.
__device__ __forceinline__ void select_bin(
    const int* histogram,
    int rank,
    int* chosen,
    int* above) {
    constexpr int bins_per_lane = radix_bins / warp_size;
    const int lane = threadIdx.x % warp_size;
    // Lane 0 owns the highest bins, so a lane-ascending prefix sum is a
    // descending suffix sum over the bins.
    const int first = (warp_size - 1 - lane) * bins_per_lane;
    int local = 0;
    for (int index = 0; index < bins_per_lane; ++index) {
        local += histogram[first + index];
    }
    int gathered = local;
    for (int offset = 1; offset < warp_size; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, gathered, offset);
        if (lane >= offset) {
            gathered += other;
        }
    }
    gathered -= local;
    // Only the lane whose own bins straddle the crossing may claim it: a lane
    // further down already has `gathered >= rank`, and a lane further up never
    // reaches `rank`. Exactly one lane qualifies, so the stores cannot race.
    int found = -1;
    if (gathered < rank) {
        for (int index = bins_per_lane - 1; index >= 0; --index) {
            const int bin = first + index;
            if (gathered + histogram[bin] >= rank) {
                found = bin;
                break;
            }
            gathered += histogram[bin];
        }
    }
    if (found >= 0) {
        *chosen = found;
        *above = gathered;
    }
}

/// One bitonic compare-exchange stage over a power-of-two key array.
///
/// `width` is the merge width of this stage (it also picks the direction of
/// each segment); `count` is the array length the stage sweeps.
__device__ __forceinline__ void bitonic_stage(
    unsigned long long* keys,
    int count,
    int width,
    int stride) {
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        const int partner = index ^ stride;
        if (partner <= index) {
            continue;
        }
        const bool descending = (index & width) == 0;
        const bool swap = descending ? keys[index] < keys[partner]
                                     : keys[index] > keys[partner];
        if (swap) {
            const unsigned long long held = keys[index];
            keys[index] = keys[partner];
            keys[partner] = held;
        }
    }
    __syncthreads();
}

/// Sorts `count` keys descending; `count` must be a power of two.
__device__ __forceinline__ void bitonic_sort_descending(
    unsigned long long* keys,
    int count) {
    for (int width = 2; width <= count; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            bitonic_stage(keys, count, width, stride);
        }
    }
}

/// Merges a bitonic sequence into descending order; `count` must be a power of
/// two. A bitonic input only needs the log2(count) merge stages, not the whole
/// sort: `indexer_group_topk` keeps both halves sorted, so the running partial
/// top-k costs ten stages instead of fifty-five.
__device__ __forceinline__ void bitonic_merge_descending(
    unsigned long long* keys,
    int count) {
    for (int stride = count >> 1; stride > 0; stride >>= 1) {
        bitonic_stage(keys, count, count, stride);
    }
}

/// Exact top-k of one score row, as sorted packed keys.
///
/// A radix select finds the exact `take`-th largest key, the survivors are
/// bitonic-sorted and returned in `candidates[0..take]` (score descending, ties
/// to the higher block index, as `torch.topk` orders them). `first_block` is the
/// absolute index of `row_scores[0]`, so a streaming group keeps the global
/// tie-break order. `histogram` is `radix_bins` ints, `candidates` holds
/// `select_topk_max` keys, and `cursor` is two ints; all three are caller
/// shared memory. Every thread of the block calls this, and it barriers.
__device__ __forceinline__ void select_topk_keys(
    const float* row_scores,
    int complete,
    int take,
    int first_block,
    int* histogram,
    unsigned long long* candidates,
    int* cursor) {
    const int tid = threadIdx.x;
    unsigned long long threshold = 0;
    int rank = take;
    int digit = 0;
    while (digit < radix_digits) {
        const int shift = (radix_digits - 1 - digit) * 8;
        const unsigned long long high_mask =
            (shift + 8 >= 64) ? 0ull : (~0ull << (shift + 8));
        if (tid < radix_bins) {
            histogram[tid] = 0;
        }
        __syncthreads();
        for (int block = tid; block < complete; block += blockDim.x) {
            const unsigned long long key =
                select_key(row_scores[block], first_block + block);
            if ((key & high_mask) == (threshold & high_mask)) {
                atomicAdd(&histogram[static_cast<int>((key >> shift) & 0xff)], 1);
            }
        }
        __syncthreads();
        if (tid < warp_size) {
            select_bin(histogram, rank, &cursor[0], &cursor[1]);
        }
        __syncthreads();
        threshold |= static_cast<unsigned long long>(cursor[0]) << shift;
        rank -= cursor[1];
        // Once the score digits are known, the index digits only matter when
        // the boundary score ties. Every tie being taken means the remaining
        // four passes cannot change the set, so they are skipped.
        const bool resolved =
            digit == score_digits - 1 && histogram[cursor[0]] == rank;
        digit += 1;
        if (resolved) {
            break;
        }
        __syncthreads();
    }

    if (tid == 0) {
        cursor[0] = 0;
    }
    __syncthreads();
    for (int block = tid; block < complete; block += blockDim.x) {
        const unsigned long long key =
            select_key(row_scores[block], first_block + block);
        if (key >= threshold) {
            const int slot = atomicAdd(&cursor[0], 1);
            if (slot < select_topk_max) {
                candidates[slot] = key + key_offset;
            }
        }
    }
    __syncthreads();

    int sort_size = 1;
    while (sort_size < take) {
        sort_size <<= 1;
    }
    for (int slot = take + tid; slot < sort_size; slot += blockDim.x) {
        candidates[slot] = 0;
    }
    __syncthreads();
    bitonic_sort_descending(candidates, sort_size);
}

/// Turns the block selection into the token list the attention kernels consume
/// (kernel 3 of the spec, steps 6 of the indexer).
///
/// One block per row over a full `[rows, blocks]` score buffer: the radix select
/// finds the exact `take`-th largest block key, the survivors are sorted,
/// expanded to their tokens, and the trailing incomplete block's raw indices are
/// appended. The streaming path uses `indexer_group_topk` instead, one bounded
/// score group at a time; this entry point is the whole-buffer reference both
/// paths are checked against.
///
/// Every stage is cooperative: the histogram scan is a warp search, the token
/// expansion is a strided write, and the four index digits are skipped when the
/// boundary score does not tie. A 65 536-block query therefore costs eight
/// strided passes over the score row (four in the common case) and no serial
/// walk whose length grows with the block count.
extern "C" __global__ void indexer_select(
    const float* scores,
    const int* positions,
    int* selected,
    int rows,
    int blocks,
    int slots,
    int ratio,
    int block_topk) {
    __shared__ int histogram[radix_bins];
    __shared__ unsigned long long candidates[select_topk_max];
    __shared__ int cursor[2];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const int visible = positions[row] + 1;
    const int complete = visible / ratio;
    const int tail = visible - complete * ratio;
    int* out = selected + static_cast<long long>(row) * slots;
    for (int slot = tid; slot < slots; slot += blockDim.x) {
        out[slot] = -1;
    }
    __syncthreads();

    const int take = complete < block_topk ? complete : block_topk;
    if (take > 0) {
        const float* row_scores = scores + static_cast<long long>(row) * blocks;
        select_topk_keys(row_scores, complete, take, 0, histogram, candidates, cursor);

        const int block_slots = take * ratio;
        for (int slot = tid; slot < block_slots; slot += blockDim.x) {
            const int item = slot / ratio;
            const unsigned long long key = candidates[item] - key_offset;
            const int block = static_cast<int>(key & 0xffffffffull);
            out[slot] = block * ratio + (slot % ratio);
        }
        for (int item = tid; item < tail; item += blockDim.x) {
            out[block_slots + item] = complete * ratio + item;
        }
    } else {
        for (int item = tid; item < tail; item += blockDim.x) {
            out[item] = item;
        }
    }
}

/// Merges one bounded score group into the running per-row partial top-k.
///
/// The streaming indexer holds at most `capacity` score columns per row; this
/// kernel selects the group's top `block_topk` blocks from those columns and
/// merges them with the partial keys the previous groups left behind. Both
/// sides are packed keys with the *absolute* block index in their low word, so
/// the merged order is score descending with ties to the higher block index —
/// the order `torch.topk` produces over the whole context.
///
/// The merge is exact: the global top-k of a union of groups is contained in
/// the union of each group's top-k, so merging one group at a time loses
/// nothing and the score buffer never grows with the context.
extern "C" __global__ void indexer_group_topk(
    const float* scores,
    const int* positions,
    unsigned long long* partial,
    int rows,
    int capacity,
    int first_block,
    int group_blocks,
    int ratio,
    int block_topk) {
    __shared__ int histogram[radix_bins];
    __shared__ unsigned long long group_keys[select_topk_max];
    __shared__ unsigned long long merged[merge_keys_max];
    __shared__ int cursor[2];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const int visible = positions[row] + 1;
    const int complete = visible / ratio;
    int visible_in_group = complete - first_block;
    if (visible_in_group < 0) {
        visible_in_group = 0;
    }
    if (visible_in_group > group_blocks) {
        visible_in_group = group_blocks;
    }
    const int take = visible_in_group < block_topk
        ? visible_in_group
        : block_topk;
    if (take > 0) {
        select_topk_keys(
            scores + static_cast<long long>(row) * capacity,
            visible_in_group,
            take,
            first_block,
            histogram,
            group_keys,
            cursor);
    }

    for (int slot = tid; slot < block_topk; slot += blockDim.x) {
        // The first group has no running partial yet; zeros are the padding key
        // that sorts below every real one, so no separate clear is needed.
        merged[slot] = first_block == 0
            ? 0ull
            : partial[static_cast<long long>(row) * block_topk + slot];
    }
    // The group's keys are already sorted descending, so lay them out
    // ascending at the tail and leave the padding zeros between the two runs.
    // The result decreases to the padding floor and then rises: a bitonic
    // sequence, which the log2 merge sorts for a tenth of the stages a full
    // sort would need.
    for (int slot = tid; slot < take; slot += blockDim.x) {
        merged[merge_keys_max - take + slot] = group_keys[take - 1 - slot];
    }
    for (int slot = block_topk + tid; slot < merge_keys_max - take;
         slot += blockDim.x) {
        merged[slot] = 0;
    }
    __syncthreads();
    bitonic_merge_descending(merged, merge_keys_max);

    for (int slot = tid; slot < block_topk; slot += blockDim.x) {
        partial[static_cast<long long>(row) * block_topk + slot] = merged[slot];
    }
}

/// Expands the merged partial top-k into the token list the attention kernels
/// consume: `ratio` tokens per selected block, then the incomplete tail.
extern "C" __global__ void indexer_expand(
    const unsigned long long* partial,
    const int* positions,
    int* selected,
    int rows,
    int slots,
    int ratio,
    int block_topk) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const int visible = positions[row] + 1;
    const int complete = visible / ratio;
    const int tail = visible - complete * ratio;
    const int take = complete < block_topk ? complete : block_topk;
    int* out = selected + static_cast<long long>(row) * slots;
    for (int slot = tid; slot < slots; slot += blockDim.x) {
        out[slot] = -1;
    }
    __syncthreads();

    const int block_slots = take * ratio;
    for (int slot = tid; slot < block_slots; slot += blockDim.x) {
        const int item = slot / ratio;
        const unsigned long long key =
            partial[static_cast<long long>(row) * block_topk + item] - key_offset;
        const int block = static_cast<int>(key & 0xffffffffull);
        out[slot] = block * ratio + (slot % ratio);
    }
    for (int item = tid; item < tail; item += blockDim.x) {
        out[block_slots + item] = complete * ratio + item;
    }
}

/// Streaming-softmax attention for one query head of one prefill row.
///
/// One warp per head: lane `lane` owns `pair_width` dims of every 64-dim pass,
/// keeps the query and the accumulator in registers, and folds each gathered
/// row in with the online softmax update. Rows are dense; the only per-row tests
/// are the causal one against the query's own position, a row past `horizon`
/// (the chunk's last query) counted as a violation, and the optional per-row
/// selection bitmap, which is what keeps a shared union gather exactly QSA.
__device__ __forceinline__ void attend_head_streaming(
    const __nv_bfloat16* query,
    const __nv_bfloat16* gathered_k,
    const __nv_bfloat16* gathered_v,
    const int* gather_index,
    __nv_bfloat16* out,
    int count,
    int position,
    int horizon,
    int kv_head,
    int kv_heads,
    int passes,
    int head_dim,
    int lane,
    float scale,
    const unsigned int* row_mask,
    int mask_words,
    int* violations) {
    float2 query_registers[lane_passes_max];
    float2 accumulator[lane_passes_max];
#pragma unroll
    for (int pass = 0; pass < lane_passes_max; ++pass) {
        query_registers[pass] = make_float2(0.0f, 0.0f);
        accumulator[pass] = make_float2(0.0f, 0.0f);
    }
#pragma unroll
    for (int pass = 0; pass < lane_passes_max; ++pass) {
        if (pass < passes) {
            const int dim = lane_dim(pass, lane);
            if (dim + 1 < head_dim) {
                query_registers[pass] = load_bf16_pair(query, dim);
            }
        }
    }

    const int row_stride = kv_heads * head_dim;
    const int head_offset = kv_head * head_dim;
    float running_max = negative_infinity;
    float running_sum = 0.0f;
    for (int slot = 0; slot < count; ++slot) {
        const int key_index = gather_index[slot];
        if (key_index < 0) {
            continue;
        }
        if (key_index > position) {
            if (key_index > horizon && violations != nullptr && lane == 0) {
                atomicAdd(violations, 1);
            }
            continue;
        }
        if (row_mask != nullptr &&
            ((row_mask[slot / 32] >> (slot % 32)) & 1u) == 0u) {
            continue;
        }
        const long long row = static_cast<long long>(slot) * row_stride + head_offset;
        float partial = 0.0f;
#pragma unroll
        for (int pass = 0; pass < lane_passes_max; ++pass) {
            if (pass < passes) {
                const int dim = lane_dim(pass, lane);
                if (dim + 1 < head_dim) {
                    const float2 key = load_bf16_pair(gathered_k + row, dim);
                    partial += query_registers[pass].x * key.x +
                        query_registers[pass].y * key.y;
                }
            }
        }
        const float score = warp_sum(partial) * scale;
        if (score > running_max) {
            const float correction = expf(running_max - score);
            running_sum *= correction;
#pragma unroll
            for (int pass = 0; pass < lane_passes_max; ++pass) {
                accumulator[pass].x *= correction;
                accumulator[pass].y *= correction;
            }
            running_max = score;
        }
        const float weight = expf(score - running_max);
        running_sum += weight;
#pragma unroll
        for (int pass = 0; pass < lane_passes_max; ++pass) {
            if (pass < passes) {
                const int dim = lane_dim(pass, lane);
                if (dim + 1 < head_dim) {
                    const float2 value = load_bf16_pair(gathered_v + row, dim);
                    accumulator[pass].x += weight * value.x;
                    accumulator[pass].y += weight * value.y;
                }
            }
        }
    }

    const float inverse = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
#pragma unroll
    for (int pass = 0; pass < lane_passes_max; ++pass) {
        if (pass < passes) {
            const int dim = lane_dim(pass, lane);
            if (dim + 1 < head_dim) {
                store_bf16_pair(
                    out,
                    dim,
                    make_float2(
                        accumulator[pass].x * inverse,
                        accumulator[pass].y * inverse));
            }
        }
    }
}

/// Dense prefill attention: `rows` queries over one gathered working set.
///
/// Grid: (ceil(heads / prefill_warps), rows); block: one warp per q head.
/// The working set is shared by the whole block, which is what the design's
/// union gather produces; each query is masked causally from its own position.
extern "C" __global__ void attn_prefill(
    const __nv_bfloat16* queries,
    const __nv_bfloat16* gathered_k,
    const __nv_bfloat16* gathered_v,
    const int* gather_index,
    const int* positions,
    __nv_bfloat16* out,
    int rows,
    int heads,
    int kv_heads,
    int head_dim,
    int count,
    int passes,
    float scale,
    const unsigned int* selection_mask,
    int* violations) {
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const int row = blockIdx.y;
    const int head = blockIdx.x * prefill_warps + warp;
    if (row >= rows || head >= heads || warp >= prefill_warps) {
        return;
    }
    const int group = heads / kv_heads;
    const long long query_offset =
        (static_cast<long long>(row) * heads + head) * head_dim;
    // Rows are ascending, so the last one is the chunk's causal horizon: a
    // selection may legitimately reach past an earlier row of the same chunk
    // (the union gather), never past the chunk.
    const int horizon = positions[rows - 1];
    const int mask_words = (count + 31) / 32;
    const unsigned int* row_mask = selection_mask != nullptr
        ? selection_mask + static_cast<long long>(row) * mask_words
        : nullptr;
    // Only the first head of a row tallies a violation: the counter then counts
    // gathered rows, not head instances of them.
    attend_head_streaming(
        queries + query_offset,
        gathered_k,
        gathered_v,
        gather_index,
        out + query_offset,
        count,
        positions[row],
        horizon,
        head / group,
        kv_heads,
        passes,
        head_dim,
        lane,
        scale,
        row_mask,
        mask_words,
        head == 0 ? violations : nullptr);
}

/// Tensor-core prefill attention: one CTA per (head, 16 query rows).
///
/// The arithmetic is `Q K^T` and `P V` for a 16-row m-tile; four warps split the
/// 64-key tile for the first and the output dims for the second, so both
/// products use every warp. Q and K are read straight from the gathered
/// working set in MMA fragment order (K's row-major layout *is* the col-major B
/// fragment), V is staged into shared memory and read with `ldmatrix.trans`,
/// and P is written to shared memory as bf16 between the two products.
///
/// Exact QSA needs a per-row mask over a shared union gather: `selection_mask`
/// is `[rows, ceil(count / 32)]` u32 bits, bit `slot` set when the row selected
/// the gathered row. A row past the query, a `-1` gather slot, a slot past
/// `count` and a cleared bit all score `-inf`; the union gather itself is
/// untouched. `selection_mask == nullptr` keeps the causal-only mask.
///
/// The head dim must be a multiple of 32 and at most 256; `prefill` falls back
/// to `attn_prefill` otherwise. `head_dim <= 256` keeps the V tile in the 48 KB
/// static shared-memory budget.
extern "C" __global__ void __launch_bounds__(mma_threads) attn_prefill_mma(
    const __nv_bfloat16* queries,
    const __nv_bfloat16* gathered_k,
    const __nv_bfloat16* gathered_v,
    const int* gather_index,
    const int* positions,
    const unsigned int* selection_mask,
    __nv_bfloat16* out,
    int rows,
    int heads,
    int kv_heads,
    int head_dim,
    int count,
    float scale,
    int* violations) {
    __shared__ float tile_scores[mma_rows][mma_keys_per_tile];
    __shared__ __half tile_p[mma_rows][mma_p_row_elements];
    __shared__ __half v_tile[mma_keys_per_tile][mma_v_row_elements];
    __shared__ int tile_key[mma_keys_per_tile];
    __shared__ unsigned int tile_mask[mma_rows][2];
    __shared__ float row_position[mma_rows];
    __shared__ float running_sum[mma_rows];
    __shared__ float row_correction[mma_rows];

    const int tid = threadIdx.x;
    const int lane = tid % warp_size;
    const int warp = tid / warp_size;
    const int head = blockIdx.x;
    const int row_base = blockIdx.y * mma_rows;
    if (head >= heads || row_base >= rows) {
        return;
    }
    const int group = heads / kv_heads;
    const int kv_head = head / group;
    const int key_stride = kv_heads * head_dim;
    const int head_offset = kv_head * head_dim;
    const int dim_tiles = (head_dim / mma_warps) / 8;
    const int dim_base = warp * (head_dim / mma_warps);
    const int k_steps = head_dim / mma_k;
    const int mask_words = (count + 31) / 32;

    for (int row = tid; row < mma_rows; row += blockDim.x) {
        const int global = row_base + row < rows ? row_base + row : rows - 1;
        row_position[row] = static_cast<float>(positions[global]);
        running_sum[row] = 0.0f;
        row_correction[row] = 0.0f;
    }
    // One CTA tallies the gathered rows past the chunk horizon, so the counter
    // counts gathered rows exactly once per launch, not once per head.
    if (head == 0 && row_base == 0 && violations != nullptr) {
        const int horizon = positions[rows - 1];
        for (int slot = tid; slot < count; slot += blockDim.x) {
            if (gather_index[slot] > horizon) {
                atomicAdd(violations, 1);
            }
        }
    }
    __syncthreads();

    float accumulator[mma_dim_tiles][4];
#pragma unroll
    for (int tile = 0; tile < mma_dim_tiles; ++tile) {
        accumulator[tile][0] = 0.0f;
        accumulator[tile][1] = 0.0f;
        accumulator[tile][2] = 0.0f;
        accumulator[tile][3] = 0.0f;
    }

    // The online softmax state of this thread's query row. It lives in a
    // register across key tiles, so no other thread ever reads a stale copy and
    // no barrier is needed between the tile maximum and the weight.
    float row_max = negative_infinity;
    float row_total = 0.0f;

    const int tiles = (count + mma_keys_per_tile - 1) / mma_keys_per_tile;
    for (int tile = 0; tile < tiles; ++tile) {
        const int tile_first = tile * mma_keys_per_tile;
        for (int index = tid; index < mma_keys_per_tile; index += blockDim.x) {
            const int slot = tile_first + index;
            tile_key[index] = slot < count ? gather_index[slot] : -1;
        }
        for (int index = tid; index < mma_rows * 2; index += blockDim.x) {
            const int local_row = index / 2;
            const int word = index % 2;
            const int global_row = row_base + local_row < rows
                ? row_base + local_row
                : rows - 1;
            const int slot_word = tile * 2 + word;
            unsigned int bits = 0xffffffffu;
            if (selection_mask != nullptr) {
                bits = slot_word < mask_words
                    ? selection_mask[static_cast<long long>(global_row) * mask_words + slot_word]
                    : 0u;
            }
            tile_mask[local_row][word] = bits;
        }
        const int chunks = head_dim / 8;
        for (int index = tid; index < mma_keys_per_tile * chunks;
             index += blockDim.x) {
            const int key_index = index / chunks;
            const int dim = (index % chunks) * 8;
            int slot = tile_first + key_index;
            if (slot >= count) {
                slot = count - 1;
            }
            if (slot < 0) {
                slot = 0;
            }
            const __nv_bfloat16* source = gathered_v +
                static_cast<long long>(slot) * key_stride + head_offset + dim;
            // fp16 for the PV tensor core; a bf16 value is exact in fp16 for
            // every normal magnitude, so this is a width change, not a
            // rounding of the operand.
            __half* target = &v_tile[key_index][dim];
            const __nv_bfloat162 low =
                *reinterpret_cast<const __nv_bfloat162*>(source);
            const __nv_bfloat162 high =
                *reinterpret_cast<const __nv_bfloat162*>(source + 2);
            const __nv_bfloat162 low2 =
                *reinterpret_cast<const __nv_bfloat162*>(source + 4);
            const __nv_bfloat162 high2 =
                *reinterpret_cast<const __nv_bfloat162*>(source + 6);
            *reinterpret_cast<__half2*>(target) = __floats2half2_rn(
                __low2float(low), __high2float(low));
            *reinterpret_cast<__half2*>(target + 2) = __floats2half2_rn(
                __low2float(high), __high2float(high));
            *reinterpret_cast<__half2*>(target + 4) = __floats2half2_rn(
                __low2float(low2), __high2float(low2));
            *reinterpret_cast<__half2*>(target + 6) = __floats2half2_rn(
                __low2float(high2), __high2float(high2));
        }

        // QK^T: warp `w` owns keys [w*16, w*16+16), two 8-key n-tiles of the
        // same 16 query rows. A fragments come from registers holding the
        // query rows; B fragments are K's row-major pairs, which is exactly the
        // col-major layout the instruction wants.
        float scores[2][4];
        scores[0][0] = 0.0f;
        scores[0][1] = 0.0f;
        scores[0][2] = 0.0f;
        scores[0][3] = 0.0f;
        scores[1][0] = 0.0f;
        scores[1][1] = 0.0f;
        scores[1][2] = 0.0f;
        scores[1][3] = 0.0f;
        const int query_row0 = row_base + lane / 4 < rows ? row_base + lane / 4 : rows - 1;
        const int query_row1 = row_base + lane / 4 + 8 < rows ? row_base + lane / 4 + 8 : rows - 1;
        const __nv_bfloat16* query0 =
            queries + (static_cast<long long>(query_row0) * heads + head) * head_dim;
        const __nv_bfloat16* query1 =
            queries + (static_cast<long long>(query_row1) * heads + head) * head_dim;
        for (int step = 0; step < k_steps; ++step) {
            const int offset = step * mma_k + 2 * (lane % 4);
            unsigned int a[4];
            a[0] = *reinterpret_cast<const unsigned int*>(query0 + offset);
            a[1] = *reinterpret_cast<const unsigned int*>(query1 + offset);
            a[2] = *reinterpret_cast<const unsigned int*>(query0 + offset + 8);
            a[3] = *reinterpret_cast<const unsigned int*>(query1 + offset + 8);
#pragma unroll
            for (int n_tile = 0; n_tile < 2; ++n_tile) {
                // B's fragment puts key `n = lane / 4` in this lane, so the
                // key row differs per lane and only the contraction offset is
                // uniform across the quad.
                int slot = tile_first + warp * mma_keys_per_warp +
                    n_tile * 8 + lane / 4;
                if (slot >= count) {
                    slot = count - 1;
                }
                if (slot < 0) {
                    slot = 0;
                }
                const __nv_bfloat16* key_row = gathered_k +
                    static_cast<long long>(slot) * key_stride + head_offset;
                unsigned int b[2];
                b[0] = *reinterpret_cast<const unsigned int*>(key_row + offset);
                b[1] = *reinterpret_cast<const unsigned int*>(key_row + offset + 8);
                mma_bf16(scores[n_tile], a, b);
            }
        }
#pragma unroll
        for (int n_tile = 0; n_tile < 2; ++n_tile) {
            const int column = warp * mma_keys_per_warp + n_tile * 8 + 2 * (lane % 4);
            tile_scores[lane / 4][column] = scores[n_tile][0] * scale;
            tile_scores[lane / 4][column + 1] = scores[n_tile][1] * scale;
            tile_scores[lane / 4 + 8][column] = scores[n_tile][2] * scale;
            tile_scores[lane / 4 + 8][column + 1] = scores[n_tile][3] * scale;
        }
        __syncthreads();

        // Softmax over the 64-key tile: eight threads per query row, eight keys
        // each, a half-warp reduction over the eight lanes of the row.
        const int soft_row = tid / mma_softmax_threads;
        const int soft_part = tid % mma_softmax_threads;
        float local_max = negative_infinity;
#pragma unroll
        for (int item = 0; item < mma_softmax_keys; ++item) {
            const int index = soft_part * mma_softmax_keys + item;
            const int slot = tile_first + index;
            const int key = tile_key[index];
            const bool selected = slot < count && key >= 0 &&
                static_cast<float>(key) <= row_position[soft_row] &&
                ((tile_mask[soft_row][index / 32] >> (index % 32)) & 1u) != 0u;
            const float score = selected
                ? tile_scores[soft_row][index]
                : negative_infinity;
            tile_scores[soft_row][index] = score;
            local_max = fmaxf(local_max, score);
        }
        for (int offset = 1; offset < mma_softmax_threads; offset <<= 1) {
            local_max = fmaxf(
                local_max,
                __shfl_xor_sync(0xffffffffu, local_max, offset));
        }
        // Every lane of the row now holds the tile maximum. The online update
        // is per-thread state (`row_max`), and one lane publishes the
        // correction the PV lanes rescale by and the running sum the final
        // normalisation divides by; the single barrier below covers both
        // stores and the P tile.
        const float combined = fmaxf(row_max, local_max);
        const float correction = row_max > negative_infinity
            ? expf(row_max - combined)
            : 0.0f;
        row_max = combined;

        float local_sum = 0.0f;
#pragma unroll
        for (int item = 0; item < mma_softmax_keys; ++item) {
            const int index = soft_part * mma_softmax_keys + item;
            const float score = tile_scores[soft_row][index];
            const float weight = score > negative_infinity
                ? expf(score - combined)
                : 0.0f;
            tile_p[soft_row][index] = __float2half(weight);
            local_sum += weight;
        }
        for (int offset = 1; offset < mma_softmax_threads; offset <<= 1) {
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, offset);
        }
        row_total = row_total * correction + local_sum;
        if (soft_part == 0) {
            row_correction[soft_row] = correction;
            running_sum[soft_row] = row_total;
        }
        __syncthreads();

        const float correction0 = row_correction[lane / 4];
        const float correction1 = row_correction[lane / 4 + 8];
#pragma unroll
        for (int tile_index = 0; tile_index < mma_dim_tiles; ++tile_index) {
            if (tile_index < dim_tiles) {
                accumulator[tile_index][0] *= correction0;
                accumulator[tile_index][1] *= correction0;
                accumulator[tile_index][2] *= correction1;
                accumulator[tile_index][3] *= correction1;
            }
        }

        // P V: warp `w` accumulates output dims [w*64, w*64+64) over the same
        // 16 rows, four 16-key contraction steps. P is the shared bf16 tile; V
        // is the shared row-major tile read transposed.
        for (int step = 0; step < mma_key_steps_pv; ++step) {
            const int p_row = lane & 15;
            const int p_column = step * mma_k + (lane >= 16 ? 8 : 0);
            unsigned int a[4];
            ldmatrix_x4(a, &tile_p[p_row][p_column]);
#pragma unroll
            for (int tile_index = 0; tile_index < mma_dim_tiles; ++tile_index) {
                if (tile_index < dim_tiles) {
                    // Only lanes 0-15 supply addresses to `ldmatrix.x2`; they
                    // are the sixteen keys of this contraction step. Clamping
                    // the rest to 0 keeps every address valid.
                    const int v_key = step * mma_k + (lane & 15);
                    const int v_dim = dim_base + tile_index * 8;
                    unsigned int b[2];
                    ldmatrix_x2_trans(b, &v_tile[v_key][v_dim]);
                    mma_f16(accumulator[tile_index], a, b);
                }
            }
        }
        __syncthreads();
    }

    const float sum0 = running_sum[lane / 4];
    const float sum1 = running_sum[lane / 4 + 8];
    const float inverse0 = sum0 > 0.0f ? 1.0f / sum0 : 0.0f;
    const float inverse1 = sum1 > 0.0f ? 1.0f / sum1 : 0.0f;
    const int row0 = row_base + lane / 4;
    const int row1 = row0 + 8;
#pragma unroll
    for (int tile_index = 0; tile_index < mma_dim_tiles; ++tile_index) {
        if (tile_index < dim_tiles) {
            const int dim = dim_base + tile_index * 8 + 2 * (lane % 4);
            if (row0 < rows) {
                __nv_bfloat16* target =
                    out + (static_cast<long long>(row0) * heads + head) * head_dim;
                store_bf16_pair(
                    target,
                    dim,
                    make_float2(
                        accumulator[tile_index][0] * inverse0,
                        accumulator[tile_index][1] * inverse0));
            }
            if (row1 < rows) {
                __nv_bfloat16* target =
                    out + (static_cast<long long>(row1) * heads + head) * head_dim;
                store_bf16_pair(
                    target,
                    dim,
                    make_float2(
                        accumulator[tile_index][2] * inverse1,
                        accumulator[tile_index][3] * inverse1));
            }
        }
    }
}

/// Dense single-token decode: one query head per block, `decode_warps` wide.
///
/// Two passes over the dense working set: scores into shared memory (one warp
/// per residue class of rows), a block-wide softmax, then the value
/// accumulation split across warps and reduced per dim. At M=1 this keeps all
/// 256 threads of the block busy instead of one warp per head, which is what
/// the per-head latency target needs.
extern "C" __global__ void attn_decode(
    const __nv_bfloat16* query,
    const __nv_bfloat16* gathered_k,
    const __nv_bfloat16* gathered_v,
    const int* gather_index,
    int count,
    int position,
    __nv_bfloat16* out,
    int heads,
    int kv_heads,
    int head_dim,
    int passes,
    float scale,
    int* violations) {
    __shared__ float probabilities[selected_slots_max];
    __shared__ float partials[decode_warps][head_dim_max];

    const int head = blockIdx.x;
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    if (head >= heads || count > selected_slots_max) {
        return;
    }

    const int group = heads / kv_heads;
    const int row_stride = kv_heads * head_dim;
    const int head_offset = (head / group) * head_dim;
    const __nv_bfloat16* query_row =
        query + static_cast<long long>(head) * head_dim;
    float2 query_registers[lane_passes_max];
#pragma unroll
    for (int pass = 0; pass < lane_passes_max; ++pass) {
        query_registers[pass] = make_float2(0.0f, 0.0f);
    }
#pragma unroll
    for (int pass = 0; pass < lane_passes_max; ++pass) {
        if (pass < passes) {
            const int dim = lane_dim(pass, lane);
            if (dim + 1 < head_dim) {
                query_registers[pass] = load_bf16_pair(query_row, dim);
            }
        }
    }

    // Pass one: one warp per residue class of the dense working set. A row past
    // the query is scored as masked instead of skipped, so the softmax below
    // stays a single dense pass.
    int dropped = 0;
    for (int slot = warp; slot < count; slot += decode_warps) {
        const int key_index = gather_index[slot];
        const bool masked = key_index < 0 || key_index > position;
        if (key_index > position && lane == 0) {
            dropped += 1;
        }
        float partial = 0.0f;
        const long long row = static_cast<long long>(slot) * row_stride + head_offset;
#pragma unroll
        for (int pass = 0; pass < lane_passes_max; ++pass) {
            if (pass < passes) {
                const int dim = lane_dim(pass, lane);
                if (dim + 1 < head_dim) {
                    const float2 key = load_bf16_pair(gathered_k + row, dim);
                    partial += query_registers[pass].x * key.x +
                        query_registers[pass].y * key.y;
                }
            }
        }
        const float score = warp_sum(partial) * scale;
        if (lane == 0) {
            probabilities[slot] = masked ? negative_infinity : score;
        }
    }
    // Every warp counts its own residue class, so the block total is the number
    // of gathered rows past the query position; one block (head 0) reports it,
    // which keeps the counter independent of the head count.
    const float dropped_total = block_sum(
        static_cast<float>(dropped), partials[0]);
    if (blockIdx.x == 0 && threadIdx.x == 0 && violations != nullptr &&
        dropped_total > 0.0f) {
        atomicAdd(violations, static_cast<int>(dropped_total));
    }
    __syncthreads();

    // Block-wide softmax over the dense scores.
    float largest = negative_infinity;
    for (int slot = threadIdx.x; slot < count; slot += blockDim.x) {
        largest = fmaxf(largest, probabilities[slot]);
    }
    largest = block_max(largest, partials[0]);
    float total = 0.0f;
    if (count > 0) {
        for (int slot = threadIdx.x; slot < count; slot += blockDim.x) {
            const float weight = expf(probabilities[slot] - largest);
            probabilities[slot] = weight;
            total += weight;
        }
    }
    total = block_sum(total, partials[0]);
    __syncthreads();

    // Pass two: warp `w` accumulates every `decode_warps`-th row, then the
    // per-warp accumulators are summed per dim.
    for (int index = threadIdx.x; index < decode_warps * head_dim;
         index += blockDim.x) {
        partials[index / head_dim][index % head_dim] = 0.0f;
    }
    __syncthreads();
    if (count > 0 && total > 0.0f) {
        for (int slot = warp; slot < count; slot += decode_warps) {
            const float weight = probabilities[slot] / total;
            if (weight == 0.0f) {
                continue;
            }
            const long long row =
                static_cast<long long>(slot) * row_stride + head_offset;
            for (int dim = lane_dim(0, lane); dim < head_dim;
                 dim += dims_per_warp_pass) {
                if (dim + 1 < head_dim) {
                    const float2 value = load_bf16_pair(gathered_v + row, dim);
                    partials[warp][dim] += weight * value.x;
                    partials[warp][dim + 1] += weight * value.y;
                }
            }
        }
    }
    __syncthreads();

    __nv_bfloat16* out_row = out + static_cast<long long>(head) * head_dim;
    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        float sum = 0.0f;
        for (int index = 0; index < decode_warps; ++index) {
            sum += partials[index][dim];
        }
        out_row[dim] = __float2bfloat16(sum);
    }
}
