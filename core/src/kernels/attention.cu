// Indexed (QSA) attention: sparse prefill attention, single-token decode, the
// indexer's block selection, and the partial mRoPE tables that feed them.
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
//   index keys  bf16 [arena_length, index_dim]  raw (un-normalised) token keys
//   pooled      bf16 [blocks, index_dim]         one pooled key per complete block
//   selected    int32 [rows, slots], -1 padded, slots = budget + ratio - 1
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

/// Blocks one CTA scores: the per-block work is 128 multiply-adds, so a CTA per
/// block leaves the GPU scheduling 65 536 near-empty CTAs at the context limit.
constexpr int scores_per_cta = 16;

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
    const int first = blockIdx.x * scores_per_cta;
    const int limit = first + scores_per_cta;
    const int last = limit < blocks ? limit : blocks;
    for (int block = first + warp; block < last; block += warps) {
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
            scores[static_cast<long long>(row) * blocks + block] =
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

/// Turns the block selection into the token list the attention kernels consume
/// (kernel 3 of the spec, steps 6 of the indexer).
///
/// One block per row: a radix select finds the exact `take`-th largest block
/// key, the survivors are bitonic-sorted, expanded to their tokens, and the
/// trailing incomplete block's raw indices are appended. Ties in the score are
/// broken by the higher block index, matching the reference.
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
                const unsigned long long key = select_key(row_scores[block], block);
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
            // Once the score digits are known, the index digits only matter
            // when the boundary score ties. Every tie being taken means the
            // remaining four passes cannot change the set, so they are skipped.
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
            const unsigned long long key = select_key(row_scores[block], block);
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

/// Streaming-softmax attention for one query head of one prefill row.
///
/// One warp per head: lane `lane` owns `pair_width` dims of every 64-dim pass,
/// keeps the query and the accumulator in registers, and folds each gathered
/// row in with the online softmax update. Rows are dense; the only per-row test
/// is the causal one against the query's own position, and a row past
/// `horizon` (the chunk's last query) is also counted as a violation.
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
        head == 0 ? violations : nullptr);
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
