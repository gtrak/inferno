// Elementwise and structural device code: RMSNorm (grouped, gated), partial
// rotary embeddings with mRoPE recomposition, hyper-connection mixing, the
// silu/sigmoid/softmax/argmax helpers and the on-device sampler.
//
// Every buffer is f32 unless a symbol says otherwise. The one reduced-precision
// entry point is `ew_rmsnorm_grouped_bf16`, because the engine keeps the
// residual stream in bf16 across the norm while the norm itself accumulates in
// f32; the reference upcasts the same way (`Qwen4ExpTextRMSNorm._norm`).
//
// The oracle for every kernel here is `core/src/ref/`: `ops.rmsnorm`,
// `ops.rmsnorm_gated`, `ops.silu`, `ops.sigmoid`, `ops.softmax`,
// `rope.cos_sin` + `rope.apply`, and `hc.forward`. The fixtures are handed to
// both sides unchanged, so a difference measured by `elementwise.zig` is a
// kernel difference, never an input-format difference.
//
// Nothing here sorts. The sampler needs a *definition* of "the largest k" and
// "the nucleus" that is exact under ties without an ordering, so both are
// expressed as thresholds on the score, found by bisection over the ordered f32
// bit pattern:
//
//   top-k  keeps every score >= the k-th largest score (ties included, so a
//          tie at the boundary retains more than k candidates);
//   top-p  keeps every score >= the smallest threshold whose retained mass
//          reaches `top_p` (ties included, so the retained mass is >= top_p);
//   min-p  keeps every score whose probability is >= `min_p` of the maximum.
//
// All three are value thresholds, so the retained set is a pure function of the
// logits and no ordering (stable or otherwise) can change it. The draw itself
// is Gumbel-max over the retained set: one uniform per candidate, `argmax` of
// `score + gumbel`, which samples the categorical distribution exactly and
// needs no cumulative array, no scan and no second launch.

#if !defined(__CUDA_ARCH_SPECIFIC__)
#error "inferno kernels must be compiled with -arch=sm_120a"
#endif

#include <cassert>
#include <cstdint>

#include <cuda_bf16.h>

// Threads per block of the row and elementwise kernels.
static constexpr int ew_threads = 256;

// Warp width, and the warps one block holds.
static constexpr int ew_warp = 32;
static constexpr int ew_warps = ew_threads / ew_warp;

// Floats a block-wide reduction needs: one partial per warp, plus the total.
static constexpr int ew_scratch = ew_warps + 1;

// `activation` codes, matching the order of `Activation` in `ref/ops.zig`.
static constexpr int activation_none = 0;
static constexpr int activation_silu = 1;
static constexpr int activation_sigmoid = 2;

// Threads per block of the RoPE kernels: one rotation touches `rotated / 2`
// dims, which is 32 at the production head size.
static constexpr int rope_threads = 32;

// Threads per block of the sampler. One block owns the whole vocabulary, so
// every reduction is a block reduction and the draw needs no second launch.
static constexpr int sampler_threads = 256;

// Block reductions the sampler runs: the total mass, the top-k count, the
// nucleus mass and the draw's retained mass. Each gets its own scratch slice.
static constexpr int sampler_scratch_slots = 4;

// Bisection steps over a value range: the ordered-bit space is 32 bits wide, so
// 32 halvings resolve the largest threshold code exactly.
static constexpr int ew_bisect_steps = 32;

// `x / (1 + exp(-x))` or `1 / (1 + exp(-x))`; both in f32, like the reference.
__device__ __forceinline__ float ew_activate(float value, int activation) {
    if (activation == activation_silu) {
        return value / (1.0f + expf(-value));
    }
    if (activation == activation_sigmoid) {
        return 1.0f / (1.0f + expf(-value));
    }
    assert(activation == activation_none);
    return value;
}

__device__ __forceinline__ unsigned int ew_min_code(unsigned int a,
                                                    unsigned int b) {
    return (a < b) ? a : b;
}

__device__ __forceinline__ unsigned int ew_max_code(unsigned int a,
                                                    unsigned int b) {
    return (a > b) ? a : b;
}

__device__ __forceinline__ int ew_min_int(int a, int b) {
    return (a < b) ? a : b;
}

__device__ __forceinline__ float ew_sigmoid(float value) {
    assert(value == value);
    const float result = 1.0f / (1.0f + expf(-value));
    assert(result >= 0.0f);
    assert(result <= 1.0f);
    return result;
}

__device__ __forceinline__ unsigned int ew_rotate_left(unsigned int value,
                                                       int shift) {
    assert(shift > 0);
    assert(shift < 32);
    return (value << shift) | (value >> (32 - shift));
}

// Sum held by the 32 lanes of one warp.
__device__ __forceinline__ float ew_warp_sum(float value) {
    for (int offset = ew_warp / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// Block-wide sum. Every thread of the block must reach both barriers, so this
// is only valid when the whole block executes it unconditionally.
__device__ __forceinline__ float ew_block_sum(float value, float *scratch) {
    const int lanes = static_cast<int>(blockDim.x);
    const int lane = static_cast<int>(threadIdx.x) % ew_warp;
    const int warp = static_cast<int>(threadIdx.x) / ew_warp;
    const int warps = lanes / ew_warp;
    assert(lanes % ew_warp == 0);
    assert(warps <= ew_warps);

    value = ew_warp_sum(value);
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    value = (warp == 0 && lane < warps) ? scratch[lane] : 0.0f;
    if (warp == 0) {
        value = ew_warp_sum(value);
        if (lane == 0) {
            scratch[warps] = value;
        }
    }
    __syncthreads();
    return scratch[warps];
}

// True when `(value, index)` outranks `(other_value, other_index)`: the larger
// value wins and an exact tie goes to the lower index, which makes the winner a
// function of the input alone.
__device__ __forceinline__ bool ew_outranks(float value, int index,
                                            float other_value,
                                            int other_index) {
    if (value > other_value) {
        return true;
    } else {
        if (value < other_value) {
            return false;
        } else {
            return index < other_index;
        }
    }
}

// Block-wide argmax over `(value, index)` pairs. `index_shared` and
// `value_shared` hold one slot per thread of the block.
__device__ __forceinline__ void ew_block_argmax(float value, int index,
                                               float *value_shared,
                                               int *index_shared,
                                               float *out_value,
                                               int *out_index) {
    const int lanes = static_cast<int>(blockDim.x);
    // The winner is read from slot 0 after the last barrier, so a second call
    // must not overwrite slot 0 before every thread has read it: the leading
    // barrier orders this call's writes after the previous call's reads.
    __syncthreads();
    value_shared[threadIdx.x] = value;
    index_shared[threadIdx.x] = index;
    __syncthreads();
    for (int stride = lanes / 2; stride >= 1; stride /= 2) {
        if (static_cast<int>(threadIdx.x) < stride) {
            const int other = static_cast<int>(threadIdx.x) + stride;
            const bool wins = ew_outranks(value_shared[other],
                                          index_shared[other],
                                          value_shared[threadIdx.x],
                                          index_shared[threadIdx.x]);
            if (wins) {
                value_shared[threadIdx.x] = value_shared[other];
                index_shared[threadIdx.x] = index_shared[other];
            }
        }
        __syncthreads();
    }
    *out_value = value_shared[0];
    *out_index = index_shared[0];
}

// `x * rsqrt(mean(x^2) + eps) * (1 + w)` over groups of `group` columns.
//
// This is `ops.rmsnorm` with `group_size` set: the HC stream normalises each
// 2560-dim group of the 10240-dim vector, and the weight is full width.
extern "C" __global__ void __launch_bounds__(ew_threads)
ew_rmsnorm_grouped(const float *__restrict__ x,
                   const float *__restrict__ weight,
                   float *__restrict__ out, int rows, int columns, int group,
                   float eps) {
    extern __shared__ float shared[];
    assert(group > 0);
    assert(columns % group == 0);

    const int groups = columns / group;
    const int block = static_cast<int>(blockIdx.x);
    const int row = block / groups;
    const int within = block - row * groups;
    assert(row < rows);

    const long start =
        static_cast<long>(row) * columns + static_cast<long>(within) * group;
    const long weight_start = static_cast<long>(within) * group;
    float sum_squares = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < group; i += ew_threads) {
        const float value = x[start + i];
        sum_squares += value * value;
    }
    const float total = ew_block_sum(sum_squares, shared);
    const float mean = total / static_cast<float>(group);
    const float scale = 1.0f / sqrtf(mean + eps);
    for (int i = static_cast<int>(threadIdx.x); i < group; i += ew_threads) {
        const long index = start + i;
        out[index] = x[index] * scale * (1.0f + weight[weight_start + i]);
    }
}

// The same norm with bf16 operands and a bf16 result.
//
// The reduction is f32 (the reference upcasts before `pow(2).mean()`), the
// weight stays f32 because the loader hands the norm a dequantised vector, and
// the single rounding is the store, which is what
// `elementwise.zig`'s half-ulp gate measures.
extern "C" __global__ void __launch_bounds__(ew_threads)
ew_rmsnorm_grouped_bf16(const __nv_bfloat16 *__restrict__ x,
                        const float *__restrict__ weight,
                        __nv_bfloat16 *__restrict__ out, int rows, int columns,
                        int group, float eps) {
    extern __shared__ float shared[];
    assert(group > 0);
    assert(columns % group == 0);

    const int groups = columns / group;
    const int block = static_cast<int>(blockIdx.x);
    const int row = block / groups;
    const int within = block - row * groups;
    assert(row < rows);

    const long start =
        static_cast<long>(row) * columns + static_cast<long>(within) * group;
    const long weight_start = static_cast<long>(within) * group;
    float sum_squares = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < group; i += ew_threads) {
        const float value = __bfloat162float(x[start + i]);
        sum_squares += value * value;
    }
    const float total = ew_block_sum(sum_squares, shared);
    const float mean = total / static_cast<float>(group);
    const float scale = 1.0f / sqrtf(mean + eps);
    for (int i = static_cast<int>(threadIdx.x); i < group; i += ew_threads) {
        const long index = start + i;
        const float value = __bfloat162float(x[index]);
        const float scaled = value * scale * (1.0f + weight[weight_start + i]);
        out[index] = __float2bfloat16(scaled);
    }
}

// `(w * (x * rsqrt(mean(x^2) + eps))) * act(z)` over one whole row.
//
// The product order is `ops.rmsnorm_gated`'s: the weight multiplies the
// normalised value first, then the gate. The gate activation is the
// checkpoint's `output_gate_type`.
extern "C" __global__ void __launch_bounds__(ew_threads)
ew_rmsnorm_gated(const float *__restrict__ x, const float *__restrict__ z,
                 const float *__restrict__ weight, float *__restrict__ out,
                 int rows, int columns, float eps, int activation) {
    extern __shared__ float shared[];
    assert(activation >= activation_none);
    assert(activation <= activation_sigmoid);

    const int row = static_cast<int>(blockIdx.x);
    assert(row < rows);

    const long start = static_cast<long>(row) * columns;
    float sum_squares = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < columns; i += ew_threads) {
        const float value = x[start + i];
        sum_squares += value * value;
    }
    const float total = ew_block_sum(sum_squares, shared);
    const float mean = total / static_cast<float>(columns);
    const float scale = 1.0f / sqrtf(mean + eps);
    for (int i = static_cast<int>(threadIdx.x); i < columns; i += ew_threads) {
        const long index = start + i;
        const float gated = ew_activate(z[index], activation);
        out[index] = (weight[i] * (x[index] * scale)) * gated;
    }
}

// Which position grid drives frequency `index` of the recomposition.
//
// `ref/rope.zig` `mrope_grid` defines this: lane 1 of every triple reads the
// height grid while the lane is inside `section_h * 3`, lane 2 reads the width
// grid, and every other lane keeps the temporal grid. The width grid is applied
// last, so it wins any overlap.
__device__ __forceinline__ int ew_rope_grid(int index, int section_h,
                                           int section_w) {
    assert(index >= 0);
    assert(section_h >= 0);

    int grid = 0;
    if (index % 3 == 1 && index < section_h * 3) {
        grid = 1;
    }
    if (index % 3 == 2 && index < section_w * 3) {
        grid = 2;
    }
    assert(grid <= 2);
    return grid;
}

// Builds the `[tokens][2 * frequency_count]` cos/sin tables.
//
// The frequency table itself is a host-side constant of the model (computed
// once from `rope_theta` by `rope.inv_freq_init`), so it arrives already built:
// re-deriving it on device would spend a `powf` per lane and could not be made
// bit-identical to the oracle anyway.
extern "C" __global__ void __launch_bounds__(ew_threads)
ew_rope_build_tables(const int *__restrict__ positions_t,
                     const int *__restrict__ positions_h,
                     const int *__restrict__ positions_w, int tokens,
                     const float *__restrict__ frequencies,
                     int frequency_count, int section_h, int section_w,
                     float *__restrict__ cos_out, float *__restrict__ sin_out) {
    assert(frequency_count > 0);
    assert(tokens > 0);

    const int token = static_cast<int>(blockIdx.x);
    assert(token < tokens);
    const long row = static_cast<long>(token) * 2 * frequency_count;
    for (int i = static_cast<int>(threadIdx.x); i < frequency_count;
         i += ew_threads) {
        const int grid = ew_rope_grid(i, section_h, section_w);
        const int *positions = positions_t;
        if (grid == 1) {
            positions = positions_h;
        }
        if (grid == 2) {
            positions = positions_w;
        }
        const float angle =
            static_cast<float>(positions[token]) * frequencies[i];
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        cos_out[row + i] = cosine;
        cos_out[row + i + frequency_count] = cosine;
        sin_out[row + i] = sine;
        sin_out[row + i + frequency_count] = sine;
    }
}

// Rotates the first `rotated` dims of one head in place.
//
// `rope.apply` is `rotate_half`: the block is split in two halves and the low
// half reads `cos[j]`/`sin[j]` while the high half reads the mirrored lanes.
// The dims past `rotated` are never touched, which is what makes the untouched
// tail bit-identical to the input rather than merely close.
__device__ __forceinline__ void ew_rope_rotate_head(float *head,
                                                    const float *cos_row,
                                                    const float *sin_row,
                                                    int rotated) {
    assert(rotated > 0);
    assert(rotated % 2 == 0);

    const int half = rotated / 2;
    for (int j = static_cast<int>(threadIdx.x); j < half;
         j += static_cast<int>(blockDim.x)) {
        const float low = head[j];
        const float high = head[j + half];
        head[j] = low * cos_row[j] - high * sin_row[j];
        head[j + half] = high * cos_row[j + half] + low * sin_row[j + half];
    }
}

// Rotates every head of a `[tokens][heads][head_dim]` tensor in place.
__device__ __forceinline__ void ew_rope_apply_tensor(
    float *data, const float *cos, const float *sin, int tokens, int heads,
    int head_dim, int rotated) {
    assert(rotated <= head_dim);
    assert(heads > 0);

    // Token on the x axis: `gridDim.y` stops at 65535, and a prefill chunk can
    // be longer than that at this model's 262144-token context.
    const int token = static_cast<int>(blockIdx.x);
    const int head_index = static_cast<int>(blockIdx.y);
    assert(token < tokens);
    assert(head_index < heads);
    const long offset =
        (static_cast<long>(token) * heads + head_index) * head_dim;
    const long table = static_cast<long>(token) * rotated;
    ew_rope_rotate_head(data + offset, cos + table, sin + table, rotated);
}

// Query and key in one launch: the attention layers rotate both.
extern "C" __global__ void __launch_bounds__(rope_threads)
ew_rope_apply_qk(float *__restrict__ q, float *__restrict__ k,
                 const float *__restrict__ cos, const float *__restrict__ sin,
                 int tokens, int heads, int head_dim, int rotated) {
    assert(k != nullptr);
    assert(rotated > 0);
    ew_rope_apply_tensor(q, cos, sin, tokens, heads, head_dim, rotated);
    ew_rope_apply_tensor(k, cos, sin, tokens, heads, head_dim, rotated);
}

// The query-only variant the indexer needs: its keys are pooled and unrotated.
extern "C" __global__ void __launch_bounds__(rope_threads)
ew_rope_apply_q(float *__restrict__ q, const float *__restrict__ cos,
                const float *__restrict__ sin, int tokens, int heads,
                int head_dim, int rotated) {
    assert(q != nullptr);
    assert(rotated > 0);
    ew_rope_apply_tensor(q, cos, sin, tokens, heads, head_dim, rotated);
}

// One hyper-connection block for every token: the grouped norm of the
// `streams * hidden` stream, the low-rank sigmoid mix, the weighted mean of the
// streams, and — when `inject_weight` is given — the per-stream injection
// weights. `hc.forward` with `use_combine = false` is the same launch with a
// null `inject_weight`, which is why the final mixer needs no second kernel.
//
// Shared memory holds the normalised row, the sigmoid mix weights it produces
// and the low-rank vector between them; at the production shape that is
// 2 * 10240 + 320 floats = 83 KiB, so the launcher opts in to >48 KiB.
extern "C" __global__ void __launch_bounds__(ew_threads)
ew_hyper_mix(const float *__restrict__ hyper_input,
             const float *__restrict__ norm_weight,
             const float *__restrict__ mix_down,
             const float *__restrict__ mix_up,
             const float *__restrict__ inject_weight, float *__restrict__ mixed,
             float *__restrict__ injection, int tokens, int hidden, int streams,
             int lowrank, float eps) {
    extern __shared__ float shared[];
    assert(hidden > 0);
    assert(streams > 0);

    const int hyper = streams * hidden;
    float *normed = shared;
    float *mix_weight = normed + hyper;
    float *lowrank_values = mix_weight + hyper;
    float *scratch = lowrank_values + lowrank;
    const int lane = static_cast<int>(threadIdx.x) % ew_warp;
    const int warp = static_cast<int>(threadIdx.x) / ew_warp;
    const float divisor = static_cast<float>(streams);
    const int token = static_cast<int>(blockIdx.x);
    assert(token < tokens);

    const float *row = hyper_input + static_cast<long>(token) * hyper;
    for (int group_index = 0; group_index < streams; ++group_index) {
        const long start = static_cast<long>(group_index) * hidden;
        float sum_squares = 0.0f;
        for (int i = static_cast<int>(threadIdx.x); i < hidden;
             i += ew_threads) {
            const float value = row[start + i];
            sum_squares += value * value;
        }
        const float total = ew_block_sum(sum_squares, scratch);
        const float mean = total / static_cast<float>(hidden);
        const float scale = 1.0f / sqrtf(mean + eps);
        for (int i = static_cast<int>(threadIdx.x); i < hidden;
             i += ew_threads) {
            const long index = start + i;
            normed[index] = row[index] * scale * (1.0f + norm_weight[index]);
        }
    }
    __syncthreads();

    for (int output = warp; output < lowrank; output += ew_warps) {
        float sum = 0.0f;
        for (int i = lane; i < hyper; i += ew_warp) {
            sum += normed[i] * mix_down[static_cast<long>(output) * hyper + i];
        }
        sum = ew_warp_sum(sum);
        if (lane == 0) {
            lowrank_values[output] =
                ew_activate(sum / divisor, activation_silu);
        }
    }
    __syncthreads();

    for (int output = warp; output < hyper; output += ew_warps) {
        float sum = 0.0f;
        for (int i = lane; i < lowrank; i += ew_warp) {
            sum += lowrank_values[i] *
                   mix_up[static_cast<long>(output) * lowrank + i];
        }
        sum = ew_warp_sum(sum);
        if (lane == 0) {
            mix_weight[output] = ew_sigmoid(sum);
        }
    }
    __syncthreads();

    for (int h = static_cast<int>(threadIdx.x); h < hidden; h += ew_threads) {
        float total = 0.0f;
        for (int c = 0; c < streams; ++c) {
            const long index = static_cast<long>(c) * hidden + h;
            total += mix_weight[index] * normed[index];
        }
        mixed[static_cast<long>(token) * hidden + h] = total / divisor;
    }

    if (inject_weight != nullptr) {
        assert(injection != nullptr);
        for (int c = warp; c < streams; c += ew_warps) {
            float sum = 0.0f;
            for (int i = lane; i < hyper; i += ew_warp) {
                sum += normed[i] *
                       inject_weight[static_cast<long>(c) * hyper + i];
            }
            sum = ew_warp_sum(sum);
            if (lane == 0) {
                injection[static_cast<long>(token) * streams + c] =
                    2.0f * ew_sigmoid(sum / divisor);
            }
        }
    }
}

// `out = silu(a) * b`, the MLP's elementwise product.
extern "C" __global__ void __launch_bounds__(ew_threads)
ew_silu_mul(const float *__restrict__ a, const float *__restrict__ b,
            float *__restrict__ out, int count) {
    assert(count > 0);
    for (int index = static_cast<int>(blockIdx.x) * ew_threads +
                     static_cast<int>(threadIdx.x);
         index < count; index += static_cast<int>(gridDim.x) * ew_threads) {
        out[index] = ew_activate(a[index], activation_silu) * b[index];
    }
}

// `out = sigmoid(a)`: the shared-expert gate and the attention output gate.
extern "C" __global__ void __launch_bounds__(ew_threads)
ew_sigmoid_gate(const float *__restrict__ a, float *__restrict__ out,
                int count) {
    assert(count > 0);
    for (int index = static_cast<int>(blockIdx.x) * ew_threads +
                     static_cast<int>(threadIdx.x);
         index < count; index += static_cast<int>(gridDim.x) * ew_threads) {
        out[index] = ew_sigmoid(a[index]);
    }
}

// In-place softmax over every row of `[rows][columns]`, matching `ops.softmax`
// including the all-`-inf` row, which becomes zeros rather than NaN.
extern "C" __global__ void __launch_bounds__(ew_threads)
ew_softmax_rows(float *__restrict__ values, int rows, int columns) {
    extern __shared__ float shared[];
    assert(columns > 0);
    assert(rows > 0);

    const int row = static_cast<int>(blockIdx.x);
    assert(row < rows);
    float *data = values + static_cast<long>(row) * columns;

    float largest = -INFINITY;
    for (int i = static_cast<int>(threadIdx.x); i < columns; i += ew_threads) {
        largest = fmaxf(largest, data[i]);
    }
    // A mask leaves the diagonal visible, so the maximum is finite; the
    // reference's zero-row path is reproduced anyway because a NaN row would
    // otherwise pass every downstream comparison by never comparing greater.
    float value_shared = largest;
    int index_shared = 0;
    float selected = 0.0f;
    int selected_index = 0;
    __shared__ float largest_shared[ew_threads];
    __shared__ int largest_index[ew_threads];
    ew_block_argmax(value_shared, index_shared, largest_shared, largest_index,
                    &selected, &selected_index);
    if (selected == -INFINITY) {
        for (int i = static_cast<int>(threadIdx.x); i < columns;
             i += ew_threads) {
            data[i] = 0.0f;
        }
        return;
    }

    float total = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < columns; i += ew_threads) {
        const float exponential = expf(data[i] - selected);
        data[i] = exponential;
        total += exponential;
    }
    const float sum = ew_block_sum(total, shared);
    assert(sum > 0.0f);
    const float inverse = 1.0f / sum;
    for (int i = static_cast<int>(threadIdx.x); i < columns; i += ew_threads) {
        data[i] *= inverse;
    }
}

// Index of the largest value of `[count]`, lowest index on an exact tie.
extern "C" __global__ void __launch_bounds__(ew_threads)
ew_argmax(const float *__restrict__ values, int count,
          int *__restrict__ out_index, float *__restrict__ out_value) {
    __shared__ float value_shared[ew_threads];
    __shared__ int index_shared[ew_threads];
    assert(count > 0);

    float best = -INFINITY;
    int best_index = -1;
    for (int index = static_cast<int>(threadIdx.x); index < count;
         index += ew_threads) {
        const float value = values[index];
        if (ew_outranks(value, index, best, best_index)) {
            best = value;
            best_index = index;
        }
    }
    float selected_value = 0.0f;
    int selected_index = 0;
    ew_block_argmax(best, best_index, value_shared, index_shared,
                    &selected_value, &selected_index);
    if (threadIdx.x == 0) {
        out_index[0] = selected_index;
        out_value[0] = selected_value;
    }
}

// Ordered f32 code: monotone in the value, which is what lets a bisection over
// integers stand in for a bisection over floats.
__device__ __forceinline__ unsigned int ew_ordered(float value) {
    const unsigned int bits = __float_as_uint(value);
    if ((bits & 0x80000000u) != 0u) {
        return ~bits;
    } else {
        return bits | 0x80000000u;
    }
}

__device__ __forceinline__ float ew_unordered(unsigned int code) {
    if ((code & 0x80000000u) != 0u) {
        return __uint_as_float(code & 0x7fffffffu);
    } else {
        return __uint_as_float(~code);
    }
}

// splitmix32, the mixer xoshiro's authors recommend for turning a seed into a
// xoshiro state.
__device__ __forceinline__ unsigned int ew_splitmix32(unsigned int *state) {
    *state += 0x9e3779b9u;
    unsigned int value = *state;
    value ^= value >> 16;
    value *= 0x21f0aaadu;
    value ^= value >> 15;
    value *= 0x735a2d97u;
    return value ^ (value >> 15);
}

// One xoshiro128** output, advancing `state`.
__device__ __forceinline__ unsigned int ew_xoshiro128ss_next(
    unsigned int *state) {
    const unsigned int result = ew_rotate_left(state[1] * 5u, 7) * 9u;
    const unsigned int shifted = state[1] << 9;
    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= shifted;
    state[3] = ew_rotate_left(state[3], 11);
    return result;
}

// One uniform in the open interval (0, 1) for `element` of `draw` under `seed`.
//
// A sequential xoshiro stream cannot serve a parallel Gumbel-max draw, so the
// stream is xoshiro128** seeded per (seed, draw, element) by splitmix32. The
// draw therefore stays a pure function of (seed, draw): the same seed replays
// the same token id, and different elements get independent variates.
__device__ __forceinline__ float ew_uniform(unsigned int seed,
                                            unsigned int draw,
                                            unsigned int element) {
    unsigned int source = seed ^ (draw * 0x9e3779b9u) ^ (element * 0x85ebca6bu);
    unsigned int state[4];
    for (int index = 0; index < 4; ++index) {
        state[index] = ew_splitmix32(&source);
    }
    const unsigned int raw = ew_xoshiro128ss_next(state);
    return (static_cast<float>(raw >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

// Repetition penalty of `torch`/`transformers`: positive logits are divided by
// `penalty`, the rest multiplied, once per distinct id in the history.
__device__ __forceinline__ void ew_apply_repetition(float *scaled,
                                                    const int *history,
                                                    int history_len, int vocab,
                                                    float penalty) {
    assert(penalty >= 1.0f);
    for (int j = static_cast<int>(threadIdx.x); j < history_len;
         j += static_cast<int>(blockDim.x)) {
        const int id = history[j];
        if (id >= 0 && id < vocab) {
            // Only the first occurrence applies the penalty, which is what the
            // reference's `logits[unique_ids]` gather does; a repeated id must
            // not compound the division.
            bool first = true;
            for (int k = 0; k < j; ++k) {
                if (history[k] == id) {
                    first = false;
                }
            }
            if (first) {
                const float value = scaled[id];
                if (value > 0.0f) {
                    scaled[id] = value / penalty;
                } else {
                    scaled[id] = value * penalty;
                }
            }
        }
    }
}

// Count of `values` at or above `threshold`, summed over the block.
__device__ __forceinline__ float ew_count_ge(const float *values, int count,
                                             float threshold,
                                             float *scratch) {
    float total = 0.0f;
    for (int index = static_cast<int>(threadIdx.x); index < count;
         index += static_cast<int>(blockDim.x)) {
        if (values[index] >= threshold) {
            total += 1.0f;
        }
    }
    return ew_block_sum(total, scratch);
}

// Probability mass of `values` at or above `threshold`, relative to `maximum`.
__device__ __forceinline__ float ew_mass_ge(const float *values, int count,
                                            float threshold, float maximum,
                                            float *scratch) {
    float total = 0.0f;
    for (int index = static_cast<int>(threadIdx.x); index < count;
         index += static_cast<int>(blockDim.x)) {
        const float value = values[index];
        if (value >= threshold) {
            total += expf(value - maximum);
        }
    }
    return ew_block_sum(total, scratch);
}

// Total probability mass of the softened scores, relative to the maximum: the
// denominator the nucleus predicate compares against.
__device__ __forceinline__ float ew_total_mass(const float *values, int count,
                                               float maximum,
                                               float *scratch) {
    float total = 0.0f;
    for (int index = static_cast<int>(threadIdx.x); index < count;
         index += static_cast<int>(blockDim.x)) {
        total += expf(values[index] - maximum);
    }
    return ew_block_sum(total, scratch);
}

// The two nucleus thresholds, as ordered codes.
struct ew_cutoffs {
    unsigned int count_code;
    unsigned int mass_code;
};

// Largest threshold code whose predicate holds, for the count and the mass
// predicate at once: 32 steps each, one pass over the vocabulary per step.
//
// Both predicates are monotone decreasing in the threshold, so the bisection is
// exact even though the value axis is dense: the answer is the largest
// representable threshold that still keeps `top_k` candidates / reaches
// `top_p`, and every value tied at that threshold is kept. The search range is
// the observed score range, whose low end always satisfies `count >= 1`, so
// neither predicate is ever evaluated on a NaN.
//
// `total` is the mass of the whole vocabulary: the nucleus condition is
// `mass >= top_p * total`, not `mass >= top_p`, because the masses here are
// unnormalised (`exp(score - maximum)` puts the maximum at 1, so the first
// candidate alone already reaches any `top_p` below 1).
__device__ __forceinline__ struct ew_cutoffs ew_find_cutoffs(
    const float *scaled, int vocab, float lowest, float maximum, int top_k,
    float top_p, float total, float *count_scratch, float *mass_scratch) {
    assert(vocab > 0);
    assert(top_k >= 1);
    assert(maximum >= lowest);

    const unsigned int lowest_code = ew_ordered(lowest);
    const unsigned int highest_code = ew_ordered(maximum);
    unsigned int low_count = lowest_code;
    unsigned int high_count = highest_code;
    unsigned int low_mass = lowest_code;
    unsigned int high_mass = highest_code;
    const float wanted = static_cast<float>(top_k);
    const float wanted_mass = top_p * total;
    for (int step = 0; step < ew_bisect_steps; ++step) {
        const unsigned int mid_count = static_cast<unsigned int>(
            (static_cast<uint64_t>(low_count) + high_count + 1u) / 2u);
        const unsigned int mid_mass = static_cast<unsigned int>(
            (static_cast<uint64_t>(low_mass) + high_mass + 1u) / 2u);
        const float count =
            ew_count_ge(scaled, vocab, ew_unordered(mid_count), count_scratch);
        const float mass = ew_mass_ge(scaled, vocab, ew_unordered(mid_mass),
                                      maximum, mass_scratch);
        if (low_count < high_count) {
            if (count >= wanted) {
                low_count = mid_count;
            } else {
                high_count = mid_count - 1u;
            }
        } else {
            assert(low_count == high_count);
        }
        if (low_mass < high_mass) {
            if (mass >= wanted_mass) {
                low_mass = mid_mass;
            } else {
                high_mass = mid_mass - 1u;
            }
        } else {
            assert(low_mass == high_mass);
        }
    }
    struct ew_cutoffs cutoffs;
    cutoffs.count_code = low_count;
    cutoffs.mass_code = low_mass;
    return cutoffs;
}

// Result of a draw: the token id and the retained probability mass.
struct ew_draw {
    int id;
    float mass;
};

// Gumbel-max over the retained set, plus the retained probability mass.
//
// One uniform per candidate, `argmax(score + gumbel)`: that samples the
// categorical distribution exactly and needs no cumulative array, so the draw
// is a single pass and no host buffer of the vocabulary is involved.
__device__ __forceinline__ struct ew_draw ew_gumbel_draw(
    const float *scaled, int vocab, float threshold, float maximum,
    unsigned int seed, unsigned int draw, float *value_shared,
    int *index_shared, float *scratch) {
    assert(vocab > 0);
    assert(threshold > -INFINITY);

    float mass = 0.0f;
    float best = -INFINITY;
    int best_index = -1;
    for (int index = static_cast<int>(threadIdx.x); index < vocab;
         index += static_cast<int>(blockDim.x)) {
        const float value = scaled[index];
        if (value >= threshold) {
            mass += expf(value - maximum);
            const float uniform =
                ew_uniform(seed, draw, static_cast<unsigned int>(index));
            const float key = value - logf(-logf(uniform));
            if (ew_outranks(key, index, best, best_index)) {
                best = key;
                best_index = index;
            }
        }
    }
    const float retained_mass = ew_block_sum(mass, scratch);
    float selected = 0.0f;
    int selected_index = -1;
    ew_block_argmax(best, best_index, value_shared, index_shared, &selected,
                    &selected_index);
    struct ew_draw result;
    result.id = selected_index;
    result.mass = retained_mass;
    return result;
}

// Temperature/top-k/top-p/min-p/repetition sampling of one token, on device.
//
// `scaled` is caller-provided scratch of `vocab` floats: the penalised,
// temperature-scaled scores are written once and every pass reads them, so no
// host buffer of the vocabulary is ever touched. `out_id` receives the token
// and `out_logprob` the natural log of its probability under the retained
// set.
//
// `temperature <= 0` is greedy: the winner is the lowest index among the tied
// maxima (the same rule `ew_argmax` follows) and the reported logprob is the
// argmax's log-softmax over the whole vocabulary, because a filtered set is
// meaningless once the distribution is a point mass.
extern "C" __global__ void __launch_bounds__(sampler_threads)
ew_sampler(const float *__restrict__ logits, const int *__restrict__ history,
           int history_len, int vocab, float penalty, float temperature,
           int top_k, float top_p, float min_p, unsigned int seed,
           unsigned int draw, float *__restrict__ scaled,
           int *__restrict__ out_id, float *__restrict__ out_logprob) {
    __shared__ float value_shared[sampler_threads];
    __shared__ int index_shared[sampler_threads];
    // One scratch slice per reduction: the block sums are inlined into this
    // kernel, and a shared slot used by two of them is a read/write hazard the
    // compiler is free to reorder across the inlined barriers.
    __shared__ float scratch[sampler_scratch_slots * ew_scratch];
    float *const mass_scratch = scratch + 0 * ew_scratch;
    float *const count_scratch = scratch + 1 * ew_scratch;
    float *const nucleus_scratch = scratch + 2 * ew_scratch;
    float *const draw_scratch = scratch + 3 * ew_scratch;
    assert(vocab > 0);
    assert(top_k >= 1);

    for (int index = static_cast<int>(threadIdx.x); index < vocab;
         index += sampler_threads) {
        scaled[index] = logits[index];
    }
    __syncthreads();
    if (penalty != 1.0f) {
        ew_apply_repetition(scaled, history, history_len, vocab, penalty);
    }
    __syncthreads();
    if (temperature > 0.0f) {
        for (int index = static_cast<int>(threadIdx.x); index < vocab;
             index += sampler_threads) {
            scaled[index] /= temperature;
        }
    }
    __syncthreads();

    float largest = -INFINITY;
    float smallest = INFINITY;
    for (int index = static_cast<int>(threadIdx.x); index < vocab;
         index += sampler_threads) {
        largest = fmaxf(largest, scaled[index]);
        smallest = fminf(smallest, scaled[index]);
    }
    float selected_value = 0.0f;
    int selected_index = 0;
    ew_block_argmax(largest, 0, value_shared, index_shared, &selected_value,
                    &selected_index);
    const float maximum = selected_value;
    ew_block_argmax(-smallest, 0, value_shared, index_shared, &selected_value,
                    &selected_index);
    const float minimum = -selected_value;
    assert(maximum >= minimum);
    assert(minimum > -INFINITY);

    const float total = ew_total_mass(scaled, vocab, maximum, mass_scratch);
    assert(total >= 1.0f);

    if (temperature <= 0.0f) {
        float greedy_value = -INFINITY;
        int greedy_index = -1;
        for (int index = static_cast<int>(threadIdx.x); index < vocab;
             index += sampler_threads) {
            if (ew_outranks(scaled[index], index, greedy_value, greedy_index)) {
                greedy_value = scaled[index];
                greedy_index = index;
            }
        }
        ew_block_argmax(greedy_value, greedy_index, value_shared, index_shared,
                        &selected_value, &selected_index);
        if (threadIdx.x == 0) {
            // The log probability of the argmax under the unfiltered
            // distribution: `(score - maximum) - log(total)` with the score at
            // the maximum, so it is `-log(total)`.
            out_id[0] = selected_index;
            out_logprob[0] = (scaled[selected_index] - maximum) - logf(total);
        }
        return;
    }

    // `top_p >= 1` filters nothing; without this guard the mass predicate is
    // satisfied by the maximum alone and the "nucleus" would collapse onto it.
    const int wanted_k = ew_min_int(top_k, vocab);
    const float wanted_p = (top_p >= 1.0f) ? 0.0f : top_p;
    const struct ew_cutoffs cutoffs =
        ew_find_cutoffs(scaled, vocab, minimum, maximum, wanted_k, wanted_p,
                        total, count_scratch, nucleus_scratch);
    unsigned int threshold_code = cutoffs.count_code;
    if (wanted_p > 0.0f) {
        threshold_code = ew_max_code(threshold_code, cutoffs.mass_code);
    }
    if (min_p > 0.0f) {
        const unsigned int min_code = ew_ordered(maximum + logf(min_p));
        threshold_code = ew_max_code(threshold_code, min_code);
    }
    const float threshold = ew_unordered(threshold_code);

    const struct ew_draw result =
        ew_gumbel_draw(scaled, vocab, threshold, maximum, seed, draw,
                       value_shared, index_shared, draw_scratch);
    if (threadIdx.x == 0) {
        // The retained set is the distribution the caller asked for, so the
        // log probability is its log-softmax: `(score - maximum) - log(mass)`.
        out_id[0] = result.id;
        out_logprob[0] = (scaled[result.id] - maximum) - logf(result.mass);
    }
}
