// Model-graph glue kernels.
//
// These are the operations the block kernels take as given: the dtype casts at
// the block boundaries, the KV row gather the attention kernel consumes, the
// residual injection the hyper-connection launches return as weights, the
// attention output gate, the routed/shared MoE combine, and the two Gated
// DeltaNet preprocessing steps the I-009 kernels deliberately leave to the
// graph (l2-normalise q/k out of the conv output, and the beta/decay gates).
//
// Every entry point is `extern "C"` so the cubin exposes unmangled symbols.
// The arithmetic is the reference's: `model_gdn_decay` reproduces
// `ref/gdn.zig`'s softplus/sigmoid, `model_moe_combine` sums the routed experts
// in the router's slot order, and the casts round to nearest even, which is
// what `weights/quant.zig` calls `f32_to_bf16`.

#include <cstdint>
#include <cuda_bf16.h>

namespace {

constexpr int glue_threads = 256;
constexpr int glue_warp = 32;

__device__ __forceinline__ float glue_sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ float glue_softplus(float value) {
    // Matches `ref/ops.zig`: the linear regime starts above 20.
    return value > 20.0f ? value : log1pf(expf(value));
}

} // namespace

// f32 -> bf16, round to nearest even.
extern "C" __global__ void __launch_bounds__(glue_threads)
model_cast_f32_bf16(const float *__restrict__ source,
                    __nv_bfloat16 *__restrict__ destination, int count) {
    for (int index = static_cast<int>(blockIdx.x) * glue_threads +
                         static_cast<int>(threadIdx.x);
         index < count; index += static_cast<int>(gridDim.x) * glue_threads) {
        destination[index] = __float2bfloat16_rn(source[index]);
    }
}

// Gather rows of a bf16 matrix by an int32 row index; a negative index is a
// padding slot the attention kernel masks, so its row is written as zero.
extern "C" __global__ void __launch_bounds__(glue_threads)
model_gather_bf16(const __nv_bfloat16 *__restrict__ source,
                  const int *__restrict__ row_index, int count, int width,
                  __nv_bfloat16 *__restrict__ destination) {
    const long long total = static_cast<long long>(count) * width;
    for (long long index = static_cast<long long>(blockIdx.x) * glue_threads +
                                static_cast<long long>(threadIdx.x);
         index < total; index += static_cast<long long>(gridDim.x) * glue_threads) {
        const int row = static_cast<int>(index / width);
        const int column = static_cast<int>(index % width);
        const int source_row = row_index[row];
        destination[index] =
            source_row < 0
                ? __float2bfloat16(0.0f)
                : source[static_cast<long long>(source_row) * width + column];
    }
}

// The hyper-connection residual update: every stream of a token receives the
// block's single-stream output scaled by that stream's injection weight.
extern "C" __global__ void __launch_bounds__(glue_threads)
model_inject(const float *__restrict__ block,
             const float *__restrict__ injection, float *__restrict__ hyper,
             int tokens, int hidden, int streams) {
    const int token = static_cast<int>(blockIdx.y);
    if (token >= tokens) {
        return;
    }
    const int per_token = hidden * streams;
    for (int local = static_cast<int>(blockIdx.x) * glue_threads +
                         static_cast<int>(threadIdx.x);
         local < per_token;
         local += static_cast<int>(gridDim.x) * glue_threads) {
        const int stream = local / hidden;
        const int column = local % hidden;
        const long long index = static_cast<long long>(token) * per_token + local;
        const float weight =
            injection[static_cast<long long>(token) * streams + stream];
        hyper[index] +=
            block[static_cast<long long>(token) * hidden + column] * weight;
    }
}

// The attention output gate: `values * sigmoid(gate)`.
extern "C" __global__ void __launch_bounds__(glue_threads)
model_sigmoid_mul(const float *__restrict__ values,
                  const float *__restrict__ gate, float *__restrict__ out,
                  int count) {
    for (int index = static_cast<int>(blockIdx.x) * glue_threads +
                         static_cast<int>(threadIdx.x);
         index < count; index += static_cast<int>(gridDim.x) * glue_threads) {
        out[index] = values[index] * glue_sigmoid(gate[index]);
    }
}

// Gated DeltaNet preprocessing: split the token-major conv output into q/k/v,
// l2-normalise q and k, scale q by `q_scale`, and cast v to f32.
//
// One block per output row. Row `t * k_heads + h` is a query head,
// `tokens * k_heads + t * k_heads + h` a key head, and the tail a value head,
// which mirrors `ref/gdn.zig`'s `forward_split`/`load_head` pair.
extern "C" __global__ void __launch_bounds__(glue_threads)
model_gdn_prepare(const __nv_bfloat16 *__restrict__ conv_out,
                  float *__restrict__ query, float *__restrict__ key,
                  float *__restrict__ value, int tokens, int k_heads,
                  int v_heads, int k_head, int v_head, float q_scale,
                  float eps) {
    __shared__ float partial[glue_threads / glue_warp];
    __shared__ float norm;
    const int k_rows = tokens * k_heads;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int key_dim = k_heads * k_head;
    const int value_dim = v_heads * v_head;
    const int conv_dim = 2 * key_dim + value_dim;
    int token = 0;
    int head = 0;
    const __nv_bfloat16 *source = nullptr;
    float *destination = nullptr;
    int width = 0;
    bool normalise = false;
    // Only the query is scaled by `1/sqrt(k_head)`; the reference l2-normalises
    // the key without it (`ref/gdn.zig`'s `load_head`).
    float scale = 1.0f;
    if (row < k_rows) {
        token = row / k_heads;
        head = row % k_heads;
        source = conv_out + static_cast<long long>(token) * conv_dim +
                 head * k_head;
        destination = query + static_cast<long long>(row) * k_head;
        width = k_head;
        normalise = true;
        scale = q_scale;
    } else if (row < 2 * k_rows) {
        const int local = row - k_rows;
        token = local / k_heads;
        head = local % k_heads;
        source = conv_out + static_cast<long long>(token) * conv_dim +
                 key_dim + head * k_head;
        destination = key + static_cast<long long>(local) * k_head;
        width = k_head;
        normalise = true;
    } else {
        const int local = row - 2 * k_rows;
        token = local / v_heads;
        head = local % v_heads;
        source = conv_out + static_cast<long long>(token) * conv_dim +
                 2 * key_dim + head * v_head;
        destination = value + static_cast<long long>(local) * v_head;
        width = v_head;
        normalise = false;
    }

    float squares = 0.0f;
    if (normalise) {
        for (int column = tid; column < width; column += glue_threads) {
            const float item = __bfloat162float(source[column]);
            squares += item * item;
        }
        for (int offset = glue_warp / 2; offset > 0; offset /= 2) {
            squares += __shfl_down_sync(0xffffffffu, squares, offset);
        }
        if (tid % glue_warp == 0) {
            partial[tid / glue_warp] = squares;
        }
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int warp = 0; warp < glue_threads / glue_warp; ++warp) {
                total += partial[warp];
            }
            norm = rsqrtf(total + eps);
        }
        __syncthreads();
    }

    for (int column = tid; column < width; column += glue_threads) {
        const float item = __bfloat162float(source[column]);
        destination[column] = normalise ? item * norm * scale : item;
    }
}

// The Gated DeltaNet gates: `beta = sigmoid(b)` and
// `g = -exp(A_log) * softplus(a + dt_bias)`.
extern "C" __global__ void __launch_bounds__(glue_threads)
model_gdn_decay(const float *__restrict__ proj_a,
                const float *__restrict__ proj_b,
                const float *__restrict__ a_log,
                const float *__restrict__ dt_bias, float *__restrict__ beta,
                float *__restrict__ decay, int tokens, int heads) {
    const int token = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(threadIdx.x);
    if (token >= tokens || head >= heads) {
        return;
    }
    const long long index = static_cast<long long>(token) * heads + head;
    beta[index] = glue_sigmoid(proj_b[index]);
    decay[index] =
        -expf(a_log[head]) * glue_softplus(proj_a[index] + dt_bias[head]);
}

// Routed experts plus the gated shared expert: the whole MoE combine.
//
// `expert_out` is `[experts][tokens][hidden]`; the routed sum is taken in the
// router's slot order. The shared expert is gated by
// `sigmoid(dot(x, gate_logit))`, as `ref/moe.zig` describes.
extern "C" __global__ void __launch_bounds__(glue_threads)
model_moe_combine(const float *__restrict__ expert_out,
                  const uint32_t *__restrict__ indices,
                  const float *__restrict__ router_weights,
                  const float *__restrict__ shared_out,
                  const float *__restrict__ gate_logit,
                  const float *__restrict__ x, float *__restrict__ out,
                  int tokens, int hidden, int top_k) {
    __shared__ float partial[glue_threads / glue_warp];
    __shared__ float gate;
    const int token = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (token >= tokens) {
        return;
    }
    const float *row = x + static_cast<long long>(token) * hidden;
    float dot = 0.0f;
    for (int column = tid; column < hidden; column += glue_threads) {
        dot += row[column] * gate_logit[column];
    }
    for (int offset = glue_warp / 2; offset > 0; offset /= 2) {
        dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    if (tid % glue_warp == 0) {
        partial[tid / glue_warp] = dot;
    }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int warp = 0; warp < glue_threads / glue_warp; ++warp) {
            total += partial[warp];
        }
        gate = glue_sigmoid(total);
    }
    __syncthreads();

    float *destination = out + static_cast<long long>(token) * hidden;
    for (int column = tid; column < hidden; column += glue_threads) {
        float total = 0.0f;
        for (int slot = 0; slot < top_k; ++slot) {
            const int expert = static_cast<int>(
                indices[static_cast<long long>(token) * top_k + slot]);
            const float weight =
                router_weights[static_cast<long long>(token) * top_k + slot];
            total += expert_out[(static_cast<long long>(expert) * tokens +
                                 token) * hidden + column] * weight;
        }
        destination[column] =
            total + gate * shared_out[static_cast<long long>(token) * hidden +
                                      column];
    }
}
