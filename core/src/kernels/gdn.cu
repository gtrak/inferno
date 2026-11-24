// Gated DeltaNet kernels: the four stages of a linear-attention layer that are
// not a GEMM.
//
// Every entry point is `extern "C"` so the cubin exposes unmangled symbols that
// `core/src/kernels/gdn.zig` looks up by name. Arithmetic order follows
// `tools/ref-trace/reference/modeling_qwen4_exp.py` lines 220-641 and is checked
// against the fp32 CPU oracle `core/src/ref/gdn.zig`.
//
//   gdn_conv        causal depthwise conv (kernel 4, SiLU) + history update
//   gdn_prefill     chunked (UT-transform) gated delta rule, chunk 64
//   gdn_decode      recurrent single-token gated delta rule + gated RMSNorm
//   gdn_gate_norm   RMSNormGated(out, z) for the prefill path
//
// Layouts are token-major wherever a token's row is consumed as a whole
// (`[tokens][width]`), because that is what the projection GEMMs produce and
// what the next stage reads. The recurrent state and the conv history are the
// two exceptions: they are the per-sequence state pool buffers, head-major and
// channel-major respectively.
//
// Dtypes follow the reference (`modeling_qwen4_exp.py:309-313` upcasts the
// rule's operands before it runs): `query`, `key`, `value`, `beta`, `decay` and
// the recurrent state are f32; the conv input/output, `out` and the gated-norm
// gate `z` are bf16. `gdn.zig`'s header table is the contract the engine codes
// against.
//
// Launch geometry (checked by the Zig wrappers):
//   gdn_conv       grid (ceil(conv_dim / 32), 1, 1)   block (32, tile, 1)
//                  `tile` is 8 for a prefill and 1 for a decode step
//   gdn_prefill    grid (num_v_heads, 1, 1)           block (256, 1, 1)
//   gdn_decode     grid (num_v_heads, 1, 1)           block (1024, 1, 1)
//   gdn_gate_norm  grid (rows, 1, 1)                  block (128, 1, 1)
//
// Shared memory: `gdn_conv` uses a fixed static tile, the other three take a
// dynamic allocation that `gdn.zig` sizes exactly (the formulas are repeated
// there, which is why the layouts are spelled out in the kernels):
//   gdn_prefill     (k_head * v_head + chunk * v_head + 4 * chunk + k_head) f32
//   gdn_decode      (k_head * v_head + 2 * k_head + v_head + 32) f32
//   gdn_gate_norm   (width + 32) f32

#include <cuda_bf16.h>

namespace {

// Chunk of the UT transform, fixed by the reference implementation.
constexpr int gdn_chunk = 64;

// Production head dimensions (`linear_key_head_dim` / `linear_value_head_dim`).
// The real dimensions travel as arguments; this is only the compile-time bound
// the thread mapping and the register budget are sized for.
constexpr int head_dim_max = 128;

// Threads of `gdn_conv` that reduce one dot product (a power of two, so the
// butterfly shuffle below needs no mask arithmetic).
constexpr int dot_threads = 4;

// `gdn_conv` geometry: the model's `linear_conv_kernel_dim` is 4, and a block
// owns 32 channels for the whole sequence so the history update is race free.
// `conv_token_tile` is the largest token tile; the caller picks the actual tile
// as `blockDim.y`, so a decode step (one token) does not launch eight slots per
// block and throw seven away.
constexpr int conv_kernel = 4;
constexpr int conv_history = conv_kernel - 1;
constexpr int conv_channels_per_block = 32;
constexpr int conv_token_tile_max = 8;

// Values of `activation`, matching the order of `Activation` in
// `core/src/ref/ops.zig`.
constexpr int activation_silu = 1;
constexpr int activation_sigmoid = 2;

// `x * sigmoid(x)`, spelled as the reference does.
__device__ __forceinline__ float gdn_silu(float value) {
    return value / (1.0f + expf(-value));
}

// `x / (1 + exp(-x))` or `1 / (1 + exp(-x))`; `activation_silu` is the model's
// `hidden_act`, the gate of `RMSNormGated` uses `output_gate_type`.
__device__ __forceinline__ float gdn_activate(float value, int activation) {
    if (activation == activation_silu) {
        return gdn_silu(value);
    }
    if (activation == activation_sigmoid) {
        return 1.0f / (1.0f + expf(-value));
    }
    return value;
}

// Round through bf16, the dtype the engine stores every intermediate in.
__device__ __forceinline__ float gdn_round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

// Sum of `value` over every thread of the block, delivered to all of them.
//
// `scratch` holds `reduction_scratch_floats` floats; the caller must be at a
// point where every thread reaches this call.
// `scratch` holds one float per warp, and the block of `gdn_decode` is 1024
// threads wide (32 warps), which is what sizes `reduction_scratch_floats` on the
// host side.
__device__ __forceinline__ float gdn_block_sum(float value, float *scratch) {
    const unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) >> 5;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();

    float total = 0.0f;
    if (threadIdx.x < warps) {
        total = scratch[threadIdx.x];
    }
    if (warp == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(mask, total, offset);
        }
        if (lane == 0) {
            scratch[0] = total;
        }
    }
    __syncthreads();
    return scratch[0];
}

// `RMSNormGated(values, gate)` over one head row held in shared memory.
//
// The reference normalises first, then multiplies by the per-head weight and
// the activated gate (`ref/ops.zig`'s `rmsnorm_gated`, which is the oracle this
// kernel is gated against). `values` arrives as fp32 accumulators and is
// rounded through bf16 first, because the engine keeps `core_attn_out` in a
// bf16 activation buffer before the norm; the normalised value itself is *not*
// re-rounded, because the oracle does not and the extra bf16 step would only
// add a rounding that the reference does not model.
__device__ __forceinline__ void gdn_gate_norm_row(
    float *values,
    const __nv_bfloat16 *gate,
    const float *weight,
    __nv_bfloat16 *out,
    int width,
    float epsilon,
    int activation,
    float *scratch) {
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        values[index] = gdn_round_bf16(values[index]);
    }
    __syncthreads();

    float squares = 0.0f;
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        squares += values[index] * values[index];
    }
    const float total = gdn_block_sum(squares, scratch);
    const float scale = rsqrtf(total / (float)width + epsilon);

    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        const float normed = values[index] * scale;
        const float gated = gdn_activate(__bfloat162float(gate[index]), activation);
        out[index] = __float2bfloat16_rn(weight[index] * normed * gated);
    }
}

}  // namespace

// Causal depthwise conv over `[history | x]`, SiLU, then the history update.
//
// `x` and `out` are `[tokens][conv_dim]`; `weight` is `[conv_dim][4]`;
// `state` is `[conv_dim][3]`, channel-major, holding the last three raw values
// of the sequence, and is updated in place. Output position t reads the window
// starting at t - 3, which is `F.conv1d(padding=k-1)[:, :, :S]`.
//
// One block owns 32 channels for every token of the call, which is what makes
// the in-place history update race free: no other block reads or writes those
// channels. Decode is the same kernel with `tokens == 1`.
extern "C" __global__ void gdn_conv(
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ weight,
    float *__restrict__ state,
    __nv_bfloat16 *__restrict__ out,
    int tokens,
    int conv_dim) {
    // window[channel][history .. history + conv_token_tile) is the current
    // tile; [0, history) is the retained tail of the previous one.
    __shared__ float window[conv_channels_per_block][conv_history + conv_token_tile_max];

    const int channel_in_block = threadIdx.x;
    const int slot = threadIdx.y;
    const int channel = blockIdx.x * conv_channels_per_block + threadIdx.x;
    const bool channel_valid = channel < conv_dim;

    float taps[conv_kernel];
    #pragma unroll
    for (int j = 0; j < conv_kernel; ++j) {
        taps[j] = 0.0f;
        if (channel_valid) {
            taps[j] = weight[channel * conv_kernel + j];
        }
    }

    if (slot == 0) {
        #pragma unroll
        for (int j = 0; j < conv_history; ++j) {
            window[channel_in_block][j] = 0.0f;
            if (channel_valid) {
                window[channel_in_block][j] = state[channel * conv_history + j];
            }
        }
    }
    __syncthreads();

    const int tile = (int)blockDim.y;
    for (int tile_start = 0; tile_start < tokens; tile_start += tile) {
        const int token = tile_start + slot;
        const bool token_valid = token < tokens;

        float loaded = 0.0f;
        if (channel_valid && token_valid) {
            loaded = __bfloat162float(x[token * conv_dim + channel]);
        }
        window[channel_in_block][conv_history + slot] = loaded;
        __syncthreads();

        if (channel_valid && token_valid) {
            float total = 0.0f;
            #pragma unroll
            for (int j = 0; j < conv_kernel; ++j) {
                total += taps[j] * window[channel_in_block][slot + j];
            }
            out[token * conv_dim + channel] = __float2bfloat16_rn(gdn_silu(total));
        }
        __syncthreads();

        // Slide the window by the number of tokens this tile actually held, so
        // a short final tile keeps the tail of the previous window.
        const int advanced = min(tile, tokens - tile_start);
        if (slot == 0) {
            #pragma unroll
            for (int j = 0; j < conv_history; ++j) {
                window[channel_in_block][j] = window[channel_in_block][advanced + j];
            }
        }
        __syncthreads();
    }

    if (slot == 0 && channel_valid) {
        #pragma unroll
        for (int j = 0; j < conv_history; ++j) {
            state[channel * conv_history + j] = window[channel_in_block][j];
        }
    }
}

// `RMSNormGated(out, z)`: one block per `(token, value head)` row.
//
// `values`, `gate` and `out` are `[rows][width]` and `weight` is `[width]`.
extern "C" __global__ void gdn_gate_norm(
    const __nv_bfloat16 *__restrict__ values,
    const __nv_bfloat16 *__restrict__ gate,
    const float *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out,
    int width,
    float epsilon,
    int activation) {
    extern __shared__ float shared[];
    float *row = shared;
    float *scratch = shared + width;

    const long base = (long)blockIdx.x * width;
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        row[index] = __bfloat162float(values[base + index]);
    }
    __syncthreads();

    gdn_gate_norm_row(
        row,
        gate + base,
        weight,
        out + base,
        width,
        epsilon,
        activation,
        scratch);
}

// Recurrent gated delta rule for one token, with the gated RMSNorm folded in.
//
// One block per value head. The 128x128 fp32 recurrent state lives in shared
// memory: 64 KiB at the production head size, so the caller raises the dynamic
// shared limit to the 99 KiB one block may opt into (asserted in `gdn.zig`).
// The state is loaded, decayed, updated by the delta rule, read once for the
// output and written back. The gate of the same head is applied by this block as well, so a decode
// step is two launches per layer: `gdn_conv` and this kernel.
//
// `query` and `key` are `[num_k_heads][k_head]`, `value` and `gate` are
// `[num_v_heads][v_head]` for this token, `beta` and `decay` are
// `[num_v_heads]`, `state` is `[num_v_heads][k_head][v_head]`.
//
// The output uses the identity
//   sum_d (s_d + k_d * delta) * q_d = sum_d s_d q_d + delta * (k . q)
// with `s` the decayed state, which halves the passes over the state in shared
// memory against reading the updated state again. It is algebraically the same
// sum as the reference's second reduction.
extern "C" __global__ void gdn_decode(
    const float *__restrict__ query,
    const float *__restrict__ key,
    const float *__restrict__ value,
    const float *__restrict__ beta,
    const float *__restrict__ decay,
    float *__restrict__ state,
    const __nv_bfloat16 *__restrict__ gate,
    const float *__restrict__ norm_weight,
    __nv_bfloat16 *__restrict__ out,
    int num_k_heads,
    int k_head,
    int v_head,
    float epsilon,
    int activation) {
    extern __shared__ float shared[];
    float *state_shared = shared;                       // [k_head][v_head]
    float *query_shared = state_shared + k_head * v_head;  // [k_head]
    float *key_shared = query_shared + k_head;             // [k_head]
    float *core_shared = key_shared + k_head;              // [v_head]
    float *scratch = core_shared + v_head;

    const int head = blockIdx.x;
    const int group = (int)gridDim.x / num_k_heads;
    const int key_head = head / group;
    const long state_base = (long)head * k_head * v_head;
    const long source = (long)key_head * k_head;

    for (int index = threadIdx.x; index < k_head * v_head; index += blockDim.x) {
        state_shared[index] = state[state_base + index];
    }
    if (threadIdx.x < k_head) {
        query_shared[threadIdx.x] = query[source + threadIdx.x];
        key_shared[threadIdx.x] = key[source + threadIdx.x];
    }
    __syncthreads();

    if (threadIdx.x < v_head) {
        const int column = threadIdx.x;
        const float step_decay = expf(decay[head]);
        const float learning_rate = beta[head];

        // Decay the state and reduce both dot products the delta rule needs in
        // one pass over the column: `kv_mem` against the key and the decayed
        // value against the query.
        float memory = 0.0f;
        float decayed_output = 0.0f;
        float key_query = 0.0f;
        for (int dim = 0; dim < k_head; ++dim) {
            const float decayed = state_shared[dim * v_head + column] * step_decay;
            state_shared[dim * v_head + column] = decayed;
            memory += decayed * key_shared[dim];
            decayed_output += decayed * query_shared[dim];
            key_query += key_shared[dim] * query_shared[dim];
        }
        const float expected = value[(long)head * v_head + column];
        const float delta = (expected - memory) * learning_rate;

        for (int dim = 0; dim < k_head; ++dim) {
            state_shared[dim * v_head + column] += key_shared[dim] * delta;
        }
        core_shared[column] = decayed_output + delta * key_query;
    }
    __syncthreads();

    gdn_gate_norm_row(
        core_shared,
        gate + (long)head * v_head,
        norm_weight,
        out + (long)head * v_head,
        v_head,
        epsilon,
        activation,
        scratch);
    __syncthreads();

    for (int index = threadIdx.x; index < k_head * v_head; index += blockDim.x) {
        state[state_base + index] = state_shared[index];
    }
}

// Chunked gated delta rule: the UT-transform form of the same recurrence.
//
// One block per value head walks the sequence in chunks of 64, keeping the
// 128x128 fp32 recurrent state in shared memory. Per chunk, exactly as the
// reference does:
//
//   1. cumulative decay inside the chunk, `cum[i] = sum_{m<=i} g[m]`;
//   2. the inter-chunk read `out[i] = q[i] * exp(cum[i]) . S_prev`, before the
//      state moves;
//   3. for each row j of the chunk, the 64 dot products that row needs: the
//      strictly-lower part of `ut_system[j][m] = beta_j (k_j . k_m) exp(cum_j -
//      cum_m)` and the column of the intra-chunk attention
//      `intra[i][j] = (q_i . k_j) exp(cum_i - cum_j)`, both taken from the same
//      `k_j` so every `(i, j)` pair is dotted exactly once;
//   4. the two unit-lower-triangular systems solved by forward substitution,
//      `new_values = solve(ut, v * beta)` and `k_cumdecay = solve(ut, k * beta *
//      exp(cum))`, one thread per right-hand-side column. This is the only
//      inherently sequential part of the chunk;
//   5. `v_new = new_values - k_cumdecay . S_prev`, the intra-chunk output
//      `out[i] += intra[i][j] v_new[j]` for `i >= j`, and, after the last row,
//      `S = S * exp(cum[63]) + k_decayed^T v_new` with
//      `k_decayed[j] = k_j exp(cum[63] - cum[j])`.
//
// Rows past `tokens` inside the final chunk behave as the reference's padding:
// zero query, key, value and beta, and zero decay, so they leave the state
// untouched and their output is discarded. Those rows are never read from the
// sequence tensors either: every load of `query`, `key` and `value` in the
// chunk is behind a `row_valid` / `other_valid` test, so no element at index
// `tokens` or beyond is addressed.
extern "C" __global__ void gdn_prefill(
    const float *__restrict__ query,
    const float *__restrict__ key,
    const float *__restrict__ value,
    const float *__restrict__ beta,
    const float *__restrict__ decay,
    float *__restrict__ state,
    __nv_bfloat16 *__restrict__ out,
    int tokens,
    int num_k_heads,
    int num_v_heads,
    int k_head,
    int v_head) {
    extern __shared__ float shared[];
    float *state_shared = shared;                          // [k_head][v_head]
    float *out_shared = state_shared + k_head * v_head;    // [chunk][v_head]
    float *cum_shared = out_shared + gdn_chunk * v_head;   // [chunk]
    float *beta_shared = cum_shared + gdn_chunk;           // [chunk]
    float *ut_row = beta_shared + gdn_chunk;               // [chunk]
    float *intra_column = ut_row + gdn_chunk;              // [chunk]
    float *cumdecay_row = intra_column + gdn_chunk;        // [k_head]

    const int head = blockIdx.x;
    const int group = (int)gridDim.x / num_k_heads;
    const int key_head = head / group;
    const int value_column = threadIdx.x;
    const int key_dimension = threadIdx.x - head_dim_max;
    const bool value_active = value_column < v_head;
    const bool key_active = key_dimension >= 0 && key_dimension < k_head;
    const long state_base = (long)head * k_head * v_head;

    // The solved right-hand side of this thread's triangular system and the
    // `v_new` rows it produces. Every thread owns exactly one system, a value
    // column (threads 0..127) or a key dimension (threads 128..255), so one
    // array serves both roles. Both are indexed by the substitution row, which
    // is why the compiler places them in local memory: the substitution is the
    // one step of the chunk that cannot be parallelised across rows.
    float substitution_rhs[gdn_chunk];
    float new_value[gdn_chunk];

    for (int index = threadIdx.x; index < k_head * v_head; index += blockDim.x) {
        state_shared[index] = state[state_base + index];
    }
    __syncthreads();

    const int chunks = (tokens + gdn_chunk - 1) / gdn_chunk;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int start = chunk * gdn_chunk;

        if (threadIdx.x == 0) {
            float running = 0.0f;
            #pragma unroll
            for (int row = 0; row < gdn_chunk; ++row) {
                const int token = start + row;
                running += (token < tokens) ? decay[(long)token * num_v_heads + head] : 0.0f;
                cum_shared[row] = running;
                beta_shared[row] = (token < tokens) ? beta[(long)token * num_v_heads + head] : 0.0f;
            }
        }
        __syncthreads();

        const float chunk_decay = expf(cum_shared[gdn_chunk - 1]);

        for (int index = threadIdx.x; index < gdn_chunk * v_head; index += blockDim.x) {
            const int row = index / v_head;
            const int column = index % v_head;
            const int token = start + row;
            float total = 0.0f;
            if (token < tokens) {
                const float row_decay = expf(cum_shared[row]);
                const long source = ((long)token * num_k_heads + key_head) * k_head;
                for (int dim = 0; dim < k_head; ++dim) {
                    const float query_decayed = query[source + dim] * row_decay;
                    total += query_decayed * state_shared[dim * v_head + column];
                }
            }
            out_shared[index] = total;
        }
        __syncthreads();

        for (int row = 0; row < gdn_chunk; ++row) {
            const int dot_index = threadIdx.x / dot_threads;
            const int part = threadIdx.x % dot_threads;
            // A dot product needs both of its operands inside the sequence: the
            // row's own key is padding once the chunk has run past `tokens`, and
            // so is the row it is dotted against. `other_valid` is nested inside
            // `row_valid` so neither operand is ever addressed past the tensors.
            const bool row_valid = start + row < tokens;
            const bool other_valid = row_valid && start + dot_index < tokens;

            float partial = 0.0f;
            if (other_valid) {
                const long key_row = ((long)(start + row) * num_k_heads + key_head) * k_head;
                const long other = ((long)(start + dot_index) * num_k_heads + key_head) * k_head;
                for (int dim = part; dim < k_head; dim += dot_threads) {
                    const float against = key[key_row + dim];
                    if (dot_index < row) {
                        partial += against * key[other + dim];
                    } else {
                        partial += against * query[other + dim];
                    }
                }
            }
            // The `dot_threads` lanes of one dot product sit in adjacent lanes,
            // so the butterfly needs no mask arithmetic.
            partial += __shfl_xor_sync(0xffffffffu, partial, 1);
            partial += __shfl_xor_sync(0xffffffffu, partial, 2);

            if (part == 0) {
                if (dot_index < row) {
                    ut_row[dot_index] = beta_shared[row] * partial *
                        expf(cum_shared[row] - cum_shared[dot_index]);
                } else {
                    intra_column[dot_index] = partial *
                        expf(cum_shared[dot_index] - cum_shared[row]);
                }
            }
            __syncthreads();

            if (value_active) {
                float total = 0.0f;
                if (row_valid) {
                    const long source = ((long)(start + row) * num_v_heads + head) * v_head +
                        value_column;
                    total = value[source] * beta_shared[row];
                }
                #pragma unroll
                for (int inner = 0; inner < gdn_chunk; ++inner) {
                    if (inner < row) {
                        total -= ut_row[inner] * substitution_rhs[inner];
                    }
                }
                substitution_rhs[row] = total;
            }
            if (key_active) {
                float total = 0.0f;
                if (row_valid) {
                    const long key_row = ((long)(start + row) * num_k_heads + key_head) * k_head;
                    total = key[key_row + key_dimension] *
                        beta_shared[row] * expf(cum_shared[row]);
                }
                #pragma unroll
                for (int inner = 0; inner < gdn_chunk; ++inner) {
                    if (inner < row) {
                        total -= ut_row[inner] * substitution_rhs[inner];
                    }
                }
                substitution_rhs[row] = total;
                cumdecay_row[key_dimension] = total;
            }
            __syncthreads();

            if (value_active) {
                float predicted = 0.0f;
                for (int dim = 0; dim < k_head; ++dim) {
                    predicted += cumdecay_row[dim] * state_shared[dim * v_head + value_column];
                }
                const float fresh = substitution_rhs[row] - predicted;
                new_value[row] = fresh;
                #pragma unroll
                for (int ahead = 0; ahead < gdn_chunk; ++ahead) {
                    if (ahead >= row) {
                        out_shared[ahead * v_head + value_column] +=
                            intra_column[ahead] * fresh;
                    }
                }
            }
            __syncthreads();
        }

        if (value_active) {
            for (int dim = 0; dim < k_head; ++dim) {
                state_shared[dim * v_head + value_column] *= chunk_decay;
            }
            const float last_decay = cum_shared[gdn_chunk - 1];
            for (int row = 0; row < gdn_chunk; ++row) {
                if (start + row < tokens) {
                    const long source = ((long)(start + row) * num_k_heads + key_head) * k_head;
                    const float decayed = expf(last_decay - cum_shared[row]);
                    for (int dim = 0; dim < k_head; ++dim) {
                        const float key_decayed = key[source + dim] * decayed;
                        state_shared[dim * v_head + value_column] +=
                            key_decayed * new_value[row];
                    }
                }
            }
            for (int row = 0; row < gdn_chunk; ++row) {
                const int token = start + row;
                if (token < tokens) {
                    out[((long)token * num_v_heads + head) * v_head + value_column] =
                        __float2bfloat16_rn(out_shared[row * v_head + value_column]);
                }
            }
        }
        __syncthreads();
    }

    for (int index = threadIdx.x; index < k_head * v_head; index += blockDim.x) {
        state[state_base + index] = state_shared[index];
    }
}
