// Per-Layer Embedding device code: the cast of the gathered n-gram rows, the
// key/value projections, the signed-sqrt gate, and the dilated depthwise conv.
//
// Everything here is f32 in and f32 out, because the CPU reference
// (`src/ref/ple.zig`) and the HF golden trace both keep this block in f32; the
// engine stores the block output in bf16 only on the way into the residual, and
// that rounding is the caller's (`src/kernels/ple.zig` documents the tolerance
// per stage). The one packed format is the n-gram table itself, which the
// checkpoint stores as F8_E4M3 rows of `head_dim` bytes with one per-tensor
// scale; the oracle fixture stores the same rows as f32, so `ple_cast_rows`
// takes the element format as an argument instead of hard-coding one.
//
// The gather is host I/O (16 scattered rows per token out of a 51 GB table), so
// no kernel here touches the table: `ple_cast_rows` reads the rows the host
// staged into one contiguous pinned buffer, which is exactly one H2D copy per
// token batch.
//
// Shape contract, asserted host-side in `ple.zig`: `head_dim > 0`, `tokens > 0`,
// `hidden_size % ple_threads` need not be 0 (every loop is bounds-checked), and
// the conv state length is at most `ple_state_max`.

#if !defined(__CUDA_ARCH_SPECIFIC__)
#error "inferno kernels must be compiled with -arch=sm_120a"
#endif

#include <cstdint>

#include <cuda_fp8.h>

// Threads per block for every kernel in this file.
static constexpr int ple_threads = 256;

// Warp width, and the warps one block holds.
static constexpr int ple_warp = 32;
static constexpr int ple_warps = ple_threads / ple_warp;

// Largest conv state length the ring update keeps in registers:
// `(ple_conv_kernel_size - 1) * ngram_size` is 9 for both the tiny fixture and
// the real checkpoint.
static constexpr int ple_state_max = 64;

// `ple_cast_rows` element formats: the checkpoint's F8_E4M3 rows and the f32
// rows of the golden fixture.
static constexpr int ple_format_f32 = 0;
static constexpr int ple_format_e4m3 = 1;

// One E4M3 (FN) code as f32. The value is exact in f32, and the hardware
// conversion is the finite format the checkpoint uses, so this agrees with
// `weights/quant.zig` `f8_e4m3_to_f32` bit for bit.
__device__ __forceinline__ float ple_e4m3_value(unsigned char code) {
    const __half decoded = __nv_cvt_fp8_to_halfraw(
        static_cast<__nv_fp8_storage_t>(code), __NV_E4M3);
    return __half2float(decoded);
}

// Block-wide sum. Every thread of the block must reach both barriers, so this is
// only valid when the whole block executes it unconditionally.
__device__ __forceinline__ float ple_block_sum(float value, float *scratch) {
    const int lane = static_cast<int>(threadIdx.x) % ple_warp;
    const int warp = static_cast<int>(threadIdx.x) / ple_warp;
    for (int offset = ple_warp / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    value = (warp == 0 && lane < ple_warps) ? scratch[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = ple_warp / 2; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) {
            scratch[ple_warps] = value;
        }
    }
    __syncthreads();
    return scratch[ple_warps];
}

// `rows[i] * scale` for every staged row element: the gather hands over raw
// table bytes, and this is the one place that decodes them.
extern "C" __global__ void __launch_bounds__(ple_threads)
ple_cast_rows(const unsigned char *__restrict__ rows, float *__restrict__ out,
              int count, int head_dim, int format, float scale) {
    const int total = count * head_dim;
    const float *widen = reinterpret_cast<const float *>(rows);
    for (int index = static_cast<int>(blockIdx.x) * ple_threads +
                         static_cast<int>(threadIdx.x);
         index < total; index += static_cast<int>(gridDim.x) * ple_threads) {
        // `format` is one of the two codes below; a third would read the f32
        // path, so it is rejected here rather than silently decoded.
        assert(format == ple_format_f32 || format == ple_format_e4m3);
        const float value =
            (format == ple_format_e4m3) ? ple_e4m3_value(rows[index]) : widen[index];
        out[index] = value * scale;
    }
}

// `out[row][col] = sum_i x[row][i] * weight[col][i]`, the `nn.Linear` layout
// (weight is `[out_features, in_features]`).
//
// One warp per output column: the weight row is read 32 lanes at a time, so it
// is fully coalesced, and the activation row is re-read by every warp of the
// block but stays in L1 for the whole block (2560 f32 = 10 KB).
extern "C" __global__ void __launch_bounds__(ple_threads)
ple_project(const float *__restrict__ x, const float *__restrict__ weight,
            float *__restrict__ out, int rows, int in_features,
            int out_features) {
    const int lane = static_cast<int>(threadIdx.x) % ple_warp;
    const int warp = static_cast<int>(threadIdx.x) / ple_warp;
    const int row = static_cast<int>(blockIdx.y);
    const int column = static_cast<int>(blockIdx.x) * ple_warps + warp;
    if (row >= rows || column >= out_features) {
        return;
    }
    const float *x_row = x + static_cast<size_t>(row) * in_features;
    const float *w_row = weight + static_cast<size_t>(column) * in_features;
    float sum = 0.0f;
    for (int i = lane; i < in_features; i += ple_warp) {
        sum = fmaf(x_row[i], w_row[i], sum);
    }
    for (int offset = ple_warp / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        out[static_cast<size_t>(row) * out_features + column] = sum;
    }
}

// Per-stream RMSNorm of the projected key and of the hidden state, the
// signed-sqrt gate, and the gated value.
//
// One block per token. The reference normalises both operands over
// `hidden_size`-sized groups (`group_size=hidden_size`), so each stream is its
// own group and is handled in turn; `1 + weight` is the reference's RMSNorm
// spelling.
extern "C" __global__ void __launch_bounds__(ple_threads)
ple_gate(const float *__restrict__ key, const float *__restrict__ hidden,
         const float *__restrict__ norm_key,
         const float *__restrict__ norm_query, float *__restrict__ key_normed,
         float *__restrict__ query_normed, float *__restrict__ gate_out,
         const float *__restrict__ value, float *__restrict__ gated_value,
         int hidden_size, int streams, float eps) {
    __shared__ float reduction[ple_warps + 1];
    const int token = static_cast<int>(blockIdx.x);
    const size_t hyper = static_cast<size_t>(hidden_size) * streams;
    const size_t base = static_cast<size_t>(token) * hyper;
    const float divisor = sqrtf(static_cast<float>(hidden_size));
    for (int stream = 0; stream < streams; ++stream) {
        // `offset` is the row of this (token, stream) in the [tokens, hyper]
        // buffers; the norm weights are one row per token, so they are indexed
        // by the stream only.
        const size_t offset = base + static_cast<size_t>(stream) * hidden_size;
        const size_t weight_offset = static_cast<size_t>(stream) * hidden_size;
        float key_squares = 0.0f;
        float query_squares = 0.0f;
        for (int i = static_cast<int>(threadIdx.x); i < hidden_size;
             i += ple_threads) {
            const float k = key[offset + i];
            const float q = hidden[offset + i];
            key_squares = fmaf(k, k, key_squares);
            query_squares = fmaf(q, q, query_squares);
        }
        const float key_scale =
            rsqrtf(ple_block_sum(key_squares, reduction) / hidden_size + eps);
        const float query_scale =
            rsqrtf(ple_block_sum(query_squares, reduction) / hidden_size + eps);
        float dot = 0.0f;
        for (int i = static_cast<int>(threadIdx.x); i < hidden_size;
             i += ple_threads) {
            const float k = key[offset + i] * key_scale *
                            (1.0f + norm_key[weight_offset + i]);
            const float q = hidden[offset + i] * query_scale *
                            (1.0f + norm_query[weight_offset + i]);
            key_normed[offset + i] = k;
            query_normed[offset + i] = q;
            dot = fmaf(k, q, dot);
        }
        const float scaled = ple_block_sum(dot, reduction) / divisor;
        const float magnitude = fmaxf(fabsf(scaled), 1e-6f);
        const float gate = (scaled < 0.0f ? -1.0f : 1.0f) * sqrtf(magnitude);
        const float weight = 1.0f / (1.0f + expf(-gate));
        if (threadIdx.x == 0u) {
            gate_out[static_cast<size_t>(token) * streams + stream] = gate;
        }
        for (int i = static_cast<int>(threadIdx.x); i < hidden_size;
             i += ple_threads) {
            gated_value[offset + i] =
                weight * value[static_cast<size_t>(token) * hidden_size + i];
        }
        // The next stream reuses the reduction buffer.
        __syncthreads();
    }
}

// `norm_conv` over the flattened gated value: one group per stream, per token.
extern "C" __global__ void __launch_bounds__(ple_threads)
ple_norm_conv(const float *__restrict__ gated_value,
              const float *__restrict__ weight, float *__restrict__ out,
              int hidden_size, int streams, float eps) {
    __shared__ float reduction[ple_warps + 1];
    const int token = static_cast<int>(blockIdx.x);
    const size_t hyper = static_cast<size_t>(hidden_size) * streams;
    const size_t base = static_cast<size_t>(token) * hyper;
    for (int stream = 0; stream < streams; ++stream) {
        // As in `ple_gate`: the norm weight has one row per token, not one per
        // token and stream.
        const size_t offset = base + static_cast<size_t>(stream) * hidden_size;
        const size_t weight_offset = static_cast<size_t>(stream) * hidden_size;
        float squares = 0.0f;
        for (int i = static_cast<int>(threadIdx.x); i < hidden_size;
             i += ple_threads) {
            const float x = gated_value[offset + i];
            squares = fmaf(x, x, squares);
        }
        const float scale =
            rsqrtf(ple_block_sum(squares, reduction) / hidden_size + eps);
        for (int i = static_cast<int>(threadIdx.x); i < hidden_size;
             i += ple_threads) {
            out[offset + i] = gated_value[offset + i] * scale *
                              (1.0f + weight[weight_offset + i]);
        }
        __syncthreads();
    }
}

// The dilated depthwise conv, `silu`, and the injection into the residual.
//
// The window is `[conv_state | normed]`, so the conv of a fresh prefill reads
// the zeroed state (the reference pads with zeros) and a decode step reads the
// ring buffer the previous step left. One thread per channel, so every tap load
// is coalesced across the block.
extern "C" __global__ void __launch_bounds__(ple_threads)
ple_conv_apply(const float *__restrict__ gated_value,
               const float *__restrict__ normed, const float *__restrict__ taps,
               const float *__restrict__ state, float *__restrict__ out,
               float *__restrict__ hyper, int tokens, int channels, int kernel,
               int dilation, int state_len) {
    const int channel =
        static_cast<int>(blockIdx.x) * ple_threads + static_cast<int>(threadIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    if (channel >= channels) {
        return;
    }
    const float *channel_taps = taps + static_cast<size_t>(channel) * kernel;
    const float *channel_state = state + static_cast<size_t>(channel) * state_len;
    float total = 0.0f;
    for (int j = 0; j < kernel; ++j) {
        const int position = token + j * dilation;
        const float value =
            (position < state_len)
                ? channel_state[position]
                : normed[static_cast<size_t>(position - state_len) * channels + channel];
        total = fmaf(channel_taps[j], value, total);
    }
    const size_t index = static_cast<size_t>(token) * channels + channel;
    const float conv = total / (1.0f + expf(-total));
    const float value = gated_value[index] + conv;
    out[index] = value;
    // Each (token, channel) is written by exactly one thread, so the residual
    // update needs no atomic.
    hyper[index] = hyper[index] + value;
}

// Keeps the last `state_len` columns of `[state | normed]` as the next state.
// One thread per channel reads its whole ring before writing any of it, so the
// left shift never overwrites a value it still needs.
extern "C" __global__ void __launch_bounds__(ple_threads)
ple_conv_state(const float *__restrict__ normed, float *__restrict__ state,
               int tokens, int channels, int state_len) {
    const int channel =
        static_cast<int>(blockIdx.x) * ple_threads + static_cast<int>(threadIdx.x);
    if (channel >= channels) {
        return;
    }
    float *channel_state = state + static_cast<size_t>(channel) * state_len;
    float keep[ple_state_max];
    for (int k = 0; k < state_len; ++k) {
        const int position = tokens + k;
        keep[k] = (position < state_len)
                      ? channel_state[position]
                      : normed[static_cast<size_t>(position - state_len) * channels + channel];
    }
    for (int k = 0; k < state_len; ++k) {
        channel_state[k] = keep[k];
    }
}
