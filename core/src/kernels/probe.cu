// Smallest possible kernel: proves the nvcc -> cubin -> embed -> driver-load
// pipeline works end to end. Kept as `extern "C"` so the cubin exposes the
// unmangled symbol `probe_fill` for the Zig loader.

extern "C" __global__ void probe_fill(float *out, float value) {
    out[threadIdx.x] = value;
}
