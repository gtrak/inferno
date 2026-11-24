// Smallest possible kernel: proves the nvcc -> cubin -> embed pipeline works.
// Kept as `extern "C"` so the cubin exposes the unmangled symbol `probe_fill`.

extern "C" __global__ void probe_fill(float *out, float value) {
    out[threadIdx.x] = value;
}
