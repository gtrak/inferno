# I-012 — elementwise and structural kernels: norms, RoPE, activations, sampler

## Launch infrastructure (shared, already written — do not reinvent)

`core/src/hw/kernel.zig` exists on main and is the only way to load a cubin and
launch a kernel. Copy it into your worktree (`cp /home/gary/dev/inferno/core/src/hw/kernel.zig core/src/hw/`)
and add exactly TWO lines to the test block in `core/src/c_api.zig`:
`_ = @import("hw/kernel.zig");` and `_ = @import("kernels/<yours>.zig");`.

```zig
var module = try kernel.Module.load(cubin_bytes);   // decompress-free: bytes are embedded
defer module.unload();
var fn_ = try module.function("your_kernel");       // extern "C" name from the .cu
try fn_.setDynamicSharedBytes(64 * 1024);           // only if you need > 48 KiB/block
var params = [_]?*anyopaque{ @ptrCast(&ptr), @ptrCast(&n) };
try fn_.launch(.{ gx, 1, 1 }, .{ bx, 1, 1 }, shared_bytes, stream.handle, &params);
```

`core/build.zig` already globs `src/kernels/*.cu` into the embedded cubin table, so
adding a `.cu` file needs no build change. **Do not edit `core/build.zig`.**
There is no per-kernel build step: your tests run under `zig build test`.

## Working agreement
- Sole writer. tiger-style binding. Oracle: `core/src/ref/ops.zig`, `ref/rope.zig`,
  `ref/hc.zig`. Reference lines 136–205 (rope recomposition, RMSNorm,
  RMSNormGated), 643–686 (partial rotary), 1017–1053 (hyper-connections).
- **Do not edit `core/build.zig`.** Add exactly one line
  `_ = @import("kernels/elementwise.zig");` to the test block in `core/src/c_api.zig`.

## What to build
`core/src/kernels/elementwise.cu` + `.zig`:
1. `rmsnorm` — with `group_size` (the HC 10240-dim vector normalises per 2560
   group) and the `x*(1+w)` convention, f32 accumulate.
2. `rmsnorm_gated` — norm first, then multiply by `silu(gate)`.
3. `rope_partial_mrope` — cos/sin table for partial rotary (`0.25 × 256 = 64`
   dims rotated) with the interleaved mRoPE `[11,11,10]` recomposition, applied
   to q/k in place on a `[tokens, heads, 256]` layout; plus the standalone
   query-only variant the indexer needs.
4. `hyper_connection` — per-2560-group RMSNorm of the 10240-dim stream,
   low-rank down/up mix, `silu(x/4)`, sigmoid, weighted mean of the 4 streams;
   plus the `2*sigmoid(block_inject/4)` injection weights; plus the final
   `use_combine=false` mixer.
5. `sampler` — temperature, top-k, top-p, min-p, repetition penalty and a
   32-bit RNG (xoshiro128**), all in f32 on device, returning one token id plus
   its logprob. Must be **deterministic for a fixed seed** and must not require
   a host round-trip of the vocabulary.
6. `silu_mul`, `sigmoid_gate`, `softmax_f32`, `argmax_f32` helpers.

## Gates
```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 900 zig build test) ; echo CORETEST=$?
```
1. Every kernel matches its oracle within `1e-5` relative (f32 domain) and
   bf16 round-trip within `4e-3`.
2. RoPE: assert the rotated dims are exactly the first 64 and that dims 64..255
   are bit-identical to the input (a full-rotary bug is otherwise invisible in
   aggregate error).
3. Sampler: same seed ⇒ identical ids over 100 draws; `temperature=0` ⇒ argmax;
   top_k=1 ⇒ argmax; and the returned logprob matches a re-computed log-softmax.
4. **Non-vacuity**: print the sampled id distribution over 1000 draws at
   temperature 1.0 (must have >5 distinct ids) and the injected-weight range.

## Report format
Implemented / Changed files / Validation / Open risks (sampler determinism and
top-p tie handling) / Recommended next step.
