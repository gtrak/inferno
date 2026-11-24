# I-09 — Gated DeltaNet kernels: causal conv + chunked prefill scan + recurrent decode

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

- Sole writer in your worktree. tiger-style for Zig; CUDA C++ with named constants.
- Algorithm reference: `tools/ref-trace/reference/modeling_qwen4_exp.py`
  lines **220–262** (`causal_conv1d_*`), **264–269** (`l2norm`),
  **271–411** (`torch_chunk_gated_delta_rule`), **413–482**
  (`torch_recurrent_gated_delta_rule`), **483–641** (`GatedDeltaNet.forward`).
  Read them; the arithmetic order matters.
- Validation oracle: `core/src/ref/gdn.zig` (I-05) and the HF goldens
  `layer0/gdn/*`.
- Real shapes: 36 of the 48 layers. `key_dim = 16 heads × 128 = 2048`,
  `value_dim = 48 heads × 128 = 6144`, `conv_dim = 10240`, conv kernel 4,
  `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)` in f32,
  q/k l2-normalised (eps 1e-6) and q scaled by `1/sqrt(128)`, each k head serves
  3 consecutive v heads, gate `z` applied through `RMSNormGated` with a **silu**
  gate, then `out_proj`.

## What to build

### 1. `core/src/kernels/gdn.cu`

**a) `gdn_prefill`** — the chunked (WY / UT-transform) form, chunk 64:
- stage 1 (`intra`): per chunk, build `ut_system = (k_beta · kᵀ) * pairwise_decay`
  with the strictly-upper part `-inf` **before** `exp`, then solve the unit
  lower-triangular systems by forward substitution for `new_values` and
  `k_cumdecay` (implement as a per-warp sequential substitution over 64 rows —
  this is the only inherently sequential part, keep the 128-wide RHS in registers
  and accept the latency);
- stage 2 (`scan`): sequential over chunks per (sequence, v-head), updating the
  128×128 fp32 recurrent state: `S ← S*chunk_decay + kᵀ·v_new`, and writing
  `out = q·S_prev + intra_attn·v_new`;
- fold the decay into q and k once per chunk exactly as the reference does
  (`q *= exp(cum_decay)`, `k *= exp(cum_decay_last - cum_decay)`).
- Write the final recurrent state back to the state pool.

**b) `gdn_decode`** — the recurrent single-token form, one block per v-head:
state 128×128 fp32 in shared memory (64 KiB — assert it fits), decay, `kv_mem`
reduction, delta update, output. 48 v-heads × 36 layers per token.

**c) `gdn_conv`** — the causal depthwise conv (kernel 4, silu) with the conv-state
ring buffer update, for prefill (whole chunk) and decode (single step).

**d) `gdn_gate_norm`** — `RMSNormGated(out, z)` (norm first, then multiply by
`silu(z)`) with the per-v-head 128-dim weight.

### 2. `core/src/kernels/gdn.zig`

Launch wrappers + a test suite (reached through the `c_api.zig` import; there is no
per-kernel build step) comparing against `ref/gdn.zig` for
`S ∈ {1, 63, 64, 65, 128, 1024}` (the non-multiples of 64 exercise the padding
path — **do not skip them**).

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build test) ; echo CORETEST=$?
```

Acceptance:
1. `max_abs_err / max|ref| <= 4e-3` for `gdn_prefill` and `gdn_decode` at every
   length above, under this **dtype contract** (corrected 2026-11-24 — the original
   "(bf16 I/O, fp32 state)" wording was wrong):

   | tensor | dtype | why |
   |---|---|---|
   | `query`, `key`, `value` | **f32** | the reference transposes and `.to(torch.float32)` these before the delta rule and keeps them f32 through `l2norm` (`modeling_qwen4_exp.py:309-313`); bf16 storage here was an engine artifact, not a model property |
   | `beta`, `decay`, recurrent state | f32 | the rule's arithmetic |
   | conv input/output, `out`, `z` | bf16 | activations crossing kernel boundaries |

   Storing q/k in bf16 cost 20 KiB/token/layer and was the sole cause of the prefill
   comparison reading 8.7e-3 instead of 2.35e-3. Issue I-013 must code against the
   table above.
2. **The central invariant:** for the same input sequence, running `gdn_prefill`
   over `S` tokens and running `gdn_decode` `S` times must agree within `4e-3`
   at every position, including the final recurrent state (`<= 1e-3` relative).
   Print the worst position and error. This catches the vast majority of
   chunking/decay/padding bugs.
3. `gdn_conv` matches `F.conv1d(padding=k-1)[:, :, :S]` + `silu` exactly, and the
   conv state after a prefill of `S` tokens equals the state after `S` decodes.
4. Non-vacuity: print `beta` and `g` ranges (must be in `(0,1)` and `< 0`
   respectively — a zeroed `g` means the decay path is untested), and confirm the
   output differs between two different inputs.
5. Informational: report decode time per layer at S=4096 context (36 layers
   total must stay well under the decode budget in THEORY.md §2).

## Report format

Implemented / Changed files / Validation (error table + the prefill-vs-recurrent
invariant + gate/`beta` ranges) / Open risks (the triangular solve is the risky
part — say exactly how you implemented and validated it) / Recommended next step.
