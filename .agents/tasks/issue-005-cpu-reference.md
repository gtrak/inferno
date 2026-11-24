# I-05 — CPU reference implementation of the full Qwen4Exp text graph (the oracle-follower)

## Why

This is the executable specification of the model in Zig, in fp32, straight from
the reference Python. Every GPU kernel in this project is validated against
**this** (and against the HF golden traces it reproduces). Get it exactly right
and the rest of the engine becomes mechanical.

## Working agreement

- Sole writer in your worktree. `.agents/skills/tiger-style/` is authoritative
  for every line of Zig you write (≥2 assertions per function, no recursion, no
  dynamic allocation after init, ≤70 lines per function, 80 columns, explicit
  integer widths, every `if` has an `else`).
- The only source of truth for the algorithm is the vendored reference:
  `tools/ref-trace/reference/modeling_qwen4_exp.py`. **Read the actual line
  ranges**, do not work from this spec's prose:
  - 136–158 `recomposition_frequencies` (mRoPE) · 160–186 `RMSNorm` (group_size,
    `x*(1+w)`) · 188–205 `RMSNormGated` · 220–262 `causal_conv1d_*`
  - 264–269 `l2norm` · **271–411 `torch_chunk_gated_delta_rule`** ·
    **413–482 `torch_recurrent_gated_delta_rule`** · 483–641 `GatedDeltaNet.forward`
  - 643–686 `rotate_half` / `apply_rotary_pos_emb` (partial rotary) ·
    **688–792 `QSAIndexer`** · 806–829 `eager_attention_forward` · 831–916 `Attention`
  - 918–933 `MLP` · 935–972 `Experts` · 974–993 `TopKRouter` · 995–1015 `SparseMoeBlock`
  - **1017–1053 `GatedResidual`** (hyper-connections) · 1055–1092 splitmix/primes ·
    **1094–1196 `NGramEmbedding`** · 1198–1272 `PLELayer`
  - 1274–1325 `DecoderLayer` · 1375–1520 `TextModel.forward`
- Goldens: `tests/golden/tiny/` (produced by I-04) with the tensor names listed
  in `.agents/tasks/issue-004-ref-trace.md` §3. Read `tests/golden/README.md`
  first, and `.agents/tasks/reviewer-followups.md` items 4–6 (they change how you
  compare and what you may touch).
- **Dependency note:** `core/src/weights/*` (issue I-003) is being written in a
  *different lane* and is **not in your tree**. Do not create or modify
  `core/src/weights/`. Instead write `core/src/ref/fixtures.zig`: a
  self-contained reader for `tests/golden/tiny/config.json` (`std.json`) and
  `tests/golden/tiny/checkpoint.safetensors` (8-byte little-endian header length
  + JSON header + raw bytes; dtypes present are BF16/F32/I64). It should expose
  `RefConfig` and a name→f32-tensor table. ~150 lines. A later issue consolidates
  it with the weights layer.
- You may add a `--greedy` mode to `tools/ref-trace/trace.py` (16 greedy tokens
  from the golden input, `torch.set_num_threads(1)`) and commit
  `tests/golden/tiny/greedy_ids.json` — nothing else in `tools/` is yours.
- Everything is **f32**. `allocation` only at `init`; all buffers are sized from
  `Config` and owned by a `Scratch` struct passed by pointer.

## What to build

```
core/src/ref/
  mod.zig          pub exports + Scratch allocator/init
  fixtures.zig     self-contained tiny-config + safetensors reader (see dependency note)
  ops.zig          rmsnorm, rmsnorm_gated, silu, sigmoid, softmax, l2norm, matmul, gemv, embed_gather
  conv.zig         causal depthwise conv1d (+ state update, dilation variant)
  rope.zig         partial RoPE + mRoPE recomposition
  gdn.zig          chunked + recurrent gated delta rule, GatedDeltaNet block
  attention.zig    indexer (QSA) selection + gated GQA attention
  moe.zig          router, expert loop, shared expert
  hc.zig           GatedResidual (hyper-connections)
  ple.zig          n-gram id derivation, embedding gather, PLE block
  layer.zig        DecoderLayer
  model.zig        full TextModel forward + lm_head + greedy generate
core/src/ref/tests.zig   golden comparison suite
```

### Staging (compile + test after every stage; do not write everything then debug)

1. `ops.zig` + `rope.zig` + `conv.zig`, with unit tests against hand-computed
   values (RMSNorm with `group_size`, `x*(1+w)`, RoPE partial-rotary on
   `head_dim*0.25 = 64` of 256, the mRoPE `H/W` interleave exactly as
   `recomposition_frequencies`, and the causal conv1d with `padding=k-1` then
   slice, plus the `silu` activation).
2. `hc.zig` — must reproduce `layer{L}/hc_attn/mixed`, `inject_weights`, and the
   final `hyper_connection_mixer` output (which has `use_combine=false`). Note
   `hc_norm` normalises **per `hidden_size` group** over the 10240-dim vector.
3. `moe.zig` — router softmax → top-k → `norm_topk_prob` renormalise (do the
   top-k on the **softmax probabilities**, in f32, exactly as the reference) →
   expert loop (`silu(gate)*up` then `down`, weighted, accumulated) → shared
   expert gated by `sigmoid(shared_expert_gate(x))`.
4. `gdn.zig` — implement **both** the chunked (prefill) and recurrent (decode)
   forms. Follow the reference arithmetic *exactly*, including: `repeat_interleave`
   of q/k heads 16→48 (each k-head serves 3 consecutive v-heads), `l2norm` with
   eps 1e-6, query scaling by `head_dim**-0.5` **after** l2norm, `beta = sigmoid(b)`,
   `g = -exp(A_log) * softplus(a + dt_bias)` (f32), the padding of the sequence
   to a multiple of chunk 64, `cumsum` of `g` in **log space**, the strictly-upper
   triangular mask set to `-inf` **before** `exp`, the UT transform solved by
   forward substitution (`unitriangular=True`), and the sequential chunk scan.
   Assert at the end: chunked and recurrent forms agree on the same input to
   < 1e-5 relative (this is a strong self-check the reference itself supports).
5. `attention.zig` — the indexer first (`index_qk_proj` → split into
   `index_n_heads*128` queries and `index_kv_heads*128` token keys → per-head
   RMSNorm → RoPE on queries → pool keys over `compress_ratio`-sized **visible**
   blocks (the trailing incomplete block is appended raw) → per-block
   `relu(q·k)` summed over heads, divided by `sqrt(128)`, top-`budget/compress_ratio`
   → token-level boolean mask). Then the gated attention: `q_proj` output is
   split into `(query, gate)`, `q_norm`/`k_norm` per head, partial RoPE,
   GQA with 24 q-heads / 2 kv-heads, causal mask AND the indexer mask,
   `attn_out * sigmoid(gate)` then `o_proj`. The temporary causal masking of a
   fully-masked query must produce a finite 0, not NaN.
6. `ple.zig` — n-gram id derivation exactly as `_build_layer_multipliers`,
   `_find_nth_prime_after`, `_shift_right_ignore_eos` (EOS-aware segments) and
   `torch.remainder` on **int64**; gather 16 rows; `key_proj`/`value_proj`;
   `norm_key`/`norm_query` grouping; the signed-sqrt gate
   `sign(g)*sqrt(max(|g|,1e-6))`; the dilated depthwise conv with
   `dilation = ngram_size`, `state_len = (k-1)*dilation`; and the residual add
   into the HC tensor. PLE is on **text layer index 1** for this checkpoint.
7. `layer.zig` / `model.zig` — assemble: embed → `repeat(1,1,hc_count)` →
   4 layers → final `hyper_connection_mixer` → `lm_head` (bf16 weights, f32
   matmul) → logits. Then `generate()`: greedy argmax loop with the caches
   (GDN recurrent state, conv states, KV, indexer keys, PLE n-gram context).

### Caches

`ref/model.zig` needs a `Cache` struct holding, per layer: conv state (10240×3),
recurrent state (48×128×128 f32 = 3.1 MB/layer), KV (2 heads × 256), indexer
keys (128), and for the PLE layer the n-gram context (2 ids) and the PLE conv
state. Decode with `seq_len == 1` must use the recurrent GDN form; prefill with
`seq_len > 1` uses the chunked form.

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && zig build refcheck) ; echo REFCHECK_EXIT=$?
```

`zig build check-goldens` (or `refcheck`) runs the golden suite and prints, per
tensor, `name  max_abs_err  max_rel_err`, failing if any `max_abs_err > 1e-4`
(scaled by the tensor's max magnitude: use `max_abs_err / max(1e-6, max|golden|)
<= 2e-5`) . At minimum these must be covered **and pass**:

```
embed/output, layer0/in, layer0/hc_attn/mixed, layer0/hc_attn/inject_weights,
layer0/gdn/q, layer0/gdn/k, layer0/gdn/g, layer0/gdn/beta, layer0/gdn/core_out,
layer0/gdn/out, layer0/moe/router_indices, layer0/moe/router_weights,
layer0/moe/out, layer0/out, layer1/ple/ngram_ids, layer1/ple/key_normed,
layer1/ple/out, layer2/attn/mask, layer2/attn/q, layer2/attn/indexer/selected,
layer2/attn/out, layer2/moe/expert_out, layer3/out, final/mixer_out, final/logits,
decode/final/logits
```

Plus, as a separate printed check: **greedy equivalence** — run
`model.generate` on the golden input for 16 tokens and compare token ids to the
ids produced by the HF reference (I-04 stores them as `tests/golden/tiny/greedy_ids.json`;
if it does not exist yet, generate it with a small addition to
`tools/ref-trace/trace.py` and mention that in your report).

Finally `assert` non-vacuity: print the number of selected tokens in
`layer2/attn/mask`, the top-k index set for 3 positions, and the max |logit|.
A run where these are degenerate is not a pass.

## Report format

Implemented / Changed files / Validation (paste the error table — the largest
per-tensor error is the headline) / Open risks (any place where the reference is
ambiguous and you had to choose; anything you could not match and why) /
Recommended next step.
