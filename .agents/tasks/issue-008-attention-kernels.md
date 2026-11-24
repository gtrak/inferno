# I-08 — indexed (QSA) attention kernels: prefill + decode

## Working agreement

- Sole writer in your worktree. tiger-style for Zig; CUDA C++ with named
  constants and checked launch configs.
- Algorithm reference: `tools/ref-trace/reference/modeling_qwen4_exp.py`
  lines **643–686** (`rotate_half` / partial RoPE), **688–792** (`QSAIndexer`),
  **806–829** (`eager_attention_forward`), **831–916** (`Attention`). Read them.
- Validation oracle: `core/src/ref/attention.zig` (I-05) plus the HF goldens
  `layer2/attn/*` in `tests/golden/tiny/trace.safetensors`.
- Real shapes: 24 q heads, 2 kv heads (GQA group 12), `head_dim = 256`,
  partial rotary on the first `256*0.25 = 64` dims, indexer: 4 q heads +
  1 kv head of dim 128, `budget 2048`, `compress_ratio 4` (⇒ `block_topk 512`),
  scale `1/sqrt(256)`, output gate `sigmoid(gate)` applied per q-element before
  `o_proj`. 12 of 48 layers use this path (`layer_types[i] == "indexed_attention"`).

## What to build

### 1. `core/src/kernels/attention.cu`

**a) `attn_prefill_sparse`** — the prefill path, driven by the indexer's
selection instead of a dense causal mask:

```
q: bf16 [M, 24, 256]        (already q_norm'd and RoPE'd by an earlier kernel)
kv_cache: bf16 [S_total, 2, 256] (k and v written by this kernel for the new rows)
selected: int32 [M, B]      (B = budget + compress_ratio - 1, -1 padded)
out: bf16 [M, 24, 256]
```

Online (streaming) softmax over the ≤B selected key positions only, grouped GQA
(each kv head serves 12 consecutive q heads), fp32 accumulation in registers.
Because the selection list *is* the mask, there is no causal mask in this kernel —
but assert on the host side that every selected index `< query_abs_pos` (the
indexer's guarantee) so a corrupted selection cannot leak future tokens.

**b) `attn_decode`** — one query token, KV cache of length S, same selection
semantics (a decode step selects ≤B cached positions plus the new one), plus:

**c) `indexer_scores`** — the QSA selection itself, which must run every step:

1. project the hidden state: `index_qk_proj` (TP-sharded) → split into
   `4*128` queries and `1*128` token keys;
2. per-head RMSNorm, RoPE on the queries only (partial rotary, mRoPE positions);
3. append the token key to the indexer key cache;
4. pool cached keys over `compress_ratio`-sized **complete** blocks (mean in f32,
   then cast back), RMSNorm the pooled keys, RoPE them with the block's **start**
   position;
5. `score = sum_h relu(q_h · k_pooled)` per block, `/ sqrt(128)`;
6. `topk(min(block_topk, n_blocks))` → expand blocks to token indices →
   append the trailing incomplete block's raw indices → write the `selected` list.

Do the top-k with a bitonic sort in shared memory per block-group; `block_topk`
≤ 512 so a simple two-pass radix or bitonic sort per query is fine.

**d) `rope_partial_mrope`** — cos/sin table generation + application for the
interleaved mRoPE (`mrope_section [11,11,10]`, rot dim 64) exactly as
`recomposition_frequencies`; used by both the main attention and the indexer.

### 2. `core/src/kernels/attention.zig`

Launch wrappers + a `zig build attntest` harness comparing against
`ref/attention.zig` at: `S_total ∈ {128, 1024, 4096}` × `M ∈ {1 (decode), 256, 1024}`,
with selection lists produced by our own `indexer_scores` **and** by a
brute-force reference top-k. Also a direct comparison of the whole layer's
`layer2/attn/out` against the HF golden.

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build attntest) ; echo ATTN=$?
```

Acceptance:
1. `attn_prefill_sparse` and `attn_decode` outputs match the f32 oracle with
   `max_abs_err/max|ref| <= 4e-3` (bf16 storage, fp32 accumulate — discuss in the
   report if you need a looser bound and why).
2. The indexer's `selected` lists must match the reference **exactly** (integer
   set equality after removing `-1` and sorting) for at least 3 queries; ties in
   the top-k must be broken identically to `torch.topk` (highest index wins on
   ties — verify against the reference rather than assuming).
3. Non-vacuity: print, for one query at S=4096, the number of selected indices
   (must be ≈ budget, not 0 and not S), and confirm the decode output for
   position p equals the prefill output row p when both use the same context
   (**a critical invariant**: decode must equal prefill at the same position,
   tolerance 4e-3).
4. Report achieved time per kernel; decode at S=4096 must be under 300 µs/head-layer
   (informational, not a gate).

## Report format

Implemented / Changed files / Validation (error table + the prefill-vs-decode
invariant result + selected-count evidence) / Open risks (anything ambiguous in
the indexer: tie-breaking, block pooling of partial blocks, mRoPE positions) /
Recommended next step.
