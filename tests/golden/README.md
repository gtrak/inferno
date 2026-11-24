# Golden fixtures — the Hugging Face Qwen4Exp reference oracle

`inv`-engine kernels and CPU references are validated against **the Hugging Face reference
implementation of Qwen4Exp**, never against another part of this repo. These fixtures are
produced *only* by that reference implementation (`tools/ref-trace/`), so a wrong oracle
cannot be hidden behind a matching wrong engine.

Provenance (also in `tools/ref-trace/reference/VERSION.txt`):

| item | value |
|---|---|
| reference | `transformers` **git main @ d8f10a82213a3a9db48385aba3a4b57ab535de50** (5.24.0.dev0) |
| vendored copy | `tools/ref-trace/reference/{modeling,configuration}_qwen4_exp.py`, byte-identical to the installed package (checked by `run.sh`) |
| torch | 2.15.0+cpu, float32, `torch.set_num_threads(1)` |
| attention | `_attn_implementation="eager"` |
| kernels | `flash-linear-attention` and `causal_conv1d` are **not** installed, so every reference path is the plain-PyTorch one (`torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`, `causal_conv1d_fn`, `causal_conv1d_update`) |

## The thread count is part of the fixture

`torch.set_num_threads(1)` is not incidental: the committed tensors are that
setting's output, and no other setting reproduces them bit for bit.

Measured on this machine (torch 2.15.0+cpu, float32, eager; the default thread
count here is 12) by re-running the trace and comparing every tensor against the
committed one:

| condition | observed max abs difference |
|---|---|
| re-run at `set_num_threads(1)` — the control, 2 runs | `0.0`: all 112 float tensors of each section are bit-identical |
| default threads, `final/logits` | 4.5e-8 (prefill), 3.0e-8 (decode) |
| default threads, worst tensor in a section | 2.4e-7 (`layer2/attn/indexer/q`, `decode/layer3/gdn/g`) |
| default threads, worst on the scaled bound `max abs err / max(1e-6, max abs golden)` | 2.9e-7 (`layer3/moe/out`), 2.4e-7 (`decode/layer3/moe/shared_out`) |

To reproduce: load `tiny/trace.safetensors`, run the same 12-token prefill and
1-token decode as `trace.py` (same input ids, the `decode_input_id` from the
fixture metadata) *without* calling `torch.set_num_threads(1)`, then take
`max |new - committed|` per tensor. `trace.py` writes the fixture in place, so
copy it aside first. The 8 integer / bool tensors per section (masks, router
indices, n-gram ids, indexer ids and counts) matched the fixture in that run
too, so the drift is confined to the float tensors.

**Consequence for consumers:** compare these fixtures bit-exactly *only* under
`set_num_threads(1)`; anywhere else use `rtol`/`atol ≈ 1e-6` in fp32, which sits
above the worst figure observed here (2.4e-7). `core/src/ref/tests.zig` applies
the scaled bound `2e-5`.

## Files

| file | what it is |
|---|---|
| `tiny/config.json` | the tiny `Qwen4ExpTextConfig` (text-only, 4 layers) |
| `tiny/model.safetensors` | same weights in the `transformers` layout (`model.*`, grouped expert tensors are split by `save_pretrained` itself) |
| `tiny/checkpoint.safetensors` | same weights in the **published checkpoint layout** (`model.language_model.*`, per-expert `mlp.experts.E.{gate,up,down}_proj.weight`, sharded n-gram table) — this is what the loader's key mapping is tested against |
| `tiny/trace.safetensors` | 240 golden tensors: 120 for the 12-token prefill, 120 for the 1-token decode step |
| `tiny/generation_config.json` | written by `save_pretrained`; not used by the trace |

`tiny/model.safetensors` is what `from_pretrained` reads; `tiny/checkpoint.safetensors` is
only for loader-mapping tests. `make_tiny.py` asserts that the two agree tensor for tensor.

## Regenerate

```bash
cd tools/ref-trace
rm -rf .venv && ./run.sh          # true bootstrap (venv + fixtures); omit `rm -rf` to refresh
```

`run.sh` creates `.venv` with `uv venv -p 3.13`, installs CPU torch, installs
`transformers @ git+https://github.com/huggingface/transformers`, fails loudly if
`Qwen4ExpForCausalLM`/`Qwen4ExpForConditionalGeneration` do not import, checks the vendored
copy against the installed package, writes `reference/VERSION.txt`, then runs
`make_tiny.py` and `trace.py`. The weights are bit-reproducible across runs (and
`model.safetensors`/`checkpoint.safetensors` are byte-identical); `trace.safetensors` holds
bit-identical *tensors*, but its header serialises the metadata map in hash order, so its
bytes can differ between runs — compare tensors, not checksums.

## The tiny model

| feature | tiny value |
|---|---|
| layers | 4: `linear_attention, linear_attention, indexed_attention(full), linear_attention` |
| hidden / heads | 128 / 4 q heads × 32, 1 kv head (GQA group 4) |
| GDN | 2 k heads, 6 v heads (group 3), 16+16 head dims, conv kernel 4 |
| hyper-connections | `hc_count=2` → 256-wide streams, `hc_lowrank=16` |
| MoE | 4 experts, top-2, intermediate 32, shared expert 32 |
| PLE | `ple_layer_ids=[2]` → text layer index 1, `ple_embed_dim=32` (4 n-gram heads × 8), conv kernel 4 dilated by `ngram_size=3` |
| n-grams | `ngram_size=3`, `heads_per_ngram=2`, vocab base 1000, `split_ngram_parts=2` → 4064 × 8 table |
| QSA indexer | 2 q heads × 16, 1 kv head, budget 8, compress ratio 2 (4 blocks) |
| rope | partial 0.25 → 8 rotary dims, `mrope_section=[2,1,1]` |
| vocab | 256, untied `lm_head` |

Every parameter except the `1 + w`-centred RMSNorm weights is non-zero, and
`ple.conv1d.weight` (which the reference initialises to zero) is re-randomised, so no
branch of the model is vacuous.

## Trace contract (`tiny/trace.safetensors`)

Metadata: `transformers_sha`, `transformers_version`, `torch_version`, `attn_implementation`,
`seed="0"`, `prefill_input_ids`, `decode_input_id`.

Prefill input ids (literal, vocab 256): `[13, 4, 199, 42, 7, 88, 211, 30, 96, 155, 61, 240]`,
positions 0..11. The decode step then feeds the argmax of the last prefill logits (**169**)
through the same cache.

| tensor | shape | notes |
|---|---|---|
| `embed/output` | `[1,12,128]` | token embedding, before the HC repeat |
| `rotary/cos`, `rotary/sin` | `[1,12,8]` | `position_embeddings` after the mRoPE recomposition |
| `final/mixer_out` | `[1,12,128]` | output of the final hyper-connection mixer (= model output, there is no final norm) |
| `final/logits` | `[1,12,256]` | `lm_head(mixer_out)` |
| `layer{L}/in` | `[1,12,256]` | HC tensor entering the decoder layer |
| `layer{L}/out` | `[1,12,256]` | HC tensor leaving the decoder layer |
| `layer{L}/hc_attn/mixed` | `[1,12,128]` | single-stream input to the attn / linear-attn block |
| `layer{L}/hc_attn/inject_weights` | `[1,12,2]` | per-stream injection weights (`2*sigmoid(...)`) |
| `layer{L}/hc_mlp/*` | as above | same for the MoE block |
| `layer{L}/moe/router_logits` | `[12,4]` | flat rows (the reference flattens to `[batch*seq, ...]`) |
| `layer{L}/moe/router_weights` | `[12,2]` | top-k probabilities, renormalised (`norm_topk_prob`) |
| `layer{L}/moe/router_indices` | `[12,2]` int64 | |
| `layer{L}/moe/shared_out` | `[12,128]` | shared expert output **after** its sigmoid gate (pre-gate value also stored as `moe/shared_out_ungated`, gate logit as `moe/shared_gate_logit`) |
| `layer{L}/moe/expert_out` | `[12,128]` | summed top-k expert output only |
| `layer{L}/moe/out` | `[1,12,128]` | block output = `expert_out + shared_out`, reshaped |
| `layer{L}/gdn/q`,`k` | `[1,12,2,16]` | post-conv, pre-l2norm (the l2norm happens inside the delta rule) |
| `layer{L}/gdn/v` | `[1,12,6,16]` | |
| `layer{L}/gdn/g` | `[1,12,6]` | `-exp(A_log)*softplus(a + dt_bias)`, log space, ≤ 0 |
| `layer{L}/gdn/beta` | `[1,12,6]` | `sigmoid(b)` |
| `layer{L}/gdn/conv_out` | `[1,160,12]` | causal conv output (pre-transpose, post-SiLU) |
| `layer{L}/gdn/core_out` | `[1,12,6,16]` | gated delta rule output, before the gated norm |
| `layer{L}/gdn/z` | `[1,12,96]` | `in_proj_z` output (same data as the `[1,12,6,16]` view) |
| `layer{L}/gdn/normed` | `[72,16]` | `RMSNormGated(core_out, z)` flattened to rows |
| `layer{L}/gdn/out` | `[1,12,128]` | `out_proj(normed)` |
| `layer{L}/gdn/conv_state` | `[1,160,4]` | final conv state from the cache |
| `layer{L}/gdn/recurrent_state` | `[1,6,16,16]` | final recurrent state (extra, not in the contract) |
| `layer{L}/attn/q` | `[1,4,12,32]` | post `q_norm`, post RoPE |
| `layer{L}/attn/k`, `v` | `[1,1,12,32]` | post `k_norm` / plain projection, post RoPE (k) and post cache append |
| `layer{L}/attn/gate` | `[1,12,128]` | **pre-sigmoid** gate logits (the reference multiplies by `sigmoid(gate)`) |
| `layer{L}/attn/mask` | `[1,1,12,12]` bool | the QSA selection mask returned by the indexer |
| `layer{L}/attn/weights` | `[1,4,12,12]` | post-softmax attention probabilities (extra) |
| `layer{L}/attn/indexer/q` | `[1,12,2,16]` | indexer queries, post `q_layernorm`, post RoPE (extra) |
| `layer{L}/attn/indexer/raw_keys` | `[1,12,16]` | per-token indexer keys (post cache update) |
| `layer{L}/attn/indexer/pooled_keys` | `[1,12,6,16]` | mean-pooled + `k_layernorm` + RoPE'd block keys, **zero-padded**; valid blocks = `attn/indexer/num_blocks` |
| `layer{L}/attn/indexer/scores` | `[1,12,6]` | `relu(q·k).sum()/sqrt(16)` per block, zero-padded the same way |
| `layer{L}/attn/indexer/num_blocks` | `[1,12]` int32 | number of complete compressed blocks visible to each query (extra) |
| `layer{L}/attn/indexer/selected` | `[1,12,9]` int32 | selected token ids per query, `-1` padded (== `selected_token_indices`) |
| `layer{L}/attn/out` | `[1,12,128]` | `o_proj(sigmoid(gate) * attn_output)` |
| `layer{L}/ple/ngram_ids` | `[1,12,4]` int64 | hashed ids fed to the n-gram table |
| `layer{L}/ple/embeddings` | `[1,12,32]` | concatenated n-gram embeddings (all heads) |
| `layer{L}/ple/key_normed` | `[1,12,256]` | `norm_key(key_proj(embeddings))`, still `[hc*H]`-flat |
| `layer{L}/ple/query_normed` | `[1,12,256]` | `norm_query(hidden_states)` (extra) |
| `layer{L}/ple/value` | `[1,12,128]` | `value_proj(embeddings)` |
| `layer{L}/ple/gate` | `[1,12,2,1]` | per-stream signed-sqrt gate, before the sigmoid |
| `layer{L}/ple/gated_value` | `[1,12,256]` | `(sigmoid(gate) * value)` flattened over streams |
| `layer{L}/ple/gated_value_normed` | `[1,12,256]` | `norm_conv(gated_value)` (extra) |
| `layer{L}/ple/conv_out` | `[1,256,12]` | dilated depthwise conv after SiLU (pre-transpose) |
| `layer{L}/ple/out` | `[1,12,256]` | `gated_value + conv_out`, added to the HC stream inside the layer |

The decode section repeats every prefill name with a `decode/` prefix for a single
position (`T=1`, position 12), including `decode/final/logits` `[1,1,256]` and the
post-step `decode/layer{L}/gdn/{conv,recurrent}_state`. `decode/rotary/cos` is `[1,13,8]`:
the indexer receives the full position history, the attention layers slice the last row.

`trace.py` refuses to write the file unless (a) every one of those names is present,
(b) all tensors are finite, and (c) the *patched* model (the capture hooks, plus the one
copied `Qwen4ExpTextQSAIndexer.forward` body) reproduces the unpatched reference's prefill
**and** decode logits bit-exactly.

## Checkpoint key mapping (`tiny/checkpoint.safetensors`)

Exactly the published layout (`~/models/qwen38-flash-next-nvfp4/model.safetensors.index.json`),
verified against that index: mapping every HF parameter through these rules reproduces all
74,923 weight tensors of the real checkpoint (with its 512 experts, 128 n-gram shards, 48
layers).

| HF (`transformers`) parameter | checkpoint tensor(s) |
|---|---|
| `model.<rest>` | `model.language_model.<rest>` |
| `lm_head.weight` | `lm_head.weight` (unchanged) |
| `model.layers.N.mlp.experts.gate_up_proj` `[E,2I,H]` | `model.language_model.layers.N.mlp.experts.{e}.gate_proj.weight` `[I,H]` (rows `[:I]`) and `...{e}.up_proj.weight` `[I,H]` (rows `[I:]`), for `e` in `0..E-1` |
| `model.layers.N.mlp.experts.down_proj` `[E,H,I]` | `...layers.N.mlp.experts.{e}.down_proj.weight` `[H,I]` |
| `model.layers.N.ple.ple_embedding.ngram_embedding.weight` `[V,d]` | `...ngram_embedding.shard_{s}.weight` for `s` in `0..split_ngram_parts-1`, contiguous row blocks |
| everything else (`embed_tokens`, `linear_attn.*`, `self_attn.*`, `mlp.gate`/`shared_expert`/`shared_expert_gate`, `*_hyper_connection.*`, `ple.*`, `hyper_connection_mixer.*`) | `model.language_model.` + the same path |

Two details that surprise loaders:

* the text model has **no final RMSNorm** — `model.language_model.hyper_connection_mixer`
  is the last module before `lm_head`;
* the PLE `layer_multipliers`, `ngram_heads_vocab_sizes` and `ngram_heads_offsets` buffers
  **are** in the checkpoint, and `linear_attn.{A_log,dt_bias}` are parameters, not buffers;
* `mtp.*` (1 MTP layer) and `model.visual.*` (vision tower) are out of scope for I-04 and
  are not present in these fixtures.

## `inspect.py`

```bash
cd tools/ref-trace
python inspect.py                                  # the tiny fixture config
python inspect.py ~/models/qwen38-flash-next-nvfp4/config.json   # the real 48-layer shape
```

It builds the model on the `meta` device (no allocation — the real shape is 176.9 G
parameters) and prints the module tree, then every parameter/buffer with its shape, dtype
and the checkpoint tensor name(s) it comes from.

### Tiny config output

```
# ignoring config keys not present in Qwen4ExpTextConfig: ['number_of_conv_states']
# config tests/golden/tiny/config.json
# model_type qwen4_exp_text layers 4 hidden 128 layer_types ['indexed_attention', 'linear_attention'] ple_layer_ids [2] experts 4 top_k 2 hc_count 2

=== module tree ===
<root>: Qwen4ExpForCausalLM
model: Qwen4ExpTextModel
  embed_tokens: Embedding
  layers [4 x Qwen4ExpTextDecoderLayer]: ModuleList
    0: Qwen4ExpTextDecoderLayer
      linear_attn: Qwen4ExpTextGatedDeltaNet
        conv1d: Conv1d
        norm: Qwen4ExpTextRMSNormGated
        out_proj: Linear
        in_proj_qkv: Linear
        in_proj_z: Linear
        in_proj_b: Linear
        in_proj_a: Linear
      mlp: Qwen4ExpTextSparseMoeBlock
        gate: Qwen4ExpTextTopKRouter
        experts: Qwen4ExpTextExperts
          act_fn: SiLUActivation
        shared_expert: Qwen4ExpTextMLP
          gate_proj: Linear
          up_proj: Linear
          down_proj: Linear
          act_fn: SiLUActivation
        shared_expert_gate: Linear
      attn_hyper_connection: Qwen4ExpTextGatedResidual
        hc_norm: Qwen4ExpTextRMSNorm
        input_mix_weight_down: Linear
        input_mix_weight_up: Linear
        block_inject_weight: Linear
      mlp_hyper_connection: Qwen4ExpTextGatedResidual
        hc_norm: Qwen4ExpTextRMSNorm
        input_mix_weight_down: Linear
        input_mix_weight_up: Linear
        block_inject_weight: Linear
    1: Qwen4ExpTextDecoderLayer
      linear_attn: Qwen4ExpTextGatedDeltaNet
        conv1d: Conv1d
        norm: Qwen4ExpTextRMSNormGated
        out_proj: Linear
        in_proj_qkv: Linear
        in_proj_z: Linear
        in_proj_b: Linear
        in_proj_a: Linear
      mlp: Qwen4ExpTextSparseMoeBlock
        gate: Qwen4ExpTextTopKRouter
        experts: Qwen4ExpTextExperts
          act_fn: SiLUActivation
        shared_expert: Qwen4ExpTextMLP
          gate_proj: Linear
          up_proj: Linear
          down_proj: Linear
          act_fn: SiLUActivation
        shared_expert_gate: Linear
      ple: Qwen4ExpTextPLELayer
        ple_embedding: Qwen4ExpTextNGramEmbedding
          ngram_embedding: Embedding
        key_proj: Linear
        value_proj: Linear
        norm_key: Qwen4ExpTextRMSNorm
        norm_query: Qwen4ExpTextRMSNorm
        norm_conv: Qwen4ExpTextRMSNorm
        conv1d: Conv1d
      attn_hyper_connection: Qwen4ExpTextGatedResidual
        hc_norm: Qwen4ExpTextRMSNorm
        input_mix_weight_down: Linear
        input_mix_weight_up: Linear
        block_inject_weight: Linear
      mlp_hyper_connection: Qwen4ExpTextGatedResidual
        hc_norm: Qwen4ExpTextRMSNorm
        input_mix_weight_down: Linear
        input_mix_weight_up: Linear
        block_inject_weight: Linear
    2: Qwen4ExpTextDecoderLayer
      self_attn: Qwen4ExpTextAttention
        q_proj: Linear
        k_proj: Linear
        v_proj: Linear
        o_proj: Linear
        q_norm: Qwen4ExpTextRMSNorm
        k_norm: Qwen4ExpTextRMSNorm
        indexer: Qwen4ExpTextQSAIndexer
          index_qk_proj: Linear
          q_layernorm: Qwen4ExpTextRMSNorm
          k_layernorm: Qwen4ExpTextRMSNorm
      mlp: Qwen4ExpTextSparseMoeBlock
        gate: Qwen4ExpTextTopKRouter
        experts: Qwen4ExpTextExperts
          act_fn: SiLUActivation
        shared_expert: Qwen4ExpTextMLP
          gate_proj: Linear
          up_proj: Linear
          down_proj: Linear
          act_fn: SiLUActivation
        shared_expert_gate: Linear
      attn_hyper_connection: Qwen4ExpTextGatedResidual
        hc_norm: Qwen4ExpTextRMSNorm
        input_mix_weight_down: Linear
        input_mix_weight_up: Linear
        block_inject_weight: Linear
      mlp_hyper_connection: Qwen4ExpTextGatedResidual
        hc_norm: Qwen4ExpTextRMSNorm
        input_mix_weight_down: Linear
        input_mix_weight_up: Linear
        block_inject_weight: Linear
    3: Qwen4ExpTextDecoderLayer
      linear_attn: Qwen4ExpTextGatedDeltaNet
        conv1d: Conv1d
        norm: Qwen4ExpTextRMSNormGated
        out_proj: Linear
        in_proj_qkv: Linear
        in_proj_z: Linear
        in_proj_b: Linear
        in_proj_a: Linear
      mlp: Qwen4ExpTextSparseMoeBlock
        gate: Qwen4ExpTextTopKRouter
        experts: Qwen4ExpTextExperts
          act_fn: SiLUActivation
        shared_expert: Qwen4ExpTextMLP
          gate_proj: Linear
          up_proj: Linear
          down_proj: Linear
          act_fn: SiLUActivation
        shared_expert_gate: Linear
      attn_hyper_connection: Qwen4ExpTextGatedResidual
        hc_norm: Qwen4ExpTextRMSNorm
        input_mix_weight_down: Linear
        input_mix_weight_up: Linear
        block_inject_weight: Linear
      mlp_hyper_connection: Qwen4ExpTextGatedResidual
        hc_norm: Qwen4ExpTextRMSNorm
        input_mix_weight_down: Linear
        input_mix_weight_up: Linear
        block_inject_weight: Linear
  rotary_emb: Qwen4ExpTextRotaryEmbedding
  hyper_connection_mixer: Qwen4ExpTextGatedResidual
    hc_norm: Qwen4ExpTextRMSNorm
    input_mix_weight_down: Linear
    input_mix_weight_up: Linear
lm_head: Linear

=== parameters ===
name                                                           shape                    dtype        checkpoint tensors
model.embed_tokens.weight                                      (256, 128)               torch.float32 model.language_model.embed_tokens.weight
model.layers.0.linear_attn.dt_bias                             (6,)                     torch.float32 model.language_model.layers.0.linear_attn.dt_bias
model.layers.0.linear_attn.A_log                               (6,)                     torch.float32 model.language_model.layers.0.linear_attn.A_log
model.layers.0.linear_attn.conv1d.weight                       (160, 1, 4)              torch.float32 model.language_model.layers.0.linear_attn.conv1d.weight
model.layers.0.linear_attn.norm.weight                         (16,)                    torch.float32 model.language_model.layers.0.linear_attn.norm.weight
model.layers.0.linear_attn.out_proj.weight                     (128, 96)                torch.float32 model.language_model.layers.0.linear_attn.out_proj.weight
model.layers.0.linear_attn.in_proj_qkv.weight                  (160, 128)               torch.float32 model.language_model.layers.0.linear_attn.in_proj_qkv.weight
model.layers.0.linear_attn.in_proj_z.weight                    (96, 128)                torch.float32 model.language_model.layers.0.linear_attn.in_proj_z.weight
model.layers.0.linear_attn.in_proj_b.weight                    (6, 128)                 torch.float32 model.language_model.layers.0.linear_attn.in_proj_b.weight
model.layers.0.linear_attn.in_proj_a.weight                    (6, 128)                 torch.float32 model.language_model.layers.0.linear_attn.in_proj_a.weight
model.layers.0.mlp.gate.weight                                 (4, 128)                 torch.float32 model.language_model.layers.0.mlp.gate.weight
model.layers.0.mlp.experts.gate_up_proj                        (4, 64, 128)             torch.float32 8 per-expert tensors: model.language_model.layers.0.mlp.experts.0.gate_proj.weight .. model.language_model.layers.0.mlp.experts.3.up_proj.weight
model.layers.0.mlp.experts.down_proj                           (4, 128, 32)             torch.float32 4 per-expert tensors: model.language_model.layers.0.mlp.experts.0.down_proj.weight .. model.language_model.layers.0.mlp.experts.3.down_proj.weight
model.layers.0.mlp.shared_expert.gate_proj.weight              (32, 128)                torch.float32 model.language_model.layers.0.mlp.shared_expert.gate_proj.weight
model.layers.0.mlp.shared_expert.up_proj.weight                (32, 128)                torch.float32 model.language_model.layers.0.mlp.shared_expert.up_proj.weight
model.layers.0.mlp.shared_expert.down_proj.weight              (128, 32)                torch.float32 model.language_model.layers.0.mlp.shared_expert.down_proj.weight
model.layers.0.mlp.shared_expert_gate.weight                   (1, 128)                 torch.float32 model.language_model.layers.0.mlp.shared_expert_gate.weight
model.layers.0.attn_hyper_connection.hc_norm.weight            (256,)                   torch.float32 model.language_model.layers.0.attn_hyper_connection.hc_norm.weight
model.layers.0.attn_hyper_connection.input_mix_weight_down.weight (16, 256)                torch.float32 model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight
model.layers.0.attn_hyper_connection.input_mix_weight_up.weight (256, 16)                torch.float32 model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight
model.layers.0.attn_hyper_connection.block_inject_weight.weight (2, 256)                 torch.float32 model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight
model.layers.0.mlp_hyper_connection.hc_norm.weight             (256,)                   torch.float32 model.language_model.layers.0.mlp_hyper_connection.hc_norm.weight
model.layers.0.mlp_hyper_connection.input_mix_weight_down.weight (16, 256)                torch.float32 model.language_model.layers.0.mlp_hyper_connection.input_mix_weight_down.weight
model.layers.0.mlp_hyper_connection.input_mix_weight_up.weight (256, 16)                torch.float32 model.language_model.layers.0.mlp_hyper_connection.input_mix_weight_up.weight
model.layers.0.mlp_hyper_connection.block_inject_weight.weight (2, 256)                 torch.float32 model.language_model.layers.0.mlp_hyper_connection.block_inject_weight.weight
model.layers.1.linear_attn.dt_bias                             (6,)                     torch.float32 model.language_model.layers.1.linear_attn.dt_bias
model.layers.1.linear_attn.A_log                               (6,)                     torch.float32 model.language_model.layers.1.linear_attn.A_log
model.layers.1.linear_attn.conv1d.weight                       (160, 1, 4)              torch.float32 model.language_model.layers.1.linear_attn.conv1d.weight
model.layers.1.linear_attn.norm.weight                         (16,)                    torch.float32 model.language_model.layers.1.linear_attn.norm.weight
model.layers.1.linear_attn.out_proj.weight                     (128, 96)                torch.float32 model.language_model.layers.1.linear_attn.out_proj.weight
model.layers.1.linear_attn.in_proj_qkv.weight                  (160, 128)               torch.float32 model.language_model.layers.1.linear_attn.in_proj_qkv.weight
model.layers.1.linear_attn.in_proj_z.weight                    (96, 128)                torch.float32 model.language_model.layers.1.linear_attn.in_proj_z.weight
model.layers.1.linear_attn.in_proj_b.weight                    (6, 128)                 torch.float32 model.language_model.layers.1.linear_attn.in_proj_b.weight
model.layers.1.linear_attn.in_proj_a.weight                    (6, 128)                 torch.float32 model.language_model.layers.1.linear_attn.in_proj_a.weight
model.layers.1.mlp.gate.weight                                 (4, 128)                 torch.float32 model.language_model.layers.1.mlp.gate.weight
model.layers.1.mlp.experts.gate_up_proj                        (4, 64, 128)             torch.float32 8 per-expert tensors: model.language_model.layers.1.mlp.experts.0.gate_proj.weight .. model.language_model.layers.1.mlp.experts.3.up_proj.weight
model.layers.1.mlp.experts.down_proj                           (4, 128, 32)             torch.float32 4 per-expert tensors: model.language_model.layers.1.mlp.experts.0.down_proj.weight .. model.language_model.layers.1.mlp.experts.3.down_proj.weight
model.layers.1.mlp.shared_expert.gate_proj.weight              (32, 128)                torch.float32 model.language_model.layers.1.mlp.shared_expert.gate_proj.weight
model.layers.1.mlp.shared_expert.up_proj.weight                (32, 128)                torch.float32 model.language_model.layers.1.mlp.shared_expert.up_proj.weight
model.layers.1.mlp.shared_expert.down_proj.weight              (128, 32)                torch.float32 model.language_model.layers.1.mlp.shared_expert.down_proj.weight
model.layers.1.mlp.shared_expert_gate.weight                   (1, 128)                 torch.float32 model.language_model.layers.1.mlp.shared_expert_gate.weight
model.layers.1.ple.ple_embedding.ngram_embedding.weight        (4064, 8)                torch.float32 2 shards: model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight .. model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight
model.layers.1.ple.key_proj.weight                             (256, 32)                torch.float32 model.language_model.layers.1.ple.key_proj.weight
model.layers.1.ple.value_proj.weight                           (128, 32)                torch.float32 model.language_model.layers.1.ple.value_proj.weight
model.layers.1.ple.norm_key.weight                             (256,)                   torch.float32 model.language_model.layers.1.ple.norm_key.weight
model.layers.1.ple.norm_query.weight                           (256,)                   torch.float32 model.language_model.layers.1.ple.norm_query.weight
model.layers.1.ple.norm_conv.weight                            (256,)                   torch.float32 model.language_model.layers.1.ple.norm_conv.weight
model.layers.1.ple.conv1d.weight                               (256, 1, 4)              torch.float32 model.language_model.layers.1.ple.conv1d.weight
model.layers.1.attn_hyper_connection.hc_norm.weight            (256,)                   torch.float32 model.language_model.layers.1.attn_hyper_connection.hc_norm.weight
model.layers.1.attn_hyper_connection.input_mix_weight_down.weight (16, 256)                torch.float32 model.language_model.layers.1.attn_hyper_connection.input_mix_weight_down.weight
model.layers.1.attn_hyper_connection.input_mix_weight_up.weight (256, 16)                torch.float32 model.language_model.layers.1.attn_hyper_connection.input_mix_weight_up.weight
model.layers.1.attn_hyper_connection.block_inject_weight.weight (2, 256)                 torch.float32 model.language_model.layers.1.attn_hyper_connection.block_inject_weight.weight
model.layers.1.mlp_hyper_connection.hc_norm.weight             (256,)                   torch.float32 model.language_model.layers.1.mlp_hyper_connection.hc_norm.weight
model.layers.1.mlp_hyper_connection.input_mix_weight_down.weight (16, 256)                torch.float32 model.language_model.layers.1.mlp_hyper_connection.input_mix_weight_down.weight
model.layers.1.mlp_hyper_connection.input_mix_weight_up.weight (256, 16)                torch.float32 model.language_model.layers.1.mlp_hyper_connection.input_mix_weight_up.weight
model.layers.1.mlp_hyper_connection.block_inject_weight.weight (2, 256)                 torch.float32 model.language_model.layers.1.mlp_hyper_connection.block_inject_weight.weight
model.layers.2.self_attn.q_proj.weight                         (256, 128)               torch.float32 model.language_model.layers.2.self_attn.q_proj.weight
model.layers.2.self_attn.k_proj.weight                         (32, 128)                torch.float32 model.language_model.layers.2.self_attn.k_proj.weight
model.layers.2.self_attn.v_proj.weight                         (32, 128)                torch.float32 model.language_model.layers.2.self_attn.v_proj.weight
model.layers.2.self_attn.o_proj.weight                         (128, 128)               torch.float32 model.language_model.layers.2.self_attn.o_proj.weight
model.layers.2.self_attn.q_norm.weight                         (32,)                    torch.float32 model.language_model.layers.2.self_attn.q_norm.weight
model.layers.2.self_attn.k_norm.weight                         (32,)                    torch.float32 model.language_model.layers.2.self_attn.k_norm.weight
model.layers.2.self_attn.indexer.index_qk_proj.weight          (48, 128)                torch.float32 model.language_model.layers.2.self_attn.indexer.index_qk_proj.weight
model.layers.2.self_attn.indexer.q_layernorm.weight            (16,)                    torch.float32 model.language_model.layers.2.self_attn.indexer.q_layernorm.weight
model.layers.2.self_attn.indexer.k_layernorm.weight            (16,)                    torch.float32 model.language_model.layers.2.self_attn.indexer.k_layernorm.weight
model.layers.2.mlp.gate.weight                                 (4, 128)                 torch.float32 model.language_model.layers.2.mlp.gate.weight
model.layers.2.mlp.experts.gate_up_proj                        (4, 64, 128)             torch.float32 8 per-expert tensors: model.language_model.layers.2.mlp.experts.0.gate_proj.weight .. model.language_model.layers.2.mlp.experts.3.up_proj.weight
model.layers.2.mlp.experts.down_proj                           (4, 128, 32)             torch.float32 4 per-expert tensors: model.language_model.layers.2.mlp.experts.0.down_proj.weight .. model.language_model.layers.2.mlp.experts.3.down_proj.weight
model.layers.2.mlp.shared_expert.gate_proj.weight              (32, 128)                torch.float32 model.language_model.layers.2.mlp.shared_expert.gate_proj.weight
model.layers.2.mlp.shared_expert.up_proj.weight                (32, 128)                torch.float32 model.language_model.layers.2.mlp.shared_expert.up_proj.weight
model.layers.2.mlp.shared_expert.down_proj.weight              (128, 32)                torch.float32 model.language_model.layers.2.mlp.shared_expert.down_proj.weight
model.layers.2.mlp.shared_expert_gate.weight                   (1, 128)                 torch.float32 model.language_model.layers.2.mlp.shared_expert_gate.weight
model.layers.2.attn_hyper_connection.hc_norm.weight            (256,)                   torch.float32 model.language_model.layers.2.attn_hyper_connection.hc_norm.weight
model.layers.2.attn_hyper_connection.input_mix_weight_down.weight (16, 256)                torch.float32 model.language_model.layers.2.attn_hyper_connection.input_mix_weight_down.weight
model.layers.2.attn_hyper_connection.input_mix_weight_up.weight (256, 16)                torch.float32 model.language_model.layers.2.attn_hyper_connection.input_mix_weight_up.weight
model.layers.2.attn_hyper_connection.block_inject_weight.weight (2, 256)                 torch.float32 model.language_model.layers.2.attn_hyper_connection.block_inject_weight.weight
model.layers.2.mlp_hyper_connection.hc_norm.weight             (256,)                   torch.float32 model.language_model.layers.2.mlp_hyper_connection.hc_norm.weight
model.layers.2.mlp_hyper_connection.input_mix_weight_down.weight (16, 256)                torch.float32 model.language_model.layers.2.mlp_hyper_connection.input_mix_weight_down.weight
model.layers.2.mlp_hyper_connection.input_mix_weight_up.weight (256, 16)                torch.float32 model.language_model.layers.2.mlp_hyper_connection.input_mix_weight_up.weight
model.layers.2.mlp_hyper_connection.block_inject_weight.weight (2, 256)                 torch.float32 model.language_model.layers.2.mlp_hyper_connection.block_inject_weight.weight
model.layers.3.linear_attn.dt_bias                             (6,)                     torch.float32 model.language_model.layers.3.linear_attn.dt_bias
model.layers.3.linear_attn.A_log                               (6,)                     torch.float32 model.language_model.layers.3.linear_attn.A_log
model.layers.3.linear_attn.conv1d.weight                       (160, 1, 4)              torch.float32 model.language_model.layers.3.linear_attn.conv1d.weight
model.layers.3.linear_attn.norm.weight                         (16,)                    torch.float32 model.language_model.layers.3.linear_attn.norm.weight
model.layers.3.linear_attn.out_proj.weight                     (128, 96)                torch.float32 model.language_model.layers.3.linear_attn.out_proj.weight
model.layers.3.linear_attn.in_proj_qkv.weight                  (160, 128)               torch.float32 model.language_model.layers.3.linear_attn.in_proj_qkv.weight
model.layers.3.linear_attn.in_proj_z.weight                    (96, 128)                torch.float32 model.language_model.layers.3.linear_attn.in_proj_z.weight
model.layers.3.linear_attn.in_proj_b.weight                    (6, 128)                 torch.float32 model.language_model.layers.3.linear_attn.in_proj_b.weight
model.layers.3.linear_attn.in_proj_a.weight                    (6, 128)                 torch.float32 model.language_model.layers.3.linear_attn.in_proj_a.weight
model.layers.3.mlp.gate.weight                                 (4, 128)                 torch.float32 model.language_model.layers.3.mlp.gate.weight
model.layers.3.mlp.experts.gate_up_proj                        (4, 64, 128)             torch.float32 8 per-expert tensors: model.language_model.layers.3.mlp.experts.0.gate_proj.weight .. model.language_model.layers.3.mlp.experts.3.up_proj.weight
model.layers.3.mlp.experts.down_proj                           (4, 128, 32)             torch.float32 4 per-expert tensors: model.language_model.layers.3.mlp.experts.0.down_proj.weight .. model.language_model.layers.3.mlp.experts.3.down_proj.weight
model.layers.3.mlp.shared_expert.gate_proj.weight              (32, 128)                torch.float32 model.language_model.layers.3.mlp.shared_expert.gate_proj.weight
model.layers.3.mlp.shared_expert.up_proj.weight                (32, 128)                torch.float32 model.language_model.layers.3.mlp.shared_expert.up_proj.weight
model.layers.3.mlp.shared_expert.down_proj.weight              (128, 32)                torch.float32 model.language_model.layers.3.mlp.shared_expert.down_proj.weight
model.layers.3.mlp.shared_expert_gate.weight                   (1, 128)                 torch.float32 model.language_model.layers.3.mlp.shared_expert_gate.weight
model.layers.3.attn_hyper_connection.hc_norm.weight            (256,)                   torch.float32 model.language_model.layers.3.attn_hyper_connection.hc_norm.weight
model.layers.3.attn_hyper_connection.input_mix_weight_down.weight (16, 256)                torch.float32 model.language_model.layers.3.attn_hyper_connection.input_mix_weight_down.weight
model.layers.3.attn_hyper_connection.input_mix_weight_up.weight (256, 16)                torch.float32 model.language_model.layers.3.attn_hyper_connection.input_mix_weight_up.weight
model.layers.3.attn_hyper_connection.block_inject_weight.weight (2, 256)                 torch.float32 model.language_model.layers.3.attn_hyper_connection.block_inject_weight.weight
model.layers.3.mlp_hyper_connection.hc_norm.weight             (256,)                   torch.float32 model.language_model.layers.3.mlp_hyper_connection.hc_norm.weight
model.layers.3.mlp_hyper_connection.input_mix_weight_down.weight (16, 256)                torch.float32 model.language_model.layers.3.mlp_hyper_connection.input_mix_weight_down.weight
model.layers.3.mlp_hyper_connection.input_mix_weight_up.weight (256, 16)                torch.float32 model.language_model.layers.3.mlp_hyper_connection.input_mix_weight_up.weight
model.layers.3.mlp_hyper_connection.block_inject_weight.weight (2, 256)                 torch.float32 model.language_model.layers.3.mlp_hyper_connection.block_inject_weight.weight
model.hyper_connection_mixer.hc_norm.weight                    (256,)                   torch.float32 model.language_model.hyper_connection_mixer.hc_norm.weight
model.hyper_connection_mixer.input_mix_weight_down.weight      (16, 256)                torch.float32 model.language_model.hyper_connection_mixer.input_mix_weight_down.weight
model.hyper_connection_mixer.input_mix_weight_up.weight        (256, 16)                torch.float32 model.language_model.hyper_connection_mixer.input_mix_weight_up.weight
lm_head.weight                                                 (256, 128)               torch.float32 lm_head.weight

=== buffers (persistent) ===
model.layers.1.ple.ple_embedding.layer_multipliers             (3,)                     torch.int64  model.language_model.layers.1.ple.ple_embedding.layer_multipliers
model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes       (4,)                     torch.int64  model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes
model.layers.1.ple.ple_embedding.ngram_heads_offsets           (4,)                     torch.int64  model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets

# parameters 645940 (0.00 GiB fp32, 0.00 GiB bf16)
```

## Gates

```bash
cd tools/ref-trace && ./run.sh

.venv/bin/python - <<'EOF'
import json, torch, safetensors.torch   # NOTE: `import safetensors` alone does not expose `.torch`
d = safetensors.torch.load_file("../../tests/golden/tiny/trace.safetensors")
need = ["embed/output","final/logits","layer0/gdn/core_out","layer2/attn/mask",
        "layer1/ple/out","layer2/moe/router_indices","layer0/out","layer3/out"]
missing = [k for k in need if k not in d]
assert not missing, missing
assert all(torch.isfinite(v).all() for v in d.values())
# non-vacuity
assert d["final/logits"].unique().numel() > 1
mask = d["layer2/attn/mask"]
assert mask.any() and (~mask).any(), "selection mask is not selective"
assert len({tuple(row) for row in d["layer2/moe/router_indices"].tolist()}) > 1, "routing is constant"
print("OK", len(d), "tensors")
EOF
du -sh ../../tests/golden/tiny      # must be < 25 MB
```

## Known limitations

* The traced reference uses the pure-PyTorch fallbacks (no `fla`, no `causal_conv1d`), so
  the numbers are the reference's semantics but not a bit-level target for fused kernels.
* `split_ngram_parts` is honoured at save time (the table is written as shards) but the
  reference never loads shards back from a directory — `from_pretrained` reads the
  unsharded `model.safetensors`. Only the checkpoint-layout copy is sharded.
* The MTP layer and the vision tower have no reference in `Qwen4ExpTextModel` and are not
  covered by I-04 fixtures.
