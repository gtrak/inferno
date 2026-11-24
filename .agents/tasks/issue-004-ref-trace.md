# I-04 — `tools/ref-trace`: HF reference + tiny random model + golden traces

## Why

Every kernel and every CPU reference implementation in this project is judged
against **the Hugging Face reference implementation of Qwen4Exp**, never against
another part of our own code. This issue builds that oracle. It is the single
most important correctness asset in the repo; a wrong oracle silently validates
a wrong engine.

## Working agreement

- Sole writer in your worktree. Skills are authoritative:
  `.agents/skills/tiger-style/` (N/A for Python, but keep it simple and explicit),
  `plan.md` (architecture), `THEORY.md` (numbers).
- Network access IS available for this issue (PyPI + GitHub); use it.
- `uv` is at `/home/gary/.local/bin/uv`. Python 3.14 is the system default but
  **create the venv with `uv venv -p 3.13`** so torch wheels are guaranteed.
- The reference implementation source is already fetched at
  `/tmp/ref/modeling_qwen4_exp.py` and
  `/tmp/ref/configuration_qwen4_exp.py` (also mirrored into
  `tools/ref-trace/reference/` by you). Read
  `modeling_qwen4_exp.py` — it is the ground truth for the architecture.

## What to build

```
tools/ref-trace/
  reference/{modeling_qwen4_exp.py,configuration_qwen4_exp.py}   vendored copy + VERSION.txt (git sha)
  pyproject.toml            deps: torch (CPU), transformers (git main), safetensors, numpy
  make_tiny.py              builds a tiny random-init Qwen4Exp checkpoint
  trace.py                  runs the reference and dumps golden tensors
  inspect.py                prints the module tree + parameter names of a config
  run.sh                    venv bootstrap + make_tiny + trace, idempotent
tests/golden/README.md      names + provenance + how to regenerate
tests/golden/tiny/          config.json, model.safetensors, trace.safetensors (committed)
```

### 1. Environment (`run.sh`)

`uv venv -p 3.13 .venv` inside `tools/ref-trace`, then
`uv pip install torch --index-url https://download.pytorch.org/whl/cpu`,
`uv pip install -e .` (pyproject pins `transformers @ git+https://github.com/huggingface/transformers`).
`run.sh` must be re-runnable from scratch and fail loudly if
`from transformers import Qwen4ExpForConditionalGeneration` (or the text-only
`Qwen4ExpForCausalLM`) does not import. Record the resolved transformers commit
into `tools/ref-trace/reference/VERSION.txt`.

### 2. Tiny model (`make_tiny.py`)

Build a `Qwen4ExpTextConfig` (text-only) small enough for CPU in seconds, with
**every feature exercised**: at least one `linear_attention` layer, at least one
`full_attention`/indexed layer, at least one PLE layer, `hc_count >= 2`, MoE with
≥4 experts and top-k ≥2, and a small n-gram table. Suggested starting point
(adjust only as the reference requires):

```
vocab_size=256, hidden_size=128, num_hidden_layers=4,
layer_types=["linear_attention","linear_attention","full_attention","linear_attention"],
num_attention_heads=4, num_key_value_heads=1, head_dim=32,
linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=16,
linear_value_head_dim=16, linear_conv_kernel_dim=4,
num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
shared_expert_intermediate_size=32,
hc_count=2, hc_lowrank=16,
ple_layer_ids=[2], ple_embed_dim=32, ple_conv_kernel_size=4,
ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=1000,
make_ngram_vocab_size_divisible_by=8, split_ngram_parts=2, seed=1234,
indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16,
indexer_budget=8, indexer_compress_ratio=2,
rope_parameters={rope_type:"default", rope_theta=10000.0,
                 partial_rotary_factor=0.25, mrope_section=[2,1,1],
                 mrope_interleaved=True},
norm_topk_prob=True, output_gate_type="sigmoid", rms_norm_eps=1e-6,
tie_word_embeddings=False, max_position_embeddings=512
```

Constraints you must respect (the reference raises otherwise): `head_dim *
partial_rotary_factor / 2 == sum(mrope_section)`; `num_attention_heads %
num_key_value_heads == 0`; `linear_num_value_heads % linear_num_key_heads == 0`;
`hidden_size % …group sizes == 0`; `indexer_budget % indexer_compress_ratio == 0`.
**Note:** `ple_layer_ids` are 1-based layer numbers, so `[2]` puts PLE on text
layer index 1 (this matches the real checkpoint, which has
`model.language_model.layers.1.ple.*`).

Procedure:
1. `torch.manual_seed(0)`, construct the model, `model.eval()`, cast to **float32**.
2. Overwrite all parameters with small deterministic random values where a
   default init would produce zeros/degeneracy (e.g. the RMSNorm weights are
   initialised to zeros and used as `1+w`; leave that alone, but make sure
   `A_log`, `dt_bias`, `layer_multipliers`, `ngram_heads_*` buffers are the
   reference's own values, not re-randomised).
3. Save with `model.save_pretrained("tests/golden/tiny")` **and** write a
   checkpoint-layout copy `tests/golden/tiny/checkpoint.safetensors` whose keys
   match the real HF repo naming (`model.language_model.layers.N.mlp.experts.E.gate_proj.weight`,
   `lm_head.weight`, `model.language_model.hyper_connection_mixer.input_mix_weight_down.weight`, …)
   so our loader's key mapping is testable against a real-shaped checkpoint.
   Document the mapping in `tests/golden/README.md`.
4. Assert: forward runs, logits shape `[1, T, vocab]`, no NaNs, and two
   consecutive forward passes are bit-identical (determinism check).

### 3. Golden traces (`trace.py`)

Use `_attn_implementation="eager"`. Input: fixed `input_ids` of length 12
(a literal list of ids, so it is reproducible), positions 0..11.

Capture and store **float32** tensors with these exact names (add more if
useful, never remove one):

- `embed/output` — token embedding before HC repeat.
- `rotary/cos`, `rotary/sin` — `position_embeddings`.
- Per layer `L` in 0..3, named `layer{L}/*`:
  - `in` (HC tensor input to the decoder layer, `[1,T,hc*H]`)
  - `hc_attn/mixed` (single-stream input to the attn/linear-attn block),
    `hc_attn/inject_weights`
  - for `linear_attention` layers: `gdn/q`, `gdn/k`, `gdn/v`, `gdn/g`,
    `gdn/beta`, `gdn/core_out`, `gdn/z`, `gdn/normed`, `gdn/out`,
    `gdn/conv_out`, `gdn/conv_state` (final)
  - for `full_attention` layers: `attn/q`, `attn/k`, `attn/v`, `attn/gate`,
    `attn/mask` (the boolean selection mask the indexer returns),
    `attn/indexer/raw_keys`, `attn/indexer/pooled_keys`,
    `attn/indexer/scores`, `attn/indexer/selected`, `attn/out`
  - `hc_mlp/mixed`, `hc_mlp/inject_weights`, `moe/router_logits`,
    `moe/router_weights`, `moe/router_indices`, `moe/shared_out`,
    `moe/expert_out`, `moe/out`
  - `out` (HC tensor after the layer)
  - for the PLE layer: `ple/ngram_ids`, `ple/embeddings`, `ple/key_normed`,
    `ple/value`, `ple/gate`, `ple/gated_value`, `ple/conv_out`, `ple/out`
- `final/mixer_out`, `final/logits`.

Implementation: wrap module `forward` with `torch.nn.modules.module.register_module_forward_hook`
(filtered by module type/name) or monkeypatch the specific classes — either is
fine, but the *names above are the contract*. Save with
`safetensors.torch.save_file(..., metadata={"transformers_sha": ..., "seed": "0"})`
into `tests/golden/tiny/trace.safetensors`.

Also record a **decode-step trace**: prefill the 12 tokens with
`use_cache=True`, then run 1 more token with the cache and store
`decode/{layer{L}/*,final/logits}` for that single position. This is what
validates our incremental path.

### 4. `inspect.py`

`python inspect.py [config.json]` prints the module tree with parameter names and
shapes — the mapping reference for the real 48-layer model. Include the output
for the tiny config in `tests/golden/README.md`.

## Gates

```
cd tools/ref-trace && ./run.sh            # from scratch: venv, tiny model, traces
.venv/bin/python - <<'EOF'
import safetensors, json, torch
d = safetensors.torch.load_file("../../tests/golden/tiny/trace.safetensors")
need = ["embed/output","final/logits","layer0/gdn/core_out","layer2/attn/mask",
        "layer1/ple/out","layer2/moe/router_indices","layer0/out","layer3/out"]
missing = [k for k in need if k not in d]
assert not missing, missing
assert all(torch.isfinite(v).all() for v in d.values())
print("OK", len(d), "tensors")
EOF
du -sh ../../tests/golden/tiny      # must be < 25 MB
```

Also verify the fixtures are **non-vacuous**: `final/logits` must have more than
one distinct value, `layer2/attn/mask` must contain both `True` and `False`, and
`layer2/moe/router_indices` must not be constant across positions. Print those
checks in the gate.

## Report format

Implemented / Changed files / Validation (paste the gate output) / Open risks
(especially: any reference-implementation behaviour you could not reproduce, and
the exact transformers commit you traced) / Recommended next step.
