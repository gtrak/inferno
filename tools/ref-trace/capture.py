"""Golden-tensor capture plumbing for the HF Qwen4Exp reference implementation.

The contract (issue I-04) is a set of tensor *names*; this module produces them by watching
the reference implementation, not by re-implementing it:

  * global module forward hooks capture everything that crosses a module boundary
    (embeddings, rotary, hyper-connections, router, experts, shared expert, norms, ...);
  * `torch.nn.Linear`/`Conv1d`/`RMSNorm` instances are dispatched by their qualified name,
    and the two or three values the reference derives from a module output with only
    view/activation ops (the attention gate, the PLE gate) are recomputed and *asserted*
    equal to the value the reference actually uses;
  * the reference's free functions (`causal_conv1d_fn`, `torch_chunk_gated_delta_rule`,
    `torch_recurrent_gated_delta_rule`, `eager_attention_forward`) are wrapped to capture
    their arguments and return values.

`Qwen4ExpTextQSAIndexer.forward` is the single exception: its block pooling happens in a
per-(batch, query) python loop with no module boundary, so the body is re-stated below
verbatim from `reference/modeling_qwen4_exp.py` with `layer_put()` calls inserted.
`trace.py` asserts that the patched model reproduces the *unpatched* model's logits
bit-exactly, so a drifted copy cannot silently pass.
"""

import inspect as pyinspect
import math
import re

import torch
import torch.nn.functional as F

from transformers.models.qwen4_exp import modeling_qwen4_exp as q4


# Active recorder; set by `patch()` and used by the free-function wrappers, which are not
# handed the module they were called from.
_REC = None

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _snapshot(value, where):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{where}: expected a tensor, got {type(value).__name__}")
    # Clone: several of these tensors are views of buffers the reference mutates in place.
    return value.detach().clone().contiguous()


class Recorder:
    """Name -> tensor store with a switchable name prefix ("", "decode/")."""

    def __init__(self, model):
        self.tensors = {}
        self.prefix = ""
        self.layer = None
        self.names = {module: name for name, module in model.named_modules()}
        self.by_name = {name: module for name, module in model.named_modules()}

    def put(self, name, value):
        key = self.prefix + name
        if key in self.tensors:
            raise AssertionError(f"duplicate capture for {key!r}")
        self.tensors[key] = _snapshot(value, key)

    def layer_put(self, module, sub, value):
        self.put(f"layer{self.layer_of(module)}/{sub}", value)

    def get(self, module, sub):
        return self.tensors[self.prefix + f"layer{self.layer_of(module)}/{sub}"]

    def layer_of(self, module):
        match = _LAYER_RE.search(self.names[module])
        if match is None:
            raise AssertionError(f"{self.names[module]} is not inside a decoder layer")
        layer = int(match.group(1))
        declared = getattr(module, "layer_idx", layer)
        if declared != layer:
            raise AssertionError(f"{self.names[module]}: layer_idx={declared} but name says {layer}")
        return layer

    def parent(self, module):
        return self.by_name[self.names[module].rsplit(".", 1)[0]]


class _Section:
    def __init__(self, recorder, prefix):
        self.recorder = recorder
        self.prefix = prefix

    def __enter__(self):
        self.saved = self.recorder.prefix
        self.recorder.prefix = self.prefix
        self.recorder.layer = None

    def __exit__(self, *exc):
        self.recorder.prefix = self.saved
        self.recorder.layer = None
        return False


def section(recorder, prefix):
    """Record all captures under `prefix` for the duration of the block."""
    return _Section(recorder, prefix)


# --------------------------------------------------------------------------------------
# Module hooks
# --------------------------------------------------------------------------------------


def _on_forward(module, args, kwargs, output):
    rec = _REC

    if isinstance(module, q4.Qwen4ExpTextDecoderLayer):
        rec.layer_put(module, "in", args[0])
        rec.layer_put(module, "out", output)
    elif isinstance(module, torch.nn.Embedding):
        name = rec.names[module]
        if name.endswith("embed_tokens"):
            rec.put("embed/output", output)
        elif name.endswith("ngram_embedding"):
            # The reference feeds the hashed n-gram ids straight into this table.
            rec.layer_put(module, "ple/ngram_ids", args[0])
        else:
            raise AssertionError(f"unexpected embedding module {name}")
    elif isinstance(module, q4.Qwen4ExpTextRotaryEmbedding):
        rec.put("rotary/cos", output[0])
        rec.put("rotary/sin", output[1])
    elif isinstance(module, q4.Qwen4ExpTextGatedResidual):
        name = rec.names[module]
        if name.endswith("attn_hyper_connection"):
            rec.layer_put(module, "hc_attn/mixed", output[0])
            rec.layer_put(module, "hc_attn/inject_weights", output[2])
        elif name.endswith("mlp_hyper_connection"):
            rec.layer_put(module, "hc_mlp/mixed", output[0])
            rec.layer_put(module, "hc_mlp/inject_weights", output[2])
        elif name.endswith("hyper_connection_mixer"):
            # use_combine=False -> the final mixer returns the mixed tensor only.
            rec.put("final/mixer_out", output)
        else:
            raise AssertionError(f"unexpected hyper-connection {name}")
    elif isinstance(module, q4.Qwen4ExpTextQSAIndexer):
        rec.layer_put(module, "attn/mask", output)
    elif isinstance(module, q4.Qwen4ExpTextAttention):
        rec.layer_put(module, "attn/out", output[0])
    elif isinstance(module, q4.Qwen4ExpTextTopKRouter):
        rec.layer_put(module, "moe/router_logits", output[0])
        rec.layer_put(module, "moe/router_weights", output[1])
        rec.layer_put(module, "moe/router_indices", output[2])
    elif isinstance(module, q4.Qwen4ExpTextMLP):
        rec.layer_put(module, "moe/shared_out_ungated", output)
    elif isinstance(module, q4.Qwen4ExpTextExperts):
        rec.layer_put(module, "moe/expert_out", output)
    elif isinstance(module, q4.Qwen4ExpTextSparseMoeBlock):
        rec.layer_put(module, "moe/out", output)
    elif isinstance(module, q4.Qwen4ExpTextGatedDeltaNet):
        rec.layer_put(module, "gdn/out", output)
    elif isinstance(module, q4.Qwen4ExpTextRMSNormGated):
        rec.layer_put(module, "gdn/normed", output)
    elif isinstance(module, q4.Qwen4ExpTextPLELayer):
        rec.layer_put(module, "ple/out", output)
    elif isinstance(module, q4.Qwen4ExpTextNGramEmbedding):
        # Return value of the n-gram block: [B, T, ple_embed_dim] (all heads flattened).
        rec.layer_put(module, "ple/embeddings", output)
    elif isinstance(module, q4.Qwen4ExpTextRMSNorm):
        _on_rmsnorm(module, args, output)
    elif isinstance(module, torch.nn.Linear):
        _on_linear(module, output)
    elif isinstance(module, torch.nn.Conv1d):
        name = rec.names[module]
        if name.endswith("ple.conv1d"):
            # The reference applies silu outside the conv module (`F.silu(self.conv1d(...))`).
            rec.layer_put(module, "ple/conv_out", F.silu(output))
    return None


def _on_pre_forward(module, args):
    # The gated delta rule / causal conv paths are free functions: they need to know which
    # layer is currently running.
    layer_idx = getattr(module, "layer_idx", None)
    if layer_idx is not None:
        _REC.layer = layer_idx
    return None


def _on_linear(module, output):
    rec = _REC
    name = rec.names[module]
    if name.endswith("self_attn.q_proj"):
        # q_proj projects to num_heads * head_dim * 2: the second half is the output gate.
        attn = rec.parent(module)
        batch, seq_len = output.shape[0], output.shape[1]
        gate = output.view(batch, seq_len, -1, 2 * attn.head_dim).chunk(2, dim=-1)[1]
        rec.layer_put(module, "attn/gate", gate.reshape(batch, seq_len, -1))
    elif name.endswith("linear_attn.in_proj_z"):
        # `z` is reshaped to [B, T, num_v_heads, head_v_dim] before the gated norm; same data.
        rec.layer_put(module, "gdn/z", output)
    elif name.endswith("ple.value_proj"):
        rec.layer_put(module, "ple/value", output)
    elif name.endswith("mlp.shared_expert_gate"):
        layer = rec.layer_of(module)
        gate_logit = output
        sigmoid_gate = gate_logit.sigmoid()
        rec.layer_put(module, "moe/shared_gate_logit", gate_logit)
        rec.layer_put(module, "moe/shared_out", sigmoid_gate * rec.get(module, "moe/shared_out_ungated"))
    elif name == "lm_head":
        rec.put("final/logits", output)


def _on_rmsnorm(module, args, output):
    rec = _REC
    name = rec.names[module]
    if name.endswith("ple.norm_key"):
        rec.layer_put(module, "ple/key_normed", output)
    elif name.endswith("ple.norm_query"):
        rec.layer_put(module, "ple/query_normed", output)
    elif name.endswith("ple.norm_conv"):
        # norm_conv's input is the flattened gated value; its pre-image is the PLE gate.
        ple = rec.parent(module)
        gated_value = args[0]
        rec.layer_put(module, "ple/gated_value_normed", output)
        key_normed = rec.get(module, "ple/key_normed").unflatten(-1, (ple.hc_count, ple.hidden_size))
        query_normed = rec.get(module, "ple/query_normed").unflatten(-1, (ple.hc_count, ple.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(ple.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        recomputed = (torch.sigmoid(gate) * rec.get(module, "ple/value").unsqueeze(-2)).flatten(-2)
        if not torch.equal(recomputed, gated_value):
            raise AssertionError(
                "PLE gate derivation drifted from the reference "
                f"(max |diff| = {(recomputed - gated_value).abs().max().item():.3e})"
            )
        rec.layer_put(module, "ple/gate", gate)
        rec.layer_put(module, "ple/gated_value", gated_value)


# --------------------------------------------------------------------------------------
# Wrapped free functions
# --------------------------------------------------------------------------------------


def _record_free(name, value):
    """Record a capture for the layer whose `layer_idx` pre-hook fired last.

    The conv and gated-delta-rule functions are not methods, so they never see their module;
    `_on_pre_forward` keeps `Recorder.layer` pointing at the layer being executed and
    `_STUBS` supplies a name for it.
    """
    rec = _REC
    if rec.layer is None:
        raise AssertionError(f"{name} captured outside a decoder layer")
    rec.layer_put(_STUBS[rec][rec.layer], name, value)


# gated-delta-rule argument -> golden tensor name.
_GDN_ARGUMENTS = (("query", "gdn/q"), ("key", "gdn/k"), ("value", "gdn/v"), ("g", "gdn/g"), ("beta", "gdn/beta"))


def _wrap_gated_delta_rule(original):
    signature = pyinspect.signature(original)

    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        out, final_state = original(*args, **kwargs)
        for argument, name in _GDN_ARGUMENTS:
            _record_free(name, bound.arguments[argument])
        _record_free("gdn/core_out", out)
        return out, final_state

    return wrapper


def _wrap_causal_conv(original):
    def wrapper(*args, **kwargs):
        out = original(*args, **kwargs)
        _record_free("gdn/conv_out", out)
        return out

    return wrapper


def _wrap_eager_attention(original):
    def wrapper(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        out, weights = original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        rec = _REC
        rec.layer_put(module, "attn/q", query)
        rec.layer_put(module, "attn/k", key)
        rec.layer_put(module, "attn/v", value)
        rec.layer_put(module, "attn/weights", weights)
        return out, weights

    return wrapper


# --------------------------------------------------------------------------------------
# Qwen4ExpTextQSAIndexer.forward: verbatim copy of the reference body plus captures
# --------------------------------------------------------------------------------------


def _indexer_forward(self, hidden_states, position_embeddings, attention_mask, cache):
    rec = _REC
    batch_size, seq_length, _ = hidden_states.shape
    hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
    # The cos/sin here are the full positions for the keys, so we need to slice to get only the current positions for the queries
    full_cos, full_sin = position_embeddings
    current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]

    qk = self.index_qk_proj(hidden_states)
    q, token_k = torch.split(
        qk,
        [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim],
        dim=-1,
    )
    q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
    q = self.q_layernorm(q)
    q = q4.apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)

    if cache is not None:
        raw_keys = cache.update_indexer(raw_keys, self.layer_idx)

    rec.layer_put(self, "attn/indexer/q", q)
    rec.layer_put(self, "attn/indexer/raw_keys", raw_keys)

    # Note that the mask is never None here as we only allow eager and sdpa, and we do not allow sdpa's mask skip
    # It's always 4D with either bool (sdpa) or float (eager) and already gives us the valid indices
    visible_token_indices = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0

    selected_token_indices = torch.full(
        (batch_size, seq_length, self.token_budget + self.compress_ratio - 1),
        -1,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    max_blocks = attention_mask.shape[-1] // self.compress_ratio
    pooled = torch.zeros(
        batch_size, seq_length, max_blocks, self.index_head_dim, dtype=raw_keys.dtype, device=hidden_states.device
    )
    scores = torch.zeros(batch_size, seq_length, max_blocks, dtype=torch.float32, device=hidden_states.device)
    num_blocks = torch.zeros(batch_size, seq_length, dtype=torch.int32, device=hidden_states.device)
    for batch_idx in range(batch_size):
        for query_idx in range(seq_length):
            local_visible_indices = torch.nonzero(
                visible_token_indices[batch_idx, 0, query_idx], as_tuple=False
            ).flatten()
            num_complete_blocks = local_visible_indices.shape[-1] // self.compress_ratio
            num_blocks[batch_idx, query_idx] = num_complete_blocks
            # Compute selected tokens
            if num_complete_blocks > 0:
                block_token_indices = local_visible_indices[: num_complete_blocks * self.compress_ratio].view(
                    num_complete_blocks, self.compress_ratio
                )

                key_groups = raw_keys[batch_idx].index_select(0, block_token_indices.flatten())
                key_groups = key_groups.view(*block_token_indices.shape, self.index_head_dim)
                pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
                pooled_keys = self.k_layernorm(pooled_keys)
                group_starts = block_token_indices[:, 0]
                block_key_states = q4.apply_rotary_pos_emb(
                    pooled_keys.unsqueeze(1),
                    cos=full_cos[batch_idx].index_select(0, group_starts),
                    sin=full_sin[batch_idx].index_select(0, group_starts),
                ).squeeze(1)

                scores_q = torch.matmul(
                    q[batch_idx, query_idx].float(), block_key_states.float().transpose(-1, -2)
                ).transpose(-1, -2)
                scores_q = torch.relu(scores_q).sum(dim=-1) / math.sqrt(self.index_head_dim)

                selected_block_indices = scores_q.topk(min(self.block_topk, num_complete_blocks), dim=0).indices
                # Remap the indices of the blocks to the indices of individual tokens
                selected_tokens = block_token_indices.index_select(0, selected_block_indices).flatten()
                pooled[batch_idx, query_idx, :num_complete_blocks] = pooled_keys
                scores[batch_idx, query_idx, :num_complete_blocks] = scores_q
            else:
                selected_tokens = torch.tensor([], device=hidden_states.device)
            tail = local_visible_indices[num_complete_blocks * self.compress_ratio :]
            selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
            selected_token_indices[batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens

    rec.layer_put(self, "attn/indexer/pooled_keys", pooled)
    rec.layer_put(self, "attn/indexer/scores", scores)
    rec.layer_put(self, "attn/indexer/num_blocks", num_blocks)
    rec.layer_put(self, "attn/indexer/selected", selected_token_indices)

    # Create the selection mask to combine with the main causal mask
    kv_length = attention_mask.shape[-1]
    selected_token_mask = torch.zeros(
        (*selected_token_indices.shape[:-1], kv_length + 1), device=attention_mask.device, dtype=torch.bool
    )
    # We absorb all the -1 by scaterring them to the last index that we will drop
    scatter_indices = torch.where(selected_token_indices >= 0, selected_token_indices, kv_length)
    selected_token_mask = selected_token_mask.scatter(-1, scatter_indices, True)[..., :kv_length].unsqueeze(1)

    return selected_token_mask


# --------------------------------------------------------------------------------------
# Patch / unpatch
# --------------------------------------------------------------------------------------


class _LayerStub:
    """Marker object so `Recorder.layer_of` can resolve a layer number for free functions."""

    def __init__(self, layer):
        self.layer_idx = layer


_STUBS = {}
_SAVED = {}


def patch(model):
    """Install all hooks/wrappers; returns the recorder. Call `unpatch()` afterwards."""
    global _REC
    recorder = Recorder(model)
    _REC = recorder
    _STUBS[recorder] = {layer: _LayerStub(layer) for layer in range(getattr(model.config, "num_hidden_layers", 0))}
    # `Recorder.layer_of` looks names up by module identity.
    for layer, stub in _STUBS[recorder].items():
        recorder.names[stub] = f"layers.{layer}."

    _SAVED["module_hooks"] = [
        torch.nn.modules.module.register_module_forward_hook(_on_forward, with_kwargs=True),
        torch.nn.modules.module.register_module_forward_pre_hook(_on_pre_forward),
    ]
    _SAVED["indexer_forward"] = q4.Qwen4ExpTextQSAIndexer.forward
    q4.Qwen4ExpTextQSAIndexer.forward = _indexer_forward
    for name, wrapper in (
        ("torch_chunk_gated_delta_rule", _wrap_gated_delta_rule),
        ("torch_recurrent_gated_delta_rule", _wrap_gated_delta_rule),
        ("causal_conv1d_fn", _wrap_causal_conv),
        ("causal_conv1d_update", _wrap_causal_conv),
        ("eager_attention_forward", _wrap_eager_attention),
    ):
        _SAVED[name] = getattr(q4, name)
        setattr(q4, name, wrapper(_SAVED[name]))
    return recorder


def unpatch():
    global _REC
    for handle in _SAVED.pop("module_hooks"):
        handle.remove()
    q4.Qwen4ExpTextQSAIndexer.forward = _SAVED.pop("indexer_forward")
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        setattr(q4, name, _SAVED.pop(name))
    for name in ("causal_conv1d_fn", "causal_conv1d_update", "eager_attention_forward"):
        setattr(q4, name, _SAVED.pop(name))
    _STUBS.clear()
    _REC = None
