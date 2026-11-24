"""Build the tiny random-init Qwen4Exp text checkpoint used as the project's oracle (I-04).

Writes `tests/golden/tiny/`:
  config.json            the tiny Qwen4ExpTextConfig
  model.safetensors      HF `transformers` layout (grouped expert tensors)
  checkpoint.safetensors the real checkpoint layout (`model.language_model.*`, per-expert
                         `mlp.experts.E.{gate,up,down}_proj.weight`, sharded n-gram table)
so the engine loader's key mapping can be tested against a real-shaped checkpoint.

The model exercises every feature of the architecture (see `check_tiny`): both attention
kinds, PLE, hyper-connections, MoE with top-k > 1, and the hashed n-gram table.
"""

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen4ExpTextConfig, Qwen4ExpForCausalLM
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT.parent.parent / "tests" / "golden" / "tiny"

TINY_CONFIG = dict(
    vocab_size=256,
    hidden_size=128,
    num_hidden_layers=4,
    layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=6,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    hc_count=2,
    hc_lowrank=16,
    ple_layer_ids=[2],
    ple_embed_dim=32,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=1000,
    make_ngram_vocab_size_divisible_by=8,
    split_ngram_parts=2,
    seed=1234,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=8,
    indexer_compress_ratio=2,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [2, 1, 1],
        "mrope_interleaved": True,
    },
    norm_topk_prob=True,
    output_gate_type="sigmoid",
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    max_position_embeddings=512,
    bos_token_id=254,
    eos_token_id=255,
)

PROBE_TOKENS = 12


def build_config():
    config = Qwen4ExpTextConfig(**TINY_CONFIG)
    # The oracle traces use the eager attention implementation; build the fixture the same way
    # so that a reload (`from_pretrained(..., _attn_implementation="eager")`) is bit-identical.
    config._attn_implementation = "eager"
    # `full_attention` is normalised to `indexed_attention` by the reference config.
    assert config.layer_types == ["linear_attention", "linear_attention", "indexed_attention", "linear_attention"]
    assert config.eos_token_id is not None  # required whenever PLE is enabled
    return config


def de_degenerate(model):
    """Re-randomise parameters whose reference init is exactly zero.

    The reference initialises `Qwen4ExpTextRMSNorm.weight` to zero (it is used as `1 + w`,
    so that is the identity) and `ple.conv1d.weight` to zero. The former is left alone; a
    zero PLE conv would make the whole dilated-conv branch vacuous, so it is randomised
    like any other conv.
    """
    generator = torch.Generator().manual_seed(0)
    for name, tensor in model.named_parameters():
        if not name.endswith("ple.conv1d.weight"):
            continue
        bound = 1.0 / math.sqrt(tensor.shape[-1])
        with torch.no_grad():
            tensor.uniform_(-bound, bound, generator=generator)


def check_tiny(model, config):
    """Assert the fixture exercises every architecture feature and is non-degenerate."""
    linear = sum(1 for kind in config.layer_types if kind == "linear_attention")
    indexed = sum(1 for kind in config.layer_types if kind == "indexed_attention")
    assert linear >= 1 and indexed >= 1, config.layer_types
    assert config.hc_count >= 2
    assert config.num_experts >= 4 and config.num_experts_per_tok >= 2
    ple_layers = [layer for layer in config.ple_layer_ids if config.layer_types[layer - 1] == "linear_attention"]
    assert ple_layers == config.ple_layer_ids, "PLE is only supported on linear_attention layers"
    ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram
    table = model.model.layers[config.ple_layer_ids[0] - 1].ple.ple_embedding
    assert table.ngram_heads == ngram_heads and table.total_vocab_size > config.ngram_vocab_size_base
    assert table.ngram_embedding.weight.shape[0] % config.split_ngram_parts == 0

    # Only the (1 + w)-centred RMSNorm weights may stay at the reference's zero init.
    zero_allowed = {
        id(tensor)
        for module in model.modules()
        if isinstance(module, Qwen4ExpTextRMSNorm)
        for tensor in module.parameters()
    }
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.is_floating_point() and not tensor.any():
            assert id(tensor) in zero_allowed, f"degenerate tensor {name}"

    input_ids = torch.arange(PROBE_TOKENS, dtype=torch.long).unsqueeze(0) % config.vocab_size
    with torch.no_grad():
        first = model(input_ids=input_ids, use_cache=False).logits
        second = model(input_ids=input_ids, use_cache=False).logits
    assert first.shape == (1, PROBE_TOKENS, config.vocab_size), first.shape
    assert first.dtype == torch.float32, first.dtype
    assert torch.isfinite(first).all(), "non-finite logits"
    assert torch.equal(first, second), "two consecutive forwards differ"
    assert first.unique().numel() > 1, "logits are constant"
    return first


def checkpoint_keys(name, config):
    """Checkpoint tensor names produced from one HF state-dict entry (see `to_checkpoint_layout`)."""
    if name.startswith("model."):
        name = "model.language_model." + name.removeprefix("model.")
    if name.endswith(".mlp.experts.gate_up_proj"):
        base = name.removesuffix(".gate_up_proj")
        return [
            f"{base}.{expert}.{leaf}_proj.weight"
            for expert in range(config.num_experts)
            for leaf in ("gate", "up")
        ]
    if name.endswith(".mlp.experts.down_proj"):
        return [f"{name.removesuffix('.down_proj')}.{expert}.down_proj.weight" for expert in range(config.num_experts)]
    if name.endswith(".ple_embedding.ngram_embedding.weight"):
        return [f"{name.removesuffix('.weight')}.shard_{shard}.weight" for shard in range(config.split_ngram_parts)]
    return [name]


def split_value(name, value, config):
    """Split one HF state-dict tensor into the pieces named by `checkpoint_keys`."""
    if name.endswith(".mlp.experts.gate_up_proj"):
        intermediate = config.moe_intermediate_size
        assert value.shape == (config.num_experts, 2 * intermediate, config.hidden_size), value.shape
        pieces = []
        for expert in range(config.num_experts):
            pieces.append(value[expert, :intermediate].clone())
            pieces.append(value[expert, intermediate:].clone())
        return pieces
    if name.endswith(".mlp.experts.down_proj"):
        assert value.shape == (config.num_experts, config.hidden_size, config.moe_intermediate_size), value.shape
        return [value[expert].clone() for expert in range(config.num_experts)]
    if name.endswith(".ple_embedding.ngram_embedding.weight"):
        assert value.shape[0] % config.split_ngram_parts == 0, (value.shape, config.split_ngram_parts)
        rows = value.shape[0] // config.split_ngram_parts
        return [value[shard * rows : (shard + 1) * rows].clone() for shard in range(config.split_ngram_parts)]
    return [value.clone()]


def to_checkpoint_layout(state_dict, config):
    """HF `transformers` state dict -> the naming used by the real published checkpoint."""
    checkpoint = {}
    for name, value in state_dict.items():
        keys = checkpoint_keys(name, config)
        for key, piece in zip(keys, split_value(name, value, config), strict=True):
            checkpoint[key] = piece
    return checkpoint


def from_checkpoint_layout(checkpoint, config):
    """Inverse of `to_checkpoint_layout`, written independently as a round-trip check."""
    shards = config.split_ngram_parts
    table_parts = {}
    expert_parts = {}
    state_dict = {}
    for name, value in checkpoint.items():
        if name.startswith("model.language_model."):
            key = "model." + name.removeprefix("model.language_model.")
        else:
            key = name
        if ".mlp.experts." in key:
            prefix, _, rest = key.partition(".mlp.experts.")
            expert, _, leaf = rest.partition(".")
            bucket = "gate_up_proj" if leaf in ("gate_proj.weight", "up_proj.weight") else "down_proj"
            expert_parts.setdefault(f"{prefix}.mlp.experts.{bucket}", []).append((int(expert), leaf, value))
        elif ".shard_" in key:
            stem, _, tail = key.partition(".shard_")
            table_parts.setdefault(stem, {})[int(tail.split(".")[0])] = value
        else:
            state_dict[key] = value
    for grouped_key, entries in expert_parts.items():
        ordered = sorted(entries, key=lambda entry: entry[0])
        if grouped_key.endswith("gate_up_proj"):
            gates = [value for _, leaf, value in ordered if leaf == "gate_proj.weight"]
            ups = [value for _, leaf, value in ordered if leaf == "up_proj.weight"]
            state_dict[grouped_key] = torch.stack([torch.cat(pair) for pair in zip(gates, ups, strict=True)])
        else:
            state_dict[grouped_key] = torch.stack([value for _, _, value in ordered])
    for stem, parts in table_parts.items():
        state_dict[stem + ".weight"] = torch.cat([parts[index] for index in range(shards)])
    return state_dict


def main():
    torch.set_num_threads(1)
    config = build_config()
    torch.manual_seed(0)
    model = Qwen4ExpForCausalLM(config)
    model.eval()
    model = model.float()
    de_degenerate(model)
    logits = check_tiny(model, config)

    GOLDEN.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(GOLDEN, safe_serialization=True)

    state_dict = model.state_dict()
    checkpoint = to_checkpoint_layout(state_dict, config)

    # transformers 5 already splits experts per expert and shards the n-gram table when it
    # saves; the only difference is the `language_model.` prefix of the multimodal wrapper.
    saved = load_file(str(GOLDEN / "model.safetensors"))
    renamed = {
        ("model.language_model." + key.removeprefix("model.") if key.startswith("model.") else key): value
        for key, value in saved.items()
    }
    assert checkpoint.keys() == renamed.keys(), (
        f"checkpoint layout differs from transformers' own save layout: {sorted(checkpoint.keys() ^ renamed.keys())}"
    )
    for key, value in renamed.items():
        assert torch.equal(checkpoint[key], value), f"checkpoint value differs from the saved one for {key}"

    round_trip = from_checkpoint_layout(checkpoint, config)
    assert round_trip.keys() == state_dict.keys(), (
        f"round-trip key mismatch: {sorted(round_trip.keys() ^ state_dict.keys())}"
    )
    for key, value in state_dict.items():
        assert torch.equal(round_trip[key], value), f"round-trip changed {key}"

    required = [
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
        "model.language_model.layers.2.self_attn.indexer.index_qk_proj.weight",
        "model.language_model.layers.3.linear_attn.A_log",
    ]
    missing = [key for key in required if key not in checkpoint]
    assert not missing, missing
    forbidden = [key for key in checkpoint if key.startswith(("mtp.", "model.visual."))]
    assert not forbidden, f"tiny checkpoint must be text-only: {forbidden[:3]}"

    save_file(checkpoint, str(GOLDEN / "checkpoint.safetensors"))

    # The checkerboard: reloaded from disk must reproduce the in-memory logits bit-exactly.
    reloaded = Qwen4ExpForCausalLM.from_pretrained(GOLDEN, dtype=torch.float32, _attn_implementation="eager")
    reloaded.eval()
    probe = torch.arange(PROBE_TOKENS, dtype=torch.long).unsqueeze(0) % config.vocab_size
    with torch.no_grad():
        assert torch.equal(reloaded(input_ids=probe, use_cache=False).logits, logits), "reload changed the logits"

    parameters = sum(tensor.numel() for tensor in model.parameters())
    print(f"wrote {GOLDEN}")
    print(f"  parameters {parameters}; hf tensors {len(state_dict)}; checkpoint tensors {len(checkpoint)}")
    print(f"  prefill logits {tuple(logits.shape)}, distinct values {logits.unique().numel()}")
    for path in sorted(GOLDEN.iterdir()):
        print(f"  {path.name} {path.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    sys.exit(main())
