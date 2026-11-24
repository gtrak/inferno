"""Print the module tree and parameter table of a Qwen4Exp config (issue I-04).

Invoked through `inspect.py` (the file name mandated by the issue; see the note there about
shadowing the stdlib module). Accepts either a text config (`model_type: qwen4_exp_text`) or
a full multimodal config with a `text_config` block (e.g.
`~/models/qwen38-flash-next-nvfp4/config.json`), and builds the model on the `meta` device,
so the real 48-layer shape is instantiable without allocating 130 GB.

Every parameter is printed with the checkpoint tensor name(s) it comes from, which is the
mapping the engine's loader has to reproduce.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

from make_tiny import checkpoint_keys

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT.parent.parent / "tests" / "golden" / "tiny" / "config.json"


def load_config(path):
    raw = json.loads(Path(path).read_text())
    raw = raw.get("text_config", raw)
    known = Qwen4ExpTextConfig.__dataclass_fields__
    unknown = sorted(set(raw) - set(known))
    if unknown:
        print(f"# ignoring config keys not present in Qwen4ExpTextConfig: {unknown}", file=sys.stderr)
    return Qwen4ExpTextConfig(**{key: value for key, value in raw.items() if key in known})


def print_tree(model):
    print("\n=== module tree ===")
    for name, module in model.named_modules():
        depth = name.count(".") if name else 0
        label = name.rsplit(".", 1)[-1] if name else "<root>"
        if isinstance(module, torch.nn.ModuleList):
            label = f"{label} [{len(module)} x {type(module[0]).__name__}]"
        print(f"{'  ' * depth}{label}: {type(module).__name__}")


def _describe(name, config):
    keys = checkpoint_keys(name, config)
    if len(keys) == 1:
        return keys[0]
    if "ngram_embedding" in name:
        return f"{len(keys)} shards: {keys[0]} .. {keys[-1]}"
    return f"{len(keys)} per-expert tensors: {keys[0]} .. {keys[-1]}"


def print_parameters(model, config):
    print("\n=== parameters ===")
    print(f"{'name':<62} {'shape':<24} {'dtype':<12} checkpoint tensors")
    parameters = 0
    for name, tensor in model.named_parameters():
        parameters += tensor.numel()
        print(f"{name:<62} {str(tuple(tensor.shape)):<24} {str(tensor.dtype):<12} {_describe(name, config)}")

    # Rotary inv_freq is non-persistent (recomputed from the config), so it is not checkpointed.
    non_persistent = {
        f"{module_name}.{buffer_name}" if module_name else buffer_name
        for module_name, module in model.named_modules()
        for buffer_name in module._non_persistent_buffers_set
    }
    print("\n=== buffers (persistent) ===")
    for name, tensor in model.named_buffers():
        if name in non_persistent:
            continue
        print(f"{name:<62} {str(tuple(tensor.shape)):<24} {str(tensor.dtype):<12} {_describe(name, config)}")
    return parameters


def main(argv):
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_CONFIG
    config = load_config(path)
    try:
        shown = path.resolve().relative_to(ROOT.parent.parent)
    except ValueError:
        shown = path
    print(f"# config {shown}")
    print(
        f"# model_type {config.model_type} layers {config.num_hidden_layers} hidden {config.hidden_size} "
        f"layer_types {sorted(set(config.layer_types))} ple_layer_ids {config.ple_layer_ids} "
        f"experts {config.num_experts} top_k {config.num_experts_per_tok} hc_count {config.hc_count}"
    )
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(config)
    print_tree(model)
    parameters = print_parameters(model, config)
    print(f"\n# parameters {parameters} ({parameters * 4 / 2**30:.2f} GiB fp32, {parameters * 2 / 2**30:.2f} GiB bf16)")
    return 0
