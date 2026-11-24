"""Run the HF Qwen4Exp reference on the tiny checkpoint and dump golden tensors (issue I-04).

Everything here is produced by the reference implementation itself (see `capture.py`). The
run is self-checking: the patched model must reproduce the unpatched model's logits
bit-exactly for both the 12-token prefill and the 1-token decode step, otherwise nothing is
written.

`--greedy` writes `tests/golden/tiny/greedy_ids.json` instead: the 16 token ids the
reference produces by greedy decoding from the same prompt. That file is the
greedy-equivalence gate for the CPU reference (I-005) and every later GPU path.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen4ExpForCausalLM

import capture
import provenance

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT.parent.parent / "tests" / "golden" / "tiny"

# Literal token ids (vocab_size is 256) so the trace never depends on a tokenizer.
INPUT_IDS = [13, 4, 199, 42, 7, 88, 211, 30, 96, 155, 61, 240]

# Greedy tokens the CPU reference must reproduce exactly (`--greedy` mode).
GREEDY_TOKENS = 16


def contract_names(config):
    """The exact tensor names required by issue I-04, derived from the tiny config."""
    names = ["embed/output", "rotary/cos", "rotary/sin", "final/mixer_out", "final/logits"]
    for layer, layer_type in enumerate(config["layer_types"]):
        prefix = f"layer{layer}/"
        names += [
            prefix + "in",
            prefix + "out",
            prefix + "hc_attn/mixed",
            prefix + "hc_attn/inject_weights",
            prefix + "hc_mlp/mixed",
            prefix + "hc_mlp/inject_weights",
            prefix + "moe/router_logits",
            prefix + "moe/router_weights",
            prefix + "moe/router_indices",
            prefix + "moe/shared_out",
            prefix + "moe/expert_out",
            prefix + "moe/out",
        ]
        if layer_type == "linear_attention":
            names += [
                prefix + name
                for name in (
                    "gdn/q", "gdn/k", "gdn/v", "gdn/g", "gdn/beta", "gdn/core_out",
                    "gdn/z", "gdn/normed", "gdn/out", "gdn/conv_out", "gdn/conv_state",
                )
            ]
        else:
            names += [
                prefix + name
                for name in (
                    "attn/q", "attn/k", "attn/v", "attn/gate", "attn/mask",
                    "attn/indexer/raw_keys", "attn/indexer/pooled_keys",
                    "attn/indexer/scores", "attn/indexer/selected", "attn/out",
                )
            ]
        if layer + 1 in config["ple_layer_ids"]:
            names += [
                prefix + name
                for name in (
                    "ple/ngram_ids", "ple/embeddings", "ple/key_normed", "ple/value",
                    "ple/gate", "ple/gated_value", "ple/conv_out", "ple/out",
                )
            ]
    return names


def _record_states(recorder, cache, config, prefix):
    """Final conv/recurrent state of every linear-attention layer, straight from the cache."""
    with capture.section(recorder, prefix):
        for layer, layer_type in enumerate(config["layer_types"]):
            if layer_type != "linear_attention":
                continue
            recorder.put(f"layer{layer}/gdn/conv_state", cache.get_conv_state(layer, state_idx=0))
            recorder.put(f"layer{layer}/gdn/recurrent_state", cache.get_recurrent_state(layer, state_idx=0))


def greedy():
    """Write the 16-token greedy continuation the CPU reference has to reproduce."""
    torch.set_num_threads(1)
    model = Qwen4ExpForCausalLM.from_pretrained(GOLDEN, dtype=torch.float32, _attn_implementation="eager")
    model.eval()
    input_ids = torch.tensor([INPUT_IDS], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=GREEDY_TOKENS,
            do_sample=False,
            use_cache=True,
            eos_token_id=None,
            pad_token_id=0,
        )
    generated = [int(t) for t in out[0, input_ids.shape[1] :].tolist()]
    if len(generated) != GREEDY_TOKENS:
        raise AssertionError(f"expected {GREEDY_TOKENS} generated ids, got {len(generated)}")
    version, commit = provenance.transformers_provenance()
    document = {
        "input_ids": INPUT_IDS,
        "generated_ids": generated,
        "max_new_tokens": GREEDY_TOKENS,
        "transformers_version": version,
        "transformers_sha": commit,
        "torch_version": torch.__version__,
        "num_threads": 1,
        "note": "greedy argmax, no sampling, no eos stop; compare exactly, token for token",
    }
    path = GOLDEN / "greedy_ids.json"
    path.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {path}: {generated}")


def main():
    # Single-threaded reductions keep the committed fixtures reproducible bit-for-bit.
    torch.set_num_threads(1)
    config = json.loads((GOLDEN / "config.json").read_text())
    model = Qwen4ExpForCausalLM.from_pretrained(GOLDEN, dtype=torch.float32, _attn_implementation="eager")
    model.eval()

    input_ids = torch.tensor([INPUT_IDS], dtype=torch.long)

    # Unpatched reference: prefill without a cache, prefill with a cache, then one greedy
    # decode step through that cache. Every later comparison is against these.
    with torch.no_grad():
        logits_no_cache = model(input_ids=input_ids, use_cache=False).logits
        prefill = model(input_ids=input_ids, use_cache=True)
        decode_input_id = int(prefill.logits[0, -1].argmax())
        decode_ids = torch.tensor([[decode_input_id]], dtype=torch.long)
        decode_logits = model(input_ids=decode_ids, cache=prefill.cache, use_cache=True).logits
    if not torch.equal(logits_no_cache, prefill.logits):
        raise AssertionError("cached and uncached prefill disagree in the reference itself")

    recorder = capture.patch(model)
    try:
        with capture.section(recorder, ""):
            with torch.no_grad():
                patched_prefill = model(input_ids=input_ids, use_cache=True)
        _record_states(recorder, patched_prefill.cache, config, "")
        with capture.section(recorder, "decode/"):
            with torch.no_grad():
                patched_decode = model(input_ids=decode_ids, cache=patched_prefill.cache, use_cache=True)
        _record_states(recorder, patched_decode.cache, config, "decode/")
    finally:
        capture.unpatch()

    for label, got, want in (
        ("prefill", patched_prefill.logits, prefill.logits),
        ("decode", patched_decode.logits, decode_logits),
    ):
        if not torch.equal(got, want):
            diff = (got - want).abs().max().item()
            raise AssertionError(f"capture layer changed the {label} logits (max |diff| = {diff:.3e})")

    tensors = recorder.tensors
    required = contract_names(config) + ["decode/" + name for name in contract_names(config)]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise AssertionError(f"contract names missing from the trace: {missing}")
    non_finite = [name for name, value in tensors.items() if not torch.isfinite(value).all()]
    if non_finite:
        raise AssertionError(f"non-finite tensors: {non_finite}")
    if decode_logits.shape != (1, 1, config["vocab_size"]):
        raise AssertionError(f"unexpected decode logits shape {tuple(decode_logits.shape)}")

    version, commit = provenance.transformers_provenance()
    metadata = {
        "transformers_sha": commit,
        "transformers_version": version,
        "torch_version": torch.__version__,
        "attn_implementation": "eager",
        "seed": "0",
        "model": str(GOLDEN.relative_to(ROOT.parent.parent)),
        "prefill_input_ids": json.dumps(INPUT_IDS),
        "decode_input_id": str(decode_input_id),
        "dtype": "float32 (float tensors); mask=bool, ids/indices=int/int32 as produced",
    }
    save_file(tensors, str(GOLDEN / "trace.safetensors"), metadata=metadata)
    print(f"wrote {GOLDEN / 'trace.safetensors'}: {len(tensors)} tensors")
    print(f"transformers {version} @ {commit}; prefill logits {tuple(prefill.logits.shape)}; "
          f"decode input id {decode_input_id}")


if __name__ == "__main__":
    if "--greedy" in sys.argv[1:]:
        greedy()
    else:
        main()
