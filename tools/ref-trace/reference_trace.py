"""Generate the Hugging Face *logits* reference trace that I-031's quality metrics grade against.

`trace.py` produces per-activation goldens for the tiny fixture (240 tensors, one forward,
kernel-validation scale). This tool produces the other kind of reference: **logits per
position over a fixed prompt set**, which is what KL / ΔPPL / router-agreement need, and it
works for any checkpoint the installed `transformers` can load — the tiny fixture today, the
real 124 GB checkpoint on a quiet box once `io-probe` says the box can hold it.

Three modes
-----------

    generate     run the reference over a prompt set, write a safetensors logits trace
    noise-floor  run the same prompt set twice (1 thread, then the default thread count) and
                 record the spread. That spread is the floor under every quality threshold:
                 a gate tighter than the reference's own noise is decoration (an earlier
                 tolerance here was 1e-6 while the reference moved 4.5e-8 with thread count)
    io-probe     no model, no GPU: read the checkpoint's safetensors headers, measure this
                 box's read bandwidth, and print the load/offload arithmetic. This is the
                 scheduling input for the expensive run

Determinism contract (a future run must reproduce the bytes)
-----------------------------------------------------------

1. `--threads 1` (i.e. `torch.set_num_threads(1)`) for any trace that will be committed. The
   tiny fixture is that setting's output and nothing else reproduces it bit for bit.
2. `--attn eager`. The reference's `sdpa`/`flash` paths change the arithmetic.
3. Greedy argmax only: no sampling, no logits warpers, no `eos` stop. The loop is open-coded
   here instead of `model.generate` so the logits at *every* generated position are captured;
   `verify_reference_trace.py` proves it reproduces `greedy_ids.json`, which `trace.py` wrote
   with `generate`, token for token.
4. One prompt per forward (batch = 1), so no padding semantics enter the picture.
5. Everything that can move the numbers is stamped into the trace metadata: config hash,
   prompt-file hash, `transformers` version + commit, `torch`/`numpy`/python versions, the
   requested *and* effective thread count, the requested and resolved device map, dtype,
   `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`CUDA_VISIBLE_DEVICES`, repo revision, hostname.
6. The model runs the first prompt twice before anything is written and refuses to continue if
   the two runs are not bit-identical: that catches a nondeterministic kernel (GPU atomics in a
   reduction path) instead of silently shipping an unreproducible reference.

Examples
--------

    # prove the plumbing on the tiny fixture (CPU, seconds)
    python reference_trace.py generate --model ../../tests/golden/tiny \
        --prompts prompts/tiny-v1.json --positions 16 --out traces/tiny-v1.safetensors
    python reference_trace.py noise-floor --model ../../tests/golden/tiny \
        --prompts prompts/tiny-v1.json --positions 1 --out-report ../../tests/golden/tiny/noise_floor.json

    # what the real run would cost, without touching the GPUs
    python reference_trace.py io-probe --model ~/models/qwen38-flash-next-nvfp4 \
        --prompts prompts/quality-v1.json --out traces/real-io-probe.json

    # the real reference, on a quiet box (see REAL_REFERENCE.md for the blockers it must clear)
    python reference_trace.py generate --model ~/models/qwen38-flash-next-nvfp4 \
        --prompts prompts/quality-v1.json --positions 64 --dtype bfloat16 \
        --device-map auto --max-memory '{"cpu":"40GiB",0:"14GiB",1:"14GiB",2:"14GiB",3:"14GiB","disk":"80GiB"}' \
        --out traces/real-v1.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

import provenance

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# dtypes the reference may be loaded in. The checkpoint's own dtype is bfloat16; float32 is
# available for the tiny fixture, where it is what the committed goldens were produced with.
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
STORE_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


# ---------------------------------------------------------------------------------------
# provenance: everything a future re-run has to match
# ---------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Repo-relative when it lives in the repo, absolute otherwise (stable committed metadata)."""
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:  # a checkout without git (tarball, container): record that, do not guess
        return "unknown"


def environment_snapshot() -> dict:
    version, commit = provenance.transformers_provenance()
    try:
        import accelerate

        accelerate_version = accelerate.__version__
    except ImportError:  # only device_map runs need it
        accelerate_version = "<not installed>"
    return {
        "transformers_version": version,
        "transformers_commit": commit,
        "accelerate_version": accelerate_version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu_count": os.cpu_count(),
        "repo_revision": git_revision(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "<unset>"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "<unset>"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "<no cuda>",
    }


def resolved_device_map(model) -> dict:
    """Where `from_pretrained` actually put every module (the requested map is recorded too)."""
    placement: dict[str, str] = {}
    for _name, parameter in model.named_parameters():
        device = str(parameter.device)
        placement[device] = placement.get(device, 0) + 1
    return {
        "parameter_devices": placement,
        "buffer_count": sum(1 for _ in model.named_buffers()),
        "dtype": str(next(model.parameters()).dtype),
    }


def vocab_size_of(model) -> int:
    """`Qwen4ExpForCausalLM` keeps the text config as `config`; the VL wrapper nests `text_config`."""
    config = getattr(model.config, "text_config", model.config)
    return int(config.vocab_size)


# ---------------------------------------------------------------------------------------
# prompt set
# ---------------------------------------------------------------------------------------


def load_prompts(path: Path) -> tuple[list[dict], str]:
    """Prompt records plus the hash of the file they came from.

    Schema: `{"name": str, "note": str, "prompts": [{"id": str, "text": str?, "input_ids": [int]}]}`.
    `input_ids` is authoritative so a tokenizer change can never silently change the reference;
    `text` is what a human reads, and `--tokenizer` re-derives the ids from it to catch drift.
    """
    document = json.loads(path.read_text())
    prompts = document["prompts"]
    if not prompts:
        raise AssertionError(f"{path}: empty prompt set")
    for prompt in prompts:
        ids = [int(token) for token in prompt["input_ids"]]
        if len(ids) < 2:
            raise AssertionError(f"{path}: prompt {prompt.get('id')} has {len(ids)} tokens")
        if any(token < 0 for token in ids):
            raise AssertionError(f"{path}: prompt {prompt.get('id')} has a negative token id")
    return prompts, sha256_file(path)


def verify_tokenizer(prompts: list[dict], tokenizer_path: str, vocab_size: int) -> None:
    """Re-derive every prompt's ids from its text with the checkpoint's own tokenizer.

    A reference trace is only comparable to another one if the token stream is identical, so a
    tokenizer change must be an error, never a silently different set of ids.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    for prompt in prompts:
        text = prompt.get("text")
        if text is None:
            continue
        remapped = [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]
        if remapped != [int(token) for token in prompt["input_ids"]]:
            raise AssertionError(
                f"prompt {prompt['id']}: tokenizer at {tokenizer_path} re-tokenises the text to "
                f"{len(remapped)} ids, the committed record has {len(prompt['input_ids'])}"
            )
        for token in prompt["input_ids"]:
            if not 0 <= int(token) < vocab_size:
                raise AssertionError(f"prompt {prompt['id']}: id {token} outside vocab {vocab_size}")


# ---------------------------------------------------------------------------------------
# the reference forward
# ---------------------------------------------------------------------------------------


def load_model(model_path: str, arguments):
    """`from_pretrained` with the settings recorded, and nothing implicit.

    `low_cpu_mem_usage` stays off unless asked for: it changes the load path (meta init plus mmap),
    and the tiny proof run must load exactly the way `trace.py` loads to be comparable bit for bit.
    """
    from transformers import AutoModelForCausalLM

    kwargs = {
        "dtype": DTYPES[arguments.dtype],
        "_attn_implementation": arguments.attn,
    }
    if arguments.low_cpu_mem_usage:
        kwargs["low_cpu_mem_usage"] = True
    if arguments.device_map and arguments.device_map != "none":
        kwargs["device_map"] = arguments.device_map
        if arguments.max_memory:
            kwargs["max_memory"] = json.loads(arguments.max_memory)
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    load_seconds = time.monotonic() - started
    model.eval()
    return model, kwargs, load_seconds


def input_device(model) -> torch.device:
    """Where token ids have to live.

    Under `device_map` accelerate's hooks align inputs anyway, but a model simply placed on one
    device is not so forgiving, and "expected all tensors on the same device" at 3 a.m. on the quiet
    box is a bad way to discover that the generator only worked on CPU.
    """
    for name, module in model.named_modules():
        if name.endswith("embed_tokens"):
            return next(module.parameters()).device
    return next(model.parameters()).device


@torch.no_grad()
def prefill_logits(model, input_ids: torch.Tensor):
    """Teacher-forced logits for every prompt position (position t predicts token t+1)."""
    output = model(input_ids=input_ids, use_cache=True)
    return output.logits.to(torch.float32).cpu().clone(), output.cache


@torch.no_grad()
def greedy_trace(model, cache, first_logits: torch.Tensor, positions: int, device):
    """Greedy continuation, capturing the logits at every generated position.

    Equivalent to `generate(do_sample=False, eos_token_id=None)`: argmax over the full vocab, no
    warpers, no early stop. The first generated token is the argmax of `first_logits`, and the
    logits returned for step k are those produced by feeding token k-1 -- so `gen_logits[0]` is
    the reference's single decode step at position P, exactly the `decode/final/logits` tensor of
    the tiny fixture.
    """
    generated_ids: list[int] = []
    generated_logits: list[torch.Tensor] = []
    logits = first_logits
    for _step in range(positions):
        token = int(logits[0, -1].argmax())
        generated_ids.append(token)
        step = model(input_ids=torch.tensor([[token]], dtype=torch.long, device=device), cache=cache, use_cache=True)
        logits = step.logits.to(torch.float32).cpu().clone()
        generated_logits.append(logits)
        cache = step.cache
    if not generated_logits:
        return generated_ids, torch.zeros(0, dtype=torch.float32)
    return generated_ids, torch.cat(generated_logits, dim=1)


def run_reference(model, prompts: list[dict], positions: int, store_dtype: torch.dtype) -> tuple[dict, dict]:
    """Produce the tensor dict + per-prompt facts for one reference run."""
    tensors: dict[str, torch.Tensor] = {}
    facts: dict[str, dict] = {}
    device = input_device(model)

    # Determinism self-check on prompt 0 before any output is written.
    probe = torch.tensor([prompts[0]["input_ids"]], dtype=torch.long, device=device)
    first, _ = prefill_logits(model, probe)
    second, _ = prefill_logits(model, probe)
    if not torch.equal(first, second):
        spread = (first - second).abs().max().item()
        raise AssertionError(
            f"the reference is not deterministic in this environment: prompt {prompts[0]['id']} "
            f"prefilled twice with max |diff| = {spread:.3e}. Fix the environment before recording "
            "a reference; a trace nobody can reproduce is worse than no trace."
        )

    for index, prompt in enumerate(prompts):
        ids = [int(token) for token in prompt["input_ids"]]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        logits, cache = prefill_logits(model, input_ids)
        generated_ids, generated = greedy_trace(model, cache, logits, positions, device)
        vocab = vocab_size_of(model)
        if logits.shape != (1, len(ids), vocab):
            raise AssertionError(f"prompt {prompt['id']}: unexpected prefill logits {tuple(logits.shape)}, want {1, len(ids), vocab}")
        if positions and generated.shape != (1, positions, logits.shape[-1]):
            raise AssertionError(f"prompt {prompt['id']}: unexpected gen logits {tuple(generated.shape)}")
        if not torch.isfinite(logits).all() or (positions and not torch.isfinite(generated).all()):
            raise AssertionError(f"prompt {prompt['id']}: non-finite logits in the reference run")
        prefix = f"p{index}"
        tensors[f"{prefix}/prompt_ids"] = torch.tensor(ids, dtype=torch.int64)
        tensors[f"{prefix}/prompt_logits"] = logits[0].to(store_dtype).contiguous()
        if positions:
            tensors[f"{prefix}/gen_ids"] = torch.tensor(generated_ids, dtype=torch.int64)
            tensors[f"{prefix}/gen_logits"] = generated[0].to(store_dtype).contiguous()
        facts[prompt["id"]] = {
            "prompt_tokens": len(ids),
            "generated_tokens": len(generated_ids),
            "first_generated_ids": generated_ids[:8],
            "logits_max": float(logits.max()),
            "logits_min": float(logits.min()),
        }
    return tensors, facts


# ---------------------------------------------------------------------------------------
# metrics for the noise-floor report
# ---------------------------------------------------------------------------------------


def log_softmax_f64(logits: torch.Tensor, chunk_rows: int = 256) -> torch.Tensor:
    """Row-wise log-softmax in float64, chunked so a 248k-vocab trace stays bounded.

    Float64 is not pedantry here. KL is *quadratic* in a logit perturbation, so the thread-noise KL
    of this reference is ~1e-15 while the logits themselves are fp32 (1e-7 resolution): computed in
    fp32 the sum is cancellation noise, it can come out negative, and it reports a floor ~1e6 times
    the truth. Anything that will be quoted as a threshold gets the wider accumulator.
    """
    rows = logits.reshape(-1, logits.shape[-1])
    result = torch.empty(rows.shape, dtype=torch.float64)
    for start in range(0, rows.shape[0], chunk_rows):
        result[start : start + chunk_rows] = torch.log_softmax(rows[start : start + chunk_rows].double(), dim=-1)
    return result.reshape(logits.shape)


def kl_per_position(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per position from float64 log-probabilities. Both from `log_softmax_f64`."""
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def distribution_metrics(reference: torch.Tensor, other: torch.Tensor) -> dict:
    """logits(reference) vs logits(other): the floor under KL, ΔPPL and top-1 agreement gates.

    `reference` is the canonical run (1 thread). Logits are fp32 as stored; every reduction is
    float64 (see `log_softmax_f64`). The *per position* KL is what I-031 grades, so it is reported
    as a distribution rather than a single mean.
    """
    if reference.shape != other.shape:
        raise AssertionError(f"shape mismatch {tuple(reference.shape)} vs {tuple(other.shape)}")
    difference = (reference.double() - other.double()).abs()
    log_p = log_softmax_f64(reference)
    log_q = log_softmax_f64(other)
    # A PPL gate needs both runs scored on the *same* fixed token stream, so the canonical run's
    # argmax ids are the target for both. A single-position tensor has no next-token pair, so its
    # PPL terms are reported as null rather than as the mean of an empty slice.
    target = reference.argmax(dim=-1)
    shift_target = target[..., 1:]
    nll_reference = nll_other = None
    if shift_target.numel():
        nll_reference = -log_p[..., :-1, :].gather(-1, shift_target[..., None]).squeeze(-1).mean()
        nll_other = -log_q[..., :-1, :].gather(-1, shift_target[..., None]).squeeze(-1).mean()
    flat_kl = kl_per_position(log_p, log_q).flatten()
    return {
        "positions": int(flat_kl.numel()),
        "max_abs_logit_diff": float(difference.max()),
        "mean_abs_logit_diff": float(difference.mean()),
        "kl_mean": float(flat_kl.mean()),
        "kl_p50": float(torch.quantile(flat_kl, 0.5)),
        "kl_p99": float(torch.quantile(flat_kl, 0.99)),
        "kl_max": float(flat_kl.max()),
        "argmax_agreement": float((reference.argmax(dim=-1) == other.argmax(dim=-1)).float().mean()),
        "ppl_reference": None if nll_reference is None else float(torch.exp(nll_reference)),
        "ppl_repeated": None if nll_other is None else float(torch.exp(nll_other)),
        "delta_ppl_floor": None if nll_other is None else float(abs(torch.exp(nll_other) - torch.exp(nll_reference))),
        "note": (
            "KL is quadratic in a logit perturbation: max|dlogit| ~4.5e-8 implies KL ~1e-15, which is "
            "why every reduction here is float64. At that magnitude a mean KL can still come out "
            "slightly negative from float64 cancellation -- read it as zero, the reference's own noise "
            "is not measurable in KL. The sensitive quantities at this scale are top-1 agreement, "
            "because two logits inside the noise flip the argmax, and dPPL."
        ),
    }


def per_prompt_metrics(reference_tensors: dict, other_tensors: dict) -> dict:
    report: dict[str, dict] = {}
    for key in sorted(reference_tensors):
        if not key.endswith("_logits"):
            continue
        reference = reference_tensors[key].to(torch.float32)
        other = other_tensors[key].to(torch.float32)
        # [positions, vocab] -> [1, positions, vocab] for the row-wise metrics.
        report[key] = distribution_metrics(reference[None], other[None])
    return report


def aggregate_metrics(per_prompt: dict) -> dict:
    """Worst tensor per metric: a floor is only a floor if it covers the noisiest tensor."""

    def extreme(field: str, worst: str) -> float | None:
        values = [metrics[field] for metrics in per_prompt.values() if metrics[field] is not None]
        if not values:
            return None
        return float(max(values) if worst == "max" else min(values))

    return {
        "max_abs_logit_diff": extreme("max_abs_logit_diff", "max"),
        "mean_abs_logit_diff": extreme("mean_abs_logit_diff", "max"),
        "kl_mean": extreme("kl_mean", "max"),
        "kl_p50": extreme("kl_p50", "max"),
        "kl_p99": extreme("kl_p99", "max"),
        "kl_max": extreme("kl_max", "max"),
        "argmax_agreement_min": extreme("argmax_agreement", "min"),
        "delta_ppl_floor_max": extreme("delta_ppl_floor", "max"),
        "tensors": len(per_prompt),
    }


# ---------------------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------------------


def apply_thread_setting(threads: int) -> int:
    """`--threads 0` means "leave torch's default alone" (that is the noise-floor's default run)."""
    if threads == 0:
        return torch.get_num_threads()
    torch.set_num_threads(threads)
    return torch.get_num_threads()


def cmd_generate(arguments, prompts, prompts_sha) -> int:
    effective_threads = apply_thread_setting(arguments.threads)
    model, requested, load_seconds = load_model(arguments.model, arguments)
    if arguments.tokenizer:
        verify_tokenizer(prompts, arguments.tokenizer, vocab_size_of(model))
    tensors, facts = run_reference(model, prompts, arguments.positions, STORE_DTYPES[arguments.store_dtype])
    metadata = {
        key: str(value)
        for key, value in {
            **environment_snapshot(),
            "mode": "generate",
            "model_path": display_path(arguments.model),
            "model_config_sha256": sha256_file(Path(arguments.model) / "config.json"),
            "prompt_file": display_path(arguments.prompts),
            "prompt_file_sha256": prompts_sha,
            "prompt_names": json.dumps([prompt["id"] for prompt in prompts]),
            "positions": str(arguments.positions),
            "dtype": arguments.dtype,
            "store_dtype": arguments.store_dtype,
            "attn_implementation": arguments.attn,
            "threads_requested": str(arguments.threads),
            "threads_effective": str(effective_threads),
            "device_map_requested": str(arguments.device_map),
            "max_memory_requested": str(arguments.max_memory),
            "low_cpu_mem_usage": str(arguments.low_cpu_mem_usage),
            "from_pretrained_kwargs": json.dumps({key: str(v) for key, v in requested.items()}),
            "module_placement": json.dumps(resolved_device_map(model)),
            "load_seconds": f"{load_seconds:.3f}",
            "per_prompt": json.dumps(facts),
            "greedy": "argmax over full vocab, no sampling, no eos stop, batch=1, logits captured per position",
        }.items()
    }
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(arguments.out), metadata=metadata)
    print(
        f"wrote {arguments.out}: {len(tensors)} tensors, "
        f"{sum(t.numel() for t in tensors.values())} elements, threads={effective_threads}, "
        f"load {load_seconds:.1f}s"
    )
    for name, fact in facts.items():
        print(f"  {name}: {fact['prompt_tokens']} prompt + {fact['generated_tokens']} gen tokens, "
              f"first ids {fact['first_generated_ids']}")
    return 0


def cmd_noise_floor(arguments, prompts, prompts_sha) -> int:
    """Run the same reference twice, only the thread count differing, and record the spread.

    Run A is the canonical setting (`torch.set_num_threads(1)`), run B is whatever torch picked by
    default. Both runs happen in one process so the weights, the device placement and the prompt
    order are identical -- thread count is the only variable, which is exactly what is being measured.
    """
    default_requested = arguments.threads_default or torch.get_num_threads()
    Path(arguments.out_report).parent.mkdir(parents=True, exist_ok=True)

    effective = apply_thread_setting(1)
    if effective != 1:
        raise AssertionError(f"torch refused 1 thread (got {effective})")
    model, requested, load_seconds = load_model(arguments.model, arguments)
    if arguments.tokenizer:
        verify_tokenizer(prompts, arguments.tokenizer, vocab_size_of(model))
    canonical, canonical_facts = run_reference(model, prompts, arguments.positions, STORE_DTYPES["float32"])
    canonical_threads = torch.get_num_threads()

    default_threads = apply_thread_setting(default_requested)
    if default_threads == 1:
        raise AssertionError(
            "the second run asked for the default thread count but torch is pinned to 1 thread, so "
            "the recorded spread would be zero and meaningless"
        )
    repeated, repeated_facts = run_reference(model, prompts, arguments.positions, STORE_DTYPES["float32"])
    # If the two runs are identical there is nothing to record, and that is usually the sign of a
    # silently ignored thread setting (torch built against a single-threaded backend).
    identical = all(torch.equal(canonical[k], repeated[k]) for k in canonical)

    per_prompt = per_prompt_metrics(canonical, repeated)
    report = {
        **environment_snapshot(),
        "mode": "noise-floor",
        "model_path": display_path(arguments.model),
        "model_config_sha256": sha256_file(Path(arguments.model) / "config.json"),
        "prompt_file": display_path(arguments.prompts),
        "prompt_file_sha256": prompts_sha,
        "positions": arguments.positions,
        "dtype": arguments.dtype,
        "attn_implementation": arguments.attn,
        "device_map_requested": arguments.device_map,
        "load_seconds": load_seconds,
        "run_a_threads": canonical_threads,
        "run_b_threads": default_threads,
        "runs_bit_identical": identical,
        "aggregate": aggregate_metrics(per_prompt),
        "per_tensor": per_prompt,
        "per_prompt_facts": {"canonical": canonical_facts, "repeated": repeated_facts},
        "how_to_read": (
            "aggregate is the worst tensor per metric. Every quality threshold (KL mean/p99, "
            "dPPL, top-1 agreement) must sit above these numbers by a stated margin, or it is "
            "measuring the reference's own reduction order rather than the engine under test."
        ),
    }
    Path(arguments.out_report).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {arguments.out_report}")
    print(f"  threads: {canonical_threads} -> {default_threads}, runs bit-identical: {identical}")
    for key, metrics in sorted(per_prompt.items()):
        delta_ppl = metrics["delta_ppl_floor"]
        print(
            f"  {key:<24} max|dlogit| {metrics['max_abs_logit_diff']:.3e}  "
            f"KL mean {metrics['kl_mean']:.3e} p99 {metrics['kl_p99']:.3e} max {metrics['kl_max']:.3e}  "
            f"argmax agree {metrics['argmax_agreement']:.4f}  "
            + (f"dPPL floor {delta_ppl:.3e}" if delta_ppl is not None else "dPPL floor n/a (1 position)")
        )
    if identical and not arguments.allow_zero_spread:
        raise AssertionError(
            "the two runs are bit-identical, so nothing was measured. Either torch's thread count is "
            "pinned by the environment (OMP_NUM_THREADS / torch.get_num_threads()) or the reference "
            "has no multi-threaded reduction on this device; pass --allow-zero-spread only if that is "
            "a fact you want recorded"
        )
    return 0


# ---------------------------------------------------------------------------------------
# io-probe: the arithmetic for the expensive run, with no model and no GPU work
# ---------------------------------------------------------------------------------------

DTYPE_BYTES = {"F8_E4M3": 1, "F8_E5M2": 1, "U8": 1, "I8": 1, "BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "BOOL": 1}


def safetensors_headers(model_dir: Path) -> list[tuple[str, dict]]:
    headers = []
    for path in sorted(model_dir.glob("*.safetensors")):
        with path.open("rb") as handle:
            length = int.from_bytes(handle.read(8), "little")
            headers.append((path.name, json.loads(handle.read(length))))
    if not headers:
        raise AssertionError(f"{model_dir}: no .safetensors files")
    return headers


def classify(name: str) -> str:
    if ".mlp.experts." in name:
        return "routed experts (NVFP4)"
    if "ngram_embedding" in name:
        return "PLE n-gram table (gather-only)"
    if name.startswith("mtp."):
        return "MTP layer"
    if name.startswith("model.visual.") or ".visual." in name:
        return "vision tower"
    if name.startswith("lm_head") or "embed_tokens" in name:
        return "embed_tokens + lm_head"
    if ".mlp.shared_expert" in name:
        return "shared expert + router"
    return "dense text path"


def drop_page_cache(paths: list[Path]) -> int:
    """Evict these files from the page cache. Returns how many files were successfully advised.

    Needed because btrfs (and ext4) will happily satisfy an O_DIRECT *read* from uptodate page cache
    pages: a repeat measurement of a file this box has already read reports memcpy speed, not the
    drive's. Dropping the cache for the files under measurement is the only way to get a number that
    a first run on a quiet box will actually reproduce. It is scoped to the files named, and it is a
    hint - other readers just fault the pages back in.
    """
    dropped = 0
    for path in paths:
        try:
            handle = os.open(str(path), os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(handle, 0, 0, os.POSIX_FADV_DONTNEED)
            dropped += 1
        except (OSError, AttributeError):
            pass
        finally:
            os.close(handle)
    return dropped


def measure_read_bandwidth(model_dir: Path, mebibytes: int, streams: int, label: str) -> dict:
    """Sequential read of `mebibytes` per stream across the shards, in one burst.

    Direct I/O is used so the read is real rather than a page-cache hit, and `drop_page_cache` is
    applied first for the reason documented there. `label` is "cold" or "warm" and says which of the
    two regimes the number describes.
    """
    paths = sorted(model_dir.glob("model-*.safetensors")) or sorted(model_dir.glob("*.safetensors"))
    if not paths:
        raise AssertionError(f"{model_dir}: no safetensors files to read")
    block = 8 << 20
    payload = mebibytes << 20
    results: list[tuple[str, int, float]] = []
    direct = getattr(os, "O_DIRECT", 0)
    chosen = [paths[(index * len(paths)) // max(streams, 1) % len(paths)] for index in range(streams)]
    evicted = drop_page_cache(chosen) if label == "cold" else 0
    try:
        probe = os.open(str(paths[0]), os.O_RDONLY | direct)
        if direct and len(os.read(probe, block)) == 0:  # short reads are legal, a zero read is not
            direct = 0
        os.close(probe)
    except OSError:
        direct = 0

    def worker(path: Path) -> None:
        handle = os.open(str(path), os.O_RDONLY | direct)
        started = time.monotonic()
        read = 0
        try:
            while read < payload:
                chunk = os.read(handle, block)
                if not chunk:
                    break
                read += len(chunk)
        except OSError as error:  # alignment or unsupported flag: fail loudly rather than reporting cache
            os.close(handle)
            raise RuntimeError(f"direct read failed: {error}") from error
        elapsed = time.monotonic() - started
        os.close(handle)
        results.append((path.name, read, elapsed))

    workers = [threading.Thread(target=worker, args=(path,)) for path in chosen]
    for worker_handle in workers:
        worker_handle.start()
    for worker_handle in workers:
        worker_handle.join()
    if not results:
        raise AssertionError("no read stream completed; cannot report a bandwidth")
    total_bytes = sum(read for _name, read, _elapsed in results)
    slowest = max(elapsed for _name, _read, elapsed in results)
    return {
        "label": label,
        "direct_io": bool(direct),
        "streams": streams,
        "mebibytes_per_stream": mebibytes,
        "files_evicted_from_page_cache": evicted,
        "total_bytes": total_bytes,
        "seconds": slowest,
        "gib_per_s": total_bytes / slowest / 2**30,
        "per_stream": [{"file": name, "bytes": read, "seconds": elapsed} for name, read, elapsed in results],
    }


def memory_snapshot() -> dict:
    """MemTotal / MemAvailable from /proc/meminfo, in bytes."""
    snapshot: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("MemTotal", "MemAvailable", "MemFree"):
            snapshot[key] = int(value.strip().split()[0]) * 1024
    if "MemAvailable" not in snapshot:
        raise AssertionError("/proc/meminfo has no MemAvailable")
    return snapshot


def nvidia_smi_memory() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # no driver, or a box without nvidia-smi
        return []
    devices = []
    for line in out.stdout.strip().splitlines():
        index, total, used, util = [value.strip() for value in line.split(",")]
        devices.append(
            {
                "index": int(index),
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "utilization_percent": int(util),
            }
        )
    return devices


def measure_shard_load(path: Path) -> dict:
    """Real load-path measurement: materialise one shard's tensors on CPU, timed, with peak RSS.

    This is the closest honest stand-in for "how long does `from_pretrained` take": it is the same
    safetensors decode + tensor materialisation, without the model construction on top. Opt-in, and
    one shard costs about its own size in RAM and page cache.
    """
    import resource

    import safetensors.torch

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    evicted = drop_page_cache([path])
    started = time.monotonic()
    state = safetensors.torch.load_file(str(path))
    mapped_seconds = time.monotonic() - started
    tensor_count = len(state)
    bytes_loaded = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
    dtypes = sorted({str(tensor.dtype) for tensor in state.values()})

    # `load_file` mmaps, so the numbers above are the header decode: nothing has paged in yet. Touch
    # one element per 4 KiB page to force the faults a real device copy would cause, and time that
    # separately -- it is the figure `from_pretrained` actually pays.
    started = time.monotonic()
    checksum = 0
    for tensor in state.values():
        if not tensor.is_contiguous() or tensor.dim() == 0:
            continue
        checksum += int(tensor.view(torch.uint8)[::4097].sum())
    touched_seconds = time.monotonic() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    del state
    return {
        "file": path.name,
        "files_evicted_from_page_cache": evicted,
        "tensors": tensor_count,
        "bytes_materialised": bytes_loaded,
        "mmap_only_seconds": mapped_seconds,
        "touch_seconds": touched_seconds,
        "touch_gib_per_s": bytes_loaded / touched_seconds / 2**30,
        "touch_checksum": checksum,
        "process_peak_rss_bytes_before": before,
        "process_peak_rss_bytes_after": peak,
        "dtypes_seen": dtypes,
    }


def cmd_io_probe(arguments, prompts, prompts_sha) -> int:
    model_dir = Path(arguments.model).expanduser().resolve()
    config = json.loads((model_dir / "config.json").read_text())
    # The VL wrapper nests the text config under `text_config`; a text-only checkpoint does not.
    text_config = config.get("text_config", config)
    headers = safetensors_headers(model_dir)

    group_bytes: dict[str, int] = {}
    group_elements: dict[str, int] = {}
    tensor_count = 0
    largest = ("", 0)
    for name, header in headers:
        for tensor, entry in header.items():
            if tensor == "__metadata__":
                continue
            tensor_count += 1
            if entry["dtype"] not in DTYPE_BYTES:
                raise AssertionError(f"{name}:{tensor}: unhandled safetensors dtype {entry['dtype']}")
            elements = int(np.prod(entry["shape"], dtype=np.int64)) if entry["shape"] else 1
            size = elements * DTYPE_BYTES[entry["dtype"]]
            bucket = classify(tensor)
            group_bytes[bucket] = group_bytes.get(bucket, 0) + size
            group_elements[bucket] = group_elements.get(bucket, 0) + elements
            if size > largest[1]:
                largest = (f"{name}:{tensor}", size)

    # Can the installed transformers even dispatch this checkpoint's quantiser? Pure config work.
    declared = config.get("quantization_config")
    quantiser = {"quant_method": (declared or {}).get("quant_method", "<none>")}
    try:
        if declared is None:
            quantiser["dispatches"] = True  # unquantised checkpoint, nothing to dispatch
        else:
            from transformers.quantizers.auto import AutoQuantizationConfig

            AutoQuantizationConfig.from_dict(declared)
            quantiser["dispatches"] = True
    except Exception as error:
        quantiser["dispatches"] = False
        quantiser["error"] = f"{type(error).__name__}: {error}"

    cold = measure_read_bandwidth(model_dir, arguments.read_mib, arguments.read_streams, "cold")
    warm = measure_read_bandwidth(model_dir, arguments.read_mib, arguments.read_streams, "warm")
    bandwidth = cold if cold["gib_per_s"] <= warm["gib_per_s"] else warm
    shard_load = measure_shard_load(model_dir / arguments.load_shard) if arguments.load_shard else None
    devices = nvidia_smi_memory()
    meminfo = memory_snapshot()

    on_disk = sum(group_bytes.values())
    vram_bytes = sum(device["memory_total_mib"] for device in devices) * 2**20
    ram_bytes = meminfo["MemAvailable"]
    # Placement budget. Everything pinned to a compute device (GPU *or* CPU, in accelerate's model)
    # is free per forward call; anything that falls out to disk is re-read on every single call.
    vram_budget = int(arguments.vram_budget_gib * 2**30) if arguments.vram_budget_gib else vram_bytes
    cpu_budget = int(arguments.cpu_budget_gib * 2**30) if arguments.cpu_budget_gib else int(ram_bytes * 0.75)
    placeable = vram_budget + cpu_budget
    spill = max(on_disk - placeable, 0)
    gib = 2**30
    # The stored bytes are the quantised footprint. `materialised_bf16` is what `from_pretrained`
    # has to hold if it has no loader for the stored format, which is exactly the NVFP4 case here.
    elements = sum(group_elements.values())
    materialised_bf16 = 2 * elements

    prompt_tokens = sum(len(prompt["input_ids"]) for prompt in prompts)
    forward_calls = len(prompts) * (1 + arguments.positions)  # one prefill, then one per token
    per_call_seconds = spill / gib / bandwidth["gib_per_s"]
    report = {
        "mode": "io-probe",
        **environment_snapshot(),
        "model_path": display_path(model_dir),
        "model_config_sha256": sha256_file(model_dir / "config.json"),
        "prompt_file": display_path(arguments.prompts),
        "prompt_file_sha256": prompts_sha,
        "prompt_tokens": prompt_tokens,
        "positions_per_prompt": arguments.positions,
        "checkpoint": {
            "files": len(headers),
            "tensors": tensor_count,
            "bytes_on_disk": on_disk,
            "elements": elements,
            "materialised_bfloat16_bytes": materialised_bf16,
            "largest_tensor": {"name": largest[0], "bytes": largest[1]},
            "by_group_bytes": group_bytes,
            "by_group_elements": group_elements,
            "materialised_bfloat16_by_group_bytes": {name: 2 * count for name, count in group_elements.items()},
            "text_layers": text_config["num_hidden_layers"],
            "text_vocab_size": text_config["vocab_size"],
            "experts": text_config["num_experts"],
            "experts_per_token": text_config["num_experts_per_tok"],
        },
        "quantiser": quantiser,
        "box": {
            "vram_total_bytes": vram_bytes,
            "ram_total_bytes": meminfo["MemTotal"],
            "ram_available_bytes": ram_bytes,
            "gpus": devices,
            "pcie_reference_from_theory": (
                "THEORY.md I-002 probe: 6.73 GiB/s H2D per GPU, 26.88 GiB/s aggregate for four GPUs "
                "sharing one bifurcated Gen4 x16. Only relevant if weights pinned in RAM have to be "
                "shipped to a GPU per call; the disk spill below is NVMe-bound, not PCIe-bound"
            ),
        },
        "read_bandwidth_measured": bandwidth,
        "read_bandwidth_warm_page_cache": warm,
        "read_bandwidth_basis": (
            f"arithmetic uses the {bandwidth['label']} measurement; the warm page-cache number is "
            "recorded only to show how far a cache-hit measurement overstates the drive"
        ),
        "shard_load_measured": shard_load,
        "arithmetic": {
            "note": (
                "accelerate re-reads every module that fell out to disk on every forward call: a "
                "prefill is one call, and so is each generated token. That traffic, not FLOPs, is "
                "the wall clock. Weights pinned to CPU cost compute instead, which this model does "
                "not attempt to bound here."
            ),
            "bytes_on_disk": on_disk,
            "vram_budget_bytes": vram_budget,
            "cpu_budget_bytes": cpu_budget,
            "placeable_bytes": placeable,
            "bytes_spilled_to_disk": spill,
            "spill_gib_per_forward_call": spill / gib,
            "seconds_per_forward_call_at_measured_read_bandwidth": per_call_seconds,
            "forward_calls": forward_calls,
            "transfer_seconds_for_the_whole_run": per_call_seconds * forward_calls,
            "cold_load_seconds_at_measured_bandwidth": on_disk / (bandwidth["gib_per_s"] * gib),
            "seconds_to_reread_each_group_per_call": {
                name: size / gib / bandwidth["gib_per_s"] for name, size in group_bytes.items()
            },
            "materialised_bf16_exceeds_vram_plus_ram": materialised_bf16 > vram_bytes + ram_bytes,
            "logit_trace_bytes_float32": (prompt_tokens + len(prompts) * arguments.positions)
            * text_config["vocab_size"]
            * 4,
        },
    }
    if arguments.out:
        Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {arguments.out}")

    print(f"checkpoint {on_disk / 2**30:.1f} GiB in {tensor_count} tensors across {len(headers)} files")
    for bucket, size in sorted(group_bytes.items(), key=lambda item: -item[1]):
        bucket_elements = group_elements[bucket]
        print(
            f"  {bucket:<34} {size / 2**30:8.2f} GiB   "
            f"{bucket_elements / 1e9:7.2f} G elements -> bf16 would be {2 * bucket_elements / 2**30:8.2f} GiB"
        )
    print(
        f"  materialised bf16 for the whole checkpoint: {materialised_bf16 / 2**30:.1f} GiB vs "
        f"{(vram_bytes + ram_bytes) / 2**30:.1f} GiB of VRAM + available RAM"
    )
    print(f"  largest tensor {largest[0]} {largest[1] / 2**20:.1f} MiB")
    print(f"quantiser: {quantiser['quant_method']} dispatches={quantiser['dispatches']}")
    if not quantiser["dispatches"]:
        print(f"  BLOCKER {quantiser['error']}")
    print(
        f"read bandwidth: {cold['gib_per_s']:.2f} GiB/s cold ({cold['streams']} streams x "
        f"{cold['mebibytes_per_stream']} MiB, direct_io={cold['direct_io']}, "
        f"{cold['files_evicted_from_page_cache']} files evicted) vs "
        f"{warm['gib_per_s']:.2f} GiB/s warm -> arithmetic uses {bandwidth['label']}"
    )
    print(
        f"box: VRAM {vram_bytes / 2**30:.1f} GiB total, RAM available {ram_bytes / 2**30:.1f} GiB, "
        f"{len(devices)} GPUs"
    )
    print(
        "placement: "
        f"{on_disk / gib:.1f} GiB on disk vs {vram_budget / gib:.1f} GiB VRAM budget + "
        f"{cpu_budget / gib:.1f} GiB CPU budget -> {spill / gib:.1f} GiB falls to disk"
    )
    if shard_load:
        print(
            f"shard load measured: {shard_load['file']} "
            f"{shard_load['bytes_materialised'] / gib:.2f} GiB -- mmap setup {shard_load['mmap_only_seconds']:.1f} s, "
            f"page-touching every 4 KiB {shard_load['touch_seconds']:.1f} s = "
            f"{shard_load['touch_gib_per_s']:.2f} GiB/s; process peak RSS "
            f"{shard_load['process_peak_rss_bytes_before'] / gib:.1f} -> "
            f"{shard_load['process_peak_rss_bytes_after'] / gib:.1f} GiB"
        )
    print(
        "arithmetic: "
        f"{spill / gib:.1f} GiB re-read per forward call -> {per_call_seconds:.1f} s per call at the "
        f"measured {bandwidth['gib_per_s']:.2f} GiB/s; {forward_calls} calls for this prompt set = "
        f"{per_call_seconds * forward_calls / 3600:.1f} h of transfer alone, plus "
        f"{report['arithmetic']['cold_load_seconds_at_measured_bandwidth'] / 60:.1f} min cold load"
    )
    return 0


# ---------------------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("generate", "noise-floor", "io-probe"), nargs="?", default="generate")
    parser.add_argument("--model", required=True, help="checkpoint directory (tiny fixture or real)")
    parser.add_argument("--prompts", default=str(HERE / "prompts" / "quality-v1.json"))
    parser.add_argument("--positions", type=int, default=0, help="greedy continuation tokens per prompt")
    parser.add_argument("--out", default=None, help="output .safetensors trace (generate) / .json report (io-probe)")
    parser.add_argument("--out-report", default=None, help="output .json report (noise-floor)")
    parser.add_argument("--dtype", default="float32", choices=sorted(DTYPES), help="reference compute dtype")
    parser.add_argument("--store-dtype", default="float32", choices=sorted(STORE_DTYPES), help="trace logits dtype")
    parser.add_argument("--attn", default="eager", help="attention implementation (eager = the exact reference)")
    parser.add_argument("--threads", type=int, default=1, help="torch threads for generate; 1 = reproducible")
    parser.add_argument("--threads-default", type=int, default=0, help="noise-floor: second run's thread count (0 = torch's default at start-up)")
    parser.add_argument("--allow-zero-spread", action="store_true", help="record a noise floor even if the two runs are bit-identical")
    parser.add_argument("--device-map", default="none", help="'none' for CPU-only, 'auto' with --max-memory for offload")
    parser.add_argument("--max-memory", default=None, help="JSON max_memory for device_map=auto")
    parser.add_argument("--tokenizer", default=None, help="re-derive prompt ids from text and fail on drift")
    parser.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        help="pass low_cpu_mem_usage=True to from_pretrained (offload runs want this; it changes the load path)",
    )
    parser.add_argument("--read-mib", type=int, default=256, help="io-probe: MiB read per stream")
    parser.add_argument("--read-streams", type=int, default=4, help="io-probe: concurrent read streams")
    parser.add_argument(
        "--load-shard",
        default=None,
        help="io-probe: also materialise this shard on CPU to measure the load path (costs its size in RAM)",
    )
    parser.add_argument("--vram-budget-gib", type=float, default=0.0, help="io-probe: weight bytes to pin in VRAM (0 = all)")
    parser.add_argument("--cpu-budget-gib", type=float, default=0.0, help="io-probe: weight bytes to pin in RAM (0 = 75%% of available)")
    return parser


def main(argv: list[str]) -> int:
    arguments = build_parser().parse_args(argv)
    prompts, prompts_sha = load_prompts(Path(arguments.prompts))
    print(f"prompt set {Path(arguments.prompts).name}: {len(prompts)} prompts, sha256 {prompts_sha[:12]}")
    if arguments.mode == "io-probe":
        return cmd_io_probe(arguments, prompts, prompts_sha)
    if arguments.mode == "noise-floor":
        if not arguments.out_report:
            raise AssertionError("noise-floor needs --out-report")
        return cmd_noise_floor(arguments, prompts, prompts_sha)
    if not arguments.out:
        raise AssertionError("generate needs --out")
    return cmd_generate(arguments, prompts, prompts_sha)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
