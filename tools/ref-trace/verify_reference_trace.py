"""Prove a generated reference trace against the committed tiny goldens (issue I-031).

`reference_trace.py generate` writes logits; `trace.py` wrote per-activation goldens. This is the
bridge: it re-derives, from a logits trace, the three facts the committed fixtures already pin down

    p0/prompt_logits   == tests/golden/tiny/trace.safetensors :: final/logits
    p0/gen_logits[0]   == tests/golden/tiny/trace.safetensors :: decode/final/logits
    p0/gen_ids         == tests/golden/tiny/greedy_ids.json   :: generated_ids

and asserts each within a stated tolerance. `gen_logits[0]` is the generator's first greedy decode
step, i.e. the same single cached step the fixture's decode section recorded, and `gen_ids` has to
reproduce `model.generate(do_sample=False)` token for token even though the loop here is open-coded.

The bound is not invented: `--noise-floor` reads `tests/golden/tiny/noise_floor.json` (measured by
`reference_trace.py noise-floor`) and allows `--margin` times that spread. At `--threads 1` the
comparison is bit-exact, which is the stronger statement; the noise floor is what a run at the
default thread count is allowed to cost. The KL gate takes the looser of the measured floor and the
analytic bound `2 * (logit bound)^2`, because KL is quadratic in a logit error and the measured KL
of a 4.5e-8 logit wobble is ~1e-15 -- no reimplementation on another machine could beat that.

Non-vacuity is mandatory here, so two injection modes exist:

    # passes: the generated trace is the committed reference
    python verify_reference_trace.py --trace traces/tiny-v1.safetensors

    # must fail: a perturbation calibrated so its own measured mean KL is 1e-3
    python verify_reference_trace.py --trace traces/tiny-v1.safetensors --inject-kl 1e-3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import safetensors.torch
import torch
from safetensors import safe_open

import reference_trace


HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parent.parent / "tests" / "golden" / "tiny"


def load_trace(path: Path) -> tuple[dict, dict]:
    with safe_open(str(path), framework="pt") as handle:
        metadata = dict(handle.metadata() or {})
    return safetensors.torch.load_file(str(path)), metadata


def perturb(logits: torch.Tensor, relative: float, seed: int = 0) -> torch.Tensor:
    """Deterministic perturbation with a known scale, for the non-vacuity run.

    `relative * max|logit| * N(0,1)` per element: it moves the distribution, so both the logit bound
    and the KL bound must react, and the argmax mostly survives -- which is exactly the failure mode
    a top-1-only gate would miss.
    """
    generator = torch.Generator().manual_seed(seed)
    scale = float(logits.abs().max()) * relative
    return logits + scale * torch.randn(logits.shape, generator=generator, dtype=torch.float32)


def kl_divergence(reference: torch.Tensor, other: torch.Tensor) -> dict:
    """Per-position KL(p_golden || p_ours), in float64 for the reason stated in reference_trace."""
    log_p = reference_trace.log_softmax_f64(reference.double())
    log_q = reference_trace.log_softmax_f64(other.double())
    per_position = reference_trace.kl_per_position(log_p, log_q).flatten()
    return {
        "kl_mean": float(per_position.mean()),
        "kl_p99": float(torch.quantile(per_position, 0.99)),
        "kl_max": float(per_position.max()),
        "argmax_agreement": float((reference.argmax(-1) == other.argmax(-1)).float().mean()),
    }


def compare(name: str, ours: torch.Tensor, golden: torch.Tensor, limit: float, kl_limit: float) -> bool:
    if ours.numel() == golden.numel() and ours.shape != golden.shape:
        # The trace stores logits as [positions, vocab]; the activation goldens keep the reference's
        # own [batch, positions, vocab]. Same data, so normalise to the fixture's rank.
        ours = ours.reshape(golden.shape)
    if ours.shape != golden.shape:
        print(f"  {name:<22} SHAPE ours {tuple(ours.shape)} vs golden {tuple(golden.shape)} -> FAIL")
        return False
    ours = ours.float()
    golden = golden.float()
    maximum = float((ours - golden).abs().max())
    metrics = kl_divergence(golden, ours)
    ok = maximum <= limit and metrics["kl_mean"] <= kl_limit
    print(
        f"  {name:<22} max|dlogit| {maximum:.3e} (<= {limit:.1e})  "
        f"KL mean {metrics['kl_mean']:.3e} p99 {metrics['kl_p99']:.3e} (<= {kl_limit:.1e})  "
        f"argmax agree {metrics['argmax_agreement']:.4f}  {'OK' if ok else 'FAIL'}"
    )
    return ok


def calibrate_perturbation(golden: torch.Tensor, target_kl: float, seed: int = 0) -> tuple[float, float]:
    """Log-bisect the perturbation amplitude whose measured mean KL is `target_kl`.

    The issue demands that a threshold be shown to refuse a *known* KL, not merely a known logit
    wobble, so the injection is calibrated against the same metric the gate uses.
    """
    low, high = 1e-12, 3.0
    ceiling = kl_divergence(golden, perturb(golden, high, seed))["kl_mean"]
    if ceiling < target_kl:
        raise AssertionError(
            f"cannot reach mean KL {target_kl:.1e} on this fixture: the largest perturbation the "
            f"calibration tries ({high}) only gets {ceiling:.1e}"
        )
    for _ in range(48):
        middle = (low + high) / 2
        achieved = kl_divergence(golden, perturb(golden, middle, seed))["kl_mean"]
        if achieved < target_kl:
            low = middle
        else:
            high = middle
    achieved = kl_divergence(golden, perturb(golden, high, seed))["kl_mean"]
    return high, achieved


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="verify a generated logits trace against the tiny goldens")
    parser.add_argument("--trace", required=True, help="a trace written by reference_trace.py generate")
    parser.add_argument("--noise-floor", default=str(GOLDEN / "noise_floor.json"))
    parser.add_argument("--margin", type=float, default=2.0, help="allowed multiple of the measured noise floor")
    parser.add_argument("--max-abs-diff", type=float, default=None, help="override the derived bound")
    parser.add_argument("--perturb", type=float, default=0.0, help="inject a relative logit perturbation")
    parser.add_argument("--inject-kl", type=float, default=0.0, help="inject a perturbation calibrated to this mean KL")
    arguments = parser.parse_args(argv)

    tensors, metadata = load_trace(Path(arguments.trace))
    print(f"trace {arguments.trace}")
    for key in ("transformers_version", "transformers_commit", "torch_version", "threads_effective",
                "attn_implementation", "dtype", "prompt_file_sha256", "load_seconds", "device_map_requested"):
        print(f"  {key:<22} {metadata.get(key, '<missing>')}")

    floor_path = Path(arguments.noise_floor)
    if not floor_path.exists():
        raise AssertionError(f"{floor_path} is missing; regenerate it with `reference_trace.py noise-floor`")
    floor = json.loads(floor_path.read_text())
    aggregate = floor["aggregate"]
    limit = arguments.max_abs_diff
    if limit is None:
        limit = arguments.margin * float(aggregate["max_abs_logit_diff"])
    # Two floors, and the KL gate takes the looser of them. The measured one is the reference's own
    # noise; the derived one is the analytic bound KL(p||q) <= 2 * (sup |logit error|)^2, which says
    # what a run sitting exactly on the logit bound is allowed to cost. Using the measured KL alone
    # would ask a different machine to beat ~1e-15, i.e. it would fail for being *better* behaved
    # than the box that produced the floor.
    derived_kl = 2.0 * limit**2
    kl_limit = max(arguments.margin * float(aggregate["kl_max"]), derived_kl)
    print(
        f"noise floor {floor_path.name}: run_a {floor['run_a_threads']} thread(s) vs run_b "
        f"{floor['run_b_threads']} threads -> max|dlogit| {aggregate['max_abs_logit_diff']:.3e}, "
        f"KL max {aggregate['kl_max']:.3e}"
    )
    print(
        f"bounds (margin {arguments.margin}x): max|dlogit| <= {limit:.3e}, KL <= {kl_limit:.3e} "
        f"= max({arguments.margin}x measured {arguments.margin * float(aggregate['kl_max']):.3e}, "
        f"derived 2*bound^2 {derived_kl:.3e})"
    )

    fixture = safetensors.torch.load_file(str(GOLDEN / "trace.safetensors"))
    greedy = json.loads((GOLDEN / "greedy_ids.json").read_text())

    scale = arguments.perturb
    if arguments.inject_kl:
        scale, achieved = calibrate_perturbation(fixture["final/logits"], arguments.inject_kl)
        print(
            f"  non-vacuity: injecting mean KL {arguments.inject_kl:.1e} -- amplitude "
            f"{scale:.3e} x max|logit| measured back as {achieved:.3e}"
        )
    elif scale:
        print(f"  non-vacuity: injecting a relative logit perturbation of {scale:.1e}")

    checks = []
    prefill = tensors["p0/prompt_logits"]
    checks.append(
        compare("prefill logits", perturb(prefill, scale) if scale else prefill, fixture["final/logits"],
                limit, kl_limit)
    )
    if "p0/gen_logits" in tensors:
        generated = tensors["p0/gen_logits"]
        checks.append(
            compare(
                "decode logits[0]",
                perturb(generated[0][None], scale) if scale else generated[0][None],
                fixture["decode/final/logits"],
                limit,
                kl_limit,
            )
        )
        ids = [int(token) for token in tensors["p0/gen_ids"].tolist()]
        want = list(greedy["generated_ids"])
        agree = sum(left == right for left, right in zip(ids, want)) / max(len(want), 1)
        ok = ids[: len(want)] == want
        print(f"  {'greedy ids':<22} {agree * 100:.1f}% of {len(want)} tokens match greedy_ids.json "
              f"{'OK' if ok else 'FAIL'}")
        print(f"    ours   {ids[: len(want)]}")
        print(f"    golden {want}")
        checks.append(ok)

    print("verify:", "PASS" if all(checks) else "FAIL")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
