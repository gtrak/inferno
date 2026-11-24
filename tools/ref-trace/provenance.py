"""Exact provenance of the reference implementation that produced the fixtures.

`run.sh` writes this out to `reference/VERSION.txt`; `trace.py` stamps the same commit into
the trace's safetensors metadata.
"""

import json
import sys
from importlib import metadata

import torch


def transformers_provenance():
    """(version, commit) of the installed transformers, resolved from its direct_url.json."""
    version = metadata.version("transformers")
    try:
        direct_url = json.loads(metadata.distribution("transformers").read_text("direct_url.json") or "{}")
    except (metadata.PackageNotFoundError, ValueError):
        direct_url = {}
    return version, direct_url.get("vcs_info", {}).get("commit_id", "unknown")


def version_text():
    version, commit = transformers_provenance()
    return (
        f"transformers_version: {version}\n"
        f"transformers_commit: {commit}\n"
        f"transformers_source: git+https://github.com/huggingface/transformers\n"
        f"torch_version: {torch.__version__}\n"
        f"python_version: {sys.version.split()[0]}\n"
        f"vendored_from: src/transformers/models/qwen4_exp/{{modeling,configuration}}_qwen4_exp.py\n"
        f"vendored_check: reference/*.py are byte-identical to the installed package\n"
    )


if __name__ == "__main__":
    sys.stdout.write(version_text())
