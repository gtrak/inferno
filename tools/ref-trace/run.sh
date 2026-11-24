#!/usr/bin/env bash
# I-04: bootstrap the Hugging Face reference environment and (re)generate the golden
# fixtures. Idempotent: re-running it rebuilds the venv contents in place and rewrites
# tests/golden/tiny/* deterministically.
set -euo pipefail
cd "$(dirname "$0")"

UV="${UV:-/home/gary/.local/bin/uv}"
PYTHON_VERSION="3.13"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
VENV_PY=".venv/bin/python"

log() { printf '==> %s\n' "$*"; }

if [ ! -x "${VENV_PY}" ]; then
  log "creating .venv (python ${PYTHON_VERSION})"
  "${UV}" venv -p "${PYTHON_VERSION}" .venv
fi

log "installing CPU torch"
"${UV}" pip install --python "${VENV_PY}" torch --index-url "${TORCH_INDEX}"

log "installing ref-trace and transformers from git main"
"${UV}" pip install --python "${VENV_PY}" -e .

log "verifying the reference imports"
"${VENV_PY}" - <<'PY'
import transformers
from transformers import Qwen4ExpConfig, Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration, Qwen4ExpTextConfig

print(f"transformers {transformers.__version__}: Qwen4ExpForCausalLM and Qwen4ExpForConditionalGeneration import")
PY

log "checking the vendored reference copy against the installed package"
for module in modeling_qwen4_exp configuration_qwen4_exp; do
  installed="$("${VENV_PY}" -c "import transformers.models.qwen4_exp.${module} as m; print(m.__file__)")"
  if ! cmp -s "reference/${module}.py" "${installed}"; then
    printf 'ERROR: reference/%s.py is not byte-identical to %s\n' "${module}" "${installed}" >&2
    exit 1
  fi
done

log "recording the resolved reference version"
"${VENV_PY}" provenance.py > reference/VERSION.txt
cat reference/VERSION.txt

log "building the tiny checkpoint (tests/golden/tiny)"
"${VENV_PY}" make_tiny.py

log "dumping golden traces (tests/golden/tiny/trace.safetensors)"
"${VENV_PY}" trace.py

log "done"
