#!/usr/bin/env bash
# Gate for the inferno repository.
#
# Formats, builds and tests the Zig core (including the nvcc cubin step), the
# Rust server, and finally asserts that the probe sees this machine's four GPUs.
# Idempotent: safe to run repeatedly from any directory.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="$CUDA_HOME/bin:$PATH"

# This machine has four RTX 5060 Ti cards; a different count is a failure.
expected_devices=4

step() {
    echo "== $1"
}

step "zig fmt --check core/src core/build.zig"
zig fmt --check core/src core/build.zig

step "zig build -Doptimize=ReleaseSafe (core, compiles kernels with nvcc)"
(cd core && zig build -Doptimize=ReleaseSafe)

# The 132 GB checkpoint is not a build input. A bare `zig build` must succeed on
# a machine that has no weights at all, so run it again with the model directory
# pointed at a path that does not exist: a build that still read the checkpoint
# (the manifest step used to hang off the install step) fails here.
step "zig build -Doptimize=ReleaseSafe with no checkpoint on disk"
absent_checkpoint="$repo_root/core/.absent-checkpoint"
if [ -e "$absent_checkpoint" ]; then
    echo "$absent_checkpoint exists; it must not" >&2
    exit 1
fi
(cd core && INFERNO_MODEL_DIR="$absent_checkpoint" zig build -Doptimize=ReleaseSafe)

step "zig build test (core)"
(cd core && zig build test)

step "cargo fmt --all -- --check"
cargo fmt --all -- --check

step "cargo build --workspace --release"
cargo build --workspace --release

step "cargo clippy --workspace --all-targets -- -D warnings"
cargo clippy --workspace --all-targets -- -D warnings

step "cargo test --workspace"
cargo test --workspace

step "./target/release/inferno-server --probe (expect $expected_devices devices)"
probe_output="$("./target/release/inferno-server" --probe)"
echo "$probe_output"
probe_check='
import json
import sys

document = json.load(sys.stdin)
assert "devices" in document, "probe document has no devices"
devices = document["devices"]
expected = int(sys.argv[1])
assert len(devices) == expected, "expected %d devices, got %d" % (
    expected,
    len(devices),
)
for index, device in enumerate(devices):
    assert device["idx"] == index, "device %d reports idx %d" % (index, device["idx"])
    assert device["vram_bytes"] > 0, "device %d reports no VRAM" % index
print(
    "probe ok: %d devices, driver %s, cuda %d"
    % (len(devices), document["driver"], document["cuda"])
)
'
printf '%s' "$probe_output" | python3 -c "$probe_check" "$expected_devices"

echo "== quant fixtures are regenerable (drift guard)"
if [ ! -f tools/ref-trace/quant_oracle.py ]; then
    echo "SKIP quant fixture drift guard: tools/ref-trace/quant_oracle.py not present"
elif [ ! -x tools/ref-trace/.venv/bin/python ]; then
    echo "SKIP quant fixture drift guard: no venv (run tools/ref-trace/run.sh)"
else
    # quant_oracle.py takes no arguments and writes tests/golden/quant in place, so
    # copy the committed bytes aside, regenerate, compare, and ALWAYS restore. The
    # trap matters: without it an interrupt or a failed cmp after the rewrite would
    # leave regenerated fixtures in the tree, and a later run would then "pass"
    # against bytes nobody reviewed.
    drift_dir="$(mktemp -d)"
    restore_fixtures() {
        cp "$drift_dir"/*.safetensors tests/golden/quant/ 2>/dev/null || true
        rm -rf "$drift_dir"
    }
    trap restore_fixtures EXIT INT TERM
    cp tests/golden/quant/*.safetensors "$drift_dir/"
    drift_status=0
    if ! ( cd tools/ref-trace && ./.venv/bin/python quant_oracle.py >/dev/null ); then
        echo "FAIL could not regenerate quant fixtures"
        drift_status=1
    fi
    if [ "$drift_status" -eq 0 ]; then
        for fixture in nvfp4_case fp8_blockwise_case; do
            if ! cmp -s "$drift_dir/$fixture.safetensors" \
                "tests/golden/quant/$fixture.safetensors"; then
                echo "FAIL tests/golden/quant/$fixture.safetensors is not reproducible from quant_oracle.py"
                drift_status=1
            fi
        done
    fi
    restore_fixtures
    trap - EXIT INT TERM
    if [ "$drift_status" -ne 0 ]; then
        exit 1
    fi
    echo "quant fixtures reproduce byte-for-byte"
fi

echo "== structural invariants (dep-free core, contained unsafe mass)"
python3 tools/invariants.py
echo "== gate passed"
