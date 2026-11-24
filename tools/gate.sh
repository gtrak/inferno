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
    # quant_oracle.py takes no arguments and writes tests/golden/quant in place,
    # so copy the committed bytes aside, regenerate, compare, then restore. The
    # tree is left exactly as it was, whether or not drift is detected.
    drift_dir="$(mktemp -d)"
    cp tests/golden/quant/*.safetensors "$drift_dir/"
    ( cd tools/ref-trace && ./.venv/bin/python quant_oracle.py >/dev/null ) || {
        echo "FAIL could not regenerate quant fixtures"
        cp "$drift_dir"/*.safetensors tests/golden/quant/
        rm -rf "$drift_dir"
        exit 1
    }
    drift_status=0
    for fixture in nvfp4_case fp8_blockwise_case; do
        if ! cmp -s "$drift_dir/$fixture.safetensors" \
            "tests/golden/quant/$fixture.safetensors"; then
            echo "FAIL tests/golden/quant/$fixture.safetensors is not reproducible from quant_oracle.py"
            drift_status=1
        fi
    done
    cp "$drift_dir"/*.safetensors tests/golden/quant/
    rm -rf "$drift_dir"
    if [ "$drift_status" -ne 0 ]; then
        exit 1
    fi
    echo "quant fixtures reproduce byte-for-byte"
fi

echo "== structural invariants (dep-free core, contained unsafe mass)"
python3 - <<'INVARIANTS'
import pathlib, re, sys
fail = []

# 1. The Zig core has no dependencies, and no Rust is allowed inside it.
zon = pathlib.Path("core/build.zig.zon").read_text()
if re.search(r"\.dependencies\s*=\s*\.\{\s*\}", zon) is None:
    fail.append("core/build.zig.zon must declare exactly '.dependencies = .{}' (dep-free core)")
for stray in pathlib.Path("core").rglob("Cargo.toml"):
    fail.append("Rust crate inside the core: %s" % stray)

# 2. The CUDA driver mass lives only in hw/ (driver layer) and bench/ (a diagnostic
#    tool). Engine code must never name a CUDA symbol: it goes through hw/.
core_src = pathlib.Path("core/src")
cuda_syms = re.compile(r"\bcu[A-Z][A-Za-z]+\b|\bCU(deviceptr|context|stream|module|function|graph)\b")
allowed_dirs = ("hw/", "bench/")
for path in core_src.rglob("*.zig"):
    rel = str(path.relative_to(core_src))
    if rel.startswith(allowed_dirs) or rel.startswith("generated"):
        continue
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if line.lstrip().startswith("//"):
            continue
        if cuda_syms.search(line):
            fail.append("CUDA symbol outside core/src/{hw,bench}/: %s:%d" % (rel, lineno))

# 3. The server may depend on crates; the core may not. Report the count for the log.
lock = pathlib.Path("Cargo.lock").read_text()
packages = lock.count("[[package]]")

if fail:
    print("\n".join("FAIL " + f for f in fail))
    sys.exit(1)
print("invariants ok: zig core dep-free, no rust in core/, CUDA mass confined to hw/+bench/, "
      "server dependency graph = %d crates" % packages)
INVARIANTS
echo "== gate passed"
