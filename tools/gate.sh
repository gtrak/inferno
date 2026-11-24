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

echo "== structural invariants (dep-free core, contained unsafe mass)"
python3 - <<'INVARIANTS'
import pathlib, re, sys
fail = []
zon = pathlib.Path("core/build.zig.zon").read_text()
if re.search(r"\.dependencies\s*=\s*\.\{\s*\}", zon) is None:
    fail.append("core/build.zig.zon must declare exactly '.dependencies = .{}' (dep-free core)")
lock = pathlib.Path("Cargo.lock").read_text()
packages = lock.count("[[package]]")
if packages != 1:
    fail.append("Cargo.lock must contain exactly 1 package (the server crate), found %d" % packages)
core_src = pathlib.Path("core/src")
cuda_syms = re.compile(r"\bcu[A-Z][A-Za-z]+\b|\bCU(deviceptr|context|stream|module|function|graph)\b")
allowed = {"hw/cuda_driver.zig", "hw/dl.zig"}
for path in core_src.rglob("*.zig"):
    rel = str(path.relative_to(core_src))
    if rel in allowed or rel.startswith("generated"):
        continue
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if line.lstrip().startswith("//"):
            continue
        if cuda_syms.search(line):
            fail.append("CUDA symbol outside core/src/hw/: %s:%d" % (rel, lineno))
if fail:
    print("\n".join("FAIL " + f for f in fail))
    sys.exit(1)
print("invariants ok: core dep-free, Cargo.lock 1 package, CUDA mass confined to core/src/hw/")
INVARIANTS

echo "== gate passed"
