#!/usr/bin/env python3
"""Structural invariants of the inferno repository (run by `tools/gate.sh`).

These are the checks a compiler cannot make, and each one exists because a
specific defect got through without it:

- the Zig core is dependency-free and holds no Rust;
- CUDA mass lives only in `core/src/hw/` (the driver layer) and
  `core/src/bench/` (the diagnostic probe);
- **G1 (I-028)**: the two files that expose the raw driver entry-point table
  (`functions`, `check_result`) are importable only from `hw/` and `bench/`.
  Before this check, `model/multi.zig` imported both directly and the gate
  still passed: it looked for `cu[A-Z]` symbol names, and the import hid them.
  A raw `cuMemAlloc` outside `hw/` is now a gate failure rather than a review
  question.
"""

import pathlib
import re
import sys

fail = []

# 1. The Zig core has no dependencies, and no Rust is allowed inside it.
zon = pathlib.Path("core/build.zig.zon").read_text()
if re.search(r"\.dependencies\s*=\s*\.\{\s*\}", zon) is None:
    fail.append(
        "core/build.zig.zon must declare exactly '.dependencies = .{}' (dep-free core)"
    )
for stray in pathlib.Path("core").rglob("Cargo.toml"):
    fail.append("Rust crate inside the core: %s" % stray)

core_src = pathlib.Path("core/src")
allowed_dirs = ("hw/", "bench/")

# 2. The CUDA driver mass lives only in hw/ (driver layer) and bench/ (a
#    diagnostic tool). Engine code must never name a CUDA symbol: it goes
#    through hw/. Comments are prose, not calls.
cuda_syms = re.compile(
    r"\bcu[A-Z][A-Za-z]+\b|\bCU(deviceptr|context|stream|module|function|graph)\b"
)
raw_symbols = re.compile(r"\bcuMemAlloc[A-Za-z0-9_]*|\bcuMemcpy[A-Za-z0-9_]*|\bcuLaunchKernel\b")

# 3. G1: the raw driver and its context wrapper are private to the hardware
#    layer. Everything else imports `hw/api.zig`, whose re-exports do not
#    include the entry-point table.
raw_import = re.compile(
    r'@import\("(?:\.\./)*hw/(?:cuda_driver|cuda_context)\.zig"\)'
)
raw_call = re.compile(r"\bcuda_driver\.(?:functions|check_result)\s*\(")

for path in core_src.rglob("*.zig"):
    rel = str(path.relative_to(core_src))
    in_hw = rel.startswith(allowed_dirs) or rel.startswith("generated")
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        if not in_hw:
            if cuda_syms.search(line) or raw_symbols.search(line):
                fail.append("CUDA symbol outside core/src/{hw,bench}/: %s:%d" % (rel, lineno))
            if raw_import.search(line):
                fail.append(
                    "G1: %s:%d imports the raw driver/context; import hw/api.zig instead"
                    % (rel, lineno)
                )
            if raw_call.search(line):
                fail.append(
                    "G1: %s:%d calls the raw driver entry-point table outside hw/"
                    % (rel, lineno)
                )

# G3 (I-035): a bulk data path must never be handed the tracked test allocator.
# std.testing.allocator maps every allocation for leak checking; over a 299,545-tensor
# checkpoint that turns a 478 ms build into a run that does not finish in ten minutes
# (measured: arena 478 ms vs tracked >600 s). Bulk entry points take an arena.
bulk_entry_points = (
    "manifest.build(",
    "manifest.plan(",
    "manifest.load(",
    "shard.plan(",
    "shard.load(",
    "safetensors.open(",
)
tracked_allocator = re.compile(r"\bstd\.testing\.allocator\b")
for path in core_src.rglob("*.zig"):
    rel = str(path.relative_to(core_src))
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        if not tracked_allocator.search(line):
            continue
        for entry in bulk_entry_points:
            if entry in line:
                fail.append(
                    "G3: %s:%d passes std.testing.allocator to a bulk entry point (%s); "
                    "use testing_alloc.arena()" % (rel, lineno, entry)
                )

# 4. The server may depend on crates; the core may not. Report the count for the log.
lock = pathlib.Path("Cargo.lock").read_text()
packages = lock.count("[[package]]")

if fail:
    print("\n".join("FAIL " + f for f in fail))
    sys.exit(1)
print(
    "invariants ok: zig core dep-free, no rust in core/, CUDA mass confined to hw/+bench/, "
    "G1 raw driver imports confined to hw/+bench/, G3 bulk paths off the tracked allocator, server dependency graph = %d crates"
    % packages
)
