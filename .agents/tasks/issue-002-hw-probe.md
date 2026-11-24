# I-02 — hardware probe: measure the real machine, replace every M-todo in THEORY.md

## Working agreement

- Sole writer in your worktree. Read `plan.md`, `THEORY.md`, and
  `.agents/skills/tiger-style/` first. All new Zig must be tiger-style compliant.
- Builds on I-01 (same worktree, already landed): `core/build.zig`, the dlopen
  driver layer in `core/src/hw/`, `tools/gate.sh`.
- 4 × RTX 5060 Ti behind **one bifurcated PCIe 4.0 x16 (4 × x4)** with **P2P
  enabled by a patched driver**. The whole point of this issue is to find the
  real numbers, which will be far below spec sheets.
- Serialize GPU work: one measurement at a time, `timeout`-wrapped, no other
  lane is running GPU work.

## What to build

`core/src/hw/` gains a benchmark module; `core/src/bench/main.zig` becomes a
second build target so `zig build probe` produces `core/zig-out/bin/inferno-probe`.

### Measurements (all: warm-up once, then **median of 6**)

1. `device_info[]`: idx, name, cc, total VRAM, SM count (`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`),
   L2 bytes, `cuDeviceGetAttribute` clock rates.
2. `h2d_bw_gib_s[]`, `d2h_bw_gib_s[]` per GPU — pinned host buffer
   (`cuMemHostAlloc` with `CU_MEMHOSTALLOC_PORTABLE`) of 1 GiB, async copy on a
   stream, `cuStreamSynchronize`; repeat at 64 MiB, 256 MiB, 1 GiB and report each.
3. `d2d_bw_gib_s[]` per GPU — 1 GiB intra-device copy.
4. `p2p[from][to]`: `cuDeviceCanAccessPeer` matrix, then for every ordered pair
   with access enabled via `cuCtxEnablePeerAccess`: 256 MiB D2D bandwidth, and
   **small-transfer latency** (4 KiB copy, median of 200, in µs).
5. Concurrency: aggregate H2D for 4 GPUs launched simultaneously on 4 streams —
   this is the real shared-link number (`aggregate_h2d_gib_s`).
6. Host: `memcpy` bandwidth over a 2 GiB buffer (call `memcpy` in a loop, median
   of 6), `std.Thread.getCpuCount()`.
7. Host↔device round-trip latency for 64 B (median of 200, µs) — relevant to
   any per-token PCIe traffic.

Do **not** attempt matmul throughput (no kernels yet); a later issue adds it
after I-06.

### Output

- `bench/hw-probe.json` (gitignored) with the schema above plus
  `{ "generated": "<iso8601>", "cuda": 13020, "driver": "610.57" }`.
- stdout: a markdown table ready to paste into THEORY.md, plus a one-line
  summary of the three numbers that matter most (per-GPU HBM BW, aggregate
  host↔device BW, worst P2P BW).
- Exit non-zero if any GPU fails to initialize or if P2P is unavailable between
  any pair (report which pair).

### Correctness rules

- Every driver call: check the `CUresult`, map to a closed error set, no
  `catch unreachable`.
- Free every allocation; verify `cuMemGetInfo` returns to the starting value at
  the end (assert leakage < 32 MiB per device).
- No dynamic allocation after init except the measurement buffers, which are
  freed before exit (document the exception with a comment).

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 300 ./zig-out/bin/inferno-probe) | tee /tmp/probe.txt ; echo PROBE_EXIT=$?
python3 - <<'EOF'
import json
d = json.load(open("bench/hw-probe.json"))
assert len(d["device_info"]) == 4, d["device_info"]
# Per-GPU floors are LINK-bound: Gen4 x4 = 7.88 GB/s = 7.34 GiB/s raw, so ~6.7 GiB/s
# is the honest ceiling. >=5 catches a host-bounce or a wrong-direction measurement.
assert all(v >= 5.0 for v in d["h2d_bw_gib_s"]), d["h2d_bw_gib_s"]
assert all(v >= 5.0 for v in d["d2h_bw_gib_s"]), d["d2h_bw_gib_s"]
# This is the floor with teeth: it proves all four x4 links are usable CONCURRENTLY,
# which the TP=4/EP=4 design depends on. Gen4 x16 = 31.5 GB/s = 29.3 GiB/s raw.
assert d["aggregate_h2d_gib_s"] >= 20.0, d["aggregate_h2d_gib_s"]
# Device-local copy, two explicit bases (never one number with a basis string):
# copied-once must be 20x any PCIe path; HBM traffic (read+write) must approach spec.
assert all(v >= 150.0 for v in d["d2d_copied_gib_s"]), d["d2d_copied_gib_s"]
assert all(v >= 300.0 for v in d["d2d_hbm_gib_s"]), d["d2d_hbm_gib_s"]
assert len(d["p2p"]) == 16, "expected a 4x4 matrix"
print("probe OK")
EOF
```

### Reporting bases (mandatory, no ambiguity)

Every bandwidth field carries an explicit name that states its basis, and the
JSON also carries a `basis` object of one-line strings:

| field | basis |
|---|---|
| `h2d_bw_gib_s`, `d2h_bw_gib_s` | bytes moved across the link, counted **once** |
| `aggregate_h2d_gib_s` | same basis, 4 GPUs on 4 concurrent streams |
| `d2d_copied_gib_s` | device-local copy, bytes copied **once** |
| `d2d_hbm_gib_s` | device-local copy, HBM traffic = read + write = 2 × copied |
| `p2p[i][j].bw_gib_s` | bytes copied **once**, over the P2P link |
| `p2p[i][j].latency_us` | median of 200, 4 KiB transfer, round trip |

Also record `link_gib_s_theoretical` per GPU (computed from the reported
`pcie.link.gen`/`width`, Gen4 x4 = 7.34) and print the achieved percentage, so the
number can be audited rather than believed. `d2d_hbm_gib_s` is the figure that
goes into THEORY.md §2's decode ceiling (every weight byte is read once);
`d2d_copied_gib_s` must never be used for that.

The d2d floors are real floors, not decoration: they must beat every PCIe path by
an order of magnitude, proving the copy is device-local. If one fails, the
measurement is wrong — fix it. Do not lower a floor: if physics makes a floor
impossible, report it to the orchestrator instead (this happened: the original
`h2d > 10` floor was above the x4 Gen4 ceiling and was corrected).

Additionally, you may update `THEORY.md` §1 and §2 with the measured numbers
(the M-todo rows), since no other lane writes that file.

## Report format

Implemented / Changed files / Validation (paste the markdown table and the gate
output) / Open risks (which numbers surprise you vs THEORY.md and why) /
Recommended next step.
