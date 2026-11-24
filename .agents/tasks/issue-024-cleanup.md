# I-024 — accumulated review findings (cleanup lane)

Small, disjoint fixes collected from the wave-1/2 review gates. Each is verified by
re-running the owning issue's gate. **No refactors, no new features.**

## Working agreement
- Sole writer. tiger-style binding. Do not edit `tools/gate.sh` or any issue spec.
- Do not edit `core/build.zig` unless a finding names it.

## Findings
### From I-007 (Rust server) — user-visible bugs
1. `server/src/telemetry.rs:26-29` + `core/src/config.rs`: `INFERNO_LOG=debug` aborts
   startup (`expected struct LogConfig`). Fix with `Env::prefixed(..).split("__").ignore(&["log"])`
   or rename the variable to `INFERNO_LOG_FILTER`. Add a test that `INFERNO_LOG=debug` starts.
2. `queue_ms` does not measure queue time (`server/src/api/generate.rs:169`). Measure
   from request accept to engine-submit completion; add a test that it is > 0 when the
   engine delays acceptance.
2b. (P2-3) the chat-golden gate is not self-contained: it must resolve its tokenizer
   from `INFERNO_MODEL_DIR` when set, else fall back to the committed
   `server/tests/fixtures/model/tokenizer.json`, and fail with a message naming the
   variable when a *configured* path is missing. **Do not add a precondition to
   `tools/gate.sh`** — that would make the gate fail on any machine without the 132 GB
   checkpoint, which is worse than the problem it fixes.
3. Any other P2 in the I-007 review report (read the artifact at
   `~/.pi/agent/sessions/--home-gary-dev-inferno--/subagent-artifacts/71816212-4caa-4c53-80f6-de14ac63edac_gate-reviewer_output.md`).

### From I-005 (CPU reference)
4. `core/src/ref/tests.zig`: three routing non-vacuity asserts are tautologies
   (`distinct > 0`, `distinct <= positions`, `ordered >= distinct`). Replace with
   `distinct > 1` and `distinct < positions` (both live: 5 and 12) so degenerate routing
   actually fails.
5. Compare the GDN cache tensors that are currently unused (`gdn/recurrent_state`,
   `gdn/conv_state`) — cheap and they are exactly what I-009 needs.

### From I-003 (weights)
6. `quant.zig` ignores the fixture's `pattern`/`codes` tensors: either assert the 16
   packed nibbles against `codes` or stop writing them.
7. tiger-style MINOR: 123 lines > 80 columns across `weights/*.zig`; `document_bytes: usize`
   should be an explicit width.

### From I-002 (probe) — mechanism now, measurement on an idle GPU
8. The "±0.5 %" stability claim lived only in the I-002 *worker report*; it is in no
   committed file (verified by grep over THEORY.md, `core/src/bench/*.zig` and the
   I-002 spec), so nothing in the tree needs correcting. What is missing is the
   mechanism: `core/src/bench/report_json.zig` must emit `stability_percent_max`,
   **null when unmeasured** (never 0 or a default that reads as a measurement), with
   a repeat runner (N ≥ 6). The *value* must be taken on an **idle host** —
   plan.md non-negotiable 4 forbids timing while other lanes hold the GPUs — so the
   measurement is scheduled separately and the field stays null until then.
   Amended 2026-11-24: the lane correctly refused to take a contaminated number.

## Gates
```
bash tools/gate.sh ; echo EXIT=$?
cargo test -p inferno-server ; echo SERVER=$?
(cd core && timeout 900 zig build refcheck) ; echo REF=$?
(cd core && zig build test) ; echo CORE=$?
```
All exits 0, and each finding has a test that would fail if the fix were reverted —
state which test pins which finding in your report.

## Report format
Implemented / Changed files / Validation / Open risks / Recommended next step.
