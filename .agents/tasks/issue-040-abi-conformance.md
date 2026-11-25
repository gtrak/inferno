# I-040 — the Rust ABI conformance suite (and the tests that must cross the boundary)

## Why

The C ABI is currently the least-tested seam in the repo: `core/src/c_api.zig` exports 4 symbols and has
7 Zig tests, all trivial (version string, cubin bounds, probe JSON). **No Rust test drives it at all.**
Meanwhile the behaviour Rust is about to own is tested inside Zig, bypassing the boundary. So a bad ABI
— wrong error code, a handle that survives `close`, a stale-generation reuse — would be found by
production, not by tests.

This issue makes the boundary a tested client surface. It is the companion to I-036 (which defines the
handles) and it must not wait for the whole migration: every export added by I-036/I-037/I-038 gets
conformance tests **as it lands**.

## Scope

A Rust test suite (in `host/tests/` after I-039, `server/tests/` before) that treats the Zig core as a
**third-party library** and asserts the contract:

1. **Lifecycle**: `model_open` → `model_close`; `seq_open` → `seq_close`; open/close in a loop does not
   leak (Rust-side counter or the Zig telemetry).
2. **Refusals, not crashes**: a closed sequence refuses `seq_step` with a defined code; a stale handle
   (generation mismatch) is refused with a *distinct* code; neither panics, and the process stays alive.
3. **Ownership**: a model closed while a sequence is live produces a defined error, not UB.
4. **Determinism**: the same seed/prompt stepped twice yields identical ids, in one process.
5. **The architecture gate through the boundary**: `model_open` the tiny fixture, step it, and assert
   the 16 golden ids `169 252 39 151 132 239 223 20 198 120 47 121 121 144 96 19`. This is the ABI's
   first real client and the cheapest proof that the boundary carries real numerics.
6. **Thread rules**: a handle that is `!Send`/`!Sync` cannot cross threads — a compile-fail test
   (`trybuild` or a `compile_fail` doctest), so the restriction is enforced by the compiler, not comments.

Device rules: same-device concurrency stays forbidden. The suite pins one device per test (or runs
serially) exactly as the Zig shards do, so it cannot reintroduce the measured 2-of-5 flake.

## Test moves that belong with other issues (do not duplicate them here)

| Zig test today | moves to Rust in |
|---|---|
| `model/runtime_check.zig` (reset equivalence, snapshot/restore, stop surface, cached == recompute) | I-037 |
| decode loop + sampling behaviour (currently only via `tiny_check`) | I-037 |
| `weights/shard.zig` pipeline + `weights/manifest.zig` completeness | I-038 |
| production-safetensors refusals | I-038 |
| **stays Zig**: kernels vs oracle, `hw/`, forward numerics vs trace, quad-rank forward | — |

## Alignment instrument

- Zig suite green, and its count may only change by tests this issue names as moved.
- Rust suite green; the gate test asserts the exact 16 ids.
- `tools/gate.sh` green.
- Each new export has at least one refusal test. An export with no client test is not done.

## Acceptance

The ABI has a Rust client suite covering lifecycle, refusals, determinism, the golden ids by hand, and
the thread rules; every export added from I-036 onward is covered by a test that fails if the export is
removed or its error code changes.
