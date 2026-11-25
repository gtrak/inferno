# Run log — overnight 2026-11-25

Read this first. Newest entries at the top of each stage. Numbers here are measured, not remembered.

---

## Stage 1 — I-036 handles (ABI foundation) — DONE, merged 3e3e6ae

**Lane outcome:** timed out at its 30-minute first-attempt ceiling while writing its report, but had
finished the work and left it uncommitted. Read the log (263/269 zig, 73 rust), verified it myself, banked
it. No retry needed — a retry would have re-read 1,700 lines to reproduce a report I could write from the
diff.

**Shipped:** `inferno_model_open/close`, `inferno_seq_open/close/reset/step`, `inferno_handle_counts`.
Zig handle table with generation counters (`core/src/model/handles.zig`), refusals as codes, no panic
across the ABI. Rust `Model`/`Sequence` with `Drop`, lifetime on `Sequence`, `!Send`/`!Sync` — the last
three enforced by **compile_fail doctests that actually run** (3 passed). `epochs` untouched on purpose:
deleting a runtime check belongs to I-037, which replaces it with a static guarantee.

**Verified by me, not taken on trust:** 263/269 zig tests, 6 skipped, +8 tests and none removed; 73 rust
tests in 8 binaries; `cargo test --doc` shows the 3 compile_fail cases executing; gate passed in 142 s
(budget 160 s) including `--probe`; architecture gate still reproduces the 16 golden ids through the new
ABI.

**Finding for I-040:** `server/build.rs`'s new mandatory symbol check uses a substring search for the
symbol name in the archive bytes. That cannot produce a false *failure*, but it could produce a false
*pass* (the name appearing in a string table without a definition). I-040 should assert definitions
(`T` symbols from `nm`-equivalent) rather than presence of the string, and add the negative test: strip a
symbol and confirm the build fails.

## Stage 1 (original dispatch) — I-036 handles (ABI foundation)

**Status:** superseded by the DONE entry above. Kept for the dispatch context.

**Spec:** `.agents/tasks/issue-036-rust-lifetime.md`

**Context at dispatch:** `core/src/c_api.zig` exports 4 symbols (`inferno_cubin_count`, `inferno_cubin_at`,
`inferno_version`, `inferno_probe_devices`). `server/src/engine/ffi.rs` declares 7 engine-ABI symbols
(`inferno_core_init/shutdown/version/submit/cancel/set_token_cb/metrics`) that Zig does not yet export, so
`server/build.rs` sets `cfg(inferno_core_abi_absent)` and the Rust crate builds with not-implemented
fallbacks. Baseline suite: 255/261 tests, 6 skipped, `zig fmt` clean.

**Known trap for stage 3:** that fallback cfg would let conformance tests *silently skip*. I-040 requires a
missing or partial ABI to be a hard failure with each symbol asserted individually.

---
