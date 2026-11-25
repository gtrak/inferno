# Run log — overnight 2026-11-25

Read this first. Newest entries at the top of each stage. Numbers here are measured, not remembered.

---

## Stage 1 — I-036 handles (ABI foundation)

**Status:** lane running in slot-1, dispatched after `dd7c6e6`.

**Spec:** `.agents/tasks/issue-036-rust-lifetime.md`

**Context at dispatch:** `core/src/c_api.zig` exports 4 symbols (`inferno_cubin_count`, `inferno_cubin_at`,
`inferno_version`, `inferno_probe_devices`). `server/src/engine/ffi.rs` declares 7 engine-ABI symbols
(`inferno_core_init/shutdown/version/submit/cancel/set_token_cb/metrics`) that Zig does not yet export, so
`server/build.rs` sets `cfg(inferno_core_abi_absent)` and the Rust crate builds with not-implemented
fallbacks. Baseline suite: 255/261 tests, 6 skipped, `zig fmt` clean.

**Known trap for stage 3:** that fallback cfg would let conformance tests *silently skip*. I-040 requires a
missing or partial ABI to be a hard failure with each symbol asserted individually.

---
