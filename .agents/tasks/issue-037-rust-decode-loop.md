# I-037 — the decode loop moves to Rust

**Depends on I-036** (opaque handles with Rust-decided lifetimes).

## Why

`model/generate.zig` (278 lines) plus the per-token control flow in `model/runtime_check.zig` is pure
control flow wrapped around KV and recurrent-state ownership: sample, check stop conditions, decide
whether to keep or roll back the last token, append to history. That is exactly the shape the borrow
checker is good at, and it collapses "request → tokens → SSE" into one language instead of splitting
request lifetime (already Rust, `admission.rs`) from sequence lifetime (currently Zig).

Measured cost of the boundary: one Rust→Zig hop is ~20 ns against ~1.68 ms/token of launch enqueue —
**0.0002 %**. Chattiness is not the question; ownership is. Keep the boundary **per token**, not per
launch.

## Scope

- Stop surface (`Sequence`/`Stop`, EOS, ceiling, predicate, output-full) moves to Rust.
- Sampling and the speculative accept/rollback decision move to Rust. The kernels stay in Zig.
- `inferno_seq_step` (I-036) is the only call on the hot path; everything else is per-request.
- **The GDN snapshot landmine moves with it**: GDN updates state in place, so a speculative rollback
  needs a ~113 MB/sequence snapshot. Whichever language owns the loop owns that copy discipline. Rust
  should make "you cannot advance without a snapshot you can restore" a type-level state machine, not
  a commented convention.
- **Last step of this issue**: delete `hw/epochs.zig`'s runtime checks for the paths Rust now owns,
  and only for those. If any check cannot be replaced by a static guarantee, it stays and the reason
  is recorded in the issue report — the point is to remove decoration, not to remove guards.

## Out of scope

Kernels, attention, MoE, placement, and the C ABI's existing engine-level entry points (I-016), which
keep working.

## Alignment instrument

- Suite green; architecture gate unchanged (16 ids exact, `ulp_error ≤ 16`, cached == recompute,
  deterministic).
- `model/runtime_check.zig`'s 12 passing checks are the specification of the loop's behaviour —
  reimplement them as Rust tests against the new API, and **delete the Zig copies only after the Rust
  ones pass**.
- Stop-surface tests must fail if reverted: EOS at exactly the boundary, ceiling, predicate, output-full.
- New test: a speculative rollback restores GDN state bit-identically after a rejected token.

## Acceptance

Decode loop is Rust; Zig exposes stepping only. Suite green, gate unchanged, runtime-check
equivalents present in Rust. Per-token cost reported before and after (not gated).
