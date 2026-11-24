# I-033 — the `epochs`/`guards` registries are global and keyed by pointer, so device contexts alias

Found by the I-014b lane and reported rather than hidden: with guard bands enabled
(`INFERNO_GUARD_BANDS=1`) the global pointer-keyed registries in `hw/epochs.zig` and `hw/buffer.zig`
**alias across device contexts**. Two primary contexts can hold the same virtual address; a global
map keyed by that value then confuses one context's buffer for another's. Tests run with bands off,
so this is latent - and the lane also logged **one unexplained flake in six suite runs** where the
`hw/buffer` guard test aborted with no message (3/3 isolated, 2/2 in the final binaries). That flake
is plausibly this bug.

This is the second of the only two type-system-catchable bugs of 2026-11-24: it is
**ownership-across-calls**, exactly the class `plan.md` §7f assigns to Rust's borrow checker. It is
the strongest concrete evidence for moving buffer *lifetime policy* (not launch sequencing) across the
boundary.

## Fix

Make the registry **context-scoped**, not global:

- key by (context identity, address), or
- give each `Context` its own registry instance and pass it explicitly, or
- make the buffer handle carry its owning context so a lookup in the wrong context is a hard error,
  naming both.

Any of these is acceptable; the requirement is that a stale or cross-context handle **cannot** silently
resolve. Prefer the one that makes the error message name the buffer kind and both contexts.

## Tests that must exist and must be able to fail

1. Guard bands ON, two contexts, a buffer in each: each verifies against its own band, and no lookup
   resolves across contexts. Assert the specific rejection when forced.
2. The `hw/buffer` guard test run repeatedly (>= 10) with bands on, to chase the logged flake; if it
   does not reproduce, record that with the run count rather than claiming it fixed.
3. `INFERNO_GUARD_BANDS=1 bash tools/gate.sh` must pass — today the suite is run with bands off,
   which means the guard machinery is not exercised in the default gate at all. Fix that: a guard
   that only runs when someone remembers to set an env var is the decoration pattern.

## Gates

```
INFERNO_GUARD_BANDS=1 bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build test) ; echo CORETEST=$?
```

Non-vacuity: force a cross-context lookup and paste the refusal.
