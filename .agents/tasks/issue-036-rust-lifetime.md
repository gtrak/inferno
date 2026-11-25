# I-036 — Rust owns lifetime policy: opaque handles across the C ABI

**Why this is first.** It is the prerequisite for I-037 and I-038, and it is the only item on the list
that removes a known weakness rather than relocating code: `hw/epochs.zig` checks write-before-read and
handle validity **at runtime**, and a runtime guard is decoration if it can be compiled out of a release
build (plan.md §7j). Rust's borrow checker makes the same guarantee statically and free.

**Migration rule that governs all three issues:** every step is **additive and behaviour-neutral** until
the step that explicitly hands ownership over. No step may leave the engine in a state where two sides
believe they own the same device memory — that is double-free and use-after-free at a language
boundary, which is worse than either language's own failure modes.

## Scope

Define a handle-based stepping ABI, and make Rust the owner of handle lifetime.

```c
// Model: device memory allocated and owned by Zig, lifetime decided by Rust.
inferno_model* inferno_model_open(const char* manifest_json, uint32_t len);
void           inferno_model_close(inferno_model*);

// Sequence: KV + recurrent state inside the model's arena.
inferno_seq*   inferno_seq_open(inferno_model*, const uint32_t* ids, uint32_t count);
void           inferno_seq_close(inferno_seq*);
int32_t        inferno_seq_reset(inferno_seq*);
int32_t        inferno_seq_step(inferno_seq*, uint32_t* out_token, float* out_stats, uint32_t stats_cap);
```

Rust side (`server/src/engine/`):

- `Model` and `Sequence` newtypes with `Drop`; `!Send`/`!Sync` where the underlying context is
  single-context, so the compiler refuses cross-thread use instead of the epoch registry.
- A handle must not outlive its model: encode that as a lifetime parameter, not as a runtime check.
- A closed sequence must be unusable: the API takes `&mut Sequence` and `close` consumes `self`.

Zig side:

- Handles are indices into a fixed-capacity table with a generation counter (no allocation after init).
- Every entry point asserts `handle.generation == slot.generation` and returns a distinct error code for
  a stale handle. **No panic across the C ABI** — errors are return codes.
- The existing `epochs` registry stays untouched in this issue. Deleting it is the *last* step of
  I-037, after Rust demonstrably owns the lifetimes; removing it earlier would trade a real runtime
  check for an unproven static one.

## Out of scope

The decode loop (I-037) and the load pipeline (I-038). This issue does not change any arithmetic, any
kernel, or any placement decision.

## Alignment instrument (what "tests keep things aligned" means here)

The guard is the existing suite plus the exact-ids architecture gate:

- `zig build test` stays green (255/261, 6 skipped at the time of writing; the count must never *drop*
  without the removed test being named in the report).
- Architecture gate unchanged: 16 greedy ids exactly `169 252 39 151 132 239 223 20 198 120 47 121 121
  144 96 19`, `ulp_error ≤ 16` (measured 0.07), cached == full recompute, deterministic across runs.
- New tests must fail if reverted: (a) a closed sequence refuses a step, (b) a stale handle is refused
  with an error code and not a crash, (c) a model cannot be closed while a live sequence exists, or if
  it can, the sequence is invalidated with a defined error rather than UB.
- Per plan.md §7e, bit-exactness is a diagnostic: any numerical difference across this move is
  *reported*, not gated. Token ids are gated.

## Acceptance

Suite green, architecture gate unchanged, the three new refusal tests present and passing, and a Rust
test that creates and drops a `Sequence` without leaking (Rust-side, using the existing telemetry).
