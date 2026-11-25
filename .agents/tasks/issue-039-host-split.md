# I-039 — split the Rust crate: `host/` (integration) vs `server/` (transport)

**Mechanical, behaviour-neutral, do this before moving behaviour into Rust.** It is small and it
prevents the next three issues from landing in the wrong place.

## Why

The Rust crate is named `server`, so the integration logic it already contains — `admission`,
`engine`, `tokenizer`, `stop`, `template`, `config` — reads as if it belonged to the HTTP layer. Only
`api/`, `server.rs` and `main.rs` are transport. Every architecture conversation has to first argue
past the name, and "move loading to Rust" sounds like "put loading in the HTTP handler". See
`.agents/design/ffi-boundary.md` ("What the boundary actually is").

## Scope

Turn the single-crate workspace into two crates under the existing workspace:

| crate | role | contents |
|---|---|---|
| `host/` → `inferno-host` | integration + ownership | `admission`, `engine`, `tokenizer`, `stop`, `template`, `config`, `telemetry`, `metrics`; and the future home of load pipeline, sequence control, lifetime policy |
| `server/` → `inferno-server` | transport only | `api/`, `server.rs`, `main.rs`; depends on `inferno-host` |

Rules:

- **No behaviour change.** Module moves, `pub use` re-exports where a test or the binary needs a path,
  and `Cargo.toml` edits only. If a move forces a logic change, stop and report it — that is a finding.
- `host` must not depend on `axum`. If it does today through a type that leaked out of `api/`, move the
  type rather than the dependency.
- `server` keeps the binary name `inferno-server`, so `tools/gate.sh`'s `--probe` step and any operator
  habit keep working.
- The dependency-graph invariant in `tools/invariants.py` reports a crate count; update what it reads
  and keep it truthful.

## Alignment instrument

- The existing Rust test suite and `cargo test` pass unchanged; Zig side untouched.
- `bash tools/gate.sh` green, including the `--probe` step against the real binary.
- Integration tests in `server/tests/` keep passing — they start the whole server in-process, so they
  are the check that the split did not sever a wire.

## Acceptance

Two crates, same binary behaviour, same tests, gate green, and a `host` crate that a future
load/sequence/scheduler issue can grow into without touching transport.
