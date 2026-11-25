# Target state — what "coherent" means, concretely

This is the end state the migration moves toward. Ordering does not matter (user, 2026-11-25); this
does. Every issue is judged by whether it lands closer to this, and no issue may create a **second** way
to do something that is listed here as **one**.

## Three layers, by role

| layer | owns | does not own |
|---|---|---|
| **Zig `core/`** | inference: kernels, the forward pass, device memory, and the layout algebra kernels agree with | when work runs, what is loaded, where bytes came from, request lifetime |
| **Rust `host/`** | integration + ownership: lifecycle, load pipeline, weight provenance, sequence control, scheduling, stop conditions, tokenisation | arithmetic on tensors, launch sequencing on the hot path |
| **Rust `server/`** | transport: HTTP, SSE, CLI | anything above |

## Ownership map (every module, one home)

### Zig — stays, permanently

| module | role |
|---|---|
| `hw/` | the unsafe seam by design: driver, context, buffers, guard bands, epochs, kernel launch |
| `kernels/` | CUDA kernels and their launch wrappers |
| `mem/placement.zig` | layout algebra: **the kernel contract**, exposed to Rust as *data* (sizes, strides, roles) |
| `model/exec.zig` | **the single forward**: embed → layers → mixer → lm_head |
| `model/glue.zig` | kernels the block kernels take as given |
| `model/state.zig` | per-sequence device caches (the state itself; the *policy* for resetting it is host) |
| `model/plan.zig` | static layer schedule derived from a config |
| `model/snapshot.zig` | device-side state copy/restore (the *decision* to snapshot is host) |
| `model/multi.zig` | rank/tier harness around `exec.Model`: peer exchange, hot/cold slabs, expert tier |
| `ref/` | the fp32 oracle every kernel is judged against — never moved, never modernised for its own sake |
| `bench/` | measurement harness |
| `c_api.zig` | the ABI surface (below) |

### Rust `host/` — integration and ownership

| responsibility | today's location |
|---|---|
| request admission + scheduling | `server/src/admission.rs` |
| decode loop, stop surface, sampling, speculative decisions | **Zig** `model/generate.zig` → moves (I-037) |
| sequence lifetime, model lifetime, handle validity | **Zig** runtime `hw/epochs.zig` + caller discipline → moves (I-036/I-037) |
| load pipeline: mapping lifetime, ordering, pacing, capacity negotiation, verification | split: Zig `weights/shard.zig` → moves (I-038) |
| production safetensors parsing | Zig `weights/safetensors.zig` → moves (I-038) |
| config reading and validation | Zig `weights/config.zig` → moves (I-038) |
| weight provenance (which dtype, which tier, which rank) | new, once loading is host-side |
| tokenizer, stop strings, chat templates, telemetry | already Rust |

### Zig — becomes test-only (fixtures, not a second production path)

| module | final role |
|---|---|
| `model/load.zig` | loads the **tiny fixture** into `exec.Model` for the architecture gate. Renamed to say so; never grows a real-checkpoint path |
| `weights/safetensors.zig` (fixture-grade reader) | parses the tiny fixture and quant goldens. A **different artifact** from the 132 GB checkpoint, so this is a split by artifact, not a duplicated parser |
| `model/tiny_check.zig`, `model/runtime_check.zig`, `model/config_bridge.zig` | tests and the one config translation |

## The ABI surface (final, small, data-oriented)

```
lifecycle    model_open(manifest) / model_close
             seq_open(model, ids, n) / seq_close / seq_reset
work         seq_step(seq, out_token, out_stats, cap)      // one call per token
layout       placement_query(out, cap) -> records          // Rust decides; Zig answers sizes/strides
identity     version() / probe_devices(out, cap) / cubin_*(i)
```

Rules: exactly one side owns device memory (**Zig**); Rust decides *when* and *what*, never *how big*
(that is `placement.zig`) and never *how* (that is a kernel). No panic crosses the ABI — every failure is
a return code. Handles are generation-counted; a stale handle is an error, not UB.

## Invariants — the checklist "coherent" means

1. **One forward.** `exec.Model` is the only forward; `multi.zig` harnesses it, the tiny path configures
   it, the real path feeds it different weights. No second implementation, ever.
2. **One production parser**, in `host/`. Zig parses fixtures only.
3. **One placement algebra**, in Zig, consumed by both kernels and host (as data).
4. **One owner of device memory** (Zig). Rust owns the *decisions* about it.
5. **One authority per behaviour.** Where Rust takes over a loop, the Zig test that encoded it is
   replaced, not duplicated — two suites describing one behaviour drift.
6. **No runtime guard where a static one exists.** `epochs` survives only for lifetimes no language
   proves; each survivor is listed with its reason.
7. **No allocation on the hot path**; bulk paths take an arena and allocate O(1).
8. **A test lives in the layer that owns the behaviour**, and cross-boundary behaviour is tested from the
   consuming side. No behaviour has two authorities.
9. **The architecture gate holds at every step**: 16 greedy ids exactly, `ulp_error ≤ 16`, cached ==
   full recompute, deterministic. This is the alignment instrument, and it is exact.

## Test ownership (user, 2026-11-25)

**A test belongs to the layer that owns the behaviour.** Cross-boundary behaviour is tested from the
**consuming** side, through the ABI. This is not tidiness — it is the only thing that keeps the ABI honest.

The measurement that motivated it: the C ABI has **4 exports and 7 Zig tests, all trivial** (version
string, cubin bounds, probe JSON), and **not one Rust test drives it**. Meanwhile the behaviour that
Rust is about to own — 33 `model/` tests including the architecture gate and `runtime_check`'s reset /
snapshot / stop / cached-vs-recompute checks — is tested *inside* Zig, bypassing the boundary entirely.
So today the seam the production consumer uses is the least-tested thing in the repo, and the tests that
would catch a bad ABI are written on the wrong side of it.

| behaviour | owner | test lives in | style |
|---|---|---|---|
| kernels vs fp32 oracle, dtype rounding | Zig | Zig | calls kernels directly with `ref/` |
| `hw/` buffers, guard bands, epochs, contexts | Zig | Zig | calls internals |
| forward numerics (ulp vs committed trace) | Zig | Zig | `tiny_check` numerics |
| quad-rank forward, peer exchange, expert tier | Zig | Zig | `multi.zig` |
| **the ABI itself**: lifecycle, stale/closed handles, generation counters, determinism across two runs, one-owner rule | **boundary** | **Rust** | calls the FFI as a client |
| **the architecture gate's ids** (16 exact) | boundary | **Rust** | through `model_open` → `seq_step` |
| sequence control: reset equivalence, snapshot/restore, stop surface, cached == recompute | host | **Rust** | through the FFI (moves with I-037) |
| decode loop, sampling, speculative accept/rollback | host | **Rust** | through the FFI (I-037) |
| load pipeline: mapping lifetime, ordering, pacing, capacity, verification | host | **Rust** | (I-038) |
| production parser refusals (overlap, trailing bytes, header past EOF, …) | host | **Rust** | move with the parser (I-038) |
| fixture-grade parsing, quant goldens | Zig | Zig | tiny artifact only |

**Two claims, two tests, not a duplicate authority.** The Zig test asserts *numerics* against the
committed trace (`ulp_error`, logit deltas). The Rust test asserts the *contract* through the ABI (same
golden ids, stable handles, determinism). Different claims about the same model — both are load-bearing
and neither implies the other.

**The discipline this buys:** if a test is awkward to write in Rust because it needs a Zig internal, that
is evidence the behaviour has not actually moved, or the ABI is too narrow. The test's address is the
design check. Corollary: Rust tests must never reimplement device setup; if one wants to, the setup is
misplaced.

## Where the inconsistencies are today (what the migration is for)

| # | inconsistency | resolved by |
|---|---|---|
| 1 | decode loop in Zig, request lifetime in Rust — one sequence has two owners | I-037 |
| 2 | handle validity is a **runtime** epoch check that a release build can drop | I-036, I-037 |
| 3 | production parser + pipeline in Zig while Rust is meant to integrate | I-038 |
| 4 | integration logic lives in a crate named `server` | I-039 |
| 7 | the ABI is an **untested seam**: 4 exports, trivial Zig tests, zero Rust clients | I-040 |
| 5 | `exec.Model` can only be fed by the tiny fixture | I-017c (weight source becomes an interface) |
| 6 | three config shapes (`TextConfig`, `RefConfig`, bridge) | keep, but bridge is the **only** translation and it is tested against the real checkpoint |

## Deliberate non-goals

- Moving `ref/` (the oracle is the authority), `kernels/`, `hw/`, or `bench/`.
- Moving `placement.zig`: it is the kernel contract, and bug class #1 was two sizing formulas drifting
  between modules.
- Making the tiny path disappear: it is the architecture gate's subject, and the gate is how every
  migration step is verified.
- Performance work during migration. First make it coherent; then make it fast, on a stable shape.

## Definition of done

- Rust `host/` owns load, sequence control, scheduling, and lifetimes; Zig receives handles and executes.
- `c_api.zig` is the surface above and nothing else.
- The architecture gate passes unchanged; the suite passes; `tools/gate.sh` passes; `epochs` is reduced
  to the entries with a written reason.
- A reader can find every responsibility in exactly one place, and that place is the layer the table at
  the top of this document says it should be.
