# Overnight runbook — Rust migration, then inference, quality, performance

User's instruction (2026-11-25, before sleeping): *"Get the rust migration done, then continue our
planned work, prove inference, prove quality, hill-climb performance."*

Target end state: `.agents/design/target-state.md`. Alignment instrument: the exact-ids architecture gate.

## Lane protocol (unchanged, restated so it survives a context restart)

- Worker model `opencode-go/deepseek-v4.1-flash`; one worktree slot per lane
  (`.agents/worktrees/slot-1..4`). Workers **never commit**, never touch main or another slot.
- **Worker timeout: 30 min on the first attempt, 90 min for every attempt after the first failure**
  (user, 2026-11-25). Only raise the ceiling after a lane has actually failed once — the first pass is
  where a badly-scoped brief shows up cheaply. 90 min is a ceiling, not a target: a lane that runs 90
  min and produces nothing committed has failed regardless.
- A timed-out worker still leaves green partial work; read its log for the last suite result, verify it
  myself, commit it, and **narrow the retry's scope** rather than re-sending the same brief with more time.

### Staying productive (user: "the work needs to stay productive")

- Every dispatch must be capable of ending in one of: a commit, or a written finding in this log. No
  lane may run without a defined output artifact.
- No idle waiting: while a lane runs, do verified non-conflicting work (read-only investigation, specs,
  review of a previous lane's diff). Never dispatch a second lane that touches the parent's files.
- If a lane returns "could not finish X", the parent splits X and says so — it does not silently retry
  the same larger task.
- Partial value is banked immediately: green + measured work gets committed before the next attempt.
- **Never trust a worker's green.** After each lane I re-run: `zig build test --summary all`,
  `cargo test`, `bash tools/gate.sh`, and record the numbers in the run log.
- Merge discipline: commit on the lane branch, `git merge --no-ff` into main, reset all slots to main.
- Every dispatch must carry: the spec path, the acceptance gates, the two measurement traps, G3 (bulk
  paths take an arena), and "state plainly what you could not do".

## Order of work

| # | issue | why now | done when |
|---|---|---|---|
| 1 | **I-036** handles (running) | prerequisite for everything | ABI exports exist; 3 refusal tests; gate ids unchanged |
| 2 | **I-039** crate split `host/` vs `server/` | behaviour must land in the right home | two crates, `host` has no `axum`, binary still `inferno-server`, gate includes `--probe` |
| 3 | **I-040** ABI conformance suite | the boundary is currently untested | Rust client suite: lifecycle, refusals, determinism, 16 golden ids, thread rules; **missing ABI is a hard failure, not a skip** |
| 4 | **I-037** decode loop → Rust | removes the runtime `epochs` checks it replaces | Rust owns loop+stop+sampling; Zig steps only; runtime_check equivalents in Rust first, then Zig deleted |
| 5 | **I-038** load pipeline → Rust | loading is integration | Rust owns mapping lifetime/ordering/pacing/capacity/verification + production parser; Zig keeps placement algebra (as data) and the fixture reader |
| 6 | **I-017c** real weights → `exec.Model` | **proves inference** | real checkpoint through the ABI → finite, deterministic, plausible logits; layer-level relative-RMS vs fp32 oracle reported |
| 7 | **I-031 grader** quality | **proves quality** | short context (2048) vs the published bf16 trace: KL, ΔPPL, router top-k agreement; ulp in the output format |
| 8 | perf hill-climb | only on a stable, correct shape | honest composed throughput vs 150 tok/s @ 262144 decode and 5000 tok/s prefill, measured once, same conditions |

4 and 5 may run in parallel **only after** 2 and 3 are merged, and only if their files do not overlap
(037 owns sequence/decode/stop; 038 owns weights/load). If they collide, serialize.

## Milestones and what counts as proof

**M-A boundary (1–3).** Zig infers, Rust integrates, and the ABI has a client suite. Proof: Rust tests
call the FFI; stale and closed handles are refused with distinct codes; `!Send`/`!Sync` enforced by a
compile-fail test.

**M-B Rust owns integration (4–5).** Proof: no Zig module owns a sequence lifetime; `epochs` reduced to
entries with a written reason; production parser and pipeline are host-side; the Zig suite still passes
with only the named tests moved.

**M-C inference proven (6).** Proof, in order: logits finite; two runs identical; `max|logit|` sane;
per-layer relative-RMS against the fp32 oracle reported as numbers; and — the milestone — **first
coherent tokens** from the real checkpoint. There is **no golden id for the real checkpoint**; do not
invent one and do not compare against another GPU path.

**M-D quality proven (7).** Short context only (§7i): 2048 positions against
`jyohukuchan/qwen38-flash-next-bf16-logits`. Report KL, ΔPPL, router top-k agreement, per-layer
relative-RMS. Cosine is a **localiser only** (scale-invariant, saturates, 0.999+ on degraded models).
Tolerance in the output format's precision, never a blanket 1e-6.

**M-E performance (8).** Numbers measured under identical conditions, with the two traps recorded:
`-Doptimize=ReleaseSafe` does **not** build the test artifact (stale binary), and piping `zig build test`
through awk gives a single timestamp. Targets: ≥150 tok/s decode at 262,144 ctx, ≥5000 tok/s prefill.

## Measurement discipline (this was my miss — user, 2026-11-25)

The timeout investigation was mine to solve and it consumed the user's attention it should not have.
What went wrong: I reported a 13.64 s figure that was **compile time**, then a ">150 s hang" I had not
isolated, and only found the real causes — orphaned `zig`/`test` processes spawned by my own aborted tool
calls, and then `std.testing.allocator` in a bulk path — after the user intervened three times
("do one at a time", "we have leftover process", "don't re-run gates").

**A checklist did not prevent this and would not have: both traps were already written in my notes when
I walked into them.** So the control is mechanical — `tools/measure.sh` refuses to produce a number on a
busy box, always runs twice, and rejects the pair when they disagree by more than 25 %. Any timing number
I report must come from its `quotable` line; hand-timed figures are not reportable. It caught a live busy
box on its first day (the running lane's `zig`/`build` processes).

Standing checklist before any timing claim or slow-path diagnosis:

1. **Prove the box is idle.** `pgrep -x zig; pgrep -x test; pgrep -x cargo` and GPU memory. My own aborted
   calls leave processes running; a timing taken with them alive is void.
2. **Separate compile from run.** Run twice; the second run is the measurement. `zig build` time is not
   test time, and `-Doptimize=ReleaseSafe` does not rebuild the test artifact at all.
3. **Instrument phases before restructuring anything.** The manifest was 0.57 s the whole time; the
   "optimisation" I nearly started would have fixed nothing.
4. **Bisect with one variable.** Same build, one allocator swapped — not a shard redesign.
5. **Never route debugging to the user.** They get decisions and authorisations, not "it's slow, help".
   If I catch myself asking for a debugging opinion, the correct move is another measurement.

## Standing constraints

- Guard bands **on** by default: iterate with `INFERNO_GUARD_BANDS=0`, verify without.
- `INFERNO_BENCH=1` for measurement sweeps; the default suite never touches 132 GB.
- Checkpoint only via `INFERNO_MODEL_DIR`; 52 GB load only via `INFERNO_SHARD_LOAD=1`.
- Disk ~82 GB free (92 % used): no large files, never copy the checkpoint.
- Ask-before-expensive (§7h) is suspended overnight for **critical-path** work only; anything
  irreversible, any download >2 GB, any >15 min single action outside the critical path waits for the user.
- A test lives in the layer that owns the behaviour (`.agents/design/target-state.md`). If a test is
  awkward in Rust because it needs a Zig internal, that is evidence the behaviour has not moved.
