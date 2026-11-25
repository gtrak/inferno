# Where the Rust/Zig FFI boundary should sit

Written 2026-11-24, after five seam bugs and one self-referential-struct bug, all in the same layer.

## The refined principle (user, 2026-11-24)

> Zig is about **composing larger unsafe subsections and removing allocation from the hot path**. It
> is not about the perf ceiling - maximum perf is simply reached sooner, unless we chase our tails for
> things a type system could prove.

This is the right frame and it supersedes a looser version of this document. Zig's value here is
**mechanism inside a big unsafe region with no hot-path allocation**; Rust's is **ownership policy
across calls**. So the boundary is drawn at a *coarse unit of work* - per token, per sequence - not
per launch. Rust hands Zig an arena and says "run N tokens"; Zig runs 48 layers x ~10 launches inside
it without crossing the line, with direct pointer arithmetic and no allocator.

### Which bugs were actually type-system-catchable

Seven bugs on 2026-11-24, and the honest accounting matters more than the principle:

| bug | shape | Rust would catch it | caught instead by |
|---|---|---|---|
| `indexScoreBytes` drift (256 vs 32768) | two sizing formulas | **no** | **comptime `Layout`** |
| `routedExpert` stride mismatch | writer/reader disagree on layout | **no** | **comptime `Layout`** |
| missing `applyRope` fence | cross-stream ordering | **no** | launch wrapper |
| mRoPE tables zeroed on reset | lifecycle | partial | `epochs` + `fillVoid` |
| `conv_state` chunk offset | arithmetic | **no** | guard band |
| `Model.init` dangling interior pointer | ownership across a move | **yes** | borrowck |
| `epochs` registry aliases across devices | global registry keyed by pointer | **yes**, designable | context-scoped handles |

**2 of 7.** An earlier version of this document claimed "every bug that was not a kernel was in the
region Rust's borrow checker covers". That was wrong. Four of the seven are arithmetic, layout or
ordering, which **no type system catches**.

**The type system that would have helped most tonight was comptime, not borrowck.** A single `Layout`
definition generating both the host byte size and a header the `.cu` includes makes bugs 1 and 2
unrepresentable - and that is Zig's strength. So the split is not "prove in Rust, do in Zig", it is:

| tool | catches |
|---|---|
| **comptime** | layout and sizing disagreement (the two most common failures) |
| **borrowck (Rust)** | ownership across calls: dangling interiors, use-after-move, lifecycle |
| **runtime guards (Zig)** | arithmetic, ordering, write-before-read - bounded cost, first-run failure |

## What the boundary actually is (refined 2026-11-25, user)

The boundary is **by role, not by language**:

- **Zig = inference.** Kernels, the forward pass, and the layout algebra those kernels agree with.
  It executes; it does not decide when, what, or from where.
- **Rust = integration.** Complex orchestration and ownership: which model is loaded and for how long,
  where each expert's bytes came from, sequence lifetime, scheduling, tokenisation, stop conditions,
  transport. The borrow checker earns its keep here because this is where the ownership lives.

The three items below were always this shape, but they were hard to see because **the Rust crate root is
called `server`** — so "move the load pipeline to Rust" read as "put loading in the HTTP layer", which
sounds absurd and invites arguments no one should have had. `server/src/lib.rs` already holds
`admission`, `engine`, `tokenizer`, `stop`, `template`; only `api/`, `server.rs` and `main.rs` are
transport. The name was misdescribing the design and steering it.

**Therefore (I-039):** split the crate before moving behaviour into it.

| crate | role | contents |
|---|---|---|
| `host/` (`inferno-host`) | integration + ownership | lifecycle, load pipeline, sequence/decode control, scheduler, tokenizer, stop, template, telemetry |
| `server/` (`inferno-server`) | transport only | `api/`, `server.rs`, `main.rs`; depends on `inferno-host` |
| `core/` (Zig) | inference | kernels, forward, placement algebra exposed **as data**, C ABI |

A consequence worth naming: **loading is Rust's**, including reading the checkpoint's headers. Zig keeps
a minimal safetensors reader for the *tiny fixture* and the quantisation goldens — a different artifact
from the 132 GB checkpoint, so it is a split by artifact, not a duplicated parser in the same job. The
production parser's validation suite (its refusal tests) moves with it to Rust.

## The principle

**Put the boundary where data is bulk or batched and latency-irrelevant; keep it coarse; let exactly
one side own device memory.** The borrow checker's value is proportional to host-side ownership and
aliasing complexity — so move the code that has that complexity and keep the code that is thin,
per-launch, or numerically authoritative.

## The chattiness objection is dead — measured

| | value |
|---|---|
| kernel launches per token (48 layers × ~10) | ~480 |
| `cuLaunchKernel` enqueue | ~3.5 µs → **~1.68 ms/token already** |
| an extra Rust→Zig C ABI hop | ~20 ns → **10 ns per token total, 0.0002 %** |

So "an ABI between Rust and the launch path would be too chatty" is wrong by four orders of
magnitude. The decision is about **ownership**, not call count.

## Ranked by leverage

### 1. The sequence runtime / decode loop → Rust (highest ratio, lowest risk)

`model/generate.zig` plus the per-token control flow. One FFI call **per token**, not per launch, so
the cost is nil. It is pure control flow plus ownership of KV and recurrent state — exactly the shape
the borrow checker is good at — and it collapses "request → tokens → SSE" into one language instead
of splitting it across the server and the core.

This is the cheapest real win available and it is already in flight as I-015.

### 2. Buffer lifetime policy → Rust; launch sequencing stays in Zig (revised)

**Revised per the refinement above.** Moving the launch sequence into Rust is the wrong move: it would
put every kernel call behind an `unsafe` block and force borrow gymnastics over device handles that
are not Rust's to reason about - fighting the language in exactly the place it is weakest and Zig is
fastest. The launch sequence is a **large unsafe region with no allocation**, which is Zig's purpose.

What *should* move is the **lifetime policy around it**: which arena is live, when a sequence may
reset, when buffers die, what a handle's validity scope is. Those are the two type-system-catchable
bugs above (`Model.init`'s dangling interior, the cross-context registry aliasing), and both are
ownership-across-calls, which is precisely Rust's domain. Rust owns handles with lifetimes; Zig gets
raw pointers into an arena it may not allocate from or outlive.

This is the big one, and it is where **all five seam bugs of 2026-11-24 lived**:

| bug | shape |
|---|---|
| 1.4 MB past a 64 KB alloc | two sizing formulas drifted between modules |
| `routedExpert` stride | writer and reader disagreed on layout |
| missing `applyRope` fence | unordered cross-stream dependency |
| mRoPE tables zeroed | container cleared, meaning not restored |
| `conv_state` reset | chunked copy never advanced its destination |

`hw/epochs.zig` currently checks write-before-read **at runtime**, with an epoch registry and a
history-erasing `fillVoid`. Move buffer ownership and the launch sequence into Rust and that whole
registry becomes **compile-time lifetime tracking** — the guard is no longer a runtime check that can
be compiled out of release builds, which is the one caveat on the current design.

Cost is real but bounded: a fatter ABI, and Rust owning the device buffers means Zig becomes a thin
"execute this kernel with these pointers" layer. The measured 0.0002 % per-hop cost says the chattiness
is affordable.

**Hard rule if this is done:** exactly one side owns device memory. Rust owns it; Zig receives opaque
handles and never allocates or frees. Two owners is double-free and use-after-free at a language
boundary, which is worse than either language's own failure modes.

### 3. The weight-loading pipeline → Rust (bulk, once, lifetime-heavy)

`weights/manifest.zig` (indexing), the sharding/placement arithmetic, and the quantisation tensor
descriptors. All host-side, all run once at load, all dominated by `mmap` lifetimes and slice indexing
into mappings — the same self-referential hazard `Model` just hit, and one Rust handles structurally
by owning the mapping and indexing by offset.

**Caveat that decides the shape:** keep a minimal safetensors reader in Zig for the *test* tree, or the
Zig tests become dependent on Rust. Move the pipeline, not the byte reader.

### 4. Keep in Zig — do not move these

| keep | why |
|---|---|
| `hw/` | it is the unsafe seam **by design**; Rust would need `unsafe` on the same lines and gain nothing |
| `kernels/*.zig` launch wrappers | thin, per-launch, latency-adjacent; moving them adds a hop to the hot enqueue loop for no ownership benefit |
| `ref/` (7.4k) | it is the **authority** every kernel is judged against, already validated, and freshly optimised (I-027 `@Vector`, dot 6.4×). Moving it discards that and re-opens the oracle |
| `bench/` | measurement harness, no ownership complexity |

## What not to do

- **Do not split one token's work across many FFI calls.** Coarse boundary, owned state, one call per
  meaningful unit.
- **Do not let both sides own device memory**, even "temporarily during migration".
- **Do not move `ref/` to win a language argument.** The oracle's value is that it is unchanged and
  trusted.

## If a second engine is ever built

Start from: Rust owns everything that is not a kernel or the driver seam. Tonight's evidence is that
**every bug that was not a kernel was in the region Rust's borrow checker covers**, and the one
self-referential bug (`Model.init` returning a struct with pointers into its own dead frame) is
unrepresentable in safe Rust.
