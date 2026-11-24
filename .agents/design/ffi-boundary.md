# Where the Rust/Zig FFI boundary should sit

Written 2026-11-24, after five seam bugs and one self-referential-struct bug, all in the same layer.

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

### 2. Buffer ownership and launch orchestration → Rust (highest absolute payoff)

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
