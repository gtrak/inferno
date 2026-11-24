# I-028 — make the buffer/stream bugs structurally hard (and cheaply detectable)

Four bugs tonight, in four different modules, all of the same two shapes:

| bug | module | shape |
|---|---|---|
| 1.4 MB written past a 64 KB alloc, clobbering `rope_cos`, layer-2 KV, MoE scratch | `model/state.zig` | **two sizing formulas drifted** (`State.indexScoreBytes` vs `Attention.indexerScoreBytes`, 256 cols vs a 32768 stride) |
| `routedExpert` wrote at a `tokens_max` stride while `moe_combine` reads `[experts][tokens][hidden]` | `model/exec.zig` | **two layout descriptions drifted** |
| `applyRope` read executor-stream casts with no fence | `model/exec.zig` + `hw/` | **unordered cross-stream dependency** |
| `reset` zeroed the mRoPE tables, nothing refilled them | `model/state.zig` | **no owner of an invariant** |

None is a kernel defect. Every kernel passes its own fp32 oracle. All four live in the seams, and
all four were found by an end-to-end test, not by review — the `routedExpert` one was found only
by bisecting against trace tensors.

The goal is **not** "impossible" — nothing in Zig or Rust prevents a kernel writing past a device
buffer, or a launch on the wrong stream. The goal is: the wrong thing needs a deliberate bypass,
and when it happens anyway it is detected on the first run rather than the fourteenth.

## 1. `core/src/hw/buffer.zig` — one owner, one size, verifiable guards

```zig
pub const Kind = enum { index_scores, kv_cache, moe_scratch, rope_tables, /* ... */ };

pub const DeviceBuffer = struct {
    bytes: u64,            // private: not settable from outside
    device: [*]u8,         // private
    kind: Kind,
    last_writer: Stream,   // who may read this without a fence
    guard: ?Guard,         // debug: canary band after the payload
};

pub fn alloc(.{ .kind, .size_fn, .stream }) !DeviceBuffer;
pub fn verifyGuardsAll() !void;   // call after each layer in debug builds
```

- `bytes` is computed **only** by `size_fn`, which must live in the module that owns the buffer
  (`Attention.indexerScoreBytes` for `index_scores`). `State` then calls the allocator and cannot
  invent a number. Bug 1 becomes unexpressible.
- **Guard bands**: allocate payload + 2 MiB of a canary pattern; `verifyGuardsAll()` scans it.
  This would have caught the 1.4 MB overrun **on the first run, deterministically**, instead of
  surfacing as three different `max_abs_err` values. `compute-sanitizer memcheck` reported 0
  errors on the same code because the write stayed inside a larger allocation — a guard band
  does not have that blind spot.
- Debug-only by default (an env var), zero cost in release.

## 2. Comptime layout as the single source of truth, shared with the `.cu`

```zig
pub const Layout = struct { dtype: DType, shape: [4]u32, /* ... */ };
pub fn bytes(comptime l: Layout) u64;
pub fn offset(comptime l: Layout, idx: [4]u32) u32;
```

`build.zig` already runs a generator (`generated_cubins.zig`); it emits
`generated_layout.h` from the same comptime definitions, and every kernel `.cu` includes it.
Then a host-side stride and a device-side stride cannot disagree, because there is one
definition of each tensor. Bug 2 becomes unexpressible.

This is cheap because the mechanism already exists — `nvcc` is already driven from `build.zig`
with a generated source file.

## 3. Stream ownership: the fence becomes automatic

`launch(buf_in, buf_out, ...)` records `last_writer` per buffer and inserts the event when a
kernel on a different stream consumes it. Writing the missing fence then requires calling a raw
launcher, which invariant G1 below forbids outside `hw/`. Bug 3 becomes a visible bypass.

## 4. The two gate invariants (`tools/gate.sh`)

**G1 — no raw device access outside `hw/`.** Extend the existing "CUDA mass confined to
`hw/`+`bench/`" check: `core/src/**` may not `@import("hw/cuda_driver.zig")` or
`hw/cuda_context.zig` outside `hw/` or `bench/`, and may not name `cuMemAlloc*`,
`cuLaunchKernel`, `cuMemcpy*`. Tonight `model/multi.zig` imports both driver and context directly
and the gate passes, because the check looks for `cu[A-Z]` symbol *names* and the wrapper hides
them. The abstraction exists in name only.

**G2 — no two GPU test binaries concurrent in one gate run.** `build.zig:243-245` makes `test`
depend on both `run_core_tests` and `run_probe_tests`, so two binaries bring up CUDA contexts on
device 0 simultaneously, and a reviewer measured **2 of 5 runs failing at different GPU tests**.
Serialise them (or exclude GPU tests from the probe module). This one is not hygiene: it is a
measured flake, and it may be contributing to the residual I-013 nondeterminism — a latent race
surfaces more readily when another process is perturbing the device.

## Priority and new evidence (user directive 2026-11-24: reliability before new implementation)

**This issue is now top priority, ahead of I-014b/I-015/I-016.** Measured evidence over three
consecutive `zig build test` runs on `main` after serialising the two GPU test binaries:

```
run1 EXIT=0   max_abs_err 7.652e-3
run2 EXIT=0   max_abs_err 7.652e-3
run3 EXIT=1   max_abs_err 1.216e-4   <- and a DIFFERENT test failed
   error: 'kernels.attention.test.attn_decode is flat across the context sweep' failed:
          config: cannot parse: MissingField
```

Three facts, each a separate defect:

**R1 — two error regimes, not drift.** `7.652e-3` is the usual outcome; `1.216e-4` (≈0.1 ulp,
essentially correct) occurred once and *passed*. Two stable outcomes rather than a continuum is
the signature of **uninitialised device memory** — `cudaMalloc` does not zero, so a buffer holds
either the previous process's contents or zeros. Guard bands + explicit zero-fill (item 1) are the
direct fix, and a canary would identify the exact buffer. **This is probably the same root cause
as the I-013 residual nondeterminism.**

**R2 — tests race within one binary.** Serialising `run_core_tests` before `run_probe_tests` fixed
the *cross-binary* concurrency (that is done, `core/build.zig`) and did **not** fix this: the
failure is one test corrupting another's fixture inside the core binary, which runs 16-wide. Every
test that opens a fixture by a path, or mutates a shared config/temp file, is a suspect. Fix by
making tests hermetic: no shared mutable files, no cwd-relative fixture writes, no reuse of a
process-global device buffer across tests.

**R3 — `config: cannot parse: MissingField` is doing double duty.** The same string appears on
green runs (a test prints it while exercising an error path) and on red ones (the attention test
genuinely failing). Any grepping of that string is meaningless; only exit codes count.

## Non-goals

- Do not attempt to prevent device-side OOB by type checks. Guard bands + a debug verify are the
  honest mechanism.
- Do not add `unsafe`-style escape-hatch ceremony. The abstraction must be **easier than the
  workaround**, or lanes will route around it — three lanes have already asked to edit
  `build.zig` to dodge a fence.
- Do not change any kernel's numerics or any tolerance.

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build test) ; echo CORETEST=$?
(cd core && timeout 900 zig build refcheck) ; echo REFCHECK=$?
```

Each new invariant must be **proven non-vacuous**: introduce a violation, paste the gate's refusal,
then revert. G2 must be shown by running the gate with the serialisation removed and observing a
failure.
