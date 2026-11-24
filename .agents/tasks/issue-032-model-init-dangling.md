# I-032 — `exec.Model.init` returns a struct whose interior pointers die on return

**Blocking, latent, and it invalidates the assumption behind every green e2e run so far.**

`core/src/model/exec.zig`'s `Model.init` returns `Model` **by value** while storing pointers into its
own callee-local `model`:

```zig
state.stream_handle = &model.ple.stream
ple_block.ple       = &model.ple
ple_block.table     = &model.table
pool.table          = &model.table
```

After the return, those addresses refer to a stack frame the caller does not own. Verified with a
two-model probe:

```zig
var a = try exec.Model.init(...);
var b = try exec.Model.init(...);
assert(a.state.stream_handle == &a.ple.stream);  // FAILS
assert(a.ple_block.ple == &a.ple);               // FAILS
```

With two models live, `a.reset(); a.forward(prompt)` writes zeros to `a`'s logits and a-vs-b prefill
of the same prompt differ at the first logit.

**Why it has gone unnoticed:** a single long-lived model never has its stack slot reused, so the
stale pointers happen to still resolve. `tiny_check` passes by luck. Any refactor, a second model,
or a different optimisation mode can break it, and the failure would look like a numerics bug.

## Fix

Make storage the caller's, so interior pointers are established at their final address:

```zig
pub fn init(self: *Model, allocator: Allocator, config: Config, ...) !void
```

then build the self-referential pointers from `self`. Every call site changes — `tiny_check`, the
runtime, and `multi.zig` — and that is the point: the correct form should be the only form.

Do **not** fix this with a per-caller `bind()`: that leaves every other caller relying on luck, and
leaves the next one to rediscover it.

## Test that must exist and must be able to fail

Two live models, both bound correctly:

- `a.state.stream_handle == &a.ple.stream`, `a.ple_block.ple == &a.ple`, `a.pool.table == &a.table`,
  and the same for `b` — asserted individually so a partial fix cannot pass;
- prefill the same prompt through `a` and `b` and require identical logits;
- `a.reset(); a.forward()` after `b` exists must equal the same run in a fresh process.

## Why Rust would have prevented this, and what we do instead

This is the one bug class in this project where the borrow checker earns its keep. A self-referential
struct cannot be *constructed* in safe Rust — there is no lifetime meaning "inside myself" — so the
workarounds all require an explicit act (`Box::pin`, `self_cell`, `Rc<RefCell<_>>`, or indices instead
of pointers). The bug could not have drifted in. For the device-memory bugs (the 1.4 MB overrun, the
stride mismatch, the missing fence) Rust helps essentially not at all, because CUDA buffers and stream
ordering sit outside its model. This one is entirely host-side, in safe code, in its domain.

Zig's answer is discipline plus a guard. So, in addition to the out-parameter `init`:

**A self-containment assertion, debug-only, at the end of `init`:**

```zig
// Every interior pointer must lie inside the Model that now owns them.
assert(inRange(self, &self.ple.stream));
assert(inRange(self, &self.ple));
assert(inRange(self, &self.table));
```

where `inRange` is `@intFromPtr(ptr) >= @intFromPtr(base) and @intFromPtr(ptr) < @intFromPtr(base) + @sizeOf(Model)`.
This is the §7d rule 2 case: a bounded runtime guard that fires on the **first** run and names the
field, instead of a probe written after the fact. It would have caught the by-value return the first
time any caller built a second model, and it costs nothing in release.

Add it to every self-referential struct in the tree, not just `Model` — `State`, `Block`, `Pool` and
anything else that stores a pointer to a sibling. Grep for `= &self.` and `= &model.` to find them.

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build test) ; echo CORETEST=$?
```

Non-vacuity: revert to a by-value `init` in a scratch copy and show the assertions fail — the probe
above already does this, so the test is known to be able to see the defect.

## Related

A *separate* bug in the same area (`ple.Block.reset`'s chunked copy never advancing its destination)
was the proven cause of "fourth reset+forward diverges at ulp 88" and is fixed. These are two
distinct defects and neither explains the other.
