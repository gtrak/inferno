# I-038 — the weight-loading pipeline moves to Rust

**Depends on I-036.** Can run in parallel with I-037 once the handle ABI is stable.

## Why

Loading is bulk, once, and dominated by `mmap` lifetimes and slice indexing into mappings — the same
self-referential hazard `Model.init` just hit (a dangling interior pointer caught only at runtime).
Rust handles it structurally by owning the mapping and indexing by offset.

## Scope — the pipeline only, two deliberate carve-outs

Moves to Rust:

- Mapping lifetimes: open, hold, and drop the 11 safetensors shards; nothing may outlive its mapping.
- Load ordering and pacing: which tensors go first, how H2D streaming is paced against the 6.7 GiB/s
  per-GPU PCIe measurement (cold 4-stream 4.56–6.11 GiB/s).
- Per-rank capacity negotiation against **measured** free VRAM, and the resulting VRAM/host split.
- Read-back verification of loaded extents (already implemented in Zig by I-017a; it moves with the
  pipeline because it is part of "prove the bytes landed").

**Stays in Zig** (both are bug-class guards, not language preference):

1. **`mem/placement.zig`'s layout algebra.** `kernels/moe.zig` and `model/multi.zig` import it, so its
   sizing formulas are part of the kernels' contract. Bug class #1 (1.4 MB written past a 64 KB buffer)
   was two sizing formulas drifting between modules; moving this across a language boundary creates the
   worst version of that bug. Expose the placement **decision as data** over the ABI instead.
2. **The safetensors byte reader** — the single parser of the checkpoint's bytes, with its 9 tests.

   The first draft of this spec justified the carve-out with "the Zig test tree depends on it", which is
   the weak reason. The real ones:

   - It is the **only** safetensors parser in the repo (nothing on the Rust side parses it today), so
     keeping it here keeps one parser rather than two. Two parsers with two validation suites is bug
     class #1 again, in the worst place: the decoder of a 132 GB artifact.
   - **8 of its 9 tests are refusal tests** — overlapping offsets, trailing bytes, a header length past
     EOF, a shape contradicting its byte length, an unsupported dtype, a rank-six shape, a shard still
     downloading keeping a complete header. Those are precisely the modes that corrupt a load
     *silently*, which is why they are load-bearing rather than incidental.
   - It is not fixture-only: `kernels/ple.zig` mmaps and reads the real 47.7 GiB n-gram table, and
     `weights/quant.zig` validates quantisation goldens through the same header machinery. Both are on
     the production path.
   - Zig core must stay testable standalone with no Rust dependency; a Zig-side reader is what lets the
     tiny path (`model/load.zig`) run the suite without linking the server.

   **Boundary shape:** Zig parses and answers "here are the tensors and their offsets"; Rust owns the
   mapping lifetime, ordering, pacing, capacity negotiation and read-back verification, and never
   re-parses bytes. Coarse: one call per shard (11 of them), not per tensor.

   **If a future issue does put bytes in Rust's hands**, the reader's validation must move with it as
   *shared conformance vectors* — one set of malformed headers that both implementations must reject —
   rather than as a second hand-written test suite.

## Hard constraints

- The checkpoint path comes from `INFERNO_MODEL_DIR` only; no default path.
- A default suite run must never touch 132 GB. The 52 GB load stays behind `INFERNO_SHARD_LOAD=1`.
- Bulk paths take an arena: `std.testing.allocator` on a bulk path is G3-blocked and >1250× slower
  (measured 478 ms vs >600 s unfinished).
- Measured reality this must reproduce: 65,667,551,616 B in 14.955 s = 4.39 GB/s across 4 ranks,
  43,738 extents/rank, 61.1 GiB into VRAM, 47.7 GiB PLE table host-resident, **413,664 B headroom**.

## Alignment instrument

- Suite green; `weights.shard` and `weights.manifest` tests keep their assertions — these are the
  specification of what the pipeline guarantees (completeness, role coverage, extent verification,
  the documented geometry of the real checkpoint).
- The Rust pipeline must reproduce the same per-rank byte totals and the same witnesses on the same
  checkpoint; a difference is a finding, not a tolerance.
- New tests must fail if reverted: incomplete shard refused, extent mismatch caught, capacity
  overcommit refused before upload rather than at `cuMemAlloc`.

## Acceptance

Rust owns the pipeline; Zig keeps placement algebra and the byte reader, exposing both over the ABI.
Same byte totals and witnesses as I-017a on the real checkpoint; suite green.
