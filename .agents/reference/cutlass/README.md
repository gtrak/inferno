# Vendored read-only reference: CUTLASS SM120 MMA definitions

Provenance: NVIDIA CUTLASS, `main` branch, fetched 2026-11-24 (BSD-3-Clause, see
`LICENSE.txt`). These files are **documentation, never compiled by our build**:

- `mma_sm120.hpp` — `cute/arch/mma_sm120.hpp`: the exact PTX strings for every
  MMA instruction available on SM120, with operand registers and constraints.
- `mma_traits_sm120.hpp` — `cute/atom/mma_traits_sm120.hpp`: the thread-fragment
  layouts (`ALayout`, `BLayout`, `SFALayout`, `SFBLayout`) for those MMAs.

Why vendored: `sm_120a` block-scaled FP4 MMA has no public per-instruction
documentation we could rely on, and getting the operand syntax or the scale-factor
fragment layout wrong produces either a `ptxas` error or silently wrong numbers.
These two files are the ground truth; see `.agents/design/mma-sm120-nvfp4.md` for
our distilled conclusions and the verified SASS.

Fallback if the files ever go stale: the same content lives at
`https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/arch/mma_sm120.hpp`
and `.../include/cute/atom/mma_traits_sm120.hpp`.
