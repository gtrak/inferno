# I-010 — MoE kernels: router, permute/group, grouped NVFP4 GEMM, combine, shared expert

## Working agreement
- Sole writer in your worktree. tiger-style binding for Zig.
- Reference: `tools/ref-trace/reference/modeling_qwen4_exp.py` lines 918–1015
  (`MLP`, `Experts`, `TopKRouter`, `SparseMoeBlock`). Oracle: `core/src/ref/moe.zig`.
- Real shapes: hidden 2560, 512 experts, top-10, expert intermediate 640,
  shared expert intermediate 640, `norm_topk_prob=True`, `output_gate_type=sigmoid`.
- `.agents/design/mma-sm120-nvfp4.md` is binding for the FP4 MMA path.
- **Do not edit `core/build.zig`** (it globs `src/kernels/*.cu` already). In
  `core/src/c_api.zig`'s test block add exactly one line,
  `_ = @import("kernels/moe.zig");`, and change nothing else there.

## What to build
`core/src/kernels/moe.cu` + `core/src/kernels/moe.zig`:

1. **`moe_router`** — `logits = x[M,2560] @ gate[512,2560]^T` (BF16, exact), then
   **softmax in f32**, **top-10 on the probabilities** (not logits), then
   renormalise the 10 to sum to 1 in f32, then cast to the activation dtype — in
   that exact order. Top-k **tie-breaking must match `torch.topk`** (lowest index
   wins); validate against the oracle on synthetic ties, do not assume.
2. **`moe_permute`** — build the expert-sorted assignment list: for each of
   `M*10` (token, slot) pairs emit a row index; sort by expert id (stable, so
   ties keep token order); emit per-expert counts + offsets (offsets a multiple
   of 16 for MMA alignment, padded rows marked invalid and skipped in the
   epilogue).
3. **`moe_grouped_gemm`** — the core kernel: for each expert, a small GEMM over
   its gathered rows with the **native block-scaled NVFP4 path**
   (`OMMA.SF` m16n8k64, E4M3 per-16 scales from the checkpoint). Two stages:
   `gate_up: [rows,2560] x [1280,2560]^T -> [rows,1280]` (load gate_proj and
   up_proj as one 1280-row matrix, repacked once at load), then
   `silu(gate)*up`, then `down: [rows,640] x [2560,640]^T -> [rows,2560]`.
   Include the exact-BF16-dequant reference variant for the same shapes.
4. **`moe_combine`** — scale each expert output by its routing weight and
   scatter-add back to the token rows in f32, then cast once.
5. **`shared_expert`** — the same MLP with the shared weights, output scaled by
   `sigmoid(shared_expert_gate(x))` where that gate is `[1,2560]`.

## Gates
```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build moe-test) ; echo MOE=$?
```
(`moe-test` = the test target you add; it must run the oracle comparisons below.)
1. `moe_router`: `router_indices` **exactly equal** to the oracle for every
   position, including the synthetic tie cases; weights within 1e-6 relative.
2. Full block: `max_abs_err/max|ref| <= 2e-2` for the FP4 MMA path vs the f32
   oracle, `<= 5e-4` for the BF16-dequant path.
3. **Non-vacuity**: print the expert hit histogram (must have >1 non-zero expert
   and sum to M*10), the number of padded rows, and confirm two different inputs
   give different outputs.
4. Report throughput at the real decode shape (M=1 and M=32) and prefill shape
   (M=2048) with `total_experts=512, top_k=10`.

## Report format
Implemented / Changed files / Validation / Open risks (especially: tie-breaking,
padding correctness, and whether the grouped GEMM hit its MMA alignment) /
Recommended next step.
