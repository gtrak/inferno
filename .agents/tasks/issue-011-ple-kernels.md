# I-011 — PLE kernels: n-gram ids, 16-row gather, key/value proj, dilated conv

## Working agreement
- Sole writer. tiger-style binding. Reference lines 1055–1272 (`_splitmix64`,
  `_build_layer_multipliers`, `_find_nth_prime_after`, `NGramEmbedding`,
  `PLELayer`). Oracle: `core/src/ref/ple.zig`.
- Real: PLE is on **text layer index 1**; `ple_embed_dim=2560`, 16 heads
  (8 per n-gram order × orders 2,3), `head_dim_per_ngram=160`, table 320M rows
  × 160 F8_E4M3 in 128 shards; `ngram_size=3`, conv kernel 4 with dilation 3,
  `hc_count=4`, `hidden_size=2560`.
- **Do not edit `core/build.zig`.** Add exactly one line
  `_ = @import("kernels/ple.zig");` to the test block in `core/src/c_api.zig`.

## What to build
`core/src/kernels/ple.cu` + `.zig`:
1. **`ple_ngram_ids`** (host, in Zig) — splitmix64 multipliers, the nth-prime head
   vocab sizes/offsets, EOS-aware `_shift_right_ignore_eos`, int64 multiply/xor/
   remainder, add head offsets. Emits 16 ids per token. Must be **exact**.
2. **`ple_gather`** — pull 16 rows/token (160 B each) from the mmap'd 51.2 GB
   table. Batch the reads: contiguous per shard where possible, otherwise
   issue them concurrently (threads or io_uring) — a serial gather is a
   per-token stall. Stage into a pinned buffer, one H2D copy of
   16×160 B/token, then cast/scale on device.
3. **`ple_project`** — `key_proj` (2560→10240) and `value_proj` (2560→2560),
   plus the per-`hidden_size`-group RMSNorm on the key, and the signed-sqrt gate
   `sign(g)*sqrt(max(|g|,1e-6))` with `g = sum(key_normed*query_normed)/sqrt(2560)`.
4. **`ple_conv`** — dilated depthwise conv (`k=4, dilation=3`) over the
   10240-wide gated value, with the conv state ring buffer for decode, then
   `silu`, then add into the HC tensor.

## Gates
```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 900 zig build ple-test) ; echo PLE=$?
```
1. `ngram_ids` **exactly equal** to the oracle for the golden input and for a
   second input containing EOS mid-sequence (the EOS-segment logic is the trap).
2. `ple/out` matches the HF golden `layer1/ple/out` within `2e-5` scaled error.
3. Gather correctness: assert the bytes fetched for a known id equal the bytes at
   that row's file offset (read the same row twice by two routes).
4. **Non-vacuity**: print ids for 2 tokens (must differ), the gate value range
   (must contain a negative), and the conv output's mean/std.
5. Report gather latency per token (p50/p99) and the achieved DRAM/NVMe read rate.

## Report format
Implemented / Changed files / Validation (ids, error, latency) / Open risks /
Recommended next step.
