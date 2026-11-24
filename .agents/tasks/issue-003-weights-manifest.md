# I-03 — weight layer: safetensors + config + tensor manifest + quant formats

## Working agreement

- Sole writer in your worktree. Read `plan.md` §3 (exact byte inventory),
  `THEORY.md`, `.agents/skills/tiger-style/` (authoritative for all Zig).
- Builds on I-01 (driver layer, build system, gate script).
- The real checkpoint is at `/home/gary/models/qwen38-flash-next-nvfp4`
  (**read-only, never copy it into the repo, never commit it**). It is ~132 GB;
  if a shard is still downloading, gate on the shards present and say so.
- Reference for the architecture: `tools/ref-trace/reference/modeling_qwen4_exp.py`
  (vendored by lane B; if absent, `/tmp/ref/modeling_qwen4_exp.py`) and
  `tools/ref-trace/reference/configuration_qwen4_exp.py`.

## What to build — all in Zig, no Rust

### 1. `core/src/weights/safetensors.zig`

- Parse the safetensors header: 8-byte little-endian `u64` length, then a JSON
  object of `name -> {dtype, shape, data_offsets}` plus optional `__metadata__`.
- Do not copy tensor bytes: `mmap` the file (`std.posix.mmap`, `MAP_PRIVATE`),
  keep `{ file_index, base_offset, byte_len }` per tensor.
- Support dtypes actually present: `BF16`, `F8_E4M3`, `F32`, `U8`, `I64`.
  Unknown dtypes are a hard error.
- Alignment/sanity: reject headers whose offsets overlap, exceed the file, or
  whose element count does not match `shape` and `dtype`.

### 2. `core/src/weights/config.zig`

- Parse the real `config.json` (`std.json`) into an explicit Zig struct covering
  the *text* model: `hidden_size 2560`, `num_hidden_layers 48`, `num_attention_heads 24`,
  `num_key_value_heads 2`, `head_dim 256`, `linear_*`, `num_experts 512`,
  `num_experts_per_tok 10`, `moe_intermediate_size 640`,
  `shared_expert_intermediate_size 640`, `hc_count 4`, `hc_lowrank 320`,
  `ple_layer_ids [2]`, `ple_embed_dim 2560`, `ngram_*`, `indexer_*`,
  `layer_types` (48 entries), `rope_parameters` (`partial_rotary_factor 0.25`,
  `mrope_section [11,11,10]`), `vocab_size 248320`, `rms_norm_eps 1e-6`,
  `eos_token_id`, `seed`. Keep the vision and `mtp` sub-configs as
  recognized-but-unused for now.
- Derive and validate the needed invariants **with assertions**: e.g.
  `head_dim * partial_rotary_factor / 2 == sum(mrope_section)`,
  `num_attention_heads % num_key_value_heads == 0`,
  `linear_num_value_heads % linear_num_key_heads == 0`,
  `hc_count * hidden_size == linear_num_key_heads*key_head_dim*2 + linear_num_value_heads*value_head_dim`
  (both are 10240 — assert it, it is a real invariant of this checkpoint),
  `layer_types` length == `num_hidden_layers`.
- **PLE layer index**: `ple_layer_ids` are 1-based layer numbers, so `[2]` means
  text layer index **1**. Assert the checkpoint agrees (that layer has `.ple.*` tensors).

### 3. `core/src/weights/manifest.zig`

Build a typed tensor table from config + the 11 safetensors files, classifying
every tensor into a closed enum of roles and validating shape/dtype per role:

- `embed_tokens`, `lm_head` (BF16 [248320, 2560])
- per layer: `attn_hyper_connection/*`, `mlp_hyper_connection/*` (BF16, 10240-wide),
  `mlp.gate` (BF16 [512,2560]), `mlp.shared_expert.{gate,up,down}_proj`,
  `mlp.shared_expert_gate`, and for each of 512 experts
  `mlp.experts.E.{gate,up,down}_proj.{weight (U8), weight_scale (F8_E4M3), weight_scale_2 (F32), input_scale (F32)}`
- `linear_attn/*` (36 layers): `in_proj_qkv [10240,2560]`, `in_proj_z [6144,2560]`,
  `out_proj [2560,6144]`, `in_proj_a/b [48,2560]`, `conv1d [10240,1,4]`,
  `A_log [48]`, `dt_bias [48]`, `norm [128]`
- `self_attn/*` (12 layers): `q_proj [12288,2560]`, `k_proj`/`v_proj [512,2560]`,
  `o_proj [2560,6144]`, `q_norm`/`k_norm [256]`, `indexer.index_qk_proj [640,2560]`,
  `indexer.q_layernorm`/`k_layernorm [128]`
- PLE (layer 1): `ple.key_proj [10240,2560]`, `ple.value_proj [2560,2560]`,
  `ple.conv1d`, `ple.norm_{key,query,conv}`, `ple.ple_embedding.ngram_embedding.shard_{0..127}.weight`
  (F8_E4M3, each shard a contiguous row range of the logical `[320M, 160]` table),
  plus the `ngram_heads_{offsets,vocab_sizes}`, `layer_multipliers`,
  `ngram_embedding.weight_scale` buffers
- `hyper_connection_mixer/*` (final, `use_combine=false`)
- `mtp.*` and `model.visual.*` — classified as `deferred` (present, unused, but
  reported in the manifest). Note `mtp` uses FP8 with 128×128 `weight_scale_inv`.

Emit `manifest.json` (via a `zig build manifest` step writing to
`core/zig-out/manifest.json`) with: per-role tensor count, per-role byte total,
per-file byte total, and the derived model dimensions. Assert
`sum(all tensors) == actual file bytes` for every file (headers included is fine
if you account for it explicitly).

### 4. `core/src/weights/quant.zig`

Decoders used later by kernels, validated here against the oracle:

- `fp4_e2m1_decode(u8 nibble) -> f32` — full 16-entry table, `-0.0` handled, and
  a `fp4x2_decode(u8 byte, *f32 low, *f32 high)` helper (low nibble = first element).
- `dequant_nvfp4_row(packed: []const u8, scales: []const f8_e4m3, scale_2: f32, out: []f32)`:
  `w = fp4 * f32(scale[block]) * scale_2`, **block = 16 consecutive weights along
  the contracted dimension**.
- `dequant_fp8_blockwise(...)` for the 128×128 `weight_scale_inv` layout (mtp only).
- `bf16_to_f32`, `f32_to_bf16` with correct round-to-nearest-even.

**Oracle test (mandatory).** Extend `tools/ref-trace` **only if** the golden
harness cannot express this; preferred: write a small Python snippet in
`tools/ref-trace/quant_oracle.py` that
1. takes a fixed 16-value pattern, quantizes it to NVFP4 the way modelopt does
   (`scale = amax/6`, `w/scale` rounded to the e2m1 grid, `scale_2 = global amax/ (6*448)`
   per NVIDIA's NVFP4 definition),
2. writes `{packed, scales, scale_2}` to `tests/golden/quant/nvfp4_case.safetensors`,
3. and a Zig unit test that dequantizes the same case and compares to the
   Python-produced float reference **bit-exactly** (fp32 values are produced by
   the same table lookups, so exact equality is achievable — do not widen the
   tolerance without documenting why).

If modelopt's exact rounding turns out not to matter for a *dequantization* test,
produce the reference by dequantizing with an independent implementation
(numpy table lookup) rather than by re-running our Zig code.

## Gates

**Rule: no gate expectation may be a hand-written numeric window.** Every expected
byte count is *derived from config* and asserted exactly, and the whole manifest
must reconcile against the checkpoint's own index metadata. (Two earlier specs in
this project asserted impossible windows — an H2D floor above the PCIe ceiling in
I-002, and a `kv_proj` range that described `k_proj` alone here. Magic ranges are
the bug; derived equalities are the fix.)

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && zig build manifest && python3 - <<'EOF'
import json
m = json.load(open("zig-out/manifest.json"))
r, d = m["roles"], m["dims"]

# --- headline invariant: the manifest accounts for every byte the index claims ---
assert m["total_bytes"] == m["index_total_size"], (m["total_bytes"], m["index_total_size"])

# --- per-role counts are structural, not empirical ---
H, L = d["hidden_size"], d["num_hidden_layers"]
E, I = d["num_experts"], d["moe_intermediate_size"]
n_full = d["layer_types"].count("indexed_attention")
n_lin = L - n_full
assert n_full == 12 and n_lin == 36, (n_full, n_lin)
assert r["expert"]["count"] == L * E * 3 * 4           # weight + 3 scalars per proj
assert r["k_proj"]["count"] == n_full
assert r["v_proj"]["count"] == n_full
assert r["ngram_table"]["count"] == 128

# --- exact byte counts, derived from shapes and dtypes ---
BF16 = 2
assert r["k_proj"]["bytes"] == n_full * 2 * 256 * H * BF16       # 2 kv heads, head_dim 256
assert r["v_proj"]["bytes"] == r["k_proj"]["bytes"]
assert r["q_proj"]["bytes"] == n_full * 2 * 24 * 256 * H * BF16  # query AND gate fused
assert r["o_proj"]["bytes"] == n_full * H * 24 * 256 * BF16
expert_proj = I * H // 2 + I * H // 16 + 8                       # fp4 pairs + e4m3/block + 2 f32
assert r["expert"]["bytes"] == L * E * 3 * expert_proj
assert r["kv_proj_total"]["bytes"] == r["k_proj"]["bytes"] + r["v_proj"]["bytes"]

# --- cross-check against THEORY.md (report deltas, never widen a tolerance) ---
for role, theory_gb in (("expert", 67.95), ("ngram_table", 51.20),
                        ("linear_attn", 4.16), ("self_attn", 1.23)):
    got = r[role]["bytes"] / 1e9
    assert abs(got - theory_gb) / theory_gb < 0.01, (role, got, theory_gb)
print("manifest OK", m["total_bytes"], "bytes reconciled")
EOF
) ; echo MANIFEST_EXIT=$?
(cd core && zig build test) ; echo TEST_EXIT=$?
```

All exits must be 0. If the whole-checkpoint reconciliation passes but a role
derivation fails, the *role taxonomy* is wrong (split or merge the role); never
relax the derivation.

### Role taxonomy decision (settled)

`k_proj` and `v_proj` are **separate roles**, each 12 tensors of
`2 kv_heads x 256 x hidden_size` BF16 (31,457,280 B each), because placement and
the per-tensor requantiser treat them independently. A derived aggregate role
`kv_proj_total` exists only for reporting. `q_proj` is its own role (query+gate
fused, `2*24*256` out-features). Do not fold k/v into one role: an aggregate that
hides two differently-sized tensors is what produced the bad window.

### Partially-downloaded files

Handle an in-flight shard by parsing the header (complete as soon as the file is
opened for append), reporting `"data_present": false` for that file, and still
counting its bytes. Only the mmap-touch check may be skipped for such a file.

## Report format

Implemented / Changed files / Validation / Open risks (anything in the manifest
that contradicts `THEORY.md` or the reference implementation) / Recommended next
step. List every tensor role you had to guess, if any.
