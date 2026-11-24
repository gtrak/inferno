# The real-model reference (I-031, parts 1–2): generator, noise floor, feasibility, published traces

Four things live here:

* **`reference_trace.py`** — runs HF `transformers` over a checkpoint for a fixed prompt set and a
  fixed number of positions and writes a *logits* trace (prompt ids, logits per position, greedy
  continuation, full provenance). Mode `io-probe` answers "what would the real run cost" without
  loading a model or touching a GPU.
* **`verify_reference_trace.py`** — proves the generator reproduces the committed goldens, and proves
  the bound refuses a *known* KL (non-vacuity).
* **`published_traces.py`** — part 2's ingestion path for the four *published* trace datasets that
  replace a reference run this box cannot host: revision-pinned fetch, checksum verification against
  each publisher's manifest, a KL reproduction, and the router-weight comparison that links the
  routing traces to our checkpoint bit for bit. Writes only into `traces/` (gitignored).
* **this file** — the noise-floor record, the regeneration recipe, the measured feasibility note, and
  the provenance record for the published traces.

Status: **the generator is built and proven on the tiny fixture. The real 124 GB trace has not been
and cannot be generated on this box**, for capacity reasons recorded in §5.3 (not for GPU contention,
and not because no engine can load the checkpoint — vLLM 0.34.0 can). Part 2 therefore ingests what
was published instead; §5.4 records what each dataset is, what it can prove, and what it cannot. No
kernel, no oracle comparison, no `core/build.zig` and no committed fixture was changed by either part;
the only file this added under `tests/` is `tests/golden/tiny/noise_floor.json`.

---

## 1. Environment, pinned

Everything below ran in `tools/ref-trace/.venv` (gitignored), bootstrapped from `requirements-real.txt`:

| component | value | why it matters |
|---|---|---|
| `transformers` | **5.24.0.dev0 @ `d8f10a82213a3a9db48385aba3a4b57ab535de50`** | the commit the committed fixtures were produced at; `reference/{modeling,configuration}_qwen4_exp.py` are byte-identical to it (re-checked by hand, and by `run.sh`) |
| `torch` | **2.15.0+cpu** | the version recorded in `tests/golden/tiny/greedy_ids.json` |
| `numpy` / `safetensors` | 2.5.5 / 0.8.0 | |
| python (venv) | 3.13.12 | |
| default `torch.get_num_threads()` | **12** | the "default threads" side of the noise floor |

```bash
cd tools/ref-trace
/home/gary/.local/bin/uv venv -p 3.13 .venv
/home/gary/.local/bin/uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu
/home/gary/.local/bin/uv pip install --python .venv/bin/python -r requirements-real.txt
```

**Pin hazard, recorded because it will bite the next person:** `run.sh` installs
`transformers @ git+https://github.com/huggingface/transformers` with no commit. `origin/HEAD` was
already at `fa9c04ea60c5e1d2ecc38064442973fa1ddccc72`, so `./run.sh` on a fresh venv today fails its
own "vendored copy must be byte-identical" check — and if that check were removed it would silently
regenerate the goldens with a different reference. `requirements-real.txt` pins the commit;
`run.sh` does not. Fixing `run.sh` was not in scope here.

## 2. The determinism contract

A reference you cannot regenerate is a file, not a reference. `reference_trace.py` enforces:

1. `--threads 1` for anything committed (`torch.set_num_threads(1)`).
2. `--attn eager`, batch = 1, one prompt per forward, `dtype float32` for the tiny fixture.
3. Greedy argmax only — no sampling, no warpers, no `eos` stop — open-coded so the logits at *every*
   generated position are captured.
4. A determinism self-check before anything is written: prompt 0 is prefilled twice and the run
   aborts unless the two are bit-identical. That is the guard against recording a reference from an
   environment with a nondeterministic kernel (GPU atomics in a reduction), which is the one case
   where "we'll average it out" is a lie.
5. Provenance stamped into every artefact: config hash, prompt-file hash, `transformers` version +
   commit, `torch`/`numpy`/python/accelerate versions, requested **and** effective thread count,
   requested device map + resolved per-device parameter placement, repo revision, hostname,
   `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`CUDA_VISIBLE_DEVICES`, load seconds.
6. Prompt ids are authoritative in `prompts/*.json`; `--tokenizer <path>` re-derives them from the
   recorded text with the checkpoint's own tokenizer and **fails** if they differ, so a tokenizer
   change can never quietly turn the reference into a different token stream.

## 3. Proof on the tiny fixture

`prompts/tiny-v1.json` is `trace.py`'s 12-token input, so the generated logits are directly
comparable with the committed `final/logits` and `decode/final/logits` tensors.

```
$ cd tools/ref-trace && ./.venv/bin/python reference_trace.py generate \
      --model ../../tests/golden/tiny --prompts prompts/tiny-v1.json \
      --positions 16 --out traces/tiny-v1.safetensors
prompt set tiny-v1.json: 1 prompts, sha256 e2458fcba306
wrote traces/tiny-v1.safetensors: 4 tensors, 7196 elements, threads=1, load 2.8s
  tiny12: 12 prompt + 16 gen tokens, first ids [169, 252, 39, 151, 132, 239, 223, 20]

$ ./.venv/bin/python verify_reference_trace.py --trace traces/tiny-v1.safetensors
trace traces/tiny-v1.safetensors
  transformers_version   5.24.0.dev0
  transformers_commit    d8f10a82213a3a9db48385aba3a4b57ab535de50
  torch_version          2.15.0+cpu
  threads_effective      1
  attn_implementation    eager
  dtype                  float32
  prompt_file_sha256     e2458fcba306ad33e685b393bee5a6811524af4ea40f2b8dd2f4ca8498156777
  load_seconds           2.791
noise floor noise_floor.json: run_a 1 thread(s) vs run_b 12 threads -> max|dlogit| 4.470e-08, KL max 5.522e-16
bounds (margin 2.0x): max|dlogit| <= 8.941e-08, KL <= 1.599e-14 = max(2.0x measured 1.104e-15, derived 2*bound^2 1.599e-14)
  prefill logits         max|dlogit| 0.000e+00 (<= 8.9e-08)  KL mean 0.000e+00 p99 0.000e+00 (<= 1.6e-14)  argmax agree 1.0000  OK
  decode logits[0]       max|dlogit| 0.000e+00 (<= 8.9e-08)  KL mean 0.000e+00 p99 0.000e+00 (<= 1.6e-14)  argmax agree 1.0000  OK
  greedy ids             100.0% of 16 tokens match greedy_ids.json OK
    ours   [169, 252, 39, 151, 132, 239, 223, 20, 198, 120, 47, 121, 121, 144, 96, 19]
    golden [169, 252, 39, 151, 132, 239, 223, 20, 198, 120, 47, 121, 121, 144, 96, 19]
verify: PASS
```

That is a stronger statement than "within the noise floor": at `threads=1` the generator's logits are
**bit-identical** to `final/logits` (0.0 difference) and to the decode-step tensor, and the
open-coded greedy loop reproduces `model.generate(do_sample=False)` on all 16 tokens of
`greedy_ids.json`. The plumbing is right.

### Non-vacuity: the bound refuses a known KL

`--inject-kl` log-bisects a perturbation until *its own measured mean KL* equals the target, then
runs the gate — so the refusal is expressed in the unit the gate uses, not in raw logit wobble.

```
$ ./.venv/bin/python verify_reference_trace.py --trace traces/tiny-v1.safetensors --inject-kl 1e-3
  non-vacuity: injecting mean KL 1.0e-03 -- amplitude 1.382e-01 x max|logit| measured back as 1.000e-03
  prefill logits         max|dlogit| 1.832e-01 (<= 8.9e-08)  KL mean 1.000e-03 p99 1.177e-03 (<= 1.6e-14)  argmax agree 0.3333  FAIL
  decode logits[0]       max|dlogit| 1.180e-01 (<= 8.9e-08)  KL mean 5.260e-04 p99 5.260e-04 (<= 1.6e-14)  argmax agree 0.0000  FAIL
verify: FAIL                                     (exit 1)

$ ... --inject-kl 1e-9      -> FAIL  (KL mean 1.000e-09, 5 orders above the floor)
$ ... --inject-kl 1e-16     -> PASS  (max|dlogit| 7.451e-08, argmax agreement 1.0000)
```

The 1e-16 run is the interesting one: it passes at `max|dlogit| 7.45e-8`, i.e. **the bound sits
exactly where the reference's own thread-to-thread spread sits** (4.47e-8 with a 2x margin). It is
not 1,000x loose and it is not a wish.

Two derived bounds are combined and the gate takes the looser one:

* `max|Δlogit| <= margin × measured spread`;
* `KL <= max(margin × measured KL, 2 × (logit bound)²)`.

The second term exists because KL is *quadratic* in a logit perturbation: a 4.5e-8 logit wobble is a
~1e-15 KL, so asking an independent machine to beat the measured KL alone would fail it for being
better behaved than the box that recorded the floor.

## 4. The reference's own noise floor

Measured by running the *same* model, prompt set and code path twice in one process, changing only
`torch.set_num_threads`: 1 (canonical) then 12 (this box's default). Recorded in
`tests/golden/tiny/noise_floor.json`.

```
$ ./.venv/bin/python reference_trace.py noise-floor --model ../../tests/golden/tiny \
      --prompts prompts/tiny-v1.json --positions 16 \
      --out-report ../../tests/golden/tiny/noise_floor.json
  threads: 1 -> 12, runs bit-identical: False
  p0/gen_logits     max|dlogit| 2.980e-08  KL mean -5.403e-17 p99 5.131e-16 max 5.522e-16  argmax agree 1.0000  dPPL floor 2.249e-07
  p0/prompt_logits  max|dlogit| 4.470e-08  KL mean -7.033e-17 p99 4.996e-16 max 5.003e-16  argmax agree 1.0000  dPPL floor 9.706e-07
```

Re-running `noise-floor` from a fresh process reproduces `aggregate` exactly (verified: identical
`max_abs_logit_diff`, `kl_*`, `delta_ppl_floor_max` and thread counts; only `load_seconds` differs),
so the floor itself is a reproducible measurement and not one lucky pair of runs.

| metric | floor (tiny, 1 thread vs 12) | what it means for a gate |
|---|---|---|
| max abs logit difference | **4.47e-8** prefill, 2.98e-8 decode | reproduces the 4.5e-8 / 3.0e-8 figures in `tests/golden/README.md` independently |
| KL mean / p99 / max | ~0 (−7e-17) / 5.0e-16 / 5.5e-16 | **KL is blind to this noise.** Any KL above ~1e-14 is signal, not thread order |
| ΔPPL | **9.7e-7** prefill, 2.2e-7 decode | a ΔPPL gate below ~1e-6 is measuring torch's reduction order |
| top-1 agreement | 1.0000 on 28 positions | this fixture has no near-tie positions; a real prompt set will, and that is where flips come from |

**The trap this found, in the instrument itself.** Computed in fp32, the same comparison reported
`KL mean 3.16e-9` for the prefill and a *negative* `−4.6e-9` for a 1e-3 perturbation: KL of two
nearly identical fp32 distributions is cancellation noise, roughly **seven orders of magnitude too
big**, and it can be negative. Every KL/PPL reduction in `reference_trace.py` is therefore float64
(`log_softmax_f64`, chunked so a 248k-vocab trace stays bounded). Had the fp32 number been taken at
face value, the "noise floor" would have been written down as 3e-9 and any real quality gate at
1e-6 would have looked generous while the actual floor was 1e-15.

## 5. Feasibility of the real 124 GB reference

Produced with `reference_trace.py io-probe`, which reads only safetensors headers, this box's
`/proc/meminfo`, `nvidia-smi`, and one shard's bytes. **No model load, no GPU work.**

```
$ ./.venv/bin/python reference_trace.py io-probe --model ~/models/qwen38-flash-next-nvfp4 \
      --prompts prompts/quality-v1.json --positions 64 --read-mib 2048 --read-streams 4 \
      --load-shard model-00002-of-00010.safetensors --out traces/real-v1-io-probe.json
checkpoint 123.5 GiB in 299545 tensors across 11 files
  routed experts (NVFP4)                65.63 GiB     70.46 G elements -> bf16 would be   131.25 GiB
  PLE n-gram table (gather-only)        47.68 GiB     51.20 G elements -> bf16 would be    95.37 GiB
  dense text path                        6.41 GiB      3.44 G elements -> bf16 would be     6.41 GiB
  embed_tokens + lm_head                 2.37 GiB      1.27 G elements -> bf16 would be     2.37 GiB
  vision tower                           0.84 GiB      0.45 G elements -> bf16 would be     0.84 GiB
  shared expert + router                 0.44 GiB      0.24 G elements -> bf16 would be     0.44 GiB
  MTP layer                              0.17 GiB      0.09 G elements -> bf16 would be     0.17 GiB
  materialised bf16 for the whole checkpoint: 236.8 GiB vs 118.2 GiB of VRAM + available RAM
  largest tensor model-00009-of-00010.safetensors:lm_head.weight 1212.5 MiB
quantiser: modelopt dispatches=False
  BLOCKER ValueError: Unknown quantization type, got modelopt - supported types are: [... 'nvfp4' ...]
read bandwidth: 4.56 GiB/s cold (4 streams x 2048 MiB, direct_io=True, 4 files evicted) vs 15.35 GiB/s warm -> arithmetic uses cold
box: VRAM 63.7 GiB total, RAM available 54.5 GiB, 4 GPUs
placement: 123.5 GiB on disk vs 63.7 GiB VRAM budget + 40.9 GiB CPU budget -> 18.9 GiB falls to disk
shard load measured: model-00002-of-00010.safetensors 9.31 GiB -- mmap setup 5.2 s, page-touching every 4 KiB 0.2 s = 49.71 GiB/s; process peak RSS 0.5 -> 1.8 GiB
arithmetic: 18.9 GiB re-read per forward call -> 4.2 s per call at the measured 4.56 GiB/s; 585 calls for this prompt set = 0.7 h of transfer alone, plus 0.5 min cold load
```

### 5.1 Measured

| quantity | value | how |
|---|---|---|
| checkpoint footprint | 123.53 GiB, 299,545 tensors, 127.15 G elements | safetensors headers |
| …materialised to bf16 | **236.8 GiB** | 2 bytes × element count |
| box capacity for weights | 63.7 GiB VRAM (4 × 16311 MiB by `nvidia-smi`; `cuDeviceTotalMem` says 62.1 GiB) + 54.5 GiB RAM available | `nvidia-smi`, `/proc/meminfo` |
| cold sequential read, 4 O_DIRECT streams | **4.56 GiB/s** and 6.11 GiB/s (two repetitions) | 4 × 2 GiB, page cache dropped first |
| same read warm | 15.35 / 17.03 GiB/s | *do not plan against this* — see the note below |
| single-stream O_DIRECT (`dd`) | 3.35 GiB/s | `dd iflag=direct`, 2 GiB at two offsets |
| load path, one 10 GB shard, cache dropped | mmap setup 5.2 s (**1.79 GiB/s**), page-touch +0.2 s, process peak RSS 0.5 → 1.8 GiB | `safetensors.torch.load_file` + one byte per 4 KiB |
| trace size, `quality-v1` @ 64 positions | **1.17 GiB** fp32 logits | (688 prompt + 576 generated) = 1,264 positions × 248,320 vocab × 4 B |

The cold number moves 34 % between repetitions, so every estimate below is computed at the
**conservative 4.56 GiB/s**; where a figure is hand-computed from another row, the divisor is that one.

Two measurement traps, both fixed in the tool rather than in the prose:

* **btrfs answers `O_DIRECT` reads from the page cache.** The first version of this probe reported
  15 GiB/s, which is memcpy, not the 980 PRO. `measure_read_bandwidth` now `posix_fadvise(DONTNEED)`s
  the files it is about to read and labels every number cold or warm. Cold is 4.56 GiB/s.
* **`safetensors.load_file` is mmap-backed.** Loading a 9.31 GiB shard moved RSS by 1.3 GiB: the
  bytes were mapped, not read. The number that matters for `from_pretrained` is the cold map/copy
  rate (1.79 GiB/s here), which is why "load time" cannot be inferred from `load_file`'s wall clock.

### 5.2 The wall-time estimate, with the arithmetic

`quality-v1` is 9 prompts / 688 tokens; at 64 positions each the run is
`9 × (1 prefill + 64 decode) = 585 forward calls`. Under `accelerate` offload every module that fell
to disk is re-read **on every forward call** — a decode token is one call — so:

| scenario | pinned (VRAM + RAM) | spills per call | seconds per call | 585 calls |
|---|---|---|---|---|
| probe default (all of VRAM, 75 % of RAM) | 104.6 GiB | 18.9 GiB | 4.1 s | **0.67 h** |
| honest budget (1.5 GiB/GPU left for context+activations, 16 GiB of RAM left for torch/staging), measured by the probe at 6.11 GiB/s and recomputed at 4.56 | 96.2 GiB | 27.3 GiB | 6.0 s | **0.97 h** |
| n-gram table ends up on disk (it is one 47.7 GiB `nn.Embedding`) | — | ≥ 47.7 GiB | ≥ 10.5 s | **≥ 1.71 h** |
| experts also spill (+65.6 GiB) | — | + 65.6 GiB | + 14.4 s | **+ 2.34 h** |
| **bf16-materialised checkpoint (236.8 GiB)** | 96.2 GiB | 140.6 GiB | 30.8 s | **5.01 h**, and staging it needs 237 GiB of disk against **95 GiB free** → not possible here |

Cold load of the whole checkpoint is *not* the problem: 123.5 GiB at 4.56 GiB/s = 27 s, plus ~9 s of
per-tensor bookkeeping at ~30 µs × 299,545 tensors (not measured). Compute is not the problem
either: the reference's own Python loops dominate it — 48 layers × 512 experts = 24,576 expert
iterations per prefill call and 48 × 10 = 480 per decode token, at ~0.2–0.5 ms each (index_select,
two GEMMs and a device sync) that is 5–12 s per prefill call and 0.1–0.24 s per token, plus the QSA
indexer's per-(batch, query) loop (12 indexed layers × T queries with an inner loop over
T/4 blocks). Those per-iteration costs are **estimates, not measurements** — measuring them means
touching the GPUs, which was out of bounds. They are an order of magnitude below the transfer term
in every scenario above.

**Answer to "how long, on a quiet box":** one pass over `quality-v1` at 64 positions is
**0.7–1.7 h of transfer** plus an estimated 0.2–0.5 h of reference compute, so **1–2 h**. The
twice-run noise-floor protocol is one load and two sweeps, so budget **2–4 h**; if the placement
also spills the routed experts it becomes **6–9 h**. It is not a day-long job at this prompt size —
but `--positions` is a direct multiplier, because each generated token is one more forward call at
~6 s: 64 positions over the 9 prompts is 576 calls ≈ 1 h of transfer, 500 would be 4,500 calls
≈ 7.5 h. Scale `--positions` deliberately, and re-run `io-probe` with the new value before launching:
it prints the call count and the transfer total without touching a GPU. And see 5.3: today the run
cannot start at all.

### 5.3 The blocker, corrected in part 2: the loader was not the wall, capacity is

Part 1 recorded the `transformers` refusal as *the* finding. That refusal is still exactly true —
`AutoQuantizationConfig.from_dict` on the checkpoint's `quantization_config` raises

```
ValueError: Unknown quantization type, got modelopt - supported types are: [...]
```

— but it was the wrong wall to stop at, and part 1's conclusion that the reference "needs
`nvidia-modelopt` … or a second weight-streaming engine" undersold the one route that actually
exists. The model card names vLLM as the supported engine, and vLLM 0.34.0 (measured in a scratch
venv: `torch 2.14.0+cu130`) registers all three Qwen4Exp architectures and the checkpoint's
quantiser:

```
vllm 0.34.0 torch 2.14.0+cu130
supported archs: 426
Qwen4Exp present: ['Qwen4ExpForCausalLM', 'Qwen4ExpForConditionalGeneration', 'Qwen4ExpMTP']
quant methods: [..., 'modelopt', 'modelopt_fp4', 'modelopt_mixed', ...]
```

So "no Python engine can load this checkpoint" is false. What is true is arithmetic:

| quantity | value | source |
|---|---:|---|
| checkpoint on disk | 123.53 GiB (132.73 GB), 299,545 tensors | safetensors headers |
| VRAM, four GPUs | 63.7 GiB | `nvidia-smi` |
| RAM available | 54.5 GiB | `/proc/meminfo` |
| vLLM `--cpu-offload-gb` capacity | pinned host memory — it cannot swap | vLLM's offload path |

123.53 GiB of weights do not fit in 118.2 GiB of VRAM + RAM, and the shortfall is structural, not a
tuning knob: TP=4 shards the routed experts across GPUs (65.63 GiB / 4 = **16.4 GiB per GPU**, more
than the 15.9 GiB a 16 GB card actually offers) before any dense weight, PLE table or KV cache is
placed. No placement of this checkpoint fits this box, so an independent *run* of the published
NVFP4 weights is unavailable here. §5.4 is what replaced it.

The same arithmetic rules out the sibling FP8 checkpoint the card benchmarks against
(`Qwen/Qwen3.8-Flash-Next-FP8`): it is **185.56 GB (172.82 GiB) in 144 files**, against 82–91 GB of
free disk on this box during this work — the download alone needs ~2x the free space. Its *loader*
is fine: `transformers` 5.24 dispatches its `quant_method: fp8` / `weight_block_size: [128,128]`
config to `FineGrainedFP8Config`, and vLLM knows the architecture. That route is closed by disk and
memory, not by software, and it is recorded here so the next person does not discover it from a
failed 186 GB download.

**Dtype facts from the model card, because they change what a reference would even be.** Only the
routed MoE experts in the main language model are NVFP4 (W4A4, group 16, MSE-calibrated scales).
Attention layers, shared experts and the other main-model layers remain **BF16**; the MTP routed
experts are **128×128 block-scaled FP8**; the PLE n-gram table is **per-tensor FP8**. Anything that
assumes "the checkpoint is FP4" would be wrong about 7 GiB of dense weights, the 47.7 GiB PLE table
and the 2.7 GiB MTP module.

**What part 1 got right and part 2 kept:** the `transformers` path is still shut, and materialising
bf16 is still 236.8 GiB against 118.2 GiB of VRAM+RAM — so a *from-scratch* HF reference remains
impossible on this box for a second, independent reason. The random-weight fixture route (item 2
below) is still the right instrument for threshold-setting; what §5.4 adds is a way to measure a
real model's behaviour without running one.

### 5.4 Published traces: what exists, what it proves, what it does not

Four datasets were ingested with `published_traces.py`. Every artefact is fetched at a **pinned
dataset revision** and checked against the publisher's own manifest before it is used; all four
blocks below distinguish what the publisher *states* from what it leaves unsaid, because a trace is
only a reference to the extent those gaps are known.

First, the baseline this all hangs off: **our local copy is the published file.** `sha256sum` of all
13 LFS-backed files under `~/models/qwen38-flash-next-nvfp4` matches the Hub's LFS oids exactly
(132.7 GB, 0 failures, `traces/published/nvfp4-local-vs-published-sha256.json`), so the model card's
statements about the published checkpoint describe the bytes on this box, not a lookalike.

| dataset | revision | size | what it is | what it cannot do |
|---|---:|---|---|---|
| `Vemloir/Qwen3.8-Flash-Next-traces` | `26fdf6eddae7` | 0.38 GiB | calibration **inputs** (`input_ids` + `lengths`) generated by the model itself | no logits, no activations: cannot grade anything |
| `satsder/qwen3.8-flash-next-routing-traces` | `6e29444b7472` | 45.85 GiB (one `.7z`) | llama.cpp-branch hidden/router/expert dumps, 85 request dirs | not downloaded (49.2 GB + 54.4 GB extracted); model/tokeniser identity uncertified |
| `jyohukuchan/qwen38-flash-next-bf16-logits` | `5294aa9ab94b` | 78.82 GiB | **BF16 reference logits** from exllamav3, 21 cases / 32,359 positions | max 4,097 positions, no states/router ids, and a documented cross-engine disagreement |
| `aswinkumar99/qwen3.8-flash-next-expert-traces` | `13969e686ba0` | 1046 GiB | FreeToken traces: top-10 experts per token per layer, top-32 logits, 5-tap states, LM-head top-8 | 5 taps only in `compact/`, no full-vocab logits, different publisher's NVFP4 export |

#### Vemloir: a calibration input set, not a reference

Measured from the safetensors headers: **only `input_ids` (int64, right-padded with 0) and
`lengths` (int64)**, plus `mm_len`/`mm_row`/`mm_start` and raw `mm_image_*` bytes in the multimodal
part. No logits and no hidden states anywhere in the three parts.

| part | `input_ids` | rows | tokens (min / max / mean) |
|---|---|---:|---|
| agents | `[40, 200863]` | 40 | 8,669 / **200,863** / 49,978 |
| chat | `[700, 43460]` | 700 | 88 / 43,460 / 3,892 |
| multimodal | `[105, 10349]` | 105 | 341 / 10,349 / 2,628 |

Provenance: generated by Qwen3.8-Flash-Next itself, served as `nvidia/Qwen3.8-Flash-Next-FP4-Local`
in vLLM (card), at four reasoning levels. **Not stated:** the model revision, a tokeniser file or
hash, and there are no outputs. What it *is* good for: 200k-token prompt ids from the model's own
distribution, for the long-context path, where the only other published artefacts stop at 4k.

#### satsder: per-token, per-layer expert ids — but not fetched, and not certified

The routing record is documented (in `DATASET_INFO/README.md`, Chinese) as:

* `experts.csv`: `graph_id,split_id,layer,tensor,token_row,rank,expert_id,sched_id` — **yes, expert
  ids at token × layer × rank granularity**;
* `router/rl_*.bin` / `rp_*.bin`: 512 float32 gate logits (2,048 bytes per single-token file), and
  **only for single-token graphs**, so not the full prefill;
* `hidden/hidden_meta.csv`: F32 `[2560, 1, nt]` hidden states, and `tokens.csv` token ids.

The archive is 49,235,820,003 bytes → 50.69 GiB uncompressed across **4,158,067 files**, against
82–91 GB free: downloading plus extracting it would consume both, so it was **not** fetched. Its
provenance is the weakest of the four: a locally modified llama.cpp branch, context 4,096, Q8 KV,
16 generated tokens, no launch/exit manifest for the `collect-*` batches, and the publisher states
the records are not enough to bind every batch to a certified model/tokeniser/collector snapshot.
A dead end worth recording, with the size that makes it one.

#### jyohukuchan: the only artefact here that can grade an output

**BF16 logits of the official `Qwen/Qwen3.8-Flash-Next` at revision `de4b8e4d43b9…`, produced by
exllamav3 (BF16 weights), FP32 raw pre-softmax rows, temperature 1, one streamed forward per case.**
21 cases / 32,359 positions, 32.14 GB of FP32 on the Hub:

| group | cases | shapes (rows × 248,320) |
|---|---|---|
| mtp-bench | 11 | 55, 129×2, 257×3, 513×2, 1,025, **4,097×2** |
| WikiText-2 | 10 | 2,103 rows each = 2,048 scored tokens behind a 55-token chat-template prefix |

Measured from the manifests: every case carries `positions`, `nonfinite_count` (all **0**),
`input_token_ids_sha256`, and `corpus/all-cases.json` publishes the exact `token_ids` — so the input
side is authoritative and checkable. Candidates: `candidate-xestrata` (EXL3 3.05 bpw, all 21 cases)
and `candidate-turboderp-exl3-3.05bpw` (WikiText-2 only, prefix rows removed).

Exact file sizes, measured: `manifest.json` 10,476 B; `reference/manifest.json` 311,791 B;
`candidate-xestrata/manifest.json` 311,759 B; `candidate-turboderp-exl3-3.05bpw/manifest.json`
196,277 B; `corpus/all-cases.json` 184,870 B. One case is `rows × 248,320 × 4` bytes — 128,133,120 B
for the 129-position `en_prose_129`, 2,088,867,840 B for a WikiText-2 row — so a single case plus
the metadata is ~2 GB, which is what this lane fetched; the other 75 GiB stayed on the Hub.

Two measured facts pin the ingestion down:

* **We reproduce the publisher's own number.** `published_traces.py kl --case wiki2_r0` re-derives
  KL(reference ‖ XeStrata) in float64 over the 248,077 valid ids: **0.4667382507872193** against the
  published **0.4667382507871942** (2.5e-14 apart), top-1 agreement 0.85986328125 identical. The
  WikiText-2 headline 0.1587 in the card is the mean over the ten rows; row 0 is the outlier.
* **The existing metrics run on it unchanged.** Converting `en_prose_129` with
  `published_traces.py to-trace` gives a `p0/prompt_ids` + `p0/prompt_logits` safetensors file;
  `reference_trace.distribution_metrics` then reports KL mean 0.074203, p50 0.043729, p99 0.354432,
  argmax agreement 0.852713, ΔPPL 321.68 — and the publisher's `results/kl-xestrata-vs-bf16.json`
  has en_prose_129 at exactly 0.074203 / 0.852713.

**The caveat that matters more than any of that:** the publisher rejected `transformers` 5.20.2
`qwen4_exp` as a reference engine — KL 3.2 (reference vs transformers) / 4.1 (reverse), top-1
agreement 0.3101, next-token accuracy 0.234 against exllamav3's 0.516 on the same inputs
(`results/engine-crosscheck.json`). Two independent implementations of this architecture do not
agree by default. **A KL gate against this trace measures engine disagreement first and
quantisation second**, and the publisher's own candidate-path noise floor (CPU-expert vs GPU-expert
path) is KL **0.0149** — 30x the BF16-vs-BF16 thread noise of the tiny fixture and the correct scale
for any threshold derived from this dataset.

#### aswinkumar99: real-text routing, and the weight-level link to our checkpoint

**FreeToken 0.1.2 (branch `expert-trace-study`) traces of `RadixArk/Qwen3.8-Flash-Next-NVFP4`**, a
different publisher's NVFP4 export with the same recipe (routed experts only, group 16, routers
excluded from quantisation and kept BF16 with no bias). 18 raw runs (1,046 GiB total), plus a
`compact/` training set: **2,222,925 decode rows in 98.1 GB**, per row

| array | dtype | shape | meaning |
|---|---|---|---|
| `experts.npy` | int16 | `[rows, 48, 10]` | routed top-10 expert ids, rank order |
| `gates.npy` | float16 | `[rows, 48, 10]` | renormalised gate weights |
| `logit_idx.npy` / `logit_val.npy` | int16 / float16 | `[rows, 48, 32]` | top-32 router ids and logits |
| `taps.npy` | float16 | `[rows, 5, 2560]` | hidden state at probes 0/12/24/36/47 |
| `stats.npy` | float16 | `[rows, 5, 48]` | entropy, top-10 mass, norms |
| `token_id.npy` etc. | int32 | `[rows]` | exact input id, request, position |

The raw `trace.bin` format (v3) additionally records `token_id`, `prev_token_ids[3]` and
**`ple_hash_ids[16]`** per token — the actual PLE table rows the model gathered, which is a direct
oracle for the n-gram id derivation I-011 has only ever seen on the tiny fixture.

Fetched here: `compact/batch-001/{experts,gates}.npy` 455,474,048 B each, `token_id.npy` 1,897,936 B,
`diagnostic/router_weights.npz` 309,433,101 B (56 tensors: the 48 gates, three hyper-connection
mixers and their MTP counterparts, all fp32), plus the manifests and `manifest.jsonl` (18 runs).
The remaining 1 TB of `trace.bin` / `taps.npy` / heads was not fetched.

**The provenance link was measured, not assumed.** `published_traces.py router` compares the
published `diagnostic/router_weights.npz` with our checkpoint's unquantised weights:

```
48/48 gate matrices bit-identical, worst max|diff| 0.0
  hc_norm.weight: bit_equal True max|d| 0.0
  input_mix_weight_down.weight: bit_equal True max|d| 0.0
  input_mix_weight_up.weight: bit_equal True max|d| 0.0
```

So RadixArk's routers and hyper-connection mixers are the same tensors as NVIDIA's, bit for bit:
`W_l @ x` with our weights is their router, and a top-10 computed from a *matching hidden state* is
the same top-10. What is **not** established is that the NVFP4 expert weights are identical between
the two exports — so the traces are a router/weight oracle, not proof that the whole forward agrees.

**And the recorded router output reproduces from those weights.** `published_traces.py router-trace`
recomputes `W_l @ x` for the first 64 rows of `compact/batch-001` out of the recorded five taps and
our own gate matrices, then compares against the recorded top-32 logits (after the recorder's per-row
max shift) and the executed top-10:

| layer | top-10 set overlap | exact rank order | max abs logit diff | mean abs logit diff |
|---|---:|---:|---:|---:|
| 0 | 0.9906 | 0.672 | 0.02214 | 0.00753 |
| 12 | 0.9969 | 0.672 | 0.02282 | 0.00771 |
| 24 | 0.9891 | 0.734 | 0.02325 | 0.00725 |
| 36 | 0.9859 | 0.547 | 0.02311 | 0.00774 |
| 47 | 0.9859 | 0.484 | 0.02664 | 0.00819 |

The mean logit difference **0.0072–0.0082** is exactly the "~0.008, pure bf16 GEMV accumulation
noise" the publisher reports, recomputed independently here. The top-10 *set* survives; the *rank
order* does not (0.48–0.73 exact), because near-tie margins are inside that noise. **Gate router
agreement on set overlap, never on rank order** — and on a real cross-machine replay the publisher
measured **95.7% mean top-10 set agreement** (100% on the same machine), with 25% of router margins
inside 0–2 bf16 ulps. That is the floor under any router-agreement gate built on this dataset.

#### The question that decides the long-context path

**None of the four gives 262,144-token behaviour.** The longest logits case is **4,097 positions**
(two mtp-bench cases), WikiText-2 is 2,048 scored positions, satsder's context is 4,096, and
aswinkumar99's runs are long-document prompts traced at per-token decode granularity, not at full
context. Vemloir has 200,863-token *inputs* but no outputs at all. So the published material can
validate short-context numerics and router behaviour; the 262,144-token path has no external
reference and must be gated on the engine's own consistency (I-018) plus the CPU reference.

## 6. How to regenerate everything

```bash
cd tools/ref-trace                       # venv: see §1, or requirements-real.txt
PY=./.venv/bin/python

# 1. generator + proof against the committed goldens (seconds, CPU, no GPU)
$PY reference_trace.py generate --model ../../tests/golden/tiny \
    --prompts prompts/tiny-v1.json --positions 16 --out traces/tiny-v1.safetensors
$PY verify_reference_trace.py --trace traces/tiny-v1.safetensors            # PASS

# 2. non-vacuity: must exit 1
$PY verify_reference_trace.py --trace traces/tiny-v1.safetensors --inject-kl 1e-3

# 3. the tiny noise floor (rewrites tests/golden/tiny/noise_floor.json)
$PY reference_trace.py noise-floor --model ../../tests/golden/tiny \
    --prompts prompts/tiny-v1.json --positions 16 \
    --out-report ../../tests/golden/tiny/noise_floor.json

# 4. cost of the real run, without touching the GPUs
$PY reference_trace.py io-probe --model ~/models/qwen38-flash-next-nvfp4 \
    --prompts prompts/quality-v1.json --positions 64 --read-mib 2048 --read-streams 4 \
    --load-shard model-00002-of-00010.safetensors --out traces/real-v1-io-probe.json

# 5. the real reference is NOT runnable on this box (§5.3): 123.53 GiB of weights against
#    63.7 GiB VRAM + 54.5 GiB RAM. Do not start the 186 GB FP8 download either.

# 6. published traces instead (no model load, no GPU; downloads land in traces/, gitignored)
$PY published_traces.py inspect      # writes traces/published/provenance.json
$PY published_traces.py verify       # sha256 every downloaded file against its manifest
$PY published_traces.py kl --case wiki2_r0                 # reproduces the publisher's KL to 2.5e-14
$PY published_traces.py router --model ~/models/qwen38-flash-next-nvfp4   # 48/48 gates bit-identical
$PY published_traces.py router-trace            # recorded top-10 reproduces from the recorded taps
$PY published_traces.py to-trace --case en_prose_129 \
    --out traces/published/en_prose_129.safetensors          # then reference_trace.distribution_metrics
```

`traces/` is gitignored: a real trace is >1 GiB and regenerable, so it is never committed. The
committed artefacts of this issue are `prompts/*.json`, `tests/golden/tiny/noise_floor.json`, the
three scripts (`reference_trace.py`, `verify_reference_trace.py`, `published_traces.py`) and this
note.

## 7. What this lane did **not** do

* No real-checkpoint trace and no real-checkpoint noise floor. §5.3 is why: not the loader (vLLM
  0.34.0 would take it), but 123.53 GiB of weights against 118.2 GiB of VRAM + RAM.
* No GPU work of any kind. `io-probe` reads headers, `/proc/meminfo` and `nvidia-smi`; the vLLM
  capability check imported the package and queried its registry without initialising CUDA; the
  published traces were ingested on CPU. The compute term in §5.2 remains arithmetic on the
  reference's loop structure, not a measurement.
* No full download of the `satsder` routing archive (45.85 GiB compressed, 50.69 GiB extracted,
  4.16 M files) or of the FP8 checkpoint (185.56 GB). Both sizes are measured and recorded above;
  neither fits the free disk.
* No byte-for-byte comparison of our PLE n-gram table against `Qwen/Qwen3.8-Flash-Next-FP8`. The
  FP8 table has the same 128 shards of `[2500012, 160]` F8_E4M3 and the same 51,200,245,760 data
  bytes as ours (headers measured), and the model card states they are byte-identical, but the
  51.2 GB range-fetch needed to prove it was not run: HF delivered 17.6 MB/s on one stream and
  33.8 MB/s on eight, i.e. ~25–50 minutes. That is the cheapest remaining provenance check.
* No real-model noise floor of our own, so the numbers in §4 remain *tiny-fixture* numbers. For the
  published logits the publisher's own floors apply instead: cross-engine KL 3.2/4.1 and top-1
  0.31 (transformers vs exllamav3), candidate-path KL 0.0149. For routing, the measured floor is the
  one in §5.4: mean |Δlogit| 0.0072–0.0082 from bf16 GEMV accumulation, 0.986–0.997 top-10 set
  overlap on one machine, and 95.7% across machines.
* No I-031 metric test under `zig build test`, no change to any kernel tolerance, oracle
  comparison, `core/build.zig`, or committed fixture.
* No edit to `run.sh`'s unpinned `transformers` install, even though it currently cannot bootstrap a
  venv that matches the fixtures (§1).
