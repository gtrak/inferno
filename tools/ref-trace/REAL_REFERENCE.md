# The real-model quality reference (I-031, part 1): generator, noise floor, feasibility

Three things live here:

* **`reference_trace.py`** — runs HF `transformers` over a checkpoint for a fixed prompt set and a
  fixed number of positions and writes a *logits* trace (prompt ids, logits per position, greedy
  continuation, full provenance). Mode `io-probe` answers "what would the real run cost" without
  loading a model or touching a GPU.
* **`verify_reference_trace.py`** — proves the generator reproduces the committed goldens, and proves
  the bound refuses a *known* KL (non-vacuity).
* **this file** — the noise-floor record, the regeneration recipe, and the measured feasibility note.

Status: **the generator is built and proven on the tiny fixture. The real 124 GB trace has not been
generated**, and §5.3 records a blocker that is not about GPU contention. No kernel, no oracle
comparison, no `core/build.zig` and no committed fixture was changed; the only file this added under
`tests/` is `tests/golden/tiny/noise_floor.json`.

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

### 5.3 The blocker (this is the finding, not the timing)

`transformers` at the pinned commit **cannot load this checkpoint at all**, quantised or not:

```
config.json: "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", ...}
AutoQuantizationConfig.from_dict(...) ->
  ValueError: Unknown quantization type, got modelopt - supported types are:
  ['awq','bitsandbytes_4bit','bitsandbytes_8bit','gptq','aqlm','quanto','quark','fouroversix',
   'fp_quant','eetq','higgs','hqq','compressed-tensors','fbgemm_fp8','torchao','bitnet','vptq',
   'spqr','fp8','gguf','nvfp4','mxfp8','auto-round','mxfp4','metal','sinq','gemma']
```

There is no `modelopt` entry in `transformers.quantizers.auto.AUTO_QUANTIZER_MAPPING`, and no
modelopt integration anywhere in the installed package (`grep -r modelopt` is empty). The closest
entry, `nvfp4`, is not a way in: `quantizers/quantizer_nvfp4.py` is a *post-training* quantiser for
full-precision checkpoints — `validate_environment` raises
`"Loading pre-quantized NVFP4 checkpoints is not supported yet"`, requires `kernels`, requires CUDA,
supports exactly one CUDA device, and states `"NVFP4 quantization does not support CPU or disk
offload"`. So even rewriting `quant_method` to `nvfp4` would be refused, and offload is precisely
what this box needs.

And the fallback the issue already ruled out is ruled out by arithmetic too: materialising bf16 costs
**236.8 GiB**, against 118.2 GiB of VRAM + available RAM and 95 GiB of free disk — it does not fit in
memory *or* on disk, and 102.4 GiB of that is the PLE n-gram table, which is gather-only (16 rows per
token) and would be re-read or re-mapped in its entirety by HF's `nn.Embedding` path.

Consequences, for the decision that belongs to the orchestrator and not to this lane:

1. **The "run HF over the real checkpoint once and cache it" plan does not survive contact with the
   loader.** It needs either (a) `nvidia-modelopt` installed into the reference venv and a confirmed
   offload path (its own `validate_environment` behaviour under `device_map` with CPU/disk tiers is
   unverified), (b) a dequant-on-load weight provider feeding the HF *model classes* — which means
   writing a second weight-streaming engine, against the spirit of "the reference is HF", and still
   does not fit in bf16, or (c) a bf16 copy of the checkpoint on a bigger box (236.8 GiB weights +
   working set; this box is 60/66 by construction).
2. **Everything else in I-031 can proceed on a random-weight checkpoint of the real shape.** The KL /
   ΔPPL / router-agreement / per-layer-RMS machinery, its thresholds, and the FP8-KV-vs-BF16-KV
   experiment do not need the published weights: a medium random Qwen4Exp checkpoint (real config, a
   fraction of the layers so it fits in RAM in fp32) is a *better-controlled* instrument for a
   precision study, and its noise floor is measurable today. `reference_trace.py` runs it unchanged —
   `--model <dir> --prompts ... --positions ...` — with the real prompt ids, since `prompts/quality-v1.json`
   only requires vocab ≤ 248,320.
3. If the number that must be quoted on the *published* weights is ΔPPL on a fixed text set, that
   can come from our own engine plus the I-018 acceptance runs (needle at 50 %/100 % context, chat +
   tools round-trip), which need no HF oracle at all. What cannot be replaced is a *per-position KL
   against HF on the published weights* — and that is the one this lane can currently neither produce
   nor measure.

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

# 5. the real reference — ONLY once §5.3 is resolved, on a quiet box, with the blockers cleared
$PY reference_trace.py generate --model ~/models/qwen38-flash-next-nvfp4 \
    --prompts prompts/quality-v1.json --positions 64 --dtype bfloat16 --threads 1 \
    --tokenizer ~/models/qwen38-flash-next-nvfp4 --low-cpu-mem-usage \
    --device-map auto --max-memory '{"cpu":"40GiB",0:"14GiB",1:"14GiB",2:"14GiB",3:"14GiB","disk":"80GiB"}' \
    --out traces/real-v1.safetensors
# and the twice-run floor, same prompt set, one model load (mode `noise-floor`):
$PY reference_trace.py noise-floor --model ~/models/qwen38-flash-next-nvfp4 \
    --prompts prompts/quality-v1.json --positions 64 --dtype bfloat16 \
    --tokenizer ~/models/qwen38-flash-next-nvfp4 --low-cpu-mem-usage \
    --device-map auto --max-memory '{"cpu":"40GiB",0:"14GiB",1:"14GiB",2:"14GiB",3:"14GiB","disk":"80GiB"}' \
    --out-report traces/real-v1-noise-floor.json
```

`traces/` is gitignored: a real trace is >1 GiB and regenerable, so it is never committed. The
committed artefacts of this issue are `prompts/*.json`, `tests/golden/tiny/noise_floor.json`, the two
scripts and this note.

## 7. What this lane did **not** do

* No real-checkpoint trace, and no real-checkpoint noise floor: not run, not even a load. §5.3 is why.
* No GPU work of any kind: `io-probe` reads headers, `/proc/meminfo` and `nvidia-smi` only. The
  compute term in §5.2 is arithmetic on the reference's loop structure, not a measurement.
* No real-model noise floor, so the numbers in §4 are *tiny-fixture* numbers. The published-weight
  reference's floor will be larger in absolute terms (bf16 logits, 248k vocab) and must be measured
  with the same command when the loader exists.
* No I-031 metric test under `zig build test` (that is part 2), no change to any kernel tolerance,
  oracle comparison, `core/build.zig`, or committed fixture.
* No edit to `run.sh`'s unpinned `transformers` install, even though it currently cannot bootstrap a
  venv that matches the fixtures (§1).
