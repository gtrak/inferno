# I-01 — build skeleton: Cargo workspace + Zig static lib + nvcc pipeline + FFI smoke

## Working agreement

You are the sole writer in your worktree. The repo lives at the main checkout;
your `cwd` is the assigned worktree slot. Global truth:

- `.agents/skills/tiger-style/{SKILL,SAFETY,PERFORMANCE,DX,CHECKLIST}.md` — ALL
  Zig code must comply. Read SAFETY.md and DX.md before writing Zig. Minimum:
  ≥2 assertions per function, no recursion, no dynamic allocation after init,
  functions ≤70 lines, 80-column limit, explicit integer widths (`u32`, never
  `usize` in hot-path signatures), every `if` has an `else`.
- `plan.md` §4/§5 — architecture and directory layout. Follow it exactly.
- Environment: `zig` 0.14.1 on PATH, `cargo`/`rustc` 1.100, CUDA 13.2 at
  `/usr/local/cuda-13.2` (`nvcc` exists there but is NOT on PATH), 4 × RTX 5060 Ti.
- No network access is needed and no third-party Rust crates are needed for this
  issue except `serde_json` (already vendored/available via crates.io).

Write first, compile early. Do not research zig build internals for more than a
few tool calls; if a Zig 0.14 API detail is uncertain, write the simplest thing
that compiles and verify with `zig build`.

## What to build

### 1. Repository skeleton (exact paths)

```
Cargo.toml                     workspace: members = ["server"], resolver = "2"
.gitignore                     target/  .zig-cache/  zig-out/  *.cubin  .venv/  bench/*.json
core/build.zig
core/build.zig.zon
core/src/c_api.zig             exported C ABI (see 4)
core/src/hw/cuda_driver.zig    minimal driver-API layer for this issue
core/src/hw/dl.zig             dlopen helper for libcuda.so.1
core/src/kernels/probe.cu      one trivial kernel (see 3)
core/src/kernels/generated_cubins.zig   BUILD-GENERATED, gitignored
server/Cargo.toml              package inferno-server
server/build.rs
server/src/main.rs
tools/gate.sh                  executable
```

### 2. `core/build.zig`

Produce a **static library `libinferno_core.a`** from `src/c_api.zig`:

- target: native x86_64-linux, `linkLibC()`, **position independent**
  (`static_lib.root_module.pic = true`) so it can link into the Rust binary.
- expose `-Doptimize=` passthrough; default `ReleaseSafe`.
- `zig build test` runs unit tests from `src/*.zig`.
- `zig fmt --check src` must be clean (part of the gate).

### 3. nvcc pipeline (must be part of `zig build`)

- A build step compiles every `core/src/kernels/*.cu` with
  `/usr/local/cuda-13.2/bin/nvcc -arch=sm_120a --cubin -O3 -lineinfo -o <out>`
  (honour `CUDA_HOME` if set, else `/usr/local/cuda-13.2`; fail the build with a
  clear error if nvcc is missing).
- Generate `src/kernels/generated_cubins.zig` (via `b.addWriteFiles()` or an
  equivalent, added as an import) containing, for each `.cu` file, its basename
  and `@embedFile`d cubin bytes, e.g.
  `pub const cubins = [_]Cubin{ .{ .name = "probe", .bytes = @embedFile("probe") }, };`
  The static lib must therefore embed the cubins — no runtime file lookups.
- `probe.cu`: `__global__ void probe_fill(float* out, float value)` writing
  `value` to `out[threadIdx.x]`, exported with `extern "C"` to keep the symbol name.

### 4. C ABI (`core/src/c_api.zig`)

Export exactly this surface for this issue (MUST be `export fn`, C ABI, no
Zig-only types in the signature):

```zig
export fn inferno_probe_devices(buf: [*]u8, cap: u32) u32;  // returns bytes written, or 0 on error
export fn inferno_version() [*:0]const u8;                 // "inferno <semver>"
```

`inferno_probe_devices` writes a JSON document into `buf` (at most `cap` bytes,
always NUL-terminated, truncated rather than overflowed) describing every CUDA
device:

```json
{"driver":"610.57","cuda":13020,"devices":[
  {"idx":0,"name":"NVIDIA GeForce RTX 5060 Ti","cc":"12.0","vram_bytes":17104306176}]}
```

Driver access: `dlopen("libcuda.so.1")` + `dlsym` of `cuInit`, `cuDriverGetVersion`,
`cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceGetAttribute`,
`cuDeviceTotalMem_v2`. **Do not link libcuda at build time.** Cache the
`dlsym` results in a static table, check every return code, and map driver
errors to a small closed error set (tiger-style: exhaustive `switch`, no
`catch unreachable` on driver calls).

### 5. `server/`

- `server/build.rs`: run `zig build -Doptimize=ReleaseSafe --prefix <OUT_DIR>/zig`
  inside `../core`, emit `cargo:rustc-link-search=native=<OUT_DIR>/zig/lib`,
  `cargo:rustc-link-lib=static=inferno_core`, `cargo:rerun-if-changed` for
  `core/src/**` and `core/build.zig`.
- `server/src/main.rs`: `--probe` prints the JSON from `inferno_probe_devices`
  (allocate a 64 KiB buffer, UTF-8 validate) and exits 0; `--version` prints
  `inferno_version()`. Default (no args) prints a usage line and exits 2.
- `extern "C"` declarations must match the Zig signatures exactly.

### 6. `tools/gate.sh`

`set -euo pipefail`, run from the repo root, export
`CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}` and
`PATH="$CUDA_HOME/bin:$PATH"`. Steps, each echoing `== step` before running:

1. `zig fmt --check core/src core/build.zig`
2. `(cd core && zig build -Doptimize=ReleaseSafe)` (must also compile cubins)
3. `(cd core && zig build test)`
4. `cargo fmt --all -- --check`
5. `cargo build --workspace --release`
6. `cargo clippy --workspace --all-targets -- -D warnings`
7. `cargo test --workspace`
8. `./target/release/inferno-server --probe` and assert the output contains
   `"devices"` and the count is 4 (parse with `python3 -c`, do not grep-only).

The script must be idempotent and pass from a clean checkout.

## Gates (all must pass, and you must paste the real output in your report)

```
bash tools/gate.sh ; echo EXIT=$?
```

Green means: EXIT=0, and `--probe` reported 4 devices. If a GPU is busy or
`cuInit` fails, that is a FAILURE to report, not to paper over.

## Report format

Implemented / Changed files / Validation (paste gate output tail) / Open
risks / Recommended next step. Note any Zig-0.14-API decision you had to make,
and any skill correction you applied.
