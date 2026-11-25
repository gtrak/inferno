# I-034 — partition the test suite so it uses all four GPUs (and lanes run only what they own)

## Measured problem

`zig build test` is **195.86 s** with guard bands on, and it runs **one binary on device 0** while
GPUs 1–3 sit at 4 MiB for the entire gate. The serialisation in `core/build.zig` was added for a real
defect — two binaries bringing up CUDA contexts **on device 0 concurrently** gave 2-of-5 gate failures
— but it was over-applied: that is a *same-device* hazard, not a reason to leave three cards idle.

Only **two** files need the multi-rank machinery:

| group | content | device need |
|---|---|---|
| **quad** | `model/multi.zig` (19 tests), `mem/placement.zig` | all 4 ranks |
| **single** | `kernels/{attention,elementwise,gemm,gemv,moe,ple,gdn}`, `model/{exec,state,generate,snapshot,runtime_check,tiny_check}`, `hw/{buffer,epochs,kernel,bench}`, `weights/*`, `ref/*` | one device |

Most of the suite is single-device, and it is all pinned to device 0.

## Work

1. **Split the test roots** in `core/build.zig` — one `addTest` per subsystem (at minimum:
   `kernels`, `model`, `hw`, `weights`, `ref`, `multi`). This is orchestrator-owned (`build.zig`), so it
   is done by me, not a lane.
2. **A `tools/gate.sh` step that shards the single-device groups across devices** with
   `CUDA_VISIBLE_DEVICES=N` per process, one process per device. Same-device concurrency stays
   forbidden — that is the flake we actually measured.
3. **The quad group runs alone**, with all four devices, before or after the sharded groups — never
   overlapping them, because it wants the whole machine.
4. **Keep `zig build test` working** as a single command that does all of the above, so lanes and CI
   need no new knowledge. A composite step that sequences the shards is fine.

## Second-order win (the real reason)

Lanes currently eat ~196 s for every gate run, many times per lane. Splitting also lets a lane run
**only its own subsystem's tests** while iterating — `zig build test-kernels` — which removes most of
that wall clock from every lane for the rest of the project. That is worth more than the gate time.

## Guard rails

- **Do not weaken the guard-band default.** Bands stay on in the gate (I-028/I-033); sharding is about
  which device runs what, not about skipping the scan.
- **Test isolation must survive parallel processes.** Any test writing a temp or config file must use a
  process-unique path. This is the same class as the `weights.config.test` collision that made 2 of 20
  runs fail — running in separate processes with separate temp paths should *fix* it, but only if the
  paths are unique.
- **No two processes on the same device**, ever. Pin explicitly; do not rely on a default.
- Prove it: run the sharded gate and the serial gate and confirm **identical pass sets**, then report
  before/after wall time and the per-device utilisation from `nvidia-smi` during the run.

## Gates

```
bash tools/gate.sh ; echo EXIT=$?
(cd core && timeout 1800 zig build test) ; echo CORETEST=$?
INFERNO_GUARD_BANDS=1 bash tools/gate.sh ; echo EXIT=$?
```
