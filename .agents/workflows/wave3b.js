// Wave 3b: I-010 MoE (deepseek), I-024 cleanup (desktop), I-011 PLE, GDN OOB fix.
const BASE = '/home/gary/dev/inferno/.agents/worktrees/slot-';
const GATE = BASE + '1/.agents/tasks/review-gate.md';
const DEEPSEEK = 'opencode-go/deepseek-v4.1-flash';
const DESKTOP = 'desktop';
const STRONG = 'synthetic/syn:large:text';

function brief(slot, spec, extra) {
  return 'Execute the task spec at ' + BASE + slot + '/' + spec + ' exactly. '
    + 'First actions in order: read that spec, then .agents/tasks/reviewer-followups.md, '
    + 'then .agents/design/mma-sm120-nvfp4.md, then plan.md. '
    + 'You are the ONLY writer in your cwd. Never touch /home/gary/dev/inferno (main) or another slot. '
    + 'Do not run git commit. Other lanes are running in parallel: stay strictly inside your own files. '
    + 'SHARED-FILE RULES: core/build.zig is owned by the orchestrator - do NOT edit it (it already globs '
    + 'src/kernels/*.cu). Copy core/src/hw/kernel.zig from main into your worktree and do NOT modify it; it is '
    + 'the only way to load a cubin and launch a kernel. In core/src/c_api.zig add EXACTLY two lines in the '
    + 'test block: _ = @import("hw/kernel.zig"); and _ = @import("kernels/<yours>.zig"); '
    + 'You may add `pub` to core/src/ref/* internals but must NEVER change a reference body or semantics. '
    + 'Kernel tests compare against core/src/ref/ with IDENTICAL inputs, and tier-1 tolerance is the OUTPUT '
    + 'format precision (ulp <= 0.5 for bf16), never a blanket 1e-6. '
    + 'Gates are: bash tools/gate.sh ; echo EXIT=$? AND cd core && zig build test ; echo EXIT=$? '
    + 'There are no per-kernel build steps and you must not add any. Wrap long runs in timeout. '
    + 'Write code first, compile early, iterate on real errors, paste real gate output (never | tail then $?).'
    + (extra || '');
}

function reviewText(spec, impl) {
  return 'Review the work described by ' + spec + ' in your cwd. First action: read ' + GATE
    + ' and follow it exactly. Verify by EXECUTION: run the gates, confirm the kernel was compared '
    + 'against core/src/ref/ with identical inputs (not against another GPU path), confirm no oracle '
    + 'file or committed fixture was edited and no tolerance widened, and spot-check at least one '
    + 'assertion for non-vacuity. bash is verification only; do not edit or commit.\n\n'
    + '--- IMPLEMENTER REPORT ---\n' + impl + '\n--- END REPORT ---';
}

const LANES = [
  { id: 'moe', slot: '1', model: DEEPSEEK, spec: '.agents/tasks/issue-010-moe-kernels.md',
    extra: ' This is the hardest remaining kernel: grouped NVFP4 GEMM across 512 experts with top-10 routing. Router top-k tie-breaking must match torch.topk exactly - validate on synthetic ties, do not assume.' },
  { id: 'cleanup', slot: '2', model: DESKTOP, spec: '.agents/tasks/issue-024-cleanup.md',
    extra: ' This lane is deliberately scoped to small mechanical fixes with a test pinning each one. If a finding is ambiguous, report it instead of guessing. Do not refactor.' },
  { id: 'ple', slot: '3', model: STRONG, spec: '.agents/tasks/issue-011-ple-kernels.md',
    extra: ' The n-gram id derivation must be exact (int64 multiply/xor/remainder, EOS-aware segments). The 51.2 GB gather latency is the number I care most about - measure p50/p99 per token.' },
  { id: 'gdnfix', slot: '4', model: STRONG, spec: '.agents/tasks/issue-009-gdn-kernels.md',
    extra: ' This lane is a SINGLE blocking fix from the final gate review, not a re-run of I-009. Fix exactly this: core/src/kernels/gdn.cu around line 506, in the triangular-solve dot-product loop, `key[key_row + dim]` is read unguarded for rows past `tokens`. Guard with `row_valid = start + row < tokens`, nest `other_valid` inside it, so the header comment at gdn.cu:409-411 about zero padding becomes true. Add a test that fails without the guard (run the prefill kernel with a chunk where the last rows are padding and assert no read past the tensor, e.g. via compute-sanitizer memcheck). Then re-run the full gate. Everything else in the lane is already reviewed and green - do not re-open it.' },
];

const rounds = [];
const state = {};
for (let i = 0; i < LANES.length; i++) {
  state[LANES[i].id] = { impl: '', review: '', blocking: true };
}

const implOpts = { agent: 'worker', timeoutMs: 5400000 };
const revOpts = { agent: 'gate-reviewer', model: STRONG, timeoutMs: 1800000 };

for (let attempt = 1; attempt <= 3; attempt++) {
  const jobs = [];
  for (let i = 0; i < LANES.length; i++) {
    const L = LANES[i];
    if (!state[L.id].blocking) { continue; }
    const task = attempt === 1
      ? brief(L.slot, L.spec, L.extra)
      : ('Fix exactly the BLOCKING findings below, per the spec at ' + BASE + L.slot + '/' + L.spec
          + '. You are the ONLY writer in your cwd; do not run git commit. Re-run the full gate and paste '
          + 'real output.\n\nBLOCKING FINDINGS:\n' + state[L.id].review);
    jobs.push({ key: 'impl-' + L.id + '-r' + attempt, lane_id: L.id, model: L.model,
                cwd: BASE + L.slot, task: task, ...implOpts });
  }
  if (jobs.length === 0) { break; }
  const done = await runs.all(jobs);
  for (let j = 0; j < done.length; j++) {
    state[done[j].lane_id] = { impl: done[j].output || '(no output)',
                               review: state[done[j].lane_id].review, blocking: true };
  }
  emit('wave3b impl round ' + attempt + ' done for ' + jobs.length + ' lane(s)');

  const revJobs = [];
  for (let j = 0; j < jobs.length; j++) {
    const L = LANES.filter(function (x) { return x.id === jobs[j].lane_id; })[0];
    revJobs.push({ key: 'rev-' + L.id + '-r' + attempt, lane_id: L.id, cwd: BASE + L.slot,
                   task: reviewText(L.spec, state[L.id].impl), ...revOpts });
  }
  const revs = await runs.all(revJobs);
  const still = [];
  for (let j = 0; j < revs.length; j++) {
    const id = revs[j].lane_id;
    state[id].review = revs[j].output || '';
    state[id].blocking = /VERDICT:\s*BLOCKING/i.test(state[id].review);
    if (state[id].blocking) { still.push(id); }
  }
  rounds.push('round ' + attempt + ': blocking=' + (still.length ? still.join(',') : 'none'));
  emit('wave3b ' + rounds[rounds.length - 1]);
  if (still.length === 0) { break; }
}

const out = { rounds: rounds };
for (let i = 0; i < LANES.length; i++) {
  out[LANES[i].id] = { blocking: state[LANES[i].id].blocking,
                       impl: state[LANES[i].id].impl,
                       review: state[LANES[i].id].review };
}
return out;
