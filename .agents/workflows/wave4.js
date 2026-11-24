// Wave 4: I-025 attention perf (deepseek), I-012 elementwise/RoPE/sampler (syn:large),
// reviewer-followups sweep (desktop). Impl -> review -> bounded fix, per lane.
// TWO HARD RULES LEARNED THE HARD WAY - both crashed this driver after a full impl round:
//  1. runs.all() items REQUIRE a unique `key`. Without it the call dies at launch.
//  2. Child results expose ONLY {runId, ok, output}; custom fields (lane_id, cwd, model)
//     do NOT come back. Recover the lane from the key you set, never from the result.
const BASE = '/home/gary/dev/inferno/.agents/worktrees/slot-';
const GATE = BASE + '1/.agents/tasks/review-gate.md';
const ART = '/home/gary/.pi/agent/sessions/--home-gary-dev-inferno--/subagent-artifacts/';
const DEEPSEEK = 'opencode-go/deepseek-v4.1-flash';
const DESKTOP = 'desktop';
const STRONG = 'synthetic/syn:large:text';

const COMMON = 'You are the ONLY writer in your cwd. Never touch /home/gary/dev/inferno (main) or another slot. '
  + 'Do not run git commit. Other lanes run in parallel: stay strictly inside your own files. '
  + 'SHARED FILES: do not edit core/build.zig (it globs src/kernels/*.cu). Copy core/src/hw/kernel.zig from main '
  + 'and do not modify it - it is the only way to load a cubin and launch a kernel. In core/src/c_api.zig add '
  + 'EXACTLY one line to the merged `test { ... }` block: _ = @import("kernels/<yours>.zig"); '
  + 'You may add `pub` to core/src/ref/* internals but NEVER change a reference body or semantics. '
  + 'Kernel tests compare against core/src/ref/ with IDENTICAL inputs; tier-1 tolerance is the OUTPUT format '
  + 'precision (ulp <= 0.5 for bf16), never a blanket 1e-6. '
  + 'Gates: bash tools/gate.sh ; echo EXIT=$? AND cd core && zig build test ; echo EXIT=$? '
  + 'No per-kernel build steps; do not add any. Wrap long runs in timeout. Paste real gate output, never | tail then $?.';

const LANES = [
  { id: 'attnperf', slot: '1', model: DEEPSEEK, spec: '.agents/tasks/issue-025-attention-perf.md',
    extra: ' Priorities in order: (1) MMA prefill path to <=5 us/token/layer from the measured 36.6; '
      + '(2) context-independent score buffer via streaming top-k, asserting the buffer stays flat as context grows; '
      + '(3) per-row selection bitmap so QSA is exact at rows>1 instead of only via union. '
      + 'Item (4), sharding, is explicitly I-014 - do not do it. If the MMA path will not reach 5 us in this pass, '
      + 'land it correct with the measured number and say so; a correct kernel with an honest number beats a target hit.' },
  { id: 'elementwise', slot: '2', model: STRONG, spec: '.agents/tasks/issue-012-elementwise-kernels.md',
    extra: ' RoPE is the exactness-critical part: partial rotary factor 0.25 and mrope_section [11,11,10] with '
      + 'interleaved section assignment must match the reference element-for-element, including the untouched tail. '
      + 'The sampler must reproduce greedy ties and temperature/top-k/top-p ordering exactly - test ties explicitly, '
      + 'do not assume a stable sort.' },
  { id: 'sweep', slot: '3', model: DESKTOP, spec: '.agents/tasks/reviewer-followups.md',
    extra: ' Mechanical only, each with a test or an executed check that fails if reverted. '
      + 'Do NOT take the F8 stability measurement - the GPUs are busy and non-negotiable 4 forbids it; '
      + 'if you reach that item, leave the field null and say so. Do not refactor.' },
];

function implTask(L) {
  return 'Execute the task spec at ' + BASE + L.slot + '/' + L.spec + ' exactly. Read that spec first, then '
    + 'plan.md (non-negotiables) and ' + GATE + '. ' + COMMON + ' ' + L.extra
    + ' HARD TIME DISCIPLINE: if a measurement is proving hard, STOP measuring, report the number you have as '
    + '"not measured" and finish the code. A green tree with an honest gap beats a perfect report that never lands.';
}

function reviewTask(L, impl) {
  return 'Review the work in your cwd against ' + BASE + L.slot + '/' + L.spec + '. First action: read ' + GATE
    + ' and follow it exactly. Judge the TREE by execution, not the report. Run bash tools/gate.sh; echo EXIT=$? '
    + 'and cd core && zig build test; echo EXIT=$?. Confirm comparisons are against core/src/ref/ with IDENTICAL '
    + 'inputs (never another GPU path), that no ref/ body or committed fixture or tolerance changed, and spot-check '
    + 'at least one assertion for non-vacuity. bash is verification only; do not edit or commit.\n\n'
    + '--- IMPLEMENTER REPORT ---\n' + impl + '\n--- END REPORT ---';
}

const state = {};
for (let i = 0; i < LANES.length; i++) {
  state[LANES[i].id] = { impl: '', review: 'BLOCKING: not yet reviewed', blocking: true };
}
const log = [];
const implOpts = { agent: 'worker', timeoutMs: 4500000 };
const revOpts = { agent: 'gate-reviewer', model: STRONG, timeoutMs: 1800000 };

for (let attempt = 1; attempt <= 3; attempt++) {
  const jobs = [];
  for (let i = 0; i < LANES.length; i++) {
    const L = LANES[i];
    if (!state[L.id].blocking) { continue; }
    const task = attempt === 1 ? implTask(L)
      : ('Fix exactly the BLOCKING findings below, per ' + BASE + L.slot + '/' + L.spec + '. ' + COMMON
         + ' Do not broaden scope.\n\nBLOCKING FINDINGS:\n' + state[L.id].review);
    jobs.push({ key: 'impl-' + L.id + '-p' + attempt, cwd: BASE + L.slot,
                model: L.model, task: task, ...implOpts });
  }
  if (jobs.length === 0) { break; }
  const impls = await runs.all(jobs);
  for (let j = 0; j < jobs.length; j++) {
    // Index the LOCAL jobs array. Results expose only {runId, ok, output}, so the
    // lane is recovered from the key I set, never from a field on the result.
    const parts = jobs[j].key.split('-');
    const laneId = parts[1];
    state[laneId].impl = impls[j].output || '(no output)';
  }
  emit('impl pass ' + attempt + ' done for ' + jobs.length + ' lane(s)');

  const revJobs = [];
  for (let j = 0; j < jobs.length; j++) {
    const laneId = jobs[j].key.split('-')[1];
    const lane = LANES.filter(function (x) { return x.id === laneId; })[0];
    revJobs.push({ key: 'rev-' + laneId + '-p' + attempt, cwd: jobs[j].cwd,
                   task: reviewTask(lane, state[laneId].impl), ...revOpts });
  }
  const revs = await runs.all(revJobs);
  const still = [];
  for (let j = 0; j < revJobs.length; j++) {
    const laneId = revJobs[j].key.split('-')[1];
    state[laneId].review = revs[j].output || '';
    state[laneId].blocking = /VERDICT:\s*BLOCKING/i.test(state[laneId].review);
    if (state[laneId].blocking) { still.push(laneId); }
  }
  log.push('pass ' + attempt + ': blocking=' + (still.length ? still.join(',') : 'none'));
  emit(log[log.length - 1]);
  if (still.length === 0) { break; }
}

const out = { log: log };
for (let i = 0; i < LANES.length; i++) {
  out[LANES[i].id] = { blocking: state[LANES[i].id].blocking, review: state[LANES[i].id].review };
}
return out;
