// Wave 5: I-013 end-to-end tiny (the architecture gate) + I-026 decode attention/indexer.
// Rules: runs.all items need a unique `key`; results expose only {runId, ok, output}.
const BASE = '/home/gary/dev/inferno/.agents/worktrees/slot-';
const GATE = BASE + '1/.agents/tasks/review-gate.md';
const DEEPSEEK = 'opencode-go/deepseek-v4.1-flash';
const STRONG = 'synthetic/syn:large:text';

const COMMON = 'You are the ONLY writer in your cwd. Never touch /home/gary/dev/inferno (main) or another slot. '
  + 'Do not run git commit. Do not edit core/build.zig or core/src/hw/kernel.zig. '
  + 'Never change a core/src/ref/ body or semantics (adding `pub` is allowed). Never weaken a tolerance or a gate. '
  + 'Gates: bash tools/gate.sh ; echo EXIT=$? AND cd core && zig build test ; echo EXIT=$? '
  + 'Wrap long runs in timeout. Paste real output, never | tail then $?. '
  + 'HARD TIME DISCIPLINE: if a measurement or experiment is proving hard, STOP, report the number you have as '
  + '"not measured" with the reason, and finish the deliverable. A green tree with an honest gap beats a perfect '
  + 'report that never lands.';

const LANES = [
  { id: 'e2e', slot: '1', model: DEEPSEEK, spec: '.agents/tasks/issue-013-end-to-end-tiny.md',
    extra: ' This is THE architecture gate: every kernel has passed its own oracle in isolation and the composed '
      + '48-layer stack has never run. Bit-identical greedy token ids against the HF tiny-model trace is the bar. '
      + 'When an id mismatches, bisect per layer against the trace tensors in core/src/ref/ rather than guessing - '
      + 'the first divergence names the bug. Do not adjust any kernel to make ids match; report the divergence point.' },
  { id: 'serial', slot: '2', model: STRONG, spec: '.agents/tasks/issue-026-decode-attention-indexer.md',
    extra: ' Item 1 is the priority: attn_decode at 195.8 us/launch for a QSA-bounded 2048-key set is ~10x too '
      + 'slow and may indicate the bound is not enforced on that path at all. Find the cause before optimising, '
      + 'and report a per-launch breakdown. Item 2 (indexer incremental top-k) must not change the selected set.' },
];

function implTask(L) {
  return 'Execute the task spec at ' + BASE + L.slot + '/' + L.spec + ' exactly. Read that spec first, then '
    + 'plan.md (non-negotiables) and ' + GATE + '. ' + COMMON + ' ' + L.extra;
}
function reviewTask(L, impl) {
  return 'Review the work in your cwd against ' + BASE + L.slot + '/' + L.spec + '. First action: read ' + GATE
    + ' and follow it exactly. Judge the TREE by execution, not the report. Run bash tools/gate.sh; echo EXIT=$? '
    + 'and cd core && zig build test; echo EXIT=$?. Confirm comparisons are against core/src/ref/ with IDENTICAL '
    + 'inputs (never another GPU path), that no ref/ body, committed fixture or tolerance changed, and spot-check '
    + 'at least one perf assertion for non-vacuity. bash is verification only; do not edit or commit.\n\n'
    + '--- IMPLEMENTER REPORT ---\n' + impl + '\n--- END REPORT ---';
}

const state = {};
for (let i = 0; i < LANES.length; i++) { state[LANES[i].id] = { impl: '', review: 'BLOCKING: not yet reviewed', blocking: true }; }
const log = [];
const implOpts = { agent: 'worker', timeoutMs: 5400000 };
const revOpts = { agent: 'gate-reviewer', model: STRONG, timeoutMs: 1800000 };

for (let attempt = 1; attempt <= 3; attempt++) {
  const jobs = [];
  for (let i = 0; i < LANES.length; i++) {
    const L = LANES[i];
    if (!state[L.id].blocking) { continue; }
    const task = attempt === 1 ? implTask(L)
      : ('Fix exactly the BLOCKING findings below, per ' + BASE + L.slot + '/' + L.spec + '. ' + COMMON
         + ' Do not broaden scope.\n\nBLOCKING FINDINGS:\n' + state[L.id].review);
    jobs.push({ key: 'impl-' + L.id + '-p' + attempt, cwd: BASE + L.slot, model: L.model, task: task, ...implOpts });
  }
  if (jobs.length === 0) { break; }
  const impls = await runs.all(jobs);
  for (let j = 0; j < jobs.length; j++) {
    const laneId = jobs[j].key.split('-')[1];
    state[laneId].impl = impls[j].output || '(no output)';
  }
  emit('impl pass ' + attempt + ': ' + jobs.length + ' lane(s)');

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
