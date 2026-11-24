// Wave 3b reviews. Impl rounds already landed in the worktrees; this script reviews
// them and runs bounded fix rounds for any BLOCKING verdict.
// NOTE: child results expose only {runId, ok, output, ...} — custom fields are NOT
// returned, so every lookup here indexes the LOCAL jobs array, never the result.
const BASE = '/home/gary/dev/inferno/.agents/worktrees/slot-';
const GATE = BASE + '1/.agents/tasks/review-gate.md';
const ART = '/home/gary/.pi/agent/sessions/--home-gary-dev-inferno--/subagent-artifacts/';
const DEEPSEEK = 'opencode-go/deepseek-v4.1-flash';
const DESKTOP = 'desktop';
const STRONG = 'synthetic/syn:large:text';

const LANES = [
  { id: 'moe', slot: '1', fixModel: DEEPSEEK, spec: '.agents/tasks/issue-010-moe-kernels.md',
    report: ART + '8744059b-9e3a-487e-88d0-3e3c3d49f28c_worker_output.md',
    note: 'Grouped NVFP4 GEMM across 512 experts, top-10. Router tie-breaking must match torch.topk.' },
  { id: 'cleanup', slot: '2', fixModel: DESKTOP, spec: '.agents/tasks/issue-024-cleanup.md',
    report: ART + 'bcb39dce-90f3-4acf-942e-362bf64cd2fd_worker_output.md',
    note: 'Eight mechanical findings. F8 is mechanism-only with a null value (measurement deferred); P2-3 belongs in server/tests, not tools/gate.sh.' },
  { id: 'ple', slot: '3', fixModel: STRONG, spec: '.agents/tasks/issue-011-ple-kernels.md',
    report: ART + 'e679a9b7-53b6-4d51-9586-ce08640bcc52_worker_output.md',
    note: 'The implementer TIMED OUT mid-report: the tree is green (orchestrator ran bash tools/gate.sh EXIT=0 and zig build test EXIT=0) but its report is partial and it burned the last hour chasing mincore cache-residency measurements. Judge the TREE, not the report. Do not require cache-residency numbers; p50/p99 gather latency is enough, and "not measured" stated plainly is acceptable.' },
  { id: 'gdnfix', slot: '4', fixModel: STRONG, spec: '.agents/tasks/issue-009-gdn-kernels.md',
    report: ART + '692c1b7b-8817-4bf6-8959-5569254cef9c_worker_output.md',
    note: 'This lane is a single blocking fix (OOB guard in the triangular solve) plus its padded-chunk test. Everything else in I-009 was already reviewed and green - do NOT re-open it, and do NOT fail it for pre-existing tiger-style line-length or test-count drift, which is tracked separately.' },
];

function reviewTask(L) {
  return 'Review the work in your cwd against the spec at ' + BASE + L.slot + '/' + L.spec + '. '
    + 'First action: read ' + GATE + ' and follow it exactly. '
    + 'The implementer report is at ' + L.report + ' — read it if useful, but judge the TREE by execution. '
    + 'Context: ' + L.note + '\n\n'
    + 'Verify by execution: run `bash tools/gate.sh; echo EXIT=$?` and `cd core && zig build test; echo EXIT=$?`; '
    + 'confirm the kernel was compared against core/src/ref/ with IDENTICAL inputs (never against another GPU path); '
    + 'confirm no core/src/ref/ body was changed (adding `pub` is allowed) and no committed fixture or tolerance was altered; '
    + 'spot-check at least one assertion for non-vacuity; check that timing gates assert rather than merely print. '
    + 'Tier-1 numeric tolerance is the OUTPUT format precision (ulp <= 0.5 for bf16), never a blanket 1e-6. '
    + 'bash is verification only; do not edit or commit.';
}

function fixTask(L, findings) {
  return 'Fix exactly the BLOCKING findings below, per the spec at ' + BASE + L.slot + '/' + L.spec + '. '
    + 'You are the ONLY writer in your cwd; do not run git commit. Do not edit core/build.zig. '
    + 'Do not broaden scope beyond the findings. Re-run the full gate and paste real output.\n\n'
    + 'HARD TIME DISCIPLINE: this lane may have limited wall clock. If a measurement is proving hard, '
    + 'STOP measuring, report the number you have as "not measured" and finish the code. '
    + 'A green tree with an honest gap beats a perfect report that never lands.\n\n'
    + 'BLOCKING FINDINGS:\n' + findings;
}

const state = {};
for (let i = 0; i < LANES.length; i++) { state[LANES[i].id] = { review: 'BLOCKING: not yet reviewed', blocking: true }; }

const revOpts = { agent: 'gate-reviewer', model: STRONG, timeoutMs: 1800000 };
const fixOpts = { agent: 'worker', timeoutMs: 3300000 };
const log = [];

for (let attempt = 1; attempt <= 3; attempt++) {
  const jobs = [];
  for (let i = 0; i < LANES.length; i++) {
    const L = LANES[i];
    if (!state[L.id].blocking) { continue; }
    jobs.push({ key: 'rev-' + L.id + '-p' + attempt, cwd: BASE + L.slot, task: reviewTask(L), ...revOpts });
  }
  if (jobs.length === 0) { break; }
  const revs = await runs.all(jobs);
  const still = [];
  for (let j = 0; j < jobs.length; j++) {
    const L = LANES.filter(function (x) { return 'rev-' + x.id + '-p' + attempt === jobs[j].key; })[0];
    state[L.id].review = revs[j].output || '';
    state[L.id].blocking = /VERDICT:\s*BLOCKING/i.test(state[L.id].review);
    if (state[L.id].blocking) { still.push(L.id); }
  }
  log.push('review pass ' + attempt + ': blocking=' + (still.length ? still.join(',') : 'none'));
  emit(log[log.length - 1]);
  if (still.length === 0) { break; }
  if (attempt === 3) { break; }

  const fixes = [];
  for (let i = 0; i < LANES.length; i++) {
    const L = LANES[i];
    if (!state[L.id].blocking) { continue; }
    fixes.push({ key: 'fix-' + L.id + '-p' + attempt, cwd: BASE + L.slot,
                 model: L.fixModel, task: fixTask(L, state[L.id].review), ...fixOpts });
  }
  if (fixes.length === 0) { break; }
  await runs.all(fixes);
  emit('fix pass ' + attempt + ' done for ' + fixes.length + ' lane(s)');
}

const out = { log: log };
for (let i = 0; i < LANES.length; i++) {
  out[LANES[i].id] = { blocking: state[LANES[i].id].blocking, review: state[LANES[i].id].review };
}
return out;
