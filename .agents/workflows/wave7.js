// Wave 7: I-013 determinism hunt + I-014 F1 race. Generated from TEMPLATE.js.
// Do not derive a new wave by editing the previous wave's script: that is how
// DESKTOP went missing from wave6 and how three other drivers failed tonight.
//
// RULES, each learned from a failure that cost a full round:
//  1. runs.all() items REQUIRE a unique `key`. Missing -> dies at launch.
//  2. Child results expose ONLY {runId, ok, output}. Custom fields do not come
//     back, so recover the lane from the key you set, never from the result.
//  3. `action:"validate"` checks structure WITHOUT executing. It will not catch
//     an undefined identifier. Define every model const below, even if unused.
//  4. A workflow driver has no access to the repo; it only launches children.
//  5. A caller must treat a FAILED/TIMED-OUT review as BLOCKING. A dead reviewer emits no
//     verdict, and "no verdict" is not "pass" - wave 5 shipped that mistake.
//  6. Never let a lane take a perf number while another lane holds a GPU. Tell
//     it to report "unmeasured, blocked on an idle box" instead.
const BASE = '/home/gary/dev/inferno/.agents/worktrees/slot-';
const GATE = BASE + '1/.agents/tasks/review-gate.md';
const ART = '/home/gary/.pi/agent/sessions/--home-gary-dev-inferno--/subagent-artifacts/';

// Every model the routing table may use. Define all of them, always.
const DEEPSEEK = 'opencode-go/deepseek-v4.1-flash'; // hard work, ONE slot only
const DESKTOP = 'desktop';                          // OFFLINE 2026-11-24: local engine on
// the same GPUs; do NOT dispatch. Kept defined so old scripts do not break.
const STRONG = 'synthetic/syn:large:text';          // general; ALWAYS the reviewer

const COMMON = 'You are the ONLY writer in your cwd. Never touch /home/gary/dev/inferno (main) '
  + 'or another slot. Do not run git commit. Do not edit core/build.zig or core/src/hw/kernel.zig. '
  + 'Never change a core/src/ref/ body or semantics (adding `pub` is allowed). Never weaken a '
  + 'tolerance or a gate. Gates: bash tools/gate.sh ; echo EXIT=$? AND cd core && zig build test ; '
  + 'echo EXIT=$? -- wrap long runs in timeout, paste real output, never | tail then $?. '
  + 'HARD TIME DISCIPLINE: if a measurement is proving hard, STOP, report the number you have as '
  + '"not measured" with the reason, and finish the deliverable. A green tree with an honest gap '
  + 'beats a perfect report that never lands.';

// ---- EDIT ONLY BELOW THIS LINE ----
const LANES = [
  { id: 'det', slot: '1', model: DEEPSEEK, spec: '.agents/tasks/issue-013-end-to-end-tiny.md',
    extra: ' The residual nondeterminism is the whole job. Order, cheapest first: '
      + '(1) run the e2e test with CUDA_LAUNCH_BLOCKING=1 - one env var, no code change; if the '
      + 'spread vanishes the bug is host async ordering, if it persists it is memory or an '
      + 'intra-kernel race. (2) compute-sanitizer --tool racecheck, then memcheck - racecheck has '
      + 'never been run on this path and names the kernel and line. (3) if ordering, add a '
      + 'cudaStreamSynchronize after every launch behind an env flag, confirm determinism, then '
      + 're-add syncs one at a time to localise the missing dependency. Do NOT guess. '
      + 'Flake rate is ~12%, so verify the final config with >=20 runs ONCE and report every '
      + 'distinct value. Do not widen the ulp ceiling (16) to absorb the spread - the spread is '
      + 'the bug. Do not add a blanket cudaDeviceSynchronize and call it done.' },
  { id: 'f1', slot: '4', model: STRONG, spec: '.agents/tasks/issue-014-multigpu.md',
    extra: ' Fix exactly F1 and the four accepted P2s from the last review. F1: prefetch orders '
      + 'the CPU refill of bounce[i] against the previous copy that read that slot, but nothing '
      + 'orders it against the WRITE BACK enqueued later in the same batch - a host slot written '
      + 'back by move N-1 of batch N can be read by the CPU at the top of batch N+1. Re-record '
      + 'every bounce_free after ready.record and pin it with a test that fails on current code. '
      + 'Also: make the comms report write fatal; rename within_bound to liveness_ok beside '
      + 'liveness_bound_ms with an explicit not_measured field; correct the "1,645 cold slabs per '
      + 'token => ~0.6 s/token" claim to <=120 slab fetches per rank per token (top_k_total=10). '
      + 'Do not attempt item 2 - it needs model/exec.zig which is not in main. No perf claims.' },
];

function implTask(L) {
  return 'Execute the task spec at ' + BASE + L.slot + '/' + L.spec + ' exactly. Read that spec first, '
    + 'then plan.md (non-negotiables) and ' + GATE + '. ' + COMMON + ' ' + L.extra;
}
function reviewTask(L, impl) {
  return 'Review the work in your cwd against ' + BASE + L.slot + '/' + L.spec + '. First action: read '
    + GATE + ' and follow it exactly. Judge the TREE by execution, not the report. Run '
    + 'bash tools/gate.sh; echo EXIT=$? and cd core && zig build test; echo EXIT=$?. Confirm comparisons '
    + 'are against core/src/ref/ with IDENTICAL inputs (never another GPU path), that no ref/ body, '
    + 'committed fixture or tolerance changed, and that at least one perf assertion is non-vacuous. '
    + 'bash is verification only; do not edit or commit.\n\n'
    + '--- IMPLEMENTER REPORT ---\n' + impl + '\n--- END REPORT ---';
}

const state = {};
for (let i = 0; i < LANES.length; i++) {
  state[LANES[i].id] = { impl: '', review: 'BLOCKING: not yet reviewed', blocking: true };
}
const log = [];
const implOpts = { agent: 'worker', timeoutMs: 5400000 };
const revOpts = { agent: 'gate-reviewer', model: STRONG, timeoutMs: 1800000 };

for (let attempt = 1; attempt <= 3; attempt++) {
  const jobs = [];
  for (let i = 0; i < LANES.length; i++) {
    const L = LANES[i];
    if (!state[L.id].blocking) { continue; }
    const task = attempt === 1 ? implTask(L)
      : ('Fix exactly the BLOCKING findings below, per ' + BASE + L.slot + '/' + L.spec + '. '
         + COMMON + ' Do not broaden scope.\n\nBLOCKING FINDINGS:\n' + state[L.id].review);
    jobs.push({ key: 'impl-' + L.id + '-p' + attempt, cwd: BASE + L.slot, model: L.model,
                task: task, ...implOpts });
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
    // A review that FAILED OR TIMED OUT has no verdict line. Treating silence as approval
    // let a broken tree through (wave 5 reported "blocking=none" for a reviewer that died).
    const revFailed = revs[j].ok === false;
    state[laneId].blocking = revFailed || /VERDICT:\s*BLOCKING/i.test(state[laneId].review);
    if (revFailed) { emit('REVIEW FAILED for ' + laneId + ' - treating as BLOCKING'); }
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
