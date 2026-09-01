# Autopilot learnings

Rewritten each run from `.superpowers/autopilot/*/findings.jsonl`. Kept short on
purpose: rules are merged rather than appended, and "Recent runs" holds only the
last few runs that produced findings, most recent first.

## Planning rules

**Every file the change falsifies belongs on the plan's file list — the
architecture doc most of all.** This run's headline major: `docs/pipeline.md`
still says the filter stage upsamples every candidate before NMS, labels the
filter-to-NMS edge "full res", describes NMS as a loop over full-resolution
coverage, and marks the issue open — after a branch that made all four false.
The file structure table never listed it, so no task touched it, and the repo's
one architecture overview now contradicts the shipped code. The same doc also
carries staleness from two *earlier* waves (NMS is bit-packed popcounts now, not
a byte-wise IoU loop; mask-encode moved into the worker), because no wave has
ever had a step that re-checks it. The smaller version of this bites constantly:
a `session-key.ts` comment enumerating fields excluded from the cache key, not
extended for the field the same brief adds a test about; a `createSegmenter`
comment naming a function the same step deletes; two playground test files
holding `counts` literals that broke the typecheck the moment a member was added
to `SegmentationCounts`. Before finalising the file list, grep the repo for the
names, numbers and diagrams the change invalidates — comments and docs included
— and put each hit on the list or say why it stays.

**Code the plan dictates verbatim has never been run. Schedule the run.** Two
majors in an earlier run came out of literal plan code that typechecked, passed
review and passed its unit tests, then failed on first contact with a real
browser: a `waitForSelector` in the default `visible` state aimed at a `<pre>`
with a genuine 0px box until the first result arrives; and a dynamic `import()`
of `@huggingface/transformers` inside a Web Worker, invisible to Vite's
dependency scanner, so first inference reloaded the page and wiped the state the
runner was polling. jsdom computes no layout and runs no bundler. This run is
the other side of the coin: it is the first in this repo where a `(ui)`
criterion actually executed instead of being skipped, and it passed — real
execution is affordable, and it is the only evidence that counts for browser,
bundler or worker-boot behaviour. Put a real end-to-end invocation in an early
task and budget for it.

**Pin the branch to what it will merge into, not to a stale merge base.** New
failure stage this run (the corpus's first `brief`-stage finding, major): the
worktree stayed at merge base `3cccb78` while `main` advanced to `3fe5e56`,
which rewrote the `DEFAULT_SEGMENTER_OPTIONS` doc comment and `batchSize` in
`types.ts` and replaced `CompareView`'s local choice constants with
`playground/option-choices.ts`. Every green gate the run reports describes a
tree that will never exist on `main`. When a run spans hours next to other
merges, re-check the base before the final review and re-run the gates on the
merge result, not on the branch tip alone.

**Check claims about the repository — including claims about your own edit —
against the repository.** A spec justified skipping browser verification with
"the repo has no `@playwright/test` dependency"; the repo pins it, with a config,
a spec and a CI lane. This run's plan prose said "four `recordFilterSub` and six
`filterSubPhases` occurrences" where the file has five and five, and "the two
lines" of a three-line region — harmless only because the quoted text was
authoritative. Any count, path or capability asserted in a brief is one `grep`
away; run it before writing the number.

**Make every verification step falsifiable, and check the plan's own text cannot
satisfy it.** Recurring in five runs. A plan claimed `npm run typecheck` proved a
property of a file `tsconfig` excludes. A plan verified "no `declare` for
`dedupeMasksReference` leaks into `index.d.ts`" with `grep -q dedupeMasksReference`
— which the JSDoc that same plan mandated matches. A plan told the implementer
`grep -n catch` should show exactly two matches; it shows five, because comments
and a `.catch()` call match too. Confirm the command's scope covers the file,
confirm nothing the plan itself mandates trips it, anchor greps to the construct
(`export`, `declare`, `catch (`), and run any count you intend to assert.

**Plan the failure path, and never let a batch runner lose completed work.**
Same shape four times: `chromium.launch()` outside the `try/finally` that closes
the server; `die()` calling `process.exit(1)` from inside that `try`, skipping
the `finally`; a pre-dispatched next-batch promise with no rejection handler,
firing an `unhandledrejection` in exactly the error case the design meant to keep
quiet; and an unguarded `await runOneRow(...)` that would have discarded every
already-measured row on one hard failure. For each resource the plan's code
acquires, name where it is released when the *next* step throws; wrap per-item
calls and record failures as results; validate up front that the inputs the run
will feed the UI are inputs the UI accepts (a trailing `--reps` with no value
yields `Number(undefined)`). And check what the failure path *says*: a page with
no `navigator.gpu` renders a different panel, so the wait died with a generic 30s
timeout instead of the intended no-adapter message.

**Compute every number in prose, and put the caveat in the artifact people
open.** A commit message called four decode paths "within roughly 2% of each
other" when the group spanned 4.0%; the fix claimed a floor of "about 2.0%"
against a true 1.46%. The measurements were right both times — only the hedge
was invented. Same rule as not putting an unmeasured magnitude in a mandated doc
comment ("roughly doubles the nms stage" against a measured ~124x). Disclosures
belong in the committed `docs/measurements/*.md` a reader opens, not in a commit
message, and the plan should name that file. One caveat learned this run: a
blanket "every figure derived from the committed artifacts" rule leaves no room
for an openly illustrative calculation (12.2 MB per candidate on a 12 MP photo,
arithmetic off a source comment) — allow it explicitly and label it, or the rule
gets quietly broken instead of followed.

**Point every test at the code that owns the invariant, and choose constants
that can fail.** A test built its variant as `baseline.map(m => ({...m}))` — four
identical IoU-1.0 pairs — so "takes the lower median" passes under the upper
median or the average. A seeded differential fixed `CANVAS_W` at 64, a multiple
of the 32-bit word, so the bit-packing test never produced the tail or straddling
word the exactness claim rests on. A real-GPU pixel spec centred a 32x24 rect in
a 64x48 image, so a uniform flip of the composite pass moves every sampled layer
together and both assertions still pass. And an assertion that derives its
expected row count from the DOM passes vacuously if the fixture list ever
empties. Pick constants off the boundary, break the symmetry the bug would hide
in, say in the plan which case each constant exists to hit, and guard any
count-derived assertion with a non-empty check. Know too which assertion does the
failing: a spec credited a bbox check with catching geometry errors, but the bbox
delta was 1px on healthy code *and* under an injected half-pixel regression,
while IoU fell to 0.92-0.95 against a 0.99 gate. Inject the failure and record
which assertion moves.

**Name the convention behind every number, and give real-clock invariants a
tolerance.** "Median" over an even count; a counter counting all iterations
beside counters counting only non-empty ones; sub-timers leaving a residual
outside the parent's instrumented region — each is defensible and each misleads.
New this run: an exactness invariant (sub-steps sum to the filter total) that is
exact in a synthetic unit test left a 0.1 ms residual on 3 of 4 real GPU rows,
because the worker takes its own `performance.now()` for the parent record and
Chrome quantises that clock to 0.1 ms. When an AC asserts an exact identity,
say whether it binds the unit test only, and give the real-data form a tolerance.

**Keep an acceptance criterion's rationale no wider than what it binds, and check
a gate in both directions.** An AC bound `timings.record('nms')` to the fast
path, which holds, but its parenthetical "the results table is never inflated"
does not — `totalMs` is wall clock. Another asserted two sub-regions tile the
filter total exactly, when the inner `recordSub` fires only on a non-empty batch.
This run shipped a comment and an analysis calling a low-res area gate "1.60x
stricter" one-directionally, when a candidate whose logits cross zero in the
letterbox pad gets area credit that never reaches the output, making the gate
*looser* there. Also avoid freezing measured figures into ACs: a precision fix
was once parked purely because it would shift numbers quoted in AC1. Put gates
and tolerances in criteria, not figures a benign change perturbs.

**Do not mandate incidentals; do mandate the assertion.** Plans are followed
literally, so an instruction to emit the same tree glyph on every row produces
exactly that, a required `data-testid` with no matching assertion step produces a
hook nothing uses, and two new table cells requested with only a
control-passthrough test get verified by code reading. Generated-output details
count: a hardcoded column index for a markdown alignment row misaligns the table
when a column is added, and an unescaped failure message interpolated into a
markdown cell destroys the row on a pipe. Related and recurring three runs
running: **render a recorded run from the recorded run** — dtype, batch size,
points per side, the grid line in an export must come from the options captured
with the result, never from live control state, and a failed re-run must clear or
mark the previous result.

**When the plan wraps or rewrites existing code, name what the old path loses.**
Wrapping the whole NMS region in `if (plan)` dropped a progress event a
zero-batch run always used to post — plan-mandated, inert at the default
`pointsPerSide`, invisible in review until self-disclosed. Worse, a verbatim
replacement for `createSegmenter.test.ts` silently dropped
`expect(filterSubPhases.threshold.count).toBe(0)`, the only coverage that the
rebuild path zero-fills an unrecorded sub-step, costing a review round. Prefer
targeted edits. Where a full replacement or a new conditional is warranted, list
what exists today and mark each item kept, renamed or deliberately dropped.

**Give every dependency and every consumer its own step, and keep the public
surface narrow.** A test importing `node:zlib` with no `@types/node`; a Playwright
harness with no `.gitignore` for `test-results/`; a worker-only dynamic import
with no `optimizeDeps.include`. This run's variant: `core/index.ts` grew
`export * from './mask-pipeline'`, publishing ~12 worker-lifecycle names carrying
a mutation-and-ordering contract meaningful only inside the worker loop, solely so
one playground file could import from `../src/segmenter` — a deep import would
have kept it narrow. Name the exact import path the new consumer will use.
Likewise bound large buffers in the plan: full-resolution `RawMask[]` per
comparison row holds hundreds of megabytes for a consumer that reduces them to
ten numbers.

**Say what the mandated code does at its edges, and require the test.** A
docstring promised a NaN threshold returns an empty mask rather than throwing,
with no test; a `resolvePadSize` double-cast reached `Partial<PadSize>` with no
`typeof` check; a `width` documented as required but not as *positive* silently
returned a wrong kept set (32 mismatches in 1000 trials at width -8). If an edge
case corrupts results instead of throwing, the doc must say so and a test must
pin it. Test-harness config is an edge too: `webServer.reuseExistingServer:!CI`
serves a stale local dev server to a local run — standard idiom, worth stating
rather than discovering.

**Meta:** the dominant failure mode is unchanged — plans over-prescriptive about
incidentals and under-specific about prerequisites and evidence. This run
sharpened two edges of it. First, a plan's blast radius is wider than its diff:
the docs, comments and sibling test literals a change falsifies are part of the
change, and the architecture overview is the one nobody schedules. Second,
reviewing plan code is not evidence about plan code — but real execution is now
demonstrably cheap here, so spend the budget on running it and on pinning the
branch to the tree it will actually merge into.

## Recent runs

**issue-6-5-8-f1-n1-f3-carry-256x256-through-filter** (2026-08-31) — filter and
dedupe at 256x256 with resampling only for survivors, behind `lowResFilterNms`; a
playground Boundary tab with a three-engine check; Compare/sweep exposure; and a
real-GPU measurement writeup. Tier large, 5 tasks, 0 parked, 0 fix rounds; verify
ran and passed 1/1 `(ui)` criteria — the first run in this repo where a `(ui)`
criterion executed rather than skipped. 16 findings, 12 plan-stage. Two majors:
`docs/pipeline.md` left describing the pre-change pipeline (and carrying
staleness from two earlier waves nobody re-checks), and the branch verified
against a merge base `main` had moved past — the corpus's first `brief`-stage
finding. The minors cluster tightly into one theme: the plan's own text against
its own change — an incomplete file list that broke the playground typecheck,
prose miscounting occurrences, a comment ordered preserved that the same step
falsifies, a doc comment not extended, a conditional that drops a zero-batch
progress event. Plus a `export *` that widened the public API to serve one
playground import, and a timing identity exact in the unit test but 0.1 ms short
on real GPU rows thanks to `performance.now()` quantisation.

**issue-16-decode-sweep** (2026-08-28) — `overlapDecodeFilter` /
`gpuResidentEmbeddings` / `keepRawMasks` behind flags, a playground Compare tab,
and a headed real-GPU Playwright sweep writing `docs/measurements/`. Tier large,
5 tasks, 1 fix round across 2 tasks; verify skipped (no ui criteria); final
whole-branch review produced a 2-major fix wave and parked 9 minors. 18 findings,
12 plan-stage. The two majors are the run's lesson: plan-mandated Playwright code
that failed only under real execution (a `visible` wait on a 0px-tall empty
`<pre>`; Vite pre-bundling a worker-only dynamic import mid-run and reloading the
page), plus an unguarded per-row `await` that would have discarded a completed
sweep on any single-row failure. The rest: cleanup skipped on early-exit paths
(3), imprecise summary statistics in prose (2), and repeats — stale control state
in an export, a vacuous fixture, an unfalsifiable `grep` count. Outcome: no decode
path is a clear win; the fp16/batch-32 group spans 1.46%.

**issue-8-7-8-f2-single-pass-resample** (2026-08-28) — fused single-pass
threshold+resample for masks. Tier standard, 2 tasks, 2 parked, 0 fix rounds;
verify skipped. 12 findings, unusually spec-heavy: an Important one where the spec
asserted the repo had no `@playwright/test` dependency (it does, with a CI lane)
to justify skipping browser verification, a spec claim that swapped which check
catches geometry errors (bbox delta stayed 1px, IoU fell to 0.92-0.95), an AC that
overstated exact tiling, and a fix wave that introduced a fresh unattributed
figure while repairing the first one. Plan-stage: a mandated test rewrite that
silently dropped the only assertion covering the rebuild path's zero-fill.

**issue-4-3-8-n2-n3-bbox-prefilter-and-bit-packed** (2026-08-27) — bbox prefilter
plus bit-packed coverage in `dedupeMasks`, with an opt-in `compareNms` A/B against
a preserved `dedupeMasksReference`. Tier standard, 2 tasks, 0 parked, 2 review
rounds, 0 fix rounds; verify skipped. 6 findings: four plan-stage (a `grep` step
satisfied by the plan's own mandated JSDoc; a `CANVAS_W` of 64 hiding the
tail-word case; a doc comment claiming "roughly doubles" against a measured ~124x;
a documented `width` with no positivity precondition) and two spec-stage (an AC
parenthetical broader than the criterion; the first `OVERRULED` verdict, review
objecting to duplication the spec deliberately requires).
