# Autopilot learnings

Rewritten each run from `.superpowers/autopilot/*/findings.jsonl`. Kept short on
purpose: rules are merged rather than appended, and "Recent runs" holds only the
last few runs that produced findings, most recent first.

## Planning rules

**Code the plan dictates verbatim has never been run. Schedule the run.** This
run's two majors both came out of literal plan code that typechecked, passed
review and passed its unit tests, and then failed on first contact with a real
browser: a `waitForSelector` in the default `visible` state aimed at a `<pre>`
that renders empty (genuine 0px box) until the first result arrives, so it timed
out deterministically every invocation; and a dynamic `import()` of
`@huggingface/transformers` inside a Web Worker, invisible to Vite's dependency
scanner, so first inference triggered a full-page reload that wiped the very
state the runner was polling. jsdom tests compute no layout and run no bundler,
so nothing before real execution could have caught either. When a plan mandates
browser automation, bundler config, or worker boot order, put a real end-to-end
invocation in an early task — and budget for it, because that task is where the
schedule actually goes.

**Check claims about the repository against the repository.** A spec justified
skipping browser verification with "the repo has no `@playwright/test`
dependency"; the repo pins `@playwright/test`, has `playwright.config.ts`, a
browser spec and a CI lane running it. The same run's fix wave then wrote an
unattributed, unreproducible IoU figure into the spec while fixing exactly that
defect class. Every factual assertion about the repo or about a measurement is a
`grep`, an `ls` or a run away — do it, and cite where the number came from.

**Plan the failure path's cleanup, not just the happy path's.** Three findings
in this run, one shape: `chromium.launch()` sitting outside the `try/finally`
that closes the server; `die()` calling `process.exit(1)` from inside that
`try`, skipping the `finally` entirely; and a pre-dispatched next-batch promise
with no rejection handler for the duration of the `await` before it, firing an
`unhandledrejection` in precisely the error case the design meant to keep quiet.
For every resource the plan's code acquires, name where it is released when the
step *after* the acquisition throws. And check what the failure path *says*: a
page with no `navigator.gpu` renders a different panel, so the runner's selector
wait died with a generic 30s Playwright timeout instead of the intended
no-adapter message — the abort was right, the diagnostic was useless.

**A batch runner must not lose completed work.** The plan's row loop called
`await runOneRow(...)` unguarded, so a single hard failure — a real hang, a
crashed worker, a config value the UI does not offer — threw out of the loop
before anything was written, discarding every row already measured and
contradicting the plan's own continue-past-a-failed-row criterion. Wrap the
per-item call, record the failure as a result, and validate up front that the
inputs the run will feed the UI are inputs the UI actually accepts. Same for the
CLI edge: a trailing `--reps` with no value yields `Number(undefined)` and a
trailing `--config` silently falls back to the default grid.

**Make every verification step falsifiable — and check the plan's own text
cannot satisfy it, and that the command means what it says.** Recurring in four
runs now. A plan claimed `npm run typecheck` proved a property of a file
`tsconfig` excludes. A plan verified "no `declare` for `dedupeMasksReference`
leaks into `index.d.ts`" with `grep -q dedupeMasksReference`, which the JSDoc
prose the same plan mandated matches. And this run told the implementer `grep -n
catch` should show exactly two matches; it shows five, because comments and a
`.catch()` method call match too. Confirm the command's scope covers the file,
confirm nothing the plan itself mandates trips it, anchor greps to the construct
(`export`, `declare`, `catch (`) rather than a bare word, and if you assert a
match count, run the command before writing the number down.

**Compute the summary statistic; do not eyeball it.** A commit message described
four decode paths as landing "within roughly 2% of each other" when the group
actually spanned 4.0%; the fix that disclosed this then claimed a floor of
"about 2.0%" when the true minimum was 1.46%. The underlying measurements were
correct both times — only the hedge wrapped around them was invented. Any
number in prose, including a hedge, is derived from data the run already has:
derive it. This is the same rule as not putting an unmeasured magnitude in a
mandated doc comment ("roughly doubles the nms stage" against a measured ~124x).

**Put the caveat in the artifact people open, not in the commit message.** The
cold-first-row artifact that inflated one row's model-load and encode times, and
the imprecise spread claim, were disclosed only in a git commit message — while
the committed `docs/measurements/*.md` report, the thing a reader actually
opens, presented the row ordering as a clean result. A caveat that changes how a
number should be read belongs in the same file as the number, and the plan
should say which file that is.

**Point every test at the code that owns the invariant, and choose constants
that can fail.** A test built its variant as `baseline.map(m => ({...m}))` —
four identical IoU-1.0 pairs — so "takes the lower median" passes identically
under the upper median or the average. A seeded differential fixed `CANVAS_W` at
64, a multiple of the 32-bit word, so the bit-packing test never produced the
tail word or the straddling word the exactness claim rests on. A test that
hand-picks inputs and then asserts a property of those inputs proves nothing.
Pick constants off the boundary, make the variant differ where the claim lives,
and say in the plan which case each constant exists to hit. If the invariant
lives somewhere the repo cannot test yet (a worker, a GPU path), say so in the
plan instead of substituting a pure-function test that looks equivalent.
Know too which assertion does the failing: a spec credited a bounding-box
check with catching geometry errors, but measured, bbox delta was 1px on
healthy code *and* 1px under an injected half-pixel regression, while IoU fell
to 0.92-0.95 against the 0.99 gate — the roles were swapped. When a plan claims
a check guards a failure mode, inject that failure and record which assertion
actually moves.

**Render a recorded run from the recorded run.** Third appearance. A row's
labels — dtype, batch size, points per side, the grid line in exported markdown
— must come from the options captured with the result, never from current
control state; this run's markdown export still stamped the default grid onto a
hand run at a different configuration. Staleness runs the other way too: a
re-run that fails must clear or mark the previous result. Say in the plan which
state a re-run invalidates.

**Give every dependency the plan's code implies its own step.** A test importing
`node:zlib` with no `@types/node`; a Playwright harness with no `.gitignore`
entry for `test-results/`; a worker-only dynamic import with no
`optimizeDeps.include`. When a task adds a package, imports from a new runtime
surface, or adds a tool that writes files, the same task adds the manifest
entry, the bundler hint and the ignore rule.

**Name the convention behind every number, and test its ambiguous case.**
"Median" over an even count, a counter that counts all iterations beside
counters that count only non-empty ones, sub-timers that leave a residual
outside the parent's instrumented region — each is defensible and each misleads
a reader assuming the obvious meaning. Decide it in the plan, require a comment
stating it, and require a test where the definitions diverge.

**Keep an acceptance criterion's rationale no wider than what it binds — and
justify deliberate duplication where the reader will meet it.** An AC bound
`timings.record('nms')` to the fast path, which holds, but its parenthetical
"the results table is never inflated" does not: `totalMs` is wall clock and
still includes the reference run. Another AC asserted two sub-regions tile the
filter total exactly, when the inner `recordSub` fires only on a non-empty
batch. Write the parenthetical about what the criterion actually checks. Where a
spec requires duplication on purpose — a reference implementation copied
verbatim *because* it is the differential baseline — say so at the duplication,
or review will flag it every time. The converse bites as well: an AC that quotes
exact measured values freezes the implementation behind it — a precision fix
was once parked purely because it would shift numbers already quoted in AC1.
Put gates and tolerances in acceptance criteria, not figures a benign change
perturbs.

**Do not mandate incidentals; do mandate the assertion.** Plans are followed
literally, so an instruction to emit the same tree glyph on every row produces
exactly that, and a required `data-testid` with no matching assertion step
produces a hook nothing uses. Small generated-output details count here too: a
hardcoded column index for a markdown alignment row silently misaligns the table
when a column is added, and a failure message interpolated unescaped into a
markdown cell destroys the row if it contains a pipe. Specify presentation only
where it carries meaning, and never add a test-only affordance without the test
that reads it.

**Bound the lifetime of large buffers in the plan.** Holding full-resolution
`RawMask[]` for every row of a comparison table keeps hundreds of megabytes
alive for the session when the only consumer reduces them to a ten-number
summary. State when they are released; prefer caching the derived summary and
dropping the source.

**Do not mandate a wholesale rewrite of an existing test file.** The plan's
verbatim replacement for `createSegmenter.test.ts` silently dropped
`expect(filterSubPhases.threshold.count).toBe(0)` — the only coverage that the
rebuild path zero-fills an unrecorded sub-step — and it took a review round and a
fix commit to restore. Prefer targeted edits. Where a full replacement is truly
warranted, the plan must list the existing assertions and say which survive,
which are renamed, and which are deliberately dropped and why.

**Say what the mandated code does at its edges, and require the test.** A
docstring promised that a NaN threshold returns an empty mask rather than
throwing, with no test for it; a `resolvePadSize` double-cast reached
`Partial<PadSize>` with no `typeof` object check, leaning on a guard clause
further down. For every documented parameter, ask what a plausible bad value does
— a `width` documented as required but not as *positive* silently returned a
wrong kept set (32 mismatches in 1000 trials at width -8). If an edge case
corrupts results instead of throwing, the doc must say so and a test must pin it;
at a trust boundary, either narrow at runtime or state in the plan which later
guard makes the cast safe.

**Meta:** the dominant failure mode is a plan that is over-prescriptive about
incidentals and under-specific about prerequisites and evidence — which command
proves what, which dependency the code needs, which constant hits the edge case,
which number in mandated prose was ever measured. This run added a sharper
version of the same thing: for anything touching a browser, a bundler or a
worker, *reviewing* plan code is not evidence about plan code. Prose and code
the plan dictates verbatim ship unexamined and reviewers trust them; spend the
planning budget there.

## Recent runs

**issue-16-decode-sweep** (2026-08-28) — `overlapDecodeFilter` /
`gpuResidentEmbeddings` / `keepRawMasks` behind flags, a playground Compare tab,
and a headed real-GPU Playwright sweep writing `docs/measurements/`. Tier large,
5 tasks, 1 fix round across 2 tasks; verify skipped (no ui criteria); final
whole-branch review produced a 2-major fix wave and parked 9 minors. 18
findings, 12 plan-stage. The two majors are the run's lesson: plan-mandated
Playwright code that failed only under real execution (a `visible` wait on a
0px-tall empty `<pre>`; Vite pre-bundling a worker-only dynamic import mid-run
and reloading the page), plus an unguarded per-row `await` that would have
discarded a completed sweep on any single-row failure. The rest cluster into
cleanup skipped on early-exit paths (3), imprecise summary statistics in written
analysis (2), and repeats of known rules — stale control state in an export, a
vacuous test fixture, an unfalsifiable `grep` count. Outcome: no decode path is
a clear win; the fp16/batch-32 group spans 1.46%.

**issue-8-7-8-f2-single-pass-resample** (2026-08-28) — fused single-pass
threshold+resample for masks. Tier standard, 2 tasks, 2 parked, 0 fix rounds;
verify skipped. 12 findings, unusually spec-heavy: an Important one where the
spec asserted the repo had no `@playwright/test` dependency (it does, with a CI
lane) to justify skipping browser verification, a spec claim that swapped which
check catches geometry errors (measured: bbox delta stayed 1px, IoU fell to
0.92-0.95), an AC that overstated exact tiling, and a fix wave that introduced a
fresh unattributed figure while repairing the first one. Plan-stage: a mandated
test rewrite that silently dropped the only assertion covering the rebuild
path's zero-fill.

**issue-4-3-8-n2-n3-bbox-prefilter-and-bit-packed** (2026-08-27) — bbox
prefilter plus bit-packed coverage in `dedupeMasks`, with an opt-in `compareNms`
A/B against a preserved `dedupeMasksReference`. Tier standard, 2 tasks, 0
parked, 2 review rounds, 0 fix rounds; verify skipped. 6 findings: four
plan-stage (a `grep` step satisfied by the plan's own mandated JSDoc; a
`CANVAS_W` of 64 hiding the tail-word case; a doc comment claiming "roughly
doubles" against a measured ~124x; a documented `width` with no positivity
precondition) and two spec-stage (an AC parenthetical broader than the
criterion; the first `OVERRULED` verdict, review objecting to duplication the
spec deliberately requires).

**issue-5-4-8-m3-m4-1-bit-indexed-png-encoded-off** (2026-08-27) — 1-bit indexed
PNGs without canvas, encode moved into the segmenter worker, dropping ~1,069 ms
per mask. Tier standard, 3 tasks, 0 parked, 0 fix rounds; verify skipped. 2
findings, both minor, both plan-stage, both the same shape: a prerequisite the
mandated code needed and the plan never scheduled — `@types/node` for the
`node:zlib` test, and `.gitignore` entries for Playwright's output.
