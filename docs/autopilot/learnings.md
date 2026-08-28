# Autopilot learnings

Rewritten each run from `.superpowers/autopilot/*/findings.jsonl`. Kept short on
purpose: rules are merged rather than appended, and "Recent runs" holds only the
last few runs that produced findings, most recent first.

## Planning rules

**Check every claim about this repository against this repository — hardest when
the claim waives work.** A spec asserted the repo has no `@playwright/test`
dependency and used that to justify skipping browser verification; the repo pins
`@playwright/test` 1.62.1, ships `playwright.config.ts`, a browser spec, and a
CI lane that runs it. The same shape recurs in verification steps: a plan claimed
`npm run typecheck` proved a property of a file `tsconfig` excludes
(`playground/`), and another verified "no `declare` leaks into `index.d.ts`" with
`grep -q dedupeMasksReference`, which the JSDoc the same plan mandated verbatim
matches — the step passes no matter what the build emits. Before a plan or spec
asserts what the repo has, what a command covers, or what a grep proves: run the
grep, read `package.json`, confirm the command's scope reaches the file, and
confirm nothing the plan itself mandates trips the check. Anchor greps to the
construct (`export`, `declare`), never to a bare identifier.

**Never write a number into mandated prose that nobody measured — and do not
freeze numbers you may still want to move.** Verbatim doc comments and spec
paragraphs ship unexamined and reviewers read them as fact. This has produced: an
option described as "roughly doubles the nms stage" when the reference path
measures ~123x the fast path (so it can sit silent for over a minute at 32 points
per side); and — in the very wave fixing a false-fact finding — a fresh
unattributed IoU figure written into the spec beside figures the suite actually
prints. Every figure in a doc must be reproducible by a named command, or stated
qualitatively, or explicitly labelled as pending developer measurement. The
converse bites too: an AC that quotes exact measured values freezes the
implementation behind it — this run parked a Float32/f64 precision asymmetry
purely because fixing it would shift numbers already quoted in AC1. Put gates and
tolerances in ACs, not figures a benign change perturbs.

**Make the test able to fail, and know which assertion does the failing.** Two
halves of one rule. Constants: the seeded NMS differential fixed `CANVAS_W` at
64 — a multiple of the 32-bit word — so it never produced a tail word or a word
straddling the shared row band, precisely the arithmetic the exactness claim
rests on (changed to 61). Attribution: a spec credited a bounding-box check with
catching geometry errors; measured, bbox delta was 1px on healthy code *and* 1px
under an injected half-pixel regression, while IoU fell to 0.92-0.95 against the
0.99 gate — the roles were swapped. When a plan claims a check guards a failure
mode, inject that failure and record which assertion actually moves; when a
change turns on alignment, remainders, or boundaries, pick constants off the
boundary and say which edge case each exists to hit.

**Do not mandate a wholesale rewrite of an existing test file.** The plan's
verbatim replacement for `createSegmenter.test.ts` silently dropped
`expect(filterSubPhases.threshold.count).toBe(0)` — the only coverage that the
rebuild path zero-fills an unrecorded sub-step — and it took a review round and a
fix commit to restore. Prefer targeted edits. Where a full replacement is truly
warranted, the plan must list the existing assertions and say which survive,
which are renamed, and which are deliberately dropped and why.

**Point every invariant test at the code that owns the invariant.** A test that
hand-picks inputs and then asserts a property of those inputs proves nothing.
This keeps recurring at seams the repo cannot execute — a Web Worker, a GPU path
(this run: the worker's fused `resampleThresholdMask` call site, where argument
order rests on cross-reading alone). Do not paper over it with a pure-function
test that looks equivalent: either scope the test infrastructure into the plan,
or state plainly in the plan that the seam is covered by review only, name what
could silently break there, and say why that is acceptable for this task.

**Give every dependency the plan's code implies its own step.** A test importing
`node:zlib` and `Buffer` with no `@types/node` left `npm run typecheck` resting
on a transitive dep from vite/vitest; a new `@playwright/test` harness with no
`.gitignore` step left `test-results/` to be deleted by hand. When a task adds a
package, imports from a new runtime surface, or adds a tool that writes files,
the same task adds the manifest entry and the ignore rule.

**Keep an acceptance criterion's rationale no wider than what it binds.** An AC
bound `timings.record('nms')` and the nms progress event to the fast path, which
holds — but its parenthetical "the results table is never inflated" does not:
`totalMs` is main-thread wall clock and still includes the reference run. Another
asserted two sub-regions tile the filter total *exactly*, while `recordSub` fires
only when `chosen.length > 0`, so an empty batch leaves a residual. Both
overshoots were inherited verbatim from surrounding code and specs; re-derive an
AC's wording from the conditional paths in the code, do not copy the shape of the
one next to it.

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

**Name the convention behind every number.** "Median" over an even count, a
counter that counts all iterations beside counters that count only non-empty
ones, sub-timers that leave a residual outside the parent's instrumented region —
each is defensible and each misleads a reader assuming the obvious reading.
Decide the convention in the plan, require a comment stating it, and require a
test exercising the case where definitions diverge. When splitting an
instrumented region, place boundaries so no work sits between the last sub-record
and the parent's record.

**Render a recorded run from the recorded run.** In a measurement UI, a row's
labels must come from the options captured with the result, never from current
control state — otherwise changing a select after a run relabels numbers measured
under the old setting and the table disagrees with the exported markdown.
Staleness runs the other way too: a re-run that fails must clear or mark the
previous result. Say in the plan which state a re-run invalidates. Related: if a
plan retains large typed arrays (full-resolution `RawMask[]` per table row ran to
hundreds of megabytes), state when they are released, and prefer caching the
derived summary over the source.

**Do not mandate incidentals; do mandate the assertion — and justify deliberate
duplication at the site.** Plans are followed literally, so "use this tree glyph
on every row" produces exactly that, review finding and all, and a required
`data-testid` with no matching assertion step produces a hook nothing uses.
Specify presentation only where it carries meaning. Where duplication is
intentional — a reference implementation copying a greedy loop verbatim *because*
the spec makes the pre-change body the differential baseline; two near-identical
loops written out inline for a perf asymmetry — say so in a comment at the
duplication, or review will keep flagging it (once as `OVERRULED`, once as
`DEFERRED`). Where two views need the same panel, plan the shared component.

**Meta:** the dominant failure mode is a plan or spec that is over-prescriptive
about incidentals and under-specific about prerequisites and evidence — which
command proves what, which dependency the code needs, which constant hits the
edge case, which number in a mandated comment was ever measured. Prose dictated
verbatim is the highest-yield place to look: it ships unexamined and reviewers
trust it. The strongest single habit is to grep the repo before asserting
anything about it, especially when the assertion is the reason to skip work.

## Recent runs

**issue-8-7-8-f2-single-pass-resample** (2026-08-27) — fuse the mask upsample and
threshold into one bilinear pass (`resampleThresholdMask`) and wire it into the
segmenter worker. Tier standard, 2 tasks, 0 fix rounds during sdd; verify skipped
(both criteria `(non-ui)`; the filter-phase before/after table is hand-measured
in the playground after landing). 10 findings. Four per-task minors were deferred
(inline duplication kept for the perf asymmetry, an untested NaN-threshold
docstring promise, no executable test for the worker call site, a double cast in
`resolvePadSize`). The whole-branch review then ran two rounds and produced the
run's only Important: the spec claimed the repo has no `@playwright/test` to
justify skipping browser verification, when it pins 1.62.1 with a config, a
browser spec and a CI lane. Fixed in `ff3970a` along with a misattributed test
check (bbox vs IoU) and a plan-mandated test rewrite that had dropped an
assertion — but that same fix wave wrote a fresh unattributed IoU figure into the
spec, the defect class it was fixing, and was parked with the Float32/f64
precision asymmetry that AC1's quoted numbers had frozen.

**issue-4-3-8-n2-n3-bbox-prefilter-and-bit-packed** (2026-08-27) — bbox prefilter
plus bit-packed coverage in `dedupeMasks`, with an opt-in `compareNms` A/B
against a preserved `dedupeMasksReference`. Tier standard, 2 tasks, 2 review
rounds, 0 fix rounds; verify skipped. 6 findings: four plan-stage (a `grep` step
satisfied by the plan's own JSDoc; a `CANVAS_W` of 64 hiding the tail-word case;
"roughly doubles" against a measured ~124x; a `width` with no positivity
precondition) and two spec-stage (an AC parenthetical broader than the criterion;
and the first `OVERRULED` verdict, review objecting to duplication the spec
deliberately requires).

**issue-5-4-8-m3-m4-1-bit-indexed-png-encoded-off** (2026-08-27) — 1-bit indexed
PNG masks without canvas, encode moved into the worker; ~1,069 ms per mask saved.
Tier standard, 3 tasks, 0 fix rounds; verify skipped. 2 findings, both minor,
both plan-stage, both the same shape — a prerequisite the mandated code needed
and the plan never scheduled (`@types/node`; `.gitignore` for Playwright output).

**issue-3-2-8-e1-d1-fp16-and-a-larger-batchsize** (2026-08-27) — fp16 and
larger-`batchSize` measurement harness in the playground. Tier standard, 3 tasks,
1 fix round. 5 findings, the first run not dominated by plan-stage faults: a
medium implementation bug (row labels from live select state against a previous
run's timings), an unreachable error cell after a failed re-run, a duplicated
WebGPU-required panel, an undocumented lower-median convention with no even-count
test, and a spec-level note on unreleased mask buffers. PR closed unmerged —
deprioritised — with the measurements recorded in issue #12.
