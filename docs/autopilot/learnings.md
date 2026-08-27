# Autopilot learnings

Rewritten each run from `.superpowers/autopilot/*/findings.jsonl`. Kept short on
purpose: rules are merged rather than appended, and "Recent runs" holds only the
last few runs that produced findings, most recent first.

## Planning rules

**Point every invariant test at the code that owns the invariant.** A test that
hand-picks inputs and then asserts a property of those inputs proves nothing
about the implementation. If the invariant actually lives somewhere the repo
cannot test yet (a Web Worker, a GPU path), do not paper over it with a
pure-function test that looks equivalent: either scope the test infrastructure
into the plan, or state plainly in the plan that the invariant is covered by
review only and why that is acceptable for this task.

**Make every verification step falsifiable — and check the plan's own text
cannot satisfy it.** Two ways this has gone wrong. A plan claimed `npm run
typecheck` proved a property of a file `tsconfig` excludes (`playground/` is
outside the repo-wide command's scope). And a plan verified "no `declare` for
`dedupeMasksReference` leaks into `dist/.../index.d.ts`" with `grep -q
dedupeMasksReference`, which the JSDoc prose the same plan mandated verbatim
matches — the step passes no matter what the build emits. Before writing a
verification step, confirm the command's scope covers the file, and confirm
nothing the plan itself mandates trips it. Anchor greps to the construct
(`export`, `declare`), not to a bare identifier.

**Choose test constants that can fail.** The seeded differential for the
bit-packed NMS fixed `CANVAS_W` at 64 — a multiple of the 32-bit word — so the
test never produced a tail word or a word straddling the shared row band, which
is precisely the arithmetic the exactness claim rests on. When a change turns on
alignment, remainders, or boundaries, pick constants that land off the boundary
(64 → 61) and say in the plan which edge case each constant exists to hit.

**Treat plan-mandated prose as code you have not checked.** Verbatim doc
comments in a plan get committed verbatim, and reviewers read them as fact. Two
defects came out of one JSDoc block: a magnitude nobody had measured (an option
described as "roughly doubles the nms stage" when the reference path measures
~123x the fast path, so the stage grows ~124x and can sit silent for over a
minute at 32 points per side), and a precondition that fails silently (the doc
required a `width` but never said positive; a negative width collapses the mask
y-extent and returns a wrong kept set with no error). Do not put a number in
mandated prose that the plan has not measured — write it qualitatively or
schedule the measurement — and for every documented parameter, ask what a
plausible bad value does; if it corrupts results instead of throwing, the doc
must say so.

**Give every dependency the plan's code implies its own step.** A test importing
`node:zlib` and `Buffer` — the first node-typed code under `src/` — with no
`@types/node` in `devDependencies` left `npm run typecheck` resting on a
transitive dep from vite/vitest; a new `@playwright/test` harness with no
`.gitignore` step left `test-results/` to be deleted by hand. When a task adds a
package, imports from a new runtime surface, or adds a tool that writes files,
the same task adds the manifest entry and the ignore rule.

**Keep an acceptance criterion's rationale no wider than what it binds — and
justify deliberate duplication where the reader will meet it.** An AC bound
`timings.record('nms')` and the nms progress event to the fast path, which
holds, but its parenthetical "the results table is never inflated" does not:
`totalMs` is main-thread wall clock and still includes the reference run in A/B
mode. Write the parenthetical about what the criterion actually checks. The same
applies to duplication a spec wants on purpose: this run's reference
implementation copies the greedy-loop scaffold verbatim *because* the spec makes
the pre-change body the differential baseline, and review flagged it as
copy-paste until the reasoning was supplied. Say so at the duplication, in the
spec and in the code.

**Render a recorded run from the recorded run.** In any measurement UI, a row's
labels (dtype, batch size, pixels-per-side) must come from the options captured
with the result, never from current control state — otherwise changing a select
after a run silently relabels numbers measured under the old setting, and the
on-screen table disagrees with the exported markdown. Staleness runs the other
way too: a re-run that fails must clear or mark the previous result. Say in the
plan which state a re-run invalidates.

**Name the convention behind every number, and test its ambiguous case.**
"Median" over an even count, a counter that counts all iterations sitting beside
counters that count only non-empty ones, sub-timers that leave a residual
outside the parent's instrumented region — each is defensible and each misleads
a reader who assumes the obvious reading. Decide the convention in the plan,
require a comment stating it, and require a test that exercises the case where
the definitions diverge. When splitting an instrumented region into sub-timers,
place the boundaries so no work sits between the last sub-record and the
parent's record; if a residual is unavoidable, say roughly how large.

**Bound the lifetime of large buffers in the plan.** Holding full-resolution
`RawMask[]` for every row of a comparison table keeps hundreds of megabytes
alive for the session when the only consumer reduces them to a ten-number
summary. If a plan retains big typed arrays, state when they are released — and
prefer caching the derived summary and dropping the source.

**Do not mandate incidentals; do mandate the assertion.** Plans are followed
literally, so an instruction to emit the same tree glyph on every row produces
exactly that, review finding and all, and a required `data-testid` with no
matching assertion step produces a hook nothing uses. Specify presentation only
where it carries meaning, state intent otherwise ("draw as a tree"), and never
add a test-only affordance without the test that reads it. Where two views need
the same panel, plan the shared component rather than a second copy that will
drift.

**Meta:** the dominant failure mode is a plan that is over-prescriptive about
incidentals and under-specific about prerequisites and evidence — which command
proves what, which dependency the code needs, which constant hits the edge case,
which number in a mandated comment was ever measured. Spend planning budget in
that proportion. Prose the plan dictates verbatim is the highest-yield place to
look: it ships unexamined and reviewers trust it.

## Recent runs

**issue-4-3-8-n2-n3-bbox-prefilter-and-bit-packed** (2026-08-27) — bbox
prefilter plus bit-packed coverage in `dedupeMasks`, with an opt-in `compareNms`
A/B against a preserved `dedupeMasksReference`. Tier standard, 2 tasks, 0
parked, 2 review rounds, 0 fix rounds; verify skipped (all criteria `(non-ui)` —
the real-image measurements are taken by hand in the playground after merge).
6 findings: four plan-stage (a `grep` verification step satisfied by the plan's
own mandated JSDoc; a `CANVAS_W` of 64 that hid the tail-word and
band-straddling cases the exactness argument needs; a mandated doc comment
claiming "roughly doubles" against a measured ~124x; a documented `width` with
no positivity precondition, which silently returns a wrong kept set) and two
spec-stage (an AC parenthetical broader than the criterion — `totalMs` still
includes the reference run; and the first `OVERRULED` verdict, review objecting
to duplication the spec deliberately requires as the differential baseline).

**issue-5-4-8-m3-m4-1-bit-indexed-png-encoded-off** (2026-08-27) — write masks
as 1-bit indexed PNGs without canvas and move the encode into the segmenter
worker, dropping ~1,069 ms per mask and the coverage-buffer transfer. Tier
standard, 3 tasks, 0 parked, 0 fix rounds; verify skipped. 2 findings, both
minor and both plan-stage, both the same shape: a prerequisite the mandated code
needed and the plan never scheduled — `@types/node` for the `node:zlib` test,
and `.gitignore` entries for Playwright's output.

**issue-3-2-8-e1-d1-fp16-and-a-larger-batchsize** (2026-08-27) — fp16 and
larger-`batchSize` measurement harness in the playground. Tier standard,
3 tasks, 0 parked, 1 fix round. 5 findings, the first not attributed to the
plan: one medium implementation bug (row labels rendered from live select state
against a previous run's timings, so the table and the copied markdown
disagreed), an unreachable error cell after a failed re-run, a duplicated
WebGPU-required panel, an undocumented lower-median convention with no
even-count test, and a spec-level note on unreleased mask buffers. PR closed
unmerged — deprioritised — with the measurements recorded in issue #12.

**issue-2-1-8-step-0-sub-timers-inside-filter** (2026-08-27) — sub-timers for
the filter stage's three regions, plus sub-rows in the playground `SegmentView`.
Tier small, 1 task, 0 parked, 0 fix rounds; verify skipped. 6 findings, all
minor and all plan-stage, none a real defect: an invariant test asserting its
own arranged inputs, a typecheck coverage claim invalidated by tsconfig scope,
an unused `data-testid` hook, a sub-microsecond timing residual, a uniform tree
glyph on all three sub-rows, and non-comparable `count` semantics between
sibling rows.
