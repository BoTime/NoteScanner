# Autopilot learnings

Rewritten each run from `.superpowers/autopilot/*/findings.jsonl`. Kept short on
purpose: rules are merged rather than appended, and "Recent runs" holds only the
runs that produced findings, most recent first.

## Planning rules

**Point every invariant test at the code that owns the invariant.** A test that
hand-picks inputs and then asserts a property of those inputs proves nothing
about the implementation. If the invariant actually lives somewhere the repo
cannot test yet (a Web Worker, a GPU path), do not paper over it with a
pure-function test that looks equivalent: either scope the test infrastructure
into the plan, or state plainly in the plan that the invariant is covered by
review only and why that is acceptable for this task.

**Check that a cited verification command really covers the file.** Before a
plan claims "`npm run typecheck` proves X", confirm the file is inside the
tsconfig `include`/`exclude` scope — `playground/` is excluded in this repo, so
nothing under it is typechecked by the repo-wide command. Name a command whose
scope you have verified, or make the evidence step something that is genuinely
repo-wide (a `grep` for every construction site, a standalone `tsc` on the
file).

**Give every dependency the plan's code implies its own step.** Twice now the
plan mandated code and left its prerequisites to chance: a test importing
`node:zlib` and `Buffer` — the first node-typed code under `src/` — with no
`@types/node` in `devDependencies`, so `npm run typecheck` came to rest on a
transitive dep from vite/vitest; and a new `@playwright/test` harness with no
step adding `test-results/` and `playwright-report/` to `.gitignore`, so its
output had to be deleted by hand. When a task adds a package, adds an import
from a new runtime surface, or adds a tool that writes files, the same task adds
the manifest entry and the ignore rule.

**Render a recorded run from the recorded run.** In any measurement UI, a row's
labels (dtype, batch size, pixels-per-side) must come from the options captured
with the result, never from current control state — otherwise changing a select
after a run silently relabels numbers that were measured under the old setting,
and the on-screen table disagrees with the exported markdown. The same rule
covers staleness in the other direction: a re-run that fails must clear or mark
the previous result, not leave last run's numbers rendered with no failure
showing. Say in the plan which state a re-run invalidates.

**Name the convention behind every number, and test its ambiguous case.**
"Median" over an even count, a counter that counts all iterations sitting beside
counters that count only non-empty ones, sub-timers that leave a residual
outside the parent's instrumented region — each is defensible and each misleads
a reader who assumes the obvious reading. Decide the convention in the plan,
require a comment stating it, and require a test that exercises the case where
the definitions diverge. When splitting an instrumented region into sub-timers,
place the boundaries so no work sits between the last sub-record and the
parent's record; if a residual is unavoidable, say so and say roughly how large.

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
proves what, which dependency the code needs, which state a re-run invalidates.
Spend planning budget in that proportion. The one `medium` finding so far was an
implementation slip of exactly this shape (live control state rendered against a
stale result) in an instrument whose only job is trustworthy numbers.

## Recent runs

**issue-5-4-8-m3-m4-1-bit-indexed-png-encoded-off** (2026-08-27) — write masks
as 1-bit indexed PNGs without canvas and move the encode into the segmenter
worker, dropping ~1,069 ms per mask and the coverage-buffer transfer. Tier
standard, 3 tasks, 0 parked, 0 fix rounds; verify skipped (no UI criteria).
2 findings, both minor and both plan-stage, both the same shape: a prerequisite
the mandated code needed and the plan never scheduled — `@types/node` for the
`node:zlib` test, and `.gitignore` entries for Playwright's output.

**issue-3-2-8-e1-d1-fp16-and-a-larger-batchsize** (2026-08-27) — fp16 and
larger-`batchSize` measurement harness in the playground (`CompareView`,
`compare.ts`). Tier standard, 3 tasks, 0 parked, 1 fix round. 5 findings, the
first not attributed to the plan: one medium implementation bug (row labels
rendered from live select state against a previous run's timings, so the table
and the copied markdown disagreed), an unreachable error cell after a failed
re-run, a duplicated WebGPU-required panel, an undocumented lower-median
convention with no even-count test, and a spec-level note on unreleased
full-resolution mask buffers. PR closed unmerged — deprioritised — with the
measurements recorded in follow-up issue #12.

**issue-2-1-8-step-0-sub-timers-inside-filter** (2026-08-27) — sub-timers for
the filter stage's three regions in the WebGPU segmenter worker, plus sub-rows
in the playground `SegmentView`. Tier small, 1 task, 0 parked, 0 fix rounds;
verify skipped. 6 findings, all minor and all plan-stage, none a real defect:
an invariant test asserting its own arranged inputs, a typecheck coverage claim
invalidated by tsconfig scope, an unused `data-testid` hook, a sub-microsecond
residual outside the instrumented sub-regions, a uniform tree glyph on all three
sub-rows, and non-comparable `count` semantics between sibling rows.
