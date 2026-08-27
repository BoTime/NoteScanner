# Autopilot learnings

Rewritten each run from `.superpowers/autopilot/*/findings.jsonl`. Kept short on
purpose: rules are merged rather than appended, and "Recent runs" holds only the
last few runs.

## Planning rules

**Close the gate gap for any file the task creates, or say out loud that the
file is unguarded.** Two runs in a row have been bitten by the same hole: this
repo's `tsconfig.json` excludes `playground/`, so `npm run typecheck` never sees
it, and `vitest.config.ts` includes only `playground/**/*.test.ts`, so no
playground `.tsx` is exercised either. Both real defects of the fp16/batchSize
run landed in `playground/CompareView.tsx` — the one changed file covered by
neither gate — while the whole suite went green. The previous run's plan had
already *noticed* the typecheck gap and deliberately skipped adding a playground
tsconfig; noticing is not enough. Before a plan cites a command as evidence,
check the config's `include`/`exclude` against the files the task will touch. If
they are outside, either scope the gate in (a playground tsconfig was measured
as free — playground typechecks clean today) or write in the plan, in the
verification step itself, that "green suite" says nothing about this file and
that review is the only cover.

**When a spec puts data in long-lived state, the spec must say when it leaves.**
The one `spec`-attributed finding: the compare harness keeps full-resolution
`RawMask[]` per row in React state for the session — a full run holds four rows
at once, plausibly 150-250MB — even though the only consumer reduces each set to
a ten-number summary. The spec said masks are released "when the row is re-run",
which is a release rule for the *overwrite* case only and silently accepts
unbounded retention otherwise. Any spec clause that stores a buffer, an image,
or a decoded model needs a matching clause naming the moment it is dropped, and
that moment should be as early as the consumer allows (here: cache the derived
summary, drop the masks once both halves of a pair are compared).

**A displayed result must be rendered from the inputs it was measured with, not
from the live controls.** Both real defects of the fp16/batchSize run were this
one bug wearing two hats. A table row rendered its dtype/pps/batch from current
select state but its timings from the stored previous run, so changing the
select relabelled already-measured numbers as a configuration that was never
run — and the copied markdown, which did read `result.options`, disagreed with
the screen. Separately, a row that succeeded and then failed on re-run kept
showing the old numbers because the error branch was only reachable in the
"not run" state. Plan the state machine explicitly: every stored result carries
its own options and is rendered from them, and every control change or re-run
either clears the stale result or marks it stale. This matters most in a
measurement tool, whose entire value is that its numbers are trustworthy.

**Plan the full result-state matrix, not just success and empty.** `not run`,
`running`, `succeeded`, `failed`, and `succeeded-then-failed` are five states,
and the last one is the one plans forget. If the runner clears the stale error
on entry, require it to clear the stale result too.

**When the acceptance criteria need hardware or downloads the pipeline cannot
have, build the instrument and say so.** Issue #3 asked for an fp16 +
`batchSize` performance change but graded it on measured numbers, which need
WebGPU, a real GPU and a ~40MB model fetch — nothing an unattended run in this
repo can produce. The run deliberately shipped a playground "Compare" tab and
left `DEFAULT_SEGMENTER_OPTIONS` untouched. That was the right call, and the
plan should make it the stated deliverable up front rather than discovering it
mid-implementation: name the harness as the output, name the default that is
*not* being changed, and name the human step that turns the numbers into the
config change.

**Point every invariant test at the code that owns the invariant.** A test that
hand-picks inputs and then asserts a property of those inputs proves nothing
about the implementation. If the invariant lives somewhere the repo cannot test
yet (a Web Worker, a GPU path), either scope the test infrastructure into the
plan or state plainly that it is covered by review only, and why that is
acceptable here.

**Pin down every statistic's convention, and test the path the examples miss.**
`compareMaskSets` takes the lower median on an even count rather than the mean
of the two middle values — defensible, but uncommented, and every median test
used sets where all IoUs were 1, so the even-count branch was never executed. If
a plan introduces a summary number, it should fix the convention in words and
require at least one case that is not degenerate.

**Say "extract" or "copy", never "same as over there".** The WebGPU-required
panel was reproduced near-verbatim from `SegmentView.tsx` into `CompareView.tsx`,
`data-testid` and browser-version copy included, differing by one word. When a
plan reuses an existing block, decide in the plan whether it becomes a shared
component; two copies of a notice about browser support will drift.

**Spend the presentation budget on meaning, not incidentals.** Plans are
followed literally, so a literal instruction to emit the same tree glyph on
every row produces exactly that, review finding and all — and a mandated
`data-testid` with no assertion behind it is dead weight. Specify presentation
only where it carries meaning; conversely, *do* specify semantics that look
cosmetic but are not, such as sibling counters with different denominators
(`select.count` counts all batches, its neighbours only non-empty ones) or
sub-timers that leave a residual outside the parent's instrumented region.

## Recent runs

**issue-3-2-8-e1-d1-fp16-and-a-larger-batchsize** (2026-08-27) — GitHub issue #3
asked for fp16 + a larger `batchSize`; the run instead built the measurement
harness the issue's own criteria require (a playground "Compare" tab running the
dtype x batchSize matrix, with IoU agreement between pairs) and deliberately
left `DEFAULT_SEGMENTER_OPTIONS` unchanged. Tier standard, 3 tasks, 0 parked,
1 fix round across 1 task; verify skipped (no UI criteria). 6 findings: 2 real
defects (1 medium, 1 low), both in `playground/CompareView.tsx` — the file
covered by neither `npm run typecheck` nor vitest — and both the same shape,
a stored result rendered against live control state. Plus 2 nits (a duplicated
WebGPU-required panel, an undocumented lower-median convention with an untested
even-count path) and 1 `spec`-attributed design note: unbounded retention of
full-resolution mask buffers in component state.

**issue-2-1-8-step-0-sub-timers-inside-filter** (2026-08-27) — sub-timers for
the filter stage's three regions in the WebGPU segmenter worker, plus sub-rows
in the playground `SegmentView`. Tier small, 1 task, 0 parked, 0 fix rounds;
verify skipped (no UI criteria). 6 findings, all minor and all attributed to the
plan stage, none a real defect: an invariant test asserting its own arranged
inputs, a typecheck coverage claim invalidated by tsconfig scope, an unused
`data-testid` hook, a sub-microsecond residual outside the instrumented
sub-regions, a uniform tree glyph on all three sub-rows, and non-comparable
`count` semantics between sibling rows.
