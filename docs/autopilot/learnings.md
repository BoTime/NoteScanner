# Autopilot learnings

Rewritten each run from `.superpowers/autopilot/*/findings.jsonl`. Kept short on
purpose: rules are merged rather than appended, and "Recent runs" holds only the
last five runs.

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

**Do not mandate a test hook with no assertion behind it.** If a plan requires
`data-testid` attributes (or any other affordance that exists only for tests),
the same plan must contain the step that asserts on them. Otherwise drop the
hook and add it when the test arrives.

**Make sub-measurements tile their parent exactly.** When a plan splits an
instrumented region into sub-timers, place the boundaries so that no work sits
between the last sub-record and the parent's record — otherwise the sub-totals
read permanently under the parent and readers will hunt for the missing time.
If a residual is unavoidable, say in the plan that it is expected and roughly
how large it is.

**Leave cosmetic detail to the implementer, or get it right.** Plans are
followed literally, so a literal instruction to emit the same tree glyph on
every row produces exactly that, review finding and all. Specify presentation
only where it carries meaning; otherwise state the intent ("draw as a tree")
and let the implementer pick the characters.

**Flag differing denominators across sibling metrics.** If one counter counts
all iterations and its neighbours count only non-empty ones, per-row averages
are not comparable even though the rows sit side by side. Either normalize the
semantics or require the display to label the difference — decide this in the
plan, not in review.

**Meta:** every finding recorded so far has been `minor`, `stage_at_fault:
plan`, and confirmed as "no actual defect". The consistent failure mode is a
plan that is over-prescriptive about incidentals (glyphs, unused hooks) and
under-specific about evidence (which command proves what). Spend planning
budget in the opposite proportion.

## Recent runs

**issue-2-1-8-step-0-sub-timers-inside-filter** (2026-08-27) — sub-timers for
the filter stage's three regions in the WebGPU segmenter worker, plus sub-rows
in the playground `SegmentView`. Tier small, 1 task, 0 parked, 0 fix rounds;
verify skipped (no UI criteria). 6 findings, all minor and all attributed to the
plan stage: an invariant test asserting its own arranged inputs, a typecheck
coverage claim invalidated by tsconfig scope, an unused `data-testid` hook, a
sub-microsecond residual outside the instrumented sub-regions, a uniform tree
glyph on all three sub-rows, and non-comparable `count` semantics between
sibling rows. None was a real defect.
