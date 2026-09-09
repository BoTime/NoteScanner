# Autopilot learnings

Rewritten each run from `.superpowers/autopilot/*/findings.jsonl`. Kept short on
purpose: rules are merged rather than appended, and "Recent runs" holds only the
last few runs that produced findings, most recent first.

## Planning rules

**Every file the change falsifies belongs on the plan's file list — the
architecture doc most of all.** A branch made four statements in `docs/pipeline.md`
false (filter upsamples before NMS, a "full res" edge label, a full-resolution NMS
loop, the issue marked open); the file structure table never listed it, so no task
touched it, and the repo's one architecture overview contradicted the shipped code —
while still carrying staleness from two earlier waves nobody re-checks. The smaller
version bites constantly: a `session-key.ts` comment enumerating excluded fields, not
extended for the field the same brief adds; a `createSegmenter` comment naming a
function the same step deletes; playground test files holding `counts` literals that
broke the typecheck the moment a member was added. Two shapes recur. *Sibling
contradiction*: this run's mandated `What ships today` bullet tagged F2 as shipped
while the same file's status table lists issue #8 open — a doc edit must be
consistent with the rest of its own file, not just true in isolation. *Half-updated
comment*: a comment beside a guard widened from `!mask` to `!mask || !encodeMask`
still described only the old drop. Before finalising the file list, grep the repo for
the names, numbers and diagrams the change invalidates — comments and docs included —
and put each hit on the list or say why it stays. And give doc insertions an
unambiguous anchor: "immediately after the *Where the budget lands* table" is
ambiguous when a prose paragraph belongs to that table.

**Code the plan dictates verbatim has never been run. Schedule the run — and never
assign an acceptance criterion to a hand run.** Two majors came out of literal plan
code that typechecked, passed review and passed its unit tests, then failed on first
contact with a real browser: a `waitForSelector` in the default `visible` state aimed
at a `<pre>` with a genuine 0px box until the first result arrives; and a dynamic
`import()` of `@huggingface/transformers` inside a Web Worker, invisible to Vite's
dependency scanner, so first inference reloaded the page and wiped the state the
runner was polling. jsdom computes no layout and runs no bundler. Real execution is
now demonstrably affordable here — three consecutive runs have had `(ui)` criteria
actually execute (1/1, 5/5, 2/2). So the other half of this rule matters: this run's
plan coverage table left AC2's running-app half — clicking edges and a thin structure
on the Segment tab with the flag on and off — to a person with a real WebGPU adapter.
A criterion parked on a hand run is a criterion with no evidence. Put a real
end-to-end invocation in an early task and budget for it. This run shows the failure
mode the rule does not yet name: plan-dictated *assertions* are unrun too, and they
fail silently rather than loudly. The AC6 spec clicked the viewer canvas and asserted
nothing about the result, under a test named "…and selects a segment" — it would have
passed with selection entirely dead. It also assumed the wrong interaction: selecting
is two steps, a canvas click opens a `role="menu"` of the masks under the cursor and
its `Select` item commits. Writing the real assertion is what surfaced that. **Any
step that performs an action and asserts nothing about its effect is not coverage.**
Before dictating an assertion about an interaction, confirm the interaction exists in
the component being driven, not in the plan's mental model of it.

**A guard is worth exactly the breaches it can detect — prove that, don't assume it.**
The AC8 test forbidding imports across the `site/` ↔ `playground/` boundary passed
while missing side-effect imports (`import '../playground/x.css'` carries no `from`)
and `new Worker(new URL('../playground/w.ts', import.meta.url))` — the two idioms this
repo itself uses, and so the realistic ways the boundary would actually break. It also
scanned only `.ts/.tsx/.html`, and recursed into `site/dist`. A vacuity check that the
file list is non-empty is necessary and was present; it is not sufficient, because it
proves the loop ran, not that the pattern matches anything real. Pair every guard with
cases that feed it each breach form and assert it catches them. The same applies to
cross-engine assertions: an AC5 button count hardcoded at 3 failed in webkit, which
correctly renders no Run button because it exposes no `navigator.gpu` — an
engine-dependent value must be derived from the engine, and the fix is to derive it
while keeping the assertion strict, never to relax it.

**Disable every path to a state change, not the obvious ones.** The public page
disabled its sample chips and file input while a run was in flight, but the drop
target on the same panel was not gated by either — so a photo dropped mid-run got the
previous photo's masks and stat line painted over it. Wrong output that looks
plausible is worse than an error. When a control is disabled for a reason, enumerate
the other routes to the same state (a drop handler, a keyboard path, a deep link) and
gate them at the one place they converge; then have the async work verify on
completion that the state it was computed for is still current.

**Pin the branch to what it will merge into.** A worktree stayed at merge base
`3cccb78` while `main` advanced to `3fe5e56`, which rewrote `DEFAULT_SEGMENTER_OPTIONS`
and `batchSize` in `types.ts` and replaced `CompareView`'s local constants with
`playground/option-choices.ts`. Every green gate that run reported described a tree
that will never exist on `main`. When a run spans hours next to other merges, re-check
the base before the final review and re-run the gates on the merge result.

**Check claims about the repository — including claims about your own edit — against
the repository, and make every verification step falsifiable.** A spec justified
skipping browser verification with "the repo has no `@playwright/test` dependency";
the repo pins it, with a config, a spec and a CI lane. Plan prose said "four
`recordFilterSub` and six `filterSubPhases` occurrences" where the file has five and
five. On the gate side, recurring in five runs: a plan claimed `npm run typecheck`
proved a property of a file `tsconfig` excludes; a plan verified "no `declare` for
`dedupeMasksReference` leaks into `index.d.ts`" with `grep -q dedupeMasksReference`,
which the JSDoc that same plan mandated matches; a plan asserted `grep -n catch` shows
two matches when it shows five. Confirm the command's scope covers the file, confirm
nothing the plan itself mandates trips it, anchor greps to the construct (`export`,
`declare`, `catch (`), and run any count you intend to assert.

**Plan the failure path, and never let a batch runner lose completed work.** Same
shape four times: `chromium.launch()` outside the `try/finally` that closes the
server; `die()` calling `process.exit(1)` from inside that `try`, skipping the
`finally`; a pre-dispatched next-batch promise with no rejection handler, firing an
`unhandledrejection` in exactly the error case the design meant to keep quiet; and an
unguarded `await runOneRow(...)` that would have discarded every already-measured row
on one hard failure. For each resource the plan's code acquires, name where it is
released when the *next* step throws; wrap per-item calls and record failures as
results; validate up front that the inputs the run will feed the UI are inputs the UI
accepts (a trailing `--reps` with no value yields `Number(undefined)`). And check what
the failure path *says*: a page with no `navigator.gpu` renders a different panel, so
the wait died with a generic 30s timeout instead of the intended no-adapter message.

**Compute every number in prose, keep it reproducible from the committed tree, and
put the caveat in the artifact people open.** A commit message called four decode
paths "within roughly 2%" when the group spanned 4.0%; the fix claimed a floor of
"about 2.0%" against a true 1.46%. A mandated doc comment said "roughly doubles the
nms stage" against a measured ~124x. Worse than a wrong hedge is an unreproducible
right one: mirrored-frame flip counts quoted in a shipped comment and a measurements
doc came from a throwaway probe spec the implementer ran once and deleted before
commit — real numbers, but nothing in the tree regenerates them and the doc did not
say so. Disclosures belong in the committed `docs/measurements/*.md` a reader opens,
not in a commit message, and the plan should name that file. One caveat: a blanket
"every figure derived from the committed artifacts" rule leaves no room for an openly
illustrative calculation — allow it explicitly and label it, or the rule gets quietly
broken instead of followed.

**Point every test at the code that owns the invariant, choose constants that can
fail, and match the shipped geometry.** This is the corpus's most repeated major.
Constants off the boundary: a seeded differential fixed `CANVAS_W` at 64, a multiple
of the 32-bit word, so the bit-packing test never produced the tail or straddling
word the exactness claim rests on. Symmetry the bug hides in: a test built its variant
as `baseline.map(m => ({...m}))` — four identical IoU-1.0 pairs — so "takes the lower
median" passes under the upper median or the average; a real-GPU pixel spec centred a
32x24 rect in a 64x48 image, so a uniform flip of the composite pass moves every
sampled layer together and both assertions still pass. New and sharpest this run: the
plan's AC2 browser substitute pinned `SCALE = 4`, an exact integer upscale at which
nearest-neighbour is lossless *by construction*, while the shipped ratio is 4.000
across and 4.006 down — the entire boundary-quantisation defect class the criterion
exists to catch lived on the fractional axis and was never touched. A substitute
fixture must reproduce the shipped geometry's *character*, not a round number.
Guard count-derived assertions with a non-empty check. And prove which assertion does
the failing: a spec credited a bbox check with catching geometry errors, but the bbox
delta was 1px on healthy code *and* under an injected half-pixel regression, while IoU
fell to 0.92-0.95 against a 0.99 gate; an AC7 "paints nothing while lost" assertion
passed identically with the guard it targets deleted, because every GL call on a lost
context is a silent no-op. Inject the failure and record which assertion moves.

**Name the convention behind every number, keep a criterion's claim no wider than
what it binds, and check gates in both directions.** "Median" over an even count; a
counter counting all iterations beside counters counting only non-empty ones;
sub-timers leaving a residual outside the parent's instrumented region — each is
defensible and each misleads. An exactness invariant (sub-steps sum to the filter
total) that is exact in a synthetic unit test left a 0.1 ms residual on 3 of 4 real
GPU rows, because Chrome quantises `performance.now()` to 0.1 ms: when an AC asserts
an exact identity, say whether it binds the unit test only, and give the real-data
form a tolerance. On scope: an AC bound `timings.record('nms')` to the fast path,
which holds, but its parenthetical "the results table is never inflated" does not —
`totalMs` is wall clock; a low-res area gate was called "1.60x stricter"
one-directionally when a candidate whose logits cross zero in the letterbox pad gets
area credit that never reaches the output, making the gate *looser* there. Also avoid
freezing measured figures into ACs — a precision fix was parked purely because it
would shift numbers quoted in AC1. Put gates and tolerances in criteria, not figures a
benign change perturbs. Finally, do not pin a differential to the one configuration
users are not in: a WebGL2-vs-canvas2d measurement pinned `dpr` to 1, excluding every
retina display.

**Do not mandate incidentals; do mandate the assertion.** Plans are followed
literally, so an instruction to emit the same tree glyph on every row produces exactly
that, a required `data-testid` with no matching assertion step produces a hook nothing
uses, two new table cells requested with only a control-passthrough test get verified
by code reading, a mandated 28-line docblock ships four times the length of its
siblings, and a mandated near-duplicate helper ships instead of the shared one.
Generated-output details count: a hardcoded column index for a markdown alignment row
misaligns the table when a column is added, and an unescaped failure message
interpolated into a markdown cell destroys the row on a pipe. Recurring three runs
running: **render a recorded run from the recorded run** — dtype, batch size, points
per side, the grid line in an export must come from the options captured with the
result, never from live control state, and a failed re-run must clear or mark the
previous result.

**When the plan wraps or rewrites existing code, name what the old path loses.**
Wrapping the whole NMS region in `if (plan)` dropped a progress event a zero-batch run
always used to post. A verbatim replacement for `createSegmenter.test.ts` silently
dropped `expect(filterSubPhases.threshold.count).toBe(0)`, the only coverage that the
rebuild path zero-fills an unrecorded sub-step, costing a review round. Prefer
targeted edits; where a full replacement or new conditional is warranted, list what
exists today and mark each item kept, renamed or deliberately dropped.

**Give every dependency and every consumer its own step; keep the public surface and
the data crossing it narrow.** A test importing `node:zlib` with no `@types/node`; a
Playwright harness with no `.gitignore` for `test-results/`; a worker-only dynamic
import with no `optimizeDeps.include`. `core/index.ts` grew `export * from
'./mask-pipeline'`, publishing ~12 worker-lifecycle names carrying a
mutation-and-ordering contract meaningful only inside the worker loop, solely so one
playground file could import from `../src/segmenter` — a deep import would have kept
it narrow. Name the exact import path the new consumer will use. Bound large payloads
in the plan too: a full-resolution `RawMask[]` per comparison row holds hundreds of
megabytes for a consumer that reduces them to ten numbers, and a `paintSequence` that
returns `frames` and `live` when its only caller destructures `{ peak }` ships ~25 x
12,288 unused pixel values over CDP per engine.

**Say what the mandated code does at its edges, carry the guards its siblings already
have, and require the test.** A docstring promised a NaN threshold returns an empty
mask rather than throwing, with no test; a `resolvePadSize` double-cast reached
`Partial<PadSize>` with no `typeof` check; a `width` documented as required but not as
*positive* silently returned a wrong kept set (32 mismatches in 1000 trials at width
-8). This run: `resolveEncodeSize` shipped without the `Math.max(1, ...)` lower bound
that `lowResMinArea` applies to the same shape of computation twelve lines above —
when the plan adds a computation shaped like one already in the file, copy its guards
or say why not. A new default's resource ceiling is an edge too: making WebGL2 the
default gave every mounted viewer a live context for its lifetime, and browsers evict
the oldest and fire `webglcontextlost` on viewers still on screen with nothing calling
`restoreContext` — undocumented in spec and README; and the spec's promised
downgrade-to-canvas2d only ever ran on a throwaway 1x1 probe, so a live-instance
failure set a private flag nothing observes and `paint()` blanked the viewer. So is
test-harness config: `webServer.reuseExistingServer: !CI` plus vite's
`strictPort: 5180` lets *one worktree's* dev server silently serve another worktree's
browser run — a standard idiom that becomes a correctness hazard the moment two
autopilot worktrees are live, and worth stating rather than discovering.

**Meta:** the dominant failure mode is unchanged — plans over-prescriptive about
incidentals and under-specific about prerequisites and evidence. The last three runs
sharpened the evidence half specifically. A plan's blast radius is wider than its
diff: the docs, comments and sibling test literals a change falsifies are part of the
change. Reviewing plan code is not evidence about plan code. And a green test is not
evidence either unless its fixture can fail — the two majors of this run and two of
the previous one were all tests that passed for reasons unrelated to the property
under test.

## Recent runs

**issue-11-create-a-public-github-page-as-a-playgro** (2026-09-08) — a public "Try it"
page at `botime.github.io/NoteScanner/`, built from a new `site/` entry, plus a shared
top-level `samples/` and a Pages workflow. Tier standard, 3 tasks, 0 parked, 1 fix
round. Verify ran: 5/5 `(ui)` criteria in chromium, webkit and firefox against the
production build under its real base path, and AC6 end to end on a real adapter (34
segments · 14.7 s, matching the `pps 16` mask count in the decode sweep). Four majors,
all found by a whole-branch review rather than by the gates, and all sitting under
green checks: a vacuous AC6 assertion; an import guard blind to the two import idioms
the repo uses; a mid-run drop painting stale results onto a new photo; and an
undecodable dropped file throwing uncaught with no user-visible feedback and a leaked
object URL. Minor: a hardcoded cross-engine button count that webkit correctly
contradicts. **The process finding matters more than any of them.** Two dispatched
stage agents were terminated by the runtime mid-run, both reported as "stopped by
user" with no interrupt record anywhere and no error in their own transcripts; the
kill notifications landed on an exact 30-second grid, at identical sub-second phase,
870.000s apart. Task 2 therefore landed with no review at all and Task 3 was written
by the orchestrator, so the per-task review gate — the thing that would normally have
caught those four defects — silently did not happen for two thirds of the branch. A
stage that cannot report its own non-completion is worse than one that parks: the run
looked finished. Treat a missing per-task review as a blocking condition and run a
whole-branch review before landing whenever one is missing.

**issue-7-6-8-m2-encode-masks-at-256x256** (2026-09-01) — encode each survivor's PNG
at the decoder's own logit window (256x162 for a 1024x649 photo) behind
`lowResMaskEncode`, with playground/sweep exposure and a real-GPU writeup. Tier
standard, 3 tasks, 2 parked, 0 fix rounds; verify ran and passed 2/2 `(ui)` criteria.
10 findings, 8 plan-stage. Both majors are about evidence rather than code: the plan's
AC2 browser substitute pinned `SCALE = 4`, an exact integer upscale at which
nearest-neighbour is trivially lossless, so the boundary-quantisation defect class AC2
exists to catch was unexercised (repaired with a second scene at a 4.025 vertical
ratio, twelve hit-test probes and hand-derived mismatch bounds); and the coverage
table left AC2's running-app half to a hand run on a machine with a real WebGPU
adapter. A third major was ruled out of scope: playwright's `reuseExistingServer` plus
vite `strictPort: 5180` lets one worktree's dev server serve another worktree's
browser run. Minors: `resolveEncodeSize` missing the `Math.max(1, ...)` floor its
sibling applies twelve lines above (fixed); a `What ships today` bullet tagging F2 as
shipped while the same file's table lists issue #8 open; two deferred
over-prescriptions (a near-duplicate resample helper, a 28-line docblock); two
contested (a comment describing half a widened guard, an ambiguous doc insertion
point). Outcome: a 16.02x pixel reduction buys only 3.0x/3.5x on the stage — ~85% of
per-mask encode cost is fixed, not pixel-proportional.

**issue-9-8-8-m5-f4-webgl2-renderer** (2026-09-01, PR #22, learnings written on its
own branch and folded in here) — a WebGL2 renderer for `SegmentViewer`, made the
default with a canvas2d fallback. Tier standard, 3 tasks, 0 parked, 1 fix round;
verify 5/5 `(ui)` criteria. 14 findings, unusually evenly spread across plan, spec and
implementation. Five majors: a real-GPU pixel spec whose centred 32x24-in-64x48 rect
cannot detect a uniform flip; an AC7 "paints nothing while lost" assertion that passes
identically with the guard deleted; a spec promising a downgrade the code implements
only in the 1x1 probe, so a live-instance failure blanks the viewer instead; an
undocumented resource ceiling now that every mounted viewer holds a live WebGL2
context; and one `NEAREST` filter shared by coverage targets and the base photo, with
the differential pinned to `dpr` 1 — the configuration most users are not in. Minors
repeat known themes: a number published from a deleted throwaway probe, a residual gap
left to inference, a plan-mandated unused return value shipping pixels over CDP, an
artifact filename dated from the plan slug rather than the run.

**issue-6-5-8-f1-n1-f3-carry-256x256-through-filter** (2026-08-31) — filter and dedupe
at 256x256 with resampling only for survivors, behind `lowResFilterNms`; a playground
Boundary tab, Compare/sweep exposure, a real-GPU writeup. Tier large, 5 tasks, 0
parked, 0 fix rounds; verify 1/1 `(ui)` — the first run in this repo where a `(ui)`
criterion executed rather than skipped. 16 findings, 12 plan-stage. Two majors:
`docs/pipeline.md` left describing the pre-change pipeline (and carrying staleness
from two earlier waves), and the branch verified against a merge base `main` had moved
past — the corpus's only `brief`-stage finding. The minors cluster into one theme, the
plan's own text against its own change: an incomplete file list that broke the
playground typecheck, prose miscounting occurrences, a comment ordered preserved that
the same step falsifies, a conditional dropping a zero-batch progress event. Plus an
`export *` widening the public API for one playground import, and a timing identity
exact in the unit test but 0.1 ms short on real GPU rows.
