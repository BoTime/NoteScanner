# Sub-timers inside the `filter` stage

## Why

`docs`-side analysis (GitHub issue #1, §3.3) reasons the internal split of the
segmenter's `filter` stage as roughly: (1) `stabilityScore` + best-of-3 per
point at 256x256, ~2-3%; (2) `post_process_masks`, ~75-85%; (3) `thresholdMask`
+ `minMaskArea` at full resolution, ~12-23%. That split is **reasoned from the
two `interpolate_4d` call shapes, not profiled**, and it is what sizes issues #6
and #8. This change makes it measurable.

Scope settled in the brainstorm: this branch ships the **instrumentation only**.
Running the two operating points on real WebGPU hardware and recording the
figures in issue #1 §3.3 stays with the human, as does the conditional #6/#8
re-scope. Those are carried into the acceptance criteria as explicitly deferred.

## `src/segmenter/core/types.ts`

Add, beside the existing `PHASE_ORDER`:

```ts
export const FILTER_SUBSTEP_ORDER = ['select', 'upscale', 'threshold'] as const;
export type FilterSubstep = (typeof FILTER_SUBSTEP_ORDER)[number];
```

and a **required** field on `TimingReport`:

```ts
export interface TimingReport {
  phases: Record<SegmentationPhase, PhaseTiming>;
  filterSubPhases: Record<FilterSubstep, PhaseTiming>;
  totalMs: number;
}
```

`PHASE_ORDER`, `SegmentationPhase` and `SegmenterFailure.phase` are **not**
touched. Three considered alternatives and why they lost:

- Adding `filter:select` / `filter:upscale` / `filter:threshold` to
  `PHASE_ORDER` is the cheapest change and renders in the playground table for
  free, but `SegmentationPhase` is public API exported from
  `note-scanner/segmenter`; it would widen `SegmenterFailure.phase` to values
  that can never be thrown, and anyone summing the table's total column would
  count `filter` twice. The document this feeds is a percentage breakdown, so a
  double-counting total is exactly the wrong footgun.
- A generic `subPhases: Partial<Record<SegmentationPhase, Record<string,
  PhaseTiming>>>` would future-proof #6 (`nms`) and #8 (`mask-encode`), but it
  is an abstraction with one call site today. YAGNI.

The field is **required, not optional**, deliberately: `createSegmenter.ts`
rebuilds the `TimingReport` as an object literal (around line 132) to splice in
the main-thread `mask-encode` phase. Making the field required means the
compiler, not a reviewer, catches a dropped passthrough across the worker
boundary.

## `src/segmenter/core/timing.ts`

`TimingAccumulator` gains one method, backed by a second `Map`:

```ts
export interface TimingAccumulator {
  record(phase: SegmentationPhase, ms: number): void;
  recordFilterSub(step: FilterSubstep, ms: number): void;
  report(totalMs: number): TimingReport;
}
```

`report()` zero-fills every entry of `FILTER_SUBSTEP_ORDER` exactly as it
already zero-fills `PHASE_ORDER`, so the sub-table has a stable shape and a
sub-step that never ran reads as "0 samples" rather than as a hole. Reuse
`summarizePhase` unchanged — it already returns both `total` and `p50`, which
are the two figures the issue asks for.

Do not introduce a generic `createSampleAccumulator<K>`: one extra `Map` and one
extra method inside the existing factory is ~15 lines and keeps a single object
owning the whole report.

## `src/segmenter/worker/segmenter.worker.ts`

Three timer regions inside the existing `filter` stage, drawn to be
**exhaustive and non-overlapping** so they sum to the stage total. Region
boundaries against the current source:

| Sub-step | Region | Resolution |
|---|---|---|
| `select` | stage start (the `dims` reads) through the end of the best-of-3 `stabilityScore` loop | 256x256 |
| `upscale` | the `selected` `Float32Array` gather **and** `await processor.post_process_masks(...)` | 256x256 -> full |
| `threshold` | the `thresholdMask` + `minMaskArea` loop | full res |

The gather is folded into `upscale` rather than timed separately or left out.
§3.3 defines step 2 as `post_process_masks` alone, but timing exactly that
leaves the gather in an unattributed residual, and a residual is precisely what
muddies the decision the measurement exists to make ("is step 2 materially
below ~75%?"). The gather is ~1.2 MB of memcpy per batch and will read
sub-millisecond, so folding it in barely perturbs the comparison against the
estimate while guaranteeing the three sum to the stage.

`upscale` and `threshold` record **only when `chosen.length > 0`**. A
zero-survivor batch executes neither region, so the sum invariant still holds,
and recording a 0 ms sample would drag their `p50` down misleadingly — `p50` is
exactly what the issue asks for.

The existing `timings.record('filter', ...)` call stays exactly as it is. Do not
change what the `filter` phase measures; the sub-steps nest inside it.

## `src/segmenter/createSegmenter.ts`

Carry `filterSubPhases` through the object literal that rebuilds the report.
No other change; the main thread contributes nothing to `filter`.

## `playground/SegmentView.tsx`

Render the three sub-steps as rows nested under `filter` in the existing
results table, driven by `FILTER_SUBSTEP_ORDER` the same way the phase rows are
driven by `PHASE_ORDER`. Indent the label (e.g. a tree prefix) so a reader can
see at a glance that they are a breakdown of `filter` and not peers of it.

`pointsPerSide` 16 and 32 are already the two `POINTS_PER_SIDE_CHOICES`, so both
operating points the issue calls for are reachable with no further change. The
instrumentation must be operating-point independent — nothing keyed to a
particular `pointsPerSide` or `batchSize`.

## Testing

`src/segmenter/core/timing.test.ts`:

- every entry of `FILTER_SUBSTEP_ORDER` is present and zero-filled when nothing
  was recorded, mirroring the existing `PHASE_ORDER` zero-fill test;
- repeated `recordFilterSub` samples accumulate, and the reported `total` and
  `p50` are correct;
- the three sub-step totals sum to a separately recorded `filter` total, which
  is the no-residual invariant stated above.

`src/segmenter/createSegmenter.test.ts`:

- `filterSubPhases` survives the worker -> main-thread boundary with `total` and
  `p50` intact. This seam is the closest thing to the issue's first acceptance
  criterion that is unit-testable, since the worker itself imports
  `@huggingface/transformers` and needs a GPU.

The worker's own three regions are not directly unit-testable for that reason;
they are covered by review against the region table above.

## Out of scope

- Producing the actual figures at `pointsPerSide` 16 and 32 on real hardware.
- Recording them in issue #1 §3.3.
- The conditional re-scope of #6 and #8.
- Any change to what `filter`, `nms` or `mask-encode` measure.
- Sub-timers for any stage other than `filter`.

## Acceptance criteria

Every criterion below is `(non-ui)`. This repository has no `@playwright/test`
dependency and its vitest environment is `node`, so nothing here is
browser-verifiable by the pipeline — including AC5, which is verified by
reading `playground/SegmentView.tsx` rather than by opening a browser.

- AC1 (non-ui) — a run emits a `total` and a `p50` for each of `filter`'s three
  sub-steps: `TimingReport.filterSubPhases` carries a `PhaseTiming` for
  `select`, `upscale` and `threshold`.
- AC2 (non-ui) — `filterSubPhases` has an entry for every member of
  `FILTER_SUBSTEP_ORDER` even when nothing was recorded, zero-filled, so a
  sub-step that never ran reads as "0 samples" rather than as a missing key.
- AC3 (non-ui) — the three sub-step totals sum to the `filter` phase total with
  no residual: the timed regions are exhaustive and non-overlapping across the
  stage.
- AC4 (non-ui) — `filterSubPhases` survives the worker -> main-thread boundary
  with `total` and `p50` intact, i.e. `createSegmenter`'s rebuilt report carries
  the worker's values through unchanged.
- AC5 (non-ui) — the playground results table renders `select`, `upscale` and
  `threshold` as visibly nested rows under `filter`, driven by
  `FILTER_SUBSTEP_ORDER`, and does so identically at both operating points
  (`pointsPerSide` 16 and 32) because the instrumentation is keyed to neither
  `pointsPerSide` nor `batchSize`.
- AC6 (non-ui) — **deferred to the human; satisfied outside this branch and not
  a gate on this PR.** Real figures for the three sub-steps at `pointsPerSide`
  16 and 32 are recorded on issue #1, replacing the estimates in §3.3.
  Producing them needs a WebGPU browser downloading SlimSAM weights and running
  ~100-170 s per operating point on representative hardware, which this
  pipeline cannot do.
- AC7 (non-ui) — **deferred to the human; satisfied outside this branch and not
  a gate on this PR.** If the measured step 2 (`upscale`) is materially below
  ~75% of the `filter` stage, issues #6 and #8 are re-scoped before they are
  started. This depends on AC6's figures and so cannot be settled here.
