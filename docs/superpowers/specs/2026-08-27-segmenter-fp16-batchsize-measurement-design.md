# Spec — issue #3 (2/8): E1 + D1, fp16 and a larger batchSize

**Issue:** [#3 — 2/8 — E1 + D1: fp16 and a larger batchSize](https://github.com/BoTime/NoteScanner/issues/3)
(parent: #1 · Wave 1 · Tier 3 · depends on nothing)
**Date:** 2026-08-27

## What this run delivers

The **measurement instrument**, not the measured numbers.

The issue's acceptance criteria are GPU measurements: `encode` + `decode`
timings at 16 and 32 points-per-side, a paired fp16-vs-fp32 mask agreement
check, and a `batchSize` curve. None of them can be produced by an unattended
pipeline — they need WebGPU, a real GPU adapter, and a ~40 MB ONNX weight
download. This repository has no `@playwright/test` and no browser-verification
harness of any kind, and its CI runs headless Ubuntu with no WebGPU adapter.

So this run builds the harness that produces those numbers, and the developer
runs it in a real browser and pastes the results into issue #3. Nothing in this
run is browser-observable by the pipeline, which is why every criterion below is
tagged `(non-ui)`.

**`DEFAULT_SEGMENTER_OPTIONS` is deliberately NOT changed.** AC3 requires the
recommended `batchSize` to be chosen from measurement, and the measurement
happens after this lands. Flipping `dtype` to `'fp16'` or raising `batchSize`
now would be choosing on the issue's estimate, which is the thing AC3 rules out.

## Why the harness has the shape it does

From the measured runs recorded in parent issue #1 (`batchSize: 8`, fp32):

| | 16 pps (Run A) | 32 pps (Run B) |
|---|---:|---:|
| `encode` + `decode` — what E1/D1 change | 9.0 s | 23.2 s |
| `filter` + `nms` + `mask-encode` — untouched here | 89.2 s | 143.2 s |
| total per run | 98.4 s | 166.4 s |

About 90% of every run is stages this issue does not touch. That is why the
matrix is kept to seven rows and run on demand rather than swept exhaustively:
the four dtype rows satisfy AC1 and AC2 in ~9 minutes, and three further rows
sketch the `batchSize` curve for AC3.

## Component 1 — the one package change

`src/segmenter/core/types.ts`:

- `SegmenterOptions` gains `keepRawMasks: boolean`, defaulting to `false` in
  `DEFAULT_SEGMENTER_OPTIONS`.
- `SegmentationResult` gains `rawMasks?: RawMask[]`.

`src/segmenter/createSegmenter.ts`:

- When `keepRawMasks` is true, the resolved result carries the `RawMask[]` the
  worker already posted; when false, the field is absent and the buffers stay
  eligible for collection as they are today.

`keepRawMasks` is a **client-side retention flag, not an inference parameter**.
The worker ignores it — it always posts masks — and it must not participate in
the worker's session cache key. Document it as such at the field, because it
sits in a bag whose every other member is a model knob.

Off by default because `SegmentView` holds its `SegmentationResult` in React
state; at 16 pps a kept set is ~50 full-resolution coverage arrays at ~0.7 MB
each, so retaining them unconditionally would pin tens of megabytes for the
lifetime of the view.

## Component 2 — `playground/compare.ts` (new, pure, unit-tested)

Playground-local by decision: nothing beyond the `keepRawMasks` opt-in is added
to the published `/segmenter` subpath. This file mirrors how `benchmark.ts`
already sits beside `BenchmarkView.tsx`.

- **`COMPARE_ROWS`** — the seven presets, as a module constant:

  | # | dtype | pointsPerSide | batchSize | serves |
  |---|---|---:|---:|---|
  | 1 | fp32 | 16 | 8 | AC1 baseline, AC2 baseline |
  | 2 | fp16 | 16 | 8 | AC1, AC2 |
  | 3 | fp32 | 32 | 8 | AC1 baseline, AC2 baseline |
  | 4 | fp16 | 32 | 8 | AC1, AC2 |
  | 5 | fp16 | 16 | 32 | AC3 curve |
  | 6 | fp16 | 16 | 64 | AC3 curve |
  | 7 | fp16 | 32 | *chosen* | AC3 confirmation |

  Row 7's `batchSize` is supplied by the caller (the view offers 32 or 64,
  defaulting to 32) — it is the confirmation of the pick AC3 asks the developer
  to make from rows 5 and 6. Every other row is fixed.

  Rows 1-4 are the pair rows: they run with `keepRawMasks: true`. Rows 5-7 do
  not, since `batchSize` cannot change the output — it only changes how many
  prompt points ride in each model call.

- **`runRow(row, deps)`** — `deps` carries the `segment` function, so tests
  drive it with a stub and never touch WebGPU. Returns a result row holding the
  per-phase totals (`encode`, `decode`, `filter`, `nms`, `mask-encode`), the
  `encode + decode` subtotal that the criteria actually care about, the overall
  total, and `SegmentationCounts`.

- **`compareMaskSets(baseline, variant)`** — greedy best-IoU pairing built on
  the existing `pairwiseIoU` exported from `src/segmenter/core`. Do not write a
  second IoU implementation; import that one. Each baseline mask takes its
  highest-IoU unclaimed partner in the variant set; a pair below a floor counts
  as unmatched on both sides. Returns baseline count, variant count, matched
  count, unmatched counts either side, and the matched-IoU distribution (mean,
  median, min).

- **`toMarkdown(rows, agreement)`** — renders the results as a markdown table
  the developer pastes into issue #3, including the fp16-vs-fp32 agreement
  summary for the pairs that have run.

## Component 3 — `playground/CompareView.tsx` (new, thin)

Renders `COMPARE_ROWS` up front as "not run". Per-row Run, plus Run all. Live
progress from the segmenter's existing `onProgress`. An agreement panel for the
1-2 and 3-4 pairs once both halves of a pair have results. A "copy as markdown"
button over `toMarkdown`.

A row whose run rejects with `SegmenterFailure` records the failing phase and
message in that row and leaves every other row runnable — one bad row must not
abort the sweep.

Where `navigator.gpu` is absent, it shows the same WebGPU-required panel the
Segment view shows and spawns no worker.

Image source defaults to the committed `playground/sample/cafe-table.jpg`, the
same image the Segment view preloads, so results stay comparable across runs;
the developer can substitute their own file the same way the Segment view
allows.

## Component 4 — small additions

- `playground/SegmentView.tsx` gains `dtype` and `batchSize` selects, following
  the existing control pattern (labelled `<select>` with a `data-testid`), for
  one-off pokes outside the matrix.
- `playground/main.tsx` gains a third tab, `Compare`, beside Benchmark and
  Segment.
- `README.md` gains a short note about the Compare tab under the `/segmenter`
  section.

## Data flow

Compare view holds the loaded image. Running a row: fetch the image URL to a
blob, `createImageBitmap`, then `segment(bitmap, {...row, keepRawMasks})`. The
bitmap is transferred to the worker and consumed, so each row creates its own.
`createSegmenter` already terminates and respawns the worker per call, so every
row pays its own `model-load`; that phase is reported separately and is excluded
from the criteria's budget, exactly as parent issue #1 excludes it.

Results accumulate in state keyed by row id. Raw masks are retained only for
rows 1-4, and a row's previous raw masks are released when it is re-run.

## Testing

vitest only — no browser, no WebGPU, no Playwright.

- `playground/compare.test.ts`: the row presets (count, the axes each row
  covers, row 7 honouring the supplied `batchSize`); `runRow` against a stub
  segmenter with fixed timings, asserting the `encode + decode` subtotal and
  that `keepRawMasks` is requested for rows 1-4 and not for 5-7;
  `compareMaskSets` on synthetic coverage arrays — identical sets match at IoU
  1, a dropped mask leaves an unmatched baseline, a shifted mask yields IoU
  below 1, and empty sets do not divide by zero; `toMarkdown` output shape.
- `src/segmenter/createSegmenter.test.ts` gains cases that `rawMasks` is
  populated when `keepRawMasks` is true and absent when it is false.

## Acceptance criteria

Every criterion is tagged `(non-ui)`: this repository has no `@playwright/test`
and no browser-verification harness, and its CI runs headless Ubuntu with no
WebGPU adapter, so nothing in this run is browser-observable by the pipeline.
The issue's criteria are satisfied by the instrument this run builds, not by
numbers this run produces — the numbers require a GPU browser session a human
drives. Each criterion below is checkable by `npm run typecheck && npm test`.

- AC1 (non-ui) — `COMPARE_ROWS` defines rows covering both `dtype: 'fp32'` and
  `dtype: 'fp16'` at both 16 and 32 points-per-side, and `runRow` reports an
  `encode + decode` subtotal alongside the per-phase totals for each row it
  runs. (From the issue's "`encode` + `decode` phase timings recorded
  before/after, at both 16 and 32 pps".)
- AC2 (non-ui) — `compareMaskSets` computes kept-mask counts for both sides and
  a per-mask IoU distribution (mean, median, min) over a greedy best-IoU
  pairing between a paired fp32 baseline run and its fp16 counterpart, with a
  stated IoU floor below which a pair counts as unmatched on both sides; the
  pairing is unit-tested on synthetic coverage arrays. (From the issue's "fp16
  does not change the output: kept-mask count and per-mask IoU against the fp32
  baseline, on the same image, within a stated tolerance.")
- AC3 (non-ui) — `COMPARE_ROWS` defines rows that sweep `batchSize` at a fixed
  `dtype` and points-per-side, plus a confirmation row whose `batchSize` the
  developer supplies from what those sweep rows measured. (From the issue's "a
  recommended `batchSize` chosen from measurement, not assumption".)
- AC4 (non-ui) — `DEFAULT_SEGMENTER_OPTIONS` is unchanged by this run: `dtype`
  remains `'fp32'` and `batchSize` remains `8`, because AC3 requires the
  recommendation to come from measurement that has not happened yet. (Settled
  in brainstorm; implicit in the issue.)
- AC5 (non-ui) — a `SegmentationResult` carries `rawMasks` when the run
  requested `keepRawMasks: true` and omits the field when it did not, so the
  AC2 comparison has masks to compare without retaining them for every run.
- AC6 (non-ui) — `npm run typecheck && npm test` both pass.

## Out of scope

- Changing any default in `DEFAULT_SEGMENTER_OPTIONS`.
- Any change to the worker's inference path, including a timing-only mode.
- Adding `@playwright/test` or any browser-driven verification.
- The other issues in the #1 breakdown (#2 sub-timers, #4 NMS work).
