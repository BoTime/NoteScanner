# F1 + N1 + F3 — carry 256x256 through filter and NMS

Reorder the worker's mask pipeline so full resolution is reached only by masks
that survive NMS: retain each chosen candidate as its 256x256 logit window plus
a 256x256 binary mask, run `dedupeMasks` on that low-res coverage, and resample
to full resolution once per survivor, fused with the PNG encode.

Implements GitHub issue #6 (F1 + N1 + F3). The keystone of issue #1's Tier 1:
it is the change that unblocks #7 (M2, encode at 256x256).

## What is true today (post #8/F2 and #19/decode-sweep)

The worker's batch loop, per batch:

1. `select` — `stabilityScore` + best-of-three per prompt point, on the
   decoder's native 256x256 logits. Cheap.
2. `resample` — `resampleThresholdMask` takes EVERY chosen candidate's low-res
   logit window straight to a full-resolution binary mask, fusing the bilinear
   resample and the threshold in one pass (F2, already landed). The full-res
   `minMaskArea` gate is applied here.

Then, once, across all batches: `dedupeMasks(candidates, iouThreshold,
originalWidth)` runs at FULL resolution, and `mask-encode` writes the survivors
as 1-bit PNGs.

So every candidate is carried at full resolution from the moment it is chosen,
and ~94% of that work is discarded by NMS (issue #1 measured 617 -> 35 at 32
points per side).

## The change

**In the batch loop**, replace the per-candidate full-res resample with:

- copy the chosen 256x256 logit window into a retained `Float32Array`
  (`predMasks.data` is reused per batch, so a `subarray` view cannot be held);
- binarize it at 256x256 with the existing `thresholdMask`;
- apply a SCALED area gate (below) and push a candidate carrying
  `{ logits, lowMask }`.

**Then** `dedupeMasks(lowMasks, iouThreshold, lowWidth)` — NMS on 256x256
coverage (N1). IoU at 256x256 is not an approximation of the signal: 256x256 is
the resolution the decoder emits, and everything above it is interpolation that
adds no information.

**Then** release every rejected candidate's logits, and for each survivor, in
one loop: `resampleThresholdMask` to full resolution -> apply the exact
full-res `minMaskArea` -> `encodeMaskPng` -> discard the coverage.

The only exception to discarding the coverage is `keepRawMasks`, which must go
on working: when it is set the survivor's full-resolution coverage is retained
in that same loop instead of being dropped, so `SegmentationResult.rawMasks`
carries exactly what it carries today. This is not incidental — the Compare
tab's `compareMaskSets` reads those buffers, and without them the kept-set
delta (AC1) cannot be measured at all.

### Memory: this is a saving, not a cost

A retained candidate is 256 KB of logits + 64 KB of low-res coverage = **320 KB,
independent of image size**. Today a candidate holds full-resolution coverage:
`W*H` bytes — 664 KB each on the 1024x649 playground sample, and 12.2 MB each on
a 12 MP photo, which at 617 candidates is ~7.5 GB and does not run at all.
Fusing the survivor resample with the encode also keeps peak FULL-resolution
memory at one mask rather than one per survivor.

### `minMaskArea` keeps its exact documented meaning (F3)

Two gates, deliberately:

- **Before NMS**, at 256x256:
  `lowMin = max(1, round(minMaskArea * lowPixels / (originalWidth * originalHeight)))`.
  This is where F3's saving now lives — fewer NMS candidates and fewer retained
  logit windows.
- **After the survivor resample**, at full resolution: the true `minMaskArea`,
  unchanged.

The low-res gate is coarse and image-size dependent (on the 1024x649 sample one
low-res pixel is ~10.1 full-res pixels, so `minMaskArea: 100` scales to ~10; on
a 12 MP photo the same option scales to 0.54 and floors to 1). The full-res
re-check is what keeps the RETURNED set exact against the option's documented
meaning, so the coarseness costs performance rather than correctness.

Admitting borderline masks into NMS cannot change any larger mask's fate:
`dedupeMasks` is greedy largest-area-first, so a smaller candidate is only ever
considered after every larger one and can never suppress one.

### Kept masks are bit-identical, by construction

A survivor is resampled by the SAME `resampleThresholdMask` call, with the same
arguments, that it would have received today. So for any mask both paths keep,
coverage is byte-for-byte equal. This change can alter WHICH masks are kept —
never what a kept mask looks like. That is the property the browser check
asserts, and it is what makes AC2 falsifiable instead of an eyeball.

The genuine behaviour deltas, both to be quantified rather than assumed:

1. IoU computed at 256x256 can flip a borderline dedupe decision.
2. The low-res area gate can drop a thin mask whose full-res area would have
   passed.

**This change is explicitly NOT behaviour-preserving**, unlike #4. The kept-set
delta is a number this run has to produce and then judge, not a risk to wave
past.

## Timing model

`PHASE_ORDER` gains **`resample`**, between `nms` and `mask-encode` — the
survivor resample is now its own stage rather than part of `filter`, so AC3's
before/after on `filter` and `nms` reads honestly instead of hiding the moved
cost. It is a legitimate `SegmenterFailure.phase`: `resampleThresholdMask`
throws on bad dimensions.

`FILTER_SUBSTEP_ORDER` becomes `['select', 'threshold']` — the low-res binarize
plus area gate replaces the `resample` substep, and the two still sum to the
`filter` total (the invariant `timing.test.ts` already asserts).

`SegmentationCounts` gains a count for the full-res area re-check, so the funnel
(`raw` -> `afterFilter` -> `afterNms` -> returned) stays readable: `afterNms` is
no longer the returned count, and a silently shrinking result set must be
visible in the counts rather than inferred from the segment list.

## How the delta gets measured — reuse, do not rebuild

The new path is a `SegmenterOptions` flag, exactly as `overlapDecodeFilter` and
`gpuResidentEmbeddings` are:

```
lowResFilterNms: boolean   // default TRUE — this is the shipped pipeline
```

It is the one flag in `DEFAULT_SEGMENTER_OPTIONS` that defaults `true`: the
low-res pipeline is what ships, and the old path exists only as a measurement
baseline. Turning it OFF restores the pre-change full-resolution path,
preserved for the same reason `dedupeMasksReference` is: so the kept-set delta
is measured rather than argued about. This is the one place the old code stays.

Like `keepRawMasks`, and unlike `gpuResidentEmbeddings`, it changes nothing
about the ONNX sessions and must NOT enter the worker's session cache key.

**No new comparison machinery.** `playground/compare.ts` already has
`compareMaskSets` — greedy best-IoU pairing reporting matched, unmatched on each
side, and mean/median/min IoU over the matched pairs, built on core's single
`pairwiseIoU` — and `keepRawMasks` already exists to supply the coverage
buffers. The Compare tab A/Bs the flag with what is already there, and the
numbers land in `docs/measurements/` following the convention #19 established.
The flag becomes a Compare-tab control with a `data-testid` beside the existing
decode-path checkboxes, and is reachable from the sweep's row options so
`scripts/sweep-decode.mjs` can walk a before/after pair in one run — without
widening the default grid.

Deliberate deviation, recorded: the brainstorm settled on a bespoke in-worker
`compareLowRes` A/B mirroring `compareNms`. PR #19 landed a strictly better
mechanism for the same intent after that answer was given. An in-worker A/B
would now be a second copy of `compareMaskSets`, which the learnings explicitly
warn against. The intent — a measured A/B on a real image, not an assertion —
is unchanged.

## Browser verification (AC2)

`resampleThresholdMask`, `thresholdMask` and `dedupeMasks` are pure and
model-free, so the boundary invariant can be checked deterministically with NO
WebGPU, NO model download and NO network.

Add a **Boundary** tab to the playground that drives both paths over committed
PROCEDURAL 256x256 logit fixtures — generated in code as smooth signed-distance
fields, not committed binaries — chosen to stress exactly what changed:

- a fine-toothed comb (thin tines: the fine structure the issue's boundary
  criterion names);
- a shape with a one-pixel bridge;
- a near-duplicate pair straddling the IoU threshold (stresses N1);
- a speck whose full-res area passes `minMaskArea` but whose low-res area
  rounds below `lowMin` (stresses F3).

It renders each fixture's baseline and new mask with a difference overlay and
numeric readouts, each addressable by `data-testid`. The Playwright spec
asserts the invariant: for every fixture kept by both paths the coverage
difference is exactly zero, and the comb fixture is kept by both. Because the
spec drives the running playground rather than encoding fixtures in Node like
`tests/browser/mask-png.spec.ts` does, the Playwright config gains a
`webServer` that serves the playground; the tab itself needs no adapter, so it
runs headless in CI.

Real-image numbers (AC1, AC3) are taken on a real GPU through the existing
Compare tab and `scripts/sweep-decode.mjs`, by hand, and committed under
`docs/measurements/`.

## Testing

Vitest: the scaled-threshold arithmetic including the ratio where it floors to
1; both area gates; that rejected candidates' logits are released; the funnel
counts; that the retained logit window is a copy and not a view into the reused
`pred_masks` buffer; and that `lowResFilterNms: false` reproduces the previous
kept set. Playwright: the boundary invariant, in a real engine.

## Out of scope

Encoding masks at 256x256 (that is #7/M2, which this unblocks), any renderer
change, and any change to `nmsIouThreshold`'s default.

## Acceptance criteria

AC2 is the run's only `(ui)` criterion, and it is browser-checkable precisely
because the Boundary tab needs no WebGPU, no model download and no network.
AC1, AC3 and AC4 need a real GPU on a real image: they are taken by hand
through the Compare tab and `scripts/sweep-decode.mjs` and committed under
`docs/measurements/`, and the `verify` stage checks them by reading those
committed artifacts, not by opening a browser.

- AC1 (non-ui) — the kept-mask set is compared against the pre-change baseline
  on a real image and the delta is QUANTIFIED, not assumed: a committed
  `docs/measurements/` artifact records, for `lowResFilterNms` on vs off at the
  same options, the returned mask count on each side, `compareMaskSets`'
  matched / unmatched-baseline / unmatched-new counts, and mean/median/min IoU
  over the matched pairs — and states explicitly, in prose, whether the delta
  is acceptable and why. A record that reports only that a measurement
  happened does not satisfy this.
- AC2 (ui) — in the playground's Boundary tab, for every procedural 256x256
  fixture that BOTH paths keep, the rendered difference between the baseline
  mask and the new mask is exactly zero pixels, and the fine-toothed comb
  fixture is kept by both paths; the tab also shows, per fixture, each path's
  kept/dropped status and area, so a fixture the new path drops is visible
  rather than silently absent. The tab renders with no WebGPU adapter, no model
  download and no network.
- AC3 (non-ui) — the `filter` and `nms` phase totals are recorded before and
  after the change at BOTH operating points (`pointsPerSide` 16 and 32) on the
  same image and machine, committed under `docs/measurements/` alongside the
  new `resample` phase total, and the record states explicitly whether the
  result is an acceptable outcome — including where time moved to rather than
  disappeared.
- AC4 (non-ui) — #2's sub-timers (`FILTER_SUBSTEP_ORDER`) have landed and still
  hold after this change: the order is `['select', 'threshold']`, both
  substeps are reported, and their totals still sum to the `filter` phase
  total.
- AC5 (non-ui) — `SegmenterOptions` carries `lowResFilterNms`, `true` in
  `DEFAULT_SEGMENTER_OPTIONS`, surviving `createSegmenter` into the worker
  request; setting it `false` reproduces the pre-change kept set exactly; and
  it does not appear in the worker's ONNX session cache key.
- AC6 (non-ui) — with the flag on, a chosen candidate is retained as a COPY of
  its 256x256 logit window: mutating or reusing the decoder's `pred_masks`
  buffer after the batch cannot change a retained candidate's logits or its
  resulting mask.
- AC7 (non-ui) — the pre-NMS gate is
  `max(1, round(minMaskArea * lowPixels / (originalWidth * originalHeight)))`,
  including the case where the ratio rounds below 1 and floors to 1, and NMS
  runs on the 256x256 coverage with the low-res width.
- AC8 (non-ui) — after NMS, each survivor is resampled once to full resolution
  and re-checked against the exact, unscaled `minMaskArea`, so the RETURNED set
  matches the option's documented full-resolution meaning; a mask that passes
  the low-res gate but fails the full-res area is not returned.
- AC9 (non-ui) — every rejected candidate's retained logits are released before
  the survivor loop runs, and full-resolution coverage exists for at most one
  mask at a time in that loop, except that with `keepRawMasks` set each
  survivor's coverage is retained and surfaced on
  `SegmentationResult.rawMasks` exactly as it is today.
- AC10 (non-ui) — `PHASE_ORDER` contains `resample` between `nms` and
  `mask-encode`, the survivor resample's time is recorded there and not in
  `filter`, and a `resampleThresholdMask` throw during that stage surfaces as a
  `SegmenterFailure` whose `phase` is `resample`.
- AC11 (non-ui) — `SegmentationCounts` reports the full-res area re-check, so
  the funnel reads `raw` -> `afterFilter` -> `afterNms` -> returned, and the
  returned count is distinguishable from `afterNms` when the re-check drops a
  mask.
- AC12 (non-ui) — the delta is measurable with the machinery that already
  exists: the Compare tab exposes a `lowResFilterNms` control with a
  `data-testid`, the flag is reachable from the sweep's row options so
  `scripts/sweep-decode.mjs` can walk a before/after pair, and no second
  implementation of mask-set comparison or IoU is added anywhere.
