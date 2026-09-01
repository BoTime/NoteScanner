# M2 — encode masks at the decoder's own resolution

Implements GitHub issue #7 (M2). Depends on #6 (hard) and #5, both landed:
#6 is what makes 256x256 logits exist at the encode step, and #5 is the 1-bit
PNG writer this change points at a smaller target.

## Problem

`segmenter.worker.ts:450` encodes every surviving mask at full image
resolution:

    const maskUrl = await encodeMaskPng(mask.coverage, originalWidth, originalHeight);

After #6 the decoder's real output is a 256x256 logit grid, of which only the
top-left `(reshapedWidth * lowWidth / padWidth) x (reshapedHeight * lowHeight /
padHeight)` window ever maps onto image pixels. For a 1024x649 photo that is
256 x 162 samples — 41,472 — being written out as 664,576 pixels. The encode is
paying a 16x interpolation for information the model never produced.

Measured `mask-encode` totals to beat: ~1.6 s at 16 points per side, ~1.2 s at
32. Target ~0.2 s / ~0.1 s.

## Two facts that shrink the issue's stated scope

1. **`encodeMaskPng(coverage, width, height)` already accepts a target size.**
   The issue's second scope bullet ("the 1-bit PNG writer needs to accept a
   target size") is already satisfied. The hardcode is in the caller, not the
   writer. `mask-encode.ts` needs no change.
2. **There is exactly one mask-PNG consumer**, `buildMaskData` at
   `SegmentViewer.tsx:79`, and it already does
   `ctx.drawImage(img, 0, 0, imageWidth, imageHeight)` — it rescales any
   intrinsic size up to image space before reading pixels back. Hit-testing,
   the bright-window union and the outline paths all operate on that
   image-space coverage and need no coordinate-scaling change. The issue's
   third scope bullet is a confirmation, not a rewrite.

## Chosen approach

Resample survivors a **second time, straight from the retained logits**, at the
encode size.

`resolveSurvivor` keeps doing its full-resolution resample unchanged, because
two things depend on it: the exact, unscaled `minMaskArea` re-check (a contract
#6 deliberately protected — "the RETURNED set is what the option documents")
and the `area` reported on `EncodedMask`. Alongside it, one more
`resampleThresholdMask` call on the *same* retained logits, at the encode size,
produces the coverage the PNG is written from.

Cost: 41k extra pixels against the 664k already being resampled, ~6% on the
`resample` phase, in exchange for turning a 664k-pixel PNG encode into a
41k-pixel one.

Rejected alternatives:

- **Decimate the full-resolution coverage inside the PNG writer.** Avoids the
  second resample but downsamples an already-binarised mask, which is exactly
  how thin structures vanish. AC2 tests thin structures specifically.
- **Drop the full-resolution resample entirely and scale the area gate.** The
  largest win, but it changes the returned kept set. That is a different issue,
  not M2.

## Design

### Encode target

`createFilterPlan` gains `encodeWidth` / `encodeHeight`, derived once per run as
the aspect-correct logit window clamped to the image:

    encodeWidth  = min(round(lowWidth  * reshapedWidth  / padWidth),  originalWidth)
    encodeHeight = min(round(lowHeight * reshapedHeight / padHeight), originalHeight)

One output pixel per decoder sample: nothing invented, nothing redundant. The
clamp is what stops a photo smaller than the window receiving an *upscaled*
mask — without it a 200x150 photo would get a mask larger than full resolution.

Note that this is the aspect-correct window, not a square 256x256, so on the
1024x649 sample the reduction is ~16x rather than the issue's estimated 10.7x.
The criterion the issue was stating — hit-testing survives a large pixel-count
reduction — is what carries over; its estimated constant does not.

### Option

`SegmenterOptions.lowResMaskEncode`, **default true**, mirroring
`lowResFilterNms`. When off, the encode target resolves to full resolution and
the path is byte-for-byte today's. It selects a resample target only, so — like
`keepRawMasks` — it must NOT enter the worker's ONNX session cache key.

### Flag interaction, stated explicitly

The encode resample reads the retained logits, which exist only on the
`lowResFilterNms: true` path (on the baseline path `MaskCandidate.logits` is
null by construction). So `createFilterPlan` forces the encode target to full
resolution whenever `lowResFilterNms` is false. Retaining logits there would add
~320 KB per candidate to a path that already holds full-resolution coverage, on
a path that exists only for measurement. AC1's before/after pair holds
`lowResFilterNms` at its shipped `true` and varies only `lowResMaskEncode`,
which is the honest comparison anyway. Document this in `mask-pipeline.ts`'s
module comment beside the existing description of the two pipelines.

### Where the time is counted

The second resample is timed into **`resample`**, not `mask-encode`.
`mask-encode` then shows the pure encode saving and `resample` grows ~6%.
Splitting it the other way would hide moved cost inside the very number this
issue claims — the failure mode `PHASE_ORDER`'s own comment was written to
prevent ("Keeping it out of `filter` is what makes a before/after read honestly
instead of hiding the moved cost").

### Viewer

One line in `buildMaskData`, before the `drawImage`:

    ctx.imageSmoothingEnabled = false;

Nearest-neighbour is both the cheaper upscale — 1 tap per output pixel instead
of bilinear's 4, and this canvas is created `{ willReadFrequently: true }` so it
is genuinely CPU-backed — and the one that keeps every read-back pixel exactly
`(255,255,255,255)` or `(0,0,0,0)`. That means the `alpha > 0 && red > 0`
coverage predicate and the contract it shares with the preview crop and the
server-side crop stay untouched. The accepted cost: boundaries quantise to
4px steps, which AC2 measures rather than assumes.

### Harness

`lowResMaskEncode` joins the Compare tab's controls and `compare.ts`'s axis
list beside `lowResFilterNms`, with the same validation shape, and gets a sweep
config so one headed real-GPU run produces all four AC1 rows
(16/32 pps x on/off) in a single session. One session matters: the decode-sweep
report already had to disclose a cold-fp16-row artifact caused by exactly the
cross-session drift that differencing two separate runs would reintroduce.

### Testing

- Unit tests for the encode-size derivation: the clamp, a landscape photo, a
  portrait photo, and the square case.
- The #5 round-trip property tests extended to the new dimensions, keeping the
  width-not-a-multiple-of-8 case — 162 and 649 both exercise it.
- A browser test that clicks near image edges and on a thin structure and
  asserts the same segment id is selected as at full resolution.
- The measurement lands as a dated report under `docs/measurements/`.

### Docs

`docs/pipeline.md` describes the encode step and will be falsified by this
change. The learnings doc's headline rule from the last run is precisely that
this file gets missed — it goes on the file list.

## Files expected to change

- `src/segmenter/core/mask-pipeline.ts` — plan fields, encode-size derivation,
  the survivor encode resample, module comment.
- `src/segmenter/core/types.ts` — `lowResMaskEncode` option + default.
- `src/segmenter/worker/segmenter.worker.ts` — call the new resample; keep the
  session cache key unchanged.
- `src/SegmentViewer.tsx` — `imageSmoothingEnabled = false`.
- `playground/CompareView.tsx`, `playground/compare.ts`,
  `playground/option-choices.ts` — the new axis.
- `scripts/sweep-decode.mjs` / a sweep config under `docs/measurements/`.
- `docs/pipeline.md` — encode step description.
- Tests alongside each.

## Out of scope

Changing which masks are returned (dropping the full-resolution resample and
scaling the area gate is a different issue), any renderer change beyond the one
`imageSmoothingEnabled` line, and any change to `mask-encode.ts` itself.

## Acceptance criteria

AC2 is the run's only `(ui)` criterion that needs a click: hit-testing is the
behaviour most at risk from shrinking the mask, and it is confirmable by a
person in the running app. AC1 needs a real GPU on a real image — it is taken
by hand through the sweep runner and committed under `docs/measurements/`, and
is checked by reading that committed artifact rather than by opening a browser.

- AC1 (non-ui) — the `mask-encode` phase total is recorded before and after
  the change at BOTH operating points (`pointsPerSide` 16 and 32) on the same
  image and machine, in a single session, committed under
  `docs/measurements/`; the record also carries the `resample` phase total on
  each side so moved cost is visible rather than hidden, and states explicitly
  in prose whether the result is an acceptable outcome. A record that reports
  only that a measurement happened does not satisfy this.
- AC2 (ui) — in the running app, with masks encoded at the reduced size,
  clicking near the image edges (all four) and on a thin structure selects the
  same segment as the same click does with `lowResMaskEncode` off at full
  resolution.
- AC3 (non-ui) — the #5 round-trip property tests pass at the new encode
  dimensions, including the width-not-a-multiple-of-8 case.
- AC4 (non-ui) — the encode target is
  `min(round(lowWidth * reshapedWidth / padWidth), originalWidth)` by
  `min(round(lowHeight * reshapedHeight / padHeight), originalHeight)`,
  covering landscape, portrait and square images, and the clamp holds: for an
  image smaller than the logit window the encode size never exceeds the image's
  own dimensions.
- AC5 (non-ui) — `SegmenterOptions` carries `lowResMaskEncode`, `true` in
  `DEFAULT_SEGMENTER_OPTIONS`, surviving `createSegmenter` into the worker
  request; setting it `false` resolves the encode target to full resolution and
  reproduces today's encoded masks exactly; and it does not appear in the
  worker's ONNX session cache key.
- AC6 (non-ui) — with `lowResFilterNms: false` the encode target is forced to
  full resolution regardless of `lowResMaskEncode`, because retained logits do
  not exist on that path; `mask-pipeline.ts`'s module comment states this
  interaction.
- AC7 (non-ui) — the survivor encode resample's time is recorded in the
  `resample` phase and not in `mask-encode`, so the `mask-encode` before/after
  in AC1 shows the encode saving alone.
- AC8 (non-ui) — the returned mask set and each `EncodedMask.area` are
  unchanged by `lowResMaskEncode`: the full-resolution survivor resample and
  the exact, unscaled `minMaskArea` re-check still run, so the option changes
  only the PNG's pixel dimensions.
- AC9 (ui) — the viewer reads a reduced-size mask back as strictly binary
  coverage: with `imageSmoothingEnabled = false` every pixel sampled from
  `buildMaskData`'s canvas is exactly `(255,255,255,255)` or `(0,0,0,0)`, so
  the highlight and outline a viewer sees have hard edges rather than a
  semi-transparent halo.
- AC10 (non-ui) — the delta is reproducible with the machinery that already
  exists: the Compare tab exposes a `lowResMaskEncode` control with a
  `data-testid`, the flag is an axis in `compare.ts` and validated in
  `option-choices.ts` with the same shape as `lowResFilterNms`, and a
  committed sweep config produces all four AC1 rows (16/32 points per side x
  on/off) in one run.
- AC11 (non-ui) — `docs/pipeline.md`'s description of the encode step matches
  the shipped behaviour, naming the reduced encode target and the viewer's
  upscale.
