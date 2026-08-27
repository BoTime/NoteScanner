# F2 — single-pass resample (GitHub issue #8)

Replace the two-pass `post_process_masks` upsample in the segmenter worker with
one direct, fused bilinear-resample-and-threshold over the 256x256 logit grid.

## Context and why the skip condition is void

Issue #8 carries a skip condition: "Do not build this if #9 is committed."
Issue #9 (M5 + F4, a WebGL2 renderer) is NOT committed and does not exist in
the repo. The developer has explicitly overridden the clause: rather than #9
later deleting the CPU upsample, renderer choice stays client-configurable —
`SegmentViewer` already accepts a `renderer?: RendererFactory` prop
(`src/types.ts:40`) and `createCanvas2DRenderer` is exported from the package
root — so the CPU path survives whatever #9 adds, and optimising it is not
wasted work.

**This run builds F2 only. `src/renderer/` is not touched.**

## The problem

`src/segmenter/worker/segmenter.worker.ts` upsamples each batch's chosen masks
by calling `processor.post_process_masks`, which does:

```
interpolate_4d(mask, size: pad_size)        // 256x256 -> 1024x1024  bilinear
  .slice(0..reshaped_h, 0..reshaped_w)      // crop, full copy
interpolate_4d(cropped, size: original)     // reshaped -> original  bilinear
```

and the worker then runs `thresholdMask` over the full-resolution
`Float32Array` result. That is two ORT dispatches, a ~33 MB 1024^2 Float32
intermediate, a multi-MB slice copy, a full-resolution Float32 output buffer
(~61 MB per batch at 1600x1200 with 8 chosen masks), and a second full pass
over that buffer to binarize it.

## The geometry, derived

Both `interpolate_4d` calls resolve to an ONNX `Resize` node whose only
attribute is `mode = "linear"` (verified by decoding the graph bytes in
`TensorOpRegistry.bilinear_interpolate_4d`: a single `Resize` node, one
attribute `mode="linear"`, opset 20). With no `coordinate_transformation_mode`
attribute, ONNX's default `half_pixel` applies to both:

```
src = (dst + 0.5) / scale - 0.5      where scale = out_size / in_size
```

Composing the two passes on the X axis:

```
pass 2:  v = (X + 0.5) * (reshapedW / origW) - 0.5
pass 1:  u = (v + 0.5) * (lowW / padW)       - 0.5
```

Substituting, the -0.5 and +0.5 cancel exactly:

```
u = (X + 0.5) * (reshapedW / origW) * (lowW / padW) - 0.5
```

So the whole chain is equivalent to ONE `half_pixel` bilinear resample of the
top-left `(reshapedH * lowH / padH) x (reshapedW * lowW / padW)` window of the
256x256 logit grid up to `origH x origW`. For SAM's 1024 pad that window is
`reshapedH/4 x reshapedW/4` — fractional in general, which is exactly why an
integer crop of the low-res grid is NOT an acceptable implementation.

This composition is provably correct for the *coordinate map*. It is not
bit-identical to the two-pass *result*: the intermediate 1024^2 grid quantises
the interpolated ramp where a sample's neighbourhood straddles a source-cell
boundary. That residual difference is small, real, and is what acceptance
criterion AC1 measures.

## Design

### New module: `src/segmenter/core/mask-resample.ts`

A pure, synchronous function — no ORT, no async, no tensors — so it is fully
testable under vitest in Node:

```ts
export function resampleThresholdMask(options: {
  logits: Float32Array;   // one mask's window; length must be lowWidth*lowHeight
  lowWidth: number; lowHeight: number;
  padWidth: number; padHeight: number;
  reshapedWidth: number; reshapedHeight: number;
  originalWidth: number; originalHeight: number;
  threshold: number;
}): BinaryMask;
```

Returns the existing `BinaryMask` (`{ coverage: Uint8Array; area: number }`) —
declared in `src/segmenter/core/mask-postprocess.ts` and re-exported from
`src/segmenter/core/nms.ts` — so `candidates`, `dedupeMasks`, the `RawMask`
mapping and the zero-copy transfer downstream are all untouched.

Implementation shape:

- Per-axis scale `sx = (reshapedWidth * lowWidth) / (originalWidth * padWidth)`,
  and `sy` likewise.
- For output column `X`: `u = (X + 0.5) * sx - 0.5`, clamped to
  `[0, lowWidth - 1]`; `x0 = min(floor(u), lowWidth - 2)` when `lowWidth >= 2`
  else `0`; `x1 = min(x0 + 1, lowWidth - 1)`; `wx = u - x0` clamped to `[0, 1]`.
  Precompute `x0`, `x1` into an `Int32Array(originalWidth)` and `wx` into a
  `Float32Array(originalWidth)` ONCE, outside the row loop.
- For each output row `Y`: compute `y0`, `y1`, `wy` once, hoist the two row
  base offsets, then the inner loop over columns is four typed-array loads, a
  lerp, a strict `>` comparison against `threshold`, a coverage store, and an
  area increment.
- The strict `>` matches `thresholdMask`'s documented semantics exactly (a
  logit equal to the threshold is NOT covered), so a mask binarized by either
  function agrees.
- Allocates nothing but the `Uint8Array` coverage plus the two small per-column
  lookup arrays. No 1024^2 intermediate, no full-resolution `Float32Array`, and
  no second memory pass to binarize.

Guards: throw on any non-positive dimension, and throw when
`logits.length !== lowWidth * lowHeight`. A mis-sized window is a caller bug
and must be loud rather than silently producing a plausible but wrong mask.

Note the input may be a `subarray` view into a much larger tensor buffer — the
same contract `mask-postprocess.ts` already documents — so the function must
not assume it owns its input and must not write to it.

### Worker changes: `src/segmenter/worker/segmenter.worker.ts`

Remove, from the `chosen.length > 0` block:

- the `selected` staging `Float32Array` and its copy loop,
- the 5-D `Tensor` construction,
- the `await processor.post_process_masks(...)` call,
- the `full` full-resolution `Float32Array` and the `thresholdMask` loop over it,
- the now-unused `fullPixels` binding.

Replace with a single loop that calls `resampleThresholdMask` once per chosen
mask, reading each mask's window directly as
`logits.subarray(chosen[k] * lowPixels, (chosen[k] + 1) * lowPixels)`, and
pushes the result into `candidates` when `mask.area >= options.minMaskArea` —
identical admission logic to today.

`padWidth` / `padHeight` are read off the processor, resolved the same way
`post_process_masks` resolves them internally
(`image_processor.pad_size ?? image_processor.size`, each `{height, width}` —
confirmed at `transformers.js:21004`), and NOT hardcoded to 1024. Resolve them
once, after the encode step, alongside `originalSizes` / `reshapedSizes`. If
neither field is present, throw with a clear message — the worker cannot
compute correct geometry without the pad size.

`processor` is still required for the encode step; only the
`post_process_masks` call goes away.

### Sub-timer changes

`FILTER_SUBSTEP_ORDER` in `src/segmenter/core/types.ts` becomes
`['select', 'resample'] as const`, and `FilterSubstep` narrows with it. The
`upscale` and `threshold` regions are now one fused loop and reporting them
separately would ship a permanently-zero row.

- `src/segmenter/core/timing.ts` keys its filter-substep accumulators off the
  constant (`report()` zero-fills by iterating `FILTER_SUBSTEP_ORDER`), so it
  follows automatically — re-confirm when implementing, do not assume.
- The worker records `recordSub('select')` after the selection loop and
  `recordSub('resample')` after the fused resample loop. The two regions must
  still tile the `filter` stage total exactly, with no work between the last
  `recordSub` and the `timings.record('filter', ...)` call.
- `playground/SegmentView.tsx` iterates `FILTER_SUBSTEP_ORDER` to render its
  sub-rows, so it needs no structural change — re-confirm when implementing,
  do not assume.
- `src/segmenter/createSegmenter.test.ts` also asserts against
  `FILTER_SUBSTEP_ORDER` and follows automatically for the same reason.

`FILTER_SUBSTEP_ORDER` is re-exported through `note-scanner/segmenter`, so this
is a public API change. It is deliberate and belongs in the PR description.

### Tests

**`src/segmenter/core/mask-resample.test.ts` — the AC1 gate.**

A local `twoPassReference` helper in the test file reproduces today's chain in
plain JS with no ORT — bilinear `half_pixel` resample to `pad`, integer crop to
`reshaped`, bilinear `half_pixel` resample to `original`, then threshold — so
the comparison runs in Node. Both paths run over several synthetic logit fields
at more than one aspect ratio (at minimum: a smooth diagonal ramp, a centred
disc, and an off-centre blob; at minimum one landscape and one portrait
`original` size). For each case assert:

- coverage IoU between the one-pass and two-pass masks `>= 0.99`, and
- the two masks' bounding boxes agree within 1 px on all four sides.

The IoU bound tolerates honest resampling drift; the bounding-box check is what
catches a geometry error, which is the dangerous failure mode here. Record the
observed max and mean divergence (as a comment or console output the run can
quote) so the criterion is quantified rather than merely gated.

**Unit tests** in the same file for behaviour and degenerate inputs:

- `lowWidth` or `lowHeight` of 1 (no second sample to interpolate towards),
- a 1x1 output,
- an all-below-threshold window returning `area === 0` and an all-zero coverage,
- a logit exactly equal to the threshold staying uncovered (agreeing with
  `thresholdMask`),
- `coverage.length === originalWidth * originalHeight`,
- the throw guards: bad dimension, and `logits.length` mismatch,
- the input `Float32Array` is not mutated, including when passed as a
  `subarray` view.

**`src/segmenter/core/timing.test.ts`** updates to the two-substep order.

### What does not change

- `src/renderer/` — untouched, per the developer's decision.
- `thresholdMask` and `stabilityScore` stay exported and tested.
  `thresholdMask` becomes unused by the worker but remains public API and is
  now the documented reference for the strict-`>` semantics that
  `resampleThresholdMask` matches. Do not delete it.
- `PHASE_ORDER` / `SegmentationPhase` — unchanged.
- The nms stage, the mask transfer, and the playground's phase table structure.

## Measurement procedure (AC2)

This procedure is the criterion. This spec deliberately contains **no** timing
numbers: the before/after table is filled in by the developer after the branch
lands in review, on real WebGPU hardware, and posted to issue #8. Nothing in
this pipeline may invent or estimate those figures.

1. `npm run playground`, load `playground/sample/cafe-table.jpg`.
2. Run at `pointsPerSide = 16`, then at `pointsPerSide = 32`.
3. Record the `filter` phase total and its sub-rows.
4. Repeat on `main` for the "before" figures — there the sub-rows are
   `select` / `upscale` / `threshold`; on this branch they are
   `select` / `resample`. The like-for-like comparison is
   old (`upscale` + `threshold`) against new (`resample`), plus the `filter`
   total either side.

## Out of scope

- `src/renderer/` and anything in issue #9 (M5 / F4).
- F5 (running `interpolate_4d` on WebGPU) — upstream-blocked in transformers.js.
- Any change to `thresholdMask`, `stabilityScore`, `dedupeMasks`,
  `pairwiseIoU`, the worker message contract, or the mask PNG encoder.
- Producing the real-hardware before/after figures — done by hand after review.

## Acceptance criteria

Every criterion below is `(non-ui)`. The two criteria stated on issue #8 are
AC1 and AC2. Neither is observable by driving this package's UI in a browser:
AC1 is a numerical comparison between two resampling paths, and AC2 needs a
WebGPU browser downloading SlimSAM weights and a human reading a timing table.
This repository has no `@playwright/test` dependency and none is being added;
the vitest environment is `node`.

- AC1 (non-ui) — **the divergence from the two-pass result is quantified, not
  assumed.** `src/segmenter/core/mask-resample.test.ts` compares
  `resampleThresholdMask` against an in-test `twoPassReference` (bilinear
  `half_pixel` to `pad`, integer crop to `reshaped`, bilinear `half_pixel` to
  `original`, then threshold) over at least three synthetic logit fields — a
  diagonal ramp, a centred disc, an off-centre blob — across at least one
  landscape and one portrait `original` size. For every case coverage IoU is
  `>= 0.99` and the two masks' bounding boxes agree within 1 px on all four
  sides, and the observed max and mean divergence are recorded in the test
  output so the difference is a number, not a shrug.
- AC2 (non-ui) — **deferred to the human; satisfied outside this branch and not
  a gate on this PR.** The `filter` phase total is recorded before and after at
  both operating points (`pointsPerSide` 16 and 32) using the procedure in
  "Measurement procedure" above, and posted to issue #8. Producing these needs
  real WebGPU hardware, which this pipeline does not have; no estimated figures
  appear in this spec, the code, or the PR.
- AC3 (non-ui) — `src/segmenter/core/mask-resample.ts` exports
  `resampleThresholdMask`, a pure synchronous function with the signature above
  that returns a `BinaryMask`. It is importable and fully exercised under
  vitest in Node with no ORT, no tensors and no async.
- AC4 (non-ui) — the fused threshold uses a strict `>` and therefore agrees
  with `thresholdMask`: a logit exactly equal to the threshold is NOT covered.
- AC5 (non-ui) — `resampleThresholdMask` throws on any non-positive dimension
  and when `logits.length !== lowWidth * lowHeight`. A mis-sized window fails
  loudly rather than producing a plausible-looking wrong mask.
- AC6 (non-ui) — the input `Float32Array` is never mutated, including when it
  is a `subarray` view into a larger buffer.
- AC7 (non-ui) — degenerate geometry is handled: `lowWidth` or `lowHeight` of
  1, a 1x1 output, and an all-below-threshold window returning `area === 0`
  with an all-zero coverage; `coverage.length === originalWidth *
  originalHeight` in every case.
- AC8 (non-ui) — the worker no longer calls `processor.post_process_masks`, no
  longer allocates the `selected` staging array, the 5-D input `Tensor`, or any
  full-resolution `Float32Array`; each chosen mask goes straight from its
  low-res `subarray` window to a `BinaryMask` in one call. Admission is
  unchanged: `mask.area >= options.minMaskArea`.
- AC9 (non-ui) — pad size is resolved at runtime from the processor
  (`image_processor.pad_size ?? image_processor.size`), never hardcoded to
  1024, and the worker throws with a clear message when neither field is
  present.
- AC10 (non-ui) — `FILTER_SUBSTEP_ORDER` is `['select', 'resample']`; the
  timing report and the playground's nested sub-rows follow it without a
  permanently-zero row; the two recorded sub-regions tile the `filter` stage
  total exactly, with no work between the last `recordSub` and
  `timings.record('filter', ...)`.
- AC11 (non-ui) — nothing outside F2's scope regresses: `src/renderer/` is
  unchanged, `thresholdMask` and `stabilityScore` remain exported and tested,
  `PHASE_ORDER` / `SegmentationPhase` are unchanged, and the full suite
  (typecheck, lint, vitest) passes on the branch.
