# M2 — encode masks at the decoder's own resolution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write each surviving mask's PNG at the decoder's own logit window (256x162 on a 1024x649 photo) instead of at full image resolution, behind `lowResMaskEncode` (default true), without changing which masks are returned or their reported `area`.

**Architecture:** `createFilterPlan` resolves an encode target once per run; survivors are resampled a SECOND time from the retained logits at that target, and the PNG is written from that smaller coverage. The existing full-resolution survivor resample and the exact, unscaled `minMaskArea` re-check are untouched, so the returned set is bit-for-bit what it is today. `SegmentViewer` already rescales any intrinsic mask size to image space; one line (`imageSmoothingEnabled = false`) makes that upscale nearest-neighbour so the read-back stays strictly binary.

**Tech Stack:** TypeScript, React 19, vitest (+ jsdom), Playwright (chromium/webkit/firefox), Vite playground, transformers.js + ONNX Runtime WebGPU in a module worker.

**Spec:** `docs/superpowers/specs/2026-09-01-low-res-mask-encode-design.md`

## Global Constraints

- **The returned mask set and every `EncodedMask.area` must not move.** The full-resolution survivor resample and the exact, unscaled `minMaskArea` re-check both still run; `lowResMaskEncode` changes the PNG's pixel dimensions and nothing else.
- **`lowResMaskEncode` must never enter the worker's ONNX session cache key** (`src/segmenter/worker/session-key.ts`), exactly like `keepRawMasks` and `lowResFilterNms`.
- **The second resample is timed into `resample`, never into `mask-encode`.** `mask-encode`'s before/after is the claim this issue makes; moved cost hidden inside it would make that claim false.
- **`mask-encode.ts` is not modified.** It already takes a target size; the hardcode is in the caller.
- **No renderer change beyond the single `imageSmoothingEnabled = false` line** in `buildMaskData`.
- **Default `lowResMaskEncode: true`** in `DEFAULT_SEGMENTER_OPTIONS`; `false` reproduces today's encoded masks exactly.
- **Encode target formula, verbatim:**
  `encodeWidth = min(round(lowWidth * reshapedWidth / padWidth), originalWidth)`,
  `encodeHeight = min(round(lowHeight * reshapedHeight / padHeight), originalHeight)`,
  forced to `originalWidth` x `originalHeight` when `lowResFilterNms` is false.
- Every claim about a phase total or a delta lands in a committed `docs/measurements/*.md`, with figures computed from the committed artifacts — not in a commit message.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/segmenter/core/types.ts` | Modify | `SegmenterOptions.lowResMaskEncode` + its doc; `lowResMaskEncode: true` in `DEFAULT_SEGMENTER_OPTIONS`; one sentence on `EncodedMask` saying the PNG's pixel size is not necessarily the image's. |
| `src/segmenter/core/mask-pipeline.ts` | Modify | `FilterNmsOptions.lowResMaskEncode`; `FilterPlan.encodeWidth`/`.encodeHeight`; `resolveEncodeSize`; `resolveEncodeMask`; module comment states the `lowResFilterNms` coupling. |
| `src/segmenter/core/mask-pipeline.test.ts` | Modify (append + one fixture line) | Derivation, clamp, rounding, forced-full, and `resolveEncodeMask` behaviour. |
| `playground/boundary.ts` | Modify (1 line) | `pathOptions` must satisfy the widened `FilterNmsOptions` or the playground typecheck breaks. |
| `src/segmenter/worker/session-key.ts` | Modify (comment) | Name `lowResMaskEncode` among the fields deliberately absent from the key. |
| `src/segmenter/worker/session-key.test.ts` | Modify (append) | The flag must not split the session cache. |
| `src/segmenter/createSegmenter.test.ts` | Modify (append) | Default true, explicit false survives into the worker request. |
| `src/segmenter/worker/segmenter.worker.ts` | Modify | Pass the flag into `createFilterPlan`; call `resolveEncodeMask` inside the `resample` timing; encode at `plan.encodeWidth/Height`; module header sentence. |
| `src/SegmentViewer.tsx` | Modify (1 line + comment) | `ctx.imageSmoothingEnabled = false` before the `drawImage`. |
| `tests/fixtures/mask-cases.ts` | Modify | The new encode dimensions as round-trip shapes (`256x162`, `162x256`). |
| `tests/browser/mask-png.spec.ts` | Modify (append) | Strict-binary upscaled read-back and hit-test agreement, in three real engines. |
| `playground/compare.ts` | Modify | `lowResMaskEncode` on `RowOptions`/`SweepConfig`/validation/grid/row id/label/markdown column. |
| `playground/compare.test.ts` | Modify | Fixture field, grid and validation coverage. |
| `playground/CompareView.tsx` | Modify | The control + one table column. |
| `playground/CompareView.test.tsx` | Modify (append) | Default on, passthrough. |
| `playground/SegmentView.tsx` | Modify | The same control, so AC2 can be exercised by hand where masks are actually clicked. |
| `playground/SegmentView.test.tsx` | Modify (append) | Default on, control present. |
| `scripts/sweep-decode.mjs` | Modify (1 line) | Drive the new checkbox. |
| `docs/measurements/2026-09-01-mask-encode.config.json` | Create | The four-row AC1 grid. |
| `docs/measurements/2026-09-01-low-res-mask-encode.md` | Create | The AC1 analysis, with a verdict. |
| `docs/pipeline.md` | Modify | The encode step as shipped, and the status table. |

**Checked and deliberately NOT changed:**

- `src/segmenter/core/mask-encode.ts` — already takes `(coverage, width, height)`. The spec's second scope bullet is already satisfied.
- `playground/option-choices.ts` — holds SELECT choice lists only (`DTYPE_CHOICES`, `BATCH_SIZE_CHOICES`, `POINTS_PER_SIDE_CHOICES`). `lowResFilterNms` is a checkbox and its axis validation lives in `compare.ts`'s `resolveConfig`, not here (`grep -n lowResFilterNms playground/option-choices.ts` returns nothing). AC10's phrase "validated in `option-choices.ts`" describes a file that does not do that job; the "same shape as `lowResFilterNms`" requirement is met in `compare.ts`. **This deviation is deliberate — do not add a boolean to `option-choices.ts`.**
- `AGENTS.md:21` — "re-decode every mask PNG at full image resolution" describes the DECODE canvas, which stays `imageWidth x imageHeight`. Still true.
- `src/types.ts` `ViewerSegment.maskUrl` — the contract is a URL, with no dimension claim. Still true.
- `docs/pipeline.md` §1 ("The pipeline before") — a record of a measured past. Not falsified by this change; leave it.

---

### Task 1: The encode target and the second resample, as pure functions (AC4, AC5, AC6, AC8)

**Files:**
- Modify: `src/segmenter/core/types.ts`
- Modify: `src/segmenter/core/mask-pipeline.ts`
- Modify: `src/segmenter/core/mask-pipeline.test.ts`
- Modify: `src/segmenter/worker/session-key.ts` (comment), `src/segmenter/worker/session-key.test.ts`
- Modify: `src/segmenter/createSegmenter.test.ts`
- Modify: `playground/boundary.ts`

**Interfaces:**
- Consumes: `MaskGeometry`, `FilterPlan`, `MaskCandidate`, `BinaryMask`, `resampleThresholdMask` — all as they exist today.
- Produces, for Task 2 and Task 3:
  - `SegmenterOptions.lowResMaskEncode: boolean` (public through `note-scanner/segmenter`), `true` in `DEFAULT_SEGMENTER_OPTIONS`.
  - `FilterNmsOptions.lowResMaskEncode: boolean` — now REQUIRED on every literal.
  - `FilterPlan.encodeWidth: number`, `FilterPlan.encodeHeight: number`.
  - `resolveEncodeSize(geometry: MaskGeometry, options: FilterNmsOptions): { width: number; height: number }`.
  - `resolveEncodeMask(candidate: MaskCandidate, mask: BinaryMask, plan: FilterPlan): BinaryMask`.

- [ ] **Step 1: Write the failing tests**

In `src/segmenter/core/mask-pipeline.test.ts`, first add the new field to the shared fixture so the file compiles — in `const OPTIONS: FilterNmsOptions` (line 37), after `lowResFilterNms: true,`:

```ts
  lowResMaskEncode: true,
```

Then add these imports to the existing import block from `./mask-pipeline`: `resolveEncodeMask`, `resolveEncodeSize`.

Append this block at the end of the file:

```ts
/**
 * SAM's real geometry: a 256x256 logit grid over a 1024x1024 pad, with the
 * resized image occupying the top-left `reshaped` window. These are the
 * numbers a photo actually produces, not scaled-down stand-ins.
 */
function samGeometry(
  originalWidth: number,
  originalHeight: number,
  reshapedWidth: number,
  reshapedHeight: number,
): MaskGeometry {
  return {
    lowWidth: 256,
    lowHeight: 256,
    padWidth: 1024,
    padHeight: 1024,
    reshapedWidth,
    reshapedHeight,
    originalWidth,
    originalHeight,
  };
}

function encodeSize(geometry: MaskGeometry, overrides: Partial<FilterNmsOptions> = {}) {
  const size = resolveEncodeSize(geometry, { ...OPTIONS, ...overrides });
  return [size.width, size.height];
}

describe('resolveEncodeSize', () => {
  it('gives one output pixel per decoder sample on a landscape photo (AC4)', () => {
    // 1024x649 resized to 1024x649 inside a 1024 pad: 256 x round(162.25).
    expect(encodeSize(samGeometry(1024, 649, 1024, 649))).toEqual([256, 162]);
  });

  it('follows the image round on a portrait photo (AC4)', () => {
    expect(encodeSize(samGeometry(649, 1024, 649, 1024))).toEqual([162, 256]);
  });

  it('is the whole grid on a square image (AC4)', () => {
    expect(encodeSize(samGeometry(1024, 1024, 1024, 1024))).toEqual([256, 256]);
  });

  it('rounds a half sample up rather than truncating (AC4)', () => {
    // 256 * 650 / 1024 = 162.5 exactly. Math.floor would give 162.
    expect(encodeSize(samGeometry(1024, 650, 1024, 650))).toEqual([256, 163]);
  });

  it('never upscales a photo smaller than the logit window (AC4)', () => {
    // 200x150 resized to 1024x768: the unclamped window would be 256x192,
    // LARGER than the image on both axes.
    expect(encodeSize(samGeometry(200, 150, 1024, 768))).toEqual([200, 150]);
  });

  it('is full resolution when lowResMaskEncode is off (AC5)', () => {
    expect(encodeSize(samGeometry(1024, 649, 1024, 649), { lowResMaskEncode: false })).toEqual([
      1024, 649,
    ]);
  });

  it('is full resolution when lowResFilterNms is off, whatever lowResMaskEncode says (AC6)', () => {
    // The encode resample reads the RETAINED LOGITS, and the baseline path
    // keeps none: `MaskCandidate.logits` is null there by construction.
    expect(
      encodeSize(samGeometry(1024, 649, 1024, 649), {
        lowResFilterNms: false,
        lowResMaskEncode: true,
      }),
    ).toEqual([1024, 649]);
  });

  it('is carried on the plan', () => {
    const plan = lowPlan();
    // GEOMETRY: 16x16 grid, 16x16 pad, 16x12 reshaped, 32x24 image.
    expect([plan.encodeWidth, plan.encodeHeight]).toEqual([16, 12]);
    expect([fullPlan().encodeWidth, fullPlan().encodeHeight]).toEqual([32, 24]);
  });
});

describe('resolveEncodeMask', () => {
  it('resamples the logit window at the encode size, leaving the survivor alone (AC8)', () => {
    const window = makeWindow([2, 9, 3, 8]);
    const plan = lowPlan();
    const { candidate } = retainCandidate(window, plan);
    const mask = resolveSurvivor(candidate!, plan)!;

    const encoded = resolveEncodeMask(candidate!, mask, plan);

    // The survivor is untouched: this is what keeps `EncodedMask.area` and the
    // returned set independent of the option.
    expect(mask.coverage.length).toBe(32 * 24);
    expect(mask.area).toBe(192);

    expect(encoded.coverage.length).toBe(16 * 12);
    expect(encoded.area).toBe(48);
    // At this geometry both scale factors are exactly 1, so the encoded mask
    // must be the thresholded top-left 16x12 window of the logit grid, sample
    // for sample — an expectation derived from the WINDOW, not from the
    // implementation.
    const expected: number[] = [];
    for (let y = 0; y < 12; y += 1) {
      for (let x = 0; x < 16; x += 1) expected.push(window[y * 16 + x] > 0 ? 1 : 0);
    }
    expect(Array.from(encoded.coverage)).toEqual(expected);
  });

  it('hands back the survivor itself when the target is full resolution (AC5)', () => {
    const plan = lowPlan({ lowResMaskEncode: false });
    const { candidate } = retainCandidate(makeWindow([2, 9, 3, 8]), plan);
    const mask = resolveSurvivor(candidate!, plan)!;
    // The SAME object, not an equal one: with the flag off the PNG is written
    // from exactly the bytes it is written from today.
    expect(resolveEncodeMask(candidate!, mask, plan)).toBe(mask);
  });

  it('hands back the survivor itself when the clamp collapsed the target', () => {
    const plan = createFilterPlan(samGeometry(200, 150, 1024, 768), OPTIONS);
    const mask = { coverage: new Uint8Array(200 * 150), area: 7 };
    expect(resolveEncodeMask({ logits: null, coverage: null }, mask, plan)).toBe(mask);
  });

  it('fails loudly on a released candidate rather than encoding garbage', () => {
    const plan = lowPlan();
    const { candidate } = retainCandidate(makeWindow([2, 9, 3, 8]), plan);
    const mask = resolveSurvivor(candidate!, plan)!;
    releaseCandidate(candidate!);
    expect(() => resolveEncodeMask(candidate!, mask, plan)).toThrow(/released/);
  });
});
```

In `src/segmenter/worker/session-key.test.ts`, append inside the `describe('sessionCacheKey', ...)` block:

```ts
  it('ignores lowResMaskEncode, which only picks a resample target (AC5)', () => {
    expect(sessionCacheKey(options({ lowResMaskEncode: true }))).toBe(
      sessionCacheKey(options({ lowResMaskEncode: false })),
    );
  });
```

In `src/segmenter/createSegmenter.test.ts`, append after the `lowResFilterNms` test (line 391):

```ts
  it('defaults lowResMaskEncode ON and passes an explicit false through (AC5)', async () => {
    const segmenter = createSegmenter({ createWorker: spawn });
    const first = segmenter.segment(fakeBitmap());
    expect(FakeWorker.instances[0].posted[0].options.lowResMaskEncode).toBe(true);
    FakeWorker.instances[0].emit(doneMessage());
    await first;

    const second = segmenter.segment(fakeBitmap(), { lowResMaskEncode: false });
    expect(FakeWorker.instances[1].posted[0].options.lowResMaskEncode).toBe(false);
    FakeWorker.instances[1].emit(doneMessage());
    await second;
  });
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
npx vitest run src/segmenter/core/mask-pipeline.test.ts src/segmenter/worker/session-key.test.ts src/segmenter/createSegmenter.test.ts
```

Expected: FAIL. `resolveEncodeSize` and `resolveEncodeMask` are not exported (import errors), `lowResMaskEncode` is not a known property of `SegmenterOptions`/`FilterNmsOptions`, and `plan.encodeWidth` is `undefined`. Confirm you see failures, not a pass.

- [ ] **Step 3: Add the option to the public type**

In `src/segmenter/core/types.ts`, immediately after `lowResFilterNms: boolean;` (line 139) add:

```ts

  /**
   * Encode each surviving mask's PNG at the decoder's own resolution instead
   * of at full image resolution.
   *
   * DEFAULT TRUE. The decoder emits a 256x256 logit grid, of which only the
   * top-left `lowWidth * reshapedWidth / padWidth` by
   * `lowHeight * reshapedHeight / padHeight` window maps onto image pixels —
   * 256x162 on a 1024x649 photo, 41k samples against 665k pixels. Encoding
   * there writes one PNG pixel per decoder sample instead of interpolating for
   * information the model never produced. `SegmentViewer` already rescales any
   * intrinsic mask size into image space, and does it with
   * `imageSmoothingEnabled = false`, so a covered pixel still reads back as
   * exactly `(255,255,255,255)`.
   *
   * It does NOT change which masks are returned, nor `EncodedMask.area`: the
   * full-resolution survivor resample and the exact, unscaled `minMaskArea`
   * re-check both still run. What it does change is boundary precision — mask
   * edges quantise to the upscale factor, about 4 px on a 1024x649 photo.
   *
   * It has NO EFFECT when `lowResFilterNms` is false: the encode resample
   * reads the retained logits, which only that path keeps.
   *
   * Like `keepRawMasks` and `lowResFilterNms`, it selects a resample target
   * and nothing about the ONNX sessions, so it must never enter the worker's
   * session cache key.
   */
  lowResMaskEncode: boolean;
```

In `DEFAULT_SEGMENTER_OPTIONS`, after `lowResFilterNms: true,`:

```ts
  lowResMaskEncode: true,
```

And extend the `EncodedMask` doc — it currently has none. Replace

```ts
export interface EncodedMask {
  maskUrl: string;
  area: number;
}
```

with

```ts
export interface EncodedMask {
  /**
   * A 1-bit indexed PNG data URL. Its pixel dimensions are the ENCODE target,
   * which under the default `lowResMaskEncode` is the decoder's own window
   * (256x162 on a 1024x649 photo) rather than the image size reported on the
   * `done` message. A consumer must scale it into image space, as
   * `SegmentViewer` does.
   */
  maskUrl: string;
  /** Covered pixels at FULL image resolution, independent of the encode target. */
  area: number;
}
```

- [ ] **Step 4: Add the encode target and the encode resample to the pipeline**

In `src/segmenter/core/mask-pipeline.ts`:

**(a)** extend `FilterNmsOptions`, after `lowResFilterNms: boolean;`:

```ts
  /** True to write survivor PNGs at the decoder's own window; see `resolveEncodeSize`. */
  lowResMaskEncode: boolean;
```

**(b)** extend `FilterPlan`, after `nmsWidth: number;`:

```ts
  /** Width of the mask PNG, in pixels. `geometry.originalWidth` when encoding full-size. */
  encodeWidth: number;
  /** Height of the mask PNG, in pixels. `geometry.originalHeight` when encoding full-size. */
  encodeHeight: number;
```

**(c)** add `resolveEncodeSize` immediately above `createFilterPlan`:

```ts
/**
 * The size a survivor's PNG is written at: one output pixel per decoder
 * sample over the window that actually maps onto the image.
 *
 * `resampleThresholdMask` samples the top-left
 * `(reshapedWidth * lowWidth / padWidth) x (reshapedHeight * lowHeight / padHeight)`
 * window of the logit grid however large an output it is asked for, so
 * rounding that window to whole samples is the size at which the resample
 * neither invents nor discards information.
 *
 * CLAMPED to the image on each axis. Without the clamp a photo smaller than
 * the window — a 200x150 thumbnail, whose window rounds to 256x192 — would be
 * handed a mask LARGER than full resolution, which is the opposite of the
 * point.
 *
 * Returns full resolution on either opt-out: `lowResMaskEncode` off, or
 * `lowResFilterNms` off (see the module comment — the baseline path retains no
 * logits to resample from).
 */
export function resolveEncodeSize(
  geometry: MaskGeometry,
  options: FilterNmsOptions,
): { width: number; height: number } {
  if (!options.lowResMaskEncode || !options.lowResFilterNms) {
    return { width: geometry.originalWidth, height: geometry.originalHeight };
  }
  return {
    width: Math.min(
      Math.round((geometry.lowWidth * geometry.reshapedWidth) / geometry.padWidth),
      geometry.originalWidth,
    ),
    height: Math.min(
      Math.round((geometry.lowHeight * geometry.reshapedHeight) / geometry.padHeight),
      geometry.originalHeight,
    ),
  };
}
```

**(d)** carry it on the plan. `createFilterPlan`'s body becomes:

```ts
export function createFilterPlan(
  geometry: MaskGeometry,
  options: FilterNmsOptions,
): FilterPlan {
  const lowPixels = geometry.lowWidth * geometry.lowHeight;
  const originalPixels = geometry.originalWidth * geometry.originalHeight;
  const encode = resolveEncodeSize(geometry, options);
  return {
    geometry,
    options,
    minArea: options.lowResFilterNms
      ? lowResMinArea(options.minMaskArea, lowPixels, originalPixels)
      : options.minMaskArea,
    nmsWidth: options.lowResFilterNms ? geometry.lowWidth : geometry.originalWidth,
    encodeWidth: encode.width,
    encodeHeight: encode.height,
  };
}
```

**(e)** append `resolveEncodeMask` at the end of the file, below `resolveSurvivor`:

```ts
/**
 * The coverage a survivor's PNG is written from.
 *
 * A SECOND resample of the same retained logits, at `plan.encodeWidth` x
 * `plan.encodeHeight`. It exists alongside `resolveSurvivor` rather than
 * replacing it because two things depend on the full-resolution mask and must
 * not move: the exact, unscaled `minMaskArea` re-check, and the `area`
 * reported on `EncodedMask`. The cost is ~41k extra output pixels against the
 * ~665k already resampled on a 1024x649 photo.
 *
 * Deliberately NOT a decimation of `mask`: downsampling an already-binarised
 * mask is how thin structures vanish. It resamples the logits, so a structure
 * one output pixel wide still crosses the threshold.
 *
 * When the target IS full resolution — the flag off, `lowResFilterNms` off, or
 * a photo small enough that the clamp collapsed the window — this returns
 * `mask` itself, so that path costs nothing and produces byte-for-byte
 * today's PNG.
 */
export function resolveEncodeMask(
  candidate: MaskCandidate,
  mask: BinaryMask,
  plan: FilterPlan,
): BinaryMask {
  const { encodeWidth, encodeHeight } = plan;
  if (
    encodeWidth === plan.geometry.originalWidth &&
    encodeHeight === plan.geometry.originalHeight
  ) {
    return mask;
  }
  if (!candidate.logits) {
    throw new Error('mask-pipeline: resolveEncodeMask called on a released candidate');
  }
  return resampleThresholdMask({
    logits: candidate.logits,
    ...plan.geometry,
    originalWidth: encodeWidth,
    originalHeight: encodeHeight,
    threshold: plan.options.maskThreshold,
  });
}
```

**(f)** the module comment. After the paragraph ending `This is the one place the old path stays.` (line 24), insert:

```
 * The ENCODE target is a third resolution decision, and it is COUPLED to the
 * one above. `lowResMaskEncode` asks for the PNG to be written at the
 * decoder's own window (`resolveEncodeSize`), and `resolveEncodeMask` produces
 * that coverage by resampling the RETAINED LOGITS — which exist only on the
 * `lowResFilterNms: true` path, where `MaskCandidate.logits` is a copy of the
 * window; on the baseline path it is null by construction. So
 * `createFilterPlan` forces the encode target back to full resolution whenever
 * `lowResFilterNms` is false, whatever `lowResMaskEncode` says. Retaining
 * logits on the baseline path to lift that would add 256 KB per candidate to a
 * path that already carries full-resolution coverage and exists only to be
 * measured against.
 *
```

- [ ] **Step 5: Fix the one other `FilterNmsOptions` literal**

`playground/boundary.ts:56` builds a `FilterNmsOptions` and will not typecheck against the widened interface. Change:

```ts
function pathOptions(lowResFilterNms: boolean): FilterNmsOptions {
  return { ...BOUNDARY_OPTIONS, lowResFilterNms };
}
```

to:

```ts
function pathOptions(lowResFilterNms: boolean): FilterNmsOptions {
  // The Boundary tab compares FULL-RESOLUTION masks between the two filter
  // paths, so it pins the encode target off: a reduced PNG target would change
  // nothing it looks at, and pinning it says so.
  return { ...BOUNDARY_OPTIONS, lowResFilterNms, lowResMaskEncode: false };
}
```

- [ ] **Step 6: Update the session-key comment**

In `src/segmenter/worker/session-key.ts`, the doc comment on `sessionCacheKey` currently reads `...and so is `lowResFilterNms`, which only changes what the worker does with `pred_masks` after the decoder has returned it.` Replace that clause with:

```
 * `keepRawMasks` is deliberately absent — it only changes what the worker
 * posts back — and so are `lowResFilterNms` and `lowResMaskEncode`, which only
 * change what the worker does with `pred_masks` after the decoder has returned
 * it, and at what size it writes the PNG. So is every
```

(keeping the rest of the sentence intact).

- [ ] **Step 7: Run the tests and the typechecks**

```bash
npm test
npm run typecheck
```

Expected: PASS, both. The typecheck's second half (`tsconfig.playground.json`) is what proves Step 5 was needed and sufficient.

- [ ] **Step 8: Commit**

```bash
git add src/segmenter playground/boundary.ts
git commit -m "feat(segmenter): resolve a mask-encode target at the decoder's own resolution"
```

---

### Task 2: Encode there, upscale without smoothing, and prove it in three engines (AC2, AC3, AC7, AC8, AC9)

**Files:**
- Modify: `src/segmenter/worker/segmenter.worker.ts`
- Modify: `src/SegmentViewer.tsx:79`
- Modify: `tests/fixtures/mask-cases.ts`
- Modify: `tests/browser/mask-png.spec.ts`

**Interfaces:**
- Consumes from Task 1: `resolveEncodeMask(candidate, mask, plan)`, `plan.encodeWidth`, `plan.encodeHeight`, `SegmenterOptions.lowResMaskEncode`.
- Consumes, unchanged: `encodeMaskPng(coverage, width, height)` from `src/segmenter/core/mask-encode`, `hitTestAll(point, width, height, masks)` and `type SegmentMaskData` from `src/core/segment-viewer-logic`.
- Produces: nothing new for Task 3 beyond the shipped behaviour.

- [ ] **Step 1: Write the failing browser tests**

Append to `tests/browser/mask-png.spec.ts`. Add `hitTestAll` and `SegmentMaskData` to the imports at the top:

```ts
import { hitTestAll, type SegmentMaskData } from '../../src/core/segment-viewer-logic';
```

Then append:

```ts
/**
 * `buildMaskData`'s decode, run verbatim in the page — INCLUDING the upscale
 * this change relies on. `SegmentViewer` draws every mask at image size
 * whatever the PNG's intrinsic size is, so a reduced-size mask is upscaled
 * here exactly as it is there.
 *
 * Returns the coverage plus a count of pixels that were NEITHER
 * `(255,255,255,255)` nor `(0,0,0,0)`, which is the property the whole
 * coverage contract rests on.
 */
async function decodeUpscaledInPage(
  page: Page,
  maskUrl: string,
  width: number,
  height: number,
  smoothing: boolean,
): Promise<{ coverage: number[]; nonBinary: number; firstNonBinary: number[] | null }> {
  return page.evaluate(
    ({ maskUrl, width, height, smoothing }) =>
      new Promise<{ coverage: number[]; nonBinary: number; firstNonBinary: number[] | null }>(
        (resolve, reject) => {
          const img = new Image();
          img.onerror = () => reject(new Error('this engine refused to decode the mask PNG'));
          img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            if (!ctx) {
              reject(new Error('no 2d context'));
              return;
            }
            ctx.imageSmoothingEnabled = smoothing;
            ctx.drawImage(img, 0, 0, width, height);
            const data = ctx.getImageData(0, 0, width, height).data;
            const coverage: number[] = [];
            let nonBinary = 0;
            let firstNonBinary: number[] | null = null;
            for (let i = 0; i < data.length; i += 4) {
              const px = [data[i], data[i + 1], data[i + 2], data[i + 3]];
              const white = px[0] === 255 && px[1] === 255 && px[2] === 255 && px[3] === 255;
              const clear = px[0] === 0 && px[1] === 0 && px[2] === 0 && px[3] === 0;
              if (!white && !clear) {
                nonBinary += 1;
                firstNonBinary ??= px;
              }
              coverage.push(px[3] > 0 && px[0] > 0 ? 1 : 0);
            }
            resolve({ coverage, nonBinary, firstNonBinary });
          };
          img.src = maskUrl;
        },
      ),
    { maskUrl, width, height, smoothing },
  );
}

/**
 * A scene at both resolutions, with an EXACTLY 4x integer ratio so the two
 * coverages are the same shape by construction and any disagreement is the
 * upscale's doing rather than the fixture's.
 *
 * Rects are given in encode coordinates as [x, y, w, h]. Every area is
 * distinct, so `hitTestAll`'s smallest-first ordering is decided by the masks
 * rather than by input order, and `comb` is one encode pixel wide — the thin
 * structure this change is most likely to destroy.
 */
const ENCODE_W = 64;
const ENCODE_H = 48;
const FULL_W = 256;
const FULL_H = 192;
const SCALE = 4;

const SCENE: ReadonlyArray<{ id: string; rect: readonly [number, number, number, number] }> = [
  { id: 'comb', rect: [32, 0, 1, 48] },        // full area   768 — 4px wide, full height
  { id: 'top-left', rect: [0, 0, 10, 8] },     // full area 1,280 — touches top and left edges
  { id: 'bottom-right', rect: [52, 38, 12, 10] }, // full area 1,920 — touches right and bottom
  { id: 'blob', rect: [20, 20, 16, 12] },      // full area 3,072 — overlaps `comb`
];

function rectCoverage(
  rect: readonly [number, number, number, number],
  width: number,
  height: number,
  scale: number,
): { coverage: Uint8Array; area: number } {
  const [rx, ry, rw, rh] = rect;
  const coverage = new Uint8Array(width * height);
  let area = 0;
  for (let y = ry * scale; y < (ry + rh) * scale; y += 1) {
    for (let x = rx * scale; x < (rx + rw) * scale; x += 1) {
      coverage[y * width + x] = 1;
      area += 1;
    }
  }
  return { coverage, area };
}

/**
 * Probe points in IMAGE space, with the ids each must select — worked out from
 * the rects above and written down, never derived from the run.
 */
const PROBES: ReadonlyArray<{ point: { x: number; y: number }; expect: string[] }> = [
  { point: { x: 0, y: 0 }, expect: ['top-left'] },
  { point: { x: 255, y: 191 }, expect: ['bottom-right'] },
  { point: { x: 255, y: 0 }, expect: [] },
  { point: { x: 0, y: 191 }, expect: [] },
  { point: { x: 129, y: 0 }, expect: ['comb'] },
  { point: { x: 131, y: 191 }, expect: ['comb'] },
  { point: { x: 127, y: 5 }, expect: [] },
  { point: { x: 132, y: 5 }, expect: [] },
  { point: { x: 128, y: 100 }, expect: ['comb', 'blob'] },
  { point: { x: 132, y: 100 }, expect: ['blob'] },
  { point: { x: 79, y: 100 }, expect: [] },
  { point: { x: 80, y: 100 }, expect: ['blob'] },
];

test('a mask encoded at the decoder size hit-tests exactly like a full-resolution one', async ({
  page,
}) => {
  const reduced: SegmentMaskData[] = [];
  const full: SegmentMaskData[] = [];

  for (const { id, rect } of SCENE) {
    const small = rectCoverage(rect, ENCODE_W, ENCODE_H, 1);
    const large = rectCoverage(rect, FULL_W, FULL_H, SCALE);

    const smallUrl = await encodeMaskPng(small.coverage, ENCODE_W, ENCODE_H);
    const largeUrl = await encodeMaskPng(large.coverage, FULL_W, FULL_H);

    const upscaled = await decodeUpscaledInPage(page, smallUrl, FULL_W, FULL_H, false);
    const native = await decodeUpscaledInPage(page, largeUrl, FULL_W, FULL_H, false);

    // AC9: nothing between covered and clear comes back from either decode.
    expect(
      upscaled.nonBinary,
      `${id}: ${upscaled.nonBinary} non-binary pixels in the upscaled read-back, first ${JSON.stringify(upscaled.firstNonBinary)}`,
    ).toBe(0);
    expect(native.nonBinary).toBe(0);

    // The ratio is an exact integer, so nearest-neighbour must reproduce the
    // full-resolution coverage pixel for pixel.
    const mismatches = upscaled.coverage.reduce(
      (n, value, i) => (value === native.coverage[i] ? n : n + 1),
      0,
    );
    expect(mismatches, `${id}: ${mismatches} pixels differ after the 4x upscale`).toBe(0);

    const area = (values: number[]) => values.reduce((n, v) => n + v, 0);
    reduced.push({ id, coverage: Uint8Array.from(upscaled.coverage), area: area(upscaled.coverage) });
    full.push({ id, coverage: Uint8Array.from(native.coverage), area: area(native.coverage) });
  }

  expect(reduced).toHaveLength(SCENE.length);

  for (const probe of PROBES) {
    const label = `(${probe.point.x},${probe.point.y})`;
    expect(hitTestAll(probe.point, FULL_W, FULL_H, full), `${label} full-res`).toEqual(probe.expect);
    expect(hitTestAll(probe.point, FULL_W, FULL_H, reduced), `${label} reduced`).toEqual(
      probe.expect,
    );
  }
});

test('smoothing is what would break the coverage contract', async ({ page, browserName }) => {
  // The negative control for the test above: it demonstrates that
  // `imageSmoothingEnabled = false` is load-bearing rather than decorative.
  // Chromium only — this is a statement about the engine the product runs on,
  // and a bilinear upscale is not a behaviour other engines owe us.
  test.skip(browserName !== 'chromium', 'control for the engine the product ships on');

  const small = rectCoverage(SCENE[3].rect, ENCODE_W, ENCODE_H, 1);
  const url = await encodeMaskPng(small.coverage, ENCODE_W, ENCODE_H);
  const smoothed = await decodeUpscaledInPage(page, url, FULL_W, FULL_H, true);
  expect(smoothed.nonBinary).toBeGreaterThan(0);
});
```

Then add the encode dimensions to the shared round-trip fixtures (AC3). In `tests/fixtures/mask-cases.ts`, extend `SHAPES` and its comment:

```ts
/**
 * 7, 9 and 17 are not multiples of 8, so their last row byte carries padding
 * bits — the case a 1-bit writer gets wrong. 1xN and Nx1 are the degenerate
 * strips; 64x64 is big enough that deflate actually has something to chew on.
 *
 * 256x162 and 162x256 are the ENCODE dimensions a 1024x649 (and 649x1024)
 * photo produces under `lowResMaskEncode` — the sizes the writer is now
 * actually asked for. 162 is not a multiple of 8 either, so the portrait case
 * keeps the padding-bit path exercised at a realistic width.
 */
const SHAPES: ReadonlyArray<readonly [number, number]> = [
  [1, 1],
  [7, 3],
  [8, 3],
  [9, 3],
  [17, 5],
  [1, 40],
  [40, 1],
  [64, 64],
  [256, 162],
  [162, 256],
];
```

- [ ] **Step 2: Run the browser tests to verify they fail**

```bash
npx playwright install --with-deps chromium webkit firefox   # only if the browsers are missing
npm run test:browser
```

**Be honest about what this step proves.** `buildMaskData` is module-private in a component jsdom cannot render (no canvas), so this spec re-runs its decode in the page rather than importing it — the same arrangement `decodeInPage`'s docstring already describes. Consequently the two new tests are expected to PASS at this point: they pin the contract `SegmentViewer` must satisfy, and Step 3 is what makes `SegmentViewer` satisfy it, checked by the anchored grep in Step 5. What must be watched here instead:

- the six new `MASK_CASES` round-trip cases (`256x162`, `162x256`) must pass in all three engines. If either fails, STOP — that is a real defect in the writer at a size this change is about to start using every run.
- the smoothing control must FAIL if you delete the `nonBinary` assertions' premise: run it once with `smoothing: false` swapped to `true` in the first test and confirm the `nonBinary` assertion fires. Put it back before moving on. This is the one check that the assertion is not vacuous.

- [ ] **Step 3: Turn smoothing off in the viewer**

In `src/SegmentViewer.tsx`, in `buildMaskData`, immediately before `ctx.drawImage(img, 0, 0, width, height);` (line 79):

```ts
  // Masks may arrive smaller than the image — the worker encodes them at the
  // decoder's own resolution under `lowResMaskEncode` — and this `drawImage`
  // is the upscale. NEAREST-NEIGHBOUR, deliberately: it is the cheaper filter
  // (one tap per output pixel, on a canvas that is CPU-backed by
  // `willReadFrequently`), and it is the only one that keeps every read-back
  // pixel exactly (255,255,255,255) or (0,0,0,0), which is what the
  // `alpha > 0 && red > 0` predicate below — and the preview and server-side
  // crops that share it — depend on. Bilinear would ring a semi-transparent
  // halo around every boundary and quietly widen every mask.
  ctx.imageSmoothingEnabled = false;
```

- [ ] **Step 4: Encode from the reduced coverage in the worker**

In `src/segmenter/worker/segmenter.worker.ts`:

**(a)** add `resolveEncodeMask,` to the import list from `'../core'` (alphabetically, between `releaseRejected,` and `resolveSurvivor,`).

**(b)** pass the flag into the plan. In the `createFilterPlan` call's options object (around line 305), after `lowResFilterNms: options.lowResFilterNms,`:

```ts
            lowResMaskEncode: options.lowResMaskEncode,
```

**(c)** replace the survivor loop's resample/encode region. Today (lines 434-452):

```ts
        phase = 'resample';
        const resampleStarted = performance.now();
        const mask = resolveSurvivor(candidate, plan);
        timings.record('resample', performance.now() - resampleStarted);
        if (!mask) {
          // Passed the coarse pre-NMS gate, failed the exact full-resolution
          // `minMaskArea`. Visible as the gap between afterNms and returned.
          releaseCandidate(candidate);
          continue;
        }

        phase = 'mask-encode';
        const encodeStarted = performance.now();
        const maskUrl = await encodeMaskPng(mask.coverage, originalWidth, originalHeight);
        timings.record('mask-encode', performance.now() - encodeStarted);
        masks.push({ maskUrl, area: mask.area });
```

becomes:

```ts
        phase = 'resample';
        const resampleStarted = performance.now();
        const mask = resolveSurvivor(candidate, plan);
        // The SECOND resample, at the encode target, timed into `resample`
        // with the first. Keeping it out of `mask-encode` is what lets that
        // stage's before/after read as the encode saving alone instead of
        // hiding the moved cost inside the very number this change claims.
        const encodeMask = mask ? resolveEncodeMask(candidate, mask, plan) : null;
        timings.record('resample', performance.now() - resampleStarted);
        if (!mask || !encodeMask) {
          // Passed the coarse pre-NMS gate, failed the exact full-resolution
          // `minMaskArea`. Visible as the gap between afterNms and returned.
          releaseCandidate(candidate);
          continue;
        }

        phase = 'mask-encode';
        const encodeStarted = performance.now();
        const maskUrl = await encodeMaskPng(
          encodeMask.coverage,
          plan.encodeWidth,
          plan.encodeHeight,
        );
        timings.record('mask-encode', performance.now() - encodeStarted);
        // `area` stays the FULL-RESOLUTION survivor's: the option moves the
        // PNG's pixel dimensions and nothing else about the returned set.
        masks.push({ maskUrl, area: mask.area });
```

Nothing else in the loop moves. In particular `rawMasks.push({ coverage: mask.coverage, area: mask.area })` and the `done` message's `width: originalWidth, height: originalHeight` stay exactly as they are — `keepRawMasks` still yields full-resolution coverage, and the image size the viewer scales into is still the image size.

**(d)** the module header. In the `Pipeline:` sentence, replace `resample ONLY the survivors to full resolution, fused with the PNG encode.` with:

```
 * batches -> filter each batch at the decoder's native 256x256 -> dedupe every
 * candidate at 256x256 -> resample ONLY the survivors to full resolution (for
 * the exact area gate and the reported area) and, by default, a second time at
 * the decoder's own window, which is what the PNG is written from. That encode
 * is fused into the same loop.
```

- [ ] **Step 5: Verify the ordering the timing claim rests on (AC7)**

```bash
grep -n "resolveSurvivor(\|resolveEncodeMask(\|timings.record('resample'\|phase = 'mask-encode'\|encodeMaskPng(" src/segmenter/worker/segmenter.worker.ts
```

Expected: exactly five lines, in this order — `resolveSurvivor(candidate, plan)`, `resolveEncodeMask(candidate, mask, plan)`, `timings.record('resample'`, `phase = 'mask-encode'`, `encodeMaskPng(`. The `resolveEncodeMask` call must sit ABOVE the `timings.record('resample'` line; if it is below, its cost is landing in `mask-encode` and AC7 fails. (The import line matches none of these patterns — each requires the call's open paren or the assignment.)

And that the returned set is untouched (AC8):

```bash
grep -n "area: mask.area" src/segmenter/worker/segmenter.worker.ts
```

Expected: exactly two lines — the `masks.push` and the `rawMasks.push`. Both must read `mask`, the full-resolution survivor; neither may read `encodeMask`.

And the viewer's one line (AC9):

```bash
grep -n -A1 "imageSmoothingEnabled" src/SegmentViewer.tsx
```

Expected: exactly one match, `ctx.imageSmoothingEnabled = false;`, on the line immediately before `ctx.drawImage(img, 0, 0, width, height);`. The identifier appears nowhere else in the file — not in the comment this plan mandates — so a second match means a second, unasked-for renderer change.

- [ ] **Step 6: Run everything**

```bash
npm test
npm run typecheck
npm run test:browser
```

Expected: PASS. All three engines run `mask-png.spec.ts`; the new hit-test spec must pass in each. If an engine's nearest-neighbour upscale disagrees pixel-for-pixel at an exact 4x ratio, STOP and report the engine and the mismatch count — do not relax the assertion.

- [ ] **Step 7: Commit**

```bash
git add src/segmenter/worker/segmenter.worker.ts src/SegmentViewer.tsx tests
git commit -m "feat(segmenter): encode survivor masks at the decoder's own resolution"
```

---

### Task 3: Expose it, document it, and measure it on a real GPU (AC1, AC10, AC11)

The last steps of this task produce EVIDENCE, not code, and need a machine with a real WebGPU adapter; the sweep refuses to measure a software rasterizer, and that refusal is correct. **If no hardware adapter is available, commit through Step 6 and then STOP and report it — do not hand-write, estimate or carry over numbers.**

**Files:**
- Modify: `playground/compare.ts`, `playground/compare.test.ts`
- Modify: `playground/CompareView.tsx`, `playground/CompareView.test.tsx`
- Modify: `playground/SegmentView.tsx`, `playground/SegmentView.test.tsx`
- Modify: `scripts/sweep-decode.mjs`
- Modify: `docs/pipeline.md`
- Create: `docs/measurements/2026-09-01-mask-encode.config.json`
- Create: `docs/measurements/2026-09-01-low-res-mask-encode.md`
- Create (by the runner): one `docs/measurements/<date>-decode-sweep*.{md,json}` pair

**Interfaces:**
- Consumes from Task 1: `SegmenterOptions.lowResMaskEncode` (default `true`).
- Produces: `RowOptions.lowResMaskEncode: boolean`; `SweepConfig.lowResMaskEncode: readonly boolean[]` (default `[true]`); row ids of the form `p16-fp32-b32-none-lowres-enclow-r1`; a `low-res-mask-encode` checkbox on BOTH the Compare and the Segment tabs.

- [ ] **Step 1: Write the failing harness tests**

In `playground/compare.test.ts`, add `lowResMaskEncode: true,` to `rowOptions`'s defaults (after `lowResFilterNms: true,`), then append inside the config/grid `describe` that already holds the `lowResFilterNms` cases:

```ts
  it('walks the encode target as its own axis, innermost after the pipeline flag', () => {
    const rows = expandGrid(
      resolveConfig({
        decodePaths: ['none'],
        batchSizes: [32],
        dtypes: ['fp32'],
        pointsPerSide: [16],
        lowResFilterNms: [true],
        lowResMaskEncode: [false, true],
      }),
    );
    expect(rows.map((row) => row.id)).toEqual([
      'p16-fp32-b32-none-lowres-encfull-r1',
      'p16-fp32-b32-none-lowres-enclow-r1',
    ]);
    expect(rows.map((row) => row.options.lowResMaskEncode)).toEqual([false, true]);
  });

  it('defaults the encode axis to the shipped value only', () => {
    expect(DEFAULT_SWEEP_CONFIG.lowResMaskEncode).toEqual([true]);
    expect(expandGrid(DEFAULT_SWEEP_CONFIG)).toHaveLength(16);
  });

  it('rejects a lowResMaskEncode axis that is empty or not boolean', () => {
    expect(() => resolveConfig({ lowResMaskEncode: [] })).toThrow(/lowResMaskEncode/);
    expect(() => resolveConfig({ lowResMaskEncode: ['yes' as never] })).toThrow(
      /lowResMaskEncode/,
    );
  });

  it('labels and tabulates a row from the encode target it was captured with', () => {
    const record = okRecord('p16-fp32-b32-none-lowres-encfull-r1', 1000, {
      batchSize: 32,
      lowResMaskEncode: false,
    });
    expect(rowLabel(record)).toContain('enc full');
    // `meta` is the fixture the other toMarkdown tests in this file use.
    const md = toMarkdown([record], meta);
    expect(md).toContain('| pps | lowres | enc |');
    // Adjacent cells, so the new column lands beside `lowres` rather than
    // anywhere that happens to render.
    expect(md).toContain('| lowres | encfull |');
  });
```

The file's existing `'aligns the table from the column names, not a hardcoded index'` test already covers the alignment row against the widened header — it must keep passing untouched.

In `playground/CompareView.test.tsx`, append:

```ts
  it('defaults the lowResMaskEncode control on and passes it through (AC10)', async () => {
    const { seen } = mount(stubResult(1000));
    const control = screen.getByTestId('low-res-mask-encode') as HTMLInputElement;
    expect(control.checked).toBe(true);

    fireEvent.click(control);
    fireEvent.click(screen.getByTestId('run-row'));
    await waitFor(() => expect(seen).toHaveLength(1));
    expect(seen[0].lowResMaskEncode).toBe(false);
  });
```

In `playground/SegmentView.test.tsx`, append inside the controls `describe`:

```ts
  it('offers the encode-target control, seeded from the package default (AC2)', () => {
    render(<SegmentView />);
    const control = screen.getByTestId('low-res-mask-encode') as HTMLInputElement;
    // The tab where masks are actually clicked is where AC2 gets exercised by
    // hand, so the flag has to be reachable from here and not only from the
    // Compare tab.
    expect(control.checked).toBe(DEFAULT_SEGMENTER_OPTIONS.lowResMaskEncode);
    fireEvent.click(control);
    expect(control.checked).toBe(false);
  });
```

(`fireEvent` is already imported in that file.)

- [ ] **Step 2: Run them to verify they fail**

```bash
npx vitest run playground/compare.test.ts playground/CompareView.test.tsx playground/SegmentView.test.tsx
```

Expected: FAIL — unknown property `lowResMaskEncode`, missing row ids, no `low-res-mask-encode` element.

- [ ] **Step 3: Add the axis to `compare.ts`**

**(a)** `RowOptions`, after `lowResFilterNms: boolean;`:

```ts
  lowResMaskEncode: boolean;
```

**(b)** `SweepConfig`, after the `lowResFilterNms` field and its comment:

```ts
  /**
   * The encode-target axis. `[true]` by default, for the same reason as
   * `lowResFilterNms`: the default grid measures the SHIPPED pipeline and
   * stays 16 rows. `[false, true]` walks the before/after pair.
   */
  lowResMaskEncode: readonly boolean[];
```

**(c)** `DEFAULT_SWEEP_CONFIG`, after `lowResFilterNms: [true],`:

```ts
  lowResMaskEncode: [true],
```

**(d)** replace the inline `lowResFilterNms` validation in `resolveConfig` with a shared helper, so the second axis cannot drift from the first. Above `resolveConfig`, beside `positiveInts`:

```ts
function booleanAxis(name: string, values: readonly boolean[]): void {
  if (values.length === 0) throw new Error(`${name} must not be empty`);
  for (const value of values) {
    if (typeof value !== 'boolean') {
      throw new Error(`${name} must be booleans, got ${JSON.stringify(value)}`);
    }
  }
}
```

and in `resolveConfig` replace the whole `lowResFilterNms` validation block (the `if (config.lowResFilterNms.length === 0)` line through the closing brace of its `for` loop) with:

```ts
  booleanAxis('lowResFilterNms', config.lowResFilterNms);
  booleanAxis('lowResMaskEncode', config.lowResMaskEncode);
```

The message strings are unchanged, so the existing `lowResFilterNms` validation tests must still pass untouched — that is the check that this refactor is behaviour-preserving.

**(e)** `expandGrid`: nest the new axis INSIDE `lowResFilterNms` and outside `rep`, and extend the id:

```ts
          for (const lowResFilterNms of config.lowResFilterNms) {
            for (const lowResMaskEncode of config.lowResMaskEncode) {
              for (let rep = 1; rep <= config.reps; rep += 1) {
                rows.push({
                  id: `p${pointsPerSide}-${dtype}-b${batchSize}-${path}-${
                    lowResFilterNms ? 'lowres' : 'fullres'
                  }-${lowResMaskEncode ? 'enclow' : 'encfull'}-r${rep}`,
                  rep,
                  warmUp: false,
                  options: {
                    dtype,
                    batchSize,
                    pointsPerSide,
                    keepRawMasks: config.keepRawMasks,
                    lowResFilterNms,
                    lowResMaskEncode,
                    ...decodeFlags(path),
                  },
                });
              }
            }
          }
```

Update the nesting-order comment above `expandGrid` to name the new axis: `... then decode path, then lowResFilterNms, then lowResMaskEncode, then rep ...`. Update the `SweepRow.id` doc comment to `p<pps>-<dtype>-b<batch>-<path>-<pipeline>-<encode>-r<rep>`.

**(f)** `rowLabel`, appending to the template literal:

```ts
  return `${decodePathOf(o)} · ${o.dtype} · batch ${o.batchSize} · pps ${o.pointsPerSide} · ${
    o.lowResFilterNms ? 'lowres' : 'fullres'
  } · ${o.lowResMaskEncode ? 'enc low' : 'enc full'}`;
```

**(g)** the markdown table: add `'enc'` to `COLUMNS` immediately after `'lowres'`, add `'enc'` to `LEFT_ALIGNED`, and add the matching cell immediately after the `o.lowResFilterNms ? 'lowres' : 'fullres',` line in the row array:

```ts
        o.lowResMaskEncode ? 'enclow' : 'encfull',
```

- [ ] **Step 4: Add the two controls**

`playground/CompareView.tsx`:
- add `lowResMaskEncode: true,` to `INITIAL_OPTIONS` (after `lowResFilterNms: true,`);
- add the control immediately after the `low-res-filter-nms` label, keeping the `{' '}` separator style:

```tsx
        <label>
          low-res mask encode:{' '}
          <input
            data-testid="low-res-mask-encode"
            type="checkbox"
            checked={options.lowResMaskEncode}
            disabled={running}
            onChange={(e) => patch({ lowResMaskEncode: e.target.checked })}
          />
        </label>
```

- in the results table add `<th align="left">enc</th>` immediately after the `lowres` header, and the matching cell immediately after the `lowres` cell:

```tsx
                <td>{o.lowResMaskEncode ? 'enclow' : 'encfull'}</td>
```

(reading `record.options`, like every other cell — never live control state).

`playground/SegmentView.tsx`: add the same control after the `gpu-resident-embeddings` label (this view has no `disabled={running}` on its checkboxes — match the file, do not introduce one):

```tsx
        {' '}
        <label>
          low-res mask encode:{' '}
          <input
            data-testid="low-res-mask-encode"
            type="checkbox"
            checked={options.lowResMaskEncode}
            onChange={(e) => patch({ lowResMaskEncode: e.target.checked })}
          />
        </label>
```

`scripts/sweep-decode.mjs`, in `runOneRow` after the `low-res-filter-nms` line:

```js
  await setCheckbox(page, 'low-res-mask-encode', o.lowResMaskEncode);
```

- [ ] **Step 5: Run the tests and typechecks**

```bash
npm test
npm run typecheck
```

Expected: PASS. `npm test` must show the pre-existing `lowResFilterNms` validation tests still passing — the `booleanAxis` refactor is only safe if they do.

- [ ] **Step 6: Update `docs/pipeline.md` (AC11) and commit the code**

Three edits, all in place:

**(a)** In §2, the `**`mask-encode` — deleted.**` paragraph, replace the sentence `At 256×256 (**M2**) it is ~10× cheaper again.` with:

```
Encoding at the decoder's own window (**M2**) shrinks it again: the PNG is
written at
`min(round(lowWidth × reshapedWidth / padWidth), originalWidth)` ×
`min(round(lowHeight × reshapedHeight / padHeight), originalHeight)` — 256×162
on the 1024×649 sample, ~16× fewer pixels than full resolution rather than the
~10× first estimated here — and `SegmentViewer` upscales it back to image space
nearest-neighbour (`imageSmoothingEnabled = false`), which keeps every
read-back pixel exactly `(255,255,255,255)` or `(0,0,0,0)`.
```

**(b)** Immediately after the "Where the budget lands" table in §2, add:

```markdown
### What ships today

Waves 1 and 2 have landed, so this section is no longer a proposal for them:

- `filter` scores and thresholds at 256×256 and retains a copy of each chosen
  candidate's logit window; no full-resolution buffer is allocated in the batch
  loop (**F1**, **F3**, behind `lowResFilterNms`, default on).
- `nms` runs on 256×256 coverage with a bbox prefilter and bit-packed popcounts
  (**N1**, **N2**, **N3**).
- Only NMS survivors are resampled, in a single pass straight from the logits
  (**F2**), and the exact, unscaled `minMaskArea` is re-applied there — the gap
  between the `afterNms` and `returned` counts.
- `mask-encode` writes a 1-bit indexed PNG inside the worker (**M3**, **M4**)
  from a SECOND resample of the same logits at the encode target above
  (**M2**, behind `lowResMaskEncode`, default on). That second resample is
  timed into `resample`, not into `mask-encode`.
- The viewer still decodes the PNG back into an image-space coverage array;
  **M1**/**M5** and the WebGL2 renderer (**F4**) remain future work.
```

**(c)** In §4's status table, change issue #6's status to `landed` and issue #7's to `landed`.

```bash
git add playground scripts docs/pipeline.md
git commit -m "feat(playground): expose lowResMaskEncode to both tabs and the sweep"
```

- [ ] **Step 7: Write the AC1 grid config**

`docs/measurements/2026-09-01-mask-encode.config.json`:

```json
{
  "decodePaths": ["none"],
  "batchSizes": [32],
  "dtypes": ["fp32"],
  "pointsPerSide": [16, 32],
  "lowResFilterNms": [true],
  "lowResMaskEncode": [false, true],
  "keepRawMasks": false,
  "reps": 1
}
```

Every axis but the one under test is pinned to a SHIPPED default: `fp32`, `batchSize: 32` (`DEFAULT_SEGMENTER_OPTIONS.batchSize`), no decode-path flags, and `lowResFilterNms: true` — the honest comparison, and the only one where `lowResMaskEncode: true` does anything at all.

- [ ] **Step 8: Run the sweep (AC1)**

```bash
npm run sweep:decode -- --config docs/measurements/2026-09-01-mask-encode.config.json
```

Headed Chromium opens; leave it alone until it prints `wrote docs/measurements/…`. Five runs happen: a discarded warm-up (a copy of row 1, so `encfull` at 16 points per side) and four measured rows, in this order:

1. `p16-fp32-b32-none-lowres-encfull-r1`
2. `p16-fp32-b32-none-lowres-enclow-r1`
3. `p32-fp32-b32-none-lowres-encfull-r1`
4. `p32-fp32-b32-none-lowres-enclow-r1`

All four in ONE session, which is the point: the decode-sweep report already had to disclose a cold-first-row artifact, and differencing two separate runs would reintroduce exactly that.

- [ ] **Step 9: Pull the numbers out of the artifact**

Substitute the JSON filename the runner printed:

```bash
node -e '
const data = JSON.parse(require("node:fs").readFileSync(process.argv[1], "utf8"));
for (const row of data.rows) {
  if (row.status !== "ok") { console.log(row.rowId, "FAILED", row.message); continue; }
  const p = row.timings.phases;
  console.log(
    row.rowId,
    "mask-encode", p["mask-encode"].total.toFixed(1),
    "(count", p["mask-encode"].count + ", p50", p["mask-encode"].p50.toFixed(2) + ")",
    "resample", p.resample.total.toFixed(1),
    "budget", row.budgetMs.toFixed(1),
    "| afterNms", row.counts.afterNms, "returned", row.counts.returned,
  );
}
' docs/measurements/<the sweep>.json
```

(`node -e` evaluates as CommonJS regardless of `"type": "module"`, so `require` is available.)

**Read the counts first.** At each `pointsPerSide`, the `encfull` and `enclow` rows must report the SAME `afterNms` and the SAME `returned`. `lowResMaskEncode` selects a resample target; if the kept set moved, something is wrong in the implementation and no timing number from this run is worth reporting. If they differ, STOP and report it.

- [ ] **Step 10: Write the analysis**

Create `docs/measurements/2026-09-01-low-res-mask-encode.md`. It must NAME the generated artifacts it draws from and contain, with real numbers copied from them:

1. **Setup** — adapter identity (from the sweep's own header), image, the config filename, and the four row ids. State plainly that the first run was a discarded warm-up.
2. **AC1 — the encode saving.** A table with one row per operating point: `mask-encode` total and per-mask p50, `resample` total, and `budget`, for `encfull` and `enclow` at `pointsPerSide` 16 and 32, plus the delta and the ratio for each. Compute every ratio; do not estimate one. Then PROSE that judges whether this is an acceptable outcome and why, naming:
   - the pixel-count ratio the encode target implies for THIS image (state the image's dimensions and the encode dimensions the formula gives, and compare the measured `mask-encode` ratio against it — if the measured saving is materially smaller, say so and say what fixed per-mask cost is left);
   - where cost MOVED: `resample` now carries a second, smaller resample per survivor, so report its delta beside the `mask-encode` delta and say whether the net is a win at both operating points.
   A record that says only that a measurement happened does not satisfy AC1.
3. **AC7 — the accounting.** State that the second resample is timed into `resample`, and show it in the numbers: `resample` up, `mask-encode` down, both reported.
4. **AC8 on real data** — the `afterNms` and `returned` counts per row from Step 9, and the statement that they are identical across the encode axis at each operating point.
5. **Caveats, in this file** — not in a commit message. At minimum: one rep per row, so single-run noise is not quantified; the discarded warm-up and what it protects; and the boundary-precision cost this buys — mask edges quantise to the upscale factor (state it for this image), and because `SegmentViewer` derives each mask's `area` from the DECODED coverage, that quantisation can in principle reorder `hitTestAll`'s smallest-first result where two masks of nearly equal area overlap. Say whether anything in this run bears on that.

Every figure in the prose must be derived from the committed artifacts, including hedges — compute a ratio before writing "roughly". An openly illustrative calculation is allowed if it is labelled as one.

- [ ] **Step 11: Commit the evidence**

```bash
git add docs/measurements
git commit -m "docs(measurements): quantify the reduced mask-encode target on a real GPU"
```

---

## Acceptance criteria coverage

| AC | Where it is satisfied | How it can fail |
|---|---|---|
| AC1 | Task 3 Steps 8-10; `docs/measurements/2026-09-01-low-res-mask-encode.md` §2 | Missing an operating point, missing the `resample` column, or no verdict in prose. |
| AC2 (ui) | Task 2's hit-test spec in chromium/webkit/firefox (12 probes incl. all four corners, both ends of a 4px-wide bar, and a boundary pair) + a hand run on the Segment tab, whose `low-res-mask-encode` control Task 3 Step 4 adds | Any probe selecting a different id set from the reduced masks than from the full-resolution ones. |
| AC3 | Task 2 Step 1's `SHAPES` additions — `256x162` and `162x256` through `mask-encode.test.ts` (Node round-trip) and `mask-png.spec.ts` (three engines); 162 is not a multiple of 8 | A padding-bit or IHDR bug at the sizes the writer is now actually given. |
| AC4 | Task 1's `resolveEncodeSize` tests: landscape, portrait, square, half-up rounding, and the clamp | A `floor`/`ceil` instead of `round`, a swapped axis, or a missing clamp (the 200x150 case would return 256x192). |
| AC5 | Task 1's `createSegmenter` and `session-key` tests, plus `'hands back the survivor itself when the target is full resolution'` — the SAME object, so the PNG bytes cannot differ | The flag missing from the worker request, present in the session key, or the off path taking a second resample. |
| AC6 | Task 1's `'is full resolution when lowResFilterNms is off'` and the module-comment paragraph | The encode target surviving on a path whose candidates carry no logits — which would throw on the first survivor. |
| AC7 | Task 2 Step 5's ordered grep; the `resample` deltas in the report's §3 | `resolveEncodeMask` called below `timings.record('resample'`. |
| AC8 | Task 1's `'resamples the logit window at the encode size, leaving the survivor alone'`; Task 2 Step 5's `area: mask.area` grep; Task 3 Step 9's counts check on real data | `masks.push` reading `encodeMask.area`, or `resolveSurvivor` being skipped when the flag is on. |
| AC9 | Task 2's `nonBinary === 0` assertions, with the chromium-only smoothing control proving the assertion can fail | A missing `imageSmoothingEnabled = false`, or an engine that ignores it. |
| AC10 | Task 3's `compare.test.ts` grid/validation/markdown tests, both control tests, and `docs/measurements/2026-09-01-mask-encode.config.json` | A control the sweep cannot drive, a widened default grid, or an alignment row that no longer matches the header. |
| AC11 | Task 3 Step 6 | `docs/pipeline.md` still projecting M2 rather than describing it, or still marking #6/#7 open. |

## Notes for the reviewer

- **`SegmenterOptions` gains a public field with a `true` default** — the second one after `lowResFilterNms`. It belongs in the PR description as a deliberate API change, as does the `EncodedMask.maskUrl` contract note: **the mask PNG's pixel dimensions are no longer the image's**, and any consumer that assumed otherwise must scale (as `SegmentViewer` already does).
- **Sweep row ids change shape again** (`…-lowres-r1` → `…-lowres-enclow-r1`). Deliberate: a row must say what it measured, and every label is derived from the options the row was captured with.
- **`FilterNmsOptions` gains a required field**, so every literal must be updated — there are exactly two outside the worker (`mask-pipeline.test.ts`'s `OPTIONS`, `playground/boundary.ts`'s `pathOptions`), and the playground half of `npm run typecheck` is what catches a miss.
- **`booleanAxis` replaces two hand-rolled validation blocks** in `compare.ts` with identical messages; the pre-existing `lowResFilterNms` validation tests passing unchanged is the evidence that the refactor is behaviour-preserving.
- **AC2's browser test hit-tests rather than clicks.** The spec's testing section asks for "a browser test that clicks near image edges"; driving the real app would need a WebGPU adapter, a model download and a full segmentation run inside Playwright, which is what `playwright.config.ts` deliberately keeps out of this suite. Instead the spec exercises the exact code that decides a click — `hitTestAll`, over coverage produced by a real PNG decode and a real canvas upscale in three engines — and the running-app half of AC2 is the hand check on the Segment tab, whose control Task 3 adds. The substance of the criterion is covered; the mechanism differs, deliberately.
- **Known, accepted cost:** boundaries quantise to the upscale factor (~4 px on a 1024x649 photo), and the viewer's per-mask `area` comes from the decoded coverage, so overlapping masks of nearly equal area could in principle swap places in `hitTestAll`'s ordering. Measured rather than assumed: the browser spec pins the probe results, and the report's caveats section states the exposure.
