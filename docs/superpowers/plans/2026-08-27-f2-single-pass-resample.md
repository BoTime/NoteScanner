# F2 — Single-Pass Resample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the segmenter worker's two-pass `post_process_masks` upsample with one fused bilinear-resample-and-threshold that goes straight from a 256x256 logit window to a full-resolution `BinaryMask`.

**Architecture:** A new pure, synchronous module `src/segmenter/core/mask-resample.ts` exports `resampleThresholdMask`, which composes the two ONNX `half_pixel` `Resize` passes into a single coordinate map (`u = (X + 0.5) * sx - 0.5`, `sx = reshapedWidth * lowWidth / (originalWidth * padWidth)`) and binarizes in the same loop. The worker calls it once per chosen mask over a `subarray` view of the decoder output, dropping the staging copy, the 5-D `Tensor`, the ORT round-trip, the 1024^2 float intermediate and the full-resolution float buffer. The `filter` stage's sub-timers collapse from `select`/`upscale`/`threshold` to `select`/`resample`, because the last two regions are now one loop.

**Tech Stack:** TypeScript (strict), vitest (`environment: 'node'`), `@huggingface/transformers` (optional peer, worker only). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-27-f2-single-pass-resample-design.md`

## Global Constraints

- **`src/renderer/` is not touched.** F2 only; issue #9 (M5 / F4) is out of scope.
- **`thresholdMask` and `stabilityScore` stay exported and tested.** `thresholdMask` becomes unused by the worker but remains public API and is the documented reference for the strict-`>` semantics. Do not delete it.
- **`PHASE_ORDER` / `SegmentationPhase` are unchanged.** Only `FILTER_SUBSTEP_ORDER` / `FilterSubstep` change.
- **No estimated or invented timing numbers** anywhere — in code, comments, tests, the plan or the PR description. AC2's before/after table is produced by hand on real WebGPU hardware after review and posted to issue #8.
- **Pad size is resolved at runtime** from `image_processor.pad_size ?? image_processor.size`, never hardcoded to 1024.
- **Verification commands for this repo are `npm run typecheck` and `npm test`.** There is no `lint` script and no ESLint config in this repository — AC11's "lint" has nothing to run. `npm run typecheck` uses `tsconfig.json`, whose `include` is `src/**/*.ts(x)` and whose `exclude` lists `playground` — so **typecheck does not cover `playground/`**. Any claim about `playground/SegmentView.tsx` must be verified by reading or grepping the file, never by typecheck.
- **The `Write` tool is blocked in this worktree.** Create files with Bash heredocs (`cat > path <<'EOF'`).

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `src/segmenter/core/mask-resample.ts` | Create | `resampleThresholdMask` — the fused resample+threshold. Pure, synchronous, no ORT, no tensors, no async. |
| `src/segmenter/core/mask-resample.test.ts` | Create | The AC1 differential gate (in-test `twoPassReference`) plus the behaviour and guard unit tests. |
| `src/segmenter/core/index.ts` | Modify (line 4-ish) | Re-export the new module so the worker imports it from `../core` like everything else. |
| `src/segmenter/core/types.ts` | Modify (lines 19-28) | `FILTER_SUBSTEP_ORDER` becomes `['select', 'resample']`; doc comment follows. |
| `src/segmenter/core/timing.test.ts` | Modify (lines 64-117) | Two-substep order; the no-residual invariant now sums two regions. |
| `src/segmenter/createSegmenter.test.ts` | Modify (lines 115-141) | The passthrough test records `resample` instead of `upscale`. |
| `src/segmenter/worker/segmenter.worker.ts` | Modify (header comment, imports, lines 11-15, 123-126, 215-242) | Resolve pad size after encode; replace the `post_process_masks` block with one `resampleThresholdMask` loop; `recordSub('resample')`. |
| `playground/SegmentView.tsx` | **Verify only, no edit expected** | Already iterates `FILTER_SUBSTEP_ORDER` (lines 285-286). Confirm by grep; do not restructure. |

Two tasks, drawn so each is one reviewable diff: Task 1 adds a self-contained pure module with its own full test suite and touches nothing that already runs; Task 2 rewires the single consumer and renames the sub-timers. The rename and the rewiring are one task on purpose — `recordSub('resample')` does not typecheck until `FilterSubstep` includes `'resample'`, and `FILTER_SUBSTEP_ORDER` without the worker change would ship a permanently-zero row. Neither half can be reviewed or tested without the other.

---

### Task 1: `resampleThresholdMask` and its differential test

**Files:**
- Create: `src/segmenter/core/mask-resample.ts`
- Test: `src/segmenter/core/mask-resample.test.ts`
- Modify: `src/segmenter/core/index.ts`

**Interfaces:**
- Consumes: `BinaryMask` (`{ coverage: Uint8Array; area: number }`) and `thresholdMask` from `./mask-postprocess`.
- Produces, for Task 2:
  ```ts
  export interface ResampleThresholdMaskOptions {
    logits: Float32Array;
    lowWidth: number;
    lowHeight: number;
    padWidth: number;
    padHeight: number;
    reshapedWidth: number;
    reshapedHeight: number;
    originalWidth: number;
    originalHeight: number;
    threshold: number;
  }
  export function resampleThresholdMask(options: ResampleThresholdMaskOptions): BinaryMask;
  ```

- [ ] **Step 1: Write the failing test file**

Create `src/segmenter/core/mask-resample.test.ts` with exactly this content.

Why these constants, so a later reader does not "tidy" them into round numbers:

- `padWidth === reshapedWidth` on the landscape geometry and `padHeight === reshapedHeight` on the portrait one, because SAM's reshape always makes the **long** side fill the pad. The short side is what makes the sampled window fractional: `reshapedHeight * lowHeight / padHeight = 181 * 64 / 256 = 45.25`. A fractional window is precisely the case an integer crop of the low-res grid would get wrong, so no geometry here may be chosen to make it land on an integer.
- The third geometry has a **non-square low grid** (`lowWidth 40`, `lowHeight 64`) with a matching non-square pad. Row indexing is `y * lowWidth`; a transposed axis survives every square-grid case and dies here.
- `originalWidth`/`originalHeight` are odd, non-multiples of the low dimensions (211x149, 143x205, 121x131) so no output pixel centre lands exactly on a source sample and every case exercises a real interpolation weight.
- The disc and blob are placed inside the sampled window (centres at ~0.32/0.30 and ~0.42/0.16 of the low grid) so their bounding boxes are interior on all four sides — a shape clipped by the image edge cannot detect a bbox error on that side. The diagonal ramp is deliberately a half-plane that *does* reach the edges; it is the case that produces the largest divergence.

```ts
import { describe, it, expect, afterAll } from 'vitest';
import { resampleThresholdMask } from './mask-resample';
import { thresholdMask } from './mask-postprocess';

const clamp = (value: number, lo: number, hi: number) =>
  value < lo ? lo : value > hi ? hi : value;

/**
 * ONNX `Resize` with `mode="linear"` and the default `half_pixel` coordinate
 * transform — the node `TensorOpRegistry.bilinear_interpolate_4d` builds, and
 * therefore what both of `post_process_masks`' `interpolate_4d` calls do.
 */
function bilinearHalfPixel(
  src: Float32Array,
  srcWidth: number,
  srcHeight: number,
  dstWidth: number,
  dstHeight: number,
): Float32Array {
  const out = new Float32Array(dstWidth * dstHeight);
  const sx = srcWidth / dstWidth;
  const sy = srcHeight / dstHeight;
  const maxX0 = srcWidth >= 2 ? srcWidth - 2 : 0;
  const maxY0 = srcHeight >= 2 ? srcHeight - 2 : 0;
  for (let y = 0; y < dstHeight; y += 1) {
    const v = clamp((y + 0.5) * sy - 0.5, 0, srcHeight - 1);
    const y0 = Math.min(Math.floor(v), maxY0);
    const y1 = Math.min(y0 + 1, srcHeight - 1);
    const wy = clamp(v - y0, 0, 1);
    for (let x = 0; x < dstWidth; x += 1) {
      const u = clamp((x + 0.5) * sx - 0.5, 0, srcWidth - 1);
      const x0 = Math.min(Math.floor(u), maxX0);
      const x1 = Math.min(x0 + 1, srcWidth - 1);
      const wx = clamp(u - x0, 0, 1);
      const top = src[y0 * srcWidth + x0] + (src[y0 * srcWidth + x1] - src[y0 * srcWidth + x0]) * wx;
      const bottom =
        src[y1 * srcWidth + x0] + (src[y1 * srcWidth + x1] - src[y1 * srcWidth + x0]) * wx;
      out[y * dstWidth + x] = top + (bottom - top) * wy;
    }
  }
  return out;
}

/** The `.slice(0..h, 0..w)` between the two `interpolate_4d` calls. */
function cropTopLeft(
  src: Float32Array,
  srcWidth: number,
  width: number,
  height: number,
): Float32Array {
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    out.set(src.subarray(y * srcWidth, y * srcWidth + width), y * width);
  }
  return out;
}

interface Geometry {
  name: string;
  lowWidth: number;
  lowHeight: number;
  padWidth: number;
  padHeight: number;
  reshapedWidth: number;
  reshapedHeight: number;
  originalWidth: number;
  originalHeight: number;
}

/**
 * Today's chain, in plain JS: bilinear to `pad`, integer crop to `reshaped`,
 * bilinear to `original`, then threshold. This is the baseline AC1 measures
 * against, and it exists here rather than being imported because the shipped
 * path is exactly what we are replacing.
 */
function twoPassReference(geometry: Geometry, logits: Float32Array, threshold: number) {
  const padded = bilinearHalfPixel(
    logits,
    geometry.lowWidth,
    geometry.lowHeight,
    geometry.padWidth,
    geometry.padHeight,
  );
  const cropped = cropTopLeft(
    padded,
    geometry.padWidth,
    geometry.reshapedWidth,
    geometry.reshapedHeight,
  );
  const full = bilinearHalfPixel(
    cropped,
    geometry.reshapedWidth,
    geometry.reshapedHeight,
    geometry.originalWidth,
    geometry.originalHeight,
  );
  return thresholdMask(full, threshold);
}

function iou(a: Uint8Array, b: Uint8Array): number {
  let intersection = 0;
  let union = 0;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] || b[i]) union += 1;
    if (a[i] && b[i]) intersection += 1;
  }
  return union === 0 ? 1 : intersection / union;
}

/**
 * `[minX, minY, maxX, maxY]`. An empty mask yields Infinity sentinels, which is
 * why every differential case asserts a non-zero area before comparing boxes.
 */
function bbox(coverage: Uint8Array, width: number): [number, number, number, number] {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < coverage.length; i += 1) {
    if (!coverage[i]) continue;
    const x = i % width;
    const y = Math.floor(i / width);
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return [minX, minY, maxX, maxY];
}

/** A half-plane crossing zero on the diagonal: reaches every image edge. */
function diagonalRamp(width: number, height: number): Float32Array {
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      out[y * width + x] = 4 * (x / (width - 1) + y / (height - 1) - 1);
    }
  }
  return out;
}

/** A disc inside the sampled window, so its bbox is interior on all four sides. */
function centredDisc(width: number, height: number): Float32Array {
  const out = new Float32Array(width * height);
  const cx = width * 0.32;
  const cy = height * 0.3;
  const radius = Math.min(width, height) * 0.17;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      out[y * width + x] = radius - Math.hypot(x - cx, y - cy);
    }
  }
  return out;
}

/** A smooth off-centre gaussian blob, also interior to the window. */
function offCentreBlob(width: number, height: number): Float32Array {
  const out = new Float32Array(width * height);
  const cx = width * 0.42;
  const cy = height * 0.16;
  const sigma = Math.min(width, height) * 0.1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const d = ((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma);
      out[y * width + x] = 6 * Math.exp(-d) - 2;
    }
  }
  return out;
}

const GEOMETRIES: Geometry[] = [
  // Landscape: the long side fills the pad exactly (as SAM's reshape always
  // does), so the fractional window is on the y axis — 181 * 64 / 256 = 45.25.
  {
    name: 'landscape',
    lowWidth: 64,
    lowHeight: 64,
    padWidth: 256,
    padHeight: 256,
    reshapedWidth: 256,
    reshapedHeight: 181,
    originalWidth: 211,
    originalHeight: 149,
  },
  // Portrait: mirror image — the fractional window is on x, 179 * 64 / 256.
  {
    name: 'portrait',
    lowWidth: 64,
    lowHeight: 64,
    padWidth: 256,
    padHeight: 256,
    reshapedWidth: 179,
    reshapedHeight: 256,
    originalWidth: 143,
    originalHeight: 205,
  },
  // Non-square low grid and non-square pad: catches a transposed axis, which
  // every square-grid case above would silently pass.
  {
    name: 'non-square-low',
    lowWidth: 40,
    lowHeight: 64,
    padWidth: 160,
    padHeight: 256,
    reshapedWidth: 160,
    reshapedHeight: 173,
    originalWidth: 121,
    originalHeight: 131,
  },
];

const FIELDS: Array<[string, (w: number, h: number) => Float32Array]> = [
  ['diagonal ramp', diagonalRamp],
  ['centred disc', centredDisc],
  ['off-centre blob', offCentreBlob],
];

// AC1 wants the divergence recorded as a number, not asserted away. Collected
// here and printed once at the end of the file's run.
const divergences: number[] = [];
let maxBboxDelta = 0;

afterAll(() => {
  if (divergences.length === 0) return;
  const max = Math.max(...divergences);
  const mean = divergences.reduce((sum, d) => sum + d, 0) / divergences.length;
  console.log(
    `[mask-resample AC1] one-pass vs two-pass over ${divergences.length} cases: ` +
      `max divergence (1 - IoU) ${max.toExponential(3)}, mean ${mean.toExponential(3)}, ` +
      `max bounding-box delta ${maxBboxDelta} px`,
  );
});

describe('resampleThresholdMask vs the two-pass reference', () => {
  for (const geometry of GEOMETRIES) {
    for (const [fieldName, field] of FIELDS) {
      it(`agrees with the two-pass chain on a ${fieldName} at ${geometry.name}`, () => {
        const logits = field(geometry.lowWidth, geometry.lowHeight);
        const onePass = resampleThresholdMask({ ...geometry, logits, threshold: 0 });
        const twoPass = twoPassReference(geometry, logits, 0);

        // A case where both masks are empty would pass every assertion below
        // while proving nothing, so pin that shut first.
        expect(onePass.area).toBeGreaterThan(0);
        expect(twoPass.area).toBeGreaterThan(0);

        const overlap = iou(onePass.coverage, twoPass.coverage);
        divergences.push(1 - overlap);
        expect(overlap).toBeGreaterThanOrEqual(0.99);

        const a = bbox(onePass.coverage, geometry.originalWidth);
        const b = bbox(twoPass.coverage, geometry.originalWidth);
        for (let side = 0; side < 4; side += 1) {
          const delta = Math.abs(a[side] - b[side]);
          if (delta > maxBboxDelta) maxBboxDelta = delta;
          expect(delta).toBeLessThanOrEqual(1);
        }
      });
    }
  }
});

describe('resampleThresholdMask geometry and degenerate inputs', () => {
  it('samples a single low-res column when lowWidth is 1', () => {
    const mask = resampleThresholdMask({
      logits: Float32Array.from([-1, -1, 1, 1]),
      lowWidth: 1,
      lowHeight: 4,
      padWidth: 1,
      padHeight: 4,
      reshapedWidth: 1,
      reshapedHeight: 4,
      originalWidth: 3,
      originalHeight: 4,
      threshold: 0,
    });
    // Every column collapses onto the one source column, so the top two rows
    // are uncovered and the bottom two are covered, all the way across.
    expect(Array.from(mask.coverage)).toEqual([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
    expect(mask.area).toBe(6);
  });

  it('samples a single low-res row when lowHeight is 1', () => {
    const mask = resampleThresholdMask({
      logits: Float32Array.from([-1, -1, 1, 1]),
      lowWidth: 4,
      lowHeight: 1,
      padWidth: 4,
      padHeight: 1,
      reshapedWidth: 4,
      reshapedHeight: 1,
      originalWidth: 4,
      originalHeight: 3,
      threshold: 0,
    });
    expect(Array.from(mask.coverage)).toEqual([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]);
    expect(mask.area).toBe(6);
  });

  it('handles a 1x1 output by averaging the four corners', () => {
    const mask = resampleThresholdMask({
      logits: Float32Array.from([0, 0, 0, 4]),
      lowWidth: 2,
      lowHeight: 2,
      padWidth: 2,
      padHeight: 2,
      reshapedWidth: 2,
      reshapedHeight: 2,
      originalWidth: 1,
      originalHeight: 1,
      threshold: 0,
    });
    // The single output centre lands at (0.5, 0.5) in source space: the mean
    // of the four corners is 1, which is above the threshold.
    expect(Array.from(mask.coverage)).toEqual([1]);
    expect(mask.area).toBe(1);
  });

  it('returns an empty mask when every logit is below the threshold', () => {
    const mask = resampleThresholdMask({
      logits: new Float32Array(16).fill(-3),
      lowWidth: 4,
      lowHeight: 4,
      padWidth: 16,
      padHeight: 16,
      reshapedWidth: 13,
      reshapedHeight: 16,
      originalWidth: 5,
      originalHeight: 3,
      threshold: 0,
    });
    expect(mask.area).toBe(0);
    expect(mask.coverage.length).toBe(15);
    expect(Array.from(mask.coverage).every((v) => v === 0)).toBe(true);
  });

  it('leaves a logit exactly at the threshold uncovered, agreeing with thresholdMask', () => {
    // Bilinear interpolation of a constant field is exactly that constant, so
    // every output sample is exactly 0.5 — the strict `>` must reject it.
    const mask = resampleThresholdMask({
      logits: new Float32Array(16).fill(0.5),
      lowWidth: 4,
      lowHeight: 4,
      padWidth: 16,
      padHeight: 16,
      reshapedWidth: 13,
      reshapedHeight: 16,
      originalWidth: 7,
      originalHeight: 9,
      threshold: 0.5,
    });
    expect(mask.area).toBe(0);
    expect(thresholdMask(Float32Array.from([0.5]), 0.5).area).toBe(0);
  });

  it('always sizes coverage as originalWidth * originalHeight', () => {
    const mask = resampleThresholdMask({
      logits: diagonalRamp(8, 8),
      lowWidth: 8,
      lowHeight: 8,
      padWidth: 32,
      padHeight: 32,
      reshapedWidth: 32,
      reshapedHeight: 23,
      originalWidth: 19,
      originalHeight: 13,
      threshold: 0,
    });
    expect(mask.coverage.length).toBe(19 * 13);
  });

  it('throws on a non-positive or non-integer dimension', () => {
    const base = {
      logits: new Float32Array(4),
      lowWidth: 2,
      lowHeight: 2,
      padWidth: 2,
      padHeight: 2,
      reshapedWidth: 2,
      reshapedHeight: 2,
      originalWidth: 2,
      originalHeight: 2,
      threshold: 0,
    };
    expect(() => resampleThresholdMask({ ...base, lowWidth: 0 })).toThrow(/lowWidth/);
    expect(() => resampleThresholdMask({ ...base, originalHeight: -1 })).toThrow(
      /originalHeight/,
    );
    expect(() => resampleThresholdMask({ ...base, padWidth: 1.5 })).toThrow(/padWidth/);
  });

  it('throws when logits.length does not match lowWidth * lowHeight', () => {
    expect(() =>
      resampleThresholdMask({
        logits: new Float32Array(5),
        lowWidth: 2,
        lowHeight: 2,
        padWidth: 2,
        padHeight: 2,
        reshapedWidth: 2,
        reshapedHeight: 2,
        originalWidth: 2,
        originalHeight: 2,
        threshold: 0,
      }),
    ).toThrow(/logits\.length/);
  });

  it('does not mutate its input, including when it is a subarray view', () => {
    // The worker passes a window into one big decoder-output buffer, so the
    // whole surrounding buffer has to come back untouched too.
    const buffer = new Float32Array(3 * 16);
    for (let i = 0; i < buffer.length; i += 1) buffer[i] = Math.sin(i) * 2;
    const before = Array.from(buffer);
    const window = buffer.subarray(16, 32);

    resampleThresholdMask({
      logits: window,
      lowWidth: 4,
      lowHeight: 4,
      padWidth: 16,
      padHeight: 16,
      reshapedWidth: 13,
      reshapedHeight: 16,
      originalWidth: 11,
      originalHeight: 9,
      threshold: 0,
    });

    expect(Array.from(buffer)).toEqual(before);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm test -- src/segmenter/core/mask-resample.test.ts`
Expected: FAIL — the whole file errors with a resolution failure on `./mask-resample` ("Failed to resolve import" / "Cannot find module"), because the module does not exist yet.

- [ ] **Step 3: Write the implementation**

Create `src/segmenter/core/mask-resample.ts` with exactly this content.

```ts
import type { BinaryMask } from './mask-postprocess';

export interface ResampleThresholdMaskOptions {
  /**
   * One mask's low-resolution logit window, row-major, exactly
   * `lowWidth * lowHeight` long. May be a `subarray` view into a much larger
   * tensor buffer; it is only ever read.
   */
  logits: Float32Array;
  /** Width of the logit grid, in samples. Positive integer. */
  lowWidth: number;
  /** Height of the logit grid, in samples. Positive integer. */
  lowHeight: number;
  /** Width the processor padded the input to. Positive integer. */
  padWidth: number;
  /** Height the processor padded the input to. Positive integer. */
  padHeight: number;
  /** Width of the resized (pre-pad) image inside the padded square. Positive integer. */
  reshapedWidth: number;
  /** Height of the resized (pre-pad) image inside the padded square. Positive integer. */
  reshapedHeight: number;
  /** Width of the output mask, in pixels. Positive integer. */
  originalWidth: number;
  /** Height of the output mask, in pixels. Positive integer. */
  originalHeight: number;
  /** Logit value a sample must exceed — strictly — to count as covered. */
  threshold: number;
}

const DIMENSION_KEYS = [
  'lowWidth',
  'lowHeight',
  'padWidth',
  'padHeight',
  'reshapedWidth',
  'reshapedHeight',
  'originalWidth',
  'originalHeight',
] as const;

function clamp(value: number, lo: number, hi: number): number {
  return value < lo ? lo : value > hi ? hi : value;
}

/**
 * Resample one low-resolution logit window straight to a full-resolution
 * binary mask, in a single pass.
 *
 * This replaces `SamProcessor.post_process_masks` + `thresholdMask`, which
 * together do: a bilinear resize of the logit grid up to the padded square, a
 * crop of that to the reshaped image, a second bilinear resize to the original
 * size, and then a separate pass to binarize. Both resizes are ONNX `Resize`
 * nodes with `mode="linear"` and no `coordinate_transformation_mode`, so both
 * use the ONNX default, `half_pixel`:
 *
 *     src = (dst + 0.5) / scale - 0.5,   scale = outSize / inSize
 *
 * Composing the two on one axis, the +0.5 and -0.5 cancel exactly and leave a
 * single map with no intermediate:
 *
 *     u = (X + 0.5) * (reshapedWidth * lowWidth) / (originalWidth * padWidth) - 0.5
 *
 * which is one `half_pixel` bilinear resample of the top-left
 * `(reshapedWidth * lowWidth / padWidth)` x `(reshapedHeight * lowHeight / padHeight)`
 * window of the logit grid. That window is fractional in general — it is NOT
 * an integer crop of the low-res grid, and implementing it as one would be
 * wrong.
 *
 * The composition is exact for the coordinate map, but the result is not
 * bit-identical to the two-pass chain: the intermediate padded grid quantises
 * the interpolated ramp wherever a sample's neighbourhood straddles a source
 * cell boundary. The residual is small and nonzero;
 * `mask-resample.test.ts` measures it against an in-test two-pass reference
 * and prints the observed max and mean, so the difference is a number rather
 * than an assumption.
 *
 * Binarization is fused into the same loop and uses a strict `>`, matching
 * `thresholdMask` (and transformers.js's own `post_process_masks`): a logit
 * exactly equal to `threshold` is NOT covered.
 *
 * Allocates only the returned `coverage` plus three small per-column lookup
 * arrays — no padded-resolution float intermediate, no full-resolution float
 * buffer, and no second memory pass to binarize.
 *
 * Bad values fail loudly rather than producing a plausible-looking wrong mask:
 * every dimension must be a positive integer (a fractional `lowWidth` would
 * silently corrupt row indexing) and `logits.length` must equal
 * `lowWidth * lowHeight`. The one input that is not validated is `threshold`,
 * because any finite value is legitimate — but note that a `NaN` threshold
 * makes every comparison false and returns an empty mask rather than throwing.
 */
export function resampleThresholdMask(options: ResampleThresholdMaskOptions): BinaryMask {
  const {
    logits,
    lowWidth,
    lowHeight,
    padWidth,
    padHeight,
    reshapedWidth,
    reshapedHeight,
    originalWidth,
    originalHeight,
    threshold,
  } = options;

  for (const key of DIMENSION_KEYS) {
    const value = options[key];
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(
        `resampleThresholdMask: ${key} must be a positive integer, received ${value}`,
      );
    }
  }
  if (logits.length !== lowWidth * lowHeight) {
    throw new Error(
      `resampleThresholdMask: logits.length ${logits.length} does not match ` +
        `lowWidth * lowHeight (${lowWidth} * ${lowHeight} = ${lowWidth * lowHeight})`,
    );
  }

  const sx = (reshapedWidth * lowWidth) / (originalWidth * padWidth);
  const sy = (reshapedHeight * lowHeight) / (originalHeight * padHeight);

  // Every output row samples the same set of columns, so the column indices
  // and weights are computed once here rather than originalHeight times.
  const columnLeft = new Int32Array(originalWidth);
  const columnRight = new Int32Array(originalWidth);
  const columnWeight = new Float32Array(originalWidth);
  // With a single-sample axis there is no neighbour to interpolate towards, so
  // both indices collapse onto 0 and the weight is 0.
  const maxLeft = lowWidth >= 2 ? lowWidth - 2 : 0;
  for (let x = 0; x < originalWidth; x += 1) {
    const u = clamp((x + 0.5) * sx - 0.5, 0, lowWidth - 1);
    const x0 = Math.min(Math.floor(u), maxLeft);
    columnLeft[x] = x0;
    columnRight[x] = Math.min(x0 + 1, lowWidth - 1);
    columnWeight[x] = clamp(u - x0, 0, 1);
  }

  const coverage = new Uint8Array(originalWidth * originalHeight);
  let area = 0;
  const maxTop = lowHeight >= 2 ? lowHeight - 2 : 0;

  for (let y = 0; y < originalHeight; y += 1) {
    const v = clamp((y + 0.5) * sy - 0.5, 0, lowHeight - 1);
    const y0 = Math.min(Math.floor(v), maxTop);
    const y1 = Math.min(y0 + 1, lowHeight - 1);
    const wy = clamp(v - y0, 0, 1);
    const topRow = y0 * lowWidth;
    const bottomRow = y1 * lowWidth;
    const outRow = y * originalWidth;

    for (let x = 0; x < originalWidth; x += 1) {
      const x0 = columnLeft[x];
      const x1 = columnRight[x];
      const wx = columnWeight[x];
      const topLeft = logits[topRow + x0];
      const bottomLeft = logits[bottomRow + x0];
      const top = topLeft + (logits[topRow + x1] - topLeft) * wx;
      const bottom = bottomLeft + (logits[bottomRow + x1] - bottomLeft) * wx;
      if (top + (bottom - top) * wy > threshold) {
        coverage[outRow + x] = 1;
        area += 1;
      }
    }
  }

  return { coverage, area };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npm test -- src/segmenter/core/mask-resample.test.ts`
Expected: PASS, all 18 tests. The console line
`[mask-resample AC1] one-pass vs two-pass over 9 cases: max divergence (1 - IoU) ...` must appear in the output — **quote that line in the task report**, because it is the number AC1 asks for. If it does not appear, the `afterAll` hook is broken and the criterion is unrecorded; fix that before moving on.

- [ ] **Step 5: Export the module from the core barrel**

In `src/segmenter/core/index.ts`, add the new module immediately after the `mask-postprocess` line, so the file reads:

```ts
export * from './types';
export * from './point-grid';
export * from './mask-postprocess';
export * from './mask-resample';
export { dedupeMasks, pairwiseIoU, type BinaryMask } from './nms';
export * from './mask-encode';
export * from './timing';
```

`mask-resample.ts` imports `BinaryMask` as a type and does not re-export it, so this adds no second `BinaryMask` export and no ambiguity with the explicit one from `./nms`.

- [ ] **Step 6: Run the full suite and typecheck**

Run: `npm run typecheck && npm test`
Expected: typecheck clean; every existing test still passes alongside the new file. Nothing outside `src/segmenter/core/` has changed yet, so any failure elsewhere is a real regression, not fallout.

- [ ] **Step 7: Commit**

```bash
git add src/segmenter/core/mask-resample.ts src/segmenter/core/mask-resample.test.ts src/segmenter/core/index.ts
git commit -m "feat(segmenter): fused single-pass mask resample and threshold"
```

---

### Task 2: Rewire the worker and collapse the filter sub-timers

**Files:**
- Modify: `src/segmenter/core/types.ts:19-28`
- Modify: `src/segmenter/core/timing.test.ts:64-117`
- Modify: `src/segmenter/createSegmenter.test.ts:115-141`
- Modify: `src/segmenter/worker/segmenter.worker.ts` (header comment, import block, lines 123-126, 215-242)
- Verify only: `playground/SegmentView.tsx:283-303`

**Interfaces:**
- Consumes from Task 1: `resampleThresholdMask(options: ResampleThresholdMaskOptions): BinaryMask`, re-exported from `../core`.
- Produces: `FILTER_SUBSTEP_ORDER = ['select', 'resample'] as const` and `type FilterSubstep = 'select' | 'resample'`, both public through `note-scanner/segmenter`.

This is a deliberate public API change (`FILTER_SUBSTEP_ORDER` narrows, and `'upscale'` / `'threshold'` stop being valid `FilterSubstep` values). It belongs in the PR description.

- [ ] **Step 1: Update the timing tests to the two-substep order**

In `src/segmenter/core/timing.test.ts`, replace the entire `describe('createTimingAccumulator filter sub-steps', ...)` block (lines 64-117) with:

```ts
describe('createTimingAccumulator filter sub-steps', () => {
  it('reports every sub-step in FILTER_SUBSTEP_ORDER, zero-filled when never recorded', () => {
    const report = createTimingAccumulator().report(0);
    expect(Object.keys(report.filterSubPhases)).toEqual([...FILTER_SUBSTEP_ORDER]);
    for (const step of FILTER_SUBSTEP_ORDER) {
      expect(report.filterSubPhases[step].count).toBe(0);
    }
  });

  it('accumulates repeated samples for one sub-step', () => {
    const timings = createTimingAccumulator();
    timings.recordFilterSub('resample', 10);
    timings.recordFilterSub('resample', 30);
    timings.recordFilterSub('resample', 20);
    const report = timings.report(60);
    expect(report.filterSubPhases.resample.count).toBe(3);
    expect(report.filterSubPhases.resample.p50).toBe(20);
    expect(report.filterSubPhases.resample.max).toBe(30);
    expect(report.filterSubPhases.resample.total).toBe(60);
    expect(report.filterSubPhases.select.count).toBe(0);
  });

  it('keeps sub-steps independent of the phases they nest inside', () => {
    const timings = createTimingAccumulator();
    timings.record('filter', 100);
    timings.recordFilterSub('select', 3);
    const report = timings.report(100);
    expect(report.phases.filter.total).toBe(100);
    expect(report.filterSubPhases.select.total).toBe(3);
    expect(report.filterSubPhases.resample.total).toBe(0);
  });

  // The no-residual invariant: the worker's two regions are drawn to be
  // exhaustive and non-overlapping across the stage, so their totals sum to
  // the filter total. A residual is exactly what would muddy the percentage
  // breakdown this instrumentation exists to produce.
  it('has the two sub-step totals sum to the filter total with no residual', () => {
    const timings = createTimingAccumulator();
    timings.record('filter', 100);
    timings.recordFilterSub('select', 3);
    timings.recordFilterSub('resample', 97);
    timings.record('filter', 50);
    timings.recordFilterSub('select', 2);
    timings.recordFilterSub('resample', 48);
    const report = timings.report(150);
    const subTotal = FILTER_SUBSTEP_ORDER.reduce(
      (sum, step) => sum + report.filterSubPhases[step].total,
      0,
    );
    expect(subTotal).toBe(report.phases.filter.total);
  });
});
```

- [ ] **Step 2: Update the createSegmenter passthrough test**

In `src/segmenter/createSegmenter.test.ts`, replace the body of
`it('carries the worker filterSubPhases through the rebuilt report', ...)` (lines 115-141) with:

```ts
  it('carries the worker filterSubPhases through the rebuilt report', async () => {
    const workerTimings = createTimingAccumulator();
    workerTimings.recordFilterSub('select', 4);
    workerTimings.recordFilterSub('select', 6);
    workerTimings.recordFilterSub('resample', 100);

    const segmenter = createSegmenter({ createWorker: spawn });
    const pending = segmenter.segment(fakeBitmap());
    FakeWorker.instances[0].emit({
      type: 'done',
      masks: [],
      width: 2,
      height: 1,
      timings: workerTimings.report(42),
      counts: { raw: 24, afterFilter: 9, afterNms: 0 },
    });

    const result = await pending;
    // Every sub-step key survives the rebuild, in order, with the worker's
    // own numbers. The zero-fill rule itself belongs to the accumulator and
    // is tested where it lives, in `core/timing.test.ts`.
    expect(Object.keys(result.timings.filterSubPhases)).toEqual([...FILTER_SUBSTEP_ORDER]);
    expect(result.timings.filterSubPhases.select.total).toBe(10);
    expect(result.timings.filterSubPhases.select.p50).toBe(6);
    expect(result.timings.filterSubPhases.resample.total).toBe(100);
    expect(result.timings.filterSubPhases.resample.p50).toBe(100);
  });
```

- [ ] **Step 3: Run both test files to verify they fail**

Run: `npm test -- src/segmenter/core/timing.test.ts src/segmenter/createSegmenter.test.ts`
Expected: FAIL. `FILTER_SUBSTEP_ORDER` is still `['select', 'upscale', 'threshold']`, so `Object.keys(...)` comparisons mismatch and `filterSubPhases.resample` is `undefined` (a `TypeError` reading `.count` / `.total`). Under vitest these are runtime failures even though `recordFilterSub('resample', ...)` is also a type error — confirm you see failures, not a silent pass.

- [ ] **Step 4: Narrow FILTER_SUBSTEP_ORDER**

In `src/segmenter/core/types.ts`, replace lines 19-26 (the doc comment and the constant) with:

```ts
/**
 * The internal split of the `filter` stage, in render order. Deliberately NOT
 * folded into `PHASE_ORDER`: `SegmentationPhase` is public API and also types
 * `SegmenterFailure.phase`, so widening it would admit values that can never
 * be thrown, and summing the results table's total column would count
 * `filter` twice.
 *
 * `resample` is one region, not two: the upsample and the threshold are a
 * single fused loop in `resampleThresholdMask`, and reporting them separately
 * would ship a permanently-zero row.
 */
export const FILTER_SUBSTEP_ORDER = ['select', 'resample'] as const;
```

Leave `export type FilterSubstep = (typeof FILTER_SUBSTEP_ORDER)[number];` on the next line untouched — it narrows automatically. `src/segmenter/core/timing.ts` keys both accumulators off the constant and zero-fills by iterating it, so it needs no edit; confirm by reading lines 60-73 rather than assuming.

- [ ] **Step 5: Rewire the worker**

Four edits in `src/segmenter/worker/segmenter.worker.ts`.

**(a) The module header.** Replace lines 11-15 (the "Pipeline:" paragraph) with:

```ts
 * Pipeline: load once -> encode the image once -> decode the prompt grid in
 * batches -> filter each batch at LOW resolution -> resample only the
 * survivors straight to binary masks -> NMS across everything. Filtering
 * before resampling is not an optimization detail: a batch of 8 points yields
 * 24 low-res masks, and taking all of them to full image resolution is
 * hundreds of megabytes per batch.
```

**(b) The import block.** In the `from '../core'` list, drop `thresholdMask,` (the worker no longer binarizes a full-resolution buffer; the function stays exported and tested for other consumers) and add `resampleThresholdMask,` in its place, keeping the list's existing value-then-type grouping and alphabetical order within the value group:

```ts
import {
  batchPoints,
  buildPointGrid,
  createTimingAccumulator,
  dedupeMasks,
  encodeMaskPng,
  resampleThresholdMask,
  stabilityScore,
  type BinaryMask,
  type EncodedMask,
  type FilterSubstep,
  type NmsComparison,
  type SegmentationPhase,
  type SegmenterOptions,
  type SegmenterRequest,
  type SegmenterResponse,
} from '../core';
```

`Tensor` stays imported from `@huggingface/transformers` — it still builds `input_points` and `input_labels`. `processor` stays too; only the `post_process_masks` call goes away.

**(c) Pad-size resolution.** Add this helper immediately after the `Session` interface (after line 64, before the `let session` block):

```ts
interface PadSize {
  height: number;
  width: number;
}

/**
 * The pad size the processor actually applied, resolved the way
 * `post_process_masks` resolves it internally (`pad_size ?? size`, each
 * `{height, width}`).
 *
 * Read at runtime rather than hardcoded to 1024: the fused resample's geometry
 * is only correct for the pad the processor used, and a different SAM
 * checkpoint may ship a different one. Both fields are typed `any` upstream,
 * so this narrows them itself rather than trusting the declaration.
 */
function resolvePadSize(processor: SamProcessor): PadSize {
  const imageProcessor = processor.image_processor as
    | { pad_size?: unknown; size?: unknown }
    | undefined;
  const candidate = imageProcessor?.pad_size ?? imageProcessor?.size;
  const size = candidate as Partial<PadSize> | undefined;
  if (!size || typeof size.height !== 'number' || typeof size.width !== 'number') {
    throw new Error(
      'processor exposes neither image_processor.pad_size nor image_processor.size ' +
        'as {height, width}; cannot compute mask resample geometry',
    );
  }
  return { height: size.height, width: size.width };
}
```

**(d) The geometry bindings and the filter block.** Replace lines 123-126 — the size destructuring and `fullPixels` — with:

```ts
    // `original_sizes` and `reshaped_input_sizes` are [height, width].
    const [originalHeight, originalWidth] = originalSizes[0];
    const [reshapedHeight, reshapedWidth] = reshapedSizes[0];
    // `phase` is still 'encode' here, so a processor with no usable pad size
    // surfaces as SegmenterFailure('encode', ...) — which is where the
    // processor came from — rather than as an unattributed throw.
    const { width: padWidth, height: padHeight } = resolvePadSize(processor);
```

Then replace the whole `if (chosen.length > 0) { ... }` block (lines 215-242) with:

```ts
      if (chosen.length > 0) {
        // One call per surviving mask, straight from its low-res window to a
        // binary mask: no staging copy, no 5-D tensor, no ORT round-trip, no
        // padded-resolution intermediate and no full-resolution float buffer.
        for (let k = 0; k < chosen.length; k += 1) {
          const mask = resampleThresholdMask({
            logits: logits.subarray(chosen[k] * lowPixels, (chosen[k] + 1) * lowPixels),
            lowWidth,
            lowHeight,
            padWidth,
            padHeight,
            reshapedWidth,
            reshapedHeight,
            originalWidth,
            originalHeight,
            threshold: options.maskThreshold,
          });
          if (mask.area >= options.minMaskArea) candidates.push(mask);
        }
        recordSub('resample');
      }
```

Also update the `recordSub` comment just above it (lines 170-171) from "the three sum to the stage total" to "the two sum to the stage total". Nothing may be added between `recordSub('resample')` and the `timings.record('filter', ...)` call that follows.

Note the unchanged behaviour: `recordSub('resample')` fires only when `chosen.length > 0`, exactly as `upscale`/`threshold` did. A batch with no survivors records `select` alone, which then covers the whole stage.

- [ ] **Step 6: Run typecheck and the full suite**

Run: `npm run typecheck && npm test`
Expected: typecheck clean and every test green. If typecheck reports `image_processor` missing on `SamProcessor`, the getter is inherited from `Processor` in `@huggingface/transformers/types/processing_utils.d.ts` — check the import is `type SamProcessor` from the package root, not a local re-declaration.

- [ ] **Step 7: Verify the removals and the playground, by grep**

Run each of these and confirm the stated result. Each is anchored to the construct, not to a bare identifier, so the plan's own prose cannot satisfy it.

```bash
# AC8: none of the removed machinery survives in the worker.
grep -n "post_process_masks\|const selected\|fullPixels\|thresholdMask" src/segmenter/worker/segmenter.worker.ts
```
Expected: **no output** (exit 1).

```bash
# AC9: pad size comes from the processor, and 1024 is nowhere near it.
grep -n "pad_size\|1024" src/segmenter/worker/segmenter.worker.ts
```
Expected: exactly one line, the `imageProcessor?.pad_size ?? imageProcessor?.size` expression. No `1024`.

```bash
# AC10 / AC11: no stale substep name anywhere in shipped code or the playground,
# and the playground still renders from the constant rather than a literal list.
grep -rn "'upscale'\|\"upscale\"\|filterSubPhases\.upscale\|filterSubPhases\.threshold" src playground
grep -n "FILTER_SUBSTEP_ORDER" playground/SegmentView.tsx
```
Expected: the first grep produces **no output**; the second produces the import line and the `.map` at ~line 285. `playground/` is excluded from `tsconfig.json`, so this grep — not `npm run typecheck` — is what covers it.

```bash
# AC11: the renderer is untouched by this branch.
git diff --name-only main -- src/renderer
```
Expected: **no output**.

```bash
# AC11: thresholdMask and stabilityScore are still exported and still tested.
grep -n "^export function thresholdMask\|^export function stabilityScore" src/segmenter/core/mask-postprocess.ts
grep -c "thresholdMask\|stabilityScore" src/segmenter/core/mask-postprocess.test.ts
```
Expected: both `export function` lines present; the test-file count is greater than 0.

- [ ] **Step 8: Commit**

```bash
git add src/segmenter/core/types.ts src/segmenter/core/timing.test.ts src/segmenter/createSegmenter.test.ts src/segmenter/worker/segmenter.worker.ts
git commit -m "perf(segmenter): fuse the mask upsample and threshold into one pass"
```

---

## Notes for the PR description

- **Public API change:** `FILTER_SUBSTEP_ORDER` narrows from `['select', 'upscale', 'threshold']` to `['select', 'resample']`, and `FilterSubstep` narrows with it. Both are re-exported through `note-scanner/segmenter`. Deliberate: the two regions are now one fused loop, and keeping them apart would ship a permanently-zero row.
- **AC1's number:** quote the `[mask-resample AC1] ...` line from the test output. Do not paraphrase it and do not round it away.
- **AC2 is not a gate on this PR.** The `filter` before/after table at `pointsPerSide` 16 and 32 is measured by hand on real WebGPU hardware following the spec's "Measurement procedure" and posted to issue #8. No estimated figures anywhere.
