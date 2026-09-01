import { expect, test, type Page } from '@playwright/test';
import { encodeMaskPng } from '../../src/segmenter/core/mask-encode';
import { hitTestAll, type SegmentMaskData } from '../../src/core/segment-viewer-logic';
import { MASK_CASES } from '../fixtures/mask-cases';

/**
 * `SegmentViewer.tsx`'s decoder, run verbatim inside the page: draw the mask
 * PNG onto a canvas and count a pixel covered when `alpha > 0 && red > 0`.
 * A data-URL image does not taint the canvas, so `getImageData` is readable,
 * and drawing at the image's natural size means no resampling.
 */
async function decodeInPage(
  page: Page,
  maskUrl: string,
  width: number,
  height: number,
): Promise<number[]> {
  return page.evaluate(
    ({ maskUrl, width, height }) =>
      new Promise<number[]>((resolve, reject) => {
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
          ctx.drawImage(img, 0, 0, width, height);
          const data = ctx.getImageData(0, 0, width, height).data;
          const coverage: number[] = [];
          for (let i = 0; i < data.length; i += 4) {
            coverage.push(data[i + 3] > 0 && data[i] > 0 ? 1 : 0);
          }
          resolve(coverage);
        };
        img.src = maskUrl;
      }),
    { maskUrl, width, height },
  );
}

test.beforeEach(async ({ page }) => {
  // An empty document is all the spec needs; the images are data URLs.
  await page.setContent('<!doctype html><meta charset="utf-8"><title>mask decode</title>');
});

for (const maskCase of MASK_CASES) {
  test(`decodes ${maskCase.name} back to the exact coverage`, async ({ page }) => {
    const maskUrl = await encodeMaskPng(maskCase.coverage, maskCase.width, maskCase.height);
    const decoded = await decodeInPage(page, maskUrl, maskCase.width, maskCase.height);
    expect(decoded).toEqual(Array.from(maskCase.coverage));
  });
}

test('reports the natural dimensions the encoder declared in IHDR', async ({ page }) => {
  // A width that is not a multiple of 8 is the case where a bad IHDR would
  // otherwise hide behind correct-looking pixels.
  const maskCase = MASK_CASES.find((entry) => entry.name === 'random 17x5')!;
  const maskUrl = await encodeMaskPng(maskCase.coverage, maskCase.width, maskCase.height);
  const size = await page.evaluate(
    (url) =>
      new Promise<{ width: number; height: number }>((resolve, reject) => {
        const img = new Image();
        img.onerror = () => reject(new Error('decode failed'));
        img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
        img.src = url;
      }),
    maskUrl,
  );
  expect(size).toEqual({ width: maskCase.width, height: maskCase.height });
});

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
