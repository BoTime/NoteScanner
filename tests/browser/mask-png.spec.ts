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

/**
 * The SECOND scene, at a deliberately NON-INTEGER ratio.
 *
 * The scene above pins an exact 4x ratio, where a nearest-neighbour upscale is
 * lossless by construction — so its `mismatches === 0` is a shape check, not a
 * measurement of what this change costs. The SHIPPED ratio is not integral: a
 * 1024x649 photo encodes at 256x162, i.e. 4.000 across and 4.006 down. The
 * defect class AC2 exists to catch — a boundary landing a pixel off, a thin
 * structure quantising away, two near-equal masks swapping places in
 * `hitTestAll`'s smallest-first order — lives entirely on the fractional axis,
 * and an exact-4x fixture cannot reach any of it.
 *
 * So: encode at 64x40 and upscale to 256x161. Across, 256/64 = 4.000 exactly.
 * Down, 161/40 = 4.025 — fractional, and the same character as the shipped
 * 4.006. Deliberately NOT 256x162 -> 1024x649: every coverage array crosses the
 * `page.evaluate` boundary as JSON, once per mask per resolution, and this
 * scene is about the RATIO, which 25x more pixels would not sharpen.
 *
 * What still holds EXACTLY here: `nonBinary === 0` on both decodes.
 * Binary-ness is not a function of the ratio — it is `buildMaskData`'s
 * `alpha > 0 && red > 0` coverage predicate having something unambiguous to
 * read — so if it ever stops holding, the contract is broken whatever the
 * geometry.
 *
 * What does NOT hold here, by design: pixel-for-pixel equality with the
 * natively-encoded full-resolution mask. At 4.025 a horizontal boundary must
 * land on a whole output row that the ideal boundary sits between, and that
 * quantisation is the documented, accepted cost of encoding at the decoder's
 * own window. Each mask therefore asserts a BOUND derived from its geometry
 * (see `mismatchBound`) — never a number read back off a run.
 */
const FRAC_ENCODE_W = 64;
const FRAC_ENCODE_H = 40;
const FRAC_FULL_W = 256;
const FRAC_FULL_H = 161;

/**
 * Both rects are written out by hand rather than scaled by a helper:
 * `rectCoverage`'s single `scale` cannot express 4.000 across and 4.025 down,
 * and spelling the full-resolution rect out IS the point — it is an
 * independent statement of where each boundary belongs, not a restatement of
 * the upscale under test.
 *
 * `full` is the image of `encode` under x -> 4x and y -> round(y * 161 / 40):
 *
 *   spine      x 30..31 -> 120..124   y  0..40 ->   0.00 ->   0 .. 161.00 -> 161
 *   corner-tl  x  0..8  ->   0..32    y  0..6  ->   0.00 ->   0 ..  24.15 ->  24
 *   corner-br  x 54..64 -> 216..256   y 32..40 -> 128.80 -> 129 .. 161.00 -> 161
 *   blob       x 24..40 ->  96..160   y 14..26 ->  56.35 ->  56 .. 104.65 -> 105
 *
 * Full areas are 644 / 768 / 1,280 / 3,136 — all distinct, and no two of them
 * within 15% of each other, so `hitTestAll`'s smallest-first ordering is
 * decided by the masks and a one-row quantisation error (worth at most the
 * mask's width, i.e. 4 / 32 / 40 / 64 pixels) cannot reorder them.
 *
 * `spine` is the thin structure: one encode pixel wide, so its thinness lies on
 * the exact-4.000 axis where it can be probed without ambiguity, and it spans
 * every row so its two horizontal boundaries are the image edges.
 *
 * `mismatchBound` = (interior horizontal boundaries) * (full width). Derivation:
 * a nearest-neighbour upscale covers output row Y when the source row it floors
 * to is covered, which puts the first covered row at `ceil(ideal)` (or, for an
 * engine sampling pixel centres, `ceil(ideal - 0.5)`), where `full` puts it at
 * `round(ideal)`. Those differ by at most one row. One displaced row costs at
 * most the mask's full width in disagreeing pixels, and only a boundary that is
 * not an image edge can move — so `spine`, whose horizontal boundaries are both
 * image edges, has a hard bound of zero.
 */
const FRAC_SCENE: ReadonlyArray<{
  id: string;
  encode: readonly [number, number, number, number];
  full: readonly [number, number, number, number];
  mismatchBound: number;
}> = [
  // full area   644 — 4px wide, full height; the thin structure
  { id: 'spine', encode: [30, 0, 1, 40], full: [120, 0, 4, 161], mismatchBound: 0 },
  // full area   768 — touches the top and left edges
  { id: 'corner-tl', encode: [0, 0, 8, 6], full: [0, 0, 32, 24], mismatchBound: 32 },
  // full area 1,280 — touches the right and bottom edges
  { id: 'corner-br', encode: [54, 32, 10, 8], full: [216, 129, 40, 32], mismatchBound: 40 },
  // full area 3,136 — overlaps `spine`; both horizontal boundaries interior
  { id: 'blob', encode: [24, 14, 16, 12], full: [96, 56, 64, 49], mismatchBound: 128 },
];

/**
 * Probe points in IMAGE space, with the ids each must select — every list
 * worked out from the `full` rects above and written down before the first run.
 *
 * MARGIN RULE, and it is what keeps this test from being flaky: a horizontal
 * boundary can sit up to one encode pixel — 4.025 output rows — away from where
 * `full` puts it, so every probe stands at least 6 output pixels clear of every
 * boundary it is not deliberately testing, inside or out. The only boundaries a
 * probe is allowed to sit near are `spine`'s left and right edges (x = 120 and
 * x = 124), which lie on the exact-4.000 axis and therefore cannot move at all,
 * and the image edges, which cannot move either. A probe placed on a quantised
 * boundary would fail intermittently, and relaxing it afterwards would empty
 * the test of meaning.
 *
 * Coverage, mirroring AC2's wording ("near the image edges (all four) and on a
 * thin structure"):
 *
 *   (0,0) (255,160)      two image corners that land INSIDE a mask
 *   (255,0) (0,160)      the other two, which must select nothing
 *   (122,2) (122,158)    both ends of the thin structure
 *   (122,80)             spine over blob — ordering decided by area, 644 < 3136
 *   (110,80) (140,80)    either side of the spine, inside blob
 *   (140,45) (140,66)    11px outside and 10px inside blob's quantised top edge
 *   (60,100)             open ground; nearest boundary 28px away
 *
 * Per-probe derivation, against `full`:
 *   (0,0)     corner-tl x 0..31 y 0..23 covers it; nothing else reaches x < 32.
 *   (255,160) corner-br x 216..255 y 129..160 covers it; nothing else is there.
 *   (255,0)   corner-br starts at y 129; corner-tl ends at x 32. Nothing.
 *   (0,160)   corner-tl ends at y 24. Nothing.
 *   (122,2)   spine x 120..123 covers it; blob starts at y 56. Spine alone.
 *   (122,158) spine covers it; blob ends at y 104, corner-br starts at x 216.
 *   (122,80)  spine AND blob (x 96..159, y 56..104). 644 < 3,136 -> spine first.
 *   (140,80)  blob only — spine ends at x 123.
 *   (110,80)  blob only — spine starts at x 120.
 *   (140,45)  above blob (y 56) and right of corner-tl (x 32). Nothing.
 *   (140,66)  inside blob. Blob alone.
 *   (60,100)  right of corner-tl, left of blob, above corner-br. Nothing.
 */
const FRAC_PROBES: ReadonlyArray<{ point: { x: number; y: number }; expect: string[] }> = [
  { point: { x: 0, y: 0 }, expect: ['corner-tl'] },
  { point: { x: 255, y: 160 }, expect: ['corner-br'] },
  { point: { x: 255, y: 0 }, expect: [] },
  { point: { x: 0, y: 160 }, expect: [] },
  { point: { x: 122, y: 2 }, expect: ['spine'] },
  { point: { x: 122, y: 158 }, expect: ['spine'] },
  { point: { x: 122, y: 80 }, expect: ['spine', 'blob'] },
  { point: { x: 140, y: 80 }, expect: ['blob'] },
  { point: { x: 110, y: 80 }, expect: ['blob'] },
  { point: { x: 140, y: 45 }, expect: [] },
  { point: { x: 140, y: 66 }, expect: ['blob'] },
  { point: { x: 60, y: 100 }, expect: [] },
];

test('a non-integer encode ratio still puts the same segment under the click (AC2)', async ({
  page,
}) => {
  const reduced: SegmentMaskData[] = [];
  const full: SegmentMaskData[] = [];
  const area = (values: number[]) => values.reduce((n, v) => n + v, 0);

  for (const entry of FRAC_SCENE) {
    const small = rectCoverage(entry.encode, FRAC_ENCODE_W, FRAC_ENCODE_H, 1);
    const large = rectCoverage(entry.full, FRAC_FULL_W, FRAC_FULL_H, 1);

    const smallUrl = await encodeMaskPng(small.coverage, FRAC_ENCODE_W, FRAC_ENCODE_H);
    const largeUrl = await encodeMaskPng(large.coverage, FRAC_FULL_W, FRAC_FULL_H);

    const upscaled = await decodeUpscaledInPage(page, smallUrl, FRAC_FULL_W, FRAC_FULL_H, false);
    const native = await decodeUpscaledInPage(page, largeUrl, FRAC_FULL_W, FRAC_FULL_H, false);

    // AC9, and it is ratio-independent: nothing between covered and clear comes
    // back from either decode, however the rows quantise.
    expect(
      upscaled.nonBinary,
      `${entry.id}: ${upscaled.nonBinary} non-binary pixels in the upscaled read-back, first ${JSON.stringify(upscaled.firstNonBinary)}`,
    ).toBe(0);
    expect(
      native.nonBinary,
      `${entry.id}: ${native.nonBinary} non-binary pixels in the native read-back, first ${JSON.stringify(native.firstNonBinary)}`,
    ).toBe(0);

    // NOT zero here, unlike the exact-4x scene: at 4.025 the two genuinely
    // differ along every interior horizontal boundary. The bound is the
    // geometric worst case from the docblock above.
    const mismatches = upscaled.coverage.reduce(
      (n, value, i) => (value === native.coverage[i] ? n : n + 1),
      0,
    );
    expect(
      mismatches,
      `${entry.id}: ${mismatches} pixels differ after the 4.025x upscale, bound ${entry.mismatchBound}`,
    ).toBeLessThanOrEqual(entry.mismatchBound);

    reduced.push({
      id: entry.id,
      coverage: Uint8Array.from(upscaled.coverage),
      area: area(upscaled.coverage),
    });
    full.push({
      id: entry.id,
      coverage: Uint8Array.from(native.coverage),
      area: area(native.coverage),
    });
  }

  expect(reduced).toHaveLength(FRAC_SCENE.length);

  // AC2's actual claim: the same click selects the same segment, in the same
  // order, whichever resolution the PNG was written at.
  for (const probe of FRAC_PROBES) {
    const label = `(${probe.point.x},${probe.point.y})`;
    expect(hitTestAll(probe.point, FRAC_FULL_W, FRAC_FULL_H, full), `${label} full-res`).toEqual(
      probe.expect,
    );
    expect(hitTestAll(probe.point, FRAC_FULL_W, FRAC_FULL_H, reduced), `${label} reduced`).toEqual(
      probe.expect,
    );
  }
});
