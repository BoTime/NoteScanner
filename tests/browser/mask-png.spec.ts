import { expect, test, type Page } from '@playwright/test';
import { encodeMaskPng } from '../../src/segmenter/core/mask-encode';
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
