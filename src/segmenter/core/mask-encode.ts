/**
 * Binary mask -> PNG data URL, because `ViewerSegment.maskUrl` is a URL and a
 * data URL is the only kind this prototype can produce without a server.
 *
 * Written to work on either side of the worker boundary: it prefers
 * `OffscreenCanvas` (available in a worker AND on the main thread in every
 * browser that has WebGPU) and falls back to `document.createElement` only
 * where `OffscreenCanvas` is missing. That is why the function is async even
 * though the DOM-canvas path is synchronous.
 */

/**
 * Covered pixels become opaque white; everything else stays fully transparent.
 *
 * `SegmentViewer.tsx`'s mask decoder counts a pixel as covered when
 * `alpha > 0 && red > 0`. Opaque white on transparent black satisfies both
 * halves of that predicate unambiguously, and a mostly-transparent image is
 * what makes these PNGs small enough to carry as data URLs.
 */
export function maskToRgba(
  coverage: Uint8Array,
  width: number,
  height: number,
): Uint8ClampedArray {
  const pixels = width * height;
  if (coverage.length !== pixels) {
    throw new Error(
      `coverage length ${coverage.length} does not match ${width}x${height} (${pixels})`,
    );
  }

  const rgba = new Uint8ClampedArray(pixels * 4);
  for (let p = 0; p < pixels; p += 1) {
    if (!coverage[p]) continue;
    const offset = p * 4;
    rgba[offset] = 255;
    rgba[offset + 1] = 255;
    rgba[offset + 2] = 255;
    rgba[offset + 3] = 255;
  }
  return rgba;
}

/**
 * `btoa` takes a string, and `String.fromCharCode(...bytes)` blows the
 * argument limit somewhere north of 100k arguments — a full-resolution mask
 * PNG is comfortably past that. Chunking is the fix, and the chunk seam is
 * exactly where this would silently corrupt data, so it has its own test.
 */
const BASE64_CHUNK = 0x8000;

export function bytesToDataUrl(bytes: Uint8Array, mime: string): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += BASE64_CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + BASE64_CHUNK));
  }
  return `data:${mime};base64,${btoa(binary)}`;
}

export async function encodeMaskPng(
  coverage: Uint8Array,
  width: number,
  height: number,
): Promise<string> {
  const image = new ImageData(
    maskToRgba(coverage, width, height) as Uint8ClampedArray<ArrayBuffer>,
    width,
    height,
  );

  if (typeof OffscreenCanvas !== 'undefined') {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D | null;
    if (!ctx) throw new Error('OffscreenCanvas 2d context unavailable');
    ctx.putImageData(image, 0, 0);
    const blob = await canvas.convertToBlob({ type: 'image/png' });
    return bytesToDataUrl(new Uint8Array(await blob.arrayBuffer()), 'image/png');
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('canvas 2d context unavailable');
  ctx.putImageData(image, 0, 0);
  return canvas.toDataURL('image/png');
}
