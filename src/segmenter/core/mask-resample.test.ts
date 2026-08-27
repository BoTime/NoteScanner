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
