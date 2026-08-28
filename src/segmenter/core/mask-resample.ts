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
