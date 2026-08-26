/**
 * Turning a mask decoder's raw logits into something worth keeping.
 *
 * Both functions take a `Float32Array` that may be a `subarray` view into a
 * much larger tensor buffer — the worker slices per-mask windows out of one
 * allocation rather than copying — so neither may assume it owns its input.
 */

export interface BinaryMask {
  /** `coverage[p] === 1` for covered pixels, else 0. */
  coverage: Uint8Array;
  /** Number of covered pixels. */
  area: number;
}

/**
 * Binarize logits at `threshold`. A logit exactly at the threshold is NOT
 * covered, matching transformers.js's own `post_process_masks`
 * (`data[i] > mask_threshold`) so a mask binarized here and one binarized
 * there agree.
 */
export function thresholdMask(logits: Float32Array, threshold: number): BinaryMask {
  const coverage = new Uint8Array(logits.length);
  let area = 0;
  for (let i = 0; i < logits.length; i += 1) {
    if (logits[i] > threshold) {
      coverage[i] = 1;
      area += 1;
    }
  }
  return { coverage, area };
}

/**
 * SAM's stability score: the IoU between the mask binarized at
 * `threshold + offset` and the mask binarized at `threshold - offset`.
 *
 * Because a higher threshold can only ever cover a subset of what a lower one
 * covers, the intersection IS the tight mask's area and the union IS the loose
 * mask's area — so the IoU collapses to one ratio and one pass, with no second
 * allocation. A mask whose boundary moves a lot when the threshold is nudged
 * is a mask the decoder was not confident about.
 *
 * Returns 0 when the loose mask is empty (nothing to be stable about).
 */
export function stabilityScore(
  logits: Float32Array,
  threshold: number,
  offset: number,
): number {
  const high = threshold + offset;
  const low = threshold - offset;
  let highCount = 0;
  let lowCount = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const value = logits[i];
    if (value > high) highCount += 1;
    if (value > low) lowCount += 1;
  }
  return lowCount === 0 ? 0 : highCount / lowCount;
}
