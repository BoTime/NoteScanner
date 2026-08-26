import type { BinaryMask } from './mask-postprocess';

export type { BinaryMask };

/**
 * IoU dedup for near-identical masks, ported from
 * `apps/api/src/segmentation/mask-dedup.ts`. That file is the settled
 * definition and stays authoritative; this is a browser-side copy, not a
 * second opinion. Read it before touching anything here.
 *
 * One deliberate difference: the API's median-area outlier filter is NOT
 * ported. It is tuned for photos of post-it notes on a wall, where every real
 * mask is roughly the same size; this prototype segments arbitrary photos,
 * where a legitimately huge mask (a table, a sky) sits beside legitimately
 * tiny ones. The spec calls that filter optional and off by default, and
 * omitting it entirely is off-by-default with no dead code.
 */

/**
 * Intersection-over-union of two masks at the same resolution. Returns 0 when
 * the union is empty (both masks have zero area).
 */
export function pairwiseIoU(a: BinaryMask, b: BinaryMask): number {
  const areaSum = a.area + b.area;
  if (areaSum === 0) return 0;

  // Contract: equal-length coverage arrays. Iterate the shorter length
  // defensively so a malformed pair can't read past the end of either array.
  const len = Math.min(a.coverage.length, b.coverage.length);
  let intersection = 0;
  for (let p = 0; p < len; p += 1) {
    if (a.coverage[p] && b.coverage[p]) intersection += 1;
  }

  const union = areaSum - intersection;
  return union === 0 ? 0 : intersection / union;
}

/**
 * Greedy "keep largest" non-maximum suppression.
 *
 * Candidates are considered largest-area-first, ties broken by lower original
 * index. A candidate is dropped only when its IoU against an already-kept mask
 * is STRICTLY GREATER than the threshold — IoU exactly equal to the threshold
 * is not a duplicate.
 *
 * Returns the kept original indices in ascending order.
 */
export function dedupeMasks(masks: readonly BinaryMask[], iouThreshold: number): number[] {
  const order = masks.map((_, index) => index);
  // Largest area first; stable on ties via lower original index.
  order.sort((i, j) => masks[j].area - masks[i].area || i - j);

  const kept: number[] = [];
  for (const candidate of order) {
    const isDuplicate = kept.some(
      (keptIndex) => pairwiseIoU(masks[candidate], masks[keptIndex]) > iouThreshold,
    );
    if (!isDuplicate) kept.push(candidate);
  }

  return kept.sort((a, b) => a - b);
}
