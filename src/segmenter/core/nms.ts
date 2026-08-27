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
 * Population count of one 32-bit word — the classic SWAR bit-twiddle.
 *
 * Exported so it can be unit-tested directly, which is worth doing: the input
 * here is the result of `a & b`, and `&` yields a SIGNED int32, so a word with
 * the high bit set arrives as a negative number. `value >>> 0` normalises that
 * back to the unsigned bit pattern before the first (arithmetic) subtraction —
 * without it, `popcount(0x80000000 | 0)` is wrong and nothing downstream says so.
 */
export function popcount(value: number): number {
  let v = value >>> 0;
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return (Math.imul(v, 0x01010101) >>> 24) & 0x3f;
}

/**
 * A candidate with its coverage packed 32 pixels to a word and its bounding
 * box precomputed. Module-private: this is an implementation detail of
 * `dedupeMasks`, built and discarded inside one call.
 */
interface PreparedMask {
  words: Uint32Array;
  area: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  /** No covered pixel at all, so the bbox is meaningless and nothing intersects it. */
  empty: boolean;
}

function prepareMask(mask: BinaryMask, width: number): PreparedMask {
  const coverage = mask.coverage;
  const pixels = coverage.length;
  const words = new Uint32Array((pixels + 31) >>> 5);
  let x0 = 0;
  let y0 = 0;
  let x1 = -1;
  let y1 = -1;
  let seen = false;

  for (let p = 0; p < pixels; p += 1) {
    if (!coverage[p]) continue;
    words[p >>> 5] |= 1 << (p & 31);
    const x = p % width;
    const y = (p / width) | 0;
    if (!seen) {
      x0 = x;
      x1 = x;
      y0 = y;
      y1 = y;
      seen = true;
      continue;
    }
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    // p ascends, so y never decreases: y0 is fixed by the first covered pixel.
    if (y > y1) y1 = y;
  }

  return { words, area: mask.area, x0, y0, x1, y1, empty: !seen };
}

/**
 * IoU of two prepared candidates. Exactly equal to `pairwiseIoU` on the masks
 * they were built from, for every input — the two rejections below only skip
 * work that provably contributes nothing to the intersection.
 */
function packedIoU(a: PreparedMask, b: PreparedMask, width: number): number {
  const areaSum = a.area + b.area;
  // Mirrors pairwiseIoU's first line, and keeps the union below from going
  // negative when both areas are 0 but coverage bits are set.
  if (areaSum === 0) return 0;
  // No covered pixel on one side: intersection is 0, so IoU is 0/areaSum.
  if (a.empty || b.empty) return 0;

  // N2 — disjoint on either axis means IoU 0 by definition. Not one pixel read.
  if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return 0;

  // N3 — only the words spanned by the SHARED row band can hold a bit set in
  // both masks. That range is a superset of the possible overlap, so
  // restricting to it is exact rather than approximate.
  const rowStart = a.y0 > b.y0 ? a.y0 : b.y0;
  const rowEnd = a.y1 < b.y1 ? a.y1 : b.y1;
  // Contract: equal-length coverage arrays. Clamp to the shorter defensively,
  // exactly as pairwiseIoU's Math.min does, so a malformed pair still cannot
  // read past the end of either array.
  const wordCount = Math.min(a.words.length, b.words.length);
  const first = (rowStart * width) >>> 5;
  let last = ((rowEnd + 1) * width - 1) >>> 5;
  if (last > wordCount - 1) last = wordCount - 1;

  let intersection = 0;
  for (let w = first; w <= last; w += 1) {
    intersection += popcount(a.words[w] & b.words[w]);
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
 * `width` is the pixel width of the coverage arrays and is REQUIRED: it is what
 * turns a flat index into an (x, y), which is what makes the bbox prefilter
 * 2-D. There is deliberately no optional-width fallback — a quiet slow path is
 * easy to end up in by accident, and a required parameter makes the compiler
 * say so instead.
 *
 * Returns the kept original indices in ascending order. Identical, for every
 * input, to `dedupeMasksReference` — see the differential test.
 */
export function dedupeMasks(
  masks: readonly BinaryMask[],
  iouThreshold: number,
  width: number,
): number[] {
  const order = masks.map((_, index) => index);
  // Largest area first; stable on ties via lower original index.
  order.sort((i, j) => masks[j].area - masks[i].area || i - j);

  // One extra linear pass over every candidate, to remove a quadratic number
  // of byte-pair reads from the loop below. Function-local, collected on return.
  const prepared = masks.map((mask) => prepareMask(mask, width));

  const kept: number[] = [];
  for (const candidate of order) {
    const isDuplicate = kept.some(
      (keptIndex) => packedIoU(prepared[candidate], prepared[keptIndex], width) > iouThreshold,
    );
    if (!isDuplicate) kept.push(candidate);
  }

  return kept.sort((a, b) => a - b);
}

/**
 * The byte-wise `dedupeMasks` this module shipped before the bbox prefilter and
 * bit packing, preserved verbatim as the baseline that "identical to baseline"
 * is measured against — by the differential test, and by the playground's
 * opt-in A/B panel.
 *
 * Reachable from `./nms` for the worker and the tests, and deliberately NOT
 * re-exported by `./index`: the package does not own a slow second NMS as
 * public API.
 */
export function dedupeMasksReference(
  masks: readonly BinaryMask[],
  iouThreshold: number,
): number[] {
  const order = masks.map((_, index) => index);
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
