import { describe, it, expect } from 'vitest';
import { dedupeMasks, pairwiseIoU, type BinaryMask } from './nms';

function mask(bits: number[]): BinaryMask {
  const coverage = Uint8Array.from(bits);
  return { coverage, area: bits.reduce((sum, b) => sum + (b ? 1 : 0), 0) };
}

describe('pairwiseIoU', () => {
  it('is 1 for identical masks', () => {
    expect(pairwiseIoU(mask([1, 1, 0, 0]), mask([1, 1, 0, 0]))).toBe(1);
  });

  it('is 0 for disjoint masks', () => {
    expect(pairwiseIoU(mask([1, 1, 0, 0]), mask([0, 0, 1, 1]))).toBe(0);
  });

  it('is smaller / larger when one mask fully contains the other', () => {
    expect(pairwiseIoU(mask([1, 0, 0, 0]), mask([1, 1, 1, 1]))).toBe(0.25);
  });

  it('is 0 rather than NaN when both masks are empty', () => {
    expect(pairwiseIoU(mask([0, 0]), mask([0, 0]))).toBe(0);
  });
});

describe('dedupeMasks', () => {
  it('returns nothing for no candidates', () => {
    expect(dedupeMasks([], 0.7)).toEqual([]);
  });

  it('keeps the larger of two duplicates and drops the smaller', () => {
    const kept = dedupeMasks([mask([1, 0, 0, 0]), mask([1, 1, 1, 0])], 0.2);
    expect(kept).toEqual([1]);
  });

  it('keeps disjoint masks and returns their indices ascending', () => {
    const kept = dedupeMasks([mask([0, 0, 1, 1]), mask([1, 1, 0, 0])], 0.5);
    expect(kept).toEqual([0, 1]);
  });

  it('keeps a candidate whose IoU is EXACTLY the threshold', () => {
    // IoU of a 1-pixel mask inside a 4-pixel mask is exactly 0.25.
    const kept = dedupeMasks([mask([1, 1, 1, 1]), mask([1, 0, 0, 0])], 0.25);
    expect(kept).toEqual([0, 1]);
  });

  it('drops that same candidate once the threshold dips just below its IoU', () => {
    const kept = dedupeMasks([mask([1, 1, 1, 1]), mask([1, 0, 0, 0])], 0.24);
    expect(kept).toEqual([0]);
  });

  it('breaks an equal-area tie in favour of the lower original index', () => {
    const kept = dedupeMasks([mask([1, 1, 0, 0]), mask([1, 1, 0, 0])], 0.5);
    expect(kept).toEqual([0]);
  });

  it('does not let a zero-area mask suppress anything', () => {
    const kept = dedupeMasks([mask([0, 0, 0, 0]), mask([1, 1, 0, 0])], 0.5);
    expect(kept).toEqual([0, 1]);
  });
});
