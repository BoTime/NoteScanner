import { describe, it, expect } from 'vitest';
import { makeSyntheticSegments, summarize } from './fixtures';

describe('makeSyntheticSegments', () => {
  it('makes exactly N segments with unique ids', () => {
    const segs = makeSyntheticSegments(10, 64, 64);
    expect(segs).toHaveLength(10);
    expect(new Set(segs.map((s) => s.id)).size).toBe(10);
  });

  it('gives every segment a data: mask URL so nothing is fetched', () => {
    for (const s of makeSyntheticSegments(3, 32, 32)) {
      expect(s.maskUrl.startsWith('data:')).toBe(true);
    }
  });

  it('numbers segments from 1', () => {
    expect(makeSyntheticSegments(2, 32, 32).map((s) => s.index)).toEqual([1, 2]);
  });
});

describe('summarize', () => {
  it('reports the distribution, not just the mean', () => {
    const s = summarize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(s.p50).toBe(6);
    expect(s.p95).toBe(10);
    expect(s.max).toBe(10);
    expect(s.count).toBe(10);
  });

  it('handles an empty sample set without NaN', () => {
    expect(summarize([])).toEqual({ p50: 0, p95: 0, max: 0, count: 0 });
  });
});
