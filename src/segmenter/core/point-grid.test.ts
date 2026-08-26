import { describe, it, expect } from 'vitest';
import { batchPoints, buildPointGrid } from './point-grid';

describe('buildPointGrid', () => {
  it('places a single point at the centre for n = 1', () => {
    expect(buildPointGrid(1)).toEqual([[0.5, 0.5]]);
  });

  it('places n * n points at cell centres, x varying fastest', () => {
    expect(buildPointGrid(2)).toEqual([
      [0.25, 0.25],
      [0.75, 0.25],
      [0.25, 0.75],
      [0.75, 0.75],
    ]);
  });

  it('produces exactly n * n points, all strictly inside the unit square', () => {
    const grid = buildPointGrid(16);
    expect(grid).toHaveLength(256);
    for (const [x, y] of grid) {
      expect(x).toBeGreaterThan(0);
      expect(x).toBeLessThan(1);
      expect(y).toBeGreaterThan(0);
      expect(y).toBeLessThan(1);
    }
  });

  it('rejects a non-positive or non-integer density', () => {
    expect(() => buildPointGrid(0)).toThrow(/positive integer/);
    expect(() => buildPointGrid(-3)).toThrow(/positive integer/);
    expect(() => buildPointGrid(2.5)).toThrow(/positive integer/);
  });
});

describe('batchPoints', () => {
  it('splits into fixed-size chunks with a short final chunk', () => {
    expect(batchPoints([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]]);
  });

  it('returns one chunk when the batch size covers everything', () => {
    expect(batchPoints([1, 2, 3], 3)).toEqual([[1, 2, 3]]);
    expect(batchPoints([1, 2, 3], 99)).toEqual([[1, 2, 3]]);
  });

  it('returns no chunks for an empty input', () => {
    expect(batchPoints([], 4)).toEqual([]);
  });

  it('rejects a batch size below 1, which would loop forever', () => {
    expect(() => batchPoints([1], 0)).toThrow(/at least 1/);
  });
});
