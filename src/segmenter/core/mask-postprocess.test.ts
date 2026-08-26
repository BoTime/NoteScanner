import { describe, it, expect } from 'vitest';
import { stabilityScore, thresholdMask } from './mask-postprocess';

describe('thresholdMask', () => {
  it('covers pixels strictly above the threshold and counts their area', () => {
    const mask = thresholdMask(Float32Array.from([-1, 0, 0.5, 2]), 0);
    expect(Array.from(mask.coverage)).toEqual([0, 0, 1, 1]);
    expect(mask.area).toBe(2);
  });

  it('treats a logit exactly at the threshold as uncovered', () => {
    const mask = thresholdMask(Float32Array.from([0, 0, 0]), 0);
    expect(mask.area).toBe(0);
    expect(Array.from(mask.coverage)).toEqual([0, 0, 0]);
  });

  it('honours a non-zero threshold', () => {
    const mask = thresholdMask(Float32Array.from([1, 2, 3]), 2);
    expect(mask.area).toBe(1);
    expect(Array.from(mask.coverage)).toEqual([0, 0, 1]);
  });

  it('returns an empty mask for empty logits', () => {
    const mask = thresholdMask(new Float32Array(0), 0);
    expect(mask.area).toBe(0);
    expect(mask.coverage).toHaveLength(0);
  });
});

describe('stabilityScore', () => {
  it('scores 1 when raising and lowering the threshold changes nothing', () => {
    // Every logit is far outside the +/- 1 band, so both thresholds agree.
    expect(stabilityScore(Float32Array.from([10, 10, -10, -10]), 0, 1)).toBe(1);
  });

  it('scores the ratio of the tight mask to the loose mask', () => {
    // At +1: only 5 survives -> 1 pixel. At -1: 5, 0 and -0.5 survive -> 3.
    expect(stabilityScore(Float32Array.from([5, 0, -0.5, -10]), 0, 1)).toBeCloseTo(1 / 3);
  });

  it('scores 0 when even the loose threshold covers nothing', () => {
    expect(stabilityScore(Float32Array.from([-10, -10]), 0, 1)).toBe(0);
  });

  it('scores 0 for empty logits rather than dividing by zero', () => {
    expect(stabilityScore(new Float32Array(0), 0, 1)).toBe(0);
  });
});
