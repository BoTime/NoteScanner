import { describe, it, expect } from 'vitest';
import { nextFocusIndex } from './popup-keys';

describe('nextFocusIndex', () => {
  it('ArrowDown advances', () => {
    expect(nextFocusIndex(0, 3, 'ArrowDown')).toBe(1);
  });

  it('ArrowDown wraps at the end', () => {
    expect(nextFocusIndex(2, 3, 'ArrowDown')).toBe(0);
  });

  it('ArrowUp wraps at the start', () => {
    expect(nextFocusIndex(0, 3, 'ArrowUp')).toBe(2);
  });

  it('Home and End jump to the edges', () => {
    expect(nextFocusIndex(1, 3, 'Home')).toBe(0);
    expect(nextFocusIndex(1, 3, 'End')).toBe(2);
  });

  it('returns the current index for unrelated keys', () => {
    expect(nextFocusIndex(1, 3, 'a')).toBe(1);
  });

  it('is a no-op on an empty menu', () => {
    expect(nextFocusIndex(0, 0, 'ArrowDown')).toBe(0);
  });
});
