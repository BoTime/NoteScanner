import { describe, it, expect } from 'vitest';
import {
  MIN_SCALE,
  MAX_SCALE,
  clampScale,
  clampTransform,
  zoomAt,
  screenToImage,
  type ViewTransform,
} from './view-transform';

describe('clampScale', () => {
  it('clamps below MIN_SCALE up to MIN_SCALE', () => {
    expect(clampScale(0.2)).toBe(MIN_SCALE);
  });
  it('clamps above MAX_SCALE down to MAX_SCALE', () => {
    expect(clampScale(99)).toBe(MAX_SCALE);
  });
  it('passes through an in-range value', () => {
    expect(clampScale(2.5)).toBe(2.5);
  });
});

describe('clampTransform', () => {
  it('forces offsets to 0 at scale 1', () => {
    const t: ViewTransform = { scale: 1, offsetX: 50, offsetY: -30 };
    expect(clampTransform(t, 200, 100)).toEqual({
      scale: 1,
      offsetX: 0,
      offsetY: 0,
    });
  });

  it('clamps scale into range and zeros offsets when clamped to 1', () => {
    const t: ViewTransform = { scale: 0.3, offsetX: 10, offsetY: 10 };
    expect(clampTransform(t, 200, 100)).toEqual({
      scale: 1,
      offsetX: 0,
      offsetY: 0,
    });
  });

  it('keeps offsets that already cover the frame', () => {
    // frame 200x100, scale 2 -> scaled canvas 400x200.
    // valid offsetX range: [200-400, 0] = [-200, 0]; offsetY: [-100, 0].
    const t: ViewTransform = { scale: 2, offsetX: -50, offsetY: -25 };
    expect(clampTransform(t, 200, 100)).toEqual({
      scale: 2,
      offsetX: -50,
      offsetY: -25,
    });
  });

  it('clamps a too-positive offset to 0 (no blank top/left gutter)', () => {
    const t: ViewTransform = { scale: 2, offsetX: 30, offsetY: 40 };
    const r = clampTransform(t, 200, 100);
    expect(r.offsetX).toBe(0);
    expect(r.offsetY).toBe(0);
  });

  it('clamps a too-negative offset to the lower bound (no blank bottom/right gutter)', () => {
    // lower bound = frameDim - scaledDim = 200 - 400 = -200 (x), -100 (y)
    const t: ViewTransform = { scale: 2, offsetX: -999, offsetY: -999 };
    const r = clampTransform(t, 200, 100);
    expect(r.offsetX).toBe(-200);
    expect(r.offsetY).toBe(-100);
  });
});

describe('zoomAt', () => {
  const frameW = 200;
  const frameH = 100;

  // Image point currently under a frame-relative anchor, given the transform.
  function imagePointUnderAnchor(
    t: ViewTransform,
    ax: number,
    ay: number,
  ): { x: number; y: number } {
    return { x: (ax - t.offsetX) / t.scale, y: (ay - t.offsetY) / t.scale };
  }

  it('keeps the anchored point fixed when zooming in from identity', () => {
    const t: ViewTransform = { scale: 1, offsetX: 0, offsetY: 0 };
    const ax = 100;
    const ay = 50;
    const before = imagePointUnderAnchor(t, ax, ay);
    const after = zoomAt(t, 2, ax, ay, frameW, frameH);
    const afterPoint = imagePointUnderAnchor(after, ax, ay);
    expect(after.scale).toBe(2);
    expect(afterPoint.x).toBeCloseTo(before.x, 5);
    expect(afterPoint.y).toBeCloseTo(before.y, 5);
  });

  it('does not exceed MAX_SCALE', () => {
    const t: ViewTransform = { scale: 4, offsetX: -100, offsetY: -50 };
    const after = zoomAt(t, 10, frameW / 2, frameH / 2, frameW, frameH);
    expect(after.scale).toBe(MAX_SCALE);
  });

  it('does not go below MIN_SCALE and re-centers there', () => {
    const t: ViewTransform = { scale: 2, offsetX: -50, offsetY: -25 };
    const after = zoomAt(t, 0.1, frameW / 2, frameH / 2, frameW, frameH);
    expect(after.scale).toBe(MIN_SCALE);
    expect(after.offsetX).toBe(0);
    expect(after.offsetY).toBe(0);
  });
});

describe('screenToImage', () => {
  // Frame is 200x100 CSS px at screen origin (left/top = 0); image is 400x200 px.
  const frameRect = { left: 0, top: 0, width: 200, height: 100 };
  const imageW = 400;
  const imageH = 200;

  it('maps frame px to image px at scale 1 (matches the old inline math)', () => {
    const t: ViewTransform = { scale: 1, offsetX: 0, offsetY: 0 };
    // Click at frame center -> image center.
    expect(screenToImage(100, 50, t, frameRect, imageW, imageH)).toEqual({
      x: 200,
      y: 100,
    });
  });

  it('accounts for left/top offset of the frame on screen', () => {
    const t: ViewTransform = { scale: 1, offsetX: 0, offsetY: 0 };
    const shifted = { left: 30, top: 10, width: 200, height: 100 };
    // Screen (130,60) -> frame (100,50) -> image center.
    expect(screenToImage(130, 60, t, shifted, imageW, imageH)).toEqual({
      x: 200,
      y: 100,
    });
  });

  it('maps correctly at scale 2 with a non-zero offset', () => {
    // scale 2, offset (-100, -50): the image point under frame px (100,50) is
    // ((100 - -100)/2, (50 - -50)/2) = (100, 50) in canvas CSS px, which is
    // (100/200, 50/100) of the image = image px (200, 100).
    const t: ViewTransform = { scale: 2, offsetX: -100, offsetY: -50 };
    const r = screenToImage(100, 50, t, frameRect, imageW, imageH);
    expect(r.x).toBeCloseTo(200, 5);
    expect(r.y).toBeCloseTo(100, 5);
  });
});
