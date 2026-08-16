// @vitest-environment jsdom
//
// This is the first DOM-touching suite in the package: the other seven core
// suites are pure and run fine in the package's default `node` environment
// (see vitest.config.ts). Canvas2DRenderer.paint() creates a scratch
// `document.createElement('canvas')` for the dim layer, so this file alone
// needs a real `document`.
import { describe, it, expect, vi } from 'vitest';
import { createCanvas2DRenderer } from './canvas2d';
import type { Renderer, Scene } from './types';

function fakeCtx() {
  return {
    setTransform: vi.fn(),
    clearRect: vi.fn(),
    drawImage: vi.fn(),
    fillRect: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
    stroke: vi.fn(),
    fill: vi.fn(),
    arc: vi.fn(),
    globalCompositeOperation: 'source-over',
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 0,
    lineJoin: '',
    lineCap: '',
  };
}

function fakeCanvas() {
  const ctx = fakeCtx();
  return {
    canvas: { width: 0, height: 0, getContext: () => ctx } as unknown as HTMLCanvasElement,
    ctx,
  };
}

function scene(over: Partial<Scene> = {}): Scene {
  return {
    base: {} as CanvasImageSource,
    imageWidth: 4,
    imageHeight: 4,
    masks: new Map(),
    selectedIds: new Set(),
    hoveredId: null,
    draftPoints: [],
    devicePixelRatio: 1,
    ...over,
  };
}

describe('Renderer conformance: Canvas2DRenderer', () => {
  it('draw() before init() is a no-op, not a throw', () => {
    const r: Renderer = createCanvas2DRenderer();
    expect(() => r.draw(scene())).not.toThrow();
  });

  it('init() then draw() paints the base image', () => {
    const { canvas, ctx } = fakeCanvas();
    const r = createCanvas2DRenderer();
    r.init(canvas);
    r.draw(scene());
    expect(ctx.drawImage).toHaveBeenCalled();
  });

  it('resize() sets the backing store from dpr', () => {
    const { canvas } = fakeCanvas();
    const r = createCanvas2DRenderer();
    r.init(canvas);
    r.resize(scene({ imageWidth: 10, imageHeight: 20, devicePixelRatio: 2 }));
    expect(canvas.width).toBe(20);
    expect(canvas.height).toBe(40);
  });

  it('dispose() is idempotent and draw() after dispose does not throw', () => {
    const { canvas } = fakeCanvas();
    const r = createCanvas2DRenderer();
    r.init(canvas);
    r.dispose();
    expect(() => r.dispose()).not.toThrow();
    expect(() => r.draw(scene())).not.toThrow();
  });

  it('draws the draft polygon when draftPoints are present', () => {
    const { canvas, ctx } = fakeCanvas();
    const r = createCanvas2DRenderer();
    r.init(canvas);
    r.draw(scene({ draftPoints: [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }] }));
    expect(ctx.stroke).toHaveBeenCalled();
  });

  it('evict() with an unknown id is a no-op, not a throw', () => {
    const r = createCanvas2DRenderer();
    // Assert the method actually exists (not just that optional chaining
    // silently no-ops on an absent method) so this fails for the right
    // reason — a missing `evict` — before it is implemented.
    expect(typeof r.evict).toBe('function');
    expect(() => r.evict?.(['nope'])).not.toThrow();
  });

  it('evict() is callable with empty/duplicate ids and does not disturb a subsequent draw()', () => {
    // The strong version of this test — draw with `a` selected and loaded,
    // evict it, draw again with a different coverage for the same id, and
    // assert the second draw recomputed rather than reused stale indices —
    // is infeasible in this environment. Populating covIdxCache/edgeIdxCache
    // only happens inside `rebuildSelectedLayers`'s `for (const id of added)`
    // loop, which runs only when `draw()` is called with a *loaded, selected*
    // id; that function is async and, for any non-empty selection, always
    // proceeds to `countsToImageData` (needs `ImageData`) and then
    // `createImageBitmap` — neither implemented by jsdom (confirmed
    // experimentally: both are undefined in this environment, and reaching
    // either produces an *unhandled rejection* that fails the whole suite,
    // not just this test, since `draw()` fires the async work as
    // `void rebuildSelectedLayers(scene)`). Stubbing both globals was tried
    // and works mechanically, but even then the cached `Uint32Array` indices
    // never leave the renderer's closure and `fakeCtx()`'s `drawImage` mock
    // only records that a bitmap was drawn, not its content — so
    // recomputation-vs-reuse still wouldn't be observable through this public
    // surface, and the extra stubbing would only be adding incidental
    // complexity. The correctness guarantee itself — evicting an id removes
    // it from both index caches, leaving others untouched — is already
    // covered directly by `evictMaskCaches`'s own test in
    // core/mask-loading.test.ts (Task 1), which is the tested primitive
    // `evict()` below calls. Here we assert the weaker, still meaningful
    // properties the plan allows as a fallback: evict is present, tolerates
    // empty/duplicate/unknown ids without throwing, and does not disturb a
    // subsequent draw() (using an empty selection, which stays on the
    // synchronous/no-bitmap path, matching every other `draw()` call already
    // in this file).
    const { canvas, ctx } = fakeCanvas();
    const r = createCanvas2DRenderer();
    r.init(canvas);
    r.draw(scene());

    expect(typeof r.evict).toBe('function');
    expect(() => r.evict?.([])).not.toThrow();
    expect(() => r.evict?.(['a'])).not.toThrow();
    expect(() => r.evict?.(['a', 'a', 'unknown'])).not.toThrow();

    expect(() => r.draw(scene())).not.toThrow();
    expect(ctx.drawImage).toHaveBeenCalled();
  });
});
