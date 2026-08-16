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
});
