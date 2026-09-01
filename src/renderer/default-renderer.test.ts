// @vitest-environment jsdom
//
// SCOPE: which backend the factory picks, and nothing else. jsdom has no
// WebGL, so "webgl2 was picked" is established by the returned renderer asking
// its canvas for a 'webgl2' context and "canvas2d was picked" by it asking for
// '2d' — which is precisely the observable difference between the two, and is
// checked against a canvas that records what it was asked for. The claim that
// the fallback still PAINTS correctly in a real browser is AC2's other half
// and lives in tests/browser/webgl2-renderer.spec.ts.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { createDefaultRenderer } from './index';
import type { Scene } from './types';

function recordingCanvas(kinds: Record<string, unknown>) {
  const asked: string[] = [];
  const canvas = {
    width: 0,
    height: 0,
    getContext: (kind: string) => {
      asked.push(kind);
      return kinds[kind] ?? null;
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  } as unknown as HTMLCanvasElement;
  return { canvas, asked };
}

function scene(): Scene {
  return {
    base: {} as CanvasImageSource,
    imageWidth: 4,
    imageHeight: 4,
    masks: new Map(),
    selectedIds: new Set(),
    hoveredId: null,
    draftPoints: [],
    devicePixelRatio: 1,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('createDefaultRenderer', () => {
  it('falls back to canvas2d when getContext("webgl2") returns null', () => {
    vi.spyOn(document, 'createElement').mockImplementation(
      () => recordingCanvas({}).canvas as unknown as HTMLElement,
    );
    const { canvas, asked } = recordingCanvas({ '2d': { setTransform: vi.fn(), clearRect: vi.fn(), drawImage: vi.fn(), fillRect: vi.fn() } });
    const r = createDefaultRenderer();
    r.init(canvas);
    expect(asked).toContain('2d');
    expect(asked).not.toContain('webgl2');
  });

  it('falls back to canvas2d when getContext throws (jsdom without the canvas package)', () => {
    vi.spyOn(document, 'createElement').mockImplementation(() => {
      return {
        width: 0,
        height: 0,
        getContext: () => {
          throw new Error('Not implemented');
        },
      } as unknown as HTMLElement;
    });
    const { canvas, asked } = recordingCanvas({ '2d': { setTransform: vi.fn(), clearRect: vi.fn(), drawImage: vi.fn(), fillRect: vi.fn() } });
    const r = createDefaultRenderer();
    r.init(canvas);
    expect(asked).toEqual(['2d']);
  });

  it('draw() on the fallback renderer is safe before anything else happens', () => {
    const r = createDefaultRenderer();
    expect(() => r.draw(scene())).not.toThrow();
    expect(() => r.dispose()).not.toThrow();
  });
});
