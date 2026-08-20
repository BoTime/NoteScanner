// @vitest-environment jsdom
//
// Second DOM-touching suite in the package (after
// renderer/renderer-conformance.test.ts) — a real DOM is needed to render
// SegmentViewer and inspect its computed inline styles. jsdom here has no
// `canvas` npm package installed, so HTMLCanvasElement.getContext('2d')
// throws; a fake `renderer` (the component's RendererFactory injection
// point) sidesteps that entirely. jsdom also never fires Image.onload on
// its own (no real network fetch), so the base-image loader is stubbed with
// a minimal fake Image class that resolves on the next microtask.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, waitFor, cleanup } from '@testing-library/react';
import { SegmentViewer } from './SegmentViewer';
import type { Renderer } from './renderer';

class FakeImage {
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  crossOrigin: string | null = null;
  private _src = '';
  get src() {
    return this._src;
  }
  set src(v: string) {
    this._src = v;
    // Mirrors a real image's async decode: onload fires after `src` is set,
    // never synchronously within the same tick.
    queueMicrotask(() => {
      this.onload?.();
    });
  }
}

function fakeRenderer(): Renderer {
  return {
    init: vi.fn(),
    draw: vi.fn(),
    resize: vi.fn(),
    dispose: vi.fn(),
  };
}

function renderViewer() {
  return render(
    <SegmentViewer
      imageUrl="https://example.com/image.png"
      imageWidth={4000}
      imageHeight={1000}
      segments={[]}
      initialSelectedIds={new Set()}
      onSelectionChange={() => {}}
      onCreateSegment={async () => {}}
      renderer={fakeRenderer}
    />,
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('SegmentViewer frame/canvas sizing', () => {
  it('does not set aspectRatio on the loading placeholder (.sv-status)', () => {
    vi.stubGlobal('Image', FakeImage);
    renderViewer();
    const status = document.querySelector('.sv-status') as HTMLElement;
    expect(status).not.toBeNull();
    expect(status.style.aspectRatio).toBe('');
    expect(status.style.width).toBe('100%');
    expect(status.style.maxWidth).toBe('100%');
    expect(status.style.maxHeight).toBe('calc(100dvh - 28rem)');
  });

  it('does not set aspectRatio on the loaded frame (.sv-frame), and the canvas carries the contain-based sizing', async () => {
    vi.stubGlobal('Image', FakeImage);
    renderViewer();

    await waitFor(() => {
      expect(document.querySelector('.sv-canvas')).not.toBeNull();
    });

    const frame = document.querySelector('.sv-frame') as HTMLElement;
    expect(frame.style.aspectRatio).toBe('');
    expect(frame.style.width).toBe('100%');
    expect(frame.style.maxWidth).toBe('100%');
    expect(frame.style.maxHeight).toBe('calc(100dvh - 28rem)');

    const canvas = document.querySelector('.sv-canvas') as HTMLCanvasElement;
    expect(canvas.style.maxWidth).toBe('100%');
    expect(canvas.style.width).toBe('auto');
    expect(canvas.style.height).toBe('auto');
    expect(canvas.style.aspectRatio).toBe('4000 / 1000');

    // Critical: NOT '100%'. `.sv-frame` has no definite height, so a percentage
    // max-height resolves to `none` and the frame's `overflow: hidden` would
    // clip a tall image instead of scaling it to fit. The canvas must carry the
    // same concrete cap the frame box uses. jsdom does no layout, so this
    // asserts the CSS value that structurally prevents the bug, not geometry.
    expect(canvas.style.maxHeight).not.toBe('100%');
    expect(canvas.style.maxHeight).toBe('calc(100dvh - 28rem)');
    expect(canvas.style.maxHeight).toBe(frame.style.maxHeight);
  });

  it('keeps the loading and loaded outer boxes sized identically (no layout jump)', async () => {
    vi.stubGlobal('Image', FakeImage);
    renderViewer();

    const status = document.querySelector('.sv-status') as HTMLElement;
    const loadingBox = {
      width: status.style.width,
      maxWidth: status.style.maxWidth,
      maxHeight: status.style.maxHeight,
    };

    await waitFor(() => {
      expect(document.querySelector('.sv-frame')).not.toBeNull();
    });
    const frame = document.querySelector('.sv-frame') as HTMLElement;
    const readyBox = {
      width: frame.style.width,
      maxWidth: frame.style.maxWidth,
      maxHeight: frame.style.maxHeight,
    };

    expect(readyBox).toEqual(loadingBox);
  });
});
