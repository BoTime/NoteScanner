/**
 * Mounts the real `SegmentViewer` in the page, so AC2's "the viewer still
 * paints correctly in that fallback" and AC6's "an explicitly passed renderer
 * prop still overrides the default" are answered by the component, not by a
 * unit test standing in for it.
 *
 * TEST-ONLY, same as harness.ts.
 */
import { createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { SegmentViewer } from '../../../src/SegmentViewer';
import type { Renderer, Scene } from '../../../src/renderer';
import { encodeMaskPng } from '../../../src/segmenter/core/mask-encode';

const W = 64;
const H = 48;

function baseImageUrl(): string {
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const ctx = c.getContext('2d')!;
  const grad = ctx.createLinearGradient(0, 0, W, H);
  grad.addColorStop(0, '#c8e6a0');
  grad.addColorStop(1, '#3050a0');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);
  return c.toDataURL('image/png');
}

function rectCoverage(x0: number, y0: number, x1: number, y1: number): Uint8Array {
  const cov = new Uint8Array(W * H);
  for (let y = y0; y < y1; y += 1) for (let x = x0; x < x1; x += 1) cov[y * W + x] = 1;
  return cov;
}

export interface MountResult {
  /** Every context kind requested from the moment the mount began. */
  contextKinds: string[];
  /** How many times the injected renderer prop was asked to paint. */
  propDrawCalls: number;
  /** RGBA of the viewer's own canvas, or null if it never appeared. */
  pixels: number[] | null;
  width: number;
  height: number;
}

let root: Root | null = null;

const raf = () => new Promise<void>((r) => requestAnimationFrame(() => r()));

/**
 * Wait for the viewer's frame to be final.
 *
 * canvas2d does not finish a frame synchronously: `rebuildSelectedLayers`
 * awaits `createImageBitmap` and calls `paint` again only once the bright and
 * outline bitmaps exist. `settle()` in harness.ts cannot be reused here —
 * React owns this canvas's paint, so there is no `repaint` callback to give
 * it — so wait on the same exact completion signal it uses rather than on a
 * guessed number of frames, which is precisely what read back a pre-bitmap dim
 * frame on firefox during Task 1.
 *
 * Measured: with this reduced to a single frame the AC2 spec still passed on
 * all three engines, because awaiting `ready` already spends a task or two. It
 * is kept because that is a timing coincidence, not a guarantee — the same
 * coincidence held for chromium and webkit in `settle()` and did not for
 * firefox — and the signal costs one accessor.
 */
async function settleViewer(): Promise<void> {
  await raf();
  for (let i = 0; i < 120 && window.__harness.pendingBitmaps() > 0; i += 1) await raf();
  // The `paint` that follows the last resolved bitmap runs in a microtask, so
  // it has already happened by the time this frame's callback fires; one more
  // frame lets its pixels land in the backing store.
  await raf();
}

export async function mountViewer(opts: {
  denyWebgl2?: boolean;
  useRendererProp?: boolean;
}): Promise<MountResult> {
  unmountViewer();
  window.__harness.setDenyWebgl2(Boolean(opts.denyWebgl2));
  const imageUrl = baseImageUrl();
  const maskUrl = await encodeMaskPng(rectCoverage(16, 12, 48, 36), W, H);
  window.__harness.resetContextLog();

  let propDrawCalls = 0;
  const propRenderer: Renderer = {
    init: () => {},
    draw: (_scene: Scene) => {
      propDrawCalls += 1;
    },
    resize: () => {},
    dispose: () => {},
  };

  const host = document.createElement('div');
  host.id = 'viewer-host';
  document.body.appendChild(host);
  root = createRoot(host);

  const ready = new Promise<void>((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error('SegmentViewer never reached status "ready"')),
      15000,
    );
    root!.render(
      createElement(SegmentViewer, {
        imageUrl,
        imageWidth: W,
        imageHeight: H,
        segments: [{ id: 'a', maskUrl }],
        initialSelectedIds: new Set(['a']),
        onSelectionChange: () => {},
        onCreateSegment: async () => {},
        onStatusChange: (status) => {
          if (status === 'ready') {
            clearTimeout(timer);
            resolve();
          }
          if (status === 'error') {
            clearTimeout(timer);
            reject(new Error('SegmentViewer reported status "error"'));
          }
        },
        ...(opts.useRendererProp ? { renderer: () => propRenderer } : {}),
      }),
    );
  });
  await ready;
  await settleViewer();

  const canvas = host.querySelector('canvas');
  let pixels: number[] | null = null;
  if (canvas) {
    const out = document.createElement('canvas');
    out.width = W;
    out.height = H;
    const ctx = out.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(canvas, 0, 0, W, H);
    pixels = Array.from(ctx.getImageData(0, 0, W, H).data);
  }
  return { contextKinds: window.__harness.contextLog(), propDrawCalls, pixels, width: W, height: H };
}

export function unmountViewer(): void {
  root?.unmount();
  root = null;
  document.getElementById('viewer-host')?.remove();
  window.__harness.setDenyWebgl2(false);
}

declare global {
  interface Window {
    __viewerHarness: { mountViewer: typeof mountViewer; unmountViewer: typeof unmountViewer };
  }
}

window.__viewerHarness = { mountViewer, unmountViewer };
