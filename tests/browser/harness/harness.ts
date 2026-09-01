/**
 * In-page API for the renderer Playwright specs. TEST-ONLY: it lives under
 * `tests/`, is never imported by `src/`, and never ships.
 */
import { createCanvas2DRenderer } from '../../../src/renderer/canvas2d';
import { createWebGL2Renderer } from '../../../src/renderer/webgl2';
import type { Renderer, Scene } from '../../../src/renderer/types';

export interface MaskSpec {
  id: string;
  /** [x, y, w, h] in IMAGE pixels. */
  rect: [number, number, number, number];
  /**
   * 1 (default) authors the mask at image resolution and leaves `width` /
   * `height` undefined, which is the only shape canvas2d can render. 2 authors
   * it at half size and sets `width` / `height`, exercising webgl2's sampler
   * upsample (the F4 mechanism). canvas2d ignores those fields by contract, so
   * a scale > 1 scene is NOT comparable across backends.
   */
  maskScale?: number;
}

export interface SceneSpec {
  width: number;
  height: number;
  masks: MaskSpec[];
  selected: string[];
  hovered?: string | null;
  draft?: { x: number; y: number }[];
}

export interface Frame {
  width: number;
  height: number;
  /** RGBA bytes, row-major from the top-left, length width * height * 4. */
  pixels: number[];
}

// --- test-only getContext instrumentation ---------------------------------
// Two things the specs need that production code must never provide:
//  1. `preserveDrawingBuffer`, so a WebGL frame survives long enough to be
//     copied into a 2D canvas and read back. Without it the back buffer's
//     contents are undefined once the current task yields.
//  2. a log of every context kind requested, which is how the fallback spec
//     proves the default factory probed 'webgl2', got null, then asked '2d'.
const contextLog: string[] = [];
let lastGl: WebGL2RenderingContext | null = null;
let denyWebgl2 = false;
const realGetContext = HTMLCanvasElement.prototype.getContext;
HTMLCanvasElement.prototype.getContext = function (
  this: HTMLCanvasElement,
  kind: string,
  opts?: unknown,
) {
  contextLog.push(kind);
  if (kind === 'webgl2') {
    if (denyWebgl2) return null;
    const ctx = realGetContext.call(this, kind, {
      ...(opts as Record<string, unknown>),
      preserveDrawingBuffer: true,
    });
    if (ctx) lastGl = ctx as WebGL2RenderingContext;
    return ctx;
  }
  return realGetContext.call(this, kind, opts as never);
} as typeof HTMLCanvasElement.prototype.getContext;

// --- test-only texture accounting (AC5) -----------------------------------
let liveTextures = 0;
let peakTextures = 0;
if (typeof WebGL2RenderingContext !== 'undefined') {
  const proto = WebGL2RenderingContext.prototype;
  const realCreate = proto.createTexture;
  const realDelete = proto.deleteTexture;
  proto.createTexture = function (this: WebGL2RenderingContext) {
    const tex = realCreate.call(this);
    if (tex) {
      liveTextures += 1;
      peakTextures = Math.max(peakTextures, liveTextures);
    }
    return tex;
  };
  proto.deleteTexture = function (this: WebGL2RenderingContext, tex: WebGLTexture | null) {
    if (tex) liveTextures -= 1;
    return realDelete.call(this, tex);
  };
}

// --- test-only in-flight async-work accounting ----------------------------
// canvas2d does not finish a frame synchronously: `rebuildSelectedLayers`
// awaits `createImageBitmap` and only then paints the bright/outline layers.
// Counting the decodes still in flight is an EXACT "is this renderer done?"
// signal, which is what `settle()` needs — see the comment there for the
// firefox failure that a frame comparison alone could not tell apart.
let pendingBitmaps = 0;
if (typeof createImageBitmap === 'function') {
  const realCreateImageBitmap = createImageBitmap as (...args: unknown[]) => Promise<ImageBitmap>;
  window.createImageBitmap = function (...args: unknown[]) {
    pendingBitmaps += 1;
    return realCreateImageBitmap(...args).finally(() => {
      pendingBitmaps -= 1;
    });
  } as unknown as typeof createImageBitmap;
}

function makeBase(width: number, height: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = width;
  c.height = height;
  const ctx = c.getContext('2d')!;
  // Deterministic, non-uniform, and never black — a uniform base would let a
  // renderer that paints nothing pass the "dim wash is darker" assertions.
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const checker = ((x >> 3) + (y >> 3)) & 1 ? 40 : 0;
      ctx.fillStyle = `rgb(${120 + checker}, ${200 - checker}, ${80 + ((x * 2) % 100)})`;
      ctx.fillRect(x, y, 1, 1);
    }
  }
  return c;
}

function buildMask(spec: MaskSpec, imageWidth: number, imageHeight: number) {
  const scale = spec.maskScale ?? 1;
  const w = Math.floor(imageWidth / scale);
  const h = Math.floor(imageHeight / scale);
  const coverage = new Uint8Array(w * h);
  const [rx, ry, rw, rh] = spec.rect;
  const x0 = Math.floor(rx / scale);
  const y0 = Math.floor(ry / scale);
  const x1 = Math.floor((rx + rw) / scale);
  const y1 = Math.floor((ry + rh) / scale);
  let area = 0;
  for (let y = Math.max(0, y0); y < Math.min(h, y1); y += 1) {
    for (let x = Math.max(0, x0); x < Math.min(w, x1); x += 1) {
      coverage[y * w + x] = 1;
      area += 1;
    }
  }
  return scale === 1
    ? { coverage, area }
    : { coverage, area, width: w, height: h };
}

let baseCache: { key: string; canvas: HTMLCanvasElement } | null = null;

function toScene(spec: SceneSpec): Scene {
  const key = `${spec.width}x${spec.height}`;
  if (!baseCache || baseCache.key !== key) {
    baseCache = { key, canvas: makeBase(spec.width, spec.height) };
  }
  const masks: Scene['masks'] = new Map();
  for (const m of spec.masks) masks.set(m.id, buildMask(m, spec.width, spec.height));
  return {
    base: baseCache.canvas,
    imageWidth: spec.width,
    imageHeight: spec.height,
    masks,
    selectedIds: new Set(spec.selected),
    hoveredId: spec.hovered ?? null,
    draftPoints: spec.draft ?? [],
    // Pinned to 1 on purpose. At dpr > 1 canvas2d upscales the base image with
    // the browser's 2D resampler while webgl2 upscales with a NEAREST sampler,
    // so the AC3 differential would be measuring two resamplers, not two
    // renderers.
    devicePixelRatio: 1,
  };
}

function readback(canvas: HTMLCanvasElement, width: number, height: number): number[] {
  const out = document.createElement('canvas');
  out.width = width;
  out.height = height;
  const ctx = out.getContext('2d', { willReadFrequently: true })!;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(canvas, 0, 0, width, height);
  return Array.from(ctx.getImageData(0, 0, width, height).data);
}

const raf = () => new Promise<void>((r) => requestAnimationFrame(() => r()));

/**
 * canvas2d rebuilds its layers through `createImageBitmap`, so its first
 * synchronous paint is not its final frame. Read until two consecutive frames
 * agree rather than sleeping for a guessed interval — a fixed sleep is exactly
 * the kind of "worked on my machine" wait that has bitten this repo before.
 *
 * Frame equality alone is NOT sufficient, and this is measured, not assumed:
 * on the first real run of this harness, chromium and webkit resolved
 * canvas2d's `createImageBitmap` inside the first animation frame while
 * firefox did not, so firefox saw two identical *pre-bitmap* dim frames and
 * settled on a frame with no bright window in it at all. Requiring
 * `pendingBitmaps === 0` closes that hole with the renderer's own completion
 * signal instead of a longer guess. webgl2 paints synchronously and never
 * raises the counter, so this costs it nothing.
 */
async function settle(canvas: HTMLCanvasElement, w: number, h: number, repaint: () => void) {
  let prev = readback(canvas, w, h);
  for (let i = 0; i < 60; i += 1) {
    await raf();
    repaint();
    const next = readback(canvas, w, h);
    if (pendingBitmaps === 0 && next.length === prev.length && next.every((v, k) => v === prev[k])) {
      return next;
    }
    prev = next;
  }
  throw new Error('renderer output never settled within 60 frames');
}

function makeRenderer(backend: 'canvas2d' | 'webgl2'): Renderer {
  return backend === 'webgl2' ? createWebGL2Renderer() : createCanvas2DRenderer();
}

async function paint(backend: 'canvas2d' | 'webgl2', spec: SceneSpec): Promise<Frame> {
  const renderer = makeRenderer(backend);
  const canvas = document.createElement('canvas');
  document.getElementById('root')!.appendChild(canvas);
  try {
    const scene = toScene(spec);
    renderer.init(canvas);
    renderer.draw(scene);
    const pixels = await settle(canvas, spec.width, spec.height, () => renderer.draw(scene));
    return { width: spec.width, height: spec.height, pixels };
  } finally {
    // Released even when settle() throws — the failure path gets the same
    // cleanup as the happy path.
    renderer.dispose();
    canvas.remove();
  }
}

/** One renderer instance across a sequence of scenes, which is what makes the
 *  texture-count bound meaningful: a fresh renderer per frame would reset the
 *  accounting and the test would pass no matter how badly the real one leaked. */
async function paintSequence(
  backend: 'canvas2d' | 'webgl2',
  specs: SceneSpec[],
): Promise<{ frames: Frame[]; peak: number; live: number }> {
  const renderer = makeRenderer(backend);
  const canvas = document.createElement('canvas');
  document.getElementById('root')!.appendChild(canvas);
  const frames: Frame[] = [];
  try {
    renderer.init(canvas);
    peakTextures = liveTextures;
    for (const spec of specs) {
      const scene = toScene(spec);
      renderer.draw(scene);
      frames.push({
        width: spec.width,
        height: spec.height,
        pixels: await settle(canvas, spec.width, spec.height, () => renderer.draw(scene)),
      });
    }
    return { frames, peak: peakTextures, live: liveTextures };
  } finally {
    renderer.dispose();
    canvas.remove();
  }
}

/** Resolve on the next `type` event, or reject with a named error rather than
 *  hanging until Playwright's 30s test timeout reports only "page.evaluate". */
function eventOrTimeout(target: EventTarget, type: string, ms = 10000): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${type} never fired within ${ms}ms`)), ms);
    target.addEventListener(
      type,
      () => {
        clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
  });
}

export interface LossResult {
  supported: boolean;
  threwWhileLost: boolean;
  before: number[];
  after: number[];
  width: number;
  height: number;
}

/** AC7: draw() while the context is lost must paint nothing and throw nothing;
 *  after restore the same scene must come back. */
async function loseAndRestore(spec: SceneSpec): Promise<LossResult> {
  const renderer = createWebGL2Renderer();
  const canvas = document.createElement('canvas');
  document.getElementById('root')!.appendChild(canvas);
  try {
    const scene = toScene(spec);
    renderer.init(canvas);
    renderer.draw(scene);
    const before = await settle(canvas, spec.width, spec.height, () => renderer.draw(scene));
    const ext = lastGl?.getExtension('WEBGL_lose_context') ?? null;
    if (!ext) {
      return {
        supported: false,
        threwWhileLost: false,
        before,
        after: before,
        width: spec.width,
        height: spec.height,
      };
    }
    const lost = eventOrTimeout(canvas, 'webglcontextlost');
    ext.loseContext();
    await lost;
    let threwWhileLost = false;
    try {
      renderer.draw(scene);
    } catch {
      threwWhileLost = true;
    }
    // Yield a whole TASK before asking for the restore. `await lost` resumes in
    // a microtask, and a microtask checkpoint runs as soon as the listener that
    // resolved it returns — i.e. still inside the browser's dispatch of
    // `webglcontextlost`. chromium and webkit only mark restoration allowed
    // once that dispatch has finished and observed `preventDefault()`, so
    // calling `restoreContext()` from the microtask is too early: measured on
    // the first run of this spec, webkit logged
    // `INVALID_OPERATION: restoreContext: context restoration not allowed`
    // and `webglcontextrestored` never fired on either engine, while firefox
    // happened to allow it. A `setTimeout(0)` lets the dispatch complete.
    await new Promise<void>((r) => setTimeout(r, 0));
    // Registered here, not alongside `lost`: `webglcontextrestored` cannot fire
    // before the call below, and a listener armed earlier would leave a live
    // rejection timer behind whenever the `lost` wait is the one that fails.
    const restored = eventOrTimeout(canvas, 'webglcontextrestored');
    ext.restoreContext();
    await restored;
    renderer.draw(scene);
    const after = await settle(canvas, spec.width, spec.height, () => renderer.draw(scene));
    return { supported: true, threwWhileLost, before, after, width: spec.width, height: spec.height };
  } finally {
    renderer.dispose();
    canvas.remove();
  }
}

export interface HarnessApi {
  webgl2Available(): boolean;
  paint(backend: 'canvas2d' | 'webgl2', spec: SceneSpec): Promise<Frame>;
  paintSequence(
    backend: 'canvas2d' | 'webgl2',
    specs: SceneSpec[],
  ): Promise<{ frames: Frame[]; peak: number; live: number }>;
  loseAndRestore(spec: SceneSpec): Promise<LossResult>;
  /**
   * How many `createImageBitmap` calls are still in flight — the same exact
   * completion signal `settle()` uses, exposed because `viewer-harness.tsx`
   * cannot use `settle()`: React owns the viewer's paint, so there is no
   * `repaint` callback to hand it. Waiting a fixed number of frames instead
   * is the firefox trap documented on `settle()`.
   */
  pendingBitmaps(): number;
  contextLog(): string[];
  resetContextLog(): void;
  textureStats(): { live: number; peak: number };
  resetTextureStats(): void;
  setDenyWebgl2(deny: boolean): void;
  lastContext(): WebGL2RenderingContext | null;
}

const api: HarnessApi = {
  webgl2Available() {
    try {
      const c = document.createElement('canvas');
      return Boolean(c.getContext('webgl2'));
    } catch {
      return false;
    }
  },
  paint,
  paintSequence,
  loseAndRestore,
  pendingBitmaps: () => pendingBitmaps,
  contextLog: () => contextLog.slice(),
  resetContextLog: () => {
    contextLog.length = 0;
  },
  textureStats: () => ({ live: liveTextures, peak: peakTextures }),
  resetTextureStats: () => {
    peakTextures = liveTextures;
  },
  setDenyWebgl2: (deny) => {
    denyWebgl2 = deny;
  },
  lastContext: () => lastGl,
};

declare global {
  interface Window {
    __harness: HarnessApi;
  }
}

window.__harness = api;
