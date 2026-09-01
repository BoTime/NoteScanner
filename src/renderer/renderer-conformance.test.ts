// @vitest-environment jsdom
//
// This is the first DOM-touching suite in the package: the other core suites
// are pure and run fine in the package's default `node` environment (see
// vitest.config.ts). Canvas2DRenderer.paint() creates a scratch
// `document.createElement('canvas')` for the dim layer, so this file alone
// needs a real `document`.
//
// SCOPE, so nobody mistakes this for GPU evidence: jsdom has no WebGL. The
// WebGL2 case below runs against a MOCKED `webgl2` context. It proves the
// lifecycle contract — draw before init, idempotent dispose, draw after
// dispose, evict tolerance, dpr-sized backing store — and proves nothing
// whatsoever about pixels, blending, or texture formats. Every claim that is
// genuinely about the GPU lives in `tests/browser/webgl2-renderer.spec.ts`,
// which runs real chromium/webkit/firefox.
import { describe, it, expect, vi } from 'vitest';
import { createCanvas2DRenderer } from './canvas2d';
import { createWebGL2Renderer } from './webgl2';
import type { Renderer, RendererFactory, Scene } from './types';

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
    canvas: {
      width: 0,
      height: 0,
      getContext: () => ctx,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    } as unknown as HTMLCanvasElement,
    ctx: ctx as unknown as Record<string, ReturnType<typeof vi.fn>>,
  };
}

/**
 * A stable pseudo-value for a GL enum name. Every `gl.SOME_ENUM` read must
 * return the same number every time, and `checkFramebufferStatus()` must
 * return exactly `gl.FRAMEBUFFER_COMPLETE` — otherwise the renderer's
 * completeness check fails against its own mock and the suite passes for the
 * wrong reason.
 */
function glEnum(name: string): number {
  let h = 0x1000;
  for (let i = 0; i < name.length; i += 1) h = (h * 31 + name.charCodeAt(i)) | 0;
  return h >>> 0;
}

/**
 * A Proxy standing in for WebGL2RenderingContext: ALL-CAPS reads are enums,
 * everything else is a memoized `vi.fn()` so `expect(gl.drawArrays)` works.
 * A hand-written mock of ~50 GL entry points would rot on the first shader
 * edit; this cannot.
 */
function fakeGl() {
  const fns = new Map<string, ReturnType<typeof vi.fn>>();
  const returns: Record<string, () => unknown> = {
    createShader: () => ({}),
    createProgram: () => ({}),
    createTexture: () => ({}),
    createFramebuffer: () => ({}),
    createBuffer: () => ({}),
    createVertexArray: () => ({}),
    getUniformLocation: () => ({}),
    getShaderParameter: () => true,
    getProgramParameter: () => true,
    getShaderInfoLog: () => '',
    getProgramInfoLog: () => '',
    checkFramebufferStatus: () => glEnum('FRAMEBUFFER_COMPLETE'),
    getExtension: () => null,
  };
  return new Proxy({} as Record<string, unknown>, {
    get(_target, prop) {
      if (typeof prop !== 'string') return undefined;
      if (/^[A-Z][A-Z0-9_]*$/.test(prop)) return glEnum(prop);
      let fn = fns.get(prop);
      if (!fn) {
        fn = vi.fn(returns[prop] ?? (() => undefined));
        fns.set(prop, fn);
      }
      return fn;
    },
  }) as unknown as Record<string, ReturnType<typeof vi.fn>>;
}

function fakeGlCanvas() {
  const gl = fakeGl();
  return {
    canvas: {
      width: 0,
      height: 0,
      getContext: (kind: string) => (kind === 'webgl2' ? gl : null),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    } as unknown as HTMLCanvasElement,
    ctx: gl,
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

type Harness = {
  factory: RendererFactory;
  makeCanvas: () => { canvas: HTMLCanvasElement; ctx: Record<string, ReturnType<typeof vi.fn>> };
  expectPainted: (ctx: Record<string, ReturnType<typeof vi.fn>>) => void;
  expectDraftPainted: (ctx: Record<string, ReturnType<typeof vi.fn>>) => void;
};

function describeRendererContract(name: string, h: Harness): void {
  describe(`Renderer conformance: ${name}`, () => {
    it('draw() before init() is a no-op, not a throw', () => {
      const r: Renderer = h.factory();
      expect(() => r.draw(scene())).not.toThrow();
    });

    it('init() then draw() paints the base image', () => {
      const { canvas, ctx } = h.makeCanvas();
      const r = h.factory();
      r.init(canvas);
      r.draw(scene());
      h.expectPainted(ctx);
    });

    it('resize() sets the backing store from dpr', () => {
      const { canvas } = h.makeCanvas();
      const r = h.factory();
      r.init(canvas);
      r.resize(scene({ imageWidth: 10, imageHeight: 20, devicePixelRatio: 2 }));
      expect(canvas.width).toBe(20);
      expect(canvas.height).toBe(40);
    });

    it('dispose() is idempotent and draw() after dispose does not throw', () => {
      const { canvas } = h.makeCanvas();
      const r = h.factory();
      r.init(canvas);
      r.dispose();
      expect(() => r.dispose()).not.toThrow();
      expect(() => r.draw(scene())).not.toThrow();
    });

    it('draws the draft polygon when draftPoints are present', () => {
      const { canvas, ctx } = h.makeCanvas();
      const r = h.factory();
      r.init(canvas);
      r.draw(
        scene({
          draftPoints: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 1, y: 1 },
          ],
        }),
      );
      h.expectDraftPainted(ctx);
    });

    it('evict() with an unknown id is a no-op, not a throw', () => {
      const r = h.factory();
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
      // is infeasible in this environment. For canvas2d, populating
      // covIdxCache/edgeIdxCache only happens inside `rebuildSelectedLayers`'s
      // `for (const id of added)` loop, which runs only when `draw()` is called
      // with a *loaded, selected* id; that function is async and, for any
      // non-empty selection, always proceeds to `countsToImageData` (needs
      // `ImageData`) and then `createImageBitmap` — neither implemented by
      // jsdom (confirmed experimentally: both are undefined in this
      // environment, and reaching either produces an *unhandled rejection*
      // that fails the whole suite, not just this test, since `draw()` fires
      // the async work as `void rebuildSelectedLayers(scene)`). Stubbing both
      // globals was tried and works mechanically, but even then the cached
      // `Uint32Array` indices never leave the renderer's closure and
      // `fakeCtx()`'s `drawImage` mock only records that a bitmap was drawn,
      // not its content — so recomputation-vs-reuse still wouldn't be
      // observable through this public surface, and the extra stubbing would
      // only be adding incidental complexity. The correctness guarantee
      // itself — evicting an id removes it from both index caches, leaving
      // others untouched — is already covered directly by `evictMaskCaches`'s
      // own test in core/mask-loading.test.ts, which is the tested primitive
      // canvas2d's `evict()` calls. For webgl2 there is nothing per-id to
      // evict at all (see the doc comment on its `evict`). Here we assert the
      // weaker, still meaningful properties: evict is present, tolerates
      // empty/duplicate/unknown ids without throwing, and does not disturb a
      // subsequent draw() (using an empty selection, which stays on the
      // synchronous/no-bitmap path, matching every other `draw()` call already
      // in this file).
      const { canvas, ctx } = h.makeCanvas();
      const r = h.factory();
      r.init(canvas);
      r.draw(scene());

      expect(typeof r.evict).toBe('function');
      expect(() => r.evict?.([])).not.toThrow();
      expect(() => r.evict?.(['a'])).not.toThrow();
      expect(() => r.evict?.(['a', 'a', 'unknown'])).not.toThrow();

      expect(() => r.draw(scene())).not.toThrow();
      h.expectPainted(ctx);
    });
  });
}

describeRendererContract('Canvas2DRenderer', {
  factory: createCanvas2DRenderer,
  makeCanvas: fakeCanvas,
  expectPainted: (ctx) => expect(ctx.drawImage).toHaveBeenCalled(),
  expectDraftPainted: (ctx) => expect(ctx.stroke).toHaveBeenCalled(),
});

describeRendererContract('WebGL2Renderer', {
  factory: createWebGL2Renderer,
  makeCanvas: fakeGlCanvas,
  expectPainted: (ctx) => expect(ctx.drawArrays).toHaveBeenCalled(),
  // The draft polygon is the only thing this renderer uploads through a
  // vertex buffer; every other pass is an attribute-less full-screen triangle.
  // So a bufferData call is exactly the signal that draft geometry was built.
  expectDraftPainted: (ctx) => expect(ctx.bufferData).toHaveBeenCalled(),
});
