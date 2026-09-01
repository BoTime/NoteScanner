# WebGL2 Renderer (issue #9 — M5 + F4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a WebGL2 painting backend alongside `canvas2d` behind the existing `RendererFactory` extension point, selected by default when a real `webgl2` context can be obtained.

**Architecture:** `createWebGL2Renderer()` mirrors `canvas2d`'s incremental accumulation on the GPU: two `R8` framebuffer-attached count textures at image resolution (bright, outline), one reused `R8` scratch texture that each added/removed mask's coverage is uploaded into, and two `R8` ping-pong targets for the separable radius-3 outline dilation. Selection deltas come from `resolveAppliedDelta` in `src/core` — the same function `canvas2d` uses — so both backends agree on delta semantics by construction. One composite pass per frame draws base + dim wash + outlines; the draft polygon is CPU-triangulated and drawn on top. `createDefaultRenderer()` probes WebGL2 for real and hands back `canvas2d` when the probe fails.

**Tech Stack:** TypeScript, WebGL2 / GLSL ES 3.00, React 19, vitest (jsdom), Playwright (chromium/webkit/firefox), Vite dev server (programmatic, as in `scripts/sweep-decode.mjs`).

**Spec:** `docs/superpowers/specs/2026-09-01-webgl2-renderer-design.md`

## Global Constraints

- The segmenter worker is **not** touched. `src/segmenter/**` must have a zero-line diff at the end of this run (AC9). No `filter` or `mask-encode` timing improvement is claimed.
- No public package API is **removed** or **made required**. `Scene.masks` entries gain `width?` / `height?` — optional, so a third-party `RendererFactory` compiles unchanged (AC8). Adding new named exports is allowed and expected.
- Hit-testing is untouched. It stays on the full-resolution CPU `coverage` array in `src/core/segment-viewer-logic.ts` (AC4). No task in this plan edits `hitTest` / `hitTestAll`.
- `canvas2d` behaviour must not change at all. It ignores `width` / `height` entirely and keeps operating at full image resolution.
- Outline geometry constant: `OUTLINE_RADIUS = 3`, matching `src/renderer/canvas2d.ts:20`. Dim wash: `rgba(0, 0, 0, 0.55)`. Outline / draft colour: `#f97316` = rgb(249, 115, 22).
- `R8` UNORM count targets are incremented by exactly one ULP (`1/255`) per covering mask, so counts are exact up to **255** concurrently selected masks. That ceiling is stated in a comment at the point where the format is chosen.
- No new npm packages. Everything this plan uses (`vite`, `@vitejs/plugin-react`, `@playwright/test`, `react`, `react-dom`, `@types/node`, `jsdom`) is already in `devDependencies`.
- **Precision about M5.** What this run removes from the main thread is the renderer's per-rebuild `countsToImageData(...)` → `createImageBitmap(...)` round trip (`src/renderer/canvas2d.ts`, `rebuildSelectedLayers` / `rebuildHoverLayers`). The per-mask PNG → `ImageData` → `Uint8Array` coverage decode at `src/SegmentViewer.tsx:80` is **not** removed — it is what produces `Scene.masks[].coverage`, and it stays until issue #7 moves mask production into the worker. Do not write a commit message, comment or doc claiming otherwise.
- Every GPU claim goes to Playwright. The jsdom conformance suite runs the WebGL2 renderer against a **mocked** context; it proves the lifecycle contract (AC1) and nothing about pixels. Say so in the file.

## File Structure

| File | Responsibility | Task |
| --- | --- | --- |
| `src/renderer/types.ts` (modify) | `Scene.masks` value type gains optional `width?` / `height?` | 1 |
| `src/renderer/webgl2.ts` (create) | `createWebGL2Renderer()` + `probeWebGL2Support()`; all GL state, shaders, passes, lifecycle | 1 |
| `src/renderer/renderer-conformance.test.ts` (modify) | one contract suite, run twice — once per factory | 1 |
| `tsconfig.tests.json` (create) | type-checks `tests/**` and `playwright.config.ts`, which `tsconfig.json`'s `include: ["src/**"]` does not cover | 1 |
| `tests/browser/dev-server.ts` (create) | starts a programmatic Vite dev server on an OS-assigned port for the harness page | 1 |
| `tests/browser/harness/index.html` (create) | harness entry page | 1 |
| `tests/browser/harness/harness.ts` (create) | in-page API the specs drive: paint a scene through either backend, read pixels back, count textures | 1 |
| `tests/browser/webgl2-renderer.spec.ts` (create) | engine WebGL2 support report + first real paint (task 1); differential / texture-bound / context-loss (task 3) | 1, 3 |
| `src/renderer/index.ts` (modify) | `createDefaultRenderer()` + re-exports | 2 |
| `src/renderer/default-renderer.test.ts` (create) | probe-fallback unit tests | 2 |
| `src/SegmentViewer.tsx` (modify) | line 158: `renderer ?? createDefaultRenderer` | 2 |
| `src/SegmentViewer.test.tsx` (modify) | AC6: an explicit `renderer` prop still wins | 2 |
| `src/index.ts` (modify) | export `createWebGL2Renderer`, `createDefaultRenderer` | 2 |
| `src/types.ts` (modify) | the `renderer` prop doc comment says "Defaults to Canvas2D"; it no longer does | 2 |
| `README.md` (modify) | the "Renderers" section says `createCanvas2DRenderer` is the default; it no longer is | 2 |
| `tests/browser/harness/viewer-harness.tsx` (create) | mounts `SegmentViewer` in-page for AC2 / AC6 | 3 |
| `docs/measurements/2026-09-01-webgl2-vs-canvas2d.md` (create) | the AC3 disagreement counts, per engine, with their caveats in the same file | 3 |

---

## Task 1: The WebGL2 renderer, the contract suite, and the first real-browser paint

**Files:**
- Modify: `src/renderer/types.ts:17`
- Create: `src/renderer/webgl2.ts`
- Modify: `src/renderer/renderer-conformance.test.ts` (restructure — see Step 3)
- Create: `tsconfig.tests.json`
- Modify: `package.json` (`typecheck` script)
- Create: `tests/browser/dev-server.ts`
- Create: `tests/browser/harness/index.html`
- Create: `tests/browser/harness/harness.ts`
- Create: `tests/browser/webgl2-renderer.spec.ts`

**Interfaces:**
- Consumes: `resolveAppliedDelta(prev, next, isLoaded) => { added: string[]; removed: string[]; applied: Set<string> }` from `src/core`; the `Renderer` / `Scene` types in `src/renderer/types.ts`.
- Produces:
  - `createWebGL2Renderer(): Renderer` — exported from `src/renderer/webgl2.ts`.
  - `probeWebGL2Support(): boolean` — exported from `src/renderer/webgl2.ts`; Task 2's `createDefaultRenderer()` calls it.
  - `startHarnessServer(): Promise<{ url: string; close: () => Promise<void> }>` — exported from `tests/browser/dev-server.ts`; Task 3 reuses it.
  - `window.__harness` in the harness page, with the members listed in Step 6. Task 3 extends this object; it does not replace it.

**Why this is one task:** the optional `width` / `height` fields exist only so `webgl2.ts` can size its scratch texture — a field and its only consumer. The contract suite cannot be written without the factory it parameterizes over. And per `docs/autopilot/learnings.md` ("Code the plan dictates verbatim has never been run"), the browser harness lands *here*, with a real `npx playwright test` invocation, rather than waiting for the verification task — the GL code below has never executed and the three engines' WebGL2 support is unknown.

- [ ] **Step 1: Add the optional per-mask dimensions**

In `src/renderer/types.ts`, replace the `masks` field:

```ts
  /**
   * Decoded masks by segment id.
   *
   * `width` / `height` are the mask's OWN natural size and are optional. They
   * are the hook for issue #7, which will make the segmenter worker emit
   * 256x256 masks instead of image-sized ones; a renderer that samples the
   * coverage as a texture then needs no change to keep working. Nothing emits
   * them yet.
   *
   * `canvas2d` ignores them entirely and keeps operating at full image
   * resolution. `webgl2` uses them for its scratch texture's dimensions when
   * present and falls back to `imageWidth` / `imageHeight` when absent.
   */
  masks: Map<string, { coverage: Uint8Array; area: number; width?: number; height?: number }>;
```

- [ ] **Step 2: Give `tests/` a typecheck lane**

`tsconfig.json` has `"include": ["src/**/*.ts", "src/**/*.tsx"]`, so `npm run typecheck` does **not** currently look at `tests/browser/**` or `playwright.config.ts`. Everything this plan writes under `tests/` would otherwise ship unchecked.

Create `tsconfig.tests.json`:

```json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "types": ["node"]
  },
  "include": ["tests/**/*.ts", "tests/**/*.tsx", "playwright.config.ts"],
  "exclude": ["node_modules", "dist"]
}
```

In `package.json`, change the `typecheck` script to:

```json
    "typecheck": "tsc --noEmit && tsc --noEmit -p tsconfig.playground.json && tsc --noEmit -p tsconfig.tests.json",
```

Run: `npm run typecheck`
Expected: PASS. The pre-existing `tests/browser/mask-png.spec.ts` and `tests/fixtures/mask-cases.ts` now type-check for the first time; if either reports an error, fix it in this task (both are small and import only from `src/`).

- [ ] **Step 3: Restructure the conformance suite to run once per backend**

The existing file has seven tests. None are dropped. Five move verbatim into a shared `describeRendererContract` helper; two (`init() then draw() paints the base image`, `draws the draft polygon when draftPoints are present`) also move in but get their backend-specific assertion supplied as a callback, because "painted" means `ctx.drawImage` for canvas2d and `gl.drawArrays` for WebGL2:

| existing test | fate |
| --- | --- |
| `draw() before init() is a no-op, not a throw` | moved verbatim into the shared contract |
| `init() then draw() paints the base image` | moved; assertion parameterized as `expectPainted` |
| `resize() sets the backing store from dpr` | moved verbatim |
| `dispose() is idempotent and draw() after dispose does not throw` | moved verbatim |
| `draws the draft polygon when draftPoints are present` | moved; assertion parameterized as `expectDraftPainted` |
| `evict() with an unknown id is a no-op, not a throw` | moved verbatim |
| `evict() is callable with empty/duplicate ids and does not disturb a subsequent draw()` | moved; its long explanatory comment is preserved verbatim, and its final `expect(ctx.drawImage)` becomes `expectPainted(ctx)` |

Replace the whole of `src/renderer/renderer-conformance.test.ts` with:

```ts
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
```

- [ ] **Step 4: Run the suite and watch it fail for the right reason**

Run: `npx vitest run src/renderer/renderer-conformance.test.ts`
Expected: FAIL — `Failed to resolve import "./webgl2"`. Both the canvas2d block and the webgl2 block fail to collect, because the import is at file scope. If instead the canvas2d tests pass and only webgl2 fails, the import was written inside the wrong scope; fix that before continuing.

- [ ] **Step 5: Implement `src/renderer/webgl2.ts`**

Create the file with exactly this content:

```ts
/**
 * WebGL2 painting backend — a peer of `createCanvas2DRenderer`.
 *
 * It mirrors canvas2d's incremental accumulation on the GPU. Two R8
 * framebuffer-attached count textures at image resolution hold, per pixel, how
 * many selected masks cover it (`bright`) and how many selected masks' dilated
 * edges cover it (`outline`). A selection toggle uploads only the toggled
 * mask's coverage into one reused scratch texture and draws it into those
 * targets with additive (select) or reverse-subtractive (deselect) blending —
 * so a toggle costs one mask's worth of work, not the whole selection's, and
 * peak VRAM is a constant number of textures regardless of how many masks are
 * selected.
 *
 * The added/removed sets come from `resolveAppliedDelta`, the same function
 * canvas2d uses, so the two backends agree on delta semantics by construction
 * rather than by two implementations happening to match.
 *
 * WHAT THIS DOES AND DOES NOT SAVE. It removes the per-rebuild
 * `countsToImageData()` -> `createImageBitmap()` round trip canvas2d performs
 * on the main thread. It does NOT remove the per-mask PNG -> ImageData ->
 * Uint8Array coverage decode in `SegmentViewer.tsx`; that is what produces
 * `Scene.masks[].coverage` and it is unchanged by this file.
 *
 * The scratch texture is uploaded at the MASK's own size (`mask.width` /
 * `mask.height` when present, else the image size) and sampled at image
 * resolution, so the sampler performs the upsample. Nothing emits smaller
 * masks yet; this is the mechanism, not a realized saving.
 *
 * COORDINATE CONVENTION — the one thing that will bite an editor. Nothing is
 * uploaded flipped (`UNPACK_FLIP_Y_WEBGL` stays 0), so for every texture here
 * "texture row 0" is "image row 0". In an FBO pass that means
 * `int(gl_FragCoord.y)` IS the image row index, which is what lets the dash
 * stipple and the edge test use the same integer coordinates canvas2d's CPU
 * loops use. The single flip happens in the composite pass's vertex shader,
 * which maps v = 0 to the TOP of the viewport.
 */
import { resolveAppliedDelta } from '../core';
import type { Renderer, Scene } from './types';

/** canvas2d's DIM_COLOR is `rgba(0, 0, 0, 0.55)`. */
const DIM_ALPHA = 0.55;
/** canvas2d's OUTLINE_COLOR / DRAFT_COLOR, `#f97316` = rgb(249, 115, 22). */
const OUTLINE_R = 249 / 255;
const OUTLINE_G = 115 / 255;
const OUTLINE_B = 22 / 255;
/** Must stay equal to canvas2d's OUTLINE_RADIUS. */
const OUTLINE_RADIUS = 3;
/** Triangles per draft vertex dot. */
const DOT_SEGMENTS = 16;
/**
 * One count in an R8 UNORM target. Additive blending at 1/255 increments a
 * UNORM8 texel by exactly one ULP, so counts are EXACT up to 255 concurrently
 * selected masks. Past 255 a texel saturates and stays lit until the selection
 * drops back under the ceiling. Raising the ceiling means moving the count
 * targets to R16F, which needs EXT_color_buffer_float and is therefore not
 * core WebGL2 — see makeTarget().
 */
const COUNT_UNIT = 1 / 255;

const CONTEXT_ATTRS: WebGLContextAttributes = {
  alpha: false,
  antialias: false,
  depth: false,
  stencil: false,
};

type Gl = WebGL2RenderingContext;
interface Target {
  tex: WebGLTexture;
  fbo: WebGLFramebuffer;
}
interface MaskRef {
  coverage: Uint8Array;
  width: number;
  height: number;
}

/** Attribute-less full-screen triangle; see the coordinate-convention note. */
const VERT_QUAD = `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

/**
 * Coverage byte 1 becomes 1/255 in an R8 texture, NOT 1.0 — every coverage
 * test in this file compares against 0.5/255.
 */
const FRAG_MASK = `#version 300 es
precision highp float;
uniform sampler2D uMask;
uniform vec2 uImageSize;
uniform float uAmount;
out vec4 fragColor;
const float COV_EPS = 0.5 / 255.0;
void main() {
  float c = texture(uMask, gl_FragCoord.xy / uImageSize).r > COV_EPS ? 1.0 : 0.0;
  fragColor = vec4(c * uAmount, 0.0, 0.0, 1.0);
}
`;

/**
 * Same rule as buildEdgeCoverage: a pixel is an edge when it is covered and at
 * least one of its 4 orthogonal neighbours is uncovered OR outside the image,
 * so a mask touching the border still gets an outline along that border.
 */
const FRAG_EDGE = `#version 300 es
precision highp float;
uniform sampler2D uMask;
uniform vec2 uImageSize;
out vec4 fragColor;
const float COV_EPS = 0.5 / 255.0;
float cov(vec2 p) {
  return texture(uMask, p / uImageSize).r > COV_EPS ? 1.0 : 0.0;
}
void main() {
  vec2 p = gl_FragCoord.xy;
  if (cov(p) == 0.0) { fragColor = vec4(0.0); return; }
  float l = p.x >= 1.0 ? cov(p - vec2(1.0, 0.0)) : 0.0;
  float r = p.x <= uImageSize.x - 1.0 ? cov(p + vec2(1.0, 0.0)) : 0.0;
  float d = p.y >= 1.0 ? cov(p - vec2(0.0, 1.0)) : 0.0;
  float u = p.y <= uImageSize.y - 1.0 ? cov(p + vec2(0.0, 1.0)) : 0.0;
  fragColor = vec4((l * r * d * u) < 1.0 ? 1.0 : 0.0, 0.0, 0.0, 1.0);
}
`;

/**
 * Half of a radius-N Chebyshev (8-connected square) dilation. It is separable,
 * so two 7-tap passes replace one 49-tap window — RADIUS is injected from
 * OUTLINE_RADIUS so the shader cannot drift from canvas2d's constant. The dash
 * stipple and the count increment ride on the second (vertical) pass.
 */
const FRAG_DILATE = `#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform vec2 uImageSize;
uniform vec2 uStep;
uniform float uAmount;
uniform bool uDashed;
out vec4 fragColor;
const float COV_EPS = 0.5 / 255.0;
const int RADIUS = ${OUTLINE_RADIUS};
void main() {
  ivec2 pi = ivec2(gl_FragCoord.xy);
  if (uDashed && ((pi.x + pi.y) & 1) == 1) { fragColor = vec4(0.0); return; }
  vec2 p = gl_FragCoord.xy;
  float m = 0.0;
  for (int i = -RADIUS; i <= RADIUS; i++) {
    vec2 q = p + uStep * float(i);
    if (q.x < 0.0 || q.y < 0.0 || q.x > uImageSize.x || q.y > uImageSize.y) continue;
    m = max(m, texture(uSrc, q / uImageSize).r > COV_EPS ? 1.0 : 0.0);
  }
  fragColor = vec4(m * uAmount, 0.0, 0.0, 1.0);
}
`;

/**
 * One pass, no per-mask work. Reproduces canvas2d's paint(): base image, then
 * a dim layer filled everywhere and punched out where the ACTIVE bright count
 * (hover when hovering, else selection) is non-zero, then the selection
 * outline and the hover outline as opaque source-over blits.
 */
const FRAG_COMPOSITE = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uBase;
uniform sampler2D uBright;
uniform sampler2D uOutline;
uniform sampler2D uHoverOutline;
uniform float uDimAlpha;
uniform vec3 uOutlineColor;
out vec4 fragColor;
const float COV_EPS = 0.5 / 255.0;
void main() {
  vec3 col = texture(uBase, vUv).rgb;
  float bright = texture(uBright, vUv).r > COV_EPS ? 1.0 : 0.0;
  col = mix(col, vec3(0.0), uDimAlpha * (1.0 - bright));
  float outline = max(
    texture(uOutline, vUv).r > COV_EPS ? 1.0 : 0.0,
    texture(uHoverOutline, vUv).r > COV_EPS ? 1.0 : 0.0);
  col = mix(col, uOutlineColor, outline);
  fragColor = vec4(col, 1.0);
}
`;

const VERT_DRAFT = `#version 300 es
in vec2 aPos;
uniform vec2 uImageSize;
void main() {
  gl_Position = vec4(
    aPos.x / uImageSize.x * 2.0 - 1.0,
    1.0 - aPos.y / uImageSize.y * 2.0,
    0.0, 1.0);
}
`;

const FRAG_DRAFT = `#version 300 es
precision highp float;
uniform vec3 uColor;
out vec4 fragColor;
void main() { fragColor = vec4(uColor, 1.0); }
`;

/**
 * Triangulate the draft polygon in IMAGE space: one quad per segment plus a
 * disc per vertex. canvas2d strokes the same path with `lineJoin: 'round'` and
 * `lineCap: 'round'`; the vertex discs (radius lineWidth * 1.8, which is wider
 * than the stroke) are what stand in for those round joins and caps, and they
 * are drawn by canvas2d too. The path is closed at 3+ points, matching
 * canvas2d's `if (draftPoints.length >= 3) ctx.closePath()`.
 *
 * Exported for the unit test in this file's sibling suite? No — kept private.
 * The geometry is verified through pixels in tests/browser, not by shape.
 */
function buildDraftGeometry(
  points: readonly { x: number; y: number }[],
  imageWidth: number,
  imageHeight: number,
): Float32Array {
  const lineWidth = Math.max(2, Math.min(imageWidth, imageHeight) / 300);
  const half = lineWidth / 2;
  const dot = lineWidth * 1.8;
  const n = points.length;
  const segments = n >= 3 ? n : n - 1;
  const out: number[] = [];
  for (let i = 0; i < segments; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % n];
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    const len = Math.hypot(dx, dy);
    if (len === 0) continue;
    dx /= len;
    dy /= len;
    const nx = -dy * half;
    const ny = dx * half;
    out.push(a.x + nx, a.y + ny, b.x + nx, b.y + ny, b.x - nx, b.y - ny);
    out.push(a.x + nx, a.y + ny, b.x - nx, b.y - ny, a.x - nx, a.y - ny);
  }
  for (const p of points) {
    for (let k = 0; k < DOT_SEGMENTS; k += 1) {
      const t0 = ((k / DOT_SEGMENTS) * Math.PI) * 2;
      const t1 = (((k + 1) / DOT_SEGMENTS) * Math.PI) * 2;
      out.push(
        p.x, p.y,
        p.x + Math.cos(t0) * dot, p.y + Math.sin(t0) * dot,
        p.x + Math.cos(t1) * dot, p.y + Math.sin(t1) * dot,
      );
    }
  }
  return new Float32Array(out);
}

function compileShader(gl: Gl, type: number, source: string): WebGLShader | null {
  const sh = gl.createShader(type);
  if (!sh) return null;
  gl.shaderSource(sh, source);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    console.warn('note-scanner: WebGL2 shader compile failed:', gl.getShaderInfoLog(sh));
    gl.deleteShader(sh);
    return null;
  }
  return sh;
}

function linkProgram(gl: Gl, vertSrc: string, fragSrc: string): WebGLProgram | null {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = vs ? compileShader(gl, gl.FRAGMENT_SHADER, fragSrc) : null;
  if (!vs || !fs) {
    if (vs) gl.deleteShader(vs);
    if (fs) gl.deleteShader(fs);
    return null;
  }
  const prog = gl.createProgram();
  if (!prog) {
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return null;
  }
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  // Only the draft program has an attribute; pinning it to 0 lets the draft
  // VAO be configured once at build time instead of queried per frame.
  gl.bindAttribLocation(prog, 0, 'aPos');
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.warn('note-scanner: WebGL2 program link failed:', gl.getProgramInfoLog(prog));
    gl.deleteProgram(prog);
    return null;
  }
  return prog;
}

function makeTexture(gl: Gl): WebGLTexture | null {
  const tex = gl.createTexture();
  if (!tex) return null;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  // NEAREST everywhere. The mask -> image-resolution step IS the F4 upsample,
  // and a LINEAR filter there would yield fractional coverage, which would
  // break the exact one-ULP count arithmetic the R8 targets rely on.
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

function makeTarget(gl: Gl, width: number, height: number): Target | null {
  const tex = makeTexture(gl);
  if (!tex) return null;
  // R8, not R16F: R8 is color-renderable in core WebGL2, R16F is not without
  // EXT_color_buffer_float. See COUNT_UNIT for the 255-mask ceiling this buys.
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R8, width, height);
  const fbo = gl.createFramebuffer();
  if (!fbo) {
    gl.deleteTexture(tex);
    return null;
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(tex);
    return null;
  }
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  return { tex, fbo };
}

/**
 * Does this browser actually give us a usable WebGL2 renderer, right now?
 *
 * A browser can expose `WebGL2RenderingContext` and still refuse a context
 * (blocklisted driver, lost GPU process), and it can hand back a context that
 * then fails to compile, link, or produce a complete R8 framebuffer. So the
 * probe does all four on a 1x1 throwaway canvas and drops the context again.
 * `createDefaultRenderer()` is the only caller.
 */
export function probeWebGL2Support(): boolean {
  if (typeof document === 'undefined') return false;
  try {
    const probe = document.createElement('canvas');
    probe.width = 1;
    probe.height = 1;
    const gl = probe.getContext('webgl2', CONTEXT_ATTRS) as Gl | null;
    if (!gl) return false;
    const prog = linkProgram(gl, VERT_QUAD, FRAG_COMPOSITE);
    const target = prog ? makeTarget(gl, 1, 1) : null;
    if (prog) gl.deleteProgram(prog);
    if (target) {
      gl.deleteFramebuffer(target.fbo);
      gl.deleteTexture(target.tex);
    }
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return Boolean(prog && target);
  } catch {
    // jsdom throws outright from getContext when the `canvas` package is
    // absent, which is exactly the "no WebGL2 here" answer we want.
    return false;
  }
}

export function createWebGL2Renderer(): Renderer {
  let canvas: HTMLCanvasElement | null = null;
  let gl: Gl | null = null;
  let failed = false;
  let contextLost = false;

  let progMask: WebGLProgram | null = null;
  let progEdge: WebGLProgram | null = null;
  let progDilate: WebGLProgram | null = null;
  let progComposite: WebGLProgram | null = null;
  let progDraft: WebGLProgram | null = null;
  let quadVao: WebGLVertexArrayObject | null = null;
  let draftVao: WebGLVertexArrayObject | null = null;
  let draftBuf: WebGLBuffer | null = null;
  const uniforms = new Map<WebGLProgram, Map<string, WebGLUniformLocation | null>>();

  let texWidth = 0;
  let texHeight = 0;
  let bright: Target | null = null;
  let outline: Target | null = null;
  let edgeA: Target | null = null;
  let edgeB: Target | null = null;
  let hoverBright: Target | null = null;
  let hoverOutline: Target | null = null;
  let scratchTex: WebGLTexture | null = null;
  let baseTex: WebGLTexture | null = null;
  let baseSource: CanvasImageSource | null = null;

  let prevSelected = new Set<string>();
  let hoverKey = '';
  /**
   * The coverage each currently-applied id was added WITH. It holds references
   * to arrays SegmentViewer already owns — no copies, no GPU memory — and
   * exists so a deselect can subtract exactly what was added even after the id
   * has dropped out of `scene.masks`. This is the direct analogue of canvas2d's
   * covIdxCache/edgeIdxCache: cache first, `scene.masks` on a miss.
   */
  const appliedMasks = new Map<string, MaskRef>();

  function uni(prog: WebGLProgram, name: string): WebGLUniformLocation | null {
    let m = uniforms.get(prog);
    if (!m) {
      m = new Map();
      uniforms.set(prog, m);
    }
    if (!m.has(name)) m.set(name, gl!.getUniformLocation(prog, name));
    return m.get(name) ?? null;
  }

  function bindTex(unit: number, tex: WebGLTexture | null, prog: WebGLProgram, name: string): void {
    const g = gl!;
    g.activeTexture(g.TEXTURE0 + unit);
    g.bindTexture(g.TEXTURE_2D, tex);
    g.uniform1i(uni(prog, name), unit);
  }

  function buildPrograms(): boolean {
    const g = gl!;
    progMask = linkProgram(g, VERT_QUAD, FRAG_MASK);
    progEdge = linkProgram(g, VERT_QUAD, FRAG_EDGE);
    progDilate = linkProgram(g, VERT_QUAD, FRAG_DILATE);
    progComposite = linkProgram(g, VERT_QUAD, FRAG_COMPOSITE);
    progDraft = linkProgram(g, VERT_DRAFT, FRAG_DRAFT);
    quadVao = g.createVertexArray();
    draftVao = g.createVertexArray();
    draftBuf = g.createBuffer();
    if (!progMask || !progEdge || !progDilate || !progComposite || !progDraft) return false;
    if (!quadVao || !draftVao || !draftBuf) return false;
    g.bindVertexArray(draftVao);
    g.bindBuffer(g.ARRAY_BUFFER, draftBuf);
    g.enableVertexAttribArray(0);
    g.vertexAttribPointer(0, 2, g.FLOAT, false, 0, 0);
    g.bindVertexArray(null);
    return true;
  }

  function destroyTargets(): void {
    const g = gl;
    if (!g) return;
    for (const t of [bright, outline, edgeA, edgeB, hoverBright, hoverOutline]) {
      if (!t) continue;
      g.deleteFramebuffer(t.fbo);
      g.deleteTexture(t.tex);
    }
    bright = outline = edgeA = edgeB = hoverBright = hoverOutline = null;
  }

  function ensureTargets(width: number, height: number): boolean {
    if (bright && texWidth === width && texHeight === height) return true;
    destroyTargets();
    const g = gl!;
    bright = makeTarget(g, width, height);
    outline = makeTarget(g, width, height);
    edgeA = makeTarget(g, width, height);
    edgeB = makeTarget(g, width, height);
    hoverBright = makeTarget(g, width, height);
    hoverOutline = makeTarget(g, width, height);
    texWidth = width;
    texHeight = height;
    // Fresh, zeroed count buffers mean nothing has been accumulated yet.
    prevSelected = new Set();
    appliedMasks.clear();
    hoverKey = '';
    if (!bright || !outline || !edgeA || !edgeB || !hoverBright || !hoverOutline) {
      destroyTargets();
      failed = true;
      return false;
    }
    return true;
  }

  function ensureBase(scene: Scene): void {
    const g = gl!;
    if (baseTex && baseSource === scene.base) return;
    if (!baseTex) baseTex = makeTexture(g);
    if (!baseTex) {
      failed = true;
      return;
    }
    g.bindTexture(g.TEXTURE_2D, baseTex);
    // Never flipped — see the coordinate-convention note at the top.
    g.pixelStorei(g.UNPACK_FLIP_Y_WEBGL, 0);
    try {
      g.texImage2D(g.TEXTURE_2D, 0, g.RGBA, g.RGBA, g.UNSIGNED_BYTE, scene.base);
    } catch (err) {
      // KNOWN EDGE: a cross-origin base image served without CORS headers
      // makes texImage2D throw SecurityError, where canvas2d's drawImage would
      // merely taint the canvas and keep painting. SegmentViewer sets
      // `crossOrigin` on its loader, so this is reachable only for a host that
      // serves the image without `Access-Control-Allow-Origin`. Refusing to
      // paint (with this warning) beats painting a black board silently.
      console.warn('note-scanner: WebGL2 renderer cannot upload the base image:', err);
      failed = true;
      return;
    }
    baseSource = scene.base;
  }

  function maskRef(scene: Scene, id: string): MaskRef | null {
    const cached = appliedMasks.get(id);
    if (cached) return cached;
    const mask = scene.masks.get(id);
    if (!mask) return null;
    const width = mask.width ?? scene.imageWidth;
    const height = mask.height ?? scene.imageHeight;
    // A coverage array that disagrees with its declared dimensions would
    // upload garbage rather than a wrong-but-plausible mask, so refuse it and
    // leave the counts untouched.
    if (mask.coverage.length !== width * height) return null;
    return { coverage: mask.coverage, width, height };
  }

  function uploadScratch(ref: MaskRef): void {
    const g = gl!;
    if (!scratchTex) scratchTex = makeTexture(g);
    g.bindTexture(g.TEXTURE_2D, scratchTex);
    g.pixelStorei(g.UNPACK_ALIGNMENT, 1);
    g.texImage2D(
      g.TEXTURE_2D, 0, g.R8, ref.width, ref.height, 0, g.RED, g.UNSIGNED_BYTE, ref.coverage,
    );
  }

  type Blend = 'add' | 'sub' | 'replace';

  function runPass(
    prog: WebGLProgram,
    target: Target,
    blend: Blend,
    setUniforms: (prog: WebGLProgram) => void,
  ): void {
    const g = gl!;
    g.bindFramebuffer(g.FRAMEBUFFER, target.fbo);
    g.viewport(0, 0, texWidth, texHeight);
    if (blend === 'replace') {
      g.disable(g.BLEND);
    } else {
      g.enable(g.BLEND);
      g.blendFunc(g.ONE, g.ONE);
      // FUNC_REVERSE_SUBTRACT is dst - src, which is the deselect direction.
      g.blendEquation(blend === 'add' ? g.FUNC_ADD : g.FUNC_REVERSE_SUBTRACT);
    }
    g.useProgram(prog);
    g.bindVertexArray(quadVao);
    setUniforms(prog);
    g.drawArrays(g.TRIANGLES, 0, 3);
    g.disable(g.BLEND);
  }

  function clearTarget(target: Target): void {
    const g = gl!;
    g.bindFramebuffer(g.FRAMEBUFFER, target.fbo);
    g.viewport(0, 0, texWidth, texHeight);
    g.disable(g.BLEND);
    g.clearColor(0, 0, 0, 0);
    g.clear(g.COLOR_BUFFER_BIT);
  }

  /** scratch -> edgeA (edge) -> edgeB (horizontal half) -> target (vertical half). */
  function runOutlinePasses(target: Target, blend: Blend, dashed: boolean): void {
    const g = gl!;
    runPass(progEdge!, edgeA!, 'replace', (prog) => {
      bindTex(0, scratchTex, prog, 'uMask');
      g.uniform2f(uni(prog, 'uImageSize'), texWidth, texHeight);
    });
    runPass(progDilate!, edgeB!, 'replace', (prog) => {
      bindTex(0, edgeA!.tex, prog, 'uSrc');
      g.uniform2f(uni(prog, 'uImageSize'), texWidth, texHeight);
      g.uniform2f(uni(prog, 'uStep'), 1, 0);
      g.uniform1f(uni(prog, 'uAmount'), 1);
      g.uniform1i(uni(prog, 'uDashed'), 0);
    });
    runPass(progDilate!, target, blend, (prog) => {
      bindTex(0, edgeB!.tex, prog, 'uSrc');
      g.uniform2f(uni(prog, 'uImageSize'), texWidth, texHeight);
      g.uniform2f(uni(prog, 'uStep'), 0, 1);
      g.uniform1f(uni(prog, 'uAmount'), COUNT_UNIT);
      g.uniform1i(uni(prog, 'uDashed'), dashed ? 1 : 0);
    });
  }

  function applyMask(scene: Scene, id: string, blend: 'add' | 'sub'): void {
    const g = gl!;
    const ref = maskRef(scene, id);
    if (!ref) {
      if (blend === 'sub') appliedMasks.delete(id);
      return;
    }
    uploadScratch(ref);
    runPass(progMask!, bright!, blend, (prog) => {
      bindTex(0, scratchTex, prog, 'uMask');
      g.uniform2f(uni(prog, 'uImageSize'), texWidth, texHeight);
      g.uniform1f(uni(prog, 'uAmount'), COUNT_UNIT);
    });
    runOutlinePasses(outline!, blend, false);
    if (blend === 'add') appliedMasks.set(id, ref);
    else appliedMasks.delete(id);
  }

  function applyDelta(scene: Scene): void {
    const { added, removed, applied } = resolveAppliedDelta(
      prevSelected,
      scene.selectedIds,
      (id) => scene.masks.has(id),
    );
    for (const id of added) applyMask(scene, id, 'add');
    for (const id of removed) applyMask(scene, id, 'sub');
    prevSelected = applied;
  }

  /** Hover is one mask that REPLACES the selection's bright window, so it
   *  never touches the count targets. Same `hoverKey` guard canvas2d uses. */
  function rebuildHover(scene: Scene): void {
    const g = gl!;
    const key = scene.hoveredId ?? '';
    if (key === hoverKey) return;
    hoverKey = key;
    const ref = scene.hoveredId ? maskRef(scene, scene.hoveredId) : null;
    if (!ref) {
      clearTarget(hoverBright!);
      clearTarget(hoverOutline!);
      return;
    }
    uploadScratch(ref);
    runPass(progMask!, hoverBright!, 'replace', (prog) => {
      bindTex(0, scratchTex, prog, 'uMask');
      g.uniform2f(uni(prog, 'uImageSize'), texWidth, texHeight);
      g.uniform1f(uni(prog, 'uAmount'), COUNT_UNIT);
    });
    runOutlinePasses(hoverOutline!, 'replace', true);
  }

  function drawDraft(scene: Scene): void {
    if (scene.draftPoints.length === 0) return;
    const g = gl!;
    const verts = buildDraftGeometry(scene.draftPoints, scene.imageWidth, scene.imageHeight);
    if (verts.length === 0) return;
    g.useProgram(progDraft!);
    g.bindVertexArray(draftVao);
    g.bindBuffer(g.ARRAY_BUFFER, draftBuf);
    g.bufferData(g.ARRAY_BUFFER, verts, g.DYNAMIC_DRAW);
    g.uniform2f(uni(progDraft!, 'uImageSize'), scene.imageWidth, scene.imageHeight);
    g.uniform3f(uni(progDraft!, 'uColor'), OUTLINE_R, OUTLINE_G, OUTLINE_B);
    g.drawArrays(g.TRIANGLES, 0, verts.length / 2);
  }

  function composite(scene: Scene): void {
    const g = gl!;
    g.bindFramebuffer(g.FRAMEBUFFER, null);
    g.viewport(0, 0, canvas!.width, canvas!.height);
    g.disable(g.BLEND);
    g.clearColor(0, 0, 0, 1);
    g.clear(g.COLOR_BUFFER_BIT);
    g.useProgram(progComposite!);
    g.bindVertexArray(quadVao);
    bindTex(0, baseTex, progComposite!, 'uBase');
    bindTex(1, scene.hoveredId ? hoverBright!.tex : bright!.tex, progComposite!, 'uBright');
    bindTex(2, outline!.tex, progComposite!, 'uOutline');
    bindTex(3, hoverOutline!.tex, progComposite!, 'uHoverOutline');
    g.uniform1f(uni(progComposite!, 'uDimAlpha'), DIM_ALPHA);
    g.uniform3f(uni(progComposite!, 'uOutlineColor'), OUTLINE_R, OUTLINE_G, OUTLINE_B);
    g.drawArrays(g.TRIANGLES, 0, 3);
    drawDraft(scene);
  }

  function paint(scene: Scene): void {
    if (!canvas || !gl || failed || contextLost) return;
    const { imageWidth: w, imageHeight: h, devicePixelRatio: dpr } = scene;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    if (!ensureTargets(w, h)) return;
    ensureBase(scene);
    if (failed) return;
    applyDelta(scene);
    rebuildHover(scene);
    composite(scene);
  }

  const onContextLost = (event: Event): void => {
    // Without preventDefault the context is never restorable.
    event.preventDefault();
    contextLost = true;
  };

  const onContextRestored = (): void => {
    contextLost = false;
    if (!gl) return;
    // Every GL object created before the loss is gone. Rebuild from nothing and
    // re-accumulate the selection on the next draw.
    uniforms.clear();
    progMask = progEdge = progDilate = progComposite = progDraft = null;
    quadVao = draftVao = null;
    draftBuf = null;
    bright = outline = edgeA = edgeB = hoverBright = hoverOutline = null;
    scratchTex = null;
    baseTex = null;
    baseSource = null;
    texWidth = texHeight = 0;
    prevSelected = new Set();
    appliedMasks.clear();
    hoverKey = '';
    failed = !buildPrograms();
  };

  return {
    init(next) {
      // SegmentViewer calls init() before every draw, so this must be cheap and
      // must not re-register the context-loss listeners.
      if (canvas === next && (gl || failed)) return;
      canvas = next;
      failed = false;
      contextLost = false;
      gl = next.getContext('webgl2', CONTEXT_ATTRS) as Gl | null;
      if (!gl) {
        failed = true;
        return;
      }
      next.addEventListener('webglcontextlost', onContextLost);
      next.addEventListener('webglcontextrestored', onContextRestored);
      failed = !buildPrograms();
    },
    draw(scene) {
      paint(scene);
    },
    resize(scene) {
      paint(scene);
    },
    dispose() {
      const g = gl;
      if (canvas) {
        canvas.removeEventListener('webglcontextlost', onContextLost);
        canvas.removeEventListener('webglcontextrestored', onContextRestored);
      }
      if (g) {
        destroyTargets();
        if (scratchTex) g.deleteTexture(scratchTex);
        if (baseTex) g.deleteTexture(baseTex);
        for (const p of [progMask, progEdge, progDilate, progComposite, progDraft]) {
          if (p) g.deleteProgram(p);
        }
        if (quadVao) g.deleteVertexArray(quadVao);
        if (draftVao) g.deleteVertexArray(draftVao);
        if (draftBuf) g.deleteBuffer(draftBuf);
      }
      uniforms.clear();
      progMask = progEdge = progDilate = progComposite = progDraft = null;
      quadVao = draftVao = null;
      draftBuf = null;
      scratchTex = null;
      baseTex = null;
      baseSource = null;
      texWidth = texHeight = 0;
      prevSelected = new Set();
      appliedMasks.clear();
      hoverKey = '';
      failed = false;
      contextLost = false;
      gl = null;
      canvas = null;
    },
    /**
     * Drop the retained coverage reference for these ids.
     *
     * Per the Renderer interface's contract this must NOT touch the count
     * targets or `prevSelected`: subtracting here as well as in
     * `resolveAppliedDelta`'s `removed` branch would double-count and drive the
     * R8 counts below zero, where they clamp at 0 and permanently un-brighten
     * pixels other selected masks still cover.
     *
     * No GPU resource is per-id in this renderer — the scratch texture is
     * re-uploaded per delta and nothing per-mask is retained (which is why
     * AC5's texture bound holds) — so this drops only the CPU-side coverage
     * reference, exactly mirroring canvas2d evicting its index caches.
     */
    evict(ids) {
      for (const id of ids) appliedMasks.delete(id);
    },
  };
}
```

- [ ] **Step 6: Run the contract suite and the typecheck**

Run: `npx vitest run src/renderer/renderer-conformance.test.ts && npm run typecheck`
Expected: PASS — 14 tests (7 per backend).

- [ ] **Step 7: Build the browser harness**

Per `docs/autopilot/learnings.md`, plan-dictated browser code that has only been *reviewed* is not evidence. The harness lands here, in the same task as the renderer, so Step 8 is a real `npx playwright test` invocation against real engines rather than a hope deferred to Task 3.

No new dependency: `vite` and `@vitejs/plugin-react` are already devDependencies, and `scripts/sweep-decode.mjs` already drives a programmatic Vite server the same way. `test-results/` and `playwright-report/` are already in `.gitignore`.

Create `tests/browser/dev-server.ts`:

```ts
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { createServer, type ViteDevServer } from 'vite';

const here = path.dirname(fileURLToPath(import.meta.url));
const harnessRoot = path.join(here, 'harness');
const repoRoot = path.resolve(here, '..', '..');

/**
 * One Vite dev server per Playwright worker, on an OS-assigned port.
 *
 * `playwright.config.ts` deliberately has no `webServer` — the mask-PNG specs
 * hand data URLs into an empty document and need nothing served. The renderer
 * specs DO need the real TS modules running in the page, so they start a
 * server themselves and close it in `afterAll`. Port 0 (rather than a fixed
 * port) is what lets chromium, webkit and firefox run fully parallel.
 *
 * `fs.allow` has to reach the repo root because the harness imports from
 * `src/`, which is outside the server root.
 */
export async function startHarnessServer(): Promise<{ url: string; close: () => Promise<void> }> {
  const server: ViteDevServer = await createServer({
    configFile: false,
    root: harnessRoot,
    plugins: [react()],
    logLevel: 'error',
    server: { port: 0, strictPort: false, fs: { allow: [repoRoot] } },
  });
  await server.listen();
  const url = server.resolvedUrls?.local?.[0];
  if (!url) {
    await server.close();
    throw new Error('harness vite server started but reported no local URL');
  }
  return { url, close: () => server.close() };
}
```

Create `tests/browser/harness/index.html`:

```html
<!doctype html>
<meta charset="utf-8" />
<title>renderer harness</title>
<style>
  body { margin: 0; background: #222; }
  canvas { display: block; }
</style>
<div id="root"></div>
<script type="module" src="./harness.ts"></script>
```

Create `tests/browser/harness/harness.ts`:

```ts
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
 */
async function settle(canvas: HTMLCanvasElement, w: number, h: number, repaint: () => void) {
  let prev = readback(canvas, w, h);
  for (let i = 0; i < 60; i += 1) {
    await raf();
    repaint();
    const next = readback(canvas, w, h);
    if (next.length === prev.length && next.every((v, k) => v === prev[k])) return next;
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

export interface HarnessApi {
  webgl2Available(): boolean;
  paint(backend: 'canvas2d' | 'webgl2', spec: SceneSpec): Promise<Frame>;
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
```

- [ ] **Step 8: Write and RUN the first browser spec**

Create `tests/browser/webgl2-renderer.spec.ts`:

```ts
import { expect, test, type Page } from '@playwright/test';
import { startHarnessServer } from './dev-server';
import type { Frame, SceneSpec } from './harness/harness';

let server: { url: string; close: () => Promise<void> };

test.beforeAll(async () => {
  server = await startHarnessServer();
});

test.afterAll(async () => {
  await server?.close();
});

const problems: string[] = [];

test.beforeEach(async ({ page }) => {
  page.on('console', (msg) => {
    if (msg.type() === 'error' || msg.text().includes('note-scanner')) problems.push(msg.text());
  });
  page.on('pageerror', (err) => problems.push(String(err)));
  problems.length = 0;
  await page.goto(server.url);
  await page.waitForFunction(() => Boolean(window.__harness));
});

async function webgl2Available(page: Page): Promise<boolean> {
  return page.evaluate(() => window.__harness.webgl2Available());
}

function pixelAt(frame: Frame, x: number, y: number): [number, number, number, number] {
  const o = (y * frame.width + x) * 4;
  return [frame.pixels[o], frame.pixels[o + 1], frame.pixels[o + 2], frame.pixels[o + 3]];
}

/** One selected rectangle, big enough to have an unambiguous interior, a
 *  dilated border ring, and a lot of untouched background. */
const ONE_MASK: SceneSpec = {
  width: 64,
  height: 48,
  masks: [{ id: 'a', rect: [16, 12, 32, 24] }],
  selected: ['a'],
};

test('reports whether this engine can obtain a webgl2 context', async ({ page }, testInfo) => {
  const available = await webgl2Available(page);
  // Never a failure: which of chromium/webkit/firefox exposes WebGL2 headless
  // is exactly the fact this spec exists to record, and every GPU assertion
  // below skips rather than fails where it does not.
  testInfo.annotations.push({
    type: 'webgl2',
    description: `${testInfo.project.name}: webgl2 ${available ? 'available' : 'UNAVAILABLE'}`,
  });
  console.log(`[webgl2-support] ${testInfo.project.name}: ${available ? 'yes' : 'no'}`);
  expect(typeof available).toBe('boolean');
});

test('the WebGL2 renderer paints a dim wash, a bright window and an orange outline', async ({
  page,
}) => {
  test.skip(!(await webgl2Available(page)), 'this engine has no webgl2 context');
  const frame = await page.evaluate(
    (spec) => window.__harness.paint('webgl2', spec),
    ONE_MASK,
  );

  const inside = pixelAt(frame, 32, 24);
  const outside = pixelAt(frame, 2, 2);
  // The dim layer is rgba(0,0,0,0.55), so background is 0.45x the base colour
  // while the bright window keeps it. Comparing the two rules out both "painted
  // nothing" and "dimmed everything".
  const lum = (p: number[]) => p[0] + p[1] + p[2];
  expect(lum(inside)).toBeGreaterThan(lum(outside) * 1.5);

  // The dilated border ring must contain the outline colour #f97316.
  let orange = 0;
  for (let i = 0; i < frame.pixels.length; i += 4) {
    if (
      Math.abs(frame.pixels[i] - 249) <= 8 &&
      Math.abs(frame.pixels[i + 1] - 115) <= 8 &&
      Math.abs(frame.pixels[i + 2] - 22) <= 8
    ) {
      orange += 1;
    }
  }
  // A radius-3 dilation of a 32x24 rectangle's 1px perimeter is a ring roughly
  // 7px thick around a ~112px perimeter, so hundreds of pixels — but the exact
  // count depends on clipping and is not what this asserts. It asserts the ring
  // exists and is not a stray pixel or two.
  expect(orange).toBeGreaterThan(200);

  expect(problems).toEqual([]);
});

test('a half-resolution mask still paints, through the sampler upsample', async ({ page }) => {
  test.skip(!(await webgl2Available(page)), 'this engine has no webgl2 context');
  // Nothing emits `width` / `height` yet (issue #7 will). This is the F4
  // MECHANISM under test, not an F4 saving: a mask authored at half size must
  // light the same region of the image as one authored at full size.
  const frame = await page.evaluate(
    (spec) => window.__harness.paint('webgl2', spec),
    { ...ONE_MASK, masks: [{ id: 'a', rect: [16, 12, 32, 24], maskScale: 2 }] } as SceneSpec,
  );
  const lum = (p: number[]) => p[0] + p[1] + p[2];
  expect(lum(pixelAt(frame, 32, 24))).toBeGreaterThan(lum(pixelAt(frame, 2, 2)) * 1.5);
  expect(problems).toEqual([]);
});
```

Run: `npx playwright test tests/browser/webgl2-renderer.spec.ts`
Expected: PASS in every project that reports webgl2 available; skipped assertions (not failures) in any that does not.

**This is the step where surprises surface.** Record in the commit message which of chromium / webkit / firefox printed `[webgl2-support] … yes`. If a shader fails to compile, the renderer's `console.warn` lands in `problems` and the test fails with the driver's own infolog — read it and fix the shader here, not in Task 3.

- [ ] **Step 9: Commit**

```bash
git add src/renderer/types.ts src/renderer/webgl2.ts src/renderer/renderer-conformance.test.ts \
        tsconfig.tests.json package.json tests/browser/dev-server.ts tests/browser/harness \
        tests/browser/webgl2-renderer.spec.ts
git commit -m "feat(renderer): WebGL2 backend, run the contract suite over both factories"
```

---

## Task 2: Select WebGL2 by default, with a real probe and a real fallback

**Files:**
- Modify: `src/renderer/index.ts`
- Create: `src/renderer/default-renderer.test.ts`
- Modify: `src/SegmentViewer.tsx:54,158`
- Modify: `src/SegmentViewer.test.tsx`
- Modify: `src/types.ts` (the `renderer` prop doc comment)
- Modify: `src/index.ts`
- Modify: `README.md:52-56`

**Interfaces:**
- Consumes: `createWebGL2Renderer()` and `probeWebGL2Support()` from Task 1's `src/renderer/webgl2.ts`; `createCanvas2DRenderer()` from `src/renderer/canvas2d.ts`.
- Produces: `createDefaultRenderer(): Renderer`, exported from `src/renderer/index.ts` and re-exported from `src/index.ts`. Task 3's viewer harness imports it.

**Why this is one task:** the factory, its only caller, and the docs sentence that names the old default cannot be reviewed apart — a reviewer approving the factory while rejecting the wiring would leave a function nothing calls.

- [ ] **Step 1: Write the failing probe tests**

Create `src/renderer/default-renderer.test.ts`:

```ts
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
```

Run: `npx vitest run src/renderer/default-renderer.test.ts`
Expected: FAIL — `createDefaultRenderer` is not exported from `./index`.

- [ ] **Step 2: Add the factory**

Replace `src/renderer/index.ts` with:

```ts
import { createCanvas2DRenderer } from './canvas2d';
import { createWebGL2Renderer, probeWebGL2Support } from './webgl2';
import type { Renderer } from './types';

export type { Renderer, RendererFactory, Scene, ScenePoint } from './types';
export { createCanvas2DRenderer } from './canvas2d';
export { createWebGL2Renderer } from './webgl2';

/**
 * The backend `SegmentViewer` uses when the caller passes no `renderer` prop.
 *
 * The choice is made by actually attempting a context (and a shader compile, a
 * link, and an R8 framebuffer-completeness check — see `probeWebGL2Support`),
 * never by sniffing for `'WebGL2RenderingContext' in window`: a browser can
 * expose the constructor and still refuse a context on a blocklisted driver or
 * after a lost GPU process. Probing for real is what stops `SegmentViewer` from
 * ever holding a half-dead WebGL renderer.
 */
export function createDefaultRenderer(): Renderer {
  return probeWebGL2Support() ? createWebGL2Renderer() : createCanvas2DRenderer();
}
```

Run: `npx vitest run src/renderer/default-renderer.test.ts`
Expected: PASS.

- [ ] **Step 3: Add the AC6 regression to the existing viewer suite**

`src/SegmentViewer.test.tsx` already renders `SegmentViewer` with `renderer={fakeRenderer}`. Add one test to it — do not restructure the file. Append inside a new `describe` at the end:

```tsx
describe('renderer selection', () => {
  it('an explicitly passed renderer prop overrides the default selection', async () => {
    vi.stubGlobal('Image', FakeImage);
    const injected = fakeRenderer();
    render(
      <SegmentViewer
        imageUrl="https://example.com/image.png"
        imageWidth={40}
        imageHeight={10}
        segments={[]}
        initialSelectedIds={new Set()}
        onSelectionChange={() => {}}
        onCreateSegment={async () => {}}
        renderer={() => injected}
      />,
    );
    // If the default factory had won, this would be a canvas2d renderer calling
    // getContext('2d') on a jsdom canvas, which throws in this environment —
    // so the injected renderer being the one that paints is the whole claim.
    await waitFor(() => expect(injected.draw).toHaveBeenCalled());
    expect(injected.init).toHaveBeenCalled();
  });
});
```

Run: `npx vitest run src/SegmentViewer.test.tsx`
Expected: PASS (the prop already wins today; this pins it before Step 4 changes the default).

- [ ] **Step 4: Wire the viewer**

In `src/SegmentViewer.tsx` line 54, change the import:

```ts
import { createDefaultRenderer, type Renderer, type Scene } from './renderer';
```

and line 158:

```ts
    rendererRef.current = (renderer ?? createDefaultRenderer)();
```

Leave the surrounding comment about the renderer being created once and never swapped mid-mount exactly as it is; it is still true.

Run: `npx vitest run`
Expected: PASS, whole suite. `src/SegmentViewer.test.tsx` still passes because every render there supplies a `renderer` prop, and `createDefaultRenderer` falls back to canvas2d in jsdom anyway.

- [ ] **Step 5: Export the new factories and fix the README**

In `src/index.ts`, replace the renderer export line with:

```ts
export { createCanvas2DRenderer, createWebGL2Renderer, createDefaultRenderer } from './renderer';
```

These are additive named exports. Nothing is removed and nothing becomes required, so AC8's "no public package API is removed or made required" still holds.

In `src/types.ts`, the `renderer?: RendererFactory` prop's doc comment still says `/** Painting backend. Defaults to Canvas2D. */`. That sentence is now false — change it to:

```ts
  /** Painting backend. Defaults to `createDefaultRenderer` (WebGL2 when the
   *  browser yields a context, canvas2d otherwise). */
  renderer?: RendererFactory;
```

In `README.md`, replace the `## Renderers` section body (currently lines 53-56) with:

```markdown
`renderer?: RendererFactory` swaps the painting backend. The default is
`createDefaultRenderer`, which paints with `createWebGL2Renderer` when the
browser actually yields a `webgl2` context (probed, not sniffed) and falls back
to `createCanvas2DRenderer` otherwise. All three are exported. The `Renderer`
interface takes coverage arrays, not images — see
[issue #9](https://github.com/BoTime/NoteScanner/issues/9).
```

- [ ] **Step 6: Verify and commit**

Run: `npm run test && npm run typecheck`
Expected: PASS.

Run: `grep -n "createDefaultRenderer" README.md src/index.ts src/renderer/index.ts src/SegmentViewer.tsx`
Expected: exactly four files, one or more lines each. (This grep is anchored to the identifier, not a prose word, and nothing else in the repo contains it.)

```bash
git add src/renderer/index.ts src/renderer/default-renderer.test.ts src/SegmentViewer.tsx \
        src/SegmentViewer.test.tsx src/index.ts src/types.ts README.md
git commit -m "feat(renderer): probe for WebGL2 and make it the default backend"
```

---

## Task 3: Browser verification — the differential, the texture bound, context loss, and the fallback

**Files:**
- Modify: `tests/browser/harness/harness.ts` (add members; do not restructure)
- Create: `tests/browser/harness/viewer-harness.tsx`
- Modify: `tests/browser/harness/index.html` (load the viewer harness too)
- Modify: `tests/browser/webgl2-renderer.spec.ts` (add specs; keep Task 1's three)
- Create: `docs/measurements/2026-09-01-webgl2-vs-canvas2d.md`

**Interfaces:**
- Consumes: `window.__harness` from Task 1; `createDefaultRenderer` from Task 2; `encodeMaskPng(coverage, width, height): Promise<string>` from `src/segmenter/core/mask-encode.ts` (canvas-free and pure, so it runs unchanged in the page — this is the production encoder, not a stand-in).
- Produces: nothing else in the repo imports. This task produces evidence.

**Why this is one task:** every criterion here is read out of the same harness page in the same Playwright run, and the measurements doc is written from that run's output. Splitting it would mean two dispatches sharing one set of numbers.

- [ ] **Step 1: Extend the harness**

Append to `tests/browser/harness/harness.ts` — before the `const api: HarnessApi` declaration, add:

```ts
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
      return { supported: false, threwWhileLost: false, before, after: before, width: spec.width, height: spec.height };
    }
    const lost = new Promise<void>((r) =>
      canvas.addEventListener('webglcontextlost', () => r(), { once: true }),
    );
    const restored = new Promise<void>((r) =>
      canvas.addEventListener('webglcontextrestored', () => r(), { once: true }),
    );
    ext.loseContext();
    await lost;
    let threwWhileLost = false;
    try {
      renderer.draw(scene);
    } catch {
      threwWhileLost = true;
    }
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
```

Add these three members to the `HarnessApi` interface and to the `api` object literal:

```ts
  paintSequence(
    backend: 'canvas2d' | 'webgl2',
    specs: SceneSpec[],
  ): Promise<{ frames: Frame[]; peak: number; live: number }>;
  loseAndRestore(spec: SceneSpec): Promise<LossResult>;
```

```ts
  paintSequence,
  loseAndRestore,
```

- [ ] **Step 2: Add the viewer harness**

Create `tests/browser/harness/viewer-harness.tsx`:

```tsx
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
    const timer = setTimeout(() => reject(new Error('SegmentViewer never reached status "ready"')), 15000);
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
  // One more frame so the paint that follows "ready" has landed.
  await new Promise<void>((r) => requestAnimationFrame(() => r()));

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
```

In `tests/browser/harness/index.html`, add a second module script AFTER the first (order matters — `viewer-harness.tsx` reads `window.__harness`):

```html
<script type="module" src="./harness.ts"></script>
<script type="module" src="./viewer-harness.tsx"></script>
```

- [ ] **Step 3: Add the AC3 differential spec**

Append to `tests/browser/webgl2-renderer.spec.ts`:

```ts
/**
 * Scenes the two backends are genuinely comparable on. Every mask is authored
 * at image resolution (maskScale 1) because canvas2d ignores `width`/`height`
 * by contract, and dpr is pinned to 1 inside the harness.
 */
const DIFF_SCENES: { name: string; gate: number; spec: SceneSpec }[] = [
  {
    name: 'single selection',
    // Both backends produce the same binary coverage, the same radius-3
    // dilation and the same 0.55 dim factor, so this should be near-exact.
    gate: 0.01,
    spec: ONE_MASK,
  },
  {
    name: 'two overlapping selections',
    gate: 0.01,
    spec: {
      width: 64,
      height: 48,
      masks: [
        { id: 'a', rect: [8, 8, 28, 28] },
        { id: 'b', rect: [24, 16, 30, 26] },
      ],
      selected: ['a', 'b'],
    },
  },
  {
    name: 'hover (dashed outline)',
    gate: 0.01,
    spec: { ...ONE_MASK, masks: [{ id: 'a', rect: [16, 12, 32, 24] }], selected: [], hovered: 'a' },
  },
  {
    name: 'draft polygon',
    // Loose ON PURPOSE and reported rather than asserted tight: canvas2d
    // strokes an antialiased path with round caps and `ctx.arc` dots, webgl2
    // draws hard-edged triangles and 16-gon discs. The disagreement here is
    // edge antialiasing, and it is the number this row exists to publish.
    gate: 0.2,
    spec: {
      ...ONE_MASK,
      draft: [
        { x: 8, y: 8 },
        { x: 52, y: 14 },
        { x: 30, y: 40 },
      ],
    },
  },
];

/** A pixel disagrees when any channel differs by more than 8/255 — enough
 *  slack for a rounding difference in the dim multiply, not enough to hide a
 *  wrong colour or a missing outline. */
function disagreements(a: Frame, b: Frame): number {
  let count = 0;
  for (let i = 0; i < a.pixels.length; i += 4) {
    for (let c = 0; c < 4; c += 1) {
      if (Math.abs(a.pixels[i + c] - b.pixels[i + c]) > 8) {
        count += 1;
        break;
      }
    }
  }
  return count;
}

for (const row of DIFF_SCENES) {
  test(`AC3: ${row.name} — per-pixel disagreement vs the canvas2d baseline`, async ({
    page,
  }, testInfo) => {
    test.skip(!(await webgl2Available(page)), 'this engine has no webgl2 context');
    const [c2d, wgl] = await page.evaluate(
      async (spec) => [
        await window.__harness.paint('canvas2d', spec),
        await window.__harness.paint('webgl2', spec),
      ],
      row.spec,
    );
    const total = row.spec.width * row.spec.height;
    const differing = disagreements(c2d, wgl);
    const fraction = differing / total;
    // Reported as a number, per AC3 — never as an unmeasured "matches".
    const line = `[ac3] ${testInfo.project.name} | ${row.name} | ${differing}/${total} px (${(fraction * 100).toFixed(3)}%)`;
    console.log(line);
    testInfo.annotations.push({ type: 'ac3', description: line });
    expect(fraction).toBeLessThanOrEqual(row.gate);
    expect(problems).toEqual([]);
  });
}
```

- [ ] **Step 4: Add the AC5 texture-bound spec**

Append:

```ts
test('AC5: live GPU textures stay bounded as the selection grows', async ({ page }, testInfo) => {
  test.skip(!(await webgl2Available(page)), 'this engine has no webgl2 context');
  // 24 non-overlapping 8x8 masks on a 64x48 board, selected one more at a
  // time. A renderer that kept a texture per mask would end at 24+ live
  // textures; this one allocates base + bright + outline + edgeA + edgeB +
  // hoverBright + hoverOutline + scratch = 8, and nothing per mask.
  const specs = [];
  const masks = [];
  for (let i = 0; i < 24; i += 1) {
    masks.push({ id: `m${i}`, rect: [(i % 8) * 8, Math.floor(i / 8) * 8, 8, 8] });
  }
  for (let n = 0; n <= 24; n += 1) {
    specs.push({ width: 64, height: 48, masks, selected: masks.slice(0, n).map((m) => m.id) });
  }
  const { peak } = await page.evaluate(
    (s) => window.__harness.paintSequence('webgl2', s),
    specs as unknown as SceneSpec[],
  );
  console.log(`[ac5] ${testInfo.project.name} | peak live textures across 0->24 masks: ${peak}`);
  testInfo.annotations.push({ type: 'ac5', description: `peak textures ${peak}` });
  // The bound, not the exact figure: an implementation detail that adds one
  // more constant target should not fail this, but one that adds a texture per
  // mask must. 24 masks with a per-mask texture would be >= 24.
  expect(peak).toBeLessThanOrEqual(12);
  expect(problems).toEqual([]);
});
```

- [ ] **Step 5: Add the AC7 context-loss spec**

Append:

```ts
test('AC7: draw() during context loss is silent, and the scene comes back after restore', async ({
  page,
}, testInfo) => {
  test.skip(!(await webgl2Available(page)), 'this engine has no webgl2 context');
  const result = await page.evaluate((spec) => window.__harness.loseAndRestore(spec), ONE_MASK);
  test.skip(!result.supported, 'this engine does not expose WEBGL_lose_context');
  expect(result.threwWhileLost).toBe(false);
  const before: Frame = { width: result.width, height: result.height, pixels: result.before };
  const after: Frame = { width: result.width, height: result.height, pixels: result.after };
  const differing = disagreements(before, after);
  const total = result.width * result.height;
  console.log(`[ac7] ${testInfo.project.name} | post-restore disagreement: ${differing}/${total} px`);
  testInfo.annotations.push({ type: 'ac7', description: `restore delta ${differing}/${total}` });
  expect(differing / total).toBeLessThanOrEqual(0.001);
  expect(problems).toEqual([]);
});
```

- [ ] **Step 6: Add the AC2 / AC6 viewer specs**

Append:

```ts
test('AC2: with webgl2 unavailable the viewer falls back to canvas2d and still paints', async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.__viewerHarness.mountViewer({ denyWebgl2: true }),
  );
  // The probe ran for real and was refused, then canvas2d took the canvas.
  expect(result.contextKinds).toContain('webgl2');
  expect(result.contextKinds).toContain('2d');
  expect(result.pixels).not.toBeNull();
  const frame: Frame = { width: result.width, height: result.height, pixels: result.pixels! };
  const lum = (p: number[]) => p[0] + p[1] + p[2];
  // Selected mask covers [16,12]-[48,36]; (32,24) is inside it, (2,2) is not.
  expect(lum(pixelAt(frame, 32, 24))).toBeGreaterThan(lum(pixelAt(frame, 2, 2)) * 1.5);
  await page.evaluate(() => window.__viewerHarness.unmountViewer());
  expect(problems).toEqual([]);
});

test('AC6: an explicitly passed renderer prop is the one that paints', async ({ page }) => {
  const result = await page.evaluate(() =>
    window.__viewerHarness.mountViewer({ useRendererProp: true }),
  );
  expect(result.propDrawCalls).toBeGreaterThan(0);
  // The default factory probes with getContext('webgl2'); the injected renderer
  // never does. Its absence from the log is what proves the prop won, in a
  // browser where webgl2 IS available.
  expect(result.contextKinds).not.toContain('webgl2');
  await page.evaluate(() => window.__viewerHarness.unmountViewer());
  expect(problems).toEqual([]);
});
```

- [ ] **Step 7: Run the whole browser suite and capture the numbers**

Run: `npx playwright test 2>&1 | tee /tmp/webgl2-browser-run.txt`
Expected: PASS. Keep the `[webgl2-support]`, `[ac3]`, `[ac5]` and `[ac7]` lines — they are the raw data for Step 8. Note which projects skipped and why.

Run: `npm run test && npm run typecheck`
Expected: PASS (the harness now type-checks under `tsconfig.tests.json` from Task 1).

- [ ] **Step 8: Write the measurements doc from that run**

Create `docs/measurements/2026-09-01-webgl2-vs-canvas2d.md`, filling every `<...>` from `/tmp/webgl2-browser-run.txt`. Do not round, do not estimate, and do not write a spread or an average you have not computed from the table itself.

```markdown
# WebGL2 vs canvas2d — per-pixel differential

Produced by `npx playwright test tests/browser/webgl2-renderer.spec.ts` on
<date>, Playwright <version>, on <OS / machine>. Raw lines are the `[ac3]`,
`[ac5]` and `[ac7]` entries in that run's stdout.

## What is and is not measured here

This is a **correctness** differential, not a performance result. Nothing in
this run changes what the segmenter worker emits, so no `filter` or
`mask-encode` timing improves and none is claimed (AC9). The F4 saving — the
sampler doing the upsample — is latent until issue #7 makes the worker emit
256x256 masks.

Read the numbers with these caveats, which change how they should be read:

- `devicePixelRatio` is pinned to 1. At dpr > 1 canvas2d upscales the base
  image with the browser's 2D resampler and webgl2 with a NEAREST sampler, so a
  dpr > 1 differential would measure two resamplers rather than two renderers.
- A pixel counts as disagreeing when ANY RGBA channel differs by more than
  8/255.
- The `draft polygon` row is expected to be the largest by a wide margin and is
  not a defect: canvas2d strokes an antialiased path with `ctx.arc` dots,
  webgl2 rasterizes hard-edged triangles and 16-gon discs.
- Engines with no headless WebGL2 context skip every row rather than reporting
  zero. Skips are listed below; an empty row is a skip, not agreement.

## Results (64x48 = 3072 pixels per scene)

| engine | scene | disagreeing px | % |
| --- | --- | --- | --- |
| <engine> | single selection | <n> | <pct> |
| <engine> | two overlapping selections | <n> | <pct> |
| <engine> | hover (dashed outline) | <n> | <pct> |
| <engine> | draft polygon | <n> | <pct> |

Engines that reported no webgl2 context and skipped: <list, or "none">.

## Texture bound (AC5)

Peak live GPU textures while the selection grew from 0 to 24 masks:
<per-engine numbers>. The renderer allocates a constant set — base, bright,
outline, two dilation ping-pong targets, two hover targets, and one reused mask
scratch — and nothing per selected mask.

## Context loss (AC7)

Post-restore disagreement against the pre-loss frame: <per-engine numbers>.
Engines with no `WEBGL_lose_context` extension: <list, or "none">.
```

- [ ] **Step 9: Record the AC4 and AC9 evidence**

Run: `git diff --stat main...HEAD -- src/segmenter`
Expected: no output. That is AC9 — the segmenter worker still calls `post_process_masks`, `thresholdMask` and `encodeMaskPng`, untouched.

Run: `git diff --stat main...HEAD -- src/core`
Expected: no output. Combined with the next command, that is AC4: hit-testing is the same full-resolution CPU code it was, and its suites still pass.

Run: `npx vitest run src/core/segment-viewer-logic.test.ts`
Expected: PASS.

If either diff is non-empty, stop: something outside this plan's file list was edited.

- [ ] **Step 10: Commit**

```bash
git add tests/browser docs/measurements/2026-09-01-webgl2-vs-canvas2d.md
git commit -m "test(renderer): browser differential, texture bound, context loss and fallback"
```

---

## Self-review

**Spec coverage.** Every section of `docs/superpowers/specs/2026-09-01-webgl2-renderer-design.md` maps to a task: `src/renderer/webgl2.ts` and the GPU-state/passes sections → Task 1 Step 5; `Scene.masks` optional dimensions → Task 1 Step 1; `src/renderer/index.ts`'s `createDefaultRenderer` and `SegmentViewer.tsx:158` → Task 2; hit-testing untouched → Global Constraints plus Task 3 Step 9; lifecycle and error handling → Task 1 Step 5 (`init`, `dispose`, `evict`, context loss) and Task 3 Step 5; testing → Task 1 Step 3 (parameterized conformance) and Task 3.

**Acceptance criteria.** AC1 → Task 1 Steps 3/6. AC2 → Task 2 Step 1 (probe fallback, unit) and Task 3 Step 6 (real browser). AC3 → Task 3 Steps 3, 7, 8. AC4 → Task 3 Step 9. AC5 → Task 3 Step 4. AC6 → Task 2 Step 3 (unit) and Task 3 Step 6 (browser). AC7 → Task 3 Step 5. AC8 → Task 1 Step 1 plus Task 2 Step 5's additive-only exports, exercised for webgl2 by Task 1 Step 8's `maskScale: 2` spec and for canvas2d by the whole existing suite continuing to pass. AC9 → Task 3 Step 9.

**Deliberate divergences from canvas2d, all documented at the code that causes them:**

- Counts saturate past 255 concurrently selected masks (R8 UNORM); canvas2d's `Uint16Array` does not. Stated at `makeTarget` / `COUNT_UNIT`.
- The draft polygon is hard-edged triangles rather than an antialiased stroke. This is the `draft polygon` row of the AC3 table, published rather than hidden.
- A cross-origin base image without CORS headers makes `texImage2D` throw where canvas2d's `drawImage` would merely taint the canvas. Handled in `ensureBase` with a `console.warn` and a refusal to paint; `SegmentViewer` sets `crossOrigin` on its loader, so this is reachable only against a host that serves the image without `Access-Control-Allow-Origin`.
- `evict()` drops only the retained coverage references. Like canvas2d's `evict`, it deliberately does not touch the count targets or `prevSelected`.

**Verification-step scope check.** `npm run typecheck` did not cover `tests/**` before this plan; Task 1 Step 2 adds `tsconfig.tests.json` and extends the script, so every line of harness and spec code this plan dictates is type-checked. The `grep -n "createDefaultRenderer"` in Task 2 Step 6 is anchored to an identifier that appears nowhere else in the repo, and nothing this plan mandates in prose matches it. No step asserts a match count that has not been derived from a real run.
