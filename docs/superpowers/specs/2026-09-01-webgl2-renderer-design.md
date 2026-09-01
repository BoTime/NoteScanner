# WebGL2 renderer (issue #9 — M5 + F4)

Two rendering paths coexist behind the existing `RendererFactory` extension
point: today's `canvas2d`, and a new WebGL2 backend. The `Renderer` interface is
otherwise unchanged, and no public package API changes.

Issue #9. Wave 4, tier 2 — the renderer half of the "remove the full-resolution
intermediate as a category" work.

## Scope, and what is deliberately deferred

The segmenter worker is **not** touched in this run. It keeps calling
`post_process_masks` (full-resolution upsample), `thresholdMask` and
`encodeMaskPng`, so every mask reaching the viewer is still an image-sized 1-bit
PNG. Issue #7 flips the worker to 256x256 later, and — because of the optional
per-mask dimensions described below — will need no renderer change when it does.

Consequently this run delivers **M5** (mask coverage uploaded straight into a
GPU texture, with no PNG -> `ImageData` -> `ImageBitmap` round trip on the main
thread) and the *mechanism* for **F4** (the sampler performs the upsample), but
F4's measured saving stays latent until something actually produces 256x256
masks.

Stated plainly, so no reader mistakes this for a perf result: **the timing table
in issue #9 is not realized by this run.** Nothing here makes `filter` or
`mask-encode` faster, because nothing here changes what the worker emits. What
lands is the structure that makes the saving collectable later, plus M5's
removal of the main-thread decode.

Also out of scope, per the issue: **N5**, the GPU compute-shader IoU matrix.
WebGL2 has no compute shaders and it would break the WebGL2 fallback story.

## Components

### `src/renderer/webgl2.ts` (new)

Exports `createWebGL2Renderer(): Renderer`, a peer of `createCanvas2DRenderer`.

### `src/renderer/index.ts`

Gains `createDefaultRenderer()`. It probes support by actually attempting
`getContext('webgl2')` on a throwaway canvas rather than sniffing for
`'WebGL2RenderingContext' in window` — a browser can expose the constructor and
still refuse a context (blocklisted driver, lost GPU process). On a failed probe
it returns a canvas2d renderer, so `SegmentViewer` never holds a half-dead
WebGL renderer.

### `src/SegmentViewer.tsx`

The `rendererRef` initializer changes from `renderer ?? createCanvas2DRenderer`
to `renderer ?? createDefaultRenderer` (`src/SegmentViewer.tsx:158`). An
explicitly passed `renderer` prop still wins, so the existing escape hatch is
intact.

### `src/renderer/types.ts`

`Scene.masks` entries gain **optional** `width?` / `height?` — the mask's own
natural size. `canvas2d` ignores them entirely and keeps operating at full image
resolution exactly as it does today. WebGL2 uses them for its texture dimensions
when present and falls back to `imageWidth` / `imageHeight` when absent. Nothing
emits them yet, so this is behaviourally a no-op in this run; it is what makes
the WebGL2 path resolution-agnostic.

Because the fields are optional, this is not a breaking interface change: a
third-party `RendererFactory` compiles and runs unchanged.

## GPU state and compositing

Approach: mirror `canvas2d`'s incremental accumulation on the GPU.

Three textures, two of them framebuffer-attached at image resolution:

- `brightTex` (`R8`, FBO) — per-pixel count of selected masks covering it.
- `outlineTex` (`R8`, FBO) — per-pixel count of selected masks whose dilated
  edge covers it.
- one reused `R8` scratch texture, `texImage2D`'d once per mask being added or
  removed.

`R8` rather than `R16F` because `R8` is color-renderable in core WebGL2 and
needs no `EXT_color_buffer_float`. Additive blending at `1/255` increments a
UNORM8 target by exactly one ULP, so the counts are exact up to **255
concurrently selected masks**. Past 255 the format would have to become `R16F`
with `EXT_color_buffer_float`; the code states that ceiling at the point where
the format is chosen.

### Passes

1. **Delta pass** — runs only when the selection changed. The added/removed sets
   come from `resolveAppliedDelta` in `src/core`, reused rather than
   reimplemented, so the WebGL2 path and the canvas2d path agree on delta
   semantics by construction. For each added id: upload its coverage into the
   scratch texture and draw a unit quad into `brightTex` with additive blending,
   then into `outlineTex` through a radius-3 dilation shader matching
   `buildEdgeCoverage`'s semantics. Removed ids draw the same two quads with
   `FUNC_REVERSE_SUBTRACT`.
2. **Hover pass** — hover is a single mask that replaces the selection's bright
   window, so it never touches the count buffers. It lives in its own texture
   pair, rebuilt only when `hoveredId` changes, behind the same `hoverKey` guard
   `canvas2d` already uses.
3. **Composite pass** — every `draw` / `resize`, one full-screen quad with no
   per-mask work: the base image texture, the dim wash applied where the active
   bright count (hover when hovering, else selection) is zero, the outline colour
   where the outline count is non-zero, then the draft polygon as line geometry
   with its vertex dots.

The base image (`scene.base`, a `CanvasImageSource`) is uploaded once per image
and released on `dispose`.

Peak VRAM is therefore a constant number of textures — base, two count targets,
scratch, and the hover pair — independent of how many masks are selected. That
is the bound acceptance criterion AC5 asks for; bit-packing masks 8-per-`RGBA8`,
which the issue offers as a fallback, is unnecessary because no per-mask texture
is retained at all.

## Hit-testing

Untouched. It stays on the CPU `coverage` array at full resolution, where it is
already pixel-exact, so there is no coordinate scaling and no accuracy traded.
That settles the issue's deliberate decision (1) in favour of keeping the CPU
array rather than adopting GPU picking: picking would buy exactness this path
already has, at the cost of a `readPixels` stall on every click.

## Lifecycle and error handling

- `evict(ids)` drops only per-id cached GL resources. Per the `Renderer`
  interface's own doc comment it must **not** touch the count buffers or
  `prevSelected` — double-subtracting would underflow the counts.
- `dispose()` deletes every texture, framebuffer, program and buffer, and is
  idempotent.
- `createDefaultRenderer()`'s pre-flight probe (`probeWebGL2Support()`) links
  every program the renderer uses on a throwaway 1x1 canvas; a failed
  `getContext`, or any of those five programs failing to compile or link, or a
  failed framebuffer completeness check, makes the probe answer `false`, and
  the factory hands back `canvas2d`. That is the only failure check that
  changes the factory's answer. A failure inside an **already-selected live
  instance** — the same checks, but after WebGL2 won — only marks that
  instance unusable; `paint()` becomes a permanent no-op for it, and nothing
  falls back to canvas2d mid-session. This is a deliberate scope limit, not an
  oversight: swapping backends mid-session is its own failure mode (which
  canvas gets the 2d context, what happens to the accumulated GPU state) that
  this run does not implement or test.
- `webglcontextlost` makes `draw()` a no-op until `webglcontextrestored`, which
  rebuilds programs and textures and repaints from the next scene.
- `draw()` before `init()` resolves a context must be a no-op, not a throw — the
  existing interface contract.
- Each renderer instance holds one live WebGL2 context for its lifetime, and
  `createDefaultRenderer()` creates and discards one more during its probe.
  Browsers cap the number of live contexts (Chrome enforces roughly 16) and
  evict the oldest on overflow, firing `webglcontextlost` on a viewer that is
  still on screen; nothing in this renderer calls `restoreContext()` on that
  path, so an evicted viewer stays blank until its scene next changes. This is
  a ceiling `canvas2d` does not have, and a consumer mounting many viewers at
  once should pass `renderer={createCanvas2DRenderer}` explicitly rather than
  rely on the default.

## Testing

- The existing vitest conformance suite in
  `src/renderer/renderer-conformance.test.ts` is parameterized over both
  factories for the contract assertions that need no real pixels: draw before
  init, idempotent dispose, draw after dispose, `evict` tolerating unknown and
  duplicate ids, `resize` sizing the backing store from dpr. jsdom has no WebGL,
  so the WebGL2 case runs against a mocked `webgl2` context.
- Every claim that is genuinely about the GPU goes to Playwright in the existing
  `tests/browser/` harness, which already runs chromium, webkit and firefox with
  no `webServer`. There: both renderers paint the same scene, and pixels are read
  back. This is where AC3 lives, reported as a quantified per-pixel disagreement
  count against the canvas2d baseline rather than an unmeasured "matches" claim,
  and where the real fallback selection assertion for AC2 lives.
- AC5 is made falsifiable in that harness by wrapping the context's
  `createTexture` / `deleteTexture` and asserting the live texture count stays
  bounded as the selection grows.
- AC4 is satisfied by construction, since hit-testing is untouched
  full-resolution CPU code; this spec records that reasoning and points at the
  existing `hitTest` / `hitTestAll` suites in
  `src/core/segment-viewer-logic.test.ts` as the standing evidence, rather than
  adding a test that asserts nothing changed.

This follows the repository's own recorded planning rule: a GPU invariant is not
papered over with a pure-function test that looks equivalent.

## Acceptance criteria

- AC1 (non-ui) — the WebGL2 renderer passes the existing `RendererFactory`
  conformance suite: `draw()` before `init()` is a no-op, `dispose()` is
  idempotent, `draw()` after `dispose()` is safe, `evict()` tolerates unknown and
  duplicate ids, and `resize()` sizes the backing store from `devicePixelRatio`.
- AC2 (ui) — `canvas2d` still passes that same suite, and is actually the
  renderer selected in a running browser when a `webgl2` context cannot be
  obtained; the viewer still paints correctly in that fallback.
- AC3 (ui) — painting the same scene through both renderers in a real browser
  and reading the pixels back yields a per-pixel disagreement count against the
  canvas2d baseline that is measured and reported as a number, not asserted as
  "matches".
- AC4 (non-ui) — click hit-testing returns the same segment ids as before this
  change, evidenced by the unchanged full-resolution CPU `hitTest` / `hitTestAll`
  suites continuing to pass.
- AC5 (ui) — with the WebGL2 renderer live in a browser, the number of GPU
  textures alive stays bounded by a small constant as the selection grows from
  empty to many masks; it does not grow per selected mask.
- AC6 (ui) — an explicitly passed `renderer` prop still overrides the default
  selection, so a caller-supplied renderer is the one that paints.
- AC7 (ui) — after a simulated `webglcontextlost`, `draw()` paints nothing and
  throws nothing; after `webglcontextrestored` the viewer repaints the scene
  correctly.
- AC8 (non-ui) — `Scene.masks` entries accept optional `width` / `height`;
  `canvas2d` ignores them and behaves exactly as today, and WebGL2 falls back to
  `imageWidth` / `imageHeight` when they are absent. No public package API is
  removed or made required.
- AC9 (non-ui) — the segmenter worker is unchanged: it still calls
  `post_process_masks`, `thresholdMask` and `encodeMaskPng`, and no `filter` or
  `mask-encode` timing improvement is claimed for this run.
