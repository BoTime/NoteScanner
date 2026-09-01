# WebGL2 vs canvas2d — per-pixel differential

Produced by `npx playwright test tests/browser/webgl2-renderer.spec.ts` on
2026-08-31, Playwright 1.62.1, on macOS 15.7.4 / Apple M4 (arm64), headless.
Engine builds: chromium 151.0.7922.34, webkit 26.5, firefox 153.0. Raw lines
are the `[ac3]`, `[ac5]` and `[ac7]` entries in that run's stdout.

## What is and is not measured here

This is a **correctness** differential, not a performance result. Nothing in
this run changes what the segmenter worker emits, so no `filter` or
`mask-encode` timing improves and none is claimed (AC9). The F4 saving — the
sampler doing the upsample — is latent until issue #7 makes the worker emit
256x256 masks.

Nor does this change remove the per-mask PNG -> `ImageData` -> `Uint8Array`
coverage decode at `src/SegmentViewer.tsx:80`; that still runs on the main
thread exactly as before. What the WebGL2 backend removes is the renderer's
per-rebuild `countsToImageData(...)` -> `createImageBitmap(...)` round trip.
No timing for either is measured here.

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
| chromium | single selection | 0 | 0.000% |
| chromium | two overlapping selections | 0 | 0.000% |
| chromium | hover (dashed outline) | 0 | 0.000% |
| chromium | draft polygon | 94 | 3.060% |
| webkit | single selection | 0 | 0.000% |
| webkit | two overlapping selections | 0 | 0.000% |
| webkit | hover (dashed outline) | 0 | 0.000% |
| webkit | draft polygon | 94 | 3.060% |
| firefox | single selection | 0 | 0.000% |
| firefox | two overlapping selections | 0 | 0.000% |
| firefox | hover (dashed outline) | 0 | 0.000% |
| firefox | draft polygon | 95 | 3.092% |

Engines that reported no webgl2 context and skipped: none. All three engines
obtained a real WebGL2 context (`[webgl2-support] chromium: yes`,
`webkit: yes`, `firefox: yes`) and ran every row.

The three mask rows disagree on zero pixels on every engine — not "near zero",
and not asserted as byte equality either: zero means no pixel differs by more
than the 8/255 threshold above. That is the expected outcome: both backends
derive the same binary coverage, the same radius-3 dilation, the same `rgb(249, 115, 22)`
outline and the same `rgba(0, 0, 0, 0.55)` dim wash, and at dpr 1 there is no
resampling anywhere to introduce a rounding difference.

The `draft polygon` rows are the only non-zero ones, and the whole disagreement
is edge antialiasing: 94 px (3.060%) on chromium and webkit, 95 px (3.092%) on
firefox. The spread across the three engines is 1 px — firefox rasterizes one
more edge pixel of the antialiased canvas2d stroke than the other two do. The
row's gate is 20%, deliberately loose and published rather than asserted tight.

### Would this differential catch a flipped composite?

Every layer in the composite fragment shader is sampled through the same
`vUv = vec2(p.x, 1.0 - p.y)`, so a wrong sign there mirrors the base image and
the mask layers together — a bug that a centred, doubly-symmetric fixture
cannot see from luminance probes alone. It is visible here, because the harness
base image is asymmetric on both axes: its checker term is
`((x >> 3) + (y >> 3)) & 1` on a 64x48 board, and both `(47 - y) >> 3` and
`(63 - x) >> 3` flip that parity, so all 3072 base pixels change by 40/255
(18/255 under the dim wash) when mirrored on either axis.

Measured, by scoring each canvas2d baseline against the MIRROR of the webgl2
frame from the same run (chromium), with a throwaway scoring pass that is not
part of the committed spec suite — it is not re-run automatically, and
reproducing the table means mirroring a captured frame and re-scoring it by
hand the same way. The reasoning above (the fixture geometry and the
`makeBase` checker formula) is what the committed suite actually enforces on
every run; this table is the one-time measurement that reasoning predicts:

| scene | true | vertical flip | horizontal flip | gate |
| --- | --- | --- | --- | --- |
| single selection | 0 | 2316 | 2316 | 30 px |
| two overlapping selections | 0 | 2056 | 2208 | 30 px |
| hover (dashed outline) | 0 | 3072 | 3072 | 30 px |
| draft polygon | 94 | 2314 | 2292 | 614 px |

Every row exceeds its gate by roughly two orders of magnitude, so a uniform
composite flip fails the differential loudly. The pixels a mirrored frame can
still agree on are the ones the constant-colour orange outline covers in BOTH
the true frame and its mirror, which is why the three solid-outline rows land
near 2/3 of the frame rather than all of it, and why the dashed-outline hover
row — whose gaps rarely line up with themselves — reaches all 3072.

The `two overlapping selections` row is additionally asymmetric in the MASKS
(x [8,36) and [24,54), y [8,36) and [16,42), none of which map to themselves
under a 64- or 48-flip), so it is the row that would also catch a flip confined
to the coverage upload or the mask pass, which would leave the base image
where it is.

## Texture bound (AC5)

Peak live GPU textures while the selection grew from 0 to 24 masks, one
renderer instance across all 25 frames: chromium 8, webkit 8, firefox 8. The
renderer allocates a constant set — base, bright, outline, two dilation
ping-pong targets, two hover targets, and one reused mask scratch — and nothing
per selected mask. A renderer holding one texture per mask would have ended at
24 or more. The spec's assertion is the bound (`peak <= 12`), not the figure.

## Context loss (AC7)

Post-restore disagreement against the pre-loss frame: chromium 0/3072, webkit
0/3072, firefox 0/3072. `draw()` while the context was lost threw on none of
the three. Engines with no `WEBGL_lose_context` extension: none.

One harness detail this exposed, recorded because it is a real cross-engine
difference and not a test artefact: `restoreContext()` must be called from a
fresh task, not from the microtask that `await`ing the `webglcontextlost` event
resumes in. That microtask checkpoint still runs inside the browser's dispatch
of the event, and chromium and webkit only mark restoration allowed once that
dispatch has finished and observed `preventDefault()`. Calling it too early
made webkit log `INVALID_OPERATION: restoreContext: context restoration not
allowed` and left `webglcontextrestored` unfired on both engines, while firefox
allowed it. `loseAndRestore` in `tests/browser/harness/harness.ts` yields a
task before restoring.

## Fallback and renderer override (AC2, AC6)

Not numeric, read out of the same page by mounting the real `SegmentViewer`:

- AC2 — with `getContext('webgl2')` refused, the context log for a mount reads
  `webgl2` (the probe, refused) followed by `2d`, the viewer reaches status
  `ready`, and its canvas paints a bright window that is more than 1.5x the
  luminance of the dimmed background. Passes on all three engines.
- AC6 — with an explicit `renderer` prop, that renderer receives the draw calls
  and `webgl2` never appears in the context log, in browsers where WebGL2 *is*
  available. Passes on all three engines.
