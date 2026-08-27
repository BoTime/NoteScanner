# M3 + M4 — 1-bit indexed PNG, encoded off the main thread

GitHub issue #5 (parent #1, Wave 1, Tier 1, depends on nothing).

## The problem, as measured

A mask carries 1 bit/px and is handed to a general-purpose lossless encoder as
32-bit truecolor + alpha. Per-mask cost is 1,069.2 ms in Run A (16 pps) and
1,069.9 ms in Run B (32 pps) — identical to within 0.7 ms across different
images, grid densities and mask counts, which is conclusive that the cost is
strictly per-mask and driven by resolution.

`mask-encode` today: 50.5 s / 51.3% (Run A), 37.4 s / 22.5% (Run B).
After M3 + M4: ~1.6 s and ~1.2 s. Saving ~48.9 s and ~36.2 s, plus the same
amount again in UI jank removed — the encode loop currently runs serially on
the main thread at `createSegmenter.ts:115-129`.

M3 is the only change in the whole optimization analysis with production value
rather than prototype-only value: R2 masks get ~32x smaller, cutting storage
and real-user transfer.

## Current shape

- `src/segmenter/core/mask-encode.ts` — `maskToRgba` expands coverage to 32-bit
  RGBA; `encodeMaskPng` puts it on an `OffscreenCanvas` (falling back to
  `document.createElement('canvas')`) and calls `convertToBlob`, then
  base64s the result through `bytesToDataUrl`.
- `src/segmenter/worker/segmenter.worker.ts:259-280` — after NMS, builds
  `RawMask[]` (`{ coverage, area }`) and posts it with the coverage buffers in
  the transfer list, ~0.7 MB each, specifically so the main thread can encode.
- `src/segmenter/createSegmenter.ts:115-129` — loops the masks serially on the
  main thread, timing each `encodeMaskPng` call and splicing a `mask-encode`
  `PhaseTiming` into the worker's report via `summarizePhase`.
- `src/SegmentViewer.tsx:80-92` — decodes a mask PNG back to coverage, counting
  a pixel covered when `alpha > 0 && red > 0`.

## Design

### Approach chosen for the compression layer

The PNG writer needs a zlib stream for IDAT. `CompressionStream('deflate')` is
used: it is a platform global, it is zlib-wrapped (exactly PNG's IDAT format —
`'deflate-raw'` would be the wrong one), and it is present in Chrome 80+,
Safari 16.4+ and Firefox 113+. Every browser that has WebGPU has it, so it adds
no compatibility floor to a package that already requires WebGPU, and it is
present in Node 22, which is what makes the round-trip testable with no new
dependency. Being native, it costs approximately nothing against a
1,069 ms/mask baseline.

Two alternatives were rejected. Stored (uncompressed) deflate blocks are ~20
lines and need no platform support, but give up all compression — and today's
masks go through canvas's *compressed* PNG encoder, so the issue's headline
production claim (R2 masks ~32x smaller) would not survive. A deflate library
(`fflate`, `pako`) works, but this package's core advertises zero runtime
dependencies and `scripts/smoke-build.mjs` enforces it; paying that for
something the platform already provides is a bad trade.

### M3 — the 1-bit indexed writer

`src/segmenter/core/mask-encode.ts` is rewritten as pure JS with no canvas
anywhere. `encodeMaskPng(coverage, width, height): Promise<string>` keeps its
exact existing signature, so no call site changes shape.

Internals:

- Pack each row to `1 + ceil(width / 8)` bytes: one `0x00` filter byte (filter
  type None for every row — deflate handles the redundancy), then the row's
  bits MSB-first, set where covered.
- Deflate the packed rows through `new CompressionStream('deflate')`.
- CRC32 via the standard table-driven reflected `0xEDB88320` polynomial.
- Assemble: signature `89 50 4E 47 0D 0A 1A 0A`, IHDR (width, height, bit depth
  1, colour type 3, compression 0, filter 0, interlace 0), PLTE
  (`00 00 00`, `FF FF FF`), tRNS (one byte, `0x00`), IDAT, IEND.
- `bytesToDataUrl` survives untouched — it is already chunk-seam tested.

`maskToRgba` and both canvas paths are deleted; nothing needs a 32-bit
expansion any more, and being canvas-free is precisely what lets the same
function run in the worker, on the main thread, and in Node.

**Decoder contract preserved exactly.** Palette index 1 is opaque white, index
0 is fully transparent (entries past the tRNS length are opaque by spec).
`SegmentViewer.tsx:83-86` counts a pixel covered when `alpha > 0 && red > 0`;
covered decodes to `(255,255,255,255)` and uncovered to `(0,0,0,0)`. The
`maskUrl: string` contract is 100% intact and the PNG stays natively
`<img>`-decodable.

### M4 — the encode loop moves into the existing worker

The worker already holds every coverage array immediately before it transfers
them out, so encoding there is both the cheapest move and the one that deletes
the most work.

- `segmenter.worker.ts` encodes right after NMS, recording each call into its
  existing timing accumulator as `mask-encode`.
- It posts `masks: EncodedMask[]` — `{ maskUrl: string; area: number }` — with
  no transfer list. The ~0.7 MB x N coverage transfer stops happening at all.
- `types.ts`: `RawMask` is replaced by `EncodedMask`; `SegmenterResponse`'s
  `done` variant carries `EncodedMask[]`. `width`/`height` stay on the
  response — they describe the result and dropping them would be a gratuitous
  extra breaking change. `PHASE_ORDER`'s doc comment loses the sentence
  claiming `mask-encode` is main-thread work; every phase is now in the worker.
- `createSegmenter.ts` collapses to a synchronous map over `message.masks`.
  The `summarizePhase` splice goes (the worker's report is now complete and is
  passed through whole), as do the `encodeMaskPng`/`summarizePhase` imports and
  the main-thread `try/catch` — an encode failure now arrives as the worker's
  own `error` message already carrying `phase: 'mask-encode'`. `totalMs` is
  still measured on the main thread so worker spawn and bitmap transfer stay
  inside the reported total.

This is serial within the one worker, which is what the issue's own ~1.6 s /
~1.2 s estimate already assumes. A worker pool was considered and rejected: it
would need a new worker file plus pool lifecycle, and the coverage arrays would
still have to cross a boundary (worker-to-worker transfer needs the main thread
or a MessageChannel to relay) — all to beat a target the simple version already
hits.

### Error handling

Strictly narrowed, not added. The one new failure mode is `CompressionStream`
being absent, which throws inside the worker's existing try block and surfaces
as `SegmenterFailure('mask-encode', ...)` exactly as a canvas failure did.

### Testing

Two layers.

**Node (`src/segmenter/core/mask-encode.test.ts`, rewritten).** Round-trips by
parsing the chunks, inflating the IDAT with `DecompressionStream('deflate')` —
Node's own zlib, genuinely independent of our encoder — unfiltering, and
unpacking MSB-first, then asserting byte-equality with the input coverage.
Fixtures: all-zero, all-one, widths of 1/7/8/9/17 (where the row padding
lives), 1xN and Nx1, and a batch generated from a seeded PRNG so the property
coverage is deterministic rather than flaky. It also asserts the padding bits
in each row's final byte are zero, that every chunk's CRC validates, that IHDR
declares bit depth 1 / colour type 3, that PLTE is 6 bytes and tRNS is the
single byte `0x00`, and that the pre-deflate IDAT input length is exactly
`height * (1 + ceil(width / 8))` — the divide-by-32 claim, stated structurally.

**Real browsers (`tests/browser/mask-png.spec.ts`, new, under a new
`playwright.config.ts` with `chromium`, `webkit` and `firefox` projects).**
Encodes in Node, hands the data URLs into the page, decodes through
`new Image()` and a canvas, applies the viewer's own `alpha > 0 && red > 0`
predicate, and compares. `page.setContent` is sufficient — the images are data
URLs, so no dev server and no WebGPU are involved, which is exactly what lets
all three browsers run headless. Same fixtures as the Node layer, including the
width-not-a-multiple-of-8 cases.

Wiring: `@playwright/test` as a devDependency and an `npm run test:browser`
script. Vitest's `include` (`src/**/*.test.ts(x)`, `playground/**/*.test.ts`)
never matches `tests/`, so `npm test` is unchanged. CI gains a step that runs
`npx playwright install --with-deps` and then `npm run test:browser`.

Note that `.claude/autopilot.json`'s `test_command` is
`npm run typecheck && npm test`, which does NOT include the browser spec —
CI on the pull request is what runs it.

### Measurement (criteria 3 and 4)

Not code. `playground/SegmentView.tsx:259` already renders every `PHASE_ORDER`
phase from `result.timings.phases`, so `mask-encode` keeps reporting itself
with no change. The recipe, to be run by the developer on a WebGPU machine
after merge: run the playground at `pointsPerSide` 16 and again at 32, read the
`mask-encode` total off the results table, and take a DevTools performance
trace across the encode window to confirm the main thread stays unblocked. The
before/after numbers get posted on issue #5.

## Files touched

- `src/segmenter/core/mask-encode.ts` — rewritten (1-bit writer, CRC32, chunks)
- `src/segmenter/core/mask-encode.test.ts` — rewritten (round-trip properties)
- `src/segmenter/core/types.ts` — `RawMask` -> `EncodedMask`, response shape,
  `PHASE_ORDER` doc comment
- `src/segmenter/worker/segmenter.worker.ts` — encode after NMS, post URLs
- `src/segmenter/createSegmenter.ts` — synchronous map, imports, timing
  passthrough
- `src/segmenter/createSegmenter.test.ts` — updated for the new message shape
- `playwright.config.ts` — new
- `tests/browser/mask-png.spec.ts` — new
- `package.json` — `@playwright/test` devDependency, `test:browser` script
- `.github/workflows/ci.yml` — browser install + browser test step

## Out of scope

M2 (encode masks at 256x256, issue #7) and M5/F4 (WebGL2 renderer, issue #9)
are separate issues. This change is deliberately independent of both.

## Acceptance criteria

Every criterion below is `(non-ui)`, and that is a decision the brainstorm
settled rather than an oversight. This issue changes no pixel in the
application: the `maskUrl: string` contract is 100% intact and `SegmentViewer`
renders identically before and after, so there is no UI surface for the
pipeline to open.

AC1 and AC2 **are** verified in real browsers — Chromium, WebKit and Firefox —
but by the `@playwright/test` spec this pull request adds and CI runs, not by
driving the app's UI. AC3 and AC4 need a real WebGPU segmentation run in the
playground on a machine with a GPU; they ship as a written measurement recipe
(see "Measurement" above) that the developer executes after merge, and nothing
in this pipeline measures them.

- AC1 (non-ui) — **round-trip is byte-identical**:
  `coverage -> 1-bit PNG -> <img> decode -> threshold` reproduces the input
  `Uint8Array` exactly.
  Property-tested over random masks and the edge cases: all-zero, all-one, and
  width not a multiple of 8 — the row-padding case is where hand-rolled 1-bit
  PNG writers break.
- AC2 (non-ui) — the encoded PNG decodes in Chrome, Safari and Firefox. `tRNS`
  on an indexed PNG is well supported but worth confirming, since the fallback
  is silent. Verified by `tests/browser/mask-png.spec.ts` running against
  `chromium`, `webkit` and `firefox` in CI.
- AC3 (non-ui) — **satisfied outside this branch by the developer, on a WebGPU
  machine; not a gate on this PR.** The `mask-encode` phase total is recorded
  before and after at both operating points (`pointsPerSide` 16 and 32) and
  posted on issue #5.
- AC4 (non-ui) — **satisfied outside this branch by the developer, on a WebGPU
  machine; not a gate on this PR.** The main thread is not blocked for the
  duration of encoding, confirmed on a DevTools performance trace rather than
  by eye.
- AC5 (non-ui) — the decoder contract is preserved exactly: a covered pixel
  decodes to `(255,255,255,255)` and an uncovered pixel to `(0,0,0,0)`, so
  `SegmentViewer`'s existing `alpha > 0 && red > 0` predicate reproduces the
  coverage with no change to `SegmentViewer.tsx`, and `maskUrl` remains a
  `string` data URL that an `<img>` decodes natively.
- AC6 (non-ui) — the encoded bytes are structurally a 1-bit indexed PNG: IHDR
  declares bit depth 1 and colour type 3, PLTE is 6 bytes, tRNS is the single
  byte `0x00`, every chunk's CRC validates, the padding bits in each row's
  final byte are zero, and the pre-deflate IDAT input length is exactly
  `height * (1 + ceil(width / 8))` — the divide-by-32 size claim, asserted
  structurally.
- AC7 (non-ui) — the encode runs in the worker, not on the main thread: the
  worker posts `masks: EncodedMask[]` (`{ maskUrl, area }`) with **no transfer
  list**, no coverage buffer crosses the worker boundary, and `mask-encode`
  appears in the worker's own timing report with `createSegmenter` splicing
  nothing in.
- AC8 (non-ui) — `mask-encode.ts` contains no canvas or `OffscreenCanvas` use
  and the package's core keeps zero runtime dependencies, so the same
  `encodeMaskPng` runs unchanged in the worker, on the main thread and in Node,
  and `scripts/smoke-build.mjs` still passes.
- AC9 (non-ui) — `npm test` (vitest) is unchanged in scope — its `include`
  never matches `tests/` — and CI gains a step that installs browsers and runs
  `npm run test:browser`, which is what actually executes AC2.
