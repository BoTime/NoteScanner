# note-scanner

In-browser post-it note scanner. A React segment/mask viewer, plus an optional
everything-mode segmenter that runs SAM on WebGPU entirely client-side.

Zero runtime dependencies — except the optional
[`/segmenter` subpath](#note-scannersegmenter-optional-prototype), which needs
`@huggingface/transformers`. `react` and `react-dom` are peers (`>=18`).

## Install

```bash
npm install note-scanner
```

## Use

```tsx
import { SegmentViewer } from 'note-scanner';
import 'note-scanner/styles.css';

<SegmentViewer
  imageUrl={url}
  imageWidth={w}
  imageHeight={h}
  segments={segments}          // { id, maskUrl, index?, excluded? }[]
  initialSelectedIds={selected}
  onSelectionChange={setSelected}
  onCreateSegment={async (points) => {}}
/>
```

The component fetches nothing except the mask images from the URLs it is
handed. The consumer owns segments and selection state.

## Theming

Five custom properties, light defaults. Redefine them under your own selector
for dark mode:

`--sv-accent`, `--sv-canvas-bg`, `--sv-border`, `--sv-dim`, `--sv-muted-fg`

## Subpaths

- `note-scanner` — the component
- `note-scanner/core` — the pure math (hit-testing, coverage, bounds,
  connected components, view transform), React-free
- `note-scanner/segmenter` — the optional WebGPU segmenter (see below)
- `note-scanner/styles.css`

## Renderers

`renderer?: RendererFactory` swaps the painting backend; the default is
`createCanvas2DRenderer`. The `Renderer` interface takes coverage arrays, not
images — see [issue #1](https://github.com/BoTime/NoteScanner/issues/1) for the
planned WebGL2 renderer.

## Development

```bash
npm install
npm run playground   # Vite dev app
npm run test
npm run typecheck
npm run smoke        # build + artifact check
```

## `note-scanner/segmenter` (optional, prototype)

An in-browser everything-mode segmenter: it runs SAM on WebGPU in a module
worker and produces `ViewerSegment[]` for `<SegmentViewer>` to render. It is a
**separate subpath** because it needs `@huggingface/transformers` plus tens of
megabytes of ONNX weights; that package is an **optional peer**, so importing
the main entry or `/core` still pulls in nothing at runtime.

```ts
import { createSegmenter, isWebGPUAvailable } from 'note-scanner/segmenter';

if (isWebGPUAvailable()) {
  const segmenter = createSegmenter();
  const { segments, timings, counts } = await segmenter.segment(bitmap, { pointsPerSide: 16 });
}
```

Requires WebGPU — there is no CPU fallback by design. The default worker is
referenced as `new URL('./worker/segmenter.worker.ts', import.meta.url)`, which
Vite and webpack 5 resolve; any other host passes its own
`createSegmenter({ createWorker })`. See `playground/SegmentView.tsx` for a
worked example.

This default only resolves when consuming the package from source (the
`development` export condition, as this repo's Vite playground does). A
consumer of the published npm tarball must pass `createSegmenter({ createWorker })`
with their own worker construction, since the published `dist/segmenter/` does
not include a bundled worker file.

### Measuring dtype and `batchSize`

`npm run playground` serves a **Compare** tab (`playground/CompareView.tsx`) that
runs a fixed seven-row matrix — fp16 vs fp32 at 16 and 32 points per side, plus a
`batchSize` curve — and renders the per-phase timings, the `encode + decode`
subtotal, and an fp16-vs-fp32 mask-agreement summary as markdown you can paste
into an issue. It needs a real GPU browser session; the presets, the greedy
best-IoU pairing and the markdown renderer live in `playground/compare.ts` and
are unit-tested without one.

The comparison uses the one addition this makes to the package surface:
`segment(bitmap, { keepRawMasks: true })` retains the full-resolution
`RawMask[]` on the result as `rawMasks`. It is a client-side retention flag, not
an inference parameter — the worker ignores it — and it is off by default
because retaining ~50 full-resolution coverage arrays pins tens of megabytes for
as long as you hold the result.

### Status

The segmenter is a **prototype**. A full everything-mode run currently takes
~98 s per image (excluding one-time model load), and ~91% of that is overhead
rather than model inference. The analysis and the staged plan to get it to
~7-10 s live in [issue #1](https://github.com/BoTime/NoteScanner/issues/1).

The default checkpoint is `Xenova/slimsam-77-uniform`, configurable via the
`modelId` option. No transformers.js-loadable MobileSAM exists on the HF Hub
today (`nielsr/mobilesam` and `bhllx/mobilesam` ship PyTorch weights only, with
no `onnx/` folder), so SlimSAM stands in; the image backbone is ~2-3% of a run,
so the choice barely moves the timings.

## License

MIT — see [LICENSE](LICENSE).
