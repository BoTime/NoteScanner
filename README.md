# @sambacollab/segment-viewer

Canvas segment/mask viewer for React. Zero runtime dependencies — except the
optional [`/segmenter` subpath](#sambacollabsegment-viewersegmenter-optional-prototype),
which needs `@huggingface/transformers`. `react` and `react-dom` are peers
(`>=18`).

## Install

Add one line to your `.npmrc`:

```
@sambacollab:registry=https://npm.pkg.github.com
```

```bash
npm install @sambacollab/segment-viewer
```

## Use

```tsx
import { SegmentViewer } from '@sambacollab/segment-viewer';
import '@sambacollab/segment-viewer/styles.css';

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

- `@sambacollab/segment-viewer` — the component
- `@sambacollab/segment-viewer/core` — the pure math (hit-testing, coverage,
  bounds, connected components, view transform), React-free
- `@sambacollab/segment-viewer/styles.css`

## Renderers

`renderer?: RendererFactory` swaps the painting backend; the default is
`createCanvas2DRenderer`.

## Development

```bash
npm run playground -w @sambacollab/segment-viewer   # Vite dev app
npm run test -w @sambacollab/segment-viewer
npm run smoke -w @sambacollab/segment-viewer        # build + artifact check
```

## `@sambacollab/segment-viewer/segmenter` (optional, prototype)

An in-browser everything-mode segmenter: it runs SAM on WebGPU in a module
worker and produces `ViewerSegment[]` for `<SegmentViewer>` to render. It is a
**separate subpath** because it needs `@huggingface/transformers` plus tens of
megabytes of ONNX weights; that package is an **optional peer**, so importing
the main entry or `/core` still pulls in nothing at runtime.

```ts
import { createSegmenter, isWebGPUAvailable } from '@sambacollab/segment-viewer/segmenter';

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
`development` export condition, as this workspace's Vite playground does). A
consumer of the published npm tarball must pass `createSegmenter({ createWorker })`
with their own worker construction, since the published `dist/segmenter/` does
not include a bundled worker file.
