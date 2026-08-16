# @sambacollab/segment-viewer

Canvas segment/mask viewer for React. Zero runtime dependencies; `react` and
`react-dom` are peers (`>=18`).

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
