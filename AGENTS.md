# `SegmentViewer`'s mask-load contract

`SegmentViewer.tsx` owns a single `useEffect` that fetches and decodes mask
PNGs and paints them onto a canvas via a `Renderer`. It is the most
performance-sensitive code in this package — every rule below exists because
getting it wrong either re-decodes work that was already done, or silently
paints a mask's geometry against the wrong pixels.

## The mask-load effect's dep array is pinned

```ts
// deps: [imageUrl, imageWidth, imageHeight, maskSignature, maskRetryNonce]
```

This list is exhaustive as written. Nothing that varies per segment may be
added to it. The effect keys on `maskSignature` — a stable string built only
from `(id, maskUrl)` pairs — rather than the consumer's `segments` array,
because any mutation to per-segment scalar state (`selected`, `reviewed`,
`flagged`, `extractedText`, …) produces a new `segments` array reference. If
the effect depended on that array, an unrelated checkbox toggle would
re-fetch and re-decode every mask PNG at full image resolution.

**Symptom of getting this wrong:** an unrelated per-segment interaction
(reviewed/flagged checkbox, inline text edit) feels laggy by hundreds of
milliseconds, because a mask decode fires on the main thread before the
paint. The fix is always "narrow the effect deps," never "memoize harder."

## `masksRef` is a long-lived, incrementally-updated map

`masksRef.current` is not reset to an empty `Map` on every effect run. A
pure planner, `planMaskDiff` (`core/mask-loading.ts`), decides — from the
previous vs. next image key and previous vs. next mask ref lists — which ids
survive as the same object references (`keep`), which must be fetched fresh
(`load`), and which must be discarded (`evict`). An entry is discarded only
when:

- its id leaves the `maskSignature` (the segment was removed), or
- its `maskUrl` changes for an existing id (the mask itself changed), or
- the **image key** changes (`imageUrl` / `imageWidth` / `imageHeight` — see
  below).

Everything else survives untouched across a segment add. This is what makes
adding one segment to a board of N cost one fetch/decode instead of N+1.

## Discarding an entry must also evict the renderer's per-id caches

Per-mask derived geometry — coverage indices, dilated edge indices — lives
in the `Renderer`'s own closure (`covIdxCache` / `edgeIdxCache` in
`renderer/canvas2d.ts`), not in the component. `getCovIndices` /
`getEdgeIndices` consult their cache **before** `scene.masks`, keyed by
segment id alone. A surviving cache entry for a reused or url-changed id
would silently paint the **old geometry** against the new mask.

So whenever the component evicts an id from `masksRef`, it must also call
`rendererRef.current?.evict?.(ids)`. `Renderer.evict` is **optional** on the
interface — `renderer?: RendererFactory` is public package API, and a
third-party renderer must not break on a minor upgrade — so every call site
is `rendererRef.current?.evict?.(...)`, never a bare `.evict(...)`.

**`Renderer.evict` clears per-id derived geometry only.** Count-buffer
(`brightCounts` / `outlineCounts`) and `prevSelected` bookkeeping stay owned
by `resolveAppliedDelta`, driven by `scene.selectedIds` on every `draw()`.
Do not make `evict` subtract counts too — the component only ever evicts an
id that has just left `selectedIds` (segment removed) or is about to be
reloaded (url changed while still selected does not arise from the flows
this package drives today), so `resolveAppliedDelta`'s `removed` branch
already performs the subtraction on the very next draw. Subtracting in both
places double-subtracts and drives the `Uint16Array` counts negative, which
underflows to ~65535 and permanently brightens those pixels. **Never evict
an id that is still in `selectedIds` and still has a live `masksRef`
entry** — the diff plan is constructed so this never happens; do not add a
code path that violates it.

## The effect's cleanup must not be a full wipe

React runs an effect's cleanup function before **every** re-run, not just
unmount. A cleanup that unconditionally nulls `baseImageRef`, resets
`masksRef`, and disposes the renderer would destroy exactly the entries the
diff path exists to preserve — on every keystroke-adjacent re-render, not
just when the image genuinely changes.

The full teardown lives in `tearDownMaskState`, called from two places only:

1. the **`fullReset` branch** inside the effect body, when `planMaskDiff`
   reports the image key changed (or this is the first run); and
2. a **dedicated unmount-only effect** — `useEffect(() => () =>
   tearDownMaskState(), [tearDownMaskState])`.

The effect's own cleanup is just `cancelled = true`. **If a new per-mask
cache is ever added to this component or its renderer, it must be added to
`tearDownMaskState`, not to the effect's inline cleanup** — that is the one
sentence that keeps the next cache from silently leaking across images.

## The image key includes width and height, not just the url

```ts
buildImageKey({ imageUrl, imageWidth, imageHeight })
// → `${imageUrl}|${imageWidth}|${imageHeight}`
```

A mask's `coverage` array is sized to `imageWidth * imageHeight`. Keying on
`imageUrl` alone would let a dimension change (the same photo re-served at a
different size) reuse masks sized for the old dimensions — a correctness
bug, not a missed optimization. The `|` separator is load-bearing: a bare
concatenation would let `(url: 'a', w: 1, h: 23)` and
`(url: 'a', w: 12, h: 3)` collide on the same key.

## The render gate and the draw-trigger are coupled

The component does not always show a loading placeholder while
`status !== 'ready'`. A same-image incremental reload (e.g. one segment
added) keeps the existing canvas and its already-decoded masks visible
instead — controlled by `hasReadyFrameRef`, a ref set true the first time a
frame is actually painted for the current image key, and cleared by
`tearDownMaskState`:

```ts
const showPlaceholder =
  status === 'error' || (status !== 'ready' && !hasReadyFrameRef.current);
```

`status === 'error'` **always** shows the placeholder regardless of the
latch — the retry button lives there, and a mask failure is
terminal-but-recoverable by design (see `viewer-status.ts`). Nothing about
`resolveViewerStatus` or the reported `status` changes; only what this
component renders while loading a same-image batch.

Because the canvas can now stay mounted through a diff-path reload instead
of unmounting/remounting, the draw effect's gate must widen to match, **and
its deps must include `masksList`** so a mask that finishes decoding
actually triggers a repaint:

```ts
const canPaint =
  status === 'ready' || (status !== 'error' && hasReadyFrameRef.current);

useEffect(() => {
  if (!canPaint) return;
  drawFrame();
  hasReadyFrameRef.current = true;
}, [canPaint, drawFrame, masksList]);
```

`masksList` is not read by `drawFrame` directly (it reads `masksRef`, a
ref) — but it gets a **new array reference on every successful mask
decode**, and `buildScene`'s own deps
(`[imageWidth, imageHeight, selectedIds, hoveredId, draftPoints]`) contain
nothing that changes when a mask lands. Without `masksList` in the draw
effect's deps, a newly drawn, auto-selected segment would stay unhighlighted
**indefinitely** rather than briefly — this is a functional bug, not a
cosmetic one, once the canvas no longer remounts to force a redraw.

**`masksList` must never migrate into the mask-load effect's own dep
array** — that array stays pinned per the rule at the top of this file. The
two effects sit close together in the source; the mistake would be silent
and would produce an infinite reload loop.

Keep `canPaint` and `showPlaceholder` derived from the same two inputs
(`status`, `hasReadyFrameRef`) so a future edit cannot drift them apart — a
`canPaint` that is `true` while the placeholder is showing would draw to a
null canvas ref.
