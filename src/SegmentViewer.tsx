'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangleIcon,
  CheckIcon,
  CropIcon,
  HandIcon,
  Loader2Icon,
  MaximizeIcon,
  MinusIcon,
  PlusIcon,
  RefreshCwIcon,
  XIcon,
} from './ui/icons';
import { Popup, type PopupItem } from './ui/Popup';
import { formatSegmentLabelDefault } from './segment-label';
import type { Point, SegmentViewerProps } from './types';
import {
  MIN_SCALE,
  MAX_SCALE,
  PAN_THRESHOLD,
  MaskLoadTracker,
  applyMenuAction,
  buildImageKey,
  buildSegmentMenuItems,
  canReuseBaseImage,
  classifyMaskFailure,
  clampTransform,
  describeMaskError,
  hitTestAll,
  isModifierKey,
  isTypingTarget,
  panModeForZoomCrossing,
  parseMaskSignature,
  planMaskDiff,
  planMaskRetry,
  resolveCursor,
  resolveViewerStatus,
  screenToImage,
  shouldHonorModifierPan,
  shouldPanOnPress,
  shouldStartPanDrag,
  shouldSuppressContextMenu,
  zoomAt,
  type CursorMode,
  type MaskError,
  type MaskRef,
  type SegmentMaskData,
  type SegmentMenuAction,
  type ViewTransform,
} from './core';
import { createCanvas2DRenderer, type Renderer, type Scene } from './renderer';

interface PopupState {
  segmentIds: string[];
  x: number;
  y: number;
}

/** A decoded mask: just the data click hit-testing and the bright-window union need. */
type LoadedMask = { coverage: Uint8Array; area: number };

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load ${src}`));
    img.src = src;
  });
}

function buildMaskData(
  img: HTMLImageElement,
  width: number,
  height: number,
): LoadedMask {
  // The decode canvas is a scratch buffer: we read its pixels into `coverage`
  // and discard it. Nothing per-segment is retained for rendering — the bright
  // window and outline are rebuilt from the coverage arrays on demand by the
  // renderer.
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Could not get 2d context for mask');
  ctx.drawImage(img, 0, 0, width, height);
  const data = ctx.getImageData(0, 0, width, height).data;
  const coverage = new Uint8Array(width * height);
  let area = 0;
  // Mask PNGs are white-on-black. A pixel counts as covered only when it is
  // non-transparent AND non-black (`alpha > 0 && red > 0`) — the shared coverage
  // contract used by the preview crop and the server-side crop.
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    if (data[i + 3] > 0 && data[i] > 0) {
      coverage[p] = 1;
      area += 1;
    }
  }
  return { coverage, area };
}

export function SegmentViewer({
  imageUrl,
  imageWidth,
  imageHeight,
  segments,
  initialSelectedIds,
  onSelectionChange,
  onCreateSegment,
  maxHeight = 'calc(100dvh - 28rem)',
  onRefreshMaskUrls,
  onStatusChange,
  formatSegmentLabel,
  renderer,
  className,
}: SegmentViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const baseImageRef = useRef<HTMLImageElement | null>(null);
  // The decoded base image is cached ACROSS effect runs, unlike `baseImageRef`
  // which the cleanup nulls on every re-run. Keyed by `imageUrl` so a genuinely
  // different image never reuses the previous one; a mask retry re-runs PHASE 2
  // against this cache instead of re-fetching a multi-megabyte photo.
  const baseImageCacheRef = useRef<{ url: string; img: HTMLImageElement } | null>(
    null,
  );
  const masksRef = useRef<Map<string, LoadedMask>>(new Map());
  // The previous run's mask refs and image identity. REFS, not state or deps:
  // the load effect's dep array must stay exactly
  // [imageUrl, imageWidth, imageHeight, maskSignature, maskRetryNonce], and
  // refs add no deps — that is precisely why they are refs.
  const prevMaskRefsRef = useRef<MaskRef[]>([]);
  const prevImageKeyRef = useRef<string | null>(null);
  // Ids that were still failing when the last load() run finished. A manual
  // retry bumps `maskRetryNonce` only — the image key and the ref list are
  // unchanged, so planMaskDiff correctly reports a no-op diff (keep =
  // everything, load = []). The effect layers these ids back on top of
  // `plan.load` to form the batch it actually fetches, so a retry re-attempts
  // exactly what failed and never re-requests an already-decoded mask.
  // Rewritten at the end of every load() run; cleared by tearDownMaskState.
  const lastFailedRefsRef = useRef<MaskRef[]>([]);
  // True once a frame has actually been painted for the CURRENT image key. It
  // is what lets the render gate keep the canvas up through an incremental
  // (same image) mask reload instead of swapping the whole board for a
  // spinner. Cleared by tearDownMaskState, so a genuinely new image still
  // shows the placeholder from scratch — no separate "is this still the same
  // image" check is needed, because tearDownMaskState runs exactly on a
  // fullReset (an image-key change) and on unmount.
  const hasReadyFrameRef = useRef(false);
  // Held in a ref so a parent recreating the callback each render cannot
  // restart the mask pipeline — the load effect must key ONLY on
  // [imageUrl, imageWidth, imageHeight, maskSignature].
  const refreshMaskUrlsRef = useRef(onRefreshMaskUrls);
  useEffect(() => {
    refreshMaskUrlsRef.current = onRefreshMaskUrls;
  }, [onRefreshMaskUrls]);

  const formatLabel = formatSegmentLabel ?? formatSegmentLabelDefault;

  // The painting backend. Created once and never swapped mid-mount: the
  // renderer owns cached bitmaps keyed to this canvas.
  const rendererRef = useRef<Renderer | null>(null);
  if (rendererRef.current === null) {
    rendererRef.current = (renderer ?? createCanvas2DRenderer)();
  }

  const selectedIds = initialSelectedIds;
  const [loaded, setLoaded] = useState(false);
  // Terminal base-image load failure — feeds the card-level gate.
  const [error, setError] = useState<string | null>(null);
  // Transient "could not create that segment" message. Kept separate from
  // `error` so a failed draw shows inline on a working board instead of
  // collapsing the whole card into the error state.
  const [drawError, setDrawError] = useState<string | null>(null);
  const [popup, setPopup] = useState<PopupState | null>(null);
  const [masksList, setMasksList] = useState<SegmentMaskData[]>([]);
  // Masks load AFTER the base image decodes, but the card is not revealed
  // until they are all in (see viewer-status.ts) — so `masksReady` now gates
  // the whole card, not just hit-testing.
  const [masksReady, setMasksReady] = useState(false);
  // Terminal-but-recoverable mask failure. Deliberately NOT the base image's
  // `error` (which permanently blanks the board and says "Failed to load
  // images"): this one offers a retry that re-runs PHASE 2 in place.
  const [maskError, setMaskError] = useState<MaskError | null>(null);
  // Bumped by the retry button to re-run the load effect. A separate nonce,
  // never `segments`: the effect's deps must stay independent of the array.
  const [maskRetryNonce, setMaskRetryNonce] = useState(0);
  const [retrying, setRetrying] = useState(false);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [drawMode, setDrawMode] = useState(false);
  const [draftPoints, setDraftPoints] = useState<Point[]>([]);
  const [creatingSegment, setCreatingSegment] = useState(false);
  const frameRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<ViewTransform>({
    scale: 1,
    offsetX: 0,
    offsetY: 0,
  });
  // The latched mode is the single source of truth for the cursor and for what
  // a drag does. Nothing reads live key state, so a missed keyup (macOS
  // suppresses keyup for keys pressed while ⌘ is held) can never strand it.
  const [panMode, setPanMode] = useState<CursorMode>('select');
  // The previous scale, so the mode rule can see a zoom CROSSING of the fit
  // boundary rather than just the current level. Seeded from the initial view
  // so the first effect run is a no-op.
  const prevScaleRef = useRef(view.scale);
  const [panning, setPanning] = useState(false);
  const panStateRef = useRef<{
    startX: number;
    startY: number;
    startOffsetX: number;
    startOffsetY: number;
    moved: boolean;
  } | null>(null);
  // One-shot guard: a pan-drag sets this so the trailing click does not
  // select a segment or place a polygon point.
  const suppressClickRef = useRef(false);

  // Stable signature of just the mask-relevant fields. The load/decode effect
  // below keys off this, not the full `segments` array, so toggling per-segment
  // state (which produces a new `segments` array reference) no longer triggers
  // a full mask reload + re-decode. The signature only changes when the actual
  // set of segments or their mask URLs changes.
  const maskSignature = useMemo(
    () => segments.map((s) => `${s.id}|${s.maskUrl}`).join('\n'),
    [segments],
  );

  // Full teardown of every decoded-mask-derived ref. Called on unmount, and on
  // the `fullReset` path inside the effect body — NOT unconditionally in the
  // effect's cleanup, because cleanup runs before every re-run and would
  // otherwise destroy the very entries the diff path exists to preserve.
  // A new per-mask cache must be added here, not to the effect cleanup.
  const tearDownMaskState = useCallback(() => {
    baseImageRef.current = null;
    masksRef.current = new Map();
    // `dispose()` releases the renderer's cached bitmaps and per-id index
    // caches so the next image rebuilds from scratch against its own masks
    // (different segment ids and dimensions). Documented idempotent.
    rendererRef.current?.dispose();
    prevMaskRefsRef.current = [];
    prevImageKeyRef.current = null;
    lastFailedRefsRef.current = [];
    hasReadyFrameRef.current = false;
  }, []);

  // Unmount teardown, and ONLY unmount: the load effect's own cleanup fires on
  // every re-run, so dropping every mask ref and disposing the renderer lives
  // here instead, where it happens exactly once when the component goes away.
  useEffect(() => () => tearDownMaskState(), [tearDownMaskState]);

  useEffect(() => {
    let cancelled = false;
    // Decide what this run must do BEFORE touching any state: which masks
    // survive, which must be refetched, and which must be thrown away.
    const nextImageKey = buildImageKey({ imageUrl, imageWidth, imageHeight });
    // Read the mask list from the SIGNATURE, not the `segments` prop: the
    // effect must stay provably independent of the array.
    const refs = parseMaskSignature(maskSignature);
    const plan = planMaskDiff({
      prevImageKey: prevImageKeyRef.current,
      nextImageKey,
      prevRefs: prevMaskRefsRef.current,
      nextRefs: refs,
    });

    // The batch this run actually fetches. `plan.load` covers what the SIGNATURE
    // says is new or changed; a retry-nonce bump changes neither the signature
    // nor the image key, so plan.load is empty by design and the ids that failed
    // last time have to be layered back on here — otherwise the retry button
    // would re-run the effect and immediately no-op.
    //   - matched by id against the CURRENT refs, so the retry uses the freshest
    //     maskUrl (a 403 refresh may have replaced it) rather than the stale one;
    //   - a failed id that has since vanished from the signature is dropped, not
    //     force-loaded;
    //   - on fullReset nothing carries over: tearDownMaskState has already
    //     cleared the failed list and plan.load is the whole board anyway.
    const failedStillPresent = plan.fullReset
      ? []
      : lastFailedRefsRef.current.flatMap((failed) => {
          const current = refs.find((r) => r.id === failed.id);
          if (!current) return [];
          return plan.load.some((l) => l.id === current.id) ? [] : [current];
        });
    const refsToLoad = [...plan.load, ...failedStillPresent];

    if (plan.fullReset) {
      // A different image (url OR dimensions), or the very first run. Nothing
      // decoded against the old image may survive: every `coverage` array is
      // sized to the old imageWidth*imageHeight and every renderer index cache
      // is keyed by a segment id that may not even belong to this image.
      tearDownMaskState();
    } else {
      // Diff path — same image, changed signature or a retry nonce bump. Keep
      // the surviving decoded masks AS THE SAME OBJECT REFERENCES; only drop
      // what the plan says to drop. The renderer's per-id caches must be
      // evicted alongside `masksRef`: they are keyed by id alone and are
      // consulted BEFORE `scene.masks`, so a survivor would otherwise paint old
      // geometry against a new mask.
      for (const id of plan.evict) masksRef.current.delete(id);
      rendererRef.current?.evict?.(plan.evict);
    }

    // A retry re-runs PHASE 2 only: the base image for THIS url already
    // decoded, so re-fetching a multi-megabyte photo would be pure waste, and
    // reusing it means the canvas can draw the instant the masks land. Assigned
    // AFTER the branch above: `tearDownMaskState` nulls `baseImageRef`.
    const cache = baseImageCacheRef.current;
    const cachedBase = canReuseBaseImage(cache?.url, imageUrl)
      ? (cache?.img ?? null)
      : null;
    baseImageRef.current = cachedBase;
    prevImageKeyRef.current = nextImageKey;
    prevMaskRefsRef.current = refs;

    // Reset SYNCHRONOUSLY, not inside `load()` after its await. On the
    // fullReset path the previous masks are already gone above, so leaving
    // `masksReady` true even for one microtask would report 'ready' and reveal
    // a board whose masks are discarded. On the diff path the survivors are NOT
    // discarded, but a freshly drawn segment is auto-selected by the host and
    // would highlight blank — so readiness still drops until the new batch
    // settles. (resolveAppliedDelta handles that selected-but-not-yet-loaded
    // window.)
    setMasksReady(false);
    setMaskError(null);
    // The hit-test list must end up holding ALL live masks, so on the diff path
    // it is seeded from the survivors rather than emptied.
    const survivors: SegmentMaskData[] = plan.fullReset
      ? []
      : plan.keep.flatMap((id) => {
          const mask = masksRef.current.get(id);
          return mask ? [{ id, coverage: mask.coverage, area: mask.area }] : [];
        });
    setMasksList(survivors);
    // Same reasoning for the base image: a new url means the old decode is
    // gone, so `loaded` must drop before any await can let a render through.
    if (!cachedBase) {
      setLoaded(false);
      setError(null);
    }

    async function load() {
      try {
        await Promise.resolve();
        if (cancelled) return;

        // PHASE 1 — base image. It decodes first so the canvas has pixels
        // ready, but nothing is revealed until PHASE 2 finishes too: painting
        // at this point is the flash-of-un-masked-image bug.
        if (cachedBase) {
          setLoaded(true);
        } else {
          const base = await loadImage(imageUrl);
          if (cancelled) return;
          baseImageRef.current = base;
          baseImageCacheRef.current = { url: imageUrl, img: base };
          setLoaded(true);
        }

        // PHASE 2 — masks, in the background, independently. Only `refsToLoad`
        // is fetched: the `plan.keep` masks are already decoded and are reused
        // as the same object references. This single check also covers an image
        // with zero segments — nothing to load means the board is ready now.
        if (refsToLoad.length === 0) {
          setMasksReady(true);
          return;
        }

        // The accumulator starts from the survivors so the hit-test list ends
        // up holding ALL live masks, not just the newly decoded ones.
        const hitMasks: SegmentMaskData[] = [...survivors];

        // Load one batch of masks independently, recording each outcome on
        // `tracker`. Returns once every ref in the batch has settled. Never
        // throws: a mask failure is data, not an exception (the catch below
        // belongs to the base image alone).
        async function loadBatch(
          batch: ReadonlyArray<{ id: string; maskUrl: string }>,
          tracker: MaskLoadTracker,
        ): Promise<void> {
          await Promise.all(
            batch.map(async (ref) => {
              try {
                const img = await loadImage(ref.maskUrl);
                if (cancelled) return;
                const mask = buildMaskData(img, imageWidth, imageHeight);
                // Mutate the LIVE map rather than building a fresh one and
                // swapping it in: the survivors must stay in place as the same
                // entries.
                masksRef.current.set(ref.id, mask);
                hitMasks.push({
                  id: ref.id,
                  coverage: mask.coverage,
                  area: mask.area,
                });
                tracker.succeed(ref.id);
                setMasksList([...hitMasks]);
              } catch (err) {
                if (cancelled) return;
                tracker.fail(ref.id, classifyMaskFailure(ref.maskUrl, err));
              }
            }),
          );
        }

        const tracker = new MaskLoadTracker(refsToLoad.map((r) => r.id));
        await loadBatch(refsToLoad, tracker);
        if (cancelled) return;

        // A presigned URL has a 300s TTL, so a page left open can 403 on a mask
        // fetch. Recovery is EXACTLY ONE refetch of the detail DTO for fresh
        // URLs plus one retry of the failed masks — deliberately not a
        // background refresh timer.
        let failures = tracker.failures;
        const decision = planMaskRetry({ failures, alreadyRetried: false });
        const refresh = refreshMaskUrlsRef.current;
        if (decision.shouldRefetch && refresh) {
          try {
            const fresh = await refresh();
            if (cancelled) return;
            const retryRefs = failures
              .map((f) => ({ id: f.id, maskUrl: fresh.get(f.id) }))
              .filter((r): r is { id: string; maskUrl: string } =>
                typeof r.maskUrl === 'string' && r.maskUrl.length > 0,
              );
            if (retryRefs.length > 0) {
              const retryTracker = new MaskLoadTracker(
                retryRefs.map((r) => r.id),
              );
              await loadBatch(retryRefs, retryTracker);
              if (cancelled) return;
              failures = retryTracker.failures;
            }
          } catch {
            if (cancelled) return;
            // Refresh itself failed — keep the original failures and surface
            // them below. No second attempt.
          }
        }

        // Remember what is still broken so the retry button's next run knows
        // what to re-attempt (see `lastFailedRefsRef`). Recorded from
        // `refsToLoad`, the batch actually attempted, and rewritten on every
        // completed run — so a successful retry leaves it empty.
        lastFailedRefsRef.current = refsToLoad.filter((r) =>
          failures.some((f) => f.id === r.id),
        );

        // A partially-masked image is never shown: any surviving failure is a
        // terminal (retryable) error, not a warning painted over the canvas.
        // `total` is the size of the batch actually attempted, not the whole
        // board — on a retry only the previously-failed masks are in flight.
        if (failures.length > 0) {
          setMaskError({ failed: failures.length, total: refsToLoad.length });
        }
        setMasksReady(true);
      } catch (err) {
        // Only the base image can reach here — mask failures are caught above.
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load images');
        }
      }
    }
    void load();
    return () => {
      // ONLY the cancel flag. React runs cleanup before EVERY re-run, not just
      // unmount, so wiping `baseImageRef` / `masksRef` / the renderer here would
      // destroy exactly the decoded entries the diff path exists to preserve —
      // this deletion looks reversible and is not. The full teardown now lives
      // in the `fullReset` branch above (image identity actually changed) and in
      // the unmount effect (component actually going away).
      cancelled = true;
    };
    // Key on the mask signature, not the full `segments` array: the effect only
    // reads `s.id`/`s.maskUrl` (via parseMaskSignature), so it must not re-run
    // when unrelated per-segment fields change. See `maskSignature` above.
    // `maskRetryNonce` is a plain counter bumped only by the retry button — it
    // is NOT derived from `segments`, so it cannot reintroduce the per-segment
    // coupling. This list is UNCHANGED by the incremental diff work and is
    // exhaustive as written; nothing may be added that varies per segment.
    // `prevMaskRefsRef` / `prevImageKeyRef` / `lastFailedRefsRef` carry this
    // effect's cross-run state and are refs precisely so they cannot appear
    // here; `tearDownMaskState` is a stable empty-dep useCallback, likewise
    // deliberately absent.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageUrl, imageWidth, imageHeight, maskSignature, maskRetryNonce]);

  // The single readiness signal the host gates the whole card on. Derived, not
  // stored, so it can never drift from the underlying load state.
  const status = resolveViewerStatus({
    baseLoaded: loaded,
    masksSettled: masksReady,
    baseError: error !== null,
    failedMaskCount: maskError?.failed ?? 0,
  });

  // Push status to the parent. Held in a ref and fired from an effect keyed on
  // `status` alone, so a parent recreating the callback each render neither
  // re-fires it nor (via a dep) re-runs anything in the mask pipeline.
  const onStatusChangeRef = useRef(onStatusChange);
  useEffect(() => {
    onStatusChangeRef.current = onStatusChange;
  }, [onStatusChange]);
  useEffect(() => {
    onStatusChangeRef.current?.(status);
  }, [status]);

  // Retry re-runs PHASE 2 against the cached base image. `retrying` keeps the
  // error state's button in a pending look until the effect flips the status.
  const retryMasks = useCallback(() => {
    setRetrying(true);
    setMaskRetryNonce((n) => n + 1);
  }, []);
  useEffect(() => {
    if (status !== 'loading') setRetrying(false);
  }, [status]);

  // Everything the renderer needs to paint a frame. The component owns state;
  // the renderer owns pixels (dim wash, bright window, outlines, draft polygon).
  const buildScene = useCallback((): Scene | null => {
    const base = baseImageRef.current;
    if (!base) return null;
    return {
      base,
      imageWidth,
      imageHeight,
      masks: masksRef.current,
      selectedIds,
      hoveredId,
      draftPoints,
      devicePixelRatio: window.devicePixelRatio || 1,
    };
  }, [imageWidth, imageHeight, selectedIds, hoveredId, draftPoints]);

  const drawFrame = useCallback(() => {
    const canvas = canvasRef.current;
    const scene = buildScene();
    if (!canvas || !scene) return;
    rendererRef.current?.init(canvas);
    rendererRef.current?.draw(scene);
  }, [buildScene]);

  // Mirrors the render gate below: whenever the <canvas> is actually on
  // screen, drawing is safe. Before the incremental-reload fix this was just
  // `status === 'ready'`, because the canvas was not mounted until then — a
  // draw fired earlier would hit a null ref and never re-fire. Now the canvas
  // can also be mounted on a same-image diff-path reload (hasReadyFrameRef),
  // so the gate widens to match. Keep `canPaint` and the render gate's
  // `showPlaceholder` derived from the same two inputs (status,
  // hasReadyFrameRef) so a future edit cannot drift them apart — a `canPaint`
  // that is true while the placeholder is showing would draw to a null ref.
  const canPaint =
    status === 'ready' || (status !== 'error' && hasReadyFrameRef.current);

  useEffect(() => {
    if (!canPaint) return;
    drawFrame();
    hasReadyFrameRef.current = true;
    // `masksList` is not read directly by `drawFrame` (it reads `masksRef`,
    // a ref), but it gets a new array reference on every successful mask
    // decode — so including it here is what makes a diff-path decode
    // actually trigger a repaint. Without it, `buildScene`'s deps
    // ([imageWidth, imageHeight, selectedIds, hoveredId, draftPoints]) never
    // change when a mask lands, `drawFrame` keeps its identity, and this
    // effect would not re-run: a newly drawn, auto-selected segment would
    // stay unhighlighted indefinitely instead of briefly. Must NOT be added
    // to the mask-load effect's own dep array — that array stays pinned.
  }, [canPaint, drawFrame, masksList]);

  function eventToImagePoint(
    e: React.MouseEvent<HTMLCanvasElement>,
  ): Point | null {
    const frame = frameRef.current;
    if (!frame) return null;
    const rect = frame.getBoundingClientRect();
    return screenToImage(
      e.clientX,
      e.clientY,
      view,
      rect,
      imageWidth,
      imageHeight,
    );
  }

  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      return;
    }
    if (shouldPanOnPress(panMode)) return;
    if (!loaded) return;
    // Masks load after the base image paints. Until they are ready there is
    // nothing to hit-test, so ignore selection clicks rather than silently
    // reporting "no segment here". Drawing mode is unaffected — it does not
    // need masks.
    if (!masksReady && !drawMode) return;
    const wrapper = wrapperRef.current;
    const point = eventToImagePoint(e);
    if (!wrapper || !point) return;
    if (drawMode) {
      setPopup(null);
      setHoveredId(null);
      if (e.detail > 1) return;
      setDraftPoints((current) => [...current, point]);
      return;
    }
    const hits = hitTestAll(point, imageWidth, imageHeight, masksList);
    if (hits.length === 0) {
      setPopup(null);
      setHoveredId(null);
      return;
    }
    const wrapperRect = wrapper.getBoundingClientRect();
    setPopup({
      segmentIds: hits,
      x: e.clientX - wrapperRect.left,
      y: e.clientY - wrapperRect.top,
    });
  }

  function handleCanvasDoubleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!drawMode) return;
    e.preventDefault();
    void finishDrawing();
  }

  function handleMenuAction(segmentId: string, action: SegmentMenuAction) {
    const result = applyMenuAction({ selectedIds }, action, segmentId);
    // Keep the changed-set guard: cheap, and it stops a redundant dispatch from
    // pushing a no-op selection update into the parent.
    if (
      result.selectedIds.size !== selectedIds.size ||
      ![...result.selectedIds].every((id) => selectedIds.has(id))
    ) {
      onSelectionChange(result.selectedIds);
    }
    setPopup(null);
    setHoveredId(null);
  }

  const cancelDrawing = useCallback(() => {
    setDraftPoints([]);
    setDrawMode(false);
    setPopup(null);
    setHoveredId(null);
  }, []);

  async function finishDrawing() {
    if (draftPoints.length < 3 || creatingSegment) return;
    setCreatingSegment(true);
    setDrawError(null);
    try {
      await onCreateSegment(draftPoints);
      setDraftPoints([]);
      setDrawMode(false);
    } catch (err) {
      // Deliberately NOT the load `error`: that one gates the whole card, and a
      // failed polygon submit must not tear down a working board.
      setDrawError(
        err instanceof Error ? err.message : 'Failed to create segment',
      );
    } finally {
      setCreatingSegment(false);
    }
  }

  useEffect(() => {
    if (!drawMode) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') cancelDrawing();
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [cancelDrawing, drawMode]);

  function frameSize(): { w: number; h: number } | null {
    const frame = frameRef.current;
    if (!frame) return null;
    const rect = frame.getBoundingClientRect();
    return { w: rect.width, h: rect.height };
  }

  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    // Primary button only. macOS synthesizes Ctrl+click as a RIGHT click
    // (button 2), and a right press never delivers a matching mouseup here — so
    // starting a pan on it would strand `panning` true and leave the image
    // dragging with no button held.
    if (!shouldStartPanDrag(panMode, e.button)) return;
    e.preventDefault();
    panStateRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startOffsetX: view.offsetX,
      startOffsetY: view.offsetY,
      moved: false,
    };
    setPanning(true);
  }

  // In pan mode ⌘/Ctrl is a mode key, and macOS turns Ctrl+click into a right
  // click — which would pop the browser's context menu over the image. Suppress
  // it there. In select mode a right click still behaves normally.
  function handleContextMenu(e: React.MouseEvent<HTMLCanvasElement>) {
    if (shouldSuppressContextMenu(panMode)) e.preventDefault();
  }

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const pan = panStateRef.current;
    if (!pan) return;
    const dx = e.clientX - pan.startX;
    const dy = e.clientY - pan.startY;
    if (!pan.moved && Math.hypot(dx, dy) > PAN_THRESHOLD) pan.moved = true;
    const size = frameSize();
    if (!size) return;
    setView((v) =>
      clampTransform(
        { ...v, offsetX: pan.startOffsetX + dx, offsetY: pan.startOffsetY + dy },
        size.w,
        size.h,
      ),
    );
  }

  function handleMouseUp() {
    const pan = panStateRef.current;
    panStateRef.current = null;
    setPanning(false);
    // Arm on EVERY pan-initiated press, not only past PAN_THRESHOLD. The browser
    // fires `click` after `mouseup`, and users usually release the modifier
    // before the button — so by click time the guards read false and the segment
    // menu would open. PAN_THRESHOLD still governs only whether the view offset
    // actually moves (see handleMouseMove).
    if (pan) suppressClickRef.current = true;
  }

  function zoomByFactor(factor: number) {
    const size = frameSize();
    if (!size) return;
    setView((v) => zoomAt(v, factor, size.w / 2, size.h / 2, size.w, size.h));
  }

  function resetView() {
    setView({ scale: 1, offsetX: 0, offsetY: 0 });
    // Fit scale is unpannable, so leaving the mode latched to pan would strand
    // the user with a hand cursor over an image that cannot move.
    setPanMode('select');
    // Fit is the "back to neutral" control: it resets the view, the drag latch
    // and draw mode together, discarding any in-progress polygon.
    cancelDrawing();
  }

  const toggleCursorMode = useCallback(() => {
    setPanMode((mode) => {
      const next = mode === 'select' ? 'pan' : 'select';
      // The other direction of mutual exclusion: turning the hand ON while
      // drawing exits draw mode and discards the in-progress polygon.
      if (next === 'pan') cancelDrawing();
      return next;
    });
  }, [cancelDrawing]);

  // Drag mode follows the zoom level, but only when the zoom CROSSES fit:
  // 1 → >1 lights the hand, >1 → 1 turns it off. Non-crossing steps return
  // null and the setter is never called, which is what lets a manual hand
  // toggle survive further zooming at the same level.
  //
  // Deps are [view.scale, drawMode] only — never `segments`. A drawMode flip
  // re-runs this with an unchanged scale; panModeForZoomCrossing returns null
  // on equal scales, so nothing is clobbered.
  useEffect(() => {
    const prevScale = prevScaleRef.current;
    prevScaleRef.current = view.scale;
    const next = panModeForZoomCrossing(prevScale, view.scale, drawMode);
    if (next) setPanMode(next);
  }, [view.scale, drawMode]);

  // Safety net for a pan drag whose mouseup never reaches the canvas — released
  // outside the window, swallowed by a native dialog, or lost to a tab switch.
  // Without this the viewer stays stuck mid-drag and the image follows the
  // pointer with no button held. Only armed while actually panning.
  useEffect(() => {
    if (!panning) return;
    function end() {
      panStateRef.current = null;
      setPanning(false);
    }
    window.addEventListener('mouseup', end);
    window.addEventListener('blur', end);
    return () => {
      window.removeEventListener('mouseup', end);
      window.removeEventListener('blur', end);
    };
  }, [panning]);

  // ⌘ (macOS) / Ctrl latch-toggles the mode on its own, exactly like clicking
  // the toggle button — press and it stays flipped. `e.repeat` is what keeps a
  // held key from flipping the mode over and over at the auto-repeat rate.
  // Matched on `e.key` because e.metaKey/e.ctrlKey are unreliable for the very
  // key being pressed. Skipped while typing, and skipped when another modifier
  // is also down so chords like ⌘⇧P are never hijacked.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      // Fully inert while drawing — no momentary pan, no cursor change, and
      // the in-progress polygon is left alone.
      if (!shouldHonorModifierPan(drawMode)) return;
      if (!isModifierKey(e.key) || e.repeat) return;
      if (e.altKey || e.shiftKey) return;
      const target = e.target as HTMLElement | null;
      if (isTypingTarget(target?.tagName, target?.isContentEditable ?? false)) {
        return;
      }
      toggleCursorMode();
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [drawMode, toggleCursorMode]);

  const rootClassName = ['sv-root', className].filter(Boolean).join(' ');

  // Same box as the real canvas frame, so swapping placeholder → content does
  // not move anything below it. `aspectRatio` + `maxHeight` mirror the frame
  // exactly; without this the card would jump by the height of the image.
  const frameBoxStyle = {
    width: '100%',
    maxWidth: '100%',
    maxHeight,
    aspectRatio: `${imageWidth} / ${imageHeight}`,
  } as const;

  // Nothing of the board is revealed until every mask has drawn on the FIRST
  // load: a base image with no overlay is indistinguishable from
  // "segmentation found nothing". But a same-image incremental reload (e.g.
  // adding one segment) keeps the board up instead: every previously decoded
  // mask is still valid and paintable, and swapping the canvas for a spinner
  // unmounts it — which is what made that flash visible rather than a
  // sub-frame blip. An error ALWAYS takes this branch regardless of
  // `hasReadyFrameRef`: the retry button lives here, and a mask failure is
  // terminal-but-recoverable by design (see viewer-status.ts). Note
  // `resolveViewerStatus` and the reported `status` are UNCHANGED by this —
  // only what this component renders while loading a same-image batch.
  const showPlaceholder =
    status === 'error' || (status !== 'ready' && !hasReadyFrameRef.current);
  if (showPlaceholder) {
    return (
      <div ref={wrapperRef} className={rootClassName}>
        <div
          className="sv-status"
          style={frameBoxStyle}
          role="status"
          aria-live="polite"
          aria-busy={status === 'loading'}
        >
          {status === 'loading' ? (
            <>
              <Loader2Icon className="sv-icon sv-icon-lg sv-spin" />
              <p className="sv-muted">Loading segments…</p>
            </>
          ) : (
            <>
              <AlertTriangleIcon className="sv-icon sv-icon-lg sv-error-icon" />
              <p className="sv-error">
                {maskError ? describeMaskError(maskError) : error}
              </p>
              {/* A base-image failure has no in-place recovery — only the mask
                  phase can be re-run against the cached image. */}
              {maskError && (
                <button
                  type="button"
                  className="sv-btn sv-btn-outline"
                  onClick={retryMasks}
                  disabled={retrying}
                >
                  {retrying ? (
                    <Loader2Icon className="sv-icon sv-spin" />
                  ) : (
                    <RefreshCwIcon className="sv-icon" />
                  )}
                  Retry
                </button>
              )}
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div ref={wrapperRef} className={rootClassName}>
      {drawError && <p className="sv-error sv-error-inline">{drawError}</p>}
      <div className="sv-header">
        {drawMode && (
          <div className="sv-muted">
            {draftPoints.length} polygon point{draftPoints.length === 1 ? '' : 's'} placed
          </div>
        )}
        <div className="sv-actions">
          {drawMode ? (
            <>
              <button
                type="button"
                className="sv-btn"
                onClick={() => void finishDrawing()}
                disabled={draftPoints.length < 3 || creatingSegment}
              >
                {creatingSegment ? (
                  <Loader2Icon className="sv-icon sv-spin" />
                ) : (
                  <CheckIcon className="sv-icon" />
                )}
                Finish
              </button>
              <button
                type="button"
                className="sv-btn sv-btn-outline"
                onClick={cancelDrawing}
                disabled={creatingSegment}
              >
                <XIcon className="sv-icon" />
                Cancel
              </button>
            </>
          ) : (
            <button
              type="button"
              className="sv-btn sv-btn-outline"
              onClick={() => {
                setDrawMode(true);
                // Draw and drag are mutually exclusive: entering draw mode
                // always drops the hand, otherwise pan silently outranks draw
                // in resolveCursor/handleCanvasClick and points cannot be
                // placed at all.
                setPanMode('select');
                setPopup(null);
                setHoveredId(null);
              }}
              disabled={!loaded}
            >
              <CropIcon className="sv-icon" />
              Draw segment
            </button>
          )}
        </div>
      </div>
      <div
        ref={frameRef}
        className="sv-frame"
        // Shared with the loading/error placeholder so the swap is layout-neutral.
        style={frameBoxStyle}
      >
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          onDoubleClick={handleCanvasDoubleClick}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onContextMenu={handleContextMenu}
          style={{
            width: '100%',
            height: '100%',
            transform: `translate(${view.offsetX}px, ${view.offsetY}px) scale(${view.scale})`,
            transformOrigin: '0 0',
            cursor: resolveCursor({
              mode: panMode,
              drawMode,
              loaded,
              panning,
            }),
            display: 'block',
          }}
          className="sv-canvas"
        />
        <div className="sv-toolbar">
          <button
            type="button"
            className="sv-btn sv-btn-icon"
            aria-label="Zoom in"
            onClick={() => zoomByFactor(1.5)}
            disabled={!loaded || view.scale >= MAX_SCALE}
          >
            <PlusIcon className="sv-icon" />
          </button>
          <button
            type="button"
            className="sv-btn sv-btn-icon"
            aria-label="Zoom out"
            onClick={() => zoomByFactor(1 / 1.5)}
            disabled={!loaded || view.scale <= MIN_SCALE}
          >
            <MinusIcon className="sv-icon" />
          </button>
          <button
            type="button"
            className="sv-btn sv-btn-icon"
            aria-label="Reset zoom"
            onClick={resetView}
            disabled={
              !loaded ||
              (view.scale === 1 && view.offsetX === 0 && view.offsetY === 0)
            }
          >
            <MaximizeIcon className="sv-icon" />
          </button>
          {/* The toggle is a mode, not an action — the rule separates it. */}
          <div className="sv-toolbar-sep" aria-hidden />
          <button
            type="button"
            // Latched fill when on, so "pressed" is unmistakable and distinct
            // from hover; quiet otherwise.
            className={
              panMode === 'pan'
                ? 'sv-btn sv-btn-icon sv-btn-latched'
                : 'sv-btn sv-btn-icon'
            }
            // Pressed tracks the LATCHED mode — that is what this button controls.
            aria-pressed={panMode === 'pan'}
            aria-label={
              panMode === 'pan'
                ? 'Pan mode on — switch to select mode (⌘/Ctrl)'
                : 'Select mode on — switch to pan mode (⌘/Ctrl)'
            }
            onClick={toggleCursorMode}
            disabled={!loaded || view.scale === MIN_SCALE}
          >
            {/* Always the hand: this button means "pan", and the highlight —
                not a swapped glyph — is what says whether it is on. A control
                whose icon changes reads as two different buttons. */}
            <HandIcon className="sv-icon" />
          </button>
        </div>
      </div>
      {popup && (
        <Popup
          x={popup.x}
          y={popup.y}
          onClose={() => {
            setPopup(null);
            setHoveredId(null);
          }}
          items={popup.segmentIds.flatMap((segmentId): PopupItem[] => {
            // A hit id always comes from a decoded mask, so it is normally in
            // `segments`; the fallback keeps the menu renderable if a mask
            // outlives its segment across a reload.
            const found = segments.findIndex((s) => s.id === segmentId);
            const segmentIndex = found >= 0 ? found : 0;
            const segment = segments[segmentIndex] ?? {
              id: segmentId,
              maskUrl: '',
            };
            const items = buildSegmentMenuItems({
              selected: selectedIds.has(segmentId),
              excluded: segment.excluded ?? false,
            });
            return items.map((item) => ({
              key: `${segmentId}:${item.action}`,
              label: `${formatLabel(segment, segmentIndex)} - ${item.label}`,
              disabled: item.disabled,
              onFocus: () => setHoveredId(segmentId),
              onPointerEnter: () => setHoveredId(segmentId),
              onPointerLeave: () => setHoveredId(null),
              onSelect: () => handleMenuAction(segmentId, item.action),
            }));
          })}
        />
      )}
    </div>
  );
}
