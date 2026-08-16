import {
  applyCountDelta,
  buildBrightWindowImageData,
  buildEdgeCoverage,
  buildOutlineImageData,
  countsToImageData,
  coverageToIndices,
  resolveAppliedDelta,
  type OutlineRgb,
} from '../core';
import type { Renderer, Scene } from './types';

const DIM_COLOR = 'rgba(0, 0, 0, 0.55)';
const DRAFT_COLOR = '#f97316';
const OUTLINE_COLOR: OutlineRgb = { r: 249, g: 115, b: 22 };
// Dilation radius (px at image resolution). The rasterized line is (2*radius+1)
// px wide at image resolution; radius 3 → a 7px line that renders ~3px wide on
// screen after the image is downscaled to fit the viewport.
const OUTLINE_RADIUS = 3;

export function createCanvas2DRenderer(): Renderer {
  let canvas: HTMLCanvasElement | null = null;
  let ctx: CanvasRenderingContext2D | null = null;
  let dim: HTMLCanvasElement | null = null;

  // Cached, incrementally-maintained layers (moved from SegmentViewer refs).
  let brightCounts: Uint16Array | null = null;
  let outlineCounts: Uint16Array | null = null;
  let prevSelected = new Set<string>();
  const covIdxCache = new Map<string, Uint32Array>();
  const edgeIdxCache = new Map<string, Uint32Array>();
  let selBright: ImageBitmap | null = null;
  let selOutline: ImageBitmap | null = null;
  let hoverBright: ImageBitmap | null = null;
  let hoverOutline: ImageBitmap | null = null;
  let hoverKey = '';

  function getCovIndices(scene: Scene, id: string): Uint32Array | null {
    const cached = covIdxCache.get(id);
    if (cached) return cached;
    const cov = scene.masks.get(id)?.coverage;
    if (!cov) return null;
    const idx = coverageToIndices(cov);
    covIdxCache.set(id, idx);
    return idx;
  }

  function getEdgeIndices(scene: Scene, id: string): Uint32Array | null {
    const cached = edgeIdxCache.get(id);
    if (cached) return cached;
    const cov = scene.masks.get(id)?.coverage;
    if (!cov) return null;
    const edge = buildEdgeCoverage(cov, scene.imageWidth, scene.imageHeight, OUTLINE_RADIUS);
    const idx = coverageToIndices(edge);
    edgeIdxCache.set(id, idx);
    return idx;
  }

  /**
   * Incrementally rebuild the committed-selection bright window and outline.
   * Moved verbatim in behavior from SegmentViewer.rebuildSelectedLayers: only
   * added/removed masks are applied as +/-1 deltas to the count buffers, so a
   * single toggle is O(toggled mask), not O(all selected * pixels).
   */
  async function rebuildSelectedLayers(scene: Scene): Promise<void> {
    const px = scene.imageWidth * scene.imageHeight;
    if (!brightCounts || brightCounts.length !== px) {
      brightCounts = new Uint16Array(px);
      outlineCounts = new Uint16Array(px);
      prevSelected = new Set();
    }
    const { added, removed, applied } = resolveAppliedDelta(
      prevSelected,
      scene.selectedIds,
      (id) => scene.masks.has(id),
    );
    if (added.length === 0 && removed.length === 0 && selBright) return;
    for (const id of added) {
      const cov = getCovIndices(scene, id);
      const edge = getEdgeIndices(scene, id);
      if (cov) applyCountDelta(brightCounts, cov, 1);
      if (edge) applyCountDelta(outlineCounts!, edge, 1);
    }
    for (const id of removed) {
      const cov = getCovIndices(scene, id);
      const edge = getEdgeIndices(scene, id);
      if (cov) applyCountDelta(brightCounts, cov, -1);
      if (edge) applyCountDelta(outlineCounts!, edge, -1);
    }
    prevSelected = applied;

    if (scene.selectedIds.size === 0) {
      selBright?.close?.();
      selBright = null;
      selOutline?.close?.();
      selOutline = null;
      paint(scene);
      return;
    }

    const [brightBmp, outlineBmp] = await Promise.all([
      createImageBitmap(countsToImageData(scene.imageWidth, scene.imageHeight, brightCounts, null)),
      createImageBitmap(
        countsToImageData(scene.imageWidth, scene.imageHeight, outlineCounts!, OUTLINE_COLOR),
      ),
    ]);
    selBright?.close?.();
    selBright = brightBmp;
    selOutline?.close?.();
    selOutline = outlineBmp;
    paint(scene);
  }

  /** Hover preview: one mask, solid window + dashed outline. Kept off the
   *  selection caches so hover never recomputes the committed selection. */
  async function rebuildHoverLayers(scene: Scene): Promise<void> {
    const key = scene.hoveredId ?? '';
    if (key === hoverKey) return;
    hoverKey = key;
    if (!scene.hoveredId) {
      hoverBright?.close?.();
      hoverBright = null;
      hoverOutline?.close?.();
      hoverOutline = null;
      paint(scene);
      return;
    }
    const ids = new Set([scene.hoveredId]);
    const [brightBmp, outlineBmp] = await Promise.all([
      createImageBitmap(
        buildBrightWindowImageData(scene.imageWidth, scene.imageHeight, ids, scene.masks),
      ),
      createImageBitmap(
        buildOutlineImageData(
          scene.imageWidth,
          scene.imageHeight,
          ids,
          scene.masks,
          OUTLINE_COLOR,
          OUTLINE_RADIUS,
          true,
        ),
      ),
    ]);
    if (hoverKey !== key) {
      brightBmp.close?.();
      outlineBmp.close?.();
      return;
    }
    hoverBright?.close?.();
    hoverBright = brightBmp;
    hoverOutline?.close?.();
    hoverOutline = outlineBmp;
    paint(scene);
  }

  /** The synchronous blit. Moved from SegmentViewer.drawFrame. */
  function paint(scene: Scene): void {
    if (!canvas || !ctx) return;
    const { imageWidth: w, imageHeight: h, devicePixelRatio: dpr } = scene;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(scene.base, 0, 0, w, h);

    if (!dim || dim.width !== w || dim.height !== h) {
      dim = document.createElement('canvas');
      dim.width = w;
      dim.height = h;
    }
    const dctx = dim.getContext('2d');
    if (dctx) {
      dctx.globalCompositeOperation = 'source-over';
      dctx.clearRect(0, 0, w, h);
      dctx.fillStyle = DIM_COLOR;
      dctx.fillRect(0, 0, w, h);
      const brightBitmap = scene.hoveredId ? hoverBright : selBright;
      if (brightBitmap) {
        dctx.globalCompositeOperation = 'destination-out';
        dctx.drawImage(brightBitmap, 0, 0);
      }
      ctx.drawImage(dim, 0, 0, w, h);
    }

    if (selOutline) ctx.drawImage(selOutline, 0, 0, w, h);
    if (hoverOutline) ctx.drawImage(hoverOutline, 0, 0, w, h);

    if (scene.draftPoints.length > 0) {
      ctx.save();
      ctx.strokeStyle = DRAFT_COLOR;
      ctx.fillStyle = DRAFT_COLOR;
      ctx.lineWidth = Math.max(2, Math.min(w, h) / 300);
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(scene.draftPoints[0].x, scene.draftPoints[0].y);
      for (const p of scene.draftPoints.slice(1)) ctx.lineTo(p.x, p.y);
      if (scene.draftPoints.length >= 3) ctx.closePath();
      ctx.stroke();
      for (const p of scene.draftPoints) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, ctx.lineWidth * 1.8, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
  }

  return {
    init(next) {
      canvas = next;
      ctx = next.getContext('2d');
    },
    draw(scene) {
      void rebuildSelectedLayers(scene);
      void rebuildHoverLayers(scene);
      paint(scene);
    },
    resize(scene) {
      paint(scene);
    },
    dispose() {
      selBright?.close?.();
      selOutline?.close?.();
      hoverBright?.close?.();
      hoverOutline?.close?.();
      selBright = selOutline = hoverBright = hoverOutline = null;
      covIdxCache.clear();
      edgeIdxCache.clear();
      brightCounts = outlineCounts = null;
      prevSelected = new Set();
      hoverKey = '';
      canvas = null;
      ctx = null;
      dim = null;
    },
  };
}
