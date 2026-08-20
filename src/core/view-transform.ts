/**
 * Pure view-transform math for the SegmentViewer zoom/pan feature. No React, no
 * DOM — every function takes plain numbers so it can be unit-tested in the node
 * Vitest environment.
 *
 * The transform is applied as a CSS `transform` on the canvas with
 * `transform-origin: 0 0`:
 *   transform: translate(offsetX px, offsetY px) scale(scale)
 * `scale` is relative to the fit-to-frame size (scale 1 == the whole image fills
 * the frame). Offsets are in frame CSS pixels.
 */
export interface ViewTransform {
  /** 1 = fit-to-frame, up to MAX_SCALE. */
  scale: number;
  /** CSS px translate applied before scaling (transform-origin 0 0). */
  offsetX: number;
  offsetY: number;
}

export const MIN_SCALE = 1;
export const MAX_SCALE = 5;
/** Movement (frame px) past which a ⌘/Ctrl drag counts as a pan, not a click. */
export const PAN_THRESHOLD = 4;

export const IDENTITY: ViewTransform = { scale: 1, offsetX: 0, offsetY: 0 };

export function clampScale(scale: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));
}

export function clampTransform(
  t: ViewTransform,
  frameW: number,
  frameH: number,
): ViewTransform {
  const scale = clampScale(t.scale);
  if (scale === MIN_SCALE) {
    return { scale, offsetX: 0, offsetY: 0 };
  }
  // With transform-origin 0,0 the scaled canvas is frameW*scale × frameH*scale.
  // To keep it covering the frame: frameDim - scaledDim <= offset <= 0.
  const minOffsetX = frameW - frameW * scale; // <= 0
  const minOffsetY = frameH - frameH * scale;
  const offsetX = Math.min(0, Math.max(minOffsetX, t.offsetX));
  const offsetY = Math.min(0, Math.max(minOffsetY, t.offsetY));
  return { scale, offsetX, offsetY };
}

export function zoomAt(
  t: ViewTransform,
  factor: number,
  anchorX: number,
  anchorY: number,
  frameW: number,
  frameH: number,
): ViewTransform {
  const newScale = clampScale(t.scale * factor);
  // Image point currently under the anchor (frame-relative CSS px):
  //   imagePoint = (anchor - offset) / scale
  // Keep it under the anchor after zoom:
  //   newOffset = anchor - imagePoint * newScale
  const imageX = (anchorX - t.offsetX) / t.scale;
  const imageY = (anchorY - t.offsetY) / t.scale;
  const offsetX = anchorX - imageX * newScale;
  const offsetY = anchorY - imageY * newScale;
  return clampTransform({ scale: newScale, offsetX, offsetY }, frameW, frameH);
}

export function screenToImage(
  screenX: number,
  screenY: number,
  t: ViewTransform,
  canvasRect: { left: number; top: number; width: number; height: number },
  imageW: number,
  imageH: number,
): { x: number; y: number } {
  // `canvasRect` is the <canvas>'s own UNTRANSFORMED layout box in screen
  // coordinates, NOT the frame's box.
  //
  // Why not the frame: the canvas sizes itself with the "replaced element
  // contain" pattern (max-width/max-height/width:auto/height:auto +
  // aspect-ratio), so whenever the frame's aspect ratio differs from the
  // image's, the canvas is strictly smaller than the frame on one axis. Canvas
  // CSS px != frame CSS px in general, and the boxes' origins can differ too,
  // so measuring the frame here maps clicks to the wrong image pixels.
  //
  // Why untransformed: the steps below undo the canvas's CSS transform
  // analytically. `getBoundingClientRect()` on the canvas reports the box with
  // that transform ALREADY applied, so feeding it in raw would double-count
  // both the translate and the scale. Callers must un-apply the transform when
  // measuring (see `canvasLayoutRect` in SegmentViewer.tsx).
  //
  // 1. Canvas-relative CSS px.
  const fx = screenX - canvasRect.left;
  const fy = screenY - canvasRect.top;
  // 2. Undo the canvas transform (translate then scale, origin 0,0) -> canvas
  //    CSS px.
  const cx = (fx - t.offsetX) / t.scale;
  const cy = (fy - t.offsetY) / t.scale;
  // 3. Canvas CSS px -> image px using the canvas's unscaled size.
  return {
    x: cx * (imageW / canvasRect.width),
    y: cy * (imageH / canvasRect.height),
  };
}
