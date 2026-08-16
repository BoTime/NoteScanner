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
  frameRect: { left: number; top: number; width: number; height: number },
  imageW: number,
  imageH: number,
): { x: number; y: number } {
  // 1. Frame-relative CSS px.
  const fx = screenX - frameRect.left;
  const fy = screenY - frameRect.top;
  // 2. Undo the canvas transform (translate then scale, origin 0,0) -> canvas
  //    CSS px. The canvas is width:100%/height:100% of the frame, so canvas CSS
  //    px == frame CSS px before the transform.
  const cx = (fx - t.offsetX) / t.scale;
  const cy = (fy - t.offsetY) / t.scale;
  // 3. Canvas CSS px -> image px using the frame's unscaled size.
  return {
    x: cx * (imageW / frameRect.width),
    y: cy * (imageH / frameRect.height),
  };
}
