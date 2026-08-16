export interface SegmentMaskData {
  id: string;
  /** Per-pixel coverage at the rendered (image-space) resolution. 0 = transparent, >0 = covered. */
  coverage: Uint8Array;
  area: number;
}

/**
 * Find every segment id covering the given image-space point, ordered from the
 * smallest mask to the largest mask.
 */
export function hitTestAll(
  point: { x: number; y: number },
  width: number,
  height: number,
  masks: SegmentMaskData[],
): string[] {
  const x = Math.floor(point.x);
  const y = Math.floor(point.y);
  if (x < 0 || y < 0 || x >= width || y >= height) return [];
  const offset = y * width + x;
  return masks
    .filter((mask) => mask.coverage[offset] > 0)
    .sort((a, b) => a.area - b.area)
    .map((mask) => mask.id);
}

/**
 * Find the topmost segment id covering the given image-space point. Topmost = the
 * smallest mask among hits, biasing toward the most specific segment when masks overlap.
 */
export function hitTest(
  point: { x: number; y: number },
  width: number,
  height: number,
  masks: SegmentMaskData[],
): string | null {
  return hitTestAll(point, width, height, masks)[0] ?? null;
}

/**
 * Build the "bright window" alpha mask: the union of every highlighted segment's
 * coverage. A pixel's alpha is 255 when any bright segment covers it, otherwise 0.
 *
 * This is the union/selection-wins rule: deselected masks are simply never
 * visited, so they can never subtract from or carve a hole in a bright segment.
 * Missing ids (no loaded mask) are skipped. RGB is left at 0 — the draw path
 * relies solely on alpha (it uses the bitmap as a `destination-out` stencil).
 */
export function buildBrightWindowImageData(
  width: number,
  height: number,
  brightIds: ReadonlySet<string>,
  masks: ReadonlyMap<string, { coverage: Uint8Array }>,
): ImageData {
  const img = new ImageData(width, height);
  const data = img.data;
  for (const id of brightIds) {
    const cov = masks.get(id)?.coverage;
    if (!cov) continue;
    for (let p = 0; p < cov.length; p += 1) {
      if (cov[p]) data[p * 4 + 3] = 255;
    }
  }
  return img;
}

/**
 * Identity key for a cached bright window. Two rebuilds may share a cached
 * bitmap only when this key matches.
 *
 * The key folds in WHICH bright masks are actually loaded, not just the bright
 * id set: a rebuild that fires before a mask has loaded (e.g. mid-reload after
 * a new segment is drawn) builds an empty highlight, and must NOT share a key
 * with the later rebuild that runs once the mask is loaded — otherwise the
 * memo suppresses the real rebuild and the highlight never appears.
 */
export function brightWindowKey(
  width: number,
  height: number,
  brightIds: ReadonlySet<string>,
  masks: ReadonlyMap<string, unknown>,
): string {
  const sortedBright = [...brightIds].sort();
  const loadedBright = sortedBright.filter((id) => masks.has(id));
  return `${width}x${height}|sel:${sortedBright.join(',')}|got:${loadedBright.join(',')}`;
}

/**
 * Derive an outline ("edge") bitmap from a coverage array.
 *
 * A pixel is an edge when it is covered AND at least one of its 4 orthogonal
 * neighbors is uncovered OR lies outside the image (so a mask touching the image
 * border still gets an outline along that border). This naturally traces the
 * outer perimeter, interior holes, and every disconnected blob.
 *
 * `radius` dilates the raw edge set by that many pixels (8-connected) so the
 * rendered line is visibly thick rather than a single-pixel hairline. radius 0
 * returns the raw 1px edge.
 *
 * Returns a new Uint8Array (1 = edge, 0 = not), same length as `coverage`.
 */
export function buildEdgeCoverage(
  coverage: Uint8Array,
  width: number,
  height: number,
  radius: number,
): Uint8Array {
  const raw = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = y * width + x;
      if (!coverage[p]) continue;
      const up = y > 0 ? coverage[p - width] : 0;
      const down = y < height - 1 ? coverage[p + width] : 0;
      const left = x > 0 ? coverage[p - 1] : 0;
      const right = x < width - 1 ? coverage[p + 1] : 0;
      if (!up || !down || !left || !right) raw[p] = 1;
    }
  }
  if (radius <= 0) return raw;

  const dilated = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!raw[y * width + x]) continue;
      for (let dy = -radius; dy <= radius; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          dilated[ny * width + nx] = 1;
        }
      }
    }
  }
  return dilated;
}

/** RGB triple for an outline color (0-255 per channel). */
export interface OutlineRgb {
  r: number;
  g: number;
  b: number;
}

/**
 * Rasterize the union of the edge outlines of `ids` into an ImageData, painting
 * each edge pixel `color` at full alpha. RGB channels carry the color (unlike
 * the bright window, which uses alpha only) because this layer is blitted as a
 * normal `source-over` image, not a stencil.
 *
 * `radius` thickens the line (see buildEdgeCoverage). When `dashed` is true,
 * edge pixels with odd `(x + y)` parity are dropped, producing a stipple that
 * reads as a dashed outline — used to distinguish a hovered (preview) segment
 * from committed selections. Missing ids are skipped.
 */
export function buildOutlineImageData(
  width: number,
  height: number,
  ids: ReadonlySet<string>,
  masks: ReadonlyMap<string, { coverage: Uint8Array }>,
  color: OutlineRgb,
  radius: number,
  dashed: boolean,
): ImageData {
  const img = new ImageData(width, height);
  const data = img.data;
  for (const id of ids) {
    const cov = masks.get(id)?.coverage;
    if (!cov) continue;
    const edge = buildEdgeCoverage(cov, width, height, radius);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const p = y * width + x;
        if (!edge[p]) continue;
        if (dashed && ((x + y) & 1) === 1) continue;
        const o = p * 4;
        data[o] = color.r;
        data[o + 1] = color.g;
        data[o + 2] = color.b;
        data[o + 3] = 255;
      }
    }
  }
  return img;
}

/**
 * Identity key for a cached outline bitmap. Like brightWindowKey, it folds in
 * which ids are actually loaded so a rebuild that fires before a mask loads does
 * not share a cache slot with the later rebuild once it loads. `tag`
 * distinguishes the solid-selected layer from the dashed-hover layer.
 */
export function outlineKey(
  width: number,
  height: number,
  ids: ReadonlySet<string>,
  masks: ReadonlyMap<string, unknown>,
  tag: string,
): string {
  const sorted = [...ids].sort();
  const loaded = sorted.filter((id) => masks.has(id));
  return `${tag}|${width}x${height}|ids:${sorted.join(',')}|got:${loaded.join(',')}`;
}

/**
 * Flatten a coverage/edge bitmap into the ascending list of set-pixel indices.
 *
 * The incremental accumulators (see applyCountDelta) add or subtract a single
 * mask's contribution per toggle. Iterating only that mask's set pixels — rather
 * than the whole width*height grid — is what makes a toggle O(mask) instead of
 * O(image). Compute this once per mask and cache it; re-toggling is then free.
 */
export function coverageToIndices(coverage: Uint8Array): Uint32Array {
  let count = 0;
  for (let p = 0; p < coverage.length; p += 1) if (coverage[p]) count += 1;
  const out = new Uint32Array(count);
  let i = 0;
  for (let p = 0; p < coverage.length; p += 1) {
    if (coverage[p]) {
      out[i] = p;
      i += 1;
    }
  }
  return out;
}

/** A tight crop rectangle in image-space pixels. */
export interface MaskBounds {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * Compute the bounding box of a coverage array, padded by `padding` px and
 * clamped to the image. Returns null when nothing is covered.
 *
 * NOTE: this is a plain min/max box over every covered pixel, so a stray speck
 * far from the main blob inflates it. Run the coverage through
 * {@link largestConnectedComponent} first to drop such specks — the preview does
 * this so a flying pixel can't inflate the crop (see PostItPreview).
 */
export function coverageToBounds(
  coverage: Uint8Array,
  width: number,
  height: number,
  padding: number,
): MaskBounds | null {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (!coverage[y * width + x]) continue;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }
  if (maxX < 0 || maxY < 0) return null;
  const left = Math.max(0, minX - padding);
  const top = Math.max(0, minY - padding);
  const right = Math.min(width - 1, maxX + padding);
  const bottom = Math.min(height - 1, maxY + padding);
  return { left, top, width: right - left + 1, height: bottom - top + 1 };
}

/**
 * Return a new coverage array containing only the largest 8-connected component
 * of `coverage`; all smaller blobs (stray "flying pixel" mask noise) are zeroed.
 * Ties break toward whichever component is found first in row-major order. An
 * all-empty input returns an all-empty copy.
 *
 * Mirrors the server-side filter in apps/api's mask-largest-component.ts so the
 * preview crop and the OCR crop drop the same noise. 8-connectivity keeps
 * diagonal strokes together; the flood fill is iterative (explicit stack) to
 * handle full-resolution masks without overflowing the call stack.
 */
export function largestConnectedComponent(
  coverage: Uint8Array,
  width: number,
  height: number,
): Uint8Array {
  const visited = new Uint8Array(coverage.length);
  const stack: number[] = [];
  let bestArea = 0;
  let bestPixels: number[] = [];

  for (let start = 0; start < coverage.length; start += 1) {
    if (!coverage[start] || visited[start]) continue;
    const pixels: number[] = [];
    stack.push(start);
    visited[start] = 1;
    while (stack.length > 0) {
      const cur = stack.pop()!;
      pixels.push(cur);
      const cx = cur % width;
      const cy = (cur - cx) / width;
      for (let dy = -1; dy <= 1; dy += 1) {
        const ny = cy + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -1; dx <= 1; dx += 1) {
          if (dx === 0 && dy === 0) continue;
          const nx = cx + dx;
          if (nx < 0 || nx >= width) continue;
          const np = ny * width + nx;
          if (coverage[np] && !visited[np]) {
            visited[np] = 1;
            stack.push(np);
          }
        }
      }
    }
    if (pixels.length > bestArea) {
      bestArea = pixels.length;
      bestPixels = pixels;
    }
  }

  const result = new Uint8Array(coverage.length);
  for (const p of bestPixels) result[p] = 1;
  return result;
}

/** The ids added to and removed from a selection between two snapshots. */
export interface SelectionDiff {
  added: string[];
  removed: string[];
}

/**
 * Diff two selection sets. The viewer applies `added` as +1 deltas and `removed`
 * as -1 deltas to the per-pixel count buffers, so a single toggle touches one
 * mask while a bulk change (initial load, transcribe resync) touches only the
 * masks that actually changed — one code path for both.
 */
export function diffSelection(
  prev: ReadonlySet<string>,
  next: ReadonlySet<string>,
): SelectionDiff {
  const added: string[] = [];
  const removed: string[] = [];
  for (const id of next) if (!prev.has(id)) added.push(id);
  for (const id of prev) if (!next.has(id)) removed.push(id);
  return { added, removed };
}

/** The applied delta plus the snapshot to record as the new "previous" set. */
export interface AppliedDelta {
  added: string[];
  removed: string[];
  applied: Set<string>;
}

/**
 * Resolve which selection deltas can actually be applied to the count buffers
 * right now, given that a mask may not be decoded yet.
 *
 * A rebuild can fire during the reload window (a new manual segment changes the
 * `segments` prop, which empties the decoded-mask map while the masks re-decode
 * asynchronously). If we recorded a not-yet-loaded id as "applied", the rebuild
 * that runs once the mask finishes loading would see an empty diff and skip —
 * leaving the highlight blank. So:
 *
 * - `added` includes only ids whose mask `isLoaded` returns true (others are
 *   deferred to a later rebuild).
 * - `removed` is always processed (a deselected mask need not be loaded).
 * - `applied` — the set to store as the next "previous" snapshot — is `prev`
 *   minus `removed` plus the *loaded* additions. Deferred (unloaded) additions
 *   are deliberately left out so the next rebuild still sees them as added.
 *
 * This restores the load-state safeguard the old `brightWindowKey`/`outlineKey`
 * `got:` segment provided before the incremental rewrite.
 */
export function resolveAppliedDelta(
  prev: ReadonlySet<string>,
  next: ReadonlySet<string>,
  isLoaded: (id: string) => boolean,
): AppliedDelta {
  const { added: rawAdded, removed } = diffSelection(prev, next);
  const added = rawAdded.filter((id) => isLoaded(id));
  const applied = new Set(prev);
  for (const id of removed) applied.delete(id);
  for (const id of added) applied.add(id);
  return { added, removed, applied };
}

/**
 * Add `sign` (+1 to select, -1 to deselect) to `counts` at each pixel index in
 * `indices`. `counts[p]` is the number of currently-selected masks contributing
 * to pixel p; a pixel renders iff its count > 0. Counting (rather than a boolean
 * union) keeps overlap correct: deselecting one mask cannot erase a pixel that
 * another selected mask still contributes (selection wins on overlap).
 */
export function applyCountDelta(
  counts: Uint16Array,
  indices: Uint32Array,
  sign: number,
): void {
  for (let i = 0; i < indices.length; i += 1) {
    counts[indices[i]] += sign;
  }
}

/**
 * Render a per-pixel count buffer to an ImageData. A pixel is painted iff its
 * count > 0. With `color` null the output is an alpha-only stencil (alpha 255,
 * RGB 0) for the dim-overlay `destination-out` blit; with an OutlineRgb the
 * output carries that color at full alpha for a `source-over` outline blit.
 */
export function countsToImageData(
  width: number,
  height: number,
  counts: Uint16Array,
  color: OutlineRgb | null,
): ImageData {
  const img = new ImageData(width, height);
  const data = img.data;
  const r = color ? color.r : 0;
  const g = color ? color.g : 0;
  const b = color ? color.b : 0;
  for (let p = 0; p < counts.length; p += 1) {
    if (counts[p] <= 0) continue;
    const o = p * 4;
    data[o] = r;
    data[o + 1] = g;
    data[o + 2] = b;
    data[o + 3] = 255;
  }
  return img;
}

export function toggleSelection(
  set: ReadonlySet<string>,
  id: string,
): Set<string> {
  const next = new Set(set);
  if (next.has(id)) {
    next.delete(id);
  } else {
    next.add(id);
  }
  return next;
}
