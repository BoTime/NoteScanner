import type { ViewerSegment } from '../src';

export interface Distribution {
  p50: number;
  p95: number;
  max: number;
  count: number;
}

/** Percentiles, not means: a 60fps mean with periodic 200ms hitches feels
 *  worse than a steady 40fps, and only the tail shows it. */
export function summarize(samples: number[]): Distribution {
  if (samples.length === 0) return { p50: 0, p95: 0, max: 0, count: 0 };
  const sorted = [...samples].sort((a, b) => a - b);
  const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  return { p50: at(0.5), p95: at(0.95), max: sorted[sorted.length - 1], count: sorted.length };
}

function maskDataUrl(w: number, h: number, x0: number, y0: number, bw: number, bh: number): string {
  if (typeof document === 'undefined') {
    return `data:image/png;base64,synthetic-${x0}-${y0}-${bw}-${bh}`;
  }
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#fff';
  ctx.fillRect(x0, y0, bw, bh);
  return canvas.toDataURL('image/png');
}

/** N non-overlapping rectangles on a grid, sized to fill the image. */
export function makeSyntheticSegments(n: number, w: number, h: number): ViewerSegment[] {
  const cols = Math.ceil(Math.sqrt(n));
  const rows = Math.ceil(n / cols);
  const cw = Math.floor(w / cols);
  const ch = Math.floor(h / rows);
  const out: ViewerSegment[] = [];
  for (let i = 0; i < n; i += 1) {
    const cx = (i % cols) * cw;
    const cy = Math.floor(i / cols) * ch;
    out.push({
      id: `seg-${i + 1}`,
      index: i + 1,
      maskUrl: maskDataUrl(w, h, cx + 2, cy + 2, Math.max(1, cw - 4), Math.max(1, ch - 4)),
    });
  }
  return out;
}
