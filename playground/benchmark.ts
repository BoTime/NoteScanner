import { summarize, type Distribution } from './fixtures';

export interface PhaseReport {
  n: number;
  decodeMs: Distribution;
  firstPaintMs: number;
  panFrameMs: Distribution;
  zoomFrameMs: Distribution;
  hitTestMs: Distribution;
  selectionRepaintMs: Distribution;
  /** Summed from coverage arrays + bitmaps. NOT performance.memory, which is
   *  Chrome-only and coarse. */
  allocatedBytes: number;
}

export class Benchmark {
  private samples = new Map<string, number[]>();
  private marks = new Map<string, number>();

  start(label: string): void {
    this.marks.set(label, performance.now());
  }

  end(label: string): void {
    const t0 = this.marks.get(label);
    if (t0 === undefined) return;
    this.marks.delete(label);
    const list = this.samples.get(label) ?? [];
    list.push(performance.now() - t0);
    this.samples.set(label, list);
  }

  report(n: number, allocatedBytes: number, firstPaintMs: number): PhaseReport {
    const d = (k: string) => summarize(this.samples.get(k) ?? []);
    return {
      n,
      decodeMs: d('decode'),
      firstPaintMs,
      panFrameMs: d('pan'),
      zoomFrameMs: d('zoom'),
      hitTestMs: d('hitTest'),
      selectionRepaintMs: d('selection'),
      allocatedBytes,
    };
  }

  reset(): void {
    this.samples.clear();
    this.marks.clear();
  }
}

/** The N values the spec calls for. The shape of the curve across these
 *  matters more than any absolute number. */
export const BENCH_SIZES = [10, 40, 100] as const;
