import type { SegmenterProgress } from '../src/segmenter';

/**
 * One line of honest progress.
 *
 * The `model-load` phase is reported by the worker only once the model is
 * already loaded — the worker posts it with the elapsed time attached, after
 * the fact. Nothing at all is posted while the weights are downloading. So the
 * interval where `progress` is still `null` is exactly the interval a
 * first-time visitor spends waiting on that download, and naming it is the
 * whole point: an undifferentiated spinner during the longest wait of the visit
 * is indistinguishable from a hung tab.
 *
 * On a warm second run that window is brief; "(first run only)" is what keeps
 * the line honest in both cases.
 */
export function progressLine(progress: SegmenterProgress | null): string {
  if (!progress) return 'Downloading the model (first run only)…';
  if (progress.phase === 'model-load') return 'Model ready — segmenting…';
  return `${progress.phase} ${progress.done}/${progress.total}`;
}

/** The post-run line: how many segments came back, and how long it took. */
export function statLine(segmentCount: number, totalMs: number): string {
  return `${segmentCount} segments · ${(totalMs / 1000).toFixed(1)} s`;
}
