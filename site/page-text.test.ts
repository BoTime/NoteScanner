import { describe, it, expect } from 'vitest';
import { progressLine, statLine } from './page-text';

describe('progressLine', () => {
  // The worker posts `model-load` only AFTER the download finishes
  // (segmenter.worker.ts:153). So the null window IS the download window, and
  // that is the window this copy has to name.
  it('names the model download while nothing has been reported yet', () => {
    expect(progressLine(null)).toBe('Downloading the model (first run only)…');
  });

  it('reports the model as loaded once model-load arrives', () => {
    expect(progressLine({ phase: 'model-load', done: 1, total: 1, ms: 616 })).toBe(
      'Model ready — segmenting…',
    );
  });

  it('names per-batch phases with their counts', () => {
    expect(progressLine({ phase: 'decode', done: 3, total: 8, ms: 12 })).toBe('decode 3/8');
    expect(progressLine({ phase: 'nms', done: 1, total: 1, ms: 4 })).toBe('nms 1/1');
  });

  it('never labels a per-batch phase as a download', () => {
    for (const phase of ['encode', 'decode', 'filter', 'nms', 'resample', 'mask-encode'] as const) {
      expect(progressLine({ phase, done: 1, total: 2, ms: 1 })).not.toMatch(/download/i);
    }
  });
});

describe('statLine', () => {
  it('renders the count and seconds to one decimal', () => {
    expect(statLine(34, 7067)).toBe('34 segments · 7.1 s');
  });

  it('renders a zero-segment run rather than hiding it', () => {
    expect(statLine(0, 812)).toBe('0 segments · 0.8 s');
  });
});
