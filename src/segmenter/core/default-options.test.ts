import { describe, expect, it } from 'vitest';

import { DEFAULT_SEGMENTER_OPTIONS } from './types';

/**
 * The two inference knobs whose defaults are an evidence-backed decision rather
 * than an arbitrary starting point. Both are pinned here so a change to either
 * has to be deliberate: the numbers behind them live in
 * `docs/measurements/2026-08-28-decode-sweep.md` and issue #12, and a silent
 * flip would invalidate the published measurements without anyone noticing.
 */
describe('DEFAULT_SEGMENTER_OPTIONS inference knobs', () => {
  it('batches 32 grid points per decode dispatch', () => {
    // Measured twice: -11.8% on `decode` (issue #12) and -4.4% on the whole
    // budget (decode sweep), both at 16 points per side.
    expect(DEFAULT_SEGMENTER_OPTIONS.batchSize).toBe(32);
  });

  it('never defaults to a batchSize that cannot allocate', () => {
    // 64 dies in the upsample with `Array buffer allocation failed`, which
    // scales as batch x width x height x 4. The default must stay under it.
    expect(DEFAULT_SEGMENTER_OPTIONS.batchSize).toBeLessThan(64);
  });

  it('keeps fp32, because fp16 changes what the segmenter detects', () => {
    // fp16 is faster but moved the kept-mask count in BOTH directions across
    // two runs (26 -> 28, and 36 -> 31). Adopting it is a product decision.
    expect(DEFAULT_SEGMENTER_OPTIONS.dtype).toBe('fp32');
  });

  it('leaves both measured-no-op decode paths off', () => {
    // The decode sweep measured `overlapDecodeFilter` as relocating time
    // between stage counters at zero net gain, and `gpuResidentEmbeddings` as
    // flat to slightly negative. Neither earns being on by default.
    expect(DEFAULT_SEGMENTER_OPTIONS.overlapDecodeFilter).toBe(false);
    expect(DEFAULT_SEGMENTER_OPTIONS.gpuResidentEmbeddings).toBe(false);
  });
});
