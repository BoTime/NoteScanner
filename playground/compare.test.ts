import { describe, it, expect, vi } from 'vitest';
import {
  CHOSEN_BATCH_SIZE_CHOICES,
  COMPARE_PAIRS,
  COMPARE_ROWS,
  DEFAULT_CHOSEN_BATCH_SIZE,
  IOU_MATCH_FLOOR,
  compareMaskSets,
  pairAgreements,
  rowOptions,
  runRow,
  toMarkdown,
  type CompareResult,
} from './compare';
import { createTimingAccumulator, type RawMask, type SegmentationResult } from '../src/segmenter';

/** A `SegmentationResult` with fixed per-phase totals, so timings are exact. */
function stubResult(
  totals: Partial<Record<'encode' | 'decode' | 'filter' | 'nms' | 'mask-encode', number>>,
  rawMasks?: RawMask[],
): SegmentationResult {
  const accumulator = createTimingAccumulator();
  for (const [phase, total] of Object.entries(totals)) {
    accumulator.record(phase as 'encode', total as number);
  }
  return {
    segments: [],
    timings: accumulator.report(1000),
    counts: { raw: 30, afterFilter: 12, afterNms: rawMasks?.length ?? 5 },
    ...(rawMasks ? { rawMasks } : {}),
  };
}

/** A square mask of `size` at (x0, y0) on a `w * h` grid. */
function boxMask(w: number, h: number, x0: number, y0: number, size: number): RawMask {
  const coverage = new Uint8Array(w * h);
  let area = 0;
  for (let y = y0; y < y0 + size; y += 1) {
    for (let x = x0; x < x0 + size; x += 1) {
      coverage[y * w + x] = 1;
      area += 1;
    }
  }
  return { coverage, area };
}

describe('COMPARE_ROWS', () => {
  it('is the seven-row matrix the criteria need', () => {
    expect(COMPARE_ROWS).toHaveLength(7);
    expect(COMPARE_ROWS.map((row) => row.id)).toEqual([
      'row-1', 'row-2', 'row-3', 'row-4', 'row-5', 'row-6', 'row-7',
    ]);
  });

  it('covers both dtypes at both 16 and 32 points per side (AC1, AC2)', () => {
    const axes = COMPARE_ROWS.map((row) => `${row.dtype}@${row.pointsPerSide}`);
    for (const axis of ['fp32@16', 'fp16@16', 'fp32@32', 'fp16@32']) {
      expect(axes).toContain(axis);
    }
  });

  it('keeps raw masks for exactly the four paired dtype rows (AC2, AC5)', () => {
    const keeping = COMPARE_ROWS.filter((row) => row.keepRawMasks).map((row) => row.id);
    expect(keeping).toEqual(['row-1', 'row-2', 'row-3', 'row-4']);
  });

  it('sweeps batchSize at a fixed dtype and points per side (AC3)', () => {
    const sweep = COMPARE_ROWS.filter((row) => row.id === 'row-5' || row.id === 'row-6');
    expect(sweep.map((row) => row.batchSize)).toEqual([32, 64]);
    expect(new Set(sweep.map((row) => `${row.dtype}@${row.pointsPerSide}`))).toEqual(
      new Set(['fp16@16']),
    );
  });

  it('pairs each fp16 row with its fp32 baseline at the same density', () => {
    expect(COMPARE_PAIRS).toHaveLength(2);
    for (const pair of COMPARE_PAIRS) {
      const baseline = COMPARE_ROWS.find((row) => row.id === pair.baselineRowId)!;
      const variant = COMPARE_ROWS.find((row) => row.id === pair.variantRowId)!;
      expect(baseline.dtype).toBe('fp32');
      expect(variant.dtype).toBe('fp16');
      expect(baseline.pointsPerSide).toBe(variant.pointsPerSide);
    }
  });
});

describe('rowOptions', () => {
  it('leaves a fixed row alone whatever batchSize the caller chose', () => {
    const row = COMPARE_ROWS.find((r) => r.id === 'row-5')!;
    expect(rowOptions(row, 64).batchSize).toBe(32);
  });

  it('gives the confirmation row the batchSize the developer supplied (AC3)', () => {
    const row = COMPARE_ROWS.find((r) => r.id === 'row-7')!;
    expect(row.batchSize).toBeNull();
    expect(rowOptions(row, 64).batchSize).toBe(64);
    expect(rowOptions(row).batchSize).toBe(DEFAULT_CHOSEN_BATCH_SIZE);
    expect(CHOSEN_BATCH_SIZE_CHOICES).toEqual([32, 64]);
  });
});

describe('runRow', () => {
  it('reports the encode + decode subtotal beside the per-phase totals (AC1)', async () => {
    const segment = vi.fn(async () =>
      stubResult({ encode: 300, decode: 8700, filter: 60000, nms: 20000, 'mask-encode': 9000 }),
    );
    const row = COMPARE_ROWS.find((r) => r.id === 'row-5')!;

    const result = await runRow(row, { createBitmap: async () => ({}) as ImageBitmap, segment });

    expect(result.rowId).toBe('row-5');
    expect(result.phases.encode).toBe(300);
    expect(result.phases.decode).toBe(8700);
    expect(result.encodeDecodeMs).toBe(9000);
    expect(result.totalMs).toBe(1000);
    expect(result.counts.afterNms).toBe(5);
  });

  it('asks for raw masks on a pair row and keeps what comes back', async () => {
    const masks = [boxMask(4, 4, 0, 0, 2)];
    const segment = vi.fn(async () => stubResult({ encode: 1, decode: 2 }, masks));
    const row = COMPARE_ROWS.find((r) => r.id === 'row-1')!;

    const result = await runRow(row, { createBitmap: async () => ({}) as ImageBitmap, segment });

    expect(segment.mock.calls[0][1]).toMatchObject({
      dtype: 'fp32', pointsPerSide: 16, batchSize: 8, keepRawMasks: true,
    });
    expect(result.rawMasks).toBe(masks);
  });

  it('does not ask for raw masks on a batchSize row — batchSize cannot move the output', async () => {
    const segment = vi.fn(async () => stubResult({ encode: 1, decode: 2 }));
    const row = COMPARE_ROWS.find((r) => r.id === 'row-6')!;

    const result = await runRow(row, { createBitmap: async () => ({}) as ImageBitmap, segment });

    expect(segment.mock.calls[0][1]).toMatchObject({ keepRawMasks: false, batchSize: 64 });
    expect('rawMasks' in result).toBe(false);
  });

  it('mints a fresh bitmap per row, because the worker consumes it', async () => {
    const createBitmap = vi.fn(async () => ({}) as ImageBitmap);
    const segment = vi.fn(async () => stubResult({ encode: 1 }));
    await runRow(COMPARE_ROWS[0], { createBitmap, segment });
    await runRow(COMPARE_ROWS[0], { createBitmap, segment });
    expect(createBitmap).toHaveBeenCalledTimes(2);
  });
});

describe('compareMaskSets', () => {
  it('matches an identical set at IoU 1 (AC2)', () => {
    const masks = [boxMask(8, 8, 0, 0, 3), boxMask(8, 8, 4, 4, 3)];
    const agreement = compareMaskSets(masks, masks);
    expect(agreement).toMatchObject({
      baselineCount: 2, variantCount: 2, matched: 2,
      unmatchedBaseline: 0, unmatchedVariant: 0,
      meanIou: 1, medianIou: 1, minIou: 1,
    });
  });

  it('leaves a baseline mask unmatched when the variant dropped it', () => {
    const baseline = [boxMask(8, 8, 0, 0, 3), boxMask(8, 8, 4, 4, 3)];
    const agreement = compareMaskSets(baseline, [baseline[0]]);
    expect(agreement.matched).toBe(1);
    expect(agreement.unmatchedBaseline).toBe(1);
    expect(agreement.unmatchedVariant).toBe(0);
  });

  it('counts an extra variant mask as unmatched on the variant side', () => {
    const baseline = [boxMask(8, 8, 0, 0, 3)];
    const agreement = compareMaskSets(baseline, [baseline[0], boxMask(8, 8, 4, 4, 3)]);
    expect(agreement.matched).toBe(1);
    expect(agreement.unmatchedVariant).toBe(1);
  });

  it('scores a shifted mask below 1 and drops it below the floor', () => {
    const baseline = [boxMask(8, 8, 0, 0, 3)];
    const shifted = [boxMask(8, 8, 1, 0, 3)];
    const loose = compareMaskSets(baseline, shifted, 0.1);
    expect(loose.matched).toBe(1);
    expect(loose.minIou).toBeGreaterThan(0);
    expect(loose.minIou).toBeLessThan(1);

    // Below the floor a pair is unmatched on BOTH sides, not a bad match.
    const strict = compareMaskSets(baseline, shifted, IOU_MATCH_FLOOR);
    expect(strict).toMatchObject({ matched: 0, unmatchedBaseline: 1, unmatchedVariant: 1 });
  });

  it('does not divide by zero on empty sets', () => {
    expect(compareMaskSets([], [])).toMatchObject({
      baselineCount: 0, variantCount: 0, matched: 0, meanIou: 0, medianIou: 0, minIou: 0,
    });
  });

  it('gives each baseline mask its best unclaimed partner', () => {
    // Two baseline boxes; the variant lists the far one first, so a
    // positional zip would mispair them and score ~0.
    const near = boxMask(16, 16, 0, 0, 4);
    const far = boxMask(16, 16, 10, 10, 4);
    const agreement = compareMaskSets([near, far], [far, near]);
    expect(agreement.matched).toBe(2);
    expect(agreement.minIou).toBe(1);
  });
});

describe('pairAgreements and toMarkdown', () => {
  function resultFor(rowId: string, rawMasks?: RawMask[]): CompareResult {
    const row = COMPARE_ROWS.find((r) => r.id === rowId)!;
    return {
      rowId,
      options: rowOptions(row),
      phases: { 'model-load': 500, encode: 300, decode: 8700, filter: 60000, nms: 20000, 'mask-encode': 9000 },
      encodeDecodeMs: 9000,
      totalMs: 98400,
      counts: { raw: 30, afterFilter: 12, afterNms: rawMasks?.length ?? 5 },
      ...(rawMasks ? { rawMasks } : {}),
    };
  }

  it('reports a pair only once both halves have masks', () => {
    const masks = [boxMask(8, 8, 0, 0, 3)];
    expect(pairAgreements({ 'row-1': resultFor('row-1', masks) })).toHaveLength(0);
    const both = pairAgreements({
      'row-1': resultFor('row-1', masks),
      'row-2': resultFor('row-2', masks),
    });
    expect(both).toHaveLength(1);
    expect(both[0]).toMatchObject({ pairId: 'pair-16', agreement: { matched: 1 } });
  });

  it('renders a markdown table plus the agreement summary', () => {
    const masks = [boxMask(8, 8, 0, 0, 3)];
    const results = { 'row-1': resultFor('row-1', masks), 'row-2': resultFor('row-2', masks) };
    const markdown = toMarkdown(Object.values(results), pairAgreements(results));

    const lines = markdown.split('\n');
    expect(lines[0]).toContain('encode + decode');
    expect(lines[1]).toMatch(/^\|[\s|:-]+\|$/);
    expect(markdown).toContain('| row-1 | fp32 | 16 | 8 |');
    expect(markdown).toContain('fp16 vs fp32');
    expect(markdown).toContain('pair-16');
  });
});
