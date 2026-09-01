import { describe, it, expect } from 'vitest';
import {
  DECODE_PATHS,
  DEFAULT_SWEEP_CONFIG,
  IOU_MATCH_FLOOR,
  OVERLAP_CAVEAT,
  budgetMsOf,
  compareMaskSets,
  decodeFlags,
  decodePathOf,
  expandGrid,
  fastestRecord,
  rankRecords,
  resolveConfig,
  rowLabel,
  toMarkdown,
  warmUpRow,
  type RowOptions,
  type RunRecord,
} from './compare';
import { createTimingAccumulator, type RawMask } from '../src/segmenter';

function rowOptions(overrides: Partial<RowOptions> = {}): RowOptions {
  return {
    dtype: 'fp32',
    batchSize: 8,
    pointsPerSide: 16,
    overlapDecodeFilter: false,
    gpuResidentEmbeddings: false,
    keepRawMasks: false,
    lowResFilterNms: true,
    ...overrides,
  };
}

/** An ok record whose budget is exactly `budget` ms. */
function okRecord(rowId: string, budget: number, overrides: Partial<RowOptions> = {}): RunRecord {
  const accumulator = createTimingAccumulator();
  accumulator.record('model-load', 500);
  accumulator.record('encode', 100);
  accumulator.record('decode', 200);
  accumulator.record('filter', 300);
  accumulator.record('nms', 40);
  accumulator.record('mask-encode', 60);
  const timings = accumulator.report(budget + 500);
  return {
    rowId,
    options: rowOptions(overrides),
    status: 'ok',
    timings,
    counts: { raw: 96, afterFilter: 31, afterNms: 12, returned: 11 },
    budgetMs: budgetMsOf(timings),
  };
}

function failedRecord(rowId: string, overrides: Partial<RowOptions> = {}): RunRecord {
  return {
    rowId,
    options: rowOptions(overrides),
    status: 'failed',
    phase: 'filter',
    message: 'Array buffer allocation failed',
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

describe('decode paths', () => {
  it('maps each named path onto its two flags, and back', () => {
    expect(DECODE_PATHS).toEqual(['none', 'overlap', 'gpuEmbeddings', 'both']);
    for (const path of DECODE_PATHS) {
      expect(decodePathOf(decodeFlags(path))).toBe(path);
    }
    expect(decodeFlags('both')).toEqual({
      overlapDecodeFilter: true,
      gpuResidentEmbeddings: true,
    });
  });
});

describe('expandGrid', () => {
  it('is exactly the 16-row default grid, with no batchSize 64 (AC6)', () => {
    const rows = expandGrid(DEFAULT_SWEEP_CONFIG);
    expect(rows).toHaveLength(16);
    expect(rows.map((row) => row.options.batchSize)).not.toContain(64);
    expect(new Set(rows.map((row) => row.options.pointsPerSide))).toEqual(new Set([16]));
    expect(new Set(rows.map((row) => row.options.dtype))).toEqual(new Set(['fp32', 'fp16']));
    expect(new Set(rows.map((row) => decodePathOf(row.options)))).toEqual(
      new Set(DECODE_PATHS),
    );
  });

  it('expands in a deterministic order with stable, unique ids (AC6)', () => {
    const first = expandGrid(DEFAULT_SWEEP_CONFIG).map((row) => row.id);
    const second = expandGrid(DEFAULT_SWEEP_CONFIG).map((row) => row.id);
    expect(second).toEqual(first);
    expect(new Set(first).size).toBe(first.length);
    expect(first[0]).toBe('p16-fp32-b8-none-lowres-r1');
    expect(first[15]).toBe('p16-fp16-b32-both-lowres-r1');
  });

  it('widens to 32 rows under --full and multiplies by --reps, without editing code (AC6)', () => {
    expect(expandGrid(resolveConfig({}, { full: true }))).toHaveLength(32);
    expect(expandGrid(resolveConfig({}, { reps: 3 }))).toHaveLength(48);
  });

  it('takes batchSize 64 from an explicit config, so the failure stays reachable (AC6)', () => {
    const rows = expandGrid(
      resolveConfig({ decodePaths: ['none'], batchSizes: [64], dtypes: ['fp16'] }),
    );
    expect(rows).toHaveLength(1);
    expect(rows[0].options.batchSize).toBe(64);
  });

  it('rejects a config that would silently expand to nothing or to garbage', () => {
    expect(() => resolveConfig({ batchSizes: [] })).toThrow(/batchSizes/);
    expect(() => resolveConfig({}, { reps: 0 })).toThrow(/reps/);
    expect(() => resolveConfig({ decodePaths: ['turbo' as never] })).toThrow(/decodePaths/);
  });

  it('keeps the default grid at 16 rows, all measuring the shipped pipeline (AC12)', () => {
    const rows = expandGrid(DEFAULT_SWEEP_CONFIG);
    expect(rows).toHaveLength(16);
    expect(rows.every((row) => row.options.lowResFilterNms)).toBe(true);
  });

  it('walks a before/after pair from a config, adjacently (AC12)', () => {
    const rows = expandGrid(
      resolveConfig({
        decodePaths: ['none'],
        batchSizes: [8],
        dtypes: ['fp32'],
        lowResFilterNms: [false, true],
      }),
    );
    expect(rows.map((row) => row.id)).toEqual([
      'p16-fp32-b8-none-fullres-r1',
      'p16-fp32-b8-none-lowres-r1',
    ]);
    // The baseline runs FIRST, so the Compare tab's retained mask set is the
    // pre-change one and every later row is compared against it.
    expect(rows[0].options.lowResFilterNms).toBe(false);
  });

  it('rejects a lowResFilterNms axis that is empty or not boolean', () => {
    expect(() => resolveConfig({ lowResFilterNms: [] })).toThrow(/lowResFilterNms/);
    expect(() => resolveConfig({ lowResFilterNms: ['yes' as never] })).toThrow(/lowResFilterNms/);
  });
});

describe('warmUpRow', () => {
  it('clones the first row under a reserved id so its numbers can be discarded (AC12)', () => {
    const rows = expandGrid(DEFAULT_SWEEP_CONFIG);
    const warm = warmUpRow(rows);
    expect(warm.warmUp).toBe(true);
    expect(warm.id).toBe('warm-up');
    expect(warm.options).toEqual(rows[0].options);
    expect(rows.some((row) => row.warmUp)).toBe(false);
  });
});

describe('rankRecords / fastestRecord', () => {
  it('ranks by budgetMs ascending regardless of input order (AC8)', () => {
    const ranked = rankRecords([okRecord('slow', 900), okRecord('fast', 100)]);
    expect(ranked.map((r) => r.rowId)).toEqual(['fast', 'slow']);
    expect(fastestRecord([okRecord('slow', 900), okRecord('fast', 100)])!.rowId).toBe('fast');
  });

  it('breaks a budget tie by grid order, so the ranking is deterministic', () => {
    const ranked = rankRecords([okRecord('a', 100), okRecord('b', 100)]);
    expect(ranked.map((r) => r.rowId)).toEqual(['a', 'b']);
  });

  it('sinks every failed row below every ok row, however fast the ok rows are (AC8)', () => {
    const ranked = rankRecords([failedRecord('boom'), okRecord('slow', 9999)]);
    expect(ranked.map((r) => r.rowId)).toEqual(['slow', 'boom']);
    expect(fastestRecord([failedRecord('boom')])).toBeNull();
  });
});

describe('toMarkdown', () => {
  const meta = {
    config: DEFAULT_SWEEP_CONFIG,
    adapter: {
      vendor: 'apple',
      architecture: 'metal-3',
      device: '',
      description: 'Apple M3 Pro',
      software: false,
    },
    generatedAt: '2026-08-28T12:00:00.000Z',
  };

  it('renders options, every phase, counts and budget for each row (AC8)', () => {
    const md = toMarkdown([okRecord('p16-fp16-b32-both-r1', 700, {
      dtype: 'fp16', batchSize: 32, overlapDecodeFilter: true, gpuResidentEmbeddings: true,
    })], meta);
    for (const column of [
      'budget', 'total', 'model-load', 'encode', 'decode', 'filter', 'nms',
      'resample', 'mask-encode', 'raw', 'afterFilter', 'afterNms', 'returned', 'status',
    ]) {
      expect(md).toContain(column);
    }
    expect(md).toContain('p16-fp16-b32-both-r1');
    expect(md).toContain('| both | fp16 | 32 | 16 | lowres |');
    // counts.raw / afterFilter / afterNms / returned, in that order.
    expect(md).toContain('| 96 | 31 | 12 | 11 |');
    expect(md).toContain('700');
  });

  it('names the fastest configuration explicitly (AC8)', () => {
    const md = toMarkdown([okRecord('slow', 900), okRecord('fast', 100, { dtype: 'fp16' })], meta);
    expect(md).toContain(`**Fastest: ${rowLabel(okRecord('fast', 100, { dtype: 'fp16' }))}`);
  });

  it('carries the overlap caveat above the table, not below it (AC8)', () => {
    const md = toMarkdown([okRecord('a', 100)], meta);
    expect(md).toContain(OVERLAP_CAVEAT);
    expect(md.indexOf(OVERLAP_CAVEAT)).toBeLessThan(md.indexOf('| rank |'));
  });

  it('renders a failed row as failed rather than as fast (AC8)', () => {
    const md = toMarkdown([failedRecord('boom'), okRecord('a', 900)], meta);
    expect(md).toContain('failed in filter: Array buffer allocation failed');
    // The failed row shows no timing numbers to be mistaken for a fast result.
    const failedLine = md.split('\n').find((line) => line.includes('boom'))!;
    expect(failedLine).not.toMatch(/\d+\.\d/);
  });

  it('records the adapter and the resolved grid, so the run is reproducible', () => {
    const md = toMarkdown([okRecord('a', 100)], meta);
    expect(md).toContain('Apple M3 Pro');
    expect(md).toContain(JSON.stringify(DEFAULT_SWEEP_CONFIG));
  });

  it('says so plainly when nothing completed', () => {
    expect(toMarkdown([failedRecord('boom')], meta)).toContain('No row completed');
  });

  it('aligns the table from the column names, not a hardcoded index', () => {
    const md = toMarkdown([okRecord('a', 100)], meta);
    const lines = md.split('\n');
    const header = lines.find((line) => line.startsWith('| rank |'))!;
    const alignment = lines[lines.indexOf(header) + 1];
    const cells = (row: string) => row.split('|').slice(1, -1).length;
    expect(cells(alignment)).toBe(cells(header));
    // `status` is the last column and is left-aligned; a positional rule
    // silently misaligns it the moment a column is added.
    expect(alignment.trim().endsWith('--- |')).toBe(true);
  });
});

describe('compareMaskSets', () => {
  it('matches identical sets at IoU 1', () => {
    const masks = [boxMask(8, 8, 0, 0, 3), boxMask(8, 8, 4, 4, 2)];
    const agreement = compareMaskSets(masks, masks);
    expect(agreement.matched).toBe(2);
    expect(agreement.unmatchedBaseline).toBe(0);
    expect(agreement.unmatchedVariant).toBe(0);
    expect(agreement.meanIou).toBe(1);
    expect(agreement.minIou).toBe(1);
    expect(agreement.floor).toBe(IOU_MATCH_FLOOR);
  });

  it('counts a below-floor pairing as unmatched on BOTH sides', () => {
    const baseline = [boxMask(8, 8, 0, 0, 4)];
    const variant = [boxMask(8, 8, 0, 0, 2)];
    const agreement = compareMaskSets(baseline, variant);
    expect(agreement.matched).toBe(0);
    expect(agreement.unmatchedBaseline).toBe(1);
    expect(agreement.unmatchedVariant).toBe(1);
  });

  it('takes the LOWER median on an even match count', () => {
    // Four matched pairs whose IoUs are distinct: the median must be a value
    // some pair actually scored, not an averaged midpoint between two.
    const baseline = [
      boxMask(20, 20, 0, 0, 10),
      boxMask(20, 20, 0, 0, 10),
      boxMask(20, 20, 0, 0, 10),
      boxMask(20, 20, 0, 0, 10),
    ];
    const variant = baseline.map((mask) => ({ ...mask }));
    const agreement = compareMaskSets(baseline, variant);
    expect(agreement.matched).toBe(4);
    expect(agreement.medianIou).toBe(agreement.minIou);
  });

  it('is empty-safe on both sides', () => {
    const agreement = compareMaskSets([], []);
    expect(agreement).toMatchObject({ matched: 0, meanIou: 0, medianIou: 0, minIou: 0 });
  });
});
