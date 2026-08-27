# Segmenter fp16 / batchSize Measurement Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the instrument that measures fp16-vs-fp32 and `batchSize` for the WebGPU segmenter — a `keepRawMasks` opt-in on the package, a pure playground compare module, and a Compare tab — without changing any segmenter default.

**Architecture:** One small, documented addition to the published `/segmenter` surface (`keepRawMasks` in, `rawMasks` out) so an fp32 baseline run and its fp16 counterpart can be compared mask-for-mask. Everything else is playground-local: `playground/compare.ts` holds the seven-row preset matrix, a `runRow` that takes its `segment` function by injection (so vitest drives it with a stub and never touches WebGPU), a greedy best-IoU `compareMaskSets` built on the existing `pairwiseIoU`, and a `toMarkdown` renderer; `playground/CompareView.tsx` is a thin view over that module.

**Tech Stack:** TypeScript, React 19, Vite (playground only), vitest (node environment), `@huggingface/transformers` (optional peer, worker-side only — untouched here).

**Spec:** `docs/superpowers/specs/2026-08-27-segmenter-fp16-batchsize-measurement-design.md`

## Global Constraints

- **`DEFAULT_SEGMENTER_OPTIONS` must not change.** `dtype` stays `'fp32'`, `batchSize` stays `8`. Choosing either now would be choosing on the issue's estimate, which AC3 rules out. (AC4)
- **No change to the worker.** `src/segmenter/worker/segmenter.worker.ts` is not edited by any task. `keepRawMasks` is a client-side retention flag, not an inference parameter: the worker always posts masks, and the flag must NOT enter the worker's session cache key (today `` `${options.modelId}|${options.dtype}` ``, worker line ~68).
- **No new dependencies.** No `@playwright/test`, no browser-driven verification, no second IoU implementation — import `pairwiseIoU` from `src/segmenter/core`.
- **Nothing beyond `keepRawMasks` is added to the published `/segmenter` subpath.** The compare matrix, the IoU pairing and the markdown renderer are playground-local, mirroring how `playground/benchmark.ts` sits beside `BenchmarkView.tsx`.
- **`tsconfig.json` excludes `playground/`**, so `npm run typecheck` does NOT typecheck playground files. Do not "fix" this by editing `tsconfig.json` — it is out of scope. Playground types must be correct by hand; `playground/compare.ts` is still exercised by vitest (`vitest.config.ts` includes `playground/**/*.test.ts`).
- **vitest runs in the `node` environment** and only picks up `playground/**/*.test.ts` (not `.tsx`). No React component test for the playground exists or is added.
- Verification for every task: `npm run typecheck && npm test`.
- Every acceptance criterion is `(non-ui)`: nothing here is browser-observable by the pipeline. The numbers come later, from a human driving a GPU browser session.

---

### Task 1: `keepRawMasks` opt-in and `rawMasks` on the result

**Files:**
- Modify: `src/segmenter/core/types.ts` (`SegmenterOptions`, `DEFAULT_SEGMENTER_OPTIONS`, `SegmentationResult`)
- Modify: `src/segmenter/createSegmenter.ts` (the `done` branch that resolves the promise, ~line 138)
- Test: `src/segmenter/createSegmenter.test.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `SegmenterOptions.keepRawMasks: boolean` — defaults to `false` in `DEFAULT_SEGMENTER_OPTIONS`.
  - `SegmentationResult.rawMasks?: RawMask[]` — present only when the run requested `keepRawMasks: true`.
  - `RawMask` already exists in `types.ts` as `{ coverage: Uint8Array; area: number }` and is already exported from `src/segmenter`.

- [ ] **Step 1: Write the failing tests**

In `src/segmenter/createSegmenter.test.ts`, first extract the two `vi.stubGlobal` calls currently inlined at the top of the `it('encodes each surviving mask into a ViewerSegment', ...)` test into a module-level helper placed just below `doneMessage`, and call `stubMaskEncoder()` as that test's first line instead:

```ts
/** Minimal `ImageData` / `OffscreenCanvas` so `encodeMaskPng` runs under node. */
function stubMaskEncoder() {
  vi.stubGlobal(
    'ImageData',
    class {
      constructor(
        readonly data: Uint8ClampedArray,
        readonly width: number,
        readonly height: number,
      ) {}
    },
  );
  vi.stubGlobal(
    'OffscreenCanvas',
    class {
      getContext() {
        return { putImageData: () => {} };
      }
      convertToBlob() {
        return Promise.resolve({
          arrayBuffer: async () => Uint8Array.from([104, 105]).buffer,
        });
      }
    },
  );
}
```

Then add these three tests. The first two go inside the existing `describe('createSegmenter', ...)`; the third is a new top-level `describe`:

```ts
  it('retains the worker masks when keepRawMasks is requested', async () => {
    stubMaskEncoder();
    const segmenter = createSegmenter({ createWorker: spawn });
    const pending = segmenter.segment(fakeBitmap(), { keepRawMasks: true });
    FakeWorker.instances[0].emit({
      type: 'done',
      masks: [{ coverage: Uint8Array.from([0, 1]), area: 1 }],
      width: 2,
      height: 1,
      timings: createTimingAccumulator().report(1),
      counts: { raw: 3, afterFilter: 1, afterNms: 1 },
    });

    const result = await pending;
    expect(result.rawMasks).toEqual([{ coverage: Uint8Array.from([0, 1]), area: 1 }]);
    // The segments are still produced — retention is additive, not a mode.
    expect(result.segments).toHaveLength(1);
  });

  it('omits rawMasks entirely when keepRawMasks is not requested', async () => {
    stubMaskEncoder();
    const segmenter = createSegmenter({ createWorker: spawn });
    const pending = segmenter.segment(fakeBitmap());
    FakeWorker.instances[0].emit({
      type: 'done',
      masks: [{ coverage: Uint8Array.from([0, 1]), area: 1 }],
      width: 2,
      height: 1,
      timings: createTimingAccumulator().report(1),
      counts: { raw: 3, afterFilter: 1, afterNms: 1 },
    });

    const result = await pending;
    // Absent, not undefined-valued: the coverage buffers must stay collectable.
    expect('rawMasks' in result).toBe(false);
  });
```

```ts
describe('DEFAULT_SEGMENTER_OPTIONS', () => {
  // A regression guard, not a tautology: issue #3 measures fp16 and a larger
  // batchSize, and the recommendation must come from that measurement rather
  // than from flipping these on the estimate.
  it('still ships fp32 at batchSize 8, with raw-mask retention off', () => {
    expect(DEFAULT_SEGMENTER_OPTIONS.dtype).toBe('fp32');
    expect(DEFAULT_SEGMENTER_OPTIONS.batchSize).toBe(8);
    expect(DEFAULT_SEGMENTER_OPTIONS.keepRawMasks).toBe(false);
  });
});
```

Add `DEFAULT_SEGMENTER_OPTIONS` to the existing `from './core'` import block at the top of the test file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/segmenter/createSegmenter.test.ts`
Expected: FAIL — TypeScript/runtime errors on the unknown `keepRawMasks` option, `result.rawMasks` undefined, and `DEFAULT_SEGMENTER_OPTIONS.keepRawMasks` undefined.

- [ ] **Step 3: Add the option and the result field**

In `src/segmenter/core/types.ts`, add to `SegmenterOptions` (after `dtype`):

```ts
  /**
   * Retain the worker's full-resolution `RawMask[]` on the result.
   *
   * A CLIENT-SIDE RETENTION FLAG, not an inference parameter — every other
   * member of this bag is a model knob, this one is not. The worker ignores
   * it (it always posts masks) and it must never enter the worker's session
   * cache key, which is keyed on `modelId` and `dtype` alone.
   *
   * Off by default because a caller holding the result in React state would
   * pin tens of megabytes: at 16 points per side that is ~50 full-resolution
   * coverage arrays at ~0.7 MB each, for the lifetime of the view. Turn it on
   * only to compare one run's masks against another's.
   */
  keepRawMasks: boolean;
```

In `DEFAULT_SEGMENTER_OPTIONS`, add `keepRawMasks: false,` after `dtype: 'fp32',`. Change nothing else in that object.

In `SegmentationResult`:

```ts
export interface SegmentationResult {
  segments: ViewerSegment[];
  timings: TimingReport;
  counts: SegmentationCounts;
  /** Present only when the run asked for `keepRawMasks`. */
  rawMasks?: RawMask[];
}
```

- [ ] **Step 4: Carry the masks through `createSegmenter`**

In `src/segmenter/createSegmenter.ts`, in the `resolve({ ... })` call inside `onMessage`, add the field after `counts: message.counts,`:

```ts
              counts: message.counts,
              // Spread so the key is ABSENT rather than undefined when the
              // caller did not ask: the buffers must stay collectable for the
              // common case, which is every run the Segment view makes.
              ...(resolved.keepRawMasks ? { rawMasks: message.masks } : {}),
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `npm run typecheck && npm test`
Expected: PASS — the whole suite, including the three new tests.

- [ ] **Step 6: Commit**

```bash
git add src/segmenter/core/types.ts src/segmenter/createSegmenter.ts src/segmenter/createSegmenter.test.ts
git commit -m "feat(segmenter): opt-in raw mask retention via keepRawMasks"
```

---

### Task 2: `playground/compare.ts` — the preset matrix, row runner, IoU pairing and markdown

**Files:**
- Create: `playground/compare.ts`
- Test: `playground/compare.test.ts`

**Interfaces:**
- Consumes: `SegmenterOptions.keepRawMasks` and `SegmentationResult.rawMasks` from Task 1; `pairwiseIoU`, `PHASE_ORDER`, `RawMask`, `SegmentationCounts`, `SegmentationPhase`, `SegmenterOptions`, `SegmentationResult`, `SegmenterProgress` from `../src/segmenter`.
- Produces, all consumed by Task 3:
  - `COMPARE_ROWS: readonly CompareRow[]` (7 entries), `COMPARE_PAIRS: readonly ComparePair[]` (2 entries)
  - `CHOSEN_BATCH_SIZE_CHOICES: readonly [32, 64]`, `DEFAULT_CHOSEN_BATCH_SIZE = 32`, `IOU_MATCH_FLOOR = 0.9`
  - `rowOptions(row: CompareRow, chosenBatchSize?: number): RowOptions`
  - `runRow(row: CompareRow, deps: RunRowDeps): Promise<CompareResult>`
  - `compareMaskSets(baseline: readonly RawMask[], variant: readonly RawMask[], floor?: number): MaskAgreement`
  - `pairAgreements(results: Readonly<Record<string, CompareResult>>): PairAgreement[]`
  - `toMarkdown(results: readonly CompareResult[], agreements: readonly PairAgreement[]): string`

- [ ] **Step 1: Write the failing tests**

Create `playground/compare.test.ts`:

```ts
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run playground/compare.test.ts`
Expected: FAIL — `Failed to resolve import "./compare"`.

- [ ] **Step 3: Write `playground/compare.ts`**

```ts
import {
  PHASE_ORDER,
  pairwiseIoU,
  type RawMask,
  type SegmentationCounts,
  type SegmentationPhase,
  type SegmentationResult,
  type SegmenterOptions,
  type SegmenterProgress,
} from '../src/segmenter';

/**
 * The measurement matrix for issue #3 (E1 + D1: fp16 and a larger batchSize).
 *
 * Playground-local on purpose: nothing here belongs in the published
 * `/segmenter` subpath, exactly as `benchmark.ts` sits beside its view.
 *
 * Seven rows, not an exhaustive sweep. About 90% of every run is `filter`,
 * `nms` and `mask-encode` — stages this issue does not touch — so a wide
 * sweep would buy minutes of wall clock and no extra signal. Four dtype rows
 * answer AC1/AC2; three further rows sketch the batchSize curve for AC3.
 */

/** The subset of `SegmenterOptions` a compare row pins. */
export interface RowOptions {
  dtype: SegmenterOptions['dtype'];
  pointsPerSide: number;
  batchSize: number;
  keepRawMasks: boolean;
}

export interface CompareRow {
  id: string;
  dtype: SegmenterOptions['dtype'];
  pointsPerSide: number;
  /** `null` on the confirmation row: the developer supplies it from rows 5-6. */
  batchSize: number | null;
  /**
   * Only the four dtype rows retain masks. batchSize cannot change the output
   * — it changes how many prompt points ride in each model call — so keeping
   * ~50 full-resolution coverage arrays for rows 5-7 would pin tens of
   * megabytes for nothing.
   */
  keepRawMasks: boolean;
  /** Which acceptance criterion this row exists for, shown in the table. */
  serves: string;
}

export const COMPARE_ROWS: readonly CompareRow[] = [
  { id: 'row-1', dtype: 'fp32', pointsPerSide: 16, batchSize: 8, keepRawMasks: true, serves: 'AC1/AC2 baseline' },
  { id: 'row-2', dtype: 'fp16', pointsPerSide: 16, batchSize: 8, keepRawMasks: true, serves: 'AC1/AC2' },
  { id: 'row-3', dtype: 'fp32', pointsPerSide: 32, batchSize: 8, keepRawMasks: true, serves: 'AC1/AC2 baseline' },
  { id: 'row-4', dtype: 'fp16', pointsPerSide: 32, batchSize: 8, keepRawMasks: true, serves: 'AC1/AC2' },
  { id: 'row-5', dtype: 'fp16', pointsPerSide: 16, batchSize: 32, keepRawMasks: false, serves: 'AC3 curve' },
  { id: 'row-6', dtype: 'fp16', pointsPerSide: 16, batchSize: 64, keepRawMasks: false, serves: 'AC3 curve' },
  { id: 'row-7', dtype: 'fp16', pointsPerSide: 32, batchSize: null, keepRawMasks: false, serves: 'AC3 confirmation' },
];

export interface ComparePair {
  id: string;
  baselineRowId: string;
  variantRowId: string;
  label: string;
}

/** The fp32 baseline / fp16 variant pairs AC2 compares, at each density. */
export const COMPARE_PAIRS: readonly ComparePair[] = [
  { id: 'pair-16', baselineRowId: 'row-1', variantRowId: 'row-2', label: '16 points per side' },
  { id: 'pair-32', baselineRowId: 'row-3', variantRowId: 'row-4', label: '32 points per side' },
];

export const CHOSEN_BATCH_SIZE_CHOICES = [32, 64] as const;
export const DEFAULT_CHOSEN_BATCH_SIZE = 32;

/**
 * The IoU at or above which two masks are the same mask. Below it a pair is
 * NOT a bad match — it is unmatched on both sides, because a low-IoU pairing
 * says the two runs found different things, not the same thing badly.
 */
export const IOU_MATCH_FLOOR = 0.9;

export function rowOptions(
  row: CompareRow,
  chosenBatchSize: number = DEFAULT_CHOSEN_BATCH_SIZE,
): RowOptions {
  return {
    dtype: row.dtype,
    pointsPerSide: row.pointsPerSide,
    batchSize: row.batchSize ?? chosenBatchSize,
    keepRawMasks: row.keepRawMasks,
  };
}

export interface CompareResult {
  rowId: string;
  options: RowOptions;
  /** Per-phase wall-clock totals, in ms. */
  phases: Record<SegmentationPhase, number>;
  /** The subtotal E1 and D1 actually move. */
  encodeDecodeMs: number;
  totalMs: number;
  counts: SegmentationCounts;
  rawMasks?: RawMask[];
}

export interface RunRowDeps {
  /**
   * A FRESH bitmap per row. The bitmap is transferred to the worker and
   * consumed, so it cannot be reused across rows.
   */
  createBitmap: () => Promise<ImageBitmap>;
  /** Injected so tests drive this with a stub and never touch WebGPU. */
  segment: (
    image: ImageBitmap,
    options: RowOptions,
    onProgress?: (event: SegmenterProgress) => void,
  ) => Promise<SegmentationResult>;
  chosenBatchSize?: number;
  onProgress?: (event: SegmenterProgress) => void;
}

export async function runRow(row: CompareRow, deps: RunRowDeps): Promise<CompareResult> {
  const options = rowOptions(row, deps.chosenBatchSize);
  const bitmap = await deps.createBitmap();
  const result = await deps.segment(bitmap, options, deps.onProgress);

  const phases = {} as Record<SegmentationPhase, number>;
  for (const phase of PHASE_ORDER) phases[phase] = result.timings.phases[phase].total;

  return {
    rowId: row.id,
    options,
    phases,
    encodeDecodeMs: phases.encode + phases.decode,
    totalMs: result.timings.totalMs,
    counts: result.counts,
    ...(result.rawMasks ? { rawMasks: result.rawMasks } : {}),
  };
}

export interface MaskAgreement {
  baselineCount: number;
  variantCount: number;
  matched: number;
  unmatchedBaseline: number;
  unmatchedVariant: number;
  /** Over the matched pairs only; 0 when nothing matched. */
  meanIou: number;
  medianIou: number;
  minIou: number;
  floor: number;
}

/**
 * Greedy best-IoU pairing between two mask sets.
 *
 * Each baseline mask takes its highest-IoU unclaimed partner in the variant
 * set. Greedy rather than optimal (Hungarian) on purpose: the sets are ~50
 * masks of a near-identical scene, so ties that would separate the two
 * algorithms do not arise, and an optimal assignment would be a second
 * algorithm to trust.
 *
 * IoU comes from `pairwiseIoU` in `src/segmenter/core` — the one definition.
 * `RawMask` is structurally a `BinaryMask`, so it passes straight in.
 */
export function compareMaskSets(
  baseline: readonly RawMask[],
  variant: readonly RawMask[],
  floor: number = IOU_MATCH_FLOOR,
): MaskAgreement {
  const claimed = new Set<number>();
  const matchedIous: number[] = [];

  for (const mask of baseline) {
    let bestIndex = -1;
    let bestIou = 0;
    for (let i = 0; i < variant.length; i += 1) {
      if (claimed.has(i)) continue;
      const iou = pairwiseIoU(mask, variant[i]);
      if (iou > bestIou) {
        bestIou = iou;
        bestIndex = i;
      }
    }
    if (bestIndex >= 0 && bestIou >= floor) {
      claimed.add(bestIndex);
      matchedIous.push(bestIou);
    }
  }

  const sorted = [...matchedIous].sort((a, b) => a - b);
  const matched = sorted.length;

  return {
    baselineCount: baseline.length,
    variantCount: variant.length,
    matched,
    unmatchedBaseline: baseline.length - matched,
    unmatchedVariant: variant.length - matched,
    meanIou: matched === 0 ? 0 : sorted.reduce((sum, iou) => sum + iou, 0) / matched,
    medianIou: matched === 0 ? 0 : sorted[Math.floor((matched - 1) / 2)],
    minIou: matched === 0 ? 0 : sorted[0],
    floor,
  };
}

export interface PairAgreement {
  pairId: string;
  label: string;
  baselineRowId: string;
  variantRowId: string;
  agreement: MaskAgreement;
}

/** Agreement for every pair whose BOTH halves have run and kept their masks. */
export function pairAgreements(
  results: Readonly<Record<string, CompareResult>>,
): PairAgreement[] {
  const out: PairAgreement[] = [];
  for (const pair of COMPARE_PAIRS) {
    const baseline = results[pair.baselineRowId]?.rawMasks;
    const variant = results[pair.variantRowId]?.rawMasks;
    if (!baseline || !variant) continue;
    out.push({
      pairId: pair.id,
      label: pair.label,
      baselineRowId: pair.baselineRowId,
      variantRowId: pair.variantRowId,
      agreement: compareMaskSets(baseline, variant),
    });
  }
  return out;
}

function ms(value: number): string {
  return value.toFixed(0);
}

/** Results as a markdown table to paste into issue #3, agreement included. */
export function toMarkdown(
  results: readonly CompareResult[],
  agreements: readonly PairAgreement[],
): string {
  const header =
    '| row | dtype | pps | batch | encode | decode | encode + decode | filter | nms | mask-encode | total | kept | model-load |';
  const divider = '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |';
  const body = results.map((result) => {
    const p = result.phases;
    return [
      result.rowId,
      result.options.dtype,
      String(result.options.pointsPerSide),
      String(result.options.batchSize),
      ms(p.encode),
      ms(p.decode),
      ms(result.encodeDecodeMs),
      ms(p.filter),
      ms(p.nms),
      ms(p['mask-encode']),
      ms(result.totalMs),
      String(result.counts.afterNms),
      ms(p['model-load']),
    ].join(' | ');
  });

  const lines = [header, divider, ...body.map((row) => `| ${row} |`)];
  lines.push('', 'All times in ms. `model-load` is one-time per run and sits outside the budget.');

  if (agreements.length > 0) {
    lines.push('', '### fp16 vs fp32 mask agreement', '');
    lines.push('| pair | density | fp32 masks | fp16 masks | matched | unmatched fp32 | unmatched fp16 | mean IoU | median IoU | min IoU |');
    lines.push('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |');
    for (const entry of agreements) {
      const a = entry.agreement;
      lines.push(
        `| ${entry.pairId} | ${entry.label} | ${a.baselineCount} | ${a.variantCount} | ${a.matched} | ${a.unmatchedBaseline} | ${a.unmatchedVariant} | ${a.meanIou.toFixed(3)} | ${a.medianIou.toFixed(3)} | ${a.minIou.toFixed(3)} |`,
      );
    }
    lines.push('', `Pairing is greedy best-IoU; a pair below IoU ${IOU_MATCH_FLOOR} counts as unmatched on both sides.`);
  }

  return lines.join('\n');
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npx vitest run playground/compare.test.ts`
Expected: PASS — every test in the file.

If `createTimingAccumulator().record(...)` does not have the signature the test stub assumes, read `src/segmenter/core/timing.ts` and adapt the `stubResult` helper to build a `TimingReport` literal directly via `summarizePhase([total])` per phase. Do not change `timing.ts`.

- [ ] **Step 5: Run the whole suite**

Run: `npm run typecheck && npm test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add playground/compare.ts playground/compare.test.ts
git commit -m "feat(playground): fp16/batchSize compare matrix, row runner and IoU agreement"
```

---

### Task 3: the Compare tab, the Segment-view selects, and the README note

**Files:**
- Create: `playground/CompareView.tsx`
- Modify: `playground/main.tsx` (the `View` union, the nav, the render)
- Modify: `playground/SegmentView.tsx` (add `dtype` and `batchSize` selects to the existing control paragraph)
- Modify: `README.md` (a note under `## note-scanner/segmenter (optional, prototype)`)

**Interfaces:**
- Consumes: everything Task 2 produces (`COMPARE_ROWS`, `COMPARE_PAIRS`, `CHOSEN_BATCH_SIZE_CHOICES`, `DEFAULT_CHOSEN_BATCH_SIZE`, `rowOptions`, `runRow`, `pairAgreements`, `toMarkdown`, `CompareResult`, `CompareRow`), plus `keepRawMasks` from Task 1.
- Produces: nothing further consumed by other tasks.

There is no vitest test in this task: `vitest.config.ts` includes only `playground/**/*.test.ts`, the environment is `node`, and no playground `.tsx` has ever had a test. The logic worth testing was extracted into `compare.ts` in Task 2 precisely so this file could stay thin.

- [ ] **Step 1: Write `playground/CompareView.tsx`**

```tsx
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  SegmenterFailure,
  createSegmenter,
  isWebGPUAvailable,
  type SegmenterProgress,
} from '../src/segmenter';
import {
  CHOSEN_BATCH_SIZE_CHOICES,
  COMPARE_ROWS,
  DEFAULT_CHOSEN_BATCH_SIZE,
  pairAgreements,
  rowOptions,
  runRow,
  toMarkdown,
  type CompareResult,
  type CompareRow,
} from './compare';
import sampleUrl from './sample/cafe-table.jpg';

function ms(value: number): string {
  return value.toFixed(0);
}

export function CompareView() {
  // Probed once: a GPU adapter does not appear part-way through a session.
  const webgpu = useMemo(() => isWebGPUAvailable(), []);
  const [imageUrl, setImageUrl] = useState<string>(sampleUrl);
  const [chosenBatchSize, setChosenBatchSize] = useState<number>(DEFAULT_CHOSEN_BATCH_SIZE);
  const [results, setResults] = useState<Record<string, CompareResult>>({});
  const [errors, setErrors] = useState<Record<string, { phase: string; message: string }>>({});
  const [runningRowId, setRunningRowId] = useState<string | null>(null);
  const [sweeping, setSweeping] = useState(false);
  const [progress, setProgress] = useState<SegmenterProgress | null>(null);
  const [copied, setCopied] = useState(false);
  const segmenterRef = useRef<ReturnType<typeof createSegmenter> | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  useEffect(
    () => () => {
      segmenterRef.current?.dispose();
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    },
    [],
  );

  const runOne = useCallback(
    async (row: CompareRow) => {
      setRunningRowId(row.id);
      setProgress(null);
      setErrors((current) => {
        const next = { ...current };
        delete next[row.id];
        return next;
      });
      try {
        // Created lazily so a browser with no WebGPU never spawns a worker.
        segmenterRef.current ??= createSegmenter();
        const segmenter = segmenterRef.current;
        const result = await runRow(row, {
          chosenBatchSize,
          // Fresh per row: the bitmap is transferred to the worker and consumed.
          createBitmap: async () => createImageBitmap(await (await fetch(imageUrl)).blob()),
          segment: (bitmap, options, onProgress) =>
            segmenter.segment(bitmap, options, onProgress),
          onProgress: setProgress,
        });
        // Replacing the entry drops the previous run's rawMasks reference, so a
        // re-run releases the old coverage buffers rather than stacking them.
        setResults((current) => ({ ...current, [row.id]: result }));
      } catch (thrown) {
        const failure = thrown instanceof SegmenterFailure ? thrown : null;
        // Recorded against this row only — one bad row must not abort the sweep.
        setErrors((current) => ({
          ...current,
          [row.id]: {
            phase: failure?.phase ?? 'unknown',
            message: thrown instanceof Error ? thrown.message : String(thrown),
          },
        }));
      } finally {
        setRunningRowId(null);
        setProgress(null);
      }
    },
    [chosenBatchSize, imageUrl],
  );

  const runAll = useCallback(async () => {
    setSweeping(true);
    // Sequential: two runs at once would contend for the same GPU and make
    // every timing in the table meaningless.
    for (const row of COMPARE_ROWS) await runOne(row);
    setSweeping(false);
  }, [runOne]);

  function pickFile(file: File | undefined) {
    if (!file) return;
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setImageUrl(url);
    setResults({});
    setErrors({});
  }

  const ordered = COMPARE_ROWS.map((row) => results[row.id]).filter(
    (result): result is CompareResult => Boolean(result),
  );
  const agreements = pairAgreements(results);
  const busy = runningRowId !== null || sweeping;

  async function copyMarkdown() {
    await navigator.clipboard.writeText(toMarkdown(ordered, agreements));
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  if (!webgpu) {
    return (
      <section
        data-testid="webgpu-required"
        style={{ border: '1px solid #f59e0b', borderRadius: 6, padding: 12 }}
      >
        <h2 style={{ margin: '0 0 4px', fontSize: '1rem' }}>WebGPU required</h2>
        <p style={{ margin: 0 }}>
          This comparison runs the whole segmentation model in your browser on{' '}
          <strong>WebGPU</strong>. This browser exposes no <code>navigator.gpu</code>{' '}
          adapter, so nothing has been downloaded and no worker has been started.
          Chrome or Edge 113+, Firefox 141+, or Safari 26+ on a supported GPU can run
          it. There is no CPU fallback by design — on CPU a single pass takes minutes.
        </p>
      </section>
    );
  }

  return (
    <>
      <p>
        Seven rows measuring <strong>fp16 vs fp32</strong> and the{' '}
        <strong>batchSize</strong> curve. Every row respawns the worker and pays its
        own <code>model-load</code>, which is reported separately and excluded from the
        budget. Expect roughly 9 minutes for the four dtype rows.
      </p>

      <p>
        <label>
          confirmation-row batch size:{' '}
          <select
            data-testid="chosen-batch-size"
            value={chosenBatchSize}
            disabled={busy}
            onChange={(e) => setChosenBatchSize(Number(e.target.value))}
          >
            {CHOSEN_BATCH_SIZE_CHOICES.map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>{' '}
        <label>
          image:{' '}
          <input
            data-testid="compare-file-input"
            type="file"
            accept="image/*"
            disabled={busy}
            onChange={(e) => pickFile(e.target.files?.[0])}
          />
        </label>{' '}
        <button data-testid="run-all" type="button" disabled={busy} onClick={() => void runAll()}>
          {sweeping ? 'Running all…' : 'Run all'}
        </button>{' '}
        <button
          data-testid="copy-markdown"
          type="button"
          disabled={ordered.length === 0}
          onClick={() => void copyMarkdown()}
        >
          {copied ? 'Copied' : 'Copy as markdown'}
        </button>
      </p>

      {busy && (
        <p data-testid="compare-progress" role="status">
          {runningRowId ?? '…'}:{' '}
          {progress
            ? `${progress.phase} ${progress.done}/${progress.total} — ${progress.ms.toFixed(1)} ms`
            : 'starting…'}
        </p>
      )}

      <table data-testid="compare-table" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th align="left">row</th>
            <th align="left">dtype</th>
            <th align="right">pps</th>
            <th align="right">batch</th>
            <th align="right">encode</th>
            <th align="right">decode</th>
            <th align="right">encode + decode</th>
            <th align="right">total</th>
            <th align="right">kept</th>
            <th align="left">serves</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {COMPARE_ROWS.map((row) => {
            const options = rowOptions(row, chosenBatchSize);
            const result = results[row.id];
            const error = errors[row.id];
            return (
              <tr key={row.id} data-testid={`compare-${row.id}`}>
                <th align="left" scope="row">{row.id}</th>
                <td>{options.dtype}</td>
                <td align="right">{options.pointsPerSide}</td>
                <td align="right">{options.batchSize}</td>
                {result ? (
                  <>
                    <td align="right">{ms(result.phases.encode)}</td>
                    <td align="right">{ms(result.phases.decode)}</td>
                    <td align="right"><strong>{ms(result.encodeDecodeMs)}</strong></td>
                    <td align="right">{ms(result.totalMs)}</td>
                    <td align="right">{result.counts.afterNms}</td>
                  </>
                ) : (
                  <td colSpan={5} align="left">
                    {error ? (
                      <span role="alert">
                        failed in <strong>{error.phase}</strong>: {error.message}
                      </span>
                    ) : (
                      'not run'
                    )}
                  </td>
                )}
                <td>{row.serves}</td>
                <td>
                  <button type="button" disabled={busy} onClick={() => void runOne(row)}>
                    {runningRowId === row.id ? 'Running…' : 'Run'}
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {agreements.length > 0 && (
        <section data-testid="agreement-panel">
          <h2 style={{ fontSize: '1rem' }}>fp16 vs fp32 mask agreement</h2>
          {agreements.map((entry) => {
            const a = entry.agreement;
            return (
              <p key={entry.pairId} data-testid={`agreement-${entry.pairId}`}>
                <strong>{entry.label}</strong> ({entry.baselineRowId} vs {entry.variantRowId}):{' '}
                {a.baselineCount} fp32 masks vs {a.variantCount} fp16, {a.matched} matched at
                IoU ≥ {a.floor} ({a.unmatchedBaseline} / {a.unmatchedVariant} unmatched). IoU
                mean {a.meanIou.toFixed(3)}, median {a.medianIou.toFixed(3)}, min{' '}
                {a.minIou.toFixed(3)}.
              </p>
            );
          })}
        </section>
      )}
    </>
  );
}
```

- [ ] **Step 2: Add the Compare tab to `playground/main.tsx`**

Add the import beside the other two view imports:

```tsx
import { CompareView } from './CompareView';
```

Widen the union and add the third button plus the render branch:

```tsx
type View = 'benchmark' | 'segment' | 'compare';
```

```tsx
        <button
          data-testid="view-compare"
          type="button"
          aria-pressed={view === 'compare'}
          onClick={() => setView('compare')}
        >
          Compare
        </button>
```

Replace the ternary render with an explicit switch on the three views:

```tsx
      {view === 'benchmark' && <BenchmarkView />}
      {view === 'segment' && <SegmentView />}
      {view === 'compare' && <CompareView />}
```

- [ ] **Step 3: Add `dtype` and `batchSize` selects to `playground/SegmentView.tsx`**

These are for one-off pokes outside the matrix. Add the choice constants beside the existing `POINTS_PER_SIDE_CHOICES`:

```tsx
const DTYPE_CHOICES = ['fp32', 'fp16', 'q8'] as const;
/** Grid points per model call. 8 is the shipped default; the rest are probes. */
const BATCH_SIZE_CHOICES = [8, 16, 32, 64] as const;
```

Add two labelled selects to the existing control paragraph, immediately after the `points per side` label, following the same pattern:

```tsx
        <label>
          dtype:{' '}
          <select
            data-testid="dtype"
            value={options.dtype}
            onChange={(e) => patch({ dtype: e.target.value as SegmenterOptions['dtype'] })}
          >
            {DTYPE_CHOICES.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </label>{' '}
        <label>
          batch size:{' '}
          <select
            data-testid="batch-size"
            value={options.batchSize}
            onChange={(e) => patch({ batchSize: Number(e.target.value) })}
          >
            {BATCH_SIZE_CHOICES.map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>{' '}
```

`SegmenterOptions` is already imported in that file as a type. Change nothing else in `SegmentView.tsx`.

- [ ] **Step 4: Add the README note**

In `README.md`, inside `## note-scanner/segmenter (optional, prototype)`, insert this immediately before the `### Status` heading:

```markdown
### Measuring dtype and `batchSize`

`npm run playground` serves a **Compare** tab (`playground/CompareView.tsx`) that
runs a fixed seven-row matrix — fp16 vs fp32 at 16 and 32 points per side, plus a
`batchSize` curve — and renders the per-phase timings, the `encode + decode`
subtotal, and an fp16-vs-fp32 mask-agreement summary as markdown you can paste
into an issue. It needs a real GPU browser session; the presets, the greedy
best-IoU pairing and the markdown renderer live in `playground/compare.ts` and
are unit-tested without one.

The comparison uses the one addition this makes to the package surface:
`segment(bitmap, { keepRawMasks: true })` retains the full-resolution
`RawMask[]` on the result as `rawMasks`. It is a client-side retention flag, not
an inference parameter — the worker ignores it — and it is off by default
because retaining ~50 full-resolution coverage arrays pins tens of megabytes for
as long as you hold the result.
```

- [ ] **Step 5: Verify**

Run: `npm run typecheck && npm test`
Expected: PASS.

Note that `tsconfig.json` excludes `playground/`, so the typecheck does not cover the three playground files this task touched. Smoke the bundle instead:

Run: `npx vite build --config playground/vite.config.ts --outDir /tmp/notescanner-compare-build --emptyOutDir`
Expected: a successful build. If it fails, read the error: a resolution or syntax error in `CompareView.tsx` / `main.tsx` / `SegmentView.tsx` must be fixed. A failure originating in `@huggingface/transformers` or ONNX worker bundling is pre-existing and out of scope — confirm it by checking out `HEAD` versions of the three files, re-running, and seeing the same failure; if so, say so in the task report and move on.

- [ ] **Step 6: Commit**

```bash
git add playground/CompareView.tsx playground/main.tsx playground/SegmentView.tsx README.md
git commit -m "feat(playground): Compare tab for the fp16 and batchSize matrix"
```

---

## Acceptance criteria mapping

| AC | Where it lands |
|---|---|
| AC1 — fp32/fp16 rows at 16 and 32 pps, `encode + decode` subtotal | Task 2 (`COMPARE_ROWS`, `runRow.encodeDecodeMs`) |
| AC2 — kept-mask counts and IoU distribution over a greedy best-IoU pairing with a stated floor | Task 2 (`compareMaskSets`, `IOU_MATCH_FLOOR`) |
| AC3 — `batchSize` sweep plus a developer-supplied confirmation row | Task 2 (rows 5-7, `rowOptions`), Task 3 (the `chosen-batch-size` select) |
| AC4 — `DEFAULT_SEGMENTER_OPTIONS` unchanged | Task 1 (the `DEFAULT_SEGMENTER_OPTIONS` regression test) |
| AC5 — `rawMasks` present on request, absent otherwise | Task 1 |
| AC6 — `npm run typecheck && npm test` pass | every task's verification step |
