# Filter Sub-Step Timers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument the segmenter's `filter` stage with three exhaustive, non-overlapping sub-timers (`select`, `upscale`, `threshold`) so the reasoned 3%/80%/17% split in issue #1 §3.3 becomes measurable.

**Architecture:** A second `Map` inside the existing `createTimingAccumulator` factory backs one new method, `recordFilterSub`. `TimingReport` gains a **required** `filterSubPhases: Record<FilterSubstep, PhaseTiming>` field — required, not optional, so the compiler catches a dropped passthrough where `createSegmenter` rebuilds the report as an object literal. The worker wraps three regions of the existing `filter` stage; the existing `timings.record('filter', ...)` call is untouched, so the sub-steps nest inside the phase rather than replacing it. The playground renders them as indented rows under `filter`.

**Tech Stack:** TypeScript, vitest (node environment), React 18 (playground only), `@huggingface/transformers` (worker only, not exercised by tests).

**Spec:** `docs/superpowers/specs/2026-08-27-filter-substep-timers-design.md`

## Global Constraints

- `PHASE_ORDER`, `SegmentationPhase` and `SegmenterFailure.phase` are **not** touched. Do not add `filter:select` etc. to `PHASE_ORDER`.
- `filterSubPhases` is **required** on `TimingReport`, never optional.
- Do **not** introduce a generic `createSampleAccumulator<K>` or a generic `subPhases: Partial<Record<SegmentationPhase, ...>>`. One extra `Map` and one extra method inside the existing factory. YAGNI.
- The existing `timings.record('filter', performance.now() - started)` call in the worker stays exactly as it is. Do not change what the `filter`, `nms` or `mask-encode` phases measure.
- The three worker regions must be **exhaustive and non-overlapping** across the stage — they sum to the `filter` total with no residual.
- `upscale` and `threshold` record **only when `chosen.length > 0`**. A zero-survivor batch executes neither region; recording a 0 ms sample would drag their `p50` down misleadingly.
- The instrumentation must be operating-point independent: nothing keyed to a particular `pointsPerSide` or `batchSize`.
- Reuse `summarizePhase` unchanged.
- Sub-step order is exactly `['select', 'upscale', 'threshold']`.
- Out of scope: producing real figures on hardware, editing issue #1, re-scoping issues #6/#8, sub-timers for any stage other than `filter`.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/segmenter/core/types.ts` | Modify (~line 15 and ~line 71) | Adds `FILTER_SUBSTEP_ORDER` / `FilterSubstep` beside `PHASE_ORDER`, and the required `filterSubPhases` field on `TimingReport`. |
| `src/segmenter/core/timing.ts` | Modify | `TimingAccumulator` gains `recordFilterSub`, backed by a second `Map`; `report()` zero-fills every sub-step. |
| `src/segmenter/core/timing.test.ts` | Modify | Zero-fill, accumulation and the no-residual sum invariant. |
| `src/segmenter/worker/segmenter.worker.ts` | Modify (lines 160-225) | Three timer regions inside the existing `filter` stage. |
| `src/segmenter/createSegmenter.ts` | Modify (~line 133) | Carries `filterSubPhases` through the rebuilt report literal. |
| `src/segmenter/createSegmenter.test.ts` | Modify | Proves the field survives the worker -> main-thread boundary. |
| `playground/SegmentView.tsx` | Modify (lines 1, 246-278) | Renders the three sub-steps as visibly nested rows under `filter`. |

No new files. `src/segmenter/core/index.ts` already does `export * from './types'` and `'./timing'`, so the new symbols are exported with no change there.

---

### Task 1: Filter sub-step timers end to end

**Files:**
- Modify: `src/segmenter/core/types.ts:8-17`, `src/segmenter/core/types.ts:71-75`
- Modify: `src/segmenter/core/timing.ts:1`, `:28-53`
- Modify: `src/segmenter/worker/segmenter.worker.ts:25-37`, `:160-225`
- Modify: `src/segmenter/createSegmenter.ts:131-143`
- Modify: `playground/SegmentView.tsx:1-13`, `:257-270`
- Test: `src/segmenter/core/timing.test.ts`, `src/segmenter/createSegmenter.test.ts`

**Interfaces:**
- Consumes: existing `summarizePhase(samples: readonly number[]): PhaseTiming`, `createTimingAccumulator(): TimingAccumulator`, `PHASE_ORDER`.
- Produces: `FILTER_SUBSTEP_ORDER: readonly ['select','upscale','threshold']`; `type FilterSubstep = 'select'|'upscale'|'threshold'`; `TimingAccumulator.recordFilterSub(step: FilterSubstep, ms: number): void`; `TimingReport.filterSubPhases: Record<FilterSubstep, PhaseTiming>` (required).

---

- [ ] **Step 1: Write the failing timing tests**

Append to `src/segmenter/core/timing.test.ts`, and change the import on line 2 to `import { PHASE_ORDER, FILTER_SUBSTEP_ORDER } from './types';`:

```ts
describe('createTimingAccumulator filter sub-steps', () => {
  it('reports every sub-step in FILTER_SUBSTEP_ORDER, zero-filled when never recorded', () => {
    const report = createTimingAccumulator().report(0);
    expect(Object.keys(report.filterSubPhases)).toEqual([...FILTER_SUBSTEP_ORDER]);
    for (const step of FILTER_SUBSTEP_ORDER) {
      expect(report.filterSubPhases[step].count).toBe(0);
    }
  });

  it('accumulates repeated samples for one sub-step', () => {
    const timings = createTimingAccumulator();
    timings.recordFilterSub('upscale', 10);
    timings.recordFilterSub('upscale', 30);
    timings.recordFilterSub('upscale', 20);
    const report = timings.report(60);
    expect(report.filterSubPhases.upscale.count).toBe(3);
    expect(report.filterSubPhases.upscale.p50).toBe(20);
    expect(report.filterSubPhases.upscale.max).toBe(30);
    expect(report.filterSubPhases.upscale.total).toBe(60);
    expect(report.filterSubPhases.select.count).toBe(0);
  });

  it('keeps sub-steps independent of the phases they nest inside', () => {
    const timings = createTimingAccumulator();
    timings.record('filter', 100);
    timings.recordFilterSub('select', 3);
    const report = timings.report(100);
    expect(report.phases.filter.total).toBe(100);
    expect(report.filterSubPhases.select.total).toBe(3);
    expect(report.filterSubPhases.threshold.total).toBe(0);
  });

  // The no-residual invariant: the worker's three regions are drawn to be
  // exhaustive and non-overlapping across the stage, so their totals sum to
  // the filter total. A residual is exactly what would muddy the percentage
  // breakdown this instrumentation exists to produce.
  it('has the three sub-step totals sum to the filter total with no residual', () => {
    const timings = createTimingAccumulator();
    timings.record('filter', 100);
    timings.recordFilterSub('select', 3);
    timings.recordFilterSub('upscale', 80);
    timings.recordFilterSub('threshold', 17);
    timings.record('filter', 50);
    timings.recordFilterSub('select', 2);
    timings.recordFilterSub('upscale', 40);
    timings.recordFilterSub('threshold', 8);
    const report = timings.report(150);
    const subTotal = FILTER_SUBSTEP_ORDER.reduce(
      (sum, step) => sum + report.filterSubPhases[step].total,
      0,
    );
    expect(subTotal).toBe(report.phases.filter.total);
  });
});
```

- [ ] **Step 2: Run the timing tests to verify they fail**

Run: `npx vitest run src/segmenter/core/timing.test.ts`
Expected: FAIL — `FILTER_SUBSTEP_ORDER` is not exported from `./types` (undefined at import / `recordFilterSub is not a function`).

- [ ] **Step 3: Add the sub-step vocabulary and the report field to `types.ts`**

In `src/segmenter/core/types.ts`, immediately after the `export type SegmentationPhase = (typeof PHASE_ORDER)[number];` line (line 17), insert:

```ts
/**
 * The internal split of the `filter` stage, in render order. Deliberately NOT
 * folded into `PHASE_ORDER`: `SegmentationPhase` is public API and also types
 * `SegmenterFailure.phase`, so widening it would admit values that can never
 * be thrown, and summing the results table's total column would count
 * `filter` twice.
 */
export const FILTER_SUBSTEP_ORDER = ['select', 'upscale', 'threshold'] as const;

export type FilterSubstep = (typeof FILTER_SUBSTEP_ORDER)[number];
```

Then replace the `TimingReport` interface (lines 71-75) with:

```ts
export interface TimingReport {
  phases: Record<SegmentationPhase, PhaseTiming>;
  /**
   * The `filter` stage broken down. Required, not optional: `createSegmenter`
   * rebuilds this report as an object literal to splice in `mask-encode`, and
   * requiring the field makes the compiler catch a dropped passthrough across
   * the worker boundary.
   */
  filterSubPhases: Record<FilterSubstep, PhaseTiming>;
  /** Wall clock for the whole run, measured on the main thread. */
  totalMs: number;
}
```

- [ ] **Step 4: Add `recordFilterSub` to the accumulator**

In `src/segmenter/core/timing.ts`, replace the import on line 1 with:

```ts
import {
  FILTER_SUBSTEP_ORDER,
  PHASE_ORDER,
  type FilterSubstep,
  type PhaseTiming,
  type SegmentationPhase,
  type TimingReport,
} from './types';
```

Replace the `TimingAccumulator` interface and `createTimingAccumulator` (lines 28-53) with:

```ts
export interface TimingAccumulator {
  record(phase: SegmentationPhase, ms: number): void;
  /** A region inside the `filter` stage. Nests under it; does not replace it. */
  recordFilterSub(step: FilterSubstep, ms: number): void;
  /** `totalMs` is wall clock for the whole run, measured by the caller. */
  report(totalMs: number): TimingReport;
}

export function createTimingAccumulator(): TimingAccumulator {
  const samples = new Map<SegmentationPhase, number[]>();
  const filterSamples = new Map<FilterSubstep, number[]>();

  const push = <K>(into: Map<K, number[]>, key: K, ms: number) => {
    const bucket = into.get(key);
    if (bucket) bucket.push(ms);
    else into.set(key, [ms]);
  };

  return {
    record(phase, ms) {
      push(samples, phase, ms);
    },
    recordFilterSub(step, ms) {
      push(filterSamples, step, ms);
    },
    report(totalMs) {
      // Every phase gets a row even when it never ran, so the results table
      // has a stable shape and a missing phase reads as "0 samples" rather
      // than as a hole.
      const phases = {} as Record<SegmentationPhase, PhaseTiming>;
      for (const phase of PHASE_ORDER) {
        phases[phase] = summarizePhase(samples.get(phase) ?? []);
      }
      // Same zero-fill rule for the filter breakdown.
      const filterSubPhases = {} as Record<FilterSubstep, PhaseTiming>;
      for (const step of FILTER_SUBSTEP_ORDER) {
        filterSubPhases[step] = summarizePhase(filterSamples.get(step) ?? []);
      }
      return { phases, filterSubPhases, totalMs };
    },
  };
}
```

- [ ] **Step 5: Run the timing tests to verify they pass**

Run: `npx vitest run src/segmenter/core/timing.test.ts`
Expected: PASS — all tests in both `describe` blocks green.

- [ ] **Step 6: Write the failing boundary test**

In `src/segmenter/createSegmenter.test.ts`, add `FILTER_SUBSTEP_ORDER,` to the import list from `'./core'` (line 4 area, alphabetically before `PHASE_ORDER`). Then add this test inside the `describe('createSegmenter', ...)` block, right after the `'resolves with counts, timings and a mask-encode phase'` test:

```ts
  it('carries the worker filterSubPhases through the rebuilt report', async () => {
    const workerTimings = createTimingAccumulator();
    workerTimings.recordFilterSub('select', 4);
    workerTimings.recordFilterSub('select', 6);
    workerTimings.recordFilterSub('upscale', 100);

    const segmenter = createSegmenter({ createWorker: spawn });
    const pending = segmenter.segment(fakeBitmap());
    FakeWorker.instances[0].emit({
      type: 'done',
      masks: [],
      width: 2,
      height: 1,
      timings: workerTimings.report(42),
      counts: { raw: 24, afterFilter: 9, afterNms: 0 },
    });

    const result = await pending;
    expect(Object.keys(result.timings.filterSubPhases)).toEqual([...FILTER_SUBSTEP_ORDER]);
    expect(result.timings.filterSubPhases.select.total).toBe(10);
    expect(result.timings.filterSubPhases.select.p50).toBe(6);
    expect(result.timings.filterSubPhases.upscale.total).toBe(100);
    expect(result.timings.filterSubPhases.upscale.p50).toBe(100);
    // The main thread contributes nothing to `filter`, so an unrecorded
    // sub-step still arrives zero-filled rather than missing.
    expect(result.timings.filterSubPhases.threshold.count).toBe(0);
  });
```

- [ ] **Step 7: Run the boundary test to verify it fails**

Run: `npx vitest run src/segmenter/createSegmenter.test.ts -t 'carries the worker filterSubPhases'`
Expected: FAIL — `Cannot convert undefined or null to object` / `result.timings.filterSubPhases` is `undefined`, because the rebuilt literal drops it.

- [ ] **Step 8: Carry the field through `createSegmenter`**

In `src/segmenter/createSegmenter.ts`, in the `resolve({ ... })` call (lines 131-143), add the passthrough to the `timings` literal:

```ts
            resolve({
              segments,
              timings: {
                phases: {
                  ...message.timings.phases,
                  'mask-encode': summarizePhase(encodeSamples),
                },
                // Straight passthrough: the main thread contributes nothing to
                // the `filter` stage.
                filterSubPhases: message.timings.filterSubPhases,
                // Wall clock from this side of the boundary, so worker spawn
                // and bitmap transfer are inside the number the table reports.
                totalMs: performance.now() - startedAt,
              },
              counts: message.counts,
            });
```

- [ ] **Step 9: Run the boundary test to verify it passes**

Run: `npx vitest run src/segmenter/createSegmenter.test.ts`
Expected: PASS — the whole file, including the pre-existing tests.

- [ ] **Step 10: Instrument the worker's three regions**

In `src/segmenter/worker/segmenter.worker.ts`, add `type FilterSubstep,` to the import block from `'../core'` (alphabetically among the other `type` imports, before `type RawMask`).

The three regions are exhaustive and non-overlapping across the stage:

| Sub-step | Region | Resolution |
|---|---|---|
| `select` | stage start (the `dims` reads) through the end of the best-of-3 `stabilityScore` loop | 256x256 |
| `upscale` | the `selected` `Float32Array` gather **and** `await processor.post_process_masks(...)`, through reading `upscaled[0].data` | 256x256 -> full |
| `threshold` | the `thresholdMask` + `minMaskArea` loop | full res |

The gather is folded into `upscale` rather than timed separately: timing `post_process_masks` alone would leave the gather as an unattributed residual, and a residual is precisely what muddies the "is step 2 materially below ~75%?" decision. It is ~1.2 MB of memcpy per batch and will read sub-millisecond.

Immediately after `started = performance.now();` on the `phase = 'filter';` line pair (line 162), add a second cursor:

```ts
      // ---- filter, at low resolution ----
      phase = 'filter';
      started = performance.now();
      // A second cursor for the sub-regions. Each region ends exactly where
      // the next begins, so the three sum to the stage total.
      let subStarted = started;
      const recordSub = (step: FilterSubstep) => {
        const now = performance.now();
        timings.recordFilterSub(step, now - subStarted);
        subStarted = now;
      };
```

Close the `select` region immediately after the best-of-3 loop — i.e. between the closing brace of `for (let p = 0; ...)` and the `if (chosen.length > 0) {` line:

```ts
        if (bestFlat >= 0) chosen.push(bestFlat);
      }
      recordSub('select');

      if (chosen.length > 0) {
```

Inside the `if (chosen.length > 0)` block, close `upscale` right after `const full = ...`, and close `threshold` at the end of the block:

```ts
        const full = upscaled[0].data as Float32Array;
        recordSub('upscale');
        for (let k = 0; k < chosen.length; k += 1) {
          const mask = thresholdMask(
            full.subarray(k * fullPixels, (k + 1) * fullPixels),
            options.maskThreshold,
          );
          if (mask.area >= options.minMaskArea) candidates.push(mask);
        }
        recordSub('threshold');
      }
      timings.record('filter', performance.now() - started);
```

Because `recordSub` is only reachable for `upscale` and `threshold` inside the `if`, a zero-survivor batch records neither — the sum invariant still holds and their `p50` is not dragged down by 0 ms samples. The `timings.record('filter', ...)` line is unchanged. Nothing here reads `options.pointsPerSide` or `options.batchSize`, so the instrumentation is operating-point independent.

- [ ] **Step 11: Render the nested rows in the playground**

In `playground/SegmentView.tsx`, change line 1 to import `Fragment`:

```tsx
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
```

and add `FILTER_SUBSTEP_ORDER,` to the `'../src/segmenter'` import list (before `PHASE_ORDER`).

Replace the `PHASE_ORDER.map(...)` block in `<tbody>` (lines 258-270) with:

```tsx
              {PHASE_ORDER.map((phase) => {
                const timing = result.timings.phases[phase];
                return (
                  <Fragment key={phase}>
                    <tr>
                      <th align="left" scope="row">{phase}</th>
                      <td align="right">{ms(timing.p50)}</td>
                      <td align="right">{ms(timing.p95)}</td>
                      <td align="right">{ms(timing.max)}</td>
                      <td align="right">{ms(timing.total)}</td>
                      <td align="right">{timing.count}</td>
                    </tr>
                    {/* The filter breakdown nests under filter and is NOT a
                        peer of it — indented and tree-prefixed so nobody sums
                        it into the phase column. */}
                    {phase === 'filter' &&
                      FILTER_SUBSTEP_ORDER.map((step) => {
                        const sub = result.timings.filterSubPhases[step];
                        return (
                          <tr key={`filter-${step}`} data-testid={`filter-substep-${step}`}>
                            <th
                              align="left"
                              scope="row"
                              style={{ paddingLeft: '1.5em', fontWeight: 'normal', opacity: 0.8 }}
                            >
                              └ {step}
                            </th>
                            <td align="right">{ms(sub.p50)}</td>
                            <td align="right">{ms(sub.p95)}</td>
                            <td align="right">{ms(sub.max)}</td>
                            <td align="right">{ms(sub.total)}</td>
                            <td align="right">{sub.count}</td>
                          </tr>
                        );
                      })}
                  </Fragment>
                );
              })}
```

Leave the `total` row and everything below it untouched. Nothing in this block references `pointsPerSide` or `batchSize`, so it renders identically at both `POINTS_PER_SIDE_CHOICES` values.

- [ ] **Step 12: Run the full suite and the typechecker**

Run: `npm test && npm run typecheck`
Expected: both PASS. The typecheck is the load-bearing half — the required `filterSubPhases` field means any other construction site of a `TimingReport` object literal would be a compile error, and this proves there are none left.

- [ ] **Step 13: Verify the acceptance criteria by reading**

- AC1/AC2: `report()` zero-fills all of `FILTER_SUBSTEP_ORDER`, and `PhaseTiming` carries `total` and `p50` — covered by the Step 1 tests.
- AC3: re-read the Step 10 diff and confirm `select` ends where `upscale` begins, `upscale` ends where `threshold` begins, and `threshold` ends before `timings.record('filter', ...)`. No statement inside the stage sits outside a region except the `recordSub` calls themselves.
- AC4: covered by the Step 6 test.
- AC5 (non-ui, verified by reading `playground/SegmentView.tsx`): the three rows render from `FILTER_SUBSTEP_ORDER`, indented under `filter`, keyed to neither `pointsPerSide` nor `batchSize`.
- AC6 and AC7 are **deferred to the human and are not a gate on this PR** — they need a WebGPU browser. Do not attempt them.

- [ ] **Step 14: Commit**

```bash
git add src/segmenter/core/types.ts src/segmenter/core/timing.ts \
  src/segmenter/core/timing.test.ts src/segmenter/worker/segmenter.worker.ts \
  src/segmenter/createSegmenter.ts src/segmenter/createSegmenter.test.ts \
  playground/SegmentView.tsx
git commit -m "feat(segmenter): sub-timers for the filter stage's three regions"
```
