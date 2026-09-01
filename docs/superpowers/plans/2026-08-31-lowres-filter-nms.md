# F1 + N1 + F3 — carry 256x256 through filter and NMS — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorder the worker's mask pipeline so full resolution is reached only by masks that survive NMS — retain each chosen candidate as a 256x256 logit window plus a 256x256 binary mask, dedupe on that low-res coverage, and resample once per survivor fused with the PNG encode.

**Architecture:** Every RULE the pipeline turns on moves into a new pure, model-free module `src/segmenter/core/mask-pipeline.ts` that implements BOTH the new low-res path and the preserved full-res baseline behind one flag. The worker keeps only the ORDER, the timing and the async encode. That module is what the playground's new Boundary tab drives with no WebGPU, no model and no network, which is what makes AC2 checkable in a real browser.

**Tech Stack:** TypeScript, vitest (node env; jsdom per-file), Playwright (`@playwright/test` 1.62.1, three engines), Vite playground, React 19.

**Spec:** `docs/superpowers/specs/2026-08-31-lowres-filter-nms-design.md` — read it before Task 1. It is the authority; this plan does not redesign it.

## Global Constraints

- **No new dependencies.** Everything below uses packages already in `package.json`.
- **`npm run typecheck` covers `src/**` and `playground/**` only.** `tsconfig.json` excludes `playground`, `tsconfig.playground.json` covers it, and NEITHER includes `tests/`. A type error in `tests/browser/*.spec.ts` surfaces only when `npm run test:browser` runs. Never cite `npm run typecheck` as evidence about a file under `tests/`.
- **`npm test` (vitest) only collects `src/**/*.test.ts(x)` and `playground/**/*.test.ts(x)`.** A test placed anywhere else runs nowhere.
- **The worker is not unit-testable.** `src/segmenter/worker/segmenter.worker.ts` binds `globalThis.addEventListener` at module scope and imports `@huggingface/transformers`; there is no seam. See "What is deliberately not unit-tested" below.
- **Harness note:** if the `Write` tool is blocked in this worktree, create files with a Bash heredoc (`cat > path <<'EOF'`).
- **Commit after every task.** Do not push, do not open a PR.

## What is deliberately not unit-tested, and why

The worker's batch loop cannot be exercised by vitest. This plan's answer is not to skip the coverage but to move the decisions out: `mask-pipeline.ts` owns the pre-NMS gate, the retained-copy rule, the NMS resolution and the full-res re-check, and is tested directly (Task 1). What stays untested-by-unit is the worker's CALL ORDER, its `phase` assignments and its timing regions — those are bound to code review, to the Boundary tab (which calls the same functions in the same order with no worker), and to the committed real-GPU measurement in Task 5, whose `counts` and `filterSubPhases` columns would expose a mis-ordered pipeline.

## File structure

| File | Change | Responsibility |
|---|---|---|
| `src/segmenter/core/mask-pipeline.ts` | **Create** | Both pipelines as pure functions: gate arithmetic, candidate retention, dedupe resolution, survivor resolution, release. |
| `src/segmenter/core/mask-pipeline.test.ts` | **Create** | AC5–AC9's arithmetic and invariants, on a geometry where full resolution is exactly 2x the low grid. |
| `src/segmenter/core/index.ts` | Modify (1 line) | Re-export the new module. |
| `src/segmenter/core/types.ts` | Modify | `PHASE_ORDER` gains `resample`; `FILTER_SUBSTEP_ORDER` becomes `['select','threshold']`; `SegmenterOptions.lowResFilterNms`; `SegmentationCounts.returned`. |
| `src/segmenter/worker/segmenter.worker.ts` | Modify | Rewired to call the new module; new `resample` stage; fused survivor resample + encode. |
| `src/segmenter/core/timing.test.ts` | Modify (targeted) | `resample` sub-step renamed to `threshold`. |
| `src/segmenter/createSegmenter.test.ts` | Modify (targeted) | `counts.returned` in fixtures; `lowResFilterNms` passthrough. |
| `src/segmenter/worker/session-key.test.ts` | Modify (append) | `lowResFilterNms` must not enter the session key. |
| `playground/SegmentView.tsx` | Modify (1 line) | The funnel line shows `returned`. |
| `playground/boundary.ts` | **Create** | Procedural 256x256 SDF fixtures + the two-path runner. |
| `playground/boundary.test.ts` | **Create** | Pins each fixture's geometry and the divergence it exists to show. |
| `playground/BoundaryView.tsx` | **Create** | The Boundary tab: per-window status, areas, difference overlay and readouts. |
| `playground/main.tsx` | Modify | Boundary tab in the nav. |
| `playground/vite.config.ts` | Modify (1 line) | `strictPort: true`, so Playwright's `webServer` URL is the URL Vite serves. |
| `playwright.config.ts` | Modify | `baseURL` + `webServer` serving the playground. |
| `tests/browser/boundary.spec.ts` | **Create** | AC2, in three real engines. |
| `playground/compare.ts` | Modify | `lowResFilterNms` on `RowOptions`/`SweepConfig`/grid/label; `resample` and `returned` columns; alignment row derived from column names. |
| `playground/compare.test.ts` | Modify (targeted) | The above. |
| `playground/CompareView.tsx` | Modify | The `lowResFilterNms` control and two new table columns. |
| `playground/CompareView.test.tsx` | Modify (targeted) | Control default + passthrough. |
| `scripts/sweep-decode.mjs` | Modify (2 lines) | Drive the new checkbox; report `returned`. |
| `docs/measurements/*` | **Create** | The committed AC1/AC3/AC4/AC11 evidence. |

Five tasks. Task 1 is additive and self-contained. Task 2 is one diff because the type change and the worker rewire do not compile apart. Task 3 is the browser-verified tab plus the spec that reads it — split them and the plan ships a `data-testid` nothing asserts. Task 4 is the measurement plumbing. Task 5 is the measurement.

---

### Task 1: The pure mask pipeline

**Files:**
- Create: `src/segmenter/core/mask-pipeline.ts`
- Create: `src/segmenter/core/mask-pipeline.test.ts`
- Modify: `src/segmenter/core/index.ts` (line 5 area)

**Interfaces:**
- Consumes: `thresholdMask(logits, threshold): BinaryMask` and `BinaryMask` from `./mask-postprocess`; `resampleThresholdMask(options): BinaryMask` from `./mask-resample`; `dedupeMasks(masks, iouThreshold, width): number[]` from `./nms`.
- Produces, all exported from `./mask-pipeline` and re-exported through `note-scanner/segmenter`:
  - `interface MaskGeometry { lowWidth; lowHeight; padWidth; padHeight; reshapedWidth; reshapedHeight; originalWidth; originalHeight }` (all `number`)
  - `interface FilterNmsOptions { maskThreshold: number; minMaskArea: number; nmsIouThreshold: number; lowResFilterNms: boolean }`
  - `interface MaskCandidate { logits: Float32Array | null; coverage: BinaryMask | null }`
  - `interface FilterPlan { geometry: MaskGeometry; options: FilterNmsOptions; minArea: number; nmsWidth: number }`
  - `interface RetainResult { gateArea: number; candidate: MaskCandidate | null }`
  - `lowResMinArea(minMaskArea: number, lowPixels: number, originalPixels: number): number`
  - `createFilterPlan(geometry: MaskGeometry, options: FilterNmsOptions): FilterPlan`
  - `retainCandidate(window: Float32Array, plan: FilterPlan): RetainResult`
  - `candidateCoverage(candidates: readonly MaskCandidate[]): BinaryMask[]`
  - `dedupeCandidates(candidates: readonly MaskCandidate[], plan: FilterPlan): number[]`
  - `releaseCandidate(candidate: MaskCandidate): void`
  - `releaseRejected(candidates: readonly MaskCandidate[], kept: readonly number[]): number`
  - `resolveSurvivor(candidate: MaskCandidate, plan: FilterPlan): BinaryMask | null`

- [ ] **Step 1: Write the failing test**

Create `src/segmenter/core/mask-pipeline.test.ts` with exactly this content:

```ts
import { describe, it, expect } from 'vitest';
import {
  candidateCoverage,
  createFilterPlan,
  dedupeCandidates,
  lowResMinArea,
  releaseCandidate,
  releaseRejected,
  resolveSurvivor,
  retainCandidate,
  type FilterNmsOptions,
  type MaskCandidate,
  type MaskGeometry,
} from './mask-pipeline';
import { thresholdMask } from './mask-postprocess';
import { resampleThresholdMask } from './mask-resample';
import { dedupeMasks } from './nms';

/**
 * A 16x16 logit grid over a 16x16 pad whose top-left 16x12 maps to a 32x24
 * image, so full resolution is EXACTLY 2x the low grid on both axes and a low
 * box covering columns a..b covers full columns 2a..2b+1. Every expected
 * number in this file was derived from that rule and then confirmed by
 * running the file — none of them is a guess.
 */
const GEOMETRY: MaskGeometry = {
  lowWidth: 16,
  lowHeight: 16,
  padWidth: 16,
  padHeight: 16,
  reshapedWidth: 16,
  reshapedHeight: 12,
  originalWidth: 32,
  originalHeight: 24,
};

const OPTIONS: FilterNmsOptions = {
  maskThreshold: 0,
  minMaskArea: 100,
  nmsIouThreshold: 0.7,
  lowResFilterNms: true,
};

function lowPlan(overrides: Partial<FilterNmsOptions> = {}) {
  return createFilterPlan(GEOMETRY, { ...OPTIONS, lowResFilterNms: true, ...overrides });
}

function fullPlan(overrides: Partial<FilterNmsOptions> = {}) {
  return createFilterPlan(GEOMETRY, { ...OPTIONS, lowResFilterNms: false, ...overrides });
}

/** +1 inside any listed inclusive box, -1 outside. */
function makeWindow(...boxes: [number, number, number, number][]): Float32Array {
  const data = new Float32Array(GEOMETRY.lowWidth * GEOMETRY.lowHeight).fill(-1);
  for (const [x0, x1, y0, y1] of boxes) {
    for (let y = y0; y <= y1; y += 1) {
      for (let x = x0; x <= x1; x += 1) data[y * GEOMETRY.lowWidth + x] = 1;
    }
  }
  return data;
}

/**
 * Six windows chosen so both drops are exercised: B duplicates A (IoU 0.818),
 * C is just below the threshold against A (0.667), D and E are specks that
 * fail both area gates, and F is a smaller mask that overlaps nothing enough
 * to be suppressed.
 */
const WINDOWS = {
  A: makeWindow([2, 11, 2, 9]),
  B: makeWindow([3, 12, 2, 9]),
  C: makeWindow([4, 13, 2, 9]),
  D: makeWindow([14, 14, 10, 10]),
  E: makeWindow([2, 2, 11, 11]),
  F: makeWindow([9, 14, 4, 9]),
};
const ORDER = ['A', 'B', 'C', 'D', 'E', 'F'] as const;
const ALL = ORDER.map((name) => WINDOWS[name]);

describe('lowResMinArea', () => {
  it('scales minMaskArea by the low-grid-to-image pixel ratio (AC7)', () => {
    // The 1024x649 playground sample: 100 * 65536 / 664576 = 9.861 -> 10.
    expect(lowResMinArea(100, 65536, 1024 * 649)).toBe(10);
  });

  it('rounds half up rather than truncating', () => {
    // 75 * 65536 / 131072 is exactly 37.5.
    expect(lowResMinArea(75, 65536, 131072)).toBe(38);
  });

  it('floors to 1 when the ratio rounds below one (AC7)', () => {
    // A 12 MP photo: 100 * 65536 / 12000000 = 0.546 -> 0 -> floored to 1.
    expect(lowResMinArea(100, 65536, 12_000_000)).toBe(1);
    expect(lowResMinArea(1, 65536, 12_000_000)).toBe(1);
  });

  it('still floors to 1 for minMaskArea 0, which is the one place the gates differ in kind', () => {
    // Documented consequence: with `minMaskArea: 0` the low-res path drops
    // zero-area masks that the full-res path would keep. Nothing downstream
    // wants an empty mask, but the asymmetry is real and pinned here.
    expect(lowResMinArea(0, 65536, 12_000_000)).toBe(1);
  });
});

describe('createFilterPlan', () => {
  it('gates and dedupes at 256x256 units on the low-res path', () => {
    const plan = lowPlan();
    // 100 * 256 / 768 = 33.33 -> 33.
    expect(plan.minArea).toBe(33);
    expect(plan.nmsWidth).toBe(GEOMETRY.lowWidth);
  });

  it('gates and dedupes at full resolution on the baseline path', () => {
    const plan = fullPlan();
    expect(plan.minArea).toBe(OPTIONS.minMaskArea);
    expect(plan.nmsWidth).toBe(GEOMETRY.originalWidth);
  });
});

describe('retainCandidate', () => {
  it('retains a COPY of the logit window, not a view into it (AC6)', () => {
    const plan = lowPlan();
    const source = Float32Array.from(WINDOWS.A);
    const retained = retainCandidate(source, plan);
    expect(retained.candidate).not.toBeNull();
    const expected = resolveSurvivor(retained.candidate!, plan);

    // Exactly what the worker does next: the decoder reuses `pred_masks`.
    source.fill(-999);

    expect(retained.candidate!.logits![0]).toBe(WINDOWS.A[0]);
    const after = resolveSurvivor(retained.candidate!, plan);
    expect(after).not.toBeNull();
    expect(after!.area).toBe(expected!.area);
    expect(Array.from(after!.coverage)).toEqual(Array.from(expected!.coverage));
  });

  it('throws when the window does not match the low grid', () => {
    expect(() => retainCandidate(new Float32Array(255), lowPlan())).toThrow(/256/);
  });

  it('reports the gate area in the units the path gates at', () => {
    expect(retainCandidate(WINDOWS.A, lowPlan()).gateArea).toBe(80);
    expect(retainCandidate(WINDOWS.A, fullPlan()).gateArea).toBe(320);
    expect(retainCandidate(WINDOWS.F, lowPlan()).gateArea).toBe(36);
    expect(retainCandidate(WINDOWS.F, fullPlan()).gateArea).toBe(144);
  });

  it('drops a speck under the pre-NMS gate on both paths', () => {
    // D is a single low-res pixel: 1 < 33 at low res, 4 < 100 at full res.
    expect(retainCandidate(WINDOWS.D, lowPlan()).candidate).toBeNull();
    expect(retainCandidate(WINDOWS.D, lowPlan()).gateArea).toBe(1);
    expect(retainCandidate(WINDOWS.D, fullPlan()).candidate).toBeNull();
    expect(retainCandidate(WINDOWS.D, fullPlan()).gateArea).toBe(4);
  });
});

describe('dedupeCandidates', () => {
  it('suppresses the duplicate at both resolutions, keeping the same set', () => {
    for (const plan of [lowPlan(), fullPlan()]) {
      const candidates = ALL.map((w) => retainCandidate(w, plan).candidate).filter(
        (candidate): candidate is MaskCandidate => candidate !== null,
      );
      // D and E already fell out at the gate, so the live set is A, B, C, F.
      expect(candidates).toHaveLength(4);
      // B duplicates A at IoU 0.818; C sits at 0.667 and survives.
      expect(dedupeCandidates(candidates, plan)).toEqual([0, 2, 3]);
    }
  });

  it('refuses to dedupe a released candidate rather than reading null', () => {
    const plan = lowPlan();
    const candidate = retainCandidate(WINDOWS.A, plan).candidate!;
    releaseCandidate(candidate);
    expect(() => candidateCoverage([candidate])).toThrow(/released/);
  });
});

describe('releaseRejected', () => {
  it('releases exactly the candidates NMS did not keep (AC9)', () => {
    const plan = lowPlan();
    const candidates = ALL.map((w) => retainCandidate(w, plan).candidate).filter(
      (candidate): candidate is MaskCandidate => candidate !== null,
    );
    const kept = dedupeCandidates(candidates, plan);
    expect(releaseRejected(candidates, kept)).toBe(1);
    expect(candidates[1].logits).toBeNull();
    expect(candidates[1].coverage).toBeNull();
    for (const index of kept) {
      expect(candidates[index].logits).not.toBeNull();
      expect(candidates[index].coverage).not.toBeNull();
    }
  });
});

describe('resolveSurvivor', () => {
  it('throws rather than returning a wrong mask for a released candidate', () => {
    const plan = lowPlan();
    const candidate = retainCandidate(WINDOWS.A, plan).candidate!;
    releaseCandidate(candidate);
    expect(() => resolveSurvivor(candidate, plan)).toThrow(/released/);
  });

  it('re-checks the exact, unscaled minMaskArea at full resolution (AC8)', () => {
    // A geometry where full resolution is 4x the low grid on both axes, so
    // the scaled pre-NMS gate is LOOSER than minMaskArea and a mask can pass
    // it and still fail the real one.
    const geometry: MaskGeometry = {
      lowWidth: 4,
      lowHeight: 4,
      padWidth: 4,
      padHeight: 4,
      reshapedWidth: 4,
      reshapedHeight: 4,
      originalWidth: 16,
      originalHeight: 16,
    };
    const window = new Float32Array(16).fill(-1);
    for (const p of [5, 6, 9, 10]) window[p] = 1;
    // Measured, not assumed: 4 low-res pixels become 60 full-res ones here.
    const fullArea = resampleThresholdMask({ logits: window, ...geometry, threshold: 0 }).area;
    const lowArea = thresholdMask(window, 0).area;

    const tooBig = createFilterPlan(geometry, { ...OPTIONS, minMaskArea: fullArea + 1 });
    // The candidate must genuinely PASS the pre-NMS gate, or this proves nothing.
    expect(tooBig.minArea).toBeLessThanOrEqual(lowArea);
    const retained = retainCandidate(window, tooBig);
    expect(retained.candidate).not.toBeNull();
    expect(resolveSurvivor(retained.candidate!, tooBig)).toBeNull();

    const exact = createFilterPlan(geometry, { ...OPTIONS, minMaskArea: fullArea });
    const kept = resolveSurvivor(retainCandidate(window, exact).candidate!, exact);
    expect(kept).not.toBeNull();
    expect(kept!.area).toBe(fullArea);
  });

  it('gives a mask kept by both paths byte-identical coverage (AC2 mechanism)', () => {
    for (const name of ['A', 'C', 'F'] as const) {
      const low = resolveSurvivor(retainCandidate(WINDOWS[name], lowPlan()).candidate!, lowPlan());
      const full = resolveSurvivor(retainCandidate(WINDOWS[name], fullPlan()).candidate!, fullPlan());
      expect(low).not.toBeNull();
      expect(full).not.toBeNull();
      expect(Array.from(low!.coverage)).toEqual(Array.from(full!.coverage));
    }
  });
});

describe('the baseline path', () => {
  it('reproduces the pre-change pipeline exactly (AC5)', () => {
    // The pipeline as it stood before this change, written out here because a
    // differential needs a baseline that is not the code under test.
    const resampled = ALL.map((w) =>
      resampleThresholdMask({ logits: w, ...GEOMETRY, threshold: OPTIONS.maskThreshold }),
    );
    const passed = resampled.filter((mask) => mask.area >= OPTIONS.minMaskArea);
    const reference = dedupeMasks(passed, OPTIONS.nmsIouThreshold, GEOMETRY.originalWidth).map(
      (index) => passed[index],
    );
    // Non-vacuous: the area gate drops two and NMS drops one, so an
    // implementation that skipped either gate would fail the comparison below.
    expect(passed).toHaveLength(4);
    expect(reference).toHaveLength(3);

    const plan = fullPlan();
    const candidates = ALL.map((w) => retainCandidate(w, plan).candidate).filter(
      (candidate): candidate is MaskCandidate => candidate !== null,
    );
    const returned = dedupeCandidates(candidates, plan)
      .map((index) => resolveSurvivor(candidates[index], plan))
      .filter((mask) => mask !== null);

    expect(returned).toHaveLength(reference.length);
    returned.forEach((mask, i) => {
      expect(mask!.area).toBe(reference[i].area);
      expect(Array.from(mask!.coverage)).toEqual(Array.from(reference[i].coverage));
    });
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/segmenter/core/mask-pipeline.test.ts`

Expected: FAIL — `Failed to resolve import "./mask-pipeline"`. Confirm you see a resolution failure, not a pass.

- [ ] **Step 3: Write the module**

Create `src/segmenter/core/mask-pipeline.ts`:

```ts
/**
 * The mask pipeline between `select` and `mask-encode`, as pure functions.
 *
 * The worker cannot be unit tested — `segmenter.worker.ts` binds
 * `globalThis.addEventListener` at module scope and imports
 * `@huggingface/transformers` — so every RULE the pipeline turns on lives
 * here instead, model-free and synchronous: the pre-NMS area gate, what a
 * retained candidate is, which resolution NMS runs at, and the
 * full-resolution re-check. The worker owns only the ORDER these are called
 * in, plus timing and the async PNG encode.
 *
 * Both pipelines live here, chosen by `FilterNmsOptions.lowResFilterNms`:
 *
 *   - `true` (the shipped pipeline): a chosen candidate is retained as a COPY
 *     of its 256x256 logit window plus a 256x256 binary mask, NMS runs on that
 *     low-res coverage, and only the survivors are resampled to full
 *     resolution — where the exact `minMaskArea` is applied.
 *   - `false` (the measurement baseline): every chosen candidate goes straight
 *     to full resolution, NMS runs there, and `resolveSurvivor` hands back the
 *     mask it already built.
 *
 * The baseline is kept for the same reason `dedupeMasksReference` is: so the
 * kept-set delta is measured rather than argued about. This is the one place
 * the old path stays.
 *
 * A survivor is resampled by `resampleFull` below on BOTH paths, with the same
 * arguments — that is the mechanical reason a mask both paths keep is
 * byte-for-byte identical, and it is what the Boundary tab asserts in a real
 * browser.
 */
import { thresholdMask, type BinaryMask } from './mask-postprocess';
import { resampleThresholdMask } from './mask-resample';
import { dedupeMasks } from './nms';

/** The mask decoder's output grid and the image geometry it maps onto. */
export interface MaskGeometry {
  /** Width of the decoder's logit grid, in samples (256 for SAM). */
  lowWidth: number;
  /** Height of the decoder's logit grid, in samples. */
  lowHeight: number;
  /** Width the processor padded the input to. */
  padWidth: number;
  /** Height the processor padded the input to. */
  padHeight: number;
  /** Width of the resized (pre-pad) image inside the padded square. */
  reshapedWidth: number;
  /** Height of the resized (pre-pad) image inside the padded square. */
  reshapedHeight: number;
  /** Width of the output mask, in pixels. */
  originalWidth: number;
  /** Height of the output mask, in pixels. */
  originalHeight: number;
}

/** The subset of `SegmenterOptions` this pipeline reads. */
export interface FilterNmsOptions {
  maskThreshold: number;
  minMaskArea: number;
  nmsIouThreshold: number;
  /** True for the shipped low-resolution pipeline; false for the baseline. */
  lowResFilterNms: boolean;
}

/**
 * One chosen candidate, retained between the batch loop and the survivor loop.
 *
 * Both fields are nullable and are MUTATED to null by `releaseCandidate`: the
 * arrays are the whole memory cost of the pipeline (320 KB per candidate on
 * the low-res path, `originalWidth * originalHeight` bytes on the baseline
 * one), and dropping the reference is how they become collectable while the
 * candidate array itself is still alive.
 */
export interface MaskCandidate {
  /** A COPY of the decoder's logit window. Null on the baseline path. */
  logits: Float32Array | null;
  /** The coverage NMS runs on: 256x256 on the low path, full-res on the baseline. */
  coverage: BinaryMask | null;
}

/** The two resolution-dependent decisions, resolved once per run. */
export interface FilterPlan {
  geometry: MaskGeometry;
  options: FilterNmsOptions;
  /** The pre-NMS area gate, in the units `retainCandidate` measures. */
  minArea: number;
  /** The width handed to `dedupeMasks`, matching the retained coverage. */
  nmsWidth: number;
}

export interface RetainResult {
  /** The area compared against `plan.minArea` — reported even when the candidate is dropped. */
  gateArea: number;
  /** Null when `gateArea < plan.minArea`: nothing was retained. */
  candidate: MaskCandidate | null;
}

/**
 * `minMaskArea` expressed in low-resolution pixels.
 *
 * CONVENTION, and it matters: `lowPixels` is the WHOLE decoder grid, including
 * the part that maps onto the processor's padding rather than onto the image.
 * On a non-square image only a fraction of the grid is ever sampled, so this
 * gate is proportionally STRICTER than `minMaskArea` — on a 1024x649 photo it
 * drops masks down to roughly 1.6x `minMaskArea` of true area. That coarseness
 * is deliberate (it is where F3's saving lives) and it is why the exact,
 * unscaled `minMaskArea` is re-applied at full resolution after NMS: the
 * RETURNED set is what the option documents, and this gate only decides what
 * NMS is allowed to consider.
 *
 * Floors at 1, so a huge image can never make the gate vacuous. That floor is
 * also the one place the two paths differ in kind: with `minMaskArea: 0` the
 * low-res path drops zero-area masks the baseline would keep.
 */
export function lowResMinArea(
  minMaskArea: number,
  lowPixels: number,
  originalPixels: number,
): number {
  return Math.max(1, Math.round((minMaskArea * lowPixels) / originalPixels));
}

export function createFilterPlan(
  geometry: MaskGeometry,
  options: FilterNmsOptions,
): FilterPlan {
  const lowPixels = geometry.lowWidth * geometry.lowHeight;
  const originalPixels = geometry.originalWidth * geometry.originalHeight;
  return {
    geometry,
    options,
    minArea: options.lowResFilterNms
      ? lowResMinArea(options.minMaskArea, lowPixels, originalPixels)
      : options.minMaskArea,
    nmsWidth: options.lowResFilterNms ? geometry.lowWidth : geometry.originalWidth,
  };
}

/**
 * The ONE full-resolution resample in this module. Both paths reach it with
 * identical arguments for the same logit window, which is what makes a mask
 * kept by both paths bit-identical.
 */
function resampleFull(logits: Float32Array, plan: FilterPlan): BinaryMask {
  return resampleThresholdMask({
    logits,
    ...plan.geometry,
    threshold: plan.options.maskThreshold,
  });
}

/**
 * Retain one chosen candidate, or report why it was dropped.
 *
 * `window` may be a `subarray` view into the decoder's reused `pred_masks`
 * buffer, so on the low-res path the retained logits are `window.slice()` — a
 * copy. `thresholdMask` and `resampleThresholdMask` both allocate their own
 * coverage, so nothing else here aliases the caller's buffer.
 */
export function retainCandidate(window: Float32Array, plan: FilterPlan): RetainResult {
  const { lowWidth, lowHeight } = plan.geometry;
  if (window.length !== lowWidth * lowHeight) {
    throw new Error(
      `retainCandidate: window.length ${window.length} does not match lowWidth * lowHeight ` +
        `(${lowWidth} * ${lowHeight} = ${lowWidth * lowHeight})`,
    );
  }

  if (!plan.options.lowResFilterNms) {
    const coverage = resampleFull(window, plan);
    return {
      gateArea: coverage.area,
      candidate: coverage.area >= plan.minArea ? { logits: null, coverage } : null,
    };
  }

  const coverage = thresholdMask(window, plan.options.maskThreshold);
  if (coverage.area < plan.minArea) return { gateArea: coverage.area, candidate: null };
  // Copied only once the gate has passed, so a rejected candidate costs no
  // 256 KB allocation at all.
  return { gateArea: coverage.area, candidate: { logits: window.slice(), coverage } };
}

/** The coverage NMS reads, with a released candidate failing loudly. */
export function candidateCoverage(candidates: readonly MaskCandidate[]): BinaryMask[] {
  return candidates.map((candidate, index) => {
    if (!candidate.coverage) {
      throw new Error(`mask-pipeline: candidate ${index} has been released`);
    }
    return candidate.coverage;
  });
}

/** Greedy keep-largest NMS at whatever resolution the candidates carry. */
export function dedupeCandidates(
  candidates: readonly MaskCandidate[],
  plan: FilterPlan,
): number[] {
  return dedupeMasks(candidateCoverage(candidates), plan.options.nmsIouThreshold, plan.nmsWidth);
}

/** Drop a candidate's retained buffers. Idempotent. */
export function releaseCandidate(candidate: MaskCandidate): void {
  candidate.logits = null;
  candidate.coverage = null;
}

/**
 * Release every candidate NMS did not keep. Returns how many were released,
 * so a caller can assert the pipeline actually let go of the ~94% it drops.
 */
export function releaseRejected(
  candidates: readonly MaskCandidate[],
  kept: readonly number[],
): number {
  const survivors = new Set(kept);
  let released = 0;
  for (let index = 0; index < candidates.length; index += 1) {
    if (survivors.has(index)) continue;
    releaseCandidate(candidates[index]);
    released += 1;
  }
  return released;
}

/**
 * One survivor at full resolution, or null when it fails the exact,
 * UNSCALED `minMaskArea`.
 *
 * On the baseline path this hands back the mask `retainCandidate` already
 * built, so the re-check is a tautology there and the coverage is the very
 * same object — which is what "the flag off reproduces the previous kept set"
 * means concretely.
 */
export function resolveSurvivor(candidate: MaskCandidate, plan: FilterPlan): BinaryMask | null {
  let mask: BinaryMask;
  if (plan.options.lowResFilterNms) {
    if (!candidate.logits) {
      throw new Error('mask-pipeline: resolveSurvivor called on a released candidate');
    }
    mask = resampleFull(candidate.logits, plan);
  } else {
    if (!candidate.coverage) {
      throw new Error('mask-pipeline: resolveSurvivor called on a released candidate');
    }
    mask = candidate.coverage;
  }
  return mask.area >= plan.options.minMaskArea ? mask : null;
}
```

- [ ] **Step 4: Export it from the core barrel**

In `src/segmenter/core/index.ts`, add a line after `export * from './mask-resample';`:

```ts
export * from './mask-pipeline';
```

- [ ] **Step 5: Run the tests and the typecheck**

```bash
npx vitest run src/segmenter/core/mask-pipeline.test.ts
npm run typecheck
```

Expected: all mask-pipeline tests PASS; typecheck clean. If any expected number disagrees with the code, STOP and report which one — the numbers were measured, so a mismatch means the module is wrong, not the test.

- [ ] **Step 6: Commit**

```bash
git add src/segmenter/core/mask-pipeline.ts src/segmenter/core/mask-pipeline.test.ts src/segmenter/core/index.ts
git commit -m "feat(segmenter): add the pure low-res filter/NMS pipeline"
```

---

### Task 2: Wire the worker to it, and re-shape the report

One task because the halves do not compile apart: `FILTER_SUBSTEP_ORDER` losing `'resample'` breaks the worker's `recordSub('resample')`, and `PHASE_ORDER` without a worker that records `resample` ships a permanently-zero row.

**Files:**
- Modify: `src/segmenter/core/types.ts` (lines 8-32, `SegmenterOptions` tail ~line 106, `DEFAULT_SEGMENTER_OPTIONS` ~line 131, `SegmentationCounts` lines 156-163)
- Modify: `src/segmenter/worker/segmenter.worker.ts` (header comment, imports, and the batch/nms/encode blocks)
- Modify: `src/segmenter/core/timing.test.ts` (lines 72-115)
- Modify: `src/segmenter/createSegmenter.test.ts` (targeted)
- Modify: `src/segmenter/worker/session-key.test.ts` (append one test)
- Modify: `playground/SegmentView.tsx` (line 336 area)

**Interfaces:**
- Consumes: everything Task 1 produced.
- Produces: `PHASE_ORDER = ['model-load','encode','decode','filter','nms','resample','mask-encode']`; `FILTER_SUBSTEP_ORDER = ['select','threshold']`; `SegmenterOptions.lowResFilterNms: boolean` (default `true`); `SegmentationCounts.returned: number`. All four are public through `note-scanner/segmenter`.

- [ ] **Step 1: Write the failing tests**

**(a)** In `src/segmenter/core/timing.test.ts`, in the `describe('createTimingAccumulator filter sub-steps', ...)` block, replace every `'resample'` sub-step with `'threshold'` — four occurrences in `recordFilterSub(...)` calls and six in `filterSubPhases.resample` reads, across the last three tests. Do NOT rewrite the file: the `PHASE_ORDER` tests above are unchanged, and the no-residual test's structure must survive intact. Concretely, in that block:
- `timings.recordFilterSub('resample', 10 | 30 | 20)` → `'threshold'`
- `report.filterSubPhases.resample.{count,p50,max,total}` → `.threshold.…`
- `expect(report.filterSubPhases.resample.total).toBe(0)` → `.threshold.total`
- in the no-residual test, `recordFilterSub('resample', 97)` / `('resample', 48)` → `'threshold'`

**(b)** In `src/segmenter/createSegmenter.test.ts`, add `returned` to every `counts` literal, keeping each one internally consistent with its `afterNms`:
- line 64 (`doneMessage`): `counts: { raw: 24, afterFilter: 9, afterNms: masks.length, returned: masks.length },`
- line 112: `expect(result.counts).toEqual({ raw: 24, afterFilter: 9, afterNms: 0, returned: 0 });`
- lines 133, 159, 323: `counts: { raw: 24, afterFilter: 9, afterNms: 0, returned: 0 },` (line 159 keeps `afterFilter: 0`, so: `{ raw: 24, afterFilter: 0, afterNms: 0, returned: 0 }`)
- line 185: `counts: { raw: 3, afterFilter: 2, afterNms: 2, returned: 2 },`
- line 208: `counts: { raw: 3, afterFilter: 1, afterNms: 1, returned: 1 },`

Then change the two `'resample'` sub-step reads in the `'carries the worker filterSubPhases through the rebuilt report'` and `'rebuilds a zero row for a sub-step the worker never recorded'` tests to `'threshold'` (`recordFilterSub('resample', 100)` → `'threshold'`; `filterSubPhases.resample.total` / `.p50` / `.count` → `.threshold.…`). Leave both tests' comments and every other assertion exactly as they are — the zero-fill assertion in the second test is the only coverage of the rebuild path's zero row.

Add this test at the end of the `describe('createSegmenter', ...)` block:

```ts
  it('defaults lowResFilterNms ON and passes an explicit false through (AC5)', async () => {
    const segmenter = createSegmenter({ createWorker: spawn });
    const first = segmenter.segment(fakeBitmap());
    // The one flag in DEFAULT_SEGMENTER_OPTIONS that is true: the low-res
    // pipeline is what ships, and the old path is the opt-in baseline.
    expect(FakeWorker.instances[0].posted[0].options.lowResFilterNms).toBe(true);
    FakeWorker.instances[0].emit(doneMessage());
    await first;

    const second = segmenter.segment(fakeBitmap(), { lowResFilterNms: false });
    expect(FakeWorker.instances[1].posted[0].options.lowResFilterNms).toBe(false);
    FakeWorker.instances[1].emit(doneMessage());
    await second;
  });

  it('reports the returned count separately from afterNms (AC11)', async () => {
    const segmenter = createSegmenter({ createWorker: spawn });
    const pending = segmenter.segment(fakeBitmap());
    FakeWorker.instances[0].emit({
      type: 'done',
      masks: [{ maskUrl: 'data:image/png;base64,aGk=', area: 1 }],
      width: 2,
      height: 1,
      timings: createTimingAccumulator().report(1),
      // Two masks survived NMS; the full-resolution area re-check dropped one.
      counts: { raw: 24, afterFilter: 9, afterNms: 2, returned: 1 },
    });

    const result = await pending;
    expect(result.counts.afterNms).toBe(2);
    expect(result.counts.returned).toBe(1);
    expect(result.segments).toHaveLength(1);
  });
```

**(c)** Append to `src/segmenter/worker/session-key.test.ts`, inside `describe('sessionCacheKey', ...)`:

```ts
  it('ignores lowResFilterNms, which changes nothing about the session (AC5)', () => {
    expect(sessionCacheKey(options({ lowResFilterNms: true }))).toBe(
      sessionCacheKey(options({ lowResFilterNms: false })),
    );
  });
```

- [ ] **Step 2: Run them to verify they fail**

```bash
npx vitest run src/segmenter/core/timing.test.ts src/segmenter/createSegmenter.test.ts src/segmenter/worker/session-key.test.ts
```

Expected: FAIL. `filterSubPhases.threshold` is `undefined` (TypeError reading `.count`/`.total`), `Object.keys(...)` comparisons mismatch, `options.lowResFilterNms` is `undefined`, and the `counts` deep-equals mismatch on the extra key. Confirm you see failures, not a pass.

- [ ] **Step 3: Change the types**

In `src/segmenter/core/types.ts`, replace lines 3-32 (the `PHASE_ORDER` comment through the `FilterSubstep` alias) with:

```ts
/**
 * The phases the segmenter times, in the order the playground's results table
 * renders them. Every one of them happens inside the worker — including
 * `mask-encode`, which used to run on the main thread.
 *
 * `resample` sits between `nms` and `mask-encode` because that is where the
 * survivor resample now happens: once per mask that SURVIVED dedup, fused with
 * the encode. Keeping it out of `filter` is what makes a before/after on
 * `filter` and `nms` read honestly instead of hiding the moved cost.
 */
export const PHASE_ORDER = [
  'model-load',
  'encode',
  'decode',
  'filter',
  'nms',
  'resample',
  'mask-encode',
] as const;

export type SegmentationPhase = (typeof PHASE_ORDER)[number];

/**
 * The internal split of the `filter` stage, in render order. Deliberately NOT
 * folded into `PHASE_ORDER`: `SegmentationPhase` is public API and also types
 * `SegmenterFailure.phase`, so widening it would admit values that can never
 * be thrown, and summing the results table's total column would count
 * `filter` twice.
 *
 * CONVENTION: `threshold` is the per-candidate RETENTION region, whatever that
 * costs on the path in force. With `lowResFilterNms` on it is a 256x256
 * binarize plus the scaled area gate; with it off it is the full-resolution
 * `resampleThresholdMask` the baseline path still does inside `filter`. The
 * two sub-steps tile the stage exactly either way.
 */
export const FILTER_SUBSTEP_ORDER = ['select', 'threshold'] as const;

export type FilterSubstep = (typeof FILTER_SUBSTEP_ORDER)[number];
```

In the same file, add this member at the END of `SegmenterOptions` (after `keepRawMasks`):

```ts
  /**
   * Carry each chosen candidate through the filter and NMS stages at the
   * decoder's native 256x256 instead of at full image resolution, and resample
   * only the survivors — fused with the PNG encode.
   *
   * DEFAULT TRUE. This is the shipped pipeline; the flag exists so the OLD one
   * survives as a measurement baseline, exactly as `dedupeMasksReference` does.
   * Turning it off restores the pre-change full-resolution path.
   *
   * It is not behaviour-preserving, and deliberately so. Two things can move:
   * IoU computed at 256x256 can flip a borderline dedupe decision, and the
   * scaled pre-NMS area gate (see `lowResMinArea`) is coarser than
   * `minMaskArea` and can drop a thin mask the full-resolution gate would have
   * kept. The RETURNED set is still exact against this option's documented
   * meaning, because `minMaskArea` is re-applied unscaled after the survivor
   * resample. `docs/measurements/` carries the measured delta.
   *
   * A retained candidate costs 256 KB of logits plus 64 KB of coverage,
   * independent of image size — where the old path cost `width * height` bytes
   * per candidate, which is 12.2 MB each on a 12 MP photo.
   *
   * Like `keepRawMasks`, and unlike `gpuResidentEmbeddings`, it changes nothing
   * about the ONNX sessions and must never enter the worker's session cache key.
   */
  lowResFilterNms: boolean;
```

Add `lowResFilterNms: true,` to `DEFAULT_SEGMENTER_OPTIONS`, after `keepRawMasks: false,`.

Replace the `SegmentationCounts` interface with:

```ts
export interface SegmentationCounts {
  /** Every mask the decoder produced, before any filtering. */
  raw: number;
  /** Survivors of the stability, best-of-three and pre-NMS area filters. */
  afterFilter: number;
  /** Survivors of NMS dedup. NOT the returned count — see `returned`. */
  afterNms: number;
  /**
   * Masks actually returned: NMS survivors that also passed the exact,
   * unscaled `minMaskArea` re-check at full resolution. Equal to `afterNms`
   * unless that re-check dropped something, which is precisely the case a
   * silently shrinking result set would otherwise hide.
   */
  returned: number;
}
```

- [ ] **Step 4: Rewire the worker**

In `src/segmenter/worker/segmenter.worker.ts`:

**(a)** Replace lines 11-16 of the header comment with:

```
 * Pipeline: load once -> encode the image once -> decode the prompt grid in
 * batches -> filter each batch at the decoder's native 256x256 -> dedupe every
 * candidate at 256x256 -> resample ONLY the survivors to full resolution,
 * fused with the PNG encode. Reaching full resolution before NMS is not a
 * detail: NMS discards ~94% of candidates, and carrying each one at
 * `width * height` bytes until then is hundreds of megabytes on a photo and
 * gigabytes on a large one.
```

**(b)** Replace the import block from `'../core'` (lines 26-43) with:

```ts
import {
  batchPoints,
  buildPointGrid,
  candidateCoverage,
  createFilterPlan,
  createTimingAccumulator,
  dedupeCandidates,
  encodeMaskPng,
  releaseCandidate,
  releaseRejected,
  resolveSurvivor,
  retainCandidate,
  stabilityScore,
  type EncodedMask,
  type FilterPlan,
  type FilterSubstep,
  type MaskCandidate,
  type NmsComparison,
  type RawMask,
  type SegmentationPhase,
  type SegmenterOptions,
  type SegmenterRequest,
  type SegmenterResponse,
} from '../core';
```

(`BinaryMask`, `dedupeMasks` and `resampleThresholdMask` stop being used here; `dedupeMasksReference` is still imported from `'../core/nms'` on line 47.)

**(c)** Replace `const candidates: BinaryMask[] = [];` (line 170) with:

```ts
    const candidates: MaskCandidate[] = [];
    /**
     * Resolved on the first batch, because `lowWidth`/`lowHeight` are read off
     * the decoder's tensor rather than hardcoded to 256. Null only when not one
     * batch ran, in which case `candidates` is empty too.
     */
    let plan: FilterPlan | null = null;
```

**(d)** Immediately after `const scores = iouScores.data as Float32Array;` (line 273), insert:

```ts

        // Assigned through a local const so the block below has a
        // non-nullable `FilterPlan` without leaning on narrowing a `let`.
        const batchPlan = (plan ??= createFilterPlan(
          {
            lowWidth,
            lowHeight,
            padWidth,
            padHeight,
            reshapedWidth,
            reshapedHeight,
            originalWidth,
            originalHeight,
          },
          {
            maskThreshold: options.maskThreshold,
            minMaskArea: options.minMaskArea,
            nmsIouThreshold: options.nmsIouThreshold,
            lowResFilterNms: options.lowResFilterNms,
          },
        ));
```

**(e)** Replace the whole `if (chosen.length > 0) { … recordSub('resample'); }` block (lines 301-321) with:

```ts
        if (chosen.length > 0) {
          // One call per chosen mask. On the shipped path this is a 256x256
          // binarize plus the scaled gate — no full-resolution buffer is
          // allocated anywhere in this loop.
          for (let k = 0; k < chosen.length; k += 1) {
            const retained = retainCandidate(
              logits.subarray(chosen[k] * lowPixels, (chosen[k] + 1) * lowPixels),
              batchPlan,
            );
            if (retained.candidate) candidates.push(retained.candidate);
          }
          recordSub('threshold');
        }
```

**(f)** Replace everything from `// ---- nms: once, across every batch.` (line 349) through the closing `);` of the `post(...)` call (line 432) with:

```ts
    // ---- nms: once, across every batch, on whatever resolution the
    // ---- candidates carry. Adjacent grid points land on the same object
    // ---- constantly, so this is where the count actually falls.
    //
    // All of it is inside `if (plan)` rather than behind a non-null assertion:
    // `plan` is null only for an empty prompt grid, and then `candidates` is
    // empty too and there is nothing to dedupe, resample or encode.
    const masks: EncodedMask[] = [];
    const rawMasks: RawMask[] = [];
    let afterNms = 0;
    let nmsComparison: NmsComparison | null = null;

    if (plan) {
      phase = 'nms';

      // The A/B, when asked for. Reference FIRST, over the identical candidate
      // array, so the comparison is against the same input and not a mutated one.
      let referenceKept: number[] | null = null;
      let referenceMs = 0;
      if (options.compareNms) {
        const referenceStarted = performance.now();
        referenceKept = dedupeMasksReference(
          candidateCoverage(candidates),
          options.nmsIouThreshold,
        );
        referenceMs = performance.now() - referenceStarted;
      }

      started = performance.now();
      const kept = dedupeCandidates(candidates, plan);
      elapsed = performance.now() - started;
      // Deliberately only the fast path: the reference's time travels in
      // nmsComparison and NOWHERE else, so the results table is never inflated
      // by the doubled work.
      timings.record('nms', elapsed);
      afterNms = kept.length;
      post({ type: 'progress', event: { phase: 'nms', done: 1, total: 1, ms: elapsed } });

      // Both are returned ascending, so element-wise equality is set equality.
      nmsComparison = referenceKept
        ? {
            referenceMs,
            fastMs: elapsed,
            identical:
              referenceKept.length === kept.length &&
              referenceKept.every((value, i) => value === kept[i]),
          }
        : null;

      // Every rejected candidate's retained buffers go NOW, before the loop
      // below allocates its first full-resolution mask.
      releaseRejected(candidates, kept);

      // ---- resample + mask-encode, fused. One survivor at a time, so full
      // ---- resolution exists for exactly one mask at a time — except under
      // ---- keepRawMasks, which retains each survivor's coverage on purpose.
      // ---- `phase` is assigned inside the loop so a resample throw surfaces
      // ---- as SegmenterFailure('resample', ...) and a CompressionStream
      // ---- failure as SegmenterFailure('mask-encode', ...).
      for (const index of kept) {
        const candidate = candidates[index];

        phase = 'resample';
        const resampleStarted = performance.now();
        const mask = resolveSurvivor(candidate, plan);
        timings.record('resample', performance.now() - resampleStarted);
        if (!mask) {
          // Passed the coarse pre-NMS gate, failed the exact full-resolution
          // `minMaskArea`. Visible as the gap between afterNms and returned.
          releaseCandidate(candidate);
          continue;
        }

        phase = 'mask-encode';
        const encodeStarted = performance.now();
        const maskUrl = await encodeMaskPng(mask.coverage, originalWidth, originalHeight);
        timings.record('mask-encode', performance.now() - encodeStarted);
        masks.push({ maskUrl, area: mask.area });

        // The surviving coverage, only when asked for. Every mask owns its own
        // ArrayBuffer — `thresholdMask` and `resampleThresholdMask` each
        // allocate a fresh one — so no buffer can appear twice in the transfer
        // list below (postMessage throws on a duplicate) and none is aliased.
        if (options.keepRawMasks) rawMasks.push({ coverage: mask.coverage, area: mask.area });

        // Dropped here rather than after the loop: 320 KB of logits plus
        // coverage per survivor, and nothing reads either again.
        releaseCandidate(candidate);
      }
    }

    // With keepRawMasks off this is empty and the call below is byte-for-byte
    // today's: `masks` is strings, and not one coverage buffer crosses the
    // worker boundary.
    const transfer = rawMasks.map((mask) => mask.coverage.buffer as ArrayBuffer);

    post(
      {
        type: 'done',
        masks,
        width: originalWidth,
        height: originalHeight,
        timings: timings.report(performance.now() - startedAt),
        counts: {
          raw: rawCount,
          afterFilter: candidates.length,
          afterNms,
          returned: masks.length,
        },
        ...(nmsComparison ? { nmsComparison } : {}),
        ...(options.keepRawMasks ? { rawMasks } : {}),
      },
      transfer.length > 0 ? transfer : undefined,
    );
```

- [ ] **Step 5: Show the returned count in the playground**

In `playground/SegmentView.tsx`, replace the two lines inside `<p data-testid="mask-counts">` (lines 336-338) with:

```tsx
            masks: {result.counts.raw} raw → {result.counts.afterFilter} after filter →{' '}
            {result.counts.afterNms} after dedup → <strong>{result.counts.returned}</strong>{' '}
            returned. The last number is the one that matters; a gap between the
            last two is the full-resolution area re-check dropping a mask.
```

- [ ] **Step 6: Run everything**

```bash
npm test
npm run typecheck
```

Expected: all PASS. Then confirm the two claims a grep can actually settle, anchored to the construct rather than to a bare word:

```bash
grep -n "recordSub('" src/segmenter/worker/segmenter.worker.ts
grep -n "timings.record('resample'" src/segmenter/worker/segmenter.worker.ts
```

Expected: the first prints exactly two lines, `recordSub('select')` and `recordSub('threshold')`; the second prints exactly one line, inside the survivor loop. If either count differs, the rewire is incomplete.

- [ ] **Step 7: Commit**

```bash
git add src/segmenter playground/SegmentView.tsx
git commit -m "feat(segmenter): filter and dedupe at 256x256, resample only survivors"
```

---

### Task 3: The Boundary tab and its browser check (AC2)

The tab and the spec are one task: a `data-testid` with no assertion reading it is a hook nothing uses, and plan-dictated browser code that has never actually run is this repo's recorded failure mode. This task ends with `npm run test:browser` executed for real.

**Files:**
- Create: `playground/boundary.ts`
- Create: `playground/boundary.test.ts`
- Create: `playground/BoundaryView.tsx`
- Create: `tests/browser/boundary.spec.ts`
- Modify: `playground/main.tsx`
- Modify: `playground/vite.config.ts` (line 8)
- Modify: `playwright.config.ts`

**Interfaces:**
- Consumes: `createFilterPlan`, `dedupeCandidates`, `releaseRejected`, `resolveSurvivor`, `retainCandidate`, `BinaryMask`, `FilterNmsOptions`, `MaskCandidate`, `MaskGeometry` from `../src/segmenter`.
- Produces: `BOUNDARY_GEOMETRY`, `BOUNDARY_OPTIONS`, `BOUNDARY_FIXTURES`, `runBoundaryPath(windows, lowResFilterNms): PathResult`, `differingPixels(a, b): number`, `type BoundaryFixture`, `type WindowOutcome`, `type PathResult` — all playground-local, nothing under `src/` learns they exist.

- [ ] **Step 1: Write the fixtures and the two-path runner**

Create `playground/boundary.ts`:

```ts
/**
 * The boundary invariant, made checkable with no GPU.
 *
 * `retainCandidate`, `dedupeCandidates` and `resolveSurvivor` are pure and
 * model-free, so both pipelines can be driven over synthetic 256x256 logit
 * windows in a plain browser tab: no WebGPU adapter, no ONNX download, no
 * network. That is what makes AC2 a real-engine assertion rather than an
 * eyeball over a photo.
 *
 * The fixtures are PROCEDURAL — smooth signed-distance fields generated here,
 * not committed binaries — so a reader can see exactly which shape stresses
 * which change.
 *
 * The geometry is a real one: a 512x333 image, resized so its longest side is
 * 1024 (giving 1024x666) and padded to a 1024x1024 square, decoded on a
 * 256x256 logit grid. Full resolution then works out to exactly 2x the low
 * grid on both axes, so a low box covering columns a..b covers full columns
 * 2a..2b+1 — which is why every expected number in `boundary.test.ts` can be
 * derived rather than guessed.
 */
import {
  createFilterPlan,
  dedupeCandidates,
  releaseRejected,
  resolveSurvivor,
  retainCandidate,
  type BinaryMask,
  type FilterNmsOptions,
  type MaskCandidate,
  type MaskGeometry,
} from '../src/segmenter';

// Re-exported so `BoundaryView` can name the type without reaching past this
// module into the package's own barrel.
export type { BinaryMask };

export const BOUNDARY_GEOMETRY: MaskGeometry = {
  lowWidth: 256,
  lowHeight: 256,
  padWidth: 1024,
  padHeight: 1024,
  reshapedWidth: 1024,
  reshapedHeight: 666,
  originalWidth: 512,
  originalHeight: 333,
};

/** The shipped defaults, minus the flag the two paths differ on. */
export const BOUNDARY_OPTIONS = {
  maskThreshold: 0,
  minMaskArea: 100,
  nmsIouThreshold: 0.7,
} as const;

function pathOptions(lowResFilterNms: boolean): FilterNmsOptions {
  return { ...BOUNDARY_OPTIONS, lowResFilterNms };
}

/**
 * Signed distance to an axis-aligned box, POSITIVE INSIDE. The box is given as
 * inclusive integer pixel bounds and inflated by half a pixel each way, so
 * pixels `a..b` land at distance >= 0.5 and pixel `a - 1` at -0.5 — a clean
 * ramp through zero exactly halfway between them.
 */
function boxSdf(x: number, y: number, a: number, b: number, c: number, d: number): number {
  const dx = Math.max(a - 0.5 - x, x - (b + 0.5));
  const dy = Math.max(c - 0.5 - y, y - (d + 0.5));
  const outside = Math.hypot(Math.max(dx, 0), Math.max(dy, 0));
  const inside = Math.min(Math.max(dx, dy), 0);
  return -(outside + inside);
}

/** The union of the listed boxes, as a 256x256 signed-distance logit window. */
function sdfWindow(...boxes: [number, number, number, number][]): Float32Array {
  const { lowWidth: w, lowHeight: h } = BOUNDARY_GEOMETRY;
  const data = new Float32Array(w * h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let value = -Infinity;
      for (const [a, b, c, d] of boxes) value = Math.max(value, boxSdf(x, y, a, b, c, d));
      data[y * w + x] = value;
    }
  }
  return data;
}

/** Eight 3px tines on a 24px pitch, hanging off a spine. */
function combBoxes(): [number, number, number, number][] {
  const boxes: [number, number, number, number][] = [[40, 215, 150, 165]];
  for (let i = 0; i < 8; i += 1) boxes.push([40 + i * 24, 42 + i * 24, 100, 149]);
  return boxes;
}

export interface BoundaryFixture {
  id: string;
  title: string;
  /** What this fixture exists to break, in one sentence. */
  stresses: string;
  /** One 256x256 logit window per prompt-point candidate. */
  windows: Float32Array[];
}

export const BOUNDARY_FIXTURES: BoundaryFixture[] = [
  {
    id: 'comb',
    title: 'fine-toothed comb',
    stresses:
      'thin structure: eight 3-pixel tines that a coarser round-trip would thin, merge or lose.',
    windows: [sdfWindow(...combBoxes())],
  },
  {
    id: 'bridge',
    title: 'one-pixel bridge',
    stresses:
      'connectivity: two blobs joined by a single-pixel isthmus, which either survives the resample or does not.',
    windows: [sdfWindow([30, 89, 98, 157], [166, 225, 98, 157], [90, 165, 127, 127])],
  },
  {
    id: 'pair',
    title: 'near-duplicate pair straddling the IoU threshold',
    stresses:
      'N1: three identical 100x100 boxes at offsets 0, 17 and 18. Against the first, IoU is 83/117 = 0.7094 (above the 0.7 threshold, suppressed) and 82/118 = 0.6949 (below it, kept).',
    windows: [
      sdfWindow([78, 177, 40, 139]),
      sdfWindow([95, 194, 40, 139]),
      sdfWindow([96, 195, 40, 139]),
    ],
  },
  {
    id: 'speck',
    title: 'speck below the scaled gate',
    stresses:
      'F3: a 6x6 box with a low-res area of 36 against a scaled gate of 38, whose full-resolution area of 144 clears minMaskArea 100 comfortably. The baseline keeps it; the low-res path never lets it reach NMS.',
    windows: [sdfWindow([125, 130, 125, 130])],
  },
];

export type WindowStatus = 'kept' | 'filtered' | 'suppressed' | 'undersized';

export interface WindowOutcome {
  /**
   * `filtered` — dropped by the pre-NMS area gate.
   * `suppressed` — dropped by NMS as a duplicate.
   * `undersized` — survived NMS, then failed the exact full-resolution minMaskArea.
   * `kept` — returned.
   */
  status: WindowStatus;
  /** The area the pre-NMS gate saw: 256x256 on the low path, full-res on the baseline. */
  gateArea: number;
  /** Full-resolution coverage. Non-null only when `status` is `kept`. */
  mask: BinaryMask | null;
}

export interface PathResult {
  /** One outcome per source window, in input order. */
  outcomes: WindowOutcome[];
  /** The pre-NMS gate this path applied, in `gateArea`'s units. */
  minArea: number;
}

/**
 * One fixture through one path, in the same order the worker calls these
 * functions in: retain every candidate, dedupe, release the rejected, resolve
 * each survivor.
 *
 * The one deliberate difference from the worker: every survivor's mask is kept
 * so the tab can render them side by side. The worker holds one at a time, and
 * that bound is the worker's to keep.
 */
export function runBoundaryPath(
  windows: readonly Float32Array[],
  lowResFilterNms: boolean,
): PathResult {
  const plan = createFilterPlan(BOUNDARY_GEOMETRY, pathOptions(lowResFilterNms));
  const outcomes: WindowOutcome[] = windows.map(() => ({
    status: 'filtered',
    gateArea: 0,
    mask: null,
  }));

  const live: MaskCandidate[] = [];
  const liveSource: number[] = [];
  windows.forEach((window, index) => {
    const retained = retainCandidate(window, plan);
    outcomes[index].gateArea = retained.gateArea;
    if (retained.candidate) {
      live.push(retained.candidate);
      liveSource.push(index);
      outcomes[index].status = 'suppressed';
    }
  });

  const kept = dedupeCandidates(live, plan);
  releaseRejected(live, kept);
  for (const index of kept) {
    const mask = resolveSurvivor(live[index], plan);
    const outcome = outcomes[liveSource[index]];
    outcome.status = mask ? 'kept' : 'undersized';
    outcome.mask = mask;
  }

  return { outcomes, minArea: plan.minArea };
}

/** Pixels covered by exactly one of the two masks. Both are full resolution. */
export function differingPixels(a: BinaryMask, b: BinaryMask): number {
  let diff = 0;
  for (let p = 0; p < a.coverage.length; p += 1) {
    if (a.coverage[p] !== b.coverage[p]) diff += 1;
  }
  return diff;
}
```

- [ ] **Step 2: Write the fixture test and run it**

Create `playground/boundary.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import {
  BOUNDARY_FIXTURES,
  BOUNDARY_GEOMETRY,
  differingPixels,
  runBoundaryPath,
  type BoundaryFixture,
} from './boundary';

function fixture(id: string): BoundaryFixture {
  const found = BOUNDARY_FIXTURES.find((candidate) => candidate.id === id);
  if (!found) throw new Error(`no fixture ${id}`);
  return found;
}

describe('the boundary fixtures', () => {
  it('gates at 38 low-res pixels, from a real 512x333 geometry', () => {
    // 100 * 65536 / 170496 = 38.438 -> 38.
    expect(runBoundaryPath(fixture('speck').windows, true).minArea).toBe(38);
    expect(runBoundaryPath(fixture('speck').windows, false).minArea).toBe(100);
  });

  it('keeps every mask both paths keep byte-for-byte identical (AC2)', () => {
    let keptByBoth = 0;
    for (const item of BOUNDARY_FIXTURES) {
      const baseline = runBoundaryPath(item.windows, false);
      const next = runBoundaryPath(item.windows, true);
      item.windows.forEach((_, index) => {
        const a = baseline.outcomes[index];
        const b = next.outcomes[index];
        if (a.status !== 'kept' || b.status !== 'kept') return;
        keptByBoth += 1;
        expect(differingPixels(a.mask!, b.mask!), `${item.id}[${index}]`).toBe(0);
      });
    }
    // Not vacuous: something has to be kept by both, or the loop above proves
    // nothing at all.
    expect(keptByBoth).toBe(4);
  });

  it('keeps the fine-toothed comb on both paths (AC2)', () => {
    const windows = fixture('comb').windows;
    const baseline = runBoundaryPath(windows, false).outcomes[0];
    const next = runBoundaryPath(windows, true).outcomes[0];
    expect(baseline.status).toBe('kept');
    expect(next.status).toBe('kept');
    // 4016 low-res pixels become 16064 at full resolution — the exact 4x the
    // 2x-per-axis mapping predicts.
    expect(next.gateArea).toBe(4016);
    expect(baseline.gateArea).toBe(16064);
    expect(next.mask!.area).toBe(16064);
  });

  it('shows the speck the scaled gate drops and the baseline returns (F3)', () => {
    const windows = fixture('speck').windows;
    const baseline = runBoundaryPath(windows, false).outcomes[0];
    const next = runBoundaryPath(windows, true).outcomes[0];
    expect(next.gateArea).toBe(36);
    expect(next.status).toBe('filtered');
    expect(baseline.gateArea).toBe(144);
    expect(baseline.status).toBe('kept');
    // The whole point: its true area clears minMaskArea 100 with room to spare.
    expect(baseline.gateArea).toBeGreaterThan(100);
  });

  it('suppresses the 0.7094 duplicate and keeps the 0.6949 one, on both paths (N1)', () => {
    const windows = fixture('pair').windows;
    for (const lowRes of [false, true]) {
      const statuses = runBoundaryPath(windows, lowRes).outcomes.map((o) => o.status);
      expect(statuses, `lowResFilterNms=${lowRes}`).toEqual(['kept', 'suppressed', 'kept']);
    }
  });

  it('keeps the one-pixel bridge on both paths', () => {
    const windows = fixture('bridge').windows;
    expect(runBoundaryPath(windows, false).outcomes[0].status).toBe('kept');
    expect(runBoundaryPath(windows, true).outcomes[0].status).toBe('kept');
  });

  it('renders masks at the image size, not the grid size', () => {
    const kept = runBoundaryPath(fixture('comb').windows, true).outcomes[0].mask!;
    expect(kept.coverage.length).toBe(
      BOUNDARY_GEOMETRY.originalWidth * BOUNDARY_GEOMETRY.originalHeight,
    );
  });
});
```

Run: `npx vitest run playground/boundary.test.ts`

Expected: all PASS. Every number above was measured against the real `resampleThresholdMask` before this plan was written; a failure means the fixture code drifted from the description, not that the numbers need editing. If one genuinely disagrees, STOP and report the observed value rather than relaxing the assertion.

- [ ] **Step 3: Build the tab**

Create `playground/BoundaryView.tsx`:

```tsx
/**
 * The Boundary tab: both mask pipelines, over procedural fixtures, with no
 * WebGPU adapter, no model download and no network. Everything renders
 * synchronously on mount — there is nothing to await.
 */
import { useEffect, useMemo, useRef } from 'react';
import {
  BOUNDARY_FIXTURES,
  BOUNDARY_GEOMETRY,
  BOUNDARY_OPTIONS,
  differingPixels,
  runBoundaryPath,
  type BinaryMask,
  type PathResult,
} from './boundary';

function MaskCanvas({
  baseline,
  next,
  testId,
}: {
  baseline: BinaryMask | null;
  next: BinaryMask | null;
  testId: string;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const { originalWidth: width, originalHeight: height } = BOUNDARY_GEOMETRY;

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const image = ctx.createImageData(width, height);
    for (let p = 0; p < width * height; p += 1) {
      const b = baseline?.coverage[p] ?? 0;
      const n = next?.coverage[p] ?? 0;
      // Blue: both. Orange: baseline only. Green: new only. Slate: neither.
      const rgb = b && n ? [37, 99, 235] : b ? [249, 115, 22] : n ? [22, 163, 74] : [17, 24, 39];
      const offset = p * 4;
      image.data[offset] = rgb[0];
      image.data[offset + 1] = rgb[1];
      image.data[offset + 2] = rgb[2];
      image.data[offset + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }, [baseline, next, width, height]);

  return (
    <canvas
      ref={ref}
      data-testid={testId}
      width={width}
      height={height}
      style={{ width: 256, imageRendering: 'pixelated', border: '1px solid #334155' }}
    />
  );
}

interface FixtureRun {
  baseline: PathResult;
  next: PathResult;
}

export function BoundaryView() {
  const runs = useMemo<FixtureRun[]>(
    () =>
      BOUNDARY_FIXTURES.map((fixture) => ({
        baseline: runBoundaryPath(fixture.windows, false),
        next: runBoundaryPath(fixture.windows, true),
      })),
    [],
  );

  const rows = runs.reduce((sum, run) => sum + run.baseline.outcomes.length, 0);
  const keptByBoth = runs.reduce(
    (sum, run) =>
      sum +
      run.baseline.outcomes.filter(
        (outcome, index) =>
          outcome.status === 'kept' && run.next.outcomes[index].status === 'kept',
      ).length,
    0,
  );

  return (
    <>
      <p data-testid="boundary-summary">
        {BOUNDARY_FIXTURES.length} procedural fixtures, {rows} candidate windows,{' '}
        {keptByBoth} kept by both paths. Baseline = <code>lowResFilterNms: false</code>{' '}
        (resample every candidate, dedupe at {BOUNDARY_GEOMETRY.originalWidth}x
        {BOUNDARY_GEOMETRY.originalHeight}); new = <code>true</code> (dedupe at{' '}
        {BOUNDARY_GEOMETRY.lowWidth}x{BOUNDARY_GEOMETRY.lowHeight}, resample the survivors).
        Gate: <code>minMaskArea</code> {BOUNDARY_OPTIONS.minMaskArea}, scaled to{' '}
        {runs[0].next.minArea} before NMS. No WebGPU, no model, no network.
      </p>

      {BOUNDARY_FIXTURES.map((fixture, f) => (
        <section key={fixture.id} data-testid={`boundary-fixture-${fixture.id}`}>
          <h2 style={{ fontSize: '1rem', marginBottom: 2 }}>{fixture.title}</h2>
          <p style={{ margin: '0 0 8px', opacity: 0.8 }}>{fixture.stresses}</p>
          <table style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th align="left">window</th>
                <th align="left">baseline</th>
                <th align="right">gate area</th>
                <th align="right">mask area</th>
                <th align="left">new</th>
                <th align="right">gate area</th>
                <th align="right">mask area</th>
                <th align="right">differing pixels</th>
                <th align="left">overlay</th>
              </tr>
            </thead>
            <tbody>
              {fixture.windows.map((_, index) => {
                const baseline = runs[f].baseline.outcomes[index];
                const next = runs[f].next.outcomes[index];
                const bothKept = baseline.status === 'kept' && next.status === 'kept';
                const diff = bothKept ? differingPixels(baseline.mask!, next.mask!) : null;
                return (
                  <tr
                    key={index}
                    data-testid={`boundary-window-${fixture.id}-${index}`}
                    data-baseline-status={baseline.status}
                    data-baseline-gate-area={baseline.gateArea}
                    data-baseline-mask-area={baseline.mask?.area}
                    data-new-status={next.status}
                    data-new-gate-area={next.gateArea}
                    data-new-mask-area={next.mask?.area}
                    data-diff={diff ?? undefined}
                  >
                    <td>#{index}</td>
                    <td>{baseline.status}</td>
                    <td align="right">{baseline.gateArea}</td>
                    <td align="right">{baseline.mask?.area ?? '—'}</td>
                    <td>{next.status}</td>
                    <td align="right">{next.gateArea}</td>
                    <td align="right">{next.mask?.area ?? '—'}</td>
                    <td align="right">
                      <strong>{diff ?? '—'}</strong>
                    </td>
                    <td>
                      <MaskCanvas
                        baseline={baseline.mask}
                        next={next.mask}
                        testId={`boundary-canvas-${fixture.id}-${index}`}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      ))}
    </>
  );
}
```

- [ ] **Step 4: Put it in the nav**

In `playground/main.tsx`:
- add `import { BoundaryView } from './BoundaryView';` after the `CompareView` import;
- change the view union to `type View = 'benchmark' | 'segment' | 'compare' | 'boundary';`;
- add this button after the Compare button, inside the same `<nav>` (keep the `{' '}` separator before it):

```tsx
        <button
          data-testid="view-boundary"
          type="button"
          aria-pressed={view === 'boundary'}
          onClick={() => setView('boundary')}
        >
          Boundary
        </button>
```
- add `{view === 'boundary' && <BoundaryView />}` after the CompareView line.

- [ ] **Step 5: Make the dev server's port deterministic and teach Playwright to start it**

In `playground/vite.config.ts`, change `server: { port: 5180 },` to:

```ts
  // strictPort so Playwright's `webServer.url` is the URL Vite actually
  // serves. `scripts/sweep-decode.mjs` passes `port: 0, strictPort: false`
  // inline, which overrides this, so the sweep still never fights a dev
  // server the developer already has running.
  server: { port: 5180, strictPort: true },
```

In `playwright.config.ts`, replace the file's doc comment and add `use` + `webServer`, leaving `testDir`, `fullyParallel`, `forbidOnly`, `reporter` and `projects` untouched:

```ts
/**
 * Browser-only verification, in three engines.
 *
 * Two specs with different needs. `mask-png.spec.ts` encodes its masks in Node
 * and hands them into the page as data URLs, so it needs nothing served.
 * `boundary.spec.ts` drives the running playground, so a `webServer` starts
 * Vite for it — which also serves the first spec harmlessly. Neither needs
 * WebGPU, a model download or the network: firefox and webkit expose no
 * `navigator.gpu` at all, which is exactly why the Boundary tab must not ask
 * for one.
 */
export default defineConfig({
  testDir: './tests/browser',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  reporter: process.env.CI ? 'github' : 'list',
  use: { baseURL: 'http://localhost:5180' },
  webServer: {
    command: 'npm run playground',
    url: 'http://localhost:5180/',
    // Reuse a dev server the developer already has up; in CI always start one.
    reuseExistingServer: !process.env.CI,
    // Vite pre-bundles @huggingface/transformers at startup (see
    // playground/vite.config.ts), which is slow the first time.
    timeout: 180_000,
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
  ],
});
```

- [ ] **Step 6: Write the browser spec**

Create `tests/browser/boundary.spec.ts`:

```ts
import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * AC2, in a real engine. The Boundary tab computes both pipelines
 * synchronously on mount, so there is nothing to poll — but every wait below
 * targets an element that renders visible TEXT, never an empty box.
 */
async function openBoundary(page: Page) {
  await page.goto('/');
  await page.click('[data-testid="view-boundary"]');
  await expect(page.getByTestId('boundary-summary')).toBeVisible();
}

async function statuses(row: Locator): Promise<[string | null, string | null]> {
  return [await row.getAttribute('data-baseline-status'), await row.getAttribute('data-new-status')];
}

test('every mask both paths keep renders a zero-pixel difference', async ({ page, baseURL }) => {
  const foreign: string[] = [];
  page.on('request', (request) => {
    if (!request.url().startsWith(baseURL!)) foreign.push(request.url());
  });

  await openBoundary(page);

  const rows = page.locator('[data-testid^="boundary-window-"]');
  const count = await rows.count();
  // Guards the loop below against passing because it iterated nothing.
  expect(count).toBeGreaterThan(0);

  let keptByBoth = 0;
  for (let i = 0; i < count; i += 1) {
    const row = rows.nth(i);
    const id = await row.getAttribute('data-testid');
    const [baseline, next] = await statuses(row);
    if (baseline !== 'kept' || next !== 'kept') continue;
    keptByBoth += 1;
    expect(await row.getAttribute('data-diff'), `${id} differs between the two paths`).toBe('0');
  }
  expect(keptByBoth).toBeGreaterThan(0);

  // No model, no CDN, no anything: everything this tab needs is its own code.
  expect(foreign, `the Boundary tab fetched ${foreign.join(', ')}`).toEqual([]);
});

test('keeps the fine-toothed comb on both paths', async ({ page }) => {
  await openBoundary(page);
  const row = page.getByTestId('boundary-window-comb-0');
  expect(await statuses(row)).toEqual(['kept', 'kept']);
  expect(await row.getAttribute('data-diff')).toBe('0');
});

test('shows the speck the scaled gate drops rather than hiding it', async ({ page }) => {
  await openBoundary(page);
  const row = page.getByTestId('boundary-window-speck-0');
  // Visible as a row with both statuses and both areas, not as an absence.
  expect(await statuses(row)).toEqual(['kept', 'filtered']);
  expect(await row.getAttribute('data-baseline-mask-area')).toBe('144');
  expect(await row.getAttribute('data-new-gate-area')).toBe('36');
});

test('renders the difference overlay for every window', async ({ page }) => {
  await openBoundary(page);
  const rows = await page.locator('[data-testid^="boundary-window-"]').count();
  await expect(page.locator('[data-testid^="boundary-canvas-"]')).toHaveCount(rows);
});
```

- [ ] **Step 7: Run everything, including the real browsers**

```bash
npm test
npm run typecheck
npm run test:browser
```

Expected: vitest and typecheck clean; `npm run test:browser` PASSES in all three projects (chromium, webkit, firefox) — 4 new boundary tests per project on top of the existing mask-png ones. If Playwright browsers are not installed, run `npx playwright install` first. If the run fails on the webServer, report the exact error rather than deleting the `webServer` block: CI runs this lane.

- [ ] **Step 8: Commit**

```bash
git add playground/boundary.ts playground/boundary.test.ts playground/BoundaryView.tsx \
        playground/main.tsx playground/vite.config.ts playwright.config.ts tests/browser/boundary.spec.ts
git commit -m "test(playground): add a Boundary tab and its three-engine boundary check"
```

---

### Task 4: Make the delta measurable with the machinery that already exists (AC12)

No new comparison code. `compareMaskSets` and `keepRawMasks` already do the work; this task only gives the flag a control, a grid axis and two report columns.

**Files:**
- Modify: `playground/compare.ts`
- Modify: `playground/compare.test.ts`
- Modify: `playground/CompareView.tsx`
- Modify: `playground/CompareView.test.tsx`
- Modify: `scripts/sweep-decode.mjs` (two lines)

**Interfaces:**
- Consumes: `SegmentationCounts.returned` and `SegmenterOptions.lowResFilterNms` from Task 2.
- Produces: `RowOptions.lowResFilterNms: boolean`; `SweepConfig.lowResFilterNms: readonly boolean[]` (default `[true]`); row ids of the form `p16-fp32-b8-none-lowres-r1`; a `low-res-filter-nms` checkbox on the Compare tab.

- [ ] **Step 1: Write the failing tests**

In `playground/compare.test.ts`:

**(a)** add `lowResFilterNms: true,` to the `rowOptions` defaults (after `keepRawMasks: false,`), and `returned: 11,` to `okRecord`'s counts so it reads `counts: { raw: 96, afterFilter: 31, afterNms: 12, returned: 11 },` — deliberately different from `afterNms`, so the two columns cannot be confused for one another.

**(b)** update the two id assertions in `'expands in a deterministic order with stable, unique ids (AC6)'`:

```ts
    expect(first[0]).toBe('p16-fp32-b8-none-lowres-r1');
    expect(first[15]).toBe('p16-fp16-b32-both-lowres-r1');
```

**(c)** update the two `toMarkdown` row assertions in `'renders options, every phase, counts and budget for each row (AC8)'`:

```ts
    expect(md).toContain('| both | fp16 | 32 | 16 | lowres |');
    // counts.raw / afterFilter / afterNms / returned, in that order.
    expect(md).toContain('| 96 | 31 | 12 | 11 |');
```
and add `'resample'` and `'returned'` to that test's list of column names.

**(d)** append these tests, the first two inside `describe('expandGrid', ...)`'s block (the one holding the id tests) and the last inside `describe('toMarkdown', ...)`:

```ts
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
```

```ts
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
```

In `playground/CompareView.test.tsx`: add `returned: 12,` to `stubResult`'s counts, and append this test to the file's main `describe`:

```ts
  it('defaults the lowResFilterNms control on and passes it through (AC12)', async () => {
    const { seen } = mount(stubResult(1000));
    const control = screen.getByTestId('low-res-filter-nms') as HTMLInputElement;
    expect(control.checked).toBe(true);

    fireEvent.click(control);
    fireEvent.click(screen.getByTestId('run-row'));
    await waitFor(() => expect(seen).toHaveLength(1));
    expect(seen[0].lowResFilterNms).toBe(false);
  });
```

Run: `npx vitest run playground/compare.test.ts playground/CompareView.test.tsx`
Expected: FAIL — unknown property `lowResFilterNms`, missing ids, no `low-res-filter-nms` element.

- [ ] **Step 2: Widen `RowOptions` and `SweepConfig`**

In `playground/compare.ts`:

Add to `RowOptions`, after `keepRawMasks: boolean;`:

```ts
  lowResFilterNms: boolean;
```

Add to `SweepConfig`, after `pointsPerSide: readonly number[];`:

```ts
  /**
   * The pipeline axis. `[true]` by default: the default grid measures the
   * SHIPPED pipeline and stays 16 rows. `[false, true]` walks a before/after
   * pair — baseline first, so the Compare tab retains the PRE-change mask set
   * as its agreement baseline.
   */
  lowResFilterNms: readonly boolean[];
```

Add `lowResFilterNms: [true],` to `DEFAULT_SWEEP_CONFIG` (after `pointsPerSide: [16],`).

Add to `resolveConfig`, beside the other checks:

```ts
  if (config.lowResFilterNms.length === 0) throw new Error('lowResFilterNms must not be empty');
  for (const value of config.lowResFilterNms) {
    if (typeof value !== 'boolean') {
      throw new Error(`lowResFilterNms must be booleans, got ${JSON.stringify(value)}`);
    }
  }
```

In `expandGrid`, wrap the rep loop in the new axis and extend the id:

```ts
        for (const path of config.decodePaths) {
          for (const lowResFilterNms of config.lowResFilterNms) {
            for (let rep = 1; rep <= config.reps; rep += 1) {
              rows.push({
                id: `p${pointsPerSide}-${dtype}-b${batchSize}-${path}-${
                  lowResFilterNms ? 'lowres' : 'fullres'
                }-r${rep}`,
                rep,
                warmUp: false,
                options: {
                  dtype,
                  batchSize,
                  pointsPerSide,
                  keepRawMasks: config.keepRawMasks,
                  lowResFilterNms,
                  ...decodeFlags(path),
                },
              });
            }
          }
        }
```

Extend `rowLabel`'s return so a rendered row says which pipeline it measured:

```ts
  return `${decodePathOf(o)} · ${o.dtype} · batch ${o.batchSize} · pps ${o.pointsPerSide} · ${
    o.lowResFilterNms ? 'lowres' : 'fullres'
  }`;
```

- [ ] **Step 3: Give the report its new columns and a name-driven alignment row**

In `playground/compare.ts`, replace the `COLUMNS` constant with:

```ts
const COLUMNS = [
  'rank', 'row', 'decode path', 'dtype', 'batch', 'pps', 'lowres', 'budget', 'total',
  'model-load', 'encode', 'decode', 'filter', 'nms', 'resample', 'mask-encode',
  'raw', 'afterFilter', 'afterNms', 'returned', 'status',
] as const;

/**
 * Text columns. Derived by NAME rather than by index: a positional rule
 * silently misaligns the whole table the moment a column is inserted.
 */
const LEFT_ALIGNED: ReadonlySet<string> = new Set([
  'rank', 'row', 'decode path', 'dtype', 'batch', 'pps', 'lowres', 'status',
]);
```

Replace the alignment line with:

```ts
  lines.push(`| ${COLUMNS.map((column) => (LEFT_ALIGNED.has(column) ? '---' : '---:')).join(' | ')} |`);
```

In the row loop, insert three cells so the array matches `COLUMNS` one for one:
- after `String(o.pointsPerSide),` → `o.lowResFilterNms ? 'lowres' : 'fullres',`
- after `ms(p?.nms.total),` → `ms(p?.resample.total),`
- after the `afterNms` cell → `record.counts ? String(record.counts.returned) : DASH,`

Replace the closing summary paragraph's last sentence with:

```ts
      `${PHASE_ORDER.length} phases. \`raw\`/\`afterFilter\`/\`afterNms\`/\`returned\` are mask ` +
      `counts: a row that is fast because it silently dropped masks is visible here, and a gap ` +
      `between \`afterNms\` and \`returned\` is the full-resolution area re-check.`,
```

- [ ] **Step 4: Give the Compare tab the control**

In `playground/CompareView.tsx`:
- add `lowResFilterNms: true,` to `INITIAL_OPTIONS` (after `keepRawMasks: false,`);
- add this control immediately after the `keep raw masks` label, keeping the `{' '}` separator pattern:

```tsx
        <label>
          low-res filter/NMS:{' '}
          <input
            data-testid="low-res-filter-nms"
            type="checkbox"
            checked={options.lowResFilterNms}
            disabled={running}
            onChange={(e) => patch({ lowResFilterNms: e.target.checked })}
          />
        </label>
```
- in the results table, add `<th align="left">lowres</th>` after the `pps` header and `<th align="right">returned</th>` after `kept`, with the matching cells `<td>{o.lowResFilterNms ? 'lowres' : 'fullres'}</td>` and `<td align="right">{record.counts?.returned ?? '—'}</td>`. Both read `record.options` / `record.counts` — the captured run, never current control state.

- [ ] **Step 5: Let the sweep drive it**

In `scripts/sweep-decode.mjs`, inside `runOneRow`, after the `keep-raw-masks` line:

```js
  await setCheckbox(page, 'low-res-filter-nms', o.lowResFilterNms);
```

and change the per-row success log to report what was actually returned:

```js
          ? `${record.budgetMs.toFixed(0)} ms budget, ${record.counts.returned} masks`
```

The new axis needs no entry in `assertConfigMatchesTabControls`: it is a checkbox, and `setChecked` accepts either boolean. `resolveConfig` (Step 2) already rejects a non-boolean before any GPU time is spent.

- [ ] **Step 6: Run everything**

```bash
npm test
npm run typecheck
node --check scripts/sweep-decode.mjs
```

Expected: all PASS. Then confirm no second comparison implementation crept in (AC12) — anchored to the definition, not to a mention:

```bash
grep -rn "function compareMaskSets\|function pairwiseIoU" src playground scripts
```

Expected: exactly two lines — `pairwiseIoU` in `src/segmenter/core/nms.ts` and `compareMaskSets` in `playground/compare.ts`.

- [ ] **Step 7: Commit**

```bash
git add playground scripts/sweep-decode.mjs
git commit -m "feat(playground): expose lowResFilterNms to the Compare tab and the sweep"
```

---

### Task 5: Measure the delta on a real GPU and commit it (AC1, AC3, AC4, AC11)

This task produces evidence, not code. It needs a machine with a real WebGPU adapter; the sweep refuses to measure a software rasterizer, and that refusal is correct. **If no hardware adapter is available, STOP and report it — do not hand-write, estimate or carry over numbers.**

**Files:**
- Create: `docs/measurements/2026-08-31-lowres-agreement.config.json`
- Create: `docs/measurements/2026-08-31-lowres-timing.config.json`
- Create: `docs/measurements/2026-08-31-lowres-filter-nms.md` (the analysis)
- Create (by the runner): two `docs/measurements/<date>-decode-sweep*.{md,json}` pairs

- [ ] **Step 1: Write the two grid configs**

`docs/measurements/2026-08-31-lowres-agreement.config.json`:

```json
{
  "decodePaths": ["none"],
  "batchSizes": [8],
  "dtypes": ["fp32"],
  "pointsPerSide": [16],
  "lowResFilterNms": [false, true],
  "keepRawMasks": true,
  "reps": 1
}
```

`docs/measurements/2026-08-31-lowres-timing.config.json`:

```json
{
  "decodePaths": ["none"],
  "batchSizes": [8],
  "dtypes": ["fp32"],
  "pointsPerSide": [16, 32],
  "lowResFilterNms": [false, true],
  "keepRawMasks": false,
  "reps": 1
}
```

Both pin the shipped defaults (`fp32`, batch 8, no decode-path flags) so the only thing moving between rows is the pipeline — and, in the timing grid, the grid density.

- [ ] **Step 2: Run the agreement sweep (AC1)**

```bash
npm run sweep:decode -- --config docs/measurements/2026-08-31-lowres-agreement.config.json
```

Headed Chromium opens; leave it alone until it prints `wrote docs/measurements/…`. Three runs happen: a warm-up (a copy of row 1, so `lowResFilterNms: false`) and the two measured rows.

Note how the agreement numbers arise, because it decides how they are read: `CompareView` retains the FIRST mask set it sees as its baseline, which is the warm-up's — same options as the `fullres` row. So the report's agreement table has two rows: `fullres` against that baseline, which is a CONTROL (same pipeline twice; expect it to match essentially perfectly), and `lowres` against it, which is AC1's actual measurement. Record both.

- [ ] **Step 3: Run the timing sweep (AC3)**

```bash
npm run sweep:decode -- --config docs/measurements/2026-08-31-lowres-timing.config.json
```

Four rows: `pointsPerSide` 16 and 32, each `fullres` then `lowres`. The `pointsPerSide: 32` fullres row is the slowest thing in this plan — issue #1 measured that operating point at ~166 s — so budget for it.

- [ ] **Step 4: Check the sub-timer invariant against the real run (AC4)**

Against the timing sweep's JSON (substitute the filename the runner printed):

```bash
node -e '
const data = JSON.parse(require("node:fs").readFileSync(process.argv[1], "utf8"));
for (const row of data.rows) {
  if (row.status !== "ok") { console.log(row.rowId, "FAILED", row.message); continue; }
  const sub = row.timings.filterSubPhases;
  const keys = Object.keys(sub).join(",");
  const sum = sub.select.total + sub.threshold.total;
  const filter = row.timings.phases.filter.total;
  console.log(
    row.rowId, "subphases:", keys,
    "select", sub.select.total.toFixed(1), "threshold", sub.threshold.total.toFixed(1),
    "sum", sum.toFixed(4), "filter", filter.toFixed(4), "residual", (filter - sum).toFixed(6),
    "| afterNms", row.counts.afterNms, "returned", row.counts.returned,
  );
}
' docs/measurements/<the timing sweep>.json
```

(`node -e` evaluates as CommonJS regardless of `"type": "module"`, so `require` is available.)

Copy the printed residuals into the analysis doc verbatim. Both sub-steps must be reported on every row and `keys` must read `select,threshold`.

- [ ] **Step 5: Write the analysis**

Create `docs/measurements/2026-08-31-lowres-filter-nms.md`. It must NAME the two generated artifacts it draws from and contain, with real numbers copied from them:

1. **Setup** — adapter identity (from the sweep's own header), image, options, and the two config filenames. State plainly that the first run in each sweep was a discarded warm-up, and that the agreement baseline is that warm-up's mask set.
2. **AC1 — kept-set delta.** For `fullres` vs `lowres`: `afterNms` and `returned` on each side, then `compareMaskSets`' matched / unmatched-baseline / unmatched-variant counts and mean / median / min IoU over the matched pairs. Report the `fullres`-vs-baseline control row beside it, and say what it demonstrates about the machinery. Then a paragraph of PROSE judging whether the delta is acceptable and why. Two effects belong in that judgement and must be named:
   - the scaled pre-NMS gate is computed over the WHOLE 256x256 grid including the padded region, so on a non-square image it is stricter than `minMaskArea` by roughly `max(w,h)/min(w,h)` — on the 1024x649 sample, about 1.6x. Masks between `minMaskArea` and that bound are dropped before NMS. Say whether the observed unmatched-baseline count is consistent with that.
   - IoU at 256x256 can flip a borderline dedupe decision. Say whether the numbers show it happening.
   A record that says only "a measurement was taken" does not satisfy AC1.
3. **AC3 — where the time went.** A table of `filter`, `nms`, `resample` and `mask-encode` stage totals plus `budget`, `fullres` vs `lowres`, at `pointsPerSide` 16 AND 32. Then prose: what got faster, what got slower, and where time MOVED rather than disappeared — `resample` is new and is exactly the cost `filter` used to carry — with an explicit verdict on whether this is an acceptable outcome.
4. **AC4** — the residuals from Step 4, per row, and the statement that `FILTER_SUBSTEP_ORDER` is `['select','threshold']` and both sub-steps are reported.
5. **AC11** — `afterNms` vs `returned` per row, and what any gap was.
6. **Caveats, in this file** — not in a commit message. At minimum: the discarded warm-up and what it protects; one rep per row, so single-run noise is not quantified; and any number whose reading a caveat changes must carry that caveat in the same section it appears in.

Every figure in the prose must be derived from the committed artifacts — including hedges. If you want to write "roughly 2x", compute the ratio first.

- [ ] **Step 6: Commit**

```bash
git add docs/measurements
git commit -m "docs(measurements): quantify the low-res filter/NMS delta on a real GPU"
```

---

## Acceptance criteria coverage

| AC | Where it is satisfied | How it can fail |
|---|---|---|
| AC1 | Task 5 Step 2 + the analysis doc's section 2 | The prose judges the delta with real `compareMaskSets` numbers, or the criterion is not met. |
| AC2 (ui) | Task 3: `boundary.test.ts` (`keptByBoth` = 4, comb kept by both) and `tests/browser/boundary.spec.ts` in chromium/webkit/firefox | A non-zero `data-diff` on any row both paths keep; a comb row that is not `kept/kept`; any request off the dev-server origin. |
| AC3 | Task 5 Step 3 + section 3 | Missing either operating point, or no verdict. |
| AC4 | Task 2 (`FILTER_SUBSTEP_ORDER` + `timing.test.ts`'s no-residual test) and Task 5 Step 4 on real data | A residual, or a sub-step missing from a real run's report. |
| AC5 | Task 1's `'reproduces the pre-change pipeline exactly'`; Task 2's `createSegmenter` and `session-key` tests | The flag missing from the worker request, present in the session key, or the baseline path diverging from the inline reference. |
| AC6 | Task 1's `'retains a COPY of the logit window'` — the source is overwritten after retention | A `subarray` view instead of `slice()` makes the post-mutation mask differ. |
| AC7 | Task 1's `lowResMinArea` and `createFilterPlan` tests, incl. the floor and the exact-.5 rounding case | Wrong formula, truncation instead of rounding, or a missing floor. |
| AC8 | Task 1's `'re-checks the exact, unscaled minMaskArea'`, on a geometry where the scaled gate is genuinely looser | Applying the scaled gate twice, or skipping the re-check. |
| AC9 | Task 1's `releaseRejected` test; the worker's single-survivor loop (review + Task 5's memory behaviour) | Rejected candidates keeping their buffers; a survivor array of full-res masks. |
| AC10 | Task 2's `PHASE_ORDER` change and the `timings.record('resample'` grep; `phase = 'resample'` set inside the survivor loop | `resample` time landing in `filter`, or a throw attributed to the wrong phase. |
| AC11 | Task 2's `'reports the returned count separately from afterNms'`; the `returned` column in Task 4; Task 5 section 5 | `returned` mirroring `afterNms` unconditionally. |
| AC12 | Task 4, incl. the `grep` for a second `compareMaskSets`/`pairwiseIoU` definition | A duplicated comparison, a widened default grid, or a control the sweep cannot drive. |

## Notes for the reviewer

- **`FILTER_SUBSTEP_ORDER` narrows again** (`['select','resample']` → `['select','threshold']`), and `PHASE_ORDER` widens with `resample`. Both are public through `note-scanner/segmenter`; both belong in the PR description as deliberate API changes, as the F2 change did.
- **Sweep row ids change shape** (`…-none-r1` → `…-none-lowres-r1`). Deliberate: a row must say which pipeline it measured, and the label is derived from the options the row was captured with.
- **`lowResFilterNms` is the only `true` in `DEFAULT_SEGMENTER_OPTIONS`.** That is the point: the low-res pipeline ships, and the old path exists to be measured against.
