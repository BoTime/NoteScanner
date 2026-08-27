# NMS bbox prefilter + bit-packed coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `dedupeMasks` reject most candidate pairs on their bounding boxes and compare the rest 32 pixels at a time, exactly — same kept set as today — and ship an opt-in A/B panel that proves it on a real image.

**Architecture:** `dedupeMasks(masks, iouThreshold, width)` gains a required `width` and, before the greedy loop, turns every candidate's `Uint8Array` coverage into a packed `Uint32Array` plus a 2-D bbox. The greedy loop's structure and tie-breaking are untouched; only the per-pair test changes — bbox-disjoint returns 0 without touching a pixel, otherwise `a & b` + SWAR popcount over just the words spanned by the two bboxes' shared row band. The current byte-wise body survives verbatim as `dedupeMasksReference` so "identical to baseline" is a real assertion in CI and a live A/B in the playground.

**Tech Stack:** TypeScript, vitest (node environment), React 19 (playground only), `@huggingface/transformers` (worker only, not exercised by tests).

**Spec:** `docs/superpowers/specs/2026-08-27-nms-bbox-prefilter-bit-packed-design.md`

## Global Constraints

- **N2 and N3 are exact.** No approximation, no tolerance, anywhere. Every assertion comparing `dedupeMasks` to `dedupeMasksReference` is exact array equality.
- **`pairwiseIoU` does not change.** Same two-argument signature, same byte-wise body. It stays public API.
- **`dedupeMasks` takes `width` as a REQUIRED third parameter.** No optional-width fallback, no overload, no default. A quiet slow mode is worse than a compile error.
- **The greedy loop's structure and tie-breaking are unchanged:** descending area, ties broken by lower original index, kept indices returned ascending, and the duplicate test stays `> iouThreshold` **strictly** (IoU exactly at the threshold survives).
- **N4 (the area-ratio early exit) is out of scope.** Do not implement it.
- `RawMask.coverage`, `thresholdMask`, `mask-encode`, and the worker message contract stay as they are apart from the one new optional `nmsComparison` field. Nothing is packed at threshold time.
- `dedupeMasksReference` and `popcount` are importable from `./nms` but must **not** be exported from `note-scanner/segmenter`.
- `timings.record('nms', ...)` and the `nms` progress event report **only** the fast path's elapsed time, even when `compareNms` is on.
- `compareNms` defaults to `false` in `DEFAULT_SEGMENTER_OPTIONS`.
- Nothing in the A/B path may be keyed to a particular `pointsPerSide` or `batchSize`.
- Out of scope: producing the real-image before/after numbers (AC10 — done by hand after merge), adding `@playwright/test`, moving NMS to 256x256 (issue #6), touching the worker message contract further (issue #5).

## Verification scope — read this before citing a command

- `npm run test` runs vitest over `src/**/*.test.ts(x)` and `playground/**/*.test.ts`.
- `npm run typecheck` (`tsc --noEmit`) covers **`src/**` only**. `tsconfig.json` lists `playground` in `exclude`, so **nothing under `playground/` is typechecked by `npm run typecheck`.** The only command in this plan that typechecks `playground/SegmentView.tsx` is the standalone `npx tsc` invocation given in Task 2 Step 9, which has been run against the current tree and exits 0.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/segmenter/core/nms.ts` | Modify | Adds `popcount`, `dedupeMasksReference`, a module-private `PreparedMask` + `prepareMask` + `packedIoU`; rewrites `dedupeMasks`'s per-pair test. `pairwiseIoU` untouched. |
| `src/segmenter/core/nms.test.ts` | Modify | Every existing case, plus popcount units, bbox-disjoint, non-multiple-of-32 tail, and the seeded differential. |
| `src/segmenter/core/index.ts` | Modify (line 4) | `export * from './nms'` narrows to an explicit named list so the slow reference and `popcount` stay off the public surface. |
| `src/segmenter/core/types.ts` | Modify | `NmsComparison`; `SegmenterOptions.compareNms`; the default; `SegmentationResult.nmsComparison?`; the `done` response's `nmsComparison?`. |
| `src/segmenter/worker/segmenter.worker.ts` | Modify (imports, nms block ~lines 250-280) | Passes `originalWidth`; runs the reference first when asked; records only the fast time; posts `nmsComparison`. |
| `src/segmenter/createSegmenter.ts` | Modify (~line 131-146) | Surfaces `nmsComparison` on `SegmentationResult`. |
| `src/segmenter/createSegmenter.test.ts` | Modify | `compareNms` passthrough and `nmsComparison` surviving the boundary. |
| `playground/SegmentView.tsx` | Modify | `compare-nms` checkbox and the A/B result row. |

No new files.

---

### Task 1: Exact bbox-prefiltered, bit-packed `dedupeMasks`

**Files:**
- Modify: `src/segmenter/core/nms.ts`
- Modify: `src/segmenter/core/index.ts:4`
- Modify: `src/segmenter/worker/segmenter.worker.ts:254` (call site only — the required third parameter breaks the build otherwise, so it lands here)
- Test: `src/segmenter/core/nms.test.ts` (rewritten)

**Interfaces:**
- Consumes: `BinaryMask { coverage: Uint8Array; area: number }` from `./mask-postprocess`.
- Produces, all from `src/segmenter/core/nms.ts`:
  - `popcount(value: number): number` — exported.
  - `dedupeMasksReference(masks: readonly BinaryMask[], iouThreshold: number): number[]` — exported from `./nms`, **not** from `./index`.
  - `dedupeMasks(masks: readonly BinaryMask[], iouThreshold: number, width: number): number[]` — exported from both.
  - `pairwiseIoU(a: BinaryMask, b: BinaryMask): number` — unchanged, exported from both.
  - `PreparedMask`, `prepareMask`, `packedIoU` are **module-private**: not exported, not tested directly.

---

- [ ] **Step 1: Rewrite the test file**

Replace the whole of `src/segmenter/core/nms.test.ts` with this. Every existing case is here — the `dedupeMasks` ones now pass a `width`, which is the only change to them.

```ts
import { describe, it, expect } from 'vitest';
import { dedupeMasks, dedupeMasksReference, pairwiseIoU, popcount, type BinaryMask } from './nms';

function mask(bits: number[]): BinaryMask {
  const coverage = Uint8Array.from(bits);
  return { coverage, area: bits.reduce((sum, b) => sum + (b ? 1 : 0), 0) };
}

/** A filled axis-aligned rectangle on a `width` x `height` canvas. */
function rect(
  width: number,
  height: number,
  x0: number,
  y0: number,
  w: number,
  h: number,
): BinaryMask {
  const coverage = new Uint8Array(width * height);
  let area = 0;
  for (let y = y0; y < y0 + h; y += 1) {
    for (let x = x0; x < x0 + w; x += 1) {
      coverage[y * width + x] = 1;
      area += 1;
    }
  }
  return { coverage, area };
}

/** mulberry32 — small, seeded, reproducible. A failure below repeats exactly. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('pairwiseIoU', () => {
  it('is 1 for identical masks', () => {
    expect(pairwiseIoU(mask([1, 1, 0, 0]), mask([1, 1, 0, 0]))).toBe(1);
  });

  it('is 0 for disjoint masks', () => {
    expect(pairwiseIoU(mask([1, 1, 0, 0]), mask([0, 0, 1, 1]))).toBe(0);
  });

  it('is smaller / larger when one mask fully contains the other', () => {
    expect(pairwiseIoU(mask([1, 0, 0, 0]), mask([1, 1, 1, 1]))).toBe(0.25);
  });

  it('is 0 rather than NaN when both masks are empty', () => {
    expect(pairwiseIoU(mask([0, 0]), mask([0, 0]))).toBe(0);
  });
});

describe('popcount', () => {
  it('counts nothing in zero', () => {
    expect(popcount(0)).toBe(0);
  });

  it('counts every bit in an all-ones word', () => {
    expect(popcount(0xffffffff)).toBe(32);
  });

  // 0x80000000 arrives from `a & b` as the NEGATIVE int32 -2147483648. Both
  // spellings must answer 1: a popcount that lets the sign bit through the
  // first subtraction gets this wrong, and nothing else in the suite would
  // notice.
  it('counts the high bit exactly once, unsigned or sign-extended', () => {
    expect(popcount(0x80000000)).toBe(1);
    expect(popcount(0x80000000 | 0)).toBe(1);
  });

  it('counts exactly one bit at every single-bit position 0..31', () => {
    for (let bit = 0; bit < 32; bit += 1) {
      expect(popcount((1 << bit) >>> 0)).toBe(1);
      expect(popcount(1 << bit)).toBe(1);
    }
  });
});

describe('dedupeMasks', () => {
  it('returns nothing for no candidates', () => {
    expect(dedupeMasks([], 0.7, 4)).toEqual([]);
  });

  it('keeps the larger of two duplicates and drops the smaller', () => {
    const kept = dedupeMasks([mask([1, 0, 0, 0]), mask([1, 1, 1, 0])], 0.2, 4);
    expect(kept).toEqual([1]);
  });

  it('keeps disjoint masks and returns their indices ascending', () => {
    const kept = dedupeMasks([mask([0, 0, 1, 1]), mask([1, 1, 0, 0])], 0.5, 4);
    expect(kept).toEqual([0, 1]);
  });

  it('keeps a candidate whose IoU is EXACTLY the threshold', () => {
    // IoU of a 1-pixel mask inside a 4-pixel mask is exactly 0.25.
    const kept = dedupeMasks([mask([1, 1, 1, 1]), mask([1, 0, 0, 0])], 0.25, 4);
    expect(kept).toEqual([0, 1]);
  });

  it('drops that same candidate once the threshold dips just below its IoU', () => {
    const kept = dedupeMasks([mask([1, 1, 1, 1]), mask([1, 0, 0, 0])], 0.24, 4);
    expect(kept).toEqual([0]);
  });

  it('breaks an equal-area tie in favour of the lower original index', () => {
    const kept = dedupeMasks([mask([1, 1, 0, 0]), mask([1, 1, 0, 0])], 0.5, 4);
    expect(kept).toEqual([0]);
  });

  it('does not let a zero-area mask suppress anything', () => {
    const kept = dedupeMasks([mask([0, 0, 0, 0]), mask([1, 1, 0, 0])], 0.5, 4);
    expect(kept).toEqual([0, 1]);
  });
});

describe('dedupeMasks matches dedupeMasksReference', () => {
  it('on a pair the bbox prefilter rejects on the x axis', () => {
    const masks = [rect(8, 4, 0, 0, 3, 4), rect(8, 4, 5, 0, 3, 4)];
    expect(dedupeMasks(masks, 0.0, 8)).toEqual(dedupeMasksReference(masks, 0.0));
    expect(dedupeMasks(masks, 0.0, 8)).toEqual([0, 1]);
  });

  it('on a pair the bbox prefilter rejects on the y axis', () => {
    const masks = [rect(8, 6, 0, 0, 8, 2), rect(8, 6, 0, 4, 8, 2)];
    expect(dedupeMasks(masks, 0.0, 8)).toEqual(dedupeMasksReference(masks, 0.0));
    expect(dedupeMasks(masks, 0.0, 8)).toEqual([0, 1]);
  });

  // 7 x 5 = 35 pixels: two words, of which 29 bits of the second are unused.
  // If the tail's unused high bits were ever set or counted, the intersection
  // would come out too large and the kept set would differ.
  it('on masks whose coverage length is not a multiple of 32', () => {
    const masks = [
      rect(7, 5, 0, 0, 7, 5),
      rect(7, 5, 0, 0, 4, 5),
      rect(7, 5, 3, 1, 4, 3),
      rect(7, 5, 6, 4, 1, 1),
    ];
    for (const threshold of [0.0, 0.2, 0.5, 0.8]) {
      expect(dedupeMasks(masks, threshold, 7)).toEqual(dedupeMasksReference(masks, threshold));
    }
  });

  // The differential. 300 overlapping rectangles on a 2-D canvas, seeded so a
  // failure reproduces. 64 x 48 rather than a literal full-resolution canvas:
  // the reference is O(pairs x pixels) byte reads, and real 617 x 0.7 MB
  // candidates would put minutes into every CI run for no extra coverage of
  // the two transformations under test.
  const CANVAS_W = 64;
  const CANVAS_H = 48;
  const candidates = (() => {
    const random = mulberry32(0x5eed1234);
    const built: BinaryMask[] = [];
    for (let i = 0; i < 300; i += 1) {
      // Every 40th is a genuinely empty mask, so the empty/zero-area path is
      // inside the differential rather than only in the canned cases above.
      if (i % 40 === 39) {
        built.push({ coverage: new Uint8Array(CANVAS_W * CANVAS_H), area: 0 });
        continue;
      }
      const w = 4 + Math.floor(random() * 20);
      const h = 4 + Math.floor(random() * 16);
      const x0 = Math.floor(random() * (CANVAS_W - w + 1));
      const y0 = Math.floor(random() * (CANVAS_H - h + 1));
      built.push(rect(CANVAS_W, CANVAS_H, x0, y0, w, h));
    }
    return built;
  })();

  it.each([0.0, 0.1, 0.25, 0.5, 0.7, 0.9])(
    'on 300 seeded overlapping candidates at threshold %s',
    (threshold) => {
      // Exact equality, no tolerance: N2 and N3 are exact transformations.
      expect(dedupeMasks(candidates, threshold, CANVAS_W)).toEqual(
        dedupeMasksReference(candidates, threshold),
      );
    },
  );
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm run test -- src/segmenter/core/nms.test.ts`

Expected: FAIL. `dedupeMasksReference` and `popcount` are not exported yet, so the whole file fails to resolve its imports.

- [ ] **Step 3: Rewrite `src/segmenter/core/nms.ts`**

Keep the file's existing top-of-file doc comment (lines 1-17) and `pairwiseIoU` (lines 19-37) **byte for byte**. Replace only the `dedupeMasks` block at the bottom (lines 39-63) with everything below.

```ts
/**
 * Population count of one 32-bit word — the classic SWAR bit-twiddle.
 *
 * Exported so it can be unit-tested directly, which is worth doing: the input
 * here is the result of `a & b`, and `&` yields a SIGNED int32, so a word with
 * the high bit set arrives as a negative number. `value >>> 0` normalises that
 * back to the unsigned bit pattern before the first (arithmetic) subtraction —
 * without it, `popcount(0x80000000 | 0)` is wrong and nothing downstream says so.
 */
export function popcount(value: number): number {
  let v = value >>> 0;
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return (Math.imul(v, 0x01010101) >>> 24) & 0x3f;
}

/**
 * A candidate with its coverage packed 32 pixels to a word and its bounding
 * box precomputed. Module-private: this is an implementation detail of
 * `dedupeMasks`, built and discarded inside one call.
 */
interface PreparedMask {
  words: Uint32Array;
  area: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  /** No covered pixel at all, so the bbox is meaningless and nothing intersects it. */
  empty: boolean;
}

function prepareMask(mask: BinaryMask, width: number): PreparedMask {
  const coverage = mask.coverage;
  const pixels = coverage.length;
  const words = new Uint32Array((pixels + 31) >>> 5);
  let x0 = 0;
  let y0 = 0;
  let x1 = -1;
  let y1 = -1;
  let seen = false;

  for (let p = 0; p < pixels; p += 1) {
    if (!coverage[p]) continue;
    words[p >>> 5] |= 1 << (p & 31);
    const x = p % width;
    const y = (p / width) | 0;
    if (!seen) {
      x0 = x;
      x1 = x;
      y0 = y;
      y1 = y;
      seen = true;
      continue;
    }
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    // p ascends, so y never decreases: y0 is fixed by the first covered pixel.
    if (y > y1) y1 = y;
  }

  return { words, area: mask.area, x0, y0, x1, y1, empty: !seen };
}

/**
 * IoU of two prepared candidates. Exactly equal to `pairwiseIoU` on the masks
 * they were built from, for every input — the two rejections below only skip
 * work that provably contributes nothing to the intersection.
 */
function packedIoU(a: PreparedMask, b: PreparedMask, width: number): number {
  const areaSum = a.area + b.area;
  // Mirrors pairwiseIoU's first line, and keeps the union below from going
  // negative when both areas are 0 but coverage bits are set.
  if (areaSum === 0) return 0;
  // No covered pixel on one side: intersection is 0, so IoU is 0/areaSum.
  if (a.empty || b.empty) return 0;

  // N2 — disjoint on either axis means IoU 0 by definition. Not one pixel read.
  if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return 0;

  // N3 — only the words spanned by the SHARED row band can hold a bit set in
  // both masks. That range is a superset of the possible overlap, so
  // restricting to it is exact rather than approximate.
  const rowStart = a.y0 > b.y0 ? a.y0 : b.y0;
  const rowEnd = a.y1 < b.y1 ? a.y1 : b.y1;
  // Contract: equal-length coverage arrays. Clamp to the shorter defensively,
  // exactly as pairwiseIoU's Math.min does, so a malformed pair still cannot
  // read past the end of either array.
  const wordCount = Math.min(a.words.length, b.words.length);
  const first = (rowStart * width) >>> 5;
  let last = ((rowEnd + 1) * width - 1) >>> 5;
  if (last > wordCount - 1) last = wordCount - 1;

  let intersection = 0;
  for (let w = first; w <= last; w += 1) {
    intersection += popcount(a.words[w] & b.words[w]);
  }

  const union = areaSum - intersection;
  return union === 0 ? 0 : intersection / union;
}

/**
 * Greedy "keep largest" non-maximum suppression.
 *
 * Candidates are considered largest-area-first, ties broken by lower original
 * index. A candidate is dropped only when its IoU against an already-kept mask
 * is STRICTLY GREATER than the threshold — IoU exactly equal to the threshold
 * is not a duplicate.
 *
 * `width` is the pixel width of the coverage arrays and is REQUIRED: it is what
 * turns a flat index into an (x, y), which is what makes the bbox prefilter
 * 2-D. There is deliberately no optional-width fallback — a quiet slow path is
 * easy to end up in by accident, and a required parameter makes the compiler
 * say so instead.
 *
 * Returns the kept original indices in ascending order. Identical, for every
 * input, to `dedupeMasksReference` — see the differential test.
 */
export function dedupeMasks(
  masks: readonly BinaryMask[],
  iouThreshold: number,
  width: number,
): number[] {
  const order = masks.map((_, index) => index);
  // Largest area first; stable on ties via lower original index.
  order.sort((i, j) => masks[j].area - masks[i].area || i - j);

  // One extra linear pass over every candidate, to remove a quadratic number
  // of byte-pair reads from the loop below. Function-local, collected on return.
  const prepared = masks.map((mask) => prepareMask(mask, width));

  const kept: number[] = [];
  for (const candidate of order) {
    const isDuplicate = kept.some(
      (keptIndex) => packedIoU(prepared[candidate], prepared[keptIndex], width) > iouThreshold,
    );
    if (!isDuplicate) kept.push(candidate);
  }

  return kept.sort((a, b) => a - b);
}

/**
 * The byte-wise `dedupeMasks` this module shipped before the bbox prefilter and
 * bit packing, preserved verbatim as the baseline that "identical to baseline"
 * is measured against — by the differential test, and by the playground's
 * opt-in A/B panel.
 *
 * Reachable from `./nms` for the worker and the tests, and deliberately NOT
 * re-exported by `./index`: the package does not own a slow second NMS as
 * public API.
 */
export function dedupeMasksReference(
  masks: readonly BinaryMask[],
  iouThreshold: number,
): number[] {
  const order = masks.map((_, index) => index);
  order.sort((i, j) => masks[j].area - masks[i].area || i - j);

  const kept: number[] = [];
  for (const candidate of order) {
    const isDuplicate = kept.some(
      (keptIndex) => pairwiseIoU(masks[candidate], masks[keptIndex]) > iouThreshold,
    );
    if (!isDuplicate) kept.push(candidate);
  }

  return kept.sort((a, b) => a - b);
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npm run test -- src/segmenter/core/nms.test.ts`

Expected: PASS, every case including the six `it.each` differential rows.

- [ ] **Step 5: Narrow the core barrel export**

`src/segmenter/core/index.ts` line 4 is `export * from './nms';`, which would publish `dedupeMasksReference` and `popcount` on `note-scanner/segmenter`. Replace that one line with:

```ts
export { dedupeMasks, pairwiseIoU, type BinaryMask } from './nms';
```

Leave lines 1-3, 5 and 6 alone. (`BinaryMask` also arrives via `export * from './mask-postprocess'`; naming it explicitly here resolves to the same declaration and is not a conflict — this exact edit has been compiled against the current tree.)

- [ ] **Step 6: Fix the one call site**

In `src/segmenter/worker/segmenter.worker.ts`, line 254 reads:

```ts
    const kept = dedupeMasks(candidates, options.nmsIouThreshold);
```

`originalWidth` is already in scope (destructured at line 118). Change it to:

```ts
    const kept = dedupeMasks(candidates, options.nmsIouThreshold, originalWidth);
```

- [ ] **Step 7: Verify the whole suite and the types**

Run: `npm run test`
Expected: PASS, whole suite.

Run: `npm run typecheck`
Expected: exit 0, no output. This covers `src/**` — which is every file touched in this task.

Run: `grep -qE "dedupeMasksReference|popcount" src/segmenter/core/index.ts && echo LEAKED || echo clean`
Expected: `clean`. That is AC8 for this task: neither name is on the public barrel.

- [ ] **Step 8: Commit**

```bash
git add src/segmenter/core/nms.ts src/segmenter/core/nms.test.ts src/segmenter/core/index.ts src/segmenter/worker/segmenter.worker.ts
git commit -m "perf(nms): bbox prefilter and bit-packed coverage in dedupeMasks"
```

---

### Task 2: Opt-in NMS A/B, worker to playground

**Files:**
- Modify: `src/segmenter/core/types.ts:30-49`, `:60-70`, `:111-115`, `:141-151`
- Modify: `src/segmenter/worker/segmenter.worker.ts:25-39`, `:250-280`
- Modify: `src/segmenter/createSegmenter.ts:131-146`
- Modify: `playground/SegmentView.tsx`
- Test: `src/segmenter/createSegmenter.test.ts`

**Interfaces:**
- Consumes, from Task 1: `dedupeMasks(masks, iouThreshold, width): number[]` (already wired at the worker call site) and `dedupeMasksReference(masks, iouThreshold): number[]`, importable only from `'../core/nms'`.
- Produces:
  - `interface NmsComparison { referenceMs: number; fastMs: number; identical: boolean }` — exported from `./core/types`, and therefore from `note-scanner/segmenter`.
  - `SegmenterOptions.compareNms: boolean` (required on the interface, `false` in `DEFAULT_SEGMENTER_OPTIONS`).
  - `SegmentationResult.nmsComparison?: NmsComparison`.
  - The `done` `SegmenterResponse` variant's `nmsComparison?: NmsComparison`.

---

- [ ] **Step 1: Write the failing `createSegmenter` tests**

Append to `src/segmenter/createSegmenter.test.ts`, inside the existing `describe('createSegmenter', ...)` block (i.e. before its closing `});` on line 264):

```ts
  it('passes compareNms through to the worker, defaulting it off', async () => {
    const segmenter = createSegmenter({ createWorker: spawn });
    const first = segmenter.segment(fakeBitmap());
    expect(FakeWorker.instances[0].posted[0].options.compareNms).toBe(false);
    FakeWorker.instances[0].emit(doneMessage());
    await first;

    const second = segmenter.segment(fakeBitmap(), { compareNms: true });
    expect(FakeWorker.instances[1].posted[0].options.compareNms).toBe(true);
    FakeWorker.instances[1].emit(doneMessage());
    await second;
  });

  it('surfaces nmsComparison from the worker on the result', async () => {
    const segmenter = createSegmenter({ createWorker: spawn });
    const pending = segmenter.segment(fakeBitmap(), { compareNms: true });
    FakeWorker.instances[0].emit({
      type: 'done',
      masks: [],
      width: 2,
      height: 1,
      timings: createTimingAccumulator().report(42),
      counts: { raw: 24, afterFilter: 9, afterNms: 0 },
      nmsComparison: { referenceMs: 900, fastMs: 30, identical: true },
    });

    const result = await pending;
    expect(result.nmsComparison).toEqual({ referenceMs: 900, fastMs: 30, identical: true });
  });

  it('leaves nmsComparison undefined when the worker sent none', async () => {
    const segmenter = createSegmenter({ createWorker: spawn });
    const pending = segmenter.segment(fakeBitmap());
    FakeWorker.instances[0].emit(doneMessage());
    expect((await pending).nmsComparison).toBeUndefined();
  });
```

- [ ] **Step 2: Run them to verify they fail**

Run: `npm run test -- src/segmenter/createSegmenter.test.ts`

Expected: FAIL — `compareNms` is not a known option, `nmsComparison` is not on the `done` message or on `SegmentationResult`.

- [ ] **Step 3: Extend `src/segmenter/core/types.ts`**

Add `compareNms` as the last field of `SegmenterOptions` (after `dtype`, line 48):

```ts
  /**
   * Run the pre-optimization byte-wise NMS alongside the fast one and report
   * both times plus whether they kept the identical set. Off by default: it
   * roughly doubles the `nms` stage, and exists so the speedup can be measured
   * on a real image rather than argued about.
   */
  compareNms: boolean;
```

Add to `DEFAULT_SEGMENTER_OPTIONS` (after `dtype: 'fp32',` on line 69):

```ts
  compareNms: false,
```

Add this interface immediately after `SegmentationCounts` (after line 102):

```ts
/**
 * The result of one opt-in A/B of the NMS implementations.
 *
 * Measurement caveat: the reference runs FIRST, so the fast path sees a warmer
 * cache. At full-resolution candidate sizes the working set is far past any
 * cache and the effect is small, but this is not a controlled benchmark —
 * do not quote the ratio to two significant figures.
 */
export interface NmsComparison {
  /** Wall clock of the byte-wise `dedupeMasksReference`. */
  referenceMs: number;
  /** Wall clock of the shipped `dedupeMasks` — the same number `timings` records. */
  fastMs: number;
  /** Whether both returned the identical kept-index array. */
  identical: boolean;
}
```

Add to `SegmentationResult` (after `counts` on line 114):

```ts
  /** Present only when `compareNms` was set. */
  nmsComparison?: NmsComparison;
```

Add to the `done` variant of `SegmenterResponse` (after `counts: SegmentationCounts;` on line 149):

```ts
      nmsComparison?: NmsComparison;
```

- [ ] **Step 4: Wire the worker**

In `src/segmenter/worker/segmenter.worker.ts`, add `type NmsComparison,` to the existing `from '../core'` import list (keep it alphabetical — it goes between `type FilterSubstep,` and `type RawMask,`), and add one new import immediately after that block (after line 39):

```ts
// Not on the `note-scanner/segmenter` surface by design, so it is imported
// from the module rather than from the barrel.
import { dedupeMasksReference } from '../core/nms';
```

Replace the `nms` block (lines 250-257, from the `// ---- nms:` comment through the `post({ type: 'progress', ... phase: 'nms' ... })` call) with:

```ts
    // ---- nms: once, across every batch. Adjacent grid points land on the
    // ---- same object constantly, so this is where the count actually falls.
    phase = 'nms';

    // The A/B, when asked for. Reference FIRST, over the identical candidate
    // array, so the comparison is against the same input and not a mutated one.
    let referenceKept: number[] | null = null;
    let referenceMs = 0;
    if (options.compareNms) {
      const referenceStarted = performance.now();
      referenceKept = dedupeMasksReference(candidates, options.nmsIouThreshold);
      referenceMs = performance.now() - referenceStarted;
    }

    started = performance.now();
    const kept = dedupeMasks(candidates, options.nmsIouThreshold, originalWidth);
    elapsed = performance.now() - started;
    // Deliberately only the fast path: the reference's time travels in
    // nmsComparison and NOWHERE else, so the results table is never inflated
    // by the doubled work.
    timings.record('nms', elapsed);
    post({ type: 'progress', event: { phase: 'nms', done: 1, total: 1, ms: elapsed } });

    // Both are returned ascending, so element-wise equality is set equality.
    const nmsComparison: NmsComparison | null = referenceKept
      ? {
          referenceMs,
          fastMs: elapsed,
          identical:
            referenceKept.length === kept.length &&
            referenceKept.every((value, i) => value === kept[i]),
        }
      : null;
```

Then in the `done` message (lines 264-276), add the field after `counts: { ... },` — spread rather than assigned, so a run without the option posts a message with no `nmsComparison` key at all rather than a key holding `undefined`:

```ts
        ...(nmsComparison ? { nmsComparison } : {}),
```

- [ ] **Step 5: Surface it on the result**

In `src/segmenter/createSegmenter.ts`, in the `resolve({ ... })` literal, add after `counts: message.counts,` (line 145):

```ts
              nmsComparison: message.nmsComparison,
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `npm run test -- src/segmenter/createSegmenter.test.ts`
Expected: PASS.

Run: `npm run test`
Expected: PASS, whole suite.

Run: `npm run typecheck`
Expected: exit 0. This covers `src/segmenter/core/types.ts`, the worker and `createSegmenter.ts`. It does **not** cover the playground file edited in the next step — see Step 9.

- [ ] **Step 7: Add the `compare-nms` control to the playground**

In `playground/SegmentView.tsx`, add a fifth control to the options paragraph, immediately after the closing `</label>` of the `NMS IoU` control (line 182) and before the paragraph's `</p>`:

```tsx
        {' '}
        <label>
          compare NMS:{' '}
          <input
            data-testid="compare-nms"
            type="checkbox"
            checked={options.compareNms}
            onChange={(e) => patch({ compareNms: e.target.checked })}
          />
        </label>
```

(`data-testid` here is the convention every other control in this file follows, not a hook for a test: this repo has no browser test runner, and no assertion in this plan reads it. AC9 names the control `compare-nms`, so the id is what makes the criterion checkable by reading the file.)

- [ ] **Step 8: Render the A/B result**

In the same file, inside the `{result && ( ... )}` block, add after the closing `</p>` of the `mask-counts` paragraph (line 309):

```tsx
          {result.nmsComparison && (
            <p data-testid="nms-comparison">
              nms A/B: reference {ms(result.nmsComparison.referenceMs)} ms → fast{' '}
              {ms(result.nmsComparison.fastMs)} ms
              {result.nmsComparison.fastMs > 0 &&
                ` (${(result.nmsComparison.referenceMs / result.nmsComparison.fastMs).toFixed(1)}x)`}
              . kept sets{' '}
              <strong>{result.nmsComparison.identical ? 'identical' : 'DIFFERENT'}</strong>.
            </p>
          )}
```

Four facts have to be on screen — reference ms, fast ms, the speedup, and a pass/fail on set equality. The exact wording and styling are yours; the guard on `fastMs > 0` is not optional, because a sub-millisecond fast path can round to 0 and divide to `Infinity`.

- [ ] **Step 9: Typecheck the playground file**

`npm run typecheck` does **not** reach `playground/` — `tsconfig.json` excludes it. Run the playground file through `tsc` directly instead:

```bash
npx tsc --noEmit --strict --jsx react-jsx --target ES2020 --module esnext \
  --moduleResolution bundler --esModuleInterop --skipLibCheck \
  --lib dom,dom.iterable,esnext \
  playground/SegmentView.tsx playground/vite-env.d.ts
```

Expected: exit 0, no output. (Verified to exit 0 against the pre-change tree, so any output is from this task's edit.)

- [ ] **Step 10: Confirm the public surface did not grow a slow NMS**

Run: `npm run smoke`
Expected: `build smoke: OK`. (`dist/` is gitignored; this rebuilds it.)

```bash
grep -q "dedupeMasksReference" dist/segmenter/index.d.ts && echo LEAKED || echo clean
grep -q "declare function dedupeMasks" dist/segmenter/index.d.ts && echo present || echo MISSING
```

Expected: `clean` then `present` — the reference implementation is reachable from `./nms` for the worker and the tests but is not on the `note-scanner/segmenter` type surface, while the fast one still is (AC8).

- [ ] **Step 11: Commit**

```bash
git add src/segmenter/core/types.ts src/segmenter/worker/segmenter.worker.ts src/segmenter/createSegmenter.ts src/segmenter/createSegmenter.test.ts playground/SegmentView.tsx
git commit -m "feat(segmenter): opt-in A/B of the reference and fast NMS paths"
```

---

## Acceptance criteria coverage

| AC | Where it is satisfied |
|---|---|
| AC1 exact differential | Task 1 Step 1's `it.each` over 300 seeded candidates x 6 thresholds; Step 4 runs it |
| AC2 existing cases | Task 1 Step 1 carries all seven `dedupeMasks` cases and all four `pairwiseIoU` cases forward |
| AC3 `popcount` unit tests | Task 1 Step 1's `describe('popcount')`, including the sign-extended high bit |
| AC4 bbox-disjoint + non-multiple-of-32 | Task 1 Step 1's first three `matches dedupeMasksReference` cases |
| AC5 required `width`, `pairwiseIoU` untouched | Task 1 Step 3 (signature; `pairwiseIoU` explicitly preserved byte for byte) + Step 6 (call site) |
| AC6 opt-in A/B path | Task 2 Steps 3-5, asserted by Step 1's three tests |
| AC7 `nms` timing is fast-path only | Task 2 Step 4 — `timings.record('nms', elapsed)` and the progress event both read `elapsed`, measured after the reference block closed |
| AC8 reference off the public surface | Task 1 Step 5 + Step 7's grep; Task 2 Step 10's `dist` grep |
| AC9 playground panel | Task 2 Steps 7-8, typechecked by Step 9 |
| AC10 real-image numbers | Deferred to the human, explicitly not a gate on this PR |
