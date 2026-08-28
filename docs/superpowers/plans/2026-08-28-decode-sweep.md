# Automated Playwright sweep of the decode phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two new opt-in decode paths in the segmenter worker and an automated, committed, config-driven Playwright sweep that measures them on a real GPU and names the winner.

**Architecture:** Three new `SegmenterOptions` flags (`overlapDecodeFilter`, `gpuResidentEmbeddings`, `keepRawMasks`), all `false` by default, change the worker's batch loop, its ONNX session creation, and what the `done` message carries. A ported-and-re-cut `playground/compare.ts` owns every piece of sweep logic (grid expansion, row labelling, ranking, markdown) as unit-tested TypeScript; a new `playground/CompareView.tsx` tab drives one configuration at a time and re-exports that logic to the page as `window.__decodeSweep`; and `scripts/sweep-decode.mjs` starts Vite through its Node API, drives headed Chromium through the Playwright **library**, and reaches the logic through the page rather than duplicating it into an untested `.mjs`.

**Tech Stack:** TypeScript, vitest (node + jsdom environments), React 19 (playground only), `@huggingface/transformers` (worker only), Vite Node API, Playwright library (`chromium` from `@playwright/test`).

**Spec:** `docs/superpowers/specs/2026-08-28-decode-sweep-design.md`

## Global Constraints

- **All three new options default to `false`** in `DEFAULT_SEGMENTER_OPTIONS`, and all three travel through `createSegmenter` into the worker request untouched.
- **The sweep ranks on `budgetMs = timings.totalMs - timings.phases['model-load'].total`, never on the `decode` row.** `overlapDecodeFilter` moves time between stage counters rather than removing it. This caveat is rendered in the generated markdown, directly above the table.
- **Every rendered row is labelled from the options captured with its result**, never from the runner's or the page's current control state.
- **`gpuResidentEmbeddings` has NO CPU fallback.** If the runtime rejects a GPU-resident tensor, the throw propagates and arrives as a `SegmenterFailure` naming its phase. No catch-and-retry anywhere in that path.
- **Exactly `image_embeddings` and `image_positional_embeddings` are named as `'gpu-buffer'`.** `pred_masks` and `iou_scores` stay on the CPU — the filter stage reads `pred_masks.data`.
- **The worker's session cache key includes `gpuResidentEmbeddings` and excludes `keepRawMasks`.**
- **`batchSize: 64` is NOT in the default grid** (issue #12 measured it dying with `Array buffer allocation failed` inside `post_process_masks`). It stays reachable through `--config`.
- **The default grid is exactly 16 measured rows:** 4 decode paths x `batchSize` {8, 32} x `dtype` {fp32, fp16} at `pointsPerSide` 16, 1 rep.
- **There is exactly one IoU implementation in this repo.** `compareMaskSets` is built on `pairwiseIoU` from `src/segmenter/core`. Do not write a second one.
- **No new npm dependencies.** `vite`, `@playwright/test` (which re-exports `chromium`), `@types/node`, `vitest`, `jsdom` and `@testing-library/react` are all already in `devDependencies`. If a step seems to need a package, stop and re-read this line.
- **No new `.gitignore` entries.** The runner's only outputs are `docs/measurements/*.md` and `*.json`, which are deliberately committed. Nothing else is written to the tree.
- **Nothing under `src/` learns that `window.__decodeSweep` exists.** The hook is playground-only, so `scripts/smoke-build.mjs`'s published-surface guarantee is untouched. `src/segmenter/worker/session-key.ts` is likewise **not** re-exported from `src/segmenter/index.ts`.
- **PR #11's branch is PORTED, not replayed.** Read `git show worktree-issue-3-2-8-e1-d1-fp16-and-a-larger-batchsize:playground/compare.ts` for reference; never rebase or cherry-pick that branch.
- Out of scope: changing `DEFAULT_SEGMENTER_OPTIONS` to the winner (a follow-up), moving `pred_masks`/`iou_scores` to GPU buffers, any change to `nms`/`mask-encode`/`filter` beyond the loop restructuring, multi-crop, a second sample image.

## Verification scope — read this before citing a command

Today, on `main`:

- `npm test` runs vitest over `src/**/*.test.ts`, `src/**/*.test.tsx` and `playground/**/*.test.ts`. **`playground/**/*.test.tsx` is NOT included** until Task 3 adds it.
- `npm run typecheck` is `tsc --noEmit` against the root `tsconfig.json`, whose `exclude` lists `playground`. **Nothing under `playground/` is typechecked by `npm run typecheck`** until Task 2 adds `tsconfig.playground.json` and chains it into the script.
- The root `tsconfig.json` `exclude` (`["node_modules", "dist", "playground"]`) is **inherited** by `extends`. `tsconfig.playground.json` therefore MUST override `exclude` — an inherited `playground` exclusion beats the child's `include` and the project would silently compile zero files.
- The `tsconfig.playground.json` shape given in Task 2 has been run against the current tree with `npx tsc --noEmit -p ...` and exits 0, so any error it reports after a task is an error that task introduced.

After Task 3, `npm run typecheck` covers `playground/CompareView.tsx` and `npm test` executes `playground/CompareView.test.tsx`. That is AC10.

**What is deliberately NOT unit-tested, and why.** The worker's batch loop and `loadSession`'s call into `SamModel.from_pretrained` are not testable in this repo: `segmenter.worker.ts` is a top-level module body that binds `globalThis.addEventListener` and calls into `@huggingface/transformers` with a live WebGPU device, with no seam to inject a fake model. Extracting the loop purely to assert "the second dispatch started before the first filter finished" would test the extraction, not the worker. **AC2's ordering claim and AC4's propagation claim are therefore bound to code review plus the sweep's own `counts` columns** — a path that reorders work incorrectly shows up as a changed `afterNms` in the committed table. What *is* extractable and *is* tested is the pure decision both hazards turn on: `sessionCacheKey` and `embeddingsSessionOptions` (Task 1), which own AC3 and the "exactly these two output names" half of AC4.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/segmenter/core/types.ts` | Modify | The three new options + defaults; `RawMask`; `SegmentationResult.rawMasks?`; the `done` response's `rawMasks?`. |
| `src/segmenter/worker/session-key.ts` | **Create** | `sessionCacheKey` and `embeddingsSessionOptions` — the two pure decisions the GPU-resident path turns on. Imports no runtime package. |
| `src/segmenter/worker/session-key.test.ts` | **Create** | AC3, and AC4's output-name half. |
| `src/segmenter/worker/segmenter.worker.ts` | Modify | `dispatch` helper; the overlapped batch loop; the no-op catch on an abandoned dispatch; `session_options`; the `rawMasks` transfer. |
| `src/segmenter/createSegmenter.ts` | Modify | Surfaces `rawMasks` on `SegmentationResult`. |
| `src/segmenter/createSegmenter.test.ts` | Modify | Option passthrough for all three flags; `rawMasks` across the worker boundary. |
| `playground/compare.ts` | **Create** | `SweepConfig`, `DEFAULT_SWEEP_CONFIG`, `resolveConfig`, `expandGrid`, `warmUpRow`, `RunRecord`, `budgetMsOf`, `rankRecords`, `fastestRecord`, `rowLabel`, `compareMaskSets`, `toMarkdown`, and the `window.__decodeSweep` type. |
| `playground/compare.test.ts` | **Create** | AC6, AC8, and the ported `compareMaskSets` cases. |
| `tsconfig.playground.json` | **Create** | Type-checks `playground/**/*` without letting the published `dts` graph see it. |
| `package.json` | Modify | `typecheck` chains the playground project; `sweep:decode` script. |
| `playground/CompareView.tsx` | **Create** | The sweep's page: six `data-testid` controls, the run control, `run-json`, `compare-table`, `copy-markdown`, `agreement-panel`, the `__decodeSweep` hook, injectable deps. |
| `playground/CompareView.test.tsx` | **Create** | AC7, AC9, AC10. |
| `playground/main.tsx` | Modify | Third tab. |
| `playground/SegmentView.tsx` | Modify | The two decode-path checkboxes beside `compare-nms`. |
| `vitest.config.ts` | Modify | Adds `playground/**/*.test.tsx`. |
| `scripts/sweep-decode.mjs` | **Create** | The runner. |
| `docs/measurements/<date>-decode-sweep.{md,json}` | **Create** (Task 5) | The committed real-GPU run. |

---

### Task 1: The two decode paths, `keepRawMasks`, and the options that select them

**Files:**
- Modify: `src/segmenter/core/types.ts`
- Create: `src/segmenter/worker/session-key.ts`
- Create: `src/segmenter/worker/session-key.test.ts`
- Modify: `src/segmenter/worker/segmenter.worker.ts`
- Modify: `src/segmenter/createSegmenter.ts`
- Modify: `src/segmenter/createSegmenter.test.ts`

**Interfaces:**
- Consumes: `SegmenterOptions`, `BinaryMask`, `thresholdMask`, `batchPoints`, `buildPointGrid` — all as they are on `main`.
- Produces, from `note-scanner/segmenter` (i.e. `src/segmenter/core/types.ts`, re-exported by `core/index.ts`'s existing `export * from './types'` — no change needed there):
  - `SegmenterOptions.overlapDecodeFilter: boolean`
  - `SegmenterOptions.gpuResidentEmbeddings: boolean`
  - `SegmenterOptions.keepRawMasks: boolean`
  - `interface RawMask { coverage: Uint8Array; area: number }`
  - `SegmentationResult.rawMasks?: RawMask[]`
- Produces, from `src/segmenter/worker/session-key.ts` (**not** re-exported from `src/segmenter/index.ts`):
  - `sessionCacheKey(options: SegmenterOptions): string`
  - `embeddingsSessionOptions(options: SegmenterOptions): { preferredOutputLocation: Record<string, 'gpu-buffer'> } | undefined`

- [ ] **Step 1: Add the three options, their defaults, and `RawMask` to `src/segmenter/core/types.ts`**

Insert into `SegmenterOptions`, immediately after the `compareNms` member:

```ts
  /**
   * Keep the next batch's model dispatch in flight while the current batch's
   * filter block runs on the CPU.
   *
   * This does NOT make decode faster. The batch loop is serial today — the GPU
   * is idle for the whole filter block and the CPU is idle for the whole model
   * call — and overlapping them moves time BETWEEN the stage counters. When
   * the GPU finished during the previous filter block, `decode` collapses
   * toward zero: the work still happened, it simply stopped being counted
   * anywhere. Anything ranking these runs must rank on wall clock.
   *
   * Peak memory rises by one batch of `pred_masks` (at `batchSize` 32 that
   * tensor is ~25 MB fp32, so ~50 MB with two in flight). A configuration that
   * exhausts memory fails the run rather than degrading quietly.
   */
  overlapDecodeFilter: boolean;
  /**
   * Ask the runtime to leave the encoder's `image_embeddings` and
   * `image_positional_embeddings` on the device instead of copying them back
   * to the CPU, so the decoder does not re-upload ~8 MB on every dispatch.
   *
   * Nothing may read `.data` on those two tensors while this is on — the
   * worker only forwards them into `model(...)`, which is what makes the path
   * possible at all. `pred_masks` and `iou_scores` are deliberately left on
   * the CPU, because the filter stage reads `pred_masks.data`.
   *
   * There is NO CPU fallback. If the runtime rejects a GPU-resident input the
   * run fails, naming its phase; a silent fallback would report a fast row
   * that measured the very path it was supposed to replace.
   */
  gpuResidentEmbeddings: boolean;
  /**
   * Additionally post the surviving masks' full-resolution coverage buffers
   * back on `SegmentationResult.rawMasks`, transferred rather than cloned.
   *
   * Off by default because a caller holding the result in React state pins
   * tens of megabytes: ~50 coverage arrays at ~0.7 MB each at 16 points per
   * side. It exists so one run's masks can be compared against another's, and
   * for nothing else. It changes nothing about the ONNX sessions and therefore
   * must never enter the worker's session cache key.
   */
  keepRawMasks: boolean;
```

Add to `DEFAULT_SEGMENTER_OPTIONS`, after `compareNms: false,`:

```ts
  overlapDecodeFilter: false,
  gpuResidentEmbeddings: false,
  keepRawMasks: false,
```

Add above `SegmenterRequest` (beside the existing `EncodedMask`):

```ts
/**
 * A surviving mask's raw coverage, at full image resolution.
 *
 * Structurally a `BinaryMask`, so it passes straight into `pairwiseIoU`. Only
 * posted when `keepRawMasks` is set; the coverage buffers are TRANSFERRED, so
 * the worker's own copies are detached once the `done` message is sent.
 */
export interface RawMask {
  coverage: Uint8Array;
  area: number;
}
```

Add to `SegmentationResult`, after `nmsComparison?`:

```ts
  /** Present only when the run asked for `keepRawMasks`. */
  rawMasks?: RawMask[];
```

Add to the `done` variant of `SegmenterResponse`, after `nmsComparison?`:

```ts
      rawMasks?: RawMask[];
```

- [ ] **Step 2: Write the failing test for the session decisions**

Create `src/segmenter/worker/session-key.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { DEFAULT_SEGMENTER_OPTIONS, type SegmenterOptions } from '../core';
import { embeddingsSessionOptions, sessionCacheKey } from './session-key';

function options(overrides: Partial<SegmenterOptions> = {}): SegmenterOptions {
  return { ...DEFAULT_SEGMENTER_OPTIONS, ...overrides };
}

describe('sessionCacheKey', () => {
  it('separates the two embedding residencies (AC3)', () => {
    const cpu = sessionCacheKey(options({ gpuResidentEmbeddings: false }));
    const gpu = sessionCacheKey(options({ gpuResidentEmbeddings: true }));
    expect(cpu).not.toBe(gpu);
  });

  it('still separates modelId and dtype', () => {
    expect(sessionCacheKey(options({ dtype: 'fp32' }))).not.toBe(
      sessionCacheKey(options({ dtype: 'fp16' })),
    );
    expect(sessionCacheKey(options({ modelId: 'a' }))).not.toBe(
      sessionCacheKey(options({ modelId: 'b' })),
    );
  });

  it('ignores keepRawMasks, which changes nothing about the session (AC3)', () => {
    expect(sessionCacheKey(options({ keepRawMasks: true }))).toBe(
      sessionCacheKey(options({ keepRawMasks: false })),
    );
  });

  it('ignores the per-call inference knobs, which are not session state', () => {
    const base = sessionCacheKey(options());
    expect(sessionCacheKey(options({ pointsPerSide: 32 }))).toBe(base);
    expect(sessionCacheKey(options({ batchSize: 64 }))).toBe(base);
    expect(sessionCacheKey(options({ overlapDecodeFilter: true }))).toBe(base);
  });
});

describe('embeddingsSessionOptions', () => {
  it('is undefined on the default CPU-resident path', () => {
    expect(embeddingsSessionOptions(options({ gpuResidentEmbeddings: false }))).toBeUndefined();
  });

  it('names EXACTLY the encoder\'s two outputs as gpu-buffer (AC4)', () => {
    const resolved = embeddingsSessionOptions(options({ gpuResidentEmbeddings: true }));
    expect(resolved).toBeDefined();
    // Exact, not a superset: naming pred_masks or iou_scores here would move a
    // tensor the filter stage reads on the CPU onto the device.
    expect(Object.keys(resolved!.preferredOutputLocation)).toEqual([
      'image_embeddings',
      'image_positional_embeddings',
    ]);
    expect(Object.values(resolved!.preferredOutputLocation)).toEqual([
      'gpu-buffer',
      'gpu-buffer',
    ]);
  });
});
```

- [ ] **Step 3: Run it and watch it fail**

Run: `npx vitest run src/segmenter/worker/session-key.test.ts`
Expected: FAIL — `Failed to resolve import "./session-key"`.

- [ ] **Step 4: Write `src/segmenter/worker/session-key.ts`**

```ts
/**
 * The two pure decisions the worker's ONNX session creation turns on.
 *
 * Split out of `segmenter.worker.ts` because that module's top-level body
 * binds `globalThis.addEventListener` and imports `@huggingface/transformers`,
 * so nothing in it can be reached from a unit test. This file imports neither.
 *
 * NOT re-exported from `src/segmenter/index.ts`: it is worker-internal, and
 * `scripts/smoke-build.mjs` guards the published surface.
 */
import type { SegmenterOptions } from '../core';

/**
 * The key under which the worker caches its loaded session.
 *
 * `gpuResidentEmbeddings` is a component because it changes HOW the sessions
 * are created. Without it, flipping the flag inside one worker would silently
 * reuse a session built the other way and the sweep would measure a lie.
 *
 * `keepRawMasks` is deliberately absent — it only changes what the worker
 * posts back — and so is every per-call inference knob (`pointsPerSide`,
 * `batchSize`, thresholds, `overlapDecodeFilter`), none of which touch the
 * session.
 */
export function sessionCacheKey(options: SegmenterOptions): string {
  return [
    options.modelId,
    options.dtype,
    options.gpuResidentEmbeddings ? 'gpu-embeddings' : 'cpu-embeddings',
  ].join('|');
}

/**
 * The `session_options` handed to `SamModel.from_pretrained`, or `undefined`
 * for the default CPU-resident path.
 *
 * `preferredOutputLocation` is keyed by OUTPUT NAME, so naming the encoder's
 * two outputs is inert for the `prompt_encoder_mask_decoder` session, which
 * produces neither. `pred_masks` and `iou_scores` are deliberately NOT named:
 * the filter stage reads `pred_masks.data` on the CPU, and moving it to a GPU
 * buffer would trade one copy for a worse one.
 */
export function embeddingsSessionOptions(
  options: SegmenterOptions,
): { preferredOutputLocation: Record<string, 'gpu-buffer'> } | undefined {
  if (!options.gpuResidentEmbeddings) return undefined;
  return {
    preferredOutputLocation: {
      image_embeddings: 'gpu-buffer',
      image_positional_embeddings: 'gpu-buffer',
    },
  };
}
```

- [ ] **Step 5: Run the test again**

Run: `npx vitest run src/segmenter/worker/session-key.test.ts`
Expected: PASS, 6 tests.

- [ ] **Step 6: Wire the session decisions into the worker's `loadSession`**

In `src/segmenter/worker/segmenter.worker.ts`, add below the `dedupeMasksReference` import:

```ts
import { embeddingsSessionOptions, sessionCacheKey } from './session-key';
```

and add `type RawMask,` to the `../core` type import list.

Replace the first four lines of `loadSession`'s body and the `SamModel.from_pretrained` call:

```ts
async function loadSession(options: SegmenterOptions): Promise<Session> {
  const key = sessionCacheKey(options);
  if (session && sessionKey === key) return session;

  const sessionOptions = embeddingsSessionOptions(options);
  const model = (await SamModel.from_pretrained(options.modelId, {
    device: 'webgpu',
    dtype: options.dtype,
    // Spread rather than passed as `undefined`, so the default path hands
    // transformers.js exactly the options object it gets today.
    ...(sessionOptions ? { session_options: sessionOptions } : {}),
  })) as unknown as SamModel;
```

The rest of `loadSession` is unchanged.

> If `SamModel.from_pretrained`'s shipped declaration rejects `session_options`, do NOT loosen `embeddingsSessionOptions`' return type. Add a single narrowly-scoped cast at this one call site with a comment naming the declaration that is too tight.

- [ ] **Step 7: Restructure the batch loop for `overlapDecodeFilter`**

In `run()`, replace everything from `const batches = batchPoints(...)` down to the closing brace of the `for (let b = 0; ...)` loop with the following. Everything inside the filter block — `recordSub`, the `chosen` selection, `post_process_masks`, `thresholdMask` — is **unchanged**; only the seams around it move.

```ts
    const batches = batchPoints(buildPointGrid(options.pointsPerSide), options.batchSize);
    const candidates: BinaryMask[] = [];
    let rawCount = 0;

    /**
     * Build one batch's prompt tensors and ISSUE the model call without
     * awaiting it. Tensor construction lives in here rather than at the call
     * site so that its CPU cost is attributed to `decode` on both paths.
     */
    const dispatch = (batch: readonly [number, number][]) => {
      // The processor's own `reshape_input_points` scales ORIGINAL-space points
      // by reshaped/original. Our grid is already normalized, so multiplying by
      // the reshaped size directly lands in the same place with one less step.
      const pointData = new Float32Array(batch.length * 2);
      for (let i = 0; i < batch.length; i += 1) {
        pointData[i * 2] = batch[i][0] * reshapedWidth;
        pointData[i * 2 + 1] = batch[i][1] * reshapedHeight;
      }
      // Every grid point is its own independent prompt: point_batch_size is the
      // batch length and nb_points_per_image is 1. Label 1 = foreground.
      const inputPoints = new Tensor('float32', pointData, [1, batch.length, 1, 2]);
      const inputLabels = new Tensor(
        'int64',
        new BigInt64Array(batch.length).fill(1n),
        [1, batch.length, 1],
      );
      return model({
        ...embeddings,
        input_points: inputPoints,
        input_labels: inputLabels,
      });
    };

    phase = 'decode';
    /**
     * The decode sample's clock.
     *
     * CONVENTION, because the two paths measure different regions and a reader
     * will assume otherwise. Both record exactly `batches.length` samples.
     *   - serial: the sample is today's exact region — tensor build + `await`.
     *   - overlapped: the sample runs from the end of the PREVIOUS batch's
     *     filter block, through the `await` of this batch's already-in-flight
     *     dispatch, to the moment the NEXT batch's dispatch has been issued.
     *     For b = 0 it starts just before the priming dispatch below, so batch
     *     0's tensor construction is inside the first sample.
     * On the overlapped path the sample is therefore dominated by WAITING: a
     * near-zero decode means the GPU finished during the previous filter block,
     * not that the work vanished.
     */
    let decodeStarted = performance.now();
    /**
     * The next batch's dispatch, kept in flight across the current batch's
     * filter block. Null on the serial path and after the last batch.
     */
    let pending: ReturnType<typeof dispatch> | null =
      options.overlapDecodeFilter && batches.length > 0 ? dispatch(batches[0]) : null;

    try {
      for (let b = 0; b < batches.length; b += 1) {
        const batchStarted = performance.now();

        // ---- decode ----
        phase = 'decode';
        let outputs;
        if (options.overlapDecodeFilter) {
          // Cleared BEFORE the await so a rejection here is handled by the
          // await alone; the catch below then has nothing to suppress.
          const inFlight = pending;
          pending = null;
          outputs = await inFlight;
          // Issued before the filter block so the GPU is busy through it.
          pending = b + 1 < batches.length ? dispatch(batches[b + 1]) : null;
        } else {
          // Exactly today's shape: dispatch, immediately awaited.
          decodeStarted = performance.now();
          outputs = await dispatch(batches[b]);
        }
        // `.to('float32')` is a no-op on an fp32 model and the required
        // conversion on an fp16 one, where ORT hands back raw float16 bits in a
        // Uint16Array on runtimes with no Float16Array.
        const predMasks = outputs.pred_masks.to('float32') as Tensor;
        const iouScores = outputs.iou_scores.to('float32') as Tensor;
        timings.record('decode', performance.now() - decodeStarted);

        // ---- filter, at low resolution ----
        phase = 'filter';
        started = performance.now();
        /* ... the ENTIRE existing filter block, verbatim, from
           `let subStarted = started;` through `timings.record('filter', ...)`.
           Do not change a line of it. ... */

        post({
          type: 'progress',
          event: {
            phase: 'decode',
            done: b + 1,
            total: batches.length,
            ms: performance.now() - batchStarted,
          },
        });

        // Starts the NEXT overlapped sample's clock. Overwritten on the serial
        // path before it is read, so it costs that path nothing but a store.
        decodeStarted = performance.now();
      }
    } catch (error) {
      // A dispatch nobody will ever await must not surface as an unhandled
      // top-level rejection and drown out the failure that actually killed the
      // run. Suppress it, then rethrow with `phase` still naming its stage.
      if (pending) {
        void pending.catch(() => {});
        pending = null;
      }
      throw error;
    }
```

- [ ] **Step 8: Post `rawMasks` when asked, with the buffers transferred**

In `run()`, replace the final `post({ type: 'done', ... })` call (and its preceding comment) with:

```ts
    // ---- keepRawMasks: the surviving coverage buffers, only when asked for.
    // Collected AFTER the encode loop above, which reads them locally.
    //
    // `thresholdMask` allocates a fresh Uint8Array per mask, so every coverage
    // owns its own ArrayBuffer: no buffer can appear twice in this transfer
    // list (postMessage throws on a duplicate) and none is aliased elsewhere.
    const rawMasks: RawMask[] = options.keepRawMasks
      ? kept.map((index) => ({
          coverage: candidates[index].coverage,
          area: candidates[index].area,
        }))
      : [];
    // With the flag off this is empty and the call below is byte-for-byte
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
          afterNms: masks.length,
        },
        ...(nmsComparison ? { nmsComparison } : {}),
        ...(options.keepRawMasks ? { rawMasks } : {}),
      },
      transfer.length > 0 ? transfer : undefined,
    );
```

- [ ] **Step 9: Surface `rawMasks` on the result in `createSegmenter.ts`**

In the `done` branch of `onMessage`, add to the `resolve({...})` object after `nmsComparison: message.nmsComparison,`:

```ts
            // Spread, not a bare `rawMasks: message.rawMasks`: the field is
            // absent unless the run asked for it, and an explicit `undefined`
            // would make `'rawMasks' in result` true for every run.
            ...(message.rawMasks ? { rawMasks: message.rawMasks } : {}),
```

- [ ] **Step 10: Add the boundary tests to `src/segmenter/createSegmenter.test.ts`**

Add `type RawMask,` to the `./core` type import list, and append these cases inside the existing `describe('createSegmenter', ...)`:

```ts
  it('defaults all three new flags off and passes each one through (AC1)', async () => {
    const segmenter = createSegmenter({ createWorker: spawn });
    const first = segmenter.segment(fakeBitmap());
    const defaults = FakeWorker.instances[0].posted[0].options;
    expect(defaults.overlapDecodeFilter).toBe(false);
    expect(defaults.gpuResidentEmbeddings).toBe(false);
    expect(defaults.keepRawMasks).toBe(false);
    FakeWorker.instances[0].emit(doneMessage());
    await first;

    const second = segmenter.segment(fakeBitmap(), {
      overlapDecodeFilter: true,
      gpuResidentEmbeddings: true,
      keepRawMasks: true,
    });
    const sent = FakeWorker.instances[1].posted[0].options;
    expect(sent.overlapDecodeFilter).toBe(true);
    expect(sent.gpuResidentEmbeddings).toBe(true);
    expect(sent.keepRawMasks).toBe(true);
    FakeWorker.instances[1].emit(doneMessage());
    await second;
  });

  it('surfaces rawMasks when the worker sends them, and omits the field otherwise (AC5)', async () => {
    const rawMasks: RawMask[] = [{ coverage: new Uint8Array([1, 0, 1, 1]), area: 3 }];

    const segmenter = createSegmenter({ createWorker: spawn });
    const withMasks = segmenter.segment(fakeBitmap(), { keepRawMasks: true });
    FakeWorker.instances[0].emit({ ...doneMessage(), type: 'done', rawMasks });
    const kept = await withMasks;
    expect(kept.rawMasks).toEqual(rawMasks);

    const without = segmenter.segment(fakeBitmap());
    FakeWorker.instances[1].emit(doneMessage());
    const plain = await without;
    // Absent, not `undefined`: a run that did not ask for masks must not carry
    // the key at all, so `'rawMasks' in result` is a usable question.
    expect('rawMasks' in plain).toBe(false);
  });
```

- [ ] **Step 11: Run the whole suite and the typecheck**

Run: `npm test`
Expected: PASS. The two new `createSegmenter` cases and the six `session-key` cases are included.

Run: `npm run typecheck`
Expected: exit 0.

- [ ] **Step 12: Re-read the worker diff against AC2 and AC4**

This is a review step, not a test — the loop is not unit-testable (see *Verification scope*). Confirm by reading `git diff src/segmenter/worker/segmenter.worker.ts`:

1. With `overlapDecodeFilter` **off**, the loop body is `await dispatch(batches[b])` with the decode clock started immediately before it — today's exact region.
2. With it **on**, `pending = dispatch(batches[b + 1])` appears strictly **before** the `phase = 'filter'` line.
3. `pending` is set to `null` before the `await` that consumes it, and the `catch` around the loop attaches `void pending.catch(() => {})` before rethrowing.
4. `decodeStarted` is assigned before the priming `dispatch(batches[0])`.
5. Nowhere in `loadSession` or the loop is there a `try`/`catch` that retries on the CPU after a GPU-resident failure. Grep to be sure: `grep -n "catch" src/segmenter/worker/segmenter.worker.ts` should show exactly two — the abandoned-dispatch suppressor and the pre-existing outer phase catch.

- [ ] **Step 13: Commit**

```bash
git add src/segmenter
git commit -m "feat(segmenter): overlapDecodeFilter, gpuResidentEmbeddings and keepRawMasks"
```

---

### Task 2: `playground/compare.ts` — the sweep's grid, ranking and report

**Files:**
- Create: `playground/compare.ts`
- Create: `playground/compare.test.ts`
- Create: `tsconfig.playground.json`
- Modify: `package.json` (the `typecheck` script)

**Interfaces:**
- Consumes: `pairwiseIoU`, `PHASE_ORDER`, `RawMask`, `SegmentationCounts`, `SegmentationPhase`, `SegmenterOptions`, `TimingReport` from `../src/segmenter` (Task 1's `RawMask` included).
- Produces, all exported from `playground/compare.ts`:
  - `type DecodePath = 'none' | 'overlap' | 'gpuEmbeddings' | 'both'`, `DECODE_PATHS`
  - `decodeFlags(path): { overlapDecodeFilter: boolean; gpuResidentEmbeddings: boolean }`
  - `decodePathOf(options): DecodePath`
  - `interface RowOptions`, `interface SweepConfig`, `DEFAULT_SWEEP_CONFIG`
  - `resolveConfig(overrides?, flags?): SweepConfig`
  - `interface SweepRow`, `expandGrid(config): SweepRow[]`, `warmUpRow(rows): SweepRow`
  - `interface RunRecord`, `budgetMsOf(timings): number`
  - `rankRecords(records): RunRecord[]`, `fastestRecord(records): RunRecord | null`, `rowLabel(record): string`
  - `IOU_MATCH_FLOOR`, `interface MaskAgreement`, `compareMaskSets(baseline, variant, floor?)`
  - `interface AdapterInfo`, `interface ReportMeta`, `OVERLAP_CAVEAT`, `toMarkdown(records, meta): string`
  - `interface DecodeSweepHook` + the `declare global` that puts it on `window.__decodeSweep`

- [ ] **Step 1: Create `tsconfig.playground.json`**

```json
{
  "extends": "./tsconfig.json",
  "include": ["playground/**/*.ts", "playground/**/*.tsx"],
  "exclude": ["node_modules", "dist"]
}
```

`exclude` is overridden deliberately: the root config lists `playground` there, `extends` inherits it, and an inherited exclusion beats this file's `include` — the project would compile zero files and pass vacuously.

- [ ] **Step 2: Chain it into `typecheck` in `package.json`**

```json
    "typecheck": "tsc --noEmit && tsc --noEmit -p tsconfig.playground.json",
```

The root project runs first and is unchanged, so `tsup`'s `dts: true` build still cannot see playground files.

- [ ] **Step 3: Confirm the new project passes on the tree as it stands**

Run: `npm run typecheck`
Expected: exit 0. If it does not, the failure is pre-existing playground code and must be fixed here, before any new file is added — otherwise every later step inherits a red baseline.

- [ ] **Step 4: Write the failing test — `playground/compare.test.ts`**

```ts
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
    counts: { raw: 96, afterFilter: 31, afterNms: 12 },
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
    expect(first[0]).toBe('p16-fp32-b8-none-r1');
    expect(first[15]).toBe('p16-fp16-b32-both-r1');
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
      'mask-encode', 'raw', 'afterFilter', 'afterNms', 'status',
    ]) {
      expect(md).toContain(column);
    }
    expect(md).toContain('p16-fp16-b32-both-r1');
    expect(md).toContain('| both | fp16 | 32 | 16 |');
    // counts.raw / afterFilter / afterNms, in that order.
    expect(md).toContain('| 96 | 31 | 12 |');
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
```

- [ ] **Step 5: Run it and watch it fail**

Run: `npx vitest run playground/compare.test.ts`
Expected: FAIL — `Failed to resolve import "./compare"`.

- [ ] **Step 6: Write `playground/compare.ts`**

```ts
/**
 * The decode sweep's grid, ranking and report — ONE implementation, unit
 * tested, living in TypeScript.
 *
 * `scripts/sweep-decode.mjs` reaches these functions THROUGH THE PAGE (see
 * `DecodeSweepHook` at the bottom of this file) rather than importing a second
 * copy into an `.mjs` where no test would reach it.
 *
 * Playground-local on purpose: nothing here belongs on the published
 * `note-scanner/segmenter` surface, exactly as `benchmark.ts` sits beside its
 * own view. Ported from the closed-unmerged PR #11 branch and re-cut for the
 * decode sweep; `compareMaskSets` is carried over unchanged in substance.
 */
import {
  PHASE_ORDER,
  pairwiseIoU,
  type RawMask,
  type SegmentationCounts,
  type SegmentationPhase,
  type SegmenterOptions,
  type TimingReport,
} from '../src/segmenter';

// ---------------------------------------------------------------- decode path

/**
 * The four points on the decode-path axis: neither new flag, each alone, and
 * both — so the sweep can see whether they compose.
 */
export const DECODE_PATHS = ['none', 'overlap', 'gpuEmbeddings', 'both'] as const;

export type DecodePath = (typeof DECODE_PATHS)[number];

export function decodeFlags(path: DecodePath): {
  overlapDecodeFilter: boolean;
  gpuResidentEmbeddings: boolean;
} {
  return {
    overlapDecodeFilter: path === 'overlap' || path === 'both',
    gpuResidentEmbeddings: path === 'gpuEmbeddings' || path === 'both',
  };
}

/**
 * The inverse. Every label in the report is derived through here from the
 * options a result was CAPTURED with, so there is exactly one source of truth
 * for which path a row measured.
 */
export function decodePathOf(options: {
  overlapDecodeFilter: boolean;
  gpuResidentEmbeddings: boolean;
}): DecodePath {
  if (options.overlapDecodeFilter && options.gpuResidentEmbeddings) return 'both';
  if (options.overlapDecodeFilter) return 'overlap';
  if (options.gpuResidentEmbeddings) return 'gpuEmbeddings';
  return 'none';
}

// --------------------------------------------------------------------- config

/** The subset of `SegmenterOptions` a sweep row pins. */
export interface RowOptions {
  dtype: SegmenterOptions['dtype'];
  batchSize: number;
  pointsPerSide: number;
  overlapDecodeFilter: boolean;
  gpuResidentEmbeddings: boolean;
  keepRawMasks: boolean;
}

export interface SweepConfig {
  decodePaths: readonly DecodePath[];
  batchSizes: readonly number[];
  dtypes: readonly SegmenterOptions['dtype'][];
  pointsPerSide: readonly number[];
  /**
   * Off by default. Sixteen rows of ~50 full-resolution coverage arrays is
   * hundreds of megabytes, and greedy best-IoU pairing over them is a
   * full-resolution O(masks squared) scan per row. The `counts` columns
   * already expose a variant that is fast because it silently dropped masks,
   * which is the signal the sweep needs; agreement is one config flag away for
   * anyone who wants the stronger check.
   */
  keepRawMasks: boolean;
  reps: number;
}

/**
 * decode path {none, overlap, gpuEmbeddings, both} x batchSize {8, 32}
 * x dtype {fp32, fp16} at pointsPerSide 16 — sixteen rows: minutes, not hours.
 *
 * `batchSize: 64` is EXCLUDED. Issue #12 measured it dying with
 * `Array buffer allocation failed` inside `post_process_masks`, whose
 * allocation scales with batch x width x height x 4. It stays reachable
 * through `--config` so the failure can be re-confirmed deliberately, but it
 * does not burn a row in every run.
 */
export const DEFAULT_SWEEP_CONFIG: SweepConfig = {
  decodePaths: DECODE_PATHS,
  batchSizes: [8, 32],
  dtypes: ['fp32', 'fp16'],
  pointsPerSide: [16],
  keepRawMasks: false,
  reps: 1,
};

const DTYPES: readonly SegmenterOptions['dtype'][] = ['fp32', 'fp16', 'q8'];

function positiveInts(name: string, values: readonly number[]): void {
  if (values.length === 0) throw new Error(`${name} must not be empty`);
  for (const value of values) {
    if (!Number.isInteger(value) || value < 1) {
      throw new Error(`${name} must be positive integers, got ${JSON.stringify(value)}`);
    }
  }
}

/**
 * The full cross-product is a FLAG, not a code change.
 *
 * `overrides` is a `--config <path>` file's contents (untrusted user input,
 * hence the validation); `flags.full` and `flags.reps` are command-line flags
 * and WIN over the config file, because a flag the operator just typed should
 * beat a file they typed last week.
 */
export function resolveConfig(
  overrides: Partial<SweepConfig> = {},
  flags: { full?: boolean; reps?: number } = {},
): SweepConfig {
  const config: SweepConfig = {
    ...DEFAULT_SWEEP_CONFIG,
    ...overrides,
    ...(flags.full ? { pointsPerSide: [16, 32] } : {}),
    ...(flags.reps !== undefined ? { reps: flags.reps } : {}),
  };

  if (config.decodePaths.length === 0) throw new Error('decodePaths must not be empty');
  for (const path of config.decodePaths) {
    if (!DECODE_PATHS.includes(path)) {
      throw new Error(`decodePaths contains an unknown path: ${JSON.stringify(path)}`);
    }
  }
  if (config.dtypes.length === 0) throw new Error('dtypes must not be empty');
  for (const dtype of config.dtypes) {
    if (!DTYPES.includes(dtype)) {
      throw new Error(`dtypes contains an unknown dtype: ${JSON.stringify(dtype)}`);
    }
  }
  positiveInts('batchSizes', config.batchSizes);
  positiveInts('pointsPerSide', config.pointsPerSide);
  if (!Number.isInteger(config.reps) || config.reps < 1) {
    throw new Error(`reps must be a positive integer, got ${JSON.stringify(config.reps)}`);
  }

  return config;
}

// ----------------------------------------------------------------------- grid

export interface SweepRow {
  /** Stable and unique: `p<pps>-<dtype>-b<batch>-<path>-r<rep>`. */
  id: string;
  rep: number;
  /** True only for the discarded first run — see `warmUpRow`. */
  warmUp: boolean;
  options: RowOptions;
}

/**
 * The measured rows, in a deterministic nesting order — pointsPerSide, then
 * dtype, then batchSize, then decode path, then rep — so two runs of the same
 * config produce the same ids in the same order and their tables line up.
 */
export function expandGrid(config: SweepConfig): SweepRow[] {
  const rows: SweepRow[] = [];
  for (const pointsPerSide of config.pointsPerSide) {
    for (const dtype of config.dtypes) {
      for (const batchSize of config.batchSizes) {
        for (const path of config.decodePaths) {
          for (let rep = 1; rep <= config.reps; rep += 1) {
            rows.push({
              id: `p${pointsPerSide}-${dtype}-b${batchSize}-${path}-r${rep}`,
              rep,
              warmUp: false,
              options: {
                dtype,
                batchSize,
                pointsPerSide,
                keepRawMasks: config.keepRawMasks,
                ...decodeFlags(path),
              },
            });
          }
        }
      }
    }
  }
  return rows;
}

/**
 * A copy of the first row under a reserved id.
 *
 * The runner walks `[warmUpRow(rows), ...rows]` and throws the first result
 * away: the very first run in a page pays cold shader compilation and a cold
 * HTTP cache, and that cost belongs to no configuration. Keeping it out of
 * `expandGrid` is what lets AC6 say "the default grid is exactly 16 rows".
 */
export function warmUpRow(rows: readonly SweepRow[]): SweepRow {
  if (rows.length === 0) throw new Error('cannot build a warm-up row from an empty grid');
  return { ...rows[0], id: 'warm-up', warmUp: true };
}

// -------------------------------------------------------------------- records

/**
 * One completed run, as the page captures it and the runner collects it.
 *
 * Deliberately NOT the whole `SegmentationResult`: `segments` carries ~50 PNG
 * data URLs and `rawMasks` carries ~35 MB of coverage, neither of which can go
 * through a `<pre>` and back over CDP for every row. Everything a number in
 * the report is derived from IS here.
 */
export interface RunRecord {
  /**
   * The page emits `''` — it drives one configuration at a time and has no
   * grid — and the runner overwrites it with the `SweepRow.id` it asked for.
   */
  rowId: string;
  /** The options the run was ACTUALLY made with, captured before it started. */
  options: RowOptions;
  status: 'ok' | 'failed';
  timings?: TimingReport;
  counts?: SegmentationCounts;
  /** `timings.totalMs - timings.phases['model-load'].total`. */
  budgetMs?: number;
  agreement?: MaskAgreement;
  phase?: SegmentationPhase | 'unknown';
  message?: string;
}

/**
 * The ranking metric: wall clock minus the one-time model load.
 *
 * `createSegmenter` terminates and respawns the worker for every run, so every
 * row pays a warm (HTTP-cached) model load that is not part of the per-image
 * cost. This matches issue #1's budget convention.
 */
export function budgetMsOf(timings: TimingReport): number {
  return timings.totalMs - timings.phases['model-load'].total;
}

/**
 * Ok rows by ascending `budgetMs`, then every failed row in input order.
 *
 * CONVENTION: a budget tie keeps input (grid) order — `Array.prototype.sort`
 * is stable in every runtime this ships to — so the ranking is reproducible.
 */
export function rankRecords(records: readonly RunRecord[]): RunRecord[] {
  const ok = records.filter((record) => record.status === 'ok');
  const failed = records.filter((record) => record.status !== 'ok');
  return [...[...ok].sort((a, b) => (a.budgetMs ?? Infinity) - (b.budgetMs ?? Infinity)), ...failed];
}

export function fastestRecord(records: readonly RunRecord[]): RunRecord | null {
  const ranked = rankRecords(records);
  return ranked.length > 0 && ranked[0].status === 'ok' ? ranked[0] : null;
}

/** Derived from the record's OWN options — never from current control state. */
export function rowLabel(record: RunRecord): string {
  const o = record.options;
  return `${decodePathOf(o)} · ${o.dtype} · batch ${o.batchSize} · pps ${o.pointsPerSide}`;
}

// ------------------------------------------------------------------ agreement

/**
 * The IoU at or above which two masks are the same mask. Below it a pair is
 * NOT a bad match — it is unmatched on both sides, because a low-IoU pairing
 * says the two runs found different things, not the same thing badly.
 */
export const IOU_MATCH_FLOOR = 0.9;

export interface MaskAgreement {
  baselineCount: number;
  variantCount: number;
  matched: number;
  unmatchedBaseline: number;
  unmatchedVariant: number;
  /** Over the matched pairs only; 0 when nothing matched. */
  meanIou: number;
  /**
   * The LOWER median: with an even number of matches this is the lower of the
   * two middle IoUs, not their average. Every value here is an IoU that was
   * actually measured on a real pair of masks, which is what a human reading
   * the table wants; an averaged midpoint would be a number no pair scored.
   */
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
 * IoU comes from `pairwiseIoU` in `src/segmenter/core` — the ONE definition in
 * this repo. `RawMask` is structurally a `BinaryMask`, so it passes straight in.
 *
 * REPORTED, NEVER ENFORCED: the sweep ranks on time and does not gate on mask
 * equality. Correctness of the decode paths is a matter for review and unit
 * tests, not for a benchmark.
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

// --------------------------------------------------------------------- report

export interface AdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  /** True when the identity matched a known software rasterizer. */
  software: boolean;
}

export interface ReportMeta {
  config: SweepConfig;
  /** Absent on a hand run from the tab, which cannot probe the adapter. */
  adapter?: AdapterInfo;
  generatedAt: string;
}

export const OVERLAP_CAVEAT =
  '`overlapDecodeFilter` does not make decode faster. It keeps the next batch on the GPU ' +
  'while the current one is filtered on the CPU, so time MOVES BETWEEN the stage counters. ' +
  'A near-zero `decode` on an overlap row means the GPU finished during the previous filter ' +
  'block — the work happened, it stopped being counted. Rank on `budget` (wall clock minus ' +
  '`model-load`). The `decode` column is not the ranking.';

const DASH = '—';

function ms(value: number | undefined): string {
  return value === undefined ? DASH : value.toFixed(1);
}

function describeAdapter(adapter: AdapterInfo | undefined): string {
  if (!adapter) return 'not recorded (hand run from the Compare tab)';
  const identity = [adapter.vendor, adapter.architecture, adapter.device, adapter.description]
    .filter(Boolean)
    .join(' / ');
  return `${identity || 'unidentified'}${adapter.software ? ' — SOFTWARE RASTERIZER' : ''}`;
}

const COLUMNS = [
  'rank', 'row', 'decode path', 'dtype', 'batch', 'pps', 'budget', 'total',
  'model-load', 'encode', 'decode', 'filter', 'nms', 'mask-encode',
  'raw', 'afterFilter', 'afterNms', 'status',
] as const;

/** The committed report. Ranked on `budget`; the caveat sits above the table. */
export function toMarkdown(records: readonly RunRecord[], meta: ReportMeta): string {
  const ranked = rankRecords(records);
  const fastest = fastestRecord(records);
  const okCount = records.filter((record) => record.status === 'ok').length;

  const lines: string[] = [];
  lines.push(`# decode sweep — ${meta.generatedAt}`, '');
  lines.push(`- adapter: ${describeAdapter(meta.adapter)}`);
  lines.push(`- grid: \`${JSON.stringify(meta.config)}\``);
  lines.push(`- rows: ${records.length} (${okCount} ok, ${records.length - okCount} failed)`);
  lines.push('');
  lines.push(`> ${OVERLAP_CAVEAT}`);
  lines.push('');
  lines.push(
    fastest
      ? `**Fastest: ${rowLabel(fastest)} — ${ms(fastest.budgetMs)} ms budget (row \`${fastest.rowId}\`).**`
      : '**No row completed, so there is no fastest configuration.**',
  );
  lines.push('');
  lines.push(`| ${COLUMNS.join(' | ')} |`);
  lines.push(`| ${COLUMNS.map((_, i) => (i <= 5 || i === 17 ? '---' : '---:')).join(' | ')} |`);

  let rank = 0;
  for (const record of ranked) {
    const o = record.options;
    const p = record.timings?.phases;
    const ok = record.status === 'ok';
    if (ok) rank += 1;
    lines.push(
      `| ${[
        ok ? String(rank) : DASH,
        record.rowId,
        decodePathOf(o),
        o.dtype,
        String(o.batchSize),
        String(o.pointsPerSide),
        ms(record.budgetMs),
        ms(record.timings?.totalMs),
        ms(p?.['model-load'].total),
        ms(p?.encode.total),
        ms(p?.decode.total),
        ms(p?.filter.total),
        ms(p?.nms.total),
        ms(p?.['mask-encode'].total),
        record.counts ? String(record.counts.raw) : DASH,
        record.counts ? String(record.counts.afterFilter) : DASH,
        record.counts ? String(record.counts.afterNms) : DASH,
        ok ? 'ok' : `failed in ${record.phase ?? 'unknown'}: ${record.message ?? ''}`,
      ].join(' | ')} |`,
    );
  }

  lines.push('');
  lines.push(
    `All times in ms. \`budget\` = \`total\` − \`model-load\`; every row respawns the worker ` +
      `and pays its own warm, HTTP-cached model load. Phase columns are stage TOTALS over ` +
      `${PHASE_ORDER.length} phases. \`raw\`/\`afterFilter\`/\`afterNms\` are mask counts: a row ` +
      `that is fast because it silently dropped masks is visible here.`,
  );

  const withAgreement = records.filter((record) => record.agreement);
  if (withAgreement.length > 0) {
    lines.push('', '### mask agreement against the baseline row', '');
    lines.push(
      '| row | baseline masks | variant masks | matched | unmatched baseline | unmatched variant | mean IoU | median IoU | min IoU |',
    );
    lines.push('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |');
    for (const record of withAgreement) {
      const a = record.agreement!;
      lines.push(
        `| ${record.rowId} | ${a.baselineCount} | ${a.variantCount} | ${a.matched} | ` +
          `${a.unmatchedBaseline} | ${a.unmatchedVariant} | ${a.meanIou.toFixed(3)} | ` +
          `${a.medianIou.toFixed(3)} | ${a.minIou.toFixed(3)} |`,
      );
    }
    lines.push(
      '',
      `Pairing is greedy best-IoU; a pair below IoU ${IOU_MATCH_FLOOR} counts as unmatched on ` +
        `both sides. REPORTED, NOT ENFORCED — the sweep ranks on time.`,
    );
  }

  return `${lines.join('\n')}\n`;
}

// ----------------------------------------------------------------- page hook

/**
 * The surface `scripts/sweep-decode.mjs` evaluates in the page.
 *
 * It exists so the runner has no second copy of grid expansion or markdown
 * rendering in an `.mjs` file that no test would reach. PLAYGROUND-ONLY:
 * nothing under `src/` knows it exists, `CompareView` installs it, and
 * `scripts/smoke-build.mjs`'s guarantee about the published package surface is
 * untouched.
 */
export interface DecodeSweepHook {
  DEFAULT_SWEEP_CONFIG: SweepConfig;
  resolveConfig: typeof resolveConfig;
  expandGrid: typeof expandGrid;
  warmUpRow: typeof warmUpRow;
  toMarkdown: typeof toMarkdown;
}

declare global {
  interface Window {
    __decodeSweep?: DecodeSweepHook;
  }
}
```

- [ ] **Step 7: Run the test**

Run: `npx vitest run playground/compare.test.ts`
Expected: PASS.

- [ ] **Step 8: Run the full suite and both typecheck projects**

Run: `npm test`
Expected: PASS.

Run: `npm run typecheck`
Expected: exit 0 — and this now covers `playground/compare.ts`. Prove the coverage is real rather than assumed: temporarily add `const broken: number = 'x';` to the top of `playground/compare.ts`, run `npm run typecheck`, confirm it FAILS naming that file, then delete the line and confirm it passes again.

- [ ] **Step 9: Commit**

```bash
git add playground/compare.ts playground/compare.test.ts tsconfig.playground.json package.json
git commit -m "feat(playground): the decode sweep's grid, ranking and report"
```

---

### Task 3: The Compare tab, and bringing the playground under checks that run

**Files:**
- Create: `playground/CompareView.tsx`
- Create: `playground/CompareView.test.tsx`
- Modify: `playground/main.tsx`
- Modify: `playground/SegmentView.tsx`
- Modify: `vitest.config.ts`

**Interfaces:**
- Consumes: everything Task 2 exported from `./compare`; `createSegmenter`, `isWebGPUAvailable`, `SegmenterFailure`, `DEFAULT_SEGMENTER_OPTIONS`, `type Segmenter`, `type SegmenterProgress` from `../src/segmenter`.
- Produces:
  - `CompareView(props: CompareViewProps)` where
    `interface CompareViewProps { createSegmenter?: () => Segmenter; createBitmap?: () => Promise<ImageBitmap> }`
  - The DOM contract the runner depends on, all as `data-testid`: `dtype`, `batch-size`, `points-per-side`, `overlap-decode-filter`, `gpu-resident-embeddings`, `keep-raw-masks`, `run-row`, `run-json`, `compare-table`, `copy-markdown`, `agreement-panel`, and on `main.tsx` the tab button `view-compare`.
  - `run-json` additionally carries **`data-run-count`**, an integer that increments once per completed run, success or failure. The runner waits on it; `CompareView.test.tsx` asserts it. It exists for no other reason.

- [ ] **Step 1: Add `.tsx` playground tests to `vitest.config.ts`**

```ts
    include: [
      'src/**/*.test.ts',
      'src/**/*.test.tsx',
      'playground/**/*.test.ts',
      'playground/**/*.test.tsx',
    ],
```

- [ ] **Step 2: Write the failing test — `playground/CompareView.test.tsx`**

```tsx
// @vitest-environment jsdom
//
// jsdom because CompareView is a React component. No WebGPU, no worker and no
// network: `createSegmenter` and `createBitmap` are both injected, mirroring
// the injection points `runRow` had on the PR #11 branch. `navigator.gpu` is
// stubbed truthy so the component renders its controls rather than the
// WebGPU-required panel.
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { CompareView } from './CompareView';
import { budgetMsOf, decodePathOf, type RunRecord } from './compare';
import {
  SegmenterFailure,
  createTimingAccumulator,
  type SegmentationResult,
  type Segmenter,
  type SegmenterOptions,
} from '../src/segmenter';

function stubResult(totalMs: number): SegmentationResult {
  const accumulator = createTimingAccumulator();
  accumulator.record('model-load', 400);
  accumulator.record('decode', 250);
  return {
    segments: [],
    timings: accumulator.report(totalMs),
    counts: { raw: 96, afterFilter: 31, afterNms: 12 },
  };
}

/** Records the options each call was made with, so AC7 can be checked. */
function stubSegmenter(outcome: SegmentationResult | Error) {
  const seen: Partial<SegmenterOptions>[] = [];
  const segmenter: Segmenter = {
    segment: vi.fn(async (_image, options) => {
      seen.push({ ...options });
      if (outcome instanceof Error) throw outcome;
      return outcome;
    }) as Segmenter['segment'],
    dispose: vi.fn(),
  };
  return { segmenter, seen };
}

function mount(outcome: SegmentationResult | Error) {
  const { segmenter, seen } = stubSegmenter(outcome);
  render(
    <CompareView
      createSegmenter={() => segmenter}
      createBitmap={async () => ({ width: 4, height: 4 }) as ImageBitmap}
    />,
  );
  return { segmenter, seen };
}

function runJson(): HTMLElement {
  return screen.getByTestId('run-json');
}

function runCount(): number {
  return Number(runJson().getAttribute('data-run-count'));
}

beforeEach(() => {
  vi.stubGlobal('navigator', { ...navigator, gpu: {}, clipboard: { writeText: vi.fn() } });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('CompareView', () => {
  it('exposes every control the sweep drives (AC9)', () => {
    mount(stubResult(1400));
    for (const id of [
      'dtype', 'batch-size', 'points-per-side', 'overlap-decode-filter',
      'gpu-resident-embeddings', 'keep-raw-masks', 'run-row', 'run-json', 'copy-markdown',
    ]) {
      expect(screen.getByTestId(id)).toBeTruthy();
    }
    // batchSize 64 is reachable by hand even though it is out of the default grid.
    const batch = screen.getByTestId('batch-size') as HTMLSelectElement;
    expect([...batch.options].map((option) => option.value)).toEqual(['8', '16', '32', '64']);
  });

  it('installs the __decodeSweep hook the runner reaches through', () => {
    mount(stubResult(1400));
    expect(typeof window.__decodeSweep?.expandGrid).toBe('function');
    expect(typeof window.__decodeSweep?.toMarkdown).toBe('function');
    expect(typeof window.__decodeSweep?.resolveConfig).toBe('function');
    expect(typeof window.__decodeSweep?.warmUpRow).toBe('function');
  });

  it('runs the injected segmenter with the controls and emits a run-json blob', async () => {
    const result = stubResult(1400);
    const { seen } = mount(result);

    fireEvent.change(screen.getByTestId('dtype'), { target: { value: 'fp16' } });
    fireEvent.change(screen.getByTestId('batch-size'), { target: { value: '32' } });
    fireEvent.click(screen.getByTestId('overlap-decode-filter'));
    expect(runCount()).toBe(0);
    fireEvent.click(screen.getByTestId('run-row'));

    await waitFor(() => expect(runCount()).toBe(1));
    expect(seen[0]).toMatchObject({
      dtype: 'fp16',
      batchSize: 32,
      overlapDecodeFilter: true,
      gpuResidentEmbeddings: false,
    });

    const record = JSON.parse(runJson().textContent!) as RunRecord;
    expect(record.status).toBe('ok');
    expect(record.counts).toEqual(result.counts);
    expect(record.timings!.totalMs).toBe(1400);
    expect(record.budgetMs).toBe(budgetMsOf(result.timings));
    expect(decodePathOf(record.options)).toBe('overlap');
    // The heavy fields never go through the <pre>.
    expect('segments' in record).toBe(false);
    expect('rawMasks' in record).toBe(false);
  });

  it('labels a captured row from its OWN options after the controls move on (AC7)', async () => {
    mount(stubResult(1400));
    fireEvent.click(screen.getByTestId('run-row'));
    await waitFor(() => expect(runCount()).toBe(1));

    // The run above was fp32. Move every control afterwards.
    fireEvent.change(screen.getByTestId('dtype'), { target: { value: 'fp16' } });
    fireEvent.change(screen.getByTestId('batch-size'), { target: { value: '32' } });
    fireEvent.click(screen.getByTestId('gpu-resident-embeddings'));

    const row = screen.getByTestId('compare-table').querySelectorAll('tbody tr')[0];
    expect(row.textContent).toContain('fp32');
    expect(row.textContent).toContain('none');
    expect(row.textContent).not.toContain('fp16');
    // And the exported markdown agrees with the table.
    expect(JSON.parse(runJson().textContent!).options.dtype).toBe('fp32');
  });

  it('records a failed run as failed and still ticks the run counter (AC12)', async () => {
    mount(new SegmenterFailure('decode', 'Array buffer allocation failed'));
    fireEvent.click(screen.getByTestId('run-row'));

    await waitFor(() => expect(runCount()).toBe(1));
    const record = JSON.parse(runJson().textContent!) as RunRecord;
    expect(record.status).toBe('failed');
    expect(record.phase).toBe('decode');
    expect(record.message).toContain('allocation failed');
    expect(record.budgetMs).toBeUndefined();
    expect(screen.getByTestId('compare-table').textContent).toContain('decode');
  });
});
```

- [ ] **Step 3: Run it and watch it fail**

Run: `npx vitest run playground/CompareView.test.tsx`
Expected: FAIL — `Failed to resolve import "./CompareView"`.

- [ ] **Step 4: Write `playground/CompareView.tsx`**

```tsx
/**
 * The decode sweep's page: one configuration at a time, driven by hand or by
 * `scripts/sweep-decode.mjs`.
 *
 * Ported from the PR #11 branch and re-cut. Two things are load-bearing:
 *   - every rendered row is labelled from the options CAPTURED with its
 *     result, never from the current control state; and
 *   - `run-json` carries the completed run's exact numbers, so the runner
 *     never has to parse rounded display text.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  SegmenterFailure,
  createSegmenter as createRealSegmenter,
  isWebGPUAvailable,
  type RawMask,
  type Segmenter,
  type SegmenterProgress,
} from '../src/segmenter';
import {
  DEFAULT_SWEEP_CONFIG,
  budgetMsOf,
  compareMaskSets,
  decodePathOf,
  expandGrid,
  resolveConfig,
  rowLabel,
  toMarkdown,
  warmUpRow,
  type RowOptions,
  type RunRecord,
} from './compare';
import sampleUrl from './sample/cafe-table.jpg';

const DTYPE_CHOICES = ['fp32', 'fp16'] as const;
/**
 * 64 is offered even though `DEFAULT_SWEEP_CONFIG` excludes it: issue #12
 * measured it dying inside `post_process_masks`, and re-confirming that by
 * hand should not require editing code.
 */
const BATCH_SIZE_CHOICES = [8, 16, 32, 64] as const;
const POINTS_PER_SIDE_CHOICES = [16, 32] as const;

const INITIAL_OPTIONS: RowOptions = {
  dtype: 'fp32',
  batchSize: 8,
  pointsPerSide: 16,
  overlapDecodeFilter: false,
  gpuResidentEmbeddings: false,
  keepRawMasks: false,
};

function ms(value: number | undefined): string {
  return value === undefined ? '—' : value.toFixed(1);
}

export interface CompareViewProps {
  /** Injected by tests so the component runs with no WebGPU and no worker. */
  createSegmenter?: () => Segmenter;
  /** Injected by tests: jsdom has no `createImageBitmap`. */
  createBitmap?: () => Promise<ImageBitmap>;
}

export function CompareView({ createSegmenter, createBitmap }: CompareViewProps = {}) {
  // Probed once: a GPU adapter does not appear part-way through a session.
  const webgpu = useMemo(() => isWebGPUAvailable(), []);
  const [options, setOptions] = useState<RowOptions>(INITIAL_OPTIONS);
  const [records, setRecords] = useState<RunRecord[]>([]);
  const [last, setLast] = useState<RunRecord | null>(null);
  /**
   * Incremented once per COMPLETED run, success or failure.
   *
   * This is the runner's synchronisation point: it reads the attribute, clicks
   * run, and waits for the number to grow. Without it the runner would have to
   * poll the blob's contents and guess whether it is looking at this run's
   * result or the previous one's.
   */
  const [runCount, setRunCount] = useState(0);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<SegmenterProgress | null>(null);
  const [copied, setCopied] = useState(false);
  const segmenterRef = useRef<Segmenter | null>(null);
  /**
   * The first retained mask set, and only that one.
   *
   * Bounded on purpose: holding every row's `RawMask[]` would pin ~35 MB per
   * row for the session. Each later row is compared against this baseline and
   * then its own masks are DROPPED — only the ten-number summary survives.
   */
  const baselineRef = useRef<{ rowId: string; masks: RawMask[] } | null>(null);

  useEffect(() => {
    window.__decodeSweep = {
      DEFAULT_SWEEP_CONFIG,
      resolveConfig,
      expandGrid,
      warmUpRow,
      toMarkdown,
    };
    return () => {
      delete window.__decodeSweep;
    };
  }, []);

  useEffect(
    () => () => {
      segmenterRef.current?.dispose();
    },
    [],
  );

  const patch = (change: Partial<RowOptions>) =>
    setOptions((current) => ({ ...current, ...change }));

  const runRow = useCallback(async () => {
    setRunning(true);
    setProgress(null);
    // Captured BEFORE the run, and never read from state again: this object is
    // what labels the row, whatever the controls do next.
    const captured: RowOptions = { ...options };
    let record: RunRecord;
    try {
      const bitmap = await (createBitmap
        ? createBitmap()
        : createImageBitmap(await (await fetch(sampleUrl)).blob()));
      // Created lazily so a browser with no WebGPU never spawns a worker.
      segmenterRef.current ??= (createSegmenter ?? createRealSegmenter)();
      const result = await segmenterRef.current.segment(bitmap, captured, setProgress);
      record = {
        rowId: '',
        options: captured,
        status: 'ok',
        timings: result.timings,
        counts: result.counts,
        budgetMs: budgetMsOf(result.timings),
      };
      if (result.rawMasks) {
        if (!baselineRef.current) {
          baselineRef.current = { rowId: rowLabel(record), masks: result.rawMasks };
        } else {
          record.agreement = compareMaskSets(baselineRef.current.masks, result.rawMasks);
        }
        // `result` goes out of scope here, so every mask set except the
        // baseline's is released as soon as its summary has been taken.
      }
    } catch (thrown) {
      const failure = thrown instanceof SegmenterFailure ? thrown : null;
      record = {
        rowId: '',
        options: captured,
        status: 'failed',
        phase: failure?.phase ?? 'unknown',
        message: thrown instanceof Error ? thrown.message : String(thrown),
      };
    }
    setRecords((current) => [...current, record]);
    setLast(record);
    setRunning(false);
    setProgress(null);
    // Last, so a runner woken by this attribute always finds the blob updated.
    setRunCount((current) => current + 1);
  }, [createBitmap, createSegmenter, options]);

  async function copyMarkdown() {
    await navigator.clipboard.writeText(
      toMarkdown(records, {
        config: DEFAULT_SWEEP_CONFIG,
        generatedAt: new Date().toISOString(),
      }),
    );
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  const agreements = records.filter((record) => record.agreement);

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
          There is no CPU fallback by design — on CPU a single pass takes minutes.
        </p>
      </section>
    );
  }

  return (
    <>
      <p>
        One decode configuration per run, on the bundled sample image. Every run
        respawns the worker and pays its own <code>model-load</code>, which is
        reported separately and excluded from the budget.{' '}
        <strong>
          overlapDecodeFilter moves time between the stage counters rather than
          removing it — rank on budget, not on the decode row.
        </strong>
      </p>

      <p>
        <label>
          dtype:{' '}
          <select
            data-testid="dtype"
            value={options.dtype}
            disabled={running}
            onChange={(e) => patch({ dtype: e.target.value as RowOptions['dtype'] })}
          >
            {DTYPE_CHOICES.map((value) => (
              <option key={value} value={value}>{value}</option>
            ))}
          </select>
        </label>{' '}
        <label>
          batch size:{' '}
          <select
            data-testid="batch-size"
            value={options.batchSize}
            disabled={running}
            onChange={(e) => patch({ batchSize: Number(e.target.value) })}
          >
            {BATCH_SIZE_CHOICES.map((value) => (
              <option key={value} value={value}>{value}</option>
            ))}
          </select>
        </label>{' '}
        <label>
          points per side:{' '}
          <select
            data-testid="points-per-side"
            value={options.pointsPerSide}
            disabled={running}
            onChange={(e) => patch({ pointsPerSide: Number(e.target.value) })}
          >
            {POINTS_PER_SIDE_CHOICES.map((value) => (
              <option key={value} value={value}>{value}</option>
            ))}
          </select>
        </label>{' '}
        <label>
          overlap decode/filter:{' '}
          <input
            data-testid="overlap-decode-filter"
            type="checkbox"
            checked={options.overlapDecodeFilter}
            disabled={running}
            onChange={(e) => patch({ overlapDecodeFilter: e.target.checked })}
          />
        </label>{' '}
        <label>
          GPU-resident embeddings:{' '}
          <input
            data-testid="gpu-resident-embeddings"
            type="checkbox"
            checked={options.gpuResidentEmbeddings}
            disabled={running}
            onChange={(e) => patch({ gpuResidentEmbeddings: e.target.checked })}
          />
        </label>{' '}
        <label>
          keep raw masks (~35 MB per run):{' '}
          <input
            data-testid="keep-raw-masks"
            type="checkbox"
            checked={options.keepRawMasks}
            disabled={running}
            onChange={(e) => patch({ keepRawMasks: e.target.checked })}
          />
        </label>
      </p>

      <p>
        <button data-testid="run-row" type="button" disabled={running} onClick={() => void runRow()}>
          {running ? 'Running…' : 'Run this configuration'}
        </button>{' '}
        <button
          data-testid="copy-markdown"
          type="button"
          disabled={records.length === 0}
          onClick={() => void copyMarkdown()}
        >
          {copied ? 'Copied' : 'Copy as markdown'}
        </button>
      </p>

      {running && (
        <p data-testid="compare-progress" role="status">
          {progress
            ? `${progress.phase} ${progress.done}/${progress.total} — ${ms(progress.ms)} ms`
            : 'starting…'}
        </p>
      )}

      <table data-testid="compare-table" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th align="left">decode path</th>
            <th align="left">dtype</th>
            <th align="right">batch</th>
            <th align="right">pps</th>
            <th align="right">budget</th>
            <th align="right">total</th>
            <th align="right">decode</th>
            <th align="right">filter</th>
            <th align="right">kept</th>
            <th align="left">status</th>
          </tr>
        </thead>
        <tbody>
          {records.map((record, index) => {
            // Every cell reads `record.options` — the options this run was
            // MEASURED under. Reading `options` here instead would relabel old
            // numbers whenever a control moved.
            const o = record.options;
            const p = record.timings?.phases;
            return (
              <tr key={index} data-testid={`compare-row-${index}`}>
                <td>{decodePathOf(o)}</td>
                <td>{o.dtype}</td>
                <td align="right">{o.batchSize}</td>
                <td align="right">{o.pointsPerSide}</td>
                <td align="right"><strong>{ms(record.budgetMs)}</strong></td>
                <td align="right">{ms(record.timings?.totalMs)}</td>
                <td align="right">{ms(p?.decode.total)}</td>
                <td align="right">{ms(p?.filter.total)}</td>
                <td align="right">{record.counts?.afterNms ?? '—'}</td>
                <td>
                  {record.status === 'ok' ? (
                    'ok'
                  ) : (
                    <span role="alert">
                      failed in <strong>{record.phase}</strong>: {record.message}
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {agreements.length > 0 && (
        <section data-testid="agreement-panel">
          <h2 style={{ fontSize: '1rem' }}>mask agreement against the baseline row</h2>
          {agreements.map((record, index) => {
            const a = record.agreement!;
            return (
              <p key={index}>
                <strong>{rowLabel(record)}</strong>: {a.baselineCount} baseline masks vs{' '}
                {a.variantCount}, {a.matched} matched at IoU ≥ {a.floor} (
                {a.unmatchedBaseline} / {a.unmatchedVariant} unmatched). IoU mean{' '}
                {a.meanIou.toFixed(3)}, median {a.medianIou.toFixed(3)}, min{' '}
                {a.minIou.toFixed(3)}. Reported, not enforced.
              </p>
            );
          })}
        </section>
      )}

      {/*
        The completed run's exact numbers. `data-run-count` is the runner's
        synchronisation point; the text is a `RunRecord` minus `segments` and
        `rawMasks`, which are megabytes and would have to cross CDP per row.
      */}
      <pre data-testid="run-json" data-run-count={runCount} style={{ overflowX: 'auto' }}>
        {last ? JSON.stringify(last) : ''}
      </pre>
    </>
  );
}
```

- [ ] **Step 5: Run the test**

Run: `npx vitest run playground/CompareView.test.tsx`
Expected: PASS, 5 tests.

- [ ] **Step 6: Add the third tab in `playground/main.tsx`**

Import `CompareView`, widen the union, and add the button:

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

and replace the ternary body with:

```tsx
      {view === 'benchmark' && <BenchmarkView />}
      {view === 'segment' && <SegmentView />}
      {view === 'compare' && <CompareView />}
```

- [ ] **Step 7: Add the two decode-path checkboxes to `playground/SegmentView.tsx`**

Immediately after the existing `compare-nms` `<label>` (and before the closing `</p>`), so the manual tuning affordance exists on the tab people already use:

```tsx
        {' '}
        <label>
          overlap decode/filter:{' '}
          <input
            data-testid="overlap-decode-filter"
            type="checkbox"
            checked={options.overlapDecodeFilter}
            onChange={(e) => patch({ overlapDecodeFilter: e.target.checked })}
          />
        </label>
        {' '}
        <label>
          GPU-resident embeddings:{' '}
          <input
            data-testid="gpu-resident-embeddings"
            type="checkbox"
            checked={options.gpuResidentEmbeddings}
            onChange={(e) => patch({ gpuResidentEmbeddings: e.target.checked })}
          />
        </label>
```

- [ ] **Step 8: Run everything and prove the new checks actually cover `CompareView.tsx` (AC10)**

Run: `npm test`
Expected: PASS, including `playground/CompareView.test.tsx`.

Run: `npm run typecheck`
Expected: exit 0.

Now falsify both claims rather than assuming them:

1. Add `const broken: number = 'x';` at the top of `playground/CompareView.tsx`. Run `npm run typecheck`. Expected: FAIL, naming `playground/CompareView.tsx`. Remove the line.
2. Change the first assertion in `CompareView.test.tsx`'s "exposes every control" case to `expect(screen.getByTestId('dtype-nope')).toBeTruthy()`. Run `npm test`. Expected: FAIL, naming `playground/CompareView.test.tsx`. Revert it.
3. Run `npm run typecheck && npm test` once more. Expected: both pass.

- [ ] **Step 9: Commit**

```bash
git add playground vitest.config.ts
git commit -m "feat(playground): Compare tab, decode-path checkboxes, playground under typecheck and test"
```

---

### Task 4: `scripts/sweep-decode.mjs` — the runner

**Files:**
- Create: `scripts/sweep-decode.mjs`
- Modify: `package.json` (the `sweep:decode` script)

**Interfaces:**
- Consumes: `createServer` from `vite`; `chromium` from `@playwright/test`; `window.__decodeSweep` (`DEFAULT_SWEEP_CONFIG`, `resolveConfig`, `expandGrid`, `warmUpRow`, `toMarkdown`) and the `data-testid` contract, both from Task 3.
- Produces: `npm run sweep:decode [-- --full] [-- --reps N] [-- --config path] [-- --headless] [-- --allow-software]`, writing `docs/measurements/<YYYY-MM-DD>-decode-sweep.md` and `.json`.

- [ ] **Step 1: Confirm the Chromium binary Playwright will drive is installed**

Run: `node -e "import('@playwright/test').then(m => console.log(m.chromium.executablePath()))"`
Expected: a path. If it prints a path that does not exist, or throws, run `npx playwright install chromium` before continuing. `@playwright/test@1.62.1` is already a devDependency — **do not add a package.**

- [ ] **Step 2: Write `scripts/sweep-decode.mjs`**

```js
/**
 * The decode sweep: a real-GPU, config-driven walk of the decode grid.
 *
 * Uses the Playwright LIBRARY, not the test runner. The test runner's retries,
 * timeouts and parallelism fight a serial GPU benchmark — and a retried GPU row
 * is a different measurement, not the same one again.
 *
 * No sweep logic lives here. Grid expansion, ranking and markdown all come from
 * `playground/compare.ts` THROUGH THE PAGE, via `window.__decodeSweep`, so
 * there is exactly one implementation and it is the unit-tested one.
 *
 *   npm run sweep:decode
 *   npm run sweep:decode -- --full --reps 3
 *   npm run sweep:decode -- --config my-grid.json
 *   npm run sweep:decode -- --headless --allow-software   # wiring smoke test only
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';
import { createServer } from 'vite';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

/** Generous: an fp32 batch-8 row at 32 points per side is minutes of GPU work. */
const ROW_TIMEOUT_MS = 15 * 60 * 1000;

/**
 * Known software rasterizers. A software run's timings are meaningless next to
 * a hardware run's, so the sweep refuses to produce them by accident.
 */
const SOFTWARE_ADAPTER = /swiftshader|lavapipe|llvmpipe|warp|basic render|software/i;

function parseArgs(argv) {
  const args = { full: false, headless: false, allowSoftware: false, reps: undefined, config: undefined };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--full') args.full = true;
    else if (arg === '--headless') args.headless = true;
    else if (arg === '--allow-software') args.allowSoftware = true;
    else if (arg === '--reps') args.reps = Number(argv[(i += 1)]);
    else if (arg === '--config') args.config = argv[(i += 1)];
    else throw new Error(`unknown argument: ${arg}`);
  }
  return args;
}

function die(message) {
  console.error(`sweep-decode: ${message}`);
  process.exit(1);
}

async function probeAdapter(page) {
  return page.evaluate(async () => {
    const gpu = navigator.gpu;
    if (!gpu) return null;
    const adapter = await gpu.requestAdapter();
    if (!adapter) return null;
    // `adapter.info` is the current API; `requestAdapterInfo()` is the older
    // one. Read whichever this Chromium ships.
    const info =
      adapter.info ??
      (typeof adapter.requestAdapterInfo === 'function' ? await adapter.requestAdapterInfo() : {});
    return {
      vendor: info.vendor ?? '',
      architecture: info.architecture ?? '',
      device: info.device ?? '',
      description: info.description ?? '',
    };
  });
}

async function setCheckbox(page, testId, value) {
  await page.locator(`[data-testid="${testId}"]`).setChecked(value);
}

async function runOneRow(page, row) {
  const o = row.options;
  await page.selectOption('[data-testid="dtype"]', o.dtype);
  await page.selectOption('[data-testid="batch-size"]', String(o.batchSize));
  await page.selectOption('[data-testid="points-per-side"]', String(o.pointsPerSide));
  await setCheckbox(page, 'overlap-decode-filter', o.overlapDecodeFilter);
  await setCheckbox(page, 'gpu-resident-embeddings', o.gpuResidentEmbeddings);
  await setCheckbox(page, 'keep-raw-masks', o.keepRawMasks);

  const before = Number(await page.getAttribute('[data-testid="run-json"]', 'data-run-count'));
  await page.click('[data-testid="run-row"]');
  // The page ticks this counter LAST, after the blob is written, so seeing it
  // grow means this run's numbers — not the previous run's — are readable.
  await page.waitForFunction(
    (n) =>
      Number(
        document.querySelector('[data-testid="run-json"]')?.getAttribute('data-run-count') ?? -1,
      ) > n,
    before,
    { timeout: ROW_TIMEOUT_MS },
  );

  const blob = await page.textContent('[data-testid="run-json"]');
  // The page drives one configuration at a time and has no grid, so it emits an
  // empty rowId; the id belongs to the row WE asked for.
  return { ...JSON.parse(blob), rowId: row.id };
}

function outputStem(dir, date) {
  let stem = `${date}-decode-sweep`;
  let n = 1;
  // A second run on the same day gets -2, -3 rather than silently overwriting.
  while (existsSync(path.join(dir, `${stem}.json`)) || existsSync(path.join(dir, `${stem}.md`))) {
    n += 1;
    stem = `${date}-decode-sweep-${n}`;
  }
  return stem;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  const server = await createServer({
    configFile: path.join(root, 'playground', 'vite.config.ts'),
    // Port 0: never fight a dev server the developer already has running.
    server: { port: 0, strictPort: false },
  });
  await server.listen();
  const url = server.resolvedUrls?.local?.[0];
  if (!url) {
    await server.close();
    die('vite started but reported no local URL');
  }

  // HEADED by default. Headless Chromium frequently resolves no real WebGPU
  // adapter and silently falls back to software, and software timings would
  // make the whole sweep meaningless. --headless is for smoke-testing wiring.
  const browser = await chromium.launch({ headless: args.headless });
  let exitCode = 0;

  try {
    const page = await browser.newPage();
    page.on('pageerror', (error) => console.error(`page error: ${error.message}`));
    await page.goto(url, { waitUntil: 'load' });
    await page.click('[data-testid="view-compare"]');
    await page.waitForSelector('[data-testid="run-json"]');

    // ---- the adapter gate, BEFORE the first measurement.
    const probed = await probeAdapter(page);
    if (!probed) die('no WebGPU adapter in this browser — refusing to measure');
    const software = SOFTWARE_ADAPTER.test(Object.values(probed).join(' '));
    const adapter = { ...probed, software };
    if (software && !args.allowSoftware) {
      die(
        `adapter looks like a software rasterizer (${JSON.stringify(adapter)}); ` +
          `pass --allow-software to measure it anyway`,
      );
    }
    // Recorded either way, so a software run is self-identifying rather than
    // quietly comparable to a real one.
    console.log(`adapter: ${JSON.stringify(adapter)}`);

    // ---- the grid, expanded by the page's own tested code.
    const overrides = args.config
      ? JSON.parse(readFileSync(path.resolve(process.cwd(), args.config), 'utf8'))
      : {};
    const config = await page.evaluate(
      ({ overrides, flags }) => window.__decodeSweep.resolveConfig(overrides, flags),
      { overrides, flags: { full: args.full, reps: args.reps } },
    );
    const rows = await page.evaluate((config) => {
      const rows = window.__decodeSweep.expandGrid(config);
      // The warm-up runs first and is discarded: the first run in a page pays
      // cold shader compilation and a cold HTTP cache.
      return [window.__decodeSweep.warmUpRow(rows), ...rows];
    }, config);

    // ---- walk every row in ONE page, so the ONNX weights stay HTTP-cached.
    const records = [];
    for (const row of rows) {
      process.stdout.write(`${row.id} … `);
      const record = await runOneRow(page, row);
      if (row.warmUp) {
        console.log('discarded (warm-up)');
        continue;
      }
      records.push(record);
      // One unsupported path must not cost the other fifteen their row.
      console.log(
        record.status === 'ok'
          ? `${record.budgetMs.toFixed(0)} ms budget, ${record.counts.afterNms} masks`
          : `FAILED in ${record.phase}: ${record.message}`,
      );
    }

    const generatedAt = new Date().toISOString();
    const markdown = await page.evaluate(
      ({ records, meta }) => window.__decodeSweep.toMarkdown(records, meta),
      { records, meta: { config, adapter, generatedAt } },
    );

    const dir = path.join(root, 'docs', 'measurements');
    mkdirSync(dir, { recursive: true });
    const stem = outputStem(dir, generatedAt.slice(0, 10));
    // The JSON embeds the resolved config and the adapter, so the report is
    // reproducible from its own artifact.
    writeFileSync(
      path.join(dir, `${stem}.json`),
      `${JSON.stringify({ generatedAt, adapter, config, rows: records }, null, 2)}\n`,
    );
    writeFileSync(path.join(dir, `${stem}.md`), markdown);
    console.log(`\nwrote docs/measurements/${stem}.{md,json}`);

    // Non-zero only if EVERY row failed: a partial sweep is still a result.
    if (!records.some((record) => record.status === 'ok')) {
      console.error('every row failed');
      exitCode = 1;
    }
  } finally {
    await browser.close();
    await server.close();
  }

  process.exit(exitCode);
}

await main();
```

- [ ] **Step 3: Add the npm script**

In `package.json`, beside `playground`:

```json
    "sweep:decode": "node scripts/sweep-decode.mjs",
```

- [ ] **Step 4: Smoke-test the wiring without claiming a measurement (AC11's flags)**

Run: `npm run sweep:decode -- --headless --config /dev/null` — expect it to fail parsing `/dev/null` as JSON. That is fine; it proves argument handling runs. Then write a one-row config to a scratch path **outside the repo** and use it:

```bash
printf '{"decodePaths":["none"],"batchSizes":[8],"dtypes":["fp32"],"pointsPerSide":[16]}' > /tmp/sweep-smoke.json
npm run sweep:decode -- --headless --allow-software --config /tmp/sweep-smoke.json
```

Expected, in order: an `adapter:` line, `warm-up … discarded (warm-up)`, one measured row line, and `wrote docs/measurements/<date>-decode-sweep.{md,json}`. The row may well come back `FAILED` under headless software — that is a correct outcome, not a bug, and the script must still write both files and exit 0 only if at least one row was `ok`.

**Delete the smoke artifacts before committing** — the committed measurement comes from Task 5's headed run, not from this one:

```bash
rm -f docs/measurements/*-decode-sweep*.md docs/measurements/*-decode-sweep*.json
```

- [ ] **Step 5: Verify the adapter gate actually aborts**

Run: `npm run sweep:decode -- --headless` (no `--allow-software`).
Expected: either `no WebGPU adapter in this browser — refusing to measure` or `adapter looks like a software rasterizer …`, exit code 1, **and no files written under `docs/measurements/`**. Confirm with `git status --short docs/measurements` — it must be empty.

If headless Chromium on this machine happens to resolve a *hardware* adapter, this step cannot fire; say so in the commit body rather than forcing it, and confirm the gate by reading the code path instead.

- [ ] **Step 6: Confirm nothing stray was added to the tree**

Run: `git status --short`
Expected: only `scripts/sweep-decode.mjs` and `package.json`. No `test-results/`, no `playwright-report/`, no `docs/measurements/` entries. The runner writes only into `docs/measurements/`, which is committed deliberately, so **no `.gitignore` change belongs in this task.**

- [ ] **Step 7: Commit**

```bash
git add scripts/sweep-decode.mjs package.json
git commit -m "feat(sweep): headed real-GPU decode sweep runner"
```

---

### Task 5: The committed real-GPU run

**Files:**
- Create: `docs/measurements/<YYYY-MM-DD>-decode-sweep.md`
- Create: `docs/measurements/<YYYY-MM-DD>-decode-sweep.json`

**Interfaces:**
- Consumes: `npm run sweep:decode` from Task 4.
- Produces: the artifacts AC13 requires. Nothing later depends on them.

> **This task requires a machine with a hardware WebGPU adapter and takes real time** — sixteen rows plus a warm-up, each a full everything-mode pass. It is not produced by the headless `verify` stage, which is exactly why the runner refuses to produce numbers there.

- [ ] **Step 1: Run the sweep, headed, with no overrides**

Run: `npm run sweep:decode`
Expected: an `adapter:` line naming real hardware (on macOS, an Apple/Metal identity), `warm-up … discarded (warm-up)`, then sixteen row lines, then `wrote docs/measurements/…`.

**If the runner aborts at the adapter gate, STOP.** Report that this machine cannot produce AC13's evidence and leave the task incomplete. Do **not** pass `--allow-software`: a software row committed under this filename is worse than no row, because the next reader will compare it against a hardware run.

- [ ] **Step 2: Check the JSON against AC13**

Run: `node -e "const r=require('./docs/measurements/'+require('fs').readdirSync('docs/measurements').find(f=>f.endsWith('.json')));console.log(r.rows.length, r.adapter, JSON.stringify(r.config))"`

Expected: `16`, an adapter object with `software: false`, and the resolved config. Confirm no row has `rowId: 'warm-up'`:

Run: `grep -c '"warm-up"' docs/measurements/*-decode-sweep.json`
Expected: `0`.

- [ ] **Step 3: Read the markdown and check it against AC8 and AC13**

Open the generated `.md` and confirm, by eye:

1. The overlap caveat paragraph sits **above** the table.
2. A `**Fastest: …**` line names one configuration with a budget in ms.
3. Ranks run 1..N over the ok rows; any failed row shows `failed in <phase>: <message>` and carries no timing numbers.
4. `raw`, `afterFilter` and `afterNms` are populated for every ok row.
5. The adapter line names real hardware and does not say `SOFTWARE RASTERIZER`.
6. The grid line reproduces the resolved config.

- [ ] **Step 4: Note anything the numbers actually say**

If a decode path failed on every row (for example the runtime rejecting a GPU-resident decoder input), that is a legitimate result. Record it in the commit body — which path, which phase, which message — rather than re-running until it passes. The sweep exists to produce this answer.

- [ ] **Step 5: Commit**

```bash
git add docs/measurements
git commit -m "docs(measurements): committed real-GPU decode sweep

<adapter identity>. <N> ok, <M> failed. Fastest: <configuration> at <X> ms budget.
<Any path that failed, with its phase and message.>"
```

---

## Self-review notes

**Spec coverage.** AC1/AC5 → Task 1 steps 1, 8–10. AC2 → Task 1 steps 7, 12 (review-bound; see *Verification scope*). AC3 + AC4's output-name half → Task 1 steps 2–6; AC4's no-fallback half → Task 1 step 12. AC6 → Task 2. AC7 → Task 3 step 2, case 4. AC8 → Task 2 step 4, `toMarkdown` cases. AC9 → Task 3 steps 2, 4, 7. AC10 → Task 3 step 8 (falsified, not assumed). AC11 → Task 4 steps 2, 4, 5. AC12 → Task 4 step 2 (`warmUp` skip, per-row record, exit code) plus Task 3's failed-run case. AC13 → Task 5.

**Type consistency.** `RowOptions` has the same six members in `compare.ts`, `CompareView.tsx` and both test files. `RunRecord.rowId` is `''` from the page and overwritten by the runner in exactly one place (`runOneRow`). `budgetMsOf` is the only definition of the ranking metric, used by `CompareView`, `toMarkdown` and the tests. `decodePathOf` is the only place a label is derived from the two flags.
