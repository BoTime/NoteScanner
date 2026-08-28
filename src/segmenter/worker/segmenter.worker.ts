/**
 * Everything-mode SAM inference on WebGPU, in a module worker.
 *
 * A worker because the first run downloads tens of megabytes of ONNX weights
 * and then runs a GPU-bound decode loop; on the main thread that is a frozen
 * tab. Nothing here is reachable from `src/index.ts` or `src/core/index.ts` —
 * `scripts/smoke-build.mjs` asserts it, which is what keeps the package's
 * zero-runtime-dependency promise intact for every consumer who does not
 * import `/segmenter`.
 *
 * Pipeline: load once -> encode the image once -> decode the prompt grid in
 * batches -> filter each batch at LOW resolution -> resample only the
 * survivors straight to binary masks -> NMS across everything. Filtering
 * before resampling is not an optimization detail: a batch of 8 points yields
 * 24 low-res masks, and taking all of them to full image resolution is
 * hundreds of megabytes per batch.
 */
import {
  AutoProcessor,
  RawImage,
  SamModel,
  type SamProcessor,
  Tensor,
} from '@huggingface/transformers';

import {
  batchPoints,
  buildPointGrid,
  createTimingAccumulator,
  dedupeMasks,
  encodeMaskPng,
  resampleThresholdMask,
  stabilityScore,
  type BinaryMask,
  type EncodedMask,
  type FilterSubstep,
  type NmsComparison,
  type RawMask,
  type SegmentationPhase,
  type SegmenterOptions,
  type SegmenterRequest,
  type SegmenterResponse,
} from '../core';

// Not on the `note-scanner/segmenter` surface by design, so it is imported
// from the module rather than from the barrel.
import { dedupeMasksReference } from '../core/nms';

import { embeddingsSessionOptions, sessionCacheKey } from './session-key';

/**
 * `globalThis` is typed as a `Window` under this package's `lib: ["dom", ...]`,
 * so the worker's own `postMessage` overload is not visible. Narrowing through
 * one named alias is cheaper and more honest than pulling in the whole
 * `webworker` lib, which would collide with `dom` on dozens of names.
 */
const scope = globalThis as unknown as {
  postMessage(message: SegmenterResponse, transfer?: Transferable[]): void;
};

function post(message: SegmenterResponse, transfer?: Transferable[]): void {
  scope.postMessage(message, transfer);
}

interface Session {
  model: SamModel;
  processor: SamProcessor;
}

interface PadSize {
  height: number;
  width: number;
}

/**
 * The pad size the processor actually applied, resolved the way
 * `post_process_masks` resolves it internally (`pad_size ?? size`, each
 * `{height, width}`).
 *
 * Read at runtime rather than hardcoded to 1024: the fused resample's geometry
 * is only correct for the pad the processor used, and a different SAM
 * checkpoint may ship a different one. Both fields are typed `any` upstream,
 * so this narrows them itself rather than trusting the declaration.
 */
function resolvePadSize(processor: SamProcessor): PadSize {
  const imageProcessor = processor.image_processor as
    | { pad_size?: unknown; size?: unknown }
    | undefined;
  const candidate = imageProcessor?.pad_size ?? imageProcessor?.size;
  const size = candidate as Partial<PadSize> | undefined;
  if (!size || typeof size.height !== 'number' || typeof size.width !== 'number') {
    throw new Error(
      'processor exposes neither image_processor.pad_size nor image_processor.size ' +
        'as {height, width}; cannot compute mask resample geometry',
    );
  }
  return { height: size.height, width: size.width };
}

/**
 * Cached for the lifetime of the worker. The client terminates and respawns
 * the worker between runs, so a poisoned session (lost adapter, half-loaded
 * weights) is never reused; within one run this just avoids reloading.
 */
let session: Session | null = null;
let sessionKey = '';

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
  const processor = (await AutoProcessor.from_pretrained(
    options.modelId,
  )) as unknown as SamProcessor;

  session = { model, processor };
  sessionKey = key;
  return session;
}

async function run(request: SegmenterRequest): Promise<void> {
  const { bitmap, options } = request;
  const timings = createTimingAccumulator();
  const startedAt = performance.now();
  // Tracks which phase a throw belongs to, so the failure names where it died.
  let phase: SegmentationPhase = 'model-load';

  try {
    // ---- model-load: reported as its own phase so a slow first download
    // ---- never reads as a hung tab.
    let started = performance.now();
    const { model, processor } = await loadSession(options);
    let elapsed = performance.now() - started;
    timings.record('model-load', elapsed);
    post({ type: 'progress', event: { phase: 'model-load', done: 1, total: 1, ms: elapsed } });

    // ---- encode: exactly once. This is the whole reason everything-mode is
    // ---- viable at all; the grid below reuses these embeddings.
    phase = 'encode';
    started = performance.now();
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('OffscreenCanvas 2d context unavailable');
    ctx.drawImage(bitmap, 0, 0);
    const inputs = await processor(RawImage.fromCanvas(canvas));
    const originalSizes = inputs.original_sizes as [number, number][];
    const reshapedSizes = inputs.reshaped_input_sizes as [number, number][];
    const embeddings = await model.get_image_embeddings({ pixel_values: inputs.pixel_values });
    elapsed = performance.now() - started;
    timings.record('encode', elapsed);
    post({ type: 'progress', event: { phase: 'encode', done: 1, total: 1, ms: elapsed } });

    // `original_sizes` and `reshaped_input_sizes` are [height, width].
    const [originalHeight, originalWidth] = originalSizes[0];
    const [reshapedHeight, reshapedWidth] = reshapedSizes[0];
    // `phase` is still 'encode' here, so a processor with no usable pad size
    // surfaces as SegmenterFailure('encode', ...) — which is where the
    // processor came from — rather than as an unattributed throw.
    const { width: padWidth, height: padHeight } = resolvePadSize(processor);

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
        // A second cursor for the sub-regions. Each region ends exactly where
        // the next begins, so the two sum to the stage total.
        let subStarted = started;
        const recordSub = (step: FilterSubstep) => {
          const now = performance.now();
          timings.recordFilterSub(step, now - subStarted);
          subStarted = now;
        };
        // dims: [1, point_batch_size, masks_per_point, lowHeight, lowWidth].
        // Read them off the tensor rather than hardcoding 3 x 256 x 256.
        const dims = predMasks.dims;
        const pointCount = dims[1];
        const masksPerPoint = dims[2];
        const lowHeight = dims[3];
        const lowWidth = dims[4];
        const lowPixels = lowHeight * lowWidth;
        const logits = predMasks.data as Float32Array;
        const scores = iouScores.data as Float32Array;

        // One mask per prompt point: the most confident of the three that is
        // also stable. SAM emits three to disambiguate whole/part/subpart, and
        // keeping all three is how you end up with three copies of everything.
        const chosen: number[] = [];
        for (let p = 0; p < pointCount; p += 1) {
          let bestFlat = -1;
          let bestScore = -Infinity;
          for (let m = 0; m < masksPerPoint; m += 1) {
            const flat = p * masksPerPoint + m;
            rawCount += 1;
            const window = logits.subarray(flat * lowPixels, (flat + 1) * lowPixels);
            const stability = stabilityScore(
              window,
              options.maskThreshold,
              options.stabilityScoreOffset,
            );
            if (stability < options.stabilityScoreThreshold) continue;
            if (scores[flat] > bestScore) {
              bestScore = scores[flat];
              bestFlat = flat;
            }
          }
          if (bestFlat >= 0) chosen.push(bestFlat);
        }
        recordSub('select');

        if (chosen.length > 0) {
          // One call per surviving mask, straight from its low-res window to a
          // binary mask: no staging copy, no 5-D tensor, no ORT round-trip, no
          // padded-resolution intermediate and no full-resolution float buffer.
          for (let k = 0; k < chosen.length; k += 1) {
            const mask = resampleThresholdMask({
              logits: logits.subarray(chosen[k] * lowPixels, (chosen[k] + 1) * lowPixels),
              lowWidth,
              lowHeight,
              padWidth,
              padHeight,
              reshapedWidth,
              reshapedHeight,
              originalWidth,
              originalHeight,
              threshold: options.maskThreshold,
            });
            if (mask.area >= options.minMaskArea) candidates.push(mask);
          }
          recordSub('resample');
        }
        timings.record('filter', performance.now() - started);

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

    // ---- mask-encode: right here, while the coverage arrays are still local.
    // ---- `phase` is set first so a `CompressionStream` failure surfaces as
    // ---- SegmenterFailure('mask-encode', ...) through the catch below.
    phase = 'mask-encode';
    const masks: EncodedMask[] = [];
    for (const index of kept) {
      const encodeStarted = performance.now();
      const maskUrl = await encodeMaskPng(
        candidates[index].coverage,
        originalWidth,
        originalHeight,
      );
      timings.record('mask-encode', performance.now() - encodeStarted);
      masks.push({ maskUrl, area: candidates[index].area });
    }

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
  } catch (error) {
    post({
      type: 'error',
      phase,
      message: error instanceof Error ? error.message : String(error),
    });
  } finally {
    bitmap.close();
  }
}

globalThis.addEventListener('message', (event) => {
  const data = (event as MessageEvent<SegmenterRequest>).data;
  if (data?.type === 'segment') void run(data);
});
