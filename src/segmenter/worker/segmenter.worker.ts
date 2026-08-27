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
 * batches -> filter each batch at LOW resolution -> upscale only the survivors
 * -> NMS across everything. Filtering before upscaling is not an optimization
 * detail: a batch of 8 points yields 24 low-res masks, and upscaling all of
 * them to full image resolution as float32 is hundreds of megabytes per batch.
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
  stabilityScore,
  thresholdMask,
  type BinaryMask,
  type EncodedMask,
  type FilterSubstep,
  type SegmentationPhase,
  type SegmenterOptions,
  type SegmenterRequest,
  type SegmenterResponse,
} from '../core';

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

/**
 * Cached for the lifetime of the worker. The client terminates and respawns
 * the worker between runs, so a poisoned session (lost adapter, half-loaded
 * weights) is never reused; within one run this just avoids reloading.
 */
let session: Session | null = null;
let sessionKey = '';

async function loadSession(options: SegmenterOptions): Promise<Session> {
  const key = `${options.modelId}|${options.dtype}`;
  if (session && sessionKey === key) return session;

  const model = (await SamModel.from_pretrained(options.modelId, {
    device: 'webgpu',
    dtype: options.dtype,
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
    const fullPixels = originalHeight * originalWidth;

    const batches = batchPoints(buildPointGrid(options.pointsPerSide), options.batchSize);
    const candidates: BinaryMask[] = [];
    let rawCount = 0;

    for (let b = 0; b < batches.length; b += 1) {
      const batch = batches[b];
      const batchStarted = performance.now();

      // ---- decode ----
      phase = 'decode';
      started = performance.now();
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
      const outputs = await model({
        ...embeddings,
        input_points: inputPoints,
        input_labels: inputLabels,
      });
      // `.to('float32')` is a no-op on an fp32 model and the required
      // conversion on an fp16 one, where ORT hands back raw float16 bits in a
      // Uint16Array on runtimes with no Float16Array.
      const predMasks = outputs.pred_masks.to('float32') as Tensor;
      const iouScores = outputs.iou_scores.to('float32') as Tensor;
      timings.record('decode', performance.now() - started);

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
        const selected = new Float32Array(chosen.length * lowPixels);
        for (let k = 0; k < chosen.length; k += 1) {
          selected.set(
            logits.subarray(chosen[k] * lowPixels, (chosen[k] + 1) * lowPixels),
            k * lowPixels,
          );
        }
        // post_process_masks indexes `masks[0]`, so a 5-D tensor shaped
        // [1, K, 1, lowH, lowW] gives it the 4-D [K, 1, lowH, lowW] batch it
        // wants and comes back as one [K, 1, origH, origW] tensor.
        const upscaled = await processor.post_process_masks(
          new Tensor('float32', selected, [1, chosen.length, 1, lowHeight, lowWidth]),
          originalSizes,
          reshapedSizes,
          { binarize: false },
        );
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

      post({
        type: 'progress',
        event: {
          phase: 'decode',
          done: b + 1,
          total: batches.length,
          ms: performance.now() - batchStarted,
        },
      });
    }

    // ---- nms: once, across every batch. Adjacent grid points land on the
    // ---- same object constantly, so this is where the count actually falls.
    phase = 'nms';
    started = performance.now();
    const kept = dedupeMasks(candidates, options.nmsIouThreshold);
    elapsed = performance.now() - started;
    timings.record('nms', elapsed);
    post({ type: 'progress', event: { phase: 'nms', done: 1, total: 1, ms: elapsed } });

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

    // No transfer list, deliberately: `masks` is now strings. Not one coverage
    // buffer crosses the worker boundary any more.
    post({
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
    });
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
