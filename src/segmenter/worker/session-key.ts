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
 * posts back — and so is `lowResFilterNms`, which only changes what the worker
 * does with `pred_masks` after the decoder has returned it. So is every
 * per-call inference knob (`pointsPerSide`, `batchSize`, thresholds,
 * `overlapDecodeFilter`), none of which touch the session.
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
