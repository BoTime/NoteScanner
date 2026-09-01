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
