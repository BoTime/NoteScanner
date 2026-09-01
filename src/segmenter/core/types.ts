import type { ViewerSegment } from '../../types';

/**
 * The phases the segmenter times, in the order the playground's results table
 * renders them. Every one of them happens inside the worker — including
 * `mask-encode`, which used to run on the main thread.
 */
export const PHASE_ORDER = [
  'model-load',
  'encode',
  'decode',
  'filter',
  'nms',
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
 * `resample` is one region, not two: the upsample and the threshold are a
 * single fused loop in `resampleThresholdMask`, and reporting them separately
 * would ship a permanently-zero row.
 */
export const FILTER_SUBSTEP_ORDER = ['select', 'resample'] as const;

export type FilterSubstep = (typeof FILTER_SUBSTEP_ORDER)[number];

export interface SegmenterOptions {
  /** Prompt-grid density. One crop layer only — no multi-crop, by design. */
  pointsPerSide: number;
  /** Grid points sent to the mask decoder per model call. */
  batchSize: number;
  /** Logit value above which a predicted mask pixel counts as covered. */
  maskThreshold: number;
  /** Minimum stability score a mask must reach to survive. */
  stabilityScoreThreshold: number;
  /** Logit offset applied either side of `maskThreshold` when scoring stability. */
  stabilityScoreOffset: number;
  /** Minimum covered pixel count a mask must have to survive. */
  minMaskArea: number;
  /** IoU above which a candidate is a duplicate of an already-kept mask. */
  nmsIouThreshold: number;
  /** Hugging Face repo id of the SAM checkpoint. */
  modelId: string;
  /** ONNX weight precision handed to transformers.js. */
  dtype: 'fp32' | 'fp16' | 'q8';
  /**
   * Run the pre-optimization byte-wise NMS alongside the fast one and report
   * both times plus whether they kept the identical set. Off by default, and
   * deliberately so: the reference is on the order of 100x slower than the
   * fast path (the exact ratio depends on candidate count and mask size), so
   * ticking this multiplies the `nms` stage by roughly that factor — a minute
   * or more at `pointsPerSide` 32, with no progress event emitted until the
   * reference run finishes. That long silence is the option working, not a
   * hung worker. It exists so the speedup can be measured on a real image
   * rather than argued about.
   */
  compareNms: boolean;
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
}

/**
 * `Xenova/slimsam-77-uniform` is the tiny distilled SAM that transformers.js
 * actually ships ONNX weights + a processor config for; no MobileSAM export on
 * the Hub is loadable by `SamModel`/`AutoProcessor` (see the plan's deviation
 * note). `dtype: 'fp32'` is the safe default — fp16 outputs come back as raw
 * float16 bits on runtimes without `Float16Array`, and while the worker
 * converts them with `tensor.to('float32')`, fp32 removes the variable when
 * the first goal is a trustworthy timing table. It is also not purely a speed
 * knob: fp16 changes the kept-mask COUNT, and in both directions — 26 → 28 in
 * issue #12's run, 36 → 31 in the decode sweep's. That makes adopting it a
 * product decision, which is why the faster dtype is still not the default.
 *
 * `batchSize: 32` IS the default, on two independent measurements: issue #12
 * (−11.8% on `decode` at 16 points per side) and
 * `docs/measurements/2026-08-28-decode-sweep.md` (−4.4% on the whole budget,
 * fp32, same density). Both also rule out going further: `batchSize: 64` dies
 * with `Array buffer allocation failed`, because the upsample allocates
 * `batch × width × height × 4` bytes at once. 32 is bounded above by memory,
 * not by diminishing returns.
 */
export const DEFAULT_SEGMENTER_OPTIONS: SegmenterOptions = {
  pointsPerSide: 16,
  batchSize: 32,
  maskThreshold: 0,
  stabilityScoreThreshold: 0.85,
  stabilityScoreOffset: 1,
  minMaskArea: 100,
  nmsIouThreshold: 0.7,
  modelId: 'Xenova/slimsam-77-uniform',
  dtype: 'fp32',
  compareNms: false,
  overlapDecodeFilter: false,
  gpuResidentEmbeddings: false,
  keepRawMasks: false,
};

export interface PhaseTiming {
  p50: number;
  p95: number;
  max: number;
  /** Number of samples recorded for the phase. */
  count: number;
  /** Sum of every sample — the phase's share of wall clock. */
  total: number;
}

export interface TimingReport {
  phases: Record<SegmentationPhase, PhaseTiming>;
  /**
   * The `filter` stage broken down. Required, not optional: the worker's whole
   * report is passed through, and requiring the field makes the compiler catch
   * a dropped passthrough across the worker boundary.
   */
  filterSubPhases: Record<FilterSubstep, PhaseTiming>;
  /** Wall clock for the whole run, measured on the main thread. */
  totalMs: number;
}

export interface SegmentationCounts {
  /** Every mask the decoder produced, before any filtering. */
  raw: number;
  /** Survivors of the stability, best-of-three and minimum-area filters. */
  afterFilter: number;
  /** Survivors of NMS dedup — the only count that means anything. */
  afterNms: number;
}

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

export interface SegmenterProgress {
  phase: SegmentationPhase;
  done: number;
  total: number;
  ms: number;
}

export interface SegmentationResult {
  segments: ViewerSegment[];
  timings: TimingReport;
  counts: SegmentationCounts;
  /** Present only when `compareNms` was set. */
  nmsComparison?: NmsComparison;
  /** Present only when the run asked for `keepRawMasks`. */
  rawMasks?: RawMask[];
}

/** A failure that names the phase it died in, so the UI can say where. */
export class SegmenterFailure extends Error {
  readonly phase: SegmentationPhase;

  constructor(phase: SegmentationPhase, message: string) {
    super(message);
    this.name = 'SegmenterFailure';
    this.phase = phase;
  }
}

/**
 * A surviving mask as the worker posts it back: already a PNG data URL, not a
 * coverage buffer. Encoding in the worker is what removes both the ~0.7 MB
 * per-mask transfer and the main-thread encode loop.
 */
export interface EncodedMask {
  maskUrl: string;
  area: number;
}

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

export type SegmenterRequest = {
  type: 'segment';
  /** Transferred, never cloned. */
  bitmap: ImageBitmap;
  options: SegmenterOptions;
};

export type SegmenterResponse =
  | { type: 'progress'; event: SegmenterProgress }
  | {
      type: 'done';
      masks: EncodedMask[];
      width: number;
      height: number;
      timings: TimingReport;
      counts: SegmentationCounts;
      nmsComparison?: NmsComparison;
      rawMasks?: RawMask[];
    }
  | { type: 'error'; phase: SegmentationPhase; message: string };
