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
 */
export const FILTER_SUBSTEP_ORDER = ['select', 'upscale', 'threshold'] as const;

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
}

/**
 * `Xenova/slimsam-77-uniform` is the tiny distilled SAM that transformers.js
 * actually ships ONNX weights + a processor config for; no MobileSAM export on
 * the Hub is loadable by `SamModel`/`AutoProcessor` (see the plan's deviation
 * note). `dtype: 'fp32'` is the safe default — fp16 outputs come back as raw
 * float16 bits on runtimes without `Float16Array`, and while the worker
 * converts them with `tensor.to('float32')`, fp32 removes the variable when
 * the first goal is a trustworthy timing table.
 */
export const DEFAULT_SEGMENTER_OPTIONS: SegmenterOptions = {
  pointsPerSide: 16,
  batchSize: 8,
  maskThreshold: 0,
  stabilityScoreThreshold: 0.85,
  stabilityScoreOffset: 1,
  minMaskArea: 100,
  nmsIouThreshold: 0.7,
  modelId: 'Xenova/slimsam-77-uniform',
  dtype: 'fp32',
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
    }
  | { type: 'error'; phase: SegmentationPhase; message: string };
