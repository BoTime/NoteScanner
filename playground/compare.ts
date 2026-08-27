import {
  PHASE_ORDER,
  pairwiseIoU,
  type RawMask,
  type SegmentationCounts,
  type SegmentationPhase,
  type SegmentationResult,
  type SegmenterOptions,
  type SegmenterProgress,
} from '../src/segmenter';

/**
 * The measurement matrix for issue #3 (E1 + D1: fp16 and a larger batchSize).
 *
 * Playground-local on purpose: nothing here belongs in the published
 * `/segmenter` subpath, exactly as `benchmark.ts` sits beside its view.
 *
 * Seven rows, not an exhaustive sweep. About 90% of every run is `filter`,
 * `nms` and `mask-encode` — stages this issue does not touch — so a wide
 * sweep would buy minutes of wall clock and no extra signal. Four dtype rows
 * answer AC1/AC2; three further rows sketch the batchSize curve for AC3.
 */

/** The subset of `SegmenterOptions` a compare row pins. */
export interface RowOptions {
  dtype: SegmenterOptions['dtype'];
  pointsPerSide: number;
  batchSize: number;
  keepRawMasks: boolean;
}

export interface CompareRow {
  id: string;
  dtype: SegmenterOptions['dtype'];
  pointsPerSide: number;
  /** `null` on the confirmation row: the developer supplies it from rows 5-6. */
  batchSize: number | null;
  /**
   * Only the four dtype rows retain masks. batchSize cannot change the output
   * — it changes how many prompt points ride in each model call — so keeping
   * ~50 full-resolution coverage arrays for rows 5-7 would pin tens of
   * megabytes for nothing.
   */
  keepRawMasks: boolean;
  /** Which acceptance criterion this row exists for, shown in the table. */
  serves: string;
}

export const COMPARE_ROWS: readonly CompareRow[] = [
  { id: 'row-1', dtype: 'fp32', pointsPerSide: 16, batchSize: 8, keepRawMasks: true, serves: 'AC1/AC2 baseline' },
  { id: 'row-2', dtype: 'fp16', pointsPerSide: 16, batchSize: 8, keepRawMasks: true, serves: 'AC1/AC2' },
  { id: 'row-3', dtype: 'fp32', pointsPerSide: 32, batchSize: 8, keepRawMasks: true, serves: 'AC1/AC2 baseline' },
  { id: 'row-4', dtype: 'fp16', pointsPerSide: 32, batchSize: 8, keepRawMasks: true, serves: 'AC1/AC2' },
  { id: 'row-5', dtype: 'fp16', pointsPerSide: 16, batchSize: 32, keepRawMasks: false, serves: 'AC3 curve' },
  { id: 'row-6', dtype: 'fp16', pointsPerSide: 16, batchSize: 64, keepRawMasks: false, serves: 'AC3 curve' },
  { id: 'row-7', dtype: 'fp16', pointsPerSide: 32, batchSize: null, keepRawMasks: false, serves: 'AC3 confirmation' },
];

export interface ComparePair {
  id: string;
  baselineRowId: string;
  variantRowId: string;
  label: string;
}

/** The fp32 baseline / fp16 variant pairs AC2 compares, at each density. */
export const COMPARE_PAIRS: readonly ComparePair[] = [
  { id: 'pair-16', baselineRowId: 'row-1', variantRowId: 'row-2', label: '16 points per side' },
  { id: 'pair-32', baselineRowId: 'row-3', variantRowId: 'row-4', label: '32 points per side' },
];

export const CHOSEN_BATCH_SIZE_CHOICES = [32, 64] as const;
export const DEFAULT_CHOSEN_BATCH_SIZE = 32;

/**
 * The IoU at or above which two masks are the same mask. Below it a pair is
 * NOT a bad match — it is unmatched on both sides, because a low-IoU pairing
 * says the two runs found different things, not the same thing badly.
 */
export const IOU_MATCH_FLOOR = 0.9;

export function rowOptions(
  row: CompareRow,
  chosenBatchSize: number = DEFAULT_CHOSEN_BATCH_SIZE,
): RowOptions {
  return {
    dtype: row.dtype,
    pointsPerSide: row.pointsPerSide,
    batchSize: row.batchSize ?? chosenBatchSize,
    keepRawMasks: row.keepRawMasks,
  };
}

export interface CompareResult {
  rowId: string;
  options: RowOptions;
  /** Per-phase wall-clock totals, in ms. */
  phases: Record<SegmentationPhase, number>;
  /** The subtotal E1 and D1 actually move. */
  encodeDecodeMs: number;
  totalMs: number;
  counts: SegmentationCounts;
  rawMasks?: RawMask[];
}

export interface RunRowDeps {
  /**
   * A FRESH bitmap per row. The bitmap is transferred to the worker and
   * consumed, so it cannot be reused across rows.
   */
  createBitmap: () => Promise<ImageBitmap>;
  /** Injected so tests drive this with a stub and never touch WebGPU. */
  segment: (
    image: ImageBitmap,
    options: RowOptions,
    onProgress?: (event: SegmenterProgress) => void,
  ) => Promise<SegmentationResult>;
  chosenBatchSize?: number;
  onProgress?: (event: SegmenterProgress) => void;
}

export async function runRow(row: CompareRow, deps: RunRowDeps): Promise<CompareResult> {
  const options = rowOptions(row, deps.chosenBatchSize);
  const bitmap = await deps.createBitmap();
  const result = await deps.segment(bitmap, options, deps.onProgress);

  const phases = {} as Record<SegmentationPhase, number>;
  for (const phase of PHASE_ORDER) phases[phase] = result.timings.phases[phase].total;

  return {
    rowId: row.id,
    options,
    phases,
    encodeDecodeMs: phases.encode + phases.decode,
    totalMs: result.timings.totalMs,
    counts: result.counts,
    ...(result.rawMasks ? { rawMasks: result.rawMasks } : {}),
  };
}

export interface MaskAgreement {
  baselineCount: number;
  variantCount: number;
  matched: number;
  unmatchedBaseline: number;
  unmatchedVariant: number;
  /** Over the matched pairs only; 0 when nothing matched. */
  meanIou: number;
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
 * IoU comes from `pairwiseIoU` in `src/segmenter/core` — the one definition.
 * `RawMask` is structurally a `BinaryMask`, so it passes straight in.
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

export interface PairAgreement {
  pairId: string;
  label: string;
  baselineRowId: string;
  variantRowId: string;
  agreement: MaskAgreement;
}

/** Agreement for every pair whose BOTH halves have run and kept their masks. */
export function pairAgreements(
  results: Readonly<Record<string, CompareResult>>,
): PairAgreement[] {
  const out: PairAgreement[] = [];
  for (const pair of COMPARE_PAIRS) {
    const baseline = results[pair.baselineRowId]?.rawMasks;
    const variant = results[pair.variantRowId]?.rawMasks;
    if (!baseline || !variant) continue;
    out.push({
      pairId: pair.id,
      label: pair.label,
      baselineRowId: pair.baselineRowId,
      variantRowId: pair.variantRowId,
      agreement: compareMaskSets(baseline, variant),
    });
  }
  return out;
}

function ms(value: number): string {
  return value.toFixed(0);
}

/** Results as a markdown table to paste into issue #3, agreement included. */
export function toMarkdown(
  results: readonly CompareResult[],
  agreements: readonly PairAgreement[],
): string {
  const header =
    '| row | dtype | pps | batch | encode | decode | encode + decode | filter | nms | mask-encode | total | kept | model-load |';
  const divider = '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |';
  const body = results.map((result) => {
    const p = result.phases;
    return [
      result.rowId,
      result.options.dtype,
      String(result.options.pointsPerSide),
      String(result.options.batchSize),
      ms(p.encode),
      ms(p.decode),
      ms(result.encodeDecodeMs),
      ms(p.filter),
      ms(p.nms),
      ms(p['mask-encode']),
      ms(result.totalMs),
      String(result.counts.afterNms),
      ms(p['model-load']),
    ].join(' | ');
  });

  const lines = [header, divider, ...body.map((row) => `| ${row} |`)];
  lines.push('', 'All times in ms. `model-load` is one-time per run and sits outside the budget.');

  if (agreements.length > 0) {
    lines.push('', '### fp16 vs fp32 mask agreement', '');
    lines.push('| pair | density | fp32 masks | fp16 masks | matched | unmatched fp32 | unmatched fp16 | mean IoU | median IoU | min IoU |');
    lines.push('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |');
    for (const entry of agreements) {
      const a = entry.agreement;
      lines.push(
        `| ${entry.pairId} | ${entry.label} | ${a.baselineCount} | ${a.variantCount} | ${a.matched} | ${a.unmatchedBaseline} | ${a.unmatchedVariant} | ${a.meanIou.toFixed(3)} | ${a.medianIou.toFixed(3)} | ${a.minIou.toFixed(3)} |`,
      );
    }
    lines.push('', `Pairing is greedy best-IoU; a pair below IoU ${IOU_MATCH_FLOOR} counts as unmatched on both sides.`);
  }

  return lines.join('\n');
}
