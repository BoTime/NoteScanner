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
  lowResFilterNms: boolean;
}

export interface SweepConfig {
  decodePaths: readonly DecodePath[];
  batchSizes: readonly number[];
  dtypes: readonly SegmenterOptions['dtype'][];
  pointsPerSide: readonly number[];
  /**
   * The pipeline axis. `[true]` by default: the default grid measures the
   * SHIPPED pipeline and stays 16 rows. `[false, true]` walks a before/after
   * pair — baseline first, so the Compare tab retains the PRE-change mask set
   * as its agreement baseline.
   */
  lowResFilterNms: readonly boolean[];
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
  lowResFilterNms: [true],
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
  if (config.lowResFilterNms.length === 0) throw new Error('lowResFilterNms must not be empty');
  for (const value of config.lowResFilterNms) {
    if (typeof value !== 'boolean') {
      throw new Error(`lowResFilterNms must be booleans, got ${JSON.stringify(value)}`);
    }
  }
  if (!Number.isInteger(config.reps) || config.reps < 1) {
    throw new Error(`reps must be a positive integer, got ${JSON.stringify(config.reps)}`);
  }

  return config;
}

// ----------------------------------------------------------------------- grid

export interface SweepRow {
  /** Stable and unique: `p<pps>-<dtype>-b<batch>-<path>-<pipeline>-r<rep>`. */
  id: string;
  rep: number;
  /** True only for the discarded first run — see `warmUpRow`. */
  warmUp: boolean;
  options: RowOptions;
}

/**
 * The measured rows, in a deterministic nesting order — pointsPerSide, then
 * dtype, then batchSize, then decode path, then lowResFilterNms, then rep — so
 * two runs of the same config produce the same ids in the same order and their
 * tables line up.
 */
export function expandGrid(config: SweepConfig): SweepRow[] {
  const rows: SweepRow[] = [];
  for (const pointsPerSide of config.pointsPerSide) {
    for (const dtype of config.dtypes) {
      for (const batchSize of config.batchSizes) {
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
  return `${decodePathOf(o)} · ${o.dtype} · batch ${o.batchSize} · pps ${o.pointsPerSide} · ${
    o.lowResFilterNms ? 'lowres' : 'fullres'
  }`;
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
  lines.push(`| ${COLUMNS.map((column) => (LEFT_ALIGNED.has(column) ? '---' : '---:')).join(' | ')} |`);

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
        o.lowResFilterNms ? 'lowres' : 'fullres',
        ms(record.budgetMs),
        ms(record.timings?.totalMs),
        ms(p?.['model-load'].total),
        ms(p?.encode.total),
        ms(p?.decode.total),
        ms(p?.filter.total),
        ms(p?.nms.total),
        ms(p?.resample.total),
        ms(p?.['mask-encode'].total),
        record.counts ? String(record.counts.raw) : DASH,
        record.counts ? String(record.counts.afterFilter) : DASH,
        record.counts ? String(record.counts.afterNms) : DASH,
        record.counts ? String(record.counts.returned) : DASH,
        ok ? 'ok' : `failed in ${record.phase ?? 'unknown'}: ${record.message ?? ''}`,
      ].join(' | ')} |`,
    );
  }

  lines.push('');
  lines.push(
    `All times in ms. \`budget\` = \`total\` − \`model-load\`; every row respawns the worker ` +
      `and pays its own warm, HTTP-cached model load. Phase columns are stage TOTALS over ` +
      `${PHASE_ORDER.length} phases. \`raw\`/\`afterFilter\`/\`afterNms\`/\`returned\` are mask ` +
      `counts: a row that is fast because it silently dropped masks is visible here, and a gap ` +
      `between \`afterNms\` and \`returned\` is the full-resolution area re-check.`,
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
