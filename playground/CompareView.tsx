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

import {
  BATCH_SIZE_CHOICES,
  DTYPE_CHOICES,
  POINTS_PER_SIDE_CHOICES,
} from './option-choices';

const INITIAL_OPTIONS: RowOptions = {
  dtype: 'fp32',
  batchSize: 8,
  pointsPerSide: 16,
  overlapDecodeFilter: false,
  gpuResidentEmbeddings: false,
  keepRawMasks: false,
  lowResFilterNms: true,
  lowResMaskEncode: true,
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
        </label>{' '}
        <label>
          low-res filter/NMS:{' '}
          <input
            data-testid="low-res-filter-nms"
            type="checkbox"
            checked={options.lowResFilterNms}
            disabled={running}
            onChange={(e) => patch({ lowResFilterNms: e.target.checked })}
          />
        </label>{' '}
        <label>
          low-res mask encode:{' '}
          <input
            data-testid="low-res-mask-encode"
            type="checkbox"
            checked={options.lowResMaskEncode}
            disabled={running}
            onChange={(e) => patch({ lowResMaskEncode: e.target.checked })}
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
            <th align="left">lowres</th>
            <th align="left">enc</th>
            <th align="right">budget</th>
            <th align="right">total</th>
            <th align="right">decode</th>
            <th align="right">filter</th>
            <th align="right">kept</th>
            <th align="right">returned</th>
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
                <td>{o.lowResFilterNms ? 'lowres' : 'fullres'}</td>
                <td>{o.lowResMaskEncode ? 'enclow' : 'encfull'}</td>
                <td align="right"><strong>{ms(record.budgetMs)}</strong></td>
                <td align="right">{ms(record.timings?.totalMs)}</td>
                <td align="right">{ms(p?.decode.total)}</td>
                <td align="right">{ms(p?.filter.total)}</td>
                <td align="right">{record.counts?.afterNms ?? '—'}</td>
                <td align="right">{record.counts?.returned ?? '—'}</td>
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
