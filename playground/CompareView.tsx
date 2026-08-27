import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  SegmenterFailure,
  createSegmenter,
  isWebGPUAvailable,
  type SegmenterProgress,
} from '../src/segmenter';
import {
  CHOSEN_BATCH_SIZE_CHOICES,
  COMPARE_ROWS,
  DEFAULT_CHOSEN_BATCH_SIZE,
  pairAgreements,
  rowOptions,
  runRow,
  toMarkdown,
  type CompareResult,
  type CompareRow,
} from './compare';
import sampleUrl from './sample/cafe-table.jpg';

function ms(value: number): string {
  return value.toFixed(0);
}

export function CompareView() {
  // Probed once: a GPU adapter does not appear part-way through a session.
  const webgpu = useMemo(() => isWebGPUAvailable(), []);
  const [imageUrl, setImageUrl] = useState<string>(sampleUrl);
  const [chosenBatchSize, setChosenBatchSize] = useState<number>(DEFAULT_CHOSEN_BATCH_SIZE);
  const [results, setResults] = useState<Record<string, CompareResult>>({});
  const [errors, setErrors] = useState<Record<string, { phase: string; message: string }>>({});
  const [runningRowId, setRunningRowId] = useState<string | null>(null);
  const [sweeping, setSweeping] = useState(false);
  const [progress, setProgress] = useState<SegmenterProgress | null>(null);
  const [copied, setCopied] = useState(false);
  const segmenterRef = useRef<ReturnType<typeof createSegmenter> | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  useEffect(
    () => () => {
      segmenterRef.current?.dispose();
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    },
    [],
  );

  const runOne = useCallback(
    async (row: CompareRow) => {
      setRunningRowId(row.id);
      setProgress(null);
      setErrors((current) => {
        const next = { ...current };
        delete next[row.id];
        return next;
      });
      try {
        // Created lazily so a browser with no WebGPU never spawns a worker.
        segmenterRef.current ??= createSegmenter();
        const segmenter = segmenterRef.current;
        const result = await runRow(row, {
          chosenBatchSize,
          // Fresh per row: the bitmap is transferred to the worker and consumed.
          createBitmap: async () => createImageBitmap(await (await fetch(imageUrl)).blob()),
          segment: (bitmap, options, onProgress) =>
            segmenter.segment(bitmap, options, onProgress),
          onProgress: setProgress,
        });
        // Replacing the entry drops the previous run's rawMasks reference, so a
        // re-run releases the old coverage buffers rather than stacking them.
        setResults((current) => ({ ...current, [row.id]: result }));
      } catch (thrown) {
        const failure = thrown instanceof SegmenterFailure ? thrown : null;
        // Recorded against this row only — one bad row must not abort the sweep.
        setErrors((current) => ({
          ...current,
          [row.id]: {
            phase: failure?.phase ?? 'unknown',
            message: thrown instanceof Error ? thrown.message : String(thrown),
          },
        }));
      } finally {
        setRunningRowId(null);
        setProgress(null);
      }
    },
    [chosenBatchSize, imageUrl],
  );

  const runAll = useCallback(async () => {
    setSweeping(true);
    // Sequential: two runs at once would contend for the same GPU and make
    // every timing in the table meaningless.
    for (const row of COMPARE_ROWS) await runOne(row);
    setSweeping(false);
  }, [runOne]);

  function pickFile(file: File | undefined) {
    if (!file) return;
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setImageUrl(url);
    setResults({});
    setErrors({});
  }

  const ordered = COMPARE_ROWS.map((row) => results[row.id]).filter(
    (result): result is CompareResult => Boolean(result),
  );
  // Memoised on `results`: comparing two mask sets is a full-resolution
  // O(masks squared) scan, and a progress event fires per decode batch — re-running
  // it on every render would freeze the sweep once a pair has both halves.
  const agreements = useMemo(() => pairAgreements(results), [results]);
  const busy = runningRowId !== null || sweeping;

  async function copyMarkdown() {
    await navigator.clipboard.writeText(toMarkdown(ordered, agreements));
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

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
          Chrome or Edge 113+, Firefox 141+, or Safari 26+ on a supported GPU can run
          it. There is no CPU fallback by design — on CPU a single pass takes minutes.
        </p>
      </section>
    );
  }

  return (
    <>
      <p>
        Seven rows measuring <strong>fp16 vs fp32</strong> and the{' '}
        <strong>batchSize</strong> curve. Every row respawns the worker and pays its
        own <code>model-load</code>, which is reported separately and excluded from the
        budget. Expect roughly 9 minutes for the four dtype rows.
      </p>

      <p>
        <label>
          confirmation-row batch size:{' '}
          <select
            data-testid="chosen-batch-size"
            value={chosenBatchSize}
            disabled={busy}
            onChange={(e) => setChosenBatchSize(Number(e.target.value))}
          >
            {CHOSEN_BATCH_SIZE_CHOICES.map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>{' '}
        <label>
          image:{' '}
          <input
            data-testid="compare-file-input"
            type="file"
            accept="image/*"
            disabled={busy}
            onChange={(e) => pickFile(e.target.files?.[0])}
          />
        </label>{' '}
        <button data-testid="run-all" type="button" disabled={busy} onClick={() => void runAll()}>
          {sweeping ? 'Running all…' : 'Run all'}
        </button>{' '}
        <button
          data-testid="copy-markdown"
          type="button"
          disabled={ordered.length === 0}
          onClick={() => void copyMarkdown()}
        >
          {copied ? 'Copied' : 'Copy as markdown'}
        </button>
      </p>

      {busy && (
        <p data-testid="compare-progress" role="status">
          {runningRowId ?? '…'}:{' '}
          {progress
            ? `${progress.phase} ${progress.done}/${progress.total} — ${progress.ms.toFixed(1)} ms`
            : 'starting…'}
        </p>
      )}

      <table data-testid="compare-table" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th align="left">row</th>
            <th align="left">dtype</th>
            <th align="right">pps</th>
            <th align="right">batch</th>
            <th align="right">encode</th>
            <th align="right">decode</th>
            <th align="right">encode + decode</th>
            <th align="right">total</th>
            <th align="right">kept</th>
            <th align="left">serves</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {COMPARE_ROWS.map((row) => {
            const options = rowOptions(row, chosenBatchSize);
            const result = results[row.id];
            const error = errors[row.id];
            return (
              <tr key={row.id} data-testid={`compare-${row.id}`}>
                <th align="left" scope="row">{row.id}</th>
                <td>{options.dtype}</td>
                <td align="right">{options.pointsPerSide}</td>
                <td align="right">{options.batchSize}</td>
                {result ? (
                  <>
                    <td align="right">{ms(result.phases.encode)}</td>
                    <td align="right">{ms(result.phases.decode)}</td>
                    <td align="right"><strong>{ms(result.encodeDecodeMs)}</strong></td>
                    <td align="right">{ms(result.totalMs)}</td>
                    <td align="right">{result.counts.afterNms}</td>
                  </>
                ) : (
                  <td colSpan={5} align="left">
                    {error ? (
                      <span role="alert">
                        failed in <strong>{error.phase}</strong>: {error.message}
                      </span>
                    ) : (
                      'not run'
                    )}
                  </td>
                )}
                <td>{row.serves}</td>
                <td>
                  <button type="button" disabled={busy} onClick={() => void runOne(row)}>
                    {runningRowId === row.id ? 'Running…' : 'Run'}
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {agreements.length > 0 && (
        <section data-testid="agreement-panel">
          <h2 style={{ fontSize: '1rem' }}>fp16 vs fp32 mask agreement</h2>
          {agreements.map((entry) => {
            const a = entry.agreement;
            return (
              <p key={entry.pairId} data-testid={`agreement-${entry.pairId}`}>
                <strong>{entry.label}</strong> ({entry.baselineRowId} vs {entry.variantRowId}):{' '}
                {a.baselineCount} fp32 masks vs {a.variantCount} fp16, {a.matched} matched at
                IoU ≥ {a.floor} ({a.unmatchedBaseline} / {a.unmatchedVariant} unmatched). IoU
                mean {a.meanIou.toFixed(3)}, median {a.medianIou.toFixed(3)}, min{' '}
                {a.minIou.toFixed(3)}.
              </p>
            );
          })}
        </section>
      )}
    </>
  );
}
