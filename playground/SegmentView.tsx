import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SegmentViewer, type ViewerSegment } from '../src';
import {
  DEFAULT_SEGMENTER_OPTIONS,
  FILTER_SUBSTEP_ORDER,
  PHASE_ORDER,
  SegmenterFailure,
  createSegmenter,
  isWebGPUAvailable,
  type SegmentationResult,
  type SegmenterOptions,
  type SegmenterProgress,
} from '../src/segmenter';
import sampleUrl from './sample/cafe-table.jpg';
import {
  BATCH_SIZE_CHOICES,
  DTYPE_CHOICES,
  POINTS_PER_SIDE_CHOICES,
} from './option-choices';

interface LoadedImage {
  url: string;
  width: number;
  height: number;
  /** Set when we minted the url ourselves and therefore have to revoke it. */
  objectUrl: boolean;
}

async function loadImage(url: string, objectUrl: boolean): Promise<LoadedImage> {
  const img = new Image();
  img.src = url;
  await img.decode();
  return { url, width: img.naturalWidth, height: img.naturalHeight, objectUrl };
}

function ms(value: number): string {
  return value.toFixed(1);
}

export function SegmentView() {
  // Probed once: a GPU adapter does not appear part-way through a session, and
  // re-probing per render would make the no-WebGPU panel flicker.
  const webgpu = useMemo(() => isWebGPUAvailable(), []);
  const [image, setImage] = useState<LoadedImage | null>(null);
  const [options, setOptions] = useState<SegmenterOptions>(DEFAULT_SEGMENTER_OPTIONS);
  const [segments, setSegments] = useState<ViewerSegment[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<SegmenterProgress | null>(null);
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [error, setError] = useState<{ phase: string; message: string } | null>(null);
  const segmenterRef = useRef<ReturnType<typeof createSegmenter> | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const replaceImage = useCallback((next: LoadedImage) => {
    // Revoked here rather than inside a state updater: React re-invokes
    // updaters in StrictMode, and a side effect in one fires twice.
    if (objectUrlRef.current && objectUrlRef.current !== next.url) {
      URL.revokeObjectURL(objectUrlRef.current);
    }
    objectUrlRef.current = next.objectUrl ? next.url : null;
    setImage(next);
    setSegments([]);
    setSelected(new Set());
    setResult(null);
    setError(null);
    setProgress(null);
  }, []);

  // Preload the committed sample so the demo is one click (AC2).
  useEffect(() => {
    let cancelled = false;
    void loadImage(sampleUrl, false).then((loaded) => {
      if (!cancelled) setImage(loaded);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(
    () => () => {
      segmenterRef.current?.dispose();
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    },
    [],
  );

  const patch = (change: Partial<SegmenterOptions>) =>
    setOptions((current) => ({ ...current, ...change }));

  async function run() {
    if (!image) return;
    setRunning(true);
    setError(null);
    setResult(null);
    setSegments([]);
    setSelected(new Set());
    setProgress(null);

    try {
      const blob = await (await fetch(image.url)).blob();
      const bitmap = await createImageBitmap(blob);
      // Created lazily so a browser with no WebGPU never spawns a worker.
      segmenterRef.current ??= createSegmenter();
      const outcome = await segmenterRef.current.segment(bitmap, options, setProgress);
      setSegments(outcome.segments);
      setResult(outcome);
    } catch (thrown) {
      const failure = thrown instanceof SegmenterFailure ? thrown : null;
      setError({
        phase: failure?.phase ?? 'unknown',
        message: thrown instanceof Error ? thrown.message : String(thrown),
      });
    } finally {
      setRunning(false);
    }
  }

  async function pickFile(file: File | undefined) {
    if (!file) return;
    replaceImage(await loadImage(URL.createObjectURL(file), true));
  }

  return (
    <>
      <p>
        <label>
          points per side:{' '}
          <select
            data-testid="points-per-side"
            value={options.pointsPerSide}
            onChange={(e) => patch({ pointsPerSide: Number(e.target.value) })}
          >
            {POINTS_PER_SIDE_CHOICES.map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>{' '}
        <label>
          dtype:{' '}
          <select
            data-testid="segment-dtype"
            value={options.dtype}
            disabled={running}
            onChange={(e) => patch({ dtype: e.target.value as SegmenterOptions['dtype'] })}
          >
            {DTYPE_CHOICES.map((value) => (
              <option key={value} value={value}>{value}</option>
            ))}
          </select>
        </label>{' '}
        <label>
          batch size:{' '}
          <select
            data-testid="segment-batch-size"
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
          mask threshold:{' '}
          <input
            data-testid="mask-threshold"
            type="number"
            step={0.1}
            value={options.maskThreshold}
            onChange={(e) => patch({ maskThreshold: Number(e.target.value) })}
          />
        </label>{' '}
        <label>
          stability threshold:{' '}
          <input
            data-testid="stability-threshold"
            type="number"
            step={0.01}
            min={0}
            max={1}
            value={options.stabilityScoreThreshold}
            onChange={(e) => patch({ stabilityScoreThreshold: Number(e.target.value) })}
          />
        </label>{' '}
        <label>
          min mask area:{' '}
          <input
            data-testid="min-mask-area"
            type="number"
            step={10}
            min={0}
            value={options.minMaskArea}
            onChange={(e) => patch({ minMaskArea: Number(e.target.value) })}
          />
        </label>{' '}
        <label>
          NMS IoU:{' '}
          <input
            data-testid="nms-iou"
            type="number"
            step={0.05}
            min={0}
            max={1}
            value={options.nmsIouThreshold}
            onChange={(e) => patch({ nmsIouThreshold: Number(e.target.value) })}
          />
        </label>
        {' '}
        <label>
          compare NMS (~100x slower nms stage):{' '}
          <input
            data-testid="compare-nms"
            type="checkbox"
            checked={options.compareNms}
            onChange={(e) => patch({ compareNms: e.target.checked })}
          />
        </label>
        {' '}
        <label>
          overlap decode/filter:{' '}
          <input
            data-testid="overlap-decode-filter"
            type="checkbox"
            checked={options.overlapDecodeFilter}
            onChange={(e) => patch({ overlapDecodeFilter: e.target.checked })}
          />
        </label>
        {' '}
        <label>
          GPU-resident embeddings:{' '}
          <input
            data-testid="gpu-resident-embeddings"
            type="checkbox"
            checked={options.gpuResidentEmbeddings}
            onChange={(e) => patch({ gpuResidentEmbeddings: e.target.checked })}
          />
        </label>
      </p>

      <p
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          void pickFile(e.dataTransfer.files[0]);
        }}
        style={{ border: '1px dashed #9ca3af', borderRadius: 6, padding: 12 }}
      >
        <label>
          use your own image (or drop one here):{' '}
          <input
            data-testid="sample-file-input"
            type="file"
            accept="image/*"
            onChange={(e) => void pickFile(e.target.files?.[0])}
          />
        </label>
      </p>

      {webgpu ? (
        <p>
          <button
            data-testid="run-segmentation"
            type="button"
            disabled={running || !image}
            onClick={() => void run()}
          >
            {running ? 'Running…' : 'Run'}
          </button>
        </p>
      ) : (
        <section
          data-testid="webgpu-required"
          style={{ border: '1px solid #f59e0b', borderRadius: 6, padding: 12 }}
        >
          <h2 style={{ margin: '0 0 4px', fontSize: '1rem' }}>WebGPU required</h2>
          <p style={{ margin: 0 }}>
            This prototype runs the whole segmentation model in your browser on{' '}
            <strong>WebGPU</strong>. This browser exposes no <code>navigator.gpu</code>{' '}
            adapter, so nothing has been downloaded and no worker has been started.
            Chrome or Edge 113+, Firefox 141+, or Safari 26+ on a supported GPU can run
            it. There is no CPU fallback by design — on CPU a single pass takes minutes.
          </p>
        </section>
      )}

      {running && (
        <p data-testid="run-progress" role="status">
          {progress
            ? `${progress.phase} ${progress.done}/${progress.total} — ${ms(progress.ms)} ms`
            : 'starting…'}
        </p>
      )}

      {error && (
        <p data-testid="run-error" role="alert">
          Failed during <strong>{error.phase}</strong>: {error.message}
        </p>
      )}

      {result && (
        <>
          <table data-testid="results-table" style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th align="left">phase</th>
                <th align="right">p50 ms</th>
                <th align="right">p95 ms</th>
                <th align="right">max ms</th>
                <th align="right">total ms</th>
                <th align="right">calls</th>
              </tr>
            </thead>
            <tbody>
              {PHASE_ORDER.map((phase) => {
                const timing = result.timings.phases[phase];
                return (
                  <Fragment key={phase}>
                    <tr>
                      <th align="left" scope="row">{phase}</th>
                      <td align="right">{ms(timing.p50)}</td>
                      <td align="right">{ms(timing.p95)}</td>
                      <td align="right">{ms(timing.max)}</td>
                      <td align="right">{ms(timing.total)}</td>
                      <td align="right">{timing.count}</td>
                    </tr>
                    {/* The filter breakdown nests under filter and is NOT a
                        peer of it — indented and tree-prefixed so nobody sums
                        it into the phase column. */}
                    {phase === 'filter' &&
                      FILTER_SUBSTEP_ORDER.map((step) => {
                        const sub = result.timings.filterSubPhases[step];
                        return (
                          <tr key={`filter-${step}`} data-testid={`filter-substep-${step}`}>
                            <th
                              align="left"
                              scope="row"
                              style={{ paddingLeft: '1.5em', fontWeight: 'normal', opacity: 0.8 }}
                            >
                              └ {step}
                            </th>
                            <td align="right">{ms(sub.p50)}</td>
                            <td align="right">{ms(sub.p95)}</td>
                            <td align="right">{ms(sub.max)}</td>
                            <td align="right">{ms(sub.total)}</td>
                            <td align="right">{sub.count}</td>
                          </tr>
                        );
                      })}
                  </Fragment>
                );
              })}
              <tr>
                <th align="left" scope="row">total</th>
                <td align="right" colSpan={3} />
                <td align="right">{ms(result.timings.totalMs)}</td>
                <td />
              </tr>
            </tbody>
          </table>
          <p data-testid="mask-counts">
            masks: {result.counts.raw} raw → {result.counts.afterFilter} after filter →{' '}
            <strong>{result.counts.afterNms}</strong> after dedup. The last number is the
            one that matters; raw counts are misleading.
          </p>
          {result.nmsComparison && (
            <p data-testid="nms-comparison">
              nms A/B: reference {ms(result.nmsComparison.referenceMs)} ms → fast{' '}
              {ms(result.nmsComparison.fastMs)} ms
              {result.nmsComparison.fastMs > 0 &&
                ` (${(result.nmsComparison.referenceMs / result.nmsComparison.fastMs).toFixed(1)}x)`}
              . kept sets{' '}
              <strong>{result.nmsComparison.identical ? 'identical' : 'DIFFERENT'}</strong>. The
              total row above includes the reference run; the nms phase row does not, so the
              two will not reconcile in A/B mode.
            </p>
          )}
        </>
      )}

      {image && (
        <SegmentViewer
          key={`${image.url}-${segments.length}`}
          imageUrl={image.url}
          imageWidth={image.width}
          imageHeight={image.height}
          segments={segments}
          initialSelectedIds={selected}
          onSelectionChange={setSelected}
          onCreateSegment={async () => {}}
          maxHeight="70vh"
        />
      )}
    </>
  );
}
