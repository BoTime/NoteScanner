import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SegmentViewer, type ViewerSegment } from '../src';
import {
  DEFAULT_SEGMENTER_OPTIONS,
  SegmenterFailure,
  createSegmenter,
  isWebGPUAvailable,
  type SegmentationResult,
  type SegmenterProgress,
} from '../src/segmenter';
import cafeTableUrl from '../samples/cafe-table.jpg';
import stickyNotesUrl from '../samples/sticky-notes.jpg';
import { progressLine, statLine } from './page-text';

const REPO_URL = 'https://github.com/BoTime/NoteScanner';

const SAMPLES = [
  { id: 'cafe-table', label: 'Café table', url: cafeTableUrl },
  { id: 'sticky-notes', label: 'Sticky notes', url: stickyNotesUrl },
] as const;

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

export function TryItPage() {
  // Probed once: a GPU adapter does not appear part-way through a session, and
  // re-probing per render would make the notice flicker.
  const webgpu = useMemo(() => isWebGPUAvailable(), []);
  const [image, setImage] = useState<LoadedImage | null>(null);
  const [sampleId, setSampleId] = useState<string | null>(SAMPLES[0].id);
  const [segments, setSegments] = useState<ViewerSegment[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<SegmenterProgress | null>(null);
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [error, setError] = useState<{ phase: string; message: string } | null>(null);
  const segmenterRef = useRef<ReturnType<typeof createSegmenter> | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  // Bumped every time the displayed image changes. `run()` captures it and
  // discards its own result if the image moved on meanwhile — otherwise a
  // photo swapped mid-run gets the previous photo's masks painted over it,
  // which reads as a plausible wrong answer rather than as an error.
  //
  // Defence in depth: `pickFile`/`pickSample` already refuse to swap the image
  // while a run is in flight, so today nothing can bump this mid-run. It is
  // kept because that refusal is the kind of thing a later change quietly
  // relaxes, and the failure it prevents is silent rather than loud.
  const imageGenRef = useRef(0);

  const replaceImage = useCallback((next: LoadedImage, nextSampleId: string | null) => {
    // Revoked here rather than inside a state updater: React re-invokes
    // updaters in StrictMode, and a side effect in one fires twice.
    if (objectUrlRef.current && objectUrlRef.current !== next.url) {
      URL.revokeObjectURL(objectUrlRef.current);
    }
    objectUrlRef.current = next.objectUrl ? next.url : null;
    imageGenRef.current += 1;
    setImage(next);
    setSampleId(nextSampleId);
    setSegments([]);
    setSelected(new Set());
    setResult(null);
    setError(null);
    setProgress(null);
  }, []);

  // A sample is on screen before the visitor does anything.
  useEffect(() => {
    let cancelled = false;
    void loadImage(SAMPLES[0].url, false).then((loaded) => {
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

  async function pickSample(sample: (typeof SAMPLES)[number]) {
    if (running) return;
    replaceImage(await loadImage(sample.url, false), sample.id);
  }

  async function pickFile(file: File | undefined) {
    // The chips and the file input are `disabled` while a run is in flight,
    // but a drop onto the panel is not gated by either — so the check lives
    // here, where every path to a new image passes through.
    if (!file || running) return;
    const url = URL.createObjectURL(file);
    try {
      replaceImage(await loadImage(url, true), null);
    } catch {
      // decode() rejects for anything that is not a real image — a dropped
      // .txt, a truncated jpeg. `accept="image/*"` does not constrain drops at
      // all, so on a public page this is a normal thing for a stranger to do.
      // The url never reached replaceImage, so nothing else would revoke it.
      URL.revokeObjectURL(url);
      setError({ phase: 'image', message: 'That file could not be read as an image.' });
    }
  }

  async function run() {
    if (!image) return;
    setRunning(true);
    setError(null);
    setResult(null);
    setSegments([]);
    setSelected(new Set());
    setProgress(null);

    // Captured before any await: everything below belongs to THIS image.
    const gen = imageGenRef.current;
    try {
      const blob = await (await fetch(image.url)).blob();
      const bitmap = await createImageBitmap(blob);
      // Created lazily so a browser with no WebGPU never spawns a worker.
      segmenterRef.current ??= createSegmenter();
      const outcome = await segmenterRef.current.segment(
        bitmap,
        DEFAULT_SEGMENTER_OPTIONS,
        setProgress,
      );
      if (imageGenRef.current !== gen) return;
      setSegments(outcome.segments);
      setResult(outcome);
    } catch (thrown) {
      // A stale run's failure is not this image's failure either — reporting
      // it would blame the new photo for the old one's error.
      if (imageGenRef.current !== gen) return;
      const failure = thrown instanceof SegmenterFailure ? thrown : null;
      setError({
        phase: failure?.phase ?? 'unknown',
        message: thrown instanceof Error ? thrown.message : String(thrown),
      });
    } finally {
      setRunning(false);
    }
  }

  return (
    <main style={{ padding: 16, fontFamily: 'system-ui', maxWidth: 1100, margin: '0 auto' }}>
      <h1 style={{ marginBottom: 4 }}>Try note-scanner</h1>
      <p data-testid="site-tagline" style={{ marginTop: 0, color: '#374151' }}>
        Segments a photo into individual notes and objects with SlimSAM, running on{' '}
        <strong>WebGPU in your own browser</strong>. Your image never leaves this page —
        nothing is uploaded.
      </p>

      <section
        data-testid="site-controls"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          void pickFile(e.dataTransfer.files[0]);
        }}
        style={{ border: '1px dashed #9ca3af', borderRadius: 6, padding: 12 }}
      >
        <p style={{ margin: '0 0 8px' }}>
          {SAMPLES.map((sample) => (
            <button
              key={sample.id}
              data-testid={`site-sample-${sample.id}`}
              type="button"
              aria-pressed={sampleId === sample.id}
              disabled={running}
              onClick={() => void pickSample(sample)}
              style={{ marginRight: 8 }}
            >
              {sample.label}
            </button>
          ))}
        </p>
        <p style={{ margin: '0 0 8px' }}>
          <label>
            Or use your own image (drop one here too):{' '}
            <input
              data-testid="site-file-input"
              type="file"
              accept="image/*"
              disabled={running}
              onChange={(e) => void pickFile(e.target.files?.[0])}
            />
          </label>
        </p>

        {webgpu ? (
          <p style={{ margin: 0 }}>
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
              This runs the whole segmentation model in your browser on{' '}
              <strong>WebGPU</strong>. This browser exposes no <code>navigator.gpu</code>{' '}
              adapter, so nothing has been downloaded and no worker has been started.
              Chrome or Edge 113+, Firefox 141+, or Safari 26+ on a supported GPU can run
              it. There is no CPU fallback by design — on CPU a single pass takes minutes.
            </p>
          </section>
        )}
      </section>

      {running && (
        <p data-testid="run-progress" role="status">
          {progressLine(progress)}
        </p>
      )}

      {error && (
        <p data-testid="run-error" role="alert">
          Failed during <strong>{error.phase}</strong>: {error.message}
        </p>
      )}

      {result && (
        <p data-testid="run-stats">{statLine(result.segments.length, result.timings.totalMs)}</p>
      )}

      {image && (
        <div
          data-testid="site-image"
          data-src={image.url}
          data-width={image.width}
          data-height={image.height}
          // Selection lives in React state and paints to a canvas, so it is
          // otherwise invisible to a browser test. Surfacing the count is what
          // lets the AC6 spec assert that click-to-select actually selected
          // something, instead of clicking and asserting nothing.
          data-selected={selected.size}
        >
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
        </div>
      )}

      <footer data-testid="site-footer" style={{ marginTop: 24, color: '#6b7280' }}>
        <a href={REPO_URL}>BoTime/NoteScanner</a> on GitHub
      </footer>
    </main>
  );
}
