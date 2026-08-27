import { useMemo, useState } from 'react';
import { SegmentViewer, createCanvas2DRenderer, type RendererFactory } from '../src';
import { makeSyntheticSegments } from './fixtures';
import { BENCH_SIZES } from './benchmark';

const IMAGE_W = 1200;
const IMAGE_H = 900;

const RENDERERS: Record<string, RendererFactory> = {
  canvas2d: createCanvas2DRenderer,
};

export function BenchmarkView() {
  const [count, setCount] = useState<number>(BENCH_SIZES[0]);
  const [rendererKey, setRendererKey] = useState('canvas2d');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const segments = useMemo(() => makeSyntheticSegments(count, IMAGE_W, IMAGE_H), [count]);
  const baseUrl = useMemo(() => {
    const c = document.createElement('canvas');
    c.width = IMAGE_W;
    c.height = IMAGE_H;
    const ctx = c.getContext('2d')!;
    const grad = ctx.createLinearGradient(0, 0, IMAGE_W, IMAGE_H);
    grad.addColorStop(0, '#fde68a');
    grad.addColorStop(1, '#93c5fd');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, IMAGE_W, IMAGE_H);
    return c.toDataURL('image/png');
  }, []);

  return (
    <>
      <p>
        <label>
          masks:{' '}
          <select
            data-testid="bench-mask-count"
            value={count}
            onChange={(e) => setCount(Number(e.target.value))}
          >
            {BENCH_SIZES.map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>{' '}
        <label>
          renderer:{' '}
          <select
            data-testid="bench-renderer"
            value={rendererKey}
            onChange={(e) => setRendererKey(e.target.value)}
          >
            {Object.keys(RENDERERS).map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
        </label>
      </p>
      <SegmentViewer
        key={`${count}-${rendererKey}`}
        imageUrl={baseUrl}
        imageWidth={IMAGE_W}
        imageHeight={IMAGE_H}
        segments={segments}
        initialSelectedIds={selected}
        onSelectionChange={setSelected}
        onCreateSegment={async () => {}}
        renderer={RENDERERS[rendererKey]}
        maxHeight="70vh"
      />
    </>
  );
}
