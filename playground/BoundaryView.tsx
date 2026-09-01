/**
 * The Boundary tab: both mask pipelines, over procedural fixtures, with no
 * WebGPU adapter, no model download and no network. Everything renders
 * synchronously on mount — there is nothing to await.
 */
import { useEffect, useMemo, useRef } from 'react';
import {
  BOUNDARY_FIXTURES,
  BOUNDARY_GEOMETRY,
  BOUNDARY_OPTIONS,
  differingPixels,
  runBoundaryPath,
  type BinaryMask,
  type PathResult,
} from './boundary';

function MaskCanvas({
  baseline,
  next,
  testId,
}: {
  baseline: BinaryMask | null;
  next: BinaryMask | null;
  testId: string;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const { originalWidth: width, originalHeight: height } = BOUNDARY_GEOMETRY;

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const image = ctx.createImageData(width, height);
    for (let p = 0; p < width * height; p += 1) {
      const b = baseline?.coverage[p] ?? 0;
      const n = next?.coverage[p] ?? 0;
      // Blue: both. Orange: baseline only. Green: new only. Slate: neither.
      const rgb = b && n ? [37, 99, 235] : b ? [249, 115, 22] : n ? [22, 163, 74] : [17, 24, 39];
      const offset = p * 4;
      image.data[offset] = rgb[0];
      image.data[offset + 1] = rgb[1];
      image.data[offset + 2] = rgb[2];
      image.data[offset + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }, [baseline, next, width, height]);

  return (
    <canvas
      ref={ref}
      data-testid={testId}
      width={width}
      height={height}
      style={{ width: 256, imageRendering: 'pixelated', border: '1px solid #334155' }}
    />
  );
}

interface FixtureRun {
  baseline: PathResult;
  next: PathResult;
}

export function BoundaryView() {
  const runs = useMemo<FixtureRun[]>(
    () =>
      BOUNDARY_FIXTURES.map((fixture) => ({
        baseline: runBoundaryPath(fixture.windows, false),
        next: runBoundaryPath(fixture.windows, true),
      })),
    [],
  );

  const rows = runs.reduce((sum, run) => sum + run.baseline.outcomes.length, 0);
  const keptByBoth = runs.reduce(
    (sum, run) =>
      sum +
      run.baseline.outcomes.filter(
        (outcome, index) =>
          outcome.status === 'kept' && run.next.outcomes[index].status === 'kept',
      ).length,
    0,
  );

  return (
    <>
      <p data-testid="boundary-summary">
        {BOUNDARY_FIXTURES.length} procedural fixtures, {rows} candidate windows,{' '}
        {keptByBoth} kept by both paths. Baseline = <code>lowResFilterNms: false</code>{' '}
        (resample every candidate, dedupe at {BOUNDARY_GEOMETRY.originalWidth}x
        {BOUNDARY_GEOMETRY.originalHeight}); new = <code>true</code> (dedupe at{' '}
        {BOUNDARY_GEOMETRY.lowWidth}x{BOUNDARY_GEOMETRY.lowHeight}, resample the survivors).
        Gate: <code>minMaskArea</code> {BOUNDARY_OPTIONS.minMaskArea}, scaled to{' '}
        {runs[0].next.minArea} before NMS. No WebGPU, no model, no network.
      </p>

      {BOUNDARY_FIXTURES.map((fixture, f) => (
        <section key={fixture.id} data-testid={`boundary-fixture-${fixture.id}`}>
          <h2 style={{ fontSize: '1rem', marginBottom: 2 }}>{fixture.title}</h2>
          <p style={{ margin: '0 0 8px', opacity: 0.8 }}>{fixture.stresses}</p>
          <table style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th align="left">window</th>
                <th align="left">baseline</th>
                <th align="right">gate area</th>
                <th align="right">mask area</th>
                <th align="left">new</th>
                <th align="right">gate area</th>
                <th align="right">mask area</th>
                <th align="right">differing pixels</th>
                <th align="left">overlay</th>
              </tr>
            </thead>
            <tbody>
              {fixture.windows.map((_, index) => {
                const baseline = runs[f].baseline.outcomes[index];
                const next = runs[f].next.outcomes[index];
                const bothKept = baseline.status === 'kept' && next.status === 'kept';
                const diff = bothKept ? differingPixels(baseline.mask!, next.mask!) : null;
                return (
                  <tr
                    key={index}
                    data-testid={`boundary-window-${fixture.id}-${index}`}
                    data-baseline-status={baseline.status}
                    data-baseline-gate-area={baseline.gateArea}
                    data-baseline-mask-area={baseline.mask?.area}
                    data-new-status={next.status}
                    data-new-gate-area={next.gateArea}
                    data-new-mask-area={next.mask?.area}
                    data-diff={diff ?? undefined}
                  >
                    <td>#{index}</td>
                    <td>{baseline.status}</td>
                    <td align="right">{baseline.gateArea}</td>
                    <td align="right">{baseline.mask?.area ?? '—'}</td>
                    <td>{next.status}</td>
                    <td align="right">{next.gateArea}</td>
                    <td align="right">{next.mask?.area ?? '—'}</td>
                    <td align="right">
                      <strong>{diff ?? '—'}</strong>
                    </td>
                    <td>
                      <MaskCanvas
                        baseline={baseline.mask}
                        next={next.mask}
                        testId={`boundary-canvas-${fixture.id}-${index}`}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      ))}
    </>
  );
}
