// @vitest-environment jsdom
//
// jsdom because CompareView is a React component. No WebGPU, no worker and no
// network: `createSegmenter` and `createBitmap` are both injected, mirroring
// the injection points `runRow` had on the PR #11 branch. `navigator.gpu` is
// stubbed truthy so the component renders its controls rather than the
// WebGPU-required panel.
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { CompareView } from './CompareView';
import { budgetMsOf, decodePathOf, type RunRecord } from './compare';
import {
  SegmenterFailure,
  createTimingAccumulator,
  type SegmentationResult,
  type Segmenter,
  type SegmenterOptions,
} from '../src/segmenter';

function stubResult(totalMs: number): SegmentationResult {
  const accumulator = createTimingAccumulator();
  accumulator.record('model-load', 400);
  accumulator.record('decode', 250);
  return {
    segments: [],
    timings: accumulator.report(totalMs),
    counts: { raw: 96, afterFilter: 31, afterNms: 12, returned: 12 },
  };
}

/** Records the options each call was made with, so AC7 can be checked. */
function stubSegmenter(outcome: SegmentationResult | Error) {
  const seen: Partial<SegmenterOptions>[] = [];
  const segmenter: Segmenter = {
    segment: vi.fn(async (_image, options) => {
      seen.push({ ...options });
      if (outcome instanceof Error) throw outcome;
      return outcome;
    }) as Segmenter['segment'],
    dispose: vi.fn(),
  };
  return { segmenter, seen };
}

function mount(outcome: SegmentationResult | Error) {
  const { segmenter, seen } = stubSegmenter(outcome);
  render(
    <CompareView
      createSegmenter={() => segmenter}
      createBitmap={async () => ({ width: 4, height: 4 }) as ImageBitmap}
    />,
  );
  return { segmenter, seen };
}

function runJson(): HTMLElement {
  return screen.getByTestId('run-json');
}

function runCount(): number {
  return Number(runJson().getAttribute('data-run-count'));
}

beforeEach(() => {
  vi.stubGlobal('navigator', { ...navigator, gpu: {}, clipboard: { writeText: vi.fn() } });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('CompareView', () => {
  it('exposes every control the sweep drives (AC9)', () => {
    mount(stubResult(1400));
    for (const id of [
      'dtype', 'batch-size', 'points-per-side', 'overlap-decode-filter',
      'gpu-resident-embeddings', 'keep-raw-masks', 'run-row', 'run-json', 'copy-markdown',
    ]) {
      expect(screen.getByTestId(id)).toBeTruthy();
    }
    // batchSize 64 is reachable by hand even though it is out of the default grid.
    const batch = screen.getByTestId('batch-size') as HTMLSelectElement;
    expect([...batch.options].map((option) => option.value)).toEqual(['8', '16', '32', '64']);
  });

  it('installs the __decodeSweep hook the runner reaches through', () => {
    mount(stubResult(1400));
    expect(typeof window.__decodeSweep?.expandGrid).toBe('function');
    expect(typeof window.__decodeSweep?.toMarkdown).toBe('function');
    expect(typeof window.__decodeSweep?.resolveConfig).toBe('function');
    expect(typeof window.__decodeSweep?.warmUpRow).toBe('function');
  });

  it('runs the injected segmenter with the controls and emits a run-json blob', async () => {
    const result = stubResult(1400);
    const { seen } = mount(result);

    fireEvent.change(screen.getByTestId('dtype'), { target: { value: 'fp16' } });
    fireEvent.change(screen.getByTestId('batch-size'), { target: { value: '32' } });
    fireEvent.click(screen.getByTestId('overlap-decode-filter'));
    expect(runCount()).toBe(0);
    fireEvent.click(screen.getByTestId('run-row'));

    await waitFor(() => expect(runCount()).toBe(1));
    expect(seen[0]).toMatchObject({
      dtype: 'fp16',
      batchSize: 32,
      overlapDecodeFilter: true,
      gpuResidentEmbeddings: false,
    });

    const record = JSON.parse(runJson().textContent!) as RunRecord;
    expect(record.status).toBe('ok');
    expect(record.counts).toEqual(result.counts);
    expect(record.timings!.totalMs).toBe(1400);
    expect(record.budgetMs).toBe(budgetMsOf(result.timings));
    expect(decodePathOf(record.options)).toBe('overlap');
    // The heavy fields never go through the <pre>.
    expect('segments' in record).toBe(false);
    expect('rawMasks' in record).toBe(false);
  });

  it('labels a captured row from its OWN options after the controls move on (AC7)', async () => {
    mount(stubResult(1400));
    fireEvent.click(screen.getByTestId('run-row'));
    await waitFor(() => expect(runCount()).toBe(1));

    // The run above was fp32. Move every control afterwards.
    fireEvent.change(screen.getByTestId('dtype'), { target: { value: 'fp16' } });
    fireEvent.change(screen.getByTestId('batch-size'), { target: { value: '32' } });
    fireEvent.click(screen.getByTestId('gpu-resident-embeddings'));

    const row = screen.getByTestId('compare-table').querySelectorAll('tbody tr')[0];
    expect(row.textContent).toContain('fp32');
    expect(row.textContent).toContain('none');
    expect(row.textContent).not.toContain('fp16');
    // And the exported markdown agrees with the table.
    expect(JSON.parse(runJson().textContent!).options.dtype).toBe('fp32');
  });

  it('records a failed run as failed and still ticks the run counter (AC12)', async () => {
    mount(new SegmenterFailure('decode', 'Array buffer allocation failed'));
    fireEvent.click(screen.getByTestId('run-row'));

    await waitFor(() => expect(runCount()).toBe(1));
    const record = JSON.parse(runJson().textContent!) as RunRecord;
    expect(record.status).toBe('failed');
    expect(record.phase).toBe('decode');
    expect(record.message).toContain('allocation failed');
    expect(record.budgetMs).toBeUndefined();
    expect(screen.getByTestId('compare-table').textContent).toContain('decode');
  });
});
