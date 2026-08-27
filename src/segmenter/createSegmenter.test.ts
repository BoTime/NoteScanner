import { afterEach, beforeEach, describe, it, expect, vi } from 'vitest';
import { createSegmenter, isWebGPUAvailable } from './createSegmenter';
import {
  FILTER_SUBSTEP_ORDER,
  PHASE_ORDER,
  SegmenterFailure,
  createTimingAccumulator,
  type RawMask,
  type SegmenterProgress,
  type SegmenterRequest,
  type SegmenterResponse,
} from './core';

/**
 * A stand-in for a real `Worker`: it records what was posted to it and lets
 * the test drive the response side by hand. No GPU, no network, no bundler.
 */
class FakeWorker extends EventTarget {
  static instances: FakeWorker[] = [];
  readonly posted: SegmenterRequest[] = [];
  readonly transfers: unknown[][] = [];
  terminated = false;

  constructor() {
    super();
    FakeWorker.instances.push(this);
  }

  postMessage(message: SegmenterRequest, transfer: unknown[] = []) {
    this.posted.push(message);
    this.transfers.push(transfer);
  }

  terminate() {
    this.terminated = true;
  }

  emit(data: SegmenterResponse) {
    this.dispatchEvent(Object.assign(new Event('message'), { data }));
  }
}

function spawn() {
  return new FakeWorker() as unknown as Worker;
}

function fakeBitmap() {
  return { width: 4, height: 4, close: vi.fn() } as unknown as ImageBitmap;
}

function doneMessage(masks: RawMask[] = []): SegmenterResponse {
  return {
    type: 'done',
    masks,
    width: 2,
    height: 1,
    // 42 deliberately differs from any main-thread clock reading, so a test
    // can prove totalMs is measured here and not copied from the worker.
    timings: createTimingAccumulator().report(42),
    counts: { raw: 24, afterFilter: 9, afterNms: masks.length },
  };
}

beforeEach(() => {
  FakeWorker.instances = [];
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('isWebGPUAvailable', () => {
  it('is false when the navigator exposes no gpu', () => {
    vi.stubGlobal('navigator', {});
    expect(isWebGPUAvailable()).toBe(false);
  });

  it('is true when it does', () => {
    vi.stubGlobal('navigator', { gpu: {} });
    expect(isWebGPUAvailable()).toBe(true);
  });
});

describe('createSegmenter', () => {
  it('posts the merged options and transfers the bitmap', async () => {
    const segmenter = createSegmenter({ createWorker: spawn });
    const bitmap = fakeBitmap();
    const pending = segmenter.segment(bitmap, { pointsPerSide: 32 });

    const worker = FakeWorker.instances[0];
    expect(worker.posted).toHaveLength(1);
    expect(worker.posted[0].options.pointsPerSide).toBe(32);
    // Untouched defaults still travel, so the worker never guesses.
    expect(worker.posted[0].options.nmsIouThreshold).toBe(0.7);
    expect(worker.transfers[0]).toEqual([bitmap]);

    worker.emit(doneMessage());
    await pending;
  });

  it('resolves with counts, timings and a mask-encode phase', async () => {
    const segmenter = createSegmenter({ createWorker: spawn });
    const pending = segmenter.segment(fakeBitmap());
    FakeWorker.instances[0].emit(doneMessage());

    const result = await pending;
    expect(result.segments).toEqual([]);
    expect(result.counts).toEqual({ raw: 24, afterFilter: 9, afterNms: 0 });
    expect(Object.keys(result.timings.phases)).toEqual([...PHASE_ORDER]);
    expect(result.timings.phases['mask-encode'].count).toBe(0);
    // totalMs is measured on the main thread, not copied from the worker.
    expect(result.timings.totalMs).not.toBe(42);
  });

  it('carries the worker filterSubPhases through the rebuilt report', async () => {
    const workerTimings = createTimingAccumulator();
    workerTimings.recordFilterSub('select', 4);
    workerTimings.recordFilterSub('select', 6);
    workerTimings.recordFilterSub('upscale', 100);

    const segmenter = createSegmenter({ createWorker: spawn });
    const pending = segmenter.segment(fakeBitmap());
    FakeWorker.instances[0].emit({
      type: 'done',
      masks: [],
      width: 2,
      height: 1,
      timings: workerTimings.report(42),
      counts: { raw: 24, afterFilter: 9, afterNms: 0 },
    });

    const result = await pending;
    expect(Object.keys(result.timings.filterSubPhases)).toEqual([...FILTER_SUBSTEP_ORDER]);
    expect(result.timings.filterSubPhases.select.total).toBe(10);
    expect(result.timings.filterSubPhases.select.p50).toBe(6);
    expect(result.timings.filterSubPhases.upscale.total).toBe(100);
    expect(result.timings.filterSubPhases.upscale.p50).toBe(100);
    // The main thread contributes nothing to `filter`, so an unrecorded
    // sub-step still arrives zero-filled rather than missing.
    expect(result.timings.filterSubPhases.threshold.count).toBe(0);
  });

  it('forwards every progress event to the callback', async () => {
    const seen: SegmenterProgress[] = [];
    const segmenter = createSegmenter({ createWorker: spawn });
    const pending = segmenter.segment(fakeBitmap(), undefined, (event) => seen.push(event));

    const worker = FakeWorker.instances[0];
    worker.emit({ type: 'progress', event: { phase: 'model-load', done: 1, total: 1, ms: 5 } });
    worker.emit({ type: 'progress', event: { phase: 'decode', done: 2, total: 8, ms: 12 } });
    worker.emit(doneMessage());
    await pending;

    expect(seen).toEqual([
      { phase: 'model-load', done: 1, total: 1, ms: 5 },
      { phase: 'decode', done: 2, total: 8, ms: 12 },
    ]);
  });

  it('rejects with a typed failure naming the phase that died', async () => {
    const segmenter = createSegmenter({ createWorker: spawn });
    const pending = segmenter.segment(fakeBitmap());
    FakeWorker.instances[0].emit({
      type: 'error',
      phase: 'encode',
      message: 'GPU adapter lost',
    });

    await expect(pending).rejects.toBeInstanceOf(SegmenterFailure);
    await pending.catch((error: SegmenterFailure) => {
      expect(error.phase).toBe('encode');
      expect(error.message).toBe('GPU adapter lost');
    });
  });

  it('starts the run after a failure on a freshly created worker', async () => {
    const segmenter = createSegmenter({ createWorker: spawn });
    const failed = segmenter.segment(fakeBitmap());
    FakeWorker.instances[0].emit({ type: 'error', phase: 'decode', message: 'boom' });
    await failed.catch(() => undefined);

    expect(FakeWorker.instances[0].terminated).toBe(true);

    const retried = segmenter.segment(fakeBitmap());
    expect(FakeWorker.instances).toHaveLength(2);
    expect(FakeWorker.instances[1].terminated).toBe(false);

    FakeWorker.instances[1].emit(doneMessage());
    await retried;
  });

  it('rejects a worker crash as a failure rather than hanging forever', async () => {
    const segmenter = createSegmenter({ createWorker: spawn });
    const pending = segmenter.segment(fakeBitmap());
    FakeWorker.instances[0].dispatchEvent(
      Object.assign(new Event('error'), { message: 'worker died' }),
    );

    await expect(pending).rejects.toThrow(/worker died/);
  });

  it('terminates the worker on dispose', async () => {
    const segmenter = createSegmenter({ createWorker: spawn });
    const pending = segmenter.segment(fakeBitmap());
    FakeWorker.instances[0].emit(doneMessage());
    await pending;

    segmenter.dispose();
    expect(FakeWorker.instances[0].terminated).toBe(true);
  });

  it('refuses to start a second run while one is in flight', async () => {
    const segmenter = createSegmenter({ createWorker: spawn });
    const first = segmenter.segment(fakeBitmap());

    await expect(segmenter.segment(fakeBitmap())).rejects.toThrow(/already in flight/);
    expect(FakeWorker.instances).toHaveLength(1);

    FakeWorker.instances[0].emit(doneMessage());
    await first;
  });
});
