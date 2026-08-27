import {
  DEFAULT_SEGMENTER_OPTIONS,
  SegmenterFailure,
  type SegmentationResult,
  type SegmenterOptions,
  type SegmenterProgress,
  type SegmenterRequest,
  type SegmenterResponse,
} from './core';

/**
 * Whether this browser can run the segmenter at all.
 *
 * Probed with `'gpu' in navigator` rather than `navigator.gpu` because
 * `Navigator.gpu` is not in TypeScript's `lib.dom` — reading the property
 * would be a compile error, and adding `@webgpu/types` for one boolean is not
 * worth a dependency.
 *
 * There is deliberately no CPU/WASM fallback: everything-mode on CPU is
 * minutes per image, which is indistinguishable from a hung tab.
 */
export function isWebGPUAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

export interface Segmenter {
  /**
   * Run one everything-mode pass. `image` is TRANSFERRED to the worker and is
   * unusable afterwards. Rejects with a `SegmenterFailure` naming the phase
   * that failed.
   */
  segment(
    image: ImageBitmap,
    options?: Partial<SegmenterOptions>,
    onProgress?: (event: SegmenterProgress) => void,
  ): Promise<SegmentationResult>;
  dispose(): void;
}

export interface CreateSegmenterConfig {
  /**
   * Overridable for two real consumers: tests inject a stub `Worker`, and a
   * host whose bundler cannot resolve `new URL(..., import.meta.url)` supplies
   * its own. Defaults to the module worker sitting beside this file.
   */
  createWorker?: () => Worker;
}

function defaultCreateWorker(): Worker {
  return new Worker(new URL('./worker/segmenter.worker.ts', import.meta.url), {
    type: 'module',
  });
}

export function createSegmenter(config: CreateSegmenterConfig = {}): Segmenter {
  const spawn = config.createWorker ?? defaultCreateWorker;
  let worker: Worker | null = null;
  let inFlight = false;

  function teardown(): void {
    worker?.terminate();
    worker = null;
  }

  return {
    segment(image, options, onProgress) {
      if (inFlight) {
        return Promise.reject(
          new SegmenterFailure('model-load', 'a segmentation run is already in flight'),
        );
      }
      inFlight = true;

      // Every run gets a brand-new worker. A previous run may have died with a
      // lost GPU adapter or a half-loaded session behind it, and there is no
      // way to interrogate that from out here — so we never try.
      teardown();
      const active = spawn();
      worker = active;

      const resolved: SegmenterOptions = { ...DEFAULT_SEGMENTER_OPTIONS, ...options };
      const startedAt = performance.now();

      return new Promise<SegmentationResult>((resolve, reject) => {
        const detach = () => {
          inFlight = false;
          active.removeEventListener('message', onMessage);
          active.removeEventListener('error', onError);
        };

        const fail = (phase: SegmenterFailure['phase'], message: string) => {
          detach();
          teardown();
          reject(new SegmenterFailure(phase, message));
        };

        const onMessage = (event: MessageEvent<SegmenterResponse>) => {
          const message = event.data;

          if (message.type === 'progress') {
            onProgress?.(message.event);
            return;
          }

          if (message.type === 'error') {
            fail(message.phase, message.message);
            return;
          }

          detach();
          // Nothing to do but name the segments: the worker encoded every mask
          // before it posted. No try/catch either — an encode failure now
          // arrives as the worker's own `error` message, already carrying
          // `phase: 'mask-encode'`.
          resolve({
            segments: message.masks.map((mask, i) => ({
              id: `segment-${i + 1}`,
              index: i + 1,
              maskUrl: mask.maskUrl,
            })),
            timings: {
              // Verbatim from the worker: every phase, `mask-encode` included,
              // is measured on that side now.
              ...message.timings,
              // Except wall clock, which is measured from here so worker spawn
              // and bitmap transfer are inside the number the table reports.
              totalMs: performance.now() - startedAt,
            },
            counts: message.counts,
          });
        };

        const onError = (event: ErrorEvent) => {
          fail('model-load', event.message || 'the segmentation worker crashed');
        };

        active.addEventListener('message', onMessage);
        active.addEventListener('error', onError);
        active.postMessage(
          { type: 'segment', bitmap: image, options: resolved } satisfies SegmenterRequest,
          [image],
        );
      });
    },

    dispose() {
      inFlight = false;
      teardown();
    },
  };
}
