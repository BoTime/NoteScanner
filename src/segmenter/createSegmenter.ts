import {
  DEFAULT_SEGMENTER_OPTIONS,
  SegmenterFailure,
  encodeMaskPng,
  summarizePhase,
  type SegmentationResult,
  type SegmenterOptions,
  type SegmenterProgress,
  type SegmenterRequest,
  type SegmenterResponse,
} from './core';
import type { ViewerSegment } from '../types';

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

        const onMessage = async (event: MessageEvent<SegmenterResponse>) => {
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
          try {
            // Mask -> PNG happens here, on the main thread, because a data URL
            // is all `SegmentViewer` ever consumes and the worker has already
            // handed the coverage buffers over.
            const encodeSamples: number[] = [];
            const segments: ViewerSegment[] = [];
            for (let i = 0; i < message.masks.length; i += 1) {
              const encodeStarted = performance.now();
              const maskUrl = await encodeMaskPng(
                message.masks[i].coverage,
                message.width,
                message.height,
              );
              encodeSamples.push(performance.now() - encodeStarted);
              segments.push({ id: `segment-${i + 1}`, index: i + 1, maskUrl });
            }

            resolve({
              segments,
              timings: {
                phases: {
                  ...message.timings.phases,
                  'mask-encode': summarizePhase(encodeSamples),
                },
                // Straight passthrough: the main thread contributes nothing to
                // the `filter` stage.
                filterSubPhases: message.timings.filterSubPhases,
                // Wall clock from this side of the boundary, so worker spawn
                // and bitmap transfer are inside the number the table reports.
                totalMs: performance.now() - startedAt,
              },
              counts: message.counts,
              // Spread so the key is ABSENT rather than undefined when the
              // caller did not ask: the buffers must stay collectable for the
              // common case, which is every run the Segment view makes.
              ...(resolved.keepRawMasks ? { rawMasks: message.masks } : {}),
            });
          } catch (error) {
            fail('mask-encode', error instanceof Error ? error.message : String(error));
          }
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
