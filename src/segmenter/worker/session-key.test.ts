import { describe, it, expect } from 'vitest';
import { DEFAULT_SEGMENTER_OPTIONS, type SegmenterOptions } from '../core';
import { embeddingsSessionOptions, sessionCacheKey } from './session-key';

function options(overrides: Partial<SegmenterOptions> = {}): SegmenterOptions {
  return { ...DEFAULT_SEGMENTER_OPTIONS, ...overrides };
}

describe('sessionCacheKey', () => {
  it('separates the two embedding residencies (AC3)', () => {
    const cpu = sessionCacheKey(options({ gpuResidentEmbeddings: false }));
    const gpu = sessionCacheKey(options({ gpuResidentEmbeddings: true }));
    expect(cpu).not.toBe(gpu);
  });

  it('still separates modelId and dtype', () => {
    expect(sessionCacheKey(options({ dtype: 'fp32' }))).not.toBe(
      sessionCacheKey(options({ dtype: 'fp16' })),
    );
    expect(sessionCacheKey(options({ modelId: 'a' }))).not.toBe(
      sessionCacheKey(options({ modelId: 'b' })),
    );
  });

  it('ignores keepRawMasks, which changes nothing about the session (AC3)', () => {
    expect(sessionCacheKey(options({ keepRawMasks: true }))).toBe(
      sessionCacheKey(options({ keepRawMasks: false })),
    );
  });

  it('ignores the per-call inference knobs, which are not session state', () => {
    const base = sessionCacheKey(options());
    expect(sessionCacheKey(options({ pointsPerSide: 32 }))).toBe(base);
    expect(sessionCacheKey(options({ batchSize: 64 }))).toBe(base);
    expect(sessionCacheKey(options({ overlapDecodeFilter: true }))).toBe(base);
  });

  it('ignores lowResFilterNms, which changes nothing about the session (AC5)', () => {
    expect(sessionCacheKey(options({ lowResFilterNms: true }))).toBe(
      sessionCacheKey(options({ lowResFilterNms: false })),
    );
  });
});

describe('embeddingsSessionOptions', () => {
  it('is undefined on the default CPU-resident path', () => {
    expect(embeddingsSessionOptions(options({ gpuResidentEmbeddings: false }))).toBeUndefined();
  });

  it("names EXACTLY the encoder's two outputs as gpu-buffer (AC4)", () => {
    const resolved = embeddingsSessionOptions(options({ gpuResidentEmbeddings: true }));
    expect(resolved).toBeDefined();
    // Exact, not a superset: naming pred_masks or iou_scores here would move a
    // tensor the filter stage reads on the CPU onto the device.
    expect(Object.keys(resolved!.preferredOutputLocation)).toEqual([
      'image_embeddings',
      'image_positional_embeddings',
    ]);
    expect(Object.values(resolved!.preferredOutputLocation)).toEqual([
      'gpu-buffer',
      'gpu-buffer',
    ]);
  });
});
