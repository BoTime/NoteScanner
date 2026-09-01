// @vitest-environment jsdom
//
// jsdom because SegmentView is a React component. Nothing here runs the model:
// `navigator.gpu` is stubbed truthy only so the component renders its controls
// instead of the WebGPU-required panel, and `Image.decode` is stubbed because
// jsdom does not implement it and the sample-image effect would otherwise
// reject unhandled. The run button stays disabled throughout — this file is
// about the controls, not about segmentation.
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';

import { SegmentView } from './SegmentView';
import { DEFAULT_SEGMENTER_OPTIONS } from '../src/segmenter';

beforeEach(() => {
  vi.stubGlobal('navigator', { ...navigator, gpu: {} });
  // jsdom has no decode(); resolve it so the sample-image effect settles.
  Object.defineProperty(HTMLImageElement.prototype, 'decode', {
    configurable: true,
    value: () => Promise.resolve(),
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('SegmentView inference controls', () => {
  it('offers dtype and batch size, seeded from the package defaults', () => {
    render(<SegmentView />);

    const dtype = screen.getByTestId('segment-dtype') as HTMLSelectElement;
    const batch = screen.getByTestId('segment-batch-size') as HTMLSelectElement;

    // Seeded from DEFAULT_SEGMENTER_OPTIONS rather than from literals, so the
    // day a default moves this test follows it instead of failing spuriously.
    expect(dtype.value).toBe(DEFAULT_SEGMENTER_OPTIONS.dtype);
    expect(batch.value).toBe(String(DEFAULT_SEGMENTER_OPTIONS.batchSize));
  });

  it('drives the options a run would use', () => {
    render(<SegmentView />);

    const batch = screen.getByTestId('segment-batch-size') as HTMLSelectElement;
    fireEvent.change(batch, { target: { value: '8' } });
    expect(batch.value).toBe('8');

    const dtype = screen.getByTestId('segment-dtype') as HTMLSelectElement;
    fireEvent.change(dtype, { target: { value: 'fp16' } });
    expect(dtype.value).toBe('fp16');
  });

  it('offers the batch size that OOMs, so it stays reachable by hand', () => {
    render(<SegmentView />);

    const batch = screen.getByTestId('segment-batch-size') as HTMLSelectElement;
    const offered = Array.from(batch.options).map((option) => option.value);
    // 64 dies in the upsample. Reproducing that must not require editing code
    // — the same reasoning the Compare tab's list carries.
    expect(offered).toContain('64');
  });

  it('does not collide with the Compare tab test ids the sweep runner drives', () => {
    render(<SegmentView />);

    // scripts/sweep-decode.mjs selects `[data-testid="batch-size"]`. If this
    // view ever claimed that id too, the runner could drive the wrong control.
    expect(screen.queryByTestId('batch-size')).toBeNull();
    expect(screen.queryByTestId('dtype')).toBeNull();
  });
});
