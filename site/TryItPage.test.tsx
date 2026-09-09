// @vitest-environment jsdom
//
// jsdom because TryItPage is a React component. `createSegmenter` is mocked so
// nothing downloads a model; everything else in ../src/segmenter — including
// the real SegmenterFailure and DEFAULT_SEGMENTER_OPTIONS — stays real, because
// the point of these tests is how the page reacts to a REAL SegmenterFailure.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';

const segment = vi.fn();

vi.mock('../src/segmenter', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/segmenter')>();
  return { ...actual, createSegmenter: () => ({ segment, dispose: vi.fn() }) };
});

import { TryItPage } from './TryItPage';
import { DEFAULT_SEGMENTER_OPTIONS, SegmenterFailure } from '../src/segmenter';

beforeEach(() => {
  segment.mockReset();
  vi.stubGlobal('navigator', { ...navigator, gpu: {} });
  vi.stubGlobal('fetch', vi.fn(async () => ({ blob: async () => new Blob() })));
  vi.stubGlobal('createImageBitmap', vi.fn(async () => ({}) as ImageBitmap));
  // Defined ON the real URL rather than replacing the global: `new URL(...)`
  // is used elsewhere in the tree, and a plain-object stand-in for URL is not
  // constructible. `vi.unstubAllGlobals` does not undo this, which is why both
  // are `configurable` and why the value is deterministic.
  for (const [name, value] of [
    ['createObjectURL', () => 'blob:stub'],
    ['revokeObjectURL', () => {}],
  ] as const) {
    Object.defineProperty(URL, name, { configurable: true, value });
  }
  // jsdom implements none of these three; without them the sample-image effect
  // rejects unhandled and the viewer never sees a size.
  Object.defineProperty(HTMLImageElement.prototype, 'decode', {
    configurable: true,
    value: () => Promise.resolve(),
  });
  Object.defineProperty(HTMLImageElement.prototype, 'naturalWidth', {
    configurable: true,
    get: () => 1024,
  });
  Object.defineProperty(HTMLImageElement.prototype, 'naturalHeight', {
    configurable: true,
    get: () => 649,
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

async function renderWithSample() {
  render(<TryItPage />);
  await screen.findByTestId('site-image');
}

describe('TryItPage failure handling (AC7)', () => {
  it('names the phase a SegmenterFailure died in', async () => {
    segment.mockRejectedValueOnce(new SegmenterFailure('decode', 'adapter lost'));
    await renderWithSample();

    fireEvent.click(screen.getByTestId('run-segmentation'));

    const error = await screen.findByTestId('run-error');
    expect(error.textContent).toBe('Failed during decode: adapter lost');
  });

  it('renders a non-Error throw under the phase "unknown"', async () => {
    segment.mockRejectedValueOnce('something fell over');
    await renderWithSample();

    fireEvent.click(screen.getByTestId('run-segmentation'));

    const error = await screen.findByTestId('run-error');
    expect(error.textContent).toBe('Failed during unknown: something fell over');
  });

  it('stays usable after a failure: another image can be chosen and run again', async () => {
    segment.mockRejectedValueOnce(new SegmenterFailure('encode', 'nope'));
    segment.mockResolvedValueOnce({
      segments: [{ id: 'a' }, { id: 'b' }],
      timings: { totalMs: 7067 },
      counts: {},
    });
    await renderWithSample();

    fireEvent.click(screen.getByTestId('run-segmentation'));
    await screen.findByTestId('run-error');

    // Choosing the other sample clears the failure...
    fireEvent.click(screen.getByTestId('site-sample-sticky-notes'));
    await waitFor(() => expect(screen.queryByTestId('run-error')).toBeNull());

    // ...and a second run completes and reports its stats.
    fireEvent.click(screen.getByTestId('run-segmentation'));
    const stats = await screen.findByTestId('run-stats');
    expect(stats.textContent).toBe('2 segments · 7.1 s');
  });
});

describe('TryItPage image sources (AC3)', () => {
  it('accepts a dropped file, not only a chosen one', async () => {
    await renderWithSample();
    const file = new File(['x'], 'note.png', { type: 'image/png' });

    fireEvent.drop(screen.getByTestId('site-controls'), { dataTransfer: { files: [file] } });

    await waitFor(() =>
      expect(screen.getByTestId('site-image').getAttribute('data-src')).toBe('blob:stub'),
    );
    // A dropped file is not a sample: neither chip stays pressed.
    for (const id of ['cafe-table', 'sticky-notes']) {
      expect(screen.getByTestId(`site-sample-${id}`).getAttribute('aria-pressed')).toBe('false');
    }
  });
});

describe('TryItPage pins the package defaults (AC10)', () => {
  it('passes DEFAULT_SEGMENTER_OPTIONS through unmodified', async () => {
    segment.mockResolvedValueOnce({ segments: [], timings: { totalMs: 1 }, counts: {} });
    await renderWithSample();

    fireEvent.click(screen.getByTestId('run-segmentation'));

    await waitFor(() => expect(segment).toHaveBeenCalledTimes(1));
    // Identity, not deep equality: the page must not build its own options
    // object, because an object it builds is an object it can drift.
    expect(segment.mock.calls[0][1]).toBe(DEFAULT_SEGMENTER_OPTIONS);
  });
});
