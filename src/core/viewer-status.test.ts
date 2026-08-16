import { describe, it, expect } from 'vitest';
import {
  canReuseBaseImage,
  describeMaskError,
  resolveViewerStatus,
  type ViewerStatusInput,
} from './viewer-status';

// Defaults describe the state on first mount: nothing loaded, nothing failed.
function input(over: Partial<ViewerStatusInput> = {}): ViewerStatusInput {
  return {
    baseLoaded: false,
    masksSettled: false,
    baseError: false,
    failedMaskCount: 0,
    ...over,
  };
}

describe('resolveViewerStatus', () => {
  it('is loading before anything has loaded', () => {
    expect(resolveViewerStatus(input())).toBe('loading');
  });

  // The bug this whole change exists to fix: PHASE 1 completing is NOT enough
  // to show the card, or the user sees a raw photo with no segment overlay.
  it('stays loading when the base image is decoded but masks are not', () => {
    expect(resolveViewerStatus(input({ baseLoaded: true }))).toBe('loading');
  });

  it('is ready only when the base image and every mask are done', () => {
    expect(
      resolveViewerStatus(input({ baseLoaded: true, masksSettled: true })),
    ).toBe('ready');
  });

  it('stays loading when masks settled but the base image has not decoded', () => {
    expect(resolveViewerStatus(input({ masksSettled: true }))).toBe('loading');
  });

  // Decision 2: a failed mask is an error state with a retry, never a warning
  // painted over a partially-masked image.
  it('is error — not ready — when a mask fails after the retry', () => {
    expect(
      resolveViewerStatus(
        input({ baseLoaded: true, masksSettled: true, failedMaskCount: 1 }),
      ),
    ).toBe('error');
  });

  it('is error as soon as a mask has failed, even before the rest settle', () => {
    // Otherwise the card would spin until the slowest surviving mask resolved,
    // only to then show an error anyway.
    expect(
      resolveViewerStatus(
        input({ baseLoaded: true, masksSettled: false, failedMaskCount: 2 }),
      ),
    ).toBe('error');
  });

  it('is error when the base image itself failed', () => {
    expect(resolveViewerStatus(input({ baseError: true }))).toBe('error');
  });

  // An image with no segments has no PHASE 2 work; the viewer marks masks
  // settled immediately. It must not hang on masks that will never arrive.
  it('reaches ready for a zero-segment image', () => {
    expect(
      resolveViewerStatus(
        input({ baseLoaded: true, masksSettled: true, failedMaskCount: 0 }),
      ),
    ).toBe('ready');
  });

  // Retry path: clearing the failure count returns the viewer to loading, and
  // a successful reload then reaches ready.
  it('goes error -> loading -> ready across a retry', () => {
    const failed = input({
      baseLoaded: true,
      masksSettled: true,
      failedMaskCount: 3,
    });
    expect(resolveViewerStatus(failed)).toBe('error');

    const retrying = { ...failed, masksSettled: false, failedMaskCount: 0 };
    expect(resolveViewerStatus(retrying)).toBe('loading');

    const recovered = { ...retrying, masksSettled: true };
    expect(resolveViewerStatus(recovered)).toBe('ready');
  });
});

describe('canReuseBaseImage', () => {
  const url = 'https://r2.example/img-1.jpg?sig=abc';

  // The point of the cache: a mask retry must not refetch the photo.
  it('reuses the cache when it belongs to the url being loaded', () => {
    expect(canReuseBaseImage(url, url)).toBe(true);
  });

  it('does not reuse a cache from a different image', () => {
    expect(canReuseBaseImage('https://r2.example/img-2.jpg', url)).toBe(false);
  });

  it('does not reuse an empty cache (first load must fetch)', () => {
    expect(canReuseBaseImage(null, url)).toBe(false);
    expect(canReuseBaseImage(undefined, url)).toBe(false);
  });

  // A refreshed presigned url is a different string, so the photo is refetched.
  // Correct-but-conservative: reuse must never outlive the exact url it cached.
  it('does not reuse across a changed query string', () => {
    expect(
      canReuseBaseImage('https://r2.example/img-1.jpg?sig=old', url),
    ).toBe(false);
  });
});

describe('describeMaskError', () => {
  it('reports the failed count against the total', () => {
    expect(describeMaskError({ failed: 3, total: 65 })).toContain('3 of 65');
  });

  it('singularises a single failure', () => {
    expect(describeMaskError({ failed: 1, total: 4 })).toContain(
      '1 of 4 segment mask failed',
    );
  });

  it('pluralises multiple failures', () => {
    expect(describeMaskError({ failed: 2, total: 4 })).toContain(
      '2 of 4 segment masks failed',
    );
  });

  it('explains that the image is withheld rather than shown partially', () => {
    expect(describeMaskError({ failed: 1, total: 2 })).toMatch(/hidden until/i);
  });
});
