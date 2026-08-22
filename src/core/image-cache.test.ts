import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadImage } from './image-cache';

// Decides, per URL, whether the next decode succeeds or fails. Rebuilt by each
// test so the eviction test can flip a URL from 'error' to 'load' and prove a
// re-attempt.
let outcomes: Map<string, 'load' | 'error'> = new Map();

// The module-level cache persists across every test in this file (vitest shares
// one module registry per file, and `afterEach` only resets `outcomes`), so
// each test must use a URL no earlier test has loaded — otherwise `loadImage`
// short-circuits on a previously resolved promise and never builds a FakeImage.
// `nextUrl()` hands out a globally unique URL per call, so no test can
// accidentally collide with a URL an earlier test already cached.
let urlCounter = 0;
const nextUrl = () => `url-${urlCounter++}`;

// Mirrors MAX_CACHED_IMAGES in image-cache.ts. The cap test inserts one more
// URL than this so it genuinely crosses the eviction threshold; if the cap is
// raised above that count, the test's eviction assertions fail loudly rather
// than silently no-op.
const CACHE_CAP = 100;

class FakeImage {
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  crossOrigin: string | null = null;
  private _src = '';
  get src(): string {
    return this._src;
  }
  set src(value: string) {
    this._src = value;
    // Mirrors a real <img>: the load/error callback fires on the next
    // microtask after `src` is assigned, never synchronously.
    queueMicrotask(() => {
      if (outcomes.get(value) === 'error') this.onerror?.();
      else this.onload?.();
    });
  }
}

afterEach(() => {
  outcomes = new Map();
  vi.unstubAllGlobals();
});

describe('loadImage', () => {
  it('shares a single promise across concurrent callers of the same URL', async () => {
    vi.stubGlobal('Image', FakeImage);
    const url = nextUrl();
    outcomes.set(url, 'load');

    const first = loadImage(url);
    const second = loadImage(url);

    expect(first).toBe(second);
    const [a, b] = await Promise.all([first, second]);
    expect(a).toBe(b);
  });

  it('sets crossOrigin "anonymous" so canvas reads stay untainted', async () => {
    vi.stubGlobal('Image', FakeImage);
    const url = nextUrl();
    outcomes.set(url, 'load');

    const img = await loadImage(url);
    expect(img.crossOrigin).toBe('anonymous');
  });

  it('keeps distinct URLs on separate promises', async () => {
    vi.stubGlobal('Image', FakeImage);
    const aUrl = nextUrl();
    const bUrl = nextUrl();
    outcomes.set(aUrl, 'load');
    outcomes.set(bUrl, 'load');

    const a = loadImage(aUrl);
    const b = loadImage(bUrl);

    expect(a).not.toBe(b);
    expect(await a).not.toBe(await b);
  });

  it('caches a resolved image for later callers', async () => {
    vi.stubGlobal('Image', FakeImage);
    const url = nextUrl();
    outcomes.set(url, 'load');

    const first = await loadImage(url);
    const again = await loadImage(url);

    expect(again).toBe(first);
  });

  it('evicts a rejected URL so a later caller re-attempts', async () => {
    vi.stubGlobal('Image', FakeImage);
    const url = nextUrl();
    outcomes.set(url, 'error');

    const failed = loadImage(url);
    await expect(failed).rejects.toThrow(`Failed to load ${url}`);

    // The failure is now evicted; flipping the outcome lets a retry succeed.
    outcomes.set(url, 'load');
    const retry = loadImage(url);
    expect(retry).not.toBe(failed);
    await expect(retry).resolves.toBeInstanceOf(FakeImage);
  });

  it('rejection evicts only the rejected URL, leaving other cached URLs intact', async () => {
    vi.stubGlobal('Image', FakeImage);
    const resolvedUrl = nextUrl();
    const rejectedUrl = nextUrl();
    outcomes.set(resolvedUrl, 'load');
    outcomes.set(rejectedUrl, 'error');

    const resolved = loadImage(resolvedUrl);
    const rejected = loadImage(rejectedUrl);
    await expect(rejected).rejects.toThrow(`Failed to load ${rejectedUrl}`);

    // Only the rejected URL was evicted: re-requesting it builds a fresh promise...
    expect(loadImage(rejectedUrl)).not.toBe(rejected);
    // ...while the resolved URL is still cached, returning the same promise
    // (and thus the same decoded image).
    const resolvedAgain = loadImage(resolvedUrl);
    expect(resolvedAgain).toBe(resolved);
    expect(await resolvedAgain).toBe(await resolved);
  });

  it('bounds the cache, evicting the oldest entry once the cap is reached', async () => {
    vi.stubGlobal('Image', FakeImage);

    // Insert one more URL than the cap. Every URL resolves, so the only
    // eviction is cap-driven, not rejection-driven.
    const oldestUrl = nextUrl();
    const oldest = loadImage(oldestUrl);

    const extraUrls = Array.from({ length: CACHE_CAP - 1 }, () => nextUrl());
    for (const url of extraUrls) {
      loadImage(url);
    }

    const newestUrl = nextUrl();
    const newest = loadImage(newestUrl);

    // The map now holds CACHE_CAP entries; the oldest was evicted to make room.
    const refetchedOldest = loadImage(oldestUrl);
    expect(refetchedOldest).not.toBe(oldest);

    // The most recently inserted URL is still cached.
    expect(loadImage(newestUrl)).toBe(newest);

    // Drain the promises so the microtask queue settles cleanly.
    await Promise.all([
      oldest,
      newest,
      refetchedOldest,
      ...extraUrls.map((url) => loadImage(url)),
    ]);
  });
});
