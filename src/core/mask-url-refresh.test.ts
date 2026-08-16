import { describe, it, expect, vi } from 'vitest';
import { createMaskUrlRefresher, resolveMaskUrl } from './mask-url-refresh';

/** A manually-resolvable promise, so a refresh can be held in-flight. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe('createMaskUrlRefresher', () => {
  it('gives every concurrent caller the same in-flight promise (one request for N rows)', async () => {
    const gate = deferred<Map<string, string>>();
    const fetchFreshUrls = vi.fn(() => gate.promise);
    const refresher = createMaskUrlRefresher(fetchFreshUrls);

    // 22 rows all fail at once and all ask for a refresh.
    const calls = Array.from({ length: 22 }, () => refresher.refresh());
    gate.resolve(new Map([['a', 'https://r2/fresh-a']]));
    const results = await Promise.all(calls);

    expect(fetchFreshUrls).toHaveBeenCalledTimes(1);
    for (const result of results) {
      expect(result).toBe(results[0]);
      expect(result.get('a')).toBe('https://r2/fresh-a');
    }
  });

  it('serves the cached result to later callers without issuing a second request', async () => {
    const fetchFreshUrls = vi.fn(async () =>
      new Map([['a', 'https://r2/fresh-a']]),
    );
    const refresher = createMaskUrlRefresher(fetchFreshUrls);

    const first = await refresher.refresh();
    const second = await refresher.refresh();

    expect(fetchFreshUrls).toHaveBeenCalledTimes(1);
    expect(second).toBe(first);
  });

  it('does not wedge into a permanently-pending state when the refresh rejects', async () => {
    const fetchFreshUrls = vi.fn(async () => {
      throw new Error('refresh failed');
    });
    const refresher = createMaskUrlRefresher(fetchFreshUrls);

    // The rejection must be observable (not swallowed into a hanging promise),
    // and the one-shot policy means no second request is ever issued.
    await expect(refresher.refresh()).rejects.toThrow('refresh failed');
    await expect(refresher.refresh()).rejects.toThrow('refresh failed');
    expect(fetchFreshUrls).toHaveBeenCalledTimes(1);
  });
});

describe('resolveMaskUrl', () => {
  it('returns the fresh url when the refresh produced one for that segment', () => {
    const fresh = new Map([['seg-1', 'https://r2/fresh-1']]);
    expect(resolveMaskUrl('seg-1', 'https://r2/stale-1', fresh)).toBe(
      'https://r2/fresh-1',
    );
  });

  it('returns the original url when no refresh has happened yet', () => {
    expect(resolveMaskUrl('seg-1', 'https://r2/stale-1', null)).toBe(
      'https://r2/stale-1',
    );
  });

  it('returns the original url when the refresh omitted that segment', () => {
    const fresh = new Map([['seg-2', 'https://r2/fresh-2']]);
    expect(resolveMaskUrl('seg-1', 'https://r2/stale-1', fresh)).toBe(
      'https://r2/stale-1',
    );
  });

  it('returns the original url when the refresh gave an empty string', () => {
    const fresh = new Map([['seg-1', '']]);
    expect(resolveMaskUrl('seg-1', 'https://r2/stale-1', fresh)).toBe(
      'https://r2/stale-1',
    );
  });
});
