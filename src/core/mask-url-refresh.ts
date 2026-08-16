/**
 * Pure coordination policy for refreshing expired presigned mask URLs in the
 * transcribed-text review panel.
 *
 * The panel itself cannot be unit-tested (no jsdom, no canvas in the Vitest
 * `node` environment), so the decision-carrying parts live here with direct
 * tests. Mirrors mask-loading.ts / segment-viewer-logic.ts.
 *
 * Why the PANEL and not each row owns this: all mask URLs are signed in a
 * single batch pass server-side, so they expire together. If each of the ~22
 * PostItPreview rows refetched the detail DTO on its own failure, one expiry
 * would fire ~22 identical requests at once. Instead every caller shares one
 * in-flight promise, and the refresh is ONE SHOT for the panel's lifetime —
 * no timer, no loop, matching the policy planMaskRetry already encodes.
 */

/** Fetches fresh presigned mask URLs for the current image: id -> maskUrl. */
export type RefreshMaskUrls = () => Promise<Map<string, string>>;

export interface MaskUrlRefresher {
  /**
   * Ask for fresh mask URLs. Concurrent callers share the single in-flight
   * promise; later callers get the same settled promise back. Rejections
   * propagate to every caller and are NOT retried.
   */
  refresh(): Promise<Map<string, string>>;
}

export function createMaskUrlRefresher(
  fetchFreshUrls: RefreshMaskUrls,
): MaskUrlRefresher {
  // Caching the promise itself (rather than a boolean + result) is what makes
  // dedupe and one-shot the same mechanism: the first call creates it, every
  // later call — during or after the request — awaits the same object. A
  // rejected promise stays cached, so a failed refresh settles as a rejection
  // for everyone instead of leaving callers pending or triggering a retry.
  let pending: Promise<Map<string, string>> | null = null;
  return {
    refresh() {
      if (pending === null) pending = fetchFreshUrls();
      return pending;
    },
  };
}

/**
 * Pick the URL a row should load: the refreshed one when the refresh produced
 * a usable URL for that segment, otherwise the original. Keeping this a plain
 * string is deliberate — PostItPreview's decode effect stays keyed on
 * `maskUrl` and never learns about segments (see apps/web/AGENTS.md).
 */
export function resolveMaskUrl(
  segmentId: string,
  originalUrl: string,
  fresh: ReadonlyMap<string, string> | null,
): string {
  const refreshed = fresh?.get(segmentId);
  return typeof refreshed === 'string' && refreshed.length > 0
    ? refreshed
    : originalUrl;
}
