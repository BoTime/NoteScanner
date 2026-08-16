/**
 * Pure policy helpers for the two-phase mask load in SegmentViewer.
 *
 * The viewer itself cannot be unit-tested (no jsdom, no canvas in the Vitest
 * `node` environment), so the parts with real decisions in them — how to read
 * the mask signature, what a failure means, and whether to retry — live here
 * with direct tests. Mirrors view-transform.ts / segment-viewer-logic.ts.
 */

/** The only two fields the mask pipeline cares about. */
export interface MaskRef {
  id: string;
  maskUrl: string;
}

/**
 * Inverse of the `maskSignature` memo in SegmentViewer:
 * `segments.map(s => `${s.id}|${s.maskUrl}`).join('\n')`.
 *
 * Deriving the mask list from the signature — rather than closing over the
 * `segments` prop — is what makes the load effect provably independent of the
 * full array, per the hard constraint in apps/web/AGENTS.md. Splits on the
 * FIRST pipe only, so presigned query strings survive intact.
 */
export function parseMaskSignature(signature: string): MaskRef[] {
  if (signature === '') return [];
  return signature.split('\n').map((line) => {
    const sep = line.indexOf('|');
    if (sep < 0) return { id: line, maskUrl: '' };
    return { id: line.slice(0, sep), maskUrl: line.slice(sep + 1) };
  });
}

export type MaskFailureKind = 'expired' | 'other';

/**
 * Classify a mask load failure.
 *
 * `HTMLImageElement.onerror` gives no status code, so the status cannot be
 * read directly. The discriminator that IS available is the URL shape: an
 * absolute presigned URL is time-boxed (300s TTL) and can 403 on a page left
 * open, whereas the same-origin proxy path is cookie-authed and never expires
 * in that way. So an absolute (non-same-origin-path) mask URL that fails is
 * treated as possibly-expired and is worth exactly one refetch-and-retry;
 * a proxy-path failure is not.
 */
export function classifyMaskFailure(
  url: string,
  _error: unknown,
): MaskFailureKind {
  return url.startsWith('/') ? 'other' : 'expired';
}

export interface RetryDecision {
  shouldRefetch: boolean;
  reason: string;
}

/**
 * Decide whether to refetch the detail DTO for fresh presigned URLs and retry
 * the failed masks. Deliberately ONE shot — no background refresh timer, no
 * loop. If the retry also fails, the error surfaces and the board stays usable
 * with whatever masks did load.
 */
export function planMaskRetry({
  failures,
  alreadyRetried,
}: {
  failures: ReadonlyArray<{ id: string; kind: MaskFailureKind }>;
  alreadyRetried: boolean;
}): RetryDecision {
  if (failures.length === 0) {
    return { shouldRefetch: false, reason: 'no failures' };
  }
  if (alreadyRetried) {
    return { shouldRefetch: false, reason: 'already retried once' };
  }
  if (!failures.some((f) => f.kind === 'expired')) {
    return { shouldRefetch: false, reason: 'no expiry-shaped failures' };
  }
  return { shouldRefetch: true, reason: 'expired presigned url' };
}

/**
 * Why a post-it preview tile has no canvas to show.
 *
 * - `error`   — the photo or mask genuinely failed to load/decode. Something IS
 *               broken and the user should be told so.
 * - `no-mask` — everything loaded fine, the mask simply covers no pixels. This
 *               is the expected, correct shape of a Text-tab "Add note", whose
 *               mask is a blank 1x1 transparent PNG stand-in (there is no crop
 *               region to preview). Nothing is broken.
 */
export type PreviewEmptyReason = 'error' | 'no-mask';

export interface PreviewEmptyState {
  reason: PreviewEmptyReason;
  /** Copy for the tile. Never implies breakage for a text-only note. */
  label: string;
}

/**
 * Map a finished-but-empty preview attempt onto the message the tile shows.
 *
 * Split out as a pure function because apps/web runs Vitest in the `node`
 * environment — PostItPreview itself (canvas, Image) cannot be rendered in a
 * test, so the decision this encodes is only directly assertable here.
 *
 * A blank mask that decoded cleanly is NOT a failure: it is how a text-only
 * note is stored. Reporting it as "Preview unavailable" made the flagship Add
 * Note flow read as broken on the very list the note was created in.
 */
export function describePreviewEmptyState(
  reason: PreviewEmptyReason,
): PreviewEmptyState {
  if (reason === 'no-mask') {
    return { reason, label: 'Text-only note' };
  }
  return { reason, label: 'Preview unavailable' };
}

/**
 * Should a preview miss trigger the panel's one-shot mask-url refresh?
 *
 * Only a real load/decode error can be an expired presigned URL. A cleanly
 * decoded blank mask will decode to blank again after any refresh, so
 * refetching for it is pure waste (and, at 35 rows, a needless request storm).
 */
export function shouldReportPreviewFailure(reason: PreviewEmptyReason): boolean {
  return reason === 'error';
}

export interface MaskLoadOutcome {
  id: string;
  ok: boolean;
  kind?: MaskFailureKind;
}

/**
 * Accumulates per-mask load outcomes so one failure cannot void the rest.
 *
 * The old load path was a single `Promise.all` over every mask: one rejection
 * discarded every successfully decoded mask and put the viewer into its
 * terminal "Failed to load images" state. Masks now load independently, so
 * successes are kept and only the failures are reported — the board stays
 * usable with whatever loaded.
 *
 * Outcomes for unexpected ids are ignored, and only the first outcome per id
 * counts, so a late callback from a superseded load cannot corrupt the state.
 */
export class MaskLoadTracker {
  private readonly expected: Set<string>;
  private readonly succeeded: string[] = [];
  private readonly failed: Array<{ id: string; kind: MaskFailureKind }> = [];
  private readonly settled = new Set<string>();

  constructor(expectedIds: ReadonlyArray<string>) {
    this.expected = new Set(expectedIds);
  }

  private accept(id: string): boolean {
    if (!this.expected.has(id) || this.settled.has(id)) return false;
    this.settled.add(id);
    return true;
  }

  succeed(id: string): void {
    if (this.accept(id)) this.succeeded.push(id);
  }

  fail(id: string, kind: MaskFailureKind): void {
    if (this.accept(id)) this.failed.push({ id, kind });
  }

  get failures(): ReadonlyArray<{ id: string; kind: MaskFailureKind }> {
    return this.failed;
  }

  get succeededIds(): string[] {
    return [...this.succeeded];
  }

  get settledCount(): number {
    return this.settled.size;
  }

  get isComplete(): boolean {
    return this.settled.size === this.expected.size;
  }
}

/**
 * Build the key that identifies "the same image" for `planMaskDiff`.
 *
 * Joined with `|` rather than bare concatenation: without a separator,
 * `('a', 1, 23)` and `('a', 12, 3)` would produce the identical string
 * `'a123'`, silently reusing masks across two genuinely different images.
 *
 * Width and height are part of the key — not just the url — because a mask's
 * `coverage` array is sized to `imageWidth * imageHeight`. Reusing a decoded
 * mask across a dimension change is a correctness bug (indices would read out
 * of bounds or misalign), not merely a missed optimization.
 */
export function buildImageKey(args: {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
}): string {
  return `${args.imageUrl}|${args.imageWidth}|${args.imageHeight}`;
}

/**
 * The result of diffing the previous and next mask ref lists: what the
 * mask-load effect must do this run.
 */
export interface MaskDiffPlan {
  /** True when the image itself changed (or this is the first run) — discard
   * every decoded mask and derived cache and reload from scratch. */
  fullReset: boolean;
  /** Ids whose decoded mask is still valid and must be reused AS THE SAME
   * OBJECT REFERENCE — never rebuilt. */
  keep: string[];
  /** Refs to fetch and decode on this run. */
  load: MaskRef[];
  /** Ids to delete from `masksRef` AND the renderer's per-id caches. */
  evict: string[];
}

/**
 * Decide, from the previous/next image keys and previous/next mask ref
 * lists, whether the mask-load effect should do a full reset or an
 * incremental diff — and if incremental, exactly which ids to keep, load,
 * and evict.
 *
 * Diffs at the `id + maskUrl` pair granularity, which mirrors the
 * `maskSignature` memo's own granularity exactly (see parseMaskSignature's
 * doc comment): an id whose `maskUrl` is unchanged is kept as-is; an id
 * whose `maskUrl` changed lands in BOTH `evict` and `load`, because its
 * previously decoded geometry cannot survive the url change (a different
 * url may mean genuinely different pixel coverage); an id absent from the
 * previous refs is a pure add and only needs `load`; an id absent from the
 * next refs is a pure removal and only needs `evict`.
 */
export function planMaskDiff(args: {
  prevImageKey: string | null;
  nextImageKey: string;
  prevRefs: ReadonlyArray<MaskRef>;
  nextRefs: ReadonlyArray<MaskRef>;
}): MaskDiffPlan {
  const { prevImageKey, nextImageKey, prevRefs, nextRefs } = args;

  if (prevImageKey === null || prevImageKey !== nextImageKey) {
    return { fullReset: true, keep: [], load: [...nextRefs], evict: [] };
  }

  const prevByUrl = new Map(prevRefs.map((r) => [r.id, r.maskUrl]));
  const keep: string[] = [];
  const load: MaskRef[] = [];
  const evict: string[] = [];

  for (const next of nextRefs) {
    const prevUrl = prevByUrl.get(next.id);
    if (prevUrl === undefined) {
      // New id — pure add.
      load.push(next);
    } else if (prevUrl === next.maskUrl) {
      // Unchanged — reuse the existing decoded mask as-is.
      keep.push(next.id);
    } else {
      // Same id, different url — cached geometry cannot survive the url
      // change, so it must be evicted AND reloaded.
      evict.push(next.id);
      load.push(next);
    }
  }

  const nextIds = new Set(nextRefs.map((r) => r.id));
  for (const prev of prevRefs) {
    if (!nextIds.has(prev.id)) {
      // Present before, gone now — pure removal.
      evict.push(prev.id);
    }
  }

  return { fullReset: false, keep, load, evict };
}

/**
 * Delete one id from an arbitrary list of caches.
 *
 * Used to evict per-mask derived geometry (e.g. the renderer's coverage /
 * edge index caches) whenever `planMaskDiff` reports an id in `evict` —
 * those caches are keyed by id alone and are consulted before `scene.masks`,
 * so a stale entry would silently paint old geometry against a new mask.
 * A no-op for an id that is not present in a given cache.
 */
export function evictMaskCaches(
  id: string,
  caches: ReadonlyArray<Map<string, unknown>>,
): void {
  for (const c of caches) c.delete(id);
}
