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
