/**
 * The viewer's externally-visible readiness, and the pure rule that derives it.
 *
 * Context: masks load AFTER the base image decodes (PR #104 took 65 mask
 * round-trips off the first-paint path). Painting the base image the moment it
 * decodes produced a ~1.7s "flash of un-masked image" — a raw photo with no
 * overlay, indistinguishable from "segmentation found nothing" in a review
 * tool. So the viewer no longer decides for itself when it is presentable: it
 * reports a status, and the detail page gates the WHOLE card on it.
 *
 * Two product rules are encoded here, and both are deliberate:
 *
 *  1. 'ready' requires the base image AND every mask. A partially-masked image
 *     is never shown, so there is no "some masks are missing" middle state.
 *  2. A mask failure is terminal-but-recoverable: it resolves to 'error' (with
 *     a retry action in the UI), never to 'ready' with a warning.
 *
 * Kept pure and separate from SegmentViewer.tsx for the same reason as
 * mask-loading.ts / view-transform.ts: the viewer itself needs a DOM and a
 * canvas, so the decisions live where they can be tested directly.
 */

export type ViewerStatus = 'loading' | 'ready' | 'error';

export interface ViewerStatusInput {
  /** Base image decoded (PHASE 1). */
  baseLoaded: boolean;
  /** Every mask has settled — succeeded or failed — after any retry (PHASE 2). */
  masksSettled: boolean;
  /** The base image itself failed. Terminal and not retryable in-place. */
  baseError: boolean;
  /** How many masks failed after the one-shot retry. */
  failedMaskCount: number;
}

/**
 * Collapse the viewer's internal load state into the single status the page
 * gates on.
 *
 * Error dominates: once anything has failed, more loading cannot make the
 * image showable, and reporting 'loading' would leave the card spinning
 * forever. Otherwise 'ready' requires BOTH phases complete — `baseLoaded`
 * alone is exactly the flash-of-un-masked-image bug.
 *
 * An image with zero segments has nothing to load in PHASE 2, so the caller
 * sets `masksSettled` immediately and it reaches 'ready' normally; it must
 * never hang waiting for masks that do not exist.
 */
export function resolveViewerStatus({
  baseLoaded,
  masksSettled,
  baseError,
  failedMaskCount,
}: ViewerStatusInput): ViewerStatus {
  if (baseError) return 'error';
  if (failedMaskCount > 0) return 'error';
  if (baseLoaded && masksSettled) return 'ready';
  return 'loading';
}

/**
 * Whether the cached decoded base image can be reused for this effect run.
 *
 * The load effect re-runs for two very different reasons: the image changed
 * (everything must be refetched) or the user hit Retry after a mask failure
 * (PHASE 2 only — refetching a multi-megabyte photo would be pure waste, and
 * keeping the decode means the canvas can paint the instant the masks land).
 * The discriminator is simply whether the cache belongs to the url being
 * loaded; a cache from a different image is never reusable, which is what stops
 * a retry on image B from painting image A.
 */
export function canReuseBaseImage(
  cachedUrl: string | null | undefined,
  imageUrl: string,
): boolean {
  return cachedUrl != null && cachedUrl === imageUrl;
}

/** The mask-failure detail the error state renders. */
export interface MaskError {
  failed: number;
  total: number;
}

/**
 * Human-readable copy for a terminal mask failure.
 *
 * Deliberately does NOT say how many masks succeeded: since a partial result is
 * never displayed, "12 of 65 loaded" would describe something the user cannot
 * see and implies the board is usable when it is not.
 */
export function describeMaskError({ failed, total }: MaskError): string {
  const noun = failed === 1 ? 'segment mask' : 'segment masks';
  return `${failed} of ${total} ${noun} failed to load. The image is hidden until every segment can be shown.`;
}
