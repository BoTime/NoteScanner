/**
 * Pure logic for the segment click-menu.
 *
 * The `SegmentViewer` popup offers exactly one per-segment action: the
 * Select/Deselect toggle. This module isolates the item-decision and
 * state-transition logic so it can be unit-tested without React (the web Vitest
 * env is `node` — no DOM). `SegmentViewer` renders `buildSegmentMenuItems` and
 * dispatches on the returned actions.
 *
 * Semantics:
 * - Select/Deselect is the only live action. Deselecting a segment skips it for
 *   transcription and drops it from the review panel, the progress numerator,
 *   and the workflow-stage eligible set.
 * - A segment persisted as `excluded` is inert in the UI: the server refuses to
 *   re-select it, so the menu shows a single DISABLED item explaining the state
 *   rather than an empty menu (which reads as a bug) or an enabled Select
 *   (which would silently fail to persist). Reversing an exclusion is not a UI
 *   affordance; it requires a direct API call.
 */

export type SegmentMenuAction = 'select' | 'deselect';

export interface SegmentMenuState {
  selected: boolean;
  excluded: boolean;
}

export interface SegmentMenuItem {
  action: SegmentMenuAction;
  label: string;
  /** Rendered as a Radix disabled item: styled dim and never dispatched. */
  disabled?: boolean;
}

/**
 * Decide which menu items to show for a single segment given its current state.
 * Always exactly one item:
 * - Excluded: a disabled "Excluded from data" notice. Its `'select'` action is
 *   never dispatched — the renderer marks the item disabled and Radix
 *   suppresses `onSelect`.
 * - Otherwise: the select/deselect toggle, keyed on `selected`.
 */
export function buildSegmentMenuItems(
  state: SegmentMenuState,
): SegmentMenuItem[] {
  if (state.excluded) {
    return [{ action: 'select', label: 'Excluded from data', disabled: true }];
  }
  return [
    state.selected
      ? { action: 'deselect', label: 'Deselect' }
      : { action: 'select', label: 'Select' },
  ];
}

/** The client-only state a menu action operates on. */
export interface SegmentClientState {
  selectedIds: ReadonlySet<string>;
}

/** The next client-only selection set after applying a menu action. */
export interface SegmentActionResult {
  selectedIds: Set<string>;
}

function withoutId(set: ReadonlySet<string>, id: string): Set<string> {
  const next = new Set(set);
  next.delete(id);
  return next;
}

function withId(set: ReadonlySet<string>, id: string): Set<string> {
  const next = new Set(set);
  next.add(id);
  return next;
}

/**
 * Compute the next client state for a menu action on `segmentId`. Pure —
 * returns a fresh set; the component applies it.
 */
export function applyMenuAction(
  current: SegmentClientState,
  action: SegmentMenuAction,
  segmentId: string,
): SegmentActionResult {
  switch (action) {
    case 'select':
      return { selectedIds: withId(current.selectedIds, segmentId) };
    case 'deselect':
      return { selectedIds: withoutId(current.selectedIds, segmentId) };
  }
}
