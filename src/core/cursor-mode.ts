/**
 * Pure mode/cursor resolution for the SegmentViewer. No React, no DOM — every
 * function takes plain values so it can be unit-tested in the node Vitest
 * environment, mirroring the view-transform.ts / segment-viewer-logic.ts split.
 *
 * The viewer has a single LATCHED mode: select (crosshair) or pan (hand). It is
 * the sole source of truth for the cursor and for what a drag does. Three
 * controls flip it, all with identical latch semantics: the toggle button, the
 * `H` shortcut, and the ⌘/Ctrl key.
 *
 * Deriving the cursor from a latched mode rather than from live key state is
 * what fixes the stuck-hand bug: macOS suppresses keyup for keys pressed while
 * Command is held, so any design that reads "is the modifier down right now"
 * can strand the cursor when that keyup never arrives.
 */

import { MIN_SCALE } from './view-transform';

/** The two viewer interaction modes. */
export type CursorMode = 'select' | 'pan';

/** Everything the cursor depends on. */
export interface CursorModeInput {
  /** The user's explicit, sticky choice — the only mode input. */
  mode: CursorMode;
  /** Polygon-draw mode is active. */
  drawMode: boolean;
  /** The base image and masks have finished decoding. */
  loaded: boolean;
  /** A pan drag is in progress right now. */
  panning: boolean;
}

/**
 * The CSS `cursor` value for the canvas. Precedence, highest first: not loaded
 * (wait), actively dragging (grabbing), pan mode (grab), draw mode (copy), else
 * select (crosshair). Pan outranks draw, matching the pre-existing behavior.
 */
export function resolveCursor(input: CursorModeInput): string {
  if (!input.loaded) return 'wait';
  if (input.panning) return 'grabbing';
  if (input.mode === 'pan') return 'grab';
  if (input.drawMode) return 'copy';
  return 'crosshair';
}

/**
 * Whether a mouse press should start a pan drag (rather than fall through to
 * selection / point placement). True exactly in pan mode.
 */
export function shouldPanOnPress(mode: CursorMode): boolean {
  return mode === 'pan';
}

/** MouseEvent.button value for the primary (left) button. */
const PRIMARY_BUTTON = 0;

/**
 * Whether a mouse press should begin a pan drag, given the mode AND which
 * button was pressed.
 *
 * The button check is load-bearing on macOS, where Ctrl+click is synthesized as
 * a RIGHT click (button 2). A right press never delivers a matching mouseup to
 * the element, so starting a pan on one strands the drag: `panning` stays true
 * and the image follows the pointer with no button held.
 */
export function shouldStartPanDrag(mode: CursorMode, button: number): boolean {
  return button === PRIMARY_BUTTON && shouldPanOnPress(mode);
}

/**
 * Whether to suppress the browser's context menu over the canvas. Only in pan
 * mode, where ⌘/Ctrl is a mode key and macOS would otherwise turn Ctrl+click
 * into a right click and pop a menu over the image. In select mode a right
 * click still behaves normally.
 */
export function shouldSuppressContextMenu(mode: CursorMode): boolean {
  return shouldPanOnPress(mode);
}

/** Flip the latched mode. */
export function toggleMode(mode: CursorMode): CursorMode {
  return mode === 'select' ? 'pan' : 'select';
}

/**
 * The single definition of "zoomed in" for mode decisions, so the threshold
 * never drifts from the transform layer's floor. `scale` is relative to
 * fit-to-frame, so MIN_SCALE (1) IS fit — at that scale `clampTransform`
 * zeroes the offsets and the image cannot be panned at all.
 */
export function isZoomedIn(scale: number): boolean {
  return scale > MIN_SCALE;
}

/**
 * Whether the ⌘/Ctrl mode-key hold should be honoured at all. It is fully inert
 * while drawing: a modifier press must not flip the mode, move the cursor, or
 * disturb the in-progress polygon.
 */
export function shouldHonorModifierPan(drawMode: boolean): boolean {
  return !drawMode;
}

/**
 * Which mode a zoom change should produce, or `null` for "no opinion — leave
 * the mode exactly as the user left it".
 *
 * Fires only on CROSSINGS of the fit boundary, never on every zoom step. That
 * is what makes a manual hand toggle stick: zooming 2x → 3x crosses nothing, so
 * this returns null, the caller skips `setPanMode`, and the user's choice
 * survives.
 *
 * The rule is asymmetric on purpose. Draw mode suppresses auto-ENABLE (zooming
 * in to place points precisely is normal drawing), but never suppresses
 * auto-DISABLE: at fit the image is unpannable, so a latched pan mode would
 * strand the user with a hand cursor over an image that cannot move.
 */
export function panModeForZoomCrossing(
  prevScale: number,
  nextScale: number,
  drawMode: boolean,
): CursorMode | null {
  const wasZoomed = isZoomedIn(prevScale);
  const isZoomed = isZoomedIn(nextScale);
  if (wasZoomed === isZoomed) return null;
  if (isZoomed) return drawMode ? null : 'pan';
  return 'select';
}

/**
 * Whether a KeyboardEvent.key names the mode-toggle modifier. Matched on `key`
 * rather than on the e.metaKey/e.ctrlKey flags, which are unreliable for the
 * very key being pressed or released.
 */
export function isModifierKey(key: string): boolean {
  return key === 'Meta' || key === 'Control';
}

/**
 * Whether keyboard focus is in a text-entry context, in which case the mode
 * shortcuts must not fire. Takes the raw tagName + contentEditable flag rather
 * than an Element so it stays DOM-free and testable.
 */
export function isTypingTarget(
  tagName: string | null | undefined,
  isContentEditable: boolean,
): boolean {
  if (isContentEditable) return true;
  if (!tagName) return false;
  const tag = tagName.toUpperCase();
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}
