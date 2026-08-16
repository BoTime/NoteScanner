import { describe, it, expect } from 'vitest';
import {
  isModifierKey,
  isTypingTarget,
  isZoomedIn,
  panModeForZoomCrossing,
  resolveCursor,
  shouldHonorModifierPan,
  shouldPanOnPress,
  shouldStartPanDrag,
  shouldSuppressContextMenu,
  toggleMode,
  type CursorMode,
  type CursorModeInput,
} from './cursor-mode';

/** Defaults for a loaded, idle, non-drawing viewer in select mode. */
function input(overrides: Partial<CursorModeInput> = {}): CursorModeInput {
  return {
    mode: 'select',
    drawMode: false,
    loaded: true,
    panning: false,
    ...overrides,
  };
}

describe('resolveCursor', () => {
  it('shows wait before the image has loaded, regardless of mode', () => {
    expect(resolveCursor(input({ loaded: false }))).toBe('wait');
    expect(
      resolveCursor(input({ loaded: false, mode: 'pan', panning: true })),
    ).toBe('wait');
  });

  it('shows grabbing while a pan drag is in progress', () => {
    expect(resolveCursor(input({ mode: 'pan', panning: true }))).toBe(
      'grabbing',
    );
  });

  it('shows grab in pan mode', () => {
    expect(resolveCursor(input({ mode: 'pan' }))).toBe('grab');
  });

  it('shows crosshair in select mode', () => {
    expect(resolveCursor(input())).toBe('crosshair');
  });

  it('shows copy in draw mode, but pan still wins over draw', () => {
    expect(resolveCursor(input({ drawMode: true }))).toBe('copy');
    expect(resolveCursor(input({ drawMode: true, mode: 'pan' }))).toBe('grab');
  });
});

describe('shouldPanOnPress', () => {
  it('pans exactly in pan mode', () => {
    expect(shouldPanOnPress('select')).toBe(false);
    expect(shouldPanOnPress('pan')).toBe(true);
  });
});

describe('shouldStartPanDrag', () => {
  const LEFT = 0;
  const MIDDLE = 1;
  const RIGHT = 2;

  it('starts a drag on a primary-button press in pan mode', () => {
    expect(shouldStartPanDrag('pan', LEFT)).toBe(true);
  });

  it('never starts a drag in select mode', () => {
    expect(shouldStartPanDrag('select', LEFT)).toBe(false);
    expect(shouldStartPanDrag('select', RIGHT)).toBe(false);
  });

  // Regression: macOS synthesizes Ctrl+click as button 2. A right press never
  // delivers a matching mouseup, so starting a pan on one strands `panning`
  // true and the image drags with no button held.
  it('ignores a right-button press in pan mode, even though the mode allows panning', () => {
    expect(shouldStartPanDrag('pan', RIGHT)).toBe(false);
  });

  it('ignores the middle button too', () => {
    expect(shouldStartPanDrag('pan', MIDDLE)).toBe(false);
  });
});

describe('shouldSuppressContextMenu', () => {
  // Regression: in pan mode ⌘/Ctrl is a mode key, and Ctrl+click on macOS pops
  // the browser context menu over the image unless it is prevented.
  it('suppresses the browser menu in pan mode', () => {
    expect(shouldSuppressContextMenu('pan')).toBe(true);
  });

  it('leaves the browser menu alone in select mode', () => {
    expect(shouldSuppressContextMenu('select')).toBe(false);
  });
});

describe('toggleMode', () => {
  it('flips between select and pan', () => {
    expect(toggleMode('select')).toBe('pan');
    expect(toggleMode('pan')).toBe('select');
  });

  it('round-trips', () => {
    const start: CursorMode = 'select';
    expect(toggleMode(toggleMode(start))).toBe(start);
  });
});

describe('isModifierKey', () => {
  it('recognizes Meta and Control by KeyboardEvent.key', () => {
    expect(isModifierKey('Meta')).toBe(true);
    expect(isModifierKey('Control')).toBe(true);
  });

  it('rejects other keys', () => {
    expect(isModifierKey('Shift')).toBe(false);
    expect(isModifierKey('h')).toBe(false);
    expect(isModifierKey('Escape')).toBe(false);
  });
});

describe('isTypingTarget', () => {
  it('treats inputs, textareas, selects and contenteditable as typing targets', () => {
    expect(isTypingTarget('INPUT', false)).toBe(true);
    expect(isTypingTarget('TEXTAREA', false)).toBe(true);
    expect(isTypingTarget('SELECT', false)).toBe(true);
    expect(isTypingTarget('DIV', true)).toBe(true);
  });

  it('is case-insensitive on the tag name', () => {
    expect(isTypingTarget('input', false)).toBe(true);
  });

  it('treats other elements and a missing target as not typing', () => {
    expect(isTypingTarget('CANVAS', false)).toBe(false);
    expect(isTypingTarget('BUTTON', false)).toBe(false);
    expect(isTypingTarget(null, false)).toBe(false);
    expect(isTypingTarget(undefined, false)).toBe(false);
  });
});

describe('isZoomedIn', () => {
  it('is false at exactly the minimum (fit) scale', () => {
    expect(isZoomedIn(1)).toBe(false);
  });

  it('is true just above the minimum scale', () => {
    expect(isZoomedIn(1.01)).toBe(true);
    expect(isZoomedIn(2)).toBe(true);
    expect(isZoomedIn(5)).toBe(true);
  });
});

describe('shouldHonorModifierPan', () => {
  it('honors the ⌘/Ctrl hold when not drawing', () => {
    expect(shouldHonorModifierPan(false)).toBe(true);
  });

  it('ignores the ⌘/Ctrl hold while drawing', () => {
    expect(shouldHonorModifierPan(true)).toBe(false);
  });
});

describe('panModeForZoomCrossing', () => {
  it('enables pan when zoom crosses fit → past fit', () => {
    expect(panModeForZoomCrossing(1, 2, false)).toBe('pan');
  });

  it('does not enable pan on the same crossing while drawing', () => {
    expect(panModeForZoomCrossing(1, 2, true)).toBeNull();
  });

  it('disables pan when zoom crosses past fit → fit', () => {
    expect(panModeForZoomCrossing(2, 1, false)).toBe('select');
  });

  it('disables pan on zoom-out to fit even while drawing', () => {
    expect(panModeForZoomCrossing(2, 1, true)).toBe('select');
  });

  it('has no opinion on non-crossing zoom steps in either direction', () => {
    expect(panModeForZoomCrossing(2, 3, false)).toBeNull();
    expect(panModeForZoomCrossing(3, 2, false)).toBeNull();
    expect(panModeForZoomCrossing(2, 3, true)).toBeNull();
    expect(panModeForZoomCrossing(3, 2, true)).toBeNull();
  });

  it('has no opinion when the scale is unchanged', () => {
    // The effect re-runs whenever drawMode flips, which invokes this with
    // prevScale === nextScale. Returning anything but null there would clobber
    // a manual override the moment the user toggles draw mode.
    expect(panModeForZoomCrossing(1, 1, false)).toBeNull();
    expect(panModeForZoomCrossing(2, 2, false)).toBeNull();
    expect(panModeForZoomCrossing(2, 2, true)).toBeNull();
  });

  it('lets a manual override survive further zooming at the same level', () => {
    // User zoomed to 2 (pan auto-enabled), then manually turned the hand off.
    // Zooming further to 3 must not re-enable it: no boundary is crossed, so
    // the function returns null and the caller never calls setPanMode.
    expect(panModeForZoomCrossing(2, 3, false)).toBeNull();
    expect(panModeForZoomCrossing(3, 4, false)).toBeNull();
  });
});
