import { describe, it, expect } from 'vitest';
import { applyMenuAction, buildSegmentMenuItems } from './segment-menu';

describe('buildSegmentMenuItems', () => {
  it('an unselected, non-excluded segment offers exactly one enabled Select item', () => {
    const items = buildSegmentMenuItems({ selected: false, excluded: false });
    expect(items).toHaveLength(1);
    expect(items[0].action).toBe('select');
    expect(items[0].label).toBe('Select');
    expect(items[0].disabled).toBeFalsy();
  });

  it('a selected, non-excluded segment offers exactly one enabled Deselect item', () => {
    const items = buildSegmentMenuItems({ selected: true, excluded: false });
    expect(items).toHaveLength(1);
    expect(items[0].action).toBe('deselect');
    expect(items[0].label).toBe('Deselect');
    expect(items[0].disabled).toBeFalsy();
  });

  it('an excluded segment offers exactly one disabled "Excluded from data" item, regardless of selection', () => {
    // The disabled item reuses the 'select' action deliberately: it is never
    // dispatched, because the renderer marks it disabled and Radix suppresses
    // the click. An empty menu would read as a bug, and a plain enabled Select
    // would silently fail against the server's excluded guard.
    for (const selected of [true, false]) {
      const items = buildSegmentMenuItems({ selected, excluded: true });
      expect(items).toEqual([
        { action: 'select', label: 'Excluded from data', disabled: true },
      ]);
    }
  });
});

describe('applyMenuAction', () => {
  const base = () => ({ selectedIds: new Set<string>(['a', 'b']) });

  it("applyMenuAction('select') adds the id", () => {
    const result = applyMenuAction(base(), 'select', 'z');
    expect([...result.selectedIds].sort()).toEqual(['a', 'b', 'z']);
  });

  it("applyMenuAction('deselect') removes the id", () => {
    const result = applyMenuAction(base(), 'deselect', 'a');
    expect([...result.selectedIds]).toEqual(['b']);
  });

  it('does not mutate the input set', () => {
    const current = base();
    applyMenuAction(current, 'deselect', 'a');
    expect(current.selectedIds.has('a')).toBe(true);
    applyMenuAction(current, 'select', 'z');
    expect(current.selectedIds.has('z')).toBe(false);
  });
});
