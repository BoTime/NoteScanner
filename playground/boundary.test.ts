import { describe, it, expect } from 'vitest';
import {
  BOUNDARY_FIXTURES,
  BOUNDARY_GEOMETRY,
  differingPixels,
  runBoundaryPath,
  type BoundaryFixture,
} from './boundary';

function fixture(id: string): BoundaryFixture {
  const found = BOUNDARY_FIXTURES.find((candidate) => candidate.id === id);
  if (!found) throw new Error(`no fixture ${id}`);
  return found;
}

describe('the boundary fixtures', () => {
  it('gates at 38 low-res pixels, from a real 512x333 geometry', () => {
    // 100 * 65536 / 170496 = 38.438 -> 38.
    expect(runBoundaryPath(fixture('speck').windows, true).minArea).toBe(38);
    expect(runBoundaryPath(fixture('speck').windows, false).minArea).toBe(100);
  });

  it('keeps every mask both paths keep byte-for-byte identical (AC2)', () => {
    let keptByBoth = 0;
    for (const item of BOUNDARY_FIXTURES) {
      const baseline = runBoundaryPath(item.windows, false);
      const next = runBoundaryPath(item.windows, true);
      item.windows.forEach((_, index) => {
        const a = baseline.outcomes[index];
        const b = next.outcomes[index];
        if (a.status !== 'kept' || b.status !== 'kept') return;
        keptByBoth += 1;
        expect(differingPixels(a.mask!, b.mask!), `${item.id}[${index}]`).toBe(0);
      });
    }
    // Not vacuous: something has to be kept by both, or the loop above proves
    // nothing at all.
    expect(keptByBoth).toBe(4);
  });

  it('keeps the fine-toothed comb on both paths (AC2)', () => {
    const windows = fixture('comb').windows;
    const baseline = runBoundaryPath(windows, false).outcomes[0];
    const next = runBoundaryPath(windows, true).outcomes[0];
    expect(baseline.status).toBe('kept');
    expect(next.status).toBe('kept');
    // 4016 low-res pixels become 16064 at full resolution — the exact 4x the
    // 2x-per-axis mapping predicts.
    expect(next.gateArea).toBe(4016);
    expect(baseline.gateArea).toBe(16064);
    expect(next.mask!.area).toBe(16064);
  });

  it('shows the speck the scaled gate drops and the baseline returns (F3)', () => {
    const windows = fixture('speck').windows;
    const baseline = runBoundaryPath(windows, false).outcomes[0];
    const next = runBoundaryPath(windows, true).outcomes[0];
    expect(next.gateArea).toBe(36);
    expect(next.status).toBe('filtered');
    expect(baseline.gateArea).toBe(144);
    expect(baseline.status).toBe('kept');
    // The whole point: its true area clears minMaskArea 100 with room to spare.
    expect(baseline.gateArea).toBeGreaterThan(100);
  });

  it('suppresses the 0.7094 duplicate and keeps the 0.6949 one, on both paths (N1)', () => {
    const windows = fixture('pair').windows;
    for (const lowRes of [false, true]) {
      const statuses = runBoundaryPath(windows, lowRes).outcomes.map((o) => o.status);
      expect(statuses, `lowResFilterNms=${lowRes}`).toEqual(['kept', 'suppressed', 'kept']);
    }
  });

  it('keeps the one-pixel bridge on both paths', () => {
    const windows = fixture('bridge').windows;
    expect(runBoundaryPath(windows, false).outcomes[0].status).toBe('kept');
    expect(runBoundaryPath(windows, true).outcomes[0].status).toBe('kept');
  });

  it('renders masks at the image size, not the grid size', () => {
    const kept = runBoundaryPath(fixture('comb').windows, true).outcomes[0].mask!;
    expect(kept.coverage.length).toBe(
      BOUNDARY_GEOMETRY.originalWidth * BOUNDARY_GEOMETRY.originalHeight,
    );
  });
});
