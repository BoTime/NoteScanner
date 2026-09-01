import { describe, it, expect } from 'vitest';
import {
  candidateCoverage,
  createFilterPlan,
  dedupeCandidates,
  lowResMinArea,
  releaseCandidate,
  releaseRejected,
  resolveSurvivor,
  retainCandidate,
  type FilterNmsOptions,
  type MaskCandidate,
  type MaskGeometry,
} from './mask-pipeline';
import { thresholdMask } from './mask-postprocess';
import { resampleThresholdMask } from './mask-resample';
import { dedupeMasks } from './nms';

/**
 * A 16x16 logit grid over a 16x16 pad whose top-left 16x12 maps to a 32x24
 * image, so full resolution is EXACTLY 2x the low grid on both axes and a low
 * box covering columns a..b covers full columns 2a..2b+1. Every expected
 * number in this file was derived from that rule and then confirmed by
 * running the file — none of them is a guess.
 */
const GEOMETRY: MaskGeometry = {
  lowWidth: 16,
  lowHeight: 16,
  padWidth: 16,
  padHeight: 16,
  reshapedWidth: 16,
  reshapedHeight: 12,
  originalWidth: 32,
  originalHeight: 24,
};

const OPTIONS: FilterNmsOptions = {
  maskThreshold: 0,
  minMaskArea: 100,
  nmsIouThreshold: 0.7,
  lowResFilterNms: true,
};

function lowPlan(overrides: Partial<FilterNmsOptions> = {}) {
  return createFilterPlan(GEOMETRY, { ...OPTIONS, lowResFilterNms: true, ...overrides });
}

function fullPlan(overrides: Partial<FilterNmsOptions> = {}) {
  return createFilterPlan(GEOMETRY, { ...OPTIONS, lowResFilterNms: false, ...overrides });
}

/** +1 inside any listed inclusive box, -1 outside. */
function makeWindow(...boxes: [number, number, number, number][]): Float32Array {
  const data = new Float32Array(GEOMETRY.lowWidth * GEOMETRY.lowHeight).fill(-1);
  for (const [x0, x1, y0, y1] of boxes) {
    for (let y = y0; y <= y1; y += 1) {
      for (let x = x0; x <= x1; x += 1) data[y * GEOMETRY.lowWidth + x] = 1;
    }
  }
  return data;
}

/**
 * Six windows chosen so both drops are exercised: B duplicates A (IoU 0.818),
 * C is just below the threshold against A (0.667), D and E are specks that
 * fail both area gates, and F is a smaller mask that overlaps nothing enough
 * to be suppressed.
 */
const WINDOWS = {
  A: makeWindow([2, 11, 2, 9]),
  B: makeWindow([3, 12, 2, 9]),
  C: makeWindow([4, 13, 2, 9]),
  D: makeWindow([14, 14, 10, 10]),
  E: makeWindow([2, 2, 11, 11]),
  F: makeWindow([9, 14, 4, 9]),
};
const ORDER = ['A', 'B', 'C', 'D', 'E', 'F'] as const;
const ALL = ORDER.map((name) => WINDOWS[name]);

describe('lowResMinArea', () => {
  it('scales minMaskArea by the low-grid-to-image pixel ratio (AC7)', () => {
    // The 1024x649 playground sample: 100 * 65536 / 664576 = 9.861 -> 10.
    expect(lowResMinArea(100, 65536, 1024 * 649)).toBe(10);
  });

  it('rounds half up rather than truncating', () => {
    // 75 * 65536 / 131072 is exactly 37.5.
    expect(lowResMinArea(75, 65536, 131072)).toBe(38);
  });

  it('floors to 1 when the ratio rounds below one (AC7)', () => {
    // A 12 MP photo: 100 * 65536 / 12000000 = 0.546 -> 0 -> floored to 1.
    expect(lowResMinArea(100, 65536, 12_000_000)).toBe(1);
    expect(lowResMinArea(1, 65536, 12_000_000)).toBe(1);
  });

  it('still floors to 1 for minMaskArea 0, which is the one place the gates differ in kind', () => {
    // Documented consequence: with `minMaskArea: 0` the low-res path drops
    // zero-area masks that the full-res path would keep. Nothing downstream
    // wants an empty mask, but the asymmetry is real and pinned here.
    expect(lowResMinArea(0, 65536, 12_000_000)).toBe(1);
  });
});

describe('createFilterPlan', () => {
  it('gates and dedupes at 256x256 units on the low-res path', () => {
    const plan = lowPlan();
    // 100 * 256 / 768 = 33.33 -> 33.
    expect(plan.minArea).toBe(33);
    expect(plan.nmsWidth).toBe(GEOMETRY.lowWidth);
  });

  it('gates and dedupes at full resolution on the baseline path', () => {
    const plan = fullPlan();
    expect(plan.minArea).toBe(OPTIONS.minMaskArea);
    expect(plan.nmsWidth).toBe(GEOMETRY.originalWidth);
  });
});

describe('retainCandidate', () => {
  it('retains a COPY of the logit window, not a view into it (AC6)', () => {
    const plan = lowPlan();
    const source = Float32Array.from(WINDOWS.A);
    const retained = retainCandidate(source, plan);
    expect(retained.candidate).not.toBeNull();
    const expected = resolveSurvivor(retained.candidate!, plan);

    // Exactly what the worker does next: the decoder reuses `pred_masks`.
    source.fill(-999);

    expect(retained.candidate!.logits![0]).toBe(WINDOWS.A[0]);
    const after = resolveSurvivor(retained.candidate!, plan);
    expect(after).not.toBeNull();
    expect(after!.area).toBe(expected!.area);
    expect(Array.from(after!.coverage)).toEqual(Array.from(expected!.coverage));
  });

  it('throws when the window does not match the low grid', () => {
    expect(() => retainCandidate(new Float32Array(255), lowPlan())).toThrow(/256/);
  });

  it('reports the gate area in the units the path gates at', () => {
    expect(retainCandidate(WINDOWS.A, lowPlan()).gateArea).toBe(80);
    expect(retainCandidate(WINDOWS.A, fullPlan()).gateArea).toBe(320);
    expect(retainCandidate(WINDOWS.F, lowPlan()).gateArea).toBe(36);
    expect(retainCandidate(WINDOWS.F, fullPlan()).gateArea).toBe(144);
  });

  it('drops a speck under the pre-NMS gate on both paths', () => {
    // D is a single low-res pixel: 1 < 33 at low res, 4 < 100 at full res.
    expect(retainCandidate(WINDOWS.D, lowPlan()).candidate).toBeNull();
    expect(retainCandidate(WINDOWS.D, lowPlan()).gateArea).toBe(1);
    expect(retainCandidate(WINDOWS.D, fullPlan()).candidate).toBeNull();
    expect(retainCandidate(WINDOWS.D, fullPlan()).gateArea).toBe(4);
  });
});

describe('dedupeCandidates', () => {
  it('suppresses the duplicate at both resolutions, keeping the same set', () => {
    for (const plan of [lowPlan(), fullPlan()]) {
      const candidates = ALL.map((w) => retainCandidate(w, plan).candidate).filter(
        (candidate): candidate is MaskCandidate => candidate !== null,
      );
      // D and E already fell out at the gate, so the live set is A, B, C, F.
      expect(candidates).toHaveLength(4);
      // B duplicates A at IoU 0.818; C sits at 0.667 and survives.
      expect(dedupeCandidates(candidates, plan)).toEqual([0, 2, 3]);
    }
  });

  it('refuses to dedupe a released candidate rather than reading null', () => {
    const plan = lowPlan();
    const candidate = retainCandidate(WINDOWS.A, plan).candidate!;
    releaseCandidate(candidate);
    expect(() => candidateCoverage([candidate])).toThrow(/released/);
  });
});

describe('releaseRejected', () => {
  it('releases exactly the candidates NMS did not keep (AC9)', () => {
    const plan = lowPlan();
    const candidates = ALL.map((w) => retainCandidate(w, plan).candidate).filter(
      (candidate): candidate is MaskCandidate => candidate !== null,
    );
    const kept = dedupeCandidates(candidates, plan);
    expect(releaseRejected(candidates, kept)).toBe(1);
    expect(candidates[1].logits).toBeNull();
    expect(candidates[1].coverage).toBeNull();
    for (const index of kept) {
      expect(candidates[index].logits).not.toBeNull();
      expect(candidates[index].coverage).not.toBeNull();
    }
  });
});

describe('resolveSurvivor', () => {
  it('throws rather than returning a wrong mask for a released candidate', () => {
    const plan = lowPlan();
    const candidate = retainCandidate(WINDOWS.A, plan).candidate!;
    releaseCandidate(candidate);
    expect(() => resolveSurvivor(candidate, plan)).toThrow(/released/);
  });

  it('re-checks the exact, unscaled minMaskArea at full resolution (AC8)', () => {
    // A geometry where full resolution is 4x the low grid on both axes, so
    // the scaled pre-NMS gate is LOOSER than minMaskArea and a mask can pass
    // it and still fail the real one.
    const geometry: MaskGeometry = {
      lowWidth: 4,
      lowHeight: 4,
      padWidth: 4,
      padHeight: 4,
      reshapedWidth: 4,
      reshapedHeight: 4,
      originalWidth: 16,
      originalHeight: 16,
    };
    const window = new Float32Array(16).fill(-1);
    for (const p of [5, 6, 9, 10]) window[p] = 1;
    // Measured, not assumed: 4 low-res pixels become 60 full-res ones here.
    const fullArea = resampleThresholdMask({ logits: window, ...geometry, threshold: 0 }).area;
    const lowArea = thresholdMask(window, 0).area;

    const tooBig = createFilterPlan(geometry, { ...OPTIONS, minMaskArea: fullArea + 1 });
    // The candidate must genuinely PASS the pre-NMS gate, or this proves nothing.
    expect(tooBig.minArea).toBeLessThanOrEqual(lowArea);
    const retained = retainCandidate(window, tooBig);
    expect(retained.candidate).not.toBeNull();
    expect(resolveSurvivor(retained.candidate!, tooBig)).toBeNull();

    const exact = createFilterPlan(geometry, { ...OPTIONS, minMaskArea: fullArea });
    const kept = resolveSurvivor(retainCandidate(window, exact).candidate!, exact);
    expect(kept).not.toBeNull();
    expect(kept!.area).toBe(fullArea);
  });

  it('gives a mask kept by both paths byte-identical coverage (AC2 mechanism)', () => {
    for (const name of ['A', 'C', 'F'] as const) {
      const low = resolveSurvivor(retainCandidate(WINDOWS[name], lowPlan()).candidate!, lowPlan());
      const full = resolveSurvivor(retainCandidate(WINDOWS[name], fullPlan()).candidate!, fullPlan());
      expect(low).not.toBeNull();
      expect(full).not.toBeNull();
      expect(Array.from(low!.coverage)).toEqual(Array.from(full!.coverage));
    }
  });
});

describe('the baseline path', () => {
  it('reproduces the pre-change pipeline exactly (AC5)', () => {
    // The pipeline as it stood before this change, written out here because a
    // differential needs a baseline that is not the code under test.
    const resampled = ALL.map((w) =>
      resampleThresholdMask({ logits: w, ...GEOMETRY, threshold: OPTIONS.maskThreshold }),
    );
    const passed = resampled.filter((mask) => mask.area >= OPTIONS.minMaskArea);
    const reference = dedupeMasks(passed, OPTIONS.nmsIouThreshold, GEOMETRY.originalWidth).map(
      (index) => passed[index],
    );
    // Non-vacuous: the area gate drops two and NMS drops one, so an
    // implementation that skipped either gate would fail the comparison below.
    expect(passed).toHaveLength(4);
    expect(reference).toHaveLength(3);

    const plan = fullPlan();
    const candidates = ALL.map((w) => retainCandidate(w, plan).candidate).filter(
      (candidate): candidate is MaskCandidate => candidate !== null,
    );
    const returned = dedupeCandidates(candidates, plan)
      .map((index) => resolveSurvivor(candidates[index], plan))
      .filter((mask) => mask !== null);

    expect(returned).toHaveLength(reference.length);
    returned.forEach((mask, i) => {
      expect(mask!.area).toBe(reference[i].area);
      expect(Array.from(mask!.coverage)).toEqual(Array.from(reference[i].coverage));
    });
  });
});
