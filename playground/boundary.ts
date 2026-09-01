/**
 * The boundary invariant, made checkable with no GPU.
 *
 * `retainCandidate`, `dedupeCandidates` and `resolveSurvivor` are pure and
 * model-free, so both pipelines can be driven over synthetic 256x256 logit
 * windows in a plain browser tab: no WebGPU adapter, no ONNX download, no
 * network. That is what makes AC2 a real-engine assertion rather than an
 * eyeball over a photo.
 *
 * The fixtures are PROCEDURAL — smooth signed-distance fields generated here,
 * not committed binaries — so a reader can see exactly which shape stresses
 * which change.
 *
 * The geometry is a real one: a 512x333 image, resized so its longest side is
 * 1024 (giving 1024x666) and padded to a 1024x1024 square, decoded on a
 * 256x256 logit grid. Full resolution then works out to exactly 2x the low
 * grid on both axes, so a low box covering columns a..b covers full columns
 * 2a..2b+1 — which is why every expected number in `boundary.test.ts` can be
 * derived rather than guessed.
 */
import {
  createFilterPlan,
  dedupeCandidates,
  releaseRejected,
  resolveSurvivor,
  retainCandidate,
  type BinaryMask,
  type FilterNmsOptions,
  type MaskCandidate,
  type MaskGeometry,
} from '../src/segmenter';

// Re-exported so `BoundaryView` can name the type without reaching past this
// module into the package's own barrel.
export type { BinaryMask };

export const BOUNDARY_GEOMETRY: MaskGeometry = {
  lowWidth: 256,
  lowHeight: 256,
  padWidth: 1024,
  padHeight: 1024,
  reshapedWidth: 1024,
  reshapedHeight: 666,
  originalWidth: 512,
  originalHeight: 333,
};

/** The shipped defaults, minus the flag the two paths differ on. */
export const BOUNDARY_OPTIONS = {
  maskThreshold: 0,
  minMaskArea: 100,
  nmsIouThreshold: 0.7,
} as const;

function pathOptions(lowResFilterNms: boolean): FilterNmsOptions {
  // The Boundary tab compares FULL-RESOLUTION masks between the two filter
  // paths, so it pins the encode target off: a reduced PNG target would change
  // nothing it looks at, and pinning it says so.
  return { ...BOUNDARY_OPTIONS, lowResFilterNms, lowResMaskEncode: false };
}

/**
 * Signed distance to an axis-aligned box, POSITIVE INSIDE. The box is given as
 * inclusive integer pixel bounds and inflated by half a pixel each way, so
 * pixels `a..b` land at distance >= 0.5 and pixel `a - 1` at -0.5 — a clean
 * ramp through zero exactly halfway between them.
 */
function boxSdf(x: number, y: number, a: number, b: number, c: number, d: number): number {
  const dx = Math.max(a - 0.5 - x, x - (b + 0.5));
  const dy = Math.max(c - 0.5 - y, y - (d + 0.5));
  const outside = Math.hypot(Math.max(dx, 0), Math.max(dy, 0));
  const inside = Math.min(Math.max(dx, dy), 0);
  return -(outside + inside);
}

/** The union of the listed boxes, as a 256x256 signed-distance logit window. */
function sdfWindow(...boxes: [number, number, number, number][]): Float32Array {
  const { lowWidth: w, lowHeight: h } = BOUNDARY_GEOMETRY;
  const data = new Float32Array(w * h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let value = -Infinity;
      for (const [a, b, c, d] of boxes) value = Math.max(value, boxSdf(x, y, a, b, c, d));
      data[y * w + x] = value;
    }
  }
  return data;
}

/** Eight 3px tines on a 24px pitch, hanging off a spine. */
function combBoxes(): [number, number, number, number][] {
  const boxes: [number, number, number, number][] = [[40, 215, 150, 165]];
  for (let i = 0; i < 8; i += 1) boxes.push([40 + i * 24, 42 + i * 24, 100, 149]);
  return boxes;
}

export interface BoundaryFixture {
  id: string;
  title: string;
  /** What this fixture exists to break, in one sentence. */
  stresses: string;
  /** One 256x256 logit window per prompt-point candidate. */
  windows: Float32Array[];
}

export const BOUNDARY_FIXTURES: BoundaryFixture[] = [
  {
    id: 'comb',
    title: 'fine-toothed comb',
    stresses:
      'thin structure: eight 3-pixel tines that a coarser round-trip would thin, merge or lose.',
    windows: [sdfWindow(...combBoxes())],
  },
  {
    id: 'bridge',
    title: 'one-pixel bridge',
    stresses:
      'connectivity: two blobs joined by a single-pixel isthmus, which either survives the resample or does not.',
    windows: [sdfWindow([30, 89, 98, 157], [166, 225, 98, 157], [90, 165, 127, 127])],
  },
  {
    id: 'pair',
    title: 'near-duplicate pair straddling the IoU threshold',
    stresses:
      'N1: three identical 100x100 boxes at offsets 0, 17 and 18. Against the first, IoU is 83/117 = 0.7094 (above the 0.7 threshold, suppressed) and 82/118 = 0.6949 (below it, kept).',
    windows: [
      sdfWindow([78, 177, 40, 139]),
      sdfWindow([95, 194, 40, 139]),
      sdfWindow([96, 195, 40, 139]),
    ],
  },
  {
    id: 'speck',
    title: 'speck below the scaled gate',
    stresses:
      'F3: a 6x6 box with a low-res area of 36 against a scaled gate of 38, whose full-resolution area of 144 clears minMaskArea 100 comfortably. The baseline keeps it; the low-res path never lets it reach NMS.',
    windows: [sdfWindow([125, 130, 125, 130])],
  },
];

export type WindowStatus = 'kept' | 'filtered' | 'suppressed' | 'undersized';

export interface WindowOutcome {
  /**
   * `filtered` — dropped by the pre-NMS area gate.
   * `suppressed` — dropped by NMS as a duplicate.
   * `undersized` — survived NMS, then failed the exact full-resolution minMaskArea.
   * `kept` — returned.
   */
  status: WindowStatus;
  /** The area the pre-NMS gate saw: 256x256 on the low path, full-res on the baseline. */
  gateArea: number;
  /** Full-resolution coverage. Non-null only when `status` is `kept`. */
  mask: BinaryMask | null;
}

export interface PathResult {
  /** One outcome per source window, in input order. */
  outcomes: WindowOutcome[];
  /** The pre-NMS gate this path applied, in `gateArea`'s units. */
  minArea: number;
}

/**
 * One fixture through one path, in the same order the worker calls these
 * functions in: retain every candidate, dedupe, release the rejected, resolve
 * each survivor.
 *
 * The one deliberate difference from the worker: every survivor's mask is kept
 * so the tab can render them side by side. The worker holds one at a time, and
 * that bound is the worker's to keep.
 */
export function runBoundaryPath(
  windows: readonly Float32Array[],
  lowResFilterNms: boolean,
): PathResult {
  const plan = createFilterPlan(BOUNDARY_GEOMETRY, pathOptions(lowResFilterNms));
  const outcomes: WindowOutcome[] = windows.map(() => ({
    status: 'filtered',
    gateArea: 0,
    mask: null,
  }));

  const live: MaskCandidate[] = [];
  const liveSource: number[] = [];
  windows.forEach((window, index) => {
    const retained = retainCandidate(window, plan);
    outcomes[index].gateArea = retained.gateArea;
    if (retained.candidate) {
      live.push(retained.candidate);
      liveSource.push(index);
      outcomes[index].status = 'suppressed';
    }
  });

  const kept = dedupeCandidates(live, plan);
  releaseRejected(live, kept);
  for (const index of kept) {
    const mask = resolveSurvivor(live[index], plan);
    const outcome = outcomes[liveSource[index]];
    outcome.status = mask ? 'kept' : 'undersized';
    outcome.mask = mask;
  }

  return { outcomes, minArea: plan.minArea };
}

/** Pixels covered by exactly one of the two masks. Both are full resolution. */
export function differingPixels(a: BinaryMask, b: BinaryMask): number {
  let diff = 0;
  for (let p = 0; p < a.coverage.length; p += 1) {
    if (a.coverage[p] !== b.coverage[p]) diff += 1;
  }
  return diff;
}
