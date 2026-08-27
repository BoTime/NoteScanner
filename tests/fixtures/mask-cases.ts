/**
 * The mask fixtures both verification layers run: the Node round-trip in
 * `src/segmenter/core/mask-encode.test.ts` and the real-browser decode in
 * `tests/browser/mask-png.spec.ts`. Shared so the two layers cannot drift
 * apart — a case added here is immediately checked in three real engines.
 *
 * Lives outside `src/` because it is test-only, but `npm run typecheck` still
 * covers it: the vitest spec imports it, and `tsc` typechecks every file it
 * pulls into the program.
 */

export interface MaskCase {
  name: string;
  width: number;
  height: number;
  coverage: Uint8Array;
}

/**
 * mulberry32. Property coverage over random masks is only worth having if a
 * failure reproduces, so nothing here calls `Math.random`.
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * 7, 9 and 17 are not multiples of 8, so their last row byte carries padding
 * bits — the case a 1-bit writer gets wrong. 1xN and Nx1 are the degenerate
 * strips; 64x64 is big enough that deflate actually has something to chew on.
 */
const SHAPES: ReadonlyArray<readonly [number, number]> = [
  [1, 1],
  [7, 3],
  [8, 3],
  [9, 3],
  [17, 5],
  [1, 40],
  [40, 1],
  [64, 64],
];

function buildMaskCases(): MaskCase[] {
  const random = mulberry32(0x5eed);
  const cases: MaskCase[] = [];
  for (const [width, height] of SHAPES) {
    const pixels = width * height;
    cases.push({
      name: `all-zero ${width}x${height}`,
      width,
      height,
      coverage: new Uint8Array(pixels),
    });
    cases.push({
      name: `all-one ${width}x${height}`,
      width,
      height,
      coverage: new Uint8Array(pixels).fill(1),
    });
    const noisy = new Uint8Array(pixels);
    for (let i = 0; i < pixels; i += 1) noisy[i] = random() < 0.5 ? 1 : 0;
    cases.push({ name: `random ${width}x${height}`, width, height, coverage: noisy });
  }
  return cases;
}

export const MASK_CASES: readonly MaskCase[] = buildMaskCases();
