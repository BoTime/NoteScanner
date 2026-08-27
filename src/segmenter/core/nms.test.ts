import { describe, it, expect } from 'vitest';
import { dedupeMasks, dedupeMasksReference, pairwiseIoU, popcount, type BinaryMask } from './nms';

function mask(bits: number[]): BinaryMask {
  const coverage = Uint8Array.from(bits);
  return { coverage, area: bits.reduce((sum, b) => sum + (b ? 1 : 0), 0) };
}

/** A filled axis-aligned rectangle on a `width` x `height` canvas. */
function rect(
  width: number,
  height: number,
  x0: number,
  y0: number,
  w: number,
  h: number,
): BinaryMask {
  const coverage = new Uint8Array(width * height);
  let area = 0;
  for (let y = y0; y < y0 + h; y += 1) {
    for (let x = x0; x < x0 + w; x += 1) {
      coverage[y * width + x] = 1;
      area += 1;
    }
  }
  return { coverage, area };
}

/** mulberry32 — small, seeded, reproducible. A failure below repeats exactly. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('pairwiseIoU', () => {
  it('is 1 for identical masks', () => {
    expect(pairwiseIoU(mask([1, 1, 0, 0]), mask([1, 1, 0, 0]))).toBe(1);
  });

  it('is 0 for disjoint masks', () => {
    expect(pairwiseIoU(mask([1, 1, 0, 0]), mask([0, 0, 1, 1]))).toBe(0);
  });

  it('is smaller / larger when one mask fully contains the other', () => {
    expect(pairwiseIoU(mask([1, 0, 0, 0]), mask([1, 1, 1, 1]))).toBe(0.25);
  });

  it('is 0 rather than NaN when both masks are empty', () => {
    expect(pairwiseIoU(mask([0, 0]), mask([0, 0]))).toBe(0);
  });
});

describe('popcount', () => {
  it('counts nothing in zero', () => {
    expect(popcount(0)).toBe(0);
  });

  it('counts every bit in an all-ones word', () => {
    expect(popcount(0xffffffff)).toBe(32);
  });

  // 0x80000000 arrives from `a & b` as the NEGATIVE int32 -2147483648. Both
  // spellings must answer 1: a popcount that lets the sign bit through the
  // first subtraction gets this wrong, and nothing else in the suite would
  // notice.
  it('counts the high bit exactly once, unsigned or sign-extended', () => {
    expect(popcount(0x80000000)).toBe(1);
    expect(popcount(0x80000000 | 0)).toBe(1);
  });

  it('counts exactly one bit at every single-bit position 0..31', () => {
    for (let bit = 0; bit < 32; bit += 1) {
      expect(popcount((1 << bit) >>> 0)).toBe(1);
      expect(popcount(1 << bit)).toBe(1);
    }
  });
});

describe('dedupeMasks', () => {
  it('returns nothing for no candidates', () => {
    expect(dedupeMasks([], 0.7, 4)).toEqual([]);
  });

  it('keeps the larger of two duplicates and drops the smaller', () => {
    const kept = dedupeMasks([mask([1, 0, 0, 0]), mask([1, 1, 1, 0])], 0.2, 4);
    expect(kept).toEqual([1]);
  });

  it('keeps disjoint masks and returns their indices ascending', () => {
    const kept = dedupeMasks([mask([0, 0, 1, 1]), mask([1, 1, 0, 0])], 0.5, 4);
    expect(kept).toEqual([0, 1]);
  });

  it('keeps a candidate whose IoU is EXACTLY the threshold', () => {
    // IoU of a 1-pixel mask inside a 4-pixel mask is exactly 0.25.
    const kept = dedupeMasks([mask([1, 1, 1, 1]), mask([1, 0, 0, 0])], 0.25, 4);
    expect(kept).toEqual([0, 1]);
  });

  it('drops that same candidate once the threshold dips just below its IoU', () => {
    const kept = dedupeMasks([mask([1, 1, 1, 1]), mask([1, 0, 0, 0])], 0.24, 4);
    expect(kept).toEqual([0]);
  });

  it('breaks an equal-area tie in favour of the lower original index', () => {
    const kept = dedupeMasks([mask([1, 1, 0, 0]), mask([1, 1, 0, 0])], 0.5, 4);
    expect(kept).toEqual([0]);
  });

  it('does not let a zero-area mask suppress anything', () => {
    const kept = dedupeMasks([mask([0, 0, 0, 0]), mask([1, 1, 0, 0])], 0.5, 4);
    expect(kept).toEqual([0, 1]);
  });
});

describe('dedupeMasks matches dedupeMasksReference', () => {
  it('on a pair the bbox prefilter rejects on the x axis', () => {
    const masks = [rect(8, 4, 0, 0, 3, 4), rect(8, 4, 5, 0, 3, 4)];
    expect(dedupeMasks(masks, 0.0, 8)).toEqual(dedupeMasksReference(masks, 0.0));
    expect(dedupeMasks(masks, 0.0, 8)).toEqual([0, 1]);
  });

  it('on a pair the bbox prefilter rejects on the y axis', () => {
    const masks = [rect(8, 6, 0, 0, 8, 2), rect(8, 6, 0, 4, 8, 2)];
    expect(dedupeMasks(masks, 0.0, 8)).toEqual(dedupeMasksReference(masks, 0.0));
    expect(dedupeMasks(masks, 0.0, 8)).toEqual([0, 1]);
  });

  // 7 x 5 = 35 pixels: two words, of which 29 bits of the second are unused.
  // If the tail's unused high bits were ever set or counted, the intersection
  // would come out too large and the kept set would differ.
  it('on masks whose coverage length is not a multiple of 32', () => {
    const masks = [
      rect(7, 5, 0, 0, 7, 5),
      rect(7, 5, 0, 0, 4, 5),
      rect(7, 5, 3, 1, 4, 3),
      rect(7, 5, 6, 4, 1, 1),
    ];
    for (const threshold of [0.0, 0.2, 0.5, 0.8]) {
      expect(dedupeMasks(masks, threshold, 7)).toEqual(dedupeMasksReference(masks, threshold));
    }
  });

  // The differential. 300 overlapping rectangles on a 2-D canvas, seeded so a
  // failure reproduces. 61 x 48 rather than a literal full-resolution canvas:
  // the reference is O(pairs x pixels) byte reads, and real 617 x 0.7 MB
  // candidates would put minutes into every CI run for no extra coverage of
  // the two transformations under test.
  //
  // The width is deliberately NOT a multiple of 32. At a multiple of 32 every
  // row starts on a word boundary and the coverage length is a whole number of
  // words, so `first` and `last` land exactly on the shared row band's edges
  // and no word ever straddles it — which is precisely the case `packedIoU`'s
  // "extra bits pulled in by a boundary word cannot be set in both masks"
  // argument exists for. 61 gives a partial tail word (61 x 48 = 2928 pixels,
  // 92 words) and rows that start mid-word, folding both into the 300-candidate
  // 6-threshold differential instead of leaving them to the canned 7 x 5 case.
  const CANVAS_W = 61;
  const CANVAS_H = 48;
  const candidates = (() => {
    const random = mulberry32(0x5eed1234);
    const built: BinaryMask[] = [];
    for (let i = 0; i < 300; i += 1) {
      // Every 40th is a genuinely empty mask, so the empty/zero-area path is
      // inside the differential rather than only in the canned cases above.
      if (i % 40 === 39) {
        built.push({ coverage: new Uint8Array(CANVAS_W * CANVAS_H), area: 0 });
        continue;
      }
      const w = 4 + Math.floor(random() * 20);
      const h = 4 + Math.floor(random() * 16);
      const x0 = Math.floor(random() * (CANVAS_W - w + 1));
      const y0 = Math.floor(random() * (CANVAS_H - h + 1));
      built.push(rect(CANVAS_W, CANVAS_H, x0, y0, w, h));
    }
    return built;
  })();

  it.each([0.0, 0.1, 0.25, 0.5, 0.7, 0.9])(
    'on 300 seeded overlapping candidates at threshold %s',
    (threshold) => {
      // Exact equality, no tolerance: N2 and N3 are exact transformations.
      expect(dedupeMasks(candidates, threshold, CANVAS_W)).toEqual(
        dedupeMasksReference(candidates, threshold),
      );
    },
  );
});
