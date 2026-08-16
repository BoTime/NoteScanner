import { describe, it, expect } from 'vitest';
import {
  applyCountDelta,
  brightWindowKey,
  buildBrightWindowImageData,
  buildEdgeCoverage,
  buildOutlineImageData,
  coverageToBounds,
  coverageToIndices,
  countsToImageData,
  diffSelection,
  hitTest,
  hitTestAll,
  largestConnectedComponent,
  outlineKey,
  resolveAppliedDelta,
  toggleSelection,
  type SegmentMaskData,
} from './segment-viewer-logic';

// The Vitest environment is `node`, which has no DOM, so `ImageData` (used by
// buildBrightWindowImageData) is undefined. Provide a minimal stand-in matching
// the real constructor: `new ImageData(width, height)` allocates a zero-filled
// RGBA buffer of width*height*4 bytes.
if (typeof globalThis.ImageData === 'undefined') {
  class FakeImageData {
    readonly width: number;
    readonly height: number;
    readonly data: Uint8ClampedArray;
    readonly colorSpace = 'srgb' as const;
    constructor(width: number, height: number) {
      this.width = width;
      this.height = height;
      this.data = new Uint8ClampedArray(width * height * 4);
    }
  }
  (globalThis as { ImageData: unknown }).ImageData = FakeImageData;
}

function makeMask(
  id: string,
  width: number,
  height: number,
  fill: (x: number, y: number) => boolean,
): SegmentMaskData {
  const coverage = new Uint8Array(width * height);
  let area = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (fill(x, y)) {
        coverage[y * width + x] = 1;
        area += 1;
      }
    }
  }
  return { id, coverage, area };
}

describe('coverageToBounds', () => {
  it('returns null when nothing is covered', () => {
    const cov = new Uint8Array(4 * 4);
    expect(coverageToBounds(cov, 4, 4, 0)).toBeNull();
  });

  it('returns a tight (unpadded) box around a single blob', () => {
    // single covered pixel at (2, 1)
    const cov = makeMask('a', 4, 4, (x, y) => x === 2 && y === 1).coverage;
    expect(coverageToBounds(cov, 4, 4, 0)).toEqual({
      left: 2,
      top: 1,
      width: 1,
      height: 1,
    });
  });

  it('pads the box and clamps to the image edges', () => {
    const cov = makeMask('a', 6, 6, (x, y) => x === 0 && y === 0).coverage;
    // padding 2 would push left/top to -2 but clamps to 0; box stays in bounds.
    expect(coverageToBounds(cov, 6, 6, 2)).toEqual({
      left: 0,
      top: 0,
      width: 3,
      height: 3,
    });
  });

  it('a stray speck inflates the box (documents why the preview must clip)', () => {
    // main blob at (5,5) plus one stray pixel at (0,0): the plain min/max box
    // spans both, which is exactly the case the mask-clipping crop neutralises.
    const cov = makeMask(
      'a',
      8,
      8,
      (x, y) => (x === 5 && y === 5) || (x === 0 && y === 0),
    ).coverage;
    expect(coverageToBounds(cov, 8, 8, 0)).toEqual({
      left: 0,
      top: 0,
      width: 6,
      height: 6,
    });
  });
});

describe('largestConnectedComponent', () => {
  // Build coverage from an ASCII grid ('#' = covered).
  function grid(rows: string[]): {
    coverage: Uint8Array;
    width: number;
    height: number;
  } {
    const height = rows.length;
    const width = rows[0]?.length ?? 0;
    const coverage = new Uint8Array(width * height);
    for (let y = 0; y < height; y += 1)
      for (let x = 0; x < width; x += 1)
        if (rows[y][x] === '#') coverage[y * width + x] = 1;
    return { coverage, width, height };
  }
  function render(cov: Uint8Array, width: number, height: number): string[] {
    const out: string[] = [];
    for (let y = 0; y < height; y += 1) {
      let row = '';
      for (let x = 0; x < width; x += 1) row += cov[y * width + x] ? '#' : '.';
      out.push(row);
    }
    return out;
  }

  it('drops a stray speck far from the main blob', () => {
    const { coverage, width, height } = grid([
      '#......',
      '.......',
      '....##.',
      '....##.',
    ]);
    const result = largestConnectedComponent(coverage, width, height);
    expect(render(result, width, height)).toEqual([
      '.......',
      '.......',
      '....##.',
      '....##.',
    ]);
  });

  it('keeps diagonally-touching pixels together (8-connectivity)', () => {
    const { coverage, width, height } = grid([
      '#....',
      '.#...',
      '.....',
      '....#',
    ]);
    const result = largestConnectedComponent(coverage, width, height);
    expect(render(result, width, height)).toEqual([
      '#....',
      '.#...',
      '.....',
      '.....',
    ]);
  });

  it('returns an all-empty copy for empty coverage', () => {
    const { coverage, width, height } = grid(['...', '...']);
    const result = largestConnectedComponent(coverage, width, height);
    expect(render(result, width, height)).toEqual(['...', '...']);
  });
});

describe('SegmentViewer — hitTest', () => {
  it('returns null when click is outside the canvas', () => {
    const masks = [makeMask('a', 4, 4, () => true)];
    expect(hitTest({ x: -1, y: 0 }, 4, 4, masks)).toBeNull();
    expect(hitTest({ x: 4, y: 0 }, 4, 4, masks)).toBeNull();
    expect(hitTest({ x: 0, y: 4 }, 4, 4, masks)).toBeNull();
  });

  it('returns null when no mask covers the point', () => {
    const masks = [makeMask('a', 4, 4, () => false)];
    expect(hitTest({ x: 1, y: 1 }, 4, 4, masks)).toBeNull();
  });

  it('returns the only segment that covers the point', () => {
    const masks = [
      makeMask('a', 4, 4, (x, y) => x < 2 && y < 2),
      makeMask('b', 4, 4, (x, y) => x >= 2 && y >= 2),
    ];
    expect(hitTest({ x: 0, y: 0 }, 4, 4, masks)).toBe('a');
    expect(hitTest({ x: 3, y: 3 }, 4, 4, masks)).toBe('b');
  });

  it('returns the smallest mask among multiple hits (most-specific bias)', () => {
    const big = makeMask('big', 4, 4, () => true); // area 16
    const small = makeMask('small', 4, 4, (x, y) => x === 1 && y === 1); // area 1
    expect(hitTest({ x: 1, y: 1 }, 4, 4, [big, small])).toBe('small');
    expect(hitTest({ x: 1, y: 1 }, 4, 4, [small, big])).toBe('small');
    // Outside the small mask, big still wins.
    expect(hitTest({ x: 0, y: 0 }, 4, 4, [big, small])).toBe('big');
  });
});

describe('SegmentViewer — hitTestAll', () => {
  it('returns every covering mask ordered smallest-first', () => {
    const big = makeMask('big', 4, 4, () => true); // area 16
    const medium = makeMask('medium', 4, 4, (x, y) => x < 3 && y < 3); // area 9
    const small = makeMask('small', 4, 4, (x, y) => x === 1 && y === 1); // area 1

    expect(hitTestAll({ x: 1, y: 1 }, 4, 4, [big, small, medium])).toEqual([
      'small',
      'medium',
      'big',
    ]);
  });

  it('returns an empty array when no mask covers the point or the point is outside', () => {
    const masks = [makeMask('a', 4, 4, (x, y) => x === 0 && y === 0)];
    expect(hitTestAll({ x: 2, y: 2 }, 4, 4, masks)).toEqual([]);
    expect(hitTestAll({ x: -1, y: 0 }, 4, 4, masks)).toEqual([]);
    expect(hitTestAll({ x: 4, y: 0 }, 4, 4, masks)).toEqual([]);
  });
});

describe('SegmentViewer — buildBrightWindowImageData', () => {
  // Read the alpha byte for pixel (x, y).
  function alphaAt(img: ImageData, x: number, y: number): number {
    return img.data[(y * img.width + x) * 4 + 3];
  }

  function covMask(
    width: number,
    height: number,
    fill: (x: number, y: number) => boolean,
  ): { coverage: Uint8Array } {
    const coverage = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (fill(x, y)) coverage[y * width + x] = 1;
      }
    }
    return { coverage };
  }

  it('produces all-zero alpha when the bright set is empty', () => {
    const masks = new Map([['a', covMask(4, 4, () => true)]]);
    const img = buildBrightWindowImageData(4, 4, new Set(), masks);
    expect(img.data.every((byte) => byte === 0)).toBe(true);
  });

  it('sets alpha 255 only for covered pixels of one selected mask', () => {
    const masks = new Map([
      ['a', covMask(4, 4, (x, y) => x < 2 && y < 2)],
    ]);
    const img = buildBrightWindowImageData(4, 4, new Set(['a']), masks);
    expect(alphaAt(img, 0, 0)).toBe(255);
    expect(alphaAt(img, 1, 1)).toBe(255);
    expect(alphaAt(img, 2, 2)).toBe(0);
    expect(alphaAt(img, 3, 3)).toBe(0);
  });

  it('unions overlapping selected masks', () => {
    const masks = new Map([
      ['a', covMask(4, 4, (x) => x < 2)], // left half
      ['b', covMask(4, 4, (x) => x >= 1)], // everything except column 0
    ]);
    const img = buildBrightWindowImageData(4, 4, new Set(['a', 'b']), masks);
    // Every pixel is covered by at least one of the two masks.
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        expect(alphaAt(img, x, y)).toBe(255);
      }
    }
  });

  it('does not let a deselected mask subtract from a selected overlapping mask', () => {
    const small = covMask(4, 4, (x, y) => x === 1 && y === 1); // selected
    const big = covMask(4, 4, () => true); // deselected, fully overlaps
    const masks = new Map([
      ['small', small],
      ['big', big],
    ]);
    // Only `small` is bright; `big` is deselected and must not be visited.
    const img = buildBrightWindowImageData(4, 4, new Set(['small']), masks);
    expect(alphaAt(img, 1, 1)).toBe(255); // selected pixel stays bright
    expect(alphaAt(img, 0, 0)).toBe(0); // deselected coverage does not light up
    expect(alphaAt(img, 3, 3)).toBe(0);
  });

  it('ignores bright ids with no loaded mask', () => {
    const masks = new Map([
      ['a', covMask(4, 4, (x, y) => x === 0 && y === 0)],
    ]);
    const img = buildBrightWindowImageData(
      4,
      4,
      new Set(['a', 'missing']),
      masks,
    );
    expect(alphaAt(img, 0, 0)).toBe(255);
    // Missing id contributes nothing; the rest stays transparent.
    expect(alphaAt(img, 1, 1)).toBe(0);
  });
});

describe('SegmentViewer — brightWindowKey', () => {
  const cov = { coverage: new Uint8Array(1) };

  it('produces the same key for the same bright set and loaded masks', () => {
    const masks = new Map([['a', cov]]);
    expect(brightWindowKey(4, 4, new Set(['a']), masks)).toBe(
      brightWindowKey(4, 4, new Set(['a']), masks),
    );
  });

  // Regression: drawing a new segment empties masksRef and reloads async. A
  // rebuild that fires before the mask loads must NOT share a key with the
  // rebuild after it loads — otherwise the memo suppresses the real rebuild
  // and the drawn segment never highlights.
  it('differs when the bright id is selected-but-not-yet-loaded vs loaded', () => {
    const empty = new Map<string, typeof cov>();
    const loaded = new Map([['new', cov]]);
    const beforeLoad = brightWindowKey(4, 4, new Set(['new']), empty);
    const afterLoad = brightWindowKey(4, 4, new Set(['new']), loaded);
    expect(beforeLoad).not.toBe(afterLoad);
  });

  it('is order-independent in the selected set', () => {
    const masks = new Map([
      ['a', cov],
      ['b', cov],
    ]);
    expect(brightWindowKey(4, 4, new Set(['a', 'b']), masks)).toBe(
      brightWindowKey(4, 4, new Set(['b', 'a']), masks),
    );
  });

  it('changes when image dimensions change', () => {
    const masks = new Map([['a', cov]]);
    expect(brightWindowKey(4, 4, new Set(['a']), masks)).not.toBe(
      brightWindowKey(8, 4, new Set(['a']), masks),
    );
  });
});

describe('SegmentViewer — toggleSelection', () => {
  it('removes the id when present', () => {
    const out = toggleSelection(new Set(['a', 'b']), 'a');
    expect(Array.from(out).sort()).toEqual(['b']);
  });

  it('adds the id when absent', () => {
    const out = toggleSelection(new Set(['a']), 'b');
    expect(Array.from(out).sort()).toEqual(['a', 'b']);
  });

  it('returns a new Set rather than mutating the input', () => {
    const input = new Set(['a']);
    const out = toggleSelection(input, 'b');
    expect(input.has('b')).toBe(false);
    expect(out).not.toBe(input);
  });
});

describe('SegmentViewer — buildEdgeCoverage', () => {
  function covArray(
    width: number,
    height: number,
    fill: (x: number, y: number) => boolean,
  ): Uint8Array {
    const coverage = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (fill(x, y)) coverage[y * width + x] = 1;
      }
    }
    return coverage;
  }

  // Set of "x,y" for every edge pixel, for readable assertions.
  function edgeSet(edge: Uint8Array, width: number, height: number): Set<string> {
    const out = new Set<string>();
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (edge[y * width + x]) out.add(`${x},${y}`);
      }
    }
    return out;
  }

  it('returns an empty edge array when nothing is covered', () => {
    const edge = buildEdgeCoverage(covArray(4, 4, () => false), 4, 4, 0);
    expect(edge.every((b) => b === 0)).toBe(true);
  });

  it('marks the perimeter of a solid interior square, not its interior', () => {
    const edge = buildEdgeCoverage(
      covArray(5, 5, (x, y) => x >= 1 && x <= 3 && y >= 1 && y <= 3),
      5,
      5,
      0,
    );
    const s = edgeSet(edge, 5, 5);
    expect(s.has('2,2')).toBe(false); // interior pixel, fully surrounded
    expect(s.has('1,1')).toBe(true);
    expect(s.has('3,3')).toBe(true);
    expect(s.has('2,1')).toBe(true);
    expect(s.has('0,0')).toBe(false); // uncovered pixels are never edges
  });

  it('treats image-border covered pixels as edges (outline closes on the edge)', () => {
    const edge = buildEdgeCoverage(covArray(3, 3, () => true), 3, 3, 0);
    const s = edgeSet(edge, 3, 3);
    expect(s.has('1,1')).toBe(false); // center: all 4 neighbors covered & in-bounds
    expect(s.has('0,0')).toBe(true);
    expect(s.has('2,2')).toBe(true);
    expect(s.has('0,1')).toBe(true);
  });

  it('marks the boundary of an interior hole', () => {
    const edge = buildEdgeCoverage(
      covArray(5, 5, (x, y) => !(x === 2 && y === 2)),
      5,
      5,
      0,
    );
    const s = edgeSet(edge, 5, 5);
    expect(s.has('2,1')).toBe(true); // above the hole
    expect(s.has('1,2')).toBe(true); // left of the hole
    expect(s.has('2,2')).toBe(false); // the hole itself is uncovered
  });

  it('marks both blobs of a disconnected mask', () => {
    const edge = buildEdgeCoverage(
      covArray(5, 1, (x) => x === 0 || x === 4),
      5,
      1,
      0,
    );
    const s = edgeSet(edge, 5, 1);
    expect(s.has('0,0')).toBe(true);
    expect(s.has('4,0')).toBe(true);
    expect(s.has('2,0')).toBe(false);
  });

  it('dilates edges by the given radius to thicken the line', () => {
    const base = covArray(5, 5, (x, y) => x === 2 && y === 2);
    const r0 = edgeSet(buildEdgeCoverage(base, 5, 5, 0), 5, 5);
    expect(r0).toEqual(new Set(['2,2']));
    const r1 = edgeSet(buildEdgeCoverage(base, 5, 5, 1), 5, 5);
    expect(r1.has('2,2')).toBe(true);
    expect(r1.has('1,1')).toBe(true);
    expect(r1.has('3,3')).toBe(true);
    expect(r1.has('2,1')).toBe(true);
    expect(r1.has('0,0')).toBe(false);
  });
});

describe('SegmentViewer — buildOutlineImageData', () => {
  function rgbaAt(img: ImageData, x: number, y: number): [number, number, number, number] {
    const i = (y * img.width + x) * 4;
    return [img.data[i], img.data[i + 1], img.data[i + 2], img.data[i + 3]];
  }
  function covMask(
    width: number,
    height: number,
    fill: (x: number, y: number) => boolean,
  ): { coverage: Uint8Array } {
    const coverage = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (fill(x, y)) coverage[y * width + x] = 1;
      }
    }
    return { coverage };
  }

  it('paints edge pixels in the given color and leaves the rest transparent', () => {
    const masks = new Map([['a', covMask(3, 3, () => true)]]);
    const img = buildOutlineImageData(
      3,
      3,
      new Set(['a']),
      masks,
      { r: 249, g: 115, b: 22 },
      0,
      false,
    );
    expect(rgbaAt(img, 0, 0)).toEqual([249, 115, 22, 255]); // edge
    expect(rgbaAt(img, 1, 1)).toEqual([0, 0, 0, 0]); // interior, transparent
  });

  it('unions edges of multiple ids', () => {
    const masks = new Map([
      ['a', covMask(5, 1, (x) => x === 0)],
      ['b', covMask(5, 1, (x) => x === 4)],
    ]);
    const img = buildOutlineImageData(
      5,
      1,
      new Set(['a', 'b']),
      masks,
      { r: 1, g: 2, b: 3 },
      0,
      false,
    );
    expect(rgbaAt(img, 0, 0)[3]).toBe(255);
    expect(rgbaAt(img, 4, 0)[3]).toBe(255);
    expect(rgbaAt(img, 2, 0)[3]).toBe(0);
  });

  it('produces fully-transparent output for an empty id set', () => {
    const masks = new Map([['a', covMask(3, 3, () => true)]]);
    const img = buildOutlineImageData(
      3,
      3,
      new Set(),
      masks,
      { r: 1, g: 2, b: 3 },
      0,
      false,
    );
    expect(img.data.every((byte) => byte === 0)).toBe(true);
  });

  it('ignores ids with no loaded mask', () => {
    const masks = new Map([['a', covMask(3, 3, () => true)]]);
    const img = buildOutlineImageData(
      3,
      3,
      new Set(['a', 'missing']),
      masks,
      { r: 9, g: 9, b: 9 },
      0,
      false,
    );
    expect(rgbaAt(img, 0, 0)[3]).toBe(255); // 'a' still drawn
  });

  it('stipples every other edge pixel when dashed=true', () => {
    const masks = new Map([['a', covMask(4, 1, () => true)]]);
    const img = buildOutlineImageData(
      4,
      1,
      new Set(['a']),
      masks,
      { r: 5, g: 5, b: 5 },
      0,
      true,
    );
    expect(rgbaAt(img, 0, 0)[3]).toBe(255);
    expect(rgbaAt(img, 1, 0)[3]).toBe(0);
    expect(rgbaAt(img, 2, 0)[3]).toBe(255);
    expect(rgbaAt(img, 3, 0)[3]).toBe(0);
  });
});

describe('SegmentViewer — outlineKey', () => {
  const cov = { coverage: new Uint8Array(1) };

  it('is stable for the same inputs', () => {
    const masks = new Map([['a', cov]]);
    expect(outlineKey(4, 4, new Set(['a']), masks, 'sel')).toBe(
      outlineKey(4, 4, new Set(['a']), masks, 'sel'),
    );
  });

  it('differs by tag so selected and hover layers never share a cache slot', () => {
    const masks = new Map([['a', cov]]);
    expect(outlineKey(4, 4, new Set(['a']), masks, 'sel')).not.toBe(
      outlineKey(4, 4, new Set(['a']), masks, 'hover'),
    );
  });

  it('differs when the id set changes', () => {
    const masks = new Map([
      ['a', cov],
      ['b', cov],
    ]);
    expect(outlineKey(4, 4, new Set(['a']), masks, 'sel')).not.toBe(
      outlineKey(4, 4, new Set(['a', 'b']), masks, 'sel'),
    );
  });

  it('distinguishes selected-but-not-loaded from loaded (mirrors brightWindowKey)', () => {
    const empty = new Map<string, typeof cov>();
    const loaded = new Map([['new', cov]]);
    expect(outlineKey(4, 4, new Set(['new']), empty, 'sel')).not.toBe(
      outlineKey(4, 4, new Set(['new']), loaded, 'sel'),
    );
  });
});

describe('SegmentViewer — incremental accumulators', () => {
  function rgbaAt(
    img: ImageData,
    x: number,
    y: number,
  ): [number, number, number, number] {
    const i = (y * img.width + x) * 4;
    return [img.data[i], img.data[i + 1], img.data[i + 2], img.data[i + 3]];
  }

  describe('coverageToIndices', () => {
    it('returns the indices of all set pixels, ascending', () => {
      const cov = new Uint8Array([0, 1, 0, 1, 1]);
      expect(Array.from(coverageToIndices(cov))).toEqual([1, 3, 4]);
    });

    it('returns an empty array when nothing is set', () => {
      expect(Array.from(coverageToIndices(new Uint8Array([0, 0, 0])))).toEqual(
        [],
      );
    });
  });

  describe('diffSelection', () => {
    it('reports ids added and removed between two sets', () => {
      const prev = new Set(['a', 'b', 'c']);
      const next = new Set(['b', 'c', 'd']);
      const { added, removed } = diffSelection(prev, next);
      expect(added.sort()).toEqual(['d']);
      expect(removed.sort()).toEqual(['a']);
    });

    it('treats a first selection (empty prev) as all-added', () => {
      const { added, removed } = diffSelection(new Set<string>(), new Set(['a', 'b']));
      expect(added.sort()).toEqual(['a', 'b']);
      expect(removed).toEqual([]);
    });

    it('returns empty added/removed for identical sets', () => {
      const { added, removed } = diffSelection(new Set(['a']), new Set(['a']));
      expect(added).toEqual([]);
      expect(removed).toEqual([]);
    });
  });

  describe('applyCountDelta', () => {
    it('increments counts at the given indices', () => {
      const counts = new Uint16Array(5);
      applyCountDelta(counts, Uint32Array.from([1, 3]), 1);
      expect(Array.from(counts)).toEqual([0, 1, 0, 1, 0]);
    });

    it('accumulates overlapping contributions and decrements without erasing shared pixels', () => {
      const counts = new Uint16Array(4);
      const maskA = Uint32Array.from([0, 1]);
      const maskB = Uint32Array.from([1, 2]); // pixel 1 shared with A
      applyCountDelta(counts, maskA, 1);
      applyCountDelta(counts, maskB, 1);
      expect(Array.from(counts)).toEqual([1, 2, 1, 0]);
      // Deselect A: pixel 1 must stay > 0 because B still covers it.
      applyCountDelta(counts, maskA, -1);
      expect(Array.from(counts)).toEqual([0, 1, 1, 0]);
    });
  });

  describe('countsToImageData', () => {
    it('alpha mode: sets alpha 255 where count > 0, color channels 0', () => {
      const counts = Uint16Array.from([0, 2, 0, 1]);
      const img = countsToImageData(2, 2, counts, null);
      expect(rgbaAt(img, 0, 0)).toEqual([0, 0, 0, 0]); // count 0
      expect(rgbaAt(img, 1, 0)).toEqual([0, 0, 0, 255]); // count 2
      expect(rgbaAt(img, 1, 1)).toEqual([0, 0, 0, 255]); // count 1
    });

    it('color mode: paints the given rgb at full alpha where count > 0', () => {
      const counts = Uint16Array.from([0, 1, 0, 3]);
      const img = countsToImageData(2, 2, counts, { r: 249, g: 115, b: 22 });
      expect(rgbaAt(img, 0, 0)).toEqual([0, 0, 0, 0]);
      expect(rgbaAt(img, 1, 0)).toEqual([249, 115, 22, 255]);
      expect(rgbaAt(img, 1, 1)).toEqual([249, 115, 22, 255]);
    });
  });
});

describe('SegmentViewer — resolveAppliedDelta', () => {
  it('applies an added id only when its mask is loaded', () => {
    const prev = new Set<string>();
    const next = new Set(['a']);
    const loadedNo = resolveAppliedDelta(prev, next, () => false);
    expect(loadedNo.added).toEqual([]); // mask not loaded → not applied
    expect(loadedNo.applied.has('a')).toBe(false); // not recorded as applied

    const loadedYes = resolveAppliedDelta(prev, next, () => true);
    expect(loadedYes.added).toEqual(['a']);
    expect(loadedYes.applied.has('a')).toBe(true);
  });

  it('regression: a rebuild fired before masks load does not poison the later rebuild', () => {
    // Reload window: selection is {a} but masksRef is empty.
    const first = resolveAppliedDelta(new Set(), new Set(['a']), () => false);
    expect(first.added).toEqual([]);
    // `applied` is what becomes the next prevSelected snapshot — must NOT include
    // the not-yet-loaded id, or the post-load run sees an empty diff and skips.
    expect(Array.from(first.applied)).toEqual([]);

    // Masks finished loading; same selection. Because `a` was never recorded as
    // applied, it is still seen as added and now gets applied.
    const second = resolveAppliedDelta(first.applied, new Set(['a']), () => true);
    expect(second.added).toEqual(['a']);
    expect(second.applied.has('a')).toBe(true);
  });

  it('always processes removals regardless of load state', () => {
    const r = resolveAppliedDelta(new Set(['a', 'b']), new Set(['a']), () => true);
    expect(r.removed).toEqual(['b']);
    expect(r.applied.has('b')).toBe(false);
    expect(r.applied.has('a')).toBe(true);
  });

  it('records the previously-applied set minus removals plus newly-loaded adds', () => {
    // prev applied {a}; now select {a,b,c} but only b is loaded.
    const r = resolveAppliedDelta(
      new Set(['a']),
      new Set(['a', 'b', 'c']),
      (id) => id === 'b',
    );
    expect(r.added.sort()).toEqual(['b']); // only loaded add applied
    expect(r.removed).toEqual([]);
    expect(Array.from(r.applied).sort()).toEqual(['a', 'b']); // c deferred
  });
});
