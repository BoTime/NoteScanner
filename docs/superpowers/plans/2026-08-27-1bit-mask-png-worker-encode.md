# 1-bit Indexed Mask PNG, Encoded Off the Main Thread — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the canvas-based 32-bit mask PNG encoder with a pure-JS 1-bit
indexed PNG writer and move the encode loop into the existing segmenter worker,
cutting the `mask-encode` phase from ~1,069 ms/mask to ~1 ms/mask and taking it
off the main thread entirely.

**Architecture:** `mask-encode.ts` becomes a canvas-free PNG writer — pack each
row MSB-first at 1 bit/px behind a filter-type-None byte, deflate through the
platform's `CompressionStream('deflate')` (zlib-wrapped, which is exactly PNG's
IDAT format), and emit signature + IHDR + PLTE + tRNS + IDAT + IEND with
table-driven CRC32. Being canvas-free is what lets the identical function run in
the worker, on the main thread and in Node. The worker then encodes right after
NMS, while it still holds the coverage arrays, and posts `{ maskUrl, area }`
with **no transfer list** — so no ~0.7 MB coverage buffer crosses the worker
boundary at all, and `createSegmenter` collapses to a synchronous map.

**Tech Stack:** TypeScript 5.9, Vitest 3 (node environment), `@playwright/test`
(new devDependency, browser layer only), Node 22 in CI. No new runtime
dependency — `CompressionStream` is a platform global.

**Spec:** `docs/superpowers/specs/2026-08-27-1bit-mask-png-worker-encode-design.md`

## Global Constraints

- **Zero runtime dependencies in the package core.** `scripts/smoke-build.mjs`
  enforces it. Do not add `fflate`, `pako`, or any other deflate library.
  `@playwright/test` is a **devDependency** only.
- **`encodeMaskPng`'s signature is frozen:**
  `encodeMaskPng(coverage: Uint8Array, width: number, height: number): Promise<string>`.
  No call site changes shape.
- **The decoder contract is frozen.** A covered pixel must decode to
  `(255, 255, 255, 255)` and an uncovered pixel to `(0, 0, 0, 0)`, so
  `SegmentViewer.tsx`'s existing `alpha > 0 && red > 0` predicate keeps working.
  **`src/SegmentViewer.tsx` is not modified by this plan.**
- **`maskUrl` stays a `string` data URL** that an `<img>` decodes natively.
- **PNG structure is fixed:** IHDR bit depth `1`, colour type `3` (indexed),
  compression `0`, filter `0`, interlace `0`; PLTE is exactly 6 bytes
  (`00 00 00`, `FF FF FF`); tRNS is exactly the single byte `0x00`; every row is
  filter type None (`0x00`); row stride is `1 + ceil(width / 8)` bytes.
- **Use `'deflate'`, never `'deflate-raw'`.** PNG's IDAT is a zlib stream.
- **No canvas anywhere in `mask-encode.ts`** — no `OffscreenCanvas`, no
  `document.createElement('canvas')`, no `ImageData`.
- **Out of scope:** M2 (256x256 encoding, issue #7) and M5/F4 (WebGL2 renderer,
  issue #9). Do not touch either.

## Verification scope — which command proves what

Read this before claiming any step passed.

- `npm run typecheck` (`tsc --noEmit`) has `include: ["src/**/*.ts",
  "src/**/*.tsx"]` and `exclude: [..., "playground"]`. It **does** cover
  `tests/fixtures/mask-cases.ts`, because `src/segmenter/core/mask-encode.test.ts`
  imports it and `tsc` pulls imported files into the program (verified). It does
  **not** cover `playwright.config.ts` or `tests/browser/mask-png.spec.ts` —
  nothing under `src/` imports them. Those two files are proven only by
  `npm run test:browser` actually running.
- `npm test` (`vitest run`) has `include: ['src/**/*.test.ts',
  'src/**/*.test.tsx', 'playground/**/*.test.ts']`. It never matches `tests/`,
  so adding the Playwright layer leaves `npm test`'s scope unchanged (AC9).
- `npm run smoke` (`npm run build && node scripts/smoke-build.mjs`) is what
  proves the zero-runtime-dependency promise survives (AC8).
- **AC7's "no transfer list" half is verified by review plus a grep, not by a
  unit test.** `src/segmenter/worker/segmenter.worker.ts` imports
  `@huggingface/transformers` and registers a `message` listener at module
  scope; there is no worker test harness in this repo and building one is not
  in scope. The observable half — that `createSegmenter` splices nothing in and
  that `mask-encode` arrives inside the worker's own report — **is** unit
  tested in Task 2. This is a deliberate, stated gap.
- **AC3 and AC4 are not gates on this branch.** They need a real WebGPU
  segmentation run; the spec's "Measurement" section is the recipe the
  developer runs after merge. No task here implements them.

## File structure

| File | Fate | Responsibility after this plan |
| --- | --- | --- |
| `src/segmenter/core/mask-encode.ts` | rewritten | Pure-JS 1-bit indexed PNG writer: row packing, CRC32, chunk framing, deflate, base64 |
| `src/segmenter/core/mask-encode.test.ts` | rewritten | Node round-trip + structural properties, inflating with Node's own zlib |
| `tests/fixtures/mask-cases.ts` | new | The mask fixtures **both** verification layers share, so they cannot drift |
| `src/segmenter/core/types.ts` | modified | `RawMask` -> `EncodedMask`; `done` response shape; two stale doc comments |
| `src/segmenter/worker/segmenter.worker.ts` | modified | Encodes after NMS; posts data URLs with no transfer list |
| `src/segmenter/createSegmenter.ts` | modified | Synchronous map over `message.masks`; timing passthrough |
| `src/segmenter/createSegmenter.test.ts` | modified | Updated for the new message shape |
| `playwright.config.ts` | new | `chromium` / `webkit` / `firefox` projects over `tests/browser` |
| `tests/browser/mask-png.spec.ts` | new | Real-browser decode through the viewer's own predicate |
| `package.json` | modified | `@playwright/test` devDependency, `test:browser` script |
| `.github/workflows/ci.yml` | modified | Browser install + browser test step |

**Breaking change, intentional:** `maskToRgba` is deleted. It is re-exported
from the `note-scanner/segmenter` subpath via `src/segmenter/core/index.ts`, so
removing it is a public API removal. The spec calls for it (nothing needs a
32-bit expansion any more) and the package is at `0.1.0`. Do not keep a
deprecated shim.

---

### Task 1: The 1-bit indexed PNG writer

Rewrites the encoder as pure JS and replaces its test with a real round-trip.
This task changes no call site — `encodeMaskPng` keeps its exact signature — so
it is reviewable and testable entirely on its own.

**Files:**
- Modify (full rewrite of the body): `src/segmenter/core/mask-encode.ts`
- Modify (full rewrite): `src/segmenter/core/mask-encode.test.ts`
- Create: `tests/fixtures/mask-cases.ts`
- Modify (delete one stale test): `src/segmenter/createSegmenter.test.ts:143-184`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `encodeMaskPng(coverage: Uint8Array, width: number, height: number): Promise<string>`
    — unchanged signature, exported from `src/segmenter/core/mask-encode.ts`.
  - `bytesToDataUrl(bytes: Uint8Array, mime: string): string` — unchanged,
    exported.
  - `maskToRgba` — **deleted**. Task 2 must not reference it.
  - `tests/fixtures/mask-cases.ts` exports
    `interface MaskCase { name: string; width: number; height: number; coverage: Uint8Array }`
    and `const MASK_CASES: readonly MaskCase[]`. Task 3 imports both.
  - A one-test hole in `createSegmenter.test.ts` that Task 2 Step 1 fills.

- [ ] **Step 1: Create the shared fixtures**

Both verification layers run the same masks, so they live in one file. Widths
7, 9 and 17 are the row-padding cases — the exact place hand-rolled 1-bit PNG
writers break.

Create `tests/fixtures/mask-cases.ts`:

```ts
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
```

- [ ] **Step 2: Write the failing test**

Replace the **entire** contents of `src/segmenter/core/mask-encode.test.ts`
with the following. The three `bytesToDataUrl` cases are carried over verbatim
— that function is untouched and its chunk seam is already covered.

The round-trip decodes with Node's own `zlib` (`DecompressionStream('deflate')`
for the IDAT, `zlib.crc32` for the chunk CRCs), which is genuinely independent
of our encoder: it proves the bytes are a real PNG, not that the encoder agrees
with itself.

```ts
import { describe, it, expect } from 'vitest';
import { crc32 as nodeCrc32 } from 'node:zlib';
import { bytesToDataUrl, encodeMaskPng } from './mask-encode';
import { MASK_CASES, type MaskCase } from '../../../tests/fixtures/mask-cases';

const DATA_URL_PREFIX = 'data:image/png;base64,';
const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

interface PngChunk {
  type: string;
  data: Uint8Array<ArrayBuffer>;
  crcOk: boolean;
}

interface ParsedPng {
  chunks: PngChunk[];
  byType: Map<string, PngChunk>;
  /** The inflated IDAT — i.e. the exact bytes the encoder handed to deflate. */
  rows: Uint8Array<ArrayBuffer>;
}

function decodeDataUrl(url: string): Uint8Array<ArrayBuffer> {
  expect(url.startsWith(DATA_URL_PREFIX)).toBe(true);
  return Uint8Array.from(Buffer.from(url.slice(DATA_URL_PREFIX.length), 'base64'));
}

/**
 * Chunk framing is `length(4) type(4) data(length) crc(4)`, and the CRC covers
 * type + data. Validated with Node's own `zlib.crc32`, deliberately not the
 * encoder's table.
 */
function parseChunks(png: Uint8Array<ArrayBuffer>): PngChunk[] {
  const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
  const chunks: PngChunk[] = [];
  let at = 8;
  while (at < png.length) {
    const length = view.getUint32(at);
    const covered = png.subarray(at + 4, at + 8 + length);
    chunks.push({
      type: String.fromCharCode(...png.subarray(at + 4, at + 8)),
      data: png.subarray(at + 8, at + 8 + length),
      crcOk: nodeCrc32(covered) === view.getUint32(at + 8 + length),
    });
    at += 12 + length;
  }
  return chunks;
}

async function inflate(bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function readPng(url: string): Promise<ParsedPng> {
  const png = decodeDataUrl(url);
  expect(Array.from(png.subarray(0, 8))).toEqual(PNG_SIGNATURE);
  const chunks = parseChunks(png);
  const byType = new Map(chunks.map((chunk) => [chunk.type, chunk]));
  const idat = byType.get('IDAT');
  if (!idat) throw new Error('the encoder produced no IDAT chunk');
  return { chunks, byType, rows: await inflate(idat.data) };
}

/** Bits are MSB-first within each byte, after a one-byte filter code. */
function unpackRows(rows: Uint8Array, width: number, height: number): Uint8Array {
  const stride = 1 + Math.ceil(width / 8);
  const coverage = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      coverage[y * width + x] = (rows[y * stride + 1 + (x >> 3)] >> (7 - (x & 7))) & 1;
    }
  }
  return coverage;
}

describe('encodeMaskPng round-trip', () => {
  const stride = (width: number) => 1 + Math.ceil(width / 8);

  for (const maskCase of MASK_CASES) {
    it(`reproduces ${maskCase.name} byte for byte`, async () => {
      const { width, height, coverage } = maskCase;
      const { rows } = await readPng(await encodeMaskPng(coverage, width, height));
      expect(Array.from(unpackRows(rows, width, height))).toEqual(Array.from(coverage));
    });

    it(`packs ${maskCase.name} into exactly height * (1 + ceil(width / 8)) bytes`, async () => {
      // The divide-by-32 size claim, stated structurally: 1 bit/px + a filter
      // byte per row, against the 4 bytes/px the canvas encoder was fed.
      const { width, height, coverage } = maskCase;
      const { rows } = await readPng(await encodeMaskPng(coverage, width, height));
      expect(rows.length).toBe(height * stride(width));
    });

    it(`writes filter type None and zero padding bits for ${maskCase.name}`, async () => {
      const { width, height, coverage } = maskCase;
      const { rows } = await readPng(await encodeMaskPng(coverage, width, height));
      const padBits = (8 - (width % 8)) % 8;
      for (let y = 0; y < height; y += 1) {
        expect(rows[y * stride(width)]).toBe(0);
        if (padBits === 0) continue;
        const lastByte = rows[y * stride(width) + Math.ceil(width / 8)];
        expect(lastByte & ((1 << padBits) - 1)).toBe(0);
      }
    });

    it(`emits valid CRCs for every chunk of ${maskCase.name}`, async () => {
      const { width, height, coverage } = maskCase;
      const { chunks } = await readPng(await encodeMaskPng(coverage, width, height));
      expect(chunks.map((chunk) => chunk.crcOk)).toEqual(chunks.map(() => true));
    });
  }
});

describe('encodeMaskPng structure', () => {
  // 17x5 is a padded width with more than one row, so it exercises the stride
  // arithmetic while still being small enough to read in a failure message.
  const sample: MaskCase = MASK_CASES.find((c) => c.name === 'random 17x5')!;

  it('emits the PNG chunks in spec order', async () => {
    const { chunks } = await readPng(
      await encodeMaskPng(sample.coverage, sample.width, sample.height),
    );
    expect(chunks.map((chunk) => chunk.type)).toEqual([
      'IHDR',
      'PLTE',
      'tRNS',
      'IDAT',
      'IEND',
    ]);
  });

  it('declares bit depth 1, colour type 3 and no interlacing in IHDR', async () => {
    const { byType } = await readPng(
      await encodeMaskPng(sample.coverage, sample.width, sample.height),
    );
    const ihdr = byType.get('IHDR')!.data;
    const view = new DataView(ihdr.buffer, ihdr.byteOffset, ihdr.byteLength);
    expect(ihdr).toHaveLength(13);
    expect(view.getUint32(0)).toBe(sample.width);
    expect(view.getUint32(4)).toBe(sample.height);
    expect(Array.from(ihdr.subarray(8))).toEqual([1, 3, 0, 0, 0]);
  });

  it('carries the decoder contract in PLTE and tRNS', async () => {
    // Index 1 is opaque white and index 0 is fully transparent, which is what
    // makes SegmentViewer's `alpha > 0 && red > 0` predicate reproduce the
    // coverage. tRNS is one byte long, so index 1 is opaque by spec.
    const { byType } = await readPng(
      await encodeMaskPng(sample.coverage, sample.width, sample.height),
    );
    expect(Array.from(byType.get('PLTE')!.data)).toEqual([0, 0, 0, 255, 255, 255]);
    expect(Array.from(byType.get('tRNS')!.data)).toEqual([0x00]);
  });

  it('rejects a coverage array that does not match the dimensions', async () => {
    await expect(encodeMaskPng(new Uint8Array(5), 2, 3)).rejects.toThrow(/does not match/);
  });
});

describe('bytesToDataUrl', () => {
  it('base64-encodes the bytes behind the given mime type', () => {
    expect(bytesToDataUrl(Uint8Array.from([104, 105]), 'image/png')).toBe(
      'data:image/png;base64,aGk=',
    );
  });

  it('encodes a payload larger than one chunk correctly', () => {
    // 0x8000 is the chunk size; go past it so a bug in the chunk seam shows.
    const bytes = new Uint8Array(0x8000 + 7).fill(65);
    const url = bytesToDataUrl(bytes, 'image/png');
    const decoded = atob(url.slice(DATA_URL_PREFIX.length));
    expect(decoded).toHaveLength(bytes.length);
    expect(decoded.charCodeAt(0)).toBe(65);
    expect(decoded.charCodeAt(decoded.length - 1)).toBe(65);
  });

  it('encodes an empty payload without throwing', () => {
    expect(bytesToDataUrl(new Uint8Array(0), 'image/png')).toBe('data:image/png;base64,');
  });
});
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `npm test -- src/segmenter/core/mask-encode.test.ts`

Expected: FAIL. The old canvas encoder is still in place, so `readPng` blows up
on the signature/chunk assertions (and under the node environment
`encodeMaskPng` cannot even find `ImageData`).

- [ ] **Step 4: Write the implementation**

Replace the **entire** contents of `src/segmenter/core/mask-encode.ts` with:

```ts
/**
 * Binary mask -> 1-bit indexed PNG data URL, in plain JS with no canvas.
 *
 * A mask carries one bit per pixel. Handing it to a canvas as 32-bit RGBA and
 * letting `convertToBlob` re-derive that fact cost ~1,069 ms per mask, which
 * was over half the wall clock of an everything-mode run. Writing the PNG
 * directly at bit depth 1 costs about a millisecond and produces roughly a
 * thirtieth of the bytes — which matters in production, not just here, because
 * these masks get stored and transferred.
 *
 * Being canvas-free is also what lets the identical function run inside the
 * worker, on the main thread, and in Node — the last of which is what makes
 * the round-trip testable at all.
 */

/** `89 50 4E 47 0D 0A 1A 0A` — the PNG magic. */
const PNG_SIGNATURE = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/**
 * Two entries: index 0 black, index 1 white. Paired with `TRANSPARENCY` below
 * this is the exact contract `SegmentViewer.tsx` decodes — a covered pixel
 * reads back `(255, 255, 255, 255)` and an uncovered one `(0, 0, 0, 0)`, so
 * its `alpha > 0 && red > 0` predicate reproduces the coverage unchanged.
 */
const PALETTE = Uint8Array.from([0, 0, 0, 255, 255, 255]);

/**
 * One byte, so it gives an alpha only for palette index 0. Entries past the
 * end of tRNS are opaque by spec, which is what makes index 1 opaque white
 * without spending a second byte saying so.
 */
const TRANSPARENCY = Uint8Array.from([0x00]);

/** The standard reflected `0xEDB88320` CRC-32 table, built once at load. */
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) {
    c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

/**
 * `length(4) type(4) data(length) crc(4)`, big-endian, with the CRC taken over
 * the type and the data but not the length.
 */
function chunk(type: string, data: Uint8Array): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i += 1) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

function header(width: number, height: number): Uint8Array<ArrayBuffer> {
  const ihdr = new Uint8Array(13);
  const view = new DataView(ihdr.buffer);
  view.setUint32(0, width);
  view.setUint32(4, height);
  ihdr[8] = 1; // bit depth: one bit per pixel
  ihdr[9] = 3; // colour type: indexed, so the bit is a palette index
  ihdr[10] = 0; // compression: deflate, the only value PNG defines
  ihdr[11] = 0; // filter: adaptive, the only value PNG defines
  ihdr[12] = 0; // interlace: none
  return ihdr;
}

/**
 * Packs the mask into PNG scanlines: `1 + ceil(width / 8)` bytes per row, a
 * leading `0x00` filter byte (filter type None — deflate already exploits the
 * row-to-row redundancy, and a real filter on 1-bit data mostly gets in the
 * way), then the row's bits MSB-first.
 *
 * Padding bits in a row's final byte stay zero. A decoder ignores them, but
 * leaving them dirty is exactly how a hand-rolled 1-bit writer starts
 * producing images that differ between engines.
 */
function packMaskRows(
  coverage: Uint8Array,
  width: number,
  height: number,
): Uint8Array<ArrayBuffer> {
  const pixels = width * height;
  if (coverage.length !== pixels) {
    throw new Error(
      `coverage length ${coverage.length} does not match ${width}x${height} (${pixels})`,
    );
  }

  const stride = 1 + Math.ceil(width / 8);
  const rows = new Uint8Array(stride * height);
  for (let y = 0; y < height; y += 1) {
    const rowStart = y * stride + 1;
    const src = y * width;
    for (let x = 0; x < width; x += 1) {
      if (coverage[src + x]) rows[rowStart + (x >> 3)] |= 0x80 >> (x & 7);
    }
  }
  return rows;
}

/**
 * IDAT is a zlib stream, so this is `'deflate'` and NOT `'deflate-raw'` —
 * the raw variant omits the zlib header and every decoder would reject it.
 *
 * `CompressionStream` is a platform global in Chrome 80+, Safari 16.4+,
 * Firefox 113+ and Node 22. Every browser with WebGPU has it, so this adds no
 * compatibility floor to a package that already requires WebGPU, and it keeps
 * the core's zero-runtime-dependency promise intact.
 */
async function deflate(bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function concatBytes(parts: readonly Uint8Array[]): Uint8Array<ArrayBuffer> {
  let total = 0;
  for (const part of parts) total += part.length;
  const out = new Uint8Array(total);
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.length;
  }
  return out;
}

/**
 * `btoa` takes a string, and `String.fromCharCode(...bytes)` blows the
 * argument limit somewhere north of 100k arguments. Chunking is the fix, and
 * the chunk seam is exactly where this would silently corrupt data, so it has
 * its own test.
 */
const BASE64_CHUNK = 0x8000;

export function bytesToDataUrl(bytes: Uint8Array, mime: string): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += BASE64_CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + BASE64_CHUNK));
  }
  return `data:${mime};base64,${btoa(binary)}`;
}

export async function encodeMaskPng(
  coverage: Uint8Array,
  width: number,
  height: number,
): Promise<string> {
  const idat = await deflate(packMaskRows(coverage, width, height));
  return bytesToDataUrl(
    concatBytes([
      PNG_SIGNATURE,
      chunk('IHDR', header(width, height)),
      chunk('PLTE', PALETTE),
      chunk('tRNS', TRANSPARENCY),
      chunk('IDAT', idat),
      chunk('IEND', new Uint8Array(0)),
    ]),
    'image/png',
  );
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `npm test -- src/segmenter/core/mask-encode.test.ts`
Expected: PASS — 103 tests (24 fixtures x 4 round-trip properties, 4 structure
cases, 3 `bytesToDataUrl` cases).

- [ ] **Step 6: Prove there is no canvas left (AC8)**

Run: `grep -niE "canvas|imagedata" src/segmenter/core/mask-encode.ts`
Expected: **no output**, exit status 1.

- [ ] **Step 7: Typecheck**

Run: `npm run typecheck`
Expected: clean. Note this also typechecks `tests/fixtures/mask-cases.ts`,
because the vitest spec imports it (see "Verification scope" above). Expect
errors here only in `createSegmenter.ts` if you accidentally left a
`maskToRgba` reference — there are none in the repo today.

- [ ] **Step 8: Delete the one test the rewrite invalidates**

`src/segmenter/createSegmenter.test.ts`'s
`it('encodes each surviving mask into a ViewerSegment', ...)` (lines 143-184)
stubs `OffscreenCanvas` and `ImageData` and asserts the exact data URL those
stubs produce. The new encoder ignores both stubs and returns a real PNG, so
this test now fails — and its subject, main-thread canvas encoding, is about to
stop existing entirely.

Delete that whole `it(...)` block, including both `vi.stubGlobal` calls inside
it. Leave the rest of the file untouched. Task 2 Step 1 puts two tests against
the new message shape in its place; this commit is knowingly one test lighter
in between.

- [ ] **Step 9: Run the whole suite**

Run: `npm test`
Expected: PASS, every file green. If anything else in
`createSegmenter.test.ts` fails, that is a real regression — stop and
investigate rather than deleting more tests.

- [ ] **Step 10: Commit**

```bash
git add src/segmenter/core/mask-encode.ts src/segmenter/core/mask-encode.test.ts src/segmenter/createSegmenter.test.ts tests/fixtures/mask-cases.ts
git commit -m "perf(mask-encode): write masks as 1-bit indexed PNGs without canvas"
```

---

### Task 2: Move the encode loop into the worker

The worker already holds every coverage array immediately before it transfers
them out, so encoding there deletes both the main-thread loop and the ~0.7 MB
per-mask transfer.

**Files:**
- Modify: `src/segmenter/core/types.ts` (`PHASE_ORDER` doc comment ~lines 3-7,
  `TimingReport.filterSubPhases` doc comment ~lines 84-90, `RawMask` ~lines
  128-132, `SegmenterResponse` ~line 145)
- Modify: `src/segmenter/worker/segmenter.worker.ts` (imports ~lines 25-38,
  the post-NMS block ~lines 259-283)
- Modify: `src/segmenter/createSegmenter.ts` (imports ~lines 1-12, `onMessage`
  ~lines 100-150)
- Test: `src/segmenter/createSegmenter.test.ts`

**Interfaces:**
- Consumes: `encodeMaskPng(coverage: Uint8Array, width: number, height: number): Promise<string>`
  from Task 1, re-exported through `src/segmenter/core/index.ts`.
- Produces:
  - `interface EncodedMask { maskUrl: string; area: number }` exported from
    `src/segmenter/core/types.ts`, replacing `RawMask`.
  - `SegmenterResponse`'s `done` variant:
    `{ type: 'done'; masks: EncodedMask[]; width: number; height: number; timings: TimingReport; counts: SegmentationCounts }`.
    `width`/`height` stay — they describe the result, and dropping them would
    be a gratuitous extra breaking change.

- [ ] **Step 1: Write the failing tests**

In `src/segmenter/createSegmenter.test.ts`:

(a) Change the type import on line 8 from `type RawMask,` to
`type EncodedMask,`, and line 51's signature to
`function doneMessage(masks: EncodedMask[] = []): SegmenterResponse {`.

(b) Task 1 Step 8 deleted the old
`it('encodes each surviving mask into a ViewerSegment', ...)`. Put these two
tests where it used to be — after
`it('carries the worker filterSubPhases through the rebuilt report', ...)` and
before `it('forwards every progress event to the callback', ...)`:

```ts
  it('maps each encoded mask into a ViewerSegment without re-encoding', async () => {
    // No canvas stub and no encoder stub: the worker already did the work, so
    // the main thread's whole job here is naming the segments.
    const segmenter = createSegmenter({ createWorker: spawn });
    const pending = segmenter.segment(fakeBitmap());
    FakeWorker.instances[0].emit({
      type: 'done',
      masks: [
        { maskUrl: 'data:image/png;base64,aGk=', area: 1 },
        { maskUrl: 'data:image/png;base64,eWE=', area: 2 },
      ],
      width: 2,
      height: 1,
      timings: createTimingAccumulator().report(1),
      counts: { raw: 3, afterFilter: 2, afterNms: 2 },
    });

    const result = await pending;
    expect(result.segments).toEqual([
      { id: 'segment-1', index: 1, maskUrl: 'data:image/png;base64,aGk=' },
      { id: 'segment-2', index: 2, maskUrl: 'data:image/png;base64,eWE=' },
    ]);
  });

  it('passes the worker mask-encode timing through, splicing nothing in', async () => {
    const workerTimings = createTimingAccumulator();
    workerTimings.record('mask-encode', 3);
    workerTimings.record('mask-encode', 5);

    const segmenter = createSegmenter({ createWorker: spawn });
    const pending = segmenter.segment(fakeBitmap());
    FakeWorker.instances[0].emit({
      type: 'done',
      masks: [{ maskUrl: 'data:image/png;base64,aGk=', area: 1 }],
      width: 2,
      height: 1,
      timings: workerTimings.report(42),
      counts: { raw: 3, afterFilter: 1, afterNms: 1 },
    });

    const result = await pending;
    // Two samples from the worker, verbatim. A main-thread splice would have
    // overwritten this with one sample (or zero).
    expect(result.timings.phases['mask-encode'].count).toBe(2);
    expect(result.timings.phases['mask-encode'].total).toBe(8);
    // totalMs is still the one field the main thread owns, so worker spawn and
    // bitmap transfer stay inside the number the results table reports.
    expect(result.timings.totalMs).not.toBe(42);
  });
```

Leave every other test in the file exactly as it is — including
`'resolves with counts, timings and a mask-encode phase'`, whose
`expect(result.timings.phases['mask-encode'].count).toBe(0)` still holds
because the worker's report zero-fills every phase it never recorded.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm test -- src/segmenter/createSegmenter.test.ts`
Expected: FAIL — `EncodedMask` is not exported yet, and `createSegmenter` still
splices its own `mask-encode` summary over the worker's.

- [ ] **Step 3: Update the shared types**

In `src/segmenter/core/types.ts`:

Replace the `PHASE_ORDER` doc comment (lines 3-7) with:

```ts
/**
 * The phases the segmenter times, in the order the playground's results table
 * renders them. Every one of them happens inside the worker — including
 * `mask-encode`, which used to run on the main thread.
 */
```

Replace the `TimingReport.filterSubPhases` doc comment (lines 84-90) with:

```ts
  /**
   * The `filter` stage broken down. Required, not optional: the worker's whole
   * report is passed through, and requiring the field makes the compiler catch
   * a dropped passthrough across the worker boundary.
   */
```

Replace `RawMask` (lines 128-132) with:

```ts
/**
 * A surviving mask as the worker posts it back: already a PNG data URL, not a
 * coverage buffer. Encoding in the worker is what removes both the ~0.7 MB
 * per-mask transfer and the main-thread encode loop.
 */
export interface EncodedMask {
  maskUrl: string;
  area: number;
}
```

And in `SegmenterResponse`'s `done` variant, change `masks: RawMask[];` to
`masks: EncodedMask[];`.

- [ ] **Step 4: Encode in the worker**

In `src/segmenter/worker/segmenter.worker.ts`, change the import block from
`'../core'` so that `type RawMask,` becomes `type EncodedMask,` and
`encodeMaskPng,` is added to the value imports. The list is alphabetised;
`encodeMaskPng` goes after `dedupeMasks`:

```ts
import {
  batchPoints,
  buildPointGrid,
  createTimingAccumulator,
  dedupeMasks,
  encodeMaskPng,
  stabilityScore,
  thresholdMask,
  type BinaryMask,
  type EncodedMask,
  type FilterSubstep,
  type SegmentationPhase,
  type SegmenterOptions,
  type SegmenterRequest,
  type SegmenterResponse,
} from '../core';
```

Then replace everything from `const masks: RawMask[] = kept.map(...)` down to
the end of the `post({ type: 'done', ... })` call — that is the block currently
at lines 259-283, ending just before the `} catch (error) {` — with:

```ts
    // ---- mask-encode: right here, while the coverage arrays are still local.
    // ---- `phase` is set first so a `CompressionStream` failure surfaces as
    // ---- SegmenterFailure('mask-encode', ...) through the catch below.
    phase = 'mask-encode';
    const masks: EncodedMask[] = [];
    for (const index of kept) {
      const encodeStarted = performance.now();
      const maskUrl = await encodeMaskPng(
        candidates[index].coverage,
        originalWidth,
        originalHeight,
      );
      timings.record('mask-encode', performance.now() - encodeStarted);
      masks.push({ maskUrl, area: candidates[index].area });
    }

    // No transfer list, deliberately: `masks` is now strings. Not one coverage
    // buffer crosses the worker boundary any more.
    post({
      type: 'done',
      masks,
      width: originalWidth,
      height: originalHeight,
      timings: timings.report(performance.now() - startedAt),
      counts: {
        raw: rawCount,
        afterFilter: candidates.length,
        afterNms: masks.length,
      },
    });
```

- [ ] **Step 5: Collapse the main-thread loop**

In `src/segmenter/createSegmenter.ts`:

Change the import block (lines 1-12) to drop `encodeMaskPng`, `summarizePhase`
and the now-unused `ViewerSegment` import entirely:

```ts
import {
  DEFAULT_SEGMENTER_OPTIONS,
  SegmenterFailure,
  type SegmentationResult,
  type SegmenterOptions,
  type SegmenterProgress,
  type SegmenterRequest,
  type SegmenterResponse,
} from './core';
```

(Delete the `import type { ViewerSegment } from '../types';` line.)

Change `const onMessage = async (event: ...)` to `const onMessage = (event: ...)`
— nothing in it awaits any more — and replace the body from `detach();` at line
113 through the closing `}` of the `catch` block at line 149 with:

```ts
          detach();
          // Nothing to do but name the segments: the worker encoded every mask
          // before it posted. No try/catch either — an encode failure now
          // arrives as the worker's own `error` message, already carrying
          // `phase: 'mask-encode'`.
          resolve({
            segments: message.masks.map((mask, i) => ({
              id: `segment-${i + 1}`,
              index: i + 1,
              maskUrl: mask.maskUrl,
            })),
            timings: {
              // Verbatim from the worker: every phase, `mask-encode` included,
              // is measured on that side now.
              ...message.timings,
              // Except wall clock, which is measured from here so worker spawn
              // and bitmap transfer are inside the number the table reports.
              totalMs: performance.now() - startedAt,
            },
            counts: message.counts,
          });
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `npm test -- src/segmenter/createSegmenter.test.ts`
Expected: PASS, all 13 tests (2 `isWebGPUAvailable`, 11 `createSegmenter`).

- [ ] **Step 7: Prove no coverage buffer crosses the boundary (AC7)**

The worker has no test harness in this repo (it imports
`@huggingface/transformers` and registers a module-scope listener), so this
half of AC7 is verified structurally rather than by a unit test.

Run: `grep -n "buffer as ArrayBuffer" src/segmenter/worker/segmenter.worker.ts`
Expected: **no output**, exit status 1 — the transfer list is gone.

Run: `grep -rn "RawMask\|maskToRgba" src playground scripts`
Expected: **no output** — both names are gone from the repo.

Then read the `post({ type: 'done', ... })` call by eye and confirm it is
called with **one argument**: no second `[...]` transfer array.

- [ ] **Step 8: Typecheck and run the whole suite**

Run: `npm run typecheck && npm test`
Expected: both clean. The typecheck is the real gate on the worker file, which
has no tests.

- [ ] **Step 9: Confirm the package still has zero runtime dependencies (AC8)**

Run: `npm run smoke`
Expected: PASS with no `unresolved external` or `imports @huggingface/transformers`
lines.

- [ ] **Step 10: Commit**

```bash
git add src/segmenter/core/types.ts src/segmenter/worker/segmenter.worker.ts src/segmenter/createSegmenter.ts src/segmenter/createSegmenter.test.ts
git commit -m "perf(segmenter): encode masks in the worker and stop transferring coverage"
```

---

### Task 3: Verify the PNG decodes in Chromium, WebKit and Firefox

`tRNS` on an indexed PNG is well supported, but the failure mode if an engine
mishandled it would be a silently blank or silently solid mask — so it gets a
real-browser check rather than an assumption. No dev server and no WebGPU are
involved: the masks are encoded in Node and handed into the page as data URLs,
which is exactly what lets all three engines run headless in CI.

**Files:**
- Create: `playwright.config.ts`
- Create: `tests/browser/mask-png.spec.ts`
- Modify: `package.json` (`devDependencies`, `scripts`)
- Modify: `.github/workflows/ci.yml` (lines 17-20)

**Interfaces:**
- Consumes: `encodeMaskPng` from `src/segmenter/core/mask-encode.ts` (Task 1)
  and `MASK_CASES` / `MaskCase` from `tests/fixtures/mask-cases.ts` (Task 1).
- Produces: an `npm run test:browser` script. Nothing in `src/` depends on it.

- [ ] **Step 1: Add the dependency and the script**

Run:

```bash
npm install --save-dev --save-exact @playwright/test
```

Then add to `package.json`'s `scripts`, immediately after the `"test:watch"`
line:

```json
    "test:browser": "playwright test",
```

Confirm `@playwright/test` landed under `devDependencies` and **not**
`dependencies`.

- [ ] **Step 2: Write the Playwright config**

Create `playwright.config.ts`:

```ts
import { defineConfig, devices } from '@playwright/test';

/**
 * Browser-only verification of the 1-bit indexed mask PNG.
 *
 * There is no `webServer` and no WebGPU here on purpose: the spec encodes the
 * masks in Node and hands them into the page as data URLs, so all three
 * engines run headless with nothing to serve. `npm test` (vitest) never sees
 * these files — its `include` only matches `src/` and `playground/`.
 */
export default defineConfig({
  testDir: './tests/browser',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  reporter: process.env.CI ? 'github' : 'list',
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
  ],
});
```

- [ ] **Step 3: Write the browser spec**

Create `tests/browser/mask-png.spec.ts`:

```ts
import { expect, test, type Page } from '@playwright/test';
import { encodeMaskPng } from '../../src/segmenter/core/mask-encode';
import { MASK_CASES } from '../fixtures/mask-cases';

/**
 * `SegmentViewer.tsx`'s decoder, run verbatim inside the page: draw the mask
 * PNG onto a canvas and count a pixel covered when `alpha > 0 && red > 0`.
 * A data-URL image does not taint the canvas, so `getImageData` is readable,
 * and drawing at the image's natural size means no resampling.
 */
async function decodeInPage(
  page: Page,
  maskUrl: string,
  width: number,
  height: number,
): Promise<number[]> {
  return page.evaluate(
    ({ maskUrl, width, height }) =>
      new Promise<number[]>((resolve, reject) => {
        const img = new Image();
        img.onerror = () => reject(new Error('this engine refused to decode the mask PNG'));
        img.onload = () => {
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d', { willReadFrequently: true });
          if (!ctx) {
            reject(new Error('no 2d context'));
            return;
          }
          ctx.drawImage(img, 0, 0, width, height);
          const data = ctx.getImageData(0, 0, width, height).data;
          const coverage: number[] = [];
          for (let i = 0; i < data.length; i += 4) {
            coverage.push(data[i + 3] > 0 && data[i] > 0 ? 1 : 0);
          }
          resolve(coverage);
        };
        img.src = maskUrl;
      }),
    { maskUrl, width, height },
  );
}

test.beforeEach(async ({ page }) => {
  // An empty document is all the spec needs; the images are data URLs.
  await page.setContent('<!doctype html><meta charset="utf-8"><title>mask decode</title>');
});

for (const maskCase of MASK_CASES) {
  test(`decodes ${maskCase.name} back to the exact coverage`, async ({ page }) => {
    const maskUrl = await encodeMaskPng(maskCase.coverage, maskCase.width, maskCase.height);
    const decoded = await decodeInPage(page, maskUrl, maskCase.width, maskCase.height);
    expect(decoded).toEqual(Array.from(maskCase.coverage));
  });
}

test('reports the natural dimensions the encoder declared in IHDR', async ({ page }) => {
  // A width that is not a multiple of 8 is the case where a bad IHDR would
  // otherwise hide behind correct-looking pixels.
  const maskCase = MASK_CASES.find((entry) => entry.name === 'random 17x5')!;
  const maskUrl = await encodeMaskPng(maskCase.coverage, maskCase.width, maskCase.height);
  const size = await page.evaluate(
    (url) =>
      new Promise<{ width: number; height: number }>((resolve, reject) => {
        const img = new Image();
        img.onerror = () => reject(new Error('decode failed'));
        img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
        img.src = url;
      }),
    maskUrl,
  );
  expect(size).toEqual({ width: maskCase.width, height: maskCase.height });
});
```

- [ ] **Step 4: Install the browsers and run the spec**

Run:

```bash
npx playwright install --with-deps
npm run test:browser
```

Expected: PASS — 25 tests x 3 projects = 75 passed. A failure in `webkit` or
`firefox` only is the `tRNS` risk this spec exists to catch; report it rather
than working around it.

- [ ] **Step 5: Confirm `npm test`'s scope is unchanged (AC9)**

Run: `npm test`
Expected: PASS, and the reported file list contains **no** path under `tests/`.
Vitest's `include` is `['src/**/*.test.ts', 'src/**/*.test.tsx',
'playground/**/*.test.ts']`, so it cannot match the new spec.

- [ ] **Step 6: Wire it into CI**

In `.github/workflows/ci.yml`, replace the step list from `- run: npm run test`
onward so the job reads:

```yaml
      - run: npm install
      - run: npm run typecheck
      - run: npm run test
      - run: npx playwright install --with-deps
      - run: npm run test:browser
      - run: npm run build
```

The browser install goes after the vitest run so a plain unit-test failure
still reports in seconds rather than after a browser download.

- [ ] **Step 7: Commit**

```bash
git add playwright.config.ts tests/browser/mask-png.spec.ts package.json package-lock.json .github/workflows/ci.yml
git commit -m "test(mask-encode): verify the 1-bit PNG decodes in chromium, webkit and firefox"
```

---

## Acceptance criteria coverage

| AC | Where it is proven |
| --- | --- |
| AC1 round-trip byte-identical | Task 1 Step 5 (Node inflate + unpack over all 24 fixtures) and Task 3 Step 4 (real `<img>` decode, all 3 engines) |
| AC2 decodes in Chrome/Safari/Firefox | Task 3 Steps 4 and 6 |
| AC3 `mask-encode` total before/after | **Not in this branch.** Spec's "Measurement" recipe, run by the developer on a WebGPU machine, posted on issue #5 |
| AC4 main thread unblocked | **Not in this branch.** Same recipe |
| AC5 decoder contract preserved | Task 1 Step 2's PLTE/tRNS test (structural) and Task 3 Step 3's spec, which applies `alpha > 0 && red > 0` literally. `SegmentViewer.tsx` is not modified |
| AC6 structurally a 1-bit indexed PNG | Task 1 Step 2: IHDR `[1, 3, 0, 0, 0]`, PLTE 6 bytes, tRNS `[0x00]`, every CRC via Node's `zlib.crc32`, zero padding bits, IDAT input length `height * (1 + ceil(width / 8))` |
| AC7 encode in the worker, no transfer list | Task 2 Steps 6 (unit-tested: no splice, worker timing passthrough) and 7 (grep — the worker itself has no test harness; stated gap) |
| AC8 canvas-free, zero runtime deps | Task 1 Step 6 (grep) and Task 2 Step 9 (`npm run smoke`) |
| AC9 `npm test` scope unchanged, CI runs browsers | Task 3 Steps 5 and 6 |
