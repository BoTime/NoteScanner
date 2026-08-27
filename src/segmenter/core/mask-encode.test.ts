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
