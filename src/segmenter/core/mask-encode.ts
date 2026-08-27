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
