import { afterEach, describe, it, expect, vi } from 'vitest';
import { bytesToDataUrl, encodeMaskPng, maskToRgba } from './mask-encode';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('maskToRgba', () => {
  it('paints covered pixels opaque white and leaves the rest fully transparent', () => {
    // SegmentViewer's decoder counts a pixel as covered when alpha > 0 AND
    // red > 0, so this is the exact encoding it reads back.
    const rgba = maskToRgba(Uint8Array.from([0, 1]), 2, 1);
    expect(Array.from(rgba)).toEqual([0, 0, 0, 0, 255, 255, 255, 255]);
  });

  it('produces width * height * 4 bytes', () => {
    expect(maskToRgba(new Uint8Array(12), 4, 3)).toHaveLength(48);
  });

  it('rejects a coverage array that does not match the dimensions', () => {
    expect(() => maskToRgba(new Uint8Array(5), 2, 3)).toThrow(/does not match/);
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
    const decoded = atob(url.slice('data:image/png;base64,'.length));
    expect(decoded).toHaveLength(bytes.length);
    expect(decoded.charCodeAt(0)).toBe(65);
    expect(decoded.charCodeAt(decoded.length - 1)).toBe(65);
  });

  it('encodes an empty payload without throwing', () => {
    expect(bytesToDataUrl(new Uint8Array(0), 'image/png')).toBe('data:image/png;base64,');
  });
});

describe('encodeMaskPng', () => {
  it('draws the mask onto an OffscreenCanvas and returns a PNG data URL', async () => {
    const putImageData = vi.fn();
    const convertToBlob = vi.fn(async () => ({
      arrayBuffer: async () => Uint8Array.from([104, 105]).buffer,
    }));
    const constructed: Array<[number, number]> = [];

    vi.stubGlobal(
      'ImageData',
      class {
        constructor(
          readonly data: Uint8ClampedArray,
          readonly width: number,
          readonly height: number,
        ) {}
      },
    );
    vi.stubGlobal(
      'OffscreenCanvas',
      class {
        constructor(width: number, height: number) {
          constructed.push([width, height]);
        }
        getContext() {
          return { putImageData };
        }
        convertToBlob = convertToBlob;
      },
    );

    const url = await encodeMaskPng(Uint8Array.from([0, 1]), 2, 1);

    expect(constructed).toEqual([[2, 1]]);
    expect(putImageData).toHaveBeenCalledTimes(1);
    expect(convertToBlob).toHaveBeenCalledWith({ type: 'image/png' });
    expect(url).toBe('data:image/png;base64,aGk=');
  });

  it('fails loudly when no 2d context is available', async () => {
    vi.stubGlobal('ImageData', class {});
    vi.stubGlobal(
      'OffscreenCanvas',
      class {
        getContext() {
          return null;
        }
      },
    );

    await expect(encodeMaskPng(new Uint8Array(1), 1, 1)).rejects.toThrow(/2d context/);
  });
});
