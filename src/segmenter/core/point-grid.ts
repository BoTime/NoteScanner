/**
 * SAM's automatic-mask prompt grid: `pointsPerSide ** 2` points at cell
 * centres, normalized to [0, 1] so the caller can scale them into whatever
 * coordinate space the model wants.
 *
 * Single crop layer only. Multi-crop is explicitly out of scope: it multiplies
 * the decode cost, and the decode cost is the thing this prototype exists to
 * measure.
 *
 * Row-major, x varying fastest, matching SAM's own `build_point_grid`.
 */
export function buildPointGrid(pointsPerSide: number): Array<[number, number]> {
  if (!Number.isInteger(pointsPerSide) || pointsPerSide < 1) {
    throw new Error(`pointsPerSide must be a positive integer, got ${pointsPerSide}`);
  }

  const points: Array<[number, number]> = [];
  for (let iy = 0; iy < pointsPerSide; iy += 1) {
    for (let ix = 0; ix < pointsPerSide; ix += 1) {
      points.push([(ix + 0.5) / pointsPerSide, (iy + 0.5) / pointsPerSide]);
    }
  }
  return points;
}

/**
 * Split a list into fixed-size chunks. Each chunk becomes one mask-decoder
 * call, which is what bounds peak GPU memory: a batch of P points produces
 * P * 3 low-resolution masks that all have to be resident at once.
 */
export function batchPoints<T>(items: readonly T[], batchSize: number): T[][] {
  if (!Number.isInteger(batchSize) || batchSize < 1) {
    throw new Error(`batchSize must be an integer of at least 1, got ${batchSize}`);
  }

  const batches: T[][] = [];
  for (let i = 0; i < items.length; i += batchSize) {
    batches.push(items.slice(i, i + batchSize));
  }
  return batches;
}
