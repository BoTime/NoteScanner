/**
 * `@sambacollab/segment-viewer/segmenter` — an OPTIONAL subpath.
 *
 * Importing it pulls in `@huggingface/transformers`, which is why it is a
 * separate export and why that package is an optional peer. Consumers of
 * `@sambacollab/segment-viewer` or `.../core` pull in nothing at runtime, and
 * `scripts/smoke-build.mjs` fails the build if that ever stops being true.
 */
export * from './core';
export {
  createSegmenter,
  isWebGPUAvailable,
  type CreateSegmenterConfig,
  type Segmenter,
} from './createSegmenter';
