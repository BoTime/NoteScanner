// Bound the cache so a long-lived tab (many navigations and refreshes mint
// fresh presigned mask URLs) cannot retain unbounded full-res decoded images.
const MAX_CACHED_IMAGES = 100;

const imageCache = new Map<string, Promise<HTMLImageElement>>();

/**
 * Load an image, deduplicating concurrent and subsequent loads of the same
 * URL through a module-level promise cache.
 *
 * The first caller for a URL pays the fetch + decode; every other caller — the
 * board and each row preview, which all receive the same full-res image URL —
 * awaits the same resolved image. On rejection the entry is evicted so a later
 * caller re-attempts (a presigned mask URL that 403s after its 300s TTL is
 * replaced by the one-shot refresh path, which hands back a fresh URL).
 */
export function loadImage(src: string): Promise<HTMLImageElement> {
  const cached = imageCache.get(src);
  if (cached) return cached;

  const promise = new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    // Masks come from R2 as absolute cross-origin presigned URLs. Without this
    // the decode canvas is tainted and getImageData throws SecurityError.
    image.crossOrigin = 'anonymous';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Failed to load ${src}`));
    image.src = src;
  });

  imageCache.set(src, promise);

  // Cap-on-insert eviction: a JS `Map` preserves insertion order, so the first
  // key is the oldest entry. Evict oldest-first while over the cap. This can
  // delete an entry before its rejection handler runs — the handler's
  // `imageCache.get(src) === promise` guard makes that a harmless no-op.
  while (imageCache.size > MAX_CACHED_IMAGES) {
    const oldestKey = imageCache.keys().next().value;
    if (oldestKey === undefined) break;
    imageCache.delete(oldestKey);
  }

  // Evict on rejection; a resolved image stays cached until the cap evicts it.
  // The rejection handler is attached synchronously so the returned promise
  // never surfaces as an unhandled rejection, and it does not alter the promise
  // callers receive.
  promise.then(undefined, () => {
    if (imageCache.get(src) === promise) imageCache.delete(src);
  });

  return promise;
}
