import { createCanvas2DRenderer } from './canvas2d';
import { createWebGL2Renderer, probeWebGL2Support } from './webgl2';
import type { Renderer } from './types';

export type { Renderer, RendererFactory, Scene, ScenePoint } from './types';
export { createCanvas2DRenderer } from './canvas2d';
export { createWebGL2Renderer } from './webgl2';

/**
 * The backend `SegmentViewer` uses when the caller passes no `renderer` prop.
 *
 * The choice is made by actually attempting a context (and a shader compile, a
 * link, and an R8 framebuffer-completeness check — see `probeWebGL2Support`),
 * never by sniffing for `'WebGL2RenderingContext' in window`: a browser can
 * expose the constructor and still refuse a context on a blocklisted driver or
 * after a lost GPU process. Probing for real is what stops `SegmentViewer` from
 * ever holding a half-dead WebGL renderer.
 */
export function createDefaultRenderer(): Renderer {
  return probeWebGL2Support() ? createWebGL2Renderer() : createCanvas2DRenderer();
}
