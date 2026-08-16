export interface ScenePoint {
  x: number;
  y: number;
}

/**
 * A plain description of what to paint. The component owns state; the renderer
 * owns pixels. Everything here is data the component already holds.
 */
export interface Scene {
  /** Decoded base image, drawn first at natural size. */
  base: CanvasImageSource;
  imageWidth: number;
  imageHeight: number;
  /** Decoded masks by segment id. */
  masks: Map<string, { coverage: Uint8Array; area: number }>;
  /** Committed selection. */
  selectedIds: Set<string>;
  /** Hover preview; when set it replaces the selection's bright window. */
  hoveredId: string | null;
  /** In-progress polygon for draw mode; empty when not drawing. */
  draftPoints: ScenePoint[];
  /** Device pixel ratio to size the backing store by. */
  devicePixelRatio: number;
}

export interface Renderer {
  /** Bind to a canvas. Called once per canvas element. */
  init(canvas: HTMLCanvasElement): void;
  /** Paint the scene. Must be safe to call before init resolves a context. */
  draw(scene: Scene): void;
  /** Re-size the backing store to the given CSS size and dpr, then repaint. */
  resize(scene: Scene): void;
  /** Release GPU/bitmap resources. Safe to call twice. */
  dispose(): void;
  /**
   * Drop per-mask cached geometry for these ids. Safe to call with unknown ids.
   *
   * Optional on purpose: `renderer?: RendererFactory` is public package API, so
   * a third-party renderer must not break on a minor upgrade just because this
   * method was added. Callers must use `rendererRef.current?.evict?.(ids)`.
   *
   * Scope: clears derived per-id geometry ONLY (e.g. cached coverage/edge
   * index arrays) — never count buffers or `prevSelected`. `resolveAppliedDelta`'s
   * `removed` branch already subtracts an id's contribution from those when it
   * drops out of `scene.selectedIds`; subtracting again here would double-count
   * and drive the `Uint16Array` counts negative, underflowing to ~65535 and
   * permanently brightening those pixels.
   */
  evict?(ids: readonly string[]): void;
}

export type RendererFactory = () => Renderer;
