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
}

export type RendererFactory = () => Renderer;
