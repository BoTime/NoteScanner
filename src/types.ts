import type { RendererFactory } from './renderer';
import type { ViewerStatus } from './core';

export interface Point {
  x: number;
  y: number;
}

export interface ViewerSegment {
  id: string;
  maskUrl: string;
  /** Menu label number; falls back to array position + 1. */
  index?: number;
  /** Menu item state. */
  excluded?: boolean;
}

export interface SegmentViewerProps {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  segments: ViewerSegment[];
  initialSelectedIds: Set<string>;
  onSelectionChange: (selected: Set<string>) => void;
  onCreateSegment: (points: Point[]) => Promise<void>;
  /** CSS max-height applied to the canvas so it stays in the viewport. */
  maxHeight?: string;
  /**
   * Fetch fresh mask URLs for the current image (presigned URLs expire after
   * 300s). Returns id → maskUrl. Called at most ONCE per load, only when a
   * presigned mask fetch fails. Omit to disable retry.
   */
  onRefreshMaskUrls?: () => Promise<Map<string, string>>;
  /** 'loading' until the base image AND every mask are done; 'error' on failure. */
  onStatusChange?: (status: ViewerStatus) => void;
  /** Menu label for a segment. Defaults to `Segment #N`. */
  formatSegmentLabel?: (segment: ViewerSegment, index: number) => string;
  /** Painting backend. Defaults to Canvas2D. */
  renderer?: RendererFactory;
  /** Passed through to the root element. */
  className?: string;
}
