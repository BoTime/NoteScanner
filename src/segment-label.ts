import type { ViewerSegment } from './types';

/** Default menu label. Deliberately domain-neutral — hosts override via
 *  the `formatSegmentLabel` prop. */
export function formatSegmentLabelDefault(segment: ViewerSegment, index: number): string {
  return `Segment #${segment.index ?? index + 1}`;
}
