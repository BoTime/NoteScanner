import { describe, it, expect } from 'vitest';
import { formatSegmentLabelDefault } from './segment-label';

describe('formatSegmentLabelDefault', () => {
  it('uses the segment index when present', () => {
    expect(formatSegmentLabelDefault({ id: 'a', maskUrl: 'u', index: 7 }, 0)).toBe('Segment #7');
  });

  it('falls back to array position + 1 when index is absent', () => {
    expect(formatSegmentLabelDefault({ id: 'a', maskUrl: 'u' }, 2)).toBe('Segment #3');
  });

  it('never says the host application vocabulary', () => {
    const label = formatSegmentLabelDefault({ id: 'a', maskUrl: 'u' }, 0);
    expect(label.toLowerCase()).not.toContain('post' + '-it');
  });
});
