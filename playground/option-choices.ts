/**
 * The option values the playground offers in its selects.
 *
 * Shared by the Segment view and the Compare tab deliberately: both drive the
 * same `SegmenterOptions`, and a value offered by one tab but not the other is
 * a bug nobody notices until a hand run cannot reproduce a swept row.
 */

export const DTYPE_CHOICES = ['fp32', 'fp16'] as const;

/**
 * 64 is offered even though `DEFAULT_SWEEP_CONFIG` excludes it: issue #12
 * measured it dying inside `post_process_masks`, and re-confirming that by
 * hand should not require editing code.
 */
export const BATCH_SIZE_CHOICES = [8, 16, 32, 64] as const;

/** The one axis prior probes showed actually moves everything-mode cost. */
export const POINTS_PER_SIDE_CHOICES = [16, 32] as const;
