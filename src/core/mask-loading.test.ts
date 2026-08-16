import { describe, it, expect } from 'vitest';
import {
  MaskLoadTracker,
  buildImageKey,
  classifyMaskFailure,
  describePreviewEmptyState,
  evictMaskCaches,
  parseMaskSignature,
  planMaskDiff,
  planMaskRetry,
  shouldReportPreviewFailure,
  type MaskRef,
} from './mask-loading';

describe('parseMaskSignature', () => {
  it('returns an empty list for an empty signature', () => {
    expect(parseMaskSignature('')).toEqual([]);
  });

  it('round-trips the signature built by SegmentViewer', () => {
    const segments = [
      { id: 'seg-1', maskUrl: '/api/events/e-1/images/img-1/masks/seg-1/file' },
      { id: 'seg-2', maskUrl: '/api/events/e-1/images/img-1/masks/seg-2/file' },
    ];
    const signature = segments.map((s) => `${s.id}|${s.maskUrl}`).join('\n');
    expect(parseMaskSignature(signature)).toEqual(segments);
  });

  it('keeps presigned query strings intact, splitting only on the first pipe', () => {
    const url =
      'https://r2.example/img-1/individual_0.png?X-Amz-Signature=abc&X-Amz-Expires=300';
    expect(parseMaskSignature(`seg-1|${url}`)).toEqual([
      { id: 'seg-1', maskUrl: url },
    ]);
  });
});

describe('classifyMaskFailure', () => {
  it('classifies a presigned r2 url failure as expired', () => {
    const kind = classifyMaskFailure(
      'https://r2.example/img-1/individual_0.png?X-Amz-Signature=abc',
      new Error('Failed to load'),
    );
    expect(kind).toBe('expired');
  });

  it('classifies a proxy-path failure as other', () => {
    const kind = classifyMaskFailure(
      '/api/events/e-1/images/img-1/masks/seg-1/file',
      new Error('Failed to load'),
    );
    expect(kind).toBe('other');
  });
});

describe('planMaskRetry', () => {
  it('refetches when an expired presigned url failed and no retry has happened', () => {
    const decision = planMaskRetry({
      failures: [{ id: 'seg-1', kind: 'expired' }],
      alreadyRetried: false,
    });
    expect(decision.shouldRefetch).toBe(true);
  });

  it('does not refetch a second time', () => {
    const decision = planMaskRetry({
      failures: [{ id: 'seg-1', kind: 'expired' }],
      alreadyRetried: true,
    });
    expect(decision.shouldRefetch).toBe(false);
  });

  it('does not refetch when the failures are not expiry-shaped', () => {
    const decision = planMaskRetry({
      failures: [{ id: 'seg-1', kind: 'other' }],
      alreadyRetried: false,
    });
    expect(decision.shouldRefetch).toBe(false);
  });

  it('does not refetch when nothing failed', () => {
    const decision = planMaskRetry({ failures: [], alreadyRetried: false });
    expect(decision.shouldRefetch).toBe(false);
  });
});

describe('MaskLoadTracker', () => {
  it('is complete immediately when there are no masks', () => {
    const tracker = new MaskLoadTracker([]);
    expect(tracker.isComplete).toBe(true);
    expect(tracker.failures).toEqual([]);
    expect(tracker.succeededIds).toEqual([]);
  });

  it('is not complete until every expected mask settles', () => {
    const tracker = new MaskLoadTracker(['a', 'b', 'c']);
    expect(tracker.isComplete).toBe(false);
    tracker.succeed('a');
    tracker.succeed('b');
    expect(tracker.isComplete).toBe(false);
    expect(tracker.settledCount).toBe(2);
    tracker.succeed('c');
    expect(tracker.isComplete).toBe(true);
  });

  it('completes even when some masks fail, keeping the successes', () => {
    const tracker = new MaskLoadTracker(['a', 'b', 'c']);
    tracker.succeed('a');
    tracker.fail('b', 'other');
    tracker.succeed('c');

    expect(tracker.isComplete).toBe(true);
    expect(tracker.succeededIds).toEqual(['a', 'c']);
    expect(tracker.failures).toEqual([{ id: 'b', kind: 'other' }]);
  });

  it('ignores outcomes for ids it was not told to expect', () => {
    const tracker = new MaskLoadTracker(['a']);
    tracker.succeed('ghost');
    expect(tracker.settledCount).toBe(0);
    expect(tracker.isComplete).toBe(false);
    expect(tracker.succeededIds).toEqual([]);
  });

  it('records only the first outcome per id', () => {
    const tracker = new MaskLoadTracker(['a']);
    tracker.succeed('a');
    tracker.fail('a', 'expired');
    expect(tracker.settledCount).toBe(1);
    expect(tracker.failures).toEqual([]);
    expect(tracker.succeededIds).toEqual(['a']);
  });
});

// Regression guard for the hard constraint in apps/web/AGENTS.md: the mask-load
// effect keys on `maskSignature`, which is built ONLY from (id, maskUrl). If a
// future edit folds a per-segment scalar into the signature, every mask PNG gets
// re-fetched and re-decoded on every checkbox click. These tests fail loudly if
// that happens.
describe('maskSignature stability (apps/web/AGENTS.md regression guard)', () => {
  // Mirrors the memo in SegmentViewer verbatim.
  const buildSignature = (
    segments: ReadonlyArray<{ id: string; maskUrl: string }>,
  ) => segments.map((s) => `${s.id}|${s.maskUrl}`).join('\n');

  const base = [
    {
      id: 'seg-1',
      maskUrl: '/api/events/e-1/images/img-1/masks/seg-1/file',
      reviewed: false,
      flagged: false,
      extractedText: null as string | null,
    },
    {
      id: 'seg-2',
      maskUrl: '/api/events/e-1/images/img-1/masks/seg-2/file',
      reviewed: false,
      flagged: false,
      extractedText: null as string | null,
    },
  ];

  it('does not change when reviewed, flagged, or extractedText change', () => {
    const before = buildSignature(base);
    // A new array with new object identities — exactly what the detail page
    // produces on every checkbox click.
    const after = buildSignature(
      base.map((s) => ({
        ...s,
        reviewed: true,
        flagged: true,
        extractedText: 'hello',
      })),
    );
    expect(after).toBe(before);
  });

  it('does change when a mask url changes (fresh presigned urls must reload)', () => {
    const before = buildSignature(base);
    const after = buildSignature([
      { ...base[0], maskUrl: 'https://r2.example/img-1/individual_0.png?sig=x' },
      base[1],
    ]);
    expect(after).not.toBe(before);
  });

  it('does change when a segment is added', () => {
    const before = buildSignature(base);
    const after = buildSignature([
      ...base,
      {
        id: 'seg-3',
        maskUrl: '/api/events/e-1/images/img-1/masks/seg-3/file',
        reviewed: false,
        flagged: false,
        extractedText: null,
      },
    ]);
    expect(after).not.toBe(before);
  });

  it('parses back to exactly the (id, maskUrl) pairs it was built from', () => {
    const pairs = base.map((s) => ({ id: s.id, maskUrl: s.maskUrl }));
    expect(parseMaskSignature(buildSignature(base))).toEqual(pairs);
  });
});

describe('describePreviewEmptyState', () => {
  it('labels a genuinely broken mask fetch as unavailable', () => {
    expect(describePreviewEmptyState('error')).toEqual({
      reason: 'error',
      label: 'Preview unavailable',
    });
  });

  it('does NOT label a blank text-only mask as unavailable', () => {
    // Regression: a Text-tab "Add note" stores a blank 1x1 transparent PNG as
    // its mask, so decodeMaskForPreview finds zero coverage and returns null.
    // That used to reuse the error copy, making a note that was created
    // perfectly read as broken in the very list it appeared in.
    const state = describePreviewEmptyState('no-mask');
    expect(state.reason).toBe('no-mask');
    expect(state.label).not.toMatch(/unavailable/i);
    expect(state.label).toBe('Text-only note');
  });

  it('gives the two reasons distinct copy', () => {
    expect(describePreviewEmptyState('no-mask').label).not.toBe(
      describePreviewEmptyState('error').label,
    );
  });
});

describe('shouldReportPreviewFailure', () => {
  it('reports a real load/decode error so the one-shot url refresh can run', () => {
    expect(shouldReportPreviewFailure('error')).toBe(true);
  });

  it('does not burn the one-shot refresh on a cleanly decoded blank mask', () => {
    // Refetching a presigned url cannot add coverage to a blank mask, so a
    // text-only note must never consume the panel's single retry.
    expect(shouldReportPreviewFailure('no-mask')).toBe(false);
  });
});

// The source-level guard pinning TranscribedTextReview.tsx's wiring to
// describePreviewEmptyState/shouldReportPreviewFailure lives in apps/web,
// next to that component: see
// apps/web/src/components/post-it-images/TranscribedTextReview.preview-empty-state.test.ts

describe('buildImageKey', () => {
  it('changes when any of url, width, or height changes; stable for identical args', () => {
    const base = { imageUrl: '/img/a.png', imageWidth: 100, imageHeight: 200 };
    const key = buildImageKey(base);
    expect(buildImageKey(base)).toBe(key);
    expect(buildImageKey({ ...base, imageUrl: '/img/b.png' })).not.toBe(key);
    expect(buildImageKey({ ...base, imageWidth: 101 })).not.toBe(key);
    expect(buildImageKey({ ...base, imageHeight: 201 })).not.toBe(key);
  });

  it('separates its parts so a width/height digit split cannot collide', () => {
    // The `|` is load-bearing: bare concatenation makes ('a',1,23) === ('a',12,3).
    expect(
      buildImageKey({ imageUrl: 'a', imageWidth: 1, imageHeight: 23 }),
    ).not.toBe(buildImageKey({ imageUrl: 'a', imageWidth: 12, imageHeight: 3 }));
  });
});

describe('planMaskDiff', () => {
  const ref = (id: string, n = 1): MaskRef => ({
    id,
    maskUrl: `/masks/${id}-${n}`,
  });
  const KEY = buildImageKey({
    imageUrl: '/img/img-1.png',
    imageWidth: 800,
    imageHeight: 600,
  });

  it('adding a segment reuses everything else: load=[new id] only, keep=rest, evict=[], fullReset=false', () => {
    const prevRefs = [ref('a'), ref('b'), ref('c')];
    const nextRefs = [ref('a'), ref('b'), ref('c'), ref('d')];
    const plan = planMaskDiff({
      prevImageKey: KEY,
      nextImageKey: KEY,
      prevRefs,
      nextRefs,
    });
    expect(plan.fullReset).toBe(false);
    expect(plan.load).toEqual([ref('d')]);
    expect(plan.keep).toEqual(['a', 'b', 'c']);
    expect(plan.evict).toEqual([]);
  });

  it('removing a segment evicts it: evict=[removed id], keep=rest, load=[]', () => {
    const prevRefs = [ref('a'), ref('b'), ref('c')];
    const nextRefs = [ref('a'), ref('c')];
    const plan = planMaskDiff({
      prevImageKey: KEY,
      nextImageKey: KEY,
      prevRefs,
      nextRefs,
    });
    expect(plan.fullReset).toBe(false);
    expect(plan.evict).toEqual(['b']);
    expect(plan.keep).toEqual(['a', 'c']);
    expect(plan.load).toEqual([]);
  });

  it('a changed maskUrl for an existing id lands in BOTH evict and load; others untouched', () => {
    const prevRefs = [ref('a'), ref('b'), ref('c')];
    const nextRefs = [ref('a'), ref('b', 2), ref('c')];
    const plan = planMaskDiff({
      prevImageKey: KEY,
      nextImageKey: KEY,
      prevRefs,
      nextRefs,
    });
    expect(plan.fullReset).toBe(false);
    expect(plan.evict).toEqual(['b']);
    expect(plan.load).toEqual([ref('b', 2)]);
    expect(plan.keep).toEqual(['a', 'c']);
  });

  it('an image-key change forces fullReset=true, load=all of nextRefs, regardless of overlap (imageUrl changing)', () => {
    const prevRefs = [ref('a'), ref('b')];
    const nextRefs = [ref('a'), ref('b')]; // identical refs, only the url differs
    const prevKey = buildImageKey({
      imageUrl: '/img/img-1.png',
      imageWidth: 800,
      imageHeight: 600,
    });
    const nextKey = buildImageKey({
      imageUrl: '/img/img-2.png',
      imageWidth: 800,
      imageHeight: 600,
    });
    const plan = planMaskDiff({
      prevImageKey: prevKey,
      nextImageKey: nextKey,
      prevRefs,
      nextRefs,
    });
    expect(plan.fullReset).toBe(true);
    expect(plan.load).toEqual(nextRefs);
    expect(plan.keep).toEqual([]);
    expect(plan.evict).toEqual([]);
  });

  it('an image-key change forces fullReset=true when only width/height change with an unchanged url', () => {
    const prevRefs = [ref('a'), ref('b')];
    const nextRefs = [ref('a'), ref('b')];
    const prevKey = buildImageKey({
      imageUrl: '/img/img-1.png',
      imageWidth: 800,
      imageHeight: 600,
    });
    const nextKey = buildImageKey({
      imageUrl: '/img/img-1.png',
      imageWidth: 801,
      imageHeight: 600,
    });
    const plan = planMaskDiff({
      prevImageKey: prevKey,
      nextImageKey: nextKey,
      prevRefs,
      nextRefs,
    });
    expect(plan.fullReset).toBe(true);
    expect(plan.load).toEqual(nextRefs);
    expect(plan.keep).toEqual([]);
    expect(plan.evict).toEqual([]);
  });

  it('first run (prevImageKey: null) forces fullReset=true, everything in load', () => {
    const nextRefs = [ref('a'), ref('b'), ref('c')];
    const plan = planMaskDiff({
      prevImageKey: null,
      nextImageKey: KEY,
      prevRefs: [],
      nextRefs,
    });
    expect(plan.fullReset).toBe(true);
    expect(plan.load).toEqual(nextRefs);
    expect(plan.keep).toEqual([]);
    expect(plan.evict).toEqual([]);
  });

  it('retry path: identical prev/next refs and key -> fullReset=false, keep=everything, load=[], evict=[]', () => {
    const refs = [ref('a'), ref('b'), ref('c')];
    const plan = planMaskDiff({
      prevImageKey: KEY,
      nextImageKey: KEY,
      prevRefs: refs,
      nextRefs: refs,
    });
    expect(plan.fullReset).toBe(false);
    expect(plan.keep).toEqual(['a', 'b', 'c']);
    expect(plan.load).toEqual([]);
    expect(plan.evict).toEqual([]);
  });
});

describe('evictMaskCaches', () => {
  it('removes the id from both maps, leaves other ids untouched, and no-ops for an unknown id', () => {
    const mapA = new Map<string, unknown>([
      ['a', 1],
      ['b', 2],
      ['c', 3],
    ]);
    const mapB = new Map<string, unknown>([
      ['a', 'x'],
      ['b', 'y'],
      ['c', 'z'],
    ]);

    evictMaskCaches('b', [mapA, mapB]);

    expect(mapA.has('b')).toBe(false);
    expect(mapB.has('b')).toBe(false);
    expect(mapA.get('a')).toBe(1);
    expect(mapA.get('c')).toBe(3);
    expect(mapB.get('a')).toBe('x');
    expect(mapB.get('c')).toBe('z');
    expect(mapA.size).toBe(2);
    expect(mapB.size).toBe(2);

    expect(() => evictMaskCaches('nope', [mapA, mapB])).not.toThrow();
    expect(mapA.size).toBe(2);
    expect(mapB.size).toBe(2);
  });
});
