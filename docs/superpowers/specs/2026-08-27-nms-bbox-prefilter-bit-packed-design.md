# N2 + N3: bbox prefilter and bit-packed NMS

Implement the bounding-box prefilter (N2) and bit-packed coverage comparison
(N3) in `src/segmenter/core/nms.ts`, with an opt-in differential A/B mode so the
real-image before/after numbers can be taken by hand in the playground.

N4 (the area-ratio early exit) is listed as optional on GitHub issue #4 and is
deliberately **out of scope** for this run.

## Why

`dedupeMasks` is the whole of the `nms` phase. It is a greedy largest-area-first
loop whose inner test is `pairwiseIoU`, and `pairwiseIoU` walks two
full-resolution `Uint8Array` coverage buffers byte by byte. At 32 points per
side there are ~617 surviving candidates of ~0.7 MB each, so the loop performs
on the order of 15.1 billion byte-pair reads. Two exact transformations remove
almost all of them:

- **N2, bbox prefilter** — most candidate pairs are nowhere near each other. A
  pair whose bounding boxes are disjoint on either axis has IoU 0 by definition,
  and can be rejected without touching a pixel.
- **N3, bit packing** — a pair that does overlap is compared 32 pixels at a
  time with `a & b` plus a popcount, instead of one pixel at a time.

Both are exact. Neither approximates, so no tolerance is involved anywhere in
this change and the kept set is identical to baseline by construction, not by
luck.

## Chosen approach (and the two rejected)

**Chosen — precompute inside `dedupeMasks`.** One pass per candidate turns its
`Uint8Array` coverage into a packed `Uint32Array` plus a bbox; the greedy loop
then rejects on bbox and, failing that, runs `a & b` + SWAR popcount over only
the word range the two bboxes share. `nms.ts` stays the module that owns the
algorithm, which is where its tests and its ported-from-the-API provenance
already live.

Cost: one extra linear pass — ~432 MB of byte reads against the ~15.1 billion
byte-pair operations it removes — and roughly 54 MB of transient packed arrays
at 32 pps. Both are function-local and collected on return.

**Rejected — pack at threshold time and carry packed coverage through the
pipeline.** It would avoid the extra pass and the allocation by making
`thresholdMask` emit packed masks, but `RawMask.coverage` crosses the worker
`postMessage` boundary and feeds `mask-encode`, so this ripples into the message
contract and the PNG encoder — which is exactly issue #5's scope, and the
256x256 move is issue #6's. It would trade this issue's stated
"depends on: nothing" property for a three-issue entanglement.

**Rejected — precompute in the worker and pass packed masks to a new
`dedupeMasksPacked`.** It moves algorithm detail into the worker, where none of
the NMS tests can reach it.

## Algorithm — `src/segmenter/core/nms.ts`

`dedupeMasks(masks, iouThreshold, width)` — `width` becomes a **required** third
parameter. This is a breaking change to the `note-scanner/segmenter` subpath
export, accepted deliberately at v0.1.0 with exactly one in-repo caller
(`src/segmenter/worker/segmenter.worker.ts`, which already has `originalWidth`
in scope at the call site), in exchange for the full 2-D bbox the issue
specifies. There is **no** optional-width fallback path: a quiet slow mode is
easy to end up in by accident, and a required parameter makes the compiler say
so instead.

`dedupeMasks` first builds, for each candidate, a private prepared record:

- `words: Uint32Array` of length `ceil(pixels / 32)`, with bit `p & 31` of word
  `p >>> 5` set for each covered pixel `p`.
- a bbox `{ x0, y0, x1, y1 }` derived from the flat index and `width`
  (`x = p % width`, `y = (p / width) | 0`).
- the mask's `area`, carried through unchanged.

Zero-area masks are marked empty and can never match anything, preserving
today's "a zero-area mask suppresses nothing" behaviour.

The greedy largest-area-first loop is **unchanged** in structure and in
tie-breaking: candidates are ordered by descending area with ties broken by
lower original index, and the kept indices are returned ascending. What changes
is only the per-pair test, in three steps:

1. **Bbox reject.** If the bboxes are disjoint on either axis
   (`a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0`) the IoU is 0 by
   definition and not a single pixel is touched.
2. **Word-range-limited intersection.** Otherwise intersect only the 32-bit
   words spanned by the shared row band: from `(y0 * width) >>> 5` through
   `((y1 + 1) * width - 1) >>> 5`, where `y0`/`y1` are the overlapping row
   range. That range is a *superset* of the pixels that could possibly overlap
   — words outside it contain no pixel set in both masks — so restricting to it
   is exact, not approximate.
   `intersection = sum of popcount(a.words[w] & b.words[w])` over that range.
3. **Union from areas.** `union = a.area + b.area - intersection`, using areas
   already known, so the union costs nothing extra.

The comparison remains `> iouThreshold`, **strictly**, so a mask at exactly the
threshold still survives. This is an existing pinned behaviour with tests either
side of it.

Every step is exact. No approximation, no tolerance.

`popcount` is the classic SWAR bit-twiddle, using unsigned shifts (`>>>`)
throughout so the high bit is never sign-extended, and it is **exported from
`nms.ts`** so it can be unit-tested directly.

**Public `pairwiseIoU` keeps its current byte-wise implementation and its
current two-argument signature.** It is published API, it is correct, and it is
now used by tests and by the reference path rather than by the hot loop.

## The A/B differential mode

The current byte-wise `dedupeMasks` body survives verbatim as
`dedupeMasksReference(masks, iouThreshold)` in `nms.ts` — same two-argument
shape it has today, still built on `pairwiseIoU`. It is the baseline that
"identical to baseline" is measured against, both in CI and in the app.

It is **not** part of the public surface. `src/segmenter/core/index.ts`
currently does `export * from './nms'`, which would re-export it and `popcount`
automatically, so that line narrows to an explicit named list — the
`BinaryMask` re-export, `pairwiseIoU`, `dedupeMasks` — leaving
`dedupeMasksReference` and `popcount` importable from `./nms` by the worker and
the tests but absent from `note-scanner/segmenter`. The project does not
permanently own a slow implementation as API.

The wiring:

- `SegmenterOptions` gains `compareNms: boolean`, defaulting to `false` in
  `DEFAULT_SEGMENTER_OPTIONS`.
- When set, the worker runs `dedupeMasksReference` **first**, then the fast
  `dedupeMasks`, over the identical candidate array; times each; and compares
  the two returned index arrays for exact equality (same length, same values in
  order — both are returned ascending).
- The `done` message carries
  `nmsComparison: { referenceMs: number; fastMs: number; identical: boolean }`,
  present only when the option was set.
- **Critical:** `timings.record('nms', ...)` still records **only** the fast
  path's elapsed time, so the results table never reports the doubled work. The
  reference time travels only in `nmsComparison`. The `progress` event for the
  `nms` phase likewise reports the fast path.
- `createSegmenter` passes the option through and surfaces the comparison on
  `SegmentationResult` (`nmsComparison?`), alongside `timings` and `counts`.
- `playground/SegmentView.tsx` gains a `compare-nms` checkbox beside the other
  option controls and, when a comparison came back, a row showing reference ms,
  fast ms, the speedup, and a pass/fail on set equality.

Off by default, so a production run pays nothing; the shipped bundle's only cost
is the dormant reference function.

**Measurement caveat.** The reference runs first, so the fast path sees a warmer
cache. At 617 candidates x ~0.7 MB the working set is far past any cache, so the
effect is small — but this is not a controlled A/B, and the resulting numbers
should not be quoted to two significant figures.

## Evidence strategy

`@playwright/test` is **not** a dependency of this repo and is **not** being
added. There is therefore no automated browser verification in this run, and
every acceptance criterion below is tagged `(non-ui)` — including the ones about
the playground panel, which are verified by reading
`playground/SegmentView.tsx`.

The two halves of "identical to baseline":

- **In CI** — the seeded differential test, which asserts index-identical
  results between `dedupeMasks` and `dedupeMasksReference` over several hundred
  overlapping full-resolution-shaped candidates across a range of thresholds.
- **By hand, after merge** — the developer runs the A/B panel this branch ships
  on two images at both operating points, reading off the `identical` flag and
  the before/after `nms` times. The existing `sample-file-input` control already
  allows loading a second image by hand, so no second sample image is added to
  the repo.

## Error handling

No new failure modes. A `width` that does not divide `coverage.length` is a
caller bug, not a runtime condition; the existing defensive `Math.min` over the
two coverage lengths carries into the word loop so a malformed pair still cannot
read past the end of either array.

## Testing

`src/segmenter/core/nms.test.ts` keeps **every** existing case — they now
exercise the new path, updated only to pass the required `width` argument — and
gains:

- `popcount` unit tests: `0` -> 0, `0xFFFFFFFF` -> 32, `0x80000000` -> 1, and
  every single-bit position 0..31. The classic SWAR bit-twiddle is easy to get
  subtly wrong on the high bit, which is exactly what the `0x80000000` and
  position-31 cases pin.
- bbox-disjoint pairs returning exactly the same kept set as the reference.
- masks whose coverage length is not a multiple of 32, proving the tail word's
  unused high bits are not counted.
- a seeded differential test generating several hundred overlapping
  full-resolution-shaped candidates, asserting `dedupeMasks` and
  `dedupeMasksReference` return index-identical results across a range of
  thresholds. Seeded, so a failure is reproducible.

`src/segmenter/createSegmenter.test.ts` covers the `compareNms` passthrough and
the `nmsComparison` field surviving the worker -> main-thread boundary, in the
same shape as the existing `filterSubPhases` passthrough test.

## Out of scope

- **N4**, the area-ratio early exit.
- Packing at threshold time, changing `RawMask.coverage`, or touching the worker
  message contract beyond adding `nmsComparison` (issue #5).
- Moving NMS to 256x256 (issue #6).
- Any change to `pairwiseIoU`'s implementation or signature.
- Producing the real-image before/after figures — done by hand after merge, with
  the panel this branch ships.

## Acceptance criteria

Every criterion below is `(non-ui)`. This repository has no `@playwright/test`
dependency, none is being added, and the vitest environment is `node`, so
nothing here is browser-verifiable by the pipeline — including AC6 and AC7,
which are verified by reading `playground/SegmentView.tsx` and
`src/segmenter/createSegmenter.ts` rather than by opening a browser.

- AC1 (non-ui) — **behaviour-preserving, exactly.** For the same candidates and
  threshold, `dedupeMasks` returns an index-identical kept set to
  `dedupeMasksReference` (the current byte-wise implementation, preserved
  verbatim): a seeded differential test over several hundred overlapping
  full-resolution-shaped candidates asserts equality across a range of
  thresholds, with no tolerance anywhere — N2 and N3 are exact.
- AC2 (non-ui) — every existing `nms.test.ts` case still passes against the new
  implementation, including the pinned "IoU exactly equal to the threshold
  survives" / "just below the threshold drops" pair, the equal-area tie broken
  by lower original index, ascending returned indices, and "a zero-area mask
  suppresses nothing".
- AC3 (non-ui) — `popcount` is exported from `nms.ts` and covered by unit tests:
  `0` -> 0, `0xFFFFFFFF` -> 32, `0x80000000` -> 1, and every single-bit position
  0..31, so a sign-extension bug on the high bit fails the suite.
- AC4 (non-ui) — a bbox-disjoint pair yields the same kept set as the reference,
  and a coverage length that is not a multiple of 32 yields the same kept set as
  the reference — the tail word's unused high bits are never counted.
- AC5 (non-ui) — `dedupeMasks` takes `width` as a required third parameter with
  no optional-width fallback, and `pairwiseIoU` keeps its current two-argument
  signature and byte-wise implementation.
- AC6 (non-ui) — the opt-in A/B path exists and is correct: `compareNms` is
  `false` in `DEFAULT_SEGMENTER_OPTIONS`; when set, the worker runs the
  reference then the fast path over the identical candidate array, and the
  `done` message carries `nmsComparison` with `referenceMs`, `fastMs` and
  `identical`; `createSegmenter` surfaces it on `SegmentationResult`.
- AC7 (non-ui) — the `nms` phase timing reports **only** the fast path even when
  `compareNms` is on: `timings.record('nms', ...)` and the `nms` progress event
  never include the reference run's elapsed time, so the results table is never
  inflated by the doubled work.
- AC8 (non-ui) — `dedupeMasksReference` is reachable from `./nms` for the worker
  and the tests but is **not** exported from `note-scanner/segmenter`; the
  package's public surface does not grow a second, slow NMS implementation.
- AC9 (non-ui) — the playground exposes a `compare-nms` checkbox and, when a
  comparison came back, renders reference ms, fast ms, the speedup, and a
  pass/fail on set equality, at both `pointsPerSide` 16 and 32 (nothing in the
  A/B path is keyed to a particular operating point).
- AC10 (non-ui) — **deferred to the human; satisfied outside this branch and not
  a gate on this PR.** Using the A/B panel this branch ships, set equality is
  confirmed on at least two images at both `pointsPerSide` 16 and 32, and the
  `nms` phase total before and after is recorded on issue #4. Producing these
  needs a WebGPU browser downloading SlimSAM weights, which this pipeline cannot
  do; the measurement caveat above (reference runs first, warmer cache for the
  fast path) travels with the recorded numbers.
