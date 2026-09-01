# Low-res filter/NMS — measured delta on a real GPU

Date: 2026-08-31 (sweeps stamped 2026-09-01T01:05Z and 2026-09-01T01:07Z UTC).

This document reports what carrying 256x256 masks through `filter` and NMS actually
cost and bought, measured on hardware. It covers AC1 (kept-set delta), AC3 (where the
time went), AC4 (filter sub-step attribution) and AC11 (`afterNms` vs `returned`).

## 1. Setup

**Adapter.** `apple` / `metal-3`, `software: false` — a real hardware adapter, not a
software fallback. Both sweeps report the same adapter.

**Image.** The playground's default sample, `playground/sample/cafe-table.jpg`,
1024x649. It is non-square, aspect ratio 1024/649 = 1.578; that ratio matters in
section 2.

**Options.** `decodePath: none`, `dtype: fp32`, `batchSize: 8`. The only variable
under test is `lowResFilterNms` (`fullres` = false, the old path; `lowres` = true, the
new path). `pointsPerSide` is the second axis in the timing sweep.

**Two sweeps, two configs.** The measurements come from two separate runs because they
want different things from the harness:

| purpose | config | artifacts | grid |
| --- | --- | --- | --- |
| agreement | `docs/measurements/2026-08-31-lowres-agreement.config.json` | `2026-09-01-decode-sweep.json` / `.md` | pps 16, `keepRawMasks: true`, reps 1 |
| timing | `docs/measurements/2026-08-31-lowres-timing.config.json` | `2026-09-01-decode-sweep-2.json` / `.md` | pps 16 and 32, `keepRawMasks: false`, reps 1 |

The agreement run keeps raw masks so `compareMaskSets` has something to compare; that
retention itself costs time, which is why the timing numbers are taken from the second
run with `keepRawMasks: false`.

**Warm-up and baseline.** The first run in each sweep was a discarded warm-up — it is
not a row in either table. Its purpose is to pay the cold-start costs (shader
compilation, pipeline creation, first-touch allocation) before anything is recorded, so
that no reported row carries them. The agreement baseline is that warm-up's mask set,
produced with the same options as the `fullres` row. Every "vs baseline" figure in
section 2 is measured against that set.

## 2. AC1 — the kept-set delta

### The numbers

Counts, from the agreement run (`2026-09-01-decode-sweep.md`, pps 16,
`keepRawMasks: true`):

| row | raw | afterFilter | afterNms | returned |
| --- | ---: | ---: | ---: | ---: |
| `p16-fp32-b8-none-fullres-r1` | 768 | 242 | 36 | 36 |
| `p16-fp32-b8-none-lowres-r1` | 768 | 240 | 34 | 34 |

Agreement against the baseline mask set, via `compareMaskSets` (greedy best-IoU
pairing; a pair below IoU 0.9 counts as unmatched on both sides):

| row | baseline | variant | matched | unmatched baseline | unmatched variant | mean IoU | median IoU | min IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `p16-fp32-b8-none-fullres-r1` (control) | 36 | 36 | 36 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| `p16-fp32-b8-none-lowres-r1` | 36 | 34 | 34 | 2 | 0 | 1.000 | 1.000 | 0.993 |

The min IoU on the `lowres` row is 0.993 as rendered; the JSON carries 0.9929.

### The control row, and why it is here

The `fullres` row is the same pipeline as the baseline, run a second time. It scores
36 / 36 / 36 matched, 0 unmatched on either side, and mean = median = min = 1.000.

That is the control, and it does real work: it establishes that this comparison
measures *the pipeline*, not run-to-run noise. The decode path is deterministic enough
that running it twice reproduces the mask set exactly. So every departure from 1.000 on
the `lowres` row is attributable to the change under test, not to jitter. Without this
row a min IoU of 0.9929 would be unreadable — you could not tell a real difference from
sampling noise. With it, the reading is unambiguous.

### Ruling: the 0.9929 is a swapped near-duplicate representative

The min IoU of 0.9929 is **a different mask being elected, not the same mask coming back
with a different edge.**

The mechanism is in NMS, not in resampling. Greedy NMS keeps largest-area-first. On the
old path it orders candidates by their **full-resolution** area; on the new path it
orders them by their **low-res (256x256)** area. Those two orderings are not identical,
and where a near-duplicate cluster contains two masks of nearly equal area, the two
orderings can elect *different members of the same cluster*. `compareMaskSets` then
pairs the elected mask against its cluster twin — a nearly-but-not-exactly identical
mask — and scores that pair at 0.9929.

This is a representative swap inside a near-duplicate cluster. It is emphatically not a
boundary difference on a shared candidate.

**What this sweep does not show.** The bit-identical-boundary claim is about the *same*
candidate: both paths resample it through the same `resampleThresholdMask` call with the
same arguments, so the same candidate yields the same boundary bits. That claim is
tested by the fixture-driven Boundary tab and its three-engine Playwright check. It is
**not** tested here, and nothing in this sweep should be read as proving it. A sweep that
compares two *sets* cannot demonstrate bit-identity of individual masks; it can only
show that the sets agree, which is a weaker statement. Do not cite this document as
evidence of bit-identity.

### Is the delta acceptable?

Two masks out of 36 are dropped by the new path (unmatched baseline 2, unmatched
variant 0 — the new path drops, it never invents). Everything it does keep, it keeps
faithfully: mean and median IoU are both 1.000, and the single worst pair is 0.9929.
There are two candidate effects, and the numbers separate them.

**(a) The scaled pre-NMS area gate is stricter on a non-square image.** The gate is now
computed over the whole 256x256 grid, *including the padded region*. On a non-square
image the real content occupies only part of that grid, so a threshold expressed as a
fraction of the grid corresponds to a larger fraction of the actual image than
`minMaskArea` intends.

The arithmetic, from `lowResMinArea` in `src/segmenter/core/mask-pipeline.ts`
(`Math.max(1, Math.round((minMaskArea * lowPixels) / originalPixels))`):

```
lowResMinArea(100, 65536, 1024 * 649)
  = max(1, round(100 * 65536 / 664576))
  = max(1, round(9.861))
  = 10 low-res pixels
```

The image letterboxes into the 256x256 grid as 256 x 162.25, so the content occupies
41536 of the 65536 grid pixels and **one content low-res pixel is 16 full-res pixels**
(1024/256 = 4 horizontally, 649/162.25 = 4 vertically). The applied gate is therefore
10 x 16 = **160 full-res pixels against a configured `minMaskArea` of 100 — 1.60x
stricter.**

Two ratios are in play and they are not the same number. The **1.578x** figure
(= 1024/649 = 65536/41536) is the over-division introduced by measuring against the
whole grid instead of the content region — the ratio *before* rounding. Rounding 9.861
up to a whole low-res pixel pushes it the rest of the way, and **1.60x is the gate that
actually dropped the masks.** The argument below rests on 1.60.

(`src/segmenter/core/mask-pipeline.test.ts:83` pins this exact geometry:
`expect(lowResMinArea(100, 65536, 1024 * 649)).toBe(10)`.)

The two unmatched-baseline masks are consistent with this. The asymmetry is the tell:
the drops are entirely one-sided (2 unmatched baseline, 0 unmatched variant), which is
exactly the signature of a *stricter gate* — a gate that only ever removes. A symmetric
disagreement would point somewhere else. The count also drops at the right place:
`afterFilter` falls 242 → 240, a loss of 2, and `afterNms` falls 36 → 34, the same 2.
The masks are lost in `filter`, before NMS ever sees them, which is where an area gate
lives. Two small masks near a 1.60x-tightened threshold is an unremarkable number for
this image.

**Independent corroboration at pps 32.** The timing sweep reproduces the same 1:1
relationship at a different grid density: `afterFilter` 951 → 947 (4 dropped) and
`afterNms` 57 → 53 (4 fewer). The same one-sided, filter-localised signature appears at
4x the sampling density, which is evidence the mechanism is the gate and not something
particular to the pps-16 candidate set. This is **counts only, no IoUs** — that sweep ran
`keepRawMasks: false`, so `compareMaskSets` had nothing to compare.

**(b) IoU at 256x256 could flip a borderline dedupe decision.** In principle, computing
overlap on a coarser grid can push a pair across the NMS IoU threshold and change
whether one suppresses the other. The counts show **no NET dedupe change**: the entire
36 → 34 delta is already accounted for by the `afterFilter` drop of 242 → 240, and NMS
removed 206 on the old path and 206 on the new path.

But identical counts removed is **not** identical behaviour, and this must not be
overstated. The 0.9929 pair documented in the ruling above is direct evidence that NMS
*did* diverge in **which** mask it kept — it elected a different member of a
near-duplicate cluster. So the accurate statement is: NMS diverged in its choice of
representative while removing the same number of masks. What is absent from the numbers
is a *net* change in dedupe outcome — a pair that suppressed on one path and survived on
the other, which would have moved the kept-set size at NMS independently of `filter`.
That did not happen here. The divergence that did happen is a tie-break effect, not a
threshold-crossing one.

**Verdict: acceptable.** The delta is 2 masks out of 36, one-sided, localised to the
`filter` stage, and mechanistically explained by a known and quantified gate asymmetry
on a non-square image. No mask is invented, no kept mask is degraded (mean and median
IoU 1.000), and the single worst pair at 0.9929 is a near-duplicate representative swap
rather than a quality loss. The one caveat worth carrying forward: the gate asymmetry
scales with aspect ratio, so a more extreme image (a panorama, a tall receipt) would see
a proportionally stricter gate and could lose more. That is a property of the scaled
gate, not of this measurement, and it is the thing to watch if the drop count ever
matters.

## 3. AC3 — where the time went

All figures from the timing run (`2026-09-01-decode-sweep-2.md`, `keepRawMasks: false`,
reps 1). All times in ms. `budget` = `total` − `model-load`.

### pointsPerSide 16

`p16-fp32-b8-none-fullres-r1` vs `p16-fp32-b8-none-lowres-r1`:

| stage | fullres | lowres | delta | speedup |
| --- | ---: | ---: | ---: | ---: |
| `filter` | 693.0 | 176.8 | −516.2 | 3.92x |
| `nms` | 292.0 | 37.5 | −254.5 | 7.79x |
| `resample` | 0.0 | 45.9 | +45.9 | new stage |
| `mask-encode` | 32.4 | 35.7 | +3.3 | — |
| `budget` | 7798.0 | 7113.0 | −685.0 | 8.78% of budget |

Stage group: `filter+nms` 985.0 → `filter+nms+resample` 260.2 = **724.8 ms saved,
3.79x faster.**

Counts: fullres `afterFilter` 242 / `afterNms` 36 / `returned` 36; lowres 240 / 34 / 34.

### pointsPerSide 32

`p32-fp32-b8-none-fullres-r1` vs `p32-fp32-b8-none-lowres-r1`:

| stage | fullres | lowres | delta | speedup |
| --- | ---: | ---: | ---: | ---: |
| `filter` | 2647.0 | 658.5 | −1988.5 | 4.02x |
| `nms` | 1045.7 | 103.4 | −942.3 | 10.11x |
| `resample` | 0.0 | 68.5 | +68.5 | new stage |
| `mask-encode` | 50.3 | 50.5 | +0.2 | — |
| `budget` | 27020.2 | 24542.6 | −2477.6 | 9.17% of budget |

Stage group: `filter+nms` 3692.7 → `filter+nms+resample` 830.4 = **2862.3 ms saved,
4.45x faster.**

Counts: fullres `raw` 3072 / `afterFilter` 951 / `afterNms` 57 / `returned` 57; lowres
3072 / 947 / 53 / 53.

### What got faster

`filter` and `nms`, both of them, at both operating points, by large factors. `filter`
is ~3.9-4.0x faster and `nms` is 7.8-10.1x faster. The direction of the `nms` scaling is
worth noticing: the speedup *grows* with `pointsPerSide` (7.79x at 16, 10.11x at 32).
NMS is quadratic in candidate count, and its per-pair cost is the IoU computation, which
is linear in mask area. Shrinking every mask to 256x256 shrinks that inner cost by a
constant factor, and the more pairs there are the more total time that constant removes.
The pps-32 row has ~4x the post-filter candidates of pps-16 (947 vs 240), and that is
where the extra speedup comes from.

### What got slower, and where time MOVED rather than vanished

`resample` is a **new stage**: 0.0 → 45.9 ms at pps 16, 0.0 → 68.5 ms at pps 32. It is
not new work in the sense of extra work. It is exactly the cost that `filter` used to
carry inline — upsampling a mask to full resolution — now deferred to the end of the
pipeline and given its own counter. On the old path this cost was paid *per candidate*,
for all 242 (or 951) survivors of the filter gate. On the new path it is paid *per kept
mask*, for the 34 (or 53) that survive NMS. That is why it is small: the work moved to a
point in the pipeline where there is 7-18x less of it to do.

So a fair reading of the `filter` saving is not "516 ms of work disappeared" but "516 ms
of work stopped happening at full resolution, and 46 ms of it reappeared downstream at
the right cardinality". The stage-group figures (`filter+nms` vs
`filter+nms+resample`) are the honest accounting, and they are the numbers to quote:
**724.8 ms at pps 16, 2862.3 ms at pps 32.**

`mask-encode` is flat: +3.3 ms at pps 16, +0.2 ms at pps 32. It was not expected to move
and it did not.

### Noise caveat — read this before quoting the budget delta

The pps-16 rows appear in **both** sweeps, under the same options. Their spread across
the two runs bounds single-run noise:

| row | agreement run budget | timing run budget | spread | spread % |
| --- | ---: | ---: | ---: | ---: |
| `p16 fullres` | 8275.0 | 7798.0 | 477.0 | 5.76% |
| `p16 lowres` | 8224.6 | 7113.0 | 1111.6 | 13.52% |

Percentages are of the **larger** budget in each pair (477.0 / 8275.0 = 5.76%;
1111.6 / 8224.6 = 13.52%). Against the smaller they are 6.12% and 15.63%.

**This is a COARSE bound, not a clean repeat.** The two sweeps differ in `keepRawMasks`
(`true` in the agreement run, `false` in the timing run), so the spread mixes genuine
run-to-run variation with the real cost of raw-mask retention. It is an upper bound on
noise contaminated by a systematic difference, not an isolated measurement of noise. It
is the best available bound and it should be read as generous rather than tight.

Taking the `p16 fullres` pair as the reference band — **477.0 ms** — the conclusion is
unchanged and sharper:

**The pps-16 budget saving of 685.0 ms is only about 1.4x that band** (685.0 / 477.0 =
1.44), and it is smaller than the 1111.6 ms spread on the `lowres` row. At one rep, the
pps-16 budget delta is *not separable from run-to-run variation*. It must not be quoted
as a solid figure, and this document does not treat it as one.

Two things comfortably exceed that band and can be quoted:

- **The stage totals.** `filter` moves by 516.2 ms and `nms` by 254.5 ms at pps 16, and
  the stage group by 724.8 ms — but more to the point, these are *ratios* of 3.92x and
  7.79x on stages whose absolute values are hundreds of ms. A budget spread of 5.8-13.5%,
  which is dominated by the `decode` stage (5.5-5.8 s of the budget), cannot manufacture
  a 4x-10x change in two stages that together account for under 1 s. The stage-level
  effect is far larger than the spread and survives it comfortably.
- **The pps-32 budget saving of 2477.6 ms**, which exceeds the 477.0 ms band by 5.2x and
  the wider 1111.6 ms figure by more than 2x.

### Ruling: the savings are far below issue #1's projection

Issue #1 projected the win from this change. The measurement does not come close:

| operating point | projected budget saving | actual budget saving | actual as % of projection | short by |
| --- | ---: | ---: | ---: | ---: |
| pps 16 | ~15 s | 0.685 s | 4.6% | 21.9x |
| pps 32 | ~29 s | 2.478 s | 8.5% | 11.7x |

On the stage-group figures — the more defensible accounting — the actuals are 0.725 s
and 2.862 s, which does not change the picture.

This is a finding, and it is reported as one rather than buried.

**Why the projection missed.** Those projections were made against the **pre-F2
baseline**. F2 — the fused resample+threshold — has since landed, and it already took
most of the filter win that issue #1 was forecasting. This change collects what F2 left
behind, not the original prize. The projection was not wrong when it was written; it was
overtaken. Re-projecting from a stale baseline is the error to avoid repeating, not the
change itself.

**What the change still buys, which is equally true.** Two things, and neither shows up
in a budget column on a 1024x649 image:

1. **It removes the per-candidate full-resolution allocation.** A candidate mask is now
   **320 KB regardless of image size**, against `width * height` bytes before. On a 12 MP
   photo that is **12.2 MB per candidate** on the old path. With hundreds of candidates
   in flight, that is the allocation pattern that made large images simply impossible —
   not slow, impossible. This sample image is small enough that the ceiling never binds,
   which is precisely why the timing table understates the change.
2. **It is the keystone that unblocks encoding at 256x256.** That work cannot proceed
   while `filter` and NMS demand full-resolution masks.

**Verdict.** Do not dress 0.7 s up as 15 s — the projection was against a superseded
baseline and the honest number is between 0.7 s and 2.9 s of stage time depending on
operating point, with the pps-16 *budget* figure not separable from noise at one rep.
Equally, do not write the change off: the stage-level speedups (3.9-4.0x on `filter`,
7.8-10.1x on `nms`) are real and survive the noise bound, the memory ceiling it removes
is the difference between working and not working on large images, and it is a
prerequisite for the 256x256 encoding work. The time win is a bonus here; the
allocation win is the reason.

## 4. AC4 — filter sub-step attribution

`FILTER_SUBSTEP_ORDER` is `['select', 'threshold']` (`src/segmenter/core/types.ts`), and
both sub-steps are reported on every row. The step 4 residual check output, verbatim:

```
p16-fp32-b8-none-fullres-r1 subphases: select,threshold select 111.4 threshold 581.5 sum 692.9000 filter 693.0000 residual 0.100000 | afterNms 36 returned 36
p16-fp32-b8-none-lowres-r1 subphases: select,threshold select 124.0 threshold 52.7 sum 176.7000 filter 176.8000 residual 0.100000 | afterNms 34 returned 34
p32-fp32-b8-none-fullres-r1 subphases: select,threshold select 438.8 threshold 2208.1 sum 2646.9000 filter 2647.0000 residual 0.100000 | afterNms 57 returned 57
p32-fp32-b8-none-lowres-r1 subphases: select,threshold select 471.1 threshold 187.4 sum 658.5000 filter 658.5000 residual 0.000000 | afterNms 53 returned 53
```

Reading the sub-steps: `select` got slightly **slower** on the low-res path for unchanged
work (111.4 → 124.0 at pps 16; 438.8 → 471.1 at pps 32). It is the cheaper half and the
regression is small, but it is a regression, not a wash.

The whole filter win is in `threshold`, which falls 581.5 → 52.7 at pps 16 and
2208.1 → 187.4 at pps 32. Those raw ratios — 11.0x and 11.8x — are **not the figure to
quote**, because the work `threshold` shed did not vanish: it reappears downstream as the
new `resample` stage. The honest comparison is `threshold + resample` against
`threshold`:

| operating point | old `threshold` | new `threshold` + `resample` | speedup |
| --- | ---: | ---: | ---: |
| pps 16 | 581.5 | 52.7 + 45.9 = 98.6 | 5.90x |
| pps 32 | 2208.1 | 187.4 + 68.5 = 255.9 | 8.63x |

**5.90x and 8.63x are the defensible sub-step numbers**; 11.0x and 11.8x count the
saving twice by ignoring where the work went. This is the same accounting point made at
stage level in section 3, one level further down.

The shape is as expected: thresholding is per-pixel work, so shrinking the mask shrinks
it directly, while `select` is per-candidate bookkeeping that does not care how large a
mask is — which is also why `select` had no win available to it and drifted slightly the
wrong way.

### Ruling: the residual is 0.1 ms on three of four rows, not zero

The sub-steps do not sum *exactly* to the filter stage total on three of the four rows.
The residual is 0.1 ms each time. This section exists so a reader learns that from the
document rather than from a surprising log line.

**Mechanism.** The worker's `record('filter', ...)` takes its own `performance.now()`
reading after the last `recordSub` has already closed. Chrome quantises
`performance.now()` to 0.1 ms. So exactly one quantisation tick sits between the two
cursors, unattributed to either sub-step.

**Evidence it is quantisation, not drift and not a leak.** Three independent signs, and
they agree:

1. **Every stage total in the JSON is a multiple of 0.1.** That is the quantum showing
   itself directly in the data.
2. **It is exactly one tick regardless of row size.** A drift or a leak would scale with
   the amount of work: the pps-32 fullres filter stage is 3.8x the pps-16 fullres stage,
   and the residual on both is 0.1 ms, not 0.38 ms. A constant that refuses to scale
   across a 3.8x change in workload is a clock artifact, not accumulated error. The
   fourth row's residual of exactly 0.000000 is the same story — sometimes the two reads
   land in the same tick.
3. **It is 0.014% of the pps-16 fullres filter stage** (0.1 / 693.0). It is below the
   resolution at which any of these numbers are being read.

**The unit test measures something different, and both are correct.** The unit test
asserts *exact* summation, and it passes. It can, because it feeds synthetic values
through a single clock read — there is no second `performance.now()` to be quantised
against, so there is no tick to strand. The real run has two independent clock reads and
therefore has a sub-tick boundary between them. The test proves the *arithmetic* is
right; the run shows the *clock* has finite resolution. Neither contradicts the other.

**This is not a code defect and it has not been fixed.** AC4 asks that both sub-steps
are reported, that the order is `['select', 'threshold']`, and that they account for the
filter stage. All three hold: both sub-steps are present on every row, the order is as
specified, and a residual of 0.014% is accounting for the stage by any reasonable
reading. A sub-tick residual does not break the acceptance criterion. No change was
made to `src/` for it, and none should be.

## 5. AC11 — `afterNms` vs `returned`

| row | afterNms | returned | gap |
| --- | ---: | ---: | ---: |
| `p16-fp32-b8-none-fullres-r1` | 36 | 36 | 0 |
| `p16-fp32-b8-none-lowres-r1` | 34 | 34 | 0 |
| `p32-fp32-b8-none-fullres-r1` | 57 | 57 | 0 |
| `p32-fp32-b8-none-lowres-r1` | 53 | 53 | 0 |

The two counts are **equal on every row, in both sweeps, at both operating points.**

**What that means.** The gap between `afterNms` and `returned` is the full-resolution
area re-check — the pass that re-measures each surviving mask at full resolution and can
drop one that only met the area threshold because of the low-res approximation. A gap of
zero on every row means **that re-check dropped nothing on this image at these options.**
The masks that survive the scaled gate and NMS also satisfy the full-resolution gate.

That is a reassuring result and a narrow one. It says the scaled gate did not let
anything through that the full-resolution gate then had to catch — the approximation
erred on the strict side (consistent with section 2's finding that the applied gate is
1.60x stricter on this image), so there was nothing left for the re-check to reject. On an image where
the gate erred the other way, this column is where it would show.

**What AC11 asks for is that the two counts are distinguishable, and they are.** The
value of the criterion is not that the gap is zero here — it is that the instrumentation
can *tell you* when it is not. Before AC11 a mask dropped by the full-resolution
re-check was invisible; it looked like NMS had simply kept fewer. Now the two counters
are separate columns in every sweep table, and a non-zero gap on a future image or a
future set of options will be legible at a glance rather than requiring a bisect. The
zeros are the current reading, not the point of the change.

## 6. Caveats

These belong with the numbers, in this file, rather than in a commit message where they
would be read once and lost.

**The discarded warm-up.** The first run in each sweep was discarded and is not a row in
either table. It absorbs the one-time costs — shader compilation, pipeline creation,
first-touch allocation — that would otherwise land entirely on whichever row happened to
run first and make it look slow for reasons that have nothing to do with the variable
under test. This protects the comparison: without it, the `fullres`/`lowres` ordering
inside the grid would confound the result. Every row reported here is a warm row, and
the `model-load` column (629-720 ms across all rows) is a warm, HTTP-cached load, not a
cold one.

**One rep per row.** `reps: 1` in both configs. There is no within-run variance estimate.
The noise bound used throughout this document comes from the fact that the pps-16 rows
were measured **twice**, once in each sweep — a cross-run spread of **477.0 ms on
`p16 fullres`** (8275.0 vs 7798.0; 5.76% of the larger, 6.12% of the smaller) and
1111.6 ms on `p16 lowres` (13.52% of the larger). It is a bound on *budget*, which is
dominated by the `decode` stage.

**It is a coarse bound rather than a clean repeat.** The two sweeps were not run under
identical settings: the agreement run used `keepRawMasks: true` and the timing run
`keepRawMasks: false`. The spread therefore mixes true run-to-run variation with the
systematic cost of retaining raw masks, and overstates noise by an unknown amount. It is
the best bound available from these artifacts, and it is generous.

The reading it changes is stated where the number appears, in section 3: **the pps-16
budget saving of 685.0 ms is only about 1.4x the 477.0 ms band, and smaller than the
1111.6 ms figure, so it is not separable from run-to-run variation at one rep.** It must
not be quoted as a solid figure. The pps-32 budget saving of 2477.6 ms exceeds the band
by 5.2x and does survive it, as do all four stage-level ratios (`filter` 3.92x and 4.02x,
`nms` 7.79x and 10.11x), which are changes far larger than a 5.8-13.5% budget spread
could produce in stages totalling under 1 s.

**Agreement and timing come from different runs.** The agreement figures are from the
`keepRawMasks: true` sweep and the timing figures from the `keepRawMasks: false` sweep,
because raw-mask retention itself costs time. Do not read a timing number out of the
agreement table or an agreement number out of the timing table; they are separate
measurements and the counts, not the times, are what is common between them. (The counts
do agree: 240/34/34 and 242/36/36 on the pps-16 rows in both sweeps.)

**Scope of the agreement evidence.** The agreement run covers pps 16 only. There is no
`compareMaskSets` evidence at pps 32; its count deltas (951 → 947 `afterFilter`,
57 → 53 `afterNms`) are consistent with the same one-sided gate effect, but that is an
inference from counts, not a measured IoU.

**One image, one adapter.** Everything here is `playground/sample/cafe-table.jpg`
(1024x649) on `apple` / `metal-3`. The gate asymmetry of section 2 is 1.60x as applied on
*this* image (1.578x before `lowResMinArea` rounds up), it scales with aspect ratio, and
it would be larger on a more elongated one. The memory
argument of section 3 is the reverse: it is invisible on an image this small and is the
dominant effect on a large one.
