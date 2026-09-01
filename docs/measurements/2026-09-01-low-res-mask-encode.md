# Low-res mask encode — measured delta on a real GPU

Date: 2026-09-01 (sweep stamped 2026-09-01T18:50:35.925Z UTC).

This document reports what encoding each survivor's PNG at the decoder's own logit
window (`lowResMaskEncode`, M2, issue #7) actually cost and bought, measured on
hardware. It covers AC1 (the encode saving), AC7 (where the moved cost is accounted)
and AC8 on real data (the kept set does not move across the encode axis).

**Artifacts this document draws from.** Every number below is copied from, or computed
from, these committed files and nothing else:

- `docs/measurements/2026-09-01-decode-sweep-3.json` — the machine-readable run.
- `docs/measurements/2026-09-01-decode-sweep-3.md` — the same run as a table.
- `docs/measurements/2026-09-01-mask-encode.config.json` — the grid that produced them.

## 1. Setup

**Adapter.** `apple` / `metal-3`, `software: false` — a real hardware adapter, not a
software fallback. Taken from the sweep's own header
(`2026-09-01-decode-sweep-3.md`, line 3, and the `adapter` object in the JSON).

**Image.** The playground's default sample, `playground/sample/cafe-table.jpg`,
1024x649 (read from the JPEG's SOF marker). Its aspect ratio is what makes the encode
target non-square, which matters in section 2.

**Config.** `docs/measurements/2026-09-01-mask-encode.config.json`:
`decodePaths: ["none"]`, `dtypes: ["fp32"]`, `batchSizes: [32]`,
`pointsPerSide: [16, 32]`, `lowResFilterNms: [true]`,
`lowResMaskEncode: [false, true]`, `keepRawMasks: false`, `reps: 1`.

Every axis but the one under test is pinned to a shipped default, and
`lowResFilterNms` is pinned **on** — the encode target only leaves full resolution on
the low-res path (`resolveEncodeSize` returns the original size whenever
`lowResFilterNms` is false), so this is the only comparison in which
`lowResMaskEncode` does anything at all.

**Rows.** Four measured rows, all in ONE browser session:

1. `p16-fp32-b32-none-lowres-encfull-r1`
2. `p16-fp32-b32-none-lowres-enclow-r1`
3. `p32-fp32-b32-none-lowres-encfull-r1`
4. `p32-fp32-b32-none-lowres-enclow-r1`

**Warm-up.** A fifth run happened first and its numbers were **discarded**: the runner
walks `[warmUpRow(rows), ...rows]`, and the warm-up is a copy of row 1 under the
reserved id `warm-up`. It is not in the table above and not in the artifacts' `rows`
array. Its purpose is to pay the cold costs — shader compilation, pipeline creation,
a cold HTTP cache, first-touch allocation — before anything is recorded, so that the
`encfull` row (which runs first among the measured four) does not carry them and read
as slow for a reason that has nothing to do with the encode target.

## 2. AC1 — the encode saving

### The numbers

All times in ms, copied from `2026-09-01-decode-sweep-3.json`. `per-mask` is the phase
total divided by that row's `returned` count (34 at pps 16, 53 at pps 32); `p50` is the
per-mask median the timing accumulator recorded directly.

| pps | encode target | `mask-encode` total | per-mask | `mask-encode` p50 | `resample` total | `budget` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | `encfull` (1024x649) | 31.6 | 0.929 | 0.80 | 43.5 | 7092.1 |
| 16 | `enclow` (256x162) | 10.5 | 0.309 | 0.20 | 47.5 | 7067.2 |
| 16 | **delta** | **−21.1** | **−0.620** | **−0.60** | **+4.0** | **−24.9** |
| 16 | **ratio (full ÷ low)** | **3.01x** | **3.01x** | **4.00x** | **0.916x** | **1.004x** |
| 32 | `encfull` (1024x649) | 49.9 | 0.942 | 0.80 | 68.3 | 23834.4 |
| 32 | `enclow` (256x162) | 14.2 | 0.268 | 0.20 | 76.4 | 23863.4 |
| 32 | **delta** | **−35.7** | **−0.674** | **−0.60** | **+8.1** | **+29.0** |
| 32 | **ratio (full ÷ low)** | **3.51x** | **3.51x** | **4.00x** | **0.894x** | **0.999x** |

`mask-encode` falls from 0.446% of the budget to 0.149% at pps 16, and from 0.209% to
0.060% at pps 32.

### Verdict: yes, but the win is small in absolute terms and smaller than the pixel count predicts

**The pixel ratio this image implies.** `resolveEncodeSize` computes
`min(round(lowWidth x reshapedWidth / padWidth), originalWidth)` x
`min(round(lowHeight x reshapedHeight / padHeight), originalHeight)`. For this image
the SAM processor's geometry is `lowWidth = lowHeight = 256`, `padWidth = padHeight =
1024`, `reshapedWidth = 1024`, `reshapedHeight = 649`, so the target is
`round(256 x 1024 / 1024) = 256` by `round(256 x 649 / 1024) = round(162.25) = 162` —
**256x162**. That is 41,472 pixels against 664,576 at full resolution, a ratio of
**16.02x**.

**The measured saving is materially smaller than that: 3.01x at pps 16 and 3.51x at
pps 32, against a 16.02x pixel reduction.** A ~16x cheaper payload buys a ~3-3.5x
cheaper stage, so most of what `mask-encode` costs per mask is **not** proportional to
pixels.

**What the fixed cost is, quantified.** Per-mask cost can be modelled as
`fixed + k x pixels`, and each operating point gives two measurements at two pixel
counts on the same hardware in the same session, which determines both terms exactly
(this is a two-point fit, so it is a model of the data, not an independent
measurement — but the two operating points were fitted independently and agree):

| pps | `k` (pixel-proportional) | `fixed` per mask | `fixed` as a share of the `enclow` per-mask cost |
| ---: | ---: | ---: | ---: |
| 16 | 9.96e-7 ms/px | 0.268 ms | 86.6% |
| 32 | 1.08e-6 ms/px | 0.223 ms | 83.3% |

The two fits agree to within 9% on `k` and 0.045 ms on `fixed`, which is the reason to
believe them. **~0.22-0.27 ms per mask survives regardless of size**, and at the
reduced target that fixed part is 83-87% of what `mask-encode` still costs. Reading
`encodeMaskPng` (`src/segmenter/core/mask-encode.ts`), that is what one would expect:
per mask it allocates a `Blob`, constructs a `CompressionStream('deflate')`, and
awaits a `Response(...).arrayBuffer()` round trip through the streams machinery, then
assembles five chunks with their CRC32s and base64-wraps the result into a data URL.
The stream setup and the async round trip are per-call costs that a smaller payload
does not reduce. Shrinking the image further would therefore buy almost nothing; the
remaining `mask-encode` cost is an argument for **M1/M5** — deleting the round trip —
not for a smaller PNG.

**Where the cost moved.** `resample` now runs a second, smaller resample per survivor
(`resolveEncodeMask`), and it is up **+4.0 ms at pps 16 (+9.2%)** and **+8.1 ms at
pps 32 (+11.9%)**. Netting the two phases:

| pps | `resample` + `mask-encode`, `encfull` | same, `enclow` | net | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 75.1 | 58.0 | **−17.1** | 1.295x |
| 32 | 118.2 | 90.6 | **−27.6** | 1.305x |

**The net is a win at both operating points** — 17.1 ms and 27.6 ms, ~1.30x on the two
phases taken together — so the second resample costs about a fifth of what the smaller
encode saves, and the change pays for itself. It is a real but small win: 17.1 ms is
0.24% of the pps-16 budget and 27.6 ms is 0.12% of the pps-32 budget.

**Honest limit: the saving is below this run's end-to-end noise floor.** The `budget`
column moves −24.9 ms at pps 16 but **+29.0 ms at pps 32** — the low-res row is
nominally *slower* end to end at the denser grid, which cannot be an effect of this
change. The same run measures its own noise: `encode`, a phase `lowResMaskEncode`
cannot touch, differs by 54.4 ms between the two pps-16 rows and 30.7 ms between the
two pps-32 rows, and `decode` differs by 38.1 ms at pps 16. Those swings are larger
than the 17.1/27.6 ms the change actually saves. So the saving is visible and
consistent in the phase counters it applies to, and invisible in wall clock. That is
the correct reading of these numbers, and it is why AC1 is stated on the phase totals
rather than on `budget`.

**So: acceptable.** The change makes the stage it targets 3.0-3.5x cheaper, nets a win
after the cost it moves, does not perturb the returned set (section 4), and sits behind
a default-on flag whose off path is the previous behaviour exactly. What it is not is a budget-level improvement — after
issue #5 moved PNG encoding into the worker and made it 1-bit, `mask-encode` was
already down to 0.2-0.45% of the budget, and this change takes a small number and makes
it smaller. The remaining pipeline cost is `decode` (5.6 s at pps 16, 22.0 s at pps 32
— 78% and 92% of budget respectively), exactly as `docs/pipeline.md` intends.

## 3. AC7 — the accounting

The second resample is timed into **`resample`**, not into `mask-encode`. In
`src/segmenter/worker/segmenter.worker.ts` the survivor loop opens
`const resampleStarted = performance.now()`, calls `resolveSurvivor` and then
`resolveEncodeMask`, and only then calls `timings.record('resample', ...)`; the
`mask-encode` timer starts afterwards and wraps `encodeMaskPng` alone.

The numbers show it rather than merely asserting it. In every pair, `resample` goes
**up** and `mask-encode` goes **down**, and both are reported:

| pps | `resample` | `mask-encode` |
| ---: | ---: | ---: |
| 16 | 43.5 → 47.5 (**+4.0**) | 31.6 → 10.5 (**−21.1**) |
| 32 | 68.3 → 76.4 (**+8.1**) | 49.9 → 14.2 (**−35.7**) |

The `resample` sample **count** is unchanged — 34 at pps 16 and 53 at pps 32 in both
rows — because both resamples for a survivor are recorded as one sample. So the
increase shows up as per-survivor cost (1.279 → 1.397 ms at pps 16, 1.289 → 1.442 ms
at pps 32) and not as extra samples. Had the second resample been timed inside
`mask-encode`, the `mask-encode` deltas above would be roughly 4-8 ms smaller and the
saving this document claims would be overstating itself by that much.

## 4. AC8 on real data — the kept set does not move

Mask counts per row, from the artifacts:

| row | `raw` | `afterFilter` | `afterNms` | `returned` |
| --- | ---: | ---: | ---: | ---: |
| `p16-fp32-b32-none-lowres-encfull-r1` | 768 | 233 | 34 | 34 |
| `p16-fp32-b32-none-lowres-enclow-r1` | 768 | 233 | 34 | 34 |
| `p32-fp32-b32-none-lowres-encfull-r1` | 3072 | 916 | 53 | 53 |
| `p32-fp32-b32-none-lowres-enclow-r1` | 3072 | 916 | 53 | 53 |

**`afterNms` and `returned` are identical across the encode axis at each operating
point**: 34/34 at pps 16 and 53/53 at pps 32. This is the check that matters, because
`lowResMaskEncode` is supposed to select a resample *target* and nothing else — the
`minMaskArea` re-check still runs against the full-resolution survivor
(`resolveSurvivor`), and `masks.push` still records `area: mask.area` from that
survivor rather than from the encode-sized copy. A moved kept set would have meant the
flag was changing which masks come back, and no timing number from such a run would be
worth reporting.

(`afterNms == returned` in all four rows here simply means no survivor failed the
exact full-resolution area re-check on this image at these settings. That gap is
expected to be non-zero in general; it is not what this section is testing.)

## 5. Caveats

**One rep per row.** `reps: 1`, so single-run noise is **not quantified** — there is no
variance estimate for any figure above. Section 2 bounds it indirectly instead, from
phases this change cannot affect (`encode` differs 54.4 ms / 30.7 ms between paired
rows, `decode` 38.1 ms / 1.4 ms), and that bound is larger than the net saving. The
per-phase deltas are consistent in sign and similar in magnitude across two independent
operating points, which is the reason to trust their direction; nobody should read the
one-decimal precision as significance.

**The discarded warm-up.** The first run in the session was thrown away. Without it,
the very first measured row — `p16-fp32-b32-none-lowres-encfull-r1`, the `encfull`
baseline — would have absorbed cold shader compilation and a cold HTTP cache, which
would have inflated the very number the saving is measured against. The
`2026-08-28-decode-sweep.md` report had to disclose exactly that artifact. It is also
why all four rows are in ONE session: differencing two separate sweeps would reintroduce
the same problem in a form that is harder to see.

**Boundary precision — what this buys is paid for here.** The PNG is 256x162 for a
1024x649 image, so `SegmentViewer` upscales it by **4.000x horizontally and 4.006x
vertically**, and mask edges therefore quantise to a ~4 px step in image space. Because
`buildMaskData` derives each mask's `area` from the DECODED coverage (it counts
`alpha > 0 && red > 0` pixels on the upscaled canvas) rather than from the `area` the
worker reports, that quantisation changes the areas `hitTestAll` orders by. In
principle two overlapping masks of nearly equal area could swap places in
`hitTestAll`'s smallest-first result.

**Nothing in this run bears on that.** This sweep ran with `keepRawMasks: false` and
never decodes a mask PNG — it measures the worker, not the viewer, and its identical
counts (section 4) show only that the same masks are returned, not that they hit-test
the same way. The evidence for the hit-test question is elsewhere: the boundary
browser spec, which hit-tests real decoded, real upscaled coverage in three engines.
This document neither confirms nor contradicts it.

**Scope.** One image, one adapter, one dtype, one batch size, `decodePath: none`. The
16.02x pixel ratio and the 4x upscale factor are properties of *this* image's aspect
ratio; a square 1024x1024 image would encode at 256x256 (a 16.00x pixel ratio and a
clean 4x on both axes), and a small image would clamp to its own dimensions and save
nothing. The fixed ~0.22-0.27 ms per mask is the part expected to generalise, because
it is per-call and not per-pixel.
