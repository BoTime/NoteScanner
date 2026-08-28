# Automated Playwright sweep of the decode phase

Build an automated, repeatable, committed measurement of the `decode` stage:
two new decode paths behind independent `SegmenterOptions` flags, a compare
harness ported from PR #11 onto current `main`, and a Node + Playwright runner
that walks a config-driven grid on a real GPU and writes a ranked table under
`docs/measurements/`.

Issue #16. Follow-up to #1 (the `decode` budget) and #12 (the hand-run
`batchSize`/`dtype` table).

## Why

Issue #1 treats `decode` as irreducible model work with two levers — D1
`batchSize` and D2 fp16 — both **estimated**. Issue #12 turned those estimates
into measurements, but by hand, once, on one image, in a single browser
session, with one sample per cell, using a harness that was never merged. Since
then `mask-encode` (M3/M4) and `nms` (N2/N3) have landed, so `decode` is a much
larger share of what is left.

Two things follow. First, the measurement itself needs to become a script
anyone can re-run and whose output is committed, rather than a table pasted
into an issue from a session nobody else can reproduce. Second, reading
`segmenter.worker.ts` against transformers.js turns up two decode levers issue
#1 never considered, and neither can be settled by argument — only by measuring
them on the real adapter.

**The goal is not to pick a decode optimization by reasoning. It is to build
the sweep that names the winner.**

## Prior art — port, do not rebase

The closed-unmerged PR #11 branch already carries a reviewed measurement
harness: `playground/compare.ts` (a preset row matrix, an injectable `runRow`,
`compareMaskSets` built on the existing `pairwiseIoU` rather than a second IoU
implementation, and `toMarkdown`), `playground/compare.test.ts`,
`playground/CompareView.tsx` (a third playground tab plus `dtype`/`batchSize`
selects), and an opt-in `keepRawMasks`.

**Those files are ported onto current `main`; that branch is not replayed onto
it.** It predates both the M3/M4 1-bit PNG worker encode and the N2/N3 NMS
rewrite, so its 8 commits show ~3,600 deletions against `main`, and replaying
them would drag two merged perf rewrites back through conflict resolution for
no gain. Issue #12 explicitly sanctions re-cutting.

One ported piece cannot be ported verbatim, and this spec says so rather than
leaving it to be discovered in review — see **`keepRawMasks` must be re-cut**
below.

**PR #11's known gap is carried forward as work, not as a footnote.**
`tsconfig.json` excludes `playground/` and `vitest.config.ts` includes only
`playground/**/*.test.ts`, so `CompareView.tsx` is checked by neither typecheck
nor tests — and both defects PR #11's review found lived in exactly that file.
This run brings the compare harness under a check that actually runs.

## The two new decode paths

Two independent `SegmenterOptions` flags, both `false` in
`DEFAULT_SEGMENTER_OPTIONS`, so the sweep can measure each alone and both
together and see whether they compose.

### `overlapDecodeFilter`

The worker's batch loop is strictly serial today: `await model(...)` (GPU busy,
CPU idle, ~171-183 ms p50 per issue #12) and then the filter block (CPU busy,
GPU idle, ~300-544 ms p50). The GPU is idle for the larger of the two.

With the flag on, the loop keeps the next batch's dispatch in flight while the
current batch is post-processed:

```
let pending = dispatch(batches[0]);
for (let b = 0; b < batches.length; b += 1) {
  const outputs = await pending;                     // <- the decode timer
  pending = b + 1 < batches.length ? dispatch(batches[b + 1]) : null;
  filter(outputs);                                   // <- the filter timer
}
```

`dispatch(batch)` builds the point/label tensors and calls `model(...)`,
returning the promise unawaited; tensor construction stays inside it so that
CPU cost is still attributed to `decode`. With the flag off the loop is exactly
today's shape — `dispatch` immediately awaited — so the serial path is not a
special case of the overlapped one and cannot regress by accident.

**This does not make decode faster. It moves time between the stage
counters.** When the GPU finished during the previous filter block, the
`await pending` resolves almost immediately and the `decode` row collapses
toward zero — but the work happened, it simply stopped being counted anywhere.
**The sweep therefore ranks on wall clock, never on the `decode` row**, and
this caveat is repeated in the generated report, where a reader meets the
numbers.

Two consequences to bind:

- **Peak memory rises by one batch of `pred_masks`.** At `batchSize` 32 that
  tensor is `[1, 32, 3, 256, 256]` fp32 ~ 25 MB, so two in flight is ~50 MB.
  Acceptable at the default grid's sizes, and a further reason `batchSize: 64`
  stays out of that grid (below): a variant that exhausts memory must be
  recorded as a **failed row**, not crash the sweep.
- **An abandoned in-flight dispatch must never become an unhandled
  rejection.** If the filter block throws while `pending` is outstanding, the
  loop attaches a no-op `catch` to `pending` before propagating, so the run
  fails with the phase it actually died in rather than with a stray top-level
  rejection.

### `gpuResidentEmbeddings`

`SamModel.forward` feeds `image_embeddings` and `image_positional_embeddings`
into the decoder session on every call. The worker holds them as CPU tensors,
so every dispatch re-uploads 2 x 256x64x64 fp32 = ~8 MB — ~256 MB across a
32-dispatch run and ~1 GB across a 128-dispatch one.

With the flag on, the session is created with a `session_options` carrying a
`preferredOutputLocation` that maps `image_embeddings` and
`image_positional_embeddings` to `'gpu-buffer'`, so the encoder's two outputs
stay resident on the device and are handed straight back to the decoder.
`preferredOutputLocation` is keyed by output name, so naming these two is inert
for every session that does not produce them. **`pred_masks` and `iou_scores`
are deliberately not named**: the filter stage reads `pred_masks.data` on the
CPU, and moving it to a GPU buffer would trade one copy for a worse one.

Nothing may read `.data` on the two GPU-resident tensors. The worker only
forwards them into `model(...)`, which is exactly what makes this path
possible.

Two hazards, both bound here:

- **The session cache key must include the flag.** `loadSession` keys on the
  model id and dtype today. It gains `gpuResidentEmbeddings` as a third
  component. Without this, flipping the flag inside one worker silently reuses
  a session built the other way and the sweep measures a lie. (`keepRawMasks`
  must **not** enter the key — it changes nothing about the session.)
- **The path must fail loudly.** If transformers.js or the runtime will not
  accept a GPU-resident tensor as decoder input, the throw propagates through
  the worker's existing phase-tracking catch and arrives as a
  `SegmenterFailure` naming `encode` or `decode`. There is **no**
  catch-and-retry on CPU: a silent fallback would report a fast row that
  measured the very path it was supposed to replace. The runner records such a
  row as **failed** and moves on.

## `keepRawMasks` must be re-cut

PR #11 described `keepRawMasks` as client-side retention with the worker
untouched, and on that branch it was: the worker posted `RawMask[]` (coverage
plus area) and the flag only decided whether `createSegmenter` kept them.

M3/M4 changed that. On current `main` the worker encodes each surviving mask to
a 1-bit PNG data URL and posts `EncodedMask[]`; **not one coverage buffer
crosses the worker boundary any more**, which is the whole point of that
change. So the ported flag has to be honoured in the worker:

- `SegmenterOptions.keepRawMasks: boolean` — `false` in
  `DEFAULT_SEGMENTER_OPTIONS`.
- When set, the `done` message additionally carries
  `rawMasks: RawMask[]` (`{ coverage: Uint8Array; area: number }`), with the
  coverage buffers in the `postMessage` transfer list so they move rather than
  clone. When unset, the message and the transfer list are exactly what they
  are today.
- `createSegmenter` surfaces `SegmentationResult.rawMasks?`.

Off by default because a caller holding the result in React state pins tens of
megabytes — ~50 full-resolution coverage arrays at ~0.7 MB each at 16 points
per side. It exists so one run's masks can be compared against another's, and
for nothing else.

The alternative — decoding the PNGs back to coverage on the main thread — is
rejected: it adds a second route to a buffer the worker already has in hand,
and it would put PNG decode time inside the comparison.

## The sweep

### Axes and the default grid

decode path {`none`, `overlap`, `gpuEmbeddings`, `both`} x `batchSize` {8, 32}
x `dtype` {fp32, fp16}, at `pointsPerSide` 16, on the bundled sample image, 1
rep, with one warm-up run discarded. Sixteen measured rows: minutes, not hours.

**`batchSize: 64` is excluded from the default grid.** Issue #12 measured it
dying with `Array buffer allocation failed` inside `post_process_masks`, whose
allocation scales with `batch x width x height x 4`. It stays reachable through
config so the failure can be re-confirmed deliberately, but it does not burn a
row in every run.

The grid is config-driven: a `DEFAULT_SWEEP_CONFIG` in `playground/compare.ts`,
widened by `--full` (which adds `pointsPerSide` 32) or replaced wholesale by
`--config <path>`, with `--reps <n>`. **The full cross-product is a flag, not a
code change.** The resolved config is embedded in the output JSON, so the
report's numbers are reproducible from the report's own artifact.

### Where the logic lives

Grid expansion, row labelling, ranking and markdown rendering live in
`playground/compare.ts` — TypeScript, unit-tested under vitest, one
implementation. The Node runner reaches them **through the page**: the Compare
tab installs a `window.__decodeSweep` hook exposing `expandGrid` and
`toMarkdown`, the runner evaluates them in the browser, and no logic is
duplicated into an `.mjs` file where no test would reach it.

The hook is playground-only. Nothing under `src/` knows it exists, and
`scripts/smoke-build.mjs`'s guarantee about the published package surface is
untouched.

`compare.ts` keeps PR #11's `compareMaskSets` in spirit and in code: greedy
best-IoU pairing over the existing `pairwiseIoU` from `src/segmenter/core`,
with the `IOU_MATCH_FLOOR` below which a pair counts as unmatched on both
sides. There is no second IoU implementation anywhere in this change.

### The runner — `scripts/sweep-decode.mjs`

Invoked as `npm run sweep:decode`. A Node script using the **Playwright
library, not the test runner** — the test runner's retries, timeouts and
parallelism fight a serial GPU benchmark, and a retried GPU row is a different
measurement, not the same one again.

It:

1. Starts the playground through Vite's Node API against
   `playground/vite.config.ts` and reads back the resolved URL.
2. Launches **headed Chromium by default**. Headless Chromium frequently
   resolves no real WebGPU adapter and silently falls back to software, and
   software timings would make the whole sweep meaningless. `--headless` is
   available for smoke-testing the wiring.
3. **Asserts a hardware adapter before the first measurement**, by requesting
   an adapter in the page and reading its reported vendor and architecture. If
   there is no adapter, or the adapter is a known software rasterizer
   (SwiftShader, lavapipe, WARP), the runner aborts. `--allow-software`
   overrides, and either way the adapter's identity is written into the output
   JSON, so a software run is self-identifying rather than quietly comparable
   to a real one.
4. Opens the Compare tab and **walks every config in one page**, so the ONNX
   weights stay HTTP-cached across rows. For each row it sets the controls by
   `data-testid` and clicks run — the way a person does — then reads the
   completed run's JSON blob.
5. Discards the first (warm-up) row's numbers.
6. On a failed row — the playground's `run-error` surface — records
   `{ status: 'failed', phase, message }` and continues to the next row. One
   unsupported path does not cost the other fifteen.
7. Writes both artifacts, and exits non-zero only if **every** row failed.

### Timings only

The sweep ranks and does **not** gate on mask-set equality. It records
`counts.raw / afterFilter / afterNms` for every row, so a variant that is fast
because it silently dropped masks is visible in the table rather than hidden by
it, and it reports `compareMaskSets` agreement against the baseline row
wherever both sides retained masks — **reported, never enforced**. Correctness
of the two new paths is a matter for review and unit tests, not for the
benchmark.

## Output

`docs/measurements/<YYYY-MM-DD>-decode-sweep.md` (rendered with `compare.ts`'s
`toMarkdown`) and `<YYYY-MM-DD>-decode-sweep.json` beside it, both committed. A
second run on the same day gets a `-2`, `-3` suffix rather than silently
overwriting the first.

**The ranking metric is wall clock minus `model-load`**, carried as `budgetMs`,
matching issue #1's budget convention: `createSegmenter` terminates and
respawns the worker for every run, so every row pays a warm (HTTP-cached) model
load that is not part of the per-image cost. The markdown names the fastest
configuration explicitly, and carries the overlap caveat — that
`overlapDecodeFilter` moves time between stage counters rather than removing
it, so the `decode` column is not the ranking — directly above the table.

**Every row is labelled from the options captured with that result**, never
from the runner's or the page's current control state. The JSON carries, per
row: the full resolved `SegmenterOptions`, the `TimingReport`, `counts`,
`budgetMs`, and status. The adapter identity and the resolved grid config sit
at the top level.

## Playground changes

The Compare tab (`playground/CompareView.tsx`, ported and re-cut) is the
sweep's page, and stays usable by hand:

- Controls, each with a `data-testid`: `dtype` (fp32/fp16), `batch-size`
  (8/16/32/64), `points-per-side`, `overlap-decode-filter`,
  `gpu-resident-embeddings`, `keep-raw-masks`.
- A run control for the current configuration; the accumulated `compare-table`
  of rows run so far; `copy-markdown`; the `agreement-panel` when two rows
  retained masks.
- `run-json` — a `<pre>` carrying the completed run's captured
  `SegmentationResult` plus the resolved options that produced it. The runner
  reads exact numbers from here rather than from the table's rounded display
  text.
- A `createSegmenter` injection point, so the component is testable without
  WebGPU. This mirrors `runRow`'s existing injectable deps.

`SegmentView` is otherwise left alone, except for the two new decode-path
checkboxes beside the existing `compare-nms` one, so the manual tuning
affordance also exists on the tab people already use.

## Bringing the playground under a check that runs

- **`tsconfig.playground.json`** — extends the root config and includes
  `playground/**/*`. The `typecheck` script runs the root project and then this
  one.

  *Rejected: widening the root `include`.* `tsup` runs with `dts: true` against
  the root tsconfig, and the published declaration graph should not be able to
  see playground files. A second project file keeps the package build exactly
  as it is while still failing CI on a playground type error.

- **`vitest.config.ts`** gains `playground/**/*.test.tsx`, and
  `playground/CompareView.test.tsx` is added with the
  `// @vitest-environment jsdom` pragma the repo already uses in
  `src/SegmentViewer.test.tsx`. The suite's global environment stays `node`.

After this run, `CompareView.tsx` is covered by both `npm run typecheck` and
`npm test`. That is the point of the exercise.

## Error handling

| Condition | Behaviour |
|---|---|
| GPU-resident embeddings rejected by the runtime | throws; surfaces as `SegmenterFailure` naming its phase; **no CPU fallback** |
| Overlapped dispatch rejects while another is in flight | the abandoned promise gets a no-op `catch`; the run fails with the real phase |
| A row exhausts memory (`batchSize` 64, or overlap's higher peak) | recorded as a failed row; the sweep continues |
| No WebGPU adapter, or a software one | the runner aborts before the first measurement, unless `--allow-software` |
| Every row failed | the runner exits non-zero |

## Testing

**Unit tests** (`playground/compare.test.ts`, `playground/CompareView.test.tsx`,
`src/segmenter/createSegmenter.test.ts`):

- grid expansion — the default grid is exactly 16 rows, contains no
  `batchSize: 64`, and expands in a deterministic order with stable row ids;
  `--full` and an explicit config expand as specified.
- row labelling — a row's label and table cells are derived from the captured
  options, so a result captured under one configuration still labels correctly
  after the controls have moved on.
- markdown rendering — the column set, the `budgetMs` ranking, `counts` per
  row, the overlap caveat present, and a failed row rendered as failed.
- `compareMaskSets` — the ported cases, unchanged.
- option plumbing — `overlapDecodeFilter`, `gpuResidentEmbeddings` and
  `keepRawMasks` default to `false`, survive `createSegmenter` into the worker
  request, and `rawMasks` survives the worker to main-thread boundary, in the
  same shape as the existing `filterSubPhases` and `compareNms` passthrough
  tests.
- `CompareView` — renders, drives an injected stub segmenter, and emits a
  `run-json` blob matching the stub's result.

**What is deliberately not unit-tested, and why.** The worker's batch loop is
not unit-testable in this repo today: it is a top-level module body that binds
`globalThis.addEventListener` and calls into `@huggingface/transformers` with a
live WebGPU device, with no seam to inject a fake model. Extracting one purely
to assert "the second dispatch started before the first filter finished" would
be a test of the extraction, not of the worker. **The correctness of the two
decode paths is therefore bound to code review plus the sweep's own reported
mask agreement and `counts` columns** — a path that reorders work incorrectly
shows up as changed `afterNms` or collapsed agreement in the committed table.
Saying so plainly is better than a pure-function test that proves nothing about
the worker.

**Browser evidence.** The evidence for this run is the sweep script's own
headed, real-GPU run, committed under `docs/measurements/`. It is not the
`verify` stage, which runs headless where there is no hardware WebGPU adapter
and whose numbers would therefore be meaningless — which is precisely why the
runner refuses to produce them. **Every acceptance criterion below is
`(non-ui)`**, matching how every prior optimization run in this repo was
verified.

## Out of scope

- Choosing a default. This run measures; changing `DEFAULT_SEGMENTER_OPTIONS`
  to the winner is a follow-up with the committed table as its evidence.
- Moving `pred_masks` or `iou_scores` to GPU buffers — the filter stage reads
  them on the CPU.
- Any change to `nms`, `mask-encode` or `filter` beyond the loop restructuring
  that `overlapDecodeFilter` requires.
- Multi-crop, a second sample image, or a second IoU implementation.
- Replaying PR #11's branch onto `main`, or reviving anything from it not named
  above.

## Acceptance criteria

Every criterion is `(non-ui)`. The browser evidence for this run is the sweep
script's own headed, real-GPU run committed under `docs/measurements/` — the
`verify` stage runs headless, where there is no hardware WebGPU adapter and any
timing it produced would be meaningless. AC9 and AC10 are verified by reading
`playground/CompareView.tsx` and by running `npm run typecheck` and `npm test`,
not by opening a browser.

- AC1 (non-ui) — `SegmenterOptions` carries `overlapDecodeFilter`,
  `gpuResidentEmbeddings` and `keepRawMasks`, all `false` in
  `DEFAULT_SEGMENTER_OPTIONS`, and all three survive `createSegmenter` into the
  worker request; the worker honours all four combinations of the two decode
  flags.
- AC2 (non-ui) — with `overlapDecodeFilter` on, batch `b + 1`'s model dispatch
  is started before batch `b`'s filter block runs, and with it off the loop is
  the serial await-then-filter shape it has today; if the filter block throws
  while a dispatch is outstanding, the run rejects with the phase it died in
  and no unhandled promise rejection escapes.
- AC3 (non-ui) — the worker's session cache key includes
  `gpuResidentEmbeddings`, so flipping that flag cannot reuse a session built
  the other way, and `keepRawMasks` is absent from that key.
- AC4 (non-ui) — with `gpuResidentEmbeddings` on, exactly `image_embeddings`
  and `image_positional_embeddings` are requested as `'gpu-buffer'` and
  `pred_masks` stays on the CPU; if the runtime rejects the GPU-resident input,
  the failure propagates as a `SegmenterFailure` naming its phase, with no
  catch-and-retry on CPU anywhere in the path.
- AC5 (non-ui) — with `keepRawMasks` set, the `done` message carries `rawMasks`
  with the coverage buffers transferred rather than cloned, and
  `createSegmenter` surfaces them on `SegmentationResult.rawMasks`; with it
  unset the message is unchanged from today and no coverage buffer crosses the
  worker boundary.
- AC6 (non-ui) — grid expansion is unit-tested: the default grid is exactly 16
  rows (4 decode paths x `batchSize` {8, 32} x `dtype` {fp32, fp16} at
  `pointsPerSide` 16), contains no `batchSize: 64`, and expands in a
  deterministic order with stable row ids; a wider grid is reachable by config
  or flag without editing code.
- AC7 (non-ui) — every rendered row is labelled from the options captured with
  its result: a test that mutates the current control state after a result is
  captured still renders that result's own configuration.
- AC8 (non-ui) — `toMarkdown` renders, per row, the resolved options, the
  per-phase timings, `counts.raw / afterFilter / afterNms`, and
  `budgetMs = totalMs - model-load`; it ranks by `budgetMs`, names the fastest
  configuration, renders a failed row as failed rather than as fast, and
  carries the caveat that `overlapDecodeFilter` moves time between stage
  counters so the `decode` column is not the ranking.
- AC9 (non-ui) — the Compare tab exposes `data-testid` controls for `dtype`,
  `batch-size`, `points-per-side`, `overlap-decode-filter`,
  `gpu-resident-embeddings` and `keep-raw-masks`, a run control, and a
  `run-json` blob carrying the completed run's `SegmentationResult` plus the
  resolved options that produced it; `SegmentView` gains the two decode-path
  checkboxes beside the existing `compare-nms` one.
- AC10 (non-ui) — the compare harness is under checks that actually run:
  `npm run typecheck` type-checks `playground/**/*.tsx` including
  `CompareView.tsx`, and `npm test` executes a `CompareView` test, so a
  deliberate type error or a broken render in that file fails CI.
- AC11 (non-ui) — `npm run sweep:decode` exists and launches headed Chromium
  through the Playwright library rather than the test runner; it aborts before
  the first measurement when no adapter resolves or the adapter is a known
  software rasterizer, `--headless` and `--allow-software` are available, and
  the adapter's identity is recorded in the output JSON.
- AC12 (non-ui) — the runner discards one warm-up run, walks every remaining
  config in a single page so the weights stay HTTP-cached, records a row that
  fails (unsupported path, exhausted memory) as `failed` with its phase and
  message and continues with the rest, and exits non-zero only when every row
  failed.
- AC13 (non-ui) — a real-GPU run is committed: `docs/measurements/` contains a
  dated `-decode-sweep.json` and `-decode-sweep.md` from one headed run on a
  hardware adapter; the markdown ranks the configurations by `budgetMs` and
  names the fastest, with `counts` alongside every row; and the JSON embeds the
  resolved grid config and the adapter identity, so the run is reproducible
  from its own artifact. Producing this requires a machine with a hardware
  WebGPU adapter — it is not produced by the headless `verify` stage.
