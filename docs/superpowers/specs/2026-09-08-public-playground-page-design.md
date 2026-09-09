# A public "Try it" page for note-scanner (issue #11)

A dedicated public demo page, built from a new `site/` entry and deployed to
GitHub Pages at `https://botime.github.io/NoteScanner/`, so a stranger can
segment a photo in their own browser without installing anything.

Issue #11 — "Create a public GitHub page as a playground for people to try the
No Scanner." The issue body is empty, so the title is the only criterion it
states: a member of the public, with no checkout and no install, can reach a URL
and try the scanner. Everything else below was settled in the brainstorm.

## What it is

The page is the only public face the package has: `@botime/note-scanner`
publishes to GitHub Packages, not npm, and there is no docs site.

It is **demo only** — not a landing page for the package. No install line, no
usage snippet, no how-it-works essay.

## Structure

New `site/` directory at the repository root, holding:

- `index.html`
- `main.tsx` — mounts the page
- the page component
- `vite.config.ts` — `base: '/NoteScanner/'`, and the same
  `optimizeDeps: { include: ['@huggingface/transformers'] }` the playground
  needs, because the worker reaches that package only through a dynamic import
  the Vite scanner cannot see

It imports `SegmentViewer` from `src/`, and `createSegmenter`,
`isWebGPUAvailable`, `DEFAULT_SEGMENTER_OPTIONS`, `SegmenterFailure` and the
progress/result types from `src/segmenter`. **It imports nothing from
`playground/`.**

The alternative considered and rejected was reusing `playground/SegmentView.tsx`
behind a `mode` prop. The reason is not bundle safety — `SegmentView` imports
nothing from the benchmark, compare or boundary modules, so a public bundle
built from it would be clean. The reason is divergent UI: the public page wants
sample chips and download-progress copy and no timing table, the dev page wants
eleven option controls and p50/p95 per phase. Gating roughly half of a 409-line
component that has taken eight commits — five of them "expose <new option> on
the Segment view" — is the expensive part.

## What the page contains

- A title and one line: it runs SlimSAM on WebGPU, and the image never leaves
  the page. Inference is entirely client-side and the page should say so,
  because "drop your photo in" otherwise reads as an upload.
- Two selectable sample images.
- A file input that doubles as a drop target, for the visitor's own image.
- One **Run** button.
- A progress line while running.
- The `SegmentViewer`, with click-to-select segments.
- A plain `N segments · X s` stat line after a run.
- A footer link back to the repository.

No tabs. No option controls. No timing table.

When `isWebGPUAvailable()` is false, the Run control is replaced by the existing
amber "WebGPU required" notice, and no worker is ever spawned. There is no CPU
fallback by design.

## Fixed segmenter options

The page pins `DEFAULT_SEGMENTER_OPTIONS` and exposes no way to change them.
That is not an arbitrary default. From
`docs/measurements/2026-09-01-decode-sweep-3.md`, on an apple/metal-3 adapter:

| settings | budget | model-load | returned masks |
|---|---:|---:|---:|
| pps 16, fp32, batch 32, lowres, enclow | 7 067 ms | 616 ms | 34 |
| pps 32, fp32, batch 32, lowres, enclow | 23 834 ms | 642 ms | 53 |

Doubling points-per-side costs 3.4x wall clock for 19 more masks. The defaults
are already the fastest measured row, so there is nothing to tune and no knob
worth spending.

## Progress, honestly

A cold visit pays a model download before any compute, and `model-load` is
already its own phase in `SegmenterProgress`. The page distinguishes
"downloading the model (first run only)" from the per-batch phases rather than
showing one undifferentiated spinner — on a first visit the download is the
longest single wait, and a spinner that looks identical to a stalled page is the
worst outcome for a stranger evaluating the demo.

The honest figure is roughly 7 seconds of compute on a fast Mac, plus that
download. The page should not imply it is instant.

## Sample images

The two samples move to a shared top-level `samples/` directory, imported by
both the public page and the playground, with one README documenting provenance
for each. This is what keeps the public app off `playground/` without either app
reaching into the other's assets; `playground/SegmentView.tsx`'s import updates
to the new path.

- `cafe-table.jpg` — already committed, CC0, provenance documented, chosen for
  its several clearly separable objects.
- A new sticky-note / whiteboard photo, so a visitor trying a *note* scanner
  sees notes. It must be **verifiably CC0 or public domain**, downscaled under
  200 KB, and recorded in the README in the same format as the existing entry.
  If no such image can be verified, the page ships with the cafe table alone and
  the run says so explicitly. A committed binary whose provenance nobody can
  state is a binary nobody can safely publish — and this one is going on a
  public page.

## Data flow

The path `SegmentView` already proves:

`fetch(url)` -> `blob()` -> `createImageBitmap` ->
`segmenter.segment(bitmap, DEFAULT_SEGMENTER_OPTIONS, onProgress)` ->
`ViewerSegment[]` -> `<SegmentViewer>`.

Lifecycle matches the existing view: object URLs for visitor-supplied files are
revoked when the image is replaced and on unmount, the segmenter is created
lazily so a browser with no WebGPU never spawns a worker, and it is disposed on
unmount.

## Error handling

`SegmenterFailure` carries the phase it failed in, so a failure renders as
"Failed during <phase>: <message>" rather than a bare stack. Any other thrown
value renders with phase "unknown". A failed run leaves the page usable — the
visitor can pick another image and run again.

## Deployment

A new `.github/workflows/pages.yml`:

- triggers on push to `main`, plus `workflow_dispatch`
- `permissions: { pages: write, id-token: write, contents: read }`
- `actions/configure-pages` with `enablement: true`, so the first run turns
  Pages on without a manual settings visit
- builds the site, `actions/upload-pages-artifact`, `actions/deploy-pages`
- a `github-pages` environment

The existing `ci.yml` is untouched. The site build gets its own npm script.

## Testing

- Unit tests (vitest) for any pure helper the page introduces.
- A Playwright spec covering the page:
  - it renders, and a sample image loads
  - the WebGPU-absent path shows the notice and offers no Run button
  - **the guard**: the public page exposes no option controls. This is what
    makes the separation from the dev playground enforceable rather than a
    convention.

GitHub's runners have no WebGPU adapter, so no CI test can run real inference.
Verifying an actual segmentation on the deployed page is a manual check on a
WebGPU-capable machine, and the acceptance criteria say so rather than
pretending CI covers it.

## Out of scope

- Publishing the dev playground (Benchmark / Compare / Boundary) anywhere.
- Any package-marketing content on the page.
- A custom domain, analytics, or a CPU fallback.

## Acceptance criteria

AC6 is the one criterion no automated check can reach: GitHub's runners expose
no WebGPU adapter, so a real segmentation — and with it the progress line, the
stat line and click-to-select on real output — is confirmed by hand on a
WebGPU-capable machine and recorded in the run, not by CI. Everything a browser
can observe without a GPU adapter is tagged `(ui)` and is checked by opening the
built site.

- AC1 (ui) — the site built from `site/` and served under the base path
  `/NoteScanner/` loads with no failed asset requests and no console errors, and
  renders a title, a one-line statement that inference runs in the visitor's own
  browser on WebGPU and that the image never leaves the page, and a footer link
  back to the repository. No install line, usage snippet or how-it-works prose
  appears.
- AC2 (ui) — a sample image is selected and displayed on first load with no
  interaction; where a second verified sample ships, both are offered as
  selectable choices and picking the other one replaces the displayed image.
- AC3 (ui) — the page offers a file input that also accepts a dropped file, and
  choosing an image replaces the displayed sample with it.
- AC4 (ui) — in a browser that exposes no WebGPU adapter, the page shows the
  amber "WebGPU required" notice in place of the Run control, offers no Run
  button, and starts no segmentation worker.
- AC5 (ui) — the page exposes no segmenter option controls anywhere in its
  rendered UI (no selects, checkboxes, radios or numeric inputs for segmenter
  options), no tabs, and no timing table. Only the sample choices, the file
  input and the single Run control are interactive.
- AC6 (non-ui) — on a WebGPU-capable machine, a manual run of the page segments
  a sample end to end: while running, the progress line names the model download
  ("first run only") distinctly from the per-batch phases; on completion the
  viewer shows the segments, clicking a segment selects it, and a
  `N segments · X s` stat line is shown. The run records what was observed
  rather than claiming CI covered it.
- AC7 (non-ui) — a `SegmenterFailure` renders as "Failed during <phase>:
  <message>", any other thrown value renders with phase "unknown", and after a
  failure the page is still usable: another image can be chosen and run again.
- AC8 (non-ui) — nothing under `site/` imports from `playground/`; the site
  imports only from `src/`, the shared `samples/` directory and third-party
  packages. `playground/` likewise does not import from `site/`.
- AC9 (non-ui) — both sample images live in a top-level `samples/` directory
  with a single README documenting provenance for each in the format the
  existing entry uses; `playground/SegmentView.tsx` imports its sample from that
  new path and the playground still loads it. The second sample ships only if it
  is verifiably CC0 or public domain and under 200 KB; if no such image can be
  verified, the page ships with `cafe-table.jpg` alone and the run states that
  explicitly.
- AC10 (non-ui) — the page pins `DEFAULT_SEGMENTER_OPTIONS` and passes no
  overridden segmenter options.
- AC11 (non-ui) — the site build runs from its own npm script, with
  `site/vite.config.ts` setting `base: '/NoteScanner/'` and
  `optimizeDeps: { include: ['@huggingface/transformers'] }`.
- AC12 (non-ui) — `.github/workflows/pages.yml` exists, triggers on push to
  `main` and on `workflow_dispatch`, declares
  `permissions: { pages: write, id-token: write, contents: read }`, uses
  `actions/configure-pages` with `enablement: true`, builds the site and deploys
  it with `actions/upload-pages-artifact` and `actions/deploy-pages` under a
  `github-pages` environment. `.github/workflows/ci.yml` is unchanged.
- AC13 (non-ui) — a Playwright spec covers the public page for AC1, AC2, AC4 and
  AC5 and passes in the configured engines; unit tests exist for any pure helper
  the page introduces; `npm run test`, `npm run test:browser` and
  `npm run typecheck` all pass.
