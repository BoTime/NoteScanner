# Public "Try it" Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a public demo page at `https://botime.github.io/NoteScanner/`, built from a new `site/` Vite entry, where a stranger can segment a photo entirely in their own browser.

**Architecture:** A second Vite app (`site/`) alongside the existing `playground/`. It imports `SegmentViewer` from `src/` and the segmenter from `src/segmenter`, and **nothing** from `playground/`. The two sample photos move to a shared top-level `samples/` directory that both apps import. A `pages.yml` workflow builds `site/` and deploys it to GitHub Pages. Verification is a Playwright spec against the real production build served under the real base path, plus a WebGPU-gated end-to-end spec for the one criterion CI's GPU-less runners cannot reach.

**Tech Stack:** Vite 7, React 19, TypeScript, vitest (unit + jsdom component), Playwright 1.62 (chromium/webkit/firefox), GitHub Actions Pages deployment.

**Spec:** `docs/superpowers/specs/2026-09-08-public-playground-page-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- `site/vite.config.ts` sets `base: '/NoteScanner/'` and `optimizeDeps: { include: ['@huggingface/transformers'] }` — exact values, verbatim.
- Nothing under `site/` imports from `playground/`; nothing under `playground/` imports from `site/`. Task 2 ships the test that enforces this.
- The page pins `DEFAULT_SEGMENTER_OPTIONS` and passes **no** overridden segmenter options. It never constructs a partial options object.
- The page's rendered UI contains **no** `<select>`, no `input[type=checkbox]`, no `input[type=radio]`, no `input[type=number]`, no `<table>`, and no tab navigation. Its only `<input>` is the single `type="file"` control.
- No install line, no usage snippet, no how-it-works prose anywhere on the page. It is a demo, not a landing page.
- No CPU fallback. When `isWebGPUAvailable()` is false the page renders the amber notice **in place of** the Run control and never calls `createSegmenter()`.
- Both committed sample images stay under 200 KB and each has a provenance entry in `samples/README.md`.
- `.github/workflows/ci.yml` is not modified by this plan.
- `.gitignore` already ignores `dist` with no leading slash, so `site/dist` is ignored at any depth. Do **not** add a redundant rule.

## Facts established by reading the repo — do not re-derive these

1. **`model-load` progress is posted only when the download has already finished.** `src/segmenter/worker/segmenter.worker.ts:153` posts `{ phase: 'model-load', done: 1, total: 1, ms: elapsed }` *after* the model is loaded. Nothing is posted during the download. So the `progress === null` window **is** the model-download window, and the page's copy must be built around that, not around a live `model-load` event. This is the single most load-bearing fact in the plan; the spec's "distinguishes the download from the per-batch phases" is satisfied by the null-state copy.
2. **Two playground files import the sample**, not one: `playground/SegmentView.tsx:14` and `playground/CompareView.tsx:33`. The spec mentions only the first.
3. **`SegmentViewer` renders `<button>` elements but no `<select>`, `<input>` or `<table>`.** So an AC5 assertion may count selects/checkboxes/radios/number-inputs/tables document-wide, but must **not** assert a document-wide button count.
4. **`SegmentViewer` paints to a `<canvas>`; it renders no `<img>`.** A browser assertion that "a sample is displayed" therefore needs an explicit wrapper carrying the decoded dimensions — see Task 2's `data-testid="site-image"`.
5. **`isWebGPUAvailable()` is `'gpu' in navigator`** (`src/segmenter/createSegmenter.ts`). `navigator.gpu` lives on `Navigator.prototype`, so a Playwright init script must `delete Navigator.prototype.gpu` — assigning `undefined` leaves `'gpu' in navigator` true and would silently not test the no-WebGPU path.
6. **`docs/measurements/*.md` reference `playground/sample/cafe-table.jpg`.** Those are records of runs performed against that path at that time. Leave them alone — rewriting a historical measurement's stated input falsifies the record. Only live code imports move.

---

## File Structure

**Created**

| path | responsibility |
|---|---|
| `samples/cafe-table.jpg` | existing sample, moved with `git mv` |
| `samples/sticky-notes.jpg` | new PD sample (see Task 1 for the only download recipe that works here) |
| `samples/README.md` | provenance for both, moved with `git mv` and extended |
| `site/index.html` | Vite entry document |
| `site/main.tsx` | mounts `TryItPage`, imports `../src/styles.css` |
| `site/TryItPage.tsx` | the whole page component |
| `site/page-text.ts` | the two pure copy helpers (`progressLine`, `statLine`) |
| `site/page-text.test.ts` | unit tests for those helpers |
| `site/TryItPage.test.tsx` | jsdom component test (error path, drop path, stat line) |
| `site/imports.test.ts` | the AC8 guard, both directions |
| `site/vite-env.d.ts` | `/// <reference types="vite/client" />` |
| `site/vite.config.ts` | base, optimizeDeps, `worker.format`, dev + preview ports |
| `tsconfig.site.json` | typechecks `site/**` |
| `tests/browser/site-url.ts` | the one place the site preview URL is written |
| `tests/browser/site.spec.ts` | AC1, AC2, AC3, AC4, AC5 in three engines |
| `tests/browser/site-webgpu.spec.ts` | AC6 end to end on a real adapter |
| `.github/workflows/pages.yml` | build + deploy to GitHub Pages |

**Modified**

| path | change |
|---|---|
| `playground/SegmentView.tsx:14` | sample import path |
| `playground/CompareView.tsx:33` | sample import path |
| `package.json` | `site`, `build:site`, `preview:site`, `test:site:gpu` scripts; `typecheck` gains `tsconfig.site.json` |
| `vitest.config.ts` | `include` gains `site/**/*.test.ts(x)` |
| `playwright.config.ts` | `webServer` becomes an array; second entry serves the built site |
| `README.md` | Development section gains the new scripts and the public URL |

---

## Task 1: The shared `samples/` directory

Moves both apps' sample assets out of `playground/` and adds the second, licence-verified photo. Nothing in `site/` exists yet, so this task is reviewable on its own: the playground must still build, typecheck and pass its tests with the sample at its new path.

**Files:**
- Move: `playground/sample/cafe-table.jpg` -> `samples/cafe-table.jpg`
- Move: `playground/sample/README.md` -> `samples/README.md` (then edit)
- Create: `samples/sticky-notes.jpg`
- Modify: `playground/SegmentView.tsx:14`
- Modify: `playground/CompareView.tsx:33`

**Interfaces:**
- Consumes: nothing.
- Produces: two importable asset paths used by Task 2 — `../samples/cafe-table.jpg` and `../samples/sticky-notes.jpg`, both resolved from a file inside `site/`.

- [ ] **Step 1: Move the directory with git, preserving history**

```bash
mkdir -p samples
git mv playground/sample/cafe-table.jpg samples/cafe-table.jpg
git mv playground/sample/README.md samples/README.md
rmdir playground/sample
```

- [ ] **Step 2: Repoint both playground imports**

`playground/SegmentView.tsx` line 14 and `playground/CompareView.tsx` line 33 both read:

```ts
import sampleUrl from './sample/cafe-table.jpg';
```

Change both to:

```ts
import sampleUrl from '../samples/cafe-table.jpg';
```

Then prove no reference to the old path survives in live code:

```bash
grep -rn "sample/cafe-table" --include='*.ts' --include='*.tsx' --include='*.html' playground src tests scripts
```

Expected: no output. (`docs/measurements/*.md` still names the old path and stays as-is — see fact 6.)

- [ ] **Step 3: Fetch the second sample**

`upload.wikimedia.org` and `thumb.wikimedia.org` return HTTP 400 from Varnish in this environment for every URL, with or without a User-Agent. Do not spend time there. `Special:FilePath` on `commons.wikimedia.org` follows the redirect server-side and works:

```bash
curl -sL -A "NoteScannerBot/1.0 (https://github.com/BoTime/NoteScanner)" \
  -o samples/sticky-notes.jpg \
  "https://commons.wikimedia.org/wiki/Special:FilePath/Off_the_wall_ideas_sticky_notes_SEI_2018_(41779807090).jpg?width=800"
```

- [ ] **Step 4: Verify what actually landed before committing it**

A failed fetch writes a ~2 KB HTML error page under a `.jpg` name, so check the type and the size, not just that the file exists:

```bash
file samples/sticky-notes.jpg
wc -c < samples/sticky-notes.jpg
```

Expected: `file` says `JPEG image data`, and the byte count is under 204800. The Commons original is 768x1024 / ~172 KB, so no downscale should be needed. If the fetched file exceeds 200 KB, re-encode it down (`sips -s format jpeg -s formatOptions 45 samples/sticky-notes.jpg --out samples/sticky-notes.jpg`) and record the re-encode in the README's **Modifications** field.

If the fetch cannot be made to produce a JPEG at all, **stop and report it**: the spec's fallback is that the page ships with `cafe-table.jpg` alone and the run says so explicitly (AC9). Do not substitute a different image without re-verifying its licence.

- [ ] **Step 5: Rewrite `samples/README.md` to cover both**

The heading changes because the directory is no longer playground-specific, and the closing note stays because it is the reason the file exists. Write the file as:

```markdown
# Sample images

Shared by both apps in this repo: the public page (`site/`) and the dev
playground (`playground/`). They live here, at the top level, so neither app
reaches into the other's assets.

## `cafe-table.jpg`

The photo both apps preload first. Chosen for its several clearly separable
objects — croissant, wooden plate, fork, spoon, glass of berries, pine cones, a
paper tag and a hand — which is what makes an everything-mode segmentation
result readable at a glance.

- **Source:** <https://commons.wikimedia.org/wiki/File:Rustic_Cafe_Table_(Unsplash).jpg>
- **Originally published:** <https://unsplash.com/photos/L_ENUhk011o>
- **Author:** Kawin Harasai
- **Licence:** CC0 1.0 Universal (Public Domain Dedication) —
  <https://creativecommons.org/publicdomain/zero/1.0/>
- **Modifications:** downscaled from 4018x2548 to 1024x649 and re-encoded as
  JPEG quality 45, to keep the committed file under 200 KB.

## `sticky-notes.jpg`

The public page's second sample: a visitor trying a *note* scanner should be
able to segment notes. Roughly sixty well-separated, high-contrast sticky notes
on glass, no identifiable faces, generic workshop content. Note density matters
— the page runs at `pointsPerSide: 16` (256 sample points), and this photo's
note count sits in the range that grid can actually resolve.

- **Source:** <https://commons.wikimedia.org/wiki/File:Off_the_wall_ideas_sticky_notes_SEI_2018_(41779807090).jpg>
- **Originally published:** <https://www.flickr.com/photos/156788110@N04/41779807090>
- **Author:** embljusocmedia (U.S. Embassy in Ljubljana Flickr stream), 2018-07-23
- **Licence:** Public domain as a work of the U.S. Department of State
  (`{{PD-USGov-DOS}}`) —
  <https://commons.wikimedia.org/wiki/Template:PD-USGov-DOS>. Independently
  Flickr-reviewed by `FlickreviewR 2` on 2024-08-16, review licence "Public
  Domain Mark". The basis is the U.S. Government authorship, not a licence tag
  an uploader applied to someone else's photo.
- **Modifications:** fetched at `width=800` via the Commons `Special:FilePath`
  renderer; otherwise unmodified.

CC0 and public domain waive copyright, so no attribution is legally required.
It is recorded here anyway: a committed binary whose provenance nobody can
state is a binary nobody can safely publish — and these two are going on a
public web page. Neither file is shipped in the npm tarball; `package.json`
sets `"files": ["dist"]`.
```

If Step 4 forced a re-encode, edit the `sticky-notes.jpg` **Modifications** line to say so — the field must describe the committed bytes, not the intended ones.

- [ ] **Step 6: Run the playground's own tests and typecheck**

```bash
npm run test
npm run typecheck
```

Expected: PASS. These exercise `playground/SegmentView.test.tsx` and `playground/CompareView.test.tsx`, both of which render components importing the moved asset — a broken import path fails resolution here.

- [ ] **Step 7: Prove the playground still serves the moved asset in a real browser**

`samples/` is outside the playground's Vite root (`playground/`), so this is a genuine `server.fs` question, not a formality.

```bash
npm run test:browser -- --project=chromium
```

Expected: PASS. If Vite refuses to serve a file outside its root, add `server: { fs: { allow: [path.resolve(__dirname, '..')] } }` to `playground/vite.config.ts` and re-run. (Vite's default workspace root is the directory holding `package-lock.json`, i.e. the repo root, so this is expected to work unchanged — confirm it rather than assume it.)

- [ ] **Step 8: Commit**

```bash
git add samples playground/SegmentView.tsx playground/CompareView.tsx
git commit -m "refactor(samples): move sample photos to a shared top-level samples/"
```

---

## Task 2: The `site/` app

The whole public page: config, entry, component, pure helpers, and the three test files covering the parts a browser cannot reach cheaply. Ends with a **production build that actually runs** — the site's build is the artifact GitHub Pages serves, and no unit test exercises it.

**Files:**
- Create: `site/vite.config.ts`, `site/vite-env.d.ts`, `site/index.html`, `site/main.tsx`, `site/page-text.ts`, `site/TryItPage.tsx`
- Test: `site/page-text.test.ts`, `site/imports.test.ts`, `site/TryItPage.test.tsx`
- Create: `tsconfig.site.json`
- Modify: `package.json` (scripts), `vitest.config.ts` (include)

**Interfaces:**
- Consumes: `samples/cafe-table.jpg`, `samples/sticky-notes.jpg` (Task 1).
- Produces, for Task 3:
  - `npm run build:site` writes `site/dist/`
  - `npm run preview:site` builds, then serves `site/dist` at `http://localhost:5181/NoteScanner/`
  - `progressLine(progress: SegmenterProgress | null): string`
  - `statLine(segmentCount: number, totalMs: number): string`
  - DOM contract, every element of which is asserted in Task 3:
    - `h1` with text `Try note-scanner`
    - `[data-testid="site-tagline"]`
    - `[data-testid="site-sample-cafe-table"]` and `[data-testid="site-sample-sticky-notes"]`, buttons carrying `aria-pressed`
    - `[data-testid="site-file-input"]`, the only input on the page, `type="file"`, `accept="image/*"`
    - `[data-testid="run-segmentation"]`, present only when WebGPU is available
    - `[data-testid="webgpu-required"]`, present only when it is not
    - `[data-testid="run-progress"]`, `[data-testid="run-error"]`, `[data-testid="run-stats"]`
    - `[data-testid="site-image"]`, wrapping the viewer, carrying `data-src`, `data-width`, `data-height`
    - `[data-testid="site-controls"]`, holding the chips, the file input and the Run control or notice; also the drop target
    - `[data-testid="site-footer"]`, containing the repository link

- [ ] **Step 1: Write the failing unit tests for the two copy helpers**

Create `site/page-text.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { progressLine, statLine } from './page-text';

describe('progressLine', () => {
  // The worker posts `model-load` only AFTER the download finishes
  // (segmenter.worker.ts:153). So the null window IS the download window, and
  // that is the window this copy has to name.
  it('names the model download while nothing has been reported yet', () => {
    expect(progressLine(null)).toBe('Downloading the model (first run only)…');
  });

  it('reports the model as loaded once model-load arrives', () => {
    expect(progressLine({ phase: 'model-load', done: 1, total: 1, ms: 616 })).toBe(
      'Model ready — segmenting…',
    );
  });

  it('names per-batch phases with their counts', () => {
    expect(progressLine({ phase: 'decode', done: 3, total: 8, ms: 12 })).toBe('decode 3/8');
    expect(progressLine({ phase: 'nms', done: 1, total: 1, ms: 4 })).toBe('nms 1/1');
  });

  it('never labels a per-batch phase as a download', () => {
    for (const phase of ['encode', 'decode', 'filter', 'nms', 'resample', 'mask-encode'] as const) {
      expect(progressLine({ phase, done: 1, total: 2, ms: 1 })).not.toMatch(/download/i);
    }
  });
});

describe('statLine', () => {
  it('renders the count and seconds to one decimal', () => {
    expect(statLine(34, 7067)).toBe('34 segments · 7.1 s');
  });

  it('renders a zero-segment run rather than hiding it', () => {
    expect(statLine(0, 812)).toBe('0 segments · 0.8 s');
  });
});
```

- [ ] **Step 2: Run it and watch it fail for the right reason**

```bash
npx vitest run site/page-text.test.ts
```

Expected: FAIL — vitest reports no test files matched, because `vitest.config.ts` does not include `site/`. Add `'site/**/*.test.ts'` and `'site/**/*.test.tsx'` to the `include` array in `vitest.config.ts`, keeping the existing four entries, then re-run.

Expected on the second run: FAIL with `Failed to resolve import "./page-text"`.

- [ ] **Step 3: Write the helpers**

Create `site/page-text.ts`:

```ts
import type { SegmenterProgress } from '../src/segmenter';

/**
 * One line of honest progress.
 *
 * The `model-load` phase is reported by the worker only once the model is
 * already loaded — the worker posts it with the elapsed time attached, after
 * the fact. Nothing at all is posted while the weights are downloading. So the
 * interval where `progress` is still `null` is exactly the interval a
 * first-time visitor spends waiting on that download, and naming it is the
 * whole point: an undifferentiated spinner during the longest wait of the visit
 * is indistinguishable from a hung tab.
 *
 * On a warm second run that window is brief; "(first run only)" is what keeps
 * the line honest in both cases.
 */
export function progressLine(progress: SegmenterProgress | null): string {
  if (!progress) return 'Downloading the model (first run only)…';
  if (progress.phase === 'model-load') return 'Model ready — segmenting…';
  return `${progress.phase} ${progress.done}/${progress.total}`;
}

/** The post-run line: how many segments came back, and how long it took. */
export function statLine(segmentCount: number, totalMs: number): string {
  return `${segmentCount} segments · ${(totalMs / 1000).toFixed(1)} s`;
}
```

- [ ] **Step 4: Run the unit tests**

```bash
npx vitest run site/page-text.test.ts
```

Expected: PASS, 6 tests.

- [ ] **Step 5: Write the AC8 import guard**

Create `site/imports.test.ts`. It inspects **extracted import specifiers**, never raw file text — the word "playground" appears in comments across this repo, including in comments this plan itself mandates, so a raw-text grep would fail for the wrong reason or match its own prose and pass for the wrong reason.

```ts
import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect } from 'vitest';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function sourceFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) out.push(...sourceFiles(full));
    else if (/\.(ts|tsx|html)$/.test(entry)) out.push(full);
  }
  return out;
}

/** Every module specifier: static imports, re-exports, dynamic import(), and
 *  the script src in an index.html. */
function specifiers(file: string): string[] {
  const text = readFileSync(file, 'utf8');
  const found: string[] = [];
  const patterns = [
    /(?:^|\s)(?:import|export)\s[^;]*?from\s*['"]([^'"]+)['"]/g,
    /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
    /<script[^>]+src=['"]([^'"]+)['"]/g,
  ];
  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) found.push(match[1]);
  }
  return found;
}

const siteFiles = sourceFiles(path.join(repoRoot, 'site'));
const playgroundFiles = sourceFiles(path.join(repoRoot, 'playground'));

describe('the public site and the dev playground stay separate (AC8)', () => {
  it('found files on both sides to check', () => {
    // Without this, the two assertions below pass by iterating nothing.
    expect(siteFiles.length).toBeGreaterThan(0);
    expect(playgroundFiles.length).toBeGreaterThan(0);
  });

  it('nothing under site/ imports from playground/', () => {
    for (const file of siteFiles) {
      for (const spec of specifiers(file)) {
        expect(spec, `${path.relative(repoRoot, file)} imports ${spec}`).not.toMatch(
          /(^|\/)playground(\/|$)/,
        );
      }
    }
  });

  it('nothing under playground/ imports from site/', () => {
    for (const file of playgroundFiles) {
      for (const spec of specifiers(file)) {
        expect(spec, `${path.relative(repoRoot, file)} imports ${spec}`).not.toMatch(
          /(^|\/)site(\/|$)/,
        );
      }
    }
  });
});
```

Run it:

```bash
npx vitest run site/imports.test.ts
```

Expected: PASS, 3 tests. Re-run it after Step 8 adds the page component and confirm it still passes with more files in scope.

- [ ] **Step 6: Write the site's Vite config**

Create `site/vite.config.ts`:

```ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  root: __dirname,
  // Served from https://botime.github.io/NoteScanner/, so every emitted asset
  // URL has to carry the repository path segment.
  base: '/NoteScanner/',
  plugins: [react()],
  // The segmenter worker is constructed as a module Worker and reaches
  // @huggingface/transformers through a DYNAMIC import. Vite's default worker
  // output format is 'iife', which cannot express a dynamic import, so a
  // production build of an app that uses the worker fails or ships a broken
  // one unless the worker is emitted as an ES module. The dev server serves
  // module workers natively, which is why the playground's config never needed
  // this: nothing in this repo has ever production-built an app using it.
  worker: { format: 'es' },
  // Same reason as the playground's config: @huggingface/transformers is
  // reachable only through that dynamic import inside the worker, so Vite's
  // dependency scanner does not see it and would otherwise discover it on the
  // first inference and force a full-page reload mid-run.
  optimizeDeps: { include: ['@huggingface/transformers'] },
  // 5181, one past the playground's 5180, so both can run at once. The preview
  // port is the one tests/browser/site-url.ts encodes; if the two ever
  // disagree, Playwright's webServer wait fails immediately and loudly.
  server: { port: 5181, strictPort: true },
  preview: { port: 5181, strictPort: true },
});
```

Create `site/vite-env.d.ts`:

```ts
/// <reference types="vite/client" />
```

- [ ] **Step 7: Write the entry document and the mount**

Create `site/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Try note-scanner</title>
    <!-- The browser requests /favicon.ico at the ORIGIN root on its own, which
         under base '/NoteScanner/' is a path this site does not serve — a 404
         that AC1's "no failed asset requests" assertion would (correctly) catch.
         This declares the icon instead of leaving it to that guess. Removing
         the request is the fix; filtering it out of the assertion is not. -->
    <link rel="icon" href="data:," />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="./main.tsx"></script>
  </body>
</html>
```

Create `site/main.tsx`:

```tsx
import { createRoot } from 'react-dom/client';
import { TryItPage } from './TryItPage';
import '../src/styles.css';

createRoot(document.getElementById('root')!).render(<TryItPage />);
```

- [ ] **Step 8: Write the page component**

Create `site/TryItPage.tsx`. The lifecycle deliberately mirrors `playground/SegmentView.tsx` — that path is proven — with the option controls, the timing table and the mask-count line removed, and sample chips plus the two copy helpers added.

```tsx
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SegmentViewer, type ViewerSegment } from '../src';
import {
  DEFAULT_SEGMENTER_OPTIONS,
  SegmenterFailure,
  createSegmenter,
  isWebGPUAvailable,
  type SegmentationResult,
  type SegmenterProgress,
} from '../src/segmenter';
import cafeTableUrl from '../samples/cafe-table.jpg';
import stickyNotesUrl from '../samples/sticky-notes.jpg';
import { progressLine, statLine } from './page-text';

const REPO_URL = 'https://github.com/BoTime/NoteScanner';

const SAMPLES = [
  { id: 'cafe-table', label: 'Café table', url: cafeTableUrl },
  { id: 'sticky-notes', label: 'Sticky notes', url: stickyNotesUrl },
] as const;

interface LoadedImage {
  url: string;
  width: number;
  height: number;
  /** Set when we minted the url ourselves and therefore have to revoke it. */
  objectUrl: boolean;
}

async function loadImage(url: string, objectUrl: boolean): Promise<LoadedImage> {
  const img = new Image();
  img.src = url;
  await img.decode();
  return { url, width: img.naturalWidth, height: img.naturalHeight, objectUrl };
}

export function TryItPage() {
  // Probed once: a GPU adapter does not appear part-way through a session, and
  // re-probing per render would make the notice flicker.
  const webgpu = useMemo(() => isWebGPUAvailable(), []);
  const [image, setImage] = useState<LoadedImage | null>(null);
  const [sampleId, setSampleId] = useState<string | null>(SAMPLES[0].id);
  const [segments, setSegments] = useState<ViewerSegment[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<SegmenterProgress | null>(null);
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [error, setError] = useState<{ phase: string; message: string } | null>(null);
  const segmenterRef = useRef<ReturnType<typeof createSegmenter> | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const replaceImage = useCallback((next: LoadedImage, nextSampleId: string | null) => {
    // Revoked here rather than inside a state updater: React re-invokes
    // updaters in StrictMode, and a side effect in one fires twice.
    if (objectUrlRef.current && objectUrlRef.current !== next.url) {
      URL.revokeObjectURL(objectUrlRef.current);
    }
    objectUrlRef.current = next.objectUrl ? next.url : null;
    setImage(next);
    setSampleId(nextSampleId);
    setSegments([]);
    setSelected(new Set());
    setResult(null);
    setError(null);
    setProgress(null);
  }, []);

  // A sample is on screen before the visitor does anything.
  useEffect(() => {
    let cancelled = false;
    void loadImage(SAMPLES[0].url, false).then((loaded) => {
      if (!cancelled) setImage(loaded);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(
    () => () => {
      segmenterRef.current?.dispose();
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    },
    [],
  );

  async function pickSample(sample: (typeof SAMPLES)[number]) {
    replaceImage(await loadImage(sample.url, false), sample.id);
  }

  async function pickFile(file: File | undefined) {
    if (!file) return;
    replaceImage(await loadImage(URL.createObjectURL(file), true), null);
  }

  async function run() {
    if (!image) return;
    setRunning(true);
    setError(null);
    setResult(null);
    setSegments([]);
    setSelected(new Set());
    setProgress(null);

    try {
      const blob = await (await fetch(image.url)).blob();
      const bitmap = await createImageBitmap(blob);
      // Created lazily so a browser with no WebGPU never spawns a worker.
      segmenterRef.current ??= createSegmenter();
      const outcome = await segmenterRef.current.segment(
        bitmap,
        DEFAULT_SEGMENTER_OPTIONS,
        setProgress,
      );
      setSegments(outcome.segments);
      setResult(outcome);
    } catch (thrown) {
      const failure = thrown instanceof SegmenterFailure ? thrown : null;
      setError({
        phase: failure?.phase ?? 'unknown',
        message: thrown instanceof Error ? thrown.message : String(thrown),
      });
    } finally {
      setRunning(false);
    }
  }

  return (
    <main style={{ padding: 16, fontFamily: 'system-ui', maxWidth: 1100, margin: '0 auto' }}>
      <h1 style={{ marginBottom: 4 }}>Try note-scanner</h1>
      <p data-testid="site-tagline" style={{ marginTop: 0, color: '#374151' }}>
        Segments a photo into individual notes and objects with SlimSAM, running on{' '}
        <strong>WebGPU in your own browser</strong>. Your image never leaves this page —
        nothing is uploaded.
      </p>

      <section
        data-testid="site-controls"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          void pickFile(e.dataTransfer.files[0]);
        }}
        style={{ border: '1px dashed #9ca3af', borderRadius: 6, padding: 12 }}
      >
        <p style={{ margin: '0 0 8px' }}>
          {SAMPLES.map((sample) => (
            <button
              key={sample.id}
              data-testid={`site-sample-${sample.id}`}
              type="button"
              aria-pressed={sampleId === sample.id}
              disabled={running}
              onClick={() => void pickSample(sample)}
              style={{ marginRight: 8 }}
            >
              {sample.label}
            </button>
          ))}
        </p>
        <p style={{ margin: '0 0 8px' }}>
          <label>
            Or use your own image (drop one here too):{' '}
            <input
              data-testid="site-file-input"
              type="file"
              accept="image/*"
              disabled={running}
              onChange={(e) => void pickFile(e.target.files?.[0])}
            />
          </label>
        </p>

        {webgpu ? (
          <p style={{ margin: 0 }}>
            <button
              data-testid="run-segmentation"
              type="button"
              disabled={running || !image}
              onClick={() => void run()}
            >
              {running ? 'Running…' : 'Run'}
            </button>
          </p>
        ) : (
          <section
            data-testid="webgpu-required"
            style={{ border: '1px solid #f59e0b', borderRadius: 6, padding: 12 }}
          >
            <h2 style={{ margin: '0 0 4px', fontSize: '1rem' }}>WebGPU required</h2>
            <p style={{ margin: 0 }}>
              This runs the whole segmentation model in your browser on{' '}
              <strong>WebGPU</strong>. This browser exposes no <code>navigator.gpu</code>{' '}
              adapter, so nothing has been downloaded and no worker has been started.
              Chrome or Edge 113+, Firefox 141+, or Safari 26+ on a supported GPU can run
              it. There is no CPU fallback by design — on CPU a single pass takes minutes.
            </p>
          </section>
        )}
      </section>

      {running && (
        <p data-testid="run-progress" role="status">
          {progressLine(progress)}
        </p>
      )}

      {error && (
        <p data-testid="run-error" role="alert">
          Failed during <strong>{error.phase}</strong>: {error.message}
        </p>
      )}

      {result && (
        <p data-testid="run-stats">{statLine(result.segments.length, result.timings.totalMs)}</p>
      )}

      {image && (
        <div
          data-testid="site-image"
          data-src={image.url}
          data-width={image.width}
          data-height={image.height}
        >
          <SegmentViewer
            key={`${image.url}-${segments.length}`}
            imageUrl={image.url}
            imageWidth={image.width}
            imageHeight={image.height}
            segments={segments}
            initialSelectedIds={selected}
            onSelectionChange={setSelected}
            onCreateSegment={async () => {}}
            maxHeight="70vh"
          />
        </div>
      )}

      <footer data-testid="site-footer" style={{ marginTop: 24, color: '#6b7280' }}>
        <a href={REPO_URL}>BoTime/NoteScanner</a> on GitHub
      </footer>
    </main>
  );
}
```

- [ ] **Step 9: Write the jsdom component test for the failure path and the drop path**

These are the behaviours a headless-browser spec cannot reach cheaply: a `SegmenterFailure` needs a segmenter that fails, and a drop needs a synthetic `DataTransfer`. Create `site/TryItPage.test.tsx`:

```tsx
// @vitest-environment jsdom
//
// jsdom because TryItPage is a React component. `createSegmenter` is mocked so
// nothing downloads a model; everything else in ../src/segmenter — including
// the real SegmenterFailure and DEFAULT_SEGMENTER_OPTIONS — stays real, because
// the point of these tests is how the page reacts to a REAL SegmenterFailure.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';

const segment = vi.fn();

vi.mock('../src/segmenter', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/segmenter')>();
  return { ...actual, createSegmenter: () => ({ segment, dispose: vi.fn() }) };
});

import { TryItPage } from './TryItPage';
import { DEFAULT_SEGMENTER_OPTIONS, SegmenterFailure } from '../src/segmenter';

beforeEach(() => {
  segment.mockReset();
  vi.stubGlobal('navigator', { ...navigator, gpu: {} });
  vi.stubGlobal('fetch', vi.fn(async () => ({ blob: async () => new Blob() })));
  vi.stubGlobal('createImageBitmap', vi.fn(async () => ({}) as ImageBitmap));
  // Defined ON the real URL rather than replacing the global: `new URL(...)`
  // is used elsewhere in the tree, and a plain-object stand-in for URL is not
  // constructible. `vi.unstubAllGlobals` does not undo this, which is why both
  // are `configurable` and why the value is deterministic.
  for (const [name, value] of [
    ['createObjectURL', () => 'blob:stub'],
    ['revokeObjectURL', () => {}],
  ] as const) {
    Object.defineProperty(URL, name, { configurable: true, value });
  }
  // jsdom implements none of these three; without them the sample-image effect
  // rejects unhandled and the viewer never sees a size.
  Object.defineProperty(HTMLImageElement.prototype, 'decode', {
    configurable: true,
    value: () => Promise.resolve(),
  });
  Object.defineProperty(HTMLImageElement.prototype, 'naturalWidth', {
    configurable: true,
    get: () => 1024,
  });
  Object.defineProperty(HTMLImageElement.prototype, 'naturalHeight', {
    configurable: true,
    get: () => 649,
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

async function renderWithSample() {
  render(<TryItPage />);
  await screen.findByTestId('site-image');
}

describe('TryItPage failure handling (AC7)', () => {
  it('names the phase a SegmenterFailure died in', async () => {
    segment.mockRejectedValueOnce(new SegmenterFailure('decode', 'adapter lost'));
    await renderWithSample();

    fireEvent.click(screen.getByTestId('run-segmentation'));

    const error = await screen.findByTestId('run-error');
    expect(error.textContent).toBe('Failed during decode: adapter lost');
  });

  it('renders a non-Error throw under the phase "unknown"', async () => {
    segment.mockRejectedValueOnce('something fell over');
    await renderWithSample();

    fireEvent.click(screen.getByTestId('run-segmentation'));

    const error = await screen.findByTestId('run-error');
    expect(error.textContent).toBe('Failed during unknown: something fell over');
  });

  it('stays usable after a failure: another image can be chosen and run again', async () => {
    segment.mockRejectedValueOnce(new SegmenterFailure('encode', 'nope'));
    segment.mockResolvedValueOnce({
      segments: [{ id: 'a' }, { id: 'b' }],
      timings: { totalMs: 7067 },
      counts: {},
    });
    await renderWithSample();

    fireEvent.click(screen.getByTestId('run-segmentation'));
    await screen.findByTestId('run-error');

    // Choosing the other sample clears the failure...
    fireEvent.click(screen.getByTestId('site-sample-sticky-notes'));
    await waitFor(() => expect(screen.queryByTestId('run-error')).toBeNull());

    // ...and a second run completes and reports its stats.
    fireEvent.click(screen.getByTestId('run-segmentation'));
    const stats = await screen.findByTestId('run-stats');
    expect(stats.textContent).toBe('2 segments · 7.1 s');
  });
});

describe('TryItPage image sources (AC3)', () => {
  it('accepts a dropped file, not only a chosen one', async () => {
    await renderWithSample();
    const file = new File(['x'], 'note.png', { type: 'image/png' });

    fireEvent.drop(screen.getByTestId('site-controls'), { dataTransfer: { files: [file] } });

    await waitFor(() =>
      expect(screen.getByTestId('site-image').getAttribute('data-src')).toBe('blob:stub'),
    );
    // A dropped file is not a sample: neither chip stays pressed.
    for (const id of ['cafe-table', 'sticky-notes']) {
      expect(screen.getByTestId(`site-sample-${id}`).getAttribute('aria-pressed')).toBe('false');
    }
  });
});

describe('TryItPage pins the package defaults (AC10)', () => {
  it('passes DEFAULT_SEGMENTER_OPTIONS through unmodified', async () => {
    segment.mockResolvedValueOnce({ segments: [], timings: { totalMs: 1 }, counts: {} });
    await renderWithSample();

    fireEvent.click(screen.getByTestId('run-segmentation'));

    await waitFor(() => expect(segment).toHaveBeenCalledTimes(1));
    // Identity, not deep equality: the page must not build its own options
    // object, because an object it builds is an object it can drift.
    expect(segment.mock.calls[0][1]).toBe(DEFAULT_SEGMENTER_OPTIONS);
  });
});
```

- [ ] **Step 10: Run the vitest suite, then prove the AC10 assertion can fail**

```bash
npm run test
```

Expected: PASS across `src/`, `playground/` and the three new `site/` files.

Then inject a failure, to establish that the AC10 assertion is load-bearing rather than incidentally true. Temporarily change the `segment(...)` call in `TryItPage.tsx` to pass `{ ...DEFAULT_SEGMENTER_OPTIONS }`, then:

```bash
npx vitest run site/TryItPage.test.tsx
```

Expected: FAIL, on the identity check only. Revert the change, re-run, confirm PASS. Record both outcomes in the task's completion notes.

- [ ] **Step 11: Add the typecheck project and the npm scripts**

Create `tsconfig.site.json`:

```json
{
  "extends": "./tsconfig.json",
  "include": ["site/**/*.ts", "site/**/*.tsx"],
  "exclude": ["node_modules", "dist"]
}
```

In `package.json`, extend `typecheck` and add four scripts, leaving every existing script untouched:

```json
"typecheck": "tsc --noEmit && tsc --noEmit -p tsconfig.playground.json && tsc --noEmit -p tsconfig.site.json && tsc --noEmit -p tsconfig.tests.json",
"site": "vite --config site/vite.config.ts",
"build:site": "vite build --config site/vite.config.ts",
"preview:site": "npm run build:site && vite preview --config site/vite.config.ts",
"test:site:gpu": "REQUIRE_WEBGPU=1 playwright test site-webgpu --project=chromium --headed",
```

Then:

```bash
npm run typecheck
```

Expected: PASS, four projects.

- [ ] **Step 12: Production-build the site and check the artifact, not the exit code**

Nothing above has run the bundler. This is where `worker.format` and the base path get their only proof before CI.

```bash
npm run build:site
```

Expected: a successful build writing `site/dist/`. Then:

```bash
grep -o '/NoteScanner/assets/[^"]*' site/dist/index.html
```

Expected: at least one match, and every script/link href the document emits begins with `/NoteScanner/`. A relative `./assets/...` or a bare `/assets/...` means `base` did not apply and the deployed page would 404 its own bundle.

If the build fails on the worker with a message about `iife` and dynamic import, `worker: { format: 'es' }` is missing or misspelled — that is the expected failure mode, and the reason Step 6 sets it.

- [ ] **Step 13: Open the built site by hand, once**

```bash
npm run preview:site
```

Open `http://localhost:5181/NoteScanner/` in a real browser. Confirm the title and tagline render, a sample photo is on screen, both chips are present, and the console is clean. Stop the server. A 60-second check that costs far less than discovering a blank page in Task 3.

- [ ] **Step 14: Commit**

```bash
git add site tsconfig.site.json package.json vitest.config.ts
git commit -m "feat(site): a public Try-it page built from its own Vite entry"
```

---

## Task 3: Browser verification and the Pages deployment

Playwright coverage for everything a GPU-less engine can observe, a WebGPU-gated spec for the one criterion that needs a real adapter, and the workflow that publishes the site.

**Files:**
- Create: `tests/browser/site-url.ts`, `tests/browser/site.spec.ts`, `tests/browser/site-webgpu.spec.ts`
- Create: `.github/workflows/pages.yml`
- Modify: `playwright.config.ts`, `README.md`
- Unchanged, deliberately: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the DOM contract and the `preview:site` / `test:site:gpu` scripts from Task 2.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Give the site's preview URL one home**

Create `tests/browser/site-url.ts`:

```ts
/**
 * The one place the built site's preview URL is written.
 *
 * The port must match `preview.port` in `site/vite.config.ts`. If the two ever
 * disagree, Playwright's `webServer` wait on this exact URL times out at start
 * of run, rather than producing a confusing per-test failure later.
 */
export const SITE_URL = 'http://localhost:5181/NoteScanner/';
```

- [ ] **Step 2: Serve the built site to Playwright**

In `playwright.config.ts`, import `SITE_URL` from `./tests/browser/site-url` and turn `webServer` into an array — keeping the existing playground entry exactly as it is, and appending a second:

```ts
  webServer: [
    {
      command: 'npm run playground',
      url: 'http://localhost:5180/',
      // Reuse a dev server the developer already has up; in CI always start one.
      reuseExistingServer: !process.env.CI,
      // Vite pre-bundles @huggingface/transformers at startup (see
      // playground/vite.config.ts), which is slow the first time.
      timeout: 180_000,
    },
    {
      // The PRODUCTION build, served under the real base path — that is the
      // artifact GitHub Pages serves, and base-path asset breakage is exactly
      // the class of bug a dev server hides.
      command: 'npm run preview:site',
      url: SITE_URL,
      // Never reused, even locally: a reused preview server is serving some
      // earlier build, and checking the current one is this spec's whole job.
      // A port clash therefore fails loudly here, which is the right outcome.
      reuseExistingServer: false,
      // Builds @huggingface/transformers and the worker bundle from cold.
      timeout: 300_000,
    },
  ],
```

Leave `use.baseURL` pointing at the playground; the two site specs navigate to `SITE_URL` absolutely.

- [ ] **Step 3: Write the browser spec for AC1 through AC5**

Create `tests/browser/site.spec.ts`:

```ts
import { expect, test, type Page } from '@playwright/test';
import { SITE_URL } from './site-url';

/** Everything the page must never expose, per AC5. */
const BANNED = 'select, input[type=checkbox], input[type=radio], input[type=number], table';

/** `isWebGPUAvailable()` is `'gpu' in navigator`, and `navigator.gpu` lives on
 *  the prototype — assigning undefined leaves the `in` check true, the page
 *  would render the Run button, and the AC4 test would prove nothing. */
async function withoutWebGPU(page: Page) {
  await page.addInitScript(() => {
    delete (window.Navigator.prototype as unknown as Record<string, unknown>).gpu;
  });
}
```

Then, in the same file, the AC1 test:

```ts
test('renders the demo, its claim and its footer, with no failed requests (AC1)', async ({
  page,
}) => {
  const consoleErrors: string[] = [];
  const failed: string[] = [];
  const foreign: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('pageerror', (err) => consoleErrors.push(String(err)));
  page.on('requestfailed', (req) => failed.push(req.url() + ' ' + req.failure()?.errorText));
  page.on('response', (res) => {
    if (res.status() >= 400) failed.push(res.url() + ' status ' + res.status());
    if (!res.url().startsWith(SITE_URL) && !res.url().startsWith('data:')) foreign.push(res.url());
  });

  await page.goto(SITE_URL);
  await expect(page.getByTestId('site-image')).toBeVisible();

  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Try note-scanner');
  const tagline = page.getByTestId('site-tagline');
  await expect(tagline).toContainText('WebGPU in your own browser');
  await expect(tagline).toContainText('never leaves this page');
  await expect(page.getByTestId('site-footer').getByRole('link')).toHaveAttribute(
    'href',
    'https://github.com/BoTime/NoteScanner',
  );

  // A demo, not a landing page: no install line, no usage snippet.
  const body = (await page.locator('body').innerText()).toLowerCase();
  expect(body).not.toContain('npm install');
  expect(body).not.toContain('yarn add');
  expect(body).not.toContain('import {');

  expect(failed, 'failed requests: ' + failed.join(', ')).toEqual([]);
  expect(consoleErrors, 'console errors: ' + consoleErrors.join(', ')).toEqual([]);
  // Loading the page must not touch a CDN. The model is fetched only on Run.
  expect(foreign, 'off-origin requests at load: ' + foreign.join(', ')).toEqual([]);
});
```

The AC2 and AC3 tests:

```ts
test('shows a sample immediately and swaps to the other one (AC2)', async ({ page }) => {
  await page.goto(SITE_URL);

  const shown = page.getByTestId('site-image');
  await expect(shown).toHaveAttribute('data-src', /cafe-table/);
  // Proves the image DECODED, not merely that a url string was set.
  expect(Number(await shown.getAttribute('data-width'))).toBeGreaterThan(0);
  await expect(page.getByTestId('site-sample-cafe-table')).toHaveAttribute('aria-pressed', 'true');

  await page.getByTestId('site-sample-sticky-notes').click();

  await expect(shown).toHaveAttribute('data-src', /sticky-notes/);
  expect(Number(await shown.getAttribute('data-width'))).toBeGreaterThan(0);
  await expect(page.getByTestId('site-sample-sticky-notes')).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.getByTestId('site-sample-cafe-table')).toHaveAttribute('aria-pressed', 'false');
});

test('takes the visitor own file (AC3)', async ({ page }) => {
  await page.goto(SITE_URL);
  const input = page.getByTestId('site-file-input');
  await expect(input).toHaveAttribute('accept', 'image/*');

  await input.setInputFiles('samples/sticky-notes.jpg');

  const shown = page.getByTestId('site-image');
  await expect(shown).toHaveAttribute('data-src', /^blob:/);
  expect(Number(await shown.getAttribute('data-width'))).toBeGreaterThan(0);
});
```

The AC4 pair — the no-adapter assertion and the positive control that keeps it from being vacuous:

```ts
test('shows the notice and no Run button without an adapter, and starts no worker (AC4)', async ({
  page,
}) => {
  const modelish: string[] = [];
  page.on('request', (req) => {
    if (/segmenter\.worker|huggingface|\.onnx|transformers/i.test(req.url())) {
      modelish.push(req.url());
    }
  });
  await withoutWebGPU(page);
  await page.goto(SITE_URL);

  // The init script has to have worked, or this test proves nothing.
  expect(await page.evaluate(() => 'gpu' in navigator)).toBe(false);

  await expect(page.getByTestId('webgpu-required')).toBeVisible();
  await expect(page.getByTestId('webgpu-required')).toContainText('no CPU fallback');
  await expect(page.getByTestId('run-segmentation')).toHaveCount(0);
  expect(modelish, 'worker or model traffic: ' + modelish.join(', ')).toEqual([]);
});

test('offers Run when an adapter IS present', async ({ page }) => {
  await page.goto(SITE_URL);
  // The positive control for the test above. Without it, that test would pass
  // in an engine where the Run button fails to render for unrelated reasons.
  test.skip(
    !(await page.evaluate(() => 'gpu' in navigator)),
    'this engine exposes no navigator.gpu at all',
  );
  await expect(page.getByTestId('run-segmentation')).toBeVisible();
  await expect(page.getByTestId('webgpu-required')).toHaveCount(0);
});
```

And the AC5 guard, the assertion that makes the separation from the dev playground enforceable rather than a convention:

```ts
test('exposes no option controls, no tabs and no timing table (AC5)', async ({ page }) => {
  await page.goto(SITE_URL);
  await expect(page.getByTestId('site-image')).toBeVisible();

  // Document-wide: SegmentViewer renders buttons, but no select, input or table.
  await expect(page.locator(BANNED)).toHaveCount(0);

  const inputs = page.locator('input');
  await expect(inputs).toHaveCount(1);
  await expect(inputs.first()).toHaveAttribute('type', 'file');

  // The page's own controls are exactly two sample chips and one Run button.
  await expect(page.getByTestId('site-controls').locator('button')).toHaveCount(3);
  await expect(page.locator('nav')).toHaveCount(0);
});
```

- [ ] **Step 4: Run the spec in all three engines**

```bash
npm run test:browser -- site.spec.ts
```

Expected: PASS in chromium, webkit and firefox. Record in the completion notes which engines skipped "offers Run when an adapter IS present" — webkit and firefox expose no `navigator.gpu` and are expected to skip; a skip in chromium too simply means headless chromium has no adapter here.

If the AC5 input-count assertion fails at 2 or more, do **not** relax the assertion — find what rendered the extra control and remove it.

- [ ] **Step 5: Write the real-adapter end-to-end spec**

AC6 is the criterion CI cannot reach. It gets a real run rather than a hand-wave: this spec skips cleanly when no adapter exists, and **fails rather than skips** when `REQUIRE_WEBGPU=1`, which is what `npm run test:site:gpu` sets. A skipped run therefore cannot be reported as a passed one.

Create `tests/browser/site-webgpu.spec.ts`:

```ts
import { expect, test } from '@playwright/test';
import { SITE_URL } from './site-url';

// A cold run downloads tens of megabytes of ONNX weights and then does roughly
// 7 s of GPU work; the default 30 s timeout is nowhere near it.
test.setTimeout(10 * 60 * 1000);

test.describe('a real segmentation on a real adapter (AC6)', () => {
  test.skip(({ browserName }) => browserName !== 'chromium', 'needs a WebGPU adapter');

  test('runs end to end, names the download, and selects a segment', async ({ page }) => {
    const progressLines: string[] = [];
    await page.goto(SITE_URL);

    const adapter = await page.evaluate(async () => {
      const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
      if (!gpu) return false;
      return Boolean(await gpu.requestAdapter());
    });
    if (!adapter) {
      // npm run test:site:gpu sets this. A silent skip there would let a run
      // with no adapter report the same green as a run that segmented an image.
      expect(
        process.env.REQUIRE_WEBGPU,
        'REQUIRE_WEBGPU is set but this browser yielded no WebGPU adapter — run it headed on a machine with a GPU',
      ).toBeFalsy();
      test.skip(true, 'no WebGPU adapter in this browser');
    }

    const progress = page.getByTestId('run-progress');
    // Sample every distinct progress line the run passes through, so the
    // assertions below are about what a visitor actually saw.
    const poll = setInterval(() => {
      void progress
        .textContent({ timeout: 500 })
        .then((text) => {
          if (text && progressLines.at(-1) !== text) progressLines.push(text);
        })
        .catch(() => {});
    }, 200);

    await page.getByTestId('run-segmentation').click();
    // Wait on the stats line, which renders real text — never on a box that is
    // 0px tall until a result arrives.
    await expect(page.getByTestId('run-stats')).toBeVisible({ timeout: 9 * 60 * 1000 });
    clearInterval(poll);

    // `?? ''` rather than a non-null assertion: textContent is `string | null`
    // and `tsc --noEmit -p tsconfig.tests.json` is strict.
    const stats = (await page.getByTestId('run-stats').textContent()) ?? '';
    expect(stats).toMatch(/^\d+ segments · \d+\.\d s$/);
    expect(Number(stats.split(' ')[0]), 'the run returned no segments').toBeGreaterThan(0);
    expect(await page.getByTestId('run-error').count()).toBe(0);

    // The download is named, and distinctly from the per-batch phases.
    const seen = progressLines.join(' | ');
    expect(
      progressLines.some((line) => /Downloading the model \(first run only\)/.test(line)),
      'progress lines seen: ' + seen,
    ).toBe(true);
    expect(
      progressLines.some((line) =>
        /^(encode|decode|filter|nms|resample|mask-encode) \d+\/\d+$/.test(line),
      ),
      'progress lines seen: ' + seen,
    ).toBe(true);

    // Click-to-select works on real output: click the middle of the viewer.
    const canvas = page.locator('canvas').first();
    const box = await canvas.boundingBox();
    // Thrown, not asserted: a null box must stop the test here, and `expect`
    // alone would leave `box` typed `null` for the click below.
    if (!box) throw new Error('the viewer canvas has no layout box');
    await canvas.click({ position: { x: box.width / 2, y: box.height / 2 } });

    // Printed, not just asserted: this is the AC6 evidence the run records.
    console.log('progress lines:', seen);
    console.log('stat line:', stats);
  });
});
```

- [ ] **Step 6: Run it for real, headed, on this machine**

```bash
npm run test:site:gpu
```

Expected: PASS with a real segmentation, not a skip — `REQUIRE_WEBGPU=1` turns a missing adapter into a failure. Copy the printed progress lines and stat line into the task's completion notes; that output is the evidence for AC6.

If it fails because headed chromium here yields no adapter, say so explicitly in the completion notes and escalate — do not silently downgrade AC6 to an unverified hand run. `scripts/sweep-decode.mjs` already drives real-GPU chromium on this machine with `chromium.launch({ headless: false })`, so an adapter is expected to be available.

- [ ] **Step 7: Confirm CI's full browser lane still passes**

```bash
npm run test:browser
```

Expected: PASS. The GPU spec skips (no adapter headless, and `REQUIRE_WEBGPU` unset); every other spec runs. Confirm the playground specs are unaffected by the second `webServer` entry.

- [ ] **Step 8: Write the Pages workflow**

Create `.github/workflows/pages.yml`:

```yaml
name: Deploy public page

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

# One deployment at a time, and never cancel one mid-flight: a cancelled
# deploy-pages can leave the environment with no live deployment.
concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: npm
      - run: npm install
      # enablement: true turns Pages on for the repository on the first run,
      # so no manual visit to the repository settings is needed.
      - uses: actions/configure-pages@v5
        with:
          enablement: true
      - run: npm run build:site
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site/dist
      - id: deployment
        uses: actions/deploy-pages@v4
```

Then confirm `ci.yml` is untouched:

```bash
git diff --name-only main -- .github/workflows/
```

Expected: `.github/workflows/pages.yml` and nothing else.

- [ ] **Step 9: Add the new scripts and the public URL to the README**

In `README.md`, replace the Development code block with:

```bash
npm install
npm run playground   # Vite dev app (all four tabs)
npm run site         # the public Try-it page, the app deployed to GitHub Pages
npm run test
npm run typecheck
npm run smoke        # build + artifact check
```

and add one sentence immediately below that block:

```markdown
The public page is deployed from `site/` to
<https://botime.github.io/NoteScanner/> by `.github/workflows/pages.yml` on
every push to `main`.
```

- [ ] **Step 10: Run every gate**

```bash
npm run test
```

```bash
npm run typecheck
```

```bash
npm run test:browser
```

```bash
npm run build:site
```

Expected: all PASS.

- [ ] **Step 11: Commit**

```bash
git add tests/browser playwright.config.ts .github/workflows/pages.yml README.md
git commit -m "test(site): browser coverage for the public page, and a Pages deploy"
```

---

## Acceptance criteria coverage

Every criterion and the exact command whose output is its evidence. Nothing is parked on an unverifiable hand run.

| AC | Evidence | Task |
|---|---|---|
| AC1 | `npm run test:browser -- site.spec.ts`, test "renders the demo, its claim and its footer" | 3 |
| AC2 | same spec, "shows a sample immediately and swaps to the other one" | 3 |
| AC3 | same spec, "takes the visitor own file"; drop branch in `site/TryItPage.test.tsx` | 2, 3 |
| AC4 | same spec, "shows the notice and no Run button without an adapter", with its positive control alongside | 3 |
| AC5 | same spec, "exposes no option controls, no tabs and no timing table" | 3 |
| AC6 | `npm run test:site:gpu` — a real headed run; `REQUIRE_WEBGPU=1` makes a missing adapter a failure, not a skip | 3 |
| AC7 | `npx vitest run site/TryItPage.test.tsx`, the three failure-handling tests | 2 |
| AC8 | `npx vitest run site/imports.test.ts`, both directions, guarded by a non-empty file-list check | 2 |
| AC9 | `file samples/sticky-notes.jpg`, `wc -c`, and the two provenance entries in `samples/README.md` | 1 |
| AC10 | the identity assertion in `site/TryItPage.test.tsx`, proven failable in Task 2 Step 10 | 2 |
| AC11 | `npm run build:site` plus the `/NoteScanner/` prefix check on the emitted `index.html` | 2 |
| AC12 | `.github/workflows/pages.yml` as written; `git diff --name-only main -- .github/workflows/` lists only the new file | 3 |
| AC13 | `npm run test`, `npm run typecheck`, `npm run test:browser` | 3 |
