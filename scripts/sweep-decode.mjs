/**
 * The decode sweep: a real-GPU, config-driven walk of the decode grid.
 *
 * Uses the Playwright LIBRARY, not the test runner. The test runner's retries,
 * timeouts and parallelism fight a serial GPU benchmark — and a retried GPU row
 * is a different measurement, not the same one again.
 *
 * No sweep logic lives here. Grid expansion, ranking and markdown all come from
 * `playground/compare.ts` THROUGH THE PAGE, via `window.__decodeSweep`, so
 * there is exactly one implementation and it is the unit-tested one.
 *
 *   npm run sweep:decode
 *   npm run sweep:decode -- --full --reps 3
 *   npm run sweep:decode -- --config my-grid.json
 *   npm run sweep:decode -- --headless --allow-software   # wiring smoke test only
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';
import { createServer } from 'vite';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

/** Generous: an fp32 batch-8 row at 32 points per side is minutes of GPU work. */
const ROW_TIMEOUT_MS = 15 * 60 * 1000;

/**
 * Known software rasterizers. A software run's timings are meaningless next to
 * a hardware run's, so the sweep refuses to produce them by accident.
 */
const SOFTWARE_ADAPTER = /swiftshader|lavapipe|llvmpipe|warp|basic render|software/i;

function parseArgs(argv) {
  const args = { full: false, headless: false, allowSoftware: false, reps: undefined, config: undefined };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--full') args.full = true;
    else if (arg === '--headless') args.headless = true;
    else if (arg === '--allow-software') args.allowSoftware = true;
    else if (arg === '--reps') args.reps = Number(argv[(i += 1)]);
    else if (arg === '--config') args.config = argv[(i += 1)];
    else throw new Error(`unknown argument: ${arg}`);
  }
  return args;
}

function die(message) {
  console.error(`sweep-decode: ${message}`);
  process.exit(1);
}

/**
 * Mirrors `BATCH_SIZE_CHOICES` / `POINTS_PER_SIDE_CHOICES` from
 * `playground/CompareView.tsx` — the exact values the Compare tab's
 * `<select data-testid="batch-size">` / `<select data-testid="points-per-side">`
 * controls render as `<option>`s. `resolveConfig`'s own validation only checks
 * that config values are positive integers, so a `--config` file asking for a
 * batch size or points-per-side the tab doesn't offer (e.g. `pointsPerSide: [8]`)
 * passes that check and would otherwise only fail inside `page.selectOption`
 * deep into the row loop — after however many earlier rows already burned real
 * GPU minutes. Keep these two lists in sync with CompareView.tsx BY HAND: there
 * is no import path from this Node script into a TSX file's local constants
 * (and adding one would be exactly the "second copy of sweep logic" this file's
 * header comment says not to create).
 */
const TAB_BATCH_SIZE_CHOICES = [8, 16, 32, 64];
const TAB_POINTS_PER_SIDE_CHOICES = [16, 32];

function assertConfigMatchesTabControls(config) {
  for (const value of config.batchSizes) {
    if (!TAB_BATCH_SIZE_CHOICES.includes(value)) {
      die(
        `config batchSizes contains ${value}, which the Compare tab's control doesn't offer ` +
          `(only ${TAB_BATCH_SIZE_CHOICES.join(', ')}) — page.selectOption would fail on it deep into the sweep`,
      );
    }
  }
  for (const value of config.pointsPerSide) {
    if (!TAB_POINTS_PER_SIDE_CHOICES.includes(value)) {
      die(
        `config pointsPerSide contains ${value}, which the Compare tab's control doesn't offer ` +
          `(only ${TAB_POINTS_PER_SIDE_CHOICES.join(', ')}) — page.selectOption would fail on it deep into the sweep`,
      );
    }
  }
}

async function probeAdapter(page) {
  return page.evaluate(async () => {
    const gpu = navigator.gpu;
    if (!gpu) return null;
    const adapter = await gpu.requestAdapter();
    if (!adapter) return null;
    // `adapter.info` is the current API; `requestAdapterInfo()` is the older
    // one. Read whichever this Chromium ships.
    const info =
      adapter.info ??
      (typeof adapter.requestAdapterInfo === 'function' ? await adapter.requestAdapterInfo() : {});
    return {
      vendor: info.vendor ?? '',
      architecture: info.architecture ?? '',
      device: info.device ?? '',
      description: info.description ?? '',
    };
  });
}

async function setCheckbox(page, testId, value) {
  await page.locator(`[data-testid="${testId}"]`).setChecked(value);
}

async function runOneRow(page, row) {
  const o = row.options;
  await page.selectOption('[data-testid="dtype"]', o.dtype);
  await page.selectOption('[data-testid="batch-size"]', String(o.batchSize));
  await page.selectOption('[data-testid="points-per-side"]', String(o.pointsPerSide));
  await setCheckbox(page, 'overlap-decode-filter', o.overlapDecodeFilter);
  await setCheckbox(page, 'gpu-resident-embeddings', o.gpuResidentEmbeddings);
  await setCheckbox(page, 'keep-raw-masks', o.keepRawMasks);
  await setCheckbox(page, 'low-res-filter-nms', o.lowResFilterNms);

  const before = Number(await page.getAttribute('[data-testid="run-json"]', 'data-run-count'));
  await page.click('[data-testid="run-row"]');
  // The page ticks this counter LAST, after the blob is written, so seeing it
  // grow means this run's numbers — not the previous run's — are readable.
  await page.waitForFunction(
    (n) =>
      Number(
        document.querySelector('[data-testid="run-json"]')?.getAttribute('data-run-count') ?? -1,
      ) > n,
    before,
    { timeout: ROW_TIMEOUT_MS },
  );

  const blob = await page.textContent('[data-testid="run-json"]');
  // The page drives one configuration at a time and has no grid, so it emits an
  // empty rowId; the id belongs to the row WE asked for.
  return { ...JSON.parse(blob), rowId: row.id };
}

function outputStem(dir, date) {
  let stem = `${date}-decode-sweep`;
  let n = 1;
  // A second run on the same day gets -2, -3 rather than silently overwriting.
  while (existsSync(path.join(dir, `${stem}.json`)) || existsSync(path.join(dir, `${stem}.md`))) {
    n += 1;
    stem = `${date}-decode-sweep-${n}`;
  }
  return stem;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  const server = await createServer({
    configFile: path.join(root, 'playground', 'vite.config.ts'),
    // Port 0: never fight a dev server the developer already has running.
    server: { port: 0, strictPort: false },
  });
  await server.listen();
  const url = server.resolvedUrls?.local?.[0];
  if (!url) {
    await server.close();
    die('vite started but reported no local URL');
  }

  // HEADED by default. Headless Chromium frequently resolves no real WebGPU
  // adapter and silently falls back to software, and software timings would
  // make the whole sweep meaningless. --headless is for smoke-testing wiring.
  const browser = await chromium.launch({ headless: args.headless });
  let exitCode = 0;

  try {
    const page = await browser.newPage();
    page.on('pageerror', (error) => console.error(`page error: ${error.message}`));
    await page.goto(url, { waitUntil: 'load' });
    await page.click('[data-testid="view-compare"]');
    // `state: 'attached'` on purpose: this `<pre>` is empty (0px height) until
    // the first row completes, so the default 'visible' wait state would time
    // out here every time. Only the DOM's existence matters at this point —
    // the actual run-completion sync is the `data-run-count` waitForFunction
    // inside runOneRow, below.
    await page.waitForSelector('[data-testid="run-json"]', { state: 'attached' });

    // ---- the adapter gate, BEFORE the first measurement.
    const probed = await probeAdapter(page);
    if (!probed) die('no WebGPU adapter in this browser — refusing to measure');
    const software = SOFTWARE_ADAPTER.test(Object.values(probed).join(' '));
    const adapter = { ...probed, software };
    if (software && !args.allowSoftware) {
      die(
        `adapter looks like a software rasterizer (${JSON.stringify(adapter)}); ` +
          `pass --allow-software to measure it anyway`,
      );
    }
    // Recorded either way, so a software run is self-identifying rather than
    // quietly comparable to a real one.
    console.log(`adapter: ${JSON.stringify(adapter)}`);

    // ---- the grid, expanded by the page's own tested code.
    const overrides = args.config
      ? JSON.parse(readFileSync(path.resolve(process.cwd(), args.config), 'utf8'))
      : {};
    const config = await page.evaluate(
      ({ overrides, flags }) => window.__decodeSweep.resolveConfig(overrides, flags),
      { overrides, flags: { full: args.full, reps: args.reps } },
    );
    // Fail fast on a config the tab's own controls can't drive, BEFORE
    // expandGrid and the row loop below spend any GPU time. This is as early
    // as the check can run: `resolveConfig` only exists inside the page (see
    // the file header — its logic is deliberately not duplicated here), so
    // the fully-resolved config isn't known until the line above.
    assertConfigMatchesTabControls(config);
    const rows = await page.evaluate((config) => {
      const rows = window.__decodeSweep.expandGrid(config);
      // The warm-up runs first and is discarded: the first run in a page pays
      // cold shader compilation and a cold HTTP cache.
      return [window.__decodeSweep.warmUpRow(rows), ...rows];
    }, config);

    // ---- walk every row in ONE page, so the ONNX weights stay HTTP-cached.
    const records = [];
    for (const row of rows) {
      process.stdout.write(`${row.id} … `);
      let record;
      try {
        record = await runOneRow(page, row);
      } catch (error) {
        // A `SegmenterFailure` inside the worker never lands here — CompareView
        // already catches that itself and resolves with a `status: 'failed'`
        // record, which is the branch below. What DOES land here is the
        // harness breaking: `waitForFunction` hitting ROW_TIMEOUT_MS (a real
        // hang, or a page/worker crash the promise never resolves from), or
        // `selectOption`/`setChecked` being handed a value the tab's control
        // doesn't render.
        const message = String(error?.message ?? error);
        if (row.warmUp) {
          // The warm-up exists to pay cold shader compilation and a cold HTTP
          // cache ONCE so no measured row absorbs it. If even the warm-up
          // can't complete, the harness itself is broken (bad selector, wedged
          // page, crashed worker) — not one unsupported config among sixteen.
          // Continuing would run the whole grid against a page that already
          // proved it can't finish a single row, producing sixteen more
          // failures instead of one clear one. So abort here rather than push
          // a record: there is no measured-rows array entry for a warm-up row
          // even on success, and there shouldn't be one on failure either.
          console.log('FAILED (harness error)');
          die(`warm-up row failed, aborting sweep: ${message}`);
        }
        console.log(`FAILED (harness error): ${message}`);
        records.push({
          rowId: row.id,
          options: row.options,
          status: 'failed',
          phase: 'unknown',
          message,
        });
        continue;
      }
      if (row.warmUp) {
        console.log('discarded (warm-up)');
        continue;
      }
      records.push(record);
      // One unsupported path must not cost the other fifteen their row.
      console.log(
        record.status === 'ok'
          ? `${record.budgetMs.toFixed(0)} ms budget, ${record.counts.returned} masks`
          : `FAILED in ${record.phase}: ${record.message}`,
      );
    }

    const generatedAt = new Date().toISOString();
    const markdown = await page.evaluate(
      ({ records, meta }) => window.__decodeSweep.toMarkdown(records, meta),
      { records, meta: { config, adapter, generatedAt } },
    );

    const dir = path.join(root, 'docs', 'measurements');
    mkdirSync(dir, { recursive: true });
    const stem = outputStem(dir, generatedAt.slice(0, 10));
    // The JSON embeds the resolved config and the adapter, so the report is
    // reproducible from its own artifact.
    writeFileSync(
      path.join(dir, `${stem}.json`),
      `${JSON.stringify({ generatedAt, adapter, config, rows: records }, null, 2)}\n`,
    );
    writeFileSync(path.join(dir, `${stem}.md`), markdown);
    console.log(`\nwrote docs/measurements/${stem}.{md,json}`);

    // Non-zero only if EVERY row failed: a partial sweep is still a result.
    if (!records.some((record) => record.status === 'ok')) {
      console.error('every row failed');
      exitCode = 1;
    }
  } finally {
    await browser.close();
    await server.close();
  }

  process.exit(exitCode);
}

await main();
