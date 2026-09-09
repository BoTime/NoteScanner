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

test('exposes no option controls, no tabs and no timing table (AC5)', async ({ page }) => {
  await page.goto(SITE_URL);
  await expect(page.getByTestId('site-image')).toBeVisible();

  // Document-wide: SegmentViewer renders buttons, but no select, input or table.
  await expect(page.locator(BANNED)).toHaveCount(0);

  const inputs = page.locator('input');
  await expect(inputs).toHaveCount(1);
  await expect(inputs.first()).toHaveAttribute('type', 'file');

  // The page's own controls are exactly the two sample chips, plus the Run
  // button — which AC4 replaces with the notice in an engine that exposes no
  // adapter (webkit has none). Deriving the count rather than hardcoding 3
  // keeps this strict: an extra control still fails it in every engine.
  const hasGpu = await page.evaluate(() => 'gpu' in navigator);
  await expect(page.getByTestId('site-controls').locator('button')).toHaveCount(hasGpu ? 3 : 2);
  await expect(page.getByTestId(hasGpu ? 'run-segmentation' : 'webgpu-required')).toBeVisible();
  await expect(page.locator('nav')).toHaveCount(0);
});
