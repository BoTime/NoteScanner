import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * AC2, in a real engine. The Boundary tab computes both pipelines
 * synchronously on mount, so there is nothing to poll — but every wait below
 * targets an element that renders visible TEXT, never an empty box.
 */
async function openBoundary(page: Page) {
  await page.goto('/');
  await page.click('[data-testid="view-boundary"]');
  await expect(page.getByTestId('boundary-summary')).toBeVisible();
}

async function statuses(row: Locator): Promise<[string | null, string | null]> {
  return [await row.getAttribute('data-baseline-status'), await row.getAttribute('data-new-status')];
}

test('every mask both paths keep renders a zero-pixel difference', async ({ page, baseURL }) => {
  const foreign: string[] = [];
  page.on('request', (request) => {
    if (!request.url().startsWith(baseURL!)) foreign.push(request.url());
  });

  await openBoundary(page);

  const rows = page.locator('[data-testid^="boundary-window-"]');
  const count = await rows.count();
  // Guards the loop below against passing because it iterated nothing.
  expect(count).toBeGreaterThan(0);

  let keptByBoth = 0;
  for (let i = 0; i < count; i += 1) {
    const row = rows.nth(i);
    const id = await row.getAttribute('data-testid');
    const [baseline, next] = await statuses(row);
    if (baseline !== 'kept' || next !== 'kept') continue;
    keptByBoth += 1;
    expect(await row.getAttribute('data-diff'), `${id} differs between the two paths`).toBe('0');
  }
  expect(keptByBoth).toBeGreaterThan(0);

  // No model, no CDN, no anything: everything this tab needs is its own code.
  expect(foreign, `the Boundary tab fetched ${foreign.join(', ')}`).toEqual([]);
});

test('keeps the fine-toothed comb on both paths', async ({ page }) => {
  await openBoundary(page);
  const row = page.getByTestId('boundary-window-comb-0');
  expect(await statuses(row)).toEqual(['kept', 'kept']);
  expect(await row.getAttribute('data-diff')).toBe('0');
});

test('shows the speck the scaled gate drops rather than hiding it', async ({ page }) => {
  await openBoundary(page);
  const row = page.getByTestId('boundary-window-speck-0');
  // Visible as a row with both statuses and both areas, not as an absence.
  expect(await statuses(row)).toEqual(['kept', 'filtered']);
  expect(await row.getAttribute('data-baseline-mask-area')).toBe('144');
  expect(await row.getAttribute('data-new-gate-area')).toBe('36');
});

test('renders the difference overlay for every window', async ({ page }) => {
  await openBoundary(page);
  const rows = await page.locator('[data-testid^="boundary-window-"]').count();
  await expect(page.locator('[data-testid^="boundary-canvas-"]')).toHaveCount(rows);
});
