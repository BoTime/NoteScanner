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

    // Click-to-select works on real output. Clicking and asserting nothing
    // would pass just as well with click-to-select dead, so this reads the
    // selection count the page exposes and requires it to actually rise.
    const shown = page.getByTestId('site-image');
    expect(await shown.getAttribute('data-selected')).toBe('0');

    const canvas = page.locator('canvas').first();
    const box = await canvas.boundingBox();
    // Thrown, not asserted: a null box must stop the test here, and `expect`
    // alone would leave `box` typed `null` for the clicks below.
    if (!box) throw new Error('the viewer canvas has no layout box');

    // Selecting is two steps, not one: a canvas click hit-tests the masks and
    // opens a role="menu" of what is under the cursor, and the "Select" item
    // in it is what actually changes the selection. A test that only clicked
    // the canvas would pass with selection entirely broken.
    const menu = page.getByRole('menu');
    // Several points, because any single one may land on background rather
    // than on a mask — that would be a flaky test, not a real failure.
    const spots = [0.5, 0.35, 0.65].flatMap((x) => [0.5, 0.35, 0.65].map((y) => ({ x, y })));
    let opened = false;
    for (const spot of spots) {
      await canvas.click({ position: { x: box.width * spot.x, y: box.height * spot.y } });
      if (await menu.isVisible().catch(() => false)) {
        opened = true;
        break;
      }
    }
    expect(opened, 'no click anywhere on the viewer hit a mask').toBe(true);

    await menu.getByRole('menuitem', { name: /- Select$/ }).first().click();

    expect(
      Number(await shown.getAttribute('data-selected')),
      'the Select menu item did not change the selection',
    ).toBeGreaterThan(0);

    // Printed, not just asserted: this is the AC6 evidence the run records.
    console.log('progress lines:', seen);
    console.log('stat line:', stats);
  });
});
