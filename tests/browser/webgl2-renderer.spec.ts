import { expect, test, type Page } from '@playwright/test';
import { startHarnessServer } from './dev-server';
import type { Frame, SceneSpec } from './harness/harness';

let server: { url: string; close: () => Promise<void> };

test.beforeAll(async () => {
  server = await startHarnessServer();
});

test.afterAll(async () => {
  await server?.close();
});

const problems: string[] = [];

test.beforeEach(async ({ page }) => {
  page.on('console', (msg) => {
    if (msg.type() === 'error' || msg.text().includes('note-scanner')) problems.push(msg.text());
  });
  page.on('pageerror', (err) => problems.push(String(err)));
  problems.length = 0;
  await page.goto(server.url);
  await page.waitForFunction(() => Boolean(window.__harness));
});

async function webgl2Available(page: Page): Promise<boolean> {
  return page.evaluate(() => window.__harness.webgl2Available());
}

function pixelAt(frame: Frame, x: number, y: number): [number, number, number, number] {
  const o = (y * frame.width + x) * 4;
  return [frame.pixels[o], frame.pixels[o + 1], frame.pixels[o + 2], frame.pixels[o + 3]];
}

/** One selected rectangle, big enough to have an unambiguous interior, a
 *  dilated border ring, and a lot of untouched background. */
const ONE_MASK: SceneSpec = {
  width: 64,
  height: 48,
  masks: [{ id: 'a', rect: [16, 12, 32, 24] }],
  selected: ['a'],
};

test('reports whether this engine can obtain a webgl2 context', async ({ page }, testInfo) => {
  const available = await webgl2Available(page);
  // Never a failure: which of chromium/webkit/firefox exposes WebGL2 headless
  // is exactly the fact this spec exists to record, and every GPU assertion
  // below skips rather than fails where it does not.
  testInfo.annotations.push({
    type: 'webgl2',
    description: `${testInfo.project.name}: webgl2 ${available ? 'available' : 'UNAVAILABLE'}`,
  });
  console.log(`[webgl2-support] ${testInfo.project.name}: ${available ? 'yes' : 'no'}`);
  expect(typeof available).toBe('boolean');
});

test('the WebGL2 renderer paints a dim wash, a bright window and an orange outline', async ({
  page,
}) => {
  test.skip(!(await webgl2Available(page)), 'this engine has no webgl2 context');
  const frame = await page.evaluate(
    (spec) => window.__harness.paint('webgl2', spec),
    ONE_MASK,
  );

  const inside = pixelAt(frame, 32, 24);
  const outside = pixelAt(frame, 2, 2);
  // The dim layer is rgba(0,0,0,0.55), so background is 0.45x the base colour
  // while the bright window keeps it. Comparing the two rules out both "painted
  // nothing" and "dimmed everything".
  const lum = (p: number[]) => p[0] + p[1] + p[2];
  expect(lum(inside)).toBeGreaterThan(lum(outside) * 1.5);

  // The dilated border ring must contain the outline colour #f97316.
  let orange = 0;
  for (let i = 0; i < frame.pixels.length; i += 4) {
    if (
      Math.abs(frame.pixels[i] - 249) <= 8 &&
      Math.abs(frame.pixels[i + 1] - 115) <= 8 &&
      Math.abs(frame.pixels[i + 2] - 22) <= 8
    ) {
      orange += 1;
    }
  }
  // A radius-3 dilation of a 32x24 rectangle's 1px perimeter is a ring roughly
  // 7px thick around a ~112px perimeter, so hundreds of pixels — but the exact
  // count depends on clipping and is not what this asserts. It asserts the ring
  // exists and is not a stray pixel or two.
  expect(orange).toBeGreaterThan(200);

  expect(problems).toEqual([]);
});

test('a half-resolution mask still paints, through the sampler upsample', async ({ page }) => {
  test.skip(!(await webgl2Available(page)), 'this engine has no webgl2 context');
  // Nothing emits `width` / `height` yet (issue #7 will). This is the F4
  // MECHANISM under test, not an F4 saving: a mask authored at half size must
  // light the same region of the image as one authored at full size.
  const frame = await page.evaluate(
    (spec) => window.__harness.paint('webgl2', spec),
    { ...ONE_MASK, masks: [{ id: 'a', rect: [16, 12, 32, 24], maskScale: 2 }] } as SceneSpec,
  );
  const lum = (p: number[]) => p[0] + p[1] + p[2];
  expect(lum(pixelAt(frame, 32, 24))).toBeGreaterThan(lum(pixelAt(frame, 2, 2)) * 1.5);
  expect(problems).toEqual([]);
});
