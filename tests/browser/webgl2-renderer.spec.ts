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
  await page.waitForFunction(() => Boolean(window.__harness && window.__viewerHarness));
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

/**
 * Scenes the two backends are genuinely comparable on. Every mask is authored
 * at image resolution (maskScale 1) because canvas2d ignores `width`/`height`
 * by contract, and dpr is pinned to 1 inside the harness.
 *
 * FLIP SENSITIVITY. Every layer in the composite fragment shader is sampled
 * through the same `vUv = vec2(p.x, 1.0 - p.y)`, so a wrong sign there mirrors
 * the base image and the mask layers TOGETHER, and a fixture that is symmetric
 * about both axes cannot see that from luminance probes alone. This
 * differential can, because the harness base image is asymmetric on both axes
 * by construction: its checker term is `((x >> 3) + (y >> 3)) & 1` on a 64x48
 * board, and `(47 - y) >> 3 = 5 - (y >> 3)` and `(63 - x) >> 3 = 7 - (x >> 3)`
 * both flip that parity, so every one of the 3072 base pixels changes by 40/255
 * (18/255 under the dim wash) when mirrored on either axis. The only pixels a
 * mirrored frame can still agree on are the ones the constant-colour orange
 * outline covers in BOTH the true frame and its mirror — which is why the
 * solid-outline rows land near two thirds of the frame rather than all of it,
 * and why the dashed hover ring, whose gaps rarely line up with themselves,
 * reaches all 3072.
 *
 * Measured, by comparing each canvas2d baseline against the MIRROR of the
 * webgl2 frame (chromium, same run as the numbers below): single selection
 * 2316 v / 2316 h, two overlapping selections 2056 v / 2208 h, hover 3072 v /
 * 3072 h, draft polygon 2314 v / 2292 h — against gates of 30 px (1%) and, for
 * draft, 614 px (20%). Every row fails loudly on a uniform flip.
 *
 * The `two overlapping selections` row goes further and is asymmetric in the
 * MASKS too (x [8,36) and [24,54), y [8,36) and [16,42), none of which map to
 * themselves under a 64- or 48-flip), so it is also the row that catches a flip
 * confined to the coverage upload or the mask pass, which would leave the base
 * image where it is.
 */
const DIFF_SCENES: { name: string; gate: number; spec: SceneSpec }[] = [
  {
    name: 'single selection',
    // Both backends produce the same binary coverage, the same radius-3
    // dilation and the same 0.55 dim factor, so this should be near-exact.
    gate: 0.01,
    spec: ONE_MASK,
  },
  {
    name: 'two overlapping selections',
    gate: 0.01,
    spec: {
      width: 64,
      height: 48,
      masks: [
        { id: 'a', rect: [8, 8, 28, 28] },
        { id: 'b', rect: [24, 16, 30, 26] },
      ],
      selected: ['a', 'b'],
    },
  },
  {
    name: 'hover (dashed outline)',
    gate: 0.01,
    spec: { ...ONE_MASK, masks: [{ id: 'a', rect: [16, 12, 32, 24] }], selected: [], hovered: 'a' },
  },
  {
    name: 'draft polygon',
    // Loose ON PURPOSE and reported rather than asserted tight: canvas2d
    // strokes an antialiased path with round caps and `ctx.arc` dots, webgl2
    // draws hard-edged triangles and 16-gon discs. The disagreement here is
    // edge antialiasing, and it is the number this row exists to publish.
    gate: 0.2,
    spec: {
      ...ONE_MASK,
      draft: [
        { x: 8, y: 8 },
        { x: 52, y: 14 },
        { x: 30, y: 40 },
      ],
    },
  },
];

/** A pixel disagrees when any channel differs by more than 8/255 — enough
 *  slack for a rounding difference in the dim multiply, not enough to hide a
 *  wrong colour or a missing outline. */
function disagreements(a: Frame, b: Frame): number {
  let count = 0;
  for (let i = 0; i < a.pixels.length; i += 4) {
    for (let c = 0; c < 4; c += 1) {
      if (Math.abs(a.pixels[i + c] - b.pixels[i + c]) > 8) {
        count += 1;
        break;
      }
    }
  }
  return count;
}

for (const row of DIFF_SCENES) {
  test(`AC3: ${row.name} — per-pixel disagreement vs the canvas2d baseline`, async ({
    page,
  }, testInfo) => {
    test.skip(!(await webgl2Available(page)), 'this engine has no webgl2 context');
    const [c2d, wgl] = await page.evaluate(
      async (spec) => [
        await window.__harness.paint('canvas2d', spec),
        await window.__harness.paint('webgl2', spec),
      ],
      row.spec,
    );
    const total = row.spec.width * row.spec.height;
    const differing = disagreements(c2d, wgl);
    const fraction = differing / total;
    // Reported as a number, per AC3 — never as an unmeasured "matches".
    const line = `[ac3] ${testInfo.project.name} | ${row.name} | ${differing}/${total} px (${(fraction * 100).toFixed(3)}%)`;
    console.log(line);
    testInfo.annotations.push({ type: 'ac3', description: line });
    expect(fraction).toBeLessThanOrEqual(row.gate);
    expect(problems).toEqual([]);
  });
}

test('AC5: live GPU textures stay bounded as the selection grows', async ({ page }, testInfo) => {
  test.skip(!(await webgl2Available(page)), 'this engine has no webgl2 context');
  // 24 non-overlapping 8x8 masks on a 64x48 board, selected one more at a
  // time. A renderer that kept a texture per mask would end at 24+ live
  // textures; this one allocates base + bright + outline + edgeA + edgeB +
  // hoverBright + hoverOutline + scratch = 8, and nothing per mask.
  const specs = [];
  const masks = [];
  for (let i = 0; i < 24; i += 1) {
    masks.push({ id: `m${i}`, rect: [(i % 8) * 8, Math.floor(i / 8) * 8, 8, 8] });
  }
  for (let n = 0; n <= 24; n += 1) {
    specs.push({ width: 64, height: 48, masks, selected: masks.slice(0, n).map((m) => m.id) });
  }
  const { peak } = await page.evaluate(
    (s) => window.__harness.paintSequence('webgl2', s),
    specs as unknown as SceneSpec[],
  );
  console.log(`[ac5] ${testInfo.project.name} | peak live textures across 0->24 masks: ${peak}`);
  testInfo.annotations.push({ type: 'ac5', description: `peak textures ${peak}` });
  // The bound, not the exact figure: an implementation detail that adds one
  // more constant target should not fail this, but one that adds a texture per
  // mask must. 24 masks with a per-mask texture would be >= 24.
  expect(peak).toBeLessThanOrEqual(12);
  expect(problems).toEqual([]);
});

test('AC7: draw() during context loss is silent, and the scene comes back after restore', async ({
  page,
}, testInfo) => {
  test.skip(!(await webgl2Available(page)), 'this engine has no webgl2 context');
  const result = await page.evaluate((spec) => window.__harness.loseAndRestore(spec), ONE_MASK);
  test.skip(!result.supported, 'this engine does not expose WEBGL_lose_context');
  expect(result.threwWhileLost).toBe(false);
  // Falsifiable, unlike threwWhileLost alone: every GL call inside a lost
  // context is a silent no-op by spec, so a missing guard would still not
  // throw. A draw() that actually reached the composite pass increments this.
  expect(result.drawCallsWhileLost).toBe(0);
  const before: Frame = { width: result.width, height: result.height, pixels: result.before };
  const after: Frame = { width: result.width, height: result.height, pixels: result.after };
  const differing = disagreements(before, after);
  const total = result.width * result.height;
  console.log(`[ac7] ${testInfo.project.name} | post-restore disagreement: ${differing}/${total} px`);
  testInfo.annotations.push({ type: 'ac7', description: `restore delta ${differing}/${total}` });
  expect(differing / total).toBeLessThanOrEqual(0.001);
  expect(problems).toEqual([]);
});

test('AC2: with webgl2 unavailable the viewer falls back to canvas2d and still paints', async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    window.__viewerHarness.mountViewer({ denyWebgl2: true }),
  );
  // The probe ran for real and was refused, then canvas2d took the canvas.
  expect(result.contextKinds).toContain('webgl2');
  expect(result.contextKinds).toContain('2d');
  expect(result.pixels).not.toBeNull();
  const frame: Frame = { width: result.width, height: result.height, pixels: result.pixels! };
  const lum = (p: number[]) => p[0] + p[1] + p[2];
  // Selected mask covers [16,12]-[48,36]; (32,24) is inside it, (2,2) is not.
  expect(lum(pixelAt(frame, 32, 24))).toBeGreaterThan(lum(pixelAt(frame, 2, 2)) * 1.5);
  await page.evaluate(() => window.__viewerHarness.unmountViewer());
  expect(problems).toEqual([]);
});

test('AC6: an explicitly passed renderer prop is the one that paints', async ({ page }) => {
  const result = await page.evaluate(() =>
    window.__viewerHarness.mountViewer({ useRendererProp: true }),
  );
  expect(result.propDrawCalls).toBeGreaterThan(0);
  // The default factory probes with getContext('webgl2'); the injected renderer
  // never does. Its absence from the log is what proves the prop won, in a
  // browser where webgl2 IS available.
  expect(result.contextKinds).not.toContain('webgl2');
  await page.evaluate(() => window.__viewerHarness.unmountViewer());
  expect(problems).toEqual([]);
});
