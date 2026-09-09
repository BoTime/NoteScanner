import { defineConfig, devices } from '@playwright/test';
import { SITE_URL } from './tests/browser/site-url';

/**
 * Browser-only verification, in three engines.
 *
 * Two specs with different needs. `mask-png.spec.ts` encodes its masks in Node
 * and hands them into the page as data URLs, so it needs nothing served.
 * `boundary.spec.ts` drives the running playground, so a `webServer` starts
 * Vite for it — which also serves the first spec harmlessly. Neither needs
 * WebGPU, a model download or the network: firefox and webkit expose no
 * `navigator.gpu` at all, which is exactly why the Boundary tab must not ask
 * for one.
 */
export default defineConfig({
  testDir: './tests/browser',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  reporter: process.env.CI ? 'github' : 'list',
  use: { baseURL: 'http://localhost:5180' },
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
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
  ],
});
