import { defineConfig, devices } from '@playwright/test';

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
  webServer: {
    command: 'npm run playground',
    url: 'http://localhost:5180/',
    // Reuse a dev server the developer already has up; in CI always start one.
    reuseExistingServer: !process.env.CI,
    // Vite pre-bundles @huggingface/transformers at startup (see
    // playground/vite.config.ts), which is slow the first time.
    timeout: 180_000,
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
  ],
});
