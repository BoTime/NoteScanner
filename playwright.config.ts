import { defineConfig, devices } from '@playwright/test';

/**
 * Browser-only verification of the 1-bit indexed mask PNG.
 *
 * There is no `webServer` and no WebGPU here on purpose: the spec encodes the
 * masks in Node and hands them into the page as data URLs, so all three
 * engines run headless with nothing to serve. `npm test` (vitest) never sees
 * these files — its `include` only matches `src/` and `playground/`.
 */
export default defineConfig({
  testDir: './tests/browser',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  reporter: process.env.CI ? 'github' : 'list',
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
  ],
});
