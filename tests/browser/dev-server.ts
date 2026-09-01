import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { createServer, type ViteDevServer } from 'vite';

const here = path.dirname(fileURLToPath(import.meta.url));
const harnessRoot = path.join(here, 'harness');
const repoRoot = path.resolve(here, '..', '..');

/**
 * One Vite dev server per Playwright worker, on an OS-assigned port.
 *
 * `playwright.config.ts` deliberately has no `webServer` — the mask-PNG specs
 * hand data URLs into an empty document and need nothing served. The renderer
 * specs DO need the real TS modules running in the page, so they start a
 * server themselves and close it in `afterAll`. Port 0 (rather than a fixed
 * port) is what lets chromium, webkit and firefox run fully parallel.
 *
 * `fs.allow` has to reach the repo root because the harness imports from
 * `src/`, which is outside the server root.
 *
 * `cacheDir` is a PRIVATE temp directory per server, not the default shared
 * `node_modules/.vite`. Observed on the first real run of this file: three
 * workers starting at once each ran the dependency optimizer against the same
 * cache directory and raced on its atomic rename, printing
 * `ENOENT ... rename '.vite/deps' -> '.vite/deps_temp_*'`. A private cache dir
 * removes the shared resource rather than serializing access to it.
 */
export async function startHarnessServer(): Promise<{ url: string; close: () => Promise<void> }> {
  const cacheDir = fs.mkdtempSync(path.join(os.tmpdir(), 'note-scanner-harness-'));
  let server: ViteDevServer;
  try {
    server = await createServer({
      configFile: false,
      root: harnessRoot,
      cacheDir,
      plugins: [react()],
      logLevel: 'error',
      server: { port: 0, strictPort: false, fs: { allow: [repoRoot] } },
    });
  } catch (err) {
    fs.rmSync(cacheDir, { recursive: true, force: true });
    throw err;
  }
  const close = async () => {
    await server.close();
    fs.rmSync(cacheDir, { recursive: true, force: true });
  };
  await server.listen();
  const url = server.resolvedUrls?.local?.[0];
  if (!url) {
    await close();
    throw new Error('harness vite server started but reported no local URL');
  }
  return { url, close };
}
