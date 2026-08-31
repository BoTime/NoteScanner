import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  root: __dirname,
  plugins: [react()],
  server: { port: 5180 },
  // `@huggingface/transformers` is only reachable through a dynamic import
  // inside the SAM worker module (src/segmenter/worker/segmenter.worker.ts),
  // not from the page's static import graph. Vite's dependency scanner does
  // not see it until the worker actually runs, so without this it gets
  // discovered lazily on first inference and Vite fires a full-page reload
  // mid-request to pre-bundle it — which, under the sweep runner, destroys
  // the in-page state a row was waiting on (data-run-count) and hangs the
  // row until ROW_TIMEOUT_MS. Listing it here forces pre-bundling at dev
  // server start, before the runner ever navigates, so no reload happens
  // mid-sweep.
  optimizeDeps: { include: ['@huggingface/transformers'] },
});
