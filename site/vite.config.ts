import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  root: __dirname,
  // Served from https://botime.github.io/NoteScanner/, so every emitted asset
  // URL has to carry the repository path segment.
  base: '/NoteScanner/',
  plugins: [react()],
  // The segmenter worker is constructed as a module Worker and reaches
  // @huggingface/transformers through a DYNAMIC import. Vite's default worker
  // output format is 'iife', which cannot express a dynamic import, so a
  // production build of an app that uses the worker fails or ships a broken
  // one unless the worker is emitted as an ES module. The dev server serves
  // module workers natively, which is why the playground's config never needed
  // this: nothing in this repo has ever production-built an app using it.
  worker: { format: 'es' },
  // Same reason as the playground's config: @huggingface/transformers is
  // reachable only through that dynamic import inside the worker, so Vite's
  // dependency scanner does not see it and would otherwise discover it on the
  // first inference and force a full-page reload mid-run.
  optimizeDeps: { include: ['@huggingface/transformers'] },
  // 5181, one past the playground's 5180, so both can run at once. The preview
  // port is the one tests/browser/site-url.ts encodes; if the two ever
  // disagree, Playwright's webServer wait fails immediately and loudly.
  server: { port: 5181, strictPort: true },
  preview: { port: 5181, strictPort: true },
});
