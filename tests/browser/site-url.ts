/**
 * The one place the built site's preview URL is written.
 *
 * The port must match `preview.port` in `site/vite.config.ts`. If the two ever
 * disagree, Playwright's `webServer` wait on this exact URL times out at start
 * of run, rather than producing a confusing per-test failure later.
 */
export const SITE_URL = 'http://localhost:5181/NoteScanner/';
