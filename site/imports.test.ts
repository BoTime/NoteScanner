import { mkdtempSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect } from 'vitest';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

/** Build output and dependencies are not sources; scanning `site/dist` would
 *  check the bundler's work rather than ours. */
const SKIP_DIRS = new Set(['dist', 'node_modules', '.vite']);

function sourceFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) {
      if (!SKIP_DIRS.has(entry)) out.push(...sourceFiles(full));
    } else if (/\.(ts|tsx|js|jsx|mjs|cjs|css|html)$/.test(entry)) {
      out.push(full);
    }
  }
  return out;
}

/** Every way one module can name another. The bare-`import` and `new URL`
 *  forms matter most: a side-effect import (`import '../playground/x.css'`)
 *  carries no `from`, and a worker is referenced as
 *  `new Worker(new URL('...', import.meta.url))` — which is exactly how
 *  `src/segmenter/createSegmenter.ts` loads its worker and how `site/main.tsx`
 *  pulls in stylesheets, so these are the realistic ways the boundary would be
 *  breached, not exotic ones. */
function specifiers(file: string): string[] {
  const text = readFileSync(file, 'utf8');
  const found: string[] = [];
  const patterns = [
    // import x from '…' / export … from '…'
    /(?:^|[\s;}])(?:import|export)\s[^;]*?from\s*['"]([^'"]+)['"]/g,
    // side-effect import, no `from` at all
    /(?:^|[\s;}])import\s*['"]([^'"]+)['"]/g,
    // dynamic import(), tolerating comments and whitespace before the string
    /\bimport\s*\(\s*(?:\/\*[\s\S]*?\*\/\s*)?['"]([^'"]+)['"]/g,
    // new URL('…', import.meta.url) — the worker/asset reference form
    /\bnew\s+URL\s*\(\s*['"]([^'"]+)['"]/g,
    /\brequire\s*\(\s*['"]([^'"]+)['"]/g,
    /<script[^>]+src=['"]([^'"]+)['"]/g,
    // CSS @import, since stylesheets are scanned too
    /@import\s+(?:url\(\s*)?['"]([^'"]+)['"]/g,
  ];
  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) found.push(match[1]);
  }
  return found;
}

const siteFiles = sourceFiles(path.join(repoRoot, 'site'));
const playgroundFiles = sourceFiles(path.join(repoRoot, 'playground'));

describe('the AC8 guard actually detects the forms it claims to', () => {
  // Without this, a regex that silently matches nothing would let every
  // assertion below pass while the boundary was wide open.
  // Assembled at runtime, never written as one literal: this file lives under
  // `site/`, so the real scan below reads it too — a literal breach path here
  // would make the guard report itself.
  const PG = `../${'play' + 'ground'}`;

  const breaches: Array<{ label: string; source: string; expected: string }> = [
    { label: 'static import', source: `import x from '${PG}/x';`, expected: `${PG}/x` },
    { label: 'side-effect import', source: `import '${PG}/x.css';`, expected: `${PG}/x.css` },
    { label: 're-export', source: `export { x } from '${PG}/x';`, expected: `${PG}/x` },
    { label: 'dynamic import', source: `void import('${PG}/x');`, expected: `${PG}/x` },
    {
      label: 'dynamic import with a comment',
      source: `void import(/* vite-ignore */ '${PG}/x');`,
      expected: `${PG}/x`,
    },
    {
      label: 'new URL worker reference',
      source: `new Worker(new URL('${PG}/w.ts', import.meta.url));`,
      expected: `${PG}/w.ts`,
    },
    { label: 'require', source: `const x = require('${PG}/x');`, expected: `${PG}/x` },
    {
      label: 'script src',
      source: `<script type="module" src="${PG}/main.tsx"></script>`,
      expected: `${PG}/main.tsx`,
    },
    { label: 'css @import', source: `@import '${PG}/theme.css';`, expected: `${PG}/theme.css` },
  ];

  for (const { label, source, expected } of breaches) {
    it(`detects a ${label}`, () => {
      const file = path.join(mkdtempSync(path.join(tmpdir(), 'ac8-')), 'probe.txt');
      writeFileSync(file, source);
      expect(specifiers(file)).toContain(expected);
    });
  }
});

describe('the public site and the dev playground stay separate (AC8)', () => {
  it('found files on both sides to check', () => {
    // Without this, the two assertions below pass by iterating nothing.
    expect(siteFiles.length).toBeGreaterThan(0);
    expect(playgroundFiles.length).toBeGreaterThan(0);
  });

  it('nothing under site/ imports from playground/', () => {
    for (const file of siteFiles) {
      for (const spec of specifiers(file)) {
        expect(spec, `${path.relative(repoRoot, file)} imports ${spec}`).not.toMatch(
          /(^|\/)playground(\/|$)/,
        );
      }
    }
  });

  it('nothing under playground/ imports from site/', () => {
    for (const file of playgroundFiles) {
      for (const spec of specifiers(file)) {
        expect(spec, `${path.relative(repoRoot, file)} imports ${spec}`).not.toMatch(
          /(^|\/)site(\/|$)/,
        );
      }
    }
  });
});
