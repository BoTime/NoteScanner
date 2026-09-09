import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect } from 'vitest';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function sourceFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) out.push(...sourceFiles(full));
    else if (/\.(ts|tsx|html)$/.test(entry)) out.push(full);
  }
  return out;
}

/** Every module specifier: static imports, re-exports, dynamic import(), and
 *  the script src in an index.html. */
function specifiers(file: string): string[] {
  const text = readFileSync(file, 'utf8');
  const found: string[] = [];
  const patterns = [
    /(?:^|\s)(?:import|export)\s[^;]*?from\s*['"]([^'"]+)['"]/g,
    /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
    /<script[^>]+src=['"]([^'"]+)['"]/g,
  ];
  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) found.push(match[1]);
  }
  return found;
}

const siteFiles = sourceFiles(path.join(repoRoot, 'site'));
const playgroundFiles = sourceFiles(path.join(repoRoot, 'playground'));

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
