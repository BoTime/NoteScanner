// Shared helpers for prepack-exports.mjs / postpack-exports.mjs.
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const PKG_PATH = path.join(ROOT, 'package.json');
// Gitignored scratch file holding the dev `exports` block while a pack is in flight.
export const BACKUP_PATH = path.join(ROOT, '.exports-dev-backup.json');

/**
 * Replace the value of the top-level `"exports"` key in a package.json source
 * string, touching nothing else.
 *
 * We splice text rather than JSON.parse/stringify the whole file because a
 * round-trip through JSON.stringify reflows every other field (collapsing or
 * expanding unrelated arrays), which would leave cosmetic churn in a
 * developer's working tree after a local `npm pack`.
 *
 * @param {string} source raw package.json contents
 * @param {unknown} value new value for the `exports` field
 * @returns {{ source: string, previous: unknown }}
 */
export function replaceExportsBlock(source, value) {
  const keyMatch = /^(\s*)"exports"\s*:\s*/m.exec(source);
  if (!keyMatch) throw new Error('could not locate a top-level "exports" key in package.json');

  const indent = keyMatch[1].replace(/^\r?\n/, '');
  const valueStart = keyMatch.index + keyMatch[0].length;
  const valueEnd = findValueEnd(source, valueStart);
  const previous = JSON.parse(source.slice(valueStart, valueEnd));

  // Indent the replacement to sit at the same depth as the key it belongs to.
  const rendered = JSON.stringify(value, null, 2).replace(/\n/g, `\n${indent}`);

  return { source: source.slice(0, valueStart) + rendered + source.slice(valueEnd), previous };
}

/** Scan forward from the start of a JSON value to just past its final character. */
function findValueEnd(source, start) {
  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let i = start; i < source.length; i += 1) {
    const ch = source[i];

    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }

    if (ch === '"') inString = true;
    else if (ch === '{' || ch === '[') depth += 1;
    else if (ch === '}' || ch === ']') {
      depth -= 1;
      if (depth === 0) return i + 1;
    }
  }

  throw new Error('unterminated "exports" value in package.json');
}
