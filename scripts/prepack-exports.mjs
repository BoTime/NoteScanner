// npm only honours a fixed allowlist inside `publishConfig` (registry, access,
// tag, provenance, ...). Arbitrary field overrides such as `publishConfig.exports`
// are a pnpm/Yarn feature and are a silent no-op on npm: the packed tarball keeps
// the dev `exports` map, which points `./styles.css` at `./src/styles.css` (never
// shipped — `files: ["dist"]`) and leaks the `development` condition to Vite-based
// consumers, routing them at the unshipped `src/index.ts`.
//
// So we rewrite the real top-level `exports` field before npm packs. Run by the
// `prepack` lifecycle hook, which fires before both `npm pack` and `npm publish`.
// `publishConfig.exports` stays in package.json as the source of truth for what
// the publish-time map should be; postpack-exports.mjs restores the dev map from
// the backup this script writes.
import { readFileSync, writeFileSync } from 'node:fs';

import { BACKUP_PATH, PKG_PATH, replaceExportsBlock } from './pack-exports-paths.mjs';

const raw = readFileSync(PKG_PATH, 'utf8');
const publishExports = JSON.parse(raw).publishConfig?.exports;

if (!publishExports) {
  console.error(
    'prepack: publishConfig.exports is missing from package.json; refusing to pack a ' +
      'tarball carrying the dev exports map.',
  );
  process.exit(1);
}

const { source, previous } = replaceExportsBlock(raw, publishExports);

// Preserve the dev map for postpack before we overwrite it.
writeFileSync(BACKUP_PATH, `${JSON.stringify(previous, null, 2)}\n`);
writeFileSync(PKG_PATH, source);

console.log('prepack: package.json exports rewritten to publishConfig.exports');
