// Restores the dev `exports` map that prepack-exports.mjs swapped out, so a
// developer running `npm pack` locally is not left with the publish-time map in
// their working tree. In CI the container is discarded after `npm publish`, so
// this is a no-op safety net there.
import { existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs';

import { BACKUP_PATH, PKG_PATH, replaceExportsBlock } from './pack-exports-paths.mjs';

if (!existsSync(BACKUP_PATH)) {
  // prepack did not run, or the restore already happened — nothing to undo.
  process.exit(0);
}

const devExports = JSON.parse(readFileSync(BACKUP_PATH, 'utf8'));
const { source } = replaceExportsBlock(readFileSync(PKG_PATH, 'utf8'), devExports);

writeFileSync(PKG_PATH, source);
rmSync(BACKUP_PATH);

console.log('postpack: package.json exports restored to the dev map');
