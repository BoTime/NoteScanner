import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
const pkg = JSON.parse(readFileSync(path.join(root, 'package.json'), 'utf8'));
const allowed = new Set([
  ...Object.keys(pkg.peerDependencies ?? {}),
  'react/jsx-runtime',
  'react/jsx-dev-runtime',
]);

const required = [
  'dist/index.js',
  'dist/index.cjs',
  'dist/index.d.ts',
  'dist/core/index.js',
  'dist/core/index.cjs',
  'dist/core/index.d.ts',
  'dist/segmenter/index.js',
  'dist/segmenter/index.cjs',
  'dist/segmenter/index.d.ts',
  'dist/segmenter/worker.js',
  'dist/segment-viewer.css',
];

let failed = false;
for (const rel of required) {
  if (!existsSync(path.join(root, rel))) {
    console.error(`missing artifact: ${rel}`);
    failed = true;
  }
}

// No runtime deps: every bare import in the ESM bundles must be a declared peer.
const bare = /from\s*["']([^".'/][^"']*|@[^"']+)["']/g;
for (const rel of ['dist/index.js', 'dist/core/index.js']) {
  const src = readFileSync(path.join(root, rel), 'utf8');
  for (const [, spec] of src.matchAll(bare)) {
    if (spec.startsWith('.') || spec.startsWith('node:')) continue;
    if (!allowed.has(spec)) {
      console.error(`unresolved external in ${rel}: ${spec}`);
      failed = true;
    }
  }
}

// The peer allow-list above cannot protect the zero-runtime-dependency promise
// any more: `@huggingface/transformers` is now a declared (optional) peer, so
// it is IN that allow-list by construction. The real invariant is per-entry —
// only the /segmenter subpath may reach for it.
const HEAVY_OPTIONAL_PEER = /["']@huggingface\/transformers["']/;
for (const rel of ['dist/index.js', 'dist/core/index.js']) {
  const src = readFileSync(path.join(root, rel), 'utf8');
  if (HEAVY_OPTIONAL_PEER.test(src)) {
    console.error(
      `${rel} imports @huggingface/transformers; only the /segmenter subpath may`,
    );
    failed = true;
  }
}

// The core subpath must be React-free — that is the point of exporting it.
const core = readFileSync(path.join(root, 'dist/core/index.js'), 'utf8');
if (/from\s*["']react/.test(core)) {
  console.error('dist/core/index.js imports react; /core must stay pure');
  failed = true;
}

// The CJS entry must actually load.
try {
  require(path.join(root, 'dist/core/index.cjs'));
} catch (err) {
  console.error(`dist/core/index.cjs failed to require: ${err.message}`);
  failed = true;
}

if (failed) process.exit(1);
console.log('build smoke: OK');
