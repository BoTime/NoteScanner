import { defineConfig } from 'tsup';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    'core/index': 'src/core/index.ts',
    'segmenter/index': 'src/segmenter/index.ts',
  },
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  splitting: false,
  external: ['react', 'react-dom', 'react/jsx-runtime'],
  injectStyle: false,
  esbuildOptions(options) {
    options.jsx = 'automatic';
  },
  onSuccess: 'cp src/styles.css dist/segment-viewer.css',
});
