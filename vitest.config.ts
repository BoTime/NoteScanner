import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: [
      'src/**/*.test.ts',
      'src/**/*.test.tsx',
      'playground/**/*.test.ts',
      'playground/**/*.test.tsx',
      'site/**/*.test.ts',
      'site/**/*.test.tsx',
    ],
  },
});
