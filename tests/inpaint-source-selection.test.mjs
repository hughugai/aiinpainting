import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createRequire } from 'node:module';
import { test } from 'node:test';

test('removal rejects a matching source rim with text hidden inside', () => {
  const output = mkdtempSync(join(tmpdir(), 'inpaint-test-'));
  try {
    execFileSync('node_modules/.bin/tsc', [
      'algorithms/js/core.ts', '--outDir', output,
      '--module', 'commonjs', '--target', 'es2020', '--skipLibCheck',
    ]);
    const { inpaint } = createRequire(import.meta.url)(join(output, 'core.js'));
    const width = 40, height = 20;
    const rgb = new Uint8Array(width * height * 3).fill(240);
    const mask = new Uint8Array(width * height);
    const target = 4 * width + 24;
    mask[target] = 1;
    // The first source has a perfect background rim, but a dark glyph at
    // exactly the offset that will be copied into the missing target pixel.
    rgb.fill(0, (3 * width + 2) * 3, (3 * width + 2) * 3 + 3);
    rgb.fill(0, target * 3, target * 3 + 3);
    const original = rgb.slice();
    const baseline = inpaint(rgb, mask, width, height, { patch: 5 });
    assert.equal(baseline.image[target * 3], 0);
    const result = inpaint(rgb, mask, width, height, {
      patch: 5, sourceTextureWeight: 1,
    });
    assert.equal(result.image[target * 3], 240);
    for (let i = 0; i < mask.length; i++)
      if (!mask[i])
        assert.deepEqual(result.image.slice(i * 3, i * 3 + 3), rgb.slice(i * 3, i * 3 + 3));
    assert.deepEqual(rgb, original);
    assert.deepEqual(inpaint(rgb, new Uint8Array(mask.length), width, height).image, rgb);
    assert.throws(() => inpaint(rgb, mask, width, height, { sourceTextureWeight: -1 }));
  } finally {
    rmSync(output, { recursive: true, force: true });
  }
});
