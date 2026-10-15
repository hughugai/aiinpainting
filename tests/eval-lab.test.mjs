import assert from 'node:assert/strict';
import { test } from 'node:test';
import { psnr, ssim } from '../eval-lab/lib/metrics.mjs';
import { makeMask, hasCleanPatch, mulberry32 } from '../eval-lab/lib/masks.mjs';

const W = 48, H = 32;
function flat(v) {
  const rgb = new Uint8Array(W * H * 3);
  for (let i = 0; i < W * H; i++) rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = v;
  return rgb;
}
function noisy(seed) {
  const rgb = new Uint8Array(W * H * 3), rng = mulberry32(seed);
  for (let i = 0; i < rgb.length; i++) rgb[i] = Math.floor(rng() * 256);
  return rgb;
}

test('psnr: identical images report the 100 dB cap', () => {
  const a = flat(120);
  assert.equal(psnr(a, a), 100);
});

test('psnr: known shift matches the closed-form value', () => {
  const a = flat(100), b = flat(90);
  const mse = 100; // every channel differs by 10
  assert.ok(Math.abs(psnr(a, b) - 10 * Math.log10(255 * 255 / mse)) < 1e-9);
});

test('psnr: larger error means lower value', () => {
  assert.ok(psnr(flat(100), flat(90)) > psnr(flat(100), flat(50)));
});

test('ssim: identical images give 1', () => {
  const a = noisy(7);
  assert.ok(Math.abs(ssim(a, a, W, H) - 1) < 1e-12);
});

test('ssim: ordering tracks structural similarity', () => {
  const a = noisy(7), b = noisy(8), c = noisy(7).map(v => Math.min(255, v + 10));
  assert.ok(ssim(a, c, W, H) > ssim(a, b, W, H), 'small brightness shift beats unrelated noise');
  assert.ok(ssim(a, b, W, H) < 0.5);
});

test('masks: deterministic for the same seed', () => {
  const a = makeMask('small', 192, 192, 12345), b = makeMask('small', 192, 192, 12345);
  assert.deepEqual([...a.mask], [...b.mask]);
  assert.equal(a.frac, b.frac);
});

test('masks: tier area bands hold across seeds', () => {
  const bands = { small: [.05, .20], thin: [.003, .035], large: [.38, .54] };
  for (const [tier, [lo, hi]] of Object.entries(bands))
    for (let seed = 0; seed < 12; seed++) {
      const m = makeMask(tier, 192, 192, seed * 977 + 1);
      assert.ok(m.frac >= lo && m.frac <= hi, `${tier} seed ${seed}: frac=${m.frac}`);
    }
});

test('masks: large masks still leave a clean 9x9 source patch', () => {
  for (let seed = 0; seed < 12; seed++) {
    const m = makeMask('large', 192, 192, seed * 131 + 7);
    assert.ok(hasCleanPatch(m.mask, 192, 192, 9), `seed ${seed}`);
  }
});

test('masks: different seeds give different holes', () => {
  const a = makeMask('thin', 192, 192, 1), b = makeMask('thin', 192, 192, 2);
  assert.ok(a.mask.some((v, i) => v !== b.mask[i]));
});
