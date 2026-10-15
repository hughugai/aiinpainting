/** PSNR and SSIM for the eval harness. Both operate on flat RGB Uint8Arrays. */

/** PSNR in dB over all channels; identical inputs report the 100 dB cap. */
export function psnr(a, b) {
  if (a.length !== b.length) throw new Error('PSNR: length mismatch');
  let se = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; se += d * d; }
  if (se === 0) return 100;
  return Math.min(100, 10 * Math.log10(255 * 255 / (se / a.length)));
}

function gray(rgb, w, h) {
  const g = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++)
    g[i] = rgb[i * 3] * .299 + rgb[i * 3 + 1] * .587 + rgb[i * 3 + 2] * .114;
  return g;
}

const K = 11, SIGMA = 1.5, C1 = (0.01 * 255) ** 2, C2 = (0.03 * 255) ** 2;
const GAUSS = (() => {
  const k = new Float64Array(K);
  let sum = 0;
  for (let i = 0; i < K; i++) { const d = i - (K - 1) / 2; k[i] = Math.exp(-d * d / (2 * SIGMA * SIGMA)); sum += k[i]; }
  for (let i = 0; i < K; i++) k[i] /= sum;
  return k;
})();

/** Separable Gaussian filter, 'valid' mode: output is (w-K+1) x (h-K+1). */
function filt(src, w, h) {
  const ow = w - K + 1, oh = h - K + 1, tmp = new Float64Array(ow * h), out = new Float64Array(ow * oh);
  for (let y = 0; y < h; y++) for (let x = 0; x < ow; x++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += src[y * w + x + k] * GAUSS[k];
    tmp[y * ow + x] = s;
  }
  for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += tmp[(y + k) * ow + x] * GAUSS[k];
    out[y * ow + x] = s;
  }
  return out;
}

/** Mean SSIM (Wang et al. 2004) on luma, 11x11 Gaussian window, valid region. */
export function ssim(a, b, w, h) {
  if (a.length !== b.length || a.length !== w * h * 3) throw new Error('SSIM: dimension mismatch');
  const x = gray(a, w, h), y = gray(b, w, h);
  const n = w * h;
  const x2 = new Float64Array(n), y2 = new Float64Array(n), xy = new Float64Array(n);
  for (let i = 0; i < n; i++) { x2[i] = x[i] * x[i]; y2[i] = y[i] * y[i]; xy[i] = x[i] * y[i]; }
  const ow = w - K + 1, oh = h - K + 1;
  const mx = filt(x, w, h), my = filt(y, w, h), mxx = filt(x2, w, h), myy = filt(y2, w, h), mxy = filt(xy, w, h);
  let total = 0;
  for (let i = 0; i < ow * oh; i++) {
    const ux = mx[i], uy = my[i];
    const vx = mxx[i] - ux * ux, vy = myy[i] - uy * uy, cov = mxy[i] - ux * uy;
    total += ((2 * ux * uy + C1) * (2 * cov + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2));
  }
  return total / (ow * oh);
}
