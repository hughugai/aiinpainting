/** Visual sanity run: 5 rows from the local full InpaintCOCO copy.
 *
 * For each row: center-crop to 256x256, paint the dataset's real object mask,
 * run feature (texture-aware on/off) + Telea, and write a side-by-side panel:
 *   original | masked | feature | feature-plain | telea
 * to eval-lab/work/sample5/ with hole-region PSNR, full PSNR/SSIM and timing.
 *
 * Usage: node eval-lab/sample5.mjs [--rows 0,250,500,750,1000]
 */
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodeImage, decodeMaskPng, centerSquareResize, resizeMask, encodePng } from './lib/imageio.mjs';
import { psnr, ssim } from './lib/metrics.mjs';
import { ALGOS, warmup } from './lib/algos.mjs';

const LAB = dirname(fileURLToPath(import.meta.url));
const DS = join(LAB, 'datasets', 'inpaintcoco');
const OUT = join(LAB, 'work', 'sample5');
const SIZE = 256;
const ALGO_LIST = ['feature', 'feature-plain', 'telea'];

const rowsArg = process.argv.indexOf('--rows');
const ROWS = rowsArg > -1 ? process.argv[rowsArg + 1].split(',').map(Number) : null;

const psnrFromMse = mse => mse === 0 ? 100 : Math.min(100, 10 * Math.log10(255 * 255 / mse));
function holeMse(out, ref, mask) {
  let s = 0, n = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i])
    for (let c = 0; c < 3; c++) { const d = out[i * 3 + c] - ref[i * 3 + c]; s += d * d; n++; }
  return n ? s / n : 0;
}

/** Side-by-side panel of equal-size RGB cells separated by white gaps. */
function panel(size, cells) {
  const GAP = 6, W = cells.length * size + (cells.length - 1) * GAP, H = size;
  const out = new Uint8Array(W * H * 3).fill(255);
  cells.forEach((rgb, k) => {
    const x0 = k * (size + GAP);
    for (let y = 0; y < H; y++) for (let x = 0; x < size; x++) {
      const s = (y * size + x) * 3, d = (y * W + x0 + x) * 3;
      out[d] = rgb[s]; out[d + 1] = rgb[s + 1]; out[d + 2] = rgb[s + 2];
    }
  });
  return { rgb: out, w: W, h: H };
}

await warmup();
mkdirSync(OUT, { recursive: true });
const meta = readFileSync(join(DS, 'metadata.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);

/** Native mask hole fraction (InpaintCOCO masks often cover most of the image —
 * only rows with a modest object are usable for classical inpainting). */
function loadCase(m) {
  const img = decodeImage(readFileSync(join(DS, 'images', m.files.images)));
  const mk = decodeMaskPng(readFileSync(join(DS, 'masks', m.files.masks)));
  if (mk.w !== img.w || mk.h !== img.h) return null;
  let hole = 0;
  for (let i = 0; i < mk.mask.length; i++) hole += mk.mask[i] ? 1 : 0;
  return { m, img, mk, frac: hole / mk.mask.length };
}

let cases;
if (ROWS) {
  cases = ROWS.map(i => meta[i]).filter(Boolean)
    .map(loadCase).filter(c => c && c.frac <= 0.9);
} else {
  // scan from the top until 5 rows with 1–10% hole coverage are found
  cases = [];
  for (const m of meta) {
    if (cases.length >= 5) break;
    const c = loadCase(m);
    if (c && c.frac >= 0.01 && c.frac <= 0.10) cases.push(c);
  }
}
console.log(`picked ${cases.length} rows (filter: native hole 1-10%)`);

for (const { m, img, mk, frac: nativeFrac } of cases) {
  const a = centerSquareResize(img.rgb, img.w, img.h, SIZE);
  const b = resizeMask(mk.mask, mk.w, mk.h, SIZE);
  let hole = 0;
  for (let i = 0; i < b.mask.length; i++) hole += b.mask[i] ? 1 : 0;
  const frac = hole / (SIZE * SIZE);

  const masked = new Uint8Array(a.rgb);
  for (let i = 0; i < SIZE * SIZE; i++)
    if (b.mask[i]) { masked[i * 3] = masked[i * 3 + 1] = masked[i * 3 + 2] = 128; }

  console.log(`\n== row ${m.row_index} (coco ${m.coco_image_id}, ${m.concept}) native hole ${(nativeFrac * 100).toFixed(1)}% -> cropped ${(frac * 100).toFixed(1)}%`);
  console.log(`   caption: ${(m.coco_caption || '').slice(0, 76)}`);

  const cells = [a.rgb, masked];
  for (const id of ALGO_LIST) {
    const t0 = performance.now();
    const out = await ALGOS[id].run(a.rgb, b.mask, SIZE, SIZE);
    const ms = Math.round(performance.now() - t0);
    const p = psnrFromMse(holeMse(out, a.rgb, b.mask));
    const f = psnr(out, a.rgb), s = ssim(out, a.rgb, SIZE, SIZE);
    console.log(`   ${id.padEnd(14)} PSNR_hole=${p.toFixed(2).padStart(6)}dB  PSNR_full=${f.toFixed(2).padStart(6)}dB  SSIM=${s.toFixed(4)}  ${ms}ms`);
    cells.push(out);
  }
  const p = panel(SIZE, cells);
  const file = join(OUT, `row-${String(m.row_index).padStart(5, '0')}-coco${m.coco_image_id}.png`);
  writeFileSync(file, encodePng(p.rgb, p.w, p.h));
  console.log(`   panel -> ${file}`);
}
