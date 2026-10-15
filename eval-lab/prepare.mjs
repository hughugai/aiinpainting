/** Build the 50-image eval dataset from Hugging Face (datasets-server rows API).
 *
 * Sources are pinned by (dataset, config, split) + deterministic offsets; the
 * manifest records sha256 of every downloaded file so drift is detectable.
 * Images and masks land in eval-lab/datasets/ (gitignored — Places365 /
 * CelebA-HQ / COCO derivatives are non-commercial research licenses).
 *
 * Usage: node eval-lab/prepare.mjs [--smoke] [--force]
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodeImage, encodePng, encodeMaskPng, decodeMaskPng, centerSquareResize, resizeMask, countHole } from './lib/imageio.mjs';
import { makeMask, hasCleanPatch } from './lib/masks.mjs';

const LAB = dirname(fileURLToPath(import.meta.url));
const DIR = join(LAB, 'datasets');
const IMG = join(DIR, 'images'), MSK = join(DIR, 'masks');
const SIZE = 192;
const SEED_BASE = 0xA11CE;

const SERVER = 'https://datasets-server.huggingface.co';
const SOURCES = [
  { id: 'places', dataset: 'dpdl-benchmark/Places365-Validation', config: 'default', split: 'train', count: 30, imageField: 'image' },
  { id: 'celeb', dataset: 'korexyz/celeba-hq-256x256', config: 'default', split: 'validation', count: 15, imageField: 'image' },
  { id: 'coco', dataset: 'phiyodr/InpaintCOCO', config: 'default', split: 'test', count: 5, imageField: 'coco_image', maskField: 'mask' },
];
const SMOKE = { places: 2, celeb: 2, coco: 1 };

const sha256 = buf => createHash('sha256').update(buf).digest('hex');
const fnv = s => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); } return h >>> 0; };

const sleep = ms => new Promise(r => setTimeout(r, ms));

/** Fetch with retries for transient datasets-server errors (502/503/429, network). */
async function fetchRetry(url, tries = 7) {
  for (let i = 0; ; i++) {
    try {
      const r = await fetch(url);
      if (r.ok) return r;
      if (i >= tries || ![429, 500, 502, 503, 504].includes(r.status)) throw new Error(`${r.status} ${url}`);
    } catch (e) {
      if (i >= tries) throw e;
      console.warn(`[retry ${i + 1}/${tries}] ${e.message}`);
    }
    await sleep(1000 * 2 ** i);
  }
}

async function fetchJson(url) {
  return (await fetchRetry(url)).json();
}

/** Total rows of a (dataset,config,split) via the size endpoint. */
async function totalRows(ds, config, split) {
  const j = await fetchJson(`${SERVER}/size?dataset=${ds}`);
  const row = (j.size?.splits || []).find(s => s.config === config && s.split === split);
  if (!row) throw new Error(`No size for ${ds} ${config}/${split}`);
  return row.num_rows;
}

/** Row at an offset; returns [] when out of range. */
async function rowAt(ds, config, split, offset) {
  const j = await fetchJson(`${SERVER}/rows?dataset=${ds}&config=${config}&split=${split}&offset=${offset}&length=1`);
  return j.rows?.[0]?.row || null;
}

async function fetchBytes(url) {
  const r = await fetchRetry(url);
  return Buffer.from(await r.arrayBuffer());
}

/** Deterministic scan: start at offset, walk forward until a usable row. */
async function pickRow(src, start, wantMask) {
  for (let probe = 0; probe < 25; probe++) {
    const row = await rowAt(src.dataset, src.config, src.split, start + probe);
    if (!row) continue;
    const imgRef = row[src.imageField];
    if (!imgRef?.src) continue;
    if (wantMask && !row[src.maskField]?.src) continue;
    return { row, offset: start + probe };
  }
  throw new Error(`No usable row for ${src.id} near offset ${start}`);
}

function genMaskChecked(tier, w, h, seed) {
  for (let attempt = 0; attempt < 5; attempt++) {
    const m = makeMask(tier, w, h, (seed + attempt * 1013) >>> 0);
    if (hasCleanPatch(m.mask, w, h, 9)) return m;
  }
  throw new Error(`No clean 9x9 patch for ${tier} seed ${seed}`);
}

async function main() {
  const smoke = process.argv.includes('--smoke');
  const force = process.argv.includes('--force');
  for (const d of [DIR, IMG, MSK]) mkdirSync(d, { recursive: true });

  const manifestPath = join(DIR, 'manifest.json');
  const manifest = { size: SIZE, generatedAt: new Date().toISOString(), images: [] };
  if (!force && existsSync(manifestPath)) {
    try {
      const prev = JSON.parse(readFileSync(manifestPath, 'utf8'));
      if (prev.size === SIZE) Object.assign(manifest, prev, { generatedAt: manifest.generatedAt });
    } catch { /* rebuild */ }
  }
  const done = new Set(manifest.images.map(e => e.id));

  for (const src of SOURCES) {
    const count = smoke ? SMOKE[src.id] : src.count;
    const total = await totalRows(src.dataset, src.config, src.split);
    const stride = Math.max(1, Math.floor(total / count));
    for (let i = 0; i < count; i++) {
      const imageId = `${src.id}-${String(i + 1).padStart(2, '0')}`;
      if (done.has(imageId) && existsSync(join(IMG, `${imageId}.png`))) {
        console.log(`[skip] ${imageId}`);
        continue;
      }
      const { row, offset } = await pickRow(src, i * stride, !!src.maskField);

      const raw = await fetchBytes(row[src.imageField].src);
      const img = decodeImage(raw);
      const small = centerSquareResize(img.rgb, img.w, img.h, SIZE);
      writeFileSync(join(IMG, `${imageId}.png`), encodePng(small.rgb, SIZE, SIZE));

      const masks = {};
      for (const tier of ['small', 'thin', 'large']) {
        const m = genMaskChecked(tier, SIZE, SIZE, (SEED_BASE ^ fnv(`${imageId}/${tier}`)) >>> 0);
        writeFileSync(join(MSK, `${imageId}.${tier}.png`), encodeMaskPng(m.mask, SIZE, SIZE));
        masks[tier] = { seed: m.seed, frac: +m.frac.toFixed(4) };
      }
      if (src.maskField) {
        const rawMask = await fetchBytes(row[src.maskField].src);
        const mimg = decodeMaskPng(rawMask);
        if (mimg.w !== img.w || mimg.h !== img.h)
          throw new Error(`${imageId}: mask/image size mismatch ${mimg.w}x${mimg.h} vs ${img.w}x${img.h}`);
        const m = resizeMask(mimg.mask, mimg.w, mimg.h, SIZE);
        const frac = countHole(m.mask) / (SIZE * SIZE);
        if (frac < 0.002 || frac > 0.65 || !hasCleanPatch(m.mask, SIZE, SIZE, 9))
          throw new Error(`${imageId}: native mask unusable (frac=${frac.toFixed(3)})`);
        writeFileSync(join(MSK, `${imageId}.object.png`), encodeMaskPng(m.mask, SIZE, SIZE));
        masks.object = { native: true, frac: +frac.toFixed(4) };
      }

      manifest.images.push({
        id: imageId, source: src.id, dataset: src.dataset, config: src.config, split: src.split,
        offset, rowId: row.img_id ?? row.id ?? null,
        sha256: sha256(raw), masks,
      });
      writeFileSync(manifestPath, JSON.stringify(manifest, null, 2)); // incremental: crash-safe
      console.log(`[ok] ${imageId} <- offset ${offset} (${img.w}x${img.h})`);
    }
  }

  manifest.generatedAt = new Date().toISOString();
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`manifest: ${manifestPath} (${manifest.images.length} images)`);
}

main().catch(e => { console.error(e); process.exit(1); });
