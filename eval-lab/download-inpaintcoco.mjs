/** Download the FULL phiyodr/InpaintCOCO test split (1260 rows) from Hugging Face.
 *
 * Saves to eval-lab/datasets/inpaintcoco/ (gitignored — COCO-derived licenses):
 *   images/{ann_id}.jpg|png     original COCO image
 *   masks/{ann_id}.jpg|png      removal mask (white = hole)
 *   inpainted/{ann_id}.jpg|png  SD-generated reference result from the dataset
 *   metadata.jsonl              one line per row: ids, captions, details, file names
 *
 * Idempotent: existing files are skipped. Usage: node eval-lab/download-inpaintcoco.mjs [--jobs N]
 */
import { createWriteStream, existsSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline } from 'node:stream/promises';
import { Readable } from 'node:stream';

const LAB = dirname(fileURLToPath(import.meta.url));
const ROOT = join(LAB, 'datasets', 'inpaintcoco');
const DIRS = { images: join(ROOT, 'images'), masks: join(ROOT, 'masks'), inpainted: join(ROOT, 'inpainted') };
const SERVER = 'https://datasets-server.huggingface.co';
const DS = { dataset: 'phiyodr/InpaintCOCO', config: 'default', split: 'test' };
const FIELDS = { images: 'coco_image', masks: 'mask', inpainted: 'inpaint_image' };

const sleep = ms => new Promise(r => setTimeout(r, ms));
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

const ext = buf => buf[0] === 0x89 ? 'png' : 'jpg';

/** Simple promise pool: `concurrency` workers over `items`. */
async function pool(concurrency, items, worker) {
  let next = 0;
  const runners = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (next < items.length) await worker(items[next++]);
  });
  await Promise.all(runners);
}

async function main() {
  const jobs = +process.argv[process.argv.indexOf('--jobs') + 1] || 10;
  for (const d of Object.values(DIRS)) mkdirSync(d, { recursive: true });

  // 1) page through the rows API (max 100 per request); index is the stable row key
  const items = [];
  for (let offset = 0; ; offset += 100) {
    const j = await (await fetchRetry(`${SERVER}/rows?dataset=${DS.dataset}&config=${DS.config}&split=${DS.split}&offset=${offset}&length=100`)).json();
    const page = j.rows || [];
    page.forEach((r, i) => items.push({ index: offset + i, row: r.row }));
    if (!j.num_rows_total || items.length >= j.num_rows_total || !page.length) break;
    console.log(`[rows] ${items.length}/${j.num_rows_total}`);
  }
  console.log(`[rows] total ${items.length}`);

  const metaPath = join(ROOT, 'metadata.jsonl');
  const meta = createWriteStream(metaPath, { flags: items.length ? 'w' : 'a' });

  let done = 0, failed = 0;
  await pool(jobs, items, async ({ index, row }) => {
    const id = `${String(index).padStart(5, '0')}-${row.coco_details?.id ?? 'x'}`;
    const files = {};
    try {
      for (const [dir, field] of Object.entries(FIELDS)) {
        const src = row[field]?.src;
        if (!src) continue;
        const file = `${id}.${ext(await (await fetchRetry(src, 2)).arrayBuffer())}`;
        files[dir] = file;
        if (!existsSync(join(DIRS[dir], file))) {
          const buf = Buffer.from(await (await fetchRetry(src)).arrayBuffer());
          await pipeline(Readable.from(buf), createWriteStream(join(DIRS[dir], file)));
        }
      }
    } catch (e) {
      failed++;
      console.warn(`[fail] ${id}: ${e.message}`);
      return;
    }
    meta.write(JSON.stringify({
      row_index: index, coco_image_id: row.coco_details?.id ?? null, concept: row.concept,
      coco_caption: row.coco_caption, inpaint_caption: row.inpaint_caption,
      files, coco_details: row.coco_details, inpaint_details: row.inpaint_details,
      mask_details: row.mask_details, worker: row.worker,
    }) + '\n');
    if (++done % 50 === 0) console.log(`[dl] ${done}/${items.length}`);
  });
  meta.end();
  console.log(`[done] ${done} rows written to metadata.jsonl, ${failed} failed`);
  if (failed) process.exit(1);
}

main().catch(e => { console.error(e); process.exit(1); });
