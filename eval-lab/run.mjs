/** Eval harness: run all algorithms over the dataset and produce JSON + MD reports.
 *
 * Usage:
 *   node eval-lab/run.mjs --label baseline
 *   node eval-lab/run.mjs --label round1 --baseline baseline
 *   node eval-lab/run.mjs --label quick --algos feature,telea --tiers thin --limit 10 --jobs 4
 *
 * Flags:
 *   --label NAME     report name (default: run)
 *   --baseline NAME  report to diff against (from eval-lab/reports/NAME.json)
 *   --algos LIST     comma-separated algorithm ids or 'all' (default all)
 *   --tiers LIST     comma-separated mask tiers or 'all' (default all)
 *   --limit N        only first N images
 *   --jobs N         worker threads (default 4)
 *   --save MODE      worst (default) | all | none  — write output PNGs to eval-lab/work/<label>/
 */
import { Worker } from 'node:worker_threads';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { encodePng } from './lib/imageio.mjs';
import { ALGOS, ALGO_IDS, warmup } from './lib/algos.mjs';

const LAB = dirname(fileURLToPath(import.meta.url));
const DIR = join(LAB, 'datasets'), REPORTS = join(LAB, 'reports'), WORK = join(LAB, 'work');

function args() {
  const a = { label: 'run', algos: 'all', tiers: 'all', limit: 0, jobs: 4, save: 'worst', baseline: null };
  const v = process.argv.slice(2);
  for (let i = 0; i < v.length; i++) {
    if (v[i] === '--label') a.label = v[++i];
    else if (v[i] === '--baseline') a.baseline = v[++i];
    else if (v[i] === '--algos') a.algos = v[++i];
    else if (v[i] === '--tiers') a.tiers = v[++i];
    else if (v[i] === '--limit') a.limit = +v[++i];
    else if (v[i] === '--jobs') a.jobs = +v[++i];
    else if (v[i] === '--save') a.save = v[++i];
    else throw new Error(`Unknown flag ${v[i]}`);
  }
  a.algos = a.algos === 'all' ? ALGO_IDS : a.algos.split(',').map(s => s.trim()).filter(Boolean);
  a.tiers = a.tiers === 'all' ? ['small', 'thin', 'large', 'object'] : a.tiers.split(',').map(s => s.trim()).filter(Boolean);
  for (const algo of a.algos) if (!ALGOS[algo]) throw new Error(`Unknown algo '${algo}' (have: ${ALGO_IDS.join(', ')})`);
  return a;
}

function buildTasks(a) {
  const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
  const images = a.limit ? manifest.images.slice(0, a.limit) : manifest.images;
  const tasks = [];
  for (const img of images)
    for (const tier of a.tiers) {
      const maskFile = join(DIR, 'masks', `${img.id}.${tier}.png`);
      if (!existsSync(maskFile)) continue;
      for (const algo of a.algos)
        tasks.push({
          id: `${algo}|${img.id}|${tier}`, algo, tier, imageId: img.id,
          imagePath: join(DIR, 'images', `${img.id}.png`), maskFile,
          keepOutput: a.save !== 'none',
        });
    }
  return tasks;
}

/** Fixed pool of workers; tasks are dispatched as workers come free. */
function makePool(size, onResult) {
  const workers = [];
  let next = 0, done = 0;
  const submitNext = w => {
    if (next >= tasksAll.length || w.busy) return;
    w.busy = true;
    w.port.postMessage(tasksAll[next++]);
  };
  let tasksAll;
  const finished = [];
  for (let i = 0; i < size; i++) {
    const w = { port: new Worker(join(LAB, 'worker.mjs')), busy: false };
    w.port.on('message', r => {
      w.busy = false;
      onResult(r);
      done++;
      submitNext(w);
      if (done === tasksAll.length) finished.forEach(f => f());
    });
    workers.push(w);
  }
  return {
    run(tasks) {
      tasksAll = tasks;
      workers.forEach(submitNext);
      return new Promise(resolve => finished.push(resolve));
    },
    async dispose() { await Promise.all(workers.map(w => w.port.terminate())); },
  };
}

const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
const median = xs => { if (!xs.length) return 0; const s = [...xs].sort((a, b) => a - b); return s[s.length >> 1]; };
const fmt = (x, d = 2) => x == null ? '—' : x.toFixed(d);

function summarize(results) {
  const by = {};
  for (const r of results) {
    const key = `${r.algo}|${r.tier}`;
    (by[key] ??= { algo: r.algo, tier: r.tier, n: 0, err: 0, psnr: [], ssim: [], ms: [] });
    by[key].n++;
    if (r.ok) { by[key].psnr.push(r.psnr); by[key].ssim.push(r.ssim); by[key].ms.push(r.ms); }
    else by[key].err++;
  }
  for (const k of Object.keys(by)) {
    const g = by[k];
    g.psnrMean = mean(g.psnr); g.psnrMed = median(g.psnr);
    g.ssimMean = mean(g.ssim); g.ssimMed = median(g.ssim);
    g.msMean = mean(g.ms);
    delete g.psnr; delete g.ssim; delete g.ms;
  }
  return Object.values(by).sort((a, b) => a.algo.localeCompare(b.algo) || a.tier.localeCompare(b.tier));
}

function markdown(meta, summary, baseline) {
  const L = [];
  L.push(`# Eval report: ${meta.label}`, '');
  L.push(`- Date: ${meta.finishedAt}`);
  L.push(`- Image size: ${meta.size}px, cases: ${meta.results.length}, jobs: ${meta.jobs}`);
  L.push(`- Algorithms: ${meta.algos.join(', ')}`);
  L.push(`- Tiers: ${meta.tiers.join(', ')}`);
  L.push(`- Dataset: eval-lab/datasets/manifest.json (${meta.images} images, generated ${meta.manifestDate})`, '');

  L.push('## Summary', '');
  L.push('| algorithm | tier | n | err | PSNR mean | PSNR med | SSIM mean | SSIM med | ms mean |');
  L.push('|---|---|---|---|---|---|---|---|---|');
  for (const s of summary)
    L.push(`| ${s.algo} | ${s.tier} | ${s.n} | ${s.err} | ${fmt(s.psnrMean)} | ${fmt(s.psnrMed)} | ${fmt(s.ssimMean, 4)} | ${fmt(s.ssimMed, 4)} | ${fmt(s.msMean, 0)} |`);
  L.push('');

  if (baseline) {
    L.push(`## Delta vs ${baseline.meta.label}`, '');
    L.push('PSNR/SSIM mean difference (positive = better than baseline).', '');
    L.push('| algorithm | tier | n | ΔPSNR mean | ΔSSIM mean | better | worse |');
    L.push('|---|---|---|---|---|---|---|');
    const base = new Map(baseline.results.map(r => [`${r.algo}|${r.imageId}|${r.tier}`, r]));
    for (const s of summary) {
      let dp = 0, ds = 0, better = 0, worse = 0, n = 0;
      for (const r of meta.results) {
        if (r.algo !== s.algo || r.tier !== s.tier || !r.ok) continue;
        const b = base.get(`${r.algo}|${r.imageId}|${r.tier}`);
        if (!b?.ok) continue;
        dp += r.psnr - b.psnr; ds += r.ssim - b.ssim; n++;
        if (r.psnr > b.psnr + 1e-6) better++; else if (r.psnr < b.psnr - 1e-6) worse++;
      }
      if (n) L.push(`| ${s.algo} | ${s.tier} | ${n} | ${fmt(dp / n, 3)} | ${fmt(ds / n, 5)} | ${better} | ${worse} |`);
    }
    L.push('');
  }

  L.push('## Worst cases', '');
  for (const s of summary) {
    const rows = meta.results.filter(r => r.algo === s.algo && r.tier === s.tier && r.ok)
      .sort((a, b) => a.psnr - b.psnr).slice(0, 3);
    if (!rows.length) continue;
    L.push(`**${s.algo} / ${s.tier}**`);
    for (const r of rows)
      L.push(`- ${r.imageId}: PSNR ${fmt(r.psnr)}, SSIM ${fmt(r.ssim, 4)}, ${r.ms}ms`);
    L.push('');
  }
  return L.join('\n');
}

async function main() {
  const a = args();
  const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
  const tasks = buildTasks(a);
  if (!tasks.length) throw new Error('No tasks — did you run eval:prepare?');

  let baseline = null;
  if (a.baseline) {
    const p = a.baseline.endsWith('.json') ? a.baseline : join(REPORTS, `${a.baseline}.json`);
    baseline = JSON.parse(readFileSync(p, 'utf8'));
  }

  await warmup();
  mkdirSync(REPORTS, { recursive: true });
  const results = [];
  const t0 = Date.now();
  const pool = makePool(Math.max(1, Math.min(a.jobs, tasks.length)), r => {
    results.push(r);
    const i = results.length;
    if (i % 25 === 0 || i === tasks.length)
      console.log(`[eval] ${i}/${tasks.length} (${Math.round((Date.now() - t0) / 1000)}s)`);
  });
  await pool.run(tasks);
  await pool.dispose();

  const records = results.map(r => {
    const [algo, imageId, tier] = r.id.split('|');
    return { algo, imageId, tier, ok: r.ok, psnr: r.psnr, ssim: r.ssim, ms: r.ms, error: r.error };
  });
  const summary = summarize(records);

  const dir = a.save === 'all' ? join(WORK, a.label) : a.save === 'worst' ? join(WORK, a.label) : null;
  if (dir) {
    const want = new Set();
    if (a.save === 'worst')
      for (const s of summary)
        records.filter(x => x.algo === s.algo && x.tier === s.tier && x.ok)
          .sort((x, y) => x.psnr - y.psnr).slice(0, 3)
          .forEach(x => want.add(`${x.algo}|${x.imageId}|${x.tier}`));
    for (const r of results) {
      if (!r.ok) { console.warn(`[error] ${r.id}: ${r.error}`); continue; }
      if (a.save === 'all' || want.has(r.id)) {
        const [algo, imageId, tier] = r.id.split('|');
        const d = join(dir, algo, tier);
        mkdirSync(d, { recursive: true });
        writeFileSync(join(d, `${imageId}.png`), encodePng(r.outputRgb, manifest.size, manifest.size));
      }
    }
  }

  const meta = {
    label: a.label, startedAt: new Date(t0).toISOString(), finishedAt: new Date().toISOString(),
    size: manifest.size, images: manifest.images.length, manifestDate: manifest.generatedAt,
    jobs: a.jobs, algos: a.algos, tiers: a.tiers, results: records,
  };
  writeFileSync(join(REPORTS, `${a.label}.json`), JSON.stringify({ meta, summary }, null, 1));
  writeFileSync(join(REPORTS, `${a.label}.md`), markdown(meta, summary, baseline));
  console.log(`[done] reports/${a.label}.json + .md`);
}

main().catch(e => { console.error(e); process.exit(1); });
