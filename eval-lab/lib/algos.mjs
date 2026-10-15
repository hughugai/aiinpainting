/** Algorithm registry for the eval harness. Every entry: run(rgb,mask,w,h) -> Uint8Array.
 * feature        core.ts, texture-aware source selection on  — current website algorithm
 * feature-plain  core.ts, texture-aware off                  — quantifies the new feature
 * shifted        core.ts ablation (priority without feature bonus)
 * criminisi      core.ts Criminisi-mode baseline
 * telea / bertalmio / efros-leung / patchmatch — paper-lab classical baselines
 */
import { execFileSync } from 'node:child_process';
import { mkdirSync, statSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';

const LAB = dirname(fileURLToPath(import.meta.url)); // eval-lab/lib
const ROOT = join(LAB, '..', '..');

let coreInpaint = null;
let refInpaint = null;

/** Compile feature-inpainting/core.ts once (tsc) and cache it in eval-lab/.cache. */
function loadCore() {
  if (coreInpaint) return coreInpaint;
  const outDir = join(LAB, '..', '.cache');
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, 'package.json'), '{"type":"commonjs"}');
  const src = join(ROOT, 'feature-inpainting', 'core.ts');
  const dst = join(outDir, 'core.js');
  let stale = true;
  try { stale = statSync(src).mtimeMs > statSync(dst).mtimeMs; } catch { /* not built yet */ }
  if (stale)
    execFileSync(join(ROOT, 'node_modules', '.bin', 'tsc'),
      [src, '--outDir', outDir, '--module', 'commonjs', '--target', 'es2020', '--skipLibCheck'],
      { stdio: 'pipe' });
  coreInpaint = createRequire(import.meta.url)(dst).inpaint;
  return coreInpaint;
}

function loadRef() {
  if (refInpaint) return refInpaint;
  refInpaint = import(join(ROOT, 'feature-inpainting', 'inpaint.js')).then(m => m.inpaint);
  return refInpaint;
}

async function classical() {
  return import(join(ROOT, 'paper-lab', 'js', 'classical.js'));
}

export const ALGOS = {
  'feature': {
    label: 'feature (core.ts, texture-aware on)',
    run: async (rgb, mask, w, h) => loadCore()(rgb, mask, w, h, { method: 'feature', sourceTextureWeight: 1 }).image,
  },
  'feature-plain': {
    label: 'feature (core.ts, texture-aware off)',
    run: async (rgb, mask, w, h) => loadCore()(rgb, mask, w, h, { method: 'feature' }).image,
  },
  'feature-ref': {
    label: 'feature (inpaint.js reference)',
    run: async (rgb, mask, w, h) => (await loadRef())(rgb, mask, w, h, { method: 'feature' }),
  },
  'shifted': {
    label: 'shifted (priority ablation)',
    run: async (rgb, mask, w, h) => loadCore()(rgb, mask, w, h, { method: 'shifted' }).image,
  },
  'criminisi': {
    label: 'criminisi (priority baseline)',
    run: async (rgb, mask, w, h) => loadCore()(rgb, mask, w, h, { method: 'criminisi' }).image,
  },
  'telea': {
    label: 'Telea 2004 FMM',
    run: async (rgb, mask, w, h) => (await classical()).telea(rgb, mask, w, h, { radius: 3 }),
  },
  'bertalmio': {
    label: 'Bertalmio 2000 isophote transport',
    run: async (rgb, mask, w, h) => (await classical()).inpaint('bertalmio', rgb, mask, w, h),
  },
  'efros-leung': {
    label: 'Efros & Leung 1999',
    run: async (rgb, mask, w, h) => (await classical()).inpaint('efros-leung', rgb, mask, w, h),
  },
  'patchmatch': {
    label: 'PatchMatch (simplified Barnes 2009)',
    run: async (rgb, mask, w, h) => (await classical()).inpaint('patchmatch', rgb, mask, w, h),
  },
};

export const ALGO_IDS = Object.keys(ALGOS);

/** Compile core.ts up front so worker threads never race on tsc output. */
export async function warmup() {
  loadCore();
  await loadRef();
}
