/** Algorithm registry for the eval harness. Every entry: run(rgb,mask,w,h) -> Uint8Array.
 * feature        core.ts, texture-aware source selection on  — current website algorithm
 * feature-plain  core.ts, texture-aware off                  — quantifies the new feature
 * shifted        core.ts ablation (priority without feature bonus)
 * criminisi      core.ts Criminisi-mode baseline
 * telea / bertalmio / efros-leung / patchmatch — algorithms/js classical baselines
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

/** Compile algorithms/js/core.ts once (tsc) and cache it in eval-lab/.cache. */
function loadCore() {
  if (coreInpaint) return coreInpaint;
  const outDir = join(LAB, '..', '.cache');
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, 'package.json'), '{"type":"commonjs"}');
  const src = join(ROOT, 'algorithms', 'js', 'core.ts');
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
  refInpaint = import(join(ROOT, 'algorithms', 'js', 'inpaint.js')).then(m => m.inpaint);
  return refInpaint;
}

async function classical() {
  return import(join(ROOT, 'algorithms', 'js', 'classical.js'));
}

/** ONNX deep baselines (LaMa / MI-GAN) via algorithms/js/onnx.js + onnxruntime-web WASM.
 * Weights live in models/*.onnx (gitignored); sessions are lazy singletons. */
const ONNX_FILES = { lama: 'lama_fp32.onnx', migan: 'migan_pipeline_v2.onnx' };
const onnxRunners = {};
async function getOnnx(method) {
  if (!onnxRunners[method]) {
    const ort = await import('onnxruntime-web/wasm');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = join(ROOT, 'node_modules', 'onnxruntime-web', 'dist') + '/';
    const { createOnnxInpainter } = await import(join(ROOT, 'algorithms', 'js', 'onnx.js'));
    const { readFile } = await import('node:fs/promises');
    const model = new Uint8Array(await readFile(join(ROOT, 'models', ONNX_FILES[method])));
    onnxRunners[method] = await createOnnxInpainter(ort, method, model);
  }
  return onnxRunners[method];
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
  'lama': {
    label: 'LaMa (ONNX fp32, 512 pad, Carve export)',
    run: async (rgb, mask, w, h) => (await getOnnx('lama')).run(rgb, mask, w, h),
  },
  'migan': {
    label: 'MI-GAN (ONNX pipeline v2)',
    run: async (rgb, mask, w, h) => (await getOnnx('migan')).run(rgb, mask, w, h),
  },
};

export const ALGO_IDS = Object.keys(ALGOS);

/** Compile core.ts up front so worker threads never race on tsc output. */
export async function warmup() {
  loadCore();
  await loadRef();
}
