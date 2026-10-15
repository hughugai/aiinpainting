/** Eval worker: runs one (algo, image, mask) case per message and reports metrics. */
import { parentPort } from 'node:worker_threads';
import { readFileSync } from 'node:fs';
import { decodeImage, decodeMaskPng } from './lib/imageio.mjs';
import { psnr, ssim } from './lib/metrics.mjs';
import { ALGOS, warmup } from './lib/algos.mjs';

await warmup();

parentPort.on('message', async task => {
  try {
    const img = decodeImage(readFileSync(task.imagePath));
    const { mask } = decodeMaskPng(readFileSync(task.maskFile));
    if (mask.length !== img.w * img.h) throw new Error('Mask/image size mismatch');
    const t0 = performance.now();
    const out = await ALGOS[task.algo].run(img.rgb, mask, img.w, img.h);
    const ms = Math.round(performance.now() - t0);
    for (let i = 0; i < mask.length; i++)
      if (!mask[i] && (out[i * 3] !== img.rgb[i * 3] || out[i * 3 + 1] !== img.rgb[i * 3 + 1] || out[i * 3 + 2] !== img.rgb[i * 3 + 2]))
        throw new Error(`Known pixel changed at index ${i}`);
    const result = { id: task.id, ok: true, psnr: +psnr(out, img.rgb).toFixed(3), ssim: +ssim(out, img.rgb, img.w, img.h).toFixed(5), ms };
    if (task.keepOutput) result.outputRgb = Buffer.from(out);
    parentPort.postMessage(result);
  } catch (e) {
    parentPort.postMessage({ id: task.id, ok: false, error: String(e.message || e) });
  }
});
