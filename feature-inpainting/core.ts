/**
 * Feature-priority patch matching inpainting core.
 *
 * Ported from feature-inpainting/inpaint.js (verified reference implementation
 * of 李旭峰 et al. 2016, 《特征优先块匹配图像修复算法》). The algorithm logic is
 * defaults preserve the reference scoring. The interactive tool additionally
 * enables texture-aware source selection for removal tasks.
 *
 * RGB is a flat 3-channel Uint8Array (length = width * height * 3).
 * Mask is length = width * height; non-zero means "missing / to fill".
 */

export type InpaintMethod = "feature" | "shifted" | "criminisi";
export type ConfidenceUpdate = "raw" | "refined";

export interface InpaintStep {
  iteration: number;
  center: [number, number];
  source: [number, number];
  confidence: number;
  data: number;
  feature: number;
  refined: number;
  priority: number;
  ssd: number;
  filled: number;
}

export interface InpaintProgress {
  remaining: number;
  step: InpaintStep;
  /**
   * Live arrays for the in-flight fill state (RGB + hole flags).
   * Only valid during the callback — copy (slice) before keeping them.
   */
  image: Uint8Array;
  hole: Uint8Array;
}

export interface InpaintOptions {
  patch?: number;
  lambda?: number;
  featureWeight?: number;
  method?: InpaintMethod;
  gradientScale?: number;
  confidenceUpdate?: ConfidenceUpdate;
  maxIterations?: number;
  /** Penalize excess texture inside the pixels that will actually be copied. */
  sourceTextureWeight?: number;
  onProgress?: (progress: InpaintProgress) => void;
}

export interface InpaintResult {
  image: Uint8Array;
  trace: InpaintStep[];
}

interface Best {
  priority: number;
  y: number;
  x: number;
  y0: number;
  y1: number;
  x0: number;
  x1: number;
  c: number;
  data: number;
  feature: number;
  refined: number;
}

export function inpaint(
  rgb: Uint8Array,
  mask: ArrayLike<number>,
  width: number,
  height: number,
  options: InpaintOptions = {},
): InpaintResult {
  const {
    patch = 9,
    lambda = 0.35,
    featureWeight = 1,
    method = "feature",
    gradientScale = 255,
    confidenceUpdate = "raw",
    maxIterations = mask.length + 1,
    sourceTextureWeight = 0,
    onProgress = () => {},
  } = options;

  const w = width;
  const h = height;
  const n = w * h;
  const r = Math.floor(patch / 2);

  if (
    !Number.isInteger(w) ||
    !Number.isInteger(h) ||
    w < 1 ||
    h < 1 ||
    rgb.length !== n * 3 ||
    mask.length !== n
  )
    throw new Error("Invalid dimensions or array lengths");
  if (!Number.isInteger(patch) || patch < 3 || patch % 2 !== 1 || patch > Math.min(w, h))
    throw new Error("Invalid patch size");
  if (
    ![lambda, featureWeight, gradientScale, sourceTextureWeight].every(Number.isFinite) ||
    sourceTextureWeight < 0 ||
    gradientScale <= 0 ||
    !Number.isInteger(maxIterations) ||
    maxIterations < 0
  )
    throw new Error("Invalid numeric options");
  if (
    !["feature", "shifted", "criminisi"].includes(method) ||
    !["raw", "refined"].includes(confidenceUpdate)
  )
    throw new Error("Invalid mode");

  const image = Uint8Array.from(rgb);
  const hole = Uint8Array.from(mask, (v) => (v ? 1 : 0));
  const initial = hole.slice();
  const confidence = Float64Array.from(hole, (v) => 1 - v);
  const trace: InpaintStep[] = [];
  const cache = new Map<string, [number, number][]>();
  let remaining = hole.reduce((a, b) => a + b, 0);

  const neighbors = (y: number, x: number): [number, number][] =>
    (
      [
        [y - 1, x],
        [y + 1, x],
        [y, x - 1],
        [y, x + 1],
      ] as [number, number][]
    ).filter(([yy, xx]) => yy >= 0 && yy < h && xx >= 0 && xx < w);

  function sources(ph: number, pw: number): [number, number][] {
    const key = ph + "," + pw;
    if (!cache.has(key)) {
      const list: [number, number][] = [];
      for (let y = 0; y <= h - ph; y++)
        for (let x = 0; x <= w - pw; x++) {
          let valid = true;
          for (let dy = 0; dy < ph && valid; dy++)
            for (let dx = 0; dx < pw; dx++)
              if (initial[(y + dy) * w + x + dx]) {
                valid = false;
                break;
              }
          if (valid) list.push([y, x]);
        }
      cache.set(key, list);
    }
    return cache.get(key) as [number, number][];
  }

  for (let iteration = 0; iteration < maxIterations && remaining; iteration++) {
    const gray = new Float64Array(n);
    const gx = new Float64Array(n);
    const gy = new Float64Array(n);
    const mag = new Float64Array(n);
    for (let i = 0; i < n; i++)
      gray[i] = image[3 * i] * 0.299 + image[3 * i + 1] * 0.587 + image[3 * i + 2] * 0.114;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (hole[i]) continue;
        const l = x > 0 && !hole[i - 1];
        const rr = x + 1 < w && !hole[i + 1];
        const u = y > 0 && !hole[i - w];
        const d = y + 1 < h && !hole[i + w];
        gx[i] = l && rr ? (gray[i + 1] - gray[i - 1]) / 2 : rr ? gray[i + 1] - gray[i] : l ? gray[i] - gray[i - 1] : 0;
        gy[i] = u && d ? (gray[i + w] - gray[i - w]) / 2 : d ? gray[i + w] - gray[i] : u ? gray[i] - gray[i - w] : 0;
        mag[i] = Math.hypot(gx[i], gy[i]);
      }

    let best: Best | null = null;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const nb = neighbors(y, x);
        if (hole[i] || !nb.some(([yy, xx]) => hole[yy * w + xx])) continue;
        const y0 = Math.max(0, y - r);
        const y1 = Math.min(h, y + r + 1);
        const x0 = Math.max(0, x - r);
        const x1 = Math.min(w, x + r + 1);
        let c = 0;
        let maxMag = -1;
        let maxI = 0;
        for (let yy = y0; yy < y1; yy++)
          for (let xx = x0; xx < x1; xx++) {
            const j = yy * w + xx;
            c += confidence[j];
            if (mag[j] > maxMag) {
              maxMag = mag[j];
              maxI = j;
            }
          }
        c /= (y1 - y0) * (x1 - x0);
        let nx = ((x + 1 < w ? hole[i + 1] : 0) - (x > 0 ? hole[i - 1] : 0)) / 2;
        let ny = ((y + 1 < h ? hole[i + w] : 0) - (y > 0 ? hole[i - w] : 0)) / 2;
        let norm = Math.hypot(nx, ny);
        if (norm === 0) {
          const found = nb.find(([yy, xx]) => hole[yy * w + xx]) as [number, number];
          const [yy, xx] = found;
          nx = xx - x;
          ny = yy - y;
          norm = Math.hypot(nx, ny);
        }
        nx /= norm;
        ny /= norm;
        const data = Math.abs(-gy[maxI] * nx + gx[maxI] * ny) / 255;
        const feature = mag[i] / gradientScale;
        let refined = method === "criminisi" ? c : c - lambda;
        if (method === "feature") refined += featureWeight * feature;
        const priority = refined * data;
        if (!best || priority > best.priority)
          best = { priority, y, x, y0, y1, x0, x1, c, data, feature, refined };
      }

    if (!best) throw new Error("No known fill front (mask may cover entire image)");
    const { y, x, y0, y1, x0, x1, c, data, feature, refined } = best;
    const ph = y1 - y0;
    const pw = x1 - x0;
    const known: [number, number][] = [];
    for (let dy = 0; dy < ph; dy++)
      for (let dx = 0; dx < pw; dx++)
        if (!hole[(y0 + dy) * w + x0 + dx]) known.push([dy, dx]);

    let minSSD = Infinity;
    let minScore = Infinity;
    let minDistance = Infinity;
    let source: [number, number] | null = null;
    // Estimate background variation from trusted context, per channel. A source
    // must also look plausible in its hidden interior, not just along its rim.
    const mean = [0, 0, 0];
    for (const [dy, dx] of known)
      for (let ch = 0; ch < 3; ch++)
        mean[ch] += image[((y0 + dy) * w + x0 + dx) * 3 + ch] / known.length;
    let contextVariance = 0;
    for (const [dy, dx] of known)
      for (let ch = 0; ch < 3; ch++) {
        const delta = image[((y0 + dy) * w + x0 + dx) * 3 + ch] - mean[ch];
        contextVariance += delta * delta / (known.length * 3);
      }
    for (const [sy, sx] of sources(ph, pw)) {
      let ssd = 0;
      for (const [dy, dx] of known) {
        const ti = ((y0 + dy) * w + x0 + dx) * 3;
        const si = ((sy + dy) * w + sx + dx) * 3;
        for (let ch = 0; ch < 3; ch++) {
          const delta = image[ti + ch] - image[si + ch];
          ssd += delta * delta;
        }
        if (ssd / (known.length * 3) > minScore) break;
      }
      let excessTexture = 0;
      let missing = 0;
      if (sourceTextureWeight > 0)
        for (let dy = 0; dy < ph; dy++)
          for (let dx = 0; dx < pw; dx++) {
            if (!hole[(y0 + dy) * w + x0 + dx]) continue;
            missing++;
            for (let ch = 0; ch < 3; ch++) {
              const delta = image[((sy + dy) * w + sx + dx) * 3 + ch] - mean[ch];
              excessTexture += delta * delta;
            }
          }
      const score = ssd / (known.length * 3) + sourceTextureWeight *
        Math.max(0, excessTexture / Math.max(1, missing * 3) - contextVariance);
      const distance = (sy - y0) ** 2 + (sx - x0) ** 2;
      if (score < minScore || (sourceTextureWeight > 0 && score === minScore && distance < minDistance)) {
        minSSD = ssd;
        minScore = score;
        minDistance = distance;
        source = [sy, sx];
      }
    }
    if (!source) throw new Error("No fully known source patch; reduce patch size");

    const [sy, sx] = source;
    let filled = 0;
    for (let dy = 0; dy < ph; dy++)
      for (let dx = 0; dx < pw; dx++) {
        const ti = (y0 + dy) * w + x0 + dx;
        if (!hole[ti]) continue;
        const si = (sy + dy) * w + sx + dx;
        for (let ch = 0; ch < 3; ch++) image[3 * ti + ch] = image[3 * si + ch];
        confidence[ti] = confidenceUpdate === "raw" ? c : refined;
        hole[ti] = 0;
        filled++;
      }

    remaining -= filled;
    trace.push({
      iteration,
      center: [x, y],
      source: [sx, sy],
      confidence: c,
      data,
      feature,
      refined,
      priority: best.priority,
      ssd: minSSD,
      filled,
    });
    onProgress({ remaining, step: trace[trace.length - 1] as InpaintStep, image, hole });
  }

  if (remaining) throw new Error("Iteration limit reached before completion");
  return { image, trace };
}
