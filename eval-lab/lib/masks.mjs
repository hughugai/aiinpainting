/** Deterministic mask generation for the eval harness. mask!=0 means hole. */

/** mulberry32 — small, fast, deterministic. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function countHole(mask) {
  let c = 0;
  for (let i = 0; i < mask.length; i++) c += mask[i] ? 1 : 0;
  return c;
}

function stampDisc(mask, w, h, cx, cy, r) {
  let added = 0;
  for (let y = Math.max(0, cy - r); y <= Math.min(h - 1, cy + r); y++)
    for (let x = Math.max(0, cx - r); x <= Math.min(w - 1, cx + r); x++) {
      const dx = x - cx, dy = y - cy;
      if (dx * dx + dy * dy <= r * r && !mask[y * w + x]) { mask[y * w + x] = 1; added++; }
    }
  return added;
}

function stampRect(mask, w, h, x0, y0, x1, y1) {
  let added = 0;
  for (let y = Math.max(0, y0); y <= Math.min(h - 1, y1); y++)
    for (let x = Math.max(0, x0); x <= Math.min(w - 1, x1); x++)
      if (!mask[y * w + x]) { mask[y * w + x] = 1; added++; }
  return added;
}

/** Random-walk brush stroke along a direction with jitter. */
function stampStroke(mask, w, h, rng, radius, angle, length) {
  let filled = 0;
  let x = radius + 1 + Math.floor(rng() * (w - 2 * radius - 2));
  let y = radius + 1 + Math.floor(rng() * (h - 2 * radius - 2));
  const steps = Math.max(4, Math.floor(length));
  for (let s = 0; s < steps; s++) {
    filled += stampDisc(mask, w, h, Math.round(x), Math.round(y), radius);
    const jit = (rng() - .5) * .6;
    x += Math.cos(angle + jit) * (radius + 1);
    y += Math.sin(angle + jit) * (radius + 1);
    if (x < -r2(radius) || x > w + r2(radius) || y < -r2(radius) || y > h + r2(radius)) break;
  }
  return filled;
  function r2(v) { return v * 3; }
}

const TIERS = {
  /** 6–18% of pixels: 1-2 rects + brush strokes — object removal, small blemishes. */
  small(w, h, rng) {
    const mask = new Uint8Array(w * h), n = w * h;
    const target = n * (.06 + rng() * .12);
    let filled = 0;
    for (let s = 0; s < 2 && filled < target * .9; s++) {
      const bw = 15 + Math.floor(rng() * 30), bh = 12 + Math.floor(rng() * 28);
      const x0 = Math.floor(rng() * (w - bw)), y0 = Math.floor(rng() * (h - bh));
      filled += stampRect(mask, w, h, x0, y0, x0 + bw - 1, y0 + bh - 1);
    }
    for (let s = 0; s < 3 && filled < target * .95; s++)
      filled += stampStroke(mask, w, h, rng, 4 + Math.floor(rng() * 5), rng() * Math.PI * 2, 20 + rng() * 40);
    return { mask, frac: countHole(mask) / n };
  },
  /** ~0.5–3%: 1-3 thin strokes (wires, cracks, watermark lines). */
  thin(w, h, rng) {
    const mask = new Uint8Array(w * h), n = w * h;
    const target = n * (.005 + rng() * .025);
    let filled = 0;
    const base = rng() < .7 ? (rng() < .5 ? 0 : Math.PI / 2) : rng() * Math.PI;
    for (let s = 0; s < 3 && filled < target * .95; s++)
      filled += stampStroke(mask, w, h, rng, 1 + Math.floor(rng() * 2), base + (rng() - .5) * .8, Math.min(w, h) * (.4 + rng() * .7));
    return { mask, frac: countHole(mask) / n };
  },
  /** 40–50%: 2 large rects + wide strokes — large missing block stress test. */
  large(w, h, rng) {
    const mask = new Uint8Array(w * h), n = w * h;
    const target = n * (.40 + rng() * .10), cap = n * .52;
    let filled = 0;
    for (let s = 0; s < 2; s++) {
      const bw = Math.floor(w * (.3 + rng() * .25)), bh = Math.floor(h * (.3 + rng() * .25));
      const x0 = Math.floor(rng() * (w - bw)), y0 = Math.floor(rng() * (h - bh));
      filled += stampRect(mask, w, h, x0, y0, Math.min(w - 1, x0 + bw - 1), Math.min(h - 1, y0 + bh - 1));
    }
    for (let s = 0; s < 8 && filled < target; s++) {
      if (filled > cap) break;
      filled += stampStroke(mask, w, h, rng, 5 + Math.floor(rng() * 6), rng() * Math.PI * 2, 30 + rng() * 60);
    }
    return { mask, frac: countHole(mask) / n };
  },
};

export const TIER_NAMES = Object.keys(TIERS);

/** Generate a mask of the given tier; retries with seed+i until the hole area
 * lands inside the tier's sane band, or the retry budget runs out. */
export function makeMask(tier, w, h, seed) {
  if (!TIERS[tier]) throw new Error('Unknown mask tier: ' + tier);
  for (let i = 0; i < 8; i++) {
    const rng = mulberry32((seed + i * 7919) >>> 0);
    const out = TIERS[tier](w, h, rng);
    const band = tier === 'small' ? [.05, .20] : tier === 'thin' ? [.003, .035] : [.38, .54];
    if (out.frac >= band[0] && out.frac <= band[1]) return { ...out, seed: seed + i * 7919 };
  }
  const rng = mulberry32(seed >>> 0);
  return { ...TIERS[tier](w, h, rng), seed };
}

/** True if at least one fully-known patch x patch window exists. */
export function hasCleanPatch(mask, w, h, patch = 9) {
  const r = patch >> 1;
  for (let y = r; y + r < h; y++)
    for (let x = r; x + r < w; x++) {
      let clean = true;
      for (let dy = -r; dy <= r && clean; dy++)
        for (let dx = -r; dx <= r; dx++)
          if (mask[(y + dy) * w + x + dx]) { clean = false; break; }
      if (clean) return true;
    }
  return false;
}
