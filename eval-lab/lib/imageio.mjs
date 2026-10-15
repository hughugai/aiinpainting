/** Image I/O and resizing for the eval harness. RGB is a flat 3-channel Uint8Array. */
import { PNG } from 'pngjs';
import jpeg from 'jpeg-js';

/** Decode a PNG or JPEG buffer into {rgb,w,h}. */
export function decodeImage(buf) {
  if (buf.length > 8 && buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47) {
    const png = PNG.sync.read(buf);
    return toRgb(png.data, png.width, png.height, 4);
  }
  if (buf.length > 3 && buf[0] === 0xff && buf[1] === 0xd8) {
    const img = jpeg.decode(buf, { useTArray: true, formatAsRGBA: true, maxMemoryUsageInMB: 2048 });
    return toRgb(img.data, img.width, img.height, 4);
  }
  throw new Error('Unsupported image format (expected PNG or JPEG)');
}

function toRgb(data, w, h, ch) {
  const n = w * h, rgb = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    rgb[i * 3] = data[i * ch];
    rgb[i * 3 + 1] = data[i * ch + 1];
    rgb[i * 3 + 2] = data[i * ch + 2];
  }
  return { rgb, w, h };
}

export function encodePng(rgb, w, h) {
  const png = new PNG({ width: w, height: h });
  for (let i = 0; i < w * h; i++) {
    png.data[i * 4] = rgb[i * 3];
    png.data[i * 4 + 1] = rgb[i * 3 + 1];
    png.data[i * 4 + 2] = rgb[i * 3 + 2];
    png.data[i * 4 + 3] = 255;
  }
  return PNG.sync.write(png);
}

/** Binary mask PNG: hole (mask!=0) -> white, known -> black. */
export function encodeMaskPng(mask, w, h) {
  const png = new PNG({ width: w, height: h });
  for (let i = 0; i < w * h; i++) {
    const v = mask[i] ? 255 : 0;
    png.data[i * 4] = png.data[i * 4 + 1] = png.data[i * 4 + 2] = v;
    png.data[i * 4 + 3] = 255;
  }
  return PNG.sync.write(png);
}

/** Binary mask from a PNG or JPEG buffer: luminance > 127 means hole. */
export function decodeMaskPng(buf) {
  const img = decodeImage(buf);
  const n = img.w * img.h, mask = new Uint8Array(n);
  for (let i = 0; i < n; i++)
    mask[i] = (img.rgb[i * 3] + img.rgb[i * 3 + 1] + img.rgb[i * 3 + 2]) / 3 > 127 ? 1 : 0;
  return { mask, w: img.w, h: img.h };
}

/** Center square crop + bilinear resize to size x size. */
export function centerSquareResize(rgb, w, h, size) {
  const s = Math.min(w, h), x0 = (w - s) >> 1, y0 = (h - s) >> 1;
  const out = new Uint8Array(size * size * 3);
  const scale = s / size;
  for (let y = 0; y < size; y++) {
    const fy = Math.min(s - 1, Math.max(0, (y + 0.5) * scale - 0.5));
    const sy = Math.floor(fy), ty = fy - sy, y1 = Math.min(s - 1, sy + 1);
    for (let x = 0; x < size; x++) {
      const fx = Math.min(s - 1, Math.max(0, (x + 0.5) * scale - 0.5));
      const sx = Math.floor(fx), tx = fx - sx, x1 = Math.min(s - 1, sx + 1);
      for (let c = 0; c < 3; c++) {
        const p00 = rgb[((y0 + sy) * w + x0 + sx) * 3 + c], p10 = rgb[((y0 + sy) * w + x0 + x1) * 3 + c];
        const p01 = rgb[((y0 + y1) * w + x0 + sx) * 3 + c], p11 = rgb[((y0 + y1) * w + x0 + x1) * 3 + c];
        out[(y * size + x) * 3 + c] = Math.round(
          p00 * (1 - tx) * (1 - ty) + p10 * tx * (1 - ty) + p01 * (1 - tx) * ty + p11 * tx * ty);
      }
    }
  }
  return { rgb: out, w: size, h: size };
}

/** Nearest-neighbor resize for binary masks, then threshold. */
export function resizeMask(mask, w, h, size) {
  const s = Math.min(w, h), x0 = (w - s) >> 1, y0 = (h - s) >> 1;
  const out = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) {
    const sy = y0 + Math.min(s - 1, Math.floor((y + 0.5) * s / size));
    for (let x = 0; x < size; x++) {
      const sx = x0 + Math.min(s - 1, Math.floor((x + 0.5) * s / size));
      out[y * size + x] = mask[sy * w + sx] ? 1 : 0;
    }
  }
  return { mask: out, w: size, h: size };
}

export function countHole(mask) {
  let c = 0;
  for (let i = 0; i < mask.length; i++) c += mask[i] ? 1 : 0;
  return c;
}
