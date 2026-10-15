# aiinpainting research

Open research repository for image inpainting algorithms — reference implementations, paper reimplementations, and reproducible tests. Companion project of **[aiinpainting.app](https://aiinpainting.app)** — a free online AI inpainting tool for object removal, watermark & text removal, and photo repair.

## What's inside

| Directory | Description |
|---|---|
| [`paper-lab/`](paper-lab/) | Independent JS + Python implementations of classic and neural inpainting methods — Efros–Leung (1999), Bertalmío (2000), Criminisi (2003), Telea (2004), PatchMatch (2009), feature-priority exemplar (2016), plus LaMa / MI-GAN pretrained ONNX inference — with cross-language verification tests |
| [`feature-inpainting/`](feature-inpainting/) | Verified reference implementation of the feature-priority block-matching inpainting algorithm (李旭峰 et al. 2016), with browser demo, Python/JS cores, and example images |
| [`eval-lab/`](eval-lab/) | Deterministic benchmark harness: 50 images sampled from Hugging Face datasets (Places365-val, CelebA-HQ, InpaintCOCO), 4 mask tiers per image, PSNR/SSIM metrics, JSON + Markdown reports with baseline diffing |
| [`tests/`](tests/) | Regression tests, including texture-aware source selection and eval-lab metrics/mask generators |

## Quick start

```sh
npm install          # also copies the pinned ONNX Runtime WASM into onnx-runtime/
npm test             # Node regression tests (source selection, eval metrics, masks)
npm run test:papers  # cross-language (JS/Python) verification of paper-lab methods
npm run verify:feature  # verification suite for the feature-priority reference core
npm run eval:prepare # download the 50-image benchmark from Hugging Face + generate masks
npm run eval:run     # run all algorithms over the benchmark, write eval-lab/reports/
```

Python methods need NumPy + Pillow: `pip install -r paper-lab/python/requirements.txt`.

## Scope & honesty

This is a selected collection of algorithms for study and comparison — not a complete history of inpainting, and no paper-benchmark parity is claimed. Neural methods run pretrained ONNX graphs (architecture not reimplemented); model weights keep their upstream licenses and are not redistributed here.

## License

MIT — see [LICENSE](LICENSE). Upstream model weights follow their own licenses (see [`paper-lab/README.md`](paper-lab/README.md)).
