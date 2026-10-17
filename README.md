# aiinpainting research

Open research repository for image inpainting algorithms — papers, reference implementations, reproducible tests, and a deterministic benchmark. Companion project of **[aiinpainting.app](https://aiinpainting.app)** — a free online AI inpainting tool for object removal, watermark & text removal, and photo repair.

## What's inside

| Directory | Description |
|---|---|
| [`papers/`](papers/README.md) | Source papers behind every implementation — links and one-line notes (Efros–Leung 1999, Bertalmío 2000, Criminisi 2003, Telea 2004, PatchMatch 2009, 李旭峰 2016, LaMa, MI-GAN, DeepFillv2, HiFill) |
| [`algorithms/`](algorithms/README.md) | Core algorithms in JS + Python: classic methods (Efros–Leung, Bertalmío, Criminisi, Telea, PatchMatch), the feature-priority exemplar core (李旭峰 et al. 2016) with browser demo, and LaMa / MI-GAN pretrained ONNX inference |
| [`tests/`](tests/README.md) | Cross-language verification (JS/Python parity, invariants, determinism) and Node regression tests |
| [`eval-lab/`](eval-lab/README.md) | Deterministic benchmark: 50 Hugging Face images × 4 mask tiers, PSNR/SSIM, JSON + Markdown reports with baseline diffing |
| [`models/`](models/) | Local pretrained ONNX weights (gitignored — download separately; see [`papers/README.md`](papers/README.md)) |

## Quick start

```sh
npm install          # also copies the pinned ONNX Runtime WASM into onnx-runtime/
npm test             # Node regression tests (source selection, eval metrics, masks)
npm run test:papers  # cross-language (JS/Python) verification of the classic methods
npm run verify:feature  # consistency suite for the 2016 feature-priority reference core
npm run eval:prepare # download the 50-image benchmark from Hugging Face + generate masks
npm run eval:run     # run all algorithms over the benchmark, write eval-lab/reports/
```

Python methods need NumPy + Pillow: `pip install -r algorithms/python/requirements.txt`.

## Scope & honesty

This is a selected collection of algorithms for study and comparison — not a complete history of inpainting, and no paper-benchmark parity is claimed. Neural methods run pretrained ONNX graphs (architecture not reimplemented); model weights keep their upstream licenses and are not redistributed here.

## License

MIT — see [LICENSE](LICENSE). Upstream model weights follow their own licenses (see [`algorithms/README.md`](algorithms/README.md)).
