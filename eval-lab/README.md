# Eval lab

Deterministic benchmark harness for comparing inpainting algorithms. Companion regression tests live in [`tests/eval-lab.test.mjs`](../tests/README.md).

- **Datasets**: 50 images sampled from Hugging Face at fixed offsets (Places365-val, CelebA-HQ, InpaintCOCO) + 4 mask tiers per image (small / thin / large / object). `manifest.json` anchors the samples by sha256 for reproducibility; downloaded images are gitignored (licenses).
- **Metrics**: PSNR / SSIM computed on masked regions (`lib/metrics.mjs`), JSON + Markdown reports with baseline diffing into `reports/`.
- **Algorithms under test** (`lib/algos.mjs`): `feature` / `feature-plain` / `shifted` / `criminisi` from `algorithms/js/core.ts` (compiled with tsc, cached in `.cache/`), reference `algorithms/js/inpaint.js`, classical baselines from `algorithms/js/classical.js`, and ONNX deep baselines (LaMa / MI-GAN weights from `models/`).

## Usage

```sh
npm run eval:prepare   # download the 50-image benchmark from Hugging Face + generate masks
npm run eval:run       # run all algorithms over the benchmark, write eval-lab/reports/
```

Use it as the regression gate when optimizing algorithms: run before/after a change and diff the reports.
