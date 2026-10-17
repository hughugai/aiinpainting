# Tests

All regression, cross-language and verification suites for this repository. Run from the repo root.

## Node regression tests (`npm test` → `node --test tests/*.test.mjs`)

| File | What it covers |
|---|---|
| `inpaint-source-selection.test.mjs` | Texture-aware source selection of `algorithms/js/core.ts` (compiles it with tsc): rejects a matching source rim with hidden text inside |
| `eval-lab.test.mjs` | `eval-lab` metric (PSNR/SSIM) and mask generator determinism/correctness |

## Cross-language verification (`npm run test:papers` → `verify-papers.py`)

Runs 30 cases (center/edge/crack/disconnected/empty holes) through `run-js.mjs` (JS, `algorithms/js/classical.js` + `onnx.js`) and the independent Python counterparts in `algorithms/python/`. Asserts JS/Python max error ≤ 1 byte, known-pixel preservation, masked-input independence, deterministic seeds, invalid-mask rejection, and ONNX tensor polarity/layout contracts.

## 2016 feature-core verification (`npm run verify:feature` → `verify-feature.py`)

Generates `verification-feature.json` and verifies Python (`algorithms/python/inpaint.py`) vs JS (`algorithms/js/inpaint.js`) byte equality, fill centers and sources, trace tolerance 1e-9, invariance to masked-region RGB, across texture/flat/structure-edge/border/crack/empty-mask/negative-priority/invalid-input cases. `run-feature-json.mjs` is the Node JSON bridge. MSE/PSNR in the JSON are computed on the mask region only; MSE=0 → PSNR null (infinite).

## Optional real-model parity (`verify-onnx.py` + `onnx-real.mjs`)

```sh
python3 tests/verify-onnx.py migan /absolute/path/migan_pipeline_v2.onnx
python3 tests/verify-onnx.py lama /absolute/path/lama_fp32.onnx
```

CPU vs WASM numerical agreement (≤ 3 byte difference) on a deterministic gradient image; test RGB output goes to the system temp directory. Weights are not bundled — download to your own paths first.

## Benchmark harness

The deterministic 50-image benchmark lives separately in [`eval-lab/`](../eval-lab/README.md) (`npm run eval:prepare` / `npm run eval:run`).
