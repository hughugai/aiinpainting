# Inpainting Paper Lab

Independent reference algorithms in JavaScript and Python, plus browser/Python ONNX inference.
Branch: `codex/inpainting-paper-lab`. No remote deployment is implied.

## Website

```sh
npm install
npm run dev
```

Open `/paper-lab`; each method has its own `/paper-lab/<method>` page. Choose an image, paint the hole, run, compare and download PNG. Work runs in a cancellable Worker. Reference classical computation is capped at 128 px on the longest edge, neural computation at 512 px; downloads use that working resolution. Existing website tools retain their 512 px default. Models are never substituted silently.

## Coverage and fidelity

| Method / page slug | JS + Python implementation | Scope |
|---|---|---|
| `efros-leung` (1999) | Gaussian weighted conditional neighborhood sampling | Hole completion adaptation, fixed fully known source windows, strongest-context frontier, seeded LCG |
| `bertalmio` (2000) | Isophote transport of Laplacian | **Educational approximation**; Telea initialization, normalized/clipped explicit transport and harmonic regularization; not the paper's full anisotropic solver |
| `criminisi` (2003/2004) | Confidence × data priority; exhaustive patch SSD | Local copy of `../feature-inpainting` reference core; retains its discretization conventions |
| `telea` (2004) | Heap-driven FMM Eikonal solve and first-order color extrapolation | Known exterior arrival time zero; only accepted colors; not OpenCV byte equivalence |
| `patchmatch` (2009) | NNF initialization, alternating propagation, shrinking random search | Single-scale completion by iterative patch voting; Telea initialization; source patches fixed to original known region |
| `feature` (2016) | Feature-priority exemplar extension | Local exemplar core copied from `../feature-inpainting`; original authors' code/benchmarks not reproduced |
| `lama` (2021/2022) | ONNX pretrained inference | Carve pretrained ONNX export; no network architecture or training reproduction |
| `migan` (2023) | Official ONNX pipeline inference | Mobile-oriented neural option; no training reproduction |

This is a selected collection, not the whole history. Image Quilting, global MRF optimization, Context Encoders, Partial/Gated Convolution, EdgeConnect, MAT/ZITS/PUT, RePaint/LDM, PowerPaint/BrushNet/TurboFill, and video ProPainter are not implemented here. Their names must not be used as aliases for the included algorithms. No paper quality benchmark or universal superiority is claimed.

## JavaScript API

```js
import {inpaint} from './paper-lab/js/classical.js';
const output = inpaint('patchmatch', rgb, mask, width, height,
  {patch:5, iterations:5, seed:1});
```

`rgb` is an interleaved `Uint8Array(H*W*3)`. `mask` has `H*W` entries: nonzero = hole. Output is new RGB bytes; inputs are not modified. No server or Node API is used by the algorithms. Call computational APIs from a Worker in production websites. Defaults: patch 5, seed 1; Telea radius 3; Bertalmio iterations 200 / dt .1. Existing exemplar defaults follow their source.

```js
import * as ort from 'onnxruntime-web/wasm';
import {createOnnxInpainter, MIGAN_URL} from './paper-lab/js/onnx.js';
ort.env.wasm.wasmPaths='/onnx-runtime/';
ort.env.wasm.numThreads=1;
const model=await createOnnxInpainter(ort,'migan',MIGAN_URL);
try { const output=await model.run(rgb,mask,width,height); }
finally { await model.dispose(); }
```

`npm install` copies the pinned runtime's WASM/MJS into `public/onnx-runtime/`; these generated assets are ignored by Git and rebuilt by `postinstall`. No CDN runtime script is needed. WASM single-thread mode avoids a COOP/COEP requirement. Actual model performance depends on the device; no latency guarantee. ONNX Runtime Web and Python ONNX Runtime execute the pretrained graphs; the model architectures are **not reimplemented** in this folder.

## Model contracts

MI-GAN defaults to the author's [official dynamic pipeline](https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx), linked by the [official repository](https://github.com/Picsart-AI-Research/MI-GAN). Inputs `image` uint8 `[1,3,H,W]`, `mask` uint8 `[1,1,H,W]`, 255 known / 0 hole. Output `result` uint8 `[1,3,H,W]`. The pipeline includes cropping/resizing. A raw MI-GAN generator export is **not** compatible. Mask polarity is inverted at the adapter boundary. Read the separate [weight license](https://github.com/Picsart-AI-Research/MI-GAN/blob/main/LICENSE-WEIGHTS) before commercial use. Weights are not committed or redistributed. Model graph code and pretrained weights retain their upstream licenses.

LaMa defaults to the [Carve big-LaMa ONNX export](https://huggingface.co/Carve/LaMa-ONNX) (about 199 MB), a third-party conversion of the original model. Inputs `image` float32 `[1,3,512,512]` in [0,1], `mask` float32 `[1,1,512,512]` with 1 hole; output NCHW RGB in [0,255]. We edge-pad images to 512 square without stretching, then crop back. Local files must follow this same contract. Other exports can use JS options `outputScale:255` for [0,1] outputs and `fixedSize:0` for dynamic shape; Python equivalents are `output_scale` and `fixed_size`. The raw four-channel internal generator is not compatible. This export replaces the Fourier unit to make it ONNX-exportable; check conversion compatibility rather than assuming every LaMa checkpoint runs in browsers. LaMa is not described as lightweight; MI-GAN is the mobile option. [LaMa original project](https://github.com/advimman/lama).

Both adapters zero masked input RGB, handle empty masks without inference, validate output shape/finite values, and composite only original hole pixels. Browser pages accept local `.onnx` files; both can download their documented default weights automatically. Model requests go to the weight host; photo pixels are not sent there. Local model files work without that download after application/runtime assets are available.

## Python

```sh
python3 -m venv /tmp/inpainting-lab-venv
/tmp/inpainting-lab-venv/bin/pip install -r paper-lab/python/requirements.txt
/tmp/inpainting-lab-venv/bin/python paper-lab/python/cli.py telea photo.png mask.png result.png
/tmp/inpainting-lab-venv/bin/python paper-lab/python/cli.py patchmatch photo.png mask.png result.png --patch 5 --seed 1
/tmp/inpainting-lab-venv/bin/python paper-lab/python/cli.py migan photo.png mask.png result.png --model /path/to/migan_pipeline_v2.onnx
```

CLI mask gray >127 means hole. Python API: add `paper-lab/python` to `sys.path`, import `classical.inpaint` or `onnx_inpaint.inpaint_onnx`; image H×W×3 uint8, mask H×W nonzero hole. Core APIs do not impose the browser resolution cap. Full masks or unavailable complete source patches fail explicitly. No automatic Python model downloads.

## Validation

```sh
npm run test:papers
npx tsc --noEmit
npm run lint
```

Tests compare independent JS/Python results on center holes, border holes, cracks and empty masks, enforce known-region preservation, masked-input independence and seeded determinism, and check ONNX channel layout/mask polarity. Cross-language numeric tolerance is one byte. These checks do not establish paper benchmark parity or perceptual quality. Real pretrained model validation is recorded separately in the Obsidian implementation handoff.

Primary sources: [Efros–Leung](https://people.eecs.berkeley.edu/~efros/research/EfrosLeung.html), [Bertalmío](https://doi.org/10.1145/344779.344972), [Criminisi](https://www.microsoft.com/en-us/research/publication/object-removal-by-exemplar-based-inpainting/), [Telea](https://webspace.science.uu.nl/~telea001/Shapes/Inpainting), [PatchMatch](https://pixl.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/index.php), [LaMa](https://github.com/advimman/lama), [MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN), [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html). Project/business handoff is maintained in Obsidian, not in additional repository documents.

Optional real-model CPU/WASM parity (download graphs to your own paths first):

```sh
python3 paper-lab/tests/verify_onnx.py migan /absolute/path/migan_pipeline_v2.onnx
python3 paper-lab/tests/verify_onnx.py lama /absolute/path/lama_fp32.onnx
```

The real-model tests use a small deterministic gradient image and verify known-region preservation plus CPU/WASM numerical agreement (at most 3 byte levels permitted). Test output RGB goes to the system temporary directory. Cross-language classical verification covers 30 cases, including disconnected holes; run from the project root.
