# Algorithms

Core inpainting implementations in JavaScript and Python, plus pretrained ONNX inference and a local browser demo. Paper sources are listed in [`papers/README.md`](../papers/README.md).

## Layout

```
algorithms/
├── js/
│   ├── classical.js   # efros-leung / bertalmio / criminisi / telea / patchmatch (+ feature variants)
│   ├── exemplar.js    # feature-priority exemplar (2016) standalone lab implementation
│   ├── inpaint.js     # verified reference core of 李旭峰 et al. 2016 (纯 JS，无依赖)
│   ├── core.ts        # website port of inpaint.js + texture-aware source selection
│   ├── onnx.js        # LaMa / MI-GAN pretrained ONNX adapters (onnxruntime-web)
│   └── demo/          # local browser demo (index.html + worker.js + examples/)
└── python/
    ├── classical.py   # Python counterparts of classical.js
    ├── exemplar.py    # Python counterpart of exemplar.js
    ├── inpaint.py     # Python core of the 2016 algorithm + image CLI
    ├── onnx_inpaint.py# Python ONNX Runtime inference
    ├── cli.py         # unified CLI over all methods
    └── requirements.txt
```

## Method coverage and fidelity

| Method | Implementation | Scope |
|---|---|---|
| `efros-leung` (1999) | Gaussian weighted conditional neighborhood sampling | Hole completion adaptation, fixed fully known source windows, strongest-context frontier, seeded LCG |
| `bertalmio` (2000) | Isophote transport of Laplacian | **Educational approximation**; Telea initialization, normalized/clipped explicit transport and harmonic regularization; not the paper's full anisotropic solver |
| `criminisi` (2003/2004) | Confidence × data priority; exhaustive patch SSD | Reference core of the 2016 implementation in Criminisi mode; retains its discretization conventions |
| `telea` (2004) | Heap-driven FMM Eikonal solve and first-order color extrapolation | Known exterior arrival time zero; only accepted colors; not OpenCV byte equivalence |
| `patchmatch` (2009) | NNF initialization, alternating propagation, shrinking random search | Single-scale completion by iterative patch voting; Telea initialization; source patches fixed to original known region |
| `feature` (2016) | Feature-priority exemplar extension | Priority `(C−λ+w·feature)·D`; original authors' code/benchmarks not reproduced |
| `lama` (2021/2022) | ONNX pretrained inference | Carve big-LaMa ONNX export; no network architecture or training reproduction |
| `migan` (2023) | ONNX pretrained inference | Official MI-GAN pipeline export; mobile-oriented; no training reproduction |

## 2016 feature-priority reference core (`inpaint.js` / `inpaint.py` / `core.ts`)

依据：李旭峰、王静、刘红敏、王志衡，2016，《特征优先块匹配图像修复算法》，计算机辅助设计与图形学学报，28(7)，1131–1137。实现由论文公式推导，未使用作者原始程序。

论文公式对应：式(2) 置信度 C(P)、式(3) 数据项 D(P)、式(4) `shifted` 模式的 Kc(P)=C(P)−λ（默认 λ=0.35）、式(5) `feature` 模式的 Rc(P)=Kc(P)+‖I_P‖、式(1) 优先级 P(P)=置信项×数据项（`feature` 模式为 `(C−λ+w*feature)*D`，Criminisi 基线为 `C*D`）；全局 RGB SSD 只比较目标块已知位置，来源块全部来自初始已知区域。

可审查约定（论文未明确、由实现自行确定）：块大小默认 9×9 可调；边界为已知区域中与缺失像素四邻接的点；灰度 `0.299R+0.587G+0.114B`、中心差分；特征项默认归一化 `hypot(gx,gy)/255`（`gradientScale:1` 可关闭）；负优先级保留符号直接取最大；置信度更新默认赋原始 C(P)（`confidenceUpdate:'refined'` 为敏感性实验开关）；SSD/优先级相等时按行优先取第一个。全图缺失、无完整源块、迭代耗尽均显式报错，不回退到其他算法。

Python CLI：

```bash
pip install -r algorithms/python/requirements.txt
python algorithms/python/inpaint.py algorithms/js/demo/examples/original.png algorithms/js/demo/examples/mask.png output.png --patch 9 --lambda 0.35
python algorithms/python/inpaint.py <img> <mask> <out> --method criminisi   # 基线对比
```

掩码与图片同尺寸，灰度 >127 为待修复；只有掩码内像素被替换。JS 核心：

```js
import {inpaint} from './algorithms/js/inpaint.js';
// rgb: H*W*3 字节；mask: H*W，非零为缺失。
const {image, trace} = inpaint(rgb, mask, width, height,
  {patch:9, lambda:0.35, featureWeight:1, method:'feature'});
```

Node JSON 接口：`node tests/run-feature-json.mjs < input.json > result.json`（输入字段 `rgb, mask, width, height, options`）。

## Browser demo

```bash
python -m http.server 8000   # 在 algorithms/js/demo/ 目录下运行
# 打开 http://localhost:8000/index.html
```

选择图片、涂抹待修复区域或点击“载入样例”；可切换论文算法 / 仅置信项偏移 / Criminisi 基线，调整 λ、块大小和特征权重，下载 PNG 与掩码。计算在 Web Worker 中执行、可取消；演示最长边限制 256 px（核心 API 不自动缩放）。需 HTTP 服务，不支持 `file://`。`demo/examples/comparison.png` 从左到右：原图、污染图、Criminisi 基线、论文算法。

## JavaScript API

```js
import {inpaint} from './algorithms/js/classical.js';
const output = inpaint('patchmatch', rgb, mask, width, height,
  {patch:5, iterations:5, seed:1});
```

`rgb` is an interleaved `Uint8Array(H*W*3)`; `mask` has `H*W` entries, nonzero = hole. Output is new RGB bytes; inputs are not modified. Defaults: patch 5, seed 1; Telea radius 3; Bertalmio iterations 200 / dt 0.1. Call computational APIs from a Worker in production.

ONNX inference (weights live in `models/`, gitignored; `npm install` copies the pinned WASM runtime into `public/onnx-runtime/`):

```js
import * as ort from 'onnxruntime-web/wasm';
import {createOnnxInpainter, MIGAN_URL} from './algorithms/js/onnx.js';
ort.env.wasm.wasmPaths='/onnx-runtime/';
ort.env.wasm.numThreads=1;
const model=await createOnnxInpainter(ort,'migan',MIGAN_URL);
try { const output=await model.run(rgb,mask,width,height); }
finally { await model.dispose(); }
```

### Model contracts

- **MI-GAN**: official [dynamic pipeline export](https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx); inputs `image` uint8 `[1,3,H,W]`, `mask` uint8 `[1,1,H,W]` (255 known / 0 hole), output `result` uint8. A raw generator export is **not** compatible. Read the [weight license](https://github.com/Picsart-AI-Research/MI-GAN/blob/main/LICENSE-WEIGHTS) before commercial use.
- **LaMa**: [Carve big-LaMa ONNX export](https://huggingface.co/Carve/LaMa-ONNX) (~199 MB); inputs `image` float32 `[1,3,512,512]` in [0,1], `mask` float32 with 1=hole; output NCHW RGB in [0,255]. Images are edge-padded to 512 square without stretching, then cropped back. Other exports may need options `outputScale:255` / `fixedSize:0` (Python: `output_scale` / `fixed_size`).

Both adapters zero masked input RGB, handle empty masks without inference, validate output shape/finite values, and composite only original hole pixels. Weights are not committed or redistributed.

## Python CLI (all methods)

```bash
python algorithms/python/cli.py telea photo.png mask.png result.png
python algorithms/python/cli.py patchmatch photo.png mask.png result.png --patch 5 --seed 1
python algorithms/python/cli.py migan photo.png mask.png result.png --model models/migan_pipeline_v2.onnx
```

CLI mask: gray >127 means hole. Python API: add `algorithms/python` to `sys.path`, import `classical.inpaint` or `onnx_inpaint.inpaint_onnx`. No automatic model downloads.

## Validation

Cross-language checks live in [`tests/`](../tests/README.md): `npm run test:papers` (30-case JS/Python parity), `npm run verify:feature` (2016 core consistency), optional real-model parity via `python3 tests/verify-onnx.py migan|lama /absolute/model.onnx`. Tolerances: one byte cross-language; at most 3 bytes CPU/WASM. These checks establish determinism and cross-language agreement, not paper-benchmark parity or perceptual quality.

Limitations: the 2016 exemplar core is globally exhaustive (≈ iterations × source candidates × known patch pixels), unvectorized in Python — slow on large images. The algorithms borrow structure/texture; they do not infer new semantic content.
