# Papers & references

Source papers behind the implementations in [`algorithms/`](../algorithms/README.md) and the pretrained graphs in `models/` (gitignored, downloaded locally). Notes and links only — no PDFs are redistributed in this public repository; fetch from the linked sources. Upstream model weights keep their own licenses and are never committed here.

## Classical algorithms (implemented in `algorithms/`)

| Year | Paper | Source | Method slug |
|---|---|---|---|
| 1999 | Texture synthesis by non-parametric sampling (Efros & Leung, ICCV) | [page](https://people.eecs.berkeley.edu/~efros/research/EfrosLeung.html) | `efros-leung` |
| 2000 | Image inpainting (Bertalmío et al., SIGGRAPH) | [doi:10.1145/344779.344972](https://doi.org/10.1145/344779.344972) | `bertalmio` |
| 2003/2004 | Object removal by exemplar-based inpainting (Criminisi, Pérez & Toyama) | [MSR publication page](https://www.microsoft.com/en-us/research/publication/object-removal-by-exemplar-based-inpainting/) | `criminisi` |
| 2004 | An image inpainting technique based on the fast marching method (Telea) | [page](https://webspace.science.uu.nl/~telea001/Shapes/Inpainting) | `telea` |
| 2009 | PatchMatch: a randomized correspondence algorithm (Barnes et al., SIGGRAPH) | [page](https://pixl.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/index.php) | `patchmatch` |
| 2016 | 特征优先块匹配图像修复算法 (李旭峰、王静、刘红敏、王志衡，计算机辅助设计与图形学学报 28(7):1131–1137) | journal paper; implemented from its equations, authors' code not used | `feature` |

## Neural methods (pretrained ONNX graphs, not reimplemented)

| Year | Paper / model | Source | Local file | Wired into `algorithms/js/onnx.js`? |
|---|---|---|---|---|
| 2021/2022 | LaMa — Resolution-robust large mask image inpainting (Suvorov et al., WACV) | [project](https://github.com/advimman/lama), [Carve ONNX export](https://huggingface.co/Carve/LaMa-ONNX) | `lama_fp32.onnx` | yes (`lama`) |
| 2022 | MI-GAN — Efficient image inpainting (Sargsyan et al.) | [repo](https://github.com/Picsart-AI-Research/MI-GAN), [official pipeline export](https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx) | `migan_pipeline_v2.onnx` | yes (`migan`) |
| 2019 | DeepFillv2 — Free-form image inpainting with gated convolution (Yu et al., ICCV) | [repo](https://github.com/JiahuiYu/generative_inpainting) | `deepfillv2.onnx` | no — third-party export under evaluation |
| 2021 | HiFill — Contextual residual aggregation for ultra-high-resolution inpainting (Yi et al., CVPR) | [repo](https://github.com/Seanseyy/HiFill) | `hifill.onnx` | no — third-party export under evaluation |

Only LaMa and MI-GAN adapters are implemented and cross-verified; the DeepFillv2 / HiFill exports are stored locally for study and may use incompatible tensor contracts. Check conversion compatibility rather than assuming any checkpoint runs.

## Scope & honesty

This is a selected collection for study and comparison — not a complete history of inpainting. Image Quilting, Context Encoders, Partial/Gated Convolution (as reimplementation), EdgeConnect, MAT/ZITS/PUT, RePaint/LDM and video ProPainter are not implemented here; their names must not be used as aliases for the included algorithms. No paper-benchmark parity is claimed.
