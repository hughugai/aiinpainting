# 特征优先块匹配图像修复算法：Python / JavaScript 复现

依据：李旭峰、王静、刘红敏、王志衡，2016，《特征优先块匹配图像修复算法》，计算机辅助设计与图形学学报，28(7)，1131–1137。实现由所提供论文推导，未使用作者原始程序。

## 快速开始

Python 3.10+，NumPy、Pillow；JavaScript 使用现代浏览器或 Node.js 18+，不需要 npm 依赖。

```bash
pip install -r requirements.txt
python inpaint.py examples/original.png examples/mask.png output.png --patch 9 --lambda 0.35 --trace trace.json
python inpaint.py examples/original.png examples/mask.png baseline.png --method criminisi
python verify.py
```

掩码须与图片尺寸相同：灰度 >127 的像素待修复，其余像素已知。只有掩码内像素被替换；被遮挡区域的 RGB 不参与匹配和优先级计算。

浏览器演示：

```bash
python -m http.server 8000
```

打开 http://localhost:8000/index.html 。选择图片后涂抹待修复区域，或者点击“载入样例”。可切换论文算法、仅置信项偏移和 Criminisi 基线，调整 λ、块大小和特征权重，下载 PNG 和掩码。演示最长边限制为 256，明确属于演示尺寸；Python / JS 核心不自动缩放。计算在 Web Worker 中执行，可取消。需 HTTP 服务，不支持直接双击 file://。

JS 核心：

```js
import {inpaint} from './inpaint.js';
// rgb: H*W*3 字节；mask: H*W，非零为缺失。
const {image, trace} = inpaint(rgb, mask, width, height, {
  patch:9, lambda:0.35, featureWeight:1, method:'feature',
  gradientScale:255, confidenceUpdate:'raw'
});
```

Node JSON 接口：`node run-json.mjs < input.json > result.json`，输入字段为 `rgb, mask, width, height, options`。输出是 RGB 数组及迭代 trace，无需图片解码依赖。

## 论文与实现对应

| 论文公式/步骤 | 实现 |
|---|---|
| 式(2) C(P)=已知像素置信度之和 / 块面积 | 初始已知=1、缺失=0，每轮计算边界块均值 |
| 式(3) D(P)=\|∇I⊥·n\| /255 | 以灰度梯度构造等照度线，乘边界单位法向量 |
| 式(4) Kc(P)=C(P)−λ | `shifted` 模式；默认 λ=0.35 |
| 式(5) Rc(P)=Kc(P)+\|IP\| | `feature` 模式；`featureWeight=1.5` 对应论文增强结构传播建议 |
| 式(1) P(P)=置信项×数据项 | 论文模式为 `(C−λ+w*feature)*D`；基线为 `C*D` |
| 全局 RGB SSD 最佳匹配 | 只比较目标块已知位置；来源块全部来自初始已知区域 |
| 填充和循环 | 仅复制目标块中缺失像素，更新置信度和边界直至完成 |

## 论文未明确的细节：本实现的可审查约定

这些设置影响结果，不能视为已确认与作者原代码完全相同。

1. **块大小**：默认 9×9，可调。靠图像边缘的目标块裁剪，来源使用相同裁剪形状。
2. **边界**：采用已知区域中与缺失像素四邻接的点，才能计算 P 点梯度；边界点按行优先遍历。
3. **灰度与梯度**：灰度 `0.299R+0.587G+0.114B`；中心差分，有一侧缺失则单侧差分，两侧都缺失则该方向为 0。不把掩码颜色边缘当作真实边缘。
4. **特征项单位**：默认 `|IP|=hypot(gx,gy)/255`，使梯度项与置信项量级可比较。这是实现选择，论文未指定归一化。Python `--gradient-scale 1` / JS `gradientScale:1` 可以使用未归一化梯度。
5. **数据项梯度**：取目标块内已知像素中梯度幅值最大的向量，而特征项取边界 P 点自己的梯度。论文未规定数据项的离散采样；本实现参考经典块修复惯例。
6. **法线**：二值掩码中心差分并单位化；细裂缝两侧抵消时，使用按上、下、左、右顺序第一个缺失邻居的方向。
7. **置信度更新**：默认新填充像素赋原始 `C(P)`；λ 和特征仅影响优先级。论文没有给出修正后的更新公式。另提供 `--confidence-update refined` / `confidenceUpdate:'refined'`，新像素赋 `Rc(P)`（shifted 时赋 Kc）。这属于敏感性实验开关，可能出现负值或增长，不宣称是哪一种原作者实现。
8. **负优先级**：保留 `(C−λ+feature)*D` 的符号，直接取最大值，不取绝对值、不截断到零，也不擅自添加 ε。如果数据项为零，允许按固定遍历顺序选择。这符合论文允许负值的表述，但固定偏移并不数学上保证优先级永不为零。
9. **源区域**：固定初始已知区域，避免再次复制已修复内容导致误差传播。候选为逐像素全局穷举；SSD 相等时选第一个（行优先），优先级相等时同样选择第一个。
10. **停止与失败**：全图缺失、没有完整源块、非法参数、迭代上限耗尽均报错，不回退成另一种修复算法。

## 验证与结果解释

运行 `python verify.py` 生成 `verification.json`，验证 Python 和 JS：输出逐字节相同、填充中心和来源相同、数值 trace 容差 1e-9；已知像素不变；改变掩码内部 RGB 不影响输出；覆盖纹理、平坦区、结构边缘、图像边界、细裂缝、空掩码、负优先级和错误输入。

`examples/comparison.png` 从左到右：原图、污染图、Criminisi 基线、论文算法。样例是可重复的合成横向结构，不是论文的原始实验图片。`verification.json` 的 MSE/PSNR **只对掩码区域计算**；MSE=0 时 JSON 的 PSNR 为 null，表示无穷大。该测试用于校验运行和跨语言一致性，不证明算法普遍优于基线。

论文插图并非未压缩的原图和精确掩码；因此这里不声称复现表1/表2的 PSNR，也不从 PDF 截图重新计算来冒充原实验。目标移除没有真实背景参照时，PSNR 不宜直接当作客观修复质量。

浏览器演示的 HTML 模块脚本、核心和 Worker 通过 Node 语法检查；当前验证环境缺少 Chromium 内核，未完成真实浏览器交互测试。Python CLI 已验证输出与核心 API 相同。

## 文件与局限

- `inpaint.py`：Python 核心与图片 CLI。
- `inpaint.js`：纯 JS 核心；`run-json.mjs`：Node JSON 接口。
- `index.html`、`worker.js`：本地浏览器演示。
- `verify.py`、`verification.json`：一致性/不变量测试与结果。
- `examples/`：完整可复现样例和输出。

这是便于核对论文的参考实现。全局穷举 SSD 的复杂度约为“迭代次数 × 来源候选数 × 已知块像素数”，Python 版本尚未向量化，高清图/大面积移除会很慢。加速应单独验证，以免改变匹配顺序。该算法依赖图中存在可借用结构/纹理，不会推理新的语义内容；边界选择、λ、梯度单位和块大小都可能明显影响结果。
