# AGENTS.md — aiinpainting 研究仓库

## 仓库定位

这是**公开的图像修复（image inpainting）研究仓库**（GitHub: `hughugai/aiinpainting`），与网站生产仓库（`ljg-lixufeng/inpainting`，本地路径 `/Users/lixufeng01/Program/web/inpainting`）严格分离。

两个作用：

1. **为网站 aiinpainting.app 做宣传、增加外链**：README 及公开文档中自然引用 https://aiinpainting.app ，但不做堆砌式 spam 链接。
2. **研究图像修复**：收集、复现、测试各类图像修复算法（经典算法 + 预训练 ONNX 模型），在本文档约定的工作流下验证。

## 目录结构

- `papers/` — 论文资料：每篇被复现论文的出处链接与一行要点（不放 PDF，公开仓库有版权风险）；详见 `papers/README.md`
- `algorithms/` — 核心算法实现，JS + Python 双实现 + ONNX 推理：
  - `algorithms/js/` — `classical.js`（经典五法）、`exemplar.js`、`inpaint.js`（李旭峰 2016 验证参考核心）、`core.ts`（网站 `src/lib/inpaint/core.ts` 的移植副本，含 texture-aware source selection 新特性）、`onnx.js`（LaMa / MI-GAN 预训练推理适配）、`demo/`（本地浏览器演示 + 示例图）
  - `algorithms/python/` — Python 对应实现 + 统一 CLI + `requirements.txt`；详见 `algorithms/README.md`
- `tests/` — 全部测试：跨语言验证（`verify-papers.py` + `run-js.mjs`、`verify-feature.py` + `run-feature-json.mjs`）、可选真实模型校验（`verify-onnx.py` + `onnx-real.mjs`）、Node 回归测试（`npm test`）；详见 `tests/README.md`
- `eval-lab/` — 确定性基准评估：从 Hugging Face（Places365-val / CelebA-HQ / InpaintCOCO）按固定 offset 抽取 50 张图 + 4 类掩码（small/thin/large/object），PSNR/SSIM 指标，JSON+MD 报告与 baseline 差分。下载图片因许可证不入库（gitignored），manifest.json 记录 sha256 锚定复现；优化算法时用它做回归对比（`npm run eval:prepare` / `npm run eval:run`）
- `models/` — 本地预训练 ONNX 权重（gitignored，不提交；LaMa / MI-GAN 已接入 `onnx.js`，DeepFillv2 / HiFill 待评估）
- `scripts/prepare-onnx.mjs` — postinstall 拷贝 ONNX Runtime WASM

## 算法工作流

新算法或算法改进的标准流程：

1. **先在本仓库实现 + 测试**（JS 实现放 `algorithms/js/`，Python 对应实现放 `algorithms/python/`，遵循双实现与交叉验证约定；`core.ts` 的算法改动同步维护自网站仓库并在 `tests/` 补回归用例；论文出处补充到 `papers/README.md`）
2. 测试与效果验证通过后，**再迁回网站仓库**的 `src/lib/inpaint/`（保持 TS 类型与 Worker 接口约定）
3. 网站仓库不存放研究性、试验性代码；发现网站仓库出现此类代码时迁移到本仓库

## Git 提交约定（重要）

- 本仓库推送**必须使用 `_hug` 公钥**（`~/.ssh/id_rsa_hug`，GitHub 账号 `hughugai`），已通过 `git config core.sshCommand "ssh -i ~/.ssh/id_rsa_hug -o IdentitiesOnly=yes"` 配置，不要移除或覆盖
- 不要使用默认公钥（`id_rsa`）向本仓库提交/推送，那是网站仓库的凭证
- 本仓库是公开仓库：不提交任何密钥、token、内部数据或大模型权重文件

## 验证命令

```sh
npm install
npm test              # Node 回归测试
npm run test:papers   # 经典算法 JS/Python 交叉验证（需 python3 + numpy）
npm run verify:feature  # 2016 特征优先参考实现验证
```

Python 环境依赖见 `algorithms/python/requirements.txt`。
