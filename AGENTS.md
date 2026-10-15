# AGENTS.md — aiinpainting 研究仓库

## 仓库定位

这是**公开的图像修复（image inpainting）研究仓库**（GitHub: `hughugai/aiinpainting`），与网站生产仓库（`ljg-lixufeng/inpainting`，本地路径 `/Users/lixufeng01/Program/web/inpainting`）严格分离。

两个作用：

1. **为网站 aiinpainting.app 做宣传、增加外链**：README 及公开文档中自然引用 https://aiinpainting.app ，但不做堆砌式 spam 链接。
2. **研究图像修复**：收集、复现、测试各类图像修复算法（经典算法 + 预训练 ONNX 模型），在本文档约定的工作流下验证。

## 目录结构

- `paper-lab/` — 论文算法复现（JS + Python 双实现 + ONNX 推理），来自网站仓库 `codex/inpainting-paper-lab` 分支；详细说明见 `paper-lab/README.md`
- `feature-inpainting/` — 李旭峰等 2016《特征优先块匹配图像修复算法》的验证参考实现（inpaint.js / inpaint.py），`core.ts` 是网站 `src/lib/inpaint/core.ts` 的移植副本（含 texture-aware source selection 新特性）
- `tests/` — 根级回归测试（Node 内置 test runner），当前测试 `feature-inpainting/core.ts` 的纹理感知源选择
- `scripts/prepare-onnx.mjs` — postinstall 拷贝 ONNX Runtime WASM

注意：`paper-lab` 内部代码通过 `../feature-inpainting` 相对路径引用参考实现，两者必须保持同级目录关系。

## 算法工作流

新算法或算法改进的标准流程：

1. **先在本仓库实现 + 测试**（放 `paper-lab/`，遵循其 JS/Python 双实现与交叉验证约定；`core.ts` 的算法改动同步维护在 `feature-inpainting/core.ts` 并在 `tests/` 补回归用例）
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
npm run test:papers   # paper-lab JS/Python 交叉验证（需 python3 + numpy）
npm run verify:feature  # feature-inpainting 参考实现验证
```

Python 环境依赖见 `paper-lab/python/requirements.txt` 与 `feature-inpainting/requirements.txt`。
