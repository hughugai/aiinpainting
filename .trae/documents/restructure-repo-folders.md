# 仓库目录重构：按「文档 / 论文 / Python 算法 / JS 算法 / 测试 / 模型」六大类整理

## Context

用户要求把 inpainting-research 仓库整理为清晰的六类结构：文档、paper、Python 核心算法实现、JS 核心算法实现、测试文件夹、模型文件。

现状：`paper-lab/`（js/ + python/ + tests/）与 `feature-inpainting/`（2016 参考实现，js+py+demo）功能重叠、组织分散；`papers/` 分类缺失；`models/` 已存在（gitignored，本地 4 个 onnx）。

已确认的全部跨目录引用点（重构需同步修改）：
- [package.json](file:///Users/lixufeng01/Program/web/inpainting-research/package.json) L9-10：`paper-lab/tests/verify.py`、`feature-inpainting/verify.py`
- [eval-lab/lib/algos.mjs](file:///Users/lixufeng01/Program/web/inpainting-research/eval-lab/lib/algos.mjs) L25/L39/L44/L56：`feature-inpainting/core.ts`、`feature-inpainting/inpaint.js`、`paper-lab/js/classical.js`、`paper-lab/js/onnx.js`（models 路径不变）
- [tests/inpaint-source-selection.test.mjs](file:///Users/lixufeng01/Program/web/inpainting-research/tests/inpaint-source-selection.test.mjs) L13：tsc 编译 `feature-inpainting/core.ts`
- `paper-lab/tests/run-js.mjs` L1-2、`onnx-real.mjs` L5：`../js/*.js`
- `paper-lab/tests/verify.py` L27：parents[2] 推导根目录 + `node paper-lab/tests/run-js.mjs`；`verify_onnx.py` L10：`node paper-lab/tests/onnx-real.mjs`
- `feature-inpainting/run-json.mjs` L2 → `./inpaint.js`；`worker.js` L1 → `./inpaint.js`；`index.html` → `examples/original.png`
- Python 文件无跨目录硬编码路径（cli.py 与被导入模块同目录搬迁即可）

## 目标结构

```
inpainting-research/
├── README.md  AGENTS.md  LICENSE  package.json  .gitignore   # 文档（根）
├── papers/                        # ② 论文资料（清单+笔记，无 PDF）
│   └── README.md
├── algorithms/                    # ③④ 核心算法实现
│   ├── README.md
│   ├── python/
│   │   ├── classical.py  exemplar.py  onnx_inpaint.py  cli.py   # ← paper-lab/python/
│   │   ├── inpaint.py             # ← feature-inpainting/inpaint.py
│   │   └── requirements.txt       # 合并两份 requirements
│   └── js/
│       ├── classical.js  exemplar.js  onnx.js                   # ← paper-lab/js/
│       ├── inpaint.js  core.ts    # ← feature-inpainting/
│       └── demo/                  # ← feature-inpainting/{index.html,worker.js,examples/}
├── tests/                         # ⑤ 全部测试
│   ├── verify-papers.py           # ← paper-lab/tests/verify.py（重命名防撞名）
│   ├── run-js.mjs  onnx-real.mjs  verify-onnx.py
│   ├── verify-feature.py  run-feature-json.mjs  verification-feature.json
│   │                              # ← feature-inpainting/{verify.py,run-json.mjs,verification.json}
│   ├── eval-lab.test.mjs          # 原地不动
│   └── inpaint-source-selection.test.mjs
├── models/                        # ⑥ 模型文件（gitignored，不动）
├── eval-lab/  scripts/  public/   # 评估 harness 与构建脚本保留原位
```

说明：文档按用户规则只保留 README.md（根 + 各子目录），不建 docs/ 目录；`feature-inpainting/package.json`、`paper-lab/package.json` 随目录消亡而删除（脚本并入根 package.json）；清理 `__pycache__/`。原 AGENTS.md「paper-lab 与 feature-inpainting 必须同级」的约束随合并自然解除。

## 实施步骤

1. **git mv 移动文件**（保留历史）：
   - `paper-lab/js/{classical,exemplar,onnx}.js` → `algorithms/js/`
   - `paper-lab/python/{classical,exemplar,onnx_inpaint,cli}.py` → `algorithms/python/`
   - `feature-inpainting/{inpaint.js,core.ts}` → `algorithms/js/`；`inpaint.py` → `algorithms/python/`
   - `feature-inpainting/{index.html,worker.js}` + `examples/` → `algorithms/js/demo/`
   - `paper-lab/tests/verify.py` → `tests/verify-papers.py`；`run-js.mjs`、`onnx-real.mjs`、`verify_onnx.py` → `tests/`（onnx 的 py 文件同步重命名 `verify-onnx.py`）
   - `feature-inpainting/{verify.py,run-json.mjs,verification.json}` → `tests/{verify-feature.py,run-feature-json.mjs,verification-feature.json}`
2. **合并 requirements**：两份 requirements.txt 合并为 `algorithms/python/requirements.txt`，删除原文件与两个子 package.json
3. **更新引用**（7 处）：
   - `eval-lab/lib/algos.mjs`：4 处 join 路径 → `algorithms/js/...`
   - `tests/inpaint-source-selection.test.mjs`：`feature-inpainting/core.ts` → `algorithms/js/core.ts`
   - `tests/run-js.mjs`：`../js/...` → `../algorithms/js/...`；`tests/onnx-real.mjs` 同理
   - `tests/verify-papers.py`：parents[2]→parents[1]，`paper-lab/tests/run-js.mjs`→`tests/run-js.mjs`
   - `tests/verify-onnx.py`：`paper-lab/tests/onnx-real.mjs`→`tests/onnx-real.mjs`
   - `tests/run-feature-json.mjs`：`./inpaint.js`→`../algorithms/js/inpaint.js`；`algorithms/js/demo/worker.js`：`./inpaint.js`→`../inpaint.js`
4. **根 package.json** 脚本：`test:papers` → `python3 tests/verify-papers.py`；`verify:feature` → `python3 tests/verify-feature.py`
5. **文档更新**：
   - 新建 `papers/README.md`（9 篇论文：Efros–Leung 1999、Bertalmío 2000、Criminisi 2003、Telea 2004、PatchMatch 2009、李旭峰 2016、LaMa 2021、DeepFillv2 2019、MI-GAN 2022，含 arXiv/出处链接与一行要点）
   - 新建 `algorithms/README.md`（方法清单、Python CLI 用法、demo 说明、依赖安装）
   - 新建 `tests/README.md` + `eval-lab/README.md`（简短：各测试用途与运行命令）
   - 重写根 `README.md`（新目录表 + Quick start 路径）与 `AGENTS.md`（目录结构、算法工作流、验证命令章节按新路径更新）
6. **清理**：删除 `feature-inpainting/__pycache__/`、残留空目录 `paper-lab/`、`feature-inpainting/`

## 验证

```sh
npm test               # 覆盖 core.ts 新路径（tsc 编译）+ eval-lab 指标/掩码
npm run test:papers    # verify-papers.py → run-js.mjs → algorithms/js 交叉验证
npm run verify:feature # verify-feature.py → run-feature-json.mjs → algorithms/js/inpaint.js
node -e "import('./eval-lab/lib/algos.mjs').then(m=>console.log(Object.keys(m.ALGOS)))"  # algos.mjs 模块加载
node -e "import('./algorithms/js/classical.js').then(()=>console.log('ok'))"             # 算法模块可加载
git status             # 移动以 rename 记录，无遗漏
```

eval-lab 数据集下载/全量跑分（`eval:prepare`/`eval:run`）不在本次验证范围（重、需网络），algos.mjs 仅改 4 个 join 路径且已被上述 import 冒烟覆盖。
