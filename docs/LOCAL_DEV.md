# 本地开发指南 / Local development guide

这份文档面向在自己电脑上继续开发本项目的人（以及在本地运行的 Claude Code）。先读这里，再读 [`HANDOFF.md`](HANDOFF.md)（进度、用户偏好、下一章计划）和 [`CLAUDE.md`](../CLAUDE.md)（项目规则）。

This guide is for continuing the project on your own machine (and for Claude Code running locally). Read it first, then [`HANDOFF.md`](HANDOFF.md) (status, preferences, next chapter) and [`CLAUDE.md`](../CLAUDE.md) (project rules).

---

## 1. 需要安装的工具

| 工具 | 版本 | 用途 |
|---|---|---|
| Git | 任意新版本 | 版本管理 |
| Node.js | **22**（见 `.nvmrc`，推荐用 nvm / fnm 安装） | Astro 网站、单元测试、Playwright 检查脚本 |
| Python | **3.11+**（CI 用 3.13） | 练习的检查与生成脚本（`requirements-dev.txt`） |
| Chromium（Playwright 自带） | 由 `npx playwright install chromium` 安装 | `site/scripts/verify/` 的浏览器检查 |
| Claude Code（可选） | 最新版 | 在本地让 Claude 继续写章节；会自动读取 `CLAUDE.md` 和 `.claude/skills/` |
| 编辑器（可选） | VS Code + Astro、MDX、Python 扩展 | 编辑 `.astro`、`.mdx`、`.tsx`、`.py` |

浏览器里的 Python（代码练习）由 **Pyodide 0.29.3** 提供，默认从 jsDelivr CDN 加载，本地不用装。只有离线或网络屏蔽 CDN 时才需要 `scripts/fetch_pyodide.sh`（见第 6 节）。

## 2. 一次性设置

```bash
git clone https://github.com/BOIIIKa/Learning_ML.git
cd Learning_ML
bash scripts/setup_local.sh            # macOS / Linux
# Windows（PowerShell）：powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1
```

脚本会做这些事：检查 Node 22 与 Python 3.11+；创建 `.venv` 并安装 `requirements-dev.txt`；在 `site/` 里 `npm ci` 并安装 Playwright 的 Chromium；生成练习数据；最后跑一遍类型检查、vitest 和 pytest。全部通过就说明环境没问题。

手动等价步骤：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cd site && npm ci && npx playwright install chromium
```

## 3. 日常开发

```bash
source .venv/bin/activate              # Windows: .\.venv\Scripts\Activate.ps1
cd site
npm run dev                            # http://localhost:4321/zh/ ，会先运行 build_exercises.py
```

- 讲义：`http://localhost:4321/zh/chapters/<slug>/`；动画课程：`/zh/course/<slug>/`；英文把 `zh` 换成 `en`。
- 改了 `exercises/` 下的文件后运行 `python scripts/build_exercises.py`（`npm run dev` 启动时会自动跑一次）。若页面上新练习不出现，停掉开发服务器，删除 `site/node_modules/.vite` 和 `site/.astro` 后重启。
- 改了 `curriculum/outline.yaml` 后运行 `python scripts/build_outline.py`。
- 预览和 GitHub Pages 一致的生产版本（带 `/Learning_ML/` 前缀）：

```bash
cd site && SITE_BASE=/Learning_ML/ npm run build && npx astro preview   # http://localhost:4321/Learning_ML/
```

### 目录速查

| 想改什么 | 去哪里 |
|---|---|
| 一章的讲义 | `site/src/content/{zh,en}/<slug>.mdx` |
| 动画课程 | `site/src/slides/{zh,en}/<slug>.mdx` |
| 一页纸总结 / 实战手册 | `site/src/recaps/…`、`site/src/handbook/…` |
| 交互组件 | `site/src/components/chapters/chNN/`；共享组件在 `components/viz`、`components/ui` |
| 组件背后的数学 | `site/src/engines/<slug>.ts` + `<slug>.test.ts` |
| 动画引擎（Clock、Write、Morph、drive） | `site/src/manim/`、`site/src/components/course/controller.ts` |
| 练习 | `exercises/chNN/`（`exercises.yaml` + `*.py` + `*_check.py`） |
| 大纲（30 章的唯一数据源） | `curriculum/outline.yaml` |

## 4. 写新的一章

```bash
python scripts/new_chapter.py 18 --widget EarlyStopRidge   # 生成全部文件，带 TODO(scaffold) 标记
```

然后按 `.claude/skills/new-chapter/SKILL.md` 的顺序：先在 Python 里把例题数字算准 → 引擎与单元测试 → 组件（看截图）→ 讲义 → 动画课程 → 一页纸 → 练习 → 手册 → 验证。`site/src/lib/scaffold.test.ts` 会在还有 `TODO(scaffold)` 时让测试失败，防止半成品被合并。

**用本地 Claude Code 继续**：在仓库根目录运行 `claude`，它会自动加载 `CLAUDE.md` 和五个 skills（new-chapter、course-deck、widget、exercise、verify-site）。`.claude/settings.json` 已经允许常用的只读与检查命令免确认。开场可以这样说：

> 读 docs/HANDOFF.md 和 docs/LOCAL_DEV.md，然后按 new-chapter skill 写第 18 章。

## 5. 检查工具（`site/` 目录下运行，需要先 `npm run dev`）

| 命令 | 作用 |
|---|---|
| `npx tsc --noEmit -p .` · `npm test` | 类型检查 · 引擎与判题的单元测试（vitest） |
| `python -m pytest -q ../tests` | 每道代码练习：参考答案全部通过、骨架全部不通过（中英两种提示） |
| `npm run verify:chapter -- <slug> zh dark` | 截下每个组件；在 Pyodide 里填入答案并运行每道代码题；控制台必须干净 |
| `npm run verify:course -- walk zh <slug>` | 逐拍走完动画课程（英文换 `en`），控制台必须干净 |
| `npm run verify:course -- film <slug> zh "#5" "2,6,10"` | 从第 5 页播放，在第 2、6、10 秒截图（看 Morph 中途、drive 效果、节奏） |
| `npm run verify:pages -- <slug>` | 首页、手册、该章在深浅两种主题下的水合与 390 px 手机宽度溢出 |
| `npm run verify:hydration -- /zh/chapters/<slug>/` | 定位 React 水合不一致：打印“+ 客户端 / − 服务端”的差异行和所在组件 |
| `npm run verify:mathdiff` | 逐位比较 Node 与浏览器的 `Math.*`（现在 cos、sin、非整数次幂不一致） |

截图在 `site/verify-out/`（已忽略）。**一定要打开看**：布局溢出、颜色看不清、组件状态不对，都不会表现为报错。检查脚本默认访问 `http://127.0.0.1:4321`，可用环境变量 `SITE_URL` 改（例如检查生产预览：`SITE_URL=http://localhost:4321/Learning_ML`）。

### 推送前

```bash
cd site && npx tsc --noEmit -p . && npx vitest run && npx astro build
cd .. && python -m pytest -q tests && python scripts/build_outline.py --check
cd site && npm run verify:chapter -- <slug> && npm run verify:course -- walk zh <slug> && npm run verify:course -- walk en <slug> && npm run verify:pages -- <slug>
```

### 分支、PR 与部署

- 在功能分支上开发，开 PR 到 `main`。CI（`.github/workflows/ci.yml`）运行 pytest、大纲检查、vitest 和构建。
- 合并到 `main` 后，`pages.yml` 以 `SITE_BASE=/Learning_ML/` 构建并部署到 https://boiiika.github.io/Learning_ML/ 。
- 约定：一章一个提交（"Add chapter NN: …"）；合并前 CI 必须全绿。

## 6. 离线或无法访问 CDN 时的 Pyodide

```bash
bash scripts/fetch_pyodide.sh                      # 下载 0.29.3 完整版（约 300 MB）到 .cache/ 并链接到 site/public/pyodide
cd site && PUBLIC_PYODIDE_URL=/pyodide/ npm run dev
rm site/public/pyodide                             # 构建或提交前删掉链接，否则 300 MB 会被复制进 dist/
```

没设 `PUBLIC_PYODIDE_URL` 而 CDN 又不通时，代码题会一直显示“加载中”，`verify:chapter` 会超时。

## 7. 心得：这个项目里真正踩过的坑

### 水合（hydration）——服务端渲染和浏览器第一次渲染必须逐字相同

- SVG 坐标一律 `toFixed(1)`（各组件里的 `f1`）；生成的数据用 `r6()` 取整；随机数只用带种子的 `rng(seed)`，不要在渲染里用 `Math.random()`。
- **Node 与浏览器的 `Math.cos`、`Math.sin`、非整数次幂 `x ** y` 在最后一位上可能不同**（`npm run verify:mathdiff` 可复现）。平时取整就够了；但不稳定的迭代（例如大学习率下的训练）会把这一位放大成完全不同的曲线。第 16 章因此用只含四则运算的 `detCos`，并把初始权重 `r6` 取整。
- SSR 的 `style` 里不要写多值属性或 CSS 变量（在 `useEffect` 里设置）；组件里不要用 `<table>`（讲义页加载后会给表格加类名）；SVG 的 `id`（clipPath、marker、渐变）要用 `useId()` 保证每个实例唯一。
- 计算量大的组件要算得快，因为服务端和浏览器各算一遍：第 16、17 章把小网络训练改成 `Float64Array` 扁平数组，快了约 20 倍（`backprop-init.ts` 的 `trainDeep` 约 40 ms）。

### 严谨性

- 讲义、课程、一页纸、练习里的**每个数字**都先在 Python 里算出来，再由 vitest 或 pytest 断言。组件说明文字里的每个论断（“5 层起停在 ln 2”“κ = 100 时 346 → 52 步”）都要有对应的单元测试。
- 实验结论不成立就改说法，而不是挑数据。例：第 16 章原想检查“Newton-CG 比 CG 迭代少”，实测相反（83 次对 36 次），于是改为检查“BFGS 的梯度计算次数约为 CG 的一半”。
- 混沌或嘈杂的结果（大学习率训练）只断言稳健的结论，并在多个随机种子下确认。
- 浏览器里的 scikit-learn 是 1.7，本地可能更新：不要用 1.7 之后的 API；`MLPRegressor` 的第一个位置参数是 `loss`，层数要写 `hidden_layer_sizes=(…)`。`verify:chapter` 会在真正的 Pyodide 里跑一遍。

### 动画课程

- 用户指定的播放方式：**点一下播完当前这一页的动画和旁白，然后暂停**留给学生思考；“连播”按钮或 A 键一直播下去。所以每页应是一个完整的想法，并以值得思考的内容收尾。
- 中英文课程必须**页数、拍数、顺序完全相同**（drive 脚本只写一次，两种语言共用）。
- 每个组件页都要有 `drive` 脚本（`s1=0.5/2`、`click:chip2`、`wait:1`）；写之前先 `grep` 组件的 Slider 和 chip 顺序。学生拖动控件时脚本会让位，不会把值抢回去。
- 每章至少一条 `<Morph>` 推导链，前后两步要共享大部分符号；看效果时截**变形中途**的帧。
- 旁白：中文每拍 15–60 字，英文 10–40 词；屏幕上一拍只放一行。

### MDX 与 KaTeX

- 行间公式只用多行 `$$` 块；正文里的 `<` 写成 `$\lt$`，不要出现裸的 `{`、`<`；不要写 `style=""` 字符串。
- KaTeX 的 CSS 版本要和 rehype-katex 内部的 katex 一致（锁在 0.16.x），否则符号错位。

### 其他

- `astro build` 之后开发服务器的 Vite 缓存可能失效（“504 Outdated Optimize Dep”、组件不水合）：停服务器，删 `site/node_modules/.vite`，重启。
- 第一次运行 `verify:chapter` 若显示 `code exercises: 0`，多半是冷编译还没渲染完，或开发服务器拿着旧的 `exercises.json`：再跑一次，或按第 3 节清缓存重启。
- 讲义页的“蓝图”风格用户很满意，**不要改风格**；颜色只用 CSS 变量（`--accent`、`--accent-2`、`--class-pos`…），否则两种主题之一会坏。
- 版权：不复制原书的文字和插图，只按章引用；所有内容中英双语、各自地道地写，不逐字翻译。

## 8. 常见问题

| 现象 | 原因 / 办法 |
|---|---|
| `Playwright not found` | `cd site && npm ci && npx playwright install chromium` |
| 代码题一直“加载中” | 访问不了 jsDelivr：见第 6 节 |
| “A tree hydrated but some attributes … didn't match” | `npm run verify:hydration -- <路径>` 找到组件；按第 7 节的水合规则修 |
| 行间公式显示成行内 | 单行 `$$…$$`，改成多行块 |
| MDX 编译错误，指向 `<` 或 `{` | 正文里用 `$\lt$`，或把它们放进行内公式 |
| pytest 里某章的骨架“意外通过” | `@blank` 区域外已经写了答案；骨架必须在每个空上失败 |
| 课程 walk 报错但讲义正常 | drive 脚本里的 `sN` / `chipN` 序号和组件不符 |
