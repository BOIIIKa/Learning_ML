# 交接文档 / Handoff

> 写于 2026-10-14，供上下文压缩或换人之后继续工作使用。先读本文，再读 `CLAUDE.md` 和 `.claude/skills/`。
> Written 2026-10-14 for picking the work up after a context reset. Read this, then `CLAUDE.md` and `.claude/skills/`.

## 1. 现状一句话

**完成 6/30 章（00、01、02、03、07、08）**，全部合并到 `main`，并已部署到 https://boiiika.github.io/Learning_ML/。

- 竖版讲义：工程蓝图风格。
- 横版动画课程：3Blue1Brown / manim 风格，自动播放。
- 写新章的工作流已沉淀为 skills、模板和脚手架。

| PR | 内容 | 状态 |
|---|---|---|
| BOIIIKa/Learning_ML#1 | 网站骨架；第 00、07、08 章；练习系统 | 已合并 |
| BOIIIKa/Learning_ML#2 | 蓝图风格；第 01–03 章 | 已合并 |
| BOIIIKa/Learning_ML#4 | 3B1B 式动画课程引擎；前端优化；skills、模板、项目总结 | 已合并 |

## 2. 用户（项目负责人）的偏好与约定

- 用中文交流。网站是中英双语，学生来自不同国家。
- **非商用，完全开源**。不得复制原书文字或插图，只引用章号。
- 受众：有数学基础、想快速具备科研能力的学生。要求推导严谨，并跟进前沿。
- 每章两种形式：
  - **竖版讲义**用于复习。用户对现有风格和内容很满意，**不要改风格**。
  - **横版动画课程**用于初学。要像 3B1B：节奏适中、自动播放，突出 manim 式动画。
- 视觉：工程草稿 / 蓝图风格，配色克制，有少量网格线，动效要能抓住注意力（用户的部分美国学生不喜欢读书）。
- 练习：
  - 分为“推导理解”和“训练实战”两类；训练实战又分“补全算法”和“组装库”（例如 LightGBM）。
  - 训练过程要可视化，例如每次更新后超平面的移动。
  - 必须有参考答案；连按两次 Tab 填入答案，填充动效要酷。
- 章节顺序由我决定。**只有用户明确同意时才合并 PR**；用户会把网站链接发给学生测试。
- 工作分支：`claude/charming-knuth-ti0cof`。每次 PR 合并后，从 `main` 重新开这个分支，再开新的 PR（草稿）。

## 3. 下一步：第 04 章（线性分类）

先按 `.claude/skills/new-chapter/SKILL.md` 执行：

```bash
python3 scripts/new_chapter.py 04 --widget PerceptronSteps
```

规划草案（大纲要求见 `curriculum/outline.yaml` 中 id "04" 的条目）：

- **情景**：延续第 07 章的医院初筛。先不问“哪条线最好”，而是问“一条线怎样从错误中学会分开两类？”，引出感知机。之后再问“我想要一个概率”，引出逻辑回归。
- **交互组件**：
  - `PerceptronSteps`：逐步更新感知机。高亮当前误分类点，画出更新向量，超平面旋转；可切换原始 / 对偶形式。可以复用 `Plot2D` 的 `step.w/b` 与 `highlight`。
  - `LogisticSurface`：拖动 w、b，sigmoid 概率场（`Plot2D` 的 field）与交叉熵损失联动更新。
  - `LossZoo`：用 `CurvePlot` 画 0-1、感知机、交叉熵、合页、指数损失关于间隔 yf(x) 的曲线，链接第 07、08 章。
  - 可选 `NewtonVsGD`：比较牛顿法（IRLS）与梯度下降的收敛速度。
- **推导**：
  - Novikoff 定理：误分类次数 ≤ (R/γ)²
  - 逻辑回归对数似然的梯度与 Hessian，并证明其凹性
  - 最大熵的拉格朗日对偶 ⇒ softmax
  - 例题：在四五个点上手算感知机更新
- **练习**：
  - 推导题：sigmoid 的导数 σ(1−σ)；Novikoff 界；交叉熵梯度 (σ−y)x；softmax 对 logit 的导数。
  - implement：感知机原始形式 + 逻辑回归牛顿法；trace 回放超平面的转动。
  - library：`LogisticRegression`（多分类、正则化路径 C）与 `SGDClassifier(loss=…)` 的对比。
- 随后是第 05 章（LDA/QDA、朴素贝叶斯，可复用第 01 章的贝叶斯分类器）和第 06 章（kNN、kd 树、样条与 GAM、维数灾难）。第 04–06 章完成后，第二部分就补齐了。

## 4. 环境备忘（云端沙箱）

这些路径只在当前会话的沙箱里有效，容器回收后需要按下面的说明重建。

- Python 虚拟环境（含 numpy、scipy、sklearn、lightgbm、pytest、pyyaml）：
  `/tmp/claude-0/-home-user-Learning-ML/d78ce949-50c6-5fa6-b1f1-4262f6405940/scratchpad/venv/bin/python`
  重建：`python3 -m venv venv && venv/bin/pip install numpy scipy scikit-learn lightgbm pytest pyyaml`
- 本地 Pyodide 0.29.3（沙箱访问不了 jsDelivr）：`$SP/pyodide-dl/pyodide`，其中 `$SP` 是上面的 scratchpad 目录。
  - 使用：`ln -sfn $SP/pyodide-dl/pyodide site/public/pyodide`，再运行 `PUBLIC_PYODIDE_URL=/pyodide/ npx astro dev --port 4321 --host 127.0.0.1`。
  - **构建或提交前删除这个软链接**（已加入 .gitignore，但构建时会把它复制进 dist）。
- Astro 7 的 dev 服务器会以守护进程运行，用 `npx astro dev stop` 停止。`astro build` 之后 dev 服务器的 Vite 缓存可能失效，需要 `rm -rf node_modules/.vite` 后重启。
- Playwright：`/opt/node-tools/node_modules/playwright`，`site/scripts/verify/_browser.mjs` 会自动找到它。
- github.io 在沙箱里访问不了。部署结果用 GitHub Actions 中 `pages.yml` 的运行状态确认。
- GitHub 操作只能用 MCP 的 github 工具（没有 `gh`），推送用 `git push -u origin claude/charming-knuth-ti0cof`。

## 5. 推送前检查清单

```bash
cd site && npx tsc --noEmit -p . && npx vitest run && npx astro build      # 65 tests, 30 pages
cd .. && python -m pytest -q tests && python3 scripts/build_outline.py --check   # 36 tests
cd site && node scripts/verify/course.mjs walk zh <slug> && node scripts/verify/course.mjs walk en <slug>
node scripts/verify/chapter.mjs <slug> zh dark     # 需要本地 Pyodide
node scripts/verify/pages.mjs <slug>
```

## 6. 压缩上下文用的关键词 / Context-compression keywords

Learning_ML · BOIIIKa/Learning_ML · boiiika.github.io/Learning_ML · 看得见的机器学习 / Visible ML · ISL + UDL(Prince) + 李航 · 重写不复制 · 中英双语 · 非商用 CC BY-NC-SA + MIT ·
6/30 章完成（00 数学工具、01 三要素、02 泛化、03 线性回归、07 SVM、08 树与集成）· 下一步 04 线性分类 → 05 生成式 → 06 非参数 ·
竖版讲义 = 蓝图风格（不改）· 横版动画课程 = 3B1B / manim 自动播放 · 六段结构（情景 / 方法 / 数学 / 例题拓展 / 课后训练 / 实战手册）+ 一页纸 recap ·
Astro 7 + React 19 + MDX + KaTeX 0.16.47 + Pyodide 0.29.3 · src/manim（Clock、Write、Morph、Indicate、Circumscribe、createSvg）· drive DSL（s1=11/4、s1=80%、click:chip2、wait:1）· Step effect（write / fade / pop / focus / morph）· `<Morph>` 推导变形 ·
练习：推导题（mathjs 数值比对）+ implement / library（@blank、双击 Tab 填答案、Trace 回放 classifier2d、field=f(x)−y）·
skills：new-chapter、course-deck、widget、exercise、verify-site · `scripts/new_chapter.py` + `templates/chapter` + `TODO(scaffold)` 守卫测试 · `site/scripts/verify`（chapter / course walk|film / pages）·
hydration 规则（f1 取整、r6 数据、SSR 中不写多值 style）· MDX 规则（多行 `$$`、正文用 \lt 不用裸 <）·
分支 claude/charming-knuth-ti0cof · PR #1 #2 #4 已合并 · 只在用户同意时合并 · 本地 Pyodide 软链接与 venv 位于 scratchpad
