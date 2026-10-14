# 交接文档 / Handoff

> 写于 2026-10-14，供上下文压缩或换人之后继续工作使用。先读本文，再读 `CLAUDE.md` 和 `.claude/skills/`。
> Written 2026-10-14 for picking the work up after a context reset. Read this, then `CLAUDE.md` and `.claude/skills/`.

## 1. 现状一句话

**完成 7/30 章（00、01、02、03、04、07、08）**。第 04 章在 PR BOIIIKa/Learning_ML#5 中，等待用户同意合并；其余已合并到 `main` 并部署到 https://boiiika.github.io/Learning_ML/。

- 竖版讲义：工程蓝图风格。
- 横版动画课程：3Blue1Brown / manim 风格，自动播放。
- 写新章的工作流已沉淀为 skills、模板和脚手架。

| PR | 内容 | 状态 |
|---|---|---|
| BOIIIKa/Learning_ML#1 | 网站骨架；第 00、07、08 章；练习系统 | 已合并 |
| BOIIIKa/Learning_ML#2 | 蓝图风格；第 01–03 章 | 已合并 |
| BOIIIKa/Learning_ML#4 | 3B1B 式动画课程引擎；前端优化；skills、模板、项目总结 | 已合并 |
| BOIIIKa/Learning_ML#5 | 交接文档；第 04 章；Morph 播放与 KaTeX 根号的修复 | 草稿，进行中 |

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

## 3. 下一步：第 05 章（生成式分类）

第 04 章已完成（感知机、逻辑回归、最大熵；组件 PerceptronSteps、LogisticSurface、MaxEntDice、LossZoo）。它的情景、数据和引擎（`engines/linear-classification.ts` 中的 `screening`、`overlapping`、`newton`）可以直接复用。

```bash
python3 scripts/new_chapter.py 05 --widget GaussianBoundaries
```

规划草案（大纲见 `curriculum/outline.yaml` 中 id "05"）：

- **情景**：同一家医院。第 04 章直接学 $P(y\mid x)$；这次反过来，先描述“需要复查的人的指标长什么样”，即 $p(x\mid y)$，再用贝叶斯公式翻转。可复用第 01 章的贝叶斯分类器。
- **交互组件**：
  - `GaussianBoundaries`：拖动两类高斯的均值与协方差椭圆，同时画出 LDA（直线）、QDA（二次曲线）和逻辑回归的边界。
  - `GenVsDisc`：训练样本量滑块，朴素贝叶斯与逻辑回归的测试误差学习曲线交叉（Ng 与 Jordan 2001，已在第 04 章拓展中预告）。
  - `NaiveBayesCounts`：一个小的文本或症状计数表，拉普拉斯平滑参数 α 滑块，显示零概率问题。
- **推导**：共享协方差 ⇒ 线性边界；不同协方差 ⇒ 二次边界；LDA 后验恰为逻辑回归形式；朴素贝叶斯的 MLE 与贝叶斯估计（拉普拉斯平滑 = Dirichlet 先验）。
- **练习**：推导题（LDA 的 $w = \Sigma^{-1}(\mu_1-\mu_0)$、偏置、拉普拉斯平滑公式、高斯朴素贝叶斯的对数后验）；implement（高斯判别分析 + 多项式朴素贝叶斯，trace 画边界）；library（`LinearDiscriminantAnalysis`、`QuadraticDiscriminantAnalysis`、`GaussianNB`、`MultinomialNB` 与逻辑回归在小样本上的对比）。
- 之后是第 06 章（kNN、kd 树、核密度、样条与 GAM、维数灾难），完成后第二部分补齐。

### 第 04 章中发现并修复的引擎问题（写新章时注意）

- 动画课程的 Morph：时间线先 `applyState()` 再 `enterStep()`，旧代码因此找不到上一拍，实际退化成重写。已改为按 DOM 顺序找上一个已显示的拍。拍 Morph 幻灯片时，要看变形**中途**的帧。
- KaTeX 的根号是绝对定位的 SVG；`.course .mj-g { display: inline-block }` 会把它挪到被开方数上面。已用 `.course svg.mj-g { display: block }` 修复。
- 同一道代码题的多条 trace 可以记录不同的 metrics；TracePlayer 会从第一条含该指标的 trace 取值，MetricChart 跳过非有限值（d3 比例尺对 NaN 返回 undefined，会让整个组件崩溃）。
- `Plot2D` 新增 `levels`（固定等高线，例如概率 0.5/0.75/0.9 对应的 logit）和 `overlay`（在数据坐标上叠加任意 SVG）。

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
cd site && npx tsc --noEmit -p . && npx vitest run && npx astro build      # 85 tests
cd .. && python -m pytest -q tests && python3 scripts/build_outline.py --check   # 42 tests
cd site && node scripts/verify/course.mjs walk zh <slug> && node scripts/verify/course.mjs walk en <slug>
node scripts/verify/chapter.mjs <slug> zh dark     # 需要本地 Pyodide
node scripts/verify/pages.mjs <slug>
```

## 6. 压缩上下文用的关键词 / Context-compression keywords

Learning_ML · BOIIIKa/Learning_ML · boiiika.github.io/Learning_ML · 看得见的机器学习 / Visible ML · ISL + UDL(Prince) + 李航 · 重写不复制 · 中英双语 · 非商用 CC BY-NC-SA + MIT ·
7/30 章完成（00 数学工具、01 三要素、02 泛化、03 线性回归、04 线性分类、07 SVM、08 树与集成）· 下一步 05 生成式 → 06 非参数 ·
竖版讲义 = 蓝图风格（不改）· 横版动画课程 = 3B1B / manim 自动播放 · 六段结构（情景 / 方法 / 数学 / 例题拓展 / 课后训练 / 实战手册）+ 一页纸 recap ·
Astro 7 + React 19 + MDX + KaTeX 0.16.47 + Pyodide 0.29.3 · src/manim（Clock、Write、Morph、Indicate、Circumscribe、createSvg）· drive DSL（s1=11/4、s1=80%、click:chip2、wait:1）· Step effect（write / fade / pop / focus / morph）· `<Morph>` 推导变形 ·
练习：推导题（mathjs 数值比对）+ implement / library（@blank、双击 Tab 填答案、Trace 回放 classifier2d、field=f(x)−y）·
skills：new-chapter、course-deck、widget、exercise、verify-site · `scripts/new_chapter.py` + `templates/chapter` + `TODO(scaffold)` 守卫测试 · `site/scripts/verify`（chapter / course walk|film / pages）·
hydration 规则（f1 取整、r6 数据、SSR 中不写多值 style）· MDX 规则（多行 `$$`、正文用 \lt 不用裸 <）·
分支 claude/charming-knuth-ti0cof · PR #1 #2 #4 已合并，#5（第 04 章）待合并 · 只在用户同意时合并 · 本地 Pyodide 软链接与 venv 位于 scratchpad
