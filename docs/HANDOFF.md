# 交接文档 / Handoff

> 写于 2026-10-14（当天收工时更新），供上下文压缩或换人之后继续工作使用。先读本文，再读 `CLAUDE.md` 和 `.claude/skills/`。
> Written 2026-10-14 for picking the work up after a context reset. Read this, then `CLAUDE.md` and `.claude/skills/`.

## 1. 现状一句话

**完成 16/30 章（00–15），全部已合并到 `main` 并部署到 https://boiiika.github.io/Learning_ML/。** 没有待合并的 PR。下一步：第 16 章“优化：从梯度下降到 Adam”。用户要求：**在保证准确性的前提下尽量精简编辑次数，持续推进**（每章只做一轮验证：单元测试 + pytest + 讲义页 Pyodide 检查 + 两种语言的课程 walk + 看组件截图）。

- 竖版讲义：工程蓝图风格。
- 横版动画课程：3Blue1Brown / manim 风格，自动播放。
- 写新章的工作流已沉淀为 skills、模板和脚手架。

| PR | 内容 | 状态 |
|---|---|---|
| BOIIIKa/Learning_ML#1 | 网站骨架；第 00、07、08 章；练习系统 | 已合并 |
| BOIIIKa/Learning_ML#2 | 蓝图风格；第 01–03 章 | 已合并 |
| BOIIIKa/Learning_ML#4 | 3B1B 式动画课程引擎；前端优化；skills、模板、项目总结 | 已合并 |
| BOIIIKa/Learning_ML#5 | 交接文档；第 04 章；Morph 播放与 KaTeX 根号的修复 | 已合并 |
| BOIIIKa/Learning_ML#6 | 第 05、06、09–13 章 | 已合并 |
| BOIIIKa/Learning_ML#7 | 动画课程：点一下播完一页再暂停；滑块被脚本抢回、reduced-motion 无动画等修复；讲义导航、手机布局、Pyodide 预热 | 已合并 |
| BOIIIKa/Learning_ML#8 | 翻页滑动动画、上一页 / 下一页 / 重播 / 页面总览按钮、滚轮与触屏翻页；朗读注释掉；尺寸优化；暂停时手动翻看动画冻结的修复 | 已合并 |
| BOIIIKa/Learning_ML#9 | 第 14 章；通用引擎 `mlp.ts` | 已合并 |
| BOIIIKa/Learning_ML#10 | 第 15 章；CurvePlot clipPath id 重复的修复 | 已合并 |

## 2. 用户（项目负责人）的偏好与约定

- 用中文交流。网站是中英双语，学生来自不同国家。
- **非商用，完全开源**。不得复制原书文字或插图，只引用章号。
- 受众：有数学基础、想快速具备科研能力的学生。要求推导严谨，并跟进前沿。
- 每章两种形式：
  - **竖版讲义**用于复习。用户对现有风格和内容很满意，**不要改风格**。
  - **横版动画课程**用于初学。要像 3B1B：节奏适中，突出 manim 式动画。**播放方式：点一下播完当前这一页的动画和旁白，然后暂停，留给学生思考；再点进入下一页**（“连播”按钮或 A 键可以一直播下去）。学生拖动滑块时，脚本不能把值抢回去。翻页要顺滑（新旧两页同时滑动），有上一页 / 下一页 / 重播本页 / 页面总览按钮；朗读功能暂时注释掉了。
- 视觉：工程草稿 / 蓝图风格，配色克制，有少量网格线，动效要能抓住注意力（用户的部分美国学生不喜欢读书）。
- 练习：
  - 分为“推导理解”和“训练实战”两类；训练实战又分“补全算法”和“组装库”（例如 LightGBM）。
  - 训练过程要可视化，例如每次更新后超平面的移动。
  - 必须有参考答案；连按两次 Tab 填入答案，填充动效要酷。
- 章节顺序由我决定。**只有用户明确同意时才合并 PR**；用户会把网站链接发给学生测试。
- 工作分支：`claude/charming-knuth-ti0cof`。每次 PR 合并后，从 `main` 重新开这个分支，再开新的 PR（草稿）。

## 3. 下一步：第 16 章（优化：从梯度下降到 Adam，slug `optimization`）

`curriculum/outline.yaml` 中第 16 章承诺的内容：
- **目标**：步长、条件数与收敛速度的关系；SGD、动量、Nesterov、Adam / AdamW 的更新规则；非凸地形中的鞍点、平坦区与学习率调度。
- **推导**：强凸光滑函数上 GD 的线性收敛 $(1-\mu/L)^t$；动量作为带阻尼的重球方程；Adam 的偏差修正、AdamW 的解耦权重衰减。
- **组件**：优化器赛跑（Rosenbrock / 鞍点 / 病态二次型，可调学习率与起点）、条件数与之字形、学习率调度（常数 / 阶梯 / 余弦 / 预热）。

建议做法：
- 引擎 `site/src/engines/optimization.ts` 写纯函数的 `gd / momentum / nesterov / rmsprop / adam / adamw` 步进器（输入梯度函数，返回轨迹），测试里验证二次型上的收敛率、Adam 偏差修正后第一步的大小 ≈ 学习率。
- 情景可以沿用老师的学生：用第 14 章的 `mlp.ts` 训练一个小网络，比较不同优化器的训练曲线（例如作为调度组件的数据）。
- 例题先在 Python 里算准：例如二次型 $f = \frac12(x^2 + \kappa y^2)$ 上最优步长 $2/(L + \mu)$ 的收敛因子 $(\kappa - 1)/(\kappa + 1)$。

```bash
python3 scripts/new_chapter.py 16 --widget OptimizerRace
```

可复用的引擎：
- `mlp.ts`：全连接层、ReLU / tanh、前向缓存、反向传播（有有限差分梯度检验）、激活模式、参数个数。
- `losses-as-likelihood.ts`：各种头的负对数似然和梯度。
- `neural-networks.ts`：铰链拟合、线性段计数、帐篷映射。
- `generalization.ts`：学生数据 `sample` 和 `truth`。
- `regression.ts`：`solve`。
- `data.ts`：`rng`、`gaussian`、`r6`、`moons`、`blobs`。

### 第 07–15 章期间新踩到的坑（写新章时注意）

- **SVG 的 id 必须每个实例唯一**：所有 CurvePlot 曾共用 `cp-clip`，同页高矮不同的图会被第一张图的尺寸裁掉。现在用 `useId()`；Astro 给每个 island 不同前缀，跨 island 也唯一。
- **scikit-learn 1.7**：`MLPRegressor` 的第一个位置参数是 `loss`，层数要写 `hidden_layer_sizes=(…)`。
- **讲义页第一次 `chapter.mjs` 常报 `code exercises: 0`**（冷编译还没渲染完）：直接再跑一次。
- **测线上行为要用生产构建**：`SITE_BASE=/Learning_ML/ npx astro build`，把 `dist` 软链到 `$SP/srv/Learning_ML`，再用 `python3 -m http.server 8080` 服务，`SITE_URL=http://127.0.0.1:8080/Learning_ML node scripts/verify/course.mjs walk …`。
- **组件文字里的每个断言都要有单元测试**（例如“覆盖率 100% / 91%”“段数 ≤ D + 1”），截图一定要看：两个 bug（clip id、公式错位）都是看截图才发现的。

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
  - 没设 `PUBLIC_PYODIDE_URL` 时会去 CDN 取 Pyodide，沙箱里代码题会一直转圈（`chapter.mjs` 超时）。新加的练习若显示 `code exercises: 0`，是开发服务器还拿着旧的 `generated/exercises.json`：`npx astro dev stop; rm -rf node_modules/.vite .astro` 后重启。
  - **构建或提交前删除这个软链接**（已加入 .gitignore，但构建时会把它复制进 dist）。
- Astro 7 的 dev 服务器会以守护进程运行，用 `npx astro dev stop` 停止。`astro build` 之后 dev 服务器的 Vite 缓存可能失效，需要 `rm -rf node_modules/.vite` 后重启。
- Playwright：`/opt/node-tools/node_modules/playwright`，`site/scripts/verify/_browser.mjs` 会自动找到它。
- github.io 在沙箱里访问不了。部署结果用 GitHub Actions 中 `pages.yml` 的运行状态确认。
- GitHub 操作只能用 MCP 的 github 工具（没有 `gh`），推送用 `git push -u origin claude/charming-knuth-ti0cof`。

## 5. 推送前检查清单

```bash
cd site && npx tsc --noEmit -p . && npx vitest run && npx astro build      # 185 tests, 70 pages
cd .. && python -m pytest -q tests && python3 scripts/build_outline.py --check   # 96 tests
cd site && node scripts/verify/course.mjs walk zh <slug> && node scripts/verify/course.mjs walk en <slug>
node scripts/verify/chapter.mjs <slug> zh dark     # 需要本地 Pyodide
node scripts/verify/pages.mjs <slug>
```

## 6. 压缩上下文用的关键词 / Context-compression keywords

Learning_ML · BOIIIKa/Learning_ML · boiiika.github.io/Learning_ML · 看得见的机器学习 / Visible ML · ISL + UDL(Prince) + 李航 · 重写不复制 · 中英双语 · 非商用 CC BY-NC-SA + MIT ·
16/30 章完成（00–15），全部已合并 · 下一步 16 优化（GD → Adam，slug optimization）· 然后 17 反向传播 … 21 Transformer，第五部分起是生成模型 ·
竖版讲义 = 蓝图风格（不改）· 横版动画课程 = 3B1B / manim · **点一下播完一页再暂停**（连播按钮 / A 键）· 翻页 = 新旧两页同时滑动 · 按钮：上一页 / 上一拍 / 播放 / 下一拍 / 下一页 / 重播本页 / 连播 / 倍速 / 页面总览 / 全屏 · 朗读已注释掉 · 学生拖动控件时 drive 脚本停下（takeover）· 课程忽略 prefers-reduced-motion ·
六段结构（情景 / 方法 / 数学 / 例题拓展 / 课后训练 / 实战手册）+ 一页纸 recap · 情景人物：老师和学生（第 01–03、14、15 章）、医院初筛（第 07 章）·
Astro 7 + React 19 + MDX + KaTeX + Pyodide 0.29.3（sklearn 1.7）· src/manim（Clock、Write、Morph、createSvg）· drive DSL（s1=11/4、s1=80%、click:chip2、wait:1）· `<Morph>` 推导变形 ·
引擎：mlp.ts（前向 + 反向传播）· neural-networks.ts · losses-as-likelihood.ts · 共享组件 Widget / Slider / Stat / CurvePlot（useId clip）/ Plot2D / TracePlayer ·
练习：4 道推导题（mathjs 数值比对）+ implement / library（@blank、双击 Tab、Trace 回放 classifier2d）· `python3 scripts/build_exercises.py` ·
skills：new-chapter、course-deck、widget、exercise、verify-site · `scripts/new_chapter.py <id> --widget X` · `TODO(scaffold)` 守卫测试 ·
hydration 规则（f1 取整、r6 数据、SSR 不写多值 style、组件里不用 table、SVG id 唯一）· MDX 规则（多行 `$$`、正文用 \lt）·
分支 claude/charming-knuth-ti0cof（每次合并后从 main 重开）· PR #1–#10 已合并 · 只在用户明确同意时合并 · 本地 Pyodide 软链接与 venv 在 scratchpad（容器回收后需重建）
