# 交接文档 / Handoff

> 写于 2026-10-14，2026-10-15 更新（转为本地开发；第 18 章完成），供上下文压缩或换人之后继续工作使用。先读本文，再读 `CLAUDE.md` 和 `.claude/skills/`。
> Written 2026-10-14 for picking the work up after a context reset. Read this, then `CLAUDE.md` and `.claude/skills/`.

## 1. 现状一句话

**完成 19/30 章（00–18）。00–17 已合并并部署到 https://boiiika.github.io/Learning_ML/，第 18 章在 PR 中等待用户同意合并。** 本地开发先读 [`LOCAL_DEV.md`](LOCAL_DEV.md)。下一步：第 19 章“卷积网络与残差网络”（slug `cnn-resnet`）。用户要求：**在保证准确性的前提下尽量精简编辑次数，持续推进**（每章只做一轮验证：单元测试 + pytest + 讲义页 Pyodide 检查 + 两种语言的课程 walk + 看组件截图）。

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
| BOIIIKa/Learning_ML#11 | 交接文档；第 16 章（优化） | 已合并 |
| BOIIIKa/Learning_ML#12 | 第 17 章（反向传播与初始化）；Pyodide 练习可用 autograd | 已合并 |
| BOIIIKa/Learning_ML#13 | 本地开发工具与指南（setup 脚本、requirements-dev.txt、Playwright 开发依赖、水合与 Math 差异检查工具、LOCAL_DEV.md） | 见 PR |
| 第 18 章 PR（基于 #13 的分支 `claude/chapter-18-dl-generalization-odq0wf`） | 第 18 章（正则化与泛化之谜）；Python 预计算实验的模式 | 见 PR |

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
- 工作分支：云端会话用 `claude/charming-knuth-ti0cof`；本地开发自己开功能分支。每次 PR 合并后从 `main` 重新开分支，再开新的 PR。

## 3. 下一步：第 19 章（卷积网络与残差网络，slug `cnn-resnet`）

`curriculum/outline.yaml` 中第 19 章承诺的内容：
- **目标**：从平移等变性推出卷积，理解感受野随深度增长；理解残差连接与归一化层为何使深层网络可训练。
- **推导**：卷积 = 与平移交换的线性算子；残差网络的梯度路径展开与“浅网络集成”视角。
- **组件**：卷积核滑动（可编辑核值，步长、填充、空洞）；感受野生长（点选深层单元，高亮输入上的感受野）。

建议做法：
- 引擎 `site/src/engines/cnn-resnet.ts`：卷积、感受野都是整数运算，不涉及水合问题。
- 第 18 章的 `LossLandscape` 已经展示了残差让 20 层网络的地形变平滑，可以在讲义里回引；第 17 章的 `varianceProfile` 可以加一个残差版本，比较有无残差时各层梯度的大小。
- 浏览器里训练卷积网络太慢：照第 18 章的做法，在 `scripts/precompute_ch18.py` 那样的脚本里用 numpy 预先训练，提交 JSON 数据，由 vitest 检查形状和讲义里的每个结论，讲义里注明数据是预先算好的。

```bash
python3 scripts/new_chapter.py 19 --widget ConvSlide
```

### 第 18 章的经验

- **实验结论要跨随机种子确认**：损失地形（3 个种子）、顿悟（权重衰减 1 和 0 各 3 个种子）的结论都在 vitest 里逐个种子断言。小网络上“小批量更平”成立，但测试准确率几乎一样，于是讲义就这样写，而不是声称平坦带来更好的泛化。
- **正则化并不总能提高测试表现**：练习里的双月数据上，Dropout / 早停不稳定地优于不加正则，所以检查只断言稳健的性质（权重范数变小、早停返回更早的轮次、准确率 > 0.8）。
- **ReLU 在 0 处的折点会让有限差分出错**：偏置初始化为 0 时，某行输入全被 Dropout 丢掉，预激活恰好是 0。梯度检查前把偏置设成非零随机数。
- **早停与岭回归的对应 α = 1/(ηt) 只是近似**：两条路径几乎重合（最远相距不到 ‖w*‖ 的 1/10），但同一时刻对应的两点最多差 ‖w*‖ 的 1/4。说明文字要写清楚是哪一种“接近”。
- **深色主题下“颜色越深”可能反了**：用 `fillOpacity` 叠加 `--accent` 的热图，在深色背景上损失大的地方更亮。写“颜色越浓”。

### 第 07–16 章期间新踩到的坑（写新章时注意）

- **浏览器的 `Math.cos` / `Math.sin` 与 Node 在末位可能不同**（`tanh`、`exp`、`log` 一致）。不稳定的训练过程（第 16 章峰值学习率 0.3）会把这一位差异放大成完全不同的曲线，导致 hydration 报错。办法：初始权重用 `r6` 取整，余弦用 `optimization.ts` 里的 `detCos`（只用四则运算）。

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

## 4. 开发环境

**本地开发**（用户从第 18 章起改在自己电脑上开发）：全部步骤、检查工具和心得都在 [`LOCAL_DEV.md`](LOCAL_DEV.md)。一次性设置是 `bash scripts/setup_local.sh`（Windows 用 `scripts/setup_local.ps1`）。

**云端沙箱**（Claude Code on the web）的特殊情况：
- 沙箱访问不了 jsDelivr，需要本地 Pyodide：`bash scripts/fetch_pyodide.sh`，再用 `PUBLIC_PYODIDE_URL=/pyodide/` 启动开发服务器；构建或提交前删掉 `site/public/pyodide` 链接。
- Playwright 浏览器预装在 `/opt/pw-browsers`（`PLAYWRIGHT_BROWSERS_PATH`），不要运行 `playwright install`。
- 访问不了 github.io：用 Actions 中 `pages.yml` 的运行状态确认部署。
- GitHub 操作只能用 MCP 的 github 工具（没有 `gh`）。
- Astro 7 的开发服务器以守护进程运行，用 `npx astro dev stop` 停止。

## 5. 推送前检查清单

```bash
cd site && npx tsc --noEmit -p . && npx vitest run && npx astro build      # 228 tests, 82 pages
cd .. && python -m pytest -q tests && python3 scripts/build_outline.py --check   # 114 tests
cd site && node scripts/verify/course.mjs walk zh <slug> && node scripts/verify/course.mjs walk en <slug>
node scripts/verify/chapter.mjs <slug> zh dark     # 本地默认从 CDN 加载 Pyodide
node scripts/verify/hydration.mjs /zh/chapters/<slug>/   # 出现水合报错时定位组件
node scripts/verify/pages.mjs <slug>
```

## 6. 压缩上下文用的关键词 / Context-compression keywords

Learning_ML · BOIIIKa/Learning_ML · boiiika.github.io/Learning_ML · 看得见的机器学习 / Visible ML · ISL + UDL(Prince) + 李航 · 重写不复制 · 中英双语 · 非商用 CC BY-NC-SA + MIT ·
19/30 章完成（00–18；18 在 PR 中）· 本地开发见 docs/LOCAL_DEV.md · 下一步 19 卷积与残差（slug cnn-resnet） · 20 Transformer · 21 图网络（第五部分：网络架构与归纳偏置） ·
竖版讲义 = 蓝图风格（不改）· 横版动画课程 = 3B1B / manim · **点一下播完一页再暂停**（连播按钮 / A 键）· 翻页 = 新旧两页同时滑动 · 按钮：上一页 / 上一拍 / 播放 / 下一拍 / 下一页 / 重播本页 / 连播 / 倍速 / 页面总览 / 全屏 · 朗读已注释掉 · 学生拖动控件时 drive 脚本停下（takeover）· 课程忽略 prefers-reduced-motion ·
六段结构（情景 / 方法 / 数学 / 例题拓展 / 课后训练 / 实战手册）+ 一页纸 recap · 情景人物：老师和学生（第 01–03、14、15 章）、医院初筛（第 07 章）·
Astro 7 + React 19 + MDX + KaTeX + Pyodide 0.29.3（sklearn 1.7）· src/manim（Clock、Write、Morph、createSvg）· drive DSL（s1=11/4、s1=80%、click:chip2、wait:1）· `<Morph>` 推导变形 ·
引擎：mlp.ts（前向 + 反向传播）· neural-networks.ts · losses-as-likelihood.ts · 共享组件 Widget / Slider / Stat / CurvePlot（useId clip）/ Plot2D / TracePlayer ·
练习：4 道推导题（mathjs 数值比对）+ implement / library（@blank、双击 Tab、Trace 回放 classifier2d）· `python3 scripts/build_exercises.py` ·
skills：new-chapter、course-deck、widget、exercise、verify-site · `scripts/new_chapter.py <id> --widget X` · `TODO(scaffold)` 守卫测试 ·
hydration 规则（f1 取整、r6 数据、SSR 不写多值 style、组件里不用 table、SVG id 唯一）· MDX 规则（多行 `$$`、正文用 \lt）·
分支 claude/charming-knuth-ti0cof（每次合并后从 main 重开）· PR #1–#10 已合并 · 只在用户明确同意时合并 · 本地 Pyodide 软链接与 venv 在 scratchpad（容器回收后需重建）
