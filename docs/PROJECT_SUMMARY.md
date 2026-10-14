# 项目总结 / Project Summary

> 截至 2026-10-14。面向接手的老师、学生贡献者和 AI 协作者。
> As of 2026-10-14, for the teachers, student contributors and AI collaborators who pick this up next.

## 1. 这是什么

**看得见的机器学习**：一套中英双语、完全开放、非商用的机器学习课程。

- 以 ISL/ISLP、Prince《Understanding Deep Learning》、李航《统计学习方法》为主线，只引用章号，所有讲解与图示均为重新创作。
- 面向有数学基础、希望快速具备科研能力的学生。
- 网址：https://boiiika.github.io/Learning_ML/

每章有两种形式：

| 形式 | 适合 | 特点 |
|---|---|---|
| **竖版讲义** `/chapters/<slug>/` | 复习、查阅、打印 | 工程蓝图风格。六段结构：情景 → 方法 → 数学 → 例题与拓展 → 课后训练 → 实战手册。正文中嵌入可拖动的交互图形 |
| **横版动画课程** `/course/<slug>/` | 第一次学 | 3Blue1Brown 式自动播放。公式先描边再填充（Write），推导原地变形（TransformMatchingTex），旁白字幕可朗读，动画自己拖动图中的滑块、点按钮 |

## 2. 当前进度

| 章 | 讲义 | 动画课 | 一页纸 | 交互组件 | 练习（推导 + 代码） | 实战手册 |
|---|---|---|---|---|---|---|
| 00 数学工具 | ✓ | ✓ | ✓ | 5 | 4 + 2 | — |
| 01 学习问题与三要素 | ✓ | ✓ | ✓ | 4 | 4 + 2 | — |
| 02 泛化与模型选择 | ✓ | ✓ | ✓ | 4 | 4 + 2 | 模型选择 |
| 03 线性回归与正则化 | ✓ | ✓ | ✓ | 4 | 4 + 2 | 线性模型 |
| 04 线性分类：感知机、逻辑回归与最大熵 | ✓ | ✓ | ✓ | 4 | 4 + 2 | 逻辑回归 |
| 05 生成式分类：朴素贝叶斯与判别分析 | ✓ | ✓ | ✓ | 3 | 4 + 2 | 生成式分类 |
| 06 非参数方法与基函数扩展 | ✓ | ✓ | ✓ | 4 | 4 + 2 | 非参数方法 |
| 09 结构化预测：HMM 与 CRF | ✓ | ✓ | ✓ | 3 | 4 + 2 | 序列标注 |
| 10 聚类 | ✓ | ✓ | ✓ | 3 | 4 + 2 | 聚类 |
| 11 EM 算法与混合模型 | ✓ | ✓ | ✓ | 3 | 4 + 2 | 高斯混合 |
| 12 矩阵分解与降维 | ✓ | ✓ | ✓ | 3 | 4 + 2 | 降维 |
| 13 马尔可夫链方法：MCMC、PageRank 与主题模型 | ✓ | ✓ | ✓ | 3 | 4 + 2 | MCMC 与主题模型 |
| 07 支持向量机与核方法 | ✓ | ✓ | ✓ | 5 | 4 + 2 | SVM |
| 08 决策树与集成学习 | ✓ | ✓ | ✓ | 5 | 4 + 2 | LightGBM |

合计：30 章中完成 14 章（00–13），共有 53 个交互组件、564 拍动画（每种语言）、56 道推导题、28 道代码题。测试有 165 项前端单元测试，以及 84 项练习测试（参考答案全部通过，骨架全部不通过）。

## 3. 架构一览

```
curriculum/outline.yaml ──► OUTLINE.{zh,en}.md、网站目录、章节元数据（目标、三要素、前沿）
exercises/chNN/*.yaml|py ──► scripts/build_exercises.py ──► site/src/generated/exercises.json
python/learnml (blanks · checking · Trace)  ← 同一份代码跑在 CPython (pytest) 和 Pyodide（浏览器）

site/ (Astro 7 + React 19 + MDX)
├─ content/   竖版讲义 MDX        ├─ slides/   动画课程 MDX（Slide / Step / Morph）
├─ recaps/    一页纸              ├─ handbook/ 实战手册
├─ engines/   纯 TS 算法 + 测试   ├─ components/chapters/chNN  交互组件
├─ manim/     动画层：时钟、Write、Morph、Indicate、Create、drive 脚本
└─ components/course/controller.ts  自动播放时间线
```

关键设计：

- **一个交互组件 = 一个想法**。算法在 `engines/` 中用纯 TypeScript 实现并写单元测试，组件只负责画。
- **动画课程是一条时间线**。每一拍都要等三件事完成后才进入下一拍：入场动画、旁白（按字数估算时长，或等语音读完）、`drive` 脚本（例如 `s1=11/4; click:chip2`，像 manim 的 ValueTracker 那样操作组件）。
- **练习的两类**：
  - 推导题：输入表达式，在随机取值上与参考答案做数值比对，所以任何等价写法都算对。
  - 代码题：“补全算法”与“组装库”两种。连按两次 Tab 填入参考答案；运行后可以回放训练过程，例如超平面怎样一步步移动。
- **两套主题**：讲义用蓝图风格（深色是蓝图纸，浅色是绘图纸）；动画课程用 manim 配色（黑底 + 3B1B 的蓝、黄、红、绿）。组件只使用 CSS 变量，所以三种配色下都能直接用。

## 4. 质量标准（已沉淀为 skills）

写剩下 24 章时，请先读 `CLAUDE.md` 和 `.claude/skills/`：

| Skill | 内容 |
|---|---|
| `new-chapter` | 一章从规划到上线的完整流程与验收标准 |
| `course-deck` | 动画课程写法：拍、旁白节奏、Morph 推导、drive 脚本 |
| `widget` | 组件与算法引擎、水合（hydration）规则、主题变量、可被驱动的控件 |
| `exercise` | 推导题与 Python 练习、@blank 语法、检查与回放 |
| `verify-site` | 本地运行、Pyodide、Playwright 检查脚本、推送前清单 |

脚手架：`python3 scripts/new_chapter.py <id>` 会按模板生成一章的全部文件，文件中带有 `TODO(scaffold)` 标记。只要还有标记没处理，测试就不会通过，所以写到一半的章节不会被误合并。

开发中踩过的坑（详见各 skill）：

- 服务端与客户端渲染不一致（hydration）：SVG 坐标要取整，数据要用固定种子生成，不能在 SSR 的 style 里写多值属性。
- MDX 中的 `$$` 必须独占多行，正文里不能出现裸的 `<` 和 `{`。
- 蒙特卡洛估计自身也有噪声：第 02 章的偏差² 需要扣掉 Var/reps，否则高次多项式时会虚高。
- 例题里的每个数字都要在代码里验算一遍。

## 5. 后续路线

2. **第三部分，第 09–13 章**：结构化预测、聚类、EM、矩阵分解、马尔可夫链。第 11 章 EM 的 ELBO 要为第 24 章 VAE 和第 25 章扩散模型埋下伏笔。
3. **第四、五部分，第 14–21 章**：深度学习。这部分需要一个小型的浏览器内自动微分引擎（放在 `engines/` 里），用于反向传播和优化器的可视化。
4. **第六至九部分，第 22–29 章**：生成模型、强化学习、大模型综述、实验方法、负责任的机器学习。前沿阅读要每学期更新一次。

可选改进：

- 如果学生反映 jsDelivr 访问慢，可以把 Pyodide 自托管到 Pages。
- 加入学生反馈入口，例如在每章末尾链接 GitHub Discussions。
- 为动画课程录制真人或高质量 TTS 配音，替换浏览器自带的朗读。

---

## English summary

A bilingual, open, non-commercial ML course. Each chapter ships as blueprint-styled **notes** (for
review) and a 3Blue1Brown-style **animated course** that plays itself: manim-like Write and
TransformMatchingTex animations, narration subtitles (optionally spoken), and drive scripts that
operate the interactive figures.

Fourteen of 30 chapters are done: 00–13. They contain 53 widgets, 564 animated beats per language,
56 derivation drills and 28 in-browser Python exercises, tested by 165 unit tests and 84 exercise tests. The workflow, quality bar and pitfalls are captured in `CLAUDE.md`,
`.claude/skills/*` and `templates/chapter/`, and scaffolded by `scripts/new_chapter.py`.

Next steps:
- Parts III–IX follow the order in `curriculum/outline.yaml`.
- Deep learning (chapters 14–21) needs a small in-browser autodiff engine first.
