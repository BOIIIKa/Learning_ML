# 章节结构与练习系统设计

> 前置文档：[课程结构设计](curriculum-design.md)。本文记录第 07 章试点中确定下来的章节结构、练习形态与技术方案。

## 1. 每章固定的六个部分

| # | 部分 | 内容 | 实现位置 |
|---|---|---|---|
| 1 | 情景引入 | 一个具体问题，配一个让学生“先动手、后讲理”的交互组件 | `site/src/content/<lang>/<slug>.mdx` |
| 2 | 方法 | 用三要素（模型–策略–算法）引出方法，配可拖拽的交互组件 | 同上 |
| 3 | 数学工具与推导 | 先回顾所需工具（定义 / 定理框），再给出完整推导，并配算法回放 | 同上 |
| 4 | 案例与拓展 | 能手算的例题，回到情景中的真实数据，再拓展到更多情景 | 同上 |
| 5 | 课后训练 | 推导理解 + 训练实战（补全算法 / 组装库训练），并配训练过程可视化 | `exercises/chXX/`，由页面模板自动渲染 |
| 6 | 实战手册与前沿 | 链接到独立的《实战手册》条目，并列出前沿文献 | `site/src/handbook/<lang>/*.mdx` 与 `curriculum/outline.yaml` |

第 1–4 部分由作者用 MDX 撰写，每部分用 `<Section k="scenario|method|math|cases">` 包裹。第 5、6 部分由章节模板根据数据自动生成，保证每章结构一致。

## 2. 两种呈现形式

同一章内容有两种呈现，两者共用同一套交互组件和练习：

| | 横版动画课程（初学推荐） | 竖版讲义（复习推荐） |
|---|---|---|
| 地址 | `/<lang>/course/<slug>/` | `/<lang>/chapters/<slug>/` |
| 源文件 | `site/src/slides/<lang>/<slug>.mdx` | `site/src/content/<lang>/<slug>.mdx` |
| 形式 | 16:9 舞台，一屏一个想法；`<Step>` 逐步揭示，可配字幕与朗读 | 完整推导与六段式结构，可滚动阅读 |
| 交互 | ← → / 空格 / 滑动翻页，P 自动播放，F 全屏，按“部分”分段的进度条 | 侧栏目录、滚动渐显、可展开解答 |
| 练习 | 最后一页链接到讲义中的练习 | 完整练习：推导题与代码题 |

两者的末尾都会显示“本章一页纸”（`site/src/recaps/<lang>/<slug>.mdx`）：讲义中放在顶部，可折叠；课程中是最后一页。只要某章同时有中英两种语言的 deck，讲义页就会自动出现“动画课程”入口。

**写 deck 的约定**：
- 每页只讲一个想法，正文是短句，完整的话写进 `say`，它会作为字幕显示，也可以朗读；
- 交互组件放进 `<div slot="visual">`；
- 公式块使用多行的 `$$ … $$`。

## 3. 练习形态

### 推导理解（`derivation`）

- 学生输入一个表达式（mathjs 语法），实时显示 KaTeX 预览。
- **判题方式是数值等价**：在 `vars` 给定的取值范围内随机采样约 32 个点，比较学生答案与参考答案的值，所以 `(4*e*(1-e))^(M/2)` 与 `(2*sqrt(e*(1-e)))^M` 都判为正确。
- 每题都附有完整的参考解答，可以展开或收起。

### 训练实战 · 补全算法（`implement`）与组装库训练（`library`）

- **单一源文件**：参考解答就是一个可直接运行的 `.py` 文件，用标记圈出需要学生填写的部分：

  ```python
  # @blank eta
  # zh: 计算 η = K_ii + K_jj − 2K_ij
  # en: Compute η = K_ii + K_jj − 2K_ij
  return K[i, i] + K[j, j] - 2.0 * K[i, j]
  # @end
  ```

  `python/learnml/blanks.py` 从这个文件生成中英两份骨架（空白处替换为 `TODO` 注释加 `raise NotImplementedError`）、参考答案和空白列表。答案只维护一份，不会出现两边不一致。
- **判题脚本** `<name>_check.py` 定义 `check(student, t)`，逐项测试并收集训练轨迹。同一份脚本既在 CPython（pytest）里跑，也在浏览器的 Pyodide 里跑。
- **CI 双向校验**（`tests/test_exercises.py`）：参考解答必须全部通过；中英两份骨架必须一项都不通过。这能防止出现“空着也能过”的题目。

### 参考答案的填充交互

- 光标位于某个空内时按一次 **Tab**，会浮出提示“再按一次 Tab 填入答案”；在 650 ms 内再按一次，参考答案会以打字机效果逐字填入，同时有渐变光效，结束时迸发一小簇粒子。
- 推导题的输入框也支持同样的双击 Tab。
- “填入全部答案”按钮会按顺序依次填充所有空。
- 全部测试通过时，结果区会弹出彩纸动画并发出绿色辉光；失败项会标红并显示断言信息。

## 4. 训练轨迹协议

手写算法与调库训练都输出同一种轨迹格式（`python/learnml/trace.py` ↔ `site/src/engines/trace.ts`）：

```json
{ "kind": "classifier2d", "title": {"zh": "…", "en": "…"}, "X": [[…]], "y": […],
  "grid": {"x": […], "y": […]},
  "steps": [{ "w": [w1, w2], "b": 0.1, "field": […], "weights": […],
              "highlight": [i, j], "support": […], "metrics": {"dual": 1.2}, "note": "…" }] }
```

`TracePlayer` 只认这一种格式，提供播放、单步、拖动时间轴和变速，并可以把两条轨迹并排同步播放（例如“你的 SMO”对比“libsvm”）。

库的内部过程不一定能拿到。scikit-learn 的 `SVC` 只能取到最终解，所以 SVM 的逐步过程来自学生自己写的 SMO；LightGBM 则可以用 `num_iteration=k` 拿到逐轮的结果。

## 5. 运行环境

- 浏览器内 Python 使用 **Pyodide 0.29.3**，内含 NumPy 2.2、scikit-learn 1.7、LightGBM 4.6。它运行在 Web Worker 中，不会卡住页面；死循环可以用 Stop 按钮终止。
- 默认从 jsDelivr CDN 加载 Pyodide；也可以用环境变量 `PUBLIC_PYODIDE_URL` 改为自托管地址，适合 CDN 访问不稳定的地区。
- 页面上的交互组件（SVM 拖拽、核函数、SMO 回放）使用 TypeScript 实现的同一算法（`site/src/engines/`），不依赖 Python，打开页面即可使用。这些实现有 vitest 测试（KKT 条件、经典例题）。

## 6. 新增一章的步骤

1. 新建 `exercises/chXX/exercises.yaml`，写推导题与代码题清单。
2. 每道代码题写一个带 `@blank` 标记的 `.py` 文件，以及对应的 `_check.py`。
3. 运行 `pytest tests/`：参考解答应全部通过，骨架应全部失败。
4. 写讲义 `site/src/content/{zh,en}/<slug>.mdx`（第 1–4 部分），章节页面会自动出现。
5. 写一页纸总结 `site/src/recaps/{zh,en}/<slug>.mdx`，以及动画课程 `site/src/slides/{zh,en}/<slug>.mdx`。
6. 如有需要，在 `site/src/handbook/{zh,en}/` 中添加实战手册条目，并在 frontmatter 的 `chapters` 中注明章号。


> 新增章节的最新流程见 [`.claude/skills/new-chapter/SKILL.md`](../.claude/skills/new-chapter/SKILL.md)，脚手架为 `scripts/new_chapter.py`。
> For the current chapter workflow see `.claude/skills/new-chapter/SKILL.md` and `scripts/new_chapter.py`.
