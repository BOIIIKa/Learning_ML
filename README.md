# Learning_ML

**看得见的机器学习：从严谨推导到科研前沿**
**Visible Machine Learning: From Rigorous Derivations to the Research Frontier**

一套中英双语、交互可视化的机器学习课程，面向有数学基础、希望快速具备科研能力的学生。
A bilingual, interactive-visual machine learning course for mathematically prepared students who want to build research ability quickly.

在线阅读 / Live: **https://boiiika.github.io/Learning_ML/** —— 每章有竖版讲义（复习）和 3Blue1Brown 式自动播放的横版动画课程（初学）。
Each chapter comes as notes (for review) and a self-playing, 3Blue1Brown-style animated course (for first-time learners).

项目总结 / Project summary: [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)

## 课程大纲 / Outline

- 中文：[curriculum/OUTLINE.zh.md](curriculum/OUTLINE.zh.md)
- English: [curriculum/OUTLINE.en.md](curriculum/OUTLINE.en.md)

主线参考书 / Main reference books:
*An Introduction to Statistical Learning* · *Understanding Deep Learning* (Prince) · 《统计学习方法》（李航）。
本项目只引用其章节结构，所有讲解与图示均为重新创作。
We cite only their chapter structure; all explanations and figures are original.

## 设计文档 / Design docs

- [教材评估与选书 / Textbook evaluation](docs/textbook-selection.md)
- [课程结构设计 / Curriculum design](docs/curriculum-design.md)
- [章节结构与练习系统 / Chapter structure & exercises](docs/exercises-design.md)

## 本地运行 / Run locally

```bash
cd site
npm install
npm run dev        # http://localhost:4321 （会先运行 scripts/build_exercises.py，需要 PyYAML）
npm test           # 算法引擎与判题单元测试 / engine & grading unit tests
pytest ../tests    # 练习：参考解答全过、骨架全不过 / exercises: solutions pass, skeletons fail
```

## 写一章 / Writing a chapter

```bash
python3 scripts/new_chapter.py 04        # 按模板生成一章的全部文件 / scaffold every file of chapter 04
cd site && npm run verify:chapter -- linear-classification   # 浏览器检查 / browser check (Playwright)
```

流程、质量标准和常见坑见 [`CLAUDE.md`](CLAUDE.md) 与 [`.claude/skills/`](.claude/skills)（new-chapter、course-deck、widget、exercise、verify-site）。
Workflow, quality bar and pitfalls: [`CLAUDE.md`](CLAUDE.md) and [`.claude/skills/`](.claude/skills).

## 贡献 / Contributing

大纲的唯一数据源是 `curriculum/outline.yaml`，修改后运行：
The outline's single source of truth is `curriculum/outline.yaml`; after editing run:

```bash
python3 scripts/build_outline.py
```

## 许可 / License

内容 / Content: [CC BY-NC-SA 4.0](LICENSE-CONTENT.md) · 代码 / Code: [MIT](LICENSE)
