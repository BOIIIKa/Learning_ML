# Learning_ML

**看得见的机器学习：从严谨推导到科研前沿**
**Visible Machine Learning: From Rigorous Derivations to the Research Frontier**

一套中英双语、交互可视化的机器学习课程，面向有数学基础、希望快速具备科研能力的学生。
A bilingual, interactive-visual machine learning course for mathematically prepared students who want to build research ability quickly.

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

## 贡献 / Contributing

大纲的唯一数据源是 `curriculum/outline.yaml`，修改后运行：
The outline's single source of truth is `curriculum/outline.yaml`; after editing run:

```bash
python3 scripts/build_outline.py
```

## 许可 / License

拟定 / Proposed: 内容 / Content CC BY-NC-SA 4.0 · 代码 / Code MIT（LICENSE 文件待添加 / LICENSE files pending）
