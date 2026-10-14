# 机器学习 / 统计学习教材评估与选书建议

> 目的：为“教材可视化”项目挑选底本。评价维度为 **前沿性**、**易读性**、**严谨性**（各 1–5 分），并额外考虑对本项目至关重要的 **可视化适配度** 与 **版权/许可**。
>
> 说明：评分是主观的综合判断；许可信息基于各书官网公开声明，正式使用前需再次核对原文。

## 1. 评分总表

### 1.1 英文教材

| 教材 | 版本/年份 | 定位 | 前沿 | 易读 | 严谨 | 可视化适配 | 免费获取 / 许可 |
|---|---|---|:-:|:-:|:-:|:-:|---|
| **Understanding Deep Learning** — Simon Prince | 2023 | 深度学习主教材 | 5 | 5 | 4 | **5** | 免费 PDF；CC BY-NC-ND |
| **Deep Learning: Foundations and Concepts** — Bishop & Bishop | 2024 | 深度学习（概率视角） | 5 | 4 | 4 | 4 | 官网可在线阅读；版权保留 |
| **Probabilistic Machine Learning: An Introduction / Advanced Topics** — Kevin Murphy | 2022 / 2023 | 百科全书式参考 | 5 | 2 | 4 | 3 | 免费 PDF；CC BY-NC-ND |
| **Dive into Deep Learning (D2L)** — Zhang, Lipton, Li, Smola | 持续更新 | 代码驱动实践 | 4 | 5 | 3 | 4 | 开源；文本 CC BY-SA 4.0，代码 MIT-0 |
| **An Introduction to Statistical Learning (ISLR2 / ISLP)** — James, Witten, Hastie, Tibshirani, Taylor | 2021 / 2023 | 统计学习入门 | 2 | **5** | 3 | **5** | 免费 PDF；版权保留 |
| **The Elements of Statistical Learning (ESL)** — Hastie, Tibshirani, Friedman | 2009 (2nd) | 统计学习经典 | 1 | 2 | 5 | 4 | 免费 PDF；版权保留 |
| **Pattern Recognition and Machine Learning (PRML)** — Bishop | 2006 | 贝叶斯视角经典 | 1 | 3 | 5 | 4 | 免费 PDF；版权保留 |
| **Deep Learning** — Goodfellow, Bengio, Courville | 2016 | 深度学习经典 | 2 | 3 | 4 | 3 | 免费 HTML；版权保留 |
| **Mathematics for Machine Learning** — Deisenroth, Faisal, Ong | 2020 | 数学预备 | 2 | 4 | 4 | **5** | 免费 PDF；版权保留 |
| **Understanding Machine Learning: From Theory to Algorithms** — Shalev-Shwartz & Ben-David | 2014 | 学习理论 | 2 | 3 | 5 | 2 | 免费 PDF（个人使用） |
| **Foundations of Machine Learning** — Mohri, Rostamizadeh, Talwalkar | 2018 (2nd) | 学习理论 | 2 | 2 | 5 | 2 | MIT Press 开放获取 |
| **Learning Theory from First Principles** — Francis Bach | 2024 | 现代学习理论 | 4 | 3 | 5 | 3 | 免费 PDF |
| **Patterns, Predictions, and Actions** — Hardt & Recht | 2022 | 现代视角的 ML 导论 | 4 | 4 | 4 | 3 | 免费；CC BY-NC-ND |
| **Reinforcement Learning: An Introduction** — Sutton & Barto | 2018 (2nd) | 强化学习标准教材 | 3 | 4 | 4 | **5** | 免费 PDF；CC BY-NC-ND |
| **The Little Book of Deep Learning** — François Fleuret | 2023 | 极简速览 | 4 | 4 | 3 | 3 | 免费；CC BY-NC-SA |
| **Hands-On Machine Learning** — Aurélien Géron | 2022 (3rd) | 工程实践 | 3 | 5 | 2 | 3 | 商业出版 |

### 1.2 中文教材

| 教材 | 版本/年份 | 定位 | 前沿 | 易读 | 严谨 | 可视化适配 | 免费获取 / 许可 |
|---|---|---|:-:|:-:|:-:|:-:|---|
| **《统计学习方法》（第 2 版）** — 李航 | 2019 | 经典算法推导 | 1 | 3 | **5** | **5** | 商业出版 |
| **《机器学习》（西瓜书）** — 周志华 | 2016 | 中文入门首选 | 2 | 4 | 3 | 4 | 商业出版 |
| **南瓜书 PumpkinBook** — Datawhale | 持续更新 | 西瓜书公式推导补全 | 2 | 4 | 4 | 3 | 开源；CC BY-NC-SA 4.0 |
| **《神经网络与深度学习》（蒲公英书）** — 邱锡鹏 | 2020 | 中文深度学习教材 | 3 | 4 | 4 | 4 | 官网免费阅读 |
| **《动手学深度学习》（第 2 版）** — 李沐等 | 持续更新 | D2L 中文版 | 4 | 5 | 3 | 4 | 开源；CC BY-SA 4.0 |
| **《机器学习理论导引》** — 周志华等 | 2020 | 学习理论 | 2 | 3 | 5 | 2 | 商业出版 |

## 2. 关键观察

1. **前沿性与严谨性普遍此消彼长。** 经典教材（ESL、PRML、李航）推导严密但止步于 2010 年前后，不含 Transformer、扩散模型、大模型；新书（Prince、Bishop 2024、Murphy）补上了前沿，但 Murphy 过于庞杂，不适合作为主线。
2. **Prince《Understanding Deep Learning》是三项指标最均衡的一本。** 2023 年出版，覆盖 Transformer / GNN / 扩散 / 流模型 / VAE / RL；作者自带大量精心设计的示意图与配套 Jupyter notebook，几乎每章都有可直接“动起来”的概念（损失曲面、激活区域、注意力矩阵、去噪过程）。
3. **ISL 是“可视化”的天然底本。** 全书以图驱动，讨论的都是二维即可直观展示的现象：偏差-方差权衡、正则化路径、交叉验证、决策边界、树与集成。易读性最高，适合作为统计学习主线的入口。
4. **李航《统计学习方法》最适合做“算法步骤动画”。** 每章都是“模型 → 策略 → 算法 → 例题”的固定结构，且附带小规模手算例题（感知机、kNN 的 kd 树、ID3/C4.5、SVM-SMO、AdaBoost、EM、HMM 前向后向/维特比、CRF），非常适合逐步回放式可视化，同时贴合中文读者。
5. **版权是选书的硬约束。** 我们做的是“可视化伴读”，应遵循：**不复制原书正文与插图**，只引用章节结构、重绘概念、写自己的讲解。
   - **CC BY-SA（D2L / 动手学深度学习）**：可改编、可商用，派生作品需同样以 CC BY-SA 发布——对项目最友好。
   - **CC BY-NC-SA（南瓜书、Fleuret）**：可改编，但不可商用。
   - **CC BY-NC-ND（Prince、Murphy、Sutton & Barto、Hardt & Recht）**：不可发布改编版，只能“引用 + 自绘 + 链接原书”。
   - **版权保留（ISL、ESL、PRML、李航、西瓜书等）**：同上，只能作为大纲参考与跳转链接。

## 3. 推荐方案

### 主线（建议先做这三本）

| 角色 | 教材 | 理由 |
|---|---|---|
| **统计学习主线** | ISL（Python 版 ISLP） | 易读性 + 可视化适配度最高；读者门槛低 |
| **深度学习主线** | Prince《Understanding Deep Learning》 | 前沿、易读、严谨三者最均衡；图示与 notebook 资源丰富 |
| **中文算法主线** | 李航《统计学习方法》第 2 版 | 严谨度高、结构规整、例题小而完整，最适合逐步算法动画 |

### 辅助资料（按需引用，不单独做可视化）

- **严谨性补充**：ESL（统计学习进阶推导）、PRML / Bishop 2024（概率视角）、Bach 2024（学习理论）
- **前沿性补充**：Murphy PML 1/2（查阅式参考）
- **数学预备**：Mathematics for Machine Learning（线代、概率、优化——本身就极适合可视化，可作“前置模块”）
- **代码与可派生内容**：D2L / 《动手学深度学习》（许可宽松，可直接改编示例）
- **中文补充**：西瓜书 + 南瓜书（概念广度）、邱锡鹏（中文深度学习术语对照）

### 建议的第一批可视化章节（验证技术路线）

1. ISL ch2 / ch6：偏差-方差权衡、Ridge / Lasso 正则化路径（交互式拖动 λ）
2. 李航 ch2 / ch7：感知机迭代过程、SVM 间隔与支持向量（可拖拽数据点）
3. Prince ch3–4 / ch12：浅层网络的分段线性区域、Transformer 注意力矩阵

这三组覆盖“统计概念 / 经典算法 / 深度学习”三种可视化形态，可用来确定组件库和交互范式，再横向扩展到全书。

## 4. 下一步待决策

- 目标读者：初学者为主，还是有数学基础的研究生？（决定 ISL 与 ESL 的比重）
- 语言：中文为主、英文为主，还是双语对照？
- 是否有商业化计划？（决定能否使用 NC 类许可的素材）
