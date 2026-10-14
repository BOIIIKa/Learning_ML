# Visible Machine Learning: From Rigorous Derivations to the Research Frontier

> Generated from `curriculum/outline.yaml` by `scripts/build_outline.py` — do not edit by hand. [中文](OUTLINE.zh.md)

- **Audience**: Students comfortable with calculus, linear algebra and probability who want to build ML research ability quickly
- **License**: Content under CC BY-NC-SA 4.0, code under MIT; no text or figures are copied from the source books
- **Source books**: `ISL` [An Introduction to Statistical Learning (2nd ed. / Python ed.)](https://www.statlearning.com/); `UDL` [Understanding Deep Learning](https://udlbook.github.io/udlbook/); `LH` [统计学习方法（第 2 版）/ Statistical Learning Methods (2nd ed.)](https://book.douban.com/subject/33437381/)

9 parts, 30 chapters, 69 interactive visuals, 58 frontier papers.

## Table of Contents

- **Prelude: The Mathematical Toolkit**
  - [00 Geometric Intuition for Linear Algebra, Probability and Optimization](#ch-00) · `Foundation`
- **Part I: The Framework of Learning**
  - [01 The Learning Problem: Model, Strategy, Algorithm](#ch-01) · `Foundation`
  - [02 Generalization, Bias–Variance and Model Selection](#ch-02) · `Foundation`
- **Part II: Classical Supervised Learning**
  - [03 Linear Regression and Regularization](#ch-03) · `Supervised`
  - [04 Linear Classification: Perceptron, Logistic Regression and Maximum Entropy](#ch-04) · `Supervised`
  - [05 Generative Classifiers: Naive Bayes and Discriminant Analysis](#ch-05) · `Supervised`
  - [06 Non-parametric Methods and Basis Expansions: kNN, Splines and GAMs](#ch-06) · `Supervised`
  - [07 Support Vector Machines and Kernel Methods](#ch-07) · `Supervised`
  - [08 Decision Trees and Ensemble Learning](#ch-08) · `Supervised`
  - [09 Structured Prediction: Hidden Markov Models and Conditional Random Fields](#ch-09) · `Supervised`
- **Part III: Unsupervised Learning and Latent Variables**
  - [10 Clustering](#ch-10) · `Unsupervised`
  - [11 The EM Algorithm and Mixture Models](#ch-11) · `Unsupervised`
  - [12 Matrix Factorization and Dimensionality Reduction: SVD, PCA and LSA](#ch-12) · `Unsupervised`
  - [13 Markov Chain Methods: MCMC, PageRank and Topic Models](#ch-13) · `Unsupervised`
- **Part IV: Foundations of Deep Learning**
  - [14 From Shallow to Deep Networks](#ch-14) · `Deep`
  - [15 Loss Functions as Maximum Likelihood](#ch-15) · `Deep`
  - [16 Optimization: From Gradient Descent to Adam](#ch-16) · `Deep`
  - [17 Backpropagation and Initialization](#ch-17) · `Deep`
  - [18 Regularization and the Generalization Puzzle in Deep Networks](#ch-18) · `Deep`
- **Part V: Architectures and Inductive Biases**
  - [19 Convolutional and Residual Networks](#ch-19) · `Deep`
  - [20 Transformers and Attention](#ch-20) · `Deep`
  - [21 Graph Neural Networks](#ch-21) · `Deep`
- **Part VI: Generative Models**
  - [22 A Map of Generative Models, and GANs](#ch-22) · `Generative`
  - [23 Normalizing Flows](#ch-23) · `Generative`
  - [24 Variational Autoencoders](#ch-24) · `Generative`
  - [25 Diffusion Models](#ch-25) · `Generative`
- **Part VII: Decision-Making and Large Models**
  - [26 Reinforcement Learning](#ch-26) · `RL`
  - [27 Large Language Models: Putting the Course Together](#ch-27) · `Frontier`
- **Part VIII: Doing Research**
  - [28 Experimental Design, Evaluation and Multiple Testing](#ch-28) · `Methodology`
  - [29 Responsible Machine Learning](#ch-29) · `Methodology`
- **[Cross-indices](#index)**

## Prelude: The Mathematical Toolkit

*Only the mathematics the rest of the course uses, each built from a geometric picture.*

<a id="ch-00"></a>

### 00 Geometric Intuition for Linear Algebra, Probability and Optimization

- **Paradigm**: Foundation
- **Source chapters**: UDL App. B, App. C; LH 附录 A–E
- **Prerequisites**: none

**Learning goals**

- See matrices as linear maps and the SVD as rotate–scale–rotate
- Master the trio of MLE / MAP, KL divergence and Jensen's inequality
- Understand convexity, gradients, the Hessian, and Lagrangian duality / KKT conditions

**Rigorous derivations**

1. MLE for the multivariate Gaussian; eigendecomposition of the covariance and its contour ellipses
2. Non-negativity of KL divergence via Jensen's inequality
3. Lagrangian duality: weak/strong duality and the geometry of KKT

**Interactive visuals**

- `viz-linear-map` **A matrix is a transformation** — Drag the four entries of a 2×2 matrix to warp a unit grid and circle, with the three-step SVD shown alongside
- `viz-gaussian-mle` **Gaussian fitting and the likelihood surface** — Click to add points; the MLE mean and covariance ellipse update live next to the log-likelihood surface
- `viz-kkt` **Constrained optimization and KKT** — Drag the constraint boundary and see when the multiplier hits zero as the optimum moves between interior and boundary (complementary slackness)

## Part I: The Framework of Learning

*Introduces the model–strategy–algorithm triad used in every chapter, and generalization as the central question.*

<a id="ch-01"></a>

### 01 The Learning Problem: Model, Strategy, Algorithm

- **Paradigm**: Foundation
- **Source chapters**: ISL 2; UDL 2; LH 1
- **Prerequisites**: [00 Geometric Intuition for Linear Algebra, Probability and Optimization](#ch-00)

**Learning goals**

- Separate the three independent choices: hypothesis space, loss / risk, optimizer
- Understand expected risk, empirical risk and structural risk minimization
- Place methods on two axes: supervised / unsupervised / RL and parametric / non-parametric

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Hypothesis space $\mathcal{F}$: a family of functions or conditional distributions | Empirical risk + regularizer (structural risk) | The numerical method that solves the resulting optimization problem |

**Rigorous derivations**

1. Under squared loss the Bayes-optimal predictor is $E[Y|X]$
2. Under 0-1 loss: the Bayes classifier and the Bayes error rate

**Interactive visuals**

- `viz-triad-explorer` **Triad composer** — Pick a model, a loss and an optimizer from three menus and watch training on one dataset — every algorithm in the course is one such combination
- `viz-risk-gap` **Empirical vs expected risk** — Change sample size n and watch the empirical risk curve approach the true risk curve

<a id="ch-02"></a>

### 02 Generalization, Bias–Variance and Model Selection

- **Paradigm**: Foundation
- **Source chapters**: ISL 2, 5; UDL 8; LH 1
- **Prerequisites**: [01 The Learning Problem: Model, Strategy, Algorithm](#ch-01)

**Learning goals**

- Derive the bias–variance decomposition and the classical U-shaped curve
- Master hold-out, k-fold cross-validation and the bootstrap
- Meet modern double descent and see where classical theory breaks down

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| A nested family of models of increasing complexity | Estimate generalization error with validation error | Cross-validation / bootstrap |

**Rigorous derivations**

1. Under squared loss: expected error = bias² + variance + noise
2. Generalization bound for a finite hypothesis space (Hoeffding + union bound)
3. The bias–variance trade-off of the k-fold CV estimator itself

**Interactive visuals**

- `viz-bias-variance` **Bias–variance simulator** — Resample training sets, overlay the fitted curves, drag polynomial degree, and see the error split into three parts live
- `viz-double-descent` **Double descent** — Sweep the number of random features past the interpolation threshold p = n and watch test error rise then fall again
- `viz-cv-folds` **Folds of cross-validation** — Animate k-fold splits; vary k to see the mean and variance of the CV estimate

**Frontier reading**

- Belkin, Hsu, Ma, Mandal (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-off. PNAS. — Introduced the double-descent curve
- Nakkiran et al. (2020). Deep Double Descent. ICLR. — Systematic double descent in deep nets across model size, data and epochs
- Zhang, Bengio, Hardt, Recht, Vinyals (2017). Understanding deep learning requires rethinking generalization. ICLR. — Networks fit random labels — a challenge to classical complexity theory

## Part II: Classical Supervised Learning

*Li Hang's rigorous derivations as the skeleton and ISL's statistical view as the flesh; every model is dissected into the triad.*

<a id="ch-03"></a>

### 03 Linear Regression and Regularization

- **Paradigm**: Supervised
- **Source chapters**: ISL 3, 6; UDL 2
- **Prerequisites**: [02 Generalization, Bias–Variance and Model Selection](#ch-02)

**Learning goals**

- Understand least squares both as a projection and as Gaussian MLE
- See the geometric difference between ridge and lasso, and why lasso gives sparse solutions
- Interpret regularization as a prior (MAP)

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| $f(x)=w^\top x+b$ | Squared loss + L2 / L1 penalty | Normal equations / coordinate descent (soft thresholding) / gradient descent |

**Rigorous derivations**

1. Normal equations and the hat matrix: $\hat y$ is the orthogonal projection of $y$ onto the column space
2. Ridge closed form and its shrinkage factors $d_j^2/(d_j^2+\lambda)$ in the SVD basis
3. Lasso subgradient optimality and the soft-thresholding operator
4. Gaussian prior ⇒ ridge; Laplace prior ⇒ lasso

**Interactive visuals**

- `viz-projection-3d` **Least squares is projection** — In 3D, drag the vector y and watch its projection onto the column-space plane and the orthogonal residual
- `viz-l1-l2-geometry` **Geometry of L1 vs L2** — Drag the centre of the loss contours and see where they touch the diamond / disc — why L1 lands on the axes
- `viz-reg-path` **Regularization path** — Slide λ to update the coefficient paths, the fitted curve and CV error together

**Frontier reading**

- Tibshirani (1996). Regression Shrinkage and Selection via the Lasso. JRSS-B. — The original lasso paper
- Hastie, Montanari, Rosset, Tibshirani (2022). Surprises in high-dimensional ridgeless least squares interpolation. Annals of Statistics. — Exact risk of minimum-norm interpolation: double descent in linear models

<a id="ch-04"></a>

### 04 Linear Classification: Perceptron, Logistic Regression and Maximum Entropy

- **Paradigm**: Supervised
- **Source chapters**: ISL 4; LH 2, 6
- **Prerequisites**: [03 Linear Regression and Regularization](#ch-03)

**Learning goals**

- Master the perceptron's primal form, dual form and convergence theorem
- View logistic regression as both Bernoulli MLE and cross-entropy minimization
- Show that the dual of the maximum-entropy model is multinomial logistic regression (softmax)

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Linear boundary $\mathrm{sign}(w^\top x+b)$ or $\sigma(w^\top x+b)$ | Sum of distances of misclassified points / negative log-likelihood / maximum entropy | SGD / Newton's method (IRLS) / improved iterative scaling |

**Rigorous derivations**

1. Novikoff's theorem: on separable data the perceptron makes at most $(R/\gamma)^2$ mistakes
2. Gradient and Hessian of the logistic log-likelihood; proof of concavity
3. Lagrangian dual of maximum entropy ⇒ exponential-family / softmax form

**Interactive visuals**

- `viz-perceptron-steps` **Perceptron step by step** — Step through: highlight the misclassified point, show the update vector and the rotating hyperplane; toggle primal / dual form
- `viz-logistic-surface` **The logistic probability surface** — Drag w and b; the 3D sigmoid surface, 2D boundary and cross-entropy loss update together
- `viz-loss-zoo` **Zoo of classification losses** — Plot 0-1, perceptron, cross-entropy, hinge and exponential losses against the margin $yf(x)$ (linked to Ch. 07 and 08)

<a id="ch-05"></a>

### 05 Generative Classifiers: Naive Bayes and Discriminant Analysis

- **Paradigm**: Supervised
- **Source chapters**: ISL 4; LH 4
- **Prerequisites**: [04 Linear Classification: Perceptron, Logistic Regression and Maximum Entropy](#ch-04)

**Learning goals**

- Grasp the core difference between generative $p(x,y)$ and discriminative $p(y|x)$ modelling
- Derive the shape of LDA / QDA decision boundaries
- Understand the naive Bayes independence assumption and Laplace smoothing

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Class prior $p(y)$ × class-conditional density $p(x|y)$ | Maximize joint likelihood (decide by posterior) | Counting / closed-form parameter estimates |

**Rigorous derivations**

1. Shared covariance ⇒ linear boundary (LDA); separate covariances ⇒ quadratic boundary (QDA)
2. The LDA posterior has logistic form — same model family, different estimation strategy
3. Maximum-likelihood vs Bayesian estimates for naive Bayes (Laplace smoothing)

**Interactive visuals**

- `viz-lda-qda` **From Gaussians to boundaries** — Drag the means and covariance ellipses of two Gaussians; LDA, QDA and logistic boundaries update live
- `viz-gen-vs-disc` **Sample efficiency: generative vs discriminative** — Vary training size and compare the crossing learning curves of naive Bayes and logistic regression

**Frontier reading**

- Ng & Jordan (2002). On Discriminative vs. Generative Classifiers: A comparison of logistic regression and naive Bayes. NeurIPS. — Generative converges faster; discriminative has lower asymptotic error

<a id="ch-06"></a>

### 06 Non-parametric Methods and Basis Expansions: kNN, Splines and GAMs

- **Paradigm**: Supervised
- **Source chapters**: ISL 2, 7; LH 3
- **Prerequisites**: [02 Generalization, Bias–Variance and Model Selection](#ch-02)

**Learning goals**

- Understand kNN's three choices: distance metric, k, and decision rule
- Master kd-tree construction and search
- Turn linear models non-linear via basis expansions; understand the curse of dimensionality

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Local averaging / linear combination of basis functions | Majority vote (0-1 loss) / penalized squared loss | kd-tree search / closed-form smoothing spline |

**Rigorous derivations**

1. The asymptotic 1-NN error is at most twice the Bayes error (Cover–Hart)
2. Volume concentration in high-dimensional shells: quantifying the curse of dimensionality
3. Degrees of freedom of cubic splines as a function of knots

**Interactive visuals**

- `viz-knn-boundary` **k and the decision boundary** — Slide k from jagged to smooth boundaries; switch between L1 / L2 / L∞ distances
- `viz-kdtree` **kd-tree construction and nearest-neighbour search** — Animate the recursive splits; click a query point to replay backtracking and pruning
- `viz-curse-dim` **The curse of dimensionality** — Raise dimension d and watch the nearest/farthest distance ratio approach 1

<a id="ch-07"></a>

### 07 Support Vector Machines and Kernel Methods

- **Paradigm**: Supervised
- **Source chapters**: ISL 9; LH 7
- **Prerequisites**: [00 Geometric Intuition for Linear Algebra, Probability and Optimization](#ch-00), [04 Linear Classification: Perceptron, Logistic Regression and Maximum Entropy](#ch-04)

**Learning goals**

- Fully derive hard / soft-margin SVMs and their duals from the max-margin principle
- Understand the kernel trick, positive-definite kernels and the intuition of an RKHS
- Master SMO's two-variable subproblem

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Separating hyperplane (possibly in a kernel feature space) | Margin maximization ⇔ hinge loss + L2 regularization | Lagrangian dual + SMO |

**Rigorous derivations**

1. Geometric margin → convex QP primal → KKT → dual problem
2. Soft-margin SVM is equivalent to unconstrained hinge loss + L2 regularization
3. Mercer's condition; feature spaces of common kernels (polynomial, RBF)
4. SMO: closed-form two-variable update and clipping

**Interactive visuals**

- `viz-svm-margin` **Draggable maximum margin** — Drag points and re-solve the SVM live; support vectors light up, and moving non-support vectors leaves the boundary unchanged
- `viz-kernel-lift` **Kernel lifting: 2D to 3D** — Concentric circles become separable after lifting by $\phi(x)=(x_1,x_2,x_1^2+x_2^2)$; tune RBF bandwidth to reshape the boundary
- `viz-smo` **SMO iteration replay** — Step through each chosen pair of α and its update path inside the box constraint

**Frontier reading**

- Rahimi & Recht (2007). Random Features for Large-Scale Kernel Machines. NeurIPS. — Random features approximate kernels, bridging kernels and two-layer networks
- Jacot, Gabriel, Hongler (2018). Neural Tangent Kernel. NeurIPS. — Training infinitely wide networks equals kernel regression (links to Ch. 18)

<a id="ch-08"></a>

### 08 Decision Trees and Ensemble Learning

- **Paradigm**: Supervised
- **Source chapters**: ISL 8; LH 5, 8
- **Prerequisites**: [02 Generalization, Bias–Variance and Model Selection](#ch-02), [04 Linear Classification: Perceptron, Logistic Regression and Maximum Entropy](#ch-04)

**Learning goals**

- Master the splitting criteria and pruning of ID3 / C4.5 / CART
- Understand how bagging / random forests reduce variance
- Derive AdaBoost as forward stagewise additive modelling under exponential loss, then generalize to gradient boosting

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Axis-aligned piecewise-constant functions / weighted sums of base learners | Information gain / Gini / exponential loss / any differentiable loss | Greedy recursive splitting + cost-complexity pruning / forward stagewise |

**Rigorous derivations**

1. Information gain, gain ratio and the Gini index
2. Exponentially decreasing upper bound on AdaBoost training error
3. Forward stagewise + exponential loss ⇒ AdaBoost's weight and coefficient formulas
4. Gradient boosting: gradient descent in function space

**Interactive visuals**

- `viz-tree-grow` **Growing a tree** — Each step shows a bar chart of candidate split gains; the chosen split cuts the plane and the tree grows in sync; tune pruning α
- `viz-adaboost-weights` **AdaBoost sample weights** — Point size encodes weight; play rounds of weak learners and reweighting while plotting training error and its bound
- `viz-bagging-variance` **Ensembles reduce variance** — Vary the number of trees and feature subsampling; compare the jitter of a single tree versus a forest

**Frontier reading**

- Friedman (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics. — The founding gradient boosting paper
- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD. — Second-order approximation plus systems engineering; an industry standard
- Grinsztajn, Oyallon, Varoquaux (2022). Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS Datasets & Benchmarks. — Why trees still win on tabular data

<a id="ch-09"></a>

### 09 Structured Prediction: Hidden Markov Models and Conditional Random Fields

- **Paradigm**: Supervised
- **Source chapters**: LH 10, 11
- **Prerequisites**: [05 Generative Classifiers: Naive Bayes and Discriminant Analysis](#ch-05)

**Learning goals**

- Master the three HMM problems: evaluation, learning, decoding
- Relate HMMs (generative) to linear-chain CRFs (discriminative)
- See forward–backward and Viterbi as dynamic programming on a trellis

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Chain-structured probabilistic graphical models | Maximize (conditional) log-likelihood | Forward–backward, Baum–Welch (EM), Viterbi, quasi-Newton |

**Rigorous derivations**

1. Forward / backward recursions and their $O(TN^2)$ cost
2. Baum–Welch as a special case of EM (links to Ch. 11)
3. Parametric form of the linear-chain CRF and computing its partition function

**Interactive visuals**

- `viz-trellis` **Dynamic programming on the trellis** — Light up forward probabilities column by column; switch to Viterbi to show max operations, back-pointers and the final best path
- `viz-hmm-vs-crf` **Graph structure of HMM vs CRF** — Directed vs undirected graphs side by side; hover a factor to see its parametrization

**Frontier reading**

- Lafferty, McCallum, Pereira (2001). Conditional Random Fields. ICML. — The original CRF paper
- Gu & Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv. — Modern state-space models — linear recurrences plus parallel scans, comparable with HMMs

## Part III: Unsupervised Learning and Latent Variables

*What to learn without labels: clusters, latent variables, low-rank structure and sampling — the direct ancestors of modern generative models.*

<a id="ch-10"></a>

### 10 Clustering

- **Paradigm**: Unsupervised
- **Source chapters**: ISL 12; LH 13, 14
- **Prerequisites**: [01 The Learning Problem: Model, Strategy, Algorithm](#ch-01)

**Learning goals**

- Master k-means and hierarchical clustering
- See k-means as coordinate descent with only local optimality guarantees
- Learn to choose k and evaluate clusterings

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| A partition / set of centroids | Minimize within-cluster sum of squares / linkage criterion | Lloyd's alternating minimization / agglomerative merging |

**Rigorous derivations**

1. Proof that each Lloyd step does not increase the objective
2. Expected $O(\log k)$ approximation ratio of k-means++ seeding

**Interactive visuals**

- `viz-kmeans` **k-means iterations** — Step through assign / update with a live objective curve; compare random vs k-means++ seeding
- `viz-dendrogram` **Cutting a dendrogram** — Drag the cut line and watch clusters change in the scatter plot; switch single / complete / average linkage

**Frontier reading**

- Arthur & Vassilvitskii (2007). k-means++: The Advantages of Careful Seeding. SODA. — Seeding with a provable guarantee

<a id="ch-11"></a>

### 11 The EM Algorithm and Mixture Models

- **Paradigm**: Unsupervised
- **Source chapters**: LH 9
- **Prerequisites**: [00 Geometric Intuition for Linear Algebra, Probability and Optimization](#ch-00), [10 Clustering](#ch-10)

**Learning goals**

- Derive EM from the ELBO and prove monotone likelihood increase
- Derive the E-step and M-step for Gaussian mixtures
- See k-means as the hard-assignment limit of a GMM

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Latent-variable model $p(x,z|\theta)$ | Maximize marginal likelihood (via a lower bound) | EM: coordinate ascent on the ELBO |

**Rigorous derivations**

1. $\log p(x) = \mathrm{ELBO}(q,\theta) + \mathrm{KL}(q\,\|\,p(z|x))$
2. E-step zeroes the KL, M-step raises the bound ⇒ monotonicity
3. Generalized EM and the maximization–maximization view (setting up VAEs)

**Interactive visuals**

- `viz-gmm-em` **EM for a GMM** — Point colour shows soft responsibilities, ellipses converge step by step, log-likelihood plotted alongside
- `viz-elbo-bound` **Touching and pushing the lower bound** — For a 1-D parameter, draw $\log p(x|\theta)$ and each step's ELBO, animating the touch-then-push cycle

**Frontier reading**

- Dempster, Laird, Rubin (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm. JRSS-B. — The original EM paper
- Kingma & Welling (2014). Auto-Encoding Variational Bayes. ICLR. — Replace the E-step with amortized neural inference ⇒ VAE (Ch. 24)

<a id="ch-12"></a>

### 12 Matrix Factorization and Dimensionality Reduction: SVD, PCA and LSA

- **Paradigm**: Unsupervised
- **Source chapters**: ISL 12; LH 15, 16, 17
- **Prerequisites**: [00 Geometric Intuition for Linear Algebra, Probability and Optimization](#ch-00)

**Learning goals**

- Prove the max-variance and min-reconstruction-error definitions of PCA coincide
- Master the Eckart–Young theorem and truncated SVD
- Understand the low-rank structure of term–document matrices through LSA

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Low-rank linear subspace | Maximize variance / minimize reconstruction error | Eigendecomposition / SVD / non-negative matrix factorization |

**Rigorous derivations**

1. PCA via Lagrange multipliers yields the covariance eigen-equation
2. Eckart–Young: truncated SVD is the best rank-k approximation in Frobenius norm
3. Multiplicative update rules for NMF

**Interactive visuals**

- `viz-pca-rotate` **Finding the principal direction** — Rotate the projection axis while projected variance and reconstruction error sum to a constant; release to snap to the optimum
- `viz-lowrank-image` **Low-rank image approximation** — Slide rank k to update the image and singular-value spectrum together, with the compression ratio
- `viz-scree` **Scree plot and explained variance** — Pick the number of components on a real dataset and see cumulative explained variance and the 2-D projection

**Frontier reading**

- Halko, Martinsson, Tropp (2011). Finding Structure with Randomness. SIAM Review. — Randomized SVD for large-scale low-rank approximation
- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR. — Low-rank factorization for fine-tuning large models

<a id="ch-13"></a>

### 13 Markov Chain Methods: MCMC, PageRank and Topic Models

- **Paradigm**: Unsupervised
- **Source chapters**: LH 18, 19, 20, 21
- **Prerequisites**: [11 The EM Algorithm and Mixture Models](#ch-11), [12 Matrix Factorization and Dimensionality Reduction: SVD, PCA and LSA](#ch-12)

**Learning goals**

- Master stationary distributions and detailed balance
- See PageRank as a random walk's stationary distribution, and MCMC as designing a chain whose stationary distribution is the target
- Understand PLSA and LDA topic models and their Gibbs sampler

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Markov chains / hierarchical Bayesian topic models | Stationary distribution / posterior inference | Power iteration / Metropolis–Hastings / Gibbs sampling / variational EM |

**Rigorous derivations**

1. Detailed balance ⇒ stationarity; deriving the M–H acceptance ratio
2. PageRank: the damped transition matrix and Perron–Frobenius
3. Collapsed Gibbs conditionals for LDA

**Interactive visuals**

- `viz-mh-sampler` **Metropolis–Hastings sampler** — Watch the chain wander over a 2-D target; tune proposal step size to trade acceptance rate against mixing
- `viz-pagerank-surfer` **The random surfer** — Play a random walk on an editable web graph; node sizes converge to PageRank; tune the damping factor
- `viz-lda-topics` **Topics emerging** — Run Gibbs sweeps on a small corpus and watch word–topic assignments go from random to structured

**Frontier reading**

- Blei, Ng, Jordan (2003). Latent Dirichlet Allocation. JMLR. — The original LDA paper
- Welling & Teh (2011). Bayesian Learning via Stochastic Gradient Langevin Dynamics. ICML. — Langevin sampling — the road to score-based and diffusion models (Ch. 25)

## Part IV: Foundations of Deep Learning

*Following UDL: what functions networks compute, which losses, how they are optimized, and why they generalize.*

<a id="ch-14"></a>

### 14 From Shallow to Deep Networks

- **Paradigm**: Deep
- **Source chapters**: ISL 10; UDL 3, 4
- **Prerequisites**: [03 Linear Regression and Regularization](#ch-03), [04 Linear Classification: Perceptron, Logistic Regression and Maximum Entropy](#ch-04)

**Learning goals**

- Understand ReLU networks as piecewise-linear functions and count their linear regions
- Understand what the universal approximation theorem does and does not say
- Understand the expressive efficiency of depth through “folding”

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Compositions of affine maps and element-wise non-linearities | (see Ch. 15) | (see Ch. 16 and 17) |

**Rigorous derivations**

1. Single-hidden-layer ReLU nets: at most $D+1$ linear pieces on 1-D input with $D$ hidden units
2. A construction where linear regions grow exponentially with depth

**Interactive visuals**

- `viz-relu-regions` **Linear regions of a ReLU network** — Tune hidden units and weights; the 2-D input space splits into polygons, coloured by region, with the output surface
- `viz-depth-folding` **Depth as folding** — Animate layer-by-layer folding of input space; compare region counts of wide-shallow and narrow-deep nets at equal parameters

**Frontier reading**

- Montúfar, Pascanu, Cho, Bengio (2014). On the Number of Linear Regions of Deep Neural Networks. NeurIPS. — Theory of linear-region counts in deep nets

<a id="ch-15"></a>

### 15 Loss Functions as Maximum Likelihood

- **Paradigm**: Deep
- **Source chapters**: UDL 5
- **Prerequisites**: [00 Geometric Intuition for Linear Algebra, Probability and Optimization](#ch-00), [14 From Shallow to Deep Networks](#ch-14)

**Learning goals**

- Master the recipe: choose an output distribution → write its negative log-likelihood → obtain the loss
- Unify the losses of Parts II and III under this recipe

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| The network outputs parameters of a distribution | Negative log-likelihood | (see Ch. 16) |

**Rigorous derivations**

1. Gaussian ⇒ squared loss; Bernoulli ⇒ binary cross-entropy; categorical ⇒ softmax cross-entropy
2. Heteroscedastic regression: predicting mean and variance together
3. Relation between cross-entropy and KL divergence

**Interactive visuals**

- `viz-loss-recipe` **The loss recipe machine** — Pick an output distribution; the loss formula, its curve and the matching classical model (linked back to earlier chapters) appear
- `viz-heteroscedastic` **Predicting uncertainty** — Compare homoscedastic and heteroscedastic fits on data whose noise varies with x

<a id="ch-16"></a>

### 16 Optimization: From Gradient Descent to Adam

- **Paradigm**: Deep
- **Source chapters**: UDL 6; LH 附录 A, 附录 B
- **Prerequisites**: [00 Geometric Intuition for Linear Algebra, Probability and Optimization](#ch-00), [15 Loss Functions as Maximum Likelihood](#ch-15)

**Learning goals**

- Relate step size, condition number and convergence rate
- Master the update rules of SGD, momentum, Nesterov and Adam / AdamW
- Recognize saddle points, plateaus and learning-rate schedules on non-convex landscapes

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Any differentiable parametrized model | Any differentiable loss | First- and second-order iterative optimization |

**Rigorous derivations**

1. Linear convergence $(1-\mu/L)^t$ of gradient descent on strongly convex smooth functions
2. Momentum as a damped heavy-ball ODE
3. Deriving Adam's bias correction; AdamW's decoupled weight decay

**Interactive visuals**

- `viz-optimizer-race` **Optimizer race** — Run several optimizers at once on Rosenbrock, saddle and ill-conditioned quadratic surfaces with adjustable learning rate and start
- `viz-condition-number` **Condition number and zig-zags** — Stretch a quadratic bowl's condition number and watch GD zig-zag and its step count
- `viz-lr-schedule` **Learning-rate schedules** — Compare training curves under constant, step, cosine and warm-up schedules

**Frontier reading**

- Kingma & Ba (2015). Adam: A Method for Stochastic Optimization. ICLR. — The original Adam paper
- Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization. ICLR. — AdamW
- Cohen et al. (2021). Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability. ICLR. — The edge-of-stability phenomenon at large learning rates

<a id="ch-17"></a>

### 17 Backpropagation and Initialization

- **Paradigm**: Deep
- **Source chapters**: UDL 7
- **Prerequisites**: [14 From Shallow to Deep Networks](#ch-14), [16 Optimization: From Gradient Descent to Adam](#ch-16)

**Learning goals**

- Derive backprop on a computational graph as reverse-mode automatic differentiation
- Understand vanishing / exploding gradients and derive He / Xavier initialization

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Computational graph | — | Chain rule + dynamic programming (reverse mode) |

**Rigorous derivations**

1. Layer-wise gradient recursion for an MLP
2. Preserving forward activation variance ⇒ $\mathrm{Var}(w)=2/n_{in}$ (He initialization)
3. Cost of forward- vs reverse-mode autodiff

**Interactive visuals**

- `viz-backprop-graph` **Forward and backward on a computational graph** — An editable expression graph: play the forward value flow, then the backward gradient flow, with local derivatives on nodes
- `viz-init-variance` **Initialization and activation variance** — Tune init scale and depth, plot per-layer activation / gradient histograms, and watch them vanish or explode

**Frontier reading**

- He, Zhang, Ren, Sun (2015). Delving Deep into Rectifiers. ICCV. — He initialization

<a id="ch-18"></a>

### 18 Regularization and the Generalization Puzzle in Deep Networks

- **Paradigm**: Deep
- **Source chapters**: UDL 8, 9, 20
- **Prerequisites**: [02 Generalization, Bias–Variance and Model Selection](#ch-02), [16 Optimization: From Gradient Descent to Adam](#ch-16), [17 Backpropagation and Initialization](#ch-17)

**Learning goals**

- Master explicit regularizers: weight decay, dropout, augmentation, early stopping
- Understand implicit regularization: SGD noise, flat minima, over-parametrization
- Map out the open questions of why deep learning works

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Over-parametrized networks | Empirical risk + explicit / implicit regularization | SGD and variants |

**Rigorous derivations**

1. In linear models early stopping approximates L2 regularization
2. Dropout as an ensemble / adaptive regularizer
3. Gradient descent converges to the minimum-norm solution in underdetermined linear regression

**Interactive visuals**

- `viz-loss-landscape` **Loss landscape slices** — Slice the loss along two random directions; compare minima flatness with / without skip connections and with large / small batches
- `viz-grokking` **Grokking** — Replay train / test accuracy on modular arithmetic; tune weight decay to see when generalization suddenly appears

**Frontier reading**

- Li, Xu, Taylor, Studer, Goldstein (2018). Visualizing the Loss Landscape of Neural Nets. NeurIPS. — Loss-landscape visualization method
- Frankle & Carbin (2019). The Lottery Ticket Hypothesis. ICLR. — Sparse trainable subnetworks
- Power, Burda, Edwards, Babuschkin, Misra (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv. — Delayed generalization

## Part V: Architectures and Inductive Biases

*Architecture is design of the model element: building the data's symmetries into the network.*

<a id="ch-19"></a>

### 19 Convolutional and Residual Networks

- **Paradigm**: Deep
- **Source chapters**: UDL 10, 11
- **Prerequisites**: [17 Backpropagation and Initialization](#ch-17)

**Learning goals**

- Derive convolution from translation equivariance; understand receptive-field growth
- Understand why residual connections and normalization make deep nets trainable

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Weight-shared local operators + residual structure | Cross-entropy | SGD + batch normalization |

**Rigorous derivations**

1. Convolution = linear operators that commute with translation
2. Unrolling gradient paths in ResNets; the ensemble-of-shallow-nets view

**Interactive visuals**

- `viz-conv-slide` **Sliding a kernel** — Drag an editable kernel over an image to see the feature map; tune stride, padding and dilation
- `viz-receptive-field` **Receptive-field growth** — Click a deep unit to highlight its receptive field on the input

**Frontier reading**

- He, Zhang, Ren, Sun (2016). Deep Residual Learning for Image Recognition. CVPR. — ResNet
- Dosovitskiy et al. (2021). An Image is Worth 16x16 Words. ICLR. — ViT replaces convolution with a Transformer (links to Ch. 20)

<a id="ch-20"></a>

### 20 Transformers and Attention

- **Paradigm**: Deep
- **Source chapters**: UDL 12
- **Prerequisites**: [17 Backpropagation and Initialization](#ch-17), [19 Convolutional and Residual Networks](#ch-19)

**Learning goals**

- Derive scaled dot-product attention as a differentiable soft lookup
- Understand multi-head attention, positional encodings, layer norm and residuals in a Transformer block
- Distinguish encoders, decoders and causal masking

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Stacks of self-attention + MLP (permutation-equivariant) | Next-token cross-entropy / masked language modelling | AdamW + warm-up + cosine schedule |

**Rigorous derivations**

1. Why divide by $\sqrt{d_k}$: variance of dot products
2. Permutation equivariance of self-attention, why positions are needed; the rotation construction of RoPE
3. The $O(n^2)$ cost of attention

**Interactive visuals**

- `viz-attention-matrix` **Interactive attention matrix** — Type a sentence and hover tokens to see per-head attention weights; toggle the causal mask
- `viz-qkv-geometry` **Geometry of Q, K, V** — Drag a query in 2-D to see dot products with keys, softmax weights and the output as a convex combination of values
- `viz-transformer-block` **Data flow through a Transformer block** — Step through tensor shapes: embedding → multi-head attention → residual & norm → MLP

**Frontier reading**

- Vaswani et al. (2017). Attention Is All You Need. NeurIPS. — The original Transformer paper
- Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv. — Rotary position embedding (RoPE)
- Dao, Fu, Ermon, Rudra, Ré (2022). FlashAttention. NeurIPS. — IO-aware exact attention
- Elhage et al. (2021). A Mathematical Framework for Transformer Circuits. Transformer Circuits Thread. — An entry point to mechanistic interpretability

<a id="ch-21"></a>

### 21 Graph Neural Networks

- **Paradigm**: Deep
- **Source chapters**: UDL 13
- **Prerequisites**: [13 Markov Chain Methods: MCMC, PageRank and Topic Models](#ch-13), [20 Transformers and Attention](#ch-20)

**Learning goals**

- Derive message passing from permutation invariance / equivariance
- Connect GCNs to the graph Laplacian spectrum and understand over-smoothing

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Neighbourhood aggregation + update function | Node / edge / graph-level cross-entropy | Full-graph or sampled mini-batch training |

**Rigorous derivations**

1. Where the GCN rule $\tilde D^{-1/2}\tilde A\tilde D^{-1/2}HW$ comes from
2. Repeated aggregation is low-pass filtering ⇒ over-smoothing

**Interactive visuals**

- `viz-message-passing` **Message passing** — Play layer-by-layer neighbour aggregation on an editable graph, colours as features; watch over-smoothing as depth grows

**Frontier reading**

- Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR. — GCN
- Veličković et al. (2018). Graph Attention Networks. ICLR. — GAT: attention on graphs

## Part VI: Generative Models

*Pushing Part III's ideas of density estimation and latent variables to their limit with deep networks.*

<a id="ch-22"></a>

### 22 A Map of Generative Models, and GANs

- **Paradigm**: Generative
- **Source chapters**: UDL 14, 15
- **Prerequisites**: [16 Optimization: From Gradient Descent to Adam](#ch-16), [19 Convolutional and Residual Networks](#ch-19)

**Learning goals**

- Classify generative models by three questions: tractable likelihood? latent variables? fast sampling?
- Derive the GAN minimax objective and its link to the JS divergence
- Understand mode collapse and the Wasserstein GAN

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Generator $G(z)$ + discriminator $D(x)$ | Minimax game | Alternating gradient descent / ascent |

**Rigorous derivations**

1. Optimal discriminator $D^*=p_{data}/(p_{data}+p_g)$ ⇒ objective becomes the JS divergence
2. Kantorovich–Rubinstein duality for the Wasserstein distance

**Interactive visuals**

- `viz-generative-map` **Map of generative models** — A clickable taxonomy of GAN / flow / VAE / diffusion / autoregressive models, coloured by the three questions and linked to chapters
- `viz-gan-1d` **1-D GAN training** — Watch real density, generated density and discriminator curve compete live; reproduce mode collapse

**Frontier reading**

- Goodfellow et al. (2014). Generative Adversarial Nets. NeurIPS. — The original GAN paper
- Arjovsky, Chintala, Bottou (2017). Wasserstein GAN. ICML. — WGAN
- Karras, Laine, Aila (2019). A Style-Based Generator Architecture for GANs. CVPR. — StyleGAN

<a id="ch-23"></a>

### 23 Normalizing Flows

- **Paradigm**: Generative
- **Source chapters**: UDL 16
- **Prerequisites**: [00 Geometric Intuition for Linear Algebra, Probability and Optimization](#ch-00), [22 A Map of Generative Models, and GANs](#ch-22)

**Learning goals**

- Write the exact likelihood with the change-of-variables formula
- Understand how coupling layers keep invertibility and a cheap Jacobian determinant

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Sequence of invertible maps $x=f_K\circ\cdots\circ f_1(z)$ | Exact negative log-likelihood | Gradient descent |

**Rigorous derivations**

1. $\log p(x)=\log p(z)-\sum_k\log|\det J_{f_k}|$
2. The triangular Jacobian of affine coupling layers

**Interactive visuals**

- `viz-flow-warp` **Kneading a Gaussian into moons** — Animate a base-distribution grid warping layer by layer into the target, with local area scaling (the determinant)

**Frontier reading**

- Dinh, Sohl-Dickstein, Bengio (2017). Density Estimation using Real NVP. ICLR. — Affine coupling layers
- Lipman, Chen, Ben-Hamu, Nickel, Le (2023). Flow Matching for Generative Modeling. ICLR. — Simulation-free training of continuous flows, bridging flows and diffusion (Ch. 25)

<a id="ch-24"></a>

### 24 Variational Autoencoders

- **Paradigm**: Generative
- **Source chapters**: UDL 17
- **Prerequisites**: [11 The EM Algorithm and Mixture Models](#ch-11), [22 A Map of Generative Models, and GANs](#ch-22)

**Learning goals**

- See the VAE as EM with a neural E-step (back to Ch. 11)
- Master the reparameterization trick
- Understand the reconstruction–KL trade-off and posterior collapse

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Decoder $p_\theta(x|z)$ + encoder $q_\phi(z|x)$ | Maximize the ELBO | Reparameterization + stochastic gradients |

**Rigorous derivations**

1. ELBO = reconstruction − KL; the closed-form Gaussian KL
2. Variance of reparameterization vs score-function (REINFORCE) gradients

**Interactive visuals**

- `viz-latent-space` **A walkable latent space** — Drag a cursor through a 2-D latent space to decode samples live; encoder posteriors drawn as ellipses
- `viz-beta-tradeoff` **The β trade-off** — Tune the KL weight β to trade reconstruction quality against latent-space regularity

**Frontier reading**

- Kingma & Welling (2014). Auto-Encoding Variational Bayes. ICLR. — The original VAE paper
- van den Oord, Vinyals, Kavukcuoglu (2017). Neural Discrete Representation Learning. NeurIPS. — VQ-VAE: discrete latents, the basis of modern image / audio tokenizers

<a id="ch-25"></a>

### 25 Diffusion Models

- **Paradigm**: Generative
- **Source chapters**: UDL 18
- **Prerequisites**: [13 Markov Chain Methods: MCMC, PageRank and Topic Models](#ch-13), [24 Variational Autoencoders](#ch-24)

**Learning goals**

- See diffusion as a hierarchical VAE: a fixed noising encoder plus a learned denoising decoder
- Derive how the ELBO reduces to a noise-prediction MSE
- Understand score functions, Langevin sampling, the SDE view, and classifier-free guidance

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Denoising network $\epsilon_\theta(x_t,t)$ | Denoising score matching (weighted ELBO) | Stochastic-gradient training + iterative sampling (ancestral / ODE solvers) |

**Rigorous derivations**

1. Closed-form forward marginal $q(x_t|x_0)$ and posterior $q(x_{t-1}|x_t,x_0)$
2. ELBO decomposition ⇒ simplified loss $\|\epsilon-\epsilon_\theta(x_t,t)\|^2$
3. Tweedie's formula: the denoiser is the score

**Interactive visuals**

- `viz-diffusion-2d` **2-D toy diffusion** — Scrub time t to morph a particle cloud between data and Gaussian, overlaid with the learned score field
- `viz-sampler-compare` **Sampler comparison** — Compare stochastic (SDE) and deterministic (ODE / DDIM) sampling trajectories and step–quality curves
- `viz-cfg` **Classifier-free guidance** — Tune guidance scale w to trade diversity against fidelity

**Frontier reading**

- Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models. NeurIPS. — DDPM
- Song et al. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. ICLR. — The unifying SDE view
- Ho & Salimans (2022). Classifier-Free Diffusion Guidance. arXiv. — Classifier-free guidance
- Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR. — Latent diffusion (Stable Diffusion)
- Peebles & Xie (2023). Scalable Diffusion Models with Transformers. ICCV. — DiT: diffusion + Transformers

## Part VII: Decision-Making and Large Models

*Reinforcement learning, and large language models as the synthesis of the whole course.*

<a id="ch-26"></a>

### 26 Reinforcement Learning

- **Paradigm**: RL
- **Source chapters**: UDL 19
- **Prerequisites**: [13 Markov Chain Methods: MCMC, PageRank and Topic Models](#ch-13), [16 Optimization: From Gradient Descent to Adam](#ch-16)

**Learning goals**

- Master MDPs, the Bellman equations, value iteration and policy iteration
- Understand the two routes: Q-learning / DQN and policy gradients / PPO

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Policy $\pi_\theta(a|s)$ and / or value function | Maximize expected cumulative return | Dynamic programming / temporal difference / policy gradient |

**Rigorous derivations**

1. The Bellman optimality operator is a γ-contraction ⇒ value iteration converges
2. The policy gradient theorem and variance reduction with baselines
3. Motivation for PPO's clipped objective

**Interactive visuals**

- `viz-gridworld` **Gridworld** — Edit walls and rewards, step through value iteration with cells coloured by V(s) and arrows for the greedy policy; tune γ
- `viz-policy-gradient` **Variance of policy gradients** — On a bandit, compare gradient-estimate distributions and learning curves with / without a baseline

**Frontier reading**

- Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature. — DQN
- Schulman, Wolski, Dhariwal, Radford, Klimov (2017). Proximal Policy Optimization Algorithms. arXiv. — PPO
- Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv. — Introduces GRPO: group-relative policy optimization without a value network

<a id="ch-27"></a>

### 27 Large Language Models: Putting the Course Together

- **Paradigm**: Frontier
- **Source chapters**: Original (no source-book chapter)
- **Prerequisites**: [15 Loss Functions as Maximum Likelihood](#ch-15), [16 Optimization: From Gradient Descent to Adam](#ch-16), [20 Transformers and Attention](#ch-20), [26 Reinforcement Learning](#ch-26)

**Learning goals**

- Revisit LLM pre-training, fine-tuning and alignment through the triad
- Understand scaling laws and compute-optimal training
- Understand how DPO is derived from the RLHF objective

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| Decoder-only Transformer (Ch. 20) | Pre-training: next-token cross-entropy (Ch. 15); alignment: KL-regularized reward maximization (Ch. 26) | AdamW (Ch. 16) + PPO / DPO / GRPO |

**Rigorous derivations**

1. Power-law scaling and the compute-optimal split between parameters and data
2. Closed-form optimal policy of KL-regularized reward maximization ⇒ the DPO loss

**Interactive visuals**

- `viz-scaling-laws` **Scaling-law calculator** — Enter a compute budget and locate the compute-optimal point on loss contours over parameters and data
- `viz-next-token` **Next-token distribution** — Show a small model's next-token probabilities; tune temperature, top-k and top-p to see sampling change

**Frontier reading**

- Kaplan et al. (2020). Scaling Laws for Neural Language Models. arXiv. — Scaling laws
- Hoffmann et al. (2022). Training Compute-Optimal Large Language Models. NeurIPS. — Chinchilla: compute-optimal training
- Ouyang et al. (2022). Training language models to follow instructions with human feedback. NeurIPS. — InstructGPT / RLHF
- Rafailov et al. (2023). Direct Preference Optimization. NeurIPS. — DPO

## Part VIII: Doing Research

*From using ML to researching it: reliable experiments, honest conclusions, responsible impact.*

<a id="ch-28"></a>

### 28 Experimental Design, Evaluation and Multiple Testing

- **Paradigm**: Methodology
- **Source chapters**: ISL 5, 13; UDL 8
- **Prerequisites**: [02 Generalization, Bias–Variance and Model Selection](#ch-02)

**Learning goals**

- Master reporting baselines, ablations, seed variance and confidence intervals
- Understand multiple testing, FWER and FDR control, and avoid leaderboard false positives
- Learn a workflow for reading and reproducing a paper

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| — | — | Bonferroni / Holm / Benjamini–Hochberg / permutation tests |

**Rigorous derivations**

1. Proof that Bonferroni controls the FWER
2. The Benjamini–Hochberg procedure and the FDR

**Interactive visuals**

- `viz-p-hacking` **p-hacking simulator** — Try many “tricks” on pure-null data, watch false positives inflate, then turn on BH correction
- `viz-seed-variance` **Seed variance** — Run one configuration over many seeds and show how misleading a single run can be

**Frontier reading**

- Benjamini & Hochberg (1995). Controlling the False Discovery Rate. JRSS-B. — FDR control
- Bouthillier et al. (2021). Accounting for Variance in Machine Learning Benchmarks. MLSys. — Sources of variance in ML benchmarks

<a id="ch-29"></a>

### 29 Responsible Machine Learning

- **Paradigm**: Methodology
- **Source chapters**: UDL 21
- **Prerequisites**: none

**Learning goals**

- Recognize bias, the incompatibility of fairness definitions, privacy and explainability
- Build the habit of reflecting on the impact of one's own research

**The triad**

| Model | Strategy | Algorithm |
|---|---|---|
| — | Risk with fairness / privacy constraints | — |

**Rigorous derivations**

1. With unequal base rates, calibration and equal error rates cannot both hold

**Interactive visuals**

- `viz-fairness-tradeoff` **Fairness trade-offs** — Drag per-group thresholds and watch fairness metrics update live, experiencing their incompatibility

**Frontier reading**

- Kleinberg, Mullainathan, Raghavan (2017). Inherent Trade-Offs in the Fair Determination of Risk Scores. ITCS. — An impossibility theorem for fairness

<a id="index"></a>

## Cross-indices

### By task

- **Regression**: [01 The Learning Problem: Model, Strategy, Algorithm](#ch-01), [02 Generalization, Bias–Variance and Model Selection](#ch-02), [03 Linear Regression and Regularization](#ch-03), [06 Non-parametric Methods and Basis Expansions: kNN, Splines and GAMs](#ch-06), [08 Decision Trees and Ensemble Learning](#ch-08), [14 From Shallow to Deep Networks](#ch-14), [15 Loss Functions as Maximum Likelihood](#ch-15)
- **Classification**: [01 The Learning Problem: Model, Strategy, Algorithm](#ch-01), [02 Generalization, Bias–Variance and Model Selection](#ch-02), [04 Linear Classification: Perceptron, Logistic Regression and Maximum Entropy](#ch-04), [05 Generative Classifiers: Naive Bayes and Discriminant Analysis](#ch-05), [06 Non-parametric Methods and Basis Expansions: kNN, Splines and GAMs](#ch-06), [07 Support Vector Machines and Kernel Methods](#ch-07), [08 Decision Trees and Ensemble Learning](#ch-08), [14 From Shallow to Deep Networks](#ch-14), [15 Loss Functions as Maximum Likelihood](#ch-15), [19 Convolutional and Residual Networks](#ch-19), [21 Graph Neural Networks](#ch-21)
- **Sequence labeling**: [09 Structured Prediction: Hidden Markov Models and Conditional Random Fields](#ch-09), [20 Transformers and Attention](#ch-20), [27 Large Language Models: Putting the Course Together](#ch-27)
- **Clustering**: [10 Clustering](#ch-10), [11 The EM Algorithm and Mixture Models](#ch-11), [13 Markov Chain Methods: MCMC, PageRank and Topic Models](#ch-13)
- **Dimensionality reduction**: [12 Matrix Factorization and Dimensionality Reduction: SVD, PCA and LSA](#ch-12), [24 Variational Autoencoders](#ch-24)
- **Density estimation**: [11 The EM Algorithm and Mixture Models](#ch-11), [13 Markov Chain Methods: MCMC, PageRank and Topic Models](#ch-13), [23 Normalizing Flows](#ch-23)
- **Generation**: [20 Transformers and Attention](#ch-20), [22 A Map of Generative Models, and GANs](#ch-22), [23 Normalizing Flows](#ch-23), [24 Variational Autoencoders](#ch-24), [25 Diffusion Models](#ch-25), [27 Large Language Models: Putting the Course Together](#ch-27)
- **Ranking**: [13 Markov Chain Methods: MCMC, PageRank and Topic Models](#ch-13)
- **Sequential decision-making**: [26 Reinforcement Learning](#ch-26)
- **Graph data**: [21 Graph Neural Networks](#ch-21)

### By strategy (loss)

- **Squared loss**: [01 The Learning Problem: Model, Strategy, Algorithm](#ch-01), [03 Linear Regression and Regularization](#ch-03), [06 Non-parametric Methods and Basis Expansions: kNN, Splines and GAMs](#ch-06), [08 Decision Trees and Ensemble Learning](#ch-08), [15 Loss Functions as Maximum Likelihood](#ch-15)
- **Cross-entropy / log-likelihood**: [04 Linear Classification: Perceptron, Logistic Regression and Maximum Entropy](#ch-04), [09 Structured Prediction: Hidden Markov Models and Conditional Random Fields](#ch-09), [15 Loss Functions as Maximum Likelihood](#ch-15), [19 Convolutional and Residual Networks](#ch-19), [20 Transformers and Attention](#ch-20), [21 Graph Neural Networks](#ch-21), [27 Large Language Models: Putting the Course Together](#ch-27)
- **Perceptron loss**: [04 Linear Classification: Perceptron, Logistic Regression and Maximum Entropy](#ch-04)
- **Hinge loss**: [07 Support Vector Machines and Kernel Methods](#ch-07)
- **Exponential loss**: [08 Decision Trees and Ensemble Learning](#ch-08)
- **Impurity (entropy / Gini)**: [08 Decision Trees and Ensemble Learning](#ch-08)
- **Maximum entropy**: [04 Linear Classification: Perceptron, Logistic Regression and Maximum Entropy](#ch-04)
- **(Marginal) likelihood**: [05 Generative Classifiers: Naive Bayes and Discriminant Analysis](#ch-05), [09 Structured Prediction: Hidden Markov Models and Conditional Random Fields](#ch-09), [11 The EM Algorithm and Mixture Models](#ch-11), [13 Markov Chain Methods: MCMC, PageRank and Topic Models](#ch-13), [15 Loss Functions as Maximum Likelihood](#ch-15), [23 Normalizing Flows](#ch-23)
- **Evidence lower bound (ELBO)**: [11 The EM Algorithm and Mixture Models](#ch-11), [24 Variational Autoencoders](#ch-24), [25 Diffusion Models](#ch-25)
- **Reconstruction error**: [12 Matrix Factorization and Dimensionality Reduction: SVD, PCA and LSA](#ch-12)
- **Within-cluster sum of squares**: [10 Clustering](#ch-10)
- **Adversarial loss**: [22 A Map of Generative Models, and GANs](#ch-22)
- **Denoising / score matching**: [25 Diffusion Models](#ch-25)
- **Expected return**: [26 Reinforcement Learning](#ch-26), [27 Large Language Models: Putting the Course Together](#ch-27)
- **Regularization / structural risk**: [01 The Learning Problem: Model, Strategy, Algorithm](#ch-01), [02 Generalization, Bias–Variance and Model Selection](#ch-02), [03 Linear Regression and Regularization](#ch-03), [06 Non-parametric Methods and Basis Expansions: kNN, Splines and GAMs](#ch-06), [07 Support Vector Machines and Kernel Methods](#ch-07), [18 Regularization and the Generalization Puzzle in Deep Networks](#ch-18)

### By algorithm (optimization / inference)

- **Closed form**: [01 The Learning Problem: Model, Strategy, Algorithm](#ch-01), [03 Linear Regression and Regularization](#ch-03), [05 Generative Classifiers: Naive Bayes and Discriminant Analysis](#ch-05), [06 Non-parametric Methods and Basis Expansions: kNN, Splines and GAMs](#ch-06)
- **Gradient descent**: [00 Geometric Intuition for Linear Algebra, Probability and Optimization](#ch-00), [01 The Learning Problem: Model, Strategy, Algorithm](#ch-01), [03 Linear Regression and Regularization](#ch-03), [16 Optimization: From Gradient Descent to Adam](#ch-16)
- **Stochastic gradient descent**: [04 Linear Classification: Perceptron, Logistic Regression and Maximum Entropy](#ch-04), [16 Optimization: From Gradient Descent to Adam](#ch-16), [18 Regularization and the Generalization Puzzle in Deep Networks](#ch-18), [19 Convolutional and Residual Networks](#ch-19), [22 A Map of Generative Models, and GANs](#ch-22), [26 Reinforcement Learning](#ch-26)
- **Adaptive methods (Adam etc.)**: [16 Optimization: From Gradient Descent to Adam](#ch-16), [20 Transformers and Attention](#ch-20), [22 A Map of Generative Models, and GANs](#ch-22), [23 Normalizing Flows](#ch-23), [24 Variational Autoencoders](#ch-24), [25 Diffusion Models](#ch-25), [27 Large Language Models: Putting the Course Together](#ch-27)
- **Newton / quasi-Newton**: [00 Geometric Intuition for Linear Algebra, Probability and Optimization](#ch-00), [04 Linear Classification: Perceptron, Logistic Regression and Maximum Entropy](#ch-04), [09 Structured Prediction: Hidden Markov Models and Conditional Random Fields](#ch-09), [16 Optimization: From Gradient Descent to Adam](#ch-16)
- **Coordinate descent**: [03 Linear Regression and Regularization](#ch-03)
- **Duality & QP (SMO)**: [07 Support Vector Machines and Kernel Methods](#ch-07)
- **Greedy splitting**: [08 Decision Trees and Ensemble Learning](#ch-08), [10 Clustering](#ch-10)
- **Forward stagewise additive**: [08 Decision Trees and Ensemble Learning](#ch-08)
- **Dynamic programming**: [09 Structured Prediction: Hidden Markov Models and Conditional Random Fields](#ch-09), [26 Reinforcement Learning](#ch-26)
- **EM algorithm**: [09 Structured Prediction: Hidden Markov Models and Conditional Random Fields](#ch-09), [11 The EM Algorithm and Mixture Models](#ch-11), [13 Markov Chain Methods: MCMC, PageRank and Topic Models](#ch-13)
- **Alternating minimization (Lloyd)**: [10 Clustering](#ch-10)
- **Eigendecomposition / SVD**: [12 Matrix Factorization and Dimensionality Reduction: SVD, PCA and LSA](#ch-12)
- **MCMC sampling**: [13 Markov Chain Methods: MCMC, PageRank and Topic Models](#ch-13), [25 Diffusion Models](#ch-25)
- **Power iteration**: [13 Markov Chain Methods: MCMC, PageRank and Topic Models](#ch-13)
- **Backpropagation**: [17 Backpropagation and Initialization](#ch-17), [19 Convolutional and Residual Networks](#ch-19), [20 Transformers and Attention](#ch-20), [21 Graph Neural Networks](#ch-21)
- **Policy gradient**: [26 Reinforcement Learning](#ch-26), [27 Large Language Models: Putting the Course Together](#ch-27)

### Source-book coverage

**ISL**

| Book chapter | Our chapters |
|---|---|
| 1 | *Not included: Introduction, folded into Ch. 01* |
| 2 | [01 The Learning Problem: Model, Strategy, Algorithm](#ch-01), [02 Generalization, Bias–Variance and Model Selection](#ch-02), [06 Non-parametric Methods and Basis Expansions: kNN, Splines and GAMs](#ch-06) |
| 3 | [03 Linear Regression and Regularization](#ch-03) |
| 4 | [04 Linear Classification: Perceptron, Logistic Regression and Maximum Entropy](#ch-04), [05 Generative Classifiers: Naive Bayes and Discriminant Analysis](#ch-05) |
| 5 | [02 Generalization, Bias–Variance and Model Selection](#ch-02), [28 Experimental Design, Evaluation and Multiple Testing](#ch-28) |
| 6 | [03 Linear Regression and Regularization](#ch-03) |
| 7 | [06 Non-parametric Methods and Basis Expansions: kNN, Splines and GAMs](#ch-06) |
| 8 | [08 Decision Trees and Ensemble Learning](#ch-08) |
| 9 | [07 Support Vector Machines and Kernel Methods](#ch-07) |
| 10 | [14 From Shallow to Deep Networks](#ch-14) |
| 11 | *Not included: Survival analysis: applied-statistics topic, outside the main line for now* |
| 12 | [10 Clustering](#ch-10), [12 Matrix Factorization and Dimensionality Reduction: SVD, PCA and LSA](#ch-12) |
| 13 | [28 Experimental Design, Evaluation and Multiple Testing](#ch-28) |

**UDL**

| Book chapter | Our chapters |
|---|---|
| 1 | *Not included: Introduction, folded into Ch. 01* |
| 2 | [01 The Learning Problem: Model, Strategy, Algorithm](#ch-01), [03 Linear Regression and Regularization](#ch-03) |
| 3 | [14 From Shallow to Deep Networks](#ch-14) |
| 4 | [14 From Shallow to Deep Networks](#ch-14) |
| 5 | [15 Loss Functions as Maximum Likelihood](#ch-15) |
| 6 | [16 Optimization: From Gradient Descent to Adam](#ch-16) |
| 7 | [17 Backpropagation and Initialization](#ch-17) |
| 8 | [02 Generalization, Bias–Variance and Model Selection](#ch-02), [18 Regularization and the Generalization Puzzle in Deep Networks](#ch-18), [28 Experimental Design, Evaluation and Multiple Testing](#ch-28) |
| 9 | [18 Regularization and the Generalization Puzzle in Deep Networks](#ch-18) |
| 10 | [19 Convolutional and Residual Networks](#ch-19) |
| 11 | [19 Convolutional and Residual Networks](#ch-19) |
| 12 | [20 Transformers and Attention](#ch-20) |
| 13 | [21 Graph Neural Networks](#ch-21) |
| 14 | [22 A Map of Generative Models, and GANs](#ch-22) |
| 15 | [22 A Map of Generative Models, and GANs](#ch-22) |
| 16 | [23 Normalizing Flows](#ch-23) |
| 17 | [24 Variational Autoencoders](#ch-24) |
| 18 | [25 Diffusion Models](#ch-25) |
| 19 | [26 Reinforcement Learning](#ch-26) |
| 20 | [18 Regularization and the Generalization Puzzle in Deep Networks](#ch-18) |
| 21 | [29 Responsible Machine Learning](#ch-29) |
| App. B | [00 Geometric Intuition for Linear Algebra, Probability and Optimization](#ch-00) |
| App. C | [00 Geometric Intuition for Linear Algebra, Probability and Optimization](#ch-00) |

**LH**

| Book chapter | Our chapters |
|---|---|
| 1 | [01 The Learning Problem: Model, Strategy, Algorithm](#ch-01), [02 Generalization, Bias–Variance and Model Selection](#ch-02) |
| 2 | [04 Linear Classification: Perceptron, Logistic Regression and Maximum Entropy](#ch-04) |
| 3 | [06 Non-parametric Methods and Basis Expansions: kNN, Splines and GAMs](#ch-06) |
| 4 | [05 Generative Classifiers: Naive Bayes and Discriminant Analysis](#ch-05) |
| 5 | [08 Decision Trees and Ensemble Learning](#ch-08) |
| 6 | [04 Linear Classification: Perceptron, Logistic Regression and Maximum Entropy](#ch-04) |
| 7 | [07 Support Vector Machines and Kernel Methods](#ch-07) |
| 8 | [08 Decision Trees and Ensemble Learning](#ch-08) |
| 9 | [11 The EM Algorithm and Mixture Models](#ch-11) |
| 10 | [09 Structured Prediction: Hidden Markov Models and Conditional Random Fields](#ch-09) |
| 11 | [09 Structured Prediction: Hidden Markov Models and Conditional Random Fields](#ch-09) |
| 12 | *Not included: Summary of supervised methods: replaced by the site's cross-index views* |
| 13 | [10 Clustering](#ch-10) |
| 14 | [10 Clustering](#ch-10) |
| 15 | [12 Matrix Factorization and Dimensionality Reduction: SVD, PCA and LSA](#ch-12) |
| 16 | [12 Matrix Factorization and Dimensionality Reduction: SVD, PCA and LSA](#ch-12) |
| 17 | [12 Matrix Factorization and Dimensionality Reduction: SVD, PCA and LSA](#ch-12) |
| 18 | [13 Markov Chain Methods: MCMC, PageRank and Topic Models](#ch-13) |
| 19 | [13 Markov Chain Methods: MCMC, PageRank and Topic Models](#ch-13) |
| 20 | [13 Markov Chain Methods: MCMC, PageRank and Topic Models](#ch-13) |
| 21 | [13 Markov Chain Methods: MCMC, PageRank and Topic Models](#ch-13) |
| 22 | *Not included: Summary of unsupervised methods: replaced by the site's cross-index views* |
| 附录 A | [16 Optimization: From Gradient Descent to Adam](#ch-16) |
| 附录 A–E | [00 Geometric Intuition for Linear Algebra, Probability and Optimization](#ch-00) |
| 附录 B | [16 Optimization: From Gradient Descent to Adam](#ch-16) |
