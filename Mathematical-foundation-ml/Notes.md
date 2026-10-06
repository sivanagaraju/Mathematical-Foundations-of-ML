# Mathematical Foundations of Machine Learning

NPTEL / IISc Bangalore · Course **106108841** · noc26-cs02 · 12 weeks · Prof. Prathosh A P

**Playlist:** [Mathematical Foundations of Machine Learning](https://www.youtube.com/playlist?list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu)  
**Channel:** NPTEL — Indian Institute of Science, Bengaluru  
**Size:** 89 videos · ~46.6 hours  
**Catalog Status:** All 89 videos cataloged, reconciled with syllabus blocks, mapped to learning sequences & study packages.  
**Production Status:** 33 packages fully completed (7-Pillar production standard), 56 mapped on roadmap.

This file is the **master course map + catalog** for the parent NPTEL recording. Per-lecture study packages (`PREREQUISITES.md`, `NOTES.md`, `references.md`, `examples/*.py`, `glossary.md`, `formulae_sheet.md`, `quiz.html`) sit in numbered folders under [`Mathematical-foundation-ml/`](./Mathematical-foundation-ml/) and are linked below.

This is the **first** of a two-course sequence. The sequel is Mathematical Foundations of **Generative AI**:

- NPTEL recording: [`Mathematical-Foundation-for-GenerativeAI/NOTES.md`](./Mathematical-Foundation-for-GenerativeAI/NOTES.md)
- IIT Madras BS recording: [`IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md`](./IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md)
- Core Mathematical Knowledge Base: [`MathsTerms/README.md`](./MathsTerms/README.md) · Concept Map: [`MathsTerms/CONCEPT_MAP.md`](./MathsTerms/CONCEPT_MAP.md)

---

## Table of Contents

1. [What the course is for](#what-the-course-is-for)
2. [Architecture of the whole course](#architecture-of-the-whole-course)
   - [Master Architecture Blueprint](#master-architecture-blueprint)
   - [Uploaded so far vs syllabus coverage](#uploaded-so-far-vs-syllabus-coverage)
   - [What later blocks cannot skip](#what-later-blocks-cannot-skip)
3. [How the blocks fit](#how-the-blocks-fit)
4. [Study packages status and catalog (all 89 videos)](#study-packages-status-and-catalog-all-89-videos)
5. [Playlist catalog (learning order)](#playlist-catalog-learning-order)
   - [Block 1: Intro and function approximation](#block-1-intro-and-function-approximation)
   - [Block 2: Probability recap](#block-2-probability-recap)
   - [Block 3: Data as samples; estimation; density](#block-3-data-as-samples-estimation-density)
   - [Block 4: Python / probability tutorials](#block-4-python--probability-tutorials)
   - [Block 5: Entropy, KL, min-KL](#block-5-entropy-kl-min-kl)
   - [Block 6: Risk minimization and Bayes](#block-6-risk-minimization-and-bayes)
   - [Block 7: MLE, latent variables, EM](#block-7-mle-latent-variables-em)
   - [Block 8: Classifier / MLE tutorials](#block-8-classifier--mle-tutorials)
   - [Block 9: EM, MAP, nonparametric](#block-9-em-map-nonparametric)
   - [Block 10: Linear models and bias–variance](#block-10-linear-models-and-biasvariance)
   - [Block 11: Regularization and SVM](#block-11-regularization-and-svm)
   - [Block 12: Neural nets, CNN, RNN](#block-12-neural-nets-cnn-rnn)
   - [Block 13: PyTorch tutorials](#block-13-pytorch-tutorials)
   - [Block 14: Attention, Transformers, transfer, optimizers](#block-14-attention-transformers-transfer-optimizers)
   - [Block 15: CNN / RNN tutorials](#block-15-cnn--rnn-tutorials)
   - [Block 16: Trees, ensembles, cross-validation](#block-16-trees-ensembles-cross-validation)
   - [Block 17: Unsupervised and contrastive](#block-17-unsupervised-and-contrastive)
   - [Block 18: Bridge to generative AI](#block-18-bridge-to-generative-ai)
6. [YouTube playlist order (chronological 1 to 89)](#youtube-playlist-order-chronological-1-to-89)
7. [Compact title → URL list](#compact-title--url-list)
8. [External resources](#external-resources)
9. [Sources](#sources)

---

## What the course is for

Machine learning is treated as **function approximation under uncertainty**, not as an arbitrary zoo of disconnected heuristics or named algorithms.

You hold a finite dataset of observations:
$$\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\} \quad \text{or} \quad \mathcal{D} = \{x_1, x_2, \dots, x_N\}$$

You must answer a query on a **new input** $x^*$ you have never seen before.
1. **Memorizing the table fails:** Generalization is impossible by pure lookup in high-dimensional continuous domains.
2. **Deterministic physical equations fail:** Semantic data (e.g., whether a chest X-ray displays cardiomegaly) cannot be modeled by first-principles mechanics.

The course's foundational fork:
- Treat inputs and targets as **random variables** $X, Y$ defined over a probability space $(\Omega, \mathcal{F}, P)$.
- Recognize that data $\mathcal{D}$ consists of **independent and identically distributed (I.I.D.) draws** from an unknown true data law $p_{X,Y}$ (or $p_X$).
- Estimate the law or its decision boundary by **minimizing a divergence or risk function**.
- Derive every major ML architecture (Linear models, SVMs, Neural Networks, CNNs, Transformers, Trees, GMMs, Contrastive Learning) as an instance of this unified statistical recipe.

`Lec` = chalk-and-talk rigorous mathematical derivation.  
`Tutorial` = worked numerical examples, mathematical problem solving, and PyTorch implementations.

---

## Architecture of the whole course

### Master Architecture Blueprint

```
                      NATURE / PHYSICAL WORLD
            Hidden Random Experiment: (Ω, F, P)
                           │
             Sensors / Measurement Maps: X, Y: Ω → R^d
                           │
                           ▼
             OBSERVED FINITE DATASET D
         D = {(x_i, y_i)}_{i=1}^N  ~  p_{X,Y} (I.I.D.)
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
 1. PARAMETRIC MODEL FAMILY           2. DISCREPANCY / OBJECTIVE
    Pick hypothesis class F:             Pick mathematical loss:
    • Density: p_θ(x), p_θ(x|z)          • KL Divergence / Cross-Entropy
    • Classifier: g_θ(x)                 • Empirical Risk: 1/N ∑ ℓ(y_i, g(x_i))
    • Deep Net / CNN / Transformer       • Margin / Hinge / MSE / Log-Loss
        │                                     │
        └──────────────────┬──────────────────┘
                           ▼
 3. OPTIMIZATION / ESTIMATION ENGINE
    θ* = arg min_θ  [ d(p_data, p_θ)  +  λ · Regularizer(θ) ]
    • Closed-form: OLS (X^T X)^{-1} X^T y, Gaussian MLE
    • Iterative / Latent: EM Algorithm (E-step Q, M-step argmax)
    • Gradient-based: Backpropagation + SGD / RMSprop / Adam
    • Convex Dual: Quadratic Programming / Lagrange Multipliers (SVM)
                           │
                           ▼
 4. DEPLOYED INFERENCE ARTIFACT
    • Discriminative Decision: g*(x) = arg max_y P(Y=y | X=x)  (Bayes optimal)
    • Nonparametric Density: Parzen windows, k-NN
    • Latent Representation: Principal Subspace (PCA), Embeddings (InfoNCE)
    • Generative Sampling: Hand-off to GenAI Sequel (GAN, VAE, Diffusion, LLMs)
```

### Uploaded so far vs syllabus coverage

```
  PHASE 1: PROBABILISTIC FOUNDATIONS & ESTIMATION [COMPLETED PACKAGES 02–14]
  Lec 01        Function Approximation: Physics vs Statistics
  Lec 02–05     Probability Recap: Triplet (Ω, F, P), RV X: Ω → R^d, CDF, Joints
  Lec 06–10     Data as Samples: Chest X-Ray in range(X), IID, Density vs Probability
  Tut 01–02     Python Basics & Probability Problem Solving
  Lec 11–13     Information Theory: Surprisal, Entropy, KL Divergence, min-KL ≡ MLE
                │
                ▼
  PHASE 2: RISK, MLE & LATENT VARIABLE MODELS [PACKAGES 15–38]
  Lec 14–16     MLE Concrete Example, Risk Minimization R(g), Bayes Optimal Classifier
  Tut 03–06     Risk Minimization, Minimax Classifiers, Neyman-Pearson & ROC Curves
  Lec 17–22     Parametric MLE: Gaussian, Discrete, Mixed laws; Latent Variables & EM
  Tut 07–09     Gaussian/Discrete MLE in code; EM for GMMs; MAP estimation
  Lec 23–27     EM Convergence, GMMs, MAP with Priors, Parzen Windows, k-NN
                │
                ▼
  PHASE 3: LINEAR MODELS, REGULARIZATION & SVM [PACKAGES 39–53]
  Lec 28–32     OLS, GLS, Linear Classification, Bias-Variance Decomposition
  Tut 10A/B     Numerical Bayes Classifier, MLE vs MAP side-by-side
  Lec 33–35     Regularization: Ridge/Lasso, Regularized ERM ≡ MAP, SGD as regularizer
  Lec 36–40     SVM: Max-margin, Primal slack, Lagrangian Dual, Kernel Trick k(x, x')
                │
                ▼
  PHASE 4: DEEP LEARNING, CONVOLUTIONS & RECURRENCE [PACKAGES 54–63]
  Lec 41–42     Neural Networks, Universal Approximation Theorem (UAT), Backpropagation
  Lec 43–44     Inductive Bias: Local Receptive Fields, Parameter Sharing, CNN as Reg-MLP
  Lec 45–47     Sequence Modeling: RNNs, BPTT, Vanishing Gradients, LSTMs & GRUs
  Tut 11–13     PyTorch Tensors, DataLoader, Building MLPs, Autograd, Training Loops
                │
                ▼
  PHASE 5: ATTENTION, TRANSFORMERS & OPTIMIZERS [PACKAGES 64–73]
  Lec 48–51     Attention Mechanism (Q, K, V), Multi-Head Attention, Transformers, Positional Encoding
  Lec 52–53     Transfer Learning, Knowledge Distillation, Modern Optimizers (SGD, RMSProp, Adam)
  Tut 14–15     PyTorch CNNs, Transfer Learning, RNN/LSTM/GRU Sequence Models
                │
                ▼
  PHASE 6: TREES, ENSEMBLES & UNSUPERVISED LEARNING [PACKAGES 74–84]
  Lec 54–59     Decision Trees, Impurity (Gini/Entropy), Regression Trees, Bagging, Gradient Boosting, AdaBoost, CV
  Lec 60–64     Unsupervised Learning, K-Means, PCA, Noise Contrastive Estimation (NCE), InfoNCE, SimCLR, JEPA
                │
                ▼
  PHASE 7: BRIDGE TO GENERATIVE AI [PACKAGES 85–89]
  Lec 65–69     Generative Models, GANs, VAEs, Large Language Models (LLMs), Reinforcement Learning (RL)
                → Handoff to "Mathematical Foundations of Generative AI"
```

### What later blocks cannot skip

| Block | Load-bearing claim | Failure mode if skipped |
|:------|:-------------------|:------------------------|
| **FA (Lec 01)** | Model $\neq$ algorithm; table lookup is not function approximation. | Treating deep nets as magic black boxes rather than statistical estimators. |
| **Probability (Lec 02–05)** | Triplet $(\Omega, \mathcal{F}, P)$; RV $X: \Omega \to \mathbb{R}^d$ pushes measure to data space. | Confusing observable digital files with the underlying generating mechanism. |
| **Data & Density (Lec 06–10)** | Dataset $\sim p$; probability density height $p(x)$ is **not** a probability. | Calling a pixel density value a probability; dividing by zero when evaluating continuous likelihoods. |
| **KL Divergence (Lec 11–13)** | Train by $\min d(p, p_\theta)$; $\min \mathrm{KL}(p \parallel p_\theta) \equiv \max \text{Likelihood}$. | Inventing arbitrary ad-hoc loss functions with no probabilistic discrepancy justification. |
| **Risk & Bayes (Lec 15–16)** | Classifier is decision rule $g(x)$; Bayes rule minimizes 0–1 risk. | Optimizing accuracy metrics without defining a loss function or cost matrix. |
| **EM & Latent Models (Lec 20–26)** | Incomplete data likelihood $\log \int p(x,z) dz$ is intractable; optimize the ELBO bound $Q$. | Trying to optimize incomplete log-likelihood directly with naive gradient descent. |
| **Regularization (Lec 33–35)** | ERM overfits; explicit penalties, MAP estimation, and SGD noise solve the same statistical problem. | Hand-tuning weight decay without understanding prior distributions. |
| **SVM & Duality (Lec 36–40)** | Primal max-margin optimization dualizes to inner products $\langle x_i, x_j \rangle$ enabling kernels. | Forcing high-dimensional explicit feature mappings instead of implicit kernels. |
| **Deep Nets (Lec 41–44)** | CNNs and RNNs are not new fundamental models; they are MLPs regularized by weight tying. | Viewing CNNs/RNNs as ad-hoc hacks rather than structured inductive biases. |
| **GenAI Coda (Lec 85–89)** | Generative modeling requires estimating $p_X$ **and** sampling new draws $x \sim p_X$. | Jumping into diffusion models and LLMs without grounding in likelihood and latent spaces. |

---

## How the blocks fit

| Block | Videos | Thematic Scope & Core Deliverables |
|:-----:|:------:|:-----------------------------------|
| **Block 1: Intro & FA** | 1–2 | Course scope; function approximation under uncertainty; physics vs statistical models. |
| **Block 2: Probability** | 3–6 | Measure space $(\Omega, \mathcal{F}, P)$, Random variables $X: \Omega \to \mathbb{R}^d$, pushforward measure, CDFs, joints, conditionals, marginals. |
| **Block 3: Data & Density** | 7–11 | Digital files in range$(X)$; I.I.D. assumption; density $p(x) \neq P$; parametric recipe: family $p_\theta \to$ divergence $d \to \arg\min$. |
| **Block 4: Python & Probability Labs** | 12–13 | Hands-on Python programming; numerical simulations of random variables, coin tosses, Bayes rule. |
| **Block 5: Info Theory & KL** | 14–16 | Surprisal $-\log p$, Shannon entropy $H$, Cross-Entropy, KL Divergence; equivalence of min-KL and MLE. |
| **Block 6: Risk & Bayes** | 17–20 | Concrete MLE derivations; Expected Risk $R(g)$; Bayes optimal classifier; worked risk minimization. |
| **Block 7: MLE & Latent Variables** | 21–26 | Gaussian MLE, Categorical/Discrete MLE, mixed distributions; latent variable models $p(x) = \int p(x,z)dz$; EM derivation. |
| **Block 8: Classifier Tutorials** | 27–31 | Minimax classifiers for adversarial priors; Neyman-Pearson likelihood ratios; ROC curves; numeric Gaussian/Discrete MLE. |
| **Block 9: EM, MAP & Nonparametrics** | 32–38 | EM convergence proof; GMM clustering; MAP estimation as penalized likelihood; Parzen window KDE; $k$-NN voting. |
| **Block 10: Linear Models & Bias-Variance** | 39–45 | Ordinary Least Squares (OLS) normal equations; Generalized Least Squares (GLS); linear classifiers; Bias-Variance-Noise decomposition. |
| **Block 11: Regularization & SVM** | 46–53 | L2 ridge penalty $\equiv$ Gaussian prior; L1 lasso $\equiv$ Laplace prior; SGD noise as implicit regularizer; SVM max-margin, dual, kernel trick. |
| **Block 12: Neural Networks & Architectures** | 54–60 | Universal Approximation Theorem (UAT); Backpropagation; CNNs as regularized MLPs (local receptive fields, parameter sharing); RNNs, BPTT, LSTMs, GRUs. |
| **Block 13: PyTorch Deep Learning Labs** | 61–63 | PyTorch tensors, autograd engine, custom `nn.Module` subclasses, `Dataset`, `DataLoader`, complete model training pipelines. |
| **Block 14: Attention, Transformers & Optimizers** | 64–69 | Scaled dot-product attention (Q, K, V); Multi-Head Attention; Transformer block; Positional embeddings; Transfer learning & distillation; SGD, RMSprop, Adam. |
| **Block 15: CNN & RNN PyTorch Labs** | 70–73 | Implementing CNN image classifiers, fine-tuning pretrained backbones, coding vanilla RNNs, LSTMs, GRUs, and stacked sequence models. |
| **Block 16: Trees, Ensembles & Validation** | 74–79 | Decision tree recursive splitting; Gini impurity & entropy; regression trees; Bagging; Gradient Boosting; AdaBoost; $K$-fold cross-validation. |
| **Block 17: Unsupervised & Contrastive** | 80–84 | Unsupervised representation learning; $K$-Means clustering; Principal Component Analysis (PCA); Noise Contrastive Estimation (NCE); InfoNCE, SimCLR, JEPA. |
| **Block 18: Bridge to Generative AI** | 85–89 | Problem formulation: estimate $p_X$ + sample; GANs (min-max game); VAEs (ELBO); Autoregressive LLMs; Reinforcement Learning (RL) for alignment. |

---

## Study packages status and catalog (all 89 videos)

All production study packages in this repository strictly adhere to the **7-Pillar Production Learning Suite standard** (`PREREQUISITES.md`, `NOTES.md`, `references.md`, `examples/*.py`, `glossary.md`, `formulae_sheet.md`, `quiz.html`).

| Learn # | Video Title | YouTube Slot | Study Package Folder | Status |
|:-------:|:------------|:------------:|:---------------------|:------:|
| 1 | Mathematical Foundations of Machine Learning (Intro) | YT #1 | `[Planned: 01-Intro-Mathematical-Foundations-of-ML]` | Roadmap |
| 2 | Lec 01 Overview of Function Approximation | YT #2 | [`02-Lec01-Overview-Function-Approximation/`](./Mathematical-foundation-ml/02-Lec01-Overview-Function-Approximation/) | **Completed** |
| 3 | Lec 02 Recap of Probability Theory - 1, Part 1 | YT #3 | [`03-Lec02-Recap-Probability-Theory-Part1/`](./Mathematical-foundation-ml/03-Lec02-Recap-Probability-Theory-Part1/) | **Completed** |
| 4 | Lec 03 Recap of Probability Theory - 1, Part 2 | YT #4 | [`04-Lec03-Recap-Probability-Theory-Part2/`](./Mathematical-foundation-ml/04-Lec03-Recap-Probability-Theory-Part2/) | **Completed** |
| 5 | Lec 04 Recap of Probability Theory - 1, Part 3 | YT #5 | [`05-Lec04-Recap-Probability-Theory-Part3/`](./Mathematical-foundation-ml/05-Lec04-Recap-Probability-Theory-Part3/) | **Completed** |
| 6 | Lec 05 Recap of Probability Theory Part 2 | YT #6 | [`06-Lec05-Recap-Probability-Theory-Part2/`](./Mathematical-foundation-ml/06-Lec05-Recap-Probability-Theory-Part2/) | **Completed** |
| 7 | Lec 06 Understanding a Chest X-Ray as Sample from Distribution | YT #7 | [`07-Lec06-XRay-Sample-From-Distribution/`](./Mathematical-foundation-ml/07-Lec06-XRay-Sample-From-Distribution/) | **Completed** |
| 8 | Lec 07 IID Assumption | YT #8 | [`08-Lec07-IID-Assumption/`](./Mathematical-foundation-ml/08-Lec07-IID-Assumption/) | **Completed** |
| 9 | Lec 08 Distribution Estimation | YT #9 | [`09-Lec08-Distribution-Estimation/`](./Mathematical-foundation-ml/09-Lec08-Distribution-Estimation/) | **Completed** |
| 10 | Lec 09 Density Function | YT #10 | [`10-Lec09-Density-Function/`](./Mathematical-foundation-ml/10-Lec09-Density-Function/) | **Completed** |
| 11 | Lec 10 Challenge With ML | YT #11 | [`11-Lec10-Challenges-of-ML/`](./Mathematical-foundation-ml/11-Lec10-Challenges-of-ML/) | **Completed** |
| 12 | Tutorial 1 : Introduction to Python Basics | YT #12 | `[Planned: 15-Tutorial01-Python-Basics]` | Roadmap |
| 13 | Tutorial 2 : Simple Problem solving in Probability Theory | YT #13 | `[Planned: 16-Tutorial02-Simple-Problem-Solving-Probability]` | Roadmap |
| 14 | Lec 11 Entropy | YT #14 | [`12-Lec11-Entropy/`](./Mathematical-foundation-ml/12-Lec11-Entropy/) | **Completed** |
| 15 | Lec 12 Kullback-Leibler (KL) Divergence | YT #15 | [`13-Lec12-KL-Divergence/`](./Mathematical-foundation-ml/13-Lec12-KL-Divergence/) | **Completed** |
| 16 | Lec 13 Minimization of KL Divergence | YT #16 | [`14-Lec13-Minimization-of-KL/`](./Mathematical-foundation-ml/14-Lec13-Minimization-of-KL/) | **Completed** |
| 17 | Lec 14 Example of ML Estimate | YT #17 | `[Planned: 17-Lec14-Example-of-ML-Estimate]` | Roadmap |
| 18 | Lec 15 Risk Minimization Framework | YT #18 | `[Planned: 18-Lec15-Risk-Minimization-Framework]` | Roadmap |
| 19 | Lec 16 Bayes Classifier | YT #19 | `[Planned: 19-Lec16-Bayes-Classifier]` | Roadmap |
| 20 | Tutorail 3 : Risk Minimization Framework | YT #20 | `[Planned: 20-Tutorial03-Risk-Minimization-Framework]` | Roadmap |
| 21 | Lec 17 MLE for Gaussian Distribution | YT #21 | `[Planned: 21-Lec17-MLE-for-Gaussian-Distribution]` | Roadmap |
| 22 | Lec 18 MLE for Generalized Discrete Random Variable | YT #22 | `[Planned: 22-Lec18-MLE-Generalized-Discrete-RV]` | Roadmap |
| 23 | Lec 19 Density Estimation for Mixed Distribution | YT #23 | `[Planned: 23-Lec19-Density-Estimation-Mixed-Distribution]` | Roadmap |
| 24 | Lec 20 Latent Variable Models | YT #24 | `[Planned: 24-Lec20-Latent-Variable-Models]` | Roadmap |
| 25 | Lec 21 MLE for Latent Variable Models | YT #25 | `[Planned: 25-Lec21-MLE-for-Latent-Variable-Models]` | Roadmap |
| 26 | Lec 22 Expectation Maximization Algorithm | YT #26 | `[Planned: 26-Lec22-Expectation-Maximization-Algorithm]` | Roadmap |
| 27 | Tutorial 4 : Minmax Classifier | YT #27 | `[Planned: 27-Tutorial04-Minmax-Classifier]` | Roadmap |
| 28 | Tutorial 5 : Neyman Pearson Classifier | YT #28 | `[Planned: 28-Tutorial05-Neyman-Pearson-Classifier]` | Roadmap |
| 29 | Tutorial 6 : Example of NP Classifier, ROC Curve | YT #29 | `[Planned: 29-Tutorial06-NP-Classifier-ROC-Curve]` | Roadmap |
| 30 | Tutorial 7A : MLE for Gaussian Distribution | YT #30 | `[Planned: 30-Tutorial07A-MLE-Gaussian-Distribution]` | Roadmap |
| 31 | Tutorial 7B : MLE for Generalized Discrete Distribution | YT #31 | `[Planned: 31-Tutorial07B-MLE-Generalized-Discrete-Distribution]` | Roadmap |
| 32 | Lec 23 Convergence of EM | YT #32 | `[Planned: 32-Lec23-Convergence-of-EM]` | Roadmap |
| 33 | Lec 24 EM for GMMs | YT #33 | `[Planned: 33-Lec24-EM-for-GMMs]` | Roadmap |
| 34 | Lec 25 MAP Estimate | YT #34 | `[Planned: 34-Lec25-MAP-Estimate]` | Roadmap |
| 35 | Lec 26 Parzen Window | YT #35 | `[Planned: 35-Lec26-Parzen-Window]` | Roadmap |
| 36 | Lec 27 Nearest Neighbor Classifier | YT #36 | `[Planned: 36-Lec27-Nearest-Neighbor-Classifier]` | Roadmap |
| 37 | Tutorial 8 : Computation of EM for GMMs | YT #37 | `[Planned: 37-Tutorial08-Computation-EM-for-GMMs]` | Roadmap |
| 38 | Tutorial 9 : MAP Estimate | YT #38 | `[Planned: 38-Tutorial09-MAP-Estimate]` | Roadmap |
| 39 | Lec 28 Ordinary Least Squares (OLS) | YT #39 | `[Planned: 39-Lec28-Ordinary-Least-Squares-OLS]` | Roadmap |
| 40 | Lec 29 Generalized Least Squares (GLS) | YT #40 | `[Planned: 40-Lec29-Generalized-Least-Squares-GLS]` | Roadmap |
| 41 | Lec 30 Linear Models for Classification | YT #41 | `[Planned: 41-Lec30-Linear-Models-for-Classification]` | Roadmap |
| 42 | Lec 31 Bias - Variance Decomposition and Analysis | YT #42 | `[Planned: 42-Lec31-Bias-Variance-Decomposition-Analysis]` | Roadmap |
| 43 | Lec 32 Bias & Variance in Practice | YT #43 | `[Planned: 43-Lec32-Bias-Variance-in-Practice]` | Roadmap |
| 44 | Tutorial 10 Part A : Numerical Example on Bayes Classifier | YT #44 | `[Planned: 44-Tutorial10A-Numerical-Example-Bayes-Classifier]` | Roadmap |
| 45 | Tutorial 10 Part B : Numerical Example on MLE and MAP Estimate | YT #45 | `[Planned: 45-Tutorial10B-Numerical-Example-MLE-MAP]` | Roadmap |
| 46 | Lec 33 Regularization | YT #46 | `[Planned: 46-Lec33-Regularization]` | Roadmap |
| 47 | Lec 34 Regularized ERM and MAP Estimate | YT #47 | `[Planned: 47-Lec34-Regularized-ERM-MAP-Estimate]` | Roadmap |
| 48 | Lec 35 Stochastic Gradient Descent as a Regularizer | YT #48 | `[Planned: 48-Lec35-SGD-as-Regularizer]` | Roadmap |
| 49 | Lec 36 Max-Margin Classifier and SVM | YT #49 | `[Planned: 49-Lec36-Max-Margin-Classifier-SVM]` | Roadmap |
| 50 | Lec 37 SVM Formulation | YT #50 | `[Planned: 50-Lec37-SVM-Formulation]` | Roadmap |
| 51 | Lec 38 Dual Function in SVM | YT #51 | `[Planned: 51-Lec38-Dual-Function-in-SVM]` | Roadmap |
| 52 | Lec 39 SVM for Non-Linear Seperable Case | YT #52 | `[Planned: 52-Lec39-SVM-Nonlinear-Separable]` | Roadmap |
| 53 | Lec 40 SVM with Kernel Function | YT #53 | `[Planned: 53-Lec40-SVM-with-Kernel-Function]` | Roadmap |
| 54 | Lec 41 Neural Networks and Universal Approximation Theorem | YT #54 | [`54-Lec41-Neural-Networks-UAT/`](./Mathematical-foundation-ml/54-Lec41-Neural-Networks-UAT/) | **Completed** |
| 55 | Lec 42 ERM on Neural Networks and Error Backpropagation | YT #55 | [`55-Lec42-ERM-Neural-Networks-Backpropagation/`](./Mathematical-foundation-ml/55-Lec42-ERM-Neural-Networks-Backpropagation/) | **Completed** |
| 56 | Lec 43 Local Receptive Field and Parameter Sharing | YT #56 | [`56-Lec43-Local-Receptive-Field-Parameter-Sharing/`](./Mathematical-foundation-ml/56-Lec43-Local-Receptive-Field-Parameter-Sharing/) | **Completed** |
| 57 | Lec 44 Convolutional Neural Networks(CNNs) as Regularized MLP | YT #57 | [`57-Lec44-CNNs-as-Regularized-MLP/`](./Mathematical-foundation-ml/57-Lec44-CNNs-as-Regularized-MLP/) | **Completed** |
| 58 | Lec 45 Recurrent Neural Networks(RNNs) | YT #58 | [`58-Lec45-Recurrent-Neural-Networks-RNNs/`](./Mathematical-foundation-ml/58-Lec45-Recurrent-Neural-Networks-RNNs/) | **Completed** |
| 59 | Lec 46 Back Prapogation in RNNs and Vanishing Gradients Problem | YT #59 | [`59-Lec46-Backpropagation-in-RNNs-Vanishing-Gradients/`](./Mathematical-foundation-ml/59-Lec46-Backpropagation-in-RNNs-Vanishing-Gradients/) | **Completed** |
| 60 | Lec 47 LSTMs and GRUs | YT #60 | [`60-Lec47-LSTMs-and-GRUs/`](./Mathematical-foundation-ml/60-Lec47-LSTMs-and-GRUs/) | **Completed** |
| 61 | Tutorial 11 : Pytorch - Tensors and Data Loaders | YT #61 | [`61-Tutorial11-PyTorch-Tensors-DataLoaders/`](./Mathematical-foundation-ml/61-Tutorial11-PyTorch-Tensors-DataLoaders/) | **Completed** |
| 62 | Tutorial 12 : Pytorch - Building MLP and Auto Grad | YT #62 | [`62-Tutorial12-PyTorch-Building-MLP-Autograd/`](./Mathematical-foundation-ml/62-Tutorial12-PyTorch-Building-MLP-Autograd/) | **Completed** |
| 63 | Tutorial 13 : Pytorch - Training the Model | YT #63 | [`63-Tutorial13-PyTorch-Training-the-Model/`](./Mathematical-foundation-ml/63-Tutorial13-PyTorch-Training-the-Model/) | **Completed** |
| 64 | Lec 48 Attention Part1 | YT #64 | [`64-Lec48-Attention-Part1/`](./Mathematical-foundation-ml/64-Lec48-Attention-Part1/) | **Completed** |
| 65 | Lec 49 Attention Part2 | YT #65 | [`65-Lec49-Attention-Part2/`](./Mathematical-foundation-ml/65-Lec49-Attention-Part2/) | **Completed** |
| 66 | Lec 50 Multi-Head Attention and Transformer Architecture | YT #66 | [`66-Lec50-Multi-Head-Attention-Transformer/`](./Mathematical-foundation-ml/66-Lec50-Multi-Head-Attention-Transformer/) | **Completed** |
| 67 | Lec 51 Positional Embeddings | YT #67 | [`67-Lec51-Positional-Embeddings/`](./Mathematical-foundation-ml/67-Lec51-Positional-Embeddings/) | **Completed** |
| 68 | Lec 52 Transfer Learning and Knowledge Distilation | YT #68 | [`68-Lec52-Transfer-Learning-Knowledge-Distillation/`](./Mathematical-foundation-ml/68-Lec52-Transfer-Learning-Knowledge-Distillation/) | **Completed** |
| 69 | Lec 53 SGD, RMS Prop, ADAM : Optimizers | YT #69 | [`69-Lec53-SGD-RMSprop-Adam-Optimizers/`](./Mathematical-foundation-ml/69-Lec53-SGD-RMSprop-Adam-Optimizers/) | **Completed** |
| 70 | Tutorial 14 Part 1 : CNNs | YT #70 | [`70-Tutorial14A-CNNs/`](./Mathematical-foundation-ml/70-Tutorial14A-CNNs/) | **Completed** |
| 71 | Tutorial 14 Part 2 : Transfer Learning using CNNs | YT #71 | [`71-Tutorial14B-Transfer-Learning-CNNs/`](./Mathematical-foundation-ml/71-Tutorial14B-Transfer-Learning-CNNs/) | **Completed** |
| 72 | Tutorial 15 Part 1 : RNNs, LSTMs and GRUs | YT #72 | [`72-Tutorial15A-RNNs-LSTMs-GRUs/`](./Mathematical-foundation-ml/72-Tutorial15A-RNNs-LSTMs-GRUs/) | **Completed** |
| 73 | Tutorial 15 Part 2 : Deep RNNs, LSTMs and GRUs | YT #73 | [`73-Tutorial15B-Deep-RNNs-LSTMs-GRUs/`](./Mathematical-foundation-ml/73-Tutorial15B-Deep-RNNs-LSTMs-GRUs/) | **Completed** |
| 74 | Lec 54 Decision Trees and Impurity Measures | YT #74 | `[Planned: 74-Lec54-Decision-Trees-Impurity-Measures]` | Roadmap |
| 75 | Lec 55 Regression Trees | YT #75 | `[Planned: 75-Lec55-Regression-Trees]` | Roadmap |
| 76 | Lec 56 Ensemble Methods, Bagging and Boosting | YT #76 | `[Planned: 76-Lec56-Ensemble-Methods-Bagging-Boosting]` | Roadmap |
| 77 | Lec 57 Gradient Boosting Algorithm | YT #77 | `[Planned: 77-Lec57-Gradient-Boosting-Algorithm]` | Roadmap |
| 78 | Lec 58 Ada-Boosting | YT #78 | `[Planned: 78-Lec58-Ada-Boosting]` | Roadmap |
| 79 | Lec 59 Cross Validation | YT #79 | `[Planned: 79-Lec59-Cross-Validation]` | Roadmap |
| 80 | Lec 60 Un-Supervised Learning | YT #80 | `[Planned: 80-Lec60-Unsupervised-Learning]` | Roadmap |
| 81 | Lec 61 K-Means Clustering | YT #81 | `[Planned: 81-Lec61-K-Means-Clustering]` | Roadmap |
| 82 | Lec 62 PCA - Principal Component Analysis | YT #82 | `[Planned: 82-Lec62-PCA-Principal-Component-Analysis]` | Roadmap |
| 83 | Lec 63 NCE - Noise Contrastive Estimation | YT #83 | `[Planned: 83-Lec63-NCE-Noise-Contrastive-Estimation]` | Roadmap |
| 84 | Lec 64 NCE, Info-NCE, SimCLR, JEPA | YT #84 | `[Planned: 84-Lec64-NCE-InfoNCE-SimCLR-JEPA]` | Roadmap |
| 85 | Lec 65 Introduction to Generative Models | YT #85 | `[Planned: 85-Lec65-Introduction-to-Generative-Models]` | Roadmap |
| 86 | Lec 66 GAN - Generative Adversarial Networks | YT #86 | `[Planned: 86-Lec66-GAN-Generative-Adversarial-Networks]` | Roadmap |
| 87 | Lec 67 Variational Auto Encoders : VAEs | YT #87 | `[Planned: 87-Lec67-Variational-Auto-Encoders-VAEs]` | Roadmap |
| 88 | Lec 68 Introduction to Large Language Models : LLMs | YT #88 | `[Planned: 88-Lec68-Introduction-to-Large-Language-Models-LLMs]` | Roadmap |
| 89 | Lec 69 Introduction to Reinforcement Learning : RL | YT #89 | `[Planned: 89-Lec69-Introduction-to-Reinforcement-Learning-RL]` | Roadmap |

---

## Playlist catalog (learning order)

Links keep the official playlist ID (`PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu`) so YouTube navigates cleanly within the playlist context.

### Block 1: Intro and function approximation

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 1 | 1 | Mathematical Foundations of Machine Learning (Intro) | 3:33 | [watch](https://www.youtube.com/watch?v=vbs9WGWjS9U&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=1) | Course trailer: ML formulated from a rigorous probabilistic viewpoint; first course in the IISc two-course sequence. | Roadmap |
| 2 | 2 | Lec 01 Overview of Function Approximation | 47:50 | [watch](https://www.youtube.com/watch?v=G2h7nD_Stxg&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=2) | Function approximation (FA) as the core ML problem: from a finite table $\mathcal{D}$, estimate unknown continuous $f$. Why table lookup fails; physics vs probability. | [`02-Lec01`](./Mathematical-foundation-ml/02-Lec01-Overview-Function-Approximation/) |

### Block 2: Probability recap

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 3 | 3 | Lec 02 Recap of Probability Theory - 1, Part 1 | 32:13 | [watch](https://www.youtube.com/watch?v=YLx3hBqt28k&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=3) | Probability space triplet $(\Omega, \mathcal{F}, P)$; sample spaces, event $\sigma$-algebras, Kolmogorov axioms. Why continuous ML requires measure theory. | [`03-Lec02`](./Mathematical-foundation-ml/03-Lec02-Recap-Probability-Theory-Part1/) |
| 4 | 4 | Lec 03 Recap of Probability Theory - 1, Part 2 | 14:30 | [watch](https://www.youtube.com/watch?v=DaBw9qBpt2s&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=4) | Random variables $X: \Omega \to \mathbb{R}^d$ as measurable measurement maps. Sensor readings convert hidden physical outcomes into numerical vectors. | [`04-Lec03`](./Mathematical-foundation-ml/04-Lec03-Recap-Probability-Theory-Part2/) |
| 5 | 5 | Lec 04 Recap of Probability Theory - 1, Part 3 | 29:06 | [watch](https://www.youtube.com/watch?v=0R6Agp4tqSU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=5) | Pushforward measure, cumulative distribution function (CDF) $F_X(x)$, and the probability density function (PDF). The density trap ($p(x) > 1$ is valid). | [`05-Lec04`](./Mathematical-foundation-ml/05-Lec04-Recap-Probability-Theory-Part3/) |
| 6 | 6 | Lec 05 Recap of Probability Theory Part 2 | 21:43 | [watch](https://www.youtube.com/watch?v=R69wew8RrPo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=6) | Joint distributions, conditionals, and marginals. Equivalence of a $d$-dimensional random vector to $d$ scalar random variables with joint dependence. | [`06-Lec05`](./Mathematical-foundation-ml/06-Lec05-Recap-Probability-Theory-Part2/) |

### Block 3: Data as samples; estimation; density

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 7 | 7 | Lec 06 Understanding a Chest X-Ray as Sample from Distribution | 26:51 | [watch](https://www.youtube.com/watch?v=bdcvsSNAHIk&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=7) | High-dimensional images live in $\mathrm{range}(X) \subset \mathbb{R}^{d}$. Individual images are not probabilities; supervised learning is sampling from joint $p_{X,Y}$. | [`07-Lec06`](./Mathematical-foundation-ml/07-Lec06-XRay-Sample-From-Distribution/) |
| 8 | 8 | Lec 07 IID Assumption | 30:42 | [watch](https://www.youtube.com/watch?v=C83xmx80tMo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=8) | Independent and Identically Distributed (I.I.D.) condition across dataset points. Total dataset likelihood factorizes into a product of individual point densities. | [`08-Lec07`](./Mathematical-foundation-ml/08-Lec07-IID-Assumption/) |
| 9 | 9 | Lec 08 Distribution Estimation | 28:47 | [watch](https://www.youtube.com/watch?v=aYb8KG9JYsg&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=9) | Given finite dataset $\mathcal{D}$, estimate unknown $P$. Estimation targets: conditional $P(Y \mid X)$, prior $P(Y)$, joint $P(X,Y)$. Discriminative vs generative. | [`09-Lec08`](./Mathematical-foundation-ml/09-Lec08-Distribution-Estimation/) |
| 10 | 10 | Lec 09 Density Function | 8:06 | [watch](https://www.youtube.com/watch?v=_QrezNPmxDk&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=10) | Mathematical properties of probability density $p(x)$: point height is not a probability; height can exceed 1 (e.g. Uniform$[0, 0.5]$ has density 2.0). | [`10-Lec09`](./Mathematical-foundation-ml/10-Lec09-Density-Function/) |
| 11 | 11 | Lec 10 Challenge With ML | 35:31 | [watch](https://www.youtube.com/watch?v=767MLwniPKE&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=11) | Fundamental dilemma of ML: true $p$ is unknown. The universal ML recipe: pick family $p_\theta$, choose divergence $d(p, p_\theta)$, solve $\arg\min_\theta$. Model $\neq$ algorithm. | [`11-Lec10`](./Mathematical-foundation-ml/11-Lec10-Challenges-of-ML/) |

### Block 4: Python / probability tutorials

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 12 | 12 | Tutorial 1 : Introduction to Python Basics | 47:10 | [watch](https://www.youtube.com/watch?v=cF025BechXo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=12) | Core Python programming: data structures, list comprehensions, control flow, functions, foundational math scripting. | Roadmap |
| 13 | 13 | Tutorial 2 : Simple Problem solving in Probability Theory | 53:51 | [watch](https://www.youtube.com/watch?v=nGwjqvLHguA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=13) | Hand-worked and numerical probability problems grounding sample spaces, Bayes rule, conditional probability, and discrete distributions. | Roadmap |

### Block 5: Entropy, KL, min-KL

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 14 | 14 | Lec 11 Entropy | 17:56 | [watch](https://www.youtube.com/watch?v=P6wjLz4dRTs&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=14) | Information theory: surprisal $-\log P(x)$, Shannon entropy $H(p) = -\mathbb{E}[\log p(x)]$. Why we need statistical divergences to train machine learning models. | [`12-Lec11`](./Mathematical-foundation-ml/12-Lec11-Entropy/) |
| 15 | 15 | Lec 12 Kullback-Leibler (KL) Divergence | 16:49 | [watch](https://www.youtube.com/watch?v=ihkGbIdbbxc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=15) | Cross-entropy and relative entropy; $\mathrm{KL}(p \parallel q) = \mathrm{CE}(p, q) - H(p)$. Jensen's inequality proof that $\mathrm{KL} \ge 0$. Asymmetry and lack of triangle inequality. | [`13-Lec12`](./Mathematical-foundation-ml/13-Lec12-KL-Divergence/) |
| 16 | 16 | Lec 13 Minimization of KL Divergence | 24:52 | [watch](https://www.youtube.com/watch?v=Ij4p5hLbfo4&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=16) | Minimizing KL divergence drops the data entropy term $H(p)$; Law of Large Numbers (LLN) replaces expectation with empirical average. **Proof that MLE $\equiv$ min-KL estimator**. | [`14-Lec13`](./Mathematical-foundation-ml/14-Lec13-Minimization-of-KL/) |

### Block 6: Risk minimization and Bayes

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 17 | 17 | Lec 14 Example of ML Estimate | 20:24 | [watch](https://www.youtube.com/watch?v=mEpXOyLwbxA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=17) | Concrete Maximum Likelihood derivation following the min-KL theorem on an explicit parametric distribution family. | Roadmap |
| 18 | 18 | Lec 15 Risk Minimization Framework | 40:14 | [watch](https://www.youtube.com/watch?v=jXCqrFVGwoU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=18) | Decision theory: decision rule $g(X)$, loss function $\ell(y, g(x))$, expected risk $R(g) = \mathbb{E}[\ell(Y, g(X))]$. Empirical Risk Minimization (ERM). | Roadmap |
| 19 | 19 | Lec 16 Bayes Classifier | 35:37 | [watch](https://www.youtube.com/watch?v=-y3SSAIhD4Y&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=19) | Exact mathematical derivation of the Bayes optimal classifier for 0–1 loss: $g^*(x) = \arg\max_y P(Y=y \mid X=x)$. Bayes risk as the irreducible error lower bound. | Roadmap |
| 20 | 20 | Tutorail 3 : Risk Minimization Framework | 59:52 | [watch](https://www.youtube.com/watch?v=AQ3einJJrr0&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=20) | Comprehensive worked problem set calculating empirical risk, expected risk, and risk minimization boundaries under asymmetric loss matrices. | Roadmap |

### Block 7: MLE, latent variables, EM

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 21 | 21 | Lec 17 MLE for Gaussian Distribution | 26:07 | [watch](https://www.youtube.com/watch?v=tF-RrzUnnYA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=21) | Closed-form log-likelihood maximization for univariate and multivariate Gaussians $\mathcal{N}(\mu, \Sigma)$; sample mean and biased/unbiased covariance estimators. | Roadmap |
| 22 | 22 | Lec 18 MLE for Generalized Discrete Random Variable | 25:48 | [watch](https://www.youtube.com/watch?v=j7jbpicYdik&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=22) | Categorical and multinomial distributions; constrained log-likelihood optimization via Lagrange multipliers; empirical frequency counts as MLE. | Roadmap |
| 23 | 23 | Lec 19 Density Estimation for Mixed Distribution | 20:41 | [watch](https://www.youtube.com/watch?v=3UmgTSDgG5Q&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=23) | Handling distributions with both discrete atoms and continuous densities (e.g. rectified signals, zero-inflated sensor measurements); mixed measure formulation. | Roadmap |
| 24 | 24 | Lec 20 Latent Variable Models | 33:23 | [watch](https://www.youtube.com/watch?v=J9QNr4UrB2c&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=24) | Unobserved latent variables $Z$; marginal likelihood $p(x) = \int p(x \mid z) p(z) dz$. Why summing/integrating inside the logarithm breaks analytical tractability. | Roadmap |
| 25 | 25 | Lec 21 MLE for Latent Variable Models | 16:08 | [watch](https://www.youtube.com/watch?v=BMj-TWtK83A&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=25) | Analysis of the computational log-sum intractability for incomplete data; introducing variational lower bounds as an optimization surrogate. | Roadmap |
| 26 | 26 | Lec 22 Expectation Maximization Algorithm | 25:34 | [watch](https://www.youtube.com/watch?v=ejma0iH1pXE&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=26) | Full mathematical derivation of the EM algorithm using Jensen's inequality: E-step computing $Q(\theta \mid \theta^{(t)}) = \mathbb{E}_{Z \mid X, \theta^{(t)}}[\log p(X, Z \mid \theta)]$; M-step $\arg\max_\theta Q$. | Roadmap |

### Block 8: Classifier / MLE tutorials

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 27 | 27 | Tutorial 4 : Minmax Classifier | 26:33 | [watch](https://www.youtube.com/watch?v=ENpzs2ycXJE&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=27) | Minimax classification theory: finding optimal decision rules when class prior probabilities are completely unknown or adversarially selected. | Roadmap |
| 28 | 28 | Tutorial 5 : Neyman Pearson Classifier | 49:00 | [watch](https://www.youtube.com/watch?v=8esVIly2TZY&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=28) | Neyman-Pearson lemma: constrained risk minimization fixing false positive rate (Type-I error $\le \alpha$) while minimizing false negative rate (Type-II error); likelihood ratio thresholding. | Roadmap |
| 29 | 29 | Tutorial 6 : Example of NP Classifier, ROC Curve | 32:40 | [watch](https://www.youtube.com/watch?v=JwQEaTqyBDw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=29) | Concrete Neyman-Pearson derivations and construction of the Receiver Operating Characteristic (ROC) curve by sweeping decision thresholds. | Roadmap |
| 30 | 30 | Tutorial 7A : MLE for Gaussian Distribution | 40:17 | [watch](https://www.youtube.com/watch?v=XA3UiD8zEF8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=30) | Step-by-step code implementation and numerical verification of Gaussian MLE, confidence intervals, and variance estimators. | Roadmap |
| 31 | 31 | Tutorial 7B : MLE for Generalized Discrete Distribution | 23:08 | [watch](https://www.youtube.com/watch?v=o5697P6KZoc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=31) | Python simulation of categorical MLE with sparse data; Laplace smoothing / pseudocount regularization. | Roadmap |

### Block 9: EM, MAP, nonparametric

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 32 | 32 | Lec 23 Convergence of EM | 19:27 | [watch](https://www.youtube.com/watch?v=zHchxrSwOu4&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=32) | Rigorous convergence proof: monotonic non-decrease of incomplete log-likelihood $\ell(\theta^{(t+1)}) \ge \ell(\theta^{(t)})$; discussion of stationary points and local maxima. | Roadmap |
| 33 | 33 | Lec 24 EM for GMMs | 30:28 | [watch](https://www.youtube.com/watch?v=TSNsiglfduQ&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=33) | EM applied to Gaussian Mixture Models: soft responsibilities $\gamma_{ik}$ in E-step; weighted update formulas for mixing weights $\pi_k$, means $\mu_k$, and covariances $\Sigma_k$ in M-step. | Roadmap |
| 34 | 34 | Lec 25 MAP Estimate | 38:24 | [watch](https://www.youtube.com/watch?v=HH9Xjjj7UN4&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=34) | Maximum A Posteriori (MAP) estimation: treating parameters as random variables with prior $p(\theta)$; $\theta_{\text{MAP}} = \arg\max_\theta [\log p(\mathcal{D} \mid \theta) + \log p(\theta)]$. Priors as regularizers. | Roadmap |
| 35 | 35 | Lec 26 Parzen Window | 29:07 | [watch](https://www.youtube.com/watch?v=boCvzXvUVMI&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=35) | Nonparametric density estimation: Parzen window / Kernel Density Estimation (KDE); kernel functions, bandwidth $h$, asymptotic consistency, curse of dimensionality. | Roadmap |
| 36 | 36 | Lec 27 Nearest Neighbor Classifier | 17:32 | [watch](https://www.youtube.com/watch?v=YZ3Xa6dEMl8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=36) | $k$-Nearest Neighbors ($k$-NN) classification; Cover-Hart theorem proving 1-NN asymptotic error is bounded by at most twice the optimal Bayes error rate. | Roadmap |
| 37 | 37 | Tutorial 8 : Computation of EM for GMMs | 40:49 | [watch](https://www.youtube.com/watch?v=sOwgRt6uiA4&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=37) | Complete runnable Python/NumPy implementation of the EM algorithm for 2D GMMs with visualization of evolving covariance ellipses. | Roadmap |
| 38 | 38 | Tutorial 9 : MAP Estimate | 22:05 | [watch](https://www.youtube.com/watch?v=7y4V0GUoyaw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=38) | Worked numerical examples comparing MLE vs MAP with Conjugate Beta-Binomial and Gaussian-Gaussian priors. | Roadmap |

### Block 10: Linear models and bias–variance

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 39 | 39 | Lec 28 Ordinary Least Squares (OLS) | 24:48 | [watch](https://www.youtube.com/watch?v=s_DfCCobgnA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=39) | Linear regression as MLE under Gaussian noise; Normal equations $\theta^* = (X^T X)^{-1} X^T y$; projection matrix and geometric orthogonality of residuals. | Roadmap |
| 40 | 40 | Lec 29 Generalized Least Squares (GLS) | 25:19 | [watch](https://www.youtube.com/watch?v=UfiHgztGgu8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=40) | GLS regression under heteroscedastic or correlated noise $\mathrm{Cov}(\epsilon) = \Sigma$; Mahalanobis whitening transformation; Gauss-Markov theorem. | Roadmap |
| 41 | 41 | Lec 30 Linear Models for Classification | 33:29 | [watch](https://www.youtube.com/watch?v=EydAoMbslkc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=41) | Hyperplane decision surfaces $w^T x + b = 0$; least-squares classification limitations; logistic regression via logit link function; Cross-Entropy loss. | Roadmap |
| 42 | 42 | Lec 31 Bias - Variance Decomposition and Analysis | 46:37 | [watch](https://www.youtube.com/watch?v=0RCDPOz3YVc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=42) | Expected generalization error decomposition: $\mathbb{E}[(y - \hat{f})^2] = \text{Bias}^2 + \text{Variance} + \sigma_{\text{noise}}^2$; model capacity vs sample size trade-offs. | Roadmap |
| 43 | 43 | Lec 32 Bias & Variance in Practice | 37:14 | [watch](https://www.youtube.com/watch?v=E-kOTTO5hK8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=43) | Diagnosing high bias (underfitting) vs high variance (overfitting) using learning curves; practical architectural remedies. | Roadmap |
| 44 | 44 | Tutorial 10 Part A : Numerical Example on Bayes Classifier | 51:38 | [watch](https://www.youtube.com/watch?v=oTEPAiwv-00&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=44) | Hand-calculated and numerical simulation of Bayes decision boundaries for 2D Gaussian classes with identical vs distinct covariance matrices (LDA vs QDA). | Roadmap |
| 45 | 45 | Tutorial 10 Part B : Numerical Example on MLE and MAP Estimate | 30:58 | [watch](https://www.youtube.com/watch?v=ysjGmQW4HOo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=45) | Side-by-side numerical comparison showing how increasing dataset size $N \to \infty$ causes MAP to converge to MLE. | Roadmap |

### Block 11: Regularization and SVM

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 46 | 46 | Lec 33 Regularization | 28:27 | [watch](https://www.youtube.com/watch?v=7F8pknXk_-o&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=46) | Preventing overfitting by constraining parameter norms: L2 Ridge penalty ($\lambda \|w\|_2^2$) and L1 Lasso penalty ($\lambda \|w\|_1$); sparsity mechanics. | Roadmap |
| 47 | 47 | Lec 34 Regularized ERM and MAP Estimate | 26:57 | [watch](https://www.youtube.com/watch?v=rypIu-ZSYBo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=47) | Formal mathematical proof that L2 regularization is strictly equivalent to Gaussian prior MAP, and L1 is equivalent to Laplace prior MAP. | Roadmap |
| 48 | 48 | Lec 35 Stochastic Gradient Descent as a Regularizer | 22:02 | [watch](https://www.youtube.com/watch?v=vKTxP9FsR90&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=48) | Implicit regularization in mini-batch SGD: gradient covariance noise drives optimization toward flat minima, boosting generalization without explicit penalties. | Roadmap |
| 49 | 49 | Lec 36 Max-Margin Classifier and SVM | 43:45 | [watch](https://www.youtube.com/watch?v=joL7g6DSxPU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=49) | Support Vector Machines (SVM): geometric margin $2/\|w\|_2$; formulation as the widest-street convex optimization problem $\min \frac{1}{2}\|w\|_2^2$. | Roadmap |
| 50 | 50 | Lec 37 SVM Formulation | 33:42 | [watch](https://www.youtube.com/watch?v=aO3FTnrf2bQ&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=50) | Soft-margin SVM for non-linearly separable data: slack variables $\xi_i$, box constraints, trade-off parameter $C$, and hinge loss formulation. | Roadmap |
| 51 | 51 | Lec 38 Dual Function in SVM | 21:10 | [watch](https://www.youtube.com/watch?v=RXzcClx44Tw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=51) | Karush-Kuhn-Tucker (KKT) conditions; deriving the Wolfe Dual problem $\max_\alpha \sum \alpha_i - \frac{1}{2}\sum \alpha_i \alpha_j y_i y_j x_i^T x_j$. Support vectors. | Roadmap |
| 52 | 52 | Lec 39 SVM for Non-Linear Seperable Case | 34:09 | [watch](https://www.youtube.com/watch?v=jQ-3gT8Mytw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=52) | Non-linear mapping $\phi(x): \mathbb{R}^d \to \mathcal{H}$; transforming non-separable raw input spaces into linearly separable feature Hilbert spaces. | Roadmap |
| 53 | 53 | Lec 40 SVM with Kernel Function | 29:53 | [watch](https://www.youtube.com/watch?v=dDIutyWTPKA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=53) | The Kernel Trick: computing inner products $k(x, x') = \langle \phi(x), \phi(x') \rangle$ without explicitly computing $\phi$; Mercer's theorem; RBF and polynomial kernels. | Roadmap |

### Block 12: Neural nets, CNN, RNN

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 54 | 54 | Lec 41 Neural Networks and Universal Approximation Theorem | 30:43 | [watch](https://www.youtube.com/watch?v=npYHSFuqnzs&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=54) | Multi-Layer Perceptrons (MLP) as flexible compositional hypothesis classes; Universal Approximation Theorem (Cybenko, Hornik) for single-hidden-layer networks. | [`54-Lec41`](./Mathematical-foundation-ml/54-Lec41-Neural-Networks-UAT/) |
| 55 | 55 | Lec 42 ERM on Neural Networks and Error Backpropagation | 47:44 | [watch](https://www.youtube.com/watch?v=dONDRwX_83E&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=55) | Empirical Risk Minimization on computational graphs; multivariable calculus chain rule; reverse-mode automatic differentiation (Backpropagation). | [`55-Lec42`](./Mathematical-foundation-ml/55-Lec42-ERM-Neural-Networks-Backpropagation/) |
| 56 | 56 | Lec 43 Local Receptive Field and Parameter Sharing | 36:12 | [watch](https://www.youtube.com/watch?v=rm0VmbTQE8Y&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=56) | Exploiting spatial structure: local receptive fields, translation equivariance, and weight sharing as statistical priors / inductive bias. | [`56-Lec43`](./Mathematical-foundation-ml/56-Lec43-Local-Receptive-Field-Parameter-Sharing/) |
| 57 | 57 | Lec 44 Convolutional Neural Networks(CNNs) as Regularized MLP | 44:57 | [watch](https://www.youtube.com/watch?v=mSYTyrXCsA8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=57) | Proof that 2D convolution is an MLP with Toeplitz/circulant weight matrix constraints (infinite L2 penalty forcing connections to zero outside filter). | [`57-Lec44`](./Mathematical-foundation-ml/57-Lec44-CNNs-as-Regularized-MLP/) |
| 58 | 58 | Lec 45 Recurrent Neural Networks(RNNs) | 38:05 | [watch](https://www.youtube.com/watch?v=E2LLi7AB9lQ&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=58) | Sequence modeling via recurrent transition function $h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b)$; parameter sharing across temporal timesteps. | [`58-Lec45`](./Mathematical-foundation-ml/58-Lec45-Recurrent-Neural-Networks-RNNs/) |
| 59 | 59 | Lec 46 Back Prapogation in RNNs and Vanishing Gradients Problem | 29:28 | [watch](https://www.youtube.com/watch?v=TVkaROL2FLw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=59) | Backpropagation Through Time (BPTT); product of Jacobians $\prod_{k=t}^T \frac{\partial h_k}{\partial h_{k-1}}$; mathematical proof of exponential vanishing and exploding gradients. | [`59-Lec46`](./Mathematical-foundation-ml/59-Lec46-Backpropagation-in-RNNs-Vanishing-Gradients/) |
| 60 | 60 | Lec 47 LSTMs and GRUs | 22:19 | [watch](https://www.youtube.com/watch?v=Pkuwu4EMRj8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=60) | Gated architectures: Long Short-Term Memory (LSTM) cell state $c_t$ with additive gradient highway; Gated Recurrent Units (GRU); vanishing gradient mitigation. | [`60-Lec47`](./Mathematical-foundation-ml/60-Lec47-LSTMs-and-GRUs/) |

### Block 13: PyTorch tutorials

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 61 | 61 | Tutorial 11 : Pytorch - Tensors and Data Loaders | 45:19 | [watch](https://www.youtube.com/watch?v=vm9FYVtM5ZA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=61) | Practical PyTorch: tensor manipulation, memory layout, broadcasting, GPU device transfers, `Dataset` and `DataLoader` batching pipelines. | [`61-Tutorial11`](./Mathematical-foundation-ml/61-Tutorial11-PyTorch-Tensors-DataLoaders/) |
| 62 | 62 | Tutorial 12 : Pytorch - Building MLP and Auto Grad | 42:47 | [watch](https://www.youtube.com/watch?v=SEEQ2A2WN9g&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=62) | Constructing modular MLPs via `nn.Module`; autograd mechanics (`requires_grad`, `.backward()`, `.grad`); inspect computation graphs. | [`62-Tutorial12`](./Mathematical-foundation-ml/62-Tutorial12-PyTorch-Building-MLP-Autograd/) |
| 63 | 63 | Tutorial 13 : Pytorch - Training the Model | 33:37 | [watch](https://www.youtube.com/watch?v=5PruFG5g1C4&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=63) | End-to-end training loop: forward pass, loss computation, zeroing gradients, optimizer step, learning rate scheduling, validation checkpoints. | [`63-Tutorial13`](./Mathematical-foundation-ml/63-Tutorial13-PyTorch-Training-the-Model/) |

### Block 14: Attention, Transformers, transfer, optimizers

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 64 | 64 | Lec 48 Attention Part1 | 23:31 | [watch](https://www.youtube.com/watch?v=00DOOSYFyJA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=64) | Beyond recurrence: content-based addressing; Queries ($Q$), Keys ($K$), and Values ($V$); similarity scores as dynamic routing weights. | [`64-Lec48`](./Mathematical-foundation-ml/64-Lec48-Attention-Part1/) |
| 65 | 65 | Lec 49 Attention Part2 | 33:56 | [watch](https://www.youtube.com/watch?v=TGzZ8jCX4y0&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=65) | Scaled Dot-Product Attention $\mathrm{softmax}(QK^T / \sqrt{d_k})V$; scale factor derivation preventing gradient vanishing; causal masking for autoregression. | [`65-Lec49`](./Mathematical-foundation-ml/65-Lec49-Attention-Part2/) |
| 66 | 66 | Lec 50 Multi-Head Attention and Transformer Architecture | 27:25 | [watch](https://www.youtube.com/watch?v=NQyNZ6Plxzs&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=66) | Multi-Head Attention (MHA) projecting into multiple representation subspaces; full Transformer block: LayerNorm, Residual Connections, Feed-Forward Networks. | [`66-Lec50`](./Mathematical-foundation-ml/66-Lec50-Multi-Head-Attention-Transformer/) |
| 67 | 67 | Lec 51 Positional Embeddings | 13:29 | [watch](https://www.youtube.com/watch?v=xbkJIeLoGyw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=67) | Permutation equivariance of self-attention; injecting sequence order via sinusoidal positional embeddings and learned positional encodings. | [`67-Lec51`](./Mathematical-foundation-ml/67-Lec51-Positional-Embeddings/) |
| 68 | 68 | Lec 52 Transfer Learning and Knowledge Distilation | 37:35 | [watch](https://www.youtube.com/watch?v=5f2_TocU_CU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=68) | Reusing pretrained representations; head swapping vs fine-tuning; Knowledge Distillation (Hinton): matching softened student logits with teacher targets. | [`68-Lec52`](./Mathematical-foundation-ml/68-Lec52-Transfer-Learning-Knowledge-Distillation/) |
| 69 | 69 | Lec 53 SGD, RMS Prop, ADAM : Optimizers | 19:28 | [watch](https://www.youtube.com/watch?v=N6F1J-wcE_E&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=69) | First-order optimization: SGD with momentum; RMSprop dividing by running root-mean-square of gradients; Adam combining momentum and adaptive learning rates with bias correction. | [`69-Lec53`](./Mathematical-foundation-ml/69-Lec53-SGD-RMSprop-Adam-Optimizers/) |

### Block 15: CNN / RNN tutorials

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 70 | 70 | Tutorial 14 Part 1 : CNNs | 33:52 | [watch](https://www.youtube.com/watch?v=0wd-LIzzfM0&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=70) | PyTorch implementation of convolutional architectures: `nn.Conv2d`, `nn.MaxPool2d`, flattening, feature map dimension tracking. | [`70-Tutorial14A-CNNs/`](./Mathematical-foundation-ml/70-Tutorial14A-CNNs/) | **Completed** |
| 71 | 71 | Tutorial 14 Part 2 : Transfer Learning using CNNs | 28:50 | [watch](https://www.youtube.com/watch?v=vocN0cxAT7I&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=71) | Loading pretrained torchvision vision models (ResNet), freezing feature extraction weights, fine-tuning classification heads on custom data. | [`71-Tutorial14B-Transfer-Learning-CNNs/`](./Mathematical-foundation-ml/71-Tutorial14B-Transfer-Learning-CNNs/) | **Completed** |
| 72 | 72 | Tutorial 15 Part 1 : RNNs, LSTMs and GRUs | 34:07 | [watch](https://www.youtube.com/watch?v=zM5-TlrmKg8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=72) | Hands-on PyTorch coding of `nn.RNN`, `nn.LSTM`, and `nn.GRU` layers for sequence classification and sentiment analysis. | [`72-Tutorial15A-RNNs-LSTMs-GRUs/`](./Mathematical-foundation-ml/72-Tutorial15A-RNNs-LSTMs-GRUs/) | **Completed** |
| 73 | 73 | Tutorial 15 Part 2 : Deep RNNs, LSTMs and GRUs | 16:53 | [watch](https://www.youtube.com/watch?v=nJivfX9VY7Y&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=73) | Stacking multi-layer recurrent nets; bidirectional LSTMs (`bidirectional=True`); sequence decoding mechanics; Many-to-One vs Many-to-Many. | [`73-Tutorial15B-Deep-RNNs-LSTMs-GRUs/`](./Mathematical-foundation-ml/73-Tutorial15B-Deep-RNNs-LSTMs-GRUs/) | **Completed** |

### Block 16: Trees, ensembles, cross-validation

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 74 | 74 | Lec 54 Decision Trees and Impurity Measures | 40:23 | [watch](https://www.youtube.com/watch?v=b2ScFHIhnB0&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=74) | Non-parametric decision trees: greedy recursive binary splitting; impurity metrics: Gini impurity $1 - \sum p_i^2$, Entropy $-\sum p_i \log p_i$, misclassification error. | Roadmap |
| 75 | 75 | Lec 55 Regression Trees | 34:09 | [watch](https://www.youtube.com/watch?v=1GZDVRskXGE&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=75) | Regression trees for continuous targets $y \in \mathbb{R}$: splitting criteria minimizing sum of squared errors (SSE); terminal leaf sample mean assignments. | Roadmap |
| 76 | 76 | Lec 56 Ensemble Methods, Bagging and Boosting | 35:55 | [watch](https://www.youtube.com/watch?v=2wyxSgVolJg&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=76) | Ensemble theory: bootstrap aggregating (Bagging) reducing variance on deep trees; sequential Boosting reducing bias on shallow stumps. | Roadmap |
| 77 | 77 | Lec 57 Gradient Boosting Algorithm | 30:24 | [watch](https://www.youtube.com/watch?v=_YKxmyP5PWU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=77) | Gradient Boosting as gradient descent in functional space: fitting successive trees to the negative gradient (pseudo-residuals) of arbitrary differentiable loss functions. | Roadmap |
| 78 | 78 | Lec 58 Ada-Boosting | 40:08 | [watch](https://www.youtube.com/watch?v=27cAa8L0JQo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=78) | AdaBoost algorithm: reweighting misclassified samples; proving AdaBoost is stagewise additive modeling optimizing exponential loss $L(y, f) = e^{-y f(x)}$. | Roadmap |
| 79 | 79 | Lec 59 Cross Validation | 12:26 | [watch](https://www.youtube.com/watch?v=R36BJ52Lr1A&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=79) | Model selection under finite sample constraints: $K$-fold cross-validation, stratified splits, unbiased risk estimation without data snooping. | Roadmap |

### Block 17: Unsupervised and contrastive

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 80 | 80 | Lec 60 Un-Supervised Learning | 20:44 | [watch](https://www.youtube.com/watch?v=5sg3d8ObcDc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=80) | Learning structure without labels: discovering underlying manifold geometry, clusters, latent variables, and self-supervised representations from $p_X$. | Roadmap |
| 81 | 81 | Lec 61 K-Means Clustering | 36:11 | [watch](https://www.youtube.com/watch?v=oFr4JX75MCc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=81) | $K$-Means clustering algorithm: alternating assignment and centroid update steps; non-convex coordinate descent; hard-assignment limit of Gaussian Mixture Models. | Roadmap |
| 82 | 82 | Lec 62 PCA - Principal Component Analysis | 45:42 | [watch](https://www.youtube.com/watch?v=BLYw-vf9q6U&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=82) | Principal Component Analysis: variance maximization and reconstruction error minimization; spectral decomposition of empirical covariance matrix $X^T X = U \Lambda U^T$. | Roadmap |
| 83 | 83 | Lec 63 NCE - Noise Contrastive Estimation | 41:02 | [watch](https://www.youtube.com/watch?v=DHBl3E0ndRA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=83) | Density estimation without computing the partition function: training a binary classifier to discriminate true data samples from noise distribution draws. | Roadmap |
| 84 | 84 | Lec 64 NCE, Info-NCE, SimCLR, JEPA | 35:36 | [watch](https://www.youtube.com/watch?v=vRX_5wwImec&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=84) | Contrastive self-supervised representations: InfoNCE mutual information lower bound; SimCLR data augmentations; LeCun's Joint Embedding Predictive Architecture (JEPA). | Roadmap |

### Block 18: Bridge to generative AI

These final 5 lectures serve as the **pedagogical bridge** and trailer for the sequel course, [Mathematical Foundations of Generative AI](./Mathematical-Foundation-for-GenerativeAI/NOTES.md).

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 85 | 85 | Lec 65 Introduction to Generative Models | 37:10 | [watch](https://www.youtube.com/watch?v=hgZ3HOMkrx0&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=85) | Shifting from classification $P(Y \mid X)$ to density estimation **and sampling** $x \sim p_X$. Handoff from classical ML risk minimization to generative modeling. | Roadmap |
| 86 | 86 | Lec 66 GAN - Generative Adversarial Networks | 41:51 | [watch](https://www.youtube.com/watch?v=nd3laZj5Cdg&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=86) | Implicit density modeling: generator $G(z)$ mapping noise to data; discriminator $D(x)$ estimating density ratio; two-player zero-sum game minimizing Jensen-Shannon divergence. | Roadmap |
| 87 | 87 | Lec 67 Variational Auto Encoders : VAEs | 42:06 | [watch](https://www.youtube.com/watch?v=8LAKmtw0WvQ&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=87) | Latent variable generative modeling: encoder $q_\phi(z \mid x)$, decoder $p_\theta(x \mid z)$; optimizing the Evidence Lower Bound (ELBO) via reparameterization trick $z = \mu + \sigma \odot \epsilon$. | Roadmap |
| 88 | 88 | Lec 68 Introduction to Large Language Models : LLMs | 25:22 | [watch](https://www.youtube.com/watch?v=41tyjrCeENA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=88) | Autoregressive generative sequence models: factorizing joint probability $p(x) = \prod_{i=1}^T p(x_i \mid x_{<i})$; next-token prediction on Decoder-only Transformer backbones. | Roadmap |
| 89 | 89 | Lec 69 Introduction to Reinforcement Learning : RL | 31:13 | [watch](https://www.youtube.com/watch?v=5AcmvmzE2zU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=89) | Sequential decision making under uncertainty: Markov Decision Processes (MDP), policy $\pi(a \mid s)$, expected returns, value functions; foundational vocabulary for RLHF/DPO. | Roadmap |

---

## YouTube playlist order (chronological 1 to 89)

The official YouTube playlist [PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu](https://www.youtube.com/playlist?list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu) is organized in chronological lecture order:

| YT # | Video Title | Duration | Video ID |
|:----:|:------------|:--------:|:--------:|
| 1 | Mathematical Foundations of Machine Learning (Intro) | 3:33 | `vbs9WGWjS9U` |
| 2 | Lec 01 Overview of Function Approximation | 47:50 | `G2h7nD_Stxg` |
| 3 | Lec 02 Recap of Probability Theory - 1, Part 1 | 32:13 | `YLx3hBqt28k` |
| 4 | Lec 03 Recap of Probability Theory - 1, Part 2 | 14:30 | `DaBw9qBpt2s` |
| 5 | Lec 04 Recap of Probability Theory - 1, Part 3 | 29:06 | `0R6Agp4tqSU` |
| 6 | Lec 05 Recap of Probability Theory Part 2 | 21:43 | `R69wew8RrPo` |
| 7 | Lec 06 Understanding a Chest X-Ray as Sample from Distribution | 26:51 | `bdcvsSNAHIk` |
| 8 | Lec 07 IID Assumption | 30:42 | `C83xmx80tMo` |
| 9 | Lec 08 Distribution Estimation | 28:47 | `aYb8KG9JYsg` |
| 10 | Lec 09 Density Function | 8:06 | `_QrezNPmxDk` |
| 11 | Lec 10 Challenge With ML | 35:31 | `767MLwniPKE` |
| 12 | Tutorial 1 : Introduction to Python Basics | 47:10 | `cF025BechXo` |
| 13 | Tutorial 2 : Simple Problem solving in Probability Theory | 53:51 | `nGwjqvLHguA` |
| 14 | Lec 11 Entropy | 17:56 | `P6wjLz4dRTs` |
| 15 | Lec 12 Kullback-Leibler (KL) Divergence | 16:49 | `ihkGbIdbbxc` |
| 16 | Lec 13 Minimization of KL Divergence | 24:52 | `Ij4p5hLbfo4` |
| 17 | Lec 14 Example of ML Estimate | 20:24 | `mEpXOyLwbxA` |
| 18 | Lec 15 Risk Minimization Framework | 40:14 | `jXCqrFVGwoU` |
| 19 | Lec 16 Bayes Classifier | 35:37 | `-y3SSAIhD4Y` |
| 20 | Tutorail 3 : Risk Minimization Framework | 59:52 | `AQ3einJJrr0` |
| 21 | Lec 17 MLE for Gaussian Distribution | 26:07 | `tF-RrzUnnYA` |
| 22 | Lec 18 MLE for Generalized Discrete Random Variable | 25:48 | `j7jbpicYdik` |
| 23 | Lec 19 Density Estimation for Mixed Distribution | 20:41 | `3UmgTSDgG5Q` |
| 24 | Lec 20 Latent Variable Models | 33:23 | `J9QNr4UrB2c` |
| 25 | Lec 21 MLE for Latent Variable Models | 16:08 | `BMj-TWtK83A` |
| 26 | Lec 22 Expectation Maximization Algorithm | 25:34 | `ejma0iH1pXE` |
| 27 | Tutorial 4 : Minmax Classifier | 26:33 | `ENpzs2ycXJE` |
| 28 | Tutorial 5 : Neyman Pearson Classifier | 49:00 | `8esVIly2TZY` |
| 29 | Tutorial 6 : Example of NP Classifier, ROC Curve | 32:40 | `JwQEaTqyBDw` |
| 30 | Tutorial 7A : MLE for Gaussian Distribution | 40:17 | `XA3UiD8zEF8` |
| 31 | Tutorial 7B : MLE for Generalized Discrete Distribution | 23:08 | `o5697P6KZoc` |
| 32 | Lec 23 Convergence of EM | 19:27 | `zHchxrSwOu4` |
| 33 | Lec 24 EM for GMMs | 30:28 | `TSNsiglfduQ` |
| 34 | Lec 25 MAP Estimate | 38:24 | `HH9Xjjj7UN4` |
| 35 | Lec 26 Parzen Window | 29:07 | `boCvzXvUVMI` |
| 36 | Lec 27 Nearest Neighbor Classifier | 17:32 | `YZ3Xa6dEMl8` |
| 37 | Tutorial 8 : Computation of EM for GMMs | 40:49 | `sOwgRt6uiA4` |
| 38 | Tutorial 9 : MAP Estimate | 22:05 | `7y4V0GUoyaw` |
| 39 | Lec 28 Ordinary Least Squares (OLS) | 24:48 | `s_DfCCobgnA` |
| 40 | Lec 29 Generalized Least Squares (GLS) | 25:19 | `UfiHgztGgu8` |
| 41 | Lec 30 Linear Models for Classification | 33:29 | `EydAoMbslkc` |
| 42 | Lec 31 Bias - Variance Decomposition and Analysis | 46:37 | `0RCDPOz3YVc` |
| 43 | Lec 32 Bias & Variance in Practice | 37:14 | `E-kOTTO5hK8` |
| 44 | Tutorial 10 Part A : Numerical Example on Bayes Classifier | 51:38 | `oTEPAiwv-00` |
| 45 | Tutorial 10 Part B : Numerical Example on MLE and MAP Estimate | 30:58 | `ysjGmQW4HOo` |
| 46 | Lec 33 Regularization | 28:27 | `7F8pknXk_-o` |
| 47 | Lec 34 Regularized ERM and MAP Estimate | 26:57 | `rypIu-ZSYBo` |
| 48 | Lec 35 Stochastic Gradient Descent as a Regularizer | 22:02 | `vKTxP9FsR90` |
| 49 | Lec 36 Max-Margin Classifier and SVM | 43:45 | `joL7g6DSxPU` |
| 50 | Lec 37 SVM Formulation | 33:42 | `aO3FTnrf2bQ` |
| 51 | Lec 38 Dual Function in SVM | 21:10 | `RXzcClx44Tw` |
| 52 | Lec 39 SVM for Non-Linear Seperable Case | 34:09 | `jQ-3gT8Mytw` |
| 53 | Lec 40 SVM with Kernel Function | 29:53 | `dDIutyWTPKA` |
| 54 | Lec 41 Neural Networks and Universal Approximation Theorem | 30:43 | `npYHSFuqnzs` |
| 55 | Lec 42 ERM on Neural Networks and Error Backpropagation | 47:44 | `dDONDRwX_83E` |
| 56 | Lec 43 Local Receptive Field and Parameter Sharing | 36:12 | `rm0VmbTQE8Y` |
| 57 | Lec 44 Convolutional Neural Networks(CNNs) as Regularized MLP | 44:57 | `mSYTyrXCsA8` |
| 58 | Lec 45 Recurrent Neural Networks(RNNs) | 38:05 | `E2LLi7AB9lQ` |
| 59 | Lec 46 Back Prapogation in RNNs and Vanishing Gradients Problem | 29:28 | `TVkaROL2FLw` |
| 60 | Lec 47 LSTMs and GRUs | 22:19 | `Pkuwu4EMRj8` |
| 61 | Tutorial 11 : Pytorch - Tensors and Data Loaders | 45:19 | `vm9FYVtM5ZA` |
| 62 | Tutorial 12 : Pytorch - Building MLP and Auto Grad | 42:47 | `SEEQ2A2WN9g` |
| 63 | Tutorial 13 : Pytorch - Training the Model | 33:37 | `5PruFG5g1C4` |
| 64 | Lec 48 Attention Part1 | 23:31 | `00DOOSYFyJA` |
| 65 | Lec 49 Attention Part2 | 33:56 | `TGzZ8jCX4y0` |
| 66 | Lec 50 Multi-Head Attention and Transformer Architecture | 27:25 | `NQyNZ6Plxzs` |
| 67 | Lec 51 Positional Embeddings | 13:29 | `xbkJIeLoGyw` |
| 68 | Lec 52 Transfer Learning and Knowledge Distilation | 37:35 | `5f2_TocU_CU` |
| 69 | Lec 53 SGD, RMS Prop, ADAM : Optimizers | 19:28 | `N6F1J-wcE_E` |
| 70 | Tutorial 14 Part 1 : CNNs | 33:52 | `0wd-LIzzfM0` |
| 71 | Tutorial 14 Part 2 : Transfer Learning using CNNs | 28:50 | `vocN0cxAT7I` |
| 72 | Tutorial 15 Part 1 : RNNs, LSTMs and GRUs | 34:07 | `zM5-TlrmKg8` |
| 73 | Tutorial 15 Part 2 : Deep RNNs, LSTMs and GRUs | 16:53 | `nJivfX9VY7Y` |
| 74 | Lec 54 Decision Trees and Impurity Measures | 40:23 | `b2ScFHIhnB0` |
| 75 | Lec 55 Regression Trees | 34:09 | `1GZDVRskXGE` |
| 76 | Lec 56 Ensemble Methods, Bagging and Boosting | 35:55 | `2wyxSgVolJg` |
| 77 | Lec 57 Gradient Boosting Algorithm | 30:24 | `_YKxmyP5PWU` |
| 78 | Lec 58 Ada-Boosting | 40:08 | `27cAa8L0JQo` |
| 79 | Lec 59 Cross Validation | 12:26 | `R36BJ52Lr1A` |
| 80 | Lec 60 Un-Supervised Learning | 20:44 | `5sg3d8ObcDc` |
| 81 | Lec 61 K-Means Clustering | 36:11 | `oFr4JX75MCc` |
| 82 | Lec 62 PCA - Principal Component Analysis | 45:42 | `BLYw-vf9q6U` |
| 83 | Lec 63 NCE - Noise Contrastive Estimation | 41:02 | `DHBl3E0ndRA` |
| 84 | Lec 64 NCE, Info-NCE, SimCLR, JEPA | 35:36 | `vRX_5wwImec` |
| 85 | Lec 65 Introduction to Generative Models | 37:10 | `hgZ3HOMkrx0` |
| 86 | Lec 66 GAN - Generative Adversarial Networks | 41:51 | `nd3laZj5Cdg` |
| 87 | Lec 67 Variational Auto Encoders : VAEs | 42:06 | `8LAKmtw0WvQ` |
| 88 | Lec 68 Introduction to Large Language Models : LLMs | 25:22 | `41tyjrCeENA` |
| 89 | Lec 69 Introduction to Reinforcement Learning : RL | 31:13 | `5AcmvmzE2zU` |

---

## Compact title → URL list

In pedagogical sequence (learning order). Every link points to the exact YouTube video in playlist context:

1. Mathematical Foundations of Machine Learning (Intro) → https://www.youtube.com/watch?v=vbs9WGWjS9U&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=1
2. Lec 01 Overview of Function Approximation → https://www.youtube.com/watch?v=G2h7nD_Stxg&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=2
3. Lec 02 Recap of Probability Theory - 1, Part 1 → https://www.youtube.com/watch?v=YLx3hBqt28k&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=3
4. Lec 03 Recap of Probability Theory - 1, Part 2 → https://www.youtube.com/watch?v=DaBw9qBpt2s&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=4
5. Lec 04 Recap of Probability Theory - 1, Part 3 → https://www.youtube.com/watch?v=0R6Agp4tqSU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=5
6. Lec 05 Recap of Probability Theory Part 2 → https://www.youtube.com/watch?v=R69wew8RrPo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=6
7. Lec 06 Understanding a Chest X-Ray as Sample from Distribution → https://www.youtube.com/watch?v=bdcvsSNAHIk&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=7
8. Lec 07 IID Assumption → https://www.youtube.com/watch?v=C83xmx80tMo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=8
9. Lec 08 Distribution Estimation → https://www.youtube.com/watch?v=aYb8KG9JYsg&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=9
10. Lec 09 Density Function → https://www.youtube.com/watch?v=_QrezNPmxDk&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=10
11. Lec 10 Challenge With ML → https://www.youtube.com/watch?v=767MLwniPKE&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=11
12. Tutorial 1 : Introduction to Python Basics → https://www.youtube.com/watch?v=cF025BechXo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=12
13. Tutorial 2 : Simple Problem solving in Probability Theory → https://www.youtube.com/watch?v=nGwjqvLHguA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=13
14. Lec 11 Entropy → https://www.youtube.com/watch?v=P6wjLz4dRTs&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=14
15. Lec 12 Kullback-Leibler (KL) Divergence → https://www.youtube.com/watch?v=ihkGbIdbbxc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=15
16. Lec 13 Minimization of KL Divergence → https://www.youtube.com/watch?v=Ij4p5hLbfo4&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=16
17. Lec 14 Example of ML Estimate → https://www.youtube.com/watch?v=mEpXOyLwbxA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=17
18. Lec 15 Risk Minimization Framework → https://www.youtube.com/watch?v=jXCqrFVGwoU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=18
19. Lec 16 Bayes Classifier → https://www.youtube.com/watch?v=-y3SSAIhD4Y&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=19
20. Tutorail 3 : Risk Minimization Framework → https://www.youtube.com/watch?v=AQ3einJJrr0&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=20
21. Lec 17 MLE for Gaussian Distribution → https://www.youtube.com/watch?v=tF-RrzUnnYA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=21
22. Lec 18 MLE for Generalized Discrete Random Variable → https://www.youtube.com/watch?v=j7jbpicYdik&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=22
23. Lec 19 Density Estimation for Mixed Distribution → https://www.youtube.com/watch?v=3UmgTSDgG5Q&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=23
24. Lec 20 Latent Variable Models → https://www.youtube.com/watch?v=J9QNr4UrB2c&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=24
25. Lec 21 MLE for Latent Variable Models → https://www.youtube.com/watch?v=BMj-TWtK83A&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=25
26. Lec 22 Expectation Maximization Algorithm → https://www.youtube.com/watch?v=ejma0iH1pXE&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=26
27. Tutorial 4 : Minmax Classifier → https://www.youtube.com/watch?v=ENpzs2ycXJE&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=27
28. Tutorial 5 : Neyman Pearson Classifier → https://www.youtube.com/watch?v=8esVIly2TZY&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=28
29. Tutorial 6 : Example of NP Classifier, ROC Curve → https://www.youtube.com/watch?v=JwQEaTqyBDw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=29
30. Tutorial 7A : MLE for Gaussian Distribution → https://www.youtube.com/watch?v=XA3UiD8zEF8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=30
31. Tutorial 7B : MLE for Generalized Discrete Distribution → https://www.youtube.com/watch?v=o5697P6KZoc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=31
32. Lec 23 Convergence of EM → https://www.youtube.com/watch?v=zHchxrSwOu4&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=32
33. Lec 24 EM for GMMs → https://www.youtube.com/watch?v=TSNsiglfduQ&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=33
34. Lec 25 MAP Estimate → https://www.youtube.com/watch?v=HH9Xjjj7UN4&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=34
35. Lec 26 Parzen Window → https://www.youtube.com/watch?v=boCvzXvUVMI&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=35
36. Lec 27 Nearest Neighbor Classifier → https://www.youtube.com/watch?v=YZ3Xa6dEMl8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=36
37. Tutorial 8 : Computation of EM for GMMs → https://www.youtube.com/watch?v=sOwgRt6uiA4&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=37
38. Tutorial 9 : MAP Estimate → https://www.youtube.com/watch?v=7y4V0GUoyaw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=38
39. Lec 28 Ordinary Least Squares (OLS) → https://www.youtube.com/watch?v=s_DfCCobgnA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=39
40. Lec 29 Generalized Least Squares (GLS) → https://www.youtube.com/watch?v=UfiHgztGgu8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=40
41. Lec 30 Linear Models for Classification → https://www.youtube.com/watch?v=EydAoMbslkc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=41
42. Lec 31 Bias - Variance Decomposition and Analysis → https://www.youtube.com/watch?v=0RCDPOz3YVc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=42
43. Lec 32 Bias & Variance in Practice → https://www.youtube.com/watch?v=E-kOTTO5hK8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=43
44. Tutorial 10 Part A : Numerical Example on Bayes Classifier → https://www.youtube.com/watch?v=oTEPAiwv-00&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=44
45. Tutorial 10 Part B : Numerical Example on MLE and MAP Estimate → https://www.youtube.com/watch?v=ysjGmQW4HOo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=45
46. Lec 33 Regularization → https://www.youtube.com/watch?v=7F8pknXk_-o&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=46
47. Lec 34 Regularized ERM and MAP Estimate → https://www.youtube.com/watch?v=rypIu-ZSYBo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=47
48. Lec 35 Stochastic Gradient Descent as a Regularizer → https://www.youtube.com/watch?v=vKTxP9FsR90&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=48
49. Lec 36 Max-Margin Classifier and SVM → https://www.youtube.com/watch?v=joL7g6DSxPU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=49
50. Lec 37 SVM Formulation → https://www.youtube.com/watch?v=aO3FTnrf2bQ&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=50
51. Lec 38 Dual Function in SVM → https://www.youtube.com/watch?v=RXzcClx44Tw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=51
52. Lec 39 SVM for Non-Linear Seperable Case → https://www.youtube.com/watch?v=jQ-3gT8Mytw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=52
53. Lec 40 SVM with Kernel Function → https://www.youtube.com/watch?dDIutyWTPKA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=53
54. Lec 41 Neural Networks and Universal Approximation Theorem → https://www.youtube.com/watch?v=npYHSFuqnzs&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=54
55. Lec 42 ERM on Neural Networks and Error Backpropagation → https://www.youtube.com/watch?v=dONDRwX_83E&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=55
56. Lec 43 Local Receptive Field and Parameter Sharing → https://www.youtube.com/watch?v=rm0VmbTQE8Y&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=56
57. Lec 44 Convolutional Neural Networks(CNNs) as Regularized MLP → https://www.youtube.com/watch?v=mSYTyrXCsA8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=57
58. Lec 45 Recurrent Neural Networks(RNNs) → https://www.youtube.com/watch?v=E2LLi7AB9lQ&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=58
59. Lec 46 Back Prapogation in RNNs and Vanishing Gradients Problem → https://www.youtube.com/watch?v=TVkaROL2FLw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=59
60. Lec 47 LSTMs and GRUs → https://www.youtube.com/watch?v=Pkuwu4EMRj8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=60
61. Tutorial 11 : Pytorch - Tensors and Data Loaders → https://www.youtube.com/watch?v=vm9FYVtM5ZA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=61
62. Tutorial 12 : Pytorch - Building MLP and Auto Grad → https://www.youtube.com/watch?v=SEEQ2A2WN9g&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=62
63. Tutorial 13 : Pytorch - Training the Model → https://www.youtube.com/watch?v=5PruFG5g1C4&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=63
64. Lec 48 Attention Part1 → https://www.youtube.com/watch?v=00DOOSYFyJA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=64
65. Lec 49 Attention Part2 → https://www.youtube.com/watch?v=TGzZ8jCX4y0&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=65
66. Lec 50 Multi-Head Attention and Transformer Architecture → https://www.youtube.com/watch?v=NQyNZ6Plxzs&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=66
67. Lec 51 Positional Embeddings → https://www.youtube.com/watch?v=xbkJIeLoGyw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=67
68. Lec 52 Transfer Learning and Knowledge Distilation → https://www.youtube.com/watch?v=5f2_TocU_CU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=68
69. Lec 53 SGD, RMS Prop, ADAM : Optimizers → https://www.youtube.com/watch?v=N6F1J-wcE_E&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=69
70. Tutorial 14 Part 1 : CNNs → https://www.youtube.com/watch?v=0wd-LIzzfM0&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=70
71. Tutorial 14 Part 2 : Transfer Learning using CNNs → https://www.youtube.com/watch?v=vocN0cxAT7I&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=71
72. Tutorial 15 Part 1 : RNNs, LSTMs and GRUs → https://www.youtube.com/watch?v=zM5-TlrmKg8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=72
73. Tutorial 15 Part 2 : Deep RNNs, LSTMs and GRUs → https://www.youtube.com/watch?v=nJivfX9VY7Y&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=73
74. Lec 54 Decision Trees and Impurity Measures → https://www.youtube.com/watch?v=b2ScFHIhnB0&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=74
75. Lec 55 Regression Trees → https://www.youtube.com/watch?v=1GZDVRskXGE&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=75
76. Lec 56 Ensemble Methods, Bagging and Boosting → https://www.youtube.com/watch?v=2wyxSgVolJg&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=76
77. Lec 57 Gradient Boosting Algorithm → https://www.youtube.com/watch?v=_YKxmyP5PWU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=77
78. Lec 58 Ada-Boosting → https://www.youtube.com/watch?v=27cAa8L0JQo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=78
79. Lec 59 Cross Validation → https://www.youtube.com/watch?v=R36BJ52Lr1A&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=79
80. Lec 60 Un-Supervised Learning → https://www.youtube.com/watch?v=5sg3d8ObcDc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=80
81. Lec 61 K-Means Clustering → https://www.youtube.com/watch?v=oFr4JX75MCc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=81
82. Lec 62 PCA - Principal Component Analysis → https://www.youtube.com/watch?v=BLYw-vf9q6U&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=82
83. Lec 63 NCE - Noise Contrastive Estimation → https://www.youtube.com/watch?v=DHBl3E0ndRA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=83
84. Lec 64 NCE, Info-NCE, SimCLR, JEPA → https://www.youtube.com/watch?v=vRX_5wwImec&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=84
85. Lec 65 Introduction to Generative Models → https://www.youtube.com/watch?v=hgZ3HOMkrx0&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=85
86. Lec 66 GAN - Generative Adversarial Networks → https://www.youtube.com/watch?v=nd3laZj5Cdg&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=86
87. Lec 67 Variational Auto Encoders : VAEs → https://www.youtube.com/watch?v=8LAKmtw0WvQ&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=87
88. Lec 68 Introduction to Large Language Models : LLMs → https://www.youtube.com/watch?v=41tyjrCeENA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=88
89. Lec 69 Introduction to Reinforcement Learning : RL → https://www.youtube.com/watch?v=5AcmvmzE2zU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=89

---

## External resources

| Resource | URL |
|:---------|:----|
| YouTube playlist (Official NPTEL) | https://www.youtube.com/playlist?list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu |
| NPTEL course page | https://nptel.ac.in/courses/106108841 |
| Swayam portal preview (noc26_cs02) | https://onlinecourses.nptel.ac.in/noc26_cs02/preview |
| Sequel — NPTEL Generative AI packages | [`Mathematical-Foundation-for-GenerativeAI/NOTES.md`](./Mathematical-Foundation-for-GenerativeAI/NOTES.md) |
| Sequel — IITM BS Generative AI catalog | [`IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md`](./IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md) |
| Core Mathematical Knowledge Base | [`MathsTerms/README.md`](./MathsTerms/README.md) |
| Unified Mathematical Concept Map | [`MathsTerms/CONCEPT_MAP.md`](./MathsTerms/CONCEPT_MAP.md) |

---

## Sources

- [NPTEL playlist](https://www.youtube.com/playlist?list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu) (Complete live audit: 89 videos, exact titles, video IDs, and durations)
- [NPTEL 106108841](https://nptel.ac.in/courses/106108841) (Official syllabus: risk minimization, density estimation, regularization, generalization)
- Completed 7-Pillar study packages in this directory (`02-Lec01` through `14-Lec13`)
- Prof. Prathosh A P lecture chalkboard derivations and course blueprint
- Authoritative reference textbooks:
  - Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* (MIT Press, 2022)
  - Christopher M. Bishop, *Pattern Recognition and Machine Learning* (Springer, 2006)
  - Trevor Hastie, Robert Tibshirani, Jerome Friedman, *The Elements of Statistical Learning* (Springer, 2009)
  - Thomas M. Cover and Joy A. Thomas, *Elements of Information Theory* (Wiley, 2006)
