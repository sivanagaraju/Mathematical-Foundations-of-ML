# Mathematical Foundations of Machine Learning

NPTEL / IISc Bangalore · Course **106108841** · noc26-cs02 · 12 weeks · Prof. Prathosh A P

**Playlist:** [Mathematical Foundations of Machine Learning](https://www.youtube.com/playlist?list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu)  
**Channel:** NPTEL — Indian Institute of Science, Bengaluru  
**Size:** 89 videos · ~46.6 hours

This file is the **course map + catalog** for the parent playlist of this repo. Per-lecture study packages (`PREREQUISITES.md` / `NOTES.md` / quiz) sit in numbered folders (`02-Lec01-…` through `14-Lec13-…`) and are linked in the tables below when they exist.

This is the **first** of a two-course sequence. The sequel is Mathematical Foundations of **Generative AI**:

- NPTEL recording: [`Mathematical-Foundation-for-GenerativeAI/`](./Mathematical-Foundation-for-GenerativeAI/)
- IIT Madras BS recording: [`IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md`](./IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md)

---

## Table of Contents

1. [What the course is for](#what-the-course-is-for)
2. [Architecture of the whole course](#architecture-of-the-whole-course)
3. [How the blocks fit](#how-the-blocks-fit)
4. [Study packages already in this repo](#study-packages-already-in-this-repo)
5. [Playlist catalog](#playlist-catalog)
   - [Intro and function approximation](#intro-and-function-approximation)
   - [Probability recap](#probability-recap)
   - [Data as samples; estimation; density](#data-as-samples-estimation-density)
   - [Python / probability tutorials](#python--probability-tutorials)
   - [Entropy, KL, min-KL](#entropy-kl-min-kl)
   - [Risk minimization and Bayes](#risk-minimization-and-bayes)
   - [MLE, latent variables, EM](#mle-latent-variables-em)
   - [Classifier / MLE tutorials](#classifier--mle-tutorials)
   - [EM, MAP, nonparametric](#em-map-nonparametric)
   - [Linear models and bias–variance](#linear-models-and-biasvariance)
   - [Regularization and SVM](#regularization-and-svm)
   - [Neural nets, CNN, RNN](#neural-nets-cnn-rnn)
   - [PyTorch tutorials](#pytorch-tutorials)
   - [Attention, Transformers, transfer, optimizers](#attention-transformers-transfer-optimizers)
   - [CNN / RNN tutorials](#cnn--rnn-tutorials)
   - [Trees, ensembles, cross-validation](#trees-ensembles-cross-validation)
   - [Unsupervised and contrastive](#unsupervised-and-contrastive)
   - [Bridge to generative AI](#bridge-to-generative-ai)
6. [Compact title → URL list](#compact-title--url-list)
7. [External resources](#external-resources)
8. [Sources](#sources)

---

## What the course is for

Machine learning is treated as **function approximation under uncertainty**, not as a zoo of named algorithms. You hold a finite table of pairs (an X-ray and a label, a sentence and the next token). You must answer a **new** input you have never seen. Memorizing the table fails. Physics-style equations often fail when the label is semantic (disease / not). The course’s fork is: treat inputs and labels as **random variables**, estimate their law from data, then decide.

NPTEL abstract: risk minimization, density estimation, regularization, and generalization — from a **probabilistic** viewpoint.

`Lec` = lecture (chalk-and-talk math). `Tutorial` = worked examples / PyTorch.

---

## Architecture of the whole course

**Worldview arc:** deterministic **function approximation** → **probability triplet / RV / $p_x$** → **estimate $p$ by minimizing a divergence** → **risk** (decide) → **models** (linear, SVM, nets, trees) → **unsupervised / contrastive** → a short **generative** coda that hands off to the GenAI course.

One recipe, reused all semester:

```
  Dataset D = {(x_i, y_i)}  or  {x_i}
  treated as draws from unknown p   (IID)

                    │
                    ▼
         pick a model family F
         (density p_θ, classifier g, net, tree, …)
                    │
                    ▼
         pick a discrepancy
         KL / NLL / risk R(g) / hinge / MSE / …
                    │
                    ▼
         train   θ* = arg min_θ  d(p, model)
         (MLE ≡ min-KL; ERM; MAP; EM; SGD)
                    │
                    ▼
         use:  classify / regress / cluster / generate
```

### Main blueprint

```
  Nature: RE → (Ω, F, P)          [almost never observed]
              │
              │  X, Y : Ω → R^d    (sensors / labels)
              ▼
  Files live in range(X):  x ∈ R^d ,  D ~ p_x  (or p_{x,y})
              │
              ▼
  Lec 01–10   FA becomes distribution estimation
              density p  ≠  probability;  recipe p_θ → d → argmin
              │
              ▼
  Lec 11–16   d = KL;  min KL ≡ MLE (drop entropy, LLN)
              risk R(g); Bayes classifier
              │
              ├─ parametric MLE / EM / MAP / Parzen / k-NN
              ├─ linear models, bias–variance, regularization
              ├─ SVM (max-margin, dual, kernel)
              ├─ nets: UAT → backprop → CNN as regularized MLP → RNN/LSTM
              ├─ attention / Transformer / transfer / Adam
              ├─ trees / boosting / CV
              └─ unsupervised: k-means, PCA, NCE / SimCLR
                        │
                        ▼
              Lec 65–69  generative coda (GAN, VAE, LLM, RL)
                        →  GenAI course (full treatment)
```

### What later blocks cannot skip

| Block | Load-bearing claim | Failure if skipped |
|-------|--------------------|--------------------|
| FA (Lec 01) | Model ≠ algorithm; table lookup is not FA | Treating nets as magic, not estimators |
| Probability (Lec 02–05) | Triplet $(\Omega,\mathcal F,P)$; RV $X:\Omega\to\mathbb R^d$ | Confusing files with the hidden world |
| Data (Lec 06–10) | Dataset $\sim p$; density height ≠ probability | Calling a pixel value a probability |
| KL (Lec 11–13) | Train by $\min d(p,p_\theta)$; MLE ≡ min-KL | Inventing a new “loss” with no $d$ |
| Risk (Lec 15–16) | Classifier is $g$; Bayes is optimal for 0–1 | Accuracy without a risk |
| EM / LVM | Incomplete data → bound, don’t pretend $z$ is observed | “Just maximize the incomplete likelihood” |
| Regularization | ERM overfits; MAP / penalty / SGD are the same job | Tuning λ with no statistical reading |
| Nets | Backprop = ERM on a compositional model | CNN/RNN as unrelated gadgets |
| GenAI coda | Estimate **and** sample | Jumping to the sequel without $p_x$ |

---

## How the blocks fit

| Block | Videos | What actually happens |
|-------|--------|------------------------|
| Intro + FA | 1–2 | Course mission. FA: estimate $f$ from a finite table. Physics vs stats. |
| Probability | 3–6 | RE → Ω → events → $P$; RV $X$; pushforward / CDF; joints, conditionals, margins. |
| Data / estimation | 7–11 | X-ray ∈ range$(X)$; IID; given $D$ estimate $p$; density $p$; recipe $p_\theta\to d\to\arg\min$. |
| Info theory | 14–16 | Surprisal, entropy, cross-entropy, KL; min KL → MLE. |
| Risk | 17–20 | MLE example; risk $R(g)$; Bayes classifier; tutorial. |
| MLE / LVM / EM | 21–31 | Gaussian / discrete / mixed MLE; latent $z$; EM; minmax / NP classifiers. |
| MAP / nonparametric | 32–38 | EM convergence; GMM; MAP; Parzen; nearest neighbour. |
| Linear + BV | 39–45 | OLS, GLS, linear classification, bias–variance. |
| Regularization + SVM | 46–53 | Penalty / MAP / SGD; max-margin; dual; kernel. |
| Neural nets | 54–63 | UAT, backprop, CNN, RNN, LSTM/GRU; PyTorch. |
| Sequence / transfer | 64–73 | Attention, Transformer, positional embeddings, distillation, Adam. |
| Trees / ensembles | 74–79 | Impurity, regression trees, bagging, boosting, AdaBoost, CV. |
| Unsupervised | 80–84 | k-means, PCA, NCE, InfoNCE / SimCLR / JEPA. |
| GenAI bridge | 85–89 | Generative models, GAN, VAE, LLMs, RL — trailer for the sequel. |

Suggested first watch: **Lec 01** (FA, 48 min), then **Lec 02–05** (probability), then **Lec 06–10** (data as $p$). That is the spine already packaged in this repo.

---

## Study packages already in this repo

| Playlist # | Video | Folder |
|------------|-------|--------|
| 2 | Lec 01 Overview of Function Approximation | [`02-Lec01-Overview-Function-Approximation/`](./02-Lec01-Overview-Function-Approximation/) |
| 3 | Lec 02 Recap of Probability Theory - 1, Part 1 | [`03-Lec02-Recap-Probability-Theory-Part1/`](./03-Lec02-Recap-Probability-Theory-Part1/) |
| 4 | Lec 03 Recap of Probability Theory - 1, Part 2 | [`04-Lec03-Recap-Probability-Theory-Part2/`](./04-Lec03-Recap-Probability-Theory-Part2/) |
| 5 | Lec 04 Recap of Probability Theory - 1, Part 3 | [`05-Lec04-Recap-Probability-Theory-Part3/`](./05-Lec04-Recap-Probability-Theory-Part3/) |
| 6 | Lec 05 Recap of Probability Theory Part 2 | [`06-Lec05-Recap-Probability-Theory-Part2/`](./06-Lec05-Recap-Probability-Theory-Part2/) |
| 7 | Lec 06 Understanding a Chest X-Ray as Sample from Distribution | [`07-Lec06-XRay-Sample-From-Distribution/`](./07-Lec06-XRay-Sample-From-Distribution/) |
| 8 | Lec 07 IID Assumption | [`08-Lec07-IID-Assumption/`](./08-Lec07-IID-Assumption/) |
| 9 | Lec 08 Distribution Estimation | [`09-Lec08-Distribution-Estimation/`](./09-Lec08-Distribution-Estimation/) |
| 10 | Lec 09 Density Function | [`10-Lec09-Density-Function/`](./10-Lec09-Density-Function/) |
| 11 | Lec 10 Challenge With ML | [`11-Lec10-Challenges-of-ML/`](./11-Lec10-Challenges-of-ML/) |
| 14 | Lec 11 Entropy | [`12-Lec11-Entropy/`](./12-Lec11-Entropy/) |
| 15 | Lec 12 Kullback-Leibler (KL) Divergence | [`13-Lec12-KL-Divergence/`](./13-Lec12-KL-Divergence/) |
| 16 | Lec 13 Minimization of KL Divergence | [`14-Lec13-Minimization-of-KL/`](./14-Lec13-Minimization-of-KL/) |

Intro, tutorials, and Lec 14 onward do not yet have packages here.

---

## Playlist catalog

Links keep the playlist id so YouTube stays in-list:

`https://www.youtube.com/watch?v=VIDEO_ID&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=N`

Summaries are from official titles plus this repo’s lecture packages (Lec 01–13) and the published course abstract. They are not substitutes for watching.

**Package** = existing study folder in this repo, when there is one.

### Intro and function approximation

| # | Video | Duration | Link | Summary | Package |
|---|--------|----------|------|---------|---------|
| 1 | Mathematical Foundations of Machine Learning (Intro) | 3:33 | [watch](https://www.youtube.com/watch?v=vbs9WGWjS9U&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=1) | Course trailer: ML from a probabilistic viewpoint; first of a two-course sequence (GenAI is next). | |
| 2 | Lec 01 Overview of Function Approximation | 47:50 | [watch](https://www.youtube.com/watch?v=G2h7nD_Stxg&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=2) | FA is the core job: from a finite table, estimate unknown $f$ for new $x$. Model ≠ algorithm. Physics often fails; probability is the fork. | [02](./02-Lec01-Overview-Function-Approximation/) |

### Probability recap

| # | Video | Duration | Link | Summary | Package |
|---|--------|----------|------|---------|---------|
| 3 | Lec 02 Recap of Probability Theory - 1, Part 1 | 32:13 | [watch](https://www.youtube.com/watch?v=YLx3hBqt28k&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=3) | Random experiment → sample space Ω → events → probability $P$. Why ML needs a measure. | [03](./03-Lec02-Recap-Probability-Theory-Part1/) |
| 4 | Lec 03 Recap of Probability Theory - 1, Part 2 | 14:30 | [watch](https://www.youtube.com/watch?v=DaBw9qBpt2s&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=4) | Random variable $X:\Omega\to\mathbb R^d$. Sensors turn hidden outcomes into number lists. | [04](./04-Lec03-Recap-Probability-Theory-Part2/) |
| 5 | Lec 04 Recap of Probability Theory - 1, Part 3 | 29:06 | [watch](https://www.youtube.com/watch?v=0R6Agp4tqSU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=5) | Pushforward / CDF. Density trap. Vector RV and a preview of several RVs. | [05](./05-Lec04-Recap-Probability-Theory-Part3/) |
| 6 | Lec 05 Recap of Probability Theory Part 2 | 21:43 | [watch](https://www.youtube.com/watch?v=R69wew8RrPo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=6) | Joints, conditionals, margins. A $d$-vector RV ≡ $d$ scalar RVs. | [06](./06-Lec05-Recap-Probability-Theory-Part2/) |

### Data as samples; estimation; density

| # | Video | Duration | Link | Summary | Package |
|---|--------|----------|------|---------|---------|
| 7 | Lec 06 Understanding a Chest X-Ray as Sample from Distribution | 26:51 | [watch](https://www.youtube.com/watch?v=bdcvsSNAHIk&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=7) | Image ∈ range$(X)$. Data is not $P$. Labels need a joint. A dataset is draws from $p_{X,Y}$. | [07](./07-Lec06-XRay-Sample-From-Distribution/) |
| 8 | Lec 07 IID Assumption | 30:42 | [watch](https://www.youtube.com/watch?v=C83xmx80tMo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=8) | Identical + independent **across** points. ML = estimate $P$ from IID files. | [08](./08-Lec07-IID-Assumption/) |
| 9 | Lec 08 Distribution Estimation | 28:47 | [watch](https://www.youtube.com/watch?v=aYb8KG9JYsg&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=9) | Given $D$, estimate $P$. Targets $P(Y\|X)$, $P(Y)$, …. Discriminative vs generative. | [09](./09-Lec08-Distribution-Estimation/) |
| 10 | Lec 09 Density Function | 8:06 | [watch](https://www.youtube.com/watch?v=_QrezNPmxDk&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=10) | Density $p$: height ≠ probability. Uniform-on-$[0,0.5]$ has height 2. Estimate $p$. | [10](./10-Lec09-Density-Function/) |
| 11 | Lec 10 Challenge With ML | 35:31 | [watch](https://www.youtube.com/watch?v=767MLwniPKE&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=11) | True $p$ is unknown. Recipe: family $p_\theta$, divergence $d$, $\arg\min$. Model ≠ algorithm. | [11](./11-Lec10-Challenges-of-ML/) |

### Python / probability tutorials

| # | Video | Duration | Link | Summary |
|---|--------|----------|------|---------|
| 12 | Tutorial 1 : Introduction to Python Basics | 47:10 | [watch](https://www.youtube.com/watch?v=cF025BechXo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=12) | Python warm-up for later numerical examples. |
| 13 | Tutorial 2 : Simple Problem solving in Probability Theory | 53:51 | [watch](https://www.youtube.com/watch?v=nGwjqvLHguA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=13) | Worked probability problems that lock Lec 02–05. |

### Entropy, KL, min-KL

| # | Video | Duration | Link | Summary | Package |
|---|--------|----------|------|---------|---------|
| 14 | Lec 11 Entropy | 17:56 | [watch](https://www.youtube.com/watch?v=P6wjLz4dRTs&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=14) | Surprisal $-\log P$. Entropy $H=-\sum p\log p$. Need a divergence $d$ to train. | [12](./12-Lec11-Entropy/) |
| 15 | Lec 12 Kullback-Leibler (KL) Divergence | 16:49 | [watch](https://www.youtube.com/watch?v=ihkGbIdbbxc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=15) | Cross-entropy; $\mathrm{KL}=\mathrm{CE}-H$. Asymmetric; not a metric. | [13](./13-Lec12-KL-Divergence/) |
| 16 | Lec 13 Minimization of KL Divergence | 24:52 | [watch](https://www.youtube.com/watch?v=Ij4p5hLbfo4&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=16) | $\min\mathrm{KL}$ drops $H$; LLN; MLE ≡ min-KL estimator. | [14](./14-Lec13-Minimization-of-KL/) |

### Risk minimization and Bayes

| # | Video | Duration | Link | Summary |
|---|--------|----------|------|---------|
| 17 | Lec 14 Example of ML Estimate | 20:24 | [watch](https://www.youtube.com/watch?v=mEpXOyLwbxA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=17) | Concrete MLE after the min-KL theorem — see the estimator on a named family. |
| 18 | Lec 15 Risk Minimization Framework | 40:14 | [watch](https://www.youtube.com/watch?v=jXCqrFVGwoU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=18) | Decision $g$; risk $R(g)=\mathbb E[\ell(Y,g(X))]$. ERM is empirical risk. |
| 19 | Lec 16 Bayes Classifier | 35:37 | [watch](https://www.youtube.com/watch?v=-y3SSAIhD4Y&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=19) | Optimal $g$ for 0–1 loss is $\arg\max_y P(Y=y\|X=x)$. |
| 20 | Tutorail 3 : Risk Minimization Framework | 59:52 | [watch](https://www.youtube.com/watch?v=AQ3einJJrr0&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=20) | Worked risk-minimization examples (title spelling is the YouTube original). |

### MLE, latent variables, EM

| # | Video | Duration | Link | Summary |
|---|--------|----------|------|---------|
| 21 | Lec 17 MLE for Gaussian Distribution | 26:07 | [watch](https://www.youtube.com/watch?v=tF-RrzUnnYA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=21) | Closed-form MLE for $\mathcal N(\mu,\sigma^2)$: sample mean / variance. |
| 22 | Lec 18 MLE for Generalized Discrete Random Variable | 25:48 | [watch](https://www.youtube.com/watch?v=j7jbpicYdik&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=22) | Categorical / multinomial MLE: frequencies. |
| 23 | Lec 19 Density Estimation for Mixed Distribution | 20:41 | [watch](https://www.youtube.com/watch?v=3UmgTSDgG5Q&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=23) | Mixed discrete–continuous laws; one density formula does not cover both. |
| 24 | Lec 20 Latent Variable Models | 33:23 | [watch](https://www.youtube.com/watch?v=J9QNr4UrB2c&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=24) | Hidden $z$: $p(x)=\int p(x\|z)p(z)\,dz$ is the incomplete likelihood. |
| 25 | Lec 21 MLE for Latent Variable Models | 16:08 | [watch](https://www.youtube.com/watch?v=BMj-TWtK83A&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=25) | Why naive MLE on incomplete data is intractable — need a bound / EM. |
| 26 | Lec 22 Expectation Maximization Algorithm | 25:34 | [watch](https://www.youtube.com/watch?v=ejma0iH1pXE&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=26) | E-step: fill in $z$ in expectation. M-step: maximize the complete-data $Q$. |

### Classifier / MLE tutorials

| # | Video | Duration | Link | Summary |
|---|--------|----------|------|---------|
| 27 | Tutorial 4 : Minmax Classifier | 26:33 | [watch](https://www.youtube.com/watch?v=ENpzs2ycXJE&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=27) | Minimax classifier when priors are unknown or adversarial. |
| 28 | Tutorial 5 : Neyman Pearson Classifier | 49:00 | [watch](https://www.youtube.com/watch?v=8esVIly2TZY&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=28) | Neyman–Pearson: fix type-I error, minimize type-II (likelihood ratio). |
| 29 | Tutorial 6 : Example of NP Classifier, ROC Curve | 32:40 | [watch](https://www.youtube.com/watch?v=JwQEaTqyBDw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=29) | NP example and ROC as the threshold sweep. |
| 30 | Tutorial 7A : MLE for Gaussian Distribution | 40:17 | [watch](https://www.youtube.com/watch?v=XA3UiD8zEF8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=30) | Numerical Gaussian MLE (pairs with Lec 17). |
| 31 | Tutorial 7B : MLE for Generalized Discrete Distribution | 23:08 | [watch](https://www.youtube.com/watch?v=o5697P6KZoc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=31) | Numerical discrete MLE (pairs with Lec 18). |

### EM, MAP, nonparametric

| # | Video | Duration | Link | Summary |
|---|--------|----------|------|---------|
| 32 | Lec 23 Convergence of EM | 19:27 | [watch](https://www.youtube.com/watch?v=zHchxrSwOu4&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=32) | EM does not decrease the observed likelihood; local maxima still exist. |
| 33 | Lec 24 EM for GMMs | 30:28 | [watch](https://www.youtube.com/watch?v=TSNsiglfduQ&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=33) | Gaussian mixture: responsibilities (E) and weighted mean/cov (M). |
| 34 | Lec 25 MAP Estimate | 38:24 | [watch](https://www.youtube.com/watch?v=HH9Xjjj7UN4&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=34) | MAP $= \arg\max_\theta p(\theta\|D)$; prior as regularizer. |
| 35 | Lec 26 Parzen Window | 29:07 | [watch](https://www.youtube.com/watch?v=boCvzXvUVMI&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=35) | Kernel density estimate: nonparametric $p$ with a window / bandwidth. |
| 36 | Lec 27 Nearest Neighbor Classifier | 17:32 | [watch](https://www.youtube.com/watch?v=YZ3Xa6dEMl8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=36) | $k$-NN as a nonparametric classifier (local votes, no $p_\theta$). |
| 37 | Tutorial 8 : Computation of EM for GMMs | 40:49 | [watch](https://www.youtube.com/watch?v=sOwgRt6uiA4&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=37) | Numeric GMM EM. |
| 38 | Tutorial 9 : MAP Estimate | 22:05 | [watch](https://www.youtube.com/watch?v=7y4V0GUoyaw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=38) | Numeric MAP (pairs with Lec 25). |

### Linear models and bias–variance

| # | Video | Duration | Link | Summary |
|---|--------|----------|------|---------|
| 39 | Lec 28 Ordinary Least Squares (OLS) | 24:48 | [watch](https://www.youtube.com/watch?v=s_DfCCobgnA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=39) | Linear regression as MLE under Gaussian noise; normal equations. |
| 40 | Lec 29 Generalized Least Squares (GLS) | 25:19 | [watch](https://www.youtube.com/watch?v=UfiHgztGgu8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=40) | GLS when noise is correlated / heteroscedastic. |
| 41 | Lec 30 Linear Models for Classification | 33:29 | [watch](https://www.youtube.com/watch?v=EydAoMbslkc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=41) | Linear decision surfaces; least-squares / logistic as linear classifiers. |
| 42 | Lec 31 Bias - Variance Decomposition and Analysis | 46:37 | [watch](https://www.youtube.com/watch?v=0RCDPOz3YVc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=42) | $\mathbb E[(f-\hat f)^2]=\mathrm{bias}^2+\mathrm{var}+\mathrm{noise}$. Capacity vs sample size. |
| 43 | Lec 32 Bias & Variance in Practice | 37:14 | [watch](https://www.youtube.com/watch?v=E-kOTTO5hK8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=43) | How under/overfit show up; what you can actually tune. |
| 44 | Tutorial 10 Part A : Numerical Example on Bayes Classifier | 51:38 | [watch](https://www.youtube.com/watch?v=oTEPAiwv-00&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=44) | Numeric Bayes classifier. |
| 45 | Tutorial 10 Part B : Numerical Example on MLE and MAP Estimate | 30:58 | [watch](https://www.youtube.com/watch?v=ysjGmQW4HOo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=45) | Numeric MLE vs MAP side by side. |

### Regularization and SVM

| # | Video | Duration | Link | Summary |
|---|--------|----------|------|---------|
| 46 | Lec 33 Regularization | 28:27 | [watch](https://www.youtube.com/watch?v=7F8pknXk_-o&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=46) | Penalty on $\theta$ to cut variance (ridge / weight decay as the prototype). |
| 47 | Lec 34 Regularized ERM and MAP Estimate | 26:57 | [watch](https://www.youtube.com/watch?v=rypIu-ZSYBo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=47) | Regularized ERM ≡ MAP for a matching prior. |
| 48 | Lec 35 Stochastic Gradient Descent as a Regularizer | 22:02 | [watch](https://www.youtube.com/watch?v=vKTxP9FsR90&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=48) | Mini-batch noise as implicit regularization, not only an optimizer trick. |
| 49 | Lec 36 Max-Margin Classifier and SVM | 43:45 | [watch](https://www.youtube.com/watch?v=joL7g6DSxPU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=49) | Widest street between classes; SVM as max-margin. |
| 50 | Lec 37 SVM Formulation | 33:42 | [watch](https://www.youtube.com/watch?v=aO3FTnrf2bQ&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=50) | Primal SVM with slack variables (soft margin). |
| 51 | Lec 38 Dual Function in SVM | 21:10 | [watch](https://www.youtube.com/watch?v=RXzcClx44Tw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=51) | Dual: solution lives on support vectors; inner products appear. |
| 52 | Lec 39 SVM for Non-Linear Seperable Case | 34:09 | [watch](https://www.youtube.com/watch?v=jQ-3gT8Mytw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=52) | Feature map $\phi(x)$ so a linear SVM in $\phi$-space is nonlinear in $x$. |
| 53 | Lec 40 SVM with Kernel Function | 29:53 | [watch](https://www.youtube.com/watch?v=dDIutyWTPKA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=53) | Kernel trick: $k(x,x')=\langle\phi(x),\phi(x')\rangle$ without building $\phi$. |

### Neural nets, CNN, RNN

| # | Video | Duration | Link | Summary |
|---|--------|----------|------|---------|
| 54 | Lec 41 Neural Networks and Universal Approximation Theorem | 30:43 | [watch](https://www.youtube.com/watch?v=npYHSFuqnzs&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=54) | MLP as a model family; UAT: wide enough nets can approximate continuous $f$. |
| 55 | Lec 42 ERM on Neural Networks and Error Backpropagation | 47:44 | [watch](https://www.youtube.com/watch?v=dONDRwX_83E&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=55) | Backprop = chain rule for ERM on a compositional net. |
| 56 | Lec 43 Local Receptive Field and Parameter Sharing | 36:12 | [watch](https://www.youtube.com/watch?v=rm0VmbTQE8Y&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=56) | Why convolution: local filters + tied weights as **inductive bias**. |
| 57 | Lec 44 Convolutional Neural Networks(CNNs) as Regularized MLP | 44:57 | [watch](https://www.youtube.com/watch?v=mSYTyrXCsA8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=57) | CNN = MLP with hard zeros and shared weights (regularization, not a new species). |
| 58 | Lec 45 Recurrent Neural Networks(RNNs) | 38:05 | [watch](https://www.youtube.com/watch?v=E2LLi7AB9lQ&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=58) | Shared transition $h_t=f(h_{t-1},x_t)$ for sequences. |
| 59 | Lec 46 Back Prapogation in RNNs and Vanishing Gradients Problem | 29:28 | [watch](https://www.youtube.com/watch?v=TVkaROL2FLw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=59) | BPTT; products of Jacobians vanish / explode. |
| 60 | Lec 47 LSTMs and GRUs | 22:19 | [watch](https://www.youtube.com/watch?v=Pkuwu4EMRj8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=60) | Gates + additive cell path so long-range gradients can survive. |

### PyTorch tutorials

| # | Video | Duration | Link | Summary |
|---|--------|----------|------|---------|
| 61 | Tutorial 11 : Pytorch - Tensors and Data Loaders | 45:19 | [watch](https://www.youtube.com/watch?v=vm9FYVtM5ZA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=61) | Tensors and `DataLoader` for the net lectures. |
| 62 | Tutorial 12 : Pytorch - Building MLP and Auto Grad | 42:47 | [watch](https://www.youtube.com/watch?v=SEEQ2A2WN9g&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=62) | `nn.Module` + autograd (pairs with Lec 42). |
| 63 | Tutorial 13 : Pytorch - Training the Model | 33:37 | [watch](https://www.youtube.com/watch?v=5PruFG5g1C4&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=63) | Train loop: loss, backward, step. |

### Attention, Transformers, transfer, optimizers

| # | Video | Duration | Link | Summary |
|---|--------|----------|------|---------|
| 64 | Lec 48 Attention Part1 | 23:31 | [watch](https://www.youtube.com/watch?v=00DOOSYFyJA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=64) | Attention as a content-based mixer: queries, keys, values. |
| 65 | Lec 49 Attention Part2 | 33:56 | [watch](https://www.youtube.com/watch?v=TGzZ8jCX4y0&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=65) | Scaled dot-product attention; masking for autoregression. |
| 66 | Lec 50 Multi-Head Attention and Transformer Architecture | 27:25 | [watch](https://www.youtube.com/watch?v=NQyNZ6Plxzs&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=66) | Multi-head + FFN + residuals = Transformer block. |
| 67 | Lec 51 Positional Embeddings | 13:29 | [watch](https://www.youtube.com/watch?v=xbkJIeLoGyw&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=67) | Attention is permutation-equivariant; positions must be injected. |
| 68 | Lec 52 Transfer Learning and Knowledge Distilation | 37:35 | [watch](https://www.youtube.com/watch?v=5f2_TocU_CU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=68) | Reuse a pretrained $f$; distillation: student matches a teacher. |
| 69 | Lec 53 SGD, RMS Prop, ADAM : Optimizers | 19:28 | [watch](https://www.youtube.com/watch?v=N6F1J-wcE_E&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=69) | SGD → RMSProp (scale by RMS of grads) → Adam (momentum + RMS). |

### CNN / RNN tutorials

| # | Video | Duration | Link | Summary |
|---|--------|----------|------|---------|
| 70 | Tutorial 14 Part 1 : CNNs | 33:52 | [watch](https://www.youtube.com/watch?v=0wd-LIzzfM0&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=70) | CNN implementation (pairs with Lec 43–44). |
| 71 | Tutorial 14 Part 2 : Transfer Learning using CNNs | 28:50 | [watch](https://www.youtube.com/watch?v=vocN0cxAT7I&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=71) | Fine-tune a pretrained CNN (pairs with Lec 52). |
| 72 | Tutorial 15 Part 1 : RNNs, LSTMs and GRUs | 34:07 | [watch](https://www.youtube.com/watch?v=zM5-TlrmKg8&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=72) | RNN / LSTM / GRU in code. |
| 73 | Tutorial 15 Part 2 : Deep RNNs, LSTMs and GRUs | 16:53 | [watch](https://www.youtube.com/watch?v=nJivfX9VY7Y&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=73) | Stacked recurrent nets. |

### Trees, ensembles, cross-validation

| # | Video | Duration | Link | Summary |
|---|--------|----------|------|---------|
| 74 | Lec 54 Decision Trees and Impurity Measures | 40:23 | [watch](https://www.youtube.com/watch?v=b2ScFHIhnB0&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=74) | Axis-aligned splits; impurity (Gini / entropy) as the split score. |
| 75 | Lec 55 Regression Trees | 34:09 | [watch](https://www.youtube.com/watch?v=1GZDVRskXGE&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=75) | Trees for $y\in\mathbb R$: split to cut squared error. |
| 76 | Lec 56 Ensemble Methods, Bagging and Boosting | 35:55 | [watch](https://www.youtube.com/watch?v=2wyxSgVolJg&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=76) | Average many weak $g$: bagging (variance) vs boosting (bias). |
| 77 | Lec 57 Gradient Boosting Algorithm | 30:24 | [watch](https://www.youtube.com/watch?v=_YKxmyP5PWU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=77) | Fit the next tree to the residual / negative gradient of the loss. |
| 78 | Lec 58 Ada-Boosting | 40:08 | [watch](https://www.youtube.com/watch?v=27cAa8L0JQo&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=78) | Reweight misclassified points; AdaBoost as exponential-loss boosting. |
| 79 | Lec 59 Cross Validation | 12:26 | [watch](https://www.youtube.com/watch?v=R36BJ52Lr1A&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=79) | $k$-fold CV as a stand-in for risk when you cannot see $p$. |

### Unsupervised and contrastive

| # | Video | Duration | Link | Summary |
|---|--------|----------|------|---------|
| 80 | Lec 60 Un-Supervised Learning | 20:44 | [watch](https://www.youtube.com/watch?v=5sg3d8ObcDc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=80) | No $y$: still estimate structure of $p_x$ (clusters, factors, embeddings). |
| 81 | Lec 61 K-Means Clustering | 36:11 | [watch](https://www.youtube.com/watch?v=oFr4JX75MCc&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=81) | Alternate assignment / centroid; hard-assignment cousin of GMM. |
| 82 | Lec 62 PCA - Principal Component Analysis | 45:42 | [watch](https://www.youtube.com/watch?v=BLYw-vf9q6U&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=82) | Linear subspace of max variance; eigendecomposition of the covariance. |
| 83 | Lec 63 NCE - Noise Contrastive Estimation | 41:02 | [watch](https://www.youtube.com/watch?v=DHBl3E0ndRA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=83) | Learn $p$ by classifying data vs noise — density without the partition function. |
| 84 | Lec 64 NCE, Info-NCE, SimCLR, JEPA | 35:36 | [watch](https://www.youtube.com/watch?v=vRX_5wwImec&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=84) | Contrastive self-supervision: InfoNCE, SimCLR, JEPA as NCE descendants. |

### Bridge to generative AI

These five lectures are a **coda**, not the full GenAI course. For the sequel use the NPTEL or IITM-BS folders linked at the top.

| # | Video | Duration | Link | Summary |
|---|--------|----------|------|---------|
| 85 | Lec 65 Introduction to Generative Models | 37:10 | [watch](https://www.youtube.com/watch?v=hgZ3HOMkrx0&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=85) | Estimate $p_x$ **and** sample. Hands the FA/risk course off to generative modelling. |
| 86 | Lec 66 GAN - Generative Adversarial Networks | 41:51 | [watch](https://www.youtube.com/watch?v=nd3laZj5Cdg&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=86) | Implicit $G(z)$; discriminator as a density-ratio / JS-style game. |
| 87 | Lec 67 Variational Auto Encoders : VAEs | 42:06 | [watch](https://www.youtube.com/watch?v=8LAKmtw0WvQ&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=87) | Encoder $q_\phi(z\|x)$, decoder $p_\theta(x\|z)$, train by ELBO (ties back to Lec 20–22). |
| 88 | Lec 68 Introduction to Large Language Models : LLMs | 25:22 | [watch](https://www.youtube.com/watch?v=41tyjrCeENA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=88) | Autoregressive $p(x)=\prod_i p(x_i\|x_{<i})$ on the Transformer from Lec 48–51. |
| 89 | Lec 69 Introduction to Reinforcement Learning : RL | 31:13 | [watch](https://www.youtube.com/watch?v=5AcmvmzE2zU&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=89) | MDP / policy / return — the vocabulary for aligning LMs (full PPO/DPO lives in the GenAI course). |

---

## Compact title → URL list

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
53. Lec 40 SVM with Kernel Function → https://www.youtube.com/watch?v=dDIutyWTPKA&list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu&index=53
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
|----------|-----|
| YouTube playlist | https://www.youtube.com/playlist?list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu |
| NPTEL course page | https://nptel.ac.in/courses/106108841 |
| Swayam preview (noc26_cs02) | https://onlinecourses.nptel.ac.in/noc26_cs02/preview |
| Sequel — NPTEL GenAI packages | [`Mathematical-Foundation-for-GenerativeAI/`](./Mathematical-Foundation-for-GenerativeAI/) |
| Sequel — IITM BS GenAI catalog | [`IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md`](./IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md) |

---

## Sources

- [NPTEL playlist](https://www.youtube.com/playlist?list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu) (89 video titles, ids, durations)
- [NPTEL 106108841](https://nptel.ac.in/courses/106108841) (course abstract: risk minimization, density estimation, regularization, generalization)
- Lecture packages in this repo for Lec 01–13 (claim-mined NOTES)
- Instructor: first of a two-course probabilistic ML → GenAI sequence
