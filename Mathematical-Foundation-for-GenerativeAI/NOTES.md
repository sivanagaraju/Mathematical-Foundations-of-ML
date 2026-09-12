# Mathematical Foundations of Generative AI

NPTEL / IISc Bangalore · Course **106108004** · noc26-cs97 · Prof. Prathosh A P

**Playlist:** [Mathematical Foundations of Generative AI](https://www.youtube.com/playlist?list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK)  
**Channel:** NPTEL — Indian Institute of Science, Bengaluru  
**Size (uploaded so far):** 36 videos · 26h 16m (~26.3 hours) · YouTube lists **newest first**  
**Catalog Status:** All 36 videos cataloged, reconciled with YouTube lecture numbering fixes, and mapped to learning sequences & study packages.

This file is the **course map + catalog** for the NPTEL recording. Per-lecture study packages live in the numbered folders beside this file and are linked below.

Same instructor and same math as the IIT Madras BS recording. **Different videos.** Full 12-week IITM catalog: [`../IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md`](../IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md)

This is the **sequel** to Mathematical Foundations of Machine Learning: [`../NOTES.md`](../NOTES.md)

---

## Table of Contents

1. [What the course is for](#what-the-course-is-for)
2. [Architecture of the whole course](#architecture-of-the-whole-course)
3. [How the uploaded blocks fit](#how-the-uploaded-blocks-fit)
4. [Study packages status and catalog (all 36 videos)](#study-packages-status-and-catalog-all-36-videos)
5. [Playlist catalog (learning order)](#playlist-catalog-learning-order)
   - [Intro, Lec 01–02, Python / NumPy](#intro-lec-0102-python--numpy)
   - [PyTorch, CNN, RNN, transfer](#pytorch-cnn-rnn-transfer)
   - [Probability and ML review tutorials](#probability-and-ml-review-tutorials)
   - [f-divergence and VDM](#f-divergence-and-vdm)
   - [GANs, WGAN, inversion](#gans-wgan-inversion)
   - [WGAN PyTorch implementations](#wgan-pytorch-implementations)
   - [Latent-variable models, VAE, beta-VAE, VQ-VAE](#latent-variable-models-vae-beta-vae-vq-vae)
   - [Diffusion Models (DDPM Engine)](#diffusion-models-ddpm-engine)
   - [Pedagogy in the Times of AI](#pedagogy-in-the-times-of-ai)
6. [YouTube playlist order (newest first)](#youtube-playlist-order-newest-first)
7. [Compact title → URL list](#compact-title--url-list)
8. [External resources](#external-resources)
9. [Sources](#sources)

---

## What the course is for

Generative AI is treated as a **probability problem**: given files (images, tokens, spectrograms) treated as draws from unknown $p_x$, you must

1. **Estimate** that law (or a surrogate good enough to sample from).
2. **Sample** a brand-new file that was never in the pile.

NPTEL abstract: deep generative models — VAEs, GANs, autoregressive models, diffusion, LLMs — with rigorous proofs and PyTorch implementations.

`Lec` = chalk-and-talk math. `Tutorial` = NumPy / PyTorch / worked probability.

> [!NOTE]
> **YouTube Title Renaming Note:**
> NPTEL previously uploaded three videos mislabeled as Lec 18, Lec 19, and Lec 20. YouTube has since updated them to their rightful sequential titles:
> - `Lec 18 WGAN` $\rightarrow$ **`Lec 06 Wasserstein GAN (WGAN)`** (folder `30-Lec18-Wasserstein-GAN`)
> - `Lec 19 Inversion with GANs and FID` $\rightarrow$ **`Lec 07 Inversion with GANs and FID`** (folder `31-Lec19-Inversion-GANs-FID`)
> - `Lec 20 Latent Variable Models & VAE` $\rightarrow$ **`Lec 08 Latent Variable Models and Introduction to Variational Autoencoder (VAE)`** (folder `32-Lec20-Latent-Variable-Models-VAE`)
>
> All lectures from **Lec 01 to Lec 18** and tutorials from **Tutorial 01 to Tutorial 17** are now live in the playlist!

---

## Architecture of the whole course

**Worldview arc:** from the MFML recipe “estimate $p$ then decide” **to** “estimate $p_x$ **and sample**” **to** four generative sampling engines:
1. **Adversarial (GAN / WGAN / WGAN-GP)**
2. **Variational (VAE / $\beta$-VAE / VQ-VAE)**
3. **Diffusion (DDPM / Score Matching / Denoising SDE)**
4. **Autoregressive / Sequence Models & Alignment**

```
  Dataset D = {x_1, …, x_n}  ⊂  R^d ,   x_i ~ p_x  (unknown)

                    │
                    ▼
         pick a generator family
         (implicit G_θ(z)  or  latent p_θ(x|z)  or  Markov diffusion p_θ(x_{t-1}|x_t))
                    │
                    ▼
         pick a discrepancy d(p_x, model)
         f-div / JS / Wasserstein / ELBO / VLB score / NLL
                    │
                    ▼
         train θ  (min  or  min_θ max_w saddle)
                    │
                    ▼
         SAMPLE  x_new  from the trained object
```

### Uploaded so far vs syllabus coverage

```
  MFML parent: FA → p → min KL / risk → nets
              │
              ▼
  Lec 01–02   Restated job: estimate p_x  AND  sample; files ∈ range(X) ⊂ R^d
              │
              ├─ Tut 01–06   Python, NumPy, PyTorch, CNN, RNN, transfer
              ├─ Tut 07–10   Probability + ML review (triplet, RV, MLE/EM)
              │
  ENGINE 1: ADVERSARIAL (IMPLICIT GENERATOR)
              ├─ Lec 03–04   f-divergence → VDM saddle (critic T vs G_θ)
              ├─ Lec 05      GAN as the JS special case of VDM
              ├─ Tut 12      Vanilla / DC / conditional GAN in PyTorch
              ├─ Lec 06      Wasserstein GAN (WGAN); 1-Lipschitz critic via Kantorovich-Rubinstein
              ├─ Tut 13–14   WGAN code: Weight Clipping vs Gradient Penalty (WGAN-GP)
              └─ Lec 07      Inverting G (BiGAN/ALI) and measuring generation with FID
              │
  ENGINE 2: VARIATIONAL (LATENT VARIABLE MODELS)
              ├─ Lec 08      Latent variables, incomplete likelihood, Jensen's ELBO
              ├─ Lec 09      VAEs Part 1: Boxed ELBO, encoder/decoder, LOTUS reparameterization
              ├─ Lec 10      VAEs Part 2: Gaussian closed-form KL, reconstruction loss, training loop
              ├─ Lec 11      Beta-VAE: Disentanglement factor β, capacity control
              ├─ Tut 15      VAE & Beta-VAE implementation in PyTorch
              ├─ Lec 12      Vector Quantised VAE (VQ-VAE): Discrete latents & codebook lookup
              └─ Tut 16      VQ-VAE implementation: Straight-Through Estimator (STE) in PyTorch
              │
  ENGINE 3: DIFFUSION MODELS (NON-EQUILIBRIUM DENOISING)
              ├─ Lec 13      Introduction to Diffusion: Physical analogy, forward vs reverse
              ├─ Lec 14      Diffusion Part 1: DDPM forward Markov chain, Gaussian kernel q(x_t|x_0)
              ├─ Lec 15      Diffusion Part 2: Variational Lower Bound (VLB), posterior q(x_{t-1}|x_t, x_0)
              ├─ Lec 17      Diffusion Part 3: Score matching, noise prediction ε_θ, simplified loss
              ├─ Lec 18      Diffusion Part 4: Ancestral sampling loop, variance schedules, guidance
              └─ Tut 17      Implementation overview: UNet, noise schedules, DDPM sampling in PyTorch
              │
  SPECIAL / PERSPECTIVE
              └─ Lec 16      Pedagogy in the Times of AI (Prof. Prathosh seminar on cognition & AI)
```

---

## How the uploaded blocks fit

| Block | Videos (learning order) | What actually happens |
|-------|-------------------------|------------------------|
| **Intro + problem formulation** | Intro, Lec 01–02 | Probability space triplet; files as vectors; GenAI = estimate $p_x$ + sample. |
| **Tooling & fundamentals** | Tut 1–6 | Python, NumPy, PyTorch, CNN, RNN, transfer learning. |
| **Probability & ML review** | Tut 7–10 | Triplet / RV / CDF / joints; Gaussian MLE, two-exponential EM algorithm. |
| **Divergence & VDM** | Lec 03–04, Tut 11 | $f$-divergence properties; VDM min–max saddle via convex conjugate $f^\star$. |
| **Adversarial (GAN / WGAN)** | Lec 05–07, Tut 12–14 | JS GAN $\rightarrow$ PyTorch GANs $\rightarrow$ WGAN 1-Lipschitz $\rightarrow$ weight clipping vs GP $\rightarrow$ inversion & FID. |
| **Variational (VAE / VQ-VAE)** | Lec 08–12, Tut 15–16 | Latent variables $\rightarrow$ ELBO $\rightarrow$ LOTUS reparam $\rightarrow$ Gaussian VAE $\rightarrow$ $\beta$-VAE $\rightarrow$ VQ-VAE discrete codes. |
| **Diffusion (DDPM Engine)** | Lec 13–15, 17–18, Tut 17 | Forward Gaussian Markov chain $\rightarrow$ closed form $q(x_t\|x_0)$ $\rightarrow$ VLB derivation $\rightarrow$ $\epsilon_\theta$ objective $\rightarrow$ ancestral sampling $\rightarrow$ PyTorch UNet. |
| **AI Pedagogy** | Lec 16 | Seminar on learning, intelligence, education transformation in the GenAI era. |

---

## Study packages status and catalog (all 36 videos)

The repository packages follow the **7-Pillar Study Package standard** (`PREREQUISITES.md`, `NOTES.md`, `references.md`, `examples/*.py`, `glossary.md`, `formulae_sheet.md`, `quiz.html`).

| Learn # | Video Title | YouTube Slot | Study Package Folder | Status |
|:-------:|:------------|:------------:|:---------------------|:------:|
| 1 | Mathematical Foundations of Generative AI (Intro) | YT #36 | `[Planned: 01-Intro]` | Roadmap |
| 2 | Lec 01 Introduction | YT #35 | [`14-Lec01-MFGAI-Introduction/`](./14-Lec01-MFGAI-Introduction/) | Completed |
| 3 | Tutorial 1 : Introduction to Python Basics | YT #34 | `[Planned: 13B-Tutorial01]` | Roadmap |
| 4 | Tutorial 2 : Introduction to Numpy | YT #33 | [`16-Tutorial02-Introduction-to-NumPy/`](./16-Tutorial02-Introduction-to-NumPy/) | Completed |
| 5 | Lec 02 Generative Models : Problem Formulation | YT #32 | [`15-Lec02-Generative-Models-Problem-Formulation/`](./15-Lec02-Generative-Models-Problem-Formulation/) | Completed |
| 6 | Tutorial 3 : PyTorch Basics | YT #31 | [`17-Tutorial03-PyTorch-Basics/`](./17-Tutorial03-PyTorch-Basics/) | Completed |
| 7 | Tutorial 4 : CNNs using PyTorch | YT #30 | [`18-Tutorial04-CNNs-PyTorch/`](./18-Tutorial04-CNNs-PyTorch/) | Completed |
| 8 | Tutorial 5 : RNNs using PyTorch | YT #29 | [`19-Tutorial05-RNNs-PyTorch/`](./19-Tutorial05-RNNs-PyTorch/) | Completed |
| 9 | Tutorial 6 : Transfer Learning with PyTorch | YT #28 | [`20-Tutorial06-Transfer-Learning-PyTorch/`](./20-Tutorial06-Transfer-Learning-PyTorch/) | Completed |
| 10 | Tutorial 7 : Review of Basic Probability 1 | YT #27 | [`21-Tutorial07-Review-Basic-Probability-1/`](./21-Tutorial07-Review-Basic-Probability-1/) | Completed |
| 11 | Tutorial 8 : Review of Basic Probability 2 | YT #26 | [`22-Tutorial08-Review-Basic-Probability-2/`](./22-Tutorial08-Review-Basic-Probability-2/) | Completed |
| 12 | Tutorial 9 : Review of Basic Probability 3 | YT #25 | [`23-Tutorial09-Review-Basic-Probability-3/`](./23-Tutorial09-Review-Basic-Probability-3/) | Completed |
| 13 | Tutorial 10 : Review of Machine Learning 1 | YT #24 | [`24-Tutorial10-Review-Machine-Learning-1/`](./24-Tutorial10-Review-Machine-Learning-1/) | Completed |
| 14 | Lec 03 f-Divergence and Examples | YT #23 | [`25-Lec03-f-Divergence-Examples/`](./25-Lec03-f-Divergence-Examples/) | Completed |
| 15 | Tutorial 11 – f-Divergence and Examples | YT #22 | [`26-Tutorial11-f-Divergence-Examples/`](./26-Tutorial11-f-Divergence-Examples/) | Completed |
| 16 | Lec 04 Variational Divergence Minimization (VDM) | YT #21 | [`27-Lec04-Variational-Divergence-Minimization/`](./27-Lec04-Variational-Divergence-Minimization/) | Completed |
| 17 | Lec 05 Generative Adversarial Networks (GANs) | YT #20 | [`28-Lec05-Generative-Adversarial-Networks/`](./28-Lec05-Generative-Adversarial-Networks/) | Completed |
| 18 | Tutorial 12 : Implementations of Vanilla GAN, DCGAN and Conditional GAN | YT #19 | [`29-Tutorial12-Implementations-Vanilla-GAN-DCGAN-cGAN/`](./29-Tutorial12-Implementations-Vanilla-GAN-DCGAN-cGAN/) | Completed |
| 19 | Lec 06 Wasserstein GAN (WGAN) | YT #18 | [`30-Lec18-Wasserstein-GAN/`](./30-Lec18-Wasserstein-GAN/) | Completed |
| 20 | Lec 07 Inversion with GANs and FID | YT #17 | [`31-Lec19-Inversion-GANs-FID/`](./31-Lec19-Inversion-GANs-FID/) | Completed |
| 21 | Lec 08 Latent Variable Models and Introduction to Variational Autoencoder (VAE) | YT #16 | [`32-Lec20-Latent-Variable-Models-VAE/`](./32-Lec20-Latent-Variable-Models-VAE/) | Completed |
| 22 | Lec 09 VAEs Part 1 | YT #15 | [`33-Lec09-VAEs-Part1/`](./33-Lec09-VAEs-Part1/) | Completed |
| 23 | Lec 10 VAEs Part 2 | YT #14 | [`34-Lec10-VAEs-Part2/`](./34-Lec10-VAEs-Part2/) | Completed |
| 24 | Lec 11 Beta- VAE | YT #13 | [`35-Lec11-Beta-VAE/`](./35-Lec11-Beta-VAE/) | Completed |
| 25 | Tutorial 15 : VAE and Beta-VAE Implementation | YT #6 | [`39-Tutorial15-VAE-Beta-VAE-Implementation/`](./39-Tutorial15-VAE-Beta-VAE-Implementation/) | Completed |
| 26 | Lec 12 Vector Quantised VAE | YT #12 | [`36-Lec12-Vector-Quantised-VAE/`](./36-Lec12-Vector-Quantised-VAE/) | Completed |
| 27 | Tutorial 16 : Implementation of VQ-VAE | YT #2 | [`40-Tutorial16-VQ-VAE-Implementation/`](./40-Tutorial16-VQ-VAE-Implementation/) | Completed |
| 28 | Tutorial 13 : Wasserstein GAN (WGAN) Implementation using Gradient Clip | YT #8 | [`37-Tutorial13-WGAN-Weight-Clipping/`](./37-Tutorial13-WGAN-Weight-Clipping/) | Completed |
| 29 | Tutorial 14 : Wasserstein GAN (WGAN) Implementation using Gradient Penalty | YT #7 | [`38-Tutorial14-WGAN-Gradient-Penalty/`](./38-Tutorial14-WGAN-Gradient-Penalty/) | Completed |
| 30 | Lec 13 Introdution to Diffusion models | YT #11 | [`41-Lec13-Introduction-to-Diffusion-Models/`](./41-Lec13-Introduction-to-Diffusion-Models/) | Completed |
| 31 | Lec 14 Diffussion Models - Part 1 | YT #10 | [`42-Lec14-Diffusion-Models-Part1/`](./42-Lec14-Diffusion-Models-Part1/) | Completed |
| 32 | Lec 15 Diffussion Models - Part 2 | YT #9 | [`43-Lec15-Diffusion-Models-Part2/`](./43-Lec15-Diffusion-Models-Part2/) | Completed |
| 33 | Lec 17 Diffussion Models - Part 3 | YT #4 | [`44-Lec17-Diffusion-Models-Part3/`](./44-Lec17-Diffusion-Models-Part3/) | Completed |
| 34 | Lec 18 Diffussion Models - Part 4 | YT #3 | [`45-Lec18-Diffusion-Models-Part4/`](./45-Lec18-Diffusion-Models-Part4/) | Completed |
| 35 | Tutorial 17 : Implementation overview of Diffusion Models | YT #1 | [`46-Tutorial17-Diffusion-Models-Implementation-Overview/`](./46-Tutorial17-Diffusion-Models-Implementation-Overview/) | Completed |
| 36 | Lec 16 Pedagogy in the Times of AI | YT #5 | [`47-Lec16-Pedagogy-in-the-Times-of-AI/`](./47-Lec16-Pedagogy-in-the-Times-of-AI/) | Completed |

---

## Playlist catalog (learning order)

Links keep the playlist ID. `index` is the **YouTube slot** (newest-first list), not the learning number.

### Intro, Lec 01–02, Python / NumPy

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 1 | 36 | Mathematical Foundations of Generative AI (Intro) | 3:29 | [watch](https://www.youtube.com/watch?v=tXduOwQ36X0&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=36) | Trailer: probabilistic GenAI — estimate a law and sample; sequel to MFML course. | Roadmap |
| 2 | 35 | Lec 01 Introduction | 1:10:53 | [watch](https://www.youtube.com/watch?v=H05WDy9Mngk&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=35) | RE $\rightarrow$ $\Omega$ $\rightarrow$ $P$ $\rightarrow$ RV $\rightarrow$ estimate $P_X$. Course roadmap (GAN, VAE, diffusion, AR, LLM). | [14](./14-Lec01-MFGAI-Introduction/) |
| 3 | 34 | Tutorial 1 : Introduction to Python Basics | 34:25 | [watch](https://www.youtube.com/watch?v=HvLLR_PScio&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=34) | Python warm-up before NumPy / PyTorch. | Roadmap |
| 4 | 33 | Tutorial 2 : Introduction to Numpy | 1:09:18 | [watch](https://www.youtube.com/watch?v=E79ld44pfGM&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=33) | Arrays $\rightarrow$ matmul $\rightarrow$ ReLU/softmax $\rightarrow$ conv/RNN sketches $\rightarrow$ logistic regression. | [16](./16-Tutorial02-Introduction-to-NumPy/) |
| 5 | 32 | Lec 02 Generative Models : Problem Formulation | 1:03:59 | [watch](https://www.youtube.com/watch?v=GKfv4l6r7hQ&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=32) | Files $\in \mathbb R^d$; dataset $\sim p_x$; GenAI = estimate $p_x$ **and** sample. Recipe: $p_\theta$, $d$, train. | [15](./15-Lec02-Generative-Models-Problem-Formulation/) |

### PyTorch, CNN, RNN, transfer

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 6 | 31 | Tutorial 3 : PyTorch Basics | 1:02:09 | [watch](https://www.youtube.com/watch?v=SEtu7Eef5ps&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=31) | Tensors, device, autograd, `Module`, `DataLoader`, MLP train loop. | [17](./17-Tutorial03-PyTorch-Basics/) |
| 7 | 30 | Tutorial 4 : CNNs using PyTorch | 39:30 | [watch](https://www.youtube.com/watch?v=BhnGtsMwUCU&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=30) | `Conv2d`, max-pool, SimpleCNN, MNIST train/eval. | [18](./18-Tutorial04-CNNs-PyTorch/) |
| 8 | 29 | Tutorial 5 : RNNs using PyTorch | 38:16 | [watch](https://www.youtube.com/watch?v=k6zF2NsvVrk&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=29) | Sequence tensors, RNN/LSTM/GRU, save/load. | [19](./19-Tutorial05-RNNs-PyTorch/) |
| 9 | 28 | Tutorial 6 : Transfer Learning with PyTorch | 29:29 | [watch](https://www.youtube.com/watch?v=ETJG9mmeL5k&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=28) | Pretrained AlexNet/VGG/ResNet; swap the head; fine-tune. | [20](./20-Tutorial06-Transfer-Learning-PyTorch/) |

### Probability and ML review tutorials

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 10 | 27 | Tutorial 7 : Review of Basic Probability 1 | 50:07 | [watch](https://www.youtube.com/watch?v=owlWCCgYx50&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=27) | Triplet, conditional/Bayes, independence, RV/CDF, discrete PMF families. | [21](./21-Tutorial07-Review-Basic-Probability-1/) |
| 11 | 26 | Tutorial 8 : Review of Basic Probability 2 | 57:13 | [watch](https://www.youtube.com/watch?v=pQIbfyjSnFk&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=26) | Continuous RV/PDF, expectation/LOTUS/var, Markov/Chebyshev/Jensen, numpy samples. | [22](./22-Tutorial08-Review-Basic-Probability-2/) |
| 12 | 25 | Tutorial 9 : Review of Basic Probability 3 | 1:13:24 | [watch](https://www.youtube.com/watch?v=eDSb3yObtB8&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=25) | Joints, marginals, conditionals, mixed/GMM, IID, Jacobian. | [23](./23-Tutorial09-Review-Basic-Probability-3/) |
| 13 | 24 | Tutorial 10 : Review of Machine Learning 1 | 47:34 | [watch](https://www.youtube.com/watch?v=wjSKM1xFoSU&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=24) | Sign-censored Normal MLE; two-exponential EM; $Q$ and a closed M-step. | [24](./24-Tutorial10-Review-Machine-Learning-1/) |

### f-divergence and VDM

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 14 | 23 | Lec 03 f-Divergence and Examples | 43:00 | [watch](https://www.youtube.com/watch?v=LR9UQXY_IU8&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=23) | Estimate + sample via $G_\theta$. $f$-div (not a metric). KL / reverse-KL / JSD; modes vs junk. | [25](./25-Lec03-f-Divergence-Examples/) |
| 15 | 22 | Tutorial 11 – f-Divergence and Examples | 48:09 | [watch](https://www.youtube.com/watch?v=GjxuVZeMSfE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=22) | $P\ll Q$; Jensen proofs; KL / $-\log$ / TV / JSD. KL fails symmetry and triangle. | [26](./26-Tutorial11-f-Divergence-Examples/) |
| 16 | 21 | Lec 04 Variational Divergence Minimization (VDM) | 58:54 | [watch](https://www.youtube.com/watch?v=4vtL3NhCkgg&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=21) | Two clouds; conjugate $f^\star$; critic $T(x)$ lower bound; $\min_\theta\max_w$ saddle. | [27](./27-Lec04-Variational-Divergence-Minimization/) |

### GANs, WGAN, inversion

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 17 | 20 | Lec 05 Generative Adversarial Networks (GANs) | 58:04 | [watch](https://www.youtube.com/watch?v=5uqga82bDNA&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=20) | GAN as VDM with a JS-style $f$. Generator $G(z)$, discriminator as density-ratio. | [28](./28-Lec05-Generative-Adversarial-Networks/) |
| 18 | 19 | Tutorial 12 : Implementations of Vanilla GAN, DCGAN and Conditional GAN | 1:18:34 | [watch](https://www.youtube.com/watch?v=dBcURX7GrwE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=19) | Three GAN families in PyTorch: vanilla, conv (DCGAN), class-conditional. | [29](./29-Tutorial12-Implementations-Vanilla-GAN-DCGAN-cGAN/) |
| 19 | 18 | Lec 06 Wasserstein GAN (WGAN) | 44:48 | [watch](https://www.youtube.com/watch?v=1neDqqgaXhE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=18) | Earth-mover / $W_1$; 1-Lipschitz critic; why $f$-div saturates and WGAN does not. *(Formerly Lec 18)* | [30](./30-Lec18-Wasserstein-GAN/) |
| 20 | 17 | Lec 07 Inversion with GANs and FID | 28:12 | [watch](https://www.youtube.com/watch?v=zw2DUzD0TLE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=17) | Find $z$ with $G(z)\approx x$ via BiGAN/ALI tuples. FID = $W_2$ of Inception Gaussians. *(Formerly Lec 19)* | [31](./31-Lec19-Inversion-GANs-FID/) |

### WGAN PyTorch implementations

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 21 | 8 | Tutorial 13 : Wasserstein GAN (WGAN) Implementation using Gradient Clip | 32:54 | [watch](https://www.youtube.com/watch?v=p0wXSsTnmw0&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=8) | WGAN in PyTorch: clamping weights in $[-c, c]$ to satisfy 1-Lipschitz condition; critic loss. | [37](./37-Tutorial13-WGAN-Weight-Clipping/) |
| 22 | 7 | Tutorial 14 : Wasserstein GAN (WGAN) Implementation using Gradient Penalty | 20:43 | [watch](https://www.youtube.com/watch?v=Mxad7Wz7ymg&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=7) | WGAN-GP in PyTorch: interpolating samples $\hat{x}$, penalizing $(\|\nabla_{\hat{x}} D\|_2 - 1)^2$, eliminating clipping pathology. | [38](./38-Tutorial14-WGAN-Gradient-Penalty/) |

### Latent-variable models, VAE, beta-VAE, VQ-VAE

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 23 | 16 | Lec 08 Latent Variable Models and Introduction to Variational Autoencoder (VAE) | 55:25 | [watch](https://www.youtube.com/watch?v=4djE9goJtKs&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=16) | Hidden $z$; incomplete likelihood; ELBO derivation; EM vs VAE when posterior is intractable. *(Formerly Lec 20)* | [32](./32-Lec20-Latent-Variable-Models-VAE/) |
| 24 | 15 | Lec 09 VAEs Part 1 | 32:56 | [watch](https://www.youtube.com/watch?v=KHiRfCIpJkI&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=15) | Boxed ELBO $\mathbb{E}_Q[\log P(X\|Z)]-KL$; encoder/decoder; reparam trick via LOTUS. | [33](./33-Lec09-VAEs-Part1/) |
| 25 | 14 | Lec 10 VAEs Part 2 | 45:07 | [watch](https://www.youtube.com/watch?v=plrfGKsZihg&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=14) | Gaussian likelihood, analytic Gaussian KL term $\frac{1}{2}\sum(\sigma_j^2+\mu_j^2-1-\log\sigma_j^2)$, full training algorithm. | [34](./34-Lec10-VAEs-Part2/) |
| 26 | 13 | Lec 11 Beta- VAE | 23:47 | [watch](https://www.youtube.com/watch?v=I89atful2qg&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=13) | Adding weight $\beta$ to KL term; constrained optimization Lagrangian; learning disentangled features vs reconstruction tradeoff. | [35](./35-Lec11-Beta-VAE/) |
| 27 | 6 | Tutorial 15 : VAE and Beta-VAE Implementation | 41:27 | [watch](https://www.youtube.com/watch?v=f_X2vwIXVz4&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=6) | PyTorch implementation of VAE and $\beta$-VAE on MNIST/CIFAR: encoder $(\mu,\sigma)$, sampling node, reconstruction loss, latent traversals. | [39](./39-Tutorial15-VAE-Beta-VAE-Implementation/) |
| 28 | 12 | Lec 12 Vector Quantised VAE | 18:06 | [watch](https://www.youtube.com/watch?v=inowo4EMmkA&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=12) | VQ-VAE: discrete latent space, codebook vector quantization, dictionary learning, Straight-Through Estimator (STE). | [36](./36-Lec12-Vector-Quantised-VAE/) |
| 29 | 2 | Tutorial 16 : Implementation of VQ-VAE | 58:35 | [watch](https://www.youtube.com/watch?v=NZQzEYuok_c&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=2) | PyTorch code for VQ-VAE: discrete codebook `nn.Embedding`, nearest neighbor assignment, straight-through gradient copy, commitment loss. | [40](./40-Tutorial16-VQ-VAE-Implementation/) |

### Diffusion Models (DDPM Engine)

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 30 | 11 | Lec 13 Introdution to Diffusion models | 30:37 | [watch](https://www.youtube.com/watch?v=DrLmFcXxhKY&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=11) | Physical motivation from non-equilibrium thermodynamics; forward noising process vs reverse generative denoising process. | [41](./41-Lec13-Introduction-to-Diffusion-Models/) |
| 31 | 10 | Lec 14 Diffussion Models - Part 1 | 35:09 | [watch](https://www.youtube.com/watch?v=Q6EUgNEPwRk&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=10) | DDPM formulation: Gaussian transition kernel $q(x_t\|x_{t-1})$; closed-form forward marginal $q(x_t\|x_0)=\mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$. | [42](./42-Lec14-Diffusion-Models-Part1/) |
| 32 | 9 | Lec 15 Diffussion Models - Part 2 | 25:32 | [watch](https://www.youtube.com/watch?v=O-Jzyk_JjME&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=9) | Variational Lower Bound (VLB) expansion: splitting into prior matching $L_T$, denoising matching $L_{t-1}$, and reconstruction $L_0$. Bayes posterior $q(x_{t-1}\|x_t,x_0)$. | [43](./43-Lec15-Diffusion-Models-Part2/) |
| 33 | 4 | Lec 17 Diffussion Models - Part 3 | 29:28 | [watch](https://www.youtube.com/watch?v=hf5E8k_Pdh0&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=4) | Connection to score matching; reparameterizing $\mu_\theta(x_t, t)$ into noise prediction $\epsilon_\theta(x_t, t)$; Ho et al. simplified loss $L_{\text{simple}}$. | [44](./44-Lec17-Diffusion-Models-Part3/) |
| 34 | 3 | Lec 18 Diffussion Models - Part 4 | 32:17 | [watch](https://www.youtube.com/watch?v=1G3kzIE44aQ&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=3) | DDPM Ancestral Sampling algorithm; noise schedules (linear $\beta_t$, cosine); conditionings and classifier-free guidance foundations. | [45](./45-Lec18-Diffusion-Models-Part4/) |
| 35 | 1 | Tutorial 17 : Implementation overview of Diffusion Models | 38:32 | [watch](https://www.youtube.com/watch?v=Jw01N9Efubw&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=1) | Full PyTorch DDPM implementation overview: UNet backbone, sinusoidal timestep embedding, forward noise schedule, training loop, ancestral sampling loop. | [46](./46-Tutorial17-Diffusion-Models-Implementation-Overview/) |

### Pedagogy in the Times of AI

| Learn | YT # | Video | Duration | Link | Summary | Package |
|:-----:|:----:|:------|:--------:|:----:|:--------|:-------:|
| 36 | 5 | Lec 16 Pedagogy in the Times of AI | 56:27 | [watch](https://www.youtube.com/watch?v=N2a1J0UPeL4&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=5) | Special lecture by Prof. Prathosh: pedagogy, learning versus generation, cognition, curriculum evolution in the generative AI era. | [47](./47-Lec16-Pedagogy-in-the-Times-of-AI/) |

---

## YouTube playlist order (newest first)

This is the exact order YouTube shows on the playlist page (`index=1` through `index=36`):

| YT # | Video Title | Duration | Video ID |
|:----:|:------------|:--------:|:--------:|
| 1 | Tutorial 17 : Implementation overview of Diffusion Models | 38:32 | `Jw01N9Efubw` |
| 2 | Tutorial 16 : Implementation of VQ-VAE | 58:35 | `NZQzEYuok_c` |
| 3 | Lec 18 Diffussion Models - Part 4 | 32:17 | `1G3kzIE44aQ` |
| 4 | Lec 17 Diffussion Models - Part 3 | 29:28 | `hf5E8k_Pdh0` |
| 5 | Lec 16 Pedagogy in the Times of AI | 56:27 | `N2a1J0UPeL4` |
| 6 | Tutorial 15 : VAE and Beta-VAE Implementation | 41:27 | `f_X2vwIXVz4` |
| 7 | Tutorial 14 : Wasserstein GAN (WGAN) Implementation using Gradient Penalty | 20:43 | `Mxad7Wz7ymg` |
| 8 | Tutorial 13 : Wasserstein GAN (WGAN) Implementation using Gradient Clip | 32:54 | `p0wXSsTnmw0` |
| 9 | Lec 15 Diffussion Models - Part 2 | 25:32 | `O-Jzyk_JjME` |
| 10 | Lec 14 Diffussion Models - Part 1 | 35:09 | `Q6EUgNEPwRk` |
| 11 | Lec 13 Introdution to Diffusion models | 30:37 | `DrLmFcXxhKY` |
| 12 | Lec 12 Vector Quantised VAE | 18:06 | `inowo4EMmkA` |
| 13 | Lec 11 Beta- VAE | 23:47 | `I89atful2qg` |
| 14 | Lec 10 VAEs Part 2 | 45:07 | `plrfGKsZihg` |
| 15 | Lec 09 VAEs Part 1 | 32:56 | `KHiRfCIpJkI` |
| 16 | Lec 08 Latent Variable Models and Introduction to Variational Autoencoder (VAE) | 55:25 | `4djE9goJtKs` |
| 17 | Lec 07 Inversion with GANs and FID | 28:12 | `zw2DUzD0TLE` |
| 18 | Lec 06 Wasserstein GAN (WGAN) | 44:48 | `1neDqqgaXhE` |
| 19 | Tutorial 12 : Implementations of Vanilla GAN, DCGAN and Conditional GAN | 1:18:34 | `dBcURX7GrwE` |
| 20 | Lec 05 Generative Adversarial Networks (GANs) | 58:04 | `5uqga82bDNA` |
| 21 | Lec 04 Variational Divergence Minimization (VDM) | 58:54 | `4vtL3NhCkgg` |
| 22 | Tutorial 11 – f-Divergence and Examples | 48:09 | `GjxuVZeMSfE` |
| 23 | Lec 03 f-Divergence and Examples | 43:00 | `LR9UQXY_IU8` |
| 24 | Tutorial 10 : Review of Machine Learning 1 | 47:34 | `wjSKM1xFoSU` |
| 25 | Tutorial 9 : Review of Basic Probability 3 | 1:13:24 | `eDSb3yObtB8` |
| 26 | Tutorial 8 : Review of Basic Probability 2 | 57:13 | `pQIbfyjSnFk` |
| 27 | Tutorial 7 : Review of Basic Probability 1 | 50:07 | `owlWCCgYx50` |
| 28 | Tutorial 6 : Transfer Learning with PyTorch | 29:29 | `ETJG9mmeL5k` |
| 29 | Tutorial 5 : RNNs using PyTorch | 38:16 | `k6zF2NsvVrk` |
| 30 | Tutorial 4 : CNNs using PyTorch | 39:30 | `BhnGtsMwUCU` |
| 31 | Tutorial 3 : PyTorch Basics | 1:02:09 | `SEtu7Eef5ps` |
| 32 | Lec 02 Generative Models : Problem Formulation | 1:03:59 | `GKfv4l6r7hQ` |
| 33 | Tutorial 2 : Introduction to Numpy | 1:09:18 | `E79ld44pfGM` |
| 34 | Tutorial 1 : Introduction to Python Basics | 34:25 | `HvLLR_PScio` |
| 35 | Lec 01 Introduction | 1:10:53 | `H05WDy9Mngk` |
| 36 | Mathematical Foundations of Generative AI (Intro) | 3:29 | `tXduOwQ36X0` |

---

## Compact title → URL list

In pedagogical sequence (learning order). URL links to the exact YouTube index.

1. Mathematical Foundations of Generative AI (Intro) → https://www.youtube.com/watch?v=tXduOwQ36X0&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=36
2. Lec 01 Introduction → https://www.youtube.com/watch?v=H05WDy9Mngk&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=35
3. Tutorial 1 : Introduction to Python Basics → https://www.youtube.com/watch?v=HvLLR_PScio&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=34
4. Tutorial 2 : Introduction to Numpy → https://www.youtube.com/watch?v=E79ld44pfGM&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=33
5. Lec 02 Generative Models : Problem Formulation → https://www.youtube.com/watch?v=GKfv4l6r7hQ&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=32
6. Tutorial 3 : PyTorch Basics → https://www.youtube.com/watch?v=SEtu7Eef5ps&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=31
7. Tutorial 4 : CNNs using PyTorch → https://www.youtube.com/watch?v=BhnGtsMwUCU&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=30
8. Tutorial 5 : RNNs using PyTorch → https://www.youtube.com/watch?v=k6zF2NsvVrk&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=29
9. Tutorial 6 : Transfer Learning with PyTorch → https://www.youtube.com/watch?v=ETJG9mmeL5k&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=28
10. Tutorial 7 : Review of Basic Probability 1 → https://www.youtube.com/watch?v=owlWCCgYx50&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=27
11. Tutorial 8 : Review of Basic Probability 2 → https://www.youtube.com/watch?v=pQIbfyjSnFk&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=26
12. Tutorial 9 : Review of Basic Probability 3 → https://www.youtube.com/watch?v=eDSb3yObtB8&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=25
13. Tutorial 10 : Review of Machine Learning 1 → https://www.youtube.com/watch?v=wjSKM1xFoSU&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=24
14. Lec 03 f-Divergence and Examples → https://www.youtube.com/watch?v=LR9UQXY_IU8&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=23
15. Tutorial 11 – f-Divergence and Examples → https://www.youtube.com/watch?v=GjxuVZeMSfE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=22
16. Lec 04 Variational Divergence Minimization (VDM) → https://www.youtube.com/watch?v=4vtL3NhCkgg&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=21
17. Lec 05 Generative Adversarial Networks (GANs) → https://www.youtube.com/watch?v=5uqga82bDNA&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=20
18. Tutorial 12 : Implementations of Vanilla GAN, DCGAN and Conditional GAN → https://www.youtube.com/watch?v=dBcURX7GrwE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=19
19. Lec 06 Wasserstein GAN (WGAN) → https://www.youtube.com/watch?v=1neDqqgaXhE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=18
20. Lec 07 Inversion with GANs and FID → https://www.youtube.com/watch?v=zw2DUzD0TLE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=17
21. Tutorial 13 : Wasserstein GAN (WGAN) Implementation using Gradient Clip → https://www.youtube.com/watch?v=p0wXSsTnmw0&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=8
22. Tutorial 14 : Wasserstein GAN (WGAN) Implementation using Gradient Penalty → https://www.youtube.com/watch?v=Mxad7Wz7ymg&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=7
23. Lec 08 Latent Variable Models and Introduction to Variational Autoencoder (VAE) → https://www.youtube.com/watch?v=4djE9goJtKs&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=16
24. Lec 09 VAEs Part 1 → https://www.youtube.com/watch?v=KHiRfCIpJkI&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=15
25. Lec 10 VAEs Part 2 → https://www.youtube.com/watch?v=plrfGKsZihg&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=14
26. Lec 11 Beta- VAE → https://www.youtube.com/watch?v=I89atful2qg&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=13
27. Tutorial 15 : VAE and Beta-VAE Implementation → https://www.youtube.com/watch?v=f_X2vwIXVz4&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=6
28. Lec 12 Vector Quantised VAE → https://www.youtube.com/watch?v=inowo4EMmkA&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=12
29. Tutorial 16 : Implementation of VQ-VAE → https://www.youtube.com/watch?v=NZQzEYuok_c&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=2
30. Lec 13 Introdution to Diffusion models → https://www.youtube.com/watch?v=DrLmFcXxhKY&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=11
31. Lec 14 Diffussion Models - Part 1 → https://www.youtube.com/watch?v=Q6EUgNEPwRk&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=10
32. Lec 15 Diffussion Models - Part 2 → https://www.youtube.com/watch?v=O-Jzyk_JjME&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=9
33. Lec 17 Diffussion Models - Part 3 → https://www.youtube.com/watch?v=hf5E8k_Pdh0&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=4
34. Lec 18 Diffussion Models - Part 4 → https://www.youtube.com/watch?v=1G3kzIE44aQ&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=3
35. Tutorial 17 : Implementation overview of Diffusion Models → https://www.youtube.com/watch?v=Jw01N9Efubw&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=1
36. Lec 16 Pedagogy in the Times of AI → https://www.youtube.com/watch?v=N2a1J0UPeL4&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=5

---

## External resources

| Resource | URL |
|:---------|:----|
| YouTube playlist (official NPTEL) | https://www.youtube.com/playlist?list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK |
| NPTEL course page | https://nptel.ac.in/courses/106108004 |
| Swayam preview (noc26_cs97) | https://onlinecourses.nptel.ac.in/noc26_cs97/preview |
| Parent — MFML catalog | [`../NOTES.md`](../NOTES.md) |
| Sibling — IITM BS GenAI catalog (full 73 videos) | [`../IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md`](../IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md) |
| IITM PyTorch notebooks (same instructor, other recording) | https://github.com/Chandan-IISc/IITM_GenAI |

---

## Sources

- [NPTEL playlist](https://www.youtube.com/playlist?list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK) (Live audit: 36 videos, IDs, exact durations, chronological & newest-first order)
- [NPTEL 106108004](https://nptel.ac.in/courses/106108004) (course syllabus & abstract)
- Lecture packages in this folder for Lec 01–09, Tutorials 02–12, and WGAN/Inversion packages
- IITM BS 12-week map for curriculum alignment across the generative modeling paradigm
