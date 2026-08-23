# Mathematical Foundations of Generative AI

NPTEL / IISc Bangalore · Course **106108004** · noc26-cs97 · Prof. Prathosh A P

**Playlist:** [Mathematical Foundations of Generative AI](https://www.youtube.com/playlist?list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK)  
**Channel:** NPTEL — Indian Institute of Science, Bengaluru  
**Size (uploaded so far):** 21 videos · ~17.6 hours · still growing (YouTube currently lists **newest first**)

This file is the **course map + catalog** for the NPTEL recording. Per-lecture study packages live in the numbered folders beside this file and are linked below when they exist.

Same instructor and same math as the IIT Madras BS recording. **Different videos.** Full 12-week IITM catalog: [`../IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md`](../IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md)

This is the **sequel** to Mathematical Foundations of Machine Learning: [`../NOTES.md`](../NOTES.md)

---

## Table of Contents

1. [What the course is for](#what-the-course-is-for)
2. [Architecture of the whole course](#architecture-of-the-whole-course)
3. [How the uploaded blocks fit](#how-the-uploaded-blocks-fit)
4. [Study packages already in this folder](#study-packages-already-in-this-folder)
5. [Playlist catalog (learning order)](#playlist-catalog-learning-order)
   - [Intro, Lec 01–02, Python / NumPy](#intro-lec-0102-python--numpy)
   - [PyTorch, CNN, RNN, transfer](#pytorch-cnn-rnn-transfer)
   - [Probability and ML review tutorials](#probability-and-ml-review-tutorials)
   - [f-divergence and VDM](#f-divergence-and-vdm)
   - [GANs, WGAN, inversion, VAE](#gans-wgan-inversion-vae)
6. [YouTube playlist order (newest first)](#youtube-playlist-order-newest-first)
7. [Compact title → URL list](#compact-title--url-list)
8. [External resources](#external-resources)
9. [Sources](#sources)

---

## What the course is for

Generative AI is treated as a **probability problem**: given files (images, tokens, spectrograms) treated as draws from unknown $p_x$, you must

1. **Estimate** that law (or a surrogate good enough to sample from).
2. **Sample** a brand-new file that was never in the pile.

NPTEL abstract: deep generative models — VAEs, GANs, autoregressive models, diffusion, LLMs — with proofs and PyTorch implementations.

`Lec` = chalk-and-talk math. `Tutorial` = NumPy / PyTorch / worked probability.

The playlist on YouTube is **newest-upload first**. Tables below are in **learning order** (Intro → Lec 01 → …). `&index=` in each URL is the actual YouTube playlist slot.

Lec 06–17 are **not on this playlist yet** (the numbering jumps from Lec 05 to Lec 18). Treat those as not uploaded, not skipped on purpose.

---

## Architecture of the whole course

**Worldview arc:** from the MFML recipe “estimate $p$ then decide” **to** “estimate $p_x$ **and sample**” **to** four sampling engines (adversarial, variational, diffusion, autoregressive) **to** aligning an LM. This NPTEL playlist currently reaches the start of the VAE engine.

One recipe, reused every week:

```
  Dataset D = {x_1, …, x_n}  ⊂  R^d ,   x_i ~ p_x  (unknown)

                    │
                    ▼
         pick a generator family
         (explicit p_θ  or  implicit G_θ(z))
                    │
                    ▼
         pick a discrepancy d(p_x, model)
         f-div / JS / Wasserstein / ELBO / score / NLL / …
                    │
                    ▼
         train θ  (min  or  min_θ max_w saddle)
                    │
                    ▼
         SAMPLE  x_new  from the trained object
```

### Uploaded so far vs still coming

```
  MFML parent: FA → p → min KL / risk → nets
              │
              ▼
  Lec 01–02   restated job: estimate p_x  AND  sample
              files ∈ range(X) ⊂ R^d
              │
              ├─ Tut 1–6    Python, NumPy, PyTorch, CNN, RNN, transfer
              ├─ Tut 7–10   probability + ML review (triplet, RV, MLE/EM)
              │
              ├─ Lec 03–04  f-divergence → VDM saddle (critic T vs G_θ)
              ├─ Lec 05     GAN as the JS special case of VDM
              ├─ Tut 12     Vanilla / DC / conditional GAN in code
              │
              │   Lec 06–17  not on the playlist yet
              │
              ├─ Lec 18–19  WGAN; inversion + FID
              └─ Lec 20     latent variables → VAE (this playlist’s current end)

  Still expected (IITM 12-week map): β-VAE / VQ-VAE, DDPM/DDIM,
  AR + Transformer, PPO/DPO, SSMs.
```

### What later lectures cannot skip

| Block | Load-bearing claim | Failure if skipped |
|-------|--------------------|--------------------|
| Lec 01–02 | Data ∈ range$(X)$; GenAI = estimate $p_x$ **and** sample | “Just generate” with no law |
| Lec 03 | $d$ is an $f$-divergence; not a metric | Treating KL/JS as distances |
| Lec 04 | Convex conjugate → critic $T$ lower-bounds $d$ | Writing a GAN loss with no VDM |
| Lec 05 | GAN value function ≡ JS-style VDM | Discriminator as a random extra net |
| Lec 18 | Wasserstein when JS saturates | Mode collapse blamed on “tuning” |
| Lec 20 | $p(x)=\int p(x\|z)p(z)\,dz$; ELBO | VAE as “autoencoder + noise” |

---

## How the uploaded blocks fit

| Block | Videos (learning order) | What actually happens |
|-------|-------------------------|------------------------|
| Intro + job | Intro, Lec 01–02 | Probability triplet recap; files as vectors; estimate $p_x$ + sample. |
| Tooling | Tut 1–6 | Python, NumPy, PyTorch, CNN, RNN, transfer — so later code is not a black box. |
| Review | Tut 7–10 | Triplet / RV / CDF / joints; MLE, EM — the MFML spine used as a warm-up. |
| Divergence | Lec 03–04, Tut 11 | $f$-div examples; VDM min–max via $f^\star$. |
| Implicit generators | Lec 05, Tut 12, Lec 18–19 | GAN → DCGAN/cGAN in code → WGAN → invert $G$ and score with FID. |
| Latent / VAE | Lec 20 | Incomplete likelihood; VAE as amortized variational inference. **Current end.** |

Suggested first watch: **Lec 02** (problem formulation, 64 min), then **Lec 03–04** ($f$-div and VDM). That is the same spine as IITM W1_L2–L4.

---

## Study packages already in this folder

| Learning # | Video | Folder |
|------------|-------|--------|
| 2 | Lec 01 Introduction | [`14-Lec01-MFGAI-Introduction/`](./14-Lec01-MFGAI-Introduction/) |
| 5 | Lec 02 Generative Models : Problem Formulation | [`15-Lec02-Generative-Models-Problem-Formulation/`](./15-Lec02-Generative-Models-Problem-Formulation/) |
| 4 | Tutorial 2 : Introduction to Numpy | [`16-Tutorial02-Introduction-to-NumPy/`](./16-Tutorial02-Introduction-to-NumPy/) |
| 6 | Tutorial 3 : PyTorch Basics | [`17-Tutorial03-PyTorch-Basics/`](./17-Tutorial03-PyTorch-Basics/) |
| 7 | Tutorial 4 : CNNs using PyTorch | [`18-Tutorial04-CNNs-PyTorch/`](./18-Tutorial04-CNNs-PyTorch/) |
| 8 | Tutorial 5 : RNNs using PyTorch | [`19-Tutorial05-RNNs-PyTorch/`](./19-Tutorial05-RNNs-PyTorch/) |
| 9 | Tutorial 6 : Transfer Learning with PyTorch | [`20-Tutorial06-Transfer-Learning-PyTorch/`](./20-Tutorial06-Transfer-Learning-PyTorch/) |
| 10 | Tutorial 7 : Review of Basic Probability 1 | [`21-Tutorial07-Review-Basic-Probability-1/`](./21-Tutorial07-Review-Basic-Probability-1/) |
| 11 | Tutorial 8 : Review of Basic Probability 2 | [`22-Tutorial08-Review-Basic-Probability-2/`](./22-Tutorial08-Review-Basic-Probability-2/) |
| 12 | Tutorial 9 : Review of Basic Probability 3 | [`23-Tutorial09-Review-Basic-Probability-3/`](./23-Tutorial09-Review-Basic-Probability-3/) |
| 13 | Tutorial 10 : Review of Machine Learning 1 | [`24-Tutorial10-Review-Machine-Learning-1/`](./24-Tutorial10-Review-Machine-Learning-1/) |
| 14 | Lec 03 f-Divergence and Examples | [`25-Lec03-f-Divergence-Examples/`](./25-Lec03-f-Divergence-Examples/) |
| 15 | Tutorial 11 – f-Divergence and Examples | [`26-Tutorial11-f-Divergence-Examples/`](./26-Tutorial11-f-Divergence-Examples/) |
| 16 | Lec 04 Variational Divergence Minimization (VDM) | [`27-Lec04-Variational-Divergence-Minimization/`](./27-Lec04-Variational-Divergence-Minimization/) |
| 17 | Lec 05 Generative Adversarial Networks (GANs) | [`28-Lec05-Generative-Adversarial-Networks/`](./28-Lec05-Generative-Adversarial-Networks/) |
| 18 | Tutorial 12 : Vanilla GAN, DCGAN, Conditional GAN | [`29-Tutorial12-Implementations-Vanilla-GAN-DCGAN-cGAN/`](./29-Tutorial12-Implementations-Vanilla-GAN-DCGAN-cGAN/) |
| 19 | Lec 18 Wasserstein GAN (WGAN) | [`30-Lec18-Wasserstein-GAN/`](./30-Lec18-Wasserstein-GAN/) |
| 20 | Lec 19 Inversion with GANs and FID | [`31-Lec19-Inversion-GANs-FID/`](./31-Lec19-Inversion-GANs-FID/) |
| 21 | Lec 20 Latent Variable Models and VAE intro | [`32-Lec20-Latent-Variable-Models-VAE/`](./32-Lec20-Latent-Variable-Models-VAE/) |

No package yet: course Intro, Tutorial 1.

PDFs in [`Notes/`](./Notes/): `NPTEL-Tutorials.pdf`, `Review-Probability.pdf`.

---

## Playlist catalog (learning order)

Links keep the playlist id. `index` is the **YouTube slot** (newest-first list), not the learning number.

### Intro, Lec 01–02, Python / NumPy

| Learn | YT # | Video | Duration | Link | Summary | Package |
|-------|------|--------|----------|------|---------|---------|
| 1 | 21 | Mathematical Foundations of Generative AI (Intro) | 3:29 | [watch](https://www.youtube.com/watch?v=tXduOwQ36X0&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=21) | Trailer: probabilistic GenAI — estimate a law and sample; sequel to the MFML course. | |
| 2 | 20 | Lec 01 Introduction | 1:10:53 | [watch](https://www.youtube.com/watch?v=H05WDy9Mngk&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=20) | RE → Ω → $P$ → RV → estimate $P_X$. Course roadmap (GAN, VAE, diffusion, AR, LLM). | [14](./14-Lec01-MFGAI-Introduction/) |
| 3 | 19 | Tutorial 1 : Introduction to Python Basics | 34:25 | [watch](https://www.youtube.com/watch?v=HvLLR_PScio&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=19) | Python warm-up before NumPy / PyTorch. | |
| 4 | 18 | Tutorial 2 : Introduction to Numpy | 1:09:18 | [watch](https://www.youtube.com/watch?v=E79ld44pfGM&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=18) | Arrays → matmul → ReLU/softmax → conv/RNN sketches → logistic regression. | [16](./16-Tutorial02-Introduction-to-NumPy/) |
| 5 | 17 | Lec 02 Generative Models : Problem Formulation | 1:03:59 | [watch](https://www.youtube.com/watch?v=GKfv4l6r7hQ&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=17) | Files ∈ $\mathbb R^d$; dataset $\sim p_x$; GenAI = estimate $p_x$ **and** sample. Recipe: $p_\theta$, $d$, train. | [15](./15-Lec02-Generative-Models-Problem-Formulation/) |

### PyTorch, CNN, RNN, transfer

| Learn | YT # | Video | Duration | Link | Summary | Package |
|-------|------|--------|----------|------|---------|---------|
| 6 | 16 | Tutorial 3 : PyTorch Basics | 1:02:09 | [watch](https://www.youtube.com/watch?v=SEtu7Eef5ps&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=16) | Tensors, device, autograd, `Module`, `DataLoader`, MLP train loop. | [17](./17-Tutorial03-PyTorch-Basics/) |
| 7 | 15 | Tutorial 4 : CNNs using PyTorch | 39:30 | [watch](https://www.youtube.com/watch?v=BhnGtsMwUCU&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=15) | `Conv2d`, max-pool, SimpleCNN, MNIST train/eval. | [18](./18-Tutorial04-CNNs-PyTorch/) |
| 8 | 14 | Tutorial 5 : RNNs using PyTorch | 38:16 | [watch](https://www.youtube.com/watch?v=k6zF2NsvVrk&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=14) | Sequence tensors, RNN/LSTM/GRU, save/load. | [19](./19-Tutorial05-RNNs-PyTorch/) |
| 9 | 13 | Tutorial 6 : Transfer Learning with PyTorch | 29:29 | [watch](https://www.youtube.com/watch?v=ETJG9mmeL5k&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=13) | Pretrained AlexNet/VGG/ResNet; swap the head; fine-tune. | [20](./20-Tutorial06-Transfer-Learning-PyTorch/) |

### Probability and ML review tutorials

| Learn | YT # | Video | Duration | Link | Summary | Package |
|-------|------|--------|----------|------|---------|---------|
| 10 | 12 | Tutorial 7 : Review of Basic Probability 1 | 50:07 | [watch](https://www.youtube.com/watch?v=owlWCCgYx50&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=12) | Triplet, conditional/Bayes, independence, RV/CDF, discrete PMF families. | [21](./21-Tutorial07-Review-Basic-Probability-1/) |
| 11 | 11 | Tutorial 8 : Review of Basic Probability 2 | 57:13 | [watch](https://www.youtube.com/watch?v=pQIbfyjSnFk&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=11) | Continuous RV/PDF, expectation/LOTUS/var, Markov/Chebyshev/Jensen, numpy samples. | [22](./22-Tutorial08-Review-Basic-Probability-2/) |
| 12 | 10 | Tutorial 9 : Review of Basic Probability 3 | 1:13:24 | [watch](https://www.youtube.com/watch?v=eDSb3yObtB8&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=10) | Joints, marginals, conditionals, mixed/GMM, IID, Jacobian. | [23](./23-Tutorial09-Review-Basic-Probability-3/) |
| 13 | 9 | Tutorial 10 : Review of Machine Learning 1 | 47:34 | [watch](https://www.youtube.com/watch?v=wjSKM1xFoSU&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=9) | Sign-censored Normal MLE; two-exponential EM; $Q$ and a closed M-step. | [24](./24-Tutorial10-Review-Machine-Learning-1/) |

### f-divergence and VDM

| Learn | YT # | Video | Duration | Link | Summary | Package |
|-------|------|--------|----------|------|---------|---------|
| 14 | 8 | Lec 03 f-Divergence and Examples | 43:00 | [watch](https://www.youtube.com/watch?v=LR9UQXY_IU8&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=8) | Estimate + sample via $G_\theta$. $f$-div (not a metric). KL / reverse-KL / JSD; modes vs junk. | [25](./25-Lec03-f-Divergence-Examples/) |
| 15 | 7 | Tutorial 11 – f-Divergence and Examples | 48:09 | [watch](https://www.youtube.com/watch?v=GjxuVZeMSfE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=7) | $P\ll Q$; Jensen proofs; KL / $-\log$ / TV / JSD. KL fails symmetry and triangle. | [26](./26-Tutorial11-f-Divergence-Examples/) |
| 16 | 6 | Lec 04 Variational Divergence Minimization (VDM) | 58:54 | [watch](https://www.youtube.com/watch?v=4vtL3NhCkgg&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=6) | Two clouds; conjugate $f^\star$; critic $T(x)$ lower bound; $\min_\theta\max_w$ saddle. | [27](./27-Lec04-Variational-Divergence-Minimization/) |

### GANs, WGAN, inversion, VAE

| Learn | YT # | Video | Duration | Link | Summary | Package |
|-------|------|--------|----------|------|---------|---------|
| 17 | 5 | Lec 05 Generative Adversarial Networks (GANs) | 58:04 | [watch](https://www.youtube.com/watch?v=5uqga82bDNA&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=5) | GAN as VDM with a JS-style $f$. Generator $G(z)$, discriminator as density-ratio. | [28](./28-Lec05-Generative-Adversarial-Networks/) |
| 18 | 4 | Tutorial 12 : Implementations of Vanilla GAN, DCGAN and Conditional GAN | 1:18:34 | [watch](https://www.youtube.com/watch?v=dBcURX7GrwE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=4) | Three GAN families in PyTorch: vanilla, conv (DCGAN), class-conditional. | [29](./29-Tutorial12-Implementations-Vanilla-GAN-DCGAN-cGAN/) |
| 19 | 3 | Lec 18 Wasserstein GAN (WGAN) | 44:48 | [watch](https://www.youtube.com/watch?v=1neDqqgaXhE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=3) | Earth-mover / $W_2$; 1-Lipschitz critic; why $f$-div saturates and WGAN does not. | [30](./30-Lec18-Wasserstein-GAN/) |
| 20 | 2 | Lec 19 Inversion with GANs and FID | 28:12 | [watch](https://www.youtube.com/watch?v=zw2DUzD0TLE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=2) | Find $z$ with $G(z)\approx x$ via BiGAN/ALI tuples. FID = $W_2$ of Inception Gaussians. | [31](./31-Lec19-Inversion-GANs-FID/) |
| 21 | 1 | Lec 20 Latent Variable Models and Introduction to Variational Autoencoder (VAE) | 55:25 | [watch](https://www.youtube.com/watch?v=4djE9goJtKs&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=1) | Hidden $z$; incomplete likelihood; ELBO; EM vs VAE when posterior intractable. **Current last lecture.** | [32](./32-Lec20-Latent-Variable-Models-VAE/) |

---

## YouTube playlist order (newest first)

This is the order YouTube shows if you press Play All.

| YT # | Video |
|------|--------|
| 1 | Lec 20 Latent Variable Models and Introduction to Variational Autoencoder (VAE) |
| 2 | Lec 19 Inversion with GANs and FID |
| 3 | Lec 18 Wasserstein GAN (WGAN) |
| 4 | Tutorial 12 : Implementations of Vanilla GAN, DCGAN and Conditional GAN |
| 5 | Lec 05 Generative Adversarial Networks (GANs) |
| 6 | Lec 04 Variational Divergence Minimization (VDM) |
| 7 | Tutorial 11 – f-Divergence and Examples |
| 8 | Lec 03 f-Divergence and Examples |
| 9 | Tutorial 10 : Review of Machine Learning 1 |
| 10 | Tutorial 9 : Review of Basic Probability 3 |
| 11 | Tutorial 8 : Review of Basic Probability 2 |
| 12 | Tutorial 7 : Review of Basic Probability 1 |
| 13 | Tutorial 6 : Transfer Learning with PyTorch |
| 14 | Tutorial 5 : RNNs using PyTorch |
| 15 | Tutorial 4 : CNNs using PyTorch |
| 16 | Tutorial 3 : PyTorch Basics |
| 17 | Lec 02 Generative Models : Problem Formulation |
| 18 | Tutorial 2 : Introduction to Numpy |
| 19 | Tutorial 1 : Introduction to Python Basics |
| 20 | Lec 01 Introduction |
| 21 | Mathematical Foundations of Generative AI (Intro) |

---

## Compact title → URL list

Learning order. Index = YouTube slot.

1. Mathematical Foundations of Generative AI (Intro) → https://www.youtube.com/watch?v=tXduOwQ36X0&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=21
2. Lec 01 Introduction → https://www.youtube.com/watch?v=H05WDy9Mngk&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=20
3. Tutorial 1 : Introduction to Python Basics → https://www.youtube.com/watch?v=HvLLR_PScio&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=19
4. Tutorial 2 : Introduction to Numpy → https://www.youtube.com/watch?v=E79ld44pfGM&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=18
5. Lec 02 Generative Models : Problem Formulation → https://www.youtube.com/watch?v=GKfv4l6r7hQ&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=17
6. Tutorial 3 : PyTorch Basics → https://www.youtube.com/watch?v=SEtu7Eef5ps&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=16
7. Tutorial 4 : CNNs using PyTorch → https://www.youtube.com/watch?v=BhnGtsMwUCU&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=15
8. Tutorial 5 : RNNs using PyTorch → https://www.youtube.com/watch?v=k6zF2NsvVrk&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=14
9. Tutorial 6 : Transfer Learning with PyTorch → https://www.youtube.com/watch?v=ETJG9mmeL5k&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=13
10. Tutorial 7 : Review of Basic Probability 1 → https://www.youtube.com/watch?v=owlWCCgYx50&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=12
11. Tutorial 8 : Review of Basic Probability 2 → https://www.youtube.com/watch?v=pQIbfyjSnFk&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=11
12. Tutorial 9 : Review of Basic Probability 3 → https://www.youtube.com/watch?v=eDSb3yObtB8&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=10
13. Tutorial 10 : Review of Machine Learning 1 → https://www.youtube.com/watch?v=wjSKM1xFoSU&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=9
14. Lec 03 f-Divergence and Examples → https://www.youtube.com/watch?v=LR9UQXY_IU8&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=8
15. Tutorial 11 – f-Divergence and Examples → https://www.youtube.com/watch?v=GjxuVZeMSfE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=7
16. Lec 04 Variational Divergence Minimization (VDM) → https://www.youtube.com/watch?v=4vtL3NhCkgg&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=6
17. Lec 05 Generative Adversarial Networks (GANs) → https://www.youtube.com/watch?v=5uqga82bDNA&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=5
18. Tutorial 12 : Implementations of Vanilla GAN, DCGAN and Conditional GAN → https://www.youtube.com/watch?v=dBcURX7GrwE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=4
19. Lec 18 Wasserstein GAN (WGAN) → https://www.youtube.com/watch?v=1neDqqgaXhE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=3
20. Lec 19 Inversion with GANs and FID → https://www.youtube.com/watch?v=zw2DUzD0TLE&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=2
21. Lec 20 Latent Variable Models and Introduction to Variational Autoencoder (VAE) → https://www.youtube.com/watch?v=4djE9goJtKs&list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK&index=1

---

## External resources

| Resource | URL |
|----------|-----|
| YouTube playlist (this file) | https://www.youtube.com/playlist?list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK |
| NPTEL course page | https://nptel.ac.in/courses/106108004 |
| Swayam preview (noc26_cs97) | https://onlinecourses.nptel.ac.in/noc26_cs97/preview |
| Parent — MFML catalog | [`../NOTES.md`](../NOTES.md) |
| Sibling — IITM BS GenAI catalog (full 73 videos) | [`../IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md`](../IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md) |
| IITM PyTorch notebooks (same instructor, other recording) | https://github.com/Chandan-IISc/IITM_GenAI |

---

## Sources

- [NPTEL playlist](https://www.youtube.com/playlist?list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK) (21 video titles, ids, durations)
- [NPTEL 106108004](https://nptel.ac.in/courses/106108004) (course abstract)
- Lecture packages in this folder for Lec 01–04 and Tutorials 2–11
- IITM BS 12-week map for the part of the syllabus not yet on this playlist
