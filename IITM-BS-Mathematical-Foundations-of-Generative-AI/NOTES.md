# Mathematical Foundations of Generative AI

IIT Madras B.S. Degree Programme · Course **BSDA5002** · Prof. Prathosh A P (EECS, IISc Bangalore)

**Playlist:** [Mathematical Foundations of Generative AI](https://www.youtube.com/playlist?list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu)
**Channel:** IIT Madras — B.S. Degree Programme
**Size:** 73 videos · ~32.3 hours · last updated 5 May 2026

This file is the **course map + catalog**. Per-lecture study packages (`PREREQUISITES.md` / `NOTES.md` / quiz) are added here as we build them.


| Playlist # | Video                                                                     | Study package                                                                                                        |
| ------------ | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 2          | W1_L2 Introduction & problem setting                                      | [`01-W1-L2-Introduction-Problem-Setting/`](./01-W1-L2-Introduction-Problem-Setting/)                                 |
| 3          | W1_L3 (YouTube title: F-divergence;**recording is PyTorch tensors**)      | [`02-W1-L3-F-Divergence/`](./02-W1-L3-F-Divergence/)                                                                 |
| 4          | W1_L4 Variational divergence minimization                                 | [`03-W1-L4-Variational-Divergence-Minimization/`](./03-W1-L4-Variational-Divergence-Minimization/)                   |
| 6          | W1_T2 Introduction to pytorch: tensors*(recording: Dataset / DataLoader)* | [`04-W1-T2-Introduction-to-PyTorch-Tensors/`](./04-W1-T2-Introduction-to-PyTorch-Tensors/)                           |
| 9          | W2_L5 (YouTube title: VDM;**recording is MLP forward/backprop**)          | [`07-W2-L5-Generative-Modelling-via-VDM/`](./07-W2-L5-Generative-Modelling-via-VDM/)                                 |
| 10         | W2_L6 Generative adversarial networks: introduction                       | [`08-W2-L6-Generative-Adversarial-Networks-Introduction/`](./08-W2-L6-Generative-Adversarial-Networks-Introduction/) |
| 11         | W2_L7 Generative adversarial networks: formulation                        | [`09-W2-L7-GAN-Formulation/`](./09-W2-L7-GAN-Formulation/)                                                           |
| 7          | W1_T3 datasets & dataloaders*(recording: VDM two-net saddle)*             | [`05-W1-T3-PyTorch-Datasets-DataLoaders/`](./05-W1-T3-PyTorch-Datasets-DataLoaders/)                                 |
| 8          | W1_T4 Introduction to pytorch: model building                             | [`06-W1-T4-Introduction-to-PyTorch-Model-Building/`](./06-W1-T4-Introduction-to-PyTorch-Model-Building/)             |
| 9          | W2_L5 VDM*(check recording)*                                              | [`07-W2-L5-Generative-Modelling-via-VDM/`](./07-W2-L5-Generative-Modelling-via-VDM/)                                 |
| 10         | W2_L6 GAN introduction*(chalk: one $f$, two-log $J$)*                     | [`08-W2-L6-Generative-Adversarial-Networks-Introduction/`](./08-W2-L6-Generative-Adversarial-Networks-Introduction/) |
| 11         | W2_L7 GAN formulation*(D ascent / G descent / freeze)*                    | [`09-W2-L7-GAN-Formulation/`](./09-W2-L7-GAN-Formulation/)                                                           |
| 12         | W2_T5 Tutorial: Implementation of GAN*(vanilla MNIST Colab)*              | [`10-W2-T5-Implementation-of-GAN/`](./10-W2-T5-Implementation-of-GAN/)                                               |
| 13         | W3_L8 GANs as classifier-guided generative sampler                         | [`11-W3-L8-GANs-as-Classifier-Guided-Generative-Sampler/`](./11-W3-L8-GANs-as-Classifier-Guided-Generative-Sampler/) |

Same instructor, similar math, **different recording** from NPTEL / IISc ([`../Mathematical-Foundation-for-GenerativeAI/`](../Mathematical-Foundation-for-GenerativeAI/)).

Same instructor, similar math, **different recording**: IIT Madras BS (this playlist) vs NPTEL / IISc.

---

## Table of Contents

- [Mathematical Foundations of Generative AI](#mathematical-foundations-of-generative-ai)
  - [Table of Contents](#table-of-contents)
  - [What the course is for](#what-the-course-is-for)
  - [Architecture of the whole course](#architecture-of-the-whole-course)
    - [Four engines, then alignment](#four-engines-then-alignment)
    - [What each engine *cannot* skip](#what-each-engine-cannot-skip)
  - [How the 12 weeks fit](#how-the-12-weeks-fit)
  - [Playlist catalog](#playlist-catalog)
    - [Week 1 — Introduction to probabilistic deep generative modelling](#week-1--introduction-to-probabilistic-deep-generative-modelling)
    - [Week 2 — Variational divergence minimization → GANs](#week-2--variational-divergence-minimization--gans)
    - [Week 3 — GANs part 1](#week-3--gans-part-1)
    - [Week 4 — WGANs, inversion, evaluation](#week-4--wgans-inversion-evaluation)
    - [Week 5 — Latent-variable models and VAE](#week-5--latent-variable-models-and-vae)
    - [Week 6 — Training VAEs, β-VAE, VQ-VAE](#week-6--training-vaes-β-vae-vq-vae)
    - [Week 7 — DDPM formulation](#week-7--ddpm-formulation)
    - [Week 8 — DDPM ELBO, training, inference](#week-8--ddpm-elbo-training-inference)
    - [Week 9 — Score-based, guided, latent, DDIM](#week-9--score-based-guided-latent-ddim)
    - [Week 10 — Autoregressive models and Transformers](#week-10--autoregressive-models-and-transformers)
    - [Week 9 tutorials (playlist order)](#week-9-tutorials-playlist-order)
    - [Week 11 — RL for language models](#week-11--rl-for-language-models)
    - [Week 12 — Reward modelling, DPO, SSMs](#week-12--reward-modelling-dpo-ssms)
  - [Compact title → URL list](#compact-title--url-list)
  - [External resources](#external-resources)
  - [Sources](#sources)

---

## What the course is for

The course treats generative AI as a **probability problem**, not a product demo. You are given files (images, tokens, spectrograms) that you treat as draws from an unknown law $p_x$. Two jobs follow:

1. **Estimate** that law (or a surrogate that is good enough to sample from).
2. **Sample** a brand-new file that was never in the pile.

Lectures are chalk-and-talk: every equation is written, with proofs. Tutorials (`T` videos) re-implement the same object in PyTorch from scratch (student: Chandan). The prescribed book is Foster, *Generative Deep Learning* (O'Reilly, 2023), plus papers.

`L` = lecture. `T` = tutorial. Week-9 tutorials sit **after** week-10 lectures in the playlist (videos 63–65).

---

## Architecture of the whole course

**Worldview arc:** from “data is samples of an unknown $p_x$” **to** four sampling engines (adversarial, variational, diffusion, autoregressive) **to** aligning an autoregressive language model with preferences.

Every family in the playlist is a different answer to the **same recipe**:

```
  Dataset D = {x_1, …, x_n}  ⊂  R^d ,   x_i ~ p_x  (unknown)

                    │
                    ▼
         pick a generator family
         (explicit p_θ  or  implicit G_θ(z))
                    │
                    ▼
         pick a discrepancy d(p_x, model)
         f-div / JS / Wasserstein / ELBO / score / NLL / preference
                    │
                    ▼
         train θ  (min  or  min_θ max_w saddle)
                    │
                    ▼
         SAMPLE  x_new  from the trained object
```

### Four engines, then alignment

```
  W1  problem: estimate p_x  AND  sample
   │
   ├─ W1–W4   implicit generator G_θ(z)
   │          distance = f-divergence → VDM saddle → GAN / WGAN
   │          leftover: no encoder, inversion is extra work
   │
   ├─ W5–W6   explicit latent p(x) = ∫ p(x|z) p(z) dz
   │          distance = −ELBO  (VAE, β-VAE, VQ-VAE)
   │          leftover: blurry samples; discrete codes wait for diffusion
   │
   ├─ W7–W9   hierarchical noising x_0 → x_T, reverse denoise
   │          distance = DDPM ELBO ≡ noise / mean / score MSE
   │          leftover: many steps; DDIM / latent / guidance fix that
   │
   └─ W10     autoregressive  p(x) = Π_i p(x_i | x_<i)
              Transformer = the mixer; NLL is the d
              leftover: next-token LMs are unaligned
                    │
                    ▼
              W11–W12  treat the LM as an RL policy
                       PPO / TRPO / reward model / DPO
                       SSM/Mamba as a long-sequence alternative
```

### What each engine *cannot* skip


| Family           | What you train                                   | How you sample                             | Typical failure if you skip the math        |
| ------------------ | -------------------------------------------------- | -------------------------------------------- | --------------------------------------------- |
| VDM / GAN        | critic$T$ vs generator $G_\theta$                | $z\sim p(z)$, $x=G_\theta(z)$              | saturation, mode collapse, no likelihood    |
| VAE              | encoder$q_\phi(z\|x)$ + decoder $p_\theta(x\|z)$ | $z\sim p(z)$ or $q_\phi$, then decode      | posterior collapse, blur, ignoring the KL   |
| DDPM / DDIM      | $\epsilon_\theta(x_t,t)$ (or score / mean)       | reverse Markov (DDPM) or shorter DDIM path | wrong noise schedule, no reverse process    |
| AR / Transformer | next-token$p_\theta(x_i\|x_{<i})$                | ancestral decode                           | teacher-forcing train ≠ free-run inference |
| RLHF / DPO       | reward$r$ or preference $\pi_\theta$             | still decode the LM                        | treating alignment as “just more NLL”     |

The NPTEL Lec 02 package in this repo ([problem formulation](../Mathematical-Foundation-for-GenerativeAI/15-Lec02-Generative-Models-Problem-Formulation/NOTES.md)) is the same *recipe* in a different recording. This IITM playlist is the full 12-week expansion of that recipe.

---

## How the 12 weeks fit


| Week | Official theme                                               | What actually happens                                                                                                      |
| ------ | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 1    | Introduction to probabilistic deep generative modelling      | State the estimate+sample problem. Introduce$f$-divergence and variational divergence minimization (VDM). PyTorch warm-up. |
| 2    | Generative modelling via variational divergence minimization | Finish VDM; specialise it to GANs (intro + value function). First GAN tutorial.                                            |
| 3    | GANs part 1                                                  | Discriminator as density-ratio / classifier-guided sampler. DCGAN, conditional GAN.                                        |
| 4    | GANs part 2 (WGANs and applications)                         | Why GAN loss saturates → Wasserstein. Inversion, BiGAN, domain-adversarial nets, evaluation.                              |
| 5    | Generative modelling via variational autoencoding            | Latent variables, ELBO, GMM+EM as the discrete-latent special case, VAE.                                                   |
| 6    | VAEs: improvisations and VQ-VAE                              | Reparameterization, train/infer, β-VAE, discrete codebook (VQ-VAE).                                                       |
| 7    | DDPM — formulation                                          | Forward noising / reverse denoising. U-Net as the score/noise network.                                                     |
| 8    | Diffusion: multiple forms and implementation                 | DDPM ELBO in full, equivalent losses, train, ancestral sample.                                                             |
| 9    | Conditional diffusion and score-based models                 | Score view, guidance, latent diffusion, DDIM. Tutorials appear after week 10 in the playlist.                              |
| 10   | Autoregressive models and LLMs                               | Factorization, attention, Transformer stack, train vs decode.                                                              |
| 11   | LLMs: RL overview, PPO, TRPO                                 | Language model = policy. Policy gradient, PPO, TRPO.                                                                       |
| 12   | LLMs — alignment (PPO leftover, DPO) + SSMs                 | Reward model, Direct Preference Optimization, state-space models / Mamba.                                                  |

Suggested first watch: **W1_L2** (problem setting, 58 min), then **W1_L3–L4** ($f$-div and VDM). That is the same spine as NPTEL Lec 02–04.

---

## Playlist catalog

Links keep the playlist id so YouTube stays in-list, same shape as:

`W1_L2: Introduction & problem setting | generative AI basics explained` → https://www.youtube.com/watch?v=HUunmwZfGzc&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=2

Summaries are from official titles + the published week syllabus. They are not substitutes for watching the lecture.

### Week 1 — Introduction to probabilistic deep generative modelling


| # | Video                                                                          | Duration | Link                                                                                                 | Summary                                                                                                                                                              |
| --- | -------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | W1_L1: Course outline deep generative models                                   | 9:33     | [watch](https://www.youtube.com/watch?v=skWhn8W9P_Y&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=1) | Roadmap of the 12-week course: GANs, VAEs, diffusion, autoregressive models, LLMs, RL alignment, SSMs.                                                               |
| 2 | W1_L2: Introduction & problem setting\| generative AI basics explained         | 58:32    | [watch](https://www.youtube.com/watch?v=HUunmwZfGzc&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=2) | Casts generative AI as: given data, estimate a density$p_x$ **and** sample from it. This is the problem the rest of the course solves.                               |
| 3 | W1_L3: F-divergence\| variational divergence minimization in generative models | 28:44    | [watch](https://www.youtube.com/watch?v=rHnrALMCyIQ&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=3) | $f$-divergence as the distance $d(p,p_\theta)$ used to train generators. KL, reverse-KL, JS; they are not metrics.                                                   |
| 4 | W1_L4: Variational divergence minimization                                     | 26:09    | [watch](https://www.youtube.com/watch?v=nfZQYopzv20&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=4) | Names$f$-divergence (KL / JS / TV). The conjugate / critic algorithm is **next** (W2_L5). Package: [`03-W1-L4-…`](./03-W1-L4-Variational-Divergence-Minimization/). |
| 5 | W1_T1: Tutorial 1: Forward pass & backpropagation                              | 42:50    | [watch](https://www.youtube.com/watch?v=VxRIqenOoQw&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=5) | Neural-net forward pass and backprop from scratch so later PyTorch tutorials are not black boxes.                                                                    |
| 6 | W1_T2: Tutorial 2: Introduction to pytorch: tensors                            | 18:26    | [watch](https://www.youtube.com/watch?v=L5n4rNrLZ_8&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=6) | Title says tensors; recording is Dataset/DataLoader. Package:[`04-W1-T2-…`](./04-W1-T2-Introduction-to-PyTorch-Tensors/).                                           |
| 7 | W1_T3: Tutorial 3: Introduction to pytorch: datasets & dataloaders             | 30:44    | [watch](https://www.youtube.com/watch?v=c2gN3TK3U74&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=7) | `Dataset` / `DataLoader` so training loops can stream minibatches.                                                                                                   |
| 8 | W1_T4: Tutorial 4: Introduction to pytorch: model building                     | 44:06    | [watch](https://www.youtube.com/watch?v=h1hEddM0aVE&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=8) | `nn.Module`, autograd, and a first trainable model.                                                                                                                  |

### Week 2 — Variational divergence minimization → GANs


| #  | Video                                                               | Duration | Link                                                                                                  | Summary                                                                                                               |
| ---- | --------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 9  | W2_L5: Generative modelling via variational divergence minimization | 54:54    | [watch](https://www.youtube.com/watch?v=stZC0Zk5KYo&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=9)  | Completes VDM: generator$G_\theta$ vs critic, saddle-point training, and how this is the parent of GANs.              |
| 10 | W2_L6: Generative adversarial networks: introduction                | 22:13    | [watch](https://www.youtube.com/watch?v=EHhURRwMEPo&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=10) | GAN idea: a generator fools a discriminator; samples come from$G(z)$ with no explicit density.                        |
| 11 | W2_L7: Generative adversarial networks: formulation                 | 35:20    | [watch](https://www.youtube.com/watch?v=pLD5Q5cS4kI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=11) | Original GAN value function and its link to Jensen–Shannon divergence.                                               |
| 12 | W2_T5: Tutorial: Implementation of generative adversarial network   | 41:07    | [watch](https://www.youtube.com/watch?v=iOb8vmlJd8o&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=12) | Vanilla GAN in PyTorch ([notebook](https://github.com/Chandan-IISc/IITM_GenAI/blob/main/IITM_DGM_Vanilla_GAN.ipynb)). |

### Week 3 — GANs part 1


| #  | Video                                              | Duration | Link                                                                                                  | Summary                                                                                                        |
| ---- | ---------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 13 | W3L8: GANs as classifier-guided generative sampler | 41:29    | [watch](https://www.youtube.com/watch?v=ga8VOW6pPeA&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=13) | Reads the discriminator as a density-ratio estimator that **guides** the generator’s samples. Package: [`11-W3-L8-…`](./11-W3-L8-GANs-as-Classifier-Guided-Generative-Sampler/). |
| 14 | W3L9: Deep Convolution GANs and Conditional GANs   | 21:26    | [watch](https://www.youtube.com/watch?v=CFymHrr5iQw&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=14) | DCGAN (conv generator/discriminator) and cGAN (class-conditional generation).                                  |
| 15 | W3T6: Tutorial: Implementation of DC-GAN           | 27:33    | [watch](https://www.youtube.com/watch?v=4-o6d8EyxCU&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=15) | DCGAN implementation ([notebook](https://github.com/Chandan-IISc/IITM_GenAI/blob/main/IITM_DGM_DC_GAN.ipynb)). |

### Week 4 — WGANs, inversion, evaluation


| #  | Video                                      | Duration | Link                                                                                                  | Summary                                                                                                              |
| ---- | -------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 16 | W4L10: Saturation of GAN training          | 27:08    | [watch](https://www.youtube.com/watch?v=2RMeQ5YxIxI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=16) | Why the original GAN loss saturates: vanishing gradients when the discriminator is too strong.                       |
| 17 | W4L11: Wasserstein GANs                    | 50:53    | [watch](https://www.youtube.com/watch?v=_IBfVkrvqAI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=17) | Earth-mover / Wasserstein-1 distance, Lipschitz critic, and why WGAN training is more stable.                        |
| 18 | W4L12: Inversion with GANs                 | 13:28    | [watch](https://www.youtube.com/watch?v=S84MmiEr-6o&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=18) | Find$z$ such that $G(z)\approx x$ — using a trained generator as an encoder.                                        |
| 19 | W4L13: Bi-directional GANs                 | 22:32    | [watch](https://www.youtube.com/watch?v=9av__QKR_xk&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=19) | BiGAN: jointly learn generator and encoder so you get latent codes, not only samples.                                |
| 20 | W4L14: GAN inversion via latent regression | 6:22     | [watch](https://www.youtube.com/watch?v=gkMIerCn8n0&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=20) | Train a regressor$x\mapsto z$ to invert $G$ without a full BiGAN.                                                    |
| 21 | W4L15: Domain Adversarial Networks         | 27:42    | [watch](https://www.youtube.com/watch?v=rWk04R1VH8Q&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=21) | Domain-adversarial nets (DANN/UDA): a discriminator on**features** for unsupervised domain adaptation.               |
| 22 | W4L16: Evaluation of Generative Models     | 22:52    | [watch](https://www.youtube.com/watch?v=5Mchnh2xedI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=22) | How to score generators when likelihood is unavailable (sample quality / diversity).                                 |
| 23 | W4T7: Tutorial: Implementation of Bi-GAN   | 14:55    | [watch](https://www.youtube.com/watch?v=eH9skKyqjJM&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=23) | BiGAN in PyTorch ([notebook](https://github.com/Chandan-IISc/IITM_GenAI/blob/main/IITM_DGM_Bi_GAN.ipynb)).           |
| 24 | W4T8: Tutorial: Implementation of UDA      | 28:55    | [watch](https://www.youtube.com/watch?v=kD708wpM14c&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=24) | Unsupervised domain adaptation ([notebook](https://github.com/Chandan-IISc/IITM_GenAI/blob/main/IIT_DGM_UDA.ipynb)). |
| 25 | W4T9: Tutorial: Implementation of WGAN     | 23:06    | [watch](https://www.youtube.com/watch?v=ZApQpSKjazs&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=25) | WGAN / WGAN-GP ([notebook](https://github.com/Chandan-IISc/IITM_GenAI/blob/main/IIT_DGM_WGAN.ipynb)).                |

### Week 5 — Latent-variable models and VAE


| #  | Video                                                              | Duration | Link                                                                                                  | Summary                                                                                      |
| ---- | -------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 26 | W5L17: Introduction to latent variable models                      | 16:51    | [watch](https://www.youtube.com/watch?v=UAnp6yU8K0A&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=26) | Why introduce a latent$z$: $p(x)=\int p(x\|z)p(z)\,dz$ and why that integral is intractable. |
| 27 | W5L18: Evidence Lower Bound (ELBO)                                 | 25:50    | [watch](https://www.youtube.com/watch?v=d9fDDUcQqq4&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=27) | Derives ELBO as a lower bound on$\log p(x)$ via a variational posterior $q(z\|x)$.           |
| 28 | W5L19: Gaussian Mixture Models: Expectation-Maximization Algorithm | 30:29    | [watch](https://www.youtube.com/watch?v=zUJNypPc-Vo&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=28) | GMM as the discrete-latent special case; E-step / M-step of EM.                              |
| 29 | W5L20: Variational Autoencoder (VAE)                               | 33:11    | [watch](https://www.youtube.com/watch?v=RN3_gkjlYoA&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=29) | Encoder$=q_\phi(z\|x)$, decoder $=p_\theta(x\|z)$; VAE is amortized variational inference.   |
| 30 | W5T10: Proof of Jensen's inequality                                | 13:55    | [watch](https://www.youtube.com/watch?v=br7oydgTero&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=30) | Jensen’s inequality — the inequality that makes ELBO a**lower** bound.                     |
| 31 | W5T11: GMM                                                         | 28:36    | [watch](https://www.youtube.com/watch?v=VB0E_tzwuxI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=31) | Implements GMM / EM numerically.                                                             |

### Week 6 — Training VAEs, β-VAE, VQ-VAE


| #  | Video                                           | Duration | Link                                                                                                  | Summary                                                                                                             |
| ---- | ------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 32 | W6L21: Training VAE: Reparameterization methods | 37:14    | [watch](https://www.youtube.com/watch?v=blh_AnhwIpw&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=32) | Reparameterization trick:$z=\mu+\sigma\odot\epsilon$ so gradients flow through sampling.                            |
| 33 | W6L22: Training VAE                             | 43:48    | [watch](https://www.youtube.com/watch?v=dAQVTNnRrEg&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=33) | Full VAE training loop: reconstruction + KL regularizer on the encoder.                                             |
| 34 | W6L23: Inference with a trained VAE             | 29:59    | [watch](https://www.youtube.com/watch?v=x85aw6DXX0w&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=34) | Encode, decode, interpolate in$z$, and generate new samples after training.                                         |
| 35 | W6L24: Beta-VAE                                 | 18:03    | [watch](https://www.youtube.com/watch?v=6ZBvXaVgAGA&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=35) | Extra weight on the KL term to encourage disentangled latents.                                                      |
| 36 | W6L25: Vector Quantized VAE (VQ-VAE)            | 24:36    | [watch](https://www.youtube.com/watch?v=8UbwFtDJEG0&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=36) | Discrete codebook latents; VQ-VAE as the discrete-token cousin of VAE (used later in latent diffusion).             |
| 37 | W6T12: Implementation of VAE                    | 34:23    | [watch](https://www.youtube.com/watch?v=ZJh_Jv0hxZM&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=37) | VAE / β-VAE in PyTorch ([notebook](https://github.com/Chandan-IISc/IITM_GenAI/blob/main/IITM_DGM_Beta_VAE.ipynb)). |
| 38 | W6T13: Implementation of VQ-VAE                 | 15:10    | [watch](https://www.youtube.com/watch?v=o2Nc6_kcgEw&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=38) | VQ-VAE implementation ([notebook](https://github.com/Chandan-IISc/IITM_GenAI/blob/main/IITM_DGM_VQ_VAE.ipynb)).     |

### Week 7 — DDPM formulation


| #  | Video                                                   | Duration | Link                                                                                                  | Summary                                                                                                                                 |
| ---- | --------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 39 | W7L26: Denoising Diffusion Probabilistic Models (DDPMs) | 17:09    | [watch](https://www.youtube.com/watch?v=N0OOnTKMYJE&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=39) | Forward noising$x_0\to x_T$ vs reverse denoising $x_T\to x_0$.                                                                          |
| 40 | W7L27: DDPM: Formulation                                | 43:16    | [watch](https://www.youtube.com/watch?v=P8AiIW0Gg0s&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=40) | Markov forward process, reverse model, and the likelihood that will be bounded by ELBO.                                                 |
| 41 | W7T14: U-Net                                            | 13:44    | [watch](https://www.youtube.com/watch?v=-sOaKgKAmaM&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=41) | U-Net backbone used as the DDPM noise predictor ([notebook](https://github.com/Chandan-IISc/IITM_GenAI/blob/main/IITM_DGM_UNET.ipynb)). |

### Week 8 — DDPM ELBO, training, inference


| #  | Video                            | Duration | Link                                                                                                  | Summary                                                                                                        |
| ---- | ---------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 42 | W8L28: ELBO for DDPM : Part 1    | 40:41    | [watch](https://www.youtube.com/watch?v=AnWitwNPnN4&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=42) | Starts the DDPM ELBO: KL terms between forward posterior and reverse transitions.                              |
| 43 | W8L29: ELBO for DDPM : Part 2    | 36:50    | [watch](https://www.youtube.com/watch?v=wPx64rVy2c4&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=43) | Finishes the ELBO expansion and isolates the terms that actually get trained.                                  |
| 44 | W8L30: Optimization of DDPM loss | 30:54    | [watch](https://www.youtube.com/watch?v=yzU0ueuABLw&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=44) | Simplifies ELBO into the practical noise-prediction MSE used in code.                                          |
| 45 | W8L31: ELBO Equivalence          | 14:51    | [watch](https://www.youtube.com/watch?v=j-xrjqvSKhU&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=45) | Several DDPM loss writings (noise / mean / sample) are equivalent.                                             |
| 46 | W8L32: Training of DDPM          | 11:37    | [watch](https://www.youtube.com/watch?v=47TD4eSLVMI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=46) | Training algorithm: sample$t$, noise $x_0$, regress $\epsilon_\theta$.                                         |
| 47 | W8L33: Inference in DDPM         | 19:33    | [watch](https://www.youtube.com/watch?v=0p7T-3WiPnQ&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=47) | Ancestral reverse sampling from$x_T\sim\mathcal N(0,I)$ down to $x_0$.                                         |
| 48 | W8T15: Implementation of DDPM    | 26:11    | [watch](https://www.youtube.com/watch?v=au-7zLxcP1k&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=48) | DDPM training/sampling ([notebook](https://github.com/Chandan-IISc/IITM_GenAI/blob/main/IITM_DGM_DDPM.ipynb)). |
| 49 | W8T16: Proofs                    | 27:49    | [watch](https://www.youtube.com/watch?v=rNxtbFa8J-s&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=49) | Algebraic proofs behind the DDPM ELBO reductions.                                                              |

### Week 9 — Score-based, guided, latent, DDIM


| #  | Video                                              | Duration | Link                                                                                                  | Summary                                                                                               |
| ---- | ---------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 50 | W9L34: alternate interpretations of DDPMs          | 18:16    | [watch](https://www.youtube.com/watch?v=30fSjB8oGN0&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=50) | DDPM as noise prediction**or** mean prediction **or** score matching — same object, different views. |
| 51 | W9L35: DDPMs as score-predictors                   | 22:46    | [watch](https://www.youtube.com/watch?v=2Sp0BqAWWXY&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=51) | $\epsilon_\theta$ estimates $\nabla_x \log p_t(x)$; Langevin / score-based sampling.                  |
| 52 | W9L36: Guided Diffusion Models                     | 33:45    | [watch](https://www.youtube.com/watch?v=kJCgO7rwo3Y&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=52) | Classifier / classifier-free guidance to steer samples with a condition (class, text).                |
| 53 | W9L37: Latent Diffusion Models                     | 13:16    | [watch](https://www.youtube.com/watch?v=qkinkLtwSyc&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=53) | Run DDPM in a VAE/VQ latent space (the Stable Diffusion idea).                                        |
| 54 | W9L38: Denoising Diffusion Implicit Models (DDIMs) | 43:09    | [watch](https://www.youtube.com/watch?v=qiMJBB8chzI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=54) | Non-Markov reverse process: fewer steps, deterministic path, same trained$\epsilon_\theta$.           |
| 55 | W9L39: Inference in DDIM                           | 22:43    | [watch](https://www.youtube.com/watch?v=3KQWN9SS7L0&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=55) | DDIM sampling,$\eta$-interpolation between stochastic DDPM and deterministic DDIM.                    |

Week-10 lectures come next in the playlist; week-9 tutorials are videos 63–65.

### Week 10 — Autoregressive models and Transformers


| #  | Video                                                    | Duration | Link                                                                                                  | Summary                                                                             |
| ---- | ---------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| 56 | W10L40: Auto-Regressive Models                           | 25:08    | [watch](https://www.youtube.com/watch?v=PtDFqdTbQUY&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=56) | $p(x)=\prod_i p(x_i\|x_{<i})$: generation as next-token (or next-pixel) prediction. |
| 57 | W10L41: Attention Mechanism                              | 32:02    | [watch](https://www.youtube.com/watch?v=LoO0zQvmArE&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=57) | Query–key–value attention as the mixing operator for sequences.                   |
| 58 | W10L42: Transformers for Auto-Regressive Models          | 25:14    | [watch](https://www.youtube.com/watch?v=uOb-dRc8yrA&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=58) | Causal (masked) Transformers as AR language models.                                 |
| 59 | W10L43: Transformers architecture                        | 33:31    | [watch](https://www.youtube.com/watch?v=NAWe0F0PmRU&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=59) | Encoder/decoder blocks, multi-head attention, FFN.                                  |
| 60 | W10L44: Transformers: Skip Connections and Normalization | 11:05    | [watch](https://www.youtube.com/watch?v=ZtGkl72SZPo&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=60) | Residual stream + LayerNorm (pre-norm vs post-norm).                                |
| 61 | W10L45: Transformers: Position Embeddings                | 11:51    | [watch](https://www.youtube.com/watch?v=_k1vd1DaMzQ&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=61) | Why attention needs positions; absolute / learned positional encodings.             |
| 62 | W10L46: Transformers: Training and Inference             | 21:42    | [watch](https://www.youtube.com/watch?v=RJmTsB7mnUk&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=62) | Teacher-forced training vs autoregressive decoding (sampling, temperature).         |

### Week 9 tutorials (playlist order)

These belong conceptually with week 9; YouTube lists them after week 10.


| #  | Video                                          | Duration | Link                                                                                                  | Summary                                                                                                                                               |
| ---- | ------------------------------------------------ | ---------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 63 | W9T17: Implementation of DDPM Noise estimation | 20:56    | [watch](https://www.youtube.com/watch?v=cmMm8Jgm3hI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=63) | Train$\epsilon_\theta$ (noise-prediction DDPM) ([notebook](https://github.com/Chandan-IISc/IITM_GenAI/blob/main/IITM_DGM_DDPM_Noise_Estimate.ipynb)). |
| 64 | W9T18: Implementation of DDIM                  | 9:53     | [watch](https://www.youtube.com/watch?v=kT-PCHIayS0&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=64) | DDIM sampler on a trained noise network ([notebook](https://github.com/Chandan-IISc/IITM_GenAI/blob/main/IITM_DGM_DDIM.ipynb)).                       |
| 65 | W9T19: Implementation of Guided DDPM           | 11:18    | [watch](https://www.youtube.com/watch?v=BKqUPihkNmY&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=65) | Guided / classifier-free diffusion ([notebook](https://github.com/Chandan-IISc/IITM_GenAI/blob/main/IITM_DGM_DDPM_Guided_Diffusion.ipynb)).           |

### Week 11 — RL for language models


| #  | Video                                           | Duration | Link                                                                                                  | Summary                                                                                  |
| ---- | ------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 66 | W11L47: An overview of Reinforcement Learning   | 26:32    | [watch](https://www.youtube.com/watch?v=4NZ6DLFF2DY&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=66) | MDP, policy, return — the RL vocabulary needed for LLM alignment.                       |
| 67 | W11L48: Policy Gradient Theorem                 | 26:20    | [watch](https://www.youtube.com/watch?v=5FOz4agFl4M&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=67) | $\nabla J=\mathbb E[\nabla\log\pi\cdot A]$: how to differentiate through sampled tokens. |
| 68 | W11L49: Expressing an AR-LM as RL policy        | 7:03     | [watch](https://www.youtube.com/watch?v=Y2KZ_zDGhKw&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=68) | Next-token LM = policy$\pi(a_t\|s_t)$ over a growing prompt/state.                       |
| 69 | W11L50: Proximal Policy Optimization (PPO)      | 30:52    | [watch](https://www.youtube.com/watch?v=mvANXESrhqc&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=69) | Clipped surrogate objective used in RLHF.                                                |
| 70 | W11L51: Trust Region Policy Optimization (TRPO) | 17:30    | [watch](https://www.youtube.com/watch?v=9MJkL3XiCsk&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=70) | KL-constrained policy updates; PPO is the practical cousin of TRPO.                      |

### Week 12 — Reward modelling, DPO, SSMs


| #  | Video                                        | Duration | Link                                                                                                  | Summary                                                                                        |
| ---- | ---------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 71 | W12L52: Reward-Modelling                     | 15:23    | [watch](https://www.youtube.com/watch?v=kCwdMhKO0x8&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=71) | Train a reward model from human (or AI) preference pairs for RLHF.                             |
| 72 | W12L53: Direct Preference Optimization (DPO) | 18:44    | [watch](https://www.youtube.com/watch?v=P4Tm0FURBFU&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=72) | Align$\pi_\theta$ to preferences **without** an explicit RL loop; DPO as a closed-form RLHF.   |
| 73 | W12L54: State-space-Models                   | 1:05:49  | [watch](https://www.youtube.com/watch?v=A7iAmVr0QE4&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=73) | SSMs / Mamba as a long-sequence alternative to Transformers (longest lecture in the playlist). |

---

## Compact title → URL list

1. W1_L1: Course outline deep generative models → https://www.youtube.com/watch?v=skWhn8W9P_Y&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=1
2. W1_L2: Introduction & problem setting | generative AI basics explained → https://www.youtube.com/watch?v=HUunmwZfGzc&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=2
3. W1_L3: F-divergence | variational divergence minimization in generative models → https://www.youtube.com/watch?v=rHnrALMCyIQ&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=3
4. W1_L4: Variational divergence minimization → https://www.youtube.com/watch?v=nfZQYopzv20&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=4
5. W1_T1: Tutorial 1: Forward pass & backpropagation → https://www.youtube.com/watch?v=VxRIqenOoQw&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=5
6. W1_T2: Tutorial 2: Introduction to pytorch: tensors → https://www.youtube.com/watch?v=L5n4rNrLZ_8&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=6
7. W1_T3: Tutorial 3: Introduction to pytorch: datasets & dataloaders → https://www.youtube.com/watch?v=c2gN3TK3U74&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=7
8. W1_T4: Tutorial 4: Introduction to pytorch: model building → https://www.youtube.com/watch?v=h1hEddM0aVE&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=8
9. W2_L5: Generative modelling via variational divergence minimization → https://www.youtube.com/watch?v=stZC0Zk5KYo&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=9
10. 
11. W2_L6: Generative adversarial networks: introduction → https://www.youtube.com/watch?v=EHhURRwMEPo&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=10
12. W2_L7: Generative adversarial networks: formulation → https://www.youtube.com/watch?v=pLD5Q5cS4kI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=11
13. W2_T5: Tutorial: Implementation of generative adversarial network → https://www.youtube.com/watch?v=iOb8vmlJd8o&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=12
14. PENDING --
15. W3L8: GANs as classifier-guided generative sampler → https://www.youtube.com/watch?v=ga8VOW6pPeA&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=13
16. 
17. W3L9: Deep Convolution GANs and Conditional GANs → https://www.youtube.com/watch?v=CFymHrr5iQw&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=14
18. W3T6: Tutorial: Implementation of DC-GAN → https://www.youtube.com/watch?v=4-o6d8EyxCU&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=15
19. W4L10: Saturation of GAN training → https://www.youtube.com/watch?v=2RMeQ5YxIxI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=16
20. W4L11: Wasserstein GANs → https://www.youtube.com/watch?v=_IBfVkrvqAI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=17
21. W4L12: Inversion with GANs → https://www.youtube.com/watch?v=S84MmiEr-6o&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=18
22. W4L13: Bi-directional GANs → https://www.youtube.com/watch?v=9av__QKR_xk&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=19
23. W4L14: GAN inversion via latent regression → https://www.youtube.com/watch?v=gkMIerCn8n0&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=20
24. W4L15: Domain Adversarial Networks → https://www.youtube.com/watch?v=rWk04R1VH8Q&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=21
25. W4L16: Evaluation of Generative Models → https://www.youtube.com/watch?v=5Mchnh2xedI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=22
26. W4T7: Tutorial: Implementation of Bi-GAN → https://www.youtube.com/watch?v=eH9skKyqjJM&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=23
27. W4T8: Tutorial: Implementation of UDA → https://www.youtube.com/watch?v=kD708wpM14c&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=24
28. W4T9: Tutorial: Implementation of WGAN → https://www.youtube.com/watch?v=ZApQpSKjazs&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=25
29. W5L17: Introduction to latent variable models → https://www.youtube.com/watch?v=UAnp6yU8K0A&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=26
30. W5L18: Evidence Lower Bound (ELBO) → https://www.youtube.com/watch?v=d9fDDUcQqq4&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=27
31. W5L19: Gaussian Mixture Models: Expectation-Maximization Algorithm → https://www.youtube.com/watch?v=zUJNypPc-Vo&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=28
32. W5L20: Variational Autoencoder (VAE) → https://www.youtube.com/watch?v=RN3_gkjlYoA&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=29
33. W5T10: Proof of Jensen's inequality → https://www.youtube.com/watch?v=br7oydgTero&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=30
34. W5T11: GMM → https://www.youtube.com/watch?v=VB0E_tzwuxI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=31
35. W6L21: Training VAE: Reparameterization methods → https://www.youtube.com/watch?v=blh_AnhwIpw&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=32
36. W6L22: Training VAE → https://www.youtube.com/watch?v=dAQVTNnRrEg&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=33
37. W6L23: Inference with a trained VAE → https://www.youtube.com/watch?v=x85aw6DXX0w&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=34
38. W6L24: Beta-VAE → https://www.youtube.com/watch?v=6ZBvXaVgAGA&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=35
39. W6L25: Vector Quantized VAE (VQ-VAE) → https://www.youtube.com/watch?v=8UbwFtDJEG0&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=36
40. W6T12: Implementation of VAE → https://www.youtube.com/watch?v=ZJh_Jv0hxZM&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=37
41. W6T13: Implementation of VQ-VAE → https://www.youtube.com/watch?v=o2Nc6_kcgEw&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=38
42. W7L26: Denoising Diffusion Probabilistic Models (DDPMs) → https://www.youtube.com/watch?v=N0OOnTKMYJE&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=39
43. W7L27: DDPM: Formulation → https://www.youtube.com/watch?v=P8AiIW0Gg0s&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=40
44. W7T14: U-Net → https://www.youtube.com/watch?v=-sOaKgKAmaM&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=41
45. W8L28: ELBO for DDPM : Part 1 → https://www.youtube.com/watch?v=AnWitwNPnN4&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=42
46. W8L29: ELBO for DDPM : Part 2 → https://www.youtube.com/watch?v=wPx64rVy2c4&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=43
47. W8L30: Optimization of DDPM loss → https://www.youtube.com/watch?v=yzU0ueuABLw&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=44
48. W8L31: ELBO Equivalence → https://www.youtube.com/watch?v=j-xrjqvSKhU&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=45
49. W8L32: Training of DDPM → https://www.youtube.com/watch?v=47TD4eSLVMI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=46
50. W8L33: Inference in DDPM → https://www.youtube.com/watch?v=0p7T-3WiPnQ&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=47
51. W8T15: Implementation of DDPM → https://www.youtube.com/watch?v=au-7zLxcP1k&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=48
52. W8T16: Proofs → https://www.youtube.com/watch?v=rNxtbFa8J-s&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=49
53. W9L34: alternate interpretations of DDPMs → https://www.youtube.com/watch?v=30fSjB8oGN0&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=50
54. W9L35: DDPMs as score-predictors → https://www.youtube.com/watch?v=2Sp0BqAWWXY&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=51
55. W9L36: Guided Diffusion Models → https://www.youtube.com/watch?v=kJCgO7rwo3Y&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=52
56. W9L37: Latent Diffusion Models → https://www.youtube.com/watch?v=qkinkLtwSyc&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=53
57. W9L38: Denoising Diffusion Implicit Models (DDIMs) → https://www.youtube.com/watch?v=qiMJBB8chzI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=54
58. W9L39: Inference in DDIM → https://www.youtube.com/watch?v=3KQWN9SS7L0&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=55
59. W10L40: Auto-Regressive Models → https://www.youtube.com/watch?v=PtDFqdTbQUY&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=56
60. W10L41: Attention Mechanism → https://www.youtube.com/watch?v=LoO0zQvmArE&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=57
61. W10L42: Transformers for Auto-Regressive Models → https://www.youtube.com/watch?v=uOb-dRc8yrA&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=58
62. W10L43: Transformers architecture → https://www.youtube.com/watch?v=NAWe0F0PmRU&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=59
63. W10L44: Transformers: Skip Connections and Normalization → https://www.youtube.com/watch?v=ZtGkl72SZPo&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=60
64. W10L45: Transformers: Position Embeddings → https://www.youtube.com/watch?v=_k1vd1DaMzQ&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=61
65. W10L46: Transformers: Training and Inference → https://www.youtube.com/watch?v=RJmTsB7mnUk&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=62
66. W9T17: Implementation of DDPM Noise estimation → https://www.youtube.com/watch?v=cmMm8Jgm3hI&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=63
67. W9T18: Implementation of DDIM → https://www.youtube.com/watch?v=kT-PCHIayS0&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=64
68. W9T19: Implementation of Guided DDPM → https://www.youtube.com/watch?v=BKqUPihkNmY&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=65
69. W11L47: An overview of Reinforcement Learning → https://www.youtube.com/watch?v=4NZ6DLFF2DY&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=66
70. W11L48: Policy Gradient Theorem → https://www.youtube.com/watch?v=5FOz4agFl4M&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=67
71. W11L49: Expressing an AR-LM as RL policy → https://www.youtube.com/watch?v=Y2KZ_zDGhKw&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=68
72. W11L50: Proximal Policy Optimization (PPO) → https://www.youtube.com/watch?v=mvANXESrhqc&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=69
73. W11L51: Trust Region Policy Optimization (TRPO) → https://www.youtube.com/watch?v=9MJkL3XiCsk&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=70
74. W12L52: Reward-Modelling → https://www.youtube.com/watch?v=kCwdMhKO0x8&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=71
75. W12L53: Direct Preference Optimization (DPO) → https://www.youtube.com/watch?v=P4Tm0FURBFU&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=72
76. W12L54: State-space-Models → https://www.youtube.com/watch?v=A7iAmVr0QE4&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=73

---

## External resources


| Resource                            | URL                                                                                            |
| ------------------------------------- | ------------------------------------------------------------------------------------------------ |
| YouTube playlist                    | https://www.youtube.com/playlist?list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu                       |
| IIT Madras course page (BSDA5002)   | https://study.iitm.ac.in/ds/course_pages/BSDA5002.html                                         |
| PyTorch notebooks (Chandan)         | https://github.com/Chandan-IISc/IITM_GenAI                                                     |
| Lecture notes (Drive)               | https://drive.google.com/file/d/1Kebb00VehPBMyleuw2ubp2qbvrBqyvZZ/view                         |
| Assignments (Drive)                 | https://drive.google.com/file/d/1YTciDIt2eCuq0nm3vdMK8k2hLYuA1UWk/view                         |
| NPTEL sibling packages in this repo | [`../Mathematical-Foundation-for-GenerativeAI/`](../Mathematical-Foundation-for-GenerativeAI/) |

---

## Sources

- [IIT Madras BS playlist](https://www.youtube.com/playlist?list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu) (73 video titles, ids, durations)
- [BSDA5002 course page](https://study.iitm.ac.in/ds/course_pages/BSDA5002.html) (official 12-week structure)
- [Chandan-IISc/IITM_GenAI](https://github.com/Chandan-IISc/IITM_GenAI) (tutorial notebooks)
- Instructor posts describing chalk-and-talk lectures + PyTorch tutorials
