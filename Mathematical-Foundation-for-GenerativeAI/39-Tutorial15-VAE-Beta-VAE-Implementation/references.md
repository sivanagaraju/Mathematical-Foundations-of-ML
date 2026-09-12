# References & Pedagogical Bridges: Tutorial 15 VAE & Beta-VAE Implementation

Welcome to the curated reference architecture for VAE and Beta-VAE Implementation in PyTorch. This document connects the technical implementation tutorial delivered by Prof. Prathosh A. P. to foundational research literature, sibling curriculum modules, elite university lectures, and production implementation guides.

---

## Curriculum & Prerequisite Bridges

The mathematical concepts in this tutorial build directly upon variational inference, probabilistic modeling, and neural autoencoders:

- [Lec 10: VAEs Part 2](../34-Lec10-VAEs-Part2/NOTES.md)
  - **Relevance & Why Read This:** Derives the variational evidence lower bound (ELBO), Gaussian assumption equivalence to MSE, and the reparameterization trick.
- [Lec 11: Beta-VAE](../35-Lec11-Beta-VAE/NOTES.md)
  - **Relevance & Why Read This:** Theoretical lecture formulating the $\beta$-weighted information bottleneck and posterior collapse dynamics.
- [MathsTerms: Reparameterization Trick](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/08-Reparameterization_Trick.md)
  - **Relevance & Why Read This:** Formal mathematical treatment of Jacobian determinants and stochastic node gradient routing.
- [MathsTerms: ELBO & Variational Inference](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/07-ELBO_and_Variational_Inference.md)
  - **Relevance & Why Read This:** Exhaustive derivations of Jensen's inequality applied to marginal log-likelihoods.

---

## Foundational & Seminal Papers

- [Kingma & Welling (2013) Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
  - **What it covers:** Foundational paper introducing the Variational Autoencoder, Stochastic Gradient Variational Bayes (SGVB) estimator, and the reparameterization trick.
  - **Why it matters:** The seminal genesis of modern deep latent variable modeling.
  - **How it maps to this lecture:** Forms the direct implementation blueprint for Topics 01, 02, and 03.
- [Higgins et al. (2017) beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
  - **What it covers:** Introduces the tunable $\beta$ hyperparameter to encourage unsupervised discovery of disentangled generative visual features.
  - **Why it matters:** Established disentanglement benchmarking on dSprites and CelebA datasets.
  - **How it maps to this lecture:** Direct basis for Topics 04 and 05.
- [Burgess et al. (2018) Understanding Computing and Disentangling in beta-VAE](https://arxiv.org/abs/1804.03599)
  - **What it covers:** Capacity annealing schedules $|D_{KL} - C|$, controlling progressive information channel capacity during training.
  - **Why it matters:** Provides practical recipes to avoid early posterior collapse in $\beta$-VAE.
  - **How it maps to this lecture:** Informs the training dynamics and warmup schedules discussed in Topic 06.

---

## University Lectures & Courses

- [Stanford CS236: Deep Generative Models — Lecture 5: Variational Autoencoders](https://deepgenerativemodels.github.io/)
  - **What it covers:** Prof. Stefano Ermon's complete derivation of the ELBO, amortized inference, and reparameterization mechanics.
  - **Why it matters:** Top-tier graduate level treatment of latent variable models.
  - **How it maps to this lecture:** Reinforces Topics 02, 03, and 04.
- [MIT 6.S191: Introduction to Deep Learning — Deep Generative Modeling](http://introtodeeplearning.com/)
  - **What it covers:** Intuitive visual explanations of latent space geometry, manifold traversals, and sample synthesis.
  - **Why it matters:** Great visual intuition for understanding latent dimension traversals.
  - **How it maps to this lecture:** Pedagogical bridge for Topic 05.

---

## Textbooks & Video Lectures

- [Deep Learning (Goodfellow, Bengio, Courville, MIT Press 2016)](https://www.deeplearningbook.org/)
  - **What it covers:** Chapter 20 provides foundational coverage of deep generative models, variational approximations, and directed latent models.
  - **Why it matters:** The definitive standard reference textbook for deep learning theory.
  - **How it maps to this lecture:** Foundational reading for Topics 01 and 02.

---

## Industry & Implementation Guides

- [PyTorch-VAE: Modular Implementations of Generative Models](https://github.com/AntixK/PyTorch-VAE)
  - **What it covers:** Production-ready PyTorch implementations of Vanilla VAE, Beta-VAE, FactorVAE, and WAE.
  - **Why it matters:** Gold-standard industrial codebase for verifying modular class designs and loss functions.
- [HuggingFace Diffusers & Generative Modeling Hub](https://github.com/huggingface/diffusers)
  - **What it covers:** Modern production convolutional autoencoders used as visual tokenizers in Stable Diffusion and FLUX.
  - **Why it matters:** Direct practical application showing how VAEs serve as perceptual compressors for foundation models.

---

## Interactive Visualizers

- [Latent Space Explorer: Interactive WebGL VAE Demo](https://magenta.tensorflow.org/assets/sketch_rnn_demo/index.html)
  - **What it covers:** Real-time browser slider manipulation across latent coordinates to observe continuous stroke generation.
  - **Why it matters:** Direct hands-on demonstration of latent traversal principles.
