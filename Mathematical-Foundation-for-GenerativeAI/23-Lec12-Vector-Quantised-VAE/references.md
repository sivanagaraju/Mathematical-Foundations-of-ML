# References & Pedagogical Bridges: Lec 12 Vector Quantised VAE

Welcome to the curated reference architecture for Vector Quantised VAE (VQ-VAE) and discrete latent representations. This document connects the mathematical lecture delivered by Prof. Prathosh A. P. to foundational research literature, sibling curriculum modules, elite university lectures, and production implementation guides.

---

## Curriculum & Prerequisite Bridges

The mathematical concepts in this lecture build directly upon prior probability, optimization, and generative representation modules:

- [Lec 10: VAEs Part 2](../21-Lec10-VAEs-Part2/NOTES.md)
  - **Relevance & Why Read This:** Details continuous Gaussian latent bottlenecks, the reparameterization trick, and posterior collapse pathologies that VQ-VAE is specifically engineered to overcome.
- [Lec 11: Beta-VAE](../22-Lec11-Beta-VAE/NOTES.md)
  - **Relevance & Why Read This:** Analyzes the information bottleneck Lagrangian trade-off between reconstruction fidelity and latent regularization pressure.
- [MathsTerms: Encodings Categorical & Embeddings](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/08-Encodings_Categorical_and_Embeddings.md)
  - **Relevance & Why Read This:** Formulates vector lookup matrices, categorical indexing, and embedding layer gradient routing.
- [MathsTerms: Exponential Moving Average](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/10-Exponential_Moving_Average_EMA.md)
  - **Relevance & Why Read This:** Establishes the mathematical tracking mechanics behind optimizer-free codebook updates.

---

## Foundational & Seminal Papers

The theoretical derivation of discrete latent variables and straight-through estimation is detailed in the seminal papers below:

- [van den Oord et al. (2017) Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)
  - **What it covers:** The foundational paper introducing VQ-VAE, vector quantization via codebooks, the straight-through estimator (STE) for discrete bottlenecks, and two-stage image/audio generation with PixelCNN priors.
  - **Why it matters:** Proved that discrete latent spaces bypass posterior collapse without sacrificing representation quality, matching continuous generative models.
  - **How it maps to this lecture:** Forms the direct curriculum basis for Topics 01 through 06.
- [Razavi et al. (2019) Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446)
  - **What it covers:** Multi-scale hierarchical discrete latent maps (top/bottom latent grids) and multi-level autoregressive self-attention priors generating 1024x1024 photo-realistic images.
  - **Why it matters:** Demonstrated that hierarchical discrete codes match or exceed BigGAN in sample quality while maintaining sample diversity.
  - **How it maps to this lecture:** Deepens Topic 06 by scaling the two-stage discrete generative architecture to deep pyramids.
- [Bengio et al. (2013) Estimating or Propagating Gradients Through Stochastic Neurons](https://arxiv.org/abs/1308.3432)
  - **What it covers:** Formal mathematical grounding of the Straight-Through Estimator (STE) heuristic across discrete thresholding and quantization operators.
  - **Why it matters:** Provides theoretical justification for why copying gradient signals across piecewise-constant boundaries yields valid optimization steps.
  - **How it maps to this lecture:** Crucial theoretical anchor for Topic 03.
- [Esser et al. (2021) Taming Transformers for High-Resolution Image Synthesis (VQGAN)](https://arxiv.org/abs/2012.09841)
  - **What it covers:** Combines VQ-VAE discrete codebooks with perceptual patch losses and adversarial discriminators, replacing PixelCNN with an autoregressive Transformer.
  - **Why it matters:** The direct predecessor to Stable Diffusion / Latent Diffusion Models (LDMs), showing how codebooks compress visual perceptual redundancy.
  - **How it maps to this lecture:** Bridges discrete autoencoding with modern generative foundation models.

---

## University Lectures & Courses

Top-tier institutional curricula covering discrete latent variables, vector quantization, and deep generative architectures:

- [Stanford CS236: Deep Generative Models — Lecture on Discrete Latent Variable Models](https://deepgenerativemodels.github.io/)
  - **What it covers:** Mathematical comparison between continuous Gaussian latent models (ELBO) and discrete variable models, addressing Gumbel-Softmax vs Vector Quantization.
  - **Why it matters:** Provides rigorous academic treatment of gradient estimation under non-differentiable bottlenecks.
  - **How it maps to this lecture:** Reinforces Topics 01 and 03.
- [UC Berkeley CS285: Deep Reinforcement Learning & Generative Models](https://rail.eecs.berkeley.edu/deeprlcourse/)
  - **What it covers:** Latent state discretization, model-based planning with discrete tokens, and trajectory quantization.
  - **Why it matters:** Connects VQ representations to modern tokenized world models (e.g., DreamerV3).
  - **How it maps to this lecture:** Broadens the perspective on discrete bottlenecks beyond vision to dynamics modeling.
- [MIT 6.S191: Introduction to Deep Learning — Generative Models](http://introtodeeplearning.com/)
  - **What it covers:** Accessible high-level overview of autoencoders, VAEs, GANs, and discrete codebook tokenization.
  - **Why it matters:** Excellent visual intuitions for codebook lookups and straight-through gradient approximations.
  - **How it maps to this lecture:** Pedagogical bridge for Topic 02.

---

## Textbooks & Video Lectures

- [DeepMind x UCL Deep Learning Lecture Series: Generative Models](https://www.youtube.com/playlist?list=PLqYmG7hTraZCDxZ44o4p3N5Anz3lLRVZF)
  - **What it covers:** Detailed technical overview delivered by DeepMind researchers on VQ-VAE, PixelCNN autoregression, and WaveNet audio tokenization.
  - **Why it matters:** Direct firsthand explanation from the research team that developed VQ-VAE.
  - **How it maps to this lecture:** Provides real-world historical context for topics discussed in the lecture.
- [Probabilistic Machine Learning: Advanced Topics (Kevin P. Murphy, MIT Press 2023)](https://probml.github.io/prml-book/)
  - **What it covers:** Chapter 24 provides an exhaustive mathematical analysis of discrete latent models, straight-through estimators, and vector quantization geometry.
  - **Why it matters:** Rigorous textbook reference for graduate machine learning theory.
  - **How it maps to this lecture:** Foundational reading for Topics 02, 03, and 04.

---

## Industry & Implementation Guides

- [PyTorch Official Examples: VQ-VAE](https://github.com/pytorch/examples/tree/main/vae)
  - **What it covers:** Canonical PyTorch implementation of VectorQuantizer module, straight-through gradient detach mechanics, and training loop on CIFAR-10.
  - **Why it matters:** Verified baseline code for production deployment.
- [Lucidrains Vector Quantize PyTorch](https://github.com/lucidrains/vector-quantize-pytorch)
  - **What it covers:** Production-grade modular vector quantization library including EMA updates, cosine similarity codebooks, and dead code reset algorithms.
  - **Why it matters:** SOTA industrial reference used across modern audio and video foundation models.

---

## Interactive Visualizers

- [OpenAI Jukebox & DALL-E Codebook Interactive Explorer](https://openai.com/research/jukebox)
  - **What it covers:** Interactive demonstration of how discrete codebook indices represent multi-instrument musical audio and hierarchical image tokens.
  - **Why it matters:** Builds visceral intuition for how continuous signals map onto discrete vocabularies.
