# References — Tutorial 17: Implementation Overview of Diffusion Models

## Primary Literature & Theoretical Notes
- **Calvin Luo (2022):** *"Understanding Diffusion Models: A Unified Perspective."* arXiv:2208.11970.  
  [https://arxiv.org/abs/2208.11970](https://arxiv.org/abs/2208.11970)  
  *Why Read This:* The exact primary reference reviewed in Tutorial 17. Systematically labels and derives equations 31, 45, 46, 58, 70, 93, 94, 99, 115, 128, 130, 143, and 148, proving the algebraic equivalence of all four diffusion parameterizations.
- **Jonathan Ho, Ajay Jain, Pieter Abbeel (2020):** *"Denoising Diffusion Probabilistic Models (DDPM)."* NeurIPS 2020.  
  [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)  
  *Why Read This:* The seminal paper introducing the unweighted simplified noise prediction loss and reverse ancestral sampling algorithm used across production generative AI.
- **Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, Surya Ganguli (2015):** *"Deep Unsupervised Learning using Nonequilibrium Thermodynamics."* ICML 2015.  
  [https://arxiv.org/abs/1503.03585](https://arxiv.org/abs/1503.03585)  
  *Why Read This:* The foundational paper establishing generative diffusion modeling inspired by physical nonequilibrium thermodynamics and reverse-time Markov chains.
- **Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole (2021):** *"Score-Based Generative Modeling through Stochastic Differential Equations."* ICLR 2021.  
  [https://arxiv.org/abs/2011.13456](https://arxiv.org/abs/2011.13456)  
  *Why Read This:* Provides the overarching continuous-time SDE perspective linking discrete DDPM steps to continuous stochastic vector fields and Probability Flow ODEs.

---

## Textbooks & Video Lectures
- **NPTEL — Mathematical Foundations of Generative AI (IISc Bengaluru):**  
  [Official YouTube Playlist](https://www.youtube.com/playlist?list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK)  
  *Why Watch This:* Comprehensive lecture series covering linear algebra, probability, information theory, VAEs, WGANs, and diffusion models taught by Prof. Chiranjib Bhattacharyya.
- **Stanford CS236: Deep Generative Models (Stefano Ermon):**  
  [https://deepgenerativemodels.github.io/](https://deepgenerativemodels.github.io/)  
  *Why Read This:* World-renowned university course covering score matching, autoregressive models, VAEs, and diffusion SDEs.

---

## Official Tutorial Notebook
- **NPTEL Tutorial 17 Official Colab Notebook:**  
  [https://colab.research.google.com/drive/15nVkKu1mySDHzEj4NaqDjDAUUvZsTxzR?usp=sharing](https://colab.research.google.com/drive/15nVkKu1mySDHzEj4NaqDjDAUUvZsTxzR?usp=sharing)  
  *Why Inspect This:* Direct PyTorch implementation accompanying the video, verifying tensor broadcasting, U-Net forward passes, and training loops.

---

## Curriculum & Prerequisite Bridges
- **Tutorial 15 & 16:** Implementation of VAE, $\beta$-VAE, and VQ-VAE discrete codebooks.
- **Lecture 13 — Introduction to Diffusion Models:** Conceptual intuition for forward noising and Markov chains.
- **Lecture 14 & 15 — Diffusion Models Parts 1 & 2:** Rigorous mathematical derivations of the ELBO.
- **Lecture 17 & 18 — Diffusion Models Parts 3 & 4:** Continuous-time score matching, SDEs, and Classifier-Free Guidance.

---

## Interactive Visualizers
- **Diffusion Explainer (Georgia Tech):**  
  [https://poloclub.github.io/diffusion-explainer/](https://poloclub.github.io/diffusion-explainer/)  
  *Why Use This:* Interactive web visualizer illustrating real-time reverse trajectory evolution and noise schedule decay across image pixels.
