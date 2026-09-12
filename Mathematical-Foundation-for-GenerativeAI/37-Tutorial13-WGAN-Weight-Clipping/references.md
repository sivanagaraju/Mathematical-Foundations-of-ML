# References & Pedagogical Bridges: Tutorial 13 WGAN Weight Clipping

Welcome to the curated reference architecture for Wasserstein GAN with Weight Clipping. This document connects the mathematical lecture delivered by Prof. Prathosh A. P. to foundational research literature, sibling curriculum modules, elite university lectures, and production implementation guides.

---

## Curriculum & Prerequisite Bridges

The mathematical concepts in this lecture build directly upon optimal transport, Lipschitz analysis, and generative game theory:

- [MathsTerms: Wasserstein Distance & EMD](../../MathsTerms/05-Information-Theory-and-Divergences/05-Wasserstein_Distance_and_EMD.md)
  - **Relevance & Why Read This:** Details primal optimal transport, Kantorovich relaxation, and Monge problem geometry.
- [MathsTerms: Lipschitz Continuity](../../MathsTerms/01-Primal-Analysis-and-Foundations/04-Lipschitz_Continuity.md)
  - **Relevance & Why Read This:** Provides rigorous mathematical proofs for Lipschitz constants and gradient norm bounds.
- [MathsTerms: Minimax Game and GANs](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/09-Minimax_Game_and_GANs.md)
  - **Relevance & Why Read This:** Establishes the standard zero-sum minimax objective and Jensen-Shannon divergence derivations.
- [Lec 11: Beta-VAE](../35-Lec11-Beta-VAE/NOTES.md)
  - **Relevance & Why Read This:** Explores trade-offs in generative density fitting and distribution mismatch.

---

## Foundational & Seminal Papers

- [Arjovsky et al. (2017) Wasserstein Generative Adversarial Networks](https://arxiv.org/abs/1701.07875)
  - **What it covers:** Foundational paper introducing WGAN, Kantorovich-Rubinstein duality, 1-Lipschitz Critic, and weight clipping.
  - **Why it matters:** Proved why standard GANs suffer from vanishing gradients and established optimal transport as a viable training objective.
  - **How it maps to this lecture:** Forms the direct curriculum basis for Topics 01 through 06.
- [Villani (2008) Optimal Transport: Old and New](https://link.springer.com/book/10.1007/978-3-540-71050-9)
  - **What it covers:** The authoritative mathematical treatise on Monge-Kantorovich optimal transport and metric topologies.
  - **Why it matters:** Rigorous foundation for why Wasserstein distance induces a weaker topology that guarantees convergence.
  - **How it maps to this lecture:** Underpins the theoretical arguments in Topics 02 and 03.
- [Gulrajani et al. (2017) Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
  - **What it covers:** Identifies the pathological failures of weight clipping (capacity underuse and gradient explosion/vanishing) and introduces gradient penalty.
  - **Why it matters:** Direct chronological and conceptual successor to WGAN weight clipping.
  - **How it maps to this lecture:** Serves as the motivating bridge for Tutorial 14.

---

## University Lectures & Courses

- [Stanford CS236: Deep Generative Models — Lecture on Optimal Transport and WGAN](https://deepgenerativemodels.github.io/)
  - **What it covers:** Formal mathematical derivation of Kantorovich-Rubinstein duality and comparison between $f$-divergences and Wasserstein distance.
  - **Why it matters:** Outstanding graduate-level slides explaining why disjoint manifolds kill JS divergence.
  - **How it maps to this lecture:** Reinforces Topics 01, 02, and 03.
- [UC Berkeley CS294-158: Deep Unsupervised Learning — Optimal Transport in Generative Models](https://sites.google.com/view/berkeley-cs294-158-sp20/home)
  - **What it covers:** Practical implementation of WGAN, tuning $n_{critic}$, and evaluating sample quality via Wasserstein loss tracking.
  - **Why it matters:** Direct practical guide matching the tutorial's implementation focus.
  - **How it maps to this lecture:** Pedagogical bridge for Topics 04 and 05.

---

## Textbooks & Video Lectures

- [Computational Optimal Transport (Peyré & Cuturi, 2019)](https://optimaltransport.github.io/)
  - **What it covers:** Comprehensive algorithms for computing optimal transport, Wasserstein metrics, and Sinkhorn divergences.
  - **Why it matters:** Bridges pure measure theory with efficient machine learning implementations.
  - **How it maps to this lecture:** Reference for understanding the computational complexity of optimal transport.

---

## Industry & Implementation Guides

- [PyTorch Official WGAN Implementation Example](https://github.com/martinarjovsky/WassersteinGAN)
  - **What it covers:** Martin Arjovsky's original repository implementing WGAN with weight clipping and RMSprop in PyTorch.
  - **Why it matters:** The canonical reference code for verifying clamp operations and Critic training loops.

---

## Interactive Visualizers

- [Optimal Transport 1D & 2D Interactive Playground](https://www.optimaltransport.org/)
  - **What it covers:** Visualizes mass transport plans, showing how dirt piles are moved to target holes under minimum cost.
  - **Why it matters:** Builds immediate physical intuition for the Earth Mover's Distance.
