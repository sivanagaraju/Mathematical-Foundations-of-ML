# Curated Academic References: Diffusion Models Part 4 (Score Theory & Latent Diffusion)

## Tier 1: Foundational Papers
1. **Score-Based Generative Modeling through Stochastic Differential Equations** (Song et al., 2021)  
   *Significance*: Groundbreaking paper unifying DDPM and score matching under continuous reverse SDEs and Probability Flow ODEs.  
   *URL*: https://arxiv.org/abs/2011.13456
2. **High-Resolution Image Synthesis with Latent Diffusion Models** (Rombach et al., 2022)  
   *Significance*: Introduced Latent Diffusion Models (Stable Diffusion), decoupling perceptual compression from semantic synthesis.  
   *URL*: https://arxiv.org/abs/2112.10752
3. **An Estimate of the Scale of a Mixture of Normal Distributions** (Tweedie, 1956)  
   *Significance*: Seminal statistical paper establishing Tweedie's formula for Gaussian empirical Bayes estimation.

## Tier 2: Top Academic Lectures
1. **Stanford CS236: Deep Generative Models - Lecture 15: Continuous SDEs and Score Modeling** (Stefano Ermon)  
   *Significance*: Complete whiteboard derivations of VP and VE SDEs, Itô calculus, and Anderson's reverse SDE theorem.  
   *URL*: https://deepgenerativemodels.github.io/
2. **MIT 6.S978: Deep Generative Models - Latent Diffusion Architecture**  
   *Significance*: In-depth analysis of VAE latent space properties and cross-attention text conditioning.  
   *URL*: https://diffusion.csail.mit.edu/
3. **UC Berkeley CS294-158: Deep Unsupervised Learning - Lecture 9: Score-Based Generative Modeling**  
   *Significance*: Rigorous treatment of Probability Flow ODEs and continuous-time Langevin dynamics.  
   *URL*: https://rail.eecs.berkeley.edu/deepunsupervisedlearning/

## Tier 3: Classic Textbooks & Monograms
1. **Stochastic Differential Equations: An Introduction with Applications** (Bernt Øksendal)  
   *Significance*: Comprehensive coverage of Itô integrals, continuous martingales, and diffusion processes.
2. **Nonparametric Empirical Bayes Methods** (Robbins, 1956)  
   *Significance*: Foundational statistical theory underlying empirical Bayes and Tweedie's estimator.

## Tier 4: Industry & Code Repositories
1. **CompVis Latent Diffusion Repository** (Rombach et al.)  
   *Significance*: Official implementation of Latent Diffusion Models and KL-VAE autoencoders.  
   *URL*: https://github.com/CompVis/latent-diffusion
2. **Song Score SDE PyTorch Repository** (Yang Song)  
   *Significance*: Reference implementation of continuous SDE solvers (Euler-Maruyama, Runge-Kutta).  
   *URL*: https://github.com/yang-song/score_sde_pytorch
