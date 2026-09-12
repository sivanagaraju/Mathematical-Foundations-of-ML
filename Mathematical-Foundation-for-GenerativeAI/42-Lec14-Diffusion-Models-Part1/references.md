# Curated Academic References: Diffusion Models Part 1 (ELBO & Posterior)

## Tier 1: Foundational Papers
1. **Denoising Diffusion Probabilistic Models (DDPM)** (Ho, Jain, & Abbeel, 2020)  
   *Significance*: Section 3 provides the definitive derivation of the ELBO expansion and the analytical posterior mean $\tilde{\mu}_t(x_t, x_0)$.  
   *URL*: https://arxiv.org/abs/2006.11239
2. **Deep Unsupervised Learning using Nonequilibrium Thermodynamics** (Sohl-Dickstein et al., 2015)  
   *Significance*: Introduces the trajectory lower bound and thermodynamic entropy production equivalence.  
   *URL*: https://arxiv.org/abs/1503.03585
3. **Variational Inference: A Review for Statisticians** (Blei, Kucukelbir, & McAuliffe, 2017)  
   *Significance*: Foundational treatment of Jensen's inequality and evidence lower bounds in latent models.  
   *URL*: https://arxiv.org/abs/1601.00670

## Tier 2: Top Academic Lectures
1. **Stanford CS236: Deep Generative Models - Lecture 12: Variational Bounds in Diffusion** (Stefano Ermon)  
   *Significance*: Step-by-step whiteboard derivation of telescoping product cancellation and Gaussian KL divergence.  
   *URL*: https://deepgenerativemodels.github.io/
2. **UC Berkeley CS294-158: Deep Unsupervised Learning - Lecture 6: Diffusion Models** (Pieter Abbeel)  
   *Significance*: Detailed breakdown of $L_T, L_{t-1}, L_0$ and why $L_T$ has zero learnable parameters.  
   *URL*: https://rail.eecs.berkeley.edu/deepunsupervisedlearning/
3. **MIT 6.S978: Deep Generative Models - Denoising Probabilistic Foundations**  
   *Significance*: Analytical expansions of Gaussian products and completions of squares.  
   *URL*: https://diffusion.csail.mit.edu/

## Tier 3: Classic Textbooks & Monograms
1. **Pattern Recognition and Machine Learning** (Christopher M. Bishop)  
   *Significance*: Chapter 10 covers variational inference, factorization, and KL divergences for Gaussians.
2. **Probabilistic Machine Learning: Advanced Topics** (Kevin P. Murphy)  
   *Significance*: Chapter 24 provides complete mathematical coverage of diffusion models and score-based methods.

## Tier 4: Industry & Code Repositories
1. **Official DDPM TensorFlow Repository** (Ho et al.)  
   *Significance*: The reference implementation of the tripartite loss and posterior mean equations.  
   *URL*: https://github.com/hojonathanho/diffusion
2. **HuggingFace Diffusers Scheduler Codebase**  
   *Significance*: Production implementation of `DDPMScheduler` with exact $\tilde{\beta}_t$ equations.  
   *URL*: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py
