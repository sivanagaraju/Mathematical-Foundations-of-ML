# Curated Academic References: Introduction to Diffusion Models

## Tier 1: Foundational Papers
1. **Deep Unsupervised Learning using Nonequilibrium Thermodynamics** (Sohl-Dickstein et al., 2015)  
   *Significance*: Seminal paper introducing the concept of diffusion models by inverting a Markov diffusion process that gradually destroys data distribution.  
   *URL*: https://arxiv.org/abs/1503.03585
2. **Denoising Diffusion Probabilistic Models (DDPM)** (Ho, Jain, & Abbeel, 2020)  
   *Significance*: Established the modern formulation connecting diffusion models to score-based generative models and denoising score matching with simplified MSE loss.  
   *URL*: https://arxiv.org/abs/2006.11239
3. **Improved Denoising Diffusion Probabilistic Models** (Nichol & Dhariwal, 2021)  
   *Significance*: Introduced the cosine noise schedule, learned reverse variance $\Sigma_\theta$, and showed that DDPMs can achieve competitive log-likelihoods.  
   *URL*: https://arxiv.org/abs/2102.09672

## Tier 2: Top Academic Lectures
1. **Stanford CS236: Deep Generative Models - Lecture 11: Diffusion Models** (Stefano Ermon)  
   *Significance*: In-depth exploration of score matching, SDE formulations, and connection between SML and DDPM.  
   *URL*: https://deepgenerativemodels.github.io/
2. **MIT 6.S978: Deep Generative Models - Diffusion Models**  
   *Significance*: Rigorous treatment of continuous-time Langevin dynamics and stochastic differential equations.  
   *URL*: https://diffusion.csail.mit.edu/
3. **UC Berkeley CS294-158: Deep Unsupervised Learning** (Pieter Abbeel)  
   *Significance*: Excellent derivations of the variational lower bound and comparison with autoregressive and flow models.  
   *URL*: https://rail.eecs.berkeley.edu/deepunsupervisedlearning/

## Tier 3: Classic Textbooks & Monograms
1. **Stochastic Differential Equations: An Introduction with Applications** (Bernt Øksendal)  
   *Significance*: Mathematical authority on Brownian motion, Itô calculus, and reverse diffusion SDEs.
2. **Information Theory, Inference, and Learning Algorithms** (David J.C. MacKay)  
   *Significance*: Comprehensive coverage of Markov chains, Gaussian processes, and variational inference.

## Tier 4: Industry & Code Repositories
1. **HuggingFace Diffusers Library**  
   *Significance*: State-of-the-art implementation of diffusion pipelines, schedulers, and U-Net backbones.  
   *URL*: https://github.com/huggingface/diffusers
2. **Lucidrains Denoising Diffusion PyTorch**  
   *Significance*: Minimalist, highly readable PyTorch implementation of DDPM and 1D/2D diffusion models.  
   *URL*: https://github.com/lucidrains/denoising-diffusion-pytorch
