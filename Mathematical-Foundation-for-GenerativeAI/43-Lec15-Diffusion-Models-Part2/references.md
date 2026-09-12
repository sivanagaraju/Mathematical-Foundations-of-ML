# Curated Academic References: Diffusion Models Part 2 (Reparameterization & Sampling)

## Tier 1: Foundational Papers
1. **Denoising Diffusion Probabilistic Models (DDPM)** (Ho, Jain, & Abbeel, 2020)  
   *Significance*: Introduced the noise prediction parameterization $\epsilon_\theta$, $L_{\text{simple}}$, and Algorithm 2 (Ancestral Sampling).  
   *URL*: https://arxiv.org/abs/2006.11239
2. **Generative Modeling by Estimating Gradients of the Data Distribution** (Song & Ermon, 2019)  
   *Significance*: Established the deep mathematical connection between denoising score matching and Langevin dynamics.  
   *URL*: https://arxiv.org/abs/1907.05600
3. **Score-Based Generative Modeling through Stochastic Differential Equations** (Song et al., 2021)  
   *Significance*: Unified DDPM and SGM under the continuous framework of reverse-time SDEs.  
   *URL*: https://arxiv.org/abs/2011.13456

## Tier 2: Top Academic Lectures
1. **Stanford CS236: Deep Generative Models - Lecture 13: Reparameterization and Score Connections** (Stefano Ermon)  
   *Significance*: Whiteboard proof showing the algebraic reduction from Gaussian KL divergence to unweighted MSE.  
   *URL*: https://deepgenerativemodels.github.io/
2. **MIT 6.S978: Deep Generative Models - DDPM to Score Matching**  
   *Significance*: Thorough analysis of Tweedie's formula and the connection between score and noise prediction.  
   *URL*: https://diffusion.csail.mit.edu/
3. **UC Berkeley CS294-158: Deep Unsupervised Learning - Lecture 7: Diffusion Implementation Details**  
   *Significance*: Detailed discussion of why $L_{\text{simple}}$ outperforms exact variational weights.  
   *URL*: https://rail.eecs.berkeley.edu/deepunsupervisedlearning/

## Tier 3: Classic Textbooks & Monograms
1. **Probabilistic Machine Learning: Advanced Topics** (Kevin P. Murphy)  
   *Significance*: Chapter 24 details ancestral sampling, Tweedie's formula, and Langevin dynamics.
2. **Deep Learning** (Ian Goodfellow, Yoshua Bengio, & Aaron Courville)  
   *Significance*: Chapter 14 covers Denoising Autoencoders and score estimation.

## Tier 4: Industry & Code Repositories
1. **OpenAI Improved Diffusion Codebase**  
   *Significance*: Production implementation of DDPM with exact sampling and $L_{\text{simple}}$.  
   *URL*: https://github.com/openai/improved-diffusion
2. **HuggingFace Diffusers DDPMPipeline**  
   *Significance*: Industry standard inference pipeline executing ancestral sampling loops.  
   *URL*: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddpm/pipeline_ddpm.py
