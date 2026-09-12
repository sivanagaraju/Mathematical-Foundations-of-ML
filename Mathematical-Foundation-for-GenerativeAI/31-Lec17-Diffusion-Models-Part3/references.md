# Curated Academic References: Diffusion Models Part 3 (Architecture & Guidance)

## Tier 1: Foundational Papers
1. **Diffusion Models Beat GANs on Image Synthesis** (Dhariwal & Nichol, 2021)  
   *Significance*: Introduced Classifier Guidance and established U-Net architectural improvements (attention heads, AdaGN).  
   *URL*: https://arxiv.org/abs/2105.05233
2. **Classifier-Free Diffusion Guidance** (Ho & Salimans, 2021)  
   *Significance*: Groundbreaking paper introducing CFG, eliminating external classifiers and enabling modern text-to-image synthesis.  
   *URL*: https://arxiv.org/abs/2207.12598
3. **Denoising Diffusion Implicit Models (DDIM)** (Song, Meng, & Ermon, 2020)  
   *Significance*: Formulated non-Markovian deterministic sampling, enabling 10x faster generation and exact latent inversion.  
   *URL*: https://arxiv.org/abs/2010.02502

## Tier 2: Top Academic Lectures
1. **Stanford CS236: Deep Generative Models - Lecture 14: Guided Diffusion and Conditioning** (Stefano Ermon)  
   *Significance*: Mathematical derivation of Bayes decomposition for classifier gradients and CFG extrapolation.  
   *URL*: https://deepgenerativemodels.github.io/
2. **MIT 6.S978: Deep Generative Models - Controllable Generation in Diffusion**  
   *Significance*: Analysis of score guidance, text conditioning, and cross-attention mechanisms.  
   *URL*: https://diffusion.csail.mit.edu/
3. **UC Berkeley CS294-158: Deep Unsupervised Learning - Lecture 8: Controllable Diffusion Models**  
   *Significance*: Detailed treatment of U-Net vs DiT backbones and guidance scale tradeoffs.  
   *URL*: https://rail.eecs.berkeley.edu/deepunsupervisedlearning/

## Tier 3: Classic Textbooks & Monograms
1. **Deep Learning** (Ian Goodfellow, Yoshua Bengio, & Aaron Courville)  
   *Significance*: Chapter 9 covers Convolutional Networks and U-Net skip connection principles.
2. **Probabilistic Machine Learning: Advanced Topics** (Kevin P. Murphy)  
   *Significance*: Chapter 24 details score-based conditioning and classifier guidance math.

## Tier 4: Industry & Code Repositories
1. **OpenAI Guided Diffusion Repository** (Dhariwal & Nichol)  
   *Significance*: Original implementation of classifier guidance and improved U-Net models.  
   *URL*: https://github.com/openai/guided-diffusion
2. **CompVis Stable Diffusion Repository** (Rombach et al.)  
   *Significance*: Flagship open-source implementation of Classifier-Free Guidance with cross-attention.  
   *URL*: https://github.com/CompVis/stable-diffusion
