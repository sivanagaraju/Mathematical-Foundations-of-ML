# Start Here: A First-Principles Route into Mathematical Foundations for AI

> **Who this is for:** a learner who is new to the mathematics used in machine learning and generative AI.
>
> **What this page does:** it gives the learning order. The numbered files in this folder are a useful catalogue, but they are not, by themselves, a complete zero-background sequence.

---

## The honest starting point

Before probability, calculus, or neural networks, a learner needs a little mathematical language: arithmetic with fractions and negative numbers; powers; equations and inequalities; the coordinate plane; and the idea of a set and a function. You do **not** need measure theory, linear algebra, or programming to begin.

This knowledge base currently teaches those later topics in several sibling folders. Follow the route below rather than choosing a page only because its filename has the smallest number.

## The learning route

| Phase / Stage | Central question | Read next | Do not move on until you can… |
| :--- | :--- | :--- | :--- |
| **0. Readiness & Log-Space** | What do symbols, sets, exponentials, and logarithms mean? | [Logarithms & Exponential Functions](./02-Logarithms_and_Exponential_Functions.md) | Explain why $\ln(ab)=\ln a+\ln b$ and why $e^x>0$. |
| **1. Geometric Foundations** | How do we represent directions, coordinate systems, and distances? | [Vectors & Matrices](../02-Linear-Algebra-Geometry-and-Tensors/01-Vectors_and_Matrices.md) $\to$ [Basis & Spans](../02-Linear-Algebra-Geometry-and-Tensors/01b-Basis_Spans_and_Orthogonality.md) $\to$ [Vector Norms](../02-Linear-Algebra-Geometry-and-Tensors/02-Vector_Norms_and_Inner_Products.md) | Explain linear independence, compute a dot product, and interpret norms as distance. |
| **2. Linear Transformations** | How do matrices warp space, scale volume, and stretch along special axes? | [Determinants & Volume Scaling](../02-Linear-Algebra-Geometry-and-Tensors/01c-Determinants_and_Volume_Scaling.md) $\to$ [Eigenvalues & Eigenvectors](../02-Linear-Algebra-Geometry-and-Tensors/05b-Eigenvalues_and_Eigenvectors.md) $\to$ [SVD](../02-Linear-Algebra-Geometry-and-Tensors/06-Singular_Value_Decomposition.md) | State what $\det = 0$ means geometrically, solve $Ax = \lambda x$, and explain SVD. |
| **3. Calculus & Optimization** | How do outputs react to input changes, and how do models learn? | [Functions & Derivatives](../03-Multivariate-Calculus-and-Optimization/01-Functions_Derivatives_and_Rules.md) $\to$ [Gradients & Jacobians](../03-Multivariate-Calculus-and-Optimization/02-Derivatives_Gradients_and_Jacobians.md) $\to$ [Hessian & Curvature](../03-Multivariate-Calculus-and-Optimization/02b-Hessian_Matrix_and_Curvature.md) $\to$ [Backprop](../03-Multivariate-Calculus-and-Optimization/04-Chain_Rule_and_Backpropagation.md) $\to$ [Gradient Descent](../03-Multivariate-Calculus-and-Optimization/09-Gradient_Descent.md) | Distinguish gradients from Hessians, identify saddle points, and trace reverse-mode backpropagation. |
| **4. Probability & Uncertainty** | How do we model randomness, update beliefs, and estimate expectations? | [Probability Basics & Axioms](./01-Probability_Basics_and_Axioms.md) $\to$ [Random Variables](../04-Probability-and-Statistical-Estimation/01-Random_Variables_and_Distributions.md) $\to$ [Law of Large Numbers & Monte Carlo](../04-Probability-and-Statistical-Estimation/01b-Law_of_Large_Numbers_and_Monte_Carlo.md) $\to$ [Joint/Conditional & Bayes](../04-Probability-and-Statistical-Estimation/03-Joint_Marginal_Conditional_Dist.md) $\to$ [MLE & NLL](../04-Probability-and-Statistical-Estimation/05-MLE.md) | Calculate conditional probabilities, explain why mini-batch SGD works via LLN, and derive MLE. |
| **5. Information & Divergences** | How do we quantify surprise, mismatch between distributions, and distance between shapes? | [Entropy & Cross-Entropy](../05-Information-Theory-and-Divergences/01-Entropy_CrossEntropy_CCE.md) $\to$ [KL Divergence](../05-Information-Theory-and-Divergences/02-KL_Divergence.md) $\to$ [Wasserstein Distance](../05-Information-Theory-and-Divergences/05-Wasserstein_Distance_and_EMD.md) | Calculate Shannon entropy by hand, explain mode covering vs mode dropping in KL, and explain optimal transport. |
| **6. Advanced Analysis & Duality** | When is optimization well-behaved, and how do supporting lines represent curves? | [Convexity & Jensen's](./03-Convexity_and_Jensens_Inequality.md) $\to$ [Bounds & Supremum](./04-Bounds_Supremum_Infimum_and_Linear_Families.md) $\to$ [Fenchel Conjugate & Duality](./05-Fenchel_Conjugate_and_Dual_Representations.md) $\to$ [Lipschitz Continuity](./06-Lipschitz_Continuity.md) | Recognize a convex function, state Jensen's inequality in words, compute a Fenchel dual, and explain 1-Lipschitz critics in WGAN. |
| **7. Deep Generative Models** | How do all these tools construct modern generative architectures? | [Autoencoders](../06-Deep-Architectures-and-Generative-Models/03-Autoencoders_and_Latent_Spaces.md) $\to$ [ELBO & VAEs](../06-Deep-Architectures-and-Generative-Models/07-ELBO_and_Variational_Inference.md) $\to$ [Reparameterization Trick](../06-Deep-Architectures-and-Generative-Models/08-Reparameterization_Trick.md) $\to$ [GANs](../06-Deep-Architectures-and-Generative-Models/09-Minimax_Game_and_GANs.md) $\to$ [Autoregressive Models](../06-Deep-Architectures-and-Generative-Models/04-Autoregressive_Models.md) | Trace the derivation of the ELBO, explain why reparameterization enables backprop, and analyze the GAN minimax saddle. |

## How to use a chapter

For a first pass, read the chapter's opening, notation decoder, worked example, and self-check. Then return for the proof and AI application. The AI material is the destination: it should make the mathematics useful, not be a barrier to learning the definition.

When a page links to an advanced term you do not yet know, use the route above. Skipping forward is fine for curiosity; it is not required for mastery.

## What these ideas eventually unlock

The route leads from basic numerical reasoning to the tools behind likelihoods in LLMs, probability models in diffusion, variational objectives in VAEs, $f$-GAN duality, and the Lipschitz critics used in Wasserstein GANs. Each later application depends on the earlier mathematical object; none needs to be memorized before the object makes sense on its own.
