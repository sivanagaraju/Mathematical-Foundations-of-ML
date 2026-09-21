# 🗺️ MathsTerms Curriculum Restructure Plan: The 6-Pillar Architecture

> `📅 Date:` September 2026  
> `🎯 Objective:` Reorganize the entire `MathsTerms/` curriculum based on pedagogical principles from Tivadar Danka's *The Roadmap of Mathematics for Machine Learning* and eliminate the cognitive dissonance of placing advanced analysis topics in Chapter 1.

---

## 1. Executive Summary & Root Cause Analysis

In the original repository layout, **Convexity & Jensen's Inequality**, **Bounds & Supremum**, **Fenchel Conjugate**, and **Lipschitz Continuity** were physically placed in the very first folder (`01-Primal-Analysis-and-Foundations/`).

### Why This Caused Cognitive Friction:
1. **Pedagogical Inversion:** A zero-math beginner opening `01` was greeted with Legendre-Fenchel duality, infimum/supremum bounds, and Lipschitz constants before even learning what a vector, a matrix, or a derivative is.
2. **True Downstream Purpose:** These four topics actually belong to **Phase 6** (Advanced Analysis & Duality) and exist strictly to serve **Phase 7** Generative AI architectures:
   - *Jensen's Inequality* $\to$ ELBO derivation in Variational Autoencoders (VAEs).
   - *Fenchel Conjugates* $\to$ Variational Divergence Minimization ($f$-GANs).
   - *Lipschitz Continuity* $\to$ Wasserstein GAN (WGAN-GP) 1-Lipschitz critics and Spectral Normalization.
3. **Stranded Foundations:** `01-Probability_Basics_and_Axioms.md` and `02-Logarithms_and_Exponential_Functions.md` were stranded in folder 01, detached from their natural homes in Probability and Calculus.

---

## 2. The Approved 6-Pillar Architecture (Proposal A)

The curriculum is structured into 6 sequential pillars that directly mirror human learning:

```text
===================================================================================================
                       THE 6-PILLAR PEDAGOGICAL STRUCTURE
===================================================================================================

  📁 01-Linear-Algebra-Geometry-and-Tensors (Phases 1 & 2)
     ├── 01-Vectors_and_Matrices.md
     ├── 01b-Basis_Spans_and_Orthogonality.md              [FOUNDATIONAL GUIDE]
     ├── 01c-Determinants_and_Volume_Scaling.md           [FOUNDATIONAL GUIDE]
     ├── 02-Vector_Norms_and_Inner_Products.md
     ├── 03-Dot_Product_and_Similarity.md
     ├── 04-Tensors_and_Shapes.md
     ├── 05-Tensor_Broadcasting.md
     ├── 05b-Eigenvalues_and_Eigenvectors.md               [FOUNDATIONAL GUIDE]
     ├── 06-Singular_Value_Decomposition.md
     ├── 07-One_Hot_Encoding.md
     ├── 08-Encodings_Categorical_and_Embeddings.md
     └── 09-Positional_Encodings.md

  📁 02-Multivariate-Calculus-and-Optimization (Phase 3)
     ├── 00-Logarithms_and_Exponential_Functions.md        [RELOCATED FROM 01]
     ├── 01-Functions_Derivatives_and_Rules.md
     ├── 02-Derivatives_Gradients_and_Jacobians.md
     ├── 02b-Hessian_Matrix_and_Curvature.md               [FOUNDATIONAL GUIDE]
     ├── 03-Jacobian_Matrix.md
     ├── 04-Chain_Rule_and_Backpropagation.md
     ├── 05-Activation_Functions.md
     ├── 06-Softmax.md
     ├── 07-Argmax.md
     ├── 08-Loss_Functions.md
     ├── 09-Gradient_Descent.md
     ├── 10-Exponential_Moving_Average_EMA.md
     └── 11-Batch_Normalization_and_Spectral_Norm.md

  📁 03-Probability-and-Statistical-Estimation (Phase 4)
     ├── 00-Probability_Basics_and_Axioms.md               [RELOCATED FROM 01]
     ├── 01-Random_Variables_and_Distributions.md
     ├── 01b-Law_of_Large_Numbers_and_Monte_Carlo.md       [FOUNDATIONAL GUIDE]
     ├── 02-Common_Probability_Distributions.md
     ├── 03-Joint_Marginal_Conditional_Dist.md
     ├── 04-Likelihood_and_Log_Likelihood.md
     ├── 05-MLE.md
     ├── 06-NLL.md
     └── 07-LOTUS_and_Empirical_Expectation_Estimation.md

  📁 04-Information-Theory-and-Divergences (Phase 5)
     ├── 01-Entropy_CrossEntropy_CCE.md
     ├── 02-KL_Divergence.md
     ├── 03-Jensen_Shannon_Divergence.md
     ├── 04-f_Divergence.md
     ├── 05-Wasserstein_Distance_and_EMD.md
     ├── 06-Variational_Divergence_Minimization_VDM.md
     └── 07-Joint_Conditional_Entropy_and_Mutual_Information.md

  📁 05-Convexity-Duality-and-Metric-Analysis (Phase 6)    [NEW DEDICATED FOLDER]
     ├── 01-Convexity_and_Jensens_Inequality.md            [RELOCATED FROM 01]
     ├── 02-Bounds_Supremum_Infimum_and_Linear_Families.md [RELOCATED FROM 01]
     ├── 03-Fenchel_Conjugate_and_Dual_Representations.md  [RELOCATED FROM 01]
     └── 04-Lipschitz_Continuity.md                        [RELOCATED FROM 01]

  📁 06-Deep-Architectures-and-Generative-Models (Phase 7)
     ├── 01-Convolution_and_Pooling.md
     ├── 02-Recurrent_Neural_Networks.md
     ├── 03-Autoencoders_and_Latent_Spaces.md
     ├── 04-Autoregressive_Models.md
     ├── 05-Latent_Variable_Models.md
     ├── 06-Expectation_Maximization_Algorithm.md
     ├── 07-ELBO_and_Variational_Inference.md
     ├── 08-Reparameterization_Trick.md
     ├── 09-Minimax_Game_and_GANs.md
     └── 10-Frechet_Inception_Distance.md
===================================================================================================
```

---

## 3. Migration Milestones

1. **Phase 1: Foundational Authoring**
   - Author 5 missing foundational guides to bridge critical gaps identified in the roadmap:
     - `01b-Basis_Spans_and_Orthogonality.md`
     - `01c-Determinants_and_Volume_Scaling.md`
     - `05b-Eigenvalues_and_Eigenvectors.md`
     - `02b-Hessian_Matrix_and_Curvature.md`
     - `01b-Law_of_Large_Numbers_and_Monte_Carlo.md`
2. **Phase 2: Physical Relocation**
   - Create `05-Convexity-Duality-and-Metric-Analysis/`.
   - Move Convexity, Bounds, Fenchel, and Lipschitz guides into `05-`.
   - Move Probability Axioms into `03-`.
   - Move Log/Exp into `02-`.
   - Rename folders to 6-pillar sequence and remove obsolete `01-Primal-Analysis-and-Foundations`.
3. **Phase 3: Automated Link Refactoring**
   - Execute regex-based global link replacement across all markdown files in the repository.
4. **Phase 4: Verification & Integrity**
   - Run verification script to guarantee zero broken relative links across all 54 guides.
