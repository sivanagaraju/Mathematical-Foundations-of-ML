# 📚 Mathematical Terms & Generative AI Foundations Catalog

Welcome to the **Mathematical Terms (`MathsTerms`) Knowledge Base** for the *Mathematical Foundations of Generative AI* masterclass.

Every guide in this directory follows a strict **10-Section Gold Standard** bridging abstract mathematical rigor directly to intuitive physical primitives, elementary algebraic proofs, micro-numerical arithmetic, runnable PyTorch/NumPy code, and modern generative AI architectures (Diffusion Models, GANs, VAEs, Transformers, Flow Matching).

```
 ===================================================================================================
                 THE 5-POINT PEDAGOGICAL BRIDGE (STANDARD ACROSS ALL GUIDES)
 ===================================================================================================
 
  👶 ELI5 Intuition  ◄──►  🔍 Plain-English  ◄──►  🔢 Micro-Numbers  ◄──►  📐 Formal Math  ◄──►  💻 Runnable Code
  (Everyday Metaphors)     (Rosetta Stones)        (Step-by-Step Maths)    (Theorems & Proofs)     (PyTorch/NumPy)
 ===================================================================================================
```

---

## 🗺️ Master Dependency Graph & Concept Roadmap

> 🚀 **Looking to connect all the dots?**  
> Explore the comprehensive **[Grand Unified Concept Map & Mathematical Dependency Graph](./CONCEPT_MAP.md)** connecting all 46 guides from first-principles axioms to Generative AI architectures (VAEs, GANs, Diffusion Models, Autoregressive LLMs, Flow Matching), featuring interactive Mermaid diagrams, ASCII pipelines, sub-term lineage, and 4 specialized learning tracks.

---

## 🧭 Master Directory Index (46 Curated Guides)

### 1. ⚡ Deep Learning, Calculus & Optimization Core

| Mathematical Guide | Core Focus & Key Formula | Primary Modules |
| :--- | :--- | :--- |
| **[Functions, Derivatives & Rules](./Functions_Derivatives_and_Rules.md)** | Power, product, quotient rules, limit definition ($f'(x) = \lim \frac{f(x+h)-f(x)}{h}$) | Tut 03, Lec 01 |
| **[The Chain Rule & Backpropagation](./Chain_Rule_and_Backpropagation.md)** | Reverse-mode automatic differentiation, DAG computation graphs ($\frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx}$) | Tut 03, Tut 04, Lec 01 |
| **[The Jacobian Matrix](./Jacobian_Matrix.md)** | Vector-valued derivatives, local volume warping, VJP/JVP ($J_{ij} = \partial f_i / \partial x_j$) | Tut 03, Tut 06, Lec 01 |
| **[Exponential Moving Average (EMA)](./Exponential_Moving_Average_EMA.md)** | Shadow weights in Diffusion, Adam momentum ($\theta_{\text{EMA}} = \beta \theta_{\text{EMA}} + (1-\beta)\theta$) | Tut 03, Lec 01, Lec 18 |
| **[Activation Functions](./Activation_Functions.md)** | Non-linearities: ReLU, LeakyReLU, GELU, SwiGLU, Sigmoid ($\sigma(z), \text{GELU}(z)$) | Lec 01, Tut 03, Tut 04 |
| **[Argmax & Soft-Argmax](./Argmax.md)** | Non-differentiable discrete selection vs differentiable Gumbel-Softmax | Tut 03, Lec 02 |
| **[Batch Normalization & Spectral Norm](./Batch_Normalization_and_Spectral_Norm.md)** | Layer normalization & 1-Lipschitz matrix spectral norm ($\|W\|_2 \le 1, W / \sigma(W)$) | Tut 04, Lec 18, Tut 12 |
| **[Convexity & Jensen's Inequality](./Convexity_and_Jensens_Inequality.md)** | Convex functions, secant lines, and lower bounds ($f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]$) | Lec 03, Lec 20 |
| **[Convolution & Pooling Operations](./Convolution_and_Pooling.md)** | Receptive fields, spatial downsampling, transposed convolutions, DCGAN | Tut 04, Tut 12 |
| **[Derivatives, Gradients & Jacobians](./Derivatives_Gradients_and_Jacobians.md)** | Multivariable calculus, autograd computational graphs, Jacobian matrices | Tut 03, Lec 04, Lec 18 |
| **[Gradient Descent & Optimizers](./Gradient_Descent.md)** | SGD, Momentum, RMSprop, Adam first/second moment updates ($\theta \leftarrow \theta - \eta \nabla L$) | Tut 03, Lec 05, Lec 19 |
| **[Lipschitz Continuity](./Lipschitz_Continuity.md)** | Bounded gradient slope $\|f(x) - f(y)\| \le K \|x - y\|$, Kantorovich-Rubinstein dual | Lec 18, Tut 12 |
| **[Logarithms & Exponential Functions](./Logarithms_and_Exponential_Functions.md)** | Numerical stability, log-space arithmetic, preventing underflow ($\ln(ab) = \ln a + \ln b$) | Tut 02, Lec 01, Tut 10 |
| **[Loss Functions in Machine Learning](./Loss_Functions.md)** | MSE, MAE, Huber, BCE, CCE, Hinge, Wasserstein objectives | Tut 03, Tut 10, Lec 05 |
| **[Recurrent Neural Networks & Backprop Through Time](./Recurrent_Neural_Networks.md)** | Sequence processing, hidden state recurrence, vanishing gradients in BPTT | Tut 05 |
| **[Softmax Function](./Softmax.md)** | Probability normalization, logits squashing, temperature scaling ($\frac{e^{z_i/\tau}}{\sum e^{z_j/\tau}}$) | Lec 01, Tut 03, Tut 10 |

---

### 2. 📐 Linear Algebra, Vectors, Matrices & Embeddings

| Mathematical Guide | Core Focus & Key Formula | Primary Modules |
| :--- | :--- | :--- |
| **[Vectors & Matrices](./Vectors_and_Matrices.md)** | Linear transformations, coordinate bases, matrix multiplication ($y = Wx + b$) | Tut 02, Tut 03, Lec 14 |
| **[Singular Value Decomposition (SVD)](./Singular_Value_Decomposition.md)** | Rotate-Stretch-Rotate factorization, Eckart-Young theorem, LoRA ($A = U \Sigma V^T$) | Tut 06, Lec 01, Lec 14 |
| **[Similarity with Dot Product](./Dot_Product_and_Similarity.md)** | Vector projections, cosine similarity, Scaled Attention ($\vec{a} \cdot \vec{b} = \|\vec{a}\|\|\vec{b}\|\cos\theta$) | Tut 02, Tut 03, Lec 14 |
| **[Tensor Broadcasting](./Tensor_Broadcasting.md)** | Memory strides, 3 golden broadcasting rules, zero-copy expansion (`stride = 0`) | Tut 02, Tut 03 |
| **[Vector Norms & Inner Products](./Vector_Norms_and_Inner_Products.md)** | $L_1, L_2, L_\infty$ geometry, Lasso sparsity vs Ridge weight decay | Tut 02, Lec 18 |
| **[Encodings & Categorical Embeddings](./Encodings_Categorical_and_Embeddings.md)** | Discrete token IDs to continuous coordinates, BPE tokenization, embedding lookup tables | Tut 03, Lec 01 |
| **[Positional Encodings & RoPE](./Positional_Encodings.md)** | Sequence order, Sinusoidal, Learned, ALiBi, and Rotary Position Embeddings in LLMs | Tut 03, Lec 01 |
| **[One-Hot Encoding](./One_Hot_Encoding.md)** | Sparse categorical representation, cross-entropy target vectors | Lec 01, Tut 10 |
| **[Tensors, Shapes & Dimensional Broadcasting](./Tensors_and_Shapes.md)** | Multidimensional arrays, PyTorch strided layouts, broadcasting arithmetic | Tut 02, Tut 03 |

---

### 3. 🎲 Probability, Information Theory & Divergences

| Mathematical Guide | Core Focus & Key Formula | Primary Modules |
| :--- | :--- | :--- |
| **[Common Probability Distributions](./Common_Probability_Distributions.md)** | Gaussian, Uniform, Bernoulli, Categorical, multivariate covariance matrices | Tut 07, Tut 08, Lec 18 |
| **[Entropy, Cross-Entropy & Categorical Cross-Entropy](./Entropy_CrossEntropy_CCE.md)** | Shannon surprise, cross-entropy loss, negative log-likelihood ($H(P, Q) = -\sum p \log q$) | Lec 01, Tut 10 |
| **[$f$-Divergence & Csiszár Generators](./f_Divergence.md)** | Unified divergence family, non-negativity, convexity ($D_f(P \parallel Q) = \int q f(p/q) dx$) | Lec 03, Tut 11, Lec 04 |
| **[Jensen-Shannon Divergence](./Jensen_Shannon_Divergence.md)** | Symmetric divergence, strict $[0, \ln 2]$ bounds, GAN minimax connection | Lec 03, Lec 05, Tut 12 |
| **[Joint, Marginal & Conditional Distributions](./Joint_Marginal_Conditional_Dist.md)** | Bayes' Theorem, continuous slices, chain rule ($p(x, z) = p(x \mid z) p(z)$) | Tut 09, Lec 19, Lec 20 |
| **[Kullback-Leibler (KL) Divergence](./KL_Divergence.md)** | Relative entropy, forward vs reverse KL mode covering/dropping ($D_{\text{KL}}(P \parallel Q) = \int p \ln \frac{p}{q} dx$) | Lec 02, Lec 03, Lec 20 |
| **[Likelihood & Log-Likelihood](./Likelihood_and_Log_Likelihood.md)** | Parameter scoring given fixed data ($L(\theta; X) = \prod p(x_i \mid \theta)$) | Tut 08, Tut 10, Lec 20 |
| **[Maximum Likelihood Estimation (MLE)](./MLE.md)** | Optimal parameter estimation, score equations ($\hat{\theta}_{\text{MLE}} = \arg\max \sum \ln p(x_i)$) | Tut 10, Lec 02, Lec 20 |
| **[Negative Log-Likelihood (NLL)](./NLL.md)** | Supervised & generative training loss ($\mathcal{L}_{\text{NLL}} = -\sum_{i=1}^N \ln p_\theta(y_i \mid x_i)$) | Lec 01, Tut 10 |
| **[Probability Basics & Kolmogorov Axioms](./Probability_Basics_and_Axioms.md)** | Sample spaces, $\sigma$-algebras, measure non-negativity, countable additivity | Tut 07 |
| **[Random Variables & Probability Distributions](./Random_Variables_and_Distributions.md)** | Discrete PMF vs continuous PDF, expectation, variance, law of total probability | Tut 07, Tut 08 |

---

### 4. 🧠 Modern Generative AI, Variational & Adversarial Foundations

| Mathematical Guide | Core Focus & Key Formula | Primary Modules |
| :--- | :--- | :--- |
| **[Autoencoders & Latent Space Manifolds](./Autoencoders_and_Latent_Spaces.md)** | Manifold hypothesis, encoder-decoder bottlenecks, spatial latent embeddings | Tut 06, Lec 19, Lec 20 |
| **[Autoregressive Generative Models](./Autoregressive_Models.md)** | Probability chain rule factorization ($p(x) = \prod_{t=1}^T p(x_t \mid x_{<t})$), causal masking | Lec 02, Tut 05 |
| **[Evidence Lower Bound (ELBO) & Variational Inference](./ELBO_and_Variational_Inference.md)** | Tractable evidence bound, encoder-decoder optimization, VAEs | Lec 20 |
| **[Expectation-Maximization (EM) Algorithm](./Expectation_Maximization_Algorithm.md)** | Incomplete likelihood, E-step responsibilities, M-step closed forms, GMM fitting | Tut 10, Lec 20 |
| **[Fenchel Conjugate & Dual Variational Representations](./Fenchel_Conjugate_and_Dual_Representations.md)** | Legendre-Fenchel transformation ($f^*(t) = \sup_u \{tu - f(u)\}$), $f$-GAN variational dual | Lec 04, Lec 05, Lec 18 |
| **[Fréchet Inception Distance (FID)](./Frechet_Inception_Distance.md)** | 2-Wasserstein metric on Inception Gaussians ($\|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$) | Tut 12, Lec 19 |
| **[Latent Variable Models](./Latent_Variable_Models.md)** | Unobserved causes $p(x, z) = p(x \mid z)p(z)$, intractable marginals, GMMs, VAEs, DDPMs | Lec 19, Lec 20 |
| **[Minimax Games & GANs](./Minimax_Game_and_GANs.md)** | Two-player zero-sum saddle $\min_G \max_D V(G, D)$, Nash equilibrium $D^*(x) = 0.5$ | Lec 04, Lec 05, Tut 12 |
| **[Reparameterization Trick](./Reparameterization_Trick.md)** | Differentiable sampling $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$, enabling backpropagation in VAEs | Lec 20 |
| **[Wasserstein Distance & Earth Mover's Distance (EMD)](./Wasserstein_Distance_and_EMD.md)** | Optimal transport, Kantorovich-Rubinstein dual ($\sup_{\|f\|_L \le 1} \mathbb{E}_P[f] - \mathbb{E}_Q[f]$), WGAN-GP | Lec 18, Tut 12 |

---

## 🔍 How to Cross-Reference in Module Notes

To link to any mathematical term from a lecture or tutorial note, use relative paths:
```markdown
[Functions, Derivatives & Rules](../../../MathsTerms/Functions_Derivatives_and_Rules.md)
[The Chain Rule & Backpropagation](../../../MathsTerms/Chain_Rule_and_Backpropagation.md)
[The Jacobian Matrix](../../../MathsTerms/Jacobian_Matrix.md)
[Singular Value Decomposition](../../../MathsTerms/Singular_Value_Decomposition.md)
[Exponential Moving Average (EMA)](../../../MathsTerms/Exponential_Moving_Average_EMA.md)
[Similarity with Dot Product](../../../MathsTerms/Dot_Product_and_Similarity.md)
[Tensor Broadcasting](../../../MathsTerms/Tensor_Broadcasting.md)
[Positional Encodings & RoPE](../../../MathsTerms/Positional_Encodings.md)
```
