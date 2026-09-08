# 📚 Mathematical Terms & Generative AI Foundations Catalog

Welcome to the **Mathematical Terms (MathsTerms) Knowledge Base** for the *Mathematical Foundations of Generative AI* masterclass.

Every guide in this directory follows a strict **10-Section Gold Standard** bridging abstract mathematical rigor directly to intuitive physical primitives, elementary algebraic proofs, micro-numerical arithmetic, runnable PyTorch/NumPy code, and modern generative AI architectures (Diffusion Models, GANs, VAEs, Transformers, Flow Matching).

`
 ===================================================================================================
                 THE 5-POINT PEDAGOGICAL BRIDGE (STANDARD ACROSS ALL GUIDES)
 ===================================================================================================
 
  👶 ELI5 Intuition  ◄──►  🔍 Plain-English  ◄──►  🔢 Micro-Numbers  ◄──►  📐 Formal Math  ◄──►  💻 Runnable Code
  (Everyday Metaphors)     (Rosetta Stones)        (Step-by-Step Maths)    (Theorems & Proofs)     (PyTorch/NumPy)
 ===================================================================================================
`

---

## 🗺️ Master Dependency Graph & Concept Roadmap

> 🚀 **Looking to connect all the dots?**  
> Explore the comprehensive **[Grand Unified Concept Map & Mathematical Dependency Graph](./CONCEPT_MAP.md)** connecting all 46 guides from first-principles axioms to Generative AI architectures (VAEs, GANs, Diffusion Models, Autoregressive LLMs, Flow Matching), featuring interactive Mermaid diagrams, ASCII pipelines, sub-term lineage, and 4 specialized learning tracks.

---

## 🧭 Master Directory Index (46 Curated Guides in 6 Sequenced Categories)

### 1. ⚡ [01-Primal-Analysis-and-Foundations](./01-Primal-Analysis-and-Foundations/README.md)

| Mathematical Guide | Core Focus & Key Formula | Primary Modules |
| :--- | :--- | :--- |
| **[Probability Basics & Axioms](./01-Primal-Analysis-and-Foundations/Probability_Basics_and_Axioms.md)** | Sample spaces, $\sigma$-algebras, measure non-negativity, countable additivity | Tut 07 |
| **[Logarithms & Exponential Functions](./01-Primal-Analysis-and-Foundations/Logarithms_and_Exponential_Functions.md)** | Numerical stability, log-space arithmetic, preventing underflow ($\ln(ab) = \ln a + \ln b$) | Tut 02, Lec 01, Tut 10 |
| **[Convexity & Jensen\'s Inequality](./01-Primal-Analysis-and-Foundations/Convexity_and_Jensens_Inequality.md)** | Convex functions, secant lines, and lower bounds ((\mathbb{E}[X]) \le \mathbb{E}[f(X)]$) | Lec 03, Lec 20 |
| **[Lipschitz Continuity](./01-Primal-Analysis-and-Foundations/Lipschitz_Continuity.md)** | Bounded gradient slope $\|f(x) - f(y)\| \le K \|x - y\|$, Kantorovich-Rubinstein dual | Lec 18, Tut 12 |
| **[Fenchel Conjugate & Dual Variational Representations](./01-Primal-Analysis-and-Foundations/Fenchel_Conjugate_and_Dual_Representations.md)** | Legendre-Fenchel transformation (^*(t) = \sup_u \{tu - f(u)\}$), $-GAN variational dual | Lec 04, Lec 05, Lec 18 |

---

### 2. 📐 [02-Linear-Algebra-Geometry-and-Tensors](./02-Linear-Algebra-Geometry-and-Tensors/README.md)

| Mathematical Guide | Core Focus & Key Formula | Primary Modules |
| :--- | :--- | :--- |
| **[Vectors & Matrices](./02-Linear-Algebra-Geometry-and-Tensors/Vectors_and_Matrices.md)** | Linear transformations, coordinate bases, matrix multiplication ( = Wx + b$) | Tut 02, Tut 03, Tut 06 |
| **[Vector Norms & Inner Products](./02-Linear-Algebra-Geometry-and-Tensors/Vector_Norms_and_Inner_Products.md)** | , L_2, L_\infty$ geometry, Lasso sparsity vs Ridge weight decay | Tut 02, Lec 18 |
| **[Similarity with Dot Product](./02-Linear-Algebra-Geometry-and-Tensors/Dot_Product_and_Similarity.md)** | Vector projections, cosine similarity, Scaled Attention ($\vec{a} \cdot \vec{b} = \|\vec{a}\|\|\vec{b}\|\cos\theta$) | Tut 02, Tut 03, Tut 06 |
| **[Tensors, Shapes & Dimensional Broadcasting](./02-Linear-Algebra-Geometry-and-Tensors/Tensors_and_Shapes.md)** | Multidimensional arrays, PyTorch strided layouts, contiguous memory buffers | Tut 02, Tut 03 |
| **[Tensor Broadcasting](./02-Linear-Algebra-Geometry-and-Tensors/Tensor_Broadcasting.md)** | Memory strides, 3 golden broadcasting rules, zero-copy expansion (stride = 0) | Tut 02, Tut 03 |
| **[Singular Value Decomposition (SVD)](./02-Linear-Algebra-Geometry-and-Tensors/Singular_Value_Decomposition.md)** | Rotate-Stretch-Rotate factorization, Eckart-Young theorem, LoRA ( = U \Sigma V^T$) | Tut 06, Lec 01 |
| **[One-Hot Encoding](./02-Linear-Algebra-Geometry-and-Tensors/One_Hot_Encoding.md)** | Sparse categorical representation, cross-entropy target vectors | Lec 01, Tut 10 |
| **[Encodings & Categorical Embeddings](./02-Linear-Algebra-Geometry-and-Tensors/Encodings_Categorical_and_Embeddings.md)** | Discrete token IDs to continuous coordinates, BPE tokenization, embedding lookup tables | Tut 03, Lec 01 |
| **[Positional Encodings & RoPE](./02-Linear-Algebra-Geometry-and-Tensors/Positional_Encodings.md)** | Sequence order, Sinusoidal, Learned, ALiBi, and Rotary Position Embeddings in LLMs | Tut 03, Lec 01 |

---

### 3. ⚡ [03-Multivariate-Calculus-and-Optimization](./03-Multivariate-Calculus-and-Optimization/README.md)

| Mathematical Guide | Core Focus & Key Formula | Primary Modules |
| :--- | :--- | :--- |
| **[Functions, Derivatives & Rules](./03-Multivariate-Calculus-and-Optimization/Functions_Derivatives_and_Rules.md)** | Power, product, quotient rules, limit definition ('(x) = \lim \frac{f(x+h)-f(x)}{h}$) | Tut 03, Lec 01 |
| **[Derivatives, Gradients & Jacobians](./03-Multivariate-Calculus-and-Optimization/Derivatives_Gradients_and_Jacobians.md)** | Multivariable calculus, autograd computational graphs, Jacobian matrices | Tut 03, Lec 04, Lec 18 |
| **[The Jacobian Matrix](./03-Multivariate-Calculus-and-Optimization/Jacobian_Matrix.md)** | Vector-valued derivatives, local volume warping, VJP/JVP ({ij} = \partial f_i / \partial x_j$) | Tut 03, Tut 06, Lec 01 |
| **[The Chain Rule & Backpropagation](./03-Multivariate-Calculus-and-Optimization/Chain_Rule_and_Backpropagation.md)** | Reverse-mode automatic differentiation, DAG computation graphs ($\frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx}$) | Tut 03, Tut 04, Lec 01 |
| **[Activation Functions](./03-Multivariate-Calculus-and-Optimization/Activation_Functions.md)** | Non-linearities: ReLU, LeakyReLU, GELU, SwiGLU, Sigmoid ($\sigma(z), \text{GELU}(z)$) | Lec 01, Tut 03, Tut 04 |
| **[Softmax Function](./03-Multivariate-Calculus-and-Optimization/Softmax.md)** | Probability normalization, logits squashing, temperature scaling ($\frac{e^{z_i/\tau}}{\sum e^{z_j/\tau}}$) | Lec 01, Tut 03, Tut 10 |
| **[Argmax & Soft-Argmax](./03-Multivariate-Calculus-and-Optimization/Argmax.md)** | Non-differentiable discrete selection vs differentiable Gumbel-Softmax | Tut 03, Lec 02 |
| **[Loss Functions in Machine Learning](./03-Multivariate-Calculus-and-Optimization/Loss_Functions.md)** | MSE, MAE, Huber, BCE, CCE, Hinge, Wasserstein objectives | Tut 03, Tut 10, Lec 05 |
| **[Gradient Descent & Optimizers](./03-Multivariate-Calculus-and-Optimization/Gradient_Descent.md)** | SGD, Momentum, RMSprop, Adam first/second moment updates ($\theta \leftarrow \theta - \eta \nabla L$) | Tut 03, Lec 05, Lec 19 |
| **[Exponential Moving Average (EMA)](./03-Multivariate-Calculus-and-Optimization/Exponential_Moving_Average_EMA.md)** | Shadow weights in Diffusion, Adam momentum ($\theta_{\text{EMA}} = \beta \theta_{\text{EMA}} + (1-\beta)\theta$) | Tut 03, Lec 01, Lec 18 |
| **[Batch Normalization & Spectral Norm](./03-Multivariate-Calculus-and-Optimization/Batch_Normalization_and_Spectral_Norm.md)** | Layer normalization & 1-Lipschitz matrix spectral norm ($\|W\|_2 \le 1, W / \sigma(W)$) | Tut 04, Lec 18, Tut 12 |

---

### 4. 🎲 [04-Probability-and-Statistical-Estimation](./04-Probability-and-Statistical-Estimation/README.md)

| Mathematical Guide | Core Focus & Key Formula | Primary Modules |
| :--- | :--- | :--- |
| **[Random Variables & Probability Distributions](./04-Probability-and-Statistical-Estimation/Random_Variables_and_Distributions.md)** | Discrete PMF vs continuous PDF, expectation, variance, law of total probability | Tut 07, Tut 08 |
| **[Common Probability Distributions](./04-Probability-and-Statistical-Estimation/Common_Probability_Distributions.md)** | Gaussian, Uniform, Bernoulli, Categorical, multivariate covariance matrices | Tut 07, Tut 08, Lec 18 |
| **[Joint, Marginal & Conditional Distributions](./04-Probability-and-Statistical-Estimation/Joint_Marginal_Conditional_Dist.md)** | Bayes\' Theorem, continuous slices, chain rule ((x, z) = p(x \mid z) p(z)$) | Tut 09, Lec 19, Lec 20 |
| **[Likelihood & Log-Likelihood](./04-Probability-and-Statistical-Estimation/Likelihood_and_Log_Likelihood.md)** | Parameter scoring given fixed data ((\theta; X) = \prod p(x_i \mid \theta)$) | Tut 08, Tut 10, Lec 20 |
| **[Maximum Likelihood Estimation (MLE)](./04-Probability-and-Statistical-Estimation/MLE.md)** | Optimal parameter estimation, score equations ($\hat{\theta}_{\text{MLE}} = \arg\max \sum \ln p(x_i)$) | Tut 10, Lec 02, Lec 20 |
| **[Negative Log-Likelihood (NLL)](./04-Probability-and-Statistical-Estimation/NLL.md)** | Supervised & generative training loss ($\mathcal{L}_{\text{NLL}} = -\sum_{i=1}^N \ln p_\theta(y_i \mid x_i)$) | Lec 01, Tut 10 |

---

### 5. 🌐 [05-Information-Theory-and-Divergences](./05-Information-Theory-and-Divergences/README.md)

| Mathematical Guide | Core Focus & Key Formula | Primary Modules |
| :--- | :--- | :--- |
| **[Entropy, Cross-Entropy & Categorical Cross-Entropy](./05-Information-Theory-and-Divergences/Entropy_CrossEntropy_CCE.md)** | Shannon surprise, cross-entropy loss, negative log-likelihood ((P, Q) = -\sum p \log q$) | Lec 01, Tut 10 |
| **[Kullback-Leibler (KL) Divergence](./05-Information-Theory-and-Divergences/KL_Divergence.md)** | Relative entropy, forward vs reverse KL mode covering/dropping ({\text{KL}}(P \parallel Q) = \int p \ln \frac{p}{q} dx$) | Lec 02, Lec 03, Lec 20 |
| **[Jensen-Shannon Divergence](./05-Information-Theory-and-Divergences/Jensen_Shannon_Divergence.md)** | Symmetric divergence, strict $[0, \ln 2]$ bounds, GAN minimax connection | Lec 03, Lec 05, Tut 12 |
| **[$-Divergence & Csiszár Generators](./05-Information-Theory-and-Divergences/f_Divergence.md)** | Unified divergence family, non-negativity, convexity ((P \parallel Q) = \int q f(p/q) dx$) | Lec 03, Tut 11, Lec 04 |
| **[Wasserstein Distance & Earth Mover\'s Distance (EMD)](./05-Information-Theory-and-Divergences/Wasserstein_Distance_and_EMD.md)** | Optimal transport, Kantorovich-Rubinstein dual ($\sup_{\|f\|_L \le 1} \mathbb{E}_P[f] - \mathbb{E}_Q[f]$), WGAN-GP | Lec 18, Tut 12 |

---

### 6. 🧠 [06-Deep-Architectures-and-Generative-Models](./06-Deep-Architectures-and-Generative-Models/README.md)

| Mathematical Guide | Core Focus & Key Formula | Primary Modules |
| :--- | :--- | :--- |
| **[Convolution & Pooling Operations](./06-Deep-Architectures-and-Generative-Models/Convolution_and_Pooling.md)** | Receptive fields, spatial downsampling, transposed convolutions, DCGAN | Tut 04, Tut 12 |
| **[Recurrent Neural Networks & Backprop Through Time](./06-Deep-Architectures-and-Generative-Models/Recurrent_Neural_Networks.md)** | Sequence processing, hidden state recurrence, vanishing gradients in BPTT | Tut 05 |
| **[Autoencoders & Latent Space Manifolds](./06-Deep-Architectures-and-Generative-Models/Autoencoders_and_Latent_Spaces.md)** | Manifold hypothesis, encoder-decoder bottlenecks, spatial latent embeddings | Tut 06, Lec 19, Lec 20 |
| **[Autoregressive Generative Models](./06-Deep-Architectures-and-Generative-Models/Autoregressive_Models.md)** | Probability chain rule factorization ((x) = \prod_{t=1}^T p(x_t \mid x_{<t})$), causal masking | Lec 02, Tut 05 |
| **[Latent Variable Models](./06-Deep-Architectures-and-Generative-Models/Latent_Variable_Models.md)** | Unobserved causes (x, z) = p(x \mid z)p(z)$, intractable marginals, GMMs, VAEs, DDPMs | Lec 19, Lec 20 |
| **[Expectation-Maximization (EM) Algorithm](./06-Deep-Architectures-and-Generative-Models/Expectation_Maximization_Algorithm.md)** | Incomplete likelihood, E-step responsibilities, M-step closed forms, GMM fitting | Tut 10, Lec 20 |
| **[Evidence Lower Bound (ELBO) & Variational Inference](./06-Deep-Architectures-and-Generative-Models/ELBO_and_Variational_Inference.md)** | Tractable evidence bound, encoder-decoder optimization, VAEs | Lec 20 |
| **[Reparameterization Trick](./06-Deep-Architectures-and-Generative-Models/Reparameterization_Trick.md)** | Differentiable sampling  = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$, enabling backpropagation in VAEs | Lec 20 |
| **[Minimax Games & GANs](./06-Deep-Architectures-and-Generative-Models/Minimax_Game_and_GANs.md)** | Two-player zero-sum saddle $\min_G \max_D V(G, D)$, Nash equilibrium ^*(x) = 0.5$ | Lec 04, Lec 05, Tut 12 |
| **[Fréchet Inception Distance (FID)](./06-Deep-Architectures-and-Generative-Models/Frechet_Inception_Distance.md)** | 2-Wasserstein metric on Inception Gaussians ($\|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$) | Tut 12, Lec 19 |

---

## 🔍 How to Cross-Reference in Module Notes

To link to any mathematical term from a lecture or tutorial note in Mathematical-Foundation-for-GenerativeAI/, use standard relative paths:
`	ext
[Functions, Derivatives & Rules]       --> ../../MathsTerms/03-Multivariate-Calculus-and-Optimization/Functions_Derivatives_and_Rules.md
[The Chain Rule & Backpropagation]    --> ../../MathsTerms/03-Multivariate-Calculus-and-Optimization/Chain_Rule_and_Backpropagation.md
[The Jacobian Matrix]                 --> ../../MathsTerms/03-Multivariate-Calculus-and-Optimization/Jacobian_Matrix.md
[Singular Value Decomposition]        --> ../../MathsTerms/02-Linear-Algebra-Geometry-and-Tensors/Singular_Value_Decomposition.md
[Exponential Moving Average (EMA)]    --> ../../MathsTerms/03-Multivariate-Calculus-and-Optimization/Exponential_Moving_Average_EMA.md
[Similarity with Dot Product]         --> ../../MathsTerms/02-Linear-Algebra-Geometry-and-Tensors/Dot_Product_and_Similarity.md
[Tensor Broadcasting]                 --> ../../MathsTerms/02-Linear-Algebra-Geometry-and-Tensors/Tensor_Broadcasting.md
[Positional Encodings & RoPE]         --> ../../MathsTerms/02-Linear-Algebra-Geometry-and-Tensors/Positional_Encodings.md
`
