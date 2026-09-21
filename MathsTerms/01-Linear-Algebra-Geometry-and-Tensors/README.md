# 📚 Linear Algebra, Geometry, Tensors & Embeddings (`01-Linear-Algebra-Geometry-and-Tensors`)

> `🏷️ Sub-Cluster:` `01-Linear-Algebra-Geometry-and-Tensors`  
> `🎯 Core Purpose:` The geometric, structural, and algebraic engine of deep learning and generative modeling: vector spaces, matrix transformations, norms, inner products, singular value decomposition (SVD), tensor ranks and strided layouts, broadcasting arithmetic, and discrete-to-continuous embedding projections (One-Hot, Dense Embeddings, RoPE).  
> `📐 Pedagogical Standard:` 14 Canonical Sections (4-Question Onboarding $\iff$ Geometric Intuition $\iff$ Pencil-and-Paper Forward & Backward Pass $\iff$ Dual-Stage Python/PyTorch Verification $\iff$ 5-Interval Spaced Return Schedule $\iff$ 5-Gate Comprehension Audit)  
> `🗺️ Master Roadmap:` [Grand Unified Concept Map & Mathematical Dependency Graph](../CONCEPT_MAP.md)  
> `🧭 Catalog Index:` [MathsTerms Master Catalog](../README.md)

---

## 🧭 Curated Mathematical Guides in this Cluster

| # | Guide Title | Core Mathematical Concept | Key Upstream Prerequisites | Modern Generative AI Application |
| :-: | :--- | :--- | :--- | :--- |
| **01** | **[Vectors and Matrices](./01-Vectors_and_Matrices.md)** | Standard vector spaces, linear maps, matrix multiplication as spatial warping, column vs row-major layouts, and outer product gradient propagation | Basic algebra & arithmetic | Linear projection layers ($W_Q, W_K, W_V$), feed-forward networks (MLP), and backpropagation gradients ($\nabla_W \mathcal{L} = \delta x^\top$) |
| **01b** | **[Basis, Spans & Orthogonality](./01b-Basis_Spans_and_Orthogonality.md)** | Spans, linear independence, basis coordinate systems, orthonormal bases, Gram-Schmidt process | [Vectors & Matrices](./01-Vectors_and_Matrices.md), [Norms](./02-Vector_Norms_and_Inner_Products.md) | VAE latent spaces, orthogonal weight initialization (`torch.nn.init.orthogonal_`), LoRA low-rank spans |
| **01c** | **[Determinants & Volume Scaling](./01c-Determinants_and_Volume_Scaling.md)** | 2D area/3D volume scaling, orientation flips, singularity, triangular matrix determinant shortcut | [Vectors & Matrices](./01-Vectors_and_Matrices.md), [Basis & Spans](./01b-Basis_Spans_and_Orthogonality.md) | Normalizing Flows (RealNVP, Glow) change-of-variables, triangular Jacobian log-determinants |
| **02** | **[Vector Norms and Inner Products](./02-Vector_Norms_and_Inner_Products.md)** | Metric spaces, $L_1$ (Manhattan), $L_2$ (Euclidean), $L_\infty$, Frobenius norms, Cauchy-Schwarz inequality, and unit hyperspheres | [Vectors & Matrices](./01-Vectors_and_Matrices.md) | Weight decay ($L_2$ regularization), RMSNorm / LayerNorm token stabilization, gradient clipping, and spectral normalization |
| **03** | **[Dot Product and Similarity](./03-Dot_Product_and_Similarity.md)** | Geometric projection, angular correlation ($\cos \theta$), Cauchy-Schwarz bound, and scaled dot-product attention logits | [Vectors & Matrices](./01-Vectors_and_Matrices.md), [Norms](./02-Vector_Norms_and_Inner_Products.md) | Scaled Dot-Product Attention in Transformers ($\text{Softmax}(Q K^\top / \sqrt{d_k})$), contrastive learning in CLIP, and vector database search |
| **04** | **[Tensors and Shapes](./04-Tensors_and_Shapes.md)** | Multidimensional coordinate tensors, stride algebra, row-major contiguous memory layouts, and view vs copy mechanics | [Vectors & Matrices](./01-Vectors_and_Matrices.md) | Multi-head attention tensors (`[B, H, S, D]`), image latents (`[B, C, H, W]`), and CUDA memory coalescing on GPU Tensor Cores |
| **05** | **[Tensor Broadcasting](./05-Tensor_Broadcasting.md)** | NumPy/PyTorch right-aligned shape broadcasting, zero-stride pointer dereferencing, and dual adjoint summation gradient reduction | [Tensors & Shapes](./04-Tensors_and_Shapes.md), [Vectors & Matrices](./01-Vectors_and_Matrices.md) | Fused bias addition ($Y = X W^\top + b$), LayerNorm affine parameters ($\gamma, \beta$), and FlashAttention SRAM memory tiling |
| **05b** | **[Eigenvalues & Eigenvectors](./05b-Eigenvalues_and_Eigenvectors.md)** | Characteristic equation $\det(A - \lambda I) = 0$, un-rotated axes, Spectral Theorem, matrix diagonalization | [Vectors & Matrices](./01-Vectors_and_Matrices.md), [Determinants](./01c-Determinants_and_Volume_Scaling.md) | Principal Component Analysis (PCA), Spectral Normalization in GANs, controlling RNN spectral radius |
| **06** | **[Singular Value Decomposition (SVD)](./06-Singular_Value_Decomposition.md)** | Universal matrix factorization ($A = U \Sigma V^\top$), rotate-stretch-rotate geometry, Eckart-Young-Mirsky low-rank theorem, and condition numbers | [Vectors & Matrices](./01-Vectors_and_Matrices.md), [Norms](./02-Vector_Norms_and_Inner_Products.md) | Low-Rank Adaptation (LoRA: $\Delta W = B \cdot A$) in LLMs & Diffusion, weight pruning, PCA, and Moore-Penrose pseudoinverses |
| **07** | **[One-Hot Encoding](./07-One_Hot_Encoding.md)** | Standard orthonormal basis vectors ($e_k \in \mathbb{R}^K$), equidistant categorical geometry ($\|e_i - e_j\|_2 = \sqrt{2}$), and label smoothing | [Vectors & Matrices](./01-Vectors_and_Matrices.md) | Discrete token representation, cross-entropy loss targets, class conditioning in cGANs/DiTs, and GPU `gather`/`scatter-add` lookup tables |
| **08** | **[Categorical Encodings & Embeddings](./08-Encodings_Categorical_and_Embeddings.md)** | Continuous dense semantic coordinate spaces ($W_E \in \mathbb{R}^{V \times D}$), subword Byte-Pair Encoding (BPE), and sparse gradient accumulation | [One-Hot Encoding](./07-One_Hot_Encoding.md), [Vectors & Matrices](./01-Vectors_and_Matrices.md) | Token embedding layers in LLMs (LLaMA-3, GPT-4), joint vision-language text encoders (CLIP), and Megatron-LM vocabulary tensor parallelism |
| **09** | **[Positional Encodings](./09-Positional_Encodings.md)** | Permutation invariance resolution, Sinusoidal waves, ALiBi linear slopes, Rotary Position Embedding (RoPE) 2D Givens rotations, and context extrapolation | [Dot Product](./03-Dot_Product_and_Similarity.md), [Embeddings](./08-Encodings_Categorical_and_Embeddings.md), [Matrices](./01-Vectors_and_Matrices.md) | Sequence order modeling in Transformers, RoPE in modern LLMs (LLaMA-3, Mistral, Gemma), long-context scaling (YaRN), and Diffusion timesteps |

---

## 🗺️ Recommended Pedagogical Reading Order

For optimal conceptual continuity and zero mathematical gaps, study these guides in the following sequential order:

```text
[01. Vectors & Matrices]
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
[01b. Basis, Spans & Orthogonality]      [02. Vector Norms & Inner Products]
       │                                          │
       ▼                                          ▼
[01c. Determinants & Volume Scaling]     [03. Dot Product & Similarity]
       │                                          │
       ├───────────────────────────┐              ▼
       ▼                           ▼     [04. Tensors & Shapes]
[05b. Eigenvalues & Eigenvectors]  │              │
       │                           │              ▼
       ▼                           │     [05. Tensor Broadcasting]
[06. Singular Value Decomposition] │              │
                                   └──────────────┼──────────────────────────┐
                                                  ▼                          ▼
                                         [07. One-Hot Encoding]     [09. Positional Encodings]
                                                  │
                                                  ▼
                                         [08. Categorical Encodings & Embeddings]
```

1. **[01-Vectors_and_Matrices.md](./01-Vectors_and_Matrices.md)** — Master vector arrows, linear transformations, matrix-vector multiplication, and outer-product gradients.
2. **[01b-Basis_Spans_and_Orthogonality.md](./01b-Basis_Spans_and_Orthogonality.md)** — Understand how directions span spaces, eliminate redundancy with linear independence, and construct orthonormal coordinate systems.
3. **[01c-Determinants_and_Volume_Scaling.md](./01c-Determinants_and_Volume_Scaling.md)** — Learn how matrices scale areas and hypervolumes, interpret orientation flips, and compute triangular determinants for Normalizing Flows.
4. **[02-Vector_Norms_and_Inner_Products.md](./02-Vector_Norms_and_Inner_Products.md)** — Measure lengths, metric distances, unit balls, and normalization gradients.
5. **[03-Dot_Product_and_Similarity.md](./03-Dot_Product_and_Similarity.md)** — Understand directional alignment, cosine similarity, and the attention logit engine.
6. **[04-Tensors_and_Shapes.md](./04-Tensors_and_Shapes.md)** — Scale from 2D matrices to $N$-D strided memory layouts in PyTorch and CUDA.
7. **[05-Tensor_Broadcasting.md](./05-Tensor_Broadcasting.md)** — Learn how GPUs execute operations across mismatched tensor dimensions with zero memory copying.
8. **[05b-Eigenvalues_and_Eigenvectors.md](./05b-Eigenvalues_and_Eigenvectors.md)** — Uncover un-rotated transformation axes, characteristic equations, and spectral stability for GANs and RNNs.
9. **[06-Singular_Value_Decomposition.md](./06-Singular_Value_Decomposition.md)** — Decompose matrices into rotations and stretching, enabling 99% parameter compression with LoRA.
10. **[07-One_Hot_Encoding.md](./07-One_Hot_Encoding.md)** — Represent discrete categories as mutually orthogonal, equidistant vectors in high-dimensional space.
11. **[08-Encodings_Categorical_and_Embeddings.md](./08-Encodings_Categorical_and_Embeddings.md)** — Map discrete tokens into continuous semantic coordinates where words become vectors.
12. **[09-Positional_Encodings.md](./09-Positional_Encodings.md)** — Break permutation invariance and inject sequence order using Sinusoids, ALiBi, and Rotary Position Embeddings (RoPE).

---

## 🔗 Cross-Cluster Interconnections

- Return to the **[Grand Unified Concept Map & Mathematical Dependency Graph](../CONCEPT_MAP.md)** to inspect how this cluster connects across all 6 mathematical tiers.
- Navigate back to the **[MathsTerms Master Catalog](../README.md)**.
