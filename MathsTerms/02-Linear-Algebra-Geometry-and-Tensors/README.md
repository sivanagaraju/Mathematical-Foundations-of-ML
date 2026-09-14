# 📚 Linear Algebra, Geometry, Tensors & Embeddings (`02-Linear-Algebra-Geometry-and-Tensors`)

> `🏷️ Sub-Cluster:` `02-Linear-Algebra-Geometry-and-Tensors`  
> `🎯 Core Purpose:` The geometric, structural, and algebraic engine of deep learning and generative modeling: vector spaces, matrix transformations, norms, inner products, singular value decomposition (SVD), tensor ranks and strided layouts, broadcasting arithmetic, and discrete-to-continuous embedding projections (One-Hot, Dense Embeddings, RoPE).  
> `📐 Pedagogical Standard:` 14 Canonical Sections (4-Question Onboarding $\iff$ Geometric Intuition $\iff$ Pencil-and-Paper Forward & Backward Pass $\iff$ Dual-Stage Python/PyTorch Verification $\iff$ 5-Interval Spaced Return Schedule $\iff$ 5-Gate Comprehension Audit)  
> `🗺️ Master Roadmap:` [Grand Unified Concept Map & Mathematical Dependency Graph](../CONCEPT_MAP.md)  
> `🧭 Catalog Index:` [MathsTerms Master Catalog](../README.md)

---

## 🧭 Curated Mathematical Guides in this Cluster

| # | Guide Title | Core Mathematical Concept | Key Upstream Prerequisites | Modern Generative AI Application |
| :-: | :--- | :--- | :--- | :--- |
| **01** | **[Vectors and Matrices](./01-Vectors_and_Matrices.md)** | Standard vector spaces, linear maps, matrix multiplication as spatial warping, column vs row-major layouts, and outer product gradient propagation | Basic algebra & arithmetic | Linear projection layers ($W_Q, W_K, W_V$), feed-forward networks (MLP), and backpropagation gradients ($\nabla_W \mathcal{L} = \delta x^\top$) |
| **02** | **[Vector Norms and Inner Products](./02-Vector_Norms_and_Inner_Products.md)** | Metric spaces, $L_1$ (Manhattan), $L_2$ (Euclidean), $L_\infty$, Frobenius norms, Cauchy-Schwarz inequality, and unit hyperspheres | [Vectors & Matrices](./01-Vectors_and_Matrices.md) | Weight decay ($L_2$ regularization), RMSNorm / LayerNorm token stabilization, gradient clipping, and spectral normalization |
| **03** | **[Dot Product and Similarity](./03-Dot_Product_and_Similarity.md)** | Geometric projection, angular correlation ($\cos \theta$), Cauchy-Schwarz bound, and scaled dot-product attention logits | [Vectors & Matrices](./01-Vectors_and_Matrices.md), [Norms](./02-Vector_Norms_and_Inner_Products.md) | Scaled Dot-Product Attention in Transformers ($\text{Softmax}(Q K^\top / \sqrt{d_k})$), contrastive learning in CLIP, and vector database search |
| **04** | **[Tensors and Shapes](./04-Tensors_and_Shapes.md)** | Multidimensional coordinate tensors, stride algebra, row-major contiguous memory layouts, and view vs copy mechanics | [Vectors & Matrices](./01-Vectors_and_Matrices.md) | Multi-head attention tensors (`[B, H, S, D]`), image latents (`[B, C, H, W]`), and CUDA memory coalescing on GPU Tensor Cores |
| **05** | **[Tensor Broadcasting](./05-Tensor_Broadcasting.md)** | NumPy/PyTorch right-aligned shape broadcasting, zero-stride pointer dereferencing, and dual adjoint summation gradient reduction | [Tensors & Shapes](./04-Tensors_and_Shapes.md), [Vectors & Matrices](./01-Vectors_and_Matrices.md) | Fused bias addition ($Y = X W^\top + b$), LayerNorm affine parameters ($\gamma, \beta$), and FlashAttention SRAM memory tiling |
| **06** | **[Singular Value Decomposition (SVD)](./06-Singular_Value_Decomposition.md)** | Universal matrix factorization ($A = U \Sigma V^\top$), rotate-stretch-rotate geometry, Eckart-Young-Mirsky low-rank theorem, and condition numbers | [Vectors & Matrices](./01-Vectors_and_Matrices.md), [Norms](./02-Vector_Norms_and_Inner_Products.md) | Low-Rank Adaptation (LoRA: $\Delta W = B \cdot A$) in LLMs & Diffusion, weight pruning, PCA, and Moore-Penrose pseudoinverses |
| **07** | **[One-Hot Encoding](./07-One_Hot_Encoding.md)** | Standard orthonormal basis vectors ($e_k \in \mathbb{R}^K$), equidistant categorical geometry ($\|e_i - e_j\|_2 = \sqrt{2}$), and label smoothing | [Vectors & Matrices](./01-Vectors_and_Matrices.md) | Discrete token representation, cross-entropy loss targets, class conditioning in cGANs/DiTs, and GPU `gather`/`scatter-add` lookup tables |
| **08** | **[Categorical Encodings & Embeddings](./08-Encodings_Categorical_and_Embeddings.md)** | Continuous dense semantic coordinate spaces ($W_E \in \mathbb{R}^{V \times D}$), subword Byte-Pair Encoding (BPE), and sparse gradient accumulation | [One-Hot Encoding](./07-One_Hot_Encoding.md), [Vectors & Matrices](./01-Vectors_and_Matrices.md) | Token embedding layers in LLMs (LLaMA-3, GPT-4), joint vision-language text encoders (CLIP), and Megatron-LM vocabulary tensor parallelism |
| **09** | **[Positional Encodings](./09-Positional_Encodings.md)** | Permutation invariance resolution, Sinusoidal waves, ALiBi linear slopes, Rotary Position Embedding (RoPE) 2D Givens rotations, and context extrapolation | [Dot Product](./03-Dot_Product_and_Similarity.md), [Embeddings](./08-Encodings_Categorical_and_Embeddings.md), [Matrices](./01-Vectors_and_Matrices.md) | Sequence order modeling in Transformers, RoPE in modern LLMs (LLaMA-3, Mistral, Gemma), long-context scaling (YaRN), and Diffusion timesteps |

---

## 🗺️ Recommended Pedagogical Reading Order

For optimal conceptual continuity and zero mathematical gaps, study these guides in the following sequential order:

```
[01. Vectors & Matrices]
       │
       ▼
[02. Vector Norms & Inner Products]
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
[03. Dot Product & Similarity]           [04. Tensors & Shapes]
       │                                          │
       │                                          ▼
       │                                 [05. Tensor Broadcasting]
       │                                          │
       ▼                                          ▼
[06. Singular Value Decomposition]       [07. One-Hot Encoding]
                                                  │
                                                  ▼
                                         [08. Categorical Encodings & Embeddings]
                                                  │
                                                  ▼
                                         [09. Positional Encodings (RoPE)]
```

1. **[01-Vectors_and_Matrices.md](./01-Vectors_and_Matrices.md)** — Master vector arrows, linear transformations, matrix-vector multiplication, and outer-product gradients.
2. **[02-Vector_Norms_and_Inner_Products.md](./02-Vector_Norms_and_Inner_Products.md)** — Measure lengths, metric distances, unit balls, and normalization gradients.
3. **[03-Dot_Product_and_Similarity.md](./03-Dot_Product_and_Similarity.md)** — Understand directional alignment, cosine similarity, and the attention logit engine.
4. **[04-Tensors_and_Shapes.md](./04-Tensors_and_Shapes.md)** — Scale from 2D matrices to $N$-D strided memory layouts in PyTorch and CUDA.
5. **[05-Tensor_Broadcasting.md](./05-Tensor_Broadcasting.md)** — Learn how GPUs execute operations across mismatched tensor dimensions with zero memory copying.
6. **[06-Singular_Value_Decomposition.md](./06-Singular_Value_Decomposition.md)** — Decompose matrices into rotations and stretching, enabling 99% parameter compression with LoRA.
7. **[07-One_Hot_Encoding.md](./07-One_Hot_Encoding.md)** — Represent discrete categories as mutually orthogonal, equidistant vectors in high-dimensional space.
8. **[08-Encodings_Categorical_and_Embeddings.md](./08-Encodings_Categorical_and_Embeddings.md)** — Map discrete tokens into continuous semantic coordinates where words become vectors.
9. **[09-Positional_Encodings.md](./09-Positional_Encodings.md)** — Break permutation invariance and inject sequence order using Sinusoids, ALiBi, and Rotary Position Embeddings (RoPE).

---

## 🔗 Cross-Cluster Interconnections

- Return to the **[Grand Unified Concept Map & Mathematical Dependency Graph](../CONCEPT_MAP.md)** to inspect how this cluster connects across all 6 mathematical tiers.
- Navigate back to the **[MathsTerms Master Catalog](../README.md)**.
