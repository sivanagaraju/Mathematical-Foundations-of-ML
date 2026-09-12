# Pedagogical Prerequisites: Tutorial 16 VQ-VAE Implementation

Welcome to the foundational implementation guide for Vector Quantised Variational Autoencoders (VQ-VAE) in PyTorch. Before writing embedding modules, straight-through identities, and codebook perplexity trackers, master these fundamental mathematical pillars.

---

## Rosetta Stone Symbol Mapping

| Mathematical Symbol | Spoken Reading | Mathematical Concept | Plain-English Intuition |
|:---|:---|:---|:---|
| $\mathcal{E} \in \mathbb{R}^{K \times D}$ | "script ee" | Codebook dictionary embedding matrix | The master dictionary holding $K$ standard visual prototypes of size $D$ |
| $z_e(x)$ | "zee-ee of eks" | Continuous encoder feature grid | The unquantized continuous features output by the convolutional encoder |
| $z_q(x)$ | "zee-kyoo of eks" | Quantized discrete feature grid | The discrete prototype vectors swapped in place of the continuous features |
| `.detach()` | "dot dee-TACH" | Autograd tensor detachment operator | A scissors operator cutting a tensor out of the backward computational graph |
| $z_e + (z_q - z_e).\text{detach}()$ | "zee-ee plus quantity zee-kyoo minus zee-ee detached" | Straight-Through Estimator (STE) trick | A clever code trick that outputs $z_q$ forward but passes gradients to $z_e$ backward |
| $\text{Perplexity} \in [1, K]$ | "per-PLEK-sih-tee" | Codebook entropy exponent | A count of how many unique dictionary words the network is actively using |
| $D_{\text{dist}} \in \mathbb{R}^{N \times K}$ | "dee dist" | Pairwise Euclidean distance matrix | A table recording how far every pixel feature is from every dictionary prototype |

---

## Curriculum & Prerequisite Bridges

| Sibling Module | Core Mathematical Concept | How it Unlocks This Lecture |
|:---|:---|:---|
| [Lec 12: VQ-VAE Foundations](../23-Lec12-Vector-Quantised-VAE/NOTES.md) | Theoretical formulation of discrete codebooks | Establishes the tripartite loss and STE theory implemented in this tutorial |
| [Tutorial 15: VAE & Beta-VAE](../26-Tutorial15-VAE-Beta-VAE-Implementation/NOTES.md) | Continuous latent autoencoders | Contrasts continuous Gaussian bottlenecks with discrete codebook lookups |
| [MathsTerms: Categorical Embeddings](../../MathsTerms/02-Linear-Algebra-Geometry-and-Tensors/08-Encodings_Categorical_and_Embeddings.md) | PyTorch `nn.Embedding` mechanics | Formulates dictionary lookup matrices and weight storage |
| [MathsTerms: Dot Product & Similarity](../../MathsTerms/02-Linear-Algebra-Geometry-and-Tensors/03-Dot_Product_and_Similarity.md) | Vector inner products and metric geometry | Explains why expanding squared norms into matrix products speeds up distance checks |

---

## Pillar 1: Codebook Embedding Initialization and Reshaping
<a id="p1-codebook-embedding"></a>

### 👶 ELI5 Intuition
Imagine a post office with 512 numbered mailboxes. When a package arrives from the factory, you can't deliver it in continuous mid-air; you have to put it into the closest numbered mailbox. PyTorch's `nn.Embedding` is that exact wall of 512 physical mailboxes, where each box stores a specific prototype feature.

### 🔢 Concrete Micro-Numbers
Let codebook size $K = 512$ and embedding dimension $D = 64$.
The weight matrix has shape `(512, 64)`.
Total parameter count: $512 \times 64 = 32,768$ floats.
Initialized uniformly within $[ -1/512, +1/512 ] = [-0.00195, +0.00195]$.

### 📐 Formal Math
The embedding dictionary $\mathcal{E} = \{e_k\}_{k=0}^{K-1}$ is parameterized by weight matrix $W \in \mathbb{R}^{K \times D}$.
A batch of images produces continuous tensor $z_e \in \mathbb{R}^{B \times D \times H \times W}$.
To enable row-wise distance evaluations against codebook rows, the tensor is permuted and flattened:
$$z_e \xrightarrow{\text{permute}(0, 2, 3, 1)} (B, H, W, D) \xrightarrow{\text{contiguous}().\text{view}(-1, D)} z_{\text{flat}} \in \mathbb{R}^{N \times D}$$
where $N = B \times H \times W$.

### 💻 Runnable Code Snippet
```python
import torch
import torch.nn as nn

B, D, H, W = 2, 8, 4, 4
K = 16
emb = nn.Embedding(K, D)
emb.weight.data.uniform_(-1.0 / K, 1.0 / K)

ze = torch.randn(B, D, H, W)
ze_perm = ze.permute(0, 2, 3, 1).contiguous()
ze_flat = ze_perm.view(-1, D)

assert ze_flat.shape == (32, 8)
assert emb.weight.shape == (16, 8)
print(f"Pillar 1 Clean: Flattened {B*H*W} vectors of dimension {D}")
```

### 🎯 Diagnostic Mini-Check / Self-Test
- **Question:** What happens if an engineer forgets to call `.contiguous()` between `.permute()` and `.view()`?
- **Answer:** PyTorch raises a `RuntimeError` because `.view()` requires a contiguous memory buffer, while `.permute()` changes tensor strides without copying data.

---

## Pillar 2: Vectorized Pairwise Distance Expansion
<a id="p2-vectorized-distance-expansion"></a>

### 👶 ELI5 Intuition
If you want to measure the distance from 100 school buses to 50 parking garages, you don't send one driver with a tape measure to walk back and forth 5,000 times. You use satellite GPS to calculate all 5,000 distances instantly in a single computer calculation.

### 🔢 Concrete Micro-Numbers
Let $z = [1.0, 2.0]$ and $e = [0.8, 1.9]$.
- Direct L2: $(1.0 - 0.8)^2 + (2.0 - 1.9)^2 = 0.04 + 0.01 = 0.05$.
- Expanded formulation:
  $\|z\|^2 = 1.0^2 + 2.0^2 = 1 + 4 = 5.0$.
  $\|e\|^2 = 0.8^2 + 1.9^2 = 0.64 + 3.61 = 4.25$.
  Dot product: $z \cdot e = 1.0(0.8) + 2.0(1.9) = 0.8 + 3.8 = 4.6$.
  Result: $\|z\|^2 + \|e\|^2 - 2(z \cdot e) = 5.0 + 4.25 - 2(4.6) = 9.25 - 9.20 = 0.05$.
Both methods give the exact same answer 0.05!

### 📐 Formal Math
The squared Euclidean distance expands as:
$$\|z_i - e_j\|_2^2 = \|z_i\|_2^2 + \|e_j\|_2^2 - 2 z_i^T e_j$$
In vectorized PyTorch operations:
$$\text{distances} = \sum z^2_{\text{dim}=1} + \sum e^2_{\text{dim}=1} - 2 (z \cdot e^T)$$
This replaces $N \times K$ distance calculations with one matrix multiplication $z e^T$.

### 💻 Runnable Code Snippet
```python
import torch

z = torch.tensor([[1.0, 2.0]])
e = torch.tensor([[0.8, 1.9], [2.0, 3.0]])

# Expanded distance
z_sq = torch.sum(z**2, dim=1, keepdim=True) # (1, 1)
e_sq = torch.sum(e**2, dim=1)               # (2,)
dots = 2 * torch.matmul(z, e.t())           # (1, 2)
dists = z_sq + e_sq - dots

assert torch.isclose(dists[0, 0], torch.tensor(0.05))
assert torch.argmin(dists, dim=1).item() == 0
print(f"Pillar 2 Clean: Expanded distance correctly identified nearest centroid: {dists.tolist()}")
```

### 🎯 Diagnostic Mini-Check / Self-Test
- **Question:** What is the computational advantage of computing distances via matrix multiplication on modern GPUs?
- **Answer:** Matrix multiplications run directly on GPU Tensor Cores, executing hundreds of times faster than element-wise difference operations.

---

## Pillar 3: Codebook Usage Perplexity
<a id="p3-codebook-perplexity"></a>

### 👶 ELI5 Intuition
Imagine a teacher who gives a vocabulary test with 100 new words. If the students only use 3 words over and over ("good", "bad", "okay"), the effective vocabulary is tiny (perplexity = 3). If the students actively use all 100 words in equal measure, the effective vocabulary is large (perplexity = 100).

### 🔢 Concrete Micro-Numbers
Suppose codebook has $K = 4$ entries.
- Case 1 (Equal usage): $p = [0.25, 0.25, 0.25, 0.25]$.
  Entropy: $H = -4 \times (0.25 \log 0.25) = -\log(0.25) = \log(4) \approx 1.3863$.
  Perplexity: $\exp(1.3863) = 4.0$ (All 4 codes actively used!).
- Case 2 (Total collapse): $p = [1.0, 0.0, 0.0, 0.0]$.
  Entropy: $H = -(1.0 \log 1.0) = 0.0$.
  Perplexity: $\exp(0.0) = 1.0$ (Only 1 code used: complete collapse!).

### 📐 Formal Math
Let empirical frequency of code $k$ across mini-batch tokens be $p_k = \frac{1}{N} \sum_{n=1}^N \mathbb{I}[k^*(n) = k]$.
The Shannon entropy is:
$$H(p) = -\sum_{k=1}^K p_k \log(p_k + \epsilon)$$
The codebook perplexity is defined as:
$$\text{Perplexity} = \exp(H(p)) = \prod_{k=1}^K p_k^{-p_k}$$

### 💻 Runnable Code Snippet
```python
import torch

# Case 1: Uniform usage over 4 codes
p_uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])
perp_uniform = torch.exp(-torch.sum(p_uniform * torch.log(p_uniform)))
assert torch.isclose(perp_uniform, torch.tensor(4.0))

# Case 2: Collapsed usage
p_collapsed = torch.tensor([1.0, 0.0, 0.0, 0.0])
perp_collapsed = torch.exp(-torch.sum(p_collapsed * torch.log(p_collapsed + 1e-10)))
assert torch.isclose(perp_collapsed, torch.tensor(1.0), atol=1e-3)
print(f"Pillar 3 Clean: Uniform Perplexity = {perp_uniform.item():.2f}, Collapsed = {perp_collapsed.item():.2f}")
```

### 🎯 Diagnostic Mini-Check / Self-Test
- **Question:** If a VQ-VAE model with $K = 1024$ codebook vectors displays perplexity of $8.5$ after 5 epochs, what problem is occurring?
- **Answer:** Severe codebook collapse: out of 1024 available prototypes, the encoder is concentrating almost all probability mass on fewer than 9 vectors.

---

## Pillar 4: The Straight-Through Detach Trick
<a id="p4-ste-detach-trick"></a>

### 👶 ELI5 Intuition
Imagine you have an electric train track with an insulated plastic bridge in the middle that doesn't conduct electricity. To make the train run, you clip a copper jumper wire across the plastic bridge. The train travels on the plastic tracks forward, but the electric signal flows through the jumper wire backward. That jumper wire is `.detach()`.

### 🔢 Concrete Micro-Numbers
Suppose continuous input is $z_e = 3.5$ and nearest code is $z_q = 4.0$.
In forward pass:
$$z_{\text{out}} = z_e + (z_q - z_e).\text{detach}() = 3.5 + (4.0 - 3.5) = 3.5 + 0.5 = 4.0$$
The output is numerically equal to the discrete code $z_q = 4.0$.
In backward pass, let loss $\mathcal{L} = (z_{\text{out}} - 10.0)^2 = (4.0 - 10.0)^2 = 36.0$.
Upstream gradient: $\frac{\partial \mathcal{L}}{\partial z_{\text{out}}} = 2(4.0 - 10.0) = -12.0$.
Because $(z_q - z_e).\text{detach}()$ has zero derivative:
$$\frac{\partial z_{\text{out}}}{\partial z_e} = \frac{\partial z_e}{\partial z_e} + 0 = 1.0 \implies \frac{\partial \mathcal{L}}{\partial z_e} = -12.0 \times 1.0 = -12.0$$

### 📐 Formal Math
The computational operator:
$$z_q^{\text{STE}} \equiv z_e + \text{sg}[z_q - z_e]$$
satisfies:
$$\text{Forward: } z_q^{\text{STE}} = z_q$$
$$\text{Backward: } \nabla_{z_e} \mathcal{L} = \nabla_{z_q^{\text{STE}}} \mathcal{L} \cdot \mathbf{I} = \nabla_{z_q} \mathcal{L}$$
This bypasses the zero derivative of the argmin function without altering the forward value.

### 💻 Runnable Code Snippet
```python
import torch

ze = torch.tensor([3.5], requires_grad=True)
zq_val = torch.tensor([4.0])

# STE detach trick
zq_ste = ze + (zq_val - ze).detach()
loss = (zq_ste - 10.0)**2
loss.backward()

assert zq_ste.item() == 4.0
assert ze.grad.item() == -12.0
print(f"Pillar 4 Clean: STE trick forward={zq_ste.item()}, backward grad={ze.grad.item()}")
```

### 🎯 Diagnostic Mini-Check / Self-Test
- **Question:** What happens if an engineer writes `z_q = z_e + (z_q - z_e)` without calling `.detach()`?
- **Answer:** Autograd attempts to differentiate through the lookup operation that produced $z_q$, causing graph cycles or failing to differentiate the encoder.
