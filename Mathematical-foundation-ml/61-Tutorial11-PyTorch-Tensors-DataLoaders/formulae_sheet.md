# Rapid Revision Formulae Sheet — Tutorial 11 : Pytorch - Tensors and Data Loaders

High-density mathematical ledger, tensor dimensionality maps, hardware memory invariants, and contrastive decision matrices for rapid revision of PyTorch tensor algebra and dataset engineering.

---

## 1. Master Equations Ledger

### 1.1 Physical Memory Offset Formula (Strided Layout)
For a rank-$k$ tensor $\mathcal{T} \in \mathbb{R}^{d_0 \times d_1 \times \dots \times d_{k-1}}$ stored at base pointer $P_0$ with stride vector $s = (s_0, s_1, \dots, s_{k-1})$:
$$
\text{Address}(i_0, i_1, \dots, i_{k-1}) = P_0 + \left( \sum_{r=0}^{k-1} i_r s_r \right) \times \text{sizeof}(\text{dtype})
$$
where for standard C-contiguous row-major storage:
$$
s_{k-1} = 1, \quad s_r = \prod_{j=r+1}^{k-1} d_j \quad (0 \le r < k-1)
$$

### 1.2 General Matrix Multiplication (GEMM Contraction)
For $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$:
$$
C = A @ B \implies C_{ij} = \sum_{p=1}^k A_{ip} B_{pj} \in \mathbb{R}^{m \times n}
$$
Inner dimension contraction: $(m \times k) \times (k \times n) \to (m \times n)$.

### 1.3 Coordinate-Wise Hadamard Product
For $A, B \in \mathbb{R}^{m \times n}$:
$$
C = A \odot B \implies C_{ij} = A_{ij} B_{ij} \in \mathbb{R}^{m \times n}
$$
Requires strictly identical or broadcast-compatible shapes; preserves dimensionality.

### 1.4 Algebraic Decomposition of 2D Convolution
For input receptive field patch $X \in \mathbb{R}^{K_h \times K_w}$ and filter kernel $W \in \mathbb{R}^{K_h \times K_w}$:
$$
Y = \sum_{u=1}^{K_h} \sum_{v=1}^{K_w} W_{uv} X_{uv} = \operatorname{vec}(W)^T \operatorname{vec}(X) = \operatorname{sum}(W \odot X)
$$

### 1.5 Image Tensor Normalization Transform
For raw uint8 image array $I \in \{0, 1, \dots, 255\}^{H \times W \times C}$:
$$
T(I)_{c, h, w} = \frac{I_{h, w, c}}{255.0} \in [0.0, 1.0]^{C \times H \times W}
$$

### 1.6 Empirical Mini-Batch Risk Collation
For batch size $B$ sampled from dataset $\mathcal{D}$:
$$
\hat{R}_B(\theta) = \frac{1}{B} \sum_{i=1}^B \ell(y_i, f_\theta(x_i))
$$

---

## 2. Tensor Dimensionality & Shape Ledger

| Operation | Input Tensors & Shapes | Output Tensor Shape | Mathematical Pre-Condition | Memory Implication |
|:----------|:-----------------------|:--------------------|:---------------------------|:-------------------|
| `torch.tensor(list)` | Python list of length $N$ | `(N,)` | Elements must share compatible numeric type | Allocates new contiguous memory buffer |
| `torch.from_numpy(arr)` | `np.ndarray` of shape `(d0, d1, ...)` | `(d0, d1, ...)` | Array must be contiguous or strided | Zero-copy view; shares existing memory pointer |
| `torch.cat([A, B], dim=0)` | $A \in \mathbb{R}^{m_1 \times n}, B \in \mathbb{R}^{m_2 \times n}$ | `(m1 + m2, n)` | All dimensions except dim 0 must match exactly | Allocates new contiguous buffer of size $(m_1+m_2)n$ |
| `torch.cat([A, B], dim=1)` | $A \in \mathbb{R}^{m \times n_1}, B \in \mathbb{R}^{m \times n_2}$ | `(m, n1 + n2)` | All dimensions except dim 1 must match exactly | Allocates new contiguous buffer of size $m(n_1+n_2)$ |
| `A @ B` (GEMM) | $A \in \mathbb{R}^{m \times k}, B \in \mathbb{R}^{k \times n}$ | `(m, n)` | Inner dimensions must match ($A.\text{shape}[1] == B.\text{shape}[0]$) | Allocates new buffer of size $m \times n$ (or reuses `out`) |
| `A * B` (Hadamard) | $A, B \in \mathbb{R}^{m \times n}$ | `(m, n)` | Shapes must match or broadcast | Pointwise operation; SIMD vectorized |
| `T.sum()` | $T \in \mathbb{R}^{d_0 \times \dots \times d_{k-1}}$ | `()` (Rank-0 Tensor) | None | Full scalar reduction; retains autograd graph |
| `T.sum().item()` | $T \in \mathbb{R}^{d_0 \times \dots \times d_{k-1}}$ | Native Python `float` | Tensor must have exactly 1 element (`numel() == 1`) | Copies scalar across PCIe/host boundary to Python runtime |
| `DataLoader` Batch | $B$ samples of shape `(C, H, W)` | `(B, C, H, W)` | All mini-batch samples must share identical spatial size | Collates individual sample tensors into 4D batch |

---

## 3. Mathematical Guarantees & Hardware Invariants

| Invariant / Law | Formal Statement | Violation Consequence / Error Mode | Architectural Rationale |
|:----------------|:-----------------|:-----------------------------------|:------------------------|
| **Device Matching Law** | $\forall \text{ op}(A, B): \text{device}(A) == \text{device}(B)$ | `RuntimeError: Expected all tensors to be on the same device` | Processors cannot directly access external memory across the PCIe bus without explicit asynchronous DMA transfers |
| **Zero-Copy NumPy Sharing** | $\text{ptr}(\text{torch.from_numpy}(X)) == \text{ptr}(X)$ | Modifying $X$ silently corrupts PyTorch tensor and vice versa | Avoids duplicating massive gigabyte-scale datasets when bridging NumPy pipelines into PyTorch |
| **Broadcasting Non-Duplication** | For dimension $r$ where $d_r = 1$, stride $s_r = 0$ | High RAM consumption if manually cloning arrays | Allows linear layers and loss functions to apply biases without memory reallocation |
| **Lazy Dataset Memory Bound** | $\text{RAM}_{\text{Dataset}} = O(N \times \text{size}(\text{metadata})) \ll O(N \times \text{size}(\text{data}))$ | System Out-Of-Memory (OOM) crash if raw images are loaded in `__init__` | Enables training on multi-terabyte datasets exceeding physical machine RAM capacity |
| **Item Scalar Detachment** | `loss.item()` returns a pure Python scalar float | GPU Out-Of-Memory memory leak if accumulating `running_loss += loss` | Prevents accumulating the entire computational tape across thousands of training steps |

---

## 4. Contrastive Decision Matrix

| Operational Need | Recommended Method | Inferior / Anti-Pattern Alternative | Rationale & Failure Mode |
|:-----------------|:-------------------|:-----------------------------------|:-------------------------|
| **Matrix Multiplication** | `A @ B` or `torch.matmul(A, B)` | `A * B` | `*` executes coordinate-wise Hadamard multiplication; fails with shape mismatch or produces incorrect point-wise products. |
| **NumPy Array Conversion** | `torch.from_numpy(arr)` | `torch.tensor(arr)` | `torch.tensor()` copies memory, doubling RAM usage; `from_numpy()` creates an instantaneous zero-copy view. |
| **Pre-Allocated Matrix GEMM** | `torch.mm(A, B, out=C)` | Creating new `C = A @ B` in a tight loop | Pre-allocating `out` eliminates garbage collection pauses and dynamic memory re-allocations in high-throughput inference engines. |
| **Extracting Metric for Logging** | `loss.item()` | `float(loss)` or `loss` | Keeping `loss` retains the GPU autograd computation graph; `loss.item()` extracts the float primitive and frees VRAM. |
| **Moving Tensors to GPU** | `tensor.to(device)` | `tensor.cuda()` hardcoded | `.to(device)` enables transparent execution on CPU, CUDA, or Apple MPS hardware without code changes. |
| **Dataset Architecture** | Custom `Dataset` with lazy `__getitem__` | Loading all images into a Python list in `__init__` | Eager loading causes OOM on real-world datasets; lazy loading streams data on-demand during batch collation. |

---

## 5. Computational Complexity & Memory Footprints

- **General Matrix Multiplication (GEMM):**
  $$
  \mathcal{O}(m \cdot k \cdot n) \text{ FLOPs}, \quad \text{Memory: } \mathcal{O}(m \cdot n) \text{ floats}
  $$
- **Coordinate-Wise Hadamard Product:**
  $$
  \mathcal{O}(m \cdot n) \text{ FLOPs}, \quad \text{Memory: } \mathcal{O}(m \cdot n) \text{ floats}
  $$
- **Tensor Concatenation along dim $d$:**
  $$
  \mathcal{O}(\text{total elements}) \text{ byte copies}, \quad \text{Memory: } \mathcal{O}\left(\left(\sum d_i\right) \cdot \prod_{j \neq d} d_j\right)
  $$
- **Dataset Streaming Footprint:**
  $$
  \text{Active RAM} = \mathcal{O}(B \cdot C \cdot H \cdot W \cdot \text{num\_workers}) \text{ bytes (bounded and constant across epochs)}
  $$

---

## 6. Numerical Stability & Traps

| Failure Mode / Trap | Root Mechanism | Mathematical / Hardware Guardrail | Production Code Pattern |
|:--------------------|:---------------|:-----------------------------------|:------------------------|
| **Division by Zero in Normalization** | Zero standard deviation or zero-norm denominator causes $\frac{x}{0} \to \text{NaN}$ | Add numerical stability epsilon $\epsilon = 10^{-7}$ or $10^{-8}$ inside square root / denominator | `norm_x = (x - mean) / torch.sqrt(var + 1e-7)` |
| **Log-Domain Probability Underflow** | Small probability $p \approx 0$ evaluated as $\log(\operatorname{softmax}(z))$ collapses to $-\infty$ | Use log-domain formulations with LogSumExp trick directly in single kernel | `log_p = torch.nn.functional.log_softmax(z, dim=-1)` |
| **Exploding Gradient Dynamic Range** | Unbounded matrix multiplication norms accumulate across layers | Bound maximum gradient norm via clipping before optimizer stepping | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` |
| **Extreme Activation Overflow** | Exponential or polynomial operations exceeding float32 ceiling ($\approx 3.4 \times 10^{38}$) | Clamp dynamic range of activations before non-linear exponentiation | `clamped_z = torch.clamp(z, min=-50.0, max=50.0)` |
| **Autograd Graph GPU Memory Leak** | Accumulating raw tensor scalar loss keeps computation graph alive across epochs | Detach scalar primitive into native Python float via `.item()` | `running_loss += loss.item()` |
| **NumPy Shared Memory Mutation Trap** | `torch.from_numpy` shares underlying storage; modifying NumPy array silently corrupts tensor | Explicitly clone when memory independence is required | `t = torch.from_numpy(arr).clone()` |
