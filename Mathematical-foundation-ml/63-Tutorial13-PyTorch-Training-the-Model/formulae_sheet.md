# Rapid Revision Formulae Sheet — Tutorial 13 : Pytorch - Training the Model

High-density mathematical ledger, tensor dimensionality maps, hardware memory invariants, optimization recursions, and contrastive decision matrices for rapid revision of neural network training loops, evaluation protocols, and model serialization.

---

## 1. Master Equations Ledger

### 1.1 Empirical Risk Minimization (Mini-Batch Stochastic Gradient Descent)
For a parameterized model $f_\theta: \mathcal{X} \to \mathbb{R}^K$ and mini-batch $B \subset \mathcal{D}$ with $|B| = b$:
$$
\mathcal{L}_B(\theta) = \frac{1}{|B|} \sum_{i \in B} \ell\left(f_\theta(x_i), y_i\right)
$$
The empirical gradient estimator is:
$$
g_t = \nabla_\theta \mathcal{L}_B(\theta_t) = \frac{1}{|B|} \sum_{i \in B} \nabla_\theta \ell\left(f_\theta(x_i), y_i\right) \quad \text{where } \mathbb{E}_{B \sim \mathcal{D}}[g_t] = \nabla_\theta \mathcal{L}_{\mathcal{D}}(\theta_t)
$$

### 1.2 Categorical Cross-Entropy Loss with LogSumExp Trick
For raw logit vector $z \in \mathbb{R}^K$ and ground-truth scalar label $y \in \{0, \dots, K-1\}$:
$$
\ell_{\text{CE}}(z, y) = -\log p_y = -\log \left( \frac{e^{z_y}}{\sum_{j=1}^K e^{z_j}} \right) = -z_y + \log \left( \sum_{j=1}^K e^{z_j} \right)
$$
Numerically stabilized via coordinate maximum $m = \max_{j} z_j$:
$$
\ell_{\text{CE}}(z, y) = -z_y + m + \log \left( \sum_{j=1}^K e^{z_j - m} \right)
$$
Gradient with respect to logit $z_k$:
$$
\frac{\partial \ell_{\text{CE}}}{\partial z_k} = p_k - \mathbb{I}(y = k) = \frac{e^{z_k}}{\sum_j e^{z_j}} - \mathbb{I}(y = k)
$$

### 1.3 Stochastic Gradient Descent with Classical Momentum
With learning rate $\eta > 0$ and momentum factor $\beta \in [0, 1)$:
$$
v_t = \beta v_{t-1} + g_t
$$
$$
\theta_t = \theta_{t-1} - \eta v_t
$$

### 1.4 Adam (Adaptive Moment Estimation) Recursions
With step size $\eta$, first-moment decay $\beta_1 \in [0, 1)$, second-moment decay $\beta_2 \in [0, 1)$, and stability scalar $\epsilon > 0$:
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\text{First Moment — Direction})
$$
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^{\odot 2} \quad (\text{Second Moment — Element-wise Variance})
$$
Bias-corrected moments:
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$
Parameter update step:
$$
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t
$$

### 1.5 Gradient Accumulation Mechanics
For gradient buffer $G \in \mathbb{R}^{\dim(\theta)}$ across $K$ consecutive mini-batches without intermediate zeroing:
$$
G_K = \sum_{k=1}^K \nabla_\theta \mathcal{L}_{B_k}(\theta) = K \cdot \nabla_\theta \left( \frac{1}{K} \sum_{k=1}^K \mathcal{L}_{B_k}(\theta) \right)
$$
Simulating effective virtual batch size $B_{\text{eff}} = K \cdot |B|$ requires scaling loss by $1/K$ before `.backward()`.

### 1.6 Relational Classification Accuracy & Zero-One Loss
For evaluation dataset $D_{\text{eval}}$ with $|D_{\text{eval}}| = N_{\text{eval}}$:
$$
\text{Accuracy} = \frac{1}{N_{\text{eval}}} \sum_{i=1}^{N_{\text{eval}}} \mathbb{I}\left( \operatorname{argmax}_{k \in \{0, \dots, K-1\}} z_{i,k} = y_i \right)
$$
$$
\mathcal{R}_{0-1}(\theta) = 1.0 - \text{Accuracy} = \frac{1}{N_{\text{eval}}} \sum_{i=1}^{N_{\text{eval}}} \mathbb{I}\left( \hat{y}_i \neq y_i \right)
$$

### 1.7 Model State Dictionary Functional Mapping
The parameter and buffer state of model $\mathcal{M}$ is defined as the ordered key-value mapping:
$$
\mathcal{S}(\mathcal{M}) = \left\{ (k, v.\operatorname{clone}()) \;\middle|\; (k, v) \in \operatorname{named\_parameters}(\mathcal{M}) \cup \operatorname{named\_buffers}(\mathcal{M}) \right\}
$$
Exact state restoration into fresh instance $\mathcal{M}'$:
$$
\mathcal{M}'.\operatorname{load\_state\_dict}(\mathcal{S}(\mathcal{M})) \implies \forall k, \quad \|\theta_k^{(\mathcal{M})} - \theta_k^{(\mathcal{M}')}\|_\infty = 0
$$

---

## 2. Tensor Dimensionality & Shape Ledger

| Operation / Step | Input Tensors & Shapes | Output Tensor Shape | Mathematical Invariant | Memory / Execution Implication |
|:---|:---|:---|:---|:---|
| **Batch Ingestion** | DataLoader batch `(X, y)` | `X: [B, 1, 28, 28]`, `y: [B]` | Batch size $B = 64$ (or remainder for final batch) | Transferred to target device via `.to(device)` |
| **Model Forward Pass** | `X: [B, 1, 28, 28]` | `logits: [B, 10]` | Unnormalized real scores: $\operatorname{dim}_1 = 10$ classes | Allocates forward activation graph unless wrapped in `no_grad()` |
| **Loss Calculation** | `logits: [B, 10]`, `y: [B]` (int64) | `loss: []` (0-dim scalar) | Scalar empirical mean cross-entropy: $\mathcal{L} \ge 0$ | Root node of autograd DAG; retains reference to entire execution graph |
| **`optimizer.zero_grad()`** | Parameter tensors $\theta$ | $\nabla_\theta \mathcal{L} \leftarrow \mathbf{0}$ or `None` | Prevents inter-iteration gradient summing | `set_to_none=True` frees gradient VRAM buffers instead of zero-filling |
| **`loss.backward()`** | Scalar `loss: []` | $\forall \theta: \theta.\text{grad} \in \mathbb{R}^{\text{shape}(\theta)}$ | Computes vector-Jacobian products via reverse-mode AD | Teardown of intermediate activation tensors in dynamic computation graph |
| **`optimizer.step()`** | $\theta$ and $\theta.\text{grad}$ | In-place update $\theta \leftarrow \theta - \Delta\theta$ | Invariant tensor shapes: $\text{shape}(\theta_{t}) == \text{shape}(\theta_{t-1})$ | Mutates weight tensors in-place; updates optimizer state ($m_t, v_t$) |
| **Prediction Argmax** | `logits: [B, 10]` | `preds: [B]` (dtype `torch.int64`) | $\hat{y}_i = \operatorname{argmax}_j z_{ij}$ | Non-differentiable discrete integer index tensor |
| **Boolean Equality** | `preds: [B]`, `y: [B]` | `correct_mask: [B]` (bool) | Element-wise indicator $\mathbb{I}(\hat{y}_i == y_i)$ | Compact 1-byte boolean tensor |
| **Accuracy Accumulator** | `correct_mask: [B]` | `batch_correct: []` (int/float) | Sum of matched predictions: $\sum_{i=1}^B \mathbb{I}(\hat{y}_i == y_i)$ | Cast to scalar float using `.item()` to decouple from PyTorch runtime |

---

## 3. Mathematical Guarantees & Hardware Invariants

| Invariant / Law | Formal Statement | Violation Consequence / Error Mode | Architectural Rationale |
|:---|:---|:---|:---|
| **Gradient Clear Invariant** | `optimizer.zero_grad()` must execute prior to `loss.backward()` | `RuntimeError` or exploding gradients: $\theta.\text{grad}$ accumulates sum across multiple mini-batches | PyTorch autograd is designed to accumulate gradients into `.grad` to facilitate multi-step batching and distributed reductions |
| **Scalar Metric Memory Invariant** | Accumulate training/eval loss via `loss.item()`, never `loss` | Catastrophic GPU VRAM exhaustion (`CUDA Out Of Memory`) | A raw loss tensor retains the full dynamic computation graph of the entire batch in memory; `.item()` extracts a pure Python float |
| **Evaluation Mode Invariant** | `model.eval()` must be set prior to validation/testing passes | Corrupted test accuracy and stochastic prediction nondeterminism | Toggles stochastic layers (Dropout, DropPath) to inactive identity and fixes BatchNorm to running population statistics |
| **Gradient Context Invariant** | `with torch.no_grad():` must wrap all evaluation and test loops | Massive unnecessary VRAM allocation ($3\times$ to $5\times$) and reduced inference speed | Disables creation of intermediate forward activation graph tapes required only for reverse-mode backpropagation |
| **Cross-Device Collocation Invariant** | $\operatorname{device}(X) \equiv \operatorname{device}(y) \equiv \operatorname{device}(\theta)$ | `RuntimeError: Expected all tensors to be on the same device` | Tensor operations (GEMM, reductions) require all operands to reside in the exact same memory space (Host RAM vs Device VRAM) |
| **Weights-Only Serialization Invariant** | `torch.load(path, weights_only=True)` must be used for model loading | Critical security vulnerability (Arbitrary Code Execution via Python pickle deserialization) | Python pickle allows arbitrary object instantiation and code execution; `weights_only=True` restricts unpickling to pure numerical tensor stores |

---

## 4. Contrastive Decision Matrix

| Operational Need | Recommended Method | Inferior / Anti-Pattern Alternative | Rationale & Failure Mode |
|:---|:---|:---|:---|
| **Gradient Reset** | `optimizer.zero_grad(set_to_none=True)` | `optimizer.zero_grad()` (default) or omitting | `set_to_none=True` releases memory buffers and skips zero-write GEMMs, yielding a $5\text{--}10\%$ memory and execution speedup. |
| **Metric Logging** | `total_loss += loss.item() * batch_size` | `total_loss += loss` | Directly accumulating the loss tensor chains every mini-batch's autograd graph across the entire epoch, triggering CUDA OOM. |
| **Inference Pass** | `with torch.no_grad(): output = model(x)` | `output = model(x)` without context | Running inference without `no_grad()` allocates activation nodes for backpropagation, wasting GPU memory and compute. |
| **Model State for Eval** | `model.eval()` before test loop | Running eval in default training mode | Dropout remains active (randomly dropping features) and BatchNorm continues updating statistics with test batch distributions. |
| **Optimizer Selection** | `torch.optim.Adam(model.parameters(), lr=1e-3)` | `torch.optim.SGD(model.parameters(), lr=0.1)` without momentum | Adam automatically adapts coordinate-wise step sizes using running moments, converging rapidly on non-convex multi-layer topologies. |
| **Checkpoint Storage** | `torch.save(model.state_dict(), path)` | `torch.save(model, path)` | Saving the whole model serializes the exact Python class definition and directory bindings, causing crashes when code refactors occur. |
| **Checkpoint Deserialization**| `torch.load(path, weights_only=True)` | `torch.load(path)` (default) | Default `torch.load` is vulnerable to arbitrary remote code execution via pickle exploits. `weights_only=True` guarantees safety. |

---

## 5. Computational Complexity & Memory Footprints

### 5.1 Training Iteration Compute (FLOPs per Mini-Batch of Size $B$)
For an MLP with layer dimensions $[d_0, d_1, \dots, d_L]$:
- **Forward Pass:** $\approx 2B \sum_{l=1}^L d_{l-1} d_l$ FLOPs.
- **Backward Pass:** $\approx 4B \sum_{l=1}^L d_{l-1} d_l$ FLOPs (2 GEMMs: one for input activation gradient $\nabla_{h_{l-1}}$, one for parameter gradient $\nabla_{W_l}$).
- **Optimizer Step:** $\mathcal{O}(P)$ operations, where $P = \sum_{l=1}^L (d_{l-1} d_l + d_l)$ is total parameter count.
- **Total Compute per Batch:** $\text{FLOPs}_{\text{train}} \approx 6B \sum_{l=1}^L d_{l-1} d_l + \mathcal{O}(P)$.

### 5.2 Memory Footprint Invariants (Single-Precision float32)
For total parameter count $P$:
- **Model Parameters:** $4P$ bytes.
- **Gradient Buffers:** $4P$ bytes.
- **Optimizer States:**
  - Standard SGD: $0$ extra bytes.
  - SGD with Momentum: $4P$ bytes (1 velocity tensor per parameter).
  - Adam / AdamW: $8P$ bytes ($4P$ for first moment $m_t$ + $4P$ for second moment $v_t$).
- **Activation Buffers (Training):** $\mathcal{O}\left(B \sum_{l=1}^{L-1} d_l\right)$ bytes retained in memory for backward pass.
- **Activation Buffers (Inference under `no_grad()`):** $\mathcal{O}\left(B \max_{l} d_l\right)$ bytes (only active layer activations retained; intermediate layers freed immediately).

---

## 6. Numerical Stability & Traps

| Failure Mode / Trap | Root Mechanism | Mathematical / Hardware Guardrail | Production Code Pattern |
|:---|:---|:---|:---|
| **Adam Division by Zero** | Small parameter gradients cause second moment $\hat{v}_t \to 0$, leading to $\eta / \sqrt{\hat{v}_t} \to \infty$ | Add numerical stability epsilon $\epsilon$ inside denominator: $\frac{1}{\sqrt{\hat{v}_t} + \epsilon}$ | PyTorch default `eps=1e-8` in `torch.optim.Adam` |
| **Log-Domain Softmax Underflow** | Evaluating $\log(\operatorname{Softmax}(z))$ when $z_k \ll 0$ rounds probabilities to $0.0$, causing $\log(0) \to -\infty$ | LogSumExp numerical identity: $\log \sum e^{z_j} = m + \log \sum e^{z_j - m}$ where $m = \max_j z_j$ | Pass raw unnormalized logits directly into `nn.CrossEntropyLoss()` |
| **Gradient Accumulation Without Scaling** | Accumulating gradients across $K$ sub-batches without dividing loss by $K$ | Scale mini-batch loss by accumulator factor: $\mathcal{L}_{\text{scaled}} = \mathcal{L}_k / K$ | `(loss / accumulation_steps).backward()` |
| **Evaluation Mode Memory Retention** | Forgetting `with torch.no_grad():` during validation loop | Dynamic computation graph records all evaluation operations, causing GPU VRAM spike | `with torch.no_grad(): for X, y in val_loader: ...` |
| **Cross-Device Scalar Comparison Panic** | Comparing tensor on GPU with CPU integer directly inside conditional loops | Transfer tensor to CPU or extract scalar value via `.item()` | `if pred.item() == target.item():` |
| **State Dict Key Mismatch on Load** | Loading weights trained with `nn.DataParallel` (prefix `module.`) into single-GPU model | Strip `module.` prefix from keys before calling `load_state_dict` | `{k.replace('module.', ''): v for k, v in state.items()}` |
