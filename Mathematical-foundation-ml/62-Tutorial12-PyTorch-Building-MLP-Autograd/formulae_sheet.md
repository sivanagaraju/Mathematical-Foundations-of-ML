# Rapid Revision Formulae Sheet — Tutorial 12 : Pytorch - Building MLP and Auto Grad

High-density mathematical ledger, tensor dimensionality maps, hardware memory invariants, autograd mechanics, and contrastive decision matrices for rapid revision of neural network construction and automatic differentiation.

---

## 1. Master Equations Ledger

### 1.1 Multi-Layer Perceptron Layer Transformation
For layer $l \in \{1, \dots, L\}$ with weights $W_l$ and bias $b_l$:
$$
h_l = \sigma\left( W_l^T h_{l-1} + b_l \right)
$$
where in PyTorch `nn.Linear` convention with input $X \in \mathbb{R}^{B \times d_{\text{in}}}$:
$$
Y = X W^T + \mathbf{1}_B b^T \in \mathbb{R}^{B \times d_{\text{out}}}
$$

### 1.2 Rectified Linear Unit (ReLU) & Subgradient
For coordinate scalar $z \in \mathbb{R}$:
$$
\sigma(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}, \quad \frac{\partial \sigma}{\partial z} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}
$$

### 1.3 Softmax Posterior Probability Normalization
For logit vector $z \in \mathbb{R}^K$:
$$
p_k = P(Y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} \quad \text{such that } \sum_{k=1}^K p_k = 1.0
$$

### 1.4 Optimal Bayes Decision Rule (Zero-One Loss)
$$
\hat{y} = \operatorname{argmax}_{k \in \{0, \dots, K-1\}} p_k \equiv \operatorname{argmax}_{k \in \{0, \dots, K-1\}} z_k
$$

### 1.5 Reverse-Mode Vector-Jacobian Product (VJP)
For node output $y = f(x)$ with upstream gradient vector $v = \nabla_y \mathcal{L}$:
$$
\nabla_x \mathcal{L} = v^T J = v^T \left( \frac{\partial y}{\partial x} \right) = \sum_{i=1}^m v_i \frac{\partial y_i}{\partial x}
$$

### 1.6 Analytical Gradients for Affine Loss Graph
For loss $\mathcal{L} = \frac{1}{2N} \sum_{i=1}^N \|x_i W + b - y_i\|_2^2$ with error residual $E = (XW + b - Y)$:
$$
\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{N} X^T E, \quad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \mathbf{1}_N^T E
$$

---

## 2. Tensor Dimensionality & Shape Ledger

| Operation | Input Tensors & Shapes | Output Tensor Shape | Mathematical Invariant | Memory Implication |
|:----------|:-----------------------|:--------------------|:-----------------------|:-------------------|
| `nn.Flatten()` | `[B, 1, 28, 28]` or `[B, 28, 28]` | `[B, 784]` | Flattens all spatial axes while preserving leading batch dimension $B$ | Zero-copy strided view if contiguous; no new allocation |
| `nn.Linear(784, 512)` | Input: `[B, 784]`, Weight: `[512, 784]`, Bias: `[512]` | `[B, 512]` | Inner feature dimension 784 contracts via GEMM: $[B, 784] \times [784, 512] \to [B, 512]$ | Allocates intermediate activation tensor of size $B \times 512 \times 4$ bytes |
| `nn.ReLU()` | `[B, 512]` | `[B, 512]` | Shape invariant; element-wise piecewise linear projection | In-place variant `nn.ReLU(inplace=True)` saves activation memory |
| `nn.Linear(512, 10)` | Input: `[B, 512]`, Weight: `[10, 512]`, Bias: `[10]` | `[B, 10]` | Generates unnormalized logit vector $z \in \mathbb{R}^{B \times 10}$ | Allocates output logit tensor |
| `nn.Softmax(dim=1)` | `[B, 10]` | `[B, 10]` | Probabilities bounded in $[0, 1]$; $\sum_{j=1}^{10} P_{ij} = 1.0$ | Allocates posterior probability tensor |
| `torch.argmax(dim=1)` | `[B, 10]` | `[B]` | Returns discrete class index vector with dtype `torch.int64` | Discards floating logits; non-differentiable discrete integer tensor |
| `loss.backward()` | Scalar Loss `[]` (Rank-0 Tensor) | Parameter `.grad` matching parameter shapes | $\text{shape}(W.\text{grad}) == \text{shape}(W)$, $\text{shape}(b.\text{grad}) == \text{shape}(b)$ | Populates gradient buffers; tears down intermediate activation DAG |

---

## 3. Mathematical Guarantees & Hardware Invariants

| Invariant / Law | Formal Statement | Violation Consequence / Error Mode | Architectural Rationale |
|:----------------|:-----------------|:-----------------------------------|:------------------------|
| **Module Initialization Contract** | `super().__init__()` must precede child assignments | `AttributeError: cannot assign module before Module.__init__() call` | PyTorch initializes internal dictionary stores (`_parameters`, `_buffers`, `_modules`) in `nn.Module.__init__` |
| **Execution Hook Invariant** | Evaluate via `model(x)` instead of `model.forward(x)` | Custom hooks, third-party profilers, and autograd tape triggers silently fail | The `__call__` method wraps `forward()` with registered pre-hooks, post-hooks, and execution context |
| **Parameter-Gradient Dual Parity** | $\dim(\nabla_\theta \mathcal{L}) \equiv \dim(\theta)$ and $\operatorname{stride}(\nabla_\theta \mathcal{L}) \equiv \operatorname{stride}(\theta)$ | Shape mismatch crash during SGD optimizer step | Gradients reside in the dual vector space of parameters, requiring coordinate-wise subtraction |
| **Leaf Tensor Grad Invariant** | Leaf parameter tensors have `grad_fn = None` and `is_leaf = True` | Setting `requires_grad=True` on non-leaves causes warning or graph bloat | Leaf tensors are external graph boundaries created directly by the user, not produced by operators |
| **DAG Single-Use Lifecycle** | DAG intermediate buffers are freed after `loss.backward()` | `RuntimeError: Trying to backward through the graph a second time` | Immediate teardown avoids catastrophic GPU memory leaks during long-running multi-epoch training |
| **Zero-Copy Detach Guarantee** | $\operatorname{ptr}(x.\operatorname{detach}()) == \operatorname{ptr}(x)$ with `requires_grad=False` | In-place modifications on detached tensors silently corrupt source tensor values | Avoids copying large multi-megabyte tensors when freezing layers or switching to inference |

---

## 4. Contrastive Decision Matrix

| Operational Need | Recommended Method | Inferior / Anti-Pattern Alternative | Rationale & Failure Mode |
|:-----------------|:-------------------|:-----------------------------------|:-------------------------|
| **Model Forward Pass** | `output = model(x)` | `output = model.forward(x)` | `model(x)` triggers `__call__`, running registered forward hooks and profiler events; `.forward(x)` bypasses hooks. |
| **Module Construction** | `super().__init__()` first in `__init__` | Omitting `super()` call | Omitting `super().__init__()` crashes with `AttributeError` because internal tracking dictionaries remain uninitialized. |
| **Inference Evaluation** | `with torch.no_grad():` | Running without context | Without `torch.no_grad()`, PyTorch records backward computation graphs, wasting massive GPU VRAM during validation. |
| **Parameter Freezing** | `param.requires_grad = False` | Re-instantiating submodules | Setting `requires_grad = False` prevents gradient allocation and backprop computation without changing model structure. |
| **Loss Evaluation** | `nn.CrossEntropyLoss(logits, y)` | `nn.NLLLoss(torch.log(nn.Softmax()(logits)), y)` | `CrossEntropyLoss` combines `log_softmax` and `NLLLoss` via the LogSumExp trick, preventing catastrophic numerical underflow. |
| **Severing Autograd Tape** | `tensor.detach()` | `tensor.clone().numpy()` | `.detach()` creates an instantaneous zero-copy view sharing storage; cloning copies memory unnecessarily. |

---

## 5. Computational Complexity & Memory Footprints

- **MLP Forward FLOPs (Batch Size $B$):**
  $$
  \text{FLOPs} = 2B \sum_{l=1}^L d_{l-1} d_l = 2B (784 \times 512 + 512 \times 512 + 512 \times 10) = 2B (668,672) \approx 1.337 \times 10^6 B \text{ FLOPs}
  $$
- **Analytical Parameter Storage:**
  $$
  N_{\text{params}} = 669,706 \implies 669,706 \times 4 \text{ bytes} \approx 2.68 \text{ MB RAM / VRAM}
  $$
- **Reverse-Mode Differentiation Complexity:**
  $$
  \text{Time}_{\text{backward}} \le 3 \times \text{Time}_{\text{forward}}, \quad \text{Memory}_{\text{DAG}} = \mathcal{O}\left(B \sum_{l=1}^{L-1} d_l\right) \text{ (activations retained for backward)}
  $$

---

## 6. Numerical Stability & Traps

| Failure Mode / Trap | Root Mechanism | Mathematical / Hardware Guardrail | Production Code Pattern |
|:--------------------|:---------------|:-----------------------------------|:------------------------|
| **Log-Domain Probability Collapse** | Evaluating $\log(\operatorname{Softmax}(z))$ when $z_k \ll 0$ collapses probabilities to $0.0$, causing $\log(0) \to -\infty$ | Use mathematically unified log-domain Softmax with LogSumExp numerical stability shift | `torch.nn.functional.log_softmax(z, dim=-1)` |
| **Extreme Activation Overflow** | Exponentiating large raw logits ($z_k > 88.7$ in float32) causes $e^{z_k} \to +\infty$ | Subtract coordinate maximum before exponentiation: $z_k - \max_j(z_j) \le 0$ | Implemented automatically inside PyTorch's `Softmax` kernel |
| **Gradient Vanishing in Deep Sigmoid/Tanh** | Saturating non-linearities produce derivative $\sigma'(z) \le 0.25$, exponentially attenuating error signals across layers | Replace saturating activations with piecewise linear activations possessing unit gradient | `nn.ReLU()` or `nn.LeakyReLU(negative_slope=0.01)` |
| **Dying ReLU Inactivation** | Heavy negative bias updates push pre-activations permanently into $z < 0$, locking gradient at identically $0.0$ | Bound gradient updates via learning rate tuning or gradient norm clipping | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` |
| **Cross-Device Layer Evaluation Panic** | Passing accelerator tensors into standalone layers residing on CPU | Explicitly port standalone layer modules before calling them with data | `layer = layer.to(x.device)` |
| **Inference VRAM Accumulation Trap** | Retaining backward computation graphs during validation epoch forward passes | Wrap all evaluation and test loops in `torch.no_grad()` | `with torch.no_grad(): eval_model(x)` |
