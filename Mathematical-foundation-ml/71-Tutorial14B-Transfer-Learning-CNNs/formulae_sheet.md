# Formulae Sheet — Tutorial 14 Part 2: Transfer Learning using CNNs

This reference sheet consolidates all mathematical formulations, gradient isolation mechanics, tensor dimensions, and architectural properties for Transfer Learning with Convolutional Neural Networks.

---

## 1. Transfer Learning Empirical Risk Formulation

Given a target dataset $\mathcal{D}_T = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N_T}$ with $C$ classes and a pretrained backbone $\mathcal{F}_{\Theta}(\mathbf{x}) \in \mathbb{R}^D$:

### Linear Probe (Frozen Trunk)
$$\min_{\Phi} \frac{1}{N_T}\sum_{i=1}^{N_T} \mathcal{L}_{\text{CE}}\left( \mathcal{G}_{\Phi}(\mathcal{F}_{\Theta_{\text{frozen}}}(\mathbf{x}_i)), \; y_i \right) + \frac{\lambda}{2}\|\Phi\|_2^2$$
where $\Theta_{\text{frozen}}$ is immutable ($\nabla_{\Theta} \mathcal{L} \equiv \mathbf{0}$) and $\Phi = \{\mathbf{W} \in \mathbb{R}^{C \times D}, \mathbf{b} \in \mathbb{R}^C\}$ is the newly initialized classification head.

### Differential Fine-Tuning
$$\Theta^{(t+1)} = \Theta^{(t)} - \eta_{\text{trunk}} \nabla_{\Theta} \mathcal{L}, \quad \Phi^{(t+1)} = \Phi^{(t)} - \eta_{\text{head}} \nabla_{\Phi} \mathcal{L}$$
where $\eta_{\text{trunk}} \ll \eta_{\text{head}}$ (typically $\eta_{\text{trunk}} = 10^{-5}, \eta_{\text{head}} = 10^{-3}$).

---

## 2. Categorical Softmax & Top-K Decision Boundaries

For an unnormalized logit vector $\mathbf{z} \in \mathbb{R}^K$:

### Softmax Posterior Distribution
$$p_i = \sigma(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)}, \quad \sum_{i=1}^K p_i = 1.0, \quad p_i > 0$$

### Top-K Selection Operator
$$\mathcal{S}_K(\mathbf{x}) = \left\{ i \in \{1, \dots, K\} \;\middle|\; \text{rank}(p_i) \le K \right\}$$
$$\mathbb{I}_{\text{Top-}K}(\mathbf{x}, y_{\text{true}}) = \begin{cases} 1 & \text{if } y_{\text{true}} \in \mathcal{S}_K(\mathbf{x}) \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. Residual Identity Gradient Flow (He et al., 2016)

For a residual block mapping input $\mathbf{x}$ to output $\mathbf{y}$:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \mathcal{W}) + \mathbf{x}$$

### Backward Gradient Propagation
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I} \right) = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$$

*Key Implication:* The gradient signal contains an unattenuated identity term $\frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \mathbf{I}$ that propagates directly across arbitrary depth without passing through weight multiplications.

---

## 4. Global Average Pooling (GAP) Reduction

For a 3D feature tensor volume $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$:

$$Z_c = \text{GAP}(\mathbf{X})_c = \frac{1}{H \cdot W} \sum_{i=1}^H \sum_{j=1}^W X_{c, i, j}, \quad \forall c \in \{1, \dots, C\}$$

*Properties:*
- Output shape: $[B, C, 1, 1] \xrightarrow{\text{flatten}} [B, C]$.
- Parameter count: Exactly $0$ learnable weights or biases.
- Spatial dimension tolerance: Valid for arbitrary input heights and widths $(H, W)$.

---

## 5. Architectural Comparison: ResNet-18 vs VGG-19

| Property | VGG-19 | ResNet-18 |
|:---|:---|:---|
| **Total Weight Layers** | 19 (16 conv + 3 dense) | 18 (17 conv + 1 dense) |
| **Total Parameters** | $143,667,240$ (~548 MB) | $11,689,512$ (~45 MB) |
| **Classifier Head Parameters** | $119,554,000$ (83.2% of total!) | $513,000$ (4.4% of total) |
| **Feature Pooling Mechanism** | MaxPool ($7 \times 7$) $\to$ Flatten | Global Average Pooling ($1 \times 1$) |
| **Embedding Dimension ($Z$)** | $25,088$ dimensions | $512$ dimensions |
| **Vanishing Gradient Safeguard** | None (sequential stacking limit) | Identity skip shortcuts ($x + F(x)$) |
| **Head Surgery Target** | `model.classifier[6]` | `model.fc` |

---

## 6. Contrastive Decision Guide

| Scenario | Recommended Strategy | Rationale |
|:---|:---|:---|
| **Small Dataset ($N < 2,000$), High Domain Similarity** | Freeze trunk, train linear probe | Prevents overfitting; generic ImageNet features suffice. |
| **Large Dataset ($N > 50,000$), Novel Domain** | Full fine-tuning ($\eta_{\text{trunk}} = 10^{-5}$) | Ample data allows adapting intermediate feature detectors. |
| **Multimodal Downstream Task (Image Captioning)**| Decouple embedding $Z \in \mathbb{R}^{512}$ | Pass compact visual vector directly into LSTM hidden state. |
| **Variable Input Resolution ($300 \times 300$)** | ResNet backbone (with GAP) | GAP produces fixed $512$-d vector regardless of input resolution. |

---

## 7. Guarantees & Invariants

1. **Parameter Immutability Under Frozen Autograd:**
   $$\forall \theta \in \Theta_{\text{frozen}}, \quad \theta.\text{requires\_grad} = \text{False} \implies \theta^{(t+1)} \equiv \theta^{(t)}$$
   Weights with `requires_grad=False` remain numerically identical across all training iterations.
2. **Convexity of Linear Probe Objective:**
   With frozen representations $Z$, the Cross-Entropy loss is strictly convex with respect to linear head parameters $\Phi = \{\mathbf{W}, \mathbf{b}\}$, guaranteeing a unique global minimum.
3. **Identity Gradient Highway:**
   In residual blocks, the subgradient of the shortcut branch is strictly $\mathbf{I}$, ensuring that gradients cannot diminish exponentially across layers.

---

## 8. Numerical Stability & Traps

1. **Evaluation Mode Enforcement:**
   Always invoke `model.eval()` before test inference. In `model.train()`, Dropout randomly zeroes 50% of activations and BatchNorm calculates batch-specific statistics, producing stochastic non-deterministic predictions.
2. **Gradient Isolation in Optimizers:**
   Avoid passing `model.parameters()` to optimizers when the trunk is frozen. Use:
   ```python
   optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
   ```
   This prevents allocating unused state buffers (first and second moments) for millions of frozen parameters.
3. **Numerical Epsilon in Softmax & Normalization:**
   When computing manual cross-entropy or division by norms, always clamp with $\epsilon = 10^{-7}$:
   $$\log(p_i + \epsilon)$$
   to prevent $\log(0) = -\infty$ and NaN propagation.
