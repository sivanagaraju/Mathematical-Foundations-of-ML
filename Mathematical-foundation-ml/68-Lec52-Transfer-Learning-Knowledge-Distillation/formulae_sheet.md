# Formulae Sheet: Lecture 52 (Transfer Learning & Knowledge Distillation)

A rapid revision ledger containing core mathematical equations, tensor shape signatures, analytical invariants, contrastive decision tables, and numerical stability guidelines for Transfer Learning, Representation Tapping, Multitask Parameter Sharing, and Knowledge Distillation.

---

## 1. Equations Index

### Composite Function Representation Tapping
For a feedforward neural network of $L$ layers:
$$G^*(X) = (g_L \circ g_{L-1} \circ \dots \circ g_1)(X)$$
Intermediate embedding tapped at layer $l$:
$$Z = G^*_l(X) \in \mathbb{R}^{D_l}$$

### Linear Probing Objective
$$\min_{\theta_{\text{head}}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}_{\text{task}}(W_{\text{head}} Z_i + b_{\text{head}}, y_i) \quad \text{subject to} \quad \nabla_{\theta_{\text{trunk}}} \mathcal{L} \triangleq 0$$

### Multitask Joint Loss and Additive Gradient Accumulation
$$\mathcal{L}_{\text{total}} = \sum_{k=1}^K \lambda_k \mathcal{L}_k(f_k(Z; \theta_k), y_k)$$
$$\nabla_Z \mathcal{L}_{\text{total}} = \sum_{k=1}^K \lambda_k \nabla_Z \mathcal{L}_k$$
$$\nabla_{\theta_{\text{trunk}}} \mathcal{L}_{\text{total}} = \left( \frac{\partial Z}{\partial \theta_{\text{trunk}}} \right)^T \nabla_Z \mathcal{L}_{\text{total}}$$

### Temperature-Scaled Softmax
For logit vector $z \in \mathbb{R}^C$ and temperature scalar $\tau > 0$:
$$p_i^\tau = \sigma_i(z / \tau) = \frac{\exp(z_i / \tau)}{\sum_{j=1}^C \exp(z_j / \tau)}$$

### Logit Probability Ratio (Dark Knowledge Metric)
$$\frac{p_i^\tau}{p_k^\tau} = \exp\left( \frac{z_i - z_k}{\tau} \right)$$

### Kullback-Leibler (KL) Divergence
$$D_{\text{KL}}(p_T^\tau \parallel p_S^\tau) = \sum_{i=1}^C p_{T, i}^\tau \ln \left( \frac{p_{T, i}^\tau}{p_{S, i}^\tau} \right) = \sum_{i=1}^C p_{T, i}^\tau \ln p_{T, i}^\tau - \sum_{i=1}^C p_{T, i}^\tau \ln p_{S, i}^\tau$$

### Hinton Combined Knowledge Distillation Loss
$$\mathcal{L}_{\text{KD}} = (1 - \alpha) \mathcal{L}_{\text{CE}}(y, \sigma(z_S)) + \alpha \tau^2 D_{\text{KL}}(p_T^\tau \parallel p_S^\tau)$$

### Asymptotic Logit Matching Limit ($\tau \to \infty$)
$$\lim_{\tau \to \infty} \tau^2 D_{\text{KL}}(p_T^\tau \parallel p_S^\tau) \approx \frac{1}{2C} \sum_{i=1}^C \left( (z_{S, i} - \bar{z}_S) - (z_{T, i} - \bar{z}_T) \right)^2$$

### Fréchet Inception Distance (FID)
$$\operatorname{FID}(P_r, P_g) = \|\mu_r - \mu_g\|_2^2 + \operatorname{Tr}\left( \Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2} \right)$$

---

## 2. Tensor Shapes & Dimension Signatures

| Tensor / Operator | Shape Signature | Physical Description |
|:------------------|:----------------|:---------------------|
| Raw Input $X$ | $[B, D_{\text{in}}]$ | Batch of input vectors or flattened images |
| Intermediate Embedding $Z = G^*_L(X)$ | $[B, D_{\text{embed}}]$ | Dense semantic feature representation |
| Trunk Parameters $\theta_{\text{trunk}}$ | Variable (Millions) | Pretrained backbone weights |
| Downstream Head Weights $W_{\text{head}}$ | $[C_{\text{target}}, D_{\text{embed}}]$ | Trainable classification layer parameters |
| Teacher Logits $z_T$ | $[B, C]$ | Raw unbounded outputs of teacher network |
| Student Logits $z_S$ | $[B, C]$ | Raw unbounded outputs of student network |
| Temperature-Scaled Probabilities $p^\tau$ | $[B, C]$ | Dilation-softened probability distribution |
| Ground-Truth Labels $y$ | $[B]$ | Class index targets $\in \{0, \dots, C-1\}$ |
| Multitask Branching Gradients $\nabla_Z \mathcal{L}_k$ | $[B, D_{\text{embed}}]$ | Per-task gradient vector arriving at branch node |

---

## 3. Analytical Invariants

- **Gradient Scale Invariance:** $\frac{\partial (\tau^2 D_{\text{KL}})}{\partial z_{S, i}} = \tau (p_{S, i}^\tau - p_{T, i}^\tau) \approx \frac{1}{C}((z_{S, i} - \bar{z}_S) - (z_{T, i} - \bar{z}_T))$ as $\tau \to \infty$. The $\tau^2$ multiplier cancels the $\frac{1}{\tau^2}$ attenuation.
- **Logit Shift Invariance:** $\sigma((z + c\mathbf{1})/\tau) = \sigma(z/\tau)$ for any constant scalar $c \in \mathbb{R}$.
- **Convexity of Linear Probing:** If loss $\mathcal{L}_{\text{task}}$ is cross-entropy or MSE and $f_{\theta_{\text{head}}}$ is linear, the optimization landscape over $\theta_{\text{head}}$ on frozen features $Z$ is strictly convex.
- **KL Non-Negativity:** $D_{\text{KL}}(p_T^\tau \parallel p_S^\tau) \ge 0$, with equality if and only if $p_T^\tau = p_S^\tau$ almost everywhere.

---

## Contrastive Decision Table

| Scenario / Constraint | Recommended Paradigm | Rationale & Trade-offs |
|:----------------------|:---------------------|:-----------------------|
| Target dataset $N < 1,000$, similar domain | **Linear Probing (Frozen Trunk)** | Prevents catastrophic overfitting; trains fast with zero risk of destroying pretrained weights. |
| Target dataset $N > 50,000$, moderate domain shift | **Full Fine-Tuning with Warmup** | Adapts all layers to target distribution; differential learning rates prevent destructive early updates. |
| Multiple related objectives on single input | **Hard Parameter Sharing Trunk** | Maximizes computational efficiency via single forward pass; provides cross-task inductive regularization. |
| Large SOTA model exceeds edge latency/memory budgets | **Knowledge Distillation ($\tau \in [3, 8]$)** | Compresses dark knowledge into compact student while allowing custom edge-tailored student architecture. |
| Unlabeled data available at scale ($>10^7$ samples) | **Self-Supervised Marginal $P(X)$ Pre-training** | Estimates marginal data manifold (masked modeling/next-token), producing universal transferable backbones. |

---

## Numerical Stability & Traps

1. **Log-Softmax Formulation in PyTorch:**  
   Never compute KL divergence by passing raw softmax into `torch.log()`:
   ```python
   # INCORRECT (Numerically Unstable):
   p_S = torch.softmax(z_S / tau, dim=-1)
   loss = torch.sum(p_T * torch.log(p_S))  # Underflows to -inf when p_S -> 0!
   
   # CORRECT (Numerically Stable):
   log_p_S = F.log_softmax(z_S / tau, dim=-1)
   loss = F.kl_div(log_p_S, p_T, reduction='batchmean') * (tau ** 2)
   ```
2. **Batchmean Reduction in `F.kl_div`:**  
   PyTorch's default `reduction='mean'` divides by the total number of elements ($B \times C$), underestimating the true KL divergence by a factor of $C$. Always use `reduction='batchmean'` to divide strictly by batch size $B$.
3. **The Logit Explosion Trap Under High Temperature:**  
   When $\tau$ is small ($\tau < 0.5$), logits divide by small fractions, causing exponent overflow ($\exp(z_i / \tau) \to \infty$). Ensure $\tau \ge 1.0$ in distillation pipelines.
