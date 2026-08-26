# Autoregressive Models: Exact Likelihood Factorization, Causal Masking, and Sequential Sampling

An **Autoregressive Model** is a generative framework that decomposes the joint probability distribution of high-dimensional data $p(x_1, \dots, x_T)$ into a product of conditional probabilities using the exact probability chain rule, predicting each token or pixel given only its preceding context ($x_t \mid x_{<t}$).

```
 ===================================================================================================
                 THE PROBABILITY CHAIN RULE & CAUSAL GENERATIVE FACTORIZATION
 ===================================================================================================
 
  JOINT DISTRIBUTION P(x₁, ..., x_T)              EXACT FACTORIZATION (CHAIN RULE)
  Full high-dimensional probability              Product of 1D conditional distributions
  ┌──────────────────────────────┐              ┌──────────────────────────────┐
  │ p(x₁, x₂, x₃, ..., x_T)      │ ═══════════► │ p(x₁) · p(x₂|x₁) · p(x₃|x₁,x₂) ...
  │ Infeasible joint table       │              │ Exactly tractable likelihood │
  │ Cannot integrate directly    │              │ No approximations needed     │
  └──────────────────────────────┘              └──────────────────────────────┘
                                                               │
                                                               ▼
  PARALLEL TRAINING (Teacher Forcing)           SEQUENTIAL INFERENCE / GENERATION
  ┌──────────────────────────────┐              ┌──────────────────────────────┐
  │ Fast O(1) step across tokens │              │ Step t=1: sample x₁ ~ p(x₁)  │
  │ Masked Attention / Conv      │              │ Step t=2: sample x₂ ~ p(x₂|x₁)
  │ Loss = -Σ log p(x_t | x_<t)  │              │ Step t=3: sample x₃ ~ p(x₃|x₁₂)
  └──────────────────────────────┘              └──────────────────────────────┘
 ===================================================================================================
```

---

### 1. 👶 ELI5 Intuition: The Storyteller & The Domino Chain

1. **The Storyteller (Probability Chain Rule):** Imagine writing a suspense story word by word.
   - Word 1: "The" ($p(x_1)$)
   - Word 2: Given "The", what comes next? "dragon" ($p(x_2 \mid x_1)$)
   - Word 3: Given "The dragon", what comes next? "breathed" ($p(x_3 \mid x_1, x_2)$)
   - Word 4: Given "The dragon breathed", what comes next? "fire!" ($p(x_4 \mid x_1, x_2, x_3)$)
2. **The Causal Blindfold (Masking):** When teaching the storyteller, we show them a sentence but blindfold them from looking ahead into future words. They can only see the past!
3. **The Generation Bottleneck:** Because word 100 depends on word 99, you *cannot* generate word 100 in advance. You must roll the dominoes one step at a time!

> 💡 **The Great AI Takeaway:** Autoregressive models (GPT-4, Claude, LLaMA, PixelCNN) provide **exact likelihood evaluation** and training parallelization, but suffer from **sequential $O(T)$ generation latency**.

---

### 2. 🔍 Plain-English Breakdown & Autoregressive Rosetta Stone

| Symbol / Term | Formal Mathematical Concept | Plain-English Software Role | Deep Learning Analogue |
| :--- | :--- | :--- | :--- |
| **$x = (x_1, \dots, x_T)$** | Sequence of discrete tokens/pixels | Input sequence or token IDs | `input_ids = [101, 2054, ...]` |
| **$x_{<t}$** | Historical prefix $(x_1, \dots, x_{t-1})$ | All preceding context tokens | Context window / Prompt buffer |
| **$p_\theta(x_t \mid x_{<t})$**| Conditional likelihood distribution | Softmax probability over vocabulary | `probs = F.softmax(logits[:, -1, :])` |
| **$M \in \{0, -\infty\}^{T \times T}$** | Causal Attention Mask | Upper-triangular mask blocking future tokens | `mask = torch.triu(..., diagonal=1)` |
| **$\mathcal{L}_{\text{NLL}}$** | Negative Log-Likelihood | Training loss penalizing wrong token predictions | `nn.CrossEntropyLoss()` |
| **$\tau > 0$** | Sampling Temperature | Temperature scaling factor flattening/sharpening logits | `logits / temperature` |
| **Top-$p$ (Nucleus)** | Cumulative probability threshold | Keeping smallest set of tokens whose sum $\ge p$ | Filtering low-probability tail tokens |

---

### 3. 📐 Formal Mathematical Formulations & Guarantees

#### A. The Exact Probability Chain Rule
For any joint probability distribution over $T$ random variables:
$$p(x_1, x_2, \dots, x_T) = p(x_1) \cdot p(x_2 \mid x_1) \cdot p(x_3 \mid x_1, x_2) \cdots p(x_T \mid x_1, \dots, x_{T-1}) = \prod_{t=1}^T p(x_t \mid x_{<t})$$

- **Exact Likelihood Property:** Unlike VAEs (which optimize a lower bound ELBO) or GANs (which have intractable density), autoregressive models compute the **exact log-likelihood**:
  $$\ln p(x) = \sum_{t=1}^T \ln p(x_t \mid x_{<t})$$

#### B. Causal Masking in Attention (Vaswani et al., 2017)
To evaluate all conditionals $p(x_t \mid x_{<t})$ for $t=1, \dots, T$ simultaneously in parallel:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{QK^\top}{\sqrt{d_k}} + M \right) V$$
where the causal mask matrix $M \in \mathbb{R}^{T \times T}$ is defined as:
$$M_{ij} = \begin{cases} 0 & \text{if } j \le i \quad (\text{Past and Present Tokens Allowed}) \\ -\infty & \text{if } j > i \quad (\text{Future Tokens Strictly Forbidden}) \end{cases}$$

#### C. Stochastic Decoding & Sampling Strategies
Given predicted logit vector $z \in \mathbb{R}^V$ at step $t$:
1. **Temperature Scaling:** $p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$  
   - As $\tau \to 0$: distribution collapses to greedy deterministic $\arg\max$.
   - As $\tau \to \infty$: distribution becomes uniform random noise.
2. **Top-$p$ (Nucleus) Truncation:** Retains the minimal subset $\mathcal{V}^{(p)} \subset \mathcal{V}$ such that:
   $$\sum_{i \in \mathcal{V}^{(p)}} p_i \ge p \quad (\text{typically } p=0.90)$$

---

### 4. 🔢 Concrete Micro-Numerical Calculation

Let vocabulary $\mathcal{V} = \{\text{cat}, \text{dog}, \text{fish}\}$ with raw logits $z = [2.0, \quad 1.0, \quad 0.0]$:

1. **Standard Softmax ($\tau = 1.0$):**
   - $e^2 \approx 7.389, \quad e^1 \approx 2.718, \quad e^0 = 1.000 \implies \sum = 11.107$
   - $p = \left[ \frac{7.389}{11.107}, \frac{2.718}{11.107}, \frac{1.000}{11.107} \right] = \mathbf{[0.665, \quad 0.245, \quad 0.090]}$
2. **High Temperature ($\tau = 2.0$ — More Creative/Diverse):**
   - $z / 2 = [1.0, \quad 0.5, \quad 0.0]$
   - $e^{1.0} \approx 2.718, \quad e^{0.5} \approx 1.649, \quad e^0 = 1.000 \implies \sum = 5.367$
   - $p_{\tau=2} = [0.506, \quad 0.307, \quad 0.186]$ (Distribution flattened!)
3. **Negative Log-Likelihood for Ground Truth Token $x_t = \text{"cat"}$ ($i=0$):**
   $$\mathcal{L}_{\text{NLL}} = -\ln(0.665) = \mathbf{0.408 \text{ nats}}$$

---

### 5. 🔗 Connecting the Dots: How Autoregression Powers Modern Generative AI

1. **Large Language Models (GPT-4, Claude 3.5, LLaMA-3, Gemini):**
   - Autoregressive causal transformers trained on trillions of internet tokens using cross-entropy next-token prediction.
2. **Autoregressive Vision Models (PixelCNN, Parti, ImageGPT):**
   - Treats images as 1D raster-scan pixel/codebook sequences ($x_1, \dots, x_{H \times W}$).
3. **Audio & Music Generation (WaveNet, MusicLM, AudioCraft):**
   - Predicts raw 16kHz/24kHz acoustic audio waveform slices conditioned on preceding audio samples.

---

### 6. 💻 Complete Standalone Executable Python/PyTorch Verification Script

```python
"""
AUTOREGRESSIVE GENERATIVE MODEL VERIFICATION SUITE
==================================================
Demonstrates exact probability factorization, causal mask generation,
and temperature-controlled autoregressive token generation in PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def run_autoregressive_verification():
    print("=" * 80)
    print("  AUTOREGRESSIVE MODELS: MATHEMATICAL & CAUSAL SAMPLING SUITE")
    print("=" * 80)

    # 1. CAUSAL ATTENTION MASK GENERATION
    print("\n[1] Generating Causal Triangular Attention Mask (Seq Len = 4)")
    seq_len = 4
    # Create upper triangular mask with -inf above diagonal
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    print(f"  * Causal Mask Matrix M (0 = Allowed, -inf = Blocked):\n{mask}")
    assert mask[0, 1] == float('-inf'), "Future position (0, 1) must be masked with -inf!"
    assert mask[1, 0] == 0.0, "Past position (1, 0) must be allowed (0.0)!"

    # 2. TOY AUTOREGRESSIVE CAUSAL MODEL
    print("\n[2] Building Modular Causal Sequence Model (Vocab = 8, Hidden = 16)")
    class ToyCausalModel(nn.Module):
        def __init__(self, vocab_size=8, hidden_dim=16):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            # Input shape: (Batch, Seq_Len)
            embeds = self.embedding(x)
            hidden, _ = self.rnn(embeds)
            logits = self.head(hidden) # (Batch, Seq_Len, Vocab)
            return logits

    model = ToyCausalModel(vocab_size=8, hidden_dim=16)
    torch.manual_seed(42)

    # 3. PARALLEL TRAINING VIA TEACHER FORCING
    print("\n[3] Parallel Next-Token Training via Teacher Forcing")
    # Training sequence: [1, 3, 5, 7]
    target_seq = torch.tensor([[1, 3, 5, 7]], dtype=torch.long)
    input_seq = target_seq[:, :-1]  # Prefix: [1, 3, 5]
    ground_truth = target_seq[:, 1:] # Next tokens: [3, 5, 7]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(60):
        optimizer.zero_grad()
        logits = model(input_seq)
        loss = criterion(logits.reshape(-1, 8), ground_truth.reshape(-1))
        loss.backward()
        optimizer.step()

    print(f"  * Final Next-Token Loss: {loss.item():.4f} nats")
    assert loss.item() < 0.1, "Model failed to learn sequential autoregressive pattern!"

    # 4. STEP-BY-STEP SEQUENTIAL AUTOREGRESSIVE GENERATION
    print("\n[4] Sequential Autoregressive Inference (Generating 3 Tokens)")
    generated_tokens = [1] # Start token prompt
    temperature = 0.7

    model.eval()
    with torch.no_grad():
        for step in range(3):
            current_input = torch.tensor([generated_tokens], dtype=torch.long)
            logits = model(current_input)
            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs).item() # Greedy argmax for deterministic verification
            generated_tokens.append(next_token)
            print(f"  * Step {step+1}: Prompt {generated_tokens[:-1]} -> Sampled Next Token: {next_token}")

    print(f"  * Full Autoregressive Sequence: {generated_tokens}")
    assert generated_tokens == [1, 3, 5, 7], "Autoregressive generation mismatch!"

    print("\n" + "=" * 80)
    print("  [PASS] ALL AUTOREGRESSIVE VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_autoregressive_verification()
```

---

### 7. 🩺 Diagnostic Mini-Checks & Common Traps

#### Diagnostic Self-Test
1. **Q:** Why can autoregressive models be trained in parallel across all time steps, but generation must be sequential?  
   *Answer:* During training, ground truth past tokens are fully known ("Teacher Forcing"), allowing parallel computation of all conditionals $p(x_t \mid x_{<t})$. During inference, step $t$ requires the generated output of step $t-1$, forcing sequential generation.
2. **Q:** How does setting temperature $\tau \to 0$ affect sampling?  
   *Answer:* It magnifies differences between logits, making the highest logit approach probability $1.0$, collapsing sampling into deterministic $\arg\max$ decoding.
3. **Q:** What is the primary theoretical advantage of Autoregressive models over GANs?  
   *Answer:* Autoregressive models compute exact, tractable likelihoods without minimax instability or mode collapse.

#### Common Engineering Traps
- ❌ **Trap 1: Information leakage across causal boundaries (Off-by-one indexing in Teacher Forcing).**  
  *Fix:* Input tokens must be `seq[:, :-1]` and targets must be `seq[:, 1:]`. If targets match inputs directly, the model learns the trivial identity shortcut.
- ❌ **Trap 2: Sampling from unnormalized raw logits without Softmax.**  
  *Fix:* Always pass scaled logits $z / \tau$ through `F.softmax(..., dim=-1)` before calling `torch.multinomial`.
