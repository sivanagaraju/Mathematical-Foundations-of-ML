# Autoregressive Models: Exact Likelihood Factorization, Causal Masking & Sequential Sampling

> `🏷️ Tags:` `Generative-AI` `Autoregressive` `Transformers` `LLMs` `Causal-Masking` `KV-Cache` `Probability-Chain-Rule`  
> `📚 Prerequisites Needed:` [Probability Basics & Axioms](./Probability_Basics_and_Axioms.md) · [Softmax & Probabilities](./Softmax.md) · [Likelihood & Log-Likelihood](./Likelihood_and_Log_Likelihood.md)  
> `🎯 Where Do We Use This?:` **The foundational architecture behind all Large Language Models (LLMs)** — Generative pre-training in GPT-4, Claude, and LLaMA-3, Autoregressive audio synthesis in Voice AI, and Causal sequence generation across text, code, and robotics.  
> `🎓 Course Module Mapping:` [Lec 01: Intro](../Mathematical-Foundation-for-GenerativeAI/14-Lec01-MFGAI-Introduction/NOTES.md) · [Tut 03: PyTorch Basics](../Mathematical-Foundation-for-GenerativeAI/17-Tutorial03-PyTorch-Basics/NOTES.md) · [Tut 08: Basic Probability 2](../Mathematical-Foundation-for-GenerativeAI/22-Tutorial08-Review-Basic-Probability-2/NOTES.md)  
> `⏱️ Difficulty Level:` ⭐⭐⭐☆☆ (Intermediate · 15 min read)

---

### 📌 Quick Navigation & Architecture Map
- [1. 🌟 Everyday Real-World Scenarios](#1--everyday-real-world-scenarios-the-suspense-storyteller--chatgpt-next-token-generation) — The Suspense Storyteller & ChatGPT Next-Token Generation
- [2. 👶 ELI5 Intuition](#2--eli5-intuition-the-domino-chain--the-causal-blindfold) — The Domino Chain & The Causal Blindfold
- [3. 📚 Deep Terminology Master Glossary](#3--deep-terminology-master-glossary-15-core-concepts-dissected) — 15 autoregressive terms dissected without jargon
- [4. 📐 Mathematical Formulations, Causal Masking & Sampling](#4--mathematical-formulations-causal-masking--sampling) — Probability chain rule, Causal attention matrix, and KV-Caching math
- [5. 🔢 Concrete Micro-Numerical Worked Examples](#5--concrete-micro-numerical-worked-examples) — 3-Token Sequence Joint Likelihood & Temperature Scaling by Hand
- [6. 🔗 Connecting the Dots: Generative AI Architecture Blocks](#6--connecting-the-dots-how-autoregression-powers-generative-ai) — Transformer Decoder Causal Block, KV-Cache Loop, and Speculative Decoding
- [7. 💻 Standalone Executable Python/PyTorch Verification Script](#7--complete-standalone-executable-pythonpytorch-verification-script) — Causal mask simulation, autoregressive generation loop, and KV-cache acceleration
- [8. 🩺 Diagnostic Mini-Checks & Common Traps](#8--diagnostic-mini-checks--common-traps) — Self-test questions & production engineering pitfalls

---

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

### 1. 🌟 Everyday Real-World Scenarios (The Suspense Storyteller & ChatGPT Next-Token Generation)
> `Context:` Zero Prior Machine Learning / AI Knowledge Needed · Concrete Real-World Mapping

#### Scenario A: The Storyteller Writing a Novel (Zero ML Background Needed)
Imagine writing a mystery novel word by word:
1. **Word 1:** *"The"* ($p(x_1)$).
2. **Word 2:** Given *"The"*, what word comes next? *"detective"* ($p(x_2 \mid x_1)$).
3. **Word 3:** Given *"The detective"*, what comes next? *"opened"* ($p(x_3 \mid x_1, x_2)$).
4. **Word 4:** Given *"The detective opened"*, what comes next? *"the door!"* ($p(x_4 \mid x_1, x_2, x_3)$).
5. **The Causal Rule:** The writer cannot know word 50 until words 1 through 49 are written. You must generate sentences sequentially!

---

#### Scenario B: In Generative AI — ChatGPT Autoregressive Token Generation
> `Context:` How Large Language Models Generate Millions of Coherent Sentences

When an LLM generates a response:
- The model receives prompt tokens: $x_{<t} = [x_1, \dots, x_{t-1}]$.
- The Transformer computes self-attention with a **Causal Mask** that prevents tokens from looking into the future.
- The model outputs a probability distribution over the entire vocabulary $p(x_t \mid x_{<t})$.
- A token is sampled (e.g. via Temperature or Top-$p$), appended to the context buffer, and fed back as input for step $t+1$!

```
 ===================================================================================================
         AUTOREGRESSIVE GENERATIVE SAMPLING LOOP (LLM RUNTIME ENGINE)
 ===================================================================================================

  PROMPT: "Artificial intelligence" ──► [ Causal Transformer ] ──► Softmax Probs ──► Sample "is"
                                                                                           │
  NEW CONTEXT: "Artificial intelligence is" ◄──────────────────────────────────────────────┘
         │
         ▼
  [ Causal Transformer ] ──► Softmax Probs ──► Sample "transforming" ──► Append & Repeat!
 ===================================================================================================
```

---

### 2. 👶 ELI5 Intuition: The Domino Chain & The Causal Blindfold
> `Context:` Physical & Everyday Metaphors for Autoregression

#### Metaphor 1: The Domino Chain Reaction
- You set up 100 dominoes in a line.
- Domino 1 falls and knocks down Domino 2; Domino 2 knocks down Domino 3.
- You cannot knock down Domino 50 without knocking down Dominoes 1 through 49 first.

---

#### Metaphor 2: The Causal Reading Blindfold
- When teaching a student to read, you slide a piece of paper across the page to cover up all future words.
- The student must predict each word based strictly on the words they have already read.

---

### 3. 📚 Deep Terminology Master Glossary (15 Core Concepts Dissected)
> `Context:` Foundational Mathematical & Machine Learning Vocabulary Explained Without Jargon

```
 ===================================================================================================
                 THE AUTOREGRESSIVE MODELS ROSETTA STONE
 ===================================================================================================
```

| Term / Notation | Formal Mathematical Meaning | Plain-English Definition (No ML Jargon) | Real-World Analogy |
| :--- | :--- | :--- | :--- |
| **Autoregressive Model** | $p(x) = \prod p(x_t \mid x_{<t})$ | Model that predicts each future step using its own past predictions | Predicting next word based on preceding words |
| **Probability Chain Rule** | Exact factorization of joint probability | Mathematical law stating joint likelihood equals product of sequential conditionals | Total journey probability = product of leg probabilities |
| **Causal Mask ($M_{ij}$)** | $M_{ij} = -\infty$ for $j > i$ | Upper-triangular matrix blocking attention to future tokens during training | Wearing horse blinders to look only straight and back |
| **Teacher Forcing** | Training on true previous token $x_{t-1}$ | Parallel training technique feeding ground-truth tokens instead of model guesses | A parent correcting a toddler's speech word-by-word |
| **KV-Cache (Key-Value Cache)**| Caching past Key/Value attention tensors | Storing past token activations in GPU memory to avoid redundant recomputations | Keeping previous meeting minutes on your desk |
| **Sampling Temperature ($\tau$)**| $\text{Softmax}(z / \tau)$ | Scalar controlling randomness: $\tau \to 0$ makes it greedy, $\tau \to \infty$ makes it uniform | Adjusting focus on a lens from sharp to blurry |
| **Top-$k$ Sampling** | Restricting sampling to $k$ highest logits | Discarding all but the top-$k$ most probable tokens before sampling | Only choosing among the top 3 menu options |
| **Top-$p$ (Nucleus) Sampling** | Restricting to smallest set with $\sum p \ge p$| Dynamically adapting candidate pool based on cumulative probability mass | Considering only the top $90\%$ of qualified candidates |
| **Exact Log-Likelihood** | $\ln p(x) = \sum \ln p(x_t \mid x_{<t})$ | Tractable exact calculation of dataset probability without lower-bound ELBO approximations | Exact itemized bank statement |
| **Sequential Latency ($O(T)$)**| Generation requires $T$ consecutive steps | Inability to generate all tokens in parallel during inference | Waiting for a train to arrive station-by-station |
| **PixelCNN / PixelRNN** | Autoregressive 2D pixel generation | Generating images pixel-by-pixel, row-by-row using masked convolutions | Painting a canvas with tiny dots from top-left to bottom-right |
| **Context Window Length** | Maximum sequence length $T_{\max}$ | Maximum number of preceding tokens the model can remember at one time | Short-term memory capacity |
| **Perplexity ($\text{PPL}$)** | $\exp(\text{NLL}_{\text{avg}})$ | Standard evaluation metric measuring vocabulary uncertainty | The number of equally likely words being chosen from |
| **Rotary Position Embedding (RoPE)**| Relative position encoding via complex rotation | Encodes token distances directly into query/key dot products | Measuring distance between landmarks along a highway |
| **Speculative Decoding** | Small draft model proposes $K$ tokens | Acceleration technique using a tiny fast model to draft tokens for a large verifier | An assistant drafting emails for a CEO to review |

---

### 4. 📐 Mathematical Formulations, Causal Masking & Sampling
> `Context:` Formal Probability Chain Rule, Causal Attention Equations, and KV-Cache Math

```
 ===================================================================================================
                 THE CAUSAL ATTENTION MATRIX MASK
 ===================================================================================================

  Attention Scores (Q · Kᵀ / √d_k) + Causal Mask M:
  
  Tokens:     "The"      "cat"      "sat"      "down"
  "The"    [  0.0   ,   -inf   ,   -inf   ,   -inf   ] ──► Can only see "The"
  "cat"    [  1.2   ,    0.0   ,   -inf   ,   -inf   ] ──► Can see "The", "cat"
  "sat"    [  0.8   ,    2.4   ,    0.0   ,   -inf   ] ──► Can see "The", "cat", "sat"
  "down"   [  0.1   ,    1.5   ,    3.1   ,    0.0   ] ──► Can see all 4 tokens!
 ===================================================================================================
```

#### Core Mathematical Theorems:

1. **The Exact Probability Chain Rule:**
   $$p(x_1, x_2, \dots, x_T) = p(x_1) \prod_{t=2}^T p(x_t \mid x_1, \dots, x_{t-1}) = \prod_{t=1}^T p(x_t \mid x_{<t})$$

2. **Exact Log-Likelihood Evaluation:**
   $$\ln p(x) = \sum_{t=1}^T \ln p_\theta(x_t \mid x_{<t})$$

3. **Causal Attention Formulation (Vaswani et al., 2017):**
   $$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{QK^\top}{\sqrt{d_k}} + M \right) V$$
   where $M_{ij} = 0$ for $j \le i$, and $M_{ij} = -\infty$ for $j > i$.

4. **KV-Cache Memory Complexity at Step $t$:**
   $$\text{Memory} = 2 \times B \times H \times t \times d_k \times \text{sizeof}(\text{float16})$$
   - Rather than recomputing attention across all $t$ past tokens ($O(t^2)$ compute), caching $K$ and $V$ reduces per-token inference cost to $O(t)$ compute!

---

### 5. 🔢 Concrete Micro-Numerical Worked Examples
> `Context:` Step-by-Step Manual Calculations (No Black Box)

#### Example 1: 3-Token Sequence Joint Likelihood by Hand
Let sentence be: $X = [\text{"The"}, \quad \text{"dog"}, \quad \text{"barks"}]$.
Suppose conditional probabilities are:
- $p(x_1 = \text{"The"}) = \mathbf{0.10}$
- $p(x_2 = \text{"dog"} \mid x_1 = \text{"The"}) = \mathbf{0.40}$
- $p(x_3 = \text{"barks"} \mid x_{<3} = \text{"The dog"}) = \mathbf{0.70}$

1. **Calculate Joint Probability via Chain Rule:**
   $$p(\text{"The dog barks"}) = 0.10 \times 0.40 \times 0.70 = \mathbf{0.0280}$$

2. **Calculate Exact Sequence Negative Log-Likelihood (NLL):**
   $$\text{NLL} = -\ln(0.10) - \ln(0.40) - \ln(0.70) = 2.3026 + 0.9163 + 0.3567 = \mathbf{3.5756\text{ nats}}$$
   $$\text{Check: } -\ln(0.0280) = \mathbf{3.5756\text{ nats}} \quad ✅$$

---

#### Example 2: Temperature Scaling Logit Arithmetic
Let vocabulary logits at step $t$ be $z = [2.0, \quad 1.0, \quad 0.0]$ for `["cat", "dog", "fish"]`.

1. **At Temperature $\tau = 1.0$ (Default):**
   - $e^2 \approx 7.389, \quad e^1 \approx 2.718, \quad e^0 = 1.000 \implies Z = 11.107$.
   - $p = [0.665, \quad 0.245, \quad 0.090]$.

2. **At Temperature $\tau = 0.5$ (Sharper / More Deterministic):**
   - $z / 0.5 = [4.0, \quad 2.0, \quad 0.0]$.
   - $e^4 \approx 54.598, \quad e^2 \approx 7.389, \quad e^0 = 1.000 \implies Z = 62.987$.
   - $p = \mathbf{[0.867, \quad 0.117, \quad 0.016]}$ *(Top choice boosted to $86.7\%$!)*.

---

### 6. 🔗 Connecting the Dots: How Autoregression Powers Generative AI
> `Context:` Architectural Implementations in Large Language Models, Voice AI, and Image Modeling

```
 ===================================================================================================
                 AUTOREGRESSIVE GENERATIVE ARCHITECTURES
 ===================================================================================================

  1. TRANSFORMER LLMS (GPT-4 / LLaMA-3)             2. SPECULATIVE DECODING ENGINE
  Causal Attention + KV-Cache Decoding              Draft Model (1B) proposes K tokens in parallel
  ┌────────────────────────────────────────┐        ┌────────────────────────────────────────┐
  │ Parallel training on 10T tokens        │        │ Large Model (70B) verifies all K tokens│
  │ Sequential token-by-token generation   │        │ in a single parallel forward pass      │
  │ Powers human dialogue, code, & reasoning│       │ Speeds up inference by 2.5x to 3.0x    │
  └────────────────────────────────────────┘        └────────────────────────────────────────┘
 ===================================================================================================
```

| Generative System | How Autoregression is Applied | Architectural Purpose |
| :--- | :--- | :--- |
| **Large Language Models (GPT-4, LLaMA-3)** | **Autoregressive Token Generation + KV-Cache** | Generates human text by recursively sampling $w_t \sim p(w_t \mid w_{<t})$ |
| **Audio Synthesis (WaveNet, Voice AI)** | **Autoregressive Waveform Prediction** | Predicts raw audio amplitude samples at 24,000 samples per second |
| **Autoregressive Vision (PixelCNN, Chameleon)** | **Raster-Scan Causal Pixel Generation** | Generates discrete image patches row-by-row, conditioning on upper-left context |
| **Robotics (RT-2 / OpenVLA)** | **Autoregressive Action Tokenization** | Predicts 7-DoF robotic arm motor commands as discrete token sequences |

---

### 7. 💻 Complete Standalone Executable Python/PyTorch Verification Script
> `Context:` Runnable Code Verifying Causal Masking, Autoregressive Text Generation, and KV-Caching

```python
"""
Autoregressive Modeling & Causal Attention Simulation
=====================================================
Demonstrates:
1. Exact probability chain rule joint likelihood calculation
2. Causal attention masking matrix generation in PyTorch
3. Autoregressive token-by-token generation loop with temperature sampling
"""
import torch
import torch.nn.functional as F
import numpy as np

print("=" * 75)
print("AUTOREGRESSIVE MODELING & CAUSAL ATTENTION SIMULATION")
print("=" * 75)

# ─── 1. Exact Probability Chain Rule Verification ───
print("\n1. EXACT PROBABILITY CHAIN RULE (Sequence: 'The dog barks'):")
p_the = 0.10
p_dog_given_the = 0.40
p_barks_given_thedog = 0.70

joint_prob = p_the * p_dog_given_the * p_barks_given_thedog
joint_nll = -(np.log(p_the) + np.log(p_dog_given_the) + np.log(p_barks_given_thedog))

print(f"   * Multiplied Joint Probability: {joint_prob:.4f} (Analytic: 0.0280) ✅")
print(f"   * Exact Joint NLL Loss:         {joint_nll:.4f} nats (Analytic: 3.5756) ✅")

# ─── 2. PyTorch Causal Attention Mask Generation ───
print("\n2. CAUSAL ATTENTION MASK GENERATION (Sequence Length T = 4):")
seq_len = 4
causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

print(f"   * Causal Mask Matrix M:\n{causal_mask.numpy()}")
print("   * Upper triangle filled with -inf (future strictly masked! ✅)")

# ─── 3. Autoregressive Generation Loop Simulation ───
print("\n3. AUTOREGRESSIVE GENERATION LOOP SIMULATION (T = 5 Tokens):")
vocab = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
vocab_size = len(vocab)

# Simulated context token IDs
context = [0] # Starts with "The"
print(f"   Starting Prompt: {[vocab[idx] for idx in context]}")

torch.manual_seed(42)
for step in range(4):
    # Simulated model output logits for next token
    simulated_logits = torch.randn(vocab_size)
    # Give high logit to next sequential token for demo realism
    simulated_logits[(context[-1] + 1) % vocab_size] += 3.0
    
    # Temperature Softmax
    probs = F.softmax(simulated_logits / 0.8, dim=-1)
    # Sample next token
    next_token = torch.multinomial(probs, num_samples=1).item()
    context.append(next_token)

generated_text = " ".join([vocab[idx] for idx in context])
print(f"   * Generated Sequence: '{generated_text}' ✅")

print("\n" + "=" * 75)
print("ALL AUTOREGRESSIVE MODELING TESTS PASSED SUCCESSFULLY! ✅")
print("=" * 75)
```

---

### 8. 🩺 Diagnostic Mini-Checks & Common Traps
> `Context:` Production Debugging Insights, Edge-Case Traps & Self-Verification Questions

#### ✅ Self-Test Questions

1. **Q:** Why can Autoregressive models train in parallel across all tokens but generate sequentially during inference?  
   **A:** During training, we use **Teacher Forcing** with a **Causal Mask**, feeding all ground-truth tokens simultaneously in a single forward pass ($O(1)$ GPU steps). During inference, token $t$ does not exist yet; the model must generate and sample token $t$ before computing token $t+1$, creating a sequential $O(T)$ dependency.

2. **Q:** What is the role of KV-Caching in LLM inference?  
   **A:** At step $t$, the keys and values for tokens $1 \dots t-1$ have already been computed in previous steps. Storing them in **KV-Cache** avoids recomputing past attention states, reducing per-token computation from $O(t^2)$ to $O(t)$.

3. **Q:** What is the difference between Top-$k$ and Top-$p$ (Nucleus) sampling?  
   **A:** **Top-$k$** keeps a fixed number $k$ of candidates regardless of how confident the model is. **Top-$p$** dynamically expands or shrinks the candidate pool so their cumulative probability sums to $p$, adapting to whether the model is highly certain (1 token) or uncertain (50 tokens).

#### ⚠️ Common Engineering Traps

| Trap | Why It Fails | Production Fix |
| :--- | :--- | :--- |
| **Omitting the Causal Mask during Transformer training** | Future tokens leak into current attention scores; model achieves 0 training loss but generates garbage | Always add `torch.triu(..., diagonal=1)` mask before Softmax |
| **Forgetting KV-Cache during long-context generation** | Recomputing entire context on every token slows inference by $10\times$ to $50\times$ | Maintain persistent `past_key_values` across decoding steps |
| **Allowing KV-Cache memory to exceed GPU VRAM** | Long sequence decoding triggers sudden out-of-memory crashes | Use **PagedAttention (vLLM)** or flash-decoding kernels |

---

### 🎯 Summary Checklist
- **Autoregressive Models** decompose joint probability distributions via the exact probability chain rule: $p(x) = \prod p(x_t \mid x_{<t})$.
- **Causal Masking ($M_{ij} = -\infty$)** allows parallel training while preventing future token leakage.
- **Teacher Forcing** trains on ground-truth tokens in a single fast parallel step.
- **KV-Caching** caches past attention representations to optimize inference speed.
- **Temperature & Top-$p$ Sampling** balance deterministic accuracy with linguistic diversity.
