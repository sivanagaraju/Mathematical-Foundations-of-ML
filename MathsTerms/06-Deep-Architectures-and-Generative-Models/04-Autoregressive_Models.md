# Autoregressive Models: Exact Likelihood Factorization and Sequential Sampling

[Module guide](README.md) · [Study routes](START_HERE.md) · Previous: [Autoencoders and latent spaces](03-Autoencoders_and_Latent_Spaces.md) · Next: [Latent variable models](05-Latent_Variable_Models.md)

## 1. What this idea helps you do

Modeling a complete high-dimensional sequence (such as a 500-word paragraph, a 10-second audio track, or a robot motor trajectory) all at once triggers an impossible combinatorial explosion. With a standard vocabulary of 50,000 tokens, the number of possible 500-token sequences is $50,000^{500} \approx 10^{2350}$—far exceeding the number of atoms in the observable universe. Attempting to assign joint probabilities across all combinations simultaneously in an unconstrained table or energy model is computationally intractable.

An **Autoregressive Model** solves this dilemma by applying the **probability chain rule** to decompose the intractable joint distribution $p(x_1, \dots, x_T)$ into a product of 1D conditional next-token probabilities $\prod_{t=1}^T p(x_t \mid x_{<t})$. Each step predicts only the single next token given the historical prefix. During training, **causal masking** enables massively parallel processing across all tokens simultaneously ($O(1)$ GPU steps). During inference, **KV-caching** avoids redundant past computations as tokens are generated sequentially ($O(T)$ steps).

```text
================================================================================
          THE PROBABILITY CHAIN RULE AND CAUSAL FACTORIZATION
================================================================================
 JOINT PROBABILITY P(x_1, ..., x_T)           EXACT FACTORIZATION (CHAIN RULE)
 Intractable high-dimensional table           Product of 1D conditional factors
 ┌──────────────────────────────┐             ┌────────────────────────────────┐
 │ p(x_1, x_2, x_3, ..., x_T)   │ ══════════► │ p(x_1) · p(x_2|x_1) ·          │
 │ Combinatorial explosion      │             │ p(x_3|x_1, x_2) ...            │
 │ Cannot integrate directly    │             │ Exactly tractable likelihood   │
 └──────────────────────────────┘             └────────────────────────────────┘
                                                              │
                                                              ▼
 PARALLEL TRAINING (Teacher Forcing)          SEQUENTIAL INFERENCE / GENERATION
 ┌──────────────────────────────┐             ┌────────────────────────────────┐
 │ Parallel across all T tokens │             │ Step 1: sample x_1 ~ p(x_1)    │
 │ Upper-triangular causal mask │             │ Step 2: sample x_2 ~ p(x_2|x_1)│
 │ Loss = -sum log p(x_t|x_<t)  │             │ Step 3: sample x_3 ~ p(x_3|x_12│
 └──────────────────────────────┘             └────────────────────────────────┘
================================================================================
```

*What to notice from the diagram:*
1. The probability chain rule decomposes an intractable joint distribution into exact, tractable 1D conditional distributions without variational approximations.
2. During training, teacher forcing and triangular causal masking allow all tokens to be trained simultaneously in parallel.
3. During inference, tokens are emitted sequentially one-by-one, where each generated token becomes part of the conditioning prefix for subsequent tokens.

**Prerequisites**

- **Required now:** Conditional probability distributions and the chain rule of probability. [Joint, marginal and conditional distributions, §3](../03-Probability-and-Statistical-Estimation/03-Joint_Marginal_Conditional_Dist.md) covers conditional conditioning $p(A \mid B) = p(A, B)/p(B)$. [Softmax function, §2](../02-Multivariate-Calculus-and-Optimization/06-Softmax.md) defines temperature-scaled categorical distributions over logits.
- **Required for optional depth:** Negative Log-Likelihood and Kullback-Leibler divergence; see [Negative log-likelihood, §4](../03-Probability-and-Statistical-Estimation/06-NLL.md). These support the maximum likelihood estimation theorems in §4 and §8.
- **Useful context:** [Recurrent neural networks](02-Recurrent_Neural_Networks.md) for sequential state transitions and [Autoencoders and latent spaces](03-Autoencoders_and_Latent_Spaces.md) for discrete tokenization via VQ-VAEs.

**Target systems:** Large Language Models (GPT-4, Claude, LLaMA-3), autoregressive audio synthesis (AudioCraft, WaveNet), raster-scan image generation (Chameleon, PixelCNN), and robotic trajectory policies (RT-2, OpenVLA).

**Study time:** About 60–90 minutes for core probability proofs, causal masking mechanics, and logit gradient derivations; another 45–60 minutes for KV-cache arithmetic, sampling algorithms, and PyTorch verification.

After studying, you should be able to:

1. Prove by induction that autoregressive probability factorizations are strictly normalized distributions summing to 1.0.
2. Compute joint sequence likelihoods, Negative Log-Likelihood (NLL) losses, and analytical logit gradients ($\nabla_z \mathcal{L} = \hat{p} - y$) by hand.
3. Derive the analytical limits of temperature-scaled softmax as $\tau \to 0^+$ (argmax with ties) and $\tau \to \infty$ (uniform distribution).
4. Formulate causal attention masks ($M_{ij} \in \{0, -\infty\}$) and explain how they prevent future token leakage during parallel training.
5. Calculate KV-cache memory footprints, byte transfer requirements, and FLOP savings during sequential inference.
6. Implement, verify, and run complete autoregressive generation loops with temperature, top-$k$, and nucleus (top-$p$) sampling in pure Python and PyTorch.

**Fast route:** §§2–4 $\to$ §7 $\to$ §9 $\to$ §11 $\to$ §12, then §10 for real-world LLM systems.  
**Deep route:** §§2–14 in order; §4 and §8 contain the full induction normalization proof, temperature limit derivations, and KV-cache arithmetic intensity analysis.

You now understand the role of causal factorization. The open question is how sequential next-token choices avoid combinatorial explosions; an intuitive sentence ladder answers that directly.

---

## 2. Start with a problem you can picture

Suppose you ask an AI model to write a 4-word sentence using a small vocabulary of 1,000 common English words. How many possible 4-word sentences exist?
$$\text{Total Candidates} = 1,000^4 = 1,000,000,000,000 \quad (1\text{ trillion combinations})$$

If you tried to train a neural network to score all 1 trillion sentences simultaneously, the output layer would require 1 trillion neurons. Storing the weights of that single layer would demand 4 terabytes of GPU memory.

Autoregressive modeling replaces this giant, impossible guess with a sequence of 4 simple, tractable choices:

```text
ALL-AT-ONCE GUESS (IMPOSSIBLE)         AUTOREGRESSIVE FACTORIZATION
1 Trillion Joint States                4 Sequential 1,000-Way Choices

       p(w_1, w_2, w_3, w_4)                  Step 1: p(w_1) ──► "The"
                 ▲                                      │
                 │                            Step 2: p(w_2 | w_1) ──► "cat"
      [ IMPOSSIBLE GIANT TABLE ]                        │
                 │                            Step 3: p(w_3 | w_<3) ──► "sat"
                 ▼                                      │
       (1 Trillion States!)                   Step 4: p(w_4 | w_<4) ──► "down"
```

*What to notice from the diagram:*
1. Instead of allocating 1 trillion output coordinates, the model uses a single reusable 1,000-dimensional softmax layer four times in sequence.
2. The context expands at each rung of the ladder: each decision conditions on all previously committed tokens.
3. The total joint probability is the simple product of the 4 individual probabilities: $p(\text{"The cat sat down"}) = p(\text{"The"}) \times p(\text{"cat"} \mid \text{"The"}) \times p(\text{"sat"} \mid \text{"The cat"}) \times p(\text{"down"} \mid \text{"The cat sat"})$.

**Predict before calculating:** Does the order in which we factorize the sequence matter? Could we factorize the sentence backwards ($p(w_4) \times p(w_3 \mid w_4) \dots$) and still obtain the exact same joint probability $p(w_1, w_2, w_3, w_4)$?

To answer this question and formalize the mathematics, we define the objects and operators governing autoregressive systems.

---

## 3. Name the objects and read the notation

Let a discrete sequence of length $T$ be $X = (x_1, x_2, \dots, x_T)$, where each token $x_t \in \mathcal{V}$ is an element of a finite vocabulary $\mathcal{V}$ of size $V \triangleq |\mathcal{V}|$.

The past history (or prefix) preceding step $t$ is denoted:

$$x_{<t} \triangleq (x_1, x_2, \dots, x_{t-1}), \quad \text{with } x_{<1} \triangleq \emptyset$$

The model produces a vector of unnormalized real-valued logits $z_t \in \mathbb{R}^V$ at each position:

$$z_t \triangleq f_\theta(x_{<t})$$

The conditional next-token probability distribution is obtained via the **Softmax function** with temperature $\tau > 0$:

$$p_\theta(x_t = v \mid x_{<t}) \triangleq \frac{\exp(z_{t, v} / \tau)}{\sum_{j=1}^V \exp(z_{t, j} / \tau)}$$

Read this equation aloud:  
*“The probability under model theta of token x-sub-t equaling vocabulary word v, conditioned on prefix x-less-than-t, is defined as the exponential of logit z-sub-t-v divided by temperature tau, divided by the sum over all vocabulary items j of the exponential of logit z-sub-t-j divided by tau.”*

The complete joint likelihood is factorized via the **probability chain rule**:

$$p_\theta(x_1, \dots, x_T) \triangleq \prod_{t=1}^T p_\theta(x_t \mid x_{<t})$$

The universal training loss is the **Negative Log-Likelihood (NLL)** (equivalent to sequence cross-entropy):

$$\mathcal{L}_{\text{NLL}}(\theta) \triangleq -\ln p_\theta(X) = -\sum_{t=1}^T \ln p_\theta(x_t \mid x_{<t})$$

| Symbol | Spoken as | Mathematical role / dimensions | Concrete toy value (§2 / §9) |
| :--- | :--- | :--- | :--- |
| $\mathcal{V}$ | “script vee” | Discrete vocabulary dictionary of tokens | $\mathcal{V} = \{\text{"barks"}, \text{"runs"}, \text{"sleeps"}\}$ ($V=3$) |
| $x_t$ | “ex sub tee” | Discrete token at timestep $t$; $x_t \in \mathcal{V}$ | $x_1 = \text{"The"}, x_2 = \text{"dog"}, x_3 = \text{"barks"}$ |
| $x_{<t}$ | “ex less than tee” | Conditioning prefix tuple $(x_1, \dots, x_{t-1})$ | $x_{<3} = (\text{"The"}, \text{"dog"})$ |
| $z_t$ | “zee sub tee” | Unnormalized logit vector at step $t$; $\mathbb{R}^V$ | $z_3 = [2.0, 1.0, 0.0]^\top$ |
| $\tau$ | “tau” | Softmax temperature hyperparameter; $\tau > 0$ | Default: $\tau = 1.0$; Focused: $\tau = 0.5$ |
| $p_\theta(x_t \mid x_{<t})$ | “p sub theta” | Categorical probability over vocabulary | $\hat{p} = [0.6652, 0.2447, 0.0900]^\top$ |
| $M$ | “capital emm” | Causal attention mask matrix; $T \times T$ | $M_{ij} = -\infty$ for $j > i$; $0$ for $j \le i$ |
| $\text{PPL}$ | “perplexity” | Geometric mean exponentiated cross-entropy | $\text{PPL} = \exp(\mathcal{L} / T) \approx 3.29$ |
| $K_t, V_t$ | “key and value” | Transformer attention key/value tensors | Stored in GPU VRAM across generation steps |

We now have the vocabulary and structural definitions. We can now derive the central relationship: why the probability chain rule is an exact mathematical identity and how temperature scaling governs generation limits.

---

## 4. Build the central relationship

### The Core "Aha!" Discovery

An autoregressive model does not need to know how to write an entire encyclopedia all at once. If it can accurately predict just the *single next word* given what came before, it can recursively generate arbitrarily long, coherent sequences of text, code, or music.

### Step-by-Step Proof: Probability Chain Rule Normalization

Why does multiplying step-by-step conditional probabilities guarantee a strictly valid, normalized joint probability distribution that sums to 1.0? Let us prove this by mathematical induction on sequence length $T$.

#### Theorem 4.1: Normalization of Autoregressive Factorization

Let $\mathcal{V}$ be a finite vocabulary. If for every prefix sequence $x_{<t} \in \mathcal{V}^{t-1}$, the conditional distribution is non-negative and normalized:

$$p(x_t \mid x_{<t}) \ge 0 \quad \forall x_t \in \mathcal{V}, \qquad \sum_{x_t \in \mathcal{V}} p(x_t \mid x_{<t}) = 1$$

then the product distribution $p(x_{1:T}) \triangleq \prod_{t=1}^T p(x_t \mid x_{<t})$ is a valid joint probability distribution over $\mathcal{V}^T$ satisfying:

$$\sum_{x_{1:T} \in \mathcal{V}^T} p(x_{1:T}) = 1$$

#### Proof by Mathematical Induction:

1. **Base Case ($T = 1$):**  
   For a sequence of length 1, $p(x_1) = p(x_1 \mid x_{<1})$. By the premise:
   $$\sum_{x_1 \in \mathcal{V}} p(x_1) = 1$$
   The base case holds trivially.

2. **Inductive Hypothesis:**  
   Assume the theorem holds for sequences of length $T - 1$, so that:
   $$\sum_{x_{1:T-1} \in \mathcal{V}^{T-1}} \prod_{t=1}^{T-1} p(x_t \mid x_{<t}) = 1$$

3. **Inductive Step for Length $T$:**  
   Summing the joint probability over all $V^T$ possible sequences of length $T$:
   $$\sum_{x_{1:T} \in \mathcal{V}^T} p(x_{1:T}) = \sum_{x_1 \in \mathcal{V}} \sum_{x_2 \in \mathcal{V}} \dots \sum_{x_T \in \mathcal{V}} \left( \prod_{t=1}^T p(x_t \mid x_{<t}) \right)$$
   Splitting the product into the prefix product and the final step factor:
   $$= \sum_{x_{1:T-1} \in \mathcal{V}^{T-1}} \sum_{x_T \in \mathcal{V}} \left( \prod_{t=1}^{T-1} p(x_t \mid x_{<t}) \right) p(x_T \mid x_{<T})$$
   Notice that the prefix product term $\prod_{t=1}^{T-1} p(x_t \mid x_{<t})$ depends only on tokens $x_1, \dots, x_{T-1}$ and is completely constant with respect to the innermost index of summation $x_T$. Factoring it out:
   $$= \sum_{x_{1:T-1} \in \mathcal{V}^{T-1}} \left( \prod_{t=1}^{T-1} p(x_t \mid x_{<t}) \right) \left[ \sum_{x_T \in \mathcal{V}} p(x_T \mid x_{<T}) \right]$$

4. **Evaluate the Inner Sum:**  
   By the premise, for any given prefix $x_{<T}$, the conditional distribution over next tokens sums to 1:
   $$\sum_{x_T \in \mathcal{V}} p(x_T \mid x_{<T}) = 1.0$$
   Substituting this back into the expression:
   $$= \sum_{x_{1:T-1} \in \mathcal{V}^{T-1}} \left( \prod_{t=1}^{T-1} p(x_t \mid x_{<t}) \right) \cdot 1.0$$

5. **Apply the Inductive Hypothesis:**  
   By the inductive hypothesis, the remaining sum over prefixes of length $T-1$ equals 1:
   $$\sum_{x_{1:T-1} \in \mathcal{V}^{T-1}} \prod_{t=1}^{T-1} p(x_t \mid x_{<t}) = 1.0$$
   Therefore:
   $$\boxed{\sum_{x_{1:T} \in \mathcal{V}^T} p(x_{1:T}) = 1.0}$$
   By mathematical induction, the factorization yields an exact, normalized probability distribution over all sequences of arbitrary length $T$. $\blacksquare$

```text
================================================================================
                    THE CAUSAL ATTENTION MASKING MECHANISM
================================================================================
 Raw Attention Scores: QK^T / sqrt(d_k)       Causal Mask Matrix M (j > i = -inf)
 ┌────────────────────────────────────┐       ┌────────────────────────────────┐
 │ S_11   S_12   S_13   S_14          │   +   │   0     -inf   -inf   -inf     │
 │ S_21   S_22   S_23   S_24          │       │   0       0    -inf   -inf     │
 │ S_31   S_32   S_33   S_34          │       │   0       0      0    -inf     │
 │ S_41   S_42   S_43   S_44          │       │   0       0      0      0      │
 └────────────────────────────────────┘       └────────────────────────────────┘
                                  │
                                  ▼
 Post-Softmax Attention Weights A = Softmax(S + M):
 exp(-inf) = 0.0 zeroes out all future attention weights strictly!
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ Token 1: [ A_11 ,  0.0 ,  0.0 ,  0.0 ] ──► Can only see Token 1             │
 │ Token 2: [ A_21 ,  A_22,  0.0 ,  0.0 ] ──► Can see Tokens 1 and 2           │
 │ Token 3: [ A_31 ,  A_32,  A_33,  0.0 ] ──► Can see Tokens 1, 2, and 3       │
 │ Token 4: [ A_41 ,  A_42,  A_43,  A_44] ──► Can see all 4 tokens!            │
 └─────────────────────────────────────────────────────────────────────────────┘
================================================================================
```

*What to notice from the diagram:*
1. Adding $-\infty$ to the upper triangle of attention logits before the softmax guarantees that $\exp(-\infty) = 0.0$.
2. Every token position $i$ is strictly blindfolded from attending to future tokens $j > i$.
3. This triangular structure allows all $T$ positions to be processed in parallel during training without future data leakage.

### Analytical Limits of Temperature Scaling

Let $z = (z_1, \dots, z_V) \in \mathbb{R}^V$ be a logit vector. The temperature-scaled categorical distribution is:

$$p_i(\tau) = \frac{\exp(z_i / \tau)}{\sum_{j=1}^V \exp(z_j / \tau)}, \qquad \tau > 0$$

#### Limit 1: Infinite Temperature ($\tau \to \infty$) — Maximum Entropy / Uniform Distribution

As $\tau \to \infty$, the exponent approaches zero:

$$\lim_{\tau \to \infty} \frac{z_i}{\tau} = 0 \implies \lim_{\tau \to \infty} \exp\left(\frac{z_i}{\tau}\right) = e^0 = 1.0 \quad \forall i \in \{1, \dots, V\}$$

Evaluating the denominator:

$$\lim_{\tau \to \infty} \sum_{j=1}^V \exp(z_j / \tau) = \sum_{j=1}^V 1.0 = V$$

Therefore:

$$\boxed{\lim_{\tau \to \infty} p_i(\tau) = \frac{1}{V} \quad \forall i \in \{1, \dots, V\}}$$

At infinite temperature, all logit differences vanish, and the model outputs pure, unbiased uniform randomness.

#### Limit 2: Zero Temperature ($\tau \to 0^+$) — Deterministic Argmax with Ties

Let $z_{\max} \triangleq \max_{j} z_j$ be the maximum logit value. Let the set of tied argmax indices be:

$$K \triangleq \left\{ k \in \{1, \dots, V\} : z_k = z_{\max} \right\}, \qquad |K| \ge 1$$

Factor out $\exp(z_{\max} / \tau)$ from numerator and denominator:

$$p_i(\tau) = \frac{\exp\left(\frac{z_i - z_{\max}}{\tau}\right)}{\sum_{j=1}^V \exp\left(\frac{z_j - z_{\max}}{\tau}\right)} = \frac{\exp\left(\frac{z_i - z_{\max}}{\tau}\right)}{\sum_{j \in K} \exp\left(\frac{0}{\tau}\right) + \sum_{j \notin K} \exp\left(\frac{z_j - z_{\max}}{\tau}\right)}$$

For any non-maximal index $j \notin K$, the difference is strictly negative: $z_j - z_{\max} < 0$. As $\tau \to 0^+$:

$$\lim_{\tau \to 0^+} \frac{z_j - z_{\max}}{\tau} = -\infty \implies \lim_{\tau \to 0^+} \exp\left(\frac{z_j - z_{\max}}{\tau}\right) = 0$$

For any maximal index $j \in K$, $z_j - z_{\max} = 0$, so $\exp(0) = 1$. The denominator becomes:

$$\lim_{\tau \to 0^+} \left[ \sum_{j \in K} 1 + \sum_{j \notin K} 0 \right] = |K|$$

Evaluating the numerator for index $i$:

$$\lim_{\tau \to 0^+} \exp\left(\frac{z_i - z_{\max}}{\tau}\right) = \begin{cases} 1 & \text{if } i \in K \\ 0 & \text{if } i \notin K \end{cases}$$

Therefore:

$$\boxed{\lim_{\tau \to 0^+} p_i(\tau) = \begin{cases} \frac{1}{|K|} & \text{if } i \in K \\ 0 & \text{if } i \notin K \end{cases}}$$

If the argmax is unique ($|K| = 1$), the distribution collapses to a Dirac delta mass on $\arg\max_j z_j$ (pure greedy decoding). If there are $|K|$ tied maximum logits, the probability mass is split uniformly among the tied candidates. $\blacksquare$

### 5-Second Mental Memory Hooks
- **"Auto" means Self, "Regressive" means Looking Back:** The model feeds its own past predictions back into itself as future inputs.
- **Teacher Forcing in Training:** All tokens are processed in a single parallel step ($O(1)$ GPU operations) using causal mask blindfolds.
- **Sequential in Inference:** Tokens are generated one-by-one ($O(T)$ steps) like a chain of dominos falling in order.

We now have the mathematical normalization proof and temperature limits. Next, we contrast autoregressive models with alternative generative paradigms.

---

## 5. Why choose this tool for this problem?

| Generative Dimension | Autoregressive Model (GPT / LLaMA) | Variational Autoencoder (VAE) | Generative Adversarial Net (GAN) | Diffusion / Flow Matching |
| :--- | :--- | :--- | :--- | :--- |
| **Likelihood Evaluation** | **Exact** ($p(X) = \prod p(x_t \mid x_{<t})$) | **Approximate Lower Bound** (ELBO) | **Implicit** (No tractable density) | **Tractable Bound** via score matching |
| **Sampling Speed** | **Sequential $O(T)$** (One token per step) | **Single-Step $O(1)$** (One forward pass) | **Single-Step $O(1)$** (One forward pass) | **Iterative $O(K)$** ($K \sim 20$–$50$ denoising steps) |
| **Mode Coverage** | **Complete** (Minimizes Forward KL $\implies$ covers all modes) | **Broad** (Covers modes, but blurs fine features) | **Prone to Mode Collapse** (Zero mass on missing modes) | **High Fidelity & Full Coverage** |
| **Discrete Sequence Data** | **Native & Exact** (Standard cross-entropy loss) | **Difficult** (Requires Gumbel-Softmax or discrete VQ) | **Fails** (Cannot backpropagate through discrete tokens) | **Emerging** (Continuous embeddings or bit diffusion) |
| **Dominant AI Domain** | **Language, Code, & Reasoning** | **Latent Image Compression** | **Real-Time Synthesis** | **Photorealistic Images & Video** |

### Concrete Mathematical Counterexample: Future Token Leakage from Causal Mask Omission

Suppose a developer trains a 4-token sequence model on $X = [x_1, x_2, x_3, x_4]$ using a Transformer without a causal mask (setting $M_{ij} = 0$ everywhere):

1. **Unmasked Attention Matrix:** The attention mechanism computes $A = \operatorname{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)$. Because $M_{ij} = 0$, position 1 is allowed to attend directly to position 2.
2. **Trivial Shortcut Learning:** The model discovers that the optimal strategy to minimize next-token cross-entropy is not to understand linguistic context, but simply to copy the embedding of token 2 directly into the output of token 1 ($x_1 \to x_2$).
3. **Spurious Zero Training Loss:** The logit for the correct target token becomes arbitrarily large ($z_{\text{true}} \to +\infty$). Within a single training epoch, the cross-entropy loss drops to nearly zero ($\mathcal{L}_{\text{NLL}} < 0.0001$). The engineer mistakenly assumes the model has learned the language.
4. **Catastrophic Inference Failure:** At deployment, the user enters a prompt consisting only of token $x_1$. Token $x_2$ does not exist in the context window yet.
5. **Gibberish Output:** Looking for the future token, the attention layer encounters zeroes or arbitrary padding tokens. The model immediately emits garbled repetition loops or unconditioned static.
6. **The Mandatory Causal Mask:** Setting $M_{ij} = -\infty$ for all $j > i$ ensures that $\exp(-\infty) = 0$, mathematically guaranteeing that token $i$ has zero information about future tokens $j > i$.

We have established why causal masking is mathematically mandatory. Next, we examine real-world mental models and their breaking points.

---

## 6. Strengthen the intuition and mark its limits

```text
================================================================================
          END-TO-END AI LIFECYCLE: AUTOREGRESSIVE GENERATION IN LLMS
================================================================================
 USER PROMPT: "The quick brown "
      │
      ▼ [1. Tokenizer maps words to integer IDs]: [464, 2068, 7586]
 CONTEXT WINDOW: 3 Tokens
      │
      ▼ [2. Transformer Forward Pass with Causal Masking]:
 Computes unnormalized vocabulary logits across 50,000 candidate words
      │
      ▼ [3. Softmax & Temperature Scaling (tau = 0.7)]:
 p("fox") = 0.88,  p("dog") = 0.08,  p("bear") = 0.03
      │
      ▼ [4. Sample Next Token]: Emits "fox" (Token ID 21831)
 CONTEXT WINDOW EXPANDS: "The quick brown fox" (4 Tokens)
      │
      ▼ [5. Store new Key and Value in KV-Cache & Repeat Loop until <EOS>]
================================================================================
```

*What to notice from the diagram:*
1. Token generation is a closed feedback loop: the output of step $t$ becomes the input for step $t+1$.
2. The tokenizer maps discrete text words into integer IDs, which index embedding matrices.
3. The generation loop terminates when the model samples the special End-of-Sequence token (`<EOS>`).

### Real-World Metaphors

#### Metaphor 1: The Domino Chain Reaction
- You arrange 1,000 dominos in a line.
- You do not push all 1,000 dominos simultaneously. You tip over Domino 1. Domino 1 strikes Domino 2, which strikes Domino 3.
- Each domino falling is an autoregressive step, triggered entirely by the momentum of the dominoes that fell before it.

#### Metaphor 2: The Improv Storyteller
- In an improv storytelling exercise, a performer can only say one word, and then their partner must immediately say the next word.
- Neither speaker knows what the entire paragraph will look like in advance.
- Each speaker listens to the words uttered so far, determines the most natural next word, and passes the microphone forward.

### Mechanical Mapping: Intuition to Mathematics and Implementation

| Physical Intuition / Metaphor | Mathematical Operation | Software / Hardware Implementation | Failure Mode / Boundary Condition |
| :--- | :--- | :--- | :--- |
| **Tipping dominoes one by one** | Sequential conditioning $p(x_t \mid x_{<t})$ | Sequential decode loop with KV-cache | Latency scales as $O(T)$; cannot parallelize token generation |
| **Parallel row of pre-arranged dominoes** | Joint sequence evaluation under teacher forcing $\prod p(x_t \mid x_{<t})$ | Causal masked attention with upper-triangular $-\infty$ | If causal mask is omitted ($M_{ij}=0$), model cheats via identity copying |
| **Improv speaker choosing next word** | Categorical sampling from temperature-scaled softmax | `torch.multinomial(F.softmax(logits / tau))` | $\tau \to 0$ collapses to greedy repetition; $\tau \gg 1$ yields incoherent babble |
| **Storyteller notes / short-term memory** | Key-Value cache persistence across decoding steps | PagedAttention / vLLM allocation in GPU HBM | VRAM explodes linearly ($2 B L H T d_k$ bytes); triggers OOM on long context |

### Where this analogy stops working

1. **Memory-Bandwidth Inference Bottleneck:** While the domino metaphor implies smooth, effortless falling, modern GPU hardware experiences an extreme memory-bandwidth bottleneck during generation. In pretraining, matrix multiplications are compute-bound on Tensor Cores. During single-token inference, the GPU must transfer all model weights (e.g., 140 GB for a 70B parameter model in float16) from High-Bandwidth Memory (HBM) to compute just **one token**. The Arithmetic Intensity drops below $1\text{ FLOP/byte}$.
2. **Compounding Exposure Bias:** In the domino chain, physical tracks guide each domino cleanly. In an LLM, the model is trained with **Teacher Forcing** (always conditioned on human ground-truth prefixes). During inference, the model conditions on its own previously generated tokens. If the model makes a slight mistake at step 5, it enters an unconditioned state it never saw in training data, causing errors to compound uncontrollably.

We now have the mental models and their operational limits. Next, we clarify essential technical terms.

---

## 7. Terms worth keeping straight

### Core Terminology Reference Table

| Term | Pronunciation | Plain-English Meaning | Formal Definition & Conditions |
| :--- | :--- | :--- | :--- |
| **Autoregressive Model** | “aw-toh-ree-GRES-iv” | Predicts the next token conditioned strictly on preceding tokens | $p(x_{1:T}) = \prod_{t=1}^T p(x_t \mid x_{<t})$. Causal factorization. |
| **Causal Masking** | “KAW-zul MASK-ing” | Upper-triangular attention mask preventing lookahead to future tokens | $M_{ij} = -\infty$ for $j > i$. Enforces autoregressive order in parallel training. |
| **Teacher Forcing** | “TEE-cher FORS-ing” | Feeding ground-truth prefix tokens at every training step | Maximizes $\sum_t \log p(x_t^* \mid x_{<t}^*)$ without compounding model errors. |
| **Free-Running Generation** | “free RUN-ing” | Autonomously generating text by feeding model's own samples back | $x_t \sim p(x \mid \hat{x}_{<t})$; serial sampling step at inference. |
| **KV Cache** | “kay-vee kash” | Storing Key and Value projection matrices to avoid recomputation | Stores $K_{1:t}, V_{1:t}$ in VRAM; transforms $O(T^2)$ generation to $O(T)$. |
| **Exposure Bias** | “ek-SPOH-zher BY-us” | Train/test distribution shift from never seeing own generation errors | Gap between teacher-forced training and autoregressive rollout. |
| **Perplexity** | “per-PLEK-sih-tee” | Geometric mean of inverse probabilities of a sequence | $\text{PPL} = \exp\left( -\frac{1}{T} \sum_{t=1}^T \log p(x_t \mid x_{<t}) \right)$. |
| **Nucleus Sampling** | “NOO-klee-us” | Restricting token sampling to cumulative probability top-$p$ mass | Selects smallest set $V^{(p)} \subset V$ with $\sum_{v \in V^{(p)}} p(v) \ge p$. |

### Confused Pairs Distinction Breakdown

1. **Autoregressive Model vs. Masked Language Model (BERT)**:
   - *Core Distinction:* Autoregressive factors probability causally forward ($p(x_t \mid x_{<t})$); masked language models predict bidirectional context ($p(x_t \mid x_{\setminus t})$).
   - *Common Confusion:* Assuming BERT can be used for autoregressive text generation simply by looping.
   - *Rule of Thumb:* Use autoregressive (GPT/Llama) for generation; use bidirectional (BERT) for classification and embeddings.
- **Autoregressive Model (Causal / Decoder-Only):** Factorizes probabilities strictly forward: $p(X) = \prod_{t=1}^T p(x_t \mid x_{<t})$. Can generate text autonomously by sampling sequentially.
- **Masked Language Model (Bidirectional / Encoder-Only):** Masks random tokens in the middle (e.g. 15% of tokens) and trains the model to predict them using both past and future context ($p(x_t \mid x_{\setminus t})$). Excellent for classification and embeddings, but cannot generate coherent text sequentially.

### 2. Teacher Forcing vs. Free-Running Autoregressive Generation
- **Teacher Forcing (Training):** At every step $t$, the network is fed the true ground-truth prefix $x_{<t}$ from the training set, regardless of what the network would have predicted. Allows parallel computation across the entire sequence.
- **Free-Running Generation (Inference):** The model samples its own output $\hat{x}_t \sim p(x_t \mid \hat{x}_{<t})$ and appends it to its own context window for step $t+1$.

### 3. KV-Cache vs. Standard Activation Memory
- **Standard Activation Memory:** Temporary intermediate tensors stored during the forward pass and kept for backpropagation during training; discarded immediately after the gradient step.
- **KV-Cache (Key-Value Cache):** Persistent GPU VRAM storage holding Key and Value projection matrices for all historical tokens during inference. It grows linearly with sequence length: $O(B \times L \times H \times T \times d_k)$.

### 4. Greedy Decoding vs. Temperature Sampling vs. Nucleus (Top-$p$) Sampling
- **Greedy Decoding ($\tau \to 0$):** Deterministically selects the token with the single highest logit: $x_t = \arg\max_v z_{t, v}$. Fast, but prone to repetitive loops.
- **Temperature Sampling ($\tau$):** Divides logits by $\tau$ and samples from the resulting categorical distribution. Controls distribution sharpness.
- **Nucleus / Top-$p$ Sampling:** Truncates the vocabulary to the smallest subset of tokens whose cumulative probability exceeds threshold $p$ (e.g., $p = 0.90$), dynamically adjusting the pool size based on model confidence.

We have clarified key terminology. Next, we work through formal mathematical equations, KV-cache scaling laws, and hardware bandwidth constraints.

---

## 8. Work through the mathematics and its conditions

### Core Mathematical Formulations

#### 1. The Sequence Negative Log-Likelihood (NLL) Loss
Given a training sequence of $T$ tokens $X = (x_1, \dots, x_T)$ and model parameters $\theta$:

$$\mathcal{L}_{\text{NLL}}(\theta) = -\sum_{t=1}^T \ln p_\theta(x_t \mid x_{<t}) = -\sum_{t=1}^T \ln\left( \frac{\exp(z_{t, x_t} / \tau)}{\sum_{j=1}^V \exp(z_{t, j} / \tau)} \right)$$

#### 2. Analytical Gradient of Cross-Entropy with Respect to Logits
Let $z_t \in \mathbb{R}^V$ be the logit vector at position $t$, and let $y_t \in \{0, 1\}^V$ be the one-hot target vector where $y_{t, k} = 1$ if $k = x_t$ and $0$ otherwise.  
Applying the multivariate chain rule:

$$\frac{\partial \mathcal{L}_t}{\partial z_{t, i}} = \hat{p}_{t, i} - y_{t, i}$$

In vector notation:

$$\boxed{\nabla_{z_t} \mathcal{L}_t = \hat{p}_t - y_t}$$

*Key mathematical property:* The sum of logit gradients across the entire vocabulary is identically zero:

$$\sum_{i=1}^V \frac{\partial \mathcal{L}_t}{\partial z_{t, i}} = \sum_{i=1}^V (\hat{p}_{t, i} - y_{t, i}) = \sum_{i=1}^V \hat{p}_{t, i} - \sum_{i=1}^V y_{t, i} = 1.0 - 1.0 = \mathbf{0}$$

This occurs because adding a constant scalar $c$ to all logits leaves softmax probabilities completely invariant: $\operatorname{Softmax}(z + c \mathbf{1}) = \operatorname{Softmax}(z)$.

#### 3. Causal Masked Scaled Dot-Product Attention
For query matrix $Q \in \mathbb{R}^{T \times d_k}$, key matrix $K \in \mathbb{R}^{T \times d_k}$, and value matrix $V \in \mathbb{R}^{T \times d_v}$:

$$\text{Attention}(Q, K, V) = \operatorname{Softmax}\left( \frac{QK^\top}{\sqrt{d_k}} + M \right) V$$

where the causal mask matrix $M \in \mathbb{R}^{T \times T}$ is defined as:

$$M_{ij} \triangleq \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

### KV-Cache Memory and Bandwidth Complexity

#### Memory Footprint Formula
In multi-head attention with batch size $B$, number of Transformer layers $L$, number of KV attention heads $H_{KV}$, head dimension $d_k$, sequence length $t$, and data precision (bytes per element):

$$\text{Memory}_{\text{KV}}(t) = 2 \times B \times L \times H_{KV} \times t \times d_k \times \text{bytes\_per\_elem}$$

The factor of $2$ accounts for storing both the **Key** tensor and the **Value** tensor.

*Grouped-Query Attention (GQA):* Modern LLMs (like LLaMA-3) set $H_{KV} = \frac{H_Q}{8}$, using 8 times fewer KV heads than query heads, reducing KV-cache VRAM consumption by an exact factor of $8\times$.

#### Hardware Arithmetic Intensity
During autoregressive generation at batch size 1:
- To generate 1 token, the GPU must stream all model parameter weights $W$ from High-Bandwidth Memory (HBM) to on-chip SRAM cache.
- For a model with $P$ parameters in 16-bit precision ($\text{bytes} = 2P$), computing the forward pass requires $2P$ FLOPs.
- The Arithmetic Intensity is:
  $$\text{Arithmetic Intensity} = \frac{\text{Compute (FLOPs)}}{\text{Memory Access (Bytes)}} = \frac{2P \text{ FLOPs}}{2P \text{ Bytes}} = \mathbf{1.0 \text{ FLOP/byte}}$$
- An NVIDIA H100 GPU can perform $1,979 \times 10^{12} \text{ FLOP/s}$ (FP16 Tensor Cores), but its memory bandwidth is only $3.35 \times 10^{12} \text{ bytes/s}$.
- To fully saturate the compute cores, an algorithm requires an arithmetic intensity of $\frac{1,979}{3.35} \approx 590 \text{ FLOP/byte}$.
- Because autoregressive inference operates at $\approx 1 \text{ FLOP/byte}$, **more than 99% of GPU compute capacity sits idle**, waiting for memory transfers from HBM.

We now have the formal loss formulations, gradient mechanics, and hardware scaling laws. Next, we work through concrete numerical calculations by hand.

---

## 9. Calculate it by hand

### Worked Example 1: 3-Token Joint Likelihood and Analytical Cross-Entropy Logit Gradients

Consider a 3-token sentence: $X = (\text{"The"}, \text{"dog"}, \text{"barks"})$.  
Given the following conditional probabilities from an autoregressive language model:
- $p(x_1 = \text{"The"}) = \mathbf{0.1000}$
- $p(x_2 = \text{"dog"} \mid x_1 = \text{"The"}) = \mathbf{0.4000}$
- $p(x_3 = \text{"barks"} \mid x_{<3} = \text{"The dog"}) = \mathbf{0.7000}$

---

#### Step 1: Forward Joint Probability, Sequence NLL, and Perplexity

1. **Joint Probability via Chain Rule:**
   $$p(\text{"The dog barks"}) = 0.1000 \times 0.4000 \times 0.7000 = 0.0400 \times 0.7000 = \mathbf{0.0280} \quad (2.80\%)$$
2. **Sequence Negative Log-Likelihood (NLL):**
   $$\text{NLL} = -\ln(0.1000) - \ln(0.4000) - \ln(0.7000)$$
   Evaluating individual terms:
   $$-\ln(0.1000) \approx 2.302585, \quad -\ln(0.4000) \approx 0.916291, \quad -\ln(0.7000) \approx 0.356675$$
   $$\text{Total NLL} = 2.302585 + 0.916291 + 0.356675 = \mathbf{3.575551\text{ nats}}$$
   *Cross-check:* $-\ln(0.0280) = \mathbf{3.575551\text{ nats}}$.
3. **Sequence Perplexity (PPL):**
   $$\text{Average NLL per token} = \frac{3.575551}{3} \approx 1.191850\text{ nats}$$
   $$\text{PPL} = \exp(1.191850) \approx \mathbf{3.2932}$$
   *Interpretation:* Across this 3-token sequence, the model was on average as uncertain as if choosing uniformly among $3.29$ equally likely words per position.

---

#### Step 2: Backward Cross-Entropy Gradient Derivation on Logits

Suppose at step 3, the candidate vocabulary logits for `["barks", "runs", "sleeps"]` are:

$$z = \begin{bmatrix} z_1 \\ z_2 \\ z_3 \end{bmatrix} = \begin{bmatrix} 2.0000 \\ 1.0000 \\ 0.0000 \end{bmatrix}$$

The ground-truth next token is $x_3 = \text{"barks"}$ (index 1), so target vector $y = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$.

1. **Evaluate Softmax Probabilities ($\tau = 1.0$):**
   $$e^{z_1} = e^{2.0} \approx 7.389056, \quad e^{z_2} = e^{1.0} \approx 2.718282, \quad e^{z_3} = e^{0.0} = 1.000000$$
   $$\text{Denominator } Z = 7.389056 + 2.718282 + 1.000000 = 11.107338$$
   $$\hat{p}_1 = \frac{7.389056}{11.107338} \approx \mathbf{0.665241}, \quad \hat{p}_2 = \frac{2.718282}{11.107338} \approx \mathbf{0.244728}, \quad \hat{p}_3 = \frac{1.000000}{11.107338} \approx \mathbf{0.090031}$$
2. **Evaluate Cross-Entropy Loss:**
   $$\mathcal{L}_3 = -\ln(\hat{p}_1) = -\ln(0.665241) \approx \mathbf{0.407606\text{ nats}}$$
3. **Compute Exact Analytical Logit Gradients ($\nabla_z \mathcal{L}_3 = \hat{p} - y$):**
   $$\frac{\partial \mathcal{L}_3}{\partial z_1} = \hat{p}_1 - y_1 = 0.665241 - 1.000000 = \mathbf{-0.334759}$$
   $$\frac{\partial \mathcal{L}_3}{\partial z_2} = \hat{p}_2 - y_2 = 0.244728 - 0.000000 = \mathbf{+0.244728}$$
   $$\frac{\partial \mathcal{L}_3}{\partial z_3} = \hat{p}_3 - y_3 = 0.090031 - 0.000000 = \mathbf{+0.090031}$$
   $$\nabla_z \mathcal{L}_3 = \begin{bmatrix} -0.334759 \\ +0.244728 \\ +0.090031 \end{bmatrix}$$
4. **Verification of Zero Sum:**
   $$\sum_{i=1}^3 \frac{\partial \mathcal{L}_3}{\partial z_i} = -0.334759 + 0.244728 + 0.090031 = \mathbf{0.000000}$$

---

#### Step 3: Gradient Descent Update and Physical Sign Interpretation

With learning rate $\eta = 0.50$:

1. **Update Logit Coordinates ($z \leftarrow z - \eta \nabla_z \mathcal{L}_3$):**
   $$z_1^{(1)} = 2.0000 - (0.50)(-0.334759) = 2.0000 + 0.167380 = \mathbf{2.167380}$$
   $$z_2^{(1)} = 1.0000 - (0.50)(+0.244728) = 1.0000 - 0.122364 = \mathbf{0.877636}$$
   $$z_3^{(1)} = 0.0000 - (0.50)(+0.090031) = 0.0000 - 0.045016 = \mathbf{-0.045016}$$
2. **Physical Sign Interpretation:**
   - **Why did $z_1$ increase ($2.00 \to 2.1674$)?** Token 1 is the correct target ($y_1 = 1$), but the model only predicted $\hat{p}_1 = 66.52\% < 100\%$. The residual error is negative ($-0.3348$). Subtracting a negative gradient increases logit $z_1$, actively raising the probability of the correct token.
   - **Why did $z_2$ and $z_3$ decrease?** Tokens 2 and 3 are incorrect ($y_2 = 0, y_3 = 0$). Their gradients are positive ($+0.2447$ and $+0.0900$). Subtracting positive gradients penalizes and suppresses incorrect competitor tokens.

---

### Worked Example 2: Temperature Scaling, Top-$k$, and Nucleus (Top-$p$) Sampling Worked by Hand

Let vocabulary logits at step $t$ be $z = [2.0000, 1.0000, 0.0000]$ for `["cat", "dog", "fish"]`.

#### 1. Temperature Scaling Comparison
- **Default Temperature $\tau = 1.0$:**
  $$p_{\tau=1.0} = [0.6652, 0.2447, 0.0900]$$
- **Focused Temperature $\tau = 0.50$:**
  Scaled logits: $z / 0.5 = [4.0, 2.0, 0.0]$
  $$e^{4.0} \approx 54.5982, \quad e^{2.0} \approx 7.3891, \quad e^{0.0} = 1.0000 \implies Z = 62.9873$$
  $$p_{\tau=0.5} = [0.8668, 0.1173, 0.0159] \quad \text{(Target probability boosted by +20.1%!)}$$
- **Creative Temperature $\tau = 2.00$:**
  Scaled logits: $z / 2.0 = [1.0, 0.5, 0.0]$
  $$e^{1.0} \approx 2.7183, \quad e^{0.5} \approx 1.6487, \quad e^{0.0} = 1.0000 \implies Z = 5.3670$$
  $$p_{\tau=2.0} = [0.5065, 0.3072, 0.1863] \quad \text{(Distribution flattens toward uniform randomness)}$$

#### 2. Top-$k$ Truncation ($k = 2$)
1. Sort candidate probabilities: $\text{"cat"} (0.6652) \ge \text{"dog"} (0.2447) > \text{"fish"} (0.0900)$.
2. Retain top $k=2$ candidates: `["cat", "dog"]`. Set $p(\text{"fish"}) = 0.0$.
3. Renormalize over surviving mass $Z_{\text{top2}} = 0.6652 + 0.2447 = 0.9099$:
   $$p_{\text{renorm}}(\text{"cat"}) = \frac{0.6652}{0.9099} \approx \mathbf{0.7311}, \quad p_{\text{renorm}}(\text{"dog"}) = \frac{0.2447}{0.9099} \approx \mathbf{0.2689}$$

#### 3. Nucleus / Top-$p$ Truncation ($p = 0.90$)
1. Compute cumulative sum of sorted probabilities:
   - "cat": $0.6652 < 0.90$
   - "cat" + "dog": $0.6652 + 0.2447 = 0.9099 \ge 0.90 \implies \text{Cutoff reached!}$
2. Candidate pool consists of `["cat", "dog"]` (renormalized to $[0.7311, 0.2689]$, eliminating "fish").

---

### Worked Example 3: KV-Cache Attention FLOP and Byte Savings by Hand

Consider generating token $t = 4$ in an autoregressive Transformer where head dimension $d_k = 64$.
- **Without KV-Cache (Recomputing from scratch):**
  - Must recompute Keys and Values for tokens $1, 2, 3, 4$: $4$ Key projections and $4$ Value projections.
  - Must compute all query-key dot products: $Q_1 K_1^\top, Q_2 K_{1..2}^\top, Q_3 K_{1..3}^\top, Q_4 K_{1..4}^\top$ ($1 + 2 + 3 + 4 = 10$ vector dot products).
  - Total dot products across $T$ tokens scales quadratically: $\sum_{t=1}^T t = \frac{T(T+1)}{2} = \mathcal{O}(T^2)$.
- **With KV-Cache (Reusing cached keys and values):**
  - Keys and Values for tokens $1, 2, 3$ are already stored in GPU VRAM.
  - Compute Key and Value for the single new token $t = 4$ ($1$ projection each).
  - Compute query vector for token $t=4$ against the 4 cached keys: $Q_4 K_{1..4}^\top$ ($1 \times 4$ dot products).
  - Operations at step $t$: exactly $t$ vector operations. Per-token computational cost drops from $\mathcal{O}(t^2)$ to $\mathcal{O}(t)$!

We now have the step-by-step arithmetic verified. Next, we examine where autoregressive modeling integrates into generative AI architectures.

---

## 10. Connect the concept to an actual system

```text
================================================================================
                    AUTOREGRESSIVE GENERATIVE ARCHITECTURES
================================================================================
 1. TRANSFORMER LLMS (GPT-4 / LLaMA-3)         2. SPECULATIVE DECODING ENGINE
 Causal Attention + KV-Cache Decoding          Draft Model (1B) proposes K tokens
 ┌────────────────────────────────────────┐    ┌───────────────────────────────┐
 │ Parallel training on 10T tokens        │    │ Large Model (70B) verifies K  │
 │ Sequential token-by-token generation   │    │ tokens in 1 parallel pass     │
 │ Powers human dialogue, code, reasoning │    │ Speeds up inference 2.5x-3.0x │
 └────────────────────────────────────────┘    └───────────────────────────────┘
================================================================================
```

*What to notice from the diagram:*
1. Modern LLMs decouple training (massively parallel matrix multiplication) from inference (memory-bandwidth bound sequential decoding).
2. Advanced inference acceleration techniques like speculative decoding use small draft models to propose multiple tokens, verifying them in parallel with the large model to bypass the memory bandwidth bottleneck.

| Mathematical Object | Role in Toy Example | Real Production System Counterpart | Hardware / Scale Approximation |
| :--- | :--- | :--- | :--- |
| **Probability Chain Rule** | Factorizes $p(\text{"The dog barks"}) = p(x_1)p(x_2\|x_1)p(x_3\|x_{<3})$ | Autoregressive token factorization in LLaMA-3 / GPT-4 | Exact mathematical decomposition; no variational or independence approximations |
| **Logit Vector $z_t$** | 3-element vector $z_3 = [2.0, 1.0, 0.0]^\top$ over vocabulary $\mathcal{V}$ | 128,256-element unnormalized logit vector from unembedding projection layer | Computed via `logits = lm_head(hidden_states)` ($B \times T \times V$ tensor in bfloat16) |
| **Causal Attention Mask $M$** | Upper-triangular matrix with $-\infty$ above diagonal | Causal attention mask in FlashAttention-2 / SDPA kernels | Fused into GPU SRAM kernel so $-\infty$ values are never materialized in HBM |
| **KV-Cache $(K_t, V_t)$** | Stores keys and values for tokens $1, 2, 3$ to avoid recomputing past attention | Persistent KV cache stored across decoding steps in GPU VRAM | Managed by PagedAttention (vLLM); occupies $2 B L H_{KV} T d_k$ bytes of HBM |

We have mapped the architectural connections. Next, we verify these formulations with executable Python and PyTorch scripts.

---

## 11. Verify the idea with a small experiment

We implement the **Dual-Stage Code Architecture**:
- **Stage 1 (Pure Python):** Transparent standard library implementation computing temperature-scaled softmax, sequence Negative Log-Likelihood, analytical logit gradients ($\hat{p} - y$), top-$k$ and nucleus (top-$p$) filtering, and an autoregressive generation loop without external libraries.
- **Stage 2 (Production PyTorch):** Implements an autograd verification suite comparing analytical gradients to PyTorch autograd, tests causal attention masking, and verifies bit-exact numerical equivalence between parallel attention and cached step attention ($< 10^{-6}$).

```python
"""
Autoregressive Modeling and Causal Decoding Verification Suite
==============================================================
Part A: Pure Python standard library (built-in math only, zero dependencies).
Part B: PyTorch industrial verification suite with autograd logit checks,
        causal masking verification, and KV-cache bit-exact equivalence.
"""

import math

print("=" * 80)
print("PART A: PURE PYTHON STDLIB AUTOREGRESSIVE & TEMPERATURE DECODING")
print("=" * 80)

# ─── 1. Pure Python Softmax with Temperature ───
def pure_python_softmax(logits, temperature=1.0):
    """Numerically stable softmax with temperature scaling."""
    scaled = [l / temperature for l in logits]
    max_val = max(scaled)
    exp_vals = [math.exp(v - max_val) for v in scaled]
    total = sum(exp_vals)
    return [v / total for v in exp_vals]

# Section 9 Example 1 Verification
logits = [2.0, 1.0, 0.0]
probs_tau1 = pure_python_softmax(logits, temperature=1.0)
probs_tau05 = pure_python_softmax(logits, temperature=0.50)
probs_tau20 = pure_python_softmax(logits, temperature=2.00)

print("1. Pure Python Temperature Scaling (Logits: [2.0, 1.0, 0.0]):")
print(f"   * tau = 1.0: {[round(p, 4) for p in probs_tau1]} (Expected: [0.6652, 0.2447, 0.0900])")
print(f"   * tau = 0.5: {[round(p, 4) for p in probs_tau05]} (Expected: [0.8668, 0.1173, 0.0159])")
print(f"   * tau = 2.0: {[round(p, 4) for p in probs_tau20]} (Expected: [0.5065, 0.3072, 0.1863])")

assert math.isclose(probs_tau1[0], 0.6652, rel_tol=1e-3)
assert math.isclose(probs_tau05[0], 0.8668, rel_tol=1e-3)
assert math.isclose(probs_tau20[0], 0.5065, rel_tol=1e-3)

# ─── 2. Cross-Entropy Loss & Analytical Logit Gradient ───
target_idx = 0  # Target: "barks"
loss_ce = -math.log(probs_tau1[target_idx])
grad_logits = [probs_tau1[i] - (1.0 if i == target_idx else 0.0) for i in range(len(logits))]

print(f"\n2. Pure Python Loss & Analytical Logit Gradients (Target: Index {target_idx}):")
print(f"   * Cross-Entropy Loss:    {loss_ce:.6f} nats (Expected: 0.407606)")
print(f"   * Analytical Logit Grad: {[round(g, 6) for g in grad_logits]}")
print(f"   * Sum of Gradients:      {sum(grad_logits):.8f} (Expected: 0.0000)")

assert math.isclose(loss_ce, 0.407606, rel_tol=1e-4)
assert math.isclose(grad_logits[0], -0.334759, rel_tol=1e-4)
assert math.isclose(grad_logits[1], +0.244728, rel_tol=1e-4)
assert math.isclose(grad_logits[2], +0.090031, rel_tol=1e-4)
assert math.isclose(sum(grad_logits), 0.0, abs_tol=1e-7)

# ─── 3. Top-k & Nucleus (Top-p) Filtering ───
def top_k_filter(probs, k=2):
    indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    surviving_indices = {idx for idx, _ in indexed[:k]}
    filtered = [p if i in surviving_indices else 0.0 for i, p in enumerate(probs)]
    total = sum(filtered)
    return [p / total for p in filtered]

def top_p_filter(probs, p_thresh=0.90):
    indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    cum = 0.0
    surviving_indices = set()
    for idx, p in indexed:
        surviving_indices.add(idx)
        cum += p
        if cum >= p_thresh:
            break
    filtered = [p if i in surviving_indices else 0.0 for i, p in enumerate(probs)]
    total = sum(filtered)
    return [p / total for p in filtered]

top_k_probs = top_k_filter(probs_tau1, k=2)
top_p_probs = top_p_filter(probs_tau1, p_thresh=0.90)

print("\n3. Pure Python Sampling Filters:")
print(f"   * Top-k (k=2) Renormalized: {[round(p, 4) for p in top_k_probs]} (Expected: [0.7311, 0.2689, 0.0])")
print(f"   * Top-p (p=0.90) Renormalized: {[round(p, 4) for p in top_p_probs]}")

assert math.isclose(top_k_probs[0], 0.7311, rel_tol=1e-3)
assert top_k_probs[2] == 0.0
assert math.isclose(top_p_probs[0], 0.7311, rel_tol=1e-3)
print("Part A Pure Python Suite: ALL CHECKS PASSED [OK]")

print("\n" + "=" * 80)
print("PART B: PYTORCH INDUSTRIAL VERIFICATION & KV-CACHE EQUIVALENCE SUITE")
print("=" * 80)

import torch
import torch.nn.functional as F
import numpy as np

# ─── 1. PyTorch Autograd vs Analytical Logit Gradient ───
z_torch = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64, requires_grad=True)
target_torch = torch.tensor(0, dtype=torch.long)
loss_torch = F.cross_entropy(z_torch.unsqueeze(0), target_torch.unsqueeze(0))
loss_torch.backward()

expected_grad = torch.tensor([-0.3347590442, +0.2447284711, +0.0900305732], dtype=torch.float64)
print("1. PyTorch Autograd vs Analytical Gradients:")
print(f"   * PyTorch Loss:       {loss_torch.item():.6f}")
print(f"   * PyTorch z.grad:     {z_torch.grad.tolist()}")
print(f"   * Analytical Grad:    {expected_grad.tolist()}")

assert torch.allclose(z_torch.grad, expected_grad, atol=1e-6)

# ─── 2. Causal Attention Mask Verification ───
seq_len = 4
d_k = 8
torch.manual_seed(42)
Q = torch.randn(1, seq_len, d_k)
K = torch.randn(1, seq_len, d_k)
V = torch.randn(1, seq_len, d_k)

scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
masked_scores = scores + causal_mask
attn_weights = F.softmax(masked_scores, dim=-1)
out_parallel = torch.matmul(attn_weights, V)

print(f"\n2. Causal Attention Mask Verification (Seq Len T = {seq_len}):")
print(f"   * Masked Upper Triangle Sum: {torch.triu(attn_weights, diagonal=1).sum().item():.8f} (Expected: 0.0)")
assert torch.allclose(torch.triu(attn_weights, diagonal=1), torch.zeros(seq_len, seq_len))

# ─── 3. KV-Cache Step Equivalence Check ───
# In step 3 (0-indexed, 4th token), we only pass Q for token 3 and reuse cached K[:4], V[:4]
q_step = Q[:, 3:4, :]  # (1, 1, d_k)
k_cache = K[:, :4, :]  # (1, 4, d_k)
v_cache = V[:, :4, :]  # (1, 4, d_k)

step_scores = torch.matmul(q_step, k_cache.transpose(-2, -1)) / math.sqrt(d_k)
step_attn = F.softmax(step_scores, dim=-1)
out_cached_step = torch.matmul(step_attn, v_cache)  # (1, 1, d_k)

cached_diff = torch.norm(out_parallel[:, 3:4, :] - out_cached_step).item()
print(f"\n3. KV-Cache Numerical Equivalence at Step 4:")
print(f"   * Difference between Parallel & Cached Step: {cached_diff:.8f}")
print(f"   * Exact Equivalence (< 1e-6): {cached_diff < 1e-6} [OK]")

assert cached_diff < 1e-6, "KV-Cache representation diverged from parallel attention!"

print("\n" + "=" * 80)
print("ALL AUTOREGRESSIVE MODELING TESTS PASSED SUCCESSFULLY! [OK]")
print("=" * 80)
```

*Expected output:*
```text
================================================================================
PART A: PURE PYTHON STDLIB AUTOREGRESSIVE & TEMPERATURE DECODING
================================================================================
1. Pure Python Temperature Scaling (Logits: [2.0, 1.0, 0.0]):
   * tau = 1.0: [0.6652, 0.2447, 0.09] (Expected: [0.6652, 0.2447, 0.0900])
   * tau = 0.5: [0.8668, 0.1173, 0.0159] (Expected: [0.8668, 0.1173, 0.0159])
   * tau = 2.0: [0.5065, 0.3072, 0.1863] (Expected: [0.5065, 0.3072, 0.1863])

2. Pure Python Loss & Analytical Logit Gradients (Target: Index 0):
   * Cross-Entropy Loss:    0.407606 nats (Expected: 0.407606)
   * Analytical Logit Grad: [-0.334759, 0.244728, 0.090031]
   * Sum of Gradients:      -0.00000000 (Expected: 0.0000)

3. Pure Python Sampling Filters:
   * Top-k (k=2) Renormalized: [0.7311, 0.2689, 0.0] (Expected: [0.7311, 0.2689, 0.0])
   * Top-p (p=0.90) Renormalized: [0.7311, 0.2689, 0.0]
Part A Pure Python Suite: ALL CHECKS PASSED [OK]

================================================================================
PART B: PYTORCH INDUSTRIAL VERIFICATION & KV-CACHE EQUIVALENCE SUITE
================================================================================
1. PyTorch Autograd vs Analytical Gradients:
   * PyTorch Loss:       0.407606
   * PyTorch z.grad:     [-0.3347590442, 0.2447284711, 0.0900305732]
   * Analytical Grad:    [-0.3347590442, 0.2447284711, 0.0900305732]

2. Causal Attention Mask Verification (Seq Len T = 4):
   * Masked Upper Triangle Sum: 0.00000000 (Expected: 0.0)

3. KV-Cache Numerical Equivalence at Step 4:
   * Difference between Parallel & Cached Step: 0.00000000
   * Exact Equivalence (< 1e-6): True [OK]

================================================================================
ALL AUTOREGRESSIVE MODELING TESTS PASSED SUCCESSFULLY! [OK]
================================================================================
```

The code confirms the analytical logit gradient derivation, causal mask integrity, and exact KV-cache numerical equivalence. Next, test your diagnostic understanding through active practice.

---

## 12. Practise, compare, and debug

Attempt all five exercises before consulting the separated diagnostic solutions.

1. **Recognize.** A language model processes an input sequence of length $T = 2048$. What is the total number of attention score entries in the full self-attention matrix before masking? How many entries are converted to zero after applying the upper-triangular causal attention mask?
2. **Calculate.** Suppose vocabulary logits at step $t$ are $z = [3.0, 1.0, 1.0]$. The ground-truth next token is token 1 ($y = [1, 0, 0]$). Compute the softmax probabilities at temperature $\tau = 1.0$, the scalar cross-entropy loss, and the exact analytical gradient vector $\nabla_z \mathcal{L}$.
3. **Contrast.** Contrast Teacher Forcing during training with Free-Running Autoregressive Generation during deployment. Why does exposure bias never appear during training, and how does it manifest during deployment?
4. **Transfer.** A model generates text using nucleus sampling with $p = 0.85$. At step $t$, the top candidate token has a predicted probability of $0.92$. How many candidate tokens will be included in the sampling pool for this step? What does this imply about the model's generation behavior?
5. **Debug.** An engineer implements a KV-cache for an open-source Transformer. The model generates the first token accurately, but starting from token 2, the model generates repetitive babble. The engineer’s decoding loop snippet is:
   ```python
   # Buggy generation loop
   past_key_values = None
   for step in range(max_new_tokens):
       # Model returns logits and updated KV cache
       logits, past_key_values = model(input_ids, past_key_values=past_key_values)
       next_token = sample(logits[:, -1, :])
       input_ids = torch.cat([input_ids, next_token], dim=-1)
   ```
   Diagnose the bug and explain why repetitive degeneration occurs.

---

<details>
<summary>Answer key and diagnostic feedback</summary>

1. **Attention Matrix Entries:**  
   - Total entries: $T \times T = 2048 \times 2048 = \mathbf{4,194,304\text{ entries}}$.  
   - Upper-triangular entries (strictly future tokens $j > i$):  
     $$\frac{T(T - 1)}{2} = \frac{2048 \times 2047}{2} = \mathbf{2,096,128\text{ entries}}$$  
   - Exactly $49.98\%$ of the matrix entries are zeroed out by the causal mask.  
   *Diagnostic feedback:* If you calculated $2048^2 / 2$, you included the diagonal entries ($j = i$). A token can always attend to itself, so the diagonal is unmasked!

2. **Calculation:**  
   - $e^3 \approx 20.0855, \quad e^1 \approx 2.7183, \quad e^1 \approx 2.7183$.  
   - Denominator $Z = 20.0855 + 2.7183 + 2.7183 = 25.5221$.  
   - $\hat{p}_1 = \frac{20.0855}{25.5221} \approx \mathbf{0.7870}, \quad \hat{p}_2 = \hat{p}_3 = \frac{2.7183}{25.5221} \approx \mathbf{0.1065}$.  
   - $\mathcal{L} = -\ln(0.7870) \approx \mathbf{0.2395\text{ nats}}$.  
   - Gradients ($\hat{p} - y$):  
     $$\frac{\partial \mathcal{L}}{\partial z_1} = 0.7870 - 1.0 = \mathbf{-0.2130}$$  
     $$\frac{\partial \mathcal{L}}{\partial z_2} = 0.1065 - 0.0 = \mathbf{+0.1065}$$  
     $$\frac{\partial \mathcal{L}}{\partial z_3} = 0.1065 - 0.0 = \mathbf{+0.1065}$$  
   *Diagnostic feedback:* Check the sum: $-0.2130 + 0.1065 + 0.1065 = 0.0000$. The gradient sum is always zero.

3. **Contrast (Teacher Forcing vs. Exposure Bias):**  
   - Under Teacher Forcing, the conditioning prefix is always a real, pristine human sentence drawn from the training corpus. The network is never evaluated on a corrupted prefix.  
   - During free-running generation, each emitted token is sampled from the model's own distribution. If the model makes an unlikely or ungrammatical choice at step 5, the prefix is now an out-of-distribution sentence that never appeared in the training corpus. The model was never trained to recover from its own errors, causing compounding nonsensical generations.

4. **Transfer (Nucleus Pool Collapse):**  
   - Because candidate 1 alone has probability $0.92 \ge 0.85$, the cumulative probability threshold is satisfied on the very first token.  
   - The sampling pool collapses to **exactly 1 token**: candidate 1. The token is chosen with $100\%$ certainty.  
   - *Implication:* Nucleus sampling automatically converts to deterministic greedy decoding when the model is confident, avoiding unnecessary randomness.

5. **Debug (Double-Counting Context in KV-Cache):**  
   - **Bug:** The loop appends `next_token` to `input_ids` on line 7 (`torch.cat([input_ids, next_token])`), and then in the next iteration passes the entire expanded `input_ids` tensor back into `model(input_ids, past_key_values=...)`.  
   - **Consequence:** If `past_key_values` is provided, the model already holds the cached keys and values for all past tokens. Passing the full sequence causes the model to recompute and duplicate keys/values for past tokens, shifting positional encodings and corrupting the attention matrix.  
   - **Fix:** When using a KV-cache, pass **only the single newly sampled token** into the model:
     ```python
     # Corrected snippet
     next_token = sample(logits[:, -1, :])
     # Pass only the 1 new token on subsequent iterations!
     input_ids = next_token.unsqueeze(-1)
     ```

</details>

---

### Transfer Challenge: Sequence Perplexity and Prefix Factorization

**Scenario:** A language model evaluates a 3-token code snippet $\boldsymbol{w} = (\text{"def"}, \text{"foo"}, \text{"():"})$. The model predicts:
- $P(\text{"def"}) = 0.50$
- $P(\text{"foo"} \mid \text{"def"}) = 0.40$
- $P(\text{"():"} \mid \text{"def foo"}) = 0.80$

1. **Calculate Joint Probability:** Compute the exact sequence probability $P(\boldsymbol{w})$.
2. **Compute Negative Log-Likelihood:** Using natural logs ($\ln 0.50 \approx -0.6931, \ln 0.40 \approx -0.9163, \ln 0.80 \approx -0.2231$), compute the sequence NLL.
3. **Calculate Perplexity:** Compute $\text{PPL} = \exp(\text{NLL} / 3)$ and explain how this metric is interpreted in language model benchmarks.

*Transfer Solution:*
1. **Joint Probability:**
   $$P(\boldsymbol{w}) = 0.50 \times 0.40 \times 0.80 = \mathbf{0.1600} \quad (16\%)$$
2. **Negative Log-Likelihood:**
   $$\text{NLL}(\boldsymbol{w}) = -[\ln(0.50) + \ln(0.40) + \ln(0.80)] = -[-0.6931 - 0.9163 - 0.2231] = \mathbf{1.8325\text{ nats}}$$
3. **Sequence Perplexity:**
   $$\text{Average NLL} = \frac{1.8325}{3} \approx 0.6108\text{ nats} \implies \text{PPL} = \exp(0.6108) \approx \mathbf{1.8420}$$
   *Interpretation:* A perplexity of $1.84$ means that at each generation step, the model is on average as uncertain as if choosing uniformly among $1.84$ equally likely candidate tokens. Lower perplexity indicates higher predictive confidence.

---

### Production Engineering Traps

| Trap | Why It Fails | Production Fix |
| :--- | :--- | :--- |
| **Omitting the Causal Mask during Transformer training** | Future tokens leak into current attention scores; model achieves 0 training loss but generates garbage | Always add `torch.triu(..., diagonal=1)` mask before Softmax |
| **Forgetting KV-Cache during long-context generation** | Recomputing entire context on every token slows inference by $10\times$ to $50\times$ | Maintain persistent `past_key_values` across decoding steps |
| **Allowing KV-Cache memory to exceed GPU VRAM** | Long sequence decoding triggers sudden out-of-memory crashes | Use **PagedAttention (vLLM)** or flash-decoding kernels |
| **Overly high temperature ($\tau > 1.5$) during factual question answering** | Probability distribution flattens, causing random hallucination of low-probability tokens | Use $\tau \le 0.2$ for code/math and $\tau \approx 0.7$ for creative prose |

---

## 13. Explain it back and return to it

**Closed-notes Feynman prompt:**  
Imagine explaining to a software engineer who only knows standard supervised classification why modern AI models write text one word at a time rather than generating a full paragraph in one shot, how an upper-triangular mask of negative infinities lets the model train on thousands of words in parallel without cheating, and why generating text uses up so much GPU memory even though the model is only predicting one word. Do not use the terms “autoregressive factorization” or “arithmetic intensity” in your initial plain-English summary. Once you finish, restore the formal terms and state the joint probability chain rule equation.

<details>
<summary>Model explanation for self-evaluation</summary>

Trying to make an AI guess an entire 500-word essay all at once is impossible because there are more combinations of words than atoms in the universe. Instead, we break the task down: the AI only ever has to guess the *single next word* based on what has already been written. Once it guesses that word, it adds it to the story and repeats the process.

To train the network quickly, we don't want to feed words one-by-one. We feed the entire training book into the GPU all at once. To stop the AI from looking ahead and reading the upcoming words, we place a digital blindfold over the future tokens in the attention grid by adding negative infinity to the upper triangle. When the numbers pass through the exponent curve, negative infinity becomes zero, physically blocking the AI from peeking into the future.

Finally, when the model generates text, it has to remember every past calculation so it doesn't recalculate the whole story from scratch. Storing these past attention keys and values takes gigabytes of memory. Furthermore, the GPU has to pull the entire multi-gigabyte neural network from memory chips to the processing cores just to produce a single word, making generation speed limited by memory bandwidth rather than processor speed.

*Restoring formal terminology:* The step-by-step breakdown is **autoregressive factorization** governed by the **probability chain rule**: $p(x_{1:T}) = \prod_{t=1}^T p(x_t \mid x_{<t})$. The blindfold is the **causal attention mask** $M_{ij} \in \{0, -\infty\}$, and the memory bottleneck is governed by **KV-cache allocation** and **memory-bandwidth bound arithmetic intensity**.

</details>

### Spaced Repetition Schedule

| Return Date | Closed-Notes Retrieval Task | Self-Verification Anchor |
| :--- | :--- | :--- |
| **Day 1** | Write out the probability chain rule for a 4-token sequence. Compute by hand the analytical cross-entropy logit gradient vector $\hat{p} - y$ for 3 vocabulary items. | Check against §9 Example 1. |
| **Day 7** | Re-derive the proof by induction showing that autoregressive factorizations sum to 1.0. Explain why temperature $\tau \to \infty$ produces a uniform distribution. | Check against Theorem 4.1 and §4 limits. |
| **Day 30** | Explain how KV-caching reduces per-token computational complexity from $O(t^2)$ to $O(t)$, and explain why single-token inference is memory-bandwidth bound. | Check against §8 arithmetic intensity analysis. |

### Self-Assessment Checklist

- [ ] I can write down the probability chain rule and prove by mathematical induction that it defines a normalized probability distribution.
- [ ] I can compute the forward joint probability, Negative Log-Likelihood, and perplexity of a sequence by hand.
- [ ] I can derive the analytical cross-entropy logit gradient $\nabla_z \mathcal{L} = \hat{p} - y$ and explain why its elements sum to zero.
- [ ] I can prove the analytical limits of temperature scaling as $\tau \to 0^+$ (argmax with ties) and $\tau \to \infty$ (uniform distribution).
- [ ] I can construct a causal attention mask matrix and explain how $-\infty$ prevents future token leakage during parallel training.
- [ ] I can calculate the KV-cache memory footprint and explain why autoregressive inference is memory-bandwidth bound on GPUs.

---

## 14. Continue with a purposeful learning path

The resources below are verified for relevance, active status, and pedagogical precision as of **2026-09-18**. Access descriptions indicate verified availability at check time.

| Resource and author | Learning job | Exact starting point | Readiness | Access | Checked date and evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Interactive visualizer:** [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), Jay Alammar | Visual step-by-step breakdown of self-attention, causal masking, and decoder blocks | Section: "The Decoder Side" and the causal masking animation | After §2 | Free open educational blog | 2026-09-18: verified active interactive animations and causal self-attention matrix walkthroughs. |
| **Video lecture:** [Neural Networks: Zero to Hero (Building GPT from scratch)](https://www.youtube.com/watch?v=kCc8FmEb1nY), Andrej Karpathy | Masterful code and theory walkthrough of causal attention and autoregressive generation | Timestamp 42:15: "Mathematical Trick in Self-Attention: The Causal Mask" | After §4 | Free YouTube video | 2026-09-18: verified active video, line-by-line PyTorch implementation of causal masking and generation loop. |
| **Foundational paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Ashish Vaswani et al. (NeurIPS 2017) | Original derivation of scaled dot-product attention and causal decoder masking | Section 3.1: "Scaled Dot-Product Attention" and Section 3.2.3: "Decoder" | After §8 | Free open-access arXiv preprint | 2026-09-18: verified original definition of upper-triangular masking ($M_{ij} = -\infty$). |
| **Textbook:** [Speech and Language Processing, Chapter 9: Language Models](https://web.stanford.edu/~jurafsky/slp3/9.pdf), Dan Jurafsky & James H. Martin | Authoritative academic treatment of probability chain rule, NLL, and perplexity | Chapter 9: §9.1 (N-gram & Neural LMs), §9.3 (Perplexity), and §9.4 (Sampling) | After §4 | Free online PDF (Stanford University, 2024) | 2026-09-18: verified section numbers, mathematical chain rule derivations, and evaluation metrics. |
| **Systems paper:** [Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180), Woosuk Kwon et al. (SOSP 2023) | Understand KV-cache memory fragmentation and production serving throughput | Section 2: "Background on LLM Serving" and Section 3: "PagedAttention" | After §8 | Free open-access arXiv preprint | 2026-09-18: verified active paper, exact KV-cache memory waste calculations, and virtual memory paging. |
| **Practice problem set:** [Stanford CS224N: NLP with Deep Learning, Assignment 4](https://web.stanford.edu/class/cs224n/), Stanford University | Implement causal attention and autoregressive generation by hand | Section 1: "Neural Machine Translation with Causal Masking" | After §12 | Free university course assignment | 2026-09-18: verified assignment problem set covering causal decoder masks and temperature sampling. |
| **Software documentation:** [Hugging Face Transformers: Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies), Hugging Face Contributors | Production engineering reference for temperature, top-k, and nucleus sampling | "Greedy Search", "Sample", and "Beam Search" sections | When running §11 | Free official documentation | 2026-09-18: verified active guide for configuring generation parameters in PyTorch. |

**Next connection:** Autoregressive models factorize joint distributions over observed tokens. In [latent variable models](05-Latent_Variable_Models.md), we explore what happens when observations depend on unobserved, continuous latent causes, proving why marginal evidence integrals become intractable and motivating the Evidence Lower Bound (ELBO).
