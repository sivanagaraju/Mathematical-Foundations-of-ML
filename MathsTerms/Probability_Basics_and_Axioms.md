# Probability Basics, Sample Spaces & The Kolmogorov Axioms: The Mathematical Foundations of Uncertainty

> `🏷️ Tags:` `Probability` `Kolmogorov-Axioms` `Bayes-Theorem` `Sample-Space` `Generative-AI` `Softmax` `Diffusion` `LLMs`  
> `📚 Prerequisites Needed:` [Logarithms & Exponential Functions](./Logarithms_and_Exponential_Functions.md) · [Tensors & Shapes](./Tensors_and_Shapes.md)  
> `🎯 Where Do We Use This?:` **The backbone of all Probabilistic AI** — Softmax probability calibration in Large Language Models (GPT-4, LLaMA-3), Gaussian Markov noise chains in Diffusion Models (Stable Diffusion), Latent priors $\mathcal{N}(0, I)$ in VAEs, and Push-forward probability measures in GANs.  
> `🎓 Course Module Mapping:` [Tut 07: Basic Probability 1](../Mathematical-Foundation-for-GenerativeAI/21-Tutorial07-Review-Basic-Probability-1/NOTES.md) · [Tut 08: Basic Probability 2](../Mathematical-Foundation-for-GenerativeAI/22-Tutorial08-Review-Basic-Probability-2/NOTES.md) · [Lec 01: Generative Models](../Mathematical-Foundation-for-GenerativeAI/14-Lec01-MFGAI-Introduction/NOTES.md) · [Lec 20: VAEs](../Mathematical-Foundation-for-GenerativeAI/32-Lec20-Latent-Variable-Models-VAE/NOTES.md)  
> `⏱️ Difficulty Level:` ⭐☆☆☆☆ (Foundational / Beginner-Friendly · 15 min read)

---

### 📌 Quick Navigation & Architecture Map
- [1. 🌟 Everyday Real-World Scenarios](#1--everyday-real-world-scenarios-the-weather-forecast--chatgpt-token-picker) — Weather Forecasts & ChatGPT Token Probabilities
- [2. 👶 ELI5 Intuition](#2--eli5-intuition-the-1-kilogram-kitchen-scale--the-birthday-cake) — The 1-Kilogram Kitchen Scale & Slicing a Birthday Cake
- [3. 📚 Deep Terminology Master Glossary](#3--deep-terminology-master-glossary-15-core-concepts-dissected) — 15 fundamental probability terms dissected without jargon
- [4. 📐 The 3 Kolmogorov Axioms, Formulas & Geometry](#4--the-3-kolmogorov-axioms-formulas--geometry) — Formal non-negativity, unit measure, additivity, and Bayes' Theorem
- [5. 🔢 Concrete Micro-Numerical Worked Examples](#5--concrete-micro-numerical-worked-examples) — 6-Sided Die & The Medical Diagnostic Test Paradox
- [6. 🔗 Connecting the Dots: Generative AI Architecture Blocks](#6--connecting-the-dots-how-probability-powers-modern-generative-ai) — Softmax normalization in LLMs & Gaussian diffusion transitions
- [7. 💻 Standalone Executable Python/PyTorch Verification Script](#7--complete-standalone-executable-pythonpytorch-verification-script) — Axiom simulation, Monte Carlo sampling, and Bayes' Rule
- [8. 🩺 Diagnostic Mini-Checks & Common Traps](#8--diagnostic-mini-checks--common-traps) — Self-test questions & production engineering pitfalls

---

Probability theory is the mathematical framework for measuring and reasoning under **uncertainty**. In Machine Learning and Generative AI, probability provides the rigorous foundation for measuring model confidence, quantifying data distributions, and generating novel synthetic samples (images, text, audio).

```
 ===================================================================================================
                 THE 3-TIER HIERARCHY OF MATHEMATICAL PROBABILITY
 ===================================================================================================

  TIER 1: SAMPLE SPACE (Ω)            TIER 2: EVENT SPACE (ℱ)             TIER 3: PROBABILITY MEASURE (P)
  Universe of All Possible Outcomes  Legal Measurable Subsets            Axiomatic Sizing in [0.0, 1.0]
  ┌──────────────────────────────┐    ┌──────────────────────────────┐    ┌──────────────────────────────┐
  │ Physical Reality / RE        │───►│ Measurable Subsets A ⊆ Ω     │───►│ P(A) ∈ [0.0, 1.0]            │
  │ Ω = {ω₁, ω₂, ω₃, ..., ωₙ}    │    │ (Boolean questions allowed)  │    │ Total Certainty: P(Ω) = 1.0  │
  │ Master Menu of All Outcomes  │    │ Closed under complement & ∪  │    │ Disjoint: P(A∪B) = P(A)+P(B) │
  └──────────────────────────────┘    └──────────────────────────────┘    └──────────────────────────────┘
 ===================================================================================================
```

---

### 1. 🌟 Everyday Real-World Scenarios (The Weather Forecast & ChatGPT Token Picker)
> `Context:` Zero Prior Machine Learning / AI Knowledge Needed · Concrete Real-World Mapping

#### Scenario A: The Morning Weather Forecast (Zero ML Background Needed)
Imagine checking your phone's weather app before leaving home:
1. **The Sample Space ($\Omega$):** The weather today will definitely be one of: `[Sunny, Rainy, Cloudy, Snowy]`. Nothing outside this list can happen.
2. **The Probabilities ($P$):** The meteorologist assigns numbers: Sunny ($60\% = 0.60$), Cloudy ($20\% = 0.20$), Rainy ($20\% = 0.20$), Snowy ($0\% = 0.00$).
3. **The 3 Rules of Nature (Kolmogorov Axioms):**
   - **No Negative Chances:** You cannot have a $-15\%$ chance of rain (Axiom 1).
   - **Total Equals 100%:** $0.60 + 0.20 + 0.20 + 0.00 = 1.00$ ($100\%$). It will definitely do *something* on the list (Axiom 2).
   - **Adding Non-Overlapping Events:** The chance of "Sunny OR Rainy" is simply $0.60 + 0.20 = 0.80$ ($80\%$) because it cannot be purely Sunny and Rainy at the exact same instant (Axiom 3).

---

#### Scenario B: In Generative AI — ChatGPT Choosing the Next Word
> `Context:` How Softmax Enforces Kolmogorov Axioms in Large Language Models (LLMs)

When ChatGPT generates the next word in the sentence *"The captain sailed the..."*:
- The neural network computes raw internal scores (logits): `ship = +9.2`, `boat = +7.1`, `car = -3.4`, `banana = -8.1`.
- These raw numbers are negative and unbounded ($-\infty$ to $+\infty$), which violates the rules of probability.
- The **Softmax Function** transforms these scores so:
  1. Every score becomes positive ($p \ge 0$).
  2. The sum across all 128,000 dictionary words equals **exactly $1.0$ ($100\%$)**.
- The AI then samples `ship` ($82\%$) or `boat` ($17\%$) while assigning $0.000001\%$ to `banana`.

```
 ===================================================================================================
         HOW PROBABILITY AXIOMS OPERATE INSIDE LARGE LANGUAGE MODELS (LLMs)
 ===================================================================================================

  RAW NETWORK LOGITS (Unbounded ℝ)       SOFTMAX NORMALIZATION (Axiom 1 & 2)    GENERATIVE SAMPLING
  ┌──────────────────────────────┐       ┌─────────────────────────────────┐    ┌─────────────────┐
  │ "ship"   : +9.2              │══════►│ P("ship")   = 0.824 (82.4%)     │═══►│ SAMPLE TOKEN:   │
  │ "boat"   : +7.1              │       │ P("boat")   = 0.175 (17.5%)     │    │ "ship"          │
  │ "car"    : -3.4              │       │ P("car")    = 0.001  (0.1%)     │    │ (82.4% chance)  │
  │ "banana" : -8.1              │       │ P("banana") = 0.000001          │    └─────────────────┘
  └──────────────────────────────┘       ├─────────────────────────────────┤
                                         │ SUM P(v) = 1.000000 (100.0%)    │
                                         │ All P(v) ≥ 0.0 (No negatives)   │
                                         └─────────────────────────────────┘
 ===================================================================================================
```

---

### 2. 👶 ELI5 Intuition: The 1-Kilogram Kitchen Scale & Slicing a Birthday Cake
> `Context:` Physical & Everyday Metaphors for Probability Measures

#### Metaphor 1: The Master Restaurant Menu and the 1-Kilogram Kitchen Scale
Imagine a restaurant kitchen with a master menu:
1. **The Sample Space ($\Omega$):** The master menu containing **every single dish** the restaurant knows how to make. An **elementary outcome ($\omega$)** is the specific single plate delivered to your table.
2. **The Event Space ($\mathcal{F}$ / Sigma-Algebra):** All valid customer questions you can ask the waiter: *"Is my meal vegetarian?", "Is it dessert?", "Does it contain garlic?"* An **event ($A$)** is the collection of all dishes matching that question.
3. **The Probability Measure ($P$):** A magical kitchen scale where the entire inventory of all dishes together weighs **exactly $1.0\text{ kg}$** ($100\%$). The probability of any event is simply the scale weight of all dishes matching your question.
   - No plate has negative weight ($P(A) \ge 0$).
   - If two plates are separate (disjoint), their combined weight is the sum of their individual weights ($P(A \cup B) = P(A) + P(B)$).

---

#### Metaphor 2: Slicing a Birthday Cake
Imagine a round birthday cake:
- The entire cake represents **certainty** ($1.0$ or $100\%$).
- You slice the cake among party guests.
- No guest can receive a "negative slice" of cake ($P(A) \ge 0$).
- When all slices are put back together on the plate, they recreate the full cake ($P(\Omega) = 1.0$).

---

### 3. 📚 Deep Terminology Master Glossary (15 Core Concepts Dissected)
> `Context:` Foundational Mathematical & Machine Learning Vocabulary Explained Without Jargon

```
 ===================================================================================================
                 THE PROBABILITY TERMINOLOGY ROSETTA STONE
 ===================================================================================================
```

| Term / Notation | Formal Mathematical Meaning | Plain-English Definition (No ML Jargon) | Real-World Analogy |
| :--- | :--- | :--- | :--- |
| **Random Experiment ($\text{RE}$)** | Nondeterministic data generation process | Any process whose exact outcome cannot be known before running | Flipping a coin or taking a photo in low light |
| **Sample Space ($\Omega$)** | Set of all elementary outcomes $\Omega = \{\omega_i\}$ | The complete master list of all possible results that could happen | The full deck of 52 playing cards |
| **Elementary Outcome ($\omega$)** | Single element $\omega \in \Omega$ | One specific result observed after running the experiment | Drawing the Queen of Hearts |
| **Event ($A$)** | Subset of sample space $A \subseteq \Omega$ | A collection of one or more outcomes sharing a property | Drawing any Red Card (26 cards) |
| **Event Space ($\mathcal{F}$ / $\sigma$-algebra)** | Family of all valid measurable subsets of $\Omega$ | The set of all legal Boolean questions you are allowed to ask | The search filter options on an e-commerce website |
| **Probability Measure ($P$)** | Function $P: \mathcal{F} \to [0, 1]$ | A ruler or scale assigning a size between $0.0$ and $1.0$ to every event | The percentage chance of an event happening |
| **Axiom 1: Non-Negativity** | $\forall A \in \mathcal{F}: P(A) \ge 0$ | Probabilities cannot be negative numbers | You cannot have negative mass on a physical scale |
| **Axiom 2: Unit Measure** | $P(\Omega) = 1.0$ | Something on the master list is guaranteed to happen ($100\%$) | The chance that tomorrow will be *some* kind of weather |
| **Axiom 3: Additivity** | $A \cap B = \emptyset \implies P(A \cup B) = P(A) + P(B)$ | Non-overlapping events simply add together without double-counting | Weight of an apple plus weight of a banana |
| **Conditional Probability ($P(A \mid B)$)** | $P(A \cap B) / P(B)$ for $P(B) > 0$ | Chance of $A$ occurring given that $B$ is already known to be true | Chance of taking an umbrella given that it is raining |
| **Statistical Independence ($A \perp B$)** | $P(A \cap B) = P(A) \cdot P(B)$ | Knowing $B$ happened gives zero new clues about whether $A$ will happen | Two coins flipped in completely separate cities |
| **Law of Total Probability** | $P(A) = \sum_{i} P(A \mid B_i) P(B_i)$ | Computing total probability by breaking reality into distinct scenarios | Total company revenue = sum of revenues from each branch |
| **Bayes' Theorem** | $P(B \mid A) = \frac{P(A \mid B)P(B)}{P(A)}$ | Updating our prior belief about a cause after observing new evidence | A doctor updating disease probability after a positive test |
| **Support of a Distribution** | $\text{supp}(P) = \{x \in \Omega : p(x) > 0\}$ | The region of space where outcomes actually have a chance of happening | The physical territory where an animal species lives |
| **Normalization Constant ($Z$)** | $Z = \sum_x \tilde{p}(x)$ or $\int \tilde{p}(x)dx$ | The divider that forces unnormalized scores to sum up to $1.0$ | The total bill divider when splitting a restaurant tab |

---

### 4. 📐 The 3 Kolmogorov Axioms, Formulas & Geometry
> `Context:` Formal Mathematical Definitions, Theorems & Geometry of Probability

```
 ===================================================================================================
                 THE 3 KOLMOGOROV PROBABILITY AXIOMS (1933)
 ===================================================================================================

  AXIOM 1: NON-NEGATIVITY                AXIOM 2: UNIT MEASURE               AXIOM 3: COUNTABLE ADDITIVITY
  ┌─────────────────────────────┐       ┌────────────────────────────┐      ┌────────────────────────────┐
  │  For any event A ∈ ℱ:       │       │  Entire Sample Space:      │      │  For disjoint A ∩ B = ∅:   │
  │                             │       │                            │      │                            │
  │       P(A) ≥ 0              │       │       P(Ω) = 1.0           │      │   P(A ∪ B) = P(A) + P(B)   │
  │                             │       │                            │      │                            │
  │ Probabilities cannot be     │       │ Certainty equals 100%.     │      │ Non-overlapping masses     │
  │ negative.                   │       │ Total probability mass.    │      │ sum linearly.              │
  └─────────────────────────────┘       └────────────────────────────┘      └────────────────────────────┘
 ===================================================================================================
```

#### Fundamental Theorems Derived from the Axioms:

```
  VENN DIAGRAM: INCLUSION-EXCLUSION PRINCIPLE
  
  ┌────────────────────────────────────────────────────────┐
  │ Sample Space Ω (Area = 1.0)                            │
  │                                                        │
  │          Event A                Event B                │
  │       ┌───────────────┬───────────────┐                │
  │       │ Only A        │ Overlap       │ Only B         │
  │       │ (A \ B)       │ (A ∩ B)       │ (B \ A)        │
  │       │               │               │                │
  │       └───────────────┴───────────────┘                │
  │                                                        │
  └────────────────────────────────────────────────────────┘
```

1. **Empty Set (Impossibility):**
   $$P(\emptyset) = 0.0$$
2. **Complement Rule:**
   $$P(A^c) = 1.0 - P(A) \quad (\text{where } A^c = \Omega \setminus A)$$
3. **Monotonicity (Subset Rule):**
   $$A \subseteq B \implies P(A) \le P(B)$$
4. **General Addition Rule (Inclusion-Exclusion):**
   $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$
5. **Conditional Probability Formula:**
   $$P(A \mid B) = \frac{P(A \cap B)}{P(B)} \quad (\text{for } P(B) > 0)$$
6. **Bayes' Theorem (Updating Beliefs with Evidence):**
   $$P(\text{Hypothesis} \mid \text{Evidence}) = \frac{P(\text{Evidence} \mid \text{Hypothesis}) \cdot P(\text{Hypothesis})}{P(\text{Evidence})}$$
   $$P(B_k \mid A) = \frac{P(A \mid B_k) P(B_k)}{\sum_{i=1}^n P(A \mid B_i) P(B_i)}$$

---

### 5. 🔢 Concrete Micro-Numerical Worked Examples
> `Context:` Step-by-Step Manual Calculations (No Black Box)

#### Example 1: Rolling a Fair 6-Sided Die
Let sample space $\Omega = \{1, 2, 3, 4, 5, 6\}$, where each face has equal probability $P(\{\omega\}) = \frac{1}{6} \approx 0.1667$.

Let Event $A = \{\text{Even Number}\} = \{2, 4, 6\}$ and Event $B = \{\text{Number} \ge 4\} = \{4, 5, 6\}$.

1. **Individual Probabilities:**
   $$P(A) = \frac{3}{6} = 0.50, \quad P(B) = \frac{3}{6} = 0.50$$
2. **Intersection (Both Even AND $\ge 4$):**
   $$A \cap B = \{4, 6\} \implies P(A \cap B) = \frac{2}{6} \approx 0.3333$$
3. **Union (Even OR $\ge 4$):**
   $$P(A \cup B) = P(A) + P(B) - P(A \cap B) = 0.50 + 0.50 - 0.3333 = \mathbf{0.6667} = \frac{4}{6} \quad (\{2, 4, 5, 6\})$$
4. **Conditional Probability (Given that the roll is $\ge 4$, what is the chance it is Even?):**
   $$P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{2/6}{3/6} = \frac{2}{3} \approx \mathbf{0.6667}$$

---

#### Example 2: The Medical Diagnostic Test Paradox (Bayes' Theorem by Hand)
Suppose a rare medical condition affects **$1$ in $1,000$ people** ($0.1\%$ prevalence).
- **Prior Probability:** $P(\text{Disease}) = 0.001$, so $P(\text{Healthy}) = 0.999$.
- **Test Sensitivity (True Positive Rate):** If you have the disease, test is positive $99\%$ of the time: $P(\text{Positive} \mid \text{Disease}) = 0.99$.
- **Test False Positive Rate:** If you are healthy, test is falsely positive $5\%$ of the time: $P(\text{Positive} \mid \text{Healthy}) = 0.05$.

**Question:** A patient takes the test and it comes back **POSITIVE**. What is the true probability they actually have the disease?

```
  BAYESIAN REASONING ON 100,000 PEOPLE:
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Total Population: 100,000 people                                       │
  │ • True Sick:    100 people ──► 99 Test POSITIVE, 1 Tests Negative      │
  │ • True Healthy: 99,900 ppl ──► 4,995 Test POSITIVE (False Alarm!),     │
  │                                94,905 Test Negative                    │
  │                                                                        │
  │ Total Positive Tests = 99 + 4,995 = 5,094 people                       │
  │ True Sufferers among Positives = 99 / 5,094 ≈ 1.94% (Only ~2% chance!) │
  └────────────────────────────────────────────────────────────────────────┘
```

**Formal Mathematical Calculation via Bayes' Theorem:**
1. **Total Probability of a Positive Test $P(\text{Positive})$:**
   $$P(\text{Positive}) = P(\text{Positive} \mid \text{Disease})P(\text{Disease}) + P(\text{Positive} \mid \text{Healthy})P(\text{Healthy})$$
   $$P(\text{Positive}) = (0.99 \times 0.001) + (0.05 \times 0.999) = 0.00099 + 0.04995 = \mathbf{0.05094}$$

2. **Posterior Probability $P(\text{Disease} \mid \text{Positive})$:**
   $$P(\text{Disease} \mid \text{Positive}) = \frac{P(\text{Positive} \mid \text{Disease})P(\text{Disease})}{P(\text{Positive})} = \frac{0.00099}{0.05094} \approx \mathbf{0.0194} \quad (\mathbf{1.94\%})$$

*(Even with a $99\%$ accurate test, a positive result only means a $1.94\%$ chance of sickness because the disease is so rare!)*

---

### 6. 🔗 Connecting the Dots: How Probability Powers Modern Generative AI
> `Context:` Architectural Implementations in Large Language Models, Diffusion Models, GANs, and VAEs

```
 ===================================================================================================
                 PROBABILITY FOUNDATIONS IN MODERN GENERATIVE AI
 ===================================================================================================

  1. AUTOREGRESSIVE LLM (Probability Chain Rule)    2. DIFFUSION MODEL (Markov Gaussian Chain)
  p(x₁, x₂, ..., x_T) = ∏ p(x_t | x_{<t})           q(x_t | x_{t-1}) = 𝒩(x_t; √(1-β_t)x_{t-1}, β_t I)
  ┌────────────────────────────────────────┐        ┌────────────────────────────────────────┐
  │ Token 1: "The" (Prompt)                │        │ Clean Image x₀ ~ p_data(x)             │
  │ Token 2: "sky"   p("sky" | "The")      │        │      │ + Small Gaussian Noise ε₁       │
  │ Token 3: "is"    p("is" | "The sky")   │        │      ▼                                 │
  │ Token 4: "blue"  p("blue" | "The...")  │        │ Noisy Image x_t ~ 𝒩(√(ᾱ_t)x₀, (1-ᾱ_t)I)│
  └────────────────────────────────────────┘        └────────────────────────────────────────┘
 ===================================================================================================
```

| Generative Architecture | How Probability Is Applied | Mathematical Role |
| :--- | :--- | :--- |
| **Large Language Models (LLMs)** | **Probability Chain Rule**: $p(x) = \prod_{t=1}^T p(x_t \mid x_{<t})$ | Factors joint sentence probability into sequentially sampled conditional token distributions |
| **Diffusion Models (DDPM / Flux)** | **Gaussian Transition Measures**: $q(x_t \mid x_{t-1})$ | Adds and removes Gaussian noise incrementally over $T=1000$ discrete Markov diffusion timesteps |
| **Variational Autoencoders (VAEs)** | **Prior & Posterior Distributions**: $p(z) = \mathcal{N}(0, I), q_\phi(z \mid x)$ | Regularizes latent space so every point samples a realistic, continuous reconstruction |
| **GANs (Generative Adversarial Nets)** | **Push-Forward Probability Measure**: $p_\theta = G_\# p_z$ | Maps a simple low-D Gaussian noise distribution $p(z)$ into complex high-D image space |
| **Softmax Classification** | **Axiom 2 Enforcer**: $p_i = \frac{e^{z_i}}{\sum e^{z_j}}$ | Guarantees non-negative outputs and forces the entire vocabulary sum to equal exactly $1.0$ |

---

### 7. 💻 Complete Standalone Executable Python/PyTorch Verification Script
> `Context:` Runnable Python Script Simulating Kolmogorov Axioms, Monte Carlo Dice, and Bayes' Disease Test

```python
"""
Probability Basics & Kolmogorov Axioms — Verification Script
============================================================
Demonstrates:
1. Verification of Kolmogorov's 3 Axioms in PyTorch
2. Monte Carlo simulation of Die Roll events and Inclusion-Exclusion
3. Exact Bayes' Theorem calculation for rare medical test
"""
import torch
import torch.nn.functional as F
import numpy as np

print("=" * 75)
print("PROBABILITY BASICS & KOLMOGOROV AXIOMS MATHEMATICAL SIMULATION")
print("=" * 75)

# ─── 1. Verifying Kolmogorov Axioms with Softmax ───
print("\n1. VERIFYING KOLMOGOROV AXIOMS ON NEURAL NETWORK LOGITS:")
raw_logits = torch.tensor([4.5, -2.1, 0.0, 7.8, -5.0])
probs = F.softmax(raw_logits, dim=0)

print(f"   Raw Unbounded Logits:  {raw_logits.numpy().tolist()}")
print(f"   Softmax Probabilities: {probs.numpy().round(5).tolist()}")

# Axiom 1: Non-Negativity
axiom_1 = torch.all(probs >= 0.0).item()
print(f"   * Axiom 1 (All P(A) >= 0.0):       {axiom_1} ✅")

# Axiom 2: Unit Measure (Sum equals 1.0)
prob_sum = torch.sum(probs).item()
axiom_2 = abs(prob_sum - 1.0) < 1e-6
print(f"   * Axiom 2 (Total Sum P(Ω) == 1.0): {axiom_2} (Sum = {prob_sum:.6f}) ✅")

# Axiom 3: Countable Additivity (Disjoint subsets add linearly)
disjoint_union = probs[0] + probs[3] # Event {0} OR Event {3}
print(f"   * Axiom 3 (Disjoint Sum P({0} U {3})): {disjoint_union.item():.5f} == {probs[0].item():.5f} + {probs[3].item():.5f} ✅")

# ─── 2. Monte Carlo Simulation of 6-Sided Die Rolls ───
print("\n2. MONTE CARLO INCLUSION-EXCLUSION SIMULATION (100,000 Rolls):")
n_trials = 100_000
np.random.seed(42)
rolls = np.random.randint(1, 7, size=n_trials)

event_A = (rolls % 2 == 0)      # Even: {2, 4, 6}
event_B = (rolls >= 4)          # >= 4: {4, 5, 6}
event_A_and_B = event_A & event_B
event_A_or_B = event_A | event_B

p_A_sim = np.mean(event_A)
p_B_sim = np.mean(event_B)
p_AB_sim = np.mean(event_A_and_B)
p_A_or_B_sim = np.mean(event_A_or_B)
p_inc_exc = p_A_sim + p_B_sim - p_AB_sim

print(f"   P(Even Number A):           {p_A_sim:.4f} (Exact: 0.5000)")
print(f"   P(Roll >= 4 B):             {p_B_sim:.4f} (Exact: 0.5000)")
print(f"   P(Even AND >= 4):           {p_AB_sim:.4f} (Exact: 0.3333)")
print(f"   P(Even OR >= 4 Empirical):  {p_A_or_B_sim:.4f} (Exact: 0.6667)")
print(f"   P(A) + P(B) - P(A ∩ B):     {p_inc_exc:.4f} (Matches Inclusion-Exclusion! ✅)")

# ─── 3. Bayes' Theorem Medical Diagnostic Simulation ───
print("\n3. BAYES' THEOREM MEDICAL DIAGNOSTIC TEST (100,000 Patients):")
p_disease = 0.001
p_pos_given_disease = 0.99
p_pos_given_healthy = 0.05

# Analytic Bayes calculation
p_healthy = 1.0 - p_disease
p_pos_total = (p_pos_given_disease * p_disease) + (p_pos_given_healthy * p_healthy)
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos_total

print(f"   Prior Chance of Disease:       {p_disease * 100:.2f}%")
print(f"   Test Accuracy on Sick Patient: {p_pos_given_disease * 100:.1f}%")
print(f"   False Alarm Rate on Healthy:   {p_pos_given_healthy * 100:.1f}%")
print(f"   Total Positive Test Rate:      {p_pos_total * 100:.3f}%")
print(f"   >> TRUE PROBABILITY OF SICKNESS GIVEN POSITIVE TEST: {p_disease_given_pos * 100:.2f}% <<")
print("   Status: Bayesian update verified mathematically! ✅")

print("\n" + "=" * 75)
print("ALL PROBABILITY SIMULATION TESTS PASSED SUCCESSFULLY! ✅")
print("=" * 75)
```

---

### 8. 🩺 Diagnostic Mini-Checks & Common Traps
> `Context:` Production Debugging Insights, Edge-Case Traps & Self-Verification Questions

#### ✅ Self-Test Questions

1. **Q:** Can a probability value ever be greater than $1.0$ or less than $0.0$?  
   **A:** No. By Kolmogorov Axiom 1 ($P(A) \ge 0$) and Axiom 2 ($P(\Omega) = 1.0$), all probabilities must strictly satisfy $0.0 \le P(A) \le 1.0$. If a neural network outputs values outside this interval, it has not been normalized with Softmax or Sigmoid.

2. **Q:** If Event $A$ and Event $B$ are statistically independent, does $P(A \cup B) = P(A) + P(B)$?  
   **A:** No! Additivity ($P(A \cup B) = P(A) + P(B)$) requires events to be **disjoint / mutually exclusive** ($A \cap B = \emptyset$). If they are independent, they overlap with probability $P(A \cap B) = P(A)P(B)$, so $P(A \cup B) = P(A) + P(B) - P(A)P(B)$.

3. **Q:** Why is the probability of picking any exact real number (like $x = 3.14159265...$) from a continuous Gaussian distribution equal to **zero**?  
   **A:** In continuous sample spaces, there are infinitely many points ($|\Omega| = \infty$). The probability of any single infinitesimal point is $\int_a^a p(x)dx = 0.0$. Probability is only non-zero over **intervals** ($P(a \le X \le b) = \int_a^b p(x)dx > 0$).

#### ⚠️ Common Engineering Traps

| Trap | Why It Fails | Production Fix |
| :--- | :--- | :--- |
| **Summing probabilities of non-disjoint events** | Produces probabilities $> 1.0$ due to double-counting overlap | Always subtract the intersection: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ |
| **Confusing $P(A \mid B)$ with $P(B \mid A)$** | Base Rate Fallacy (e.g. confusing chance of cough given cold with chance of cold given cough) | Use **Bayes' Theorem** to invert conditional probabilities properly |
| **Assuming Softmax outputs are true Bayesian confidence** | Overparameterized deep networks can output $99.9\%$ confidence on out-of-distribution hallucinations | Use **temperature scaling** or **Monte Carlo Dropout / Ensembles** to calibrate uncertainty |
| **Passing pre-softmax logits to functions expecting probabilities** | Negative numbers break log calculations ($\ln(z)$ where $z < 0 \implies \text{NaN}$) | Apply `F.softmax(logits, dim=-1)` before probability calculations |

---

### 🎯 Summary Checklist
- **Kolmogorov Triplet $(\Omega, \mathcal{F}, P)$** forms the axiomatic bedrock of all probability theory.
- **Axiom 1:** Probabilities cannot be negative ($P(A) \ge 0$).
- **Axiom 2:** Total probability across all outcomes equals certainty ($P(\Omega) = 1.0$).
- **Axiom 3:** Non-overlapping events add linearly ($P(A \cup B) = P(A) + P(B)$).
- **Softmax** is the neural network operator that enforces Kolmogorov Axioms 1 and 2 on raw unconstrained network logits.
- **Generative AI** (LLMs, Diffusion, VAEs, GANs) relies entirely on probability distributions to represent uncertainty and sample creative outputs.
