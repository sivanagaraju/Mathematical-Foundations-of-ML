# Prerequisites — warm-up before Lec 11 (entropy)

> **Do this first** if “log of a probability,” “expectation,” or “surprisal” still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Builds on [Lec 10](../11-Lec10-Challenges-of-ML/PREREQUISITES.md) (recipe: model → distance → train).  
> Strong basics: every formula below is decoded with micro numbers.

---

## ⚡ 3-Minute Executive Fast-Track Card

```
              FROM SURPRISAL OF AN EVENT TO SHANNON ENTROPY
              
   [ Single Event A ] ──► Probability P(A) ──► Surprisal I(A) = -log_2 P(A)
                                                  (Rare = High Surprise)
                                                           │
   [ Entire Law P ]   ──► Outcome Probabilities {p_i}      │
            │                                              ▼
            └─► Expected Surprise: H(P) = E[-log_2 P(X)] = - ∑ p_i log_2 p_i
                                  (1 Fair Coin = 1.0 Bit)
                                  (Certain Event = 0.0 Bits)
```

### 3 Core Mental Shifts
1. **Surprisal is Inversely Related to Probability:** Highly certain events ($P=1$) carry zero surprise ($I=0$); near-impossible events ($P \to 0$) carry infinite surprise ($I \to \infty$).
2. **Entropy is a Single Law’s Internal Spread:** Entropy $H(P)$ evaluates the average uncertainty inside one distribution $P$. It is NOT a distance metric between two distributions (that is a divergence, like KL).
3. **Logarithms Enable Additivity:** Logs turn probability products into sums ($-\log(P(A)P(B)) = I(A) + I(B)$), making information from independent events purely additive.

### 3-Question Instant Readiness Gate
1. *If an event has probability $P(A) = 1.0$, what is its surprisal in bits?*  
   <details><summary>Reveal Answer</summary><b>0.0 bits.</b> $-\log_2(1.0) = 0$. Certain events communicate zero information.</details>
2. *If an outcome has $p(x) = 0$, what is its value in the entropy formula?*  
   <details><summary>Reveal Answer</summary><b>0.</b> By mathematical convention, $\lim_{p \to 0^+} p \log p = 0$, so $0 \log 0 := 0$.</details>
3. *Can Shannon entropy $H(p_\theta)$ serve as the loss function $d(p, p_\theta)$ in the ML recipe?*  
   <details><summary>Reveal Answer</summary><b>No.</b> Entropy measures the spread of a single distribution. The ML recipe requires a comparative divergence (like KL) that scores two distributions.</details>

---

```
  After this warm-up you can say:

  "Surprisal of an event A is I(A) = −log P(A): rare events surprise more."
  "I(Ω)=0 and I(empty) is infinite; independent events add surprisals."
  "Entropy of a discrete law is the average surprisal: H = −∑ p_i log p_i."
  "Use 0 log 0 := 0 when writing the sum (only sum over outcomes with p>0)."
  "Entropy is one law’s average surprise; a divergence scores two laws."
  "ML needs divergences for recipe step 2; entropy is the info-theory building block."
```

**Warm-up → lecture boxes**

```
  §1  Log of a probability (why −log)   ──► Topic 3
  §2  Events, Ω, empty set reload       ──► Topic 3
  §3  Independence → products of P      ──► Topic 3
  §4  Discrete PMF + expectation        ──► Topic 4
  §5  Surprisal → entropy (formula)     ──► Topics 3–5
  §6  Entropy vs divergence (not same)  ──► Topics 2, 5
  §7  Recipe step 2 needs d             ──► Topics 1–2
  §8  Log base (bits vs nats)           ──► Topics 3–4
```

---

## Math Terminology Rosetta Stone

Before diving into the foundational pillars, use this reference table to decode mathematical shorthand and notation into plain English, spoken phonetics, and software implementations.

| Symbol / Notation | Spoken English (Phonetics) | Mathematical Concept | Plain-English Intuition | Dedicated MathsTerm Link |
| :--- | :--- | :--- | :--- | :--- |
| $I(A) = -\log_2 P(A)$ | **INFO OF AY EQUALS NEG-uh-tiv LOG-TWO OF PEE OF AY** | Self-Information (Surprisal) | Bits of uncertainty or shock received upon learning event $A$ occurred | [Entropy & Cross-Entropy](../../MathsTerms/04-Information-Theory-and-Divergences/01-Entropy_CrossEntropy_CCE.md) |
| $H(P) = -\sum_{x} P(x) \log_2 P(x)$ | **AYCH OF PEE** | Shannon Entropy | Expected or average surprise across all outcomes; minimum bits needed to encode a draw | [Entropy & Cross-Entropy](../../MathsTerms/04-Information-Theory-and-Divergences/01-Entropy_CrossEntropy_CCE.md) |
| $\text{bits vs nats}$ | **BITS VERSUS NATS** | Information Units | Log base 2 measures information in bits; natural log ($\ln$) measures in nats | [Logarithms & Exponential Functions](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/00-Logarithms_and_Exponential_Functions.md) |
| $\lim_{p \to 0^+} p \log p = 0$ | **LIH-mit AZ PEE GOES TO ZERO FROM RIGHT** | Limiting Entropy Zero-Product | Impossible events carry zero total entropy contribution because they never happen | [Logarithms & Exponential Functions](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/00-Logarithms_and_Exponential_Functions.md) |
| $H(P) \le \log_2 |\mathcal{X}|$ | **AYCH LESS THAN LOG CARDINALITY** | Maximum Entropy Bound | Uniform distribution over $|\mathcal{X}|$ states has maximal uncertainty and highest entropy | [Entropy & Cross-Entropy](../../MathsTerms/04-Information-Theory-and-Divergences/01-Entropy_CrossEntropy_CCE.md) |

---

## 1. Why $-\log$ of a probability?

<a id="p1-neglog"></a>

### 👶 Physical Analogy & Intuition
Imagine reading breaking news headlines. If a headline reads "The sun rose this morning in the east" ($P = 1.0$), you gain zero new knowledge. If a headline reads "A massive meteor was just discovered on a collision course with Earth tomorrow" ($P = 0.000001$), your phone buzzes with shock. Information (surprisal) is inversely related to probability—the rarer the event, the greater the shock.

### 🔍 Plain-English Breakdown
The foundation of information theory rests on this definition of event surprisal:
$$I(A) = -\log P(A)$$

**Why logarithms?**
Because independent events multiply probabilities ($P(A \cap B) = P(A)P(B)$), their information should naturally add:
$$\log(ab) = \log a + \log b$$
Logs convert multiplicative probabilities into additive information scores.

**Why the minus sign?**
For any legitimate event, $P(A) \in (0, 1]$. In this range, $\log P(A) \le 0$. The negative sign flips this negative value to a positive number so information is non-negative ($I(A) \ge 0$).

| $P(A)$ | $-\log_2 P(A)$ (bits) |
|--------|------------------------|
| $1$ | $0$ |
| $1/2$ | $1$ |
| $1/4$ | $2$ |
| $1/8$ | $3$ |
| $\to 0$ | $\to +\infty$ |

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
- Coin toss landing Heads:
  $$P(\text{Heads}) = 0.50 \implies I(\text{Heads}) = -\log_2(0.50) = -(-1.0) = 1.00 \text{ bit}$$
- Fair 8-sided die showing face 1:
  $$P(\text{Face 1}) = \frac{1}{8} = 0.125 \implies I(\text{Face 1}) = -\log_2(0.125) = -(-3.0) = 3.00 \text{ bits}$$
- The 8-sided die outcome is 4 times rarer ($0.125$ vs $0.50$), producing $3.0 - 1.0 = 2.0$ additional bits of surprise.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

p_coin = 0.50
p_die8 = 0.125

i_coin = -np.log2(p_coin)
i_die8 = -np.log2(p_die8)

assert np.isclose(i_coin, 1.0), "Coin surprisal must equal 1 bit"
assert np.isclose(i_die8, 3.0), "8-sided die surprisal must equal 3 bits"
assert i_die8 > i_coin
print(f"Surprisal of coin: {i_coin:.1f} bit, 8-sided die: {i_die8:.1f} bits")
```

### 🩺 Diagnostic Mini-Check
If an event has probability $P(A) = 0.25$, how many bits of surprisal does it carry under base-2 logarithm?
<details><summary>Reveal Answer</summary>
$I(A) = -\log_2(0.25) = -\log_2(2^{-2}) = -(-2) = 2.0 \text{ bits}$.
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Shannon proved that any continuous information measure $I: (0, 1] \to [0, \infty)$ satisfying monotonicity ($p_1 < p_2 \implies I(p_1) > I(p_2)$) and additivity for independent events ($I(p_1 p_2) = I(p_1) + I(p_2)$) must uniquely take the form:
$$I(p) = -c \log_b(p) \quad (c > 0, b > 1)$$
Setting $c=1, b=2$ defines the standard bit (shannon) unit of information.
</details>

---

## 2. Events, $\Omega$, and empty set (reload)

<a id="p2-events"></a>

### 👶 Physical Analogy & Intuition
Imagine a weather forecast stating: "Tomorrow, either the sun will shine, or it will rain, or it will snow, or it will be cloudy" (the full sample space $\Omega$). Since this guarantees reality 100%, you learned nothing ($0$ surprise). If the forecast claimed "Tomorrow, time will flow backwards" (the impossible null event $\emptyset$), your surprise would be infinite.

### 🔍 Plain-English Breakdown
He validates the formula against the mathematical boundary conditions:

| Event | Probability | Surprisal should be… |
|-------|-------------|----------------------|
| Sample space $\Omega$ (“something happens”) | $P(\Omega)=1$ | **zero** information |
| Empty set $\emptyset$ (“impossible event”) | $P(\emptyset)=0$ | **infinite** information |
| Ordinary event $A$ | $0<P(A)<1$ | positive finite |

When $X$ is discrete, the event is singleton $A = \{X = x_i\}$, with probability $P(A) = p(x_i)$. The surprisal of outcome $x_i$ is $I(\{X = x_i\}) = -\log p(x_i)$.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
- Boundary 1 (Certainty):
  $$P(\Omega) = 1.0 \implies I(\Omega) = -\log_2(1.0) = -0.0 = 0.00 \text{ bits}$$
- Boundary 2 (Near-impossibility):
  $$p = 0.0001 = 10^{-4} \implies I(p) = -\log_{10}(10^{-4}) = -(-4.0) = 4.0 \text{ dits} = 4.0 \times 3.3219 \approx 13.29 \text{ bits}$$
- As $p \to 0^+$, the surprisal strictly diverges: $\lim_{p \to 0^+} -\log_2(p) = +\infty$.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

p_certain = 1.0
p_rare = 1e-4

i_certain = -np.log2(p_certain)
i_rare = -np.log2(p_rare)

assert np.isclose(i_certain, 0.0)
assert np.isclose(i_rare, 13.2877, atol=1e-3)
print(f"Surprisal of certainty: {i_certain:.1f} bits, Rare (p=1e-4): {i_rare:.2f} bits")
```

### 🩺 Diagnostic Mini-Check
If a casino game guarantees that you will receive at least \$0 on every spin ($P=1.0$), what is the surprisal value of that guarantee?
<details><summary>Reveal Answer</summary>
$0.0$ bits. A certain outcome yields zero surprise.
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $(\Omega, \mathcal{F}, P)$ be a probability space. The information function $I: \mathcal{F} \to [0, \infty]$ satisfies:
$$I(\Omega) = -\log P(\Omega) = -\log(1) = 0$$
$$\lim_{P(A) \to 0} I(A) = \lim_{p \to 0^+} -\log(p) = +\infty$$
making $I$ a continuous extended real-valued functional on the measure algebra.
</details>

---

## 3. Independence (for the additivity rule)

<a id="p3-independence"></a>

### 👶 Physical Analogy & Intuition
Imagine flipping a coin in New York and rolling a die in Tokyo. The coin result gives you 1 bit of surprise; the die gives you 2.58 bits of surprise. Because the two events have zero causal or physical link, your total surprise upon learning both results is the pure arithmetic sum: $1.0 + 2.58 = 3.58$ bits.

### 🔍 Plain-English Breakdown
When two events $A$ and $B$ are statistically independent, their joint probability factors:
$$P(A \cap B) = P(A)P(B)$$
Taking the negative logarithm on both sides:
$$-\log P(A \cap B) = -\log(P(A)P(B)) = -\log P(A) - \log P(B)$$
which yields the additive information rule:
$$I(A \cap B) = I(A) + I(B)$$

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Suppose two fair coins are tossed independently:
- Event $A$ (Coin 1 Heads): $P(A) = 0.50 \implies I(A) = -\log_2(0.50) = 1.0$ bit.
- Event $B$ (Coin 2 Heads): $P(B) = 0.50 \implies I(B) = -\log_2(0.50) = 1.0$ bit.
- Joint Event $A \cap B$ (Both Heads):
  $$P(A \cap B) = 0.50 \times 0.50 = 0.25$$
  $$I(A \cap B) = -\log_2(0.25) = -(-2.0) = 2.00 \text{ bits}$$
- Arithmetic sum: $I(A) + I(B) = 1.0 + 1.0 = 2.00$ bits. Additivity verified!

### 💻 Standalone Executable Python Verification
```python
import numpy as np

p_A = 0.50
p_B = 0.50
p_joint = p_A * p_B

I_A = -np.log2(p_A)
I_B = -np.log2(p_B)
I_joint = -np.log2(p_joint)

assert np.isclose(I_joint, I_A + I_B)
assert np.isclose(I_joint, 2.00)
print(f"I(A) + I(B) = {I_A:.1f} + {I_B:.1f} = {I_joint:.1f} bits (Pure Additivity)")
```

### 🩺 Diagnostic Mini-Check
If three independent events each have probability $P = 0.50$, what is the total surprisal of observing all three events simultaneously?
<details><summary>Reveal Answer</summary>
$1.0 + 1.0 + 1.0 = 3.0 \text{ bits}$ (since $-\log_2(0.5^3) = -\log_2(0.125) = 3$).
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $\mathcal{G}_1, \dots, \mathcal{G}_n$ be mutually independent $\sigma$-algebras on $(\Omega, \mathcal{F}, P)$. For any sequence of events $A_i \in \mathcal{G}_i$:
$$I\left(\bigcap_{i=1}^n A_i\right) = -\log P\left(\bigcap_{i=1}^n A_i\right) = -\log \prod_{i=1}^n P(A_i) = \sum_{i=1}^n -\log P(A_i) = \sum_{i=1}^n I(A_i)$$
</details>

---

## 4. Discrete PMF and expectation

<a id="p4-pmf-expectation"></a>

### 👶 Physical Analogy & Intuition
Think of a classroom where the teacher calculates the class average test score. You multiply each possible score (say 80, 90, 100) by the fraction of students who got that score, and add them up. Expectation is just a weighted average where the probabilities act as weights!

### 🔍 Plain-English Breakdown
Entropy begins in the discrete domain:
$$p(x_i)=P(X=x_i),\qquad \sum_i p(x_i)=1,\quad p(x_i)\ge 0$$

Unlike continuous density curves, **evaluating a PMF at a point produces an exact probability**.
*Continuous trap:* For a continuous RV, $P(X=x) = 0$ for every single point, so naive $-\log P(X=x)$ is infinite everywhere. That is why entropy is built first on discrete PMFs.

**Expectation of a function $f(X)$:**
$$\mathbb{E}[f(X)] = \sum_i p(x_i) f(x_i)$$
For Shannon entropy, we set $f(x) = -\log p(x)$.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let $X$ be a fair 6-sided die with outcomes $\{1, 2, 3, 4, 5, 6\}$ and $p(x_i) = \frac{1}{6}$:
- Expected face value ($f(x) = x$):
  $$\mathbb{E}[X] = \sum_{x=1}^6 x \cdot \frac{1}{6} = \frac{1 + 2 + 3 + 4 + 5 + 6}{6} = \frac{21}{6} = 3.50$$
- Expected constant function ($f(x) = 5.0$):
  $$\mathbb{E}[5.0] = \sum_{x=1}^6 5.0 \cdot \frac{1}{6} = 5.0 \times \frac{6}{6} = 5.00$$

### 💻 Standalone Executable Python Verification
```python
import numpy as np

faces = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.full(6, 1.0 / 6.0)

expected_val = np.sum(faces * probabilities)

assert np.isclose(expected_val, 3.50)
assert np.isclose(np.sum(probabilities), 1.0)
print(f"Expected face value of fair die: {expected_val:.2f}")
```

### 🩺 Diagnostic Mini-Check
Why can't we plug individual point values of a continuous probability density $p(x)$ into $-\log P(X=x)$?
<details><summary>Reveal Answer</summary>
Because for any continuous random variable, the exact point probability $P(X=x) = 0$. Taking $-\log(0)$ would yield undefined/infinite surprisal at every continuous point.
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $X: \Omega \to \mathcal{X}$ be a discrete random variable on countable alphabet $\mathcal{X}$. The expectation operator is the Lebesgue integral with respect to pushforward measure $P_X$:
$$\mathbb{E}[f(X)] = \int_{\mathcal{X}} f(x) \, dP_X(x) = \sum_{x \in \mathcal{X}} f(x) p(x)$$
Linearity of expectation holds unconditionally: $\mathbb{E}[a f(X) + b g(X)] = a\mathbb{E}[f(X)] + b\mathbb{E}[g(X)]$.
</details>

---

## 5. From surprisal of one event to entropy of a law

<a id="p5-entropy-formula"></a>

### 👶 Physical Analogy & Intuition
Imagine watching a coin flip tournament. If the coin is heavily biased and lands Heads 99.9% of the time, the games are boring and predictable—almost zero average suspense (low entropy). But if the coin is perfectly balanced 50/50, every toss is completely unpredictable—maximum suspense (high entropy). Entropy measures the average level of suspense or surprise per draw!

### 🔍 Plain-English Breakdown
We move from the surprisal of one outcome to the expected surprisal of the entire distribution:
$$\boxed{H(p) = \mathbb{E}_{X\sim p}\big[-\log p(X)\big] = -\sum_i p(x_i)\log p(x_i)}$$

| Piece | Meaning |
|-------|---------|
| $-\log p(x_i)$ | surprisal if outcome $x_i$ occurs |
| $p(x_i)$ | probability weight of how often $x_i$ occurs |
| $-\sum$ | expected / average surprisal across all possible draws |

**Convention for $p=0$:**
If an outcome has $p=0$, the term $0 \log 0 := 0$ because $\lim_{p \to 0^+} p \log p = 0$.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
1. **Fair Coin:** $p = [0.50, 0.50]$ (base 2):
   $$H = -\left[0.5 \log_2(0.5) + 0.5 \log_2(0.5)\right] = -\left[0.5(-1) + 0.5(-1)\right] = 1.00 \text{ bit}$$
2. **Certain Outcome:** $p = [1.0, 0.0]$:
   $$H = -[1.0 \log_2(1.0) + 0] = -[0 + 0] = 0.00 \text{ bits}$$
3. **Three Outcomes:** $p = [0.50, 0.25, 0.25]$:
   $$H = -[0.5 \log_2(0.5) + 0.25 \log_2(0.25) + 0.25 \log_2(0.25)]$$
   $$= -[0.5(-1) + 0.25(-2) + 0.25(-2)] = -[-0.5 - 0.5 - 0.5] = 1.50 \text{ bits}$$

### 💻 Standalone Executable Python Verification
```python
import numpy as np

p_fair = np.array([0.5, 0.5])
p_certain = np.array([1.0, 0.0])
p_three = np.array([0.5, 0.25, 0.25])

h_fair = -np.sum(p_fair * np.log2(p_fair))
# Avoid log(0) using boolean indexing
h_certain = -np.sum(p_certain[p_certain > 0] * np.log2(p_certain[p_certain > 0]))
h_three = -np.sum(p_three * np.log2(p_three))

assert np.isclose(h_fair, 1.00)
assert np.isclose(h_certain, 0.00)
assert np.isclose(h_three, 1.50)
print(f"Fair: {h_fair:.1f} bit, Certain: {h_certain:.1f} bits, Three: {h_three:.2f} bits")
```

### 🩺 Diagnostic Mini-Check
What is the theoretical maximum Shannon entropy for a discrete distribution over $K=4$ possible outcomes?
<details><summary>Reveal Answer</summary>
The uniform distribution $p = [1/4, 1/4, 1/4, 1/4]$ achieves maximum entropy: $H_{\max} = \log_2(4) = 2.0 \text{ bits}$.
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

For any probability vector $\mathbf{p} \in \Delta^{K-1}$ on a finite alphabet of cardinality $K$:
$$0 \le H(P) \le \log_2 K$$
Lower bound $H(P) = 0$ holds if and only if $P$ is a Dirac mass $P = \delta_k$. Upper bound $H(P) = \log_2 K$ holds if and only if $P$ is the uniform distribution $p_k = 1/K$ for all $k$.
</details>

---

## 6. Entropy vs divergence (do not mix)

<a id="p6-entropy-vs-div"></a>

### 👶 Physical Analogy & Intuition
Imagine checking the fuel tank of a single car ($H(P)$—a single internal status measurement of that car) versus measuring the distance on the highway between two separate cars ($D_{\text{KL}}(P \parallel Q)$—a comparative relational measurement between two cars). Entropy evaluates **one** law; divergence evaluates **two** laws.

### 🔍 Plain-English Breakdown
Do not confuse these two concepts in the ML recipe:

| Object | Inputs | Meaning |
|--------|--------|---------|
| **Entropy** $H(p)$ | **One** distribution $p$ | average internal surprisal inside $p$ |
| **Divergence** $d(p,q)$ | **Two** distributions $p$ and $q$ | statistical discrepancy between $p$ and $q$ |

ML recipe Step 2 needs a score between **two** distributions: true data $p$ and candidate model $p_\theta$. Entropy alone is not Step 2—it is the foundational brick used to build divergences like Kullback-Leibler (KL) divergence.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let $p = [0.80, 0.20]$ and $q = [0.50, 0.50]$:
- Single-law Entropy of $p$ (internal uncertainty):
  $$H(p) = -[0.80 \ln(0.80) + 0.20 \ln(0.20)] = -[0.8(-0.2231) + 0.2(-1.6094)] = 0.5004 \text{ nats}$$
- Pairwise Divergence $D_{\text{KL}}(p \parallel q)$ (gap between $p$ and $q$):
  $$D_{\text{KL}}(p \parallel q) = 0.80 \ln\left(\frac{0.80}{0.50}\right) + 0.20 \ln\left(\frac{0.20}{0.50}\right) = 0.80(0.4700) + 0.20(-0.9163) = 0.1927 \text{ nats}$$
- Notice that $H(p) = 0.5004 \neq D_{\text{KL}}(p \parallel q) = 0.1927$.

### 💻 Standalone Executable Python Verification
```python
import numpy as np

p = np.array([0.80, 0.20])
q = np.array([0.50, 0.50])

entropy_p = -np.sum(p * np.log(p))
kl_div = np.sum(p * np.log(p / q))

assert np.isclose(entropy_p, 0.5004, atol=1e-3)
assert np.isclose(kl_div, 0.1927, atol=1e-3)
assert not np.isclose(entropy_p, kl_div)
print(f"Single Law Entropy: {entropy_p:.4f} nats, Pairwise Divergence: {kl_div:.4f} nats")
```

### 🩺 Diagnostic Mini-Check
If a student suggests minimizing $H(p_\theta)$ to fit a model to data, what will happen?
<details><summary>Reveal Answer</summary>
The model will collapse to a deterministic spike ($H(p_\theta) \to 0$) completely independent of the dataset! Fitting requires minimizing the divergence $D(p_{\text{data}} \parallel p_\theta)$, not the model's internal entropy.
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Cross-entropy decomposes into the sum of self-entropy and relative entropy (KL divergence):
$$H(P, Q) = -\sum_x P(x) \log Q(x) = H(P) + D_{\mathrm{KL}}(P \parallel Q)$$
Since $H(P)$ depends purely on the data-generating distribution and is independent of model parameters $\theta$, minimizing cross-entropy with respect to $\theta$ is mathematically identical to minimizing $D_{\mathrm{KL}}(P \parallel Q_\theta)$.
</details>

---

## 7. Why this lecture exists in the ML recipe

<a id="p7-recipe-link"></a>

### 👶 Physical Analogy & Intuition
Before you can invent a speedometer to measure how fast a car is moving relative to the speed limit, you must first define what a mile and an hour mean. Shannon entropy defines the fundamental currency of uncertainty (bits/nats). Once this currency exists, we can measure the information-distance between our model and reality.

### 🔍 Plain-English Breakdown
Connecting back to Lecture 10's 3-step master recipe:
```
  (1) choose model p_θ
  (2) distance d(true, model)   ← NEED INFORMATION THEORY
  (3) θ* = argmin d
```
Step 2 is impossible without a principled way to score distance between probability distributions. Entropy is step 1 of that toolkit.

### Linear model types (lecture clarification)
In the affine expression $w_1^\top x + w_2$ on input vector $x \in \mathbb{R}^d$:

| Symbol | Correct Type |
|--------|--------------|
| $w_1$ | vector in $\mathbb{R}^d$ |
| $w_1^\top x$ | **scalar** |
| $w_2$ | **scalar** bias (not a second vector) |

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
Let $x = [2.0, 3.0] \in \mathbb{R}^2$, $w_1 = [0.4, -0.2] \in \mathbb{R}^2$, and scalar bias $w_2 = 1.5 \in \mathbb{R}$:
- Dot product:
  $$w_1^T x = (0.4 \times 2.0) + (-0.2 \times 3.0) = 0.80 - 0.60 = 0.20 \text{ (scalar)}$$
- Affine sum:
  $$w_1^T x + w_2 = 0.20 + 1.50 = 1.70 \text{ (scalar)}$$

### 💻 Standalone Executable Python Verification
```python
import numpy as np

x = np.array([2.0, 3.0])
w1 = np.array([0.4, -0.2])
w2 = 1.5  # scalar

dot_product = np.dot(w1, x)
affine_output = dot_product + w2

assert np.isclose(dot_product, 0.20)
assert np.isclose(affine_output, 1.70)
assert affine_output.shape == ()
print(f"Affine scalar output: {affine_output:.2f}")
```

### 🩺 Diagnostic Mini-Check
In an affine model $f(x) = w_1^T x + w_2$ where $x \in \mathbb{R}^{512}$, what are the shapes of $w_1$ and $w_2$?
<details><summary>Reveal Answer</summary>
$w_1 \in \mathbb{R}^{512}$ (a 512-dimensional vector), and $w_2 \in \mathbb{R}$ (a 1-dimensional scalar bias).
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Under the Information Geometry framework (Amari, 2000), parametric model families form smooth statistical manifolds $\mathcal{M} = \{p_\theta : \theta \in \Theta\}$. The Fisher Information Metric defines a Riemannian metric $g_{ij}(\theta) = \mathbb{E}\left[\frac{\partial \log p}{\partial \theta_i}\frac{\partial \log p}{\partial \theta_j}\right]$, locally approximating the second-order Taylor expansion of the KL divergence.
</details>

---

## 8. Log base: bits vs nats

<a id="p8-log-base"></a>

### 👶 Physical Analogy & Intuition
Measuring length in inches versus centimeters. If a table is 36 inches long, it is also 91.44 centimeters long. Changing the measurement unit does not change the physical length of the table. Likewise, computing entropy in bits ($\log_2$) versus nats ($\ln$) is simply a unit conversion!

### 🔍 Plain-English Breakdown
Professors often write $\log$ without specifying the base:

| Base | Unit of entropy / surprisal |
|------|-----------------------------|
| $\log_2$ | **bits** (standard in computer science / coding) |
| $\ln=\log_e$ | **nats** (standard in calculus / deep learning papers) |
| $\log_{10}$ | dits / bans (rare in modern ML) |

Changing base multiplies all surprisal and entropy values by a positive constant:
$$\log_b u = \frac{\ln u}{\ln b}$$
Because the constant is positive, all rankings, optimal parameter locations, and conclusions remain identical.

### 🔢 Concrete Micro-Numbers (Hand-Calculated)
For an event with $P(A) = 0.50$:
- In base 2 (bits):
  $$I_2(A) = -\log_2(0.50) = 1.0000 \text{ bit}$$
- In natural log (nats):
  $$I_e(A) = -\ln(0.50) = \ln(2) \approx 0.6931 \text{ nats}$$
- Scale factor:
  $$\frac{I_2(A)}{I_e(A)} = \frac{1.0000}{0.6931} = \frac{1}{\ln 2} \approx 1.4427$$
  $$1 \text{ nat} \approx 1.4427 \text{ bits}$$

### 💻 Standalone Executable Python Verification
```python
import numpy as np

p = 0.50
i_bits = -np.log2(p)
i_nats = -np.log(p)

ratio = i_bits / i_nats

assert np.isclose(i_bits, 1.0000)
assert np.isclose(i_nats, np.log(2))
assert np.isclose(ratio, 1.0 / np.log(2))
print(f"Surprisal: {i_bits:.2f} bits = {i_nats:.4f} nats (Ratio: {ratio:.4f})")
```

### 🩺 Diagnostic Mini-Check
If you switch from PyTorch's natural log cross-entropy to a base-2 logarithm, does the optimal parameter vector $\theta^* = \arg\min_\theta L(\theta)$ move to a different location in parameter space?
<details><summary>Reveal Answer</summary>
<b>No.</b> Changing log base multiplies the objective by constant $\frac{1}{\ln 2} > 0$. Multiplying an objective by a positive constant preserves the exact location of all local and global minima.
</details>

<details><summary><b>📐 Deep Formal Mathematical Formulation & Guarantees (Click to expand)</b></summary>

Let $H_b(P)$ denote entropy computed in base $b > 1$. By the logarithmic change of base formula:
$$H_b(P) = \frac{1}{\ln b} H_e(P)$$
The gradient with respect to parameter $\theta$ satisfies $\nabla_\theta H_b(P_\theta) = \frac{1}{\ln b} \nabla_\theta H_e(P_\theta)$. Consequently, the stationary points $\{\theta : \nabla H = \mathbf{0}\}$ and Hessian eigenspaces are strictly invariant to base selection.
</details>

---

### Paper check

1. Compute $I(A)$ if $P(A)=1/8$ with $\log_2$.  
2. Why is $I(\Omega)=0$? Why is discrete first (not continuous points)?  
3. Independent fair coins both heads: check $I(A\cap B)=I(A)+I(B)$.  
4. Write $H$ for a fair coin, a sure outcome, and $p=(1/2,1/4,1/4)$.  
5. What do you do with a $p=0$ term in $-\sum p\log p$?  
6. Is entropy a distance between two distributions? What is a divergence?  
7. Which recipe step needs distributional divergences?  
8. Correct types: $w_1$ and $w_2$ in $w_1^\top x+w_2$.  
9. In one sentence: what does “average surprisal per draw” mean for $H(p)$?

---

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html).  
Prior: [Lec 10 Challenges](../11-Lec10-Challenges-of-ML/NOTES.md).

---

## 🗝️ Mathematical Foundations & MathsTerms Bridge

> [!TIP]
> **Foundational Knowledge Base:** This module directly relies upon formal mathematical constructs systematically defined and verified in our central [`MathsTerms`](../../MathsTerms) repository. For visual dependency graphs and multi-track learning roadmaps, consult the [Grand Unified Concept Map](../../MathsTerms/CONCEPT_MAP.md).

| Mathematical Concept | Dedicated Guide | Role & Significance in This Lecture |
| :--- | :--- | :--- |
| **Entropy, Cross-Entropy & Categorical Cross-Entropy** | [Entropy, Cross-Entropy & Categorical Cross-Entropy](../../MathsTerms/04-Information-Theory-and-Divergences/01-Entropy_CrossEntropy_CCE.md) | Surprisal $-\log p(x)$, Shannon entropy $H(p)$, and expected uncertainty |
| **Kullback-Leibler (KL) Divergence** | [Kullback-Leibler (KL) Divergence](../../MathsTerms/04-Information-Theory-and-Divergences/02-KL_Divergence.md) | Transitioning from self-entropy to relative divergence between two distributions |
| **Loss Functions** | [Loss Functions](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/08-Loss_Functions.md) | Cross-entropy loss as the negative log-likelihood of categorical targets |

---

## 🌉 Curriculum & Sibling Course Prerequisite Bridges

| Prerequisite Concept | Source Module / Resource | Target Application in Lecture 11 | Verification Check |
| :--- | :--- | :--- | :--- |
| **IID Assumption & Joint Probability** | [Lecture 07: IID Assumption](../../Mathematical-foundation-ml/08-Lec07-IID-Assumption/NOTES.md) | Logarithm turns probability products into sums: $\log \prod p_i = \sum \log p_i$ | Show why independence enforces additivity of information |
| **Continuous Densities vs PMFs** | [Lecture 09: Density Function](../../Mathematical-foundation-ml/10-Lec09-Density-Function/NOTES.md) | Understanding why continuous differential entropy can be negative while discrete entropy $\ge 0$ | Evaluate $h(X) < 0$ when density height exceeds 1 |
| **The Three-Step Recipe** | [Lecture 10: Challenges of ML](../../Mathematical-foundation-ml/11-Lec10-Challenges-of-ML/NOTES.md) | Formulating Step 2 (distance metric) using information-theoretic divergence | Identify the role of entropy in defining KL divergence |
| **Kullback-Leibler Divergence** | [Lecture 12: KL-Divergence](../../Mathematical-foundation-ml/13-Lec12-KL-Divergence/NOTES.md) | Relative entropy $D_{KL}(P \parallel Q) = \sum P(x) \log(P(x)/Q(x)) = H(P, Q) - H(P)$ | Contrast self-entropy with relative divergence |
| **Minimization of KL** | [Lecture 13: Minimization of KL](../../Mathematical-foundation-ml/14-Lec13-Minimization-of-KL/NOTES.md) | Showing that minimizing KL divergence to the empirical data distribution reduces to minimizing cross-entropy | Connect entropy to empirical log-likelihood maximization |

---
