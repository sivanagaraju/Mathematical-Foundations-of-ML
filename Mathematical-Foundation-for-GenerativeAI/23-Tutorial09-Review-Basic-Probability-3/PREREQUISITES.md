# Prerequisites & Foundational Warm-Up: Review of Basic Probability 3

> **Target Audience:** Engineers, data scientists, and STEM students returning to advanced mathematics after 10–15 years.  
> **Course:** NPTEL / IISc — *Mathematical Foundations of Generative AI* (Tutorial 9).  
> **Previous Modules:** [Tutorial 7](../21-Tutorial07-Review-Basic-Probability-1/NOTES.md) (Discrete RVs, PMFs & CDFs) and [Tutorial 8](../22-Tutorial08-Review-Basic-Probability-2/NOTES.md) (Continuous RVs, PDFs, LOTUS & Inequalities).  
> **Next Step:** After completing this warm-up, open [NOTES.md](./NOTES.md) at the **Executive Summary**.

---

```
  ╔═══════════════════════════════════════════════════════════════════════════════════════╗
  ║                        CORE MENTAL MODELS YOU WILL MASTER TODAY                       ║
  ╠═══════════════════════════════════════════════════════════════════════════════════════╣
  ║ 1. "A random vector (X,Y) is two different measurement stickers on ONE experiment Ω."║
  ║ 2. "The Joint CDF F(x,y) measures probability accumulated South-West of point (x,y)." ║
  ║ 3. "A 2D rectangle window uses four corners: F(x2,y2) - F(x2,y1) - F(x1,y2) + F(x1,y1)."║
  ║ 4. "Marginals are shadow projections: summing a row or integrating out the other axis."║
  ║ 5. "A joint uniquely gives marginals; but marginals alone CANNOT reconstruct a joint."║
  ║ 6. "Conditionals are knife-edge slices re-normalized: p(x|y) = p(x,y) / p_Y(y)."       ║
  ║ 7. "Independence means EVERY 2D window factors: p(x,y) = p_X(x) * p_Y(y) for all x,y."║
  ║ 8. "Transforming random vectors Y = g(X) scales joint density by the Jacobian |det J|."║
  ╚═══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 🗺️ Roadmap: Warm-Up Pillars to Lecture Topics

```
  ┌────────────────────────────────────────────────────────┐       ┌────────────────────────────────────────────────────────┐
  │ PREREQUISITE FOUNDATIONAL PILLAR                       │       │ LECTURE TOPIC MAPPING IN NOTES.md                      │
  ├────────────────────────────────────────────────────────┤       ├────────────────────────────────────────────────────────┤
  │ §1. Same Experiment (Ω), Two Random Variables (X, Y)   │ ────► │ Topic 1 (Pair & Joint CDF) & Topic 2 (CDF Properties)  │
  │ §2. Joint CDFs & The 2D Rectangle Window Formula       │ ────► │ Topic 1 (Cylindrical Sets) & Topic 2 (CDF Axioms)      │
  │ §3. Joint PMF as 2D Contingency Cookie Trays           │ ────► │ Topic 3 (Joint PMF & Two Dice (Max, Sum))              │
  │ §4. Joint PDF as 3D Terrain Height over 2D Regions     │ ────► │ Topic 4 (Joint PDF on a Triangle: p=2 on 0<x<y<1)      │
  │ §5. Marginals as Shadow Projections (Peeling One Axis) │ ────► │ Topic 5 (Marginals: Unique Forward, Underdetermined)  │
  │ §6. Conditional Distributions: Knife Slices & Band Lim │ ────► │ Topic 6 (Discrete Conditional) & Topic 7 (Cont Bayes)  │
  │ §7. Statistical Independence as Cartesian Factorization│ ────► │ Topic 9 (Independence & Vector Random Variables)       │
  │ §8. IID Vectors, Transformations & The Jacobian |det J│ │ ────► │ Topic 8 (Mixed GMM) & Topic 10 (IID & Jacobian Map)   │
  └────────────────────────────────────────────────────────┘       └────────────────────────────────────────────────────────┘
```

---

## 💎 Math Terminology Rosetta Stone

If you haven't manipulated multivariate distributions, double integrals, or Jacobian matrices in years, this table is your instant plain-English decoder.

| Symbol | Formal Math Name | How to Pronounce It | Plain-English Translation | Everyday Physical Analogy | Dedicated MathsTerm Guide |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $(X, Y): \Omega \to \mathbb{R}^2$ | Random Vector / Pair of RVs | *"Pair X comma Y"* | Two simultaneous numerical measurements recorded from the exact same physical event. | Reading both Temperature ($X$) and Humidity ($Y$) from the same weather station balloon ($\omega$). | [Joint, Marginal & Conditional Dist](../../../MathsTerms/Joint_Marginal_Conditional_Dist.md) |
| $F_{XY}(x, y)$ | Joint Cumulative Distribution Function | *"Joint CDF of X and Y"* | $P(X \le x, Y \le y)$: Total probability mass located South-West of $(x, y)$. | Measuring how much tablecloth is pinned down to the bottom-left of coordinate $(x, y)$. | [Joint, Marginal & Conditional Dist](../../../MathsTerms/Joint_Marginal_Conditional_Dist.md) |
| $p_{XY}(x, y)$ | Joint PDF / Joint PMF | *"Joint density / mass of X, Y"* | Height of probability density (or mass of discrete pile) at 2D coordinate $(x, y)$. | 3D terrain elevation on a topographic map; or mass of cookies in tray cell $(x, y)$. | [Joint, Marginal & Conditional Dist](../../../MathsTerms/Joint_Marginal_Conditional_Dist.md) |
| $p_X(x) = \int p_{XY}(x, y)\,dy$ | Marginal Distribution | *"Marginal density of X"* | The probability distribution of $X$ alone, obtained by integrating (or summing) out all possible values of $Y$. | Casting a 2D shadow of a 3D object onto the X-axis wall. | [Joint, Marginal & Conditional Dist](../../../MathsTerms/Joint_Marginal_Conditional_Dist.md) |
| $p(x \mid y) = \frac{p(x, y)}{p_Y(y)}$ | Conditional Distribution | *"p of x given y"* | The probability distribution of $X$ after someone reveals the exact observed value $Y = y$. | Taking a knife slice across a loaf of bread at coordinate $y$ and reweighing the slice to 100%. | [Joint, Marginal & Conditional Dist](../../../MathsTerms/Joint_Marginal_Conditional_Dist.md) |
| $X \perp\!\!\!\perp Y$ | Statistical Independence | *"X is independent of Y"* | Learning $Y$ gives zero new information about $X$: $p_{XY}(x, y) = p_X(x) p_Y(y)$ for all $(x, y)$. | Two completely unplugged, unlinked electronic dice rolled in different rooms. | [Probability Basics & Axioms](../../../MathsTerms/Probability_Basics_and_Axioms.md) |
| $\text{IID}$ | Independent & Identically Distributed | *"I-I-D"* | A collection of random variables that are mutually independent and all follow the exact same probability law. | Taking 10,000 separate snapshots using the exact same camera sensor. | [Probability Basics & Axioms](../../../MathsTerms/Probability_Basics_and_Axioms.md) |
| $\mathbf{J} = \frac{\partial (x_1, \dots, x_n)}{\partial (y_1, \dots, y_n)}$ | Jacobian Matrix of the Inverse Map | *"Jacobian matrix J"* | The matrix of all first-order partial derivatives describing local spatial deformation. | The local stretch/compression factor of a rubber sheet when pulled in multiple directions. | [Derivatives, Gradients & Jacobians](../../../MathsTerms/Derivatives_Gradients_and_Jacobians.md) |
| $\lvert \det \mathbf{J} \rvert$ | Absolute Jacobian Determinant | *"Absolute determinant of J"* | The local volume expansion or contraction factor when mapping coordinates $X \to Y$. | How many times larger or smaller a square becomes after twisting and stretching it. | [Derivatives, Gradients & Jacobians](../../../MathsTerms/Derivatives_Gradients_and_Jacobians.md) |
| $\text{GMM}$ | Gaussian Mixture Model | *"G-M-M"* | A probability density constructed by mixing several different Gaussian bell curves with discrete weights. | A choir where singers have different vocal ranges (soprano, tenor, bass), creating a multi-peaked sound. | [Common Probability Distributions](../../../MathsTerms/Common_Probability_Distributions.md) |

---

## Pillar 1: Same Experiment ($\Omega$), Two Random Variables ($X, Y$)

<a id="p1-same"></a>

### 1. 👶 ELI5 Intuition
Imagine you take a high-resolution photograph of an outdoor concert:
- The single photograph is **one underlying experimental outcome** $\omega \in \Omega$.
- From this *single photograph*, you extract two different metrics:
  - Metric 1 ($X$): The overall brightness level of the image (e.g., $X(\omega) = 142.5$ lux).
  - Metric 2 ($Y$): The count of human faces detected (e.g., $Y(\omega) = 12$ people).
- You did **not** take one photo on Monday for brightness and a different photo on Tuesday for faces. **Both numbers come from the exact same physical event $\omega$!**
- Together, $(X, Y)$ forms a **pair** (a 2D vector) mapping the sample space $\Omega \to \mathbb{R}^2$.

```
  Sample Space Ω (Single Physical Outcome ω)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  Measurement X(ω)        Measurement Y(ω)
  (e.g., Brightness)      (e.g., Face Count)
         │                       │
         └───────────┬───────────┘
                     ▼
             Pair (X, Y) ∈ ℝ²
```

---

### 2. 🔍 Plain-English Breakdown
1. **The Single-Universe Prerequisite:**
   - Before you can discuss joint probabilities, correlations, or conditionals, random variables $X$ and $Y$ **must be defined on the exact same probability space** $(\Omega, \mathcal{F}, P)$.
   - If $X$ is the stock price of Apple on Monday and $Y$ is the temperature in Tokyo in 1985 from an unrelated experiment, they do not form a joint probability pair unless modeled on a combined product space.
2. **Vector Random Variable Mapping:**
   - Mathematically, a pair of random variables is a vector-valued function:
     $$\mathbf{Z}: \Omega \to \mathbb{R}^2, \quad \text{where } \mathbf{Z}(\omega) = \bigl(X(\omega), Y(\omega)\bigr)$$
3. **Simultaneous Events:**
   - An event involving both variables, such as $\{X \le 3, Y \le 5\}$, is the **set intersection** of outcomes where both conditions hold:
     $$\{\omega \in \Omega : X(\omega) \le 3\} \cap \{\omega \in \Omega : Y(\omega) \le 5\}$$

---

### 3. 📐 Formal Mathematics

#### Definition: Random Vector of Dimension 2
Let $(\Omega, \mathcal{F}, P)$ be a probability space. A 2D random vector is a Borel-measurable mapping $\mathbf{X} = (X, Y): \Omega \to \mathbb{R}^2$ such that for every 2D Borel set $B \in \mathcal{B}(\mathbb{R}^2)$:
$$\mathbf{X}^{-1}(B) = \{\omega \in \Omega : (X(\omega), Y(\omega)) \in B\} \in \mathcal{F}$$

---

### 4. 🔢 Concrete Numerical Micro-Example
Roll two fair 6-sided dice simultaneously: $\Omega = \{(w_1, w_2) : w_1, w_2 \in \{1, \dots, 6\}\}$ (36 elementary outcomes).
Let $X(w_1, w_2) = \max(w_1, w_2)$ and $Y(w_1, w_2) = w_1 + w_2$:
- Suppose the roll outcome is $\omega = (2, 5)$.
- Then $X(2, 5) = \max(2, 5) = 5$.
- And $Y(2, 5) = 2 + 5 = 7$.
- The pair evaluates to $(X, Y) = (5, 7)$ for this single outcome $\omega$.

---

### 5. 💻 Runnable Python / NumPy Snippet

```python
import numpy as np

# Rolling two fair dice on ONE experiment Ω
rng = np.random.default_rng(42)
N = 100_000

# Generate 100,000 pairs of dice rolls (omega_1, omega_2)
die1 = rng.integers(1, 7, size=N)
die2 = rng.integers(1, 7, size=N)

# Two measurements on the same outcome:
X_max = np.maximum(die1, die2)
Y_sum = die1 + die2

print(f"First 5 simulated outcomes (die1, die2):")
for i in range(5):
    print(f"  omega = ({die1[i]}, {die2[i]}) -> (X=max={X_max[i]}, Y=sum={Y_sum[i]})")
```

---

### 6. 🧠 Diagnostic Mini-Check
1. **Question:** Can two random variables form a joint distribution if they are measured from two completely unrelated experiments with no shared sample space?  
   *Answer:* No. A joint pair $(X, Y)$ requires both mappings to originate from the same underlying probability space $(\Omega, \mathcal{F}, P)$.
2. **Question:** If $\omega = (4, 4)$ in the two-dice experiment, what are the values of $X = \max$ and $Y = \text{sum}$?  
   *Answer:* $X = 4$ and $Y = 8$.
3. **Question:** What is the mathematical meaning of the joint event $\{X \le x, Y \le y\}$?  
   *Answer:* The set intersection $\{\omega \in \Omega : X(\omega) \le x\} \cap \{\omega \in \Omega : Y(\omega) \le y\}$.

---

## Pillar 2: Joint CDFs & The 2D Rectangle Window Formula

<a id="p2-window"></a>

### 1. 👶 ELI5 Intuition
Imagine pinning down a large rectangular tablecloth onto a flat table:
- The function $F(x, y)$ measures the **total weight of cloth** lying strictly to the **South-West** (bottom-left) of the coordinate pin $(x, y)$.
- If you want to find the weight of cloth inside a specific rectangular window $[x_1, x_2] \times [y_1, y_2]$, you cannot just evaluate $F$ at one corner!
- You take the big rectangle at the top-right corner $(x_2, y_2)$, subtract the two long side strips at $(x_2, y_1)$ and $(x_1, y_2)$, and notice you subtracted the bottom-left corner $(x_1, y_1)$ **twice**!
- So you must **add back the small corner $+F(x_1, y_1)$** to get the exact weight of the window.

```
  The 2D Inclusion-Exclusion Rectangle Formula:
  
        y2 ┌──────────────────* (x2, y2)  [+ F(x2, y2)]
           │                  │
           │   Target Window  │
           │     Region R     │
        y1 *──────────────────* (x2, y1)  [- F(x2, y1)]
         (x1, y1)           (x2, y1)
         [+ F(x1, y1)]      [- F(x1, y2)]
          x1                 x2
  
  P((X,Y) ∈ Window) = F(x2, y2) - F(x2, y1) - F(x1, y2) + F(x1, y1)
```

---

### 2. 🔍 Plain-English Breakdown
1. **Definition of Joint CDF:**
   $$F_{XY}(x, y) \triangleq P(X \le x, Y \le y)$$
2. **The 4 Fundamental Axioms of a 2D Joint CDF:**
   - **Null Boundaries:** If *either* coordinate goes to $-\infty$, the accumulated probability is zero:
     $$\lim_{x \to -\infty} F(x, y) = 0, \quad \lim_{y \to -\infty} F(x, y) = 0$$
   - **Total Plane Normalization:** As *both* coordinates go to $+\infty$, the whole universe is covered:
     $$\lim_{x \to \infty, y \to \infty} F(x, y) = 1.0$$
   - **Monotonicity:** $F(x, y)$ is non-decreasing in $x$ (for fixed $y$) and non-decreasing in $y$ (for fixed $x$).
   - **Right-Continuity:** $F(x, y)$ is right-continuous in both variables ($F(x^+, y^+) = F(x, y)$).
3. **The 2D Rectangle Probability Formula:**
   For any window $x_1 < x_2$ and $y_1 < y_2$:
   $$P(x_1 < X \le x_2, \, y_1 < Y \le y_2) = F(x_2, y_2) - F(x_2, y_1) - F(x_1, y_2) + F(x_1, y_1) \ge 0$$

---

### 3. 📐 Formal Mathematics

#### Theorem: Rectangle Measure via 2D Cumulative Distribution Function
Let $R = (x_1, x_2] \times (y_1, y_2]$. We decompose the quarter-plane $(-\infty, x_2] \times (-\infty, y_2]$ into disjoint regions:
$$(-\infty, x_2] \times (-\infty, y_2] = R \cup \bigl((-\infty, x_1] \times (-\infty, y_2]\bigr) \cup \bigl((-\infty, x_2] \times (-\infty, y_1]\bigr)$$
Applying the principle of inclusion-exclusion for probability measures:
$$P(R) = F_{XY}(x_2, y_2) - F_{XY}(x_1, y_2) - F_{XY}(x_2, y_1) + F_{XY}(x_1, y_1)$$
*(Note: A valid 2D CDF must guarantee that $P(R) \ge 0$ for all possible rectangles $R$).*

---

### 4. 🔢 Concrete Numerical Micro-Example
Suppose a joint CDF is given by $F_{XY}(x, y) = x \cdot y$ for $(x, y) \in [0, 1] \times [0, 1]$:
Let's compute the probability in the window $[0.2, 0.6] \times [0.3, 0.8]$:
1. $F(x_2, y_2) = F(0.6, 0.8) = 0.6 \times 0.8 = 0.48$
2. $F(x_2, y_1) = F(0.6, 0.3) = 0.6 \times 0.3 = 0.18$
3. $F(x_1, y_2) = F(0.2, 0.8) = 0.2 \times 0.8 = 0.16$
4. $F(x_1, y_1) = F(0.2, 0.3) = 0.2 \times 0.3 = 0.06$
5. Calculate Rectangle Probability:
   $$P(0.2 < X \le 0.6, \, 0.3 < Y \le 0.8) = 0.48 - 0.18 - 0.16 + 0.06 = 0.20 = 20\%$$
   *(Direct Area Check: Width $\times$ Height $= (0.6 - 0.2) \times (0.8 - 0.3) = 0.4 \times 0.5 = 0.20 \checkmark$)*

---

### 5. 💻 Runnable Python / NumPy Snippet

```python
# Verifying the 2D Rectangle Formula
def F_joint(x, y):
    # F(x, y) = x * y on [0, 1] x [0, 1]
    x_c = np.clip(x, 0.0, 1.0)
    y_c = np.clip(y, 0.0, 1.0)
    return x_c * y_c

x1, x2 = 0.2, 0.6
y1, y2 = 0.3, 0.8

rect_prob = F_joint(x2, y2) - F_joint(x2, y1) - F_joint(x1, y2) + F_joint(x1, y1)
print(f"Calculated 2D Rectangle Probability: {rect_prob:.4f} (Exact: 0.2000)")
assert np.isclose(rect_prob, 0.20)
```

---

### 6. 🧠 Diagnostic Mini-Check
1. **Question:** What is $F_{XY}(x, -\infty)$ for any real $x$?  
   *Answer:* Exactly $0.0$.
2. **Question:** Why do we add $+F(x_1, y_1)$ at the end of the 2D rectangle formula?  
   *Answer:* Because subtracting the two infinite strips $-F(x_2, y_1)$ and $-F(x_1, y_2)$ subtracts the bottom-left overlap region twice; adding it back corrects the count.
3. **Question:** What is $F_{XY}(+\infty, +\infty)$?  
   *Answer:* Exactly $1.0$ (100% total probability).

---

## Pillar 3: Joint PMF as 2D Contingency Cookie Trays

<a id="p3-table"></a>

### 1. 👶 ELI5 Intuition
Imagine a baker's tray divided into a grid of 6 rows and 11 columns (like an ice cube tray):
- You have 36 total cookie crumbs (total probability $1.0$).
- When you roll two dice, you look up the max $X$ (row) and the sum $Y$ (column), and drop a crumb into that cell.
- If two different rolls produce the same cell (like $(2,5)$ and $(5,2)$ both giving $\max=5, \text{sum}=7$), that cell gets **2 crumbs** ($2/36$).
- If only one roll produces that cell (like $(3,3)$ giving $\max=3, \text{sum}=6$), that cell gets **1 crumb** ($1/36$).
- Impossible combinations (like $\max=2, \text{sum}=9$) remain **completely empty** ($0/36$).

```
  2D Joint PMF Table (Dice Max vs Sum):
  
  X = Max ^
        6 │  .  .  .  .  .  1/36 2/36 2/36 2/36 2/36 1/36   (Row Total = 11/36)
        5 │  .  .  .  .  1/36 2/36 2/36 2/36 1/36  .    .    (Row Total = 9/36)
        4 │  .  .  .  1/36 2/36 2/36 1/36  .    .    .    .  (Row Total = 7/36)
        3 │  .  .  1/36 2/36 1/36  .    .    .    .    .    . (Row Total = 5/36)
        2 │  .  1/36 1/36  .    .    .    .    .    .    .    . (Row Total = 3/36)
        1 │ 1/36  .    .    .    .    .    .    .    .    .    . (Row Total = 1/36)
        ──┴───────────────────────────────────────────────► Y = Sum
            2   3    4    5    6    7    8    9   10   11   12
```

---

### 2. 🔍 Plain-English Breakdown
1. **Definition of Joint PMF:**
   $$p_{XY}(x_i, y_j) \triangleq P(X = x_i, Y = y_j)$$
2. **The Two Discrete Axioms:**
   - $0 \le p_{XY}(x_i, y_j) \le 1.0$ for all pairs $(x_i, y_j)$.
   - The double sum over all rows and columns equals $1.0$:
     $$\sum_{i} \sum_{j} p_{XY}(x_i, y_j) = 1.0$$
3. **The Diagonal Symmetry ($n = 2m$):**
   - In the $(\max, \text{sum})$ dice experiment, outcome cell $(m, n)$ can only be formed by dice rolls $(m, n-m)$ or $(n-m, m)$.
   - When $n = 2m$ (e.g. $\max=3, \text{sum}=6$), both dice rolled $3$. There is only **1 outcome** $(3, 3) \implies \text{mass} = 1/36$.
   - When $n \ne 2m$, there are **2 distinct outcomes** $\implies \text{mass} = 2/36$.

---

### 3. 📐 Formal Mathematics

#### Double Summation Total Probability
Let $S_X = \{x_1, x_2, \dots\}$ and $S_Y = \{y_1, y_2, \dots\}$ be the discrete supports of $X$ and $Y$.
$$\sum_{x_i \in S_X} \sum_{y_j \in S_Y} p_{XY}(x_i, y_j) = P\left( \bigcup_{i, j} \{X = x_i, Y = y_j\} \right) = P(\Omega) = 1.0$$

---

### 4. 🔢 Concrete Numerical Micro-Example
For two fair dice with $X = \max, Y = \text{sum}$:
1. What is $p_{XY}(3, 6)$?
   $n = 6 = 2(3) = 2m \implies$ Only outcome is $(3, 3) \implies p(3, 6) = 1/36 \approx 0.0278$.
2. What is $p_{XY}(5, 7)$?
   $n = 7 \ne 2(5) \implies$ Outcomes are $(5, 2)$ and $(2, 5) \implies p(5, 7) = 2/36 \approx 0.0556$.
3. What is $p_{XY}(2, 8)$?
   If $\max=2$, max possible sum is $2+2=4 < 8 \implies p(2, 8) = 0.0$.

---

### 5. 💻 Runnable Python / NumPy Snippet

```python
# Constructing the exact 2D Joint PMF Table in NumPy
joint_pmf = np.zeros((7, 13)) # 1-indexed for max (1..6) and sum (2..12)

for d1 in range(1, 7):
    for d2 in range(1, 7):
        m = max(d1, d2)
        s = d1 + d2
        joint_pmf[m, s] += 1.0 / 36.0

print(f"p(X=3, Y=6) (Single roll (3,3)): {joint_pmf[3, 6]:.4f} (Exact: 1/36 = {1/36:.4f})")
print(f"p(X=5, Y=7) (Two rolls (5,2),(2,5)): {joint_pmf[5, 7]:.4f} (Exact: 2/36 = {2/36:.4f})")
print(f"Total PMF Mass Summed:          {np.sum(joint_pmf):.4f} (Must equal 1.0)")
assert np.isclose(np.sum(joint_pmf), 1.0)
```

---

### 6. 🧠 Diagnostic Mini-Check
1. **Question:** In the two-dice joint PMF of $X = \max, Y = \text{sum}$, why is $p(4, 8) = 1/36$ while $p(4, 7) = 2/36$?  
   *Answer:* A sum of 8 with max 4 can only happen from $(4, 4)$ (1 outcome), whereas a sum of 7 with max 4 happens from $(4, 3)$ and $(3, 4)$ (2 outcomes).
2. **Question:** Must the number of rows in a joint PMF table equal the number of columns?  
   *Answer:* No! $X$ and $Y$ can have completely different support cardinalities (e.g., 6 rows vs 11 columns).
3. **Question:** What is the value of $\sum_x \sum_y p_{XY}(x, y)$ for any legal discrete joint distribution?  
   *Answer:* Exactly $1.0$.

---

## Pillar 4: Joint PDF as 3D Terrain Height over 2D Regions

<a id="p4-jam"></a>

### 1. 👶 ELI5 Intuition
Imagine spreading exactly 1 cup of cake frosting across a triangular cookie sheet:
- The cookie sheet is the **2D support region** (e.g., the triangle $0 < x < y < 1$).
- The **Joint PDF $p(x, y)$ is the height or thickness** of the frosting at coordinate $(x, y)$.
- Because the triangle has a small base area ($1/2$ square units), the frosting must stand **2 units high** to make the total volume equal to $1.0$ ($0.5 \times 2 = 1.0$).
- If someone asks: *"What is the probability that $(X, Y)$ lands in a sub-region $A$?"*, you simply measure the **volume of frosting** sitting on top of shape $A$!

```
  Joint PDF: Volume under 3D Surface over 2D Support Region
  
  y ^
  1 ┼        /|
    │       / |     Support Region: Triangle 0 < x < y < 1 (Base Area = 0.5)
    │      /  |     Uniform Height p(x,y) = 2.0
    │     /   |     Total Volume = Base Area × Height = 0.5 × 2.0 = 1.0
    │    /    |
  0 ┴───┴─────┴───► x
    0         1
```

---

### 2. 🔍 Plain-English Breakdown
1. **Definition of Joint PDF:**
   - A function $p_{XY}: \mathbb{R}^2 \to \mathbb{R}$ such that the joint CDF is the double integral:
     $$F_{XY}(x, y) = \int_{-\infty}^y \int_{-\infty}^x p_{XY}(u, v)\,du\,dv$$
2. **The 2 Joint PDF Axioms:**
   - **Non-negativity:** $p_{XY}(x, y) \ge 0$ for all $(x, y) \in \mathbb{R}^2$.
   - **Unit Total Volume Normalization:**
     $$\int_{-\infty}^\infty \int_{-\infty}^\infty p_{XY}(x, y)\,dx\,dy = 1.0$$
3. **Probability as 2D Region Volume:**
   - For any 2D geometric event region $A \subseteq \mathbb{R}^2$:
     $$P\bigl((X, Y) \in A\bigr) = \iint_A p_{XY}(x, y)\,dx\,dy$$
4. **The Drawing Trap:** A 2D plot of the triangle $0 < x < y < 1$ is a drawing of the **support region in the $(x, y)$ plane**, NOT the 3D graph of $p(x, y)$ (which rises upward out of the page).

---

### 3. 📐 Formal Mathematics

#### Double Integration over Non-Rectangular Domains
To integrate a function over the triangle $0 < x < y < 1$, we set integration limits:
$$\iint_{\text{Triangle}} p(x, y)\,dx\,dy = \int_{y=0}^1 \left( \int_{x=0}^y 2 \, dx \right) dy = \int_{0}^1 2y \, dy = \left[ y^2 \right]_0^1 = 1.0 \quad \blacksquare$$

---

### 4. 🔢 Concrete Numerical Micro-Example
Let $p_{XY}(x, y) = 2$ on $0 < x < y < 1$. Compute $P(Y > X + 0.5)$:
1. The sub-region $A = \{(x, y) : 0 < x < y < 1 \text{ and } y > x + 0.5\}$.
2. This defines an upper triangular sliver where $y$ goes from $0.5$ to $1.0$, and $x$ goes from $0$ to $y - 0.5$:
   $$P(Y > X + 0.5) = \int_{y=0.5}^1 \left( \int_{x=0}^{y-0.5} 2 \, dx \right) dy = \int_{0.5}^1 2(y - 0.5) \, dy = 2 \left[ \frac{y^2}{2} - 0.5y \right]_{0.5}^1$$
   $$= 2 \left[ (0.5 - 0.5) - (0.125 - 0.25) \right] = 2 \left[ 0 - (-0.125) \right] = 2 \times 0.125 = 0.25 = 25\%$$

---

### 5. 💻 Runnable Python / NumPy Snippet

```python
from scipy import integrate

# Numerical Double Integration of the Triangle Joint PDF
def joint_pdf_triangle(x, y):
    if 0 < x < y < 1:
        return 2.0
    return 0.0

# 1. Total Volume Integral
total_vol, _ = integrate.dblquad(
    lambda x, y: 2.0,
    0.0, 1.0,          # y limits: 0 to 1
    lambda y: 0.0,     # x lower limit: 0
    lambda y: y        # x upper limit: y
)

# 2. Sub-region P(Y > X + 0.5)
sub_prob, _ = integrate.dblquad(
    lambda x, y: 2.0,
    0.5, 1.0,          # y limits: 0.5 to 1.0
    lambda y: 0.0,     # x lower limit: 0
    lambda y: y - 0.5  # x upper limit: y - 0.5
)

print(f"Total Triangle Volume:        {total_vol:.4f} (Must equal 1.0)")
print(f"P(Y > X + 0.5) Sub-region:    {sub_prob:.4f} (Theory: 0.2500)")
assert np.isclose(total_vol, 1.0)
assert np.isclose(sub_prob, 0.25)
```

---

### 6. 🧠 Diagnostic Mini-Check
1. **Question:** Can a joint PDF evaluate to $p_{XY}(0.2, 0.5) = 2.0$?  
   *Answer:* Yes! A joint PDF represents 3D density height; total volume must equal $1.0$, but height can exceed $1.0$.
2. **Question:** For the uniform triangle density $p(x, y) = 2$ on $0 < x < y < 1$, what is $P(X = Y)$?  
   *Answer:* $0.0$. The diagonal line $x = y$ has zero 2D area, so volume under it is zero.
3. **Question:** What is $P((X, Y) \in A)$ for any region $A$ in terms of joint density $p$?  
   *Answer:* $\iint_A p_{XY}(x, y)\,dx\,dy$.

---

## Pillar 5: Marginal Distributions as Shadow Projections

<a id="p5-margin"></a>

### 1. 👶 ELI5 Intuition
Imagine holding a complex 3D clay sculpture above a table with an overhead light shining down:
- The 2D shadow cast on the table below shows the distribution of coordinate $X$, completely ignoring height $Y$.
- **Marginalization is Shadow-Casting:** To find the probability distribution of $X$ alone, you **collapse or integrate out** the entire $Y$-axis.
- **The One-Way Street Rule:**
  - If you have the full 3D clay sculpture (the Joint Distribution), you can **uniquely and easily calculate its shadow** (the Marginal Distributions).
  - But if someone *only gives you the flat 2D shadow*, you **CANNOT uniquely reconstruct the 3D sculpture**! (Infinitely many different 3D shapes can cast the exact same 2D shadow).

```
  Joint Distribution (Full 2D Landscape)
              │
              │  Collapse / Integrate out unwanted variable
              ▼
  Marginal p_X(x) = ∫ p(x,y) dy     Marginal p_Y(y) = ∫ p(x,y) dx
  
  ONE-WAY DIRECTION: Joint ──► Unique Marginals (Always!)
                     Marginals ──► Unique Joint (FALSE! Underdetermined)
```

---

### 2. 🔍 Plain-English Breakdown
1. **Definition of Marginals:**
   - **Discrete:** Sum the rows or columns of the table:
     $$p_X(x_i) = \sum_j p_{XY}(x_i, y_j), \quad p_Y(y_j) = \sum_i p_{XY}(x_i, y_j)$$
   - **Continuous:** Integrate over the entire domain of the other variable:
     $$p_X(x) = \int_{-\infty}^\infty p_{XY}(x, y)\,dy, \quad p_Y(y) = \int_{-\infty}^\infty p_{XY}(x, y)\,dx$$
2. **Why They Are Called "Marginals":** Historically, statisticians summed rows and columns of probability tables and wrote the totals in the physical **margins** of the paper.
3. **The Conversions Trap:** Having two marginals $p_X(x)$ and $p_Y(y)$ does **not** give you the joint distribution $p_{XY}(x, y)$ unless you are explicitly given that $X$ and $Y$ are independent.

---

### 3. 📐 Formal Mathematics

#### Derivation of Marginal PDF from Joint CDF
$$F_X(x) = P(X \le x) = P(X \le x, Y < \infty) = F_{XY}(x, \infty) = \int_{-\infty}^x \left( \int_{-\infty}^\infty p_{XY}(u, y)\,dy \right) du$$
Differentiating with respect to $x$ via Leibniz's rule:
$$p_X(x) = \frac{d}{dx} F_X(x) = \int_{-\infty}^\infty p_{XY}(x, y)\,dy \quad \blacksquare$$

---

### 4. 🔢 Concrete Numerical Micro-Example
On the triangle $p_{XY}(x, y) = 2$ for $0 < x < y < 1$:
1. Derive Marginal $p_X(x)$ for a fixed $x \in (0, 1)$ ($y$ ranges from $x$ to $1$):
   $$p_X(x) = \int_x^1 2 \, dy = 2 [y]_x^1 = 2(1 - x) \quad (0 < x < 1)$$
   Check: $\int_0^1 2(1 - x)\,dx = 2 \left[ x - \frac{x^2}{2} \right]_0^1 = 2(0.5) = 1.0 \checkmark$
2. Derive Marginal $p_Y(y)$ for a fixed $y \in (0, 1)$ ($x$ ranges from $0$ to $y$):
   $$p_Y(y) = \int_0^y 2 \, dx = 2 [x]_0^y = 2y \quad (0 < y < 1)$$
   Check: $\int_0^1 2y\,dy = [y^2]_0^1 = 1.0 \checkmark$

---

### 5. 💻 Runnable Python / NumPy Snippet

```python
# Verifying Marginal Densities via Numerical Integration
x_val = 0.3
# Analytical p_X(0.3) = 2*(1 - 0.3) = 1.40
p_x_num, _ = integrate.quad(lambda y: 2.0, x_val, 1.0)

y_val = 0.8
# Analytical p_Y(0.8) = 2*(0.8) = 1.60
p_y_num, _ = integrate.quad(lambda x: 2.0, 0.0, y_val)

print(f"Marginal p_X(0.3): {p_x_num:.4f} (Exact: {2*(1-x_val):.4f})")
print(f"Marginal p_Y(0.8): {p_y_num:.4f} (Exact: {2*y_val:.4f})")
assert np.isclose(p_x_num, 1.40)
assert np.isclose(p_y_num, 1.60)
```

---

### 6. 🧠 Diagnostic Mini-Check
1. **Question:** How do you compute $p_X(x)$ from a joint PDF $p_{XY}(x, y)$?  
   *Answer:* Integrate out $y$: $p_X(x) = \int_{-\infty}^\infty p_{XY}(x, y)\,dy$.
2. **Question:** If you are given marginals $p_X(x) = \mathcal{N}(0, 1)$ and $p_Y(y) = \mathcal{N}(0, 1)$, is the joint distribution uniquely determined?  
   *Answer:* No! Infinitely many bivariate distributions share standard normal marginals with varying correlation structures.
3. **Question:** What is $F_X(x)$ in terms of the joint CDF $F_{XY}$?  
   *Answer:* $F_X(x) = F_{XY}(x, +\infty)$.

---

## Pillar 6: Conditional Distributions: Knife Slices & The Band Limit

<a id="p6-slice"></a>

### 1. 👶 ELI5 Intuition
Imagine cutting a loaf of raisin bread with a sharp knife at position $y$:
- You pull out that single thin slice of bread.
- The raisins on that slice represent the distribution of $X$ **given that $Y = y$**.
- Because the slice is thin, its total weight is small. To turn it into a valid probability distribution, you must **re-scale (normalize)** the raisin count by the total thickness of the slice so the area under the slice equals **100% (1.0)**!
- **The Knife-Edge Paradox:** In continuous probability, the probability of hitting an exact knife line $P(Y = y) = 0$. So we define the slice as a thin band of width $\Delta$ and take the mathematical limit as $\Delta \to 0$.

```
  Continuous Conditioning: Slicing the 2D Joint Density at Y = y
  
  y ^
  1 ┼        /|
    │       / |
  y ┼──────*──*── Knife Slice at Y = y (Length = y)
    │     /   |
  0 ┴────┴────┴───► x
    0    x    1
  
  Re-normalized slice: p(x|y) = p(x,y) / p_Y(y) = 2 / (2y) = 1/y  -> Unif(0, y)!
```

---

### 2. 🔍 Plain-English Breakdown
1. **Definition of Conditional Distribution:**
   - **Discrete:**
     $$p(x_i \mid y_j) = \frac{p_{XY}(x_i, y_j)}{p_Y(y_j)} \quad (\text{provided } p_Y(y_j) > 0)$$
   - **Continuous:**
     $$p(x \mid y) = \frac{p_{XY}(x, y)}{p_Y(y)} \quad (\text{provided } p_Y(y) > 0)$$
2. **Infinitesimal Band Definition:**
   $$F(x \mid y) = \lim_{\Delta \to 0^+} \frac{P(X \le x, \, y \le Y \le y + \Delta)}{P(y \le Y \le y + \Delta)}$$
3. **Each Slice is a Legitimate 1D Probability Distribution:**
   - For every fixed $y$, $\int_{-\infty}^\infty p(x \mid y)\,dx = \int_{-\infty}^\infty \frac{p(x, y)}{p_Y(y)}\,dx = \frac{p_Y(y)}{p_Y(y)} = 1.0$.

---

### 3. 📐 Formal Mathematics

#### Continuous Bayes' Rule
Using the conditional product decomposition $p_{XY}(x, y) = p(x \mid y) p_Y(y) = p(y \mid x) p_X(x)$:
$$p(y \mid x) = \frac{p(x \mid y) \, p_Y(y)}{p_X(x)} = \frac{p(x \mid y) \, p_Y(y)}{\int_{-\infty}^\infty p(x \mid y') \, p_Y(y') \, dy'}$$

---

### 4. 🔢 Concrete Numerical Micro-Example
On the triangle $p_{XY}(x, y) = 2$ on $0 < x < y < 1$, where $p_Y(y) = 2y$:
1. Compute the conditional density $p(x \mid y)$ for a known $Y = y \in (0, 1)$:
   $$p(x \mid y) = \frac{p_{XY}(x, y)}{p_Y(y)} = \frac{2}{2y} = \frac{1}{y} \quad \text{for } x \in (0, y)$$
   *This proves that given $Y = y$, $X$ is distributed uniformly on the interval $[0, y]$: $X \mid Y = y \sim \text{Unif}[0, y]$!*
2. If $Y = 0.8$, what is $P(X \le 0.4 \mid Y = 0.8)$?
   $$P(X \le 0.4 \mid Y = 0.8) = \int_0^{0.4} \frac{1}{0.8} \, dx = \frac{0.4}{0.8} = 0.50 = 50\%$$

---

### 5. 💻 Runnable Python / NumPy Snippet

```python
# Simulating Conditional Uniformity on the Triangle
rng = np.random.default_rng(42)
N = 500_000

# Sample uniformly in unit square, keep points in triangle 0 < x < y < 1
u1 = rng.uniform(0, 1, size=N)
u2 = rng.uniform(0, 1, size=N)
x_pts = np.minimum(u1, u2)
y_pts = np.maximum(u1, u2)

# Condition on Y being close to 0.8 (thin band [0.79, 0.81])
band_mask = (y_pts >= 0.79) & (y_pts <= 0.81)
x_given_y = x_pts[band_mask]

# Empirical probability P(X <= 0.4 | Y ≈ 0.8)
prob_cond = np.mean(x_given_y <= 0.4)
print(f"Empirical P(X <= 0.4 | Y ≈ 0.8): {prob_cond:.4f} (Theory: 0.4/0.8 = 0.5000)")
assert np.isclose(prob_cond, 0.50, atol=0.02)
```

---

### 6. 🧠 Diagnostic Mini-Check
1. **Question:** What is the formula for conditional density $p(x \mid y)$?  
   *Answer:* $p(x \mid y) = \frac{p_{XY}(x, y)}{p_Y(y)}$ when $p_Y(y) > 0$.
2. **Question:** On the triangle $0 < x < y < 1$, what is the distribution of $X \mid Y = y$?  
   *Answer:* Uniform on $[0, y]$ ($\text{Unif}[0, y]$).
3. **Question:** What is $\int_{-\infty}^\infty p(x \mid y)\,dx$ for any fixed $y$?  
   *Answer:* Exactly $1.0$.

---

## Pillar 7: Statistical Independence as Cartesian Factorization

<a id="p7-prod"></a>

### 1. 👶 ELI5 Intuition
Imagine two completely separate light switches in two different houses across town:
- Flipping Switch 1 in House A has zero physical effect on Switch 2 in House B.
- **Statistical Independence:** Knowing the exact state of $Y$ gives you **zero new clues** about $X$.
- The probability that both events happen simultaneously is simply the **product of their individual probabilities**:
  $$P(\text{Switch 1 ON and Switch 2 ON}) = P(\text{Switch 1 ON}) \times P(\text{Switch 2 ON})$$
- **The "All Windows" Rule:** Independence requires this multiplication to work for **EVERY possible region in the 2D plane**, not just one lucky test point!

```
  Statistical Independence: The 2D Landscape Factors into 1D Components
  
  Joint Density p(x, y)  =  p_X(x)  ×  p_Y(y)   (For ALL x, y)
  Joint CDF F(x, y)      =  F_X(x)  ×  F_Y(y)   (For ALL x, y)
  Conditional p(x | y)   =  p_X(x)              (Learning y changes nothing!)
```

---

### 2. 🔍 Plain-English Breakdown
1. **Definition of Independence:** Random variables $X$ and $Y$ are independent ($X \perp\!\!\!\perp Y$) if and only if for all Borel subsets $B_1, B_2 \subseteq \mathbb{R}$:
   $$P(X \in B_1, Y \in B_2) = P(X \in B_1) \cdot P(Y \in B_2)$$
2. **Equivalent Factorization Criteria:**
   - **Joint CDF:** $F_{XY}(x, y) = F_X(x) \cdot F_Y(y)$ for all $(x, y) \in \mathbb{R}^2$.
   - **Joint PMF / PDF:** $p_{XY}(x, y) = p_X(x) \cdot p_Y(y)$ for all $(x, y) \in \mathbb{R}^2$.
   - **Conditionals:** $p(x \mid y) = p_X(x)$ and $p(y \mid x) = p_Y(y)$.
3. **The Rare Exception:** Independence is the **only situation where marginal distributions uniquely determine the joint distribution**!

---

### 3. 📐 Formal Mathematics

#### Theorem: Product of Expectations under Independence
If $X$ and $Y$ are independent random variables with finite expectations:
$$\mathbb{E}[X Y] = \mathbb{E}[X] \cdot \mathbb{E}[Y]$$
*Proof (Continuous Case):*
$$\mathbb{E}[XY] = \int_{-\infty}^\infty \int_{-\infty}^\infty x y \, p_{XY}(x, y)\,dx\,dy = \int_{-\infty}^\infty x p_X(x)\,dx \cdot \int_{-\infty}^\infty y p_Y(y)\,dy = \mathbb{E}[X] \cdot \mathbb{E}[Y] \quad \blacksquare$$

---

### 4. 🔢 Concrete Numerical Micro-Example
Let $X \sim \text{Unif}[0, 1]$ and $Y \sim \text{Unif}[0, 2]$ be independent random variables:
1. $p_X(x) = 1$ on $[0, 1]$ and $p_Y(y) = 0.5$ on $[0, 2]$.
2. Compute the joint PDF $p_{XY}(x, y)$:
   $$p_{XY}(x, y) = p_X(x) \cdot p_Y(y) = 1 \cdot 0.5 = 0.5 \quad \text{on } [0, 1] \times [0, 2]$$
3. Compute $P(X \le 0.5, Y \le 1.0)$:
   $$P(X \le 0.5, Y \le 1.0) = P(X \le 0.5) \cdot P(Y \le 1.0) = 0.5 \times \frac{1.0}{2.0} = 0.5 \times 0.5 = 0.25 = 25\%$$

---

### 5. 💻 Runnable Python / NumPy Snippet

```python
# Verifying Independence Factorization in NumPy
rng = np.random.default_rng(42)
N = 100_000

# Independent draws
X = rng.uniform(0.0, 1.0, size=N)
Y = rng.uniform(0.0, 2.0, size=N)

# P(X <= 0.5 and Y <= 1.0)
joint_prob = np.mean((X <= 0.5) & (Y <= 1.0))
prob_X = np.mean(X <= 0.5)
prob_Y = np.mean(Y <= 1.0)

print(f"Empirical Joint P(X<=0.5, Y<=1.0): {joint_prob:.4f}")
print(f"Product of Marginals P(X)*P(Y):    {prob_X * prob_Y:.4f}")
assert np.isclose(joint_prob, prob_X * prob_Y, atol=0.01)
```

---

### 6. 🧠 Diagnostic Mini-Check
1. **Question:** If $F(1, 2) = F_X(1) \cdot F_Y(2)$ at one specific point $(1, 2)$, does that prove $X$ and $Y$ are independent?  
   *Answer:* No! Factorization must hold for **all** $(x, y) \in \mathbb{R}^2$ and all event sets.
2. **Question:** What is $p(x \mid y)$ if $X$ and $Y$ are independent?  
   *Answer:* $p(x \mid y) = p_X(x)$.
3. **Question:** When are marginal distributions sufficient to uniquely reconstruct the joint distribution?  
   *Answer:* When the random variables are statistically independent.

---

## Pillar 8: IID Vectors, Transformations & The Jacobian Determinant

<a id="p8-iid"></a>

### 1. 👶 ELI5 Intuition
- **IID (Independent and Identically Distributed):** Making 100 exact photocopies of the same drawing and scattering them into 100 independent envelopes. Every sample has the exact same distribution and does not talk to any other sample.
- **The Jacobian Rubber Sheet:** Imagine drawing a 2D density circle on a rubber sheet and stretching it into a rotated diamond with transformation $Y = g(X)$:
  - As the rubber sheet stretches to cover **twice the area**, the density of the ink becomes **half as thick**.
  - The **Jacobian Determinant $|\det \mathbf{J}|$** is the exact mathematical zoom-factor that keeps track of how much 2D area expands or shrinks so total probability volume remains $1.0$!

```
  2D Coordinate Transformation via Invertible Map Y = g(X)
  
  X-Space (x1, x2) ─────────────► Y-Space (y1, y2)
  Density p_X(x1, x2)             Density p_Y(y1, y2) = p_X(h(y)) × |det J|
                                  where J = [∂x_i / ∂y_j]
```

---

### 2. 🔍 Plain-English Breakdown
1. **IID Definition:** Random variables $X_1, X_2, \dots, X_n$ are IID if:
   - They are mutually independent: $p(x_1, \dots, x_n) = \prod_{i=1}^n p(x_i)$.
   - Each variable follows the exact same identical distribution: $p_{X_i}(u) = p(u)$ for all $i$.
2. **Multivariate Transformation Rule:**
   - Let $\mathbf{Y} = g(\mathbf{X})$ be an invertible, differentiable mapping from $\mathbb{R}^n \to \mathbb{R}^n$ with inverse $\mathbf{X} = h(\mathbf{Y})$.
   - The transformed density is:
     $$p_{\mathbf{Y}}(\mathbf{y}) = p_{\mathbf{X}}\bigl(h(\mathbf{y})\bigr) \cdot \left| \det \mathbf{J}_h(\mathbf{y}) \right|$$
   - Where $\mathbf{J}_h(\mathbf{y})$ is the Jacobian matrix of partial derivatives of the inverse map:
     $$\mathbf{J}_h(\mathbf{y}) = \begin{pmatrix} \frac{\partial x_1}{\partial y_1} & \cdots & \frac{\partial x_1}{\partial y_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial x_n}{\partial y_1} & \cdots & \frac{\partial x_n}{\partial y_n} \end{pmatrix}$$

---

### 3. 📐 Formal Mathematics

#### 2×2 Linear Transformation Example
Let $Y_1 = X_1 + X_2$ and $Y_2 = X_1 - X_2$.
1. Solve for inverse map $X = h(Y)$:
   $$X_1 = \frac{Y_1 + Y_2}{2}, \quad X_2 = \frac{Y_1 - Y_2}{2}$$
2. Compute the Jacobian Matrix:
   $$\mathbf{J} = \begin{pmatrix} \frac{\partial X_1}{\partial Y_1} & \frac{\partial X_1}{\partial Y_2} \\ \frac{\partial X_2}{\partial Y_1} & \frac{\partial X_2}{\partial Y_2} \end{pmatrix} = \begin{pmatrix} 0.5 & 0.5 \\ 0.5 & -0.5 \end{pmatrix}$$
3. Compute the Determinant and Absolute Value:
   $$\det \mathbf{J} = (0.5)(-0.5) - (0.5)(0.5) = -0.25 - 0.25 = -0.50 \implies |\det \mathbf{J}| = 0.50 = \frac{1}{2}$$
4. Final Transformed Joint Density:
   $$p_{Y_1 Y_2}(y_1, y_2) = \frac{1}{2} \, p_{X_1 X_2}\left( \frac{y_1 + y_2}{2}, \, \frac{y_1 - y_2}{2} \right)$$

---

### 4. 🔢 Concrete Numerical Micro-Example
Suppose $(X_1, X_2)$ is uniform on the unit square $[0, 1] \times [0, 1]$ ($p_{X_1 X_2}(x_1, x_2) = 1.0$).
Transform $Y_1 = X_1 + X_2$ and $Y_2 = X_1 - X_2$:
- What is the value of the joint density $p_{Y_1 Y_2}$ at $(y_1, y_2) = (1.0, 0.2)$?
  1. Calculate inverse coordinates:
     $$x_1 = \frac{1.0 + 0.2}{2} = 0.60, \quad x_2 = \frac{1.0 - 0.2}{2} = 0.40$$
  2. Check if $(0.60, 0.40) \in [0, 1] \times [0, 1] \implies$ Yes!
  3. Evaluate density:
     $$p_Y(1.0, 0.2) = 1.0 \times \left|\det \mathbf{J}\right| = 1.0 \times 0.5 = 0.50$$

---

### 5. 💻 Runnable Python / NumPy Snippet

```python
# Verifying 2D Linear Transformation & Jacobian Scaling in NumPy
rng = np.random.default_rng(42)
N = 1_000_000

# Base density: X1, X2 ~ Unif[0, 1]
X1 = rng.uniform(0.0, 1.0, size=N)
X2 = rng.uniform(0.0, 1.0, size=N)

# Transform
Y1 = X1 + X2
Y2 = X1 - X2

# Check small neighborhood around (y1=1.0, y2=0.2) of width dy1=0.05, dy2=0.05
dy = 0.05
in_box = np.mean((np.abs(Y1 - 1.0) <= dy/2) & (np.abs(Y2 - 0.2) <= dy/2))
empirical_density = in_box / (dy * dy)

print(f"Empirical Density p_Y(1.0, 0.2): {empirical_density:.4f} (Theory: 0.5000)")
assert np.isclose(empirical_density, 0.50, atol=0.03)
```

---

### 6. 🧠 Diagnostic Mini-Check
1. **Question:** What does IID stand for?  
   *Answer:* Independent and Identically Distributed.
2. **Question:** Why do we take the absolute value of the Jacobian determinant $|\det \mathbf{J}|$ in the density transformation formula?  
   *Answer:* A negative determinant reflects a change in coordinate orientation (e.g., flipping axes), but probability density and volume must strictly remain non-negative.
3. **Question:** For $Y_1 = X_1 + X_2, Y_2 = X_1 - X_2$, what is $|\det \mathbf{J}|$ of the inverse map?  
   *Answer:* $1/2 = 0.5$.

---

## 🎯 Master Diagnostic Self-Assessment

Before opening [NOTES.md](./NOTES.md), test your mastery with this 7-question diagnostic check:

| # | Diagnostic Check Question | Hidden Answer Peek |
| :- | :--- | :--- |
| **1** | Must random variables $X$ and $Y$ share the same underlying sample space $\Omega$ to form a pair? | Yes, a random vector maps one underlying experiment $\Omega \to \mathbb{R}^2$. |
| **2** | How many corners of the Joint CDF are needed to compute a 2D rectangle probability? | 4 corners: $F(x_2, y_2) - F(x_2, y_1) - F(x_1, y_2) + F(x_1, y_1)$. |
| **3** | Can you uniquely reconstruct a joint distribution from its two marginals alone? | No, many joint distributions can share identical marginals unless independence is assumed. |
| **4** | What is the formula for the conditional density $p(x \mid y)$? | $p(x \mid y) = \frac{p_{XY}(x, y)}{p_Y(y)}$. |
| **5** | If $p_{XY}(x, y) = 2$ on $0 < x < y < 1$, what is the conditional distribution $X \mid Y = y$? | $\text{Unif}[0, y]$ with density $1/y$. |
| **6** | What single condition defines statistical independence between $X$ and $Y$? | $p_{XY}(x, y) = p_X(x) \cdot p_Y(y)$ for all $(x, y) \in \mathbb{R}^2$. |
| **7** | When transforming random vectors $\mathbf{Y} = g(\mathbf{X})$, what scaling factor multiplies the density? | The absolute Jacobian determinant $|\det \mathbf{J}|$ of the inverse mapping. |

---

**You are now 100% prepared! Proceed to [NOTES.md](./NOTES.md) for the complete lecture deep-dive.**
