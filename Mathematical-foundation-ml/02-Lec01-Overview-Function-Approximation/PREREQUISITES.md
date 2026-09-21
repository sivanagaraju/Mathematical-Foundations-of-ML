# Prerequisites — warm-up before Lec 01

> **Do this first.** Then open [NOTES.md](./NOTES.md) at the **Executive Summary** map.  
> Basics only — not a second lecture. They unlock words on the master map if you are rusty.

```
  After this warm-up you can say:

  "A function maps each allowed input to one output."
  "Data is a finite list of (input, output) pairs."
  "A model is an estimate of a hidden rule, not the rule itself."
  "An image can be stacked into one long vector of numbers."
  "When physics cannot write the rule, counting many repeats still helps."
```

---

## Math Terminology Rosetta Stone

Before diving into the foundational pillars, use this reference table to decode mathematical shorthand and notation into plain English, spoken phonetics, and software implementations.

| Symbol / Notation | Spoken English (Phonetics) | Mathematical Concept | Plain-English Intuition | Dedicated MathsTerm Link |
| :--- | :--- | :--- | :--- | :--- |
| $f: X \to Y$ | **EFF FROM EKS TO WYE** | Target Mapping Function | The true ground-truth rule converting inputs into outputs | [Functions & Derivatives](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/01-Functions_Derivatives_and_Rules.md) |
| $\hat{f}(x; \theta)$ | **EFF-HAT OF EKS GIV-un THAY-tuh** | Parametric Approximator | A neural network or polynomial model with trainable tensor weights | [Functions & Derivatives](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/01-Functions_Derivatives_and_Rules.md) |
| $\theta \in \mathbb{R}^p$ | **THAY-tuh IN AR-PEE** | Parameter Vector | The collection of all trainable weights and biases (`model.parameters()`) | [Vectors & Matrices](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/01-Vectors_and_Matrices.md) |
| $\mathbf{x} \in \mathbb{R}^d$ | **BOLD EKS IN AR-DEE** | Feature Vector | Input data sample flattened into a 1D continuous tensor of length $d$ | [Tensors & Shapes](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/04-Tensors_and_Shapes.md) |
| $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ | **DEE EQUALS SET OF PAIRS** | Supervised Training Dataset | A finite batch of labeled input-output examples loaded into memory | [Random Variables & Distributions](../../MathsTerms/03-Probability-and-Statistical-Estimation/01-Random_Variables_and_Distributions.md) |
| $\langle \mathbf{w}, \mathbf{x} \rangle$ | **INNER PRODUCT OF DUB-yoo AND EKS** | Euclidean Dot Product | Linear projection multiplying coordinate elements and summing: `torch.dot(w, x)` | [Dot Product & Similarity](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/03-Dot_Product_and_Similarity.md) |
| $\mathcal{L}(\theta)$ | **CAL-ih-GRAF-ik EL OF THAY-tuh** | Empirical Loss Function | Scalar error penalty computed over training instances: `criterion(pred, target)` | [Loss Functions](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/08-Loss_Functions.md) |

---

## 🌉 Curriculum & Sibling Course Prerequisite Bridges

Before diving into the foundational pillars, review these key concepts from sibling course series and standalone mathematical foundations:

| Assumed Concept | Primary Series Foundation | MathsTerms Deep-Dive | 1-Sentence Intuition Refresher |
| :--- | :--- | :--- | :--- |
| **Probability Triplet** | [Lec 02: Probability Theory Part 1](../../Mathematical-foundation-ml/03-Lec02-Recap-Probability-Theory-Part1/NOTES.md) | [Probability Basics & Axioms](../../MathsTerms/03-Probability-and-Statistical-Estimation/00-Probability_Basics_and_Axioms.md) | Moving from deterministic function approximation to statistical modeling requires formal sample spaces. |
| **Vector-Valued Data** | [Lec 06: X-Ray Sample from Distribution](../../Mathematical-foundation-ml/07-Lec06-XRay-Sample-From-Distribution/NOTES.md) | [Tensors & Shapes](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/04-Tensors_and_Shapes.md) | Sensory matrices are flattened into high-dimensional continuous coordinate spaces $\mathbb{R}^d$. |
| **Distribution Estimation** | [Lec 08: Distribution Estimation](../../Mathematical-foundation-ml/09-Lec08-Distribution-Estimation/NOTES.md) | [Random Variables & Distributions](../../MathsTerms/03-Probability-and-Statistical-Estimation/01-Random_Variables_and_Distributions.md) | The fundamental paradigm shift from fitting $f(x) \approx y$ to learning probability densities $p(x, y)$. |

---

## 1. Sets: bags of allowed things

<a id="p1-sets"></a>

A **set** is a collection of allowed objects. Picture a labeled bag: members are in; non-members are out.

Mini scene: exam scores from 0 to 100 form a set. Score 73 is in; 150 is out.

```
  set X = { allowed "x" values }
  set Y = { allowed "y" values }
```

The lecture’s **domain** and **range** are just two bags with different jobs.

---

## 2. Functions: a rule that never freelances

<a id="p2-functions"></a>

A **function** $f$ from $X$ to $Y$ sends **each** element of $X$ to **exactly one** element of $Y$.

Like a vending machine: same button → same product every time.

```
  x  ──f──►  y = f(x)

  OK:     one y per x
  NOT OK: same x sometimes y1, sometimes y2
```

$$
f : X \to Y
$$

In words: $f$ takes anything allowed in $X$ and returns one thing in $Y$.

You do **not** need a formula. “$f$ exists but is unknown” is allowed — that is the learning problem.

---

## 3. Ordered pairs and a data notebook

<a id="p3-pairs-data"></a>

An **ordered pair** $(x,y)$ means “this input with this observed output.” Order matters: $(2,5)$ is not $(5,2)$.

**Data** $D$ is a **finite list of pairs** you actually saw:

$$
D = \{(x_1,y_1),\ldots,(x_n,y_n)\}
$$

In words: $D$ is a notebook of $n$ experiments — snapshots, not the full rule.

```
  Notebook D
  night 1 → position A
  night 2 → position B
  night 3 → position C
  night 10 → ??? (never written)
```

---

## 4. Domain and range in plain English

<a id="p4-domain-range"></a>

| Word | Plain meaning |
|------|----------------|
| **Domain** | What you may put in ($X$) |
| **Range** (here) | What kind of answers are allowed out ($Y$) |

“Input/output” is fine casually. The lecturer prefers domain/range because later these connect to **random variables**.

```
  domain X ----f----> range Y
    time               position
    X-ray image        disease 0/1
    speech vector      phone / word
```

---

## 5. Estimate vs truth (models before jargon)

<a id="p5-estimate"></a>

True rule $f$ generated the pairs. You almost never hold $f$. You build a stand-in $\hat{f}$.

- **Truth $f$:** hidden rule  
- **Estimate / model:** rule you write and use  
- **Useful:** good enough on questions you care about  

```
  reality:  x ──f──► y      (unknown)
  practice: x ──hat f──► prediction
```

Slogan later: “all models are wrong but some are useful” sits on this split.

---

## 6. Vectors and matrices (enough for X-rays)

<a id="p6-vectors"></a>

A **vector** is an ordered list of numbers, e.g. $(2,5,1)$ in $\mathbb{R}^3$.

A **matrix** is a rectangle of numbers. A grayscale image is a matrix of pixel intensities (often 0–255).

```
  Tiny 2×3 "image":
    10  20  30
    40  50  60

  Stack columns → one vector length 6:
  (10, 40, 20, 50, 30, 60)
```

The lecture treats each X-ray as one point in $\mathbb{R}^{P Q}$ after this stacking.

---

## 7. High dimension without fear

<a id="p7-high-dim"></a>

2D needs 2 numbers; 3D needs 3; $d$ dimensions need $d$ numbers. You cannot draw $d=10{,}000$, but distance and averaging still work on paper and in code.

```
  d=2:  point on a plane
  d=3:  point in a room
  d=PQ: one whole image after stacking — still one point
```

“High-dimensional” means **many numbers per example**, not magic.

---

## 8. Chance from many repeats (pre-probability)

<a id="p8-repeats"></a>

If a process is too messy for clean physics (coin in turbulent air), you can still **repeat** and **count**.

```
  toss 10_000 times
  count heads H
  rough chance of heads ≈ H / 10_000
```

You did not solve the fluid dynamics. You still got a useful number. The lecture calls related ideas **statistical methods** and motivates **probability theory** as the precise language for “repeated observations → estimates.” Deeper objects (sample space, random variable, measure) are previewed at the end of NOTES and taught next lectures.

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.


---

## 🗝️ Mathematical Foundations & MathsTerms Bridge

> [!TIP]
> **Foundational Knowledge Base:** This module directly relies upon formal mathematical constructs systematically defined and verified in our central [`MathsTerms`](../../MathsTerms) repository. For visual dependency graphs and multi-track learning roadmaps, consult the [Grand Unified Concept Map](../../MathsTerms/CONCEPT_MAP.md).

| Mathematical Concept | Dedicated Guide | Role & Significance in This Lecture |
| :--- | :--- | :--- |
| **Functions, Derivatives & Rules** | [Functions, Derivatives & Rules](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/01-Functions_Derivatives_and_Rules.md) | Function approximation fundamentals, domain, range, and unknown mapping $f: X \to Y$ |
| **Vectors & Matrices** | [Vectors & Matrices](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/01-Vectors_and_Matrices.md) | Vector representations of physical observations and data points |
| **Tensors, Dimensions & Shapes** | [Tensors, Dimensions & Shapes](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/04-Tensors_and_Shapes.md) | Image stacking into high-dimensional coordinate vectors $\mathbb{R}^{PQ}$ |
| **Random Variables & Probability Distributions** | [Random Variables & Probability Distributions](../../MathsTerms/03-Probability-and-Statistical-Estimation/01-Random_Variables_and_Distributions.md) | Transitioning from deterministic function approximation to probabilistic modeling |

---
