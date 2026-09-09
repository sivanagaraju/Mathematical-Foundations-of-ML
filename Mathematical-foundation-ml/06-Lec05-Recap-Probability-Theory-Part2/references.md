# External References & Prerequisite Bridges: Lec 05 Probability Recap Part 2

> **Document Role:** Authoritative, deeply annotated reference hub for Lecture 05.  
> **How to Use:** While [NOTES.md](./NOTES.md) teaches the core narrative and derivations, consult this file for seminal research papers, textbook chapter cross-references, sibling course prerequisites, and interactive visualizers.

---

## 1. 🌉 Curriculum & Prerequisite Bridges

When concepts in this lecture build upon mathematical foundations taught elsewhere in the curriculum or catalogued in [`MathsTerms/`](../../MathsTerms/), use these exact links to revisit first principles:

| Concept / Technique | Sibling Lecture / Source | MathsTerms Deep-Dive | Why Revisit? (The Dot Connected) |
| :--- | :--- | :--- | :--- |
| **Random Variables & Preimages** | [Lec 03: Probability Recap 2](../../Mathematical-foundation-ml/04-Lec03-Recap-Probability-Theory-Part2/NOTES.md) | [Random Variables & Distributions](../../MathsTerms/04-Probability-and-Statistical-Estimation/01-Random_Variables_and_Distributions.md) | Extends scalar measurable functions $X: \Omega \to \mathbb{R}$ to joint vector spaces $\mathbb{R}^d$. |
| **Product Geometry & Pushforward** | [Lec 04: Probability Recap 3](../../Mathematical-foundation-ml/05-Lec04-Recap-Probability-Theory-Part3/NOTES.md) | [Joint, Marginal & Conditional Distributions](../../MathsTerms/04-Probability-and-Statistical-Estimation/03-Joint_Marginal_Conditional_Dist.md) | Connects 2D CDF bounding rectangles to joint density marginalization. |
| **Chest X-Ray Vector Sampling** | [Lec 06: X-Ray Sample from Distribution](../../Mathematical-foundation-ml/07-Lec06-XRay-Sample-From-Distribution/NOTES.md) | [Tensors & Shapes](../../MathsTerms/02-Linear-Algebra-Geometry-and-Tensors/04-Tensors_and_Shapes.md) | Applies joint probability $P(X, Y)$ to medical feature vectors $X \in \mathbb{R}^d$ and disease diagnoses $Y \in \{0, 1\}$. |
| **Distribution Estimation** | [Lec 08: Distribution Estimation](../../Mathematical-foundation-ml/09-Lec08-Distribution-Estimation/NOTES.md) | [Random Variables & Distributions](../../MathsTerms/04-Probability-and-Statistical-Estimation/01-Random_Variables_and_Distributions.md) | Connects learning conditional $P(Y \mid X)$ (supervised) vs marginal $P(X)$ (unsupervised/generative). |

---

## 2. 📄 Foundational & Seminal Research Papers

Curated landmark research publications establishing joint probability calculus and Bayesian conditioning:

### 2.1 [An Essay towards solving a Problem in the Doctrine of Chances]
- **Authors:** Thomas Bayes, communicated by Richard Price
- **Publication:** Philosophical Transactions of the Royal Society of London (1763)
- **Direct Link:** [Bayes 1763 Royal Society](https://royalsocietypublishing.org/doi/10.1098/rstl.1763.0053)
- **Core Insight:** Introduces the mathematical theorem for inverting conditional probabilities given prior and likelihood observations.
- **Why Read This:** Section 2 lays down the fundamental law relating joint probabilities to conditional updates: $P(A \mid B) = \frac{P(A \cap B)}{P(B)}$.

### 2.2 [Sugli integrali doppi (Fubini's Theorem on Product Measures)]
- **Authors:** Guido Fubini
- **Publication:** Rendiconti della Reale Accademia dei Lincei (1907)
- **Direct Link:** [Fubini 1907 Historical Paper](https://archive.org/details/rendicontidellar161907)
- **Core Insight:** Proves that the double integral of a joint measurable function over product spaces can be computed as iterated 1D integrals in either order: $\iint f(x,y) dx dy = \int (\int f(x,y) dy) dx$.
- **Why Read This:** Guarantees that marginalizing joint continuous distributions is commutative and mathematically well-behaved.

---

## 3. 📚 Authoritative Textbooks & University Video Lectures

Curated chapters and lecture timestamps from foundational machine learning literature:

### 3.1 Textbooks
1. **Pattern Recognition and Machine Learning (Christopher M. Bishop, Springer 2006)**
   - *Relevant Chapters:* Chapter 1 (Section 1.2: The Rules of Sum and Product, Bayes' Theorem), Chapter 2 (Probability Distributions).
   - *Direct Resource:* [Microsoft Research Bishop PRML](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
   - *Key Takeaway:* Pristine geometric diagrams illustrating joint surfaces, marginal projections, and conditional slices.
2. **Probabilistic Machine Learning: An Introduction (Kevin P. Murphy, MIT Press 2022)**
   - *Relevant Chapters:* Chapter 2 (Probability: Univariate and Multivariate Models), Chapter 3 (Linear Gaussian Models).
   - *Direct Resource:* [probml.github.io](https://probml.github.io/)
   - *Key Takeaway:* Exact analytical derivations of conditional and marginal distributions for multivariate Gaussian random variables.
3. **Probability and Random Processes (Geoffrey Grimmett & David Stirzaker, Oxford University Press 2001)**
   - *Relevant Chapters:* Chapter 3 (Discrete Random Variables, Conditioning), Chapter 4 (Continuous Random Variables, Joints).
   - *Key Takeaway:* Rigorous proofs of independence, conditional expectation, and product $\sigma$-algebras.

### 3.2 Video Lectures & Course Series
1. **Mathematical Foundations of Generative AI (Prof. Prathosh A. P., IIT Madras / IISc)**
   - *Series Focus:* Why a vector-valued random variable is mathematically equivalent to $d$ scalar random variables sharing one common sample space $\Omega$.
2. **MIT 6.041SC: Probabilistic Systems Analysis (Prof. John Tsitsiklis, MIT OpenCourseWare)**
   - *Direct Resource:* [MIT OCW 6.041SC Lecture 4 & 5](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/)
   - *Key Takeaway:* Exceptional visual problem-solving on joint probability tables and conditional conditioning.

---

## 4. 🛠️ Industry Implementation Guides & Production Engineering

Real-world architectural guides and probabilistic computing frameworks:

1. **PyTorch Distributions: Multivariate Normal Implementations**
   - *MultivariateNormal Guide:* [PyTorch MultivariateNormal](https://pytorch.org/docs/stable/distributions.html#multivariatenormal) — Batched covariance factorization via Cholesky decomposition (`torch.linalg.cholesky`) for stable sampling.
2. **Scikit-Learn Probability Calibration & Bayes Classifiers**
   - *Naive Bayes User Guide:* [Scikit-Learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) — Industrial implementations factoring high-dimensional joint distributions under conditional independence.

---

## 5. 🎛️ Interactive Visualizers & Educational Demos

Interactive web studios and animations that build physical intuition for joint distributions:

1. **Seeing Theory: Compound Probability & Conditioning (Brown University)**
   - *Resource:* [Seeing Theory Compound Probability](https://seeing-theory.brown.edu/compound-probability/index.html)
   - *Relevance:* Interactive 2D scatter plots and marginal projection sliders demonstrating joint densities and correlation.
2. **3Blue1Brown: Bayes' Theorem Visualized (Grant Sanderson)**
   - *Resource:* [3Blue1Brown Bayes Theorem](https://www.3blue1brown.com/lessons/bayes-theorem)
   - *Relevance:* Visualizing probability space as a unit area square partitioned into conditional likelihood blocks.
