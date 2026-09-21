# 📚 Annotated References Hub: Density Functions

This reference hub provides foundational textbooks, peer-reviewed literature, interactive mathematical visualizers, industry implementation guides, and cross-course curriculum bridges for **Lecture 09: Density Function** in *Mathematical Foundations of Machine Learning*.

---

## 1. Primary Textbooks & Classical Treatises

1. **Casella, G., & Berger, R. L. (2002).** *Statistical Inference* (2nd ed.). Duxbury Press.
   - **Why Read This:** Essential graduate treatise covering transformations and expectations of continuous random variables, Radon-Nikodym derivatives, support sets, and Jacobian determinants under monotonic mappings.
   - *Direct Relevance:* Topics 1, 2, 4.

2. **Billingsley, P. (1995).** *Probability and Measure* (3rd ed.). John Wiley & Sons.
   - **Why Read This:** Provides measure-theoretic foundation showing that continuous densities represent Radon-Nikodym derivatives with respect to Lebesgue measure, proving singletons $\{x\}$ possess measure zero ($\lambda(\{x\}) = 0$).
   - *Direct Relevance:* Topics 1, 3.

3. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer.
   - **Why Read This:** Section 1.2.1 offers visual and mathematical intuition for continuous probability densities and demonstrates why the mode/maximum of a density is not invariant under coordinate transformations.
   - *Direct Relevance:* Topics 2, 5.

4. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press.
   - **Why Read This:** Chapter 3 formalizes continuous random variables, probability density functions, Dirac delta distributions, and empirical density matching principles utilized throughout modern generative models.
   - *Direct Relevance:* Topics 4, 5.

---

## 2. Peer-Reviewed Papers & Seminal Works

1. **Parzen, E. (1962).** *On Estimation of a Probability Density Function and Mode*. The Annals of Mathematical Statistics, 33(3), 1065–1076.
   - **Why Read This:** Seminal mathematical derivation establishing consistent non-parametric kernel density estimators (KDE) and asymptotic normality.

2. **Silverman, B. W. (1986).** *Density Estimation for Statistics and Data Analysis*. Chapman and Hall.
   - **Why Read This:** The benchmark monograph detailing optimal bandwidth selection, boundary bias correction, and the curse of dimensionality in multivariate continuous spaces.

---

## 3. Industry & Implementation Guides

1. **[PyTorch Probability Distributions Package (`torch.distributions`)](https://pytorch.org/docs/stable/distributions.html)**
   - **Why Read This:** Production documentation demonstrating how modern deep learning frameworks evaluate log-density (`log_prob`), handle batch dimensions, and avoid underflow when evaluating continuous densities.

2. **[SciPy Statistical Functions (`scipy.stats`)](https://docs.scipy.org/doc/scipy/reference/stats.html)**
   - **Why Read This:** Standard industry library for continuous distributions, numerical integration of probability density functions, and kernel density estimation.

---

## 4. Interactive Web Visualizers & Simulation Tools

1. **[Seeing Theory — Continuous Distributions](https://seeingtheory.brown.edu/probability-distributions/index.html#section2)**
   - **Why Read This:** Interactive browser simulation showing how shrinking histogram bin width $\Delta x \to 0$ converges to continuous density curves $f(x)$ while individual point frequencies vanish to zero.

2. **[Distributions Tool (Setosa.io)](https://setosa.io/ev/markov-chains/)**
   - **Why Read This:** Visualizes continuous areas under density functions and demonstrates the difference between probability mass $P(X = k)$ and density height $f(x)$.

---

## 5. 🗝️ Central MathsTerms Knowledge Base Bridges

| Mathematical Concept | Repository Guide | Role in Lecture 09 |
| :--- | :--- | :--- |
| **Continuous Random Variables** | [Random Variables & Distributions](../../MathsTerms/03-Probability-and-Statistical-Estimation/01-Random_Variables_and_Distributions.md) | Continuous densities $p(x)$, infinitesimal probability $p(x)dx$, and integration |
| **Common Probability Distributions** | [Common Probability Distributions](../../MathsTerms/03-Probability-and-Statistical-Estimation/02-Common_Probability_Distributions.md) | Density height properties (why $p(x)$ can exceed 1) and normalization $\int p(x)dx = 1$ |
| **Joint, Marginal & Conditional Distributions** | [Joint, Marginal & Conditional Dist.](../../MathsTerms/03-Probability-and-Statistical-Estimation/03-Joint_Marginal_Conditional_Dist.md) | Joint density functions, volume integrals, and marginal density extraction |
| **Likelihood Formulation** | [Likelihood & Log-Likelihood](../../MathsTerms/03-Probability-and-Statistical-Estimation/04-Likelihood_and_Log_Likelihood.md) | Transitioning from discrete likelihood $\prod P(X=x_i)$ to continuous product density $\prod f(x_i)$ |
| **Derivatives & Calculus Rules** | [Functions, Derivatives & Rules](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/01-Functions_Derivatives_and_Rules.md) | Calculus of densities: fundamental theorem linking $F(x)$ and $f(x) = F'(x)$ |

---

## 6. 🌉 Sibling Course Curriculum Bridges

| Course Module | Focus & Prerequisite Connection | Path |
| :--- | :--- | :--- |
| **Lecture 04: Recap Probability Theory Part 3** | Cumulative Distribution Functions (CDFs), right-continuity, and pushforward measures | [Lecture 04 NOTES.md](../../Mathematical-foundation-ml/05-Lec04-Recap-Probability-Theory-Part3/NOTES.md) |
| **Lecture 05: Recap Probability Theory Part 2** | Conditioning, joint distributions, and continuous vs discrete transformations | [Lecture 05 NOTES.md](../../Mathematical-foundation-ml/06-Lec05-Recap-Probability-Theory-Part2/NOTES.md) |
| **Lecture 08: Distribution Estimation** | Setting up the foundational ML objective: estimating underlying distribution $P$ or density $p$ from data | [Lecture 08 NOTES.md](../../Mathematical-foundation-ml/09-Lec08-Distribution-Estimation/NOTES.md) |
| **Lecture 10: Challenges of ML** | The three-step recipe and challenges when fitting continuous density models in high dimensions | [Lecture 10 NOTES.md](../../Mathematical-foundation-ml/11-Lec10-Challenges-of-ML/NOTES.md) |
| **Lecture 11: Entropy** | Continuous differential entropy $h(X) = -\int f(x)\log f(x)dx$ and negative entropy properties | [Lecture 11 NOTES.md](../../Mathematical-foundation-ml/12-Lec11-Entropy/NOTES.md) |
