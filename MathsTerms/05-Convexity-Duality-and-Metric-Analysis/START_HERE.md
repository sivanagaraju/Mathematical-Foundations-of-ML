# Start Here: Convexity, Duality & Metric Analysis

> **Who this is for:** A learner preparing to understand the deep mathematical derivations of Generative AI (VAEs, GANs, and Flow Matching).
>
> **What this page does:** It explains why these four advanced topics exist and how they unlock modern generative architectures.

---

## The Honest Starting Point

You should **not** begin your mathematical journey here!
Before studying this cluster, you should already have completed:
1. **Linear Algebra:** Vectors, norms, and dot products.
2. **Calculus:** Derivatives, gradients, and multivariable functions.
3. **Probability:** Random variables, expectations, and distributions.

Once you have those tools, the concepts in this cluster will make immediate, intuitive sense rather than feeling like abstract hurdles.

---

## Why Generative AI Needs This Cluster

| Concept | The Question It Answers | The Generative AI Architecture It Unlocks |
| :--- | :--- | :--- |
| **Convexity & Jensen's Inequality** | When we cannot compute $\log \mathbb{E}[p(x \mid z)]$, how do we push the logarithm inside to get a computable lower bound? | **Variational Autoencoders (VAEs):** Deriving the Evidence Lower Bound ($\mathcal{L}_{\text{ELBO}}$). |
| **Bounds & Supremum** | What is the tightest possible ceiling or floor for an approximation, even if the boundary is never reached? | **Variational Objectives:** Proving that the ELBO is the tightest possible lower bound on data log-likelihood. |
| **Fenchel Conjugate & Duality** | How do we turn an intractable integral into a two-player game over functions? | **$f$-GANs & VDM:** Replacing intractable density ratios with discriminator neural network probes. |
| **Lipschitz Continuity** | How do we stop discriminator gradients from exploding while measuring distance between probability distributions? | **Wasserstein GANs (WGAN-GP):** The 1-Lipschitz critic condition and the Kantorovich-Rubinstein dual. |

---

## The Reading Order

1. **[Convexity and Jensen's Inequality](./01-Convexity_and_Jensens_Inequality.md)**
2. **[Bounds, Supremum, Infimum & Linear Families](./02-Bounds_Supremum_Infimum_and_Linear_Families.md)**
3. **[Fenchel Conjugate and Dual Representations](./03-Fenchel_Conjugate_and_Dual_Representations.md)**
4. **[Lipschitz Continuity](./04-Lipschitz_Continuity.md)**
