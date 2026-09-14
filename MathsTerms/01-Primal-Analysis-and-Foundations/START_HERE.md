# Start Here: A First-Principles Route into Mathematical Foundations for AI

> **Who this is for:** a learner who is new to the mathematics used in machine learning and generative AI.
>
> **What this page does:** it gives the learning order. The numbered files in this folder are a useful catalogue, but they are not, by themselves, a complete zero-background sequence.

---

## The honest starting point

Before probability, calculus, or neural networks, a learner needs a little mathematical language: arithmetic with fractions and negative numbers; powers; equations and inequalities; the coordinate plane; and the idea of a set and a function. You do **not** need measure theory, linear algebra, or programming to begin.

This knowledge base currently teaches those later topics in several sibling folders. Follow the route below rather than choosing a page only because its filename has the smallest number.

## The learning route

| Stage | Central question | Read next | Do not move on until you can… |
| :--- | :--- | :--- | :--- |
| 0. Mathematical readiness | What do symbols, sets, intervals, equations, and graphs mean? | Review school algebra if these are unfamiliar. | Read $x \in S$, solve a simple inequality, and plot a line. |
| 1. Exponentials and logarithms | How do repeated multiplication and its inverse work? | [Logarithms & Exponential Functions](./02-Logarithms_and_Exponential_Functions.md) | Explain why $\ln(ab)=\ln a+\ln b$ and why $e^x>0$. |
| 2. Probability | How do we reason about uncertainty using outcomes and events? | [Probability Basics & Axioms](./01-Probability_Basics_and_Axioms.md) | Calculate a conditional probability and explain Bayes' rule with a small table. |
| 3. Functions and change | What is a function, and how does its output react to an input change? | [Functions, Derivatives & Rules](../03-Multivariate-Calculus-and-Optimization/01-Functions_Derivatives_and_Rules.md) | Read a graph, find a simple derivative, and interpret its sign. |
| 4. Vectors and distances | How do we represent many numbers, directions, and distances at once? | [Vectors & Matrices](../02-Linear-Algebra-Geometry-and-Tensors/01-Vectors_and_Matrices.md) then [Vector Norms & Inner Products](../02-Linear-Algebra-Geometry-and-Tensors/02-Vector_Norms_and_Inner_Products.md) | Compute a dot product and explain a vector norm as a distance or size. |
| 5. Multivariable change and convexity | When is optimization well behaved, and how do averages interact with curves? | [Derivatives, Gradients & Jacobians](../03-Multivariate-Calculus-and-Optimization/02-Derivatives_Gradients_and_Jacobians.md), then [Convexity & Jensen's Inequality](./03-Convexity_and_Jensens_Inequality.md) | Recognize a convex curve and state Jensen's inequality in words. |
| 6. Tight bounds | What is the best possible ceiling or floor, even when it is never reached? | [Bounds, Supremum, Infimum & Linear Families](./04-Bounds_Supremum_Infimum_and_Linear_Families.md) | Distinguish maximum from supremum using $[0,1)$. |
| 7. Dual representations | How can a curved convex function be represented through its supporting lines? | [Fenchel Conjugate & Dual Representations](./05-Fenchel_Conjugate_and_Dual_Representations.md) | Compute one simple conjugate and state Fenchel--Young in words. |
| 8. Global stability | How can we limit how fast a model's output changes? | [Lipschitz Continuity](./06-Lipschitz_Continuity.md) | Explain what a 1-Lipschitz critic means and why WGAN uses one. |

## How to use a chapter

For a first pass, read the chapter's opening, notation decoder, worked example, and self-check. Then return for the proof and AI application. The AI material is the destination: it should make the mathematics useful, not be a barrier to learning the definition.

When a page links to an advanced term you do not yet know, use the route above. Skipping forward is fine for curiosity; it is not required for mastery.

## What these ideas eventually unlock

The route leads from basic numerical reasoning to the tools behind likelihoods in LLMs, probability models in diffusion, variational objectives in VAEs, $f$-GAN duality, and the Lipschitz critics used in Wasserstein GANs. Each later application depends on the earlier mathematical object; none needs to be memorized before the object makes sense on its own.
