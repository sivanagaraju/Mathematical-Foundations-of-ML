# References & Learning Bridges: Lecture 53 (SGD, RMSprop, Adam: Optimizers)

A curated repository of foundational papers, curriculum bridges, textbooks, industry implementations, and interactive visualizers for First-Order Stochastic Optimization, Heavy Ball Momentum, Coordinate-Wise Scaling, Adam, Hessian Eigenvalue Geometry, and Generalization Dynamics.

---

## Curriculum & Prerequisite Bridges

- [Lecture 12: Empirical Risk Minimization](../13-Lec12-Empirical-Risk-Minimisation/NOTES.md)  
  *Why Read This:* Establishes the core empirical risk objective $\hat{R}_N(\theta)$ that SGD, RMSprop, and Adam iteratively minimize across training minibatches.
- [Lecture 52: Transfer Learning & Knowledge Distillation](../68-Lec52-Transfer-Learning-Knowledge-Distillation/NOTES.md)  
  *Why Read This:* Details how model checkpoints produced by Adam optimization are transferred, frozen, and distilled for downstream tasks.
- [Lecture 50: Multi-Head Attention and Transformer Architecture](../66-Lec50-Multi-Head-Attention-Transformer/NOTES.md)  
  *Why Read This:* Explores the deep transformer architecture where ill-conditioned coordinate scales make Adam and AdamW mandatory.
- [MathsTerms: Gradient Descent](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/09-Gradient_Descent.md)  
  *Why Read This:* Formulates the classical first-order steepest descent framework and step-size contraction guarantees.
- [MathsTerms: Hessian Matrix and Curvature](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/02b-Hessian_Matrix_and_Curvature.md)  
  *Why Read This:* Provides the mathematical definition of second-order partial derivative tensors, condition numbers $\kappa$, and spectral decomposition.
- [MathsTerms: Exponential Moving Average](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/10-Exponential_Moving_Average_EMA.md)  
  *Why Read This:* Details the recursive filters and geometric weighting series underpinning momentum and RMSprop second-moment tracking.

---

## Foundational & Seminal Papers

- [Kingma & Ba (2014) — Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)  
  *Why Read This:* The seminal paper introducing the Adam optimizer, deriving the first and second moment estimators, analytical bias correction terms, and empirical convergence proofs.
- [Polyak (1964) — Some Methods of Speeding Up the Convergence of Iteration Methods](https://www.sciencedirect.com/science/article/abs/pii/0041555364901375)  
  *Why Read This:* The original paper establishing the Heavy Ball method, proving the quadratic acceleration from $\mathcal{O}(\kappa)$ to $\mathcal{O}(\sqrt{\kappa})$ on ill-conditioned quadratics.
- [Nesterov (1983) — A Method for Unconstrained Convex Minimization Problem with the Rate of Convergence O(1/k^2)](https://ci.nii.ac.jp/naid/10029969234/)  
  *Why Read This:* Introduces Nesterov Accelerated Gradient (NAG), evaluating gradients at look-ahead positions to provide an anticipatory braking mechanism.
- [Tieleman & Hinton (2012) — Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)  
  *Why Read This:* The original presentation of RMSprop in Hinton's Coursera lecture, introducing coordinate-wise normalization by the root mean square of recent gradients.
- [Duchi, Hazan, & Singer (2011) — Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://jmlr.org/papers/v12/duchi11a.html)  
  *Why Read This:* The foundational AdaGrad paper establishing diagonal coordinate preconditioning for sparse and frequent features.
- [Loshchilov & Hutter (2019) — Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)  
  *Why Read This:* Discovers the fundamental flaw in combining Adam with L2 regularization, introducing **AdamW** by decoupling weight decay from adaptive gradient scaling.
- [Dauphin et al. (2014) — Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization](https://arxiv.org/abs/1406.2572)  
  *Why Read This:* Seminal paper demonstrating through statistical physics and random matrix theory that saddle points proliferate exponentially compared to local minima in deep learning.
- [Ge et al. (2015) — Escaping From Saddle Points — Online Stochastic Gradient Descent for Tensor Decomposition](https://arxiv.org/abs/1503.02101)  
  *Why Read This:* Proves mathematically that stochastic noise in SGD enables polynomial-time escape from strict saddle points.
- [Reddi, Kale, & Kumar (2018) — On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237)  
  *Why Read This:* Identifies theoretical non-convergence issues in Adam under non-stationary online environments, proposing AMSGrad with non-decreasing second-moment memory.
- [Bottou, Curtis, & Nocedal (2018) — Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838)  
  *Why Read This:* Comprehensive survey detailing stochastic approximations, condition numbers, minibatch variance reduction, and second-order methods.

---

## Textbooks & Video Lectures

- [Goodfellow, Bengio, & Courville (2016) — Deep Learning (Chapter 8: Optimization for Training Deep Models)](https://www.deeplearningbook.org)  
  *Why Read This:* The standard reference textbook covering non-convex optimization landscapes, ill-conditioning, momentum variants, and adaptive algorithms.
- [Nocedal & Wright (2006) — Numerical Optimization (Chapters 2 & 3: Fundamentals of Unconstrained Optimization)](https://link.springer.com/book/10.1007/978-0-387-40065-5)  
  *Why Read This:* Authoritative classical text covering Taylor series expansions, line search conditions, Hessian condition numbers, and Newton-Raphson methods.
- [Andrej Karpathy (CS231n) — Neural Networks Part 3: Learning and Evaluation](https://cs231n.github.io/neural-networks-3/)  
  *Why Read This:* Highly accessible visual and intuitive explanation of SGD, Momentum, RMSprop, Adam, learning rate schedules, and model checkpointing.

---

## Industry & Implementation Guides

- [PyTorch Documentation — `torch.optim.AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)  
  *Why Read This:* Official documentation and implementation notes for the standard production AdamW optimizer in PyTorch.
- [Hugging Face Documentation — Optimization for Large Language Models](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)  
  *Why Read This:* Industry guide detailing learning rate warm-up schedules, cosine annealing, and 8-bit Adam memory optimizations for multi-billion parameter LLMs.
