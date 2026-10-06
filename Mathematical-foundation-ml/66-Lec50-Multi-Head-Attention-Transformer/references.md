# References & Learning Bridges: Lecture 50 (Multi-Head Attention and Transformer Architecture)

A curated repository of foundational papers, curriculum bridges, textbooks, industry implementations, and interactive visualizers for Multi-Head Attention, Normalization Mechanics, and the complete Transformer architecture.

---

## Curriculum & Prerequisite Bridges

- [Lecture 48: Attention Part 1](../64-Lec48-Attention-Part1/NOTES.md)  
  *Why Read This:* Establishes the historical genesis of attention, recurrent encoder sequence bottlenecks, and the initial motivation for learned Query, Key, and Value projections.
- [Lecture 49: Attention Part 2](../65-Lec49-Attention-Part2/NOTES.md)  
  *Why Read This:* Provides the mathematical derivation of scaled dot-product attention, variance scaling proofs, and row-simplex probability geometry.
- [Lecture 51: Positional Embeddings](../67-Lec51-Positional-Embeddings/NOTES.md)  
  *Why Read This:* Introduces sinusoidal, learned, and rotary positional encodings that break permutation equivariance in the Transformer blocks built in Lecture 50.
- [Lecture 52: Transfer Learning and Knowledge Distillation](../68-Lec52-Transfer-Learning-Knowledge-Distillation/NOTES.md)  
  *Why Read This:* Explores how Transformer foundation models trained on web-scale text are adapted to downstream enterprise tasks.

---

## Foundational & Seminal Papers

- [Vaswani et al. (2017) — Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
  *Why Read This:* The seminal paper introducing Multi-Head Attention, the complete encoder-decoder Transformer block, and the output projection matrix $W^O$.
- [Ba, Kiros, & Hinton (2016) — Layer Normalization](https://arxiv.org/abs/1607.06450)  
  *Why Read This:* Formulates Layer Normalization across feature dimensions, proving its independence from batch sizes and superiority in recurrent and sequence architectures.
- [Ioffe & Szegedy (2015) — Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)  
  *Why Read This:* The original paper demonstrating how standardizing intermediate layer activations accelerates optimization and regularizes deep feedforward networks.
- [He et al. (2016) — Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)  
  *Why Read This:* Establishes the mathematics of residual shortcut connections $X + F(X)$ that enable gradient propagation across networks exceeding 100 layers.
- [Dosovitskiy et al. (2020) — An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)  
  *Why Read This:* Introduces the Vision Transformer (ViT), demonstrating how decomposing 2D images into non-overlapping patches enables sequence transformers to surpass CNNs.
- [Gu & Dao (2023) — Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)  
  *Why Read This:* Illustrates the modern return to structured recurrence, providing linear $\mathcal{O}(T)$ inference time while matching Transformer expressivity.

---

## Textbooks & Video Lectures

- [Jurafsky & Martin — Speech and Language Processing (3rd ed., Chapter 10: Transformers)](https://web.stanford.edu/~jurafsky/slp3/10.pdf)  
  *Why Read This:* Authoritative textbook chapter covering Multi-Head Attention equations, tensor dimensions, and encoder-decoder stacks.
- [Bishop & Bishop (2024) — Deep Learning: Foundations and Concepts (Chapter 12: Transformers)](https://www.bishopbook.com.html)  
  *Why Read This:* Rigorous statistical and architectural analysis of multi-head projections, residual stream geometry, and LayerNorm dynamics.
- [Andrej Karpathy — Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)  
  *Why Read This:* Highly pedagogical implementation of multi-head attention, residual connections, and Pre-LN blocks in clean PyTorch.
- [Umar Jamil — Attention is all you need: Detailed Architecture Walkthrough](https://www.youtube.com/watch?v=bCz4OMemCcA)  
  *Why Read This:* Step-by-step mathematical dissection of tensor shapes, multi-head splitting, and residual stream transformations.

---

## Industry & Implementation Guides

- [PyTorch Documentation — torch.nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)  
  *Why Read This:* Official documentation for PyTorch's native multi-head attention module, detailing batch-first parameterization and key padding masks.
- [Hugging Face — Transformers Architecture Philosophy](https://huggingface.co/docs/transformers/philosophy)  
  *Why Read This:* Engineering guidelines on modular transformer sub-layers, Pre-LN vs Post-LN stability, and model scaling configurations.
- [NVIDIA — Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)  
  *Why Read This:* Shows how multi-head attention matrices $W^Q, W^K, W^V$ and $W^O$ are split across multiple GPUs using tensor parallel matrix multiplication.

---

## Interactive Visualizers

- [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)  
  *Why Read This:* Renowned visual guide showing multi-head attention splitting, residual add & norm layers, and feedforward blocks in clear diagrams.
- [Poloclub at Georgia Tech — CNN vs Transformer Explainer](https://poloclub.github.io/transformer-explainer/)  
  *Why Read This:* Interactive browser tool allowing exploration of attention heads and feature distributions across deep transformer layers.
