# References & Learning Bridges: Lecture 51 (Positional Embeddings)

A curated repository of foundational papers, curriculum bridges, textbooks, industry implementations, and interactive visualizers for Positional Embeddings, Permutation Equivariance, and Cross-Attention.

---

## Curriculum & Prerequisite Bridges

- [Lecture 48: Attention Part 1](../64-Lec48-Attention-Part1/NOTES.md)  
  *Why Read This:* Establishes the historical genesis of attention, recurrent encoder sequence bottlenecks, and the initial motivation for learned Query, Key, and Value projections.
- [Lecture 49: Attention Part 2](../65-Lec49-Attention-Part2/NOTES.md)  
  *Why Read This:* Provides the mathematical derivation of scaled dot-product attention, variance scaling proofs, and row-simplex probability geometry.
- [Lecture 50: Multi-Head Attention and Transformer Architecture](../66-Lec50-Multi-Head-Attention-Transformer/NOTES.md)  
  *Why Read This:* Builds the multi-head subspace projections, LayerNorm, and residual stream that receive these positional encodings.
- [Lecture 52: Transfer Learning and Knowledge Distillation](../68-Lec52-Transfer-Learning-Knowledge-Distillation/NOTES.md)  
  *Why Read This:* Explores how pretrained Transformer representations are adapted to downstream enterprise tasks.
- [Lecture 53: SGD, RMSProp, and Adam Optimizers](../69-Lec53-SGD-RMSprop-Adam-Optimizers/NOTES.md)  
  *Why Read This:* Details first-order optimization algorithms that train positional and semantic parameters.

---

## Foundational & Seminal Papers

- [Vaswani et al. (2017) — Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
  *Why Read This:* Section 3.5 introduces the canonical sinusoidal positional encoding equations (Equations 1 & 2), detailing the geometric progression of wavelengths from $2\pi$ to $20000\pi$ and the linear relative shift property.
- [Su et al. (2024) — RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)  
  *Why Read This:* Derives Rotary Position Embedding (RoPE), proving that rotating queries and keys by block-diagonal orthogonal 2D matrices naturally enforces that inner products depend exclusively on relative token displacement $m - n$.
- [Shaw, Uszkoreit, & Vaswani (2018) — Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)  
  *Why Read This:* Modifies attention scores to incorporate learnable relative position vectors $a_{ij}^K$ and $a_{ij}^V$ clipped at a maximum distance $k$.
- [Press, Smith, & Lewis (2022) — Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization](https://arxiv.org/abs/2108.12409)  
  *Why Read This:* Introduces ALiBi, subtracting a non-learned linear penalty $m(i-j)$ from pre-softmax attention logits to achieve length generalization.
- [Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)  
  *Why Read This:* Demonstrates the practical trade-offs of learned positional embedding tables ($E_{\text{pos}} \in \mathbb{R}^{512 \times D}$) versus fixed sinusoidal encodings.
- [Radford et al. (2019) — Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)  
  *Why Read This:* Documents learned absolute positional embeddings in autoregressive decoder-only Transformers up to 1024 tokens.
- [Bahdanau, Cho, & Bengio (2015) — Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)  
  *Why Read This:* Introduces soft alignment weights between encoder and decoder hidden states, the direct predecessor to Transformer cross-attention.

---

## Textbooks & Video Lectures

- [Strang (2019) — Linear Algebra and Learning from Data](https://math.mit.edu/~gs/learningfromdata/)  
  *Why Read This:* Chapter II.4 covers orthogonal matrices, 2D rotations in $\mathrm{SO}(2)$, and spectral decomposition of harmonic functions.
- [Jurafsky & Martin (2024) — Speech and Language Processing (3rd ed., Chapter 10: Transformers)](https://web.stanford.edu/~jurafsky/slp3/10.pdf)  
  *Why Read This:* Authoritative textbook chapter detailing absolute vs relative positional encodings and cross-attention sequence alignments.
- [Bishop & Bishop (2024) — Deep Learning: Foundations and Concepts (Chapter 12: Transformers)](https://www.bishopbook.com)  
  *Why Read This:* Rigorous statistical and architectural analysis of permutation symmetry, geometric embeddings, and coordinate spaces.
- [Andrej Karpathy — Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)  
  *Why Read This:* Demonstrates how position embeddings are created as `nn.Embedding(block_size, n_embd)` and summed directly with token embeddings.
- [Umar Jamil — Rotary Positional Embeddings (RoPE) Deep Dive](https://www.youtube.com/watch?v=o29P0Kpobz0)  
  *Why Read This:* Step-by-step mathematical dissection of complex numbers, 2D rotation matrices, and PyTorch implementation of RoPE.

---

## Industry & Implementation Guides

- [PyTorch Documentation — Rotary Position Embedding Implementation in Torchtune](https://pytorch.org/torchtune/stable/modules/generated/torchtune.modules.RotaryPositionalEmbeddings.html)  
  *Why Read This:* Official PyTorch implementation of complex rotary position embeddings with cache management.
- [Hugging Face — RoFormer Model Card & Technical Specs](https://huggingface.co/docs/transformers/model_doc/roformer)  
  *Why Read This:* Production documentation on rotary embeddings and position interpolation in large foundation models.
- [FlashAttention — Hardware-Efficient RoPE Kernel Fusions](https://github.com/Dao-AILab/flash-attention)  
  *Why Read This:* Demonstrates how rotary transformations are computed in GPU SRAM tiles during attention forward and backward passes.
