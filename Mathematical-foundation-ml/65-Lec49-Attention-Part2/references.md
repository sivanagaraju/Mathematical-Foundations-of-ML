# References & Learning Bridges: Lecture 49 (Attention Part 2)

A curated repository of foundational papers, curriculum bridges, textbooks, industry implementations, and interactive visualizers for scaled dot-product attention and feedforward transformations.

---

## Curriculum & Prerequisite Bridges

- [Lecture 48: Attention Part 1](../64-Lec48-Attention-Part1/NOTES.md)  
  *Why Read This:* Establishes the historical genesis of attention, recurrent encoder sequence bottlenecks, and the initial motivation for learned Query, Key, and Value projections.
- [Lecture 50: Multi-Head Attention and Transformer Architecture](../66-Lec50-Multi-Head-Attention-Transformer/NOTES.md)  
  *Why Read This:* Extends single-head scaled dot-product attention into multi-head parallel subspaces, analyzing projection matrices, concatenation, and output projection $W^O$.
- [Lecture 51: Positional Embeddings](../67-Lec51-Positional-Embeddings/NOTES.md)  
  *Why Read This:* Solves the permutation equivariance limitation proven in Topic 3 by introducing absolute sinusoidal and learned positional encodings.
- [Lecture 47: Sequence Modeling Foundations](../63-Lec47-Sequence-Models/NOTES.md)  
  *Why Read This:* Provides the background on sequence tensor geometry, token representations, and temporal Markovian transitions before attention.

---

## Foundational & Seminal Papers

- [Vaswani et al. (2017) — Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
  *Why Read This:* The seminal paper introducing scaled dot-product attention, multi-head attention, and the full Transformer architecture, establishing the $1/\sqrt{D_k}$ scaling law.
- [Bahdanau et al. (2014) — Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)  
  *Why Read This:* Introduces the foundational additive attention mechanism, illustrating how dynamic alignment vectors bypass fixed-length encoder bottlenecks.
- [Dao et al. (2022) — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)  
  *Why Read This:* Demonstrates how tiling and online softmax evaluation eliminate the need to materialize the $O(T^2)$ attention matrix in GPU HBM, achieving $2\times-4\times$ speedups.
- [Ba et al. (2016) — Layer Normalization](https://arxiv.org/abs/1607.06450)  
  *Why Read This:* Details the normalization mechanics applied across token feature channels before and after attention and feedforward sub-layers.
- [Geva et al. (2021) — Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)  
  *Why Read This:* Empirically proves that post-attention two-layer MLPs act as associative memories storing factual knowledge extracted by attention routing.

---

## Textbooks & Video Lectures

- [Jurafsky & Martin — Speech and Language Processing (3rd ed., Chapter 10: Transformers)](https://web.stanford.edu/~jurafsky/slp3/10.pdf)  
  *Why Read This:* Comprehensive textbook treatment of self-attention tensor mechanics, mathematical formulations, and sequence classification pipelines.
- [Bishop & Bishop (2024) — Deep Learning: Foundations and Concepts (Chapter 12: Transformers)](https://www.bishopbook.com.html)  
  *Why Read This:* Rigorous mathematical derivation of attention as kernel smoothing and non-parametric regression over vector representations.
- [Andrej Karpathy — Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)  
  *Why Read This:* Clear line-by-line Python walkthrough implementing scaled dot-product attention, variance scaling, and feedforward blocks in PyTorch.
- [Stanford CS224N — Lecture 8: Self-Attention and Transformers](https://www.youtube.com/watch?v=ptGAkoW10AM)  
  *Why Read This:* Detailed pedagogical lecture by Christopher Manning explaining query-key-value retrieval mechanics and linguistic representations.

---

## Industry & Implementation Guides

- [PyTorch Documentation — torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)  
  *Why Read This:* Official documentation for PyTorch's native C++ FlashAttention and fused scaled dot-product kernel.
- [Hugging Face — Transformers Attention Implementation Details](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertSelfAttention)  
  *Why Read This:* Production reference implementation showing batched tensor shapes, attention head splitting, and dropout integration.
- [NVIDIA Developer Blog — Optimizing Transformer Attention with Tensor Cores](https://developer.nvidia.com/blog/)  
  *Why Read This:* Technical deep dive into mapping matrix multiplications $Q K^T$ and $A V$ to Hopper and Blackwell FP16/BF16 tensor cores.

---

## Interactive Visualizers

- [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)  
  *Why Read This:* Intuitive step-by-step visual diagrams depicting vector projections, scaled dot-product matrix multiplication, and value aggregation.
- [Polo Club at Georgia Tech — Transformer Explainer](https://poloclub.github.io/transformer-explainer/)  
  *Why Read This:* Real-time in-browser interactive visualization allowing users to hover over tokens and observe row-simplex attention probabilities.
