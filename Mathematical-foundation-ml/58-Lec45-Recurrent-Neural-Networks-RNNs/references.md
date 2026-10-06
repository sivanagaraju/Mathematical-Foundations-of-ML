# References & Pedagogical Resources: Recurrent Neural Networks & Sequence Modeling

This curated bibliography provides formal academic citations, seminal papers, foundational textbooks, and interactive visualizers directly supporting **Lecture 45: Recurrent Neural Networks (RNNs)**.

---

## 1. Curriculum & Prerequisite Bridges

1. **Lecture 42: ERM on Neural Networks & Error Backpropagation**
   - **Internal Link:** [`55-Lec42-ERM-Neural-Networks-Backpropagation`](../55-Lec42-ERM-Neural-Networks-Backpropagation/NOTES.md) — **Why Read This:** Establishes reverse-mode automatic differentiation on DAG computational graphs, forming the mathematical backbone of unrolled network training.

2. **Lecture 43: Local Receptive Fields & Parameter Sharing**
   - **Internal Link:** [`56-Lec43-Local-Receptive-Field-Parameter-Sharing`](../56-Lec43-Local-Receptive-Field-Parameter-Sharing/NOTES.md) — **Why Read This:** Details parameter tying and translation equivariance in spatial grids, contrasting spatial parameter sharing with temporal parameter sharing across time steps.

3. **Lecture 44: CNNs as Regularized MLPs**
   - **Internal Link:** [`57-Lec44-CNNs-as-Regularized-MLP`](../57-Lec44-CNNs-as-Regularized-MLP/NOTES.md) — **Why Read This:** Formulates convolutional networks as regularized MLPs with Toeplitz matrix constraints, setting up the exact conceptual template for viewing RNNs as causal block-Toeplitz MLPs.

4. **Core Math Concept: Chain Rule & Backpropagation**
   - **Internal Link:** [`Chain Rule & Backpropagation`](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/04-Chain_Rule_and_Backpropagation.md) — **Why Read This:** Provides multivariable calculus foundations for composite function derivatives and adjoint error sensitivity propagation.

---

## 2. Foundational & Seminal Papers

5. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).**
   - **Paper Link:** [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) — **Why Read This:** Seminal publication introducing modern backpropagation and formalizing unrolled recurrent networks as multi-stage computational graphs.

6. **Jordan, M. I. (1986).**
   - **Paper Link:** [Attractor dynamics and parallelism in a connectionist sequential machine](https://escholarship.org/uc/item/25b8856n) — **Why Read This:** Foundational early connectionist architecture introducing recurrence via feedback loops from output units back into recurrent state layers.

7. **Elman, J. L. (1990).**
   - **Paper Link:** [Finding structure in time](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1) — **Why Read This:** Landmark paper introducing the canonical vanilla RNN ("Elman Network") with hidden-to-hidden recurrent transitions $h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t)$.

8. **Werbos, P. J. (1990).**
   - **Paper Link:** [Backpropagation through time: what it does and how to do it](https://ieeexplore.ieee.org/document/80202) — **Why Read This:** Establishes the formal mathematical framework for unrolling recurrent networks into feedforward graphs and executing gradient descent across temporal sequences.

9. **Bengio, Y., Simard, P., & Frasconi, P. (1994).**
   - **Paper Link:** [Learning long-term dependencies with gradient descent is difficult](https://ieeexplore.ieee.org/document/279181) — **Why Read This:** Seminal mathematical proof demonstrating that gradient descent on unrolled recurrent networks encounters exponential gradient decay or explosion as sequence length grows.

10. **Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014).**
    - **Paper Link:** [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — **Why Read This:** Formulates the Sequence-to-Sequence Encoder-Decoder paradigm, compressing variable-length source sequences into fixed context vectors to drive auto-regressive generation.

11. **Sutskever, I., Vinyals, O., & Le, Q. V. (2014).**
    - **Paper Link:** [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — **Why Read This:** Landmark empirical demonstration of deep multi-layer LSTM encoder-decoder architectures outperforming classical phrase-based statistical machine translation.

---

## 3. Textbooks & Comprehensive Reference Guides

12. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press.
    - **Book Link:** [Deep Learning Book Chapter 10 (Sequence Modeling: Recurrent and Recursive Nets)](https://www.deeplearningbook.org/contents/rnn.html) — **Why Read This:** Definitive theoretical reference covering computational graphs, state transition mechanics, teacher forcing, and the causal formulation of unrolled recurrent networks.

13. **Graves, A. (2012).** *Supervised Sequence Labelling with Recurrent Neural Networks.* Springer.
    - **Book Link:** [Supervised Sequence Labelling with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/phd.pdf) — **Why Read This:** Rigorous PhD thesis providing mathematical derivations for loss gradients, CTC loss, and recurrent state dynamics across continuous sequences.

14. **Jurafsky, D., & Martin, J. H. (2024).** *Speech and Language Processing (3rd ed. draft).*
    - **Book Link:** [Speech and Language Processing Chapter 9 (RNNs and LSTMs)](https://web.stanford.edu/~jurafsky/slp3/9.pdf) — **Why Read This:** Masterful natural language processing text explaining sequence classification, auto-regressive language generation, perplexity, and token-level conditioning.

---

## 4. Industry & Framework Implementation Blueprints

15. **PyTorch Core Development Team.**
    - **Documentation Link:** [PyTorch torch.nn.RNN Architectural Reference](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) — **Why Read This:** Authoritative technical specification detailing PyTorch tensor layout conventions `(seq_len, batch, input_size)`, fused GEMM optimizations, and cuDNN recurrent acceleration.

16. **Karpathy, A. (2015).**
    - **Blog Link:** [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) — **Why Read This:** Classic tutorial detailing character-level auto-regressive language modeling, sequence sampling mechanics, and hidden state interpretation.

---

## 5. Interactive Demonstrations & Visualizers

17. **Olah, C. (2015).**
    - **Blog Link:** [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — **Why Read This:** The gold standard pedagogical visual walkthrough of unrolled recurrent networks, information flow diagrams, and state transitions.

18. **Carter, S., Armstrong, Z., Schubert, L., Johnson, I., & Olah, C. (2016).**
    - **Distill Article:** [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/) — **Why Read This:** Interactive visual diagrams exploring the structural limitations of fixed-size RNN hidden memory bottlenecks and the conceptual evolution toward attention mechanisms.
