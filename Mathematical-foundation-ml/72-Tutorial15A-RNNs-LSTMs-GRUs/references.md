# References & Academic Citations — Tutorial 15 Part 1: RNNs, LSTMs and GRUs

This document provides seminal academic papers, textbook chapters, production engineering guides, and interactive visualizers supporting **Tutorial 15 Part 1: Recurrent Neural Networks, LSTMs, and GRUs**.

---

## 1. Foundational & Seminal Research Papers

1. **Hochreiter, S., & Schmidhuber, J. (1997).**  
   *Long Short-Term Memory.*  
   *Neural Computation*, 9(8), 1735–1780.  
   - [Neural Computation Paper](https://dl.acm.org/doi/10.1162/neco.1997.9.8.1735)
   - **Why Read This:** Landmark paper introducing the LSTM architecture, formulating the constant error carousel (CEC) to solve exponential gradient decay in backpropagation through time.

2. **Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014).**  
   *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.*  
   *EMNLP 2014*, 1724–1734.  
   - [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
   - **Why Read This:** Proposes the Gated Recurrent Unit (GRU) with reset and update gates, establishing an expressive 25% parameter reduction over LSTMs for sequence-to-sequence translation.

3. **Bengio, Y., Simard, P., & Frasconi, P. (1994).**  
   *Learning long-term dependencies with gradient descent is difficult.*  
   *IEEE Transactions on Neural Networks*, 5(2), 157–166.  
   - [IEEE Paper](https://ieeexplore.ieee.org/document/279181)
   - **Why Read This:** Formal mathematical analysis establishing the fundamental trade-off between dynamical stability and gradient decay in recurrent systems.

4. **Pascanu, R., Mikolov, T., & Bengio, Y. (2013).**  
   *On the difficulty of training recurrent neural networks.*  
   *International Conference on Machine Learning (ICML 2013)*, 1310–1318.  
   - [arXiv:1211.5063](https://arxiv.org/abs/1211.5063)
   - **Why Read This:** Demonstrates the geometric origins of exploding gradients and proposes gradient norm clipping as an exact stabilization technique for recurrent models.

5. **Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014).**  
   *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.*  
   *NIPS Deep Learning Workshop 2014*.  
   - [arXiv:1412.3555](https://arxiv.org/abs/1412.3555)
   - **Why Read This:** Comprehensive empirical study comparing vanilla RNN, LSTM, and GRU across speech and polyphonic music modeling datasets, confirming gated models dramatically outperform simple RNNs.

---

## 2. Textbooks & Course Lectures

6. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).**  
   *Deep Learning.* MIT Press.  
   - [Deep Learning Book Chapter 10: Sequence Modeling: Recurrent and Recursive Nets](https://www.deeplearningbook.org/contents/rnn.html)
   - **Why Read This:** Authoritative treatment of computational graphs unrolled across time, parameter sharing, BPTT, and gated recurrent architectures.

7. **Jurafsky, D., & Martin, J. H. (2024).**  
   *Speech and Language Processing (3rd ed. draft).*  
   - [Chapter 9: Deep Learning Architectures for Sequence Processing](https://web.stanford.edu/~jurafsky/slp3/9.pdf)
   - **Why Read This:** Excellent pedagogical explanation of RNNs, LSTMs, and GRUs applied to natural language classification, sentiment analysis, and token sequence generation.

---

## 3. Industry & Implementation Guides

8. **PyTorch Documentation (2024).**  
   *Recurrent Layers: nn.RNN, nn.LSTM, nn.GRU.*  
   - [PyTorch RNN Documentation](https://pytorch.org/docs/stable/nn.html#recurrent-layers)
   - **Why Read This:** Canonical API reference detailing the exact mathematical equations, weight layouts (`weight_ih`, `weight_hh`), and `batch_first` semantics implemented in PyTorch C++ backends.

9. **Olah, C. (2015).**  
   *Understanding LSTM Networks.*  
   - [Colah's Blog Post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
   - **Why Read This:** World-renowned visual walkthrough detailing step-by-step gate flow diagrams, conveyor-belt intuition, and GRU design trade-offs.

---

## 4. Interactive Visualizers & Demos

10. **Karpathy, A. (2015).**  
    *The Unreasonable Effectiveness of Recurrent Neural Networks.*  
    - [Andrej Karpathy Blog & Visualizer](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
    - **Why Read This:** Interactive text generation visualizations illustrating character-level RNN hidden cell activations tracking quotation marks, code indentation, and markdown formatting.

11. **Strobelt, H., Gehrmann, S., Pfister, H., & Rush, A. M. (2018).**  
    *LSTMVis: A Tool for Visual Analysis of Hidden State Dynamics in Recurrent Neural Networks.*  
    - [LSTMVis Interactive Tool](http://lstm.seas.harvard.edu/)
    - **Why Read This:** Browser-based interactive tool for tracking and exploring how hidden dimensions in trained LSTMs activate on specific linguistic constructs.

---

## 5. Lecture Timestamp Cross-References

| Concept | Video Timestamp | Mathematical & Architectural Details |
|:---|:---|:---|
| **Sequence Data & Tensor Dimensions** | 00:00–04:15 | Tensor layout $X \in \mathbb{R}^{B \times T \times D}$, batch-first convention, token embeddings |
| **Vanilla RNN Cell Formulation** | 04:15–06:15 | Recurrent transition $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)$, weight sharing across time |
| **BPTT & Vanishing Gradient Limits** | 06:15–09:12 | Unrolled computation graphs, repeated multiplication by $W_{hh}^T$, exponential gradient decay |
| **Gated Recurrent Unit (GRU) Mechanics** | 09:12–14:30 | Reset gate $r_t$, update gate $z_t$, candidate state $\tilde{h}_t$, convex combination update |
| **LSTM Dual State & Triple Gating** | 14:30–22:45 | Cell state $C_t$ vs hidden state $h_t$, forget gate $f_t$, input gate $i_t$, output gate $o_t$, CEC |
| **Parameter Counting Arithmetic** | 22:45–28:30 | Deriving $P = K \cdot [H(D + H) + 2H]$ for $K \in \{1, 3, 4\}$, exact 25% GRU savings |
| **PyTorch Layers & Sequence Classification** | 28:30–34:06 | `nn.RNN`, `nn.LSTM`, `nn.GRU`, output sequence vs $h_T$, linear classification head |

---

## 6. Curriculum & Prerequisite Bridges

- **Preceding Milestone (Package 64: Lec 48 Attention Part 1):**  
  [Package 64](../64-Lec48-Attention-Part1/) develops content-based addressing and dynamic key-value routing, contrasting with the fixed-size hidden memory bottleneck of recurrent models explored here.
- **Preceding Milestone (Package 68: Lec 52 Transfer Learning & Distillation):**  
  [Package 68](../68-Lec52-Transfer-Learning-Knowledge-Distillation/) establishes pretrained feature reuse, enabling CNN visual backbones to feed embeddings directly into the recurrent decoders developed in this tutorial.
- **Preceding Milestone (Package 70: Tutorial 14 Part 1 CNNs):**  
  [Package 70](../70-Tutorial14A-CNNs/) establishes PyTorch tensor operations, dimension tracking, and module design applied here to temporal sequence models.
- **Succeeding Milestone (Package 73: Tutorial 15 Part 2 Deep RNNs):**  
  Directly extends the single-layer recurrent cells developed here to deep stacked networks, bidirectional processing (`bidirectional=True`), and variable-length sequence packing (`pack_padded_sequence`).
