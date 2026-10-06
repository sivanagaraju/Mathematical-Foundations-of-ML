# Glossary & Terminology Index — Tutorial 15 Part 1: RNNs, LSTMs and GRUs

This glossary provides formal definitions, spoken pronunciation guides, pedagogical analogies, and cross-course links to canonical mathematical terms for **Tutorial 15 Part 1: Recurrent Neural Networks, Long Short-Term Memory, and Gated Recurrent Units**.

---

## Technical Terms & Concepts

| Term | Spoken English (Phonetics) | Mathematical & Architectural Definition | Conceptual Role & Analogy | MathsTerms Bridge |
|:-----|:---------------------------|:----------------------------------------|:--------------------------|:------------------|
| **Recurrent Neural Network (RNN)** | /rɪˈkɜːr.ənt/ (ree-KUR-unt neural net-wurk) | A neural network architecture that processes temporal sequences by updating a persistent hidden state vector $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)$ across time steps. | Like reading a book sentence-by-sentence, carrying an evolving mental summary in mind rather than reading the entire book simultaneously. | [Recurrent Neural Networks](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/02-Recurrent_Neural_Networks.md) |
| **Hidden State** | /ˈhɪd.ən steɪt/ (HID-un stayt) | A continuous vector $h_t \in \mathbb{R}^H$ summarizing the temporal context and historical inputs from step $1$ to step $t$. | The short-term working memory of the network, updated at every word. | [Vectors & Matrices](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/01-Vectors_and_Matrices.md) |
| **Cell State** | /sɛl steɪt/ (sell stayt) | An unconstrained linear memory channel $C_t \in \mathbb{R}^H$ in LSTMs that carries long-term information across sequences with minimal non-linear attenuation. | A high-speed conveyor belt transporting raw memories directly across time steps without intermediate squashing. | [Recurrent Neural Networks](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/02-Recurrent_Neural_Networks.md) |
| **Backpropagation Through Time (BPTT)** | /bækˌprɒp/ (back-prop-uh-GAY-shun throo time) | An extension of reverse-mode automatic differentiation where the recurrent computational graph is unrolled over $T$ steps to calculate weight gradients. | Replaying a movie backwards frame-by-frame to determine how an early scene influenced the climax. | [Chain Rule & Backpropagation](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/04-Chain_Rule_and_Backpropagation.md) |
| **Vanishing Gradient** | /ˈvæn.ɪ.ʃɪŋ/ (VAN-ish-ing GRAY-dee-unt) | The exponential decay of backpropagated error gradients towards zero caused by repeated multiplication by transition matrices with spectral norm $< 1$. | A whisper traveling down a long line of people that fades into complete silence before reaching the end. | [Derivatives & Gradients](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/02-Derivatives_Gradients_and_Jacobians.md) |
| **Long Short-Term Memory (LSTM)** | /ɛl.ɛs.tiː.ɛm/ (el-ess-tee-em) | A gated recurrent architecture equipped with a linear cell state and three non-linear gates (forget, input, output) that prevents gradient decay. | A dynamic notebook where you explicitly decide what to erase, what new facts to write down, and what to read aloud. | [Recurrent Neural Networks](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/02-Recurrent_Neural_Networks.md) |
| **Gated Recurrent Unit (GRU)** | /ɡruː/ (jee-ar-yoo or groo) | A streamlined variant of LSTM that combines cell and hidden states and uses two gates (reset and update) for computational efficiency. | A lightweight digital tablet that merges the notebook and display into a single streamlined screen. | [Recurrent Neural Networks](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/02-Recurrent_Neural_Networks.md) |
| **Forget Gate** | /fərˈɡɛt ɡeɪt/ (fer-GET gayt) | A sigmoid-activated layer $f_t = \sigma(W_f [x_t, h_{t-1}] + b_f)$ that computes an elementwise retention percentage in $[0, 1]^H$ for the past cell state. | An eraser dial deciding whether an old memory is obsolete and should be wiped clean. | [Activation Functions](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/05-Activation_Functions.md) |
| **Input Gate** | /ˈɪn.pʊt ɡeɪt/ (IN-put gayt) | A sigmoid-activated layer $i_t = \sigma(W_i [x_t, h_{t-1}] + b_i)$ that scales candidate memories $\tilde{C}_t$ before adding them to the cell state. | A filter controlling which incoming news headlines are important enough to archive. | [Activation Functions](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/05-Activation_Functions.md) |
| **Hadamard Product** | /ˈhæd.əˌmɑːrd/ (HAD-uh-mard PRAH-dukt) | An algebraic operator $\odot$ denoting elementwise multiplication between two matrices or vectors of identical shape: $(A \odot B)_{ij} = A_{ij} B_{ij}$. | Two volume sliders acting point-by-point on individual frequency bands. | [Vectors & Matrices](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/01-Vectors_and_Matrices.md) |

---

## Mathematical Notation & Tensor Shape Legend

- $B$: Batch size (number of independent sequences).
- $T$: Sequence length (number of temporal tokens).
- $D$: Input feature dimension (e.g. word embedding size).
- $H$: Hidden state dimension (number of recurrent memory units).
- $C$: Number of classification categories.
- $\sigma(z) = \frac{1}{1 + e^{-z}}$: Sigmoid logistic function mapping $\mathbb{R} \to (0, 1)$.
- $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$: Hyperbolic tangent mapping $\mathbb{R} \to (-1, 1)$.
