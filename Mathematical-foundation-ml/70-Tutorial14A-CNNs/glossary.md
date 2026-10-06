# Glossary of Core Terms: Tutorial 14 Part 1 (CNNs)

A rigorous mathematical glossary defining the terminology, spoken phonetics, and operational definitions of 2D Discrete Convolutions, Spatial Pooling, Receptive Fields, LeNet-5, and Scaled Vision Architectures.

---

## Terminology Dictionary

| Term | Spoken English (Phonetics) | Mathematical Symbol | Formal Mathematical Definition | Course / Conceptual Context | Foundation Link |
|:-----|:---------------------------|:--------------------|:-------------------------------|:----------------------------|:----------------|
| **2D Convolution** | *kon-vuh-LOO-shun* | $Y_k = b_k + \sum_c \tilde{X}_c * W_{k,c}$ | Discrete 2D cross-correlation sliding parameterized filter tensor over multi-channel input volume. | Foundational building block of computer vision architectures. | [01-Convolution_and_Pooling.md](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md) |
| **Spatial Pooling** | *SPAY-shul POO-ling* | $P_c(i, j) = \mathcal{R}(Z_c(\mathcal{N}_{i,j}))$ | Parameter-free spatial downsampling operator (Max or Average) applied independently per channel. | Reduces spatial resolution by 75% while introducing local translation invariance. | [01-Convolution_and_Pooling.md](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md) |
| **Weight Sharing** | *WAYT SHAIR-ing* | $W(u, v) \text{ shared } \forall (i, j)$ | Structural constraint applying identical kernel weights across all spatial grid coordinates. | Enforces translation equivariance and eliminates parameter explosion ($80,000\times$ savings). | [01-Convolution_and_Pooling.md](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md) |
| **Tensor** | *TEN-sor* | $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$ | Multi-dimensional numerical array indexing batch, channel depth, height, and width. | Fundamental data structure for perceptual image representation in deep learning. | [04-Tensors_and_Shapes.md](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/04-Tensors_and_Shapes.md) |
| **Receptive Field** | *rih-SEP-tiv FEELD* | $RF_L = 1 + \sum (F_l - 1)$ | The spatial area in the input volume that contributes to a specific neuron's activation. | Governs the scope of visual context captured by deep representation layers. | [01-Convolution_and_Pooling.md](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md) |
| **Frobenius Dot Product** | *FROH-ben-ee-us DOT PROD-ukt*| $\langle A, B \rangle_F = \sum A_{c,u,v} B_{c,u,v}$ | Sum of element-wise multiplications between receptive field patch and filter weights. | Core arithmetic evaluation executed at every sliding convolution window. | [03-Dot_Product_and_Similarity.md](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/03-Dot_Product_and_Similarity.md) |
| **Rectified Linear Unit** | *REK-tih-fyde LIN-ee-er YOO-nit* | $f(z) = \max(0, z)$ | Non-linear activation function with constant unit subgradient for positive inputs. | Eliminates vanishing gradients during multi-stage backpropagation in deep CNNs. | [05-Activation_Functions.md](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/05-Activation_Functions.md) |
| **Softmax Function** | *SOFT-maks* | $\sigma(z)_i = \frac{e^{z_i}}{\sum e^{z_j}}$ | Normalized exponential operator mapping real logits into categorical probability distribution. | Standard activation for the final classification layer across $K$ classes. | [06-Softmax.md](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/06-Softmax.md) |
| **Cross-Entropy Loss** | *KROSS EN-truh-pee LOSS* | $\mathcal{L}_{\text{CE}} = -\sum y_k \log \hat{y}_k$ | Maximum likelihood loss objective penalizing divergence between ground truth and predictions. | Standard classification objective optimized by Adam and SGD in vision models. | [08-Loss_Functions.md](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/08-Loss_Functions.md) |
| **Backpropagation** | *back-prop-uh-GAY-shun* | $\frac{\partial \mathcal{L}}{\partial W} = \sum \frac{\partial \mathcal{L}}{\partial Y} \frac{\partial Y}{\partial W}$ | Reverse-mode automatic differentiation computing loss gradients across layered tensors. | Core algorithm propagating error signals from classifier heads back to input conv kernels. | [04-Chain_Rule_and_Backpropagation.md](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/04-Chain_Rule_and_Backpropagation.md) |

---

### Convolution (Discrete 2D Cross-Correlation)
- **Spoken / Phonetic:** /kɒn.vəˈluː.ʃən/ (kon-vuh-LOO-shun)
- **Formal Definition:** A mathematical operation on two functions (or discrete tensors) where a parameterized sliding kernel $\mathbf{W} \in \mathbb{R}^{D_{\text{in}} \times F \times F}$ computes local element-wise products across an input volume $\mathbf{X} \in \mathbb{R}^{D_{\text{in}} \times H \times W}$ and sums the results, producing a spatially structured 2D feature map.
- **Intuitive Explanation:** A small magnifying template (filter) that scans across every position in an image looking for specific visual cues like vertical edges, textures, or diagonal strokes.
- **Reference Term:** [01-Convolution_and_Pooling.md](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md)

---

### Spatial Pooling (Max & Average Pooling)
- **Spoken / Phonetic:** /ˈspeɪ.ʃəl ˈpuː.lɪŋ/ (SPAY-shul POO-ling)
- **Formal Definition:** A parameter-free spatial downsampling operation $\mathcal{R}: \mathbb{R}^{C \times H \times W} \to \mathbb{R}^{C \times H' \times W'}$ that applies a local window reduction (maximum or arithmetic mean) independently across each 2D channel slice with stride $S_p$.
- **Intuitive Explanation:** Compressing an image by summarizing local neighborhoods (e.g. keeping only the brightest highlight in each $2 \times 2$ block) to make the representation smaller and resistant to small pixel shifts.
- **Reference Term:** [01-Convolution_and_Pooling.md](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md)

---

### Weight Sharing (Parameter Tying)
- **Spoken / Phonetic:** /weɪt ˈʃeə.rɪŋ/ (WAYT SHAIR-ing)
- **Formal Definition:** The structural constraint that the same kernel tensor $\mathbf{W}$ is applied at all spatial coordinates $(i, j)$ across the entire input grid, enforcing the translation equivariance prior and making parameter count independent of input spatial resolution.
- **Intuitive Explanation:** Using the exact same edge detector at the top-left, center, and bottom-right of an image rather than relearning a new detector at every single pixel.
- **Reference Term:** [01-Convolution_and_Pooling.md](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md)

---

### Tensor & Multi-Channel Volume
- **Spoken / Phonetic:** /ˈtɛn.sər/ (TEN-sor)
- **Formal Definition:** A multi-dimensional array of numerical values indexed over multiple coordinate axes (e.g., a 4D tensor $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$ indexing batch, channel, height, and width).
- **Intuitive Explanation:** A multi-layered grid of numbers. A color photo is a 3D block (Red, Green, Blue sheets of pixels), and a stack of photos is a 4D tensor.
- **Reference Term:** [04-Tensors_and_Shapes.md](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/04-Tensors_and_Shapes.md)

---

### Dot Product (Frobenius Inner Product)
- **Spoken / Phonetic:** /dɒt ˈprɒd.ʌkt/ (DOT PROD-ukt)
- **Formal Definition:** The scalar inner product between two tensors of matching shape: $\langle \mathbf{A}, \mathbf{B} \rangle_F = \sum_{c, i, j} A_{c, i, j} B_{c, i, j}$, evaluating the degree of directional alignment between the input receptive field and the filter weights.
- **Intuitive Explanation:** Multiplying corresponding numbers between a visual patch and a filter, then adding them up to measure how closely the patch matches the template.
- **Reference Term:** [03-Dot_Product_and_Similarity.md](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/03-Dot_Product_and_Similarity.md)

---

### Rectified Linear Unit (ReLU)
- **Spoken / Phonetic:** /ˈrɛk.tɪ.faɪd ˈlɪn.i.ər ˈjuː.nɪt/ (REK-tih-fyde LIN-ee-er YOO-nit)
- **Formal Definition:** A non-linear element-wise activation function defined as $f(z) = \max(0, z)$, possessing a constant subgradient of 1 for $z > 0$ and 0 for $z < 0$.
- **Intuitive Explanation:** A simple threshold switch that lets positive signals pass through unchanged while completely blocking negative signals, eliminating vanishing gradients in deep networks.
- **Reference Term:** [05-Activation_Functions.md](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/05-Activation_Functions.md)

---

### Softmax Function
- **Spoken / Phonetic:** /ˈsɒft.mæks/ (SOFT-maks)
- **Formal Definition:** A normalized exponential map $\sigma: \mathbb{R}^K \to \Delta^{K-1}$ converting unnormalized real-valued logits $\mathbf{z}$ into a valid probability distribution: $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$.
- **Intuitive Explanation:** Turning arbitrary network output numbers into clean percentage probabilities that sum to 100%.
- **Reference Term:** [06-Softmax.md](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/06-Softmax.md)

---

### Cross-Entropy Loss
- **Spoken / Phonetic:** /krɒs ˈɛn.trə.pi lɒs/ (KROSS EN-truh-pee LOSS)
- **Formal Definition:** The empirical risk objective for categorical distribution fitting: $\mathcal{L}_{\text{CE}}(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{k=1}^K y_k \log \hat{y}_k$, penalizing divergence between ground-truth labels and predicted class probabilities.
- **Intuitive Explanation:** A mathematical penalty score that heavily punishes the model when it is confident in the wrong answer.
- **Reference Term:** [08-Loss_Functions.md](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/08-Loss_Functions.md)

---

### Backpropagation & Chain Rule
- **Spoken / Phonetic:** /ˌbæk.prɒp.əˈɡeɪ.ʃən/ (back-prop-uh-GAY-shun)
- **Formal Definition:** Reverse-mode automatic differentiation applying the multivariable calculus chain rule to recursively compute partial derivatives of a scalar loss function $\mathcal{L}$ with respect to all internal tensor parameters.
- **Intuitive Explanation:** Tracing errors backward from the final prediction through every layer of the network to determine exactly how each weight should be adjusted.
- **Reference Term:** [04-Chain_Rule_and_Backpropagation.md](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/04-Chain_Rule_and_Backpropagation.md)

---

### Receptive Field
- **Spoken / Phonetic:** /rɪˈsɛp.tɪv fiːld/ (rih-SEP-tiv FEELD)
- **Formal Definition:** The sub-lattice region $\mathcal{R} \subset \Omega_{\text{in}}$ of input space that influences the activation of a particular neuron in layer $l$, expanding linearly with network depth: $RF_l = RF_{l-1} + (F_l - 1)$.
- **Intuitive Explanation:** How much of the original photograph a single artificial brain cell can see.
- **Reference Term:** [01-Convolution_and_Pooling.md](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md)
