# References & Further Reading

> **Package:** `56-Lec43-Local-Receptive-Field-Parameter-Sharing`  
> **Course:** NPTEL / IISc — Mathematical Foundations of Machine Learning  
> **Instructor:** Prof. Prathosh A P (IISc Bengaluru)  
> **Skill Standard:** Canonical 7-Pillar Production Learning Suite (`/youtube-lecture-tutor`)

---

## 1. Internal Curriculum Bridges & Sibling Tracks

1. **Lecture 41: Neural Networks and Universal Approximation Theorem**
   - **Pointer:** [`../54-Lec41-Neural-Networks-UAT/NOTES.md`](../54-Lec41-Neural-Networks-UAT/NOTES.md)
   - **Why Read This:** Establishes the theoretical baseline that a single hidden layer can approximate any continuous function, motivating why architectural restrictions (such as local receptive fields) are necessary for optimization feasibility rather than existential expressivity.

2. **Lecture 42: ERM on Neural Networks and Error Backpropagation**
   - **Pointer:** [`../55-Lec42-ERM-Neural-Networks-Backpropagation/NOTES.md`](../55-Lec42-ERM-Neural-Networks-Backpropagation/NOTES.md)
   - **Why Read This:** Provides the formal multivariable calculus chain rule and error sensitivity derivations ($\delta_j \equiv \frac{\partial \mathcal{L}}{\partial z_j}$) that are extended in this lecture to accumulate gradients over shared convolutional weights.

3. **Lecture 44: Convolutional Neural Networks as Regularized MLPs**
   - **Pointer:** `[Planned: 57-Lec44-CNNs-as-Regularized-MLP]`
   - **Why Read This:** Direct continuation of this lecture, formally proving that 2D convolution is an MLP with doubly-blocked circulant / Toeplitz weight matrices subjected to infinite $L_2$ structural regularization.

4. **Lecture 33 & 34: Regularization and MAP Parameter Estimation**
   - **Pointer:** [`../46-Lec33-Regularization/`](../46-Lec33-Regularization/) & [`../47-Lec34-MAP-Estimation/`](../47-Lec34-MAP-Estimation/)
   - **Why Read This:** Grounds the mathematical equivalence between structural architectural constraints (cutting connections) and singular Dirac delta parameter priors $p(w) = \delta_0(w)$.

5. **MathsTerms Reference Track: Convolutions and Pooling**
   - **Pointer:** [`../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md`](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md)
   - **Why Read This:** Provides rapid reference mathematical definitions, tensor shapes, and 2D discrete filtering formulae.

---

## 2. Seminal Academic Literature

6. **Hubel, D. H., & Wiesel, T. N. (1962).** *Receptive fields, binocular interaction and functional architecture in the cat's visual cortex.* The Journal of Physiology, 160(1), 106–154.
   - **Why Read This:** Nobel Prize-winning physiology work establishing that biological neurons in primary visual cortex (V1) have localized receptive fields that fire specifically for oriented light bars and edges.

7. **Fukushima, K. (1980).** *Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position.* Biological Cybernetics, 36(4), 193–202.
   - **Why Read This:** The foundational architectural predecessor to modern CNNs, introducing alternating layers of S-cells (local receptive field feature extraction) and C-cells (spatial pooling for shift invariance).

8. **LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W. E., & Jackel, L. D. (1989).** *Backpropagation applied to handwritten zip code recognition.* Neural Computation, 1(4), 541–551.
   - **Why Read This:** The seminal paper demonstrating that combining local receptive fields, shared weights, and backpropagation solves real-world handwriting recognition (MNIST) with orders of magnitude fewer parameters than fully connected networks.

9. **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).** *Gradient-based learning applied to document recognition.* Proceedings of the IEEE, 86(11), 2278–2324.
   - **Why Read This:** The canonical LeNet-5 architecture reference detailing modern convolutional layers, parameter sharing mechanics, and spatial subsampling.

10. **Zhou, D. X. (2020).** *Universality of deep convolutional neural networks.* Applied and Computational Harmonic Analysis, 48(2), 787–794.
    - **Why Read This:** Rigorous mathematical proof establishing the Universal Approximation Theorem for deep CNNs, proving that weight sharing and local receptive fields do not sacrifice continuous function density.

11. **Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017).** *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* IEEE ICCV, 618–626.
    - **Why Read This:** The primary empirical attribution method cited in lecture for visualizing which spatial receptive fields and feature maps cause a deep network to fire.

---

## 3. Textbooks & Monographs

12. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press.
    - **Chapter 9: Convolutional Networks.**
    - **Why Read This:** The gold-standard textbook treatment of sparse interactions, parameter sharing, translation equivariance proofs, and efficient tensor multidimensional convolutions.

13. **Bishop, C. M., & Bishop, H. (2023).** *Deep Learning: Foundations and Concepts.* Springer.
    - **Chapter 10: Convolutional Networks.**
    - **Why Read This:** Rigorous modern treatment deriving 1D and 2D convolution from continuous Lie group translation symmetries and reproducing kernel Hilbert spaces.

14. **Gonzalez, R. C., & Woods, R. E. (2018).** *Digital Image Processing (4th Edition).* Pearson.
    - **Chapter 3 & 10: Spatial Filtering and Edge Detection.**
    - **Why Read This:** Details the classical hand-engineered spatial filters (Sobel, Prewitt, Laplacian, Canny, Gabor wavelets) discussed by Prof. Prathosh as human-crafted precursors to learnable CNN kernels.

---

## 4. Industry & Framework Engineering Guides

15. **PyTorch Engineering Guide: `torch.nn.Conv1d` and `torch.nn.Conv2d` Internals**
    - **URL:** [pytorch.org/docs/stable/generated/torch.nn.Conv2d.html](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    - **Why Read This:** Explains how production deep learning engines lower discrete 2D convolutions into dense GEMM matrix multiplications via `im2col` (image-to-column) memory transformations.

---

## 5. Interactive Visualizers & Educational Tools

16. **Dumoulin, V., & Visin, F. (2016).** *A guide to convolution arithmetic for deep learning.*
    - **URL:** [github.com/vdumoulin/conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic)
    - **Why Read This:** The definitive animated visual guide illustrating receptive fields, strides, valid vs same padding, and transposed convolutions.
