# References & Pedagogical Resources: Multi-Channel Convolutions & Deep Vision Architectures

This curated bibliography provides formal academic citations, seminal papers, foundational textbooks, and interactive visualizers directly supporting **Lecture 44: Convolutional Neural Networks (CNNs) as Regularized MLPs**.

---

## 1. Curriculum & Prerequisite Bridges

1. **Lecture 41: Neural Networks & Universal Approximation Theorem**
   - **Internal Link:** [`54-Lec41-Neural-Networks-UAT`](../54-Lec41-Neural-Networks-UAT/NOTES.md) — **Why Read This:** Explains the foundational theoretical UAT guarantee and connects existential representation with empirical optimization constraints.

2. **Lecture 42: ERM on Neural Networks & Error Backpropagation**
   - **Internal Link:** [`55-Lec42-ERM-Neural-Networks-Backpropagation`](../55-Lec42-ERM-Neural-Networks-Backpropagation/NOTES.md) — **Why Read This:** Explains reverse-mode automatic differentiation and derives the multi-variable chain rule on DAGs.

3. **Lecture 43: Local Receptive Fields & Parameter Sharing**
   - **Internal Link:** [`56-Lec43-Local-Receptive-Field-Parameter-Sharing`](../56-Lec43-Local-Receptive-Field-Parameter-Sharing/NOTES.md) — **Why Read This:** Explains 1D local receptive fields and parameter tying as the core foundational priors of CNNs.

4. **Core Math Concept: Convolution and Pooling Operators**
   - **Internal Link:** [`Convolution & Pooling`](../../MathsTerms/06-Deep-Architectures-and-Generative-Models/01-Convolution_and_Pooling.md) — **Why Read This:** Provides reference tensor formulas, multi-channel marginalization, and subgradient pooling rules.

---

## 2. Foundational & Seminal Papers

5. **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).**
   - **Paper Link:** [Gradient-Based Learning Applied to Document Recognition (LeNet-5)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) — **Why Read This:** Foundational paper establishing modern convolutional networks with alternating convolutions and pooling layers.

6. **Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).**
   - **Paper Link:** [ImageNet Classification with Deep CNNs (AlexNet)](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) — **Why Read This:** Landmark paper proving empirical supremacy of deep CNNs trained on GPUs with ReLU activations.

7. **Simonyan, K., & Zisserman, A. (2014).**
   - **Paper Link:** [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)](https://arxiv.org/abs/1409.1556) — **Why Read This:** Explains why stacking small 3x3 convolutions reduces parameters while increasing non-linear representation depth.

8. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).**
   - **Paper Link:** [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385) — **Why Read This:** Proves identity residual skip connections solve vanishing gradients in very deep networks.

9. **Szegedy, C., et al. (2015).**
   - **Paper Link:** [Going Deeper with Convolutions (Inception/GoogLeNet)](https://arxiv.org/abs/1409.4842) — **Why Read This:** Demonstrates 1x1 convolutions for channel dimensionality reduction and multi-scale feature extraction.

10. **Ronneberger, O., Fischer, P., & Brox, T. (2015).**
    - **Paper Link:** [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — **Why Read This:** The seminal foundation for encoder-decoder networks with lateral skip connections for pixel-wise dense prediction.

11. **Long, J., Shelhamer, E., & Darrell, T. (2015).**
    - **Paper Link:** [Fully Convolutional Networks for Semantic Segmentation (FCN)](https://arxiv.org/abs/1411.4038) — **Why Read This:** Demonstrates end-to-end pixel-wise dense semantic prediction via transposed convolutions.

---

## 3. Textbooks & Video Lectures

12. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press.
    - **Book Link:** [Deep Learning Book Chapter 9 (Convolutional Networks)](https://www.deeplearningbook.org/contents/convnets.html) — **Why Read This:** Essential textbook chapter detailing spatial cross-correlation, parameter sharing, and equivariance proofs.

13. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning.* Springer.
    - **Book Link:** [Pattern Recognition and Machine Learning (Chapter 5: Neural Networks)](https://www.microsoft.com/en-us/research/publication/pattern-recognition-and-machine-learning/) — **Why Read This:** Rigorous foundation formulating neural network parameter estimation, Bayesian priors, and structural weight constraints.

14. **Dumoulin, V., & Visin, F. (2016).**
    - **Guide Link:** [A Guide to Convolution Arithmetic for Deep Learning](https://arxiv.org/abs/1603.07285) — **Why Read This:** The definitive mathematical guide for spatial dimension arithmetic across stride, padding, dilation, and transposed convolutions.

---

## 4. Industry & Implementation Guides

15. **PyTorch Documentation: `torch.nn.Conv2d` & `torch.nn.ConvTranspose2d`**
    - **Guide Link:** [PyTorch Conv2d Documentation](https://pytorch.org/docs/stable/nn.html) — **Why Read This:** Production documentation explaining tensor shape expectations `[B, C_in, H, W]`, weight memory formats, and cuDNN acceleration.

16. **NVIDIA cuDNN Developer Guide: 2D Convolution Algorithms**
    - **Guide Link:** [NVIDIA cuDNN Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html) — **Why Read This:** Explains high-performance GPU implementations of 2D convolutions via GEMM (im2col), Winograd filtering, and FFT algorithms.

---

## 5. Interactive Visualizers & Simulations

17. **CNN Explainer (Georgia Tech & Poloclub)**
    - **Tool Link:** [Interactive CNN Explainer](https://poloclub.github.io/cnn-explainer/) — **Why Read This:** Real-time browser-based interactive 3D visualization showing exact multi-channel inner products and tensor activations.

18. **Convolution Arithmetic Animations (Vincent Dumoulin)**
    - **Tool Link:** [Convolution Arithmetic Animations](https://github.com/vdumoulin/conv_arithmetic) — **Why Read This:** Demonstrates animated visual mechanics of stride, padding, dilation, and transposed convolutions.
