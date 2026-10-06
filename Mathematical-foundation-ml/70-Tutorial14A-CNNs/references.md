# References & Academic Citations — Tutorial 14 Part 1: CNNs

This document contains authoritative primary literature, milestone computer vision papers, foundational textbooks, and curriculum cross-references supporting **Tutorial 14 Part 1: Convolutional Neural Networks (CNNs)**.

---

## 1. Primary Academic Literature

1. **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).**  
   *Gradient-Based Learning Applied to Document Recognition.*  
   *Proceedings of the IEEE*, 86(11), 2278–2324.  
   - [Paper PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
   - **Why Read This:** Essential historical paper introducing the LeNet-5 architecture ($28 \times 28 \times 1$ MNIST). It establishes the standard computer vision paradigm of alternating convolutional feature extraction, subsampling (pooling), and dense classification heads.

2. **Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).**  
   *ImageNet Classification with Deep Convolutional Neural Networks.*  
   *Advances in Neural Information Processing Systems (NeurIPS 2012)*, 25, 1097–1105.  
   - [NeurIPS Paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
   - **Why Read This:** The foundational AlexNet paper that launched modern deep learning by winning ImageNet by 10.8 percentage points. Demonstrates GPU training, ReLU non-linearities, overlapping pooling, and dropout.

3. **Simonyan, K., & Zisserman, A. (2014).**  
   *Very Deep Convolutional Networks for Large-Scale Image Recognition.*  
   *International Conference on Learning Representations (ICLR 2015)*.  
   - [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
   - **Why Read This:** Introduces VGG-16 and VGG-19, proving mathematically and empirically that stacks of small $3 \times 3$ filters outperform large kernels while reducing parameter count by 28% and increasing non-linear depth.

4. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).**  
   *Deep Residual Learning for Image Recognition.*  
   *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016)*, 770–778.  
   - [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
   - **Why Read This:** Landmark paper introducing residual learning (ResNet) with identity shortcuts $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$, resolving the degradation problem and allowing training of 152+ layer models.

5. **Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009).**  
   *ImageNet: A Large-Scale Hierarchical Image Database.*  
   *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2009)*, 248–255.  
   - [CVPR Paper](https://ieeexplore.ieee.org/document/5206848)
   - **Why Read This:** Foundational benchmark establishing the 1,000-class dataset and standardized $224 \times 224 \times 3$ RGB image resolution for visual representation evaluation.

6. **Zeiler, M. D., & Fergus, R. (2014).**  
   *Visualizing and Understanding Convolutional Networks.*  
   *European Conference on Computer Vision (ECCV 2014)*, 818–833.  
   - [arXiv:1311.2901](https://arxiv.org/abs/1311.2901)
   - **Why Read This:** Visualizes internal representations with deconvolutional networks, proving that early layers detect edges and textures while deeper layers detect semantic object parts.

7. **Lin, M., Chen, Q., & Yan, S. (2013).**  
   *Network In Network.*  
   *arXiv preprint arXiv:1312.4400*.  
   - [arXiv:1312.4400](https://arxiv.org/abs/1312.4400)
   - **Why Read This:** Introduces $1 \times 1$ convolutions for cross-channel pooling and Global Average Pooling (GAP) to replace parameter-heavy dense classifier heads.

8. **Szegedy, C., et al. (2015).**  
   *Going Deeper with Convolutions.*  
   *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2015)*, 1–9.  
   - [arXiv:1409.4842](https://arxiv.org/abs/1409.4842)
   - **Why Read This:** Introduces GoogLeNet (Inception), processing visual inputs across multiple parallel filter resolutions simultaneously.

---

## 2. Textbooks & Authoritative Course Notes

9. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).**  
   *Deep Learning.* MIT Press.  
   - [Deep Learning Book Chapter 9](https://www.deeplearningbook.org/contents/convnets.html)
   - **Why Read This:** Chapter 9 provides the definitive mathematical treatment of discrete cross-correlation, sparse connectivity, weight sharing, and equivariant representations.

10. **Karpathy, A., Johnson, J., & Fei-Fei, L. (2016).**  
    *CS231n: Convolutional Neural Networks for Visual Recognition.* Stanford University.  
    - [Stanford CS231n Module](https://cs231n.github.io/convolutional-networks/)
    - **Why Read This:** Canonical interactive lecture notes and animated visualizers walking through the exact $5 \times 5 \times 3$ stride/padding arithmetic referenced in lecture.

11. **Paszke, A., Gross, S., Massa, F., et al. (2019).**  
    *PyTorch: An Imperative Style, High-Performance Deep Learning Library.*  
    *NeurIPS 2019*, 32, 8024–8035.  
    - [PyTorch Conv2d API Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    - **Why Read This:** Official production specifications and tensor shape formulas for `torch.nn.Conv2d`, `torch.nn.AvgPool2d`, and `torch.nn.Sequential`.

---

## 3. Lecture Timestamp Cross-References

| Concept | Video Timestamp | Mathematical & Architectural Details |
|:---|:---|:---|
| **Local Receptive Fields & Weight Sharing** | 00:01–01:29 | CNNs as regularized MLPs constraining dense weights to spatial neighborhoods |
| **Filter Depth Matching Input Channels** | 01:29–02:57 | Kernel volume $F \times F \times D_{\text{in}}$ matching input channel depth |
| **Zero-Padding Boundary Preservation** | 02:57–03:52 | Padding spatial edges with $P$ zeros to avoid resolution decay |
| **CS231n Animated Convolution Walkthrough** | 03:52–07:33 | Sliding filter demonstration across $5 \times 5 \times 3$ input volume |
| **Spatial Output Dimension Formula** | 07:33–09:55 | Exact derivation of $H_{\text{out}} = \lfloor \frac{H - F + 2P}{S} \rfloor + 1$ and $D_{\text{out}} = K$ |
| **Numerical Hand Calculation (-6 Result)** | 09:55–12:19 | Step-by-step element-wise summation: $(-3) + (-1) + (-2) + 0 = -6$ |
| **Spatial Pooling (Max vs Average)** | 13:06–15:10 | $2 \times 2$ stride 2 downsampling halving spatial area by 75% |
| **Decoupling Features & Classifier Head** | 15:10–16:36 | Decoupling representation trunk from task-specific linear heads |
| **LeNet-5 Architectural Transitions** | 17:03–21:15 | Detailed breakdown of Yann LeCun's 1998 MNIST network |
| **PyTorch Implementation of LeNet-5** | 21:15–28:08 | Writing modular `nn.Sequential` code with `nn.Conv2d` and `nn.AvgPool2d` |
| **Sigmoid to ReLU Modernization** | 25:46–26:43 | Mitigating vanishing gradients in multi-stage networks |
| **ImageNet, AlexNet, VGG & ResNet** | 29:09–33:52 | Scaling up to 1,000 classes, GPU acceleration, and residual skip connections |

---

## 4. Curriculum Cross-References & Bridges

- **Preceding Milestone (Package 68: Lec 52 Transfer Learning & Distillation):**  
  Establishes how representations learned by large models transfer to downstream tasks; reinforced in this lecture by decoupling `self.features` for sequence models (e.g. image captioning).
- **Preceding Milestone (Package 69: Lec 53 Optimizers):**  
  Establishes Adam and momentum dynamics used directly in this lecture's PyTorch LeNet-5 training loop (`torch.optim.Adam(lr=1e-3)`).
- **Succeeding Milestone (Package 71: Tutorial 14 Part 2 Transfer Learning with CNNs):**  
  Applies the CNN architectures introduced here (AlexNet, VGG, ResNet) with pretrained weights from `torchvision.models`, parameter freezing (`requires_grad = False`), and classification head fine-tuning.
- **Sequential Modeling Milestone (Package 72: Tutorial 15 Part 1 RNNs & LSTMs):**  
  Pairs the convolutional embedding vectors $Z$ extracted here with recurrent networks to process temporal visual sequences and generate textual captions.
