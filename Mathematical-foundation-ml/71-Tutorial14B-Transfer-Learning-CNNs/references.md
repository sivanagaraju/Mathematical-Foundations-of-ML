# References & Academic Citations — Tutorial 14 Part 2: Transfer Learning using CNNs

This document contains authoritative primary literature, landmark transfer learning papers, vision backbone citations, and curriculum cross-references supporting **Tutorial 14 Part 2: Transfer Learning using CNNs**.

---

## 1. Primary Academic Literature

1. **Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014).**  
   *How transferable are features in deep neural networks?*  
   *Advances in Neural Information Processing Systems (NeurIPS 2014)*, 27, 3320–3328.  
   - [NeurIPS Paper](https://proceedings.neurips.cc/paper/2014/file/375c71349b295fbe2dcdca9206f20a06-Paper.pdf)
   - **Why Read This:** Landmark empirical study quantifying the transferability of features across layers. Demonstrates that early convolutional layers learn universal Gabor-like filters that transfer across domains, while deeper layers become task-specific.

2. **Simonyan, K., & Zisserman, A. (2014).**  
   *Very Deep Convolutional Networks for Large-Scale Image Recognition.*  
   *International Conference on Learning Representations (ICLR 2015)*.  
   - [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
   - **Why Read This:** Details the VGG-16 and VGG-19 architectures (configurations A through E) and establishes homogeneous $3 \times 3$ convolutional stacks; discusses depth limitations up to 19 layers.

3. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).**  
   *Deep Residual Learning for Image Recognition.*  
   *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016)*, 770–778.  
   - [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
   - **Why Read This:** Introduces ResNet-18, 34, 50, and 152 with residual identity skip connections ($\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$), resolving the vanishing gradient problem and enabling stable optimization of deep networks.

4. **Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015).**  
   *Show and Tell: A Neural Image Caption Generator.*  
   *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2015)*, 3156–3164.  
   - [arXiv:1411.4555](https://arxiv.org/abs/1411.4555)
   - **Why Read This:** Seminal multimodal paper demonstrating decoupled embedding extraction: routing a pretrained CNN's latent visual representation $Z$ into an LSTM decoder for natural language sentence generation.

5. **Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009).**  
   *ImageNet: A Large-Scale Hierarchical Image Database.*  
   *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2009)*, 248–255.  
   - [CVPR Paper](https://ieeexplore.ieee.org/document/5206848)
   - **Why Read This:** Details the 1.4-million-image, 1,000-class benchmark dataset and $224 \times 224$ RGB standardization that underpins all vision backbones used in transfer learning.

6. **Lin, M., Chen, Q., & Yan, S. (2013).**  
   *Network In Network.*  
   *arXiv preprint arXiv:1312.4400*.  
   - [arXiv:1312.4400](https://arxiv.org/abs/1312.4400)
   - **Why Read This:** Introduces Global Average Pooling (GAP), which replaces parameter-heavy dense classifier heads with parameter-free spatial averaging, adopted directly by ResNet.

7. **Kornblith, S., Shlens, J., & Le, Q. V. (2019).**  
   *Do Better ImageNet Models Transfer Better?*  
   *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2019)*, 2673–2682.  
   - [arXiv:1905.05142](https://arxiv.org/abs/1905.05142)
   - **Why Read This:** Demonstrates strong empirical correlation between ImageNet top-1 accuracy and linear probe transfer performance across 12 downstream vision datasets.

---

## 2. Textbooks & Authoritative Course Notes

8. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).**  
   *Deep Learning.* MIT Press.  
   - [Deep Learning Book Chapter 15: Representation Learning](https://www.deeplearningbook.org/contents/representation.html)
   - **Why Read This:** In-depth discussion of inductive transfer, shared representations, and domain adaptation theory.

9. **Karpathy, A., Johnson, J., & Fei-Fei, L. (2016).**  
   *CS231n: Transfer Learning and Fine-tuning Convolutional Networks.* Stanford University.  
   - [CS231n Transfer Learning Guide](https://cs231n.github.io/transfer-learning/)
   - **Why Read This:** Practical guidelines on when to freeze backbones versus fine-tune, categorized by target dataset size and domain similarity.

---

## 3. Industry & Implementation Guides

10. **PyTorch Official Engineering Team (2024).**  
    *Transfer Learning for Computer Vision Tutorial.*  
    - [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
    - **Why Read This:** Canonical implementation guide illustrating how to freeze feature extractor layers, swap out classifier heads, and schedule learning rates during fine-tuning.

11. **Torchvision Documentation (2024).**  
    *Torchvision Models and Pretrained Weights.*  
    - [Torchvision Models Hub](https://pytorch.org/vision/stable/models.html)
    - **Why Read This:** Comprehensive API reference for instantiating VGG, ResNet, EfficientNet backbones with `weights=...DEFAULT`.

---

## 4. Interactive Visualizers & Demos

12. **Harley, A. W. (2015).**  
    *An Interactive Node-Link Visualization of Convolutional Neural Networks.*  
    - [Interactive 3D CNN Visualization](https://adamharley.com/nn_vis/)
    - **Why Read This:** Interactive browser demo visualizing 2D convolutional feature maps, activation volumes, and dense classification layers in real time.

---

## 3. Lecture Timestamp Cross-References

| Concept | Video Timestamp | Mathematical & Architectural Details |
|:---|:---|:---|
| **Pretrained Model Paradigm & VGG Family** | 00:06–03:59 | Downloading published ImageNet weights; VGG configurations A to E (11 to 19 layers) |
| **VGG 19-Layer Depth Limit & ResNet Skips** | 03:59–05:56 | Vanishing gradient ceiling in sequential convs; ResNet residual identity pathways $x + F(x)$ |
| **ResNet-18/34 Layer Count Breakdown** | 06:25–08:20 | 4 stages $\times$ 2 residual blocks (16 convs) + initial conv + FC = 18 layers |
| **Loading Models & Weights in PyTorch** | 08:51–10:47 | Using `torchvision.models` with `weights=...DEFAULT`; placing models in evaluation mode `eval()` |
| **Forward Pass & Top-5 Evaluation** | 11:17–14:10 | Generating 1,000 logits, computing softmax posteriors, and extracting top-5 candidates via `torch.topk` |
| **Inductive Transfer Foundations** | 15:08–15:36 | Low-level universal primitives (edges, corners) transferring across disparate optical domains |
| **Bicycle to Motorbike Analogy** | 15:36–16:05 | Direct transfer of existing balancing and spatial motor skills to novel vehicle tasks |
| **VGG Classifier Head Surgery** | 16:35–18:09 | Modifying layer index 6: `vgg.classifier[6] = nn.Linear(4096, num_classes)` |
| **ResNet Classifier Head Surgery** | 18:37–19:34 | Modifying `resnet.fc = nn.Linear(512, num_classes)` |
| **Parameter Freezing (requires_grad)** | 20:31–21:02 | Setting `requires_grad = False` to lock feature extraction trunk during head training |
| **Non-Standard Spatial Resizing** | 21:58–22:54 | Managing dimension mismatches when feeding non-$224 \times 224$ images into VGG/ResNet |
| **Decoupled Embedding Extraction** | 23:00–27:12 | Decoupling representation trunks to extract 25,088-d (VGG) and 512-d (ResNet) latent vectors $Z$ |
| **Image Captioning & Sequence Bridge** | 23:21–24:21 | Pipelining visual embeddings into sequence-to-sequence models (RNN/LSTM) for caption decoders |
| **Lego-Block Architecture Philosophy** | 27:12–28:37 | Assembling modular vision systems by combining standard trunks, pooling blocks, and custom heads |

---

## 4. Curriculum Cross-References & Bridges

- **Preceding Milestone (Package 68: Lec 52 Transfer Learning & Distillation):**  
  Establishes the theoretical foundations of representation transfer and domain adaptation applied practically here.
- **Preceding Milestone (Package 70: Tutorial 14 Part 1 CNNs):**  
  Constructs the fundamental discrete convolution, pooling, and LeNet-5 building blocks scaled here to ImageNet.
- **Succeeding Milestone (Package 72: Tutorial 15 Part 1 RNNs, LSTMs & GRUs):**  
  Directly consumes the decoupled visual embeddings $Z \in \mathbb{R}^{512}$ generated here to train recurrent sequence-to-sequence models for automated image caption generation.
- **Succeeding Milestone (Package 73: Tutorial 15 Part 2 Deep RNNs):**  
  Explores deep stacked recurrent networks paired with convolutional encoders for video classification and temporal action recognition.
