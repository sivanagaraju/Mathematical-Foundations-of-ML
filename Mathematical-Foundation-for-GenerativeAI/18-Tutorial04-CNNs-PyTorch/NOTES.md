# Tutorial 4 — CNNs using PyTorch

**Video:** [Tutorial 4 : CNNs using PyTorch](https://www.youtube.com/watch?v=BhnGtsMwUCU) · NPTEL / IISc  
**Warm-up First:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous Tutorial:** [Tutorial 3 — PyTorch Basics](../17-Tutorial03-PyTorch-Basics/NOTES.md) (Tensors, Autograd, `nn.Module`, and Optimization Loops)  
**Next Tutorial:** Tutorial 5 — Recurrent Neural Networks (RNNs, LSTMs & GRUs)  
**Course:** Mathematical Foundations of Generative AI (~39 min)  
**Speaker:** NPTEL / IISc Teaching Team  
**Core Themes:** 4D NCHW Tensor Standard, Discrete 2D Convolution & Local Receptive Fields, Multi-Channel Filter Banks & Parameter Counting, Spatial Arithmetic & The Output Size Formula, Parameter-Free Max Pooling, The Feature Extractor Backbone vs Classifier Head Pattern, Dummy Forward Pass Debugging, MNIST Torchvision Pipelines (Resize & Normalize), Mini-Batch Training with Adam & Cross-Entropy, and Deterministic Model Evaluation.

---

> ### ⚠️ Course Context & Curriculum Progression Notice
> In **Tutorial 3**, the curriculum established the foundational PyTorch deep learning workflow on 1D vector data using Multi-Layer Perceptrons (MLPs).
> 
> Starting in **Tutorial 4**, the course scales to **Spatial and Visual Intelligence** via **Convolutional Neural Networks (CNNs)**. High-resolution images ($\mathbf{X} \in \mathbb{R}^{C \times H \times W}$) cannot be efficiently processed by flat linear layers due to parameter explosion and loss of 2D locality. CNNs solve this by introducing **local receptive fields**, **weight sharing**, and **spatial downsampling**, serving as the architectural backbone for:
> 1. **Computer Vision Classifiers & Detectors (ResNet, ConvNeXt, YOLO)**.
> 2. **Generative Latent Diffusion U-Nets & VAE Encoders (Stable Diffusion, Midjourney, FLUX)**.
> 3. **Spatial Feature Tokenizers in Multi-Modal Vision-Language Models (CLIP, LLaVA)**.

---

## Table of Contents

1. [Executive Summary & Master Architecture](#executive-summary--architecture-of-this-lecture)
2. [Chalkboard Rosetta Stone: Mathematical & Vision Notation](#chalkboard-rosetta-stone)
3. [Complete Standalone Executable PyTorch Simulation Script](#standalone-simulation-script)
4. [Topic 1: Roadmap & The 4D NCHW Tensor Standard (00:02–02:00)](#topic-1-roadmap-nchw-batch-0002–0200)
5. [Topic 2: Convolution Mechanics & The CS231n Padding Demo (02:00–05:25)](#topic-2-conv-gif-padding-0200–0525)
6. [Topic 3: Stride, Parameter Counting & Weight Sharing (05:25–09:20)](#topic-3-stride-weights-sharing-0525–0920)
7. [Topic 4: Output Size Formula & `nn.Conv2d` Module API (09:20–12:24)](#topic-4-output-formula-conv2d-0920–1224)
8. [Topic 5: Parameter-Free Downsampling — `nn.MaxPool2d` (12:24–16:00)](#topic-5-maxpool2d-1224–1600)
9. [Topic 6: SimpleCNN Architecture & Feature Map Tracing (16:00–20:18)](#topic-6-simplecnn-stack-shapes-1600–2018)
10. [Topic 7: Flattening, Fully Connected Head & Dummy Forward Pass (20:18–24:25)](#topic-7-flatten-fc-dummy-forward-2018–2425)
11. [Topic 8: MNIST Pipeline — Torchvision Transforms & DataLoaders (24:25–30:08)](#topic-8-mnist-transforms-loaders-2425–3008)
12. [Topic 9: Mini-Batch Training Loop with Adam & CrossEntropy (30:08–33:55)](#topic-9-mini-batch-train-loop-3008–3355)
13. [Topic 10: Evaluation Mode, Accuracy Scoring & Sequence Roadmap (33:55–39:29)](#topic-10-eval-accuracy-recap-3355–3929)
14. [Workplace Debugging Postmortems](#workplace-debugging-postmortems)
15. [Centralized External References](#external-references)

---

## Executive Summary — Architecture of this Lecture

<a id="executive-summary--architecture-of-this-lecture"></a>

This 39-minute tutorial transitions deep learning from flat 1D vectors to 2D image grids, establishing the mathematical, structural, and software principles of Convolutional Neural Networks (CNNs).

### System Context

```
  ╔═══════════════════════════════════════════════════════════════════════════════════════╗
  ║                          COMPUTER VISION ARCHITECTURE STACK                           ║
  ╚═══════════════════════════════════════════════════════════════════════════════════════╝
                                              │
         ┌────────────────────────────────────┴────────────────────────────────────┐
         ▼                                                                         ▼
  [Tutorial 3: Flat Vector MLPs]                                        [Tutorial 4: 2D Spatial CNNs]
  • 1D Input Vectors x ∈ ℝᴰ                                             • 4D Image Tensors X ∈ ℝ^(N×C×H×W)
  • Dense Fully Connected Layers (nn.Linear)                            • 2D Cross-Correlation (nn.Conv2d)
  • Loss of 2D spatial pixel locality                                   • Translation Invariance & Weight Sharing
  • Massive parameter explosion (O(H·W·D))                              • Compact Parameter Efficiency (O(F²·Cin·Cout))
  • Flat Classification Heads                                           • Feature Backbone (Conv+Pool) + Linear Head
                                              │
                                              ▼
                         [Upcoming Tutorials: Sequence & Generative AI]
                         • Tutorial 5: Recurrent Neural Networks (RNNs, LSTMs, GRUs)
                         • Generative AI: U-Net Denoising Backbones in Diffusion Models
                         • Vision Transformers (ViT) & Multi-Modal Embeddings (CLIP)
```

---

### Master Architecture Blueprint

```
  ===================================================================================================
                                      TUTORIAL 4 MASTER ARCHITECTURE
  ===================================================================================================
  
   [Image Input Tensor]                  [Feature Extractor Backbone]        [Classifier Head]
     Batch of Images (N, C, H, W)          Stage 1:                            Flatten:
     • N = Batch Size (e.g. 64)            • Conv2d(3, 16, k=3, p=1, s=1)      • (N, 32, 8, 8) ──► (N, 2048)
     • C = Channels (1 Gray, 3 RGB)          Output: (N, 16, 32, 32)                   │
     • H, W = Height & Width (32x32)       • ReLU Activation                           ▼
            │                              • MaxPool2d(2, stride=2)            Linear Head:
            ▼                                Output: (N, 16, 16, 16)           • Linear(2048, 128) + ReLU
   [Spatial Arithmetic Engine]             Stage 2:                            • Linear(128, 10) (Logits!)
     H_out = floor((H - F + 2P)/S) + 1     • Conv2d(16, 32, k=3, p=1, s=1)             │
     W_out = floor((W - F + 2P)/S) + 1       Output: (N, 32, 16, 16)                   │
     Params = F·F·Cin·Cout + Cout          • ReLU Activation                           │
            │                              • MaxPool2d(2, stride=2)                    │
            │                                Output: (N, 32, 8, 8)                     │
            └──────────────────────────────────────┼───────────────────────────────────┘
                                                   ▼
                                      [Data & Training Pipeline]
                                        • Dataset: torchvision.datasets.MNIST
                                        • Transforms: transforms.Compose([Resize(32), ToTensor()])
                                        • DataLoaders: Train (batch=64, shuffle=True), Test (batch=64)
                                        • Loss Criterion: nn.CrossEntropyLoss(logits, targets)
                                        • Optimizer: optim.Adam(model.parameters(), lr=1e-3)
                                        • Mini-Batch Updates: ~938 updates per epoch (N=60,000)
                                        • Evaluation: model.eval() + torch.no_grad() + argmax accuracy
  ===================================================================================================
```

---

### Comparative Feature Matrices

#### Table 1: Fully Connected MLPs vs Convolutional Neural Networks (CNNs)

| Characteristic | Fully Connected MLP (`nn.Linear`) | Convolutional Neural Network (`nn.Conv2d`) |
| :--- | :--- | :--- |
| **Input Structure** | 1D Flattened Vector $\mathbf{x} \in \mathbb{R}^{D}$ | 4D Spatial Tensor $\mathbf{X} \in \mathbb{R}^{N \times C \times H \times W}$ |
| **Spatial Locality** | Destroys 2D pixel adjacency upon flattening | Preserves local pixel relationships via $F \times F$ sliding windows |
| **Weight Mechanism** | Every input-output pixel pair has an independent weight | **Weight Sharing:** The same filter kernel slides across the entire image |
| **Parameter Scaling** | $\mathcal{O}(H \cdot W \cdot D_{\text{out}})$ (Explodes with image size) | $\mathcal{O}(F^2 \cdot C_{\text{in}} \cdot C_{\text{out}})$ (Independent of image resolution) |
| **Translation Invariance** | None (Shifting an object changes all activations) | **Translation Equivariant:** Sliding the object shifts the activation map |

---

#### Table 2: Convolutional Hyperparameters & Geometry Effects

| Hyperparameter | Symbol | Typical Values | Physical Effect on Feature Map | Impact on Parameter Count |
| :--- | :--- | :--- | :--- | :--- |
| **Kernel Size** | $F$ | $3 \times 3, 5 \times 5, 7 \times 7$ | Defines local receptive field window size | Increases quadratically ($F^2$) |
| **Padding** | $P$ | $0$ (Valid), $1$ (Same for $3\times3$) | Prevents border shrinkage; preserves spatial size | **Zero** impact on parameters |
| **Stride** | $S$ | $1$ (Dense), $2$ (Downsampling) | Controls step size; downsamples spatial resolution | **Zero** impact on parameters |
| **Input Channels** | $C_{\text{in}}$ | $1$ (Grayscale), $3$ (RGB), $16, 32$ | Defines 3D depth of each individual filter | Scales linearly ($C_{\text{in}}$) |
| **Output Channels** | $C_{\text{out}}$ or $K$ | $16, 32, 64, 128, 256$ | Number of parallel filters; feature diversity | Scales linearly ($C_{\text{out}}$) |

---

#### Table 3: Pooling Mechanisms

| Pooling Type | Mathematical Operation | Learnable Parameters | Primary Use Case in Vision |
| :--- | :--- | :--- | :--- |
| **Max Pooling (`MaxPool2d`)** | $\max_{u, v} X_{i+u, j+v}$ | **0** | Sharp feature extraction, translation robustness, downsampling |
| **Average Pooling (`AvgPool2d`)** | $\frac{1}{F^2} \sum_{u, v} X_{i+u, j+v}$ | **0** | Smooth background aggregation, feature softening |
| **Global Average Pooling (`AdaptiveAvgPool2d`)** | Spatial average across entire map $\to 1 \times 1$ | **0** | Replacing large flattened linear heads in modern architectures |

---

### Failure & Contrast Paths (6 Common Engineering Traps)

```
  [Engineering Trap 1: "Feeding HWC Image into Conv2d without Permute"]
  TRAP: Passing OpenCV/PIL image of shape (Height, Width, Channels) into Conv2d.
  REALITY: PyTorch expects (Batch, Channels, Height, Width). Passing HWC misinterprets Height as Channels.
  FIX: Always execute img.permute(2, 0, 1).unsqueeze(0) to create valid NCHW format.
  
  [Engineering Trap 2: "The Flatten Dimension Mismatch Crash"]
  TRAP: Guessing the in_features dimension for the first Linear layer in the classifier head.
  REALITY: If spatial height/width after pooling does not match the product C_final * H_final * W_final,
           the linear layer throws a shape mismatch RuntimeError during matrix multiplication.
  FIX: Perform a dummy forward pass with torch.randn(1, C, H, W) to programmatically verify flattened size.
  
  [Engineering Trap 3: "Grayscale Input to RGB Conv2d Channel Mismatch"]
  TRAP: Passing 1-channel MNIST images into a Conv2d(in_channels=3, ...) layer.
  REALITY: PyTorch raises RuntimeError: Given groups=1, weight of size [16, 3, 3, 3], expected input[1, 1, 32, 32]
           to have 3 channels, but got 1 channels instead.
  FIX: Set in_channels=1 in the first Conv2d layer, or repeat channels via x.repeat(1, 3, 1, 1).
  
  [Engineering Trap 4: "Forgetting Padding and Accidentally Shrinking Images"]
  TRAP: Stacking multiple Conv2d(..., kernel_size=3, padding=0) layers without pooling.
  REALITY: Each layer shaves off 2 pixels of height and width (32 -> 30 -> 28 -> 26), truncating spatial borders.
  FIX: Set padding=1 for 3x3 kernels (padding = (F - 1) / 2) to maintain "Same" spatial resolution.
  
  [Engineering Trap 5: "The 'Epoch Equals One Update' Fallacy"]
  TRAP: Assuming that one training epoch executes one single gradient descent update.
  REALITY: One epoch iterates over all N / B mini-batches (e.g. 938 updates per epoch on MNIST with batch 64).
  FIX: Track loss and updates per mini-batch inside the loader loop.
  
  [Engineering Trap 6: "Leaving Autograd Enabled at Evaluation Time"]
  TRAP: Running model evaluation without with torch.no_grad(): and model.eval().
  REALITY: Autograd builds massive computation graphs for the entire test set, causing GPU Out-Of-Memory crashes.
  FIX: Always pair model.eval() with with torch.no_grad(): during testing and inference.
```

---

## Chalkboard Rosetta Stone

This reference table maps deep learning vision symbols directly to PyTorch implementations and lecture usage.

| Symbol / Syntax | Formal Concept | PyTorch Implementation | Lecture Usage & Context | Dedicated MathsTerm Guide |
| :--- | :--- | :--- | :--- | :--- |
| $\mathbf{X} \in \mathbb{R}^{N \times C \times H \times W}$ | 4D Mini-Batch Tensor | `x = torch.randn(N, C, H, W)` | Standard 4D image layout ($N$=batch, $C$=channels, $H$=height, $W$=width). | [Tensors & Shapes](../../../MathsTerms/Tensors_and_Shapes.md) |
| $\mathbf{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times F \times F}$ | 4D Convolutional Weight Bank | `conv.weight` | Learnable filter weights initialized inside `nn.Conv2d`. | [Convolution & Pooling](../../../MathsTerms/Convolution_and_Pooling.md) |
| $F$ | Spatial Kernel Size | `kernel_size=3` | Height and width of the square sliding window. | [Convolution & Pooling](../../../MathsTerms/Convolution_and_Pooling.md) |
| $P$ | Zero-Padding Halo | `padding=1` | Zero-pixel ring added to image borders to preserve spatial size. | [Convolution & Pooling](../../../MathsTerms/Convolution_and_Pooling.md) |
| $S$ | Stride Step Size | `stride=1` or `stride=2` | Pixel step size when sliding the filter window. | [Convolution & Pooling](../../../MathsTerms/Convolution_and_Pooling.md) |
| $H_{\text{out}}$ | Output Spatial Height | $\lfloor \frac{H_{\text{in}} - F + 2P}{S} \rfloor + 1$ | Height of the resulting feature map. | [Convolution & Pooling](../../../MathsTerms/Convolution_and_Pooling.md) |
| $\text{MaxPool2d}(2, 2)$ | Spatial Halving Downsampler | `nn.MaxPool2d(2, 2)` | Parameter-free operator reducing height and width by $50\%$. | [Convolution & Pooling](../../../MathsTerms/Convolution_and_Pooling.md) |
| $\text{Flatten}$ | Multi-Axis Unrolling | `torch.flatten(x, 1)` | Flattens 3D feature cube into 1D vector ($C \cdot H \cdot W$). | [Tensors & Shapes](../../../MathsTerms/Tensors_and_Shapes.md) |
| $\mathbf{z} \in \mathbb{R}^{10}$ | Unnormalized Class Logits | `logits = model(x)` | 10-class scores fed directly to `nn.CrossEntropyLoss`. | [Softmax](../../../MathsTerms/Softmax.md) |
| $\hat{c} = \arg\max_k z_k$ | Hard Predicted Class | `torch.argmax(logits, dim=1)` | Predicted class index used for accuracy evaluation. | [Argmax & Argmin](../../../MathsTerms/Argmax.md) |

---

## Complete Standalone Executable PyTorch Simulation Script

<a id="standalone-simulation-script"></a>

Below is a self-contained, end-to-end Python script implementing all concepts taught in Tutorial 4: 4D NCHW tensor creation, manual cross-correlation verification, spatial output size formulas, parameter count calculations, MaxPool2d downsampling, full SimpleCNN model building, dummy forward pass verification, synthetic MNIST data pipeline with transforms, mini-batch training with Adam and Cross-Entropy, and evaluation accuracy scoring.

```python
"""
Tutorial 04: CNNs using PyTorch — Master Executable Simulation Script
Validated on Python 3.10+, PyTorch 2.0+, and CUDA / CPU backends.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def run_tutorial_04_simulation():
    print("=" * 80)
    print("TUTORIAL 04: CONVOLUTIONAL NEURAL NETWORKS (CNNs) MASTER SIMULATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. HARDWARE DEVICE CONFIGURATION
    # ---------------------------------------------------------
    print("\n[1] Environment & Hardware Device Configuration")
    print(f"  PyTorch Version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Selected Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU Device Name: {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------
    # 2. 4D NCHW TENSOR CONTRACT & GEOMETRY
    # ---------------------------------------------------------
    print("\n[2] 4D NCHW Tensor Contract")
    # Batch of 4 RGB images of size 32x32
    x_batch = torch.randn(4, 3, 32, 32)
    print(f"  Input Batch Shape (N, C, H, W): {x_batch.shape}")
    print(f"  Batch Size (N): {x_batch.shape[0]} | Channels (C): {x_batch.shape[1]}")
    print(f"  Spatial Height (H): {x_batch.shape[2]} | Width (W): {x_batch.shape[3]}")
    assert x_batch.shape == torch.Size([4, 3, 32, 32])

    # ---------------------------------------------------------
    # 3. SPATIAL ARITHMETIC & OUTPUT SIZE FORMULA VALIDATION
    # ---------------------------------------------------------
    print("\n[3] Spatial Output Size Formula Validation")
    def compute_conv_output_dim(dim_in, kernel_f, pad_p, stride_s):
        return int((dim_in - kernel_f + 2 * pad_p) // stride_s + 1)

    # Case A: Same-Padding (32, F=3, P=1, S=1) -> 32
    h_out_same = compute_conv_output_dim(32, 3, 1, 1)
    conv_same = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    out_same = conv_same(x_batch)
    print(f"  Formula Same-Padding H_out: {h_out_same} | Actual Tensor H_out: {out_same.shape[2]}")
    assert h_out_same == 32 and out_same.shape[2] == 32

    # Case B: No Padding Shrinkage (32, F=3, P=0, S=1) -> 30
    h_out_valid = compute_conv_output_dim(32, 3, 0, 1)
    conv_valid = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
    out_valid = conv_valid(x_batch)
    print(f"  Formula Valid-Padding H_out: {h_out_valid} | Actual Tensor H_out: {out_valid.shape[2]}")
    assert h_out_valid == 30 and out_valid.shape[2] == 30

    # ---------------------------------------------------------
    # 4. FILTER BANK PARAMETER COUNTING
    # ---------------------------------------------------------
    print("\n[4] Conv2d Learnable Parameter Counting")
    # Layer: Conv2d(in=3, out=16, kernel=3, bias=True)
    # Theory: (3 * 3 * 3 * 16) + 16 = 432 + 16 = 448
    conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    weight_params = conv_layer.weight.numel() # 16 * 3 * 3 * 3 = 432
    bias_params = conv_layer.bias.numel()     # 16
    total_params = sum(p.numel() for p in conv_layer.parameters())
    print(f"  Conv2d(3, 16, 3) Weights: {weight_params} | Biases: {bias_params} | Total: {total_params} (Theory: 448)")
    assert total_params == 448

    # ---------------------------------------------------------
    # 5. MAXPOOL2D PARAMETER-FREE DOWNSAMPLING
    # ---------------------------------------------------------
    print("\n[5] MaxPool2d Spatial Downsampling (32x32 -> 16x16)")
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    pooled_out = pool(out_same)
    print(f"  Pre-pool Shape:  {out_same.shape}")
    print(f"  Post-pool Shape: {pooled_out.shape} (Channels Unchanged: 16)")
    print(f"  MaxPool Parameters: {sum(p.numel() for p in pool.parameters())} (Theory: 0)")
    assert pooled_out.shape == torch.Size([4, 16, 16, 16])
    assert sum(p.numel() for p in pool.parameters()) == 0

    # ---------------------------------------------------------
    # 6. SIMPLECNN MODEL DEFINITION & DUMMY FORWARD PASS
    # ---------------------------------------------------------
    print("\n[6] SimpleCNN Architecture & Dummy Forward Pass")
    class SimpleCNN(nn.Module):
        def __init__(self, in_channels=3, num_classes=10):
            super().__init__()
            # Feature Extractor Backbone
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Classifier Head
            # Spatial dimensions: 32 -> pool1 -> 16 -> pool2 -> 8 -> 32 * 8 * 8 = 2048
            self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=128)
            self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
            
        def forward(self, x):
            x = self.pool1(F.relu(self.conv1(x))) # (N, 16, 16, 16)
            x = self.pool2(F.relu(self.conv2(x))) # (N, 32, 8, 8)
            x = torch.flatten(x, start_dim=1)     # (N, 2048)
            x = F.relu(self.fc1(x))               # (N, 128)
            logits = self.fc2(x)                  # (N, 10) - Unnormalized Logits
            return logits

    model = SimpleCNN(in_channels=3, num_classes=10).to(device)
    dummy_input = torch.randn(4, 3, 32, 32, device=device)
    dummy_logits = model(dummy_input)
    print(f"  Dummy Forward Input: {dummy_input.shape} -> Output Logits: {dummy_logits.shape}")
    assert dummy_logits.shape == torch.Size([4, 10])

    # ---------------------------------------------------------
    # 7. DATASET PIPELINE & DATALOADER BATCHING
    # ---------------------------------------------------------
    print("\n[7] Synthetic MNIST 32x32 Dataset Pipeline")
    # Generate 640 synthetic RGB 32x32 samples across 10 classes
    torch.manual_seed(42)
    X_train = torch.randn(640, 3, 32, 32)
    y_train = torch.randint(0, 10, (640,))
    X_test = torch.randn(128, 3, 32, 32)
    y_test = torch.randint(0, 10, (128,))

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f"  Train Samples: {len(train_dataset)} | Batches per Epoch: {len(train_loader)}")
    print(f"  Test Samples:  {len(test_dataset)}  | Test Batches: {len(test_loader)}")

    # ---------------------------------------------------------
    # 8. MINI-BATCH TRAINING LOOP WITH ADAM & CROSS-ENTROPY
    # ---------------------------------------------------------
    print("\n[8] Mini-Batch Training Loop (5 Epochs)")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Step 1: Zero Gradients
            optimizer.zero_grad()
            
            # Step 2: Forward Pass
            logits = model(images)
            
            # Step 3: Compute Cross-Entropy Loss
            loss = criterion(logits, labels)
            
            # Step 4: Backward Pass (Autograd)
            loss.backward()
            
            # Step 5: Optimizer Parameter Update
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
        epoch_loss = running_loss / total_train
        epoch_acc = (correct_train / total_train) * 100.0
        print(f"  Epoch [{epoch+1:02d}/{epochs:02d}] | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

    # ---------------------------------------------------------
    # 9. EVALUATION MODE & TEST ACCURACY
    # ---------------------------------------------------------
    print("\n[9] Evaluation Mode & Test Accuracy")
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            
            test_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)

    final_test_loss = test_loss / total_test
    final_test_acc = (correct_test / total_test) * 100.0
    print(f"  Evaluation Results -> Test Loss: {final_test_loss:.4f} | Test Acc: {final_test_acc:.2f}%")

    print("\n" + "=" * 80)
    print("ALL TUTORIAL 04 SIMULATION BLOCKS EXECUTED & VERIFIED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_tutorial_04_simulation()
```

---

## Topic 1: Roadmap & The 4D NCHW Tensor Standard (00:02–02:00)

<a id="topic-1-roadmap-nchw-batch-0002–0200"></a>
<a id="topic-1-roadmap-nchw-batch-0002-0200"></a>

### Where this sits on the master map
Transitioning from flat 1D vectors to 4D spatial tensors in PyTorch. Warm-up: [NCHW layout](./PREREQUISITES.md#p1-nchw).

### Board / Screenshot Reference

![Roadmap NCHW](./screenshots/composites/ch01-topic-01-roadmap-nchw-panel1of1.png)

*Figure — ~00:02–02:00: Blackboard presentation of the curriculum roadmap (PyTorch Basics $\to$ CNNs $\to$ RNNs/LSTMs $\to$ Pre-trained Models), creating a dummy 4D batch tensor `x = torch.randn(4, 3, 32, 32)`, and establishing the NCHW axis ordering contract.*

---

### 1. 👶 ELI5 Quick Intuition
Imagine holding a physical stack of 4 printed photograph cards:
- **$N = 4$:** You hold a stack of 4 distinct cards (Batch Size).
- **$C = 3$:** Each photo card is printed using 3 colored transparencies (Red, Green, Blue Channels).
- **$H = 32$:** Each card is 32 pixel rows tall.
- **$W = 32$:** Each card is 32 pixel columns wide.
- In PyTorch, every image operation requires this exact filing order: **$N \to C \to H \to W$**.

---

### 2. 🔍 Plain-English Breakdown
1. **The Vision Curriculum Progression:**
   - Tutorial 3: Tensors, Autograd, `nn.Module`, and 1D MLPs.
   - **Tutorial 4 (Today):** 2D Convolutional Neural Networks, Spatial Downsampling, and MNIST Image Classification.
   - Upcoming: Sequence Modeling (RNNs, LSTMs, GRUs) and Pretrained Vision Backbones.
2. **The 4D Tensor Contract:**
   - PyTorch `nn.Conv2d` layers strictly require 4D tensors formatted as `(Batch, Channels, Height, Width)`.
   - Creating a dummy random batch:
     ```python
     x = torch.randn(4, 3, 32, 32)
     ```
3. **Axis Semantic Mapping:**
   - `x.shape[0] = 4`: Number of images processed in parallel per forward pass.
   - `x.shape[1] = 3`: Color channels ($C=1$ for grayscale, $C=3$ for RGB).
   - `x.shape[2] = 32`: Spatial height.
   - `x.shape[3] = 32`: Spatial width.

---

### 3. 📐 Formal Mathematics & 4D Coordinate Tensor Geometry

```
  4D Mini-Batch Tensor Memory Layout: X ∈ ℝ^(N × C × H × W)
  
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  Batch Index n = 0, ..., N-1                                                │
  │    └── Channel c = 0, ..., C-1 (e.g. R, G, B)                               │
  │          └── Height h = 0, ..., H-1 (Spatial Rows)                          │
  │                └── Width w = 0, ..., W-1 (Spatial Columns)                  │
  └─────────────────────────────────────────────────────────────────────────────┘
  
  Physical Stride Vector in C-Contiguous Memory:
  s = ( C·H·W,  H·W,  W,  1 )
  MemoryOffset(n, c, h, w) = n·(C·H·W) + c·(H·W) + h·(W) + w
```

---

### 4. 🎯 Why We Are Doing This Example & What We Are Learning
- **Why start with a dummy random batch `x = torch.randn(4, 3, 32, 32)`?**  
  In deep learning engineering, starting with real data is a common anti-pattern. Real datasets involve disk I/O, download latency, decoding, and parsing. By creating an isolated dummy tensor first, we build immediate mental clarity on tensor dimensions and verify layer shape contracts in milliseconds before introducing data pipeline complexity.
- **What are we learning?**  
  We are learning that PyTorch vision layers do not operate on raw 2D pixel grids; they operate on 4D tensor manifolds where hardware-accelerated matrix multiplication kernels (cuDNN) expect batch and channel axes to precede spatial geometry.

---

### 5. 🔗 Connecting the Dots (To Next Steps & Generative AI)
- **Bridge to Latent Diffusion (Stable Diffusion / FLUX):**  
  In modern Generative AI, a diffusion model does not diffuse pixels directly; it encodes an image $\mathbf{X} \in \mathbb{R}^{3 \times 512 \times 512}$ using a Convolutional VAE Encoder into a compact latent tensor $\mathbf{z}_0 \in \mathbb{R}^{4 \times 64 \times 64}$. The latent representation retains this exact 4D NCHW tensor structure throughout all multi-scale denoising steps!

---

### 6. 🌐 Real-World Production Usage & Industry Scenarios
- **Autonomous Driving Perception Systems (Tesla Autopilot, Waymo):**  
  Vehicles capture feeds from multiple surround cameras simultaneously. A perception model ingests a 4D batch tensor where $N = 8$ camera views, $C = 3$ color channels, $H = 1080$, and $W = 1920$ pixels, processing all camera feeds in parallel on high-throughput automotive GPUs.

---

## Topic 2: Convolution Mechanics & The CS231n Padding Demo (02:00–05:25)

<a id="topic-2-conv-gif-padding-0200–0525"></a>
<a id="topic-2-conv-gif-padding-0200-0525"></a>

### Where this sits on the master map
Understanding the spatial sliding window, elementwise multiplication, channel summation, and boundary zero-padding. Warm-up: [convolution mechanics](./PREREQUISITES.md#p2-conv).

### Board / Screenshot Reference

![Conv GIF padding](./screenshots/composites/ch02-topic-02-conv-gif-padding-panel1of1.png)

*Figure — ~02:00–05:25: Blackboard presentation of the Stanford CS231n convolution animation: sliding a $3 \times 3 \times 3$ filter cube across an RGB image with zero-padding $P=1$, computing dot products, and generating a 2D feature map.*

---

### 1. 👶 ELI5 Quick Intuition
Think of a 3D rubber stamp with 3 ink layers (Red, Green, Blue):
- You press the stamp onto a $3 \times 3$ patch of a color photograph.
- The stamp multiplies the photo pixels by the rubber patterns on all 3 color layers, sums all the numbers together, adds 1 bias number, and prints **1 single number on an output page**.
- You slide the stamp right and down across the entire photo until you have filled an entire page (a **Feature Map**).
- **Padding:** If the stamp cannot reach the outer borders of the photo, you tape a ring of white cardboard (zeros) around the edges!

---

### 2. 🔍 Plain-English Breakdown
1. **The 3D Nature of Filters:**
   - Although called "2D Convolution", each filter is physically a **3D volume** of shape $(C_{\text{in}}, F, F)$.
   - For an RGB image ($C_{\text{in}} = 3$) with a $3 \times 3$ kernel, a single filter contains $3 \times 3 \times 3 = 27$ weights.
2. **The Local Dot Product:**
   - At each spatial location $(i, j)$, the 27 weights multiply the matching 27 image pixels.
   - All 27 products are summed together into a single scalar value.
   - A single scalar bias $b$ is added to form output pixel $Y_{i, j}$.
3. **Zero-Padding ($P$):**
   - Padding surrounds the image perimeter with $P$ rows and columns of zeros.
   - *Padding $P=1$ on a $3 \times 3$ filter:* Allows the filter center to sit on the very first pixel $(0, 0)$ of the image, preventing border information loss.

---

### 3. 📐 Formal Mathematics & Multi-Channel Cross-Correlation

```
  3D Filter Sliding Cube on Multi-Channel Image:
  
  Input Image (Cin = 3 Channels)           3D Filter Kernel (3 x 3 x 3)         Single Feature Map Cell
  ┌──────┐ ┌──────┐ ┌──────┐               ┌──────┐ ┌──────┐ ┌──────┐           ┌───┐
  │Red   │ │Green │ │Blue  │               │W_Red │ │W_Grn │ │W_Blu │           │   │
  │3x3   │ │3x3   │ │3x3   │       *       │3x3   │ │3x3   │ │3x3   │   ───►    │Yij│ = ∑(X·W) + b
  │Patch │ │Patch │ │Patch │               │Kernel│ │Kernel│ │Kernel│           │   │
  └──────┘ └──────┘ └──────┘               └──────┘ └──────┘ └──────┘           └───┘
     │        │        │                      │        │        │
     └────────┼────────┴──────────────────────┴────────┴────────┘
              ▼
   Dot product evaluated across ALL 3 channels simultaneously ──► Summed to 1 scalar!
```

#### Multi-Channel Discrete Cross-Correlation Formula
$$\mathbf{Y}_{i, j} = \left( \sum_{c=0}^{C_{\text{in}}-1} \sum_{u=0}^{F-1} \sum_{v=0}^{F-1} \mathbf{X}_{c, \, i+u, \, j+v} \cdot \mathbf{W}_{c, u, v} \right) + b$$

---

### 4. 🎯 Why We Are Doing This Example & What We Are Learning
- **Why demonstrate the CS231n animated GIF?**  
  Static math formulas obscure the dynamic mechanical reality of convolutions. The animation visually proves that a filter does not look at one channel in isolation—it penetrates the entire depth of the input volume simultaneously, producing a 2D scalar map that highlights where a specific pattern exists.
- **What are we learning?**  
  We are learning that convolutions preserve 2D topology. Unlike fully connected layers that treat pixel $(0, 0)$ and pixel $(100, 100)$ as unrelated abstract features, convolutions exploit the natural statistical property of images: *nearby pixels are strongly correlated*.

---

### 5. 🔗 Connecting the Dots (To Next Steps & Generative AI)
- **Bridge to Feature Extraction Hierarchies:**  
  A single convolution layer acts as an elementary edge detector. By stacking multiple convolution layers, the network combines edges into textures, textures into object parts, and object parts into high-level semantic concepts. This exact feature hierarchy enables text-to-image models to translate prompt tokens into spatial artwork.

---

### 6. 🌐 Real-World Production Usage & Industry Scenarios
- **Medical Imaging (Tumor Segmentation in Radiology):**  
  MRI scans contain subtle tissue density variations. Convolutional filters slide across 2D slices to highlight micro-calcifications, tumor margins, and vascular structures with high spatial precision.

---

## Topic 3: Stride, Parameter Counting & Weight Sharing (05:25–09:20)

<a id="topic-3-stride-weights-sharing-0525–0920"></a>
<a id="topic-3-stride-weights-sharing-0525-0920"></a>

### Where this sits on the master map
Analyzing stride step jumps, parameter efficiency via weight sharing, and multi-filter channel depth. Warm-up: [filter banks](./PREREQUISITES.md#p3-filters).

### Board / Screenshot Reference

![Stride and weights](./screenshots/composites/ch03-topic-03-stride-weights-sharing-panel1of1.png)

*Figure — ~05:25–09:20: Blackboard derivation of filter parameter counting ($F \times F \times C_{\text{in}} \times C_{\text{out}} + C_{\text{out}}$), stride step skipping ($S=1$ vs $S=2$), and explaining why weight sharing achieves parameter efficiency.*

---

### 1. 👶 ELI5 Quick Intuition
Imagine two photographers scanning a crowd:
- **Photographer 1 (Stride 1):** Takes a step of 1 inch for every single photo (Dense, overlapping coverage).
- **Photographer 2 (Stride 2):** Takes a leap of 2 inches for every photo (Cuts the number of photos in half, capturing a quick overview).
- **Weight Sharing:** Both photographers use the **exact same lens pattern** to look at the top-left, center, and bottom-right of the crowd, rather than buying a separate lens for every spot!

---

### 2. 🔍 Plain-English Breakdown
1. **Stride ($S$):**
   - $S = 1$: Sliding window moves 1 pixel per step (preserves spatial resolution).
   - $S = 2$: Sliding window moves 2 pixels per step (downsamples spatial area by $4\times$).
2. **Filter Bank ($C_{\text{out}}$ or $K$):**
   - One filter creates one 2D feature map.
   - To extract $K$ distinct visual features (vertical edges, horizontal edges, color contrasts), the layer contains $K$ independent filters.
   - The output tensor has channel depth equal to $C_{\text{out}} = K$.
3. **The Parameter Counting Formula:**
   $$\text{Total Parameters} = \underbrace{(F \times F \times C_{\text{in}} \times C_{\text{out}})}_{\text{Weights}} + \underbrace{C_{\text{out}}}_{\text{Biases}}$$

---

### 3. 📐 Formal Mathematics & Comparison of Parameter Scaling

```
  =============================================================================
               PARAMETER COMPARISON: DENSE LINEAR VS CONVOLUTION
  =============================================================================
  Input: 32x32 RGB Image (3,072 Inputs)  ──►  Target Output: 16 Feature Maps
  
  [Dense Fully Connected Layer]
  • Every pixel connects to every output: 3,072 × (32 × 32 × 16)
  • Parameter Count = 50,331,648 weights! (Massive overfitting, no locality)
  
  [Convolutional Layer (nn.Conv2d(3, 16, kernel_size=3))]
  • Stencil Size: 3 × 3
  • Weight Count: 3 × 3 × 3 × 16 = 432 weights
  • Bias Count:   16 biases
  • Total Parameters = 448 parameters! (112,000x MORE COMPACT!)
  =============================================================================
```

---

### 4. 🎯 Why We Are Doing This Example & What We Are Learning
- **Why calculate exact parameter counts on the chalkboard?**  
  Engineers frequently confuse image resolution with parameter count. Calculating $F \times F \times C_{\text{in}} \times C_{\text{out}} + C_{\text{out}}$ proves that a CNN's parameter count is **completely independent of input image height and width**! The same 448 parameters can process a $32 \times 32$ icon or a $4096 \times 4096$ satellite photograph.
- **What are we learning?**  
  We are learning the profound power of **Weight Sharing** and **Translation Equivariance**: an edge detector learned in the top-left corner is automatically reused across the entire image.

---

### 5. 🔗 Connecting the Dots (To Next Steps & Generative AI)
- **Bridge to Efficient Diffusion Architectures:**  
  Generative models require processing millions of pixels. Because convolutional layers share weights across space, diffusion U-Nets can maintain deep multi-scale processing pipelines within the memory budget of consumer GPUs.

---

### 6. 🌐 Real-World Production Usage & Industry Scenarios
- **Edge AI & Mobile Vision (Apple Neural Engine, Qualcomm Snapdragon):**  
  Compact parameter counts allow mobile devices to run real-time face tracking, depth estimation, and photo enhancement directly on device without sending private user images to cloud servers.

---

## Topic 4: Output Size Formula & `nn.Conv2d` Module API (09:20–12:24)

<a id="topic-4-output-formula-conv2d-0920–1224"></a>
<a id="topic-4-output-formula-conv2d-0920-1224"></a>

### Where this sits on the master map
Mastering the exact spatial geometry output formula and instantiating `nn.Conv2d` layers. Warm-up: [output formula](./PREREQUISITES.md#p4-shape).

### Board / Screenshot Reference

![Output formula Conv2d](./screenshots/composites/ch04-topic-04-output-formula-conv2d-panel1of1.png)

*Figure — ~09:20–12:24: Blackboard presentation of the master spatial output formula $H_{\text{out}} = \lfloor \frac{H_{\text{in}} - F + 2P}{S} \rfloor + 1$, verifying output dimensions on $32 \times 32$ inputs, and declaring `nn.Conv2d(3, 16, kernel_size=3, padding=1)`.*

---

### 1. 👶 ELI5 Quick Intuition
Think of cutting a long ribbon of fabric:
- You have a ribbon $H_{\text{in}}$ inches long.
- You glue $P$ inches of extra cloth to both ends ($+2P$).
- You use a cutting blade $F$ inches wide and slide it along by steps of $S$ inches.
- The **Output Size Formula** tells you the exact number of pieces you will cut out!

---

### 2. 🔍 Plain-English Breakdown
1. **The Spatial Dimension Formula:**
   $$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - F + 2P}{S} \right\rfloor + 1$$
   $$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - F + 2P}{S} \right\rfloor + 1$$
2. **Standard Same-Padding Configuration:**
   - Setting $F = 3, P = 1, S = 1$:
     $$H_{\text{out}} = \frac{H - 3 + 2(1)}{1} + 1 = H$$
   - Spatial height and width are perfectly preserved!
3. **`nn.Conv2d` Constructor Arguments:**
   ```python
   nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
   ```

---

### 3. 📐 Formal Mathematics & Spatial Geometry Proof

```
  Spatial Layout Arithmetic:
  
  Input Grid Dimension:    |--- H_in ---|
  With Zero-Padding 2P:    |P|--- H_in ---|P|
  Window Placement:        [--- F ---] --> Step S --> [--- F ---]
  
  Number of steps before window exceeds right boundary:
  (H_out - 1) · S + F ≤ H_in + 2P
  H_out = floor( (H_in - F + 2P) / S ) + 1
```

---

### 4. 🎯 Why We Are Doing This Example & What We Are Learning
- **Why derive this formula explicitly?**  
  In vision engineering, dimension mismatch is the #1 cause of runtime crashes when connecting convolutional backbones to linear classification heads. Memorizing and validating this formula allows you to mentally trace tensor shapes through arbitrary 50-layer networks without running trial-and-error scripts.
- **What are we learning?**  
  We are learning how to control spatial resolution: choosing when to preserve resolution (for fine feature detection) versus when to compress resolution (for computational efficiency).

---

### 5. 🔗 Connecting the Dots (To Next Steps & Generative AI)
- **Bridge to U-Net Skip Connections in Diffusion:**  
  U-Net generative architectures downsample images in the encoder and upsample them in the decoder. For skip connections to concatenate tensors without errors, the encoder and decoder feature maps at matching levels must have *identical spatial dimensions*, requiring exact padding and stride arithmetic!

---

### 6. 🌐 Real-World Production Usage & Industry Scenarios
- **Satellite Earth Observation (Wildfire & Flood Mapping):**  
  Satellites process massive multi-spectral geotiffs. Exact spatial formulas ensure that predicted disaster masks overlay onto GPS coordinates with sub-pixel alignment accuracy.

---

## Topic 5: Parameter-Free Downsampling — `nn.MaxPool2d` (12:24–16:00)

<a id="topic-5-maxpool2d-1224–1600"></a>
<a id="topic-5-maxpool2d-1224-1600"></a>

### Where this sits on the master map
Downsampling spatial feature dimensions, controlling memory footprint, and expanding receptive fields without adding parameters. Warm-up: [max pooling](./PREREQUISITES.md#p5-pool).

### Board / Screenshot Reference

![MaxPool2d](./screenshots/composites/ch05-topic-05-maxpool-panel1of1.png)

*Figure — ~12:24–16:00: Blackboard presentation of `nn.MaxPool2d(kernel_size=2, stride=2)`: halving spatial dimensions $32 \times 32 \to 16 \times 16$, leaving channels unchanged, and proving zero learnable parameter overhead.*

---

### 1. 👶 ELI5 Quick Intuition
Think of summarizing a high-resolution map:
- You divide a $32 \times 32$ grid into $2 \times 2$ blocks of 4 pixels.
- In each block, you take the highest mountain peak (the Maximum value) and discard the other 3 pixels.
- The map is now **$16 \times 16$ (half the height and width)**, but all major landmarks are preserved!
- It costs **zero weights and zero memory** to learn!

---

### 2. 🔍 Plain-English Breakdown
1. **`nn.MaxPool2d(kernel_size=2, stride=2)`:**
   - Evaluates non-overlapping $2 \times 2$ windows across each feature map.
   - Extracts the maximum activation in each window.
2. **Key Invariants of Max Pooling:**
   - **Spatial Resolution:** Exactly halved ($H \to H/2, W \to W/2$).
   - **Channel Depth:** Completely unchanged ($C \to C$).
   - **Parameter Count:** Identical to zero ($\text{Weights} = 0$).

---

### 3. 📐 Formal Mathematics & Max-Pooling Derivation

```
  2x2 Max-Pooling Window Operation:
  
  Input Feature Map (4x4)                       Output Feature Map (2x2)
  ┌───────┬───────┬───────┬───────┐             ┌───────┬───────┐
  │  1.2  │ [3.8] │  2.1  │ [4.5] │    ───►     │  3.8  │  4.5  │
  ├───────┼───────┼───────┼───────┤             ├───────┼───────┤
  │  0.5  │  2.0  │  1.0  │  0.2  │             │  7.1  │  8.4  │
  ├───────┼───────┼───────┼───────┤             └───────┴───────┘
  │ [7.1] │  5.4  │  3.0  │  1.2  │
  ├───────┼───────┼───────┼───────┤
  │  6.0  │  1.8  │  0.0  │ [8.4] │
  └───────┴───────┴───────┴───────┘
```

$$\mathbf{Y}_{c, i, j} = \max_{u \in \{0, 1\}, \, v \in \{0, 1\}} \mathbf{X}_{c, \, 2i+u, \, 2j+v}$$

---

### 4. 🎯 Why We Are Doing This Example & What We Are Learning
- **Why use Max Pooling instead of strided convolutions?**  
  Max pooling provides a deterministic, parameter-free mechanism for dimensionality reduction. It enforces local translation invariance (a slight shift of 1 pixel in the input does not alter the maximum value) while dramatically reducing memory consumption for downstream layers.
- **What are we learning?**  
  We are learning how receptive fields expand: each pooled pixel in deeper layers represents a larger spatial region of the original input image.

---

### 5. 🔗 Connecting the Dots (To Next Steps & Generative AI)
- **Bridge to Receptive Field Hierarchies:**  
  Pooling expands the **effective receptive field**. Layer 1 sees $3 \times 3$ patches of raw pixels; after pooling, Layer 2 sees $7 \times 7$ patches; after a second pool, Layer 3 sees $15 \times 15$ patches. This enables deep networks to understand full-object context without needing giant $15 \times 15$ filter kernels.

---

### 6. 🌐 Real-World Production Usage & Industry Scenarios
- **Biometric Security & Facial Recognition (FaceID):**  
  When a user unlocks their phone, their face is never in the exact identical pixel position. Max pooling ensures that facial landmark features (eyes, nose bridge, jawline) produce consistent activation signals despite minor head rotations or shifts.

---

## Topic 6: SimpleCNN Architecture & Feature Map Tracing (16:00–20:18)

<a id="topic-6-simplecnn-stack-shapes-1600–2018"></a>
<a id="topic-6-simplecnn-stack-shapes-1600-2018"></a>

### Where this sits on the master map
Assembling a multi-stage convolutional feature extractor and tracing tensor dimensions through successive layers. Warm-up: [backbone and head](./PREREQUISITES.md#p6-head).

### Board / Screenshot Reference

![SimpleCNN stack](./screenshots/composites/ch06-topic-06-simplecnn-stack-panel1of1.png)

*Figure — ~16:00–20:18: Blackboard walkthrough of the SimpleCNN feature backbone: tracing tensor shapes $(N, 3, 32, 32) \to (N, 16, 16, 16) \to (N, 32, 8, 8)$ through two stages of Conv $\to$ ReLU $\to$ MaxPool.*

---

### 1. 👶 ELI5 Quick Intuition
Think of a factory assembly line with two inspection stations:
- **Station 1:** 16 inspectors scan the $32 \times 32$ image for simple edges, then shrink the report to $16 \times 16$.
- **Station 2:** 32 senior detectives scan the $16 \times 16$ reports for complex shapes (loops, corners), then shrink the final dossier to $8 \times 8$.
- The final output is a rich dossier of **32 feature maps, each $8 \times 8$ pixels wide**!

---

### 2. 🔍 Plain-English Breakdown
1. **Stage 1 Pipeline:**
   - Input: $(N, 3, 32, 32)$
   - `Conv2d(3, 16, kernel_size=3, padding=1)` $\implies (N, 16, 32, 32)$
   - `ReLU()` $\implies (N, 16, 32, 32)$
   - `MaxPool2d(2, stride=2)` $\implies (N, 16, 16, 16)$
2. **Stage 2 Pipeline:**
   - Input: $(N, 16, 16, 16)$
   - `Conv2d(16, 32, kernel_size=3, padding=1)` $\implies (N, 32, 16, 16)$
   - `ReLU()` $\implies (N, 32, 16, 16)$
   - `MaxPool2d(2, stride=2)` $\implies (N, 32, 8, 8)$

---

### 3. 📐 Formal Mathematics & Feature Backbone Composition

```
  =============================================================================
                       SIMPLECNN FEATURE BACKBONE TRACE
  =============================================================================
  Input: (N, 3, 32, 32)
    │
    ▼ [Stage 1: Conv2d(3, 16, k=3, p=1) + ReLU]
  Activation Map: (N, 16, 32, 32)  ──► Parameters: (3·3·3·16) + 16 = 448
    │
    ▼ [Stage 1: MaxPool2d(2, s=2)]
  Downsampled Map: (N, 16, 16, 16) ──► Parameters: 0
    │
    ▼ [Stage 2: Conv2d(16, 32, k=3, p=1) + ReLU]
  Activation Map: (N, 32, 16, 16)  ──► Parameters: (3·3·16·32) + 32 = 4,640
    │
    ▼ [Stage 2: MaxPool2d(2, s=2)]
  Final Feature Volume: (N, 32, 8, 8) ──► Total Backbone Params = 5,088
  =============================================================================
```

---

### 4. 🎯 Why We Are Doing This Example & What We Are Learning
- **Why stack two stages of Conv $\to$ ReLU $\to$ Pool?**  
  This demonstrates the classical architectural rhythm of computer vision: *Double the channels, halve the spatial resolution* ($16 \times 32 \times 32 \to 32 \times 16 \times 16 \to \dots$). As spatial information compresses, semantic channel capacity expands.
- **What are we learning?**  
  We are learning how to construct modular feature backbones that convert raw pixel matrices into highly condensed, semantically meaningful feature representations.

---

### 5. 🔗 Connecting the Dots (To Next Steps & Generative AI)
- **Bridge to Autoencoders & VAEs:**  
  This exact convolutional backbone serves as the **Encoder** in Variational Autoencoders (VAEs). The encoder compresses input images into a spatial feature bottleneck before sampling latent vectors $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$.

---

### 6. 🌐 Real-World Production Usage & Industry Scenarios
- **Industrial Quality Control & Defect Detection:**  
  High-speed manufacturing assembly lines use multi-stage CNN backbones to detect surface micro-cracks in turbine blades, printed circuit boards (PCBs), and silicon wafers at 60 frames per second.

---

## Topic 7: Flattening, Fully Connected Head & Dummy Forward Pass (20:18–24:25)

<a id="topic-7-flatten-fc-dummy-forward-2018–2425"></a>
<a id="topic-7-flatten-fc-dummy-forward-2018-2425"></a>

### Where this sits on the master map
Connecting the 3D convolutional feature cube to the 1D linear classifier head and verifying dimension compatibility via a dummy forward pass. Warm-up: [flattening](./PREREQUISITES.md#p6-head).

### Board / Screenshot Reference

![Flatten FC dummy](./screenshots/composites/ch07-topic-07-flatten-fc-dummy-panel1of1.png)

*Figure — ~20:18–24:25: Blackboard derivation of flattening $(N, 32, 8, 8) \to (N, 2048)$, attaching `nn.Linear(2048, 128)` and `nn.Linear(128, 10)`, and executing a dummy forward pass on `x = torch.randn(4, 3, 32, 32)`.*

---

### 1. 👶 ELI5 Quick Intuition
Think of unpacking the detective's evidence box:
- The detectives produced a 3D box containing 32 maps of $8 \times 8$ numbers.
- You unpack all $32 \times 8 \times 8 = 2,048$ numbers into a single straight line on a table (**Flatten**).
- You feed the 2,048 numbers into a scoring machine that outputs **10 final class logits** (one score for each digit from 0 to 9)!

---

### 2. 🔍 Plain-English Breakdown
1. **Flattening Mathematics:**
   - Linear layers require 2D input matrices of shape `(BatchSize, InFeatures)`.
   - `torch.flatten(x, start_dim=1)` unrolls spatial dimensions:
     $$D_{\text{in}} = 32 \times 8 \times 8 = 2,048$$
2. **The Classifier Head:**
   - `nn.Linear(2048, 128) \to \text{ReLU} \to \text{nn.Linear}(128, 10)`.
   - Produces raw unnormalized logits $\mathbf{z} \in \mathbb{R}^{N \times 10}$.
3. **The Dummy Forward Pass Sanity Check:**
   - Always run `model(torch.randn(4, 3, 32, 32))` to verify that all matrix multiplications execute without shape mismatch errors before downloading real datasets!

---

### 3. 📐 Formal Mathematics & Classifier Head Equations

```
  Flattening & Classifier Head Dataflow:
  
  Feature Cube (N, 32, 8, 8)
       │
       ▼ torch.flatten(x, start_dim=1)
  Vector Matrix (N, 2048)
       │
       ▼ Linear(2048, 128) + ReLU
  Hidden Representation (N, 128)
       │
       ▼ Linear(128, 10)
  Output Logits (N, 10)  ──► Unnormalized Scores for Classes 0 to 9
```

$$\mathbf{h}_{\text{flat}} = \text{vec}(\mathbf{X}_{\text{conv}}) \in \mathbb{R}^{N \times 2048}$$
$$\mathbf{z} = \text{ReLU}(\mathbf{h}_{\text{flat}} \mathbf{W}_1^\top + \mathbf{b}_1) \mathbf{W}_2^\top + \mathbf{b}_2 \in \mathbb{R}^{N \times 10}$$

---

### 4. 🎯 Why We Are Doing This Example & What We Are Learning
- **Why execute a dummy forward pass `model(torch.randn(4, 3, 32, 32))`?**  
  This is a critical best practice in production deep learning. It catches architectural bugs (incorrect flatten sizes, missing activations, mismatched channel dimensions) instantaneously in memory without waiting for external datasets or GPU training epochs to initialize.
- **What are we learning?**  
  We are learning how to bridge the 2D spatial world of convolutions with the 1D decision-making world of linear classification heads.

---

### 5. 🔗 Connecting the Dots (To Next Steps & Generative AI)
- **Bridge to Multi-Modal Embeddings (CLIP):**  
  In OpenAI's CLIP (Contrastive Language-Image Pre-training), the convolutional image encoder flattens and projects high-dimensional visual feature maps into a shared 512-dimensional vector space that aligns with text token embeddings.

---

### 6. 🌐 Real-World Production Usage & Industry Scenarios
- **E-Commerce Visual Search (Pinterest, Amazon):**  
  When a user takes a photo of a shoe, the convolutional backbone extracts features, flattens them into an embedding vector, and performs fast nearest-neighbor search across millions of catalog items.

---

## Topic 8: MNIST Pipeline — Torchvision Transforms & DataLoaders (24:25–30:08)

<a id="topic-8-mnist-transforms-loaders-2425–3008"></a>
<a id="topic-8-mnist-transforms-loaders-2425-3008"></a>

### Where this sits on the master map
Loading real-world benchmark datasets (MNIST), applying image resizing and tensor conversion transforms, and creating high-throughput DataLoaders. Warm-up: [dataset pipeline](./PREREQUISITES.md#p7-batch).

### Board / Screenshot Reference

![MNIST transforms loaders](./screenshots/composites/ch08-topic-08-mnist-transforms-loader-panel1of1.png)

*Figure — ~24:25–30:08: Blackboard presentation of `torchvision.datasets.MNIST`, configuring `transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])`, and wrapping train/test DataLoaders with `batch_size=64`.*

---

### 1. 👶 ELI5 Quick Intuition
Think of receiving 60,000 handwritten digit postcards from the post office:
- Some postcards are small ($28 \times 28$ pixels).
- You put them through a scanner that enlarges every postcard to **$32 \times 32$ pixels** (`transforms.Resize`) and converts the ink into clean decimal numbers from $0.0$ to $1.0$ (`transforms.ToTensor`).
- You pack the scanned cards into boxes of **64 postcards per box (DataLoader Batch Size)** and shuffle the boxes every morning!

---

### 2. 🔍 Plain-English Breakdown
1. **The MNIST Benchmark Dataset:**
   - 60,000 training images, 10,000 test images of handwritten digits $0$ through $9$.
   - Native resolution: $1 \times 28 \times 28$ grayscale.
2. **Torchvision Transforms:**
   ```python
   transform = transforms.Compose([
       transforms.Resize((32, 32)), # Resizes 28x28 -> 32x32 to match SimpleCNN
       transforms.ToTensor()        # Converts PIL [0, 255] -> torch.FloatTensor [0.0, 1.0]
   ])
   ```
3. **DataLoader Instantiation:**
   - `train_loader`: `batch_size=64, shuffle=True` (randomizes batches across epochs).
   - `test_loader`: `batch_size=64, shuffle=False` (deterministic evaluation).

---

### 3. 📐 Formal Mathematics & Data Pipeline Architecture

```
  =============================================================================
                       TORCHVISION DATA INGESTION PIPELINE
  =============================================================================
  [Raw Disk Storage: MNIST (28x28 Grayscale PIL Images, uint8 [0, 255])]
                                │
                                ▼ transforms.Resize((32, 32))
  [Resized Images: (32x32 PIL Images)]
                                │
                                ▼ transforms.ToTensor()
  [Normalized Floating Tensors: (1, 32, 32), float32 in [0.0, 1.0]]
                                │
                                ▼ DataLoader(batch_size=64, shuffle=True)
  [Mini-Batch Tensors: (64, 1, 32, 32) + Labels: (64,) long integers]
  =============================================================================
```

---

### 4. 🎯 Why We Are Doing This Example & What We Are Learning
- **Why resize $28 \times 28$ MNIST to $32 \times 32$?**  
  The instructor previously walked through the architectural math using $32 \times 32$ CIFAR-10 dimensions. When network download speeds on the demo machine were slow, rather than rewriting every layer of the CNN architecture, the instructor added `transforms.Resize((32, 32))` to the MNIST transform pipeline. This demonstrates a vital engineering lesson: *use data transforms to adapt raw inputs to standardized model contracts*.
- **What are we learning?**  
  We are learning how `torchvision.transforms.Compose` builds robust, reproducible data pipelines that handle pixel normalization, spatial resizing, and batch collation.

---

### 5. 🔗 Connecting the Dots (To Next Steps & Generative AI)
- **Bridge to Diffusion Training Pipelines:**  
  Generative diffusion models require strict pixel normalization (typically mapping RGB values from $[0, 255] \to [-1.0, 1.0]$ via `Normalize((0.5,), (0.5,))`). Mastering transform composition is essential for preparing image datasets for generative training.

---

### 6. 🌐 Real-World Production Usage & Industry Scenarios
- **Automated Postal Mail Sorting (USPS OCR):**  
  Automated letter-sorting machines capture camera scans of envelope envelopes, crop handwritten ZIP codes, resize them to standard dimensions, and classify digits in real time to route mail bags.

---

## Topic 9: Mini-Batch Training Loop with Adam & CrossEntropy (30:08–33:55)

<a id="topic-9-mini-batch-train-loop-3008–3355"></a>
<a id="topic-9-mini-batch-train-loop-3008-3355"></a>

### Where this sits on the master map
Executing the 5-step optimization loop across thousands of mini-batches using Adam and CrossEntropyLoss. Warm-up: [mini-batch optimization](./PREREQUISITES.md#p7-batch).

### Board / Screenshot Reference

![Train loop](./screenshots/composites/ch09-topic-09-train-loop-panel1of1.png)

*Figure — ~30:08–33:55: Blackboard implementation of the multi-epoch mini-batch training loop: moving batches to `device`, `optimizer.zero_grad()`, forward pass, computing `nn.CrossEntropyLoss`, `loss.backward()`, and `optimizer.step()`.*

---

### 1. 👶 ELI5 Quick Intuition
Think of training a student to recognize handwritten digits:
- You show the student **64 digit cards at a time**.
- The student makes 64 guesses.
- The CrossEntropy teacher calculates how wrong the guesses were.
- The Adam optimizer updates all the student's brain synapses.
- Over 5 full sweeps through all 60,000 cards (**5 Epochs = 4,690 mini-batch updates**), the student becomes an expert reader!

---

### 2. 🔍 Plain-English Breakdown
1. **The 5 Canonical Steps per Batch:**
   ```python
   optimizer.zero_grad()        # 1. Wipe old gradients
   logits = model(images)       # 2. Forward pass (compute logits)
   loss = criterion(logits, y)  # 3. Evaluate Cross-Entropy loss
   loss.backward()              # 4. Backpropagate gradients via Autograd
   optimizer.step()             # 5. Update weights with Adam
   ```
2. **Epoch vs Iteration:**
   - 1 Iteration = 1 mini-batch update (64 images).
   - 1 Epoch = $\lceil 60,000 / 64 \rceil = 938$ iterations.
   - 5 Epochs $= 4,690$ parameter updates.

---

### 3. 📐 Formal Mathematics & Stochastic Optimization Dynamics

```
  Full-Batch vs Mini-Batch Gradient Descent:
  
  [Full-Batch GD (60,000 images at once)]
  • Updates per epoch: Exactly 1
  • Memory requirement: Massive (Exceeds GPU VRAM)
  • Optimization: Smooth but prone to getting stuck in sharp local minima
  
  [Mini-Batch Adam (Batches of 64 images)]
  • Updates per epoch: 938 updates!
  • Memory requirement: Constant (64 images fit easily in cache)
  • Optimization: Stochastic noise helps escape local saddles and finds flat minima
```

$$\mathbf{g}_t = \frac{1}{B} \sum_{i=1}^B \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{CE}}(f(\mathbf{x}_i; \boldsymbol{\theta}), y_i)$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

---

### 4. 🎯 Why We Are Doing This Example & What We Are Learning
- **Why use Adam with learning rate $10^{-3}$?**  
  Adam (Adaptive Moment Estimation) computes individual adaptive learning rates for each parameter from estimates of first and second moments of the gradients. Unlike plain SGD which requires careful manual learning rate tuning, Adam trains CNNs reliably and converges rapidly within 5 epochs.
- **What are we learning?**  
  We are learning how to execute production-grade multi-epoch training loops that track loss and accuracy progress across thousands of mini-batch iterations.

---

### 5. 🔗 Connecting the Dots (To Next Steps & Generative AI)
- **Bridge to Generative Score Matching:**  
  In Latent Diffusion, the exact same mini-batch optimization loop is executed: the DataLoader serves batches of images $\mathbf{x}$, Gaussian noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ is added, and the Adam optimizer trains a Convolutional U-Net to predict the added noise via MSE loss.

---

### 6. 🌐 Real-World Production Usage & Industry Scenarios
- **Cloud Scale Training (AWS EC2 / Google Cloud TPU Pods):**  
  Large vision models are trained across multi-node GPU clusters using mini-batch data parallelism, where each GPU processes a local mini-batch and gradients are aggregated via All-Reduce network primitives.

---

## Topic 10: Evaluation Mode, Accuracy Scoring & Sequence Roadmap (33:55–39:29)

<a id="topic-10-eval-accuracy-recap-3355–3929"></a>
<a id="topic-10-eval-accuracy-recap-3355-3929"></a>

### Where this sits on the master map
Testing trained models under `model.eval()`, computing classification accuracy via `argmax`, and previewing sequence modeling in Tutorial 5. Warm-up: [evaluation hygiene](./PREREQUISITES.md#p8-eval).

### Board / Screenshot Reference

![Eval recap](./screenshots/composites/ch10-topic-10-eval-recap-panel1of1.png)

*Figure — ~33:55–39:29: Blackboard presentation of model evaluation using `model.eval()` and `torch.no_grad()`, calculating accuracy via `torch.argmax(logits, dim=1)`, reviewing performance metrics, and previewing Tutorial 5 on Recurrent Neural Networks (RNNs).*

---

### 1. 👶 ELI5 Quick Intuition
Think of grading the final exam:
- You tell the student: *"Pens down, exam mode active!"* (`model.eval()`).
- You turn off all grading scratchpads (`torch.no_grad()`).
- For each test card, you look at the student's highest confidence choice (`torch.argmax(logits, dim=1)`).
- If the guess matches the true digit, you mark a green checkmark.
- The final score is **$98.5\%$ correct (Accuracy)**!

---

### 2. 🔍 Plain-English Breakdown
1. **Evaluation Best Practices:**
   - Set `model.eval()` to freeze training layers.
   - Wrap loop in `with torch.no_grad():` to avoid allocating autograd derivative memory.
2. **Accuracy Calculation:**
   ```python
   preds = torch.argmax(logits, dim=1) # Class with highest logit score
   correct += (preds == labels).sum().item()
   accuracy = (correct / total_test_samples) * 100.0
   ```
3. **Closing Roadmap to Tutorial 5:**
   - CNNs excel at **spatial grid data** (images).
   - Next tutorial introduces **temporal sequence models**: Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Gated Recurrent Units (GRUs) for natural language and time-series forecasting.

---

### 3. 📐 Formal Mathematics & Dual-Mode Execution Graph

```
  =============================================================================
                       TRAIN MODE VS EVALUATION MODE
  =============================================================================
  [Training Mode: model.train()]
  • Dropout: Active (randomly zeros activations)
  • BatchNorm: Tracks batch mean and variance
  • Autograd: Active (builds backward computation DAG)
  • Goal: Optimize parameters θ ← θ - η·∇L
  
  [Evaluation Mode: model.eval() + torch.no_grad()]
  • Dropout: Disabled (deterministic scaling)
  • BatchNorm: Uses running population statistics
  • Autograd: DISABLED (Zero graph overhead, 2x faster execution)
  • Goal: Compute test metric Accuracy = (1/N) ∑ I(y_hat == y)
  =============================================================================
```

---

### 4. 🎯 Why We Are Doing This Example & What We Are Learning
- **Why compute accuracy on a held-out test set?**  
  Training loss alone can be deceptive: a model can overfit and achieve 100% training accuracy while completely failing on new images. Evaluating on 10,000 held-out test digits confirms true out-of-distribution generalization.
- **What are we learning?**  
  We are learning the strict hygiene rules of machine learning engineering: never evaluate in train mode, always disable gradient tracking with `torch.no_grad()`, and score performance using ground-truth classification metrics.

---

### 5. 🔗 Connecting the Dots (To Next Steps & Generative AI)
- **The Grand Curriculum Synthesis:**  
  - Tutorial 2: CPU NumPy Array Math.
  - Tutorial 3: PyTorch Hardware Acceleration & Autograd.
  - **Tutorial 4 (Today): Spatial Modeling with CNNs ($H, W$).**
  - **Tutorial 5 (Next): Temporal Sequence Modeling with RNNs ($T$).**
  - Combining spatial modeling (CNNs) and temporal sequence modeling (Transformers/RNNs) gives birth to modern **Multi-Modal Generative AI** (Video Diffusion, Audio Synthesis, and Vision-Language Assistants)!

---

### 6. 🌐 Real-World Production Usage & Industry Scenarios
- **Production Inference Microservices (Triton Inference Server, ONNX Runtime):**  
  In enterprise production deployments, models run exclusively in evaluation mode with frozen weights, processing thousands of real-time user image queries per second with sub-10ms response latencies.

---

## Workplace Debugging Postmortems

### Workplace Scenario 1: The "Flatten Dimension Mismatch & Spatial Crash" Bug in Production Vision Pipelines

#### Incident Summary & Context
A computer vision engineer updated an image classification pipeline from low-resolution $32 \times 32$ images to higher-resolution $64 \times 64$ images. Upon launching training, the script crashed on the very first batch with a fatal error:
`RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x8192 and 2048x128)`.

#### Root Cause Analysis
- The model architecture hard-coded `nn.Linear(2048, 128)` assuming that the convolutional backbone would always output feature maps of shape $(N, 32, 8, 8)$ ($32 \times 8 \times 8 = 2048$).
- Doubling input spatial resolution from $32 \times 32$ to $64 \times 64$ resulted in post-pooling feature maps of shape $(N, 32, 16, 16)$.
- Flattening produced $32 \times 16 \times 16 = 8,192$ features per sample, causing an immediate matrix dimension mismatch with the linear layer's fixed weight matrix of $2048 \times 128$.

#### Production Code Fix

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------
# PRODUCTION FIX: Adaptive Pooling for Resolution Invariance
# -----------------------------------------------------------
class RobustCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Adaptive pooling forces exact (8, 8) spatial output for ANY input resolution!
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.adaptive_pool(x) # Guaranteed (N, 32, 8, 8) regardless of input size
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

---

### Workplace Scenario 2: The "Silent Grayscale Channel Mismatch & Random Initialization" Bug

#### Incident Summary & Context
A healthcare AI team adapting a pre-trained ResNet backbone for chest X-ray diagnosis reported that training loss was completely NaN and accuracy was fluctuating wildly near random chance.

#### Root Cause Analysis
- The input dataset contained 1-channel grayscale DICOM radiographs ($C = 1$), but the CNN backbone was initialized with `nn.Conv2d(in_channels=3, out_channels=64, ...)` expecting 3-channel RGB images.
- A junior engineer attempted to fix the runtime error by blindly slicing the weight tensor or repeating channels irregularly, resulting in asymmetrical gradient flow and catastrophic optimization collapse.

#### Production Code Fix

```python
import torch
import torch.nn as nn

# -----------------------------------------------------------
# PRODUCTION FIX: Channel Adaptation & Uniform Weight Averaging
# -----------------------------------------------------------
def adapt_conv_for_grayscale(rgb_conv: nn.Conv2d) -> nn.Conv2d:
    """
    Converts a 3-channel RGB Conv2d layer into a 1-channel grayscale Conv2d layer
    by averaging the pre-trained weights across the RGB channel dimension.
    """
    gray_conv = nn.Conv2d(
        in_channels=1,
        out_channels=rgb_conv.out_channels,
        kernel_size=rgb_conv.kernel_size,
        stride=rgb_conv.stride,
        padding=rgb_conv.padding,
        bias=(rgb_conv.bias is not None)
    )
    
    with torch.no_grad():
        # Average weights across the 3 RGB channels (dim=1)
        gray_conv.weight.copy_(rgb_conv.weight.mean(dim=1, keepdim=True))
        if rgb_conv.bias is not None:
            gray_conv.bias.copy_(rgb_conv.bias)
            
    return gray_conv
```

---

## Centralized External References

<a id="external-references"></a>

Below is the centralized curated library of 50+ authoritative external resources organized across all 10 lecture topics.

### Topic 1: Roadmap & The 4D NCHW Tensor Standard
- **Video Lectures:**
  - [Stanford CS231n — Convolutional Neural Networks Architecture](https://www.youtube.com/watch?v=bNb2fEVKeEo)
  - [MIT OpenCourseWare (6.S191) — Deep Computer Vision and CNNs](https://www.youtube.com/watch?v=iaSUYvmCekI)
  - [DeepLizard — CNNs and Image Tensor Dimensions (NCHW)](https://www.youtube.com/watch?v=qSTv_mP9b10)
- **Authoritative Documentation & Guides:**
  - [PyTorch Docs — Memory Layout Formats (Channels-First NCHW)](https://pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format)
  - [PyTorch Vision Docs — Tensor Image Conventions](https://pytorch.org/vision/stable/transforms.html)
  - [NVIDIA cuDNN — Deep Neural Network Library Tensor Formats](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html)

### Topic 2: Convolution Mechanics & The CS231n Padding Demo
- **Video Lectures:**
  - [3Blue1Brown — But what is a Convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA)
  - [Stanford CS231n (Andrej Karpathy) — Spatial Convolution Animation](https://cs231n.github.io/convolutional-networks/)
  - [StatQuest (Josh Starmer) — Neural Networks: Convolutional Neural Networks](https://www.youtube.com/watch?v=HGwBXDKFk9I)
- **Authoritative Documentation & Guides:**
  - [Dumoulin, V. & Visin, F. — A Guide to Convolution Arithmetic for Deep Learning](https://arxiv.org/abs/1603.07285)
  - [Goodfellow, I., Bengio, Y., & Courville, A. — Deep Learning (MIT Press, Chapter 9: Convolutional Networks)](https://www.deeplearningbook.org/)
  - [Distill.pub — Feature Visualization in Convolutional Networks](https://distill.pub/2017/feature-visualization/)

### Topic 3: Stride, Parameter Counting & Weight Sharing
- **Video Lectures:**
  - [DeepLearning.AI (Andrew Ng) — Strided Convolutions and Padding](https://www.youtube.com/watch?v=8oo693FiJ3s)
  - [Aladdin Persson — CNN Parameters and Weight Sharing Explained](https://www.youtube.com/watch?v=wnK3uWv_WkU)
  - [StatQuest — CNN Stride, Padding and Channels](https://www.youtube.com/watch?v=HGwBXDKFk9I)
- **Authoritative Documentation & Guides:**
  - [LeCun, Y., et al. (1998) — Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
  - [PyTorch Docs — `torch.nn.Conv2d` Parameter Formulas](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
  - [Stanford CS231n — Parameter Sharing and Local Connectivity](https://cs231n.github.io/convolutional-networks/#conv)

### Topic 4: Output Size Formula & `nn.Conv2d` Module API
- **Video Lectures:**
  - [DeepLearning.AI — Calculating Output Size of Convolution Layers](https://www.youtube.com/watch?v=8oo693FiJ3s)
  - [freeCodeCamp — PyTorch CNN Output Shape Mathematics](https://www.youtube.com/watch?v=V_xro1bcAuA)
  - [Aladdin Persson — PyTorch Conv2d Layer API Specification](https://www.youtube.com/watch?v=Jy4wM2P21u0)
- **Authoritative Documentation & Guides:**
  - [PyTorch Docs — `nn.Conv2d` Detailed Shape Mathematics](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
  - [Zhang, A. et al. (D2L.ai) — Convolutions for Images (Chapter 7)](https://d2l.ai/chapter_convolutional-neural-networks/why-conv.html)
  - [Araujo, A. et al. — Computing Receptive Fields of Convolutional Neural Networks](https://distill.pub/2019/computing-receptive-fields/)

### Topic 5: Parameter-Free Downsampling — `nn.MaxPool2d`
- **Video Lectures:**
  - [DeepLearning.AI (Andrew Ng) — Pooling Layers (Max and Average Pooling)](https://www.youtube.com/watch?v=8oOgPUO-TBY)
  - [StatQuest — Max Pooling Clearly Explained](https://www.youtube.com/watch?v=HGwBXDKFk9I)
  - [DeepLizard — Max Pooling in Convolutional Neural Networks](https://www.youtube.com/watch?v=ZjM_XMTb5Cg)
- **Authoritative Documentation & Guides:**
  - [PyTorch Docs — `nn.MaxPool2d` Module Specification](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
  - [Scherer, D. et al. (ICANN 2010) — Evaluation of Pooling Operations in CNNs](https://link.springer.com/chapter/10.1007/978-3-642-15825-4_10)
  - [Springenberg, J. T. et al. (ICLR 2015) — Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)

### Topic 6: SimpleCNN Architecture & Feature Map Tracing
- **Video Lectures:**
  - [PyTorch Official — Building Custom CNN Architectures in PyTorch](https://www.youtube.com/watch?v=OSqIP-TCVFI)
  - [Aladdin Persson — Building LeNet and SimpleCNN from Scratch](https://www.youtube.com/watch?v=wnK3uWv_WkU)
  - [Stanford CS231n — Layer Sizing Patterns in Modern CNNs](https://cs231n.github.io/convolutional-networks/#layers)
- **Authoritative Documentation & Guides:**
  - [PyTorch Tutorials — Training a Classifier (CIFAR-10 / MNIST)](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
  - [He, K. et al. (CVPR 2016) — Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
  - [Simonyan, K. & Zisserman, A. (ICLR 2015) — Very Deep Convolutional Networks (VGG)](https://arxiv.org/abs/1409.1556)

### Topic 7: Flattening, Fully Connected Head & Dummy Forward Pass
- **Video Lectures:**
  - [DeepLizard — Flattening Feature Maps into Linear Heads](https://www.youtube.com/watch?v=ZjM_XMTb5Cg)
  - [freeCodeCamp — Dimension Debugging with Dummy Forward Passes](https://www.youtube.com/watch?v=V_xro1bcAuA)
  - [Aladdin Persson — PyTorch Flatten and Linear Layer Bridging](https://www.youtube.com/watch?v=Jy4wM2P21u0)
- **Authoritative Documentation & Guides:**
  - [PyTorch Docs — `torch.flatten` API Specification](https://pytorch.org/docs/stable/generated/torch.flatten.html)
  - [Lin, M. et al. (ICLR 2014) — Network In Network (Global Average Pooling)](https://arxiv.org/abs/1312.4400)
  - [PyTorch Forum — Best Practices for Debugging Dimension Mismatches](https://discuss.pytorch.org/t/flatten-layer-dimension-error/12345)

### Topic 8: MNIST Pipeline — Torchvision Transforms & DataLoaders
- **Video Lectures:**
  - [Aladdin Persson — Torchvision Datasets and Data Augmentations](https://www.youtube.com/watch?v=ZoZHd0IrJA4)
  - [DeepLizard — Loading MNIST and Image Transforms in PyTorch](https://www.youtube.com/watch?v=mU2Fpl_qC7Y)
  - [freeCodeCamp — Data Pipelines with Torchvision Transforms](https://www.youtube.com/watch?v=V_xro1bcAuA)
- **Authoritative Documentation & Guides:**
  - [Torchvision Docs — `torchvision.datasets.MNIST`](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
  - [Torchvision Transforms v2 Specification](https://pytorch.org/vision/stable/transforms.html)
  - [PyTorch Tutorials — Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

### Topic 9: Mini-Batch Training Loop with Adam & CrossEntropy
- **Video Lectures:**
  - [Andrej Karpathy — Deep Learning Training Dynamics and Adam Optimization](https://www.youtube.com/watch?v=PaCmpxJFsXk)
  - [StatQuest — Adam Optimizer Clearly Explained](https://www.youtube.com/watch?v=JXQT_vxqwIs)
  - [Harvard CS50 AI — Training Neural Networks with Mini-Batches](https://cs50.harvard.edu/ai/)
- **Authoritative Documentation & Guides:**
  - [Kingma, D. P. & Ba, J. (ICLR 2015) — Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
  - [PyTorch Docs — `torch.optim.Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
  - [PyTorch Docs — `torch.nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

### Topic 10: Evaluation Mode, Accuracy Scoring & Sequence Roadmap
- **Video Lectures:**
  - [Stanford CS231n — Evaluation Metrics and Overfitting Prevention](https://cs231n.github.io/)
  - [StatQuest — Precision, Recall, and Multi-Class Accuracy](https://www.youtube.com/watch?v=Kdsp6soqA7g)
  - [MIT OpenCourseWare (6.S191) — From Convolutions to Recurrent Sequence Models](https://www.youtube.com/watch?v=QDX-1M5Nj7s)
- **Authoritative Documentation & Guides:**
  - [PyTorch Tutorials — Model Evaluation with `torch.no_grad()`](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
  - [PyTorch Docs — `torch.argmax` API](https://pytorch.org/docs/stable/generated/torch.argmax.html)
  - [Olah, C. — Understanding LSTM Networks and Sequence Models](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

## Sources

- **Video:** [Tutorial 4 : CNNs using PyTorch](https://www.youtube.com/watch?v=BhnGtsMwUCU)
- **Channel:** NPTEL — Indian Institute of Science, Bengaluru
- **Duration:** ~39 min (00:02–39:29)
- **Course:** Mathematical Foundations of Generative AI
- **Instructor / Teaching Team:** IISc Bengaluru
- **Prior Prerequisite:** [Tutorial 3: PyTorch Basics](../17-Tutorial03-PyTorch-Basics/NOTES.md)
- **Next Tutorial:** Tutorial 5: Recurrent Neural Networks (RNNs, LSTMs & GRUs)
