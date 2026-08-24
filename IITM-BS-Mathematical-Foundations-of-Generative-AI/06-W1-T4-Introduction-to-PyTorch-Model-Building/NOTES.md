# W1_T4 — Introduction to PyTorch: Model Building

> **Prerequisites:** Please read the warm-up in [PREREQUISITES.md](./PREREQUISITES.md) first to build intuition on Multi-Layer Perceptrons, spatial flattening, logits vs. softmax, `nn.Module`, Cross-Entropy loss, and PyTorch training loops.  
> **Interactive Quiz:** Test your mastery in [quiz.html](./quiz.html).

---

> ℹ️ **Course & Series Context Note:**  
> This tutorial (**W1_T4**) bridges the data pipelines constructed in W1_T2 (Dataset & DataLoader) to a complete, trainable neural network in PyTorch. The instructor (TA Chandan) live-codes an MLP classifier on Fashion-MNIST, establishes the official 4-step training loop with gradient flushing, demonstrates test evaluation with `torch.no_grad()`, and saves the model weights to `.pth`. This practical workflow is the exact engineering template required for building GAN Discriminators and Generators in upcoming weeks.

---

## Table of Contents

- [Executive Summary — architecture of this lecture](#executive-summary--architecture-of-this-lecture)
- [Chalkboard & PyTorch Rosetta Stone](#chalkboard--pytorch-rosetta-stone)
- [Complete Executable Python / PyTorch Implementation](#complete-executable-python--pytorch-implementation)
- [Topic 1: Data ready; MLP / CNN / ViT (00:16–02:49)](#topic-1-data-ready-mlp--cnn--vit-0016–0249)
- [Topic 2: Choices, imports, Colab GPU (02:49–05:46)](#topic-2-choices-imports-colab-gpu-0249–0546)
- [Topic 3: NeuralNetwork, flatten, 10 logits (05:46–09:11)](#topic-3-neuralnetwork-flatten-10-logits-0546–0911)
- [Topic 4: Sequential; Linear 784 to 512 (09:11–13:07)](#topic-4-sequential-linear-784-to-512-0911–1307)
- [Topic 5: ReLU stack, logits, P(Y|X) (13:07–17:36)](#topic-5-relu-stack-logits-pyx-1307–1736)
- [Topic 6: .to(device) and a dummy predict (17:36–19:58)](#topic-6-todevice-and-a-dummy-predict-1736–1958)
- [Topic 7: SGD / Adam, CE, hyperparameters (19:58–23:54)](#topic-7-sgd--adam-ce-hyperparameters-1958–2354)
- [Topic 8: train_loop: four steps (23:54–28:38)](#topic-8-train_loop-four-steps-2354–2838)
- [Topic 9: test_loop, accuracy, curves (28:38–36:31)](#topic-9-test_loop-accuracy-curves-2838–3631)
- [Topic 10: Save/load, CNN preview, GANs next (36:31–44:06)](#topic-10-saveload-cnn-preview-gans-next-3631–4406)
- [Apply it (scenarios)](#apply-it-scenarios)
- [External references](#external-references)
- [Sources](#sources)

---

## Executive Summary — architecture of this lecture

This tutorial builds an end-to-end classification pipeline in PyTorch. It defines an `nn.Module` Multi-Layer Perceptron (MLP) on Fashion-MNIST, vectors 2D photos through `nn.Flatten()`, stacks linear layers with `nn.ReLU` hinges, and outputs raw logits. It implements the canonical 4-step optimization loop with gradient flushing, evaluates test accuracy via `torch.no_grad()`, and serializes learned weights to disk.

**Worldview Arc:** From raw tensor batches in memory to a fully trained, evaluated, and saved `nn.Module` neural network ready for downstream inference.

### System Context

```
   ┌────────────────────────────────┐                 ┌───────────────────────────────┐
   │ Fashion-MNIST Dataset          │                 │ Hardware Accelerator Target   │
   │ 60,000 Train / 10,000 Test     │                 │ "cuda" (NVIDIA GPU) or "cpu"  │
   │ X: (B, 1, 28, 28), y: (B,)     │                 │ Runtime environment check     │
   └───────────────┬────────────────┘                 └───────────────┬───────────────┘
                   │                                                  │
                   ▼                                                  ▼
   ┌──────────────────────────────────────────────────────────────────────────────────┐
   │                         PYTORCH MODEL BUILDING PIPELINE                          │
   │                                                                                  │
   │  1. Flatten: (B, 1, 28, 28) ──> (B, 784)                                         │
   │  2. Linear ReLU Stack: 784 ──> 512 ──> 512 ──> 10 Logits                         │
   │  3. Loss & Optimizer: nn.CrossEntropyLoss() + optim.SGD(lr=1e-3)                 │
   │  4. Train Loop: predict ──> loss ──> backward ──> step ──> zero_grad              │
   │  5. Test Loop: model.eval() + torch.no_grad() ──> argmax vs true y               │
   └────────────────────────────────────────┬─────────────────────────────────────────┘
                                            │
                                            ▼
                           ┌──────────────────────────────────┐
                           │ Serialized Model Checkpoint      │
                           │ torch.save(state_dict, "m.pth")  │
                           └──────────────────────────────────┘
```

### Main Architecture Blueprint

```
[ STAGE 1: HARDWARE & DATAFLOW SETUP ]
   Input Image Batch: X in R^{B x 1 x 28 x 28}, Labels: y in {0..9}^B
   Device Allocation: device = "cuda" if torch.cuda.is_available() else "cpu"
   Transfer Tensors: X.to(device), y.to(device)
                                │
                                ▼
[ STAGE 2: NEURAL ARCHITECTURE (nn.Module) ]
   class NeuralNetwork(nn.Module):
      ├── self.flatten = nn.Flatten()                    ──> Reshapes (B, 1, 28, 28) to (B, 784)
      └── self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 512),  nn.ReLU(),             ──> Layer 1: W1*x + b1 (401,920 params)
            nn.Linear(512, 512),  nn.ReLU(),             ──> Layer 2: W2*h1 + b2 (262,656 params)
            nn.Linear(512, 10)                           ──> Layer 3: W3*h2 + b3 (5,130 params, NO ReLU)
          )
                                │
                                ▼
[ STAGE 3: RAW LOGITS & LOSS CHOKEPOINT ]
   Forward Pass: logits = model(X) in R^{B x 10} (Raw unnormalized scores)
   Loss Computation: loss = nn.CrossEntropyLoss()(logits, y)
   Formula: L = (1/B) ∑ [ -log( e^{z_{y_i}} / ∑_j e^{z_j} ) ]
                                │
                                ▼
[ STAGE 4: CANONICAL TRAINING STEP (train_loop) ]
   Step 1: pred = model(X)                               ──> Forward pass generates prediction logits
   Step 2: loss = loss_fn(pred, y)                       ──> Measure error against true ground truth
   Step 3: loss.backward()                               ──> Autograd computes ∂L/∂W and stores in W.grad
   Step 4: optimizer.step()                              ──> Update weights: W <- W - η * W.grad
   Step 5: optimizer.zero_grad()                         ──> Flush gradient buffers (prevents accumulation)
                                │
                                ▼
[ STAGE 5: EVALUATION & SERIALIZATION (test_loop & save) ]
   Evaluation Mode: model.eval()                         ──> Deactivates training-specific stochasticity
   Context Manager: with torch.no_grad():                ──> Suppresses computation graph (saves VRAM)
   Metric Computation: (pred.argmax(1) == y).sum()       ──> Calculate overall test accuracy (~71%)
   Persistence: torch.save(model.state_dict(), "m.pth")  ──> Save learned weights to disk
```

### Comparative Feature Matrices

#### Matrix 1: Architectural Trade-offs: MLP vs. CNN vs. Vision Transformer (ViT)
| Metric / Property | Multi-Layer Perceptron (MLP) | Convolutional Neural Network (CNN) | Vision Transformer (ViT) |
| :--- | :--- | :--- | :--- |
| **Spatial Invariance** | None (pixel positions are flattened) | High (translation equivariance via sliding kernels) | Learned via global self-attention & positional embeddings |
| **Parameter Count** | $\mathcal{O}(D_{\text{in}} \cdot D_{\text{out}})$ (explodes with high resolution) | $\mathcal{O}(K \cdot C_{\text{in}} \cdot C_{\text{out}})$ (weight sharing keeps parameters small) | $\mathcal{O}(D_{\text{model}}^2 \cdot L)$ (moderate to large) |
| **Inductive Bias** | Minimal (must learn all spatial relationships from scratch) | Strong spatial locality and hierarchical feature bias | Weak inductive bias (requires large datasets or pre-training) |
| **Role in This Course** | Today's tutorial baseline classifier & future GAN discriminator/generator backbone | Previewed in Topic 10; standard for computer vision | Future advanced vision modeling |

#### Matrix 2: PyTorch Execution Modes: Training vs. Evaluation
| Feature | `model.train()` | `model.eval()` + `torch.no_grad()` |
| :--- | :--- | :--- |
| **Autograd Graph Construction** | Yes (dynamically tracks forward operations for backprop) | No (graph building completely bypassed inside `torch.no_grad()`) |
| **Memory (VRAM) Consumption** | High (stores all intermediate layer activations for backward pass) | Low (activations discarded immediately after layer execution) |
| **Dropout & BatchNorm Behavior** | Dropout randomly drops nodes ($p=0.5$); BatchNorm computes batch statistics | Dropout disabled (scale $1.0$); BatchNorm uses running population stats |
| **Gradient Updates** | `optimizer.step()` updates model weights | Weights strictly frozen |

#### Matrix 3: Optimization Algorithms: SGD vs. Adam
| Dimension | Standard SGD (`torch.optim.SGD`) | Adam (`torch.optim.Adam`) |
| :--- | :--- | :--- |
| **Update Formula** | $\theta \leftarrow \theta - \eta \nabla_\theta L$ | $\theta \leftarrow \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ (adaptive moments) |
| **Hyperparameters** | Learning rate $\eta$ (e.g. $10^{-3}$) | Learning rate $\eta$ (e.g. $10^{-3}$), $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$ |
| **Convergence Speed** | Slower; sensitive to learning rate tuning | Faster initial convergence; adapts learning rate per parameter |
| **Chalkboard Status** | Used in today's live-code cell | Mentioned as standard alternative optimizer |

### Scenario Walkthrough

1. **Environment Initialization:** Check if an NVIDIA GPU is available (`device = "cuda" if torch.cuda.is_available() else "cpu"`).
2. **Model Definition:** Define `class NeuralNetwork(nn.Module)` with `nn.Flatten()` and three `nn.Linear` layers with `nn.ReLU` activations between them. Instantiate on device (`model = NeuralNetwork().to(device)`).
3. **Loss & Optimizer Setup:** Set loss function to `nn.CrossEntropyLoss()` and optimizer to `torch.optim.SGD(model.parameters(), lr=1e-3)`.
4. **Training Execution (`train_loop`):** Set `model.train()`, iterate over training batches, run forward prediction, compute Cross-Entropy loss, backpropagate gradients (`loss.backward()`), update weights (`optimizer.step()`), and flush gradients (`optimizer.zero_grad()`).
5. **Evaluation & Persistence (`test_loop`):** Set `model.eval()`, wrap in `with torch.no_grad():`, compute test accuracy via `pred.argmax(1) == y`, and save weights via `torch.save(model.state_dict(), "model.pth")`.

### Failure and Contrast Path

```
   INCORRECT TRAIN LOOP (BROKEN / CRASHES):
   [x] Step 1: Compute loss = loss_fn(pred, y)
   [x] Step 2: loss.backward()
   [x] Step 3: optimizer.step()
   [x] Bug: Forgetting optimizer.zero_grad()
   ──> Result: Gradients from batch 1 accumulate into batch 2, causing exploding gradient updates and diverging loss!
   
   CORRECT IDIOMATIC TRAIN LOOP (STABLE):
   [√] Step 1: pred = model(X)
   [√] Step 2: loss = loss_fn(pred, y)
   [√] Step 3: loss.backward()
   [√] Step 4: optimizer.step()
   [√] Step 5: optimizer.zero_grad()  ──> Clears gradient buffers for the next batch!
```

### Out of Scope (Later Lectures)
- Deep Convolutional Neural Network (CNN) feature map mathematics (covered in Week 2 / CNN tutorials).
- Vision Transformer (ViT) self-attention patch tokenization.
- Generative Adversarial Network (GAN) minimax saddle-point training (covered in Week 2 & 3).

### Load-Bearing Claims of this Lecture
- `ToTensor` is the essential torchvision transform converting PIL images into float32 tensors scaled to $[0.0, 1.0]$.
- Image classification models can be implemented using MLPs, CNNs, or ViTs; MLPs require flattening 2D inputs into 1D vectors.
- Neural network design requires selecting layer count, hidden node dimensions, and activation functions.
- PyTorch models inherit from `nn.Module` and call `super().__init__()` in the constructor to register parameter tracking.
- `nn.Flatten()` reshapes $(B, 1, 28, 28)$ into $(B, 784)$ while preserving batch dimension $B$.
- Linear layers compute affine transformations $y = W^\top x + b$; stacked linear layers require non-linear activations (e.g. ReLU) to avoid collapsing into a single linear map.
- The final linear layer outputs raw logits; `nn.CrossEntropyLoss` internally applies log-softmax for numerical stability.
- Calling `model(X)` executes `nn.Module.__call__`, triggering forward hooks before running `forward()`.
- The 4-step training cycle consists of prediction, loss calculation, backward gradient computation, and optimizer weight update.
- `optimizer.zero_grad()` must be executed to prevent gradient accumulation across successive mini-batches.
- `model.eval()` and `torch.no_grad()` are mandatory during evaluation to disable gradient computation and ensure deterministic inference.
- Learned model parameters are saved to disk using `torch.save(model.state_dict(), "model.pth")`.

*Course:* IIT Madras BS Degree Programme in Data Science / Generative AI (Prof. Prathosh A. P., Tutorial by TA Chandan).

---

## Chalkboard & PyTorch Rosetta Stone

Quick reference for mathematical notation and PyTorch APIs used in this tutorial:

| Notation / API | Type | Role in Pipeline |
| :--- | :--- | :--- |
| `torch.device("cuda" if ...)` | Device Config | Selects GPU accelerator if present, else defaults to CPU |
| `nn.Module` | Base Class | Base class for all neural network modules; tracks parameters and submodules |
| `super().__init__()` | Constructor Call | Initializes base `nn.Module` bookkeeping and parameter registration |
| `nn.Flatten()` | Layer | Flattens 2D spatial dimensions $(1, 28, 28) \to 784$ without altering batch size $B$ |
| `nn.Linear(784, 512)` | Layer | Dense affine mapping $z = W x + b$ containing $401,920$ trainable parameters |
| `nn.ReLU()` | Activation | Elementwise non-linear hinge activation $\max(0, z)$ |
| `nn.Sequential(...)` | Container | Cascades layers sequentially where output of layer $i$ is input to layer $i+1$ |
| `nn.CrossEntropyLoss()` | Loss Function | Combines `LogSoftmax` and `NLLLoss` into a single numerically stable loss metric |
| `torch.optim.SGD(params, lr=1e-3)` | Optimizer | Implements stochastic gradient descent parameter update rule $\theta \leftarrow \theta - \eta \nabla_\theta L$ |
| `pred = model(X)` | Inference Call | Executes `__call__` dunder method running pre-hooks, `forward(x)`, and post-hooks |
| `loss.backward()` | Autograd Call | Computes partial derivatives $\frac{\partial L}{\partial \theta}$ and populates `.grad` attributes |
| `optimizer.step()` | Optimization Call | Updates parameter tensor values using accumulated gradients |
| `optimizer.zero_grad()` | Cleanup Call | Sets all `.grad` tensors to zero before the next iteration |
| `model.eval()` | Mode Toggle | Switches network to evaluation mode (configures Dropout & BatchNorm) |
| `torch.no_grad()` | Context Manager | Context manager disabling autograd graph tracking to reduce VRAM usage |
| `loss.item()` | Tensor Method | Extracts 0D scalar float value, detaching from the computational graph |
| `model.state_dict()` | Serialization API | Returns an `OrderedDict` of all parameter tensors for disk persistence |

---

## Complete Executable Python / PyTorch Implementation

Below is a self-contained, complete, runnable PyTorch script that implements the full Fashion-MNIST model building, training, evaluation, and serialization pipeline. It includes a fallback synthetic data generator so you can run it immediately without external downloads:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Device Selection & Synthetic Data Preparation
# =====================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {device}")

def create_synthetic_dataloaders(batch_size=64):
    X_train = torch.randn(600, 1, 28, 28)
    y_train = torch.randint(0, 10, (600,))
    X_test = torch.randn(200, 1, 28, 28)
    y_test = torch.randint(0, 10, (200,))
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

train_dataloader, test_dataloader = create_synthetic_dataloaders(batch_size=64)

# =====================================================================
# 2. Neural Network Architecture Definition (MLP)
# =====================================================================
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)  # Output raw logits for 10 classes (NO ReLU on last layer)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(f"\nModel Architecture:\n{model}")

# =====================================================================
# 3. Loss Function & Optimizer Setup
# =====================================================================
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# =====================================================================
# 4. Training Loop Implementation
# =====================================================================
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # Set to training mode
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 1. Forward Pass (Compute prediction logits)
        pred = model(X)
        
        # 2. Compute Loss
        loss = loss_fn(pred, y)

        # 3. Backward Pass (Compute gradients)
        loss.backward()
        
        # 4. Optimizer Step (Update weights)
        optimizer.step()
        
        # 5. Zero Gradients (Flush gradient buffers)
        optimizer.zero_grad()

        if batch % 4 == 0:
            loss_val, current = loss.item(), batch * len(X) + len(X)
            print(f"  loss: {loss_val:>7f}  [{current:>3d}/{size:>3d}]")

# =====================================================================
# 5. Testing Loop Implementation
# =====================================================================
def test_loop(dataloader, model, loss_fn):
    model.eval() # Set to evaluation mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad(): # Disable gradient graph construction
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

# =====================================================================
# 6. Epoch Execution & Weight Persistence
# =====================================================================
epochs = 5
print("\nStarting Training Execution...")
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

# Save learned parameters
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# Verify loading
loaded_model = NeuralNetwork().to(device)
loaded_model.load_state_dict(torch.load("model.pth", map_location=device))
loaded_model.eval()
print("Successfully loaded model.pth into new instance!")
```

---

## Topic 1: Data ready; MLP / CNN / ViT (00:16–02:49)

> **Quick Intuition:** Before building a building, the lumber and concrete must be delivered to the construction site. Our data is downloaded, converted to tensors, and batched into shopping carts. Now we choose the right construction crane: an MLP for simple baseline classification, a CNN for vision filters, or a Transformer for large-scale modeling.

### Where this sits on the master map
We begin at **Stage 1: Hardware & Dataflow Setup**. The previous tutorial configured torchvision downloads and `ToTensor()` transformations. Now we transition from data preparation to selecting the neural architecture for image classification. See the [MLP foundation warm-up](./PREREQUISITES.md#p1-mlp).

### Board / screenshot

![Composite ch01](screenshots/composites/ch01-topic-01-transforms-then-build-mlp-cnn-vit-panel1of1.png)
*Board: The instructor confirms data readiness after `ToTensor()`, defines the image classification objective on Fashion-MNIST, and introduces the three primary architectural candidates: Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), and Vision Transformer (ViT).*

### What he is establishing
The tutorial begins by establishing that the dataset pipeline is fully initialized. The mandatory preprocessing transform is `transforms.ToTensor()`, which converts raw PIL images or NumPy arrays in range $[0, 255]$ into floating-point PyTorch tensors in range $[0.0, 1.0]$. While optional data augmentations such as `RandomCrop` or `Resize` exist in production workflows, the instructor defers them to keep the focus purely on model architecture.

With data ready, the engineering task is **image classification** on Fashion-MNIST (assigning each $28\times 28$ grayscale clothing image to one of 10 discrete categories). The instructor explains that while image classification is the concrete testbed, identical architectural principles generalize across other data modalities including tabular data, speech audio, and NLP tokens.

A common beginner mistake is assuming that any deep neural network will automatically understand 2D image structure without architectural consideration. The wrong move is expecting an MLP to recognize spatial patterns across shifted images; an MLP flattens pixels and cannot exploit translation invariance. In contrast, the right move matches the network architecture to the data domain:
1. **Multi-Layer Perceptrons (MLPs):** The simplest fully connected architecture, where all 784 pixel brightness values are flattened into an unrolled 1D vector and passed through dense matrix multiplications.
2. **Convolutional Neural Networks (CNNs):** Spatial architectures using sliding 2D convolution kernels with parameter sharing, which preserve translation invariance and local 2D pixel neighborhoods.
3. **Vision Transformers (ViTs):** Modern attention-based architectures dividing images into patch tokens and processing them via multi-head self-attention mechanisms.

While CNNs and Transformers dominate modern computer vision benchmarks, the instructor selects the **MLP** as today's foundational model to teach `nn.Module` subclassing, tensor flattening, and the core optimization loop mechanics before introducing convolutional layers.

You can now explain the three fundamental neural architectures for image classification and justify selecting an MLP as the starting baseline. What is still missing is setting up the computational environment and importing the necessary PyTorch modules.

### Analogy for this topic only
A package delivery hub sorts 10 categories of parcels. The raw delivery parcels have been unloaded and stacked onto standard wooden pallets (`ToTensor`). Now the hub manager must choose a sorting mechanism: a manual assembly line where clerks inspect the entire unrolled parcel at once (MLP), a series of specialized magnifying scanners that glide over the package surface looking for local postal stamps (CNN), or a satellite tracking network that analyzes relationships between all parcel corners simultaneously (ViT).

What breaks if you assume an MLP sees images like a human? If you shuffle all 784 pixel locations with a fixed permutation, a human can no longer recognize the shoe, but an MLP achieves the exact same classification loss because it treats pixels as an unordered list of features.

In lecture words: the wooden pallets are the prepared tensors, and the manual sorting line is the Multi-Layer Perceptron (MLP) built today.

### Local picture

```
   DATA PREPARATION COMPLETED (W1_T2):
   Raw Fashion-MNIST ──> ToTensor() ──> DataLoader(batch_size=64)
                                                │
                                                ▼
   ARCHITECTURAL CANDIDATE SELECTION:
   ┌───────────────────────┬───────────────────────┬───────────────────────┐
   │ Multi-Layer Perceptron│ Convolutional Net     │ Vision Transformer    │
   │ (MLP - Built Today)   │ (CNN - Previewed T10) │ (ViT - Future Scope)  │
   │ 784 -> 512 -> 512-> 10│ Conv2d -> MaxPool2d   │ Patches -> Attention  │
   └───────────────────────┴───────────────────────┴───────────────────────┘
```
*Notice: The pipeline transitions from data loading to neural network construction, selecting an MLP for foundational clarity.*

### Bridge
With the architectural choice settled on an MLP, we must now configure the hardware runtime in Google Colab and import the necessary PyTorch libraries.

---

## Topic 2: Choices, imports, Colab GPU (02:49–05:46)

> **Quick Intuition:** When building a high-performance race car, you first select the garage workspace and check whether you have access to a V8 supercharged engine (NVIDIA GPU) or a standard family sedan engine (CPU).

### Where this sits on the master map
We remain in **Stage 1: Hardware & Dataflow Setup**. Here we configure the execution device (`"cuda"` vs. `"cpu"`), understand Google Colab hardware acceleration, and import the core `torch.nn` package.

### Board / screenshot

![Composite ch02](screenshots/composites/ch02-topic-02-imports-colab-gpu-cuda-panel1of1.png)
*Board: The instructor outlines MLP design choices (layers, hidden units, activations), imports `torch`, `nn`, `DataLoader`, and `datasets`, and demonstrates Google Colab runtime selection (`cuda` vs. `cpu`).*

### What he is establishing
Designing a Multi-Layer Perceptron involves three fundamental engineering hyperparameters:
1. **Depth (Number of Layers):** How many linear transformations to stack in sequence.
2. **Width (Nodes per Layer):** The hidden dimensionality ($d_{\text{hidden}}$) of intermediate feature representations.
3. **Activation Functions:** The specific non-linear functions (e.g. ReLU, LeakyReLU, Sigmoid) placed between linear layers.

The instructor imports the foundational libraries: `import os`, `import torch`, `from torch import nn`, along with `DataLoader`, `datasets`, and `transforms` from `torchvision`. The core namespace `torch.nn` contains all neural network building blocks (layers, activations, containers, and loss functions).

A critical operational pitfall occurs in Google Colab: under *Runtime $\to$ Change runtime type*, users can select between **CPU**, **T4 GPU**, and **TPU**. The wrong move is assuming that switching runtimes in Colab preserves variables already created in memory. In reality, switching the hardware accelerator terminates the active Python virtual machine, clearing all notebook state and requiring every prior cell to be re-run.

To prevent hardware hard-coding errors where code fails when moved from a GPU server to a CPU laptop, the instructor introduces the standard device query idiom:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
The wrong approach is writing `model.cuda()` unconditionally, which crashes on machines without an NVIDIA GPU. The right approach evaluates `torch.cuda.is_available()`, assigning `"cuda"` when NVIDIA drivers and hardware are present, and falling back gracefully to `"cpu"`.

You can now configure an execution device and import the required PyTorch modules. What is still missing is subclassing `nn.Module` to build the neural network.

### Analogy for this topic only
A computational chemist prepares to run molecular dynamics simulations. The chemist checks whether the laboratory has booked time on the university's supercomputer cluster (GPU) or must run the calculations on a desktop workstation (CPU). Switching from the desktop to the supercomputer requires logging into a brand new terminal session, so all setup scripts must be rerun from scratch.

What breaks if you write scripts hardcoded to supercomputer commands only? The moment you test the code offline on your laptop, the entire pipeline crashes on line 1 because the hardware driver is missing.

In lecture words: the supercomputer is CUDA / GPU, the desktop is CPU, and the device check string ensures the script runs automatically on whichever hardware is available.

### Local picture

```
   GOOGLE COLAB RUNTIME SELECTION:
   ┌─────────────────────────────────────────────────────────────┐
   │ Runtime -> Change runtime type -> Hardware Accelerator      │
   │  [ ] CPU        [*] T4 GPU (CUDA)        [ ] TPU            │
   │  (Warning: Switching runtime terminates current session!)   │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   HARDWARE-AGNOSTIC DEVICE BINDING:
   device = "cuda" if torch.cuda.is_available() else "cpu"
```
*Notice: Selecting the hardware accelerator sets the device string that routes all subsequent model weights and data batches.*

### Bridge
With the CUDA device verified, we now write the object-oriented Python class that defines the neural network architecture.

---

## Topic 3: NeuralNetwork, flatten, 10 logits (05:46–09:11)

> **Quick Intuition:** In Python, creating a neural network is like creating a specialized employee class. You register the employee with the company payroll (`super().__init__()`), give them a paper shredder to unroll 2D photos into 1D ribbons (`nn.Flatten()`), and specify that their job ends with 10 output confidence scores.

### Where this sits on the master map
We step into **Stage 2: Neural Architecture (`nn.Module`)**. Here we subclass `nn.Module`, call `super().__init__()`, instantiate `nn.Flatten()`, and fix the input and output dimensionalities ($784 \to 10$). Review the [flattening warm-up](./PREREQUISITES.md#p2-flatten) and [`nn.Module` lifecycle warm-up](./PREREQUISITES.md#p4-module).

### Board / screenshot

![Composite ch03](screenshots/composites/ch03-topic-03-nn-module-flatten-784-10-panel1of1.png)
*Board: Subclassing `nn.Module`: `class NeuralNetwork(nn.Module)`, calling `super().__init__()`, initializing `self.flatten = nn.Flatten()`, and setting input dimension to 784 and output logits to 10.*

### What he is establishing
In PyTorch, custom neural network architectures are defined as object-oriented Python classes that inherit from `torch.nn.Module`.

The constructor `__init__(self)` must begin with:
```python
super().__init__()
```
A frequent bug made by beginner engineers is omitting `super().__init__()`. Without this call, PyTorch cannot initialize the underlying C++ hooks and internal dictionaries (`_parameters`, `_modules`, `_buffers`), causing immediate runtime errors when attempting to track weights or migrate models to GPU.

The instructor analyzes the dimensional requirements of Fashion-MNIST:
- Each input image is a single-channel grayscale image of size $28 \times 28$ pixels.
- The wrong approach is passing a 4D tensor `(B, 1, 28, 28)` directly into a linear matrix multiplication, which causes a dimension mismatch error.
- The right approach uses `self.flatten = nn.Flatten()`, which stacks rows end-to-end to vectorize each image into a 1D column vector of size $28 \times 28 = 784$ while keeping batch size $B$ intact (`(B, 784)`).

The output dimensionality is strictly constrained by the dataset: Fashion-MNIST contains **10 mutually exclusive clothing classes**. Therefore, the network's final layer must output a vector of **10 raw logits**, which are subsequently mapped to probabilities via Softmax. While the input dimension ($784$) and output dimension ($10$) are fixed by the problem, the number of hidden layers and their widths ($512$) are architectural design choices.

You can now define an `nn.Module` subclass, invoke `super().__init__()`, and configure `nn.Flatten()` to bridge 2D image tensors to 1D feature vectors. What is still missing is defining the hidden linear layers and non-linear activations inside `nn.Sequential`.

### Analogy for this topic only
An automated mail sorting facility accepts photographic postcards ($28\times 28$ pixels). The first machine on the sorting conveyor is a rotary slicer (`nn.Flatten()`) that cuts the 28 rows of the postcard and lays them out into a single continuous ribbon of 784 dots. At the end of the conveyor line sit 10 sorting bins, waiting to receive score stamps for each package.

What breaks if the rotary slicer cuts across the batch instead of across individual images? You would merge 64 distinct customer postcards into one giant monster document, destroying individual package identities. `nn.Flatten(start_dim=1)` guarantees each postcard is unrolled independently.

In lecture words: the postcard is the Fashion-MNIST image, the rotary slicer is `nn.Flatten()`, and the 10 bins correspond to the 10 output logits.

### Local picture

```
   FASHION-MNIST IMAGE TENSOR:                SPATIAL FLATTENING OPERATION:
   ┌───────────────────────────┐              ┌───────────────────────────────────────┐
   │ Shape: (B, 1, 28, 28)     │  nn.Flatten  │ Shape: (B, 784)                       │
   │ 28 rows x 28 columns      │  ──────────> │ Vectorized pixel values:              │
   │ Single grayscale channel  │              │ [ p0,0, p0,1, ..., p27,26, p27,27 ]   │
   └───────────────────────────┘              └───────────────────────────────────────┘
                                                                  │
                                                                  ▼
                                              Fixed Output Dimension: 10 Class Logits
```
*Notice: `nn.Flatten()` unrolls the $28\times 28$ spatial grid into a 784-dimensional feature vector, preserving the batch size $B$.*

### Bridge
Now that the input vectorization and output dimensions are established, we construct the hidden transformation layers using `nn.Sequential` and `nn.Linear`.

---

## Topic 4: Sequential; Linear 784 to 512 (09:11–13:07)

> **Quick Intuition:** `nn.Sequential` is like an industrial pipeline: the fluid that leaves pipe 1 flows straight into pipe 2 without any manual pumping in between. `nn.Linear` is the pump itself, mixing and multiplying 784 incoming streams into 512 outgoing streams.

### Where this sits on the master map
We continue in **Stage 2: Neural Architecture (`nn.Module`)**. Here we define `nn.Sequential` and construct the first hidden linear mapping: `nn.Linear(28*28, 512)`. Review the [linear layer warm-up](./PREREQUISITES.md#p1-mlp).

### Board / screenshot

![Composite ch04](screenshots/composites/ch04-topic-04-sequential-linear-wx-b-panel1of1.png)
*Board: Defining `self.linear_relu_stack = nn.Sequential(...)`, constructing `nn.Linear(28*28, 512)`, and deriving the affine transformation $y = W^\top x + b$ with weight matrix and bias vector.*

### What he is establishing
To structure intermediate transformations cleanly without manually writing nested function calls, the instructor introduces the `nn.Sequential` container:
```python
self.linear_relu_stack = nn.Sequential(...)
```
`nn.Sequential` chains layers together such that the output of layer $i$ is piped directly into the input of layer $i+1$.

The first computational block inside `nn.Sequential` is a dense linear layer:
```python
nn.Linear(28 * 28, 512)
```
The wrong move is treating `nn.Linear` as a pure linear matrix multiplication $y = W x$. In PyTorch, `nn.Linear` implements an **affine transformation**:
$$y = x W^\top + b$$
where:
- $x \in \mathbb{R}^{B \times 784}$ is the input feature batch.
- $W \in \mathbb{R}^{512 \times 784}$ is the learnable weight matrix.
- $b \in \mathbb{R}^{512}$ is the learnable bias offset vector.
- $y \in \mathbb{R}^{B \times 512}$ is the transformed output feature batch.

The instructor calculates the parameter count:
$$\text{Parameters} = (784 \times 512) + 512 = 401,408 + 512 = \mathbf{401,920} \text{ parameters}$$

The bias term $b$ is mathematically essential: without $b$, an input vector of all zeros ($x = \mathbf{0}$) would be forced to output zero ($y = \mathbf{0}$), anchoring the decision boundary to the origin. Adding $b$ allows the decision boundary to shift freely across the feature space.

You can now explain how `nn.Sequential` orchestrates layer execution and calculate the weight matrix and bias vector dimensions for any linear layer. What is still missing is inserting the non-linear activation functions between successive linear layers.

### Analogy for this topic only
An audio mixing board takes 784 raw microphone audio channels. The sound engineer constructs a mixing bank (`nn.Linear(784, 512)`) with 401,408 volume slider knobs ($W$) and 512 master pre-amp trim dials ($b$) to blend the 784 microphone feeds into 512 balanced intermediate audio tracks.

What breaks if the audio engineer forgets to install the trim dials ($b$)? If all 784 musicians stop playing simultaneously (input is zero), the speakers are forced to absolute zero volume; the engineer cannot add background ambiance or baseline tone.

In lecture words: the microphone feeds are input pixels, the volume knobs are weights $W$, the trim dials are biases $b$, and the intermediate tracks are the 512 hidden activations.

### Local picture

```
   FLATTENED INPUT (784 Features)
   [ x_1, x_2, ..., x_784 ]
         │
         │   Weight Matrix W: (512 x 784)
         ├─── Multiplying 784 inputs across 512 nodes
         │   Bias Vector b: (512,)
         ▼
   AFFINE TRANSFORMATION: z = x * W^T + b
         │
         ▼
   OUTPUT ACTIVATIONS (512 Hidden Nodes)
   [ z_1, z_2, ..., z_512 ]
```
*Notice: `nn.Linear(784, 512)` maps 784 input dimensions to 512 hidden dimensions via learned weights and biases.*

### Bridge
If we simply stack another linear layer directly onto this 512-dimensional output, the two matrices would multiply together and collapse into a single linear layer. To prevent this collapse, we must insert non-linear activation functions.

---

## Topic 5: ReLU stack, logits, P(Y|X) (13:07–17:36)

> **Quick Intuition:** If you fold a piece of paper along a straight crease, you have created a bend. Stacking multiple folded creases allows you to origami the paper into any complex shape. `nn.ReLU` is the fold that gives the network the power to sculpt intricate decision boundaries.

### Where this sits on the master map
We complete **Stage 2: Neural Architecture (`nn.Module`)**. Here we assemble the complete 3-layer MLP stack with `nn.ReLU` non-linearities, explain why the final layer omits activation functions, and contrast raw logits with posterior probabilities $P(Y \mid X)$. Review the [logits vs. softmax warm-up](./PREREQUISITES.md#p3-logits).

### Board / screenshot

![Composite ch05](screenshots/composites/ch05-topic-05-relu-stack-logits-softmax-panel1of1.png)
*Board: The complete MLP pipeline: `Linear(784, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 10)`, explaining why the final layer outputs raw logits and contrasting logits with Softmax probabilities $P(Y \mid X)$.*

### What he is establishing
The instructor completes the definition of `self.linear_relu_stack`:
```python
self.linear_relu_stack = nn.Sequential(
    nn.Linear(28 * 28, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)
```

The non-linear activation function chosen is the **Rectified Linear Unit (`nn.ReLU`)**:
$$\text{ReLU}(z) = \max(0, z)$$
`nn.ReLU` applies an elementwise hinge: positive inputs pass through with derivative $1$, while negative inputs are clamped to $0$ with derivative $0$.

The instructor emphasizes a critical contrast: the wrong move is placing an activation function (like ReLU or Softmax) after the final linear layer (`Linear(512, 10)`).
- If you place a ReLU after the final layer, all negative class scores are truncated to zero, destroying comparative ranking information.
- If you place a Softmax after the final layer, computing `nn.CrossEntropyLoss` on the output performs a "double softmax", resulting in squashed gradients and severe numerical instability.

The right move emits **raw, unbounded logits** ($z \in \mathbb{R}^{10}$) directly from `forward()`.
- **Logits ($z$):** Unconstrained real scores ($-\infty < z_k < +\infty$).
- **Softmax ($\sigma(z)$):** Normalized probabilities $P(Y=k \mid X) \in [0, 1]$ where $\sum_{k=0}^9 P(Y=k \mid X) = 1.0$.

When calculating losses, PyTorch's `nn.CrossEntropyLoss` accepts raw logits directly and combines log-softmax with negative log-likelihood in a single numerically stable formula.

You can now explain the role of ReLU activations in deep networks and justify why the final output layer returns raw logits without activation. What is still missing is implementing the `forward()` method and executing a test prediction on hardware.

### Analogy for this topic only
A water filtration plant runs dirty river water through three chambers. Chambers 1 and 2 each have a one-way pressure valve (`ReLU`) that lets water flow forward but snaps shut if pressure reverses (negative values). Chamber 3 has 10 open output pressure gauges (logits) measuring the final flow rates. You do not cap the gauges with valves—you leave them open so the pressure meters can swing freely into positive or negative readings.

What breaks if you install a one-way pressure valve on the final gauges? Any gauge reading negative pressure would be forced to read zero, making it impossible to distinguish between a slightly bad guess and a disastrously wrong guess.

In lecture words: the pressure valves are `nn.ReLU()` activations, and the 10 open gauges are the raw output logits.

### Local picture

```
   INPUT (784) ──> [ Linear(784, 512) ] ──> (512) ──> [ ReLU Bend ] ──> (512)
                                                                            │
   ┌────────────────────────────────────────────────────────────────────────┘
   │
   ▼
   (512) ──> [ Linear(512, 512) ] ──> (512) ──> [ ReLU Bend ] ──> (512)
                                                                      │
   ┌──────────────────────────────────────────────────────────────────┘
   │
   ▼
   (512) ──> [ Linear(512, 10) ] ──> 10 Raw Logits (NO ReLU!)
                                            │
                                            ├── Softmax ──> Probabilities P(Y|X)
                                            └── Argmax  ──> Predicted Class Label
```
*Notice: ReLUs sit between hidden layers, but the final output layer emits raw, unconstrained logits.*

### Bridge
Now that the architecture is fully constructed inside `__init__`, we implement the `forward()` method, migrate the model to GPU memory via `.to(device)`, and run a dummy prediction.

---

## Topic 6: .to(device) and a dummy predict (17:36–19:58)

> **Quick Intuition:** Having a blueprint of a machine on your laptop doesn't manufacture parts. You must upload the blueprint to the factory robot on the assembly floor (`.to(device)`) and feed a test scrap of material through the machine to verify that all motors spin.

### Where this sits on the master map
We step into **Stage 3: Raw Logits & Loss Chokepoint**. Here we instantiate the network, migrate its parameter tensors to GPU memory using `.to(device)`, and verify the forward pass by feeding a dummy input tensor. Review the [`nn.Module` forward execution warm-up](./PREREQUISITES.md#p4-module).

### Board / screenshot

![Composite ch06](screenshots/composites/ch06-topic-06-forward-to-device-dummy-predict-panel1of1.png)
*Board: Implementing `forward(self, x)`: `x = self.flatten(x)`, `logits = self.linear_relu_stack(x)`, returning `logits`. Instantiating `model = NeuralNetwork().to(device)`, printing architecture, and running a dummy forward pass on `X = torch.rand(1, 28, 28, device=device)`.*

### What he is establishing
First, the instructor implements the execution dataflow:
```python
def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits
```
When running the model, the wrong move is calling `model.forward(x)` directly. Calling `model.forward(x)` bypasses PyTorch's internal execution hooks (profilers, backward gradient references, activation trackers). The right move is calling `model(x)`, which triggers `__call__` and runs all registration hooks before invoking `forward()`.

Second, the instructor instantiates the model on the target hardware:
```python
model = NeuralNetwork().to(device)
print(model)
```
Until `.to(device)` is invoked, model weights reside in CPU memory. Moving the model to `"cuda"` allocates the parameter weights directly into NVIDIA VRAM.

Third, the instructor runs a dummy verification pass:
```python
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```
The wrong move is passing a CPU tensor into a GPU model, which triggers a `RuntimeError: Expected all tensors to be on the same device`. The right move ensures `X` and `model` share the identical device string (`device=device`). Feeding random input produces an arbitrary class prediction (e.g. class 7), verifying that all tensor shapes align across layers.

You can now instantiate a neural network, migrate it to GPU, and verify its forward pass using dummy inputs. What is still missing is setting up the loss function and optimizer to train the weights.

### Analogy for this topic only
A car manufacturer builds a prototype engine on a test bench. Before taking the car onto a public highway, the engineers mount the engine into the vehicle chassis (`.to(device)`), pour in a single cup of test fuel (`dummy X`), turn the key, and verify that the drive shaft spins without stalling.

What breaks if the test fuel is stored in a separate warehouse across town? The engine cannot ignite because the fuel line and combustion chamber reside in disconnected physical locations (CPU vs. GPU memory mismatch).

In lecture words: the test bench is CPU memory, mounting into the chassis is `.to(device)`, and the single cup of test fuel is the dummy random input tensor.

### Local picture

```
   MODEL INSTANTIATION & GPU MIGRATION:
   NeuralNetwork() (In CPU RAM) ──> .to(device) ──> model (In GPU VRAM)
                                                          │
   DUMMY FORWARD PASS VALIDATION:                         │
   X: torch.rand(1, 28, 28, device=device)                │
         │                                                │
         ▼                                                │
   model(X) ──> logits (1 x 10) ──> Softmax ──> argmax ──> Predicted Class ID (e.g. 7)
```
*Notice: `.to(device)` deploys model parameters to GPU memory, and a dummy forward pass verifies that data flows cleanly through all layers.*

### Bridge
Now that the network is operational, we must define the objective function that measures prediction errors and the optimizer that updates the weights.

---

## Topic 7: SGD / Adam, CE, hyperparameters (19:58–23:54)

> **Quick Intuition:** Training a network is like learning archery. You need a target board that scores how far off-center your arrow landed (Cross-Entropy Loss), and a coach who tells you how many millimeters to adjust your bow grip for the next shot (the Optimizer and Learning Rate).

### Where this sits on the master map
We step into **Stage 3 & 4: Loss & Optimization Setup**. Here we define hyperparameters (epochs, batch size, learning rate), configure `nn.CrossEntropyLoss()`, and initialize `torch.optim.SGD`. Review the [gradient descent and cross-entropy warm-up](./PREREQUISITES.md#p5-gd).

### Board / screenshot

![Composite ch07](screenshots/composites/ch07-topic-07-sgd-adam-ce-hyperparams-panel1of1.png)
*Board: Defining hyperparameters (learning rate $\alpha=10^{-3}$, batch size $64$, epochs $10$), instantiating `loss_fn = nn.CrossEntropyLoss()`, and initializing `optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)`.*

### What he is establishing
The instructor introduces the three fundamental **hyperparameters** governing optimization:
1. **Learning Rate ($\eta$ or $\alpha$):** The scaling multiplier applied to gradient updates. Setting $\eta = 10^{-3} = 0.001$.
2. **Batch Size ($B$):** The number of training examples processed before updating parameters ($B = 64$).
3. **Epochs:** The total number of complete cycles through the entire training dataset ($10$ epochs).

Next, the loss function is configured:
```python
loss_fn = nn.CrossEntropyLoss()
```
The wrong approach for multi-class classification is using Mean Squared Error (MSE), which suffers from gradient saturation when outputs approach 0 or 1. The right approach is **Categorical Cross-Entropy Loss**, which evaluates the negative log-likelihood of the ground-truth class:
$$L = -\frac{1}{B} \sum_{i=1}^B \log\left( \frac{e^{z_{i, y_i}}}{\sum_{j=0}^9 e^{z_{i, j}}} \right)$$
If the network assigns high probability to the true class $y_i$, $L \to 0$. If it assigns near-zero probability to the correct class, $L$ grows logarithmically to infinity.

Finally, the optimizer is instantiated:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```
The optimizer receives the model parameter tensors and executes the parameter update rule:
$$\theta \leftarrow \theta - \eta \nabla_\theta L$$
While standard Stochastic Gradient Descent (SGD) is used in today's live code, the instructor highlights that **Adam** (`torch.optim.Adam`) utilizes running momentum and adaptive per-parameter learning rates for faster convergence.

You can now configure training hyperparameters, explain how Cross-Entropy loss evaluates multi-class predictions, and instantiate an SGD optimizer. What is still missing is assembling these pieces into the 4-step training loop.

### Analogy for this topic only
A student studies for a 10-subject exam. The student chooses to study in 64-flashcard sessions (batch size), plans to review the entire syllabus 10 times (epochs), and sets a moderate revision pace (learning rate). The grading rubric (Cross-Entropy loss) docks points logarithmically for every wrong answer, and the study coach (SGD) directs the student to focus attention on the subjects with the highest error.

What breaks if the student sets the revision pace (learning rate) to 100x speed? The student wildly overreacts to every single missed question, discarding previously mastered rules and failing the entire exam.

In lecture words: the flashcard sessions are mini-batches, the grading rubric is `nn.CrossEntropyLoss()`, and the study coach is `torch.optim.SGD`.

### Local picture

```
   HYPERPARAMETER CONFIGURATION:
   learning_rate = 1e-3,  batch_size = 64,  epochs = 10
              │
              ├───> loss_fn = nn.CrossEntropyLoss()
              │     Combines LogSoftmax + Negative Log-Likelihood
              │
              └───> optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
                    Applies parameter updates: θ <- θ - η * ∇_θ L
```
*Notice: The loss function and optimizer are bound to the model parameters, ready to execute training steps.*

### Bridge
With the loss function and optimizer initialized, we now write the canonical 4-step inner training loop that iterates over data batches and updates model weights.

---

## Topic 8: train_loop: four steps (23:54–28:38)

> **Quick Intuition:** A carpenter building chairs follows a 4-step loop: (1) measure the wood, (2) check the error with a ruler, (3) calculate where to cut, (4) cut the wood, and (5) sweep the sawdust off the workbench before starting the next chair (`zero_grad()`).

### Where this sits on the master map
We arrive at **Stage 4: Canonical Training Step (`train_loop`)**. Here we implement the core 4-step backpropagation cycle, explain why `optimizer.zero_grad()` is mandatory, and display batch progress. Review the [training dynamics warm-up](./PREREQUISITES.md#p6-epoch).

### Board / screenshot

![Composite ch08](screenshots/composites/ch08-topic-08-train-loop-four-steps-panel1of1.png)
*Board: The canonical 4-step training loop: `pred = model(X)`, `loss = loss_fn(pred, y)`, `loss.backward()`, `optimizer.step()`, and `optimizer.zero_grad()`, with batch progress logging.*

### What he is establishing
The instructor implements the training loop function:
```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # Enable training mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
```

Inside the loop, the **canonical 4-step optimization sequence** is executed:
1. **Predict:** `pred = model(X)` — Generates $(64, 10)$ prediction logits.
2. **Loss:** `loss = loss_fn(pred, y)` — Computes the scalar cross-entropy loss.
3. **Backward:** `loss.backward()` — Autograd calculates partial derivatives $\frac{\partial L}{\partial W}$ via the Chain Rule and stores them in `param.grad`.
4. **Step:** `optimizer.step()` — Updates weights using the computed gradients: $W \leftarrow W - \eta \cdot W.\text{grad}$.

Following these four steps, the instructor emphasizes the essential gradient flush:
```python
optimizer.zero_grad()
```
The wrong move is omitting `optimizer.zero_grad()`. In PyTorch, parameter gradient buffers accumulate values across successive `loss.backward()` calls. If not reset, batch 2's gradients will be added onto batch 1's gradients, causing gradient magnitudes to explode and training to diverge. The right move flushes all `.grad` tensors to zero before the next iteration.

To log progress, the instructor tracks loss values:
```python
if batch % 100 == 0:
    loss_val = loss.item()
    current = batch * len(X) + len(X)
    print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
```
The wrong move is logging with `total_loss += loss`, which stores references to the entire autograd computation graph and rapidly triggers GPU Out-Of-Memory errors. The right move calls `loss.item()`, extracting a pure Python `float` and dropping graph references.

You can now implement the canonical 4-step PyTorch training loop and explain the necessity of `optimizer.zero_grad()` and `loss.item()`. What is still missing is implementing the evaluation loop to measure model accuracy on unseen test data.

### Analogy for this topic only
An archery student practices on a target range. On each round of 64 arrows:
1. The student shoots 64 arrows (`pred = model(X)`).
2. The coach measures the distance from the bullseye (`loss = loss_fn(pred, y)`).
3. The coach calculates how the student's arm angles must change (`loss.backward()`).
4. The student adjusts their muscle posture (`optimizer.step()`).
5. The student erases the chalk marks from the whiteboard (`optimizer.zero_grad()`) so the next round's feedback isn't added to the previous round.

What breaks if the student forgets to erase the whiteboard (`optimizer.zero_grad()`)? By round 20, 20 rounds of accumulated corrections are applied simultaneously, wrenching the student's arm so violently that the arrows fly completely off the target range.

In lecture words: shooting the arrows is the forward pass, measuring the distance is the loss, muscle adjustment is `optimizer.step()`, and wiping the whiteboard is `optimizer.zero_grad()`.

### Local picture

```
   THE CANONICAL 4-STEP TRAINING CYCLE:
   ┌─────────────────────────────────────────────────────────────┐
   │ (1) pred = model(X)           ──> Forward Prediction        │
   │ (2) loss = loss_fn(pred, y)   ──> Cross-Entropy Loss        │
   │ (3) loss.backward()           ──> Autograd Backpropagation  │
   │ (4) optimizer.step()          ──> Parameter Update (SGD)    │
   │ (5) optimizer.zero_grad()     ──> Reset Gradient Buffers!   │
   └─────────────────────────────────────────────────────────────┘
```
*Notice: The 4-step sequence executes for every mini-batch, with `zero_grad()` clearing the gradient buffers.*

### Bridge
Having built the training loop to update model weights, we must now build the evaluation loop to test whether the network generalizes to unseen data without updating weights.

---

## Topic 9: test_loop, accuracy, curves (28:38–36:31)

> **Quick Intuition:** Taking a practice quiz during study hours is different from taking the final exam. During the final exam, you don't have an instructor giving you correction hints (no gradients), and your answers are strictly graded as either right or wrong (accuracy percentage).

### Where this sits on the master map
We step into **Stage 5: Evaluation & Serialization (`test_loop`)**. Here we implement the evaluation loop using `model.eval()`, disable gradient tracking with `with torch.no_grad():`, compute classification accuracy, and analyze learning curves across 10 epochs. Review the [eval mode warm-up](./PREREQUISITES.md#p7-eval).

### Board / screenshot

![Composite ch09](screenshots/composites/ch09-topic-09-test-loop-accuracy-curves-panel1of1.png)
*Board: Implementing `test_loop(dataloader, model, loss_fn)`: setting `model.eval()`, using `with torch.no_grad():`, computing cumulative test loss and accuracy via `(pred.argmax(1) == y).type(torch.float).sum().item()`, and reviewing the 10-epoch training progression.*

### What he is establishing
The instructor implements the evaluation loop:
```python
def test_loop(dataloader, model, loss_fn):
    model.eval() # Set model to evaluation mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad(): # Disable gradient graph construction
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
```

Two critical evaluation practices are established:
1. **`model.eval()`:** The wrong move is evaluating with the model still in training mode, which causes layers like `Dropout` to randomly drop features and `BatchNorm` to distort predictions. The right move calls `model.eval()`, ensuring deterministic evaluation behavior.
2. **`with torch.no_grad():`:** The wrong move is evaluating without a gradient context manager, which forces PyTorch to allocate intermediate activation graph nodes in GPU memory for all 10,000 test images. The right move wraps the loop in `with torch.no_grad():`, deactivating autograd history and reducing VRAM consumption to near zero.

To calculate test accuracy:
- `pred.argmax(1)` identifies the predicted class index with the highest logit score.
- `(pred.argmax(1) == y)` creates a boolean tensor checking prediction correctness.
- `.type(torch.float).sum().item()` casts booleans to 1.0/0.0, sums them, and extracts the total count of correct predictions.

The instructor executes the full 10-epoch training and validation run:
- **Epoch 1:** Accuracy begins at approximately **$42.4\%$** with loss $\approx 2.16$.
- Over 10 epochs, loss steadily declines while accuracy improves.
- **Epoch 10:** Accuracy reaches **$\approx 71.0\%$** (with variations around $68.7\%$).

The instructor analyzes the resulting learning curve: loss declines sharply initially before asymptotically flattening. When determining how many epochs to train, practitioners stop training when validation loss stops improving (early stopping) to avoid overfitting.

You can now implement a test evaluation loop using `model.eval()` and `torch.no_grad()` and calculate accuracy metrics. What is still missing is saving the trained model weights to disk for reuse.

### Analogy for this topic only
A flight simulator student finishes a 10-hour flight training module. For the final certification test, the flight instructor turns off all automated guidance prompts and recording cameras (`torch.no_grad()`). The student flies 200 simulated landing scenarios (`test_loader`), and the examiner tallies the exact percentage of successful runway touchdowns (accuracy).

What breaks if the instructor leaves the flight simulator in student training mode (`model.train()`)? Random simulated engine failures (Dropout) will continue firing during the final exam, penalizing the student unfairly on routine landings.

In lecture words: the certification test is `test_loop`, turning off guidance prompts is `torch.no_grad()`, and the touchdown tally is test accuracy.

### Local picture

```
   EVALUATION PIPELINE:
   model.eval() ──> with torch.no_grad():
                          │
                          ▼  Iterate over Test DataLoader
   Batch X ──> model(X) ──> pred.argmax(1) == y ──> Sum correct predictions
                                                          │
                                                          ▼
   Epoch 1:  Accuracy ~ 42.4%  |  Avg Loss: 2.16
   ...
   Epoch 10: Accuracy ~ 71.0%  |  Avg Loss: 0.84  (Model learned successfully!)
```
*Notice: Disabling gradients during evaluation enables fast, memory-efficient accuracy computation across test batches.*

### Bridge
Now that the model is fully trained and validated, we must save the learned parameter weights to disk so they can be reloaded without retraining from scratch.

---

## Topic 10: Save/load, CNN preview, GANs next (36:31–44:06)

> **Quick Intuition:** After baking an award-winning cake, you don't throw away the recipe and reinvent baking tomorrow. You write down the exact ingredient weights in a recipe book (`state_dict`) and save it in a drawer (`model.pth`) so you can bake the same cake instantly anytime.

### Where this sits on the master map
We arrive at **Stage 5: Serialization (`save/load`) & Course Horizon**. Here we serialize model weights to `.pth` files using `state_dict`, preview Convolutional Neural Network (CNN) parameters, recap the Week 1 foundations, and look ahead to coding GANs. Review the [model saving warm-up](./PREREQUISITES.md#p8-save).

### Board / screenshot

![Composite ch10](screenshots/composites/ch10-topic-10-save-load-cnn-recap-gans-panel1of1.png)
*Board: Saving and loading: `torch.save(model.state_dict(), "model.pth")` and `model.load_state_dict(torch.load("model.pth"))`. Previewing CNN layer parameters (in_channels, out_channels, kernel_size, stride, padding) and bridging to upcoming GAN coding tutorials.*

### What he is establishing
The instructor demonstrates how to serialize and persist learned model parameters:
```python
torch.save(model.state_dict(), "model.pth")
```
The wrong approach is using Python's standard `pickle.dump(model)` to serialize the entire class instance. If the class definition or folder structure is later refactored, pickled model objects fail to deserialize. The right approach saves `model.state_dict()`, an `OrderedDict` mapping parameter names to raw weight tensors.

To reload weights into a fresh instance:
```python
model = NeuralNetwork().to(device) # 1. Create clean instance
model.load_state_dict(torch.load("model.pth", map_location=device)) # 2. Load weight tensors
model.eval() # 3. Set to evaluation mode
```

Next, the instructor previews **Convolutional Neural Networks (CNNs)**:
- In a CNN, instead of flattening pixels immediately, 2D convolution layers (`nn.Conv2d`) extract local spatial feature hierarchies.
- Each convolutional layer requires configuring:
  - `in_channels`: Number of input feature maps (e.g., $1$ for grayscale, $3$ for RGB).
  - `out_channels`: Number of learned convolutional filters.
  - `kernel_size`: Spatial filter dimensions (e.g., $3 \times 3$ or $5 \times 5$).
  - `stride` and `padding`: Step size and zero-padding borders controlling output spatial resolution.
- A CNN acts as a spatial feature extractor, whose final feature maps are flattened and fed into an MLP classification head.
- **The Key Architectural Insight:** Switching from an MLP to a CNN requires modifying **only the `nn.Module` class definition**; the `train_loop`, `test_loop`, loss functions, and optimizers remain **100% identical**.

The tutorial concludes with a full recap of the Week 1 foundational trajectory:
1. **Tensors:** Multidimensional arrays with autograd tracking on GPUs.
2. **Datasets & DataLoaders:** Batching, shuffling, and multi-threaded loading.
3. **Transforms:** Normalizing and preparing image data (`ToTensor`).
4. **Model Building:** Subclassing `nn.Module`, defining layers, and implementing `forward()`.
5. **Optimization:** Running the 4-step gradient descent loop and evaluating performance.

With the core PyTorch engineering toolkit mastered, the instructor announces the next major milestone: **coding Generative Adversarial Networks (GANs)** from scratch.

You can now save and load PyTorch model weights using `state_dict`, explain the configuration parameters of convolutional layers, and connect this model-building pipeline to upcoming GAN implementations.

### Analogy for this topic only
An automotive factory builds two different vehicle models: a compact electric sedan (MLP) and an all-terrain utility truck (CNN). The robotic assembly line (`train_loop`), the quality inspection station (`test_loop`), the fuel supply (`DataLoader`), and the shipping warehouse (`model.pth`) remain completely unchanged. Only the vehicle chassis blueprint (`nn.Module`) is swapped out.

What breaks if you try to load the sedan's suspension weights (`state_dict`) into the heavy-duty truck chassis? PyTorch's `load_state_dict()` throws a `KeyError / SizeMismatchError` because the tensor shapes between the two blueprints do not match.

In lecture words: the vehicle chassis blueprint is the neural network class, while the shared assembly line is the standardized training and evaluation loop.

### Local picture

```
   MODEL PERSISTENCE (STATE_DICT):
   model (In VRAM) ──> model.state_dict() ──> torch.save() ──> "model.pth" on disk
                                                                      │
   RELOAD INTO NEW INSTANCE:                                          │
   fresh_model = NeuralNetwork() ──> load_state_dict(torch.load()) <──┘
   
   =============================================================================
   ROADMAP RECAP & HORIZON:
   [ Tensors ] ──> [ DataLoaders ] ──> [ MLP Model ] ──> [ Train/Test Loops ]
                                                                │
                                                                ▼
                                                    NEXT: Coding GANs from Scratch!
```
*Notice: Saving `state_dict` enables lightweight model persistence, and the standard training loop transfers directly to future GAN architectures.*

### Bridge
With the PyTorch model building, optimization, and serialization lifecycle complete, you now possess the core deep learning skillset required to build custom generative architectures, discriminator critics, and generator samplers.

---

## Apply it (scenarios)

These real-world debugging scenarios illustrate common failure modes encountered when building and training PyTorch models in production:

### Scenario 1: The Gradient Accumulation / Forgotten `zero_grad()` Bug
- **Symptom:** During training, the training loss decreases slightly in epoch 1, but by epoch 2 the loss explodes to `NaN` or oscillates wildly. The model fails to converge, even with a tiny learning rate ($\eta = 10^{-5}$).
- **Root Cause Analysis:** PyTorch does not reset parameter gradient buffers (`param.grad`) automatically between iterations. If `optimizer.zero_grad()` is omitted from the batch loop, gradients calculated across successive mini-batches continuously accumulate ($G_{\text{effective}} = \sum_{b=1}^B G_b$). By batch 100, the effective gradient magnitude is 100 times larger than intended, causing catastrophic weight explosion.
- **Production Code Fix:** Ensure `optimizer.zero_grad()` is called inside the batch loop (either immediately before `loss.backward()` or right after `optimizer.step()`).

```python
# FIX: Proper gradient buffer flushing in training loop
def robust_train_step(batch_x, batch_y, model, loss_fn, optimizer):
    optimizer.zero_grad()        # 1. Flush accumulated gradients BEFORE backward
    pred = model(batch_x)        # 2. Forward pass
    loss = loss_fn(pred, batch_y)# 3. Compute loss
    loss.backward()              # 4. Populate fresh gradients
    optimizer.step()             # 5. Apply weight update
    return loss.item()
```

### Scenario 2: Evaluation Memory Leak & BatchNorm Inconsistency
- **Symptom:** The model trains smoothly, but when evaluating on the validation set, the GPU suddenly crashes with `torch.cuda.OutOfMemoryError: CUDA out of memory`. Additionally, validation accuracy fluctuates erratically between evaluation runs on the exact same data.
- **Root Cause Analysis:** Two distinct bugs:
  1. Evaluating without `with torch.no_grad():` forces PyTorch to build a dynamic computation graph for the entire validation dataset, storing all intermediate layer activations in VRAM until memory is exhausted.
  2. Forgetting `model.eval()` leaves layers like `nn.Dropout` and `nn.BatchNorm2d` in training mode. Dropout continues randomly zeroing activations during testing, producing non-deterministic evaluation scores.
- **Production Code Fix:** Always wrap evaluation routines with `model.eval()` and `with torch.no_grad():`.

```python
# FIX: Memory-safe and deterministic evaluation routine
def safe_evaluate(test_loader, model, device):
    model.eval() # 1. Freeze Dropout and BatchNorm statistics
    correct = 0
    total = 0
    
    with torch.no_grad(): # 2. Deactivate autograd graph tracking to save VRAM
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
    return correct / total
```

---

## External references

A curated collection of authoritative documentation, university lecture notes, and engineering guides expanding on the PyTorch model-building principles established in this tutorial:

| Resource | Matches lecture… | Why it helps |
| :--- | :--- | :--- |
| **PyTorch Official Tutorial: Build the Neural Network** ([pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)) | Topics 1, 3, 4, 5, 6 (`nn.Module`, `nn.Flatten`, `nn.Sequential`, `nn.Linear`, `nn.ReLU`, logits vs. softmax) | The official PyTorch documentation sheet directly walked through on the instructor's screen, covering class inheritance, layer mechanics, and device allocation. |
| **PyTorch Official Tutorial: Optimizing Model Parameters** ([pytorch.org/tutorials/beginner/basics/optimization_tutorial.html](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)) | Topics 7, 8, 9 (Hyperparameters, `nn.CrossEntropyLoss`, `optim.SGD`, `train_loop`, `test_loop`, `zero_grad`) | The official optimization guide detailing the 4-step training cycle, loss calculation, backward gradient computation, and validation metrics. |
| **PyTorch Official Tutorial: Save and Load the Model** ([pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)) | Topic 10 (`state_dict`, `.pth` file persistence, `torch.save`, `torch.load`) | Comprehensive best-practice guide for serializing model weights via `state_dict` vs. full TorchScript checkpoints. |
| **Stanford CS231n: Convolutional Neural Networks for Visual Recognition — Neural Networks Architecture** ([cs231n.stanford.edu](https://cs231n.stanford.edu/)) by Prof. Fei-Fei Li & Andrej Karpathy | Topics 1, 3, 4, 5, 10 (MLP vs. CNN inductive biases, linear layer parameterization, activation functions) | Stanford's premier computer vision course notes detailing the geometric mechanics of linear layers, ReLU activations, and spatial feature extraction. |
| **Deep Learning (Chapter 6: Deep Feedforward Networks)** ([deeplearningbook.org](https://www.deeplearningbook.org/)) by Ian Goodfellow, Yoshua Bengio, & Aaron Courville | Topics 1, 4, 5, 7 (Universal Approximation Theorem, gradient-based learning, cross-entropy derivation) | The foundational textbook chapter establishing the mathematical theory behind multi-layer perceptrons, backpropagation, and categorical cross-entropy loss. |

---

## Sources

- **Lecture Video:** IIT Madras BS Degree Programme — *W1_T4: Tutorial 4: Introduction to pytorch: model building*
- **Video URL:** [https://www.youtube.com/watch?v=h1hEddM0aVE](https://www.youtube.com/watch?v=h1hEddM0aVE)
- **Course Page:** IIT Madras BS in Data Science and Applications — Generative AI (BSDA5002) by Prof. Prathosh A. P. (Tutorial by TA Chandan)
- **Duration:** 44 minutes 06 seconds (2646 seconds)
- **Package Status:** Validated current package (`youtube-lecture-tutor` v3).
