# Prerequisites & Foundational Warm-Up: PyTorch Basics

> **Target Audience:** Engineers, data scientists, and STEM students returning to deep learning and computational mathematics after 10–15 years.  
> **Course:** NPTEL / IISc — *Mathematical Foundations of Generative AI* (Tutorial 3).  
> **Previous Modules:** [Tutorial 2 — NumPy](../16-Tutorial02-Introduction-to-NumPy/NOTES.md) (Array Indexing, Vectorization & Linear Algebra).  
> **Next Step:** After completing this warm-up, open [NOTES.md](./NOTES.md) at the **Executive Summary**.

---

```
  ╔═══════════════════════════════════════════════════════════════════════════════════════╗
  ║                        CORE MENTAL MODELS YOU WILL MASTER TODAY                       ║
  ╠═══════════════════════════════════════════════════════════════════════════════════════╣
  ║ 1. "A Tensor is a NumPy array with two superpowers: GPU execution and Autograd."     ║
  ║ 2. "Hardware device placement is strict: operands in an operation must share device." ║
  ║ 3. "Shape operations re-stride memory; permute rearranges axes without moving data."  ║
  ║ 4. "Autograd builds a dynamic Directed Acyclic Graph (DAG) during the forward pass."  ║
  ║ 5. "Gradients accumulate into .grad; you must zero them before each backward pass."   ║
  ║ 6. "nn.Module automatically registers sub-modules and leaf parameters for optimizers."║
  ║ 7. "CrossEntropyLoss expects unnormalized raw logits, never softmax probabilities."   ║
  ║ 8. "Dataset defines item-level indexing; DataLoader orchestrates batching & shuffle."  ║
  ╚═══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 🗺️ Roadmap: Warm-Up Pillars to Lecture Topics

```
  ┌────────────────────────────────────────────────────────┐       ┌────────────────────────────────────────────────────────┐
  │ PREREQUISITE FOUNDATIONAL PILLAR                       │       │ LECTURE TOPIC MAPPING IN NOTES.md                      │
  ├────────────────────────────────────────────────────────┤       ├────────────────────────────────────────────────────────┤
  │ §1. Tensors vs NumPy Arrays: Metadata & Hardware Speed │ ────► │ Topic 1 (Imports & Device) & Topic 2 (Tensor Basics)   │
  │ §2. Compute Devices & Memory Hierarchy: CPU vs CUDA    │ ────► │ Topic 1 (Device Checking) & Topic 10 (Model to Device) │
  │ §3. Tensor Geometry: Reshape, Squeeze, Unsqueeze, Perm │ ────► │ Topic 3 (Reshape, Unsqueeze, Squeeze, Permute)        │
  │ §4. Computation Graphs & Reverse-Mode Autograd         │ ────► │ Topic 4 (Matmul & Autograd) & Topic 5 (Manual Linreg)  │
  │ §5. The 5-Step Deep Learning Optimization Loop         │ ────► │ Topic 5 (Manual GD) & Topic 6 (nn.Module & Optimizer)  │
  │ §6. Object-Oriented Architecture: nn.Module & Linear   │ ────► │ Topic 6 (nn.Module & nn.Linear) & Topic 10 (Full MLP)  │
  │ §7. Activations, Logits & Numerically Stable Losses    │ ────► │ Topic 7 (Activations, Logits & CrossEntropyLoss)       │
  │ §8. The Data Pipeline: Dataset Contract vs DataLoader  │ ────► │ Topic 8 (Custom Dataset) & Topic 9 (DataLoader Shuffle)│
  └────────────────────────────────────────────────────────┘       └────────────────────────────────────────────────────────┘
```

---

## 🪨 Math & System Terminology Rosetta Stone

This reference table bridges formal mathematical symbols, deep learning software abstractions, plain-English translations, and intuitive physical analogies.

| Symbol / Object | Formal Mathematical Concept | Plain-English Software Meaning | Everyday Physical Metaphor | Dedicated MathsTerm Guide |
| :--- | :--- | :--- | :--- | :--- |
| **`torch.Tensor`** | Multidimensional array $\mathbf{T} \in \mathbb{R}^{D_1 \times \dots \times D_k}$ | Typed n-dimensional array on CPU or GPU | A grid of spreadsheet cells stored on an ultra-fast hardware calculator. | [Tensors & Shapes](../../../MathsTerms/Tensors_and_Shapes.md) |
| **`device`** | Compute context $\mathcal{D} \in \{\text{CPU}, \text{CUDA}\}$ | Memory location where operations physically execute | The choice of kitchen: cooking on a home stove vs an industrial pizza furnace. | [Tensors & Shapes](../../../MathsTerms/Tensors_and_Shapes.md) |
| **$\mathbf{W}, \mathbf{b}$** | Weight matrix $\mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$ & Bias vector $\mathbb{R}^{D_{\text{out}}}$ | Learnable parameters of an affine layer | The slope and height adjustment knobs on an analog sound mixer. | [Tensors & Shapes](../../../MathsTerms/Tensors_and_Shapes.md) |
| **$\hat{\mathbf{y}} = \mathbf{x}\mathbf{W}^\top + \mathbf{b}$** | Affine linear mapping $\mathbb{R}^{D_{\text{in}}} \to \mathbb{R}^{D_{\text{out}}}$ | `nn.Linear` forward operation on mini-batches | A pasta extruder shaping incoming dough into structured pasta tubes. | [Tensors & Shapes](../../../MathsTerms/Tensors_and_Shapes.md) |
| **$\nabla_{\mathbf{w}} \mathcal{L}$** | Gradient vector $\left[ \frac{\partial \mathcal{L}}{\partial w_1}, \dots, \frac{\partial \mathcal{L}}{\partial w_p} \right]^\top$ | Direction and magnitude of steepest loss increase (`w.grad`) | The slope of a mountain beneath your hiking boots indicating steepest uphill. | [Derivatives, Gradients & Jacobians](../../../MathsTerms/Derivatives_Gradients_and_Jacobians.md) |
| **$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} \mathcal{L}$** | Gradient Descent step with learning rate $\eta > 0$ | Parameter update executed by `optimizer.step()` | Taking a measured step downhill into the fog to find the valley floor. | [Gradient Descent](../../../MathsTerms/Gradient_Descent.md) |
| **$\sigma(\mathbf{z})$ / $\text{ReLU}(z)$** | Nonlinear activation $\max(0, z)$ | Elementwise nonlinear transform breaking linear collapse | A one-way audio diode that allows positive signals and silences negative hum. | [Activation Functions](../../../MathsTerms/Activation_Functions.md) |
| **$\mathbf{z} \in \mathbb{R}^C$** | Raw score vector (Logits) | Unbounded model outputs before probability normalization | Raw judge scorecards before calculating percentage shares of the prize pot. | [Softmax](../../../MathsTerms/Softmax.md) |
| **$\text{Softmax}(\mathbf{z})_k$** | $\frac{e^{z_k}}{\sum_{j=1}^C e^{z_j}} \in (0, 1)$ | Probability distribution vector over $C$ mutually exclusive classes | Converting raw judge points into exact percentage slices of a pie. | [Softmax](../../../MathsTerms/Softmax.md) |
| **$\mathcal{L}_{\text{CE}}(\mathbf{z}, y)$** | Negative log-likelihood $-\log\left( \frac{e^{z_y}}{\sum e^{z_j}} \right)$ | Cross-entropy loss penalizing low confidence on the true target | A severe penalty fine proportional to how surprised the referee is by the outcome. | [Loss Functions](../../../MathsTerms/Loss_Functions.md) |

---

## Pillar 1: Tensors vs NumPy Arrays — Metadata & Hardware Speed

<a id="p1-tensor"></a>

### 1. 👶 ELI5 Quick Intuition
Think of a **NumPy array** as a ledger handwritten in a paper notebook on your desk (your computer's CPU). It is reliable and easy to read, but calculating complex formulas page by page takes time.
A **PyTorch Tensor** is that exact same ledger, but displayed on an ultra-responsive glass tablet connected to a supercomputer GPU. With a single tap, the tablet calculates millions of additions at the speed of light—and keeps a digital video recording of every math step so you can run the calculations in reverse (Autograd)!

```
  NumPy ndarray                      PyTorch Tensor
  ┌───────────────────────────┐      ┌────────────────────────────────────────────────────────┐
  │ • Fixed to CPU RAM        │      │ • Executes on CPU or NVIDIA CUDA GPU                   │
  │ • No gradient tracking    │ ───► │ • Automatic Differentiation Graph Engine (Autograd)    │
  │ • Traditional Data Science│      │ • Native Backbone for Deep Learning & Generative AI   │
  └───────────────────────────┘      └────────────────────────────────────────────────────────┘
```

---

### 2. 🔍 Plain-English Breakdown
- **Core Abstraction:** A `torch.Tensor` represents a multidimensional grid of numbers of a uniform data type (`dtype`).
- **Crucial Metadata:** Every tensor carries three core identity attributes:
  1. **`.shape` (`torch.Size`):** The dimensional layout (e.g., `torch.Size([32, 3, 224, 224])` for 32 color images).
  2. **`.dtype` (`torch.dtype`):** The numerical precision. Deep learning models standardly use single-precision floating point (`torch.float32`), while target classification indices use 64-bit integers (`torch.int64` / `torch.long`).
  3. **`.device` (`torch.device`):** The hardware physical address where the data buffer resides (e.g., `device(type='cpu')` or `device(type='cuda', index=0)`).
- **Extraction:** When a tensor contains a single scalar value (such as a calculated loss), calling `.item()` extracts the value into a native Python `float`.

---

### 3. 📐 Formal Mathematics & Tensor Notation
Mathematically, a $k$-th order tensor $\mathbf{T}$ with shape $(D_1, D_2, \dots, D_k)$ is a multilinear mapping defined over the Cartesian product of discrete index sets:
$$\mathbf{T} \in \mathbb{R}^{D_1 \times D_2 \times \dots \times D_k} \implies \mathbf{T}(i_1, i_2, \dots, i_k) \in \mathbb{R}, \quad \text{where } 0 \le i_j < D_j$$

Memory layout is governed by a **stride vector** $\mathbf{s} = (s_1, s_2, \dots, s_k)$, where the 1D flat memory offset for index tuple $(i_1, \dots, i_k)$ is:
$$\text{MemoryOffset}(i_1, \dots, i_k) = \sum_{j=1}^k i_j \cdot s_j$$

---

### 4. 🔢 Concrete Numerical Micro-Example
Let us construct a 2D matrix $\mathbf{A} \in \mathbb{R}^{2 \times 3}$:
$$\mathbf{A} = \begin{bmatrix} 1.5 & 2.5 & 3.5 \\ 4.5 & 5.5 & 6.5 \end{bmatrix}$$
- Total element count: $N = 2 \times 3 = 6$.
- Data type: `torch.float32` (each element consumes 4 bytes; total tensor memory = $6 \times 4 = 24$ bytes).
- Stride vector for row-major order: $\mathbf{s} = (3, 1)$ (stepping down 1 row jumps 3 floats in flat RAM; stepping right 1 column jumps 1 float).

---

### 5. 💻 Standalone Runnable Python / PyTorch Snippet

```python
import torch
import numpy as np

# 1. Create tensor from Python list
x_list = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=torch.float32)
print("Tensor from list:\n", x_list)
print("Shape:", x_list.shape, "| Dtype:", x_list.dtype, "| Device:", x_list.device)
print("Strides:", x_list.stride())

# 2. Interoperability with NumPy (Zero-copy sharing on CPU)
np_arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
x_from_np = torch.from_numpy(np_arr)
print("\nTensor from NumPy:", x_from_np)

# 3. Pulling a scalar loss value
scalar_tensor = torch.tensor(42.50)
python_scalar = scalar_tensor.item()
print("Scalar via .item():", python_scalar, "Type:", type(python_scalar))
```

---

### 🧠 Diagnostic Mini-Checks (Pillar 1)
1. **Question:** What are the three primary metadata attributes that define a `torch.Tensor`?  
   *Answer:* `.shape` (dimensional sizes), `.dtype` (numerical precision type), and `.device` (hardware location: CPU or CUDA).
2. **Question:** Why should you use floating-point literals (e.g. `1.0`) instead of integer literals (`1`) when initializing weights or training tensors?  
   *Answer:* Neural network parameters require floating-point data types (`float32`) to compute continuous fractional gradients during backpropagation.
3. **Question:** What does `.item()` do, and what happens if you call it on a tensor with shape `(2, 2)`?  
   *Answer:* `.item()` extracts a single scalar into a native Python number. Calling it on a multi-element tensor raises a `RuntimeError: a Tensor with 4 elements cannot be converted to Scalar`.

---

## Pillar 2: Compute Devices & Hardware Placement — CPU vs CUDA

<a id="p2-device"></a>

### 1. 👶 ELI5 Quick Intuition
Imagine preparing dinner in a restaurant:
- The **CPU** is the chef's wooden prep station: great for chopping vegetables and organizing ingredients.
- The **GPU** is a massive 1,000-burner industrial stove: it can simmer thousands of sauce pans simultaneously at blinding speed.
- **The Golden Rule of Kitchen Safety:** You cannot stir a pan that is sitting on the industrial stove using a spoon that is locked inside a drawer in the wooden prep station! Every ingredient and utensil in an operation must physically sit in the **exact same kitchen**.

```
  HOST SYSTEM (CPU)                               ACCELERATOR (NVIDIA GPU)
  ┌─────────────────────────────────┐             ┌─────────────────────────────────┐
  │ Host RAM (e.g., 32 GB DDR5)     │   PCIe Bus  │ High-Bandwidth VRAM (e.g. 16GB) │
  │ • DataLoaders load disk images  │ ──────────► │ • Model Weights & Biases        │
  │ • Batch preprocessing runs here │   (Slow!)   │ • Massive Parallel Matrix Mult  │
  └─────────────────────────────────┘             └─────────────────────────────────┘
```

---

### 2. 🔍 Plain-English Breakdown
- **Hardware Separation:** Modern deep learning workstations maintain two separate physical memory spaces: Host System RAM (managed by CPU) and Video RAM / VRAM (managed by the GPU accelerator).
- **The Device Mismatch Trap:** Attempting an operation (such as addition or matrix multiplication) between a CPU tensor and a CUDA tensor causes an immediate hardware execution halt: `RuntimeError: Expected all tensors to be on the same device`.
- **The Canonical Idiom:** Professional PyTorch code dynamically detects hardware availability at runtime and stores a single device handle reused throughout the entire codebase:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```

---

### 3. 📐 Formal Mathematics & Hardware Transfer Model
Moving data across the system bus incurs a latency penalty modeled by bandwidth $\mathcal{B}_{\text{PCIe}}$ and transfer initialization latency $\tau_{\text{init}}$:
$$\text{Time}_{\text{transfer}}(\mathbf{T}) = \tau_{\text{init}} + \frac{N \times \text{sizeof}(\text{dtype})}{\mathcal{B}_{\text{PCIe}}}$$
For maximum throughput, deep learning pipelines minimize host-to-device transfers by loading data asynchronously in large batches rather than transferring individual scalars.

---

### 4. 🔢 Concrete Numerical Micro-Example
Suppose you move an image tensor $\mathbf{X} \in \mathbb{R}^{64 \times 3 \times 224 \times 224}$ of `float32` elements from CPU to GPU:
- Total elements $N = 64 \times 3 \times 224 \times 224 = 9,633,792$ floats.
- Memory size $= 9,633,792 \times 4 \text{ bytes} \approx 38.535 \text{ MB}$.
- Over a PCIe 4.0 x16 link ($\approx 31.5 \text{ GB/s}$ real-world bandwidth), transfer time is $\approx 1.22 \text{ ms}$.

---

### 5. 💻 Standalone Runnable Python / PyTorch Snippet

```python
import torch

# 1. Dynamic device query
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Target compute device:", device)

if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Capability:", torch.cuda.get_device_capability(0))

# 2. Moving tensors between devices
x_cpu = torch.randn(4, 4)
print("Initial device:", x_cpu.device)

x_target = x_cpu.to(device)
print("Target device:", x_target.device)

# 3. Safe operations verification
y_target = torch.ones(4, 4, device=device)
z_target = x_target @ y_target
print("Result device:", z_target.device)
```

---

### 🧠 Diagnostic Mini-Checks (Pillar 2)
1. **Question:** What happens if you execute `a + b` when tensor `a` is on `cpu` and tensor `b` is on `cuda:0`?  
   *Answer:* PyTorch raises a `RuntimeError` due to a device mismatch; operands must reside on the identical device.
2. **Question:** If a neural network model is moved to GPU via `model.to(device)`, does a DataLoader automatically yield batches on the GPU?  
   *Answer:* No. DataLoaders produce CPU tensors by default. Each batch must be explicitly transferred inside the training loop via `images = images.to(device)`.
3. **Question:** Why should you avoid hard-coding `device = "cuda"` in your scripts?  
   *Answer:* Hard-coding `"cuda"` crashes immediately on machines without NVIDIA GPUs (such as standard CI/CD servers or local CPU laptops).

---

## Pillar 3: Tensor Geometry — Reshape, Squeeze, Unsqueeze & Permute

<a id="p3-shapeops"></a>

### 1. 👶 ELI5 Quick Intuition
Imagine you have 12 chocolate bars:
- **Reshape:** You can arrange them in a $3 \times 4$ flat box or a $2 \times 6$ long tray. You still have 12 chocolates in the exact same flavor order.
- **Unsqueeze:** You put your $3 \times 4$ box inside a shipping crate with 1 compartment (`shape: 1 x 3 x 4`). It adds a container layer (a batch dimension) without altering the chocolates.
- **Squeeze:** You remove the empty 1-compartment crate layer.
- **Permute:** You take a photo printed on a postcard ($224 \times 224 \times 3$) and swap the physical axes so the color channels sit on top ($3 \times 224 \times 224$), which is the layout convolutional neural networks require!

```
  RESHAPE (12 elements)                UNSQUEEZE(0)                     PERMUTE(2, 0, 1)
  ┌───┬───┬───┬───┐                    ┌──────────────────┐             Height x Width x Color (HWC)
  │ 0 │ 1 │ 2 │ 3 │                    │ ┌───┬───┬───┐    │                      │
  ├───┼───┼───┼───┤   ────────►        │ │ 0 │ 1 │ 2 │    │   ────────►          ▼
  │ 4 │ 5 │ 6 │ 7 │   shape: (1, 3, 4) │ └───┴───┴───┘    │             Color x Height x Width (CHW)
  └───┴───┴───┴───┘                    └──────────────────┘             (Channels-First Vision Standard)
```

---

### 2. 🔍 Plain-English Breakdown
- **`reshape(*shape)` vs `view(*shape)`:** Both alter the dimensional view of a tensor without changing underlying data. `view()` strictly requires that the memory buffer is contiguous in RAM, while `reshape()` automatically copies data if memory is fragmented. **Best Practice:** Default to `.reshape()`.
- **`unsqueeze(dim)`:** Inserts a new dimension of size 1 at the specified index `dim`. Vital when feeding a single image `(3, 224, 224)` into a model expecting a batch `(1, 3, 224, 224)`.
- **`squeeze(dim=None)`:** Collapses all (or a specified) size-1 dimensions.
- **`permute(*dims)`:** Reorders the coordinate axes of the tensor. Unlike `reshape()`, which flattens and re-slices memory in row-major order, `permute()` swaps the semantic meaning of the coordinate axes.

---

### 3. 📐 Formal Mathematics & Permutation Tensor Algebra
Let $\mathbf{T}$ be a 3rd-order tensor with shape $(D_0, D_1, D_2)$. The permutation operator $\mathcal{P}_{\pi}$ defined by permutation index $\pi = (\pi_0, \pi_1, \pi_2)$ produces tensor $\mathbf{U} = \mathcal{P}_{\pi}(\mathbf{T})$ where:
$$\mathbf{U}_{i_{\pi_0}, i_{\pi_1}, i_{\pi_2}} = \mathbf{T}_{i_0, i_1, i_2}$$
For an image tensor converting from channels-last $(H, W, C)$ to channels-first $(C, H, W)$, $\pi = (2, 0, 1)$.

---

### 4. 🔢 Concrete Numerical Micro-Example
Let vector $\mathbf{v} = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]$ (length 12):
1. `v.reshape(3, 4)` creates a $3 \times 4$ matrix. Total elements $= 3 \times 4 = 12 \checkmark$.
2. `v.reshape(3, 5)` fails with `RuntimeError: shape '[3, 5]' is invalid for input of size 12` because $3 \times 5 = 15 \ne 12$.
3. `v.reshape(3, 4).unsqueeze(0)` creates shape `(1, 3, 4)`.

---

### 5. 💻 Standalone Runnable Python / PyTorch Snippet

```python
import torch

# 1. Reshaping
x = torch.arange(12)
grid = x.reshape(3, 4)
print("Reshaped Grid (3x4):\n", grid)

# 2. Unsqueeze and Squeeze
batched_grid = grid.unsqueeze(0)  # Shape: (1, 3, 4)
print("Batched Shape:", batched_grid.shape)
squeezed_grid = batched_grid.squeeze(0)  # Shape: (3, 4)
print("Squeezed Shape:", squeezed_grid.shape)

# 3. Permute for Vision Layout (HWC -> CHW)
img_hwc = torch.randn(224, 224, 3)  # Height=224, Width=224, Channels=3
img_chw = img_hwc.permute(2, 0, 1)  # Axis 2 -> 0, Axis 0 -> 1, Axis 1 -> 2
print("Original Vision Shape:", img_hwc.shape)
print("Permuted Vision Shape:", img_chw.shape)
```

---

### 🧠 Diagnostic Mini-Checks (Pillar 3)
1. **Question:** What is the shape of tensor `t` of shape `(3, 4)` after calling `t.unsqueeze(1)`?  
   *Answer:* `torch.Size([3, 1, 4])`.
2. **Question:** Why does PyTorch vision models prefer `(C, H, W)` layout over standard OpenCV `(H, W, C)`?  
   *Answer:* PyTorch's optimized CUDA convolution kernels (cuDNN) expect channels-first memory layout for efficient parallel caching.
3. **Question:** If you have an image tensor with shape `(28, 28, 1)`, does `reshape(1, 28, 28)` do the exact same thing as `permute(2, 0, 1)`?  
   *Answer:* No! `reshape` blindly reads memory sequentially and distorts spatial geometry; `permute` swaps the semantic coordinate axes correctly.

---

## Pillar 4: Computation Graphs & Reverse-Mode Automatic Differentiation (Autograd)

<a id="p4-autograd"></a>

### 1. 👶 ELI5 Quick Intuition
Think of **Autograd** as an automatic cash register receipt:
- Every time you add, multiply, or transform a tensor marked with `requires_grad=True`, PyTorch prints a line item on a **digital tape receipt** (the Computation Graph).
- When you reach the total bill (the scalar Loss $\mathcal{L}$), you call **`.backward()`**.
- PyTorch reads the receipt tape **backwards from bottom to top**, calculates exactly how much each ingredient contributed to the final total using the calculus Chain Rule, and writes the answer directly into **`.grad`**!

```
  FORWARD PASS (Building Graph)               BACKWARD PASS (Chain Rule)
  x (Leaf: requires_grad=True)                x.grad = dL/dx = 2*x = 6.0
       │                                           ▲
       ▼ [pow(2)]                                  │ [Reverse Derivative]
  y = x²                                      dy/dx = 2*x
       │                                           ▲
       ▼ [mean()]                                  │ [Reverse Derivative]
  Loss L = y ───────────────────────────────► loss.backward()
```

---

### 2. 🔍 Plain-English Breakdown
- **Leaf Nodes:** Tensors created directly by the user with `requires_grad=True` (such as model weights $\mathbf{W}$ and biases $\mathbf{b}$). They receive and store gradients in their `.grad` field.
- **Intermediate Nodes:** Non-leaf tensors created by mathematical operations (e.g. $\mathbf{z} = \mathbf{x}\mathbf{W} + \mathbf{b}$). They do not retain gradients by default to conserve memory.
- **The Scalar Requirement:** `loss.backward()` requires a single scalar value ($0$-dimensional tensor) to start the reverse-mode accumulation. If you have a vector loss, you must reduce it using `.mean()` or `.sum()`.
- **Gradient Accumulation:** Calling `.backward()` does **not** overwrite `.grad`; it **adds** to existing values ($\mathbf{g} \leftarrow \mathbf{g} + \nabla \mathcal{L}$). You must explicitly zero gradients before each training iteration!

---

### 3. 📐 Formal Mathematics & Vector-Jacobian Products (VJPs)
For a composition of vector functions $\mathbf{y} = f(\mathbf{x})$ and scalar loss $\mathcal{L} = g(\mathbf{y})$, the multivariable chain rule gives:
$$\nabla_{\mathbf{x}} \mathcal{L} = \mathbf{J}_f(\mathbf{x})^\top \nabla_{\mathbf{y}} \mathcal{L}$$
where $\mathbf{J}_f(\mathbf{x})$ is the Jacobian matrix $\left[ \frac{\partial y_i}{\partial x_j} \right]$. PyTorch implements **Reverse-Mode Automatic Differentiation (Vector-Jacobian Product / VJP)**, which evaluates the vector transpose product from right to left in $\mathcal{O}(1)$ sweep time relative to the forward pass.

---

### 4. 🔢 Concrete Numerical Micro-Example
Let scalar $x = 3.0$ with `requires_grad=True`, and polynomial $y = x^2 + 3x + 1$:
1. Forward evaluation:
   $$y = (3.0)^2 + 3(3.0) + 1 = 9 + 9 + 1 = 19.0$$
2. Analytical derivative:
   $$\frac{dy}{dx} = \frac{d}{dx}(x^2 + 3x + 1) = 2x + 3$$
3. At $x = 3.0$:
   $$\left.\frac{dy}{dx}\right|_{x=3.0} = 2(3.0) + 3 = 9.0$$
4. Calling `y.backward()` writes `x.grad = 9.0`.

---

### 5. 💻 Standalone Runnable Python / PyTorch Snippet

```python
import torch

# 1. Scalar Autograd Verification
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 3*x + 1
y.backward()
print("x:", x.item(), "| dy/dx computed by Autograd:", x.grad.item())
assert x.grad.item() == 9.0

# 2. Vector Autograd with Reduction
w = torch.tensor([2.0, -1.0, 4.0], requires_grad=True)
x_in = torch.tensor([1.0, 3.0, 2.0])
# Loss = sum( (w * x)^2 )
loss = torch.sum((w * x_in) ** 2)
loss.backward()
# dLoss/dw_i = 2 * (w_i * x_i) * x_i = 2 * w_i * x_i^2
# For w0: 2 * 2.0 * (1.0)^2 = 4.0
# For w1: 2 * (-1.0) * (3.0)^2 = -18.0
# For w2: 2 * 4.0 * (2.0)^2 = 32.0
print("Vector w.grad:", w.grad)
```

---

### 🧠 Diagnostic Mini-Checks (Pillar 4)
1. **Question:** What happens if you call `y.backward()` a second time without clearing gradients or setting `retain_graph=True`?  
   *Answer:* PyTorch throws an error because the intermediate computation graph was already freed after the first backward pass.
2. **Question:** Why does PyTorch accumulate gradients rather than overwriting `.grad` on every `.backward()` call?  
   *Answer:* Gradient accumulation enables training large effective batch sizes across multiple forward-backward micro-steps on memory-constrained GPUs.
3. **Question:** When updating parameter tensors manually in Python (e.g. `w -= lr * w.grad`), why is `with torch.no_grad():` mandatory?  
   *Answer:* Without `torch.no_grad()`, the update operation itself would be recorded onto the computation graph, causing memory leaks and corrupted derivative graphs.

---

## Pillar 5: The 5-Step Deep Learning Optimization Loop

<a id="p5-trainstep"></a>

### 1. 👶 ELI5 Quick Intuition
Training a neural network is like learning to steer a racing car around a track:
1. **`model.train()`:** Put the car into race mode (enable training systems like Dropout).
2. **`optimizer.zero_grad()`:** Clear your memory of yesterday's steering adjustments.
3. **`pred = model(x)`:** Look at the road and steer the wheel forward (Forward Pass).
4. **`loss.backward()`:** Feel how far off course you drifted and compute how to correct (Backward Pass).
5. **`optimizer.step()`:** Physically nudge the steering wheel in the corrective direction (Parameter Update).

```
  ╔═══════════════════════════════════════════════════════════════════════════════════════╗
  ║                        THE CANONICAL 5-STEP PYTORCH TRAINING LOOP                     ║
  ╠═══════════════════════════════════════════════════════════════════════════════════════╣
  ║  Step 0: model.train()          ──► Enable training mode behaviors                     ║
  ║  Step 1: optimizer.zero_grad()  ──► Wipe accumulated gradients from previous batch    ║
  ║  Step 2: y_pred = model(x)      ──► Forward pass: compute predictions via DAG          ║
  ║  Step 3: loss.backward()        ──► Backward pass: reverse-mode autodiff fills .grad   ║
  ║  Step 4: optimizer.step()       ──► Apply optimizer update rule to all model params    ║
  ╚═══════════════════════════════════════════════════════════════════════════════════════╝
```

---

### 2. 🔍 Plain-English Breakdown
- **The Execution Order Invariant:** `zero_grad()` must occur before `step()`, and `backward()` must precede `step()`.
- **Manual Gradient Descent:** In early tutorials, you update weights manually:
  ```python
  with torch.no_grad():
      w -= lr * w.grad
      b -= lr * b.grad
      w.grad.zero_()
      b.grad.zero_()
  ```
- **Automated Optimizers:** `torch.optim.SGD` or `torch.optim.Adam` encapsulates this manual math across thousands of parameters in one clean `optimizer.step()` call.

---

### 3. 📐 Formal Mathematics & Stochastic Gradient Descent (SGD)
Given dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ and model parameterized by $\boldsymbol{\theta} \in \mathbb{R}^P$, the empirical risk minimization objective is:
$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{B} \sum_{i=1}^B \ell(f(\mathbf{x}_i; \boldsymbol{\theta}), y_i)$$
The standard SGD parameter update at iteration $t$ with learning rate $\eta > 0$ is:
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$$

---

### 4. 🔢 Concrete Numerical Micro-Example
Suppose parameter $w = 2.0$, target $y = 5.0$, input $x = 2.0$, prediction $\hat{y} = w \cdot x = 4.0$, loss $\mathcal{L} = (\hat{y} - y)^2 = (4.0 - 5.0)^2 = 1.0$, learning rate $\eta = 0.1$:
1. Gradient computation:
   $$\frac{\partial \mathcal{L}}{\partial w} = 2(\hat{y} - y) \cdot x = 2(4.0 - 5.0) \cdot 2.0 = -4.0$$
2. Parameter update:
   $$w_{\text{new}} = w - \eta \frac{\partial \mathcal{L}}{\partial w} = 2.0 - 0.1(-4.0) = 2.0 + 0.4 = 2.4$$
3. New prediction: $\hat{y}_{\text{new}} = 2.4 \times 2.0 = 4.8$ (closer to target 5.0!).

---

### 5. 💻 Standalone Runnable Python / PyTorch Snippet

```python
import torch

# Synthetic dataset: y = 3*x + 2
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[5.0], [8.0], [11.0], [14.0]])

# Model weights
w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, 1, requires_grad=True)
lr = 0.05

for epoch in range(100):
    # Forward Pass
    y_pred = X @ w + b
    loss = torch.mean((y_pred - Y)**2)
    
    # Backward Pass
    loss.backward()
    
    # Parameter Update under no_grad
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        
        # Zero Gradients
        w.grad.zero_()
        b.grad.zero_()

print(f"Learned Weight: {w.item():.3f} (True: 3.000) | Bias: {b.item():.3f} (True: 2.000)")
```

---

### 🧠 Diagnostic Mini-Checks (Pillar 5)
1. **Question:** What happens if you run `optimizer.step()` before calling `loss.backward()`?  
   *Answer:* The optimizer updates weights using outdated or zero gradients because `.grad` has not been populated for the current batch.
2. **Question:** What happens if you forget `optimizer.zero_grad()` in a training loop?  
   *Answer:* Gradients continuously accumulate across epochs, causing explosive gradient norms and catastrophic training divergence.
3. **Question:** Why is `model.eval()` called before evaluating validation accuracy?  
   *Answer:* `model.eval()` disables training-specific behaviors like Dropout and switches BatchNorm to use running population statistics.

---

## Pillar 6: Object-Oriented Neural Architecture — `nn.Module` & `nn.Linear`

<a id="p6-module"></a>

### 1. 👶 ELI5 Quick Intuition
Building neural networks without `nn.Module` is like building an engine out of loose screws, pipes, and spark plugs rolling around on the floor.
`nn.Module` is the **chassis**:
- In **`__init__()`**, you bolt on standardized components (like `nn.Linear` gearboxes and `nn.ReLU` filters).
- In **`forward()`**, you connect the fuel line, showing how data flows through the parts.
- When you tell the chassis to move to the GPU (`model.to(device)`), every single component bolted inside moves automatically!

```
  nn.Module Subclass Blueprint
  ┌────────────────────────────────────────────────────────┐
  │ class MultiLayerPerceptron(nn.Module):                 │
  │   def __init__(self):                                  │
  │     super().__init__()                                 │
  │     self.fc1 = nn.Linear(in_features=2, out_features=4)│
  │     self.fc2 = nn.Linear(in_features=4, out_features=1)│
  │                                                        │
  │   def forward(self, x):                                │
  │     x = F.relu(self.fc1(x))                            │
  │     return self.fc2(x)                                 │
  └────────────────────────────────────────────────────────┘
```

---

### 2. 🔍 Plain-English Breakdown
- **`super().__init__()` is Mandatory:** Calling the superclass constructor initializes internal registration dictionaries (`_parameters`, `_modules`, `_buffers`).
- **`nn.Linear(in_features, out_features, bias=True)`:** Implements the classic fully-connected layer:
  $$\mathbf{Y} = \mathbf{X}\mathbf{W}^\top + \mathbf{b}$$
  where weights have shape `(out_features, in_features)` and bias has shape `(out_features)`.
- **Batch Size Agnosticism:** You only declare feature dimensions in `nn.Linear`; the layer automatically handles any batch size $B$ across input tensors with shape `(B, in_features)`.

---

### 3. 📐 Formal Mathematics & Layer Composition
An $L$-layer Multi-Layer Perceptron (MLP) mapping input $\mathbf{x} \in \mathbb{R}^{D_0}$ to output $\hat{\mathbf{y}} \in \mathbb{R}^{D_L}$ is formalized as:
$$\mathbf{h}^{(l)} = \sigma\left( \mathbf{h}^{(l-1)} \mathbf{W}^{(l)\top} + \mathbf{b}^{(l)} \right), \quad l = 1, \dots, L-1$$
$$\hat{\mathbf{y}} = \mathbf{h}^{(L-1)} \mathbf{W}^{(L)\top} + \mathbf{b}^{(L)}$$
where $\mathbf{h}^{(0)} = \mathbf{x}$ and $\sigma(\cdot)$ is an activation function.

---

### 4. 🔢 Concrete Numerical Micro-Example
For `nn.Linear(in_features=3, out_features=2)`:
- Weight matrix $\mathbf{W}$ shape: `(2, 3)` (6 parameters).
- Bias vector $\mathbf{b}$ shape: `(2)` (2 parameters).
- Total learnable parameters $= 6 + 2 = 8$.
- Given batch input $\mathbf{X} \in \mathbb{R}^{5 \times 3}$, output tensor shape is `(5, 2)`.

---

### 5. 💻 Standalone Runnable Python / PyTorch Snippet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return out

model = SimpleMLP(in_features=4, hidden_dim=8, out_features=2)
print("Model Architecture:\n", model)

# Parameter inspection
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total Learnable Parameters:", total_params)

# Forward execution with batch of 10 samples
x_sample = torch.randn(10, 4)
y_out = model(x_sample)  # Calls __call__ -> forward
print("Output Shape:", y_out.shape)
```

---

### 🧠 Diagnostic Mini-Checks (Pillar 6)
1. **Question:** What occurs if you omit `super().__init__()` inside a custom `nn.Module` constructor?  
   *Answer:* Instantiation fails with `AttributeError: cannot assign module before Module.__init__() call`.
2. **Question:** Why should you always call `model(x)` rather than `model.forward(x)`?  
   *Answer:* Calling `model(x)` triggers `__call__`, which executes PyTorch forward hooks, profiling utilities, and input validation before routing to `forward()`.
3. **Question:** How many total learnable parameters exist in an `nn.Linear(10, 20)` layer with `bias=True`?  
   *Answer:* $(10 \times 20) + 20 = 220$ parameters.

---

## Pillar 7: Nonlinearity & Classification Losses — Activations, Logits & Cross-Entropy

<a id="p7-loss"></a>

### 1. 👶 ELI5 Quick Intuition
Think of a multi-class game show with 3 doors (Cat, Dog, Bird):
- **Logits:** The 3 raw scores shouted by the judges: `[2.0, 5.0, 1.0]`. They can be negative, zero, or huge numbers.
- **Softmax:** The accountant takes those raw scores and turns them into clean betting odds that add up to 100%: `[4.7%, 94.7%, 0.6%]`.
- **The Golden Rule of CrossEntropyLoss:** PyTorch's `nn.CrossEntropyLoss` is a strict judge who wants the **RAW judge scores (Logits)** directly. If you feed it the already-converted percentages, the judge calculates percentages on your percentages (Double-Softmax), corrupting all gradients!

```
  Raw Network Output (Logits)
  z = [2.0, 5.0, 1.0]  ─── (Unbounded Real Numbers)
          │
          ├──────────────────────────────────────────┐
          │                                          │
          ▼ [F.softmax(z, dim=1)]                    ▼ [Direct Input to nn.CrossEntropyLoss]
  Probabilities: [0.047, 0.947, 0.006]       nn.CrossEntropyLoss(z, target=1)
  (Use ONLY for inference / display)          (Applies LogSoftmax internally for stability!)
```

---

### 2. 🔍 Plain-English Breakdown
- **Nonlinear Activations:** Without nonlinear activations like ReLU ($\max(0, x)$), stacking 100 Linear layers collapses mathematically into a single flat affine transformation $\mathbf{W}_{\text{total}}\mathbf{x} + \mathbf{b}_{\text{total}}$.
- **Logits:** The unnormalized output scores produced by the final linear layer of a classification network.
- **Target Formatting:** For multi-class classification with $C$ classes, target labels must be 1D tensors of integer class indices $y \in \{0, 1, \dots, C-1\}$ with dtype `torch.long`, **NOT** one-hot encoded vectors.

---

### 3. 📐 Formal Mathematics & Numerical Stability of Log-Softmax
The Softmax function converts logits $\mathbf{z} \in \mathbb{R}^C$ into probability distribution $\mathbf{p}$:
$$p_k = \frac{e^{z_k}}{\sum_{j=1}^C e^{z_j}}$$
The Multi-Class Cross-Entropy Loss for true class index $y^*$ is:
$$\mathcal{L}_{\text{CE}}(\mathbf{z}, y^*) = -\log p_{y^*} = -\log\left( \frac{e^{z_{y^*}}}{\sum_{j=1}^C e^{z_j}} \right) = -z_{y^*} + \log\left(\sum_{j=1}^C e^{z_j}\right)$$
Evaluating $e^{z_j}$ directly causes numerical overflow if $z_j > 88$ (for float32). PyTorch's `nn.CrossEntropyLoss` combines $\text{LogSoftmax}$ and $\text{NLLLoss}$ using the **Log-Sum-Exp trick**:
$$\log\left(\sum_{j=1}^C e^{z_j}\right) = M + \log\left(\sum_{j=1}^C e^{z_j - M}\right), \quad \text{where } M = \max_j(z_j)$$

---

### 4. 🔢 Concrete Numerical Micro-Example
Let logits $\mathbf{z} = [1.0, 3.0, 0.0]$ and true target $y^* = 1$ (Class 1):
1. Compute exponentials: $e^1 \approx 2.718, e^3 \approx 20.086, e^0 = 1.0$.
2. Denominator sum $= 2.718 + 20.086 + 1.0 = 23.804$.
3. True class probability: $p_1 = \frac{20.086}{23.804} \approx 0.8438$.
4. Cross-Entropy Loss: $\mathcal{L} = -\log(0.8438) \approx 0.170$.

---

### 5. 💻 Standalone Runnable Python / PyTorch Snippet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Raw Logits for batch of 2 samples across 3 classes
logits = torch.tensor([[1.0, 3.0, 0.0],
                       [2.5, 0.5, 0.1]], dtype=torch.float32)

# True integer target labels
targets = torch.tensor([1, 0], dtype=torch.long)

# 2. Correct Loss Computation
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
print(f"Correct CrossEntropyLoss: {loss.item():.4f}")

# 3. Model Inference (Probabilities & Predictions)
with torch.no_grad():
    probs = F.softmax(logits, dim=1)
    predictions = torch.argmax(logits, dim=1)
    
print("Softmax Probabilities:\n", probs)
print("Predicted Classes:", predictions.tolist(), "| Ground Truth:", targets.tolist())
```

---

### 🧠 Diagnostic Mini-Checks (Pillar 7)
1. **Question:** What catastrophic bug occurs if you apply `F.softmax` to your model output before passing it into `nn.CrossEntropyLoss`?  
   *Answer:* The "Double-Softmax" bug: `CrossEntropyLoss` applies log-softmax to probabilities, resulting in distorted gradients and near-zero learning rates.
2. **Question:** What data type must classification target labels have when passed to `nn.CrossEntropyLoss`?  
   *Answer:* 64-bit integer (`torch.long` / `torch.int64`).
3. **Question:** Along which dimension should you compute `argmax` for a logits tensor of shape `(batch_size, num_classes)`?  
   *Answer:* `dim=1` (across the class columns for each sample row).

---

## Pillar 8: The Data Production Line — `Dataset` Contract vs `DataLoader` Batch Pipeline

<a id="p8-data"></a>

### 1. 👶 ELI5 Quick Intuition
Think of feeding hungry customers at a busy cafeteria:
- **`Dataset`:** The master cookbook in the back pantry. If you ask for recipe #42, it reaches onto the shelf and hands you one fresh sandwich and its label (`__getitem__(42)`).
- **`DataLoader`:** The conveyor belt cart in the dining hall. It loads **16 sandwiches at a time** (Batch Size), shuffles the tray order every morning (`shuffle=True`), and uses multiple kitchen assistants (`num_workers`) so the chefs never wait for ingredients!

```
  CUSTOM DATASET CONTRACT                             DATALOADER PIPELINE
  ┌─────────────────────────────────┐                 ┌──────────────────────────────────────┐
  │ class CustomDataset(Dataset):   │                 │ loader = DataLoader(dataset,         │
  │   def __len__(self):            │ ──────────────► │                     batch_size=16,   │
  │     return total_samples        │                 │                     shuffle=True)    │
  │   def __getitem__(self, idx):   │                 │ Yields: (batch_x, batch_y) mini-batch│
  │     return x[idx], y[idx]       │                 └──────────────────────────────────────┘
  └─────────────────────────────────┘
```

---

### 2. 🔍 Plain-English Breakdown
- **The Dataset Protocol:** Any custom dataset in PyTorch must subclass `torch.utils.data.Dataset` and implement two essential magic methods:
  1. **`__len__(self) -> int`:** Returns the total dataset count.
  2. **`__getitem__(self, idx: int) -> Tuple[Tensor, Tensor]`:** Returns the input sample and label for integer index `idx`.
- **The DataLoader Engine:** Wraps any `Dataset` instance to provide automated mini-batch collation, stochastic shuffling across epochs, memory pinning (`pin_memory=True`), and multi-process asynchronous loading (`num_workers`).
- **The Incomplete Last Batch:** When dataset size $N$ is not evenly divisible by `batch_size`, the final batch will have size $N \pmod{\text{batch\_size}}$ unless `drop_last=True` is specified.

---

### 3. 📐 Formal Mathematics & Stochastic Mini-Batch Approximations
Full-batch gradient descent evaluates the exact gradient over all $N$ training examples:
$$\nabla \mathcal{L}_{\text{full}}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N \nabla \ell_i(\boldsymbol{\theta})$$
The `DataLoader` creates a stochastic partition of the index set $\{1, \dots, N\} = \bigcup_{k=1}^K \mathcal{B}_k$ where $|\mathcal{B}_k| = B$. The mini-batch gradient is an **unbiased estimator** of the full gradient:
$$\mathbb{E}_{\mathcal{B}}\left[ \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla \ell_i(\boldsymbol{\theta}) \right] = \nabla \mathcal{L}_{\text{full}}(\boldsymbol{\theta})$$
Shuffling ensures independent, identically distributed (I.I.D.) mini-batch samples across training epochs.

---

### 4. 🔢 Concrete Numerical Micro-Example
Suppose you have a dataset of $N = 100$ samples with `batch_size = 16`:
- Total batches per epoch: $\lceil 100 / 16 \rceil = 7$ batches.
- Batches 1 through 6 have shape `(16, D)`.
- Batch 7 has shape `(4, D)` because $100 - (6 \times 16) = 4$.

---

### 5. 💻 Standalone Runnable Python / PyTorch Snippet

```python
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Custom Dataset Implementation
class SyntheticDataset(Dataset):
    def __init__(self, num_samples: int = 100):
        # 100 samples, 2 continuous features
        self.features = torch.randn(num_samples, 2)
        # Binary target: 1 if x0 + x1 > 0 else 0
        self.labels = (self.features.sum(dim=1) > 0).long()
        
    def __len__(self) -> int:
        return len(self.features)
        
    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

# 2. DataLoader Pipeline
dataset = SyntheticDataset(num_samples=100)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"Total Dataset Samples: {len(dataset)}")
print(f"Total Batches per Epoch: {len(loader)}")

for batch_idx, (batch_x, batch_y) in enumerate(loader):
    if batch_idx == 0 or batch_idx == len(loader) - 1:
        print(f"  Batch {batch_idx}: Features Shape {batch_x.shape} | Labels Shape {batch_y.shape}")
```

---

### 🧠 Diagnostic Mini-Checks (Pillar 8)
1. **Question:** What two methods must every map-style `torch.utils.data.Dataset` implement?  
   *Answer:* `__len__(self)` and `__getitem__(self, idx)`.
2. **Question:** Why is `shuffle=True` critical when training deep neural networks?  
   *Answer:* It prevents the optimizer from memorizing spurious sample ordering dependencies and reduces gradient variance across epochs.
3. **Question:** If your model crashes on the final batch of an epoch with a dimension error, what is the most likely cause?  
   *Answer:* The final batch has fewer samples than `batch_size`, causing shape mismatches in code that hard-coded the batch dimension.

---

## 🎯 Master Diagnostic Self-Assessment

Before proceeding to [NOTES.md](./NOTES.md), verify your foundational mastery across all 8 pillars:

| Pillar & Topic | Core Verification Test | Status |
| :--- | :--- | :--- |
| **§1. Tensor vs NumPy** | Can you name the 3 core attributes of a tensor and explain what `.item()` does? | [ ] Mastered |
| **§2. Compute Devices** | Can you write the canonical `device` selection line and explain why device mismatches crash? | [ ] Mastered |
| **§3. Shape Operations** | Can you explain the difference between `.reshape()` and `.permute()` on an image tensor? | [ ] Mastered |
| **§4. Autograd & Graph** | Can you explain why `y.backward()` requires a scalar and why gradients accumulate in `.grad`? | [ ] Mastered |
| **§5. Optimization Loop** | Can you list the 5 canonical training steps in exact order and explain `with torch.no_grad():`? | [ ] Mastered |
| **§6. nn.Module & Linear** | Can you write a minimal `nn.Module` subclass with `__init__` and `forward`? | [ ] Mastered |
| **§7. Activations & Loss** | Can you explain why `nn.CrossEntropyLoss` requires unnormalized logits and integer targets? | [ ] Mastered |
| **§8. Dataset & Loader** | Can you implement `__len__` and `__getitem__` and wrap a dataset with `DataLoader(shuffle=True)`? | [ ] Mastered |

---

### 🚀 You are ready for the lecture!
Proceed directly to [NOTES.md](./NOTES.md) starting at the **Executive Summary & Master Architecture**.
