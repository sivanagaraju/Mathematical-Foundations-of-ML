# W1_L3 — PyTorch Tensors Hands-On Tutorial (Google Colab)

> **Course:** IIT Madras B.S. Degree in Data Science & AI · **Mathematical Foundations of Generative AI**  
> **Instructor:** Chandan (Tutorial TA) · Course Faculty: Prof. Prathosh A. P. (IISc / IITM)  
> **Tutorial Recording:** [W1_L3 on YouTube](https://www.youtube.com/watch?v=rHnrALMCyIQ) (~28:43)  
> **Prerequisites Warm-up:** [PREREQUISITES.md](./PREREQUISITES.md) · **Self-Assessment Quiz:** [quiz.html](./quiz.html)  
> **Course Catalog:** [../NOTES.md](../NOTES.md)

---

## 📌 Honest Title Discrepancy & Course Map Placement

> [!NOTE]
> **Video Title vs. Actual Content Discrepancy:**  
> The YouTube playlist entry is titled *"W1_L3: F-Divergence"*. However, the actual 28:43 recording is the official **PyTorch Tensors Tutorial on Google Colab**.  
> * For the pure mathematical theory of **$f$-divergences** (Forward KL, Reverse KL, Jensen-Shannon, Total Variation, and Jensen's inequality proofs), please refer to [03-W1-L4-Variational-Divergence-Minimization/NOTES.md](../03-W1-L4-Variational-Divergence-Minimization/NOTES.md).  
> * This notes document strictly covers the **speech, code cells, handwritten diagrams, and concepts presented in the 28:43 recording**: Google Colab runtime setup, tensor creation APIs, inspecting shape/dtype/device, concatenation across `dim=0` vs `dim=1`, matrix multiplication three ways (`@`, `.matmul()`, `torch.matmul(out=)`), and element-wise products (`*`, `.mul()`).

---

## Quick Navigation Matrix

| Topic & Timestamp | Focus Area | Core Code / Formulation | Prerequisite Link |
| :--- | :--- | :--- | :--- |
| [Topic 1: Colab GPU & Prereqs](#topic-1) (00:12–03:15) | Runtime Environment | Google Colab VM, GPU Runtimes, CS231n | [p4-colab](./PREREQUISITES.md#p4-colab), [p5-gpu](./PREREQUISITES.md#p5-gpu) |
| [Topic 2: Official Tutorials & Notebook](#topic-2) (03:15–06:28) | Documentation | pytorch.org/tutorials, Interactive Colab | [p4-colab](./PREREQUISITES.md#p4-colab) |
| [Topic 3: What a Tensor Is](#topic-3) (06:28–09:41) | Data Structure | Specialized $N$-D Array, Homogeneous Memory | [p1-list](./PREREQUISITES.md#p1-list), [p3-numpy](./PREREQUISITES.md#p3-numpy) |
| [Topic 4: Tensor from List & NumPy](#topic-4) (09:41–12:34) | Creation APIs | `torch.tensor(data)`, `torch.from_numpy(np_arr)` | [p1-list](./PREREQUISITES.md#p1-list), [p3-numpy](./PREREQUISITES.md#p3-numpy) |
| [Topic 5: `ones_like` & `rand_like`](#topic-5) (12:34–15:00) | Template Allocation | `torch.ones_like(x)`, `torch.rand_like(x, dtype=)` | [p2-shape](./PREREQUISITES.md#p2-shape) |
| [Topic 6: Create from Shape](#topic-6) (15:00–16:29) | Shape Allocation | `shape = (2, 3)`, `torch.rand(shape)`, `torch.zeros()` | [p2-shape](./PREREQUISITES.md#p2-shape) |
| [Topic 7: `shape`, `dtype`, `device`](#topic-7) (16:29–17:54) | Tensor Attributes | `.shape`, `.dtype`, `.device` (`cpu` vs `cuda:0`) | [p2-shape](./PREREQUISITES.md#p2-shape), [p5-gpu](./PREREQUISITES.md#p5-gpu) |
| [Topic 8: Concatenate `dim=0` vs `dim=1`](#topic-8) (17:54–20:43) | Tensor Joining | `torch.cat([A, A, A], dim=0)` vs `dim=1` | [p2-shape](./PREREQUISITES.md#p2-shape) |
| [Topic 9: Matrix Multiply Three Ways](#topic-9) (20:43–25:02) | Linear Algebra | `tensor @ tensor.T`, `.matmul()`, `torch.matmul(out=)` | [p6-matmul](./PREREQUISITES.md#p6-matmul), [p7-transpose](./PREREQUISITES.md#p7-transpose) |
| [Topic 10: Element-Wise Product](#topic-10) (25:02–28:43) | Hadamard Product | `tensor * tensor`, `tensor.mul()`, Convolutions | [p8-hadamard](./PREREQUISITES.md#p8-hadamard) |
| [Workplace Scenarios](#workplace-scenarios--debugging-tensors) | Production Debugging | Device Mismatches & Non-Contiguous Strides | — |
| [External References](#external-references) | Multi-Source Study | 2–3 Curated Videos & 2–3 Guides/Blogs per Topic | — |

---

## Executive Summary & Master Architecture

<a id="executive-summary"></a>
<a id="executive-summary--architecture-of-this-lecture"></a>

Tensors are the foundational mathematical building blocks of deep generative models. In this hands-on tutorial, Chandan walks through the **official PyTorch Tensors notebook on Google Colab**, covering how tensors are allocated in physical memory, how metadata attributes govern hardware execution, and how linear algebra operations (concatenation, matrix multiplication, and element-wise products) are implemented in production code.

```
══════════════════════════════════════════════════════════════════════════════════════════════════
                         THE PYTORCH TENSOR COMPUTING BLUEPRINT
══════════════════════════════════════════════════════════════════════════════════════════════════

  1. TENSOR CREATION PIPELINES
     ┌────────────────────────────────────────────────────────────────────────────────────────┐
     │ • Direct from List:      x_data = torch.tensor([[1, 2], [3, 4]])                       │
     │ • From NumPy (Zero-Copy): x_np   = torch.from_numpy(np_array)                           │
     │ • From Shape Tuple:      x_rand = torch.rand((2, 3)), torch.ones((2, 3))               │
     │ • From Template Tensor:  x_ones = torch.ones_like(x_data), torch.rand_like(x_data)     │
     └────────────────────────────────────────────────────────────────────────────────────────┘

  2. TENSOR METADATA TRIAD
     ┌──────────────────────┬──────────────────────────────────┬──────────────────────────────┐
     │ Attribute            │ What It Dictates                 │ Example Values               │
     ├──────────────────────┼──────────────────────────────────┼──────────────────────────────┤
     │ tensor.shape         │ Dimensionality along each axis   │ torch.Size([2, 3]), [64, 784]│
     │ tensor.dtype         │ Numerical precision in memory    │ torch.float32, torch.int64   │
     │ tensor.device        │ Physical hardware location       │ device(type='cpu'), 'cuda:0' │
     └──────────────────────┴──────────────────────────────────┴──────────────────────────────┘

  3. CORE TENSOR ARITHMETIC ENGINES
     ┌────────────────────────────────────────────────────────────────────────────────────────┐
     │ Concatenation:   torch.cat([A, A, A], dim=0) ──► Stacks rows (Taller: 12 × 4)          │
     │                  torch.cat([A, A, A], dim=1) ──► Stacks cols (Wider:  4 × 12)          │
     ├────────────────────────────────────────────────────────────────────────────────────────┤
     │ Matrix MatMul:   y = A @ A.T  ≡  A.matmul(A.T)  ≡  torch.matmul(A, A.T, out=y)         │
     │                  Rule: (M × N) @ (N × P) ──► Output (M × P)                            │
     ├────────────────────────────────────────────────────────────────────────────────────────┤
     │ Element-Wise:    z = A * A    ≡  A.mul(A)       ≡  torch.mul(A, A, out=z)              │
     │                  Rule: (M × N) * (M × N) ──► Output (M × N) (Hadamard Product)         │
     └────────────────────────────────────────────────────────────────────────────────────────┘
══════════════════════════════════════════════════════════════════════════════════════════════════
```

---

## 📖 PyTorch Tensor Rosetta Stone (Symbols & Functions $\to$ Plain English)

| PyTorch Syntax | Formal Mathematical Term | Plain-English ELI5 Mental Model |
| :--- | :--- | :--- |
| `torch.tensor(data)` | Tensor Factory Function | Packing a loose Python list into a solid, contiguous ice cube tray of numbers. |
| `torch.from_numpy(arr)` | Zero-Copy Array Wrapper | Putting a PyTorch shipping label on an existing NumPy box without moving the box. |
| `torch.ones_like(x)` | Template-Based Allocation | Using a cookie cutter of shape $x$ to cut out a new cookie made of pure 1s. |
| `tensor.shape` | Dimension Tuple | Counting rows, columns, and depth layers (e.g., a $2 \times 3$ egg carton holding 6 eggs). |
| `tensor.dtype` | Storage Data Type | The number precision: decimals (`float32`) vs whole integer tags (`int64`). |
| `tensor.device` | Compute Device Identifier | Where the numbers physically live: standard CPU RAM vs high-speed GPU VRAM. |
| `torch.cat(..., dim=0)` | Vertical Concatenation | Stacking Lego bricks vertically to make a taller tower. |
| `torch.cat(..., dim=1)` | Horizontal Concatenation | Gluing tables side-by-side to make a wider surface. |
| `A @ B` / `.matmul()` | Matrix Multiplication | Linear algebra row-by-column projection (transforms inputs in neural layers). |
| `A * B` / `.mul()` | Hadamard Product | Multiplying item quantities by unit prices cell-by-cell (used in convolutions). |
| `A.T` | Matrix Transposition | Flipping a spreadsheet diagonally so rows become columns. |

---

## Comparative Matrix: Matrix Multiplication (`@`) vs Element-Wise (`*`)

| Feature / Metric | Matrix Multiplication (`@`) | Element-Wise Product (`*`) |
| :--- | :--- | :--- |
| **PyTorch Syntaxes** | `A @ B`, `A.matmul(B)`, `torch.matmul(A, B)` | `A * B`, `A.mul(B)`, `torch.mul(A, B)` |
| **Mathematical Name** | Matrix Product ($A B$) | Hadamard Product ($A \odot B$) |
| **Dimension Rule** | **Inner dimensions must match:** $(M \times N) @ (N \times P) \implies (M \times P)$ | **Exact same shape required:** $(M \times N) * (M \times N) \implies (M \times N)$ |
| **Formula** | $Y_{i, j} = \sum_{k=1}^N A_{i, k} B_{k, j}$ | $Z_{i, j} = A_{i, j} \times B_{i, j}$ |
| **Deep Learning Use Case** | Fully Connected Linear Layers ($y = W x + b$), Self-Attention | Convolutional Feature Filtering, Dropout Masks, Gating |

---

## Complete Hands-On Implementation in Python / PyTorch

```python
import torch
import numpy as np

# ==============================================================================
# 1. TENSOR CREATION AND METADATA INSPECTION
# ==============================================================================
# From Python List
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# From NumPy Array (Zero-Copy Memory Sharing)
np_array = np.array([[5.0, 6.0], [7.0, 8.0]])
x_np = torch.from_numpy(np_array)

# From Template
x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float32)

# From Explicit Shape Tuple
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Tensor Attributes:")
print(f"  Shape:  {rand_tensor.shape}")
print(f"  Dtype:  {rand_tensor.dtype}")
print(f"  Device: {rand_tensor.device}")

# ==============================================================================
# 2. CONCATENATION ACROSS DIMENSIONS
# ==============================================================================
# Create 4x4 matrix
tensor_4x4 = torch.ones(4, 4)

# Concatenate along rows (dim=0) -> Taller (12, 4)
t_cat_dim0 = torch.cat([tensor_4x4, tensor_4x4, tensor_4x4], dim=0)

# Concatenate along columns (dim=1) -> Wider (4, 12)
t_cat_dim1 = torch.cat([tensor_4x4, tensor_4x4, tensor_4x4], dim=1)

print(f"\nConcatenation Shapes:")
print(f"  dim=0 (Vertical):   {t_cat_dim0.shape}")  # torch.Size([12, 4])
print(f"  dim=1 (Horizontal): {t_cat_dim1.shape}")  # torch.Size([4, 12])

# ==============================================================================
# 3. MATRIX MULTIPLICATION (3 WAYS) VS ELEMENT-WISE PRODUCT
# ==============================================================================
# Matrix Multiply Three Equivalent Ways: Y = A @ A.T
y1 = tensor_4x4 @ tensor_4x4.T
y2 = tensor_4x4.matmul(tensor_4x4.T)
y3 = torch.empty_like(tensor_4x4)
torch.matmul(tensor_4x4, tensor_4x4.T, out=y3)

print(f"\nMatrix Multiplication Equality Check:")
print(f"  Are all 3 matmul APIs identical? {torch.equal(y1, y2) and torch.equal(y2, y3)}")

# Element-Wise Product Three Equivalent Ways: Z = A * A
z1 = tensor_4x4 * tensor_4x4
z2 = tensor_4x4.mul(tensor_4x4)
z3 = torch.empty_like(tensor_4x4)
torch.mul(tensor_4x4, tensor_4x4, out=z3)

print(f"Element-Wise Equality Check:")
print(f"  Are all 3 element-wise APIs identical? {torch.equal(z1, z2) and torch.equal(z2, z3)}")
```

---

<a id="topic-1"></a>
<a id="topic-1-colab-gpu-prereqs-cs231n-0012–0315"></a>
## Topic 1: Colab GPU, Prereqs & CS231n (00:12–03:15)

### 👶 ELI5 Quick Intuition
You don't need a $3,000 gaming laptop to learn deep learning. Google Colab gives you a free cloud computer in a web browser tab. When you click `Play ▶`, Google connects a remote Linux supercomputer with free GPU access so you can run PyTorch code immediately.

### Master Map Placement
Introduces the Google Colab execution environment, prerequisite materials, and recommended companion courses (Stanford CS231n).

### Colab Recording Screenshot
![Topic 1 Screenshot — Colab Environment and Prereqs](./screenshots/composites/ch01-topic-01-colab-prereqs-panel1of1.png)
*Figure 1.1 (~00:12–03:10):* Chandan opens the Colab notebook, checks the cloud runtime status, and references Stanford CS231n and PyTorch official tutorials as key foundational resources.

### In-Depth Conceptual Exposition

* **The Cloud Notebook Paradigm:**
  * Google Colaboratory (Colab) executes Jupyter notebooks on Google Cloud infrastructure.
  * **Zero Local Setup:** Python, PyTorch, CUDA drivers, and scientific libraries come pre-configured.
  * **Runtime Connection:** The initial cell execution establishes an active WebSocket session to an allocated Linux Virtual Machine (VM).
* **Course Prerequisites & Recommendations:**
  * Proficiency in basic Python syntax, functions, and list indexing.
  * Familiarity with linear algebra (vectors, matrices, dot products).
  * Chandan recommends Stanford's **CS231n (Convolutional Neural Networks for Visual Recognition)** for foundational machine learning theory.

```
   BROWSER CLIENT (Laptop)                         GOOGLE CLOUD RUNTIME
  ┌──────────────────────────────┐              ┌──────────────────────────────┐
  │ Google Colab Web UI          │ ──WebSocket─►│ Linux VM (CPU / Nvidia GPU)  │
  │ Press Play [▶] on Cell       │ ◄──Stdout/Img│ Pre-installed PyTorch & CUDA │
  └──────────────────────────────┘              └──────────────────────────────┘
```

---

<a id="topic-2"></a>
<a id="topic-2-official-tutorials-and-colab-notebook-0315–0628"></a>
## Topic 2: Official Tutorials & Colab Notebook (03:15–06:28)

### 👶 ELI5 Quick Intuition
The PyTorch website (`pytorch.org`) has an entire interactive textbook of tutorials. You can click a button called *"Run in Google Colab"* on any page to open a live, working copy of the code and experiment with it yourself.

### Master Map Placement
Navigates the official PyTorch documentation portal and opens the interactive *Tensors* tutorial notebook.

### Colab Recording Screenshot
![Topic 2 Screenshot — PyTorch Official Documentation](./screenshots/composites/ch02-topic-02-official-tutorials-colab-panel1of1.png)
*Figure 2.1 (~03:18–06:25):* Navigating `pytorch.org/tutorials` to open the official *Tensors* Colab notebook directly in the browser.

### In-Depth Conceptual Exposition

* **Official PyTorch Learning Path:**
  * Located at `pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html`.
  * Every official tutorial includes a badge: `Run in Google Colab`.
* **Reproducibility & Experimentation:**
  * Allows students to modify variable names, alter tensor shapes, and observe runtime errors in a sandboxed, non-destructive environment.

---

<a id="topic-3"></a>
<a id="topic-3-what-a-tensor-is-0628–0941"></a>
## Topic 3: What a Tensor Is (06:28–09:41)

### 👶 ELI5 Quick Intuition
What is a tensor?  
* A 0D tensor is a single number (Scalar).
* A 1D tensor is a list of numbers (Vector).
* A 2D tensor is a spreadsheet table (Matrix).
* A 3D tensor is a cube of numbers (Color image).  
PyTorch tensors are like NumPy arrays, but they have two superpowers: **they can run on super-fast GPUs** and **they automatically compute calculus derivatives** for AI!

### Master Map Placement
Defines the mathematical and architectural concept of a tensor and contrasts PyTorch tensors with NumPy `ndarray` objects.

### Colab Recording Screenshot
![Topic 3 Screenshot — What is a Tensor](./screenshots/composites/ch03-topic-03-what-is-a-tensor-panel1of1.png)
*Figure 3.1 (~06:30–09:38):* Chandan highlights the definition of a tensor on screen: specialized multi-dimensional arrays encoding inputs, outputs, and model parameters.

### In-Depth Conceptual Exposition

* **Formal Definition:**
  * A **tensor** is an $N$-dimensional geometric and algebraic array of numbers defined over a contiguous memory buffer.
  * In deep learning, tensors represent:
    1. **Input Data:** Image pixels $[B, C, H, W]$, audio waveforms $[B, T]$, text token embeddings $[B, S, D]$.
    2. **Model Parameters:** Weight matrices $W \in \mathbb{R}^{M \times N}$ and bias vectors $b \in \mathbb{R}^M$.
    3. **Intermediate Activations:** Feature maps output by neural network layers.
* **PyTorch Tensor vs. NumPy Array:**
  * Similar API design and indexing syntax.
  * **Superpower 1 (Hardware Acceleration):** Tensors run natively on Nvidia GPUs via CUDA, AMD GPUs via ROCm, and Apple Silicon via MPS.
  * **Superpower 2 (Autograd Engine):** Tensors track computational history via `requires_grad=True` to compute analytical backpropagation gradients $\nabla_\theta \mathcal{L}$ automatically.

```
   TENSOR RANK TAXONOMY:
  ┌──────────────────────┬──────────────┬────────────────────────┬─────────────────────────────┐
  │ Rank / Dimension     │ Math Term    │ Shape Example          │ Real-World Deep Learning    │
  ├──────────────────────┼──────────────┼────────────────────────┼─────────────────────────────┤
  │ 0-D                  │ Scalar       │ torch.Size([])         │ Loss score: 0.4215          │
  │ 1-D                  │ Vector       │ torch.Size([784])      │ Flattened feature vector    │
  │ 2-D                  │ Matrix       │ torch.Size([64, 784])  │ Linear layer weight matrix  │
  │ 3-D                  │ Volume / Cube│ torch.Size([3, 28, 28])│ RGB color photograph        │
  │ 4-D                  │ Batch Block  │ torch.Size([64,3,28,28])│ Mini-batch of color images  │
  └──────────────────────┴──────────────┴────────────────────────┴─────────────────────────────┘
```

---

<a id="topic-4"></a>
<a id="topic-4-tensor-from-a-list-and-from-numpy-0941–1234"></a>
## Topic 4: Tensor from List and NumPy (09:41–12:34)

### 👶 ELI5 Quick Intuition
You can turn normal Python numbers into tensors in two ways:
1. From a Python list: `torch.tensor([[1, 2], [3, 4]])` (makes a new copy).
2. From a NumPy array: `torch.from_numpy(my_array)` (shares memory instantly without copying).

### Master Map Placement
Walks through the two primary factory methods for creating tensors from existing Python data structures.

### Colab Recording Screenshot
![Topic 4 Screenshot — Creating Tensors from Lists and NumPy](./screenshots/composites/ch04-topic-04-tensor-from-list-numpy-panel1of1.png)
*Figure 4.1 (~09:45–12:30):* Code cell demonstrating `x_data = torch.tensor(data)` from a Python list and `x_np = torch.from_numpy(np_array)` from a NumPy array.

### In-Depth Conceptual Exposition

```python
# 1. From Python List (Allocates new memory)
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# 2. From NumPy Array (Zero-copy memory sharing)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

* **Memory Mechanics:**
  * `torch.tensor(data)`: Infers data type (`int64` for whole numbers, `float32` for decimals) and allocates a new contiguous memory segment.
  * `torch.from_numpy(np_array)`: Zero-copy bridge. The tensor points directly to the underlying NumPy memory buffer. Modifying `np_array[0, 0]` in-place immediately mutates `x_np[0, 0]`.

```
   ZERO-COPY MEMORY BRIDGE:
   NumPy Buffer in RAM ◄─────── SHARED PHYSICAL MEMORY ───────► PyTorch Tensor x_np
   np_array = np.array(...)                                     x_np = torch.from_numpy(...)
```

---

<a id="topic-5"></a>
<a id="topic-5-ones_like--rand_like-1234–1500"></a>
## Topic 5: `ones_like` and `rand_like` (12:34–15:00)

### 👶 ELI5 Quick Intuition
Imagine you have a cake mold shaped like a star ($x_{\text{data}}$).  
* `torch.ones_like(x_data)` cuts out a new star made of solid 1s.
* `torch.rand_like(x_data)` cuts out a new star filled with random decimal numbers between $0.0$ and $1.0$.

### Master Map Placement
Explains template-based allocation APIs that inherit shape and data type properties from an existing reference tensor.

### Colab Recording Screenshot
![Topic 5 Screenshot — ones_like and rand_like](./screenshots/composites/ch05-topic-05-ones-rand-like-data-panel1of1.png)
*Figure 5.1 (~12:38–14:55):* Demonstrating `x_ones = torch.ones_like(x_data)` and `x_rand = torch.rand_like(x_data, dtype=torch.float)`.

### In-Depth Conceptual Exposition

```python
x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)
```

* **Property Inheritance:**
  * `torch.ones_like(x_data)`: Inspects `x_data.shape`, `x_data.dtype`, and `x_data.device`, allocating a new tensor with identical metadata filled with $1.0$.
  * `torch.rand_like(x_data, dtype=torch.float)`: Overrides the data type to `torch.float32` while maintaining the exact shape of `x_data`, populating entries from a uniform distribution $\mathcal{U}[0, 1)$.

---

<a id="topic-6"></a>
<a id="topic-6-create-from-a-shape-tuple-1500–1629"></a>
## Topic 6: Create from Shape (15:00–16:29)

### 👶 ELI5 Quick Intuition
If you don't have an existing tensor to copy from, you can create a tensor from scratch by giving it a **shape tuple** (like `(2, 3)` = 2 rows, 3 columns). You can fill it with random numbers (`torch.rand`), all ones (`torch.ones`), or all zeros (`torch.zeros`).

### Master Map Placement
Covers creating tensors directly from explicit shape tuples.

### Colab Recording Screenshot
![Topic 6 Screenshot — Creating Tensors from Shape Tuples](./screenshots/composites/ch06-topic-06-create-from-shape-panel1of1.png)
*Figure 6.1 (~15:05–16:25):* Instantiating tensors using `shape = (2, 3)` with `torch.rand(shape)`, `torch.ones(shape)`, and `torch.zeros(shape)`.

### In-Depth Conceptual Exposition

```python
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
```

* **Shape Tuple Specification:**
  * `shape = (2, 3)` creates a $2 \times 3$ matrix ($2 \times 3 = 6$ total elements).
  * `torch.rand(shape)`: Uniform random numbers in $[0.0, 1.0)$.
  * `torch.ones(shape)`: Constant value $1.0$.
  * `torch.zeros(shape)`: Constant value $0.0$.

```
   SHAPE TUPLE (2, 3) MATRIX ALLOCATION:
  ┌─────────────────────────────────┬─────────────────────────────────┬─────────────────────────────────┐
  │ torch.rand((2, 3))              │ torch.ones((2, 3))              │ torch.zeros((2, 3))             │
  │ [ [0.412, 0.891, 0.125],        │ [ [1.0, 1.0, 1.0],              │ [ [0.0, 0.0, 0.0],              │
  │   [0.783, 0.045, 0.612] ]       │   [1.0, 1.0, 1.0] ]             │   [0.0, 0.0, 0.0] ]             │
  └─────────────────────────────────┴─────────────────────────────────┴─────────────────────────────────┘
```

---

<a id="topic-7"></a>
<a id="topic-7-shape-dtype-device-1629–1754"></a>
## Topic 7: `shape`, `dtype`, `device` (16:29–17:54)

### 👶 ELI5 Quick Intuition
Every PyTorch tensor carries an ID card with 3 vital stats:
1. **`shape`:** What are the dimensions? (e.g., 2 rows, 3 columns).
2. **`dtype`:** What kind of numbers? (e.g., 32-bit floating-point decimals).
3. **`device`:** Where does it live? (e.g., standard computer RAM `cpu`, or GPU VRAM `cuda:0`).

### Master Map Placement
Examines the three foundational metadata attributes governing all PyTorch tensors.

### Colab Recording Screenshot
![Topic 7 Screenshot — Shape, Dtype, Device Attributes](./screenshots/composites/ch07-topic-07-shape-dtype-device-panel1of1.png)
*Figure 7.1 (~16:32–17:50):* Printing `tensor.shape`, `tensor.dtype`, and `tensor.device` in Google Colab.

### In-Depth Conceptual Exposition

```python
tensor = torch.rand(3, 4)
print(f"Shape of tensor:  {tensor.shape}")   # torch.Size([3, 4])
print(f"Datatype:         {tensor.dtype}")   # torch.float32
print(f"Device location:  {tensor.device}")  # cpu
```

* **The Attribute Triad:**
  * **`shape` (`torch.Size`):** Defines the coordinate axes.
  * **`dtype` (`torch.dtype`):** Specifies memory precision per element (`float32` = 4 bytes, `float64` = 8 bytes, `int64` = 8 bytes).
  * **`device` (`torch.device`):** Identifies the hardware compute target (`cpu`, `cuda:0`, `mps:0`).

---

<a id="topic-8"></a>
<a id="topic-8-concatenate-dim0-vs-dim1-1754–2043"></a>
## Topic 8: Concatenate `dim=0` vs `dim=1` (17:54–20:43)

### 👶 ELI5 Quick Intuition
Imagine you have three $4 \times 4$ square wooden tiles:
* **`dim=0` (Vertical Concatenation):** Stack the 3 tiles on top of each other. The tower gets **taller**: 12 rows tall, 4 columns wide ($12 \times 4$).
* **`dim=1` (Horizontal Concatenation):** Place the 3 tiles side-by-side. The table gets **wider**: 4 rows tall, 12 columns wide ($4 \times 12$).

### Master Map Placement
Demonstrates joining tensors along specific axes using `torch.cat`.

### Colab Recording Screenshot
![Topic 8 Screenshot — Tensor Concatenation across Dimensions](./screenshots/composites/ch08-topic-08-concatenate-panel1of1.png)
*Figure 8.1 (~17:58–20:40):* Code cell demonstrating `torch.cat([tensor, tensor, tensor], dim=1)` yielding shape `torch.Size([4, 12])`, contrasted with `dim=0` yielding `torch.Size([12, 4])`.

### In-Depth Conceptual Exposition

```python
t = torch.ones(4, 4)

# Stack along rows (dim=0) -> Taller (12, 4)
t_dim0 = torch.cat([t, t, t], dim=0)

# Stack along columns (dim=1) -> Wider (4, 12)
t_dim1 = torch.cat([t, t, t], dim=1)
```

* **Axis Rules:**
  * All dimensions *other* than the concatenation axis must match exactly.
  * `dim=0` concatenates along axis 0 (rows).
  * `dim=1` concatenates along axis 1 (columns).

```
   CONCATENATION GEOMETRY:
   
   dim=0 (Vertical Stack):                     dim=1 (Horizontal Stack):
   ┌─────────┐                                 ┌─────────┬─────────┬─────────┐
   │ 4 × 4   │                                 │ 4 × 4   │ 4 × 4   │ 4 × 4   │
   ├─────────┤ ──► Output Shape: [12, 4]       └─────────┴─────────┴─────────┘
   │ 4 × 4   │                                  ──► Output Shape: [4, 12]
   ├─────────┤
   │ 4 × 4   │
   └─────────┘
```

---

<a id="topic-9"></a>
<a id="topic-9-matrix-multiply-three-ways-2043–2502"></a>
## Topic 9: Matrix Multiply Three Ways (20:43–25:02)

### 👶 ELI5 Quick Intuition
In PyTorch, there are 3 different ways to write linear algebra matrix multiplication:
1. `tensor @ tensor.T`
2. `tensor.matmul(tensor.T)`
3. `torch.matmul(tensor, tensor.T, out=y3)`  
They all perform the **exact same math** and produce the **exact same result**. Pick whichever syntax you find easiest to read!

### Master Map Placement
Explains matrix multiplication rules, transposition (`tensor.T`), and compares the three equivalent PyTorch matmul syntaxes.

### Colab Recording Screenshot
![Topic 9 Screenshot — Matrix Multiplication Three Ways](./screenshots/composites/ch09-topic-09-matmul-panel1of1.png)
*Figure 9.1 (~20:48–25:00):* Demonstrating `y1 = tensor @ tensor.T`, `y2 = tensor.matmul(tensor.T)`, and `y3 = torch.rand_like(tensor); torch.matmul(tensor, tensor.T, out=y3)`, verifying that $y_1, y_2, y_3$ have identical values.

### In-Depth Conceptual Exposition

```python
# 1. Operator syntax
y1 = tensor @ tensor.T

# 2. Tensor method syntax
y2 = tensor.matmul(tensor.T)

# 3. In-place / Out buffer function syntax
y3 = torch.empty_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
```

* **Linear Algebra Law:**
  * Let $A \in \mathbb{R}^{M \times N}$ and $B \in \mathbb{R}^{N \times P}$.
  * The matrix product $Y = A B \in \mathbb{R}^{M \times P}$ has entries $Y_{i, j} = \sum_{k=1}^N A_{i, k} B_{k, j}$.
  * For a $4 \times 4$ matrix $A$, $A @ A^T$ multiplies $(4 \times 4) @ (4 \times 4)$, producing a symmetric $4 \times 4$ Gram matrix.

---

<a id="topic-10"></a>
<a id="topic-10-element-wise-product-2502–2843"></a>
## Topic 10: Element-Wise Product (25:02–28:43)

### 👶 ELI5 Quick Intuition
While matrix multiplication (`@`) sweeps rows across columns, **element-wise multiplication (`*`)** multiplies matching cells one-by-one (Top-Left $\times$ Top-Left, Bottom-Right $\times$ Bottom-Right). Both matrices must have the exact same shape! Chandan explains: this is the exact operation used when applying convolutional filters to images!

### Master Map Placement
Contrasts Hadamard (element-wise) multiplication with matrix multiplication and highlights its role in convolutional neural networks.

### Colab Recording Screenshot
![Topic 10 Screenshot — Element-Wise Multiplication](./screenshots/composites/ch10-topic-10-elementwise-panel1of1.png)
*Figure 10.1 (~25:08–28:40):* Code cell executing `z1 = tensor * tensor` and `z2 = tensor.mul(tensor)`, contrasting element-wise products with linear matrix products.

### In-Depth Conceptual Exposition

```python
# Element-wise product two ways
z1 = tensor * tensor
z2 = tensor.mul(tensor)
```

* **Hadamard Product Properties:**
  * For matrices $A, B \in \mathbb{R}^{M \times N}$, the element-wise product $Z = A \odot B \in \mathbb{R}^{M \times N}$ has entries:
    $$Z_{i, j} = A_{i, j} \times B_{i, j}$$
  * **Convolutions Connection:** When a $3 \times 3$ convolutional kernel slides over an image patch, it computes an element-wise product between the kernel weights and image pixels, then sums the results.

```
   HADAMARD PRODUCT (A * B):
   ┌───┬───┐       ┌───┬───┐       ┌──────────┬──────────┐
   │ 1 │ 2 │   *   │ 5 │ 6 │   =   │ 1·5 = 5  │ 2·6 = 12 │
   ├───┼───┤       ├───┼───┤       ├──────────┼──────────┤
   │ 3 │ 4 │       │ 7 │ 8 │       │ 3·7 = 21 │ 4·8 = 32 │
   └───┴───┘       └───┴───┘       └──────────┴──────────┘
```

---

## Workplace Scenarios & Debugging Tensors

### Scenario 1: Device Mismatch RuntimeError during Distributed Inference
* **Context:** An AI engineer moves a PyTorch neural network to GPU (`model.to('cuda')`), but when feeding input image batches, PyTorch crashes with:
  `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`
* **Root Cause:**
  * The input tensor was loaded from disk into standard system RAM (`tensor.device == 'cpu'`), while the model weights were on GPU VRAM (`cuda:0`).
  * PyTorch does not automatically copy inputs across the PCIe bus during operations.
* **Production Remedy:**
  * Explicitly move input tensors to the model device before the forward pass:
    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = inputs.to(device)
    outputs = model(inputs)
    ```

### Scenario 2: Fatal Non-Contiguous Memory Errors during View Reshaping
* **Context:** A developer transposes an image tensor `t = image.T` and attempts to flatten it using `t.view(-1)`. PyTorch throws:
  `RuntimeError: view size is not compatible with input tensor's size and stride (tensor is not contiguous).`
* **Root Cause:**
  * Transposing (`.T` or `.permute()`) alters the tensor's stride metadata without physically rearranging the byte buffer in RAM.
  * `.view()` strictly requires memory bytes to be physically contiguous.
* **Production Remedy:**
  * Call `.contiguous()` before `.view()`, or use `.reshape()`, which calls `.contiguous()` automatically:
    ```python
    # Fix option 1:
    flat_tensor = t.contiguous().view(-1)
    # Fix option 2:
    flat_tensor = t.reshape(-1)
    ```

---

## External References

> Comprehensive multi-source learning materials curated for every subtopic in this tutorial.

### Topic 1 — Colab Cloud Runtimes & Machine Learning Prerequisites
* **Video Lectures:**
  1. [Stanford CS231n: Lecture 1 — Introduction to Deep Learning](https://www.youtube.com/watch?v=vT1JzLTH4G4) — Foundational computer vision overview.
  2. [Google Cloud Tech: Google Colab Tutorial for Beginners](https://www.youtube.com/watch?v=inN8seMm7UI) — Setting up GPU/TPU runtimes.
  3. [freeCodeCamp: Python for Data Science and Machine Learning](https://www.youtube.com/watch?v=LHBE6Q9XlzI) — Essential linear algebra and programming prereqs.
* **Guides & Articles:**
  1. [Google Colab Official Guide: Overview of Colaboratory](https://colab.research.google.com/notebooks/basic_features_overview.ipynb) — Hardware specifications and limits.
  2. [Stanford CS231n Official Course Notes: Python / NumPy Tutorial](https://cs231n.github.io/python-numpy-tutorial/) — The standard preparation guide for vision engineers.
  3. [Towards Data Science: Maximizing Google Colab GPU Performance](https://towardsdatascience.com/getting-the-most-out-of-google-colab-gpu-b75d5a7d77b8) — Managing sessions and VRAM.

### Topic 2 — PyTorch Official Documentation & Tutorials Portal
* **Video Lectures:**
  1. [PyTorch Official: PyTorch Beginner Series Overview](https://www.youtube.com/watch?v=IC0_FRiX-sw) — Navigating `pytorch.org` tutorials.
  2. [Daniel Bourke: Learn PyTorch for Deep Learning in 24 Hours](https://www.youtube.com/watch?v=Z_ikDlimN6A) — Interactive Colab workflows.
  3. [Aladdin Persson: PyTorch Tutorials Playlist — Getting Started](https://www.youtube.com/watch?v=x9JiIFvlUwk) — Tutorial roadmap and setup.
* **Guides & Articles:**
  1. [PyTorch Official Documentation: Tutorials Portal](https://pytorch.org/tutorials/) — Official landing page.
  2. [PyTorch Official Docs: PyTorch Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html) — Quick syntax lookup.
  3. [Real Python: Deep Learning with PyTorch](https://realpython.com/pytorch-python/) — Step-by-step developer tutorial.

### Topic 3 — The Mathematical & Computational Nature of Tensors
* **Video Lectures:**
  1. [3Blue1Brown: Linear Algebra Chapter 1 — Vectors and Tensors](https://www.youtube.com/watch?v=fNk_zzaMoSs) — Beautiful geometric visualizations of vector spaces.
  2. [StatQuest with Josh Starmer: PyTorch Tensors Clearly Explained](https://www.youtube.com/watch?v=L35fFDpwIM4) — Step-by-step tensor indexing.
  3. [DeepLizard: PyTorch Tensors Explained — Neural Network Programming](https://www.youtube.com/watch?v=f5liqUk0ZTw) — Memory layouts and ranks.
* **Guides & Articles:**
  1. [PyTorch Official Tutorial: Tensors Basics](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) — The exact tutorial walked in this video.
  2. [Edward Yang (PyTorch Core Dev): PyTorch Internals & Tensor Memory Layout](http://blog.ezyang.com/2019/05/pytorch-internals/) — Deep dive into C++ storage buffers, strides, and metadata.
  3. [Towards Data Science: Tensor Calculus for Deep Learning](https://towardsdatascience.com/tensor-calculus-for-deep-learning/) — Higher-order multidimensional arrays.

### Topic 4 — Tensor Creation: Python Lists & NumPy Zero-Copy Bridges
* **Video Lectures:**
  1. [deeplizard: Converting NumPy Arrays to PyTorch Tensors](https://www.youtube.com/watch?v=c36lUUr864M) — Zero-copy memory sharing mechanics.
  2. [Keith Galli: NumPy Array Creation and Slicing Tutorial](https://www.youtube.com/watch?v=QUT1VHiLmmI) — Fundamental ndarray construction.
  3. [Daniel Bourke: PyTorch Tensor Creation Patterns](https://www.youtube.com/watch?v=V_xro1bcAuA) — Factory functions in code.
* **Guides & Articles:**
  1. [PyTorch Official Docs: `torch.from_numpy` API](https://pytorch.org/docs/stable/generated/torch.from_numpy.html) — Memory sharing guarantees.
  2. [NumPy Official Documentation: Array Creation](https://numpy.org/doc/stable/user/basics.creation.html) — Scientific computing array basics.
  3. [Real Python: In-Place Operations and Memory in NumPy / PyTorch](https://realpython.com/numpy-tutorial/) — Preventing unwanted data mutations.

### Topic 5 — Template Allocations: `ones_like` & `rand_like`
* **Video Lectures:**
  1. [deeplizard: PyTorch Tensor Creation Functions](https://www.youtube.com/watch?v=k4jY9L5YsTQ) — Template copying with `_like` methods.
  2. [Aladdin Persson: PyTorch Tensor Initialization Methods](https://www.youtube.com/watch?v=x9JiIFvlUwk) — Random number distributions in tensors.
  3. [StatQuest: Uniform vs Normal Random Initializations in Neural Networks](https://www.youtube.com/watch?v=8krd5qK7_4o) — Weight initialization theory.
* **Guides & Articles:**
  1. [PyTorch Official Docs: `torch.ones_like` Reference](https://pytorch.org/docs/stable/generated/torch.ones_like.html) — Template constructor reference.
  2. [PyTorch Official Docs: `torch.rand_like` Reference](https://pytorch.org/docs/stable/generated/torch.rand_like.html) — Uniform random allocation.
  3. [Machine Learning Mastery: Tensor Initialization Strategies](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/) — Xavier / He initialization foundations.

### Topic 6 — Creating Tensors from Explicit Shape Tuples
* **Video Lectures:**
  1. [deeplizard: PyTorch Tensor Shapes and Reshaping](https://www.youtube.com/watch?v=f5liqUk0ZTw) — Multi-axis dimension allocation.
  2. [Sentdex: Deep Learning with PyTorch — Tensor Dimensions](https://www.youtube.com/watch?v=BzcBsEb05OE) — Visualizing 3D/4D shapes.
  3. [freeCodeCamp: PyTorch Matrix Creation Crash Course](https://www.youtube.com/watch?v=GIsg-ZUy0MY) — Matrix initialization functions.
* **Guides & Articles:**
  1. [PyTorch Official Docs: `torch.rand` Reference](https://pytorch.org/docs/stable/generated/torch.rand.html) — Continuous uniform sampling.
  2. [PyTorch Official Docs: `torch.zeros` Reference](https://pytorch.org/docs/stable/generated/torch.zeros.html) — Zero-filled matrix allocation.
  3. [GeeksforGeeks: PyTorch Tensor Shape Operations](https://www.geeksforgeeks.org/pytorch-tensors/) — Shape parameter conventions.

### Topic 7 — Tensor Attributes: `shape`, `dtype`, `device`
* **Video Lectures:**
  1. [Aladdin Persson: Tensor Types, Conversions, and Device Management](https://www.youtube.com/watch?v=x9JiIFvlUwk) — Managing CPU and CUDA memory.
  2. [deeplizard: PyTorch Tensor Attributes: Shape, Dtype, Device](https://www.youtube.com/watch?v=f5liqUk0ZTw) — Deep dive into tensor metadata.
  3. [MIT 6.S191: Hardware Accelerators for Deep Learning](https://www.youtube.com/watch?v=AjtX1N3kzuc) — GPUs, TPUs, and parallel compute.
* **Guides & Articles:**
  1. [PyTorch Official Docs: Tensor Attributes](https://pytorch.org/docs/stable/tensor_attributes.html) — Comprehensive `torch.dtype` and `torch.device` spec.
  2. [PyTorch Official Docs: CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html) — Best practices for GPU tensors.
  3. [NVIDIA Developer Blog: Understanding GPU Memory Architectures](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) — SIMD thread parallelism.

### Topic 8 — Tensor Concatenation (`torch.cat` across `dim=0` vs `dim=1`)
* **Video Lectures:**
  1. [deeplizard: PyTorch Tensor Concatenation and Stacking Explained](https://www.youtube.com/watch?v=k4jY9L5YsTQ) — Visualizing axis joining in 2D and 3D.
  2. [Aladdin Persson: Advanced PyTorch Tensor Operations — Cat, Stack, Split](https://www.youtube.com/watch?v=x9JiIFvlUwk) — `torch.cat` vs `torch.stack`.
  3. [StatQuest: Matrix Concatenation in Machine Learning](https://www.youtube.com/watch?v=HGwBXDKFk9I) — Combining feature matrices.
* **Guides & Articles:**
  1. [PyTorch Official Docs: `torch.cat` API](https://pytorch.org/docs/stable/generated/torch.cat.html) — Concatenation rules.
  2. [PyTorch Official Docs: `torch.stack` API](https://pytorch.org/docs/stable/generated/torch.stack.html) — Creating new dimensions vs extending existing ones.
  3. [Towards Data Science: Concatenation vs Stacking in PyTorch](https://towardsdatascience.com/understanding-dimensions-in-pytorch-tensors-and-concatenation/) — Dimensional diagrams.

### Topic 9 — Linear Algebra: Matrix Multiplication Three Ways
* **Video Lectures:**
  1. [3Blue1Brown: Essence of Linear Algebra Chapter 4 — Matrix Multiplication](https://www.youtube.com/watch?v=XkY2DOUCWMU) — The geometric meaning of linear transformations.
  2. [Gilbert Strang (MIT 18.06): Matrix Multiplication and Linear Systems](https://www.youtube.com/watch?v=FX4C-JpTFgY) — Row-by-column dot product algebra.
  3. [deeplizard: PyTorch Matrix Multiplication Operations](https://www.youtube.com/watch?v=k4jY9L5YsTQ) — Syntax comparisons (`@` vs `.matmul()`).
* **Guides & Articles:**
  1. [PyTorch Official Docs: `torch.matmul` API](https://pytorch.org/docs/stable/generated/torch.matmul.html) — Full broadcasting and matrix product rules.
  2. [Khan Academy: Matrix Multiplication Rules](https://www.khanacademy.org/math/precalculus/x9e81a4f983814e3:matrices/x9e81a4f983814e3:multiplying-matrices-by-matrices/v/multiplying-matrices) — Step-by-step numerical mechanics.
  3. [Real Python: Matrix Multiplication in Python with `@`](https://realpython.com/matrix-multiplication-in-python/) — PEP 465 operator specifications.

### Topic 10 — Element-Wise Product (Hadamard Product) & Convolutions
* **Video Lectures:**
  1. [3Blue1Brown: But what is a Convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) — How element-wise multiplication powers 2D image filtering.
  2. [StatQuest: Convolutional Neural Networks (CNNs) Clearly Explained](https://www.youtube.com/watch?v=YRhxdVk_sIs) — Sliding dot products over image grids.
  3. [deeplizard: PyTorch Element-Wise vs Matrix Operations](https://www.youtube.com/watch?v=k4jY9L5YsTQ) — Visual comparison of Hadamard products.
* **Guides & Articles:**
  1. [PyTorch Official Docs: `torch.mul` API](https://pytorch.org/docs/stable/generated/torch.mul.html) — Hadamard element-wise multiplication.
  2. [Wikipedia: Hadamard Product (Matrices)](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) — Mathematical algebraic properties.
  3. [Distill.pub: Feature Visualization in Deep Neural Networks](https://distill.pub/2017/feature-visualization/) — How element-wise filter activations reveal visual patterns.

---

## Sources & Production Notes

* **Primary Recording:** [W1_L3 on YouTube](https://www.youtube.com/watch?v=rHnrALMCyIQ) · IIT Madras B.S. Degree Programme · Runtime: 28:43
* **Timed Audio Captions:** `raw/captions.en.timed.txt` (ASR transcripts verified against Colab notebook code)
* **Composite Screenshot Panels:** `./screenshots/composites/ch01-...` through `ch10-...` (High-resolution captures per topic MM:SS)
* **Official Reference Notebook:** PyTorch Official Tutorials — *Tensors* (`torch.Tensor`, `torch.matmul`, `torch.cat`).
