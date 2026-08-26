# Tensors, Shapes, Strides, and Memory Layouts

In modern machine learning and Generative AI, **Tensors** are multi-dimensional arrays of uniform numerical data. Understanding tensor shapes, strides, broadcasting, and memory contiguity is essential for designing neural architectures and eliminating shape-mismatch bugs.

```
 ===================================================================================================
                      THE TENSOR DIMENSIONALITY & RANK HIERARCHY
 ===================================================================================================
 
  RANK 0: SCALAR ()            RANK 1: VECTOR (D,)            RANK 2: MATRIX (M, N)          RANK 4: BATCH IMAGE (B, C, H, W)
  Single Number                1D Array / Embedding           2D Table / Linear Layer        Batch of Color Images
  ┌──────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────────┐
  │ 3.1415               │    │ [1.2, -0.5, 3.8, 0.0]   │    │ [[1, 2, 3],             │    │ [32, 3, 224, 224]       │
  │ Loss, Learning Rate  │    │ Word Token, 1D Signal   │    │  [4, 5, 6]]             │    │ Batch=32, RGB=3, 224x224│
  └──────────────────────┘    └─────────────────────────┘    └─────────────────────────┘    └─────────────────────────┘
 ===================================================================================================
```

---

### 1. 👶 ELI5 Intuition: The Multi-Story Apartment Building

Imagine organizing items in real physical space:

1. **A Point / Scalar ($0\text{D}$):** A single marble in your hand (`shape = ()`).
2. **A Line / Vector ($1\text{D}$):** A row of 5 lockers on a hallway wall (`shape = (5,)`).
3. **A Grid / Matrix ($2\text{D}$):** A spreadsheet with 10 rows and 4 columns (`shape = (10, 4)`).
4. **A Building / 3D Tensor:** A 5-story apartment building with 10 floors and 4 rooms per floor (`shape = (5, 10, 4)`).
5. **A City Block / 4D Tensor:** A block of 32 identical apartment buildings (`shape = (32, 5, 10, 4)`). In computer vision, this is a batch of 32 color images: `(Batch, Channels, Height, Width)`.

> 💡 **The Great AI Takeaway:** Deep learning models do not see physical photos or text; they only perform high-speed linear algebra on structured grids of numbers called tensors.

---

### 2. 🔍 Plain-English Breakdown & Notation Rosetta Stone

| Mathematical Symbol / Operation | Formal Name | Plain-English Software Meaning | PyTorch / NumPy Function |
| :--- | :--- | :--- | :--- |
| **$\text{Rank}(T)$** | **Tensor Rank / Order** | Number of axes/dimensions in the tensor (e.g. 1D, 2D, 4D). | `tensor.ndim` |
| **$\text{Shape}(T) = (d_1, \dots, d_k)$** | **Tensor Shape** | The exact size tuple along each dimension. | `tensor.shape` |
| **$T \in \mathbb{R}^{B \times C \times H \times W}$** | **Image Batch Tensor** | $B$ images, $C$ color channels, $H$ pixel height, $W$ pixel width. | `torch.randn(32, 3, 64, 64)` |
| **Flattening / Stacking** | **Vectorization** | Reshaping an image grid $(C, H, W)$ into a long 1D vector $\mathbb{R}^{C \cdot H \cdot W}$. | `x.view(B, -1)` or `x.flatten(1)` |
| **Broadcasting** | **Dimension Expansion** | Automatically stretching smaller tensors to match larger ones during arithmetic. | `x + bias` (adds $(1, D)$ to $(B, D)$) |
| **Strides** | **Memory Step Size** | Number of memory addresses skipped in RAM to move 1 step along an axis. | `tensor.stride()` |
| **Contiguity** | **Contiguous Layout** | Whether elements are stored in adjacent, unbroken RAM addresses. | `tensor.is_contiguous()` |

---

### 3. 📐 The Rules of Broadcasting & Matrix Multiplication

#### A. The NumPy / PyTorch Broadcasting Rules
When adding or multiplying two tensors of different shapes $A$ and $B$, compare their dimension sizes **from right to left**:
1. Two dimensions are compatible if they are **equal**, or **one of them is 1**.
2. If a dimension is missing in one tensor, it is padded with 1 on the left.

```
 EXAMPLE 1: Matrix + Bias Row (Broadcasting Works!)
   Matrix A:  (Batch=32, Features=128)
   Bias B:               (Features=128)  --> Automatically padded to (1, 128)
   ─────────────────────────────────────────────────────────────────────────────
   Result:    (32, 128)                  --> Bias is stamped across all 32 rows!

 EXAMPLE 2: Incompatible Shapes (Broadcasting Crashes!)
   Tensor A:  (32, 64)
   Tensor B:  (32, 10)  --> 64 ≠ 10 and neither is 1! Runtime Error: Shape Mismatch!
```

#### B. Matrix Multiplication Inner Dimension Rule ($M \times K) \cdot (K \times N) \to (M \times N)$
For linear layers $Y = X W^\top + b$:
$$(B \times D_{\text{in}}) \cdot (D_{\text{in}} \times D_{\text{out}}) \to (B \times D_{\text{out}})$$
- The **inner dimensions must match exactly** ($D_{\text{in}} = D_{\text{in}}$).
- The outer dimensions determine the output batch size and feature count.

---

### 4. 🔢 Concrete Micro-Numerical Calculation

Let input batch $X$ have shape $(2, 3)$ and weight matrix $W$ have shape $(3, 2)$:
$$X = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \quad W = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad b = \begin{bmatrix} 10 & 20 \end{bmatrix}$$

1. **Compute Matrix Multiply $X \cdot W$:**
   - Row 1: $[1(1)+2(0)+3(1), \quad 1(0)+2(1)+3(1)] = [1+0+3, \quad 0+2+3] = [4, 5]$
   - Row 2: $[4(1)+5(0)+6(1), \quad 4(0)+5(1)+6(1)] = [4+0+6, \quad 0+5+6] = [10, 11]$
   $$XW = \begin{bmatrix} 4 & 5 \\ 10 & 11 \end{bmatrix}$$
2. **Apply Broadcasted Bias Addition $XW + b$:**
   $$Y = \begin{bmatrix} 4+10 & 5+20 \\ 10+10 & 11+20 \end{bmatrix} = \begin{bmatrix} 14 & 25 \\ 20 & 31 \end{bmatrix}$$

---

### 5. 🔗 Connecting the Dots: Image Flattening in Generative AI

In Lecture 02 of Mathematical Foundations of Generative AI:
1. **Images as Vectors in $\mathbb{R}^D$:** A $64 \times 64$ RGB color image has $3 \times 64 \times 64 = 12{,}288$ pixel intensity values. We **flatten** the 3D grid into a single coordinate point $x \in \mathbb{R}^{12288}$.
2. **The Manifold Sheet:** Natural realistic images do not fill all of $\mathbb{R}^{12288}$. They lie on a very thin, low-dimensional curved manifold embedded within the high-dimensional space.

```
 ===================================================================================================
                 IMAGE TENSOR FLATTENING INTO HIGH-DIMENSIONAL EUCLIDEAN SPACE
 ===================================================================================================
 
  2D PIXEL GRID (H x W)                3D COLOR TENSOR (C x H x W)           1D FLATTENED VECTOR (ℝ^D)
  ┌───────────────────────────┐        ┌───────────────────────────┐         ┌───────────────────────────┐
  │ [ [255, 128, ...],        │ ─────► │ Red   (64 x 64)           │ ──────► │ x = [255, 128, ..., 0]    │
  │   [0,   240, ...] ]       │        │ Green (64 x 64)           │         │ Dimension D = 12,288      │
  │ Grayscale Matrix          │        │ Blue  (64 x 64)           │         │ Single point in ℝ^12288   │
  └───────────────────────────┘        └───────────────────────────┘         └───────────────────────────┘
 ===================================================================================================
```

---

### 6. 💻 Complete Standalone Executable Python Verification Script

```python
"""
TENSORS, SHAPES, STRIDES, AND BROADCASTING DEMONSTRATION
========================================================
Demonstrates multi-dimensional tensor construction, stride inspection,
broadcasting mechanics, and batch linear forward transformations in PyTorch.
"""

import torch
import torch.nn as nn

def run_tensor_simulation():
    print("=" * 70)
    print("  TENSORS, SHAPES, STRIDES & BROADCASTING VERIFICATION")
    print("=" * 70)

    # 1. Create a 4D Batch Image Tensor: (Batch=4, Channels=3, Height=8, Width=8)
    images = torch.randn(4, 3, 8, 8)
    print(f"[4D Batch Image Tensor]:")
    print(f"  * Shape:   {images.shape}")
    print(f"  * Rank:    {images.ndim}D")
    print(f"  * Strides: {images.stride()} (Steps in memory between axes)")
    print(f"  * Total Elements: {images.numel()} floats")

    # 2. Flattening into High-Dimensional Euclidean Space R^D
    # Shape becomes (Batch=4, Features = 3*8*8 = 192)
    flattened = images.flatten(start_dim=1)
    print(f"\n[Flattening to Euclidean Space R^D]:")
    print(f"  * Flattened Shape: {flattened.shape} (D = 192)")

    # 3. Broadcasting Mechanics
    bias_row = torch.ones(1, 192) * 5.0
    broadcasted_sum = flattened + bias_row
    print(f"\n[Broadcasting Validation]:")
    print(f"  * Tensor A Shape: {flattened.shape}")
    print(f"  * Tensor B Shape: {bias_row.shape}")
    print(f"  * Result Shape:   {broadcasted_sum.shape} -> Match!")

    # 4. Linear Forward Layer: Y = X W^T + b
    # Project from R^192 to R^10 (Class logits)
    linear_layer = nn.Linear(in_features=192, out_features=10)
    logits = linear_layer(flattened)
    print(f"\n[Linear Layer Projection]:")
    print(f"  * Input Shape:  {flattened.shape}")
    print(f"  * Weight Shape: {linear_layer.weight.shape}")
    print(f"  * Output Logits Shape: {logits.shape} (Batch=4, Classes=10)")

    print("\n[PASS] All Tensor Shape and Transformation Operations Verified Successfully!")

if __name__ == "__main__":
    run_tensor_simulation()
```

---

### 7. 🩺 Diagnostic Mini-Checks & Common Traps

1. **Trap 1: "Does `tensor.view()` create a copy of data in memory?"**
   - *Correction:* No! `view()` creates a new shape metadata view over the **existing memory buffer** without copying data. However, it requires the tensor to be **contiguous** in memory (call `.contiguous()` if transposed).
2. **Trap 2: "What is the difference between `tensor.reshape()` and `tensor.view()`?"**
   - *Correction:* `view()` strictly requires contiguous memory and fails with an error if memory is non-contiguous. `reshape()` will return a view if possible, or automatically allocate a contiguous copy if needed.
3. **Diagnostic Check:** If $A$ has shape $(16, 1, 64)$ and $B$ has shape $(1, 32, 64)$, what is the shape of $A + B$?
   - *Answer:* Comparing right-to-left: $64=64$, $\max(1, 32)=32$, $\max(16, 1)=16 \implies \mathbf{(16, 32, 64)}$.
