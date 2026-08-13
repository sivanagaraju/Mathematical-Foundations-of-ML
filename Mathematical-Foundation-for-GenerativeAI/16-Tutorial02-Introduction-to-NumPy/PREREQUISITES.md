0

# Prerequisites — warm-up before Tutorial 2 (Introduction to NumPy)

> **Do this first** if arrays, shapes, indexing, or “matrix multiply vs elementwise” still blur.
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.
> Course: NPTEL Mathematical Foundations of Generative AI · Tutorial 2.
> **Beginner deep warm-up:** definition · micro example · analogy · notice · mini-check.
> Assumes Tutorial 1 Python basics (lists, loops, functions).

```
  After this warm-up you can say:

  "A NumPy array is a typed multi-dimensional grid with a shape tuple."
  "shape tells rows×cols×…; dtype is the type of every entry."
  "Indexing picks elements; slicing picks sub-arrays (views when possible)."
  "reshape re-packs the same data into a new shape; flatten makes a 1D vector."
  "Broadcasting lets small arrays stretch to match larger ones for ops."
  "Elementwise ops use *, +; true matrix multiply uses @ or np.matmul."
  "A linear layer is y = X @ W + b (shapes must align)."
  "ReLU zeros negatives; softmax turns logits into a probability vector."
```

**Warm-up → tutorial boxes**

```
  §1  Arrays vs Python lists              ──► Topics 1–2
  §2  Shape and dtype                     ──► Topics 1, 3–4
  §3  Indexing and slicing                ──► Topic 2
  §4  Reshape / flatten                   ──► Topic 3
  §5  Elementwise vs matrix multiply      ──► Topic 4
  §6  Broadcasting intuition              ──► Topic 4
  §7  Linear layer + bias                 ──► Topics 4–5
  §8  Activations, softmax, loss, labels  ──► Topics 5–10 (incl. one-hot, train/test, conv idea)
```

---

## 1. Arrays vs Python lists

<a id="p1-array"></a>

### Purpose for the video

NumPy’s star is `ndarray`. Lists are fine for general Python; **numeric bulk work** needs arrays.

### Definitions


| Idea                        | Meaning                                                          |
| ----------------------------- | ------------------------------------------------------------------ |
| **Python list**             | Flexible sequence; can mix types; slow loops for big math        |
| **NumPy array (`ndarray`)** | Homogeneous (one`dtype`), multi-dimensional, fast vectorized ops |
| **Vectorized**              | One operation on whole array without writing a Python`for`       |

### Worked micro

```python
# list: flexible but slow for big math
xs = [1, 2, 3]

# array: same numbers, one type, shape (3,)
import numpy as np
a = np.array([1, 2, 3])
print(type(a))   # <class 'numpy.ndarray'>
```

### Analogy — cardboard boxes vs shipping pallets

Lists are loose cardboard boxes of mixed goods. NumPy arrays are **pallets**: same item type, stacked in a regular grid, forklifts (vectorized ops) move whole pallets at once.

### Notice

- `np.array([1, 2, 3.5])` promotes to float.
- Prefer arrays for ML tensors; keep lists for non-numeric structure.

### Mini-check

1. Why not train a neural net with nested Python lists of floats?
2. What does “vectorized” mean in one sentence?

---

## 2. Shape and dtype

<a id="p2-shape"></a>

### Purpose for the video

Almost every DL bug is a **shape** bug. The lecture hammers shapes.

### Definitions


| Term      | Meaning                                                          |
| ----------- | ------------------------------------------------------------------ |
| **shape** | Tuple of sizes along each axis, e.g.`(2, 5)` = 2 rows, 5 columns |
| **ndim**  | Number of axes (length of shape)                                 |
| **dtype** | Type of each element (`float64`, `int32`, …)                    |
| **size**  | Total number of elements (= product of shape entries)            |

### Worked micro

```python
z = np.zeros((2, 5))
print(z.shape)   # (2, 5)
print(z.dtype)   # float64 by default
print(z.size)    # 10
```

```
  shape (2, 5):

    · · · · ·
    · · · · ·
```

### Analogy — apartment floor plan

Shape is the floor plan: 2 floors × 5 rooms. dtype is “every room holds a float.” Wrong plan → furniture (operations) will not fit.

### Notice

- `(3,)` is a 1D vector of length 3; `(3, 1)` is a column; `(1, 3)` is a row.
- Batch dimension is often the **first** axis in ML: `(batch, features)`.

### Mini-check

1. Image batch of 32 grayscale 28×28 images → typical shape?
2. What is `size` of shape `(4, 3, 2)`?

---

## 3. Indexing and slicing

<a id="p3-index"></a>

### Purpose for the video

You constantly grab rows, columns, patches, and time steps.

### Definitions


| Syntax         | Meaning                                  |
| ---------------- | ------------------------------------------ |
| `a[i]`         | element / row at index`i` (0-based)      |
| `a[i:j]`       | slice from`i` inclusive to `j` exclusive |
| `a[i:j:k]`     | slice with step`k`                       |
| `a[r, c]`      | 2D pick row`r`, column `c`               |
| `a[:, c]`      | all rows, column`c`                      |
| Negative index | count from end (`-1` = last)             |

### Worked micro

```python
a = np.arange(10)          # 0..9
print(a[2:7])              # [2 3 4 5 6]
print(a[::2])              # [0 2 4 6 8]

M = np.arange(12).reshape(3, 4)
print(M[1, 2])             # single entry
print(M[:, 1])             # second column
```

### Analogy — spreadsheet selection

Indexing is clicking one cell. Slicing is drag-selecting a block of cells. `:` means “whole row/column.”

### Notice

- Slices often return **views** (shared memory) — changing the slice can change the original.
- Out-of-range index → error; empty slice → empty array.

### Mini-check

1. From `0..9`, what is `a[3:8:2]`?
2. How do you take the last column of a 2D array?

---

## 4. Reshape and flatten

<a id="p4-reshape"></a>

### Purpose for the video

MNIST images enter as matrices and become **vectors** for an MLP.

### Definitions


| Op                  | Meaning                                           |
| --------------------- | --------------------------------------------------- |
| **reshape**         | Same data, new shape; product of sizes must match |
| **flatten / ravel** | Collapse to 1D                                    |
| **-1 in reshape**   | “Infer this dimension”                          |

### Worked micro

```python
img = np.arange(28 * 28).reshape(28, 28)
vec = img.reshape(784)          # or img.flatten()
batch = np.random.rand(32, 28, 28)
flat = batch.reshape(32, -1)    # (32, 784)
```

### Analogy — folding a letter

The paper (data) is the same; folding changes the layout (shape). Flatten is fully unfolding into one long strip.

### Notice

- `reshape` does not invent data — only rearranges.
- Wrong product of dimensions → error.

### Mini-check

1. Can you reshape `(2, 3)` to `(3, 3)`? Why?
2. MNIST 28×28 → vector length?

---

## 5. Elementwise ops vs matrix multiply

<a id="p5-mul"></a>

### Purpose for the video

`*` and `@` are **not** the same. Confusing them is a classic bug.

### Definitions


| Op                           | Meaning                                       |
| ------------------------------ | ----------------------------------------------- |
| `a + b`, `a * b`             | **Elementwise** (same shape or broadcastable) |
| `a @ b` or `np.matmul(a, b)` | **Matrix product** (linear algebra)           |
| `np.dot`                     | Related; prefer`@` for clarity in 2D          |

### Worked micro

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[10, 20], [30, 40]])
print(A * B)    # [[10,40],[90,160]] elementwise
print(A @ B)    # true matrix product
```

### Analogy — cooking

Elementwise `*`: season each piece of food separately.
`@`: a recipe that mixes ingredients with fixed weights (rows combine columns).

### Notice

- For vectors of length $n$, `u * v` is elementwise; `u @ v` is a scalar dot product.
- Shapes for `A @ B`: last dim of `A` must match second-to-last of `B`.

### Mini-check

1. When do `A * B` and `A @ B` agree for 2×2 matrices?
2. Shape of `(4, 5) @ (5, 3)`?

---

## 6. Broadcasting

<a id="p6-broadcast"></a>

### Purpose for the video

Add a bias vector to every row; add a column vector to every column — without Python loops.

### Idea

NumPy **virtually stretches** smaller arrays along missing or size-1 axes so shapes align for elementwise ops.

### Worked micro

```python
X = np.ones((3, 4))
b = np.array([1.0, 2.0, 3.0, 4.0])   # shape (4,)
Y = X + b                            # b stretches to (3, 4)
```

```
  X (3,4)          b (4,)
  · · · ·          1 2 3 4
  · · · ·    +     (repeat row)
  · · · ·
```

### Analogy — rubber stamp

Bias is a one-line stamp. Broadcasting presses that stamp onto every row of the batch.

### Notice

- Not all shapes broadcast; incompatible sizes raise `ValueError`.
- Mental check: walk axes from the right; each pair must match or one side is 1.

### Mini-check

1. Can you add shape `(3, 1)` to `(3, 4)`?
2. Can you add `(2, 3)` to `(3, 2)`?

---

## 7. Linear layer $XW + b$

<a id="p7-linear"></a>

### Purpose for the video

The lecture builds `XW + b` (or `xW + b`) as the workhorse of MLPs.

### Shape contract (common ML layout)


| Tensor        | Example shape                 | Role                    |
| --------------- | ------------------------------- | ------------------------- |
| $X$           | `(batch, in_features)`        | inputs                  |
| $W$           | `(in_features, out_features)` | weights                 |
| $b$           | `(out_features,)`             | bias                    |
| $Y = X@W + b$ | `(batch, out_features)`       | pre-activation / logits |

### Worked micro

```python
X = np.random.randn(8, 4)     # 8 examples, 4 features
W = np.random.randn(4, 5)     # 4 → 5
b = np.random.randn(5)
Y = X @ W + b                 # (8, 5)
```

### Analogy — exam scoring

Each student is a row of answers ($X$). Each column of $W$ is a rubric for one skill score. Bias is a constant curve applied to every student. Output: skill scores per student.

### Notice

- Some notes write $Wx$ with column vectors; this tutorial often uses **row vectors / batch-first** and `x @ W`.
- Always print `.shape` after each step when debugging.

### Mini-check

1. If $X$ is `(32, 784)` and $W$ is `(784, 10)`, what is $Y$’s shape?
2. Why must $b$ broadcast to that shape?

---

## 8. ReLU, softmax, and loss (ideas only)

<a id="p8-act"></a>

### Purpose for the video

After linear maps: **nonlinearity**, **probabilities**, **loss**.

### Definitions


| Piece             | Meaning                                                            |
| ------------------- | -------------------------------------------------------------------- |
| **ReLU**          | $\mathrm{ReLU}(z)=\max(0,z)$ — zeros negatives                    |
| **Logits**        | Raw scores before softmax                                          |
| **Softmax**       | Maps a vector of logits to positive numbers that sum to 1          |
| **Cross-entropy** | Loss that punishes wrong class probability:$-\log p_{\text{true}}$ |

### Worked micro

```python
z = np.array([-1.0, 0.5, 2.0])
relu = np.maximum(0, z)              # [0, 0.5, 2]

logits = np.array([1.0, 2.0, 3.0])
e = np.exp(logits - logits.max())    # stable softmax
p = e / e.sum()                      # probabilities
```

### Analogy — contest scoring

Logits are raw judge scores. Softmax turns them into a **share of the prize pot** (probabilities). Cross-entropy is a fine if the winner you claimed is not the true winner.

### One-hot labels (preview for Topics 7–8)

Class id `2` with **4** classes → vector `[0, 0, 1, 0]`: length = number of classes, single `1` at the true index.

```python
n_classes = 4
label = 2
oh = np.zeros(n_classes)
oh[label] = 1
# batch of labels [0, 2, 1, 3] → matrix (4, 4) with one 1 per row
```

**Analogy:** a multiple-choice bubble sheet — only one bubble filled per question.

### Train / test split + accuracy (preview for Topic 9)

Shuffle **shared indices**, then slice 80% train / 20% test so features and labels stay aligned.

```python
idx = np.arange(N)
np.random.shuffle(idx)
cut = int(0.8 * N)
train, test = idx[:cut], idx[cut:]
acc = np.mean(y_pred == y_true)   # fraction correct
```

**Trap:** shuffle only `X` and not `y` → desynced labels.

### Convolution idea (preview for Topic 7)

A small **kernel** slides over an image; each placement: elementwise multiply + sum → one output pixel.
Output size (no pad, stride 1): $H - k + 1$ on each side.
**Pooling** (max/avg) shrinks spatial size (e.g. 4×4 → 2×2 with 2×2 pools, stride 2).

**Analogy:** a flashlight window scanning a wall painting — each flash records one summary number.

### Notice

- Softmax over classes is per **example** (per row of a batch).
- ReLU does not change shape.
- Training = lower loss by adjusting $W, b$ (gradients appear in logistic regression section).
- MSE is for regression-style targets; CE/BCE for classification with probabilities.
- Always print `.shape` after each layer when debugging.

### Mini-check

1. Softmax outputs must sum to what?
2. What does ReLU do to $-3$?
3. One-hot for class `1` with 3 classes?
4. Why shuffle shared indices for train/test?

---

### Paper check

1. Create a `(3, 4)` array of zeros.
2. Shape of `(2, 3) @ (3, 5)`?
3. Difference between `*` and `@`?
4. Flatten one 28×28 image → length?
5. ReLU of `[-2, 0, 4]`?
6. One-hot for class `0` with 3 classes?
7. 4×4 image, 2×2 kernel, no pad, stride 1 → output spatial size?

**Peek:** (1) `np.zeros((3,4))` (2) `(2,5)` (3) elementwise vs matmul (4) 784 (5) `[0,0,4]` (6) `[1,0,0]` (7) `3×3`.

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).
Quiz: [quiz.html](./quiz.html) Part A = this file.
