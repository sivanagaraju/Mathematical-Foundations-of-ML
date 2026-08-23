# Prerequisites — warm-up before this PyTorch tensor session

> **Do this first** if “list vs array,” “matrix multiply,” “GPU,” or “notebook cell” still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Playlist slot is **W1_L3** (YouTube title says *F-divergence*). **The 28-minute recording is PyTorch tensors on Colab.**  
> **Beginner deep warm-up:** definition · worked micro · analogy · notice · mini-check.

```
  After this warm-up you can say:

  "A Python list of lists is a nested Python object; a tensor is a rectangular block of numbers."
  "Shape (2, 3) means 2 rows and 3 columns."
  "NumPy ndarray is the CPU cousin of a tensor."
  "A notebook has text cells and code cells; play runs one cell."
  "Colab runs on a Google machine, not (by default) your laptop GPU."
  "Matrix multiply needs inner dimensions to match: (M×N)(N×P)=(M×P)."
  "Element-wise multiply needs the SAME shape: each pair of entries multiplies."
  "Transpose flips rows and columns."
```

**Warm-up → lecture boxes**

```
  §1  Python list / nested list     ──► Topic 4
  §2  Shape (rows, columns)         ──► Topics 5–7
  §3  NumPy ndarray                 ──► Topics 3–4
  §4  Notebook cells + Colab        ──► Topics 1–2
  §5  CPU vs GPU                    ──► Topics 3, 7
  §6  Matrix multiply rule          ──► Topic 9
  §7  Transpose                     ──► Topic 9
  §8  Element-wise vs matmul        ──► Topic 10
```

---

## 1. Python list vs a rectangle of numbers

<a id="p1-list"></a>

### Purpose for the video

He starts from `data = [[1, 2], [3, 4]]` — a **list of lists** — then wraps it in `torch.tensor`.

### Worked micro

`[[1, 2], [3, 4]]` is two Python lists inside a list. It *looks* like a 2×2 matrix but Python does not enforce a rectangle. A tensor **does**.

### Analogy — shopping bags vs a spreadsheet

A list of bags can hold mixed junk. A spreadsheet cell grid is rectangular. `torch.tensor` copies the numbers into a spreadsheet.

### Mini-check

1. Is `[[1, 2], [3]]` a valid tensor of shape (2, 2)?  
2. Does `torch.tensor(data)` change the *values* or the *type*?

---

## 2. Shape `(rows, columns)`

<a id="p2-shape"></a>

### Purpose for the video

`shape = (2, 3)` means 2 rows × 3 columns. He passes this tuple to `torch.rand` / `ones` / `zeros`.

### Worked micro

A (2, 3) ones tensor:

```
  1  1  1
  1  1  1
```

Two numbers in the shape; six entries.

### Analogy — hotel floors

`(2, 3)` = 2 floors, 3 rooms each. `rand(shape)` puts a random number in every room.

### Mini-check

How many numbers in `torch.zeros((4, 4))`?

---

## 3. NumPy `ndarray`

<a id="p3-numpy"></a>

### Purpose for the video

He converts `np.array(data)` → `torch.from_numpy(np_array)`.

### Worked micro

NumPy is the standard **CPU** grid of numbers in Python science. PyTorch tensors can share that memory (`bridge-to-np` on screen).

### Analogy — two brands of graph paper

Same grid of numbers. NumPy paper stays on the desk (CPU). Tensor paper can be carried to a GPU desk.

### Mini-check

Do you need `import numpy as np` if you only have a Python list?

---

## 4. Notebook cells and Colab

<a id="p4-colab"></a>

### Purpose for the video

`.ipynb` = text cells + code cells. Play button runs **one** code cell on a **Google** machine.

### Worked micro

First click is slow: Colab must **connect** to a server. Packages are already there — no `pip` drama for this tutorial.

### Analogy — borrowed kitchen

You type a recipe; a kitchen in another city cooks it. Your laptop is the notepad, not the stove.

### Mini-check

If Colab is in the browser, whose GPU is the default 12 GB he mentions?

---

## 5. CPU vs GPU

<a id="p5-device"></a>

### Purpose for the video

`tensor.device` printed **cpu** in the demo. GPU is optional; CPU always exists.

### Worked micro

Same 3×4 random tensor can live on `cpu` or `cuda`. Moving copies bytes; large copies are expensive (official warning on screen).

### Analogy — desk vs warehouse forklift

CPU = desk calculator. GPU = forklift for thousands of multiplies at once. This hour stays at the desk.

### Mini-check

If `device` prints `cpu`, did we fail the tutorial?

---

## 6. Matrix-multiply shape rule

<a id="p6-matmul"></a>

### Purpose for the video

He writes $A_{M\times N} \times B_{N\times P} = C_{M\times P}$. Inner $N$ must match.

### Worked micro

(2×3) times (3×4) is OK → (2×4). (2×3) times (2×4) is **illegal**.

### Analogy — handshake

The outgoing hands of A must equal incoming hands of B.

```
  A  M × N     B  N × P
         └──match──┘
  C  M × P
```

### Mini-check

Can you `@` a 4×4 with another 4×4? With its transpose?

---

## 7. Transpose

<a id="p7-transpose"></a>

### Purpose for the video

`tensor.T` flips the rectangle so `tensor @ tensor.T` is always legal for a 2-D tensor.

### Worked micro

```
  [[1, 2, 3],        [[1, 4],
   [4, 5, 6]]   T →   [2, 5],
                      [3, 6]]
```

### Analogy — turning a poster

Rows become columns. The ink does not change; the orientation does.

---

## 8. Element-wise vs matrix multiply

<a id="p8-hadamard"></a>

### Purpose for the video

`*` / `.mul` = same-shape entry-by-entry (used in **convolutions**). `@` / `.matmul` = linear-algebra product.

### Worked micro

```
  A = [1  2]     B = [4  2]     A * B = [ 4   4]
      [-2 3]         [1 -1]             [-2  -3]
```

`A @ B` would be a **different** 2×2 (row-dot-column).

### Analogy — mixing paint vs composing machines

Element-wise: each can’s pigment mixes with the matching can. Matmul: every input talks to every output through weights — the NN layer.

### Mini-check

If shapes differ, which operation still might work: `*` or `@`?

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz: [quiz.html](./quiz.html) Part A = this file.
