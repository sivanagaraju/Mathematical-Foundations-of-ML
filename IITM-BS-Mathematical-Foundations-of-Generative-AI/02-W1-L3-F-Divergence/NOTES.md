# W1_L3 — PyTorch tensors (playlist title: F-divergence)

> **Video:** [W1_L3 on the IITM playlist](https://www.youtube.com/watch?v=rHnrALMCyIQ&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=3) · **~29 min**  
> **Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md) · **Quiz:** [quiz.html](./quiz.html)

**Do PREREQUISITES before this article.**

**Title vs recording:** YouTube title and description say *F-divergence / variational divergence minimization*. The **28:43 file is a Colab walkthrough of the official PyTorch tensor tutorial** (create, attributes, `cat`, `@` vs `*`). This package follows **what is on screen and in the audio**, not the metadata. For actual $f$-divergence chalk, use [NPTEL Lec 03](../../Mathematical-Foundation-for-GenerativeAI/25-Lec03-f-Divergence-Examples/NOTES.md) or playlist [W1_L4 VDM](https://www.youtube.com/watch?v=nfZQYopzv20).

**Previous (true lecture):** [W1_L2 problem setting](../01-W1-L2-Introduction-Problem-Setting/NOTES.md) · **Catalog:** [../NOTES.md](../NOTES.md)

| When he hits… | Warm-up |
|---------------|---------|
| List → tensor | [p1-list](./PREREQUISITES.md#p1-list) |
| `shape=(2,3)` | [p2-shape](./PREREQUISITES.md#p2-shape) |
| NumPy | [p3-numpy](./PREREQUISITES.md#p3-numpy) |
| Colab / ipynb | [p4-colab](./PREREQUISITES.md#p4-colab) |
| `tensor.device` | [p5-device](./PREREQUISITES.md#p5-device) |
| $M\times N$ times $N\times P$ | [p6-matmul](./PREREQUISITES.md#p6-matmul) |
| `.T` | [p7-transpose](./PREREQUISITES.md#p7-transpose) |
| `*` vs `@` | [p8-hadamard](./PREREQUISITES.md#p8-hadamard) |

---

## Table of Contents

1. [Topic 1: Colab GPU, prereqs, CS231n](#topic-1-colab-gpu-prereqs-cs231n-0011–0442) (00:11–04:42)
2. [Topic 2: Official tutorials and the Colab notebook](#topic-2-official-tutorials-and-the-colab-notebook-0442–0814) (04:42–08:14)
3. [Topic 3: What a tensor is](#topic-3-what-a-tensor-is-0814–0941) (08:14–09:41)
4. [Topic 4: Tensor from a list and from NumPy](#topic-4-tensor-from-a-list-and-from-numpy-0941–1234) (09:41–12:34)
5. [Topic 5: `ones_like` / `rand_like`](#topic-5-ones_like--rand_like-1234–1500) (12:34–15:00)
6. [Topic 6: Create from a shape](#topic-6-create-from-a-shape-1500–1629) (15:00–16:29)
7. [Topic 7: shape, dtype, device](#topic-7-shape-dtype-device-1629–1754) (16:29–17:54)
8. [Topic 8: Concatenate `dim=0` vs `dim=1`](#topic-8-concatenate-dim0-vs-dim1-1754–2043) (17:54–20:43)
9. [Topic 9: Matrix multiply three ways](#topic-9-matrix-multiply-three-ways-2043–2502) (20:43–25:02)
10. [Topic 10: Element-wise product](#topic-10-element-wise-product-2502–2843) (25:02–28:43)
11. [External references](#external-references)
12. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

You need a number-grid that can sit on a GPU so later generative models are not stuck in NumPy. The method is the official PyTorch **Tensors** notebook on **Google Colab**: wrap a list or ndarray, or name a shape; then join, multiply, or Hadamard-multiply. The fork that matters is `@` (matmul, inner dims match) versus `*` (same shape, used in convolutions).

**Worldview arc:** from “we will code simple DGM models on Colab” **to** a tensor you can create, inspect, concatenate, and multiply two different ways.

### System context

```
  ╔════════════════════════════════════╗
  ║ Later: GANs / VAEs on small data   ║
  ║ Outside: TensorFlow (optional)     ║
  ║ Outside: CS231n if CNN rusty       ║
  ╚══════════════╤═════════════════════╝
                 │ this 29 min
                 ▼
        ┌─────────────────────┐
        │ Tensor toolkit      │
        └─────────────────────┘
                 │
                 ▼
        STOP: DataLoader, Autograd, nn.Module
        NOT this file: f-divergence math
```

### Main blueprint

```
  Colab  (.ipynb on a Google GPU box)
     │  official pytorch.org/tutorials
     ▼
  import torch [, numpy as np]
     │
     ├─ from data ──► torch.tensor(list)
     │                torch.from_numpy(ndarray)
     ├─ from template ──► ones_like / rand_like  (dtype can override)
     └─ from shape ──► torch.rand / ones / zeros (2, 3)
                 │
                 ▼
  attributes:  .shape   .dtype   .device  (demo stays CPU)
                 │
                 ├─ torch.cat([...], dim=1)  → more COLUMNS
                 ├─ torch.cat([...], dim=0)  → more ROWS
                 ├─ @  / .matmul / torch.matmul(..., out=)
                 │     A(M×N) @ B(N×P) = C(M×P)
                 └─ *  / .mul             same shape (convolutions)
```

**Commented hour in one block** (what he actually ran):

```python
import torch
import numpy as np

# --- create ---
data = [[1, 2], [3, 4]]              # Python list of lists
x_data = torch.tensor(data)          # values unchanged; type → Tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)     # same shape as x_data, all 1s
x_rand = torch.rand_like(x_data, dtype=torch.float)  # override dtype

shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# --- inspect ---
tensor = torch.rand(3, 4)
tensor.shape, tensor.dtype, tensor.device   # e.g. cpu, float32

# --- join ---
t_cols = torch.cat([tensor, tensor, tensor], dim=1)  # more columns
t_rows = torch.cat([tensor, tensor, tensor], dim=0)  # more rows

# --- matmul (three APIs, same numbers) ---
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# --- element-wise (same shape; used in conv) ---
z1 = tensor * tensor
z2 = tensor.mul(tensor)
```

### Scenario walkthrough

Open [pytorch.org/tutorials](https://pytorch.org/tutorials/) → Tensors → **Open in Colab**. First play connects to a Google box. Build `[[1,2],[3,4]]` as a tensor, print `shape`/`dtype`/`device`, then `@ tensor.T` vs `* tensor`. You now have the two products later CNN/GAN notebooks will call without explaining.

### Failure / contrast

```
  * when you meant @          ──X──►  wrong 2×2
  @ with inner dims mismatch  ──X──►  runtime error
  cat without choosing dim    ──X──►  4×12 vs 12×4 surprise
  assuming local GPU          ──X──►  device is cpu until you .to()
```

### STOP

No DataLoader, no autograd, no `nn.Module`. Slicing is skipped. **No $f$-divergence** despite the YouTube title.

### Load-bearing claims

- This hour is the official **tensor** notebook on **Colab**.
- Tensor ≈ NumPy array that can live on a GPU.
- Create from list, NumPy, `*_like`, or explicit `shape`.
- Inspect `.shape`, `.dtype`, `.device`.
- `cat(..., dim=1)` columns; `dim=0` rows.
- Three matmul APIs agree; inner dimensions must match.
- `*` is same-size Hadamard (convolutions); not `matmul`.

**Speaker:** IIT Madras BS tutorial TA (not the chalk f-div lecture).

---

## Topic 1: Colab GPU, prereqs, CS231n (00:11–04:42)

### Where this sits on the master map

**SETUP.** He frames the rest of the course’s **code** track: simple DGM models on a small GPU. [Colab](./PREREQUISITES.md#p4-colab) unlocks “not your laptop.”

### Board / screenshot

![PyTorch tutorials homepage on Colab](./screenshots/composites/ch01-topic-01-colab-prereqs-panel1of1.png)
**Figure — ~00:14–02:45:** official “Welcome to PyTorch Tutorials” page; he will click into Learn the Basics.

### What he is establishing

This is a **coding** session: PyTorch so you can implement the theory later on **simple datasets**. Google Colab provides about a **12 GB GPU**; notebooks are written so you do not need more. Bigger models are an “if you have compute” extension.

Last tutorial: forward / backward / training loop. Today: PyTorch **basics**. TensorFlow exists; he will not use it.

Assumed: MLP from last time, and a **basic CNN**. If CNNs are rusty, he points at Stanford **CS231n** notes — neural nets, convolution, parameter sharing, features. That is homework if you freeze, not this hour’s content.

You can now name the kitchen (Colab + PyTorch). You have not created a tensor yet.

### Analogy for this topic only

A cooking class: last week knives (backprop). This week the **pots** (tensors). He assumes you know what a stove is (MLP/CNN). If the stove is fuzzy, he hands you the CS231n manual rather than re-teaching fire.

**If you skip CS231n and never saw a conv, will this hour teach CNNs?** No — he only assumes them.

### Local picture

```
  theory lectures (W1_L2 …)     coding track (this video)
           │                            │
           ▼                            ▼
     p_x, G_θ, d                  Colab 12GB + PyTorch
```

Notice: playlist title “F-divergence” does not match this board.

### Bridge

Where do the notebooks live, and whose machine runs the play button?

---

## Topic 2: Official tutorials and the Colab notebook (04:42–08:14)

### Where this sits on the master map

**ENV.** Search “pytorch tutorial” → pytorch.org. [Notebook cells](./PREREQUISITES.md#p4-colab).

### Board / screenshot

![Quickstart + Tensors notebook opened in Colab](./screenshots/composites/ch02-topic-02-official-tutorials-colab-panel1of1.png)
**Figure — ~04:47–08:14:** Learn the Basics nav; Tensors notebook with `import torch`.

### What he is establishing

They **do not write a custom course notebook from scratch**. They walk the **public official tutorial**, then slow down where students usually freeze.

Start with **Quickstart / Tensors**, not the whole site (recipes, C++, audio, …). Each page offers Colab, download, or GitHub. **This session uses Colab.**

`.ipynb` = interactive Python notebook. **Text cells** explain; **code cells** have a play button. First run is slow: Colab must attach a **Google server**. Your laptop is not doing the FLOPs. Packages are already installed.

Python is assumed. If lists and `import` are new, pause and learn Python first.

```python
# Cell 0 on screen — comments from the official notebook
# https://pytorch.org/tutorials/beginner/colab
%matplotlib inline
```

You can now open the same URL he opened. You still do not know what a tensor *is*.

### Analogy for this topic only

A lab that already stocked the reagents. **Do you pip-install PyTorch on your laptop for this hour?** He says no — Colab’s kitchen is stocked.

### Local picture

```
  browser  →  Colab ipynb  →  Google VM (CPU/GPU)
                 │
                 ▼
           code cell ▶  (first click: connect)
```

Notice: “device is not your local resources.”

### Bridge

Why not just use NumPy?

---

## Topic 3: What a tensor is (08:14–09:41)

### Where this sits on the master map

**TENSOR** object. [NumPy](./PREREQUISITES.md#p3-numpy), [device](./PREREQUISITES.md#p5-device).

### Board / screenshot

![Official Tensors page: ndarray + GPU + autograd](./screenshots/composites/ch03-topic-03-what-is-a-tensor-panel1of1.png)
**Figure — ~08:14–09:12:** “Tensors are a specialized data structure… similar to NumPy’s ndarrays, except that tensors can run on GPUs…”

### What he is establishing

Tensors are PyTorch’s **building blocks** — arrays/matrices with extra powers. Like `ndarray`, but they can run on a **GPU** (and other accelerators). That is why these libraries exist for deep learning.

On-screen official text (keep it): tensors hold **inputs, outputs, and model parameters**; they can **share memory** with NumPy; they will later support **autograd**.

Wrong move: treating a tensor as “just a Python list.” Right move: rectangular numbers **plus a device**.

### Analogy for this topic only

Graph paper (NumPy) vs graph paper you can slide onto a forklift (GPU). **Does the 2×2 of ones care which desk it sits on?** Same numbers; different speed.

### Local picture

```
  Python list     →  slow, not rectangular-enforced
  np.ndarray      →  fast CPU grid
  torch.Tensor    →  same grid + .device (cpu/cuda) + autograd later
```

Notice: GPU is the *point* of tensors; this demo still stays on CPU.

### Bridge

How do you *make* one?

---

## Topic 4: Tensor from a list and from NumPy (09:41–12:34)

### Where this sits on the master map

**CREATE from data.** [List](./PREREQUISITES.md#p1-list), [NumPy](./PREREQUISITES.md#p3-numpy).

### Board / screenshot

![data=[[1,2],[3,4]]; torch.tensor; torch.from_numpy](./screenshots/composites/ch04-topic-04-tensor-from-list-numpy-panel1of1.png)
**Figure — ~09:54–12:20:** `import torch` / `import numpy as np`; `x_data = torch.tensor(data)`; `x_np = torch.from_numpy(np_array)`.

### What he is establishing

Always `import torch`. Import NumPy only if you start from an array.

From a list (no extra library):

```python
import torch
import numpy as np

data = [[1, 2], [3, 4]]       # Python list; type(data) is list
x_data = torch.tensor(data)   # same numbers; now a Tensor
```

Values are **not** modified. Only the container type changes.

From NumPy:

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

First Colab **play** can stall while the VM connects — that is normal.

You can now wrap data you already have. You cannot yet spawn a 2×3 of zeros without a template.

### Analogy for this topic only

Pouring juice from a bottle (list) or a carton (NumPy) into the same glass (tensor). **Does the juice change?** No. The glass can go to the GPU later.

### Local picture

```
  list ──torch.tensor──► Tensor
  ndarray ──from_numpy──► Tensor
```

Notice: `type(data)` stays `list` after you also have `x_data`.

### Bridge

What if you do not have numbers yet — only a desired size?

---

## Topic 5: `ones_like` / `rand_like` (12:34–15:00)

### Where this sits on the master map

**CREATE from a template tensor.** [Shape](./PREREQUISITES.md#p2-shape).

### Board / screenshot

![ones_like and rand_like with dtype override](./screenshots/composites/ch05-topic-05-ones-rand-like-data-panel1of1.png)
**Figure — ~12:20–14:50:** `torch.ones_like(x_data)`; `torch.rand_like(x_data, dtype=torch.float)` “overrides the datatype of x_data.”

### What he is establishing

Two creation philosophies: (A) copy **structure** from existing data; (B) name a **shape**. This box is (A).

```python
x_ones = torch.ones_like(x_data)   # same shape (and properties), all ones
x_rand = torch.rand_like(x_data, dtype=torch.float)  # random; dtype override
```

If `x_data` was integer, `rand_like` without override would stay integer-ish; he **overrides** to `float`. Run `rand` twice → **different** numbers (sampling).

Wrong move: `torch.ones(x_data)` as if the tensor were a shape tuple. Right move: `ones_like`.

### Analogy for this topic only

A cookie cutter from an existing cookie. **Do you need to re-measure the cookie?** No — `*_like` stole the outline. Filling can still change the dough (`dtype`).

### Local picture

```
  x_data  ──ones_like──►  same shape, all 1
  x_data  ──rand_like──►  same shape, random (dtype optional)
```

Notice: properties copy **unless** you override.

### Bridge

Route (B): you only know `(2, 3)`.

---

## Topic 6: Create from a shape (15:00–16:29)

### Where this sits on the master map

**CREATE from a tuple.** [Shape](./PREREQUISITES.md#p2-shape).

### Board / screenshot

![shape=(2,3); rand / ones / zeros](./screenshots/composites/ch06-topic-06-create-from-shape-panel1of1.png)
**Figure — ~15:14–16:04:** `shape = (2, 3)` then `torch.rand(shape)`, `torch.ones(shape)`, `torch.zeros(shape)` with printed output.

### What he is establishing

```python
shape = (2, 3)
rand_tensor = torch.rand(shape)     # Uniform(0,1) noise, 2×3
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
```

First print is random; second all ones; third all zeros. Together with Topic 5 you have **both** doors into a given size: template tensor, or explicit shape.

### Analogy for this topic only

Ordering a blank hotel floor plan: “2 floors, 3 rooms.” **Do you pass a filled guest list?** No — only the shape. `rand` puts strangers in the rooms; `zeros` leaves them empty.

### Local picture

```
  (2, 3) ─┬─ rand   →  2×3 noise
          ├─ ones   →  2×3 ones
          └─ zeros  →  2×3 zeros
```

Notice: `shape` is a **tuple**, not a tensor.

### Bridge

Once it exists, what can you **ask** it?

---

## Topic 7: shape, dtype, device (16:29–17:54)

### Where this sits on the master map

**ATTRIBUTES.** [Device](./PREREQUISITES.md#p5-device).

### Board / screenshot

![print shape, dtype, device — cpu](./screenshots/composites/ch07-topic-07-shape-dtype-device-panel1of1.png)
**Figure — ~16:29–17:45:** `tensor = torch.rand(3, 4)` then `.shape`, `.dtype`, `.device`. Screen also shows the optional `.to(accelerator)` snippet he postpones.

### What he is establishing

```python
tensor = torch.rand(3, 4)
print(tensor.shape)    # torch.Size([3, 4])
print(tensor.dtype)    # torch.float32
print(tensor.device)   # cpu   ← this demo
```

CPU is always present. GPU is optional. He is **not** using GPU yet. Official warning: copying huge tensors between devices is expensive.

He **skips** slicing for the course (NumPy-identical if you already know it). On screen you still see `tensor[0]`, `tensor[:, 0]`, `tensor[..., -1]`, `tensor[:,1] = 0` — he does not dwell.

```python
# Official, postponed:
# if torch.accelerator.is_available():
#     tensor = tensor.to(torch.accelerator.current_accelerator())
```

Wrong move: reading `device: cpu` as failure. Right move: that is the default until you move it.

### Analogy for this topic only

A shipping label: **how big**, **what’s inside**, **which warehouse**. **If the warehouse says “desk” not “forklift,” is the box empty?** No — it just hasn’t moved.

### Local picture

```
  tensor
    .shape   how many along each axis
    .dtype   int vs float32 vs …
    .device  cpu | cuda | …
```

Notice: slicing skipped on purpose.

### Bridge

How do you glue tensors along an axis?

---

## Topic 8: Concatenate `dim=0` vs `dim=1` (17:54–20:43)

### Where this sits on the master map

**JOIN.** You must answer “beside or below?”

### Board / screenshot

![torch.cat dim=1 then dim=0](./screenshots/composites/ch08-topic-08-concatenate-panel1of1.png)
**Figure — ~18:07–20:29:** `t1 = torch.cat([tensor, tensor, tensor], dim=1)` then the same with `dim=0`. Official note: `torch.stack` is a cousin.

### What he is establishing

A tensor is a matrix. Concatenate **next to** (more columns) or **under** (more rows)? That is `dim`.

```python
# tensor here is 4×4 ones in his demo (torch.ones(4, 4))
t_cols = torch.cat([tensor, tensor, tensor], dim=1)
# 4 rows, 12 columns  (three 4×4 side by side)

t_rows = torch.cat([tensor, tensor, tensor], dim=0)
# 12 rows, 4 columns  (stacked)
```

`dim=1` → columns. `dim=0` → rows. `torch.stack` is mentioned on screen as a related join (adds a **new** axis rather than extending an existing one — he does not drill it).

Wrong move: calling `cat` and being surprised by 4×12 vs 12×4. Right move: name the dim first.

### Analogy for this topic only

Taping three photos. **Do you tape them in a strip or a stack?** Strip = dim 1 (wider). Stack = dim 0 (taller). Same photos.

### Local picture

```
  dim=1:  [ A | A | A ]     wider
  dim=0:  [ A ]
          [ A ]             taller
          [ A ]
```

Notice: all inputs to `cat` need matching sizes on the **other** axis.

### Bridge

Neural nets’ forward pass is not taping — it is **matrix multiply**.

---

## Topic 9: Matrix multiply three ways (20:43–25:02)

### Where this sits on the master map

**MATMUL.** [Rule](./PREREQUISITES.md#p6-matmul), [transpose](./PREREQUISITES.md#p7-transpose). He says the whole forward pass is this.

### Board / screenshot

![A(m×n) B(n×p)=C(m×p); @ / matmul / torch.matmul out=](./screenshots/composites/ch09-topic-09-matmul-panel1of1.png)
**Figure — ~20:43–25:04:** whiteboard shape rule; Colab three APIs with comment “y1, y2, y3 will have the same value.”

### What he is establishing

Matrix multiply is the workhorse of neural nets. A layer’s forward pass **is** a multiply (plus bias/activation later).

Shape law:

$$
A_{M\times N}\, B_{N\times P} = C_{M\times P}
$$

In words: **columns of A = rows of B**, else illegal.

`.T` = transpose. For a 4×4, `tensor.T` is still 4×4, so `tensor @ tensor.T` is legal.

Three equivalent APIs (official comments: same value):

```python
# 1) @  and  .T
y1 = tensor @ tensor.T

# 2) method on the left tensor
y2 = tensor.matmul(tensor.T)

# 3) torch.matmul with a preallocated output
y3 = torch.rand_like(y1)                 # only the shape/dtype
torch.matmul(tensor, tensor.T, out=y3)   # writes into y3
```

Wrong move: `@` with inner dims that do not match. Right move: check $N$, or multiply by `.T` when you mean “self Gram.”

You can now write a linear layer’s multiply three ways. Convolution still needs a **different** product.

### Analogy for this topic only

A factory: every incoming conveyor (N) must meet every outgoing (P) through M workers. **If N does not match, do the doors even meet?** No — the plant will not start. Three badges (`@`, `.matmul`, `torch.matmul`) open the **same** plant.

### Local picture

```
  A  M × N     ×     B  N × P     =     C  M × P
           └── must match ──┘

  tensor @ tensor.T
  tensor.matmul(tensor.T)
  torch.matmul(tensor, tensor.T, out=y3)
```

Notice: `out=` needs a tensor that already has the result shape.

### Bridge

Convolutions use a **same-size, entry-by-entry** product, not this plant.

---

## Topic 10: Element-wise product (25:02–28:43)

### Where this sits on the master map

**HADAMARD.** [Element-wise vs matmul](./PREREQUISITES.md#p8-hadamard). Last box of “the tensor idea.”

### Board / screenshot

![A,B same size; C_ij = A_ij B_ij; tensor * tensor and .mul](./screenshots/composites/ch10-topic-10-elementwise-panel1of1.png)
**Figure — ~25:19–28:25:** whiteboard “A & B should be of Same Size”; Colab `z1 = tensor * tensor`, `z2 = tensor.mul(tensor)`, `torch.mul(..., out=z3)`.

### What he is establishing

Element-wise (Hadamard) multiply is a **core convolution** operation. **A and B must be the same size.**

Board numbers (he writes 2×2):

```
  A = [ 1   2 ]     B = [ 4   2 ]      C = [ 1·4    2·2 ]  = [ 4   4 ]
      [-2   3 ]         [ 1  -1 ]          [-2·1    3·(-1)]    [-2  -3]
```

(One frame shows a slightly different B entry; the **rule** is pair matching entries.)

```python
z1 = tensor * tensor          # same shape, entry-wise
z2 = tensor.mul(tensor)       # same
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```

**Not** `matmul`. `mul` ≠ `matmul`.

Recap he wants you to leave with: create (rand/ones), attributes, **two products**. Addition/subtraction are “trivial” so the official page does not linger. **That completes the first idea: the tensor.**

Wrong move: `*` when the forward pass needs `@`. Right move: same-size → `*`; inner-dim handshake → `@`.

### Analogy for this topic only

Two paint-by-number sheets of the **same** picture. **Can you mix cell 1 of A with cell 7 of B?** No — only matching cells. That is convolution’s local multiply. The factory in Topic 9 mixed **every** aisle with **every** dock.

### Local picture

```
  * / mul     same shape     C_ij = A_ij × B_ij     (conv guts)
  @ / matmul  inner dims     C_ik = Σ_j A_ij B_jk   (linear layer)
```

Notice: he ends the module here. Datasets, autograd, `nn.Module` are later official chapters.

### Bridge

Next coding tutorials (playlist T3+) load data and build models. Next **math** lecture on the playlist is VDM / f-divergence if the titles are to be believed — this file was not that lecture.

---

## External references

Grouped **by this recording’s topics** (video + docs/blog). All opened. This is a **code** hour, so official PyTorch docs beat chalk notes.

### Topic 1 — setup / CNN prereq

| Kind | Resource | Why |
|------|----------|-----|
| Notes | [CS231n Convolutional Networks](https://cs231n.github.io/convolutional-networks/) | The CNN manual he actually names |
| Notes | [CS231n Neural Networks 1](https://cs231n.github.io/neural-networks-1/) | MLP structure he assumes |
| Video | [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) | $28\times 28$ as a tensor of 784 numbers |

### Topic 2 — official tutorials / Colab

| Kind | Resource | Why |
|------|----------|-----|
| Docs | [PyTorch tutorials index](https://pytorch.org/tutorials/) | Exact site he googles |
| Docs | [Tensors tutorial (Colab)](https://pytorch.org/tutorials/beginner/basics/tensor_tutorial.html) | The notebook on screen |
| Docs | [Colab helper he comments](https://pytorch.org/tutorials/beginner/colab) | First cell comment URL |

### Topics 3–4 — tensor vs ndarray; create from data

| Kind | Resource | Why |
|------|----------|-----|
| Docs | [PyTorch tensor tutorial — initializing](https://pytorch.org/tutorials/beginner/basics/tensor_tutorial.html) | `torch.tensor` / `from_numpy` |
| Blog | [PyTorch tensor vs NumPy](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html) | GPU + autograd extras |
| Tutorial | [CS231n NumPy tutorial](https://cs231n.github.io/python-numpy-tutorial/) | Arrays before tensors |

### Topics 5–7 — `*_like`, shape, attributes

| Kind | Resource | Why |
|------|----------|-----|
| Docs | [torch.ones_like](https://pytorch.org/docs/stable/generated/torch.ones_like.html) | Template copy + dtype override |
| Docs | [Tensor attributes](https://pytorch.org/docs/stable/tensor_attributes.html) | `.shape` `.dtype` `.device` |
| Docs | [torch.rand](https://pytorch.org/docs/stable/generated/torch.rand.html) | Shape-tuple creation |

### Topic 8 — concatenate

| Kind | Resource | Why |
|------|----------|-----|
| Docs | [torch.cat](https://pytorch.org/docs/stable/generated/torch.cat.html) | `dim=` is the whole lesson |
| Docs | [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html) | Cousin named on screen |
| Video | [Aladdin Persson — PyTorch tutorial tensors](https://www.youtube.com/watch?v=x9JiIFvlUwk) | Same APIs, extra examples |

### Topics 9–10 — `@` vs `*`

| Kind | Resource | Why |
|------|----------|-----|
| Docs | [torch.matmul](https://pytorch.org/docs/stable/generated/torch.matmul.html) | The three-API family |
| Docs | [torch.mul](https://pytorch.org/docs/stable/generated/torch.mul.html) | Element-wise; same size |
| Video | [3Blue1Brown — matrix multiply as composition](https://www.youtube.com/watch?v=XkY2DOUCWMU) | Why inner dims match |
| Notes | [CS231n conv notes (element-wise in windows)](https://cs231n.github.io/convolutional-networks/) | Why he says Hadamard shows up in conv |

**If you came for $f$-divergence:** [Stanford CS236 GAN / $f$-GAN notes](https://deepgenerativemodels.github.io/notes/gan/) · [NPTEL Lec 03 package](../../Mathematical-Foundation-for-GenerativeAI/25-Lec03-f-Divergence-Examples/NOTES.md) · playlist [W1_L4 VDM](https://www.youtube.com/watch?v=nfZQYopzv20).

---

## Apply it (scenarios)

*Workplace-style situations that use **this video’s** APIs only.*

### Scenario 1: Colab vs laptop GPU

**Context:** A teammate clones the official tensor notebook and runs it on a laptop without NVIDIA drivers.  
**Challenge:** First cell takes 40 s, then `tensor.device` prints `cpu`. They think Colab is “broken.”  
**Questions:**  
1. Whose machine is running the cell if they had used **Open in Colab** as he did?  
2. Is `cpu` a failure in this hour’s demo?

<details>
<summary>Show solution sketch</summary>

- Topic 2: Colab = Google VM; first play **connects**. Topic 7: default device is **cpu** until `.to(accelerator)`.
- Steps: use Colab if you want his 12 GB GPU story; on a CPU laptop, tensors still work — just slower.

</details>

### Scenario 2: Wrong product in a “linear layer”

**Context:** Someone writes `h = x * W` for a 128-dim hidden layer (`x` is 32×128, `W` is 128×64).  
**Challenge:** Shape error or silent wrong 32×128 if they broadcast.  
**Questions:**  
1. Which product did this lecture say a forward pass **is**?  
2. Write the legal `@` using `.T` if needed.

<details>
<summary>Show solution sketch</summary>

- Topic 9: forward pass ≈ **matmul**, inner dims match: `(32×128) @ (128×64) → (32×64)`.
- `*` is Topic 10 (same shape / conv). Use `x @ W`, not `x * W`.

</details>

### Scenario 3: Concatenate batch vs features

**Context:** Three 4×4 feature maps. Product wants them **side by side** as 4×12.  
**Challenge:** Intern used `dim=0` and shipped 12×4.  
**Questions:**  
1. Which `dim` is “more columns”?  
2. What cousin of `cat` did the official page name?

<details>
<summary>Show solution sketch</summary>

- Topic 8: `torch.cat([A,A,A], dim=1)` → 4×12. `dim=0` → 12×4.
- `torch.stack` adds a **new** axis (not drilled, but named on screen).

</details>

### Scenario 4: Build a zeros buffer like an existing batch

**Context:** You have `x_data` (integer 2×2) and need a **float** noise tensor of the **same shape**.  
**Challenge:** `torch.rand(x_data)` is the wrong call.  
**Questions:**  
1. Template API vs shape-tuple API?  
2. How do you override dtype?

<details>
<summary>Show solution sketch</summary>

- Topic 5: `torch.rand_like(x_data, dtype=torch.float)`.
- Topic 6 alternative: `torch.rand(x_data.shape)`.

</details>

---

## Sources

- Video: [rHnrALMCyIQ](https://www.youtube.com/watch?v=rHnrALMCyIQ&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=3) · IIT Madras BS · ~28:43 · **audio+screen = PyTorch tensors**
- Official notebook shown: [tensor_tutorial.html](https://pytorch.org/tutorials/beginner/basics/tensor_tutorial.html)
- Timed captions: `raw/captions.en.timed.txt` (ASR matches this coding session)
- Boards: `screenshots/composites/ch01-…`–`ch10-…` (unique per topic)
