# Tutorial 3 — PyTorch Basics

**Video:** [Tutorial 3 : PyTorch Basics](https://www.youtube.com/watch?v=SEtu7Eef5ps) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Tutorial 2 — NumPy](../16-Tutorial02-Introduction-to-NumPy/NOTES.md)  
**Course:** Mathematical Foundations of **Generative AI** (~62 min)  
**Speaker:** NPTEL IISc · Tensors, AutoGrad, Datasets and DataLoaders, MLP

---

## Table of Contents

1. [Topic 1 — Imports version device](#topic-1-imports-version-device-0004–0430) (00:04–04:30)
2. [Topic 2 — Tensor basics index](#topic-2-tensor-basics-index-0430–1000) (04:30–10:00)
3. [Topic 3 — Reshape unsqueeze squeeze permute](#topic-3-reshape-unsqueeze-squeeze-permute-1000–1600) (10:00–16:00)
4. [Topic 4 — Matmul autograd intro](#topic-4-matmul-autograd-intro-1600–2200) (16:00–22:00)
5. [Topic 5 — Multi-var grad manual linreg](#topic-5-multi-var-grad-manual-linreg-2200–2830) (22:00–28:30)
6. [Topic 6 — nn.Module Linear optim](#topic-6-nnmodule-linear-optim-2830–3600) (28:30–36:00)
7. [Topic 7 — Activations CrossEntropy](#topic-7-activations-crossentropy-3600–4200) (36:00–42:00)
8. [Topic 8 — Custom Dataset](#topic-8-custom-dataset-4200–4800) (42:00–48:00)
9. [Topic 9 — DataLoader shuffle](#topic-9-dataloader-shuffle-4800–5230) (48:00–52:30)
10. [Topic 10 — MLP train accuracy](#topic-10-mlp-train-accuracy-5230–6208) (52:30–62:08)
11. [Apply it (scenarios)](#apply-it-scenarios)
12. [External references](#external-references)
13. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

**Job:** move from NumPy arrays to a **GPU-capable training stack** — tensors, autograd, modules, data pipeline, and a full MLP loop.  
**Method:** same shape language as Tutorial 2, plus `device`, `requires_grad` / `backward`, `nn.Module` + `optim`, `Dataset` / `DataLoader`, then CE + accuracy.  
**Fork:** own the **standard train step** (`zero_grad → forward → loss → backward → step`) so later CNN/RNN lectures only change the architecture, not the plumbing.

**Worldview arc:** from NumPy-style numeric arrays **to** “PyTorch = tensors on a device + automatic gradients + modules + data loaders + train loop.”

### System context

```
  ╔══════════════════════════════════════╗
  ║ Outside: production CNN / RNN / LLM  ║
  ║ Outside: multi-GPU, distributed train║
  ╚══════════════╤═══════════════════════╝
                 │ this tutorial (~62 min)
                 ▼
        ┌────────────────────────────┐
        │ PyTorch basics stack       │
        │ tensor · autograd · Module │
        │ Dataset · DataLoader · MLP │
        └────────────────────────────┘
                 │
                 ▼
        next unit: CNNs / RNNs (same train pattern)
```

### Main blueprint

```
  NumPy ndarray (Tutorial 2)
          │
          │  import torch · nn · optim · F
          ▼
  device = cuda if available else cpu
          │
          ▼
  torch.Tensor  +  .shape / .dtype / .device
          │
    ┌─────┼──────────┬──────────┐
    ▼     ▼          ▼          ▼
  create index   reshape    permute
  zeros  slice   unsqueeze  (HWC→CHW)
  ones           squeeze
  rand / randn / eye / arange
          │
          ▼
  matmul  @     ·  .to(device)
          │
          ▼
  requires_grad=True  →  y.backward()  →  .grad
          │
          ▼
  manual GD: ŷ=wx+b · MSE · no_grad update · grad.zero_()
          │
          ▼
  nn.Module + nn.Linear + criterion + optim
  (predict → loss → zero_grad → backward → step)
          │
          ▼
  F.relu / sigmoid / tanh · softmax · CrossEntropyLoss(logits)
          │
          ▼
  Dataset (__init__/__len__/__getitem__)
          │
          ▼
  DataLoader(batch_size, shuffle=True)
          │
          ▼
  MLP: FC→ReLU→FC→ReLU→FC  +  CE + Adam  +  argmax accuracy
```

### Scenario walkthrough

1. Import `torch`, `nn`, `optim`, `F`; print version; pick `device` (CUDA if available).
2. Build tensors from lists and factories; inspect `.shape`, `.dtype`, `.device`; index like NumPy.
3. Reshape / view; unsqueeze / squeeze; permute HWC → CHW for ImageNet-style images.
4. Matmul with `@`; move tensors with `.to(device)`; first scalar autograd (`x² + 3x + 1`).
5. Multi-variable gradient of mean-of-squares; full manual linear regression loop.
6. Rewrite linreg as `nn.Module` + `nn.Linear` + `MSELoss` + `SGD`.
7. Multi-feature Linear shapes; activations; softmax; **CrossEntropy wants logits**.
8. Custom `Dataset` with three required methods; wrap with `DataLoader`.
9. Understand `shuffle=True`; design a 3-layer MLP skeleton.
10. Train MLP on toy binary data: CE + Adam, device-aware batches, loss + accuracy.

### Failure / contrast path

```
  Leave tensors on CPU while model is on CUDA     ──X──► device mismatch crash
  Feed softmax probs into CrossEntropyLoss        ──X──► double-softmax, bad grads
  Skip optimizer.zero_grad() / .grad.zero_()      ──X──► gradient accumulation
  Update params without torch.no_grad() (manual)  ──X──► pollutes the graph
  view() on non-contiguous tensor                 ──X──► RuntimeError (prefer reshape)
  Forget last partial batch is smaller            ──X──► surprise shapes
  Shuffle only features, not shared indices       ──X──► label desync (Dataset fixes this)
  Softmax on wrong dim for (B, C) logits          ──X──► nonsense probabilities
```

### STOP / out of scope

CNN and RNN architectures (cliffhanger for next segment); multi-GPU patterns beyond “one device for everything”; custom loss engineering; production torchvision pipelines; distributed training; full hyperparameter search.

### Load-bearing claims (closed-book)

- PyTorch is NumPy’s shape language **plus device, autograd, modules, and data pipeline**.
- Always know your **`device`** before training; reuse it for `.to(device)` on tensors and models.
- New tensors default to **CPU** until you move them; all operands in an op must share device.
- **`requires_grad=True` → `backward()` → `.grad`** is the gradient workhorse.
- Gradients **accumulate**; clear them each step (`zero_()` / `optimizer.zero_grad()`).
- Every model is a **child of `nn.Module`** with `super().__init__()`, layers, and `forward`.
- Standard train step: **predict → loss → zero_grad → backward → step**.
- **`nn.CrossEntropyLoss` wants raw logits**, not probabilities; targets are **integer class indices**.
- **`Dataset`** returns one sample; **`DataLoader`** batches (and can shuffle each epoch).
- Accuracy for classification: **`argmax(logits, dim=1)`** vs labels; count correct / total.

**Speaker / course:** NPTEL IISc · Mathematical Foundations of Generative AI · Tutorial 3.

---

## Topic 1: Imports version device (00:04–04:30)

### Where this sits on the master map

**TORCH + DEVICE** — After NumPy arrays (Tutorial 2), move to **PyTorch**: base `torch`, `nn`, `optim`, `F`. Always pick a **device** (CUDA GPU if available, else CPU). Tensors are the GPU-capable cousins of NumPy arrays. Warm-up: [tensor](./PREREQUISITES.md#p1-tensor) · [device](./PREREQUISITES.md#p2-device).

### Board / screenshot

![Imports version device](./screenshots/composites/ch01-topic-01-imports-device-panel1of1.png)

**Figure — ~00:04–04:30:** Canonical imports; `torch.__version__`; device selection CUDA vs CPU; GPU name when available.

### What he is establishing

Prior tutorials covered basic **Python** constructs and implementing regression / CNN-style ideas with **NumPy**. The next library is **PyTorch**. The bootcamp roadmap is: (1) basic PyTorch functionalities, (2) linear models / MLPs, (3) CNNs, RNNs, LSTMs, (4) loading **pre-trained** models and adapting them for **downstream tasks**. This first tutorial focuses on **basics** — setup, tensors, and the autograd path before full `nn.Module` fluency.

Core import: **`import torch`** is the base library for tensors and low-level ops. Neural-network building blocks live under **`torch.nn`**, conventionally `import torch.nn as nn`. Optimizers come from **`torch.optim`**, conventionally `import torch.optim as optim`. Functional activation and loss helpers live under **`torch.nn.functional`**, conventionally `import torch.nn.functional as F` (e.g. `F.relu`).

```python
# ============================================================
# PyTorch Bootcamp for Advanced Deep Learning
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
```

That is the canonical starter import block for this notebook: base library, modules, optimizers, and functional helpers under short aliases.

Check the installed version with **`torch.__version__`**. The demo environment reports something like **2.11.0** with a CUDA build tag (e.g. `2.11.0+cu128`).

```python
print("PyTorch version:", torch.__version__)
# e.g. PyTorch version: 2.11.0+cu128
```

This prints which PyTorch build you are running so notebooks and bugs can be compared across machines.

Motivation slogan: plain **lists** and **NumPy arrays** are designed for **CPU**. Deep learning needs data structures that run on **GPUs / accelerators** — PyTorch tensors plus explicit device placement. Free GPU option: **Google Colab** — Runtime → Change runtime type → hardware accelerator **T4 GPU** (or TPU); CPU is the default free tier. Prefer **T4 GPU** for most class work when available; save the runtime setting after switching CPU → GPU. Workflow tip for later: you can write and sanity-run on CPU first, then enable GPU to conserve free quota.

Device selection pattern: build a **`torch.device(...)`** that is **`"cuda"`** if `torch.cuda.is_available()` else **`"cpu"`**.

```python
# ============================================================
# Check GPU Availability
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

That one-liner picks CUDA when a GPU is usable and otherwise falls back to CPU, then prints the choice.

If CUDA is available, print the GPU name with **`torch.cuda.get_device_name(0)`**. Demo machine: **Tesla T4**, device string `cuda`.

```python
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
# Using device: cuda
# GPU name: Tesla T4
```

This only runs the name query when a CUDA device exists, avoiding crashes on CPU-only machines.

Procedure slogan: **always know your device** before training — the `device` variable is reused later for `.to(device)` on tensors and models.

You can now open a notebook, import the four packages, print the version, and fix a `device` for the rest of the session. Still missing: actual tensors with shape, dtype, and indexing.

A common trap is assuming Colab always has a GPU on — the free tier defaults to CPU until you change the runtime type and re-check `torch.cuda.is_available()`. A second trap is hard-coding `"cuda"` in every cell: a CPU-only laptop crashes, and a multi-GPU box may put model and data on different cards. Store **`device` once** and reuse `.to(device)`.

### Analogy for this topic only

NumPy is a **home kitchen** (CPU only). PyTorch is the same recipes with an optional **industrial oven** (CUDA). The `device` line is deciding **which kitchen every pot will sit in before you start cooking** — not after the food is half-done. Imports are the knives and pans you lay out first so the rest of the notebook does not rummage through drawers mid-recipe.

Question: **Why store `device` once instead of hard-coding `"cuda"` everywhere?**

In lecture words: this box is the setup spine every later cell reuses.

### Local picture

```
  import torch
  import torch.nn as nn
  import torch.optim as optim
  import torch.nn.functional as F
          │
          ▼
  torch.__version__   →  e.g. 2.11.0+cu128
          │
          ▼
  device = "cuda" if cuda available else "cpu"
          │
          ├── print device
          └── if cuda: get_device_name(0)  →  Tesla T4 (demo)
```

**Notice:** `device` is a reusable object, not a one-time print.

### Bridge

With imports and a device fixed, the next skill is creating **tensors**, inspecting attributes, and indexing them the way you indexed NumPy arrays.

---

## Topic 2: Tensor basics index (04:30–10:00)

### Where this sits on the master map

**TENSOR = array + device** — Create tensors from lists / factories (`zeros`, `ones`, `rand`, `randn`, `eye`). Inspect **shape / dtype / device**. Default device is **CPU** until you move them. Indexing/slicing mirrors NumPy. Warm-up: [tensor](./PREREQUISITES.md#p1-tensor) · [device](./PREREQUISITES.md#p2-device).

### Board / screenshot

![Tensor basics index](./screenshots/composites/ch02-topic-02-tensor-basics-panel1of1.png)

**Figure — ~04:30–10:00:** `torch.tensor` / factories; attributes; 1D/2D indexing; `arange` + reshape rule.

### What he is establishing

Colab tip: coding in Colab does **not** mean the GPU must always be on. Workflow: write code and **sanity-run on CPU**, then enable GPU and re-run to conserve free GPU quota.

Create a tensor from a Python list with **`torch.tensor(...)`** — analogous to `np.array` in the NumPy tutorial. Float tensors use float literals so the dtype becomes floating point.

```python
# ============================================================
# Creating Tensors
# ============================================================
# Tensor from list
x = torch.tensor([1, 2, 3, 4])

# Float tensor
y = torch.tensor([1.0, 2.0, 3.0])
```

The first line builds an integer-ish 1D tensor; the second builds floats suitable for neural-net math.

Sample a random tensor from the **standard normal** with **`torch.randn(3, 4)`** — shape is specified as separate ints (here 3×4). Core attributes to inspect: **`.shape`**, **`.dtype`**, **`.device`**.

```python
# ============================================================
# Tensor Attributes
# ============================================================
x = torch.randn(3, 4)
print("Tensor:")
print(x)
print("Shape:", x.shape)
print("Data type:", x.dtype)
print("Device:", x.device)
# Shape: torch.Size([3, 4])
# Data type: torch.float32
# Device: cpu
```

This creates a 3×4 random normal tensor and prints the three attributes you should always check. Slogan: newly created tensors default to **CPU**. You must **explicitly port** them to the GPU later (`.to(device)`).

Shape axes are indexable: `x.shape[0]` = number of **rows**, `x.shape[1]` = number of **columns** (for 2D).

```python
print("Rows:", x.shape[0], "Cols:", x.shape[1])  # 3, 4
```

That peels the `torch.Size` tuple into human “rows vs columns” language.

Need **special tensors** by design: all zeros, all ones, samples from a distribution, or an **identity** matrix — common for init and bookkeeping.

```python
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 5)
print("Zeros:")
print(zeros)
print("Ones:")
print(ones)
```

`zeros` is a 3×4 zero matrix; `ones` is a 2×5 matrix of ones — both useful for init and masks.

```python
random_uniform = torch.rand(3, 3)
random_normal = torch.randn(3, 3)
print("Random uniform:")
print(random_uniform)
print("Random normal:")
print(random_normal)
```

**`torch.rand`** samples each entry from **Uniform(0, 1)**; **`torch.randn`** samples from the **standard normal**.

```python
identity = torch.eye(4)
print("Identity matrix:")
print(identity)
```

**`torch.eye(4)`** builds a **4×4 identity**. Identity is defined only for **square** matrices — one size argument is enough.

Factory summary covering the common “make a tensor of known kind” cases:

```python
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 5)
random_uniform = torch.rand(3, 3)
random_normal = torch.randn(3, 3)
identity = torch.eye(4)
```

Five factories in one place: zeros, ones, uniform, normal, identity.

**Indexing** on tensors matches **lists / NumPy arrays**: 0-based index, negative end, slicing. 1D demo:

```python
x = torch.tensor([10, 20, 30, 40, 50])
print(x[0])    # 10
print(x[-1])   # 50
print(x[:3])   # tensor([10, 20, 30])
print(x[2:])   # tensor([30, 40, 50])
```

First, last, prefix, and suffix slices on a length-5 vector work exactly as in Python.

2D indexing: row then column, 0-based. Full row `matrix[0]`; column `matrix[:, 1]`.

```python
matrix = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90],
])
print(matrix[0, 1])   # 20
print(matrix[0])      # first row
print(matrix[:, 1])   # second column: tensor([20, 50, 80])
```

Element (0,1) is 20; the first row is `[10,20,30]`; the second column is `[20,50,80]`.

Submatrix / block slice: `matrix[:2, :2]` → top-left 2×2. Same idea as NumPy; the container is a **tensor**, not an `ndarray`.

```python
print("Top-left 2x2 block:")
print(matrix[:2, :2])
# tensor([[10, 20],
#         [40, 50]])
```

That grabs rows 0–1 and columns 0–1 as a contiguous block.

Motivation for **reshape**: change layout of the same elements — e.g. CNN features **flatten** into a classification head, or an MLP vector **reshape** into a matrix (image-like) view. Index factory: **`torch.arange(12)`** yields integers **0…11**. Original shape is length-12.

```python
# ============================================================
# Tensor Reshaping
# ============================================================
x = torch.arange(12)
print("Original tensor:")
print(x)  # tensor([0, 1, 2, ..., 11])
print("Original shape:", x.shape)  # torch.Size([12])
```

`arange` builds a clean integer sequence ideal for reshape demos.

Reshape rule: total element count must be **compatible**. `x.reshape(3, 4)` is valid (12 = 3×4); **`reshape(3, 5)` is invalid**.

```python
x_reshaped = x.reshape(3, 4)
print("Reshaped tensor:")
print(x_reshaped)
print("Reshaped shape:", x_reshaped.shape)  # torch.Size([3, 4])
# x.reshape(3, 5)  # RuntimeError: shape is invalid for input of size 12
```

Compatible reshape succeeds; incompatible product of dimensions raises a runtime error.

You can now create tensors, read their three attributes, index like NumPy, and reshape when the element count matches. Still missing: safer shape tools (`view` nuance), unsqueeze/squeeze, and permute for image layouts.

A common trap is assuming new tensors are already on the GPU — they are not; `.device` will say `cpu` until you call `.to(device)`. Another trap: integer `torch.tensor([1, 2, 3])` for model math — prefer floats (`1.0`) so dtypes match `float32` layers and losses.

### Analogy for this topic only

A tensor is a **shipping pallet with a destination tag**. The numbers are the cargo (like NumPy). The tag is **device**. Indexing opens one box or a row of boxes; factories (`zeros`, `ones`, `rand`, `randn`, `eye`) are pre-packed standard crates the warehouse already stocks; reshape is re-stacking the same cargo on a different pallet floor plan without inventing new crates.

Question: **What three attributes should you print after every important tensor create?**

In lecture words: this box is the tensor fluency you need before shape gymnastics.

### Local picture

```
  list ──torch.tensor──► Tensor
                              │
                              ├── .shape   torch.Size([...])
                              ├── .dtype   float32 (common)
                              └── .device  cpu (default)

  factories:
    zeros(r,c)  ones(r,c)  rand  randn  eye(n)  arange(n)

  index (like NumPy):
    x[0]  x[-1]  x[:3]
    matrix[r, c]  matrix[:, j]  matrix[:2, :2]

  reshape: product of dims must match element count
```

**Notice:** `torch.Size` behaves like a tuple for indexing (`shape[0]`, `shape[1]`).

### Bridge

Reshape is only the start of shape plumbing — next come **view**, **unsqueeze / squeeze**, and **permute** for CNN-style axis order.

---

## Topic 3: Reshape unsqueeze squeeze permute (10:00–16:00)

### Where this sits on the master map

**SHAPE GYMNASTICS** — `reshape`/`view` re-lay elements (row-wise); `unsqueeze`/`squeeze` add/remove size-1 axes; `permute` reorders axes (NHWC ↔ NCHW for ImageNet 224×224×3). Critical for CNN/RNN pipelines. Warm-up: [shape ops](./PREREQUISITES.md#p3-shapeops).

### Board / screenshot

![Reshape unsqueeze squeeze permute](./screenshots/composites/ch03-topic-03-reshape-permute-panel1of1.png)

**Figure — ~10:00–16:00:** Row-wise reshape; view vs reshape; unsqueeze/squeeze; permute `(224,224,3)` → `(3,224,224)`.

### What he is establishing

After a valid reshape to **3×4**, fill order is **row-wise**: first four elements `0,1,2,3` become row 0, next four row 1, last four row 2. Slogan: reshape converts a **length-12 arrangement into a 3×4 matrix** without inventing new values — same data, new view of axes.

```python
x = torch.arange(12)
x_reshaped = x.reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
# shape: torch.Size([3, 4])
```

Row-wise fill lays sequential integers into matrix form without changing any value.

Alternative to reshape: **`.view(...)`**. Both change tensor shape; you can usually use either for simple contiguous cases. Memory nuance: **`view` requires contiguous memory**; **`reshape` can work when memory is non-contiguous**. Prefer **`reshape` as generally safer**; both often look the same in demos.

```python
# ============================================================
# view() and reshape()
# Both are used to change tensor shape
# reshape() is generally safer
# ============================================================
x = torch.arange(12)
x_view = x.view(3, 4)
# x.reshape(3, 4)  # equivalent for this contiguous case
```

On a fresh contiguous `arange` tensor, `view(3, 4)` matches reshape; prefer reshape when unsure about contiguity.

Beyond flattening matrices: sometimes you need to **add** a dimension (e.g. turn length-12 into **1×12** or **1×1×12**) for CNN/RNN shape plumbing. Definitions: **`unsqueeze`** **adds** a dimension of size 1; **`squeeze`** **removes** dimensions of size 1.

```python
# ============================================================
# unsqueeze() adds a dimension
# squeeze() removes dimensions of size 1
# ============================================================
x = torch.tensor([1, 2, 3])
print("Original shape:", x.shape)                  # torch.Size([3])
x_unsqueezed = x.unsqueeze(0)
print("After unsqueeze(0):", x_unsqueezed.shape)   # torch.Size([1, 3])
x_unsqueezed_2 = x.unsqueeze(1)
print("After unsqueeze(1):", x_unsqueezed_2.shape) # torch.Size([3, 1])
x_squeezed = x_unsqueezed.squeeze()
print("After squeeze:", x_squeezed.shape)          # torch.Size([3])
```

`unsqueeze(0)` makes a row-like `[1, 3]`; `unsqueeze(1)` makes a column-like `[3, 1]`; `squeeze` removes size-1 axes and restores `[3]`. Axis counting is always **0-based**.

Another core skill: **change the order of dimensions** without changing content — **`permute`**. Image convention clash: people often speak of images as **height × width × channels** (**H×W×C / NHWC**), but many PyTorch CNN layers expect **channels first** (**C×H×W / NCHW**). Concrete ImageNet-style example: shape **224 × 224 × 3** may need to become **3 × 224 × 224**. Whiteboard map: axes `0=H, 1=W, 2=C` for `(224, 224, 3)`; target order puts old axis **2 first**, then **0**, then **1** → permute **`(2, 0, 1)`**.

```python
# ============================================================
# permute() changes the order of dimensions
# ============================================================
x = torch.randn(224, 224, 3)   # H, W, C  (ImageNet-style spatial size)
print("Before:", x.shape)     # torch.Size([224, 224, 3])
x_perm = x.permute(2, 0, 1)   # C, H, W
print("After:", x_perm.shape) # torch.Size([3, 224, 224])
```

That reorders axes from HWC to CHW without rewriting pixel values. Slogan: **permute = reorder axes**, not reshape the element count — content layout semantics change by axis order only.

Elementwise **+ − × ÷** and **exponentiation** on tensors mirror **NumPy** behavior (same mental model as Tutorial 2).

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(a + b)       # elementwise add
print(a * b)       # elementwise mul
print(a ** 2)      # elementwise square
```

Same-shape tensors combine entrywise; `*` is still not matrix multiplication. Next highlight operation: **matrix multiplication** — central linear-algebra op for neural nets (continued in Topic 4 with `@`).

You can now reshape safely, add/remove singleton axes, and convert HWC images to CHW. Still missing: matmul, device moves for training, and autograd.

A common trap is confusing **permute** (reorder existing axes) with **reshape** (reinterpret a flat buffer under a new shape product) — wrong tool, wrong layout semantics.

### Analogy for this topic only

Reshape is **repacking the same clothes** into a different suitcase grid. Unsqueeze adds an empty labeled pocket (batch dim). Squeeze removes empty pockets. Permute is rotating the suitcase so the “color” face is on top for the CNN conveyor belt (CHW), not inventing new fabric.

Question: **Why does `permute(2, 0, 1)` turn `(224, 224, 3)` into `(3, 224, 224)`?**

In lecture words: this box is the shape gymnastic toolkit for every vision pipeline.

### Local picture

```
  arange(12) ──reshape/view──► 3×4 (row-wise)
                    │
                    ├── view: needs contiguous memory
                    └── reshape: safer when non-contiguous

  [1,2,3]  unsqueeze(0) → [1,3]
           unsqueeze(1) → [3,1]
           squeeze()    → [3]

  (224, 224, 3)  permute(2,0,1)  →  (3, 224, 224)
       H   W  C                         C   H    W

  elementwise: a+b, a*b, a**2   (NumPy-like)
```

**Notice:** prefer `reshape` when contiguity is unclear.

### Bridge

With shape tools ready, next is **`@` matmul**, moving tensors onto **`device`**, and the first **`requires_grad` / `backward`** examples.

---

## Topic 4: Matmul autograd intro (16:00–22:00)

### Where this sits on the master map

**@ + AUTOGRAD** — Matrix multiply with `@`. Move tensors with `.to(device)`; all operands in an op must share device. Enable tracking with **`requires_grad=True`**, then **`y.backward()`** to fill **`.grad`** (workhorse for gradient descent). Warm-up: [device](./PREREQUISITES.md#p2-device) · [autograd](./PREREQUISITES.md#p4-autograd).

### Board / screenshot

![Matmul autograd intro](./screenshots/composites/ch04-topic-04-matmul-autograd-panel1of1.png)

**Figure — ~16:00–22:00:** `A @ B`; `.to(device)`; scalar `y = x²+3x+1` → `backward`; vector mean loss setup.

### What he is establishing

Matrix multiplication for compatible 2×2 tensors uses the **`@`** operator (same idea as NumPy).

```python
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0],
                  [7.0, 8.0]])
C = A @ B
print("A @ B:")
print(C)
# tensor([[19., 22.],
#         [43., 50.]])
```

True matrix product: each entry is a row–column dot product, not elementwise multiply.

Slogan: tensors are created on **CPU** by default; training on GPU requires **moving** them (and the model) onto the device. Device-consistency rule: for any op (e.g. forward pass with X, Y, and a model), **all involved tensors must be on the same device**. Partial GPU residency is not allowed. Multi-GPU caution: if the system has several cards, model, X, and Y must live on the **same** GPU card unless you use advanced multi-device patterns. Default teaching rule: **one device for everything**.

```python
# ============================================================
# Moving Tensor to GPU
# ============================================================
x = torch.randn(3, 4)
print("Before moving:", x.device)   # cpu
x = x.to(device)
print("After moving:", x.device)    # cuda:0  (if CUDA available)
```

`.to(device)` ports the tensor; before: `cpu`; after (when CUDA available): `cuda:0`.

Core ML workhorse is **gradient descent** → you must understand how **gradients are computed** in PyTorch (**autograd**). Enable gradient tracking by creating a tensor with **`requires_grad=True`**. Whiteboard scalar example: **x = 2**, **y = x² + 3x + 1**. Analytic derivative **dy/dx = 2x + 3**; at x=2 → **7**. Call **`y.backward()`** to reverse-mode differentiate; then read **`x.grad`**.

```python
# ============================================================
# Autograd
# PyTorch automatically calculates gradients
# ============================================================
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3 * x + 1
y.backward()
print("dy/dx at x=2:", x.grad)  # tensor(7.)
```

Autograd builds the graph for the polynomial, walks it backward, and stores ∂y/∂x = 7 on `x.grad`. Gradients live on the **same device** as the tensors that produced them. In this demo nothing was moved — **CPU** throughout.

Extend scalar autograd to a **vector**. Let **x = [1, 2, 3]** with `requires_grad=True`. Elementwise square: **y = x²** → content **[1, 4, 9]**. Reduce to a **scalar loss** before backward: **`loss = y.mean()`** = (1+4+9)/3 = **14/3 ≈ 4.667**. `backward()` requires a scalar output (or an explicit gradient vector).

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x**2
# y content: [1, 4, 9]
loss = y.mean()
print(loss)  # ~ tensor(4.6667, grad_fn=<MeanBackward0>)
# continued in Topic 5: loss.backward(); print(x.grad)
```

Vector square plus mean yields a scalar loss ready for multi-variable gradients next.

You can now matmul, move tensors to the shared device, and trust scalar `backward` against an analytic derivative. Still missing: multi-variable ∂loss/∂x and a full **manual training loop**.

A common trap is leaving one tensor on CPU and another on CUDA for the same `@` or forward — PyTorch errors rather than silently mixing devices.

### Analogy for this topic only

`@` is the **mixer** of neural nets. Device is **which kitchen** the mixer runs in. Autograd is a **receipt tape**: every operation prints a line; `backward()` walks the receipt upside-down and writes how much each input should change (`x.grad`).

Question: **Why must `y.mean()` (or another reduction) happen before `backward` on a vector y?**

In lecture words: this box is the first proof that autograd matches hand derivatives.

### Local picture

```
  A @ B  →  matrix product (not elementwise *)

  x (cpu) ──.to(device)──► x (cuda:0)
  rule: all operands of an op share one device

  requires_grad=True
        │
        ▼
  y = f(x)  →  y.backward()  →  x.grad = ∂y/∂x

  scalar demo: y = x² + 3x + 1, x=2 → grad 7
  vector setup: y = x², loss = y.mean() ≈ 4.667
```

**Notice:** `.grad` sits on the leaf tensor you asked to track, not on intermediate temps by default.

### Bridge

Finish the vector gradient analytically, then run **manual linear regression** with MSE, `no_grad` updates, and `grad.zero_()`.

---

## Topic 5: Multi-var grad manual linreg (22:00–28:30)

### Where this sits on the master map

**MANUAL GD** — Multi-variable ∂loss/∂x via mean-of-squares; then full **manual linear regression** `ŷ = wx + b` with MSE, **`loss.backward()`**, **`torch.no_grad()`** parameter update, and **`.grad.zero_()`** (gradient accumulation). Bridge to `nn.Module` next. Warm-up: [autograd](./PREREQUISITES.md#p4-autograd) · [train step](./PREREQUISITES.md#p5-trainstep).

### Board / screenshot

![Multi-var grad manual linreg](./screenshots/composites/ch05-topic-05-manual-linreg-panel1of1.png)

**Figure — ~22:00–28:30:** Multi-var grads match whiteboard; 100-epoch manual linreg with no_grad + zero_grad.

### What he is establishing

Continuing the vector demo: **loss = (1/3)(y₁ + y₂ + y₃)** with **yᵢ = xᵢ²**, so **loss = (1/3)(x₁² + x₂² + x₃²)** ≈ **4.667**. Analytic gradient for each coordinate: **∂loss/∂xᵢ = (2/3) xᵢ**. At **x = [1, 2, 3]**: gradients **[2/3, 4/3, 2] ≈ [0.667, 1.333, 2.0]**. Autograd matches this vector.

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x**2
loss = y.mean()          # (1+4+9)/3 = 14/3 ≈ 4.667
loss.backward()
print(x.grad)
# tensor([0.6667, 1.3333, 2.0000])
```

This is the first full **multi-variable** autograd calculation so you can trust `x.grad` before training loops.

Manual **linear regression** with autograd. Inputs: four examples **x = 1, 2, 3, 4** as a column. Targets: **y_true = 3, 5, 7, 9** — consistent with true relation **y ≈ 2x + 1**. Model form: **ŷ = w x + b**. Initialize **w** and **b** with **`torch.randn(1, requires_grad=True)`**. Learning rate **`0.01`**. Training loop: **100 epochs**, full-batch. Loss = **mean squared error**. Wrap the update in **`with torch.no_grad():`** so **w ← w − lr·w.grad** is **not tracked**. PyTorch **accumulates** gradients by default — clear with **`w.grad.zero_()`** and **`b.grad.zero_()`**. Log every 10 epochs; print learned **w** and **b**. Demo final params near true **2** and **1** (e.g. ≈ **w = 2.13**, **b = 0.61** depending on init).

```python
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0], [9.0]])
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
learning_rate = 0.01

for epoch in range(100):
    y_pred = x * w + b
    loss = torch.mean((y_pred - y_true) ** 2)
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Learned weight:", w.item())
print("Learned bias:", b.item())
# Demo-ish readout (depends on randn seed):
# Epoch 90, Loss: 0.0270
# Learned weight: ~2.13
# Learned bias:   ~0.61
```

Each epoch: forward → MSE → backward → no_grad SGD step → clear grads → optional log. Loss decreases; parameters approach the true line.

Transition teaser: redo the same problem using **packages/modules** — every PyTorch model should be a **child of `nn.Module`**.

```python
# ============================================================
# nn.Module
# All PyTorch neural networks are written using nn.Module
# ============================================================
class LinearRegressionModel(nn.Module):
    ...  # filled in Topic 6
```

That class shell is the contract for every network you will write next.

You can now verify multi-var gradients and run a complete **manual** GD loop. Still missing: `nn.Linear`, built-in losses, and optimizers that hide the arithmetic.

A common trap is forgetting **`grad.zero_()`** — the next `backward` **adds** to old grads instead of replacing them, so steps drift.

### Analogy for this topic only

Manual linreg is **steering a car by hand** each second: look at error (loss), feel which way to turn (`grad`), turn the wheel yourself (`w -= lr * grad`), then wipe the old feel (`zero_`) so the next second is clean. `no_grad` means “turning the wheel is not another entry on the receipt.”

Question: **Why wrap the update in `torch.no_grad()` instead of treating `w -= ...` as another tracked op?**

In lecture words: this box is the pedagogical GD loop you will never fully abandon when debugging.

### Local picture

```
  vector: loss = mean(x²)  →  ∂/∂xᵢ = (2/3) xᵢ
          [1,2,3] → grads ≈ [0.67, 1.33, 2.0]

  linreg data: x=[1,2,3,4]ᵀ, y=[3,5,7,9]ᵀ  (true ~ 2x+1)

  each epoch:
    y_pred = x*w + b
    loss = mean((y_pred - y_true)²)
    loss.backward()
    with no_grad:
      w -= lr * w.grad
      b -= lr * b.grad
      zero_ grads
```

**Notice:** `.item()` pulls a Python float from a 0-dim tensor for printing.

### Bridge

Replace hand-rolled `w,b` with **`nn.Module` + `nn.Linear` + criterion + optimizer** — same math, standard API.

---

## Topic 6: nn.Module Linear optim (28:30–36:00)

### Where this sits on the master map

**METHOD** — Replace hand-rolled $w,b$ + manual GD with **`nn.Module` + `nn.Linear` + criterion + optimizer**. Same math $wx+b$, but parameters live inside the module; the **standard train step** is predict → loss → `zero_grad` → `backward` → `step`. Warm-up: [module](./PREREQUISITES.md#p6-module) · [train step](./PREREQUISITES.md#p5-trainstep).

### Board / screenshot

![nn.Module Linear optim](./screenshots/composites/ch06-topic-06-nn-module-optim-panel1of1.png)

**Figure — ~28:30–36:00:** `LinearRegressionModel`; MSE + SGD; 5-step train loop; `named_parameters`; multi-feature Linear teaser.

### What he is establishing

All PyTorch neural networks are written as a **child class of `nn.Module`**. In `__init__`, first call **`super().__init__()`**, then declare layers as attributes. **`nn.Linear(in_features, out_features)`** is the built-in fully connected / affine layer implementing $wx+b$. **Models are batch-agnostic:** when declaring `nn.Linear(in, out)`, think only about the **per-example** feature size. For scalar regression, per-example input and target are scalars → `nn.Linear(1, 1)`. **Bias is True by default**. You must implement **`forward(self, x)`** — calling `model(x)` runs `forward`.

```python
import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
print(model)
# Linear(in_features=1, out_features=1, bias=True)
```

Subclass, register one Linear, define forward as that Linear — instantiate and inspect. With the module you **do not** manually create `W` and `b` tensors — parameters live inside `nn.Linear`.

Use a **predefined loss**: `criterion = nn.MSELoss()`. Optimizer: **`optim.SGD`** (or Adam); pass **`model.parameters()`** so the optimizer owns the update list. Learning rate demo: **`lr=0.01`**. Advanced note: a model can have multiple parameter subsets, each updated by a different optimizer — not used here. Same data as manual linreg.

```python
import torch.optim as optim

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0], [9.0]])
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

Criterion is MSE; SGD is wired to every learnable tensor the module registered.

**Standard training procedure:** (1) prediction, (2) compute loss, (3) `zero_grad`, (4) `loss.backward`, (5) `optimizer.step`. `zero_grad` can sit after the loss computation as long as it is **before** `backward`. `loss.item()` extracts the Python float for printing.

```python
for epoch in range(100):
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

Five-step loop: forward → MSE → clear grads → backward → SGD step; print every 10 epochs.

Demo result: loss around **~0.017** by epoch ~90 (vs manual GD earlier ~0.027). Inspect learned parameters with **`model.named_parameters()`** — demo recovers weight ≈ **2.03** and bias ≈ **0.90** (true line near $y=2x+1$).

```python
for name, param in model.named_parameters():
    print(name, param.data)
# linear.weight ≈ [[2.03]]
# linear.bias   ≈ [0.90]
```

Named parameters expose the Linear’s weight and bias tensors after training.

Bridge to multi-feature / batches: declare `nn.Linear(in_features=4, out_features=3)` — each input size 4, each output size 3. Weight matrix maps last dim; with a batch of 5, $X\in\mathbb{R}^{5\times4}$ maps to $\mathbb{R}^{5\times3}$.

```python
# Linear layer shapes:
# Input shape:  batch_size x input_features
# Output shape: batch_size x output_features
linear_layer = nn.Linear(in_features=4, out_features=3)
x = torch.randn(5, 4)
output = linear_layer(x)  # (5, 3)
```

Five examples × four features become five × three outputs — batch size is free; only feature widths are declared.

You can now write models as `nn.Module` children and train with criterion + optimizer. Still missing: nonlinear activations and **classification** loss (CrossEntropy).

A common trap is forgetting **`super().__init__()`** — parameters may not register, and `model.parameters()` stays empty for the optimizer.

### Analogy for this topic only

`nn.Module` is a **LEGO kit with a manual**: pieces (`Linear`) + instructions (`forward`). The optimizer only turns knobs the kit **registered**. Manual GD was hand-tightening bolts; `optimizer.step()` is the powered screwdriver on the whole kit.

Question: **What five steps make the standard train iteration?**

In lecture words: this box is the train-step contract every later architecture reuses.

### Local picture

```
  class Model(nn.Module):
    __init__: super(); self.linear = nn.Linear(in, out)
    forward:  return self.linear(x)

  criterion = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=0.01)

  loop:
    pred = model(x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  Linear(4,3) on (5,4) → (5,3)   # batch-agnostic declaration
```

**Notice:** bias is on by default; batch size never appears in `nn.Linear(...)`.

### Bridge

Stack multi-dim Linear shapes with **elementwise activations**, **softmax**, and **`nn.CrossEntropyLoss` on logits**.

---

## Topic 7: Activations CrossEntropy (36:00–42:00)

### Where this sits on the master map

**METHOD** — Multi-dim **`nn.Linear` / $Wx+b$ shapes**, then **elementwise nonlinearities** (`F.relu`, `sigmoid`, `tanh`) and **softmax → posteriors**. Classification loss is **`nn.CrossEntropyLoss`**, which takes **raw logits** (softmax is inside). Warm-up: [activations & CE](./PREREQUISITES.md#p7-loss).

### Board / screenshot

![Activations CrossEntropy](./screenshots/composites/ch07-topic-07-activations-ce-panel1of1.png)

**Figure — ~36:00–42:00:** Linear 4→3 shapes; ReLU/sigmoid/tanh; softmax dim=1; CE on logits + integer targets.

### What he is establishing

Multi-dim affine: $W$ maps 4-D features to 3-D; batch $X\in\mathbb{R}^{5\times4}$ → output $\in\mathbb{R}^{5\times3}$. Interpretation: **5 examples**, each in **4-D** feature space, transformed into **3-D** space.

```python
linear_layer = nn.Linear(in_features=4, out_features=3)
x = torch.randn(5, 4)       # 5 examples × 4 features
output = linear_layer(x)
print(output.shape)         # torch.Size([5, 3])
```

Last dimension is rewritten from 4 to 3; batch axis 0 is free.

**Neural nets without nonlinearities are of little use** for most interesting tasks — need activation functions between linear maps. Famous activations: **ReLU**, **sigmoid**, **tanh** (others exist). Activations are applied **elementwise**. Functional API: `F.relu(x)`; also `torch.sigmoid(x)` and `torch.tanh(x)`. **ReLU** zeroes negatives and keeps positives: `[-2,-1,0,1,2] → [0,0,0,1,2]`.

```python
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
relu_output = F.relu(x)
sigmoid_output = torch.sigmoid(x)
tanh_output = torch.tanh(x)
print("Input:", x)
print("ReLU:", relu_output)
print("Sigmoid:", sigmoid_output)
print("Tanh:", tanh_output)
# F.relu: negatives → 0; non-negatives unchanged
# [-2, -1, 0, 1, 2] → [0, 0, 0, 1, 2]
```

Three classic elementwise maps on the same 1D input for side-by-side comparison.

Convert **logits → posteriors** with **softmax** — get $P(y\mid x)$ for classification. Softmax along a chosen dimension: `F.softmax(logits, dim=1)` — **`dim=1`** means apply along the class axis. Outputs **sum to 1**. Softmax entry at class $c$ is $P(y=c\mid x)$.

```python
logits = torch.tensor([[2.0, 1.0, 0.1]])
probabilities = F.softmax(logits, dim=1)
print("Logits:", logits)
print("Probabilities:", probabilities)
print("Sum:", probabilities.sum())  # ≈ 1.0
```

One example’s three scores become a valid probability distribution over three classes.

**Critical slogan for CE:** input to **`nn.CrossEntropyLoss` is logits**, not probabilities — conversion to probs is **inside** the loss; do not pre-softmax for CE. CE demo: two examples, three classes; targets are **class indices** (not one-hot). CE ≈ mean $-\log p_{\mathrm{true}}$. **Loss is always a scalar** after reduction. Use **`loss.item()`** for logging.

```python
# CrossEntropyLoss
# Important: input = logits (NOT probabilities)
criterion = nn.CrossEntropyLoss()
logits = torch.tensor([
    [2.0, 1.0, 0.1],   # example 0
    [0.5, 2.5, 0.3],   # example 1
])
targets = torch.tensor([0, 1])  # true classes
loss = criterion(logits, targets)
print("Loss:", loss.item())  # e.g. ≈ 0.3185...
```

CE scores raw logits against integer labels and returns one scalar for the batch. Bridge: next skill is **custom datasets** so real / non-built-in data can be batched for mini-batch SGD.

You can now activate Linear outputs and score classification with CE on logits. Still missing: packaging real samples into **Dataset** and **DataLoader**.

A common trap is **softmax then CrossEntropy** — double softmax collapses gradients and mis-scales the loss.

### Analogy for this topic only

Logits are **raw contest scores**. Softmax turns them into **prize shares** that sum to 1. CrossEntropy scolds you if the true winner’s share is tiny. The CE module wants the **raw scores**, not the already-shared prizes — it does the sharing internally.

Question: **Does `nn.CrossEntropyLoss` want probabilities or logits?**

In lecture words: this box is the classification head + loss contract for MLPs.

### Local picture

```
  (5,4) ──Linear(4,3)──► (5,3)

  activations (elementwise):
    ReLU max(0,z) · sigmoid · tanh

  logits ──softmax(dim=1)──► probabilities (sum=1)
              │
              └── for CE: do NOT pre-softmax

  CrossEntropyLoss(logits, class_indices) → scalar
```

**Notice:** targets are integers like `[0, 1]`, not one-hot rows.

### Bridge

Public toys exist (MNIST, …), but real apps need a **custom Dataset** with three methods, then a **DataLoader** for mini-batches.

---

## Topic 8: Custom Dataset (42:00–48:00)

### Where this sits on the master map

**DATA PIPELINE** — Public toy sets (MNIST / Fashion-MNIST / CIFAR-10) exist in PyTorch; **real apps need custom data**. Subclass **`Dataset`** with **`__init__` / `__len__` / `__getitem__`**, then wrap with **`DataLoader`** for mini-batches. Warm-up: [Dataset vs DataLoader](./PREREQUISITES.md#p8-data).

### Board / screenshot

![Custom Dataset](./screenshots/composites/ch08-topic-08-dataset-panel1of1.png)

**Figure — ~42:00–48:00:** Toy public sets; SimpleDataset three methods; DataLoader batch_size=16 shapes.

### What he is establishing

Many **toy datasets** needed for learning are **publicly available** in the PyTorch ecosystem / torchvision-style repositories: **MNIST**, **Fashion-MNIST**, **CIFAR-10**. MNIST and Fashion-MNIST are **grayscale**; **CIFAR-10** is **color** at smaller spatial size. Public sets cover various tasks, but **real applications** often need data **not** in the public repo. Goal of a data pipeline: obtain examples, **put them into batches**, iterate for **mini-batch SGD**.

Analogy to models: models subclass **`nn.Module`**; datasets subclass **`torch.utils.data.Dataset`**. A custom Dataset **must** implement three methods: **`__init__`**, **`__len__`**, **`__getitem__`**. `__init__` stores tensors/paths; `__len__` returns how many examples; `__getitem__(index)` returns the **single** example (and label) at that index.

Toy construction: `self.X = torch.randn(100, 2)` — **100 examples**, each with **2 features**. Labels: $y_i = 1$ if $X_{i0}+X_{i1} > 0$, else $0$ — cast boolean to long so True→1, False→0.

```python
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self):
        self.X = torch.randn(100, 2)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
```

Full toy dataset: random 100×2 features, linear-threshold labels, length and indexing contracts for the loader.

```python
dataset = SimpleDataset()
print(len(dataset))           # 100
sample_x, sample_y = dataset[0]
print("Sample features:", sample_x)
print("Sample label:", sample_y)
```

`len` hits `__len__`; `dataset[0]` hits `__getitem__(0)` and returns one pair.

Why batching: 2-D toy data is easy whole; **ImageNet-scale** images e.g. **$224\times224\times3$** cannot all be held/processed at once → need batches. **`DataLoader`** wraps a Dataset and yields batches. Demo: `batch_size=16`. With 100 examples, the **last batch is smaller** (remainder). **Models remain batch-size agnostic**.

```python
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
)

for batch_x, batch_y in dataloader:
    print("Batch input shape:", batch_x.shape)
    print("Batch label shape:", batch_y.shape)
    break  # inspect one batch
```

One iteration yields stacked features and labels for up to 16 examples; break after inspecting shapes.

You can now wrap custom data as a Dataset and iterate mini-batches. Still missing: why **`shuffle=True`** matters, and the **MLP architecture** that consumes those batches.

A common trap is implementing only `__init__` and forgetting `__len__` / `__getitem__` — DataLoader will fail when it tries to index the dataset.

### Analogy for this topic only

**Dataset** is the **library catalog** (item by index) — `__len__` is how many books exist; `__getitem__(i)` hands you book *i* and its sticker (label). **DataLoader** is a **book cart** that brings up to 16 books at a time and can reshuffle the queue. Public MNIST shelves exist for homework; your company data is a private wing you catalog yourself with the **same three methods**. The cart’s last trip may carry fewer books when 100 is not divisible by 16 — that is expected, not a bug.

Question: **What three methods must every custom Dataset implement?**

In lecture words: this box is the data side of the train loop.

### Local picture

```
  public toys: MNIST · Fashion-MNIST · CIFAR-10
  real apps → custom Dataset

  Dataset:
    __init__     store X, y (or paths)
    __len__      → N
    __getitem__  → (x_i, y_i)

  SimpleDataset: X ∈ R^{100×2}, y = 1[x0+x1 > 0]

  DataLoader(dataset, batch_size=16, shuffle=True)
       │
       ▼
  for batch_x, batch_y in loader:  # last batch may be <16
```

**Notice:** models do not hard-code batch size 16 — any batch length with matching feature dim is valid.

### Bridge

Nail down **shuffle each epoch**, then design a multi-layer **MLP** as `nn.Module` with stacked FC layers.

---

## Topic 9: DataLoader shuffle (48:00–52:30)

### Where this sits on the master map

**DATA + ARCH** — **`shuffle=True`** reshuffles after each epoch so batch membership changes; then design a multi-layer **MLP** as `nn.Module` with stacked **`nn.Linear` (FC)** layers and hidden widths as free design choices. Warm-up: [data](./PREREQUISITES.md#p8-data) · [module](./PREREQUISITES.md#p6-module).

### Board / screenshot

![DataLoader shuffle](./screenshots/composites/ch09-topic-09-dataloader-panel1of1.png)

**Figure — ~48:00–52:30:** shuffle vs fixed order; MLP width design; FC1/FC2/FC3 skeleton with ReLU.

### What he is establishing

Key DataLoader flag: **`shuffle=True`** — do not always stream batches in fixed sequential order. Without shuffle, sequential splits repeat: batch 0 = indices 0–15, batch 1 = 16–31, etc., **same order every epoch**. With **`shuffle=True`**, after each epoch the data is reshuffled so an example that was in the first batch need not stay there. Iteration yields **`batch_x`, `batch_y`**; full batches have **16 examples** when `batch_size=16`. This Dataset → DataLoader pattern is the **standard procedure**; only Dataset contents change (MNIST vs CIFAR-10 vs custom).

```python
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
)

for batch_x, batch_y in dataloader:
    # batch_x: (≤16, n_features); batch_y: (≤16,)
    ...
```

Shuffle re-randomizes batch membership across epochs while keeping the same Dataset.

Next build: **MLP for (binary) classification**, kept **agnostic** to `input_dim` and `num_classes` via constructor args. Forward math: $x \mapsto \phi(xW_1+b_1) \mapsto \phi(\cdot W_2+b_2) \mapsto (\cdot W_3+b_3)$ — three affine maps with nonlinearities after the first two. Activations after hidden layers can differ; **last layer usually has no activation** for classification when using CrossEntropy (logits out). Board sketch with three layer blobs is **not** “three nodes” — dots are schematic; node counts come from dimensions. **Fixed by the task/data:** `input_dim` and `num_classes`. **Free design choices:** hidden dimensions. **FC = fully connected** = `nn.Linear`. Design choice in this demo: **same hidden width** for both hidden layers.

```python
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x  # logits
```

Three FC layers with ReLU between them; final Linear returns logits for CE. Constructor args make widths reusable for other feature sizes and class counts.

You can now shuffle batches properly and declare an MLP skeleton. Still missing: parameter counts, device placement for model **and** batches, Adam + CE training, and **accuracy**.

A common trap is treating schematic “three dots” on a board as three neurons — widths come from `input_dim` / `hidden_dim` / `num_classes`, not from the drawing.

### Analogy for this topic only

Without shuffle, the model always sees the same **classroom seating chart** — batch 0 is always the same 16 students, so it can cheat on “whoever sits in row 1.” With shuffle, seats remix every epoch so batch membership is not a free hint. The MLP is a **three-room corridor**: entrance width fixed by feature count, exit width fixed by number of classes, middle room sizes (`hidden_dim`) you choose as architecture design. The last room posts **raw scorecards** (logits), not prize shares — CE will do the sharing.

Question: **Which dimensions of an MLP are fixed by the dataset, and which are free design choices?**

In lecture words: this box is the architecture declaration before the full train loop.

### Local picture

```
  shuffle=False:  indices 0–15, 16–31, ... every epoch
  shuffle=True:   reshuffle membership after each epoch

  data fixes:   input_dim, num_classes
  you choose:   hidden_dim (here reused twice)

  x ─FC1─ReLU─FC2─ReLU─FC3─► logits
      W1         W2         W3
```

**Notice:** no activation after FC3 when pairing with CrossEntropyLoss.

### Bridge

Wire sizes **2 → 32 → 32 → 2**, count parameters, put model and batches on **device**, train with **CE + Adam**, track **loss and accuracy**.

---

## Topic 10: MLP train accuracy (52:30–62:08)

### Where this sits on the master map

**METHOD + RECAP** — Wire the 3-layer **MLP** (`FC1→ReLU→FC2→ReLU→FC3`), count parameters, put model **and batches on device**, train with **CE + Adam**, track **loss and accuracy** (`argmax` vs labels). Tutorial closes; next segment = CNNs/RNNs. Warm-up: [train step](./PREREQUISITES.md#p5-trainstep) · [module](./PREREQUISITES.md#p6-module) · [loss](./PREREQUISITES.md#p7-loss) · [data](./PREREQUISITES.md#p8-data).

### Board / screenshot

![MLP train accuracy panel 1](./screenshots/composites/ch10-topic-10-mlp-train-panel1of2.png)

![MLP train accuracy panel 2](./screenshots/composites/ch10-topic-10-mlp-train-panel2of2.png)

**Figure — ~52:30–62:08:** Forward chain and param count; full train loop with device, CE, Adam, argmax accuracy; recap / next CNNs RNNs.

### What he is establishing

Forward recipe: **FC1 → ReLU → FC2 → ReLU → FC3** (no activation after final linear). Demo sizes: **`input_dim=2`**, **`hidden_dim=32`**, **`num_classes=2`**. For binary classification you **could** use a single output + BCE; with **two logits** + CrossEntropy is also fine.

Weight shapes: $W_1\in\mathbb{R}^{2\times32}$ (64), $W_2\in\mathbb{R}^{32\times32}$ (1024), $W_3\in\mathbb{R}^{32\times2}$ (64). Biases: **32 + 32 + 2**. Total weights $64+1024+64=1152$ plus biases $66$ → **1218** parameters. Create the model and **immediately `.to(device)`**.

```python
def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    return x

model = MLP(input_dim=2, hidden_dim=32, num_classes=2).to(device)
print(model)
n_params = sum(p.numel() for p in model.parameters())
# weights: 2*32 + 32*32 + 32*2 = 64 + 1024 + 64 = 1152
# biases:  32 + 32 + 2 = 66
# total:   1218
```

Instantiate on the shared device and count every learnable scalar the optimizer will touch.

Training setup reuses **SimpleDataset** + **DataLoader** (`batch_size=16`, `shuffle=True`). Loss: **`nn.CrossEntropyLoss()`**. Optimizer: **`optim.Adam(model.parameters(), lr=0.001)`**. Train **20 epochs**. **`model.train()`** puts the module in training mode (habit for dropout/batchnorm later). Epoch counters: `total_loss`, `correct`, `total`. **Device port for batches:** DataLoader tensors default to CPU — **`.to(device)` each batch**. Forward: `logits = model(batch_x)` — **do not** softmax before CE. Targets are **class index vectors**. Optim step: `zero_grad` → `backward` → `step`. Accumulate `total_loss += loss.item()`. Decision = **`argmax`**, not the max value — **argument** of the max is the predicted class; use **`dim=1`** for `(B, C)` logits. Count correct with boolean mask sum; `batch_y.size(0)` is batch length; `accuracy = correct / total`.

```python
dataset = SimpleDataset()
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
)

model = MLP(input_dim=2, hidden_dim=32, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)
    accuracy = correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
```

Full mini-batch loop: move batch → logits → CE → Adam step → argmax accuracy. Demo trajectory: accuracy starts near **0.50** (chance on a balanced binary toy) and rises to about **0.98+** by ~epoch 15; later near **0.99–1.00**. That fast climb is expected on a linearly separable synthetic label rule — real images will train slower and need the CNN/RNN architectures next.

Closing slogan: **this is how you work with an MLP** in PyTorch — Module + Dataset/DataLoader + CE + optimizer loop + accuracy. Next tutorial segment will declare and train **CNNs** and **RNNs** similarly. Tutorial ends after MLP basics; thank-you / meet next segment.

You can now train a small classifier end-to-end on custom data with device-aware batches and accuracy. Still missing (next units): convolutional and recurrent architectures — the **train step stays the same**.

A common trap is leaving **batches on CPU** while the model is on CUDA — move **both** sides of every step, not only the model at construction. A second trap: printing `max(logits)` instead of **`argmax`** — the max *value* is a score; the predicted **class index** is the argument of that max along `dim=1`.

### Analogy for this topic only

The full loop is a **daily factory shift**: put workers and materials on the same floor (device), assemble product (forward), score defects (loss), wipe old corrections (`zero_grad`), feel which knobs to turn (`backward`), turn them (`step`), then count good units (accuracy). The parameter count (**1218**) is every bolt the shift is allowed to tighten. CNNs/RNNs next week change the assembly-line layout, not the shift protocol.

Question: **Why is predicted class `argmax(logits, dim=1)` rather than the maximum logit value itself?**

In lecture words: this box closes the PyTorch basics stack for the course.

### Local picture

```
  widths: 2 → 32 → 32 → 2
  params: 1152 weights + 66 biases = 1218

  model.to(device)
  each batch: batch_x/y.to(device)

  train:
    model.train()
    logits = model(batch_x)          # no softmax
    loss = CE(logits, batch_y)       # integer labels
    zero_grad → backward → step
    pred = argmax(logits, dim=1)
    acc = (# pred == y) / N

  demo: acc ~0.50 → ~0.98+ by epoch 15
  next: CNN / RNN (same train pattern)
```

**Notice:** `size(0)` is the current batch length (handles partial last batch).

### Bridge

You own the PyTorch basics spine. Next lecture units swap the Module guts for **convolutions and recurrence** while reusing device, Dataset/DataLoader, CE, and the train step.

---

## Apply it (scenarios)

Practice these without the video open.

1. **Device + tensor audit.** Create `x = torch.randn(4, 3)`, print `.shape`, `.dtype`, `.device`, then `x = x.to(device)` and re-print device. Confirm a CPU/CUDA matmul mismatch errors if you force one operand back to the other device.
2. **Shape gym.** From `torch.arange(12)`, reshape to `(3, 4)`, unsqueeze batch dim to `(1, 3, 4)`, squeeze back. Permute a fake image `(224, 224, 3)` to CHW.
3. **Autograd check.** With `x = torch.tensor(2.0, requires_grad=True)`, form $y = x^2 + 3x + 1$, `backward`, assert `x.grad == 7`.
4. **Manual vs Module linreg.** Run 100-step manual $wx+b$ MSE; then the same data with `nn.Linear(1,1)` + SGD. Compare final weight/bias to the true line $2x+1$.
5. **CE logits rule.** Compute CE on logits; then wrongly softmax first and compare losses — notice the double-softmax pathology.
6. **Dataset → MLP.** Implement `SimpleDataset`, wrap with `DataLoader(batch_size=16, shuffle=True)`, train the Topic-10 MLP for 20 epochs, and plot or print accuracy rising from ~0.5.

---

## External references

**How to use:** finish the NOTES chain first (video closed if you can). When one map box still feels thin, open **only that topic’s group** below — **2–3 companions each** (prefer **teaching video + blog/notes + official doc**). All links live **here**, not inside topic bodies.

Prefer free, well-known teaching channels and official tutorials. Skip Wikipedia dumps and random SEO posts.

### Topic 1 — Imports, version, device

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Patrick Loeber — Deep Learning with PyTorch (full course; start: install + setup)](https://www.youtube.com/watch?v=c36lUUr864M) | Video | Same import / CUDA / Colab spirit as the lecture opening |
| [PyTorch — Get Started locally](https://pytorch.org/get-started/locally/) | Official | Match CPU vs CUDA wheel to your machine |
| [Google Colab — notebook runtime (T4 GPU)](https://colab.research.google.com/) | Tooling | Free accelerator path the instructor demonstrates |

### Topic 2 — Tensor create, attributes, index

| Resource | Type | Why it helps |
|----------|------|--------------|
| [PyTorch official — Tensors tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) | Official tutorial | Create, attributes, indexing in library words |
| [freeCodeCamp / Aladdin — PyTorch for Deep Learning (tensor basics chapter)](https://www.youtube.com/watch?v=V_xro1bcAuA) | Video | Long beginner walkthrough; use the early tensor sections |
| [Real Python — PyTorch tensors intro](https://realpython.com/pytorch-tensors/) | Blog | Readable create / shape / dtype walkthrough |

### Topic 3 — Reshape, unsqueeze, squeeze, permute

| Resource | Type | Why it helps |
|----------|------|--------------|
| [torch.reshape docs](https://pytorch.org/docs/stable/generated/torch.reshape.html) | Docs | Element-count rule; safer default than `view` |
| [torch.permute docs](https://pytorch.org/docs/stable/generated/torch.permute.html) | Docs | Axis reorder (HWC ↔ CHW) with clear signature |
| [AssemblyAI — PyTorch tensor reshape & view (short)](https://www.youtube.com/watch?v=fCVuiW9AFzY) | Video | Visual reshape vs view / contiguity intuition |

### Topic 4 — Matmul + autograd intro

| Resource | Type | Why it helps |
|----------|------|--------------|
| [PyTorch official — Autograd tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) | Official tutorial | `requires_grad`, graph, `backward` |
| [3Blue1Brown — Gradient descent](https://www.youtube.com/watch?v=IHZwWFHWa-w) | Video | Why $\theta \leftarrow \theta - \eta\nabla L$ appears |
| [Jovian — PyTorch basics & gradient descent](https://www.youtube.com/watch?v=m_tkL7DufPk) | Video | Tensors → autograd → GD in one beginner pass |

### Topic 5 — Multi-var grad + manual linreg

| Resource | Type | Why it helps |
|----------|------|--------------|
| [3Blue1Brown — What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U) | Video | Multi-variable chain-rule intuition matching the board |
| [PyTorch — Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) | Docs | Accumulation, leaves, when `zero_` is required |
| [Patrick Loeber — Linear regression from scratch in PyTorch](https://www.youtube.com/watch?v=VVDHU_TWwUg) | Video | Manual `w,b` + `no_grad` update sibling to Topic 5 |

### Topic 6 — nn.Module, Linear, optimizer

| Resource | Type | Why it helps |
|----------|------|--------------|
| [PyTorch official — Build the Model](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) | Official tutorial | `nn.Module`, `Linear`, `forward` |
| [StatQuest — Neural Networks Pt. 1 (intuition)](https://www.youtube.com/watch?v=CqOfi41LfDw) | Video | What a fully connected layer is doing before code |
| [torch.optim package guide](https://pytorch.org/docs/stable/optim.html) | Docs | SGD/Adam API; `zero_grad` / `step` contract |

### Topic 7 — Activations + CrossEntropy (logits)

| Resource | Type | Why it helps |
|----------|------|--------------|
| [StatQuest — Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU) | Video | $-\log p_{\mathrm{true}}$ intuition for CE |
| [StatQuest — Softmax function](https://www.youtube.com/watch?v=KpKog-L9veg) | Video | Scores → probabilities (and why CE wants logits raw) |
| [PyTorch — CrossEntropyLoss docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) | Docs | **Logits in**, integer targets, internal log-softmax |

### Topic 8 — Custom Dataset

| Resource | Type | Why it helps |
|----------|------|--------------|
| [PyTorch official — Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) | Official tutorial | Custom Dataset pattern used in class |
| [Patrick Loeber — Dataset and DataLoader](https://www.youtube.com/watch?v=PXOzkkB5eC0) | Video | `__init__` / `__len__` / `__getitem__` live coding |
| [torchvision datasets overview](https://pytorch.org/vision/stable/datasets.html) | Docs | MNIST / Fashion-MNIST / CIFAR-10 public toys |

### Topic 9 — DataLoader shuffle + MLP skeleton

| Resource | Type | Why it helps |
|----------|------|--------------|
| [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) | Video | Stacked Linear + nonlinearity worldview |
| [PyTorch official — Optimization loop](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) | Tutorial | How loaders feed the train step |
| [DataLoader docs (`batch_size`, `shuffle`)](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) | Docs | Flags the lecture relies on |

### Topic 10 — Full MLP train + accuracy

| Resource | Type | Why it helps |
|----------|------|--------------|
| [PyTorch official — Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) | Official tutorial | End-to-end Module + loader + train/test loop |
| [3Blue1Brown — Gradient descent](https://www.youtube.com/watch?v=IHZwWFHWa-w) | Video | Why Adam/SGD steps drive loss down over epochs |
| [Machine Learning Mastery — PyTorch training loop patterns](https://machinelearningmastery.com/training-a-pytorch-model/) | Blog | Readable zero_grad → backward → step + metrics |

### Whole-map companions

| Resource | Type | Why it helps |
|----------|------|--------------|
| [pytorch.org — tutorials hub](https://pytorch.org/tutorials/) | Docs home | Canonical path while replaying any cell |
| [Patrick Loeber — Deep Learning with PyTorch (full course)](https://www.youtube.com/watch?v=c36lUUr864M) | Video series | Parallel curriculum: tensors → autograd → train → Dataset |
| [3Blue1Brown — Neural networks playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | Video series | Geometry behind GD / layers |
| [PREREQUISITES.md (this package)](./PREREQUISITES.md) | Warm-up | Beginner definitions (#p1–#p8) for every topic |
| [Tutorial 2 NumPy NOTES](../16-Tutorial02-Introduction-to-NumPy/NOTES.md) | Prior unit | Shape language this lecture extends |

---

## Sources

- Video: [Tutorial 3 : PyTorch Basics](https://www.youtube.com/watch?v=SEtu7Eef5ps)
- Channel: NPTEL — Indian Institute of Science, Bengaluru
- Course: Mathematical Foundations of Generative AI
- Instructor / speaker: NPTEL IISc
- Duration: ~62 min (00:04–62:08)
- Skill: `youtube-lecture-tutor` · `content_type: code_tutorial`
- Captions cleaned via timed transcript / claim sheets (restructure: **10 topics**)
- Warm-up: [PREREQUISITES.md](./PREREQUISITES.md)
- Previous: [Tutorial 2 — Introduction to NumPy](../16-Tutorial02-Introduction-to-NumPy/NOTES.md)
- Package path: `17-Tutorial03-PyTorch-Basics`
