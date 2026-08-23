# Prerequisites — warm-up before Tutorial 3 (PyTorch Basics)

> **Do this first** if tensors, GPU device, autograd, or `nn.Module` still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Course: NPTEL Mathematical Foundations of Generative AI · Tutorial 3.  
> Builds on [Tutorial 2 NumPy](../16-Tutorial02-Introduction-to-NumPy/NOTES.md).  
> **Beginner deep warm-up:** definition · micro example · analogy · notice · mini-check.

```
  After this warm-up you can say:

  "A torch.Tensor is like a NumPy array that can live on CPU or CUDA."
  "device chooses where the data and compute run; move tensors with .to(device)."
  "requires_grad=True builds a computation graph for automatic derivatives."
  "loss.backward() fills .grad; optimizer.step() updates parameters."
  "nn.Module packages layers; nn.Linear is Wx+b."
  "CrossEntropyLoss wants raw logits, not softmax probabilities."
  "Dataset gives one sample; DataLoader batches and can shuffle."
  "Train loop: zero_grad → forward → loss → backward → step."
```

**Warm-up → tutorial boxes**

```
  §1  Tensor vs NumPy array               ──► Topics 1–3
  §2  Device (CPU / CUDA)                 ──► Topics 1–2, 4, 10
  §3  Shape ops: reshape, squeeze, permute ──► Topic 3
  §4  Autograd & requires_grad            ──► Topics 4–5
  §5  Train step: zero_grad/backward/step ──► Topics 5–6, 10
  §6  nn.Module & nn.Linear               ──► Topics 6, 9–10
  §7  Activations & CrossEntropy (logits) ──► Topic 7
  §8  Dataset vs DataLoader               ──► Topics 8–9
```

---

## 1. Tensor vs NumPy array

<a id="p1-tensor"></a>

### Purpose for the video

PyTorch’s core data structure is **`torch.Tensor`** — multi-dimensional arrays with optional GPU placement and optional gradient tracking. Tutorial 2 taught you the shape language on NumPy; this tutorial keeps that language and adds **device + autograd**.

### Definitions

| Idea | Meaning |
|------|---------|
| **NumPy `ndarray`** | CPU numeric array (Tutorial 2) |
| **`torch.Tensor`** | Array + **device** + optional **gradient tracking** |
| **dtype** | Element type (`float32` is the deep-learning default) |
| **shape** | Sizes along each axis — same idea as NumPy (`torch.Size` acts like a tuple) |
| **`.item()`** | Pull a Python scalar out of a 0-dim tensor (for printing losses) |

### Worked micro

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])   # float literals → float dtype
print(x.shape)   # torch.Size([3])
print(x.dtype)   # torch.float32 (typical)
print(x.device)  # cpu until you move it
```

Integer literals alone can make integer tensors; neural-net math almost always wants **floats**. Prefer `1.0` style literals (or set `dtype=torch.float32`) when you mean training data or weights.

### Analogy — notebook vs lab server

A NumPy array is homework done in a **paper notebook** (CPU only). A tensor is the same homework that can be **uploaded to a lab server** (CUDA) and, if you mark it, comes with a **receipt of every calculation** so you can reverse the grades later (autograd). Same numbers, more superpowers.

### Notice

- Many APIs feel like NumPy (`+`, `*`, `@`, indexing, slicing).  
- Interop exists (`torch.from_numpy`, `.numpy()`), but both sides must be on **CPU**, and sharing memory can surprise you if you mutate one side.  
- New tensors default to **CPU** — always print `.device` once after create until that habit sticks.  
- `shape[0]` / `shape[1]` work like a tuple (rows / cols for 2D).

### Mini-check

1. Name two things a tensor can do that a plain NumPy array cannot (in standard use).  
2. Why prefer `torch.tensor([1.0, 2.0])` over `torch.tensor([1, 2])` for model math?  
3. What does `shape` report?

---

## 2. Device: CPU and CUDA

<a id="p2-device"></a>

### Purpose for the video

The tutorial picks a **device once** and reuses that variable for **every** `.to(device)` on tensors and models. Device mismatch is the #1 crash for beginners on Colab.

### Definitions

| Term | Meaning |
|------|---------|
| **CPU** | Default host processor; always available |
| **CUDA** | NVIDIA GPU API used by PyTorch for acceleration |
| **`torch.cuda.is_available()`** | `True` if a CUDA GPU can be used right now |
| **`torch.device(...)`** | A small object naming *where* to put data |
| **`.to(device)`** | Move a tensor **or** an `nn.Module` to that device |

### Worked micro

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(2, 3).to(device)
print(x.device)   # cuda:0  or  cpu

# Colab free GPU: Runtime → Change runtime type → T4 GPU
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # e.g. Tesla T4
```

### Analogy — home kitchen vs industrial oven

CPU is a **home kitchen**. CUDA is an **industrial oven**. Recipes (ops) must put **ingredients and tools in the same kitchen** — a CPU tensor cannot `@` a CUDA tensor. The `device` variable is the sticky note on the fridge: “today we cook **here**.” Do not hard-code `"cuda"` everywhere; machines without a GPU will break, and multi-GPU boxes need the same **card** for model and data unless you use advanced patterns.

### Notice

- Gradients live on the **same device** as their tensors.  
- DataLoader batches start on **CPU** — you still `.to(device)` each batch even if the model already moved.  
- Colab free tier defaults to **CPU** until you switch runtime and re-run.  
- Teaching rule for this course: **one device for everything**.

### Mini-check

1. What happens if you matmul a CPU tensor with a CUDA tensor?  
2. Write one line that chooses CUDA if available else CPU.  
3. Model is on CUDA; batch still on CPU — who needs `.to(device)`?

---

## 3. Reshape, unsqueeze, squeeze, permute

<a id="p3-shapeops"></a>

### Purpose for the video

Deep learning constantly **re-layouts** tensors without inventing new numbers — flatten for an MLP head, add a batch axis, put channels first for a CNN.

### Definitions

| Op | Meaning |
|----|---------|
| **`reshape` / `view`** | New shape; **same number of elements** |
| **`view`** | Like reshape but requires **contiguous** memory |
| **`reshape`** | Safer default when contiguity is unclear |
| **`unsqueeze(dim)`** | Insert a size-1 axis at `dim` (e.g. add batch) |
| **`squeeze`** | Remove size-1 axes |
| **`permute`** | **Reorder** axes (e.g. HWC → CHW) — not the same as reshape |

### Worked micro

```python
x = torch.arange(12)
y = x.reshape(3, 4)          # row-wise fill; 12 = 3×4
b = y.unsqueeze(0)           # (1, 3, 4) add batch dim
img = torch.randn(224, 224, 3)
chw = img.permute(2, 0, 1)   # (3, 224, 224)  HWC → CHW
```

`reshape(3, 5)` on 12 elements **errors** — product of dimensions must match.  
`permute(2, 0, 1)` means: new axis 0 ← old 2, new 1 ← old 0, new 2 ← old 1.

### Analogy — packing a suitcase

- **Reshape:** same clothes, different packing grid on the suitcase floor.  
- **Unsqueeze:** add an empty labeled pocket (batch slot).  
- **Squeeze:** remove empty pockets.  
- **Permute:** rotate the suitcase so the “color” face is on top for the CNN conveyor belt — you did not invent new fabric; you changed **which edge is which**.

### Notice

- Many vision layers want **NCHW** (`batch, channel, height, width`). Images loaded as HWC need `permute` first.  
- Prefer **`reshape`** over `view` unless you know the tensor is contiguous.  
- Elementwise `*` is **not** matmul; matmul is `@` (next box in the lecture).

### Mini-check

1. Shape after `unsqueeze(0)` on `(3, 4)`?  
2. Why permute `(224,224,3)` → `(3,224,224)`?  
3. Is `permute` the same as `reshape` to those sizes?

---

## 4. Autograd and `requires_grad`

<a id="p4-autograd"></a>

### Purpose for the video

**Automatic differentiation** is how PyTorch trains models without you writing ∂L/∂w by hand for every layer.

### Definitions

| Term | Meaning |
|------|---------|
| **`requires_grad=True`** | Track ops on this tensor for gradients |
| **Computation graph** | Record of ops from inputs → loss |
| **`loss.backward()`** | Walk the graph reverse-mode; fill `.grad` on leaves |
| **`.grad`** | Accumulated gradient tensor (same shape as the leaf) |
| **Leaf** | A tensor you created with `requires_grad=True` (e.g. weights) |

### Worked micro

```python
x = torch.tensor(3.0, requires_grad=True)
y = x * x          # y = x²
y.backward()
print(x.grad)      # dy/dx = 2x = 6

# Vector: reduce to a scalar before backward
v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (v ** 2).mean()
loss.backward()
print(v.grad)      # [2/3, 4/3, 2]
```

`backward()` needs a **scalar** output (or an explicit gradient vector). That is why training losses use `mean` / `sum` / a loss module that reduces to one number.

### Analogy — receipt tape

Every operation prints a line on a **store receipt** (the graph). `backward()` walks the receipt **upside-down** and writes how much each input should change to reduce the final total (loss). The writing lands in `.grad`. If you run `backward()` twice without wiping, the second walk **adds** more ink on the same receipt line — that is **gradient accumulation**.

### Notice

- Gradients **accumulate** — call `optimizer.zero_grad()` or `x.grad.zero_()` before each new backward in a normal step.  
- Only **floating** tensors get gradients.  
- Use **`torch.no_grad()`** when you update weights manually or run inference so those ops are **not** taped onto the receipt.  
- `.item()` is for logging a scalar loss; it is not required for `backward`.

If this still blurs, 3Blue1Brown’s gradient-descent video is the popular picture; the official autograd tutorial is the matching code. Pointers live at the end of NOTES.

### Mini-check

1. What attribute holds dy/dx after `backward`?  
2. Why zero gradients between steps?  
3. Why call `y.mean()` before `backward` when `y` is a vector?

---

## 5. The train step pattern

<a id="p5-trainstep"></a>

### Purpose for the video

Almost every PyTorch loop uses the same five moves. CNN/RNN lectures later **reuse this spine** and only change the architecture.

### Pattern

```python
model.train()               # 0 training mode (Dropout/BatchNorm later)
optimizer.zero_grad()       # 1 clear old grads
pred = model(x)             # 2 forward
loss = criterion(pred, y)
loss.backward()             # 3 compute grads
optimizer.step()            # 4 update parameters
```

Order note from the lecture: `zero_grad` can sit after computing the loss **as long as it is before `backward`**. Never `step` before `backward` — there would be nothing useful in `.grad` yet.

### Manual update (no optimizer)

```python
with torch.no_grad():
    w -= lr * w.grad
    w.grad.zero_()
```

`no_grad` means “turning the wheel is not another line on the receipt.”

### Analogy — steering a car each second

1. Put the car in drive (`model.train()`).  
2. Forget last second’s steering correction (`zero_grad`).  
3. Look at the road (`forward`).  
4. Measure error (`loss`).  
5. Feel which way to turn (`backward`).  
6. Actually turn the wheel (`step`).  

If you skip step 2, yesterday’s pull on the wheel **adds** to today’s — the car drifts.

### Notice

- `model.eval()` + `torch.no_grad()` is the inference pair (not the focus of this tutorial, but know it exists).  
- Log with `loss.item()` so you print a Python float, not a 0-dim tensor graph node.  
- Same pattern for **manual** linreg (Topic 5) and **Module + optim** (Topics 6 and 10).

### Mini-check

1. Order of `backward` and `step`?  
2. When use `no_grad`?  
3. What goes wrong if you forget `zero_grad`?

---

## 6. `nn.Module` and `nn.Linear`

<a id="p6-module"></a>

### Purpose for the video

Models are **classes** inheriting `nn.Module`, not loose weight tensors floating in a script. The module **registers** parameters so optimizers and `.to(device)` see them.

### Definitions

| Piece | Meaning |
|-------|---------|
| **`nn.Module`** | Base class: registers parameters; `.to(device)`, `.train()` / `.eval()`, `parameters()` |
| **`nn.Linear(in, out)`** | Learns $y = xW^\top + b$ for a batch of row vectors |
| **`forward(self, x)`** | How input becomes output — call via `model(x)` |
| **Parameters** | Tensors with `requires_grad` managed by the module |
| **Batch-agnostic** | You declare **feature widths**, not batch size — `(B, in) → (B, out)` for any B |

### Worked micro

```python
import torch.nn as nn

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()          # required
        self.fc = nn.Linear(1, 1)   # bias=True by default
    def forward(self, x):
        return self.fc(x)

model = Tiny()
y = model(torch.randn(4, 1))      # batch of 4 → shape (4, 1)
```

`nn.Linear(4, 3)` on `x` with shape `(5, 4)` → `(5, 3)`. Batch size 5 never appears in the Linear constructor.

### Analogy — LEGO kit with a manual

`nn.Module` is a **kit**: pieces (`Linear`) + instructions (`forward`). The optimizer only turns knobs the kit **registered**. Skip `super().__init__()` and the kit never stamps the pieces as official — `model.parameters()` can look empty and SGD does nothing useful.

Manual GD (Topic 5) is hand-tightening bolts. `optimizer.step()` is the powered screwdriver on the whole kit.

### Notice

- Always call **`super().__init__()`**.  
- Use `model(x)`, which calls `forward` (do not usually call `forward` by hand in training code).  
- `model.parameters()` / `named_parameters()` feed the optimizer and debugging.  
- `model.to(device)` moves **all** registered parameters.

### Mini-check

1. What method defines the computation?  
2. What does `nn.Linear(3, 5)` map?  
3. Why is batch size missing from `nn.Linear(...)`?

---

## 7. Activations and CrossEntropy (logits)

<a id="p7-loss"></a>

### Purpose for the video

Stacked Linear layers without nonlinearities are still one big affine map — activations make deep nets useful. For multi-class classification, **CrossEntropy** is the standard loss, and its input convention is a classic footgun.

### Definitions

| Piece | Meaning |
|-------|---------|
| **ReLU / sigmoid / tanh** | Elementwise nonlinearities (`F.relu`, `torch.sigmoid`, `torch.tanh`) |
| **Logits** | Raw class scores **before** softmax |
| **Softmax** | Turns scores into a probability vector (sums to 1 along class dim) |
| **`nn.CrossEntropyLoss`** | Expects **logits** + **integer class labels**; applies log-softmax internally |
| **`dim=1`** | For shape `(B, C)`, class axis is 1 — use that for softmax / argmax |

### Worked micro

```python
import torch.nn.functional as F
logits = torch.tensor([[1.0, 2.0, 0.5]])   # one example, 3 classes
probs = F.softmax(logits, dim=1)           # for display / sampling
# criterion = nn.CrossEntropyLoss()
# loss = criterion(logits, torch.tensor([1]))  # class index 1 — NOT probs
```

ReLU on `[-2, -1, 0, 1, 2]` → `[0, 0, 0, 1, 2]`.

### Analogy — contest scores vs prize shares

Logits = raw **judge scores**. Softmax = **prize shares** that sum to 1. CrossEntropy scolds you if the true winner’s share is tiny: roughly $-\log p_{\mathrm{true}}$. The CE module wants the **raw scores**, not the already-shared prizes — it does the sharing **inside**. Softmax-then-CE is **double sharing**: wrong scale, bad gradients.

### Notice

- Targets are **class indices** (`0`, `1`, …), not one-hot vectors.  
- MSE is natural for regression; CE for multi-class classification.  
- Binary classification can use 1 logit + BCE **or** 2 logits + CE (the tutorial uses 2 + CE).  
- Last MLP layer usually has **no** activation when CE is next.

If this still blurs, StatQuest’s Softmax and Cross Entropy shorts, then the official CrossEntropyLoss page.

### Mini-check

1. Does `CrossEntropyLoss` want probabilities or logits?  
2. Name three activations shown in the tutorial family.  
3. For logits shape `(16, 3)`, which `dim` for `argmax` class prediction?

---

## 8. Dataset vs DataLoader

<a id="p8-data"></a>

### Purpose for the video

Separate **what one example is** from **how batches are served**. Same pattern for MNIST toys and your private company data.

### Definitions

| Object | Role |
|--------|------|
| **`Dataset`** | Implements `__len__` and `__getitem__(idx)` → **one** sample (and label) |
| **`DataLoader`** | Wraps a dataset: **batch_size**, **shuffle**, iteration over batches |
| **Batch** | Stack of several samples for one forward/backward |
| **`shuffle=True`** | Reshuffle example order each epoch so batch membership changes |

### Worked micro

```python
from torch.utils.data import Dataset, DataLoader

class Toy(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

loader = DataLoader(Toy(X, y), batch_size=16, shuffle=True)
for xb, yb in loader:
    # last batch may have length < 16 if N is not divisible by 16
    ...
```

Public toys (MNIST, Fashion-MNIST, CIFAR-10) often ship ready-made in **torchvision** — mentioned in the lecture. Real apps still use this same three-method Dataset contract.

### Analogy — library vs book cart

**Dataset** = the library catalog (item by index).  
**DataLoader** = a cart that brings **16 books at a time** and can reshuffle the queue each day (epoch). Without shuffle, the model always sees the same seating chart and can overfit order. The last trip of the cart may carry fewer than 16 books — models stay batch-size agnostic, so that is fine.

### Notice

- `shuffle=True` mixes examples so the model does not memorize “batch 0 is always indices 0–15.”  
- Models do **not** hard-code batch size 16 in `nn.Linear`.  
- Move **each batch** to `device` inside the loop (DataLoader does not magically use CUDA).

### Mini-check

1. Which object returns one `(x, y)`?  
2. What does `shuffle=True` change?  
3. Why might the last batch have shape `(4, …)` when `batch_size=16`?

---

### Paper check

1. Choose device CUDA-if-available.  
2. What does `requires_grad=True` enable?  
3. Five steps of a train iteration (including mode)?  
4. CE loss input: logits or softmax probs?  
5. Difference between Dataset and DataLoader?  
6. Why `.to(device)` on batches even after `model.to(device)`?

**Peek:** (1) `torch.device("cuda" if ...)` (2) autograd graph tracking (3) `train` → zero_grad → forward → backward → step (4) **logits** (5) one sample vs batched iterator (6) loader tensors start on CPU.

---

**Second teachers (names only here).** Official PyTorch docs, 3Blue1Brown, StatQuest. The actual pointers live at the end of [NOTES.md](./NOTES.md#external-references) — a short famous list, not a link dump.

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz: [quiz.html](./quiz.html) Part A = this file.
