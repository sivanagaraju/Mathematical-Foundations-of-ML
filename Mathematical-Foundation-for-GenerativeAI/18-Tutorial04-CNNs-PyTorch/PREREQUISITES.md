# Prerequisites — warm-up before Tutorial 4 (CNNs using PyTorch)

> **Do this first** if images, convolution, pooling, or NCHW shapes still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Course: NPTEL Mathematical Foundations of Generative AI · Tutorial 4.  
> Builds on [Tutorial 3 PyTorch Basics](../17-Tutorial03-PyTorch-Basics/NOTES.md).  
> **Beginner deep warm-up:** definition · micro example · analogy · notice · mini-check.

```
  After this warm-up you can say:

  "Images are tensors with batch, channel, height, width (NCHW)."
  "A conv filter slides a small window, multiplies, and sums into a feature map."
  "Out channels = number of filters; each filter depth matches in channels."
  "Padding and stride control the output height and width."
  "MaxPool halves spatial size (with 2×2, stride 2) and has no weights."
  "CNN extracts features; flatten + Linear head outputs class logits."
  "Train with mini-batches; one epoch = many batch updates."
  "Eval uses model.eval() and torch.no_grad(); score with argmax accuracy."
```

**Warm-up → tutorial boxes**

```
  §1  Images as NCHW tensors              ──► Topics 1–2, 4
  §2  What a convolution does             ──► Topics 2–4
  §3  Filters, channels, params           ──► Topics 2–4
  §4  Padding, stride, output size        ──► Topics 3–5
  §5  Max pooling                         ──► Topics 5–6
  §6  Feature extractor + classifier head ──► Topics 6–7
  §7  Mini-batch train vs full-batch      ──► Topics 8–9
  §8  Eval mode, no_grad, accuracy        ──► Topic 10
```

---

## 1. Images as NCHW tensors

<a id="p1-nchw"></a>

### Purpose for the video

PyTorch vision layers expect a 4-D tensor, not a 2-D matrix of pixels.

### Definitions

| Axis | Name | Meaning |
|------|------|---------|
| **N** | batch | how many images in one forward |
| **C** | channels | 1 = grayscale; 3 = RGB |
| **H** | height | rows of pixels |
| **W** | width | columns of pixels |

### Worked micro

```python
import torch
# 8 RGB images, each 32×32  →  NCHW
x = torch.randn(8, 3, 32, 32)
print(x.shape)          # torch.Size([8, 3, 32, 32])
print("batch", x.shape[0], "channels", x.shape[1])
print("H×W", x.shape[2], "×", x.shape[3])

# One MNIST-like gray image (no batch yet)
digit = torch.randn(1, 28, 28)           # CHW
batch = digit.unsqueeze(0)               # NCHW → (1, 1, 28, 28)
```

MNIST digits are naturally **1×28×28**. This lecture often **resizes to 32×32** and designs a network for **3 channels** (CIFAR-style). When you train on real MNIST you must either set `in_channels=1` **or** repeat the gray channel to 3 (`x.repeat(1, 3, 1, 1)`).

### Analogy — photo album stack

One photo is a grid of pixels. A **batch** is a **stack of photos** on your desk. **Channels** are color plates under the same photo (or one gray plate). NCHW is the filing order: “how many photos, how many plates, how tall, how wide.” If you file plates last (HWC), the librarian (`Conv2d`) cannot find the right folder.

### Notice

- OpenCV/matplotlib often use **HWC**. PyTorch Conv2d wants **CHW** per image, **NCHW** with batch.  
- Tutorial 3’s `permute` skill is exactly for HWC → CHW (e.g. `img.permute(2, 0, 1)` then `unsqueeze(0)` for batch).  
- The **first** axis is almost always batch in training code — never put height first for Conv2d.

### Mini-check

1. What does shape `(64, 1, 28, 28)` mean?  
2. Which axis is batch?  
3. How would you turn `(28, 28, 1)` HWC into NCHW with batch size 1?

---

## 2. What a convolution does

<a id="p2-conv"></a>

### Purpose for the video

Convolution is the core image operation: a small **filter** (kernel) slides across the image and produces a **feature map**.

### Definitions

| Term | Meaning |
|------|---------|
| **Kernel / filter** | Small learnable weight tensor (e.g. 3×3 per input channel) |
| **Feature map** | One 2-D output plane produced by one filter |
| **Receptive field** | The input patch that influences one output location |

### Micro (1 channel, 1 filter)

Imagine a 5×5 gray image and a 3×3 filter. At each place, multiply the 9 numbers by the 9 weights, **add them up**, add a bias → one number. Slide right/down → fill the output map.

Tiny numeric sketch (one location, no pad):

```
  patch 3×3:     filter 3×3:      products:
  1 2 0          1 0 −1           1 0  0
  1 0 1          1 0 −1           1 0 −1
  0 2 1          1 0 −1           0 0 −1
  sum of products = 1+0+0 + 1+0−1 + 0+0−1 = 0  →  one output cell
```

With **3 input channels**, do that mul-sum **per channel**, then **add the three scalars** (+ bias) → still **one** output cell for that filter at that location.

### Analogy — stamp rolling over a stamp pad

The filter is a rubber stamp pattern. Rolling it across the image “inks” how much the local patch matches that pattern. Edge detectors, blob detectors, texture detectors are all filters — **learned from data**, not hand-drawn. RGB means the stamp is **three layers thick** so it can smell red, green, and blue at once.

### Notice

- Convolution is **local** (only a patch) and **shared** (same stamp everywhere).  
- That is cheaper and more image-sensible than a full MLP on all pixels.  
- One filter → one **feature map** (2-D). Sixteen filters → sixteen maps stacked as channels.

### Mini-check

1. Does one filter produce one or many feature maps?  
2. Is the filter different at every pixel location? (weight sharing)  
3. On RGB, is a “3×3 filter” really only 9 weights?

---

## 3. Filters, channels, and parameter count

<a id="p3-filters"></a>

### Purpose for the video

Students mix up “channels” and “filters.” The lecture’s CS231 GIF is built for this.

### Rules of thumb

| Quantity | Equals |
|----------|--------|
| **Depth of each filter** | **in_channels** (Din) |
| **Number of filters** | **out_channels** (K) |
| **Weights (no bias)** | $F \times F \times D_{\mathrm{in}} \times K$ |
| **Biases** | $K$ (one per filter) |

### Worked micro

RGB image, 16 filters of size 3×3:

- Each filter: $3 \times 3 \times 3 = 27$ weights  
- All filters: $27 \times 16 = 432$ weights  
- Plus 16 biases → **448** parameters  

```python
import torch.nn as nn
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
n = sum(p.numel() for p in conv.parameters())
print(n)  # 448
```

### Analogy — orchestra mics

Input channels = microphones (R,G,B). Each **filter** is a mixing board preset that listens to **all** mics in a small spatial window. **16 filters** = 16 different mix presets → 16 output tracks (feature maps).

### Notice

- `out_channels=16` does **not** mean 16× bigger spatial image; it means 16 feature planes.  
- Spatial size is controlled by padding/stride (next section).

### Mini-check

1. Kernel depth for a 1-channel MNIST input?  
2. Params of `Conv2d(1, 8, 3)` ignoring bias? $3\times3\times1\times8=72$

---

## 4. Padding, stride, and the output-size formula

<a id="p4-shape"></a>

### Purpose for the video

Every CNN notebook eventually asks: “what is H after this layer?”

### Formula (lecture)

$$
H_{\mathrm{out}} = \frac{H_{\mathrm{in}} - F + 2P}{S} + 1
$$

Same for $W$. $D_{\mathrm{out}} = K$ (number of filters).

| Symbol | Meaning |
|--------|---------|
| **F** | kernel size (assume square) |
| **P** | padding (pixels of zeros on each side) |
| **S** | stride |

### Worked micros

1. $H=5, F=3, P=1, S=2$ → $(5-3+2)/2+1 = 3$  
2. $H=32, F=3, P=1, S=1$ → $(32-3+2)/1+1 = 32$ (same size)  
3. MaxPool $H=32, F=2, P=0, S=2$ → $(32-2)/2+1 = 16$ (half)  
4. Trap: $H=32, F=3, P=0, S=1$ → $(32-3)/1+1 = 30$ (forgot pad)

```python
import torch.nn as nn
conv_same = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
print(conv_same(torch.randn(2, 3, 32, 32)).shape)  # (2, 16, 32, 32)
```

### Analogy — window washing on a skyscraper

- **Kernel F:** how wide your squeegee is.  
- **Stride S:** how far you step sideways.  
- **Padding P:** blank canvas taped around the window so the squeegee can center on edges.  

Big stride → fewer positions → smaller output grid. No padding on a tall thin window means you never clean the outer rim — $H$ shrinks.

### Notice

- Defaults in `nn.Conv2d`: **stride=1**, **padding=0** if you omit them.  
- “Same” spatial size usually means **odd F** with **P = (F−1)/2** and S=1 (e.g. 3×3, pad 1).  
- Always compute $H_{\mathrm{out}}$ **before** you write the next Linear flatten size.

### Mini-check

1. Compute $H_{\mathrm{out}}$ for 28, F=5, P=0, S=1.  
2. Why pad=1 with 3×3 and stride 1 keeps H?  
3. What is $H_{\mathrm{out}}$ if you forget pad on 32 with F=3, S=1?

---

## 5. Max pooling

<a id="p5-pool"></a>

### Purpose for the video

Pooling shrinks maps so deeper layers see a larger fraction of the image with fewer pixels.

### Definitions

| Term | Meaning |
|------|---------|
| **MaxPool2d** | In each window, keep the **maximum** activation |
| **Standard 2×2, s=2** | Non-overlapping windows → **half** H and W |
| **No parameters** | No weights; pure downsample |

### Worked micro

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# (N, C, 32, 32) → (N, C, 16, 16)  channels unchanged
```

Example 2×2 block with values `{1, 3; 2, 6}` → max is **6**.

### Analogy — summarizing a neighborhood vote

Instead of storing every house’s number, you only keep the **loudest** house on each block. You lose fine location detail; you gain a smaller map and some noise robustness.

### Notice

- Channels **do not** change under pooling. If you see 16 before and 16 after, that is expected — not a coincidence with spatial 16.  
- Average pooling exists; this lecture uses **max**.

### Mini-check

1. Params in MaxPool2d?  
2. Shape after pool on `(8, 16, 32, 32)` with 2,2?

---

## 6. Feature extractor + classifier head

<a id="p6-head"></a>

### Purpose for the video

A modern image classifier is almost always **two blocks**:

1. **Feature extractor** — stack of Conv → activation → (Pool)  
2. **Classifier head** — Flatten → Linear → … → **logits**

### Shape story (this lecture’s SimpleCNN)

```
(N, 3, 32, 32)
  Conv 3→16, k=3, p=1  → (N, 16, 32, 32)
  ReLU                 → same shape
  MaxPool 2            → (N, 16, 16, 16)
  Conv 16→32, k=3, p=1 → (N, 32, 16, 16)
  ReLU
  MaxPool 2            → (N, 32, 8, 8)
  Flatten              → (N, 32*8*8) = (N, 2048)
  Linear 2048→128 → 10 → logits (N, 10)
```

### Analogy — detective then judge

Convs are detectives collecting local clues (edges, blobs, textures). Flatten is bagging the evidence. Linear layers are the judge that outputs a **scorecard** (logits) per class. Softmax turns scores into shares; **CrossEntropy** wants the raw scores.

### Notice

- Last Linear outputs **logits**, not probabilities, when you train with `nn.CrossEntropyLoss` (Tutorial 3).  
- `x.view(x.size(0), -1)` keeps batch and flattens the rest.

### Mini-check

1. Why flatten before Linear?  
2. What is $32 \times 8 \times 8$?

---

## 7. Mini-batch training vs full-batch

<a id="p7-batch"></a>

### Purpose for the video

Tutorial 3’s tiny linreg could feed all 4 points at once. MNIST has **60,000** train images — you must **batch**.

### Definitions

| Term | Meaning |
|------|---------|
| **Batch** | Subset of examples (e.g. 64) in one forward/backward |
| **Epoch** | One full pass over the **training set** |
| **Iteration / step** | One weight update (usually **one batch**) |
| **Updates per epoch** | $\approx N / \text{batch_size}$ (e.g. 60000/64) |

### Pattern (same as Tutorial 3, but inside a loader loop)

```python
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    logits = model(images)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Analogy — grading a class of 60,000

You do not grade every paper before changing your teaching plan. You grade a **stack of 64**, update how you teach, then take the next stack. One **epoch** = you have seen every paper once; you may have changed your plan hundreds of times that day.

### Notice

- **One epoch ≠ one update.** Multiple batches → multiple updates.  
- `shuffle=True` on the train loader remixes stacks each epoch.

### Mini-check

1. ~How many updates in one epoch with 60k samples, batch 64?  
2. Does test loader need shuffle?

---

## 8. Eval mode, `no_grad`, accuracy

<a id="p8-eval"></a>

### Purpose for the video

Training and evaluation are different **modes** and different **metrics**.

### Definitions

| Piece | Role |
|-------|------|
| **`model.train()`** | Training mode (Dropout/BatchNorm behave for train) |
| **`model.eval()`** | Evaluation mode |
| **`torch.no_grad()`** | Do not build autograd graph (faster, less memory) |
| **Accuracy** | $\#\{\hat y = y\} / N$ for balanced multi-class |
| **argmax(dim=1)** | Class index from logits shape `(N, C)` |

### Worked micro

```python
model.eval()
with torch.no_grad():
    logits = model(images.to(device))
    pred = logits.argmax(dim=1)
    correct += (pred == labels.to(device)).sum().item()
```

### Analogy — open-book vs closed-book

Training is open-book practice with a red pen (gradients). Evaluation is a **closed-book exam**: fixed weights, no red pen, just mark right/wrong. Accuracy is the fraction of green checks.

### Notice

- Classification loss (CE) is **not** the same as accuracy.  
- For skewed classes you may need precision/recall/F1 (lecture mentions; MNIST is balanced).  
- Train accuracy can be higher than test accuracy — expected.

### Mini-check

1. Why `no_grad` at test time?  
2. Why `dim=1` for argmax on `(N, 10)` logits?

---

### Paper check

1. Write the NCHW meaning of `(16, 3, 32, 32)`.  
2. Params of `Conv2d(3, 8, 3)` with bias?  
3. $H_{\mathrm{out}}$ for 32, F=3, P=1, S=1?  
4. What does MaxPool2d(2,2) do to `(N,C,32,32)`?  
5. Flatten size after two pools from 32 with this lecture’s stack?  
6. Epoch vs iteration?

**Peek:** (1) 16 RGB 32×32 images (2) $3\cdot3\cdot3\cdot8+8=224$ (3) 32 (4) `(N,C,16,16)` (5) $32\times8\times8=2048$ (6) full pass vs one update.

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz: [quiz.html](./quiz.html) Part A = this file.
