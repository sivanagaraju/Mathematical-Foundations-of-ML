# Prerequisites — warm-up before Tutorial 6 (Transfer Learning with PyTorch)

> **Do this first** if pretrained models, fine-tuning, or ImageNet 224 still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Course: NPTEL Mathematical Foundations of Generative AI · Tutorial 6.  
> Builds on [Tutorial 4 CNNs](../18-Tutorial04-CNNs-PyTorch/NOTES.md) and [Tutorial 5 RNNs](../19-Tutorial05-RNNs-PyTorch/NOTES.md).  
> **Beginner deep warm-up:** definition · micro example · analogy · notice · mini-check.

```
  After this warm-up you can say:

  "PyTorch layers are LEGO blocks if shapes match."
  "ImageNet models expect about 224×224×3 inputs."
  "Transfer learning reuses features learned on a big dataset."
  "Fine-tuning continues training from pretrained weights."
  "Usually we replace only the last classifier for our C classes."
  "Train transforms can flip; test usually only resize + normalize."
  "ImageFolder builds a Dataset from one folder per class."
  "AlexNet/VGG use classifier[-1]; ResNet uses model.fc."
```

**Warm-up → tutorial boxes**

```
  §1  Layers as LEGO (shapes)              ──► Topics 1, 10
  §2  ImageNet size 224 & 1000 classes     ──► Topics 2–4, 8
  §3  Transfer learning vs from scratch    ──► Topics 6–7
  §4  Fine-tuning & the classification head ──► Topics 3, 9
  §5  AlexNet / VGG / ResNet at a glance   ──► Topics 2, 4, 5
  §6  Train vs test transforms             ──► Topic 8
  §7  ImageFolder data layout              ──► Topic 9
  §8  Replace last Linear in code          ──► Topics 9–10
```

---

## 1. Layers as LEGO (shape matching)

<a id="p1-lego"></a>

### Purpose for the video

The instructor’s first slogan: **modules are LEGO blocks**. Programming-wise you may stack almost anything if **tensor shapes line up**.

### Definitions

| Idea | Meaning |
|------|---------|
| **Module / layer** | A callable `nn` piece (`Conv2d`, `Linear`, …) |
| **Shape contract** | Output of block $i$ must match input of block $i+1$ |
| **Mismatch** | Runtime error (or silent nonsense) when dims disagree |

### Worked micro

```python
import torch
import torch.nn as nn
x = torch.randn(2, 3, 224, 224)
y = nn.Conv2d(3, 64, 3, padding=1)(x)
# y: (2, 64, 224, 224) — next layer must accept 64 channels
z = nn.Linear(64 * 224 * 224, 10)(y.flatten(1))  # OK after flatten
```

### Analogy — LEGO studs

Every brick has **studs on top** (output shape) and **holes underneath** (input shape). Wrong stud count → the tower falls. Whether the tower is a *good building* is architecture taste; the **stud rule** is non-negotiable.

### Notice

- “Can I connect A to B?” is a **shape** question.  
- “Should I?” is a **design** question. The lecture separates both.  
- Pretrained nets are just **big pre-snapped subassemblies** — you still check studs when you attach a new head.  
- Vision batches are usually **NCHW**: `(batch, channels, height, width)` — e.g. `(32, 3, 224, 224)`. A `Linear` head wants a flat vector per sample, so something must **flatten** or **global-pool** first.

### Mini-check

1. What fails if `Linear(1000, 4)` receives a vector of length 4096?  
2. Name one shape axis `Conv2d` cares about.  
3. Is “shape-legal” the same as “good architecture”?  
4. Write the four letters of the batch layout `32×3×224×224` uses in PyTorch.

---

## 2. ImageNet size 224 and 1000 classes

<a id="p2-imagenet"></a>

### Purpose for the video

Classic vision backbones were built for **ImageNet**: roughly **224×224 RGB** images and a **1000-way** classifier.

### Definitions

| Term | Meaning |
|------|---------|
| **ImageNet** | Huge labeled image collection (~1000 classes) |
| **224×224×3** | Common spatial/channel input for AlexNet/VGG/ResNet family demos |
| **1000 logits** | Final head size for original ImageNet models |

### Worked micro

```python
# Conceptual pretrained head
fc = nn.Linear(4096, 1000)  # AlexNet/VGG-style
# Your MRI task might need:
fc_mri = nn.Linear(4096, 4)
```

### Analogy — airport baggage template

The “standard suitcase slot” on the conveyor is **224×224×3**. If your bag is a weird size, either **resize the bag** to fit the slot or rebuild the whole airport. Pretrained weights assume the standard slot.

### Notice

- Lecture: ~**1300 images/class**, whole set ~**200 GB** scale — why we download **weights**, not retrain ImageNet.  
- Softmax turns 1000 logits into a probability vector over classes.  
- “224” is a **convention of these checkpoints**, not a law of physics. Other models use 299, 384, … — always match the **weights** you load.

### Mini-check

1. Why is 224×224×3 “important”?  
2. What does the final 1000 mean on AlexNet?  
3. If you refuse to resize, what must you rewrite?

---

## 3. Transfer learning vs training from scratch

<a id="p3-transfer"></a>

### Purpose for the video

**Transfer learning** = reuse knowledge (weights) from a source task on a target task.

### Definitions

| Mode | Start weights | Typical use |
|------|---------------|-------------|
| **From scratch** | Random | Lots of labeled target data, different domain |
| **Transfer / fine-tune** | Pretrained (e.g. ImageNet) | Smaller target data; related visual features |

### Analogy — bicycle → motorcycle (lecture)

Learning to **balance** on a bicycle is the hard part. On a motorcycle you mostly learn **controls**, not balance again. ImageNet already taught the net “edges, textures, parts”; your MRI task mostly teaches “which tumor patterns matter.”

Second lecture analogy: riders who know **clutch + gear** on bikes adapt faster to **manual cars** than moped-only riders.

### Notice

- Transfer is not magic if domains differ wildly and you refuse to adapt the head/input.  
- **Fine-tuning** = continue gradient steps from the transferred weights (backbone may move a little or a lot).  
- Optional mode: **feature extraction** — freeze backbone (`requires_grad=False`), train **only** the new head (cheaper, less risk of wrecking good features on tiny data).  
- Lecture demos mostly **load weights + swap head + train** (full or near-full fine-tune narrative), not a long freeze-only lesson — but you should know both dials.

```python
# Feature-extract mode (optional): only head learns
for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():  # AlexNet/VGG: classifier[6]
    p.requires_grad = True
# Full fine-tune: leave every p.requires_grad = True (default after load)
```

Think of freeze vs fine-tune as **two gears**: first gear = only teach the new label map; second gear = gently retune the whole camera for hospital lighting.

### Mini-check

1. What is transferred in weight space?  
2. Why not always train from scratch on 200 GB?  
3. Freeze backbone vs fine-tune whole net — which trains fewer parameters?  
4. In one sentence: transfer vs fine-tune.

---

## 4. Fine-tuning and the classification head

<a id="p4-head"></a>

### Purpose for the video

The usual surgical move: keep the **feature extractor**, replace the **classification head** for your $C$ classes.

### Pattern

```
  backbone (conv stack) → vector → Linear(H, 1000)   # old
  backbone (same)       → vector → Linear(H, C)      # new
```

### Worked micro

```python
import torchvision.models as models
m = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
in_f = m.classifier[6].in_features  # often 4096
m.classifier[6] = nn.Linear(in_f, 4)  # MRI: 4 classes
```

### Analogy — camera body + new lens plate

The expensive **camera body** (backbone) already focuses light well. You only swap the **back plate** that maps features to your labels. Rebuild the whole camera only if the lens mount size (input resolution) is wrong.

### Notice

- Prefer **resize inputs to 224** over rewriting every internal layer.  
- If resize is impossible, pretrained weights may not help without heavy redesign.  
- After you assign a **new** `nn.Linear`, that module is born on **CPU** — call `model.to(device)` again (or move the whole net) so head and backbone share a device.

### Mini-check

1. What changes for a 4-class problem?  
2. What do you prefer not to change?  
3. Why re-run `.to(device)` after head surgery?

---

## 5. AlexNet, VGG, ResNet at a glance

<a id="p5-archs"></a>

### Purpose for the video

Enough architecture literacy to load the right model and know where the head lives.

### Snapshot table

| Family | Signature idea | Head location (typical) |
|--------|----------------|-------------------------|
| **AlexNet** | Large early kernels, FC 4096 | `model.classifier[6]` |
| **VGG** | Stacks of 3×3 convs; huge FC | `model.classifier[6]` |
| **ResNet** | **Skip connections** (add residual) | `model.fc` |

### Skip connection (one sentence)

After a small transform block, **add the block’s input** to its output so the network can keep or transform features:

$$
y = F(x) + x
$$

### Analogy — detour road + through road

A skip is a **through lane** next to a construction detour $F(x)$. Traffic can take the detour, the through lane, or a blend (addition).

### Notice

- ResNet-18/34/50… are depth variants of the same idea.  
- VGG parameter counts are **hundreds of millions**.

### Mini-check

1. Where is ResNet’s final Linear usually attached?  
2. What does a skip add?

---

## 6. Train vs test transforms

<a id="p6-tf"></a>

### Purpose for the video

Training may **augment**; evaluation should not invent random flips.

### Typical pair (lecture)

| Split | Ops |
|-------|-----|
| **Train** | Resize(224) → RandomHorizontalFlip → ToTensor → Normalize(ImageNet mean/std) |
| **Test** | Resize(224) → ToTensor → Normalize (same stats) |

### ImageNet stats (standard)

Mean ≈ `(0.485, 0.456, 0.406)`, Std ≈ `(0.229, 0.224, 0.225)` per RGB channel (values as commonly used; lecture cites ImageNet channel stats).

### Analogy — practice vs exam

Train flips = **practice with slightly messy desks**. Test = **exam under fixed lighting** — still resized and normalized so the pretrained eyes recognize the paper.

### Notice

- Augmentations act as **regularizers** (more variety without new labeled scans).  
- Lecture exercise: try different mean/std train vs test and watch metrics.  
- Order matters: **Resize → (optional Flip) → ToTensor → Normalize**. Flipping after Normalize is legal but messier to reason about; lecture uses the standard order above.  
- `ToTensor` maps pixel bytes to floats in roughly `[0, 1]` before ImageNet normalize centers them.

### Mini-check

1. Should test use RandomHorizontalFlip?  
2. Why normalize with ImageNet stats when loading ImageNet weights?  
3. Name one augmentation that is OK on train but not on test in this lecture.

---

## 7. ImageFolder data layout

<a id="p7-folder"></a>

### Purpose for the video

If each class is a **subfolder**, you do not need a custom `Dataset`.

### Layout

```
  data/train/class_a/*.png
  data/train/class_b/*.png
  data/test/class_a/...
```

```python
from torchvision import datasets, transforms
train_ds = datasets.ImageFolder("data/train", transform=train_tf)
# train_ds.classes → list of class names
```

### Analogy — library shelves

Each shelf is a class; every book on the shelf gets that shelf’s label. `ImageFolder` is the librarian who walks shelves for you.

### Notice

- Contrast with annotation-file custom Datasets from earlier tutorials.  
- Batch after loader: often `(B, 3, 224, 224)`.  
- Folder **names** become class strings; integer labels follow **sorted** folder names (`class_to_idx`). Renaming a folder silently renumbers labels — keep train and test trees consistent.  
- MRI demo folders: **glioma / meningioma / notumor / pituitary** under `Training/` and `Testing/`.

### Mini-check

1. When is `ImageFolder` enough?  
2. What attribute lists class names?  
3. Why must train and test use the same folder-name set?

---

## 8. Replace the last Linear in code

<a id="p8-replace"></a>

### Purpose for the video

Know the **exact attribute** to overwrite per architecture.

### Patterns

```python
# AlexNet / VGG
m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)

# ResNet
m.fc = nn.Linear(m.fc.in_features, num_classes)
```

Always read **`in_features`** from the old layer so the width stays correct (e.g. 4096→4, not a guessed size).

### Analogy — power plug adapter

The wall socket is `in_features`. Your country needs a different number of pins (`num_classes`). Swap only the **adapter plate**, not the house wiring (backbone).

### Notice

- After replace, `model.to(device)` then train with CE as usual.  
- Optional advanced: freeze backbone `p.requires_grad=False` (lecture focuses on head swap + fine-tune narrative).  
- Helper pattern in the notebook: `get_alexnet(num_classes)`, `get_resnet18(num_classes)` wrap load + replace so the train dict stays clean.

### Mini-check

1. Why use `in_features` instead of hardcoding 4096 always?  
2. ResNet uses `classifier[6]` or `fc`?  
3. After head swap, do you need to move the **new** Linear to the same device as the backbone?

---

### Paper check

1. LEGO rule for stacking layers?  
2. ImageNet spatial size used here?  
3. Transfer vs fine-tune in one line each?  
4. Train flip vs test flip?  
5. How to point ImageFolder at disk?  
6. AlexNet last layer attribute?  
7. What does `requires_grad=False` on the backbone do?  
8. Typical batch tensor shape after ImageFolder + Resize(224)?

**Peek:** (1) shapes must match (2) 224×224×3 (3) reuse weights; continue training them (4) flip train only (5) `ImageFolder(root, transform=…)` (6) `classifier[6]` (7) freezes backbone (feature extract) (8) `(B, 3, 224, 224)`

---

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html) Part A = this file.
