# Prerequisites — warm-up before W1_T2 (PyTorch Dataset / DataLoader)

> **Do this first** if “tensor,” “batch,” “`__getitem__`,” or “shuffle” still freeze you.  
> Then [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> IITM BS · Week 1 Tutorial 2 · TA live-codes (Chandan).  
> **Beginner:** purpose · definition · micro · analogy · ASCII · notice · mini-check.

YouTube **title** says “tensors.” This **recording** is **18 minutes**, not a one-hour lab, and it spends almost the whole tape on **Dataset and DataLoader** (Fashion-MNIST + a custom class). Warm-up matches **what he types**, not the title.

```
  After this warm-up you can say:

  "A tensor is a numbered grid of numbers (PyTorch’s ndarray that can sit on GPU)."
  "Train data teaches the map; test data grades it once."
  "A class label is an integer 0–9, not the word 'dress'."
  "Dataset hands one (x, y); DataLoader hands a batch of 64."
  "Subclass Dataset: __init__, __len__, __getitem__."
  "CSV column 0 is the path; column 1 is the label — do not swap."
  "A batch is 64 pairs, not 64 raw JPEGs."
  "Shuffle on train; usually not on test."
```

```
  §1  Tensor = numbered grid            ──► Topics 1, 4
  §2  Train vs test                     ──► Topic 4
  §3  Class / label 0–9                 ──► Topics 2, 5
  §4  Dataset vs DataLoader             ──► Topics 3, 8
  §5  Inheritance and dunder methods    ──► Topic 7
  §6  CSV rows and iloc                 ──► Topics 6–7
  §7  Batch of 64                       ──► Topic 8
  §8  Shuffle and epochs                ──► Topic 8
```

**One scene through all eight.** A **clothing warehouse** with 10 bins (T-shirt … ankle boot). Photos of garments live in drawers. The training robot only lifts **numbered grids** in **carts of 64**, each garment tagged with a **bin number**.

```
  ┌────────── WAREHOUSE ──────────┐
  │  drawers: JPEG / PNG files    │
  │  sticker: integer 0–9 (bin)   │
  │  robot food: TENSOR           │
  │  one trip: BATCH of 64 pairs  │
  └──────────────┬────────────────┘
                 │
     lucky: public catalog already in PyTorch
     unlucky: your private drawer + packing list (CSV)
                 │
                 ▼
           Dataset  →  DataLoader  →  carts D1, D2, … Dk
```

---

## 1. A tensor is a numbered grid

<a id="p1-tensor"></a>

**Purpose.** He starts: all data must be a tensor before weights can update.

**Definition.** A **tensor** is a multi-dimensional array of numbers with a **shape** (how many slots along each axis) and a **dtype**. A grayscale 28×28 photo is a tensor of shape `(1, 28, 28)` or `(28, 28)`. PyTorch’s type is `torch.Tensor`. Unlike a Python list of lists, a tensor supports fast batched math (and later, GPU).

**Micro.** One Fashion-MNIST image: 28 rows × 28 columns of brightness. Flattened that is $28\times 28=784$ numbers. After `ToTensor()`, values are typically in `[0, 1]`, shape often `(1, 28, 28)` = (channel, height, width). A Python list of 784 ints is **not** yet a tensor. A `.png` on disk is **not** yet a tensor either.

Tiny grid you can hold:

```
  2×2 brightness (toy, not Fashion-MNIST):

     0.0  0.5
     1.0  0.2

  as a tensor: shape (1, 2, 2) after ToTensor
  as a JPEG: the robot cannot multiply it yet
```

**Analogy.** A spreadsheet of pixel brightness. “Convert to tensor” = put the spreadsheet in the format the warehouse robot can lift. A JPEG on disk is a **sealed envelope**; `ToTensor` opens it into cells. Handing the robot the envelope is the wrong move.

Second picture: a stack of 64 envelopes is still envelopes. A batch of 64 tensors is what Topic 7–8 produce.

```
  JPEG file  --ToTensor-->  tensor shape (1, 28, 28)
       sealed                    numbered cells
```

**Notice.** This hour barely *creates* tensors by hand (`torch.tensor([[1,2]])` is the **title**, not the tape). It **loads images and converts** them.

**Mini-check.** Is a 28×28 photo already a tensor when it sits as a `.png` on disk? What shape do you expect after `ToTensor`?

---

## 2. Train vs test

<a id="p2-train-test"></a>

**Purpose.** Fashion-MNIST is two piles on purpose.

**Definition.** **Training** data: examples used to **learn** the input–output map (weights). **Test** data: a **held-out** pile used to **evaluate** that map. They must not be mixed. Same disk folder `data/` can hold both; the **`train=` flag** chooses which pile.

**Micro.** `train=True` vs `train=False` in `FashionMNIST`. `root="data"` means a folder named `data` under the **current working directory** (where the notebook lives), not “somewhere on the internet.” `download=True` fills that folder if it is empty.

**Analogy.** Practice exams vs the final. You may reshuffle practice packets every night. You grade the final **once**. If you study the answer key of the final, you destroyed the grade — that is training on test.

Wrong move: two constructors both with `train=True`, then wondering why “test accuracy” looks like training loss.

```
  cwd/
    notebook.ipynb
    data/                    ← root="data"
      FashionMNIST/
         training pile  ← train=True
         test pile      ← train=False
```

```
  practice packets  =  training_data   (learn)
  sealed final      =  test_data       (grade once)
```

**Notice.** He will shuffle train batches; he will say shuffle on test is **not required**.

**Mini-check.** If you train on the test split, what did you destroy? Where on disk does `root="data"` point?

---

## 3. Class labels 0–9

<a id="p3-label"></a>

**Purpose.** Clothes have names; the net wants integers.

**Definition.** A **class** is one of 10 garment types. A **label** `y` is an integer `0…9`. The net does not read the English word “dress.” He pointed at **coat, dress, sandal** and said number them.

**Micro.** Three photos on the counter:

```
  photo A  coat     →  sticker 4
  photo B  dress    →  sticker 3
  photo C  sandal   →  sticker 5
```

The Dataset returns `(tensor_A, 4)`, not `(tensor_A, "coat")`. Two different coats can share label `4`.

**Analogy.** Warehouse bins numbered 0–9. The sticker on a photo is the bin number, not a paragraph. If you write `"dress"` in the CSV instead of `3`, `__getitem__` will hand the net a string and training will explode.

Wrong: a 10-class problem with labels `1…10`. PyTorch classifiers almost always want `0…9` (ten bins, zero-based).

```
  photo  -->  tensor x          (pixels)
  bin    -->  integer y in {0,1,...,9}

  human: "that's a dress"
  net:   y = 3
```

**Notice.** Custom CSV column 1 must be that integer (or something `target_transform` can map to it).

**Mini-check.** Can two different photos share the same label? What goes in column 1 of the CSV — `"coat"` or `4`?

---

## 4. Dataset vs DataLoader

<a id="p4-dataset-loader"></a>

**Purpose.** Two objects, two jobs. Mixing them is the #1 beginner freeze.

**Definition.** A **`Dataset`** knows: how many examples (`__len__`) and how to fetch example `i` (`__getitem__`) as `(x, y)`. A **`DataLoader`** **wraps** a Dataset and **groups** examples into **batches**, optionally **shuffles**, and gives you an **iterator**.

**Micro.** Dataset: “give me item 17” → one coat tensor and label `4`. DataLoader: “give me the next plate” → 64 such pairs stacked. Fashion-MNIST train has 60,000 examples, so `len(dataset)=60000` and `len(dataloader)≈60000/64=937` full batches (plus a remainder).

**Analogy.** Dataset = the **catalog** of every garment photo. DataLoader = the **cart** that brings 64 tagged photos per trip. Asking the catalog for item 17 is not the same as rolling a cart. Filling 937 carts by writing `for i in range(60000)` yourself is what the loader exists to avoid.

```
  Dataset[i]           →  (x_i, y_i)          one garment
  next(DataLoader)     →  batch of 64 pairs   one cart

  training loop talks to the CART, not the catalog
```

```
  catalog (Dataset)
     │  wrap
     ▼
  cart (DataLoader, 64, shuffle?)
     │  next / for
     ▼
  (images, labels)  shapes ~ (64, 1, 28, 28)  and  (64,)
```

**Notice.** You almost never loop `for i in range(len(dataset))` in training. You loop the loader. Batch size is a **loader** argument, not a Dataset field.

**Mini-check.** Who decides batch size — Dataset or DataLoader? What does `next(iter(loader))` return?

---

## 5. Inheritance and dunder methods

<a id="p5-dunder"></a>

**Purpose.** He assumes OOP. The custom class is a **child** of `Dataset`.

**Definition.** **`__init__`** = constructor (runs once when you create the object). **`self`** = this instance. **`__len__`** = `len(obj)`. **`__getitem__`** = `obj[idx]`. **Inheritance:** `class CustomImageDataset(Dataset)` copies the Dataset *protocol* so DataLoader can call those three methods without knowing your class name.

**Micro.** If you forget `__len__`, `DataLoader` cannot know how many batches exist. `dataset[0]` is Python sugar for `dataset.__getitem__(0)`. He also passes `transform` (on **X**, the photo) and `target_transform` (on **Y**, the sticker), both default **None**.

**Analogy.** Dataset is a job description: “must answer how many, and fetch #i.” Your custom class is an employee who signed that contract. Renaming the methods `size` and `get` is like changing your job title so the warehouse radio cannot find you.

Wrong: a class that loads all 60,000 images in `__init__`. `__init__` should remember **paths**; `__getitem__` reads **one** file when asked.

```
  Dataset          (parent / protocol)
     ▲
  CustomImageDataset
     __init__(csv, folder, transform=None, target_transform=None)
     __len__        →  how many rows
     __getitem__(i) →  (tensor, label)
```

```
  Python sugar
     len(ds)   ==  ds.__len__()
     ds[0]     ==  ds.__getitem__(0)
```

**Notice.** He says those methods should be “almost as they are” — don’t rename them.

**Mini-check.** What Python expression calls `__getitem__(self, 0)`? Does `__init__` read every JPEG?

---

## 6. CSV rows and `iloc`

<a id="p6-csv"></a>

**Purpose.** Custom data = folder of files + a table of names and labels.

**Definition.** A **CSV** here has **two columns**: (0) relative path / filename, (1) label. **`pandas.read_csv`** loads it into a table (DataFrame). **`.iloc[idx, col]`** = row `idx`, column `col`, **0-based**. `os.path.join(folder, relative_path)` builds a **full** path the OS can open.

**Micro.** Three rows:

```
  annotations.csv          images/
  relpath,        label      coat/001.png
  coat/001.png,   4          dress/019.png
  dress/019.png,  3          sandal/007.png
  sandal/007.png, 5
```

`idx=0`: `iloc[0,0]` → `coat/001.png`; join with `images/` → `images/coat/001.png`; `iloc[0,1]` → `4`.

**Analogy.** A packing list: column A = “which photo,” column B = “which bin.” Swap A and B and you try to open a file named `"4"` — `read_image` dies. Absolute `C:\Users\...` paths break on another laptop; **relative** + `join` is the point.

```
  iloc[idx, 0] = path     (join with img_dir)
  iloc[idx, 1] = label    (integer 0–9)

  join("images", "coat/001.png")  →  images/coat/001.png
```

**Notice.** He allows other modalities (even a sequence) but still two columns: identity of the example, and its label.

**Mini-check.** If you write `iloc[idx, 1]` for the path, what breaks? Why `join` instead of string `+` on Windows vs Linux?

---

## 7. A batch of 64

<a id="p7-batch"></a>

**Purpose.** Weight updates use **groups**, not one photo. He repeats **64**.

**Definition.** **`batch_size=64`** means each step of the loader returns 64 examples **stacked**: `x` shape like `(64, 1, 28, 28)` and `y` shape `(64,)`. He draws the split `D1, D2, … Dk` where each **Di** is tuples `(x_j, y_j)` for `j=1…64`.

**Micro.** 192 train images → $192/64=3$ full carts (`D1,D2,D3`). 200 images → three full carts plus a short last cart of 8 (he does not dwell on the remainder; know it exists). 60,000 Fashion-MNIST train images → 937 batches of 64 plus leftover.

**Analogy.** One shopping cart holds 64 tagged garments. The clerk prices the **cart**, not one sock. Filling 64 carts by hand in a `for i` loop is what DataLoader exists to avoid.

Wrong: thinking batch 64 means “64 pixels” or “64 classes.” It is 64 **examples**.

```
  one item:     (x, y)                    shapes (1,28,28)  and  ()
  one batch:    64 of those, stacked      (64,1,28,28)     and  (64,)

  D1 | D2 | D3 | ... | Dk
  └── 64 pairs each (last may be shorter)
```

**Notice.** Dataset never saw “64.” The **loader** adds batching.

**Mini-check.** If you have 128 train images and batch 64, how many train batches per epoch (ignore remainder)? What is the first dimension of `images` after `next(iter(loader))`?

---

## 8. Shuffle and epochs

<a id="p8-shuffle"></a>

**Purpose.** His interesting argument: `shuffle=True`.

**Definition.** **Shuffle** = randomize example order when forming batches. **Epoch** = one **full pass** over training data (all carts D1…Dk once). He shuffles **train** so batches are **not the same order every time**. **Test:** you evaluate **once**; multiple epochs on test are not a thing; **`shuffle=False` makes sense**.

**Micro.** Night 1 train order: carts 7, 2, 9, … Night 2: 3, 11, 1, … Same 60,000 photos, new trips. Test night: one sitting, same order is fine.

**Analogy.** Practice: reshuffle the flashcards every night so you do not memorize “card 3 always follows card 2.” Final exam: one sitting; reshuffling between “epochs” you will not run is theatre.

Wrong: `shuffle=True` on test “because the official snippet did,” then watching accuracy jitter for no modeling reason.

```
  train DataLoader(..., shuffle=True)   # new cart order each epoch
  test  DataLoader(..., shuffle=False)  # his advice: one eval

  epoch 1:  D3, D1, D2, ...
  epoch 2:  D2, D5, D1, ...     (train only)
```

**Notice.** Official PyTorch sample often sets test `shuffle=True` too. This recording prefers False on test. Follow **this** tape for the quiz.

**Mini-check.** Why would repeating the same train batch order every epoch be a bad habit? How many times does he want you to walk the test set?

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz later: [quiz.html](./quiz.html) Part A = this file.
