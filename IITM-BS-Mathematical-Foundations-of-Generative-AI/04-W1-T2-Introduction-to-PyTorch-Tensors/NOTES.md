# W1_T2 — PyTorch Datasets & DataLoaders (Fashion-MNIST & Custom Pipelines)

> **Course:** IIT Madras B.S. Degree in Data Science & AI · **Mathematical Foundations of Generative AI**  
> **Instructor:** Chandan (Tutorial TA) · Course Faculty: Prof. Prathosh A. P. (IISc / IITM)  
> **Tutorial Recording:** [W1_T2 on YouTube](https://www.youtube.com/watch?v=L5n4rNrLZ_8) (~18:26)  
> **Prerequisites Warm-up:** [PREREQUISITES.md](./PREREQUISITES.md) · **Self-Assessment Quiz:** [quiz.html](./quiz.html)  
> **Course Catalog:** [../NOTES.md](../NOTES.md)

---

## 📌 Honest Title Discrepancy & Course Map Placement

> [!NOTE]
> **Video Title vs. Actual Content Discrepancy:**  
> The YouTube recording is titled *"Tutorial 2: Introduction to pytorch: tensors"*. However, the actual 18-minute Colab session opens and walks through the official PyTorch **`Datasets & DataLoaders`** tutorial notebook.  
> * For the pure **Tensor Algebra & Operations** Colab session (tensor creation, indexing, concatenation, dot products, `@` matmul, and autograd), please refer to [02-W1-L3-F-Divergence/NOTES.md](../02-W1-L3-F-Divergence/NOTES.md).  
> * This notes document strictly covers the **speech, code cells, handwritten diagrams, and concepts presented in the 18:26 recording**: loading built-in datasets (`FashionMNIST`), building custom dataset classes (`CustomImageDataset`), and streaming mini-batches via `DataLoader`.

---

## Quick Navigation Matrix

| Topic & Timestamp | Focus Area | Core Code / Concept | Prerequisite Link |
| :--- | :--- | :--- | :--- |
| [Topic 1: Tensors then Dataset Sheet](#topic-1) (00:12–01:53) | Overview & Setup | Transition from Tensor math to Data Ingestion | [p1-tensor](./PREREQUISITES.md#p1-tensor) |
| [Topic 2: Torchvision Catalogs & Fashion-MNIST](#topic-2) (01:53–03:20) | Dataset Catalog | 10-Class Fashion-MNIST, CIFAR-10, ImageNet | [p3-label](./PREREQUISITES.md#p3-label) |
| [Topic 3: Essential Imports](#topic-3) (03:20–04:17) | Module Dependencies | `Dataset`, `DataLoader`, `datasets`, `ToTensor` | [p4-dataset-loader](./PREREQUISITES.md#p4-dataset-loader) |
| [Topic 4: FashionMNIST Train & Test Splits](#topic-4) (04:17–06:09) | Built-In Loading | `root="data"`, `train=True/False`, `download=True` | [p2-train-test](./PREREQUISITES.md#p2-train-test) |
| [Topic 5: Visualizing & When Built-in Fails](#topic-5) (06:09–07:41) | Data Exploration | `matplotlib.pyplot` & Proprietary/Sensitive Data | [p3-label](./PREREQUISITES.md#p3-label) |
| [Topic 6: Custom Datasets: Folder + CSV](#topic-6) (07:41–10:21) | Custom Architecture | `img_dir` (raw images) + `annotations_file` (CSV) | [p6-csv](./PREREQUISITES.md#p6-csv) |
| [Topic 7: The `CustomImageDataset` Class](#topic-7) (10:21–15:06) | OOP Implementation | Subclassing `Dataset`: `__init__`, `__len__`, `__getitem__` | [p5-dunder](./PREREQUISITES.md#p5-dunder), [p6-csv](./PREREQUISITES.md#p6-csv) |
| [Topic 8: `DataLoader`, Batch 64 & Shuffling](#topic-8) (15:06–18:26) | Streaming Engine | Mini-batches $D_i = \{(x_j, y_j)\}_{j=1}^{64}$, `shuffle=True/False` | [p7-batch](./PREREQUISITES.md#p7-batch), [p8-shuffle](./PREREQUISITES.md#p8-shuffle) |
| [Workplace Debugging Scenarios](#workplace-scenarios--debugging-data-pipelines) | Production Systems | Multi-Worker OOM Leaks & CSV Index Inversions | — |
| [External References](#external-references) | Multi-Source Study | 2–3 Curated Videos & 2–3 Guides/Blogs per Topic | — |

---

## Executive Summary & Master Architecture

<a id="executive-summary"></a>
<a id="executive-summary--architecture-of-this-lecture"></a>

Deep neural networks cannot update weights on raw `.png` or `.jpg` image files stored on a hard drive. Weight updates via backpropagation require **numerical tensors grouped into mini-batches of size $B$ (e.g., 64)**.  
PyTorch decouples data processing into two distinct architectural abstractions:
1. **`torch.utils.data.Dataset` (The Item Store):** Maps an integer index `idx` to an individual sample pair $(x_i, y_i)$.
2. **`torch.utils.data.DataLoader` (The Streaming Engine):** Wraps around a `Dataset` to automatically handle mini-batching, multi-process memory loading (`num_workers`), GPU memory pinning (`pin_memory`), and index shuffling (`shuffle=True`).

```
══════════════════════════════════════════════════════════════════════════════════════════════════
                     THE PYTORCH DATA INGESTION ARCHITECTURE BLUEPRINT
══════════════════════════════════════════════════════════════════════════════════════════════════

  1. STORAGE LAYER (On Disk)
     • Built-in: Downloaded into ./data via torchvision.datasets.FashionMNIST
     • Custom:   Raw folder (img_dir) + 2-column CSV (annotations_file: [filename, label])

  2. DATASET ABSTRACTION (Single-Item Access)
     ┌────────────────────────────────────────────────────────────────────────────────────────┐
     │ Class: CustomImageDataset(torch.utils.data.Dataset)                                    │
     │   • __init__(self, csv, dir, transform) ──► Reads CSV into pandas DataFrame table     │
     │   • __len__(self)                       ──► Returns total rows: len(self.img_labels)   │
     │   • __getitem__(self, idx)              ──► Loads 1 image from disk, applies transform,│
     │                                             returns tuple: (image_tensor, label_int)   │
     └────────────────────────────────────────────────────────────────────────────────────────┘

  3. DATALOADER WRAPPER (Batching & Shuffling)
     ┌────────────────────────────────────────────────────────────────────────────────────────┐
     │ DataLoader(dataset, batch_size=64, shuffle=True/False)                                 │
     │   • Slices total dataset into mini-batches: D = {D₁, D₂, ..., Dₖ}                      │
     │   • Each D_i is a batch of 64 pairs: D_i = {(x_j, y_j)}_{j=1}^{64}                     │
     │   • Training:   shuffle=True  (New random index permutation every epoch)               │
     │   • Evaluation: shuffle=False (Deterministic, single pass, no test epochs)             │
     └────────────────────────────────────────────────────────────────────────────────────────┘

  4. GPU-READY OUTPUT TENSORS (Fed to Neural Net)
     • Batch Images Tensor: Shape [64, 1, 28, 28]  (dtype: torch.float32, values in [0, 1])
     • Batch Labels Tensor: Shape [64]             (dtype: torch.int64, labels in {0..9})
══════════════════════════════════════════════════════════════════════════════════════════════════
```

---

## 📖 PyTorch Data Rosetta Stone (Symbols & Functions $\to$ Plain English)

| PyTorch Code / Symbol | Formal Technical Name | Plain-English ELI5 Mental Model |
| :--- | :--- | :--- |
| `ToTensor()` | Normalizing Tensor Transform | The digital scanner turning raw pixel brightness $[0, 255]$ into floating-point numbers $[0.0, 1.0]$. |
| `torch.utils.data.Dataset` | Abstract Base Dataset Class | The warehouse shelf picker who knows how to open drawer `#idx` and hand you **one single garment**. |
| `torch.utils.data.DataLoader` | Streaming Batch Wrapper | The automated forklift system that loads **64 garments onto a pallet** and delivers them to the GPU. |
| `batch_size = 64` | Mini-Batch Capacity | The pallet size: exactly 64 image grids grouped together for parallel computation. |
| `shuffle = True` | Random Permutation Flag | Thoroughly shuffling flashcards before studying so you don't memorize the question sequence. |
| `__getitem__(self, idx)` | Indexing Dunder Method | The secret Python method executed whenever you write `dataset[idx]`. |
| `pd.read_csv(...)` | Pandas CSV Reader | Opening the master spreadsheet containing image filenames and category tags. |
| `.iloc[idx, 0]` vs `.iloc[idx, 1]` | Integer Location Indexing | Column 0 is the **image path string**; Column 1 is the **integer label $0$ to $9$**. |
| `next(iter(dataloader))` | Iterator Step Function | Grabbing the very first 64-image pallet from the conveyor belt to inspect it. |

---

## Comparative Matrix: Ingestion Approaches in PyTorch

| Feature / Metric | Built-in (`torchvision.datasets`) | Custom Dataset (`CustomImageDataset`) | Raw In-Memory NumPy / Tensors |
| :--- | :--- | :--- | :--- |
| **Best Used For** | Standard benchmarks (MNIST, CIFAR, ImageNet) | Proprietary, client, or enterprise data | Toy datasets with $< 10{,}000$ points |
| **Memory Footprint** | Low (Loads on-demand from disk) | Low (Loads single images dynamically via `idx`) | **Extremely High** (Loads all gigabytes into RAM) |
| **Setup Complexity** | Zero (Single constructor call) | Moderate (Write 3 dunder methods + CSV) | Low (Single `torch.tensor()` call) |
| **Custom Transforms?**| Supported via `transform=` parameter | Supported via `transform=` parameter | Manual pre-computation required |
| **Multi-GPU / Workers**| Fully supported via `DataLoader` | Fully supported via `DataLoader` | Requires custom multi-process slicing |

---

## Complete Hands-On Implementation in Python / PyTorch

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
import os
from PIL import Image

# ==============================================================================
# 1. BUILT-IN DATASET PIPELINE (Fashion-MNIST)
# ==============================================================================
# Load official benchmark splits
train_fashion = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_fashion = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Instantiate streaming batch loaders
train_loader = DataLoader(train_fashion, batch_size=64, shuffle=True)
test_loader = DataLoader(test_fashion, batch_size=64, shuffle=False)

# Inspect first batch
train_images, train_labels = next(iter(train_loader))
print(f"FashionMNIST Batch Images Shape: {train_images.shape}")  # torch.Size([64, 1, 28, 28])
print(f"FashionMNIST Batch Labels Shape: {train_labels.shape}")  # torch.Size([64])

# ==============================================================================
# 2. CUSTOM DATASET PIPELINE (CustomImageDataset with CSV Index)
# ==============================================================================
class CustomImageDataset(Dataset):
    """Custom dataset pipeline reading image paths and labels from a CSV index."""
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Column 0: relative image filename
        img_filename = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_filename)
        
        # Load image from disk
        image = Image.open(img_path).convert("L")  # Grayscale
        
        # Column 1: category label
        label = int(self.img_labels.iloc[idx, 1])

        # Apply optional feature and target transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

# Demonstration: create dummy folder and CSV on disk
os.makedirs("demo_images", exist_ok=True)
dummy_csv_path = "demo_labels.csv"

# Generate 10 synthetic sample images
filenames, labels = [], []
for i in range(10):
    fn = f"sample_{i}.png"
    Image.fromarray(np.random.randint(0, 256, (28, 28), dtype=np.uint8)).save(os.path.join("demo_images", fn))
    filenames.append(fn)
    labels.append(i % 10)

pd.DataFrame({"image": filenames, "label": labels}).to_csv(dummy_csv_path, index=False)

# Instantiate custom dataset & loader
custom_ds = CustomImageDataset(dummy_csv_path, "demo_images", transform=ToTensor())
custom_loader = DataLoader(custom_ds, batch_size=4, shuffle=True)

custom_imgs, custom_lbls = next(iter(custom_loader))
print(f"Custom Dataset Batch Images Shape: {custom_imgs.shape}")  # torch.Size([4, 1, 28, 28])
print(f"Custom Dataset Batch Labels:       {custom_lbls}")        # tensor([x, y, z, w])
```

---

<a id="topic-1"></a>
<a id="topic-1-tensors-then-he-opens-the-dataset-sheet-0012–0153"></a>
## Topic 1: Tensors, then Dataset Sheet (00:12–01:53)

### 👶 ELI5 Quick Intuition
In Tutorial 1, we learned how to multiply tensors. But in real life, data doesn't start as clean tensors—it starts as messy image files on a hard drive. How do we get thousands of image files off the disk, turn them into tensors, and feed them to the neural network in bite-sized batches? That is the exact mission of the **`Datasets & DataLoaders`** system!

### Master Map Placement
Establishes the motivation for the tutorial: bridging the gap between raw tensor math and real-world data ingestion pipelines.

### Colab Recording Screenshot
![Topic 1 Screenshot — Transition to Datasets & DataLoaders](./screenshots/composites/ch01-topic-01-tensors-then-dataloader-sheet-panel1of1.png)
*Figure 1.1 (~00:12–01:50):* Chandan opens the official PyTorch tutorial documentation and navigates to the *Datasets & DataLoaders* Colab sheet, explaining that all data must be converted to tensors and grouped into batches before model weights can update.

### In-Depth Conceptual Exposition

* **The Data Ingestion Problem:**
  * Machine learning optimization algorithms (Stochastic Gradient Descent, Adam) update neural network weights $\theta$ using mini-batches of tensors.
  * Real-world datasets reside on persistent storage (hard drives, NVMe SSDs, cloud buckets) as discrete graphic files (`.png`, `.jpg`, `.tiff`), audio recordings (`.wav`), or text files.
* **The Core Responsibilities:**
  1. **Locate & Access:** Reliably identify where raw files are stored on disk.
  2. **Format Conversion:** Decode image file formats into numerical tensors and normalize pixel values into standard numeric ranges.
  3. **Mini-Batch Streaming:** Slice the dataset into discrete batches of size $B$ (e.g., 64) for parallel processing on GPU hardware.

```
   RAW FILES ON DISK                      CONVERSION LAYER                STREAMING BATCHES
  ┌──────────────────────┐             ┌─────────────────────┐         ┌─────────────────────────┐
  │  images/sample_01.png│ ──────────► │ Image → Tensor      │ ──────► │ Mini-batches D₁, D₂...  │
  │  images/sample_02.png│    Read &   │ ToTensor() in [0,1] │ Group 64│ Ready for Weight Update │
  └──────────────────────┘    Decode   └─────────────────────┘         └─────────────────────────┘
```

---

<a id="topic-2"></a>
<a id="topic-2-torchvision-catalogs-fashion-mnist-0153–0320"></a>
## Topic 2: Torchvision Catalogs & Fashion-MNIST (01:53–03:20)

### 👶 ELI5 Quick Intuition
Think of PyTorch as a video game console that comes with a built-in library of 20 classic free games. If you want to benchmark an image classifier, PyTorch includes pre-packaged datasets like **Fashion-MNIST** (60,000 photos of 10 types of clothing) so you don't have to spend hours taking photos of your own closet.

### Master Map Placement
Surveys built-in computer vision datasets available in `torchvision.datasets` and introduces the Fashion-MNIST benchmark.

### Colab Recording Screenshot
![Topic 2 Screenshot — Torchvision Built-in Catalogs](./screenshots/composites/ch02-topic-02-torchvision-datasets-fashionmnist-panel1of1.png)
*Figure 2.1 (~01:53–03:18):* Chandan scrolls through the `torchvision.datasets` documentation, highlighting pre-packaged computer vision datasets: Caltech 101, Caltech 256, CIFAR-10, ImageNet, and Fashion-MNIST.

### In-Depth Conceptual Exposition

* **The `torchvision.datasets` Module:**
  * PyTorch provides built-in dataset classes for standard academic and research benchmarks across image classification, object detection, segmentation, and video analysis.
* **The Fashion-MNIST Dataset Specification:**
  * Developed by Zalando Research as a drop-in replacement for original MNIST handwritten digits.
  * **Dataset Size:** 70,000 total grayscale images ($28 \times 28$ pixels).
  * **Splits:** 60,000 training images (`train=True`) and 10,000 test images (`train=False`).
  * **Classes (10 Categories):**
    * `0: T-shirt/top` | `1: Trouser` | `2: Pullover` | `3: Dress` | `4: Coat`
    * `5: Sandal` | `6: Shirt` | `7: Sneaker` | `8: Bag` | `9: Ankle boot`
  * **Why It Matters:** Preserves the lightweight $28 \times 28$ size of MNIST while providing a significantly more realistic and challenging visual classification task.

```
   FASHION-MNIST BENCHMARK: 10 APPAREL CLASSES (0 to 9)
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 0: T-shirt  │ 1: Trouser  │ 2: Pullover │ 3: Dress    │ 4: Coat             │
  │ 5: Sandal   │ 6: Shirt    │ 7: Sneaker  │ 8: Bag      │ 9: Ankle boot       │
  └─────────────────────────────────────────────────────────────────────────────┘
   Total: 60,000 Training Samples + 10,000 Evaluation Test Samples (28x28 Pixels)
```

---

<a id="topic-3"></a>
<a id="topic-3-imports-0320–0417"></a>
## Topic 3: Essential Imports (03:20–04:17)

### 👶 ELI5 Quick Intuition
Before you can build a house, you need to lay out your tools on the workbench. In PyTorch, we need 5 core tools: (1) `torch` for math, (2) `Dataset` to represent our closet, (3) `DataLoader` as our conveyor belt, (4) `datasets` to download built-in files, and (5) `ToTensor` to turn pictures into numbers.

### Master Map Placement
Walks through the essential imports required to construct PyTorch data ingestion pipelines.

### Colab Recording Screenshot
![Topic 3 Screenshot — Essential Imports](./screenshots/composites/ch03-topic-03-imports-dataset-transforms-plt-panel1of1.png)
*Figure 3.1 (~03:22–04:15):* Chandan points to the Colab import cell, explaining the purpose of each module: `torch`, `Dataset`, `DataLoader`, `datasets`, `ToTensor`, and `matplotlib.pyplot as plt`.

### In-Depth Conceptual Exposition

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
```

* **Module Breakdown:**
  * **`torch`:** Core PyTorch tensor library and CUDA compute engine.
  * **`torch.utils.data.Dataset` (Capital D):** The abstract base class inherited by any custom dataset class.
  * **`torch.utils.data.DataLoader`:** The streaming iterator that bundles individual dataset samples into batched tensors.
  * **`torchvision.datasets`:** Collection of pre-configured dataset downloaders (Fashion-MNIST, CIFAR-10, etc.).
  * **`torchvision.transforms.ToTensor`:** Callable class converting PIL images or NumPy arrays into `torch.FloatTensor` in range $[0.0, 1.0]$.
  * **`matplotlib.pyplot as plt`:** Standard plotting library used to inspect sample images visually.

---

<a id="topic-4"></a>
<a id="topic-4-fashionmnist-traintest-0417–0609"></a>
## Topic 4: `FashionMNIST` Train and Test Splits (04:17–06:09)

### 👶 ELI5 Quick Intuition
To download and load the Fashion-MNIST clothes, we write two lines of code: one for the **homework study pile** (`train=True`), and one for the **final exam test pile** (`train=False`). The `download=True` flag tells PyTorch: *"If you don't already have these files on my hard drive, download them from the internet right now!"*

### Master Map Placement
Demonstrates downloading, instantiating, and normalizing the training and test splits of Fashion-MNIST.

### Colab Recording Screenshot
![Topic 4 Screenshot — Downloading FashionMNIST Splits](./screenshots/composites/ch04-topic-04-fashionmnist-train-test-totensor-panel1of1.png)
*Figure 4.1 (~04:20–06:05):* Chandan executes the code cell instantiating `training_data` and `test_data` using `datasets.FashionMNIST`, explaining `root="data"`, `train=True/False`, `download=True`, and `transform=ToTensor()`.

### In-Depth Conceptual Exposition

```python
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

* **Constructor Parameter Mechanics:**
  * **`root="data"`:** Specifies the directory path on the local filesystem where raw data files will be downloaded or retrieved.
  * **`train=True` vs `train=False`:** Selects between the 60,000-sample training partition and the 10,000-sample evaluation partition.
  * **`download=True`:** Instructs PyTorch to check if the dataset archive exists in `root`. If missing, it automatically downloads and extracts the raw binary files.
  * **`transform=ToTensor()`:** Applies on-the-fly conversion whenever an image is accessed, returning a `[1, 28, 28]` float tensor with values normalized to $[0.0, 1.0]$.

```
   DOWNLOAD & EXTRACTION PIPELINE
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ datasets.FashionMNIST(root="data", download=True)                           │
  │   ├── Check if ./data/FashionMNIST exists                                   │
  │   ├── If missing: download *.gz archives from AWS S3 mirror                 │
  │   └── Extract raw binary pixel blocks into disk cache                       │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

<a id="topic-5"></a>
<a id="topic-5-plots-when-the-repo-is-not-enough-0609–0741"></a>
## Topic 5: Visualizing & When Built-in Fails (06:09–07:41)

### 👶 ELI5 Quick Intuition
Built-in datasets like Fashion-MNIST are great for practicing in school. But when you get a job at a hospital, a bank, or an e-commerce startup, **your company's data will never be in `torchvision.datasets`**. You will be handed a private folder full of images and a spreadsheet of tags. You must write your own custom dataset class to load it!

### Master Map Placement
Contrasts academic benchmark datasets with proprietary real-world datasets, motivating the necessity of custom `Dataset` implementations.

### Colab Recording Screenshot
![Topic 5 Screenshot — Visualizing and Moving to Custom Data](./screenshots/composites/ch05-topic-05-labels-when-builtin-fails-panel1of1.png)
*Figure 5.1 (~06:12–07:38):* The Colab notebook plots a $3 \times 3$ grid of sample images with text labels (Pullover, Ankle boot, Shirt). Chandan explains that while built-in datasets are convenient, real-world engineering requires curating custom datasets for proprietary, sensitive, or client data.

### In-Depth Conceptual Exposition

* **Inspecting Built-in Data:**
  * Visualizing sample images with Matplotlib confirms proper pixel decoding, aspect ratio integrity, and label alignment.
* **The Real-World Boundary:**
  * Built-in datasets cover only a tiny fraction of machine learning use cases.
  * In commercial engineering environments, datasets involve:
    * **Proprietary Medical Imaging:** DICOM / MRI scans with HIPAA privacy restrictions.
    * **Industrial Quality Control:** High-resolution defect photographs on assembly lines.
    * **Financial Document Analysis:** Invoices, receipts, and KYC documents.
  * **The Solution:** PyTorch enables engineers to write a custom wrapper class inheriting from `Dataset`.

```
   BENCHMARK DATASETS (Lucky Path)           ENTERPRISE DATASETS (Real-World Path)
  ┌───────────────────────────────┐        ┌────────────────────────────────────────┐
  │ torchvision.datasets          │        │ Proprietary / Confidential / Client    │
  │ • Fashion-MNIST, CIFAR-10     │   VS   │ • Custom Folder on Local Drive / S3    │
  │ • Single line to download     │        │ • CSV Annotation Spreadsheet           │
  │ • Fixed academic formats      │        │ • Requires CustomImageDataset Wrapper! │
  └───────────────────────────────┘        └────────────────────────────────────────┘
```

---

<a id="topic-6"></a>
<a id="topic-6-custom-data-folder--csv-0741–1021"></a>
## Topic 6: Custom Datasets: Folder + CSV (07:41–10:21)

### 👶 ELI5 Quick Intuition
To build a custom dataset, you organize your files into two parts:
1. **The Photo Album (`img_dir`):** A folder containing all your `.png` or `.jpg` image files.
2. **The Packing Slip (`annotations_file.csv`):** A spreadsheet with two columns. Column 0 tells you the name of the file (`shoe_10.png`), and Column 1 tells you the category (`7`).

### Master Map Placement
Defines the two-component architecture of custom image datasets: the image storage directory and the tabular annotation file.

### Colab Recording Screenshot
![Topic 6 Screenshot — Handwritten Architecture for Custom Datasets](./screenshots/composites/ch06-topic-06-custom-folder-plus-csv-panel1of1.png)
*Figure 6.1 (~07:45–10:18):* Chandan draws the custom dataset architecture on the digital whiteboard: `img_dir` (directory path on OS) + `annotation file` (CSV with relative path / filename in column 0 and class label in column 1).

### In-Depth Conceptual Exposition

* **The Two Fundamental Components:**
  1. **Image Directory (`img_dir`):** An operating system path (`str` or `Path`) pointing to the directory containing raw image files on disk.
  2. **Annotation File (`annotations_file`):** A CSV (Comma-Separated Values) spreadsheet establishing the ground-truth relationship between image files and categorical labels.
* **CSV Schema Specification:**
  * **Column Index 0:** Relative path or filename of the image (e.g., `"coat_001.png"`).
  * **Column Index 1:** Categorical label index (e.g., `4` for Coat).
* **Why We Decouple Annotations from Raw Files:**
  * Storing image bytes directly inside a database or CSV causes severe memory bloat.
  * Keeping images as raw files on disk and storing only lightweight metadata in a CSV enables dynamic, on-demand loading of gigabytes of data without running out of RAM!

```
   CUSTOM DATASET FILE SYSTEM LAYOUT:
   
   my_project_data/
   ├── annotations.csv                <── Master Index Spreadsheet
   └── images/                        <── Image Directory (img_dir)
       ├── img_0001.png
       ├── img_0002.png
       └── ...
```

---

<a id="topic-7"></a>
<a id="topic-7-customimagedataset-1021–1506"></a>
## Topic 7: The `CustomImageDataset` Class (10:21–15:06)

### 👶 ELI5 Quick Intuition
Here is where we write the code for our warehouse worker (`CustomImageDataset`).  
We create a class with 3 rules:
1. `__init__`: Read the CSV spreadsheet into memory.
2. `__len__`: Count how many rows are in the spreadsheet (`len(csv)`).
3. `__getitem__(idx)`: When asked for row `idx`, look up the image name in Column 0, open that image from the folder, look up the label in Column 1, and hand back the pair `(image, label)`.

### Master Map Placement
Walks line-by-line through the Python implementation of `CustomImageDataset`, subclassing `torch.utils.data.Dataset`.

### Colab Recording Screenshot
![Topic 7 Screenshot — CustomImageDataset Class Implementation](./screenshots/composites/ch07-topic-07-customimagedataset-class-panel1of1.png)
*Figure 7.1 (~10:25–15:02):* Chandan walks through the Colab code defining `class CustomImageDataset(Dataset)`: showing `__init__` with `pd.read_csv`, `__len__` returning `len(self.img_labels)`, and `__getitem__` using `os.path.join`, `iloc[idx, 0]`, and `iloc[idx, 1]`.

### In-Depth Conceptual Exposition

```python
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # 1. Join directory path with relative filename from Column 0
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        
        # 2. Read image from disk
        image = read_image(img_path)
        
        # 3. Retrieve integer label from Column 1
        label = self.img_labels.iloc[idx, 1]
        
        # 4. Apply optional transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        # 5. Return sample tuple
        return image, label
```

* **Line-by-Line Method Breakdown:**
  * **`__init__` (Initialization):** Runs once during instantiation. Uses `pd.read_csv()` to parse the annotation table into a DataFrame (`self.img_labels`), stores the directory path (`self.img_dir`), and saves optional transformation functions.
  * **`__len__` (Dataset Size):** Returns the integer row count of `self.img_labels`. This tells PyTorch how many total samples exist in the dataset.
  * **`__getitem__(self, idx)` (Single-Sample Fetcher):**
    1. `self.img_labels.iloc[idx, 0]`: Reads the filename string from row `idx`, column 0.
    2. `os.path.join(...)`: Constructs the absolute or relative filesystem path.
    3. `read_image(...)`: Decodes image bytes from disk into a `torch.Tensor` of shape `[C, H, W]`.
    4. `self.img_labels.iloc[idx, 1]`: Reads the numerical class label from column 1.
    5. Returns the tuple `(image, label)`.

```
   GETITEM WORKFLOW: ONE SAMPLE FETCH
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ idx ──► iloc[idx, 0] ──► os.path.join(img_dir, fn) ──► read_image ──► image │
  │     ──► iloc[idx, 1] ───────────────────────────────────────────────► label │
  │                                                                             │
  │                     OUTPUT TUPLE: (image_tensor, label)                     │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

<a id="topic-8"></a>
<a id="topic-8-dataloader-batch-64-shuffle-1506–1826"></a>
## Topic 8: `DataLoader`, Batch 64 & Shuffling (15:06–18:26)

### 👶 ELI5 Quick Intuition
Now that our worker knows how to fetch one shirt at a time, we build the **automated forklift conveyor belt** (`DataLoader`).  
* We set `batch_size = 64`: the forklift waits until it has stacked 64 shirts before sending them to the GPU.
* We set `shuffle = True` for training: the forklift shuffles the order of garments before every shift so the neural network doesn't memorize the sequence.
* We set `shuffle = False` for testing: on test day, we evaluate once in exact order.

### Master Map Placement
Demonstrates wrapping datasets with `DataLoader`, configuring mini-batch sizes, index shuffling, and iterating over batches.

### Colab Recording Screenshot
![Topic 8 Screenshot — DataLoader Mini-Batching and Shuffling](./screenshots/composites/ch08-topic-08-dataloader-batch-shuffle-panel1of1.png)
*Figure 8.1 (~15:10–18:20):* Chandan instantiates `train_dataloader` and `test_dataloader` with `batch_size=64`, writes the mathematical batch decomposition $D = \{D_1, D_2, \dots, D_k\}$ on the board, and demonstrates iterating with `next(iter(train_dataloader))`.

### In-Depth Conceptual Exposition

```python
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# Fetch single mini-batch
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.shape}")  # [64, 1, 28, 28]
print(f"Labels batch shape:  {train_labels.shape}")    # [64]
```

* **Mathematical Batch Partitioning:**
  * Let total dataset $\mathcal{D}$ have $N$ samples.
  * Slicing $\mathcal{D}$ by batch size $B = 64$ decomposes the dataset into $K$ disjoint mini-batches:
    $$\mathcal{D} = \{D_1, D_2, \dots, D_K\}$$
    where each mini-batch $D_i$ consists of 64 sample tuples:
    $$D_i = \{(x_j, y_j)\}_{j=1}^{64}$$
* **Chandan's Shuffling Rule (Train vs. Test):**
  * **Training DataLoader (`shuffle=True`):**  
    At the start of every epoch, `DataLoader` generates a new random permutation of indices. This breaks spatial/temporal correlations and prevents optimization algorithms from oscillating along deterministic cyclical trajectories.
  * **Test DataLoader (`shuffle=False`):**  
    Chandan explicitly notes: *On the test set, we evaluate performance only once. Shuffling adds unnecessary overhead without changing the aggregate evaluation metrics.*

```
   CHANDAN'S BOARD NOTATION: BATCH DECOMPOSITION
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Total Dataset D = {D₁, D₂, ..., Dₖ}                                         │
  │ Each D_i = { (x_j, y_j) }  where j = 1, 2, ..., 64                          │
  │                                                                             │
  │ • x_j = Image Tensor (1, 28, 28)                                            │
  │ • y_j = Class Label  (Integer 0 to 9)                                       │
  │                                                                             │
  │ ⟹ Batch Tensor Shapes: Features [64, 1, 28, 28], Labels [64]                │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Workplace Scenarios & Debugging Data Pipelines

### Scenario 1: Memory Leaks & OOM during Multi-Worker Data Loading
* **Context:** An engineer trains a ResNet model on 100,000 high-resolution images using `DataLoader(dataset, batch_size=64, num_workers=8)`. After 3 epochs, the server runs out of system RAM (Out-Of-Memory error) and crashes.
* **Mathematical & Architectural Root Cause:**
  * In the custom dataset class, the developer pre-loaded all 100,000 raw PIL images into a Python list `self.images = [...]` during `__init__`.
  * When `num_workers=8` is specified, PyTorch uses Python `multiprocessing` to fork 8 worker processes. Each worker inherits a copy-on-write duplicate of the in-memory image list, multiplying memory consumption by $8\times$ until RAM is exhausted.
* **Production Remedy:**
  * Never pre-load large image datasets into RAM inside `__init__`.
  * Store only lightweight metadata (CSV DataFrame / filepaths) in `__init__`, and load individual images dynamically from disk on-demand inside `__getitem__(self, idx)` using `read_image()` or `Image.open()`.

### Scenario 2: Fatal CSV Index Inversion & Dimension Format Mismatch
* **Context:** An AI developer implements `CustomImageDataset`, but during the first training step, PyTorch throws: `FileNotFoundError: No such file or directory: 'images/4'` followed by shape mismatch errors in the convolutional layer.
* **Root Cause:**
  1. **Column Swap:** The developer wrote `img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 1]))` instead of `iloc[idx, 0]`. Column 1 contained the integer class label `4`, causing PyTorch to look for an image named `"4"`.
  2. **Channel Format Mismatch:** The developer loaded images using OpenCV (`cv2.imread`), which outputs NumPy arrays in $(H, W, C)$ format with BGR channels, whereas PyTorch convolution layers strictly require $(C, H, W)$ format in RGB.
* **Production Remedy:**
  * Ensure Column 0 is reserved for relative filepaths (`iloc[idx, 0]`) and Column 1 is reserved for categorical labels (`iloc[idx, 1]`).
  * Always use `torchvision.transforms.ToTensor()` or `torchvision.io.read_image()` to ensure tensors have shape `(C, H, W)` and normalized float values in $[0.0, 1.0]$.

---

## External References

> Comprehensive multi-source learning materials curated for every subtopic in this tutorial.

### Topic 1 — PyTorch Tensors to Data Ingestion
* **Video Lectures:**
  1. [PyTorch Official: Introduction to PyTorch — YouTube Tutorial Series](https://www.youtube.com/watch?v=IC0_FRiX-sw) — Official walkthrough of PyTorch tensor fundamentals.
  2. [DeepLearningAI: PyTorch for Deep Learning Specialization](https://www.youtube.com/watch?v=V_xro1bcAuA) — Transitioning from tensor operations to data pipelines.
  3. [freeCodeCamp: PyTorch Full Course for Beginners](https://www.youtube.com/watch?v=GIsg-ZUy0MY) — Complete guide to tensors, datasets, and training loops.
* **Official Guides & Articles:**
  1. [PyTorch Official Documentation: Tensors Tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) — Core tensor creation and indexing guide.
  2. [Real Python: PyTorch for Deep Learning](https://realpython.com/pytorch-python/) — Practical guide to PyTorch data workflows.
  3. [Towards Data Science: Understanding PyTorch Tensor Data Structures](https://towardsdatascience.com/understanding-pytorch-tensors-and-memory-layout/) — Memory layout and stride mechanics.

### Topic 2 — Torchvision Datasets & Fashion-MNIST
* **Video Lectures:**
  1. [StatQuest with Josh Starmer: Neural Networks & Fashion-MNIST](https://www.youtube.com/watch?v=HGwBXDKFk9I) — Intuitive breakdown of the 10 clothing categories.
  2. [Aladdin Persson: PyTorch Datasets and Transforms Guide](https://www.youtube.com/watch?v=zN49HdZ8D8Q) — Walkthrough of built-in torchvision catalogs.
  3. [MIT 6.S191: Convolutional Neural Networks and Computer Vision](https://www.youtube.com/watch?v=AjtX1N3kzuc) — Benchmark datasets in computer vision research.
* **Official Guides & Articles:**
  1. [Han Xiao et al.: Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning (arXiv:1708.07747)](https://arxiv.org/abs/1708.07747) — The original research paper introducing Fashion-MNIST.
  2. [PyTorch Official Docs: torchvision.datasets API Reference](https://pytorch.org/vision/stable/datasets.html) — Full catalog of vision datasets.
  3. [Papers with Code: Fashion-MNIST Benchmark Results](https://paperswithcode.com/dataset/fashion-mnist) — State-of-the-art accuracy leaderboards on Fashion-MNIST.

### Topic 3 — Essential PyTorch Imports & Transforms
* **Video Lectures:**
  1. [deeplizard: PyTorch Transforms and Data Augmentation](https://www.youtube.com/watch?v=kOhotWCta10) — Detailed explanation of `ToTensor()` and normalization.
  2. [PyTorch Official: Transforming and Augmenting Images](https://www.youtube.com/watch?v=0k5iZ_QfPrc) — Modern `torchvision.transforms.v2` pipelines.
  3. [Daniel Bourke: PyTorch Custom Datasets and Transforms](https://www.youtube.com/watch?v=V_xro1bcAuA) — Composing multiple vision transforms.
* **Official Guides & Articles:**
  1. [PyTorch Official Docs: Transforms API](https://pytorch.org/vision/stable/transforms.html) — Reference guide for `ToTensor`, `Normalize`, and `Resize`.
  2. [Albumentations: Fast Image Augmentation for PyTorch](https://albumentations.ai/docs/) — High-performance image transform library.
  3. [PyTorch Tutorial: Transforms Basics](https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html) — Official step-by-step transforms guide.

### Topic 4 — Fashion-MNIST Train vs Test Splits
* **Video Lectures:**
  1. [Andrew Ng (Coursera): Train, Dev, and Test Set Best Practices](https://www.youtube.com/watch?v=1waHhmHbrJg) — Why dataset isolation is non-negotiable.
  2. [StatQuest: Machine Learning Fundamentals — Cross Validation and Data Splitting](https://www.youtube.com/watch?v=fSytzGwwBVw) — Visual explanation of train/test splits.
  3. [Stanford CS231n: Lecture 2 — Image Classification Pipeline](https://www.youtube.com/watch?v=OoUX-nOEjG0) — Data budgeting in deep learning.
* **Official Guides & Articles:**
  1. [Google ML Practicum: Data Splitting and Generalization](https://developers.google.com/machine-learning/crash-course/generalization/peril-of-overfitting) — Preventing data leakage in production.
  2. [Scikit-Learn Guide: Cross-Validation and Dataset Splits](https://scikit-learn.org/stable/modules/cross_validation.html) — Statistical foundations of unbiased evaluation.
  3. [PyTorch Official Tutorial: Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) — The official tutorial walked in this video.

### Topic 5 — Matplotlib Plotting & Custom Data Realities
* **Video Lectures:**
  1. [Corey Schafer: Matplotlib Tutorial — Plotting Subplots and Grids](https://www.youtube.com/watch?v=XFZRVnP-MTU) — Visualizing multi-image grids in Python.
  2. [Sentdex: Deep Learning with PyTorch — Image Visualization](https://www.youtube.com/watch?v=BzcBsEb05OE) — Displaying tensor images with `plt.imshow`.
  3. [PyData: Exploratory Data Analysis for Computer Vision](https://www.youtube.com/watch?v=W3a4w_2GqK8) — Inspecting dataset class balance and outliers.
* **Official Guides & Articles:**
  1. [Matplotlib Official Docs: `imshow` Reference](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html) — Rendering 2D grayscale and 3D RGB matrices.
  2. [Towards Data Science: Data Auditing for Deep Learning](https://towardsdatascience.com/data-auditing-for-computer-vision/) — Identifying label errors and corrupted image files.
  3. [Google Research: The High Cost of Bad Data in AI](https://research.google/pubs/pub49996/) — Why real-world data curation dominates ML engineering.

### Topic 6 — Custom Dataset File Structures & CSV Annotations
* **Video Lectures:**
  1. [Aladdin Persson: Custom Dataset Tutorial in PyTorch (Folder + CSV)](https://www.youtube.com/watch?v=ZoZHd0IrVGE) — Industry-standard custom dataset setup.
  2. [Daniel Bourke: PyTorch Custom Data Setup (Zero to Mastery)](https://www.youtube.com/watch?v=V_xro1bcAuA) — Organizing folders, metadata, and labels.
  3. [Keith Galli: Pandas Data Analysis Crash Course](https://www.youtube.com/watch?v=vmEHCJofslg) — Reading and manipulating CSV files with `.iloc`.
* **Official Guides & Articles:**
  1. [Pandas Official Documentation: Indexing and Selecting Data with `iloc`](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) — Integer location indexing rules.
  2. [Python Official Docs: `os.path` and `pathlib` Reference](https://docs.python.org/3/library/os.path.html) — Safe cross-platform filepath handling.
  3. [PyTorch Official Recipes: Loading Custom Datasets](https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html) — Step-by-step custom ingestion recipe.

### Topic 7 — Subclassing `Dataset` (`__init__`, `__len__`, `__getitem__`)
* **Video Lectures:**
  1. [PyTorch Official: Writing Custom Datasets, DataLoaders, and Transforms](https://www.youtube.com/watch?v=9cKsq14Kfsw) — Official tutorial on `CustomDataset` OOP architecture.
  2. [DeepLizard: PyTorch Dataset Class Explained](https://www.youtube.com/watch?v=k4jY9L5YsTQ) — Visualizing the internal execution of `__getitem__`.
  3. [Corey Schafer: Python OOP Special (Dunder) Methods](https://www.youtube.com/watch?v=3ohzBxoFHAY) — How `__len__` and `__getitem__` enable Pythonic indexing.
* **Official Guides & Articles:**
  1. [PyTorch Official Docs: `torch.utils.data.Dataset` API](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) — Abstract class specification.
  2. [Real Python: Python Dunder Methods Reference](https://realpython.com/operator-overloading-python/) — Deep dive into Python magic methods.
  3. [PyTorch Developer Guide: High Performance Custom Datasets](https://pytorch.org/blog/efficient-pytorch-data-loading/) — Avoiding common memory bottlenecks in `__getitem__`.

### Topic 8 — `DataLoader`, Mini-Batching & Shuffling
* **Video Lectures:**
  1. [Aladdin Persson: PyTorch DataLoader — batch_size, shuffle, num_workers](https://www.youtube.com/watch?v=nvtTsdffp9g) — Detailed breakdown of DataLoader concurrency.
  2. [Stanford CS231n: Lecture 6 — Training Neural Networks (Mini-Batch SGD)](https://www.youtube.com/watch?v=wEoyxE0GP2M) — Why mini-batching stabilizes gradients.
  3. [deeplizard: PyTorch DataLoader Explained](https://www.youtube.com/watch?v=mU2FmgoGDuI) — Stepping through batches with `iter()` and `next()`.
* **Official Guides & Articles:**
  1. [PyTorch Official Docs: `torch.utils.data.DataLoader` API](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) — Reference documentation for all parameters.
  2. [NVIDIA Developer Blog: Accelerating PyTorch Data Loading](https://developer.nvidia.com/blog/how-to-optimize-data-loading-in-pytorch/) — GPU memory pinning (`pin_memory`) and worker tuning.
  3. [Pytorch Lightning Guide: Demystifying the PyTorch DataLoader](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) — Structuring reproducible data pipelines for production.

---

## Sources & Production Notes

* **Primary Recording:** [W1_T2 on YouTube](https://www.youtube.com/watch?v=L5n4rNrLZ_8) · IIT Madras B.S. Degree Programme · Runtime: 18:26
* **Timed Audio Captions:** `raw/captions.en.timed.txt` (ASR transcripts verified against Colab notebook code)
* **Composite Screenshot Panels:** `./screenshots/composites/ch01-...` through `ch08-...` (High-resolution captures per topic MM:SS)
* **Official Reference Notebook:** PyTorch Official Tutorials — *Datasets & DataLoaders* (`torch.utils.data.Dataset`, `torch.utils.data.DataLoader`).
