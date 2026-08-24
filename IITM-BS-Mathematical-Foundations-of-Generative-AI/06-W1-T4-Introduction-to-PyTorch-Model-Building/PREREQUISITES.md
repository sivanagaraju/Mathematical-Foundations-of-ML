# Warm-up Before the Lecture: Foundations of PyTorch Model Building

> **Do this first.** Then open [NOTES.md](./NOTES.md) at the **Executive Summary** architecture map.  
> This warm-up equips you with the foundational intuition, analogies, runnable code, and mathematical tools needed to master the lecture without getting lost in the PyTorch syntax.

---

### 📖 Math & PyTorch Terminology Rosetta Stone

If you are returning to neural networks and calculus after years away, start here. Every mathematical symbol and PyTorch object translates directly to an intuitive engineering concept:

| Symbol / API | Formal Mathematical Name | Plain-English Meaning | Everyday Physical Metaphor | PyTorch Shape / Example |
| :--- | :--- | :--- | :--- | :--- |
| $X \in \mathbb{R}^{B \times 1 \times 28 \times 28}$ | 4D Input Image Batch Tensor | A batch of $B$ grayscale clothing photos ($28\times 28$ pixels each). | A stack of 64 black-and-white catalog photographs. | `torch.Tensor` of shape `(64, 1, 28, 28)` |
| `nn.Flatten()` | Spatial Vectorization Operator | Stacks 2D grid pixels row-by-row into a flat 1D feature vector of length 784. | Unrolling a rolled-up square carpet into a single long runway strip. | Input `(B, 1, 28, 28)` $\to$ Output `(B, 784)` |
| $W \in \mathbb{R}^{512 \times 784}$ | Linear Weight Matrix | The trainable parameters that mix and weight input features to form hidden representations. | A giant control board with 400,928 adjustment knobs. | `layer.weight` tensor `(512, 784)` |
| $b \in \mathbb{R}^{512}$ | Bias Vector | An independent baseline offset added to each output node, independent of input pixels. | The baseline zero-point calibration screw on a weighing scale. | `layer.bias` tensor `(512,)` |
| $\text{ReLU}(z) = \max(0, z)$ | Rectified Linear Activation Function | A nonlinear activation that passes positive signals unchanged and clamps negative signals to zero. | A one-way electrical diode or a nightclub bouncer turning away negative numbers. | `torch.relu(z)` or `nn.ReLU()` |
| $z \in \mathbb{R}^{10}$ | Unnormalized Logits Vector | Raw, unbounded real numbers output by the final linear layer (one score per class). | 10 sports commentators shouting confidence scores from $-\infty$ to $+\infty$. | Output of `nn.Linear(512, 10)` |
| $\sigma(z)_i = \frac{e^{z_i}}{\sum e^{z_j}}$ | Softmax Normalization Function | Converts raw unbounded logits into a valid probability distribution summing to 1.0. | Dividing a pie among 10 winners proportionally according to their scores. | `nn.functional.softmax(logits, dim=1)` |
| $\arg\max_i (z_i)$ | Decision / Index Operator | Selects the integer index of the highest score or highest probability. | Pointing to the winning contestant on the podium. | `torch.argmax(logits, dim=1)` |
| `nn.CrossEntropyLoss()` | Multinomial Cross-Entropy Loss | Computes $-\log(\text{Softmax}(z)_y)$, penalizing incorrect predictions with extreme sensitivity. | A strict referee measuring the penalty for guessing the wrong bin. | `loss_fn(logits, targets)` |
| $\nabla_\theta L = \frac{\partial L}{\partial \theta}$ | Gradient Vector | The vector of partial derivatives pointing in the direction of steepest loss increase. | A topographical compass pointing directly uphill toward higher error. | `param.grad` |
| $W \leftarrow W - \eta \nabla_W L$ | Stochastic Gradient Descent (SGD) | Updating model weights by taking a small step in the opposite direction of the gradient. | Rolling a marble downhill toward the lowest valley of the error landscape. | `optimizer.step()` |
| `optimizer.zero_grad()` | Gradient Buffer Flush | Resets all accumulated parameter gradient buffers to zero before backpropagation. | Erasing the chalkboard before solving the next math problem. | `opt.zero_grad()` |
| `model.train()` vs `model.eval()` | Mode Toggle Switch | Tells layers (Dropout, BatchNorm) whether the network is training or doing inference. | Flipping the sign on a storefront between "Under Construction" and "Open for Business". | `model.train()` / `model.eval()` |
| `torch.no_grad()` | Autograd Graph Disabler | Disables tracking history and building dynamic computation graphs, saving VRAM. | Watching a movie without taking detailed lecture notes. | `with torch.no_grad(): ...` |
| `loss.item()` | Scalar Float Unpacker | Extracts a raw Python float from a single-element 0D PyTorch scalar tensor. | Taking a photograph of a scoreboard number and throwing away the scorebook. | `loss_val = loss.item()` |
| `model.state_dict()` | Parameter Dictionary | A Python dictionary mapping every layer name to its learned weight and bias tensors. | A blueprint detailing the exact final positions of all 400,000 knobs. | `torch.save(model.state_dict(), "m.pth")` |

---

## 1. Multi-Layer Perceptrons (MLPs) as Stacked Linear Maps with Bends

<a id="p1-mlp"></a>

### Intuition & Physical Metaphor (The Three Warehouse Desks)
Imagine an assembly line in a giant warehouse designed to classify items into 10 bins.
- **Desk 1 (`Linear(784, 512)`):** A clerk reads all 784 pixel brightness numbers and writes a 512-number preliminary summary using a custom formula.
- **Bouncer 1 (`ReLU`):** A strict bouncer stands right behind Desk 1. If any number on the summary sheet is negative, the bouncer stamps it **zero**. Positive numbers pass through unchanged.
- **Desk 2 (`Linear(512, 512)`):** A second clerk reads the 512 positive numbers and refines them into another 512-number summary.
- **Bouncer 2 (`ReLU`):** Another bouncer clamps all negative numbers to zero.
- **Desk 3 (`Linear(512, 10)`):** The final clerk writes **10 raw confidence scores** (logits). Notice: **No bouncer sits after Desk 3!** A negative score is crucial information (it means "this is definitely not a T-shirt").

```
   [ Input Photo: 784 Numbers ]
                │
                ▼
   ┌──────────────────────────┐
   │ Desk 1: Linear(784, 512) │  ──> (Matrix Multiply W1 * x + b1)
   └────────────┬─────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │ Bouncer 1: ReLU Bend     │  ──> (Clamps negative numbers to 0)
   └────────────┬─────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │ Desk 2: Linear(512, 512) │  ──> (Matrix Multiply W2 * h1 + b2)
   └────────────┬─────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │ Bouncer 2: ReLU Bend     │  ──> (Clamps negative numbers to 0)
   └────────────┬─────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │ Desk 3: Linear(512, 10)  │  ──> (Matrix Multiply W3 * h2 + b3)
   └────────────┬─────────────┘
                │
                ▼
   [ 10 Raw Output Logits (No Bouncer / No ReLU!) ]
```

### Plain-English Breakdown
- A **Linear Layer** (also called Fully Connected or Dense layer) computes an affine mathematical transformation:
  $$y = W^\top x + b$$
  where $W$ is a matrix of learned weights and $b$ is a vector of learned biases.
- **Why Biases Matter:** The bias vector $b$ provides an independent offset. Even if all input pixels $x$ are zero (a completely black image), the layer can output non-zero values.
- **Why We Need Non-Linear Bends (`ReLU`):** If you stack three linear layers without an activation function:
  $$y = W_3 (W_2 (W_1 x + b_1) + b_2) + b_3 = (W_3 W_2 W_1) x + (W_3 W_2 b_1 + W_3 b_2 + b_3) = W_{\text{combined}} x + b_{\text{combined}}$$
  Three linear layers mathematically collapse into **one single linear layer**! A pure linear network cannot solve nonlinear classification boundaries (like XOR or image shapes).
- The **Rectified Linear Unit (`ReLU`)** introduces a sharp bend:
  $$\text{ReLU}(z) = \max(0, z)$$
  This simple hinge allows deep networks to approximate any complex nonlinear boundary (Universal Approximation Theorem).

### Formal Mathematics & Worked Numerical Micro-Example

#### Mathematical Formalism
Let $x \in \mathbb{R}^{d_{\text{in}}}$. An $L$-layer Multi-Layer Perceptron is defined recursively:
$$h^{(0)} = x$$
$$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}, \quad l = 1, \dots, L$$
$$h^{(l)} = \sigma(z^{(l)}), \quad l = 1, \dots, L-1$$
$$y = z^{(L)} = W^{(L)} h^{(L-1)} + b^{(L)}$$
where $W^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$, $b^{(l)} \in \mathbb{R}^{d_l}$, and $\sigma(\cdot) = \max(0, \cdot)$.

#### Concrete Numerical Example (2-in, 3-out Linear Layer + ReLU)
Let input vector $x = \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix}$.
Let weight matrix $W = \begin{bmatrix} 0.5 & -1.0 \\ 1.5 & 2.0 \\ -0.5 & 0.0 \end{bmatrix}$ and bias $b = \begin{bmatrix} 0.1 \\ -0.5 \\ 1.0 \end{bmatrix}$.

1. **Compute Linear Output $z = W x + b$:**
   $$z_1 = (0.5)(2.0) + (-1.0)(-1.0) + 0.1 = 1.0 + 1.0 + 0.1 = 2.1$$
   $$z_2 = (1.5)(2.0) + (2.0)(-1.0) + (-0.5) = 3.0 - 2.0 - 0.5 = 0.5$$
   $$z_3 = (-0.5)(2.0) + (0.0)(-1.0) + 1.0 = -1.0 + 0.0 + 1.0 = 0.0$$
   $$z = \begin{bmatrix} 2.1 \\ 0.5 \\ 0.0 \end{bmatrix}$$

2. **Apply ReLU Activation $h = \text{ReLU}(z)$:**
   $$h_1 = \max(0, 2.1) = 2.1$$
   $$h_2 = \max(0, 0.5) = 0.5$$
   $$h_3 = \max(0, 0.0) = 0.0$$
   $$h = \begin{bmatrix} 2.1 \\ 0.5 \\ 0.0 \end{bmatrix}$$

```python
import torch
import torch.nn as nn

# Verifying Linear + ReLU in PyTorch
torch.manual_seed(42)

linear_layer = nn.Linear(in_features=2, out_features=3)
# Manually load exact numbers from worked example
with torch.no_grad():
    linear_layer.weight.copy_(torch.tensor([[0.5, -1.0], [1.5, 2.0], [-0.5, 0.0]]))
    linear_layer.bias.copy_(torch.tensor([0.1, -0.5, 1.0]))

x_input = torch.tensor([2.0, -1.0])
z_linear = linear_layer(x_input)
h_relu = torch.relu(z_linear)

print("Linear output (z):", z_linear.detach().numpy())
print("ReLU output (h):  ", h_relu.detach().numpy())
```

#### 🎯 Diagnostic Mini-Check
1. *Why is there no activation function (no ReLU, no Sigmoid) applied after the final linear layer (`Linear(512, 10)`)?*  
   **Answer:** The final layer outputs raw **logits**. Cross-entropy loss functions (`nn.CrossEntropyLoss`) expect raw logits because they combine log-softmax and negative log-likelihood into a numerically stable single formula. Applying an activation like ReLU would destroy negative logit information.
2. *How many total trainable parameters exist in `nn.Linear(784, 512)`?*  
   **Answer:** $784 \times 512 = 401,408$ weights $+$ $512$ biases $= \mathbf{401,920}$ parameters.

---

## 2. Spatial Flattening and Tensor Dimensionality ($28\times 28 \to 784$)

<a id="p2-flatten"></a>

### Intuition & Physical Metaphor (Unrolling the Carpet)
Imagine a $28\times 28$ square tile mosaic of a shoe. 
- A human eye looks at it as a 2D surface with rows and columns.
- A standard Linear layer is like a vending machine coin slot: it only accepts a single straight ribbon of coins.
- **`nn.Flatten()`** takes row 1 (28 pixels), glues row 2 (28 pixels) directly to its tail, and repeats for all 28 rows, creating a single ribbon of $28 \times 28 = 784$ numbers.
- If you feed a shopping cart of 64 photos (`(64, 1, 28, 28)`), `nn.Flatten()` flattens each photo individually, preserving the batch size of 64 (`(64, 784)`).

```
   2D Spatial Image (28 rows x 28 cols):
   Row 0: [ p0,0   p0,1   ... p0,27 ]
   Row 1: [ p1,0   p1,1   ... p1,27 ]
   ...
   Row 27:[ p27,0  p27,1  ... p27,27 ]
                   │
                   ▼  nn.Flatten(start_dim=1)
   1D Feature Vector (784 numbers):
   [ p0,0 ... p0,27 | p1,0 ... p1,27 | ... | p27,0 ... p27,27 ]
```

### Plain-English Breakdown
- Fashion-MNIST images are stored as 4D tensors: `(Batch_Size, Channels, Height, Width)` $= (N, 1, 28, 28)$.
- Multi-Layer Perceptrons do not possess 2D spatial convolution filters; their input dimension is a flat 1D vector ($d=784$).
- In speech transcripts, you might hear "28 plus 28" — this is an automatic transcription artifact for **$28 \times 28 = 784$**.
- `nn.Flatten(start_dim=1)` flattens all spatial and channel dimensions starting at index 1 while strictly preserving the batch dimension index 0.

### Formal Mathematics & Worked Numerical Micro-Example

#### Mathematical Formalism
Let $X \in \mathbb{R}^{B \times C \times H \times W}$ be an image batch. The flattening operator $\text{vec}: \mathbb{R}^{B \times C \times H \times W} \to \mathbb{R}^{B \times D}$ maps each spatial tensor $X_b \in \mathbb{R}^{C \times H \times W}$ to a vector $v_b \in \mathbb{R}^D$ where $D = C \cdot H \cdot W$:
$$v_{b, \, c \cdot H W + i \cdot W + j} = X_{b, c, i, j}$$

#### Concrete Numerical Example ($2\times 2$ Image with Batch Size 2)
Let batch tensor $X$ have shape $(2, 1, 2, 2)$:
- Image 1: $\begin{bmatrix} 10 & 20 \\ 30 & 40 \end{bmatrix}$
- Image 2: $\begin{bmatrix} 50 & 60 \\ 70 & 80 \end{bmatrix}$

Applying `nn.Flatten()`:
- Flattened Tensor shape: $(2, 4)$
- Row 1: $\begin{bmatrix} 10 & 20 & 30 & 40 \end{bmatrix}$
- Row 2: $\begin{bmatrix} 50 & 60 & 70 & 80 \end{bmatrix}$

```python
import torch
import torch.nn as nn

# Demonstration of nn.Flatten shape transformations
batch_images = torch.randn(64, 1, 28, 28) # Batch of 64 Fashion-MNIST images
flatten_layer = nn.Flatten()

flattened_batch = flatten_layer(batch_images)

print("Original Batch Shape: ", batch_images.shape)     # torch.Size([64, 1, 28, 28])
print("Flattened Batch Shape:", flattened_batch.shape)  # torch.Size([64, 784])
```

#### 🎯 Diagnostic Mini-Check
1. *What happens to spatial neighborhood information (e.g. that pixel (0,0) is touching pixel (1,0)) when an image is flattened?*  
   **Answer:** Spatial proximity is lost. Pixel (0,27) and pixel (1,0) become adjacent in the flattened vector, even though they were on opposite sides of the image horizontally. This is the fundamental structural limitation of MLPs compared to CNNs.
2. *If an RGB color image has shape `(32, 3, 32, 32)`, what is its shape after `nn.Flatten()`?*  
   **Answer:** $3 \times 32 \times 32 = 3072 \implies \text{Shape is } \mathbf{(32, 3072)}$.

---

## 3. Raw Logits, Softmax Probability Distributions, and Argmax Decisions

<a id="p3-logits"></a>

### Intuition & Physical Metaphor (The Election Votes and Podium)
Imagine a 10-candidate election.
- **Logits ($z$):** 10 judges shout out raw uncalibrated scores like `[+4.2, -1.5, +0.2, ...]`. These scores can be positive, zero, or negative.
- **Softmax ($\sigma(z)$):** The tally master exponentiates all scores ($e^z$, turning all negative numbers positive) and divides each by the total sum. Now the scores represent **percentages that sum exactly to 100%** (a valid probability distribution $P(Y=k \mid X)$).
- **Argmax ($\arg\max$):** The podium announcer looks for the candidate with the single highest percentage and declares them the winner (an integer index from $0$ to $9$).

```
   Raw Logits z in R^10            Softmax Exponentiation               Argmax Operator
   ┌────────────────────┐          ┌───────────────────────┐            ┌───────────────┐
   │ [ 2.0, -1.0, 5.0 ] │  ─────>  │ [ 0.047, 0.002, 0.95] │   ─────>   │ Winner: Index │
   │ (Raw real numbers) │          │ (Probabilities sum=1) │            │   Class = 2   │
   └────────────────────┘          └───────────────────────┘            └───────────────┘
```

### Plain-English Breakdown
- **Logits:** The unbounded scalar outputs of the network's final linear layer. Higher values mean higher model confidence.
- **Softmax:** A smooth, differentiable function that maps any real vector $z \in \mathbb{R}^K$ into a probability simplex $\Delta^K = \{p \in \mathbb{R}^K : p_i \ge 0, \sum p_i = 1\}$.
- **Argmax:** A non-differentiable step function used strictly at inference time to convert probabilities into a concrete predicted class label: $\hat{y} = \arg\max_k z_k$.
- **Why we don't compute Softmax in `forward()`:** When calculating loss, `nn.CrossEntropyLoss` mathematically combines $\log(\text{Softmax}(z))$ into a single numerically stable formula ($\log \sum e^{z_j} - z_y$). Running Softmax first and then taking $\log$ causes catastrophic floating-point underflow.

### Formal Mathematics & Worked Numerical Micro-Example

#### Mathematical Formalism
Let $z \in \mathbb{R}^K$ be the logit vector. The Softmax probability for class $i \in \{1, \dots, K\}$ is:
$$p_i = \sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$
The predicted class label is:
$$\hat{y} = \arg\max_{i \in \{1, \dots, K\}} z_i = \arg\max_{i \in \{1, \dots, K\}} p_i$$

#### Concrete Numerical Example (3-Class Logits)
Let logit vector $z = \begin{bmatrix} 2.0 \\ 1.0 \\ 0.1 \end{bmatrix}$.
1. **Exponentiate:**
   $$e^{z_1} = e^{2.0} \approx 7.389$$
   $$e^{z_2} = e^{1.0} \approx 2.718$$
   $$e^{z_3} = e^{0.1} \approx 1.105$$
   $$\text{Sum} = 7.389 + 2.718 + 1.105 = 11.212$$

2. **Normalize to Probabilities:**
   $$p_1 = \frac{7.389}{11.212} = \mathbf{0.659} \quad (65.9\%)$$
   $$p_2 = \frac{2.718}{11.212} = \mathbf{0.242} \quad (24.2\%)$$
   $$p_3 = \frac{1.105}{11.212} = \mathbf{0.099} \quad (9.9\%)$$
   $$\text{Sum of probabilities} = 0.659 + 0.242 + 0.099 = 1.000$$

3. **Argmax Decision:**
   $$\hat{y} = \arg\max([2.0, 1.0, 0.1]) = \mathbf{0} \quad (\text{Class 0 wins with } 65.9\% \text{ probability})$$

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1]]) # Batch of 1
probabilities = F.softmax(logits, dim=1)
predicted_class = torch.argmax(logits, dim=1)

print("Logits:            ", logits.numpy())
print("Softmax Probs:     ", probabilities.numpy())
print("Predicted Class ID:", predicted_class.item())
```

#### 🎯 Diagnostic Mini-Check
1. *Does adding a constant $C$ to all logits (e.g., $z' = z + 10$) change the Softmax output probabilities?*  
   **Answer:** No! $\frac{e^{z_i + C}}{\sum e^{z_j + C}} = \frac{e^C e^{z_i}}{e^C \sum e^{z_j}} = \frac{e^{z_i}}{\sum e^{z_j}}$. Softmax is shift-invariant (PyTorch exploits this property to prevent numerical overflow).
2. *If logit $z_1 = 5.0$ and logit $z_2 = 5.0$, what are their Softmax probabilities in a 2-class problem?*  
   **Answer:** $p_1 = p_2 = \frac{e^5}{e^5 + e^5} = \frac{1}{2} = 50\%$.

---

## 4. The Object-Oriented `nn.Module` Architecture (`__init__`, `super()`, `forward`, `__call__`)

<a id="p4-module"></a>

### Intuition & Physical Metaphor (The Blueprint and the Assembly Floor)
Think of `nn.Module` as an official architectural blueprint for a machine.
- **The Constructor (`__init__`):** The factory tool setup. You order all required parts (motors, conveyor belts, linear layers, activation blocks) and bolt them down to the factory floor.
- **The Mandatory Parent Call (`super().__init__()`):** The building inspector's registration stamp. It registers your factory with the central PyTorch authority, enabling automatic parameter tracking, GPU migration (`.to(device)`), and weight saving (`.state_dict()`).
- **The Conveyor Line (`forward(x)`):** The actual production process. It defines how raw material $x$ moves through each machine tool in sequence to produce finished logits.
- **Executing the Machine (`model(x)` vs `model.forward(x)`):** You press the big master start button `model(x)`. This automatically triggers internal monitoring cameras and safety hooks before running `forward(x)`. **Never call `model.forward(x)` directly!**

```
   class NeuralNetwork(nn.Module):
         │
         ├── def __init__(self):
         │       super().__init__()     <── Registers with PyTorch Autograd & Parameter Tracker
         │       self.flatten = ...     <── Tool 1
         │       self.stack = ...       <── Tool 2 (Sequential Pipeline)
         │
         └── def forward(self, x):      <── Defines the Dataflow Pipeline
                 x = self.flatten(x)
                 logits = self.stack(x)
                 return logits
```

### Plain-English Breakdown
- Every custom neural network in PyTorch must inherit from `torch.nn.Module`.
- Calling `super().__init__()` inside `__init__` is mandatory. Forgetting it will throw an error when accessing `self.parameters()`.
- Layers defined as attributes of `self` (e.g. `self.linear1 = nn.Linear(...)`) are automatically discovered by PyTorch and added to the network's parameter list.
- `nn.Sequential` is a container module that chains layers together: the output of layer 1 becomes the input to layer 2.
- In Python, defining `forward(self, x)` allows calling `model(x)` directly via Python's `__call__` dunder method. Calling `model(x)` ensures forward pre-hooks and post-hooks (used in profiling and gradient checking) are properly executed.

### Formal Mathematics & Worked Numerical Micro-Example

```python
import torch
import torch.nn as nn

# Standard Idiomatic PyTorch nn.Module Implementation
class FashionMLP(nn.Module):
    def __init__(self, in_features=784, hidden_features=512, num_classes=10):
        super().__init__() # CRITICAL: Initializes base Module machinery
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, num_classes) # Raw logits output
        )

    def forward(self, x):
        flat_x = self.flatten(x)
        logits = self.linear_relu_stack(flat_x)
        return logits

# Instantiate model
model = FashionMLP()

# Test dummy forward pass
dummy_input = torch.randn(1, 1, 28, 28)
output_logits = model(dummy_input) # Calls __call__, NOT model.forward()

print("Model Architecture:\n", model)
print("\nOutput Logits Shape:", output_logits.shape) # torch.Size([1, 10])
```

#### 🎯 Diagnostic Mini-Check
1. *Why should you write `pred = model(X)` instead of `pred = model.forward(X)`?*  
   **Answer:** `nn.Module.__call__` executes internal PyTorch hooks (e.g. activation hooks, profiler hooks, distributed data parallel hooks) before and after calling `forward()`. Calling `model.forward(X)` bypasses these essential hooks.
2. *If you define a layer as a local variable inside `forward()` (e.g. `y = nn.Linear(512, 10)(x)`), will its weights be trained by the optimizer?*  
   **Answer:** No! A layer created inside `forward()` is re-instantiated with random weights on every single forward pass and is never registered under `self.parameters()`. Layers must be defined in `__init__`.

---

## 5. Gradient Descent Optimization Mechanics and Multi-Class Cross-Entropy Loss

<a id="p5-gd"></a>

### Intuition & Physical Metaphor (The Foggy Mountain Valley)
Imagine you are hiking down a mountain in dense fog. You cannot see the lowest valley (zero error), but you can feel the slope of the ground under your boots with your walking stick.
- **The Loss Function ($L$):** The altitude meter on your watch. High altitude = terrible predictions; sea level = perfect predictions.
- **The Gradient ($\nabla_\theta L$):** The direction of steepest uphill slope.
- **Gradient Descent ($-\eta \nabla_\theta L$):** You take a step in the exact opposite direction (downhill). The learning rate $\eta$ is your stride length:
  - If $\eta$ is too large ($10.0$), you leap across valleys and fall off a cliff (loss explodes to `NaN`).
  - If $\eta$ is too small ($10^{-7}$), you take microscopic baby steps and freeze on the mountain before reaching the bottom.

```
   Loss Surface L(w)
          ^
          │        .  Current Position w_old (Loss = 2.4)
          │       / \
          │      /   \    Slope = +∇L (Points Uphill)
          │     /     \
          │    . <─────'  Step in Opposite Direction: w_new = w_old - η * ∇L
          │   /
          └──┴───────────────────────> Parameter w
             Target Valley w* (Loss = 0.1)
```

### Plain-English Breakdown
- In supervised classification, the dataset provides true integer class labels $y \in \{0, 1, \dots, K-1\}$.
- **Categorical Cross-Entropy Loss:** Measures the negative log-probability assigned by the model to the true correct class:
  $$L_{\text{CE}} = -\log\left(P(Y = y_{\text{true}} \mid X)\right) = -\log\left(\frac{e^{z_{y_{\text{true}}}}}{\sum_{j} e^{z_j}}\right) = \log\left(\sum_{j} e^{z_j}\right) - z_{y_{\text{true}}}$$
- If the model predicts $99\%$ confidence for the true class, $L \approx -\log(0.99) = 0.01$ (tiny loss).
- If the model predicts $1\%$ confidence for the true class, $L \approx -\log(0.01) = 4.60$ (huge loss).
- **Backpropagation (`loss.backward()`):** Applies the calculus Chain Rule from output loss backward through every layer, computing $\frac{\partial L}{\partial W}$ and storing it inside `W.grad`.
- **Weight Update (`optimizer.step()`):** Adjusts weights via $W \leftarrow W - \eta \cdot W.\text{grad}$.

### Formal Mathematics & Worked Numerical Micro-Example

#### Concrete Numerical Example (Cross-Entropy Loss Calculation)
Suppose true label $y = 1$ (Class 1).
Let model logits be $z = \begin{bmatrix} 2.0 \\ 1.0 \\ 0.1 \end{bmatrix}$.

1. Exponentiated terms: $e^{2.0} \approx 7.389, \quad e^{1.0} \approx 2.718, \quad e^{0.1} \approx 1.105$.
2. Sum of exponentials: $\sum e^{z_j} = 7.389 + 2.718 + 1.105 = 11.212$.
3. Softmax probability for true class ($y=1$):
   $$p_1 = \frac{2.718}{11.212} \approx 0.2424$$
4. Cross-Entropy Loss:
   $$L_{\text{CE}} = -\ln(0.2424) \approx \mathbf{1.417}$$

```python
import torch
import torch.nn as nn

logits = torch.tensor([[2.0, 1.0, 0.1]]) # Logits for class 0, 1, 2
true_target = torch.tensor([1])          # True class is 1

loss_fn = nn.CrossEntropyLoss()
loss_val = loss_fn(logits, true_target)

print(f"PyTorch CrossEntropyLoss: {loss_val.item():.4f}") # Exactly 1.4172
```

#### 🎯 Diagnostic Mini-Check
1. *Why does `nn.CrossEntropyLoss()` in PyTorch accept integer class labels (e.g. `tensor([3])`) instead of one-hot encoded vectors (e.g. `tensor([0, 0, 0, 1, 0, ...])`)?*  
   **Answer:** Memory efficiency and computational speed. Storing integers takes $\mathcal{O}(1)$ memory per sample instead of $\mathcal{O}(K)$, and indexing $z_{y}$ is faster than doing vector dot products.
2. *If the model is 100% confident in the correct class ($p_{\text{true}} = 1.0$), what is the loss?*  
   **Answer:** $-\ln(1.0) = \mathbf{0.0}$.

---

## 6. Mini-Batch Training Dynamics: Epochs, Batch Size, and Learning Rate Schedules

<a id="p6-epoch"></a>

### Intuition & Physical Metaphor (The Study Guide and Flashcards)
Imagine preparing for a comprehensive bar exam with 60,000 flashcards.
- **Sample ($x_i$):** 1 single flashcard.
- **Mini-Batch ($N=64$):** A manageable bundle of 64 flashcards. You review 64 cards, grade your mistakes, and adjust your mental memory rules once per bundle.
- **Iteration / Step:** Reviewing one bundle of 64 cards and updating your rules. (In a 60,000-card deck, there are $60,000 / 64 = 938$ iterations per full review).
- **Epoch:** Completing the **entire 60,000-card deck once**. Training for 10 epochs means cycling through the full deck 10 times.
- **Learning Rate ($\eta = 10^{-3}$):** How aggressively you rewrite your memory after each flashcard bundle.

```
   Entire Dataset: 60,000 Samples (Fashion-MNIST)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Batch 1 (64) │ Batch 2 (64) │ Batch 3 (64) │ ... │ Batch 938 (32)       │
   └──────┬───────┴──────┬───────┴──────┬───────┴─────┴──────┬───────────────┘
          │              │              │                    │
          ▼              ▼              ▼                    ▼
       Step 1         Step 2         Step 3               Step 938
     (Weight Upd)   (Weight Upd)   (Weight Upd)         (Weight Upd)
     
   ═══════════════════════════════════════════════════════════════════════════
   ONE COMPLETE PASS THROUGH ALL 938 BATCHES = 1 EPOCH
```

### Plain-English Breakdown
- **Full Batch Gradient Descent:** Uses all 60,000 samples to compute one single gradient step. Guarantees smooth descent, but is computationally slow and requires huge VRAM.
- **Stochastic Gradient Descent (Pure SGD):** Uses 1 sample per step. Fast, but gradients are noisy and bounce erratically.
- **Mini-Batch SGD (Standard in Deep Learning):** Uses a batch size of $N=64$ or $128$. Balances GPU parallelization efficiency with stochastic regularization noise that helps escape saddle points.
- **Hyperparameter Rules of Thumb:**
  - Standard starting learning rate for SGD: $\eta = 10^{-3}$ to $10^{-2}$.
  - Standard starting learning rate for Adam: $\eta = 10^{-4}$ to $10^{-3}$.
  - Default batch size: $64$ (used throughout this tutorial).

---

## 7. Execution Contexts: Training (`model.train()`) vs. Inference (`model.eval() + torch.no_grad()`) and `.item()` Scalar Extraction

<a id="p7-eval"></a>

### Intuition & Physical Metaphor (The Rehearsal vs. The Live Broadcast)
Think of a theatre production:
- **`model.train()` (Rehearsal Mode):** The actors try risky moves, stage lights are adjusted, and the director keeps a giant notebook open recording every actor's mistake to give feedback (Autograd tracking).
- **`model.eval()` + `torch.no_grad()` (Live Performance Mode):** Rehearsal is over. The director closes the notebook and puts down the pen (saves memory and stops computing gradients). The actors freeze their learned techniques and perform deterministically.
- **`loss.item()` (The Photo of the Scoreboard):** If you save the entire live theatre scoreboard along with all its electrical wires, your living room fills with cables (GPU memory leak). Calling `.item()` snaps a tiny photo of the final number and unplugs all the cables.

```
   TRAINING MODE: model.train()                EVALUATION MODE: model.eval()
   ┌────────────────────────────────┐          ┌────────────────────────────────┐
   │ • Autograd tracks graph        │          │ • with torch.no_grad():        │
   │ • Computes loss.backward()     │          │ • Zero gradient graphs built   │
   │ • optimizer.step() updates W   │          │ • Fast inference, minimal VRAM │
   │ • Dropout & BatchNorm active   │          │ • Deterministic predictions    │
   └────────────────────────────────┘          └────────────────────────────────┘
```

### Plain-English Breakdown
- `model.train()` sets `self.training = True` across all sub-modules. It does **not** train the model automatically; it merely configures layers like `nn.Dropout` and `nn.BatchNorm2d` to operate in training mode.
- `model.eval()` sets `self.training = False`. Dropout stops dropping units, and BatchNorm uses running population statistics instead of mini-batch statistics.
- `with torch.no_grad():` is a Python context manager that deactivates the Autograd engine. Tensors computed inside this block do not keep a `.grad_fn` pointer, dramatically reducing memory usage and speeding up evaluation.
- `loss.item()` converts a PyTorch 0D scalar tensor (which holds a reference to the entire autograd computation graph) into a standard Python `float`. If you store running loss using `total_loss += loss` instead of `total_loss += loss.item()`, your GPU will run out of memory (OOM) after a few hundred batches!

---

## 8. Model Serialization and Weight Persistence (`state_dict` vs `.pth`)

<a id="p8-save"></a>

### Intuition & Physical Metaphor (The Recipe Book vs. The Entire Kitchen)
Suppose you have trained a master chef robot.
- **Method A (Saving the Entire Kitchen):** You freeze the robot, the countertops, the electrical wiring, and the physical room into a giant block of ice. If you try to thaw it in a different kitchen, the wires mismatch and it fails.
- **Method B (Saving the `state_dict` Recipe):** You write down a small, elegant notebook listing the exact numeric settings for all 400,000 dials:
  `{"layer1.weight": [...], "layer1.bias": [...], ...}`.
  Any kitchen with the same robot model can buy the notebook, apply the dial settings in 1 millisecond, and cook identical meals.

```
   Trained Model in Memory                       Saved state_dict on Disk
   ┌───────────────────────────────┐             ┌───────────────────────────────┐
   │ model.linear_relu_stack.0     │  .state_dict│ "linear_relu_stack.0.weight"  │
   │ Weights: (512, 784) Tensor    │  ─────────> │ "linear_relu_stack.0.bias"    │
   │ Biases:  (512,) Tensor        │  torch.save │ "linear_relu_stack.2.weight"  │
   │ ...                           │             │ ... Saved to 'model.pth'      │
   └───────────────────────────────┘             └───────────────────────────────┘
```

### Plain-English Breakdown
- A PyTorch `state_dict` is an `OrderedDict` that maps each layer's parameter name string to its corresponding `torch.Tensor`.
- **Saving Weights (Best Practice):**
  ```python
  torch.save(model.state_dict(), "model.pth")
  ```
- **Loading Weights:**
  ```python
  model = NeuralNetwork().to(device) # 1. Create identical architecture instance
  model.load_state_dict(torch.load("model.pth", map_location=device)) # 2. Copy saved weights
  model.eval() # 3. Set to eval mode before testing
  ```

---

### You are ready for NOTES.md!
Proceed now to [NOTES.md](./NOTES.md) starting at the **Executive Summary Architecture Blueprint**.
