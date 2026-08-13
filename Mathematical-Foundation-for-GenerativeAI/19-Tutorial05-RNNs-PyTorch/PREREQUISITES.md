# Prerequisites — warm-up before Tutorial 5 (RNNs using PyTorch)

> **Do this first** if sequences, hidden states, or gates still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Course: NPTEL Mathematical Foundations of Generative AI · Tutorial 5.  
> Builds on [Tutorial 4 CNNs](../18-Tutorial04-CNNs-PyTorch/NOTES.md) and [Tutorial 3 PyTorch Basics](../17-Tutorial03-PyTorch-Basics/NOTES.md).  
> **Beginner deep warm-up:** definition · micro example · analogy · notice · mini-check.  
> **Note:** This video is **RNNs / LSTM / GRU**, not another CNN lecture.

```
  After this warm-up you can say:

  "A sequence batch is shaped (N, T, D) with batch_first=True."
  "An RNN reuses the same cell weights at every time step."
  "h_t = tanh(W_x x_t + W_h h_{t-1} + b)."
  "LSTM has two states: cell c (long) and hidden h (short)."
  "GRU has one state with reset and update gates."
  "For classification we often feed the last hidden into a Linear head."
  "torch.save(state_dict) stores weights; load needs matching architecture."
  "Train and eval loops can be reusable functions for any classifier."
```

**Warm-up → tutorial boxes**

```
  §1  What is a sequence?                 ──► Topics 1–2
  §2  Tensor (N, T, D) batch_first        ──► Topics 2–4
  §3  RNN cell + weight sharing in time   ──► Topics 3–4
  §4  Vanishing gradients → LSTM          ──► Topics 5–6
  §5  GRU vs LSTM vs RNN                  ──► Topics 5–7
  §6  Last-hidden classifier head         ──► Topics 8–9
  §7  Save / load state_dict              ──► Topic 10
  §8  Reusable train / eval loops         ──► Topics 9–10
```

---

## 1. What is a sequence?

<a id="p1-seq"></a>

### Purpose for the video

Images are a grid of pixels processed by **convolutions**. Many problems are **ordered in time or position**: words in a sentence, frames in a clip, sensor readings over time. Those are **sequences**.

### Definitions

| Term | Meaning |
|------|---------|
| **Sequence** | Ordered list of vectors $x_1, x_2, \ldots, x_T$ |
| **Time step / token** | One position $t$ in the sequence |
| **Temporal data** | Data where order matters |

### Worked micro

A weather series of length 5, each day described by 3 numbers (temp, humidity, wind):

```
  day:   1      2      3      4      5
  x_t:  [·]    [·]    [·]    [·]    [·]   each [·] ∈ R³
```

Predicting tomorrow’s weather or classifying “stormy week?” needs memory of earlier days — not treating days as a bag.

### Analogy — telling a joke

The punchline only works if you heard the **setup first**. Shuffle the sentences and the joke dies. Sequences are jokes for machines: **order is part of the meaning**.

### Notice

- CNNs are great when **local spatial** patterns matter. RNNs/LSTMs/GRUs are the lecture’s tools when **order along one axis (time)** matters.  
- Later models (Transformers) also handle sequences; this tutorial builds the classic recurrent stack.

### Mini-check

1. Give one non-image example of a sequence.  
2. Why is order important?

---

## 2. Tensor layout $(N, T, D)$ and `batch_first`

<a id="p2-ntd"></a>

### Purpose for the video

PyTorch RNN modules need a consistent axis order. This lecture uses **`batch_first=True`**.

### Definitions

| Axis | Name | Meaning |
|------|------|---------|
| **N** | batch | how many sequences in the forward |
| **T** | sequence length | how many time steps |
| **D** | input size / feature dim | size of each $x_t$ |

### Worked micro

```python
import torch
# 4 sequences, each length 5, each step has 3 features
x = torch.randn(4, 5, 3)
print(x.shape)  # torch.Size([4, 5, 3])  → (N, T, D)
print("one sequence:", x[0].shape)       # (5, 3)
print("one token:   ", x[0, 2].shape)    # (3,)  features at t=2

# Wrong mental model: (4, 3, 5) would mean length 3 and feature 5 — breaks Wx matmuls
```

Without `batch_first`, many APIs expect `(T, N, D)` instead. Always match the flag to your tensor — the lecture consistently uses **`batch_first=True`**.

### Analogy — classroom seating chart

- **N** = number of students (examples).  
- **T** = number of questions on the quiz (time steps).  
- **D** = number of score components per question (features).  

`batch_first=True` means: “list students first, then their answers in order.” `batch_first=False` is “question 1 for everyone, then question 2…” — same data, different filing cabinet.

### Notice

- Demo in the lecture: **(4, 5, 3)**.  
- Toy train later: **(32, 10, 3)** batches from a Dataset of 1000 sequences.  
- **T is not hidden size H** — T is how long the story is; H is how wide the memory vector is.

### Mini-check

1. What does `(8, 20, 50)` mean with batch_first?  
2. Is sequence length the same as hidden size?  
3. What is the shape of `x[:, -1, :]` for `x` shaped `(4, 5, 3)`?

---

## 3. RNN cell and weight sharing in time

<a id="p3-rnn"></a>

### Purpose for the video

A **recurrent cell** updates a **hidden state** $h_t$ using the new input $x_t$ and the previous $h_{t-1}$.

### Core equation (lecture)

$$
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
$$

Demo sizes: $x_t \in \mathbb{R}^3$, $h_t \in \mathbb{R}^8$ → $W_x$ is $8\times 3$, $W_h$ is $8\times 8$, $b$ is length 8.

### Unroll picture

```
  h0=0 ──► cell ──h1──► cell ──h2──► … ──► cell ──hT
              ↑           ↑                  ↑
             x1          x2                 xT
         (same W_x, W_h, b every step)
```

### Optional readout

$$
y_t = W_y h_t + b_y
$$

### Analogy — rolling journal

Each day you write a short summary ($h_t$) using **today’s news** ($x_t$) and **yesterday’s summary** ($h_{t-1}$). You use the **same pen and rules** every day (weight sharing). You do not buy a new notebook style each morning.

### Notice

- `nn.RNN(input_size=3, hidden_size=8, num_layers=1, batch_first=True)`.  
- Forward returns **`(output, h_n)`**: all time-step outputs, plus final hidden.  
- Optional $y_t = W_y h_t + b_y$ is a **readout**, not a second recurrence.

### Mini-check

1. Do time steps share weights?  
2. What is $h_0$ often initialized to?  
3. If $x_t\in\mathbb{R}^3$ and $h_t\in\mathbb{R}^8$, what shape is $W_x$?

---

## 4. Why LSTM? Vanishing gradients and two memories

<a id="p4-lstm"></a>

### Purpose for the video

Plain RNNs struggle to remember far-back signals when gradients **vanish** through many multiplies. **LSTM** adds a **cell state highway**.

### Definitions

| Piece | Role (colloquial + math) |
|-------|---------------------------|
| **Forget gate $f_t$** | How much old $c$ to keep (sigmoid ≈ %) |
| **Input gate $i_t$** | How much new candidate to write |
| **Candidate $\tilde c_t$** | New content (often tanh) |
| **Output gate $o_t$** | How much cell to expose as $h_t$ |
| **Cell $c_t$** | Long-term memory |
| **Hidden $h_t$** | Short-term / exposed state |

Sketch (lecture spirit):

$$
\begin{align*}
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t \\
h_t &= o_t \odot \tanh(c_t)
\end{align*}
$$

### Analogy — desk drawer + sticky note

- **Cell $c$** = a **desk drawer** of important papers you keep for months.  
- **Hidden $h$** = a **sticky note** on the monitor for today’s work.  
- Gates = keys: what to throw out, what to file, what to write on the sticky note.

### Notice

- LSTM outputs **two** states → `nn.LSTM` returns `(output, (h_n, c_n))`.  
- Counting every weight matrix is a good exercise (lecture assigns it).  
- On some boards the speaker may say “long-term” near $h$ while drawing — in standard language **$c$ is long-term cell**, **$h$ is short-term exposed state**. Trust the **update formulas**, not a single misspoken label.

### Mini-check

1. Name the two LSTM states.  
2. Why is sigmoid used on gates?  
3. In $c_t = f\odot c_{t-1} + i\odot\tilde c$, what does $f\approx 0$ mean?

---

## 5. GRU vs LSTM vs RNN

<a id="p5-gru"></a>

### Purpose for the video

**GRU (Gated Recurrent Unit)** is a simpler gated design: **one state**, two main gates.

### Comparison table

| Model | States | Main idea |
|-------|--------|-----------|
| **RNN** | $h$ | Single tanh mix of $x_t$ and $h_{t-1}$ |
| **LSTM** | $h$ and $c$ | Forget/input/output + cell highway |
| **GRU** | $h$ only | Reset $r$ + update $z$; blend old $h$ and candidate |

Lecture-style GRU core:

$$
\begin{align*}
r_t, z_t &= \sigma(\ldots) \\
\tilde h_t &= \tanh(W_x x_t + W_h (r_t \odot h_{t-1}) + \ldots) \\
h_t &= (1-z_t)\odot \tilde h_t + z_t \odot h_{t-1}
\end{align*}
$$

### Analogy — light dimmer

Update gate $z$ is a **dimmer** between “keep old hidden” and “use new candidate.” Reset gate $r$ decides whether yesterday’s note is allowed to influence the new draft.

### Notice

- API of `nn.GRU` feels like `nn.RNN` (one hidden, not separate cell).  
- Prefer thinking **gates as linear+σ transforms**, not only marketing names.

### Mini-check

1. How many states does a GRU carry?  
2. Which model has an explicit cell $c$?

---

## 6. Last-hidden classifier head

<a id="p6-head"></a>

### Purpose for the video

For **sequence classification**, compress the whole sequence into a vector, then map to class logits.

### Pattern

```
  X (N,T,D)  →  LSTM/GRU/RNN  →  h_last (N, H)  →  Linear(H, C)  →  logits (N,C)
```

Often: take **last time step** of the output, or the final `h_n` (for single-layer, carefully index).

### Worked micro

```python
# conceptual
out, (h_n, c_n) = lstm(x)      # out: (N,T,H)
logits = fc(h_n[-1])           # last layer's h: (N,H) → (N,num_classes)
# or: logits = fc(out[:, -1, :])
```

### Analogy — book report

You read all chapters in order (recurrent steps), then write **one summary paragraph** (last $h$) and a **grade** (class logits). You could grade after every chapter ($y_t$ each step) — possible, not required for this toy.

### Notice

- Binary: either **2 logits + CE** or **1 logit + BCE**; lecture uses 2-class CE.  
- Dummy demo: `(8,5,3) → (8,2)`.

### Mini-check

1. Why take the *last* hidden for whole-sequence labels?  
2. What loss wants logits not softmax?

---

## 7. Saving and loading weights

<a id="p7-save"></a>

### Purpose for the video

A **model** at inference time is its **weights**. Saving avoids re-training from scratch; loading enables fine-tune later.

### Pattern

```python
torch.save(model.state_dict(), "lstm_classifier.pth")

model2 = LSTMClassifier(input_dim=3, hidden_dim=32, num_classes=2)
model2.load_state_dict(torch.load("lstm_classifier.pth", map_location=device))
model2.to(device)
model2.eval()
```

### Analogy — recipe card

Training invents the recipe. `state_dict` is the **index card of measurements**. Loading pours those measurements into a **same-shaped bowl** (identical architecture). Wrong bowl size → crash.

### Notice

- Extension **`.pth`** common.  
- `map_location=device` places tensors on CPU/GPU correctly.  
- Same pattern works for CNN/MLP (lecture says so).

### Mini-check

1. What fails if you load hidden_dim=32 weights into hidden_dim=35?  
2. Why `model.eval()` after load for inference?

---

## 8. Reusable train / eval loops

<a id="p8-loops"></a>

### Purpose for the video

CNN, MLP, and sequence classifiers share the **same classification training spine**.

### Train one epoch (idea)

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    # for batch: to(device) → logits → loss → zero_grad → backward → step
    # track loss / accuracy; return averages
```

### Evaluate (idea)

```python
def evaluate(model, loader, criterion, device):
    model.eval()
    with torch.no_grad():
        # forward only; accumulate loss / correct
    # return averages  (no optimizer)
```

### Analogy — gym circuit card

Whether you train chest or legs, the **circuit order** (warmup → sets → log) is the same. Architecture is the exercise; train/eval functions are the **clipboard**.

### Notice

- Call `train_one_epoch` once per epoch.  
- Eval needs **no** optimizer and **no** grad.  
- Next tutorials reuse these helpers for pretrained models.

### Mini-check

1. Does `evaluate` call `optimizer.step`?  
2. List the five train-step moves inside the batch loop.

---

### Paper check

1. Shape of batch-first sequence with 16 examples, length 12, feature 8?  
2. RNN update formula for $h_t$?  
3. LSTM’s two states?  
4. GRU gate names?  
5. What vector goes into the classification Linear?  
6. API to save weights?

**Peek:** (1) `(16,12,8)` (2) $\tanh(W_x x_t+W_h h_{t-1}+b)$ (3) $h$, $c$ (4) reset, update (5) last hidden (6) `torch.save(model.state_dict(), path)`

---

Ready → [NOTES.md](./NOTES.md) (start at **Executive Summary**).  
Quiz: [quiz.html](./quiz.html) Part A = this file.
