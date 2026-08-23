# Tutorial 5 — RNNs using PyTorch

**Video:** [Tutorial 5 : RNNs using PyTorch](https://www.youtube.com/watch?v=k6zF2NsvVrk) · NPTEL / IISc  
**Warm-up first:** [PREREQUISITES.md](./PREREQUISITES.md)  
**Previous:** [Tutorial 4 — CNNs](../18-Tutorial04-CNNs-PyTorch/NOTES.md)  
**Course:** Mathematical Foundations of **Generative AI** (~38 min)  
**Speaker:** NPTEL IISc · Sequences, RNN, LSTM, GRU, save/load, reusable loops  

> **Clarify:** Playlist item is **RNNs**, not a second CNN tutorial (despite similar “code + PyTorch” framing).

---

## Table of Contents

1. [Topic 1 — Recap sequences intro](#topic-1-recap-sequences-intro-0003–0131) (00:03–01:31)
2. [Topic 2 — Sequence tensor NTD](#topic-2-sequence-tensor-ntd-0131–0429) (01:31–04:29)
3. [Topic 3 — RNN cell math unroll](#topic-3-rnn-cell-math-unroll-0429–0924) (04:29–09:24)
4. [Topic 4 — RNN output hidden shapes](#topic-4-rnn-output-hidden-shapes-0924–1101) (09:24–11:01)
5. [Topic 5 — LSTM gates cell](#topic-5-lstm-gates-cell-1101–1659) (11:01–16:59)
6. [Topic 6 — nn.LSTM API](#topic-6-nnlstm-api-1659–1926) (16:59–19:26)
7. [Topic 7 — GRU gates](#topic-7-gru-gates-1926–2422) (19:26–24:22)
8. [Topic 8 — Classifier last hidden](#topic-8-classifier-last-hidden-2422–2622) (24:22–26:22)
9. [Topic 9 — Toy dataset train](#topic-9-toy-dataset-train-2622–2951) (26:22–29:51)
10. [Topic 10 — Save load reusable recap](#topic-10-save-load-reusable-recap-2951–3815) (29:51–38:15)
11. [Apply it (scenarios)](#apply-it-scenarios)
12. [External references](#external-references)
13. [Sources](#sources)

---

## Executive Summary — architecture of this lecture

Last hour classified MNIST photos with a **CNN** (convolutional neural network). This hour treats data that arrives **in order**: weather days, tokens, any sequence. You store a batch as **(N, T, D)** — **N** sequences, **T** time steps, **D** features per step — and run one shared **RNN** (recurrent neural network) cell along time. **LSTM** (long short-term memory) and **GRU** (gated recurrent unit) add gates so the memory can last longer than a plain tanh cell. A last-hidden Linear head classifies a toy sequence; you save the weights and factor the train/eval loops. The payoff is the same CrossEntropy + Adam contract as images, now with memory across time.

**Worldview arc:** from “CNN on MNIST images” **to** “RNN/LSTM/GRU on sequences + production hygiene (save, load, reusable loops).”

**Hour at a glance (whole video).** The first half is *how a sequence sits in a tensor, then how a cell steps through it*. He closes the CNN chapter (accuracy was fine because MNIST is balanced) and builds a dummy batch `(4, 5, 3)` so you can name N, T, and D with `batch_first=True`. The plain RNN comes next because you cannot call `nn.LSTM` honestly until you have written $h_t=\tanh(W_x x_t+W_h h_{t-1}+b)$ and seen the **same** weights unroll across T. He prints `output` (every step) versus `hidden` (final state), then motivates LSTM: many tanh multiplies make early-step gradients vanish, so the cell needs a second highway $c_t$ and four gates (forget, input, candidate, output).

The rest of the hour is *the gated APIs, one classifier, and hygiene*. `nn.LSTM(..., batch_first=True)` returns three objects: all-step output plus the pair $(h_n, c_n)$. GRU keeps **one** state and two main gates (reset, update); its call looks like `nn.RNN`. He builds `LSTMClassifier`: last hidden → Linear → logits, dummy `(8,5,3)→(8,2)`. A toy Dataset (sum rule) trains for a few epochs with the usual mini-batch loop. He finishes by writing `state_dict` to a `.pth` file, loading into a matching architecture, and extracting `train_one_epoch` / `evaluate` so the next unit (pretrained vision) can reuse the same clipboard.

### System context

```
  ╔══════════════════════════════════════╗
  ║ Outside: Transformers, big NLP/ASR   ║
  ║ Outside: VGG/ResNet MRI (next unit)  ║
  ╚══════════════╤═══════════════════════╝
                 │ this tutorial (~38 min)
                 ▼
        ┌────────────────────────────┐
        │ Sequences in PyTorch       │
        │ RNN · LSTM · GRU · head    │
        │ save/load · train helpers  │
        └────────────────────────────┘
```

### Main blueprint

```
  X ∈ R^{N × T × D}   (batch_first=True)
          │            [N = batch, T = time steps, D = features per step]
          ▼
  Shared cell over time:                 [same weights at every t]
    RNN:  h_t = tanh(W_x x_t + W_h h_{t-1} + b)   [plain recurrent cell]
    LSTM: gates f,i,o + c̃ → c_t, h_t              [long + short memory]
    GRU:  r, z + candidate → h_t                   [one state, two gates]
          │
          ├── output (N, T, H)   all steps
          └── h_n [/ c_n]        final states
          │
          ▼
  h_last → Linear → logits (N, C)        [one summary vector per sequence]
          │
          ▼
  CE + Adam mini-batch train
  torch.save(state_dict) / load_state_dict
  train_one_epoch · evaluate (reusable)
```

### Scenario walkthrough

Walk this **one** story through the blueprint above. Each step answers “so what?” for the next box.

**Story:** one short sequence of 5 vectors (think five days of weather, three numbers each day) must become a **yes/no class score**, then the same path must train on a pile of such sequences.

1. **Why leave CNNs?** A photo is a grid. This sequence is a **comic strip**: order is part of the meaning. Accuracy-on-MNIST was the last vision report card; today the axis of interest is **time**. That is the SETUP box.

2. **How do you store the strip?** Shape `(4,5,3)` with `batch_first=True`: four stories, five steps, three features. Swap T and D and every later matrix multiply explodes. That is the DATA box.

3. **What does the cell do at one step?** $h_t=\tanh(W_x x_t+W_h h_{t-1}+b)$. Day 3’s hidden state uses day 3’s input **and** day 2’s memory. **Same** $W_x,W_h$ at every t — that is weight sharing in time.

4. **What comes out of `nn.RNN`?** `output` is the hidden vector at **every** step `(N,T,H)`. `hidden` is the **last** state. You need both names before you pick a classifier.

5. **Why LSTM?** A long chain of tanh multiplies can wipe the early-day signal (vanishing gradients). LSTM (long short-term memory) adds a cell highway $c_t$ and four gates: forget, input, candidate, output. Two states: $c$ long, $h$ short.

6. **What does the PyTorch call return?** `nn.LSTM(..., batch_first=True)` → `output, (h_n, c_n)`. Three objects, not two. Index them wrong and you feed the Linear the cell by mistake.

7. **Why mention GRU?** A gated recurrent unit keeps **one** state and two gates (reset, update). Same job, smaller API — it looks like `nn.RNN`. Use it when you want gates without a separate $c$.

8. **How do you classify the whole strip?** Take the **last** hidden (the summary after day 5) → Linear → logits. Dummy check: `(8,5,3)→(8,2)`. Using `output[:,0]` is the first-day summary, not the whole story.

9. **How do you train?** A toy Dataset (label from a sum rule), DataLoader batches, CrossEntropy + Adam, many updates per epoch — the Tutorial 3/4 spine.

10. **How do you keep the work?** `torch.save(model.state_dict(), "*.pth")` and load into a **same-shaped** net. Then wrap `train_one_epoch` / `evaluate` so the next tutorial (pretrained MRI) does not rewrite the clipboard.

```
  one sequence of 5 steps
         │  file as (N, T, D), batch first
         ▼
  shared cell: RNN → LSTM / GRU
         │  last hidden is the summary
         ▼
  Linear → class logits
         │  mini-batch CE + Adam
         ▼
  save state_dict · reusable train/eval   =  a sequence classifier
```

### Failure / contrast path

```
  Feed (T, N, D) while batch_first=True          ──X──► silent axis swap
  Use output[:,0] instead of the last time step  ──X──► summary of the start, not the story
  Load weights into a different hidden_dim       ──X──► size mismatch crash
  Forget model.eval / no_grad at test            ──X──► wrong mode and a wasted graph
  Softmax the logits, then call CrossEntropy     ──X──► double softmax, bad gradients
```

### STOP / out of scope

Attention / Transformers; multi-layer deep analysis; real language corpora; VGG/ResNet/MRI (next tutorial).

### Load-bearing claims (closed-book)

- Sequence batches use **(N, T, D)** with **`batch_first=True`**: batch, time, features.
- An RNN **shares** $W_x$ and $W_h$ across time; $h_t=\tanh(W_x x_t+W_h h_{t-1}+b)$.
- A forward pass returns the **hidden vector at every step** and the **final hidden**.
- LSTM fixes **vanishing gradients** with **two states** ($c$ long, $h$ short) and four gates.
- GRU keeps **one state** and two main gates: reset and update.
- Sequence classification: **last hidden → Linear → logits**.
- **`state_dict` save/load** only works if the loaded net has the **same architecture**.
- **`train_one_epoch` / `evaluate`** are reusable for any classifier in this bootcamp.

**Speaker / course:** NPTEL IISc · Mathematical Foundations of Generative AI · Tutorial 5.

---

## Topic 1: Recap sequences intro (00:03–01:31)

### Where this sits on the master map

**SETUP** — Close the CNN chapter and open **temporal sequences**. Warm-up: [sequence](./PREREQUISITES.md#p1-seq).

### Board / screenshot

![Recap sequences intro](./screenshots/composites/ch01-topic-01-recap-sequences-panel1of1.png)

**Figure — ~00:03–01:31:** Bridge from CNN MNIST; introduce sequences / RNN / LSTM / GRU.

### What he is establishing

Last tutorial: **image classification with CNNs** on **MNIST**, including training and evaluation. Because the data was **balanced**, the metric was plain **accuracy**; he reminds you that **precision, recall, F1, AUC** exist for skewed settings.

Today continues from that point into **sequences** — **temporal data**. Tools: **recursive / recurrent neural networks (RNNs)** and upgraded versions **LSTMs** and **GRUs**.

You can now place this lecture after CNN training. Still missing: how a sequence is stored as a tensor.

A common trap is treating this as “another CNN notebook” — the data axis of interest is **time**, not height/width. Another trap: assuming every classification problem uses accuracy; the instructor flags **precision / recall / F1 / AUC** when classes are skewed.

### Analogy for this topic only

Tutorial 4 was **photos** on a desk. Tutorial 5 is **stories told in order** — a comic strip, not a single frame. Same kitchen (PyTorch Module, CE, Adam); new ingredient is **memory across steps**. Accuracy on MNIST was like grading a fair multiple-choice test; skewed data is like grading a test where almost every answer is “A” — you need different scorecards.

Question: **Why mention precision/recall if MNIST used accuracy?**

In lecture words: this box is the handoff from vision to sequences.

### Local picture

```
  Tutorial 4: CNN · MNIST · accuracy (balanced)
          │
          ▼
  Tutorial 5: sequences (temporal)
          │
          ├── RNN
          ├── LSTM
          └── GRU
```

**Notice:** evaluation metric choice depends on class balance — carried over idea, not re-derived.

### Bridge

You need a concrete **batch of sequences** with clear **N, T, D** axes before any `nn.RNN` call.

---

## Topic 2: Sequence tensor NTD (01:31–04:29)

### Where this sits on the master map

**DATA SHAPE** — Build `(batch, seq_len, input_dim)` with `batch_first` semantics. Warm-up: [N,T,D](./PREREQUISITES.md#p2-ntd).

### Board / screenshot

![Sequence tensor NTD](./screenshots/composites/ch02-topic-02-sequence-tensor-panel1of1.png)

**Figure — ~01:31–04:29:** Four examples × length 5 × $R^3$; shape (4,5,3); import/device refresh.

### What he is establishing

Construct a sequence batch of **size 4**. Each example has **sequence length 5**: $x_{i1},\ldots,x_{i5}$. Each $x_{ij} \in \mathbb{R}^3$.

Layout with **batch first**: dimension **0 = batch**, **1 = sequence length**, **2 = input feature dimension**. Printed shape **(4, 5, 3)**.

If the Colab session is cold, re-run the **import + device** cells from PyTorch basics before the RNN cell.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# batch=4, T=5, D=3
x = torch.randn(4, 5, 3)
print(x.shape)  # torch.Size([4, 5, 3])
N, T, D = x.shape
print(f"N={N}, T={T}, D={D}")
```

You can now allocate sequence batches correctly. Still missing: how an **RNN cell** turns each $x_t$ into $h_t$.

A common trap is swapping T and D — length 3 and feature 5 looks “almost right” until matmul shapes explode.

### Analogy for this topic only

Four students (N), each answers a 5-question quiz (T), each answer is a 3-part rubric (D). The tensor is the stack of answer sheets.

Question: **Which axis is sequence length in `(4,5,3)` with batch_first?**

In lecture words: this box is the sequence data contract.

### Local picture

```
  example i:  x_i1, x_i2, x_i3, x_i4, x_i5
                 each x_ij ∈ R³

  X.shape = (4, 5, 3)
             N  T  D
```

**Notice:** `batch_first=True` later must match this layout.

### Bridge

Define **`nn.RNN`** and the **tanh recurrence** that reuses weights across the five steps.

---

## Topic 3: RNN cell math unroll (04:29–09:24)

### Where this sits on the master map

**RNN METHOD** — Cell equation, matrix sizes, unroll with shared weights, optional $y_t$. Warm-up: [RNN cell](./PREREQUISITES.md#p3-rnn).

### Board / screenshot

![RNN cell math unroll](./screenshots/composites/ch03-topic-03-rnn-cell-math-panel1of1.png)

**Figure — ~04:29–09:24:** $h_t=\tanh(\ldots)$; Wx 8×3, Wh 8×8; unrolled chain; y from h.

### What he is establishing

**`nn.RNN`** processes **one timestamp at a time**. Demo: each $x_t$ size **3** maps into hidden size **8**; **num_layers=1**; **`batch_first=True`** so `shape[0]` is batch.

Recurrence:

$$
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
$$

with $x_t\in\mathbb{R}^3$, $h_t\in\mathbb{R}^8$. Compatible sizes: $W_x\in\mathbb{R}^{8\times3}$, $W_h\in\mathbb{R}^{8\times8}$, bias length 8. **tanh** is elementwise.

**Unroll:** start from $h_0$ (often zero). Feed $x_1$ → $h_1$; **same cell / same weights** with $x_2$ → $h_2$; … → $h_T$. Optional readout $y_t = W_y h_t + b_y$ at each step.

```python
rnn = nn.RNN(
    input_size=3,
    hidden_size=8,
    num_layers=1,
    batch_first=True,
)
x = torch.randn(4, 5, 3)
output, h_n = rnn(x)
print(output.shape, h_n.shape)
# output: (4, 5, 8)  all time steps
# h_n:    (1, 4, 8)  num_layers × batch × hidden
```

You can now explain the shared-cell recurrence. Still missing: what **output vs hidden** mean in the API return.

A common trap is thinking a new set of weights is learned for each time index — that would not be an RNN.

### Analogy for this topic only

One **stamp** (weights) pressed on every day of the diary ($x_t$), each time dipping in yesterday’s ink ($h_{t-1}$). The stamp does not change mid-week.

Question: **What stays shared from $t=1$ to $t=5$?**

In lecture words: this box is the atomic RNN.

### Local picture

```
  h0 ─►[cell]─h1─►[cell]─h2─► … ─►[cell]─hT
           ↑         ↑                ↑
          x1        x2               xT
        Wx,Wh,b identical every step

  h_t = tanh(Wx x_t + Wh h_{t-1} + b)
  y_t = Wy h_t + by   (optional)
```

**Notice:** first axis of module input is batch when `batch_first=True`.

### Bridge

Read the **two tensors** returned by `rnn(x)` and match them to the board’s $y_{1:T}$ and $h$ story.

---

## Topic 4: RNN output hidden shapes (09:24–11:01)

### Where this sits on the master map

**RNN API SHAPES** — `output` vs `hidden` for batch (4,5,3) and H=8. Warm-up: [RNN](./PREREQUISITES.md#p3-rnn).

### Board / screenshot

![RNN output hidden shapes](./screenshots/composites/ch04-topic-04-rnn-shapes-panel1of1.png)

**Figure — ~09:24–11:01:** Return pair; output all steps (4,5,8); hidden last (layers,batch,8).

### What he is establishing

`nn.RNN` returns **two** objects: **`output`** and **`hidden`**.

- **output:** for each batch item, the representation at **every** time step — “$y_1,\ldots,y_n$” / per-step hidden projections in API terms → shape **(4, 5, 8)** for the demo.  
- **hidden:** the **final** hidden state(s) — for each of the 4 examples, an **8-vector** (and a layer axis when stacked).

Input **(4,5,3)** → after H=8 RNN: time-wise maps stay length 5 with width 8.

```python
x = torch.randn(4, 5, 3)
rnn = nn.RNN(3, 8, batch_first=True)
output, h_n = rnn(x)
print("output", output.shape)  # (4, 5, 8)  all time steps
print("h_n   ", h_n.shape)     # (1, 4, 8)  layers × batch × H
# last step of output relates to final h (single layer):
print("match last step?", torch.allclose(output[:, -1, :], h_n[-1], atol=1e-5))

# Wrong: feed whole (N,T,H) into Linear(H, C) without picking a time
# Right for whole-seq class: Linear on output[:, -1, :] or h_n[-1]
```

You can now unpack RNN returns. Still missing: **why LSTM** when plain RNNs struggle on long chains.

A common trap is using `output` shape as if it were `(N, H)` for a Linear without selecting a time index — you will get a rank/size error or silently average the wrong axis.

### Analogy for this topic only

**output** is the **full CCTV recording** of every second of the shift. **hidden** is the **guard’s final briefing** after the doors close. Whole-sequence classification often only needs the briefing; token-level tagging needs every frame of the recording.

Question: **What is `output.shape[1]`?**

In lecture words: this box is shape hygiene for recurrent modules.

### Local picture

```
  x (4,5,3)
     │
     ▼
  RNN H=8
     │
     ├── output (4,5,8)  ← all T
     └── h_n    (1,4,8)  ← final (per layer)
```

**Notice:** layer axis on `h_n` is first when using the default layout of RNN hidden states.

### Bridge

Long dependencies motivate **LSTM** with an explicit **cell state** and **gates**.

---

## Topic 5: LSTM gates cell (11:01–16:59)

### Where this sits on the master map

**LSTM METHOD** — Vanishing gradients; four gates; $c_t$ and $h_t$ updates. Warm-up: [LSTM](./PREREQUISITES.md#p4-lstm).

### Board / screenshot

![LSTM gates cell](./screenshots/composites/ch05-topic-05-lstm-gates-panel1of1.png)

**Figure — ~11:01–16:59:** Forget/input/candidate/output; cell update; h from o and tanh(c).

### What he is establishing

Imagine a 100-step series where step 1 holds a critical signal. A plain RNN multiplies that signal through many tanh layers; gradients often **shrink toward zero** — the **vanishing gradient** problem. **LSTM** is the fix the lecture adopts.

Concrete picture: at each time you still read $x_t$ (size 3 in the earlier demo), but you maintain **two** vectors of size H (demo H=8): a **cell** $c_t$ (long-term drawer) and a **hidden** $h_t$ (short sticky note). Four soft switches (each a linear mix of $x_t$ and $h_{t-1}$ passed through **sigmoid** or **tanh**) decide the rewrite:

- **Forget** $f_t$: how much of yesterday’s drawer $c_{t-1}$ to keep.  
- **Input** $i_t$: how much new ink to accept.  
- **Candidate** $\tilde c_t$: proposed new content (tanh).  
- **Output** $o_t$: how much of the drawer to show as $h_t$.

The numbers update as:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t,\qquad
h_t = o_t \odot \tanh(c_t).
$$

Sigmoid values act like **percentages** — that is why “forget 20% / write 80%” language appears. Same weights reuse every step (like RNN). He asks you to **count matrix sizes** as homework, mirroring the $W_x\in\mathbb{R}^{8\times3}$ exercise.

```python
# Tiny numerical feel for gates (not the full nn.LSTM internals)
# Suppose H=1 for illustration only:
f = torch.tensor(0.2)   # forget 20% of old cell? actually keep fraction f
i = torch.tensor(0.7)
c_prev = torch.tensor(1.0)
c_tilde = torch.tensor(0.5)
c = f * c_prev + i * c_tilde   # 0.2*1 + 0.7*0.5 = 0.55
o = torch.tensor(0.9)
h = o * torch.tanh(c)
print(c.item(), h.item())
```

You can now narrate LSTM memory flow with a worked mental example. Still missing: **`nn.LSTM` call signature** and the three-way return.

A common trap is memorizing gate names without the $c_t$ formula — the highway is the point.

### Analogy for this topic only

Think of a **hotel front desk**. The **safe** in the back office is the cell state: guests’ passports stay there for days. The **sticky note** on the counter is the hidden state: only what today’s clerk needs. Forget/input/output are **keys and stamps** that open the safe a little, file a new form, or copy a line onto the sticky note. The math symbols just rename those keys.

Question: **Which state is called long-term memory?**

In lecture words: this box is the LSTM storyboard.

### Local picture

```
  x_t, h_{t-1}, c_{t-1}
         │
    ┌────┼────┬────────┐
    f    i    c̃        o     (σ, σ, tanh, σ)
    │    │    │        │
    └─► c_t = f⊙c_{t-1} + i⊙c̃
              │
              h_t = o ⊙ tanh(c_t)
```

**Notice:** same weights reused across time, just like RNN — only richer internals.

### Bridge

Declare **`nn.LSTM`** with `batch_first` and inspect **output, h, c** shapes.

---

## Topic 6: nn.LSTM API (16:59–19:26)

### Where this sits on the master map

**LSTM API** — Layer weights, `nn.LSTM(...)`, three return tensors. Warm-up: [LSTM](./PREREQUISITES.md#p4-lstm).

### Board / screenshot

![nn.LSTM API](./screenshots/composites/ch06-topic-06-nn-lstm-api-panel1of1.png)

**Figure — ~16:59–19:26:** Weight reuse across time; multi-layer note; LSTM(3,8) shapes.

### What he is establishing

You can still form $y_t$ by a linear map of $h_t$. One cell has **many** weight matrices; those weights **repeat across time**. If you stack **layers**, **layer 1 and layer 2 have different weights**.

API demo: `input_size=3`, `hidden_size=8`, `num_layers=1`, **`batch_first=True`**. Forward yields **output**, **hidden**, and **cell** shapes — three pieces because of **two states**.

```python
lstm = nn.LSTM(
    input_size=3,
    hidden_size=8,
    num_layers=1,
    batch_first=True,
)
x = torch.randn(4, 5, 3)
output, (h_n, c_n) = lstm(x)
print(output.shape)           # (4, 5, 8)
print(h_n.shape, c_n.shape)   # (1, 4, 8), (1, 4, 8)

# Two layers → h_n / c_n first dim = 2
lstm2 = nn.LSTM(3, 8, num_layers=2, batch_first=True)
out2, (h2, c2) = lstm2(x)
print(h2.shape, c2.shape)     # (2, 4, 8) each layer has its own state

# Param count (all gates packed inside module)
print("LSTM params:", sum(p.numel() for p in lstm.parameters()))
```

You can now call LSTM like RNN but unpack **two** state tensors. Still missing: the **simpler gated cousin GRU**.

A common trap is writing `output, h = lstm(x)` — missing `c` unpacking will error. Another trap: stacking layers but still reading only `h_n[0]` when you wanted the top layer’s summary (`h_n[-1]`).

### Analogy for this topic only

RNN return was a **duet**. LSTM return is a **trio**: the full performance tape (output), the final short memo on the counter ($h$), and the final contents of the back-office safe ($c$). Multi-layer means **stacked trios** — each floor of the hotel has its own safe and sticky notes.

Question: **Why does LSTM return three objects?**

In lecture words: this box is the PyTorch LSTM surface.

### Local picture

```
  nn.LSTM(3, 8, batch_first=True)
  x (N,T,3) → output (N,T,8), h_n (L,N,8), c_n (L,N,8)

  time share: same W for all t
  layer stack: different W per layer
```

**Notice:** `L` = `num_layers`.

### Bridge

**GRU** reduces to **one state** while keeping gating — next board.

---

## Topic 7: GRU gates (19:26–24:22)

### Where this sits on the master map

**GRU METHOD** — Reset/update gates; single state; API like RNN. Warm-up: [GRU](./PREREQUISITES.md#p5-gru).

### Board / screenshot

![GRU gates](./screenshots/composites/ch07-topic-07-gru-panel1of1.png)

**Figure — ~19:26–24:22:** r and z gates; candidate with reset; h blend; nn.GRU shapes.

### What he is establishing

**GRUs (Gated Recurrent Units)** are **simpler than LSTMs** but still gated. **RNN: one state. LSTM: two states. GRU: one state again.**

Gates: **reset $r_t$**, **update $z_t$** (both sigmoid mixes of $x_t$ and $h_{t-1}$). Candidate $\tilde h_t$ uses **tanh**, with reset applied as $r_t \odot h_{t-1}$ (why “reset”). Final:

$$
h_t = (1-z_t)\odot \tilde h_t + z_t \odot h_{t-1}
$$

(sigmoid $z$ as soft mix percentage). Optional $y_t$ from $h_t$. I/O similar to RNN; use **`nn.GRU`**.

```python
gru = nn.GRU(input_size=3, hidden_size=8, num_layers=1, batch_first=True)
output, h_n = gru(torch.randn(4, 5, 3))
print(output.shape, h_n.shape)  # (4,5,8), (1,4,8)  — no separate c_n

# Side-by-side family card
x = torch.randn(2, 5, 3)
for name, mod in [
    ("RNN", nn.RNN(3, 8, batch_first=True)),
    ("GRU", nn.GRU(3, 8, batch_first=True)),
    ("LSTM", nn.LSTM(3, 8, batch_first=True)),
]:
    y = mod(x)
    if name == "LSTM":
        out, (h, c) = y
        print(name, out.shape, h.shape, c.shape)
    else:
        out, h = y
        print(name, out.shape, h.shape)
```

You can now contrast the three recurrent families. Still missing: **wiring a classifier** on the last hidden state.

A common trap is assuming GRU has a separate $c$ tensor like LSTM — it does not.

### Analogy for this topic only

Picture a **DJ crossfader**. The update gate is the slider between the **old track** still playing and the **new take** you just recorded. The reset gate is a mute button on yesterday’s track before you even start the new take. Fewer knobs than an LSTM studio — still more control than a plain RNN.

Question: **Which model is simpler and still gated: GRU or LSTM?**

In lecture words: this box completes the cell zoo.

### Local picture

```
  RNN:  h only, tanh mix
  LSTM: h + c, four gates
  GRU:  h only, r + z (+ candidate)

  h_t = (1-z)⊙h̃ + z⊙h_{t-1}
```

**Notice:** prefer “transforms + $\sigma$/tanh” over memorizing marketing names alone.

### Bridge

Build **`LSTMClassifier`**: last hidden → **Linear** → class logits; smoke-test shapes.

---

## Topic 8: Classifier last hidden (24:22–26:22)

### Where this sits on the master map

**HEAD** — Sequence classification via last hidden + FC. Warm-up: [head](./PREREQUISITES.md#p6-head).

### Board / screenshot

![Classifier last hidden](./screenshots/composites/ch08-topic-08-classifier-dummy-panel1of1.png)

**Figure — ~24:22–26:22:** LSTM + FC; dummy batch (8,5,3)→(8,2).

### What he is establishing

For classification you need a **fully connected** map from the sequence summary to classes. Standard choice: take the **hidden state at the last time** (last instance) and Linear it to `num_classes`. You *could* map every time step if the task needs it.

**LSTMClassifier** fields: `input_dim`, `hidden_dim`, `num_classes`; `nn.LSTM(..., batch_first=True)`; forward returns logits from **last h**. LSTM returns **output, hidden, cell** — use the hidden for the head.

Dummy: `input_dim=3`, `hidden=16`, `classes=2`; `X` with **8** examples, **T=5**, **D=3** → logits **(8, 2)**. Binary alternative: one logit + sigmoid — optional.

```python
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (N, T, D)
        output, (h_n, c_n) = self.lstm(x)
        # h_n: (1, N, H) → last layer hidden (N, H)
        last_h = h_n[-1]
        logits = self.fc(last_h)  # (N, num_classes)
        return logits

model = LSTMClassifier(3, 16, 2)
logits = model(torch.randn(8, 5, 3))
print(logits.shape)  # torch.Size([8, 2])
```

You can now attach a decision head. Still missing: a **labeled toy Dataset** and a real train loop.

A common trap is `h_n[0]` vs `h_n[-1]` multi-layer confusion — for one layer they match; be explicit.

### Analogy for this topic only

The LSTM reads the whole novel; the Linear is the **exam blue book** with two score boxes. You hand in **one** blue book per novel (last $h$), not one per page — unless the teacher asked for page-level grades.

Question: **What is the shape of logits for batch 8 and 2 classes?**

In lecture words: this box is the architecture used in training next.

### Local picture

```
  (N,T,D) → LSTM → h_last (N,H) → Linear → (N,C)
  demo: (8,5,3) → (8,2)
```

**Notice:** CE still wants **logits**, not softmax probs.

### Bridge

Manufacture a **1000-sample** Dataset with a simple sum rule and train with Adam.

---

## Topic 9: Toy dataset train (26:22–29:51)

### Where this sits on the master map

**DATA + TRAIN** — Custom Dataset, loader, CE/Adam loop. Warm-up: [head](./PREREQUISITES.md#p6-head) · [loops](./PREREQUISITES.md#p8-loops).

### Board / screenshot

![Toy dataset train](./screenshots/composites/ch09-topic-09-toy-train-panel1of1.png)

**Figure — ~26:22–29:51:** Sum-based labels; DataLoader (32,10,3); train curves.

### What he is establishing

Toy rule: if a **sum of designated sequence values** (he walks components of each $x_t$) is **> 0**, label **1**, else **0**.

Custom **`Dataset`**: must implement `__init__`, `__len__`, `__getitem__`. Generate **1000** samples, **T=10**, **D=3** random features; labels from the rule. **`DataLoader` batch_size=32** → batches **(32, 10, 3)**.

Train: **LSTM** with `input_dim=3`, `hidden_dim=32`, `num_classes=2`; **`nn.CrossEntropyLoss`**; **`Adam` lr=1e-3**; **10 epochs**. Loop is the **same story** as MLP/CNN: device port → forward → loss → zero_grad → backward → step. Loss decreases; accuracy rises.

```python
from torch.utils.data import Dataset, DataLoader

class SumRuleDataset(Dataset):
    def __init__(self, n=1000, T=10, D=3):
        self.x = torch.randn(n, T, D)
        # lecture-style: sum selected features across time → label
        s = self.x[:, :, :2].sum(dim=(1, 2))  # illustrative
        self.y = (s > 0).long()

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

loader = DataLoader(SumRuleDataset(), batch_size=32, shuffle=True)
xb, yb = next(iter(loader))
print(xb.shape, yb.shape)  # (32,10,3), (32,)

model = LSTMClassifier(3, 32, 2).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = crit(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
```

You can now train a sequence classifier end-to-end. Still missing: **saving weights** and **factoring** the repeated train/eval code.

A common trap is forgetting `.long()` labels for CE integer targets.

### Analogy for this topic only

The toy label is a **fake exam key** so you can grade whether your “student” (LSTM) learned the grading rule. Same red-pen loop as CNN day — only the homework is sequences.

Question: **What batch shape does the loader yield?**

In lecture words: this box is the mini-batch train proof for sequences.

### Local picture

```
  Dataset 1000 × (T=10, D=3) · y from sum rule
  DataLoader → (32,10,3)
  LSTMClassifier(3,32,2) + CE + Adam · 10 epochs
  same: zero_grad → backward → step
```

**Notice:** architecture choice (RNN vs LSTM vs GRU) swaps inside the Module; the loop stays.

### Bridge

**Persist** the weights with `torch.save`, **reload** into a fresh instance, then extract **reusable** train/eval functions and recap the whole bootcamp.

---

## Topic 10: Save load reusable recap (29:51–38:15)

### Where this sits on the master map

**PRODUCTION + CLOSE** — `state_dict` I/O; `train_one_epoch` / `evaluate`; full stack recap; next pretrained/MRI. Warm-up: [save](./PREREQUISITES.md#p7-save) · [loops](./PREREQUISITES.md#p8-loops).

### Board / screenshot

![Save load reusable recap](./screenshots/composites/ch10-topic-10-save-load-recap-panel1of1.png)

**Figure — ~30:50–38:15:** torch.save/load_state_dict; train_one_epoch & evaluate helpers; bootcamp recap; next pretrained / MRI.

### What he is establishing

**Model = weights.** Save them to reuse inference later or to **fine-tune** instead of random init. API: **`torch.save(model.state_dict(), "lstm_classifier.pth")`**. Load: construct **same architecture** → **`load_state_dict(torch.load(..., map_location=device))`** → **`model.eval()`**. Hidden dim or input dim mismatch **will not work**. Pattern is **not sequence-specific** — CNN/MLP too.

Next tutorial section: **pretrained** models (VGG, ResNet, …) and an **MRI tumor classification** case study.

Because train/eval code was rewritten for every classifier, factor it:

- **`train_one_epoch(model, loader, criterion, optimizer, device)`** — `model.train()`, batch loop, CE, zero_grad/backward/step, bookkeeping, return avg loss/acc. Call once per epoch.  
- **`evaluate(model, loader, criterion, device)`** — **no optimizer**; `model.eval()` + **`torch.no_grad()`**; forward, metrics, return averages.

**Bootcamp recap:** tensors, shapes, indexing/reshape, GPU move, autograd, `nn.Module`, Linear/Conv2d/RNN/LSTM/GRU, MSE & CE, SGD & Adam (and friends), custom/public Dataset, MLP, CNN, RNN family, **reusable train/eval**, **save/load**. Students ready for advanced topics and writing their own nets.

```python
# save
torch.save(model.state_dict(), "lstm_classifier.pth")

# load
loaded = LSTMClassifier(3, 32, 2).to(device)
loaded.load_state_dict(torch.load("lstm_classifier.pth", map_location=device))
loaded.eval()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return total_loss / max(len(loader), 1), correct / max(total, 1)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return total_loss / max(len(loader), 1), correct / max(total, 1)
```

You can now ship weights and reuse training utilities. Still missing (next unit): **loading pretrained vision nets** and adapting them (MRI).

A common trap is `torch.save(model, ...)` whole-object pickles that break across code versions — prefer **`state_dict`**.

### Analogy for this topic only

Training writes the **recipe card** (`state_dict`). Save puts it in a **recipe box** (`.pth`). Load opens the card into a **same-sized mixing bowl**. `train_one_epoch` / `evaluate` are **laminated checklist cards** you reuse whether dinner is CNN pasta or RNN stew.

Question: **What two arguments does `evaluate` deliberately omit vs train?**

In lecture words: this box closes the basics bootcamp and points at transfer learning.

### Local picture

```
  torch.save(state_dict, "model.pth")
  new = Arch(...); new.load_state_dict(torch.load(..., map_location=device))
  new.eval()

  train_one_epoch(...)  # needs optimizer
  evaluate(...)         # no_grad, no optimizer

  Recap: tensors→Module→MLP/CNN/RNN→save/load→helpers
  Next: pretrained VGG/ResNet · MRI tumor
```

**Notice:** architecture hyperparameters at load must match save.

### Bridge

You own sequence models in PyTorch. The leftover problem is **reusing giant pretrained vision weights** — next tutorials.

---

## Apply it (scenarios)

1. **Shape audit.** For `nn.LSTM(10, 32, num_layers=2, batch_first=True)` and `x=(16,20,10)`, predict `output`, `h_n`, `c_n` shapes.  
2. **Swap cell.** Replace LSTM with GRU in `LSTMClassifier` (rename); train 3 epochs on the toy set.  
3. **Wrong last-h.** Compare accuracy using `output[:,0]` vs `output[:,-1]` as summary.  
4. **Save round-trip.** Save, delete model, load, verify `evaluate` matches pre-save.  
5. **Mismatch.** Try loading with `hidden_dim=31`; read the error.  
6. **Helpers only.** Train 5 epochs using only `train_one_epoch` / `evaluate`.

### Minimal end-to-end skeleton

```python
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SumRuleDataset(Dataset):
    def __init__(self, n=1000, T=10, D=3):
        self.x = torch.randn(n, T, D)
        self.y = (self.x[:, :, :2].sum((1, 2)) > 0).long()
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

class SeqClassifier(nn.Module):
    def __init__(self, d_in=3, h=32, n_cls=2, cell="lstm"):
        super().__init__()
        Cell = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.cell_type = cell
        self.rnn = Cell(d_in, h, batch_first=True)
        self.fc = nn.Linear(h, n_cls)
    def forward(self, x):
        out = self.rnn(x)
        h = out[1][0] if self.cell_type == "lstm" else out[1]
        return self.fc(h[-1])

loader = DataLoader(SumRuleDataset(), batch_size=32, shuffle=True)
model = SeqClassifier(cell="lstm").to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()
for _ in range(5):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = crit(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
torch.save(model.state_dict(), "seq_clf.pth")
```

---

## External references

Two layers, **both kept**.

1. **Start here** — the newer high-signal companions (famous teachers, mapped to this lecture’s hard boxes).
2. **Full topic map** — the previous per-topic list (2–3 companions each) **plus** any new entries already woven above. Use a group when one box still feels thin.

### Start here — high-signal companions

Only a few **widely used** companions — the ones people actually finish. Not a pile of random blogs. Use them after the matching topic, with this tutorial still closed.

**If the tensor axes still swap (Topics 1–2).** Official [`nn.RNN`](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) / [`nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) docs name `(N, T, D)` when `batch_first=True` — the dummy `(4, 5, 3)` on the board.

**If the unroll still feels like magic (Topics 3–4).** Josh Starmer’s [StatQuest — Recurrent Neural Networks](https://www.youtube.com/watch?v=AsNTP8Kwu80) is the slow English version of shared weights across time. Andrej Karpathy’s [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) is the essay the field actually points beginners at.

**If LSTM gates are a blur of arrows (Topics 5–7).** Christopher Olah’s [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) is the canonical diagram post (GRU is in the same article). For a classroom voice on the two memories, use [StatQuest — Long Short-Term Memory](https://www.youtube.com/watch?v=YCzL96nL7j0).

**If the PyTorch return values still swap in your head (Topics 4, 6, 8–10).** Stay official: [`nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) for `(output, (h_n, c_n))`, the [char-RNN classification tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) for last-hidden → class, and [saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) for `state_dict`.

**How to use.** Gate fog → colah *after* Topic 5. Unroll fog → StatQuest *before* you write `LSTMClassifier`. Do not open ten tabs. One famous teacher per stuck idea.

---

### Full topic map — previous list plus new entries

**How to use:** finish the NOTES chain first (video closed if you can). When one map box still feels thin, open **only that topic’s group** below — **2–3 companions each** (prefer **teaching video + blog/notes + official docs**). All links live **here**, not inside topic bodies.

Prefer free teaching channels and official docs. Skip Wikipedia dumps and random SEO posts.

### Topic 1 — Recap + sequences intro

| Resource | Type | Why it helps |
|----------|------|--------------|
| [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) | Video | Layered transforms before adding time |
| [Tutorial 4 CNNs NOTES](../18-Tutorial04-CNNs-PyTorch/NOTES.md) | Prior unit | Exact previous playlist step |
| [Olah — Neural Networks, Types, and FP](https://colah.github.io/posts/2015-09-NN-Types-FP/) | Blog | Sequences as a first-class data type |

### Topic 2 — Sequence tensor (N, T, D)

| Resource | Type | Why it helps |
|----------|------|--------------|
| [PyTorch `nn.RNN` (`batch_first`)](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) | Docs | Axis convention used in class |
| [PyTorch tensors tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) | Official tutorial | Shape discipline |
| [Real Python — PyTorch tensors](https://realpython.com/pytorch-tensors/) | Blog | Readable N/T/D mental model |

### Topic 3 — RNN cell math + unroll

| Resource | Type | Why it helps |
|----------|------|--------------|
| [StatQuest — Recurrent Neural Networks](https://www.youtube.com/watch?v=AsNTP8Kwu80) | Video | Unroll + shared weights |
| [Karpathy — Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) | Blog | Why recurrence is powerful |
| [3Blue1Brown — Gradient descent](https://www.youtube.com/watch?v=IHZwWFHWa-w) | Video | Why repeated multiplies hurt long chains |

### Topic 4 — RNN output vs hidden shapes

| Resource | Type | Why it helps |
|----------|------|--------------|
| [PyTorch `nn.RNN` docs](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) | Docs | `output` vs `h_n` shapes |
| [Patrick Loeber — RNN from scratch (PyTorch)](https://www.youtube.com/watch?v=0_PgWWmauHk) | Video | Live shape prints |
| [Made With ML — RNNs](https://madewithml.com/courses/foundations/recurrent-neural-networks/) | Notes | Worked (N,T,H) examples |

### Topic 5 — LSTM gates + cell

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Colah — Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | Blog | **Canonical** gate diagrams matching the board |
| [StatQuest — Long Short-Term Memory](https://www.youtube.com/watch?v=YCzL96nL7j0) | Video | Plain-language memory |
| [Stanford CS224n LSTM notes (slide/video search)](https://web.stanford.edu/class/cs224n/) | Course | Vanishing gradients → LSTM |

### Topic 6 — `nn.LSTM` API

| Resource | Type | Why it helps |
|----------|------|--------------|
| [PyTorch `nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) | Docs | `(output, (h_n, c_n))` contract |
| [PyTorch sequence models / char-RNN tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) | Official tutorial | LSTM/RNN in classification code |
| [Sebastian Raschka — DL course RNN/LSTM notes](https://sebastianraschka.com/blog/2021/dl-course.html) | Notes | Module patterns for sequences |

### Topic 7 — GRU gates

| Resource | Type | Why it helps |
|----------|------|--------------|
| [Colah LSTM post (GRU section)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | Blog | GRU as simplified gated unit |
| [PyTorch `nn.GRU`](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) | Docs | API twin of RNN |
| [Illustrated Guide to LSTM & GRU (Michael Phi)](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21) | Blog | Side-by-side pictures |

### Topic 8 — Last-hidden classifier

| Resource | Type | Why it helps |
|----------|------|--------------|
| [PyTorch char-RNN classification tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) | Official tutorial | Last-hidden → class head |
| [StatQuest — SoftMax / ArgMax](https://www.youtube.com/watch?v=KpKog-L9veg) | Video | Logits → decision |
| [CrossEntropyLoss docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) | Docs | Logits in, integer labels |

### Topic 9 — Toy Dataset + train

| Resource | Type | Why it helps |
|----------|------|--------------|
| [PyTorch Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) | Official tutorial | Custom Dataset contract |
| [PyTorch Optimization loop](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) | Official tutorial | zero_grad → backward → step |
| [Patrick Loeber — RNN/LSTM name classification](https://www.youtube.com/watch?v=WEV61GmmPrk) | Video | Code-first sequence train loop |

### Topic 10 — Save/load + reusable loops + recap

| Resource | Type | Why it helps |
|----------|------|--------------|
| [PyTorch save/load models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) | Official tutorial | `state_dict` best practice |
| [PyTorch Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) | Official tutorial | Factor train/test functions |
| [ML Mastery — Save and load PyTorch models](https://machinelearningmastery.com/saving-and-loading-a-pytorch-model/) | Blog | Practical load pitfalls |

### Whole-map companions

| Resource | Type | Why it helps |
|----------|------|--------------|
| [PREREQUISITES.md (this package)](./PREREQUISITES.md) | Warm-up | #p1–#p8 sequence basics |
| [Tutorial 3 PyTorch Basics](../17-Tutorial03-PyTorch-Basics/NOTES.md) | Prior unit | Module / CE / DataLoader |
| [Tutorial 4 CNNs](../18-Tutorial04-CNNs-PyTorch/NOTES.md) | Prior unit | Previous classification stack |
| [Patrick Loeber — Deep Learning with PyTorch](https://www.youtube.com/watch?v=c36lUUr864M) | Video course | Full stack including RNN section |
| [Colah — Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | Blog home | Best companion while replaying gate boards |

---


## Sources

- Video: [Tutorial 5 : RNNs using PyTorch](https://www.youtube.com/watch?v=k6zF2NsvVrk)
- Channel: NPTEL — Indian Institute of Science, Bengaluru
- Course: Mathematical Foundations of Generative AI
- Duration: ~38 min (00:03–38:15)
- Skill: `youtube-lecture-tutor` · `code_tutorial`
- Captions cleaned; **10 topics** merged from denser moves
- Warm-up: [PREREQUISITES.md](./PREREQUISITES.md)
- Previous: [Tutorial 4 CNNs](../18-Tutorial04-CNNs-PyTorch/NOTES.md)
- Package: `19-Tutorial05-RNNs-PyTorch`
