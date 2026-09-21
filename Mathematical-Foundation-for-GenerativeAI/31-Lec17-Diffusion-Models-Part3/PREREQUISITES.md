> [!NOTE]
> **Warm-up & Orientation**: This package bridges unconditional DDPM to controlled conditional synthesis and guided diffusion. Ensure mastery of neural denoisers, score matching, and Bayes' rule on class conditionals.

# Prerequisites & Foundations: Diffusion Models Part 3 (Architecture, Sampling & Guidance)

## Math Terminology Rosetta Stone

| Symbol / Notation | Explicit Meaning | Standard Units / Domain | Common Alternative Notations | Mental Anchor / Reading Translation |
| :--- | :--- | :--- | :--- | :--- |
| $\nabla_{x_t} \log p(x_t)$ | Unconditional score function | $\mathbb{R}^D$ vector field | $s(x_t)$ | "The vector pointing toward higher data density in noise space" |
| $\nabla_{x_t} \log p(y \mid x_t)$ | Classifier gradient guidance vector | $\mathbb{R}^D$ vector field | $\nabla \log c(x_t)$ | "The vector steering the image toward class label $y$" |
| $s \in \mathbb{R}^+$ | Guidance scale hyperparameter | Dimensionless scalar $(1.5 \to 7.5)$ | $w, \gamma$ | "The volume knob for how aggressively the prompt is enforced" |
| $\epsilon_\theta(x_t, t, y)$ | Conditional noise prediction network | $\mathbb{R}^D$ continuous | $\hat{\epsilon}_c$ | "Noise predicted when given class or text condition $y$" |
| $\epsilon_\theta(x_t, t, \emptyset)$ | Unconditional noise prediction network | $\mathbb{R}^D$ continuous | $\hat{\epsilon}_u$ | "Noise predicted when prompt condition is dropped" |
| $\tilde{\epsilon}_\theta(x_t, t, y)$ | Extrapolated CFG noise vector | $\mathbb{R}^D$ continuous | $\epsilon_{\text{guided}}$ | "The boosted noise vector pushing toward the condition" |
| $\tau(t)$ | Sinusoidal time embedding vector | $\mathbb{R}^{d_{\text{emb}}}$ | $e_t, \text{pos}(t)$ | "The clock signal telling the U-Net what noise scale it sees" |

---

## Foundational Pillar 1: Isomorphic Neural Architectures and U-Nets

### Mathematical Derivation & Concept
In diffusion models, both the input $x_t$ and output $\epsilon_\theta(x_t, t)$ inhabit the exact same dimensional space:
$$f_\theta: \mathbb{R}^{B \times C \times H \times W} \times \mathbb{R} \to \mathbb{R}^{B \times C \times H \times W}$$
Unlike classifiers or standard autoencoders that compress spatial dimensions to a low-dimensional scalar or bottleneck vector, the diffusion backbone must be isomorphic (preserving dimensions).
The standard architecture is a U-Net:
1. **Downsampling Path**: Successive convolutional residual blocks reduce spatial resolution $(H, W) \to (H/2, W/2) \dots \to (H/16, W/16)$ while expanding channels.
2. **Bottleneck**: Captures global context and long-range semantic dependencies using self-attention.
3. **Upsampling Path**: Transposed convolutions or nearest-neighbor interpolations restore spatial resolution.
4. **Skip Connections**: Concatenate high-resolution feature maps from the downsampling path directly to the upsampling path, preserving pixel-level edge alignments.

```
   Input x_t (H x W) -----------------[ Skip Connection ]-----------------> Output eps (H x W)
         |                                                                      ^
    [ Conv 2x ]                                                            [ Conv 2x ]
         v                                                                      |
     (H/2 x W/2) ---------------------[ Skip Connection ]-----------------> (H/2 x W/2)
         |                                                                      ^
         +-------------------------> [ Bottleneck ] ----------------------------+
```

### Micro-Number Numerical Verification
Let an input image have shape $(1, 3, 32, 32)$ (3,072 dimensions).
The U-Net processes features through downsampling levels:
$(1, 64, 16, 16) \to (1, 128, 8, 8) \to (1, 256, 4, 4)$.
The upsampling path reconstructs $(1, 3, 32, 32)$ exactly, outputting 3,072 predicted noise components matching the input dimension.

### Physical Analogy
A precision optical lens system. Incoming light rays are focused into a tight focal point to extract deep semantic properties, then projected cleanly back onto a full-sized screen without losing high-frequency sharpness.

### Runnable Python Verification
```python
import torch
import torch.nn as nn

class MiniUNetBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
    def forward(self, x):
        return self.conv(x)

x = torch.randn(2, 3, 16, 16)
block = MiniUNetBlock(3, 3)
out = block(x)
assert out.shape == x.shape
print(f"Isomorphic shape verified: in={tuple(x.shape)} -> out={tuple(out.shape)}")
```

---

## Foundational Pillar 2: Sinusoidal Positional Time Embeddings

### Mathematical Derivation & Concept
Because a single neural network with shared weights $\theta$ denoises inputs across all $T=1000$ scales, it must know whether it is processing $t=999$ (pure noise) or $t=5$ (near-clean image).
Vaswani et al.'s sinusoidal position encoding is adapted to continuous diffusion time:
$$\tau(t)_{2i} = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad \tau(t)_{2i+1} = \cos\left(\frac{t}{10000^{2i/d}}\right)$$
for $i \in \{0, \dots, d/2 - 1\}$, where $d$ is the embedding dimension.
This embedding is passed through a multi-layer perceptron (MLP) and added or multiplied (via Adaptive Group Normalization / AdaGN) into every residual block:
$$h_{\text{next}} = \text{Conv}(h) + W_t \tau(t)$$

### Micro-Number Numerical Verification
Let $d = 4$, $t = 100$.
For $i=0$: $\omega_0 = 10000^0 = 1.0 \implies \sin(100) \approx -0.5064, \cos(100) \approx 0.8623$.
For $i=1$: $\omega_1 = 10000^{2/4} = 100.0 \implies \sin(100/100) = \sin(1) \approx 0.8415, \cos(1) \approx 0.5403$.
Embedding vector: $[-0.5064, 0.8623, 0.8415, 0.5403]$.

### Physical Analogy
A metronome ticking in the ear of an orchestra conductor. Every musician knows which measure of the symphony is currently being performed by listening to the rhythmic frequency.

### Runnable Python Verification
```python
import torch
import math

def get_time_embedding(timesteps, dim):
    half_dim = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim).float() / half_dim)
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return embedding

t = torch.tensor([50, 500])
emb = get_time_embedding(t, 16)
assert emb.shape == (2, 16)
print(f"Time embedding generated with shape: {emb.shape}")
```

---

## Foundational Pillar 3: Bayes' Rule on Class-Conditional Score

### Mathematical Derivation & Concept
To generate images belonging to a specific class $y$ (e.g., "cat"), we wish to sample from the conditional distribution $p(x_t \mid y)$.
Applying Bayes' rule in log-density form:
$$\log p(x_t \mid y) = \log p(x_t) + \log p(y \mid x_t) - \log p(y)$$
Taking the spatial gradient with respect to $x_t$ on both sides:
$$\nabla_{x_t} \log p(x_t \mid y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y \mid x_t)$$
Notice that $\nabla_{x_t} \log p(y) = 0$ because the prior probability of class $y$ is independent of image coordinates $x_t$.

The conditional score is decomposed into two distinct terms:
1. **Unconditional Score** $\nabla_{x_t} \log p(x_t)$: Guides samples toward realistic images generally.
2. **Classifier Gradient** $\nabla_{x_t} \log p(y \mid x_t)$: Guides samples toward the specific class $y$.

```
   Conditional Score:  nabla_{x_t} log p(x_t | y)
                              ||
   [ Unconditional Score ]     +     [ Classifier Steering Vector ]
   nabla_{x_t} log p(x_t)      +     nabla_{x_t} log p(y | x_t)
   (Makes image realistic)           (Forces image to be a "Cat")
```

### Micro-Number Numerical Verification
Suppose unconditional score vector points along $[+2.0, -1.0]$ and classifier gradient points along $[+1.0, +3.0]$.
Total conditional score vector:
$[2.0 + 1.0, -1.0 + 3.0] = [+3.0, +2.0]$.
The combined vector simultaneously increases realism and class certainty.

### Physical Analogy
Steering a sailing ship. The ocean tide and wind provide the unconditional current ($\nabla \log p(x_t)$) that keeps the ship buoyant in deep water; the rudder provides the directional thrust ($\nabla \log p(y \mid x_t)$) that aims the ship toward your chosen destination harbor.

### Runnable Python Verification
```python
import torch

def verify_conditional_score_decomposition():
    uncond_score = torch.tensor([2.0, -1.0])
    classifier_grad = torch.tensor([1.0, 3.0])
    cond_score = uncond_score + classifier_grad
    assert torch.allclose(cond_score, torch.tensor([3.0, 2.0]))
    print(f"Decomposition verified: {cond_score.tolist()}")

verify_conditional_score_decomposition()
```

---

## Foundational Pillar 4: Classifier-Free Guidance (CFG) Extrapolation

### Mathematical Derivation & Concept
Classifier guidance requires training an auxiliary image classifier $p(y \mid x_t)$ on noisy images $x_t$, which is computationally expensive and prone to adversarial gradient vulnerabilities.
Ho & Salimans (2021) introduced Classifier-Free Guidance (CFG). Using the score-noise relationship $\nabla_{x_t} \log p(x_t) \propto -\epsilon_\theta$, we rewrite the conditional score with guidance scale $s \ge 1$:
$$\tilde{\nabla}_{x_t} \log p(x_t \mid y) = \nabla_{x_t} \log p(x_t) + s \nabla_{x_t} \log p(y \mid x_t)$$
Using Bayes' rule, $\nabla_{x_t} \log p(y \mid x_t) = \nabla_{x_t} \log p(x_t \mid y) - \nabla_{x_t} \log p(x_t)$.
Substituting this into the guided equation:
$$\tilde{\nabla}_{x_t} \log p(x_t \mid y) = \nabla_{x_t} \log p(x_t) + s \left( \nabla_{x_t} \log p(x_t \mid y) - \nabla_{x_t} \log p(x_t) \right)$$
$$= (1 - s) \nabla_{x_t} \log p(x_t) + s \nabla_{x_t} \log p(x_t \mid y)$$

Translating to noise predictions:
$$\tilde{\epsilon}_\theta(x_t, t, y) = (1 - s) \epsilon_\theta(x_t, t, \emptyset) + s \epsilon_\theta(x_t, t, y)$$
Equivalently:
$$\tilde{\epsilon}_\theta(x_t, t, y) = \epsilon_\theta(x_t, t, \emptyset) + s \left( \epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset) \right)$$

When $s = 1$, we get standard conditional diffusion. When $s > 1$, we extrapolate along the conditioning direction, dramatically amplifying prompt adherence.

### Micro-Number Numerical Verification
Let unconditional prediction $\epsilon_\emptyset = 0.40$, conditional prediction $\epsilon_y = 0.60$, and scale $s = 3.0$.
$$\tilde{\epsilon} = 0.40 + 3.0 \times (0.60 - 0.40) = 0.40 + 3.0 \times 0.20 = 0.40 + 0.60 = 1.00$$
Notice how the condition's influence is boosted from $0.20$ to $0.60$.

### Physical Analogy
Contrast enhancement on a television. Increasing the contrast slider ($s > 1$) does not invent new colors; it stretches the distance between bright and dark pixels, making edges crisp and distinct.

### Runnable Python Verification
```python
import torch

def verify_cfg_extrapolation():
    eps_uncond = torch.tensor([0.4])
    eps_cond = torch.tensor([0.6])
    s = 3.0
    eps_guided = eps_uncond + s * (eps_cond - eps_uncond)
    assert abs(eps_guided.item() - 1.0) < 1e-5
    print(f"CFG Extrapolation verified: guided noise={eps_guided.item():.4f}")

verify_cfg_extrapolation()
```

---

## Curriculum Bridges

| Concept | Upstream Prerequisite | Downstream Application in Lecture 17 |
| :--- | :--- | :--- |
| Convolutional Neural Networks | [04-Tensors_and_Shapes.md](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/04-Tensors_and_Shapes.md) | Constructing isomorphic U-Net down/up blocks |
| Positional Embeddings | [09-Positional_Encodings.md](../../MathsTerms/01-Linear-Algebra-Geometry-and-Tensors/09-Positional_Encodings.md) | Modulating shared model layers across continuous time $t$ |
| Conditional Probability & Bayes | [03-Joint_Marginal_Conditional_Dist.md](../../MathsTerms/03-Probability-and-Statistical-Estimation/03-Joint_Marginal_Conditional_Dist.md) | Decomposing conditional score into unconditional plus classifier guidance |

---

## Diagnostics Self-Assessment

1. **Why does Classifier-Free Guidance require training with random label dropping ($y = \emptyset$)?**  
   *Answer*: To enable a single neural network to predict both unconditional noise $\epsilon_\theta(x_t, t, \emptyset)$ and conditional noise $\epsilon_\theta(x_t, t, y)$ without maintaining two separate models.

2. **What trade-off occurs when the guidance scale $s$ is set too high (e.g. $s = 20$)?**  
   *Answer*: High guidance reduces sample diversity, causes color over-saturation, and creates harsh high-contrast artifacts due to clipping and extreme extrapolation beyond the training distribution.
