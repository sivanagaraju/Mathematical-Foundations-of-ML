# Convolution and Pooling: Shared Local Computation

[Module guide](README.md) · [Study routes](START_HERE.md) · Next: [Recurrent networks](02-Recurrent_Neural_Networks.md)

## 1. What this idea helps you do

A camera image contains the same kinds of local patterns at many positions. A convolutional layer applies one learned local rule repeatedly, producing a spatial map of responses. Pooling summarizes small regions of that map. The engineering decision is what information to share, what to retain, and what to discard—not whether an image can be stored as a flat array.

**Prerequisites**

- **Required now:** multiplication, finite sums, indexing, and the derivative of a weighted sum. [Vectors and matrices, §9](../02-Linear-Algebra-Geometry-and-Tensors/01-Vectors_and_Matrices.md) works through a linear layer and its gradients. [Tensors and shapes, §§2–3](../02-Linear-Algebra-Geometry-and-Tensors/04-Tensors_and_Shapes.md) introduces logical image axes. This chapter restates the conventions it uses.
- **Required for optional depth:** the multibranch chain rule and transpose of a linear map; see [Chain rule and backpropagation, §4, Proofs 2–3](../03-Multivariate-Calculus-and-Optimization/04-Chain_Rule_and_Backpropagation.md). These support the input-gradient and adjoint proofs in §8.
- **Useful context:** [autoencoders](03-Autoencoders_and_Latent_Spaces.md) for image compression and [GANs](09-Minimax_Game_and_GANs.md) for the training objective of the generator in §10. Neither is needed for the numerical convolution.

**Target systems:** convolutional image encoders, image generators, and local components inside larger vision architectures. **Study time:** about 60–90 minutes for the core lesson, another 45–60 minutes for proofs, code inspection, and exercises.

After studying, you should be able to:

1. Calculate convolution and pooling outputs, including stride, padding, and dilation.
2. Explain precisely when a shifted input produces a shifted output.
3. Derive shared-kernel and input gradients and verify them independently.
4. Distinguish a transposed convolution from an inverse and trace it through an image generator.
5. Choose a local operator using its information loss, parameter count, and boundary behavior.

**Fast route:** §§2–4 → §7 → §9 → §11 → §12, then §10 for the system connection. Read the assumptions beside each formula. **Deep route:** §§2–14 in order; §8 contains the fuller proofs. The references are follow-up learning, not a substitute for the internal derivation.

You now know the task and required tools. The open question is what a shared local rule actually computes; a nine-number image lets us answer that without architectural jargon.

## 2. Start with a problem you can picture

Suppose a sensor produces this small intensity grid. We want one local score for every complete two-by-two patch. Use the same four coefficients at every position:

$$
X=\begin{bmatrix}1&2&0\\0&3&1\\2&0&1\end{bmatrix},
\qquad K=\begin{bmatrix}1&0\\-1&2\end{bmatrix}.
$$

**Predict before calculating:** How many complete patches fit if we move one cell at a time? Is the score of the upper-left patch its maximum value, or a weighted sum? Will moving a pattern leave the entire output unchanged?

```text
Input X                       Shared kernel K       Output positions
 1  2  0                        1  0                 Y[0,0]  Y[0,1]
 0  3  1                       -1  2                 Y[1,0]  Y[1,1]
 2  0  1

First patch:  [1  2]   multiply matching cells, then add -> 1+0+0+6 = 7
              [0  3]
```

The kernel fits at four starting positions, so the output is two by two. The score 7 is a weighted sum, not a maximum; reusing the rule gives comparable scores at different positions.

The other scores will be 1, −2, and 5. A separate max-pooling operation applied to this whole two-by-two output returns 7; average pooling returns 2.75. These summaries answer different questions, and neither preserves the original map.

We have a concrete operation and a prediction to revisit. To extend it beyond this one grid, we need names for the arrays, positions, and movement rules.

## 3. Name the objects and read the notation

An **input channel** is one array of values at spatial positions; RGB images have three. A **kernel** is an array of coefficients. An **output channel**, also called a feature map, records the scores from one kernel spanning all input channels. A **batch** holds several independent inputs.

We use zero-based indices. `NCHW` means batch, channel, height, width; it describes logical axes, not necessarily the order of physical memory. Our first example has one batch item and one channel, so we write only its two spatial axes.

For one channel, no padding, unit stride and unit dilation, define cross-correlation by

$$Y_{ij}\coloneqq\sum_{a=0}^{k_h-1}\sum_{b=0}^{k_w-1}K_{ab}X_{i+a,j+b}.$$

Read this aloud: “Y at row i, column j is defined as the sum over kernel rows a and kernel columns b of K at a, b times X at i plus a, j plus b.” The symbol $\coloneqq$ marks a **definition**, not a theorem. Deep-learning libraries conventionally call this operation convolution; mathematical convolution flips the kernel in both spatial directions.

| Symbol | Spoken as | Role / dimensions | Toy value |
| :--- | :--- | :--- | :--- |
| $X$ | “ex” | input, $H\times W$ here | $3\times3$ array in §2 |
| $K$ | “kay” | learned kernel, $k_h\times k_w$ | $2\times2$ array in §2 |
| $Y$ | “why” | output, $H_o\times W_o$ | $\begin{bmatrix}7&1\\-2&5\end{bmatrix}$ |
| $N,C_{\rm in},C_{\rm out}$ | “en, see in, see out” | batch and channel counts | $1,1,1$ |
| $s,p,d$ | “ess, pee, dee” | stride, padding per side, kernel-point spacing | $1,0,1$ |
| $i,j,a,b$ | “eye, jay, ay, bee” | output and kernel indices | all start at 0 |
| $G$ | “jee” | upstream gradient, same shape as $Y$ | $\begin{bmatrix}1&-0.5\\0.2&-1\end{bmatrix}$ in §9 |
| $\mathcal L$ | “ell” | scalar loss | $\sum_{ij}G_{ij}Y_{ij}$ in §9 |
| $\nabla_K\mathcal L$ | “NAB-luh kay ell” | array of loss derivatives | shape $2\times2$ |
| $\eta$ | “AY-tuh” | learning rate | $0.1$ in §9 |
| $\tau_v$ | “tau sub vee” | translation by integer vector $v$ | a one-cell shift |
| $A_K^\top$ | “ay kay transpose” | adjoint of the fixed-kernel linear map | sends output gradients to input gradients |

The upstream gradient says how a later scalar objective reacts to each output. It is not automatically the gradient of max pooling; §9 deliberately defines the objective that produces this particular $G$.

We can now read the operation without guessing the symbols. The remaining issue is why its output shape and response to a shift follow from the definition; counting valid positions and substituting indices will answer both.

## 4. Build the central relationship

### From one patch to the whole map

At the first position, substitute the four kernel/input pairs into the defining sum:

$$Y_{00}=1(1)+0(2)+(-1)(0)+2(3)=7.$$

Moving right changes only which input entries are read:

$$Y_{01}=1(2)+0(0)+(-1)(3)+2(1)=1.$$

The coefficients are shared. That is a restriction on the function class: we do not learn an unrelated four-number rule for each location.

```text
X[0:2,0:2] -- same K --> 7      X[0:2,1:3] -- same K --> 1
X[1:3,0:2] -- same K --> -2     X[1:3,1:3] -- same K --> 5
                              |
                              v
                         Y = [ 7   1 ]
                             [-2   5 ]
```

Each arrow uses a different patch but the same four parameters. The two-by-two arrangement preserves where the four responses occurred; pooling will deliberately discard part of that location information.

### Count output positions, including dilation

Consider one spatial axis of length $H$. Assume integers $H,k,s,d\geq1$ and $p\geq0$, symmetric zero padding, and floor-mode output sizing. A kernel with $k$ sample points spaced $d$ cells apart spans

$$k_{\rm eff}=1+d(k-1).$$

There are $k-1$ gaps, each of length $d$, plus the initial point. Padding makes the axis length $H+2p$. A patch starting at padded index $is$ is valid precisely when

$$is+(k_{\rm eff}-1)\leq H+2p-1.$$

Subtracting $k_{\rm eff}-1$ gives $is\leq H+2p-k_{\rm eff}$. Dividing by positive $s$ and taking the largest integer start index gives $i_{\max}=\lfloor(H+2p-k_{\rm eff})/s\rfloor$. Counting indices from 0 adds one:

$$H_o=\left\lfloor\frac{H+2p-d(k-1)-1}{s}\right\rfloor+1.$$

Apply the same reasoning to width, with its own parameters if needed. Require $H+2p\geq k_{\rm eff}$; otherwise there is no valid patch and our implementation rejects the configuration. The toy has $(3-2)/1+1=2$ outputs on each axis. With $H=64,k=3,p=1,s=d=1$, there are 64; with $k=4,p=1,s=2,d=1$, there are 32. Halving is a property of those choices, not the definition of convolution.

### Prove the shift relationship, with its domain

Let $X$ be defined on the full integer grid $\mathbb Z^2$, let $K$ have finite support, and use stride 1. Define $(\tau_vX)_{ij}=X_{i-v_1,j-v_2}$. Finite support makes every following sum finite. Substitution gives

$$
\begin{aligned}
\operatorname{Conv}_K(\tau_vX)_{ij}
&=\sum_{a,b}K_{ab}(\tau_vX)_{i+a,j+b}
&&\text{(cross-correlation definition)}\\
&=\sum_{a,b}K_{ab}X_{i+a-v_1,j+b-v_2}
&&\text{(translation definition)}\\
&=\operatorname{Conv}_K(X)_{i-v_1,j-v_2}
&&\text{(regroup integer additions)}\\
&=(\tau_v\operatorname{Conv}_K(X))_{ij}
&&\text{(translation definition).}
\end{aligned}
$$

This is **equivariance**: shifting the input shifts the output. **Invariance** would mean that the output does not change at all. They are different equations.

For a finite image, the proof applies to locations whose needed patches remain inside the image, or to a compatible periodic shift with circular boundaries. Cropping after a shift can discard input values, so fixed-size zero-padded images need not satisfy the equation at their edges. With stride $s$, an input shift of $sr$ corresponds to an output shift of $r$ on the unrestricted grid; a one-cell shift generally changes which sampling phase is observed.

We now know the local computation, its dimensions, and its qualified symmetry. But a rule can be well defined without being the right architectural choice; comparing what each candidate can represent makes the tradeoff explicit.

## 5. Why choose this tool for this problem?

| Property | Shared convolution | Dense map on flattened input | Self-attention | Max pooling |
| :--- | :--- | :--- | :--- | :--- |
| Main job | Learned local response at each position | Arbitrary linear mixing between specified input/output coordinates | Content-dependent interaction among tokens | Fixed local summary |
| Learned weights, excluding biases | $C_{out}C_{in}k_hk_w$ with one group | $D_{out}D_{in}$ | Typically $O(D^2)$ projection weights at fixed model width | 0 |
| Spatial information | Retains an output grid; stride may discard detail | Flattening preserves data if shape/order are known; no built-in sharing | Depends on tokens, position information, and masking | Loses nonmaximum values and, without saved indices, winner locations |
| Interaction range | Kernel support; grows through layers | Entire supplied input | Potentially all supplied tokens | Pool window |
| Good fit | Similar local rule needed at many positions | Coordinate-specific global map | Distant, input-dependent relationships | Presence of a strong response matters more than exact position |

Attention's $O(T^2D)$ conventional pairwise computation for $T$ tokens is **compute**, not its parameter count. A dense layer is not permutation invariant: $[1,2]\cdot[1,0]=1$, while permuting the input to $[0,1]$ gives 2. Flattening itself can be undone; the missing assumption is local weight sharing.

**Matched-output parameter counterexample.** Take one $32\times32$ single-channel image and request a $32\times32$ single-channel output. An arbitrary dense map has $1024^2=1,048,576$ weights. A $3\times3$ convolution with padding 1 has 9 weights and the same output shape. At four bytes per weight, that is 4 MiB versus 36 bytes for weights alone. The dense map can express more coordinate-specific functions; it is unsuitable if a budget requires fewer than 1,000 weights, but it is not mathematically incapable of learning a shared local rule.

For the larger, different-output example, a flattened $256\times256\times3$ input mapped to 1,024 features uses $196,608\times1,024=201,326,592$ weights. A 64-channel $3\times3$ convolution uses $64\times3\times9=1,728$, but emits a spatial map, not a 1,024-vector. Those counts explain scaling, not a like-for-like replacement.

**Pooling counterexample.** One-dimensional max pooling of width/stride 2 sends $[0,1,0,0]$ to $[1,0]$. Shifting right by one cell produces $[0,0,1,0]$ and pooled output $[0,1]$. Pooling is invariant to rearrangements *within a fixed window*, but not to arbitrary translations across windows. If precise location is needed to reconstruct an image, max pooling alone is insufficient.

We can now choose an operator based on a concrete requirement. The remaining intuition to build is how one shared parameter receives feedback from many places; a reusable signal-processing stencil makes that accumulation tangible.

## 6. Strengthen the intuition and mark its limits

Imagine a programmable four-tap sensor stencil. It reads a local patch, multiplies each reading by its calibrated gain, and adds the results. Move the stencil to another location without recalibrating it. During training, all locations send suggestions about how the same gains should change, so those suggestions must be added.

| Engineering element | Mathematical symbol | Exact intuition mapped |
| :--- | :--- | :--- |
| Sensor readings in a local window | $X_{i+a,j+b}$ | Values read at one stencil position |
| Shared programmable gain | $K_{ab}$ | The same coefficient used at every position |
| Sliding schedule | $s$ | Distance between successive starting positions |
| Summary circuit | max or average | Chooses a strongest response or computes its mean |
| Sum of calibration feedback | $\sum_{ij}G_{ij}X_{i+a,j+b}$ | One parameter receives contributions from all uses |

### Where this analogy stops working

A kernel can have negative coefficients; it is not a physical cutout that merely recognizes an outline. It also mixes channels, which may be learned features rather than visible colors. A $1\times1$ kernel mixes channels at a single position without reading neighboring positions.

The stencil does not imply that useful features will be learned, that pooling increases semantic understanding, or that all influential input positions matter equally. The **theoretical receptive field** lists possible dependencies; the observed influence of those positions depends on learned weights, activations, and data. No Gaussian influence profile follows from the convolution definition.

The mechanical picture explains sharing and feedback. To keep that picture from becoming a source of errors, we next separate terms that look similar in architecture diagrams but obey different equations.

## 7. Terms worth keeping straight

### Core Terminology Reference Table

| Term | Pronunciation | Plain-English Meaning | Formal Definition & Conditions |
| :--- | :--- | :--- | :--- |
| **Convolution** | “kon-vuh-LOO-shun” | Flipping a pattern and sliding it across a signal to compute weighted overlap | $(f * g)(x) = \int f(y) g(x-y) dy$. In discrete arrays, reverses spatial indices before summation. |
| **Cross-Correlation** | “kross-kor-uh-LAY-shun” | Sliding a pattern without flipping to compute local matching scores | $Y_{ij} = \sum_{a,b} K_{ab} X_{i+a,j+b}$. The actual operation implemented by PyTorch and deep learning libraries. |
| **Equivariance** | “ee-kwih-VAIR-ee-uns” | Transforming the input transforms the output by the exact same transformation | $f(\tau_v X) = \tau_v f(X)$. Shifting an image shifts its feature map accordingly. |
| **Invariance** | “in-VAIR-ee-uns” | Transforming the input leaves the output completely unchanged | $f(\tau_v X) = f(X)$. Global pooling produces shift-invariant classification logits. |
| **Stride ($s$)** | “stryde” | Step size in pixels between consecutive sliding window evaluations | Subsamples output resolution by factor $s$; $H_o = \lfloor (H + 2p - k)/s \rfloor + 1$. |
| **Dilation ($d$)** | “dy-LAY-shun” | Spacing between kernel taps, expanding receptive field without added weights | Effective kernel span $k_{\text{eff}} = d(k-1) + 1$. Regular convolution corresponds to $d=1$. |
| **Transposed Convolution** | “trans-POHZD kon-vuh-LOO-shun” | Linear adjoint operator scattering outputs back to inputs (gradient of convolution) | $y = A_K^\top g$. Increases spatial resolution; generally NOT the mathematical inverse $A_K^{-1}$. |
| **Receptive Field** | “ree-SEP-tiv feeld” | The spatial span of input pixels that can influence a given output activation | Expands linearly with layer depth: $R_\ell = R_{\ell-1} + (k_\ell - 1) J_{\ell-1}$. |

### Confused Pairs Distinction Breakdown

1. **Convolution vs. Cross-Correlation**:
   - *Core Definition:* Convolution flips the kernel spatially ($K_{-a, -b}$); cross-correlation preserves kernel orientation ($K_{a, b}$).
   - *Common Confusion:* Deep learning frameworks (PyTorch `nn.Conv2d`, TensorFlow `tf.nn.conv2d`) use the word "convolution" in their API names but implement cross-correlation internally.
   - *Rule of Thumb:* Because weights are learned from scratch by gradient descent, whether the kernel is flipped or unflipped is mathematically absorbed into the learned parameters. Only flip if comparing against formal signal processing theory.

2. **Equivariance vs. Invariance**:
   - *Core Definition:* Equivariance means moving the input moves the output score map identically. Invariance means moving the input leaves the scalar output identical.
   - *Common Confusion:* Saying "CNNs are invariant to translations."
   - *Rule of Thumb:* Convolutional feature maps are **equivariant**; global pooling layers and classification heads are **invariant**.

3. **Stride vs. Dilation**:
   - *Core Definition:* Stride skips input sample positions (moves the window by $s > 1$). Dilation skips kernel sampling points (inserts $d-1$ gaps inside the kernel).
   - *Common Confusion:* Conflating how resolution is reduced with how receptive field is expanded.
   - *Rule of Thumb:* Stride shrinks feature map dimensions; dilation expands receptive field while keeping feature map dimensions unchanged (if stride is 1).

4. **Transposed Convolution vs. Matrix Inverse**:
   - *Core Definition:* Transposed convolution multiplies by the adjoint matrix $A^\top$; inverse multiplies by $A^{-1}$.
   - *Common Confusion:* Assuming transposed convolution reconstructs the exact original high-resolution image without information loss.
   - *Rule of Thumb:* An adjoint reverses signal flow for backpropagation; it cannot recover information discarded by striding or pooling. $A^\top A \ne I$.

The words now identify distinct operations. What remains is to derive how changing a parameter or input changes a loss, including the exceptional pooling cases; finite-sum differentiation provides that bridge to training.

## 8. Work through the mathematics and its conditions

### The multichannel operation

For real finite arrays, one group, and zero extension of $X$ outside its spatial domain, define

$$
Y_{n,o,i,j}=\beta_o+
\sum_{c=0}^{C_{in}-1}\sum_{a=0}^{k_h-1}\sum_{b=0}^{k_w-1}
K_{o,c,a,b}X_{n,c,is_h-p_h+ad_h,js_w-p_w+bd_w}.
$$

Here $\beta_o$ (“BAY-tuh sub oh”) is a learned scalar bias per output channel. The indices $n,o,c$ select the batch item, output channel, and input channel. All other symbols retain §3's meanings. Read the formula as “one output equals its channel bias plus all channel-and-kernel weighted input readings at this starting position.” This is affine in the input when the bias is present, and linear when the bias is zero.

The weight shape is $(C_{out},C_{in},k_h,k_w)$. With $g$ groups dividing both channel counts, each output connects only to its assigned $C_{in}/g$ channels, so the weight count becomes $C_{out}(C_{in}/g)k_hk_w$. A depthwise layer with multiplier 1 has $C_{in}k_hk_w$ weights; following it with a pointwise layer adds $C_{in}C_{out}$. The reduction comes with the restriction that each input channel first uses one spatial filter before channel mixing.

### Derive both gradients from the scalar chain rule

Return to one channel, stride/dilation 1, no padding, and zero bias. Let $\mathcal L$ be a differentiable scalar function of all $Y_{ij}$ and define $G_{ij}=\partial\mathcal L/\partial Y_{ij}$. First differentiate one output with respect to a specific coefficient:

$$
\frac{\partial Y_{ij}}{\partial K_{ab}}
=\frac{\partial}{\partial K_{ab}}
\sum_{u,v}K_{uv}X_{i+u,j+v}
=X_{i+a,j+b}.
$$

The finite-sum derivative distributes across terms. All terms with $(u,v)\ne(a,b)$ have zero derivative; the remaining term uses $\partial(K_{ab}x)/\partial K_{ab}=x$. Applying the multivariable chain rule and adding all output paths gives

$$\boxed{\frac{\partial\mathcal L}{\partial K_{ab}}
=\sum_{i,j}G_{ij}X_{i+a,j+b}.}$$

One input pixel can also be used at multiple positions. Its local derivative is

$$\frac{\partial Y_{ij}}{\partial X_{uv}}=
\sum_{a,b}K_{ab}\,\mathbf1\{u=i+a,\ v=j+b\},$$

where $\mathbf1\{\cdot\}$ (“indicator”) is 1 if both index equalities hold and 0 otherwise. Substituting this into the chain rule gives

$$\boxed{\frac{\partial\mathcal L}{\partial X_{uv}}=
\sum_{i,j,a,b}G_{ij}K_{ab}\mathbf1\{u=i+a,\ v=j+b\}.}$$

Thus the backward computation scatters each $G_{ij}K$ into the corresponding input patch and adds overlapping contributions. In the full multichannel formula, also sum over batch/output positions for a shared weight, over output channels for an input gradient, and over batch/spatial positions for a bias gradient. In particular, $\partial\mathcal L/\partial\beta_o=\sum_{n,i,j}G_{n,o,i,j}$ because each local bias derivative is 1.

### Why the input gradient is a transposed convolution

Fix $K$, set bias to zero, flatten arrays in a declared row-major order, and write $y=A_Kx$. $A_K$ is sparse: each row contains the kernel coefficients at the input coordinates used by one output. It is a mathematical representation; a library need not materialize this matrix.

For an arbitrary input perturbation $h$ and output gradient vector $g$,

$$
\begin{aligned}
\langle A_Kh,g\rangle
&=\sum_r\left(\sum_t(A_K)_{rt}h_t\right)g_r
&&\text{(matrix multiplication and inner product)}\\
&=\sum_t h_t\left(\sum_r(A_K)_{rt}g_r\right)
&&\text{(reorder finite sums)}\\
&=\langle h,A_K^\top g\rangle
&&\text{(transpose definition).}
\end{aligned}
$$

Since $d\mathcal L=g^Tdy=g^TA_Kdx=(A_K^Tg)^Tdx$, the input gradient is $A_K^Tg$. This pairing property defines the **adjoint** for real arrays with the ordinary inner product.

It does not imply $A_K^TA_K=I$. For example, $A=[1\ 1]$ maps both $(1,0)$ and $(0,1)$ to 1. Applying its transpose sends 1 to $(1,1)$, recovering neither original. A transpose remains well defined even when an inverse cannot exist.

For one spatial axis, the transpose associated with a compatible convolution has size

$$H_{out}=(H_{in}-1)s-2p+d(k-1)+1+o,$$

where $o$ is `output_padding`. To derive the ambiguity, write the forward division as $H_{original}+2p-k_{eff}=s(H_{in}-1)+r$ with remainder $0\le r<s$. Rearranging gives $H_{original}=s(H_{in}-1)-2p+k_{eff}+r$. Choosing $o=r$ recovers the intended **shape**, not the lost input values. Here we use $0\le o<s$; framework APIs have additional admissibility details for dilated configurations. `output_padding` selects an output shape; it is not an instruction to append a border of zero-valued outputs.

### Pooling derivatives, directional limits, and subdifferentials

For a window with $m$ entries $u_1,\ldots,u_m$, define $q_{avg}=m^{-1}\sum_r u_r$. Differentiating the sum gives $\partial q_{avg}/\partial u_r=1/m$. With upstream derivative $h$, the multivariable chain rule multiplies this by $h$.

For $q_{max}=\max_r u_r$, suppose one entry $u_t$ strictly exceeds all others: $u_t > u_r$ for all $r \ne t$. Small enough perturbations preserve that strict inequality, so locally $q_{max}=u_t$. Its gradient has a 1 at index $t$ and zeros elsewhere.

At a tie, the ordinary Fréchet derivative does not exist. We prove this via one-sided directional limits. Let $u \in \mathbb{R}^m$ and let $\mathcal{I}(u) = \{i \in \{1,\ldots,m\} : u_i = \max_j u_j\}$ be the set of active maximum indices. The directional derivative of $\max$ along direction vector $v \in \mathbb{R}^m$ is defined as the limit:

$$
\lim_{t \to 0^+} \frac{\max(u + tv) - \max(u)}{t}.
$$

When $t > 0$ is sufficiently small, the maximum of $u + tv$ is attained exclusively on a subset of $\mathcal{I}(u)$. Specifically, for any $j \notin \mathcal{I}(u)$, the gap $\Delta = \max(u) - u_j > 0$ ensures $u_j + t v_j < \max(u) + t \min_{i \in \mathcal{I}(u)} v_i$ for all $t < \Delta / (\|v\|_\infty + \max_i |v_i|)$. Therefore:

$$
\lim_{t \to 0^+} \frac{\max(u + tv) - \max(u)}{t} = \max_{i \in \mathcal{I}(u)} v_i.
$$

When $|\mathcal{I}(u)| > 1$, the mapping $v \mapsto \max_{i \in \mathcal{I}(u)} v_i$ is sublinear and convex, but **not linear** in $v$ (for example, $\max(v_1, v_2) + \max(-v_1, -v_2) \ge 0$, with strict inequality whenever $v_1 \ne v_2$). Because linearity of the directional derivative is a necessary condition for Fréchet differentiability, the ordinary gradient does not exist at ties.

However, the Clarke generalized subdifferential $\partial \max(u)$ is well defined, non-empty, convex, and compact:

$$
\partial \max(u) = \operatorname{conv} \{ e_i \in \mathbb{R}^m : i \in \mathcal{I}(u) \},
$$

where $\operatorname{conv}\{\cdot\}$ denotes the convex hull and $e_i$ is the $i$-th standard basis vector. Any subgradient $g \in \partial \max(u)$ satisfies $g_i \ge 0$, $\sum_{i \in \mathcal{I}(u)} g_i = 1$, and $g_j = 0$ for $j \notin \mathcal{I}(u)$. Our pure Python reference routes to the first index in $\mathcal{I}(u)$ ($g = e_{\min \mathcal{I}(u)}$); PyTorch's autograd engine also selects one valid extreme point of the subdifferential without guaranteeing a particular tied index.

Overlapping pool windows require **adding** gradients where an input pixel participates in multiple pooling windows. Average pooling with padding also requires specifying whether padded zero entries count in the denominator; our examples use no padding. Max pooling's negative-infinity padding convention ensures boundary pads never win the maximum.

### Continuity: small changes in values, not arbitrary spatial shifts

For a fixed finite kernel, let $M=\sum_{a,b}|K_{ab}|$ and let $\|X\|_\infty = \max_{u,v} |X_{uv}|$. At any valid output location $(i,j)$, the triangle inequality gives:

$$
|\operatorname{Conv}_K(X)_{ij}-\operatorname{Conv}_K(Z)_{ij}|
=\left|\sum_{a,b}K_{ab}(X_{i+a,j+b}-Z_{i+a,j+b})\right|
\leq\sum_{a,b}|K_{ab}|\,|X_{i+a,j+b}-Z_{i+a,j+b}|
\leq M\|X-Z\|_\infty.
$$

Taking the maximum over all output coordinates $(i,j)$ yields $\|\operatorname{Conv}_K(X) - \operatorname{Conv}_K(Z)\|_\infty \leq M\|X-Z\|_\infty$. If $M>0$, given any $\epsilon>0$, choosing $\delta=\epsilon/M$ guarantees that $\|X-Z\|_\infty<\delta$ implies $\|\operatorname{Conv}_K(X) - \operatorname{Conv}_K(Z)\|_\infty < \epsilon$. If $M=0$, the output is identically zero. This proves uniform Lipschitz continuity on $\mathbb{R}^{H \times W}$. Consequently, if an input sequence $X^{(n)} \to X$ in the $\|\cdot\|_\infty$ norm, the output sequence converges uniformly: $\lim_{n \to \infty} \operatorname{Conv}_K(X^{(n)}) = \operatorname{Conv}_K(X)$.

Max pooling is also Lipschitz continuous with constant $L=1$, including across ties. Let $r=\|u-v\|_\infty$. For every index $i$, $u_i \le v_i + r \le \max_j v_j + r$. Taking the maximum over $i$ on the left side gives $\max_i u_i \le \max_j v_j + r$, or $\max_i u_i - \max_j v_j \le \|u-v\|_\infty$. Swapping $u$ and $v$ gives the reverse inequality, yielding:

$$
|\max_i u_i - \max_i v_i| \le \|u-v\|_\infty.
$$

Continuous dependence on input values does not mean differentiability at ties, and a spatial translation $\tau_v X$ of one pixel can produce a large $\|\tau_v X - X\|_\infty$ difference at sharp edges.

### Step-by-step proof of translation equivariance

In continuous real analysis, let $f, g \in L^1(\mathbb{R}^2)$ be integrable functions on the plane. The continuous convolution is defined as $(f * g)(x) = \int_{\mathbb{R}^2} f(y) g(x-y) dy$. For any translation vector $v \in \mathbb{R}^2$, define the shift operator $(\tau_v f)(x) = f(x-v)$. We prove that continuous convolution commutes with translation:

$$
\begin{aligned}
(\tau_v (f * g))(x)
&= (f * g)(x - v)
&& \text{(definition of translation operator)} \\
&= \int_{\mathbb{R}^2} f(y) g((x - v) - y) \, dy
&& \text{(definition of continuous convolution)} \\
&= \int_{\mathbb{R}^2} f(y) g((x - y) - v) \, dy
&& \text{(associativity and commutativity of subtraction in } \mathbb{R}^2\text{)} \\
&= \int_{\mathbb{R}^2} f(y) (\tau_v g)(x - y) \, dy
&& \text{(definition of translation applied to kernel } g\text{)} \\
&= (f * \tau_v g)(x).
\end{aligned}
$$

Furthermore, substituting the dummy integration variable $u = y - v$ (with Jacobian determinant $|\det(I)| = 1$):

$$
\begin{aligned}
\int_{\mathbb{R}^2} f(y) g(x - v - y) \, dy
&= \int_{\mathbb{R}^2} f(u + v) g(x - v - (u + v)) \, du \\
&= \int_{\mathbb{R}^2} (\tau_{-v} f)(u) g(x - u) \, du \\
&= ((\tau_v f) * g)(x) \quad \text{(using } \tau_v f(y) = f(y-v)\text{)}.
\end{aligned}
$$

Thus, $\tau_v (f * g) = (\tau_v f) * g$. In deep learning, discrete cross-correlation on finite arrays satisfies this identity **if and only if**:
1. Stride $s = 1$ (strides $s > 1$ preserve shift equivariance only for shifts that are exact integer multiples of $s$).
2. Boundary effects are eliminated via periodic boundary wrapping (circular padding) or infinite support. Under zero-padding, boundary entries lose context, causing equivariance to fail at the edges.

### Computational and hardware reality: im2col, Winograd, and memory bandwidth

Modern deep-learning frameworks almost never execute 2D convolution via nested Python or C `for` loops. The computation is mapped directly onto GPU hardware using two primary algorithms:

1. **im2col + GEMM (Image-to-Column Matrix Multiplication):**
   - **Mechanism:** For an input tensor of shape $(C_{\text{in}}, H, W)$ and kernel $(C_{\text{out}}, C_{\text{in}}, k_h, k_w)$, each local receptive field patch of size $C_{\text{in}} \times k_h \times k_w$ is extracted and flattened into a column of a large 2D matrix of shape $(C_{\text{in}} k_h k_w, H_{\text{out}} W_{\text{out}})$. The weights are flattened into shape $(C_{\text{out}}, C_{\text{in}} k_h k_w)$. Spatial convolution then becomes a single dense matrix multiplication:
     $$
     Y_{\text{flat}} = K_{\text{flat}} \times X_{\text{im2col}} \in \mathbb{R}^{C_{\text{out}} \times (H_{\text{out}} W_{\text{out}})}.
     $$
     This leverages NVIDIA cuBLAS / CUTLASS GEMM kernels, achieving $>90\%$ of peak Tensor Core FLOPS.
   - **Memory footprint bottleneck:** im2col duplicates overlapping pixel data in GPU High Bandwidth Memory (HBM). For a $3 \times 3$ kernel with unit stride, each input pixel appears in up to 9 separate columns, expanding activation memory footprint by up to $k_h k_w = 9\times$. For high-resolution image generators (e.g., $1024 \times 1024$ latent diffusion), materializing $X_{\text{im2col}}$ would cause Out-Of-Memory (OOM) errors. Production engines use **Implicit GEMM**, where address offsets are calculated on-the-fly in GPU registers without materializing the expanded matrix in global VRAM.

2. **Winograd Minimal Filtering Algorithm ($F(m \times m, r \times r)$):**
   - **Mechanism:** For small kernels (e.g., $3 \times 3$, $r=3$) and unit stride, Winograd's algorithm transforms tiles of input data and weights into the Winograd domain:
     $$
     Y = A^\top \left[ (G g G^\top) \odot (B^\top d B) \right] A,
     $$
     where $g$ is the $3 \times 3$ kernel, $d$ is a $4 \times 4$ input tile, and $G, B, A$ are fixed transformation matrices. For $F(2 \times 2, 3 \times 3)$, computing 4 output points requires only $4 \times 4 = 16$ multiplications instead of the direct $2 \times 2 \times 3 \times 3 = 36$ multiplications—a **$2.25\times$ theoretical arithmetic reduction**.
   - **Numerical stability hazard:** Winograd transformation matrices contain rational powers that amplify floating-point rounding errors. In reduced precision (`fp16` or `bf16`), Winograd can cause severe gradient divergence or loss spikes. Consequently, production libraries (cuDNN) fall back to implicit GEMM or require `float32` accumulation for Winograd tiles.

3. **Memory bandwidth vs. Compute bounds:**
   - In low-channel layers (early encoder stages), arithmetic intensity is low ($\approx 10\text{ FLOPs/byte}$), making execution **memory-bandwidth bound**. In deep bottleneck layers ($C_{\text{in}} = C_{\text{out}} = 512$), arithmetic intensity rises to $>100\text{ FLOPs/byte}$, fully saturating GPU Tensor Cores.

### Optional depth: receptive-field span

Let $R_0=1$ and $J_0=1$, where $R_\ell$ is the dependency span in original-input cells and $J_\ell$ is the spacing between adjacent layer-$\ell$ positions measured in original cells. A new kernel joins $k_\ell$ earlier positions spaced $d_\ell J_{\ell-1}$ apart. The first position already covers $R_{\ell-1}$ cells; the other $k_\ell-1$ gaps extend that span:

$$R_\ell=R_{\ell-1}+(k_\ell-1)d_\ell J_{\ell-1},
\qquad J_\ell=s_\ell J_{\ell-1}.$$

This is a span, not necessarily a count of distinct used pixels when dilation creates gaps. A size-5, stride-2 layer gives $R_1=5,J_1=2$; a following size-2, stride-2 pool gives $R_2=5+(2-1)2=7,J_2=4$. Boundaries may include padded coordinates rather than actual data.

We have derived the training equations and separated smooth behavior from ties and information loss. The remaining check is whether those equations agree with the original numbers; the next section calculates every contribution before using a framework.

## 9. Calculate it by hand

### Forward values and a precisely defined diagnostic loss

Use the exact $X,K$ from §2:

$$
\begin{aligned}
Y_{00}&=1(1)+0(2)-1(0)+2(3)=7,\\
Y_{01}&=1(2)+0(0)-1(3)+2(1)=1,\\
Y_{10}&=1(0)+0(3)-1(2)+2(0)=-2,\\
Y_{11}&=1(3)+0(1)-1(0)+2(1)=5.
\end{aligned}
$$

Define the diagnostic scalar loss $\mathcal L=\sum_{ij}G_{ij}Y_{ij}$ with fixed $G=\begin{bmatrix}1&-0.5\\0.2&-1\end{bmatrix}$. Then

$$\mathcal L=1(7)-0.5(1)+0.2(-2)-1(5)=1.1.$$

Each derivative $\partial\mathcal L/\partial Y_{ij}$ is its fixed coefficient $G_{ij}$. This linear diagnostic loss is chosen to test a nonuniform backward signal; it is not a bounded image-training objective and has no finite unconstrained minimum when its parameter gradient is nonzero.

### Shared-kernel gradient

Take the corresponding input entry from all four patches for each coefficient:

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial K_{00}}
&=1(1)-0.5(2)+0.2(0)-1(3)=-3,\\
\frac{\partial\mathcal L}{\partial K_{01}}
&=1(2)-0.5(0)+0.2(3)-1(1)=1.6,\\
\frac{\partial\mathcal L}{\partial K_{10}}
&=1(0)-0.5(3)+0.2(2)-1(0)=-1.1,\\
\frac{\partial\mathcal L}{\partial K_{11}}
&=1(3)-0.5(1)+0.2(0)-1(1)=1.5.
\end{aligned}
$$

The positive and negative upstream values can cancel. The gradient is a sum of all uses of the shared weight, not a recommendation from one favored output.

### Input gradient: scatter and add

The four output positions contribute $1K,-0.5K,0.2K,-1K$ to their input patches. Listing each input cell's nonzero or potentially nonzero paths gives

$$
\nabla_X\mathcal L=
\begin{bmatrix}
1(1)&1(0)-0.5(1)&-0.5(0)\\
1(-1)+0.2(1)&1(2)-0.5(-1)+0.2(0)-1(1)&-0.5(2)-1(0)\\
0.2(-1)&0.2(2)-1(-1)&-1(2)
\end{bmatrix}
=\begin{bmatrix}1&-0.5&0\\-0.8&1.5&-1\\-0.2&1.4&-2\end{bmatrix}.
$$

### One parameter update

Hold $X$ fixed and use $K'=K-\eta\nabla_K\mathcal L$ with $\eta=0.1$:

$$K'=\begin{bmatrix}1+0.3&0-0.16\\-1+0.11&2-0.15\end{bmatrix}
=\begin{bmatrix}1.3&-0.16\\-0.89&1.85\end{bmatrix}.$$

Recomputing the four weighted sums gives

$$Y'=\begin{bmatrix}1.3-0.32+5.55&2.6-2.67+1.85\\-0.48-1.78&3.9-0.16+1.85\end{bmatrix}
=\begin{bmatrix}6.53&1.78\\-2.26&5.59\end{bmatrix}.$$

Therefore $\mathcal L'=6.53-0.89-0.452-5.59=-0.402$. This also follows from linearity: $\mathcal L'=\mathcal L-\eta\|\nabla_K\mathcal L\|_F^2=1.1-0.1(9+2.56+1.21+2.25)=-0.402$. A negative value is permitted for this diagnostic loss. This exact single-step decrease for a linear loss is not a convergence claim about neural-network training.

### Contrasting objectives: max versus average

For the original $Y$, max pooling gives $q=7$. If the loss is now **defined instead** as $q$, the output gradient is $G_{max}=\begin{bmatrix}1&0\\0&0\end{bmatrix}$, so $\nabla_Kq=\begin{bmatrix}1&2\\0&3\end{bmatrix}$, the winning patch. It is not the earlier $\nabla_K\mathcal L$.

Average pooling gives $(7+1-2+5)/4=2.75$, output gradient $1/4$ everywhere, and

$$\nabla_Kq_{avg}=\tfrac14\begin{bmatrix}1+2+0+3&2+0+3+1\\0+3+2+0&3+1+0+1\end{bmatrix}
=\begin{bmatrix}1.5&1.5\\1.25&1.25\end{bmatrix}.$$

For a tied window $[2,2;0,0]$, the maximum is 2, but there is no single ordinary gradient at that point. Moving only the first entry upward changes the maximum; moving it slightly downward leaves the second entry as the winner.

### Contrasting operator: a transposed stamp

Use a separate kernel $K_t=\begin{bmatrix}1&2\\0&1\end{bmatrix}$ and one input value 3. With no padding and unit stride, its transposed convolution places $3K_t=\begin{bmatrix}3&6\\0&3\end{bmatrix}$ on the output grid. Applying the corresponding ordinary convolution to that result gives $3+12+0+3=18$, not 3. This is an explicit adjoint-versus-inverse counterexample.

### Second Case: Strided & Padded Boundary Evaluation

To test boundary conditions and downsampling, evaluate a 1D input array $X_{\text{case2}} = [3.0, 1.0, 4.0]$ with kernel $K = [2.0, -1.0]$, stride $s=2$, and explicit zero-padding $p=1$:
1. **Zero-padded input:** Pad 1 zero on each boundary:
   $$X_{\text{pad}} = [0.0, 3.0, 1.0, 4.0, 0.0]$$
2. **Output size formula:**
   $$W_o = \left\lfloor \frac{W + 2p - k}{s} \right\rfloor + 1 = \left\lfloor \frac{3 + 2(1) - 2}{2} \right\rfloor + 1 = \left\lfloor \frac{3}{2} \right\rfloor + 1 = 1 + 1 = 2$$
3. **Forward evaluations:**
   - Window 0 (offset 0): $2.0(X_{\text{pad}}[0]) - 1.0(X_{\text{pad}}[1]) = 2.0(0.0) - 1.0(3.0) = -3.0$
   - Window 1 (offset 2, shifted by stride $s=2$): $2.0(X_{\text{pad}}[2]) - 1.0(X_{\text{pad}}[3]) = 2.0(1.0) - 1.0(4.0) = 2.0 - 4.0 = -2.0$
   - Output: $Y_{\text{case2}} = [-3.0, -2.0]$
4. **Backward gradients:** For scalar diagnostic loss $\mathcal{L} = Y_0 + Y_1 = -5.0$ (upstream $G = [1.0, 1.0]$):
   - Kernel gradient:
     $$\frac{\partial \mathcal{L}}{\partial K_0} = 1.0(X_{\text{pad}}[0]) + 1.0(X_{\text{pad}}[2]) = 0.0 + 1.0 = +1.0$$
     $$\frac{\partial \mathcal{L}}{\partial K_1} = 1.0(X_{\text{pad}}[1]) + 1.0(X_{\text{pad}}[3]) = 3.0 + 4.0 = +7.0$$
     $$\nabla_K \mathcal{L} = [1.0, 7.0]$$
   - Padded input gradient:
     $$\nabla_{X_{\text{pad}}} \mathcal{L} = [K_0(1), K_1(1), K_0(1), K_1(1), 0] = [2.0, -1.0, 2.0, -1.0, 0.0]$$
     Stripping the padding gives input gradient $\nabla_{X_{\text{case2}}} \mathcal{L} = [-1.0, 2.0, -1.0]$.
5. **Parameter update ($\eta = 0.1$):**
   $$K' = K - \eta \nabla_K \mathcal{L} = [2.0 - 0.1(1.0), -1.0 - 0.1(7.0)] = [1.9, -1.7]$$
   Updated output on padded input: $Y' = [1.9(0) - 1.7(3), 1.9(1) - 1.7(4)] = [-5.1, -4.9]$, yielding updated loss $\mathcal{L}' = -5.1 - 4.9 = -10.0 < -5.0$.

The numbers now agree with the derived gradients and distinguish three different losses/operators across both interior and boundary regimes. We still need to know where these computations appear in a real generator; an inspected architecture connects the small arrays to channel and resolution changes.

## 10. Connect the concept to an actual system

Consider **DCGAN**, a convolutional image generator, using Nathan Inkawhich's official PyTorch reference implementation. Its generator expands a random latent vector into a $64\times64$ RGB image using transposed convolutions; its discriminator uses strided convolutions. This is a concrete implemented architecture, not a claim about a particular deployed service. The source is linked in §14; [chapter 09](09-Minimax_Game_and_GANs.md) develops the adversarial losses.

With latent width 100 and base channel width 64, its generator shapes are:

```text
Forward:  z (N,100,1,1)
             |
          transpose conv (k=4, s=1, p=0)
             v
          (N,512,4,4)
             |
          transpose conv blocks (k=4, s=2, p=1)
             v
          256x8x8 -> 128x16x16 -> 64x32x32 -> 3x64x64 -> tanh -> image
                                                                        |
                                                          discriminator -> loss
Backward: generator weights <- chain rule <- image gradient <-----------+
```

The first layer creates a four-by-four map; the later layers double each spatial dimension by the transpose-size formula. Intermediate generator blocks also contain normalization and nonlinear activation; the spatial operators alone do not define the full network.

During a generator update, differentiation passes through the discriminator into the generated image, then through these layers into generator parameters. During a discriminator update, generated images are treated as fixed inputs for that update. The gradients derived here are the local pieces of the larger chain rule.

| Mathematical Object | Role in Toy Example | Real Production System Counterpart | What Changes or is Approximate in Practice |
| :--- | :--- | :--- | :--- |
| $X$ (Layer Input) | 9 real numbers ($3\times 3$ grid) | Batched feature maps | Logical shape $(N,C,H,W)$; contiguous NCHW vs channels-last NHWC memory layouts |
| $K$ (Kernel Weights) | 4 shared coefficients ($2\times 2$) | Learned multi-channel spatial filter banks | A $512\to 256$ transpose layer with $4\times 4$ kernel has $512\times 256\times 16=2,097,152$ weights (8 MiB in float32) |
| $Y$ (Output Activations) | 4 response values ($2\times 2$) | Intermediate generator latent activations | $(1,64,32,32)$ has 65,536 values: 256 KiB in float32 or 128 KiB in fp16/bf16 |
| Upstream Gradient $G$ | Prescribed $2\times 2$ gradient | Loss derivative arriving from discriminator | Backpropagation additionally caches activations for gradient computation; total training VRAM is $\approx 3\times$ forward memory |

These are element-count calculations, not GPU benchmarks. Patch extraction followed by matrix multiplication is one mathematical implementation of convolution. A backend can choose a different algorithm without changing the intended operator; explicit patch materialization can consume extra memory. Reduced precision changes rounding, and small finite-difference perturbations can disappear in float32 or bf16, so §11 uses CPU float64 for verification. Faster execution from a layout or algorithm must be measured on the actual workload.

Upsampling can also use resize followed by convolution. Distill's visual analysis in §14 explains how uneven overlap can create checkerboard patterns in transposed convolutions; choosing kernel sizes divisible by stride removes one overlap imbalance but does not rule out learned artifacts. A generator need not use max pooling, and a diffusion architecture need not use this DCGAN upsampling scheme.

We have located the mathematics in an actual network and accounted for shapes and bytes. The unanswered question is whether our hand calculations and edge cases survive independent execution; a small two-stage program can check that without downloading a model.

## 11. Verify the idea with a small experiment

Save this single block as a Python file and run it in an environment with PyTorch installed. It uses synthetic values only. Stage 1 exposes the loops and computes analytical and central-difference gradients; Stage 2 independently asks PyTorch to compute the same quantities and checks additional shapes, adjoints, and boundaries. Fixed seeds make the randomly generated checks repeatable on the tested CPU setup.

```python
import math
import random
from copy import deepcopy

random.seed(42)

# Stage 1: explicit finite sums using only the standard library.
def shape(a):
    if not a or not a[0] or any(len(row) != len(a[0]) for row in a):
        raise ValueError("Expected a nonempty rectangular matrix")
    if not all(math.isfinite(v) for row in a for v in row):
        raise ValueError("All entries must be finite")
    return len(a), len(a[0])


def corr(x, k, stride=1, padding=0, dilation=1):
    h, w = shape(x)
    kh, kw = shape(k)
    if any(type(v) is not int for v in (stride, padding, dilation)):
        raise ValueError("Operator settings must be integers")
    if stride < 1 or dilation < 1 or padding < 0:
        raise ValueError("Invalid operator settings")
    eh, ew = 1 + dilation * (kh - 1), 1 + dilation * (kw - 1)
    if h + 2 * padding < eh or w + 2 * padding < ew:
        raise ValueError("Kernel span exceeds padded input")
    oh = (h + 2 * padding - eh) // stride + 1
    ow = (w + 2 * padding - ew) // stride + 1
    y = [[0.0] * ow for _ in range(oh)]
    for i in range(oh):
        for j in range(ow):
            for a in range(kh):
                for b in range(kw):
                    u = i * stride - padding + a * dilation
                    v = j * stride - padding + b * dilation
                    if 0 <= u < h and 0 <= v < w:
                        y[i][j] += k[a][b] * x[u][v]
    return y


def pool(x, size=2, stride=2, mode="max"):
    h, w = shape(x)
    if (type(size) is not int or type(stride) is not int
            or min(size, stride) < 1 or min(h, w) < size
            or mode not in ("max", "avg")):
        raise ValueError("Invalid pooling configuration")
    y = []
    for i in range(0, h - size + 1, stride):
        row = []
        for j in range(0, w - size + 1, stride):
            values = [x[i+a][j+b] for a in range(size) for b in range(size)]
            row.append(max(values) if mode == "max" else sum(values)/len(values))
        y.append(row)
    return y


def backward(x, k, g):
    # Restricted, deliberately readable: valid, stride=dilation=1.
    h, w = shape(x)
    kh, kw = shape(k)
    assert shape(g) == (h - kh + 1, w - kw + 1)
    dx = [[0.0] * w for _ in range(h)]
    dk = [[0.0] * kw for _ in range(kh)]
    for i in range(len(g)):
        for j in range(len(g[0])):
            for a in range(kh):
                for b in range(kw):
                    dk[a][b] += g[i][j] * x[i+a][j+b]
                    dx[i+a][j+b] += g[i][j] * k[a][b]
    return dx, dk


def dot(a, b):
    assert shape(a) == shape(b)
    return sum(a[i][j] * b[i][j] for i in range(len(a)) for j in range(len(a[0])))


def close(a, b, tol=1e-8):
    return shape(a) == shape(b) and all(
        math.isclose(u, v, rel_tol=tol, abs_tol=tol)
        for row_a, row_b in zip(a, b) for u, v in zip(row_a, row_b))


def finite_difference(f, a, step=1e-5):
    out = [[0.0] * len(a[0]) for _ in a]
    for i in range(len(a)):
        for j in range(len(a[0])):
            plus, minus = deepcopy(a), deepcopy(a)
            plus[i][j] += step
            minus[i][j] -= step
            out[i][j] = (f(plus) - f(minus)) / (2 * step)
    return out


X = [[1., 2., 0.], [0., 3., 1.], [2., 0., 1.]]
K = [[1., 0.], [-1., 2.]]
G = [[1., -.5], [.2, -1.]]
Y = corr(X, K)
dX, dK = backward(X, K, G)
assert close(Y, [[7., 1.], [-2., 5.]])
assert close(dK, [[-3., 1.6], [-1.1, 1.5]])
assert close(dX, [[1., -.5, 0.], [-.8, 1.5, -1.], [-.2, 1.4, -2.]])
assert close(dK, finite_difference(lambda k: dot(corr(X, k), G), K))
assert close(dX, finite_difference(lambda x: dot(corr(x, K), G), X))
K_new = [[K[i][j] - .1*dK[i][j] for j in range(2)] for i in range(2)]
assert math.isclose(dot(Y, G), 1.1, abs_tol=1e-12)
assert math.isclose(dot(corr(X, K_new), G), -.402, abs_tol=1e-12)
assert pool(Y) == [[7.]]
assert pool(Y, mode="avg") == [[2.75]]
# Four separate windows: this catches accidental whole-array pooling.
P = [[float(4*i+j) for j in range(4)] for i in range(4)]
assert pool(P) == [[5., 7.], [13., 15.]]
assert pool(P, mode="avg") == [[2.5, 4.5], [10.5, 12.5]]
try:
    corr([[1.]], K)
except ValueError:
    pass
else:
    raise AssertionError("An oversized kernel must be rejected")

# Stage 2: production operators, CPU float64, and independent autograd.
import torch
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def tensor(a):
    return torch.tensor(a, dtype=torch.float64)


x = tensor(X).reshape(1, 1, 3, 3).requires_grad_()
k = tensor(K).reshape(1, 1, 2, 2).requires_grad_()
g = tensor(G).reshape(1, 1, 2, 2)
y = F.conv2d(x, k)
loss = (y * g).sum()
loss.backward()
assert torch.allclose(y[0, 0], tensor(Y), atol=1e-12, rtol=0)
assert torch.allclose(k.grad[0, 0], tensor(dK), atol=1e-12, rtol=0)
assert torch.allclose(x.grad[0, 0], tensor(dX), atol=1e-12, rtol=0)
# The module API has the same weight ordering; bias disabled for this toy.
layer = torch.nn.Conv2d(1, 1, 2, bias=False)
with torch.no_grad():
    layer.weight.copy_(k)
assert torch.allclose(layer(x), y, atol=1e-12, rtol=0)

for mode in ("max", "avg"):
    py = tensor(Y).reshape(1, 1, 2, 2).requires_grad_()
    q = F.max_pool2d(py, 2) if mode == "max" else F.avg_pool2d(py, 2)
    q.sum().backward()
    expected = [[1., 0.], [0., 0.]] if mode == "max" else [[.25]*2]*2
    assert torch.allclose(py.grad[0, 0], tensor(expected), atol=1e-12, rtol=0)
    assert torch.allclose(q[0, 0], tensor(pool(Y, mode=mode)))
tied = tensor([[2., 2.], [0., 0.]]).reshape(1, 1, 2, 2).requires_grad_()
F.max_pool2d(tied, 2).sum().backward()
assert tied.grad.min() >= 0 and tied.grad.sum() == 1
assert torch.count_nonzero(tied.grad[0, 0, 1]) == 0

# Dilation, padding, and stride agree with the explicit indexing formula.
x5 = [[float(i*5+j) for j in range(5)] for i in range(5)]
for s, p, d in [(1, 0, 2), (2, 1, 1), (2, 2, 2)]:
    actual = F.conv2d(tensor(x5)[None, None], k.detach(), stride=s, padding=p, dilation=d)
    assert torch.allclose(actual[0, 0], tensor(corr(x5, K, s, p, d)))

# Adjoint identity with stride 2: output_padding=1 selects a 5x5 input space.
a = torch.randn(1, 1, 5, 5, requires_grad=True)
w = torch.randn(1, 1, 2, 2)
b = F.conv2d(a, w, stride=2)
v = torch.randn_like(b)
atv = F.conv_transpose2d(v, w, stride=2, output_padding=1)
assert atv.shape == a.shape
assert torch.allclose((b*v).sum(), (a*atv).sum(), atol=1e-12, rtol=0)
(b*v).sum().backward()
assert torch.allclose(a.grad, atv, atol=1e-12, rtol=0)
kt = tensor([[1., 2.], [0., 1.]])[None, None]
stamp = F.conv_transpose2d(tensor([[[[3.]]]]), kt)
assert torch.allclose(stamp[0, 0], tensor([[3., 6.], [0., 3.]]))
assert F.conv2d(stamp, kt).item() == 18.

# Periodic boundaries + stride 1 preserve cyclic translation equivariance.
u = torch.randn(1, 1, 5, 5)
filt = torch.randn(1, 1, 3, 3)
def periodic(t):
    return F.conv2d(F.pad(t, (1, 1, 1, 1), mode="circular"), filt)
shifted = torch.roll(u, (1, -1), (2, 3))
assert torch.allclose(periodic(shifted), torch.roll(periodic(u), (1, -1), (2, 3)))
# Zero-filled finite shifts need not commute with boundary convolution.
e = tensor([[[[1., 0., 0.]]]])
def right_zero(t):
    return F.pad(t[..., :-1], (1, 0))
def edge_conv(t):
    return F.conv2d(t, torch.ones(1, 1, 1, 3), padding=(0, 1))
assert not torch.allclose(edge_conv(right_zero(e)), right_zero(edge_conv(e)))
assert pool([[0., 1., 0., 0.]], size=1) == [[0., 0.]]  # stride is 2
assert pool([[0., 0., 1., 0.]], size=1) == [[0., 1.]]
q1 = F.max_pool1d(tensor([[[0., 1., 0., 0.]]]), 2)
q2 = F.max_pool1d(tensor([[[0., 0., 1., 0.]]]), 2)
assert not torch.equal(q1, q2)

print("Y =", Y, "loss =", loss.item())
print("dK =", dK, "dX =", dX)
print("Updated loss =", dot(corr(X, K_new), G))
print("All reference, gradient, shape, pooling, and adjoint checks passed.")
```

Expected main values are $Y=[[7,1],[-2,5]]$, loss 1.1, and updated loss −0.402, up to floating-point rounding. `allclose` checks a stated tolerance, not bitwise identity. Finite differences are an independent diagnostic at smooth points, not a proof; §8 supplies the algebra, and the tied-max test checks a valid subgradient rather than a nonexistent ordinary derivative. The tests do not establish GPU speed, training convergence, or perceptual image quality.

We have compared independent implementations and tested the assumptions where failures are expected. The next step is to see whether you can select and use the idea without following the same numbers; the exercises deliberately change the task as well as the data.

## 12. Practise, compare, and debug

Attempt all five questions before opening the answer key.

1. **Recognize.** A unit-stride detector runs on a periodic image grid. After a one-cell cyclic input shift, its score peak moves one cell too. Is this invariance or equivariance? Would fixed-size zero-filled shifts support the same statement at every boundary?
2. **Calculate.** Let $X=[2,1,3,0]$, $K=[1,-1]$, valid stride-1 cross-correlation, and $\mathcal L=Y_0+2Y_1-Y_2$. Find $Y$, $\mathcal L$, the two kernel derivatives, and the result of a learning-rate-0.1 kernel update. This is a one-dimensional version of the same indexing rule.
3. **Contrast.** You need to preserve the precise location of a single bright pixel inside each two-by-two block so that a decoder can recover it. Is retaining only its maximum value sufficient? Compare keeping the whole block, maximum plus winner index, and a learned strided layer.
4. **Transfer.** A one-dimensional audio network applies a size-3 kernel with dilation 2 and stride 2, followed by a size-3 kernel with dilation 1 and stride 1. What are its theoretical receptive-field span and output spacing in original samples? Is every sample inside the span necessarily used?
5. **Debug.** An implementation computes one `max` across the entire array while accepting a `pool_size` argument it never reads. Its only test uses a two-by-two array and `pool_size=2`. Explain why the test passes, design the smallest useful larger test, and explain why a transposed convolution cannot generally undo that pooling.

<details>
<summary>Answer key and diagnostic feedback</summary>

1. **Equivariance**, under the stated compatible periodic boundary convention. Invariance would leave the complete output array unchanged. “The detector recognizes the same pattern” is tempting language, but it does not say whether its location changed. Zero-filled finite shifts can discard information and break the equation at boundaries; §11 contains such a counterexample.
2. $Y=[2-1,1-3,3-0]=[1,-2,3]$, so $\mathcal L=1+2(-2)-3=-6$. The coefficient derivatives are $2+2(1)-3=1$ and $1+2(3)-0=7$. Thus $K'=[0.9,-1.7]$. Its outputs are $[1.8-1.7,0.9-5.1,2.7-0]=[0.1,-4.2,2.7]$, and its loss is $0.1-8.4-2.7=-11$. Updating only from the first patch gives $[2,1]$, which overlooks shared use at the other positions. The decrease also equals $0.1(1^2+7^2)=5$ for this linear objective.
3. Maximum alone is insufficient: all four placements of one value 1 among three zeros give maximum 1. The full block preserves all values. Maximum plus winner index reconstructs this restricted single-bright-pixel family, but not an arbitrary block's discarded nonmaximum values. A learned strided layer may preserve needed information using enough output channels, but striding alone supplies no invertibility guarantee. “Learned” does not imply information-preserving.
4. Start at $R_0=J_0=1$. First layer: $R_1=1+(3-1)2=5$, $J_1=2$. Second: $R_2=5+(3-1)1(2)=9$, $J_2=2$. The possible offsets here are $\{0,2,4,6,8\}$, so odd offsets are absent even though the span is nine samples. Adding kernel sizes without accumulated spacing incorrectly treats intermediate positions as adjacent original samples.
5. The whole array is also the only pooling window in the original test. Use the four-by-four array 0 through 15 in row-major order: size/stride-2 max pooling must return $[[5,7],[13,15]]$, not just 15. Discarded values are not encoded in that output, so a transpose cannot reconstruct them. An adjoint reverses an inner-product pairing for a fixed linear map; max pooling is nonlinear and has many inputs for one output.

</details>

The exercises check recognition, calculation, operator choice, transfer, and debugging separately. What remains is reliable recall after the derivation is no longer in front of you; reconstructing the explanation is a stronger check than recognizing its wording.

## 13. Explain it back and return to it

**Closed-notes explanation:** Explain to a colleague, without using “equivariance,” “adjoint,” or “receptive field,” why one four-number rule can be useful across an image, why all its uses contribute to one update, and why making a larger image from a smaller one does not recover lost data. Then restore those three terms and write their equations.

<details>
<summary>Model explanation to compare after your attempt</summary>

We apply the same small weighted sum at many positions because the desired local computation should be shared. A later error depends on several outputs, so each use contributes to the derivative of each shared coefficient. Moving a pattern moves its score map when the sampling and boundaries permit the same computation at both positions. A reverse linear operation distributes output feedback over earlier input positions, but many earlier images can have the same compressed output.

The restored relationships are $\operatorname{Conv}(\tau_vX)=\tau_v\operatorname{Conv}(X)$ on the compatible unit-stride domain; $\langle Ax,g\rangle=\langle x,A^Tg\rangle$; and $R_\ell=R_{\ell-1}+(k_\ell-1)d_\ell J_{\ell-1}$ with $J_\ell=s_\ell J_{\ell-1}$. The kernel gradient is the sum $\sum_{ij}G_{ij}X_{i+a,j+b}$, not one patch's contribution.

</details>

| Return date | Closed-notes task | Check after attempting |
| :--- | :--- | :--- |
| Day 1 | Derive the output-size formula by counting starting positions. Explain the final +1 and the dilation term. | §4: count starts from 0 and use span $1+d(k-1)$. |
| Day 7 | Repeat exercise 2 with upstream coefficients $[1,0,1]$ instead of $[1,2,-1]$. | Outputs remain $[1,-2,3]$; loss 4; kernel gradient $[5,1]$; updated kernel $[0.5,-1.1]$; new loss 1.4. |
| Day 30 | Design a strided sensor operator for an audio stream. Explain which shifts preserve its output relationship and show two inputs that a pooling step cannot distinguish. | State the input/output shift ratio and boundaries; use the same maximum with different discarded entries as a failure example. |

- [ ] I can compute the output shape with stride, padding, and dilation without guessing.
- [ ] I can state the domain of the shift proof and give a boundary or stride counterexample.
- [ ] I can derive and calculate both shared-kernel and input gradients.
- [ ] I can explain max-pooling ties and the difference between continuity and differentiability.
- [ ] I can demonstrate that a transposed convolution is generally not an inverse.
- [ ] I can connect the operator to generator shapes and distinguish memory counts from speed claims.

You have a way to test both immediate understanding and delayed recall. The final choice is where to deepen a specific weak point; the resources below have distinct jobs and verified starting locations.

## 14. Continue with a purposeful learning path

Use the visual source first if patch motion is still unclear, the textbook for another derivation, and the numbered exercises to test transfer. The five required tiers are identified below. Source contents and named starting points were inspected on **2026-09-18**; access descriptions refer to that check, not a promise of permanent availability.

| Resource and author | Learning job | Exact starting point | Readiness | Access | Checked date and evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Interactive visualizer:** [CNN Explainer](https://poloclub.github.io/cnn-explainer/), Jay Wang, Robert Turko, Omar Shaikh, Haekyu Park, Nilaksh Das, Fred Hohman, Minsuk Kahng, Polo Chau | Interactively inspect 10-layer CNN forward activations, kernel weights, 2D convolution arithmetic, and pooling on live images | Click any Conv2D layer in the interactive Tiny VGG network to inspect sliding kernel calculations and feature map outputs | After §2 | Free open-source interactive browser tool | 2026-09-18: verified interactive 2D convolution sliding window animation, pooling maps, and real-time tensor slicing. |
| **Video lecture:** [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA), 3Blue1Brown (Grant Sanderson) | Geometric intuition of sliding windows, impulse responses, and discrete cross-correlation vs mathematical convolution | “Convolutions in 2D / Image Processing” (timestamp 14:20 to end) | After §2 | Free YouTube video | 2026-09-18: verified Grant Sanderson's visual breakdown of moving kernel weighted sums, boundary effects, and spatial filtering. |
| **Visual / beginner:** [CS231n convolutional-network notes](https://cs231n.github.io/convolutional-networks/), Stanford CS231n course | See sliding local connectivity and spatial pooling | “Convolutional Layer,” then “Pooling Layer” and its two illustrated panels | After §2 | Free HTML with diagrams/animation | 2026-09-18: inspected convolution and pooling material, shape discussion, and backward-routing explanation. Older dilation convention counts gaps; here/PyTorch count point spacing. |
| **Formal foundation:** [CS231n convolutional-network notes](https://cs231n.github.io/convolutional-networks/), Stanford CS231n course | Relate sharing, spatial arrangements, and matrix implementation | “Parameter Sharing,” “Implementation as Matrix Multiplication,” and “Backpropagation” | After §§4 and 8 | Free university notes | 2026-09-18: verified named passages and local-connection/weight-sharing treatment. Historical architecture advice is not a current universal prescription. |
| **Textbook:** [Dive into Deep Learning, §7.2](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html), Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola | Rebuild cross-correlation and connect it to learned filters | Web edition **1.0.3**, §§7.2.1, 7.2.5, 7.2.6 | After §3 | Free HTML; book publisher is Cambridge University Press, 2023 | 2026-09-18: inspected numerical cross-correlation, kernel-flip distinction, receptive-field section; book landing page confirms authors/publisher. Section numbers refer to this web edition. |
| **Practice:** [D2L §7.2.8 exercises](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html#exercises), same authors | Change orientation and translate a local operation into a matrix | Exercise **1, parts 1–3**, then exercise **4** | After §§8–12 | Free numbered exercises | 2026-09-18: inspected the numbered list: diagonal-edge/transpose questions and matrix representation. These are external problems, separate from this chapter's exercises. |
| **Visual technical blog:** [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/), Augustus Odena, Vincent Dumoulin, Chris Olah, 2016 | Understand overlapping transpose contributions and resize alternatives | “Deconvolution & Overlap,” then “Better Upsampling” | After §8's adjoint explanation | Free illustrated article | 2026-09-18: inspected overlap illustrations, stride divisibility caveat, and resize-convolution discussion; divisibility alone does not eliminate learned artifacts. |
| **Software:** [Conv2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html), PyTorch contributors | Match mathematical indices to weights and output shape | PyTorch **2.9**, operation definition, “Shape,” `groups` and `dilation` | When running §11 | Free official API documentation | 2026-09-18: inspected cross-correlation definition, shape equations, groups, and weight layout. |
| **Software:** [ConvTranspose2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.ConvTranspose2d.html), PyTorch contributors | Resolve transpose shape ambiguity | PyTorch **2.9**, opening description and `output_padding` notes | After §8 | Free official API documentation | 2026-09-18: inspected input-gradient interpretation, inverse warning, and output-shape controls. |
| **Software:** [MaxPool2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.MaxPool2d.html) and [AvgPool2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.AvgPool2d.html), PyTorch contributors | Check pooling padding and denominator conventions | PyTorch **2.9**, operation definitions; `ceil_mode`; average `count_include_pad` | After pooling calculation in §9 | Free official API documentation | 2026-09-18: inspected negative-infinity max padding and average denominator setting. The chapter uses no padding and floor mode. |
| **System implementation:** [DCGAN tutorial source](https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/dcgan_faces_tutorial.py), Nathan Inkawhich / PyTorch | Trace an implemented image generator and discriminator | “Generator,” `class Generator`, then “Discriminator” | After §10; chapter 09 for objectives | Free official source; full tutorial training requires image data, unlike §11 | 2026-09-18: inspected author, layer definitions, shapes and forward calls. The rendered tutorial exceeded the fetch limit; this source was directly inspected instead. |

**Next connection:** A convolutional encoder can feed the bottleneck in [autoencoders and latent spaces](03-Autoencoders_and_Latent_Spaces.md). A generator's transposed layers create an image, while [GAN objectives](09-Minimax_Game_and_GANs.md) determine how its weights are trained. Local spatial computation and the training objective are separate pieces of the same system.
