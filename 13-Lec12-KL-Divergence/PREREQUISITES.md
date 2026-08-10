# Prerequisites — warm-up before Lec 12 (KL divergence)

> **Do this first** if “cross-entropy,” “$p\log(p/q)$,” or “asymmetric distance” still blur.  
> Then open [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Builds on [Lec 11 Entropy](../12-Lec11-Entropy/PREREQUISITES.md) (surprisal; $H=-\sum p\log p$).  
> Strong basics: every formula decoded with micro numbers.

```
  After this warm-up you can say:

  "Cross-entropy H(p,q) = E_p[−log q]: surprise under q when data come from p."
  "If p=q, cross-entropy collapses to ordinary entropy H(p); always H(p,q) ≥ H(p)."
  "KL(p‖q) = H(p,q) − H(p) = ∑ p log(p/q) ≥ 0, and =0 iff p=q."
  "True law is the first slot (expectation under p); model is q."
  "KL is asymmetric, so it is a divergence, not a true metric."
  "Minimizing KL(p‖p_θ) is the same as minimizing H(p,p_θ) because H(p) does not depend on θ."
```

**Warm-up → lecture boxes**

```
  §1  Entropy reload                     ──► Topics 1–3
  §2  Two PMFs on the same alphabet      ──► Topics 1–2
  §3  Cross-entropy                      ──► Topic 2
  §4  KL formula + non-negativity idea   ──► Topics 3–4
  §5  Asymmetry / not a metric           ──► Topic 4
  §6  Continuous densities twin          ──► Topic 5
  §7  Recipe step 2 link                 ──► Topics 1, 5
  §8  Worked binary micros               ──► Topics 2–3
```

---

## 1. Entropy reload (one distribution)

<a id="p1-entropy"></a>

### Purpose for the video

KL is built from entropy + cross-entropy.

For discrete $p$:

$$
H(p)=\mathbb{E}_{X\sim p}\big[-\log p(X)\big]=-\sum_i p(x_i)\log p(x_i)
$$

| Reading | Meaning |
|---------|---------|
| Process | average surprisal per draw when $X\sim p$ |
| Units | bits if $\log_2$, nats if $\ln$ |
| Scope | **one** law $p$ only |

Micro (base 2): fair coin $H=1$ bit; sure outcome $H=0$.

---

## 2. Two mass functions on the same space

<a id="p2-two-pmfs"></a>

### Purpose for the video

Pairwise scores need two laws $p$ and $q$ on the **same** outcomes.

$$
p(x_i)\ge 0,\ \sum_i p(x_i)=1;\qquad
q(x_i)\ge 0,\ \sum_i q(x_i)=1
$$

**Notation care (he stresses):** here $p,q$ are **PMFs** (discrete), not the abstract probability measure $P$ on $\Omega$. Continuous twin uses **densities** (small $p$).

**Support caution:** if $p(x)>0$ but $q(x)=0$, then $\log(p/q)$ blows up — KL can be $+\infty$. Models should put mass wherever truth has mass.

---

## 3. Cross-entropy

<a id="p3-cross-entropy"></a>

### Purpose for the video

The middle object of the lecture.

$$
\boxed{H(p,q)=\mathbb{E}_{X\sim p}\big[-\log q(X)\big]=-\sum_i p(x_i)\log q(x_i)}
$$

| Piece | Role |
|-------|------|
| $-\log q(x_i)$ | surprisal **as if** the law were $q$ |
| weight by $p(x_i)$ | data actually come from **true** $p$ |
| sum | average surprise of the **wrong** model on true data |

**Special case:** if $q=p$, then $H(p,p)=H(p)$.

**Not symmetric:** $H(p,q)$ and $H(q,p)$ swap who is “truth” vs “model.”

### Process story

1. Nature samples $X$ from true $p$.  
2. You score with model $q$ using surprisal $-\log q(X)$.  
3. Average over many draws → cross-entropy.

### Why expect $H(p,q)\ge H(p)$? (intuition, not a full proof)

$H(p)$ is average surprise when you use the **correct** scoring rule $-\log p$.  
$H(p,q)$ is average surprise when you use a **possibly wrong** rule $-\log q$ on the same true data.  
A wrong codebook cannot make average surprise *smaller* than the true entropy (Gibbs inequality / information inequality — lecture leaves as homework). Equality only when $q=p$.

---

## 4. KL divergence = cross-entropy − entropy

<a id="p4-kl"></a>

### Purpose for the video

The pairwise score for “how wrong is $q$ for $p$.”

$$
\boxed{D_{\mathrm{KL}}(p\|q)=H(p,q)-H(p)
=\sum_i p(x_i)\log\frac{p(x_i)}{q(x_i)}
=\mathbb{E}_{p}\Big[\log\frac{p(X)}{q(X)}\Big]}
$$

| Form | Reading |
|------|---------|
| $H(p,q)-H(p)$ | extra average surprise beyond true entropy |
| $\sum p\log(p/q)$ | same algebra (log laws) |
| $D_{\mathrm{KL}}(p\|q)$ | “KL of $p$ **to** $q$” (double bar $=$ directed) |

**Order matters:** the first argument is the **reference / true** law (expectation under $p$); the second is the **model / approximation**.

### Algebra bridge (must see once)

$$
\begin{aligned}
H(p,q)-H(p)
&= \Big(-\sum_i p_i\log q_i\Big)
  -\Big(-\sum_i p_i\log p_i\Big) \\
&= \sum_i p_i\log p_i - \sum_i p_i\log q_i
  = \sum_i p_i\log\frac{p_i}{q_i}.
\end{aligned}
$$

Same story with $\log(p/q)=\log p-\log q$ inside the expectation under $p$.

### Why non-negative? (idea only)

From CE $\ge$ H: $D_{\mathrm{KL}}=H(p,q)-H(p)\ge 0$, with equality **iff** $p=q$.  
Homework: prove non-negativity and the iff using log laws + definitions (lecture does not fully prove it on the board).

**Board caution:** he briefly wrote “entropy minus cross-entropy”; class corrected to **cross-entropy minus entropy** so the difference is non-negative.

### Training link (why people minimize CE)

If truth $p$ is fixed (or fixed dataset) and model is $q=p_\theta$,

$$
\arg\min_\theta D_{\mathrm{KL}}(p\|p_\theta)
=\arg\min_\theta H(p,p_\theta)
$$

because $H(p)$ does **not** depend on $\theta$. So minimizing cross-entropy training loss is the same as minimizing KL to the true law (when that makes sense).

---

## 5. Asymmetry and “not a metric”

<a id="p5-not-metric"></a>

### Purpose for the video

Lec 10 put “metric” in quotes — KL is the classic reason.

| Property | True metric | KL |
|----------|-------------|-----|
| Non-negative | yes | yes (when defined) |
| $=0$ iff same | yes | yes ($p=q$) |
| Symmetric $d(a,b)=d(b,a)$ | required | **fails** in general |
| Triangle inequality | required | **fails** (already fails symmetry) |

So KL is a **divergence**, not a distance metric. Still excellent for “how far is model from truth.”

$$
D_{\mathrm{KL}}(p\|q)\ \neq\ D_{\mathrm{KL}}(q\|p)\quad\text{(in general)}
$$

---

## 6. Continuous twin (densities)

<a id="p6-continuous"></a>

### Purpose for the video

Most ML data treated as continuous.

**Differential** (continuous) objects:

$$
h(p)=-\int p(x)\log p(x)\,dx,\qquad
H(p,q)=-\int p(x)\log q(x)\,dx,\qquad
D_{\mathrm{KL}}(p\|q)=\int p(x)\log\frac{p(x)}{q(x)}\,dx
$$

(same CE−H story when the integrals exist).

- $p,q$ are **densities** (heights; not probabilities at points).  
- Structure matches discrete sum → integral.  
- Differential entropy has a subtler interpretation than discrete $H$ (can be negative, etc.) — still used the same way to build KL.  
- Lecture micro: true density mean $0$ vs model mean $3$ is about scoring samples from the true density with the wrong density (cross-entropy / KL story).

---

## 7. Why this fills recipe step 2

<a id="p7-recipe"></a>

### Purpose for the video

```
  (1) model p_θ
  (2) d(true, model)     ← KL is one standard choice
  (3) θ* = argmin d
```

Training often amounts to **minimizing KL** (or cross-entropy — same $\arg\min$ over $\theta$; see §4).  
KL is one **$f$-divergence** example (whole family deferred to later / other courses).

**Convention in the recipe:** score $d(p_{\text{true}}, p_\theta)=D_{\mathrm{KL}}(p_{\text{true}}\|p_\theta)$ — truth first, model second (extra surprise of the model on true data). The reverse $D_{\mathrm{KL}}(p_\theta\|p_{\text{true}})$ is a *different* objective (sometimes called reverse KL); not the same training problem.

---

## 8. Worked binary micros

<a id="p8-micro"></a>

### Fair coin vs biased model (base 2)

True $p=(1/2,1/2)$. Model $q=(3/4,1/4)$.

$$
\begin{aligned}
H(p)&=1\\
H(p,q)&=-\tfrac12\log_2\tfrac34-\tfrac12\log_2\tfrac14
\approx 0.5\cdot 0.415+0.5\cdot 2=1.207\\
D_{\mathrm{KL}}(p\|q)&=H(p,q)-H(p)\approx 0.207\text{ bits}
\end{aligned}
$$

Check the sum form:  
$\tfrac12\log_2\frac{1/2}{3/4}+\tfrac12\log_2\frac{1/2}{1/4}
=\tfrac12\log_2\frac23+\tfrac12\log_2 2\approx 0.5(-0.585)+0.5(1)=0.207$.

### Swap (asymmetry) — different number

Now truth $=q=(3/4,1/4)$, model $=p=(1/2,1/2)$:

$$
\begin{aligned}
H(q)&=-\tfrac34\log_2\tfrac34-\tfrac14\log_2\tfrac14
\approx 0.811\\
H(q,p)&=-\tfrac34\log_2\tfrac12-\tfrac14\log_2\tfrac12=1\\
D_{\mathrm{KL}}(q\|p)&=1-0.811\approx 0.189\text{ bits}
\end{aligned}
$$

$\approx 0.189\neq 0.207$ — **asymmetric**. Same pair of laws, two different directed costs.

### Match

If $q=p$, then $H(p,q)=H(p)$ and $D_{\mathrm{KL}}=0$.

### Paper check

1. Write $H(p)$ and $H(p,q)$ in expectation form.  
2. Expand $H(p,q)-H(p)$ into $\sum p\log(p/q)$ line by line.  
3. Why is $H(p,q)$ not equal to $H(q,p)$ in words?  
4. Compute both $D_{\mathrm{KL}}(p\|q)$ and $D_{\mathrm{KL}}(q\|p)$ for the binary micro.  
5. Why does $\arg\min_\theta\mathrm{KL}(p\|p_\theta)=\arg\min_\theta H(p,p_\theta)$?  
6. List three KL properties from the homework.  
7. Why is KL not a metric?  
8. Write continuous KL with an integral.  
9. Which recipe step does KL fill? Which argument is the model?

---

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html).  
Prior: [Lec 11 Entropy](../12-Lec11-Entropy/NOTES.md).
