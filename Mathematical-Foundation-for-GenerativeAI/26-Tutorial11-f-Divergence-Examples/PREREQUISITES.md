# Prerequisites — warm-up before Tutorial 11 (f-Divergence proofs)

> **Do this first** if Jensen, “almost surely,” or “metric” still blur.  
> Then [NOTES.md](./NOTES.md) at the **Executive Summary**.  
> Complements [Lec 03](../25-Lec03-f-Divergence-Examples/NOTES.md).  
> **Beginner:** purpose · definition · micro · analogy · notice · mini-check.

```
  After this warm-up you can say:

  "P is a law; p is its height."
  "R = p/q is a ratio of two heights — only legal if P ≪ Q."
  "A convex cup never sits above its chords."
  "Jensen: average of f ≥ f of the average."
  "An expectation under q is ∫ q(x) g(x) dx."
  "A metric needs four tickets, not two."
  "A Bernoulli is two piles that add to one."
  "Almost surely means except on a set q ignores."
```

```
  §1  Law P vs density p          ──► Topics 1, 5
  §2  Likelihood ratio R=p/q      ──► Topics 2–4
      and absolute continuity
  §3  Convex / strictly convex    ──► Topics 3–4, 7
  §4  Jensen's inequality         ──► Topics 3–4
  §5  Expectation as ∫ q g        ──► Topic 3
  §6  Four metric axioms          ──► Topics 9–10
  §7  Two-point Bernoulli         ──► Topics 9–10
  §8  Almost surely               ──► Topic 4
```

---

## 1. A law and its height

<a id="p1-law"></a>

### Purpose

The tablet writes $D_f(P\|Q)$ but the integral uses **densities** $p$ and $q$. If those two alphabets collapse in your head, every later formula is noise.

### Definitions

**P** (capital) is a **distribution** / **law**: the mass of a *region*, a number in $[0,1]$.  
**p** (small) is a **density**: a *height*. Area under $p$ is probability. Height $p(x)$ can be $3$. It is not $P(\{x\})$.

### Micro

Uniform on $[0,0.5]$ has $p=2$. Legal. $P([0,0.25])=0.5$ is the probability.  
Uniform on $[0,1]$ has $p=1$. Same *kind* of object, different height.

### Analogy

Jam thickness at a crumb vs how much jam a *bite* of toast gets. A photo of ten crumbs is neither.

### Notice

He assumes every law here has a density. Not every random variable does. When a density exists, $P(A)=\int_A p(x)\,dx$.

### Mini-check

1. Can $p(x)=4$?  
2. Is $p(x)$ the chance of that exact $x$?

---

## 2. A ratio of two heights (and when it is defined)

<a id="p2-ratio"></a>

### Purpose

The board’s **red line** (Topic 2): assume $P$ is **absolutely continuous** with respect to $Q$ so $p/q$ is well-defined whenever needed. That is *not* “assume $P=Q$.” Treating those as the same sentence is the one mistake that makes the rest of the hour look circular.

### Definitions

$$
R(x)=\frac{p(x)}{q(x)}
$$

is the **likelihood ratio**. Where $q$ is tall and $p$ is short, $R$ is small. Where they match, $R=1$.

**Absolutely continuous** $P\ll Q$ means: if $Q$ gives a set mass zero, $P$ must too. Then $P$ has a density with respect to $Q$, and you may divide. If $Q$ never visits a street, $P$ is not allowed to hide a mountain there — otherwise $p/q$ blows up.

$P=Q$ is much stronger: then $R\equiv 1$ everywhere $q$ cares, and $D_f$ will be zero. The red line only licenses the *division*.

Logs you will need constantly: $\log(a/b)=\log a-\log b$. That is how $f(u)=-\log u$ becomes $\int q\log(q/p)$.

### Micro

$p=2$, $q=1$ ⇒ $R=2$. $p=q$ ⇒ $R=1$.  
If $q(x)=0$ on a whole interval but $p>0$ there, $R$ is not a legal height — absolute continuity failed.

### Analogy

Two maps of one town. $R$ is “how many times taller is map A at this church?” You may only ask that on streets **map B bothers to draw**. Streets B left blank cannot hide A’s cathedral.

Asking “are the two maps *the same* map?” is a later question (Topic 4). Absolute continuity is only the license to *compare heights*.

### Notice

$f$ eats the **number** $R(x)$, not the whole function at once. The integral then averages those readings using $q$ as the visiting schedule.

### Mini-check

1. If $p=q$, what is $R$?  
2. Why is “$P=Q$” stronger than “$P\ll Q$”?

---

## 3. A convex cup (and a strictly convex one)

<a id="p3-convex"></a>

### Purpose

Every legal $f$ in this hour is **convex**. The “$=0$ only when the hills match” theorem needs a stronger adjective: **strictly convex at $1$**. Total variation’s $f$ fails that extra adjective — that is why TV is homework.

### Definitions

$f$ is **convex** if it never sits above a chord:

$$
f(\alpha u+(1-\alpha)v)\le \alpha f(u)+(1-\alpha)f(v).
$$

**Strictly convex at $1$** means the cup is *curved* there — the chord is *strictly* above the graph except at the endpoints. Then Jensen’s “$=$” is rare: it forces the random input to be a single number.

### Micro

$u\log u$ and $-\log u$ are strictly convex on $(0,\infty)$.  
$|u-1|$ is convex but **not** strictly convex (it is a V). On each side of $1$ the graph *is* a straight chord.

### Analogy

A hanging chain (strict sag) vs a V-shaped roof (flat sides). On the V, a whole segment of the chord *is* the roof — Jensen can be equal without $U$ being a single number. That is why TV needs a separate “$=0$ iff” argument.

### Notice

Convexity is what lets him *invoke* Jensen. Strict convexity at $1$ is what lets him *read equality*. Two different jobs.

### Mini-check

1. Is $|u-1|$ strictly convex at $1$?  
2. Why does the KL spring $u\log u$ get the cheap Jensen-equality story?

---

## 4. Jensen: average of $f$ versus $f$ of the average

<a id="p4-jensen"></a>

### Purpose

The non-negativity proof is three lines, and Jensen is line two. Tutorial 8 only *stated* it. This tutorial *uses* it. He will not reprove it (Wikipedia is fine for that proof).

### Definitions

If $f$ is convex and $U$ is a random number,

$$
\mathbb{E}[f(U)]\ge f\bigl(\mathbb{E}[U]\bigr).
$$

Equality, when $f$ is strictly convex, only if $U$ is **constant** (almost surely). If $f$ is merely convex, equality can hold on a whole segment.

### Micro

$U\in\{0,2\}$ each with chance $1/2$, $f(u)=u^2$. $\mathbb{E}[U^2]=2 > 1=f(1)$.  
If instead $U\equiv 1$, both sides are $1$. That is the equality case he needs.

### Analogy

The average height of two people is not the height of the “average person” if you first square the heights. The square is a cup; cups sit above their chords.

### Notice

He applies this to $U=R=p/q$ under the law $Q$. The right-hand side will collapse to $f(1)$ because $\mathbb{E}_Q[R]=1$.

### Mini-check

1. For $f(u)=u^2$, $U\in\{0,2\}$ fair, which is larger: $\mathbb{E}[U^2]$ or $f(\mathbb{E}[U])$?  
2. When does equality hold if $f$ is strictly convex?

---

## 5. Expectation under $q$ is an integral

<a id="p5-expect"></a>

### Purpose

$D_f$ looks like a scary integral. After one rewrite it is an **expectation**, and Jensen has something to eat.

### Definitions

$$
\mathbb{E}_Q[g(X)]=\int q(x)\,g(x)\,dx.
$$

The $f$-divergence **is** $\mathbb{E}_Q[f(R)]$. The visiting schedule is $q$, not $p$. Swap the names and you get a different number (that is why KL is not symmetric).

### Micro

If $g\equiv 1$, $\mathbb{E}_Q[1]=1$.  
If $g=R=p/q$, $\mathbb{E}_Q[R]=\int q\cdot(p/q)\,dx=\int p=1$.  
That cancellation is the whole trick of Topic 3.

### Analogy

$q$ is how often you visit each street; $g$ is what you measure there; the expectation is the city-wide average. Changing the visiting schedule changes the average even if the measurements stay put.

### Notice

Forward KL, after $f(u)=u\log u$, *cancels $q$* and becomes an expectation under **$P$**. Reverse KL keeps the $q$-weight. Same two hills, two different commutes.

### Mini-check

1. Why does $\mathbb{E}_Q[p/q]$ collapse to $1$?  
2. Under which law is $D_f$ an expectation — $P$ or $Q$?

---

## 6. Four tickets for a metric

<a id="p6-metric"></a>

### Purpose

The advertised takedown: **KL is not a distance**. “Distance / metric” is a four-ticket word. Missing any one ticket already kills it.

### Definitions

A **distance / metric** $d$ must have:

1. **Non-negativity** $d(a,b)\ge 0$
2. **Identity of indiscernibles** $d(a,b)=0$ iff $a=b$
3. **Symmetry** $d(a,b)=d(b,a)$
4. **Triangle** $d(a,c)\le d(a,b)+d(b,c)$

A **divergence** usually keeps (1)–(2) and drops (3) and/or (4). That is not a slur. It is a classification.

### Micro

Walking uphill A→B is not the same length as B→A. Useful. Not a ruler.  
A “shortcut” that is *longer* than going via a third town fails (4).

### Analogy

A highway map that reads two different mileages for the same pair of cities, or that says the direct road is longer than a detour, is not a mileage table. You can still use it as a *surprise score*. That is KL.

### Notice

KL already has (1) and (2) from the $f$-divergence theorems. This hour only has to *break* (3) and (4), with numbers. TV is assigned as the one that *keeps* all four.

### Mini-check

1. If $d(A,B)\neq d(B,A)$, how many tickets remain?  
2. Does failing *one* ticket already kill “metric”?

---

## 7. A two-point Bernoulli

<a id="p7-bern"></a>

### Purpose

Every number on the last two boards is a two-pile coin. If you cannot write a two-term KL by hand, the $0.368$ vs $0.511$ takedown is just a rumor.

### Definitions

A **Bernoulli** here is a coin with two piles of mass that add to $1$. Example: $P=(0.9,0.1)$ means $P(\text{heads})=0.9$.

KL on two-point laws is a **sum**, not an integral:

$$
D_{\mathrm{KL}}(P\|Q)=0.9\log\frac{0.9}{0.5}+0.1\log\frac{0.1}{0.5}.
$$

### Micro

That number is $\approx 0.368$. Swap $P$ and $Q$ and you get $\approx 0.511$.  
A third coin $R=(0.9,0.1)$ opposite $p=(0.1,0.9)$ gives $D_{\mathrm{KL}}(p,r)\approx 1.758$.

Use **natural log** or any base $>1$ — the inequalities $0.368\neq 0.511$ and $1.758>0.879$ do not care. He just writes $\log$.

### Analogy

Two jars: $90$ black / $10$ white versus $50/50$. “How surprised is jar A by jar B’s mix” is not the same number as the reverse. A third jar that swaps the $90/10$ piles is a *far* trip under that surprise score, even though a fair jar sits in the middle.

### Notice

He first mumbled $P$ as $(0.1,\ldots)$ then wrote the $0.9\log(0.9/0.5)$ line. Trust the **formula**, not the first mumble.

### Mini-check

1. Write $D_{\mathrm{KL}}(Q\|P)$ as a two-term sum for $P=(0.9,0.1)$, $Q=(0.5,0.5)$.  
2. Why is this a *sum*, not an integral?

---

## 8. Almost surely

<a id="p8-as"></a>

### Purpose

Jensen’s equality does not say “$R$ is literally constant at every $x$.” It says **$Q$-almost surely**. Without that phrase, Topic 4 sounds sloppier than it is.

### Definitions

**$Q$-almost surely** means: except maybe on a set $Q$ gives mass **zero**. Those exceptions do not change integrals against $q$.

If $p/q=1$ $Q$-a.s., then $p=q$ **almost everywhere** (the two hills agree wherever $q$ cares).

### Micro

Suppose $p$ and $q$ disagree only at one point $x_0$ (or on a set of Lebesgue measure zero). Then $\int q\,f(p/q)$ never sees that disagreement. For this integral they are the same pair.

### Analogy

Two maps that disagree only on a street nobody lives on. For city planning they are the same map.

### Notice

This is also why $P\ll Q$ is the right license: $P$ is forbidden from hiding mass on streets $Q$ ignores. If it did, “almost surely under $Q$” would miss a mountain.

### Mini-check

1. If $p$ and $q$ differ only where $q=0$, does $Q$ notice?  
2. How does this license “$p=q$ almost everywhere”?

---

**Second teachers (names only here).** Lilian Weng, StatQuest, Grosse, Hiroaki Hayashi, NannyML, Seeing Theory. The actual pointers live at the end of [NOTES.md](./NOTES.md#external-references) — a handful of well-known items, not a link dump.

Ready → [NOTES.md](./NOTES.md).  
Quiz: [quiz.html](./quiz.html).
