# Mathematical Foundations — Study Notes

Packages for NPTEL / IISc lectures (generative AI math foundations and related), plus the IIT Madras BS playlist catalog.

## NPTEL — Mathematical Foundations of Machine Learning

Parent course of this repo. Prof. Prathosh A P · IISc · **106108841** · 89 videos · ~46.6 h.

| File | What it is |
|------|------------|
| [`NOTES.md`](./NOTES.md) | Course map + full playlist catalog (title, link, summary, existing packages) · [YouTube playlist](https://www.youtube.com/playlist?list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu) |

Start at [`NOTES.md`](./NOTES.md). Numbered folders below (`02-Lec01-…`) are per-lecture study packages for a subset of that playlist.

## NPTEL — Mathematical Foundations of Generative AI

Sequel to the MFML course. Same instructor. Course **106108004**. Playlist still growing (21 videos uploaded; YouTube lists newest first).

| Folder | What it is |
|--------|------------|
| [`Mathematical-Foundation-for-GenerativeAI/NOTES.md`](./Mathematical-Foundation-for-GenerativeAI/NOTES.md) | Course map + full playlist catalog (learning order, links, summaries, existing packages) · [YouTube playlist](https://www.youtube.com/playlist?list=PLgMDNELGJ1CaWZJn3tyRPI8JDrMQ_RqWK) |

Start at that [`NOTES.md`](./Mathematical-Foundation-for-GenerativeAI/NOTES.md). Numbered folders inside it (`14-Lec01-…`) are per-lecture study packages.

## IIT Madras BS — Mathematical Foundations of Generative AI

Different recording from the NPTEL folder above. Same instructor (Prof. Prathosh A P). Course **BSDA5002**. 73 videos.

| Folder | What it is |
|--------|------------|
| [`IITM-BS-Mathematical-Foundations-of-Generative-AI/`](./IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md) | Course map + catalog · packages include [W1_L2](./IITM-BS-Mathematical-Foundations-of-Generative-AI/01-W1-L2-Introduction-Problem-Setting/), [W1_T4 model building](./IITM-BS-Mathematical-Foundations-of-Generative-AI/06-W1-T4-Introduction-to-PyTorch-Model-Building/), [W2_T5 vanilla GAN](./IITM-BS-Mathematical-Foundations-of-Generative-AI/10-W2-T5-Implementation-of-GAN/), [W3_L8 classifier sampler](./IITM-BS-Mathematical-Foundations-of-Generative-AI/11-W3-L8-GANs-as-Classifier-Guided-Generative-Sampler/) · [playlist](https://www.youtube.com/playlist?list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu) |

Start at [`NOTES.md`](./IITM-BS-Mathematical-Foundations-of-Generative-AI/NOTES.md) (catalog). Lecture packages: [`01-W1-L2-Introduction-Problem-Setting/`](./IITM-BS-Mathematical-Foundations-of-Generative-AI/01-W1-L2-Introduction-Problem-Setting/), [`02-W1-L3-F-Divergence/`](./IITM-BS-Mathematical-Foundations-of-Generative-AI/02-W1-L3-F-Divergence/) (playlist title says F-divergence; **recording is PyTorch tensors**).

## Skill

**`/youtube-lecture-tutor`** — **only** skill for full lecture packages:

```
PREREQUISITES.md → NOTES.md (architecture Exec Summary + topics) → quiz.html
```

Law: `~/.grok/skills/youtube-lecture-tutor/`

## Lectures

| Folder | Video | Type | Notes |
|--------|-------|------|--------|
| [`02-Lec01-Overview-Function-Approximation/`](./02-Lec01-Overview-Function-Approximation/) | [Lec 01 Overview of Function Approximation](https://www.youtube.com/watch?v=G2h7nD_Stxg) | math_technical | FA pipeline |
| [`03-Lec02-Recap-Probability-Theory-Part1/`](./03-Lec02-Recap-Probability-Theory-Part1/) | [Lec 02 Recap of Probability Theory - 1, Part 1](https://www.youtube.com/watch?v=YLx3hBqt28k) | math_technical | RE→Ω→events→P |
| [`04-Lec03-Recap-Probability-Theory-Part2/`](./04-Lec03-Recap-Probability-Theory-Part2/) | [Lec 03 Recap of Probability Theory - 1, Part 2](https://www.youtube.com/watch?v=DaBw9qBpt2s) | math_technical | RV X:Ω→R^d |
| [`05-Lec04-Recap-Probability-Theory-Part3/`](./05-Lec04-Recap-Probability-Theory-Part3/) | [Lec 04 Recap of Probability Theory - 1, Part 3](https://www.youtube.com/watch?v=0R6Agp4tqSU) | math_technical | Pushforward/CDF; density trap; multi-RV preview |
| [`06-Lec05-Recap-Probability-Theory-Part2/`](./06-Lec05-Recap-Probability-Theory-Part2/) | [Lec 05 Recap of Probability Theory Part 2](https://www.youtube.com/watch?v=R69wew8RrPo) | math_technical | Joints · conditionals · margins; vector ≡ d scalars |
| [`07-Lec06-XRay-Sample-From-Distribution/`](./07-Lec06-XRay-Sample-From-Distribution/) | [Lec 06 X-Ray as Sample from Distribution](https://www.youtube.com/watch?v=bdcvsSNAHIk) | math_technical | Image∈range(X); data≠P; joint labels; dataset ~ P_{X,Y} |
| [`08-Lec07-IID-Assumption/`](./08-Lec07-IID-Assumption/) | [Lec 07 IID Assumption](https://www.youtube.com/watch?v=C83xmx80tMo) | math_technical | Identical + independent(across points); ML = estimate P |
| [`09-Lec08-Distribution-Estimation/`](./09-Lec08-Distribution-Estimation/) | [Lec 08 Distribution Estimation](https://www.youtube.com/watch?v=aYb8KG9JYsg) | math_technical | Given D estimate P; targets P(Y\|X)/P(Y)/…; disc vs gen |
| [`10-Lec09-Density-Function/`](./10-Lec09-Density-Function/) | [Lec 09 Density Function](https://www.youtube.com/watch?v=_QrezNPmxDk) | math_technical | Density p; height≠prob; Uniform height 2; estimate p |
| [`11-Lec10-Challenges-of-ML/`](./11-Lec10-Challenges-of-ML/) | [Lec 10 Challenge With ML](https://www.youtube.com/watch?v=767MLwniPKE) | math_technical | Unknown p; recipe p_θ→d→argmin; model≠algo; ERM next |
| [`12-Lec11-Entropy/`](./12-Lec11-Entropy/) | [Lec 11 Entropy](https://www.youtube.com/watch?v=P6wjLz4dRTs) | math_technical | Surprisal −log P; entropy H=−∑p log p; need d |
| [`13-Lec12-KL-Divergence/`](./13-Lec12-KL-Divergence/) | [Lec 12 KL Divergence](https://www.youtube.com/watch?v=ihkGbIdbbxc) | math_technical | Cross-entropy; KL=CE−H; asymmetric divergence |
| [`14-Lec13-Minimization-of-KL/`](./14-Lec13-Minimization-of-KL/) | [Lec 13 Minimization of KL Divergence](https://www.youtube.com/watch?v=Ij4p5hLbfo4) | math_technical | min KL → drop H → LLN → MLE ≡ min-KL estimator |
| [`Mathematical-Foundation-for-GenerativeAI/14-Lec01-MFGAI-Introduction/`](./Mathematical-Foundation-for-GenerativeAI/14-Lec01-MFGAI-Introduction/) | [Lec 01 Introduction (MF Generative AI)](https://www.youtube.com/watch?v=H05WDy9Mngk) | math_technical | RE→Ω→P→RV→estimate P_X; GenAI roadmap |
| [`Mathematical-Foundation-for-GenerativeAI/15-Lec02-Generative-Models-Problem-Formulation/`](./Mathematical-Foundation-for-GenerativeAI/15-Lec02-Generative-Models-Problem-Formulation/) | [Lec 02 Generative Models: Problem Formulation](https://www.youtube.com/watch?v=GKfv4l6r7hQ) | math_technical | data∈R^d; estimate p_x + sample; p_θ + min d |
| [`Mathematical-Foundation-for-GenerativeAI/16-Tutorial02-Introduction-to-NumPy/`](./Mathematical-Foundation-for-GenerativeAI/16-Tutorial02-Introduction-to-NumPy/) | [Tutorial 2: Introduction to NumPy](https://www.youtube.com/watch?v=E79ld44pfGM) | code_tutorial | arrays→matmul→ReLU/softmax→conv/RNN→logreg |
| [`Mathematical-Foundation-for-GenerativeAI/17-Tutorial03-PyTorch-Basics/`](./Mathematical-Foundation-for-GenerativeAI/17-Tutorial03-PyTorch-Basics/) | [Tutorial 3: PyTorch Basics](https://www.youtube.com/watch?v=SEtu7Eef5ps) | code_tutorial | tensors·device·autograd·Module·DataLoader·MLP train |
| [`Mathematical-Foundation-for-GenerativeAI/18-Tutorial04-CNNs-PyTorch/`](./Mathematical-Foundation-for-GenerativeAI/18-Tutorial04-CNNs-PyTorch/) | [Tutorial 4: CNNs using PyTorch](https://www.youtube.com/watch?v=BhnGtsMwUCU) | code_tutorial | Conv2d·MaxPool·SimpleCNN·MNIST train/eval |
| [`Mathematical-Foundation-for-GenerativeAI/19-Tutorial05-RNNs-PyTorch/`](./Mathematical-Foundation-for-GenerativeAI/19-Tutorial05-RNNs-PyTorch/) | [Tutorial 5: RNNs using PyTorch](https://www.youtube.com/watch?v=k6zF2NsvVrk) | code_tutorial | seq tensors·RNN/LSTM/GRU·save/load·train helpers |
| [`Mathematical-Foundation-for-GenerativeAI/20-Tutorial06-Transfer-Learning-PyTorch/`](./Mathematical-Foundation-for-GenerativeAI/20-Tutorial06-Transfer-Learning-PyTorch/) | [Tutorial 6: Transfer Learning with PyTorch](https://www.youtube.com/watch?v=ETJG9mmeL5k) | code_tutorial | pretrained AlexNet/VGG/ResNet·head swap·MRI fine-tune |
| [`Mathematical-Foundation-for-GenerativeAI/21-Tutorial07-Review-Basic-Probability-1/`](./Mathematical-Foundation-for-GenerativeAI/21-Tutorial07-Review-Basic-Probability-1/) | [Tutorial 7: Review of Basic Probability 1](https://www.youtube.com/watch?v=owlWCCgYx50) | math_technical | triplet·cond/Bayes·independence flavors·RV/CDF·discrete PMF families |
| [`Mathematical-Foundation-for-GenerativeAI/22-Tutorial08-Review-Basic-Probability-2/`](./Mathematical-Foundation-for-GenerativeAI/22-Tutorial08-Review-Basic-Probability-2/) | [Tutorial 8: Review of Basic Probability 2](https://www.youtube.com/watch?v=pQIbfyjSnFk) | mixed | CRV/PDF·CoV·E/LOTUS/Var·Markov/Chebyshev/Jensen·numpy samples |
| [`Mathematical-Foundation-for-GenerativeAI/23-Tutorial09-Review-Basic-Probability-3/`](./Mathematical-Foundation-for-GenerativeAI/23-Tutorial09-Review-Basic-Probability-3/) | [Tutorial 9: Review of Basic Probability 3](https://www.youtube.com/watch?v=eDSb3yObtB8) | math_technical | joints·marginals·conditionals·mixed/GMM·IID·Jacobian |
| [`Mathematical-Foundation-for-GenerativeAI/24-Tutorial10-Review-Machine-Learning-1/`](./Mathematical-Foundation-for-GenerativeAI/24-Tutorial10-Review-Machine-Learning-1/) | [Tutorial 10: Review of Machine Learning 1](https://www.youtube.com/watch?v=wjSKM1xFoSU) | math_technical | sign-censored Normal MLE · Φ⁻¹ · two-exp EM · Q · closed M-step |
| [`Mathematical-Foundation-for-GenerativeAI/25-Lec03-f-Divergence-Examples/`](./Mathematical-Foundation-for-GenerativeAI/25-Lec03-f-Divergence-Examples/) | [Lec 03: f-Divergence and Examples](https://www.youtube.com/watch?v=LR9UQXY_IU8) | math_technical | estimate+sample · G_θ · f-div (not a metric) · KL/rev/JSD · modes vs junk |
| [`Mathematical-Foundation-for-GenerativeAI/26-Tutorial11-f-Divergence-Examples/`](./Mathematical-Foundation-for-GenerativeAI/26-Tutorial11-f-Divergence-Examples/) | [Tutorial 11: f-Divergence and Examples](https://www.youtube.com/watch?v=GjxuVZeMSfE) | math_technical | P≪Q · Jensen proofs · KL/−log/TV/JSD · KL fails symmetry & triangle (0.368/1.758) |
| [`Mathematical-Foundation-for-GenerativeAI/27-Lec04-Variational-Divergence-Minimization/`](./Mathematical-Foundation-for-GenerativeAI/27-Lec04-Variational-Divergence-Minimization/) | [Lec 04 Variational Divergence Minimization (VDM)](https://www.youtube.com/watch?v=4vtL3NhCkgg) | math_technical | two clouds · f* unzip · T(x) lower bound · min_θ max_w saddle |
| [`Mathematical-Foundation-for-GenerativeAI/28-Lec05-Generative-Adversarial-Networks/`](./Mathematical-Foundation-for-GenerativeAI/28-Lec05-Generative-Adversarial-Networks/) | [Lec 05 Generative Adversarial Networks (GANs)](https://www.youtube.com/watch?v=5uqga82bDNA) | math_technical | GAN = one f of VDM · not JSD · alternate freeze · cGAN concat · StyleGAN |
| [`Mathematical-Foundation-for-GenerativeAI/29-Tutorial12-Implementations-Vanilla-GAN-DCGAN-cGAN/`](./Mathematical-Foundation-for-GenerativeAI/29-Tutorial12-Implementations-Vanilla-GAN-DCGAN-cGAN/) | [Tutorial 12: Vanilla GAN, DCGAN, Conditional GAN](https://www.youtube.com/watch?v=dBcURX7GrwE) | code_tutorial | MLP/cGAN/DCGAN · detach · non-sat G · FID 92.93/104/21.5 |
| [`Mathematical-Foundation-for-GenerativeAI/30-Lec18-Wasserstein-GAN/`](./Mathematical-Foundation-for-GenerativeAI/30-Lec18-Wasserstein-GAN/) | [Lec 18 Wasserstein GAN (WGAN)](https://www.youtube.com/watch?v=1neDqqgaXhE) | math_technical | OT / earth-mover · manifold hyp · KR dual · \|W\|_2=1 · more stable than naive GAN |
| [`Mathematical-Foundation-for-GenerativeAI/31-Lec19-Inversion-GANs-FID/`](./Mathematical-Foundation-for-GenerativeAI/31-Lec19-Inversion-GANs-FID/) | [Lec 19 Inversion with GANs and FID](https://www.youtube.com/watch?v=zw2DUzD0TLE) | math_technical | BiGAN/ALI tuple D · invert E then G · FID = W2 of Inception Gaussians |
| [`Mathematical-Foundation-for-GenerativeAI/32-Lec20-Latent-Variable-Models-VAE/`](./Mathematical-Foundation-for-GenerativeAI/32-Lec20-Latent-Variable-Models-VAE/) | [Lec 20 LVM and VAE intro](https://www.youtube.com/watch?v=4djE9goJtKs) | math_technical | p(x)=∫p(x,z) · ELBO · EM vs AEVB · reparam next class |

## Study one package

1. `PREREQUISITES.md`  
2. `NOTES.md` **architecture** Executive Summary first  
3. Topics in order  
4. `quiz.html` (browser)  
