# Mathematical Foundations — Study Notes

Packages for NPTEL / IISc lectures (generative AI math foundations and related).

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

## Study one package

1. `PREREQUISITES.md`  
2. `NOTES.md` **architecture** Executive Summary first  
3. Topics in order  
4. `quiz.html` (browser)  
