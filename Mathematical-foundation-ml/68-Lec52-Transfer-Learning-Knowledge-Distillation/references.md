# References & Learning Bridges: Lecture 52 (Transfer Learning & Knowledge Distillation)

A curated repository of foundational papers, curriculum bridges, textbooks, industry implementations, and interactive visualizers for Transfer Learning, Representation Tapping, Multitask Parameter Sharing, and Knowledge Distillation.

---

## Curriculum & Prerequisite Bridges

- [Lecture 12: Empirical Risk Minimization](../13-Lec12-Empirical-Risk-Minimisation/NOTES.md)  
  *Why Read This:* Establishes the mathematical foundation of Empirical Risk Minimization (ERM) that both pre-trained backbones and student distillation objectives optimize over sample data.
- [Lecture 13: Minimization of KL Divergence](../14-Lec13-Minimation-of-KL/NOTES.md)  
  *Why Read This:* Provides the formal derivation and information-theoretic properties of Kullback-Leibler divergence used in soft-target knowledge distillation.
- [Lecture 18: Logistic Regression Part 3 (Cross-Entropy Loss)](../19-Lec18-LogisticRegressionPart3-Cross-Entropy-Loss/NOTES.md)  
  *Why Read This:* Formulates categorical negative log-likelihood and softmax that temperature scaling dilates to reveal dark knowledge.
- [Lecture 50: Multi-Head Attention and Transformer Architecture](../66-Lec50-Multi-Head-Attention-Transformer/NOTES.md)  
  *Why Read This:* Details the transformer backbone architectures tapped for BERT and foundation model transfer learning.
- [Lecture 51: Positional Embeddings](../67-Lec51-Positional-Embeddings/NOTES.md)  
  *Why Read This:* Explains how sequence coordinates are injected into backbones whose intermediate representations are transferred downstream.
- [Lecture 53: SGD, RMSProp, and Adam Optimizers](../69-Lec53-SGD-RMSprop-Adam-Optimizers/NOTES.md)  
  *Why Read This:* Details first-order adaptive optimization algorithms used during full fine-tuning and student distillation.

---

## Foundational & Seminal Papers

- [Hinton, Vinyals, & Dean (2015) — Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)  
  *Why Read This:* The seminal paper establishing dark knowledge, temperature-scaled softmax, and the combined cross-entropy / KL divergence distillation objective with the $\tau^2$ normalization factor.
- [Buciluǎ, Caruana, & Niculescu-Mizil (2006) — Model Compression](https://dl.acm.org/doi/10.1145/1150402.1150464)  
  *Why Read This:* The pioneering paper demonstrating that complex ensembles can be compressed into a single compact neural network by training on synthetic ensemble pseudo-labels.
- [Ba & Caruana (2014) — Do Deep Nets Really Need to be Deep?](https://arxiv.org/abs/1312.6184)  
  *Why Read This:* Proves empirically that shallow neural networks can match the accuracy of deep networks when trained on the logits of deep teacher models.
- [Caruana (1997) — Multitask Learning](https://link.springer.com/article/10.1023/A:1007379606734)  
  *Why Read This:* The foundational treatise on inductive transfer and hard parameter sharing, proving that shared representations regularize generalization across tasks.
- [He et al. (2016) — Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)  
  *Why Read This:* Documents pre-training on ImageNet and linear probing / fine-tuning on downstream PASCAL VOC and MS COCO tasks.
- [Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)  
  *Why Read This:* Demonstrates the power of self-supervised marginal pre-training (Masked LM) for universal NLP transfer learning.
- [He et al. (2022) — Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)  
  *Why Read This:* Demonstrates that masking 75% of vision patches and reconstructing pixel values estimates marginal image structure $P(X)$ yielding superior transfer backbones.
- [Gou et al. (2021) — Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)  
  *Why Read This:* Comprehensive survey categorizing response-based distillation, feature-based representation matching, and relation-based student distillation.
- [Heo et al. (2019) — A Comprehensive Overhaul of Feature Distillation](https://arxiv.org/abs/1904.01866)  
  *Why Read This:* Explores intermediate layer activation tapping, margin ReLU transforms, and partial distance metrics for teacher-student representation alignment.
- [Furlanello et al. (2018) — Born-Again Neural Networks](https://arxiv.org/abs/1805.04770)  
  *Why Read This:* Discovers self-distillation, showing that distilling a teacher network into a student of identical architecture and capacity achieves superior accuracy.

---

## Textbooks & Video Lectures

- [Goodfellow, Bengio, & Courville (2016) — Deep Learning (Chapter 15: Representation Learning)](https://www.deeplearningbook.org)  
  *Why Read This:* Authoritative textbook chapter on representation learning, transfer learning mechanics, and the trade-off between conditional and marginal density modeling.
- [Bishop & Bishop (2024) — Deep Learning: Foundations and Concepts (Chapter 19: Transfer and Self-Supervised Learning)](https://www.bishopbook.com)  
  *Why Read This:* Rigorous statistical and architectural treatment of pre-training, parameter freezing, linear probing, and distillation theory.
- [Andrej Karpathy — Deep Learning Lecture: Pretraining, Fine-Tuning, and LLM Alignment](https://www.youtube.com/watch?v=zjkBMFhNj_g)  
  *Why Read This:* Visual breakdown of pretraining on internet-scale marginal text corpora followed by downstream supervised fine-tuning.
- [Yann LeCun — Self-Supervised Learning: The Dark Matter of Intelligence](https://www.youtube.com/watch?v=7I0Qt7GALVk)  
  *Why Read This:* Seminal keynote on why supervised learning $P(Y \mid X)$ provides too little feedback per sample and why self-supervised marginal modeling $P(X)$ is fundamental.

---

## Industry & Implementation Guides

- [PyTorch Tutorials — Transfer Learning for Computer Vision](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)  
  *Why Read This:* Official PyTorch guide implementing feature extraction via `param.requires_grad = False` and full fine-tuning with differential optimizers.
- [Hugging Face Documentation — DistilBERT: A Distilled Version of BERT](https://huggingface.co/docs/transformers/model_doc/distilbert)  
  *Why Read This:* Industry case study detailing how DistilBERT retains 97% of BERT's language understanding while being 40% smaller and 60% faster via triple-loss distillation.
