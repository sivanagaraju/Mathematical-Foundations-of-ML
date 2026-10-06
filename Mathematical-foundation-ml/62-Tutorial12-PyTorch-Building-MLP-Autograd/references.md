# Curated References & Further Reading — Tutorial 12 : Pytorch - Building MLP and Auto Grad

A curated, authoritative reference hub connecting PyTorch neural network construction, `nn.Module` subclassing contracts, dynamic computational tapes, reverse-mode automatic differentiation, and Directed Acyclic Graph (DAG) mechanics to seminal literature, official engineering specifications, and sibling curriculum modules.

---

## 1. Curriculum & Prerequisite Bridges

- [Lecture 41: Neural Networks and the Universal Approximation Theorem](../54-Lec41-Neural-Networks-UAT/)
  — **Why Read This:** Establishes the mathematical theory of Multi-Layer Perceptrons as compositional hypothesis classes and proves Cybenko's theorem on the density of single-hidden-layer networks.
- [Lecture 42: Empirical Risk Minimization and Error Backpropagation](../55-Lec42-ERM-Neural-Networks-Backpropagation/)
  — **Why Read This:** Provides the complete mathematical derivation of reverse-mode error backpropagation and the multivariable chain rule implemented by PyTorch's `autograd` engine.
- [Tutorial 11: PyTorch - Tensors and Data Loaders](../61-Tutorial11-PyTorch-Tensors-DataLoaders/)
  — **Why Read This:** Direct predecessor tutorial establishing tensor construction, strided memory layouts, GPU device migration via `.to(device)`, and batch collation pipelines feeding into model forward passes.
- [Tutorial 13: PyTorch - Training the Model](../63-Tutorial13-PyTorch-Training-the-Model/)
  — **Why Read This:** Direct sequel tutorial taking the model architectures and gradient vectors produced here into complete epoch optimization loops, loss backward stepping, and learning rate scheduling.
- [Lecture 13: Minimization of KL Divergence](../14-Lec13-Minimization-of-KL/)
  — **Why Read This:** Derives the statistical connection between maximum likelihood estimation, cross-entropy minimization, and why Softmax logit normalization represents optimal empirical risk minimization.
- [MathsTerms: Chain Rule and Backpropagation](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/04-Chain_Rule_and_Backpropagation.md)
  — **Why Read This:** Dedicated mathematical dossier with complete scalar, vector, and matrix calculus derivations of recursive backward sensitivities on computational graphs.
- [MathsTerms: Activation Functions](../../MathsTerms/02-Multivariate-Calculus-and-Optimization/05-Activation_Functions.md)
  — **Why Read This:** Detailed mathematical analysis of ReLU, Sigmoid, and Tanh activation gates, subgradients, piecewise linearity, and vanishing gradient mitigation.

---

## 2. Seminal Papers & Framework Foundations

- **Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019).** *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* Advances in Neural Information Processing Systems (NeurIPS 2019), 32, 8026–8037.  
  [https://arxiv.org/abs/1912.01703](https://arxiv.org/abs/1912.01703)  
  — **Why Read This:** The foundational architectural treatise explaining PyTorch's tape-based dynamic automatic differentiation engine, `torch.autograd`, and the object-oriented design of `torch.nn.Module`.
- **Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., ... & Lerer, A. (2017).** *Automatic differentiation in PyTorch.* NIPS 2017 Autodiff Workshop.  
  [https://openreview.net/forum?id=BJJsrmfCZ](https://openreview.net/forum?id=BJJsrmfCZ)  
  — **Why Read This:** Early conference paper detailing the tape-based design of PyTorch's autograd system, contrasting tape-based reverse-mode differentiation with static graph tracing.
- **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** *Learning representations by back-propagating errors.* Nature, 323(6088), 533–536.  
  [https://doi.org/10.1038/323533a0](https://doi.org/10.1038/323533a0)  
  — **Why Read This:** The seminal historical publication that popularized error backpropagation for training multi-layer neural network representations.
- **Griewank, A., & Walther, A. (2008).** *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation (2nd ed.).* SIAM.  
  [https://doi.org/10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)  
  — **Why Read This:** The definitive mathematical textbook on automatic differentiation, proving that reverse-mode algorithmic differentiation evaluates gradients in $\mathcal{O}(1)$ multiples of forward evaluation time.
- **Cybenko, G. (1989).** *Approximation by superpositions of a sigmoidal function.* Mathematics of Control, Signals and Systems, 2(4), 303–314.  
  [https://doi.org/10.1007/BF02551274](https://doi.org/10.1007/BF02551274)  
  — **Why Read This:** Formally establishes the Universal Approximation Theorem for feed-forward neural networks analyzed in Topic 1.

---

## 3. Authoritative Textbooks & Course Modules

- **Stevens, E., Antiga, L., & Viehmann, T. (2020).** *Deep Learning with PyTorch.* Manning Publications, Chapter 5: "The mechanics of learning" and Chapter 6: "Using a neural network to fit the data."  
  [https://pytorch.org/deep-learning-with-pytorch](https://pytorch.org/deep-learning-with-pytorch)  
  — **Why Read This:** Superb practical and conceptual treatment of autograd graphs, leaf versus non-leaf tensors, `loss.backward()`, and subclassing `nn.Module`.
- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press, Chapter 6: "Deep Feedforward Networks."  
  [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)  
  — **Why Read This:** Standard academic reference for the architecture of MLPs, choice of hidden units, cost functions, and computational graphs for backpropagation.
- **Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023).** *Dive into Deep Learning.* Cambridge University Press, Chapter 5: "Builders' Guide" and Chapter 2.5: "Automatic Differentiation."  
  [https://d2l.ai/chapter_builders-guide/](https://d2l.ai/chapter_builders-guide/)  
  — **Why Read This:** Detailed visual explanations of module construction, parameter inspection, execution hooks, and autograd tape execution.

---

## 4. Industry Technical Implementation Guides & Documentation

- **PyTorch Documentation: Automatic Differentiation Mechanics (`torch.autograd`).**  
  [https://pytorch.org/docs/stable/notes/autograd.html](https://pytorch.org/docs/stable/notes/autograd.html)  
  — **Why Read This:** Official PyTorch guide explaining forward and backward passes, the `grad_fn` computation graph, in-place operation safety checks, and memory management.
- **PyTorch Documentation: `torch.nn.Module` Specification & Parameter Hooks.**  
  [https://pytorch.org/docs/stable/generated/torch.nn.Module.html](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)  
  — **Why Read This:** Comprehensive API documentation detailing constructor contracts, parameter registration, buffer management, and forward hook interception.
- **Yang, E. (2021).** *PyTorch Autograd Explained (ezyang's blog).*  
  [http://blog.ezyang.com/2020/11/pytorch-distributed-autograd/](http://blog.ezyang.com/2020/11/pytorch-distributed-autograd/)  
  — **Why Read This:** Under-the-hood analysis by a core PyTorch developer explaining how the C++ `Node` and `Edge` graph representations orchestrate reverse-mode Vector-Jacobian Products.

---

## 5. Interactive Visualizations & Debugging Tools

- **Netron: Interactive Neural Network Visualizer.**  
  [https://netron.app/](https://netron.app/)  
  — **Why Read This:** Open-source visualization application that ingests PyTorch and ONNX models, rendering interactive layer graphs, tensor dimensions, and parameter hierarchies.
- **PyTorchViz: Visualizing PyTorch Execution Graphs and Autograd DAGs.**  
  [https://github.com/szagoruyko/pytorchviz](https://github.com/szagoruyko/pytorchviz)  
  — **Why Read This:** Graphviz-based tool that renders the dynamic Directed Acyclic Graph produced by `torch.autograd`, showing leaf parameters, intermediate tensors, and `grad_fn` nodes.
