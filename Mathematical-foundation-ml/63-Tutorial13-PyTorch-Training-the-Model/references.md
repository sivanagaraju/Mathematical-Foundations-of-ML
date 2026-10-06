# References & Further Reading — Tutorial 13 : Pytorch - Training the Model

A curated, annotated bibliography connecting empirical risk minimization, stochastic approximation algorithms, adaptive moment optimizers, numerical cross-entropy computation, and production model serialization.

---

## 1. Foundational Papers & Historical Literature

1. **Robbins, H., & Monro, S. (1951).** *A Stochastic Approximation Method.* The Annals of Mathematical Statistics, 22(3), 400–407.  
   — **Why Read This:** The seminal mathematical formulation of stochastic approximation proving almost-sure convergence of iterative parameter sequences under noisy gradient evaluations, underpinning modern mini-batch SGD.

2. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** *Learning representations by back-propagating errors.* Nature, 323(6088), 533–536.  
   — **Why Read This:** Landmark paper introducing generalized backpropagation across multi-layer networks, establishing the exact reverse-mode error propagation equations evaluated by `loss.backward()`.

3. **Kingma, D. P., & Ba, J. (2014).** *Adam: A Method for Stochastic Optimization.* arXiv preprint arXiv:1412.6980.  
   — **Why Read This:** Introduced the Adam optimization algorithm combining first-moment momentum with second-moment coordinate variance rescaling, including bias-correction factors for non-stationary objectives.

4. **Tieleman, T., & Hinton, G. (2012).** *Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude.* COURSERA: Neural Networks for Machine Learning.  
   — **Why Read This:** The historical origin of RMSprop, providing intuition on exponential moving average gradient scaling in neural network optimization landscapes.

5. **Duchi, J., Hazan, E., & Singer, Y. (2011).** *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.* Journal of Machine Learning Research, 12, 2121–2159.  
   — **Why Read This:** Introduced AdaGrad, establishing theoretical convergence guarantees for coordinate-wise learning rate adaptations based on historical squared gradient accumulation.

6. **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).** *Dropout: A simple way to prevent neural networks from overfitting.* Journal of Machine Learning Research, 15(1), 1929–1958.  
   — **Why Read This:** Explains the mathematical mechanism behind `model.train()` and `model.eval()`, detailing how Bernoulli mask sampling regularizes feature co-adaptation during training while expecting expectation scaling at inference.

---

## 2. Textbooks & Monographs

7. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press.  
   — **Why Read This:** Chapters 6, 7, and 8 provide rigorous derivations of empirical risk minimization, stochastic mini-batching, adaptive optimizers, and generalization bounds.

8. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning.* Springer.  
   — **Why Read This:** Chapter 4 derives multiclass cross-entropy loss from maximum likelihood on categorical distributions and connects decision theory to argmax label assignment.

9. **Boyd, S., & Vandenberghe, L. (2004).** *Convex Optimization.* Cambridge University Press.  
   — **Why Read This:** Chapters 9 and 10 analyze gradient descent convergence rates, condition numbers, and the geometry of sub-optimal convergence in ill-conditioned loss valleys.

10. **Shalev-Shwartz, S., & Ben-David, S. (2014).** *Understanding Machine Learning: From Theory to Algorithms.* Cambridge University Press.  
    — **Why Read This:** Chapters 2 and 3 formulate the PAC learning framework, uniform convergence, and the formal justification for Empirical Risk Minimization.

---

## 3. Official Documentation & Framework Standards

11. **PyTorch Core Team. (2024).** *PyTorch Optimization Documentation (`torch.optim`).* https://pytorch.org/docs/stable/optim.html  
    — **Why Read This:** Official API specification covering optimizer parameter groups, learning rate decay, and the computational efficiency of `optimizer.zero_grad(set_to_none=True)`.

12. **PyTorch Core Team. (2024).** *Saving and Loading Models in PyTorch.* https://pytorch.org/tutorials/beginner/saving_loading_models.html  
    — **Why Read This:** Definitive guide detailing why serializing `model.state_dict()` via `torch.save` is mandatory for production over whole-model pickling.

13. **PyTorch Core Team. (2024).** *CrossEntropyLoss API Reference (`torch.nn.CrossEntropyLoss`).* https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html  
    — **Why Read This:** Documents the fused LogSoftmax + NLLLoss formulation and specifies class weighting, ignore indices, and label smoothing parameters.

14. **PyTorch Core Team. (2024).** *Inference Mode & Context Managers (`torch.no_grad` vs `torch.inference_mode`).* https://pytorch.org/docs/stable/generated/torch.no_grad.html  
    — **Why Read This:** Technical breakdown of how thread-local flags disable DAG record-keeping to halve memory allocation during validation passes.

---

## 4. Curated Tutorials & Technical Deep Dives

15. **Karpathy, A. (2022).** *A Recipe for Training Neural Networks.* Personal Blog. https://karpathy.github.io/2019/04/25/recipe/  
    — **Why Read This:** Pragmatic industry distillation of debugging protocols: overfit a single mini-batch, verify initial loss values ($-\log(1/C)$), and monitor training versus validation trajectories.

16. **Ruder, S. (2016).** *An overview of gradient descent optimization algorithms.* arXiv preprint arXiv:1609.04747.  
    — **Why Read This:** Highly visual and comprehensive taxonomy comparing SGD, Momentum, Nesterov, AdaGrad, RMSprop, and Adam trajectories across complex loss surfaces.

17. **Grokking PyTorch Team. (2023).** *Understanding PyTorch Gradient Accumulation and Memory Management.* PyTorch Developer Discussions.  
    — **Why Read This:** Detailed analysis explaining why PyTorch accumulates gradients by default and detailing how `.item()` prevents DAG reference chaining in metric logging.

---

## 5. Interactive Visualizers & Profiling Tools

18. **Loss Landscape Interactive Visualizer.** https://losslandscape.com/  
    — **Why Read This:** Interactive WebGL 3D visualizer showing non-convex optimization paths, saddle points, and curvature differences across SGD, Momentum, and Adam.

19. **Weights & Biases (W&B) Training Dashboard.** https://wandb.ai/  
    — **Why Read This:** Standard industry platform for visualizing multi-epoch loss curves, gradient norms, and hyperparameter sweeps in real time.

---

## 6. Curriculum & Prerequisite Bridges

20. **Prathosh, A. P. (2024).** *Tutorial 11: PyTorch - Tensors and Data Loaders.* Mathematical Foundations of Machine Learning. [`../61-Tutorial11-PyTorch-Tensors-DataLoaders/`](../61-Tutorial11-PyTorch-Tensors-DataLoaders/)  
    — **Why Read This:** Provides the tensor memory layouts, stride semantics, and custom `Dataset`/`DataLoader` pipelines that feed the training loops in this tutorial.

21. **Prathosh, A. P. (2024).** *Tutorial 12: PyTorch - Building MLP and Auto Grad.* Mathematical Foundations of Machine Learning. [`../62-Tutorial12-PyTorch-Building-MLP-Autograd/`](../62-Tutorial12-PyTorch-Building-MLP-Autograd/)  
    — **Why Read This:** Details the `nn.Module` subclassing architecture, forward pass dispatch via `__call__`, and the dynamic autograd DAG engine executed during `loss.backward()`.

22. **Prathosh, A. P. (2024).** *Lec 53: SGD, RMSProp, ADAM: Optimizers.* Mathematical Foundations of Machine Learning. [`../69-Lec53-SGD-RMSprop-Adam-Optimizers/`](../69-Lec53-SGD-RMSprop-Adam-Optimizers/)  
    — **Why Read This:** The dedicated theoretical lecture companion deriving the mathematical convergence proofs and hyperparameter sensitivities of SGD, RMSProp, and Adam.
