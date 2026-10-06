# Curated References & Further Reading — Tutorial 11 : Pytorch - Tensors and Data Loaders

A curated, authoritative reference hub connecting PyTorch tensor internals, contiguous memory layouts, GPU hardware architectures, and production dataset streaming pipelines to canonical literature, official engineering specifications, and sibling curriculum modules.

---

## 1. Curriculum & Prerequisite Bridges

- [Lecture 01: Overview of Function Approximation](../02-Lec01-Overview-Function-Approximation/)
  — **Why Read This:** Establishes the core problem formulation of machine learning as function approximation under uncertainty, motivating why multidimensional arrays in code represent realizations of random variables.
- [Lecture 07: The I.I.D. Assumption](../08-Lec07-IID-Assumption/)
  — **Why Read This:** Provides the theoretical statistical foundation for why datasets must be shuffled (`DataLoader(shuffle=True)`) to draw independent and identically distributed mini-batches from the data law.
- [Lecture 41: Neural Networks and the Universal Approximation Theorem](../54-Lec41-Neural-Networks-UAT/)
  — **Why Read This:** Derives the composition of linear layers and non-linearities, providing the architectural purpose for the General Matrix Multiplications (`@`, `torch.matmul`) introduced in this tutorial.
- [Lecture 43: Local Receptive Fields and Parameter Sharing](../56-Lec43-Local-Receptive-Field-Parameter-Sharing/)
  — **Why Read This:** Connects coordinate-wise Hadamard products and patch summations to 2D convolutional operations on image tensors.
- [Lecture 47: LSTMs and GRUs](../60-Lec47-LSTMs-and-GRUs/)
  — **Why Read This:** Bridges element-wise Hadamard tensor multiplication (`*`, `torch.mul`) to coordinate-wise retention and candidate injection gates along unattenuated memory highways.
- [Tutorial 12: PyTorch - Building MLP and Autograd](../62-Tutorial12-PyTorch-Building-MLP-Autograd/)
  — **Why Read This:** The direct sequel tutorial, demonstrating how the tensors created and manipulated here are equipped with computational tape history (`requires_grad=True`) for automatic reverse-mode differentiation.

---

## 2. Seminal Papers & Framework Foundations

- **Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019).** *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* Advances in Neural Information Processing Systems (NeurIPS 2019), 32, 8026–8037.  
  [https://arxiv.org/abs/1912.01703](https://arxiv.org/abs/1912.01703)  
  — **Why Read This:** The definitive architectural publication describing PyTorch's core design philosophy: imperative eager execution, zero-overhead C++ backend (`ATen`), Pythonic tensor interfaces, and transparent CUDA accelerator dispatch.
- **Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... & Oliphant, T. E. (2020).** *Array programming with NumPy.* Nature, 585(7825), 357–362.  
  [https://doi.org/10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)  
  — **Why Read This:** Documents the foundational strided N-dimensional array abstraction upon which PyTorch's tensor memory model and broadcasting rules are built.
- **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).** *Gradient-based learning applied to document recognition.* Proceedings of the IEEE, 86(11), 2278–2324.  
  [http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)  
  — **Why Read This:** Introduces the MNIST handwritten digit benchmark dataset utilized throughout this tutorial's academic dataset pipeline demonstrations.
- **Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., ... & Zheng, X. (2016).** *TensorFlow: A system for large-scale machine learning.* 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), 265–283.  
  [https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)  
  — **Why Read This:** Provides historical and contrastive context on static symbolic dataflow graphs versus dynamic eager tensor execution models.

---

## 3. Authoritative Textbooks & Course Modules

- **Stevens, E., Antiga, L., & Viehmann, T. (2020).** *Deep Learning with PyTorch.* Manning Publications, Chapters 2 & 3: "It starts with a tensor" and "Learning mechanics."  
  [https://pytorch.org/deep-learning-with-pytorch](https://pytorch.org/deep-learning-with-pytorch)  
  — **Why Read This:** The gold-standard comprehensive guide to PyTorch tensor storage internals, strides, contiguous blocks, NumPy interoperability, and custom dataset engineering.
- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press, Chapter 2: "Linear Algebra" and Chapter 4: "Numerical Computation."  
  [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)  
  — **Why Read This:** Provides rigorous mathematical treatments of tensor operations, vector norms, broadcasting geometries, and floating-point roundoff errors.
- **Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023).** *Dive into Deep Learning.* Cambridge University Press, Chapter 2: "Preliminaries: Data Manipulation."  
  [https://d2l.ai/chapter_preliminaries/ndarray.html](https://d2l.ai/chapter_preliminaries/ndarray.html)  
  — **Why Read This:** Offers excellent visual diagrams and interactive code sandboxes illustrating multi-axis tensor indexing, memory reallocation avoidance, and hardware accelerator mapping.

---

## 4. Industry Technical Implementation Guides & Documentation

- **Yang, E. (2021).** *PyTorch Internals (ezyang's blog).*  
  [http://blog.ezyang.com/2019/05/pytorch-internals/](http://blog.ezyang.com/2019/05/pytorch-internals/)  
  — **Why Read This:** An essential deep-dive into PyTorch's internal architecture: the separation between `TensorImpl` (metadata, shape, strides) and `Storage` (raw byte pointers), and how autograd dynamically wraps storage buffers.
- **PyTorch Documentation: Tensor Attributes & Memory Layout.**  
  [https://pytorch.org/docs/stable/tensor_attributes.html](https://pytorch.org/docs/stable/tensor_attributes.html)  
  — **Why Read This:** The authoritative API specification for `torch.dtype`, `torch.device`, and `torch.layout`, detailing memory strides and byte representations across hardware backends.
- **PyTorch Documentation: Writing Custom Datasets, DataLoaders & Samplers.**  
  [https://pytorch.org/tutorials/beginner/data_loading_tutorial.html](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)  
  — **Why Read This:** The canonical implementation tutorial for subclassing `Dataset`, designing callable transformation classes, handling image I/O, and tuning multi-process workers in `DataLoader`.
- **NVIDIA Corporation (2024).** *CUDA C++ Programming Guide: Memory Hierarchy & Tensor Core Architecture.*  
  [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)  
  — **Why Read This:** Explains the physical hardware reality of GPU thread blocks, warp scheduling, shared memory, and how SIMD tensor cores execute $4 \times 4 \times 4$ GEMM operations in a single cycle.

---

## 5. Interactive Visualizers & Educational Sandboxes

- **Polo Club of Data Science (Georgia Tech).** *CNN Explainer: Interactive Visualization of Convolutional Operations.*  
  [https://poloclub.github.io/cnn-explainer/](https://poloclub.github.io/cnn-explainer/)  
  — **Why Read This:** Provides an interactive 3D browser-based visualization of 2D image tensors, showing how multi-channel tensors are sliced, multiplied via Hadamard products, and summed across receptive fields.
- **Netron: Visualizer for Neural Network, Deep Learning, and Machine Learning Models.**  
  [https://netron.app/](https://netron.app/)  
  — **Why Read This:** Allows students to upload exported computational models and visually inspect input/output tensor shapes, strides, and memory attributes across layer boundaries.
