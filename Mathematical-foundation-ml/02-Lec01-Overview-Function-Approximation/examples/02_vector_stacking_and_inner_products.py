"""
02_vector_stacking_and_inner_products.py
========================================
Lecture 01 Simulation: Vector Stacking & Isomorphic Inner Products

Demonstrates how multi-dimensional sensor data (e.g. chest X-rays of size P x Q)
is stacked into high-dimensional Euclidean coordinate vectors in R^{PQ}, proving
that linear projections and Frobenius inner products on matrices are mathematically
isomorphic to standard vector dot products.

Mathematical Guarantee:
    <W, X>_F = Tr(W^T X) = sum_{i,j} W_{ij} X_{ij} == vec(W)^T vec(X)
    Linear operators on flattened image vectors preserve all inner products and Euclidean norms.

Verification:
    Asserts torch.allclose between matrix Frobenius inner product and vector dot product.
"""

import sys
import torch
import numpy as np


def main() -> int:
    print("=== Simulation 02: Vector Stacking & Inner Products ===")
    torch.manual_seed(42)

    # 1. Define synthetic 2D sensor image (e.g., 8x8 chest X-ray patch)
    H, W = 8, 8
    image_matrix = torch.randn(H, W, dtype=torch.float32)

    # 2. Vector stacking: flatten 2D spatial grid into 1D coordinate vector in R^{HW}
    vector_x = image_matrix.flatten()
    assert vector_x.shape == (H * W,), f"Expected shape ({H*W},), got {vector_x.shape}"

    # 3. Define 2D linear filter / receptive field weight matrix
    weight_matrix = torch.randn(H, W, dtype=torch.float32)
    weight_vector = weight_matrix.flatten()

    # 4. Compute 2D Frobenius inner product: <W, I>_F = sum_{i,j} W_{i,j} * I_{i,j}
    frobenius_inner_prod = torch.sum(weight_matrix * image_matrix)
    trace_inner_prod = torch.trace(weight_matrix.T @ image_matrix)

    # 5. Compute 1D vector dot product: w^T x
    vector_dot_prod = torch.dot(weight_vector, vector_x)

    print(f"Frobenius Inner Product: {frobenius_inner_prod.item():.6f}")
    print(f"Matrix Trace Tr(W^T I):  {trace_inner_prod.item():.6f}")
    print(f"1D Vector Dot Product:   {vector_dot_prod.item():.6f}")

    # Mathematical identity verification
    assert torch.allclose(frobenius_inner_prod, trace_inner_prod, atol=1e-5), (
        "Matrix Frobenius product must equal trace Tr(W^T I)"
    )
    assert torch.allclose(frobenius_inner_prod, vector_dot_prod, atol=1e-5), (
        "Vector dot product must identically match matrix Frobenius inner product"
    )

    # 6. Batch simulation: N = 100 images stacked into design matrix X in R^{N x D}
    N = 100
    D = H * W
    batch_images = torch.randn(N, H, W, dtype=torch.float32)
    batch_vectors = batch_images.view(N, D)

    # Multi-class linear classification projection: Z = X W^T + b
    num_classes = 2  # Binary diagnostic: healthy vs pneumonia
    classifier_W = torch.randn(num_classes, D, dtype=torch.float32)
    classifier_b = torch.randn(num_classes, dtype=torch.float32)

    logits_vectorized = batch_vectors @ classifier_W.T + classifier_b  # [N, num_classes]

    # Manual loop accumulation verifying batch vectorization
    logits_manual = torch.zeros(N, num_classes, dtype=torch.float32)
    for i in range(N):
        for c in range(num_classes):
            logits_manual[i, c] = torch.sum(classifier_W[c].view(H, W) * batch_images[i]) + classifier_b[c]

    assert torch.allclose(logits_vectorized, logits_manual, atol=1e-5), (
        "Vectorized batch matrix multiplication must match spatial filter convolutions"
    )

    print(f"Successfully verified batch projection for N={N} image vectors in R^{D}.")
    print("SUCCESS: 02_vector_stacking_and_inner_products executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
