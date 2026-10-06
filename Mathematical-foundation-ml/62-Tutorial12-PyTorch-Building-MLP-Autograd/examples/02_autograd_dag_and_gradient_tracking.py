"""
Script 02: Autograd Engine, Directed Acyclic Graphs (DAG), and Gradient Controls
================================================================================
Demonstrates and validates:
1. Leaf tensor initialization with requires_grad=True.
2. Computational graph formation and grad_fn backward operator inspection.
3. Reverse-mode automatic differentiation via loss.backward().
4. Analytical gradient validation against autograd accumulated .grad values.
5. Inference acceleration via with torch.no_grad(): context manager.
6. Subgraph decoupling via zero-copy tensor.detach().
7. Selective parameter freezing for modular transfer learning.
"""

import sys
import torch
import torch.nn as nn


def main():
    print("[1/5] Constructing Toy Linear Computational Graph...")
    # Input batch of 2 samples, 3 features
    x = torch.tensor([[1.0, 2.0, -1.0],
                      [0.5, -1.0, 2.0]], dtype=torch.float32)  # Shape: [B, D_in] = [2, 3]

    # Trainable leaf parameters (requires_grad=True)
    w = torch.tensor([[0.2, -0.1],
                      [0.4,  0.5],
                      [-0.3, 0.1]], dtype=torch.float32, requires_grad=True)  # Shape: [D_in, D_out] = [3, 2]
    b = torch.tensor([0.05, -0.05], dtype=torch.float32, requires_grad=True)  # Shape: [D_out] = [2]

    assert w.is_leaf and b.is_leaf, "w and b must be leaf nodes"
    assert w.grad_fn is None and b.grad_fn is None, "Leaf tensors must have grad_fn = None"

    # Forward Affine Projection: Z = X @ W + b
    z = torch.matmul(x, w) + b  # Shape: [B, D_out] = [2, 2]
    assert not z.is_leaf, "z must be a non-leaf node"
    assert z.grad_fn is not None, "z must possess a grad_fn"
    assert "AddBackward" in type(z.grad_fn).__name__, f"Expected AddBackward, got {type(z.grad_fn).__name__}"
    print(f"      Forward output shape: {tuple(z.shape)}, grad_fn: {z.grad_fn}")

    # Scalar Loss: Mean Squared Error against synthetic targets
    y_target = torch.tensor([[1.0, 0.0],
                             [0.0, 1.0]], dtype=torch.float32)
    # L = 0.5 * sum((Z - Y)^2) / N
    diff = z - y_target
    loss = 0.5 * torch.mean(diff ** 2)
    print(f"      Loss value: {loss.item():.6f}, loss grad_fn: {loss.grad_fn}")

    # Step 2: Trigger Reverse Mode Automatic Differentiation
    print("[2/5] Executing reverse-mode differentiation via loss.backward()...")
    loss.backward()

    assert w.grad is not None, "w.grad must be populated"
    assert b.grad is not None, "b.grad must be populated"
    assert w.grad.shape == w.shape, "Gradient shape must match parameter shape"
    assert b.grad.shape == b.shape, "Bias gradient shape must match bias shape"

    # Step 3: Analytical Gradient Verification
    print("[3/5] Validating autograd gradients against analytical matrix calculus...")
    # Loss: L = (1 / (2 * N * K)) * sum_{i,j} (Z_ij - Y_ij)^2 where N=2, K=2
    # dL/dZ = (1 / (N * K)) * (Z - Y)
    # dL/dW = X^T @ (dL/dZ)
    # dL/db = sum_rows(dL/dZ)
    num_elements = diff.numel()  # 4
    dL_dZ = diff.detach() / float(num_elements)
    analytic_w_grad = torch.matmul(x.t(), dL_dZ)
    analytic_b_grad = dL_dZ.sum(dim=0)

    assert torch.allclose(w.grad, analytic_w_grad, atol=1e-6), f"w.grad mismatch: {w.grad} vs {analytic_w_grad}"
    assert torch.allclose(b.grad, analytic_b_grad, atol=1e-6), f"b.grad mismatch: {b.grad} vs {analytic_b_grad}"
    print("      Analytical and autograd partial derivatives match within 1e-6 precision.")

    # Step 4: Disabling Autograd via torch.no_grad()
    print("[4/5] Testing torch.no_grad() inference context...")
    with torch.no_grad():
        z_eval = torch.matmul(x, w) + b
        assert not z_eval.requires_grad, "z_eval must not require grad under no_grad()"
        assert z_eval.grad_fn is None, "z_eval must not have grad_fn under no_grad()"
    print("      torch.no_grad() successfully prevented computational graph formation.")

    # Step 5: Severing History via .detach() and Selective Parameter Freezing
    print("[5/5] Testing tensor.detach() and parameter freezing...")
    z_fresh = torch.matmul(x, w) + b
    z_detached = z_fresh.detach()

    assert z_detached.data_ptr() == z_fresh.data_ptr(), "detach() must share identical memory buffer!"
    assert not z_detached.requires_grad, "detached tensor must have requires_grad=False"
    assert z_detached.grad_fn is None, "detached tensor must have grad_fn=None"

    # Test selective parameter freezing
    w.requires_grad = False
    assert not w.requires_grad, "w must now be frozen"
    w.requires_grad = True  # Reset

    print("      detach() zero-copy sharing and parameter freezing verified.")
    print("All autograd and DAG assertion checks passed cleanly!")


if __name__ == "__main__":
    main()
