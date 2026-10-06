"""
Script 01: Multi-Layer Perceptron (MLP) Architecture & Forward Inference Pipeline
==================================================================================
Demonstrates and validates:
1. Object-oriented neural network definition via torch.nn.Module and super().__init__().
2. Feed-forward pipeline encapsulation via torch.nn.Sequential and nn.Flatten().
3. Exact analytical parameter count verification (669,706 parameters).
4. Forward pass execution via __call__ dispatch with shape invariants.
5. Softmax normalization, posterior probability bounds, and Bayes MAP argmax prediction.
"""

import sys
import torch
import torch.nn as nn


class MNISTMultiLayerPerceptron(nn.Module):
    """
    Two-hidden-layer Multi-Layer Perceptron matching Tutorial 12 specification.
    Input: (B, 1, 28, 28) or (B, 28, 28)
    Hidden 1: 512 units with ReLU
    Hidden 2: 512 units with ReLU
    Output: 10 logits
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        flat_x = self.flatten(x)
        logits = self.linear_relu_stack(flat_x)
        return logits


def main():
    print("[1/5] Instantiating MNIST Multi-Layer Perceptron...")
    model = MNISTMultiLayerPerceptron()

    # Step 1: Verify total learnable parameters analytically
    # Layer 1: 784 * 512 + 512 = 401,920
    # Layer 2: 512 * 512 + 512 = 262,656
    # Layer 3: 512 * 10 + 10   = 5,130
    # Total: 401920 + 262656 + 5130 = 669,706
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total parameters: {total_params:,} (expected: 669,706)")
    assert total_params == 669706, f"Expected 669,706 parameters, got {total_params}"
    assert trainable_params == total_params, "All parameters must be trainable by default"

    # Step 2: Test Forward Pass Execution with Synthetic Batches
    print("[2/5] Verifying forward propagation and tensor shape progression...")
    batch_size = 8
    dummy_input = torch.randn(batch_size, 1, 28, 28)  # Shape: [B, C, H, W] = [8, 1, 28, 28]

    # Calling model(x) invokes __call__ which runs hooks then forward()
    logits = model(dummy_input)  # Shape: [B, D] = [8, 10]
    assert logits.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {logits.shape}"
    print(f"      Input shape: {tuple(dummy_input.shape)} -> Output Logits shape: {tuple(logits.shape)}")

    # Step 3: Verify Forward Hook Execution via __call__
    print("[3/5] Verifying execution dispatch via __call__ vs direct .forward()...")
    hook_fired = {"count": 0}
    def forward_hook(module, input_tensor, output_tensor):
        hook_fired["count"] += 1

    hook_handle = model.register_forward_hook(forward_hook)
    _ = model(dummy_input)
    assert hook_fired["count"] == 1, "Calling model(x) must trigger forward hooks!"

    _ = model.forward(dummy_input)
    assert hook_fired["count"] == 1, "Direct model.forward(x) must bypass forward hooks!"
    hook_handle.remove()
    print("      Forward hook dispatch contract verified.")

    # Step 4: Verify Softmax Normalization and Posterior Probabilities
    print("[4/5] Verifying Softmax normalization and Bayes decision invariance...")
    softmax = nn.Softmax(dim=1)
    probs = softmax(logits)

    assert probs.shape == (batch_size, 10), "Posterior shape mismatch!"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities must be in [0, 1]"
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(batch_size), atol=1e-5), "Probabilities must sum to 1.0 per sample"

    # Step 5: Argmax Class Extraction Equivalence
    pred_from_probs = torch.argmax(probs, dim=1)
    pred_from_logits = torch.argmax(logits, dim=1)
    assert torch.equal(pred_from_probs, pred_from_logits), "Argmax over logits must equal argmax over Softmax!"
    print(f"      Predicted classes for mini-batch of {batch_size}: {pred_from_logits.tolist()}")

    print("[5/5] All MLP architecture and inference assertions passed cleanly!")


if __name__ == "__main__":
    main()
