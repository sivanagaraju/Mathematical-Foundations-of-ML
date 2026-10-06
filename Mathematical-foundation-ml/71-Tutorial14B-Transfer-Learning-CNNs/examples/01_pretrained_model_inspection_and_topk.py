"""
Tutorial 14B: Example 1 - Pretrained Vision Model Inspection & Top-K Evaluation
==============================================================================
This script demonstrates:
1. Loading standard vision architectures (ResNet-18 / VGG-19) in PyTorch.
2. Setting inference evaluation mode (`model.eval()`) to disable stochastic dropout
   and fix batch normalization running statistics.
3. Passing standard ImageNet resolution tensor batches [Batch, 3, 224, 224].
4. Verifying forward shape progression producing unnormalized class logits [Batch, 1000].
5. Computing posterior class probabilities via `torch.nn.functional.softmax(dim=1)`
   and verifying sum-to-one constraints.
6. Extracting Top-K candidate classes and confidence scores using `torch.topk(probs, k=5)`.
"""

import torch
import torch.nn.functional as F
import torchvision.models as models

def run_resnet_inspection_and_topk():
    print("=== Step 1: Loading ResNet-18 Architecture & Setting Eval Mode ===")
    # Construct ResNet-18 architecture (weights=None ensures 100% offline deterministic execution)
    torch.manual_seed(42)
    model = models.resnet18(weights=None)
    model.eval()
    
    # Assert model is in evaluation mode
    assert not model.training, "Model should be in eval mode"
    print("Model successfully placed in evaluation mode (model.eval()).")

    # Step 2: Dummy input matching standard ImageNet resolution [B=2, C=3, H=224, W=224]
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)  # Shape: [B, C, H, W] = [2, 3, 224, 224]
    print(f"\n=== Step 2: Forward Pass on ImageNet Input Batch {list(x.shape)} ===")
    
    with torch.no_grad():
        logits = model(x)  # Shape: [B, num_classes] = [2, 1000]
        
    print(f"Logits output shape: {list(logits.shape)}")
    assert logits.shape == (batch_size, 1000), f"Expected [2, 1000], got {logits.shape}"
    print("[PASS] Forward pass generated unnormalized class logits for 1,000 classes.")

    # Step 3: Softmax posterior computation
    print("\n=== Step 3: Softmax Probability Normalization ===")
    probs = F.softmax(logits, dim=1)  # Shape: [B, 1000]
    
    # Verify probability constraints
    assert probs.shape == (batch_size, 1000)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    prob_sums = probs.sum(dim=1)
    for b in range(batch_size):
        assert torch.isclose(prob_sums[b], torch.tensor(1.0)), f"Batch {b} probabilities do not sum to 1.0"
    print(f"Probabilities verified: strictly non-negative and row-sums = {prob_sums.tolist()}")

    # Step 4: Top-K prediction extraction
    k = 5
    print(f"\n=== Step 4: Extracting Top-{k} Predictions via torch.topk ===")
    topk_probs, topk_indices = torch.topk(probs, k=k, dim=1)  # Shape: [B, K] = [2, 5]
    
    print(f"Top-{k} Probabilities shape: {list(topk_probs.shape)}")
    print(f"Top-{k} Class Indices shape: {list(topk_indices.shape)}")
    assert topk_probs.shape == (batch_size, k)
    assert topk_indices.shape == (batch_size, k)

    for b in range(batch_size):
        print(f"\nSample {b} Top-{k} Predictions:")
        for rank in range(k):
            c_id = topk_indices[b, rank].item()
            p_val = topk_probs[b, rank].item()
            print(f"  Rank {rank+1}: Class ID {c_id:<4d} | Probability: {p_val*100:.2f}%")

    print("\n[PASS] Top-5 evaluation successfully executed.")


if __name__ == "__main__":
    run_resnet_inspection_and_topk()
    print("\nAll Pretrained Model Inspection and Top-K checks passed successfully!")
