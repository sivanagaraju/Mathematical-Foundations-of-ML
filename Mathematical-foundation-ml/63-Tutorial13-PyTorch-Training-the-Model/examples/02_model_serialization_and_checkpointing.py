"""
Script 02: Model Persistence, State Dictionary Serialization, & Restoration
===========================================================================
Demonstrates and validates:
1. Model parameter extraction via model.state_dict().
2. Secure tensor serialization to disk via torch.save(model.state_dict(), path).
3. Independent model re-instantiation and parameter divergence verification.
4. Parameter restoration via model.load_state_dict(torch.load(path, weights_only=True)).
5. Exact floating-point parameter identity across all layers.
6. Forward inference output invariance between original and restored models.
"""

import os
import tempfile
import torch
import torch.nn as nn


class MNISTClassifier(nn.Module):
    """Multi-Layer Perceptron architecture for MNIST classification."""
    def __init__(self, in_features=784, hidden_dim=128, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.flatten(x))


def main():
    print("--- Running Script 02: Model Persistence & Checkpointing ---")
    torch.manual_seed(123)

    # Step 1: Instantiate primary model and generate trained mock weights
    model_orig = MNISTClassifier()
    dummy_input = torch.randn(16, 1, 28, 28)  # Shape: [B, C, H, W] = [16, 1, 28, 28]

    # Perform a mock forward/backward step to alter weights from default init
    criterion = nn.CrossEntropyLoss()
    target = torch.randint(0, 10, (16,))  # Shape: [B] = [16]
    orig_output = model_orig(dummy_input)  # Shape: [B, 10] = [16, 10]
    loss = criterion(orig_output, target)
    loss.backward()

    with torch.no_grad():
        for p in model_orig.parameters():
            p.sub_(0.01 * p.grad)

    print("[*] Primary model initialized and weights trained.")

    # Step 2: Extract state_dict and inspect structure
    state_dict = model_orig.state_dict()
    print(f"[*] State dict extracted with {len(state_dict)} parameter entries:")
    for key, tensor in state_dict.items():
        print(f"    - Key '{key}': Shape {tuple(tensor.shape)}, Dtype {tensor.dtype}")

    # Step 3: Serialize state_dict to temporary storage
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
        checkpoint_path = tmp_file.name

    try:
        torch.save(model_orig.state_dict(), checkpoint_path)
        file_size_kb = os.path.getsize(checkpoint_path) / 1024
        print(f"[*] Serialized state_dict to '{checkpoint_path}' ({file_size_kb:.2f} KB).")

        # Step 4: Instantiate second independent model
        model_restored = MNISTClassifier()

        # Verify that restored model currently has different weights from model_orig
        orig_first_param = next(model_orig.parameters())
        restored_first_param = next(model_restored.parameters())
        assert not torch.allclose(orig_first_param, restored_first_param), "Independent models must have different weights before loading!"

        # Step 5: Load serialized state_dict into restored model
        loaded_state = torch.load(checkpoint_path, weights_only=True)
        model_restored.load_state_dict(loaded_state)
        print("[*] Loaded state_dict into fresh model architecture with weights_only=True.")

        # Step 6: Verify exact parameter numerical equality across all layers
        for (name_orig, param_orig), (name_restored, param_restored) in zip(
            model_orig.named_parameters(), model_restored.named_parameters()
        ):
            assert name_orig == name_restored, f"Parameter name mismatch: {name_orig} vs {name_restored}"
            assert torch.equal(param_orig, param_restored), f"Parameter values for '{name_orig}' do not match!"
        print("[*] All parameter tensors verified bit-for-bit identical.")

        # Step 7: Verify forward inference equivalence
        model_orig.eval()
        model_restored.eval()
        with torch.no_grad():
            out_orig = model_orig(dummy_input)      # Shape: [B, 10]
            out_restored = model_restored(dummy_input)  # Shape: [B, 10]

        assert torch.allclose(out_orig, out_restored, atol=1e-6), "Forward pass logits must match identically between original and restored models!"
        pred_orig = out_orig.argmax(dim=1)          # Shape: [B]
        pred_restored = out_restored.argmax(dim=1)  # Shape: [B]
        assert torch.equal(pred_orig, pred_restored), "Class predictions must match identically!"
        print(f"[*] Inference prediction parity verified for all {dummy_input.size(0)} batch samples.")

    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("[*] Cleaned up temporary checkpoint file.")

    print("--- Script 02 Completed Successfully (Exit Code 0) ---")


if __name__ == "__main__":
    main()
