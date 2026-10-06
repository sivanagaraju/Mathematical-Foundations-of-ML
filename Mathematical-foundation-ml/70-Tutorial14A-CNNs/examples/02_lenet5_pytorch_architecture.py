"""
Tutorial 14A: Example 2 - LeNet-5 Architecture in PyTorch (Modernized with ReLU & Decoupled Trunk)
==================================================================================================
This script implements:
1. LeNet-5 neural network architecture decoupled into:
   - self.features: Convolutional representation trunk extracting spatial embeddings
   - self.classifier: Dense classification MLP head mapping embeddings to class logits
2. Modernization discussed in lecture (Topic 5):
   - Replacing original saturating Sigmoid activations with nn.ReLU
   - Average pooling layers (AvgPool2d, kernel=2, stride=2) halving spatial dimensions
3. Downstream embedding extraction:
   - Extracting latent representations (Z) from self.features for sequence models
     or transfer learning (as emphasized by the instructor)
4. Shape tracing and tensor dimension verification through every layer
5. One full training iteration (forward, CrossEntropyLoss, backward, Adam step)
6. Parameter count comparison against a naive fully-connected MLP.
"""

import torch
import torch.nn as nn
import torch.optim as optim

class ModernLeNet5(nn.Module):
    """
    Modernized LeNet-5 for MNIST (28x28 grayscale images, 10 classes).
    Decoupled into feature extractor trunk and classifier head.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Feature Extractor Trunk
        # Input: [Batch, 1, 28, 28]
        self.features = nn.Sequential(
            # Conv1: 1 in_channel -> 6 out_channels, 5x5 kernel, padding=2
            # Shape: [B, 6, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # Pool1: 2x2 average pool with stride 2 -> halves spatial resolution
            # Shape: [B, 6, 14, 14]
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # Conv2: 6 in_channels -> 16 out_channels, 5x5 kernel, padding=0
            # Shape: [B, 16, 10, 10]
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            # Pool2: 2x2 average pool with stride 2
            # Shape: [B, 16, 5, 5]
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # Flatten 3D spatial tensor [B, 16, 5, 5] to 1D vector [B, 400]
            nn.Flatten()
        )
        
        # Dense Classification Head
        # Input: [Batch, 400]
        self.classifier = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts latent visual embeddings Z for downstream transfer tasks."""
        # Shape: [B, 400]
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Feature Extraction
        # Shape: [B, 400]
        features = self.features(x)
        # Step 2: Classification
        # Shape: [B, 10]
        logits = self.classifier(features)
        return logits


def trace_shapes_and_forward():
    print("=== Step 1: Tracing Layer-by-Layer Tensor Shapes ===")
    model = ModernLeNet5(num_classes=10)
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    print(f"Input shape: {list(x.shape)}")

    # Trace each layer in features
    h = x
    layer_names = [
        "Conv1 (5x5, pad=2, out=6)",
        "ReLU1",
        "AvgPool1 (2x2, stride=2)",
        "Conv2 (5x5, pad=0, out=16)",
        "ReLU2",
        "AvgPool2 (2x2, stride=2)",
        "Flatten"
    ]
    for name, layer in zip(layer_names, model.features):
        h = layer(h)
        print(f"  After {name:<26} -> Shape: {list(h.shape)}")
    
    assert h.shape == (batch_size, 400), f"Expected [4, 400], got {h.shape}"

    # Trace classifier
    clf_names = [
        "Linear1 (400 -> 120)",
        "ReLU3",
        "Linear2 (120 -> 84)",
        "ReLU4",
        "Linear3 (84 -> 10)"
    ]
    for name, layer in zip(clf_names, model.classifier):
        h = layer(h)
        print(f"  After {name:<26} -> Shape: {list(h.shape)}")

    assert h.shape == (batch_size, 10), f"Expected [4, 10], got {h.shape}"
    print("[PASS] Tensor shape progression verified.")
    return model


def verify_embedding_extraction(model: ModernLeNet5):
    print("\n=== Step 2: Downstream Embedding Extraction Verification ===")
    x = torch.randn(2, 1, 28, 28)
    z = model.extract_features(x)
    print(f"Extracted feature embedding shape: {list(z.shape)}")
    assert z.shape == (2, 400), f"Expected [2, 400], got {z.shape}"
    print("[PASS] Latent visual embedding successfully decoupled from classifier.")


def verify_training_step(model: ModernLeNet5):
    print("\n=== Step 3: Backward Pass & Optimizer Update Verification ===")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy batch of 8 samples
    x = torch.randn(8, 1, 28, 28)
    y_true = torch.tensor([0, 3, 5, 9, 2, 7, 1, 4])
    
    # Forward pass
    logits = model(x)
    loss = criterion(logits, y_true)
    initial_loss = loss.item()
    print(f"Initial loss: {initial_loss:.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient detected in {name}"
    
    # Optimizer step
    optimizer.step()
    
    # Re-evaluate
    with torch.no_grad():
        new_loss = criterion(model(x), y_true).item()
    print(f"Post-step loss: {new_loss:.4f}")
    assert new_loss < initial_loss, f"Loss should decrease after Adam step: {new_loss} vs {initial_loss}"
    print("[PASS] Backward pass and Adam optimizer step executed successfully.")


def compare_parameter_counts(model: ModernLeNet5):
    print("\n=== Step 4: Parameter Count Comparison (CNN vs Dense MLP) ===")
    cnn_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    feature_params = sum(p.numel() for p in model.features.parameters() if p.requires_grad)
    classifier_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    
    # Naive MLP with comparable layer widths: 784 -> 1024 -> 512 -> 10
    # Layer 1: 784 * 1024 + 1024 = 803,840
    # Layer 2: 1024 * 512 + 512 = 524,800
    # Layer 3: 512 * 10 + 10 = 5,130
    # Total MLP params = 1,333,770
    mlp_params = (784 * 1024 + 1024) + (1024 * 512 + 512) + (512 * 10 + 10)
    
    print(f"LeNet-5 Feature Trunk Parameters:   {feature_params:>8,d}")
    print(f"LeNet-5 Classifier Head Parameters: {classifier_params:>8,d}")
    print(f"Total LeNet-5 Parameters:           {cnn_params:>8,d}")
    print(f"Comparable Naive MLP Parameters:    {mlp_params:>8,d}")
    ratio = mlp_params / cnn_params
    print(f"Parameter Efficiency: LeNet-5 is ~{ratio:.1f}x more compact than dense MLP!")
    
    assert cnn_params < 70000, f"Expected <70k params for LeNet-5, got {cnn_params}"
    print("[PASS] Parameter efficiency benchmark verified.")


if __name__ == "__main__":
    trained_model = trace_shapes_and_forward()
    verify_embedding_extraction(trained_model)
    verify_training_step(trained_model)
    compare_parameter_counts(trained_model)
    print("\nAll LeNet-5 PyTorch architecture checks passed successfully!")
