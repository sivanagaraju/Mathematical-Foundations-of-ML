"""
Simulation: Transfer Learning & Feature Extraction Mechanics
Focus: Linear Probing vs Full Fine-Tuning with Parameter Freezing

This script implements:
1. A pre-trained feature extractor G*(x) simulating an upstream foundation model.
2. Feature extraction (linear probing) with frozen trunk (requires_grad = False).
3. Gradient isolation verification (zero grads on frozen trunk, non-zero on head).
4. Unfrozen full fine-tuning with differential learning rates.
5. Numerical assertions verifying representation shapes and gradient flows.
"""

import torch
import torch.nn as nn
import torch.optim as optim

def run_transfer_learning_simulation():
    print("=" * 60)
    print("RUNNING TRANSFER LEARNING FEATURE EXTRACTION SIMULATION")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    batch_size = 16
    in_features = 64
    hidden_dim = 32
    num_classes = 4
    
    # -------------------------------------------------------------
    # 1. Define Pre-Trained Backbone G*(x)
    # -------------------------------------------------------------
    class Backbone(nn.Module):
        def __init__(self, in_dim, h_dim):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, 48)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(48, h_dim)
            
        def forward(self, x):
            # Input shape: [batch_size, in_features]
            # Output representation z: [batch_size, hidden_dim]
            a1 = self.act(self.fc1(x))
            z = self.fc2(a1)
            return z
            
    class TransferModel(nn.Module):
        def __init__(self, backbone, h_dim, n_classes):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(h_dim, n_classes)
            
        def forward(self, x):
            z = self.backbone(x)
            logits = self.head(z)
            return z, logits

    # Instantiate pre-trained model
    backbone = Backbone(in_features, hidden_dim)
    model = TransferModel(backbone, hidden_dim, num_classes)
    
    # Generate synthetic downstream target dataset
    x_target = torch.randn(batch_size, in_features)  # Shape: [16, 64]
    y_target = torch.randint(0, num_classes, (batch_size,))  # Shape: [16]
    
    # -------------------------------------------------------------
    # 2. Phase 1: Linear Probing (Feature Extractor Freezing)
    # -------------------------------------------------------------
    print("\n[Phase 1] Linear Probing with Frozen Backbone...")
    
    # Freeze all backbone parameters
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    for param in model.head.parameters():
        param.requires_grad = True
        
    criterion = nn.CrossEntropyLoss()
    optimizer_probe = optim.SGD(model.head.parameters(), lr=0.1)
    
    # Verify forward pass and intermediate representation tapping
    z_embed, logits_probe = model(x_target)
    
    # Assert shapes
    assert z_embed.shape == (batch_size, hidden_dim), f"Expected shape [16, 32], got {z_embed.shape}"
    assert logits_probe.shape == (batch_size, num_classes), f"Expected shape [16, 4], got {logits_probe.shape}"
    
    # Backward pass
    loss_probe = criterion(logits_probe, y_target)
    optimizer_probe.zero_grad()
    loss_probe.backward()
    
    # Strict assertions: backbone gradients MUST be None
    for name, param in model.backbone.named_parameters():
        assert param.grad is None, f"Backbone param {name} should have NO gradient when frozen!"
    
    # Head gradients MUST be computed
    assert model.head.weight.grad is not None, "Downstream head must receive gradient"
    assert model.head.bias.grad is not None, "Downstream head bias must receive gradient"
    
    print("PASS: Frozen backbone gradients are strictly None.")
    print("PASS: Downstream head gradients successfully computed.")
    
    # Run 20 linear probing update steps
    initial_probe_loss = loss_probe.item()
    for _ in range(20):
        optimizer_probe.zero_grad()
        _, lgt = model(x_target)
        loss = criterion(lgt, y_target)
        loss.backward()
        optimizer_probe.step()
        
    final_probe_loss = loss.item()
    print(f"Linear Probe Loss: {initial_probe_loss:.4f} -> {final_probe_loss:.4f}")
    assert final_probe_loss < initial_probe_loss, "Linear probe should minimize downstream empirical risk"
    
    # -------------------------------------------------------------
    # 3. Phase 2: Full Fine-Tuning with Differential Learning Rates
    # -------------------------------------------------------------
    print("\n[Phase 2] Full Fine-Tuning with Unfrozen Backbone...")
    
    # Unfreeze all backbone parameters
    for param in model.backbone.parameters():
        param.requires_grad = True
        
    # Differential learning rates: conservative on backbone, aggressive on head
    optimizer_finetune = optim.SGD([
        {'params': model.backbone.parameters(), 'lr': 0.005},
        {'params': model.head.parameters(), 'lr': 0.05}
    ])
    
    optimizer_finetune.zero_grad()
    z_ft, logits_ft = model(x_target)
    loss_ft = criterion(logits_ft, y_target)
    loss_ft.backward()
    
    # Strict assertions: backbone gradients MUST now be non-zero
    for name, param in model.backbone.named_parameters():
        assert param.grad is not None, f"Backbone param {name} should now receive gradient in fine-tuning!"
        assert param.grad.abs().sum() > 0, f"Backbone param {name} gradient should be non-zero"
        
    optimizer_finetune.step()
    print("PASS: Full fine-tuning propagates gradients through entire composite graph.")
    print("=" * 60)
    print("SIMULATION COMPLETED CLEANLY WITH ALL ASSERTIONS SATISFIED")
    print("=" * 60)

if __name__ == '__main__':
    run_transfer_learning_simulation()
