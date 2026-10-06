"""
Tutorial 14B: Example 2 - Classifier Head Surgery, Parameter Freezing & Decoupled Embeddings
============================================================================================
This script demonstrates:
1. Transfer learning via surgical head replacement:
   - Modifying ResNet-18 final fully connected layer from 1,000 ImageNet classes
     to 5 custom target classes (e.g. medical pathology classification).
2. Parameter freezing:
   - Setting `requires_grad = False` on the feature extraction trunk to lock representations.
   - Initializing only the new classification head with learnable parameters.
3. Gradient isolation verification:
   - Proving that backward pass computes gradients exclusively for the new head (`grad is not None`)
     while the convolutional trunk receives zero gradient (`grad is None`).
4. Decoupled latent visual embedding extraction:
   - Building a specialized feature extractor module extracting 512-dimensional latent vectors (Z)
     for downstream multimodal transfer (e.g. image captioning).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class ResNetEmbeddingExtractor(nn.Module):
    """
    Decoupled vision backbone extracting latent representations Z (512-d).
    Reuses layers from ResNet up to the average pooling stage, omitting model.fc.
    """
    def __init__(self, base_resnet: nn.Module):
        super().__init__()
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4
        self.avgpool = base_resnet.avgpool
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial downsampling through residual blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        z = self.flatten(x)
        # Output shape: [Batch, 512]
        return z


def run_transfer_learning_pipeline():
    print("=== Step 1: Loading Backbone & Freezing Feature Trunk ===")
    torch.manual_seed(42)
    model = models.resnet18(weights=None)
    
    # Freeze all parameters in the base network
    for param in model.parameters():
        param.requires_grad = False
        
    print("All backbone parameters frozen (requires_grad = False).")

    # Step 2: Surgically replace the final classification head
    num_classes = 5
    in_features = model.fc.in_features
    print(f"\n=== Step 2: Replacing Classifier Head (in_features={in_features} -> out_features={num_classes}) ===")
    model.fc = nn.Linear(in_features=in_features, out_features=num_classes)
    
    # Verify parameter gradient states
    assert model.fc.weight.requires_grad == True, "New head weights must have requires_grad=True"
    assert model.fc.bias.requires_grad == True, "New head bias must have requires_grad=True"
    assert model.conv1.weight.requires_grad == False, "Trunk conv1 must remain frozen"
    assert model.layer4[1].conv2.weight.requires_grad == False, "Trunk layer4 must remain frozen"
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Frozen parameters (Trunk):    {frozen_params:>10,d}")
    print(f"Trainable parameters (Head):  {trainable_params:>10,d}")
    assert trainable_params == 512 * 5 + 5  # 2,565 parameters
    print("[PASS] Classifier head surgery verified.")

    # Step 3: Gradient isolation check during backpropagation
    print("\n=== Step 3: Gradient Isolation & Parameter Update Check ===")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    
    # Store initial weights to verify immutability of frozen trunk
    initial_conv1_weight = model.conv1.weight.clone()
    initial_fc_weight = model.fc.weight.clone()
    
    # Dummy batch of 4 samples
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 2, 4, 1])
    
    # Forward pass
    logits = model(x)
    assert logits.shape == (4, num_classes)
    loss = criterion(logits, y)
    print(f"Initial loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Assert gradients exist on head, but are strictly None on frozen trunk
    assert model.fc.weight.grad is not None, "Head should receive gradients"
    assert model.fc.bias.grad is not None, "Head bias should receive gradients"
    assert model.conv1.weight.grad is None, "Frozen conv1 should NOT receive gradients (grad must be None)"
    assert model.layer1[0].conv1.weight.grad is None, "Frozen layer1 should NOT receive gradients"
    print("[PASS] Gradient isolation confirmed: Trunk received None, Head received gradients.")

    # Optimizer step
    optimizer.step()
    
    # Verify head weights changed, while trunk weights remained identical
    assert not torch.equal(model.fc.weight, initial_fc_weight), "Head weights should be updated"
    assert torch.equal(model.conv1.weight, initial_conv1_weight), "Frozen conv1 weights must NOT change"
    print("[PASS] Parameter update verified: Head updated, Trunk remained strictly immutable.")

    # Step 4: Decoupled Embedding Extraction for downstream sequence models
    print("\n=== Step 4: Extracting Decoupled Visual Embeddings Z ===")
    embed_extractor = ResNetEmbeddingExtractor(model)
    with torch.no_grad():
        z_embed = embed_extractor(x)
        
    print(f"Input batch shape:              {list(x.shape)}")
    print(f"Decoupled visual embedding Z:   {list(z_embed.shape)}")
    assert z_embed.shape == (4, 512), f"Expected [4, 512], got {z_embed.shape}"
    print("[PASS] 512-dimensional latent embedding successfully decoupled from classifier.")


if __name__ == "__main__":
    run_transfer_learning_pipeline()
    print("\nAll Transfer Learning & Embedding Extraction checks passed successfully!")
