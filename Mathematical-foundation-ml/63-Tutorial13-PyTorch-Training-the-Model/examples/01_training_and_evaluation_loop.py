"""
Script 01: Complete Training & Evaluation Pipeline in PyTorch
============================================================
Demonstrates and validates:
1. Multi-Layer Perceptron architecture setup for classification.
2. Data ingestion and DataLoader batching with I.I.D. shuffling.
3. The four-phase training iteration: model(X) -> loss -> zero_grad -> backward -> step.
4. Gradient accumulation detection and zero_grad verification.
5. Deterministic evaluation loop under model.eval() and with torch.no_grad():.
6. Relational accuracy calculation via argmax and boolean tensor casting.
7. Scalar memory isolation using loss.item().
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleMLP(nn.Module):
    """3-layer feedforward MLP for 784-dim input and 10-class output."""
    def __init__(self, in_features=784, hidden_dim=64, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(self.flatten(x))


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    """Executes one training epoch across mini-batches."""
    model.train()  # Set model to training mode
    total_loss = 0.0

    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # Shape: X is [B, 1, 28, 28], y is [B]

        # Phase 1: Forward Pass (produce raw unbounded logits)
        pred = model(X)  # Shape: [B, 10]

        # Phase 2: Compute Loss
        loss = loss_fn(pred, y)  # Scalar loss in R

        # Phase 3: Zero Gradients to prevent accumulation
        optimizer.zero_grad(set_to_none=True)

        # Phase 4: Reverse Autodiff
        loss.backward()

        # Phase 5: Optimizer Parameter Update
        optimizer.step()

        # Isolate scalar metric to prevent DAG retention leak
        total_loss += loss.item() * len(X)

    return total_loss / len(dataloader.dataset)


def evaluate(dataloader, model, loss_fn, device):
    """Executes deterministic evaluation over held-out dataset."""
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    correct = 0

    with torch.no_grad():  # Disable dynamic DAG tape allocation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # Shape: X is [B, 1, 28, 28], y is [B]

            pred = model(X)  # Shape: [B, 10]
            test_loss += loss_fn(pred, y).item() * len(X)

            # Compute relational accuracy
            predicted_classes = pred.argmax(dim=1)  # Shape: [B]
            correct += (predicted_classes == y).type(torch.float).sum().item()

    avg_loss = test_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy


def main():
    print("--- Running Script 01: Training & Evaluation Loop ---")
    torch.manual_seed(42)
    device = torch.device("cpu")

    # Step 1: Create synthetic dataset (500 train, 100 test samples)
    n_train, n_test = 500, 100
    X_train = torch.randn(n_train, 1, 28, 28)  # Shape: [B, C, H, W] = [500, 1, 28, 28]
    y_train = torch.randint(0, 10, (n_train,))  # Shape: [B] = [500]
    X_test = torch.randn(n_test, 1, 28, 28)    # Shape: [B, C, H, W] = [100, 1, 28, 28]
    y_test = torch.randint(0, 10, (n_test,))    # Shape: [B] = [100]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    model = SimpleMLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"[*] Dataset initialized: Train={len(train_loader.dataset)}, Test={len(test_loader.dataset)}")
    print(f"[*] Total trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 2: Verify zero_grad necessity by inspecting gradient accumulation
    sample_X, sample_y = next(iter(train_loader))
    out1 = model(sample_X)
    loss1 = loss_fn(out1, sample_y)
    optimizer.zero_grad()
    loss1.backward()

    first_weight_grad = model.network[0].weight.grad.clone()

    # Second backward without zero_grad accumulates
    out2 = model(sample_X)
    loss2 = loss_fn(out2, sample_y)
    loss2.backward()

    accumulated_grad = model.network[0].weight.grad.clone()
    assert torch.allclose(accumulated_grad, 2 * first_weight_grad, atol=1e-4), "Gradients must accumulate by default without zero_grad!"
    print("[*] Gradient accumulation mechanism verified: unzeroed buffers double.")

    # Clean gradient state before actual training
    optimizer.zero_grad(set_to_none=True)

    # Step 3: Run Multi-Epoch Training Loop
    print("[*] Executing 5 training epochs...")
    initial_loss, initial_acc = evaluate(test_loader, model, loss_fn, device)
    print(f"    Initial State: Val Loss = {initial_loss:.4f}, Val Acc = {initial_acc:.2%}")

    losses = []
    for epoch in range(1, 6):
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(test_loader, model, loss_fn, device)
        losses.append(train_loss)
        print(f"    Epoch {epoch}/5: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2%}")

    # Step 4: Verify convergence
    assert losses[-1] < losses[0], f"Training loss must decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
    print("[*] Training convergence verified: final loss decreased from initial loss.")
    print("--- Script 01 Completed Successfully (Exit Code 0) ---")


if __name__ == "__main__":
    main()
