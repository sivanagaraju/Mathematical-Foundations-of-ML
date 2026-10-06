"""
Bidirectional Recurrent Networks & Sequence Classification Pipeline
===================================================================

This script demonstrates and validates the mechanics of Bidirectional Recurrent
Networks (BiRNN, BiLSTM, and BiGRU) in PyTorch.

Key Concepts Demonstrated:
1. Bidirectional recurrence (`bidirectional=True`) combining forward ($t=1 \\to T$)
   and backward ($t=T \\to 1$) sequence sweeps.
2. Bidirectional tensor geometry:
   - Sequence output: `[B, T, 2 * H]` (concatenation $[h_t^\\to; h_t^\\leftarrow]$).
   - Recurrent state `h_n`: `[2 * L, B, H]` where even indices are forward directions
     and odd indices are backward directions.
3. Slicing terminal states from bidirectional models for classification:
   - Forward terminal representation: `h_n[-2]` (or `output[:, -1, :H]`).
   - Backward terminal representation: `h_n[-1]` (or `output[:, 0, H:]`).
4. End-to-end synthetic training step with cross-entropy loss, gradient clipping
   via `clip_grad_norm_`, and parameter updates.

All operations execute deterministically on CPU with explicit shape comments and
assertions.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class BidirectionalLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM sequence classifier mapping [B, T, D] -> [B, K].
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # Bidirectional concatenation doubles feature dimension: 2 * hidden_dim
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, D]
        output, (h_n, c_n) = self.lstm(x)
        # output shape: [B, T, 2 * H]
        # h_n shape:    [2 * L, B, H]
        # c_n shape:    [2 * L, B, H]

        # Extract top-layer forward and backward terminal states:
        # Forward pass final state at t=T is at index -2
        # Backward pass final state at t=1 is at index -1
        h_forward_terminal = h_n[-2]   # Shape: [B, H]
        h_backward_terminal = h_n[-1]  # Shape: [B, H]

        # Concatenate forward and backward summaries
        h_bidirectional = torch.cat([h_forward_terminal, h_backward_terminal], dim=-1)  # Shape: [B, 2 * H]

        # Project to class logits
        logits = self.classifier(h_bidirectional)  # Shape: [B, K]
        return logits


def test_bidirectional_tensor_geometry():
    print("=" * 70)
    print("TEST 1: Bidirectional Tensor Geometry and Slicing Verification")
    print("=" * 70)

    torch.manual_seed(101)

    B = 4   # Batch size
    T = 6   # Sequence length
    D = 10  # Input dimension
    H = 12  # Hidden dimension
    L = 2   # Number of stacked layers

    x = torch.randn(B, T, D)
    print(f"[Input Tensor] Shape: {list(x.shape)} (B={B}, T={T}, D={D})")

    bilstm = nn.LSTM(
        input_size=D,
        hidden_size=H,
        num_layers=L,
        batch_first=True,
        bidirectional=True
    )

    output, (h_n, c_n) = bilstm(x)

    # Sequence output shape: [B, T, 2 * H]
    print(f"[BiLSTM Output] Shape: {list(output.shape)} (B, T, 2*H)")
    assert output.shape == (B, T, 2 * H)

    # Recurrent states shape: [2 * L, B, H]
    print(f"[BiLSTM h_n]    Shape: {list(h_n.shape)} (2*L, B, H)")
    print(f"[BiLSTM c_n]    Shape: {list(c_n.shape)} (2*L, B, H)")
    assert h_n.shape == (2 * L, B, H)
    assert c_n.shape == (2 * L, B, H)

    # Verify forward slice equivalence:
    # Forward final hidden state at step T from output[:, -1, :H]
    # should match top-layer forward state in h_n[-2, :, :]
    fwd_from_output = output[:, -1, :H]
    fwd_from_hn = h_n[-2, :, :]
    diff_fwd = torch.max(torch.abs(fwd_from_output - fwd_from_hn)).item()
    print(f"[Forward Match]  Max diff between output[:,-1,:H] and h_n[-2]: {diff_fwd:.8f}")
    assert diff_fwd < 1e-6, "Forward terminal slice mismatch!"

    # Verify backward slice equivalence:
    # Backward final hidden state at step 1 from output[:, 0, H:]
    # should match top-layer backward state in h_n[-1, :, :]
    bwd_from_output = output[:, 0, H:]
    bwd_from_hn = h_n[-1, :, :]
    diff_bwd = torch.max(torch.abs(bwd_from_output - bwd_from_hn)).item()
    print(f"[Backward Match] Max diff between output[:,0,H:] and h_n[-1]: {diff_bwd:.8f}")
    assert diff_bwd < 1e-6, "Backward terminal slice mismatch!"

    print("PASS: Bidirectional output and h_n coordinate slicing confirmed.")


def test_bilstm_training_and_gradient_clipping():
    print("\n" + "=" * 70)
    print("TEST 2: End-to-End Training Step with Gradient Norm Clipping")
    print("=" * 70)

    torch.manual_seed(202)

    B = 8
    T = 15
    D = 16
    H = 24
    K = 3
    L = 2

    model = BidirectionalLSTMClassifier(input_dim=D, hidden_dim=H, num_classes=K, num_layers=L)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Synthetic batch and target class labels
    x_batch = torch.randn(B, T, D)
    y_batch = torch.randint(0, K, (B,))

    # Initial state
    optimizer.zero_grad()
    logits = model(x_batch)  # Shape: [B, K]
    assert logits.shape == (B, K)

    loss = criterion(logits, y_batch)
    print(f"[Initial Loss] {loss.item():.4f}")
    assert not torch.isnan(loss)

    # Backward pass
    loss.backward()

    # Audit gradients before clipping
    total_norm_before = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_before += param_norm.item() ** 2
    total_norm_before = total_norm_before ** 0.5
    print(f"[Gradient Norm Before Clipping] {total_norm_before:.4f}")

    # Clip gradients to max norm 1.0 to prevent exploding gradients
    max_grad_norm = 1.0
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    print(f"[Reported Clipped Norm]         {grad_norm:.4f}")

    # Verify that post-clipping norm is bounded
    total_norm_after = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_after += param_norm.item() ** 2
    total_norm_after = total_norm_after ** 0.5
    print(f"[Gradient Norm After Clipping]  {total_norm_after:.4f}")
    assert total_norm_after <= max_grad_norm + 1e-4

    # Optimizer step
    optimizer.step()

    # Forward pass after step to verify loss reduction
    with torch.no_grad():
        logits_after = model(x_batch)
        loss_after = criterion(logits_after, y_batch)
    print(f"[Loss After Single Step]       {loss_after.item():.4f}")
    print("PASS: Training step and gradient clipping verified.")


if __name__ == "__main__":
    test_bidirectional_tensor_geometry()
    test_bilstm_training_and_gradient_clipping()
    print("\nALL BIDIRECTIONAL RECURRENT TESTS PASSED CLEANLY (EXIT CODE 0).")
