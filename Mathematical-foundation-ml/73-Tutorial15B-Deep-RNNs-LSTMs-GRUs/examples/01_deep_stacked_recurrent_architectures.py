"""
Deep Stacked Recurrent Architectures: Multi-Layer Propagation & Parameter Audits
===============================================================================

This script demonstrates and validates the mechanics of deep stacked recurrent
neural networks (RNN, LSTM, and GRU) in PyTorch.

Key Concepts Demonstrated:
1. Multi-layer vertical stacking (`num_layers >= 2`) and layer-to-layer propagation.
2. Tensor geometry tracking: sequence tensor `output` (top-layer across time) vs
   recurrent state tensor `h_n` (terminal time-step across all vertical layers).
3. Numerical equivalence proof: `output[:, -1, :] == h_n[-1, :, :]` for the top layer.
4. Parameter audits: distinguishing Layer 1 affine dimensions (D -> H) from
   intermediate/top layer affine dimensions (H -> H).
5. Comparing Many-to-One (classification) vs Many-to-Many (sequence tagging) pipelines.

All operations execute deterministically on CPU with explicit shape comments and
assertions.
"""

import torch
import torch.nn as nn


def audit_stacked_rnn_parameters(
    cell_type: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    has_bias: bool = True
) -> int:
    """
    Computes exact theoretical parameter count for multi-layer stacked recurrent models.

    Layer 1:
        Weight matrices: W_ih (H x D), W_hh (H x H)
        Biases: b_ih (H), b_hh (H) if has_bias
    Layers 2 to L:
        Weight matrices: W_ih (H x H), W_hh (H x H)
        Biases: b_ih (H), b_hh (H) if has_bias
    Multiplier M:
        RNN: 1, GRU: 3, LSTM: 4
    """
    mult = {"RNN": 1, "GRU": 3, "LSTM": 4}[cell_type.upper()]
    bias_mult = 2 if has_bias else 0

    # Layer 1
    l1_params = mult * (hidden_size * input_size + hidden_size * hidden_size + bias_mult * hidden_size)

    # Layers 2 to num_layers
    subsequent_params = 0
    if num_layers > 1:
        per_layer = mult * (hidden_size * hidden_size + hidden_size * hidden_size + bias_mult * hidden_size)
        subsequent_params = (num_layers - 1) * per_layer

    return l1_params + subsequent_params


def test_deep_stacked_rnn_forward_pass():
    print("=" * 70)
    print("TEST 1: Deep Stacked RNN Geometry and Layer Equivalence")
    print("=" * 70)

    torch.manual_seed(42)

    batch_size = 3       # B
    seq_len = 5          # T
    input_size = 6       # D
    hidden_size = 8      # H
    num_layers = 3       # L
    num_classes = 4      # K

    # Synthetic mini-batch
    # Shape: [B, T, D] = [3, 5, 6]
    x = torch.randn(batch_size, seq_len, input_size)
    assert x.shape == (3, 5, 6)
    print(f"[Input Tensor] Shape: {list(x.shape)} (B={batch_size}, T={seq_len}, D={input_size})")

    # Instantiate 3-layer stacked RNN
    rnn = nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=False
    )

    # Forward pass through stacked RNN
    # output shape: [B, T, H] = [3, 5, 8]  (top layer only across all time steps)
    # h_n shape:    [L, B, H] = [3, 3, 8]  (terminal step T across all L layers)
    output, h_n = rnn(x)

    print(f"[RNN Output]   Shape: {list(output.shape)} (B, T, H) -> top layer only")
    print(f"[RNN h_n]      Shape: {list(h_n.shape)} (L, B, H) -> all layers at step T")

    assert output.shape == (batch_size, seq_len, hidden_size)
    assert h_n.shape == (num_layers, batch_size, hidden_size)

    # Load-bearing equivalence:
    # Top-layer output at final time step T-1 MUST match h_n at layer index L-1
    top_layer_terminal_from_output = output[:, -1, :]  # Shape: [B, H] = [3, 8]
    top_layer_terminal_from_hn = h_n[-1, :, :]         # Shape: [B, H] = [3, 8]

    diff = torch.max(torch.abs(top_layer_terminal_from_output - top_layer_terminal_from_hn)).item()
    print(f"[Equivalence Check] Max diff between output[:, -1, :] and h_n[-1, :, :]: {diff:.8f}")
    assert diff < 1e-6, "Mismatch between output terminal step and h_n top layer!"
    print("PASS: Top-layer final time step matches h_n[-1] identically.")

    # Contrast with Layer 1 terminal step
    l1_terminal = h_n[0, :, :]
    diff_l1_top = torch.max(torch.abs(l1_terminal - top_layer_terminal_from_output)).item()
    print(f"[Contrast] Diff between Layer 1 terminal and Top Layer terminal: {diff_l1_top:.4f}")
    assert diff_l1_top > 1e-3, "Layer 1 and Layer 3 should produce distinct representations!"


def test_parameter_audits_across_cell_types():
    print("\n" + "=" * 70)
    print("TEST 2: Exact Parameter Counting Across Stacked Cell Types")
    print("=" * 70)

    input_size = 10
    hidden_size = 16
    num_layers = 2

    # 1. Vanilla RNN
    rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
    actual_rnn_params = sum(p.numel() for p in rnn.parameters())
    expected_rnn_params = audit_stacked_rnn_parameters("RNN", input_size, hidden_size, num_layers)
    print(f"[Stacked RNN]  Expected: {expected_rnn_params}, Actual: {actual_rnn_params}")
    assert actual_rnn_params == expected_rnn_params

    # 2. GRU
    gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
    actual_gru_params = sum(p.numel() for p in gru.parameters())
    expected_gru_params = audit_stacked_rnn_parameters("GRU", input_size, hidden_size, num_layers)
    print(f"[Stacked GRU]  Expected: {expected_gru_params}, Actual: {actual_gru_params}")
    assert actual_gru_params == expected_gru_params

    # 3. LSTM
    lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
    actual_lstm_params = sum(p.numel() for p in lstm.parameters())
    expected_lstm_params = audit_stacked_rnn_parameters("LSTM", input_size, hidden_size, num_layers)
    print(f"[Stacked LSTM] Expected: {expected_lstm_params}, Actual: {actual_lstm_params}")
    assert actual_lstm_params == expected_lstm_params

    # Check LSTM 4x ratio vs RNN
    # Layer 1: RNN has 1*(16*10 + 16*16 + 2*16) = 160 + 256 + 32 = 448
    # Layer 2: RNN has 1*(16*16 + 16*16 + 2*16) = 256 + 256 + 32 = 544
    # Total RNN = 992. Total LSTM = 4 * 992 = 3968.
    assert expected_rnn_params == 992
    assert expected_lstm_params == 3968
    assert expected_gru_params == 3 * 992
    print("PASS: Theoretical parameter formulas match PyTorch parameter counts exactly.")


def test_deep_lstm_classifier_vs_tagger():
    print("\n" + "=" * 70)
    print("TEST 3: Deep Stacked LSTM: Many-to-One vs Many-to-Many Pipelines")
    print("=" * 70)

    class DeepLSTMClassifier(nn.Module):
        """Many-to-One sequence classification model."""
        def __init__(self, d_in: int, d_h: int, d_out: int, n_layers: int):
            super().__init__()
            self.lstm = nn.LSTM(d_in, d_h, num_layers=n_layers, batch_first=True)
            self.head = nn.Linear(d_h, d_out)

        def forward(self, x):
            # x shape: [B, T, D]
            output, (h_n, c_n) = self.lstm(x)
            # output shape: [B, T, H]
            # h_n shape: [L, B, H]
            # c_n shape: [L, B, H]
            # Extract top-layer terminal hidden state
            h_terminal = h_n[-1]  # Shape: [B, H]
            logits = self.head(h_terminal)  # Shape: [B, K]
            return logits

    class DeepLSTMSequenceTagger(nn.Module):
        """Many-to-Many sequence tagging / generation model."""
        def __init__(self, d_in: int, d_h: int, d_out: int, n_layers: int):
            super().__init__()
            self.lstm = nn.LSTM(d_in, d_h, num_layers=n_layers, batch_first=True)
            self.head = nn.Linear(d_h, d_out)

        def forward(self, x):
            # x shape: [B, T, D]
            output, (h_n, c_n) = self.lstm(x)
            # Project every time step representation independently
            # head operates on the last dimension: [B, T, H] -> [B, T, K]
            logits = self.head(output)  # Shape: [B, T, K]
            return logits

    B, T, D, H, K, L = 4, 7, 12, 16, 5, 2
    x = torch.randn(B, T, D)

    # 1. Classification
    classifier = DeepLSTMClassifier(D, H, K, L)
    cls_logits = classifier(x)
    print(f"[Many-to-One Classifier] Logits Shape: {list(cls_logits.shape)} (B={B}, K={K})")
    assert cls_logits.shape == (B, K)

    # Verify softmax yields valid probabilities summing to 1.0
    probs = torch.softmax(cls_logits, dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B)), "Probabilities must sum to 1.0!"
    print(f"[Probabilities Sum] Batch check: {sums.tolist()}")

    # 2. Sequence Tagging
    tagger = DeepLSTMSequenceTagger(D, H, K, L)
    tag_logits = tagger(x)
    print(f"[Many-to-Many Tagger]     Logits Shape: {list(tag_logits.shape)} (B={B}, T={T}, K={K})")
    assert tag_logits.shape == (B, T, K)

    print("PASS: Both Many-to-One and Many-to-Many pipelines operate correctly.")


if __name__ == "__main__":
    test_deep_stacked_rnn_forward_pass()
    test_parameter_audits_across_cell_types()
    test_deep_lstm_classifier_vs_tagger()
    print("\nALL DEEP STACKED RECURRENT TESTS PASSED CLEANLY (EXIT CODE 0).")
