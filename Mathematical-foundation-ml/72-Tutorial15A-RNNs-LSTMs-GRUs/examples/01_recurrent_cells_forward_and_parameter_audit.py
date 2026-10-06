"""
Tutorial 15A: Example 1 - Recurrent Cells Forward Pass & Parameter Audit
=========================================================================
This script demonstrates:
1. Mathematical formulation of Vanilla RNN, GRU, and LSTM cell state transitions.
2. Step-by-step tensor operations for gating mechanisms (reset, update, forget, input, output).
3. Exact parameter count verification comparing mathematical theory against PyTorch modules.
4. Numerical assertion of mathematical equivalence between manual tensor math and nn.Cell modules.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def test_vanilla_rnn_cell():
    print("=== 1. Vanilla RNN Cell Mathematical Verification ===")
    torch.manual_seed(42)
    B, D, H = 2, 4, 3
    
    x = torch.randn(B, D)  # Shape: [B, D]
    h_prev = torch.randn(B, H)  # Shape: [B, H]
    
    # PyTorch reference cell
    rnn_cell = nn.RNNCell(input_size=D, hidden_size=H, nonlinearity="tanh")
    
    # Manual forward pass: h_t = tanh(x W_ih^T + b_ih + h_prev W_hh^T + b_hh)
    with torch.no_grad():
        w_ih = rnn_cell.weight_ih  # Shape: [H, D]
        w_hh = rnn_cell.weight_hh  # Shape: [H, H]
        b_ih = rnn_cell.bias_ih    # Shape: [H]
        b_hh = rnn_cell.bias_hh    # Shape: [H]
        
        # Linear projections
        proj_x = F.linear(x, w_ih, b_ih)        # Shape: [B, H]
        proj_h = F.linear(h_prev, w_hh, b_hh)  # Shape: [B, H]
        h_manual = torch.tanh(proj_x + proj_h)  # Shape: [B, H]
        
        # PyTorch cell forward
        h_pytorch = rnn_cell(x, h_prev)        # Shape: [B, H]
        
    assert torch.allclose(h_manual, h_pytorch, atol=1e-6), "Vanilla RNN manual math mismatch!"
    
    # Parameter count verification: W_ih (H*D) + W_hh (H*H) + b_ih (H) + b_hh (H)
    expected_params = H * D + H * H + 2 * H
    actual_params = sum(p.numel() for p in rnn_cell.parameters())
    assert actual_params == expected_params, f"RNN param mismatch: {actual_params} != {expected_params}"
    print(f"[PASS] Vanilla RNN Cell: Shape={list(h_pytorch.shape)}, Params={actual_params} (Expected={expected_params})")


def test_gru_cell():
    print("\n=== 2. GRU Cell Mathematical Verification ===")
    torch.manual_seed(42)
    B, D, H = 2, 4, 3
    
    x = torch.randn(B, D)  # Shape: [B, D]
    h_prev = torch.randn(B, H)  # Shape: [B, H]
    
    gru_cell = nn.GRUCell(input_size=D, hidden_size=H)
    
    with torch.no_grad():
        # GRU packs reset (r), update (z), and new candidate (n) into 3*H slices
        w_ih = gru_cell.weight_ih  # Shape: [3*H, D]
        w_hh = gru_cell.weight_hh  # Shape: [3*H, H]
        b_ih = gru_cell.bias_ih    # Shape: [3*H]
        b_hh = gru_cell.bias_hh    # Shape: [3*H]
        
        gi = F.linear(x, w_ih, b_ih)        # Shape: [B, 3*H]
        gh = F.linear(h_prev, w_hh, b_hh)  # Shape: [B, 3*H]
        
        i_r, i_z, i_n = gi.chunk(3, dim=1)  # Each Shape: [B, H]
        h_r, h_z, h_n = gh.chunk(3, dim=1)  # Each Shape: [B, H]
        
        reset_gate = torch.sigmoid(i_r + h_r)   # Shape: [B, H]
        update_gate = torch.sigmoid(i_z + h_z)  # Shape: [B, H]
        
        # Candidate state with modulated hidden context
        cand_state = torch.tanh(i_n + reset_gate * h_n)  # Shape: [B, H]
        
        # Convex combination update
        h_manual = (1.0 - update_gate) * cand_state + update_gate * h_prev  # Shape: [B, H]
        h_pytorch = gru_cell(x, h_prev)  # Shape: [B, H]
        
    assert torch.allclose(h_manual, h_pytorch, atol=1e-6), "GRU manual math mismatch!"
    
    # 3 gates: 3 * [H*D + H*H + 2*H]
    expected_params = 3 * (H * D + H * H + 2 * H)
    actual_params = sum(p.numel() for p in gru_cell.parameters())
    assert actual_params == expected_params, f"GRU param mismatch: {actual_params} != {expected_params}"
    print(f"[PASS] GRU Cell: Shape={list(h_pytorch.shape)}, Params={actual_params} (Expected={expected_params})")


def test_lstm_cell():
    print("\n=== 3. LSTM Cell Mathematical Verification ===")
    torch.manual_seed(42)
    B, D, H = 2, 4, 3
    
    x = torch.randn(B, D)  # Shape: [B, D]
    h_prev = torch.randn(B, H)  # Shape: [B, H]
    c_prev = torch.randn(B, H)  # Shape: [B, H]
    
    lstm_cell = nn.LSTMCell(input_size=D, hidden_size=H)
    
    with torch.no_grad():
        # LSTM packs input (i), forget (f), candidate (g/c), output (o) into 4*H slices
        w_ih = lstm_cell.weight_ih  # Shape: [4*H, D]
        w_hh = lstm_cell.weight_hh  # Shape: [4*H, H]
        b_ih = lstm_cell.bias_ih    # Shape: [4*H]
        b_hh = lstm_cell.bias_hh    # Shape: [4*H]
        
        gates = F.linear(x, w_ih, b_ih) + F.linear(h_prev, w_hh, b_hh)  # Shape: [B, 4*H]
        i_gate, f_gate, c_cand, o_gate = gates.chunk(4, dim=1)          # Each Shape: [B, H]
        
        i = torch.sigmoid(i_gate)   # Shape: [B, H]
        f = torch.sigmoid(f_gate)   # Shape: [B, H]
        c_tilde = torch.tanh(c_cand)  # Shape: [B, H]
        o = torch.sigmoid(o_gate)   # Shape: [B, H]
        
        # Additive cell state update
        c_manual = f * c_prev + i * c_tilde    # Shape: [B, H]
        # Emitted hidden state
        h_manual = o * torch.tanh(c_manual)    # Shape: [B, H]
        
        h_pytorch, c_pytorch = lstm_cell(x, (h_prev, c_prev))  # Shape: [B, H], [B, H]
        
    assert torch.allclose(h_manual, h_pytorch, atol=1e-6), "LSTM hidden state mismatch!"
    assert torch.allclose(c_manual, c_pytorch, atol=1e-6), "LSTM cell state mismatch!"
    
    # 4 gates: 4 * [H*D + H*H + 2*H]
    expected_params = 4 * (H * D + H * H + 2 * H)
    actual_params = sum(p.numel() for p in lstm_cell.parameters())
    assert actual_params == expected_params, f"LSTM param mismatch: {actual_params} != {expected_params}"
    print(f"[PASS] LSTM Cell: Shapes=h:{list(h_pytorch.shape)}, c:{list(c_pytorch.shape)}, Params={actual_params} (Expected={expected_params})")


if __name__ == "__main__":
    test_vanilla_rnn_cell()
    test_gru_cell()
    test_lstm_cell()
    print("\nAll Recurrent Cell forward operations and parameter counts passed successfully!")
