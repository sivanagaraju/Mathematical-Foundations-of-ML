"""
Tutorial 15A: Example 2 - PyTorch Sequence Classification Pipeline
===================================================================
This script demonstrates:
1. End-to-end Sequence Classifier architecture using PyTorch recurrent backbones (RNN, GRU, LSTM).
2. Proper usage of `batch_first=True` with 3D sequence tensors [Batch, SeqLen, InDim].
3. Extracting terminal hidden representation h_T for many-to-one classification.
4. Loss computation, backpropagation through time, and gradient clipping to stabilize training.
5. Verification of tensor shapes throughout the network pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceClassifier(nn.Module):
    def __init__(self, cell_type: str, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.cell_type = cell_type.upper()
        if self.cell_type == "RNN":
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type}")
            
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x Shape: [B, T, D]
        if self.cell_type == "LSTM":
            out_seq, (h_n, c_n) = self.rnn(x)  # out_seq: [B, T, H], h_n: [1, B, H], c_n: [1, B, H]
        else:
            out_seq, h_n = self.rnn(x)         # out_seq: [B, T, H], h_n: [1, B, H]
            
        # Terminal hidden state h_T: out_seq[:, -1, :] is identical to h_n.squeeze(0)
        h_t = out_seq[:, -1, :]  # Shape: [B, H]
        logits = self.head(h_t)   # Shape: [B, num_classes]
        return logits, out_seq


def test_pipeline_for_cell(cell_type: str):
    print(f"\n--- Testing SequenceClassifier with {cell_type} ---")
    torch.manual_seed(42)
    B, T, D, H, C = 4, 8, 16, 24, 3
    
    x = torch.randn(B, T, D)           # Shape: [B, T, D]
    targets = torch.randint(0, C, (B,))  # Shape: [B]
    
    model = SequenceClassifier(cell_type, input_dim=D, hidden_dim=H, num_classes=C)
    model.train()
    
    logits, out_seq = model(x)  # logits: [B, C], out_seq: [B, T, H]
    
    # Assert tensor shapes
    assert logits.shape == (B, C), f"Logits shape mismatch: {logits.shape}"
    assert out_seq.shape == (B, T, H), f"Output sequence shape mismatch: {out_seq.shape}"
    
    # Loss computation
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)  # Shape: scalar []
    assert torch.isfinite(loss), "Loss must be finite"
    
    # Backpropagation
    loss.backward()
    
    # Verify gradients exist and are non-zero
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
        
    # Gradient clipping test
    max_norm = 1.0
    total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    assert total_norm > 0.0, "Total grad norm should be positive"
    
    probs = F.softmax(logits, dim=-1)  # Shape: [B, C]
    predictions = torch.argmax(probs, dim=-1)  # Shape: [B]
    
    print(f"[PASS] {cell_type}: Loss={loss.item():.4f}, GradNorm={total_norm:.4f}, Preds={predictions.tolist()}")


if __name__ == "__main__":
    for cell in ["RNN", "GRU", "LSTM"]:
        test_pipeline_for_cell(cell)
    print("\nAll PyTorch Sequence Classification pipelines verified successfully!")
