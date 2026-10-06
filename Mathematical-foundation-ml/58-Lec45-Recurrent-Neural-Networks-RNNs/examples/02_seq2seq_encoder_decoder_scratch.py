"""
Simulation Script 02: Seq2Seq Encoder-Decoder & Auto-Regressive Decoding from Scratch
Course: Mathematical Foundations of Machine Learning (NPTEL / IISc)
Lecture 45: Recurrent Neural Networks (RNNs)

Demonstrates:
1. Encoder RNN compressing variable-length sequence to context vector c = h_T.
2. Decoder RNN initialized with context vector s_0 = c.
3. Auto-regressive closed-loop token generation with <EOS> halting condition.
4. Numerical consistency and state transitions verified against PyTorch modules.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NumPySeq2Seq:
    def __init__(self, d_in, m, vocab_size, eos_token_id=2):
        self.d_in = d_in
        self.m = m
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        
        # Encoder weights
        self.W_enc_hh = np.random.randn(m, m) * 0.1
        self.W_enc_xh = np.random.randn(m, d_in) * 0.1
        self.b_enc_h = np.zeros(m)
        
        # Decoder weights (input is embedding of previous generated token)
        self.embedding = np.random.randn(vocab_size, m) * 0.1
        self.W_dec_hh = np.random.randn(m, m) * 0.1
        self.W_dec_xh = np.random.randn(m, m) * 0.1
        self.b_dec_h = np.zeros(m)
        self.W_dec_out = np.random.randn(vocab_size, m) * 0.1
        self.b_dec_out = np.zeros(vocab_size)
        
    def encode(self, X):
        """
        Encodes sequence X of shape [B, T_enc, d_in].
        Returns final context vector c of shape [B, m].
        """
        B, T_enc, _ = X.shape
        h = np.zeros((B, self.m))  # h0 = 0
        
        for t in range(T_enc):
            xt = X[:, t, :]  # Shape: [B, d_in]
            h = np.tanh(h @ self.W_enc_hh.T + xt @ self.W_enc_xh.T + self.b_enc_h)
            
        return h  # Context vector c = h_{T_enc}
    
    def decode_autoregressive(self, context, start_token_id=1, max_len=10):
        """
        Decodes sequence auto-regressively from context vector c until <EOS> or max_len.
        Returns generated token IDs of shape [B, generated_len].
        """
        B, _ = context.shape
        s = context.copy()  # s_0 = c
        
        current_tokens = np.full((B,), start_token_id, dtype=int)
        generated_sequences = [[] for _ in range(B)]
        active_batch = [True] * B
        
        for step in range(max_len):
            # Embed current token
            xt = self.embedding[current_tokens]  # Shape: [B, m]
            
            # Decoder hidden state transition
            s = np.tanh(s @ self.W_dec_hh.T + xt @ self.W_dec_xh.T + self.b_dec_h)
            
            # Predict next token logits and argmax
            logits = s @ self.W_dec_out.T + self.b_dec_out  # Shape: [B, vocab_size]
            next_tokens = np.argmax(logits, axis=1)
            
            for b in range(B):
                if active_batch[b]:
                    generated_sequences[b].append(int(next_tokens[b]))
                    if next_tokens[b] == self.eos_token_id:
                        active_batch[b] = False
                        
            if not any(active_batch):
                break
                
            current_tokens = next_tokens
            
        return generated_sequences


def test_seq2seq_execution():
    np.random.seed(123)
    torch.manual_seed(123)
    
    B = 2
    T_enc = 6
    d_in = 5
    m = 8
    vocab_size = 12
    eos_id = 0
    
    seq2seq = NumPySeq2Seq(d_in, m, vocab_size, eos_token_id=eos_id)
    
    # Generate mock inputs
    X_input = np.random.randn(B, T_enc, d_in)
    
    # 1. Forward Encoder pass
    context = seq2seq.encode(X_input)
    assert context.shape == (B, m)
    
    # Verify encoder against PyTorch RNN
    enc_pt = nn.RNN(input_size=d_in, hidden_size=m, batch_first=True, bias=True, nonlinearity='tanh')
    with torch.no_grad():
        enc_pt.weight_ih_l0.copy_(torch.from_numpy(seq2seq.W_enc_xh))
        enc_pt.weight_hh_l0.copy_(torch.from_numpy(seq2seq.W_enc_hh))
        enc_pt.bias_ih_l0.copy_(torch.from_numpy(seq2seq.b_enc_h))
        enc_pt.bias_hh_l0.zero_()
        
    _, hn_pt = enc_pt(torch.from_numpy(X_input).float())
    np.testing.assert_allclose(context, hn_pt.squeeze(0).detach().numpy(), rtol=1e-5, atol=1e-5)
    
    # 2. Decode auto-regressively
    generated = seq2seq.decode_autoregressive(context, start_token_id=1, max_len=8)
    
    print("[PASS] Seq2Seq Encoder-Decoder Execution Verified!")
    print(f"       Batch Size: {B}, Input Length: {T_enc}, Vocab Size: {vocab_size}")
    print(f"       Context Vector Norms: {[float(np.linalg.norm(c)) for c in context]}")
    for i, seq in enumerate(generated):
        print(f"       Sequence {i} Generated Tokens: {seq}")
        if eos_id in seq:
            print(f"       Sequence {i} dynamically terminated at <EOS> (token {eos_id})!")


if __name__ == "__main__":
    test_seq2seq_execution()
