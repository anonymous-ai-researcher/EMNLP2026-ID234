"""TopK Sparse Autoencoder (Appendix C.5).

Architecture: encoder maps d -> E*d with TopK activation (K=32).
Trained on residual-stream activations from the from-scratch Transformer.
"""
import torch
import torch.nn as nn

class TopKSAE(nn.Module):
    def __init__(self, d_model=128, expansion=8, k=32):
        super().__init__()
        self.d_dict = d_model * expansion  # 1024
        self.k = k
        self.encoder = nn.Linear(d_model, self.d_dict)
        self.decoder = nn.Linear(self.d_dict, d_model)
    
    def encode(self, x):
        z = self.encoder(x)
        topk_vals, topk_idx = torch.topk(z, self.k, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(-1, topk_idx, 1.0)
        return z * mask
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z
    
    def reconstruction_loss(self, x, x_hat):
        return ((x - x_hat) ** 2).mean()
