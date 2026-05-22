"""2-layer Transformer for from-scratch training (Appendix C.1).
Architecture: d=128, H=4, GELU, learned positional embeddings, Xavier init.
Total parameters: ~0.42M.
"""
import torch, torch.nn as nn
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size=135, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, max_seq_len=30, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, 2)
        self._init_weights()
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    def forward(self, input_ids, target_pos=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = self.encoder(x)
        x = x[torch.arange(B), target_pos] if target_pos is not None else x[:, -1]
        return self.classifier(x)
    def get_residual_stream(self, input_ids, layer, target_pos):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        for i, mod in enumerate(self.encoder.layers):
            x = mod(x)
            if i == layer: return x[torch.arange(B), target_pos]
        return x[torch.arange(B), target_pos]
