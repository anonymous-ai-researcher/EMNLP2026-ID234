"""Cayley parameterization for orthogonal rotation matrices (Appendix C.3).

R = (I - A)(I + A)^{-1} where A is skew-symmetric, guaranteeing R in O(d).
"""
import torch
import torch.nn as nn

class CayleyRotation(nn.Module):
    def __init__(self, d: int, rank: int):
        super().__init__()
        self.d = d
        self.rank = rank
        self.A = nn.Parameter(torch.randn(d, d) * 0.01)
    
    def forward(self):
        A_skew = self.A - self.A.T
        I = torch.eye(self.d, device=self.A.device)
        R = torch.linalg.solve(I + A_skew, I - A_skew)
        return R[:, :self.rank]
    
    def get_subspace(self):
        return self.forward()
