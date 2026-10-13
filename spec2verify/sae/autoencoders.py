"""Sparse autoencoders (Section 5.4; Appendices C.5, D.5, D.10, F.5; Table 10).

TopK (K = 32), JumpReLU and Gated SAEs, independent PyTorch implementations of
the published formulations, trained on one target-position activation per
training sentence at the alignment site:

* inputs are centered and divided by one scalar s = sqrt(mean ||h - mu||^2 / d), so
  the mean squared coordinate is 1; dev/test reuse mu and s;
* Adam (lr 1e-3, betas 0.9/0.999, eps 1e-8, no weight decay), 50,000 updates of
  batch 4,096 drawn with replacement; dictionary 8d (expansion ablation 4/8/16/32);
* decoder columns are renormalized to unit norm after every update (untied
  encoder and decoder);
* TopK:      a = W_enc(x - b_dec) + b_enc;  z = TopK(ReLU(a), 32);  x^ = W_dec z + b_dec;
* JumpReLU:  z_i = a_i 1[a_i > theta_i], theta = exp(eta) (initially 0.01); loss adds
             lambda * sum_i 1[a_i > theta_i] with a rectangle-kernel straight-through
             estimator of bandwidth 0.001;
* Gated:     z = 1[pi_gate > 0] * ReLU(W_mag(x - b_dec) + b_mag), W_mag = diag(exp(r)) W_gate;
             loss adds lambda * ||ReLU(pi_gate)||_1 and an auxiliary reconstruction from
             ReLU(pi_gate) through a stop-gradient decoder (weight 1); L0 counts open gates.

The sparsity coefficient is swept over {1, 3} x 10^{-6..-3} and chosen on development
data as the value whose mean L0 is closest to 32 (ties: lower MSE, then smaller
lambda); at 8x this gives 1e-4 for JumpReLU and 3e-4 for Gated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = ("topk", "jumprelu", "gated")
EXPANSIONS = (4, 8, 16, 32)
LAMBDA_GRID = tuple(m * 10.0 ** e for e in (-6, -5, -4, -3) for m in (1, 3))
SELECTED_LAMBDA = {"jumprelu": 1e-4, "gated": 3e-4}
PHENOMENON_INDEX = {"agreement": 0, "npi": 1, "binding": 2, "concord": 3}


def sae_seed(model_seed: int, phenomenon: str, layer: int, arch: str, expansion: int) -> int:
    return (model_seed + 100_000 + 1_000 * PHENOMENON_INDEX[phenomenon] + 100 * layer
            + 10 * ARCHS.index(arch) + EXPANSIONS.index(expansion))


class Normalizer:
    def __init__(self, H: torch.Tensor):
        self.mu = H.mean(0)
        self.s = torch.sqrt(((H - self.mu) ** 2).sum(1).mean() / H.shape[1])

    def __call__(self, H):
        return (H - self.mu) / self.s

    def inverse(self, X):
        return X * self.s + self.mu


class _Step(torch.autograd.Function):
    """Heaviside step with a rectangle-kernel pseudo-derivative w.r.t. the threshold."""

    @staticmethod
    def forward(ctx, a, theta, bw):
        ctx.save_for_backward(a, theta); ctx.bw = bw
        return (a > theta).to(a.dtype)

    @staticmethod
    def backward(ctx, g):
        a, theta = ctx.saved_tensors
        k = ((a - theta).abs() < ctx.bw / 2).to(a.dtype) / ctx.bw
        return None, (-g * k).sum(0), None


class _JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, theta, bw):
        ctx.save_for_backward(a, theta); ctx.bw = bw
        return a * (a > theta).to(a.dtype)

    @staticmethod
    def backward(ctx, g):
        a, theta = ctx.saved_tensors
        gate = (a > theta).to(a.dtype)
        k = ((a - theta).abs() < ctx.bw / 2).to(a.dtype) / ctx.bw
        return g * gate, (-g * theta * k).sum(0), None


class SAE(nn.Module):
    def __init__(self, d: int, n_features: int, arch: str, k: int = 32, lam: float = 0.0,
                 bandwidth: float = 1e-3, theta0: float = 0.01):
        super().__init__()
        self.arch, self.k, self.lam, self.bw = arch, k, lam, bandwidth
        self.W_dec = nn.Parameter(torch.randn(d, n_features) / math.sqrt(d))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.normalize_decoder()
        if arch == "gated":
            self.W_gate = nn.Parameter(self.W_dec.data.T.clone())
            self.b_gate = nn.Parameter(torch.zeros(n_features))
            self.r_mag = nn.Parameter(torch.zeros(n_features))
            self.b_mag = nn.Parameter(torch.zeros(n_features))
        else:
            self.W_enc = nn.Parameter(self.W_dec.data.T.clone())
            self.b_enc = nn.Parameter(torch.zeros(n_features))
        if arch == "jumprelu":
            self.log_theta = nn.Parameter(torch.full((n_features,), math.log(theta0)))

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xc = x - self.b_dec
        if self.arch == "topk":
            a = F.relu(xc @ self.W_enc.T + self.b_enc)
            v, i = a.topk(self.k, dim=-1)
            return torch.zeros_like(a).scatter_(-1, i, v)
        if self.arch == "jumprelu":
            a = xc @ self.W_enc.T + self.b_enc
            return _JumpReLU.apply(a, self.log_theta.exp(), self.bw)
        pi = xc @ self.W_gate.T + self.b_gate
        mag = F.relu(xc @ (self.W_gate * self.r_mag.exp()[:, None]).T + self.b_mag)
        return (pi > 0).to(x.dtype) * mag

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.T + self.b_dec

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        xc = x - self.b_dec
        if self.arch == "topk":
            z = self.encode(x); xh = self.decode(z)
            mse = ((xh - x) ** 2).mean(-1).mean(); l0 = (z > 0).float().sum(-1).mean()
            return mse, {"mse": mse.item(), "l0": l0.item()}
        if self.arch == "jumprelu":
            a = xc @ self.W_enc.T + self.b_enc
            theta = self.log_theta.exp()
            z = _JumpReLU.apply(a, theta, self.bw); xh = self.decode(z)
            mse = ((xh - x) ** 2).mean(-1).mean()
            l0 = _Step.apply(a, theta, self.bw).sum(-1).mean()
            return mse + self.lam * l0, {"mse": mse.item(), "l0": l0.item()}
        pi = xc @ self.W_gate.T + self.b_gate
        mag = F.relu(xc @ (self.W_gate * self.r_mag.exp()[:, None]).T + self.b_mag)
        z = (pi > 0).to(x.dtype) * mag; xh = self.decode(z)
        mse = ((xh - x) ** 2).mean(-1).mean()
        sparsity = F.relu(pi).sum(-1).mean()
        aux = ((F.relu(pi) @ self.W_dec.detach().T + self.b_dec.detach() - x) ** 2).mean(-1).mean()
        l0 = (pi > 0).float().sum(-1).mean()
        return mse + self.lam * sparsity + aux, {"mse": mse.item(), "l0": l0.item()}


def train_sae(H: torch.Tensor, arch: str, expansion: int = 8, seed: int = 0, steps: int = 50_000,
              batch_size: int = 4_096, lr: float = 1e-3, lam: Optional[float] = None, k: int = 32
              ) -> Tuple[SAE, Normalizer]:
    torch.manual_seed(seed)
    norm = Normalizer(H)
    X = norm(H)
    d = X.shape[1]
    sae = SAE(d, expansion * d, arch, k=k, lam=lam if lam is not None else SELECTED_LAMBDA.get(arch, 0.0)).to(X.device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, X.shape[0], (batch_size,), generator=g).to(X.device)
        loss, _ = sae.loss(X[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        sae.normalize_decoder()
    return sae.eval(), norm


@torch.no_grad()
def quality(sae: SAE, norm: Normalizer, H: torch.Tensor) -> Dict[str, float]:
    """Held-out reconstruction MSE (normalized units, mean over coordinates), mean L0 and
    the share of features that never fire."""
    X = norm(H); Z = sae.encode(X)
    return {"mse": ((sae.decode(Z) - X) ** 2).mean().item(), "l0": (Z > 0).float().sum(1).mean().item(),
            "dead": ((Z > 0).sum(0) == 0).float().mean().item()}


def select_lambda(H_train: torch.Tensor, H_dev: torch.Tensor, arch: str, expansion: int, seed: int,
                  steps: int = 50_000) -> float:
    """Mean L0 closest to 32 on dev; ties: lower MSE, then smaller lambda."""
    rows = []
    for lam in LAMBDA_GRID:
        sae, norm = train_sae(H_train, arch, expansion, seed, steps=steps, lam=lam)
        q = quality(sae, norm, H_dev)
        rows.append((abs(q["l0"] - 32), q["mse"], lam))
    return sorted(rows)[0][2]
