"""Scoring SAE features and neurons with interchange interventions (Appendix C.5).

For k units selected on the training split by absolute correlation with the
variable U (ties: smaller index), base and source activations are encoded, the
base values of these units are replaced by the source values, the result is
decoded and the base reconstruction error h_b - dec(enc(h_b)) is added back, so
an empty set leaves the activation unchanged; IIA is then computed exactly as
for DAS.  For neurons the units are coordinates of the residual stream.

Also here: the largest single-unit correlation with its permutation null
(1,000 relabelings with the full selection repeated; p = (b + 1) / 1001), and
"SAE span": DAS restricted to the span of the 32 selected decoder directions.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..alignment.backbone import Backbone
from ..alignment.das import Rotation
from ..data.pairs import Pair
from .autoencoders import SAE, Normalizer

PERMUTATION_SEED = 17012026


def abs_correlation(Z: np.ndarray, u: np.ndarray) -> np.ndarray:
    Zc = Z - Z.mean(0); uc = u - u.mean()
    den = np.sqrt((Zc ** 2).sum(0) * (uc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, (Zc * uc[:, None]).sum(0) / den, 0.0)
    return np.abs(r)


def select_units(Z: np.ndarray, u: np.ndarray, k: int) -> np.ndarray:
    r = abs_correlation(Z, u)
    order = np.lexsort((np.arange(len(r)), -r))      # descending |r|, ties to smaller index
    return order[:k]


@torch.no_grad()
def feature_patch(sae: SAE, norm: Normalizer, hb: torch.Tensor, hs: torch.Tensor, units) -> torch.Tensor:
    xb, xs = norm(hb), norm(hs)
    zb, zs = sae.encode(xb), sae.encode(xs)
    err = xb - sae.decode(zb)
    z = zb.clone()
    if len(units):
        z[:, units] = zs[:, units]
    return norm.inverse(sae.decode(z) + err)


@torch.no_grad()
def neuron_patch(hb: torch.Tensor, hs: torch.Tensor, units) -> torch.Tensor:
    h = hb.clone()
    if len(units):
        h[:, units] = hs[:, units]
    return h


@torch.no_grad()
def interchange_iia(bb: Backbone, pairs: Sequence[Pair], site: str, patch) -> float:
    hb = bb.vectors([p.base for p in pairs], site); hs = bb.vectors([p.source for p in pairs], site)
    y = torch.tensor([p.label for p in pairs], device=hb.device)
    return (bb.patched_logits([p.base for p in pairs], site, patch(hb, hs)).argmax(-1) == y).float().mean().item()


def max_correlation_test(Z: np.ndarray, u: np.ndarray, n_perm: int = 1_000, seed: int = PERMUTATION_SEED):
    """Largest |Pearson r| of a single unit, and its permutation p-value."""
    obs = abs_correlation(Z, u).max()
    rng = np.random.default_rng(seed)
    b = sum(abs_correlation(Z, rng.permutation(u)).max() >= obs for _ in range(n_perm))
    return float(obs), (b + 1) / (n_perm + 1)


def span_das(bb: Backbone, split: Dict[str, Sequence[Pair]], site: str, sae: SAE, norm: Normalizer,
             units, rank: int = 2, seed: int = 701, epochs: int = 100, batch_size: int = 256, lr: float = 5e-3):
    """DAS whose subspace lies in the span of the selected decoder directions (mapped back to
    activation units)."""
    D = (sae.W_dec[:, units] * norm.s).detach()
    Q, _ = torch.linalg.qr(D)                             # (d, m) orthonormal basis
    rot = Rotation(Q.shape[1], seed).to(Q.device)
    opt = torch.optim.Adam(rot.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    tr = split["train"]
    hb = bb.vectors([p.base for p in tr], site); hs = bb.vectors([p.source for p in tr], site)
    y = torch.tensor([p.label for p in tr], device=hb.device)

    def patch(hb_, hs_, R):
        B = (R[:rank] @ Q.T)                               # rank x d, orthonormal rows
        return hb_ + ((hs_ - hb_) @ B.T) @ B

    g = torch.Generator().manual_seed(seed); best = (-1.0, None)
    for _ in range(epochs):
        perm = torch.randperm(len(tr), generator=g)
        for i in range(0, len(tr), batch_size):
            idx = perm[i:i + batch_size]
            logits = bb.patched_logits([tr[j].base for j in idx.tolist()], site, patch(hb[idx], hs[idx], rot.matrix()))
            loss = F.cross_entropy(logits, y[idx.to(hb.device)])
            opt.zero_grad(); loss.backward(); opt.step()
        R = rot.matrix().detach()
        dev = interchange_iia(bb, split["dev"], site, lambda a, b: patch(a, b, R))
        if dev > best[0]:
            best = (dev, R)
    R = best[1]
    return best[0], interchange_iia(bb, split["test"], site, lambda a, b: patch(a, b, R))
