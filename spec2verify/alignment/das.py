"""Distributed Alignment Search (Section 2.3, Eq. 2; Appendix C.3).

R = R0 (I - Omega)(I + Omega)^{-1} with Omega = A - A^T (Cayley transform; A is
trainable, initialized at zero), where R0 is the orthogonal factor of a seeded
Gaussian matrix with the signs of its QR factor fixed so that diag(R) >= 0.  The
interchange is h_b + R^T Pi_r R (h_s - h_b), with Pi_r selecting the first r
coordinates.  A is optimized with Adam (betas 0.9/0.999, eps 1e-8, lr 5e-3, no
weight decay, no schedule, gradient-norm clip 1.0) for 100 epochs of batch 256;
the epoch with the highest development IIA is kept (the earlier one on ties).
The network is frozen.

The (rank, layer) search fits every candidate and selects by development IIA,
breaking ties toward smaller ranks, then shallower layers; the selected
configuration is reported on the test portion.  Each DAS initialization makes
its own selection.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.pairs import Pair
from .backbone import Backbone

DAS_SEEDS = (701, 809, 907, 1013, 1109)
RANKS = (1, 2, 4, 8)


class Rotation(nn.Module):
    def __init__(self, d: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        q, r = torch.linalg.qr(torch.randn(d, d, generator=g))
        q = q * torch.sign(torch.diagonal(r)).where(torch.diagonal(r) != 0, torch.ones(d))[None]
        self.register_buffer("R0", q)
        self.A = nn.Parameter(torch.zeros(d, d))

    def matrix(self) -> torch.Tensor:
        omega = self.A - self.A.T
        eye = torch.eye(self.A.shape[0], device=self.A.device, dtype=self.A.dtype)
        # X = (I - Omega)(I + Omega)^{-1}  <=>  (I + Omega)^T X^T = (I - Omega)^T
        x_t = torch.linalg.solve((eye + omega).T, (eye - omega).T)
        return self.R0 @ x_t.T


def interchange(hb: torch.Tensor, hs: torch.Tensor, R: torch.Tensor, coords: slice) -> torch.Tensor:
    """h_b + R^T Pi R (h_s - h_b) for the coordinates in ``coords``."""
    Rr = R[coords]
    return hb + ((hs - hb) @ Rr.T) @ Rr


@dataclass
class DASResult:
    site: str
    rank: int
    seed: int
    dev_iia: float
    test_iia: float
    epoch: int
    R: Optional[torch.Tensor] = None
    extra: Dict = field(default_factory=dict)


def _labels(pairs: Sequence[Pair], device) -> torch.Tensor:
    return torch.tensor([p.label for p in pairs], device=device)


def _vectors(bb: Backbone, pairs: Sequence[Pair], site: str):
    bases = [p.base for p in pairs]; sources = [p.source for p in pairs]
    return bases, bb.vectors(bases, site), bb.vectors(sources, site)


@torch.no_grad()
def evaluate(bb: Backbone, pairs: Sequence[Pair], site: str, R: torch.Tensor, rank: int,
             labels: Optional[torch.Tensor] = None, bs: int = 512) -> float:
    if not pairs:
        return float("nan")
    bases, hb, hs = _vectors(bb, pairs, site)
    y = labels if labels is not None else _labels(pairs, hb.device)
    correct = 0
    for i in range(0, len(pairs), bs):
        new = interchange(hb[i:i + bs], hs[i:i + bs], R.to(hb.device), slice(0, rank))
        pred = bb.patched_logits(bases[i:i + bs], site, new).argmax(-1)
        correct += (pred == y[i:i + bs]).sum().item()
    return correct / len(pairs)


def fit(bb: Backbone, train: Sequence[Pair], dev: Sequence[Pair], site: str, rank: int, seed: int,
        epochs: int = 100, batch_size: int = 256, lr: float = 5e-3, grad_clip: float = 1.0,
        train_labels: Optional[torch.Tensor] = None, init: Optional[Rotation] = None
        ) -> Tuple[torch.Tensor, float, int]:
    """Fit one rotation; returns (R at the best development epoch, its dev IIA, the epoch)."""
    bases, hb, hs = _vectors(bb, train, site)
    device = hb.device
    y = (train_labels if train_labels is not None else _labels(train, device)).to(device)
    rot = (init or Rotation(bb.d, seed)).to(device)
    opt = torch.optim.Adam(rot.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    best = (-1.0, 0, rot.matrix().detach().clone())
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(train), generator=g)
        for i in range(0, len(train), batch_size):
            idx = perm[i:i + batch_size]
            R = rot.matrix()
            new = interchange(hb[idx], hs[idx], R, slice(0, rank))
            logits = bb.patched_logits([bases[j] for j in idx.tolist()], site, new)
            loss = F.cross_entropy(logits, y[idx.to(device)])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(rot.parameters(), grad_clip)
            opt.step()
        R = rot.matrix().detach()
        dev_iia = evaluate(bb, dev, site, R, rank)
        if dev_iia > best[0]:
            best = (dev_iia, epoch, R.clone())
    return best[2], best[0], best[1]


def search(bb: Backbone, split: Dict[str, Sequence[Pair]], seed: int, sites: Optional[Sequence[str]] = None,
           ranks: Sequence[int] = RANKS, **kw) -> DASResult:
    """Fit every (site, rank); select on dev IIA (ties: smaller rank, then shallower site)."""
    sites = list(sites or bb.sites)
    cands = []
    for si, site in enumerate(sites):
        for rank in ranks:
            R, dev_iia, ep = fit(bb, split["train"], split["dev"], site, rank, seed, **kw)
            cands.append((-dev_iia, rank, si, site, R, ep))
    cands.sort(key=lambda c: (c[0], c[1], c[2]))
    neg_dev, rank, _, site, R, ep = cands[0]
    test_iia = evaluate(bb, split["test"], site, R, rank)
    res = DASResult(site, rank, seed, -neg_dev, test_iia, ep, R.cpu())
    res.extra["all"] = [(c[3], c[1], -c[0]) for c in cands]
    return res


def refit_fixed(bb: Backbone, split: Dict[str, Sequence[Pair]], site: str, rank: int, seed: int,
                shuffle_labels: bool = False, **kw) -> DASResult:
    """Competing specifications and negative controls reuse the site and rank selected for
    the prescribed specification and refit only R (Appendix C.3).  ``shuffle_labels``
    permutes the counterfactual labels of the training pairs (negative control iii)."""
    y = None
    if shuffle_labels:
        lab = _labels(split["train"], "cpu")
        y = lab[torch.randperm(len(lab), generator=torch.Generator().manual_seed(seed))]
    R, dev_iia, ep = fit(bb, split["train"], split["dev"], site, rank, seed, train_labels=y, **kw)
    return DASResult(site, rank, seed, dev_iia, evaluate(bb, split["test"], site, R, rank), ep, R.cpu())


@torch.no_grad()
def iia_by_stratum(bb: Backbone, pairs: Sequence[Pair], res: DASResult, key=lambda p: p.load) -> Dict:
    groups: Dict = {}
    for p in pairs:
        groups.setdefault(key(p), []).append(p)
    return {k: evaluate(bb, v, res.site, res.R, res.rank) for k, v in sorted(groups.items())}
