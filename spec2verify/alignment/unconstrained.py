"""Constrained vs. unconstrained alignment (Appendix E.1, Table 25).

Agreement only (subj_num at the first-block site, verb_form at the compositional
site, both at the main verb).  DAS uses rank 2.  The unconstrained maps use a
learned encoder E and decoder D and set the first two encoded coordinates from
the source:

    h' = h_b + D(E(h_b) + M(E(h_s) - E(h_b))) - D(E(h_b)),

so that source = base returns h_b exactly.  Linear: E and D are independent
affine 128 -> 128 maps (not constrained to be inverses or orthogonal).  MLP: E and
D are 128 -> 256 -> 128 with GELU and no output activation.  All methods share
pairs, sites, the optimizer of Appendix C.3 (Adam, lr 5e-3, 100 epochs, batch
256, gradient clip 1.0, epoch chosen on dev IIA) and the frozen network; the
random network is the step-0 checkpoint of each training seed.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.pairs import Pair
from .backbone import Backbone


class EncoderDecoderMap(nn.Module):
    def __init__(self, d: int, kind: str, k: int = 2, hidden: int = 256):
        super().__init__()
        if kind == "linear":
            self.E, self.D = nn.Linear(d, d), nn.Linear(d, d)
        elif kind == "mlp":
            self.E = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
            self.D = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        else:
            raise ValueError(kind)
        self.k = k

    def forward(self, hb: torch.Tensor, hs: torch.Tensor) -> torch.Tensor:
        eb, es = self.E(hb), self.E(hs)
        m = torch.zeros_like(eb); m[:, :self.k] = 1.0
        return hb + self.D(eb + m * (es - eb)) - self.D(eb)


def _iia(bb, pairs, site, fmap) -> float:
    with torch.no_grad():
        hb = bb.vectors([p.base for p in pairs], site); hs = bb.vectors([p.source for p in pairs], site)
        y = torch.tensor([p.label for p in pairs], device=hb.device)
        return (bb.patched_logits([p.base for p in pairs], site, fmap(hb, hs)).argmax(-1) == y).float().mean().item()


def fit_map(bb: Backbone, split: Dict[str, Sequence[Pair]], site: str, kind: str, seed: int,
            epochs: int = 100, batch_size: int = 256, lr: float = 5e-3) -> Tuple[float, float]:
    """Returns (dev IIA, test IIA) at the best development epoch."""
    torch.manual_seed(seed)
    tr = split["train"]
    hb = bb.vectors([p.base for p in tr], site); hs = bb.vectors([p.source for p in tr], site)
    y = torch.tensor([p.label for p in tr], device=hb.device)
    fmap = EncoderDecoderMap(bb.d, kind).to(hb.device)
    opt = torch.optim.Adam(fmap.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    g = torch.Generator().manual_seed(seed)
    best = (-1.0, None)
    for _ in range(epochs):
        perm = torch.randperm(len(tr), generator=g)
        for i in range(0, len(tr), batch_size):
            idx = perm[i:i + batch_size]
            logits = bb.patched_logits([tr[j].base for j in idx.tolist()], site, fmap(hb[idx], hs[idx]))
            loss = F.cross_entropy(logits, y[idx.to(hb.device)])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(fmap.parameters(), 1.0); opt.step()
        dev = _iia(bb, split["dev"], site, fmap)
        if dev > best[0]:
            best = (dev, {k: v.clone() for k, v in fmap.state_dict().items()})
    fmap.load_state_dict(best[1])
    return best[0], _iia(bb, split["test"], site, fmap)
