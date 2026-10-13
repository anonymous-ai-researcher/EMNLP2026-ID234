"""Training trajectories (Section 5.3; Appendices C.1, E.2, E.7, F.3; Figures 3 and 5).

Per-variable IIA is computed at step 0 and every 500 steps with the site and
rank selected at the final checkpoint; R is refit (not warm-started) at every
checkpoint from the first three DAS seeds (701, 809, 907) and the three runs are
averaged.  A variable *crosses* at the first checkpoint whose mean IIA is
strictly above 0.90; a run that never crosses is right-censored, and a lag is
defined only when both variables cross.

Excluded loss (Figure 5): logistic directions for subject number and verb number
are refit at each checkpoint on the first-block residual at the verb (training
items), orthonormalized to Z, projected out (h -> mu + (I - ZZ^T)(h - mu)) and the
remaining network gives the binary cross-entropy (natural log) on 10,000 fixed
test sentences.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

THRESHOLD = 0.90


def crossing_step(steps: Sequence[int], iia: Sequence[float], threshold: float = THRESHOLD) -> Optional[int]:
    for s, v in zip(steps, iia):
        if v > threshold:
            return s
    return None


def lag(atomic: Optional[int], compositional: Optional[int]) -> Optional[int]:
    return None if atomic is None or compositional is None else compositional - atomic


@torch.no_grad()
def excluded_loss(model, vocab, train_sents, test_sents, labels_subject: Sequence[int],
                  labels_verb: Sequence[int], device: str = "cpu") -> float:
    from ..models.training import encode_batch
    site = "block1"

    def site_vectors(sents):
        ids, tgt, y = encode_batch(sents, vocab, device)
        h = model.encode(ids, site)
        return ids, tgt, y, h

    _, tgt_tr, _, h_tr = site_vectors(train_sents)
    v_tr = h_tr[torch.arange(len(train_sents)), tgt_tr].cpu().numpy()
    dirs = []
    for lab in (labels_subject, labels_verb):
        clf = LogisticRegression(C=1.0, max_iter=2000, tol=1e-8).fit(v_tr, np.asarray(lab))
        dirs.append(clf.coef_[0])
    Z, _ = np.linalg.qr(np.stack(dirs, 1))
    Z = torch.tensor(Z, dtype=h_tr.dtype, device=h_tr.device)
    mu = torch.tensor(v_tr.mean(0), dtype=h_tr.dtype, device=h_tr.device)
    ids, tgt, y, h = site_vectors(test_sents)
    b = torch.arange(len(test_sents))
    x = h[b, tgt]
    h = h.clone(); h[b, tgt] = mu + (x - mu) - ((x - mu) @ Z) @ Z.T
    logits = model.decode(h, site, ids, tgt)
    return F.cross_entropy(logits, y).item()
