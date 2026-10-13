"""Joint alignment (Section 5.1; Appendix F.1, Table 31).

The atomic subspace (rank 1) and the compositional subspace (rank 2) are fitted
together.  Each variable sits at its own site (the one selected for it in the
per-variable search) and at its consumption position; when the two sites
coincide the subspaces are disjoint coordinate blocks of one rotation,
otherwise each site has its own rotation.  An atomic patch propagates through
the network to the compositional site.  Training uses equal thirds of single
atomic, single compositional and double interventions (batches cycle 86/85/85)
with the DAS optimizer; all six evaluation conditions use the same alignment:

* single:        one variable set from a source;
* double:        atomic from source 1, compositional from source 2 (the SCM predicts
                 source 2's compositional value);
* screening-off: atomic from a source, compositional subspace pinned to its base
                 value (the SCM predicts no change);
* mediation:     atomic from a source; the compositional value is read from the
                 compositional coordinates by an L2 logistic decoder (C = 1) fitted on
                 unintervened training sentences, and compared with the value the SCM
                 propagates;
* bypass:        the source is patched into a fixed random rank-1 direction orthogonal
                 to both learned subspaces (seed 51031) on the single-atomic pairs and
                 scored against the SCM's atomic-intervention label, so an ineffective
                 patch scores 0.5 (half of these pairs keep the base output).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from ..data.pairs import Pair, label_pair, _env
from ..brasp.scm import SCM
from .backbone import Backbone
from .das import Rotation, interchange

BYPASS_SEED = 51031


@dataclass
class Triple:
    """(base, atomic source, compositional source) with the SCM's labels."""
    base: object
    s_atomic: object
    s_comp: object
    label_double: int
    label_screen: int
    comp_after_atomic: int     # compositional value after the atomic intervention (mediation)


class JointAlignment(torch.nn.Module):
    def __init__(self, d: int, site_a: str, site_c: str, seed: int):
        super().__init__()
        self.site_a, self.site_c = site_a, site_c
        self.same = site_a == site_c
        self.rot_a = Rotation(d, seed)
        self.rot_c = self.rot_a if self.same else Rotation(d, seed + 1)
        self.ca = slice(0, 1)
        self.cc = slice(1, 3) if self.same else slice(0, 2)

    def matrices(self):
        Ra = self.rot_a.matrix()
        return Ra, (Ra if self.same else self.rot_c.matrix())


def _vec(bb: Backbone, sents, site):
    return bb.vectors(list(sents), site)


def logits_single(bb, J, sents_b, hb_a, hs_a, hb_c, hs_c, which: str):
    Ra, Rc = J.matrices()
    if which == "atomic":
        return bb.patched_logits(sents_b, J.site_a, interchange(hb_a, hs_a, Ra, J.ca))
    return bb.patched_logits(sents_b, J.site_c, interchange(hb_c, hs_c, Rc, J.cc))


def logits_double(bb, J, sents_b, hb_a, hs1_a, hb_c, hs2_c, pin_base: bool = False):
    """Atomic from source 1 at site_a, then (at site_c, on the propagated activation) the
    compositional coordinates are set from source 2, or pinned to the clean base value."""
    Ra, Rc = J.matrices()
    target_c = hb_c if pin_base else hs2_c
    if J.same:
        new = hb_a + ((hs1_a - hb_a) @ Ra[J.ca].T) @ Ra[J.ca]
        new = new + ((target_c - new) @ Rc[J.cc].T) @ Rc[J.cc]
        return bb.patched_logits(sents_b, J.site_a, new)
    new_a = interchange(hb_a, hs1_a, Ra, J.ca)
    fn = lambda h: h + ((target_c - h) @ Rc[J.cc].T) @ Rc[J.cc]
    return bb.patched_logits(sents_b, J.site_a, new_a, second=(J.site_c, fn))


def fit_joint(bb: Backbone, atomic: Dict[str, List[Pair]], comp: Dict[str, List[Pair]],
              triples: Dict[str, List[Triple]], site_a: str, site_c: str, seed: int,
              epochs: int = 100, batch_size: int = 256, lr: float = 5e-3) -> JointAlignment:
    J = JointAlignment(bb.d, site_a, site_c, seed)
    dev = bb.vectors([atomic["train"][0].base], site_a).device
    J.to(dev)
    opt = torch.optim.Adam(J.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    tr_a, tr_c, tr_d = atomic["train"], comp["train"], triples["train"]
    A = (_vec(bb, [p.base for p in tr_a], site_a), _vec(bb, [p.source for p in tr_a], site_a))
    C = (_vec(bb, [p.base for p in tr_c], site_c), _vec(bb, [p.source for p in tr_c], site_c))
    Dd = (_vec(bb, [t.base for t in tr_d], site_a), _vec(bb, [t.s_atomic for t in tr_d], site_a),
          _vec(bb, [t.base for t in tr_d], site_c), _vec(bb, [t.s_comp for t in tr_d], site_c))
    ya = torch.tensor([p.label for p in tr_a], device=dev); yc = torch.tensor([p.label for p in tr_c], device=dev)
    yd = torch.tensor([t.label_double for t in tr_d], device=dev)
    g = torch.Generator().manual_seed(seed)
    third = batch_size // 3
    sizes = [(third + 1, third, third), (third, third + 1, third), (third, third, third + 1)]
    steps = max(len(tr_a), len(tr_c), len(tr_d)) * 3 // batch_size + 1
    best = (-1.0, None)
    for epoch in range(epochs):
        pa, pc, pd = (torch.randperm(len(x), generator=g) for x in (tr_a, tr_c, tr_d))
        for step in range(steps):
            na, nc, nd = sizes[step % 3]
            ia = pa[(step * na) % len(tr_a):][:na]; ic = pc[(step * nc) % len(tr_c):][:nc]
            idd = pd[(step * nd) % len(tr_d):][:nd]
            loss = 0.0
            if len(ia):
                la = logits_single(bb, J, [tr_a[i].base for i in ia.tolist()], A[0][ia], A[1][ia], None, None, "atomic")
                loss = loss + F.cross_entropy(la, ya[ia.to(dev)], reduction="sum")
            if len(ic):
                lc = logits_single(bb, J, [tr_c[i].base for i in ic.tolist()], None, None, C[0][ic], C[1][ic], "comp")
                loss = loss + F.cross_entropy(lc, yc[ic.to(dev)], reduction="sum")
            if len(idd):
                ld = logits_double(bb, J, [tr_d[i].base for i in idd.tolist()], Dd[0][idd], Dd[1][idd], Dd[2][idd], Dd[3][idd])
                loss = loss + F.cross_entropy(ld, yd[idd.to(dev)], reduction="sum")
            loss = loss / max(1, len(ia) + len(ic) + len(idd))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(J.parameters(), 1.0); opt.step()
        score = np.mean([evaluate_joint(bb, J, atomic["dev"], comp["dev"], triples["dev"])[k]
                         for k in ("single_atomic", "single_compositional", "double")])
        if score > best[0]:
            best = (score, {k: v.detach().clone() for k, v in J.state_dict().items()})
    J.load_state_dict(best[1])
    return J


def _bypass_direction(J: JointAlignment, d: int) -> torch.Tensor:
    """A random unit vector orthogonal to the learned atomic and compositional subspaces."""
    Ra, Rc = (m.detach() for m in J.matrices())
    learned = torch.cat([Ra[J.ca], Rc[J.cc]], 0) if J.same else Ra[J.ca]
    g = torch.Generator().manual_seed(BYPASS_SEED)
    v = torch.randn(d, generator=g).to(learned.device)
    q, _ = torch.linalg.qr(learned.T)
    v = v - q @ (q.T @ v)
    return v / v.norm()


@torch.no_grad()
def evaluate_joint(bb: Backbone, J: JointAlignment, atomic: List[Pair], comp: List[Pair],
                   triples: List[Triple], mediation_decoder: Optional[LogisticRegression] = None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    hb_a = _vec(bb, [p.base for p in atomic], J.site_a); hs_a = _vec(bb, [p.source for p in atomic], J.site_a)
    y = torch.tensor([p.label for p in atomic], device=hb_a.device)
    out["single_atomic"] = (logits_single(bb, J, [p.base for p in atomic], hb_a, hs_a, None, None, "atomic")
                            .argmax(-1) == y).float().mean().item()
    hb_c = _vec(bb, [p.base for p in comp], J.site_c); hs_c = _vec(bb, [p.source for p in comp], J.site_c)
    y = torch.tensor([p.label for p in comp], device=hb_c.device)
    out["single_compositional"] = (logits_single(bb, J, [p.base for p in comp], None, None, hb_c, hs_c, "comp")
                                   .argmax(-1) == y).float().mean().item()
    tb = [t.base for t in triples]
    T = (_vec(bb, tb, J.site_a), _vec(bb, [t.s_atomic for t in triples], J.site_a),
         _vec(bb, tb, J.site_c), _vec(bb, [t.s_comp for t in triples], J.site_c))
    yd = torch.tensor([t.label_double for t in triples], device=T[0].device)
    ys = torch.tensor([t.label_screen for t in triples], device=T[0].device)
    out["double"] = (logits_double(bb, J, tb, *T).argmax(-1) == yd).float().mean().item()
    out["screening_off"] = (logits_double(bb, J, tb, *T, pin_base=True).argmax(-1) == ys).float().mean().item()
    # bypass on the single-atomic pairs
    v = _bypass_direction(J, bb.d)
    new = hb_a + ((hs_a - hb_a) @ v)[:, None] * v[None]
    y = torch.tensor([p.label for p in atomic], device=hb_a.device)
    out["bypass"] = (bb.patched_logits([p.base for p in atomic], J.site_a, new).argmax(-1) == y).float().mean().item()
    if mediation_decoder is not None:
        out["mediation"] = mediation_score(bb, J, triples, mediation_decoder)
    return out


def comp_coordinates(bb: Backbone, J: JointAlignment, sents) -> np.ndarray:
    _, Rc = J.matrices()
    return (_vec(bb, sents, J.site_c) @ Rc[J.cc].T).detach().cpu().numpy()


def fit_mediation_decoder(bb: Backbone, J: JointAlignment, sents, comp_labels: Sequence[int]) -> LogisticRegression:
    clf = LogisticRegression(C=1.0, max_iter=2000, tol=1e-8)
    clf.fit(comp_coordinates(bb, J, sents), np.asarray(comp_labels))
    return clf


@torch.no_grad()
def mediation_score(bb: Backbone, J: JointAlignment, triples: List[Triple], clf: LogisticRegression) -> float:
    """Atomic patch, then read the compositional coordinates at the compositional site."""
    Ra, Rc = J.matrices()
    tb = [t.base for t in triples]
    hb_a = _vec(bb, tb, J.site_a); hs_a = _vec(bb, [t.s_atomic for t in triples], J.site_a)
    captured = []

    def grab(h):
        captured.append(h.clone()); return h
    if J.same:
        h_c = interchange(hb_a, hs_a, Ra, J.ca)
    else:
        bb.patched_logits(tb, J.site_a, interchange(hb_a, hs_a, Ra, J.ca), second=(J.site_c, grab))
        h_c = captured[0]
    coords = (h_c @ Rc[J.cc].T).cpu().numpy()
    pred = clf.predict(coords)
    return float(np.mean(pred == np.array([t.comp_after_atomic for t in triples])))


def make_triples(atomic: List[Pair], comp: List[Pair], scm: SCM, rng) -> List[Triple]:
    """Pair each atomic pair with a compositional source of the same base sentence."""
    import numpy as _np
    by_base = {id(p.base): p for p in comp}
    comps = list(comp)
    out = []
    a_var, c_var = scm.program.atomic, scm.program.compositional
    for p in atomic:
        q = by_base.get(id(p.base)) or comps[rng.randrange(len(comps))]
        b, s1, s2 = p.base, p.source, q.source
        be, s1e, s2e = _env(b), _env(s1), _env(s2)
        pos = b.target
        comp_s2 = scm.value(s2e, c_var, s2.target)
        a_val = scm.run(s1e)[a_var]
        ov_a = _np.ma.masked_all(len(b.words), dtype=bool); ov_a[pos] = a_val[pos]
        ov_c = _np.ma.masked_all(len(b.words), dtype=bool); ov_c[pos] = comp_s2
        lab_double = int(scm.run(be, {a_var: ov_a, c_var: ov_c})[scm.output][pos])
        lab_screen = int(scm.output_at(be, pos))
        after = int(scm.run(be, {a_var: ov_a})[c_var][pos])
        out.append(Triple(b, s1, s2, lab_double, lab_screen, after))
    return out
