"""Alignment as a predictor of failure (Section 5.1; Appendix F.4, Table 36).

Items: 1,200 per phenomenon from the 40 pre-trained templates (30 each), with
intervening phrases added in all four phenomena: 0-8 prepositional phrases for
agreement and binding, 0-4 number-neutral separators for NPI and concord, spread
evenly over the levels.  Errors are those of Pythia-1.4B under sentence
log-probability; the alignment score uses the atomic alignment selected for the
phenomenon (layer and rank from the core suite) and 32 fixed bases per source.
"""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import torch

from ..alignment import das as DAS
from ..alignment import probes as PROBES
from ..analysis import failure as FA
from ..data import datasets as D
from ..data import lexicon as lx
from ..data import pairs as PR
from ..data import pretrained_templates as PT
from ..pretrained import hf as HF
from .common import FULL, Scale, save
from .pretrained import PHENOMENA, pretrained_pairs

LOADS = {"agreement": range(9), "binding": range(9), "npi": range(5), "concord": range(5)}
PROBE_LAYERS = {"agreement": 9, "binding": 9, "npi": 8, "concord": 7}


def items(ph: str, per_template: int = 30, seed: int = 0) -> List:
    out = []
    for tid in range(PT.N_TEMPLATES):
        rng = D.stream(seed, "failure-items", ph, tid)
        loads = list(LOADS[ph])
        for k in range(per_template):
            load = loads[k % len(loads)]
            f = PT.GENERATORS[ph](rng, tid, load=load)
            s = f.good if rng.random() < 0.5 else f.bad
            s.meta = dict(s.meta, item=f"{ph}-{tid}-{k}", grammatical=s.label, pair=f)
            out.append(s)
    return out


def run(model_name: str = "pythia-1.4b", scale: Scale = FULL, device: str = "cuda") -> Dict:
    from .common import load as load_res
    sel = load_res(f"pretrained_core_{model_name}")
    model, tok = HF.load(model_name, device=device)
    bb = HF.HFBackbone(model, tok, model_name, device)
    res: Dict = {}
    for ph in PHENOMENA:
        its = items(ph, scale.n(30))
        # behavioral error: the grammatical member does not win under sentence log-probability
        good = [HF.detokenize(s.meta["pair"].good.words) for s in its]
        bad = [HF.detokenize(s.meta["pair"].bad.words) for s in its]
        lg, lb = bb.sentence_logprob(good), bb.sentence_logprob(bad)
        err = np.array([int(not (a > b)) for a, b in zip(lg, lb)])
        # alignment score: 32 fixed, correctly handled development bases per source (16 per target label)
        var = PR.VARIABLES[ph][0]; scm = PR.scm_for(ph)
        site, rank = sel[f"{ph}/{var}"]["site"][0], sel[f"{ph}/{var}"]["rank"][0]
        pool = pretrained_pairs(bb, ph, var, scale)
        R = DAS.refit_fixed(bb, pool, site, rank, 701, epochs=scale.e(100)).R
        dev_bases = [p.base for p in pool["dev"]]
        scores = []
        for s in its:
            rng = random.Random(f"{FA.BASE_SEED}|{ph}|{s.meta['item']}")
            cand = rng.sample(dev_bases, min(len(dev_bases), 64))
            pairs = []
            for b in cand:
                lab, base_out, changed = PR.label_pair(scm, b, s, var)
                pairs.append(PR.Pair(b, s, var, lab, base_out, b.family))
            pos = [p for p in pairs if p.label == 1][:16]; neg = [p for p in pairs if p.label == 0][:16]
            scores.append(DAS.evaluate(bb, pos + neg, site, R, rank))
        scores = np.array(scores)
        # probe confidence for the atomic variable (separate CFG-style probe sentences)
        layer = f"layer{PROBE_LAYERS[ph]}"
        tr = [s for f in PT.pretrained_pool(ph, scale.n(50), seed=1) for s in f.sentences if bb.keep(s)]
        lab = lambda ss: [scm.value(PR._env(x), var, x.target) for x in ss]
        clf, st = PROBES.fit_probe(bb.vectors(tr, layer).cpu().numpy(), lab(tr))
        conf = PROBES.confidence_true(clf, st, bb.vectors(its, layer).cpu().numpy(), lab(its))
        # B1 features
        content = {w for p in lx.NOUNS + lx.VERBS for w in p}
        corpus = [x.words for x in tr]
        pf, rmin, rmax = FA.lexical_tables([[w.lower() for w in c] for c in corpus], content)
        X1 = np.array([FA.b1_features([w.lower() for w in s.words], len(tok(HF.detokenize(s.words))["input_ids"]),
                                      s.load, pf, rmin, rmax, content) for s in its], float)
        groups = [s.template for s in its]
        p_b1 = FA.oof_logistic(X1, err, groups)
        p_b1a = FA.oof_logistic(np.c_[X1, 1 - scores], err, groups)
        p_b1p = FA.oof_logistic(np.c_[X1, 1 - conf], err, groups)
        p_b1pa = FA.oof_logistic(np.c_[X1, 1 - conf, 1 - scores], err, groups)
        p_a = FA.oof_logistic((1 - scores)[:, None], err, groups)
        p_p = FA.oof_unpenalized_1d(1 - conf, err, groups)
        res[ph] = {"error_rate": float(err.mean()),
                   "B1": FA.auroc(err, p_b1), "B1+align": FA.template_bootstrap(err, p_b1, p_b1a, groups),
                   "B1+probe": FA.auroc(err, p_b1p),
                   "B1+probe+align": FA.template_bootstrap(err, p_b1p, p_b1pa, groups),
                   "align_minus_probe": FA.template_bootstrap(err, p_p, p_a, groups, seed=9302026),
                   "AP": {"B1": FA.average_precision(err, p_b1), "B1+align": FA.average_precision(err, p_b1a)},
                   "tertiles_vs_B1": FA.tertile_gains(err, p_b1, p_b1a, groups)}
    save(f"failure_{model_name}", res)
    return res
