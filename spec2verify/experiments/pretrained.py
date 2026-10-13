"""Pre-trained experiments: Table 2 (pre-trained columns), Tables 9/12/14/26/30/43/44,
Figure 2, Figure 6 and the German and larger-model analyses (Appendices C.3, D.2, E.3,
E.4, E.8, F.2, F.8).

Pairs: 6,000/2,000/2,000 per model and variable (40 templates x 150/50/50, split by
family); German and larger-model analyses use 1,536/512/512 per stratum.  Bases and
sources are kept only when the frozen model handles both correctly and the critical
word has at most three subwords.  Each of the five DAS initializations (701, 809, 907,
1013, 1109) searches all L block outputs x ranks {1, 2, 4, 8} on development pairs.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..alignment import das as DAS
from ..alignment import joint as JOINT
from ..analysis import stats as ST
from ..brasp import programs as P
from ..brasp.scm import compile_program
from ..data import datasets as D
from ..data import pairs as PR
from ..data import pretrained_templates as PT
from ..data import templates as T
from ..pretrained import attention as ATT
from ..pretrained import blimp as BL
from ..pretrained import hf as HF
from .common import FULL, Scale, save
from .scratch import UNCHANGED_FRACTION

PHENOMENA = ("agreement", "npi", "binding", "concord")
RANDOM_INIT_SEEDS = (1103, 2207, 3319, 4421, 5531)
PER_TEMPLATE = (150, 50, 50)


def correct(bb: HF.HFBackbone, sents, scm) -> List[bool]:
    """Whether the frozen model's kappa equals the SCM output on each sentence."""
    pred = bb.logits(list(sents)).argmax(-1).tolist()
    return [p == int(scm.output_at(PR._env(s), s.target)) for p, s in zip(pred, sents)]


def pretrained_pairs(bb: HF.HFBackbone, ph: str, variable: str, scale: Scale = FULL, shift: str = "id",
                     per_template=PER_TEMPLATE, seed: int = 0, filter_correct: bool = True) -> Dict[str, List[PR.Pair]]:
    scm = PR.scm_for(ph)
    out = {"train": [], "dev": [], "test": []}
    for tid in range(PT.N_TEMPLATES):
        rng = D.stream(seed, "pt-pairs", ph, variable, shift, tid)

        def make(r):
            frame = r.randrange(10) if shift in ("template", "joint") else None
            load = r.randint(9, 16) if shift in ("load", "joint") else 0
            return PT.GENERATORS[ph](r, tid, lexical=shift in ("lexical", "joint"), load=load, frame=frame)
        quotas = [scale.n(q) for q in per_template]
        fams = D.family_pool(make, rng, 3 * sum(quotas))
        parts = D.split_families(fams, [3 * q for q in quotas], rng)
        for split, q in zip(("train", "dev", "test"), quotas):
            sents = [s for f in parts["val" if split == "dev" else split] for s in f.sentences if bb.keep(s)]
            pairs = PR.make_pairs(sents, variable, rng, scm, twins=PR.twins_of(sents),
                                  unchanged_fraction=UNCHANGED_FRACTION)
            if filter_correct and pairs:
                ok_b = correct(bb, [p.base for p in pairs], scm); ok_s = correct(bb, [p.source for p in pairs], scm)
                pairs = [p for p, a, b in zip(pairs, ok_b, ok_s) if a and b]
            out[split] += PR.balance_labels(pairs, rng)[:q]
    return out


def search_all(bb, pairs, scale: Scale, seeds=DAS.DAS_SEEDS, ranks=DAS.RANKS):
    return [DAS.search(bb, pairs, s, ranks=ranks, epochs=scale.e(100)) for s in seeds]


def core_suite(model_name: str, scale: Scale = FULL, device: str = "cuda") -> Dict:
    """Table 2 (pre-trained), negative controls, Table 12 statistics, Table 14 and Figure 2 per-layer
    curves, the rank stability of Appendix D.1 and the shifts of Table 43."""
    model, tok = HF.load(model_name, device=device)
    bb = HF.HFBackbone(model, tok, model_name, device)
    res: Dict = {"model": model_name}
    larger = model_name in HF.LARGER          # Table 44: 1,536/512/512 pairs per variable
    per_t = (39, 13, 13) if larger else PER_TEMPLATE
    pools = {}
    for ph in PHENOMENA:
        for var in PR.VARIABLES[ph]:
            p = pretrained_pairs(bb, ph, var, scale, per_template=per_t)
            if larger:
                p = {k: v[:scale.n(n)] for (k, v), n in zip(p.items(), (1_536, 512, 512))}
            pools[(ph, var)] = p
    for ph in PHENOMENA:
        for kind, var in zip(("atomic", "compositional"), PR.VARIABLES[ph]):
            pairs = pools[(ph, var)]
            runs = search_all(bb, pairs, scale)
            key = f"{ph}/{var}"
            res[key] = {"test_iia": [r.test_iia for r in runs], "site": [r.site for r in runs],
                        "rank": [r.rank for r in runs], "dev_iia": [r.dev_iia for r in runs],
                        "per_layer_dev": [r.extra["all"] for r in runs]}
            # per-item correctness of the selected alignment (template and item bootstrap)
            items = []
            for r in runs:
                with torch.no_grad():
                    hb = bb.vectors([p.base for p in pairs["test"]], r.site)
                    hs = bb.vectors([p.source for p in pairs["test"]], r.site)
                    pred = bb.patched_logits([p.base for p in pairs["test"]], r.site,
                                             DAS.interchange(hb, hs, r.R.to(hb.device), slice(0, r.rank))).argmax(-1)
                items.append([int(a == p.label) for a, p in zip(pred.tolist(), pairs["test"])])
            res[key]["items"] = {"template": [p.base.template for p in pairs["test"]], "correct": items}
            # controls at the selected site and rank
            sel = runs[0]
            res[key]["shuffled"] = [DAS.refit_fixed(bb, pairs, r.site, r.rank, r.seed, shuffle_labels=True,
                                                    epochs=scale.e(100)).test_iia for r in runs]
            for shift in ("template", "lexical", "load", "joint"):
                sp = pretrained_pairs(bb, ph, var, scale, shift=shift, per_template=(0, 0, 50), filter_correct=False)
                res[key][f"shift/{shift}"] = [DAS.evaluate(bb, sp["test"], r.site, r.R, r.rank) for r in runs]
            res[key]["rank16"] = DAS.refit_fixed(bb, pairs, sel.site, 16, sel.seed, epochs=scale.e(100)).test_iia
    # random-initialization control: rebuild the architecture with seeds 1103..5531
    for s_model, s_das in zip(RANDOM_INIT_SEEDS, DAS.DAS_SEEDS):
        rmodel, _ = HF.load(model_name, random_init=True, seed=s_model, device=device)
        rbb = HF.HFBackbone(rmodel, tok, model_name, device)
        for ph in PHENOMENA:
            for var in PR.VARIABLES[ph]:
                key = f"{ph}/{var}"
                i = DAS.DAS_SEEDS.index(s_das)
                pairs = pools[(ph, var)]
                res[key].setdefault("random_init", []).append(DAS.refit_fixed(
                    rbb, pairs, res[key]["site"][i], res[key]["rank"][i], s_das, epochs=scale.e(100)).test_iia)
        del rmodel
    # Table 12: atomic minus compositional, paired over DAS initializations
    diffs = []
    for k in range(5):
        a = np.mean([res[f"{ph}/{PR.VARIABLES[ph][0]}"]["test_iia"][k] for ph in PHENOMENA])
        c = np.mean([res[f"{ph}/{PR.VARIABLES[ph][1]}"]["test_iia"][k] for ph in PHENOMENA])
        diffs.append(a - c)
    res["atomic_minus_compositional"] = {"diffs": diffs, "t_test": ST.one_sample_t(diffs, 0.0)}
    save(f"pretrained_core_{model_name}", res)
    return res


def wrong_scm(model_name: str, scale: Scale = FULL, device: str = "cuda") -> Dict:
    """Negative control (i) for the pre-trained models: each SCM on another phenomenon's pairs,
    at the site and rank selected for the prescribed specification."""
    from .common import load as load_res
    sel = load_res(f"pretrained_core_{model_name}")
    model, tok = HF.load(model_name, device=device)
    bb = HF.HFBackbone(model, tok, model_name, device)
    res: Dict = {}
    for ph in PHENOMENA:
        for kind, var in zip((0, 1), PR.VARIABLES[ph]):
            pairs = pretrained_pairs(bb, ph, var, scale)
            for other in PHENOMENA:
                if other == ph:
                    continue
                wrong = compile_program(P.core_program(other)); wvar = PR.VARIABLES[other][kind]
                rel = {k: [PR.Pair(p.base, p.source, wvar, PR.label_pair(wrong, p.base, p.source, wvar)[0],
                                   p.base_output, p.family) for p in v] for k, v in pairs.items()}
                key = f"{ph}/{var}"
                res.setdefault(f"{ph}<-{other}/{var}", []).extend(
                    DAS.refit_fixed(bb, rel, sel[key]["site"][i], sel[key]["rank"][i], s, epochs=scale.e(100)).test_iia
                    for i, s in enumerate(DAS.DAS_SEEDS))
    save(f"pretrained_wrong_scm_{model_name}", res)
    return res


def blimp_suite(model_name: str, scale: Scale = FULL, device: str = "cuda", blimp_dir: Optional[str] = None) -> Dict:
    """Behavioral accuracy on the 23 paradigms (Table 6, Table 30) and, for Pythia-1.4B, the
    per-paradigm compositional IIA at the fixed sites of Appendix E.3 (Table 26)."""
    model, tok = HF.load(model_name, device=device)
    bb = HF.HFBackbone(model, tok, model_name, device)
    res: Dict = {}
    for ph, names in BL.PARADIGMS.items():
        for name in names:
            rows = BL.load_paradigm(name, f"{blimp_dir}/{name}.jsonl" if blimp_dir else None)
            acc = HF.minimal_pair_accuracy(bb, [r["sentence_good"] for r in rows],
                                           [r["sentence_bad"] for r in rows],
                                           [r.get("pair_id", i) for i, r in enumerate(rows)])
            res[name] = {"phenomenon": ph, "accuracy": acc}
            if model_name in BL.FIXED_LAYERS:
                a_layer, c_layer = BL.FIXED_LAYERS[model_name][ph]
                rank = 2
                pairs = BL.paradigm_pairs(name, rows, ph, sizes=tuple(scale.n(x) for x in (3_000, 1_000, 1_000)))
                res[name]["compositional_iia"] = [DAS.refit_fixed(bb, pairs, f"layer{c_layer}", rank, s,
                                                                  epochs=scale.e(100)).test_iia for s in DAS.DAS_SEEDS]
    save(f"blimp_{model_name}", res)
    return res


def joint_pretrained(model_name: str, scale: Scale = FULL, device: str = "cuda") -> Dict:
    from .common import load as load_res
    sel = load_res(f"pretrained_core_{model_name}")
    model, tok = HF.load(model_name, device=device)
    bb = HF.HFBackbone(model, tok, model_name, device)
    res: Dict = {}
    for ph in PHENOMENA:
        a_var, c_var = PR.VARIABLES[ph]; scm = PR.scm_for(ph)
        A = pretrained_pairs(bb, ph, a_var, scale); C = pretrained_pairs(bb, ph, c_var, scale)
        for i, s in enumerate(DAS.DAS_SEEDS):
            rng = random.Random(s)
            Tr = {k: JOINT.make_triples(A[k], C[k], scm, rng) for k in A}
            J = JOINT.fit_joint(bb, A, C, Tr, sel[f"{ph}/{a_var}"]["site"][i], sel[f"{ph}/{c_var}"]["site"][i], s,
                                epochs=scale.e(100))
            for k, v in JOINT.evaluate_joint(bb, J, A["test"], C["test"], Tr["test"]).items():
                res.setdefault(f"{ph}/{k}", []).append(v)
    save(f"joint_{model_name}", res)
    return res


def attention_suite(model_name: str, scale: Scale = FULL, device: str = "cuda") -> Dict:
    """Head selection by atomic-IIA drop on development pairs and attention mass on the
    target word (Appendix F.8)."""
    from .common import load as load_res
    sel = load_res(f"pretrained_core_{model_name}")
    model, tok = HF.load(model_name, device=device)
    bb = HF.HFBackbone(model, tok, model_name, device)
    n_layers = len(bb.layers); n_heads = model.config.num_attention_heads
    res: Dict = {}
    for ph in PHENOMENA:
        var = PR.VARIABLES[ph][0]
        pairs = pretrained_pairs(bb, ph, var, scale)
        site, rank = sel[f"{ph}/{var}"]["site"][0], sel[f"{ph}/{var}"]["rank"][0]
        R = DAS.refit_fixed(bb, pairs, site, rank, 701, epochs=scale.e(100)).R

        def dev_iia(c):
            if c is None:
                return DAS.evaluate(bb, pairs["dev"], site, R, rank)
            with ATT.ablate_hf_head(model, c[0], c[1], lambda: bb.current_positions):
                return DAS.evaluate(bb, pairs["dev"], site, R, rank)
        cands = [(l, h) for l in range(1, n_layers + 1) for h in range(n_heads)]
        layer, head = ATT.select_head(cands, dev_iia)
        target_key = {"agreement": "subj_pos", "binding": "subj_pos", "concord": "det_pos", "npi": "lic_pos"}[ph]
        masses, largest = [], []
        for rep in ATT.REPEAT_SEEDS:
            rng = random.Random(rep)
            test = [p.base for p in pairs["test"]]
            items = rng.sample(test, min(len(test), scale.n(1_000)))
            rows = bb.attention_rows(items, layer, head)
            m_rep, l_rep = [], []
            for s, row in zip(items, rows):
                q = bb.position(s)
                spans = [bb.word_span(s, k) for k in range(s.target)]
                m, is_max = ATT.attention_mass(row, q, spans[s.meta.get(target_key, 0)], spans)
                m_rep.append(m); l_rep.append(is_max)
            masses.append(float(np.mean(m_rep))); largest.append(float(np.mean(l_rep)))
        res[ph] = {"head": (layer, head), "attention_mass": masses, "target_largest": largest}
    save(f"attention_{model_name}", res)
    return res
