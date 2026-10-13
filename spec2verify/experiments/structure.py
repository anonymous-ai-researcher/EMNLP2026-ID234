"""Lookups follow structure (Section 5.2; Table 3; Appendix F.2, Tables 32-34; Table 44).

English: the hierarchical SCM H against the linear (L) and nearest-noun (N)
competitors on decisive pairs only (T-A, T-B, T-C; T-C has no H-vs-N pairs), for
agreement and binding.  The same alignment scores a competitor at exactly 1 - H
on each opponent-specific subset; independently fitted competitor alignments
are reported separately.  The 2-layer Transformer is trained on canonical order
only or on the mixed data (30% decisive).

German: the case-first SCM KO against L, N and S on F1-F4 (decisive pairs only),
can_nom at the last subword of the noun and subj_num/verb_form at the verb
prefix; strata weighted F1, F2, F3 by 1/8 each and the five F4 distances by 5/8
in total.  F5 measures the preference for the first noun phrase as subject.
"""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np

from ..alignment import das as DAS
from ..alignment.backbone import ScratchBackbone
from ..data import datasets as D
from ..data import pairs as PR
from ..data import templates as T
from ..data.vocab import vocab_for
from ..models.architectures import site_for
from ..models.training import MODEL_SEEDS, accuracy
from .common import FULL, Scale, save
from .scratch import config, get_model

COMPETITORS = {"english": ("L", "N"), "german": ("L", "N", "S")}
GERMAN_WEIGHTS = {("F1", 0): 1 / 8, ("F2", 0): 1 / 8, ("F3", 0): 1 / 8, **{("F4", d): 1 / 8 for d in range(5)}}


def decisive_pairs(sents, dataset: str, variable: str, competitor: str, rng, n_test: int) -> Dict[str, List]:
    scm = PR.scm_for(dataset, "H" if dataset != "german" else "KO")
    pairs = PR.make_pairs(sents, variable, rng, scm, balance=False)
    pairs = PR.decisive_filter(pairs, dataset, competitor)
    sp = PR.split_by_family(pairs, rng)
    sp["test"] = sp["test"][:n_test]
    return sp


def english_scratch(scale: Scale = FULL) -> Dict:
    res: Dict = {}
    for ph in ("agreement", "binding"):
        ds = f"decisive_{ph}"; vocab = vocab_for(ds)
        for regime in ("canonical", "mixed"):
            data = D.english_decisive_dataset(ph, mixed=regime == "mixed")
            for seed in MODEL_SEEDS:
                model, _ = get_model(f"decisive/{ph}/{regime}", data["train"], vocab, config("transformer", seed, scale))
                bb = ScratchBackbone(model, vocab)
                for tpl, ss in data["test"].items():
                    res.setdefault(f"{ph}/{regime}/accuracy/{tpl}", []).append(accuracy(model, ss, vocab, bb.device))
                for comp in COMPETITORS["english"]:
                    rng = random.Random(f"{seed}{ph}{regime}{comp}")
                    sents = [s for tpl, ss in data["test"].items() if not (comp == "N" and tpl == "T-C") for s in ss]
                    sp = decisive_pairs(sents, ds, "subj_H", comp, rng, scale.n(1_536 if comp == "L" else 1_024))
                    h = DAS.refit_fixed(bb, sp, site_for(model, "atomic"), 1, 701, epochs=scale.e(100))
                    res.setdefault(f"{ph}/{regime}/H_vs_{comp}", []).append(h.test_iia)
                    # an independently fitted competitor alignment
                    cscm = PR.scm_for(ds, comp); cvar = cscm.program.atomic
                    rel = {k: [PR.Pair(p.base, p.source, cvar, PR.label_pair(cscm, p.base, p.source, cvar)[0],
                                       p.base_output, p.family) for p in v] for k, v in sp.items()}
                    res.setdefault(f"{ph}/{regime}/{comp}_independent", []).append(
                        DAS.refit_fixed(bb, rel, site_for(model, "atomic"), 1, 701, epochs=scale.e(100)).test_iia)
    save("structure_english_scratch", res)
    return res


def german_scratch(scale: Scale = FULL) -> Dict:
    res: Dict = {}
    vocab = vocab_for("german")
    for regime in ("R1", "R2"):
        data = D.german_dataset(regime)
        for seed in MODEL_SEEDS:
            model, _ = get_model(f"german/{regime}", data["train"], vocab, config("transformer", seed, scale, max_len=64))
            bb = ScratchBackbone(model, vocab)
            for key, ss in data["test"].items():
                if key[0] == "F5":
                    good = [s for s in ss]
                    logits = bb.logits(good)
                    res.setdefault(f"{regime}/F5_preference", []).append(float((logits.argmax(-1) == 1).float().mean()))
                    continue
                res.setdefault(f"{regime}/accuracy/{key}", []).append(accuracy(model, ss, vocab, bb.device))
            for comp in COMPETITORS["german"]:
                vals = []
                for key, ss in data["test"].items():
                    if key[0] == "F5":
                        continue
                    rng = random.Random(f"{seed}{regime}{comp}{key}")
                    sp = decisive_pairs(ss, "german", "subj_num", comp, rng, scale.n(512))
                    if len(sp["test"]) == 0 or len(sp["train"]) == 0:
                        continue
                    r = DAS.refit_fixed(bb, sp, "block1", 1, 701, epochs=scale.e(100))
                    vals.append((GERMAN_WEIGHTS[key], r.test_iia))
                w = sum(a for a, _ in vals)
                res.setdefault(f"{regime}/KO_vs_{comp}", []).append(sum(a * b for a, b in vals) / w if w else None)
    save("structure_german_scratch", res)
    return res


def english_pretrained(model_name: str, scale: Scale = FULL, device: str = "cuda", n_test_per_template: int = 512) -> Dict:
    """Decisive English pairs for a pre-trained model (natural-text versions of T-A, T-B, T-C)."""
    from ..pretrained import hf as HF
    from .common import load as load_res
    model, tok = HF.load(model_name, device=device)
    bb = HF.HFBackbone(model, tok, model_name, device)
    sel = load_res(f"pretrained_core_{model_name}")
    res: Dict = {}
    for ph in ("agreement", "binding"):
        ds = f"decisive_{ph}"
        for comp in COMPETITORS["english"]:
            sents = []
            for tpl in T.DECISIVE_TEMPLATES:
                if comp == "N" and tpl == "T-C":
                    continue
                rng = D.stream(0, "pt-decisive", ph, tpl)
                for f in D.family_pool(lambda r: T.decisive_english(r, tpl, ph), rng, scale.n(3 * n_test_per_template)):
                    for s in f.sentences:
                        s.words = [w for w in s.words if w != "<eos>"]
                        s.words[0] = s.words[0].capitalize()
                        s.words.append(".")
                        sents.append(s)
            sents = [s for s in sents if bb.keep(s)]
            rng = random.Random(f"{model_name}{ph}{comp}")
            sp = decisive_pairs(sents, ds, "subj_H", comp, rng, scale.n(n_test_per_template * 3))
            i = 0
            site = sel[f"{ph}/{PR.VARIABLES[ph][0]}"]["site"][i]; rank = sel[f"{ph}/{PR.VARIABLES[ph][0]}"]["rank"][i]
            res[f"{ph}/H_vs_{comp}"] = [DAS.refit_fixed(bb, sp, site, rank, s, epochs=scale.e(100)).test_iia
                                        for s in DAS.DAS_SEEDS]
    save(f"structure_english_{model_name}", res)
    return res


def german_pretrained(model_name: str, scale: Scale = FULL, device: str = "cuda") -> Dict:
    """German pre-trained analysis: full layer search per variable and stratum (1,536/512/512
    pairs per stratum); can_nom is read at the last subword of the noun, with rank 3 added."""
    from ..pretrained import hf as HF
    model, tok = HF.load(model_name, device=device)
    bb = HF.HFBackbone(model, tok, model_name, device)
    data = D.german_dataset("R2")
    res: Dict = {}
    for key, ss in data["test"].items():
        if key[0] == "F5":
            continue
        for s in ss:
            s.words = [w for w in s.words if w != "<eos>"]
        rng = random.Random(f"{model_name}{key}")
        for var in ("subj_num", "verb_form", "can_nom"):
            sents = list(ss)
            made = PR.make_pairs(sents, var, rng, PR.scm_for("german", "KO"))
            if var == "can_nom":      # both members are read at the noun's last subword
                for p in made:
                    p.base = PR._sub(p.base, {}, "base"); p.base.meta["intervene_at"] = p.source.meta["intervene_at"]
            pairs = PR.split_by_family(made, rng)
            ranks = (1, 2, 3, 4, 8) if var == "can_nom" else DAS.RANKS
            runs = [DAS.search(bb, pairs, s, ranks=ranks, epochs=scale.e(100)) for s in DAS.DAS_SEEDS]
            res[f"{key}/{var}"] = [r.test_iia for r in runs]
    save(f"structure_german_{model_name}", res)
    return res
