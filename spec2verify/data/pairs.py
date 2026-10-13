"""Counterfactual pairs for interchange interventions (Appendix B.3).

A pair (b, s) edits one token of the base b: the critical token of an atomic
variable (the subject or antecedent noun, the licensor slot, the determiner) or
the target token of a compositional one (the verb, the reflexive, the noun; for
NPI the source is the minimal-pair twin, whose licensor sits in the other
clause).  The counterfactual label is computed by the compiled SCM,
C_{U <- s}(b), at the consumption position.  Edited pairs whose source equals
the base or leaves U unchanged are discarded; optionally (``unchanged_fraction``),
a share of the pairs instead uses a neutral edit that keeps U and the output, so
that an intervention without effect scores 0.5.  Labels are balanced, and the
60/20/20 split is made by sentence family before pairs are formed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..brasp import programs as P
from ..brasp.scm import SCM, compile_program
from . import lexicon as lx
from .templates import Sentence
from .vocab import env_for


@dataclass
class Pair:
    base: Sentence
    source: Sentence
    variable: str
    label: int                 # C_{U <- s}(b)
    base_output: int           # C(b)
    family: str
    load: int = 0
    stratum: str = ""


# ----------------------------------------------------------------------------- number flips
def _pairs_map(pairs: Sequence[Tuple[str, str]]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for sg, pl in pairs:
        m.setdefault(sg, pl); m.setdefault(pl, sg)
    return m


FLIP = {}
for _p in (lx.NOUNS, lx.VERBS, lx.DETERMINERS, lx.SHIFT_NOUNS, lx.SHIFT_VERBS):
    FLIP.update(_pairs_map(_p))
FLIP.update({"himself": "themselves", "herself": "themselves", "themselves": "himself",
             "has": "have", "have": "has"})


def flip(word: str) -> str:
    return FLIP.get(word, word)


def _sub(s: Sentence, edits: Dict[int, str], tag: str) -> Sentence:
    w = list(s.words)
    for i, x in edits.items():
        w[i] = x
    return Sentence(w, -1, s.target, s.family, s.dataset, s.load, s.template, dict(s.meta, edit=tag))


# ----------------------------------------------------------------------------- per-dataset specification
VARIABLES: Dict[str, Tuple[str, str]] = {
    "agreement": ("subj_num", "verb_form"), "distance": ("subj_num", "verb_form"),
    "binding": ("ante_num", "refl_ok"), "npi": ("has_lic", "npi_ok"),
    "concord": ("det_num", "conc_ok"), "concord_stress": ("det_num", "conc_ok"),
    "decisive_agreement": ("subj_H", "verb_form"), "decisive_binding": ("subj_H", "refl_ok"),
    "german": ("subj_num", "verb_form"), "npi3": ("x1", "npi_ok3"), "boolean": ("a", "out"),
}


def program_for(dataset: str, rule: str = "", boolean_fn: str = "XOR"):
    if dataset in ("agreement", "distance"):
        return P.agreement()
    if dataset == "binding":
        return P.binding()
    if dataset == "npi":
        return P.npi()
    if dataset in ("concord", "concord_stress"):
        return P.concord()
    if dataset.startswith("decisive_"):
        ph = dataset.split("_", 1)[1]
        return {"": P.english_hierarchical, "H": P.english_hierarchical, "L": P.english_linear,
                "N": P.english_nearest}[rule](ph)
    if dataset == "german":
        return P.german(rule or "KO")
    if dataset == "npi3":
        return P.npi_three_input()
    if dataset == "boolean":
        return P.boolean_task(boolean_fn)
    raise ValueError(dataset)


def edit(s: Sentence, variable: str, rng: random.Random, twin: Optional[Sentence] = None) -> Optional[Sentence]:
    """The source sentence for an interchange on ``variable``."""
    ds, w, t = s.dataset, s.words, s.target
    if variable in ("subj_num", "ante_num", "subj_H") and ds != "german":
        i = s.meta.get("subj_pos", 1)
        return _sub(s, {i: flip(w[i])}, "atomic")
    if variable == "det_num":
        return _sub(s, {0: flip(w[0])}, "atomic")
    if variable == "has_lic":
        lic = "the" if w[0] in lx.LICENSORS else ("no" if w[1] in FLIP and w[1] in {p for _, p in lx.NOUNS}
                                                  else rng.choice(["no", "neither"]))
        return _sub(s, {0: lic}, "atomic")
    if variable in ("verb_form", "refl_ok", "conc_ok") and ds != "german":
        return _sub(s, {t: flip(w[t])}, "compositional")
    if variable == "npi_ok":
        return twin if twin is not None else _sub(s, {0: "the" if w[0] in lx.LICENSORS else "no"}, "compositional")
    if ds == "german":
        if variable == "verb_form":
            sg, pl = s.meta["verb"]
            return _sub(s, {t: pl if w[t] == sg else sg}, "compositional")
        if variable == "subj_num":
            for art, noun_pos, noun, case in s.meta["nps"]:
                if noun_pos == s.meta["subj_pos"]:
                    pl = not s.meta["gold_number"][noun_pos]
                    src = _sub(s, {art: lx.german_article(case, noun[3], pl),
                                   noun_pos: lx.german_noun_form(noun, case, pl)}, "atomic")
                    gold = dict(s.meta["gold_number"]); gold[noun_pos] = pl
                    src.meta["gold_number"] = gold
                    return src
        if variable == "can_nom":
            # swap a masculine singular article between nominative and accusative (der <-> den)
            for art, noun_pos, noun, case in s.meta["nps"]:
                if w[art] in ("der", "den") and noun[3] == "masc" and not s.meta["gold_number"][noun_pos] \
                        and case in ("nom", "acc"):
                    src = _sub(s, {art: "den" if w[art] == "der" else "der"}, "can_nom")
                    src.meta["intervene_at"] = noun_pos
                    return src
            return None
    if ds == "npi3":
        c2 = w.index("and") + 1
        if variable == "npi_ok3":
            variable = rng.choice(["x1", "x2", "c"])
        if variable in ("x1", "x2"):
            i = 0 if variable == "x1" else c2
            plural = w[i + 1] in {p for _, p in lx.NOUNS}
            return _sub(s, {i: "the" if w[i] in lx.LICENSORS else ("no" if plural else rng.choice(["no", "neither"]))},
                        variable)
        if variable == "c":
            e = {i: ("once" if x == "ever" else "ever") for i, x in enumerate(w) if x in ("ever", "once")}
            src = _sub(s, e, "c")
            src.target = next(i for i, x in enumerate(src.words) if x == "ever")
            return src
    if ds == "boolean":
        if variable == "out":
            variable = rng.choice(["a", "b"])
        i = 1 if variable == "a" else 4
        tok = w[i]
        return _sub(s, {i: tok[0] + ("0" if tok[1] == "1" else "1")}, variable)
    raise ValueError((ds, variable))


def neutral_edit(s: Sentence, rng: random.Random) -> Optional[Sentence]:
    """A source that leaves the prescribed variables unchanged: one non-critical word is
    replaced (an adverb, the binding verb, a Boolean distractor, or the number of a noun
    other than the subject, antecedent or head noun)."""
    w = s.words
    critical = {0, s.target, s.meta.get("subj_pos", 1), s.meta.get("det_pos", 0)}
    nouns = {n for p in lx.NOUNS + lx.SHIFT_NOUNS for n in p}
    for i in rng.sample(range(len(w)), len(w)):
        x = w[i].lower()
        if i in critical:
            continue
        if x in lx.ADVERBS:
            return _sub(s, {i: rng.choice([a for a in lx.ADVERBS if a != x])}, "neutral")
        if x in lx.BINDING_VERBS:
            return _sub(s, {i: [v for v in lx.BINDING_VERBS if v != x][0]}, "neutral")
        if w[i].startswith("F") and w[i][1:].isdigit():
            return _sub(s, {i: rng.choice([f for f in lx.BOOL_DISTRACTORS if f != w[i]])}, "neutral")
        if x in nouns:
            return _sub(s, {i: flip(x)}, "neutral")
    return None


def scm_for(dataset: str, rule: str = "", boolean_fn: str = "XOR") -> SCM:
    return compile_program(program_for(dataset, rule, boolean_fn))


def _env(s: Sentence):
    return env_for(s.words, s.dataset, s.family, gold_number=s.meta.get("gold_number"))


def label_pair(scm: SCM, base: Sentence, source: Sentence, variable: str) -> Tuple[int, int, bool]:
    """(counterfactual label, base output, whether U differs between base and source)."""
    be, se = _env(base), _env(source)
    if variable == "can_nom":       # noun-level variable: intervene at the noun, read the verb
        j = source.meta["intervene_at"]
        lab = scm.interchange(be, {variable: se}, j, output_position=base.target)
        return int(lab), int(scm.output_at(be, base.target)), scm.value(be, variable, j) != scm.value(se, variable, j)
    pos = base.target
    src_pos = source.target
    bu = scm.value(be, variable, pos); su = scm.value(se, variable, src_pos)
    if src_pos != pos:   # e.g. the clause of the NPI changes in the three-input construction
        run = scm.run(se)[variable]
        import numpy as np
        val = np.array(scm.run(be)[variable]); val[pos] = run[src_pos]
        ov = np.ma.masked_all(len(val), dtype=bool); ov[pos] = run[src_pos]
        lab = bool(scm.run(be, {variable: ov})[scm.output][pos])
    else:
        lab = scm.interchange(be, {variable: se}, pos)
    return int(lab), int(scm.output_at(be, pos)), bu != su


def make_pairs(sentences: Sequence[Sentence], variable: str, rng: random.Random, scm: Optional[SCM] = None,
               twins: Optional[Dict[int, Sentence]] = None, balance: bool = True,
               require_change: bool = True, unchanged_fraction: float = 0.0) -> List[Pair]:
    """One candidate pair per base sentence; filtered and label-balanced.

    ``unchanged_fraction`` > 0 replaces that share of the sources by neutral edits, whose
    counterfactual label equals the base output (used for the joint-alignment pairs, where
    half of the targets equal the base output, Appendix F.1)."""
    scm = scm or scm_for(sentences[0].dataset)
    out: List[Pair] = []
    for k, b in enumerate(sentences):
        if unchanged_fraction and rng.random() < unchanged_fraction:
            s = neutral_edit(b, rng)
            if s is None:
                continue
            s.label = int(scm.output_at(_env(s), s.target))
            lab, base_out, _ = label_pair(scm, b, s, variable)
            out.append(Pair(b, s, variable, lab, base_out, b.family, b.load, b.template))
            continue
        s = edit(b, variable, rng, (twins or {}).get(k))
        if s is None or s.words == b.words:
            continue
        if s.label < 0:
            s.label = int(scm.output_at(_env(s), s.target))
        lab, base_out, changed = label_pair(scm, b, s, variable)
        if require_change and not changed:
            continue
        out.append(Pair(b, s, variable, lab, base_out, b.family, b.load, b.template))
    return balance_labels(out, rng) if balance else out


def balance_labels(pairs: List[Pair], rng: random.Random) -> List[Pair]:
    pos = [p for p in pairs if p.label == 1]; neg = [p for p in pairs if p.label == 0]
    n = min(len(pos), len(neg))
    rng.shuffle(pos); rng.shuffle(neg)
    out = pos[:n] + neg[:n]; rng.shuffle(out)
    return out


def split_by_family(pairs: Sequence[Pair], rng: random.Random, fractions=(0.6, 0.2, 0.2)
                    ) -> Dict[str, List[Pair]]:
    """60/20/20 by sentence family, so a base and its edit never cross portions."""
    fams = sorted({p.family for p in pairs}); rng.shuffle(fams)
    cut1 = int(len(fams) * fractions[0]); cut2 = cut1 + int(len(fams) * fractions[1])
    where = {f: ("train" if i < cut1 else "dev" if i < cut2 else "test") for i, f in enumerate(fams)}
    out = {"train": [], "dev": [], "test": []}
    for p in pairs:
        out[where[p.family]].append(p)
    for k in out:
        out[k] = balance_labels(out[k], rng)
    return out


def twins_of(sentences: Sequence[Sentence]) -> Dict[int, Sentence]:
    """Map each sentence index to the other member of its minimal pair (if present)."""
    by_fam: Dict[str, List[int]] = {}
    for i, s in enumerate(sentences):
        by_fam.setdefault(s.family, []).append(i)
    out = {}
    for idx in by_fam.values():
        if len(idx) == 2:
            out[idx[0]] = sentences[idx[1]]; out[idx[1]] = sentences[idx[0]]
    return out


def decisive_filter(pairs: Sequence[Pair], dataset: str, competitor: str) -> List[Pair]:
    """Keep pairs on which the structural SCM and a competitor predict different
    intervened outputs (Appendix F.2)."""
    h = scm_for(dataset, "H" if dataset != "german" else "KO")
    c = scm_for(dataset, competitor)
    out = []
    for p in pairs:
        var_c = c.program.atomic if p.variable == h.program.atomic else p.variable
        lh, _, _ = label_pair(h, p.base, p.source, p.variable)
        lc, _, _ = label_pair(c, p.base, p.source, var_c)
        if lh != lc:
            out.append(p)
    return out
