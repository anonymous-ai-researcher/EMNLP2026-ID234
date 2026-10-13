"""Template generators (Appendix B.1; Appendices D.1, D.8, D.10, F.2, F.7).

Every generator returns *families*: one lexical assignment that yields a
grammatical sentence and its minimally different ungrammatical twin.  The
classification target is the position where the prescribed compositional
variable is evaluated (the verb, the NPI, the reflexive or the head noun).

Lexical choices are drawn with a ``random.Random`` instance, so a family is a
deterministic function of the generator's seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import lexicon as lx


@dataclass
class Sentence:
    words: List[str]
    label: int                      # 1 grammatical, 0 ungrammatical
    target: int                     # classification / consumption position
    family: str                     # family identifier shared by both members
    dataset: str
    load: int = 0
    template: str = ""
    meta: Dict = field(default_factory=dict)


@dataclass
class Family:
    key: Tuple                      # uniquely identifies the lexical assignment
    good: Sentence
    bad: Sentence

    @property
    def sentences(self) -> List[Sentence]:
        return [self.good, self.bad]


def _num(rng: random.Random) -> int:
    return rng.randrange(2)          # 0 singular, 1 plural


def _form(pair: Tuple[str, str], plural: int) -> str:
    return pair[plural]


def _pp_chain(rng: random.Random, d: int, opposite_to: Optional[int] = None) -> List[str]:
    """``(Prep the N)^d`` with place nouns of random number."""
    out: List[str] = []
    for _ in range(d):
        n = rng.choice(lx.PLACE_NOUNS)
        num = _num(rng) if opposite_to is None else 1 - opposite_to
        out += [rng.choice(lx.PP_PREPOSITIONS), lx.NEUTRAL_DET, _form(n, num)]
    return out


def _family(dataset: str, key: Tuple, good: List[str], bad: List[str], target: int, load: int = 0,
            template: str = "", meta: Optional[Dict] = None) -> Family:
    fid = f"{dataset}:" + "|".join(map(str, key))
    m = dict(meta or {})
    return Family(key, Sentence(good + [lx.EOS], 1, target, fid, dataset, load, template, m),
                  Sentence(bad + [lx.EOS], 0, target, fid, dataset, load, template, m))


# ----------------------------------------------------------------------------- core English
def agreement(rng: random.Random, load: int) -> Family:
    """S -> NP_subj PP^d VP;  NP_subj -> the N;  PP -> Prep the N;  VP -> V Adv."""
    n = rng.choice(lx.ANIMATE_NOUNS); sn = _num(rng)
    pp = _pp_chain(rng, load)
    v = rng.choice(lx.INTRANSITIVE); adv = rng.choice(lx.ADVERBS)
    prefix = [lx.NEUTRAL_DET, _form(n, sn)] + pp
    good = prefix + [_form(v, sn), adv]
    bad = prefix + [_form(v, 1 - sn), adv]
    return _family("agreement", tuple(good), good, bad, len(prefix), load,
                   meta={"subj_pl": sn, "subj_pos": 1})


def binding(rng: random.Random, load: int) -> Family:
    """S -> NP_subj PP^d V_trans,past Refl."""
    n = rng.choice(lx.ANIMATE_NOUNS); sn = _num(rng)
    pp = _pp_chain(rng, load)
    v = rng.choice(lx.BINDING_VERBS); r = rng.choice(lx.BINDING_REFLEXIVES)
    prefix = [lx.NEUTRAL_DET, _form(n, sn)] + pp + [v]
    good = prefix + [_form(r, sn)]
    bad = prefix + [_form(r, 1 - sn)]
    return _family("binding", tuple(good), good, bad, len(prefix), load,
                   meta={"subj_pl": sn, "subj_pos": 1})


def _a_ok(word: str) -> bool:
    """*a* is grammatical before this word (consonant sound)."""
    return word[0] not in "aeiou" or word.startswith("use")


def concord(rng: random.Random, load: int = 0, modifiers: int = 0) -> Family:
    """S -> Det_{sg/pl} Mod^m N VP.  The core template has m = 0 (adjacent determiner and
    noun); the separation suite of Appendix D.8 fills m distinct slots in their fixed order."""
    n = rng.choice(lx.NOUNS); nn = _num(rng)
    slots = sorted(rng.sample(range(len(lx.MODIFIER_SLOTS)), modifiers))
    mods = [rng.choice(lx.MODIFIER_SLOTS[s]) for s in slots]
    v = rng.choice(lx.INTRANSITIVE); adv = rng.choice(lx.ADVERBS)
    tail = mods + [_form(n, nn), _form(v, nn), adv]
    d = rng.choice([p for p in lx.DETERMINERS if p[0] != "a" or _a_ok(tail[0])])
    good = [_form(d, nn)] + tail
    bad = [_form(d, 1 - nn)] + tail
    ds = "concord_stress" if modifiers else "concord"
    return _family(ds, tuple(good), good, bad, 1 + len(mods), modifiers,
                   meta={"det_pl": nn, "det_pos": 0})


def _np_licensor(rng: random.Random, plural: int) -> str:
    opts = [w for w, nums in lx.NP_LICENSORS.items() if ("pl" if plural else "sg") in nums]
    return rng.choice(opts)


def _aux(rng: random.Random, plural: int) -> str:
    return rng.choice([("has", "have")[plural], "had"])


NPI_PAIR_TYPES = ("matrix_vs_embedded", "matrix_vs_unlicensed_with_rc", "matrix_vs_unlicensed_no_rc")


def npi(rng: random.Random, load: int = 0, pair_type: Optional[str] = None) -> Family:
    """Scope-sensitive NPI templates (Appendix B.1).

    S_matrix -> Lic N RC VP_npi;  S_embed -> the N Rel V_trans Lic N VP_npi;
    S_unlic -> the N RC? VP_npi;  RC -> Rel V_trans the N;  VP_npi -> Aux ever responded.
    Determiners are the number-neutral *the* so that *that* is unambiguously relative.
    """
    pair_type = pair_type or rng.choice(NPI_PAIR_TYPES)
    n1 = rng.choice(lx.ANIMATE_NOUNS); s1 = _num(rng)
    n2 = rng.choice([x for x in lx.ANIMATE_NOUNS if x != n1]); s2 = _num(rng)
    rel = rng.choice(lx.RELATIVE_PRONOUNS); vt = rng.choice(lx.TRANSITIVE)
    aux = _aux(rng, s1)
    vp = [aux, "ever", lx.PAST_PARTICIPLE]
    lic1 = _np_licensor(rng, s1)
    head = _form(n1, s1)
    rc_verb = _form(vt, s1)
    if pair_type == "matrix_vs_embedded":
        # one licensor compatible with both nouns, so the two members share a token multiset
        lic = "no" if (s1 or s2) else rng.choice(sorted(lx.NP_LICENSORS))
        good = [lic, head, rel, rc_verb, lx.NEUTRAL_DET, _form(n2, s2)] + vp
        bad = [lx.NEUTRAL_DET, head, rel, rc_verb, lic, _form(n2, s2)] + vp
    elif pair_type == "matrix_vs_unlicensed_with_rc":
        good = [lic1, head, rel, rc_verb, lx.NEUTRAL_DET, _form(n2, s2)] + vp
        bad = [lx.NEUTRAL_DET] + good[1:]
    else:
        good = [lic1, head] + vp
        bad = [lx.NEUTRAL_DET, head] + vp
    target = len(good) - 2
    return _family("npi", (pair_type,) + tuple(good), good, bad, target, 0, pair_type,
                   meta={"pair_type": pair_type})


CORE_GENERATORS: Dict[str, Callable[[random.Random, int], Family]] = {
    "agreement": agreement, "npi": npi, "binding": binding, "concord": concord,
}
#: Structural-load levels of each core template (Appendix B.4).
LOAD_LEVELS = {"agreement": list(range(9)), "binding": list(range(9)), "npi": [0], "concord": [0]}


# ----------------------------------------------------------------------------- D.1: three inputs
def npi_three_input(rng: random.Random, combo: Optional[Tuple[int, int, int]] = None) -> Family:
    """[L1 N1 Aux A1 responded] and [L2 N2 Aux A2 responded]; exactly one A is *ever*,
    the other *once*; x1, x2: licensor in clause 1, 2; c: clause of the NPI.
    The ungrammatical twin flips the licensor status of the NPI's clause."""
    x1, x2, c = combo if combo else (rng.randrange(2), rng.randrange(2), rng.randrange(2))

    def clause(lic: int, adv: str) -> List[str]:
        n = rng.choice(lx.ANIMATE_NOUNS); s = _num(rng)
        det = _np_licensor(rng, s) if lic else lx.NEUTRAL_DET
        return [det, _form(n, s), _aux(rng, s), adv, lx.PAST_PARTICIPLE]

    a1, a2 = ("ever", "once") if c == 0 else ("once", "ever")
    c1, c2 = clause(x1, a1), clause(x2, a2)
    words = c1 + ["and"] + c2
    label = int((x1 if c == 0 else x2))
    target = 3 if c == 0 else len(c1) + 1 + 3
    flipped = list(words)
    i = 0 if c == 0 else len(c1) + 1
    plural = words[i + 1] in {p for _, p in lx.NOUNS}
    flipped[i] = lx.NEUTRAL_DET if words[i] != lx.NEUTRAL_DET else _np_licensor(rng, int(plural))
    good, bad = (words, flipped) if label else (flipped, words)
    return _family("npi3", (x1, x2, c) + tuple(words), good, bad, target, 0,
                   meta={"x1": x1, "x2": x2, "c": c})


# ----------------------------------------------------------------------------- D.10: Boolean tasks
def boolean(rng: random.Random, fn: Callable[[int, int], int], a: Optional[int] = None,
            b: Optional[int] = None) -> Sentence:
    """<bos> X_a F F Y_b F <read> <eos>; the label f(a, b) is read at <read>."""
    a = rng.randrange(2) if a is None else a
    b = rng.randrange(2) if b is None else b
    f = rng.sample(lx.BOOL_DISTRACTORS, 3)
    words = ["<bos>", f"X{a}", f[0], f[1], f"Y{b}", f[2], "<read>", "<eos>"]
    return Sentence(words, int(fn(a, b)), 6, "bool:" + " ".join(words), "boolean", 0, "",
                    {"a": a, "b": b})


# ----------------------------------------------------------------------------- F.7: distance data
def distance(rng: random.Random, d: int) -> Family:
    """Det N (Prep Det N)^d V Adv <eos>; the verb sits at position 3d + 2 (0-based)."""
    f = agreement(rng, d)
    for s in f.sentences:
        s.dataset = "distance"
    return f


# ----------------------------------------------------------------------------- F.2: English decisive
DECISIVE_TEMPLATES = ("T-A", "T-B", "T-C")


def decisive_english(rng: random.Random, template: str, phenomenon: str = "agreement",
                     numbers: Optional[Tuple[int, ...]] = None) -> Family:
    """T-A: Near the N1 the N2 by the N3 V;  T-B: when the N1 V1 , the N2 by the N3 V;
    T-C: Near the N1 the N2 V.  N2 is the subject; the numbers of all nouns are crossed."""
    k = 2 if template == "T-C" else 3
    nums = numbers or tuple(_num(rng) for _ in range(k))
    place1 = rng.choice(lx.PLACE_NOUNS); subj = rng.choice(lx.ANIMATE_NOUNS)
    if template == "T-B":
        n1 = rng.choice(lx.ANIMATE_NOUNS)
        v1 = rng.choice(lx.INTRANSITIVE[:4])
        front = ["when", lx.NEUTRAL_DET, _form(n1, nums[0]), _form(v1, nums[0]), ","]
    else:
        front = [rng.choice(lx.PP_PREPOSITIONS), lx.NEUTRAL_DET, _form(place1, nums[0])]
    core = [lx.NEUTRAL_DET, _form(subj, nums[1])]
    if template != "T-C":
        core += [rng.choice(lx.PP_PREPOSITIONS), lx.NEUTRAL_DET, _form(rng.choice(lx.PLACE_NOUNS), nums[2])]
    prefix = front + core
    sn = nums[1]
    if phenomenon == "agreement":
        v = rng.choice(lx.INTRANSITIVE)
        good, bad = prefix + [_form(v, sn)], prefix + [_form(v, 1 - sn)]
    else:
        r = rng.choice(lx.BINDING_REFLEXIVES); pv = [rng.choice(lx.BINDING_VERBS)]
        good, bad = prefix + pv + [_form(r, sn)], prefix + pv + [_form(r, 1 - sn)]
    return _family(f"decisive_{phenomenon}", (template,) + tuple(good), good, bad, len(good) - 1, 0,
                   template, meta={"numbers": nums, "template": template, "subj_pos": len(front) + 1})


# ----------------------------------------------------------------------------- F.2: German
GERMAN_CONDITIONS = ("F1", "F2", "F3", "F4", "F5")


def german(rng: random.Random, condition: str, distance_d: int = 0,
           subject_plural: Optional[int] = None) -> Family:
    """Ich glaube , dass [clause] .  Minimal pairs change only the finite verb's number.

    F1: S O_acc V.  F2: O_acc (masculine, *den*) S V.  F3: O_dat (feminine, *der Frau*) S V_dat.
    F4: S PP_dat^d O_acc V (d = 0..4; PP nouns take the number opposite to the subject's).
    F5: two nominative-compatible NPs of opposite number; both verb forms are grammatical.
    """
    clause: List[str] = []
    gold: Dict[int, bool] = {}
    offset = len(lx.GERMAN_FRAME)

    nps: List[Tuple[int, int, Tuple, str]] = []

    def np_(case: str, plural: int, gender: Optional[str] = None) -> int:
        pool = [n for n in lx.GERMAN_NOUNS if gender is None or n[3] == gender]
        noun = rng.choice(pool)
        clause.append(lx.german_article(case, noun[3], bool(plural)))
        clause.append(lx.german_noun_form(noun, case, bool(plural)))
        gold[offset + len(clause) - 1] = bool(plural)
        nps.append((offset + len(clause) - 2, offset + len(clause) - 1, noun, case))
        return offset + len(clause) - 1

    sn = _num(rng) if subject_plural is None else subject_plural
    if condition == "F1":
        subj = np_("nom", sn); np_("acc", _num(rng)); v = rng.choice(lx.GERMAN_ACC_VERBS)
    elif condition == "F2":
        np_("acc", 0, "masc"); subj = np_("nom", sn); v = rng.choice(lx.GERMAN_ACC_VERBS)
    elif condition == "F3":
        np_("dat", 0, "fem"); subj = np_("nom", sn); v = rng.choice(lx.GERMAN_DAT_VERBS)
    elif condition == "F4":
        subj = np_("nom", sn)
        for _ in range(distance_d):
            clause.append(rng.choice(lx.GERMAN_DAT_PREPOSITIONS)); np_("dat", 1 - sn)
        np_("acc", _num(rng)); v = rng.choice(lx.GERMAN_ACC_VERBS)
    elif condition == "F5":
        # die/das noun phrases are nominative or accusative alike; opposite numbers
        subj = np_("nom", sn, rng.choice(["fem", "neut"])); np_("acc", 1 - sn, rng.choice(["fem", "neut"]))
        v = rng.choice(lx.GERMAN_ACC_VERBS)
    else:
        raise ValueError(condition)
    prefix = list(lx.GERMAN_FRAME) + clause
    good = prefix + [v[sn], lx.GERMAN_PERIOD]
    bad = prefix + [v[1 - sn], lx.GERMAN_PERIOD]
    fam = _family("german", (condition, distance_d) + tuple(good), good, bad, len(prefix), distance_d,
                  condition, meta={"condition": condition, "subj_pl": sn, "gold_number": gold,
                                   "subj_pos": subj, "nps": nps, "verb": v})
    if condition == "F5":
        fam.bad.label = 1      # both continuations are grammatical; F5 measures preference
    return fam
