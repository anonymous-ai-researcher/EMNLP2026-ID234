"""Natural-text templates for the pre-trained models (Appendices C.3 and F.8).

Each phenomenon has 40 templates, indexed 0..39 and decoded into three factors
(5 x 4 x 2).  Sentences are capitalized and end with a period.  A sentence's
``target`` is its critical word (the verb, the NPI, the reflexive or the head
noun); the language model is read at the preceding position, where its
prediction of the critical word consumes the prescribed variable, and the
two candidate forms of that word are compared (``critical_alternative``).

The distribution shifts of Appendix F.8 are implemented here as well:
template (ten held-out outer frames), lexical (held-out nouns and verbs), load
(9-16 phrases or separators) and joint (all three).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from . import lexicon as lx
from .templates import Family, Sentence, _family, _form, _num

N_TEMPLATES = 40
NPI_ALTERNATIVE = {"ever": "once", "any": "some"}
SEPARATORS = ["really", "truly", "certainly", "clearly", "surely", "indeed", "quite", "very"]


def _factors(tid: int) -> Tuple[int, int, int]:
    return tid % 5, (tid // 5) % 4, tid // 20


def _pp(rng: random.Random, k: int) -> List[str]:
    out: List[str] = []
    for _ in range(k):
        out += [rng.choice(lx.PP_PREPOSITIONS), "the", _form(rng.choice(lx.PLACE_NOUNS), _num(rng))]
    return out


def _subject_modifier(rng: random.Random, kind: int, plural: int, nouns, verbs) -> List[str]:
    """0: none; 1: one PP; 2: two PPs; 3: object relative clause; 4: subject relative clause."""
    if kind in (1, 2):
        return _pp(rng, kind)
    if kind == 3:
        n2 = rng.choice(nouns); s2 = _num(rng)
        return ["that", "the", _form(n2, s2), _form(rng.choice(verbs), s2)]
    if kind == 4:
        n2 = rng.choice(nouns)
        return ["who", _form(rng.choice(verbs), plural), "the", _form(n2, _num(rng))]
    return []


def _finish(words: List[str]) -> List[str]:
    words = ["I" if w == "i" else w for w in words]
    return [words[0][0].upper() + words[0][1:]] + words[1:] + ["."]


def _pt_family(dataset: str, key, good: List[str], bad: List[str], target: int, load: int, template: str,
               meta: Dict) -> Family:
    """Pre-trained sentences: capitalized, period-final, no end-of-sentence token."""
    fam = _family(dataset, key, _finish(good), _finish(bad), target, load, template, meta)
    for s in fam.sentences:
        s.words = s.words[:-1]
    from .pairs import flip
    g, b = fam.good.words[target].lower(), fam.bad.words[target].lower()
    alt = NPI_ALTERNATIVE.get(g) or (flip(g) if g == b else None)
    fam.good.meta = dict(fam.good.meta, alternative=alt or b)
    fam.bad.meta = dict(fam.bad.meta, alternative=alt or g)
    return fam


def _wrap(rng: random.Random, words: List[str], frame: Optional[int]) -> Tuple[List[str], int]:
    """Template shift: one of ten outer frames; returns the words and the target offset."""
    if frame is None:
        return words, 0
    if frame < 5:
        return [lx.SHIFT_FRAMES_BEFORE[frame]] + words, 1
    return words + [lx.SHIFT_FRAMES_AFTER[frame - 5]], 0


def agreement(rng: random.Random, tid: int, lexical: bool = False, load: int = 0,
              frame: Optional[int] = None) -> Family:
    mod, pred, tail = _factors(tid)
    nouns = lx.SHIFT_NOUNS if lexical else lx.ANIMATE_NOUNS
    verbs = lx.SHIFT_VERBS if lexical else lx.TRANSITIVE
    n = rng.choice(nouns); sn = _num(rng)
    prefix = ["the", _form(n, sn)] + (_pp(rng, load) if load else _subject_modifier(rng, mod, sn, nouns, verbs))
    if pred == 0:
        v = rng.choice(lx.INTRANSITIVE[:4]); rest = [rng.choice(lx.ADVERBS)]
    elif pred == 1:
        v = rng.choice(verbs); rest = ["the", _form(rng.choice(nouns), _num(rng))]
    elif pred == 2:
        v = ("is", "are"); rest = ["here"]
    else:
        v = ("has", "have"); rest = ["left"]
    if tail:
        rest = rest + ["today"]
    good, bad = prefix + [_form(v, sn)] + rest, prefix + [_form(v, 1 - sn)] + rest
    good, off = _wrap(rng, good, frame); bad, _ = _wrap(rng, bad, frame)
    fam = _pt_family("agreement", ("pt", tid, frame, load) + tuple(good), good, bad,
                  len(prefix) + off, load, f"agr-{tid}", meta={"subj_pos": 1 + off, "subj_pl": sn})
    return fam


def binding(rng: random.Random, tid: int, lexical: bool = False, load: int = 0,
            frame: Optional[int] = None) -> Family:
    mod, verb, tail = _factors(tid)
    nouns = lx.SHIFT_NOUNS if lexical else lx.ANIMATE_NOUNS
    verbs = lx.SHIFT_VERBS if lexical else lx.TRANSITIVE
    n = rng.choice(nouns); sn = _num(rng)
    prefix = ["the", _form(n, sn)] + (_pp(rng, load) if load else _subject_modifier(rng, mod, sn, nouns, verbs))
    prefix += [["admired", "watched", "praised", "trusted"][verb]]
    r = rng.choice(lx.BINDING_REFLEXIVES)
    rest = ["today"] if tail else []
    good, bad = prefix + [_form(r, sn)] + rest, prefix + [_form(r, 1 - sn)] + rest
    good, off = _wrap(rng, good, frame); bad, _ = _wrap(rng, bad, frame)
    return _pt_family("binding", ("pt", tid, frame, load) + tuple(good), good, bad,
                   len(prefix) + off, load, f"bind-{tid}", meta={"subj_pos": 1 + off, "subj_pl": sn})


def concord(rng: random.Random, tid: int, lexical: bool = False, load: int = 0,
            frame: Optional[int] = None) -> Family:
    """Determiner--noun concord; the factors are the number of modifiers (0-4), the
    sentence frame of the noun phrase and the determiner pair family."""
    nmod, ctx, detfam = _factors(tid)
    nouns = lx.SHIFT_NOUNS if lexical else lx.NOUNS
    n = rng.choice(nouns); nn = _num(rng)
    k = load if load else nmod
    if load:
        mods = [rng.choice(SEPARATORS) for _ in range(max(0, k - 1))] + [rng.choice(lx.MODIFIER_SLOTS[1])]
    else:
        slots = sorted(rng.sample(range(8), k)); mods = [rng.choice(lx.MODIFIER_SLOTS[s]) for s in slots]
    nxt = (mods + [_form(n, nn)])[0]
    det = rng.choice(lx.DETERMINERS[:2] if detfam == 0 else
                     [p for p in lx.DETERMINERS[2:] if p[0] != "a" or nxt[0] not in "aeiou" or nxt.startswith("use")])
    lead = [[], ["i", "saw"], ["we", "talked", "about"], ["they", "liked"]][ctx]
    head = lead + [_form(det, nn)] + mods
    rest = [_form(rng.choice(lx.INTRANSITIVE[:4]), nn), rng.choice(lx.ADVERBS)] if ctx == 0 else []
    good = head + [_form(n, nn)] + rest
    bad = lead + [_form(det, 1 - nn)] + mods + [_form(n, nn)] + rest
    good, off = _wrap(rng, good, frame); bad, _ = _wrap(rng, bad, frame)
    fam = _pt_family("concord", ("pt", tid, frame, load) + tuple(good), good, bad,
                  len(head) + off, load, f"conc-{tid}", meta={"det_pos": len(lead) + off, "det_pl": nn})
    return fam


def npi(rng: random.Random, tid: int, lexical: bool = False, load: int = 0,
        frame: Optional[int] = None) -> Family:
    """Factors: licensor structure (matrix vs embedded, with *who* or *that*; matrix vs
    unlicensed with or without a relative clause; five types), the NPI frame (four) and
    the licensor (*no* or *neither*)."""
    struct, npiframe, licw = _factors(tid)
    nouns = lx.SHIFT_NOUNS if lexical else lx.ANIMATE_NOUNS
    verbs = lx.SHIFT_VERBS if lexical else lx.TRANSITIVE
    n1 = rng.choice(nouns); s1 = 0 if licw else _num(rng)
    n2 = rng.choice([x for x in nouns if x != n1]); s2 = 0 if licw else _num(rng)
    lic = "neither" if licw else "no"
    rel = "that" if struct == 1 else "who"
    rc = [rel, _form(rng.choice(verbs), s1), "the", _form(n2, s2)]
    sep = [rng.choice(SEPARATORS) for _ in range(load)]
    aux = ("has", "have")[s1]
    item, cont = [("ever", ["responded"]), ("ever", ["left"]), ("any", ["money"]), ("any", ["time"])][npiframe]
    verb_np = [aux] + sep if npiframe < 2 else [aux] + sep
    if npiframe >= 2:
        verb_np = [("has", "have")[s1]] + sep
    if struct in (0, 1):
        good = [lic, _form(n1, s1)] + rc + verb_np
        bad = ["the", _form(n1, s1)] + rc[:2] + [lic] + rc[3:] + verb_np
    elif struct == 2:
        good = [lic, _form(n1, s1)] + rc + verb_np; bad = ["the"] + good[1:]
    else:
        good = [lic, _form(n1, s1)] + verb_np; bad = ["the"] + good[1:]
    if npiframe >= 2:
        good = good + []; bad = bad + []
    target = len(good)
    good, bad = good + [item] + cont, bad + [item] + cont
    good, off = _wrap(rng, good, frame); bad, _ = _wrap(rng, bad, frame)
    return _pt_family("npi", ("pt", tid, frame, load) + tuple(good), good, bad,
                   target + off, load, f"npi-{tid}", meta={"lic_pos": off})


GENERATORS = {"agreement": agreement, "npi": npi, "binding": binding, "concord": concord}


def critical_alternative(s: Sentence) -> str:
    """The competing candidate for the critical word (compared under the language model):
    the minimal-pair twin's word, or for an NPI its non-polarity counterpart."""
    if "alternative" in s.meta:
        return s.meta["alternative"]
    from .pairs import flip
    w = s.words[s.target].lower()
    return NPI_ALTERNATIVE.get(w) or flip(w)


def pretrained_pool(phenomenon: str, per_template: int, seed: int, shift: str = "id",
                    exclude=()) -> List[Family]:
    """``per_template`` distinct families for each of the 40 templates.

    shift: id | template | lexical | load | joint (Appendix F.8)."""
    from .datasets import family_pool, stream
    gen = GENERATORS[phenomenon]
    out: List[Family] = []
    for tid in range(N_TEMPLATES):
        rng = stream(seed, "pretrained", phenomenon, shift, tid)

        def make(r, tid=tid):
            frame = r.randrange(10) if shift in ("template", "joint") else None
            load = r.randint(9, 16) if shift in ("load", "joint") else 0
            return gen(r, tid, lexical=shift in ("lexical", "joint"), load=load, frame=frame)
        out += family_pool(make, rng, per_template, exclude=exclude)
    return out
