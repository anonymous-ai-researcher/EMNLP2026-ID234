"""Token vocabularies and the map from tokens to B-RASP initial vectors (Appendix A.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from . import lexicon as lx


@dataclass
class Vocab:
    name: str
    tokens: List[str]

    def __post_init__(self):
        self.index: Dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        assert len(self.index) == len(self.tokens), f"duplicate token in {self.name}"

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def pad(self) -> int:
        return self.index[lx.PAD]

    def encode(self, words: Sequence[str]) -> List[int]:
        return [self.index[w] for w in words]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.tokens[i] for i in ids]


#: Sizes stated in the paper (Appendices C.1, D.1, D.8, D.10, F.2).  The decisive
#: English vocabulary is the core one plus the comma and *when*.
EXPECTED_SIZES = {"core": 135, "npi": 137, "concord_stress": 167, "npi3": 139, "boolean": 40,
                  "german": 66, "decisive": 137}
LABEL_TOKENS = ["<label_0>", "<label_1>"]


def build(name: str, label_tokens: bool = False) -> Vocab:
    """``label_tokens`` adds the two sentence-initial label tokens of Appendix F.3."""
    core = [lx.PAD, lx.EOS] + lx.english_core_words()
    if name == "core":
        toks = core
    elif name == "npi":
        toks = core + [w for w in lx.NPI_ONLY if w not in core]
    elif name == "npi3":
        toks = build("npi").tokens + lx.THREE_INPUT_EXTRA
    elif name == "concord_stress":
        toks = core + [w for slot in lx.MODIFIER_SLOTS for w in slot]
    elif name == "decisive":
        toks = core + lx.DECISIVE_EXTRA
    elif name == "boolean":
        toks = [lx.PAD] + lx.BOOL_SPECIAL + lx.BOOL_INPUTS + lx.BOOL_DISTRACTORS
    elif name == "german":
        toks = [lx.PAD, lx.EOS] + lx.german_words()
    else:
        raise ValueError(name)
    v = Vocab(name, list(toks))
    assert len(v) == EXPECTED_SIZES[name], (name, len(v))
    if label_tokens:
        v = Vocab(name + "+label", v.tokens + LABEL_TOKENS)
    return v


#: Which vocabulary each dataset uses.
DATASET_VOCAB = {"agreement": "core", "binding": "core", "concord": "core", "npi": "npi", "npi3": "npi3",
                 "concord_stress": "concord_stress", "boolean": "boolean", "german": "german",
                 "decisive_agreement": "decisive", "decisive_binding": "decisive", "distance": "core"}


def vocab_for(dataset: str, label_tokens: bool = False) -> Vocab:
    return build(DATASET_VOCAB[dataset], label_tokens)


# ---------------------------------------------------------------------------
# B-RASP input encoding
# ---------------------------------------------------------------------------

_SG = {sg for sg, _ in lx.NOUNS + lx.SHIFT_NOUNS}
_PL = {pl for _, pl in lx.NOUNS + lx.SHIFT_NOUNS}
_VSG = {sg for sg, _ in lx.VERBS + lx.SHIFT_VERBS} | {"has"}
_VPL = {pl for _, pl in lx.VERBS + lx.SHIFT_VERBS} | {"have"}
_DSG = {sg for sg, _ in lx.DETERMINERS}
_DPL = {pl for _, pl in lx.DETERMINERS}
_RSG = {sg for sg, _ in lx.REFLEXIVES}
_RPL = {pl for _, pl in lx.REFLEXIVES}

ENGLISH_FEATURES = ["Q_noun", "Q_n_sg", "Q_n_pl", "Q_verb", "Q_v_pl", "Q_det", "Q_det_pl", "Q_lic",
                    "Q_rel", "Q_npi", "Q_refl", "Q_refl_pl", "Q_prep", "Q_comma", "Q_and",
                    "Q_numbered", "Q_any_pl"]


def english_env(words: Sequence[str], dataset: str, family: str = "") -> Dict[str, np.ndarray]:
    """Initial vectors of an English sentence.

    In NPI data *that* is the relative pronoun; elsewhere it is the singular
    demonstrative.  *the* is a number-neutral determiner.
    """
    n = len(words)
    f = {k: np.zeros(n, dtype=bool) for k in ENGLISH_FEATURES}
    npi_data = dataset in ("npi", "npi3")
    for i, w in enumerate(words):
        w = w.lower()
        if w in _SG or w in _PL:
            f["Q_noun"][i] = True
            f["Q_n_pl"][i] = w in _PL
            f["Q_n_sg"][i] = w in _SG
        elif w in _VSG or w in _VPL:
            f["Q_verb"][i] = True
            f["Q_v_pl"][i] = w in _VPL and w not in _VSG
        elif w == "that" and npi_data:
            f["Q_rel"][i] = True
        elif w in _DSG or w in _DPL or w == lx.NEUTRAL_DET:
            f["Q_det"][i] = True
            f["Q_det_pl"][i] = w in _DPL
        elif w in lx.LICENSORS:
            f["Q_lic"][i] = True
        elif w in lx.RELATIVE_PRONOUNS:
            f["Q_rel"][i] = True
        elif w in lx.NPIS:
            f["Q_npi"][i] = True
        elif w in _RSG or w in _RPL:
            f["Q_refl"][i] = True
            f["Q_refl_pl"][i] = w in _RPL
        elif w in lx.PREPOSITIONS:
            f["Q_prep"][i] = True
        elif w == ",":
            f["Q_comma"][i] = True
        elif w == "and":
            f["Q_and"][i] = True
    marked_det = f["Q_det"] & np.array([w.lower() != lx.NEUTRAL_DET for w in words])
    f["Q_numbered"] = f["Q_noun"] | f["Q_verb"] | marked_det | f["Q_refl"]
    f["Q_any_pl"] = f["Q_n_pl"] | f["Q_v_pl"] | f["Q_det_pl"] | f["Q_refl_pl"]
    f["_family"] = family
    return f


_GER_FORMS: Dict[str, tuple] = {}
for _sg, _pl, _dpl, _g in lx.GERMAN_NOUNS:
    _GER_FORMS[_sg] = (_g, False)
    _GER_FORMS[_pl] = (_g, True)
    _GER_FORMS[_dpl] = (_g, True)
_GER_VSG = {sg for sg, _ in lx.GERMAN_ACC_VERBS + lx.GERMAN_DAT_VERBS}
_GER_VPL = {pl for _, pl in lx.GERMAN_ACC_VERBS + lx.GERMAN_DAT_VERBS}

GERMAN_FEATURES = ["Q_noun", "Q_pl", "Q_masc", "Q_fem", "Q_neut", "Q_der", "Q_die", "Q_das", "Q_den",
                   "Q_dem", "Q_des", "Q_det", "Q_dass", "Q_verb", "Q_v_pl"]


def german_env(words: Sequence[str], family: str = "", gold_number: Dict[int, bool] = None
               ) -> Dict[str, np.ndarray]:
    """Word-level symbols.  A form shared by two cells of a paradigm (e.g. *Jungen*)
    takes its plural reading unless ``gold_number`` gives the number of that position."""
    n = len(words)
    f = {k: np.zeros(n, dtype=bool) for k in GERMAN_FEATURES}
    for i, w in enumerate(words):
        if w in _GER_FORMS:
            g, pl = _GER_FORMS[w]
            f["Q_noun"][i] = True
            f["Q_pl"][i] = pl if not gold_number or i not in gold_number else gold_number[i]
            f["Q_" + g][i] = True
        elif w in lx.GERMAN_DETERMINERS:
            f["Q_" + w][i] = True
            f["Q_det"][i] = True
        elif w == "dass":
            f["Q_dass"][i] = True
        elif w in _GER_VSG or w in _GER_VPL:
            f["Q_verb"][i] = True
            f["Q_v_pl"][i] = w in _GER_VPL
    f["_family"] = family
    return f


def boolean_env(words: Sequence[str]) -> Dict[str, np.ndarray]:
    n = len(words)
    f = {k: np.zeros(n, dtype=bool) for k in ["Q_X", "Q_X1", "Q_Y", "Q_Y1"]}
    for i, w in enumerate(words):
        if w in ("X0", "X1"):
            f["Q_X"][i] = True
            f["Q_X1"][i] = w == "X1"
        if w in ("Y0", "Y1"):
            f["Q_Y"][i] = True
            f["Q_Y1"][i] = w == "Y1"
    return f


def env_for(words: Sequence[str], dataset: str, family: str = "", **kw) -> Dict[str, np.ndarray]:
    if dataset == "german":
        return german_env(words, family, kw.get("gold_number"))
    if dataset == "boolean":
        return boolean_env(words)
    return english_env(words, dataset, family)
