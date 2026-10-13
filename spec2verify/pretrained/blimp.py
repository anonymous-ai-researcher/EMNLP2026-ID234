"""BLiMP evaluation (Appendices B.2 and E.3; Tables 6, 26, 30).

Behavioral accuracy uses the 1,000 original minimal pairs of each paradigm,
scored by sentence log-probability (``hf.minimal_pair_accuracy``).

Per-paradigm IIA uses counterfactual pairs built from each paradigm's frame:
items are split by pair ID 60/20/20 into 3,000/1,000/1,000 base-source pairs
(both directions equally often), with fixed sites and ranks per phenomenon
(Pythia-1.4B: atomic/compositional layer 9/15 agreement, 8/14 NPI, 9/15
binding, 7/13 concord; rank 1/2, and rank 2 for the three-valued gender
variable).  The adapter reads each pair as (context, critical word): the
critical word is the first word on which the two sentences differ, or, in the
NPI paradigms, the polarity item that follows the differing licensor.  The
context's class is the value the prescribed variable takes there (the number,
gender or licensing the critical word needs), and the counterfactual label of an
interchange from source s into base b is whether b's own critical word fits the
class of s's context.  Three groups use paradigm-specific variables:
``anaphor_gender_agreement`` (gender instead of number), the two ``only_npi``
paradigms (*only* as a licensor) and ``superlative_quantifiers_1``
(not(negative scope and superlative quantifier)).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..data.templates import Sentence
from ..data.pairs import Pair

PARADIGMS: Dict[str, List[str]] = {
    "agreement": ["irregular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_1",
                  "distractor_agreement_relational_noun", "distractor_agreement_relative_clause",
                  "irregular_plural_subject_verb_agreement_2", "regular_plural_subject_verb_agreement_2"],
    "npi": ["npi_present_1", "npi_present_2", "sentential_negation_npi_licensor_present",
            "sentential_negation_npi_scope", "only_npi_licensor_present", "only_npi_scope",
            "superlative_quantifiers_1"],
    "binding": ["anaphor_gender_agreement", "anaphor_number_agreement"],
    "concord": ["determiner_noun_agreement_1", "determiner_noun_agreement_2",
                "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2",
                "determiner_noun_agreement_with_adj_irregular_1", "determiner_noun_agreement_with_adj_2",
                "determiner_noun_agreement_with_adjective_1", "determiner_noun_agreement_with_adj_irregular_2"],
}
FIXED_LAYERS = {"pythia-1.4b": {"agreement": (9, 15), "npi": (8, 14), "binding": (9, 15), "concord": (7, 13)}}
NPI_WORDS = {"ever", "any", "anyone", "anything", "anybody", "yet"}
IRREGULAR_PLURALS = {"men", "women", "children", "people", "mice", "geese", "teeth", "feet", "oxen", "cacti",
                     "fungi", "stimuli", "theses", "analyses", "crises", "hypotheses", "oases", "alumni",
                     "radii", "bacteria", "criteria", "phenomena", "data", "media", "larvae", "vertebrae",
                     "dice", "lice", "nuclei", "appendices", "matrices", "indices"}
GENDER = {"himself": 0, "herself": 1, "itself": 2}


def load_paradigm(name: str, path: Optional[str] = None) -> List[Dict]:
    """Rows with sentence_good, sentence_bad and pair_id, from a local JSONL file
    (``path``) or from the Hugging Face dataset ``nyu-mll/blimp``."""
    if path:
        with open(path) as f:
            return [json.loads(l) for l in f]
    from datasets import load_dataset
    return list(load_dataset("nyu-mll/blimp", name)["train"])


def _words(s: str) -> List[str]:
    return s.replace(".", " .").replace(",", " ,").split()


def number_class(word: str) -> int:
    """0 singular / 1 plural for verbs, nouns and reflexives (inflection heuristics)."""
    w = word.lower()
    if w in ("themselves", "are", "were", "have", "do", "these", "those"):
        return 1
    if w in ("himself", "herself", "itself", "is", "was", "has", "does", "this", "that"):
        return 0
    if w in IRREGULAR_PLURALS:
        return 1
    return -1


@dataclass
class Item:
    sentence: Sentence
    context_class: int


def adapt(paradigm: str, row: Dict, phenomenon: str) -> List[Item]:
    """Both members of a BLiMP pair as (context, critical word) items."""
    g, b = _words(row["sentence_good"]), _words(row["sentence_bad"])
    diff = [i for i in range(min(len(g), len(b))) if g[i] != b[i]]
    if not diff:
        return []
    c = diff[0]
    if phenomenon == "npi" and paradigm != "superlative_quantifiers_1":
        after = [i for i in range(diff[-1], len(g)) if g[i].lower() in NPI_WORDS]
        c = after[0] if after else diff[-1]
    items = []
    for words, grammatical in ((g, 1), (b, 0)):
        if phenomenon == "npi":
            cls = grammatical                                   # licensed context or not
        elif paradigm == "anaphor_gender_agreement":
            if not grammatical:
                continue                                         # a mismatch does not fix the gender
            cls = GENDER.get(words[c].lower(), -1)
        else:
            wc = number_class(words[c])
            if wc < 0:
                # regular inflection: a verb ending in -s is singular, a noun ending in -s plural
                is_noun = phenomenon == "concord"
                wc = int(words[c].lower().endswith("s")) if is_noun else int(not words[c].lower().endswith("s"))
            cls = wc if grammatical else 1 - wc
        s = Sentence(words, grammatical, c, f"{paradigm}:{row.get('pair_id', 0)}", phenomenon, 0, paradigm,
                     {"alternative": (b if grammatical else g)[c] if phenomenon != "npi" else
                      {"ever": "once", "any": "some", "yet": "now"}.get(words[c].lower(), "once")})
        items.append(Item(s, cls))
    return items


def word_class(item: Item, paradigm: str, phenomenon: str) -> int:
    w = item.sentence.words[item.sentence.target]
    if phenomenon == "npi":
        return 1                                                  # the polarity item needs a license
    if paradigm == "anaphor_gender_agreement":
        return GENDER.get(w.lower(), -1)
    wc = number_class(w)
    if wc < 0:
        wc = int(w.lower().endswith("s")) if phenomenon == "concord" else int(not w.lower().endswith("s"))
    return wc


def paradigm_pairs(paradigm: str, rows: Sequence[Dict], phenomenon: str, seed: int = 0,
                   sizes=(3_000, 1_000, 1_000)) -> Dict[str, List[Pair]]:
    """Counterfactual pairs within each pair-ID split; labels balanced."""
    rng = random.Random(f"{seed}|{paradigm}")
    ids = sorted({r.get("pair_id", i) for i, r in enumerate(rows)}); rng.shuffle(ids)
    cut1, cut2 = int(0.6 * len(ids)), int(0.8 * len(ids))
    where = {pid: ("train" if i < cut1 else "dev" if i < cut2 else "test") for i, pid in enumerate(ids)}
    pools: Dict[str, List[Item]] = {"train": [], "dev": [], "test": []}
    for i, r in enumerate(rows):
        pools[where[r.get("pair_id", i)]] += adapt(paradigm, r, phenomenon)
    out = {}
    for split, n in zip(("train", "dev", "test"), sizes):
        items = pools[split]; pos, neg = [], []
        tries = 0
        while (len(pos) < n // 2 or len(neg) < n // 2) and tries < 50 * n and len(items) > 1:
            tries += 1
            b, s = rng.sample(items, 2)
            wc = word_class(b, paradigm, phenomenon)
            if wc < 0 or s.context_class < 0 or b.context_class == s.context_class:
                continue                                          # the variable must differ
            lab = int(wc == s.context_class)
            pr = Pair(b.sentence, s.sentence, "context", lab, int(wc == b.context_class), b.sentence.family)
            (pos if lab else neg).append(pr) if len(pos if lab else neg) < n // 2 else None
        out[split] = pos + neg
        rng.shuffle(out[split])
    return out
