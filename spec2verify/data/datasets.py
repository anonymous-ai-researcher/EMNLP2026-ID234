"""Dataset assembly: quotas, seeds and sentence-family splits (Appendix B.4 and the
dataset descriptions of Appendices D.1, D.8, D.10, F.2, F.3 and F.7).

Splits are made by *sentence family* (both members of a minimal pair, and every
edit of a sentence, stay in one split).  A family pool is generated per stratum,
deduplicated, and partitioned in proportion to the split quotas.  When a stratum
has fewer distinct families than its quota (e.g. agreement at load 0, or the
twelve-noun German lexicon), records are drawn with replacement *within* the
split's own pool, so no sentence ever crosses splits.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import lexicon as lx
from . import templates as T
from .templates import Family, Sentence

SPLITS = ("train", "val", "test")

#: Data-generation seeds (Appendix B.4).
SEEDS = {"core": 42, "npi": 20270723, "concord_stress": 20270722}

#: Sentence counts (train, validation, test) of Table 7.
CORE_SIZES = {"agreement": (50_000, 5_000, 10_000), "npi": (40_000, 4_000, 8_000),
              "binding": (40_000, 4_000, 8_000), "concord": (40_000, 4_000, 8_000)}


def stream(seed: int, *keys) -> random.Random:
    """Independent, reproducible random stream for a (seed, key...) combination."""
    return random.Random("|".join([str(seed)] + [str(k) for k in keys]))


def family_pool(make: Callable[[random.Random], Family], rng: random.Random, needed: int,
                max_tries: Optional[int] = None, exclude: Iterable[Tuple] = ()) -> List[Family]:
    """Up to ``needed`` distinct families (fewer if the lexical space is exhausted)."""
    seen = set(exclude); out: List[Family] = []
    tries = 0; max_tries = max_tries or max(20 * needed, 2_000)
    while len(out) < needed and tries < max_tries:
        f = make(rng); tries += 1
        if f.key in seen:
            continue
        seen.add(f.key); out.append(f)
    return out


def split_families(pool: List[Family], quotas: Sequence[int], rng: random.Random
                   ) -> Dict[str, List[Family]]:
    """Partition a pool by family in proportion to ``quotas`` and fill each quota,
    drawing with replacement inside a split only when its pool is too small."""
    pool = list(pool); rng.shuffle(pool)
    total = sum(quotas); out: Dict[str, List[Family]] = {}
    start = 0
    for k, (name, q) in enumerate(zip(SPLITS, quotas)):
        share = len(pool) - start if k == len(quotas) - 1 else max(1, round(len(pool) * q / total))
        part = pool[start:start + share]; start += share
        if not q:
            out[name] = []
        elif len(part) >= q:
            out[name] = part[:q]
        else:
            out[name] = part + [rng.choice(part) for _ in range(q - len(part))]
    return out


def even(total: int, k: int, i: int) -> int:
    """The i-th of k near-equal integer shares of ``total``."""
    return total // k + (1 if i < total % k else 0)


def to_sentences(fams: List[Family]) -> List[Sentence]:
    return [s for f in fams for s in f.sentences]


# ----------------------------------------------------------------------------- core datasets
def core_dataset(phenomenon: str, sizes: Optional[Tuple[int, int, int]] = None,
                 seed: Optional[int] = None) -> Dict[str, List[Sentence]]:
    """Agreement, NPI, binding or concord; loads 0..8 evenly where the template has them."""
    sizes = sizes or CORE_SIZES[phenomenon]
    seed = seed if seed is not None else SEEDS["npi" if phenomenon == "npi" else "core"]
    levels = T.LOAD_LEVELS[phenomenon]
    gen = T.CORE_GENERATORS[phenomenon]
    out = {s: [] for s in SPLITS}
    for li, load in enumerate(levels):
        quotas = [even(n // 2, len(levels), li) for n in sizes]   # families per split
        rng = stream(seed, phenomenon, load)
        pool = family_pool(lambda r: gen(r, load), rng, int(sum(quotas) * 1.0))
        for split, fams in split_families(pool, quotas, rng).items():
            out[split] += to_sentences(fams)
    for split in SPLITS:
        stream(seed, phenomenon, split, "order").shuffle(out[split])
    return out


def concord_stress_dataset(sizes=(45_000, 4_500, 9_000), seed: int = SEEDS["concord_stress"]):
    """Appendix D.8: 0..8 prenominal modifiers, evenly over the nine levels."""
    out = {s: [] for s in SPLITS}
    for m in range(9):
        quotas = [even(n // 2, 9, m) for n in sizes]
        rng = stream(seed, "concord_stress", m)
        pool = family_pool(lambda r: T.concord(r, 0, m), rng, sum(quotas))
        for split, fams in split_families(pool, quotas, rng).items():
            out[split] += to_sentences(fams)
    return out


def npi3_dataset(sizes=(40_000, 4_000, 8_000), seed: int = SEEDS["npi"]):
    """Appendix D.1: balanced over the eight (x1, x2, c) input combinations."""
    out = {s: [] for s in SPLITS}
    combos = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    for combo in combos:
        quotas = [n // 8 for n in sizes]
        rng = stream(seed, "npi3", combo)
        pool = family_pool(lambda r: T.npi_three_input(r, combo), rng, sum(quotas))
        for split, fams in split_families(pool, quotas, rng).items():
            # keep the member whose inputs are this combination
            licensed = combo[0] if combo[2] == 0 else combo[1]
            out[split] += [f.good if licensed else f.bad for f in fams]
    return out


def boolean_dataset(fn: Callable[[int, int], int], sizes=(32_000, 4_000, 8_000), seed: int = 42):
    """Appendix D.10: uniform over the four input combinations; distractor sequences are
    not shared across splits."""
    rng = stream(seed, "boolean"); out = {s: [] for s in SPLITS}; used = set()
    for split, n in zip(SPLITS, sizes):
        for a in (0, 1):
            for b in (0, 1):
                k = 0
                while k < n // 4:
                    s = T.boolean(rng, fn, a, b)
                    key = tuple(s.words)
                    if key in used:
                        continue
                    used.add(key); out[split].append(s); k += 1
    return out


def distance_dataset(seed: int = SEEDS["core"]):
    """Appendix F.7: train 10,000 and dev 1,000 sentences per distance 0..4; 2,000 test
    items per distance 0..16."""
    out = {"train": [], "val": [], "test": {}}
    for d in range(17):
        rng = stream(seed, "distance", d)
        quotas = (5_000, 500, 1_000) if d <= 4 else (0, 0, 1_000)
        pool = family_pool(lambda r: T.distance(r, d), rng, sum(quotas))
        parts = split_families(pool, quotas, rng)
        out["train"] += to_sentences(parts["train"]); out["val"] += to_sentences(parts["val"])
        out["test"][d] = to_sentences(parts["test"])
    return out


# ----------------------------------------------------------------------------- F.2 decisive
def english_decisive_dataset(phenomenon: str, mixed: bool, seed: int = SEEDS["core"]):
    """Canonical-only: the core training set.  Mixed: 35,000 canonical and 15,000 decisive
    sentences (5,000 per template); dev 5,000 with 1,500 decisive; 1,000 test pairs per template."""
    core = core_dataset(phenomenon, sizes=(50_000, 5_000, 10_000) if phenomenon == "agreement" else None)
    for split in SPLITS:
        for s in core[split]:
            s.dataset = f"decisive_{phenomenon}"
    out = {"train": list(core["train"]), "val": list(core["val"]), "test": {}}
    for tpl in T.DECISIVE_TEMPLATES:
        rng = stream(seed, "decisive", phenomenon, tpl)
        pool = family_pool(lambda r: T.decisive_english(r, tpl, phenomenon), rng, 2_500 + 250 + 1_000)
        parts = split_families(pool, (2_500, 250, 1_000), rng)
        if mixed:
            out["train"] = out["train"][:len(out["train"]) - 5_000] + to_sentences(parts["train"])
            out["val"] = out["val"][:len(out["val"]) - 500] + to_sentences(parts["val"])
        out["test"][tpl] = to_sentences(parts["test"])
    return out


#: Share of F2 and F3 test pairs whose first noun phrase has the subject's number.
SAME_NUMBER_SHARE = {"F2": 0.33, "F3": 0.30}

GERMAN_STRATA = [("F1", 0), ("F2", 0), ("F3", 0)] + [("F4", d) for d in range(5)]


def german_dataset(regime: str, seed: int = SEEDS["core"]):
    """R1 canonical-only: F1, 40,000 train / 4,000 dev sentences.  R2 mixed: the eight strata
    F1, F2, F3, F4 (d = 0..4) in equal shares (5,000 train + 500 dev each).  Test: 1,000
    minimal pairs per stratum and 512 F5 contexts.  F1 and F4 with d = 0 are the same frame
    and share family identifiers."""
    out = {"train": [], "val": [], "test": {}}
    for cond, d in GERMAN_STRATA:
        rng = stream(seed, "german", cond, d)
        frame = "F1" if (cond, d) == ("F4", 0) else cond
        tr = (20_000, 2_000) if regime == "R1" and cond == "F1" else ((2_500, 250) if regime == "R2" else (0, 0))
        quotas = (tr[0], tr[1], 1_000)
        pool = family_pool(lambda r: T.german(r, frame, d), rng, sum(quotas))
        parts = split_families(pool, quotas, rng)
        out["train"] += to_sentences(parts["train"]); out["val"] += to_sentences(parts["val"])
        if cond in SAME_NUMBER_SHARE:
            # the share of test pairs whose first noun phrase has the subject's number
            # (the object is singular, so: singular subjects) is fixed by design
            n_sg = round(1_000 * SAME_NUMBER_SHARE[cond])
            test = []
            for plural, n in ((0, n_sg), (1, 1_000 - n_sg)):
                r2 = stream(seed, "german-test", cond, plural)
                fams = family_pool(lambda r: T.german(r, cond, 0, plural), r2, n,
                                   exclude={f.key for f in parts["train"] + parts["val"]})
                test += fams + [r2.choice(fams) for _ in range(n - len(fams))]
            out["test"][(cond, d)] = to_sentences(test)
        else:
            out["test"][(cond, d)] = to_sentences(parts["test"])
    rng = stream(seed, "german", "F5")
    out["test"][("F5", 0)] = [f.good for f in family_pool(lambda r: T.german(r, "F5"), rng, 512)]
    return out


# ----------------------------------------------------------------------------- F.3 held-out combinations
def lemma_index() -> Tuple[Dict[str, int], Dict[str, int]]:
    nouns = sorted(sg for sg, _ in lx.NOUNS)
    verbs = sorted(sg for sg, _ in lx.VERBS)
    ni = {}
    for sg, pl in lx.NOUNS:
        ni[sg] = ni[pl] = nouns.index(sg)
    vi = {}
    for sg, pl in lx.VERBS:
        vi[sg] = vi[pl] = verbs.index(sg)
    vi["admired"] = vi["admires"]; vi["watched"] = vi["watches"]
    return ni, vi


def combination_split(sentence: Sentence) -> str:
    """Appendix F.3: the 500 noun-verb lemma combinations go 300/100/100 to train, dev and
    test by (i + j) mod 5 over sorted lemma indices.  The key pairs the subject, antecedent
    or head noun with the sentence's main (or relative-clause) verb; a sentence without a
    content verb is assigned by its noun index alone."""
    ni, vi = lemma_index()
    noun = next((w for w in sentence.words if w in ni), None)
    verb = next((w for w in sentence.words if w in vi), None)
    g = (ni.get(noun, 0) + (vi[verb] if verb is not None else 0)) % 5
    return "train" if g < 3 else ("val" if g == 3 else "test")


def combination_dataset(phenomenon: str, seed: int = SEEDS["core"]):
    """Training-objective runs of Appendix F.3: the core data re-split by held-out combinations."""
    data = core_dataset(phenomenon, seed=seed)
    allsent = [s for split in SPLITS for s in data[split]]
    out = {s: [] for s in SPLITS}
    for s in allsent:
        out[combination_split(s)].append(s)
    return out
