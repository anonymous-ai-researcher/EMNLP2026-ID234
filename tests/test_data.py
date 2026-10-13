import random

from spec2verify.data import datasets as D
from spec2verify.data import pairs as PR
from spec2verify.data import pretrained_templates as PT
from spec2verify.data import templates as T
from spec2verify.data import vocab as V


def acc(scm, sents):
    return sum(scm.output_at(PR._env(s), s.target) == s.label for s in sents) / len(sents)


def test_vocabulary_sizes():
    for name, n in V.EXPECTED_SIZES.items():
        assert len(V.build(name)) == n


def test_core_datasets_sizes_balance_and_splits():
    for ph in ("agreement", "npi", "binding", "concord"):
        d = D.core_dataset(ph)
        assert tuple(len(d[s]) for s in D.SPLITS) == D.CORE_SIZES[ph]
        assert abs(sum(s.label for s in d["train"]) / len(d["train"]) - 0.5) < 1e-9
        fam = {s: {x.family for x in d[s]} for s in D.SPLITS}
        assert not (fam["train"] & fam["test"]) and not (fam["train"] & fam["val"])
        assert max(len(s.words) for s in d["train"]) <= 30
        assert acc(PR.scm_for(ph), d["test"]) == 1.0


def test_npi_pairs_share_token_multisets():
    rng = random.Random(0)
    for _ in range(200):
        f = T.npi(rng, 0, "matrix_vs_embedded")
        assert sorted(f.good.words) == sorted(f.bad.words)


def test_other_suites_follow_their_programs():
    assert acc(PR.scm_for("concord_stress"), D.concord_stress_dataset((900, 90, 180))["train"]) == 1.0
    assert acc(PR.scm_for("npi3"), D.npi3_dataset((800, 80, 160))["train"]) == 1.0
    e = D.english_decisive_dataset("agreement", True)
    for tpl, ss in e["test"].items():
        assert acc(PR.scm_for("decisive_agreement", "H"), ss) == 1.0
        assert 0.4 < acc(PR.scm_for("decisive_agreement", "L"), ss) < 0.6


def test_german_linear_rule_matches_word_order_share():
    g = D.german_dataset("R2")
    lin = PR.scm_for("german", "L")
    assert round(acc(lin, g["test"][("F2", 0)]), 2) == 0.33
    assert round(acc(lin, g["test"][("F3", 0)]), 2) == 0.30
    for k, ss in g["test"].items():
        if k[0] != "F5":
            assert acc(PR.scm_for("german", "KO"), ss) == 1.0


def test_pairs_balanced_and_family_split():
    d = D.core_dataset("agreement", (2000, 200, 400))
    rng = random.Random(1)
    for var in PR.VARIABLES["agreement"]:
        pairs = PR.make_pairs(d["train"], var, rng, PR.scm_for("agreement"))
        assert abs(sum(p.label for p in pairs) / len(pairs) - 0.5) < 1e-9
        sp = PR.split_by_family(pairs, rng)
        fams = {k: {p.family for p in v} for k, v in sp.items()}
        assert not (fams["train"] & fams["test"])


def test_pretrained_templates():
    for ph in PT.GENERATORS:
        fams = PT.pretrained_pool(ph, 2, 0)
        assert len({f.good.template for f in fams}) == PT.N_TEMPLATES
        assert acc(PR.scm_for(ph), [s for f in fams for s in f.sentences]) == 1.0
        assert all(s.words[-1] == "." and s.words[0][0].isupper() for f in fams for s in f.sentences)


def test_combination_split():
    c = D.combination_dataset("agreement")
    n = sum(len(v) for v in c.values())
    assert 0.55 < len(c["train"]) / n < 0.65
