from spec2verify.pretrained import blimp as BL


def rows(pairs):
    return [{"sentence_good": g, "sentence_bad": b, "pair_id": i} for i, (g, b) in enumerate(pairs)]


def test_adapter_builds_balanced_pairs():
    agr = rows([("The dogs near the car run.", "The dogs near the car runs."),
                ("The dog near the cars runs.", "The dog near the cars run."),
                ("The boys are here.", "The boys is here."),
                ("The boy is here.", "The boy are here.")] * 10)
    out = BL.paradigm_pairs("regular_plural_subject_verb_agreement_1", agr, "agreement", sizes=(20, 6, 6))
    assert out["train"] and abs(sum(p.label for p in out["train"]) / len(out["train"]) - 0.5) < 1e-9
    npi = rows([("No dog has ever run.", "The dog has ever run."),
                ("Nobody has ever left.", "Somebody has ever left.")] * 10)
    out = BL.paradigm_pairs("npi_present_1", npi, "npi", sizes=(10, 4, 4))
    assert all(p.base.words[p.base.target] == "ever" for p in out["train"])


def test_number_classes():
    assert BL.number_class("themselves") == 1 and BL.number_class("himself") == 0
    assert BL.number_class("children") == 1 and BL.number_class("runs") == -1
