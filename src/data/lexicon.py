"""Lexicon definitions for CFG-based data generation (Table 5, Appendix B.1).
25 noun pairs (50) + 20 verb pairs (40) + 4+3 det (7, "the" shared) + 8 prep
+ 3 lic + 3 npi + 4+3 refl (7, no dup) + 3 aux + 15 adv = 133 unique + 2 special = 135.
"""
NOUNS_SG = ["dog","cat","boy","girl","teacher","student","doctor","nurse","artist","writer",
            "farmer","driver","singer","dancer","player","manager","lawyer","engineer","professor",
            "captain","senator","officer","judge","pilot","mayor"]  # 25
NOUNS_PL = [n+"s" for n in NOUNS_SG]  # 25

VERBS_SG = ["runs","walks","reads","writes","sings","dances","plays","eats","sleeps","works",
            "speaks","drives","teaches","watches","knows","likes","needs","sees","hears","finds"]  # 20
VERBS_PL = ["run","walk","read","write","sing","dance","play","eat","sleep","work",
            "speak","drive","teach","watch","know","like","need","see","hear","find"]  # 20

TRANSITIVE_VERBS = ["admired","watched","followed","helped","trusted","avoided","noticed",
                    "praised","blamed","thanked","ignored","greeted","injured","amused","inspired"]  # 15 (subset, not in vocab separately)

DETERMINERS_SG = ["the","this","that","every"]  # 4
DETERMINERS_PL = ["the","these","those","many"]  # 4 ("the" overlaps -> 7 unique)

PREPOSITIONS = ["near","behind","beside","above","below","beneath","beyond","toward"]  # 8

LICENSORS = ["no","never","neither"]  # 3
NPIS = ["ever","any","anyone"]  # 3

REFLEXIVES_SG = ["himself","herself","itself","oneself"]  # 4
REFLEXIVES_PL = ["themselves","ourselves","yourselves"]  # 3 (no duplicate)

AUXILIARIES = ["has","had","would"]  # 3

ADVERBS = ["quickly","slowly","often","rarely","always","carefully","quietly","gently","firmly",
           "simply","usually","hardly","nearly","merely","lately"]  # 15

SPECIAL_TOKENS = {"<pad>": 0, "<eos>": 1}

def build_vocab():
    tokens = set()
    for lst in [NOUNS_SG, NOUNS_PL, VERBS_SG, VERBS_PL,
                DETERMINERS_SG, DETERMINERS_PL,
                LICENSORS, NPIS, REFLEXIVES_SG, REFLEXIVES_PL,
                PREPOSITIONS, ADVERBS]:
        tokens.update(lst)
    vocab = {t: i + 2 for i, t in enumerate(sorted(tokens))}
    vocab.update(SPECIAL_TOKENS)
    assert len(vocab) == 135, f"Expected 135, got {len(vocab)}"
    return vocab
