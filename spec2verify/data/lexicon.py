"""Lexica (Appendix B.1, Table 5; Appendices D.1, D.8, D.10, F.2, F.8).

The vocabulary sizes are fixed by the paper and asserted in ``vocab.py``:

* 135 tokens for agreement, binding and concord (lexicon below plus padding
  and end-of-sentence);
* 137 for NPI (adds the relative pronoun *who* and the past participle
  *responded*; *that* is shared with the determiners);
* 167 for the determiner--noun separation suite (adds 32 modifiers);
* 139 for the three-input NPI construction (adds *and* and *once* to NPI);
* 40 for the Boolean tasks; 66 for German.

English text is lower-cased and uses word-level tokens; sentences end with the
end-of-sentence token instead of a period.
"""

from __future__ import annotations

from typing import List, Tuple

PAD, EOS = "<pad>", "<eos>"

# ----------------------------------------------------------------------------- English
#: 25 noun pairs.  The last five are the place and object nouns used inside
#: prepositional phrases; the first twenty are animate and serve as subjects.
NOUNS: List[Tuple[str, str]] = [
    ("bird", "birds"), ("dog", "dogs"), ("boy", "boys"), ("girl", "girls"), ("pilot", "pilots"),
    ("teacher", "teachers"), ("student", "students"), ("doctor", "doctors"), ("guard", "guards"),
    ("actor", "actors"), ("dancer", "dancers"), ("singer", "singers"), ("researcher", "researchers"),
    ("worker", "workers"), ("friend", "friends"), ("visitor", "visitors"), ("author", "authors"),
    ("artist", "artists"), ("farmer", "farmers"), ("nurse", "nurses"),
    ("lake", "lakes"), ("tree", "trees"), ("fence", "fences"), ("book", "books"), ("table", "tables"),
]
ANIMATE_NOUNS = NOUNS[:20]
PLACE_NOUNS = NOUNS[20:]

#: 20 verb pairs (singular, plural): 5 intransitive and 15 transitive.  The
#: transitive pairs are the relative-clause verbs of the NPI templates.
INTRANSITIVE: List[Tuple[str, str]] = [
    ("sings", "sing"), ("runs", "run"), ("laughs", "laugh"), ("barks", "bark"), ("is", "are"),
]
TRANSITIVE: List[Tuple[str, str]] = [
    ("admires", "admire"), ("helps", "help"), ("knows", "know"), ("likes", "like"),
    ("meets", "meet"), ("sees", "see"), ("trusts", "trust"), ("praises", "praise"),
    ("follows", "follow"), ("visits", "visit"), ("thanks", "thank"), ("calls", "call"),
    ("greets", "greet"), ("watches", "watch"), ("hates", "hate"),
]
VERBS = INTRANSITIVE + TRANSITIVE

#: Past-tense transitive verbs of the binding template (no number marking).
BINDING_VERBS: List[str] = ["admired", "watched"]

#: 4 number-marked determiner pairs and the number-neutral *the*.
DETERMINERS: List[Tuple[str, str]] = [("this", "these"), ("that", "those"), ("a", "some"), ("one", "two")]
NEUTRAL_DET = "the"

PREPOSITIONS = ["near", "behind", "beside", "by", "under", "above", "across", "to"]
#: Prepositions used in the PP slot (all eight are locative except *to*).
PP_PREPOSITIONS = PREPOSITIONS[:7]
LICENSORS = ["no", "neither", "never"]
#: Licensors that can occupy the determiner slot of the NPI templates.
NP_LICENSORS = {"no": ("sg", "pl"), "neither": ("sg",)}
NPIS = ["ever", "any", "anyone"]
REFLEXIVES: List[Tuple[str, str]] = [
    ("himself", "themselves"), ("herself", "themselves"), ("itself", "themselves"), ("oneself", "themselves"),
]
#: Reflexives that the binding templates use with animate subjects.
BINDING_REFLEXIVES: List[Tuple[str, str]] = REFLEXIVES[:2]
AUXILIARIES = ["has", "have", "had"]
ADVERBS = ["loudly", "quietly", "quickly", "slowly", "often", "rarely", "again", "today", "here", "now"]

# NPI-only words.  *that* is read as a relative pronoun only in NPI data.
RELATIVE_PRONOUNS = ["who", "that"]
NPI_ONLY = ["who", "responded"]
PAST_PARTICIPLE = "responded"

#: Three-input NPI construction (Appendix D.1): two clauses joined by *and*.
THREE_INPUT_EXTRA = ["and", "once"]

#: English decisive constructions (Appendix F.2): fronted clause words.
DECISIVE_EXTRA = [",", "when"]

#: Separation suite (Appendix D.8): eight ordered slots of four modifiers.
MODIFIER_SLOTS: List[List[str]] = [
    ["beautiful", "useful", "elegant", "remarkable"],   # opinion
    ["small", "large", "tiny", "massive"],              # size
    ["old", "new", "ancient", "modern"],                # age
    ["round", "square", "narrow", "flat"],              # shape
    ["red", "blue", "green", "black"],                  # color
    ["french", "italian", "chinese", "german"],         # origin
    ["wooden", "metal", "plastic", "stone"],            # material
    ["reference", "school", "travel", "research"],      # purpose
]

#: Lexical shift (Appendix F.8): content words absent from the alignment data.
SHIFT_NOUNS: List[Tuple[str, str]] = [
    ("sailor", "sailors"), ("baker", "bakers"), ("lawyer", "lawyers"), ("musician", "musicians"),
    ("engineer", "engineers"), ("painter", "painters"), ("driver", "drivers"), ("poet", "poets"),
    ("scientist", "scientists"), ("soldier", "soldiers"),
]
SHIFT_VERBS: List[Tuple[str, str]] = [
    ("greets", "greet"), ("assists", "assist"), ("inspects", "inspect"), ("notices", "notice"),
    ("respects", "respect"),
]
#: Template shift (Appendix F.8): ten held-out outer frames.
SHIFT_FRAMES_BEFORE = ["yesterday", "today", "usually", "apparently", "perhaps"]
SHIFT_FRAMES_AFTER = ["today", "yesterday", "outside", "nearby", "again"]


def unique(words: List[str]) -> List[str]:
    out: List[str] = []
    for w in words:
        if w not in out:
            out.append(w)
    return out


def english_core_words() -> List[str]:
    words: List[str] = []
    for pairs in (NOUNS, VERBS, DETERMINERS):
        for sg, pl in pairs:
            words += [sg, pl]
    words += [NEUTRAL_DET] + PREPOSITIONS + LICENSORS + NPIS
    for sg, pl in REFLEXIVES:
        words += [sg, pl]
    words += AUXILIARIES + ADVERBS + BINDING_VERBS
    return unique(words)


# ----------------------------------------------------------------------------- German (F.2)
#: (singular, plural, dative plural, gender)
GERMAN_NOUNS: List[Tuple[str, str, str, str]] = [
    ("Mann", "Männer", "Männern", "masc"), ("Junge", "Jungen", "Jungen", "masc"),
    ("Lehrer", "Lehrer", "Lehrern", "masc"), ("Forscher", "Forscher", "Forschern", "masc"),
    ("Arzt", "Ärzte", "Ärzten", "masc"),
    ("Frau", "Frauen", "Frauen", "fem"), ("Lehrerin", "Lehrerinnen", "Lehrerinnen", "fem"),
    ("Forscherin", "Forscherinnen", "Forscherinnen", "fem"), ("Ärztin", "Ärztinnen", "Ärztinnen", "fem"),
    ("Kind", "Kinder", "Kindern", "neut"), ("Mädchen", "Mädchen", "Mädchen", "neut"),
    ("Mitglied", "Mitglieder", "Mitgliedern", "neut"),
]
#: Weak masculine noun: non-nominative singular form.
GERMAN_OBLIQUE_SG = {"Junge": "Jungen"}
GERMAN_ACC_VERBS = [("sieht", "sehen"), ("besucht", "besuchen"), ("kennt", "kennen"),
                    ("trifft", "treffen"), ("beobachtet", "beobachten"), ("lobt", "loben")]
GERMAN_DAT_VERBS = [("hilft", "helfen"), ("dankt", "danken"), ("folgt", "folgen"),
                    ("vertraut", "vertrauen"), ("gefällt", "gefallen")]
GERMAN_DETERMINERS = ["der", "die", "das", "den", "dem", "des"]
GERMAN_DAT_PREPOSITIONS = ["mit", "bei", "von", "zu"]
GERMAN_FRAME = ["Ich", "glaube", ",", "dass"]
GERMAN_PERIOD = "."


def german_words() -> List[str]:
    words = list(GERMAN_FRAME) + [GERMAN_PERIOD] + GERMAN_DETERMINERS + GERMAN_DAT_PREPOSITIONS
    for sg, pl, dpl, _ in GERMAN_NOUNS:
        words += [sg, pl, dpl]
    for sg, pl in GERMAN_ACC_VERBS + GERMAN_DAT_VERBS:
        words += [sg, pl]
    return unique(words)


def german_article(case: str, gender: str, plural: bool) -> str:
    """Definite article by case, gender and number (nominative/accusative/dative)."""
    if plural:
        return {"nom": "die", "acc": "die", "dat": "den"}[case]
    table = {
        "masc": {"nom": "der", "acc": "den", "dat": "dem"},
        "fem": {"nom": "die", "acc": "die", "dat": "der"},
        "neut": {"nom": "das", "acc": "das", "dat": "dem"},
    }
    return table[gender][case]


def german_noun_form(noun: Tuple[str, str, str, str], case: str, plural: bool) -> str:
    sg, pl, dpl, _ = noun
    if plural:
        return dpl if case == "dat" else pl
    if case != "nom" and sg in GERMAN_OBLIQUE_SG:
        return GERMAN_OBLIQUE_SG[sg]
    return sg


# ----------------------------------------------------------------------------- Boolean tasks
BOOL_SPECIAL = ["<bos>", "<read>", "<eos>"]
BOOL_INPUTS = ["X0", "X1", "Y0", "Y1"]
BOOL_DISTRACTORS = [f"F{i:02d}" for i in range(32)]
