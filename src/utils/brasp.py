"""B-RASP program definitions for four syntactic phenomena (Section 3, Appendix A.2)."""
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class AttentionOp:
    name: str; mask: str; score: str; value: str; default: str

@dataclass
class PointwiseOp:
    name: str; inputs: List[str]; func: str

@dataclass
class BRASPProgram:
    phenomenon: str; atomic_var: str; comp_var: str
    operations: List; output_position: str; boolean_function: str

AGREEMENT = BRASPProgram("agreement", "SUBJ_NUM", "VERB_FORM", [
    AttentionOp("SUBJ_NUM", "j < i", "noun(j)", "plural(j)", "0"),
    PointwiseOp("VERB_FORM", ["SUBJ_NUM", "v_pl"], "biconditional"),
], "verb position", "biconditional")

NPI = BRASPProgram("npi", "HAS_LIC", "NPI_OK", [
    AttentionOp("HAS_LIC", "j < i", "licensor(j)", "1", "0"),
    PointwiseOp("NPI_OK", ["HAS_LIC", "npi_present"], "material_implication"),
], "NPI position", "material_implication")

BINDING = BRASPProgram("binding", "ANTE_NUM", "REFL_OK", [
    AttentionOp("ANTE_NUM", "j < i AND same_clause(j,i)", "noun(j)", "plural(j)", "0"),
    PointwiseOp("REFL_OK", ["ANTE_NUM", "refl_pl"], "biconditional"),
], "reflexive position", "biconditional")

CONCORD = BRASPProgram("concord", "DET_NUM", "CONC_OK", [
    AttentionOp("DET_NUM", "j < i", "determiner(j)", "plural(j)", "0"),
    PointwiseOp("CONC_OK", ["DET_NUM", "noun_pl"], "biconditional"),
], "noun position", "biconditional")

ALL_PROGRAMS = {"agreement": AGREEMENT, "npi": NPI, "binding": BINDING, "concord": CONCORD}
def get_program(phen): return ALL_PROGRAMS[phen]
def get_variable_names(phen):
    p = get_program(phen); return p.atomic_var, p.comp_var
