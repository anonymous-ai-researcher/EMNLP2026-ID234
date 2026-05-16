"""Structural Causal Model definitions (Section 2.2, Appendix A.3)."""
import torch

class SCM:
    def __init__(self, phenomenon, variables, comp_func):
        self.phenomenon = phenomenon
        self.variables = variables
        self.comp_func = comp_func

class AgreementSCM(SCM):
    def __init__(self):
        super().__init__("agreement", ("SUBJ_NUM", "VERB_FORM"), "biconditional")
    def compute_atomic(self, sent): return int(sent["subject_is_plural"])
    def compute_comp(self, atomic, feat): return int(atomic == feat)

class NPISCM(SCM):
    def __init__(self):
        super().__init__("npi", ("HAS_LIC", "NPI_OK"), "material_implication")
    def compute_atomic(self, sent): return int(sent["has_licensor"])
    def compute_comp(self, atomic, feat): return int(atomic or not feat)

class BindingSCM(SCM):
    def __init__(self):
        super().__init__("binding", ("ANTE_NUM", "REFL_OK"), "biconditional")
    def compute_atomic(self, sent): return int(sent["antecedent_is_plural"])
    def compute_comp(self, atomic, feat): return int(atomic == feat)

class ConcordSCM(SCM):
    def __init__(self):
        super().__init__("concord", ("DET_NUM", "CONC_OK"), "biconditional")
    def compute_atomic(self, sent): return int(sent["determiner_is_plural"])
    def compute_comp(self, atomic, feat): return int(atomic == feat)

SCMS = {"agreement": AgreementSCM, "npi": NPISCM, "binding": BindingSCM, "concord": ConcordSCM}
def get_scm(phen): return SCMS[phen]()
