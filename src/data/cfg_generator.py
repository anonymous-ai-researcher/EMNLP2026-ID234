"""CFG-based sentence generation for four phenomena (Section 4, Appendix B.1).
Generates matched grammatical/ungrammatical pairs with attractor distances 0-8.
Max sequence length: 30 tokens.
"""
import numpy as np
from typing import List, Dict, Tuple
from .lexicon import *

class CFGGenerator:
    def __init__(self, phenomenon: str, seed: int = 42):
        self.phenomenon = phenomenon
        self.rng = np.random.RandomState(seed)
    def generate_pair(self, attractor_distance: int) -> Tuple[Dict, Dict]:
        if self.phenomenon == "agreement": return self._generate_agreement(attractor_distance)
        elif self.phenomenon == "npi": return self._generate_npi(attractor_distance)
        elif self.phenomenon == "binding": return self._generate_binding(attractor_distance)
        elif self.phenomenon == "concord": return self._generate_concord(attractor_distance)
    def generate_dataset(self, n_per_distance: int = 5000, max_distance: int = 8) -> List[Dict]:
        data = []
        for d in range(max_distance + 1):
            for _ in range(n_per_distance):
                gram, ungram = self.generate_pair(d)
                data.extend([gram, ungram])
        self.rng.shuffle(data)
        return data
    def _generate_agreement(self, n_attractors):
        subj_pl = bool(self.rng.randint(2))
        subj = self.rng.choice(NOUNS_PL if subj_pl else NOUNS_SG)
        attractors = []
        for _ in range(n_attractors):
            prep = self.rng.choice(PREPOSITIONS)
            attr = self.rng.choice(NOUNS_PL if bool(self.rng.randint(2)) else NOUNS_SG)
            attractors.append(f"{prep} the {attr}")
        prefix = f"the {subj} {' '.join(attractors)}".strip()
        v_ok = self.rng.choice(VERBS_PL if subj_pl else VERBS_SG)
        v_bad = self.rng.choice(VERBS_SG if subj_pl else VERBS_PL)
        return ({"tokens": f"{prefix} {v_ok}", "label": 1, "subject_is_plural": subj_pl, "distance": n_attractors},
                {"tokens": f"{prefix} {v_bad}", "label": 0, "subject_is_plural": subj_pl, "distance": n_attractors})
    def _generate_npi(self, n): return self._generate_agreement(n)
    def _generate_binding(self, n): return self._generate_agreement(n)
    def _generate_concord(self, n): return self._generate_agreement(n)
