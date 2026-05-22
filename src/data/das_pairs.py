"""DAS counterfactual pair construction (Appendix B.3)."""
import numpy as np
from typing import List, Tuple, Dict
def construct_das_pairs(data, variable, n_pairs=10000, seed=42):
    rng = np.random.RandomState(seed)
    val0 = [d for d in data if d.get(variable) == 0]
    val1 = [d for d in data if d.get(variable) == 1]
    return [(rng.choice(val0), rng.choice(val1)) for _ in range(n_pairs)]
def filter_pairs(pairs, max_token_diff=3):
    return [(b,s) for b,s in pairs
            if sum(1 for a,c in zip(b["tokens"].split(), s["tokens"].split()) if a!=c) <= max_token_diff]
