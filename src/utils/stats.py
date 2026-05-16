"""Statistical testing utilities (Appendix C.7)."""
import numpy as np
from scipy import stats

def paired_t_test(x, y, alpha=0.05):
    t_stat, p_val = stats.ttest_rel(x, y)
    return {"t": t_stat, "p": p_val, "significant": p_val < alpha}

def bootstrap_test(x, y, n_resamples=10000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    diffs = x - y
    observed = np.mean(diffs)
    boot_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_resamples)]
    p_val = np.mean([b >= 0 for b in boot_means]) if observed < 0 else np.mean([b <= 0 for b in boot_means])
    p_val = 2 * min(p_val, 1 - p_val)
    return {"observed_diff": observed, "p": p_val, "significant": p_val < alpha}

def bonferroni_correct(p_values, alpha=0.05):
    n = len(p_values)
    adjusted = [min(p * n, 1.0) for p in p_values]
    return adjusted, [p < alpha for p in adjusted]
