"""Statistical tests (Appendix C.7) and resampling.

All tests are two-sided at alpha = 0.05 unless noted.  Independent units are
training seeds for from-scratch models and DAS initializations for fixed
pre-trained checkpoints (three DAS runs per seed are averaged within the seed
first).  Percentile intervals interpolate linearly between order statistics
(``numpy.quantile(method="linear")``).

Fixed resampling seeds: 62026 (template/item bootstrap of Table 12), 6022026
(paradigm bootstrap of Table 30), 17012026 (SAE permutation null), 4112026 and
9302026 (failure prediction), 7112026 (seed-grid bootstrap of Appendix F.7).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

SEEDS = {"atomcomp_bootstrap": 62026, "npi_paradigm_bootstrap": 6022026, "sae_permutation": 17012026,
         "failure_auc": 4112026, "failure_extra": 9302026, "scaling_bootstrap": 7112026}


def one_sample_t(x: Sequence[float], mu: float = 0.5) -> Tuple[float, float]:
    r = stats.ttest_1samp(np.asarray(x, float), mu)
    return float(r.statistic), float(r.pvalue)


def paired_t(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """(mean difference, t, p) with df = n - 1."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = stats.ttest_rel(a, b)
    return float((a - b).mean()), float(r.statistic), float(r.pvalue)


def t_interval(x: Sequence[float], level: float = 0.95) -> Tuple[float, float]:
    x = np.asarray(x, float); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * se
    return float(m - h), float(m + h)


def cohens_d(diff: Sequence[float]) -> float:
    d = np.asarray(diff, float)
    return float(d.mean() / d.std(ddof=1))


def exact_sign_test(diff: Sequence[float]) -> Tuple[int, int, float]:
    d = np.asarray(diff, float); d = d[d != 0]
    k = int((d > 0).sum()); n = len(d)
    return k, n, float(stats.binomtest(k, n, 0.5).pvalue)


def wilcoxon(diff: Sequence[float]) -> Tuple[float, float]:
    """Signed-rank statistic W+ and its normal-approximation p-value (ties corrected)."""
    r = stats.wilcoxon(np.asarray(diff, float), method="approx", correction=False)
    d = np.asarray(diff, float)
    ranks = stats.rankdata(np.abs(d[d != 0]))
    w_plus = float(ranks[d[d != 0] > 0].sum())
    return w_plus, float(r.pvalue)


def benjamini_hochberg(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q


def bonferroni(p: Sequence[float]) -> np.ndarray:
    return np.minimum(np.asarray(p, float) * len(p), 1.0)


def one_way_anova(groups: Sequence[Sequence[float]]) -> Tuple[float, float]:
    r = stats.f_oneway(*[np.asarray(g, float) for g in groups])
    return float(r.statistic), float(r.pvalue)


def tost(x: Sequence[float], margin: float) -> float:
    """Equivalence of a mean change to zero within +/- margin (two one-sided t-tests)."""
    x = np.asarray(x, float); n = len(x); se = x.std(ddof=1) / np.sqrt(n)
    p_low = 1 - stats.t.cdf((x.mean() + margin) / se, n - 1)
    p_high = stats.t.cdf((x.mean() - margin) / se, n - 1)
    return float(max(p_low, p_high))


def one_sided_t_greater(x: Sequence[float], mu: float = 0.0) -> float:
    return float(stats.ttest_1samp(np.asarray(x, float), mu, alternative="greater").pvalue)


def percentile_interval(samples: Sequence[float], level: float = 0.95) -> Tuple[float, float]:
    s = np.asarray(samples, float)
    return (float(np.quantile(s, 0.5 - level / 2, method="linear")),
            float(np.quantile(s, 0.5 + level / 2, method="linear")))


def paired_cluster_bootstrap(diff_by_cluster: Dict, n_boot: int = 10_000, seed: int = SEEDS["atomcomp_bootstrap"]
                             ) -> Tuple[float, float]:
    """Paired bootstrap over clusters (templates or items) of per-cluster mean differences."""
    rng = np.random.default_rng(seed)
    keys = list(diff_by_cluster); vals = np.array([np.mean(diff_by_cluster[k]) for k in keys])
    boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)]
    return percentile_interval(boots)


def group_difference_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000,
                               seed: int = SEEDS["npi_paradigm_bootstrap"]) -> float:
    """One-sided bootstrap p-value that mean(a) < mean(b) (paradigms resampled within group)."""
    rng = np.random.default_rng(seed); a = np.asarray(a, float); b = np.asarray(b, float)
    d = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
         for _ in range(n_boot)]
    return float((np.sum(np.asarray(d) >= 0) + 1) / (n_boot + 1))


def variance_decomposition(iia: np.ndarray) -> float:
    """Within-seed DAS variance as a share of between-seed model variance for a
    (seeds x DAS runs) array (Appendix C.3)."""
    within = iia.var(axis=1, ddof=1).mean()
    between = iia.mean(axis=1).var(ddof=1)
    return float(within / between) if between > 0 else float("inf")
