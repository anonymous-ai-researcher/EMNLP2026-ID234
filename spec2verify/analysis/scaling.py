"""Parameter requirement versus dependency distance (Section 5.5; Appendix F.7, Table 41).

P*(d) is the smallest tested parameter count whose unrounded mean test accuracy
across five seeds is at least 0.90 (right-censored when no width reaches it).
log P*(d) is fitted on d = 2..10 (n = 9, the distances at which neither
sequential architecture is censored) against d (exponential) and against log d
(power law), compared by AIC within one architecture; the per-phrase growth
factor is exp(slope).  Uncertainty: 10,000 resamples of the five complete seed
grids (seed 7112026; a resampled seed keeps all its distances and widths),
recomputing the thresholds and the fit; percentile intervals.  Sensitivity:
interpolating accuracy linearly in log parameter count between the last
failing and the first passing width; a quadratic in d; and an exact enumeration
of the 126 multisets of five seeds weighted by their multinomial probability.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .stats import SEEDS, percentile_interval

FIT_DISTANCES = tuple(range(2, 11))


def p_star(acc: np.ndarray, params: Sequence[int], threshold: float = 0.90, interpolate: bool = False
           ) -> Optional[float]:
    """acc: mean accuracy per width (ordered by parameter count)."""
    params = np.asarray(params, float)
    for k, a in enumerate(acc):
        if a >= threshold:
            if not interpolate or k == 0:
                return float(params[k])
            a0, a1 = acc[k - 1], a
            l0, l1 = math.log(params[k - 1]), math.log(params[k])
            return float(math.exp(l0 + (threshold - a0) / (a1 - a0) * (l1 - l0)))
    return None


def fit(ds: Sequence[int], pstar: Sequence[float], form: str = "exponential") -> Dict[str, float]:
    y = np.log(np.asarray(pstar, float)); d = np.asarray(ds, float)
    X = {"exponential": np.c_[np.ones_like(d), d], "power": np.c_[np.ones_like(d), np.log(d)],
         "quadratic": np.c_[np.ones_like(d), d, d ** 2]}[form]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(((y - X @ beta) ** 2).sum()); n, k = len(y), X.shape[1]
    aic = n * math.log(rss / n) + 2 * k
    return {"beta": beta.tolist(), "aic": aic, "growth": float(math.exp(beta[1])) if form != "power" else float("nan")}


def grid_pstar(grid: np.ndarray, params: Sequence[int], ds: Sequence[int], interpolate: bool = False):
    """grid: (seeds, distances, widths) accuracies."""
    mean = grid.mean(0)
    return [p_star(mean[i], params, interpolate=interpolate) for i in range(len(ds))]


def growth_with_uncertainty(grid: np.ndarray, params: Sequence[int], ds: Sequence[int], n_boot: int = 10_000,
                            seed: int = SEEDS["scaling_bootstrap"], interpolate: bool = False) -> Dict:
    idx = [list(ds).index(d) for d in FIT_DISTANCES]
    ps = grid_pstar(grid, params, ds, interpolate)
    point = fit(FIT_DISTANCES, [ps[i] for i in idx])
    rng = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        g = grid[rng.integers(0, grid.shape[0], grid.shape[0])]
        p = grid_pstar(g, params, ds, interpolate)
        if all(p[i] is not None for i in idx):
            boots.append(fit(FIT_DISTANCES, [p[i] for i in idx])["growth"])
    power = fit(FIT_DISTANCES, [ps[i] for i in idx], "power")
    quad = fit(FIT_DISTANCES, [ps[i] for i in idx], "quadratic")
    return {"growth": point["growth"], "ci": percentile_interval(boots),
            "aic_power_minus_exponential": power["aic"] - point["aic"],
            "aic_exponential_minus_quadratic": point["aic"] - quad["aic"], "pstar": ps}


def exact_enumeration(grid: np.ndarray, params, ds) -> Tuple[float, float]:
    """Percentile interval from the exact distribution over the 126 seed multisets."""
    n = grid.shape[0]; idx = [list(ds).index(d) for d in FIT_DISTANCES]
    vals, weights = [], []
    for counts in itertools.product(range(n + 1), repeat=n):
        if sum(counts) != n:
            continue
        g = np.concatenate([np.repeat(grid[i:i + 1], c, 0) for i, c in enumerate(counts) if c])
        p = grid_pstar(g, params, ds)
        if all(p[i] is not None for i in idx):
            w = math.factorial(n) / np.prod([math.factorial(c) for c in counts]) / n ** n
            vals.append(fit(FIT_DISTANCES, [p[i] for i in idx])["growth"]); weights.append(w)
    order = np.argsort(vals); v = np.asarray(vals)[order]; w = np.cumsum(np.asarray(weights)[order])
    w /= w[-1]
    return float(np.interp(0.025, w, v)), float(np.interp(0.975, w, v))
