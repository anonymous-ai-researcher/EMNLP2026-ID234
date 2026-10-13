"""Alignment as a predictor of failure (Section 5.1; Appendix F.4, Table 36).

Items: 1,200 held-out items per phenomenon from 40 templates (30 each); errors are
Pythia-1.4B's under sentence log-probability.  An item's alignment score is the
fraction of 32 interventions in which the item, used as the source, changes the
output of a correctly handled base as the SCM predicts (32 distinct bases, 16 per
target label, drawn once from correctly handled development items with seed
4312026, keyed by phenomenon and item).

Predictors are L2 logistic regressions (C = 1, L-BFGS, 2,000 iterations, tol 1e-8)
fitted with 5-fold cross-validation grouped by template:
  B1:          length in Pythia tokens, structural load, rarity and mean frequency of
               the content-word lemmas (add-one-smoothed relative frequencies in the
               phenomenon's CFG training corpus; rarity = mean -log p(w), min-max
               scaled to [0, 1] over training sentences), standardized within each
               training fold;
  +probe:      1 - probe confidence;  +align: 1 - alignment score.
Probe-only: an unpenalized logistic model of 1 - probe confidence (BFGS).
AUROC (ties count 0.5) and non-interpolated average precision use out-of-fold
probabilities; intervals: 10,000 template-level bootstrap replicates of the fixed
out-of-fold predictions.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .stats import SEEDS, percentile_interval

BASE_SEED = 4312026


def lexical_tables(corpus: Sequence[Sequence[str]], content: set):
    counts = Counter(w for s in corpus for w in s if w in content)
    N, V = sum(counts.values()), len(content)
    p = lambda w: (counts.get(w, 0) + 1) / (N + V)
    R = [np.mean([-math.log(p(w)) for w in s if w in content]) for s in corpus if any(w in content for w in s)]
    return p, min(R), max(R)


def b1_features(words: Sequence[str], n_tokens: int, load: int, p, rmin: float, rmax: float, content: set):
    cw = [w for w in words if w in content]
    R = np.mean([-math.log(p(w)) for w in cw]) if cw else rmin
    rarity = float(np.clip((R - rmin) / (rmax - rmin), 0, 1)) if rmax > rmin else 0.0
    freq = float(np.mean([p(w) for w in cw])) if cw else 0.0
    return [n_tokens, load, rarity, freq]


def oof_logistic(X: np.ndarray, y: np.ndarray, groups: Sequence, n_splits: int = 5) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, tol=1e-8)
        clf.fit((X[tr] - mu) / sd, y[tr])
        oof[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    return oof


def oof_unpenalized_1d(x: np.ndarray, y: np.ndarray, groups: Sequence, n_splits: int = 5) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(x, y, groups):
        def nll(b):
            z = b[0] + b[1] * x[tr]
            return np.sum(np.logaddexp(0, z) - y[tr] * z)
        b = minimize(nll, np.zeros(2), method="BFGS", options={"gtol": 1e-8}).x
        oof[te] = 1 / (1 + np.exp(-(b[0] + b[1] * x[te])))
    return oof


def template_bootstrap(y, p_a, p_b, templates, metric=roc_auc_score, n_boot: int = 10_000,
                       seed: int = SEEDS["failure_auc"]):
    """Point difference metric(p_b) - metric(p_a) and its paired template-bootstrap interval."""
    y, p_a, p_b, templates = map(np.asarray, (y, p_a, p_b, templates))
    uniq = np.unique(templates); idx = {t: np.flatnonzero(templates == t) for t in uniq}
    rng = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[t] for t in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[sel])) < 2:
            continue
        boots.append(metric(y[sel], p_b[sel]) - metric(y[sel], p_a[sel]))
    return metric(y, p_b) - metric(y, p_a), percentile_interval(boots)


def tertile_gains(y, b1, plus, templates, seed: int = SEEDS["failure_extra"]) -> List:
    """Gain of a predictor over B1 within tertiles of the out-of-fold B1 scores."""
    y, b1, plus = map(np.asarray, (y, b1, plus))
    cuts = np.quantile(b1, [1 / 3, 2 / 3])
    out = []
    for lo, hi in ((-np.inf, cuts[0]), (cuts[0], cuts[1]), (cuts[1], np.inf)):
        m = (b1 > lo) & (b1 <= hi)
        out.append(template_bootstrap(y[m], b1[m], plus[m], np.asarray(templates)[m], seed=seed))
    return out


def auroc(y, p) -> float:
    return float(roc_auc_score(y, p))


def average_precision(y, p) -> float:
    return float(average_precision_score(y, p))
