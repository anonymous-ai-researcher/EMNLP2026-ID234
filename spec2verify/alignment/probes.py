"""Linear probes (Appendix E.5 and the probe baseline of Appendix F.4).

L2 logistic regression, C = 1, with intercept, L-BFGS, at most 2,000
iterations, tolerance 1e-8, no class weighting, threshold 0.5.  Features are
standardized with the probe-training mean and standard deviation (zero-variance
dimensions stay 0).  The probe reads the DAS site and position of the variable;
each sentence counts once, and probe-training and test sentences are disjoint.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


class Standardizer:
    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mu = X.mean(0); sd = X.std(0)
        self.sd = np.where(sd > 0, sd, 1.0); self.zero = sd == 0
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        Z[:, self.zero] = 0.0
        return Z


def fit_probe(X: np.ndarray, y: Sequence[int]) -> Tuple[LogisticRegression, Standardizer]:
    st = Standardizer().fit(X)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, tol=1e-8, fit_intercept=True)
    clf.fit(st(X), np.asarray(y))
    return clf, st


def probe_accuracy(clf, st, X: np.ndarray, y: Sequence[int]) -> float:
    return float(np.mean((clf.predict_proba(st(X))[:, 1] > 0.5).astype(int) == np.asarray(y)))


def confidence_true(clf, st, X: np.ndarray, y: Sequence[int]) -> np.ndarray:
    """Probability assigned to each item's true value (the probe score of Appendix F.4)."""
    p = clf.predict_proba(st(X))
    return p[np.arange(len(y)), np.asarray(y, dtype=int)]
