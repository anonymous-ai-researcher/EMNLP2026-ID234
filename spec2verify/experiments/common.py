"""Shared helpers for the experiment runners: result files, run directories, scaling."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

RESULTS = os.environ.get("SPEC2VERIFY_RESULTS", "results")
RUNS = os.environ.get("SPEC2VERIFY_RUNS", "runs")


@dataclass
class Scale:
    """Multiplies dataset sizes, training steps and DAS epochs.  1.0 reproduces the paper's
    protocol; small values give quick end-to-end smoke runs."""
    data: float = 1.0
    steps: float = 1.0
    epochs: float = 1.0

    def n(self, x: int) -> int:
        return max(8, int(round(x * self.data)))

    def s(self, x: int) -> int:
        return max(10, int(round(x * self.steps)))

    def e(self, x: int) -> int:
        return max(1, int(round(x * self.epochs)))


FULL = Scale()
QUICK = Scale(data=0.02, steps=0.01, epochs=0.03)


def _default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def save(name: str, obj: Dict) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name + ".json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_default)
    return path


def load(name: str) -> Dict:
    with open(os.path.join(RESULTS, name + ".json")) as f:
        return json.load(f)


def run_dir(*parts) -> str:
    p = os.path.join(RUNS, *[str(x) for x in parts])
    os.makedirs(p, exist_ok=True)
    return p
