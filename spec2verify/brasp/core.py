"""Boolean RASP (B-RASP).

A program reads a string ``w = w_1 ... w_n`` and manipulates Boolean vectors
``P in {0,1}^n``, one entry per position.  The initial vectors ``Q_sigma`` mark
the positions holding symbol ``sigma``; every further vector is computed by
one of two operations.

* Pointwise (combine): ``P_t(i) := f(P_a(i), P_b(i), ...)``.
* Attention (lookup):  ``P_t(i) := <|_j [M(i,j), S(i,j)] V(i,j) : D(i)``.

Attention semantics (Appendix A.1 of the paper):

1. ``J_i = {j : M(i,j) = 1}`` is the set of positions visible from ``i``.
2. ``B_i = {j in J_i : S(i,j) = 1}`` is the set of visible positions with
   score 1.  If ``B_i`` is non-empty, ``<|`` selects ``j* = min(B_i)``
   (the variant ``|>`` selects ``max(B_i)``).
3. ``P_t(i) = V(i, j*)`` if ``B_i`` is non-empty, and ``D(i)`` otherwise.

Masks are strict (``j < i``, ``j > i``) or absent.  Vectors are stored as
``numpy`` Boolean arrays in an environment dictionary keyed by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np

Env = Dict[str, np.ndarray]
# A matrix-valued predicate receives the environment and two index grids
# (query positions ``qi`` and key positions ``kj``, both of shape (n, n)).
MatFn = Callable[[Env, np.ndarray, np.ndarray], np.ndarray]
# A vector-valued predicate receives the environment and returns shape (n,).
VecFn = Callable[[Env], np.ndarray]


class Direction(Enum):
    LEFTMOST = "leftmost"     # <|
    RIGHTMOST = "rightmost"   # |>


def length(env: Env) -> int:
    return len(next(v for v in env.values() if isinstance(v, np.ndarray)))


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def mask_left(env: Env, qi: np.ndarray, kj: np.ndarray) -> np.ndarray:
    """``M(i,j) = (j < i)``: strict future masking."""
    return kj < qi


def mask_right(env: Env, qi: np.ndarray, kj: np.ndarray) -> np.ndarray:
    """``M(i,j) = (j > i)``: strict past masking."""
    return kj > qi


def mask_none(env: Env, qi: np.ndarray, kj: np.ndarray) -> np.ndarray:
    return np.ones_like(qi, dtype=bool)


# ---------------------------------------------------------------------------
# Small helpers for building predicates
# ---------------------------------------------------------------------------

def key(name: str) -> MatFn:
    """Lift a vector to a (query, key) predicate that reads the key position."""
    return lambda env, qi, kj: env[name][kj]


def const_true() -> MatFn:
    return lambda env, qi, kj: np.ones_like(qi, dtype=bool)


def zero() -> VecFn:
    """``D(i) = 0``."""
    return lambda env: np.zeros(length(env), dtype=bool)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

@dataclass
class Pointwise:
    """``P_t(i) := f(...)`` evaluated independently at every position."""

    name: str
    fn: VecFn
    reads: List[str] = field(default_factory=list)
    kind: str = "pointwise"

    def __call__(self, env: Env) -> np.ndarray:
        return np.asarray(self.fn(env), dtype=bool)


@dataclass
class Attention:
    """``P_t(i) := <|_j [M(i,j), S(i,j)] V(i,j) : D(i)`` (or ``|>``)."""

    name: str
    mask: MatFn
    score: MatFn
    value: MatFn
    default: VecFn
    direction: Direction = Direction.LEFTMOST
    reads: List[str] = field(default_factory=list)
    kind: str = "attention"

    def select(self, env: Env) -> np.ndarray:
        """Return ``j*`` for every query position (``-1`` when ``B_i`` is empty)."""
        n = length(env)
        qi, kj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        visible = np.asarray(self.mask(env, qi, kj), dtype=bool)
        scored = np.asarray(self.score(env, qi, kj), dtype=bool) & visible  # B_i
        out = np.full(n, -1, dtype=int)
        for i in range(n):
            cand = np.flatnonzero(scored[i])
            if cand.size:
                out[i] = cand[0] if self.direction is Direction.LEFTMOST else cand[-1]
        return out

    def __call__(self, env: Env) -> np.ndarray:
        n = length(env)
        qi, kj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        v = np.asarray(self.value(env, qi, kj), dtype=bool)
        d = np.asarray(self.default(env), dtype=bool)
        sel = self.select(env)
        out = d.copy()
        has = sel >= 0
        out[has] = v[np.arange(n)[has], sel[has]]
        return out


Operation = Pointwise | Attention


@dataclass
class Program:
    """An ordered list of operations; ``output`` names the read-out vector."""

    name: str
    operations: List[Operation]
    output: str
    atomic: Optional[str] = None          # the lookup variable studied
    compositional: Optional[str] = None   # the combine variable studied

    @property
    def length(self) -> int:
        return len(self.operations)

    def attention_depth(self) -> int:
        """Largest number of attention operations along a dependency chain.

        A multi-head masked hard-attention Transformer with this many layers
        simulates the program; it is at most ``length``.
        """
        depth: Dict[str, int] = {}
        for op in self.operations:
            base = max([depth.get(r, 0) for r in op.reads] + [0])
            depth[op.name] = base + (1 if op.kind == "attention" else 0)
        return max(depth.values()) if depth else 0

    def run(self, env: Env, overrides: Optional[Dict[str, np.ndarray]] = None) -> Env:
        """Execute the program; ``overrides`` fixes chosen vectors (interventions).

        An override replaces the computed vector *after* it is computed, so
        every downstream operation reads the intervened value.  Overrides may
        set a vector at chosen positions only: pass a masked array whose
        unmasked entries are written.
        """
        env = dict(env)
        overrides = overrides or {}
        for op in self.operations:
            val = op(env)
            if op.name in overrides:
                ov = overrides[op.name]
                if isinstance(ov, np.ma.MaskedArray):
                    val = val.copy()
                    keep = ~np.ma.getmaskarray(ov)
                    val[keep] = np.asarray(ov.data, dtype=bool)[keep]
                else:
                    val = np.asarray(ov, dtype=bool)
            env[op.name] = val
        return env
