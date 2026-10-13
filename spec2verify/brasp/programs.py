"""The B-RASP programs of the paper (Section 3 and Appendix A.2).

Core programs, one per phenomenon, each with an atomic variable (one
attention operation, a lookup) and a compositional variable (a pointwise
Boolean function of the atomic variable and one input feature):

    (A.1)  subj_num(i)  := <|_j[j<i, Q_noun(j)] Q_n_pl(j) : 0
    (A.2)  verb_form(i) := subj_num(i) <-> Q_v_pl(i)
    (A.3)  has_lic(i)   := <|_j[j<i, Q_lic(j) or Q_rel(j)] Q_lic(j) : 0
    (A.4)  npi_ok(i)    := has_lic(i) or not Q_npi(i)
    (A.5)  ante_num(i)  := <|_j[j<i, Q_noun(j)] Q_n_pl(j) : 0
    (A.6)  refl_ok(i)   := ante_num(i) <-> Q_refl_pl(i)
    (A.7)  det_num(i)   := <|_j[j<i, Q_det(j)] Q_det_pl(j) : 0
    (A.8)  conc_ok(i)   := det_num(i) <-> Q_n_pl(i)

Decisive constructions (Appendix A.2, "Programs for Decisive Constructions"):
the English hierarchical program H (A.9-A.11) and the German case-first
program KO (A.12-A.14), with their competitors L, N and S.

Also defined here: the within-phenomenon alternative SCMs (Appendix D.4),
the three-input NPI program (Appendix D.1) and the two-input Boolean
functions of the coverage study (Appendix D.10, F.5).
"""

from __future__ import annotations

import hashlib
from typing import Callable, Dict, Tuple

import numpy as np

from .core import (Attention, Direction, Env, Pointwise, Program, const_true, key,
                   length, mask_left, mask_none, mask_right, zero)

L, R = Direction.LEFTMOST, Direction.RIGHTMOST


def _iff(a: str, b: str) -> Callable[[Env], np.ndarray]:
    return lambda env: env[a] == env[b]


def _implies_npi(env: Env) -> np.ndarray:
    return env["has_lic"] | ~env["Q_npi"]


# ---------------------------------------------------------------------------
# Core programs (A.1-A.8)
# ---------------------------------------------------------------------------

def agreement(atomic_name: str = "subj_num") -> Program:
    return Program(
        name="agreement",
        operations=[
            Attention(atomic_name, mask_left, key("Q_noun"), key("Q_n_pl"), zero(), L,
                      reads=["Q_noun", "Q_n_pl"]),
            Pointwise("verb_form", _iff(atomic_name, "Q_v_pl"), reads=[atomic_name, "Q_v_pl"]),
        ],
        output="verb_form", atomic=atomic_name, compositional="verb_form")


def npi() -> Program:
    return Program(
        name="npi",
        operations=[
            Attention("has_lic", mask_left,
                      lambda env, qi, kj: env["Q_lic"][kj] | env["Q_rel"][kj],
                      key("Q_lic"), zero(), L, reads=["Q_lic", "Q_rel"]),
            Pointwise("npi_ok", _implies_npi, reads=["has_lic", "Q_npi"]),
        ],
        output="npi_ok", atomic="has_lic", compositional="npi_ok")


def binding() -> Program:
    return Program(
        name="binding",
        operations=[
            Attention("ante_num", mask_left, key("Q_noun"), key("Q_n_pl"), zero(), L,
                      reads=["Q_noun", "Q_n_pl"]),
            Pointwise("refl_ok", _iff("ante_num", "Q_refl_pl"), reads=["ante_num", "Q_refl_pl"]),
        ],
        output="refl_ok", atomic="ante_num", compositional="refl_ok")


def concord() -> Program:
    return Program(
        name="concord",
        operations=[
            Attention("det_num", mask_left, key("Q_det"), key("Q_det_pl"), zero(), L,
                      reads=["Q_det", "Q_det_pl"]),
            Pointwise("conc_ok", _iff("det_num", "Q_n_pl"), reads=["det_num", "Q_n_pl"]),
        ],
        output="conc_ok", atomic="det_num", compositional="conc_ok")


CORE = {"agreement": agreement, "npi": npi, "binding": binding, "concord": concord}
VARIABLES: Dict[str, Tuple[str, str]] = {
    "agreement": ("subj_num", "verb_form"),
    "npi": ("has_lic", "npi_ok"),
    "binding": ("ante_num", "refl_ok"),
    "concord": ("det_num", "conc_ok"),
}


def core_program(phenomenon: str) -> Program:
    return CORE[phenomenon]()


# ---------------------------------------------------------------------------
# English decisive constructions: hierarchical subject H (A.9-A.11)
# ---------------------------------------------------------------------------

def english_hierarchical(phenomenon: str = "agreement") -> Program:
    """H: mark nouns inside a PP or a fronted clause, take the leftmost other noun."""
    feature, comp = ("Q_v_pl", "verb_form") if phenomenon == "agreement" else ("Q_refl_pl", "refl_ok")
    ops = [
        # after_prep(k) := |>_m[m<k, 1] Q_prep(m) : 0   (the token right before k is a preposition)
        Attention("after_prep", mask_left, const_true(), key("Q_prep"), zero(), R, reads=["Q_prep"]),
        # (A.9) in_pp(j) := |>_k[k<j, Q_det(k) or Q_prep(k)] Q_prep(k) or after_prep(k) : 0
        Attention("in_pp", mask_left,
                  lambda env, qi, kj: env["Q_det"][kj] | env["Q_prep"][kj],
                  lambda env, qi, kj: env["Q_prep"][kj] | env["after_prep"][kj],
                  zero(), R, reads=["Q_det", "Q_prep", "after_prep"]),
        # (A.10) pre_cl(j) := <|_k[k>j, Q_comma(k)] Q_comma(k) : 0
        Attention("pre_cl", mask_right, key("Q_comma"), key("Q_comma"), zero(), L, reads=["Q_comma"]),
        # (A.11) subj_H(i) := <|_j[j<i, Q_noun(j) and not in_pp(j) and not pre_cl(j)] Q_n_pl(j) : 0
        Attention("subj_H", mask_left,
                  lambda env, qi, kj: env["Q_noun"][kj] & ~env["in_pp"][kj] & ~env["pre_cl"][kj],
                  key("Q_n_pl"), zero(), L, reads=["Q_noun", "in_pp", "pre_cl", "Q_n_pl"]),
        Pointwise(comp, _iff("subj_H", feature), reads=["subj_H", feature]),
    ]
    return Program(f"{phenomenon}:H", ops, output=comp, atomic="subj_H", compositional=comp)


def english_linear(phenomenon: str = "agreement") -> Program:
    """L: the leftmost noun, i.e. (A.1) itself."""
    feature, comp = ("Q_v_pl", "verb_form") if phenomenon == "agreement" else ("Q_refl_pl", "refl_ok")
    ops = [Attention("subj_L", mask_left, key("Q_noun"), key("Q_n_pl"), zero(), L, reads=["Q_noun", "Q_n_pl"]),
           Pointwise(comp, _iff("subj_L", feature), reads=["subj_L", feature])]
    return Program(f"{phenomenon}:L", ops, output=comp, atomic="subj_L", compositional=comp)


def english_nearest(phenomenon: str = "agreement") -> Program:
    """N: the nearest preceding noun (|> instead of <|)."""
    feature, comp = ("Q_v_pl", "verb_form") if phenomenon == "agreement" else ("Q_refl_pl", "refl_ok")
    ops = [Attention("subj_N", mask_left, key("Q_noun"), key("Q_n_pl"), zero(), R, reads=["Q_noun", "Q_n_pl"]),
           Pointwise(comp, _iff("subj_N", feature), reads=["subj_N", feature])]
    return Program(f"{phenomenon}:N", ops, output=comp, atomic="subj_N", compositional=comp)


# ---------------------------------------------------------------------------
# German verb-final clauses: case first, order as tie-breaker (A.12-A.14)
# ---------------------------------------------------------------------------

def _nom_ok(env: Env, qi: np.ndarray, kj: np.ndarray) -> np.ndarray:
    """nom_ok(k, j): determiner k and noun j are compatible with nominative case."""
    der, die, das = env["Q_der"][kj], env["Q_die"][kj], env["Q_das"][kj]
    masc, fem, neut, pl = env["Q_masc"][qi], env["Q_fem"][qi], env["Q_neut"][qi], env["Q_pl"][qi]
    return (der & masc & ~pl) | (die & (fem | pl)) | (das & neut & ~pl)


def _form_only(env: Env, qi: np.ndarray, kj: np.ndarray) -> np.ndarray:
    """Competitor S: der, die and das count as nominative, regardless of gender and number."""
    return env["Q_der"][kj] | env["Q_die"][kj] | env["Q_das"][kj]


def german(rule: str = "KO") -> Program:
    """KO (case first), L (linear), N (nearest) or S (determiner form only)."""
    nom = _form_only if rule == "S" else _nom_ok
    ops = [
        # (A.12) can_nom(j) := |>_k[k<j, Q_det(k)] nom_ok(k, j) : 0
        Attention("can_nom", mask_left, key("Q_det"), nom, zero(), R,
                  reads=["Q_det", "Q_der", "Q_die", "Q_das", "Q_masc", "Q_fem", "Q_neut", "Q_pl"]),
        # (A.13) in_cl(j) := |>_k[k<j, Q_dass(k)] Q_dass(k) : 0
        Attention("in_cl", mask_left, key("Q_dass"), key("Q_dass"), zero(), R, reads=["Q_dass"]),
    ]
    if rule in ("KO", "S"):
        score = lambda env, qi, kj: env["Q_noun"][kj] & env["can_nom"][kj] & env["in_cl"][kj]
        direction = L
    elif rule == "L":
        score, direction = (lambda env, qi, kj: env["Q_noun"][kj] & env["in_cl"][kj]), L
    elif rule == "N":
        score, direction = (lambda env, qi, kj: env["Q_noun"][kj] & env["in_cl"][kj]), R
    else:
        raise ValueError(rule)
    # (A.14) subj_num(i) := <|_j[j<i, Q_noun(j) and can_nom(j) and in_cl(j)] Q_pl(j) : 0
    ops.append(Attention("subj_num", mask_left, score, key("Q_pl"), zero(), direction,
                         reads=["Q_noun", "can_nom", "in_cl", "Q_pl"]))
    ops.append(Pointwise("verb_form", _iff("subj_num", "Q_v_pl"), reads=["subj_num", "Q_v_pl"]))
    return Program(f"german:{rule}", ops, output="verb_form", atomic="subj_num", compositional="verb_form")


# ---------------------------------------------------------------------------
# Within-phenomenon alternative SCMs (Appendix D.4)
# ---------------------------------------------------------------------------

RANDOM_POSITION_SEED = 44017


def _family_choice(env: Env, candidates: np.ndarray) -> int:
    """A random candidate, fixed per sentence family (seed 44017)."""
    family = env.get("_family", "")
    h = hashlib.sha256(f"{RANDOM_POSITION_SEED}:{family}".encode()).digest()
    idx = np.flatnonzero(candidates)
    return int(idx[int.from_bytes(h[:4], "little") % len(idx)]) if len(idx) else -1


class _Lookup(Pointwise):
    """A heuristic lookup that is not a single B-RASP attention (nearest in either
    direction, or a random position); evaluated directly on the input."""


def _nearest_any(score: str, value: str) -> Callable[[Env], np.ndarray]:
    def fn(env: Env) -> np.ndarray:
        n = length(env); out = np.zeros(n, dtype=bool)
        pos = np.flatnonzero(env[score])
        for i in range(n):
            cand = pos[pos != i]
            if cand.size:
                j = cand[np.argmin(np.abs(cand - i) + 1e-3 * (cand > i))]   # ties go left
                out[i] = env[value][j]
        return out
    return fn


def _random_pos(score: str, value: str) -> Callable[[Env], np.ndarray]:
    def fn(env: Env) -> np.ndarray:
        j = _family_choice(env, env[score])
        return np.full(length(env), bool(env[value][j]) if j >= 0 else False)
    return fn


ALTERNATIVES = {
    "agreement": ["nearest_noun", "most_recent_noun", "random_noun"],
    "npi": ["any_licensor_left", "adjacent_licensor"],
    "binding": ["nearest_noun_any", "most_recent_noun"],
    "concord": ["noun_number_copy", "following_verb_proxy", "random_number_feature"],
}


def alternative(phenomenon: str, variant: str) -> Program:
    """Replace only the atomic variable's selection rule; keep the Boolean function."""
    atom_name, comp = VARIABLES[phenomenon]
    if phenomenon in ("agreement", "binding"):
        feat = "Q_v_pl" if phenomenon == "agreement" else "Q_refl_pl"
        if variant in ("nearest_noun", "nearest_noun_any"):
            atom = _Lookup(atom_name, _nearest_any("Q_noun", "Q_n_pl"), reads=["Q_noun", "Q_n_pl"])
        elif variant == "most_recent_noun":
            atom = Attention(atom_name, mask_left, key("Q_noun"), key("Q_n_pl"), zero(), R)
        elif variant == "random_noun":
            atom = _Lookup(atom_name, _random_pos("Q_noun", "Q_n_pl"))
        else:
            raise ValueError(variant)
        head = Pointwise(comp, _iff(atom_name, feat))
    elif phenomenon == "npi":
        if variant == "any_licensor_left":       # ignores the clause boundary
            atom = Attention(atom_name, mask_left, key("Q_lic"), key("Q_lic"), zero(), L)
        elif variant == "adjacent_licensor":     # closest preceding licensor-or-noun is a licensor
            atom = Attention(atom_name, mask_left,
                             lambda env, qi, kj: env["Q_lic"][kj] | env["Q_noun"][kj],
                             key("Q_lic"), zero(), R)
        else:
            raise ValueError(variant)
        head = Pointwise(comp, _implies_npi)
    elif phenomenon == "concord":
        if variant == "noun_number_copy":        # reads the noun's own number
            atom = Pointwise(atom_name, lambda env: env["Q_n_pl"].copy())
        elif variant == "following_verb_proxy":  # number of the following verb
            atom = Attention(atom_name, mask_right, key("Q_verb"), key("Q_v_pl"), zero(), L)
        elif variant == "random_number_feature":
            atom = _Lookup(atom_name, _random_pos("Q_numbered", "Q_any_pl"))
        else:
            raise ValueError(variant)
        head = Pointwise(comp, _iff(atom_name, "Q_n_pl"))
    else:
        raise ValueError(phenomenon)
    return Program(f"{phenomenon}:{variant}", [atom, head], output=comp,
                   atomic=atom_name, compositional=comp)


# ---------------------------------------------------------------------------
# Three-input NPI construction (Appendix D.1)
# ---------------------------------------------------------------------------

def npi_three_input() -> Program:
    """npi_ok3 = (not c and x1) or (c and x2), evaluated at the NPI.

    x1: the first clause starts with a licensor; x2: the second clause does;
    c: the NPI is in the second clause (an "and" precedes it).
    """
    ops = [
        Attention("x1", mask_left, const_true(), key("Q_lic"), zero(), L, reads=["Q_lic"]),
        Attention("after_and", mask_left, const_true(), key("Q_and"), zero(), R, reads=["Q_and"]),
        Attention("x2", mask_left, key("after_and"), key("Q_lic"), zero(), L, reads=["after_and", "Q_lic"]),
        Attention("c", mask_left, key("Q_and"), key("Q_and"), zero(), L, reads=["Q_and"]),
        Pointwise("npi_ok3", lambda env: (~env["c"] & env["x1"]) | (env["c"] & env["x2"]),
                  reads=["x1", "x2", "c"]),
    ]
    return Program("npi3", ops, output="npi_ok3", atomic="x1", compositional="npi_ok3")


# ---------------------------------------------------------------------------
# Two-input Boolean functions (Appendix D.10, F.5)
# ---------------------------------------------------------------------------

BOOLEAN_FUNCTIONS: Dict[str, Tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]] = {
    "XOR": ("Non-separable", lambda a, b: a ^ b),
    "XNOR": ("Non-separable", lambda a, b: ~(a ^ b)),
    "AND": ("Conjunctive", lambda a, b: a & b),
    "OR": ("Disjunctive", lambda a, b: a | b),
    "NAND": ("Negated conjunction", lambda a, b: ~(a & b)),
    "NOR": ("Negated disjunction", lambda a, b: ~(a | b)),
    "Implication": ("Asymmetric", lambda a, b: ~a | b),
}


def boolean_task(fn_name: str) -> Program:
    """a := the value of the X token, b := the value of the Y token, out := f(a, b)."""
    f = BOOLEAN_FUNCTIONS[fn_name][1]
    ops = [
        Attention("a", mask_left, key("Q_X"), key("Q_X1"), zero(), L, reads=["Q_X", "Q_X1"]),
        Attention("b", mask_left, key("Q_Y"), key("Q_Y1"), zero(), L, reads=["Q_Y", "Q_Y1"]),
        Pointwise("out", lambda env: f(env["a"], env["b"]), reads=["a", "b"]),
    ]
    return Program(f"bool:{fn_name}", ops, output="out", atomic="a", compositional="out")
