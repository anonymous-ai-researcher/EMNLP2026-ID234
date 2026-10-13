"""Compilation of a B-RASP program into a structural causal model (Appendix A.3).

Every operation ``P_t`` becomes one endogenous variable ``U_t`` whose
structural equation is the operation itself; its parents are the vectors the
operation reads.  The output of the SCM is the program's output vector at a
designated position.

An interchange intervention on ``U`` sets ``U`` to the value it takes under a
source input and recomputes everything downstream; the counterfactual label
``C_{U <- s}(b)`` is the resulting output on the base input ``b``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .core import Env, Program


@dataclass
class SCM:
    program: Program
    inputs: List[str] = field(default_factory=list)
    endogenous: List[str] = field(default_factory=list)
    parents: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def output(self) -> str:
        return self.program.output

    def run(self, env: Env, overrides: Optional[Dict[str, np.ndarray]] = None) -> Env:
        return self.program.run(env, overrides)

    def value(self, env: Env, variable: str, position: int) -> bool:
        return bool(self.run(env)[variable][position])

    def output_at(self, env: Env, position: int) -> bool:
        return self.value(env, self.output, position)

    def interchange(self, base: Env, sources: Mapping[str, Env], position: int,
                    pinned: Sequence[str] = (), output_position: Optional[int] = None) -> bool:
        """``C_{U <- s}(b)`` at ``position``.

        ``sources`` maps each intervened variable to the input it is taken
        from (one entry for a single intervention, two for a double one).
        ``pinned`` variables are held at their *base* value (screening-off).
        The intervention acts at ``position``, the site where the variable
        is consumed; the output is read at ``output_position`` (default: the
        same position).
        """
        overrides: Dict[str, np.ndarray] = {}
        base_run = self.run(base)
        for var, src in sources.items():
            src_val = self.run(src)[var]
            ov = np.ma.masked_all(len(base_run[var]), dtype=bool)
            ov[position] = src_val[position]
            overrides[var] = ov
        for var in pinned:
            ov = np.ma.masked_all(len(base_run[var]), dtype=bool)
            ov[position] = base_run[var][position]
            overrides[var] = ov
        out = position if output_position is None else output_position
        return bool(self.run(base, overrides)[self.output][out])


def compile_program(program: Program, input_names: Sequence[str] = ()) -> SCM:
    endo = [op.name for op in program.operations]
    parents = {op.name: list(op.reads) for op in program.operations}
    ins = sorted(set(input_names) | {r for op in program.operations for r in op.reads} - set(endo))
    return SCM(program=program, inputs=ins, endogenous=endo, parents=parents)
