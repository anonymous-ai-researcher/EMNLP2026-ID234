import numpy as np

from spec2verify.brasp.core import Attention, Direction, Pointwise, Program, key, mask_left, zero
from spec2verify.brasp import programs as P
from spec2verify.brasp.scm import compile_program


def test_attention_semantics_default_and_direction():
    env = {"Q": np.array([0, 1, 0, 1, 0], bool), "V": np.array([0, 1, 0, 0, 0], bool)}
    left = Attention("x", mask_left, key("Q"), key("V"), zero(), Direction.LEFTMOST)
    right = Attention("y", mask_left, key("Q"), key("V"), zero(), Direction.RIGHTMOST)
    # position 0 and 1 see no scored position: default 0
    assert left(env).tolist() == [False, False, True, True, True]
    assert right(env).tolist() == [False, False, True, True, False]


def test_programs_have_one_attention_each():
    for ph in ("agreement", "npi", "binding", "concord"):
        assert P.core_program(ph).attention_depth() == 1


def test_interchange_flips_biconditional():
    from spec2verify.data.vocab import english_env
    scm = compile_program(P.agreement())
    b = english_env("the birds near the lake sing loudly <eos>".split(), "agreement")
    s = english_env("the bird near the lake sing loudly <eos>".split(), "agreement")
    assert scm.output_at(b, 5) is True
    assert scm.interchange(b, {"subj_num": s}, 5) is False
