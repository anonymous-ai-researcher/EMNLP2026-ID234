import torch

from spec2verify.models.architectures import build_model, count_parameters, parameter_formula


def test_parameter_formulas_are_exact():
    for w in (32, 48, 128, 640):
        for arch in ("transformer", "lstm", "mamba"):
            m = build_model(arch, 135, w, max_len=64, d_state=256)
            assert count_parameters(m) == parameter_formula(arch, w)


def test_shared_configuration_sizes():
    sizes = {a: count_parameters(build_model(a, 135, 128, max_len=30, d_state=16)) for a in ("transformer", "lstm", "mamba")}
    assert round(sizes["transformer"] / 1e6, 2) == 0.42
    assert round(sizes["lstm"] / 1e6, 2) == 0.28
    assert round(sizes["mamba"] / 1e6, 2) == 0.25


def test_staged_forward_matches_full_forward():
    ids = torch.randint(2, 135, (4, 9)); tgt = torch.tensor([3, 4, 5, 8])
    for arch in ("transformer", "lstm", "mamba"):
        m = build_model(arch, 135, 32).eval()
        full = m(ids, tgt)
        for site in m.sites:
            assert torch.allclose(m.decode(m.encode(ids, site), site, ids, tgt), full, atol=1e-5)
