import numpy as np
import torch

from spec2verify.alignment.das import Rotation, interchange
from spec2verify.analysis import stats as ST
from spec2verify.sae.autoencoders import SAE, Normalizer
from spec2verify.sae.interchange import feature_patch


def test_rotation_is_orthogonal_and_full_rank_interchange_copies_source():
    rot = Rotation(16, 701)
    with torch.no_grad():
        rot.A.normal_()
    R = rot.matrix()
    assert torch.allclose(R @ R.T, torch.eye(16), atol=1e-5)
    hb, hs = torch.randn(3, 16), torch.randn(3, 16)
    assert torch.allclose(interchange(hb, hs, R, slice(0, 16)), hs, atol=1e-5)
    assert torch.allclose(interchange(hb, hs, R, slice(0, 0)), hb)


def test_sae_patch_with_no_units_is_identity():
    H = torch.randn(64, 8)
    for arch in ("topk", "jumprelu", "gated"):
        sae, norm = SAE(8, 32, arch, k=4, lam=1e-4), Normalizer(H)
        assert torch.allclose(feature_patch(sae, norm, H[:5], H[5:10], []), H[:5], atol=1e-5)


def test_ordering_statistics():
    k, n, p = ST.exact_sign_test(np.ones(20))
    assert (k, n) == (20, 20) and round(p * 1e6) == 2
    w, p = ST.wilcoxon(np.arange(1, 21))
    assert w == 210
