import numpy as np
import pytest
from bezierv.classes.convolver import Convolver
from bezierv.classes.bezierv import Bezierv

def triangular_cdf(z):
    """
    CDF of Z = X+Y with X,Y ~ U(0,1):
      F_Z(z) = 0                 (z ≤ 0)
               z² / 2            (0 < z < 1)
               1 - (2 - z)² / 2  (1 ≤ z < 2)
               1                 (z ≥ 2)
    """
    if z <= 0:
        return 0.0
    if z < 1:
        return 0.5 * z * z
    if z < 2:
        return 1 - 0.5 * (2 - z) ** 2
    return 1.0

def test_cdf_z_matches_triangle(two_uniform_bezierv):
    bz_list = [i for i in two_uniform_bezierv]
    conv = Convolver(bz_list)
    bz_conv = conv.convolve(n_sims=1000, rng=42)

    for x in [0, 0.2, 0.8, 1.0, 1.4, 2]:
        val = bz_conv.cdf_x(x)
        expected = triangular_cdf(x)
        assert val == pytest.approx(expected, abs=5e-2)

def test_conv_calls_distfit_and_returns(two_uniform_bezierv):
    bz_list = [i for i in two_uniform_bezierv]
    conv = Convolver(bz_list)
    bez_out = conv.convolve(method="projgrad")
    assert isinstance(bez_out, Bezierv)
    assert bez_out.check_ordering() is True