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
    assert np.all(np.diff(bez_out.controls_x) >= 0)
    assert np.all(np.diff(bez_out.controls_z) >= 0)


def test_exact_cdf_two_bezierv_matches_triangle(two_uniform_bezierv):
    bz_list = [i for i in two_uniform_bezierv]
    conv = Convolver(bz_list)
    
    test_points = [0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
    
    for z in test_points:
        exact_cdf_val = conv.exact_cdf_two_bezierv(z)
        expected = triangular_cdf(z)
        assert exact_cdf_val == pytest.approx(expected, abs=1e-3)

def test_exact_cdf_two_bezierv_boundary_conditions(two_uniform_bezierv):
    bz_list = [i for i in two_uniform_bezierv]
    conv = Convolver(bz_list)
    assert conv.exact_cdf_two_bezierv(-1.0) == pytest.approx(0.0, abs=1e-6)
    assert conv.exact_cdf_two_bezierv(3.0) == pytest.approx(1.0, abs=1e-6)
    assert conv.exact_cdf_two_bezierv(0.0) == pytest.approx(0.0, abs=1e-6)
    assert conv.exact_cdf_two_bezierv(2.0) == pytest.approx(1.0, abs=1e-6)