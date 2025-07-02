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
    bx, by = two_uniform_bezierv
    conv = Convolver(bx, by, grid=50)

    for z in [0.0, 0.2, 0.8, 1.0, 1.4, 2.0]:
        val = conv.cdf_z(z)
        expected = triangular_cdf(z)
        assert val == pytest.approx(expected, abs=5e-3)

def test_conv_calls_distfit_and_returns(two_uniform_bezierv):
    bx, by = two_uniform_bezierv
    conv = Convolver(bx, by, grid=20)
    bez_out, mse = conv.conv(method="projgrad")
    assert isinstance(bez_out, Bezierv)
    assert bez_out.check_ordering() is True