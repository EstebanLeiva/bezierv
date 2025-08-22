import numpy as np
import pytest
from bezierv.classes.bezierv import Bezierv

def test_combinations_and_deltas(linear_bezierv):
    bz = linear_bezierv
    assert np.allclose(bz.comb,       [1, 1])
    assert np.allclose(bz.comb_minus, [1])
    assert np.allclose(bz.deltas_x,   [1.0])
    assert np.allclose(bz.deltas_z,   [1.0])


def test_bernstein_and_polys(linear_bezierv):
    bz = linear_bezierv
    t = 0.37
    assert bz.bernstein(t, 0, bz.comb, 1) == pytest.approx(1 - t)
    assert bz.poly_x(t) == pytest.approx(t)
    assert bz.poly_z(t) == pytest.approx(t)


def test_root_find_eval_x(linear_bezierv):
    bz = linear_bezierv
    for x in (0.0, 0.2, 0.9, 1.0):
        assert bz.root_find(x) == pytest.approx(x)
        px, pz = bz.eval_x(x)
        assert (px, pz) == pytest.approx((x, x))


def test_cdf_and_quantile(linear_bezierv):
    bz = linear_bezierv
    assert bz.cdf_x(-0.1) == 0
    assert bz.cdf_x( 1.1) == 1
    x = 0.42
    assert bz.cdf_x(x) == pytest.approx(x)
    alpha = 0.77
    assert bz.quantile(alpha) == pytest.approx(alpha)


def test_pdf_uniform(linear_bezierv):
    bz = linear_bezierv
    for x in (0.1, 0.5, 0.9):
        assert bz.pdf_x(x) == pytest.approx(1.0)
    for t in (0.25, 0.8):
        assert bz.pdf_t(t) == pytest.approx(1.0)


def test_moments(linear_bezierv):
    bz = linear_bezierv
    bz.update_bezierv(bz.controls_x, bz.controls_z)
    assert bz.get_mean() == pytest.approx(0.5)
    bz.bounds = bz.support
    assert bz.get_variance() == pytest.approx(1/12, rel=1e-3)

def test_plot_functions_do_not_crash(linear_bezierv):
    bz = linear_bezierv
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    bz.plot_cdf(ax=ax)
    bz.plot_pdf(ax=ax)