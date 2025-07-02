import numpy as np
import pytest
from bezierv.classes.bezierv import Bezierv

def test_combinations_and_deltas(linear_bezierv):
    bezierv = linear_bezierv
    assert np.allclose(bezierv.comb, [1, 1])
    assert np.allclose(bezierv.comb_minus,  [1])
    assert np.allclose(bezierv.deltas_x, [1.0])
    assert np.allclose(bezierv.deltas_z, [1.0])


def test_bernstein_and_poly_eval(linear_bezierv):
    bezierv = linear_bezierv
    t = 0.37
    assert bezierv.bernstein(t, 0, bezierv.comb, 1) == pytest.approx(1 - t)
    assert bezierv.poly_x(t) == pytest.approx(t)
    assert bezierv.poly_z(t) == pytest.approx(t)


def test_root_find_and_eval_x(linear_bezierv):
    bezierv = linear_bezierv
    for x in (0.0, 0.25, 0.8, 1.0):
        assert bezierv.root_find(x) == pytest.approx(x)
        p_x, p_z = bezierv.eval_x(x)
        assert p_x == pytest.approx(x)
        assert p_z == pytest.approx(x)


def test_cdf_and_quantile(linear_bezierv):
    bezierv = linear_bezierv
    # Outside support
    assert bezierv.cdf_x(-0.1) == 0
    assert bezierv.cdf_x( 1.1) == 1
    # Inside
    x = 0.42
    assert bezierv.cdf_x(x) == pytest.approx(x)
    # Inverse
    alpha = 0.77
    assert bezierv.quantile(alpha) == pytest.approx(alpha)


def test_pdf_uniform(linear_bezierv):
    bezierv = linear_bezierv
    for x in (0.1, 0.5, 0.9):
        assert bezierv.pdf_x(x) == pytest.approx(1.0)
    for t in (0.2, 0.7):
        assert bezierv.pdf_t(t) == pytest.approx(1.0)


def test_moments_mean_and_variance(linear_bezierv):
    bezierv = linear_bezierv
    bezierv.update_bezierv(bezierv.controls_x, bezierv.controls_z, (0.0, 1.0))
    assert bezierv.get_mean() == pytest.approx(0.5)
    assert bezierv.get_variance() == pytest.approx(1/12, rel=1e-3)


def test_check_ordering(linear_bezierv):
    good = linear_bezierv
    assert good.check_ordering() is True
    bad = Bezierv(1,
                     controls_x=np.array([0.8, 0.2]),
                     controls_z=np.array([0.0, 1.0]))
    with pytest.raises(TypeError):
        bad.check_ordering()


def test_plot_functions_do_not_crash(linear_bezierv):
    bezierv = linear_bezierv
    bezierv.plot_cdf()
    bezierv.plot_pdf()