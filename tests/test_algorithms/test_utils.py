import numpy as np
import pytest
from bezierv.classes.bezierv import Bezierv
from bezierv.algorithms import utils as utils

def test_root_find_linear_case():
    """
    For the linear Bézier curve x(t) = t, root_find should return t = data_point.
    """
    n = 1
    bez = Bezierv(n, controls_x=np.array([0.0, 1.0]), controls_z=np.array([0.0, 1.0]))
    controls_x = np.array([0.0, 1.0])
    for dp in [0.0, 0.25, 0.9, 1.0]:
        t = utils.root_find(n, bez, controls_x, dp)
        assert abs(t - dp) < 1e-12, f"Expected {dp}, got {t}"


def test_get_t_returns_vector_of_roots():
    """
    get_t should call root_find for each data point and return the same values
    when the mapping is linear.
    """
    n = 1
    m = 5
    data = np.linspace(0, 1, m)
    controls_x = np.array([0.0, 1.0])
    bez = Bezierv(n, controls_x=np.array([0.0, 1.0]), controls_z=np.array([0.0, 1.0]))

    t_vals = utils.get_t(n, m, data, bez, controls_x)
    np.testing.assert_allclose(t_vals, data, atol=1e-12)


def test_root_find_errors_outside_interval():
    """
    Raises ValueError when the data point is outside [0,1].
    root_find should propagate that error.
    """
    n = 1
    bez = Bezierv(n, controls_x=np.array([0.0, 1.0]), controls_z=np.array([0.0, 1.0]))
    controls_x = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        _ = utils.root_find(n, bez, controls_x, 1.5)