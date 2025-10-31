import numpy as np
from bezierv.algorithms import proj_subgrad as ps
from bezierv.algorithms import utils as utils
from bezierv.classes.bezierv import Bezierv

def test_subgrad_zero_when_perfect_fit():
    n = 1
    t = np.array([0.0, 0.5, 1.0])
    m = t.size
    emp_cdf = t.copy()
    controls_z = np.array([0.0, 1.0])
    bez = Bezierv(n, controls_x=np.array([0.0, 1.0]), controls_z=np.array([0.0, 1.0]))
    g_x, g_z = ps.subgrad(n, m, bez, t, controls_z, emp_cdf)
    np.testing.assert_allclose(g_x, 0.0)
    np.testing.assert_allclose(g_z, 0.0)

def test_fit_converges_to_linear_solution():
    """
    The full projected-subgradient solver should find the perfect fit on a
    toy linear CDF — i.e. MSE near 0 and final z ≈ [0,1].
    """
    n = 1
    t_init = np.array([0.0, 0.5, 1.0])
    m = t_init.size
    data = t_init.copy()
    emp_cdf = t_init.copy()
    init_x = np.array([2.0, 8.0])
    init_z = np.array([0.4, 0.7])
    bez = Bezierv(n, controls_x=np.array([0.0, 1.0]), controls_z=np.array([0.0, 1.0]))

    _, mse = ps.fit(
        n=n,
        m=m,
        data=data,
        bezierv=bez,
        init_x=init_x,
        init_z=init_z,
        init_t=t_init,
        emp_cdf_data=emp_cdf,
        step_size=0.5,
        max_iter=200
    )

    assert mse < 1e-6, "Expected near-perfect fit"
    # check the Bézier object got updated to the best values
    np.testing.assert_allclose(bez.controls_z, np.array([0.0, 1.0]), atol=1e-3)
    np.testing.assert_allclose(bez.controls_x, data[[0, -1]], atol=1e-3)