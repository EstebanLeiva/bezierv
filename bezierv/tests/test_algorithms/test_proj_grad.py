import numpy as np
from bezierv.algorithms import proj_grad as pg
from bezierv.algorithms import utils as utils
from bezierv.classes.bezierv import Bezierv


def test_grad_zero_error_returns_zero_gradient():
    """When the Bézier curve matches the empirical CDF exactly, the gradient is 0."""
    n = 1                      # linear curve: z(t) = t
    t = np.array([0.0, 0.5, 1.0])
    m = t.size
    emp_cdf = t.copy()         # perfect fit
    controls_z = np.array([0.0, 1.0])
    bez = Bezierv(n)

    g = pg.grad(n, m, t, bez, controls_z, emp_cdf)
    np.testing.assert_allclose(g, np.zeros_like(g), atol=1e-12)

def test_project_z_clips_sorts_and_enforces_bounds():
    """
    Test the project_z function to ensure it clips, sorts, and enforces bounds correctly.
    """
    raw = np.array([-0.2, 0.9, 0.7, 1.2])
    projected = pg.project_z(raw.copy())
    expected = np.array([0.0, 0.7, 0.9, 1.0])
    np.testing.assert_allclose(projected, expected)
    np.testing.assert_allclose(pg.project_z(projected.copy()), expected)

def test_fit_converges_and_returns_low_mse():
    """
    On a toy linear CDF the projected-gradient solver should converge
    to z = [0,1] and yield (almost) zero MSE.
    """
    n = 1
    t = np.array([0.0, 0.5, 1.0])
    m = t.size
    data = t.copy()
    emp_cdf = t.copy()
    init_x = np.array([0.0, 1.0])
    init_z = np.array([0.2, 0.8])
    bez = Bezierv(n)

    bezierv , mse = pg.fit(
        n=n,
        m=m,
        data=data,
        bezierv=bez,
        init_x=init_x,
        init_z=init_z,
        t=t,
        emp_cdf_data=emp_cdf,
        step_size=0.2,
        max_iter=1_000,
        threshold=1e-6,
    )

    assert isinstance(bezierv, Bezierv), "Expected Bezierv instance as output"
    np.testing.assert_allclose(bezierv.controls_z, np.array([0., 1.])), "Expected z control points to be [0, 1]"
    np.testing.assert_allclose(bezierv.controls_x, np.array([0., 1.])), "Expected x control points to be [0, 1]"
    assert mse < 1e-6, "Solver failed to reach a near-perfect fit"
    


