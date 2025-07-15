import numpy as np
from bezierv.algorithms import nelder_mead as nm
from bezierv.classes.bezierv import Bezierv


def test_objective_function_zero_for_perfect_fit():
    n = 1
    m = 3
    data = np.array([0.0, 0.5, 1.0])
    x = np.array([0.0, 1.0])
    z = np.array([0.0, 1.0])
    emp_cdf = data.copy()
    bez = Bezierv(n)

    concat = np.concatenate((x, z))
    val = nm.objective_function(concat, n, m, data, bez, emp_cdf)
    assert val < 1e-12, "Perfect fit should give ~0 MSE"

def test_objective_function_lagrangian_adds_penalty():
    n = 1
    m = 3
    data = np.array([0.0, 0.5, 1.0])
    x_bad = np.array([1.0, -0.5])
    z_bad = np.array([0.2, 0.2])
    emp_cdf = data.copy()
    bez = Bezierv(n)
    good = np.concatenate((np.array([0.0, 1.0]), np.array([0.0, 1.0])))
    bad  = np.concatenate((x_bad, z_bad))
    mse_good = nm.objective_function_lagrangian(good, n, m, data, bez, emp_cdf)
    mse_bad  = nm.objective_function_lagrangian(bad,  n, m, data, bez, emp_cdf)
    assert mse_bad > mse_good + 1.0, "Violating constraints should increase objective markedly"

def test_fit_returns_low_mse_and_updates_bezier():
    n = 1
    m = 3
    data = np.array([0.0, 0.5, 1.0])
    init_x = np.array([-0.05, 1.05])
    init_z = np.array([0, 1])
    emp_cdf = data.copy()
    bez = Bezierv(n)

    fitted_bez, mse = nm.fit(
        n=n,
        m=m,
        data=data,
        bezierv=bez,
        init_x=init_x,
        init_z=init_z,
        emp_cdf_data=emp_cdf,
        max_iter=1000
        )

    assert mse < 1e-5, "Expected near-perfect fit with dummy minimise"
    np.testing.assert_allclose(fitted_bez.controls_x, np.array([0.0, 1.0]), atol=1e-2)
    np.testing.assert_allclose(fitted_bez.controls_z, np.array([0.0, 1.0]), atol=1e-2)