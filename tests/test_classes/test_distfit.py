import numpy as np
import pytest
from bezierv.classes.distfit import DistFit

def test_quantile_initial_x(normal_data):
    d = DistFit(normal_data, n=4)
    expected = np.quantile(normal_data, np.linspace(0, 1, 5))
    np.testing.assert_allclose(d.init_x, expected)

def test_uniform_initial_x(normal_data):
    d = DistFit(normal_data, n=3, method_init_x="uniform")
    expected = np.linspace(np.min(normal_data), np.max(normal_data), 4)
    np.testing.assert_allclose(d.init_x, expected)

@pytest.mark.parametrize(
    "method, target_mse",
    [
        ("projgrad", 1e-2),
        #("nonlinear", 1e-2),
        ("neldermead", 1e-2),
        ("projsubgrad", 1e-2),
    ],
)

def test_fit_dispatch_and_mse(normal_data, method, target_mse):
    df = DistFit(normal_data, n=3)
    bez, mse = df.fit(method=method, max_iter_PS = 100, max_iter_PG=100)
    assert mse <= target_mse

def test_bad_method_raises(normal_data):
    df = DistFit(normal_data)
    with pytest.raises(ValueError):
        df.fit(method="does-not-exist")