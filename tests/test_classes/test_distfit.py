import numpy as np
import pytest
from bezierv.classes import distfit as distfit_mod
from bezierv.classes.distfit import DistFit, ProjGradOptions, NonLinearOptions, NelderMeadOptions, MLEOptions

def test_quantile_initial_x(normal_data):
    d = DistFit(normal_data, n=4)
    expected = np.quantile(normal_data, np.linspace(0, 1, 5))
    np.testing.assert_allclose(d.init_x, expected)

def test_uniform_initial_x(normal_data):
    d = DistFit(normal_data, n=3, method_init_x="uniform")
    expected = np.linspace(np.min(normal_data), np.max(normal_data), 4)
    np.testing.assert_allclose(d.init_x, expected)

@pytest.mark.parametrize(
    "method, algorithm, options, target_metric",
    [
        ("mse", "projgrad", ProjGradOptions(max_iter=100), 1e-2),
        ("mse", "neldermead", NelderMeadOptions(), 1e-2),
        ("mle", None, MLEOptions(), 140.0),
    ],
)
def test_fit_dispatch_and_metric(normal_data, method, algorithm, options, target_metric):
    df = DistFit(normal_data, n=3)
    bez, metric = df.fit(method=method, algorithm=algorithm, options=options)
    assert metric <= target_metric

def test_fit_nonlinear_dispatch(normal_data, monkeypatch):
    df = DistFit(normal_data, n=3)
    opts = NonLinearOptions()
    captured = {}

    def fake_fit(n, m, data, bezierv, init_x, init_z, init_t, emp_cdf_data, solver, solver_options, feas_tol):
        captured.update(
            n=n, m=m, data=data, bezierv=bezierv, init_x=init_x, init_z=init_z,
            init_t=init_t, emp_cdf_data=emp_cdf_data, solver=solver, solver_options=solver_options,
            feas_tol=feas_tol,
        )
        return bezierv, 0.123

    monkeypatch.setattr(distfit_mod.nl, "fit", fake_fit)

    _, metric = df.fit(method="mse", algorithm="nonlinear", options=opts)

    assert metric == 0.123
    assert df.mse == 0.123
    assert captured["n"] == df.n
    assert captured["m"] == df.m
    assert captured["data"] is df.data
    assert captured["bezierv"] is df.bezierv
    assert captured["init_x"] is df.init_x
    assert captured["init_z"] is df.init_z
    assert captured["init_t"] is df.init_t
    assert captured["emp_cdf_data"] is df.emp_cdf_data
    assert captured["solver"] == opts.solver
    assert captured["solver_options"] == opts.solver_options
    assert captured["feas_tol"] == opts.feas_tol


def test_bad_method_raises(normal_data):
    df = DistFit(normal_data)
    with pytest.raises(ValueError):
        df.fit(method="does-not-exist")