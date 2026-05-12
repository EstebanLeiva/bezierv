import numpy as np
import pytest
from pyomo.opt import TerminationCondition

from bezierv.algorithms import non_linear as nl
from bezierv.classes.bezierv import Bezierv


class _FakeSolverResults:
    class _Solver:
        def __init__(self, status, termination_condition):
            self.status = status
            self.termination_condition = termination_condition

    def __init__(self, status, termination_condition):
        self.solver = self._Solver(status, termination_condition)


class _FakeSolver:
    """Stand-in for a Pyomo solver. Optionally writes values into the model."""

    def __init__(self, termination_condition, x_values=None, z_values=None,
                 t_values=None, f_hat_values=None, status="ok", raise_exc=None):
        self.termination_condition = termination_condition
        self.x_values = x_values
        self.z_values = z_values
        self.t_values = t_values
        self.f_hat_values = f_hat_values
        self.status = status
        self.raise_exc = raise_exc
        self.options = {}
        self.solve_called = False
        self.last_solve_kwargs = None

    def solve(self, model, **kwargs):
        self.solve_called = True
        self.last_solve_kwargs = kwargs
        if self.raise_exc is not None:
            raise self.raise_exc

        if self.x_values is not None:
            for i, val in enumerate(self.x_values):
                model.x[i].set_value(float(val))
        if self.z_values is not None:
            for i, val in enumerate(self.z_values):
                model.z[i].set_value(float(val))
        if self.t_values is not None:
            for j, val in enumerate(self.t_values, start=1):
                model.t[j].set_value(float(val))
        if self.f_hat_values is not None:
            for j, val in enumerate(self.f_hat_values, start=1):
                model.F_hat[j].set_value(float(val))

        return _FakeSolverResults(self.status, self.termination_condition)


def _linear_problem():
    """Small degree-1 fitting problem with a known perfect linear solution."""
    n = 1
    data = np.array([0.0, 0.5, 1.0])
    m = data.size
    emp_cdf = data.copy()
    init_x = np.array([0.0, 1.0])
    init_z = np.array([0.2, 0.8])
    init_t = np.array([0.0, 0.5, 1.0])
    bez = Bezierv(n,
                  controls_x=np.array([0.0, 1.0]),
                  controls_z=np.array([0.0, 1.0]))
    return dict(n=n, m=m, data=data, bezierv=bez, init_x=init_x,
                init_z=init_z, init_t=init_t, emp_cdf_data=emp_cdf)


def _install_fake_solver(monkeypatch, fake):
    monkeypatch.setattr(nl, "SolverFactory", lambda name: fake)


def test_fit_returns_bezierv_and_mse_on_optimal(monkeypatch):
    p = _linear_problem()
    fake = _FakeSolver(
        termination_condition=TerminationCondition.optimal,
        x_values=[0.0, 1.0],
        z_values=[0.0, 1.0],
        t_values=[0.0, 0.5, 1.0],
        f_hat_values=[0.0, 0.5, 1.0],
    )
    _install_fake_solver(monkeypatch, fake)

    bezierv, mse = nl.fit(solver="fake", **p)

    assert isinstance(bezierv, Bezierv)
    assert np.isfinite(mse)
    np.testing.assert_allclose(bezierv.controls_x, [0.0, 1.0])
    np.testing.assert_allclose(bezierv.controls_z, [0.0, 1.0])


def test_fit_updates_bezierv_in_place_on_success(monkeypatch):
    p = _linear_problem()
    original_bez = p["bezierv"]
    fake = _FakeSolver(
        termination_condition=TerminationCondition.optimal,
        x_values=[0.0, 1.0],
        z_values=[0.0, 1.0],
        t_values=[0.0, 0.5, 1.0],
        f_hat_values=[0.0, 0.5, 1.0],
    )
    _install_fake_solver(monkeypatch, fake)

    returned, _ = nl.fit(solver="fake", **p)

    assert returned is original_bez
    np.testing.assert_allclose(returned.controls_x, [0.0, 1.0])
    np.testing.assert_allclose(returned.controls_z, [0.0, 1.0])


def test_fit_accepts_locally_optimal_with_warning(monkeypatch):
    p = _linear_problem()
    fake = _FakeSolver(
        termination_condition=TerminationCondition.locallyOptimal,
        x_values=[0.0, 1.0],
        z_values=[0.0, 1.0],
        t_values=[0.0, 0.5, 1.0],
        f_hat_values=[0.0, 0.5, 1.0],
    )
    _install_fake_solver(monkeypatch, fake)

    with pytest.warns(RuntimeWarning, match="did not reach optimality"):
        bezierv, mse = nl.fit(solver="fake", **p)

    assert np.isfinite(mse)
    np.testing.assert_allclose(bezierv.controls_z, [0.0, 1.0])


def test_fit_accepts_max_iterations_with_warning(monkeypatch):
    p = _linear_problem()
    fake = _FakeSolver(
        termination_condition=TerminationCondition.maxIterations,
        x_values=[0.0, 1.0],
        z_values=[0.0, 1.0],
        t_values=[0.0, 0.5, 1.0],
        f_hat_values=[0.0, 0.5, 1.0],
    )
    _install_fake_solver(monkeypatch, fake)

    with pytest.warns(RuntimeWarning, match="did not reach optimality"):
        bezierv, mse = nl.fit(solver="fake", **p)

    assert np.isfinite(mse)
    np.testing.assert_allclose(bezierv.controls_x, [0.0, 1.0])


def test_fit_solver_options_forwarded(monkeypatch):
    p = _linear_problem()
    fake = _FakeSolver(
        termination_condition=TerminationCondition.optimal,
        x_values=[0.0, 1.0],
        z_values=[0.0, 1.0],
        t_values=[0.0, 0.5, 1.0],
        f_hat_values=[0.0, 0.5, 1.0],
    )
    _install_fake_solver(monkeypatch, fake)

    opts = {"timelimit": 30, "tee": True, "options": {"max_iter": 1234, "tol": 1e-9}}
    nl.fit(solver="fake", solver_options=opts, **p)

    assert fake.last_solve_kwargs["timelimit"] == 30
    assert fake.last_solve_kwargs["tee"] is True
    assert fake.last_solve_kwargs["options"] == {"max_iter": 1234, "tol": 1e-9}


def test_fit_solver_options_none_is_safe(monkeypatch):
    p = _linear_problem()
    fake = _FakeSolver(
        termination_condition=TerminationCondition.optimal,
        x_values=[0.0, 1.0],
        z_values=[0.0, 1.0],
        t_values=[0.0, 0.5, 1.0],
        f_hat_values=[0.0, 0.5, 1.0],
    )
    _install_fake_solver(monkeypatch, fake)

    nl.fit(solver="fake", solver_options=None, **p)

    assert fake.solve_called
    assert fake.last_solve_kwargs == {}


def test_fit_warns_and_returns_nan_on_solver_failure(monkeypatch):
    p = _linear_problem()
    bez = p["bezierv"]
    original_x = bez.controls_x.copy()
    original_z = bez.controls_z.copy()

    fake = _FakeSolver(
        termination_condition=TerminationCondition.infeasible,
        status="warning",
    )
    _install_fake_solver(monkeypatch, fake)

    with pytest.warns(RuntimeWarning, match="Solver failed"):
        bezierv, mse = nl.fit(solver="fake", **p)

    assert np.isnan(mse)
    np.testing.assert_array_equal(bezierv.controls_x, original_x)
    np.testing.assert_array_equal(bezierv.controls_z, original_z)


def test_fit_propagates_solver_exception(monkeypatch):
    p = _linear_problem()
    bez = p["bezierv"]
    original_x = bez.controls_x.copy()
    original_z = bez.controls_z.copy()

    fake = _FakeSolver(
        termination_condition=TerminationCondition.optimal,
        raise_exc=RuntimeError("solver blew up"),
    )
    _install_fake_solver(monkeypatch, fake)

    with pytest.raises(RuntimeError, match="solver blew up"):
        nl.fit(solver="fake", **p)

    np.testing.assert_array_equal(bez.controls_x, original_x)
    np.testing.assert_array_equal(bez.controls_z, original_z)
