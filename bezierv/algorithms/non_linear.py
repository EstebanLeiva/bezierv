import warnings
import pyomo.environ as pyo
import numpy as np
from bezierv.classes.bezierv import Bezierv
from pyomo.opt import SolverFactory, TerminationCondition

def _max_infeasibility(model) -> tuple[float, float]:
    """
    Return ``(max_constraint_violation, max_bound_violation)`` for ``model``.

    Iterates all active constraints and variables and returns the worst
    absolute violation found. Used to verify that an incumbent returned by
    the NLP solver (e.g. on ``maxIterations`` / ``maxTimeLimit``) actually
    satisfies the problem before it is trusted.
    """
    max_ctr = 0.0
    for ctr in model.component_data_objects(pyo.Constraint, active=True):
        body = pyo.value(ctr.body)
        if ctr.lower is not None:
            lower = pyo.value(ctr.lower)
            if body < lower:
                max_ctr = max(max_ctr, lower - body)
        if ctr.upper is not None:
            upper = pyo.value(ctr.upper)
            if body > upper:
                max_ctr = max(max_ctr, body - upper)

    max_bnd = 0.0
    for var in model.component_data_objects(pyo.Var, active=True):
        v = var.value
        if v is None:
            continue
        if var.lb is not None and v < var.lb:
            max_bnd = max(max_bnd, var.lb - v)
        if var.ub is not None and v > var.ub:
            max_bnd = max(max_bnd, v - var.ub)

    return max_ctr, max_bnd


def fit(n: int,
        m: int,
        data: np.ndarray,
        bezierv: Bezierv,
        init_x: np.ndarray,
        init_z: np.ndarray,
        init_t: np.ndarray,
        emp_cdf_data: np.ndarray,
        solver: str,
        solver_options: dict = None,
        feas_tol: float = 1e-4) -> tuple[Bezierv, float]:
    """
    Fit a Bézier distribution via an external solver (e.g. IPOPT).

    Formulates the minimum-error estimation problem as a Pyomo
    ``ConcreteModel`` and solves it with the specified NLP solver.

    Parameters
    ----------
    n : int
        Degree of the Bézier curve (``n + 1`` control points).
    m : int
        Sample size.
    data : numpy.ndarray, shape (m,)
        Sorted observed sample values.
    bezierv : Bezierv
        Bézier random variable to be updated with the fitted parameters.
    init_x : numpy.ndarray, shape (n + 1,)
        Initial x-coordinates of the control points.
    init_z : numpy.ndarray, shape (n + 1,)
        Initial z-coordinates (CDF values) of the control points.
    init_t : numpy.ndarray, shape (m,)
        Initial parameter values ``t ∈ [0, 1]`` for each observation.
    emp_cdf_data : numpy.ndarray, shape (m,)
        Empirical CDF values at each observation.
    solver : str
        Name of the NLP solver recognized by Pyomo (e.g. ``'ipopt'``).
    solver_options : dict, optional
        Keyword arguments forwarded to ``pyo_solver.solve(model, **solver_options)``.
        Supports Pyomo's ``solve`` kwargs (e.g. ``tee``, ``timelimit``) and
        solver-native options nested under the ``options`` key
        (e.g. ``{'timelimit': 60, 'tee': False, 'options': {'max_iter': 5000, 'tol': 1e-8}}``
        for IPOPT). Defaults to ``None``.
    feas_tol : float, optional
        Maximum allowed constraint or bound violation when accepting an
        incumbent solution from a non-optimal termination (e.g.
        ``maxIterations``, ``maxTimeLimit``). If the worst violation
        exceeds this tolerance, a :class:`RuntimeError` is raised.
        Defaults to ``1e-4``, matching IPOPT's ``constr_viol_tol`` default.

    Returns
    -------
    bezierv : Bezierv
        Updated Bézier random variable. Unchanged if the solver does not
        reach an optimal solution.
    mse : float
        Final MSE at the optimal solution, or ``nan`` if the solver fails.

    Raises
    ------
    RuntimeError
        If the solver returns a non-optimal acceptable termination
        (``maxIterations``, ``maxTimeLimit``, ``locallyOptimal``,
        ``feasible``) but the incumbent violates the constraints or
        variable bounds by more than ``feas_tol``, or if the solver
        terminates with any other (non-acceptable) status.
    Exception
        Any exception raised by the Pyomo solver call is propagated to
        the caller unchanged.

    Notes
    -----
    Constraints enforce: boundary conditions ``z₀ = 0``, ``zₙ = 1``;
    monotonicity of ``x`` and ``z``; support bounds ``x₀ ≤ data[0]``,
    ``xₙ ≥ data[-1]``; and parameter boundary ``t₁ = 0``, ``tₘ = 1``.
    """
    # Defining the optimization model
    
    model = pyo.ConcreteModel()

    # Sets
    model.N = pyo.Set(initialize=list(range(n + 1))) # N = 0,...,i,...,n
    model.N_n = pyo.Set(initialize=list(range(n))) # N = 0,...,i,...,n-1
    model.M = pyo.Set(initialize=list(range(1, m + 1))) # M = 1,...,j,...,m
    model.M_m = pyo.Set(initialize=list(range(1, m))) # M = 1,...,j,...,m-1

    # Decision variables
    # Control points. Box constraints.
    X_min = data[0];
    X_max = data[-1];
    # var x{i in 0..n} >=X[1], <=X[m];
    # Initialization:
    def init_x_rule(model, i):
      return float(init_x[i])
    model.x = pyo.Var(model.N, within=pyo.Reals, bounds=(X_min, X_max), initialize=init_x_rule) 
    # var z{i in 0..n} >=0, <=1;
    # Initialization:
    def init_z_rule(model, i):
      return float(init_z[i])
    model.z = pyo.Var(model.N, within=pyo.NonNegativeReals, bounds=(0, 1), initialize=init_z_rule) 
    # Bezier 'time' parameter t for the j-th sample point.
    # var t{j in 1..m} >=0, <= 1;
    # Initialization:  
    def init_t_rule(model, j):
      return float(init_t[j - 1])  # j starts from 1, so we access init_t with j-1
    model.t = pyo.Var(model.M, within=pyo.NonNegativeReals, bounds=(0,1), initialize=init_t_rule )         
    # Estimated cdf for the j-th sample point.
    # var F_hat{j in 1..m} >=0, <= 1;
    model.F_hat = pyo.Var(model.M, within=pyo.NonNegativeReals, bounds=(0,1) ) 

    # Objective function
    # minimize mean_square_error:
    #    1/m * sum {j in 1..m} ( ( j/m - F_hat[j] )^2);
    def mse_rule(model):
      return (1 / m) * sum((emp_cdf_data[j - 1] - model.F_hat[j])**2 for j in model.M)
    model.mse = pyo.Objective(rule=mse_rule, sense=pyo.minimize )

    # Constraints
    # subject to F_hat_estimates {j in 1..m}:
    #    sum{i in 0..n}( comb[i]*t[j]^i*(1-t[j])^(n-i)*z[i] ) = F_hat[j];
    def F_hat_rule(model, j):
      return sum(bezierv.comb[i] * model.t[j]**i * (1 - model.t[j])**(n - i) * model.z[i] for i in model.N ) == model.F_hat[j]
    model.ctr_F_hat = pyo.Constraint(model.M , rule=F_hat_rule)

    # subject to data_sample {j in 1..m}:
    #    sum{i in 0..n}( comb[i]*t[j]^i*(1-t[j])^(n-i)*x[i] ) = X[j];
    def data_sample_rule(model, j):
      return sum(bezierv.comb[i] * model.t[j]**i * (1 - model.t[j])**(n - i) * model.x[i] for i in model.N ) == data[j-1]
    model.ctr_sample = pyo.Constraint(model.M , rule=data_sample_rule)
    
    # subject to convexity_x {i in 0..n-1}:
    #    x[i] <= x[i+1];
    def convexity_x_rule(model, i):
      return model.x[i] <= model.x[i + 1]
    model.ctr_convexity_x = pyo.Constraint(model.N_n , rule=convexity_x_rule)

    # subject to convexity_z {i in 0..n-1}:
    #    z[i] <= z[i+1];
    def convexity_z_rule(model, i):
      return model.z[i] <= model.z[i + 1]
    model.ctr_convexity_z = pyo.Constraint(model.N_n , rule=convexity_z_rule)

    # subject to first_control_x:
    #    x[0] = X[1];
    model.first_control_x = pyo.Constraint(expr=model.x[0] <= data[0])
    # subject to first_control_z:
    #    z[0] = 0;
    model.first_control_z = pyo.Constraint(expr=model.z[0] == 0)

    # subject to last_control_x:
    #    x[n] = X[m];
    model.last_control_x = pyo.Constraint(expr=model.x[n] >= data[-1]) 
    # subject to last_control_z:
    #    z[n] = 1;
    model.last_control_z = pyo.Constraint(expr=model.z[n] == 1)
    
    # subject to first_data_t:
    #    t[1] = 0;
    model.first_t = pyo.Constraint(expr=model.t[1] == 0)
    # subject to last_data_t:
    #    t[m] = 1;
    model.last_t = pyo.Constraint(expr=model.t[m] == 1)
 
    # Set solver
    pyo_solver = SolverFactory(solver)
    results = pyo_solver.solve(model, **(solver_options or {}))

    acceptable = {
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.feasible,
        TerminationCondition.maxIterations,
        TerminationCondition.maxTimeLimit,
    }

    tc = results.solver.termination_condition
    mse = np.nan
    if tc in acceptable:
        if tc != TerminationCondition.optimal:
            max_ctr_viol, max_bnd_viol = _max_infeasibility(model)
            if max(max_ctr_viol, max_bnd_viol) > feas_tol:
                raise RuntimeError(
                    f"Solver terminated with '{tc}' but the incumbent is "
                    f"infeasible: max constraint violation = {max_ctr_viol:.3e}, "
                    f"max bound violation = {max_bnd_viol:.3e} "
                    f"(feas_tol = {feas_tol:.3e})."
                )
            warnings.warn(
                f"Solver did not reach optimality (termination: {tc}); "
                f"returning best incumbent solution.",
                RuntimeWarning,
            )
        controls_x = np.array([model.x[i]() for i in model.N])
        controls_z = np.array([model.z[i]() for i in model.N])
        mse = model.mse()
        bezierv.update_bezierv(controls_x, controls_z)
    else:
        raise RuntimeError(
            f"Solver failed (status={results.solver.status}, termination={tc}); "
            f"bezierv unchanged."
        )

    return bezierv, mse