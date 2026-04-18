import numpy as np
from bezierv.classes.bezierv import Bezierv
from scipy.optimize import minimize
import bezierv.algorithms.utils as utils

def objective_function(concatenated: np.ndarray,
                       n: int,
                       m: int,
                       data: np.ndarray,
                       bezierv: Bezierv,
                       emp_cdf_data: np.ndarray) -> float:
    """
    Compute the MSE between the fitted Bézier CDF and the empirical CDF.

    Evaluates the minimum-error objective ``(1/m) ∑ⱼ (F(xⱼ) - F̂ⱼ)²``
    where ``F`` is the Bézier CDF parameterized by the given control points
    and ``F̂ⱼ`` are the empirical CDF values (Section 3 of the paper).

    Parameters
    ----------
    concatenated : numpy.ndarray, shape (2*(n + 1),)
        Concatenated control point vector ``[x₀, …, xₙ, z₀, …, zₙ]``.
    n : int
        Degree of the Bézier curve (``n + 1`` control points).
    m : int
        Sample size.
    data : numpy.ndarray, shape (m,)
        Sorted observed sample values.
    bezierv : Bezierv
        Bézier random variable providing the CDF evaluator ``poly_z``.
    emp_cdf_data : numpy.ndarray, shape (m,)
        Empirical CDF values at each observation.

    Returns
    -------
    float
        Mean squared error between the fitted CDF and the empirical CDF.
    """
    x = concatenated[0 : n + 1]
    z = concatenated[n + 1:]
    t = utils.get_t(n, m, data, bezierv, x)
    se = 0
    for j in range(m):
        se += (bezierv.poly_z(t[j], z) - emp_cdf_data[j])**2
    return se / m

def objective_function_lagrangian(concatenated: np.ndarray,
                                  n: int,
                                  m: int,
                                  data: np.ndarray,
                                  bezierv: Bezierv,
                                  emp_cdf_data: np.ndarray,
                                  penalty_weight: float = 1e3) -> float:
    """
    Compute the penalized MSE objective for unconstrained Nelder-Mead optimization.

    Augments :func:`objective_function` with a Lagrangian penalty that
    enforces the Bézier feasibility constraints — boundary conditions
    ``z₀ = 0``, ``zₙ = 1``, monotonicity of ``z`` and ``x``, and support
    matching ``x₀ = data[0]``, ``xₙ = data[-1]`` — since Nelder-Mead does
    not support constraints natively (Section 3 of the paper, Wagner
    baseline method).

    Parameters
    ----------
    concatenated : numpy.ndarray, shape (2*(n + 1),)
        Concatenated control point vector ``[x₀, …, xₙ, z₀, …, zₙ]``.
    n : int
        Degree of the Bézier curve (``n + 1`` control points).
    m : int
        Sample size.
    data : numpy.ndarray, shape (m,)
        Sorted observed sample values.
    bezierv : Bezierv
        Bézier random variable providing the CDF evaluator ``poly_z``.
    emp_cdf_data : numpy.ndarray, shape (m,)
        Empirical CDF values at each observation.
    penalty_weight : float, optional
        Scaling factor for the constraint violation penalty. Default is
        ``1e3``.

    Returns
    -------
    float
        Penalized objective ``MSE + penalty_weight * constraint_violation``.
        Returns ``inf`` if root-finding fails for the given ``x`` knots.
    """
    
    x = concatenated[0 : n + 1]
    z = concatenated[n + 1 : ]

    try:
        t = utils.get_t(n, m, data, bezierv, x)
    except ValueError as e:
        return np.inf
    
    se = 0
    for j in range(m):
        se += (bezierv.poly_z(t[j], z) - emp_cdf_data[j])**2
    mse = se / m

    penalty = 0.0
    penalty += abs(z[0] - 0.0)
    penalty += abs(z[-1] - 1.0)
    delta_zs = np.diff(z)
    delta_xs = np.diff(x)
    penalty += np.sum(abs(np.minimum(0, delta_zs)))
    penalty += np.sum(abs(np.minimum(0, delta_xs)))
    penalty += abs(x[0] - data[0])
    penalty += abs(data[-1] - x[-1])

    return mse + penalty_weight * penalty

def fit(n: int,
        m: int,
        data: np.ndarray,
        bezierv: Bezierv,
        init_x: np.ndarray,
        init_z: np.ndarray,
        emp_cdf_data: np.ndarray,
        max_iter: int) -> tuple[Bezierv, float]:
    """
    Fit a Bézier distribution using the Nelder-Mead derivative-free method.

    Minimizes :func:`objective_function_lagrangian` via ``scipy.optimize.minimize``
    with ``method='Nelder-Mead'``. This is the derivative-free baseline
    (Wagner's PRIME approach) benchmarked against the first-order methods
    in Section 7 of the paper.

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
    emp_cdf_data : numpy.ndarray, shape (m,)
        Empirical CDF values at each observation.
    max_iter : int
        Maximum number of Nelder-Mead iterations.

    Returns
    -------
    bezierv : Bezierv
        Updated Bézier random variable with fitted control points.
    mse : float
        Final mean squared error of the fit, evaluated via
        :func:`objective_function`.
    """
    start = np.concatenate((init_x, init_z))
    result = minimize(
        fun=objective_function_lagrangian,
        args=(n, m, data, bezierv, emp_cdf_data),
        x0=start,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'disp': False})
    sol = result.x
    controls_x = sol[0 : n + 1]
    controls_z = sol[n + 1: ]
    bezierv.update_bezierv(controls_x, controls_z)
    mse = objective_function(sol, n, m, data, bezierv, emp_cdf_data)
    return bezierv, mse