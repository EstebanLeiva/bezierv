import numpy as np
from bezierv.classes.bezierv import Bezierv
from scipy.optimize import minimize
import bezierv.algorithms.utils as utils

def objective_function(concatenated: np.array, 
                       n: int, 
                       m: int,
                       data:np.array,
                       bezierv: Bezierv,
                       emp_cdf_data: np.array) -> float:
    """
    Compute the objective function value for the given control points.

    This method calculates the sum of squared errors between the Bezier random variable's CDF
    and the empirical CDF data.

    Parameters
    ----------
    concatenated : np.array
        A concatenated array containing the control points for z and x coordinates.
        The first n+1 elements are the z control points, and the remaining elements are the x control points.
    n : int
        The number of control points minus one for the Bezier curve.
    m : int
        The number of empirical CDF data points.
    data : np.array
        The sorted data points used to fit the Bezier distribution.
    bezierv : Bezierv
        An instance of the Bezierv class representing the Bezier random variable.
    emp_cdf_data : np.array
        The empirical CDF data points used for fitting.

    Returns
    -------
    float
        The value of the objective function (MSE).
    """
    x = concatenated[0 : n + 1]
    z = concatenated[n + 1:]
    t = utils.get_t(n, m, data, bezierv, x)
    se = 0
    for j in range(m):
        se += (bezierv.poly_z(t[j], z) - emp_cdf_data[j])**2
    return se / m

def objective_function_lagrangian(concatenated: np.array,
                                  n: int,
                                  m: int,
                                  data: np.array, 
                                  bezierv: Bezierv,
                                  emp_cdf_data: np.array,
                                  penalty_weight: float=1e3) -> float:
    """
    Compute the objective function value for the given control points.

    This method calculates the sum of squared errors between the Bezier random variable's CDF
    and the empirical CDF data.

    Parameters
    ----------
    concatenated : np.array
        A concatenated array containing the control points for z and x coordinates.
        The first n+1 elements are the z control points, and the remaining elements are the x control points.
    n : int
        The number of control points minus one for the Bezier curve.
    m : int
        The number of empirical CDF data points.
    data : np.array
        The sorted data points used to fit the Bezier distribution.
    bezierv : Bezierv
        An instance of the Bezierv class representing the Bezier random variable.
    emp_cdf_data : np.array
        The empirical CDF data points used for fitting.
    penalty_weight : float, optional
        The weight for the penalty term in the objective function (default is 1e3).

    Returns
    -------
    float
        The value of the objective function + penalty (MSE + penalty).
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
        data: np.array,
        bezierv: Bezierv,
        init_x: np.array,
        init_z: np.array,
        emp_cdf_data: np.array,
        max_iter: int
        ) -> Bezierv:
    """
    Fit the Bezier random variable to the empirical CDF data using the Nelder-Mead optimization algorithm.

    Parameters
    ----------
    n : int
        The number of control points minus one for the Bezier curve.
    m : int
        The number of empirical CDF data points.
    data : np.array
        The sorted data points used to fit the Bezier distribution.
    bezierv : Bezierv
        An instance of the Bezierv class representing the Bezier random variable.
    init_x : np.array
        Initial guess for the x-coordinates of the control points.
    init_z : np.array
        Initial guess for the z-coordinates of the control points.
    emp_cdf_data : np.array
        The empirical CDF data points used for fitting.
    
    Returns
    -------
    Bezierv
        The fitted Bezierv object with updated control points.
    float
        The mean squared error (MSE) of the fit.
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
    bezierv.update_bezierv(controls_x, controls_z, (data[0], data[-1]))
    mse = objective_function(sol, n, m, data, bezierv, emp_cdf_data)
    return bezierv, mse