import numpy as np
from bezierv.classes.bezierv import Bezierv
from scipy.optimize import minimize

def objective_function(distfit: DistFit, concatenated: np.array) -> float:
    """
    Compute the objective function value for the given control points.

    This method calculates the sum of squared errors between the Bezier random variable's CDF
    and the empirical CDF data.

    Parameters
    ----------
    distfit : DistFit
        An instance of the DistFit class containing the Bezierv object and empirical CDF data.
    concatenated : np.array
        A concatenated array containing the control points for z and x coordinates.
        The first n+1 elements are the z control points, and the remaining elements are the x control points.

    Returns
    -------
    float
        The value of the objective function (MSE).
    """
    x = concatenated[0 : distfit.n + 1]
    z = concatenated[distfit.n + 1:]
    t = distfit.get_t(x, distfit.data)
    se = 0
    for j in range(distfit.m):
        se += (distfit.bezierv.poly_z(t[j], z) - distfit.emp_cdf_data[j])**2
    return se / distfit.m

def objective_function_lagrangian(distfit:DistFit, 
                                  concatenated: np.array, 
                                  penalty_weight: float=1e3) -> float:
    """
    Compute the objective function value for the given control points.

    This method calculates the sum of squared errors between the Bezier random variable's CDF
    and the empirical CDF data.

    Parameters
    ----------
    distfit : DistFit
        An instance of the DistFit class containing the Bezierv object and empirical CDF data.
    concatenated : np.array
        A concatenated array containing the control points for z and x coordinates.
        The first n+1 elements are the z control points, and the remaining elements are the x control points.

    Returns
    -------
    float
        The value of the objective function (MSE).
    """
    
    x = concatenated[0 : distfit.n + 1]
    z = concatenated[distfit.n + 1 : ]

    try:
        t = distfit.get_t(x, distfit.data)
    except ValueError as e:
        return np.inf
    
    se = 0
    for j in range(distfit.m):
        se += (distfit.bezierv.poly_z(t[j], z) - distfit.emp_cdf_data[j])**2
    mse = se / distfit.m

    penalty = 0.0
    penalty += abs(z[0] - 0.0)
    penalty += abs(z[-1] - 1.0)
    delta_zs = np.diff(z)
    delta_xs = np.diff(x)
    penalty += np.sum(abs(np.minimum(0, delta_zs)))
    penalty += np.sum(abs(np.minimum(0, delta_xs)))
    penalty += abs(x[0] - distfit.data[0])
    penalty += abs(distfit.data[-1] - x[-1])

    return mse + penalty_weight * penalty

def fit(distfit: DistFit, controls_x0: np.array, controls_z0: np.array) -> Bezierv:
    """
    Fit the Bezier random variable to the empirical CDF data using the Nelder-Mead optimization algorithm.

    Parameters
    ----------
    distfit : DistFit
        An instance of the DistFit class containing the Bezierv object and empirical CDF data.
    controls_x0 : np.array
        Initial guess for the x-coordinates of the control points.
    controls_z0 : np.array
        Initial guess for the z-coordinates of the control points.

    Returns
    -------
    Bezierv
        The fitted Bezierv object with updated control points.
    """
    start = np.concatenate((controls_x0, controls_z0))
    result = minimize(
        fun=objective_function_lagrangian,
        x0=start,
        method='Nelder-Mead')
    sol = result.x
    controls_x = sol[0:distfit.n + 1]
    controls_z = sol[distfit.n + 1:]
    distfit.bezierv.update_bezierv(controls_x, controls_z, (distfit.data[0], distfit.data[-1]))
    distfit.mse = distfit.objective_function(sol)
    return distfit.bezierv