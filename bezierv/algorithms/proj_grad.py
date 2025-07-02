import numpy as np
from bezierv.classes.bezierv import Bezierv

def grad(n: int, 
         m:int, 
         t: float, 
         bezierv: Bezierv, 
         controls_z: np.array, 
         emp_cdf_data: np.array) -> np.array:
    """
    Compute the gradient of the fitting objective with respect to the z control points.

    Parameters
    ----------
    n : int
        The number of control points minus one.
    m : int
        The number of empirical CDF data points.
    t : np.array
        The parameter values corresponding to the data points. Expected to be an array of shape (m,).
    bezierv : Bezierv
        An instance of the Bezierv class representing the Bezier random variable.
    controls_z : np.array
        The current z-coordinates of the control points.
    emp_cdf_data : np.array
        The empirical CDF data points used for fitting.

    Returns
    -------
    np.array
        An array representing the gradient with respect to each z control point.
    """
    grad_z = np.zeros(n + 1)
    for j in range(m):
        inner_sum = np.zeros(n + 1)
        for i in range(n + 1):
            inner_sum[i] = bezierv.bernstein(t[j], i, bezierv.comb, n)
        grad_z += 2 * (bezierv.poly_z(t[j], controls_z) - emp_cdf_data[j]) * inner_sum
    return grad_z
    
def project_z(controls_z: np.array):
    """
    Project the z control points onto the feasible set.

    The projection is performed by first clipping the control points to the [0, 1] interval,
    sorting them in ascending order, and then enforcing the boundary conditions by setting
    the first control point to 0 and the last control point to 1.

    Parameters
    ----------
    controls_z : np.array
        The current z-coordinates of the control points.

    Returns
    -------
    np.array
        The projected z control points that satisfy the constraints.
    """
    z_prime = np.clip(controls_z, a_min= 0, a_max=1)
    z_prime.sort()
    z_prime[0] = 0
    z_prime[-1] = 1
    return z_prime

def fit(n: int, 
        m: int, 
        data: np.array,
        bezierv: Bezierv,
        init_x: np.array,
        init_z: np.array,
        t: float,
        emp_cdf_data: np.array, 
        step_size: float, 
        max_iter: int,
        threshold: float) -> tuple:
    """
    Fit the Bezier random variable to the empirical CDF data using projected gradient descent.

    Starting from an initial guess for the z control points, this method iteratively
    updates the z control points by taking gradient descent steps and projecting the
    result back onto the feasible set. The process continues until convergence is
    achieved (i.e., the change in control points is less than a set threshold) or until
    the maximum number of iterations is reached. After convergence, the Bezierv curve is
    updated with the new control points and the fitting error is computed.

    Parameters
    ----------
    n : int
        The number of control points minus one.
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
    t : float or np.array
        The parameter values corresponding to the data points. Expected to be an array of shape (m,).
    emp_cdf_data : np.array
        The empirical CDF data points used for fitting.
    step_size : float
        The step size for the gradient descent updates.
    max_iter : int
        The maximum number of iterations allowed for the fitting process.
    threshold : float
        The convergence threshold for the change in control points.

    Returns
    -------
    Bezierv
        The updated Bezierv instance with fitted control points.
    float
        The mean squared error (MSE) of the fit.
    """
    z = init_z
    for i in range(max_iter):
        grad_z = grad(n, m, t, bezierv, z, emp_cdf_data)
        z_prime = project_z(z - step_size * grad_z)
        if np.linalg.norm(z_prime - z) < threshold:
            z = z_prime
            break
        z = z_prime
    
    se = 0
    for j in range(m):
        se += (bezierv.poly_z(t[j], z) - emp_cdf_data[j])**2
    bezierv.update_bezierv(init_x, z, (data[0], data[-1]))
    mse = se/m
    return bezierv, mse