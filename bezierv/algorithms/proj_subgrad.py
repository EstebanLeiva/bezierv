import numpy as np
from bezierv.classes.bezierv import Bezierv
from bezierv.algorithms import utils as utils

def subgrad(n: int,
            m: int,
            bezierv: Bezierv,
            t: float, 
            controls_z: np.array,
            emp_cdf_data: np.array) -> tuple:
    """
    Compute the subgradient of the fitting objective with respect to the (x, z) control points.

    Parameters
    ----------
    n : int
        The number of control points minus one.
    m : int
        The number of empirical CDF data points.
    bezierv : Bezierv
        An instance of the Bezierv class representing the Bezier random variable.
    t : np.array
        The parameter values corresponding to the data points. Expected to be an array of shape (m,).
    controls_z : np.array
        The current z-coordinates of the control points.
    emp_cdf_data : np.array
        The empirical CDF data points used for fitting.

    Returns
    -------
    np.array, np.array
        A tuple containing the subgradient with respect to the x control points and the gradient with respect to the z control points.
    """
    grad_z = np.zeros(n + 1)
    subgrad_x = np.zeros(n + 1)
    for j in range(m):
        inner_sum = np.zeros(n + 1)
        for i in range(n + 1):
            inner_sum[i] = bezierv.bernstein(t[j], i, bezierv.comb, n)

        subgrad_x += 2 * (bezierv.poly_z(t[j], controls_z) - emp_cdf_data[j]) * (1 / m) * inner_sum
        grad_z += 2 * (bezierv.poly_z(t[j], controls_z) - emp_cdf_data[j]) * inner_sum
    
    grad_z = np.zeros(n + 1)
    for j in range(m):
        inner_sum = np.zeros(n + 1)
        for i in range(n + 1):
            inner_sum[i] = bezierv.bernstein(t[j], i, bezierv.comb, n)

        grad_z += 2 * (bezierv.poly_z(t[j], controls_z) - emp_cdf_data[j]) * inner_sum

    return subgrad_x, grad_z

def project_x(data:np.array,
              controls_x: np.array):
    """
    Project the x control points onto the feasible set.

    The projection is performed by first clipping the control points to the [X_(1), X_(m)] interval,
    sorting them in ascending order, and then enforcing the boundary conditions by setting
    the first control point to X_(1) and the last control point to X_(m).

    Parameters
    ----------
    controls_x : np.array
        The current z-coordinates of the control points.

    Returns
    -------
    np.array
        The projected z control points that satisfy the constraints.
    """
    x_prime = np.clip(controls_x, a_min= data[0], a_max=data[-1])
    x_prime.sort()
    x_prime[0] = data[0]
    x_prime[-1] = data[-1]
    return x_prime

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

def objective_function(m: int,
                       bezierv: Bezierv,
                       emp_cdf_data: np.array,
                       z: np.array, 
                       t: np.array):
    """
    Compute the objective function value for the given z control points.

    This method calculates the sum of squared errors between the Bezier random variable's CDF
    and the empirical CDF data.

    Parameters
    ----------
    z : np.array
        The z-coordinates of the control points.
    t : np.array
        The parameter values corresponding to the data points.

    Returns
    -------
    float
        The value of the objective function (MSE).
    """
    se = 0
    for j in range(m):
        se += (bezierv.poly_z(t[j], z) - emp_cdf_data[j])**2
    return se / m

def fit(n: int, 
        m: int, 
        data: np.array,
        bezierv: Bezierv,
        init_x: np.array,
        init_z: np.array,
        init_t: float,
        emp_cdf_data: np.array, 
        step_size_x: float,
        step_size_z: float,
        max_iter: int,):
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
    controls_z0 : np.array
        The initial guess for the z control points.
    step : float, optional
        The step size (learning rate) for the gradient descent updates (default is 0.01).
    maxiter : int, optional
        The maximum number of iterations allowed for the fitting process (default is 1000).

    Returns
    -------
    Bezierv
        The updated Bezierv instance with fitted control points.
    float
        The mean squared error (MSE) of the fit.
    """
    f_best = np.inf
    x_best = None
    z_best = None
    x = init_x
    z = init_z
    t = init_t
    for i in range(max_iter):
        subgrad_x, grad_z = subgrad(n, m, bezierv, t, z, emp_cdf_data)
        z_prime = project_z(z - step_size_z * grad_z)
        x_prime = project_x(data, x - step_size_x * subgrad_x)
        t_prime = utils.get_t(n, m, data, bezierv, x_prime)
        mse_prime = objective_function(m, bezierv, emp_cdf_data, z_prime, t_prime)
        if mse_prime < f_best:
            f_best = mse_prime
            x_best = x_prime
            z_best = z_prime
        x = x_prime
        z = z_prime
        t = t_prime

    bezierv.update_bezierv(x_best, z_best, (data[0], data[-1]))

    return bezierv, f_best