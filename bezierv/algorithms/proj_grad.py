import numpy as np
from bezierv.classes.bezierv import Bezierv
from bezierv.algorithms.isotonic_reg import project

def compute_bernstein_basis(n: int, t: np.array, bezierv: Bezierv) -> np.array:
    """
    Helper function to precompute the Bernstein basis matrix.
    
    Returns
    -------
    np.array
        Shape (m, n+1), where element [j, i] is the i-th Bernstein 
        basis polynomial evaluated at t[j].
    """
    i_vals = np.arange(n + 1)
    t_col = t[:, np.newaxis]
    term1 = np.power(t_col, i_vals)
    term2 = np.power(1 - t_col, n - i_vals)
    B = bezierv.comb * term1 * term2
    return B

def grad(n: int, 
         m: int, 
         t: np.array, 
         bezierv: Bezierv, 
         controls_z: np.array, 
         emp_cdf_data: np.array,
         basis_matrix: np.array = None) -> np.array:
    """
    Compute the gradient of the fitting objective with respect to the z control points.
    
    Optimized to use matrix multiplication. Accepts an optional precomputed
    basis_matrix for performance in iterative loops.
    """
    if basis_matrix is None:
        B = compute_bernstein_basis(n, t, bezierv)
    else:
        B = basis_matrix

    predictions = B @ controls_z
    residuals = predictions - emp_cdf_data
    grad_z = 2 * (B.T @ residuals)
    
    return grad_z

def project_z(controls_z: np.array) -> np.array:
    """
    Project the z control points onto the feasible set.
    """
    # Assuming project handles numpy arrays efficiently
    z_prime = project(controls_z, lower=0.0, upper=1.0)
    return z_prime

def fit(n: int, 
        m: int, 
        data: np.array,
        bezierv: Bezierv,
        init_x: np.array,
        init_z: np.array,
        t: np.array,
        emp_cdf_data: np.array, 
        step_size: float, 
        max_iter: int,
        threshold: float) -> tuple:
    """
    Fit the Bezier random variable to the empirical CDF data using projected gradient descent.
    """

    B = compute_bernstein_basis(n, t, bezierv)
    
    z = init_z.copy()
    
    for i in range(max_iter):
        grad_z = grad(n, m, t, bezierv, z, emp_cdf_data, basis_matrix=B)
        z_step = z - step_size * grad_z
        z_prime = project_z(z_step)
        if np.linalg.norm(z_prime - z) < threshold:
            z = z_prime
            break
        z = z_prime
    
    final_predictions = B @ z
    mse = np.mean((final_predictions - emp_cdf_data)**2)
    bezierv.update_bezierv(init_x, z)
    return bezierv, mse