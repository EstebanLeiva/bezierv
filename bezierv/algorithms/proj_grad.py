import numpy as np
from bezierv.classes.bezierv import Bezierv
from bezierv.algorithms.isotonic_reg import project

def compute_bernstein_basis(n: int,
                            t: np.ndarray,
                            bezierv: Bezierv) -> np.ndarray:
    """
    Precompute the degree-``n`` Bernstein basis matrix for the Bézier CDF.

    Evaluates all ``n + 1`` degree-``n`` Bernstein basis polynomials at each
    parameter value in ``t``. The resulting matrix ``B`` satisfies
    ``B @ z = F̂`` in the minimum-error objective (Section 3 of the paper).
    Unlike the basis in ``primal_grad``, this uses degree ``n`` (for the CDF)
    rather than degree ``n - 1`` (for the PDF).

    Parameters
    ----------
    n : int
        Degree of the Bézier curve (``n + 1`` control points).
    t : numpy.ndarray, shape (m,)
        Parameter values in ``[0, 1]`` at which to evaluate the basis.
    bezierv : Bezierv
        Bézier random variable providing the precomputed binomial
        coefficients ``comb``.

    Returns
    -------
    numpy.ndarray, shape (m, n + 1)
        Basis matrix where ``B[j, i]`` is the ``i``-th degree-``n``
        Bernstein basis polynomial evaluated at ``t[j]``.
    """
    i_vals = np.arange(n + 1)
    t_col = t[:, np.newaxis]
    term1 = np.power(t_col, i_vals)
    term2 = np.power(1 - t_col, n - i_vals)
    B = bezierv.comb * term1 * term2
    return B

def grad(n: int,
         t: np.ndarray,
         bezierv: Bezierv,
         controls_z: np.ndarray,
         emp_cdf_data: np.ndarray,
         basis_matrix: np.ndarray = None) -> np.ndarray:
    """
    Compute the gradient of the MSE objective w.r.t. the z control points.

    Evaluates ``∇_z ‖Bz - F̂‖² = 2 Bᵀ(Bz - F̂)`` where ``B`` is the
    Bernstein basis matrix and ``F̂`` are the empirical CDF values
    (Section 3 of the paper).

    Parameters
    ----------
    n : int
        Degree of the Bézier curve.
    t : numpy.ndarray, shape (m,)
        Parameter values in ``[0, 1]`` for each observation.
    bezierv : Bezierv
        Bézier random variable providing the Bernstein basis evaluator.
    controls_z : numpy.ndarray, shape (n + 1,)
        Current z-coordinates of the control points.
    emp_cdf_data : numpy.ndarray, shape (m,)
        Empirical CDF values at each observation.
    basis_matrix : numpy.ndarray, shape (m, n + 1), optional
        Precomputed basis matrix from :func:`compute_bernstein_basis`.
        Computed on the fly if not provided.

    Returns
    -------
    numpy.ndarray, shape (n + 1,)
        Gradient ``2 Bᵀ(Bz - F̂)``.
    """
    if basis_matrix is None:
        B = compute_bernstein_basis(n, t, bezierv)
    else:
        B = basis_matrix

    predictions = B @ controls_z
    residuals = predictions - emp_cdf_data
    grad_z = 2 * (B.T @ residuals)
    
    return grad_z

def project_z(controls_z: np.ndarray) -> np.ndarray:
    """
    Project the z control points onto the bounded monotone feasible set.

    Delegates to :func:`~bezierv.algorithms.isotonic_reg.project` with
    boundary conditions ``z₀ = 0`` and ``zₙ = 1`` (CDF constraints from
    Section 3 of the paper).

    Parameters
    ----------
    controls_z : numpy.ndarray, shape (n + 1,)
        Current z-coordinates of the control points.

    Returns
    -------
    numpy.ndarray, shape (n + 1,)
        Projected control points satisfying ``0 = z[0] <= ... <= z[n] = 1``.
    """
    z_prime = project(controls_z, lower=0.0, upper=1.0)
    return z_prime

def fit(n: int,
        bezierv: Bezierv,
        init_x: np.ndarray,
        init_z: np.ndarray,
        t: np.ndarray,
        emp_cdf_data: np.ndarray,
        step_size: float,
        max_iter: int,
        threshold: float) -> tuple[Bezierv, float]:
    """
    Fit a Bézier distribution via projected gradient descent on the MSE objective.

    Minimizes ``(1/m) ‖Bz - F̂‖²`` over the bounded monotone feasible set
    by iterating a gradient step followed by projection via :func:`project_z`.
    Implements Algorithm 1 from Section 3 of the paper.

    Parameters
    ----------
    n : int
        Degree of the Bézier curve (``n + 1`` control points).
    bezierv : Bezierv
        Bézier random variable to be updated with the fitted parameters.
    init_x : numpy.ndarray, shape (n + 1,)
        Fixed x-coordinates of the control points (support knots).
    init_z : numpy.ndarray, shape (n + 1,)
        Initial z-coordinates of the control points.
    t : numpy.ndarray, shape (m,)
        Pre-computed parameter values for each observation, from
        :func:`~bezierv.algorithms.utils.get_t`.
    emp_cdf_data : numpy.ndarray, shape (m,)
        Empirical CDF values at each observation.
    step_size : float
        Gradient descent step size ``η``.
    max_iter : int
        Maximum number of projected gradient iterations.
    threshold : float
        Convergence tolerance on ``‖z_{k+1} - z_k‖``.

    Returns
    -------
    bezierv : Bezierv
        Updated Bézier random variable with fitted control points.
    mse : float
        Final mean squared error between the fitted CDF and the empirical CDF.
    """

    B = compute_bernstein_basis(n, t, bezierv)
    z = init_z.copy()
    
    for i in range(max_iter):
        grad_z = grad(n, t, bezierv, z, emp_cdf_data, basis_matrix=B)
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