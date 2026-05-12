import numpy as np
from bezierv.classes.bezierv import Bezierv

def compute_bernstein_basis(n: int,
                            t: np.ndarray,
                            bezierv: Bezierv) -> np.ndarray:
    """
    Precompute the Bernstein basis matrix for the Bézier density.

    Evaluates all ``n`` degree-``(n-1)`` Bernstein basis polynomials at each
    parameter value in ``t``. The resulting matrix ``A`` is used by the
    primal gradient algorithm so that ``A @ w`` gives the density evaluated
    at each ``t[j]`` (see Section 4 of the paper).

    Parameters
    ----------
    n : int
        Number of basis functions (equal to the number of Bézier weights
        ``w``). The polynomial degree is ``n - 1``.
    t : numpy.ndarray, shape (m,)
        Parameter values in ``[0, 1]`` at which to evaluate the basis.
    bezierv : Bezierv
        Fitted Bézier random variable providing the precomputed binomial
        coefficients ``comb_minus``.

    Returns
    -------
    numpy.ndarray, shape (m, n)
        Basis matrix where ``A[j, i]`` is the ``i``-th degree-``(n-1)``
        Bernstein basis polynomial evaluated at ``t[j]``.
    """
    i_vals = np.arange(n)
    t_col = t[:, np.newaxis]
    term1 = np.power(t_col, i_vals)
    term2 = np.power(1 - t_col, (n - 1) - i_vals)
    A = bezierv.comb_minus * term1 * term2
    return A


def grad(A: np.ndarray, 
         w: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the negative log-likelihood.

    Evaluates ``∇l(w) = -Aᵀ (Aw)⁻¹`` where ``l(w) = -∑ⱼ ln(aⱼᵀ w)``
    is the negative log-likelihood of the Bézier density (Section 4 of
    the paper).

    Parameters
    ----------
    A : numpy.ndarray, shape (m, n)
        Bernstein basis matrix from :func:`compute_bernstein_basis`.
    w : numpy.ndarray, shape (n,)
        Current weight vector on the probability simplex.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Gradient ``∇l(w)``.

    Raises
    ------
    ValueError
        If any density value ``(Aw)[j] <= 0``, indicating ``w`` is outside
        the feasible region.
    """
    s = A @ w                       
    if np.any(s <= 0):
        raise ValueError("Some a_j^T w <= 0.")
    inv_s = 1.0 / s                 
    g = - (A.T @ inv_s)             
    return g

def compute_alpha(g: np.ndarray, m_param: float, w: np.ndarray) -> np.ndarray:
    """
    Compute the shifted gradient vector used in the simplex projection step.

    Parameters
    ----------
    g : numpy.ndarray, shape (n,)
        Gradient ``∇l(w)`` from :func:`grad`.
    m_param : float
        Sample size ``m``.
    w : numpy.ndarray, shape (n,)
        Current weight vector.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Vector ``α = g + m / w``.
    """
    return g + (m_param / w)

def solve_lambda_safeguarded(alpha: np.ndarray,
                             n: int,
                             m: int,
                             tol_lambda: float = 1e-12,
                             tol_res: float = 1e-12,
                             max_iters: int = 100) -> float:
    """
    Solve ``φ(λ) = ∑ᵢ m / (αᵢ + λ) = 1`` for the dual variable ``λ``.

    Uses a safeguarded Newton method with bisection fallback. This is the
    dual problem arising from projecting the gradient step onto the
    probability simplex (Section 4 of the paper).

    Parameters
    ----------
    alpha : numpy.ndarray, shape (n,)
        Shifted gradient vector ``α = ∇l(w) + m / w``.
    n : int
        Number of Bézier weights (dimension of the simplex).
    m : int
        Sample size.
    tol_lambda : float, optional
        Convergence tolerance on the bracket width ``b - a``.
    tol_res : float, optional
        Convergence tolerance on the residual ``|φ(λ) - 1|``.
    max_iters : int, optional
        Maximum number of Newton iterations.

    Returns
    -------
    float
        Optimal dual variable ``λ*`` satisfying ``φ(λ*) = 1``.

    Raises
    ------
    RuntimeError
        If a valid bracket ``[a, b]`` with ``φ(a) > 1 > φ(b)`` cannot be
        found.
    """
    def phi(lmbd):
        return np.sum(m / (alpha + lmbd))
    
    s = np.min(alpha)
    eps = 1e-6
    a = -s + eps
    phi_a = phi(a)
    iter_a = 0
    while not (phi_a > 1) and iter_a < 10:
        eps *= 1e-1
        a = -s + eps
        phi_a = phi(a) 
        iter_a += 1

    delta = 1e-8
    b = m * n - s + delta
    phi_b = phi(b)
    iter_b = 0
    while not (phi_b < 1.0) and iter_b < 60:
        b = b + (m * n) * (2 ** iter_b)
        phi_b = phi(b)
        iter_b += 1
    
    if not (phi_a > 1) or not (phi_b < 1):
        raise RuntimeError("Unable to find bracket in solve_lambda_safeguarded")

    lambda0 = max(a, min(b, m * n - np.mean(alpha)))
    lam = lambda0
    for it in range(max_iters):
        denom = alpha + lam

        if np.any(denom <= 0): #fallback to bisection if we get out of bounds
            lam = 0.5 * (a + b)
            denom = alpha + lam
        phi_lam = np.sum(m / denom)
        res = phi_lam - 1.0

        if abs(res) <= tol_res or (b - a) <= tol_lambda:
            return lam
        
        grad_phi_lam = -np.sum(m / (denom ** 2))
        lam_new = lam - res / grad_phi_lam
        
        if (lam_new <= a) or (lam_new >= b) or not np.isfinite(lam_new): #fallback to bisection if newton step goes out of bounds or is not finite
            lam_new = 0.5 * (a + b)
        
        phi_new = np.sum(m / (alpha + lam_new))
        if phi_new > 1.0:
            a = lam_new
        else:
            b = lam_new
        
        lam = lam_new
    
    return 0.5 * (a + b) # Return midpoint if max iters reached without convergence


def primal_update_w(w: np.ndarray,
                    g: np.ndarray,
                    n: int,
                    m: int,
                    tol_res_root: float,
                    tol_lambda_root: float,
                    max_iters_root: int) -> np.ndarray:
    """
    Perform one primal gradient step projected onto the probability simplex.

    Computes the next iterate ``w_{k+1}`` by solving the dual problem
    ``φ(λ*) = 1`` via :func:`solve_lambda_safeguarded` and recovering
    ``w = m / (α + λ*)``. See Section 4 of the paper.

    Parameters
    ----------
    w : numpy.ndarray, shape (n,)
        Current weight vector on the probability simplex.
    g : numpy.ndarray, shape (n,)
        Gradient ``∇l(w)`` at the current iterate.
    n : int
        Number of Bézier weights.
    m : int
        Sample size.
    tol_res_root : float
        Residual tolerance passed to :func:`solve_lambda_safeguarded`.
    tol_lambda_root : float
        Bracket-width tolerance passed to :func:`solve_lambda_safeguarded`.
    max_iters_root : int
        Maximum Newton iterations for the dual solve.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Updated weight vector on the probability simplex.
    """
    alpha = compute_alpha(g, m, w)
    lam_star = solve_lambda_safeguarded(alpha, n, m, tol_res=tol_res_root, tol_lambda=tol_lambda_root, max_iters=max_iters_root)
    w_next = m / (alpha + lam_star)
    # numerical safety: renormalize to sum to 1 and clip positives
    w_next = np.maximum(w_next, 0.0)
    w_next = w_next / np.sum(w_next)
    return w_next

def fit(n: int,
        m: int,
        bezierv: Bezierv,
        init_x: np.ndarray,
        init_w: np.ndarray,
        t: np.ndarray,
        max_iter: int,
        tol: float,
        tol_res_root: float,
        tol_lambda_root: float,
        max_iters_root: int) -> tuple:
    """
    Fit a Bézier distribution via MLE using the primal gradient method.

    Minimizes ``l(w) = -∑ⱼ ln(aⱼᵀ w)`` over the probability simplex by
    iterating :func:`primal_update_w` until convergence, then updates the
    ``bezierv`` object in-place. Implements Algorithm 2 from Section 4 of
    the paper.

    Parameters
    ----------
    n : int
        Number of Bézier weights (``n + 1`` control points for the cdf).
    m : int
        Sample size.
    bezierv : Bezierv
        Bézier random variable to be updated with the fitted parameters.
    init_x : numpy.ndarray, shape (n + 1,)
        Fixed x-coordinates of the control points (support knots).
    init_w : numpy.ndarray, shape (n,)
        Initial weight vector on the probability simplex.
    t : numpy.ndarray, shape (m,)
        Pre-computed parameter values corresponding to each observation,
        from :func:`~bezierv.algorithms.utils.get_t`.
    max_iter : int
        Maximum number of primal gradient iterations.
    tol : float
        Convergence tolerance on ``‖w_{k+1} - w_k‖``.
    tol_res_root : float
        Residual tolerance for the inner dual solve.
    tol_lambda_root : float
        Bracket-width tolerance for the inner dual solve.
    max_iters_root : int
        Maximum Newton iterations for the inner dual solve.

    Returns
    -------
    bezierv : Bezierv
        Updated Bézier random variable with fitted control points.
    nll : float
        Negative log-likelihood at the fitted weights (adjusted by the
        x-spacing term ``∑ ln(aⱼᵀ Δx)``).
    """
    A = compute_bernstein_basis(n, t, bezierv)
    w = init_w.copy()

    for i in range(max_iter):
        g = grad(A, w)
        w_next = primal_update_w(w, g, n, m, tol_res_root, tol_lambda_root, max_iters_root)
        if np.linalg.norm(w_next - w) < tol:
            w = w_next
            break
        w = w_next

    delta_x = np.diff(init_x)
    nll = -np.sum(np.log(A @ w)) + np.sum(np.log(A @ delta_x))
    z = np.zeros(n + 1)
    z[0] = 0.0
    z[1:] = np.cumsum(w)
    bezierv.update_bezierv(init_x, z)
    return bezierv, nll