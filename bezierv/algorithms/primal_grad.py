import numpy as np
from bezierv.classes.bezierv import Bezierv

def compute_bernstein_basis(n: int, 
                            t: np.array, 
                            bezierv: Bezierv) -> np.array:
    """
    Helper function to precompute the Bernstein basis matrix.
    
    Returns
    -------
    np.array
        Shape (m, n), where element [j, i] is the i-th (n-1) Bernstein 
        basis polynomial evaluated at t[j].
    """
    i_vals = np.arange(n)
    t_col = t[:, np.newaxis]
    term1 = np.power(t_col, i_vals)
    term2 = np.power(1 - t_col, (n - 1) - i_vals)
    A = bezierv.comb_minus * term1 * term2
    return A


def grad(A: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute gradient g = nabla l(w) for l(w) = - sum_j ln(a_j^T w).
    """
    s = A @ w                       
    if np.any(s <= 0):
        raise ValueError("Some a_j^T w <= 0.")
    inv_s = 1.0 / s                 
    g = - (A.T @ inv_s)             
    return g

def compute_alpha(g: np.ndarray, m_param: float, w: np.ndarray) -> np.ndarray:
    return g + (m_param / w)

def solve_lambda_safeguarded(alpha: np.ndarray,
                             n: int,
                             m: int,
                             tol_lambda: float = 1e-12,
                             tol_res: float = 1e-12,
                             max_iters: int = 100) -> float:
    """
    Solve phi(lambda) = sum_i m / (alpha_i + lambda) = 1 for lambda > -min(alpha).
    Returns lambda^*.
    """
    def phi(lmbd):
        return np.sum(m / (alpha + lmbd))
    
    s = np.min(alpha)
    eps = 1e-6
    a = -s + eps
    phi_a = phi(a)
    iter_a = 0
    while not (phi_a > 1) and iter_a < 10:
        a = -s + eps * 1e-1
        phi_a = phi(a) 

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
    alpha = compute_alpha(g, m, w)
    lam_star = solve_lambda_safeguarded(alpha, n, m, tol_res=tol_res_root, tol_lambda=tol_lambda_root, max_iters=max_iters_root)
    w_next = m / (alpha + lam_star)
    # numerical safety: renormalize to sum to 1 and clip positives
    w_next = np.maximum(w_next, 0.0)
    w_next = w_next / np.sum(w_next)
    return w_next

def fit(n: int,
        m: int,
        data: np.array,
        bezierv: Bezierv,
        init_x: np.array,
        init_w: np.array,
        t: np.array,
        emp_cdf_data: np.array,
        max_iter: int,
        tol: float,
        tol_res_root: float,
        tol_lambda_root: float,
        max_iters_root: int) -> tuple:
    """
    Run primal gradient scheme to minimize l(w) = -sum_j ln(a_j^T w) subject to w in simplex.
    Returns final w, optional mse (if emp_target and compute_pred_from_w provided), and lam history.
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