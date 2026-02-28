import numpy as np
from bezierv.classes.bezierv import Bezierv

def compute_bernstein_basis_deltas(t: np.ndarray, bezierv: Bezierv) -> np.ndarray:
    """
    Returns B_deltas of shape (m, n) where
    B_deltas[j, i] = comb_minus[i] * t[j]^i * (1-t[j])^(n-1-i)
    """
    t = np.asarray(t)
    combm = np.asarray(bezierv.comb_minus)
    n = combm.shape[0]
    i_vals = np.arange(n)
    t_col = t[:, np.newaxis]
    term1 = np.power(t_col, i_vals)
    term2 = np.power(1.0 - t_col, (n - 1) - i_vals)
    B = combm[np.newaxis, :] * term1 * term2
    return B

def backtracking_line_search(w: np.ndarray, 
                             dw: np.ndarray, 
                             grad_F: np.ndarray, 
                             objective_f: callable, 
                             t_bar: float, 
                             B_d: np.ndarray, 
                             reg_eps: float) -> tuple:
    alpha = 1.0
    neg_idx = dw < 0
    if np.any(neg_idx):
        alpha = min(alpha, 0.99 * np.min(-w[neg_idx] / dw[neg_idx]))
    beta = 0.5
    c = 1e-4
    curr_F = t_bar * objective_f(w) + (-np.sum(np.log(w)))
    while alpha > 1e-16:
        w_new = w + alpha * dw
        if np.any(w_new <= 0):
            alpha *= beta
            continue
        s_new = B_d @ w_new + reg_eps
        if np.any(s_new <= 0):
            alpha *= beta
            continue
        F_new = t_bar * objective_f(w_new) + (-np.sum(np.log(w_new)))
        if F_new <= curr_F + c * alpha * (grad_F @ dw):
            w = w_new
            break
        else:
            alpha *= beta
    return w, alpha

def newton(w :np.ndarray, 
           B_d: np.ndarray,
           t_bar: float,
           n: int,
           objective_f: callable,
           max_newton: int,
           newton_tol: float,
           reg_eps: float) -> np.ndarray:
    
    for newton_iter in range(max_newton):
        s = B_d @ w + reg_eps
        inv_s = 1.0 / s
        inv_s2 = inv_s * inv_s

        grad_f = - (B_d.T @ inv_s)          # (n,)
        WB = (inv_s2[:, None] * B_d)        # (m,n)
        H_f = B_d.T @ WB                    # (n,n)

        grad_phi = -1.0 / w
        H_phi = np.diag(1.0 / (w * w))

        grad_F = t_bar * grad_f + grad_phi
        H = t_bar * H_f + H_phi

        # KKT
        rhs = np.concatenate([-grad_F, [-(np.sum(w) - 1.0)]])
        KKT = np.zeros((n + 1, n + 1))
        KKT[:n, :n] = H
        KKT[:n, n] = 1.0
        KKT[n, :n] = 1.0

        try:
            sol = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            KKT[:n, :n] += 1e-8 * np.eye(n)
            sol = np.linalg.solve(KKT, rhs)

        dw = sol[:n]

        newton_decrement = -grad_F @ dw
        if newton_decrement / 2.0 <= newton_tol:
            break

        w, alpha = backtracking_line_search(w, dw, grad_F, objective_f, t_bar, B_d, reg_eps)

        if alpha <= 1e-16:
            break

    return w

def fit(n: int,
        m: int,
        data: np.ndarray,
        bezierv: Bezierv,
        init_x: np.ndarray,
        init_z: np.ndarray,
        t: np.ndarray,
        emp_cdf_data: np.ndarray,
        t0: float = 1.0,
        mu: float = 10.0,
        reg_eps: float = 1e-12,
        newton_tol: float = 1e-8,
        max_newton: int = 100,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False):
    """
    Interior-point solver operating on x = deltas_z (length n) where bezierv.n == degree.
    - bezierv.controls_z has length n+1.
    - returns updated bezierv and final objective value.
    """
    # degree n = bezierv.n, #deltas = n, #controls = n+1
    if n != bezierv.n:
        raise ValueError("n must match bezierv.n.")
    if m != len(t):
        raise ValueError("m must match len(t).")
    assert len(bezierv.controls_z) == n + 1

    B_d = compute_bernstein_basis_deltas(t, bezierv)  # (m, n)
    if B_d.shape[1] != n:
        raise RuntimeError("Unexpected deltas basis shape mismatch.")

    if init_z is None:
        w = np.ones(n) / n
    else:
        z = np.asarray(init_z, dtype=float)
        if z.shape[0] != n + 1:
            raise ValueError("init_z must have length n+1 matching bezierv.")
        w = z[1:] - z[:-1]
        w = np.maximum(w, 1e-12)
        w = w / np.sum(w)

    def objective_f(x_vec):
        s = B_d @ x_vec
        s = np.maximum(s, reg_eps)
        return -np.sum(np.log(s))

    t_bar = t0
    nu = n

    for iteration in range(max_iter):
        w = newton(w, B_d, t_bar, n, objective_f, max_newton, newton_tol, reg_eps)

        # stopping criterion
        if (nu / t_bar) <= tol:
            break

        t_bar *= mu

    z_out = np.empty(n + 1, dtype=float)
    z_out[0] = 0.0
    np.cumsum(w, out=z_out[1:])
    z_out[-1] = 1.0

    bezierv.update_bezierv(init_x, z_out)
    final_obj = objective_f(w)

    return bezierv, final_obj