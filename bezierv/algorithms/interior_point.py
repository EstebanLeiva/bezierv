import numpy as np
from bezierv.classes.bezierv import Bezierv

def compute_bernstein_basis_controls(t: np.ndarray, bezierv: Bezierv) -> np.ndarray:
    """
    Returns B_controls of shape (m, n+1) where
    B_controls[j, i] = comb(n, i) * t[j]^i * (1-t[j])^(n-i)
    (Matches bezierv.controls_z which has length n+1).
    """
    t = np.asarray(t)
    m = t.shape[0]
    comb = np.asarray(bezierv.comb)           # length n+1
    k = comb.shape[0]                         # k = n+1
    i_vals = np.arange(k)                     # 0 .. n
    t_col = t[:, np.newaxis]                  # (m,1)
    term1 = np.power(t_col, i_vals)           # (m,k)
    term2 = np.power(1.0 - t_col, (k - 1) - i_vals)
    B = comb[np.newaxis, :] * term1 * term2   # (m,k)
    return B

def compute_bernstein_basis_deltas(t: np.ndarray, bezierv: Bezierv) -> np.ndarray:
    """
    Returns B_deltas of shape (m, n) where
    B_deltas[j, i] = comb_minus[i] * t[j]^i * (1-t[j])^(n-1-i)
    (Matches x = deltas_z which has length n).
    """
    t = np.asarray(t)
    m = t.shape[0]
    combm = np.asarray(bezierv.comb_minus)    # length n
    k = combm.shape[0]                        # k = n
    i_vals = np.arange(k)                     # 0 .. n-1
    t_col = t[:, np.newaxis]
    term1 = np.power(t_col, i_vals)
    term2 = np.power(1.0 - t_col, (k - 1) - i_vals)
    B = combm[np.newaxis, :] * term1 * term2  # (m,k)
    return B

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
            max_outer: int = 100,
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

    # Precompute basis for x (deltas) of shape (m, n)
    B_d = compute_bernstein_basis_deltas(t, bezierv)  # (m, n)
    if B_d.shape[1] != n:
        raise RuntimeError("Unexpected deltas basis shape mismatch.")
    # Optional: basis for controls (for plotting/validation)
    B_controls = compute_bernstein_basis_controls(t, bezierv)  # (m, n+1)

    # Initialize x from init_z (if provided) otherwise uniform
    if init_z is None:
        # uniform z increasing from 0..1 -> deltas uniform = 1/(n+1)?? careful:
        # For standard delta definition we want sum x = 1, so use uniform x = 1/n
        x = np.ones(n) / n
    else:
        z = np.asarray(init_z, dtype=float)
        if z.shape[0] != n + 1:
            raise ValueError("init_z must have length n+1 matching bezierv.")
        # compute deltas from init_z
        x = z[1:] - z[:-1]
        # small interior projection
        x = np.maximum(x, 1e-12)
        x = x / np.sum(x)

    def objective_f(x_vec):
        # s = B_d @ x  (length m)
        s = B_d @ x_vec
        s = np.maximum(s, reg_eps)
        return -np.sum(np.log(s))

    t_bar = t0
    nu = n
    outer = 0

    while outer < max_outer:
        # inner Newton
        for newton_iter in range(max_newton):
            s = B_d @ x + reg_eps
            inv_s = 1.0 / s
            inv_s2 = inv_s * inv_s

            grad_f = - (B_d.T @ inv_s)          # (n,)
            WB = (inv_s2[:, None] * B_d)        # (m,n)
            H_f = B_d.T @ WB                    # (n,n)

            grad_phi = -1.0 / x
            H_phi = np.diag(1.0 / (x * x))

            grad_F = t_bar * grad_f + grad_phi
            H = t_bar * H_f + H_phi

            # KKT
            rhs = np.concatenate([-grad_F, [-(np.sum(x) - 1.0)]])
            KKT = np.zeros((n + 1, n + 1))
            KKT[:n, :n] = H
            KKT[:n, n] = 1.0
            KKT[n, :n] = 1.0

            try:
                sol = np.linalg.solve(KKT, rhs)
            except np.linalg.LinAlgError:
                KKT[:n, :n] += 1e-8 * np.eye(n)
                sol = np.linalg.solve(KKT, rhs)

            dx = sol[:n]

            newton_resid = np.linalg.norm(rhs)
            if newton_resid < newton_tol:
                break

            # Step-length ensuring interior and positivity of s
            alpha = 1.0
            neg_idx = dx < 0
            if np.any(neg_idx):
                alpha = min(alpha, 0.99 * np.min(-x[neg_idx] / dx[neg_idx]))
            beta = 0.5
            c = 1e-4
            curr_F = t_bar * objective_f(x) + (-np.sum(np.log(x)))
            while alpha > 1e-16:
                x_new = x + alpha * dx
                if np.any(x_new <= 0):
                    alpha *= beta
                    continue
                s_new = B_d @ x_new + reg_eps
                if np.any(s_new <= 0):
                    alpha *= beta
                    continue
                F_new = t_bar * (-np.sum(np.log(s_new))) + (-np.sum(np.log(x_new)))
                if F_new <= curr_F + c * alpha * (grad_F @ dx):
                    x = x_new
                    break
                else:
                    alpha *= beta
            if alpha <= 1e-16:
                # failed to make progress; tiny step
                break

        # stopping criterion
        if (nu / t_bar) <= tol:
            break

        t_bar *= mu
        outer += 1

    # map back to z (length n+1)
    z_out = np.empty(n + 1, dtype=float)
    z_out[0] = 0.0
    np.cumsum(x, out=z_out[1:])  # z_out[1:] becomes cumulative sums of x
    z_out[-1] = 1.0  # enforce exact endpoint

    # update bezierv with new controls (keep controls_x unchanged)
    bezierv.update_bezierv(init_x, z_out)

    final_obj = objective_f(x)
    return bezierv, final_obj