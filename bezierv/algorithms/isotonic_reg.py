import numpy as np

def MA(g, w, A, a, b):
    """
    Computes the bounded weighted average M(A) as defined in 
    Barlow, Bartholomew, Bremner & Brunk (1972), p. 57.
    
    This is a direct Python translation of your R 'MA' function.
    
    Args:
        g (np.ndarray): Data points (called 'y' in the main function).
        w (np.ndarray): Corresponding weights.
        A (list or np.ndarray): A list of 0-based indices for the subset.
        a (np.ndarray): Vector of lower bounds.
        b (np.ndarray): Vector of upper bounds.
        
    Returns:
        float: The bounded weighted average, or np.nan if bounds are invalid.
    """
    a_up = np.nanmax(a[A])
    b_low = np.nanmin(b[A])
    res = np.nan
    
    if a_up <= b_low:
        weighted_avg = np.sum(g[A] * w[A]) / np.sum(w[A])
        clamped_avg = np.minimum(weighted_avg, b_low)
        res = np.maximum(clamped_avg, a_up)
        
    return res

def bounded_iso_mean(y, w, a=None, b=None):
    """
    Pool-adjacent-violaters-algorithm (PAVA) for a weighted isotonic
    mean, bounded by a lower bound 'a' and an upper bound 'b'.
    
    This is a Python translation of the R function 'BoundedIsoMean'.
    
    Args:
        y (np.ndarray): Data points (1D array).
        w (np.ndarray): Corresponding weights (1D array).
        a (np.ndarray, optional): Vector of lower bounds. Defaults to -Inf.
        b (np.ndarray, optional): Vector of upper bounds. Defaults to +Inf.
        
    Returns:
        np.ndarray: The bounded isotonic mean vector.
    """
    n = len(y)
    
    if a is None:
        a = np.full(n, -np.inf)
    if b is None:
        b = -a

    y = np.asarray(y)
    w = np.asarray(w)
    a = np.asarray(a)
    b = np.asarray(b)
    k = np.zeros(n, dtype=int)
    ghat = np.zeros(n)
    c = 0
    k[0] = 0
    ghat[0] = MA(g=y, w=w, A=[0], a=a, b=b)

    for j in range(1, n):
        c += 1
        k[c] = j
        ghat[c] = MA(g=y, w=w, A=[j], a=a, b=b)

        while (c >= 1) and (ghat[c - 1] >= ghat[c]):
            start_idx = k[c - 1]
            end_idx = j
            A_indices = list(range(start_idx, end_idx + 1))
            ghat[c - 1] = MA(g=y, w=w, A=A_indices, a=a, b=b)
            c -= 1
    current_n_index = n - 1 
    while (current_n_index >= 0):
        start_idx = k[c]
        val_to_fill = ghat[c]
        ghat[start_idx : current_n_index + 1] = val_to_fill
        current_n_index = k[c] - 1
        c -= 1

    return ghat

def project(controls: np.array, lower: float, upper: float) -> np.array:
    n = len(controls)
    if n <= 2:
        projected = np.zeros(n)
        projected[0] = lower
        projected[-1] = upper
    else: 
        w = np.ones(n)
        projected = np.zeros(n)
        projected[1:n-1] = bounded_iso_mean(controls[1:n-1], w, a=np.full(n, lower), b=np.full(n, upper))
        projected[0] = lower
        projected[-1] = upper
    return projected