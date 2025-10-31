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
    # R: a.up <- max(a[A], na.rm = TRUE)
    # np.nanmax replicates the behavior of na.rm = TRUE
    a_up = np.nanmax(a[A])
    
    # R: b.low <- min(b[A], na.rm = TRUE)
    # np.nanmin replicates the behavior of na.rm = TRUE
    b_low = np.nanmin(b[A])
    
    # R: res <- NA
    res = np.nan
    
    # R: if (a.up <= b.low){ ... }
    if a_up <= b_low:
        # R: res <- max(min(sum((g * w)[A]) / sum(w[A]), b.low), a.up)
        
        # Calculate weighted average for the subset
        weighted_avg = np.sum(g[A] * w[A]) / np.sum(w[A])
        
        # Clamp the average
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
    
    # Handle default bounds
    if a is None:
        a = np.full(n, -np.inf)
    if b is None:
        # This replicates the R logic: b <- -a
        # If 'a' was also None, 'a' is now -Inf, so 'b' becomes +Inf.
        b = -a

    # Ensure inputs are NumPy arrays
    y = np.asarray(y)
    w = np.asarray(w)
    a = np.asarray(a)
    b = np.asarray(b)

    # 'k' will store the 0-based *start index* of each block
    k = np.zeros(n, dtype=int)
    ghat = np.zeros(n)
    
    # 'c' is the 0-based *index* (and counter) of the current block
    c = 0
    
    # R: k[1] <- 1
    k[0] = 0
    
    # R: ghat[1] <- MA(y, w, A = 1, a, b)
    # Note: R's A=1 (1-based index) becomes [0] (0-based index list)
    ghat[0] = MA(g=y, w=w, A=[0], a=a, b=b)

    # R: for (j in 2:n){
    for j in range(1, n):
        # R: c <- c + 1
        c += 1
        
        # R: k[c] <- j
        k[c] = j
        
        # R: ghat[c] <- MA(y, w, A = j, a, b)
        ghat[c] = MA(g=y, w=w, A=[j], a=a, b=b)

        # R: while ((c >= 2) && (ghat[max(1, c - 1)] >= ghat[c])){
        # We use c >= 1 because 'c' is 0-based
        while (c >= 1) and (ghat[c - 1] >= ghat[c]):
            
            # R: ind <- k[c - 1]:j
            # Create a 0-based index list for the merged block
            start_idx = k[c - 1]
            end_idx = j
            A_indices = list(range(start_idx, end_idx + 1))
            
            # R: ghat[c - 1] <- MA(y, w, A = ind, a, b)
            ghat[c - 1] = MA(g=y, w=w, A=A_indices, a=a, b=b)
            
            # R: c <- c-1
            c -= 1
        # end while
    # end for j

    # This loop back-fills the 'ghat' array with the block averages
    
    # R: while (n >= 1){
    # We use 'current_n_index' (0-based) to track the end of the
    # section we are filling.
    current_n_index = n - 1 
    while (current_n_index >= 0):
        
        # R: for (j in k[c]:n){ghat[j] <- ghat[c]}
        start_idx = k[c]
        val_to_fill = ghat[c]
        ghat[start_idx : current_n_index + 1] = val_to_fill
        
        # R: n <- k[c] - 1
        current_n_index = k[c] - 1
        
        # R: c <- c - 1
        c -= 1
    # end while

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