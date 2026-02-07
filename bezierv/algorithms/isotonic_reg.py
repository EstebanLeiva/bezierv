import numpy as np

def bounded_iso_mean(y, w, a=None, b=None):
    """
    Compute the bounded isotonic mean using a stack-based Pool Adjacent Violators Algorithm (PAVA).
    
    This is an O(N) implementation that computes the weighted isotonic regression of y with respect 
    to weights w, subject to element-wise lower bounds a and upper bounds b. The algorithm uses a 
    stack-based approach to merge adjacent blocks that violate the isotonicity constraint (i.e., 
    when a previous block's value is greater than or equal to the current block's value).
    
    The isotonic regression finds the best monotonically non-decreasing approximation to the input 
    data by minimizing the weighted squared error, while respecting the provided bounds on each element.
    
    Parameters
    ----------
    y : array_like
        Input values to be fitted. Expected to be a 1D array of length n.
    w : array_like
        Weights for each element in y. Must have the same length as y. Weights should be positive.
    a : array_like, optional
        Lower bounds for each element. If None, defaults to -infinity for all elements.
        Must have the same length as y if provided.
    b : array_like, optional
        Upper bounds for each element. If None, defaults to +infinity for all elements.
        Must have the same length as y if provided.
    
    Returns
    -------
    np.ndarray
        The bounded isotonic regression result as a 1D array of length n. The output is 
        monotonically non-decreasing and each element satisfies a[i] <= result[i] <= b[i].
        If bounds are invalid (a[i] > b[i]) for any merged block, that block's value is NaN.
    
    Notes
    -----
    The algorithm uses a stack-based PAVA approach that maintains O(1) incremental updates. 
    Each block on the stack stores:
    - Weighted sum (wy)
    - Sum of weights (w)
    - Maximum lower bound (a) across merged elements
    - Minimum upper bound (b) across merged elements
    - Count of original elements in the block
    - Current bounded average value
    
    When merging blocks, the bounds are combined by taking the maximum of lower bounds and
    the minimum of upper bounds to ensure all constraints are satisfied.
    
    Examples
    --------
    >>> y = np.array([3.0, 1.0, 2.0, 4.0])
    >>> w = np.ones(4)
    >>> bounded_iso_mean(y, w)
    array([2., 2., 2., 4.])
    
    >>> y = np.array([3.0, 1.0, 2.0, 4.0])
    >>> w = np.ones(4)
    >>> a = np.array([0.0, 0.0, 0.0, 0.0])
    >>> b = np.array([5.0, 5.0, 5.0, 5.0])
    >>> bounded_iso_mean(y, w, a, b)
    array([2., 2., 2., 4.])
    """
    n = len(y)

    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    
    if a is None:
        a = np.full(n, -np.inf)
    else:
        a = np.asarray(a, dtype=np.float64)
        
    if b is None:
        b = np.full(n, np.inf)
    else:
        b = np.asarray(b, dtype=np.float64)

    stack_wy = []
    stack_w = []
    stack_a = []
    stack_b = []
    stack_count = []
    stack_val = []

    for i in range(n):
        curr_wy = y[i] * w[i]
        curr_w = w[i]
        curr_a = a[i]
        curr_b = b[i]
        curr_count = 1
        
        raw_val = curr_wy / curr_w
        val = min(raw_val, curr_b)
        val = max(val, curr_a)
        
        while stack_val and stack_val[-1] >= val:
            prev_wy = stack_wy.pop()
            prev_w = stack_w.pop()
            prev_a = stack_a.pop()
            prev_b = stack_b.pop()
            prev_count = stack_count.pop()
            stack_val.pop()
            
            curr_wy += prev_wy
            curr_w += prev_w
            curr_count += prev_count
            
            curr_a = max(curr_a, prev_a)
            curr_b = min(curr_b, prev_b)

            if curr_a > curr_b:
                val = np.nan
            else:
                raw_val = curr_wy / curr_w
                val = min(raw_val, curr_b)
                val = max(val, curr_a)

        stack_wy.append(curr_wy)
        stack_w.append(curr_w)
        stack_a.append(curr_a)
        stack_b.append(curr_b)
        stack_count.append(curr_count)
        stack_val.append(val)

    result = np.empty(n, dtype=np.float64)
    cursor = 0
    for val, count in zip(stack_val, stack_count):
        result[cursor : cursor + count] = val
        cursor += count
        
    return result

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