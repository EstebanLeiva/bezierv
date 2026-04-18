import numpy as np

def bounded_iso_mean(y: np.ndarray, 
                     w: np.ndarray, 
                     a: np.ndarray = None, 
                     b: np.ndarray = None) -> np.ndarray:
    """
    Weighted isotonic regression with element-wise bounds via PAVA.

    Solves ``min_{x} sum_i w[i] * (x[i] - y[i])^2`` subject to
    ``x[0] <= x[1] <= ... <= x[n-1]`` and ``a[i] <= x[i] <= b[i]``,
    using an O(n) stack-based Pool Adjacent Violators Algorithm. This is
    the projection subroutine used by the first-order fitting algorithms
    described in Section 3 of the paper.

    Parameters
    ----------
    y : array_like, shape (n,)
        Input values to make isotonic.
    w : array_like, shape (n,)
        Positive weights for each element.
    a : array_like, shape (n,), optional
        Per-element lower bounds. Defaults to ``-inf`` if not provided.
    b : array_like, shape (n,), optional
        Per-element upper bounds. Defaults to ``+inf`` if not provided.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Non-decreasing sequence satisfying ``a[i] <= result[i] <= b[i]``.
        Entries belonging to a block whose merged bounds are infeasible
        (``a > b``) are set to ``NaN``.

    Notes
    -----
    Each stack block tracks: weighted sum ``wy``, total weight ``w``,
    tightest lower bound ``max(a)``, tightest upper bound ``min(b)``,
    element count, and current clipped mean. Blocks are merged whenever
    the top-of-stack value would violate monotonicity.

    Examples
    --------
    >>> y = np.array([3.0, 1.0, 2.0, 4.0])
    >>> w = np.ones(4)
    >>> bounded_iso_mean(y, w)
    array([2., 2., 2., 4.])

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

def project(controls: np.ndarray, 
            lower: float, 
            upper: float) -> np.ndarray:
    """
    Project control points onto the bounded monotone feasible set.

    Fixes the boundary control points at ``lower`` and ``upper`` and projects
    the ``n - 1`` interior points onto the set of non-decreasing sequences
    in ``[lower, upper]`` via :func:`bounded_iso_mean`. This corresponds to
    the projection operator used in the projected gradient algorithms of the
    paper (Section 3).

    Parameters
    ----------
    controls : numpy.ndarray, shape (n + 1,)
        Current z-coordinates of the ``n + 1`` Bézier control points.
    lower : float
        Lower boundary value; enforced as ``controls[0]``. Corresponds to
        ``z_0 = 0`` (cdf boundary condition) in the paper.
    upper : float
        Upper boundary value; enforced as ``controls[-1]``. Corresponds to
        ``z_n = 1`` (cdf boundary condition) in the paper.

    Returns
    -------
    numpy.ndarray, shape (n + 1,)
        Projected control points satisfying ``lower = p[0] <= p[1] <= ... <=
        p[n] = upper``.
    """
    n = len(controls)
    projected = np.zeros(n)
    projected[0] = lower
    projected[-1] = upper
    if n > 2:
        w = np.ones(n - 2)
        projected[1:n-1] = bounded_iso_mean(controls[1:n-1], w, a=np.full(n-2, lower), b=np.full(n-2, upper))
    return projected