import numpy as np
from scipy.optimize import brentq
from bezierv.classes.bezierv import Bezierv

def root_find(n: int,
              bezierv: Bezierv,
              controls_x: np.ndarray,
              data_point: float) -> float:
    """
    Find the parameter t such that the Bézier x-coordinate equals a given data point.

    Uses Brent's method to solve ``B_x(t) - data_point = 0`` on ``[0, 1]``,
    where ``B_x(t)`` is the x-component of the degree-``n`` Bézier curve (Eq. 2
    of the paper).

    Parameters
    ----------
    n : int
        Degree of the Bézier curve. A degree-``n`` curve has ``n + 1`` control
        points.
    bezierv : Bezierv
        Fitted Bézier random variable providing the Bernstein basis evaluator.
    controls_x : numpy.ndarray, shape (n + 1,)
        x-coordinates of the ``n + 1`` control points.
    data_point : float
        Observed value for which the inverse parameter ``t`` is sought.

    Returns
    -------
    float
        Parameter value ``t`` in ``[0, 1]`` satisfying ``B_x(t) = data_point``.

    Raises
    ------
    ValueError
        If ``brentq`` cannot bracket a root, i.e. ``data_point`` lies outside
        the range of ``B_x`` on ``[0, 1]``.
    """
    def poly_x_sample(t, controls_x, data_point):
        p_x = 0
        for i in range(n + 1):
            p_x += bezierv.bernstein(t, i, bezierv.comb, bezierv.n) * controls_x[i]
        return p_x - data_point
    t = brentq(poly_x_sample, 0, 1, args=(controls_x, data_point))
    return t
    
def get_t(n: int,
          m: int,
          data: np.ndarray,
          bezierv: Bezierv,
          controls_x: np.ndarray) -> np.ndarray:
    """
    Compute the Bézier parameter t for each data point via root-finding.

    For each observation in ``data``, solves ``B_x(t) = data[i]`` using
    :func:`root_find` to obtain the corresponding parameter value on ``[0, 1]``.

    Parameters
    ----------
    n : int
        Degree of the Bézier curve (``n + 1`` control points).
    m : int
        Number of observations (``len(data)``).
    data : numpy.ndarray, shape (m,)
        Observed sample values in ``[a, b]``.
    bezierv : Bezierv
        Fitted Bézier random variable providing the Bernstein basis evaluator.
    controls_x : numpy.ndarray, shape (n + 1,)
        x-coordinates of the ``n + 1`` control points.

    Returns
    -------
    numpy.ndarray, shape (m,)
        Parameter values ``t[i]`` in ``[0, 1]`` such that ``B_x(t[i]) = data[i]``
        for each ``i``.
    """
    t = np.zeros(m)
    for i in range(m):
        t[i] = root_find(n, bezierv, controls_x, data[i])
    return t