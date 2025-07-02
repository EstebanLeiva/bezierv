import numpy as np
from scipy.optimize import brentq
from bezierv.classes.bezierv import Bezierv

def root_find(n: int,
              bezierv: Bezierv,
              controls_x: np.array, 
              data_point: float) -> float:
    """
    Find the parameter value t such that the Bezier random variable's x-coordinate equals data_i.

    This method defines a function representing the difference between the x-coordinate
    of the Bezier curve and the given data point. It then uses Brent's method to find a 
    root of this function, i.e., a value t in [0, 1] that satisfies the equality.

    Parameters
    ----------
    n : int
        The number of control points minus one for the Bezier curve.
    bezierv : Bezierv
        An instance of the Bezierv class representing the Bezier random variable.
    controls_x : np.array
        The control points for the x-coordinates of the Bezier curve.
    data_point : float
        The data point for which to find the corresponding value of t.

    Returns
    -------
    float
        The value of t in the interval [0, 1] such that the Bezier random variable's x-coordinate
        is approximately equal to data_point.
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
          data: np.array,
          bezierv: Bezierv,
          controls_x: np.array) -> np.array:
    """
    Compute values of the parameter t for each data point using root-finding.

    For each sorted data point, this method finds the corresponding value 't' such that the 
    x-coordinate of the Bezier random variable matches the data point. 

    Parameters
    ----------
    controls_x : np.array
        The control points for the x-coordinates of the Bezier curve.
    data : np.array
        Array of data points for which to compute the corresponding values of t.

    Returns
    -------
    np.array
        An array of values of t corresponding to each data point.
    """
    t = np.zeros(m)
    for i in range(m):
        t[i] = root_find(n, bezierv, controls_x, data[i])
    return t