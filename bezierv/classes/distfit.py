import numpy as np
from bezierv.classes.bezierv import Bezierv
from typing import List
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.optimize import brentq
from bezierv.algorithms import proj_grad2 as pj
#from bezierv.algorithms import non_linear2 as nl
#from bezierv.algorithms import nelder_mead2 as alg


class DistFit:
    """
    """
    def __init__(self, 
                 data: List, 
                 n: int=5, 
                 init_x: np.array=None, 
                 init_z: np.array=None, 
                 emp_cdf_data: np.array=None, 
                 method_init_x: str='quantile'
                 ):
        """
        """
        self.data = np.sort(data)
        self.n = n
        self.bezierv = Bezierv(n)
        self.m = len(data)
        self.mse = np.inf

        if init_x is None:
            self.init_x = self.get_controls_x(method_init_x)
        else:
            self.init_x = init_x

        self.init_t = self.get_t(self.init_x, self.data)

        if init_z is None:
            self.init_z = self.get_controls_z()
        else:
            self.init_z = init_z

        if emp_cdf_data is None:
            emp_cdf = ECDF(self.data)
            self.emp_cdf_data = emp_cdf(self.data)
        else:
            self.emp_cdf_data = emp_cdf_data

    def fit(self,
            method: str='projgrad',
            step_size_PG: float=0.001,
            max_iter_PG: float=1000,
            threshold_PG: float=1e-3,
            solver_NL: str='ipopt',
            step_size_NM=0.1,
            no_improve_thr_NM=10e-6,
            no_improv_break_NM=10, 
            max_iter_NM=0,
            alpha_NM=1., 
            gamma_NM=2., 
            rho_NM=-0.5, 
            sigma_NM=0.5) -> Bezierv:
        """
        Fit the specified distribution to the data.

        This method uses the specified fitting method (e.g., 'projgrad') to fit the Bezier random variable
        to the empirical CDF data.

        Returns
        -------
        Bezierv
            The fitted Bezierv instance with updated control points.
        """
        if method == 'projgrad':
            self.bezierv, self.mse = pj.fit(self.n, 
                                            self.m, 
                                            self.data, 
                                            self.bezierv, 
                                            self.init_x, 
                                            self.init_z, 
                                            self.init_t,
                                            self.emp_cdf_data, 
                                            step_size_PG, 
                                            max_iter_PG, 
                                            threshold_PG)
        elif method == 'nonlinear':
            None
        elif method == 'neldermead':
            None

        return self.bezierv
    
    def get_controls_z(self) -> np.array:
        """
        Compute the control points for the z-coordinates of the Bezier curve.

        'quantile' method is used to determine the control points based on the data quantiles.

        Parameters
        ----------
        data : np.array
            The sorted data points.

        Returns
        -------
        np.array
            The control points for the z-coordinates of the Bezier curve.
        """
        controls_z = np.linspace(0, 1, self.n + 1)
        return controls_z

    def get_controls_x(self, method: str) -> np.array:
        """
        Compute the control points for the x-coordinates of the Bezier curve.

        'quantile' method is used to determine the control points based on the data quantiles.

        Parameters
        ----------
        data : np.array
            The sorted data points.

        Returns
        -------
        np.array
            The control points for the x-coordinates of the Bezier curve.
        """
        controls_x = np.zeros(self.n + 1)
        if method == 'quantile':
            for i in range(self.n + 1):
                controls_x[i] = np.quantile(self.data, i/self.n)
        else:
            raise ValueError("Method not recognized. Use 'quantile'.")
        return controls_x
    
    def root_find(self, controls_x: np.array, data_point:float) -> float:
        """
        Find the parameter value t such that the Bezier random variable's x-coordinate equals data_i.

        This method defines a function representing the difference between the x-coordinate
        of the Bezier curve and the given data point. It then uses Brent's method to find a 
        root of this function, i.e., a value t in [0, 1] that satisfies the equality.

        Parameters
        ----------
        controls_x : np.array
            The control points for the x-coordinates of the Bezier curve.
        data_point : float
            A single data point for which to find the corresponding parameter value.

        Returns
        -------
        float
            The value of t in the interval [0, 1] such that the Bezier random variable's x-coordinate
            is approximately equal to data_point.
        """
        def poly_x_sample(t, controls_x, data_point):
            p_x = 0
            for i in range(self.n + 1):
                p_x += self.bezierv.bernstein(t, i, self.bezierv.comb, self.bezierv.n) * controls_x[i]
            return p_x - data_point
        t = brentq(poly_x_sample, 0, 1, args=(controls_x, data_point))
        return t
    
    def get_t(self, controls_x, data) -> np.array:
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
        t = np.zeros(self.m)
        for i in range(self.m):
            t[i] = self.root_find(controls_x, data[i])
        return t