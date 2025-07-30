import numpy as np
import copy
from bezierv.classes.bezierv import Bezierv
from typing import List
from statsmodels.distributions.empirical_distribution import ECDF
from bezierv.algorithms import proj_grad as pg
from bezierv.algorithms import non_linear as nl
from bezierv.algorithms import nelder_mead as nm
from bezierv.algorithms import proj_subgrad as ps
from bezierv.algorithms import utils as utils


class DistFit:
    """
    A class to fit a Bezier random variable to empirical data.

    Attributes
    ----------
    data : List
        The empirical data to fit the Bezier random variable to.
    n : int
        The number of control points minus one for the Bezier curve.
    init_x : np.array
        Initial control points for the x-coordinates of the Bezier curve.
    init_z : np.array
        Initial control points for the z-coordinates of the Bezier curve.
    init_t : np.array
        Initial parameter values.
    emp_cdf_data : np.array
        The empirical cumulative distribution function (CDF) data derived from the empirical data.
    bezierv : Bezierv
        An instance of the Bezierv class representing the Bezier random variable.
    m : int
        The number of empirical data points.
    mse : float
        The mean squared error of the fit, initialized to infinity.
    """
    def __init__(self, 
                 data: List, 
                 n: int=5, 
                 init_x: np.array=None, 
                 init_z: np.array=None,
                 init_t: np.array=None,
                 emp_cdf_data: np.array=None, 
                 method_init_x: str='quantile'
                 ):
        """
        Initialize the DistFit class with empirical data and parameters for fitting a Bezier random variable.

        Parameters
        ----------
        data : List
            The empirical data to fit the Bezier random variable to.
        n : int, optional
            The number of control points minus one for the Bezier curve (default is 5).
        init_x : np.array, optional
            Initial control points for the x-coordinates of the Bezier curve (default is None).
        init_z : np.array, optional
            Initial control points for the z-coordinates of the Bezier curve (default is None).
        emp_cdf_data : np.array, optional
            The empirical cumulative distribution function (CDF) data derived from the empirical data (default is None).
        method_init_x : str, optional
            Method to initialize the x-coordinates of the control points (default is 'quantile').
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

        if init_t is None:
            self.init_t = utils.get_t(self.n, self.m, self.data, self.bezierv, self.init_x)
        else:
            self.init_t = init_t

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
            step_size_PS: float=0.001,
            max_iter_PS: int=1000,
            solver_NL: str='ipopt',
            max_iter_NM: int=1000) -> Bezierv:
        """
        Fit the bezierv distribution to the data.

        Parameters
        ----------
        method : str, optional
            The fitting method to use. Options are 'projgrad', 'nonlinear', 'projsubgrad', or 'neldermead'.
            Default is 'projgrad'.
        step_size_PG : float, optional
            The step size for the projected gradient descent method (default is 0.001).
        max_iter_PG : int, optional
            The maximum number of iterations for the projected gradient descent method (default is 1000).
        threshold_PG : float, optional
            The convergence threshold for the projected gradient descent method (default is 1e-3).
        solver_NL : str, optional
            The solver to use for the nonlinear fitting method (default is 'ipopt').
        
        Returns
        -------
        Bezierv
            The fitted Bezierv instance with updated control points.
        """
        if method == 'projgrad':
            self.bezierv, self.mse = pg.fit(self.n, 
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
            self.bezierv, self.mse = nl.fit(self.n,
                                            self.m,
                                            self.data,
                                            self.bezierv,
                                            self.init_x,
                                            self.init_z,
                                            self.init_t,
                                            self.emp_cdf_data,
                                            solver_NL)
        elif method == 'neldermead':
            self.bezierv, self.mse = nm.fit(self.n,
                                            self.m,
                                            self.data,
                                            self.bezierv,
                                            self.init_x,
                                            self.init_z,
                                            self.emp_cdf_data,
                                            max_iter_NM)
        elif method == 'projsubgrad':
            self.bezierv, self.mse = ps.fit(self.n,
                                            self.m,
                                            self.data,
                                            self.bezierv,
                                            self.init_x,
                                            self.init_z,
                                            self.init_t,
                                            self.emp_cdf_data,
                                            step_size_PS,
                                            max_iter_PS)
        else:
            raise ValueError("Method not recognized. Use 'projgrad', 'nonlinear', or 'neldermead'.")

        return copy.copy(self.bezierv), copy.copy(self.mse)
    
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
        if method == 'quantile':
            controls_x = np.zeros(self.n + 1)
            for i in range(self.n + 1):
                controls_x[i] = np.quantile(self.data, i/self.n)
        elif method == 'uniform':
            controls_x = np.linspace(np.min(self.data), np.max(self.data), self.n + 1)
        else:
            raise ValueError("Method not recognized. Use 'quantile' or 'uniform'.")
        return controls_x