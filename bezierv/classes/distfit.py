import numpy as np
import copy
from bezierv.classes.bezierv import Bezierv
from typing import List
from statsmodels.distributions.empirical_distribution import ECDF
from bezierv.algorithms import proj_grad as pg
from bezierv.algorithms import non_linear as nl
from bezierv.algorithms import nelder_mead as nm
from bezierv.algorithms import utils as utils
from bezierv.algorithms import primal_grad as primg


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
                 init_w: np.array=None,
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
        self.nll = np.inf

        if init_x is None:
            init_x = self.get_controls_x(method_init_x)
            if np.diff(init_x).min() <= 0:
                raise ValueError("init_x must be distinct (strictly increasing).")
            self.init_x = init_x
        else:
            init_x = np.asarray(init_x, dtype=float)
            if np.diff(init_x).min() <= 0:
                raise ValueError("init_x must be distinct (strictly increasing).")
            self.init_x = init_x
            

        if init_t is None:
            self.init_t = utils.get_t(self.n, self.m, self.data, self.bezierv, self.init_x)
        else:
            self.init_t = init_t

        if init_z is None:
            self.init_z = self.get_controls_z()
        else:
            init_z = np.asarray(init_z, dtype=float)
            self.init_z = init_z

        if init_w is None:
            self.init_w = np.ones(self.n) / self.n
        else:
            init_w = np.asarray(init_w, dtype=float)
            self.init_w = init_w

        if emp_cdf_data is None:
            emp_cdf = ECDF(self.data)
            self.emp_cdf_data = emp_cdf(self.data)
        else:
            self.emp_cdf_data = emp_cdf_data

    def fit(self,
            method: str='mse',
            algorithm: str='projgrad',
            step_size_PG: float=0.001,
            max_iter_PG: int=1000,
            threshold_PG: float=1e-3,
            solver_NL: str='ipopt',
            max_iter_NM: int=1000,
            max_iter: int=1000,
            tol: float=1e-3,
            tol_res_root: float=1e-5,
            tol_lambda_root: float=1e-5,
            max_iters_root: int=100) -> Bezierv:
        """
        Fit the bezierv distribution to the data.

        Parameters
        ----------
        method : str, optional
            The fitting method to use. Options are 'mse' for mean squared error or 'mle' for maximum likelihood estimation (default is 'mse').
        algorithm : str, optional
            The fitting algorithm to use. Options are 'projgrad', 'nonlinear', or 'neldermead'.
            Default is 'projgrad'.
        step_size_PG : float, optional
            The step size for the projected gradient descent method (default is 0.001).
        max_iter_PG : int, optional
            The maximum number of iterations for the projected gradient descent method (default is 1000).
        threshold_PG : float, optional
            The convergence threshold for the projected gradient descent method (default is 1e-3).
        solver_NL : str, optional
            The solver to use for the nonlinear fitting method (default is 'ipopt').
        max_iter_NM : int, optional
            The maximum number of iterations for the Nelder-Mead optimization method (default is 1000).
        tol : float, optional
            The tolerance for convergence (default is 1e-3).
        tol_res_root : float, optional
            The tolerance for the root-finding in the primal gradient method (default is 1e-5).
        tol_lambda_root : float, optional
            The tolerance for the lambda root-finding in the primal gradient method (default is 1e-5).
        max_iters_root : int, optional
            The maximum number of iterations for the root-finding in the primal gradient method (default is 100).
        Returns
        -------
        Bezierv
            The fitted Bezierv instance with updated control points.
        """
        if method == 'mse':
            if algorithm == 'projgrad':
                self.bezierv, metric = pg.fit(
                                                self.n, 
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
                self.mse = metric
            elif algorithm == 'nonlinear':
                self.bezierv, metric = nl.fit(
                                                self.n,
                                                self.m,
                                                self.data,
                                                self.bezierv,
                                                self.init_x,
                                                self.init_z,
                                                self.init_t,
                                                self.emp_cdf_data,
                                                solver_NL)
                self.mse = metric

            elif algorithm == 'neldermead':
                self.bezierv, metric = nm.fit(
                                                self.n,
                                                self.m,
                                                self.data,
                                                self.bezierv,
                                                self.init_x,
                                                self.init_z,
                                                self.emp_cdf_data,
                                                max_iter_NM)
                self.mse = metric
            else:
                raise ValueError("Algorithm not recognized. Use 'projgrad', 'nonlinear', or 'neldermead'.")
            
        elif method == 'mle':
            self.bezierv, metric = primg.fit(
                                            self.n,
                                            self.m,
                                            self.data,
                                            self.bezierv,
                                            self.init_x,
                                            self.init_w,
                                            self.init_t,
                                            self.emp_cdf_data,
                                            max_iter,
                                            tol,
                                            tol_res_root,
                                            tol_lambda_root,
                                            max_iters_root)
            self.nll = metric
        else:
            raise ValueError("Method not recognized. Use 'mse' or 'mle'.")

        return copy.copy(self.bezierv), metric
    
    def get_controls_z(self) -> np.array:
        """
        Compute the control points for the z-coordinates of the Bezier curve.
        """
        controls_z = np.linspace(0, 1, self.n + 1)
        return controls_z

    def get_controls_x(self, method: str) -> np.array:
        """
        Compute the control points for the x-coordinates of the Bezier curve.

        'quantile' method is used to determine the control points based on the data quantiles.

        Parameters
        ----------
        method : str
            The method to use for initializing the x-coordinates of the control points.

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