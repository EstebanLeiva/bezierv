import numpy as np
import copy
from dataclasses import dataclass, field
from bezierv.classes.bezierv import Bezierv
from typing import List, Union
from statsmodels.distributions.empirical_distribution import ECDF
from bezierv.algorithms import proj_grad as pg
from bezierv.algorithms import non_linear as nl
from bezierv.algorithms import nelder_mead as nm
from bezierv.algorithms import utils as utils
from bezierv.algorithms import primal_grad as primg


@dataclass
class ProjGradOptions:
    step_size: float = 0.001
    max_iter: int = 1000
    threshold: float = 1e-3


@dataclass
class NonLinearOptions:
    solver: str = 'ipopt'
    solver_options: dict = field(default_factory=lambda: {'timelimit': 60, 'tee': False})


@dataclass
class NelderMeadOptions:
    max_iter: int = 1000


@dataclass
class MLEOptions:
    max_iter: int = 1000
    tol: float = 1e-3
    tol_res_root: float = 1e-5
    tol_lambda_root: float = 1e-5
    max_iters_root: int = 100


FitOptions = Union[ProjGradOptions, NonLinearOptions, NelderMeadOptions, MLEOptions]


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
                 n: int=10, 
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
            The number of control points minus one for the Bezier curve (default is 10).
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
            options: FitOptions=None) -> Bezierv:
        """
        Fit the bezierv distribution to the data.

        Parameters
        ----------
        method : str, optional
            The fitting method to use. Options are 'mse' for mean squared error
            or 'mle' for maximum likelihood estimation (default is 'mse').
        algorithm : str, optional
            The fitting algorithm to use when ``method='mse'``. Options are
            'projgrad', 'nonlinear', or 'neldermead'. Ignored when
            ``method='mle'``. Default is 'projgrad'.
        options : FitOptions, optional
            Algorithm-specific options. Must match the chosen method/algorithm:
            ``ProjGradOptions`` for ('mse', 'projgrad'),
            ``NonLinearOptions`` for ('mse', 'nonlinear'),
            ``NelderMeadOptions`` for ('mse', 'neldermead'),
            ``MLEOptions`` for ('mle'). Defaults to the matching dataclass
            with its default field values.

        Returns
        -------
        Bezierv
            The fitted Bezierv instance with updated control points.
        """
        if method == 'mse':
            if algorithm == 'projgrad':
                opts = options if options is not None else ProjGradOptions()
                self._check_options(opts, ProjGradOptions)
                self.bezierv, metric = pg.fit(
                    self.n,
                    self.bezierv,
                    self.init_x,
                    self.init_z,
                    self.init_t,
                    self.emp_cdf_data,
                    opts.step_size,
                    opts.max_iter,
                    opts.threshold)
                self.mse = metric
            elif algorithm == 'nonlinear':
                opts = options if options is not None else NonLinearOptions()
                self._check_options(opts, NonLinearOptions)
                self.bezierv, metric = nl.fit(
                    self.n,
                    self.m,
                    self.data,
                    self.bezierv,
                    self.init_x,
                    self.init_z,
                    self.init_t,
                    self.emp_cdf_data,
                    opts.solver,
                    opts.solver_options)
                self.mse = metric
            elif algorithm == 'neldermead':
                opts = options if options is not None else NelderMeadOptions()
                self._check_options(opts, NelderMeadOptions)
                self.bezierv, metric = nm.fit(
                    self.n,
                    self.m,
                    self.data,
                    self.bezierv,
                    self.init_x,
                    self.init_z,
                    self.emp_cdf_data,
                    opts.max_iter)
                self.mse = metric
            else:
                raise ValueError("Algorithm not recognized. Use 'projgrad', 'nonlinear', or 'neldermead'.")

        elif method == 'mle':
            opts = options if options is not None else MLEOptions()
            self._check_options(opts, MLEOptions)
            self.bezierv, metric = primg.fit(
                self.n,
                self.m,
                self.bezierv,
                self.init_x,
                self.init_w,
                self.init_t,
                opts.max_iter,
                opts.tol,
                opts.tol_res_root,
                opts.tol_lambda_root,
                opts.max_iters_root)
            self.nll = metric
        else:
            raise ValueError("Method not recognized. Use 'mse' or 'mle'.")

        return copy.copy(self.bezierv), metric

    @staticmethod
    def _check_options(options, expected):
        if not isinstance(options, expected):
            raise TypeError(
                f"Expected options of type {expected.__name__}, got {type(options).__name__}."
            )
    
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