import copy
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.optimize import brentq, minimize
from bezierv.classes.bezierv import Bezierv

class NelderMeadSolver:
    def __init__(self, bezierv: Bezierv, data: np.array, 
                 emp_cdf_data: np.array=None,
                 step=0.1, no_improve_thr=10e-6,
                 no_improv_break=10, max_iter=0,
                 alpha=1., gamma=2., rho=-0.5, sigma=0.5):
        self.bezierv = bezierv
        self.n = bezierv.n
        self.data = np.sort(data)
        self.m = len(data)
        self.mse = np.inf

        if emp_cdf_data is None:
            emp_cdf = ECDF(data)
            self.emp_cdf_data = emp_cdf(data)
        else:
            self.emp_cdf_data = emp_cdf_data

        self.step = step
        self.no_improve_thr = no_improve_thr
        self.no_improv_break = no_improv_break
        self.max_iter = max_iter
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
    
    def get_t_data(self, controls_x, data):
        """
        Compute values of t for each data point using root-finding.

        For each sorted data point, this method finds the corresponding value 't' such that the 
        x-coordinate of the Bezier random variable matches the data point. 

        Parameters
        ----------
        data : np.array
            Array of data points for which to compute the corresponding values of t.

        Returns
        -------
        np.array
            An array of values of t corresponding to each data point.
        """
        t_data = np.zeros(self.m)
        for i in range(self.m):
            t_data[i] = self.root_find(controls_x, data[i])
        return t_data
    
    def root_find(self, controls_x, data_i):
        """
        Find the parameter value t such that the Bezier random variable's x-coordinate equals data_i.

        This method defines a function representing the difference between the x-coordinate
        of the Bezier curve and the given data point. It then uses Brent's method to find a 
        root of this function, i.e., a value t in [0, 1] that satisfies the equality.

        Parameters
        ----------
        controls_x : np.array
            The control points for the x-coordinates of the Bezier curve.
        data_i : float
            A single data point for which to find the corresponding parameter value.

        Returns
        -------
        float
            The value of t in the interval [0, 1] such that the Bezier random variable's x-coordinate
            is approximately equal to data_i.
        """
        def poly_x_sample(t, controls_x, data_i):
            p_x = 0
            for i in range(self.n + 1):
                p_x += self.bezierv.bernstein(t, i, self.bezierv.comb, self.bezierv.n) * controls_x[i]
            return p_x - data_i
        t = brentq(poly_x_sample, 0, 1, args=(controls_x, data_i))
        return t

    def objective_function(self, concatenated: np.array):
        """
        Compute the objective function value for the given control points.

        This method calculates the sum of squared errors between the Bezier random variable's CDF
        and the empirical CDF data.

        Parameters
        ----------
        concatenated : np.array
            A concatenated array containing the control points for z and x coordinates.
            The first n+1 elements are the z control points, and the remaining elements are the x control points.

        Returns
        -------
        float
            The value of the objective function (MSE).
        """
        x = concatenated[0:self.n + 1]
        z = concatenated[self.n + 1:]
        t = self.get_t_data(x, self.data)
        se = 0
        for j in range(self.m):
            se += (self.bezierv.poly_z(t[j], z) - self.emp_cdf_data[j])**2
        return se / self.m

    def objective_function_lagrangian(self, concatenated: np.array):
        """
        Compute the objective function value for the given control points.

        This method calculates the sum of squared errors between the Bezier random variable's CDF
        and the empirical CDF data.

        Parameters
        ----------
        concatenated : np.array
            A concatenated array containing the control points for z and x coordinates.
            The first n+1 elements are the z control points, and the remaining elements are the x control points.

        Returns
        -------
        float
            The value of the objective function (MSE).
        """
        
        x = concatenated[0:self.n + 1]
        z = concatenated[self.n + 1:]
        try:
            t = self.get_t_data(x, self.data)
        except ValueError as e:
            return np.inf
        se = 0
        for j in range(self.m):
            se += (self.bezierv.poly_z(t[j], z) - self.emp_cdf_data[j])**2

        mse = se / self.m
        # --- Adaptive Penalty Calculation ---
        penalty = 0.0
        # The penalty_weight is a tunable hyperparameter. 
        # A value between 1e3 and 1e7 is often a good starting point.
        penalty_weight = 1e3

        # 1. Penalty for boundary constraints on z (z[0] should be 0, z[-1] should be 1)
        penalty += abs(z[0] - 0.0)
        penalty += abs(z[-1] - 1.0)

        # 2. Penalty for non-monotonicity in z and x
        # This penalizes any elements in the difference array that are negative.
        delta_zs = np.diff(z)
        delta_xs = np.diff(x)
        penalty += np.sum(abs(np.minimum(0, delta_zs)))
        penalty += np.sum(abs(np.minimum(0, delta_xs)))

        # 3. Penalty for x control points being outside the data range
       
        penalty += abs(x[0] - self.data[0])
        penalty += abs(self.data[-1] - x[-1])
    
        return mse + penalty_weight * penalty

    def fit(self, controls_x0, controls_z0):
        start = np.concatenate((controls_x0, controls_z0))
        result = minimize(
            fun=self.objective_function_lagrangian, # Use the function with penalties for optimization
            x0=start,
            method='Nelder-Mead')
        sol = result.x
        controls_x = sol[0:self.n + 1]
        controls_z = sol[self.n + 1:]
        self.bezierv.update_bezierv(controls_x, controls_z, (self.data[0], self.data[-1]))
        self.mse = self.objective_function(sol)

        return self.bezierv
