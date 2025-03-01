import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.optimize import brentq

from bezierv.classes.bezierv import Bezierv

class ProjGrad:
    def __init__(self, bezierv: Bezierv, controls_x: np.array, data: np.array, emp_cdf_data: np.array=None):
        """
        Initialize the ProjGrad instance with a Bezierv object and data to fit.

        This constructor sets up the instance by storing the provided Bezierv object,
        control points for the x-dimension, and the data points. It computes the
        corresponding parameter values (t_data) for the data points using a root-finding
        procedure and calculates the empirical CDF values if they are not provided.

        Parameters
        ----------
        bezierv : Bezierv
            An instance of the Bezierv class representing the Bezier random variable.
        controls_x : np.array
            The control points for the x-coordinates of the Bezier random variable.
        data : np.array
            The data points to be fitted; they will be sorted.
        emp_cdf_data : np.array, optional
            The empirical CDF values corresponding to the data. If not provided, they
            are computed using ECDF from the sorted data.

        Attributes
        ----------
        bezierv : Bezierv
            The Bezierv object.
        n : int
            The degree of the Bezier random variable (inferred from bezierv.n).
        controls_x : np.array
            The x-coordinates of the control points.
        data : np.array
            The sorted data points.
        m : int
            The number of data points.
        t_data : np.array
            The values of t corresponding to the data points on the Bezier random variable.
        fit_error : float
            The fitting error, initialized to infinity.
        emp_cdf_data : np.array
            The empirical CDF values for the data points.
        """
        self.bezierv = bezierv
        self.n = bezierv.n
        self.controls_x = controls_x
        self.data = np.sort(data)
        self.m = len(data)
        self.t_data = self.get_t_data(data)
        self.fit_error = np.inf

        if emp_cdf_data is None:
            emp_cdf = ECDF(data)
            self.emp_cdf_data = emp_cdf(data)
        else:
            self.emp_cdf_data = emp_cdf_data

    def get_t_data(self, data):
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
            t_data[i] = self.root_find(self.controls_x, data[i])
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
    
    def grad(self, t: float, controls_z: np.array):
        """
        Compute the gradient of the fitting objective with respect to the z control points.

        Parameters
        ----------
        t : float or np.array
            The parameter values corresponding to the data points. Expected to be an array of shape (m,).
        controls_z : np.array
            The current z-coordinates of the control points.

        Returns
        -------
        np.array
            An array representing the gradient with respect to each z control point.
        """
        grad_z = np.zeros(self.n + 1)
        for j in range(self.m):
            inner_sum = np.zeros(self.n + 1)
            for i in range(self.n + 1):
                inner_sum[i] = self.bezierv.bernstein(t[j], i, self.bezierv.comb, self.n)

            grad_z += 2 * (self.bezierv.poly_z(t[j], controls_z) - self.emp_cdf_data[j]) * inner_sum

        return grad_z
    
    def project_z(self, controls_z: np.array):
        """
        Project the z control points onto the feasible set.

        The projection is performed by first clipping the control points to the [0, 1] interval,
        sorting them in ascending order, and then enforcing the boundary conditions by setting
        the first control point to 0 and the last control point to 1.

        Parameters
        ----------
        controls_z : np.array
            The current z-coordinates of the control points.

        Returns
        -------
        np.array
            The projected z control points that satisfy the constraints.
        """
        z_prime = np.clip(controls_z.copy(), a_min= 0, a_max=1)
        z_prime.sort()
        z_prime[0] = 0
        z_prime[-1] = 1
        return z_prime

    def fit(self, controls_z0: np.array, step = 0.01, maxiter = 1000):
        """
        Fit the Bezier curve to the empirical CDF data using projected gradient descent.

        Starting from an initial guess for the z control points, this method iteratively
        updates the z control points by taking gradient descent steps and projecting the
        result back onto the feasible set. The process continues until convergence is
        achieved (i.e., the change in control points is less than a set threshold) or until
        the maximum number of iterations is reached. After convergence, the Bezierv curve is
        updated with the new control points and the fitting error is computed.

        Parameters
        ----------
        controls_z0 : np.array
            The initial guess for the z control points.
        step : float, optional
            The step size (learning rate) for the gradient descent updates (default is 0.01).
        maxiter : int, optional
            The maximum number of iterations allowed for the fitting process (default is 1000).

        Returns
        -------
        Bezierv
            The updated Bezierv instance with fitted control points.
        """
        z = controls_z0
        for i in range(maxiter):
            grad_z = self.grad(self.t_data, z)
            z_prime = self.project_z(z - step * grad_z)
            
            if np.linalg.norm(z_prime - z) < 1e-4:
                z = z_prime
                print(f'Converged in {i} iterations')
                break
    
            z = z_prime
        
        fit_error = 0
        for j in range(self.m):
            fit_error += (self.bezierv.poly_z(self.t_data[j], z) - self.emp_cdf_data[j])**2
        
        self.bezierv.update_bezierv(self.controls_x, z)
        self.fit_error = fit_error/self.m

        return self.bezierv