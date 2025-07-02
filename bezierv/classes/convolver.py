import numpy as np
from scipy.integrate import quad
from bezierv.classes.distfit import DistFit
from bezierv.classes.bezierv import Bezierv

class Convolver:
    def __init__(self, bezierv_x: Bezierv, bezierv_y: Bezierv, grid: int=100):
        """
        Initialize a ConvBezier instance for convolving two Bezier curves.

        This constructor sets up the convolution object by storing the provided Bezierv
        random variables, and creates a new Bezierv instance to hold the convolution 
        result. It also initializes the number of data points to be used in the numerical
        convolution process.

        Parameters
        ----------
        bezierv_x : Bezierv
            A Bezierv instance representing the first Bezier random variable.
        bezierv_y : Bezierv
            A Bezierv instance representing the second Bezier random variable.
        grid : int
            The number of data points to generate for the convolution (more data points
            induce a better approximation).

        Attributes
        ----------
        bezierv_x : Bezierv
            The Bezierv instance representing the first random variable. Control points must be in non-decreasing order.
        bezierv_y : Bezierv
            The Bezierv instance representing the second random variable. Control points must be in non-decreasing order.
        bezierv_conv : Bezierv
            A Bezierv instance that will store the convolution result. Its number of control
            points is set to the maximum of control points between bezierv_x and bezierv_y.
        m : int
            The number of data points used in the convolution.
        """
        if bezierv_x.check_ordering() is False:
            raise ValueError("bezierv_x controls are not in non-decreasing order.")
        if bezierv_y.check_ordering() is False:
            raise ValueError("bezierv_y controls are not in non-decreasing order.")
        
        self.bezierv_x = bezierv_x
        self.bezierv_y = bezierv_y
        self.n = max(bezierv_x.n, bezierv_y.n)
        self.grid = grid

    def cdf_z (self, z: float) -> float:
        """
        Numerically compute the cumulative distribution function (CDF) at a given z-value for the
        sum of Bezier random variables.

        This method evaluates the CDF at the specified value z by integrating over the
        parameter t of the sum of Bezier random variables.

        Parameters
        ----------
        z : float
            The value at which to evaluate the cumulative distribution function.

        Returns
        -------
        float
            The computed CDF value at z.
        """
        def integrand(t: float) -> float:
            y_val = z - self.bezierv_x.poly_x(t)
            if y_val < self.bezierv_y.controls_x[0]:
                cumul_z = 0
            elif y_val > self.bezierv_y.controls_x[-1]:
                cumul_z = self.bezierv_x.pdf_numerator_t(t)
            else:
                y_inv = self.bezierv_y.root_find(y_val)
                cumul_z = self.bezierv_x.pdf_numerator_t(t) * self.bezierv_y.eval_t(y_inv)[1]
            return cumul_z

        result, _ = quad(integrand, 0, 1)
        return self.bezierv_x.n * result

    def conv(self, method='projgrad') -> tuple:
        """
        Numerically compute the convolution of two Bezier random variables.

        This method performs the convolution by:
          - Defining the lower and upper bounds for the convolution as the sum of the
            smallest and largest control points of bezierv_x and bezierv_y, respectively.
          - Generating a set of data points over these bounds.
          - Evaluating the cumulative distribution function (CDF) at each data point using the
            cdf_z method.
          - Initializing a ProjGrad instance with the computed data and empirical CDF values.
          - Fitting the convolution Bezier random variable via projected gradient descent and updating
            bezierv_conv.

        Parameters
        ----------
        method : str, optional
            The fitting method to use for the convolution. Default is 'projgrad'.
            Other options may include 'nonlinear' or 'neldermead'.

        Returns
        -------
        Bezierv
            The updated Bezierv instance representing the convolution of the two input Bezier 
            random variables.
        float
            The mean squared error (MSE) of the fit.
        """
        low_bound = self.bezierv_x.controls_x[0] + self.bezierv_y.controls_x[0]
        up_bound = self.bezierv_x.controls_x[-1] + self.bezierv_y.controls_x[-1]
        data = np.linspace(low_bound, up_bound, self.grid)

        emp_cdf_data = np.zeros(self.grid)
        for i in range(self.grid):
            emp_cdf_data[i] = self.cdf_z(data[i])

        if self.bezierv_x.n == self.bezierv_y.n:
            controls_x = self.bezierv_x.controls_x + self.bezierv_y.controls_x
            distfit = DistFit(data, emp_cdf_data=emp_cdf_data, init_x=controls_x, n=self.n)
            bezierv, mse = distfit.fit(method=method)
        else:
            distfit = DistFit(data, emp_cdf_data=emp_cdf_data, n=self.n, method_init_x='uniform')
            bezierv, mse = distfit.fit(method=method)

        return bezierv, mse
