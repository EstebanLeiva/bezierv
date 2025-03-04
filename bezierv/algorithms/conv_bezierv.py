import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from bezierv.classes.bezierv import Bezierv
from bezierv.algorithms.proj_grad import ProjGrad

class ConvBezier:
    def __init__(self, bezierv_x: Bezierv, bezierv_y: Bezierv, m: int):
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
        m : int
            The number of data points to generate for the convolution (more data points
            induce a better approximation).

        Attributes
        ----------
        bezierv_x : Bezierv
            The Bezierv instance representing the first random variable.
        bezierv_y : Bezierv
            The Bezierv instance representing the second random variable.
        bezierv_conv : Bezierv
            A Bezierv instance that will store the convolution result. Its number of control
            points is set to the maximum of control points between bezierv_x and bezierv_y.
        m : int
            The number of data points used in the convolution.
        """
        self.bezierv_x = bezierv_x
        self.bezierv_y = bezierv_y
        self.bezierv_conv = Bezierv(max(bezierv_x.n, bezierv_y.n))
        self.m = m

    def cdf_z (self, z: float):
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
        def integrand(t: float):
            y_val = z - self.bezierv_x.poly_x(t)
            if y_val < self.bezierv_y.controls_x[0]:
                cumul_z = 0
            elif y_val > self.bezierv_y.controls_x[-1]:
                cumul_z = self.bezierv_x.pdf_numerator(t)
            else:
                y_inv = self.bezierv_y.root_find(y_val)
                cumul_z = self.bezierv_x.pdf_numerator(t) * self.bezierv_y.eval_t(y_inv)[1]
            return cumul_z

        result, _ = quad(integrand, 0, 1)
        return self.bezierv_x.n * result

    def conv(self):
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

        Returns
        -------
        Bezierv
            The updated Bezierv instance representing the convolution of the two input Bezier 
            random variables.
        """
        low_bound = self.bezierv_x.controls_x[0] + self.bezierv_y.controls_x[0]
        up_bound = self.bezierv_x.controls_x[-1] + self.bezierv_y.controls_x[-1]
        data = np.linspace(low_bound, up_bound, self.m)

        emp_cdf_data = np.zeros(self.m)
        for i in range(self.m):
            emp_cdf_data[i] = self.cdf_z(data[i])

        if self.bezierv_x.n == self.bezierv_y.n:
            controls_x = self.bezierv_x.controls_x + self.bezierv_y.controls_x
        else:
            #TODO: Implement the case where the number of control points is different with quantiles
            raise NotImplementedError
        proj_grad = ProjGrad(self.bezierv_conv, data, controls_x, emp_cdf_data)
        controls_z = np.linspace(0, 1, self.bezierv_conv.n + 1)
        self.bezierv_conv = proj_grad.fit(controls_z)
        return self.bezierv_conv, emp_cdf_data, data
