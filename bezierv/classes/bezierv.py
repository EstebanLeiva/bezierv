import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.optimize import brentq, bisect
from scipy.integrate import quad
from statsmodels.distributions.empirical_distribution import ECDF

class Bezierv:
    def __init__(self, n: int, controls_x=None, controls_z=None):
        """
        Initialize a Bezierv instance representing a Bezier random variable.

        This constructor initializes the Bezier curve with a given number of control points `n` and 
        optionally with provided control points for the x and z coordinates. If control points are not 
        provided, they are initialized as zero arrays. It also sets up auxiliary arrays for differences 
        (deltas) and binomial coefficients, and initializes moment attributes to NaN.

        Parameters
        ----------
        n : int
            The number of control points of the Bezier random variable.
        controls_x : array-like, optional
            The x-coordinates of the control points (length n+1). If None, a zero array is created.
        controls_z : array-like, optional
            The z-coordinates of the control points (length n+1). If None, a zero array is created.

        Raises
        ------
        ValueError
            If only one of controls_x or controls_z is provided.

        Attributes
        ----------
        n : int
            The number of control points of the Bezier random variables.
        deltas_x : np.array
            Differences between consecutive x control points.
        deltas_z : np.array
            Differences between consecutive z control points.
        comb : np.array
            Array of binomial coefficients (n).
        comb_minus : np.array
            Array of binomial coefficients (n - 1).
        support : tuple
            The support of the Bezier random variable (initialized as (-inf, inf)).
        controls_x : np.array
            x-coordinates of the control points.
        controls_z : np.array
            z-coordinates of the control points.
        mean : float
            The mean of the curve (initialized as NaN).
        var : float
            The variance of the curve (initialized as NaN).
        skew : float
            The skewness of the curve (initialized as NaN).
        kurt : float
            The kurtosis of the curve (initialized as NaN).
        """
        self.n = n
        self.deltas_x = np.zeros(n)
        self.deltas_z = np.zeros(n)
        self.comb = np.zeros(n + 1)
        self.comb_minus = np.zeros(n)
        self.support = (-np.inf, np.inf)

        if controls_x is None and controls_z is None:
            self.controls_x = np.zeros(n + 1)
            self.controls_z = np.zeros(n + 1)
        elif controls_x is not None and controls_z is not None:
            self.controls_x = controls_x
            self.controls_z = controls_z
        else:
            raise ValueError('Either all or none of the parameters controls_x and controls_y must be provided')

        # moments
        self.mean = math.nan
        self.variance = math.nan
        self.skewness = math.nan
        self.kurtosis = math.nan

        # initialize
        self.combinations()

    def update_bezierv(self, controls_x, controls_z, bounds):
        """
        Update the control points for the Bezier curve, bounds and recalculate deltas.

        Parameters
        ----------
        controls_x : array-like
            The new x-coordinates of the control points.
        controls_z : array-like
            The new z-coordinates of the control points.

        Returns
        -------
        None
        """
        self.controls_x = controls_x
        self.controls_z = controls_z
        self.bounds = bounds
        self.deltas()

    def combinations(self):
        """
        Compute and store binomial coefficients.

        This method computes two sets of binomial coefficients:
        
        - For each integer i in the range [0, self.n] (inclusive), it calculates
        "self.n choose i" using math.comb and assigns the result to self.comb[i].
        - For each integer i in the range [0, self.n - 1], it calculates
        "(self.n - 1) choose i" and assigns the result to self.comb_minus[i].

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        n = self.n
        for i in range(0, n + 1):
            self.comb[i] = math.comb(n, i)
            if i < n:
                self.comb_minus[i] = math.comb(n - 1, i)

    def deltas(self):
        """
        Compute the differences between consecutive control points.

        This method updates the attributes `deltas_x` and `deltas_z` by computing the difference
        between successive control points in `controls_x` and `controls_z`, respectively.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        n = self.n
        for i in range(n):
            self.deltas_x[i] = self.controls_x[i + 1] - self.controls_x[i]
            self.deltas_z[i] = self.controls_z[i + 1] - self.controls_z[i]

    def bernstein(self, t, i, combinations, n):
        """
        Compute the Bernstein basis polynomial value.

        Parameters
        ----------
        t : float
            The parameter value (in the interval [0, 1]).
        i : int
            The index of the Bernstein basis polynomial.
        combinations : array-like
            An array of binomial coefficients to use in the computation.
        n : int
            The degree for the Bernstein polynomial.

        Returns
        -------
        float
            The value of the Bernstein basis polynomial at t.
        """
        return combinations[i] * t**i * (1 - t)**(n - i)

    def poly_x(self, t, controls_x = None):
        """
        Evaluate the x-coordinate at a given t value.

        This method computes the weighted sum of the x control points using Bernstein basis
        polynomials. If `controls_x` is not provided, the instance's control points are used.

        Parameters
        ----------
        t : float
            The parameter value at which to evaluate (in [0, 1]).
        controls_x : array-like, optional
            An array of control points for the x-coordinate. Defaults to self.controls_x.

        Returns
        -------
        float
            The evaluated x-coordinate at t.
        """
        if controls_x is None:
            controls_x = self.controls_x
        n = self.n
        p_x = 0
        for i in range(n + 1):
           p_x  += self.bernstein(t, i, self.comb, self.n) * controls_x[i]
        return p_x
    
    def poly_z(self, t, controls_z = None):
        """
        Evaluate the z-coordinate at a given t value.

        This method computes the weighted sum of the z control points using Bernstein basis
        polynomials. If `controls_z` is not provided, the instance's control points are used.

        Parameters
        ----------
        t : float
            The parameter value at which to evaluate the curve (typically in [0, 1]).
        controls_z : array-like, optional
            An array of control points for the z-coordinate. Defaults to self.controls_z.

        Returns
        -------
        float
            The evaluated z-coordinate at t.
        """
        if controls_z is None:
            controls_z = self.controls_z
        n = self.n
        p_z = 0
        for i in range(n + 1):
           p_z  += self.bernstein(t, i, self.comb, self.n) * controls_z[i]
        return p_z
    
    def root_find(self, x, method='brentq'):
        """
        Find t such that the Bezier curve's x-coordinate equals a given value.

        This method solves for the root of the equation poly_x(t) - x = 0 using a specified root-finding
        algorithm. The search is performed in the interval [0, 1].

        Parameters
        ----------
        x : float
            The x-coordinate for which to find the corresponding parameter t.
        method : {'brentq', 'bisect'}, optional
            The root-finding method to use. Default is 'brentq'.

        Returns
        -------
        float
            The parameter t in the interval [0, 1] such that poly_x(t) is approximately equal to x.
        """
        def poly_x_zero(t, x):
            return self.poly_x(t) - x
        if method == 'brentq':
            t = brentq(poly_x_zero, 0, 1, args=(x,))
        elif method == 'bisect':
            t = bisect(poly_x_zero, 0, 1, args=(x,))
        return t

    def eval_t(self, t):
        """
        Evaluate the CDF of the Bezier random variable at a given parameter value t.

        This method computes both the x and z coordinates of the Bezier curve at parameter t
        using the Bernstein basis polynomials and the current control points.

        Parameters
        ----------
        t : float
            The parameter value at which to evaluate the curve (typically in [0, 1]).

        Returns
        -------
        tuple of floats
            A tuple (p_x, p_z) where p_x is the evaluated x-coordinate and p_z is the evaluated z-coordinate.
        """
        n = self.n
        p_x = 0
        p_z = 0
        for i in range(n + 1):
            p_x += self.comb[i] * t**i * (1 - t)**(n - i) * self.controls_x[i]
            p_z += self.comb[i] * t**i * (1 - t)**(n - i) * self.controls_z[i]
        return p_x, p_z
    
    def eval_x(self, x):
        """
        Evaluate the CDF of the Bezier random variable at a given x-coordinate.

        This method first finds the corresponding parameter value t such that the Bezier curve's x-coordinate
        is equal to the given x, and then evaluates the curve at that parameter.

        Parameters
        ----------
        x : float
            The x-coordinate at which to evaluate the Bezier curve.

        Returns
        -------
        tuple of floats
            A tuple (p_x, p_z) where p_x is the x-coordinate and p_z is the z-coordinate of the curve at x.
        """
        t = self.root_find(x)
        return self.eval_t(t)
    
    def cdf_x(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given x-coordinate.

        This method evaluates the CDF of the Bezier random variable at the specified x-coordinate
        by finding the corresponding parameter t and then computing the z-coordinate at that t.

        Parameters
        ----------
        x : float
            The x-coordinate at which to evaluate the CDF.

        Returns
        -------
        float
            The CDF value at the given x-coordinate.
        """
        if x < self.controls_x[0]:
            return 0
        if x > self.controls_x[-1]:
            return 1
        _, p_z = self.eval_x(x)
        return p_z

    def get_quantile(self, alpha, method='brentq'):
        """
        Compute the quantile function (inverse CDF) for a given probability level alpha.

        This method finds the value of t such that the cumulative distribution function (CDF) at t equals alpha,
        and then evaluates the x-coordinate at that t.

        Parameters
        ----------
        alpha : float
            The probability level for which to compute the quantile (in [0, 1]).

        Returns
        -------
        float
            The quantile value corresponding to the given alpha.
        """
        def cdf_t(t, alpha):
            return self.poly_z(t) - alpha
        
        if method == 'brentq':
            t = brentq(cdf_t, 0, 1, args=(alpha,))
        elif method == 'bisect':
            t = bisect(cdf_t, 0, 1, args=(alpha,))
        return self.poly_x(t)
    
    def pdf_t(self, t):
        """
        Compute the probability density function (PDF) of the Bezier random variable with respect to t.

        Parameters
        ----------
        t : float
            The value at which to compute the PDF (in [0, 1]).

        Returns
        -------
        float
            The computed PDF value at t.
        """
        n = self.n
        pdf_num_z = 0
        pdf_denom_x = 0
        for i in range(n):
            pdf_num_z += self.bernstein(t, i, self.comb_minus, n - 1) * self.deltas_z[i]
            pdf_denom_x += self.bernstein(t, i, self.comb_minus, n - 1) * self.deltas_x[i]
        return pdf_num_z/pdf_denom_x
    
    def pdf_x(self, x):
        """
        Compute the probability density function (PDF) of the Bezier random variable at a given x.

        This method finds the corresponding parameter t for the x-coordinate using `root_find`, and then
        computes the PDF with respect to t.

        Parameters
        ----------
        x : float
            The x-coordinate at which to evaluate the PDF.

        Returns
        -------
        float
            The computed PDF value at x.
        """
        t = self.root_find(x)
        return self.pdf_t(t)
    
    def pdf_numerator(self, t):
        """
        Compute the numerator part of the PDF for the Bezier random variable with respect to t.

        Parameters
        ----------
        t : float
            The value at which to compute the PDF numerator (in [0, 1]).

        Returns
        -------
        float
            The numerator of the PDF at t.
        """
        pdf_num_z = 0
        for i in range(self.n):
            pdf_num_z += self.bernstein(t, i, self.comb_minus, self.n - 1) * self.deltas_z[i]
        return pdf_num_z

    def get_mean(self):
        """
        Compute and return the mean of the distribution.

        This method calculates the mean by integrating x * pdf(x) over the interval
        defined by self.bounds. The integration is performed using the quad function.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The mean value of the distribution.
        """
        a, b = self.bounds
        self.mean, _ = quad(lambda x: x * self.pdf_x(x), a, b)
        return self.mean

    def get_variance(self):
        #TODO: check, it is not giving correct values
        a, b = self.bounds
        E_x2, _ = quad(lambda x: x**2 * self.pdf_x(x), a, b)
        if self.mean == math.nan:
            self.variance = E_x2 - self.mean()**2
        else:
            self.variance = E_x2 - self.mean**2
        return self.variance

    def get_skewness(self):
        #TODO: check, it is not giving correct values
        a, b = self.bounds
        if self.mean == math.nan:
            self.mean()
        if self.variance == math.nan:
            self.variance()
        mu_3, _ = quad(lambda x: (x - self.mean)**3 * self.pdf_x(x), a, b)
        self.skewness = mu_3 / self.variance**(3/2)
        return self.skewness

    def get_kurtosis(self):
        #TODO: check, it is not giving correct values
        a, b = self.bounds
        if self.mean == math.nan:
            self.mean()
        if self.variance == math.nan:
            self.variance()
        mu_4, _ = quad(lambda x: (x - self.mean)**4 * self.pdf_x(x), a, b)
        self.kurtosis = mu_4 / self.variance**2
        return self.kurtosis
    
    def check_ordering(self):
        """
        Check if the control points are ordered in non-decreasing order.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if the control points are ordered in non-decreasing order, False otherwise.
        """
        if not np.all(np.diff(self.controls_x) >= 0):
            raise False
        if not np.all(np.diff(self.controls_z) >= 0):
            raise False
        return True
    
    def plot_cdf(self, data=None, num_points=100, ax=None):
        """
        Plot the cumulative distribution function (CDF) of the Bezier random variable alongside 
        the empirical CDF (if data is provided).

        If no data is provided, a linspace is generated based on the minimum and maximum of the 
        control points in the x-direction.

        Parameters
        ----------
        data : array-like, optional
            The data points at which to evaluate and plot the CDF. If None, a linspace is used.
        num_points : int, optional
            The number of points to use in the linspace when data is not provided (default is 100).

        Returns
        -------
        None
        """
        data_bool = True
        if data is None:
            data_bool = False
            data = np.linspace(np.min(self.controls_x), np.max(self.controls_x), num_points)
        
        x_bezier = np.zeros(len(data))
        cdf_x_bezier = np.zeros(len(data))

        for i in range(len(data)):
            p_x, p_z = self.eval_x(data[i])
            x_bezier[i] = p_x
            cdf_x_bezier[i] = p_z

        if ax is None:
            ax = plt.gca() 

        if data_bool:
            ecdf_fn = ECDF(data)
            ax.plot(data, ecdf_fn(data), label='Empirical cdf', linestyle='--', color='black')

        ax.plot(x_bezier, cdf_x_bezier, label='Bezier cdf', linestyle='--')
        ax.scatter(self.controls_x, self.controls_z, label='Control Points', color='red')
        ax.legend()

    def plot_pdf(self, data=None, num_points=100, ax=None):
        """
        Plot the probability density function (PDF) of the Bezier random variable.

        If no data is provided, a linspace is generated based on the minimum and maximum of the 
        control points in the x-direction.

        Parameters
        ----------
        data : array-like, optional
            The data points at which to evaluate and plot the PDF. If None, a linspace is used.
        num_points : int, optional
            The number of points to use in the linspace when data is not provided (default is 100).

        Returns
        -------
        None
        """
        if data is None:
            data = np.linspace(np.min(self.controls_x), np.max(self.controls_x), num_points)

        x_bezier = np.zeros(len(data))
        pdf_x_bezier = np.zeros(len(data))

        if ax is None:
            ax = plt.gca()

        for i in range(len(data)):
            p_x, _ = self.eval_x(data[i])
            x_bezier[i] = p_x
            pdf_x_bezier[i] = self.pdf_x(data[i])

        ax.plot(x_bezier, pdf_x_bezier, label='Bezier pdf', linestyle='-')
        ax.legend()