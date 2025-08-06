import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.optimize import brentq, bisect
from scipy.integrate import quad
from statsmodels.distributions.empirical_distribution import ECDF

from numpy import tanh, arctanh

class Bezierv:
    def __init__(self, 
                 n: int, 
                 controls_x=None, 
                 controls_z=None):
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
            The mean of the curve (initialized as np.inf).
        var : float
            The variance of the curve (initialized as np.inf).
        skew : float
            The skewness of the curve (initialized as np.inf).
        kurt : float
            The kurtosis of the curve (initialized as np.inf).
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
            controls_x = np.asarray(controls_x, dtype=float)
            controls_z = np.asarray(controls_z, dtype=float)
            self._validate_lengths(controls_x, controls_z)
            self._validate_ordering(controls_x, controls_z)
            self.controls_x = controls_x
            self.controls_z = controls_z
            self.support = (controls_x[0], controls_x[-1])
        else:
            raise ValueError('Either all or none of the parameters controls_x and controls_y must be provided')

        self.mean = np.inf
        self.variance = np.inf
        self.skewness = np.inf
        self.kurtosis = np.inf

        self.combinations()
        self.deltas()

    def update_bezierv(self, 
                       controls_x: np.array, 
                       controls_z: np.array):
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
        controls_x = np.asarray(controls_x, dtype=float)
        controls_z = np.asarray(controls_z, dtype=float)
        self._validate_lengths(controls_x, controls_z)
        self._validate_ordering(controls_x, controls_z)

        self.controls_x = controls_x
        self.controls_z = controls_z
        self.support = (controls_x[0], controls_x[-1])

        self.deltas()

    def combinations(self):
        """
        Compute and store binomial coefficients.

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

        Parameters
        ----------
        t : float
            The parameter value at which to evaluate (in [0, 1]).
        controls_x : np.array, optional
            An array of control points for the x-coordinate. Defaults to self.controls_x.

        Returns
        -------
        float
            The evaluated x-coordinate at t.
        """
        if controls_x is None:
            self._ensure_initialized()
            controls_x = self.controls_x
        n = self.n
        p_x = 0
        for i in range(n + 1):
           p_x  += self.bernstein(t, i, self.comb, self.n) * controls_x[i]
        return p_x
    
    def poly_z(self, t, controls_z = None):
        """
        Evaluate the z-coordinate at a given t value.

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
            self._ensure_initialized()
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
        self._ensure_initialized()
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

        Parameters
        ----------
        t : float
            The parameter value at which to evaluate the curve (typically in [0, 1]).

        Returns
        -------
        tuple of floats
            A tuple (p_x, p_z) where p_x is the evaluated x-coordinate and p_z is the evaluated z-coordinate.
        """
        self._ensure_initialized()
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

        Parameters
        ----------
        x : float
            The x-coordinate at which to evaluate the Bezier curve.

        Returns
        -------
        tuple of floats
            A tuple (p_x, p_z) where p_x is the x-coordinate and p_z is the z-coordinate of the curve at x.
        """
        self._ensure_initialized()
        t = self.root_find(x)
        return self.eval_t(t)
    
    def cdf_x(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given x-coordinate.

        Parameters
        ----------
        x : float
            The x-coordinate at which to evaluate the CDF.

        Returns
        -------
        float
            The CDF value at the given x-coordinate.
        """
        self._ensure_initialized()
        if x < self.controls_x[0]:
            return 0
        if x > self.controls_x[-1]:
            return 1
        _, p_z = self.eval_x(x)
        return p_z

    def quantile(self, alpha, method='brentq'):
        """
        Compute the quantile function (inverse CDF) for a given probability level alpha.

        Parameters
        ----------
        alpha : float
            The probability level for which to compute the quantile (in [0, 1]).

        Returns
        -------
        float
            The quantile value corresponding to the given alpha.
        """
        self._ensure_initialized()
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
        self._ensure_initialized()
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

        Parameters
        ----------
        x : float
            The x-coordinate at which to evaluate the PDF.

        Returns
        -------
        float
            The computed PDF value at x.
        """
        self._ensure_initialized()
        t = self.root_find(x)
        return self.pdf_t(t)
    
    def pdf_numerator_t(self, t):
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
        self._ensure_initialized()
        pdf_num_z = 0
        for i in range(self.n):
            pdf_num_z += self.bernstein(t, i, self.comb_minus, self.n - 1) * self.deltas_z[i]
        return pdf_num_z

    def get_mean(self, closed_form: bool=True):
        """
        Compute and return the mean of the distribution.

        Parameters
        ----------
        closed_form : bool, optional
            If True, use a closed-form solution for the mean. If False, compute it numerically (default is True).

        Returns
        -------
        float
            The mean value of the distribution.
        """
        self._ensure_initialized()
        if self.mean == np.inf:
            if closed_form:
                total = 0.0
                for ell in range(self.n + 1):
                    inner_sum = 0.0
                    for i in range(self.n):
                        denom = math.comb(2 * self.n - 1, ell + i)
                        inner_sum += (self.comb_minus[i] / denom) * self.deltas_z[i]
                    total += self.comb[ell] * self.controls_x[ell] * inner_sum
                self.mean = 0.5 * total
            else:
                a, b = self.bounds
                self.mean, _ = quad(lambda x: x * self.pdf_x(x), a, b) 
        return self.mean

    def get_variance(self):
        self._ensure_initialized()
        a, b = self.bounds
        E_x2, _ = quad(lambda x: (x)**2 * self.pdf_x(x), a, b)
        if self.mean == np.inf:
            self.variance = E_x2 - self.get_mean()**2
        else:
            self.variance = E_x2 - self.mean**2
        return self.variance
    
    def random(self, 
               n_sims: int,
               *,
               rng: np.random.Generator | int | None = None):
        """
        Generate random samples from the Bezier random variable.

        This method generates `n_sims` random samples from the Bezier random variable by evaluating
        the inverse CDF at uniformly distributed random values in the interval [0, 1].

        Parameters
        ----------
        n_sims : int
            The number of random samples to generate.
        rng : numpy.random.Generator | int | None, optional
            Pseudorandom-number generator state.  If *None* (default), a new
            ``numpy.random.Generator`` is created with fresh OS entropy.  Any
            value accepted by :func:`numpy.random.default_rng` (e.g. a seed
            integer or :class:`~numpy.random.SeedSequence`) is also allowed.


        Returns
        -------
        np.array
            An array of shape (n_sims,) containing the generated random samples from the Bezier random variable.
        """
        self._ensure_initialized()
        rng = np.random.default_rng(rng)
        u = rng.uniform(0, 1, n_sims)
        samples = np.zeros(n_sims)
        for i in range(n_sims):
            samples[i] = self.quantile(u[i])
        return samples

    
    def plot_cdf(self, data=None, num_points=100, ax=None):
        """
        Plot the cumulative distribution function (CDF) of the Bezier random variable alongside 
        the empirical CDF (if data is provided).

        Parameters
        ----------
        data : array-like, optional
            The data points at which to evaluate and plot the CDF. If None, a linspace is used.
        num_points : int, optional
            The number of points to use in the linspace when data is not provided (default is 100).
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the CDF. If None, the current axes are used.

        Returns
        -------
        None
        """
        self._ensure_initialized()
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

        show = False
        if ax is None:
            ax = plt.gca()
            show = True

        if data_bool:
            ecdf_fn = ECDF(data)
            ax.plot(data, ecdf_fn(data), label='Empirical cdf', linestyle='--', color='black')

        ax.plot(x_bezier, cdf_x_bezier, label='Bezier cdf', linestyle='--')
        ax.scatter(self.controls_x, self.controls_z, label='Control Points', color='red')
        ax.legend()
        if show:
            plt.show()

    def plot_pdf(self, data=None, num_points=100, ax=None):
        """
        Plot the probability density function (PDF) of the Bezier random variable.

        Parameters
        ----------
        data : array-like, optional
            The data points at which to evaluate and plot the PDF. If None, a linspace is used.
        num_points : int, optional
            The number of points to use in the linspace when data is not provided (default is 100).
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the PDF. If None, the current axes are used.

        Returns
        -------
        None
        """
        self._ensure_initialized()
        if data is None:
            data = np.linspace(np.min(self.controls_x), np.max(self.controls_x), num_points)

        x_bezier = np.zeros(len(data))
        pdf_x_bezier = np.zeros(len(data))

        show = False
        if ax is None:
            ax = plt.gca()
            show = True

        for i in range(len(data)):
            p_x, _ = self.eval_x(data[i])
            x_bezier[i] = p_x
            pdf_x_bezier[i] = self.pdf_x(data[i])

        ax.plot(x_bezier, pdf_x_bezier, label='Bezier pdf', linestyle='-')
        ax.legend()
        if show:
            plt.show()

    def _validate_lengths(self, controls_x, controls_z):
        if len(controls_x) != len(controls_z):
            raise ValueError("controls_x and controls_z must have the same length.")
        if len(controls_x) != self.n + 1:
            raise ValueError(f"controls arrays must have length n+1 (= {self.n + 1}).")

    def _validate_ordering(self, controls_x, controls_z):
        if np.any(np.diff(controls_x) < 0):
            raise ValueError("controls_x must be nondecreasing.")
        if np.any(np.diff(controls_z) < 0):
            raise ValueError("controls_z must be nondecreasing.")

    def _ensure_initialized(self):
        if np.allclose(self.controls_x, 0) and np.allclose(self.controls_z, 0):
            raise RuntimeError(
                "Bezier controls are all zeros (placeholder). "
                "Provide valid controls in the constructor or call update_bezierv()."
            )