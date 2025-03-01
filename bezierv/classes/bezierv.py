import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.optimize import brentq, bisect
from statsmodels.distributions.empirical_distribution import ECDF

class Bezierv:
    def __init__(self, n: int, controls_x=None, controls_z=None):
        self.n = n
        self.deltas_x = np.zeros(n)
        self.deltas_z = np.zeros(n)
        self.comb = np.zeros(n + 1)
        self.comb_minus = np.zeros(n)

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
        self.var = math.nan
        self.skew = math.nan
        self.kurt = math.nan

        # initialize
        self.combinations()

    def update_bezierv(self, controls_x, controls_z):
        """
        Update the control points for the Bezier curve and recalculate deltas.

        This method sets the control points for the curve by updating the instance
        attributes `controls_x` and `controls_z` with the provided values. It then
        calls the `deltas` method to update any dependent computations.

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
        n = self.n
        for i in range(n):
            self.deltas_x[i] = self.controls_x[i + 1] - self.controls_x[i]
            self.deltas_z[i] = self.controls_z[i + 1] - self.controls_z[i]

    def bernstein(self, t, i, combinations, n):
        return combinations[i] * t**i * (1 - t)**(n - i)

    def poly_x(self, t, controls_x = None):
        if controls_x is None:
            controls_x = self.controls_x
        n = self.n
        p_x = 0
        for i in range(n + 1):
           p_x  += self.bernstein(t, i, self.comb, self.n) * controls_x[i]
        return p_x
    
    def poly_z(self, t, controls_z = None):
        if controls_z is None:
            controls_z = self.controls_z
        n = self.n
        p_z = 0
        for i in range(n + 1):
           p_z  += self.bernstein(t, i, self.comb, self.n) * controls_z[i]
        return p_z
    
    def root_find(self, x, method='brentq'):
        def poly_x_zero(t, x):
            return self.poly_x(t) - x
        if method == 'brentq':
            t = brentq(poly_x_zero, 0, 1, args=(x,))
        elif method == 'bisect':
            t = bisect(poly_x_zero, 0, 1, args=(x,))
        return t

    def eval_t(self, t):
        n = self.n
        p_x = 0
        p_z = 0
        for i in range(n + 1):
            p_x += self.comb[i] * t**i * (1 - t)**(n - i) * self.controls_x[i]
            p_z += self.comb[i] * t**i * (1 - t)**(n - i) * self.controls_z[i]
        return p_x, p_z
    
    def eval_x(self, x):
        t = self.root_find(x)
        return self.eval_t(t)
    
    def pdf_t(self, t):
        n = self.n
        pdf_num_z = 0
        pdf_denom_x = 0
        for i in range(n):
            pdf_num_z += self.bernstein(t, i, self.comb_minus, n - 1) * self.deltas_z[i]
            pdf_denom_x += self.bernstein(t, i, self.comb_minus, n - 1) * self.deltas_x[i]
        return pdf_num_z/pdf_denom_x
    
    def pdf_x(self, x):
        t = self.root_find(x)
        return self.pdf_t(t)
    
    def pdf_numerator(self, t):
        pdf_num_z = 0
        for i in range(self.n):
            pdf_num_z += self.bernstein(t, i, self.comb_minus, self.n - 1) * self.deltas_z[i]
        return pdf_num_z
    
    def plot_cdf(self, data, ecdf=True):
        x_bezier = np.zeros(len(data))
        cdf_x_bezier = np.zeros(len(data))

        for i in range(len(data)):
            p_x, p_z = self.eval_x(data[i])
            x_bezier[i] = p_x
            cdf_x_bezier[i] = p_z

        if ecdf:
            ecdf =ECDF(data)
            plt.plot(data, ecdf(data), label='Empirical CDF', linestyle='--', color='black')
            
        plt.plot(x_bezier, cdf_x_bezier, label='Bezier CDF', linestyle='--')
        plt.scatter(self.controls_x, self.controls_z, label='Control Points', color='red')
        plt.legend()
        plt.show()

    def plot_pdf(self, data):
        x_bezier = np.zeros(len(data))
        pdf_x_bezier = np.zeros(len(data))

        for i in range(len(data)):
            p_x, _ = self.eval_x(data[i])
            x_bezier[i] = p_x
            pdf_x_bezier[i] = self.pdf_x(data[i])

        plt.scatter(x_bezier, pdf_x_bezier, color='red')
        plt.legend()
        plt.show()






        
    
    
    

