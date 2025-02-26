import numpy as np
import math

from scipy.optimize import brentq, bisect

class Bezierv:
    def __init__(self, n: int, controls_x=None, controls_z=None, supp_a=None, supp_b=None):
        self.n = n
        self.deltas_x = np.zeros(n)
        self.deltas_z = np.zeros(n)
        self.comb = np.zeros(n + 1)

        if controls_x is None and controls_z is None and supp_a is None and supp_b is None:
            self.controls_x = np.zeros(n + 1)
            self.controls_z = np.zeros(n + 1)
            self.supp_a = math.nan
            self.supp_b = math.nan
        elif controls_x is not None and controls_z is not None and supp_a is not None and supp_b is not None:
            self.controls_x = controls_x
            self.controls_z = controls_z
            self.supp_a = supp_a
            self.supp_b = supp_b
        else:
            raise ValueError('Either all or none of the parameters controls, supp_a, and supp_b must be provided')

        # moments
        self.mean = math.nan
        self.var = math.nan
        self.skew = math.nan
        self.kurt = math.nan

        #initialize
        self.combinations()

    def combinations(self):
        n = self.n
        self.comb[0] = 1
        for i in range(1, n + 1):
            self.comb[i] = self.comb[i - 1] * (n - i + 1) // i

    def deltas(self):
        n = self.n
        for i in range(n):
            self.deltas_x[i] = self.controls_x[i + 1] - self.controls_x[i]
            self.deltas_z[i] = self.controls_z[i + 1] - self.controls_z[i]

    def bernstein(self, t, i):
        n = self.n
        return self.comb[i] * t**i * (1 - t)**(n - i)

    def poly_x(self, t, controls_x = None):
        if controls_x is None:
            controls_x = self.controls_x
        n = self.n
        p_x = 0
        for i in range(n + 1):
           p_x  += self.bernstein(t, i) * controls_x[i]
        return p_x
    
    def poly_z(self, t, controls_z = None):
        if controls_z is None:
            controls_z = self.controls_z
        n = self.n
        p_z = 0
        for i in range(n + 1):
           p_z  += self.bernstein(t, i) * controls_z[i]
        return p_z

    def eval_t(self, t):
        n = self.n
        p_x = 0
        p_z = 0
        for i in range(n + 1):
            p_x += self.comb[i] * t**i * (1 - t)**(n - i) * self.controls_x[i]
            p_z += self.comb[i] * t**i * (1 - t)**(n - i) * self.controls_z[i]
        return p_x, p_z
    
    def eval_x(self, x, method='brentq'):
        def poly_x_zero(t, x):
            return self.poly_x(t)[0] - x
        
        if method == 'brentq':
            t = brentq(poly_x_zero, 0, 1, args=(x,))
        elif method == 'bisect':
            t = bisect(poly_x_zero, 0, 1, args=(x,))
        
        return self.eval_t(t)
            


        
    
    
    

