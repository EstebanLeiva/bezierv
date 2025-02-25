import numpy as np
import math

from scipy.optimize import brentq, bisect

class Bezierv:
    def __init__(self, n, controls, deltas, supp_a, supp_b):
        self.n = n
        self.controls = controls
        self.deltas = deltas
        self.supp_a = supp_a
        self.supp_b = supp_b
        self.comb = np.zeros(n + 1)

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

    def bernstein(self, t, i):
        n = self.n
        return self.comb[i] * t**i * (1 - t)**(n - i)

    def poly(self, t, controls = None):
        if controls is None:
            controls = self.controls
        n = self.n
        tup = np.zeros(2)
        for i in range(n + 1):
           tup  += self.bernstein(t, i) * controls[i]
        return tup

    def eval_t(self, t):
        n = self.n
        p = np.zeros(2)
        for i in range(n + 1):
            p += self.comb[i] * t**i * (1 - t)**(n - i) * self.controls[i]
        return p
    
    def eval_x(self, x, method='brentq'):
        def poly_x_zero(t, x):
            return self.poly(t)[0] - x
        
        if method == 'brentq':
            t = brentq(poly_x_zero, 0, 1, args=(x,))
        elif method == 'bisect':
            t = bisect(poly_x_zero, 0, 1, args=(x,))
        
        return self.eval_t(t)
            


        
    
    
    

