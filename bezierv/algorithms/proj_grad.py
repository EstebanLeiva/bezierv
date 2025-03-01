import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.optimize import brentq

from bezierv.classes.bezierv import Bezierv

class ProjGrad:
    def __init__(self, bezierv: Bezierv, controls_x: np.array, data: np.array, emp_cdf_data: np.array=None):
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
        t_data = np.zeros(self.m)
        for i in range(self.m):
            t_data[i] = self.root_find(self.controls_x, data[i])
        return t_data
    
    def root_find(self, controls_x, data_i):
        def poly_x_sample(t, controls_x, data_i):
            p_x = 0
            for i in range(self.n + 1):
                p_x += self.bezierv.bernstein(t, i, self.bezierv.comb, self.bezierv.n) * controls_x[i]
            return p_x - data_i
        t = brentq(poly_x_sample, 0, 1, args=(controls_x, data_i))
        return t
    
    def grad(self, t: float, controls_z: np.array):
        grad_z = np.zeros(self.n + 1)
        for j in range(self.m):
            inner_sum = np.zeros(self.n + 1)
            for i in range(self.n + 1):
                inner_sum[i] = self.bezierv.bernstein(t[j], i, self.bezierv.comb, self.n)

            grad_z += 2 * (self.bezierv.poly_z(t[j], controls_z) - self.emp_cdf_data[j]) * inner_sum

        return grad_z
    
    def project_z(self, controls_z: np.array):
        z_prime = np.clip(controls_z.copy(), a_min= 0, a_max=1)
        z_prime.sort()
        z_prime[0] = 0
        z_prime[-1] = 1
        return z_prime

    def fit(self, controls_z0: np.array, step = 0.01, maxiter = 1000):
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