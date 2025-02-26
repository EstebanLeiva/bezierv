import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.optimize import brentq

from bezierv.classes.bezierv import Bezierv

class ProjGrad:
    def __init__(self, bezierv: Bezierv, data: np.array):
        self.bezierv = bezierv
        self.data = np.sort(data)
        self.n = bezierv.n
        self.m = len(data)
        self.emp_cdf = ECDF(data)

    def grad(self, t: float, controls_x: np.array, controls_z: np.array):
        n = self.n
        m = self.m
        data = self.data
        empirical_cdf = self.emp_cdf
        poly_x = self.bezierv.poly_x
        poly_z = self.bezierv.poly_z

        grad_x = np.zeros(n + 1)
        grad_z = np.zeros(n + 1)
        for j in range(m):
            inner_sum = np.zeros(n + 1)
            for i in range(n + 1):
                inner_sum[i] = self.bezierv.bernstein(t[j], i)

            grad_x += 2 * (poly_x(t[j], controls_x)- data[j]) * inner_sum
            grad_z += 2 * (poly_z(t[j], controls_z) - empirical_cdf(data[j])) * inner_sum

        return grad_x, grad_z
    
    def poly_x_sample(self, t, controls_x, data_x):
        p_x = 0
        for i in range(self.n + 1):
            p_x += self.bezierv.bernstein(t, i) * controls_x[i]
        return p_x - data_x
    
    def root_find(self, controls_x, data_x):
        t = brentq(self.poly_x_sample, 0, 1, args=(controls_x, data_x))
        return t
    
    def project_z(self, controls_z):
        z_prime = np.clip(controls_z.copy(), a_min= 0, a_max=1)
        z_prime.sort()
        z_prime[0] = 0
        z_prime[-1] = 1
        return z_prime
    
    def project_x(self, controls_x, data):
        x_prime = np.clip(controls_x.copy(), a_min=data[0], a_max=data[-1])
        x_prime.sort()
        x_prime[0] = data[0]
        x_prime[-1] = data[-1]
        return x_prime

    def fit(self, controls_x0, controls_z0, step = 0.01, maxiter = 10000):
        m = self.m
        data = self.data
        poly_z = self.bezierv.poly_z

        t_0 = np.zeros(m)
        for j in range(m):
            t_0[j] = self.root_find(controls_x0, data[j])
        
        x = controls_x0
        z = controls_z0
        t = t_0

        for i in range(maxiter):
            grad_x, grad_z = self.grad(t, x, z)
            x_prime = self.project_x(x - step * grad_x, data)
            z_prime = self.project_z(z - step * grad_z)

            t_prime = np.zeros(m)
            for j in range(m):
                t_prime[j] = self.root_find(x_prime, data[j])
            
            if np.linalg.norm(x_prime - x) < 1e-4 and np.linalg.norm(z_prime - z) < 1e-4 and np.linalg.norm(t_prime - t) < 1e-4:
                x = x_prime
                z = z_prime
                t = t_prime
                print(f'Converged in {i} iterations')
                break
            
            x = x_prime
            z = z_prime
            t = t_prime
        
        fit_error = 0
        for j in range(m):
            fit_error += (poly_z(t[j], z) - self.emp_cdf(data[j]))**2
        
        return x, z, t, fit_error