import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.optimize import brentq

class ProjGrad:
    def __init__(self, bezierv, data):
        self.bezierv = bezierv
        self.data = np.sort(data)
        self.n = bezierv.n
        self.m = len(data)
        self.emp_cdf = ECDF(data)

    def grad(self, t, controls):
        """
        controls = [x, z]
        """
        n = self.n
        m = self.m
        data = self.data
        empirical_cdf = self.emp_cdf
        poly = self.bezierv.poly

        grad_x = np.zeros(n + 1)
        grad_z = np.zeros(n + 1)
        for j in range(m):
            inner_sum = np.zeros(n + 1)
            for i in range(n + 1):
                inner_sum[i] = self.bernstein(t[j], i)

            bern_eval = poly(t[j], controls)
            grad_x += 2 * (bern_eval[0][j]- data[j]) * inner_sum
            grad_z += 2 * (bern_eval[1][j] - empirical_cdf(data[j])) * inner_sum

        return grad_x, grad_z
    
    def poly_x_sample(self, t, controls_x, data_x):
        p_x = 0
        for i in range(self.n + 1):
            p_x += self.bezierv.bernstein(t, i) * controls_x[i]
        return p_x - data_x
    
    def root_find(self, controls_x, data_x):
        t = brentq(self.poly_x_sample, 0, 1, args=(controls_x, data_x))
        return t
    
    def project_z(self, control_z):
        z_prime = np.clip(control_z.copy(), a_min= 0, a_max=1)
        z_prime.sort()
        z_prime[0] = 0
        z_prime[-1] = 1
        return z_prime
    
    def project_x(self, control_x, data):
        x_prime = np.clip(control_x.copy(), a_min=data[0], a_max=data[-1])
        x_prime.sort()
        x_prime[0] = data[0]
        x_prime[-1] = data[-1]
        return x_prime

    def fit(self, controls_0, step = 0.01, maxiter = 1000):
        m = self.m
        n = self.n
        data = self.data

        t_0 = np.zeros(m)
        for j in range(m):
            t_0[j] = self.root_find(controls_0[0], data[j])
        
        controls = controls_0
        x = controls[0]
        z = controls[1]
        t = t_0

        for i in range(maxiter):
            grad_x, grad_z = self.grad(t, data, controls)
            x_prime = self.project_x(x - step * grad_x, data)
            z_prime = self.project_z(z - step * grad_z)

            t_prime = np.zeros(m)
            for j in range(m):
                t_prime[j] = self.root_find(x_prime, data[j])
            
            if np.linalg.norm(x_prime - x) < 1e-4 and np.linalg.norm(z_prime - z) < 1e-4 and np.linalg.norm(t_prime - t) < 1e-4:
                x = x_prime
                z = z_prime
                controls = np.array([x, z])
                t = t_prime
                break
            
            x = x_prime
            z = z_prime
            controls = np.array([x, z])
            t = t_prime
        
        fit_error = 0
        for j in range(m):
            fit_error += (self.bezierv.poly(t[j], controls)[1] - data[j])**2
        
        return controls, t, fit_error