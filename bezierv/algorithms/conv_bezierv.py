import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from bezierv.classes.bezierv import Bezierv
from bezierv.algorithms.proj_grad import ProjGrad

class ConvBezier:
    def __init__(self, bezierv_x: Bezierv, bezierv_y: Bezierv, m: int):
        self.bezierv_x = bezierv_x
        self.bezierv_y = bezierv_y
        self.bezierv_conv = Bezierv(max(bezierv_x.n, bezierv_y.n))
        self.m = m
        self.data = None

    def cdf_z (self, z: float):
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
        low_bound = self.bezierv_x.controls_x[0] + self.bezierv_y.controls_x[0]
        up_bound = self.bezierv_x.controls_x[-1] + self.bezierv_y.controls_x[-1]
        data = np.linspace(low_bound, up_bound, self.m)
        self.data = data

        emp_cdf_data = np.zeros(self.m)
        for i in range(self.m):
            emp_cdf_data[i] = self.cdf_z(data[i])

        controls_x = np.linspace(low_bound, up_bound, self.bezierv_conv.n + 1)
        proj_grad = ProjGrad(self.bezierv_conv, controls_x, data, emp_cdf_data)
        controls_z = np.linspace(0, 1, self.bezierv_conv.n + 1)
        self.bezierv_conv = proj_grad.fit(controls_z)
        return self.bezierv_conv
