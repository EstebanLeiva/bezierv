# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bezierv.classes.bezierv import Bezierv
from bezierv.algorithms.non_linear_solver import NonLinearSolver

# %%
np.random.seed(123)
x = np.sort(np.random.normal(1, 1, 60))
y = np.sort(np.random.normal(1, 1, 100))

# %%
n = 10
bezierv_x = Bezierv(n)
bezierv_y = Bezierv(n)

# %%
non_linear_solver = NonLinearSolver(bezierv_x, x)
bezierv_x_fitted = non_linear_solver.fit()

bezierv_x_fitted.plot_cdf(x, ecdf=None)


