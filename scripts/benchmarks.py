import numpy as np
import time
import matplotlib.pyplot as plt

from bezierv.classes.bezierv import Bezierv
from bezierv.algorithms.non_linear_solver import NonLinearSolver
from bezierv.algorithms.proj_grad import ProjGrad

def run_benchmark_n():
    np.random.seed(123)
    times_proj_grad = []
    mse_proj_grad = []
    times_non_linear_solver = []
    mse_non_linear_solver = []

    for n in range(5, 20, 5):
        x = np.random.normal(1, 1, 1000)

        bezierv_x = Bezierv(n)
        bezierv_y = Bezierv(n)

        controls_z0 = np.linspace(0, 1, n + 1)

        non_linear_solver = NonLinearSolver(bezierv_x, x)
        start_time = time.time()
        bezierv_x_fitted = non_linear_solver.fit()
        elapsed = time.time() - start_time
        times_non_linear_solver.append(elapsed)
        mse_non_linear_solver.append(non_linear_solver.mse)

        proj_grad = ProjGrad(bezierv_y, x)
        start_time = time.time()
        bezierv_y_fitted = proj_grad.fit(controls_z0)
        elapsed = time.time() - start_time
        times_proj_grad.append(elapsed)
        mse_proj_grad.append(proj_grad.mse)
    
    return times_proj_grad, mse_proj_grad, times_non_linear_solver, mse_non_linear_solver

def run_benchmark_distributions():
    np.random.seed(123)
    times_proj_grad = []
    mse_proj_grad = []

    n = 7
    controls_z0 = np.linspace(0, 1, n + 1)
    #normal
    x = np.random.normal(1, 1, 1000)
    bezierv_x = Bezierv(n)
    proj_grad = ProjGrad(bezierv_x, x)
    start_time = time.time()
    bezierv_x_fitted = proj_grad.fit(controls_z0)
    elapsed = time.time() - start_time
    times_proj_grad.append(elapsed)
    mse_proj_grad.append(proj_grad.mse)

    #uniform
    x = np.random.uniform(0, 1, 1000)
    bezierv_x = Bezierv(n)
    proj_grad = ProjGrad(bezierv_x, x)
    start_time = time.time()
    bezierv_x_fitted = proj_grad.fit(controls_z0)
    elapsed = time.time() - start_time
    times_proj_grad.append(elapsed)
    mse_proj_grad.append(proj_grad.mse)

    #exponential
    x = np.random.exponential(1, 1000)
    bezierv_x = Bezierv(n)
    proj_grad = ProjGrad(bezierv_x, x)
    start_time = time.time()
    bezierv_x_fitted = proj_grad.fit(controls_z0)
    elapsed = time.time() - start_time
    times_proj_grad.append(elapsed)
    mse_proj_grad.append(proj_grad.mse)

    # lognormal
    x = np.random.lognormal(0, 1, 1000)
    bezierv_x = Bezierv(n)
    proj_grad = ProjGrad(bezierv_x, x)
    start_time = time.time()
    bezierv_x_fitted = proj_grad.fit(controls_z0)
    elapsed = time.time() - start_time
    times_proj_grad.append(elapsed)
    mse_proj_grad.append(proj_grad.mse)

    # gamma
    x = np.random.gamma(1, 1, 1000)
    bezierv_x = Bezierv(n)
    proj_grad = ProjGrad(bezierv_x, x)
    start_time = time.time()
    bezierv_x_fitted = proj_grad.fit(controls_z0)
    elapsed = time.time() - start_time
    times_proj_grad.append(elapsed)
    mse_proj_grad.append(proj_grad.mse)

    #bimodal distribution
    x = np.concatenate([np.random.normal(1, 1, 500), np.random.normal(3, 1, 500)])
    bezierv_x = Bezierv(n)
    proj_grad = ProjGrad(bezierv_x, x)
    start_time = time.time()
    bezierv_x_fitted = proj_grad.fit(controls_z0)
    elapsed = time.time() - start_time
    times_proj_grad.append(elapsed)

    return times_proj_grad, mse_proj_grad

times_proj_grad, mse_proj_grad, times_non_linear_solver, mse_non_linear_solver = run_benchmark_n()

print("##############################################")
print("Benchmark n results")
print("##############################################")
print("times_proj_grad:", times_proj_grad)
print("mse_proj_grad:", mse_proj_grad)
print("times_non_linear_solver:", times_non_linear_solver)
print("mse_non_linear_solver:", mse_non_linear_solver)

times_proj_grad, mse_proj_grad = run_benchmark_distributions()

print("##############################################")
print("Benchmark distributions results")
print("##############################################")
print("times_proj_grad:", times_proj_grad)
print("mse_proj_grad:", mse_proj_grad)