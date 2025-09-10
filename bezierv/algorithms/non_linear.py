from pyexpat import model
import pyomo.environ as pyo
import numpy as np
from bezierv.classes.bezierv import Bezierv
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

def fit(n: int,
        m: int,
        data: np.array,
        bezierv: Bezierv,
        init_x: np.array,
        init_z: np.array,
        init_t: np.array,
        emp_cdf_data: np.array,
        solver: str) -> Bezierv:
        """Fits a Bézier random variable to empirical CDF data.

        This method uses a nonlinear optimization solver to find the optimal
        Bézier curve control points that best represent the empirical cumulative
        distribution function (CDF) of the provided data.

        Parameters
        ----------
        n : int
            The degree of the Bézier curve (number of control points - 1).
        m : int
            The number of empirical CDF data points.
        data : np.ndarray
            The sorted data points used to fit the Bézier distribution.
        bezierv : Bezierv
            An instance of the `Bezierv` class to be fitted.
        init_x : np.ndarray
            Initial guess for the x-coordinates of the control points.
        init_z : np.ndarray
            Initial guess for the z-coordinates of the control points.
        init_t : np.ndarray
            Initial guess for the Bézier 'time' parameters `t` in [0, 1].
        emp_cdf_data : np.ndarray
            The empirical CDF values corresponding to the `data` points.
        solver : str, optional
            The name of the solver to use for optimization. Defaults to 'ipopt'.

        Returns
        -------
        tuple[Bezierv, float]
            A tuple containing:
            - bezierv (Bezierv): The fitted `Bezierv` object with updated
              control points.
            - mse (float): The final mean squared error (MSE) of the fit.

        Raises
        ------
        Exception
            If the optimization solver fails to find a solution.

        Notes
        -----
        The optimization is subject to several constraints to ensure a valid
        CDF representation:
        - The control points and 'time' parameters are kept sorted.
        - Convexity constraints are applied to the control points.
        - The first and last control points are fixed to the data's range.
        - The first and last 'time' parameters are fixed to 0 and 1.
        """        
        # Defining the optimization model
        model = pyo.ConcreteModel()

        # Sets
        model.N = pyo.Set(initialize=list(range(n + 1))) # N = 0,...,i,...,n
        model.N_n = pyo.Set(initialize=list(range(n))) # N = 0,...,i,...,n-1
        model.M = pyo.Set(initialize=list(range(1, m + 1))) # M = 1,...,j,...,m
        model.M_m = pyo.Set(initialize=list(range(1, m))) # M = 1,...,j,...,m-1

        # Decision variables
        # Control points. Box constraints.
        X_min = data[0];
        X_max = data[-1];
        # var x{i in 0..n} >=X[1], <=X[m];
        # Initialization:
        def init_x_rule(model, i):
          return float(init_x[i])
        model.x = pyo.Var(model.N, within=pyo.Reals, bounds=(X_min, X_max), initialize=init_x_rule) 
        # var z{i in 0..n} >=0, <=1;
        # Initialization:
        def init_z_rule(model, i):
          return float(init_z[i])
        model.z = pyo.Var(model.N, within=pyo.NonNegativeReals, bounds=(0, 1), initialize=init_z_rule) 
        # Bezier 'time' parameter t for the j-th sample point.
        # var t{j in 1..m} >=0, <= 1;
        # Initialization:  
        def init_t_rule(model, j):
          return float(init_t[j - 1])  # j starts from 1, so we access init_t with j-1
        model.t = pyo.Var(model.M, within=pyo.NonNegativeReals, bounds=(0,1), initialize=init_t_rule )         
        # Estimated cdf for the j-th sample point.
        # var F_hat{j in 1..m} >=0, <= 1;
        model.F_hat = pyo.Var(model.M, within=pyo.NonNegativeReals, bounds=(0,1) ) 

        # Objective function
        # minimize mean_square_error:
        #    1/m * sum {j in 1..m} ( ( j/m - F_hat[j] )^2);
        def mse_rule(model):
          return (1 / m) * sum((emp_cdf_data[j - 1] - model.F_hat[j])**2 for j in model.M)
        model.mse = pyo.Objective(rule=mse_rule, sense=pyo.minimize )

        # Constraints
        # subject to F_hat_estimates {j in 1..m}:
        #    sum{i in 0..n}( comb[i]*t[j]^i*(1-t[j])^(n-i)*z[i] ) = F_hat[j];
        def F_hat_rule(model, j):
          return sum(bezierv.comb[i] * model.t[j]**i * (1 - model.t[j])**(n - i) * model.z[i] for i in model.N ) == model.F_hat[j]
        model.ctr_F_hat = pyo.Constraint(model.M , rule=F_hat_rule)

        # subject to data_sample {j in 1..m}:
        #    sum{i in 0..n}( comb[i]*t[j]^i*(1-t[j])^(n-i)*x[i] ) = X[j];
        def data_sample_rule(model, j):
          return sum(bezierv.comb[i] * model.t[j]**i * (1 - model.t[j])**(n - i) * model.x[i] for i in model.N ) == data[j-1]
        model.ctr_sample = pyo.Constraint(model.M , rule=data_sample_rule)
        
        # subject to convexity_x {i in 0..n-1}:
        #    x[i] <= x[i+1];
        def convexity_x_rule(model, i):
          return model.x[i] <= model.x[i + 1]
        model.ctr_convexity_x = pyo.Constraint(model.N_n , rule=convexity_x_rule)

        # subject to convexity_z {i in 0..n-1}:
        #    z[i] <= z[i+1];
        def convexity_z_rule(model, i):
          return model.z[i] <= model.z[i + 1]
        model.ctr_convexity_z = pyo.Constraint(model.N_n , rule=convexity_z_rule)

        # subject to first_control_x:
        #    x[0] = X[1];
        model.first_control_x = pyo.Constraint(expr=model.x[0] <= data[0])
        # subject to first_control_z:
        #    z[0] = 0;
        model.first_control_z = pyo.Constraint(expr=model.z[0] == 0)

        # subject to last_control_x:
        #    x[n] = X[m];
        model.last_control_x = pyo.Constraint(expr=model.x[n] >= data[-1]) 
        # subject to last_control_z:
        #    z[n] = 1;
        model.last_control_z = pyo.Constraint(expr=model.z[n] == 1)
        
        # subject to first_data_t:
        #    t[1] = 0;
        model.first_t = pyo.Constraint(expr=model.t[1] == 0)
        # subject to last_data_t:
        #    t[m] = 1;
        model.last_t = pyo.Constraint(expr=model.t[m] == 1)

        delta_z = 0.0001
        delta_x = 0.5
        # Left end: x1 ~= x0, z1 ~= z0
        #model.end_close_z_left  = pyo.Constraint(expr = model.z[1] - model.z[0] <= delta_z)
        #model.end_close_x_left  = pyo.Constraint(expr = model.x[1] - model.x[0] <= delta_x)

        # Right end: xn ~= xn-1, zn ~= zn-1
        #model.end_close_x_right = pyo.Constraint(expr = model.x[n] - model.x[n-1] <= delta_x)
        #model.end_close_z_right = pyo.Constraint(expr = model.z[n] - model.z[n-1] <= delta_z)
 
        # Set solver
        pyo_solver = SolverFactory(solver)
        
        try:
            results = pyo_solver.solve(model, tee=False, timelimit=60)
            if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
                controls_x = np.array([model.x[i]() for i in model.N])
                controls_z = np.array([model.z[i]() for i in model.N])
                mse = model.mse()
                bezierv.update_bezierv(controls_x, controls_z)
        except Exception as e:
            print("NonLinearSolver [fit]: An exception occurred during model evaluation:", e)

        return bezierv, mse