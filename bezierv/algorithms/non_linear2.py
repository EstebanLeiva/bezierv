import pyomo.environ as pyo
import numpy as np
from bezierv.classes.bezierv import Bezierv
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition


def fit(distfit:DistFit, solver: str='ipopt') -> Bezierv:
        """
        Fits a Bézier distribution to the given data using n control points.
        This method sorts the input data, computes the empirical cumulative 
        distribution function (CDF), and sets up an optimization model to 
        fit a Bézier distribution. The control points and the empirical CDF 
        are automatically saved. The method returns the mean square error (MSE) 
        of the fit.
        
        Parameters:
        distfit (DistFit): An instance of the DistFit class containing the data and initial parameters.
        solver (str): The name of the solver to use for optimization. Default is 'ipopt'.

        Returns:
        float: The mean squared error (MSE) of the fit.

        Raises:
        Exception: If the solver fails to find an optimal solution.

        Notes:
        - The method uses the IPOPT solver for optimization.
        - The control points are constrained to lie within the range of the data.
        - The method ensures that the control points and the Bézier 'time' parameters are sorted.
        - Convexity constraints are applied to the control points and the Bézier 'time' parameters.
        - The first and last control points are fixed to the minimum and maximum of the data, respectively.
        - The first and last Bézier 'time' parameters are fixed to 0 and 1, respectively.
        """        
        # Defining the optimization model
        model = pyo.ConcreteModel()

        # Sets
        model.N = pyo.Set(initialize=list(range(distfit.n + 1))) # N = 0,...,i,...,n
        model.N_n = pyo.Set(initialize=list(range(distfit.n))) # N = 0,...,i,...,n-1
        model.M = pyo.Set(initialize=list(range(1, distfit.m + 1))) # M = 1,...,j,...,m
        model.M_m = pyo.Set(initialize=list(range(1, distfit.m))) # M = 1,...,j,...,m-1

        # Decision variables
        # Control points. Box constraints.
        X_min = distfit.data[0];
        X_max = distfit.data[-1];
        # var x{i in 0..n} >=X[1], <=X[m];
        # Initialization: let {i in 1..n-1} x[i] := quantile(i/n);
        def init_x_rule(model, i):
          return np.quantile(distfit.data, i / distfit.n)
        model.x = pyo.Var(model.N, within=pyo.Reals, bounds=(X_min, X_max), initialize=init_x_rule) 
        # var z{i in 0..n} >=0, <=1;
        # Initialization: let {i in 1..n-1} z[i] := i*(1/n);
        def init_z_rule(model, i):
          return i*(1 / distfit.n)
        model.z = pyo.Var(model.N, within=pyo.NonNegativeReals, bounds=(0, 1), initialize=init_z_rule) 
        # Bezier 'time' parameter t for the j-th sample point.
        # var t{j in 1..m} >=0, <= 1;
        # Initialization: let {j in 2..m-1} t[j] := j*(1/m);        
        def init_t_rule(model, j):
          return j*(1 / distfit.m)
        model.t = pyo.Var(model.M, within=pyo.NonNegativeReals, bounds=(0,1), initialize=init_t_rule )         
        # Estimated cdf for the j-th sample point.
        # var F_hat{j in 1..m} >=0, <= 1;
        model.F_hat = pyo.Var(model.M, within=pyo.NonNegativeReals, bounds=(0,1) ) 

        # Objective function
        # minimize mean_square_error:
        #    1/m * sum {j in 1..m} ( ( j/m - F_hat[j] )^2);
        def mse_rule(model):
          return (1 / distfit.m) * sum(((j / distfit.m) - model.F_hat[j])**2 for j in model.M)
        model.mse = pyo.Objective(rule=mse_rule, sense=pyo.minimize )

        # Constraints
        # subject to F_hat_estimates {j in 1..m}:
        #    sum{i in 0..n}( comb[i]*t[j]^i*(1-t[j])^(n-i)*z[i] ) = F_hat[j];
        def F_hat_rule(model, j):
          return sum(distfit.bezierv.comb[i] * model.t[j]**i * (1 - model.t[j])**(distfit.n - i) * model.z[i] for i in model.N ) == model.F_hat[j]
        model.ctr_F_hat = pyo.Constraint(model.M , rule=F_hat_rule)

        # subject to data_sample {j in 1..m}:
        #    sum{i in 0..n}( comb[i]*t[j]^i*(1-t[j])^(n-i)*x[i] ) = X[j];
        def data_sample_rule(model, j):
          return sum(distfit.bezierv.comb[i] * model.t[j]**i * (1 - model.t[j])**(distfit.n - i) * model.x[i] for i in model.N ) == distfit.data[j-1]
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
        model.first_control_x = pyo.Constraint(expr=model.x[0] <= distfit.data[0])
        # subject to first_control_z:
        #    z[0] = 0;
        model.first_control_z = pyo.Constraint(expr=model.z[0] == 0)

        # subject to last_control_x:
        #    x[n] = X[m];
        model.last_control_x = pyo.Constraint(expr=model.x[distfit.n] >= distfit.data[-1]) 
        # subject to last_control_z:
        #    z[n] = 1;
        model.last_control_z = pyo.Constraint(expr=model.z[distfit.n] == 1)
        
        # subject to first_data_t:
        #    t[1] = 0;
        model.first_t = pyo.Constraint(expr=model.t[1] == 0)
        # subject to last_data_t:
        #    t[m] = 1;
        model.last_t = pyo.Constraint(expr=model.t[distfit.m] == 1)
 
        # Set solver
        pyo_solver = SolverFactory(solver)
        
        try:
            results = pyo_solver.solve(model, tee=False, timelimit=60)
            if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
                controls_x = np.array([model.x[i]() for i in model.N])
                controls_z = np.array([model.z[i]() for i in model.N])
                distfit.mse = model.mse()
                distfit.bezierv.update_bezierv(controls_x, controls_z, (distfit.data[0], distfit.data[-1]))
        except Exception as e:
            print("NonLinearSolver [fit]: An exception occurred during model evaluation:", e)

        return distfit.bezierv