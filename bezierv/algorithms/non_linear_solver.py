import pyomo.environ as pyo
import numpy as np

from statsmodels.distributions.empirical_distribution import ECDF
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

from bezierv.classes.bezierv import Bezierv

class NonLinearSolver:
    def __init__(self, bezierv: Bezierv, data: np.array, emp_cdf_data: np.array=None):
        """
        Initialize the NonLinearSolver instance with a Bezierv object and data to fit.

        Parameters
        ----------
        bezierv : Bezierv
            An instance of the Bezierv class representing the Bezier random variable.
        data : np.array
            The data points to be fitted; they will be sorted.
        emp_cdf_data : np.array, optional
            The empirical CDF values corresponding to the data. If not provided, they
            are computed using ECDF from the sorted data.

        Attributes
        ----------
        bezierv : Bezierv
            The Bezierv object.
        n : int
            The degree of the Bezier random variable (inferred from bezierv.n).
        data : np.array
            The sorted data points.
        m : int
            The number of data points.
        t_data : np.array
            The values of t corresponding to the data points on the Bezier random variable.
        fit_error : float
            The fitting error, initialized to infinity.
        emp_cdf_data : np.array
            The empirical CDF values for the data points.
        """
        self.bezierv = bezierv
        self.n = bezierv.n
        self.data = np.sort(data)
        self.m = len(data)
        self.mse = np.inf

        if emp_cdf_data is None:
            emp_cdf = ECDF(data)
            self.emp_cdf_data = emp_cdf(data)
        else:
            self.emp_cdf_data = emp_cdf_data

    def fit(self, solver='ipopt'):
        """
        Fits a Bézier distribution to the given data using n control points.
        This method sorts the input data, computes the empirical cumulative 
        distribution function (CDF), and sets up an optimization model to 
        fit a Bézier distribution. The control points and the empirical CDF 
        are automatically saved. The method returns the mean square error (MSE) 
        of the fit.
        
        Parameters:
        data (list): A list of data points to fit the Bézier distribution.
        n (int): The number of control points for the Bézier distribution.
        solver (str): The SolverFactory str in pyomo. Values: "ipopt","knitroampl"

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
        model.N = pyo.Set(initialize=list(range(self.n + 1))) # N = 0,...,i,...,n
        model.N_n = pyo.Set(initialize=list(range(self.n))) # N = 0,...,i,...,n-1
        model.M = pyo.Set(initialize=list(range(1, self.m + 1))) # M = 1,...,j,...,m
        model.M_m = pyo.Set(initialize=list(range(1, self.m))) # M = 1,...,j,...,m-1

        # Decision variables
        # Control points. Box constraints.
        X_min = self.data[0];
        X_max = self.data[self.m - 1];
        # var x{i in 0..n} >=X[1], <=X[m];
        # Initialization: let {i in 1..n-1} x[i] := X[1]+i*(X[m]-X[1])/n;
        def init_x_rule(model, i):
          return X_min + i*(X_max - X_min) / self.n
        model.x = pyo.Var(model.N, within=pyo.Reals, bounds=(X_min, X_max), initialize=init_x_rule) 
        # var z{i in 0..n} >=0, <=1;
        # Initialization: let {i in 1..n-1} z[i] := i*(1/n);
        def init_z_rule(model, i):
          return i*(1 / self.n)
        model.z = pyo.Var(model.N, within=pyo.NonNegativeReals, bounds=(0, 1), initialize=init_z_rule) 
        # Bezier 'time' parameter t for the j-th sample point.
        # var t{j in 1..m} >=0, <= 1;
        # Initialization: let {j in 2..m-1} t[j] := j*(1/m);        
        def init_t_rule(model, j):
          return j*(1 / self.m)
        model.t = pyo.Var(model.M, within=pyo.NonNegativeReals, bounds=(0,1), initialize=init_t_rule )         
        # Estimated cdf for the j-th sample point.
        # var F_hat{j in 1..m} >=0, <= 1;
        model.F_hat = pyo.Var(model.M, within=pyo.NonNegativeReals, bounds=(0,1) ) 

        # Objective function
        # minimize mean_square_error:
        #    1/m * sum {j in 1..m} ( ( j/m - F_hat[j] )^2);
        def mse_rule(model):
          return (1 / self.m) * sum(((j / self.m) - model.F_hat[j])**2 for j in model.M)
        model.mse = pyo.Objective(rule=mse_rule, sense=pyo.minimize )

        # Constraints
        # subject to F_hat_estimates {j in 1..m}:
        #    sum{i in 0..n}( comb[i]*t[j]^i*(1-t[j])^(n-i)*z[i] ) = F_hat[j];
        def F_hat_rule(model, j):
          return sum(self.bezierv.comb[i] * model.t[j]**i * (1 - model.t[j])**(self.n - i) * model.z[i] for i in model.N ) == model.F_hat[j]
        model.ctr_F_hat = pyo.Constraint(model.M , rule=F_hat_rule)

        # subject to data_sample {j in 1..m}:
        #    sum{i in 0..n}( comb[i]*t[j]^i*(1-t[j])^(n-i)*x[i] ) = X[j];
        def data_sample_rule(model, j):
          return sum(self.bezierv.comb[i] * model.t[j]**i * (1 - model.t[j])**(self.n - i) * model.x[i] for i in model.N ) == self.data[j-1]
        model.ctr_sample = pyo.Constraint(model.M , rule=data_sample_rule)

        # subject to sorted_t{j in 1..m-1}:
        #    t[j] <= t[j+1];
        def sorted_t_rule(model, j):
          return model.t[j] <= model.t[j + 1]
        model.ctr_sorted_t = pyo.Constraint(model.M_m , rule=sorted_t_rule)
        
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
        model.first_control_x = pyo.Constraint(expr=model.x[0] == self.data[0])
        # subject to first_control_z:
        #    z[0] = 0;
        model.first_control_z = pyo.Constraint(expr=model.z[0] == 0)

        # subject to last_control_x:
        #    x[n] = X[m];
        model.last_control_x = pyo.Constraint(expr=model.x[self.n] == self.data[self.m - 1])
        # subject to last_control_z:
        #    z[n] = 1;
        model.last_control_z = pyo.Constraint(expr=model.z[self.n] == 1)
        
        # subject to first_data_t:
        #    t[1] = 0;
        model.first_t = pyo.Constraint(expr=model.t[1] == 0)
        # subject to last_data_t:
        #    t[m] = 1;
        model.last_t = pyo.Constraint(expr=model.t[self.m] == 1)
 
        # Set solver
        pyo_solver = SolverFactory(solver)
  
        # Write the model to an NL file    
        results = pyo_solver.solve(model, tee=True)
        
        try:
            if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
                # Solution is optimal (local or global) and feasible
                controls_x = np.array([model.x[i]() for i in model.N])
                controls_z = np.array([model.z[i]() for i in model.N])
                
                # Stores fitted cdf in a list of tuples (X_j, F_hat(X_j))
                fcdf = [(self.data[j - 1], model.F_hat[j]()) for j in model.M]
                # Stores "time" parameter t in a list of tuples (X_j, t_j)
                t = [(self.data[j - 1], model.t[j]()) for j in model.M]
                # Retrieves mean square error from model's objective
                self.mse = model.mse()

                self.bezierv.update_bezierv(controls_x, controls_z, (self.data[0], self.data[-1]))
        except Exception as e:
            print("NonLinearSolver [fit]: An exception occurred during model evaluation:", e)

        return self.bezierv
