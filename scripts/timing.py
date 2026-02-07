import numpy as np
import time
from bezierv.classes.distfit import DistFit

# Set random seed for reproducibility
np.random.seed(42)

# Generate data from a normal distribution
# Mean = 0, Standard deviation = 1, Sample size = 1000
data = np.random.gamma(shape=2.0, scale=2.0, size=1000)

# Create DistFit instance with the data
# n=5 means we use 6 control points (n+1)
distfit = DistFit(data=data, n=20)
distfit.init_x = distfit.get_controls_x(method='uniform')

# Measure the time taken to fit using projgrad
start_time = time.time()
bezierv, mse = distfit.fit(method='projgrad', 
                           step_size_PG=0.001, 
                           max_iter_PG=10000, 
                           threshold_PG=1e-3,)
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

#plot
bezierv.plot_cdf(data=data, show=True)

# Print results
print(f"Fitting completed!")
print(f"Time taken: {elapsed_time:.4f} seconds")
print(f"MSE: {mse:.6f}")
