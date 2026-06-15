import numpy as np
from bezierv import DistFit

# Synthetic data
rng = np.random.default_rng(42)
data = rng.beta(2, 4, 1000)  # replace with your data

fitter = DistFit(data, n=5)  # n = degree

# Fit: method is 'mse'; algorithm is 'projected_gradient'
bz, mse = fitter.fit(method='mse', algorithm='projected_gradient')

samples = bz.random(10_000, rng=42)                  # draw samples
q90 = bz.quantile(0.90)                              # 90% quantile
mean, variance = bz.mean(), bz.variance()    # mean and variance
bz.plot_cdf(data)                                    # overlays ECDF and Bézier CDF