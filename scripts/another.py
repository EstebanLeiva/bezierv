# test the ipm method
import numpy as np
from bezierv.classes.distfit import DistFit


# do the same for 10 other distributions
distributions = ['normal', 'exponential', 'uniform', 'beta', 'gamma', 'weibull', 'lognormal', 'pareto', 'cauchy', 'triangular']
for dist in distributions:
    if dist == 'normal':
        data = np.random.normal(loc=0.0, scale=10.0, size=1000)
    elif dist == 'exponential':
        data = np.random.exponential(scale=1.0, size=1000)
    elif dist == 'uniform':
        data = np.random.uniform(low=-5.0, high=5.0, size=1000)
    elif dist == 'beta':
        data = np.random.beta(a=2.0, b=5.0, size=1000)
    elif dist == 'gamma':
        data = np.random.gamma(shape=2.0, scale=2.0, size=1000)

    distfit = DistFit(data=data, n=10)
    bezierv, mse = distfit.fit(method='mse')
    bezierv.plot_cdf(data=data, show=True)
    print(f"{dist.capitalize()} MSE: {mse:.6f}")