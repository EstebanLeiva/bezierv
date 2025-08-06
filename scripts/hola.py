import numpy as np

normal_data = np.random.normal(loc=0, scale=1, size=1000)

from bezierv.classes.distfit import DistFit

df = DistFit(normal_data, n=3)
bez, mse = df.fit(method='projgrad', max_iter_PG=100)