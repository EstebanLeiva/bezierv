from bezierv.classes.bezierv import Bezierv
import numpy as np
from bezierv.classes.distfit import DistFit
import matplotlib.pyplot as plt
from bezierv.classes.convolver import Convolver

bz_list = []
for i in range(0, 5):
    print(i)
    normal = np.random.normal(loc=1, scale=1, size=1000)
    fitter = DistFit(normal, n=5)
    bz_list.append(fitter.fit()[0])

convolver = Convolver(bz_list)
convolved_bz = convolver.convolve(n_sims=2000, rng=42)
print(convolved_bz.get_mean())
print(convolved_bz.get_variance())
convolved_bz.plot_cdf()
convolved_bz.plot_pdf()
