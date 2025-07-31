from bezierv.classes.bezierv import Bezierv
import numpy as np
from bezierv.classes.distfit import DistFit
import matplotlib.pyplot as plt
from bezierv.classes.convolver import Convolver
from statsmodels.distributions.empirical_distribution import ECDF


def iqr_filter(x: np.ndarray, k: float = 1.5) -> np.ndarray:
    """
    Remove outliers using the IQR rule:
    keep values in [Q1 - k*IQR, Q3 + k*IQR].
    """
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return x[(x >= lo) & (x <= hi)]


bz_list = []
normal_sum = np.zeros(1000)
for i in range(0, 5):
    print(i)
    normal = np.random.uniform(0, 1, size=1000)
    normal_sum += normal
    #normal = iqr_filter(normal, k=1.5)
    fitter = DistFit(normal, n=6)
    bez = fitter.fit(method='nonlinear')[0]
    #bez.plot_cdf(np.sort(normal))
    bz_list.append(bez)

plt.hist(normal_sum, bins=30, density=True, alpha=0.5, label='Sum Histogram')
plt.show()
convolver = Convolver(bz_list)
convolved_bz = convolver.convolve(n_sims=1000, rng=42, n=6, method='projgrad')

ecdf = ECDF(normal_sum)

fig, ax = plt.subplots()
ax.plot(ecdf.x, ecdf.y, label='Empirical CDF of Normal Sum')
convolved_bz.plot_cdf(ax=ax)
