from bezierv.classes.bezierv import Bezierv
import numpy as np

bez1 = Bezierv(n=1, controls_x=np.array([0.0, 1.0]), controls_z=np.array([0.0, 1.0]))
bez2 = Bezierv(n=1, controls_x=np.array([0.0, 1.0]), controls_z=np.array([0.0, 1.0]))
bz_list = [bez1, bez2]
from bezierv.classes.convolver import Convolver
convolver = Convolver(bz_list)
convolved_bz = convolver.convolve(n_sims=1000, rng=42, n=6, method='projgrad')
convolved_bz.plot_cdf()