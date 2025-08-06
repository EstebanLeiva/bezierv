import numpy as np
from scipy.integrate import quad
from bezierv.classes.distfit import DistFit
from bezierv.classes.bezierv import Bezierv

class Convolver:
    def __init__(self, list_bezierv: list[Bezierv]):
        """
        Initialize a ConvBezier instance for convolving Bezier curves.

        This constructor sets up the convolution object by storing the provided Bezierv
        random variables, and creates a new Bezierv instance to hold the convolution 
        result. It also initializes the number of data points to be used in the numerical
        convolution process.

        Parameters
        ----------
        list_bezierv : list[Bezierv]
            A list of Bezierv instances representing the Bezier random variables to be convolved.
        """
        for bez in list_bezierv:
            bez._validate_lengths(bez.controls_x, bez.controls_z)
            bez._validate_ordering(bez.controls_x, bez.controls_z)
            bez._ensure_initialized()
        
        self.list_bezierv = list_bezierv

    
    def convolve(self,
                 n_sims: int = 1000,
                 *,
                 rng: np.random.Generator | int | None = None,
                 **kwargs) -> Bezierv:
        """
        Convolve the Bezier RVs via Monte Carlo and fit a Bezierv to the sum.

        Parameters
        ----------
        n_sims : int
            Number of Monte Carlo samples.
        rng : numpy.random.Generator | int | None, optional
            Shared PRNG stream for *all* sampling.
        **kwargs :
            Init options for DistFit(...):
                n, init_x, init_z, init_t, emp_cdf_data, method_init_x
            Fit options for DistFit.fit(...):
                method, step_size_PG, max_iter_PG, threshold_PG,
                step_size_PS, max_iter_PS, solver_NL, max_iter_NM
        """
        rng = np.random.default_rng(rng)

        bezierv_sum = np.zeros(n_sims)
        for bz in self.list_bezierv:
            samples = bz.random(n_sims, rng=rng)
            bezierv_sum += samples

        init_keys = {
            "n", "init_x", "init_z", "init_t", "emp_cdf_data", "method_init_x"
        }
        fit_keys = {
            "method", "step_size_PG", "max_iter_PG", "threshold_PG",
            "step_size_PS", "max_iter_PS", "solver_NL", "max_iter_NM"
        }

        init_kwargs = {k: v for k, v in kwargs.items() if k in init_keys}
        fit_kwargs  = {k: v for k, v in kwargs.items() if k in fit_keys}

        unknown = set(kwargs).difference(init_keys | fit_keys)
        if unknown:
            raise TypeError(f"Unknown keyword(s) for convolve: {sorted(unknown)}")

        fitter = DistFit(bezierv_sum, **init_kwargs)
        bezierv_result, _ = fitter.fit(**fit_kwargs)
        return bezierv_result