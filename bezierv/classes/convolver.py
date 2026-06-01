import numpy as np
from scipy.integrate import quad
from bezierv.classes.distfit import DistFit
from bezierv.classes.bezierv import Bezierv
from typing import Any, List

class Convolver:
    """
    Convolution of independent Bézier random variables.

    Supports Monte Carlo convolution for any number of variables and 
    exact convolution via numerical integration for the two-variable case.

    Attributes
    ----------
    list_bezierv : List[Bezierv]
        Bézier random variables to be convolved. Each element is validated
        (lengths, ordering, initialization) at construction time.
    """
    def __init__(self, list_bezierv: List[Bezierv]):
        """
        Initialize a Convolver from a list of Bézier random variables.

        Parameters
        ----------
        list_bezierv : List[Bezierv]
            Bézier random variables to be convolved.

        Raises
        ------
        ValueError
            If any element fails the length, ordering, or initialization
            checks performed by :class:`~bezierv.classes.bezierv.Bezierv`.
        
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
                 **kwargs:Any) -> tuple[Bezierv, float]:
        """
        Convolve the Bezier RVs via Monte Carlo and fit a Bezierv to the sum.
                                                                                    
        Draws ``n_sims`` samples from each Bézier RV in :attr:`list_bezierv`
        using a shared PRNG stream, sums them, and fits a new Bézier random        
        variable to the resulting sample via :class:`DistFit`.

        Parameters
        ----------
        n_sims : int, optional
            Number of Monte Carlo samples. Default is ``1000``.
        rng : numpy.random.Generator | int | None, optional
            Shared PRNG stream for *all* sampling.
        **kwargs :
            Init options for DistFit(...):
                n, init_x, init_z, init_t, emp_cdf_data, method_init_x
            Fit options for DistFit.fit(...):
                method, algorithm, options

        Returns
        -------
        bezierv : Bezierv
            Bézier random variable fitted to the convolved sample.
        metric : float
            Final objective value reported by :meth:`DistFit.fit`
            (MSE for ``method='mse'``, NLL for ``method='mle'``).

        Raises
        ------
        TypeError
            If ``kwargs`` contains a key that is not recognized as either a
            ``DistFit`` init option or a ``DistFit.fit`` option.
        """

        rng = np.random.default_rng(rng)

        bezierv_sum = np.zeros(n_sims)
        for bz in self.list_bezierv:
            samples = bz.random(n_sims, rng=rng)
            bezierv_sum += samples

        init_keys = {
            "n", "init_x", "init_z", "init_t", "init_w", "emp_cdf_data", "method_init_x"
        }
        fit_keys = {"method", "algorithm", "options"}

        init_kwargs = {k: v for k, v in kwargs.items() if k in init_keys}
        fit_kwargs  = {k: v for k, v in kwargs.items() if k in fit_keys}

        unknown = set(kwargs).difference(init_keys | fit_keys)
        if unknown:
            raise TypeError(f"Unknown keyword(s) for convolve: {sorted(unknown)}")

        fitter = DistFit(bezierv_sum, **init_kwargs)
        bezierv_result, val = fitter.fit(**fit_kwargs)
        return bezierv_result, val
    
    def convolve_exact(self, 
                       n_points: int = 1000,
                       **kwargs) -> tuple[Bezierv, float]:
        """
        Perform convolution of two Bezier random variables using numerical integration.
        
        This method generates a grid of points and computes the exact CDF at each point
        using the exact_cdf_two_bezierv method, then fits a Bezierv to the resulting CDF values.
        
        Parameters
        ----------
        n_points : int, optional
            Number of points to evaluate the exact CDF (default is 1000).
        **kwargs :
            Init options for DistFit(...):
                n, init_x, init_z, init_t, emp_cdf_data, method_init_x
            Fit options for DistFit.fit(...):
                method, algorithm, options

        Returns
        -------
        bezierv: Bezierv
            The fitted Bezierv representing the convolution.
        metric : float
          Final objective value reported by :meth:`DistFit.fit`
          (MSE for ``method='mse'``, NLL for ``method='mle'``).
            
        Raises
        ------
        ValueError
            If the number of Bezier RVs is not exactly 2.
        """
        if len(self.list_bezierv) != 2:
            raise ValueError("Exact convolution is only implemented for two Bezier RVs.")
        
        bz_x, bz_y = self.list_bezierv
        
        z_min = bz_x.support[0] + bz_y.support[0]
        z_max = bz_x.support[1] + bz_y.support[1]
        
        z_points = np.linspace(z_min, z_max, n_points)
        
        cdf_values = np.zeros(n_points)
        for i, z in enumerate(z_points):
            cdf_values[i] = self.exact_cdf_two_bezierv(z)
        
        init_keys = {
            "n", "init_x", "init_z", "init_t", "init_w", "emp_cdf_data", "method_init_x"
        }
        fit_keys = {"method", "algorithm", "options"}

        init_kwargs = {k: v for k, v in kwargs.items() if k in init_keys}
        fit_kwargs = {k: v for k, v in kwargs.items() if k in fit_keys}

        unknown = set(kwargs).difference(init_keys | fit_keys)
        if unknown:
            raise TypeError(f"Unknown keyword(s) for convolve_exact: {sorted(unknown)}")
        
        if 'emp_cdf_data' not in init_kwargs:
            init_kwargs['emp_cdf_data'] = cdf_values
        
        fitter = DistFit(z_points, **init_kwargs)
        bezierv_result, val = fitter.fit(**fit_kwargs)
        
        return bezierv_result, val
    
    def exact_cdf_two_bezierv(self, z: float) -> float:
        """
        Compute the exact CDF of the convolution Z = X + Y at point z using numerical integration.
        
        Parameters
        ----------
        z : float
            The point at which to evaluate the CDF.
            
        Returns
        -------
        float
            The CDF value F_Z(z) at point z.
        """
        if len(self.list_bezierv) != 2:
            raise ValueError("Exact CDF computation is only implemented for two Bezier RVs.")
        
        bz_x, bz_y = self.list_bezierv
        n_x = bz_x.n
        
        def integrand(t_x):
            """
            The integrand function: [∑ᵢ₌₀ⁿˣ⁻¹ B_{n_X-1,i}(t_X) · Δz_i^X] [F_Y(z - x(t_X))]
            """
            pdf_numerator = bz_x.pdf_numerator_t(t_x)
            y_val = z - bz_x.poly_x(t_x)
            cdf_y = bz_y.cdf_x(y_val)
            return pdf_numerator * cdf_y

        try:
            result, _ = quad(integrand, 0, 1)
            return n_x * result
        except Exception as e:
            raise RuntimeError(f"Numerical integration failed for z={z} with error: {e}") from e