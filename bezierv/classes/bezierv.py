import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.optimize import brentq, bisect
from scipy.integrate import quad
from statsmodels.distributions.empirical_distribution import ECDF

# import arrays for type hinting
from numpy.typing import ArrayLike

class Bezierv:
    def __init__(self, 
                 n: int, 
                 controls_x: ArrayLike=None, 
                 controls_z: ArrayLike=None):
        """
        Initialize a Bézier random variable.

        Sets up control points, binomial coefficient arrays, and consecutive
        difference arrays. Moment attributes are set to ``np.inf`` as a
        sentinel indicating they have not yet been computed.

        Parameters
        ----------
        n : int
            Degree of the Bézier curve. The curve has ``n + 1`` control
            points (paper notation: degree-``n`` Bézier distribution).
        controls_x : array-like of shape (n + 1,), optional
            x-coordinates of the control points. Must be strictly increasing.
            If ``None``, a zero array is created (placeholder state).
        controls_z : array-like of shape (n + 1,), optional
            z-coordinates (CDF values) of the control points. Must be
            non-decreasing with ``z[0] = 0`` and ``z[n] = 1``.
            If ``None``, a zero array is created (placeholder state).

        Attributes
        ----------
        n : int
            Degree of the Bézier curve.
        deltas_x : numpy.ndarray, shape (n,)
            Differences between consecutive x control points.
        deltas_z : numpy.ndarray, shape (n,)
            Differences between consecutive z control points.
        comb : numpy.ndarray, shape (n + 1,)
            Binomial coefficients ``C(n, i)`` for ``i = 0, …, n``.
        comb_minus : numpy.ndarray, shape (n,)
            Binomial coefficients ``C(n-1, i)`` for ``i = 0, …, n-1``
            (used for PDF evaluation).
        support : tuple of float
            ``(controls_x[0], controls_x[-1])``, or ``(-inf, inf)`` in
            placeholder state.
        controls_x : numpy.ndarray, shape (n + 1,)
            x-coordinates of the control points.
        controls_z : numpy.ndarray, shape (n + 1,)
            z-coordinates of the control points.
        raw_moments : dict[int, float]
            Cache of computed raw moments ``E[X^r]`` keyed by order ``r``.
            Cleared by ``update_bezierv``.
        """
        self.n = n
        self.deltas_x = np.zeros(n)
        self.deltas_z = np.zeros(n)
        self.comb = np.zeros(n + 1)
        self.comb_minus = np.zeros(n)
        self.support = (-np.inf, np.inf)

        if controls_x is None and controls_z is None:
            self.controls_x = np.zeros(n + 1)
            self.controls_z = np.zeros(n + 1)
        elif controls_x is not None and controls_z is not None:
            controls_x = np.asarray(controls_x, dtype=float)
            controls_z = np.asarray(controls_z, dtype=float)
            self._validate_lengths(controls_x, controls_z)
            self._validate_ordering(controls_x, controls_z)
            self.controls_x = controls_x
            self.controls_z = controls_z
            self.support = (controls_x[0], controls_x[-1])
        else:
            raise ValueError('Either all or none of the parameters controls_x and controls_z must be provided')

        self.raw_moments: dict[int, float] = {}

        self.combinations()
        self.deltas()

    def update_bezierv(self, 
                       controls_x: np.array, 
                       controls_z: np.array):
        """
        Update the control points, support, and delta arrays in-place.

        Parameters
        ----------
        controls_x : array-like of shape (n + 1,)
            New x-coordinates of the control points. Must be strictly increasing.
        controls_z : array-like of shape (n + 1,)
            New z-coordinates of the control points. Must be non-decreasing.
        """
        controls_x = np.asarray(controls_x, dtype=float)
        controls_z = np.asarray(controls_z, dtype=float)
        self._validate_lengths(controls_x, controls_z)
        self._validate_ordering(controls_x, controls_z)

        self.controls_x = controls_x
        self.controls_z = controls_z
        self.support = (controls_x[0], controls_x[-1])
        self.raw_moments = {}

        self.deltas()

    def combinations(self):
        """
        Compute and store binomial coefficients.
        """
        n = self.n
        for i in range(0, n + 1):
            self.comb[i] = math.comb(n, i)
            if i < n:
                self.comb_minus[i] = math.comb(n - 1, i)

    def deltas(self):
        """
        Compute the differences between consecutive control points.
        """
        n = self.n
        for i in range(n):
            self.deltas_x[i] = self.controls_x[i + 1] - self.controls_x[i]
            self.deltas_z[i] = self.controls_z[i + 1] - self.controls_z[i]

    def bernstein(self, t: float, i: int, combinations: np.ndarray, n: int) -> float:
        """
        Evaluate the ``i``-th degree-``n`` Bernstein basis polynomial at ``t``.

        Computes ``B_{n,i}(t) = C(n,i) * t^i * (1-t)^(n-i)`` as defined
        in Eq. (1) of the paper.

        Parameters
        ----------
        t : float
            Parameter value in ``[0, 1]``.
        i : int
            Index of the basis polynomial, ``0 <= i <= n``.
        combinations : numpy.ndarray
            Precomputed binomial coefficients; pass ``self.comb`` for
            degree ``n`` or ``self.comb_minus`` for degree ``n - 1``.
        n : int
            Degree of the Bernstein polynomial.

        Returns
        -------
        float
            Value of ``B_{n,i}(t)``.
        """
        return combinations[i] * t**i * (1 - t)**(n - i)

    def poly_x(self, t: float, controls_x: np.ndarray = None) -> float:
        """
        Evaluate the x-component of the Bézier curve at parameter ``t``.

        Computes ``B_x(t) = ∑ᵢ B_{n,i}(t) * xᵢ`` (Eq. 2 of the paper).

        Parameters
        ----------
        t : float
            Parameter value in ``[0, 1]``.
        controls_x : numpy.ndarray of shape (n + 1,), optional
            x-coordinates of the control points. Defaults to
            ``self.controls_x``.

        Returns
        -------
        float
            x-coordinate of the Bézier curve at ``t``.
        """
        if controls_x is None:
            self._ensure_initialized()
            controls_x = self.controls_x
        n = self.n
        p_x = 0
        for i in range(n + 1):
           p_x  += self.bernstein(t, i, self.comb, self.n) * controls_x[i]
        return p_x

    def poly_z(self, t: float, controls_z: np.ndarray = None) -> float:
        """
        Evaluate the CDF of the Bézier distribution at parameter ``t``.

        Computes ``F(t) = B_z(t) = ∑ᵢ B_{n,i}(t) * zᵢ`` (Eq. 2 of the
        paper). Returns a value in ``[0, 1]`` when control points satisfy
        the CDF boundary conditions.

        Parameters
        ----------
        t : float
            Parameter value in ``[0, 1]``.
        controls_z : numpy.ndarray of shape (n + 1,), optional
            z-coordinates of the control points. Defaults to
            ``self.controls_z``.

        Returns
        -------
        float
            CDF value at parameter ``t``.
        """
        if controls_z is None:
            self._ensure_initialized()
            controls_z = self.controls_z
        n = self.n
        p_z = 0
        for i in range(n + 1):
           p_z  += self.bernstein(t, i, self.comb, self.n) * controls_z[i]
        return p_z

    def root_find(self, x: float, method: str = 'brentq') -> float:
        """
        Find the parameter ``t`` such that ``B_x(t) = x``.

        Solves ``poly_x(t) - x = 0`` on ``[0, 1]`` using the specified
        root-finding method.

        Parameters
        ----------
        x : float
            Target x-coordinate within the support ``[controls_x[0],
            controls_x[-1]]``.
        method : {'brentq', 'bisect'}, optional
            Root-finding algorithm. Default is ``'brentq'``.

        Returns
        -------
        float
            Parameter ``t`` in ``[0, 1]`` satisfying ``B_x(t) ≈ x``.

        Raises
        ------
        ValueError
            If the root-finder cannot bracket a root (``x`` outside support).
        """
        self._ensure_initialized()
        def poly_x_zero(t, x):
            return self.poly_x(t) - x
        if method == 'brentq':
            t = brentq(poly_x_zero, 0, 1, args=(x,))
        elif method == 'bisect':
            t = bisect(poly_x_zero, 0, 1, args=(x,))
        return t

    def eval_t(self, t: float) -> tuple[float, float]:
        """
        Evaluate the Bezier random variable at a given parameter value t.

        Parameters
        ----------
        t : float
            The parameter value at which to evaluate the curve (typically in [0, 1]).

        Returns
        -------
        tuple[float, float]
            A tuple containing the (x, z) coordinates of the point on the curve w.r.t. t.
        """
        self._ensure_initialized()
        n = self.n
        p_x = 0
        p_z = 0
        for i in range(n + 1):
            p_x += self.comb[i] * t**i * (1 - t)**(n - i) * self.controls_x[i]
            p_z += self.comb[i] * t**i * (1 - t)**(n - i) * self.controls_z[i]
        return p_x, p_z

    def eval_x(self, x: float) -> tuple[float, float]:
        """
        Evaluate the Bezier random variable at a given x-coordinate.

        Parameters
        ----------
        x : float
            The x-coordinate at which to evaluate the Bezier curve.

        Returns
        -------
        tuple[float, float]
            A tuple containing the (x, z) coordinates of the point on the curve w.r.t. x.
        """
        self._ensure_initialized()
        t = self.root_find(x)
        return self.eval_t(t)

    def cdf_x(self, x: float) -> float:
        """
        Compute the cumulative distribution function (CDF) at a given x-coordinate.

        Parameters
        ----------
        x : float
            The x-coordinate at which to evaluate the CDF.

        Returns
        -------
        float
            The CDF value at the given x-coordinate.
        """
        self._ensure_initialized()
        if x < self.controls_x[0]:
            return 0
        if x > self.controls_x[-1]:
            return 1
        _, p_z = self.eval_x(x)
        return p_z

    def quantile(self, alpha: float, method: str = 'brentq') -> float:
        """
        Compute the quantile (inverse CDF) at probability level ``alpha``.

        Parameters
        ----------
        alpha : float
            Probability level in ``[0, 1]``.
        method : {'brentq', 'bisect'}, optional
            Root-finding algorithm used to invert the CDF. Default is
            ``'brentq'``.

        Returns
        -------
        float
            ``x`` such that ``F(x) = alpha``.
        """
        self._ensure_initialized()
        def cdf_t(t, alpha):
            return self.poly_z(t) - alpha
        
        if method == 'brentq':
            t = brentq(cdf_t, 0, 1, args=(alpha,))
        elif method == 'bisect':
            t = bisect(cdf_t, 0, 1, args=(alpha,))
        return self.poly_x(t)

    def pdf_t(self, t: float) -> float:
        """
        Evaluate the Bézier PDF at parameter ``t``.

        Computes ``f(t) = B'_z(t) / B'_x(t)`` where the derivatives are
        degree-``(n-1)`` Bézier curves in the differences ``Δzᵢ`` and
        ``Δxᵢ`` (Section 2 of the paper).

        Parameters
        ----------
        t : float
            Parameter value in ``[0, 1]``.

        Returns
        -------
        float
            PDF value at ``t``.
        """
        self._ensure_initialized()
        n = self.n
        pdf_num_z = 0
        pdf_denom_x = 0
        for i in range(n):
            pdf_num_z += self.bernstein(t, i, self.comb_minus, n - 1) * self.deltas_z[i]
            pdf_denom_x += self.bernstein(t, i, self.comb_minus, n - 1) * self.deltas_x[i]
        return pdf_num_z/pdf_denom_x
    
    def pdf_x(self, x: float) -> float:
        """
        Compute the probability density function (PDF) of the Bezier random variable at a given x.

        Parameters
        ----------
        x : float
            The x-coordinate at which to evaluate the PDF.

        Returns
        -------
        float
            The computed PDF value at x.
        """
        self._ensure_initialized()
        t = self.root_find(x)
        return self.pdf_t(t)

    def pdf_numerator_t(self, t: float) -> float:
        """
        Evaluate the numerator of the Bézier PDF at parameter ``t``.

        Computes ``B'_z(t) = ∑ᵢ B_{n-1,i}(t) * Δzᵢ``, the z-derivative
        component of the PDF. Used internally when only the numerator is
        needed (e.g. for MLE gradient computations).

        Parameters
        ----------
        t : float
            Parameter value in ``[0, 1]``.

        Returns
        -------
        float
            Numerator ``B'_z(t)`` of the PDF at ``t``.
        """
        self._ensure_initialized()
        pdf_num_z = 0
        for i in range(self.n):
            pdf_num_z += self.bernstein(t, i, self.comb_minus, self.n - 1) * self.deltas_z[i]
        return pdf_num_z

    def raw_moment(self, r: int) -> float:
        """
        Compute and cache the raw moment ``E[X^r]`` of the distribution.

        ``r = 1`` uses the closed-form expression derived from the Bézier
        curve properties; ``r >= 2`` integrates ``x^r * f(x)`` numerically
        over the support via :func:`scipy.integrate.quad`.

        Parameters
        ----------
        r : int
            Order of the raw moment (``r >= 1``).

        Returns
        -------
        float
            ``E[X^r]``.
        """
        self._ensure_initialized()
        if r in self.raw_moments:
            return self.raw_moments[r]
        if r == 1:
            total = 0.0
            for ell in range(self.n + 1):
                inner_sum = 0.0
                for i in range(self.n):
                    denom = math.comb(2 * self.n - 1, ell + i)
                    inner_sum += (self.comb_minus[i] / denom) * self.deltas_z[i]
                total += self.comb[ell] * self.controls_x[ell] * inner_sum
            value = 0.5 * total
        else:
            a, b = self.support
            value, _ = quad(lambda x: x**r * self.pdf_x(x), a, b)
        self.raw_moments[r] = value
        return value

    def mean(self) -> float:
        """Mean ``E[X]`` (closed form)."""
        return self.raw_moment(1)

    def variance(self) -> float:
        """Variance ``E[X^2] - E[X]^2``."""
        m1 = self.raw_moment(1)
        return self.raw_moment(2) - m1**2

    def skewness(self) -> float:
        """Skewness ``E[(X - μ)^3] / σ^3``."""
        m1 = self.raw_moment(1)
        var = self.raw_moment(2) - m1**2
        return (self.raw_moment(3) - 3 * m1 * var - m1**3) / var**1.5

    def kurtosis(self) -> float:
        """Kurtosis ``E[(X - μ)^4] / σ^4`` (Pearson, not excess)."""
        m1 = self.raw_moment(1)
        var = self.raw_moment(2) - m1**2
        return (
            self.raw_moment(4)
            - 4 * m1 * self.raw_moment(3)
            + 6 * m1**2 * var
            + 3 * m1**4
        ) / var**2
    
    def random(self, 
               n_sims: int,
               *,
               rng: np.random.Generator | int | None = None):
        """
        Generate random samples from the Bezier random variable.

        This method generates `n_sims` random samples from the Bezier random variable by evaluating
        the inverse CDF at uniformly distributed random values in the interval [0, 1].

        Parameters
        ----------
        n_sims : int
            The number of random samples to generate.
        rng : numpy.random.Generator | int | None, optional
            Pseudorandom-number generator state.  If *None* (default), a new
            ``numpy.random.Generator`` is created with fresh OS entropy.  Any
            value accepted by :func:`numpy.random.default_rng` (e.g. a seed
            integer or :class:`~numpy.random.SeedSequence`) is also allowed.


        Returns
        -------
        numpy.ndarray, shape (n_sims,)
            Random samples drawn from the Bézier distribution.
        """
        self._ensure_initialized()
        rng = np.random.default_rng(rng)
        u = rng.uniform(0, 1, n_sims)
        samples = np.zeros(n_sims)
        for i in range(n_sims):
            samples[i] = self.quantile(u[i])
        return samples


    def plot_cdf(self, data: np.ndarray = None, num_points: int = 100, ax: plt.Axes = None, show: bool = True):
        """
        Plot the Bézier CDF, optionally alongside the empirical CDF.

        Parameters
        ----------
        data : array-like, optional
            Observed sample. If provided, the empirical CDF is overlaid.
            If ``None``, a linspace over the support is used.
        num_points : int, optional
            Number of evaluation points when ``data`` is ``None``. Default
            is ``100``.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Defaults to ``plt.gca()``.
        show : bool, optional
            If ``True`` (default), call ``plt.show()`` after plotting.
        """
        self._ensure_initialized()
        data_bool = True
        if data is None:
            data_bool = False
            data = np.linspace(np.min(self.controls_x), np.max(self.controls_x), num_points)

        data = np.sort(data)
        x_bezier = np.zeros(len(data))
        cdf_x_bezier = np.zeros(len(data))

        for i in range(len(data)):
            p_x, p_z = self.eval_x(data[i])
            x_bezier[i] = p_x
            cdf_x_bezier[i] = p_z

        if ax is None:
            ax = plt.gca()

        if data_bool:
            ecdf_fn = ECDF(data)
            ax.plot(data, ecdf_fn(data), label='Empirical cdf', linestyle='--', color='black')

        ax.plot(x_bezier, cdf_x_bezier, label='Bezier cdf', linestyle='--')
        ax.scatter(self.controls_x, self.controls_z, label='Control Points', color='red')
        ax.legend()
        if show:
            plt.show()

    def plot_pdf(self, data: np.ndarray = None, num_points: int = 100, ax: plt.Axes = None, show: bool = True):
        """
        Plot the Bézier PDF.

        Parameters
        ----------
        data : array-like, optional
            Points at which to evaluate the PDF. If ``None``, a linspace
            over the support is used.
        num_points : int, optional
            Number of evaluation points when ``data`` is ``None``. Default
            is ``100``.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Defaults to ``plt.gca()``.
        show : bool, optional
            If ``True`` (default), call ``plt.show()`` after plotting.
        """
        self._ensure_initialized()
        if data is None:
            data = np.linspace(np.min(self.controls_x), np.max(self.controls_x), num_points)

        x_bezier = np.zeros(len(data))
        pdf_x_bezier = np.zeros(len(data))

        if ax is None:
            ax = plt.gca()

        for i in range(len(data)):
            p_x, _ = self.eval_x(data[i])
            x_bezier[i] = p_x
            pdf_x_bezier[i] = self.pdf_x(data[i])

        ax.plot(x_bezier, pdf_x_bezier, label='Bezier pdf', linestyle='-')
        ax.legend()
        if show:
            plt.show()

    def _validate_lengths(self, controls_x, controls_z):
        if len(controls_x) != len(controls_z):
            raise ValueError("controls_x and controls_z must have the same length.")
        if len(controls_x) != self.n + 1:
            raise ValueError(f"controls arrays must have length n+1 (= {self.n + 1}).")

    def _validate_ordering(self, controls_x, controls_z):
        if np.any(np.diff(controls_x) <= 0):
            raise ValueError("controls_x must be strictly increasing.")
        if np.any(np.diff(controls_z) < 0):
            raise ValueError("controls_z must be nondecreasing.")

    def _ensure_initialized(self):
        if np.allclose(self.controls_x, 0) and np.allclose(self.controls_z, 0):
            raise RuntimeError(
                "Bezier controls are all zeros (placeholder). "
                "Provide valid controls in the constructor or call update_bezierv()."
            )
        
from bokeh.plotting import figure, curdoc
from bokeh.models import (ColumnDataSource, PointDrawTool, Button, CustomJS, 
                          DataTable, TableColumn, NumberFormatter, TapTool)
from bokeh.layouts import column, row

class InteractiveBezierv:
    """Manages an interactive Bezier distribution in a Bokeh plot."""
    
    def __init__(self, controls_x, controls_z):
        self._is_updating = False

        n = len(controls_x) - 1
        self.bezier = Bezierv(n=n, controls_x=controls_x, controls_z=controls_z)

        self.controls_source = ColumnDataSource(data=self._get_controls_data())
        self.curve_source = ColumnDataSource(data=self._get_curve_data())
        
        # Create CDF plot
        self.plot = figure(
            height=400, width=900,
            title="Interactive Bézier CDF Editor",
            x_axis_label="x", y_axis_label="CDF",
            y_range=(-0.05, 1.05)
        )
        
        self.plot.line(x='x', y='y', source=self.curve_source, line_width=3, legend_label="Bézier CDF", color="navy")
        self.plot.line(x='x', y='y', source=self.controls_source, line_dash="dashed", color="gray")
        
        controls_renderer = self.plot.scatter(
            x='x', y='y', source=self.controls_source, size=12,
            legend_label="Control Points", color="firebrick"
        )
        self.plot.legend.location = "top_left"

        draw_tool = PointDrawTool(renderers=[controls_renderer], add=True)
        self.plot.add_tools(draw_tool)

        self.plot.toolbar.active_tap = draw_tool
        self.plot.toolbar.active_drag = draw_tool

        # Create PDF plot below CDF
        self.pdf_plot = figure(
            height=300, width=900,
            title="Bézier PDF",
            x_axis_label="x", y_axis_label="PDF"
        )
        
        self.pdf_plot.line(x='x', y='pdf', source=self.curve_source, line_width=3, legend_label="Bézier PDF", color="green")
        self.pdf_plot.legend.location = "top_right"

        formatter = NumberFormatter(format="0.000")
        columns = [
            TableColumn(field="x", title="X", formatter=formatter),
            TableColumn(field="y", title="Z", formatter=formatter)
        ]
        
        self.data_table = DataTable(
            source=self.controls_source,
            columns=columns, 
            width=250, 
            height=700, 
            editable=True
        )

        self.download_button = Button(
            label="Download Control Points as CSV", 
            button_type="success", 
            width=250
        )

        def _delete_selected():
            selected = list(self.controls_source.selected.indices)
            if not selected:
                return
            
            # Create a copy of current data
            data = dict(self.controls_source.data)
            
            # Filter out selected indices
            keep = [i for i in range(len(data["x"])) if i not in selected]
            
            # If we are deleting too many, stop (prevent going below 2 points)
            if len(keep) < 2:
                print("Cannot delete: minimum 2 points required.")
                return

            new_data = {
                "x": [data["x"][i] for i in keep],
                "y": [data["y"][i] for i in keep]
            }
            
            # Update source (this triggers _update_callback)
            self.controls_source.data = new_data
            
            # Clear selection so we don't get stuck selecting non-existent points
            self.controls_source.selected.indices = []

        self.delete_button = Button(
            label="Delete Selected Control Point",
            button_type="danger",
            width=250
        )

        self.delete_button.on_click(_delete_selected)

        callback = CustomJS(args=dict(source=self.controls_source), code="""
            const data = source.data;
            const file_name = 'control_points.csv';
            let csv_content = 'X,Z_CDF\\n'; // CSV Header

            // Iterate over the data and build the CSV string
            for (let i = 0; i < data.x.length; i++) {
                const row = [data.x[i], data.y[i]];
                csv_content += row.join(',') + '\\n';
            }

            // Create a Blob and trigger the download
            const blob = new Blob([csv_content], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = file_name;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        """)
        
        self.download_button.js_on_click(callback)

        self.controls_source.on_change('data', self._update_callback)

        # Create layout with CDF and PDF plots stacked vertically
        plots_layout = column(self.plot, self.delete_button, self.pdf_plot)
        widgets_layout = column(plots_layout, self.download_button)
        self.layout = row(widgets_layout, self.data_table)

    def _get_controls_data(self):
        """Returns the current control points from the Bezierv instance."""
        return {'x': self.bezier.controls_x, 'y': self.bezier.controls_z}

    def _get_curve_data(self, num_points=200):
        """Calculates and returns the CDF curve points."""
        t = np.linspace(0, 1, num_points)
        curve_x = [self.bezier.poly_x(ti) for ti in t]
        curve_z = [self.bezier.poly_z(ti) for ti in t]
        curve_pdf = [self.bezier.pdf_t(ti) for ti in t]
        return {'x': curve_x, 'y': curve_z, 'pdf': curve_pdf}

    def _update_callback(self, attr, old, new):
        """Handles moving, adding, and deleting control points."""
        
        if self._is_updating:
            return

        try:
            self._is_updating = True

            new_x = new['x']
            new_z = new['y']

            is_point_added = len(new_x) > len(old['x'])
            
            if is_point_added:
                old_points_sorted = sorted(zip(old['x'], old['y']))
                
                if len(old_points_sorted) < 2:
                    raise ValueError("Cannot add a point, need at least 2 existing points.")

                mid_index = len(old_points_sorted) // 2

                p_before = old_points_sorted[mid_index - 1]
                p_after = old_points_sorted[mid_index]

                x_new = (p_before[0] + p_after[0]) / 2.0
                y_new = (p_before[1] + p_after[1]) / 2.0
                
                final_points = old_points_sorted
                final_points.insert(mid_index, (x_new, y_new))
                
                final_x, final_z = zip(*final_points)
                final_x, final_z = list(final_x), list(final_z)
            
            else:
                sorted_points = sorted(zip(new_x, new_z))
                if not sorted_points:
                    final_x, final_z = [], []
                else:
                    final_x, final_z = zip(*sorted_points)
                    final_x, final_z = list(final_x), list(final_z)
                    final_z = sorted(list(final_z))

            if len(final_x) < 2:
                raise ValueError("At least two control points are required.")

            final_z[0] = 0.0
            final_z[-1] = 1.0
            
            new_n = len(final_x) - 1
            if new_n != self.bezier.n:
                self.bezier = Bezierv(n=new_n, controls_x=final_x, controls_z=final_z)
            else:
                self.bezier.update_bezierv(np.array(final_x), np.array(final_z))
            
            self.controls_source.data = {
                'x': list(self.bezier.controls_x), 
                'y': list(self.bezier.controls_z)
            }
            self.curve_source.data = self._get_curve_data()

        except Exception as e:
            print(f"An unexpected error occurred: {e}. Reverting.")
            self.controls_source.data = dict(old)

        finally:
            self._is_updating = False