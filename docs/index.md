# Bezierv: A Python Package for Bézier Distributions

Bezierv provides utilities to **fit, analyze, sample, and convolve** Bézier-based
random variables from empirical data. It includes multiple fitting algorithms
(projected gradient, projected subgradient, nonlinear (Pyomo/IPOPT), and Nelder–Mead),
plus helpers for plotting and Monte Carlo convolution.

> **Tip:** See the [API reference](reference.md) for auto-generated docs from the code.

---

## Quickstart

Install your package (editable mode recommended while developing):

```bash
pip install bezierv
```

Fit a Bézier random variable to data with `DistFit` and sample from it:

```python
import numpy as np
from bezierv.classes.distfit import DistFit

# Synthetic bounded data
rng = np.random.default_rng(42)
data = rng.beta(2, 5, 1000)  # replace with your data

fitter = DistFit(data, n=5)            # n = degree (n control segments, n+1 control points)
bz, mse = fitter.fit(method="projgrad")  # or: 'nonlinear', 'projsubgrad', 'neldermead'
print("MSE:", mse)

samples = bz.random(10_000, rng=42)     # draw samples via inverse CDF
q90 = bz.quantile(0.90)                 # 90% quantile
print("90% quantile:", q90)
```

Plot empirical vs. fitted CDF (optional):

```python
bz.plot_cdf(data)       # overlays ECDF and Bézier CDF
```

---

## Convolution (sum of independent Bézier RVs)

Use `Convolver` to approximate the **sum** of several fitted distributions via
Monte Carlo, then fit another Bézier RV to the result:

```python
from bezierv.classes.convolver import Convolver
from bezierv.classes.distfit import DistFit

# Fit two Bezier RVs separately (bz1, bz2) ... then:
conv = Convolver([bz, bz])          # example: sum with itself
bz_sum = conv.convolve(n_sims=1_000, rng=123, method="projgrad", n=7)
```

Plot fitted CDF (optional):

```python
bz_sum.plot_cdf(data)       # overlays ECDF and Bézier CDF
```

---

## Fitting methods at a glance

- **Projected Gradient (`projgrad`)** – fast and simple; optimizes *z* controls with projection.
- **Projected Subgradient (`projsubgrad`)** – updates both *x* and *z* with projection.
- **Nonlinear (`nonlinear`)** – solves a constrained model via Pyomo (e.g., IPOPT).
- **Nelder–Mead (`neldermead`)** – derivative-free simplex search with penalties.

---

## Next steps

- Browse the **[API reference](reference.md)** for the full class and function docs.